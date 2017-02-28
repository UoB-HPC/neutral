#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <omp.h>
#include "bright.h"
#include "../bright_interface.h"
#include "../../comms.h"
#include "../../shared.h"
#include "../../shared_data.h"
#include "../../params.h"

#ifdef MPI
#include "mpi.h"
#endif

// Performs a solve of dependent variables for particle transport.
void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny, 
    const int x_off, const int y_off, const double dt, const int ntotal_particles,
    int* nlocal_particles, uint64_t* master_key, const int* neighbours, 
    Particle* particles, const double* density, const double* edgex, 
    const double* edgey, const double* edgedx, const double* edgedy, 
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table, 
    double* scalar_flux_tally, double* energy_deposition_tally, RNPool* rn_pools)
{
  // Initial idea is to use a kind of queue for handling the particles. Presumably
  // this doesn't have to be a carefully ordered queue but lets see how that goes.

  // This is the known starting number of particles
  int facets = 0;
  int collisions = 0;
  int nparticles = *nlocal_particles;
  int nparticles_sent[NNEIGHBOURS];

  if(!nparticles) {
    printf("out of particles\n");
    return;
  }

  // Communication isn't required for edges
  for(int ii = 0; ii < NNEIGHBOURS; ++ii) {
    nparticles_sent[ii] = 0;
  }

  handle_particles(
      global_nx, global_ny, nx, ny, x_off, y_off, dt, neighbours, density, edgex, 
      edgey, &facets, &collisions, nparticles_sent, master_key, ntotal_particles, 
      nparticles, &nparticles, particles, cs_scatter_table, 
      cs_absorb_table, scalar_flux_tally, energy_deposition_tally, rn_pools);

#if 0
#ifdef MPI
  while(1) {
    int nneighbours = 0;
    int nparticles_recv[NNEIGHBOURS];
    MPI_Request recv_req[NNEIGHBOURS];
    MPI_Request send_req[NNEIGHBOURS];
    for(int ii = 0; ii < NNEIGHBOURS; ++ii) {
      // Initialise received particles
      nparticles_recv[ii] = 0;

      // No communication required at edge
      if(neighbours[ii] == EDGE) {
        continue;
      }

      // Check which neighbours are sending some particles
      MPI_Irecv(
          &nparticles_recv[ii], 1, MPI_INT, neighbours[ii],
          TAG_SEND_RECV, MPI_COMM_WORLD, &recv_req[nneighbours]);
      MPI_Isend(
          &nparticles_sent[ii], 1, MPI_INT, neighbours[ii],
          TAG_SEND_RECV, MPI_COMM_WORLD, &send_req[nneighbours++]);
    }

    MPI_Waitall(
        nneighbours, recv_req, MPI_STATUSES_IGNORE);
    nneighbours = 0;

    // Manage all of the received particles
    int nunprocessed_particles = 0;
    const int unprocessed_start = nparticles;
    for(int ii = 0; ii < NNEIGHBOURS; ++ii) {
      if(neighbours[ii] == EDGE) {
        continue;
      }

      // Receive the particles from this neighbour
      for(int jj = 0; jj < nparticles_recv[ii]; ++jj) {
        MPI_Recv(
            &particles[unprocessed_start+nunprocessed_particles], 
            1, particle_type, neighbours[ii], TAG_PARTICLE, 
            MPI_COMM_WORLD, MPI_STATUSES_IGNORE);
        nunprocessed_particles++;
      }

      nparticles_recv[ii] = 0;
      nparticles_sent[ii] = 0;
    }

    nparticles += nunprocessed_particles;
    if(nunprocessed_particles) {
      handle_particles(
          global_nx, global_ny, nx, ny, x_off, y_off, 0, dt, neighbours,
          density, edgex, edgey, edgedx, edgedy, &facets, &collisions, 
          nparticles_sent, ntotal_particles, nunprocessed_particles, &nparticles, 
          &particles[unprocessed_start], particles_out, cs_scatter_table, 
          cs_absorb_table, scalar_flux_tally, energy_deposition_tally, rn_pools);
    }

    // Check if any of the ranks had unprocessed particles
    int particles_to_process;
    MPI_Allreduce(
        &nunprocessed_particles, &particles_to_process, 
        1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // All ranks have reached census
    if(!particles_to_process) {
      break;
    }
  }
#endif

  barrier();
#endif // if 0

  *nlocal_particles = nparticles;

  printf("facets %d collisions %d\n", facets, collisions);
}

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const double dt, const int* neighbours, 
    const double* density, const double* edgex, const double* edgey, int* facets, 
    int* collisions, int* nparticles_sent, uint64_t* master_key, 
    const int ntotal_particles, const int nparticles_to_process, 
    int* nparticles, Particle* particles_start, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table, double* scalar_flux_tally, 
    double* energy_deposition_tally, RNPool* rn_pools)
{
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }

  int* events = (int*)malloc(sizeof(int)*ntotal_particles);

  const double inv_ntotal_particles = 1.0/(double)ntotal_particles;

  int initialised = 0;
  int nparticles_out = 0;
  int nparticles_dead = 0;

  while(1) {
    update_rn_pool_master_keys(
        rn_pools, nthreads+1, (*master_key)++);

    /* INITIALISATION */
    if(!initialised) {
      // Generate random numbers for every particle...
      fill_rn_buffer(
          &rn_pools[nthreads], ntotal_particles);

      // Initialise all particles to the correct census time and mfp to collision
#pragma omp parallel for simd
      for(int ii = 0; ii < ntotal_particles; ++ii) {
        Particle* particle = &particles_start[ii];
        particle->dt_to_census = dt;

        // Reset living particles
        if(particle->next_event != DEAD) {
          particle->next_event = FACET; 
        }

        const double microscopic_cs_scatter = 
          microscopic_cs_for_energy(cs_scatter_table, particle->e, &particle->scatter_cs_index);
        const int cellx = particle->cellx-x_off+PAD;
        const int celly = particle->celly-y_off+PAD;
        const double local_density = density[celly*(nx+2*PAD)+cellx];
        const double number_density = (local_density*AVOGADROS/MOLAR_MASS);
        const double rn0 = rn_pools[nthreads].buffer[ii]; // Make this a function
        const double macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
        particle->mfp_to_collision = -log(rn0)/macroscopic_cs_scatter;
      }
      initialised = 1;
    }

    /* DISTANCE TO FACET */
#pragma omp parallel for simd
    for(int ii = 0; ii < ntotal_particles; ++ii) {
      Particle* particle = &particles_start[ii];
      if(particle->next_event == DEAD || particle->next_event == CENSUS) {
        continue;
      }

      double particle_velocity = sqrt((2.0*particle->e*eV_TO_J)/PARTICLE_MASS);

      // Check the timestep required to move the particle along a single axis
      // If the velocity is positive then the top or right boundary will be hit
      const int cellx = particle->cellx-x_off+PAD;
      const int celly = particle->celly-y_off+PAD;
      double u_x_inv = 1.0/(particle->omega_x*particle_velocity);
      double u_y_inv = 1.0/(particle->omega_y*particle_velocity);

      // The bound is open on the left and bottom so we have to correct for this and
      // required the movement to the facet to go slightly further than the edge
      // in the calculated values, using OPEN_BOUND_CORRECTION, which is the smallest
      // possible distance we can be from the closed bound e.g. 1.0e-14.
      double dt_x = (particle->omega_x >= 0.0)
        ? ((edgex[cellx+1])-particle->x)*u_x_inv
        : ((edgex[cellx]-OPEN_BOUND_CORRECTION)-particle->x)*u_x_inv;
      double dt_y = (particle->omega_y >= 0.0)
        ? ((edgey[celly+1])-particle->y)*u_y_inv
        : ((edgey[celly]-OPEN_BOUND_CORRECTION)-particle->y)*u_y_inv;

      // Calculated the projection to be
      // a = vector on first edge to be hit
      // u = velocity vector

      double mag_u0 = particle_velocity;

      particle->x_facet = (dt_x < dt_y) ? 1 : 0;
      if(particle->x_facet) {
        // cos(theta) = ||(x, 0)||/||(u_x', u_y')|| - u' is u at boundary
        // cos(theta) = (x.u)/(||x||.||u||)
        // x_x/||u'|| = (x_x, 0)*(u_x, u_y) / (x_x.||u||)
        // x_x/||u'|| = (x_x.u_x / x_x.||u||)
        // x_x/||u'|| = u_x/||u||
        // ||u'|| = (x_x.||u||)/u_x
        // We are centered on the origin, so the y component is 0 after travelling
        // aint the x axis to the edge (ax, 0).(x, y)
        particle->distance_to_facet = (particle->omega_x >= 0.0)
          ? ((edgex[cellx+1])-particle->x)*mag_u0*u_x_inv
          : ((edgex[cellx]-OPEN_BOUND_CORRECTION)-particle->x)*mag_u0*u_x_inv;
      }
      else {
        // We are centered on the origin, so the x component is 0 after travelling
        // along the y axis to the edge (0, ay).(x, y)
        particle->distance_to_facet = (particle->omega_y >= 0.0)
          ? ((edgey[celly+1])-particle->y)*mag_u0*u_y_inv
          : ((edgey[celly]-OPEN_BOUND_CORRECTION)-particle->y)*mag_u0*u_y_inv;
      }
    }

    /* CALCULATE THE EVENTS */
    int nfacets = 0;
    int ncollisions = 0;
#pragma omp parallel for simd reduction(+: ncollisions, nfacets)
    for(int ii = 0; ii < ntotal_particles; ++ii) {
      Particle* particle = &particles_start[ii];
      if(particle->next_event == DEAD || particle->next_event == CENSUS) {
        continue;
      }

      double microscopic_cs_scatter = 
        microscopic_cs_for_energy(cs_scatter_table, particle->e, &particle->scatter_cs_index);
      double microscopic_cs_absorb = 
        microscopic_cs_for_energy(cs_absorb_table, particle->e, &particle->absorb_cs_index);
      int cellx = particle->cellx-x_off+PAD;
      int celly = particle->celly-y_off+PAD;
      double local_density = density[celly*(nx+2*PAD)+cellx];
      double number_density = (local_density*AVOGADROS/MOLAR_MASS);
      double macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
      double macroscopic_cs_absorb = number_density*microscopic_cs_absorb*BARNS;
      double particle_velocity = sqrt((2.0*particle->e*eV_TO_J)/PARTICLE_MASS);
      double cell_mfp = 1.0/(macroscopic_cs_scatter+macroscopic_cs_absorb);
      const double distance_to_collision = particle->mfp_to_collision*cell_mfp;
      const double distance_to_census = particle_velocity*particle->dt_to_census;

      if(distance_to_collision < distance_to_census && 
          distance_to_collision < particle->distance_to_facet) {
        particle->next_event = COLLISION;
        ncollisions++;
      }
      else if(particle->distance_to_facet < distance_to_census) {
        particle->next_event = FACET;
        nfacets++;
      }
      else {
        particle->next_event = CENSUS;
      }
    }

    if(!nfacets && !ncollisions) {
      break;
    }

    printf("calculated the events collision %d facets %d census/dead %d\n",
        ncollisions, nfacets, (ntotal_particles-nfacets-ncollisions));

    *facets = nfacets;
    *collisions = ncollisions;

    /* HANDLE FACET ENCOUNTERS */
#pragma omp parallel for simd reduction(+:nparticles_out)
    for(int ii = 0; ii < ntotal_particles; ++ii) {
      Particle* particle = &particles_start[ii];
      if(particle->next_event != FACET) {
        continue;
      }
      int cellx = particle->cellx-x_off+PAD;
      int celly = particle->celly-y_off+PAD;
      double local_density = density[celly*(nx+2*PAD)+cellx];
      double microscopic_cs_scatter = 
        microscopic_cs_for_energy(cs_scatter_table, particle->e, &particle->scatter_cs_index);
      double microscopic_cs_absorb = 
        microscopic_cs_for_energy(cs_absorb_table, particle->e, &particle->absorb_cs_index);
      double number_density = (local_density*AVOGADROS/MOLAR_MASS);
      double macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
      double macroscopic_cs_absorb = number_density*microscopic_cs_absorb*BARNS;
      double cell_mfp = 1.0/(macroscopic_cs_scatter+macroscopic_cs_absorb);
      double particle_velocity = sqrt((2.0*particle->e*eV_TO_J)/PARTICLE_MASS);

      // Update the mean free paths until collision
      particle->mfp_to_collision -= (particle->distance_to_facet/cell_mfp);
      particle->dt_to_census -= (particle->distance_to_facet/particle_velocity);

      // Update the tallies
      double inv_cell_volume = 1.0/edgex[cellx]*edgey[celly];
      double scalar_flux = particle->weight*particle->distance_to_facet*inv_cell_volume;
      double energy_deposition = calculate_energy_deposition(
          particle, particle->distance_to_facet, number_density, 
          microscopic_cs_absorb, microscopic_cs_scatter+microscopic_cs_absorb);
      update_tallies(
          nx, x_off, y_off, particle, inv_ntotal_particles, energy_deposition,
          scalar_flux, scalar_flux_tally, energy_deposition_tally);

      // Encounter facet, and jump out if particle left this rank's domain
      if(handle_facet_encounter(
            global_nx, global_ny, nx, ny, x_off, y_off, neighbours, 
            particle->distance_to_facet, particle->x_facet, 
            nparticles_sent, particle)) {
        nparticles_out++;
        continue;
      }
    }

    /* HANDLE COLLISIONS */
#pragma omp parallel for simd reduction(+:nparticles_dead)
    for(int ii = 0; ii < ntotal_particles; ++ii) {
      Particle* particle = &particles_start[ii];
      if(particle->next_event != COLLISION) {
        continue;
      }

      // Don't need to tally into mesh on collision
      int cellx = particle->cellx-x_off+PAD;
      int celly = particle->celly-y_off+PAD;
      double local_density = density[celly*(nx+2*PAD)+cellx];
      double microscopic_cs_scatter = 
        microscopic_cs_for_energy(cs_scatter_table, particle->e, &particle->scatter_cs_index);
      double microscopic_cs_absorb = 
        microscopic_cs_for_energy(cs_absorb_table, particle->e, &particle->absorb_cs_index);
      double number_density = (local_density*AVOGADROS/MOLAR_MASS);
      double macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
      double macroscopic_cs_absorb = number_density*microscopic_cs_absorb*BARNS;
      double cell_mfp = 1.0/(macroscopic_cs_scatter+macroscopic_cs_absorb);
      const double distance_to_collision = particle->mfp_to_collision*cell_mfp;
      double inv_cell_volume = 1.0/edgex[cellx]*edgey[celly];

      // Calculate the energy deposition in the cell
      double scalar_flux = particle->weight*distance_to_collision*inv_cell_volume;
      double energy_deposition = calculate_energy_deposition(
          particle, distance_to_collision, number_density, microscopic_cs_absorb, 
          microscopic_cs_scatter+microscopic_cs_absorb);

      // Moves the particle to the collision site
      particle->x += distance_to_collision*particle->omega_x;
      particle->y += distance_to_collision*particle->omega_y;

      const double p_absorb = macroscopic_cs_absorb/
        (macroscopic_cs_absorb+macroscopic_cs_scatter);

      RNPool* local_rn_pool = &rn_pools[omp_get_thread_num()];
      if(getrand(local_rn_pool) < p_absorb) {
        /* Model particle absorption */

        // Find the new particle weight after absorption, saving the energy change
        const double new_weight = particle->weight*(1.0 - p_absorb);
        particle->weight = new_weight;

        if(particle->e < MIN_ENERGY_OF_INTEREST) {
          // Energy is too low, so mark the particle for deletion
          particle->next_event = 1;
          nparticles_dead++;
        }
      }
      else {
        /* Model elastic particle scattering */

        // Choose a random scattering angle between -1 and 1
        // TODO: THIS RANDOM NUMBER SELECTION DOESN'T WORK
        const double mu_cm = 1.0 - 2.0*getrand(local_rn_pool);

        // Calculate the new energy based on the relation to angle of incidence
        const double e_new = particle->e*
          (MASS_NO*MASS_NO + 2.0*MASS_NO*mu_cm + 1.0)/
          ((MASS_NO + 1.0)*(MASS_NO + 1.0));

        // Convert the angle into the laboratory frame of reference
        double cos_theta =
          0.5*((MASS_NO+1.0)*sqrt(e_new/particle->e) - 
              (MASS_NO-1.0)*sqrt(particle->e/e_new));

        // Alter the direction of the velocities
        const double sin_theta = sin(acos(cos_theta));
        const double omega_x_new =
          (particle->omega_x*cos_theta - particle->omega_y*sin_theta);
        const double omega_y_new =
          (particle->omega_x*sin_theta + particle->omega_y*cos_theta);
        particle->omega_x = omega_x_new;
        particle->omega_y = omega_y_new;
        particle->e = e_new;
      }

      // Need to store tally information as finished with particle
      update_tallies(
          nx, x_off, y_off, particle, inv_ntotal_particles, energy_deposition,
          scalar_flux, scalar_flux_tally, energy_deposition_tally);

      double particle_velocity = sqrt((2.0*particle->e*eV_TO_J)/PARTICLE_MASS);
      particle->mfp_to_collision = -log(getrand(local_rn_pool))/macroscopic_cs_scatter;
      particle->dt_to_census -= distance_to_collision/particle_velocity;
    }
  }

  printf("left the main loop\n");

  /* HANDLE THE CENSUS EVENTS */
#pragma omp parallel for simd
  for(int ii = 0; ii < ntotal_particles; ++ii) {
    Particle* particle = &particles_start[ii];
    if(particle->next_event != CENSUS) {
      continue;
    }

    double particle_velocity = sqrt((2.0*particle->e*eV_TO_J)/PARTICLE_MASS);
    const double distance_to_census = particle_velocity*particle->dt_to_census;
    double microscopic_cs_scatter = 
      microscopic_cs_for_energy(cs_scatter_table, particle->e, &particle->scatter_cs_index);
    double microscopic_cs_absorb = 
      microscopic_cs_for_energy(cs_absorb_table, particle->e, &particle->absorb_cs_index);
    int cellx = particle->cellx-x_off+PAD;
    int celly = particle->celly-y_off+PAD;
    double local_density = density[celly*(nx+2*PAD)+cellx];
    double number_density = (local_density*AVOGADROS/MOLAR_MASS);
    double macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
    double macroscopic_cs_absorb = number_density*microscopic_cs_absorb*BARNS;
    double cell_mfp = 1.0/(macroscopic_cs_scatter+macroscopic_cs_absorb);
    // We have not changed cell or energy level at this stage
    particle->x += distance_to_census*particle->omega_x;
    particle->y += distance_to_census*particle->omega_y;
    particle->mfp_to_collision -= (distance_to_census/cell_mfp);

    double inv_cell_volume = 1.0/edgex[cellx]*edgey[celly];
    double scalar_flux = particle->weight*distance_to_census*inv_cell_volume;

    // Calculate the energy deposition in the cell
    double energy_deposition = calculate_energy_deposition(
        particle, distance_to_census, number_density, microscopic_cs_absorb, 
        microscopic_cs_scatter+microscopic_cs_absorb);

    // Need to store tally information as finished with particle
    update_tallies(
        nx, x_off, y_off, particle, inv_ntotal_particles, energy_deposition,
        scalar_flux, scalar_flux_tally, energy_deposition_tally);

    particle->dt_to_census = 0.0;
  }

  // Have now handled all events...

  // Correct the new total number of particles
  *nparticles -= (nparticles_dead+nparticles_out);

  printf("handled %d particles, with %d particles deleted\n", 
      nparticles_to_process, nparticles_dead+nparticles_out);
}

// Tallies both the scalar flux and energy deposition in the cell
void update_tallies(
    const int nx, const int x_off, const int y_off, Particle* particle, 
    const double inv_ntotal_particles, const double energy_deposition,
    const double scalar_flux, double* scalar_flux_tally, 
    double* energy_deposition_tally)
{
  // Store the scalar flux
  const int cellx = particle->cellx-x_off;
  const int celly = particle->celly-y_off;

#pragma omp atomic update 
  scalar_flux_tally[celly*nx+cellx] += 
    scalar_flux*inv_ntotal_particles; 

#pragma omp atomic update
  energy_deposition_tally[celly*nx+cellx] += 
    energy_deposition*inv_ntotal_particles;
}

// Makes the necessary updates to the particle given that
// the facet was encountered
int handle_facet_encounter(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const int* neighbours, 
    const double distance_to_facet, int x_facet, int* nparticles_sent, 
    Particle* particle)
{
  // TODO: Make sure that the roundoff is handled here, perhaps actually set it
  // fully to one of the edges here
  particle->x += distance_to_facet*particle->omega_x;
  particle->y += distance_to_facet*particle->omega_y;

  // This use of x_facet is a slight misnoma, as it is really a facet
  // along the y dimensions
  if(x_facet) {
    if(particle->omega_x > 0.0) {
      // Reflect at the boundary
      if(particle->cellx >= (global_nx-1)) {
        particle->omega_x = -(particle->omega_x);
      }
      else {
        // Definitely moving to right cell
        particle->cellx += 1;

        // Check if we need to pass to another process
        if(particle->cellx >= nx+x_off) {
          send_and_mark_particle(neighbours[EAST], particle);
          nparticles_sent[EAST]++;
          return 1;
        }
      }
    }
    else if(particle->omega_x < 0.0) {
      if(particle->cellx <= 0) {
        // Reflect at the boundary
        particle->omega_x = -(particle->omega_x);
      }
      else {
        // Definitely moving to left cell
        particle->cellx -= 1;

        // Check if we need to pass to another process
        if(particle->cellx < x_off) {
          send_and_mark_particle(neighbours[WEST], particle);
          nparticles_sent[WEST]++;
          return 1;
        }
      }
    }
  }
  else {
    if(particle->omega_y > 0.0) {
      // Reflect at the boundary
      if(particle->celly >= (global_ny-1)) {
        particle->omega_y = -(particle->omega_y);
      }
      else {
        // Definitely moving to north cell
        particle->celly += 1;

        // Check if we need to pass to another process
        if(particle->celly >= ny+y_off) {
          send_and_mark_particle(neighbours[NORTH], particle);
          nparticles_sent[NORTH]++;
          return 1;
        }
      }
    }
    else if(particle->omega_y < 0.0) {
      // Reflect at the boundary
      if(particle->celly <= 0) {
        particle->omega_y = -(particle->omega_y);
      }
      else {
        // Definitely moving to south cell
        particle->celly -= 1;

        // Check if we need to pass to another process
        if(particle->celly < y_off) {
          send_and_mark_particle(neighbours[SOUTH], particle);
          nparticles_sent[SOUTH]++;
          return 1;
        }
      }
    }
  }

  return 0;
}

// Sends a particle to a neighbour and replaces in the particle list
void send_and_mark_particle(
    const int destination, Particle* particle)
{
#ifdef MPI
  if(destination == EDGE) {
    return;
  }

  particle->next_event = 1;

  // Send the particle
  MPI_Send(
      particle, 1, particle_type, destination, TAG_PARTICLE, MPI_COMM_WORLD);
#else
  TERMINATE("Unreachable - shouldn't send particles unless MPI enabled.\n");
#endif
}

// Calculate the energy deposition in the cell
double calculate_energy_deposition(
    Particle* particle, const double path_length, const double number_density, 
    const double microscopic_cs_absorb, const double microscopic_cs_total)
{
  // Calculate the energy deposition based on the path length
  const double average_exit_energy_absorb = 0.0;
  const double absorption_heating = 
    (microscopic_cs_absorb/microscopic_cs_total)*average_exit_energy_absorb;
  const double average_exit_energy_scatter = 
    particle->e*(MASS_NO*MASS_NO+2)/((MASS_NO+1)*(MASS_NO+1));
  const double scattering_heating = 
    (1.0-(microscopic_cs_absorb/microscopic_cs_total))*average_exit_energy_scatter;
  const double heating_response =
    (particle->e-scattering_heating-absorption_heating);

  return particle->weight*path_length*(microscopic_cs_total*BARNS)*
    heating_response*number_density;
}

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy(
    const CrossSection* cs, const double energy, int* cs_index)
{
  /* Attempt an optimisation of the search by performing a linear operation
   * if there is an existing location. We assume that the energy has
   * reduced rather than increased, which seems to be a legitimate 
   * approximation in this particular case */

  int ind = 0; 
  double* key = cs->key;
  double* value = cs->value;

  // Minor optimisation for times when energy hasn't changed
  if(key[*cs_index] == energy) {
    return energy;
  }

  // Determine the correct search direction required to move towards the
  // new energy
  const int direction = (energy-cs->value[*cs_index] > 0.0) ? 1 : -1; 

#if 0
  // TODO: The problem that occurred with the binary search appears to be an
  // actual bug rather than just a performance issue. Now it is resolved it might
  // not be necessary to actually have the linear search.
  if(*cs_index > -1) {
    // This search will move in the correct direction towards the new energy group
    for(int ind = *cs_index; ind >= 0 && ind < cs->nentries; ind += direction) {
      // Check if we have found the new energy group index
      if(key[ind-1] > energy || key[ind] <= energy) {
        break;
      }
    }
  }
  else {
#endif // if 0
    // Use a simple binary search to find the energy group
    ind = cs->nentries/2;
    int width = ind/2;
    while(key[ind-1] > energy || key[ind] < energy) {
      ind += (key[ind] > energy) ? -width : width;
      width = max(1, width/2); // To handle odd cases, allows one extra walk
    }
#if 0
  }
#endif // if 0

  *cs_index = ind;

  // TODO: perform some interesting interpolation here
  // Center weighted is poor accuracy but might even out over enough particles
  return (value[ind-1] + value[ind])/2.0;
}

// Validates the results of the simulation
void validate(
    const int nx, const int ny, const char* params_filename, 
    const int rank, double* energy_deposition_tally)
{
  double local_energy_tally = 0.0;
  for(int ii = 0; ii < nx*ny; ++ii) {
    local_energy_tally += energy_deposition_tally[ii];
  }

  double global_energy_tally = reduce_all_sum(local_energy_tally);

  if(rank != MASTER) {
    return;
  }

  printf("\nFinal global_energy_tally %.15e\n", global_energy_tally);

  int nresults = 0;
  char* keys = (char*)malloc(sizeof(char)*MAX_KEYS*(MAX_STR_LEN+1));
  double* values = (double*)malloc(sizeof(double)*MAX_KEYS);
  if(!get_key_value_parameter(
        params_filename, NEUTRAL_TESTS, keys, values, &nresults)) {
    printf("Warning. Test entry was not found, could NOT validate.\n");
    return;
  }

  printf("Expected %.12e, result was %.12e.\n", values[0], global_energy_tally);
  if(within_tolerance(values[0], global_energy_tally, VALIDATE_TOLERANCE)) {
    printf("PASSED validation.\n");
  }
  else {
    printf("FAILED validation.\n");
  }

  free(keys);
  free(values);
}

