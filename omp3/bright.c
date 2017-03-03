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

// Performs a solve of dependent variables for particles transport.
void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny, 
    const int x_off, const int y_off, const double dt, const int ntotal_particles,
    int* nlocal_particles, uint64_t* master_key, const int* neighbours, 
    Particles* particles, const double* density, const double* edgex, 
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
    int* nparticles, Particles* particles, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table, double* scalar_flux_tally, 
    double* energy_deposition_tally, RNPool* rn_pools)
{
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }

  int initialised = 0;
  int nparticles_out = 0;
  int nparticles_dead = 0;

  while(1) {
    START_PROFILING(&compute_profile);
    update_rn_pool_master_keys(
        rn_pools, nthreads+1, (*master_key)++);
    STOP_PROFILING(&compute_profile, "update rn pool master keys");

    /* INITIALISATION */
    if(!initialised) {
      START_PROFILING(&compute_profile);
      event_initialisation(
          ntotal_particles, nx, x_off, y_off, particles, dt, density,
          nthreads, rn_pools, cs_scatter_table, cs_absorb_table);
      initialised = 1;
      STOP_PROFILING(&compute_profile, "initialisation");
    }

    // Calculates the distance to the facet for all cells
    START_PROFILING(&compute_profile);
    calc_distance_to_facet(
        ntotal_particles, x_off, y_off, particles, edgex, edgey);
    STOP_PROFILING(&compute_profile, "calc dist to facet");

    START_PROFILING(&compute_profile);
    const int all_census = 
      calc_next_event(ntotal_particles, particles, facets, collisions);
    STOP_PROFILING(&compute_profile, "calc next event");

    if(all_census) {
      break;
    }

    START_PROFILING(&compute_profile);
    handle_facets(
        ntotal_particles, global_nx, global_ny, nx, ny, x_off, y_off, 
        neighbours, nparticles_sent, particles, edgex, edgey, density, 
        &nparticles_out, scalar_flux_tally, energy_deposition_tally,
        cs_scatter_table, cs_absorb_table);
    STOP_PROFILING(&compute_profile, "handle facets");

    START_PROFILING(&compute_profile);
    handle_collisions( 
        ntotal_particles, nx, x_off, y_off, particles, edgex, edgey, 
        rn_pools, &nparticles_dead, scalar_flux_tally, energy_deposition_tally);
    STOP_PROFILING(&compute_profile, "handle collisions");
  }

  START_PROFILING(&compute_profile);
  handle_census(
      ntotal_particles, nx, x_off, y_off, particles, density, edgex, 
      edgey, scalar_flux_tally, energy_deposition_tally);
  STOP_PROFILING(&compute_profile, "handle census");

  printf("left the main loop\n");

  // Have now handled all events...

  // Correct the new total number of particles
  *nparticles -= (nparticles_dead+nparticles_out);

  printf("handled %d particles, with %d particles deleted\n", 
      nparticles_to_process, nparticles_dead+nparticles_out);
}

// Initialises ready for the event cycles
void event_initialisation(
    const int ntotal_particles, const int nx, const int x_off, const int y_off, 
    Particles* particles, const double dt, const double* density,
    const int nthreads, RNPool* rn_pools, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table)
{
  // Generate random numbers for every particles...
  fill_rn_buffer(
      &rn_pools[nthreads], ntotal_particles);

  // Initialise all of the particles with their starting state
#pragma omp parallel for simd
  for(int ii = 0; ii < ntotal_particles; ++ii) {
    particles->dt_to_census[ii] = dt;

    // Reset living particles
    if(particles->next_event[ii] != DEAD) {
      particles->next_event[ii] = FACET; 
    }

    particles->microscopic_cs_scatter[ii] = microscopic_cs_for_energy(
          cs_scatter_table, particles->e[ii], &particles->scatter_cs_index[ii]);
    particles->microscopic_cs_absorb[ii] = microscopic_cs_for_energy(
          cs_absorb_table, particles->e[ii], &particles->absorb_cs_index[ii]);
    particles->particle_velocity[ii] =
      sqrt((2.0*particles->e[ii]*eV_TO_J)/PARTICLE_MASS);

    int cellx = particles->cellx[ii]-x_off+PAD;
    int celly = particles->celly[ii]-y_off+PAD;
    particles->local_density[ii] = density[celly*(nx+2*PAD)+cellx];
    double number_density = (particles->local_density[ii]*AVOGADROS/MOLAR_MASS);
    double macroscopic_cs_scatter = number_density*particles->microscopic_cs_scatter[ii]*BARNS;
    double macroscopic_cs_absorb = number_density*particles->microscopic_cs_absorb[ii]*BARNS;
    particles->cell_mfp[ii] = 1.0/(macroscopic_cs_scatter+macroscopic_cs_absorb);
    const double rn0 = rn_pools[nthreads].buffer[ii]; // Make this a function
    particles->mfp_to_collision[ii] = -log(rn0)/macroscopic_cs_scatter;
  }
}

// Calculates the next event for each particle
int calc_next_event(
    const int ntotal_particles, Particles* particles, int* facets, int* collisions)
{
  /* CALCULATE THE EVENTS */
  int nfacets = 0;
  int ncollisions = 0;
#pragma omp parallel for simd reduction(+: ncollisions, nfacets)
  for(int ii = 0; ii < ntotal_particles; ++ii) {
    if(particles->next_event[ii] == DEAD || particles->next_event[ii] == CENSUS) {
      continue;
    }

    particles->particle_velocity[ii] = sqrt((2.0*particles->e[ii]*eV_TO_J)/PARTICLE_MASS);
    const double distance_to_collision = particles->mfp_to_collision[ii]*particles->cell_mfp[ii];
    const double distance_to_census = 
      particles->particle_velocity[ii]*particles->dt_to_census[ii];

    if(distance_to_collision < distance_to_census && 
        distance_to_collision < particles->distance_to_facet[ii]) {
      particles->next_event[ii] = COLLISION;
      ncollisions++;
    }
    else if(particles->distance_to_facet[ii] < distance_to_census) {
      particles->next_event[ii] = FACET;
      nfacets++;
    }
    else {
      particles->next_event[ii] = CENSUS;
    }
  }

  *facets += nfacets;
  *collisions += ncollisions;

#if 0
  printf("calculated the events collision %d facets %d census/dead %d\n",
      ncollisions, nfacets, (ntotal_particles-nfacets-ncollisions));
#endif // if 0

  return (!nfacets && !ncollisions);
}

// Handle all of the facet encounters
void handle_facets(
    const int ntotal_particles, const int global_nx, const int global_ny, 
    const int nx, const int ny, const int x_off, const int y_off, 
    const int* neighbours, int* nparticles_sent, Particles* particles, 
    const double* edgex, const double* edgey, const double* density, 
    int* nparticles_out, double* scalar_flux_tally, double* energy_deposition_tally,
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table)
{
  const double inv_ntotal_particles = 1.0/ntotal_particles;

  int np_out_east = 0;
  int np_out_west = 0;
  int np_out_north = 0;
  int np_out_south = 0;

  /* HANDLE FACET ENCOUNTERS */
#pragma omp parallel for simd \
  reduction(+:np_out_east, np_out_west, np_out_north, np_out_south) 
  for(int ii = 0; ii < ntotal_particles; ++ii) {
    if(particles->next_event[ii] != FACET) {
      continue;
    }
    int cellx = particles->cellx[ii]-x_off+PAD;
    int celly = particles->celly[ii]-y_off+PAD;
    double number_density = (particles->local_density[ii]*AVOGADROS/MOLAR_MASS);

    double macroscopic_cs_scatter = 
      number_density*particles->microscopic_cs_scatter[ii]*BARNS;
    double macroscopic_cs_absorb = 
      number_density*particles->microscopic_cs_absorb[ii]*BARNS;

    // Update the mean free paths until collision
    particles->mfp_to_collision[ii] -=
      (particles->distance_to_facet[ii]*(macroscopic_cs_scatter+macroscopic_cs_absorb));
    particles->dt_to_census[ii] -=
      (particles->distance_to_facet[ii]/particles->particle_velocity[ii]);

    // Update the tallies
    double inv_cell_volume = 1.0/(edgex[cellx]*edgey[celly]);
    double scalar_flux =
      particles->weight[ii]*particles->distance_to_facet[ii]*inv_cell_volume;
    double energy_deposition = calculate_energy_deposition(
        ii, particles, particles->distance_to_facet[ii], number_density, 
        particles->microscopic_cs_absorb[ii], 
        particles->microscopic_cs_scatter[ii]+particles->microscopic_cs_absorb[ii]);
    update_tallies(
        ii, nx, x_off, y_off, particles, cellx, celly, inv_ntotal_particles, 
        energy_deposition, scalar_flux, scalar_flux_tally, energy_deposition_tally);

    // TODO: Make sure that the roundoff is handled here, perhaps actually set it
    // fully to one of the edges here
    particles->x[ii] += particles->distance_to_facet[ii]*particles->omega_x[ii];
    particles->y[ii] += particles->distance_to_facet[ii]*particles->omega_y[ii];

    // This use of x_facet is a slight misnoma, as it is really a facet
    // along the y dimensions
    if(particles->x_facet[ii]) {
      if(particles->omega_x[ii] > 0.0) {
        // Reflect at the boundary
        if(particles->cellx[ii] >= (global_nx-1)) {
          particles->omega_x[ii] = -(particles->omega_x[ii]);
        }
        else {
          // Definitely moving to right cell
          particles->cellx[ii] += 1;

          // Check if we need to pass to another process
          if(particles->cellx[ii] >= nx+x_off) {
            send_and_mark_particle(neighbours[EAST], ii, particles);
            np_out_east++;
            continue;
          }
        }
      }
      else if(particles->omega_x[ii] < 0.0) {
        if(particles->cellx[ii] <= 0) {
          // Reflect at the boundary
          particles->omega_x[ii] = -(particles->omega_x[ii]);
        }
        else {
          // Definitely moving to left cell
          particles->cellx[ii] -= 1;

          // Check if we need to pass to another process
          if(particles->cellx[ii] < x_off) {
            send_and_mark_particle(neighbours[WEST], ii, particles);
            np_out_west++;
            continue;
          }
        }
      }
    }
    else {
      if(particles->omega_y[ii] > 0.0) {
        // Reflect at the boundary
        if(particles->celly[ii] >= (global_ny-1)) {
          particles->omega_y[ii] = -(particles->omega_y[ii]);
        }
        else {
          // Definitely moving to north cell
          particles->celly[ii] += 1;

          // Check if we need to pass to another process
          if(particles->celly[ii] >= ny+y_off) {
            send_and_mark_particle(neighbours[NORTH], ii, particles);
            np_out_north++;
            continue;
          }
        }
      }
      else if(particles->omega_y[ii] < 0.0) {
        // Reflect at the boundary
        if(particles->celly[ii] <= 0) {
          particles->omega_y[ii] = -(particles->omega_y[ii]);
        }
        else {
          // Definitely moving to south cell
          particles->celly[ii] -= 1;

          // Check if we need to pass to another process
          if(particles->celly[ii] < y_off) {
            send_and_mark_particle(neighbours[SOUTH], ii, particles);
            np_out_south++;
            continue;
          }
        }
      }
    }

    particles->local_density[ii] = density[celly*(nx+2*PAD)+cellx];
    number_density = (particles->local_density[ii]*AVOGADROS/MOLAR_MASS);
    macroscopic_cs_scatter = 
      number_density*particles->microscopic_cs_scatter[ii]*BARNS;
    macroscopic_cs_absorb = 
      number_density*particles->microscopic_cs_absorb[ii]*BARNS;
    particles->cell_mfp[ii] = 1.0/(macroscopic_cs_scatter+macroscopic_cs_absorb);
  }

  nparticles_sent[EAST] = np_out_east;
  nparticles_sent[WEST] = np_out_west;
  nparticles_sent[NORTH] = np_out_north;
  nparticles_sent[SOUTH] = np_out_south;
  *nparticles_out = np_out_west+np_out_north+np_out_south+np_out_east;
}

// Handle all of the collision events
void handle_collisions(
    const int ntotal_particles, const int nx, const int x_off, const int y_off, 
    Particles* particles, const double* edgex, const double* edgey, 
    RNPool* rn_pools, int* nparticles_dead, double* scalar_flux_tally, 
    double* energy_deposition_tally)
{
  const double inv_ntotal_particles = 1.0/ntotal_particles;
  int np_dead = 0;

  /* HANDLE COLLISIONS */
#pragma omp parallel for simd reduction(+:np_dead)
  for(int ii = 0; ii < ntotal_particles; ++ii) {
    if(particles->next_event[ii] != COLLISION) {
      continue;
    }

    // Don't need to tally into mesh on collision
    int cellx = particles->cellx[ii]-x_off+PAD;
    int celly = particles->celly[ii]-y_off+PAD;
    double number_density = 
      (particles->local_density[ii]*AVOGADROS/MOLAR_MASS);
    double macroscopic_cs_scatter = 
      number_density*particles->microscopic_cs_scatter[ii]*BARNS;
    double macroscopic_cs_absorb = 
      number_density*particles->microscopic_cs_absorb[ii]*BARNS;
    const double distance_to_collision = 
      particles->mfp_to_collision[ii]*particles->cell_mfp[ii];
    double inv_cell_volume = 1.0/(edgex[cellx]*edgey[celly]);

    // Calculate the energy deposition in the cell
    double scalar_flux = particles->weight[ii]*distance_to_collision*inv_cell_volume;
    double energy_deposition = calculate_energy_deposition(
        ii, particles, distance_to_collision, number_density, 
        particles->microscopic_cs_absorb[ii], 
        particles->microscopic_cs_scatter[ii]+particles->microscopic_cs_absorb[ii]);

    // Moves the particles to the collision site
    particles->x[ii] += distance_to_collision*particles->omega_x[ii];
    particles->y[ii] += distance_to_collision*particles->omega_y[ii];

    const double p_absorb = macroscopic_cs_absorb*particles->cell_mfp[ii];

    RNPool* local_rn_pool = &rn_pools[omp_get_thread_num()];
    if(getrand(local_rn_pool) < p_absorb) {
      /* Model particles absorption */

      // Find the new particles weight after absorption, saving the energy change
      const double new_weight = particles->weight[ii]*(1.0 - p_absorb);
      particles->weight[ii] = new_weight;

      if(particles->e[ii] < MIN_ENERGY_OF_INTEREST) {
        // Energy is too low, so mark the particles for deletion
        particles->next_event[ii] = DEAD;
        np_dead++;
      }
    }
    else {
      /* Model elastic particles scattering */

      // Choose a random scattering angle between -1 and 1
      // TODO: THIS RANDOM NUMBER SELECTION DOESN'T WORK
      const double mu_cm = 1.0 - 2.0*getrand(local_rn_pool);

      // Calculate the new energy based on the relation to angle of incidence
      const double e_new = particles->e[ii]*
        (MASS_NO*MASS_NO + 2.0*MASS_NO*mu_cm + 1.0)/
        ((MASS_NO + 1.0)*(MASS_NO + 1.0));

      // Convert the angle into the laboratory frame of reference
      double cos_theta =
        0.5*((MASS_NO+1.0)*sqrt(e_new/particles->e[ii]) - 
            (MASS_NO-1.0)*sqrt(particles->e[ii]/e_new));

      // Alter the direction of the velocities
      const double sin_theta = sin(acos(cos_theta));
      const double omega_x_new =
        (particles->omega_x[ii]*cos_theta - particles->omega_y[ii]*sin_theta);
      const double omega_y_new =
        (particles->omega_x[ii]*sin_theta + particles->omega_y[ii]*cos_theta);
      particles->omega_x[ii] = omega_x_new;
      particles->omega_y[ii] = omega_y_new;
      particles->e[ii] = e_new;
    }

    // Need to store tally information as finished with particles
    update_tallies(
        ii, nx, x_off, y_off, particles, cellx, celly, inv_ntotal_particles, 
        energy_deposition, scalar_flux, scalar_flux_tally, energy_deposition_tally);

    particles->mfp_to_collision[ii] = -log(getrand(local_rn_pool))/macroscopic_cs_scatter;
    particles->dt_to_census[ii] -= distance_to_collision/particles->particle_velocity[ii];
  }

  *nparticles_dead = np_dead;
}

// Handles all of the census events
void handle_census(
    const int ntotal_particles, const int nx, const int x_off, const int y_off, 
    Particles* particles, const double* density, const double* edgex, 
    const double* edgey, double* scalar_flux_tally, double* energy_deposition_tally)
{
  // Not sure these make any difference with the event based approach
  const double inv_ntotal_particles = 1.0/ntotal_particles;

  /* HANDLE THE CENSUS EVENTS */
#pragma omp parallel for simd
  for(int ii = 0; ii < ntotal_particles; ++ii) {
    if(particles->next_event[ii] != CENSUS) {
      continue;
    }

    const double distance_to_census = 
      particles->particle_velocity[ii]*particles->dt_to_census[ii];
    int cellx = particles->cellx[ii]-x_off+PAD;
    int celly = particles->celly[ii]-y_off+PAD;
    double local_density = density[celly*(nx+2*PAD)+cellx];
    double number_density = (local_density*AVOGADROS/MOLAR_MASS);
    double macroscopic_cs_scatter = 
      number_density*particles->microscopic_cs_scatter[ii]*BARNS;
    double macroscopic_cs_absorb = 
      number_density*particles->microscopic_cs_absorb[ii]*BARNS;

    // We have not changed cell or energy level at this stage
    particles->x[ii] += distance_to_census*particles->omega_x[ii];
    particles->y[ii] += distance_to_census*particles->omega_y[ii];
    particles->mfp_to_collision[ii] -= 
      (distance_to_census*(macroscopic_cs_scatter+macroscopic_cs_absorb));

    double inv_cell_volume = 1.0/(edgex[cellx]*edgey[celly]);
    double scalar_flux = particles->weight[ii]*distance_to_census*inv_cell_volume;

    // Calculate the energy deposition in the cell
    double energy_deposition = calculate_energy_deposition(
        ii, particles, distance_to_census, number_density,
        particles->microscopic_cs_absorb[ii], 
        particles->microscopic_cs_scatter[ii]+particles->microscopic_cs_absorb[ii]);

    // Need to store tally information as finished with particles
    update_tallies(
        ii, nx, x_off, y_off, particles, cellx, celly, inv_ntotal_particles, 
        energy_deposition, scalar_flux, scalar_flux_tally, energy_deposition_tally);

    particles->dt_to_census[ii] = 0.0;
  }
}

// Calculates the distance to the facet for all cells
void calc_distance_to_facet(
    const int ntotal_particles, const int x_off, const int y_off, 
    Particles* particles, const double* edgex, const double* edgey)
{
  /* DISTANCE TO FACET */
#pragma omp parallel for simd
  for(int ii = 0; ii < ntotal_particles; ++ii) {
    if(particles->next_event[ii] == DEAD || particles->next_event[ii] == CENSUS) {
      continue;
    }

    // Check the timestep required to move the particles along a single axis
    // If the velocity is positive then the top or right boundary will be hit
    const int cellx = particles->cellx[ii]-x_off+PAD;
    const int celly = particles->celly[ii]-y_off+PAD;
    double u_x_inv = 1.0/(particles->omega_x[ii]*particles->particle_velocity[ii]);
    double u_y_inv = 1.0/(particles->omega_y[ii]*particles->particle_velocity[ii]);

    // The bound is open on the left and bottom so we have to correct for this and
    // required the movement to the facet to go slightly further than the edge
    // in the calculated values, using OPEN_BOUND_CORRECTION, which is the smallest
    // possible distance we can be from the closed bound e.g. 1.0e-14.
    double dt_x = (particles->omega_x[ii] >= 0.0)
      ? ((edgex[cellx+1])-particles->x[ii])*u_x_inv
      : ((edgex[cellx]-OPEN_BOUND_CORRECTION)-particles->x[ii])*u_x_inv;
    double dt_y = (particles->omega_y[ii] >= 0.0)
      ? ((edgey[celly+1])-particles->y[ii])*u_y_inv
      : ((edgey[celly]-OPEN_BOUND_CORRECTION)-particles->y[ii])*u_y_inv;

    // Calculated the projection to be
    // a = vector on first edge to be hit
    // u = velocity vector

    double mag_u0 = particles->particle_velocity[ii];

    particles->x_facet[ii] = (dt_x < dt_y) ? 1 : 0;
    if(particles->x_facet[ii]) {
      // cos(theta) = ||(x, 0)||/||(u_x', u_y')|| - u' is u at boundary
      // cos(theta) = (x.u)/(||x||.||u||)
      // x_x/||u'|| = (x_x, 0)*(u_x, u_y) / (x_x.||u||)
      // x_x/||u'|| = (x_x.u_x / x_x.||u||)
      // x_x/||u'|| = u_x/||u||
      // ||u'|| = (x_x.||u||)/u_x
      // We are centered on the origin, so the y component is 0 after travelling
      // aint the x axis to the edge (ax, 0).(x, y)
      particles->distance_to_facet[ii] = (particles->omega_x[ii] >= 0.0)
        ? ((edgex[cellx+1])-particles->x[ii])*mag_u0*u_x_inv
        : ((edgex[cellx]-OPEN_BOUND_CORRECTION)-particles->x[ii])*mag_u0*u_x_inv;
    }
    else {
      // We are centered on the origin, so the x component is 0 after travelling
      // along the y axis to the edge (0, ay).(x, y)
      particles->distance_to_facet[ii] = (particles->omega_y[ii] >= 0.0)
        ? ((edgey[celly+1])-particles->y[ii])*mag_u0*u_y_inv
        : ((edgey[celly]-OPEN_BOUND_CORRECTION)-particles->y[ii])*mag_u0*u_y_inv;
    }
  }
}

// Tallies both the scalar flux and energy deposition in the cell
#pragma omp declare simd
void update_tallies(
    const int pindex, const int nx, const int x_off, const int y_off, 
    Particles* particles, const int cellx_pad, const int celly_pad, 
    const double inv_ntotal_particles, const double energy_deposition,
    const double scalar_flux, double* scalar_flux_tally, 
    double* energy_deposition_tally)
{
  const int cellx = cellx_pad-PAD;
  const int celly = celly_pad-PAD;

  //#pragma omp atomic update 
  scalar_flux_tally[celly*nx+cellx] += 
    scalar_flux*inv_ntotal_particles; 

//#pragma omp atomic update
  energy_deposition_tally[celly*nx+cellx] += 
    energy_deposition*inv_ntotal_particles;
}

// Sends a particles to a neighbour and replaces in the particles list
void send_and_mark_particle(
    const int destination, const int pindex, Particles* particles)
{
#ifdef MPI
  if(destination == EDGE) {
    return;
  }

  particles->next_event[pindex] = 1;

  // Send the particles
  MPI_Send(
      particles, 1, particle_type, destination, TAG_PARTICLE, MPI_COMM_WORLD);
#else
  TERMINATE("Unreachable - shouldn't send particles unless MPI enabled.\n");
#endif
}

// Calculate the energy deposition in the cell
#pragma omp declare simd
double calculate_energy_deposition(
    const int pindex, Particles* particles, const double path_length, 
    const double number_density, const double microscopic_cs_absorb, 
    const double microscopic_cs_total)
{
  // Calculate the energy deposition based on the path length
  const double average_exit_energy_absorb = 0.0;
  const double pabsorb = (microscopic_cs_absorb/microscopic_cs_total);
  const double absorption_heating = pabsorb*average_exit_energy_absorb;
  const double average_exit_energy_scatter = 
    particles->e[pindex]*((MASS_NO*MASS_NO+MASS_NO+1)/((MASS_NO+1)*(MASS_NO+1)));
  const double scattering_heating = (1.0-pabsorb)*average_exit_energy_scatter;
  const double heating_response =
    (particles->e[pindex]-scattering_heating-absorption_heating);

  return particles->weight[pindex]*path_length*(microscopic_cs_total*BARNS)*
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
    return value[*cs_index];
  }

  // Determine the correct search direction required to move towards the
  // new energy
  const int direction = (energy-cs->value[*cs_index] > 0.0) ? 1 : -1; 

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
    // Use a simple binary search to find the energy group
    ind = cs->nentries/2;
    int width = ind/2;
    while(key[ind-1] > energy || key[ind] < energy) {
      ind += (key[ind] > energy) ? -width : width;
      width = max(1, width/2); // To handle odd cases, allows one extra walk
    }
  }

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

