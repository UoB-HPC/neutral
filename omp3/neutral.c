#include "neutral.h"
#include "../../comms.h"
#include "../../params.h"
#include "../../shared.h"
#include "../../shared_data.h"
#include "../neutral_interface.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef MPI
#include "mpi.h"
#endif

// Performs a solve of dependent variables for particle transport
void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const int pad, const int x_off, const int y_off, const double dt,
    const int ntotal_particles, int* nlocal_particles, uint64_t* master_key,
    const int* neighbours, Particle* particles, const double* density,
    const double* edgex, const double* edgey, const double* edgedx,
    const double* edgedy, CrossSection* cs_scatter_table,
    CrossSection* cs_absorb_table, double* energy_deposition_tally,
    uint64_t* reduce_array0, uint64_t* reduce_array1, uint64_t* reduce_array2,
    uint64_t* facet_events, uint64_t* collision_events) {

  // This is the known starting number of particles
  int nparticles = *nlocal_particles;
  int nparticles_sent[NNEIGHBOURS];

  if (!nparticles) {
    printf("Out of particles\n");
    return;
  }

  handle_particles(global_nx, global_ny, nx, ny, pad, x_off, y_off, 1, dt,
                   neighbours, density, edgex, edgey, edgedx, edgedy, facet_events,
                   collision_events, nparticles_sent, master_key, ntotal_particles,
                   nparticles, particles, cs_scatter_table, cs_absorb_table,
                   energy_deposition_tally);

  *nlocal_particles = nparticles;
}

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny,
    const int pad, const int x_off, const int y_off, const int initial,
    const double dt, const int* neighbours, const double* density,
    const double* edgex, const double* edgey, const double* edgedx,
    const double* edgedy, uint64_t* facets, uint64_t* collisions,
    int* nparticles_sent, uint64_t* master_key, const int ntotal_particles,
    const int nparticles_to_process, Particle* particles_start,
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table,
    double* energy_deposition_tally) {

  // Maintain a master key, to not encounter the same random number streams
  (*master_key)++;

  int nthreads = 0;
#pragma omp parallel
  { nthreads = omp_get_num_threads(); }

  uint64_t nfacets = 0;
  uint64_t ncollisions = 0;
  uint64_t nparticles = 0;

  const int nb = nparticles_to_process / BLOCK_SIZE;
  const int nb_per_thread = nb / nthreads;
  const int nb_remainder = nb % nthreads;
  const double inv_ntotal_particles = 1.0 / (double)ntotal_particles;

// The main particle loop
#pragma omp parallel reduction(+ : nfacets, ncollisions, nparticles)
  {
    struct Profile tp;

    // (1) particle can stream and reach census
    // (2) particle can collide and either
    //      - the particle will be absorbed
    //      - the particle will scatter (this means the energy changes)
    // (3) particle encounters boundary region, transports to another cell

    const int tid = omp_get_thread_num();

    // Calculate the particles offset, accounting for some remainder
    const int thread_block_off = tid*nb_per_thread*BLOCK_SIZE;

    int counter_off[BLOCK_SIZE];
    double rn[BLOCK_SIZE][NRANDOM_NUMBERS];
    int x_facet[BLOCK_SIZE];
    int absorb_cs_index[BLOCK_SIZE];
    int scatter_cs_index[BLOCK_SIZE];
    double cell_mfp[BLOCK_SIZE];
    double local_density[BLOCK_SIZE];
    double microscopic_cs_scatter[BLOCK_SIZE];
    double microscopic_cs_absorb[BLOCK_SIZE];
    double number_density[BLOCK_SIZE];
    double macroscopic_cs_scatter[BLOCK_SIZE];
    double macroscopic_cs_absorb[BLOCK_SIZE];
    double speed[BLOCK_SIZE];
    double energy_deposition[BLOCK_SIZE];
    double distance_to_facet[BLOCK_SIZE];
    int next_event[BLOCK_SIZE];

    // Populate the counter offset
    for(int cc = 0; cc < BLOCK_SIZE; ++cc) {
      counter_off[cc] = 2*cc;
    }

    for(int b = 0; b < nb_per_thread; ++b) {
      // Current particle
      Particle* particles = &particles_start[thread_block_off + b*BLOCK_SIZE];

      uint64_t counter = 0;

      START_PROFILING(&tp);

      // Initialise cached particle data
#pragma omp simd reduction(+: nparticles, counter)
      for (int ip = 0; ip < BLOCK_SIZE; ++ip) {
        if (particles[ip].dead) {
          continue;
        }

        nparticles++;

        x_facet[ip] = 0;
        absorb_cs_index[ip] = -1;
        scatter_cs_index[ip] = -1;
        cell_mfp[ip] = 0.0;
        energy_deposition[ip] = 0.0;

        // Determine the current cell
        const int cellx = particles[ip].cellx - x_off + pad;
        const int celly = particles[ip].celly - y_off + pad;
        local_density[ip] = density[celly*(nx + 2 * pad) + cellx];

        // Fetch the cross sections and prepare related quantities
        microscopic_cs_scatter[ip] = microscopic_cs_for_energy_binary(
            cs_scatter_table, particles[ip].energy, &scatter_cs_index[ip]);
        microscopic_cs_absorb[ip] = microscopic_cs_for_energy_binary(
            cs_absorb_table, particles[ip].energy, &absorb_cs_index[ip]);
        number_density[ip] = (local_density[ip] * AVOGADROS / MOLAR_MASS);
        macroscopic_cs_scatter[ip] =
            number_density[ip] * microscopic_cs_scatter[ip] * BARNS;
        macroscopic_cs_absorb[ip] =
            number_density[ip] * microscopic_cs_absorb[ip] * BARNS;
        speed[ip] = sqrt((2.0 * particles[ip].energy * eV_TO_J) / PARTICLE_MASS);

        // Set time to census and MFPs until collision, unless travelled
        // particle
        if (initial) {
          particles[ip].dt_to_census = dt;
          generate_random_numbers(*master_key, particles[ip].key, counter++,
                                  &rn[ip][0], &rn[ip][1]);
          particles[ip].mfp_to_collision =
              -log(rn[ip][0]) / macroscopic_cs_scatter[ip];
        }
      }

      STOP_PROFILING(&tp, "cache_init");

      // Loop until we have reached census
      while (1) {
        uint64_t ncompleted = 0;

        START_PROFILING(&tp);
#pragma omp simd reduction(+: ncompleted, ncollisions, nfacets)
        for (int ip = 0; ip < BLOCK_SIZE; ++ip) {
          if (particles[ip].dead) {
            next_event[ip] = PARTICLE_DEAD;
            ncompleted++;
            continue;
          }

          cell_mfp[ip] =
              1.0 / (macroscopic_cs_scatter[ip] + macroscopic_cs_absorb[ip]);

          // Work out the distance until the particle hits a facet
          calc_distance_to_facet(
              global_nx, particles[ip].x, particles[ip].y, pad, x_off, y_off,
              particles[ip].omega_x, particles[ip].omega_y, speed[ip],
              particles[ip].cellx, particles[ip].celly, &distance_to_facet[ip],
              &x_facet[ip], edgex, edgey);
          const double distance_to_collision =
              particles[ip].mfp_to_collision * cell_mfp[ip];
          const double distance_to_census =
              speed[ip] * particles[ip].dt_to_census;

          if (distance_to_collision < distance_to_facet[ip] &&
             distance_to_collision < distance_to_census) {
            next_event[ip] = PARTICLE_COLLISION;
            ncollisions++;
          } else if (distance_to_facet[ip] < distance_to_census) {
            next_event[ip] = PARTICLE_FACET;
            nfacets++;
          } else {
            next_event[ip] = PARTICLE_CENSUS;
            ncompleted++;
          }
        }
        STOP_PROFILING(&tp, "calc_events");

        if (ncompleted == BLOCK_SIZE) {
          break;
        }

        double failed_energy = -1.0;

        START_PROFILING(&tp);
//#pragma omp simd
        for (int ip = 0; ip < BLOCK_SIZE; ++ip) {
          if (next_event[ip] != PARTICLE_COLLISION) {
            continue;
          }

          const double distance_to_collision =
              particles[ip].mfp_to_collision * cell_mfp[ip];
          collision_event(
              global_nx, nx, x_off, y_off, inv_ntotal_particles,
              distance_to_collision, local_density[ip], cs_scatter_table,
              cs_absorb_table, &particles[ip], counter_off[ip] + counter, master_key,
              &energy_deposition[ip], &number_density[ip],
              &microscopic_cs_scatter[ip], &microscopic_cs_absorb[ip],
              &macroscopic_cs_scatter[ip], &macroscopic_cs_absorb[ip],
              energy_deposition_tally, &scatter_cs_index[ip],
              &absorb_cs_index[ip], rn[ip], &speed[ip], &failed_energy);
        }
        STOP_PROFILING(&tp, "collision");

        if(failed_energy >= 0.0) {
          TERMINATE("Could not find lookup table entry for energy %.12f\n", failed_energy);
        }

        // Have to adjust the counter for next usage
        counter += 2*BLOCK_SIZE;

#ifdef TALLY_OUT
        START_PROFILING(&tp);
        for(int ip = 0; ip < BLOCK_SIZE; ++ip) {
          // Store tallies before we perform facet encounter
          if (next_event[ip] != PARTICLE_FACET || 
              (particles[ip].dead && next_event[ip] == PARTICLE_COLLISION)) {
            continue;
          }

          // Update the tallies for all particles leaving cells
          energy_deposition[ip] += calculate_energy_deposition(
              global_nx, nx, x_off, y_off, &particles[ip], inv_ntotal_particles,
              distance_to_facet[ip], number_density[ip], microscopic_cs_absorb[ip],
              microscopic_cs_scatter[ip] + microscopic_cs_absorb[ip]);
          update_tallies(nx, x_off, y_off, &particles[ip], inv_ntotal_particles,
              energy_deposition[ip], energy_deposition_tally);
          energy_deposition[ip] = 0.0;
        }
        STOP_PROFILING(&tp, "energy_deposition");
#endif

        START_PROFILING(&tp);
#pragma omp simd
        for (int ip = 0; ip < BLOCK_SIZE; ++ip) {
          if (next_event[ip] != PARTICLE_FACET) {
            continue;
          }

          facet_event(
              global_nx, global_ny, nx, ny, x_off, y_off, inv_ntotal_particles,
              distance_to_facet[ip], speed[ip], cell_mfp[ip], x_facet[ip],
              density, neighbours, &particles[ip], &energy_deposition[ip],
              &number_density[ip], &microscopic_cs_scatter[ip],
              &microscopic_cs_absorb[ip], &macroscopic_cs_scatter[ip],
              &macroscopic_cs_absorb[ip], energy_deposition_tally,
              nparticles_sent, &local_density[ip]);
        }
        STOP_PROFILING(&tp, "facet");
      }

      START_PROFILING(&tp);
      for (int ip = 0; ip < BLOCK_SIZE; ++ip) {
        if (next_event[ip] != PARTICLE_CENSUS) {
          continue;
        }

        const double distance_to_census = speed[ip] * particles[ip].dt_to_census;
        census_event(global_nx, nx, x_off, y_off, inv_ntotal_particles,
            distance_to_census, cell_mfp[ip], &particles[ip],
            &energy_deposition[ip], &number_density[ip],
            &microscopic_cs_scatter[ip], &microscopic_cs_absorb[ip],
            energy_deposition_tally);
      }
      STOP_PROFILING(&tp, "census");
    }
    PRINT_PROFILING_RESULTS(&tp);
  }

  // Store a total number of facets and collisions
  *facets += nfacets;
  *collisions += ncollisions;


  printf("Particles  %llu\n", nparticles);
}

// Handles a collision event
static inline void collision_event(
    const int global_nx, const int nx, const int x_off, const int y_off,
    const double inv_ntotal_particles, const double distance_to_collision,
    const double local_density, const CrossSection* cs_scatter_table,
    const CrossSection* cs_absorb_table, Particle* particle, uint64_t counter_off,
    const uint64_t* master_key, double* energy_deposition,
    double* number_density, double* microscopic_cs_scatter,
    double* microscopic_cs_absorb, double* macroscopic_cs_scatter,
    double* macroscopic_cs_absorb, double* energy_deposition_tally,
    int* scatter_cs_index, int* absorb_cs_index, double rn[NRANDOM_NUMBERS],
    double* speed, double* failed_energy) {

  // Energy deposition stored locally for collision, not in tally mesh
  *energy_deposition += calculate_energy_deposition(
      global_nx, nx, x_off, y_off, particle, inv_ntotal_particles,
      distance_to_collision, *number_density, *microscopic_cs_absorb,
      *microscopic_cs_scatter + *microscopic_cs_absorb);

  // Moves the particle to the collision site
  particle->x += distance_to_collision * particle->omega_x;
  particle->y += distance_to_collision * particle->omega_y;

  const double p_absorb = *macroscopic_cs_absorb /
    (*macroscopic_cs_scatter + *macroscopic_cs_absorb);

  double rn1[NRANDOM_NUMBERS];
  generate_random_numbers(*master_key, particle->key, counter_off, &rn1[0],
      &rn1[1]);

  if (rn1[0] < p_absorb) {
    /* Model particle absorption */

    // Find the new particle weight after absorption, saving the energy change
    particle->weight *= (1.0 - p_absorb);

    if (particle->energy < MIN_ENERGY_OF_INTEREST) {
      // Energy is too low, so mark the particle for deletion
      particle->dead = 1;

#ifndef TALLY_OUT
      // Update the tallies for all particles leaving cells
      update_tallies(nx, x_off, y_off, particle, inv_ntotal_particles,
          *energy_deposition, energy_deposition_tally);
      *energy_deposition = 0.0;
#endif

    }
  } else {

    /* Model elastic particle scattering */

    // The following assumes that all particles reside within a two-dimensional
    // plane, which solves a different equation. Change so that we consider
    // the full set of directional cosines, allowing scattering between planes.

    // Choose a random scattering angle between -1 and 1
    const double mu_cm = 1.0 - 2.0 * rn1[1];

    // Calculate the new energy based on the relation to angle of incidence
    const double e_new = particle->energy *
      (MASS_NO * MASS_NO + 2.0 * MASS_NO * mu_cm + 1.0) /
      ((MASS_NO + 1.0) * (MASS_NO + 1.0));

    // Convert the angle into the laboratory frame of reference
    double cos_theta = 0.5 * ((MASS_NO + 1.0) * sqrt(e_new / particle->energy) -
        (MASS_NO - 1.0) * sqrt(particle->energy / e_new));

    // Alter the direction of the velocities
    const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    const double omega_x_new =
      (particle->omega_x * cos_theta - particle->omega_y * sin_theta);
    const double omega_y_new =
      (particle->omega_x * sin_theta + particle->omega_y * cos_theta);
    particle->omega_x = omega_x_new;
    particle->omega_y = omega_y_new;
    particle->energy = e_new;
  }

  // Leave if particle is dead
  if (particle->dead) {
    return;
  }

  // Energy has changed so update the cross-sections
  *microscopic_cs_scatter = microscopic_cs_for_energy_binary(
      cs_scatter_table, particle->energy, scatter_cs_index, failed_energy);
  *microscopic_cs_absorb = microscopic_cs_for_energy_binary(
      cs_absorb_table, particle->energy, absorb_cs_index, failed_energy);
  *number_density = (local_density * AVOGADROS / MOLAR_MASS);
  *macroscopic_cs_scatter = *number_density * (*microscopic_cs_scatter) * BARNS;
  *macroscopic_cs_absorb = *number_density * (*microscopic_cs_absorb) * BARNS;

  // Re-sample number of mean free paths to collision
  generate_random_numbers(*master_key, particle->key, counter_off+1, &rn[0],
      &rn[1]);
  particle->mfp_to_collision = -log(rn[0]) / *macroscopic_cs_scatter;
  particle->dt_to_census -= distance_to_collision / *speed;
  *speed = sqrt((2.0 * particle->energy * eV_TO_J) / PARTICLE_MASS);
}

// Handle facet event
static inline void facet_event(const int global_nx, const int global_ny, const int nx,
    const int ny, const int x_off, const int y_off,
    const double inv_ntotal_particles,
    const double distance_to_facet, const double speed,
    const double cell_mfp, const int x_facet, const double* density,
    const int* neighbours, Particle* particle,
    double* energy_deposition, double* number_density,
    double* microscopic_cs_scatter, double* microscopic_cs_absorb,
    double* macroscopic_cs_scatter, double* macroscopic_cs_absorb,
    double* energy_deposition_tally, int* nparticles_sent,
    double* local_density) {

#ifndef TALLY_OUT
  // Update the tallies for all particles leaving cells
  *energy_deposition += calculate_energy_deposition(
      global_nx, nx, x_off, y_off, particle, inv_ntotal_particles,
      distance_to_facet, *number_density, *microscopic_cs_absorb,
      *microscopic_cs_scatter + *microscopic_cs_absorb);
  update_tallies(nx, x_off, y_off, particle, inv_ntotal_particles,
      *energy_deposition, energy_deposition_tally);
  *energy_deposition = 0.0;
#endif

  // Update the mean free paths until collision
  particle->mfp_to_collision -= (distance_to_facet / cell_mfp);
  particle->dt_to_census -= (distance_to_facet / speed);

  // Move the particle to the facet
  particle->x += distance_to_facet * particle->omega_x;
  particle->y += distance_to_facet * particle->omega_y;

  if (x_facet) {
    particle->omega_x = 
      ((particle->cellx >= (global_nx - 1) || particle->cellx <= 0)) 
      ? -particle->omega_x : particle->omega_x;
    particle->cellx += ((particle->omega_x > 0.0) && (particle->cellx < (global_nx - 1))) ? 1 : 0;
    particle->cellx += ((particle->omega_x < 0.0) && (particle->cellx > 0)) ? -1 : 0;
  } else {
    particle->omega_y = 
      ((particle->celly >= (global_ny - 1) || particle->celly <= 0)) 
      ? -particle->omega_y : particle->omega_y;
    particle->celly += ((particle->omega_y > 0.0) && (particle->celly < (global_ny - 1))) ? 1 : 0;
    particle->celly += ((particle->omega_y < 0.0) && (particle->celly > 0)) ? -1 : 0;
  }

  // Update the data based on new cell
  const int cellx = particle->cellx - x_off;
  const int celly = particle->celly - y_off;
  *local_density = density[celly*nx + cellx];
  *number_density = (*local_density * AVOGADROS / MOLAR_MASS);
  *macroscopic_cs_scatter = *number_density * *microscopic_cs_scatter * BARNS;
  *macroscopic_cs_absorb = *number_density * *microscopic_cs_absorb * BARNS;
}

// Handles the census event
static inline void census_event(const int global_nx, const int nx, const int x_off,
    const int y_off, const double inv_ntotal_particles,
    const double distance_to_census, const double cell_mfp,
    Particle* particle, double* energy_deposition,
    double* number_density, double* microscopic_cs_scatter,
    double* microscopic_cs_absorb,
    double* energy_deposition_tally) {

  // We have not changed cell or energy level at this stage
  particle->x += distance_to_census * particle->omega_x;
  particle->y += distance_to_census * particle->omega_y;
  particle->mfp_to_collision -= (distance_to_census / cell_mfp);

  // Need to store tally information as finished with particle
  *energy_deposition += calculate_energy_deposition(
      global_nx, nx, x_off, y_off, particle, inv_ntotal_particles,
      distance_to_census, *number_density, *microscopic_cs_absorb,
      *microscopic_cs_scatter + *microscopic_cs_absorb);
  update_tallies(nx, x_off, y_off, particle, inv_ntotal_particles,
      *energy_deposition, energy_deposition_tally);
  particle->dt_to_census = 0.0;
}

// Tallies the energy deposition in the cell
//#pragma omp declare simd
static inline void update_tallies(const int nx, const int x_off, const int y_off,
    Particle* particle, const double inv_ntotal_particles,
    const double energy_deposition,
    double* energy_deposition_tally) {

  const int cellx = particle->cellx - x_off;
  const int celly = particle->celly - y_off;

#ifdef MANUAL_ATOMIC

  uint64_t* ptr = (uint64_t*)&energy_deposition_tally[celly * nx + cellx];  
  double tally = energy_deposition * inv_ntotal_particles;

  uint64_t old0;
  uint64_t old1 = *ptr;

  do {
    old0 = old1;
    double new = *((double*)&old0) + tally;
    old1 = __sync_val_compare_and_swap(ptr, old0, *((uint64_t*)&new));
  }
  while(old0 != old1);

#else

#pragma omp atomic update
  energy_deposition_tally[celly * nx + cellx] +=
    energy_deposition * inv_ntotal_particles;

#endif
}

// Sends a particle to a neighbour and replaces in the particle list
void send_and_mark_particle(const int destination, Particle* particle) {}

// Calculate the distance to the next facet
static inline void calc_distance_to_facet(const int global_nx, const double x, const double y,
    const int pad, const int x_off, const int y_off,
    const double omega_x, const double omega_y,
    const double speed, const int particle_cellx,
    const int particle_celly, double* distance_to_facet,
    int* x_facet, const double* edgex,
    const double* edgey) {

  // Check the timestep required to move the particle along a single axis
  // If the velocity is positive then the top or right boundary will be hit
  const int cellx = particle_cellx - x_off + pad;
  const int celly = particle_celly - y_off + pad;
  double u_x_inv = 1.0 / (omega_x * speed);
  double u_y_inv = 1.0 / (omega_y * speed);

  // The bound is open on the left and bottom so we have to correct for this
  // and required the movement to the facet to go slightly further than the
  // edge
  // in the calculated values, using OPEN_BOUND_CORRECTION, which is the
  // smallest possible distance from the closed bound e.g. 1.0e-14.
  double dt_x = (omega_x >= 0.0)
    ? ((edgex[cellx + 1]) - x) * u_x_inv
    : ((edgex[cellx] - OPEN_BOUND_CORRECTION) - x) * u_x_inv;
  double dt_y = (omega_y >= 0.0)
    ? ((edgey[celly + 1]) - y) * u_y_inv
    : ((edgey[celly] - OPEN_BOUND_CORRECTION) - y) * u_y_inv;
  *x_facet = (dt_x < dt_y) ? 1 : 0;

  // Calculated the projection to be
  // a = vector on first edge to be hit
  // u = velocity vector

  double mag_u0 = speed;
  *distance_to_facet = (*x_facet) ? dt_x*mag_u0 : dt_y*mag_u0;
}

// Calculate the energy deposition in the cell
static inline double calculate_energy_deposition(
    const int global_nx, const int nx, const int x_off, const int y_off,
    Particle* particle, const double inv_ntotal_particles,
    const double path_length, const double number_density,
    const double microscopic_cs_absorb, const double microscopic_cs_total) {

  // Calculate the energy deposition based on the path length
  const double average_exit_energy_absorb = 0.0;
  const double absorption_heating =
    (microscopic_cs_absorb / microscopic_cs_total) *
    average_exit_energy_absorb;
  const double average_exit_energy_scatter =
    particle->energy *
    ((MASS_NO * MASS_NO + MASS_NO + 1) / ((MASS_NO + 1) * (MASS_NO + 1)));
  const double scattering_heating =
    (1.0 - (microscopic_cs_absorb / microscopic_cs_total)) *
    average_exit_energy_scatter;
  const double heating_response =
    (particle->energy - scattering_heating - absorption_heating);
  return particle->weight * path_length * (microscopic_cs_total * BARNS) *
    heating_response * number_density;
}

// Fetch the cross section for a particular energy value
static inline double microscopic_cs_for_energy_binary(
    const CrossSection* cs, const double energy, int* cs_index) {

  double* keys = cs->keys;
  double* values = cs->values;

  // Use a simple binary search to find the energy group
  int ind = cs->nentries / 2;
  int width = ind / 2;
  while (energy < keys[ind] || energy >= keys[ind + 1]) {
    ind += (energy < keys[ind]) ? -width : width;
    width = max(1, width / 2); // To handle odd cases, allows one extra walk
  }

  *cs_index = ind;

  // Return the value linearly interpolated
  return values[ind] +
    ((energy - keys[ind]) / (keys[ind + 1] - keys[ind])) *
    (values[ind + 1] - values[ind]);
}

// Validates the results of the simulation
void validate(const int nx, const int ny, const char* params_filename,
    const int rank, double* energy_deposition_tally) {

  // Reduce the entire energy deposition tally locally
  double local_energy_tally = 0.0;
  for (int ii = 0; ii < nx * ny; ++ii) {
    local_energy_tally += energy_deposition_tally[ii];
  }

  // Finalise the reduction globally
  double global_energy_tally = reduce_all_sum(local_energy_tally);

  if (rank != MASTER) {
    return;
  }

  printf("\nFinal global_energy_tally %.15e\n", global_energy_tally);

  int nresults = 0;
  char* keys = (char*)malloc(sizeof(char) * MAX_KEYS * (MAX_STR_LEN + 1));
  double* values = (double*)malloc(sizeof(double) * MAX_KEYS);
  if (!get_key_value_parameter(params_filename, NEUTRAL_TESTS, keys, values,
        &nresults)) {
    printf("Warning. Test entry was not found, could NOT validate.\n");
    return;
  }

  // Check the result is within tolerance
  printf("Expected %.12e, result was %.12e.\n", values[0], global_energy_tally);
  if (within_tolerance(values[0], global_energy_tally, VALIDATE_TOLERANCE)) {
    printf("PASSED validation.\n");
  } else {
    printf("FAILED validation.\n");
  }

  free(keys);
  free(values);
}

// Initialises a new particle ready for tracking
size_t inject_particles(const int nparticles, const int global_nx,
    const int local_nx, const int local_ny, const int pad,
    const double local_particle_left_off,
    const double local_particle_bottom_off,
    const double local_particle_width,
    const double local_particle_height, const int x_off,
    const int y_off, const double dt, const double* edgex,
    const double* edgey, const double initial_energy,
    const uint64_t master_key, Particle** particles) {

  *particles = (Particle*)malloc(sizeof(Particle) * nparticles * 2);
  if (!*particles) {
    TERMINATE("Could not allocate particle array.\n");
  }

#pragma omp parallel for
  for (int kk = 0; kk < nparticles; ++kk) {
    Particle* particle = &(*particles)[kk];

    double rn[NRANDOM_NUMBERS];
    generate_random_numbers(master_key, 0, kk, &rn[0], &rn[1]);

    // Set the initial nandom location of the particle inside the source
    // region
    particle->x = local_particle_left_off + rn[0] * local_particle_width;
    particle->y = local_particle_bottom_off + rn[1] * local_particle_height;

    // Check the location of the specific cell that the particle sits within.
    // We have to check this explicitly because the mesh might be non-uniform.
    int cellx = 0;
    int celly = 0;
    for (int ii = 0; ii < local_nx; ++ii) {
      if (particle->x >= edgex[ii + pad] && particle->x < edgex[ii + pad + 1]) {
        cellx = x_off + ii;
        break;
      }
    }
    for (int ii = 0; ii < local_ny; ++ii) {
      if (particle->y >= edgey[ii + pad] && particle->y < edgey[ii + pad + 1]) {
        celly = y_off + ii;
        break;
      }
    }

    particle->cellx = cellx;
    particle->celly = celly;

    // Generating theta has uniform density, however 0.0 and 1.0 produce the
    // same
    // value which introduces very very very small bias...
    generate_random_numbers(master_key, 1, kk, &rn[0], &rn[1]);
    const double theta = 2.0 * M_PI * rn[0];
    particle->omega_x = cos(theta);
    particle->omega_y = sin(theta);

    // This approximation sets mono-energetic initial state for source
    // particles
    particle->energy = initial_energy;

    // Set a weight for the particle to track absorption
    particle->weight = 1.0;
    particle->dt_to_census = dt;
    particle->mfp_to_collision = 0.0;
    particle->dead = 0;
    particle->key = kk;
  }

  return (sizeof(Particle) * nparticles * 2);
}

// Generates a pair of random numbers
void generate_random_numbers(const uint64_t master_key,
    const uint64_t secondary_key, const uint64_t gid,
    double* rn0, double* rn1) {

  threefry2x64_ctr_t counter;
  threefry2x64_ctr_t key;
  counter.v[0] = gid;
  counter.v[1] = 0;
  key.v[0] = master_key;
  key.v[1] = secondary_key;

  // Generate the random numbers
  threefry2x64_ctr_t rand = threefry2x64(counter, key);

  // Turn our random numbers from integrals to double precision
  uint64_t max_uint64 = UINT64_C(0xFFFFFFFFFFFFFFFF);
  const double factor = 1.0 / (max_uint64 + 1.0);
  const double half_factor = 0.5 * factor;
  *rn0 = rand.v[0] * factor + half_factor;
  *rn1 = rand.v[1] * factor + half_factor;
}
