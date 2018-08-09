#include "neutral.h"
#include "../../comms.h"
#include "../../params.h"
#include "../../raja/shared.h"
#include "../../shared.h"
#include "../../shared_data.h"
#include "../neutral_interface.h"
#include "pcg_variants.h"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef MPI
#include "mpi.h"
#endif

#define max(a, b) (((a) > (b)) ? (a) : (b))

#define MASTER_KEY_OFF (1000000000000000ULL)
#define PARTICLE_KEY_OFF (10000ULL)

// Performs a solve of dependent variables for particle transport
void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const uint64_t master_key, const int pad, const int x_off, const int y_off,
    const double dt, const int ntotal_particles, int* nparticles,
    const int* neighbours, Particle* particles, const double* density,
    const double* edgex, const double* edgey, const double* edgedx,
    const double* edgedy, CrossSection* cs_scatter_table,
    CrossSection* cs_absorb_table, double* energy_deposition_tally,
    uint64_t* reduce_array0, uint64_t* reduce_array1, uint64_t* reduce_array2,
    uint64_t* facet_events, uint64_t* collision_events) {

  if (!(*nparticles)) {
    printf("Out of particles\n");
    return;
  }

  handle_particles(global_nx, global_ny, nx, ny, master_key, pad, x_off, y_off,
                   1, dt, neighbours, density, edgex, edgey, edgedx, edgedy,
                   facet_events, collision_events, ntotal_particles,
                   *nparticles, particles, cs_scatter_table, cs_absorb_table,
                   energy_deposition_tally);
}

// Handles the current active batch of particles
void handle_particles(const int global_nx, const int global_ny, const int nx,
                      const int ny, const uint64_t master_key, const int pad,
                      const int x_off, const int y_off, const int initial,
                      const double dt, const int* neighbours,
                      const double* density, const double* edgex,
                      const double* edgey, const double* edgedx,
                      const double* edgedy, uint64_t* facets,
                      uint64_t* collisions, const int ntotal_particles,
                      const int nparticles_to_process,
                      Particle* particles_start, CrossSection* cs_scatter_table,
                      CrossSection* cs_absorb_table,
                      double* energy_deposition_tally) {

  const double* cs_scatter_keys = cs_scatter_table->keys;
  const double* cs_scatter_values = cs_scatter_table->values;
  const int cs_scatter_nentries = cs_scatter_table->nentries;
  const double* cs_absorb_keys = cs_absorb_table->keys;
  const double* cs_absorb_values = cs_absorb_table->values;
  const int cs_absorb_nentries = cs_absorb_table->nentries;

  RAJA::ReduceSum<reduce_policy, uint64_t> nfacets(0);
  RAJA::ReduceSum<reduce_policy, uint64_t> ncollisions(0);
  RAJA::ReduceSum<reduce_policy, uint64_t> nparticles(0);

  RAJA::forall<exec_policy>(
      RAJA::RangeSegment(0, nparticles_to_process), [=] RAJA_DEVICE(int pid) {

        int result = PARTICLE_CONTINUE;

        // (1) particle can stream and reach census
        // (2) particle can collide and either
        //      - the particle will be absorbed
        //      - the particle will scatter (this means the energy changes)
        // (3) particle encounters boundary region, transports to another cell

        // Current particle
        Particle* particle = &particles_start[pid];

        const uint64_t pkey = pid;

        if (!particle->dead) {

          nparticles += 1;

          int x_facet = 0;
          int absorb_cs_index = -1;
          int scatter_cs_index = -1;
          double cell_mfp = 0.0;

          // Determine the current cell
          int cellx = particle->cellx - x_off + pad;
          int celly = particle->celly - y_off + pad;
          double local_density = density[celly * (nx + 2 * pad) + cellx];

          // Fetch the cross sections and prepare related quantities
          double microscopic_cs_scatter = microscopic_cs_for_energy(
              cs_scatter_keys, cs_scatter_values, cs_scatter_nentries, 
              particle->energy, &scatter_cs_index);
          double microscopic_cs_absorb = microscopic_cs_for_energy(
              cs_absorb_keys, cs_absorb_values, cs_absorb_nentries, 
              particle->energy, &absorb_cs_index);
          double number_density = (local_density * AVOGADROS / MOLAR_MASS);
          double macroscopic_cs_scatter =
              number_density * microscopic_cs_scatter * BARNS;
          double macroscopic_cs_absorb =
              number_density * microscopic_cs_absorb * BARNS;
          double speed =
              sqrt((2.0 * particle->energy * eV_TO_J) / PARTICLE_MASS);
          double energy_deposition = 0.0;

          const double inv_ntotal_particles = 1.0 / (double)ntotal_particles;

          uint64_t counter = 0;
          double rn[NRANDOM_NUMBERS];

          // Set time to census and MFPs until collision, unless travelled
          // particle
          if (initial) {
            particle->dt_to_census = dt;
            double rn0 = generate_random_numbers(pkey, master_key, counter++);
            particle->mfp_to_collision = -log(rn0) / macroscopic_cs_scatter;
          }

          // Loop until we have reached census
          while (particle->dt_to_census > 0.0) {
            cell_mfp = 1.0 / (macroscopic_cs_scatter + macroscopic_cs_absorb);

            // Work out the distance until the particle hits a facet
            double distance_to_facet = 0.0;
            calc_distance_to_facet(
                global_nx, particle->x, particle->y, pad, x_off, y_off,
                particle->omega_x, particle->omega_y, speed, particle->cellx,
                particle->celly, &distance_to_facet, &x_facet, edgex, edgey);

            const double distance_to_collision =
                particle->mfp_to_collision * cell_mfp;
            const double distance_to_census = speed * particle->dt_to_census;

            // Check if our next event is a collision
            if (distance_to_collision < distance_to_facet &&
                distance_to_collision < distance_to_census) {

              // Track the total number of collisions
              ncollisions += 1;

              // Handles a collision event
              result = collision_event(
                  global_nx, nx, x_off, y_off, pid, master_key,
                  inv_ntotal_particles, distance_to_collision, local_density,
                  cs_scatter_keys, cs_scatter_values, cs_scatter_nentries, cs_absorb_keys, 
                  cs_absorb_values, cs_absorb_nentries, particle, &counter,
                  &energy_deposition, &number_density, &microscopic_cs_scatter,
                  &microscopic_cs_absorb, &macroscopic_cs_scatter,
                  &macroscopic_cs_absorb, energy_deposition_tally,
                  &scatter_cs_index, &absorb_cs_index, rn, &speed);

              if (result != PARTICLE_CONTINUE) {
                break;
              }
            }
            // Check if we have reached facet
            else if (distance_to_facet < distance_to_census) {

              // Track the number of fact encounters
              nfacets += 1;

              result = facet_event(
                  global_nx, global_ny, nx, ny, x_off, y_off,
                  inv_ntotal_particles, distance_to_facet, speed, cell_mfp,
                  x_facet, density, neighbours, particle, &energy_deposition,
                  &number_density, &microscopic_cs_scatter,
                  &microscopic_cs_absorb, &macroscopic_cs_scatter,
                  &macroscopic_cs_absorb, energy_deposition_tally, &cellx,
                  &celly, &local_density);

              if (result != PARTICLE_CONTINUE) {
                break;
              }

            } else {

              census_event(global_nx, nx, x_off, y_off, inv_ntotal_particles,
                           distance_to_census, cell_mfp, particle,
                           &energy_deposition, &number_density,
                           &microscopic_cs_scatter, &microscopic_cs_absorb,
                           energy_deposition_tally);

              break;
            }
          }
        }
      });

  // Store a total number of facets and collisions
  uint64_t nf = nfacets.get();
  uint64_t nc = ncollisions.get();
  *facets += nf;
  *collisions += nc;

  printf("Particles  %llu\n", nparticles.get());
}

// Handles a collision event
RAJA_DEVICE int collision_event(
    const int global_nx, const int nx, const int x_off, const int y_off,
    const uint64_t pkey, const uint64_t master_key,
    const double inv_ntotal_particles, const double distance_to_collision,
    const double local_density, const double* cs_scatter_keys, 
    const double* cs_scatter_values, const int cs_scatter_nentries, 
    const double* cs_absorb_keys, const double* cs_absorb_values, 
    const int cs_absorb_nentries, Particle* particle, uint64_t* counter,
    double* energy_deposition, double* number_density,
    double* microscopic_cs_scatter, double* microscopic_cs_absorb,
    double* macroscopic_cs_scatter, double* macroscopic_cs_absorb,
    double* energy_deposition_tally, int* scatter_cs_index,
    int* absorb_cs_index, double rn[NRANDOM_NUMBERS], double* speed) {

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

  double rn0 = generate_random_numbers(pkey, master_key, (*counter)++);

  if (rn0 < p_absorb) {
    /* Model particle absorption */

    // Find the new particle weight after absorption, saving the energy change
    particle->weight *= (1.0 - p_absorb);

    if (particle->energy < MIN_ENERGY_OF_INTEREST) {
      // Energy is too low, so mark the particle for deletion
      particle->dead = 1;

      // Need to store tally information as finished with particle
      update_tallies(nx, x_off, y_off, particle, inv_ntotal_particles,
                     *energy_deposition, energy_deposition_tally);
      *energy_deposition = 0.0;
      return PARTICLE_DEAD;
    }
  } else {

    /* Model elastic particle scattering */

    // The following assumes that all particles reside within a two-dimensional
    // plane, which solves a different equation. Change so that we consider
    // the full set of directional cosines, allowing scattering between planes.

    // Choose a random scattering angle between -1 and 1
    double rn1 = generate_random_numbers(pkey, master_key, (*counter)++);
    const double mu_cm = 1.0 - 2.0 * rn1;

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

  // Energy has changed so update the cross-sections
  *microscopic_cs_scatter = microscopic_cs_for_energy(
      cs_scatter_keys, cs_scatter_values, cs_scatter_nentries, particle->energy, scatter_cs_index);
  *microscopic_cs_absorb = microscopic_cs_for_energy(
      cs_absorb_keys, cs_absorb_values, cs_absorb_nentries, particle->energy, absorb_cs_index);
  *number_density = (local_density * AVOGADROS / MOLAR_MASS);
  *macroscopic_cs_scatter = *number_density * (*microscopic_cs_scatter) * BARNS;
  *macroscopic_cs_absorb = *number_density * (*microscopic_cs_absorb) * BARNS;

  // Re-sample number of mean free paths to collision
  double rn2 = generate_random_numbers(pkey, master_key, (*counter)++);
  particle->mfp_to_collision = -log(rn2) / *macroscopic_cs_scatter;
  particle->dt_to_census -= distance_to_collision / *speed;
  *speed = sqrt((2.0 * particle->energy * eV_TO_J) / PARTICLE_MASS);

  return PARTICLE_CONTINUE;
}

// Handle facet event
RAJA_DEVICE int
facet_event(const int global_nx, const int global_ny, const int nx,
            const int ny, const int x_off, const int y_off,
            const double inv_ntotal_particles, const double distance_to_facet,
            const double speed, const double cell_mfp, const int x_facet,
            const double* density, const int* neighbours, Particle* particle,
            double* energy_deposition, double* number_density,
            double* microscopic_cs_scatter, double* microscopic_cs_absorb,
            double* macroscopic_cs_scatter, double* macroscopic_cs_absorb,
            double* energy_deposition_tally, int* cellx, int* celly,
            double* local_density) {

  // Update the mean free paths until collision
  particle->mfp_to_collision -= (distance_to_facet / cell_mfp);
  particle->dt_to_census -= (distance_to_facet / speed);

  *energy_deposition += calculate_energy_deposition(
      global_nx, nx, x_off, y_off, particle, inv_ntotal_particles,
      distance_to_facet, *number_density, *microscopic_cs_absorb,
      *microscopic_cs_scatter + *microscopic_cs_absorb);

  // Update tallies as we leave a cell
  update_tallies(nx, x_off, y_off, particle, inv_ntotal_particles,
                 *energy_deposition, energy_deposition_tally);
  *energy_deposition = 0.0;

  // Move the particle to the facet
  particle->x += distance_to_facet * particle->omega_x;
  particle->y += distance_to_facet * particle->omega_y;

  if (x_facet) {
    if (particle->omega_x > 0.0) {
      // Reflect at the boundary
      if (particle->cellx >= (global_nx - 1)) {
        particle->omega_x = -(particle->omega_x);
      } else {
        // Moving to right cell
        particle->cellx++;
      }
    } else if (particle->omega_x < 0.0) {
      if (particle->cellx <= 0) {
        // Reflect at the boundary
        particle->omega_x = -(particle->omega_x);
      } else {
        // Moving to left cell
        particle->cellx--;
      }
    }
  } else {
    if (particle->omega_y > 0.0) {
      // Reflect at the boundary
      if (particle->celly >= (global_ny - 1)) {
        particle->omega_y = -(particle->omega_y);
      } else {
        // Moving to north cell
        particle->celly++;
      }
    } else if (particle->omega_y < 0.0) {
      // Reflect at the boundary
      if (particle->celly <= 0) {
        particle->omega_y = -(particle->omega_y);
      } else {
        // Moving to south cell
        particle->celly--;
      }
    }
  }

  // Update the data based on new cell
  *cellx = particle->cellx - x_off;
  *celly = particle->celly - y_off;
  *local_density = density[*celly * nx + *cellx];
  *number_density = (*local_density * AVOGADROS / MOLAR_MASS);
  *macroscopic_cs_scatter = *number_density * *microscopic_cs_scatter * BARNS;
  *macroscopic_cs_absorb = *number_density * *microscopic_cs_absorb * BARNS;

  return PARTICLE_CONTINUE;
}

// Handles the census event
RAJA_DEVICE void
census_event(const int global_nx, const int nx, const int x_off,
             const int y_off, const double inv_ntotal_particles,
             const double distance_to_census, const double cell_mfp,
             Particle* particle, double* energy_deposition,
             double* number_density, double* microscopic_cs_scatter,
             double* microscopic_cs_absorb, double* energy_deposition_tally) {

  // We have not changed cell or energy level at this stage
  particle->x += distance_to_census * particle->omega_x;
  particle->y += distance_to_census * particle->omega_y;
  particle->mfp_to_collision -= (distance_to_census / cell_mfp);
  *energy_deposition += calculate_energy_deposition(
      global_nx, nx, x_off, y_off, particle, inv_ntotal_particles,
      distance_to_census, *number_density, *microscopic_cs_absorb,
      *microscopic_cs_scatter + *microscopic_cs_absorb);

  // Need to store tally information as finished with particle
  update_tallies(nx, x_off, y_off, particle, inv_ntotal_particles,
                 *energy_deposition, energy_deposition_tally);

  particle->dt_to_census = 0.0;
}

// Tallies the energy deposition in the cell
RAJA_DEVICE void update_tallies(const int nx, const int x_off, const int y_off,
                                Particle* particle,
                                const double inv_ntotal_particles,
                                const double energy_deposition,
                                double* energy_deposition_tally) {

  const int cellx = particle->cellx - x_off;
  const int celly = particle->celly - y_off;

#if defined(RAJA_USE_CUDA)
  atomicAdd(
      &energy_deposition_tally[celly * nx + cellx],
      energy_deposition*inv_ntotal_particles);
#else
#pragma omp atomic update
      energy_deposition_tally[celly * nx + cellx] += 
        energy_deposition*inv_ntotal_particles;
#endif

#if 0
  RAJA::atomic::atomicAdd<atomic_policy>(
      &energy_deposition_tally[celly * nx + cellx],
      energy_deposition*inv_ntotal_particles);
#endif // if 0
}

// Calculate the distance to the next facet
RAJA_DEVICE void
calc_distance_to_facet(const int global_nx, const double x, const double y,
                       const int pad, const int x_off, const int y_off,
                       const double omega_x, const double omega_y,
                       const double speed, const int particle_cellx,
                       const int particle_celly, double* distance_to_facet,
                       int* x_facet, const double* edgex, const double* edgey) {

  // Check the master_key required to move the particle along a single axis
  // If the velocity is positive then the top or right boundary will be hit
  const int cellx = particle_cellx - x_off + pad;
  const int celly = particle_celly - y_off + pad;
  double u_x_inv = 1.0 / (omega_x * speed);
  double u_y_inv = 1.0 / (omega_y * speed);

  // The bound is open on the left and bottom so we have to correct for this
  // and required the movement to the facet to go slightly further than the edge
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

  if (*x_facet) {
    // We are centered on the origin, so the y component is 0 after travelling
    // aint the x axis to the edge (ax, 0).(x, y)
    *distance_to_facet =
        (omega_x >= 0.0)
            ? ((edgex[cellx + 1]) - x) * mag_u0 * u_x_inv
            : ((edgex[cellx] - OPEN_BOUND_CORRECTION) - x) * mag_u0 * u_x_inv;
  } else {
    // We are centered on the origin, so the x component is 0 after travelling
    // along the y axis to the edge (0, ay).(x, y)
    *distance_to_facet =
        (omega_y >= 0.0)
            ? ((edgey[celly + 1]) - y) * mag_u0 * u_y_inv
            : ((edgey[celly] - OPEN_BOUND_CORRECTION) - y) * mag_u0 * u_y_inv;
  }
}

// Calculate the energy deposition in the cell
RAJA_DEVICE double calculate_energy_deposition(
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
RAJA_DEVICE double microscopic_cs_for_energy(const double* keys, 
    const double* values, const int nentries, const double energy, int* cs_index) {

  // Use a simple binary search to find the energy group
  int ind = nentries / 2;
  int width = ind / 2;
  while (energy < keys[ind] || energy >= keys[ind + 1]) {
    ind += (energy < keys[ind]) ? -width : width;
    width = max(1, width / 2); // To handle odd cases, allows one extra walk
  }

  // Return the value linearly interpolated
  return values[ind] +
         ((energy - keys[ind]) / (keys[ind + 1] - keys[ind])) *
             (values[ind + 1] - values[ind]);
}

// Validates the results of the simulation
void validate(const int nx, const int ny, const char* params_filename,
              const int rank, double* energy_deposition_tally) {

  // Reduce the entire energy deposition tally locally
  RAJA::ReduceSum<reduce_policy, double> local_energy_tally(0.0);

  RAJA::forall<exec_policy>(
      RAJA::RangeSegment(0, nx * ny), [=] RAJA_DEVICE(int ii) {
    local_energy_tally += energy_deposition_tally[ii];
  });

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
                        Particle** particles) {

  Particle* p;

#if defined(RAJA_USE_CUDA)
  cudaMalloc(&p, sizeof(Particle)*nparticles*2);
#else
  p = (Particle*)malloc(sizeof(Particle) * nparticles * 2);
  if (!p) {
    TERMINATE("Could not allocate particle array.\n");
  }
#endif

  START_PROFILING(&compute_profile);

  RAJA::forall<exec_policy>(
      RAJA::RangeSegment(0, nparticles), [=] RAJA_DEVICE(int kk) {

    Particle* particle = &p[kk];

    double rn0 = generate_random_numbers(kk, 0, 0);
    double rn1 = generate_random_numbers(kk, 0, 1);

    // Set the initial nandom location of the particle inside the source
    // region
    particle->x = local_particle_left_off + rn0 * local_particle_width;
    particle->y = local_particle_bottom_off + rn1 * local_particle_height;

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
    // same value which introduces very very very small bias...
    double rn2 = generate_random_numbers(kk, 0, 2);
    const double theta = 2.0 * M_PI * rn2;
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
  });

  STOP_PROFILING(&compute_profile, "initialising particles");

  *particles = p;

  return (sizeof(Particle) * nparticles * 2);
}

RAJA_HOST_DEVICE double my_ldexp(uint64_t val) {
  // Turn our random numbers from integrals to double precision
  uint64_t max_uint64 = UINT64_C(0xFFFFFFFFFFFFFFFF);
  const double factor = 1.0 / (max_uint64 + 1.0);
  const double half_factor = 0.5 * factor;
  return val * factor + half_factor;
}

RAJA_HOST_DEVICE double pcg64u01f_random_r(struct pcg_state_64* rng) {
  const uint64_t uval = pcg64si_random_r(rng);
  const double x = my_ldexp(uval);
  return x;
}

RAJA_HOST_DEVICE double generate_random_numbers(const uint64_t pkey,
                                      const uint64_t master_key,
                                      const uint64_t counter) {

  uint64_t seed =
      counter + MASTER_KEY_OFF * master_key + PARTICLE_KEY_OFF * pkey;

  pcg64si_random_t rng;
  pcg64si_srandom_r(&rng, seed);
  return pcg64u01f_random_r(&rng);
}
