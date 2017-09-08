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

// Performs a solve of dependent variables for particle transport.
void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const int pad, const int x_off, const int y_off, const double dt,
    const int ntotal_particles, int* nlocal_particles, uint64_t* master_key,
    const int* neighbours, Particle* particles, const double* density,
    const double* edgex, const double* edgey, const double* edgedx,
    const double* edgedy, CrossSection* cs_scatter_table,
    CrossSection* cs_absorb_table, double* energy_deposition_tally,
    int* reduce_array0, int* reduce_array1) {
  // This is the known starting number of particles
  uint64_t facets = 0;
  uint64_t collisions = 0;
  int nparticles = *nlocal_particles;
  int nparticles_sent[NNEIGHBOURS];

  if (!nparticles) {
    printf("out of particles\n");
    return;
  }

  // Communication isn't required for edges
  for (int ii = 0; ii < NNEIGHBOURS; ++ii) {
    nparticles_sent[ii] = 0;
  }

  handle_particles(global_nx, global_ny, nx, ny, pad, x_off, y_off, 1, dt,
                   neighbours, density, edgex, edgey, edgedx, edgedy, &facets,
                   &collisions, nparticles_sent, master_key, ntotal_particles,
                   nparticles, &nparticles, particles, cs_scatter_table,
                   cs_absorb_table, energy_deposition_tally);

  *nlocal_particles = nparticles;

  printf("facets %llu collisions %llu\n", facets, collisions);
}

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny,
    const int pad, const int x_off, const int y_off, const int initial,
    const double dt, const int* neighbours, const double* density,
    const double* edgex, const double* edgey, const double* edgedx,
    const double* edgedy, uint64_t* facets, uint64_t* collisions,
    int* nparticles_sent, uint64_t* master_key, const int ntotal_particles,
    const int nparticles_to_process, int* nparticles, Particle* particles_start,
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table,
    double* energy_deposition_tally) {
  // Have to maintain a master key, so that particles don't keep seeing
  // the same random number streams.
  // TODO: THIS IS NOT GOING TO WORK WITH MPI...
  (*master_key)++;

  uint64_t nfacets = 0;
  uint64_t ncollisions = 0;
  int nparticles_deleted = 0;

  // Localising to get around the fact that CCE doesn't support
  // array data that is part of a struct...
  double* cs_scatter_table_keys = cs_scatter_table->keys;
  double* cs_scatter_table_values = cs_scatter_table->values;
  int cs_scatter_table_nentries = cs_scatter_table->nentries;
  double* cs_absorb_table_keys = cs_absorb_table->keys;
  double* cs_absorb_table_values = cs_absorb_table->values;
  int cs_absorb_table_nentries = cs_absorb_table->nentries;

  double* x = particles_start->x;
  double* y = particles_start->y;
  double* omega_x = particles_start->omega_x;
  double* omega_y = particles_start->omega_y;
  double* energy = particles_start->energy;
  double* weight = particles_start->weight;
  double* dt_to_census = particles_start->dt_to_census;
  double* mfp_to_collision = particles_start->mfp_to_collision;
  int* cellx = particles_start->cellx;
  int* celly = particles_start->celly;
  int* dead = particles_start->dead;
  uint64_t local_master_key = *master_key;

#pragma omp target teams distribute parallel for simd map(                     \
    tofrom : ncollisions, nfacets, nparticles_deleted)                         \
        reduction(+ : ncollisions, nfacets, nparticles_deleted)
  for (int pp = 0; pp < nparticles_to_process; ++pp) {
    if (dead[pp]) {
      continue;
    }

    const int result = handle_particle(
        global_nx, global_ny, nx, ny, pad, x_off, y_off, neighbours, dt,
        initial, ntotal_particles, density, edgex, edgey, edgedx, edgedy,
        cs_absorb_table_keys, cs_scatter_table_keys, cs_absorb_table_values,
        cs_scatter_table_values, cs_absorb_table_nentries,
        cs_scatter_table_nentries, nparticles_sent, &nfacets, &ncollisions, pp,
        &x[pp], &y[pp], &omega_x[pp], &omega_y[pp], &energy[pp], &weight[pp],
        &dt_to_census[pp], &mfp_to_collision[pp], &cellx[pp], &celly[pp],
        &dead[pp], energy_deposition_tally, local_master_key);

    nparticles_deleted += (result == PARTICLE_SENT || result == PARTICLE_DEAD);
  }

  *facets = nfacets;
  *collisions = ncollisions;

  printf("handled %d particles, with %d particles deleted\n",
         nparticles_to_process, nparticles_deleted);
}

// Handles an individual particle.
int handle_particle(
    const int global_nx, const int global_ny, const int nx, const int ny,
    const int pad, const int x_off, const int y_off, const int* neighbours,
    const double dt, const int initial, const int ntotal_particles,
    const double* density, const double* edgex, const double* edgey,
    const double* edgedx, const double* edgedy,
    const double* cs_absorb_table_keys, const double* cs_scatter_table_keys,
    const double* cs_absorb_table_values, const double* cs_scatter_table_values,
    const int cs_absorb_table_nentries, const int cs_scatter_table_nentries,
    int* nparticles_sent, uint64_t* facets, uint64_t* collisions, const int pp,
    double* p_x, double* p_y, double* p_omega_x, double* p_omega_y,
    double* p_energy, double* p_weight, double* p_dt_to_census,
    double* p_mfp_to_collision, int* p_cellx, int* p_celly, int* p_dead,
    double* energy_deposition_tally, const uint64_t master_key) {
  // (1) particle can stream and reach census
  // (2) particle can collide and either
  //      - the particle will be absorbed
  //      - the particle will scatter
  // (3) particle hits a boundary region and needs transferring to another
  // process

  int x_facet = 0;
  int absorb_cs_index = -1;
  int scatter_cs_index = -1;
  double cell_mfp = 0.0;
  uint64_t local_key = 0;
  uint64_t counter = pp;

  // Update the cross sections, referencing into the padded mesh
  int cellx = *p_cellx - x_off + pad;
  int celly = *p_celly - y_off + pad;
  double local_density = density[celly * (nx + 2 * pad) + cellx];

  // This makes some assumption about the units of the data stored globally.
  // Might be worth making this more explicit somewhere.
  double microscopic_cs_scatter = microscopic_cs_for_energy(
      cs_scatter_table_keys, cs_scatter_table_values, cs_scatter_table_nentries,
      *p_energy, &scatter_cs_index);
  double microscopic_cs_absorb = microscopic_cs_for_energy(
      cs_absorb_table_keys, cs_absorb_table_values, cs_absorb_table_nentries,
      *p_energy, &absorb_cs_index);
  double number_density = (local_density * AVOGADROS / MOLAR_MASS);
  double macroscopic_cs_scatter =
      number_density * microscopic_cs_scatter * BARNS;
  double macroscopic_cs_absorb = number_density * microscopic_cs_absorb * BARNS;
  double speed = sqrt((2.0 * (*p_energy) * eV_TO_J) / PARTICLE_MASS);
  double energy_deposition = 0.0;
  const double inv_ntotal_particles = 1.0 / (double)ntotal_particles;

  double rn0;
  double rn1;

  // Set time to census and MFPs until collision, unless travelled particle
  if (initial) {
    *p_dt_to_census = dt;
    generate_random_numbers(master_key, local_key++, counter, &rn0, &rn1);
    *p_mfp_to_collision = -log(rn0) / macroscopic_cs_scatter;
  }

  // Loop until we have reached census
  while (*p_dt_to_census > 0.0) {
    cell_mfp = 1.0 / (macroscopic_cs_scatter + macroscopic_cs_absorb);

    // Work out the distance until the particle hits a facet
    double distance_to_facet = 0.0;
    calc_distance_to_facet(global_nx, *p_x, *p_y, pad, x_off, y_off, *p_omega_x,
                           *p_omega_y, speed, *p_cellx, *p_celly,
                           &distance_to_facet, &x_facet, edgex, edgey);

    const double distance_to_collision = (*p_mfp_to_collision) * cell_mfp;
    const double distance_to_census = speed * (*p_dt_to_census);

    // Check if our next event is a collision
    if (distance_to_collision < distance_to_facet &&
        distance_to_collision < distance_to_census) {
      (*collisions)++;

      // Don't need to tally into mesh on collision
      energy_deposition += calculate_energy_deposition(
          global_nx, nx, x_off, y_off, inv_ntotal_particles,
          distance_to_collision, *p_energy, *p_weight, number_density,
          microscopic_cs_absorb,
          microscopic_cs_scatter + microscopic_cs_absorb);

      // The cross sections for scattering and absorption were calculated on
      // a previous iteration for our given energy
      if (handle_collision(p_x, p_y, p_omega_x, p_omega_y, p_energy, p_weight,
                           p_dt_to_census, p_mfp_to_collision, p_cellx, p_celly,
                           p_dead, macroscopic_cs_absorb, counter, &local_key,
                           macroscopic_cs_scatter + macroscopic_cs_absorb,
                           distance_to_collision, master_key)) {

        // Need to store tally information as finished with particle
        update_tallies(nx, x_off, y_off, *p_cellx, *p_celly,
                       inv_ntotal_particles, energy_deposition,
                       energy_deposition_tally);

        return PARTICLE_DEAD;
      }

      // Energy has changed so update the cross-sections
      microscopic_cs_scatter = microscopic_cs_for_energy(
          cs_scatter_table_keys, cs_scatter_table_values,
          cs_scatter_table_nentries, *p_energy, &scatter_cs_index);
      microscopic_cs_absorb = microscopic_cs_for_energy(
          cs_absorb_table_keys, cs_absorb_table_values,
          cs_absorb_table_nentries, *p_energy, &absorb_cs_index);
      number_density = (local_density * AVOGADROS / MOLAR_MASS);
      macroscopic_cs_scatter = number_density * microscopic_cs_scatter * BARNS;
      macroscopic_cs_absorb = number_density * microscopic_cs_absorb * BARNS;

      // Re-sample number of mean free paths to collision
      generate_random_numbers(master_key, local_key++, counter, &rn0, &rn1);
      *p_mfp_to_collision = -log(rn0) / macroscopic_cs_scatter;
      *p_dt_to_census -= distance_to_collision / speed;
      speed = sqrt((2.0 * (*p_energy) * eV_TO_J) / PARTICLE_MASS);
    }
    // Check if we have reached facet
    else if (distance_to_facet < distance_to_census) {
      (*facets)++;

      // Update the mean free paths until collision
      *p_mfp_to_collision -= (distance_to_facet / cell_mfp);
      *p_dt_to_census -= (distance_to_facet / speed);

      // Don't need to tally into mesh on collision
      energy_deposition += calculate_energy_deposition(
          global_nx, nx, x_off, y_off, inv_ntotal_particles, distance_to_facet,
          *p_energy, *p_weight, number_density, microscopic_cs_absorb,
          microscopic_cs_scatter + microscopic_cs_absorb);

      // Update tallies as we leave a cell
      update_tallies(nx, x_off, y_off, *p_cellx, *p_celly, inv_ntotal_particles,
                     energy_deposition, energy_deposition_tally);
      energy_deposition = 0.0;

      // Encounter facet, and jump out if particle left this rank's domain
      if (handle_facet_encounter(global_nx, global_ny, nx, ny, x_off, y_off,
                                 neighbours, distance_to_facet, x_facet,
                                 nparticles_sent, p_x, p_y, p_omega_x,
                                 p_omega_y, p_cellx, p_celly, p_dead)) {
        return PARTICLE_SENT;
      }

      // Update the data based on new cell
      cellx = (*p_cellx) - x_off + pad;
      celly = (*p_celly) - y_off + pad;
      local_density = density[celly * (nx + 2 * pad) + cellx];
      number_density = (local_density * AVOGADROS / MOLAR_MASS);
      macroscopic_cs_scatter = number_density * microscopic_cs_scatter * BARNS;
      macroscopic_cs_absorb = number_density * microscopic_cs_absorb * BARNS;
    }
    // Check if we have reached census
    else {
      // We have not changed cell or energy level at this stage
      *p_x += distance_to_census * (*p_omega_x);
      *p_y += distance_to_census * (*p_omega_y);
      *p_mfp_to_collision -= (distance_to_census / cell_mfp);
      energy_deposition += calculate_energy_deposition(
          global_nx, nx, x_off, y_off, inv_ntotal_particles, distance_to_census,
          *p_energy, *p_weight, number_density, microscopic_cs_absorb,
          microscopic_cs_scatter + microscopic_cs_absorb);

      // Need to store tally information as finished with particle
      update_tallies(nx, x_off, y_off, *p_cellx, *p_celly, inv_ntotal_particles,
                     energy_deposition, energy_deposition_tally);

      *p_dt_to_census = 0.0;
      break;
    }
  }

  return PARTICLE_CENSUS;
}

// Tallies the energy deposition in the cell
void update_tallies(const int nx, const int x_off, const int y_off,
                    const int p_cellx, const int p_celly,
                    const double inv_ntotal_particles,
                    const double energy_deposition,
                    double* energy_deposition_tally) {
  const int cellx = p_cellx - x_off;
  const int celly = p_celly - y_off;

#pragma omp atomic update
  energy_deposition_tally[celly * nx + cellx] +=
      energy_deposition * inv_ntotal_particles;
}

// Handle the collision event, including absorption and scattering
int handle_collision(double* p_x, double* p_y, double* p_omega_x,
                     double* p_omega_y, double* p_energy, double* p_weight,
                     double* p_dt_to_census, double* p_mfp_to_collision,
                     int* p_cellx, int* p_celly, int* p_dead,
                     const double macroscopic_cs_absorb, uint64_t counter,
                     uint64_t* local_key, const double macroscopic_cs_total,
                     const double distance_to_collision, uint64_t master_key) {
  // Moves the particle to the collision site
  *p_x += distance_to_collision * *p_omega_x;
  *p_y += distance_to_collision * *p_omega_y;

  const double p_absorb = macroscopic_cs_absorb / macroscopic_cs_total;

  double rn0;
  double rn1;

  generate_random_numbers(master_key, (*local_key)++, counter, &rn0, &rn1);

  if (rn0 < p_absorb) {
    /* Model particle absorption */

    // Find the new particle weight after absorption, saving the energy change
    *p_weight *= (1.0 - p_absorb);

    if (*p_energy < MIN_ENERGY_OF_INTEREST) {
      // Energy is too low, so mark the particle for deletion
      *p_dead = 1;
    }
  } else {

    /* Model elastic particle scattering */

    // TODO: This approximation is not realistic as far as I can tell.
    // This considers that all particles reside within a single two-dimensional
    // plane, which solves a different equation. Change so that we consider the
    // full set of directional cosines, allowing scattering between planes.
    // Choose a random scattering angle between -1 and 1
    const double mu_cm = 1.0 - 2.0 * rn1;

    // Calculate the new energy based on the relation to angle of incidence
    const double e_new = (*p_energy) *
                         (MASS_NO * MASS_NO + 2.0 * MASS_NO * mu_cm + 1.0) /
                         ((MASS_NO + 1.0) * (MASS_NO + 1.0));

    // Convert the angle into the laboratory frame of reference
    double cos_theta = 0.5 * ((MASS_NO + 1.0) * sqrt(e_new / (*p_energy)) -
                              (MASS_NO - 1.0) * sqrt((*p_energy) / e_new));

    // Alter the direction of the velocities
    const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    const double omega_x_new =
        ((*p_omega_x) * cos_theta - (*p_omega_y) * sin_theta);
    const double omega_y_new =
        ((*p_omega_x) * sin_theta + (*p_omega_y) * cos_theta);
    *p_omega_x = omega_x_new;
    *p_omega_y = omega_y_new;
    *p_energy = e_new;
  }

  return *p_dead;
}

// Makes the necessary updates to the particle given that
// the facet was encountered
int handle_facet_encounter(const int global_nx, const int global_ny,
                           const int nx, const int ny, const int x_off,
                           const int y_off, const int* neighbours,
                           const double distance_to_facet, int x_facet,
                           int* nparticles_sent, double* p_x, double* p_y,
                           double* p_omega_x, double* p_omega_y, int* p_cellx,
                           int* p_celly, int* p_dead) {
  // TODO: Make sure that the roundoff is handled here, perhaps actually set it
  // fully to one of the edges here
  *p_x += distance_to_facet * (*p_omega_x);
  *p_y += distance_to_facet * (*p_omega_y);

  // This use of x_facet is a slight misnoma, as it is really a facet
  // along the y dimensions
  if (x_facet) {
    if (*p_omega_x > 0.0) {
      // Reflect at the boundary
      if (*p_cellx >= (global_nx - 1)) {
        *p_omega_x = -(*p_omega_x);
      } else {
        // Definitely moving to right cell
        (*p_cellx)++;

#if 0
        // Check if we need to pass to another process
        if(*p_cellx >= nx+x_off) {
          send_and_mark_particle(neighbours[EAST], p_dead);
          nparticles_sent[EAST]++;
          return 1;
        }
#endif // if 0
      }
    } else if (*p_omega_x < 0.0) {
      if (*p_cellx <= 0) {
        // Reflect at the boundary
        *p_omega_x = -(*p_omega_x);
      } else {
        // Definitely moving to left cell
        (*p_cellx)--;

#if 0
        // Check if we need to pass to another process
        if(*p_cellx < x_off) {
          send_and_mark_particle(neighbours[WEST], p_dead);
          nparticles_sent[WEST]++;
          return 1;
        }
#endif // if 0
      }
    }
  } else {
    if (*p_omega_y > 0.0) {
      // Reflect at the boundary
      if (*p_celly >= (global_ny - 1)) {
        *p_omega_y = -(*p_omega_y);
      } else {
        // Definitely moving to north cell
        (*p_celly)++;

#if 0
        // Check if we need to pass to another process
        if(*p_celly >= ny+y_off) {
          send_and_mark_particle(neighbours[NORTH], p_dead);
          nparticles_sent[NORTH]++;
          return 1;
        }
#endif // if 0
      }
    } else if (*p_omega_y < 0.0) {
      // Reflect at the boundary
      if (*p_celly <= 0) {
        *p_omega_y = -(*p_omega_y);
      } else {
        // Definitely moving to south cell
        (*p_celly)--;

#if 0
        // Check if we need to pass to another process
        if(*p_celly < y_off) {
          send_and_mark_particle(neighbours[SOUTH], p_dead);
          nparticles_sent[SOUTH]++;
          return 1;
        }
#endif // if 0
      }
    }
  }

  return 0;
}

// Sends a particle to a neighbour and replaces in the particle list
void send_and_mark_particle(const int destination, int* p_dead) {
#if 0
#ifdef MPI
  if(destination == EDGE) {
    return;
  }

  *p_dead = 1;

  // Send the particle
  MPI_Send(
      particle, 1, particle_type, destination, TAG_PARTICLE, MPI_COMM_WORLD);
#else
  TERMINATE("Unreachable - shouldn't send particles unless MPI enabled.\n");
#endif
#endif // if 0
}

// Calculate the distance to the next facet
void calc_distance_to_facet(const int global_nx, const double x, const double y,
                            const int pad, const int x_off, const int y_off,
                            const double omega_x, const double omega_y,
                            const double speed, const int p_cellx,
                            const int p_celly, double* distance_to_facet,
                            int* x_facet, const double* edgex,
                            const double* edgey) {
  // Check the timestep required to move the particle along a single axis
  // If the velocity is positive then the top or right boundary will be hit
  const int cellx = p_cellx - x_off + pad;
  const int celly = p_celly - y_off + pad;
  double u_x_inv = 1.0 / (omega_x * speed);
  double u_y_inv = 1.0 / (omega_y * speed);

  // The bound is open on the left and bottom so we have to correct for this and
  // required the movement to the facet to go slightly further than the edge
  // in the calculated values, using OPEN_BOUND_CORRECTION, which is the
  // smallest
  // possible distance we can be from the closed bound energy.g. 1.0e-14.
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
    // cos(theta) = ||(x, 0)||/||(u_x', u_y')|| - u' is u at boundary
    // cos(theta) = (x.u)/(||x||.||u||)
    // x_x/||u'|| = (x_x, 0)*(u_x, u_y) / (x_x.||u||)
    // x_x/||u'|| = (x_x.u_x / x_x.||u||)
    // x_x/||u'|| = u_x/||u||
    // ||u'|| = (x_x.||u||)/u_x
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
double calculate_energy_deposition(
    const int global_nx, const int nx, const int x_off, const int y_off,
    const double inv_ntotal_particles, const double path_length,
    const double p_energy, const double p_weight, const double number_density,
    const double microscopic_cs_absorb, const double microscopic_cs_total) {
  // Calculate the energy deposition based on the path length
  const double average_exit_energy_absorb = 0.0;
  const double absorption_heating =
      (microscopic_cs_absorb / microscopic_cs_total) *
      average_exit_energy_absorb;
  const double average_exit_energy_scatter =
      p_energy *
      ((MASS_NO * MASS_NO + MASS_NO + 1) / ((MASS_NO + 1) * (MASS_NO + 1)));
  const double scattering_heating =
      (1.0 - (microscopic_cs_absorb / microscopic_cs_total)) *
      average_exit_energy_scatter;
  const double heating_response =
      (p_energy - scattering_heating - absorption_heating);
  return p_weight * path_length * (microscopic_cs_total * BARNS) *
         heating_response * number_density;
}

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy(const double* keys, const double* values,
                                 const int nentries, const double energy,
                                 int* cs_index) {
  int ind = 0;

  if (*cs_index > -1) {
    // Determine the correct search direction required to move towards the
    // new energy
    const int direction = (energy > keys[*cs_index]) ? 1 : -1;

    // This search will move in the correct direction towards the new energy
    // group
    int found = 0;
    for (ind = *cs_index; ind >= 0 && ind < nentries; ind += direction) {
      // Check if we have found the new energy group index
      if (energy >= keys[ind] && energy < keys[ind + 1]) {
        found = 1;
        break;
      }
    }
  } else {
    // Use a simple binary search to find the energy group
    ind = nentries / 2;
    int width = ind / 2;
    while (energy < keys[ind] || energy >= keys[ind + 1]) {
      ind += (energy < keys[ind]) ? -width : width;
      width = max(1, width / 2); // To handle odd cases, allows one extra walk
    }
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
  double local_energy_tally = 0.0;

#pragma omp target teams distribute parallel for map(                          \
    tofrom : local_energy_tally) reduction(+ : local_energy_tally)
  for (int ii = 0; ii < nx * ny; ++ii) {
    local_energy_tally += energy_deposition_tally[ii];
  }

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
  *particles = (Particle*)malloc(sizeof(Particle));
  if (!*particles) {
    TERMINATE("Could not allocate particle array.\n");
  }

  Particle* particle = *particles;
  size_t allocation = 0;
  allocation += allocate_data(&particle->x, nparticles * 1.5);
  allocation += allocate_data(&particle->y, nparticles * 1.5);
  allocation += allocate_data(&particle->omega_x, nparticles * 1.5);
  allocation += allocate_data(&particle->omega_y, nparticles * 1.5);
  allocation += allocate_data(&particle->energy, nparticles * 1.5);
  allocation += allocate_data(&particle->weight, nparticles * 1.5);
  allocation += allocate_data(&particle->dt_to_census, nparticles * 1.5);
  allocation += allocate_data(&particle->mfp_to_collision, nparticles * 1.5);
  allocation += allocate_int_data(&particle->cellx, nparticles * 1.5);
  allocation += allocate_int_data(&particle->celly, nparticles * 1.5);
  allocation += allocate_int_data(&particle->dead, nparticles * 1.5);

  double* p_x = particle->x;
  double* p_y = particle->y;
  double* p_omega_x = particle->omega_x;
  double* p_omega_y = particle->omega_y;
  double* p_energy = particle->energy;
  double* p_weight = particle->weight;
  double* p_dt_to_census = particle->dt_to_census;
  double* p_mfp_to_collision = particle->mfp_to_collision;
  int* p_cellx = particle->cellx;
  int* p_celly = particle->celly;
  int* p_dead = particle->dead;

  START_PROFILING(&compute_profile);

#pragma omp target teams distribute parallel for
  for (int pp = 0; pp < nparticles; ++pp) {
    double rn0;
    double rn1;
    generate_random_numbers(master_key, 0, pp, &rn0, &rn1);

    // Set the initial nandom location of the particle inside the source region
    p_x[pp] = local_particle_left_off + rn0 * local_particle_width;
    p_y[pp] = local_particle_bottom_off + rn1 * local_particle_height;

    // Check the location of the specific cell that the particle sits within.
    // We have to check this explicitly because the mesh might be non-uniform.
    int cellx = 0;
    int celly = 0;
    for (int pp = 0; pp < local_nx; ++pp) {
      if (p_x[pp] >= edgex[pp + pad] && p_x[pp] < edgex[pp + pad + 1]) {
        cellx = x_off + pp;
        break;
      }
    }
    for (int pp = 0; pp < local_ny; ++pp) {
      if (p_y[pp] >= edgey[pp + pad] && p_y[pp] < edgey[pp + pad + 1]) {
        celly = y_off + pp;
        break;
      }
    }

    p_cellx[pp] = cellx;
    p_celly[pp] = celly;

    // Generating theta has uniform density, however 0.0 and 1.0 produce the
    // same
    // value which introduces very very very small bias...
    generate_random_numbers(master_key, 1, pp, &rn0, &rn1);
    const double theta = 2.0 * M_PI * rn0;
    p_omega_x[pp] = cos(theta);
    p_omega_y[pp] = sin(theta);

    // This approximation sets mono-energetic initial state for source particles
    p_energy[pp] = initial_energy;

    // Set a weight for the particle to track absorption
    p_weight[pp] = 1.0;
    p_dt_to_census[pp] = dt;
    p_mfp_to_collision[pp] = 0.0;
    p_dead[pp] = 0;
  }

  STOP_PROFILING(&compute_profile, "initialising particles");

  return allocation;
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
