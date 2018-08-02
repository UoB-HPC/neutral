#include "../../cuda/reduction.k"
#include "../../cuda/shared.h"

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
__device__ double atomicAdd(double* address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN !=
    // NaN)
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

// Fetch the cross section for a particular energy value
__device__ double microscopic_cs_for_energy(const double* key,
                                            const double* value,
                                            const int nentries,
                                            const double energy,
                                            int* cs_index) {

  // Use a simple binary search to find the energy group
  int ind = nentries / 2;
  int width = ind / 2;
  while (energy < key[ind] || energy >= key[ind + 1]) {
    ind += (energy < key[ind]) ? -width : width;
    width = max(1, width / 2); // To handle odd cases, allows one extra walk
  }

  *cs_index = ind;

  // Return the value linearly interpolated
  return value[ind] +
         ((energy - key[ind]) / (key[ind + 1] - key[ind])) *
             (value[ind + 1] - value[ind]);
}

// Calculate the energy deposition in the cell
__device__ double calculate_energy_deposition(
    const double path_length, const double e, const double weight,
    const double number_density, const double microscopic_cs_absorb,
    const double microscopic_cs_total, const double pabsorb) {

  // Calculate the energy deposition based on the path length
  const double average_exit_energy_absorb = 0.0;
  const double absorption_heating = pabsorb * average_exit_energy_absorb;
  const double average_exit_energy_scatter =
      e * ((MASS_NO * MASS_NO + MASS_NO + 1) / ((MASS_NO + 1) * (MASS_NO + 1)));
  const double scattering_heating =
      (1.0 - pabsorb) * average_exit_energy_scatter;
  const double heating_response = (e - scattering_heating - absorption_heating);

  return weight * path_length * (microscopic_cs_total * BARNS) *
         heating_response * number_density;
}

// Generates a pair of random numbers
__device__ void generate_random_numbers(const uint64_t pkey,
                                        const uint64_t master_key,
                                        const uint64_t counter, double* rn0,
                                        double* rn1) {

  threefry2x64_ctr_t ctr;
  threefry2x64_ctr_t key;
  ctr.v[0] = counter;
  ctr.v[1] = 0;
  key.v[0] = pkey;
  key.v[1] = master_key;

  // Generate the random numbers
  threefry2x64_ctr_t rand = threefry2x64(ctr, key);

  // Turn our random numbers from integrals to double precision
  uint64_t max_uint64 = UINT64_C(0xFFFFFFFFFFFFFFFF);
  const double factor = 1.0 / (max_uint64 + 1.0);
  const double half_factor = 0.5 * factor;
  *rn0 = rand.v[0] * factor + half_factor;
  *rn1 = rand.v[1] * factor + half_factor;
}

__global__ void inject_particles_kernel(
    const int local_nx, const int local_ny, const int pad, const int x_off,
    const int y_off, const double local_particle_left_off,
    const double local_particle_bottom_off, const double local_particle_width,
    const double local_particle_height, const int nparticles, const double dt,
    const double initial_energy, const double* edgex, const double* edgey,
    double* x, double* y, int* particle_cellx, int* particle_celly,
    double* omega_x, double* omega_y, double* e, double* weight,
    double* dt_to_census, double* mfp_to_collision) {

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= nparticles)
    return;

  // Generate a pool of random numbers
  double rn[NRANDOM_NUMBERS];
  generate_random_numbers(gid, 0, 0, &rn[0], &rn[1]);

  // Set the initial nandom location of the particle inside the source region
  x[gid] = local_particle_left_off + rn[0] * local_particle_width;
  y[gid] = local_particle_bottom_off + rn[1] * local_particle_height;

  // Check the location of the specific cell that the particle sits within.
  // We have to check this explicitly because the mesh might be non-uniform.
  int cellx = 0;
  int celly = 0;
  for (int cc = 0; cc < local_nx; ++cc) {
    if (x[gid] >= edgex[cc + pad] && x[gid] < edgex[cc + pad + 1]) {
      cellx = x_off + cc;
      break;
    }
  }
  for (int cc = 0; cc < local_ny; ++cc) {
    if (y[gid] >= edgey[cc + pad] && y[gid] < edgey[cc + pad + 1]) {
      celly = y_off + cc;
      break;
    }
  }

  particle_cellx[gid] = cellx;
  particle_celly[gid] = celly;

  // Generating theta has uniform density, however 0.0 and 1.0 produce the same
  // value which introduces very very very small bias...

  generate_random_numbers(gid, 1, 0, &rn[0], &rn[1]);
  const double theta = 2.0 * M_PI * rn[0];
  omega_x[gid] = cos(theta);
  omega_y[gid] = sin(theta);

  // This approximation sets mono-energetic initial state for source particles
  e[gid] = initial_energy;

  // Set a weight for the particle to track absorption
  weight[gid] = 1.0;
  dt_to_census[gid] = dt;
  mfp_to_collision[gid] = 0.0;
}

// Handles an individual particle.
__global__ void handle_particles_kernel(
    const int nparticles, const int global_nx, const int global_ny,
    const int nx, const int ny, const uint64_t master_key, const int pad,
    const int x_off, const int y_off, const double dt, const int initial,
    const int ntotal_particles, const double* density, const double* edgex,
    const double* edgey, const double* edgedx, const double* edgedy,
    double* energy_deposition_tally, int* particle_cellx, int* particle_celly,
    const int cs_scatter_nentries, const int cs_absorb_nentries,
    const double* cs_scatter_keys, const double* cs_scatter_values,
    const double* cs_absorb_keys, const double* cs_absorb_values,
    double* particle_e, double* particle_dt_to_census,
    double* particle_mfp_to_collision, double* particle_weight,
    double* particle_omega_x, double* particle_omega_y, double* particle_x,
    double* particle_y, uint64_t* nfacets_reduce_array,
    uint64_t* ncollisions_reduce_array, uint64_t* nprocessed_reduce_array) {

  // (1) particle can stream and reach census
  // (2) particle can collide and either
  //      - the particle will be absorbed
  //      - the particle will scatter (this presumably means the energy changes)
  // (3) particle encounters boundary region, transports to another cell

  uint64_t nfacets = 0;
  uint64_t ncollisions = 0;
  uint64_t nprocessed = 0;

  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int pind = gid;
  double e = particle_e[pind];

  if (gid < nparticles && e >= MIN_ENERGY_OF_INTEREST) {
    nprocessed++;

    int x_facet = 0;
    int absorb_cs_index = -1;
    int scatter_cs_index = -1;
    double cell_mfp = 0.0;
    uint64_t local_key = 0;

    // Determine the current cell
    int cellx = particle_cellx[pind] - x_off + pad;
    int celly = particle_celly[pind] - y_off + pad;

    // Fetch particle data
    double x = particle_x[pind];
    double y = particle_y[pind];
    double weight = particle_weight[pind];
    double omega_x = particle_omega_x[pind];
    double omega_y = particle_omega_y[pind];
    double local_density = density[celly * (nx + 2 * pad) + cellx];
    double dt_to_census = particle_dt_to_census[pind];
    double mfp_to_collision = particle_mfp_to_collision[pind];

    // Fetch the cross sections and prepare related quantities
    double microscopic_cs_scatter =
        microscopic_cs_for_energy(cs_scatter_keys, cs_scatter_values,
                                  cs_scatter_nentries, e, &scatter_cs_index);
    double microscopic_cs_absorb =
        microscopic_cs_for_energy(cs_absorb_keys, cs_absorb_values,
                                  cs_absorb_nentries, e, &absorb_cs_index);
    double number_density = (local_density * AVOGADROS / MOLAR_MASS);
    double macroscopic_cs_scatter =
        number_density * microscopic_cs_scatter * BARNS;
    double macroscopic_cs_absorb =
        number_density * microscopic_cs_absorb * BARNS;
    double speed = sqrt(2.0 * e * eV_TO_J / PARTICLE_MASS);
    double energy_deposition = 0.0;
    const double inv_ntotal_particles = 1.0 / (double)ntotal_particles;

    double rn[NRANDOM_NUMBERS];

    // Set time to census and MFPs until collision, unless travelled particle
    if (initial) {
      dt_to_census = dt;
      generate_random_numbers(pind, master_key, local_key++, &rn[0], &rn[1]);
      mfp_to_collision = -log(rn[0]) / macroscopic_cs_scatter;
    }

    // Loop until we have reached census
    while (dt_to_census > 0.0) {
      cell_mfp = 1.0 / (macroscopic_cs_scatter + macroscopic_cs_absorb);

      // Check the timestep required to move the particles along a single axis
      // If the speed is positive then the top or right boundary will be hit
      double speed = sqrt(2.0 * e * eV_TO_J / PARTICLE_MASS);
      double u_x_inv = 1.0 / (omega_x * speed);
      double u_y_inv = 1.0 / (omega_y * speed);

      double x0 = edgex[cellx];
      double x1 = edgex[cellx + 1];
      double y0 = edgey[celly];
      double y1 = edgey[celly + 1];

      // The bound is open on the left and bottom so we have to correct for this
      // and required the movement to the facet to go slightly further than the
      // edge in the calculated values, using OPEN_BOUND_CORRECTION, which is
      // the
      // smallest possible distance we can be from the closed bound e.g.
      // 1.0e-14.
      double dt_x = (omega_x >= 0.0)
                        ? (x1 - x) * u_x_inv
                        : ((x0 - OPEN_BOUND_CORRECTION) - x) * u_x_inv;
      double dt_y = (omega_y >= 0.0)
                        ? (y1 - y) * u_y_inv
                        : ((y0 - OPEN_BOUND_CORRECTION) - y) * u_y_inv;

      // Calculated the projection to be
      // a = vector on first edge to be hit
      // u = speed vector
      double mag_u0 = speed;

      // Work out the distance until the particle hits a facet
      double distance_to_facet = 0.0;

      x_facet = (dt_x < dt_y) ? 1 : 0;
      if (x_facet) {
        // We are centered on the origin, so the y component is 0 after
        // travelling along the x axis to the edge (ax, 0).(x, y)
        distance_to_facet =
            (omega_x >= 0.0)
                ? (x1 - x) * mag_u0 * u_x_inv
                : ((x0 - OPEN_BOUND_CORRECTION) - x) * mag_u0 * u_x_inv;
      } else {
        // We are centered on the origin, so the x component is 0 after
        // travelling along the y axis to the edge (0, ay).(x, y)
        distance_to_facet =
            (omega_y >= 0.0)
                ? (y1 - y) * mag_u0 * u_y_inv
                : ((y0 - OPEN_BOUND_CORRECTION) - y) * mag_u0 * u_y_inv;
      }

      // Calculate the distances to next events
      const double distance_to_collision = mfp_to_collision * cell_mfp;
      const double distance_to_census = speed * dt_to_census;
      const double pabsorb = macroscopic_cs_absorb * cell_mfp;

      // Check if our next event is a collision
      if (distance_to_collision < distance_to_facet &&
          distance_to_collision < distance_to_census) {

        /* Collision */

        // Tally the number of collisions
        ncollisions++;

        // Moves the particle to the collision site
        x += distance_to_collision * omega_x;
        y += distance_to_collision * omega_y;
        dt_to_census -= distance_to_collision / speed;

        // Don't need to tally into mesh on collision
        energy_deposition += calculate_energy_deposition(
            distance_to_collision, e, weight, number_density,
            microscopic_cs_absorb,
            microscopic_cs_scatter + microscopic_cs_absorb, pabsorb);

        generate_random_numbers(pind, master_key, local_key++, &rn[0], &rn[1]);

        if (rn[0] < pabsorb) {

          /* Model particle absorption */

          // Find the new particle weight after absorption
          weight *= (1.0 - pabsorb);
        } else {

          /* Model elastic particle scattering */

          // Choose a random scattering angle between -1 and 1
          generate_random_numbers(pind, master_key, local_key++, &rn[0],
                                  &rn[1]);
          const double mu_cm = 1.0 - 2.0 * rn[0];

          // Calculate the new energy based on the relation to angle of
          // incidence
          const double e_new =
              e * (MASS_NO * MASS_NO + 2.0 * MASS_NO * mu_cm + 1.0) /
              ((MASS_NO + 1.0) * (MASS_NO + 1.0));

          // Convert the angle into the laboratory frame of reference
          double cos_theta = 0.5 * ((MASS_NO + 1.0) * sqrt(e_new / e) -
                                    (MASS_NO - 1.0) * sqrt(e / e_new));

          // Alter the direction of the velocities
          const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
          const double omega_x_new =
              (omega_x * cos_theta - omega_y * sin_theta);
          const double omega_y_new =
              (omega_x * sin_theta + omega_y * cos_theta);
          omega_x = omega_x_new;
          omega_y = omega_y_new;
          e = e_new;
          speed = sqrt(2.0 * e * eV_TO_J / PARTICLE_MASS);
        }

        if (e < MIN_ENERGY_OF_INTEREST) {
          // Energy is too low, so mark the particle for deletion
          // Need to store tally information as finished with particle
          atomicAdd(
              &energy_deposition_tally[(celly - pad) * nx + (cellx - pad)],
              energy_deposition * inv_ntotal_particles);
          energy_deposition = 0.0;
          break;
        }

        // Energy has changed so update the cross-sections
        microscopic_cs_scatter = microscopic_cs_for_energy(
            cs_scatter_keys, cs_scatter_values, cs_scatter_nentries, e,
            &scatter_cs_index);
        microscopic_cs_absorb =
            microscopic_cs_for_energy(cs_absorb_keys, cs_absorb_values,
                                      cs_absorb_nentries, e, &absorb_cs_index);
        macroscopic_cs_scatter =
            number_density * microscopic_cs_scatter * BARNS;
        macroscopic_cs_absorb = number_density * microscopic_cs_absorb * BARNS;

        // Re-sample number of mean free paths to collision
        mfp_to_collision = -log(rn[1]) / macroscopic_cs_scatter;
      }
      // Check if we have reached facet
      else if (distance_to_facet < distance_to_census) {

        /* Facet */

        // Tally the number of facets
        nfacets++;

        // Moves the particle to the facet
        x += distance_to_facet * omega_x;
        y += distance_to_facet * omega_y;

        // Update the mean free paths until collision
        mfp_to_collision -= (distance_to_facet *
                             (macroscopic_cs_scatter + macroscopic_cs_absorb));
        dt_to_census -= (distance_to_facet / speed);

        energy_deposition += calculate_energy_deposition(
            distance_to_facet, e, weight, number_density, microscopic_cs_absorb,
            microscopic_cs_scatter + microscopic_cs_absorb, pabsorb);

        // Update tallies as we leave a cell
        atomicAdd(&energy_deposition_tally[(celly - pad) * nx + (cellx - pad)],
                  energy_deposition * inv_ntotal_particles);
        energy_deposition = 0.0;

        if (x_facet) {
          if (omega_x > 0.0) {
            // Reflect at the boundary
            if (cellx >= (global_nx - 1)) {
              omega_x = -(omega_x);
            } else {
              // Moving to right cell
              cellx++;
            }
          } else if (omega_x < 0.0) {
            if (cellx <= 0) {
              // Reflect at the boundary
              omega_x = -(omega_x);
            } else {
              // Moving to left cell
              cellx--;
            }
          }
        } else {
          if (omega_y > 0.0) {
            // Reflect at the boundary
            if (celly >= (global_ny - 1)) {
              omega_y = -(omega_y);
            } else {
              // Moving to north cell
              celly++;
            }
          } else if (omega_y < 0.0) {
            // Reflect at the boundary
            if (celly <= 0) {
              omega_y = -(omega_y);
            } else {
              // Moving to south cell
              celly--;
            }
          }
        }

        // Update the data based on new cell
        local_density = density[celly * (nx + 2 * pad) + cellx];
        number_density = (local_density * AVOGADROS / MOLAR_MASS);
        macroscopic_cs_scatter =
            number_density * microscopic_cs_scatter * BARNS;
        macroscopic_cs_absorb = number_density * microscopic_cs_absorb * BARNS;
      } else {

        /* Census */

        // We have not changed cell or energy level at this stage
        x += distance_to_census * omega_x;
        y += distance_to_census * omega_y;
        dt_to_census = 0.0;
        mfp_to_collision -= (distance_to_facet *
                             (macroscopic_cs_scatter + macroscopic_cs_absorb));

        energy_deposition += calculate_energy_deposition(
            distance_to_census, e, weight, number_density,
            microscopic_cs_absorb,
            microscopic_cs_scatter + microscopic_cs_absorb, pabsorb);

        // Need to store tally information as finished with particle
        atomicAdd(&energy_deposition_tally[(celly - pad) * nx + (cellx - pad)],
                  energy_deposition * inv_ntotal_particles);
        energy_deposition = 0.0;
        break;
      }
    }

    // Store the final particle data
    particle_e[pind] = e;
    particle_x[pind] = x;
    particle_y[pind] = y;
    particle_weight[pind] = weight;
    particle_omega_x[pind] = omega_x;
    particle_omega_y[pind] = omega_y;
    particle_dt_to_census[pind] = dt_to_census;
    particle_mfp_to_collision[pind] = mfp_to_collision;
    particle_cellx[pind] = cellx;
    particle_celly[pind] = celly;
  }

  // Finalise the balance tally reductions
  __shared__ uint64_t nfacets_shared[NTHREADS];
  __shared__ uint64_t ncollisions_shared[NTHREADS];
  __shared__ uint64_t nprocessed_shared[NTHREADS];

  const int tid = threadIdx.x;
  nfacets_shared[tid] = nfacets;
  ncollisions_shared[tid] = ncollisions;
  nprocessed_shared[tid] = nprocessed;

  __syncthreads();

  sum_reduce_in_shared<uint64_t, NTHREADS>(tid, nfacets_shared);
  sum_reduce_in_shared<uint64_t, NTHREADS>(tid, ncollisions_shared);
  sum_reduce_in_shared<uint64_t, NTHREADS>(tid, nprocessed_shared);
  if (tid == 0) {
    nfacets_reduce_array[blockIdx.x] = nfacets_shared[0];
    ncollisions_reduce_array[blockIdx.x] = ncollisions_shared[0];
    nprocessed_reduce_array[blockIdx.x] = nprocessed_shared[0];
  }
}
