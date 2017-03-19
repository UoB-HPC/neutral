#include "../../cuda/shared.h"
#include "../../cuda/reduction.k"

#if __CUDA_ARCH__ < 600 
__device__ double atomicAdd(double* address, double val) 
{ 
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do { 
    assumed = old; 
    old = atomicCAS(address_as_ull, assumed, 
        __double_as_longlong(val + __longlong_as_double(assumed))); 
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) 
  } while (assumed != old); 
  return __longlong_as_double(old); 
} 
#endif

// Fetch the cross section for a particular energy value
__device__ double microscopic_cs_for_energy(
    const double* key, const double* value, const int nentries, 
    const double energy, int* cs_index)
{
  /* Attempt an optimisation of the search by performing a linear operation
   * if there is an existing location. We assume that the energy has
   * reduced rather than increased, which seems to be a legitimate 
   * approximation in this particular case */

  int ind = 0; 

  if(*cs_index > -1) {
    // Determine the correct search direction required to move towards the
    // new energy
    const int direction = (energy > key[*cs_index]) ? 1 : -1; 

    // This search will move in the correct direction towards the new energy group
    int found = 0;
    for(ind = *cs_index; ind >= 0 && ind < nentries; ind += direction) {
      // Check if we have found the new energy group index
      if(energy >= key[ind] && energy < key[ind+1]) {
        found = 1;
        break;
      }
    }

    if(!found) {
      // TODO: ADD AN ERROR HANDLER HERE...
    }
  }
  else {
    // Use a simple binary search to find the energy group
    ind = nentries/2;
    int width = ind/2;
    while(energy < key[ind] || energy >= key[ind+1]) {
      ind += (energy < key[ind]) ? -width : width;
      width = max(1, width/2); // To handle odd cases, allows one extra walk
    }
  }

  *cs_index = ind;

  // TODO: perform some interesting interpolation here
  // Center weighted is poor accuracy but might even out over enough particles
  return 0.5*(value[ind+1] + value[ind]);
}

// Calculate the energy deposition in the cell
__device__ double calculate_energy_deposition(
    const int pindex, const double path_length, const double* e,
    const double* weight, const double number_density, 
    const double microscopic_cs_absorb, const double microscopic_cs_total)
{
  // Calculate the energy deposition based on the path length
  const double average_exit_energy_absorb = 0.0;
  const double pabsorb = (microscopic_cs_absorb/microscopic_cs_total);
  const double absorption_heating = pabsorb*average_exit_energy_absorb;
  const double average_exit_energy_scatter = 
    e[pindex]*((MASS_NO*MASS_NO+MASS_NO+1)/((MASS_NO+1)*(MASS_NO+1)));
  const double scattering_heating = (1.0-pabsorb)*average_exit_energy_scatter;
  const double heating_response =
    (e[pindex]-scattering_heating-absorption_heating);

  return weight[pindex]*path_length*(microscopic_cs_total*BARNS)*
    heating_response*number_density;
}

__device__ void generate_random_numbers(
    const uint64_t master_key, const uint64_t secondary_key, 
    const uint64_t gid, double* rn0, double* rn1)
{
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
  const double factor = 1.0/(max_uint64 + 1.0);
  const double half_factor = 0.5*factor;
  *rn0 = rand.v[0]*factor+half_factor;
  *rn1 = rand.v[1]*factor+half_factor;
}

__global__ void event_initialisation_kernel(
    const int nparticles, const int particles_offset, const double dt,
    const int nx, const int x_off, const int y_off, 
    const int cs_scatter_nentries, const int cs_absorb_nentries,
    const double* e, const int* particle_cellx, const int* particle_celly,
    const double* cs_scatter_keys, const double* cs_scatter_values,
    const double* cs_absorb_keys, const double* cs_absorb_values,
    const double* density, double* dt_to_census, int* particle_event, 
    int* scatter_cs_index, int* absorb_cs_index, double* particle_velocity, 
    double* local_density, double* cell_mfp, double* mfp_to_collision,
    int master_key, double* energy_deposition)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
#if 0
  if(gid*NRANDOM_NUMBERS >= nparticles) return;
#endif // if 0
  if(gid >= nparticles) return;

  double rn[NRANDOM_NUMBERS];
  generate_random_numbers(master_key, 0, gid, &rn[0], &rn[1]);

  // Use all generated random numbers
#if 0
  for(int ii = 0; ii < NRANDOM_NUMBERS; ++ii) {
    const int index = (blockIdx.x*blockDim.x*NRANDOM_NUMBERS)+ii*blockDim.x+threadIdx.x;
    if(index >= nparticles) return;
    const int pindex = particles_offset+index;
#endif // if 0
    const int pindex = particles_offset+gid;
    const int ii = 0;

    dt_to_census[pindex] = dt;

    // Reset living particles
    if(particle_event[pindex] != DEAD) {
      particle_event[pindex] = FACET; 
    }

    const double microscopic_cs_scatter = microscopic_cs_for_energy(
        cs_scatter_keys, cs_scatter_values, cs_scatter_nentries, e[pindex], 
        &scatter_cs_index[pindex]);
    const double microscopic_cs_absorb = microscopic_cs_for_energy(
        cs_absorb_keys, cs_absorb_values, cs_absorb_nentries, e[pindex], 
        &absorb_cs_index[pindex]);
    particle_velocity[pindex] =
      sqrt((2.0*e[pindex]*eV_TO_J)*INV_PARTICLE_MASS);

    const int cellx = particle_cellx[pindex]-x_off+PAD;
    const int celly = particle_celly[pindex]-y_off+PAD;
    local_density[pindex] = density[celly*(nx+2*PAD)+cellx];
    const double number_density = (local_density[pindex]*AVOGADROS/MOLAR_MASS);
    const double macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
    const double macroscopic_cs_absorb = number_density*microscopic_cs_absorb*BARNS;
    cell_mfp[pindex] = 1.0/(macroscopic_cs_scatter+macroscopic_cs_absorb);
    mfp_to_collision[pindex] = -log(rn[ii])/macroscopic_cs_scatter;
    energy_deposition[pindex] = 0.0;
#if 0
  }
#endif // if 0
}

__global__ void calc_next_event_kernel(
    const int nparticles, const int particles_offset, const double* mfp_to_collision, 
    const double* cell_mfp, const double* particle_velocity, const double* dt_to_census, 
    const double* distance_to_facet, int* particle_event, int* reduce_array0,
    int* reduce_array1)
{
  const int tid = threadIdx.x;
  __shared__ int ncol_shared[NTHREADS];
  __shared__ int nfac_shared[NTHREADS];
  ncol_shared[tid] = 0;
  nfac_shared[tid] = 0;

  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  if(gid < nparticles) {
    const int pindex = particles_offset+gid;
    if(particle_event[pindex] != DEAD && particle_event[pindex] != CENSUS) {
      const double distance_to_collision = mfp_to_collision[pindex]*cell_mfp[pindex];
      const double distance_to_census = particle_velocity[pindex]*dt_to_census[pindex];

      if(distance_to_collision < distance_to_census && 
          distance_to_collision < distance_to_facet[pindex]) {
        particle_event[pindex] = COLLISION;
        ncol_shared[tid]++;
      }
      else if(distance_to_facet[pindex] < distance_to_census) {
        particle_event[pindex] = FACET;
        nfac_shared[tid]++;
      }
      else {
        particle_event[pindex] = CENSUS;
      }
    }
  }
  __syncthreads();

  sum_reduce_in_shared<int, NTHREADS>(tid, ncol_shared);
  sum_reduce_in_shared<int, NTHREADS>(tid, nfac_shared);
  if(tid == 0) {
    reduce_array0[blockIdx.x] = ncol_shared[0];
    reduce_array1[blockIdx.x] = nfac_shared[0];
  }
}

__global__ void handle_facets_kernel(
    const int nparticles, const int particles_offset, const int global_nx, 
    const int global_ny, const int nx, const int x_off, const int y_off, 
    const int cs_scatter_nentries, const int cs_absorb_nentries, const double* e, 
    const double* distance_to_facet, const double* weight,
    const double* cs_scatter_keys, const double* cs_scatter_values,
    const double* cs_absorb_keys, const double* cs_absorb_values,
    const double* density, double* energy_deposition, double* x, double* y, 
    double* omega_x, double* omega_y, int* x_facet, int* particle_cellx, 
    int* particle_celly, double* dt_to_census, int* particle_event, 
    int* scatter_cs_index, int* absorb_cs_index, double* particle_velocity, 
    double* local_density, double* cell_mfp, double* mfp_to_collision,
    double* energy_deposition_tally)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;

  for(int ii = gid; ii < nparticles; ii += gridDim.x*blockDim.x) {
    const int pindex = particles_offset+ii;
    if(particle_event[pindex] != FACET) {
      continue;
    }

    const int cellx = particle_cellx[pindex]-x_off+PAD;
    const int celly = particle_celly[pindex]-y_off+PAD;
    double number_density = (local_density[pindex]*AVOGADROS/MOLAR_MASS);

    const double microscopic_cs_scatter = microscopic_cs_for_energy(
        cs_scatter_keys, cs_scatter_values, cs_scatter_nentries, e[pindex], 
        &scatter_cs_index[pindex]);
    const double microscopic_cs_absorb = microscopic_cs_for_energy(
        cs_absorb_keys, cs_absorb_values, cs_absorb_nentries, e[pindex], 
        &absorb_cs_index[pindex]);
    double macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
    double macroscopic_cs_absorb = number_density*microscopic_cs_absorb*BARNS;

    // Update the mean free paths until collision
    mfp_to_collision[pindex] -=
      (distance_to_facet[pindex]*(macroscopic_cs_scatter+macroscopic_cs_absorb));
    dt_to_census[pindex] -=
      (distance_to_facet[pindex]/particle_velocity[pindex]);

    const double ed = calculate_energy_deposition(
        pindex, distance_to_facet[pindex], e, weight, number_density, 
        microscopic_cs_absorb, microscopic_cs_scatter+microscopic_cs_absorb);
    atomicAdd(&energy_deposition_tally[gid],//celly*nx+cellx],
        (energy_deposition[pindex]+ed)/nparticles);
    energy_deposition[pindex] = 0.0;

    // TODO: Make sure that the roundoff is handled here, perhaps actually set it
    // fully to one of the edges here
    x[pindex] += distance_to_facet[pindex]*omega_x[pindex];
    y[pindex] += distance_to_facet[pindex]*omega_y[pindex];

    // This use of x_facet is a slight misnoma, as it is really a facet
    // along the y dimensions
    if(x_facet[pindex]) {
      if(omega_x[pindex] > 0.0) {
        // Reflect at the boundary
        if(particle_cellx[pindex] >= (global_nx-1)) {
          omega_x[pindex] = -(omega_x[pindex]);
        }
        else {
          // Definitely moving to right cell
          particle_cellx[pindex]++;
        }
      }
      else if(omega_x[pindex] < 0.0) {
        if(particle_cellx[pindex] <= 0) {
          // Reflect at the boundary
          omega_x[pindex] = -(omega_x[pindex]);
        }
        else {
          // Definitely moving to left cell
          particle_cellx[pindex]--;
        }
      }
    }
    else {
      if(omega_y[pindex] > 0.0) {
        // Reflect at the boundary
        if(particle_celly[pindex] >= (global_ny-1)) {
          omega_y[pindex] = -(omega_y[pindex]);
        }
        else {
          // Definitely moving to north cell
          particle_celly[pindex]++;
        }
      }
      else if(omega_y[pindex] < 0.0) {
        // Reflect at the boundary
        if(particle_celly[pindex] <= 0) {
          omega_y[pindex] = -(omega_y[pindex]);
        }
        else {
          // Definitely moving to south cell
          particle_celly[pindex]--;
        }
      }
    }

    local_density[pindex] = density[celly*(nx+2*PAD)+cellx];
    number_density = (local_density[pindex]*AVOGADROS/MOLAR_MASS);
    macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
    macroscopic_cs_absorb = number_density*microscopic_cs_absorb*BARNS;
    cell_mfp[pindex] = 1.0/(macroscopic_cs_scatter+macroscopic_cs_absorb);
  }
}

__global__ void handle_collisions_kernel(
    const int nparticles, const int particles_offset,  
    const int nx, const int x_off, const int y_off, 
    const int cs_scatter_nentries, const int cs_absorb_nentries, double* e, 
    const double* distance_to_facet, double* weight,
    const double* cs_scatter_keys, const double* cs_scatter_values,
    const double* cs_absorb_keys, const double* cs_absorb_values,
    double* energy_deposition, double* x, double* y, 
    double* omega_x, double* omega_y, int* x_facet, int* particle_cellx, 
    int* particle_celly, double* dt_to_census, int* particle_event, 
    int* scatter_cs_index, int* absorb_cs_index, double* particle_velocity, 
    double* local_density, double* cell_mfp, double* mfp_to_collision,
    int* reduce_array, const int master_key, double* energy_deposition_tally)
{
  const int nthreads = 1024;
  __shared__ int ndead_shared[nthreads];
  const int tid = threadIdx.x;
  ndead_shared[tid] = 0;

  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  for(int ii = gid; ii < nparticles; ii += gridDim.x*blockDim.x) {
    const int pindex = particles_offset+ii;
    if(particle_event[pindex] == COLLISION) {
      double number_density = 
        (local_density[pindex]*AVOGADROS/MOLAR_MASS);
      const double microscopic_cs_scatter = microscopic_cs_for_energy(
          cs_scatter_keys, cs_scatter_values, cs_scatter_nentries, e[pindex], 
          &scatter_cs_index[pindex]);
      const double microscopic_cs_absorb = microscopic_cs_for_energy(
          cs_absorb_keys, cs_absorb_values, cs_absorb_nentries, e[pindex], 
          &absorb_cs_index[pindex]);
      double macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
      double macroscopic_cs_absorb = number_density*microscopic_cs_absorb*BARNS;
      const double distance_to_collision = mfp_to_collision[pindex]*cell_mfp[pindex];

      energy_deposition[pindex] += calculate_energy_deposition(
          pindex, distance_to_collision, e, weight, number_density, 
          microscopic_cs_absorb, microscopic_cs_scatter+microscopic_cs_absorb);

      // Moves the particles to the collision site
      x[pindex] += distance_to_collision*omega_x[pindex];
      y[pindex] += distance_to_collision*omega_y[pindex];

      const double p_absorb = macroscopic_cs_absorb*cell_mfp[pindex];

      double rn[NRANDOM_NUMBERS];
      generate_random_numbers(master_key, 1, gid, &rn[0], &rn[1]);

      if(rn[0] < p_absorb) {
        /* Model particles absorption */

        // Find the new particles weight after absorption, saving the energy change
        const double new_weight = weight[pindex]*(1.0 - p_absorb);
        weight[pindex] = new_weight;

        if(e[pindex] < MIN_ENERGY_OF_INTEREST) {
          // Energy is too low, so mark the particles for deletion
          particle_event[pindex] = DEAD;
          ndead_shared[tid]++;
          int cellx = particle_cellx[pindex]-x_off+PAD;
          int celly = particle_celly[pindex]-y_off+PAD;
          atomicAdd(&energy_deposition_tally[celly*nx+cellx],
              (energy_deposition[pindex]/nparticles));
          continue;
        }
      }
      else {
        /* Model elastic particles scattering */

        // TODO: HAVE TO DO A REFILL OF THE RN SET, WASTE A SINGLE RN
        generate_random_numbers(master_key, 2, gid, &rn[0], &rn[1]);

        // Choose a random scattering angle between -1 and 1
        const double mu_cm = 1.0 - 2.0*rn[0];

        // Calculate the new energy based on the relation to angle of incidence
        const double e_new = e[pindex]*
          (MASS_NO*MASS_NO + 2.0*MASS_NO*mu_cm + 1.0)/
          ((MASS_NO + 1.0)*(MASS_NO + 1.0));

        // Convert the angle into the laboratory frame of reference
        double cos_theta =
          0.5*((MASS_NO+1.0)*sqrt(e_new/e[pindex]) - 
              (MASS_NO-1.0)*sqrt(e[pindex]/e_new));

        // Alter the direction of the velocities
        const double sin_theta = sqrt(1.0-cos_theta*cos_theta);
        const double omega_x_new =
          (omega_x[pindex]*cos_theta - omega_y[pindex]*sin_theta);
        const double omega_y_new =
          (omega_x[pindex]*sin_theta + omega_y[pindex]*cos_theta);
        omega_x[pindex] = omega_x_new;
        omega_y[pindex] = omega_y_new;
        e[pindex] = e_new;
        particle_velocity[pindex] = sqrt((2.0*e_new*eV_TO_J)*INV_PARTICLE_MASS);
      }

      // TODO: IN THE SCATTERING CASE THIS COULD BE SAVED....
      mfp_to_collision[pindex] = -log(rn[1])/macroscopic_cs_scatter;
      dt_to_census[pindex] -= distance_to_collision/particle_velocity[pindex];
    }
  }

  __syncthreads();
  sum_reduce_in_shared<int, nthreads>(tid, ndead_shared);
  if(tid == 0) {
    reduce_array[blockIdx.x] = ndead_shared[0];
  }
}

__global__ void handle_census_kernel(
    const int nparticles, const int nx, const int x_off, const int y_off, 
    const int particles_offset, const int* particle_event,
    const double* particle_velocity, double* dt_to_census, int* particle_cellx,
    int* particle_celly, double* local_density, const double* e, 
    int* scatter_cs_index, int* absorb_cs_index, double* x, double* y, 
    const double* omega_x, const double* omega_y, double* mfp_to_collision,
    double* energy_deposition, const double* density, const double* cs_scatter_keys,
    const double* cs_absorb_keys, const double* cs_scatter_values, const double* cs_absorb_values,
    const int cs_scatter_nentries, const int cs_absorb_nentries, const double* weight,
    double* energy_deposition_tally)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  const int pindex = particles_offset+gid;
  if(gid >= nparticles || particle_event[pindex] != CENSUS) {
    return;
  }

  const double distance_to_census = 
    particle_velocity[pindex]*dt_to_census[pindex];
  double number_density = (local_density[pindex]*AVOGADROS/MOLAR_MASS);
  const double microscopic_cs_scatter = microscopic_cs_for_energy(
      cs_scatter_keys, cs_scatter_values, cs_scatter_nentries, e[pindex], 
      &scatter_cs_index[pindex]);
  const double microscopic_cs_absorb = microscopic_cs_for_energy(
      cs_absorb_keys, cs_absorb_values, cs_absorb_nentries, e[pindex], 
      &absorb_cs_index[pindex]);
  double macroscopic_cs_scatter = number_density*microscopic_cs_scatter*BARNS;
  double macroscopic_cs_absorb = number_density*microscopic_cs_absorb*BARNS;

  // We have not changed cell or energy level at this stage
  x[pindex] += distance_to_census*omega_x[pindex];
  y[pindex] += distance_to_census*omega_y[pindex];
  mfp_to_collision[pindex] -= 
    (distance_to_census*(macroscopic_cs_scatter+macroscopic_cs_absorb));

  // Calculate the energy deposition in the cell
  const double ed = calculate_energy_deposition(
      pindex, distance_to_census, e, weight, number_density, 
      microscopic_cs_absorb, microscopic_cs_scatter+microscopic_cs_absorb);
  const int cellx = particle_cellx[pindex]-x_off+PAD;
  const int celly = particle_celly[pindex]-y_off+PAD;
  atomicAdd(&energy_deposition_tally[celly*nx+cellx], (energy_deposition[pindex]+ed)/nparticles);
  dt_to_census[pindex] = 0.0;
}

__global__ void calc_distance_to_facet_kernel(
    const int nparticles, const int particles_offset, 
    const int x_off, const int y_off, 
    double* distance_to_facet, double* x, double* y, 
    double* omega_x, double* omega_y, int* x_facet, int* particle_cellx, 
    int* particle_celly, double* dt_to_census, int* particle_event, 
    int* scatter_cs_index, int* absorb_cs_index, double* particle_velocity, 
    double* cell_mfp, double* mfp_to_collision,
    const double* edgex, const double* edgey)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  if(gid >= nparticles) return;

  const int pindex = particles_offset+gid;
  if(particle_event[pindex] == DEAD || particle_event[pindex] == CENSUS) {
    return;
  }

  // Check the timestep required to move the particles along a single axis
  // If the velocity is positive then the top or right boundary will be hit
  const int cellx = particle_cellx[pindex]-x_off+PAD;
  const int celly = particle_celly[pindex]-y_off+PAD;
  double u_x_inv = 1.0/(omega_x[pindex]*particle_velocity[pindex]);
  double u_y_inv = 1.0/(omega_y[pindex]*particle_velocity[pindex]);

  double x0 = edgex[cellx];
  double x1 = edgex[cellx+1];
  double y0 = edgey[celly];
  double y1 = edgey[celly+1];

  // The bound is open on the left and bottom so we have to correct for this and
  // required the movement to the facet to go slightly further than the edge
  // in the calculated values, using OPEN_BOUND_CORRECTION, which is the smallest
  // possible distance we can be from the closed bound e.g. 1.0e-14.
  double dt_x = (omega_x[pindex] >= 0.0)
    ? (x1-x[pindex])*u_x_inv
    : ((x0-OPEN_BOUND_CORRECTION)-x[pindex])*u_x_inv;
  double dt_y = (omega_y[pindex] >= 0.0)
    ? (y1-y[pindex])*u_y_inv
    : ((y0-OPEN_BOUND_CORRECTION)-y[pindex])*u_y_inv;

  // Calculated the projection to be
  // a = vector on first edge to be hit
  // u = velocity vector

  double mag_u0 = particle_velocity[pindex];

  x_facet[pindex] = (dt_x < dt_y) ? 1 : 0;
  if(x_facet[pindex]) {
    // cos(theta) = ||(x, 0)||/||(u_x', u_y')|| - u' is u at boundary
    // cos(theta) = (x.u)/(||x||.||u||)
    // x_x/||u'|| = (x_x, 0)*(u_x, u_y) / (x_x.||u||)
    // x_x/||u'|| = (x_x.u_x / x_x.||u||)
    // x_x/||u'|| = u_x/||u||
    // ||u'|| = (x_x.||u||)/u_x
    // We are centered on the origin, so the y component is 0 after travelling
    // aint the x axis to the edge (ax, 0).(x, y)
    distance_to_facet[pindex] = (omega_x[pindex] >= 0.0)
      ? (x1-x[pindex])*mag_u0*u_x_inv
      : ((x0-OPEN_BOUND_CORRECTION)-x[pindex])*mag_u0*u_x_inv;
  }
  else {
    // We are centered on the origin, so the x component is 0 after travelling
    // along the y axis to the edge (0, ay).(x, y)
    distance_to_facet[pindex] = (omega_y[pindex] >= 0.0)
      ? (y1-y[pindex])*mag_u0*u_y_inv
      : ((y0-OPEN_BOUND_CORRECTION)-y[pindex])*mag_u0*u_y_inv;
  }
}

__global__ void update_tallies_kernel(
    const int nparticles, const int particles_offset, const int tally_census, 
    const int nx, const int x_off, const int y_off,
    const double inv_nparticles_total, int* particle_event, int* particle_cellx, 
    int* particle_celly, double* energy_deposition, double* energy_deposition_tally)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;

  for(int ii = gid; ii < nparticles; ii += blockDim.x*gridDim.x) {
    const int pindex = particles_offset+ii;
    if((!tally_census && particle_event[pindex] != FACET) || 
        (tally_census && particle_event[pindex] != CENSUS)) {
      return;
    }

    double ed = energy_deposition[pindex]*inv_nparticles_total;
    energy_deposition[pindex] = 0.0;
    const int cellx = particle_cellx[pindex]-x_off;
    const int celly = particle_celly[pindex]-y_off;
    atomicAdd(&energy_deposition_tally[celly*nx+cellx], (ed/nparticles));
  }
}

__global__ void inject_particles_kernel(
    const int local_nx, const int local_ny, const int x_off, const int y_off, 
    const double local_particle_left_off, 
    const double local_particle_bottom_off, const double local_particle_width, 
    const double local_particle_height, const int nparticles, const double dt,
    const double initial_energy, const int master_key, const double* edgex,
    const double* edgey, double* x, double* y,
    int* particle_cellx, int* particle_celly, double* omega_x, double* omega_y,
    double* e, double* weight, double* dt_to_census, double* mfp_to_collision,
    int* scatter_cs_index, int* absorb_cs_index, int* particle_event)
{
  const int gid = blockIdx.x*blockDim.x+threadIdx.x;
  if(gid >= nparticles) return;

  double rn[NRANDOM_NUMBERS];
  generate_random_numbers(master_key, 3, gid, &rn[0], &rn[1]);

  // Set the initial nandom location of the particle inside the source region
  x[gid] = local_particle_left_off + rn[0]*local_particle_width;
  y[gid] = local_particle_bottom_off + rn[1]*local_particle_height;

  // Check the location of the specific cell that the particle sits within.
  // We have to check this explicitly because the mesh might be non-uniform.
  int cellx = 0;
  int celly = 0;
  for(int cc = 0; cc < local_nx; ++cc) {
    if(x[gid] >= edgex[cc+PAD] && 
        x[gid] < edgex[cc+PAD+1]) {
      cellx = x_off+cc;
      break;
    }
  }
  for(int cc = 0; cc < local_ny; ++cc) {
    if(y[gid] >= edgey[cc+PAD] && 
        y[gid] < edgey[cc+PAD+1]) {
      celly = y_off+cc;
      break;
    }
  }

  particle_cellx[gid] = cellx;
  particle_celly[gid] = celly;

  // Generating theta has uniform density, however 0.0 and 1.0 produce the same 
  // value which introduces very very very small bias...


  // TODO: THIS CAN BE OPTIMISED, WE COULD BATCH UP AS WITH THE EVENT INITIALISATION
  generate_random_numbers(master_key, 4, gid, &rn[0], &rn[1]);
  const double theta = 2.0*M_PI*rn[0];
  omega_x[gid] = cos(theta);
  omega_y[gid] = sin(theta);

  // This approximation sets mono-energetic initial state for source particles  
  e[gid] = initial_energy;

  // Set a weight for the particle to track absorption
  weight[gid] = 1.0;
  dt_to_census[gid] = dt;
  mfp_to_collision[gid] = 0.0;
  scatter_cs_index[gid] = -1;
  absorb_cs_index[gid] = -1;
  particle_event[gid] = FACET;
}

