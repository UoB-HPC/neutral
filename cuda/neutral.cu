#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include "neutral.h"
#include "neutral.k"
#include "../neutral_interface.h"
#include "../../comms.h"
#include "../../shared.h"
#include "../../shared_data.h"
#include "../../params.h"

#if 0
#ifdef MPI
#include "mpi.h"
#endif
#endif // if 0

// Performs a solve of dependent variables for particles transport.
void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny, 
    const int x_off, const int y_off, const double dt, const int nparticles_total,
    int* nlocal_particles, uint64_t* master_key, const int* neighbours, 
    Particles* particles, const double* density, const double* edgex, 
    const double* edgey, const double* edgedx, const double* edgedy, 
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table, 
    double* scalar_flux_tally, double* energy_deposition_tally, RNPool* rn_pools,
    int* reduce_array0, int* reduce_array1)
{
  // This is the known starting number of particles
  uint64_t facets = 0;
  uint64_t collisions = 0;
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
      edgey, &facets, &collisions, nparticles_sent, master_key, nparticles_total, 
      nparticles, &nparticles, particles, cs_scatter_table, 
      cs_absorb_table, scalar_flux_tally, energy_deposition_tally, rn_pools,
      reduce_array0, reduce_array1);

  *nlocal_particles = nparticles;

  printf("facets %llu collisions %llu\n", facets, collisions);
}

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const double dt, const int* neighbours, 
    const double* density, const double* edgex, const double* edgey, uint64_t* facets, 
    uint64_t* collisions, int* nparticles_sent, uint64_t* master_key, 
    const int nparticles_total, const int nparticles_to_process, 
    int* nparticles, Particles* particles, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table, double* scalar_flux_tally, 
    double* energy_deposition_tally, RNPool* rn_pools, int* reduce_array0,
    int* reduce_array1)
{
  int nthreads                  = 0;

  int nparticles_out = 0;
  int nparticles_dead = 0;

  cudaDeviceSynchronize();

  // Block over the events
  const int block_size = nparticles_total;
  const int nblocks = ceil(nparticles_total/(double)block_size);
  for(int bb = 0; bb < nblocks; ++bb) {
    const int particles_offset = bb*block_size;

    int initialised = 0;

    while(1) {
      START_PROFILING(&compute_profile);
      update_rn_pool_master_keys(rn_pools, 1, (*master_key)++);
      STOP_PROFILING(&compute_profile, "update rn pool master keys");

      /* INITIALISATION */
      if(!initialised) {
        START_PROFILING(&compute_profile);
        event_initialisation(
            block_size, particles_offset, nx, x_off, y_off, particles, dt, 
            density, nthreads, rn_pools, cs_scatter_table, cs_absorb_table);
        initialised = 1;
        STOP_PROFILING(&compute_profile, "initialisation");
      }

      START_PROFILING(&compute_profile);
      const int all_census = calc_next_event(
          block_size, particles_offset, particles, facets, collisions, 
          reduce_array0, reduce_array1, x_off, y_off, edgex, edgey);
      STOP_PROFILING(&compute_profile, "calc next event");

      if(all_census) {
        break;
      }

      START_PROFILING(&compute_profile);
      handle_facets(
          block_size, particles_offset, global_nx, global_ny, nx, ny, x_off, 
          y_off, neighbours, nparticles_sent, particles, edgex, edgey, density, 
          &nparticles_out, scalar_flux_tally, energy_deposition_tally,
          cs_scatter_table, cs_absorb_table);
      STOP_PROFILING(&compute_profile, "handle facets");

      START_PROFILING(&compute_profile);
      handle_collisions( 
          block_size, particles_offset, nx, x_off, y_off, particles, edgex, edgey, 
          rn_pools, &nparticles_dead, cs_scatter_table, cs_absorb_table,
          scalar_flux_tally, energy_deposition_tally, reduce_array0);
      STOP_PROFILING(&compute_profile, "handle collisions");
    }

    START_PROFILING(&compute_profile);
    handle_census(
        block_size, particles_offset, nx, x_off, y_off, particles, density, edgex, 
        edgey, cs_scatter_table, cs_absorb_table, scalar_flux_tally, 
        energy_deposition_tally);
    STOP_PROFILING(&compute_profile, "handle census");
  }

  printf("handled %d particles, with %d particles deleted\n", 
      nparticles_to_process, nparticles_dead+nparticles_out);
}

// Initialises ready for the event cycles
void event_initialisation(
    const int nparticles, const int particles_offset, const int nx, 
    const int x_off, const int y_off, Particles* particles, const double dt, 
    const double* density, const int nthreads, RNPool* rn_pools, 
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table)
{
  RNPool* master_pool = &rn_pools[0];

  // Initialise all of the particles with their starting state
#if 0
  const int nblocks = ceil(nparticles/((double)NTHREADS*NRANDOM_NUMBERS)); 
#endif // if 0
  const int nblocks = ceil(nparticles/((double)NTHREADS)); 
  event_initialisation_kernel<<<nblocks, NTHREADS>>>(
      nparticles, particles_offset, dt, nx, x_off, y_off, cs_scatter_table->nentries, 
      cs_absorb_table->nentries, particles->e, particles->cellx, 
      particles->celly, cs_scatter_table->keys, cs_scatter_table->values, 
      cs_absorb_table->keys, cs_absorb_table->values, 
      cs_scatter_table->log_width, cs_absorb_table->log_width,
      density, particles->dt_to_census, 
      particles->next_event, particles->scatter_cs_index, 
      particles->absorb_cs_index, particles->particle_velocity, 
      particles->local_density, particles->cell_mfp, particles->mfp_to_collision,
      master_pool->key.v[0], particles->energy_deposition);  

  // TODO: BE CAREFUL PASSING MASTER KEY HERE, MAKE SURE IT IS INITIALISED
  // PROPERLY ETC..
}

// Calculates the next event for each particle
int calc_next_event(
    const int nparticles, const int particles_offset, Particles* particles, 
    uint64_t* facets, uint64_t* collisions, int* reduce_array0, int* reduce_array1,
    const int x_off, const int y_off, const double* edgex, const double* edgey)
{
  /* CALCULATE THE EVENTS */
  const int nblocks = ceil(nparticles/(double)NTHREADS); 
  calc_next_event_kernel<<<nblocks, NTHREADS>>>(
      nparticles, particles_offset, particles->mfp_to_collision, 
      particles->cell_mfp, particles->particle_velocity, particles->dt_to_census, 
      particles->distance_to_facet, particles->next_event, reduce_array0,
      reduce_array1, particles->e, x_off, y_off, particles->x, particles->y, 
      particles->omega_x, particles->omega_y, particles->x_facet, 
      particles->cellx, particles->celly, particles->scatter_cs_index, 
      particles->absorb_cs_index, edgex, edgey);

  int nfacets = 0;
  int ncollisions = 0;
  finish_sum_int_reduce(nblocks, reduce_array0, &ncollisions);
  finish_sum_int_reduce(nblocks, reduce_array1, &nfacets);
  *facets += nfacets;
  *collisions += ncollisions;

#if 0
  printf("calculated the events collision %d facets %d census/dead %d\n",
      ncollisions, nfacets, (nparticles-nfacets-ncollisions));
#endif // if 0

  return (!nfacets && !ncollisions);
}

// Handle all of the facet encounters
void handle_facets(
    const int nparticles, const int particles_offset, const int global_nx, 
    const int global_ny, const int nx, const int ny, const int x_off, 
    const int y_off, const int* neighbours, int* nparticles_sent, 
    Particles* particles, const double* edgex, const double* edgey, 
    const double* density, int* nparticles_out, double* scalar_flux_tally, 
    double* energy_deposition_tally, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table)
{
  const int nthreads = NTHREADS;
  const int nblocks = ceil(nparticles/(double)NTHREADS); 
  handle_facets_kernel<<<nblocks, nthreads>>>(
      nparticles, particles_offset, global_nx, global_ny, nx, x_off, y_off, 
      cs_scatter_table->nentries, cs_absorb_table->nentries, particles->e, 
      particles->distance_to_facet, particles->weight, cs_scatter_table->keys, 
      cs_scatter_table->values, cs_absorb_table->keys, cs_absorb_table->values, 
      cs_scatter_table->log_width, cs_absorb_table->log_width, density, 
      particles->energy_deposition, particles->x, particles->y, particles->omega_x, 
      particles->omega_y, particles->x_facet, particles->cellx, particles->celly,
      particles->dt_to_census, particles->next_event, particles->scatter_cs_index,
      particles->absorb_cs_index, particles->particle_velocity, particles->local_density,
      particles->cell_mfp, particles->mfp_to_collision, energy_deposition_tally);

  *nparticles_out = 0;
}

// Handle all of the collision events
void handle_collisions(
    const int nparticles, const int particles_offset, const int nx, 
    const int x_off, const int y_off, Particles* particles, const double* edgex, 
    const double* edgey, RNPool* rn_pools, int* nparticles_dead, 
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table, 
    double* scalar_flux_tally, double* energy_deposition_tally, int* reduce_array)
{
  RNPool* master_pool = &rn_pools[0];

  int np_dead = 0;

  const int nblocks = ceil(nparticles/(double)NTHREADS); 
  handle_collisions_kernel<<<nblocks, NTHREADS>>>(
      nparticles, particles_offset, nx, x_off, y_off, 
      cs_scatter_table->nentries, cs_absorb_table->nentries, particles->e, 
      particles->distance_to_facet, particles->weight, cs_scatter_table->keys, 
      cs_scatter_table->values, cs_absorb_table->keys, cs_absorb_table->values,
      cs_scatter_table->log_width, cs_absorb_table->log_width,
      particles->energy_deposition, particles->x, particles->y, 
      particles->omega_x, particles->omega_y, particles->x_facet, particles->cellx, 
      particles->celly, particles->dt_to_census, particles->next_event, 
      particles->scatter_cs_index, particles->absorb_cs_index, 
      particles->particle_velocity, particles->local_density, particles->cell_mfp, 
      particles->mfp_to_collision, reduce_array, master_pool->key.v[0],
      energy_deposition_tally);

  finish_sum_int_reduce(nblocks, reduce_array, &np_dead);
  *nparticles_dead += np_dead;
}

// Handles all of the census events
void handle_census(
    const int nparticles, const int particles_offset, const int nx, 
    const int x_off, const int y_off, Particles* particles, const double* density, 
    const double* edgex, const double* edgey, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table, double* scalar_flux_tally, 
    double* energy_deposition_tally)
{
  /* HANDLE THE CENSUS EVENTS */
  const int nblocks = ceil(nparticles/(double)NTHREADS); 
  handle_census_kernel<<<nblocks, NTHREADS>>>(
      nparticles, nx, x_off, y_off, particles_offset, particles->next_event, 
      particles->particle_velocity, particles->dt_to_census, particles->cellx, 
      particles->celly, particles->local_density, particles->e, 
      particles->scatter_cs_index, particles->absorb_cs_index, particles->x, 
      particles->y, particles->omega_x, particles->omega_y, particles->mfp_to_collision, 
      particles->energy_deposition, density, cs_scatter_table->keys, 
      cs_absorb_table->keys, cs_scatter_table->values, cs_absorb_table->values,
      cs_scatter_table->log_width, cs_absorb_table->log_width,
      cs_scatter_table->nentries, cs_absorb_table->nentries, particles->weight,
      energy_deposition_tally);
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

#if 0
  // Send the particles
  MPI_Send(
      particles, 1, particle_type, destination, TAG_PARTICLE, MPI_COMM_WORLD);
#endif // if 0
#else
  TERMINATE("Unreachable - shouldn't send particles unless MPI enabled.\n");
#endif
}

// Acts as a particle source
void inject_particles(
    Mesh* mesh, const int local_nx, const int local_ny, 
    const double local_particle_left_off, const double local_particle_bottom_off,
    const double local_particle_width, const double local_particle_height, 
    const int nparticles, const double initial_energy, RNPool* rn_pools,
    Particles* particles)
{
  RNPool* master_pool = &rn_pools[0];

  START_PROFILING(&compute_profile);

  const int nblocks = ceil(nparticles/(double)NTHREADS); 
  inject_particles_kernel<<<nblocks, NTHREADS>>>(
      local_nx, local_ny, mesh->x_off, mesh->y_off, local_particle_left_off, 
      local_particle_bottom_off, local_particle_width, local_particle_height, 
      nparticles, mesh->dt, initial_energy, master_pool->key.v[0], mesh->edgex, mesh->edgey, 
      particles->x, particles->y, particles->cellx, particles->celly, 
      particles->omega_x, particles->omega_y, particles->e, particles->weight, 
      particles->dt_to_census, particles->mfp_to_collision, 
      particles->scatter_cs_index, particles->absorb_cs_index, particles->next_event);

  STOP_PROFILING(&compute_profile, "initialising particles");
}

// Validates the results of the simulation
void validate(
    const int nx, const int ny, const char* params_filename, 
    const int rank, double* energy_deposition_tally)
{
  double* h_energy_deposition_tally;
  allocate_host_data(&h_energy_deposition_tally, nx*ny);
  copy_buffer(nx*ny, &energy_deposition_tally, &h_energy_deposition_tally, RECV);

  double local_energy_tally = 0.0;
  for(int ii = 0; ii < nx*ny; ++ii) {
    local_energy_tally += h_energy_deposition_tally[ii];
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

