#include "../../comms.h"
#include "../../cuda/shared.h"
#include "../../params.h"
#include "../../shared.h"
#include "../../shared_data.h"
#include "../neutral_interface.h"
#include "neutral.h"
#include "neutral.k"
#include <assert.h>
#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Performs a solve of dependent variables for particle transport.
void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const int pad, const int x_off, const int y_off, const float dt,
    const int nparticles_total, int* nlocal_particles, uint64_t* master_key,
    const int* neighbours, Particle* particles, const float* density,
    const float* edgex, const float* edgey, const float* edgedx,
    const float* edgedy, CrossSection* cs_scatter_table,
    CrossSection* cs_absorb_table, float* energy_deposition_tally,
    uint64_t* nfacets_reduce_array, uint64_t* ncollisions_reduce_array,
    uint64_t* nprocessed_reduce_array, uint64_t* facet_events, 
    uint64_t* collision_events) {

  // This is the known starting number of particles
  int nparticles = *nlocal_particles;
  int nparticles_sent[NNEIGHBOURS];

  if (!nparticles) {
    printf("Out of particles\n");
    return;
  }

  handle_particles(global_nx, global_ny, nx, ny, pad, x_off, y_off, 1, dt,
                   neighbours, density, edgex, edgey, edgedx, edgedy, facet_events,
                   collision_events, nparticles_sent, master_key, nparticles_total,
                   nparticles, particles, cs_scatter_table, cs_absorb_table, 
                   energy_deposition_tally, nfacets_reduce_array,
                   ncollisions_reduce_array, nprocessed_reduce_array);
}

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny,
    const int pad, const int x_off, const int y_off, const int initial,
    const float dt, const int* neighbours, const float* density,
    const float* edgex, const float* edgey, const float* edgedx,
    const float* edgedy, uint64_t* facets, uint64_t* collisions,
    int* nparticles_sent, uint64_t* master_key, const int nparticles_total,
    const int nparticles_to_process, Particle* particles,
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table,
    float* energy_deposition_tally, uint64_t* nfacets_reduce_array, 
    uint64_t* ncollisions_reduce_array, uint64_t* nprocessed_reduce_array) {

  const int nthreads = NTHREADS;
  const int nblocks = ceil(nparticles_total / (float)nthreads);
  handle_particles_kernel<<<nblocks, nthreads>>>(
      nparticles_total, global_nx, global_ny, nx, ny, pad, x_off, y_off, dt,
      initial, nparticles_total, density, edgex, edgey, edgedx, edgedy,
      energy_deposition_tally, particles->cellx, particles->celly,
      cs_scatter_table->nentries, cs_absorb_table->nentries,
      cs_scatter_table->keys, cs_scatter_table->values, cs_absorb_table->keys,
      cs_absorb_table->values, particles->energy, particles->dt_to_census,
      particles->mfp_to_collision, particles->weight, particles->omega_x,
      particles->omega_y, particles->x, particles->y, (*master_key)++,
      nfacets_reduce_array, ncollisions_reduce_array, nprocessed_reduce_array);

  // Finalise the reduction of the balance tallies
  uint64_t nfacets = 0;
  uint64_t ncollisions = 0;
  uint64_t nprocessed = 0;
  finish_sum_uint64_reduce(nblocks, nfacets_reduce_array, &nfacets);
  finish_sum_uint64_reduce(nblocks, ncollisions_reduce_array, &ncollisions);
  finish_sum_uint64_reduce(nblocks, nprocessed_reduce_array, &nprocessed);

  *facets = nfacets;
  *collisions = ncollisions;

  printf("Particles  %llu\n", nprocessed);
}

// Initialises a new particle ready for tracking
size_t inject_particles(const int nparticles, const int global_nx,
                        const int local_nx, const int local_ny, const int pad,
                        const float local_particle_left_off,
                        const float local_particle_bottom_off,
                        const float local_particle_width,
                        const float local_particle_height, const int x_off,
                        const int y_off, const float dt, const float* edgex,
                        const float* edgey, const float initial_energy,
                        const uint64_t master_key, Particle** particles) {

  // Allocate a Particle structure
  *particles = (Particle*)malloc(sizeof(Particle));
  if (!*particles) {
    TERMINATE("Could not allocate particle array.\n");
  }

  // Allocate all of the Particle data arrays
  Particle* particle = *particles;
  size_t allocation = 0;
  allocation += allocate_float_data(&particle->x, nparticles * 1.5);
  allocation += allocate_float_data(&particle->y, nparticles * 1.5);
  allocation += allocate_float_data(&particle->omega_x, nparticles * 1.5);
  allocation += allocate_float_data(&particle->omega_y, nparticles * 1.5);
  allocation += allocate_float_data(&particle->energy, nparticles * 1.5);
  allocation += allocate_float_data(&particle->weight, nparticles * 1.5);
  allocation += allocate_float_data(&particle->dt_to_census, nparticles * 1.5);
  allocation += allocate_float_data(&particle->mfp_to_collision, nparticles * 1.5);
  allocation += allocate_int_data(&particle->cellx, nparticles * 1.5);
  allocation += allocate_int_data(&particle->celly, nparticles * 1.5);

  // Initialise all of the particle data
  const int nthreads = NTHREADS;
  const int nblocks = ceil(nparticles / (float)nthreads);
  inject_particles_kernel<<<nblocks, nthreads>>>(
      local_nx, local_ny, pad, x_off, y_off, local_particle_left_off,
      local_particle_bottom_off, local_particle_width, local_particle_height,
      nparticles, dt, initial_energy, 0, edgex, edgey, (*particles)->x,
      (*particles)->y, (*particles)->cellx, (*particles)->celly,
      (*particles)->omega_x, (*particles)->omega_y, (*particles)->energy,
      (*particles)->weight, (*particles)->dt_to_census,
      (*particles)->mfp_to_collision);

  return allocation;
}

size_t initialise_float_mesh(NeutralData* neutral_data, Mesh* mesh, SharedData* shared_data) {

  size_t allocation = allocate_float_data(&neutral_data->edgex, mesh->local_nx+1);
  allocation += allocate_float_data(&neutral_data->edgey, mesh->local_ny+1);
  allocation += allocate_float_data(&neutral_data->edgedx, mesh->local_nx+1);
  allocation += allocate_float_data(&neutral_data->edgedy, mesh->local_ny+1);
  allocation += allocate_float_data(&neutral_data->density, mesh->local_nx*mesh->local_ny);

  int nthreads = NTHREADS;
  int nblocks = ceil((mesh->local_nx+1) / (double)nthreads);
  initialise_x_mesh<<<nblocks, nthreads>>>(
      mesh->local_nx+1, neutral_data->edgex, neutral_data->edgedx, mesh->edgex, mesh->edgedx);

  nblocks = ceil((mesh->local_ny+1) / (double)nthreads);
  initialise_y_mesh<<<nblocks, nthreads>>>(
      mesh->local_ny+1, neutral_data->edgey, neutral_data->edgedy, mesh->edgey, mesh->edgedy);

  nblocks = ceil((mesh->local_nx*mesh->local_ny) / (double)nthreads);
  initialise_density<<<nblocks, nthreads>>>(
      mesh->local_nx*mesh->local_ny, neutral_data->density, shared_data->density);

  return allocation;
}

// Sends a particle to a neighbour and replaces in the particle list
void send_and_mark_particle(const int destination, Particle* particle) {
}

// Validates the results of the simulation
void validate(const int nx, const int ny, const char* params_filename,
              const int rank, float* energy_deposition_tally) {

  float* h_energy_deposition_tally;
  allocate_host_float_data(&h_energy_deposition_tally, nx * ny);
  copy_float_buffer(nx * ny, &energy_deposition_tally, &h_energy_deposition_tally,
              RECV);

  // Reduce the energy deposition tally locally
  float local_energy_tally = 0.0;
  for (int ii = 0; ii < nx * ny; ++ii) {
    local_energy_tally += h_energy_deposition_tally[ii];
  }

  // Finalise the reduction globally
  float global_energy_tally = reduce_all_sum(local_energy_tally);

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

  // Check the value is within tolerance
  printf("Expected %.12e, result was %.12e.\n", values[0], global_energy_tally);
  if (within_tolerance(values[0], global_energy_tally, VALIDATE_TOLERANCE)) {
    printf("PASSED validation.\n");
  } else {
    printf("FAILED validation.\n");
  }

  free(keys);
  free(values);
}
