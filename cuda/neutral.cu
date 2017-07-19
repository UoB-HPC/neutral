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

#if 0
#ifdef MPI
#include "mpi.h"
#endif
#endif // if 0

// Performs a solve of dependent variables for particle transport.
void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const int pad, const int x_off, const int y_off, const double dt,
    const int nparticles_total, int* nlocal_particles, uint64_t* master_key,
    const int* neighbours, Particle* particles, const double* density,
    const double* edgex, const double* edgey, const double* edgedx,
    const double* edgedy, CrossSection* cs_scatter_table,
    CrossSection* cs_absorb_table, double* energy_deposition_tally,
    int* reduce_array0, int* reduce_array1) {
  // Initial idea is to use a kind of queue for handling the particles.
  // Presumably
  // this doesn't have to be a carefully ordered queue but lets see how that
  // goes.

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
                   &collisions, nparticles_sent, master_key, nparticles_total,
                   nparticles, &nparticles, particles, cs_scatter_table,
                   cs_absorb_table, energy_deposition_tally, reduce_array0,
                   reduce_array1);

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
          nparticles_sent, nparticles_total, nunprocessed_particles, &nparticles, 
          &particles[unprocessed_start], particles_out, cs_scatter_table, 
          cs_absorb_table, energy_deposition_tally, rn_pools);
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

  printf("facets %llu collisions %llu\n", facets, collisions);
}

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny,
    const int pad, const int x_off, const int y_off, const int initial,
    const double dt, const int* neighbours, const double* density,
    const double* edgex, const double* edgey, const double* edgedx,
    const double* edgedy, uint64_t* facets, uint64_t* collisions,
    int* nparticles_sent, uint64_t* master_key, const int nparticles_total,
    const int nparticles_to_process, int* nparticles, Particle* particles,
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table,
    double* energy_deposition_tally, int* reduce_array0, int* reduce_array1) {
  int nparticles_deleted = 0;

  const int nthreads = NTHREADS;
  const int nblocks = ceil(nparticles_total / (double)NTHREADS);
  handle_particles_kernel<<<nblocks, nthreads>>>(
      nparticles_total, global_nx, global_ny, nx, ny, pad, x_off, y_off, dt,
      initial, nparticles_total, density, edgex, edgey, edgedx, edgedy,
      energy_deposition_tally, particles->cellx, particles->celly,
      cs_scatter_table->nentries, cs_absorb_table->nentries,
      cs_scatter_table->keys, cs_scatter_table->values, cs_absorb_table->keys,
      cs_absorb_table->values, particles->energy, particles->dt_to_census,
      particles->mfp_to_collision, particles->weight, particles->omega_x,
      particles->omega_y, particles->x, particles->y, (*master_key)++,
      reduce_array0, reduce_array1);

  int nfacets = 0;
  int ncollisions = 0;
  finish_sum_int_reduce(nblocks, reduce_array0, &nfacets);
  finish_sum_int_reduce(nblocks, reduce_array1, &ncollisions);
  *facets = nfacets;
  *collisions = ncollisions;

  printf("handled %d particles, with %d particles deleted\n",
         nparticles_to_process, nparticles_deleted);
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

  const int nthreads = NTHREADS;
  const int nblocks = ceil(nparticles / (double)NTHREADS);
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

// Sends a particle to a neighbour and replaces in the particle list
void send_and_mark_particle(const int destination, Particle* particle) {
#if 0
#ifdef MPI
  if(destination == EDGE) {
    return;
  }

  particle->dead = 1;

  // Send the particle
  MPI_Send(
      particle, 1, particle_type, destination, TAG_PARTICLE, MPI_COMM_WORLD);
#else
  TERMINATE("Unreachable - shouldn't send particles unless MPI enabled.\n");
#endif
#endif // if 0
}

// Validates the results of the simulation
void validate(const int nx, const int ny, const char* params_filename,
              const int rank, double* energy_deposition_tally) {
  double* h_energy_deposition_tally;
  allocate_host_data(&h_energy_deposition_tally, nx * ny);
  copy_buffer(nx * ny, &energy_deposition_tally, &h_energy_deposition_tally,
              RECV);

  double local_energy_tally = 0.0;
  for (int ii = 0; ii < nx * ny; ++ii) {
    local_energy_tally += h_energy_deposition_tally[ii];
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
