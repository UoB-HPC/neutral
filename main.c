#include <string.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "neutral_interface.h"
#include "../mesh.h"
#include "../shared_data.h"
#include "../comms.h"
#include "../profiler.h"
#include "../params.h"

#ifdef MPI
#include "mpi.h"
#endif

void plot_particle_density(
    NeutralData* neutral_data, Mesh* mesh, const int tt, 
    const int nparticles, const double elapsed_sim_time);

int main(int argc, char** argv)
{
  if(argc != 2) {
    TERMINATE("usage: ./neutral.exe <param_file>\n");
  }

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  NeutralData neutral_data = {0};
  neutral_data.neutral_params_filename = argv[1];
  mesh.global_nx = get_int_parameter("nx", neutral_data.neutral_params_filename);
  mesh.global_ny = get_int_parameter("ny", neutral_data.neutral_params_filename);
  mesh.local_nx = mesh.global_nx + 2*PAD;
  mesh.local_ny = mesh.global_ny + 2*PAD;
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.dt = get_double_parameter("dt", neutral_data.neutral_params_filename);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.niters = get_int_parameter("iterations", neutral_data.neutral_params_filename);
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.ndims = 2;

  // Get the number of threads and initialise the random number pool
#pragma omp parallel
  {
    neutral_data.nthreads = omp_get_num_threads();
  }

  printf("Starting up with %d OpenMP threads.\n", neutral_data.nthreads);
  printf("Loading problem from %s.\n", neutral_data.neutral_params_filename);
#ifdef ENABLE_PROFILING
  /* The timing code has to be called so many times that the API calls 
   * actually begin to influence the performance dramatically. */
  fprintf(stderr, 
      "Warning. Profiling is enabled and will increase the runtime.\n\n");
#endif

  // Initialise enough pools for every thread and a master pool
  RNPool* rn_pools = (RNPool*)malloc(sizeof(RNPool)*(neutral_data.nthreads+1));

  // Initialise the master rn pool
  init_rn_pool(&rn_pools[neutral_data.nthreads], 0xfffff);

  // Perform the general initialisation steps for the mesh etc
  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_devices(mesh.rank);
  initialise_comms(&mesh);
  initialise_mesh_2d(&mesh);
  SharedData shared_data = {0};
  initialise_shared_data_2d(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, mesh.width, mesh.height, 
      neutral_data.neutral_params_filename, mesh.edgex, mesh.edgey, &shared_data);
  initialise_neutral_data(
      &neutral_data, &mesh, &rn_pools[neutral_data.nthreads]);

  // Make sure initialisation phase is complete
  barrier();

  // Main timestep loop where we will track each particle through time
  int tt;
  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;
  uint64_t master_key = 1;
  for(tt = 1; tt <= mesh.niters; ++tt) {

    if(mesh.rank == MASTER) {
      printf("\nIteration %d\n", tt);
    }

#ifdef VISIT_DUMP
    plot_particle_density(
        &neutral_data, &mesh, tt, nparticles, elapsed_sim_time);
#endif

    double w0 = omp_get_wtime();

    // Begin the main solve step
    solve_transport_2d(
        mesh.local_nx-2*PAD, mesh.local_ny-2*PAD, mesh.global_nx, mesh.global_ny, 
        mesh.x_off, mesh.y_off, mesh.dt, neutral_data.nparticles, 
        &neutral_data.nlocal_particles, &master_key, mesh.neighbours, 
        neutral_data.local_particles, shared_data.rho, mesh.edgex, mesh.edgey, 
        mesh.edgedx, mesh.edgedy, neutral_data.cs_scatter_table, 
        neutral_data.cs_absorb_table, neutral_data.scalar_flux_tally, 
        neutral_data.energy_deposition_tally, rn_pools, neutral_data.reduce_array0,
        neutral_data.reduce_array1);

    barrier();

    wallclock += omp_get_wtime()-w0;
    elapsed_sim_time += mesh.dt;

#ifdef VISIT_DUMP
    char tally_name[100];
    sprintf(tally_name, "energy%d", tt);
    int dneighbours[NNEIGHBOURS] = { EDGE, EDGE,  EDGE,  EDGE,  EDGE,  EDGE }; 
    write_all_ranks_to_visit(
        mesh.global_nx, mesh.global_ny, mesh.local_nx-2*PAD, mesh.local_ny-2*PAD,
        mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, dneighbours, 
        neutral_data.energy_deposition_tally, tally_name, 0, elapsed_sim_time);
#endif

    // Leave the simulation if we have reached the simulation end time
    if(elapsed_sim_time >= mesh.sim_end) {
      if(mesh.rank == MASTER)
        printf("reached end of simulation time\n");
      break;
    }
  }

#ifdef VISIT_DUMP
  plot_particle_density(
      &neutral_data, &mesh, tt, nparticles, elapsed_sim_time);
#endif

  validate(
      mesh.local_nx-2*PAD, mesh.local_ny-2*PAD, 
      neutral_data.neutral_params_filename, mesh.rank, 
      neutral_data.energy_deposition_tally);

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&p);

    printf("Wallclock %.9fs, Elapsed Simulation Time %.6fs\n", 
        wallclock, elapsed_sim_time);
  }

#if 0
  char visit_name[256];
  sprintf(visit_name, "density_%d", mesh.rank);
  write_to_visit(mesh.local_nx, mesh.local_ny, state.rho, "wet_density", tt, elapsed_sim_time);
#endif // if 0

  return 0;
}

// This is a bit hacky and temporary for now
void plot_particle_density(
    NeutralData* neutral_data, Mesh* mesh, const int tt, 
    const int nparticles, const double elapsed_sim_time)
{
  double* temp = (double*)malloc(sizeof(double)*mesh->local_nx*mesh->local_ny);
  if(!temp) {
    TERMINATE("Could not allocate data for printing.\n");
  }

  for(int ii = 0; ii < nparticles; ++ii) {
    Particle* particle = &neutral_data->local_particles[ii];
#ifdef SoA
    const int cellx = particle->cellx[ii]-mesh->x_off;
    const int celly = particle->celly[ii]-mesh->y_off;
#else 
    const int cellx = particle->cellx-mesh->x_off;
    const int celly = particle->celly-mesh->y_off;
#endif
    temp[celly*(mesh->local_nx-2*PAD)+cellx] += 1.0;
  }

  // Dummy neighbours that stops any padding from happening
  int neighbours[NNEIGHBOURS] = { EDGE, EDGE,  EDGE,  EDGE,  EDGE,  EDGE }; 
  char particles_name[100];
  sprintf(particles_name, "particles%d", tt);
  write_all_ranks_to_visit(
      mesh->global_nx, mesh->global_ny, mesh->local_nx-2*PAD, 
      mesh->local_ny-2*PAD, mesh->x_off, mesh->y_off, mesh->rank, 
      mesh->nranks, neighbours, temp, particles_name, 0, elapsed_sim_time);
  free(temp);
}

