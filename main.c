#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "bright_interface.h"
#include "../mesh.h"
#include "../shared_data.h"
#include "../comms.h"
#include "../profiler.h"
#include "../params.h"

#ifdef MPI
#include "mpi.h"
#endif

void plot_particle_density(
    BrightData* bright_data, Mesh* mesh, const int tt, 
    const int nparticles, const double elapsed_sim_time);

int main(int argc, char** argv)
{
  if(argc != 2) {
    TERMINATE("usage: ./bright.exe <param_file>\n");
  }

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  BrightData bright_data = {0};
  bright_data.neutral_params_filename = argv[1];
  mesh.global_nx = get_int_parameter("nx", bright_data.neutral_params_filename);
  mesh.global_ny = get_int_parameter("ny", bright_data.neutral_params_filename);
  mesh.local_nx = mesh.global_nx + 2*PAD;
  mesh.local_ny = mesh.global_ny + 2*PAD;
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.dt = get_double_parameter("dt", bright_data.neutral_params_filename);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.niters = get_int_parameter("iterations", bright_data.neutral_params_filename);
  mesh.rank = MASTER;
  mesh.nranks = 1;
  mesh.ndims = 2;

  // Get the number of threads and initialise the random number pool
#pragma omp parallel
  {
    bright_data.nthreads = omp_get_num_threads();
  }

  printf("Starting up with %d OpenMP threads.\n", bright_data.nthreads);
  printf("Loading problem from %s.\n", bright_data.neutral_params_filename);
#ifdef ENABLE_PROFILING
  /* The timing code has to be called so many times that the API calls 
   * actually begin to influence the performance dramatically. */
  fprintf(stderr, 
      "Warning. Profiling is enabled and will increase the runtime.\n\n");
#endif

  // Perform the general initialisation steps for the mesh etc
  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_devices(mesh.rank);
  initialise_comms(&mesh);
  initialise_mesh_2d(&mesh);
  SharedData shared_data = {0};
  initialise_shared_data_2d(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, mesh.width, mesh.height, 
      bright_data.neutral_params_filename, mesh.edgex, mesh.edgey, &shared_data);
  initialise_bright_data(
      &bright_data, &mesh);

  // Make sure initialisation phase is complete
  barrier();

  // Main timestep loop where we will track each particle through time
  int tt;
  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;
  uint64_t master_key = 0;
  for(tt = 1; tt <= mesh.niters; ++tt) {

    if(mesh.rank == MASTER) {
      printf("\nIteration %d\n", tt);
    }

#if 0
    plot_particle_density(
        &bright_data, &mesh, tt, nparticles, elapsed_sim_time);
#endif // if 0

    double w0 = omp_get_wtime();

    // Begin the main solve step
    solve_transport_2d(
        mesh.local_nx-2*PAD, mesh.local_ny-2*PAD, mesh.global_nx, mesh.global_ny, 
        mesh.x_off, mesh.y_off, mesh.dt, bright_data.nparticles, 
        &bright_data.nlocal_particles, &master_key, mesh.neighbours, 
        bright_data.local_particles, shared_data.rho, mesh.edgex, mesh.edgey, 
        mesh.edgedx, mesh.edgedy, bright_data.cs_scatter_table, 
        bright_data.cs_absorb_table, bright_data.scalar_flux_tally, 
        bright_data.energy_deposition_tally, bright_data.rn_pools,
        bright_data.reduce_array0, bright_data.reduce_array1);

    barrier();

    wallclock += omp_get_wtime()-w0;
    elapsed_sim_time += mesh.dt;

#if 0
    char tally_name[100];
    sprintf(tally_name, "energy%d", tt);
    int dneighbours[NNEIGHBOURS] = { EDGE, EDGE,  EDGE,  EDGE,  EDGE,  EDGE };
    write_all_ranks_to_visit(
        mesh.global_nx, mesh.global_ny, mesh.local_nx-2*PAD, mesh.local_ny-2*PAD,
        mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, dneighbours, 
        bright_data.energy_deposition_tally, tally_name, 0, elapsed_sim_time);
#endif // if 0

    // Leave the simulation if we have reached the simulation end time
    if(elapsed_sim_time >= mesh.sim_end) {
      if(mesh.rank == MASTER)
        printf("reached end of simulation time\n");
      break;
    }
  }

#if 0
  plot_particle_density(
      &bright_data, &mesh, tt, nparticles, elapsed_sim_time);
#endif // if 0


#if 0
  // TODO: WHAT SHOULD THE VALUE OF NINITIALPARTICLES BE IF FISSION ETC.
  validate(
      mesh.local_nx, mesh.local_ny, bright_data.neutral_params_filename, 
     mesh.rank, bright_data.energy_deposition_tally);
#endif // if 0

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);

    printf("Wallclock %.9fs, Elapsed Simulation Time %.6fs\n", 
        wallclock, elapsed_sim_time);
  }

  return 0;
}

// This is a bit hacky and temporary for now
void plot_particle_density(
    BrightData* bright_data, Mesh* mesh, const int tt, 
    const int nparticles, const double elapsed_sim_time)
{
  double* temp = (double*)malloc(sizeof(double)*mesh->local_nx*mesh->local_ny);
  if(!temp) {
    TERMINATE("Could not allocate data for printing.\n");
  }

  for(int ii = 0; ii < nparticles; ++ii) {
    Particles* particles = bright_data->local_particles;
    const int cellx = particles->cellx[ii]-mesh->x_off;
    const int celly = particles->celly[ii]-mesh->y_off;
    temp[celly*(mesh->local_nx)+cellx] += 1.0;
  }

  // Dummy neighbours that stops any padding from happening
  char particles_name[100];
  sprintf(particles_name, "particles%d", tt);
  write_all_ranks_to_visit(
      mesh->global_nx, mesh->global_ny, mesh->local_nx, mesh->local_ny, 
      mesh->x_off, mesh->y_off, mesh->rank, mesh->nranks, mesh->neighbours, temp, 
      particles_name, 0, elapsed_sim_time);
  free(temp);
}

