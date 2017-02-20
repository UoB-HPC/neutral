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
    const int ninitial_particles, const double elapsed_sim_time);

int main(int argc, char** argv)
{
  if(argc != 4) {
    TERMINATE("usage: ./bright.exe <nx> <ny> <niters>\n");
  }

#ifdef ENABLE_PROFILING
  /* The timing code has to be called so many times that the API calls 
   * actually begin to influence the performance dramatically. */
  fprintf(stderr, "Warning. Profiling is enabled and will increase the runtime.\n\n");
#endif

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  mesh.global_nx = atoi(argv[1]);
  mesh.global_ny = atoi(argv[2]);
  mesh.local_nx = atoi(argv[1]) + 2*PAD;
  mesh.local_ny = atoi(argv[2]) + 2*PAD;
  mesh.width = get_double_parameter("width", ARCH_ROOT_PARAMS);
  mesh.height = get_double_parameter("height", ARCH_ROOT_PARAMS);
  mesh.dt = get_double_parameter("dt", NEUTRAL_PARAMS);
  mesh.sim_end = get_double_parameter("sim_end", ARCH_ROOT_PARAMS);
  mesh.rank = MASTER;
  mesh.niters = atoi(argv[3]);
  mesh.nranks = 1;

  int nthreads = 0;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_devices(mesh.rank);
  initialise_comms(&mesh);
  initialise_mesh_2d(&mesh);

  SharedData shared_data = {0};
  initialise_shared_data_2d(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, &shared_data);

  RNPool* rn_pool = (RNPool*)malloc(sizeof(RNPool)*nthreads);
#pragma omp parallel
  {
    init_rn_pool(&rn_pool[omp_get_thread_num()], omp_get_thread_num());
  }

  BrightData bright_data = {0};
  initialise_bright_data(
      &bright_data, &mesh, rn_pool);

  // TODO: Currently considering that we have a time dependent system where
  // there are no secondary particles, and reflective boundary conditions.
  // This should mean that the final shared_data of the system has conserved the number
  // of particles from initialisation, minus the particles that have disappeared
  // due to leakage and absorption
  //
  // Modelling two types of collision, scattering and absorption, but not 
  // modelling fission in this particular simulation

  // Presumably the timestep will have been set by the fluid dynamics,
  // given that it has the tightest timestep control requirements
  //set_timestep();

  // Make sure initialisation phase is complete
  barrier();

  const int ninitial_particles = bright_data.nparticles;

  // Main timestep loop where we will track each particle through time
  int tt;
  double wallclock = 0.0;
  double elapsed_sim_time = 0.0;
  for(tt = 1; tt <= mesh.niters; ++tt) {
    if(mesh.rank == MASTER) {
      printf("Iteration %d\n", tt);
    }

    plot_particle_density(
        &bright_data, &mesh, tt, ninitial_particles, elapsed_sim_time);

    double w0 = omp_get_wtime();

    // At this stage all of the particles have been moved for their timestep
    // and the tallying of the energy deposition throughout the system must
    // have been resolved.
    solve_transport_2d(
        mesh.local_nx-2*PAD, mesh.local_ny-2*PAD, 
        mesh.global_nx, mesh.global_ny, 
        mesh.x_off, mesh.y_off, mesh.dt, 
        bright_data.nparticles, &bright_data.nlocal_particles, 
        mesh.neighbours, 
        bright_data.local_particles, 
        shared_data.rho, 
        mesh.edgex, mesh.edgey, 
        mesh.edgedx, mesh.edgedy, 
        bright_data.out_particles, 
        bright_data.cs_scatter_table, bright_data.cs_absorb_table, 
        bright_data.scalar_flux_tally, bright_data.energy_deposition_tally,
        rn_pool);

    barrier();

    wallclock += omp_get_wtime()-w0;
    elapsed_sim_time += mesh.dt;

    char tally_name[100];
    sprintf(tally_name, "energy%d", tt);
    int dneighbours[NNEIGHBOURS] = { EDGE, EDGE,  EDGE,  EDGE,  EDGE,  EDGE }; 
    write_all_ranks_to_visit(
        mesh.global_nx, mesh.global_ny, 
        mesh.local_nx-2*PAD, mesh.local_ny-2*PAD,
        mesh.x_off, mesh.y_off, 
        mesh.rank, mesh.nranks, dneighbours, 
        bright_data.energy_deposition_tally, tally_name, 0, elapsed_sim_time);

    // Leave the simulation if we have reached the simulation end time
    if(elapsed_sim_time >= mesh.sim_end) {
      if(mesh.rank == MASTER)
        printf("reached end of simulation time\n");
      break;
    }
  }

  plot_particle_density(
      &bright_data, &mesh, tt, ninitial_particles, elapsed_sim_time);

  // TODO: WHAT SHOULD THE VALUE OF NINITIALPARTICLES BE IF FISSION ETC.
  validate(
      mesh.local_nx-2*PAD, mesh.local_ny-2*PAD, ninitial_particles, mesh.dt, 
      mesh.niters, mesh.rank, bright_data.energy_deposition_tally);

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);

    printf("Wallclock %.2fs, Elapsed Simulation Time %.6fs\n", 
        wallclock, elapsed_sim_time);
  }

  return 0;
}

// This is a bit hacky and temporary for now
void plot_particle_density(
    BrightData* bright_data, Mesh* mesh, const int tt, 
    const int ninitial_particles, const double elapsed_sim_time)
{
  double* temp = (double*)malloc(sizeof(double)*mesh->local_nx*mesh->local_ny);
  for(int ii = 0; ii < ninitial_particles; ++ii) {
    Particle* particle = &bright_data->local_particles[ii];
    const int cellx = (particle->cell%mesh->global_nx)-mesh->x_off;
    const int celly = (particle->cell/mesh->global_nx)-mesh->y_off;
    temp[celly*(mesh->local_nx-2*PAD)+cellx] += 1.0;
  }

  // Dummy neighbours that stops any padding from happening
  int neighbours[NNEIGHBOURS] = { EDGE, EDGE,  EDGE,  EDGE,  EDGE,  EDGE }; 
  char particles_name[100];
  sprintf(particles_name, "particles%d", tt);
  write_all_ranks_to_visit(
      mesh->global_nx, mesh->global_ny, mesh->local_nx-2*PAD, mesh->local_ny-2*PAD, 
      mesh->x_off, mesh->y_off, mesh->rank, mesh->nranks, neighbours, 
      temp, particles_name, 0, elapsed_sim_time);
  free(temp);
}

