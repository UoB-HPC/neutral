#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include "bright_interface.h"
#include "../mesh.h"
#include "../shared_data.h"
#include "../comms.h"
#include "../profiler.h"
#include "mt19937.h"




#ifdef MPI
#include "mpi.h"
#endif




int main(int argc, char** argv)
{
  if(argc != 4) {
    printf("usage: ./bright.exe <nx> <ny> <niters>\n");
    exit(1);
  }

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  mesh.global_nx = atoi(argv[1]);
  mesh.global_ny = atoi(argv[2]);
  mesh.local_nx = atoi(argv[1]) + 2*PAD;
  mesh.local_ny = atoi(argv[2]) + 2*PAD;
  mesh.width = WIDTH;
  mesh.height = HEIGHT;
  mesh.dt = MAX_DT;
  mesh.rank = MASTER;
  mesh.niters = atoi(argv[3]);
  mesh.nranks = 1;

  initialise_mpi(argc, argv, &mesh.rank, &mesh.nranks);
  initialise_devices(mesh.rank);
  initialise_comms(&mesh);
  initialise_mesh_2d(&mesh);

  mesh.dt *= 0.01;

  SharedData shared_data = {0};
  initialise_shared_data_2d(
      mesh.global_nx, mesh.global_ny, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, &shared_data);

  BrightData bright_data = {0};
  bright_data.nparticles = NPARTICLES;

  initialise_bright_data(
      &bright_data, &mesh);

  // Seed the mersenne twister
  sgenrand(100UL);

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

  // Prepare for solve
  struct Profile wallclock = {0};
  double elapsed_sim_time = 0.0;

  double* temp = (double*)malloc(sizeof(double)*mesh.local_nx*mesh.local_ny);
  for(int ii = 0; ii < bright_data.nlocal_particles; ++ii) {
    Particle* particle = &bright_data.local_particles[ii];
    const int cellx = (particle->cell%mesh.global_nx)-mesh.x_off+PAD;
    const int celly = (particle->cell/mesh.global_nx)-mesh.y_off+PAD;
    temp[celly*mesh.local_nx+cellx] = particle->e;
  }

  write_all_ranks_to_visit(
      mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, 
      mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
      temp, "particles0", 0, elapsed_sim_time);

  // Main timestep loop where we will track each particle through time
  int tt;
  for(tt = 0; tt < mesh.niters; ++tt) {
    if(mesh.rank == MASTER) 
      printf("Iteration %d\n", tt+1);

    START_PROFILING(&wallclock);

    // At this stage all of the particles have been moved for their timestep
    // and the tallying of the energy deposition throughout the system must
    // have been resolved.
    solve_transport_2d(
        mesh.local_nx-2*PAD, mesh.local_ny-2*PAD, mesh.global_nx, mesh.global_ny, 
        mesh.x_off, mesh.y_off, mesh.dt, &bright_data.nlocal_particles, 
        mesh.neighbours, bright_data.local_particles, shared_data.rho, 
        mesh.edgex, mesh.edgey, bright_data.out_particles, 
        bright_data.cs_scatter_table, bright_data.cs_absorb_table, 
        bright_data.energy_tally);

    elapsed_sim_time += mesh.dt;
    if(elapsed_sim_time >= SIM_END) {
      if(mesh.rank == MASTER)
        printf("reached end of simulation time\n");
      break;
    }

    STOP_PROFILING(&wallclock, "wallclock");

    barrier();

    temp = (double*)malloc(sizeof(double)*mesh.local_nx*mesh.local_ny);
    for(int ii = 0; ii < bright_data.nlocal_particles; ++ii) {
      Particle* particle = &bright_data.local_particles[ii];
      const int cellx = (particle->cell%mesh.global_nx)-mesh.x_off+PAD;
      const int celly = (particle->cell/mesh.global_nx)-mesh.y_off+PAD;
      temp[celly*mesh.local_nx+cellx] = particle->e;
    }

    char particles_name[100];
    char tally_name[100];
    sprintf(particles_name, "particles%d", tt+1);
    sprintf(tally_name, "energy%d", tt+1);
    write_all_ranks_to_visit(
        mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, 
        mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
        temp, particles_name, 0, elapsed_sim_time);
    write_all_ranks_to_visit(
        mesh.global_nx, mesh.global_ny, mesh.local_nx-2*PAD, mesh.local_ny-2*PAD,
        mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
        bright_data.energy_tally, tally_name, 0, elapsed_sim_time);
  }

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&compute_profile);

    struct ProfileEntry pe = profiler_get_profile_entry(&wallclock, "wallclock");
    printf("Wallclock %.2fs, Elapsed Simulation Time %.4fs\n", 
        pe.time, elapsed_sim_time);
  }

  return 0;
}

