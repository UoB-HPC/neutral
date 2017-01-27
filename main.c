#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>

#include "main.h"
#include "profiler.h"

#define ind0 (ii*nx + jj)
#define ind1 (ii*(nx+1) + jj)

int main(int argc, char** argv)
{
  if(argc != 4) {
    printf("usage: ./bright.exe <local_nx> <local_ny> <niters>\n");
    exit(1);
  }

  // Store the dimensions of the mesh
  Mesh mesh = {0};
  mesh.global_nx = atoi(argv[1]);
  mesh.global_ny = atoi(argv[2]);
  mesh.local_nx = atoi(argv[1]) + 2*PAD;
  mesh.local_ny = atoi(argv[2]) + 2*PAD;
  mesh.niters = atoi(argv[3]);
  mesh.rank = MASTER;
  mesh.nranks = 1;
  initialise_mesh(&mesh);

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
  // This should mean that the final state of the system has conserved the number
  // of particles from initialisation, minus the particles that have disappeared
  // due to leakage and absorption
  //
  // Modelling two types of collision, scattering and absorption, but not 
  // modelling fission in this particular simulation as it does not match the
  // behaviour seen in SNAP

  // Field initialisation stage
  // Add a predefined numbar of particles into the system at random
  for(int ii = 0; ii < nparticles; ++ii) {
    initialise_particle(&particles[ii], &mesh);
  }

  // Presumably the timestep will have been set by the fluid dynamics,
  // given that it has the tightest timestep control requirements
  //set_timestep();

  // Prepare for solve
  struct Profile p = {0};
  struct Profile wallclock = {0};
  double elapsed_sim_time = 0.0;

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
    for(int pp = 0; pp < nparticles; ++pp) {
      // TODO: handle the different particle events
      //
      // (1) particle can stream and reach census
      // (2) particle can collide and either
      //      - the particle will be absorbed
      //      - the particle will scatter (this presumably means the energy changes)
      // (3) particle hits a boundary region and needs transferring to another process

      // Check the timestep required to move the particle aint a single axis
      // If the velocity is positive then the top or right boundary will be hit
      double u_x_inv = 1.0/particles[pp].u_x;
      double u_y_inv = 1.0/particles[pp].u_y;
      double dt_a = (particles[pp].u_x > 0.0) 
        ? (mesh.edgex[particles[pp].cellx+1]-particles[pp].x)*u_x_inv
        : (mesh.edgex[particles[pp].cellx]-particles[pp].x)*u_x_inv;
      double dt_b = (particles[pp].u_y > 0.0) 
        ? (mesh.edgey[particles[pp].celly+1]-particles[pp].y)*u_y_inv
        : (mesh.edgey[particles[pp].celly]-particles[pp].y)*u_y_inv;

      printf("dt_a %.12e dt_b %.12e\n", dt_a, dt_b);

      double mag_u0 = 
        sqrt(particles[pp].u_x*particles[pp].u_x+particles[pp].u_y*particles[pp].u_y);

      // Calculated the projection to be 
      // a = vector on first edge to be hit
      // u = velocity vector
      double mag_u1;
      if(dt_a < dt_b) {
        // cos(theta) = ||(x, 0)||/||(u_x', u_y')|| - u' is u at boundary
        // cos(theta) = (x.u)/(||x||.||u||)
        // x_x/||u'|| = (x_x, 0)*(u_x, u_y) / (x_x.||u||)
        // x_x/||u'|| = (x_x.u_x / x_x.||u||)
        // x_x/||u'|| = u_x/||u||
        // ||u'|| = (x_x.||u||)/u_x
        // We are centered on the origin, so the y component is 0 after travelling
        // aint the x axis to the edge (ax, 0).(x, y)
        mag_u1 = (particles[pp].u_x > 0.0) 
          ? (mesh.edgex[particles[pp].cellx+1]-particles[pp].x)*mag_u0*u_x_inv
          : (mesh.edgex[particles[pp].cellx]-particles[pp].x)*mag_u0*u_x_inv;
      }
      else {
        // We are centered on the origin, so the x component is 0 after travelling
        // aint the y axis to the edge (0, ay).(x, y)
        mag_u1 = (particles[pp].u_y > 0.0) 
          ? (mesh.edgey[particles[pp].celly+1]-particles[pp].y)*mag_u0*u_y_inv
          : (mesh.edgey[particles[pp].celly]-particles[pp].y)*mag_u0*u_y_inv;
      }

      // Scale the velocity by the scaling factor
      double dx = (particles[pp].u_x)*(mag_u1/mag_u0);
      double dy = (particles[pp].u_y)*(mag_u1/mag_u0);

      printf("u_x %.12e u_y %.12e\n", particles[pp].u_x, particles[pp].u_y);
      printf("x0 %.12e y0 %.12e\n", particles[pp].x, particles[pp].y);
      printf("x1 %.12e y1 %.12e\n", particles[pp].x+dx, particles[pp].y+dy);
    }

    char particles_name[100];
    char tally_name[100];
    sprintf(particles_name, "particles%d", tt+1);
    sprintf(tally_name, "energy%d", tt+1);
    write_all_ranks_to_visit(
        mesh.global_nx+2*PAD, mesh.global_ny+2*PAD, mesh.local_nx, mesh.local_ny, 
        mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, mesh.neighbours, 
        temp, particles_name, 0, elapsed_sim_time);
    int dneighbours[NNEIGHBOURS] = { EDGE, EDGE,  EDGE,  EDGE,  EDGE,  EDGE }; // Fudges away padding
    write_all_ranks_to_visit(
        mesh.global_nx, mesh.global_ny, mesh.local_nx-2*PAD, mesh.local_ny-2*PAD,
        mesh.x_off, mesh.y_off, mesh.rank, mesh.nranks, dneighbours, 
        bright_data.energy_tally, tally_name, 0, elapsed_sim_time);
  }

  if(mesh.rank == MASTER) {
    PRINT_PROFILING_RESULTS(&p);

    struct ProfileEntry pe = profiler_get_profile_entry(&wallclock, "wallclock");
    printf("Wallclock %.2fs, Elapsed Simulation Time %.4fs\n", 
        pe.time, elapsed_sim_time);
  }

#if 0
  char visit_name[256];
  sprintf(visit_name, "density_%d", mesh.rank);
  write_to_visit(mesh.local_nx, mesh.local_ny, state.rho, "wet_density", tt, elapsed_sim_time);
#endif // if 0

  return 0;
}

// Initialises a new particle ready for tracking
static inline void initialise_particle(Particle* particle, Mesh* mesh)
{
  // Set the initial random location of the particle
  particle->x = (double)prng(particle->seed)*mesh->discrete_step_x;
  particle->y = (double)prng(particle->seed)*mesh->discrete_step_y;

  for(int ii = 0; ii < mesh->local_nx; ++ii) {
    if(particle->x >= mesh->edgex[ii] && particle->x < mesh->edgex[ii+1]) {
      particle->cellx = ii;
      break;
    }
  }
  for(int ii = 0; ii < mesh->local_ny; ++ii) {
    if(particle->y >= mesh->edgey[ii] && particle->y < mesh->edgey[ii+1]) {
      particle->celly = ii;
      break;
    }
  }
  printf("cellx %d celly %d\n", particle->cellx, particle->celly);

  // TODO: Work out how the velocities can be calculated
  // It feels like it might be possible to look at the mean free path of a particle
  // Also, it is probable that a particle starting at a particular energy level will
  // have a specific velocity to begin with, perhaps this can then be weighted 
  // randomly in a particular direction
  // For testing purposes just going to set this to some simple fixed value
  // for the modulo
  const double upper_bound_vel = 1.0;
  const double discrete_step_u_x = (upper_bound_vel / (double)INT_MAX);
  const double discrete_step_u_y = (upper_bound_vel / (double)INT_MAX);
  particle->u_x = (double)prng(particle->seed)*discrete_step_u_x;
  particle->u_y = (double)prng(particle->seed)*discrete_step_u_y;

  // TODO: Again, how does the particle energy get initialised
  // Presumably neutrons can start in any continuous energy, but it's not
  // clear how we decide this at initialisation. Going to set an upper and lower
  // bound for the short term, the unit seemed to be meV...
  const double lower_bound_e = 1.0;
  const double upper_bound_e = 10.0;
  const double discrete_step_e = (upper_bound_e-lower_bound_e)/(double)INT_MAX;
  particle->e = lower_bound_e + (double)prng(particle->seed)*discrete_step_e;

  // TODO: What are we meant to initialise the number of physical particles
  // per simulated particle, 10 is an arbitrary default
  particle->nparticles = 10;

  // Important to store aint with the particle in case the particle gets sent across
  // boundaries, when it will need to continue on another process
  // TODO: It seems logical to initialise this to the current timestep, which 
  // will be coupled with the timestep provided by the fluid dynamics solver
  particle->dt_till_census = mesh->dt;

  // Initilise the most recent even to be a census event, which counts as the 
  // default starting state for a particle
  particle->most_recent_event = CENSUS;
}

// Initialises the mesh for particle transportation
// TODO: Work out if this can be complementarily stored with the other applications
// data for meshes etc.
static inline void initialise_mesh(Mesh* mesh) 
{
  mesh->width = 10.0;
  mesh->height = 10.0;
  mesh->local_nx = 100; 
  mesh->local_ny = 100; 
  mesh->x_off = 0;
  mesh->y_off = 0;

  const double cell_dx = mesh->width / (double)mesh->local_nx;
  const double cell_dy = mesh->height / (double)mesh->local_ny;

  mesh->edgex = (double*)malloc(sizeof(double)*(mesh->local_nx+1));
  mesh->edgey = (double*)malloc(sizeof(double)*(mesh->local_ny+1));
  mesh->cellx = (double*)malloc(sizeof(double)*(mesh->local_nx+1));
  mesh->celly = (double*)malloc(sizeof(double)*(mesh->local_ny+1));


  // Calculate the mesh positions, currently equal, but should be capable of 
  // having differing values, eventually with grid motion
  for(int ii = 0; ii < mesh->local_nx+1; ++ii) {
    mesh->edgex[ii] = (mesh->x_off+ii)*cell_dx;
    mesh->cellx[ii] = (mesh->x_off+ii+0.5)*cell_dx;
  }
  for(int ii = 0; ii < mesh->local_ny+1; ++ii) {
    mesh->edgey[ii] = (mesh->y_off+ii)*cell_dy;
    mesh->celly[ii] = (mesh->y_off+ii+0.5)*cell_dy;
  }

  // Calculate the discrete steps that the particles can be initialised onto
  // TODO: Check whether this is an acceptable idea, given that it potentially
  // limits the random-ness of the positioning of the particles at initialisation
  mesh->discrete_step_x = (mesh->width / (double)INT_MAX);
  mesh->discrete_step_y = (mesh->height / (double)INT_MAX);
  printf("dsx %.12e dsy %.12e\n", mesh->discrete_step_x, mesh->discrete_step_y);
}

static inline void initialise_cross_section()
{

}

static inline int prng(int seed)
{
  // Get a random number for a particular coin flip
  return rand();
}

