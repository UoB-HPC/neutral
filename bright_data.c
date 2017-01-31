#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "bright_data.h"
#include "../profiler.h"
#include "../shared.h"
#include "mt19937.h"

// Reads a cross section file
void read_cs_file(
    const char* filename, CrossSection* cs);

// Initialises the set of cross sections
void initialise_cross_sections(
    BrightData* bright_data);

// Initialises a new particle ready for tracking
void initialise_particle(
    const int global_nx, const int global_ny, const int local_nx, 
    const int local_ny, const double mesh_width, const double mesh_height, 
    const double particle_off_x, const double particle_off_y, 
    const int local_particle_nx, const int local_particle_ny, 
    const int x_off, const int y_off, const double dt, 
    const double* edgex, const double* edgey, Particle* particle);

// Initialises all of the Bright-specific data structures.
void initialise_bright_data(
    BrightData* bright_data, Mesh* mesh)
{
  const int local_nx = mesh->local_nx-2*PAD;
  const int local_ny = mesh->local_ny-2*PAD;

  // SMALL 1/5 SQUARE AT LEFT OF MESH
  const int global_particle_start_x = 0;
  const int global_particle_start_y = (2*mesh->global_ny/10);
  const int global_particle_nx = (2*mesh->global_nx/10);
  const int global_particle_ny = (2*mesh->global_ny/10);
#if 0
  // SMALL 1/5 SQUARE IN MIDDLE OF MESH
  const int global_particle_start_x = (4*mesh->global_nx/10);
  const int global_particle_start_y = (4*mesh->global_ny/10);
  const int global_particle_nx = (2*mesh->global_nx/10);
  const int global_particle_ny = (2*mesh->global_ny/10);
#endif // if 0
#if 0
  // RANDOM ACROSS WHOLE MESH
  const int global_particle_start_x = 0;
  const int global_particle_nx = mesh->global_nx;
  const int global_particle_start_y = 0;
  const int global_particle_ny = mesh->global_ny;
#endif // if 0

  // Check if the start of data region is before or after our patch starts
  const int local_particle_left_off = 
    max(0, global_particle_start_x-mesh->x_off);
  const int local_particle_bottom_off = 
    max(0, global_particle_start_y-mesh->y_off);
  const int local_particle_right_off = 
    max(0, (mesh->x_off+local_nx)-(global_particle_start_x+global_particle_nx));
  const int local_particle_top_off = 
    max(0, (mesh->y_off+local_ny)-(global_particle_start_y+global_particle_ny));

  // The area of the active region shaded by this rank
  const int local_particle_nx =
    max(0, (local_nx-local_particle_left_off-local_particle_right_off));
  const int local_particle_ny = 
    max(0, (local_ny-local_particle_bottom_off-local_particle_top_off));

  bright_data->nlocal_particles = NPARTICLES*
    ((double)local_particle_nx*local_particle_ny)/
    ((double)global_particle_nx*global_particle_ny);

  // TODO: FIX THE OVERESTIMATED PARTICLE POPULATION MAXIMUMS
  bright_data->local_particles = (Particle*)malloc(sizeof(Particle)*NPARTICLES*2);
  bright_data->out_particles = (Particle*)malloc(sizeof(Particle)*NPARTICLES);
  bright_data->energy_tally = (double*)malloc(sizeof(double)*local_nx*local_ny);

  for(int ii = 0; ii < local_nx*local_ny; ++ii) {
    bright_data->energy_tally[ii] = 0.0;
  }

  // Check we are injecting some particle into this part of the mesh
  if(global_particle_start_x+global_particle_nx >= mesh->x_off && 
      global_particle_start_x < mesh->x_off+local_nx &&
      global_particle_start_y+global_particle_ny >= mesh->y_off && 
      global_particle_start_y < mesh->y_off+local_ny) {

    inject_particles(
        mesh, local_nx, local_ny, local_particle_left_off, 
        local_particle_bottom_off, local_particle_nx, local_particle_ny, 
        bright_data->nlocal_particles, bright_data->local_particles);
  }

  initialise_cross_sections(
      bright_data);

#ifdef MPI
  // Had to initialise this in the package directly as the data structure is not
  // general enough to place in the multi-package 
  int blocks[2] = { 8, 1 };
  MPI_Datatype types[2] = { MPI_DOUBLE, MPI_INT };
  MPI_Aint displacements[2] = { 0, blocks[0]*sizeof(double) };
  MPI_Type_create_struct(
      2, blocks, displacements, types, &particle_type);
  MPI_Type_commit(&particle_type);
#endif
}

// Acts as a particle source
void inject_particles(
    Mesh* mesh, const int local_nx, const int local_ny, 
    const int local_particle_left_off, const int local_particle_bottom_off,
    const int local_particle_nx, const int local_particle_ny, 
    const int nparticles, Particle* particles)
{
  START_PROFILING(&compute_profile);
  for(int ii = 0; ii < nparticles; ++ii) {
    initialise_particle(
        mesh->global_nx, mesh->global_ny, local_nx, local_ny, mesh->width, 
        mesh->height, mesh->edgex[local_particle_left_off+PAD], 
        mesh->edgey[local_particle_bottom_off+PAD], 
        local_particle_nx, local_particle_ny, mesh->x_off, 
        mesh->y_off, mesh->dt, mesh->edgex, mesh->edgey, &particles[ii]);
  }
  STOP_PROFILING(&compute_profile, "initialising particles");
}

// Initialises a new particle ready for tracking
void initialise_particle(
    const int global_nx, const int global_ny, const int local_nx, 
    const int local_ny, const double mesh_width, const double mesh_height, 
    const double particle_off_x, const double particle_off_y, 
    const int local_particle_nx, const int local_particle_ny, 
    const int x_off, const int y_off, const double dt, 
    const double* edgex, const double* edgey, Particle* particle)
{
  // Set the initial random location of the particle inside the source region
  particle->x = particle_off_x + 
    genrand()*(((double)local_particle_nx/global_nx)*mesh_width);
  particle->y = particle_off_y +
    genrand()*(((double)local_particle_ny/global_ny)*mesh_height);

  int cellx = 0;
  int celly = 0;

  // Have to check this way as mesh doesn't have to be uniform
  for(int ii = 0; ii < local_nx; ++ii) {
    if(particle->x >= edgex[ii+PAD] && particle->x < edgex[ii+PAD+1]) {
      cellx = x_off+ii;
      break;
    }
  }
  for(int ii = 0; ii < local_ny; ++ii) {
    if(particle->y >= edgey[ii+PAD] && particle->y < edgey[ii+PAD+1]) {
      celly = y_off+ii;
      break;
    }
  }
  particle->cell = celly*global_nx+cellx;

  // Generating theta has uniform density, however 0.0 and 1.0 produce the same 
  // value which introduces very very very small bias...
  const double theta = 2.0*M_PI*genrand();
  particle->omega_x = cos(theta);
  particle->omega_y = sin(theta);

  // This approximation sets mono-energetic initial state for source particles  
  particle->e = INITIAL_ENERGY;

  // Set a weight for the particle to track absorption
  particle->weight = 1.0;
  particle->dt_to_census = dt;
  particle->mfp_to_collision = 0.0;
}

// Reads in a cross-sectional data file
void read_cs_file(
    const char* filename, CrossSection* cs) 
{
  FILE* fp = fopen(filename, "r");
  int ch;

  cs->nentries = 0;
  while ((ch = fgetc(fp)) != EOF) {
    if(ch == '\n')
      cs->nentries++;
  }
  printf("File %s contains %d entries\n", filename, cs->nentries);

  rewind(fp);

  cs->key = (double*)malloc(sizeof(double)*cs->nentries);
  cs->value = (double*)malloc(sizeof(double)*cs->nentries);

  for(int ii = 0; ii < cs->nentries; ++ii) {
    // Skip tokens
    while((ch = fgetc(fp)) == ' ' || ch == '\n' || ch == '\r'){};

    // Jump out if we reach the end of the file early
    if(ch == EOF) {
      cs->nentries = ii;
      break;
    }

    ungetc(ch, fp);
    fscanf(fp, "%lf", &cs->key[ii]);
    while((ch = fgetc(fp)) == ' '){};
    ungetc(ch, fp);
    fscanf(fp, "%lf", &cs->value[ii]);
  }
}

// Initialises the state 
void initialise_cross_sections(
    BrightData* bright_data)
{
  bright_data->cs_scatter_table = (CrossSection*)malloc(sizeof(CrossSection));
  bright_data->cs_absorb_table = (CrossSection*)malloc(sizeof(CrossSection));
  read_cs_file(CS_SCATTER_FILENAME, bright_data->cs_scatter_table);
  read_cs_file(CS_CAPTURE_FILENAME, bright_data->cs_absorb_table);
}

