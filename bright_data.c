#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "bright_data.h"
#include "../profiler.h"
#include "../shared.h"
#include "../params.h"

// Reads a cross section file
void read_cs_file(
    const char* filename, CrossSection* cs, Mesh* mesh);

// Initialises the set of cross sections
void initialise_cross_sections(
    BrightData* bright_data, Mesh* mesh);

// Initialises a new particle ready for tracking
void initialise_particle(
    const int index, const int global_nx, const int local_nx, const int local_ny, 
    const double local_particle_left_off, const double local_particle_bottom_off, 
    const double local_particle_width, const double local_particle_height, 
    const int x_off, const int y_off, const double dt, const double* edgex, 
    const double* edgey, const double initial_energy, RNPool* rn_pool, 
    Particles* particles);

// Initialises all of the Bright-specific data structures.
void initialise_bright_data(
    BrightData* bright_data, Mesh* mesh)
{
  const int local_nx = mesh->local_nx-2*PAD;
  const int local_ny = mesh->local_ny-2*PAD;

  bright_data->nparticles = 
    get_int_parameter("nparticles", bright_data->neutral_params_filename);
  bright_data->initial_energy = 
    get_double_parameter("initial_energy", bright_data->neutral_params_filename);

  // Initialise enough pools for every thread and a master pool
  bright_data->nrn_pools = bright_data->nthreads+1;
  bright_data->rn_pool_master_index = bright_data->nrn_pools-1;
  bright_data->rn_pools = (RNPool*)malloc(sizeof(RNPool)*(bright_data->nrn_pools));
  init_rn_pools(bright_data->rn_pools, bright_data->nrn_pools, bright_data->nparticles);

  int nkeys = 0;
  char* keys = (char*)malloc(sizeof(char)*MAX_KEYS*MAX_STR_LEN);
  double* values = (double*)malloc(sizeof(double)*MAX_KEYS);

  if(!get_key_value_parameter(
        "source", bright_data->neutral_params_filename, keys, values, &nkeys)) {
    TERMINATE("Parameter file %s did not contain a source entry.\n", 
        bright_data->neutral_params_filename);
  }

  // Fetch the width and height of the mesh
  const double mesh_width = mesh->edgex[mesh->global_nx+PAD];
  const double mesh_height = mesh->edgey[mesh->global_ny+PAD];

  // The last four keys are the bound specification
  const double source_xpos = values[nkeys-4]*mesh_width;
  const double source_ypos = values[nkeys-3]*mesh_height;
  const double source_width = values[nkeys-2]*mesh_width;
  const double source_height = values[nkeys-1]*mesh_height;
  const double rank_xpos_0 = mesh->edgex[mesh->x_off+PAD];
  const double rank_ypos_0 = mesh->edgey[mesh->y_off+PAD];
  const double rank_xpos_1 = mesh->edgex[local_nx+mesh->x_off+PAD];
  const double rank_ypos_1 = mesh->edgey[local_ny+mesh->y_off+PAD];

  // Calculate the shaded bounds
  const double local_particle_left_off =
    max(0.0, source_xpos-rank_xpos_0);
  const double local_particle_bottom_off =
    max(0.0, source_ypos-rank_ypos_0);
  const double local_particle_right_off =
    max(0.0, rank_xpos_1-(source_xpos+source_width));
  const double local_particle_top_off =
    max(0.0, rank_ypos_1-(source_ypos+source_height));
  const double local_particle_width = 
    max(0.0, (rank_xpos_1-rank_xpos_0)-
        (local_particle_right_off+local_particle_left_off));
  const double local_particle_height = 
    max(0.0, (rank_ypos_1-rank_ypos_0)-
        (local_particle_top_off+local_particle_bottom_off));

  // Calculate the number of particles we need based on the shaded area that
  // is covered by our source
  const double nlocal_particles_real = 
    bright_data->nparticles*
    (local_particle_width*local_particle_height)/(source_width*source_height);

  // Rounding hack to make sure correct number of particles is selected
  bright_data->nlocal_particles = nlocal_particles_real + 0.5;

  // TODO: SHOULD PROBABLY PERFORM A REDUCTION OVER THE NUMBER OF LOCAL PARTICLES
  // TO MAKE SURE THAT THEY ALL SUM UP TO THE CORRECT VALUE!

  const int vec_align = 256;
  // THIS IS A LOT OF DATA...
  bright_data->local_particles = 
    (Particles*)_mm_malloc(sizeof(Particles), vec_align);
  bright_data->local_particles->x = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->y = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->omega_x = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->omega_y = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->e = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->weight = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->dt_to_census = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->mfp_to_collision = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->distance_to_facet = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->microscopic_cs_absorb = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->microscopic_cs_scatter = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->local_density = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->cell_mfp = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->particle_velocity = 
    (double*)_mm_malloc(sizeof(double)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->x_facet = 
    (int*)_mm_malloc(sizeof(int)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->cellx = 
    (int*)_mm_malloc(sizeof(int)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->celly = 
    (int*)_mm_malloc(sizeof(int)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->scatter_cs_index = 
    (int*)_mm_malloc(sizeof(int)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->absorb_cs_index = 
    (int*)_mm_malloc(sizeof(int)*bright_data->nparticles*1.5, vec_align);
  bright_data->local_particles->next_event = 
    (int*)_mm_malloc(sizeof(int)*bright_data->nparticles*1.5, vec_align);

  if(!bright_data->local_particles) {
    TERMINATE("Could not allocate particle array.\n");
  }

  allocate_data(&bright_data->scalar_flux_tally, (mesh->local_nx)*(mesh->local_ny));
  allocate_data(&bright_data->energy_deposition_tally, (mesh->local_nx)*(mesh->local_ny));

#pragma omp parallel for
  for(int ii = 0; ii < (mesh->local_ny); ++ii) {
    for(int jj = 0; jj < (mesh->local_nx); ++jj) {
      const int ind = ii*(mesh->local_nx)+jj;
      bright_data->scalar_flux_tally[ind] = 0.0;
      bright_data->energy_deposition_tally[ind] = 0.0;
    }
  }

  // Inject some particles into the mesh if we need to
  if(bright_data->nlocal_particles) {
    inject_particles(
        mesh, local_nx, local_ny, local_particle_left_off, 
        local_particle_bottom_off, local_particle_width, local_particle_height, 
        bright_data->nlocal_particles, bright_data->initial_energy, 
        bright_data->rn_pools, bright_data->local_particles);
  }

  initialise_cross_sections(
      bright_data, mesh);

#ifdef MPI
  // Had to initialise this in the package directly as the data structure is not
  // general enough to place in the multi-package 
  const int blocks[3] = { 8, 1, 1 };
  MPI_Datatype types[3] = { MPI_DOUBLE, MPI_UINT64_T, MPI_INT };
  MPI_Aint disp[3] = { 0, blocks[0]*sizeof(double), disp[0]+sizeof(uint64_t) };
  MPI_Type_create_struct(
      2, blocks, disp, types, &particle_type);
  MPI_Type_commit(&particle_type);
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
  START_PROFILING(&compute_profile);

#pragma omp parallel for
  for(int ii = 0; ii < nparticles; ++ii) {
    initialise_particle(
        ii, mesh->global_nx, local_nx, local_ny, local_particle_left_off, 
        local_particle_bottom_off, local_particle_width, local_particle_height, 
        mesh->x_off, mesh->y_off, mesh->dt, mesh->edgex, mesh->edgey, 
        initial_energy, &rn_pools[omp_get_thread_num()], particles);
  }
  STOP_PROFILING(&compute_profile, "initialising particles");
}

// Initialises a new particle ready for tracking
void initialise_particle(
    const int index, const int global_nx, const int local_nx, const int local_ny, 
    const double local_particle_left_off, const double local_particle_bottom_off, 
    const double local_particle_width, const double local_particle_height, 
    const int x_off, const int y_off, const double dt, const double* edgex, 
    const double* edgey, const double initial_energy, RNPool* rn_pool, 
    Particles* particles)
{
  // Set the initial nandom location of the particle inside the source region
  particles->x[index] = local_particle_left_off + 
    getrand(rn_pool)*local_particle_width;
  particles->y[index] = local_particle_bottom_off + 
    getrand(rn_pool)*local_particle_height;

  // Check the location of the specific cell that the particle sits within.
  // We have to check this explicitly because the mesh might be non-uniform.
  int cellx = 0;
  int celly = 0;
  for(int ii = 0; ii < local_nx; ++ii) {
    if(particles->x[index] >= edgex[ii+PAD] && 
        particles->x[index] < edgex[ii+PAD+1]) {
      cellx = x_off+ii;
      break;
    }
  }
  for(int ii = 0; ii < local_ny; ++ii) {
    if(particles->y[index] >= edgey[ii+PAD] && 
        particles->y[index] < edgey[ii+PAD+1]) {
      celly = y_off+ii;
      break;
    }
  }

  particles->cellx[index] = cellx;
  particles->celly[index] = celly;

  // Generating theta has uniform density, however 0.0 and 1.0 produce the same 
  // value which introduces very very very small bias...
  const double theta = 2.0*M_PI*getrand(rn_pool);
  particles->omega_x[index] = cos(theta);
  particles->omega_y[index] = sin(theta);

  // This approximation sets mono-energetic initial state for source particles  
  particles->e[index] = initial_energy;

  // Set a weight for the particle to track absorption
  particles->weight[index] = 1.0;
  particles->dt_to_census[index] = dt;
  particles->mfp_to_collision[index] = 0.0;
  particles->scatter_cs_index[index] = -1;
  particles->absorb_cs_index[index] = -1;
  particles->next_event[index] = FACET;
}

// Reads in a cross-sectional data file
void read_cs_file(
    const char* filename, CrossSection* cs, Mesh* mesh) 
{
  FILE* fp = fopen(filename, "r");
  if(!fp) {
    TERMINATE("Could not open the cross section file: %s\n", filename);
  }

  // Count the number of entries in the file
  int ch;
  cs->nentries = 0;
  while ((ch = fgetc(fp)) != EOF) {
    if(ch == '\n') {
      cs->nentries++;
    }
  }

  if(mesh->rank == MASTER) {
    printf("File %s contains %d entries\n", filename, cs->nentries);
  }

  rewind(fp);

  cs->key = (double*)malloc(sizeof(double)*cs->nentries);
  cs->value = (double*)malloc(sizeof(double)*cs->nentries);

  for(int ii = 0; ii < cs->nentries; ++ii) {
    // Skip whitespace tokens
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
    BrightData* bright_data, Mesh* mesh)
{
  bright_data->cs_scatter_table = (CrossSection*)malloc(sizeof(CrossSection));
  bright_data->cs_absorb_table = (CrossSection*)malloc(sizeof(CrossSection));
  read_cs_file(CS_SCATTER_FILENAME, bright_data->cs_scatter_table, mesh);
  read_cs_file(CS_CAPTURE_FILENAME, bright_data->cs_absorb_table, mesh);
}

