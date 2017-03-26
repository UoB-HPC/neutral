#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "neutral_data.h"
#include "../profiler.h"
#include "../shared.h"
#include "../params.h"

// Reads a cross section file
void read_cs_file(
    const char* filename, CrossSection* cs, Mesh* mesh);

// Initialises the set of cross sections
void initialise_cross_sections(
    NeutralData* neutral_data, Mesh* mesh);

// Initialises a new particle ready for tracking
void initialise_particle(
    const int global_nx, const int local_nx, const int local_ny, 
    const double local_particle_left_off, const double local_particle_bottom_off, 
    const double local_particle_width, const double local_particle_height, 
    const int x_off, const int y_off, const double dt, const double* edgex, 
    const double* edgey, const double initial_energy, RNPool* rn_pool, 
    Particle* particle);

// Initialises all of the neutral-specific data structures.
void initialise_neutral_data(
    NeutralData* neutral_data, Mesh* mesh, RNPool* rn_pool)
{
  const int local_nx = mesh->local_nx-2*PAD;
  const int local_ny = mesh->local_ny-2*PAD;

  neutral_data->nparticles = 
    get_int_parameter("nparticles", neutral_data->neutral_params_filename);
  neutral_data->initial_energy = 
    get_double_parameter("initial_energy", neutral_data->neutral_params_filename);

  int nkeys = 0;
  char* keys = (char*)malloc(sizeof(char)*MAX_KEYS*MAX_STR_LEN);
  double* values = (double*)malloc(sizeof(double)*MAX_KEYS);

  if(!get_key_value_parameter(
        "source", neutral_data->neutral_params_filename, keys, values, &nkeys)) {
    TERMINATE("Parameter file %s did not contain a source entry.\n", 
        neutral_data->neutral_params_filename);
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
    neutral_data->nparticles*
    (local_particle_width*local_particle_height)/(source_width*source_height);

  // Rounding hack to make sure correct number of particles is selected
  neutral_data->nlocal_particles = nlocal_particles_real + 0.5;

  // TODO: SHOULD PROBABLY PERFORM A REDUCTION OVER THE NUMBER OF LOCAL PARTICLES
  // TO MAKE SURE THAT THEY ALL SUM UP TO THE CORRECT VALUE

#ifdef SoA
  neutral_data->local_particles = (Particle*)malloc(sizeof(Particle));
#else
  neutral_data->local_particles = 
    (Particle*)malloc(sizeof(Particle)*neutral_data->nparticles*2);
#endif

  if(!neutral_data->local_particles) {
    TERMINATE("Could not allocate particle array.\n");
  }

  allocate_data(&neutral_data->scalar_flux_tally, local_nx*local_ny);
  allocate_data(&neutral_data->energy_deposition_tally, local_nx*local_ny);

#pragma omp parallel for
  for(int ii = 0; ii < local_nx*local_ny; ++ii) {
    neutral_data->scalar_flux_tally[ii] = 0.0;
    neutral_data->energy_deposition_tally[ii] = 0.0;
  }

  allocate_int_data(&neutral_data->reduce_array0, local_nx*local_ny);
  allocate_int_data(&neutral_data->reduce_array1, local_nx*local_ny);

  // Inject some particles into the mesh if we need to
  if(neutral_data->nlocal_particles) {
    inject_particles(
        neutral_data->nparticles, mesh->global_nx, mesh->local_nx, mesh->local_ny, 
        local_particle_left_off, local_particle_bottom_off, local_particle_width, 
        local_particle_height, mesh->x_off, mesh->y_off, mesh->dt, mesh->edgex, 
        mesh->edgey, neutral_data->initial_energy, rn_pool, 
        neutral_data->local_particles);
  }

  initialise_cross_sections(
      neutral_data, mesh);

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
    NeutralData* neutral_data, Mesh* mesh)
{
  neutral_data->cs_scatter_table = (CrossSection*)malloc(sizeof(CrossSection));
  neutral_data->cs_absorb_table = (CrossSection*)malloc(sizeof(CrossSection));
  read_cs_file(CS_SCATTER_FILENAME, neutral_data->cs_scatter_table, mesh);
  read_cs_file(CS_CAPTURE_FILENAME, neutral_data->cs_absorb_table, mesh);
}

