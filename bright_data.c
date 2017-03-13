#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "bright_data.h"
#include "bright_interface.h"
#include "../profiler.h"
#include "../shared.h"
#include "../params.h"

// Initialises the set of cross sections
void initialise_cross_sections(
    BrightData* bright_data, Mesh* mesh);

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

  // The last four keys are the bound specification
  const double source_xpos = values[nkeys-4]*mesh->width;
  const double source_ypos = values[nkeys-3]*mesh->height;
  const double source_width = values[nkeys-2]*mesh->width;
  const double source_height = values[nkeys-1]*mesh->height;

  double* mesh_edgex_0 = &mesh->edgex[mesh->x_off+PAD];
  double* mesh_edgey_0 = &mesh->edgey[mesh->y_off+PAD];
  double* mesh_edgex_1 = &mesh->edgex[local_nx+mesh->x_off+PAD];
  double* mesh_edgey_1 = &mesh->edgey[local_ny+mesh->y_off+PAD];
  double* rank_xpos_0;
  double* rank_ypos_0;
  double* rank_xpos_1;
  double* rank_ypos_1;
  allocate_host_data(&rank_xpos_0, 1);
  allocate_host_data(&rank_ypos_0, 1);
  allocate_host_data(&rank_xpos_1, 1);
  allocate_host_data(&rank_ypos_1, 1);
  sync_data(1, &mesh_edgex_0, &rank_xpos_0, RECV);
  sync_data(1, &mesh_edgey_0, &rank_ypos_0, RECV);
  sync_data(1, &mesh_edgex_1, &rank_xpos_1, RECV);
  sync_data(1, &mesh_edgey_1, &rank_ypos_1, RECV);

  // Calculate the shaded bounds
  const double local_particle_left_off =
    max(0.0, source_xpos-*rank_xpos_0);
  const double local_particle_bottom_off =
    max(0.0, source_ypos-*rank_ypos_0);
  const double local_particle_right_off =
    max(0.0, *rank_xpos_1-(source_xpos+source_width));
  const double local_particle_top_off =
    max(0.0, *rank_ypos_1-(source_ypos+source_height));
  const double local_particle_width = 
    max(0.0, (*rank_xpos_1-*rank_xpos_0)-
        (local_particle_right_off+local_particle_left_off));
  const double local_particle_height = 
    max(0.0, (*rank_ypos_1-*rank_ypos_0)-
        (local_particle_top_off+local_particle_bottom_off));

  deallocate_host_data(rank_xpos_0);
  deallocate_host_data(rank_ypos_0);
  deallocate_host_data(rank_xpos_1);
  deallocate_host_data(rank_ypos_1);

  // Calculate the number of particles we need based on the shaded area that
  // is covered by our source
  const double nlocal_particles_real = 
    bright_data->nparticles*
    (local_particle_width*local_particle_height)/(source_width*source_height);

  // Rounding hack to make sure correct number of particles is selected
  bright_data->nlocal_particles = nlocal_particles_real + 0.5;

  // TODO: SHOULD PROBABLY PERFORM A REDUCTION OVER THE NUMBER OF LOCAL PARTICLES
  // TO MAKE SURE THAT THEY ALL SUM UP TO THE CORRECT VALUE!


  // THIS IS A LOT OF DATA...
  bright_data->local_particles = 
    (Particles*)_mm_malloc(sizeof(Particles), VEC_ALIGN);

  Particles* particle = bright_data->local_particles;

  size_t allocation = 0;
  allocation += allocate_data(&particle->x,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->y,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->omega_x,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->omega_y,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->e,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->weight,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->dt_to_census,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->mfp_to_collision,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->distance_to_facet,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->local_density,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->cell_mfp,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->particle_velocity,bright_data->nparticles*1.5);
  allocation += allocate_data(&particle->energy_deposition,bright_data->nparticles*1.5);
  allocation += allocate_int_data(&particle->x_facet,bright_data->nparticles*1.5);
  allocation += allocate_int_data(&particle->cellx,bright_data->nparticles*1.5);
  allocation += allocate_int_data(&particle->celly,bright_data->nparticles*1.5);
  allocation += allocate_int_data(&particle->scatter_cs_index,bright_data->nparticles*1.5);
  allocation += allocate_int_data(&particle->absorb_cs_index,bright_data->nparticles*1.5);
  allocation += allocate_int_data(&particle->next_event,bright_data->nparticles*1.5);
  allocation += 
    allocate_data(&bright_data->scalar_flux_tally, (mesh->local_nx)*(mesh->local_ny));
  allocation += 
    allocate_data(&bright_data->energy_deposition_tally, (mesh->local_nx)*(mesh->local_ny));

  printf("Allocating %zu bytes of data.\n", allocation);

  if(!bright_data->local_particles) {
    TERMINATE("Could not allocate particle array.\n");
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

#if 0
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
#endif // if 0
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

  double* h_keys;
  double* h_values;
  allocate_host_data(&h_keys, cs->nentries);
  allocate_host_data(&h_values, cs->nentries);

  for(int ii = 0; ii < cs->nentries; ++ii) {
    // Skip whitespace tokens
    while((ch = fgetc(fp)) == ' ' || ch == '\n' || ch == '\r'){};

    // Jump out if we reach the end of the file early
    if(ch == EOF) {
      cs->nentries = ii;
      break;
    }

    ungetc(ch, fp);
    fscanf(fp, "%lf", &h_keys[ii]);
    while((ch = fgetc(fp)) == ' '){};
    ungetc(ch, fp);
    fscanf(fp, "%lf", &h_values[ii]);
  }

  // Copy the cross sectional table into device memory if appropriate
  allocate_data(&cs->keys, cs->nentries);
  allocate_data(&cs->values, cs->nentries);
  sync_data(cs->nentries, &h_keys, &cs->keys, SEND);
  sync_data(cs->nentries, &h_values, &cs->values, SEND);
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

