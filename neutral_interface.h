#pragma once

#include "../mesh.h"
#include "../shared_data.h"
#include "neutral_data.h"

#ifdef __cplusplus
extern "C" {
#endif

void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const int pad, const int x_off, const int y_off, const float dt,
    const int ntotal_particles, int* nlocal_particles, uint64_t* master_key,
    const int* neighbours, Particle* particles, const float* density,
    const float* edgex, const float* edgey, const float* edgedx,
    const float* edgedy, CrossSection* cs_scatter_table,
    CrossSection* cs_absorb_table, float* energy_deposition_tally,
    uint64_t* reduce_array0, uint64_t* reduce_array1, uint64_t* reduce_array2,
    uint64_t* facet_events, uint64_t* collision_events);

// Initialises a new particle ready for tracking
size_t inject_particles(const int nparticles, const int global_nx,
                        const int local_nx, const int local_ny, const int pad,
                        const float local_particle_left_off,
                        const float local_particle_bottom_off,
                        const float local_particle_width,
                        const float local_particle_height, const int x_off,
                        const int y_off, const float dt, const float* edgex,
                        const float* edgey, const float initial_energy,
                        const uint64_t master_key, Particle** particles);

size_t initialise_float_mesh(NeutralData* neutral_data, Mesh* mesh, SharedData* shared_data);

// Validates the results of the simulation
void validate(const int nx, const int ny, const char* params_filename,
              const int rank, float* energy_tally);

#ifdef __cplusplus
}
#endif
