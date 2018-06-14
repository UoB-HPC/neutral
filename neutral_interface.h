#pragma once

#include "../mesh.h"
#include "../shared_data.h"
#include "neutral_data.h"

#ifdef __cplusplus
extern "C" {
#endif

void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny,
    const uint64_t master_key, const int pad, const int x_off, const int y_off, 
    const double dt, const int ntotal_particles, int* nlocal_particles,
    const int* neighbours, Particle* particles, const double* density,
    const double* edgex, const double* edgey, const double* edgedx,
    const double* edgedy, CrossSection* cs_scatter_table,
    CrossSection* cs_absorb_table, double* energy_deposition_tally,
    uint64_t* reduce_array0, uint64_t* reduce_array1, uint64_t* reduce_array2,
    uint64_t* facet_events, uint64_t* collision_events);

// Initialises a new particle ready for tracking
size_t inject_particles(const int nparticles, const int global_nx,
    const int local_nx, const int local_ny, const int pad,
    const double local_particle_left_off,
    const double local_particle_bottom_off,
    const double local_particle_width,
    const double local_particle_height, const int x_off,
    const int y_off, const double dt, const double* edgex,
    const double* edgey, const double initial_energy,
    Particle** particles);


// Validates the results of the simulation
void validate(const int nx, const int ny, const char* params_filename,
              const int rank, double* energy_tally);

#ifdef __cplusplus
}
#endif
