#pragma once

#include "../mesh.h"
#include "../shared_data.h"
#include "bright_data.h"

#ifdef __cplusplus
extern "C" {
#endif

// Performs a solve of dependent variables for particle transport.
void solve_transport_2d(
    const int nx, const int ny, const int global_nx, const int global_ny, 
    const int x_off, const int y_off, const double dt, int* nlocal_particles, 
    const int* neighbours, Particle* particles, const double* density, 
    const double* edgex, const double* edgey, Particle* out_particles, 
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table, 
    double* energy_tally);

#ifdef __cplusplus
}
#endif

