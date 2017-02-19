#pragma once

#include "bright_data.h"
#include "../mesh.h"
#include "../shared_data.h"

#ifdef __cplusplus
extern "C" {
#endif

  // Performs a solve of dependent variables for particle transport.
  void solve_transport_2d(
      const int nx, const int ny, const int global_nx, const int global_ny, 
      const int x_off, const int y_off, const double dt, const int ntotal_particles,
      int* nlocal_particles, const int* neighbours, Particle* particles, 
      const double* density, const double* edgex, const double* edgey, 
      const double* edgedx, const double* edgedy, Particle* particles_out, 
      CrossSection* cs_scatter_table, CrossSection* cs_absorb_table, 
      double* scalar_flux_tally, double* energy_deposition_tally);

  // Validates the results of the simulation
  void validate(
      const int nx, const int ny, const int nglobal_particles, const double dt,
      const int niters, const int rank, double* energy_tally);

#ifdef __cplusplus
}
#endif

