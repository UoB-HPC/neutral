#include <stdint.h>
#include "../neutral_data.h"

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const int initial, const double dt, 
    const int* neighbours, const double* density, const double* edgex, 
    const double* edgey, const double* edgedx, const double* edgedy, uint64_t* facets, 
    uint64_t* collisions, int* nparticles_sent, uint64_t* master_key, 
    const int ntotal_particles, const int nparticles_to_process, 
    int* nparticles, Particle* particles_start, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table, double* energy_deposition_tally, 
    int* reduce_array0, int* reduce_array1);

