#include "../neutral_data.h"
#include <stdint.h>

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny,
    const int pad, const int x_off, const int y_off, const int initial,
    const float dt, const int* neighbours, const float* density,
    const float* edgex, const float* edgey, const float* edgedx,
    const float* edgedy, uint64_t* facets, uint64_t* collisions,
    int* nparticles_sent, uint64_t* master_key, const int nparticles_total,
    const int nparticles_to_process, Particle* particles,
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table,
    float* energy_deposition_tally, uint64_t* nfacets_reduce_array, 
    uint64_t* ncollisions_reduce_array, uint64_t* nprocessed_reduce_array);

