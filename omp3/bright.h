#include "../bright_interface.h"

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const double dt, const int* neighbours, 
    const double* density, const double* edgex, const double* edgey, 
    uint64_t* facets, uint64_t* collisions, int* nparticles_sent, 
    uint64_t* master_key, const int nparticles_total, 
    const int nparticles_to_process, int* nparticles, Particles* particles, 
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table, 
    double* scalar_flux_tally, double* energy_deposition_tally, RNPool* rn_pools);

// Initialises ready for the event cycles
void event_initialisation(
    const int nparticles, const int particles_offset, const int nx, 
    const int x_off, const int y_off, Particles* particles, const double dt, 
    const double* density, const int nthreads, RNPool* rn_pools, 
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table);

// Calculates the next event for each particle
int calc_next_event(
    const int nparticles, const int particles_offset, Particles* particles, 
    uint64_t* facets, uint64_t* collisions, const int x_off, 
    const int y_off, const double* edgex, const double* edgey);

// Handle all of the facet encounters
void handle_facets(
    const int nparticles, const int particles_offset, const int global_nx, 
    const int global_ny, const int nx, const int ny, const int x_off, 
    const int y_off, const int* neighbours, int* nparticles_sent, 
    Particles* particles, const double* edgex, const double* edgey, 
    const double* density, int* nparticles_out, double* scalar_flux_tally, 
    double* energy_deposition_tally, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table);

// Handle all of the collision events
void handle_collisions(
    const int nparticles, const int particles_offset, const int nx, 
    const int x_off, const int y_off, Particles* particles, const double* edgex, 
    const double* edgey, RNPool* rn_pools, int* nparticles_dead, 
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table, 
    double* scalar_flux_tally, double* energy_deposition_tally);

// Handles all of the census events
void handle_census(
    const int nparticles, const int particles_offset, const int nx, 
    const int x_off, const int y_off, Particles* particles, const double* density, 
    const double* edgex, const double* edgey, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table, double* scalar_flux_tally, 
    double* energy_deposition_tally);

// Calculates the distance to the facet for all cells
void calc_distance_to_facet(
    const int nparticles, const int particles_offset, const int x_off, 
    const int y_off, Particles* particles, 
    const double* edgex, const double* edgey);

// Tallies both the scalar flux and energy deposition in the cell
void update_tallies(
    const int nx, Particles* particles, const int x_off, const int y_off, 
    const int nparticles, const int particles_offset, const int exclude_census, 
    double* scalar_flux_tally, double* energy_deposition_tally);

// Sends a particles to a neighbour and replaces in the particles list
void send_and_mark_particle(
    const int destination, const int pindex, Particles* particles);

// Calculate the energy deposition in the cell
#pragma omp declare simd
double calculate_energy_deposition(
    const int pindex, Particles* particles, const double path_length, 
    const double number_density, const double microscopic_cs_absorb, 
    const double microscopic_cs_total);

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy(
    const CrossSection* cs, const double energy, int* cs_index);

// Validates the results of the simulation
void validate(
    const int nx, const int ny, const char* params_filename, 
    const int rank, double* energy_deposition_tally);

