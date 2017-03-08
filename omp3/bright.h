#include "../bright_interface.h"

enum { PARTICLE_SENT, PARTICLE_DEAD, PARTICLE_CENSUS };

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy(
    const CrossSection* cs, const double energy, int* cs_index);

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const int initial, const double dt, 
    const int* neighbours, const double* density, const double* edgex, 
    const double* edgey, const double* edgedx, const double* edgedy, int* facets, 
    int* collisions, int* nparticles_sent, uint64_t* master_key, 
    const int ntotal_particles, const int nparticles_to_process, 
    int* nparticles, Particle* particles_start, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table, double* scalar_flux_tally, 
    double* energy_deposition_tally, RNPool* rn_pools);

// Handles an individual particle.
int handle_particle(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const int* neighbours, const double dt,
    const int initial, const int ntotal_particles, const double* density, 
    const double* edgex, const double* edgey, const double* edgedx, 
    const double* edgedy, const CrossSection* cs_scatter_table, 
    const CrossSection* cs_absorb_table, int* nparticles_sent, int* facets, 
    int* collisions, Particle* particle, 
    double* scalar_flux_tally, double* energy_deposition_tally, RNPool* rn_pool);

// Calculate the distance to the next facet
void calc_distance_to_facet(
    const int global_nx, const double x, const double y, const int x_off,
    const int y_off, const double omega_x, const double omega_y,
    const double particle_velocity, const int particle_cellx, 
    const int particle_celly, double* distance_to_facet,
    int* x_facet, const double* edgex, const double* edgey);

// Makes the necessary updates to the particle given that
// the facet was encountered
int handle_facet_encounter(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const int* neighbours, 
    const double distance_to_facet, int x_facet, int* nparticles_sent, 
    Particle* particle);

// Performs a binary search.
int binary_search(
    CrossSection* cs, const double energy);

// Handle the collision event, including absorption and scattering
int handle_collision(
    Particle* particle, const double macroscopic_cs_absorb, 
    const double macroscopic_cs_total, const double distance_to_collision, 
    RNPool* rn_pool);

// Sends a particle to a neighbour and replaces in the particle list
void send_and_mark_particle(
    const int destination, Particle* particle_to_replace);

// Tallies both the scalar flux and energy deposition in the cell
void update_tallies(
    const int nx, const int x_off, const int y_off, Particle* particle, 
    const double inv_ntotal_particles, const double energy_deposition,
    const double scalar_flux, double* scalar_flux_tally, 
    double* energy_deposition_tally);

void compress_particle_list(
    const int nparticles_to_process, Particle* particles_start,
    int nparticles_deleted);

// Calculate the energy deposition in the cell
double calculate_energy_deposition(
    const int global_nx, const int nx, const int x_off, const int y_off, 
    Particle* particle, const double inv_ntotal_particles, const double path_length, 
    const double number_density, const double microscopic_cs_absorb, 
    const double microscopic_cs_total);

