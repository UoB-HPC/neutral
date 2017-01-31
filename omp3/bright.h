#include "../bright_interface.h"

enum { PARTICLE_SENT, PARTICLE_DEAD, PARTICLE_CENSUS };

// Fetch the cross section for a particular energy value
double total_cs_for_energy(
    const CrossSection* cs, const double energy, const double rho);

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const int initial, const double dt, 
    const int* neighbours, const double* density, const double* edgex, 
    const double* edgey, int* facets, int* collisions, int* nparticles_sent, 
    const int nparticles_to_process, int* nparticles, Particle* particles_start, 
    Particle* particles_out, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table, double* energy_tally);

// Handles an individual particle.
int handle_particle(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const int* neighbours, const double dt,
    const int initial, const double* density, const double* edgex, 
    const double* edgey, const CrossSection* cs_scatter_table, 
    const CrossSection* cs_absorb_table, Particle* particles_end, 
    int* nparticles_sent, int* facets, int* collisions, Particle* particle, 
    Particle* particle_out, double* energy_tally);

// Calculate the distance to the next facet
void calc_distance_to_facet(
    const int global_nx, const double x, const double y, const int x_off,
    const int y_off, const double omega_x, const double omega_y,
    const double particle_velocity, const int cell, double* distance_to_facet,
    int* x_facet, const double* edgex, const double* edgey);

// Makes the necessary updates to the particle given that
// the facet was encountered
int handle_facet_encounter(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const int* neighbours, 
    const double distance_to_facet, int x_facet, int* nparticles_sent, 
    Particle* particle, Particle* particles_end, Particle* particle_out);

// Performs a binary search.
int binary_search(
    CrossSection* cs, const double energy);

// Handle the collision event, including absorption and scattering
int handle_collision(
    Particle* particle, Particle* particles_end, const int global_nx, 
    const int nx, const int x_off, const int y_off, const double cs_absorb, 
    const double cs_total, const double distance_to_collision, 
    double* energy_tally);

// Sends a particle to a neighbour and replaces in the particle list
void send_and_replace_particle(
    const int destination, Particle* particles_end, Particle* particle_to_replace, 
    Particle* particle_out);

