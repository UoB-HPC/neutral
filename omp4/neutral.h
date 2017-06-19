#include "../neutral_interface.h"

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int pad, const int x_off, const int y_off, const int initial, const double dt, 
    const int* neighbours, const double* density, const double* edgex, 
    const double* edgey, const double* edgedx, const double* edgedy, uint64_t* facets, 
    uint64_t* collisions, int* nparticles_sent, uint64_t* master_key, 
    const int ntotal_particles, const int nparticles_to_process, 
    int* nparticles, Particle* particles_start, CrossSection* cs_scatter_table, 
    CrossSection* cs_absorb_table, double* energy_deposition_tally);

#pragma omp declare target

// Handles an individual particle.
int handle_particle(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int pad, const int x_off, const int y_off, const int* neighbours, const double dt,
    const int initial, const int ntotal_particles, const double* density, 
    const double* edgex, const double* edgey, const double* edgedx, const double* edgedy, 
    const double* cs_absorb_table_keys, const double* cs_scatter_table_keys,
    const double* cs_absorb_table_values, const double* cs_scatter_table_values,
    const int cs_absorb_table_nentries, const int cs_scatter_table_nentries,
    int* nparticles_sent, uint64_t* facets, uint64_t* collisions, const int pp, 
    double* p_x, double* p_y, double* p_omega_x, double* p_omega_y, 
    double* p_energy, double* p_weight, double* p_dt_to_census, 
    double* p_mfp_to_collision, int* p_cellx, int* p_celly, int* p_dead, 
    double* energy_deposition_tally, const uint64_t master_key);

// Tallies the energy deposition in the cell
void update_tallies(
    const int nx, const int x_off, const int y_off, const int p_cellx, 
    const int p_celly, const double inv_ntotal_particles, 
    const double energy_deposition, double* energy_deposition_tally);

// Handle the collision event, including absorption and scattering
int handle_collision(
    double* p_x, double* p_y, double* p_omega_x, double* p_omega_y, double* p_e, 
    double* p_weight, double* p_dt_to_census, double* p_mfp_to_collision, 
    int* p_cellx, int* p_celly, int* p_dead, const double macroscopic_cs_absorb, 
    uint64_t counter, uint64_t* local_key, const double macroscopic_cs_total, 
    const double distance_to_collision, uint64_t master_key);

// Makes the necessary updates to the particle given that
// the facet was encountered
int handle_facet_encounter(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const int* neighbours, 
    const double distance_to_facet, int x_facet, int* nparticles_sent, 
    double* p_x, double* p_y, double* p_omega_x, double* p_omega_y,
    int* p_cellx, int* p_celly, int* p_dead);

// Sends a particle to a neighbour and replaces in the particle list
void send_and_mark_particle(
    const int destination, int* p_dead);

// Calculate the distance to the next facet
void calc_distance_to_facet(
    const int global_nx, const double x, const double y, const int pad, 
    const int x_off, const int y_off, const double omega_x, 
    const double omega_y,
    const double speed, const int particle_cellx, 
    const int particle_celly, double* distance_to_facet,
    int* x_facet, const double* edgex, const double* edgey);

// Calculate the energy deposition in the cell
double calculate_energy_deposition(
    const int global_nx, const int nx, const int x_off, const int y_off, 
    const double inv_ntotal_particles, const double path_length, 
    const double p_energy, const double p_weight, const double number_density, 
    const double microscopic_cs_absorb, const double microscopic_cs_total);

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy(
    const double* keys, const double* values, const int nentries,
    const double energy, int* cs_index);

// Generates a pair of random numbers
void generate_random_numbers(
    const uint64_t master_key, const uint64_t secondary_key, 
    const uint64_t gid, double* rn0, double* rn1);

// Random123 methods
threefry2x64_ctr_t threefry2x64_R(
        unsigned int Nrounds, threefry2x64_ctr_t counter, threefry2x64_key_t key);
uint64_t RotL_64(
        uint64_t x, unsigned int N);

#pragma omp end declare target

// Validates the results of the simulation
void validate(
    const int nx, const int ny, const char* params_filename, 
    const int rank, double* energy_deposition_tally);

