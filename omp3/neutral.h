#include "../neutral_interface.h"

// Handles the current active batch of particles
void handle_particles(
    const int global_nx, const int global_ny, const int nx, const int ny,
    const int pad, const int x_off, const int y_off, const int initial,
    const double dt, const int* neighbours, const double* density,
    const double* edgex, const double* edgey, const double* edgedx,
    const double* edgedy, uint64_t* facets, uint64_t* collisions,
    uint64_t* master_key, const int ntotal_particles,
    const int nparticles_to_process, Particle* particles_start,
    CrossSection* cs_scatter_table, CrossSection* cs_absorb_table,
    double* energy_deposition_tally);


// Handle facet event
static inline void facet_event(
    const int global_nx, const int global_ny, const int nx, const int ny, 
    const int x_off, const int y_off, const double inv_ntotal_particles, 
    const double* distance_to_facet, const double* speed, const double* cell_mfp, 
    const int* x_facet, const double* density, const int* neighbours, const int ip, 
    double* energy_deposition, double* number_density, double* microscopic_cs_scatter, 
    double* microscopic_cs_absorb, double* macroscopic_cs_scatter, 
    double* macroscopic_cs_absorb, double* energy_deposition_tally, 
    double* local_density, double* p_energy, double* p_weight, int* p_cellx, 
    int* p_celly, double* p_mfp_to_collision, double* p_dt_to_census, double* p_x, 
    double* p_y, double* p_omega_x, double* p_omega_y);

// Handles a collision event
static inline void collision_event(
    const int ip, const int global_nx, const int nx, const int x_off, const int y_off,
    const double inv_ntotal_particles, const double distance_to_collision,
    const double local_density, const CrossSection* cs_scatter_table,
    const CrossSection* cs_absorb_table, uint64_t counter_off,
    const uint64_t* master_key, double* energy_deposition,
    double* number_density, double* microscopic_cs_scatter,
    double* microscopic_cs_absorb, double* macroscopic_cs_scatter,
    double* macroscopic_cs_absorb, double* energy_deposition_tally,
    int* scatter_cs_index, int* absorb_cs_index, 
    double* speed, double* p_x, double* p_y, int* p_dead, double* p_energy, 
    double* p_omega_x, double* p_omega_y, uint64_t* p_key, 
    double* p_mfp_to_collision, double* p_dt_to_census, double* p_weight,
    int* p_cellx, int* p_celly, int* found);

// Handles the census event
void census_event(const int global_nx, const int nx, const int x_off,
    const int y_off, const double inv_ntotal_particles,
    const double distance_to_census, const double cell_mfp,
    const int ip, double* energy_deposition,
    double* number_density, double* microscopic_cs_scatter,
    double* microscopic_cs_absorb, double* energy_deposition_tally, double* p_x, 
    double* p_y, double* p_omega_x, double* p_omega_y, 
    double* p_mfp_to_collision, double* p_dt_to_census, double* p_energy, 
    double* p_weight, int* p_cellx, int* p_celly);

// Tallies the energy deposition in the cell
void update_tallies(const int nx, const int x_off, const int y_off,
                    const int ip, const double inv_ntotal_particles,
                    const double energy_deposition,
                    double* energy_deposition_tally, int* p_cellx, 
                    int* p_celly);

// Handle the collision event, including absorption and scattering
int handle_collision(Particle* particle, const double macroscopic_cs_absorb,
                     uint64_t* counter, const double macroscopic_cs_total,
                     const double distance_to_collision, uint64_t master_key);

// Sends a particle to a neighbour and replaces in the particle list
void send_and_mark_particle(const int destination, Particle* particle);

// Calculate the distance to the next facet
void calc_distance_to_facet(const int global_nx, const double x, const double y,
                            const int pad, const int x_off, const int y_off,
                            const double omega_x, const double omega_y,
                            const double speed, const int particle_cellx,
                            const int particle_celly, double* distance_to_facet,
                            int* x_facet, const double* edgex,
                            const double* edgey);

// Calculate the energy deposition in the cell
double calculate_energy_deposition(
    const int global_nx, const int nx, const int x_off, const int y_off,
    const int ip, const double inv_ntotal_particles,
    const double path_length, const double number_density,
    const double microscopic_cs_absorb, const double microscopic_cs_total,
    double* p_energy, double* p_weight);

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy_binary(
    const CrossSection* cs, const double energy, int* cs_index);

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy_linear(
    const CrossSection* cs, const double energy, int* cs_index, int* found);

// Generates a pair of random numbers
void generate_random_numbers(const uint64_t master_key,
                             const uint64_t secondary_key, const uint64_t gid,
                             double* rn0, double* rn1, double* rn2, double* rn3);
