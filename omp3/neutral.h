#include "../neutral_interface.h"

// Handles the current active batch of particles
void handle_particles(const int global_nx, const int global_ny, const int nx,
                      const int ny, const uint64_t master_key, const int pad,
                      const int x_off, const int y_off, const int initial,
                      const double dt, const int* neighbours,
                      const double* density, const double* edgex,
                      const double* edgey, const double* edgedx,
                      const double* edgedy, uint64_t* facets,
                      uint64_t* collisions, const int ntotal_particles,
                      const int nparticles_to_process,
                      Particle* particles_start, CrossSection* cs_scatter_table,
                      CrossSection* cs_absorb_table,
                      double* energy_deposition_tally);

// Handle facet event
int facet_event(const int global_nx, const int global_ny, const int nx,
                const int ny, const int x_off, const int y_off,
                const double inv_ntotal_particles,
                const double distance_to_facet, const double speed,
                const double cell_mfp, const int x_facet, const double* density,
                const int* neighbours, Particle* particle,
                double* energy_deposition, double* number_density,
                double* microscopic_cs_scatter, double* microscopic_cs_absorb,
                double* macroscopic_cs_scatter, double* macroscopic_cs_absorb,
                double* energy_deposition_tally, int* cellx, int* celly,
                double* local_density);

// Handles a collision event
int collision_event(
    const int global_nx, const int nx, const int x_off, const int y_off,
    const uint64_t pkey, const uint64_t master_key,
    const double inv_ntotal_particles, const double distance_to_collision,
    const double local_density, const CrossSection* cs_scatter_table,
    const CrossSection* cs_absorb_table, Particle* particle, uint64_t* counter,
    double* energy_deposition, double* number_density,
    double* microscopic_cs_scatter, double* microscopic_cs_absorb,
    double* macroscopic_cs_scatter, double* macroscopic_cs_absorb,
    double* energy_deposition_tally, int* scatter_cs_index,
    int* absorb_cs_index, double rn[NRANDOM_NUMBERS], double* speed);

void census_event(const int global_nx, const int nx, const int x_off,
                  const int y_off, const double inv_ntotal_particles,
                  const double distance_to_census, const double cell_mfp,
                  Particle* particle, double* energy_deposition,
                  double* number_density, double* microscopic_cs_scatter,
                  double* microscopic_cs_absorb,
                  double* energy_deposition_tally);

// Tallies the energy deposition in the cell
void update_tallies(const int nx, const int x_off, const int y_off,
                    Particle* particle, const double inv_ntotal_particles,
                    const double energy_deposition,
                    double* energy_deposition_tally);

// Handle the collision event, including absorption and scattering
int handle_collision(Particle* particle, const double macroscopic_cs_absorb,
                     uint64_t* counter, const double macroscopic_cs_total,
                     const double distance_to_collision);

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
    Particle* particle, const double inv_ntotal_particles,
    const double path_length, const double number_density,
    const double microscopic_cs_absorb, const double microscopic_cs_total);

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy(const CrossSection* cs, const double energy,
                                 int* cs_index);

void generate_random_numbers(const uint64_t pkey, const uint64_t master_key,
                             const uint64_t counter, double* rn0, double* rn1);
