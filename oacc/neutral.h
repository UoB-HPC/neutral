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

// Handles a collision event
int collision_event(
    const int global_nx, const int nx, const int x_off, const int y_off,
    const uint64_t master_key, const double inv_ntotal_particles,
    const double distance_to_collision, const double local_density,
    const double* cs_absorb_table_keys, const double* cs_scatter_table_keys,
    const double* cs_absorb_table_values, const double* cs_scatter_table_values,
    const int cs_absorb_table_nentries, const int cs_scatter_table_nentries,
    const uint64_t pp, double* p_x, double* p_y, int* p_cellx, int* p_celly,
    double* p_weight, double* p_energy, int* p_dead, double* p_omega_x,
    double* p_omega_y, double* p_dt_to_census, double* p_mfp_to_collision,
    uint64_t* counter, double* energy_deposition, double* number_density,
    double* microscopic_cs_scatter, double* microscopic_cs_absorb,
    double* macroscopic_cs_scatter, double* macroscopic_cs_absorb,
    double* energy_deposition_tally, double rn[NRANDOM_NUMBERS], double* speed);

// Handle facet event
int facet_event(const int global_nx, const int global_ny, const int nx,
                const int ny, const int x_off, const int y_off,
                const double inv_ntotal_particles,
                const double distance_to_facet, const double speed,
                const double cell_mfp, const int x_facet, const double* density,
                const int* neighbours, const uint64_t pp, double* p_energy,
                double* p_weight, double* p_mfp_to_collision,
                double* p_dt_to_census, double* p_x, double* p_y,
                double* p_omega_x, double* p_omega_y, int* p_cellx,
                int* p_celly, double* energy_deposition, double* number_density,
                double* microscopic_cs_scatter, double* microscopic_cs_absorb,
                double* macroscopic_cs_scatter, double* macroscopic_cs_absorb,
                double* energy_deposition_tally, int* cellx, int* celly,
                double* local_density);

// Handles the census event
void census_event(const int global_nx, const int nx, const int x_off,
                  const int y_off, const double inv_ntotal_particles,
                  const double distance_to_census, const double cell_mfp,
                  const uint64_t pp, double* p_weight, double* p_energy,
                  double* p_x, double* p_y, double* p_omega_x,
                  double* p_omega_y, double* p_mfp_to_collision,
                  double* p_dt_to_census, int* p_cellx, int* p_celly,
                  double* energy_deposition, double* number_density,
                  double* microscopic_cs_scatter, double* microscopic_cs_absorb,
                  double* energy_deposition_tally);

// Tallies the energy deposition in the cell
void update_tallies(const int nx, const int x_off, const int y_off,
                    const int p_cellx, const int p_celly,
                    const double inv_ntotal_particles,
                    const double energy_deposition,
                    double* energy_deposition_tally);

// Sends a particle to a neighbour and replaces in the particle list
void send_and_mark_particle(const int destination, Particle* particle);

// Calculate the distance to the next facet
void calc_distance_to_facet(const int global_nx, const double p_x,
                            const double p_y, const int pad, const int x_off,
                            const int y_off, const double p_omega_x,
                            const double p_omega_y, const double speed,
                            const int particle_cellx, const int particle_celly,
                            double* distance_to_facet, int* x_facet,
                            const double* edgex, const double* edgey);

// Calculate the energy deposition in the cell
double calculate_energy_deposition(const int global_nx, const int nx,
                                   const int x_off, const int y_off,
                                   const double p_energy, const double p_weight,
                                   const double inv_ntotal_particles,
                                   const double path_length,
                                   const double number_density,
                                   const double microscopic_cs_absorb,
                                   const double microscopic_cs_total);

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy_binary(double* keys, double* values,
                                        const int nentries, const double energy);

// Fetch the cross section for a particular energy value
double microscopic_cs_for_energy_linear(double* keys, double* values,
                                        const int nentries, const double energy);

// Validates the results of the simulation
void validate(const int nx, const int ny, const char* params_filename,
              const int rank, double* energy_deposition_tally);

double generate_random_numbers(const uint64_t pkey, const uint64_t master_key,
                               const uint64_t counter);
