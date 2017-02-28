#pragma once 

#include "rand.h"
#include "../mesh.h"
#include "../comms.h"

#ifdef MPI
#include "mpi.h"
#endif

/* Problem-Independent Constants */
#define eV_TO_J                1.60217646e-19       // 1 eV to Joules
#define AVOGADROS              6.02214085774e23     // Avogadro's constant
#define BARNS                  1.0e-28              // The barns unit in m^2
#define PARTICLE_MASS          1.674927471213e-27   // Mass taken from wiki
#define MASS_NO                1.0e2                // Mass num of the particle
#define MOLAR_MASS             1.0e-2               // Dummy kg per mole
#define MIN_ENERGY_OF_INTEREST 1.0e0                // Energy to kill particles
#define OPEN_BOUND_CORRECTION  1.0e-13              // Fixes open bounds
#define TAG_SEND_RECV          100
#define TAG_PARTICLE           1
#define VALIDATE_TOLERANCE     1.0e-3

/* Data tables */
#define CS_SCATTER_FILENAME    "elastic_scatter.cs" // Elastic scattering cs file
#define CS_CAPTURE_FILENAME    "capture.cs"         // Capture cs file

#define ARCH_ROOT_PARAMS "../arch.params"
#define NEUTRAL_TESTS    "neutral.tests"

// Represents a cross sectional table for resonance data
typedef struct {
  double* key;
  double* value;
  int nentries;

} CrossSection;

// Represents an individual particle
typedef struct {
  double x;                  // x position in space
  double y;                  // y position in space
  double omega_x;            // x direction
  double omega_y;            // y direction
  double e;                  // energy
  double weight;             // weight of the particle
  double dt_to_census;       // the time until census is reached
  double mfp_to_collision;   // the mean free paths until a collision
  double distance_to_facet;  // the distance until a facet is encountered
  uint64_t key;
  int cellx;
  int celly;
  int dead;

} Particle;

enum { COLLISION, FACET, CENSUS };

// Contains the configuration and state data for the application
typedef struct {
  CrossSection* cs_scatter_table;
  CrossSection* cs_absorb_table;
  Particle* local_particles;

  double initial_energy;

  int nthreads;
  int nparticles;
  int nlocal_particles;

  double* scalar_flux_tally;
  double* energy_deposition_tally;

  const char* neutral_params_filename;

} BrightData;

#ifdef MPI
// Global MPI particle type
MPI_Datatype particle_type;
#endif

// Initialises all of the Bright-specific data structures.
void initialise_bright_data(
    BrightData* bright_data, Mesh* mesh, RNPool* rn_pool);

// Acts as a particle source
void inject_particles(
    Mesh* mesh, const int local_nx, const int local_ny, 
    const double local_particle_left_off, const double local_particle_bottom_off,
    const double local_particle_width, const double local_particle_height, 
    const int nparticles, const double initial_energy, RNPool* rn_pool,
    Particle* particles);

