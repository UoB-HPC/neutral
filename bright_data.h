#pragma once 

#include "../mesh.h"
#include "../comms.h"

#define eV                     1.60217646e-19       // eV in C
#define INITIAL_ENERGY         1.0e+5               // random test eV
#define MIN_ENERGY_OF_INTEREST 1.0e-5               // Artificially low value
#define PARTICLE_MASS          1.674927471213e-27   // Mass taken from wiki
#define MASS_NO                100.0                
#define OPEN_BOUND_CORRECTION  1.0e-13
#define BARNS                  1.0e-28              // The barns unit in m^2
#define AVOGADROS              6.02214085774e23     // Avogadro's constant
#define MOLAR_MASS             2.0e-2               // Dummy kg per mole
#define CS_SCATTER_FILENAME    "elastic_scatter.cs"
#define CS_CAPTURE_FILENAME    "capture.cs"
#define TAG_SEND_RECV          100
#define TAG_PARTICLE           1
#define NPARTICLES             100000

#ifdef MPI
#include "mpi.h"
#endif

typedef enum { CENSUS = 0, COLLISION = 1, FACET_CROSS = 2 } MostRecentEvent;

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
  double weight;             // weight of the particle (Nphys/Nsim)
  double dt_to_census;       // the time until census is reached
  double mfp_to_collision; // the mean free paths until a collision

  int tracer;
  int cell;
  int dead;

} Particle;

#ifdef MPI
  // Global MPI particle type
  MPI_Datatype particle_type;
#endif

typedef struct {
  CrossSection* cs_scatter_table;
  CrossSection* cs_absorb_table;
  Particle* local_particles;
  Particle* out_particles;

  int nparticles;
  int nlocal_particles;

  double* energy_tally;

} BrightData;

// Initialises all of the Bright-specific data structures.
void initialise_bright_data(
    BrightData* bright_data, Mesh* mesh);

// Acts as a particle source
void inject_particles(
    Mesh* mesh, const int local_nx, const int local_ny, 
    const int local_particle_left_off, const int local_particle_bottom_off,
    const int local_particle_nx, const int local_particle_ny, 
    const int nparticles, Particle* particles);

