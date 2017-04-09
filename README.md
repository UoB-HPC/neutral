# neutral
A Monte Carlo Neutron Transport Mini-App

# Purpose

This application is a simplified Monte Carlo neutral particle transport mini-app that support a number of physical processes as particle histories are tracked.

# Build

The build process is intended to be simple, and has been tested on a number of platforms.

```
make KERNELS=omp3 COMPILER=INTEL
```

A number of other switches and options are provided:

- `DEBUG=<yes/no>` - 'yes' switches off optimisation and adds debug flags
- `MPI=<yes/no>` - 'yes' turns off any use of MPI within the application.
- The `OPTIONS` makefile variable is used to allow visit dumps, with `-DVISIT_DUMP`, and profiling, with `-DENABLE_PROFILING`.

Please note: We do not support granular profiling with the over particles parallelisation scheme because it has a negative impact on the performance of the application and gives spurious results.

# Configuration Files

The configuration files expose a number of key parameters for the application.

- `iterations` - the number of iterations the application will proceed through
- `dt` - the timestep for the application
- `nx` - the number of cells in the x-dimension
- `ny` - the number of cells in the y-dimension
- `initial_energy` - the initial energy that all particles will be set to

The performance of the Monte Carlo application is highly problem dependent, and so we provide multiple configuration files that present different computation problems:

- `problems/scatter` - the particles will mostly collide within the slab
- `problems/stream` - the particles stream across the mesh multiple times per timestep, colliding infrequently
- `problems/csp` - the particles stream until they encounter a region of high density in the center of the slab
- `problems/split` - the particles are spread evenly in regions of high and low density to match the number of events of each type

TODO: Describe the `problem` and `source` descriptions in the parameter file.

# Development Status

The implementation is currently in an active development phase. There are multiple branches that are exploring algorithmic changes and other optimisations in order to test the performance of the application on modern architectures.

- `master` - the main branch of the code, where parallelisation is over particles.
- `event-based` - adjusts the parallelisation strategy so that events are handled for all particles in a synchronous manner.
- `hybrid` - supports event-based parallelisation with progress through successive events.
- `master-soa` - changes the master branch to use an SoA data structure.
- `tiled` - attempts to tile the over particles parallelisation strategy to improve cache locality.

The mini-app currently supports elastic scattering, with realistic cross sections. We intend to extend the application to include particle production.

Random123 is a highly usable counter-based random number generator that we use for random number generation, https://www.deshawresearch.com/resources_random123.html.

