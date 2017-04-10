#pragma once 

#include "Random123/threefry.h"
#include "neutral_data.h"

#define NRANDOM_NUMBERS        2            // Precomputed random nums
#define BUF_LENGTH             1            // Precomputed random nums

typedef struct {
  double buffer[BUF_LENGTH*NRANDOM_NUMBERS];   // The pool of random numbers
  int available;                    // The number of available random numbers

  threefry2x64_ctr_t counter;
  threefry2x64_key_t key;

} RNPool;

#pragma omp declare target

// Initialises the random number pool
void init_rn_pool(RNPool* rn_pool, const uint64_t master_key);

// Prepare the random number pool
void prepare_rn_pool(RNPool* rn_pool, const uint64_t key);

// Generates a random number used the Random 123 library
double genrand(RNPool* rn_pool);


// Fills the rn buffer with random numbers
void fill_rn_buffer(RNPool* rn_pool);

// Increment the counter, handling carry as necessary
void safe_increment(uint64_t* v);

// Random123 methods
threefry2x64_ctr_t threefry2x64_R(
    unsigned int Nrounds, threefry2x64_ctr_t counter, threefry2x64_key_t key);
uint64_t RotL_64(
    uint64_t x, unsigned int N);

#pragma omp end declare target

