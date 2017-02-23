#pragma once 

#include "Random123/threefry.h"
#include "bright_data.h"

#define NRANDOM_NUMBERS        2            // Precomputed random nums

typedef struct {
  double buffer[NRANDOM_NUMBERS];   // The pool of random numbers
  int available;                    // The number of available random numbers

  threefry2x64_ctr_t counter;
  threefry2x64_key_t key;

  int loop_init;

} RNPool;

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

