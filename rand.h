#pragma once 

#include "Random123/threefry.h"
#include "bright_data.h"

#define NRANDOM_NUMBERS        ((128)*(4))            // Precomputed random nums

typedef struct {
  double buffer[NRANDOM_NUMBERS];   // The pool of random numbers
  int available;                    // The number of available random numbers

  threefry2x64_ctr_t counter;
  threefry2x64_key_t key;

} RNPool;

// Initialise the RNPool for a thread
void init_rn_pool(RNPool* rn_pool, const int key);

// Fills the rn buffer with random numbers
void fill_rn_buffer(RNPool* rn_pool);

// Generates a random number used the Random 123 library
double genrand(RNPool* rn_pool);

