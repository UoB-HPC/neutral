#pragma once 

#include "Random123/threefry.h"
#include "bright_data.h"

#define NRANDOM_NUMBERS        2            // Precomputed random nums
#define BUF_LENGTH             1            // Precomputed random nums

typedef struct {
  int buf_len;
  double* buffer;
  int available;                    // The number of available random numbers

  threefry2x64_ctr_t counter;
  threefry2x64_key_t key;

} RNPool;

// Initialises the random number pool
void init_rn_pool(RNPool* rn_pool, const uint64_t master_key, 
    const int nrandom_numbers);

// Prepare the random number pool
void prepare_rn_pool(
    RNPool* rn_pool, const uint64_t key, const int nrandom_numbers);

// Generates a random number used the Random 123 library
double genrand(RNPool* rn_pool);

// Fills the rn buffer with random numbers
void fill_rn_buffer(
    RNPool* rn_pool, const int nrandom_numbers);

