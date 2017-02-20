#include <stdint.h>
#include <stdio.h>
#include "bright_data.h"
#include "rand.h"

// Initialise the RNPool for a thread
void init_rn_pool(RNPool* rn_pool, const int key)
{
  rn_pool->counter.v[0] = 0;
  rn_pool->counter.v[1] = 0;
  rn_pool->key.v[0] = key;
  rn_pool->key.v[1] = 0;
  fill_rn_buffer(rn_pool);
}

// Fills the rn buffer with random numbers
void fill_rn_buffer(RNPool* rn_pool)
{
  // Generate the random numbers
  for(int ii = 0; ii < NRANDOM_NUMBERS; ii += 2) {
    threefry2x64_ctr_t rand = threefry2x64(rn_pool->counter, rn_pool->key);

    // Turn our random numbers from integrals to double precision
    const double factor = 1.0/(UINT64_MAX + 1.0);
    const double half_factor = 0.5*factor;
    rn_pool->buffer[ii] = rand.v[0]*factor+half_factor;
    rn_pool->buffer[ii+1] = rand.v[1]*factor+half_factor;

    // Carry the increment across both variables
    if(rn_pool->counter.v[0] == UINT64_MAX) {
      printf("OK. We actually used over %lld random numbers.\n", UINT64_MAX);
      rn_pool->counter.v[1]++;
      rn_pool->counter.v[0] = 0;
    }
    else {
      rn_pool->counter.v[0]++;
    }
  }

  rn_pool->available = NRANDOM_NUMBERS;
}

// Generates a random number used the Random 123 library
double genrand(RNPool* rn_pool)
{
  if(!rn_pool->available) {
    fill_rn_buffer(rn_pool);
  }

  return rn_pool->buffer[--rn_pool->available];
}

