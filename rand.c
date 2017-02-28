#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "bright_data.h"
#include "rand.h"

// Initialises the random number pool
void init_rn_pool(RNPool* rn_pool, const uint64_t master_key)
{
  // The master key is necessary to stop the particles seeing the same RN stream
  // every time the pool is reset
  rn_pool->key.v[0] = 0;
  rn_pool->key.v[1] = master_key;
  rn_pool->counter.v[0] = 0;
  rn_pool->counter.v[1] = 0;
  rn_pool->available = 0;
  rn_pool->buf_len = 0;
}

// Prepare the random number pool
void prepare_rn_pool(
    RNPool* rn_pool, const uint64_t key, const int nrandom_numbers)
{
  rn_pool->counter.v[0] = 0;
  rn_pool->counter.v[1] = 0;
  rn_pool->key.v[0] = key;

  if(rn_pool->buf_len < nrandom_numbers) {
    printf("reallocing rn pool\n");
    // Reallocate the random number space to be larger
    free(rn_pool->buffer);
    rn_pool->buf_len = max(rn_pool->buf_len, nrandom_numbers);
    rn_pool->buffer = (double*)malloc(sizeof(double)*1.5*nrandom_numbers);
  }

  fill_rn_buffer(rn_pool, nrandom_numbers);
}

// Generates a random number used the Random 123 library
double genrand(RNPool* rn_pool)
{
  // If we have run out of space, just generate the minimum required number 
  // of random numbers
  if(!rn_pool->available) {
    // TODO: WOULD BE NICE TO COUNT THE NUMBER OF TIMES THAT THIS ACTUALLY HAPPENS
    fill_rn_buffer(rn_pool, NRANDOM_NUMBERS);

    // TODO: CHECK FOR OVERFLOW CAUSED BY THE INCREMENT
    if(rn_pool->counter.v[0]+NRANDOM_NUMBERS >= UINT64_MAX) {
      TERMINATE("Overran the allowed space for the counter, our logic doesn't \
          permit carry yet.\n");
    }
    else {
      rn_pool->counter.v[0] += NRANDOM_NUMBERS;
    }
  }

  return rn_pool->buffer[--rn_pool->available];
}

// Fills the rn buffer with random numbers
void fill_rn_buffer(
    RNPool* rn_pool, const int nrandom_numbers)
{
  assert(nrandom_numbers%NRANDOM_NUMBERS == 0);

  uint64_t counter_start = rn_pool->counter.v[0];

#pragma omp parallel for
  for(int ii = 0; ii < nrandom_numbers/NRANDOM_NUMBERS; ++ii) {
    threefry2x64_ctr_t counter;
    counter.v[0] = counter_start+ii;

    // Generate the random numbers
    threefry2x64_ctr_t rand = threefry2x64(counter, rn_pool->key);

    // Turn our random numbers from integrals to double precision
    const double factor = 1.0/(UINT64_MAX + 1.0);
    const double half_factor = 0.5*factor;
    rn_pool->buffer[ii*NRANDOM_NUMBERS] = rand.v[0]*factor+half_factor;
    rn_pool->buffer[ii*NRANDOM_NUMBERS+1] = rand.v[1]*factor+half_factor;
  }

  rn_pool->available = nrandom_numbers;
}

