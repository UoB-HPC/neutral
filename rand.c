#include <stdint.h>
#include <stdio.h>
#include "neutral_data.h"
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
}

// Prepare the random number pool
void prepare_rn_pool(RNPool* rn_pool, const uint64_t key)
{
  rn_pool->counter.v[0] = 0;
  rn_pool->counter.v[1] = 0;
  rn_pool->key.v[0] = key;
  fill_rn_buffer(rn_pool);
}

// Generates a random number used the Random 123 library
double genrand(RNPool* rn_pool)
{
  if(!rn_pool->available) {
    fill_rn_buffer(rn_pool);
  }
  return rn_pool->buffer[--rn_pool->available];
}

// Fills the rn buffer with random numbers
void fill_rn_buffer(RNPool* rn_pool)
{
  for(int ii = 0; ii < BUF_LENGTH; ++ii) {
    // Generate the random numbers
    threefry2x64_ctr_t rand = threefry2x64(rn_pool->counter, rn_pool->key);

    // Turn our random numbers from integrals to double precision
    const double factor = 1.0/(UINT64_MAX + 1.0);
    const double half_factor = 0.5*factor;
    rn_pool->buffer[0] = rand.v[0]*factor+half_factor;
    rn_pool->buffer[1] = rand.v[1]*factor+half_factor;
    safe_increment(rn_pool->counter.v);
  }

  rn_pool->available = BUF_LENGTH*NRANDOM_NUMBERS;
}

// Increment the counter, handling carry as necessary
void safe_increment(uint64_t* v)
{
  if(v[0] == UINT64_MAX) {
    printf("OK. We actually spilled our counter or key!\n");
    v[1]++;
    v[0] = 0;
  }
  else {
    v[0]++;
  }
}

