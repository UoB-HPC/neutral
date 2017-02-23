#include <stdint.h>
#include <stdio.h>
#include "bright_data.h"
#include "rand.h"

// Initialises the random number pool
void init_rn_pool(RNPool* rn_pool)
{
  rn_pool->key.v[0] = 0;
  rn_pool->key.v[1] = 0;
  rn_pool->counter.v[0] = 0;
  rn_pool->counter.v[1] = 0;
  rn_pool->loop_init = 0;
}

// Prepare the random number pool
void prepare_rn_pool(RNPool* rn_pool, const uint64_t counter_start)
{
  rn_pool->counter.v[0] = counter_start;
  rn_pool->counter.v[1] = 0;
  rn_pool->loop_init = 1;
  safe_increment(rn_pool->key.v);
  fill_rn_buffer(rn_pool);
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

    safe_increment(rn_pool->counter.v);
  }

  rn_pool->available = NRANDOM_NUMBERS;
}

// Generates a random number used the Random 123 library
double genrand(RNPool* rn_pool, )
{
  if(!rn_pool->available) {
    fill_rn_buffer(rn_pool);
  }

  return rn_pool->buffer[--rn_pool->available];
}

