#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "../neutral_data.h"
#include "../rand.h"

// Generates NRANDOM_NUMBERS and places them in the buffer
void generate_random_numbers(
    RNPool* rn_pool, int index, uint64_t counter_start);

// Prepare the random number pool
void init_rn_pools(
    RNPool* rn_pools, const int nrn_pools, const int buf_len)
{
  // Initialise all pools to some sensible default state
  for(int ii = 0; ii < nrn_pools; ++ii) {
    rn_pools[ii].key.v[0] = 0;
    rn_pools[ii].key.v[1] = ii;
    rn_pools[ii].counter.v[0] = 0;
    rn_pools[ii].counter.v[1] = 0;
    rn_pools[ii].available = 0;
    rn_pools[ii].buf_len = 0;
    rn_pools[ii].master = 0;
    rn_pools[ii].buffer = NULL;
  }

  // Setup the master pool to have access to the whole random number buffer
  const int nnon_master_pools = (nrn_pools-1);
  const int master_pool_index = nnon_master_pools;

  allocate_data(&rn_pools[master_pool_index].buffer, buf_len);
  rn_pools[master_pool_index].buf_len = buf_len;
  rn_pools[master_pool_index].master = 1;

  // Give each of the non master random number pools an allocation in the
  // global space
  const int nlocal_slots = buf_len/nnon_master_pools;
  for(int ii = 0; ii < nnon_master_pools; ++ii) {
    rn_pools[ii].buffer = &rn_pools[master_pool_index].buffer[ii*nlocal_slots];
    rn_pools[ii].buf_len = nlocal_slots;
  }
}

// Updates the master key of the set of rn pools
void update_rn_pool_master_keys(
    RNPool* rn_pools, const int nrn_pools, uint64_t master_key)
{
  for(int ii = 0; ii < nrn_pools; ++ii) {
    rn_pools[ii].key.v[0] = master_key;
  }
}

// Generates a random number used the Random 123 library
double getrand(RNPool* rn_pool)
{
  // If we have run out of space, just generate the minimum required number 
  // of random numbers
  if(!rn_pool->available) {
    fill_rn_buffer(rn_pool, rn_pool->buf_len);

    if(rn_pool->counter.v[0]+rn_pool->buf_len >= UINT64_MAX) {
      TERMINATE("Overran the allowed space for the counter, our logic doesn't \
          permit carry yet.\n");
    }
    else {
      rn_pool->counter.v[0] += rn_pool->buf_len;
    }
  }

  return rn_pool->buffer[--rn_pool->available];
}

// Fills the rn buffer with random numbers
void fill_rn_buffer(
    RNPool* rn_pool, const int nrandom_numbers)
{
  assert(nrandom_numbers <= rn_pool->buf_len);

  // The master pool can initialise random numbers in parallel
  uint64_t counter_start = rn_pool->counter.v[0];
  if(rn_pool->master) {
#pragma omp parallel for
    for(int ii = 0; ii < nrandom_numbers/NRANDOM_NUMBERS; ++ii) {
      generate_random_numbers(rn_pool, ii, counter_start);
    }
  }
  else {
    for(int ii = 0; ii < nrandom_numbers/NRANDOM_NUMBERS; ++ii) {
      generate_random_numbers(rn_pool, ii, counter_start);
    }
  }

  rn_pool->available = nrandom_numbers;
}

// Generates NRANDOM_NUMBERS and places them in the buffer
void generate_random_numbers(
    RNPool* rn_pool, int index, uint64_t counter_start)
{
  threefry2x64_ctr_t counter;
  counter.v[0] = counter_start+index;

  // Generate the random numbers
  threefry2x64_ctr_t rand = threefry2x64(counter, rn_pool->key);

  // Turn our random numbers from integrals to double precision
  const double factor = 1.0/(UINT64_MAX + 1.0);
  const double half_factor = 0.5*factor;
  rn_pool->buffer[index*NRANDOM_NUMBERS] = rand.v[0]*factor+half_factor;
  rn_pool->buffer[index*NRANDOM_NUMBERS+1] = rand.v[1]*factor+half_factor;
}

