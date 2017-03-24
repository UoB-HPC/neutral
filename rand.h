#pragma once 

#include "Random123/threefry.h"
#include "neutral_data.h"

#define NRANDOM_NUMBERS        2            // Precomputed random nums
#define BUF_LENGTH             1            // Precomputed random nums

typedef struct {
  int buf_len;          // The local length of the random number buffer
  double* buffer;       // A pointer to the local part of the global buffer
  int available;        // The number of available local random numbers
  int master;

  threefry2x64_ctr_t counter;
  threefry2x64_key_t key;

} RNPool;


#ifdef __cplusplus
extern "C" {
#endif

  // Prepare the random number pool
  void init_rn_pools(
      RNPool* rn_pools, const int nrn_pools, const int buf_len);

  // Updates the master key of the set of rn pools
  void update_rn_pool_master_keys(
      RNPool* rn_pools, const int nrn_pools, uint64_t master_key);

  // Generates a random number used the Random 123 library
  double getrand(RNPool* rn_pool);

  // Fills the rn buffer with random numbers
  void fill_rn_buffer(
      RNPool* rn_pool, const int nrandom_numbers);

#ifdef __cplusplus
}
#endif

