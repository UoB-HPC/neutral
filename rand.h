#pragma once

#include "Random123/threefry.h"
#include "neutral_data.h"

#define NRANDOM_NUMBERS 4 // Precomputed random nums

#define MAX_UINT64 (uint64_t)UINT64_C(0xFFFFFFFFFFFFFFFF)
#define FACTOR (1.0 / ((double)MAX_UINT64 + 1.0))
#define HALF_FACTOR (0.5 * FACTOR)

