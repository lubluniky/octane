/**
 * buffer_ops_neon.h - High-performance batch gathering for ReplayBuffer
 *
 * Optimized for Apple M4 (ARM NEON) with:
 * - Software prefetching for reduced cache misses
 * - NEON SIMD for vectorized memory operations
 * - Optional index sorting for improved cache locality
 * - Loop unrolling for pipeline efficiency
 *
 * Part of RocketRL - High-performance Reinforcement Learning Library
 *
 * Copyright (C) 2024 RocketRL Authors
 * SPDX-License-Identifier: GPL-2.0
 */

#ifndef BUFFER_OPS_NEON_H
#define BUFFER_OPS_NEON_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Configuration flags for gather operations.
 */
typedef enum {
    GATHER_FLAG_NONE         = 0,
    GATHER_FLAG_SORT_INDICES = 1 << 0,  /* Sort indices for better cache locality */
    GATHER_FLAG_PREFETCH     = 1 << 1,  /* Enable software prefetching */
    GATHER_FLAG_DEFAULT      = GATHER_FLAG_PREFETCH  /* Default: prefetch enabled */
} GatherFlags;

/**
 * Statistics from gather operations (for benchmarking).
 */
typedef struct {
    uint64_t cycles;          /* CPU cycles (if available) */
    uint64_t cache_misses;    /* Estimated cache misses */
    size_t   bytes_gathered;  /* Total bytes processed */
    double   bandwidth_gbps;  /* Effective bandwidth in GB/s */
} GatherStats;

/**
 * Batch gather from a 2D float array.
 *
 * Gathers rows from src into dst using random access indices.
 * Optimized with NEON SIMD and software prefetching.
 *
 * @param src        Source buffer, row-major [capacity, dim]
 * @param indices    Random indices to gather [batch_size]
 * @param dst        Destination buffer [batch_size, dim]
 * @param batch_size Number of rows to gather
 * @param dim        Dimension of each row
 * @param capacity   Total capacity of source buffer (for bounds checking)
 *
 * Memory layout:
 *   src[i][j] = src[i * dim + j]
 *   dst[k][j] = src[indices[k] * dim + j]
 */
void gather_batch_f32(
    const float*  src,
    const size_t* indices,
    float*        dst,
    size_t        batch_size,
    size_t        dim,
    size_t        capacity
);

/**
 * Batch gather with configurable flags.
 *
 * @param src        Source buffer [capacity, dim]
 * @param indices    Indices to gather [batch_size]
 * @param dst        Destination buffer [batch_size, dim]
 * @param batch_size Number of rows to gather
 * @param dim        Dimension of each row
 * @param capacity   Total capacity
 * @param flags      Configuration flags (GatherFlags)
 * @param stats      Optional output statistics (can be NULL)
 */
void gather_batch_f32_ex(
    const float*  src,
    const size_t* indices,
    float*        dst,
    size_t        batch_size,
    size_t        dim,
    size_t        capacity,
    uint32_t      flags,
    GatherStats*  stats
);

/**
 * Strided batch gather for Structure-of-Arrays (SoA) ReplayBuffer.
 *
 * Gathers from multiple parallel arrays simultaneously, maximizing
 * cache efficiency by processing all arrays for each index together.
 *
 * This matches the ReplayBuffer storage layout:
 *   - observations:      [capacity * obs_dim]
 *   - actions:           [capacity * action_dim]
 *   - rewards:           [capacity]
 *   - next_observations: [capacity * obs_dim]
 *   - dones:             [capacity]
 *
 * @param obs            Source observations [capacity * obs_dim]
 * @param actions        Source actions [capacity * action_dim]
 * @param rewards        Source rewards [capacity]
 * @param next_obs       Source next observations [capacity * obs_dim]
 * @param dones          Source done flags [capacity]
 * @param indices        Indices to gather [batch_size]
 * @param obs_batch      Output observations [batch_size * obs_dim]
 * @param actions_batch  Output actions [batch_size * action_dim]
 * @param rewards_batch  Output rewards [batch_size]
 * @param next_obs_batch Output next observations [batch_size * obs_dim]
 * @param dones_batch    Output done flags [batch_size]
 * @param batch_size     Number of transitions to gather
 * @param obs_dim        Observation dimension
 * @param action_dim     Action dimension
 * @param capacity       Buffer capacity
 */
void gather_batch_strided(
    const float*  obs,
    const float*  actions,
    const float*  rewards,
    const float*  next_obs,
    const float*  dones,
    const size_t* indices,
    float*        obs_batch,
    float*        actions_batch,
    float*        rewards_batch,
    float*        next_obs_batch,
    float*        dones_batch,
    size_t        batch_size,
    size_t        obs_dim,
    size_t        action_dim,
    size_t        capacity
);

/**
 * Extended strided gather with configuration.
 *
 * @param flags  Configuration flags (GatherFlags)
 * @param stats  Optional output statistics
 */
void gather_batch_strided_ex(
    const float*  obs,
    const float*  actions,
    const float*  rewards,
    const float*  next_obs,
    const float*  dones,
    const size_t* indices,
    float*        obs_batch,
    float*        actions_batch,
    float*        rewards_batch,
    float*        next_obs_batch,
    float*        dones_batch,
    size_t        batch_size,
    size_t        obs_dim,
    size_t        action_dim,
    size_t        capacity,
    uint32_t      flags,
    GatherStats*  stats
);

/**
 * Scatter priorities for Prioritized Experience Replay (PER) updates.
 *
 * Updates priorities at random indices. Optimized for random write patterns.
 *
 * @param priorities      Priority buffer [capacity]
 * @param indices         Indices to update [batch_size]
 * @param new_priorities  New priority values [batch_size]
 * @param batch_size      Number of priorities to update
 */
void scatter_priorities_f32(
    float*        priorities,
    const size_t* indices,
    const float*  new_priorities,
    size_t        batch_size
);

/**
 * Scatter with optional index sorting for cache efficiency.
 *
 * @param flags  Configuration flags
 */
void scatter_priorities_f32_ex(
    float*        priorities,
    const size_t* indices,
    const float*  new_priorities,
    size_t        batch_size,
    uint32_t      flags
);

/**
 * Benchmark utilities.
 */

/**
 * Run gather benchmark and return performance statistics.
 *
 * @param dim        Test dimension
 * @param batch_size Test batch size
 * @param capacity   Test capacity
 * @param iterations Number of iterations
 * @param stats      Output statistics
 */
void benchmark_gather_f32(
    size_t        dim,
    size_t        batch_size,
    size_t        capacity,
    size_t        iterations,
    GatherStats*  stats
);

/**
 * Run strided gather benchmark.
 */
void benchmark_gather_strided(
    size_t        obs_dim,
    size_t        action_dim,
    size_t        batch_size,
    size_t        capacity,
    size_t        iterations,
    GatherStats*  stats
);

/**
 * Get NEON availability status.
 *
 * @return 1 if NEON is available, 0 otherwise
 */
int neon_available(void);

/**
 * Get implementation version string.
 */
const char* buffer_ops_version(void);

#ifdef __cplusplus
}
#endif

#endif /* BUFFER_OPS_NEON_H */
