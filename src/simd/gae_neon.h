/**
 * @file gae_neon.h
 * @brief ARM NEON optimized Generalized Advantage Estimation for Apple M4
 */

#ifndef GAE_NEON_H
#define GAE_NEON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Compute GAE (Generalized Advantage Estimation) using NEON SIMD
 *
 * @param rewards      Rewards array [buffer_size, num_envs]
 * @param values       Value estimates [buffer_size, num_envs]
 * @param dones        Done flags [buffer_size, num_envs] (1.0 = done)
 * @param advantages   Output advantages [buffer_size, num_envs]
 * @param buffer_size  Number of timesteps
 * @param num_envs     Number of parallel environments
 * @param gamma        Discount factor (typically 0.99)
 * @param gae_lambda   GAE lambda (typically 0.95)
 * @param last_values  Bootstrap values for last step [num_envs]
 */
void gae_compute_neon(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    size_t buffer_size,
    size_t num_envs,
    float gamma,
    float gae_lambda,
    const float* last_values
);

/**
 * Compute GAE and returns (value targets) in one pass
 */
void gae_compute_with_returns_neon(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    float* returns,
    size_t buffer_size,
    size_t num_envs,
    float gamma,
    float gae_lambda,
    const float* last_values
);

/**
 * Normalize advantages in-place: (adv - mean) / (std + eps)
 */
void gae_normalize_neon(
    float* advantages,
    size_t count,
    float eps
);

/**
 * Compute GAE and normalize in one call
 */
void gae_compute_normalized_neon(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    size_t buffer_size,
    size_t num_envs,
    float gamma,
    float gae_lambda,
    const float* last_values,
    float eps
);

#ifdef __cplusplus
}
#endif

#endif /* GAE_NEON_H */
