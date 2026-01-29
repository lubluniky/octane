/**
 * @file gaussian_neon.h
 * @brief High-performance Gaussian sampling with ARM NEON for Apple M4
 *
 * This header provides vectorized Gaussian sampling using the Box-Muller
 * transform optimized for ARM NEON SIMD instructions on Apple Silicon.
 *
 * Features:
 * - Vectorized xoroshiro128+ RNG (4 parallel streams)
 * - SIMD Box-Muller transform
 * - Fast math approximations (rsqrt, log, sin, cos)
 * - Reparameterization trick support: output = mean + std * noise
 *
 * @copyright 2024 RocketRL Project
 * @license GPL-2.0
 */

#ifndef GAUSSIAN_NEON_H
#define GAUSSIAN_NEON_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RNG state size: 4 parallel xoroshiro128+ states = 4 * 2 * uint64_t = 8 uint64_t */
#define RNG_STATE_SIZE 8

/* Alignment for NEON operations */
#define NEON_ALIGNMENT 16

/**
 * @brief Initialize the parallel RNG state from a single seed
 *
 * Creates 4 independent xoroshiro128+ states using a SplitMix64 seed expansion.
 * Each stream will produce independent sequences suitable for parallel sampling.
 *
 * @param seed Initial seed value
 * @param state Output array of size RNG_STATE_SIZE (8 uint64_t values)
 *              Must be aligned to 16 bytes for optimal performance
 */
void init_rng_state(uint64_t seed, uint64_t* state);

/**
 * @brief Sample from standard normal distribution N(0,1) using NEON
 *
 * Generates samples using vectorized Box-Muller transform with:
 * - 4 parallel xoroshiro128+ RNG streams
 * - Fast polynomial approximations for log/sqrt
 * - Branchless NEON operations
 *
 * @param output Output array for samples (should be 16-byte aligned for best performance)
 * @param count Number of samples to generate
 * @param rng_state RNG state array (modified in place)
 *
 * @note For best performance, count should be a multiple of 8
 */
void sample_standard_normal_neon(float* output, size_t count, uint64_t* rng_state);

/**
 * @brief Sample from Gaussian distribution with reparameterization
 *
 * Computes: output[i] = mean[i] + std[i] * N(0,1)
 *
 * This implements the reparameterization trick used in VAEs and policy
 * gradient methods, allowing gradients to flow through the sampling operation.
 *
 * @param mean Array of means (size: count)
 * @param std Array of standard deviations (size: count, must be positive)
 * @param output Output array for samples (size: count)
 * @param count Number of samples (batch_size * action_dim)
 * @param rng_state RNG state array (modified in place)
 *
 * @note All arrays should be 16-byte aligned for optimal NEON performance
 */
void sample_gaussian_neon(const float* mean, const float* std,
                          float* output, size_t count, uint64_t* rng_state);

/**
 * @brief Batch sample for RL action selection
 *
 * Optimized version for batched action sampling in reinforcement learning.
 * Generates samples for (batch_size x action_dim) configuration.
 *
 * @param mean Mean array of shape [batch_size, action_dim] (row-major)
 * @param std Std array of shape [batch_size, action_dim] (row-major)
 * @param output Output array of shape [batch_size, action_dim]
 * @param batch_size Number of parallel environments/samples
 * @param action_dim Dimension of action space
 * @param rng_state RNG state array
 */
void sample_gaussian_batch_neon(const float* mean, const float* std,
                                float* output, size_t batch_size,
                                size_t action_dim, uint64_t* rng_state);

/**
 * @brief Sample and compute log probability simultaneously
 *
 * Efficiently computes both samples and their log probabilities:
 * - sample[i] = mean[i] + std[i] * z, where z ~ N(0,1)
 * - log_prob[i] = -0.5 * (log(2*pi) + 2*log(std[i]) + z^2)
 *
 * This is useful for policy gradient methods where both the action
 * and its log probability are needed.
 *
 * @param mean Mean array
 * @param std Standard deviation array
 * @param output Sampled values output
 * @param log_prob Log probability output
 * @param count Number of samples
 * @param rng_state RNG state
 */
void sample_gaussian_with_logprob_neon(const float* mean, const float* std,
                                       float* output, float* log_prob,
                                       size_t count, uint64_t* rng_state);

#ifdef __cplusplus
}
#endif

#endif /* GAUSSIAN_NEON_H */
