//! SIMD-accelerated Generalized Advantage Estimation (GAE) computation.
//!
//! This module provides high-performance GAE computation optimized for multiple
//! architectures:
//!
//! - **ARM NEON**: Apple Silicon (M1/M2/M3) using native Rust intrinsics
//! - **x86_64 AVX2**: Intel/AMD processors with AVX2+FMA support
//!
//! The key optimization is inverting the loop order:
//! - Outer loop: iterate time `t` backwards (sequential dependency on GAE accumulator)
//! - Inner loop: process environments in SIMD chunks (parallel, no data dependency)
//!
//! This layout enables vectorization across environments while maintaining the
//! sequential time dependency required by the GAE recurrence relation.
//!
//! # GAE Equation
//!
//! ```text
//! delta[t] = reward[t] + gamma * V(s[t+1]) * (1 - done[t]) - V(s[t])
//! gae[t] = delta[t] + gamma * lambda * (1 - done[t]) * gae[t+1]
//! returns[t] = gae[t] + V(s[t])
//! ```
//!
//! # Performance
//!
//! With 64+ environments, this implementation achieves:
//! - ~4x speedup on ARM NEON (4-wide vectors)
//! - ~8x speedup on AVX2 (8-wide vectors)
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::simd::gae::compute_gae_simd;
//!
//! let num_steps = 2048;
//! let num_envs = 64;
//! let rewards = vec![1.0f32; num_steps * num_envs];
//! let values = vec![0.5f32; num_steps * num_envs];
//! let terminated = vec![0.0f32; num_steps * num_envs];
//! let truncated = vec![0.0f32; num_steps * num_envs];
//! let last_values = vec![0.5f32; num_envs];
//!
//! let (advantages, returns) = compute_gae_simd(
//!     &rewards, &values, &terminated, &truncated,
//!     num_steps, num_envs,
//!     0.99, 0.95,
//!     &last_values,
//! )?;
//! ```

#![allow(unsafe_code)]

use super::{Result, SimdError};

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
use std::arch::aarch64::*;

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
use std::arch::x86_64::*;

// ============================================================================
// Public API
// ============================================================================

/// Compute GAE using the best available SIMD instruction set.
///
/// Automatically selects:
/// - ARM NEON on aarch64 with `simd` feature
/// - AVX2 on x86_64 with `avx2` feature
/// - Scalar fallback otherwise
///
/// # Arguments
///
/// * `rewards` - Rewards array [num_steps, num_envs] in row-major order
/// * `values` - Value estimates [num_steps, num_envs]
/// * `terminated` - Terminal flags (1.0 if episode truly ended, 0.0 otherwise) [num_steps, num_envs]
/// * `truncated` - Truncation flags (1.0 if time-limit cutoff, 0.0 otherwise) [num_steps, num_envs]
/// * `num_steps` - Number of time steps in the rollout
/// * `num_envs` - Number of parallel environments
/// * `gamma` - Discount factor (typically 0.99)
/// * `gae_lambda` - GAE lambda parameter (typically 0.95)
/// * `last_values` - Bootstrap values for the state after the last step [num_envs]
///
/// # Returns
///
/// Tuple of (advantages, returns) each with shape [num_steps, num_envs]
///
/// # Data Layout
///
/// Input arrays should be in row-major order: `data[step * num_envs + env]`
/// This matches the typical vectorized environment output where all envs
/// step together, producing a batch of results per timestep.
pub fn compute_gae_simd(
    rewards: &[f32],
    values: &[f32],
    terminated: &[f32],
    truncated: &[f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    // Validate input sizes
    let expected_len = num_steps * num_envs;
    if rewards.len() != expected_len {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: rewards.len(),
        });
    }
    if values.len() != expected_len {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: values.len(),
        });
    }
    if terminated.len() != expected_len {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: terminated.len(),
        });
    }
    if truncated.len() != expected_len {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: truncated.len(),
        });
    }
    if last_values.len() != num_envs {
        return Err(SimdError::SizeMismatch {
            expected: num_envs,
            actual: last_values.len(),
        });
    }

    let mut advantages = vec![0.0f32; expected_len];
    let mut returns = vec![0.0f32; expected_len];

    // Select the best available implementation
    compute_gae_impl(
        rewards,
        values,
        terminated,
        truncated,
        &mut advantages,
        &mut returns,
        num_steps,
        num_envs,
        gamma,
        gae_lambda,
        last_values,
    );

    Ok((advantages, returns))
}

/// Compute GAE with in-place output buffers.
///
/// This variant allows reusing pre-allocated buffers to avoid allocation overhead
/// in hot paths.
///
/// # Safety
///
/// The `advantages` and `returns` buffers must have length `num_steps * num_envs`.
pub fn compute_gae_simd_inplace(
    rewards: &[f32],
    values: &[f32],
    terminated: &[f32],
    truncated: &[f32],
    advantages: &mut [f32],
    returns: &mut [f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) -> Result<()> {
    let expected_len = num_steps * num_envs;

    // Validate all buffer sizes
    if rewards.len() != expected_len
        || values.len() != expected_len
        || terminated.len() != expected_len
        || truncated.len() != expected_len
        || advantages.len() != expected_len
        || returns.len() != expected_len
    {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: rewards
                .len()
                .min(values.len())
                .min(terminated.len())
                .min(truncated.len())
                .min(advantages.len())
                .min(returns.len()),
        });
    }
    if last_values.len() != num_envs {
        return Err(SimdError::SizeMismatch {
            expected: num_envs,
            actual: last_values.len(),
        });
    }

    // Select the best available implementation
    compute_gae_impl(
        rewards,
        values,
        terminated,
        truncated,
        advantages,
        returns,
        num_steps,
        num_envs,
        gamma,
        gae_lambda,
        last_values,
    );

    Ok(())
}

// ============================================================================
// ARM NEON Implementation (4-wide f32 vectors)
// ============================================================================

/// NEON-optimized GAE computation.
///
/// Processes 4 environments at a time using 128-bit NEON registers.
/// Uses FMA (vfmaq_f32) for optimal performance on Apple Silicon.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn compute_gae_neon(
    rewards: &[f32],
    values: &[f32],
    terminated: &[f32],
    truncated: &[f32],
    advantages: &mut [f32],
    returns: &mut [f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) {
    let gamma_vec = vdupq_n_f32(gamma);
    let _gae_lambda_vec = vdupq_n_f32(gae_lambda); // Used if needed for future optimizations
    let one_vec = vdupq_n_f32(1.0);
    let gamma_lambda = vdupq_n_f32(gamma * gae_lambda);

    // Number of complete 4-env chunks
    let chunks = num_envs / 4;

    // Process environments in chunks of 4
    for chunk in 0..chunks {
        let env_offset = chunk * 4;

        // Initialize GAE accumulator and next value from last_values
        let mut last_gae = vdupq_n_f32(0.0);
        let mut next_value = vld1q_f32(last_values.as_ptr().add(env_offset));

        // Backward pass through time (outer loop - sequential)
        for step in (0..num_steps).rev() {
            let idx = step * num_envs + env_offset;

            // Load data for 4 environments (contiguous in memory)
            let reward = vld1q_f32(rewards.as_ptr().add(idx));
            let value = vld1q_f32(values.as_ptr().add(idx));
            let terminated_mask = vld1q_f32(terminated.as_ptr().add(idx));
            let truncated_mask = vld1q_f32(truncated.as_ptr().add(idx));

            // bootstrap_mask = 1 - terminated
            let bootstrap_mask = vsubq_f32(one_vec, terminated_mask);
            // trace_mask = 1 - (terminated OR truncated)
            let trace_mask = vsubq_f32(
                one_vec,
                vminq_f32(vaddq_f32(terminated_mask, truncated_mask), one_vec),
            );

            // delta = reward + gamma * next_value * bootstrap_mask - value
            // Using FMA: delta = reward + gamma * (next_value * mask) - value
            let next_masked = vmulq_f32(next_value, bootstrap_mask);
            let gamma_next = vmulq_f32(gamma_vec, next_masked);
            let delta = vsubq_f32(vaddq_f32(reward, gamma_next), value);

            // last_gae = delta + gamma * lambda * trace_mask * last_gae
            // Using FMA: last_gae = vfmaq_f32(delta, gamma_lambda * mask, last_gae)
            let gae_decay = vmulq_f32(gamma_lambda, trace_mask);
            last_gae = vfmaq_f32(delta, gae_decay, last_gae);

            // Store advantages
            vst1q_f32(advantages.as_mut_ptr().add(idx), last_gae);

            // returns = advantages + values
            let ret = vaddq_f32(last_gae, value);
            vst1q_f32(returns.as_mut_ptr().add(idx), ret);

            // Update next_value for next iteration
            next_value = value;
        }
    }

    // Handle remaining environments with scalar code
    let remainder_start = chunks * 4;
    if remainder_start < num_envs {
        compute_gae_scalar_range(
            rewards,
            values,
            terminated,
            truncated,
            advantages,
            returns,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            last_values,
            remainder_start,
            num_envs,
        );
    }
}

// ============================================================================
// x86_64 AVX2 Implementation (8-wide f32 vectors)
// ============================================================================

/// AVX2-optimized GAE computation.
///
/// Processes 8 environments at a time using 256-bit AVX2 registers.
/// Uses FMA (_mm256_fmadd_ps) for optimal performance.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_gae_avx2(
    rewards: &[f32],
    values: &[f32],
    terminated: &[f32],
    truncated: &[f32],
    advantages: &mut [f32],
    returns: &mut [f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) {
    let gamma_vec = _mm256_set1_ps(gamma);
    let one_vec = _mm256_set1_ps(1.0);
    let gamma_lambda = _mm256_set1_ps(gamma * gae_lambda);

    // Number of complete 8-env chunks
    let chunks = num_envs / 8;

    // Process environments in chunks of 8
    for chunk in 0..chunks {
        let env_offset = chunk * 8;

        // Initialize GAE accumulator and next value from last_values
        let mut last_gae = _mm256_setzero_ps();
        let mut next_value = _mm256_loadu_ps(last_values.as_ptr().add(env_offset));

        // Backward pass through time (outer loop - sequential)
        for step in (0..num_steps).rev() {
            let idx = step * num_envs + env_offset;

            // Load data for 8 environments (contiguous in memory)
            let reward = _mm256_loadu_ps(rewards.as_ptr().add(idx));
            let value = _mm256_loadu_ps(values.as_ptr().add(idx));
            let terminated_mask = _mm256_loadu_ps(terminated.as_ptr().add(idx));
            let truncated_mask = _mm256_loadu_ps(truncated.as_ptr().add(idx));

            // bootstrap_mask = 1 - terminated
            let bootstrap_mask = _mm256_sub_ps(one_vec, terminated_mask);
            // trace_mask = 1 - (terminated OR truncated)
            let trace_mask = _mm256_sub_ps(
                one_vec,
                _mm256_min_ps(_mm256_add_ps(terminated_mask, truncated_mask), one_vec),
            );

            // delta = reward + gamma * next_value * bootstrap_mask - value
            let next_masked = _mm256_mul_ps(next_value, bootstrap_mask);
            let gamma_next = _mm256_mul_ps(gamma_vec, next_masked);
            let delta = _mm256_sub_ps(_mm256_add_ps(reward, gamma_next), value);

            // last_gae = delta + gamma * lambda * trace_mask * last_gae
            // Using FMA: last_gae = fmadd(gae_decay, last_gae, delta)
            let gae_decay = _mm256_mul_ps(gamma_lambda, trace_mask);
            last_gae = _mm256_fmadd_ps(gae_decay, last_gae, delta);

            // Store advantages
            _mm256_storeu_ps(advantages.as_mut_ptr().add(idx), last_gae);

            // returns = advantages + values
            let ret = _mm256_add_ps(last_gae, value);
            _mm256_storeu_ps(returns.as_mut_ptr().add(idx), ret);

            // Update next_value for next iteration
            next_value = value;
        }
    }

    // Handle remaining environments with scalar code
    let remainder_start = chunks * 8;
    if remainder_start < num_envs {
        compute_gae_scalar_range(
            rewards,
            values,
            terminated,
            truncated,
            advantages,
            returns,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            last_values,
            remainder_start,
            num_envs,
        );
    }
}

// ============================================================================
// Implementation Dispatcher
// ============================================================================

/// Internal dispatcher that selects the best implementation.
/// This avoids the unreachable code warnings from conditional returns.
#[inline]
fn compute_gae_impl(
    rewards: &[f32],
    values: &[f32],
    terminated: &[f32],
    truncated: &[f32],
    advantages: &mut [f32],
    returns: &mut [f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) {
    // NEON implementation for aarch64
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            compute_gae_neon(
                rewards,
                values,
                terminated,
                truncated,
                advantages,
                returns,
                num_steps,
                num_envs,
                gamma,
                gae_lambda,
                last_values,
            );
        }
    }

    // AVX2 implementation for x86_64
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                compute_gae_avx2(
                    rewards,
                    values,
                    terminated,
                    truncated,
                    advantages,
                    returns,
                    num_steps,
                    num_envs,
                    gamma,
                    gae_lambda,
                    last_values,
                );
            }
            return;
        }
    }

    // Scalar fallback for all other cases
    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(
            target_arch = "x86_64",
            target_feature = "avx2",
            target_feature = "fma"
        )
    )))]
    {
        compute_gae_scalar(
            rewards,
            values,
            terminated,
            truncated,
            advantages,
            returns,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            last_values,
        );
    }
}

// ============================================================================
// Scalar Fallback Implementation
// ============================================================================

/// Scalar GAE computation (fallback for non-SIMD paths).
///
/// Still uses the optimized loop order (time-outer, env-inner) for better
/// cache utilization compared to the naive env-outer implementation.
#[inline]
#[allow(dead_code)] // Used in non-SIMD builds
fn compute_gae_scalar(
    rewards: &[f32],
    values: &[f32],
    terminated: &[f32],
    truncated: &[f32],
    advantages: &mut [f32],
    returns: &mut [f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) {
    compute_gae_scalar_range(
        rewards,
        values,
        terminated,
        truncated,
        advantages,
        returns,
        num_steps,
        num_envs,
        gamma,
        gae_lambda,
        last_values,
        0,
        num_envs,
    );
}

/// Scalar GAE for a range of environments [env_start, env_end).
///
/// Used for:
/// 1. Complete scalar fallback when SIMD is unavailable
/// 2. Handling remainder environments after SIMD chunks
#[inline]
fn compute_gae_scalar_range(
    rewards: &[f32],
    values: &[f32],
    terminated: &[f32],
    truncated: &[f32],
    advantages: &mut [f32],
    returns: &mut [f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
    env_start: usize,
    env_end: usize,
) {
    let gamma_lambda = gamma * gae_lambda;

    // Process each environment independently
    for env in env_start..env_end {
        let mut last_gae = 0.0f32;
        let mut next_value = last_values[env];

        // Backward pass through time
        for step in (0..num_steps).rev() {
            let idx = step * num_envs + env;

            let reward = rewards[idx];
            let value = values[idx];
            let terminated_mask = terminated[idx];
            let truncated_mask = truncated[idx];
            let bootstrap_mask = 1.0 - terminated_mask;
            let trace_mask = 1.0 - (terminated_mask + truncated_mask).min(1.0);

            // delta = reward + gamma * next_value * bootstrap_mask - value
            let delta = reward + gamma * next_value * bootstrap_mask - value;

            // last_gae = delta + gamma * lambda * trace_mask * last_gae
            last_gae = delta + gamma_lambda * trace_mask * last_gae;

            advantages[idx] = last_gae;
            returns[idx] = last_gae + value;

            next_value = value;
        }
    }
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Normalize advantages in-place using SIMD.
///
/// Computes: `advantages = (advantages - mean) / (std + epsilon)`
///
/// This is commonly used before PPO updates to stabilize training.
pub fn normalize_advantages_simd(advantages: &mut [f32], epsilon: f32) {
    if advantages.is_empty() {
        return;
    }

    let n = advantages.len();
    let n_f = n as f32;

    // Compute mean
    let sum: f32 = advantages.iter().sum();
    let mean = sum / n_f;

    // Compute variance
    let var_sum: f32 = advantages.iter().map(|&x| (x - mean).powi(2)).sum();
    let std = (var_sum / n_f + epsilon).sqrt();

    // Dispatch to the best available implementation
    normalize_advantages_impl(advantages, mean, std);
}

/// Internal dispatcher for advantage normalization.
#[inline]
fn normalize_advantages_impl(advantages: &mut [f32], mean: f32, std: f32) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        unsafe {
            normalize_advantages_neon(advantages, mean, std);
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                normalize_advantages_avx2(advantages, mean, std);
            }
            return;
        }
    }

    #[cfg(not(any(
        all(target_arch = "aarch64", target_feature = "neon"),
        all(target_arch = "x86_64", target_feature = "avx2")
    )))]
    // Scalar fallback
    {
        let inv_std = 1.0 / std;
        for x in advantages.iter_mut() {
            *x = (*x - mean) * inv_std;
        }
    }
}

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn normalize_advantages_neon(advantages: &mut [f32], mean: f32, std: f32) {
    let mean_vec = vdupq_n_f32(mean);
    let inv_std_vec = vdupq_n_f32(1.0 / std);

    let chunks = advantages.len() / 4;

    for i in 0..chunks {
        let idx = i * 4;
        let v = vld1q_f32(advantages.as_ptr().add(idx));
        let centered = vsubq_f32(v, mean_vec);
        let normalized = vmulq_f32(centered, inv_std_vec);
        vst1q_f32(advantages.as_mut_ptr().add(idx), normalized);
    }

    // Handle remainder
    let inv_std = 1.0 / std;
    for i in (chunks * 4)..advantages.len() {
        advantages[i] = (advantages[i] - mean) * inv_std;
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn normalize_advantages_avx2(advantages: &mut [f32], mean: f32, std: f32) {
    let mean_vec = _mm256_set1_ps(mean);
    let inv_std_vec = _mm256_set1_ps(1.0 / std);

    let chunks = advantages.len() / 8;

    for i in 0..chunks {
        let idx = i * 8;
        let v = _mm256_loadu_ps(advantages.as_ptr().add(idx));
        let centered = _mm256_sub_ps(v, mean_vec);
        let normalized = _mm256_mul_ps(centered, inv_std_vec);
        _mm256_storeu_ps(advantages.as_mut_ptr().add(idx), normalized);
    }

    // Handle remainder
    let inv_std = 1.0 / std;
    for i in (chunks * 8)..advantages.len() {
        advantages[i] = (advantages[i] - mean) * inv_std;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation for testing.
    fn compute_gae_reference(
        rewards: &[f32],
        values: &[f32],
        terminated: &[f32],
        truncated: &[f32],
        num_steps: usize,
        num_envs: usize,
        gamma: f32,
        gae_lambda: f32,
        last_values: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let mut advantages = vec![0.0f32; num_steps * num_envs];
        let mut returns = vec![0.0f32; num_steps * num_envs];

        for env in 0..num_envs {
            let mut last_gae = 0.0f32;
            let mut next_value = last_values[env];

            for step in (0..num_steps).rev() {
                let idx = step * num_envs + env;
                let bootstrap_mask = 1.0 - terminated[idx];
                let trace_mask = 1.0 - (terminated[idx] + truncated[idx]).min(1.0);
                let delta = rewards[idx] + gamma * next_value * bootstrap_mask - values[idx];
                last_gae = delta + gamma * gae_lambda * trace_mask * last_gae;

                advantages[idx] = last_gae;
                returns[idx] = last_gae + values[idx];
                next_value = values[idx];
            }
        }

        (advantages, returns)
    }

    #[test]
    fn test_gae_basic() {
        let num_steps = 128;
        let num_envs = 4;
        let rewards = vec![1.0f32; num_steps * num_envs];
        let values = vec![0.5f32; num_steps * num_envs];
        let terminated = vec![0.0f32; num_steps * num_envs];
        let truncated = vec![0.0f32; num_steps * num_envs];
        let last_values = vec![0.5f32; num_envs];
        let gamma = 0.99;
        let gae_lambda = 0.95;

        let (adv_simd, ret_simd) = compute_gae_simd(
            &rewards,
            &values,
            &terminated,
            &truncated,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            &last_values,
        )
        .unwrap();

        let (adv_ref, ret_ref) = compute_gae_reference(
            &rewards,
            &values,
            &terminated,
            &truncated,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            &last_values,
        );

        // Compare results
        for i in 0..(num_steps * num_envs) {
            assert!(
                (adv_simd[i] - adv_ref[i]).abs() < 1e-5,
                "Advantage mismatch at {}: {} vs {}",
                i,
                adv_simd[i],
                adv_ref[i]
            );
            assert!(
                (ret_simd[i] - ret_ref[i]).abs() < 1e-5,
                "Return mismatch at {}: {} vs {}",
                i,
                ret_simd[i],
                ret_ref[i]
            );
        }
    }

    #[test]
    fn test_gae_with_dones() {
        let num_steps = 16;
        let num_envs = 8;
        let rewards = vec![1.0f32; num_steps * num_envs];
        let values = vec![0.5f32; num_steps * num_envs];
        let terminated = vec![0.0f32; num_steps * num_envs];
        let mut truncated = vec![0.0f32; num_steps * num_envs];

        // Set some episodes as truncated
        truncated[5 * num_envs] = 1.0; // Env 0 time-limit cutoff at step 5
        truncated[10 * num_envs + 3] = 1.0; // Env 3 time-limit cutoff at step 10

        let last_values = vec![0.5f32; num_envs];
        let gamma = 0.99;
        let gae_lambda = 0.95;

        let (adv_simd, ret_simd) = compute_gae_simd(
            &rewards,
            &values,
            &terminated,
            &truncated,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            &last_values,
        )
        .unwrap();

        let (adv_ref, ret_ref) = compute_gae_reference(
            &rewards,
            &values,
            &terminated,
            &truncated,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            &last_values,
        );

        for i in 0..(num_steps * num_envs) {
            assert!(
                (adv_simd[i] - adv_ref[i]).abs() < 1e-5,
                "Advantage mismatch at {}: {} vs {}",
                i,
                adv_simd[i],
                adv_ref[i]
            );
            assert!(
                (ret_simd[i] - ret_ref[i]).abs() < 1e-5,
                "Return mismatch at {}: {} vs {}",
                i,
                ret_simd[i],
                ret_ref[i]
            );
        }
    }

    #[test]
    fn test_gae_non_aligned_envs() {
        // Test with num_envs not divisible by SIMD width
        let num_steps = 64;
        let num_envs = 13; // Not divisible by 4 or 8
        let rewards = vec![1.0f32; num_steps * num_envs];
        let values = vec![0.5f32; num_steps * num_envs];
        let terminated = vec![0.0f32; num_steps * num_envs];
        let truncated = vec![0.0f32; num_steps * num_envs];
        let last_values = vec![0.5f32; num_envs];
        let gamma = 0.99;
        let gae_lambda = 0.95;

        let (adv_simd, ret_simd) = compute_gae_simd(
            &rewards,
            &values,
            &terminated,
            &truncated,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            &last_values,
        )
        .unwrap();

        let (adv_ref, ret_ref) = compute_gae_reference(
            &rewards,
            &values,
            &terminated,
            &truncated,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            &last_values,
        );

        for i in 0..(num_steps * num_envs) {
            assert!(
                (adv_simd[i] - adv_ref[i]).abs() < 1e-5,
                "Advantage mismatch at {}: {} vs {}",
                i,
                adv_simd[i],
                adv_ref[i]
            );
            assert!(
                (ret_simd[i] - ret_ref[i]).abs() < 1e-5,
                "Return mismatch at {}: {} vs {}",
                i,
                ret_simd[i],
                ret_ref[i]
            );
        }
    }

    #[test]
    fn test_gae_varying_values() {
        let num_steps = 32;
        let num_envs = 16;

        // Create varying rewards and values
        let rewards: Vec<f32> = (0..num_steps * num_envs)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let values: Vec<f32> = (0..num_steps * num_envs)
            .map(|i| (i as f32 * 0.05).cos() * 0.5)
            .collect();
        let terminated = vec![0.0f32; num_steps * num_envs];
        let truncated = vec![0.0f32; num_steps * num_envs];
        let last_values: Vec<f32> = (0..num_envs).map(|i| i as f32 * 0.1).collect();
        let gamma = 0.99;
        let gae_lambda = 0.95;

        let (adv_simd, ret_simd) = compute_gae_simd(
            &rewards,
            &values,
            &terminated,
            &truncated,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            &last_values,
        )
        .unwrap();

        let (adv_ref, ret_ref) = compute_gae_reference(
            &rewards,
            &values,
            &terminated,
            &truncated,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            &last_values,
        );

        for i in 0..(num_steps * num_envs) {
            assert!(
                (adv_simd[i] - adv_ref[i]).abs() < 1e-4,
                "Advantage mismatch at {}: {} vs {}",
                i,
                adv_simd[i],
                adv_ref[i]
            );
            assert!(
                (ret_simd[i] - ret_ref[i]).abs() < 1e-4,
                "Return mismatch at {}: {} vs {}",
                i,
                ret_simd[i],
                ret_ref[i]
            );
        }
    }

    #[test]
    fn test_normalize_advantages() {
        let mut advantages = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        normalize_advantages_simd(&mut advantages, 1e-8);

        // Check mean is approximately 0
        let mean: f32 = advantages.iter().sum::<f32>() / advantages.len() as f32;
        assert!(mean.abs() < 1e-5, "Mean {} should be ~0", mean);

        // Check std is approximately 1
        let var: f32 = advantages.iter().map(|x| x * x).sum::<f32>() / advantages.len() as f32;
        let std = var.sqrt();
        assert!((std - 1.0).abs() < 1e-5, "Std {} should be ~1", std);
    }

    #[test]
    fn test_size_validation() {
        let result = compute_gae_simd(
            &[1.0; 10], &[0.5; 10], &[0.0; 10], &[0.0; 10], 5, 2, // 5 * 2 = 10
            0.99, 0.95, &[0.5; 3], // Wrong size!
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_gae_large_scale() {
        // Test with realistic PPO parameters
        let num_steps = 2048;
        let num_envs = 64;
        let rewards = vec![1.0f32; num_steps * num_envs];
        let values = vec![0.5f32; num_steps * num_envs];
        let terminated = vec![0.0f32; num_steps * num_envs];
        let truncated = vec![0.0f32; num_steps * num_envs];
        let last_values = vec![0.5f32; num_envs];

        let (advantages, returns) = compute_gae_simd(
            &rewards,
            &values,
            &terminated,
            &truncated,
            num_steps,
            num_envs,
            0.99,
            0.95,
            &last_values,
        )
        .unwrap();

        assert_eq!(advantages.len(), num_steps * num_envs);
        assert_eq!(returns.len(), num_steps * num_envs);

        // All advantages should be positive (constant positive reward)
        for &adv in &advantages {
            assert!(adv > 0.0);
        }
    }

    #[test]
    fn test_gae_bootstraps_truncated_steps() {
        let rewards = vec![1.0f32];
        let values = vec![0.5f32];
        let terminated = vec![0.0f32];
        let truncated = vec![1.0f32];
        let last_values = vec![0.75f32];

        let (advantages, returns) = compute_gae_simd(
            &rewards,
            &values,
            &terminated,
            &truncated,
            1,
            1,
            0.99,
            0.95,
            &last_values,
        )
        .unwrap();

        assert!(
            (advantages[0] - 1.2425).abs() < 1e-4,
            "unexpected advantage: {advantages:?}"
        );
        assert!(
            (returns[0] - 1.7425).abs() < 1e-4,
            "unexpected return: {returns:?}"
        );
    }
}
