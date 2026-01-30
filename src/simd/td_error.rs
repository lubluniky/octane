//! SIMD-accelerated TD-error computation for off-policy algorithms.
//!
//! This module provides vectorized temporal difference (TD) error computation
//! for off-policy reinforcement learning algorithms including SAC, TD3, DDPG, and DQN.
//!
//! TD-error is the core signal for value function learning:
//! `td_error = reward + gamma * (1 - done) * next_value - current_value`
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::simd::td_error::compute_td_errors_batch;
//!
//! let rewards = vec![1.0f32; 256];
//! let next_values = vec![0.5f32; 256];
//! let dones = vec![0.0f32; 256];
//! let current_values = vec![0.3f32; 256];
//! let gamma = 0.99;
//!
//! let td_errors = compute_td_errors_batch(
//!     &rewards, &next_values, &dones, &current_values, gamma
//! )?;
//! ```

#![allow(unsafe_code)]

use super::{Result, SimdError};

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
use std::arch::aarch64::*;

// ============================================================================
// TD-Error Computation
// ============================================================================

/// Compute TD-errors for a batch of transitions.
///
/// The TD-error formula is:
/// `td_error = reward + gamma * (1 - done) * next_value - current_value`
///
/// # Arguments
///
/// * `rewards` - Immediate rewards [batch_size]
/// * `next_values` - Value estimates for next states [batch_size]
/// * `dones` - Episode termination flags (1.0 if done, 0.0 otherwise) [batch_size]
/// * `current_values` - Value estimates for current states [batch_size]
/// * `gamma` - Discount factor (typically 0.99)
///
/// # Returns
///
/// Vector of TD-errors [batch_size]
pub fn compute_td_errors_batch(
    rewards: &[f32],
    next_values: &[f32],
    dones: &[f32],
    current_values: &[f32],
    gamma: f32,
) -> Result<Vec<f32>> {
    let batch_size = rewards.len();

    // Validate input sizes
    if next_values.len() != batch_size {
        return Err(SimdError::SizeMismatch {
            expected: batch_size,
            actual: next_values.len(),
        });
    }
    if dones.len() != batch_size {
        return Err(SimdError::SizeMismatch {
            expected: batch_size,
            actual: dones.len(),
        });
    }
    if current_values.len() != batch_size {
        return Err(SimdError::SizeMismatch {
            expected: batch_size,
            actual: current_values.len(),
        });
    }

    let mut td_errors = vec![0.0f32; batch_size];

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        if super::x86::is_avx2_available() {
            unsafe {
                compute_td_errors_avx2(
                    rewards,
                    next_values,
                    dones,
                    current_values,
                    &mut td_errors,
                    gamma,
                );
            }
            return Ok(td_errors);
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        if super::is_neon_available() {
            unsafe {
                compute_td_errors_neon(
                    rewards,
                    next_values,
                    dones,
                    current_values,
                    &mut td_errors,
                    gamma,
                );
            }
            return Ok(td_errors);
        }
    }

    // Scalar fallback
    compute_td_errors_scalar(
        rewards,
        next_values,
        dones,
        current_values,
        &mut td_errors,
        gamma,
    );

    Ok(td_errors)
}

/// Compute TD-errors with target network (for double Q-learning).
///
/// Uses separate target network values for stability:
/// `td_error = reward + gamma * (1 - done) * target_next_value - current_value`
///
/// # Arguments
///
/// * `rewards` - Immediate rewards [batch_size]
/// * `target_next_values` - Target network value estimates for next states [batch_size]
/// * `dones` - Episode termination flags [batch_size]
/// * `current_values` - Current network value estimates [batch_size]
/// * `gamma` - Discount factor
///
/// # Returns
///
/// Vector of TD-errors [batch_size]
pub fn compute_td_errors_with_target(
    rewards: &[f32],
    target_next_values: &[f32],
    dones: &[f32],
    current_values: &[f32],
    gamma: f32,
) -> Result<Vec<f32>> {
    // Same implementation, just different semantic meaning
    compute_td_errors_batch(rewards, target_next_values, dones, current_values, gamma)
}

/// Compute TD-errors for SAC with entropy bonus.
///
/// SAC uses a modified TD target that includes entropy:
/// `td_error = reward + gamma * (1 - done) * (next_value - alpha * log_prob) - current_value`
///
/// # Arguments
///
/// * `rewards` - Immediate rewards [batch_size]
/// * `next_values` - Q-value estimates for next states [batch_size]
/// * `next_log_probs` - Log probabilities of next actions [batch_size]
/// * `dones` - Episode termination flags [batch_size]
/// * `current_values` - Q-value estimates for current states [batch_size]
/// * `gamma` - Discount factor
/// * `alpha` - Temperature parameter for entropy regularization
///
/// # Returns
///
/// Vector of TD-errors [batch_size]
pub fn compute_sac_td_errors(
    rewards: &[f32],
    next_values: &[f32],
    next_log_probs: &[f32],
    dones: &[f32],
    current_values: &[f32],
    gamma: f32,
    alpha: f32,
) -> Result<Vec<f32>> {
    let batch_size = rewards.len();

    // Validate input sizes
    if next_values.len() != batch_size
        || next_log_probs.len() != batch_size
        || dones.len() != batch_size
        || current_values.len() != batch_size
    {
        return Err(SimdError::SizeMismatch {
            expected: batch_size,
            actual: next_values
                .len()
                .min(next_log_probs.len())
                .min(dones.len())
                .min(current_values.len()),
        });
    }

    let mut td_errors = vec![0.0f32; batch_size];

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        if super::x86::is_avx2_available() {
            unsafe {
                compute_sac_td_errors_avx2(
                    rewards,
                    next_values,
                    next_log_probs,
                    dones,
                    current_values,
                    &mut td_errors,
                    gamma,
                    alpha,
                );
            }
            return Ok(td_errors);
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        if super::is_neon_available() {
            unsafe {
                compute_sac_td_errors_neon(
                    rewards,
                    next_values,
                    next_log_probs,
                    dones,
                    current_values,
                    &mut td_errors,
                    gamma,
                    alpha,
                );
            }
            return Ok(td_errors);
        }
    }

    // Scalar fallback
    for i in 0..batch_size {
        let mask = 1.0 - dones[i];
        let target = next_values[i] - alpha * next_log_probs[i];
        td_errors[i] = rewards[i] + gamma * mask * target - current_values[i];
    }

    Ok(td_errors)
}

/// Compute TD-errors for TD3 with clipped double Q-learning.
///
/// TD3 uses the minimum of two Q-networks to prevent overestimation:
/// `target_q = reward + gamma * (1 - done) * min(q1_next, q2_next)`
/// `td_error = target_q - current_value`
///
/// # Arguments
///
/// * `rewards` - Immediate rewards [batch_size]
/// * `q1_next` - First Q-network values for next states [batch_size]
/// * `q2_next` - Second Q-network values for next states [batch_size]
/// * `dones` - Episode termination flags [batch_size]
/// * `current_values` - Current Q-values [batch_size]
/// * `gamma` - Discount factor
///
/// # Returns
///
/// Vector of TD-errors [batch_size]
pub fn compute_td3_td_errors(
    rewards: &[f32],
    q1_next: &[f32],
    q2_next: &[f32],
    dones: &[f32],
    current_values: &[f32],
    gamma: f32,
) -> Result<Vec<f32>> {
    let batch_size = rewards.len();

    if q1_next.len() != batch_size
        || q2_next.len() != batch_size
        || dones.len() != batch_size
        || current_values.len() != batch_size
    {
        return Err(SimdError::SizeMismatch {
            expected: batch_size,
            actual: q1_next
                .len()
                .min(q2_next.len())
                .min(dones.len())
                .min(current_values.len()),
        });
    }

    let mut td_errors = vec![0.0f32; batch_size];

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    {
        if super::x86::is_avx2_available() {
            unsafe {
                compute_td3_td_errors_avx2(
                    rewards,
                    q1_next,
                    q2_next,
                    dones,
                    current_values,
                    &mut td_errors,
                    gamma,
                );
            }
            return Ok(td_errors);
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        if super::is_neon_available() {
            unsafe {
                compute_td3_td_errors_neon(
                    rewards,
                    q1_next,
                    q2_next,
                    dones,
                    current_values,
                    &mut td_errors,
                    gamma,
                );
            }
            return Ok(td_errors);
        }
    }

    // Scalar fallback
    for i in 0..batch_size {
        let mask = 1.0 - dones[i];
        let min_q = q1_next[i].min(q2_next[i]);
        let target = rewards[i] + gamma * mask * min_q;
        td_errors[i] = target - current_values[i];
    }

    Ok(td_errors)
}

/// Compute absolute TD-errors for prioritized experience replay.
///
/// Returns |td_error| + epsilon for numerical stability.
///
/// # Arguments
///
/// * `td_errors` - TD-errors computed from any method
/// * `epsilon` - Small constant to prevent zero priorities (typically 1e-6)
///
/// # Returns
///
/// Absolute TD-errors suitable for PER priorities
pub fn compute_priorities(td_errors: &[f32], epsilon: f32) -> Vec<f32> {
    let batch_size = td_errors.len();
    let mut priorities = vec![0.0f32; batch_size];

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if super::x86::is_avx2_available() {
            unsafe {
                compute_priorities_avx2(td_errors, &mut priorities, epsilon);
            }
            return priorities;
        }
    }

    // Scalar fallback
    for i in 0..batch_size {
        priorities[i] = td_errors[i].abs() + epsilon;
    }

    priorities
}

// ============================================================================
// AVX2 Implementations
// ============================================================================

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_td_errors_avx2(
    rewards: &[f32],
    next_values: &[f32],
    dones: &[f32],
    current_values: &[f32],
    td_errors: &mut [f32],
    gamma: f32,
) {
    let batch_size = rewards.len();
    let chunks = batch_size / 8;

    let gamma_vec = _mm256_set1_ps(gamma);
    let one_vec = _mm256_set1_ps(1.0);

    for i in 0..chunks {
        let idx = i * 8;

        let r = _mm256_loadu_ps(rewards.as_ptr().add(idx));
        let nv = _mm256_loadu_ps(next_values.as_ptr().add(idx));
        let d = _mm256_loadu_ps(dones.as_ptr().add(idx));
        let cv = _mm256_loadu_ps(current_values.as_ptr().add(idx));

        // mask = 1 - done
        let mask = _mm256_sub_ps(one_vec, d);

        // target = reward + gamma * mask * next_value
        let gamma_nv = _mm256_mul_ps(gamma_vec, nv);
        let masked = _mm256_mul_ps(gamma_nv, mask);
        let target = _mm256_add_ps(r, masked);

        // td_error = target - current_value
        let error = _mm256_sub_ps(target, cv);

        _mm256_storeu_ps(td_errors.as_mut_ptr().add(idx), error);
    }

    // Handle remainder
    for i in (chunks * 8)..batch_size {
        let mask = 1.0 - dones[i];
        td_errors[i] = rewards[i] + gamma * mask * next_values[i] - current_values[i];
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_sac_td_errors_avx2(
    rewards: &[f32],
    next_values: &[f32],
    next_log_probs: &[f32],
    dones: &[f32],
    current_values: &[f32],
    td_errors: &mut [f32],
    gamma: f32,
    alpha: f32,
) {
    let batch_size = rewards.len();
    let chunks = batch_size / 8;

    let gamma_vec = _mm256_set1_ps(gamma);
    let alpha_vec = _mm256_set1_ps(alpha);
    let one_vec = _mm256_set1_ps(1.0);

    for i in 0..chunks {
        let idx = i * 8;

        let r = _mm256_loadu_ps(rewards.as_ptr().add(idx));
        let nv = _mm256_loadu_ps(next_values.as_ptr().add(idx));
        let nlp = _mm256_loadu_ps(next_log_probs.as_ptr().add(idx));
        let d = _mm256_loadu_ps(dones.as_ptr().add(idx));
        let cv = _mm256_loadu_ps(current_values.as_ptr().add(idx));

        // mask = 1 - done
        let mask = _mm256_sub_ps(one_vec, d);

        // entropy_bonus = next_value - alpha * log_prob
        let alpha_lp = _mm256_mul_ps(alpha_vec, nlp);
        let v_minus_entropy = _mm256_sub_ps(nv, alpha_lp);

        // target = reward + gamma * mask * (next_value - alpha * log_prob)
        let gamma_v = _mm256_mul_ps(gamma_vec, v_minus_entropy);
        let masked = _mm256_mul_ps(gamma_v, mask);
        let target = _mm256_add_ps(r, masked);

        // td_error = target - current_value
        let error = _mm256_sub_ps(target, cv);

        _mm256_storeu_ps(td_errors.as_mut_ptr().add(idx), error);
    }

    // Handle remainder
    for i in (chunks * 8)..batch_size {
        let mask = 1.0 - dones[i];
        let target = next_values[i] - alpha * next_log_probs[i];
        td_errors[i] = rewards[i] + gamma * mask * target - current_values[i];
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_td3_td_errors_avx2(
    rewards: &[f32],
    q1_next: &[f32],
    q2_next: &[f32],
    dones: &[f32],
    current_values: &[f32],
    td_errors: &mut [f32],
    gamma: f32,
) {
    let batch_size = rewards.len();
    let chunks = batch_size / 8;

    let gamma_vec = _mm256_set1_ps(gamma);
    let one_vec = _mm256_set1_ps(1.0);

    for i in 0..chunks {
        let idx = i * 8;

        let r = _mm256_loadu_ps(rewards.as_ptr().add(idx));
        let q1 = _mm256_loadu_ps(q1_next.as_ptr().add(idx));
        let q2 = _mm256_loadu_ps(q2_next.as_ptr().add(idx));
        let d = _mm256_loadu_ps(dones.as_ptr().add(idx));
        let cv = _mm256_loadu_ps(current_values.as_ptr().add(idx));

        // mask = 1 - done
        let mask = _mm256_sub_ps(one_vec, d);

        // min_q = min(q1, q2)
        let min_q = _mm256_min_ps(q1, q2);

        // target = reward + gamma * mask * min_q
        let gamma_q = _mm256_mul_ps(gamma_vec, min_q);
        let masked = _mm256_mul_ps(gamma_q, mask);
        let target = _mm256_add_ps(r, masked);

        // td_error = target - current_value
        let error = _mm256_sub_ps(target, cv);

        _mm256_storeu_ps(td_errors.as_mut_ptr().add(idx), error);
    }

    // Handle remainder
    for i in (chunks * 8)..batch_size {
        let mask = 1.0 - dones[i];
        let min_q = q1_next[i].min(q2_next[i]);
        let target = rewards[i] + gamma * mask * min_q;
        td_errors[i] = target - current_values[i];
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn compute_priorities_avx2(td_errors: &[f32], priorities: &mut [f32], epsilon: f32) {
    let batch_size = td_errors.len();
    let chunks = batch_size / 8;

    let epsilon_vec = _mm256_set1_ps(epsilon);
    let sign_mask = _mm256_set1_ps(-0.0); // For abs via andnot

    for i in 0..chunks {
        let idx = i * 8;

        let td = _mm256_loadu_ps(td_errors.as_ptr().add(idx));

        // abs(td) = andnot(sign_mask, td)
        let abs_td = _mm256_andnot_ps(sign_mask, td);

        // priority = abs(td) + epsilon
        let priority = _mm256_add_ps(abs_td, epsilon_vec);

        _mm256_storeu_ps(priorities.as_mut_ptr().add(idx), priority);
    }

    // Handle remainder
    for i in (chunks * 8)..batch_size {
        priorities[i] = td_errors[i].abs() + epsilon;
    }
}

// ============================================================================
// NEON Implementations (ARM)
// ============================================================================

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
unsafe fn compute_td_errors_neon(
    rewards: &[f32],
    next_values: &[f32],
    dones: &[f32],
    current_values: &[f32],
    td_errors: &mut [f32],
    gamma: f32,
) {
    let batch_size = rewards.len();
    let chunks = batch_size / 4;

    let gamma_vec = vdupq_n_f32(gamma);
    let one_vec = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let idx = i * 4;

        let r = vld1q_f32(rewards.as_ptr().add(idx));
        let nv = vld1q_f32(next_values.as_ptr().add(idx));
        let d = vld1q_f32(dones.as_ptr().add(idx));
        let cv = vld1q_f32(current_values.as_ptr().add(idx));

        // mask = 1 - done
        let mask = vsubq_f32(one_vec, d);

        // target = reward + gamma * mask * next_value
        let gamma_nv = vmulq_f32(gamma_vec, nv);
        let masked = vmulq_f32(gamma_nv, mask);
        let target = vaddq_f32(r, masked);

        // td_error = target - current_value
        let error = vsubq_f32(target, cv);

        vst1q_f32(td_errors.as_mut_ptr().add(idx), error);
    }

    // Handle remainder
    for i in (chunks * 4)..batch_size {
        let mask = 1.0 - dones[i];
        td_errors[i] = rewards[i] + gamma * mask * next_values[i] - current_values[i];
    }
}

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
unsafe fn compute_sac_td_errors_neon(
    rewards: &[f32],
    next_values: &[f32],
    next_log_probs: &[f32],
    dones: &[f32],
    current_values: &[f32],
    td_errors: &mut [f32],
    gamma: f32,
    alpha: f32,
) {
    let batch_size = rewards.len();
    let chunks = batch_size / 4;

    let gamma_vec = vdupq_n_f32(gamma);
    let alpha_vec = vdupq_n_f32(alpha);
    let one_vec = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let idx = i * 4;

        let r = vld1q_f32(rewards.as_ptr().add(idx));
        let nv = vld1q_f32(next_values.as_ptr().add(idx));
        let nlp = vld1q_f32(next_log_probs.as_ptr().add(idx));
        let d = vld1q_f32(dones.as_ptr().add(idx));
        let cv = vld1q_f32(current_values.as_ptr().add(idx));

        // mask = 1 - done
        let mask = vsubq_f32(one_vec, d);

        // entropy_bonus = next_value - alpha * log_prob
        let alpha_lp = vmulq_f32(alpha_vec, nlp);
        let v_minus_entropy = vsubq_f32(nv, alpha_lp);

        // target = reward + gamma * mask * (next_value - alpha * log_prob)
        let gamma_v = vmulq_f32(gamma_vec, v_minus_entropy);
        let masked = vmulq_f32(gamma_v, mask);
        let target = vaddq_f32(r, masked);

        // td_error = target - current_value
        let error = vsubq_f32(target, cv);

        vst1q_f32(td_errors.as_mut_ptr().add(idx), error);
    }

    // Handle remainder
    for i in (chunks * 4)..batch_size {
        let mask = 1.0 - dones[i];
        let target = next_values[i] - alpha * next_log_probs[i];
        td_errors[i] = rewards[i] + gamma * mask * target - current_values[i];
    }
}

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
unsafe fn compute_td3_td_errors_neon(
    rewards: &[f32],
    q1_next: &[f32],
    q2_next: &[f32],
    dones: &[f32],
    current_values: &[f32],
    td_errors: &mut [f32],
    gamma: f32,
) {
    let batch_size = rewards.len();
    let chunks = batch_size / 4;

    let gamma_vec = vdupq_n_f32(gamma);
    let one_vec = vdupq_n_f32(1.0);

    for i in 0..chunks {
        let idx = i * 4;

        let r = vld1q_f32(rewards.as_ptr().add(idx));
        let q1 = vld1q_f32(q1_next.as_ptr().add(idx));
        let q2 = vld1q_f32(q2_next.as_ptr().add(idx));
        let d = vld1q_f32(dones.as_ptr().add(idx));
        let cv = vld1q_f32(current_values.as_ptr().add(idx));

        // mask = 1 - done
        let mask = vsubq_f32(one_vec, d);

        // min_q = min(q1, q2)
        let min_q = vminq_f32(q1, q2);

        // target = reward + gamma * mask * min_q
        let gamma_q = vmulq_f32(gamma_vec, min_q);
        let masked = vmulq_f32(gamma_q, mask);
        let target = vaddq_f32(r, masked);

        // td_error = target - current_value
        let error = vsubq_f32(target, cv);

        vst1q_f32(td_errors.as_mut_ptr().add(idx), error);
    }

    // Handle remainder
    for i in (chunks * 4)..batch_size {
        let mask = 1.0 - dones[i];
        let min_q = q1_next[i].min(q2_next[i]);
        let target = rewards[i] + gamma * mask * min_q;
        td_errors[i] = target - current_values[i];
    }
}

// ============================================================================
// Scalar Fallback
// ============================================================================

fn compute_td_errors_scalar(
    rewards: &[f32],
    next_values: &[f32],
    dones: &[f32],
    current_values: &[f32],
    td_errors: &mut [f32],
    gamma: f32,
) {
    for i in 0..rewards.len() {
        let mask = 1.0 - dones[i];
        td_errors[i] = rewards[i] + gamma * mask * next_values[i] - current_values[i];
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_td_errors_basic() {
        let rewards = vec![1.0f32; 16];
        let next_values = vec![0.5f32; 16];
        let dones = vec![0.0f32; 16];
        let current_values = vec![0.3f32; 16];
        let gamma = 0.99;

        let td_errors =
            compute_td_errors_batch(&rewards, &next_values, &dones, &current_values, gamma)
                .unwrap();

        assert_eq!(td_errors.len(), 16);

        // Expected: 1.0 + 0.99 * 1.0 * 0.5 - 0.3 = 1.0 + 0.495 - 0.3 = 1.195
        let expected = 1.0 + gamma * 0.5 - 0.3;
        for &td in &td_errors {
            assert!((td - expected).abs() < 1e-5, "Expected {}, got {}", expected, td);
        }
    }

    #[test]
    fn test_td_errors_with_done() {
        let rewards = vec![1.0f32; 8];
        let next_values = vec![0.5f32; 8];
        let mut dones = vec![0.0f32; 8];
        dones[0] = 1.0; // First transition is terminal
        let current_values = vec![0.3f32; 8];
        let gamma = 0.99;

        let td_errors =
            compute_td_errors_batch(&rewards, &next_values, &dones, &current_values, gamma)
                .unwrap();

        // Terminal: 1.0 + 0.99 * 0.0 * 0.5 - 0.3 = 0.7
        assert!((td_errors[0] - 0.7).abs() < 1e-5);

        // Non-terminal: 1.0 + 0.99 * 1.0 * 0.5 - 0.3 = 1.195
        let expected_non_terminal = 1.0 + gamma * 0.5 - 0.3;
        assert!((td_errors[1] - expected_non_terminal).abs() < 1e-5);
    }

    #[test]
    fn test_sac_td_errors() {
        let rewards = vec![1.0f32; 16];
        let next_values = vec![0.5f32; 16];
        let next_log_probs = vec![-1.0f32; 16]; // log prob of -1 means prob ~0.37
        let dones = vec![0.0f32; 16];
        let current_values = vec![0.3f32; 16];
        let gamma = 0.99;
        let alpha = 0.2;

        let td_errors = compute_sac_td_errors(
            &rewards,
            &next_values,
            &next_log_probs,
            &dones,
            &current_values,
            gamma,
            alpha,
        )
        .unwrap();

        assert_eq!(td_errors.len(), 16);

        // Expected: 1.0 + 0.99 * 1.0 * (0.5 - 0.2 * (-1.0)) - 0.3
        //         = 1.0 + 0.99 * (0.5 + 0.2) - 0.3 = 1.0 + 0.6930 - 0.3 = 1.393
        let expected = 1.0 + gamma * (0.5 - alpha * (-1.0)) - 0.3;
        for &td in &td_errors {
            assert!((td - expected).abs() < 1e-5, "Expected {}, got {}", expected, td);
        }
    }

    #[test]
    fn test_td3_td_errors() {
        let rewards = vec![1.0f32; 16];
        let q1_next = vec![0.6f32; 16];
        let q2_next = vec![0.4f32; 16]; // q2 is lower
        let dones = vec![0.0f32; 16];
        let current_values = vec![0.3f32; 16];
        let gamma = 0.99;

        let td_errors = compute_td3_td_errors(
            &rewards,
            &q1_next,
            &q2_next,
            &dones,
            &current_values,
            gamma,
        )
        .unwrap();

        assert_eq!(td_errors.len(), 16);

        // Expected: 1.0 + 0.99 * 1.0 * min(0.6, 0.4) - 0.3 = 1.0 + 0.396 - 0.3 = 1.096
        let expected = 1.0 + gamma * 0.4 - 0.3;
        for &td in &td_errors {
            assert!((td - expected).abs() < 1e-5, "Expected {}, got {}", expected, td);
        }
    }

    #[test]
    fn test_priorities() {
        let td_errors = vec![-1.0f32, 0.5, -0.3, 0.0, 2.0];
        let epsilon = 1e-6;

        let priorities = compute_priorities(&td_errors, epsilon);

        assert_eq!(priorities.len(), 5);
        assert!((priorities[0] - (1.0 + epsilon)).abs() < 1e-5);
        assert!((priorities[1] - (0.5 + epsilon)).abs() < 1e-5);
        assert!((priorities[2] - (0.3 + epsilon)).abs() < 1e-5);
        assert!((priorities[3] - epsilon).abs() < 1e-5);
        assert!((priorities[4] - (2.0 + epsilon)).abs() < 1e-5);
    }

    #[test]
    fn test_size_mismatch_error() {
        let rewards = vec![1.0f32; 16];
        let next_values = vec![0.5f32; 8]; // Wrong size
        let dones = vec![0.0f32; 16];
        let current_values = vec![0.3f32; 16];

        let result =
            compute_td_errors_batch(&rewards, &next_values, &dones, &current_values, 0.99);
        assert!(result.is_err());
    }
}
