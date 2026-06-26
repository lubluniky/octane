//! SIMD-accelerated log probability computations for continuous distributions.
//!
//! This module provides vectorized log probability computation for Gaussian
//! distributions used in continuous action space RL algorithms (SAC, TD3, DDPG, PPO).
//!
//! # Supported Distributions
//!
//! - **Diagonal Gaussian**: Standard normal distribution with diagonal covariance
//! - **Squashed Gaussian**: Gaussian with tanh squashing for bounded actions (SAC)
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::simd::log_prob::{gaussian_log_prob_simd, squashed_gaussian_log_prob_simd};
//!
//! let actions = vec![0.5f32; 256];
//! let means = vec![0.0f32; 256];
//! let log_stds = vec![0.0f32; 256]; // std = exp(0) = 1
//!
//! // Standard Gaussian log probability
//! let log_probs = gaussian_log_prob_simd(&actions, &means, &log_stds)?;
//!
//! // Squashed Gaussian (for SAC with bounded actions)
//! let squashed_log_probs = squashed_gaussian_log_prob_simd(&actions, &means, &log_stds)?;
//! ```

#![allow(unsafe_code)]

use super::{Result, SimdError};

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
use std::arch::x86_64::*;

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
use std::arch::aarch64::*;

// ============================================================================
// Constants
// ============================================================================

/// Log(2 * PI) constant.
const LOG_2PI: f32 = 1.837_877;

/// Small epsilon for numerical stability.
const EPSILON: f32 = 1e-6;

// ============================================================================
// Gaussian Log Probability
// ============================================================================

/// Compute log probability of samples under a diagonal Gaussian distribution.
///
/// The log probability is:
/// `log_prob = -0.5 * (log(2*pi) + 2*log_std + ((x - mean) / std)^2)`
///
/// For a batch of samples with dimension D, the total log probability is the sum
/// over dimensions.
///
/// # Arguments
///
/// * `actions` - Sampled actions [batch_size * action_dim]
/// * `means` - Distribution means [batch_size * action_dim]
/// * `log_stds` - Log standard deviations [batch_size * action_dim] (log_std, not std)
///
/// # Returns
///
/// Log probabilities per element [batch_size * action_dim]. Sum over action_dim for total.
pub fn gaussian_log_prob_simd(
    actions: &[f32],
    means: &[f32],
    log_stds: &[f32],
) -> Result<Vec<f32>> {
    let n = actions.len();

    if means.len() != n {
        return Err(SimdError::SizeMismatch {
            expected: n,
            actual: means.len(),
        });
    }
    if log_stds.len() != n {
        return Err(SimdError::SizeMismatch {
            expected: n,
            actual: log_stds.len(),
        });
    }

    let mut log_probs = vec![0.0f32; n];

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        if super::x86::is_avx2_available() {
            unsafe {
                gaussian_log_prob_avx2(actions, means, log_stds, &mut log_probs);
            }
            return Ok(log_probs);
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        if super::is_neon_available() {
            unsafe {
                gaussian_log_prob_neon(actions, means, log_stds, &mut log_probs);
            }
            return Ok(log_probs);
        }
    }

    // Scalar fallback
    gaussian_log_prob_scalar(actions, means, log_stds, &mut log_probs);

    Ok(log_probs)
}

/// Compute log probability and reduce over action dimension.
///
/// # Arguments
///
/// * `actions` - Sampled actions [batch_size, action_dim]
/// * `means` - Distribution means [batch_size, action_dim]
/// * `log_stds` - Log standard deviations [batch_size, action_dim]
/// * `batch_size` - Number of samples
/// * `action_dim` - Action dimensionality
///
/// # Returns
///
/// Total log probabilities per sample [batch_size]
pub fn gaussian_log_prob_batch(
    actions: &[f32],
    means: &[f32],
    log_stds: &[f32],
    batch_size: usize,
    action_dim: usize,
) -> Result<Vec<f32>> {
    let expected_len = batch_size * action_dim;

    if actions.len() != expected_len {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: actions.len(),
        });
    }

    // First compute element-wise log probs
    let element_log_probs = gaussian_log_prob_simd(actions, means, log_stds)?;

    // Then reduce over action dimension
    let mut batch_log_probs = vec![0.0f32; batch_size];

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        if super::x86::is_avx2_available() && action_dim >= 8 {
            unsafe {
                reduce_sum_avx2(
                    &element_log_probs,
                    &mut batch_log_probs,
                    batch_size,
                    action_dim,
                );
            }
            return Ok(batch_log_probs);
        }
    }

    // Scalar reduction
    for b in 0..batch_size {
        let start = b * action_dim;
        batch_log_probs[b] = element_log_probs[start..start + action_dim].iter().sum();
    }

    Ok(batch_log_probs)
}

// ============================================================================
// Squashed Gaussian Log Probability (for SAC)
// ============================================================================

/// Compute log probability for squashed Gaussian (tanh-transformed).
///
/// For SAC, actions are squashed through tanh: `a = tanh(u)` where `u ~ N(mean, std)`.
/// The log probability includes a Jacobian correction:
/// `log_prob = log_prob_gaussian(u) - sum(log(1 - tanh(u)^2 + eps))`
///
/// # Arguments
///
/// * `pre_squash_actions` - Pre-tanh actions (u values) [batch_size * action_dim]
/// * `means` - Distribution means [batch_size * action_dim]
/// * `log_stds` - Log standard deviations [batch_size * action_dim]
///
/// # Returns
///
/// Corrected log probabilities per element [batch_size * action_dim]
pub fn squashed_gaussian_log_prob_simd(
    pre_squash_actions: &[f32],
    means: &[f32],
    log_stds: &[f32],
) -> Result<Vec<f32>> {
    let n = pre_squash_actions.len();

    if means.len() != n {
        return Err(SimdError::SizeMismatch {
            expected: n,
            actual: means.len(),
        });
    }
    if log_stds.len() != n {
        return Err(SimdError::SizeMismatch {
            expected: n,
            actual: log_stds.len(),
        });
    }

    let mut log_probs = vec![0.0f32; n];

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        if super::x86::is_avx2_available() {
            unsafe {
                squashed_gaussian_log_prob_avx2(
                    pre_squash_actions,
                    means,
                    log_stds,
                    &mut log_probs,
                );
            }
            return Ok(log_probs);
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        if super::is_neon_available() {
            unsafe {
                squashed_gaussian_log_prob_neon(
                    pre_squash_actions,
                    means,
                    log_stds,
                    &mut log_probs,
                );
            }
            return Ok(log_probs);
        }
    }

    // Scalar fallback
    squashed_gaussian_log_prob_scalar(pre_squash_actions, means, log_stds, &mut log_probs);

    Ok(log_probs)
}

/// Compute squashed Gaussian log probability from already-squashed actions.
///
/// Given squashed actions `a = tanh(u)`, recovers `u` via atanh and computes log probability.
///
/// # Arguments
///
/// * `squashed_actions` - Post-tanh actions in [-1, 1] [batch_size * action_dim]
/// * `means` - Distribution means [batch_size * action_dim]
/// * `log_stds` - Log standard deviations [batch_size * action_dim]
///
/// # Returns
///
/// Corrected log probabilities per element [batch_size * action_dim]
pub fn squashed_gaussian_log_prob_from_squashed(
    squashed_actions: &[f32],
    means: &[f32],
    log_stds: &[f32],
) -> Result<Vec<f32>> {
    let n = squashed_actions.len();

    if means.len() != n {
        return Err(SimdError::SizeMismatch {
            expected: n,
            actual: means.len(),
        });
    }
    if log_stds.len() != n {
        return Err(SimdError::SizeMismatch {
            expected: n,
            actual: log_stds.len(),
        });
    }

    let mut log_probs = vec![0.0f32; n];

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        if super::x86::is_avx2_available() {
            unsafe {
                squashed_gaussian_log_prob_from_squashed_avx2(
                    squashed_actions,
                    means,
                    log_stds,
                    &mut log_probs,
                );
            }
            return Ok(log_probs);
        }
    }

    // Scalar fallback
    for i in 0..n {
        // Clamp to prevent atanh from returning inf
        let a_clamped = squashed_actions[i].clamp(-1.0 + EPSILON, 1.0 - EPSILON);

        // Recover pre-squash action: u = atanh(a)
        let u = 0.5 * ((1.0 + a_clamped) / (1.0 - a_clamped)).ln();

        // Compute Gaussian log probability
        let std = log_stds[i].exp();
        let z = (u - means[i]) / std;
        let gaussian_lp = -0.5 * (LOG_2PI + 2.0 * log_stds[i] + z * z);

        // Jacobian correction: -log(1 - tanh(u)^2 + eps) = -log(1 - a^2 + eps)
        let jacobian = -(1.0 - a_clamped * a_clamped + EPSILON).ln();

        log_probs[i] = gaussian_lp + jacobian;
    }

    Ok(log_probs)
}

/// Compute squashed Gaussian log probability and reduce over action dimension.
///
/// # Arguments
///
/// * `pre_squash_actions` - Pre-tanh actions [batch_size, action_dim]
/// * `means` - Distribution means [batch_size, action_dim]
/// * `log_stds` - Log standard deviations [batch_size, action_dim]
/// * `batch_size` - Number of samples
/// * `action_dim` - Action dimensionality
///
/// # Returns
///
/// Total log probabilities per sample [batch_size]
pub fn squashed_gaussian_log_prob_batch(
    pre_squash_actions: &[f32],
    means: &[f32],
    log_stds: &[f32],
    batch_size: usize,
    action_dim: usize,
) -> Result<Vec<f32>> {
    let expected_len = batch_size * action_dim;

    if pre_squash_actions.len() != expected_len {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: pre_squash_actions.len(),
        });
    }

    // First compute element-wise log probs
    let element_log_probs = squashed_gaussian_log_prob_simd(pre_squash_actions, means, log_stds)?;

    // Then reduce over action dimension
    let mut batch_log_probs = vec![0.0f32; batch_size];

    for b in 0..batch_size {
        let start = b * action_dim;
        batch_log_probs[b] = element_log_probs[start..start + action_dim].iter().sum();
    }

    Ok(batch_log_probs)
}

// ============================================================================
// Entropy Computation
// ============================================================================

/// Compute entropy of diagonal Gaussian distribution.
///
/// Entropy = 0.5 * (1 + log(2*pi) + 2*log_std) = 0.5 * (1 + log(2*pi*std^2))
///
/// # Arguments
///
/// * `log_stds` - Log standard deviations [batch_size * action_dim]
///
/// # Returns
///
/// Entropy per dimension [batch_size * action_dim]
pub fn gaussian_entropy_simd(log_stds: &[f32]) -> Result<Vec<f32>> {
    let n = log_stds.len();
    let mut entropy = vec![0.0f32; n];

    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        if super::x86::is_avx2_available() {
            unsafe {
                gaussian_entropy_avx2(log_stds, &mut entropy);
            }
            return Ok(entropy);
        }
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    {
        if super::is_neon_available() {
            unsafe {
                gaussian_entropy_neon(log_stds, &mut entropy);
            }
            return Ok(entropy);
        }
    }

    // Scalar fallback
    let half = 0.5f32;
    let one_plus_log_2pi = 1.0 + LOG_2PI;

    for i in 0..n {
        entropy[i] = half * (one_plus_log_2pi + 2.0 * log_stds[i]);
    }

    Ok(entropy)
}

// ============================================================================
// AVX2 Implementations
// ============================================================================

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gaussian_log_prob_avx2(
    actions: &[f32],
    means: &[f32],
    log_stds: &[f32],
    log_probs: &mut [f32],
) {
    let n = actions.len();
    let chunks = n / 8;

    let log_2pi_vec = _mm256_set1_ps(LOG_2PI);
    let neg_half = _mm256_set1_ps(-0.5);
    let two = _mm256_set1_ps(2.0);

    for i in 0..chunks {
        let idx = i * 8;

        let a = _mm256_loadu_ps(actions.as_ptr().add(idx));
        let m = _mm256_loadu_ps(means.as_ptr().add(idx));
        let ls = _mm256_loadu_ps(log_stds.as_ptr().add(idx));

        // std = exp(log_std)
        let std = fast_exp_avx2(ls);

        // z = (action - mean) / std
        let diff = _mm256_sub_ps(a, m);
        let z = _mm256_div_ps(diff, std);

        // z^2
        let z_sq = _mm256_mul_ps(z, z);

        // 2 * log_std
        let two_ls = _mm256_mul_ps(two, ls);

        // log_2pi + 2*log_std + z^2
        let inner = _mm256_add_ps(log_2pi_vec, _mm256_add_ps(two_ls, z_sq));

        // -0.5 * inner
        let lp = _mm256_mul_ps(neg_half, inner);

        _mm256_storeu_ps(log_probs.as_mut_ptr().add(idx), lp);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        let std = log_stds[i].exp();
        let z = (actions[i] - means[i]) / std;
        log_probs[i] = -0.5 * (LOG_2PI + 2.0 * log_stds[i] + z * z);
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn squashed_gaussian_log_prob_avx2(
    pre_squash: &[f32],
    means: &[f32],
    log_stds: &[f32],
    log_probs: &mut [f32],
) {
    let n = pre_squash.len();
    let chunks = n / 8;

    let log_2pi_vec = _mm256_set1_ps(LOG_2PI);
    let neg_half = _mm256_set1_ps(-0.5);
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let eps = _mm256_set1_ps(EPSILON);

    for i in 0..chunks {
        let idx = i * 8;

        let u = _mm256_loadu_ps(pre_squash.as_ptr().add(idx));
        let m = _mm256_loadu_ps(means.as_ptr().add(idx));
        let ls = _mm256_loadu_ps(log_stds.as_ptr().add(idx));

        // Compute Gaussian log probability
        let std = fast_exp_avx2(ls);
        let diff = _mm256_sub_ps(u, m);
        let z = _mm256_div_ps(diff, std);
        let z_sq = _mm256_mul_ps(z, z);
        let two_ls = _mm256_mul_ps(two, ls);
        let inner = _mm256_add_ps(log_2pi_vec, _mm256_add_ps(two_ls, z_sq));
        let gaussian_lp = _mm256_mul_ps(neg_half, inner);

        // Compute tanh(u) for Jacobian
        let tanh_u = fast_tanh_avx2(u);
        let tanh_sq = _mm256_mul_ps(tanh_u, tanh_u);

        // Jacobian correction: -log(1 - tanh(u)^2 + eps)
        let one_minus_tanh_sq = _mm256_sub_ps(one, tanh_sq);
        let arg = _mm256_add_ps(one_minus_tanh_sq, eps);
        let log_arg = fast_log_avx2(arg);
        let jacobian = _mm256_sub_ps(_mm256_setzero_ps(), log_arg); // Negate

        // Total log prob = gaussian_lp - jacobian (but jacobian is already negated)
        let lp = _mm256_add_ps(gaussian_lp, jacobian);

        _mm256_storeu_ps(log_probs.as_mut_ptr().add(idx), lp);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        let std = log_stds[i].exp();
        let z = (pre_squash[i] - means[i]) / std;
        let gaussian_lp = -0.5 * (LOG_2PI + 2.0 * log_stds[i] + z * z);

        let tanh_u = pre_squash[i].tanh();
        let jacobian = -(1.0 - tanh_u * tanh_u + EPSILON).ln();

        log_probs[i] = gaussian_lp + jacobian;
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn squashed_gaussian_log_prob_from_squashed_avx2(
    squashed: &[f32],
    means: &[f32],
    log_stds: &[f32],
    log_probs: &mut [f32],
) {
    let n = squashed.len();
    let chunks = n / 8;

    let log_2pi_vec = _mm256_set1_ps(LOG_2PI);
    let neg_half = _mm256_set1_ps(-0.5);
    let half = _mm256_set1_ps(0.5);
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);
    let eps = _mm256_set1_ps(EPSILON);
    let clamp_lo = _mm256_set1_ps(-1.0 + EPSILON);
    let clamp_hi = _mm256_set1_ps(1.0 - EPSILON);

    for i in 0..chunks {
        let idx = i * 8;

        let a = _mm256_loadu_ps(squashed.as_ptr().add(idx));
        let m = _mm256_loadu_ps(means.as_ptr().add(idx));
        let ls = _mm256_loadu_ps(log_stds.as_ptr().add(idx));

        // Clamp squashed actions
        let a_clamped = _mm256_max_ps(_mm256_min_ps(a, clamp_hi), clamp_lo);

        // Compute atanh(a) = 0.5 * ln((1+a)/(1-a))
        let one_plus_a = _mm256_add_ps(one, a_clamped);
        let one_minus_a = _mm256_sub_ps(one, a_clamped);
        let ratio = _mm256_div_ps(one_plus_a, one_minus_a);
        let ln_ratio = fast_log_avx2(ratio);
        let u = _mm256_mul_ps(half, ln_ratio);

        // Compute Gaussian log probability
        let std = fast_exp_avx2(ls);
        let diff = _mm256_sub_ps(u, m);
        let z = _mm256_div_ps(diff, std);
        let z_sq = _mm256_mul_ps(z, z);
        let two_ls = _mm256_mul_ps(two, ls);
        let inner = _mm256_add_ps(log_2pi_vec, _mm256_add_ps(two_ls, z_sq));
        let gaussian_lp = _mm256_mul_ps(neg_half, inner);

        // Jacobian correction: -log(1 - a^2 + eps)
        let a_sq = _mm256_mul_ps(a_clamped, a_clamped);
        let one_minus_a_sq = _mm256_sub_ps(one, a_sq);
        let arg = _mm256_add_ps(one_minus_a_sq, eps);
        let log_arg = fast_log_avx2(arg);
        let jacobian = _mm256_sub_ps(_mm256_setzero_ps(), log_arg);

        let lp = _mm256_add_ps(gaussian_lp, jacobian);

        _mm256_storeu_ps(log_probs.as_mut_ptr().add(idx), lp);
    }

    // Handle remainder
    for i in (chunks * 8)..n {
        let a_clamped = squashed[i].clamp(-1.0 + EPSILON, 1.0 - EPSILON);
        let u = 0.5 * ((1.0 + a_clamped) / (1.0 - a_clamped)).ln();
        let std = log_stds[i].exp();
        let z = (u - means[i]) / std;
        let gaussian_lp = -0.5 * (LOG_2PI + 2.0 * log_stds[i] + z * z);
        let jacobian = -(1.0 - a_clamped * a_clamped + EPSILON).ln();
        log_probs[i] = gaussian_lp + jacobian;
    }
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn gaussian_entropy_avx2(log_stds: &[f32], entropy: &mut [f32]) {
    let n = log_stds.len();
    let chunks = n / 8;

    let half = _mm256_set1_ps(0.5);
    let one_plus_log_2pi = _mm256_set1_ps(1.0 + LOG_2PI);
    let two = _mm256_set1_ps(2.0);

    for i in 0..chunks {
        let idx = i * 8;
        let ls = _mm256_loadu_ps(log_stds.as_ptr().add(idx));

        // entropy = 0.5 * (1 + log_2pi + 2*log_std)
        let two_ls = _mm256_mul_ps(two, ls);
        let inner = _mm256_add_ps(one_plus_log_2pi, two_ls);
        let ent = _mm256_mul_ps(half, inner);

        _mm256_storeu_ps(entropy.as_mut_ptr().add(idx), ent);
    }

    for i in (chunks * 8)..n {
        entropy[i] = 0.5 * (1.0 + LOG_2PI + 2.0 * log_stds[i]);
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn reduce_sum_avx2(input: &[f32], output: &mut [f32], batch_size: usize, action_dim: usize) {
    let chunks = action_dim / 8;

    for b in 0..batch_size {
        let offset = b * action_dim;

        let mut sum_vec = _mm256_setzero_ps();

        for c in 0..chunks {
            let v = _mm256_loadu_ps(input.as_ptr().add(offset + c * 8));
            sum_vec = _mm256_add_ps(sum_vec, v);
        }

        // Horizontal sum
        let mut sum_arr = [0.0f32; 8];
        _mm256_storeu_ps(sum_arr.as_mut_ptr(), sum_vec);
        let mut sum: f32 = sum_arr.iter().sum();

        // Handle remainder
        for i in (chunks * 8)..action_dim {
            sum += input[offset + i];
        }

        output[b] = sum;
    }
}

// ============================================================================
// Fast Math Helpers (AVX2)
// ============================================================================

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
    let log2e = _mm256_set1_ps(1.4426950408889634);

    let x_clamped = _mm256_max_ps(
        _mm256_min_ps(x, _mm256_set1_ps(88.0)),
        _mm256_set1_ps(-88.0),
    );

    let t = _mm256_mul_ps(x_clamped, log2e);
    let t_floor = _mm256_floor_ps(t);
    let f = _mm256_sub_ps(t, t_floor);
    let n = _mm256_cvtps_epi32(t_floor);

    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(0.6931471805599453);
    let c2 = _mm256_set1_ps(0.2402265069591007);
    let c3 = _mm256_set1_ps(0.0555041086648216);
    let c4 = _mm256_set1_ps(0.0096181291076284);
    let c5 = _mm256_set1_ps(0.0013333558146428);

    let mut p = c5;
    p = _mm256_fmadd_ps(p, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);

    let n_scaled = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    let scale = _mm256_castsi256_ps(n_scaled);

    _mm256_mul_ps(p, scale)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn fast_log_avx2(x: __m256) -> __m256 {
    let one = _mm256_set1_ps(1.0);
    let ln2 = _mm256_set1_ps(0.6931471805599453);

    let xi = _mm256_castps_si256(x);
    let exp = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
    let exp_f = _mm256_cvtepi32_ps(exp);

    let mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
    let one_bits = _mm256_set1_epi32(0x3F800000);
    let m_int = _mm256_or_si256(_mm256_and_si256(xi, mantissa_mask), one_bits);
    let m = _mm256_castsi256_ps(m_int);

    let m_minus_1 = _mm256_sub_ps(m, one);

    let c1 = _mm256_set1_ps(0.9999964239);
    let c2 = _mm256_set1_ps(-0.4998741238);
    let c3 = _mm256_set1_ps(0.3317990258);
    let c4 = _mm256_set1_ps(-0.2407338084);
    let c5 = _mm256_set1_ps(0.1676540711);
    let c6 = _mm256_set1_ps(-0.0953293897);

    let mut result = c6;
    result = _mm256_fmadd_ps(result, m_minus_1, c5);
    result = _mm256_fmadd_ps(result, m_minus_1, c4);
    result = _mm256_fmadd_ps(result, m_minus_1, c3);
    result = _mm256_fmadd_ps(result, m_minus_1, c2);
    result = _mm256_fmadd_ps(result, m_minus_1, c1);
    result = _mm256_mul_ps(result, m_minus_1);

    _mm256_fmadd_ps(exp_f, ln2, result)
}

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx2",
    target_feature = "fma"
))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn fast_tanh_avx2(x: __m256) -> __m256 {
    // tanh(x) = (exp(2x) - 1) / (exp(2x) + 1)
    // Or use approximation for better performance
    let two = _mm256_set1_ps(2.0);
    let one = _mm256_set1_ps(1.0);

    let two_x = _mm256_mul_ps(two, x);
    let exp_2x = fast_exp_avx2(two_x);

    let num = _mm256_sub_ps(exp_2x, one);
    let den = _mm256_add_ps(exp_2x, one);

    _mm256_div_ps(num, den)
}

// ============================================================================
// NEON Implementations (ARM)
// ============================================================================

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
unsafe fn gaussian_log_prob_neon(
    actions: &[f32],
    means: &[f32],
    log_stds: &[f32],
    log_probs: &mut [f32],
) {
    let n = actions.len();
    let chunks = n / 4;

    let log_2pi_vec = vdupq_n_f32(LOG_2PI);
    let neg_half = vdupq_n_f32(-0.5);
    let two = vdupq_n_f32(2.0);

    for i in 0..chunks {
        let idx = i * 4;

        let a = vld1q_f32(actions.as_ptr().add(idx));
        let m = vld1q_f32(means.as_ptr().add(idx));
        let ls = vld1q_f32(log_stds.as_ptr().add(idx));

        // std = exp(log_std) - using scalar for simplicity
        let mut std_arr = [0.0f32; 4];
        let mut ls_arr = [0.0f32; 4];
        vst1q_f32(ls_arr.as_mut_ptr(), ls);
        for j in 0..4 {
            std_arr[j] = ls_arr[j].exp();
        }
        let std = vld1q_f32(std_arr.as_ptr());

        // z = (action - mean) / std
        let diff = vsubq_f32(a, m);
        let z = vdivq_f32(diff, std);

        // z^2
        let z_sq = vmulq_f32(z, z);

        // 2 * log_std
        let two_ls = vmulq_f32(two, ls);

        // log_2pi + 2*log_std + z^2
        let inner = vaddq_f32(log_2pi_vec, vaddq_f32(two_ls, z_sq));

        // -0.5 * inner
        let lp = vmulq_f32(neg_half, inner);

        vst1q_f32(log_probs.as_mut_ptr().add(idx), lp);
    }

    // Handle remainder
    for i in (chunks * 4)..n {
        let std = log_stds[i].exp();
        let z = (actions[i] - means[i]) / std;
        log_probs[i] = -0.5 * (LOG_2PI + 2.0 * log_stds[i] + z * z);
    }
}

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
unsafe fn squashed_gaussian_log_prob_neon(
    pre_squash: &[f32],
    means: &[f32],
    log_stds: &[f32],
    log_probs: &mut [f32],
) {
    let n = pre_squash.len();

    // Use scalar implementation for NEON since tanh/log are complex
    for i in 0..n {
        let std = log_stds[i].exp();
        let z = (pre_squash[i] - means[i]) / std;
        let gaussian_lp = -0.5 * (LOG_2PI + 2.0 * log_stds[i] + z * z);

        let tanh_u = pre_squash[i].tanh();
        let jacobian = -(1.0 - tanh_u * tanh_u + EPSILON).ln();

        log_probs[i] = gaussian_lp + jacobian;
    }
}

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
unsafe fn gaussian_entropy_neon(log_stds: &[f32], entropy: &mut [f32]) {
    let n = log_stds.len();
    let chunks = n / 4;

    let half = vdupq_n_f32(0.5);
    let one_plus_log_2pi = vdupq_n_f32(1.0 + LOG_2PI);
    let two = vdupq_n_f32(2.0);

    for i in 0..chunks {
        let idx = i * 4;
        let ls = vld1q_f32(log_stds.as_ptr().add(idx));

        let two_ls = vmulq_f32(two, ls);
        let inner = vaddq_f32(one_plus_log_2pi, two_ls);
        let ent = vmulq_f32(half, inner);

        vst1q_f32(entropy.as_mut_ptr().add(idx), ent);
    }

    for i in (chunks * 4)..n {
        entropy[i] = 0.5 * (1.0 + LOG_2PI + 2.0 * log_stds[i]);
    }
}

// ============================================================================
// Scalar Fallbacks
// ============================================================================

fn gaussian_log_prob_scalar(
    actions: &[f32],
    means: &[f32],
    log_stds: &[f32],
    log_probs: &mut [f32],
) {
    for i in 0..actions.len() {
        let std = log_stds[i].exp();
        let z = (actions[i] - means[i]) / std;
        log_probs[i] = -0.5 * (LOG_2PI + 2.0 * log_stds[i] + z * z);
    }
}

fn squashed_gaussian_log_prob_scalar(
    pre_squash: &[f32],
    means: &[f32],
    log_stds: &[f32],
    log_probs: &mut [f32],
) {
    for i in 0..pre_squash.len() {
        let std = log_stds[i].exp();
        let z = (pre_squash[i] - means[i]) / std;
        let gaussian_lp = -0.5 * (LOG_2PI + 2.0 * log_stds[i] + z * z);

        let tanh_u = pre_squash[i].tanh();
        let jacobian = -(1.0 - tanh_u * tanh_u + EPSILON).ln();

        log_probs[i] = gaussian_lp + jacobian;
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_log_prob() {
        let actions = vec![0.0f32, 0.5, 1.0, -0.5];
        let means = vec![0.0f32; 4];
        let log_stds = vec![0.0f32; 4]; // std = 1

        let log_probs = gaussian_log_prob_simd(&actions, &means, &log_stds).unwrap();

        assert_eq!(log_probs.len(), 4);

        // At mean (z=0): log_prob = -0.5 * log(2*pi) = -0.9189
        let expected_at_mean = -0.5 * LOG_2PI;
        assert!(
            (log_probs[0] - expected_at_mean).abs() < 1e-4,
            "At mean: expected {}, got {}",
            expected_at_mean,
            log_probs[0]
        );

        // Log prob should decrease as we move away from mean
        assert!(log_probs[0] > log_probs[1]);
        assert!(log_probs[1] > log_probs[2]);
    }

    #[test]
    fn test_gaussian_log_prob_batch() {
        let batch_size = 4;
        let action_dim = 3;
        let n = batch_size * action_dim;

        let actions = vec![0.0f32; n];
        let means = vec![0.0f32; n];
        let log_stds = vec![0.0f32; n];

        let batch_log_probs =
            gaussian_log_prob_batch(&actions, &means, &log_stds, batch_size, action_dim).unwrap();

        assert_eq!(batch_log_probs.len(), batch_size);

        // Each sample should have log_prob = action_dim * (-0.5 * log(2*pi))
        let expected = (action_dim as f32) * (-0.5 * LOG_2PI);
        for &lp in &batch_log_probs {
            assert!(
                (lp - expected).abs() < 1e-4,
                "Expected {}, got {}",
                expected,
                lp
            );
        }
    }

    #[test]
    fn test_squashed_gaussian_log_prob() {
        let pre_squash = vec![0.0f32, 0.5, -0.5, 1.0];
        let means = vec![0.0f32; 4];
        let log_stds = vec![0.0f32; 4];

        let log_probs = squashed_gaussian_log_prob_simd(&pre_squash, &means, &log_stds).unwrap();

        assert_eq!(log_probs.len(), 4);

        // Jacobian correction should make log probs different from standard Gaussian
        let gaussian_log_probs = gaussian_log_prob_simd(&pre_squash, &means, &log_stds).unwrap();

        // At u=0, tanh(0)=0, so jacobian = -log(1 - 0 + eps) ~= 0
        // So squashed log prob should be close to gaussian log prob at u=0
        assert!(
            (log_probs[0] - gaussian_log_probs[0]).abs() < 0.1,
            "At u=0, squashed {} should be close to gaussian {}",
            log_probs[0],
            gaussian_log_probs[0]
        );

        // At larger |u|, tanh^2 -> 1, so 1 - tanh^2 -> 0 and the tanh-Jacobian
        // correction -log(1 - tanh^2 + eps) grows large and POSITIVE. The
        // squashed density concentrates near the bound, so its log-prob is
        // GREATER (less negative) than the underlying Gaussian:
        //   log p(a) = log p(u) - log(1 - tanh^2(u) + eps),  with log(.) < 0.
        assert!(
            log_probs[3] > gaussian_log_probs[3],
            "At u=1, squashed {} should be greater than gaussian {}",
            log_probs[3],
            gaussian_log_probs[3]
        );
    }

    #[test]
    fn test_gaussian_entropy() {
        let log_stds = vec![0.0f32, 0.5, -0.5, 1.0];

        let entropy = gaussian_entropy_simd(&log_stds).unwrap();

        assert_eq!(entropy.len(), 4);

        // entropy = 0.5 * (1 + log(2*pi) + 2*log_std)
        for i in 0..4 {
            let expected = 0.5 * (1.0 + LOG_2PI + 2.0 * log_stds[i]);
            assert!(
                (entropy[i] - expected).abs() < 1e-4,
                "Expected {}, got {}",
                expected,
                entropy[i]
            );
        }

        // Higher std = higher entropy
        assert!(entropy[1] > entropy[0]); // log_std=0.5 > log_std=0
        assert!(entropy[3] > entropy[1]); // log_std=1.0 > log_std=0.5
        assert!(entropy[0] > entropy[2]); // log_std=0 > log_std=-0.5
    }

    #[test]
    fn test_squashed_from_squashed() {
        let squashed = vec![0.0f32, 0.5, -0.5, 0.9];
        let means = vec![0.0f32; 4];
        let log_stds = vec![0.0f32; 4];

        let log_probs =
            squashed_gaussian_log_prob_from_squashed(&squashed, &means, &log_stds).unwrap();

        assert_eq!(log_probs.len(), 4);

        // All log probs should be finite
        for &lp in &log_probs {
            assert!(lp.is_finite(), "Log prob {} should be finite", lp);
        }

        // Edge case: squashed actions near boundaries should still work
        let edge_squashed = vec![0.99f32, -0.99, 0.999, -0.999];
        let edge_log_probs =
            squashed_gaussian_log_prob_from_squashed(&edge_squashed, &means, &log_stds).unwrap();

        for &lp in &edge_log_probs {
            assert!(lp.is_finite(), "Edge log prob {} should be finite", lp);
        }
    }

    #[test]
    fn test_size_mismatch_errors() {
        let actions = vec![0.0f32; 10];
        let means = vec![0.0f32; 8]; // Wrong size
        let log_stds = vec![0.0f32; 10];

        let result = gaussian_log_prob_simd(&actions, &means, &log_stds);
        assert!(result.is_err());

        let result2 = squashed_gaussian_log_prob_simd(&actions, &means, &log_stds);
        assert!(result2.is_err());
    }
}
