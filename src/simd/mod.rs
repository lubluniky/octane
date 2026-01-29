//! SIMD-accelerated operations for Octane
//!
//! This module provides high-performance SIMD operations optimized for
//! Apple Silicon (ARM NEON) processors. All operations have safe Rust wrappers
//! with proper alignment checking and error handling.
//!
//! # Available Operations
//!
//! - **Buffer operations**: Fast batch gathering for replay buffers
//! - **Gaussian sampling**: Vectorized reparameterization sampling
//! - **Softmax**: SIMD-accelerated softmax computation
//! - **Categorical sampling**: Gumbel-max trick for discrete actions
//! - **GAE computation**: Generalized Advantage Estimation
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::simd::{GaussianSampler, compute_gae};
//!
//! // Initialize Gaussian sampler with a seed
//! let mut sampler = GaussianSampler::new(42);
//!
//! // Sample actions with reparameterization
//! let means = vec![0.0f32; 64];
//! let stds = vec![1.0f32; 64];
//! let samples = sampler.sample(&means, &stds)?;
//! ```

#![allow(unsafe_code)] // FFI requires unsafe

// Metal GPU integration (macOS only, requires "metal" feature)
#[cfg(target_os = "macos")]
pub mod metal;

#[cfg(all(target_os = "macos", feature = "metal"))]
pub use self::metal::{MetalContext, MetalError};

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
use std::ffi::CStr;

use thiserror::Error;

/// Errors that can occur in SIMD operations.
#[derive(Debug, Error)]
pub enum SimdError {
    /// Memory alignment error.
    #[error("Memory not aligned to {required} bytes (got alignment {actual})")]
    AlignmentError {
        /// Required alignment in bytes.
        required: usize,
        /// Actual alignment.
        actual: usize,
    },

    /// Buffer size mismatch.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    SizeMismatch {
        /// Expected size.
        expected: usize,
        /// Actual size.
        actual: usize,
    },

    /// Index out of bounds.
    #[error("Index {index} out of bounds for capacity {capacity}")]
    IndexOutOfBounds {
        /// Invalid index.
        index: usize,
        /// Buffer capacity.
        capacity: usize,
    },

    /// NEON not available.
    #[error("NEON SIMD instructions not available on this platform")]
    NeonNotAvailable,

    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Result type for SIMD operations.
pub type Result<T> = std::result::Result<T, SimdError>;

/// Required alignment for NEON operations (16 bytes).
pub const NEON_ALIGNMENT: usize = 16;

/// RNG state size for gaussian (xoroshiro128+ × 4 streams = 8 x u64).
pub const GAUSSIAN_RNG_STATE_SIZE: usize = 8;

/// RNG state size for categorical (xoshiro256** = 4 x u64).
pub const CATEGORICAL_RNG_STATE_SIZE: usize = 4;

// ============================================================================
// FFI Declarations
// ============================================================================

#[cfg(all(target_arch = "aarch64", feature = "simd"))]
mod ffi {
    use std::ffi::c_char;

    /// Gather operation flags.
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    pub struct GatherFlags(pub u32);

    impl GatherFlags {
        pub const NONE: Self = Self(0);
        pub const SORT_INDICES: Self = Self(1 << 0);
        pub const PREFETCH: Self = Self(1 << 1);
        pub const DEFAULT: Self = Self(1 << 1); // PREFETCH
    }

    /// Statistics from gather operations.
    #[repr(C)]
    #[derive(Debug, Clone, Default)]
    pub struct GatherStats {
        pub cycles: u64,
        pub cache_misses: u64,
        pub bytes_gathered: usize,
        pub bandwidth_gbps: f64,
    }

    extern "C" {
        // Buffer operations (buffer_ops_neon.h)
        pub fn gather_batch_f32(
            src: *const f32,
            indices: *const usize,
            dst: *mut f32,
            batch_size: usize,
            dim: usize,
            capacity: usize,
        );

        pub fn gather_batch_f32_ex(
            src: *const f32,
            indices: *const usize,
            dst: *mut f32,
            batch_size: usize,
            dim: usize,
            capacity: usize,
            flags: u32,
            stats: *mut GatherStats,
        );

        pub fn gather_batch_strided(
            obs: *const f32,
            actions: *const f32,
            rewards: *const f32,
            next_obs: *const f32,
            dones: *const f32,
            indices: *const usize,
            obs_batch: *mut f32,
            actions_batch: *mut f32,
            rewards_batch: *mut f32,
            next_obs_batch: *mut f32,
            dones_batch: *mut f32,
            batch_size: usize,
            obs_dim: usize,
            action_dim: usize,
            capacity: usize,
        );

        pub fn gather_batch_strided_ex(
            obs: *const f32,
            actions: *const f32,
            rewards: *const f32,
            next_obs: *const f32,
            dones: *const f32,
            indices: *const usize,
            obs_batch: *mut f32,
            actions_batch: *mut f32,
            rewards_batch: *mut f32,
            next_obs_batch: *mut f32,
            dones_batch: *mut f32,
            batch_size: usize,
            obs_dim: usize,
            action_dim: usize,
            capacity: usize,
            flags: u32,
            stats: *mut GatherStats,
        );

        pub fn scatter_priorities_f32(
            priorities: *mut f32,
            indices: *const usize,
            new_priorities: *const f32,
            batch_size: usize,
        );

        pub fn scatter_priorities_f32_ex(
            priorities: *mut f32,
            indices: *const usize,
            new_priorities: *const f32,
            batch_size: usize,
            flags: u32,
        );

        pub fn neon_available() -> i32;

        pub fn buffer_ops_version() -> *const c_char;

        // Gaussian sampling (gaussian_neon.h)
        pub fn init_rng_state(seed: u64, state: *mut u64);

        pub fn sample_standard_normal_neon(output: *mut f32, count: usize, rng_state: *mut u64);

        pub fn sample_gaussian_neon(
            mean: *const f32,
            std: *const f32,
            output: *mut f32,
            count: usize,
            rng_state: *mut u64,
        );

        pub fn sample_gaussian_batch_neon(
            mean: *const f32,
            std: *const f32,
            output: *mut f32,
            batch_size: usize,
            action_dim: usize,
            rng_state: *mut u64,
        );

        pub fn sample_gaussian_with_logprob_neon(
            mean: *const f32,
            std: *const f32,
            output: *mut f32,
            log_prob: *mut f32,
            count: usize,
            rng_state: *mut u64,
        );

        // GAE computation (gae_neon.h)
        pub fn gae_compute_neon(
            rewards: *const f32,
            values: *const f32,
            dones: *const f32,
            advantages: *mut f32,
            buffer_size: usize,
            num_envs: usize,
            gamma: f32,
            gae_lambda: f32,
            last_values: *const f32,
        );

        pub fn gae_compute_with_returns_neon(
            rewards: *const f32,
            values: *const f32,
            dones: *const f32,
            advantages: *mut f32,
            returns: *mut f32,
            buffer_size: usize,
            num_envs: usize,
            gamma: f32,
            gae_lambda: f32,
            last_values: *const f32,
        );

        pub fn gae_normalize_neon(advantages: *mut f32, count: usize, eps: f32);

        // Softmax (categorical_neon.h)
        pub fn softmax_neon(
            logits: *const f32,
            probs: *mut f32,
            batch_size: usize,
            num_actions: usize,
        );

        pub fn log_softmax_neon(
            logits: *const f32,
            log_probs: *mut f32,
            batch_size: usize,
            num_actions: usize,
        );

        // Categorical sampling with Gumbel-max trick (categorical_neon.h)
        // rng_state is 4x uint64_t (32 bytes), pass as *mut u64
        pub fn rng_init(rng: *mut u64, seed: u64);

        pub fn categorical_sample_gumbel_neon(
            logits: *const f32,
            actions: *mut u32,
            batch_size: usize,
            num_actions: usize,
            rng: *mut u64,
        );

        pub fn categorical_forward_neon(
            logits: *const f32,
            probs: *mut f32,
            log_probs: *mut f32,
            actions: *mut u32,
            action_log_probs: *mut f32,
            entropy: *mut f32,
            batch_size: usize,
            num_actions: usize,
            rng: *mut u64,
            deterministic: i32,
        );
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if memory is properly aligned for NEON operations.
#[inline]
#[allow(dead_code)]
fn check_alignment<T>(ptr: *const T, _name: &str) -> Result<()> {
    let addr = ptr as usize;
    if addr % NEON_ALIGNMENT != 0 {
        return Err(SimdError::AlignmentError {
            required: NEON_ALIGNMENT,
            actual: addr % NEON_ALIGNMENT,
        });
    }
    Ok(())
}

/// Validate indices are within capacity.
fn validate_indices(indices: &[usize], capacity: usize) -> Result<()> {
    for &idx in indices {
        if idx >= capacity {
            return Err(SimdError::IndexOutOfBounds {
                index: idx,
                capacity,
            });
        }
    }
    Ok(())
}

/// Check if NEON is available on this platform.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn is_neon_available() -> bool {
    unsafe { ffi::neon_available() != 0 }
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn is_neon_available() -> bool {
    false
}

/// Get the version string of the buffer operations library.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn buffer_ops_version() -> String {
    unsafe {
        let ptr = ffi::buffer_ops_version();
        if ptr.is_null() {
            return String::from("unknown");
        }
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn buffer_ops_version() -> String {
    String::from("N/A (not aarch64)")
}

// ============================================================================
// Gather Operations
// ============================================================================

/// Configuration for gather operations.
#[derive(Debug, Clone, Copy, Default)]
pub struct GatherConfig {
    /// Sort indices for better cache locality.
    pub sort_indices: bool,
    /// Enable software prefetching.
    pub prefetch: bool,
}

impl GatherConfig {
    /// Create default config (prefetch enabled).
    pub fn new() -> Self {
        Self {
            sort_indices: false,
            prefetch: true,
        }
    }

    /// Enable index sorting for cache optimization.
    pub fn with_sorting(mut self) -> Self {
        self.sort_indices = true;
        self
    }

    /// Disable prefetching.
    pub fn without_prefetch(mut self) -> Self {
        self.prefetch = false;
        self
    }

    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    fn to_flags(&self) -> u32 {
        let mut flags = 0u32;
        if self.sort_indices {
            flags |= 1 << 0;
        }
        if self.prefetch {
            flags |= 1 << 1;
        }
        flags
    }
}

/// Statistics from gather operations.
#[derive(Debug, Clone, Default)]
pub struct GatherStats {
    /// CPU cycles (if available).
    pub cycles: u64,
    /// Estimated cache misses.
    pub cache_misses: u64,
    /// Total bytes processed.
    pub bytes_gathered: usize,
    /// Effective bandwidth in GB/s.
    pub bandwidth_gbps: f64,
}

/// Gather rows from a 2D array using random indices.
///
/// # Arguments
///
/// * `src` - Source buffer of shape [capacity, dim]
/// * `indices` - Indices to gather [batch_size]
/// * `dim` - Dimension of each row
/// * `capacity` - Total capacity of source buffer
///
/// # Returns
///
/// Gathered data of shape [batch_size, dim]
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn gather_batch_f32(
    src: &[f32],
    indices: &[usize],
    dim: usize,
    capacity: usize,
) -> Result<Vec<f32>> {
    if !is_neon_available() {
        return Err(SimdError::NeonNotAvailable);
    }

    validate_indices(indices, capacity)?;

    let batch_size = indices.len();
    let expected_src_len = capacity * dim;
    if src.len() < expected_src_len {
        return Err(SimdError::SizeMismatch {
            expected: expected_src_len,
            actual: src.len(),
        });
    }

    let mut dst = vec![0.0f32; batch_size * dim];

    unsafe {
        ffi::gather_batch_f32(
            src.as_ptr(),
            indices.as_ptr(),
            dst.as_mut_ptr(),
            batch_size,
            dim,
            capacity,
        );
    }

    Ok(dst)
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn gather_batch_f32(
    src: &[f32],
    indices: &[usize],
    dim: usize,
    _capacity: usize,
) -> Result<Vec<f32>> {
    // Fallback implementation
    let batch_size = indices.len();
    let mut dst = vec![0.0f32; batch_size * dim];

    for (i, &idx) in indices.iter().enumerate() {
        let src_start = idx * dim;
        let dst_start = i * dim;
        dst[dst_start..dst_start + dim].copy_from_slice(&src[src_start..src_start + dim]);
    }

    Ok(dst)
}

/// Strided gather for ReplayBuffer batch sampling.
///
/// Efficiently gathers from multiple parallel arrays (observations, actions,
/// rewards, next_observations, dones) in a single pass.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn gather_replay_batch(
    obs: &[f32],
    actions: &[f32],
    rewards: &[f32],
    next_obs: &[f32],
    dones: &[f32],
    indices: &[usize],
    obs_dim: usize,
    action_dim: usize,
    capacity: usize,
) -> Result<ReplayBatchBuffers> {
    if !is_neon_available() {
        return Err(SimdError::NeonNotAvailable);
    }

    validate_indices(indices, capacity)?;

    let batch_size = indices.len();

    let mut obs_batch = vec![0.0f32; batch_size * obs_dim];
    let mut actions_batch = vec![0.0f32; batch_size * action_dim];
    let mut rewards_batch = vec![0.0f32; batch_size];
    let mut next_obs_batch = vec![0.0f32; batch_size * obs_dim];
    let mut dones_batch = vec![0.0f32; batch_size];

    unsafe {
        ffi::gather_batch_strided(
            obs.as_ptr(),
            actions.as_ptr(),
            rewards.as_ptr(),
            next_obs.as_ptr(),
            dones.as_ptr(),
            indices.as_ptr(),
            obs_batch.as_mut_ptr(),
            actions_batch.as_mut_ptr(),
            rewards_batch.as_mut_ptr(),
            next_obs_batch.as_mut_ptr(),
            dones_batch.as_mut_ptr(),
            batch_size,
            obs_dim,
            action_dim,
            capacity,
        );
    }

    Ok(ReplayBatchBuffers {
        observations: obs_batch,
        actions: actions_batch,
        rewards: rewards_batch,
        next_observations: next_obs_batch,
        dones: dones_batch,
    })
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn gather_replay_batch(
    obs: &[f32],
    actions: &[f32],
    rewards: &[f32],
    next_obs: &[f32],
    dones: &[f32],
    indices: &[usize],
    obs_dim: usize,
    action_dim: usize,
    _capacity: usize,
) -> Result<ReplayBatchBuffers> {
    let batch_size = indices.len();

    let mut obs_batch = vec![0.0f32; batch_size * obs_dim];
    let mut actions_batch = vec![0.0f32; batch_size * action_dim];
    let mut rewards_batch = vec![0.0f32; batch_size];
    let mut next_obs_batch = vec![0.0f32; batch_size * obs_dim];
    let mut dones_batch = vec![0.0f32; batch_size];

    for (i, &idx) in indices.iter().enumerate() {
        // Observations
        let src_start = idx * obs_dim;
        let dst_start = i * obs_dim;
        obs_batch[dst_start..dst_start + obs_dim]
            .copy_from_slice(&obs[src_start..src_start + obs_dim]);
        next_obs_batch[dst_start..dst_start + obs_dim]
            .copy_from_slice(&next_obs[src_start..src_start + obs_dim]);

        // Actions
        let src_start = idx * action_dim;
        let dst_start = i * action_dim;
        actions_batch[dst_start..dst_start + action_dim]
            .copy_from_slice(&actions[src_start..src_start + action_dim]);

        // Scalars
        rewards_batch[i] = rewards[idx];
        dones_batch[i] = dones[idx];
    }

    Ok(ReplayBatchBuffers {
        observations: obs_batch,
        actions: actions_batch,
        rewards: rewards_batch,
        next_observations: next_obs_batch,
        dones: dones_batch,
    })
}

/// Buffers returned from replay batch gathering.
#[derive(Debug, Clone)]
pub struct ReplayBatchBuffers {
    /// Observations [batch_size, obs_dim].
    pub observations: Vec<f32>,
    /// Actions [batch_size, action_dim].
    pub actions: Vec<f32>,
    /// Rewards [batch_size].
    pub rewards: Vec<f32>,
    /// Next observations [batch_size, obs_dim].
    pub next_observations: Vec<f32>,
    /// Done flags [batch_size].
    pub dones: Vec<f32>,
}

/// Scatter priority updates for Prioritized Experience Replay.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn scatter_priorities(
    priorities: &mut [f32],
    indices: &[usize],
    new_priorities: &[f32],
) -> Result<()> {
    if indices.len() != new_priorities.len() {
        return Err(SimdError::SizeMismatch {
            expected: indices.len(),
            actual: new_priorities.len(),
        });
    }

    validate_indices(indices, priorities.len())?;

    unsafe {
        ffi::scatter_priorities_f32(
            priorities.as_mut_ptr(),
            indices.as_ptr(),
            new_priorities.as_ptr(),
            indices.len(),
        );
    }

    Ok(())
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn scatter_priorities(
    priorities: &mut [f32],
    indices: &[usize],
    new_priorities: &[f32],
) -> Result<()> {
    if indices.len() != new_priorities.len() {
        return Err(SimdError::SizeMismatch {
            expected: indices.len(),
            actual: new_priorities.len(),
        });
    }

    for (i, &idx) in indices.iter().enumerate() {
        priorities[idx] = new_priorities[i];
    }

    Ok(())
}

// ============================================================================
// Gaussian Sampling
// ============================================================================

/// High-performance Gaussian sampler using NEON SIMD.
///
/// Uses vectorized Box-Muller transform with parallel xoroshiro128+ RNG streams.
pub struct GaussianSampler {
    rng_state: Vec<u64>,
}

impl GaussianSampler {
    /// Create a new sampler with the given seed.
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    pub fn new(seed: u64) -> Self {
        let mut rng_state = vec![0u64; GAUSSIAN_RNG_STATE_SIZE];
        unsafe {
            ffi::init_rng_state(seed, rng_state.as_mut_ptr());
        }
        Self { rng_state }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
    pub fn new(seed: u64) -> Self {
        use rand::SeedableRng;
        // Fallback: just store seed for later use with rand
        Self {
            rng_state: vec![seed; GAUSSIAN_RNG_STATE_SIZE],
        }
    }

    /// Sample from standard normal distribution N(0, 1).
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    pub fn sample_standard_normal(&mut self, count: usize) -> Result<Vec<f32>> {
        if !is_neon_available() {
            return Err(SimdError::NeonNotAvailable);
        }

        let mut output = vec![0.0f32; count];
        unsafe {
            ffi::sample_standard_normal_neon(
                output.as_mut_ptr(),
                count,
                self.rng_state.as_mut_ptr(),
            );
        }
        Ok(output)
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
    pub fn sample_standard_normal(&mut self, count: usize) -> Result<Vec<f32>> {
        use rand::{RngCore, SeedableRng};
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_state[0]);
        let output: Vec<f32> = (0..count)
            .map(|_| StandardNormal.sample(&mut rng))
            .collect();
        self.rng_state[0] = rng.next_u64();
        Ok(output)
    }

    /// Sample with reparameterization: output = mean + std * N(0, 1).
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    pub fn sample(&mut self, mean: &[f32], std: &[f32]) -> Result<Vec<f32>> {
        if mean.len() != std.len() {
            return Err(SimdError::SizeMismatch {
                expected: mean.len(),
                actual: std.len(),
            });
        }

        if !is_neon_available() {
            return Err(SimdError::NeonNotAvailable);
        }

        let count = mean.len();
        let mut output = vec![0.0f32; count];

        unsafe {
            ffi::sample_gaussian_neon(
                mean.as_ptr(),
                std.as_ptr(),
                output.as_mut_ptr(),
                count,
                self.rng_state.as_mut_ptr(),
            );
        }

        Ok(output)
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
    pub fn sample(&mut self, mean: &[f32], std: &[f32]) -> Result<Vec<f32>> {
        if mean.len() != std.len() {
            return Err(SimdError::SizeMismatch {
                expected: mean.len(),
                actual: std.len(),
            });
        }

        use rand::{RngCore, SeedableRng};
        use rand_distr::{Distribution, StandardNormal};

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_state[0]);
        let output: Vec<f32> = mean
            .iter()
            .zip(std.iter())
            .map(|(&m, &s)| {
                let z: f32 = StandardNormal.sample(&mut rng);
                m + s * z
            })
            .collect();
        self.rng_state[0] = rng.next_u64();
        Ok(output)
    }

    /// Sample and compute log probability simultaneously.
    ///
    /// Returns (samples, log_probs) for policy gradient methods.
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    pub fn sample_with_log_prob(&mut self, mean: &[f32], std: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if mean.len() != std.len() {
            return Err(SimdError::SizeMismatch {
                expected: mean.len(),
                actual: std.len(),
            });
        }

        if !is_neon_available() {
            return Err(SimdError::NeonNotAvailable);
        }

        let count = mean.len();
        let mut output = vec![0.0f32; count];
        let mut log_prob = vec![0.0f32; count];

        unsafe {
            ffi::sample_gaussian_with_logprob_neon(
                mean.as_ptr(),
                std.as_ptr(),
                output.as_mut_ptr(),
                log_prob.as_mut_ptr(),
                count,
                self.rng_state.as_mut_ptr(),
            );
        }

        Ok((output, log_prob))
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
    pub fn sample_with_log_prob(&mut self, mean: &[f32], std: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
        if mean.len() != std.len() {
            return Err(SimdError::SizeMismatch {
                expected: mean.len(),
                actual: std.len(),
            });
        }

        use rand::{RngCore, SeedableRng};
        use rand_distr::{Distribution, StandardNormal};

        let log_2pi: f32 = (2.0 * std::f32::consts::PI).ln();
        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_state[0]);

        let mut output = Vec::with_capacity(mean.len());
        let mut log_prob = Vec::with_capacity(mean.len());

        for (&m, &s) in mean.iter().zip(std.iter()) {
            let z: f32 = StandardNormal.sample(&mut rng);
            output.push(m + s * z);
            // log_prob = -0.5 * (log(2*pi) + 2*log(std) + z^2)
            log_prob.push(-0.5 * (log_2pi + 2.0 * s.ln() + z * z));
        }

        self.rng_state[0] = rng.next_u64();
        Ok((output, log_prob))
    }

    /// Batch sample for RL action selection.
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    pub fn sample_batch(
        &mut self,
        mean: &[f32],
        std: &[f32],
        batch_size: usize,
        action_dim: usize,
    ) -> Result<Vec<f32>> {
        let expected_len = batch_size * action_dim;
        if mean.len() != expected_len || std.len() != expected_len {
            return Err(SimdError::SizeMismatch {
                expected: expected_len,
                actual: mean.len().min(std.len()),
            });
        }

        if !is_neon_available() {
            return Err(SimdError::NeonNotAvailable);
        }

        let mut output = vec![0.0f32; expected_len];

        unsafe {
            ffi::sample_gaussian_batch_neon(
                mean.as_ptr(),
                std.as_ptr(),
                output.as_mut_ptr(),
                batch_size,
                action_dim,
                self.rng_state.as_mut_ptr(),
            );
        }

        Ok(output)
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
    pub fn sample_batch(
        &mut self,
        mean: &[f32],
        std: &[f32],
        batch_size: usize,
        action_dim: usize,
    ) -> Result<Vec<f32>> {
        let expected_len = batch_size * action_dim;
        if mean.len() != expected_len || std.len() != expected_len {
            return Err(SimdError::SizeMismatch {
                expected: expected_len,
                actual: mean.len().min(std.len()),
            });
        }

        self.sample(mean, std)
    }
}

// ============================================================================
// GAE Computation
// ============================================================================

/// Compute Generalized Advantage Estimation (GAE) using NEON SIMD.
///
/// # Arguments
///
/// * `rewards` - Rewards [num_steps, num_envs]
/// * `values` - Value estimates [num_steps, num_envs]
/// * `dones` - Episode termination flags [num_steps, num_envs]
/// * `num_steps` - Number of time steps
/// * `num_envs` - Number of parallel environments
/// * `gamma` - Discount factor
/// * `gae_lambda` - GAE lambda parameter
/// * `last_values` - Bootstrap values [num_envs]
///
/// # Returns
///
/// Tuple of (advantages, returns) each with shape [num_steps, num_envs]
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let expected_len = num_steps * num_envs;
    if rewards.len() != expected_len || values.len() != expected_len || dones.len() != expected_len
    {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: rewards.len().min(values.len()).min(dones.len()),
        });
    }

    if last_values.len() != num_envs {
        return Err(SimdError::SizeMismatch {
            expected: num_envs,
            actual: last_values.len(),
        });
    }

    if !is_neon_available() {
        return Err(SimdError::NeonNotAvailable);
    }

    let mut advantages = vec![0.0f32; expected_len];
    let mut returns = vec![0.0f32; expected_len];

    unsafe {
        ffi::gae_compute_with_returns_neon(
            rewards.as_ptr(),
            values.as_ptr(),
            dones.as_ptr(),
            advantages.as_mut_ptr(),
            returns.as_mut_ptr(),
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            last_values.as_ptr(),
        );
    }

    Ok((advantages, returns))
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn compute_gae(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let expected_len = num_steps * num_envs;
    if rewards.len() != expected_len || values.len() != expected_len || dones.len() != expected_len
    {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: rewards.len().min(values.len()).min(dones.len()),
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

    // Fallback scalar implementation
    for env in 0..num_envs {
        let mut last_gae = 0.0f32;
        let mut next_value = last_values[env];

        for step in (0..num_steps).rev() {
            let idx = step * num_envs + env;
            let mask = 1.0 - dones[idx];

            let delta = rewards[idx] + gamma * next_value * mask - values[idx];
            last_gae = delta + gamma * gae_lambda * mask * last_gae;

            advantages[idx] = last_gae;
            returns[idx] = last_gae + values[idx];
            next_value = values[idx];
        }
    }

    Ok((advantages, returns))
}

// ============================================================================
// Softmax
// ============================================================================

/// Compute softmax using NEON SIMD.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn softmax(input: &[f32]) -> Result<Vec<f32>> {
    if !is_neon_available() {
        return Err(SimdError::NeonNotAvailable);
    }

    let mut output = vec![0.0f32; input.len()];
    unsafe {
        // Single row softmax: batch_size=1, num_actions=input.len()
        ffi::softmax_neon(input.as_ptr(), output.as_mut_ptr(), 1, input.len());
    }
    Ok(output)
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn softmax(input: &[f32]) -> Result<Vec<f32>> {
    let max_val = input.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = input.iter().map(|&x| (x - max_val).exp()).sum();
    let output: Vec<f32> = input.iter().map(|&x| (x - max_val).exp() / exp_sum).collect();
    Ok(output)
}

/// Batch softmax for multiple samples.
#[cfg(all(target_arch = "aarch64", feature = "simd"))]
pub fn softmax_batch(input: &[f32], batch_size: usize, num_classes: usize) -> Result<Vec<f32>> {
    if input.len() != batch_size * num_classes {
        return Err(SimdError::SizeMismatch {
            expected: batch_size * num_classes,
            actual: input.len(),
        });
    }

    if !is_neon_available() {
        return Err(SimdError::NeonNotAvailable);
    }

    let mut output = vec![0.0f32; input.len()];
    unsafe {
        ffi::softmax_neon(
            input.as_ptr(),
            output.as_mut_ptr(),
            batch_size,
            num_classes,
        );
    }
    Ok(output)
}

#[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
pub fn softmax_batch(input: &[f32], batch_size: usize, num_classes: usize) -> Result<Vec<f32>> {
    if input.len() != batch_size * num_classes {
        return Err(SimdError::SizeMismatch {
            expected: batch_size * num_classes,
            actual: input.len(),
        });
    }

    let mut output = vec![0.0f32; input.len()];
    for b in 0..batch_size {
        let start = b * num_classes;
        let slice = &input[start..start + num_classes];
        let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = slice.iter().map(|&x| (x - max_val).exp()).sum();
        for (i, &x) in slice.iter().enumerate() {
            output[start + i] = (x - max_val).exp() / exp_sum;
        }
    }
    Ok(output)
}

// ============================================================================
// Categorical Sampling
// ============================================================================

/// Categorical sampler using Gumbel-max trick with NEON SIMD.
pub struct CategoricalSampler {
    rng_state: Vec<u64>,
}

impl CategoricalSampler {
    /// Create a new categorical sampler.
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    pub fn new(seed: u64) -> Self {
        let mut rng_state = vec![0u64; CATEGORICAL_RNG_STATE_SIZE];
        unsafe {
            ffi::rng_init(rng_state.as_mut_ptr(), seed);
        }
        Self { rng_state }
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
    pub fn new(seed: u64) -> Self {
        Self {
            rng_state: vec![seed; CATEGORICAL_RNG_STATE_SIZE],
        }
    }

    /// Sample categorical actions from logits using Gumbel-max trick.
    ///
    /// # Arguments
    ///
    /// * `logits` - Unnormalized log probabilities [batch_size, num_classes]
    /// * `batch_size` - Number of samples
    /// * `num_classes` - Number of discrete actions
    ///
    /// # Returns
    ///
    /// Sampled action indices [batch_size]
    #[cfg(all(target_arch = "aarch64", feature = "simd"))]
    pub fn sample(&mut self, logits: &[f32], batch_size: usize, num_classes: usize) -> Result<Vec<u32>> {
        if logits.len() != batch_size * num_classes {
            return Err(SimdError::SizeMismatch {
                expected: batch_size * num_classes,
                actual: logits.len(),
            });
        }

        if !is_neon_available() {
            return Err(SimdError::NeonNotAvailable);
        }

        let mut output = vec![0u32; batch_size];
        unsafe {
            ffi::categorical_sample_gumbel_neon(
                logits.as_ptr(),
                output.as_mut_ptr(),
                batch_size,
                num_classes,
                self.rng_state.as_mut_ptr(),
            );
        }
        Ok(output)
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "simd")))]
    pub fn sample(&mut self, logits: &[f32], batch_size: usize, num_classes: usize) -> Result<Vec<u32>> {
        if logits.len() != batch_size * num_classes {
            return Err(SimdError::SizeMismatch {
                expected: batch_size * num_classes,
                actual: logits.len(),
            });
        }

        use rand::{RngCore, SeedableRng};
        use rand_distr::{Distribution, Uniform};

        let mut rng = rand::rngs::StdRng::seed_from_u64(self.rng_state[0]);
        let uniform = Uniform::new(0.0f32, 1.0f32);

        let mut output = vec![0u32; batch_size];

        for b in 0..batch_size {
            let start = b * num_classes;

            // Gumbel-max trick
            let mut best_idx = 0;
            let mut best_val = f32::NEG_INFINITY;

            for c in 0..num_classes {
                let u = uniform.sample(&mut rng);
                let gumbel = -(-u.ln()).ln();
                let score = logits[start + c] + gumbel;
                if score > best_val {
                    best_val = score;
                    best_idx = c as u32;
                }
            }
            output[b] = best_idx;
        }

        self.rng_state[0] = rng.next_u64();
        Ok(output)
    }

    /// Sample and compute log probabilities.
    pub fn sample_with_log_prob(
        &mut self,
        logits: &[f32],
        batch_size: usize,
        num_classes: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        // Sample actions
        let actions = self.sample(logits, batch_size, num_classes)?;

        // Compute log probabilities using log-softmax
        let mut log_probs = vec![0.0f32; batch_size];
        for b in 0..batch_size {
            let start = b * num_classes;
            let slice = &logits[start..start + num_classes];
            let max_val = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let log_sum_exp: f32 = slice.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;
            log_probs[b] = logits[start + actions[b] as usize] - log_sum_exp;
        }

        Ok((actions, log_probs))
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gaussian_sampler() {
        let mut sampler = GaussianSampler::new(42);
        let mean = vec![0.0f32; 64];
        let std = vec![1.0f32; 64];

        let samples = sampler.sample(&mean, &std).unwrap();
        assert_eq!(samples.len(), 64);
    }

    #[test]
    fn test_gae_computation() {
        let num_steps = 128;
        let num_envs = 4;
        let rewards = vec![1.0f32; num_steps * num_envs];
        let values = vec![0.5f32; num_steps * num_envs];
        let dones = vec![0.0f32; num_steps * num_envs];
        let last_values = vec![0.5f32; num_envs];

        let (advantages, returns) =
            compute_gae(&rewards, &values, &dones, num_steps, num_envs, 0.99, 0.95, &last_values)
                .unwrap();

        assert_eq!(advantages.len(), num_steps * num_envs);
        assert_eq!(returns.len(), num_steps * num_envs);
    }

    #[test]
    fn test_softmax() {
        let input = vec![1.0f32, 2.0, 3.0, 4.0];
        let output = softmax(&input).unwrap();

        assert_eq!(output.len(), 4);
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_categorical_sampler() {
        let mut sampler = CategoricalSampler::new(42);
        let logits = vec![0.0f32; 4 * 10]; // 4 samples, 10 classes

        let actions = sampler.sample(&logits, 4, 10).unwrap();
        assert_eq!(actions.len(), 4);
        for &a in &actions {
            assert!(a < 10);
        }
    }

    #[test]
    fn test_gather_batch() {
        let dim = 16;
        let capacity = 1000;
        let batch_size = 32;

        let src: Vec<f32> = (0..capacity * dim).map(|i| i as f32).collect();
        let indices: Vec<usize> = (0..batch_size).map(|i| i * 10).collect();

        let result = gather_batch_f32(&src, &indices, dim, capacity).unwrap();
        assert_eq!(result.len(), batch_size * dim);

        // Verify first element
        assert_eq!(result[0], src[indices[0] * dim]);
    }
}
