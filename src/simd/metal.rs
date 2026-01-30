//! Metal GPU integration for Octane
//!
//! This module provides GPU-accelerated operations using Apple's Metal API.
//! It includes compute shaders for common RL operations:
//!
//! - Gaussian log probability computation
//! - PPO loss calculation (clipped surrogate + value loss + entropy bonus)
//! - Batch matrix operations
//! - Softmax and categorical sampling
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::simd::metal::MetalContext;
//!
//! let ctx = MetalContext::new()?;
//!
//! // Compute Gaussian log probabilities on GPU
//! let log_probs = ctx.gaussian_log_prob(&actions, &means, &log_stds)?;
//!
//! // Compute PPO loss
//! let (policy_loss, value_loss, entropy) = ctx.ppo_loss(
//!     &log_probs, &old_log_probs, &advantages, &values, &returns,
//!     clip_range, vf_coef, ent_coef,
//! )?;
//! ```

#![allow(unsafe_code)]

use thiserror::Error;

#[cfg(all(target_os = "macos", feature = "metal"))]
use ::metal::{
    Buffer, CommandQueue, ComputePipelineState, Device, Library, MTLResourceOptions, MTLSize,
};

/// Errors that can occur in Metal operations.
#[derive(Debug, Error)]
pub enum MetalError {
    /// Metal device not available.
    #[error("Metal device not available")]
    DeviceNotAvailable,

    /// Failed to create Metal library.
    #[error("Failed to create Metal library: {0}")]
    LibraryCreationFailed(String),

    /// Failed to create compute pipeline.
    #[error("Failed to create compute pipeline for function '{0}': {1}")]
    PipelineCreationFailed(String, String),

    /// Buffer creation failed.
    #[error("Failed to create Metal buffer")]
    BufferCreationFailed,

    /// Command execution failed.
    #[error("Command execution failed: {0}")]
    ExecutionFailed(String),

    /// Invalid buffer size.
    #[error("Invalid buffer size: expected {expected}, got {actual}")]
    InvalidBufferSize {
        /// Expected buffer size.
        expected: usize,
        /// Actual buffer size.
        actual: usize,
    },

    /// Shader compilation failed.
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationFailed(String),
}

/// Result type for Metal operations.
pub type Result<T> = std::result::Result<T, MetalError>;

/// Metal compute shaders source code.
const METAL_SHADERS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Constants
constant float LOG_2PI = 1.8378770664093453f;
constant float EPSILON = 1e-8f;

// ============================================================================
// Gaussian Distribution Operations
// ============================================================================

/// Compute log probability of Gaussian distribution
/// log_prob = -0.5 * (log(2*pi) + 2*log_std + ((x - mean) / exp(log_std))^2)
kernel void gaussian_log_prob(
    device const float* actions [[buffer(0)]],
    device const float* means [[buffer(1)]],
    device const float* log_stds [[buffer(2)]],
    device float* log_probs [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float action = actions[id];
    float mean = means[id];
    float log_std = log_stds[id];

    float std = exp(log_std);
    float z = (action - mean) / (std + EPSILON);

    log_probs[id] = -0.5f * (LOG_2PI + 2.0f * log_std + z * z);
}

/// Compute entropy of Gaussian distribution
/// entropy = 0.5 * (1 + log(2*pi) + 2*log_std)
kernel void gaussian_entropy(
    device const float* log_stds [[buffer(0)]],
    device float* entropy [[buffer(1)]],
    uint id [[thread_position_in_grid]]
) {
    entropy[id] = 0.5f * (1.0f + LOG_2PI + 2.0f * log_stds[id]);
}

// ============================================================================
// PPO Loss Computation
// ============================================================================

/// PPO clipped surrogate loss per sample
/// ratio = exp(log_prob - old_log_prob)
/// clipped_ratio = clamp(ratio, 1 - clip_range, 1 + clip_range)
/// loss = -min(ratio * advantage, clipped_ratio * advantage)
kernel void ppo_policy_loss(
    device const float* log_probs [[buffer(0)]],
    device const float* old_log_probs [[buffer(1)]],
    device const float* advantages [[buffer(2)]],
    device float* losses [[buffer(3)]],
    device float* clip_fractions [[buffer(4)]],
    constant float& clip_range [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    float ratio = exp(log_probs[id] - old_log_probs[id]);
    float advantage = advantages[id];

    float clipped_ratio = clamp(ratio, 1.0f - clip_range, 1.0f + clip_range);

    float surr1 = ratio * advantage;
    float surr2 = clipped_ratio * advantage;

    losses[id] = -min(surr1, surr2);
    clip_fractions[id] = (ratio < 1.0f - clip_range || ratio > 1.0f + clip_range) ? 1.0f : 0.0f;
}

/// Value function loss (optionally clipped)
/// loss = 0.5 * (value - return)^2  or clipped version
kernel void ppo_value_loss(
    device const float* values [[buffer(0)]],
    device const float* old_values [[buffer(1)]],
    device const float* returns [[buffer(2)]],
    device float* losses [[buffer(3)]],
    constant float& clip_range [[buffer(4)]],
    constant int& use_clipping [[buffer(5)]],
    uint id [[thread_position_in_grid]]
) {
    float value = values[id];
    float old_value = old_values[id];
    float ret = returns[id];

    if (use_clipping) {
        float clipped_value = old_value + clamp(value - old_value, -clip_range, clip_range);
        float loss1 = (value - ret) * (value - ret);
        float loss2 = (clipped_value - ret) * (clipped_value - ret);
        losses[id] = 0.5f * max(loss1, loss2);
    } else {
        float diff = value - ret;
        losses[id] = 0.5f * diff * diff;
    }
}

/// Combined PPO loss computation
kernel void ppo_combined_loss(
    device const float* log_probs [[buffer(0)]],
    device const float* old_log_probs [[buffer(1)]],
    device const float* advantages [[buffer(2)]],
    device const float* values [[buffer(3)]],
    device const float* returns [[buffer(4)]],
    device const float* entropy [[buffer(5)]],
    device float* policy_losses [[buffer(6)]],
    device float* value_losses [[buffer(7)]],
    device float* total_losses [[buffer(8)]],
    constant float& clip_range [[buffer(9)]],
    constant float& vf_coef [[buffer(10)]],
    constant float& ent_coef [[buffer(11)]],
    uint id [[thread_position_in_grid]]
) {
    // Policy loss
    float ratio = exp(log_probs[id] - old_log_probs[id]);
    float advantage = advantages[id];
    float clipped_ratio = clamp(ratio, 1.0f - clip_range, 1.0f + clip_range);
    float policy_loss = -min(ratio * advantage, clipped_ratio * advantage);

    // Value loss
    float diff = values[id] - returns[id];
    float value_loss = 0.5f * diff * diff;

    // Total loss
    policy_losses[id] = policy_loss;
    value_losses[id] = value_loss;
    total_losses[id] = policy_loss + vf_coef * value_loss - ent_coef * entropy[id];
}

// ============================================================================
// Softmax and Categorical Sampling
// ============================================================================

/// Numerically stable softmax (single row)
/// For batch processing, call with appropriate offsets
kernel void softmax_row(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device float* max_val [[buffer(2)]],
    device float* sum_exp [[buffer(3)]],
    constant uint& num_classes [[buffer(4)]],
    uint batch_id [[threadgroup_position_in_grid]],
    uint local_id [[thread_position_in_threadgroup]],
    uint local_size [[threads_per_threadgroup]]
) {
    uint offset = batch_id * num_classes;

    // Find max (parallel reduction would be better for large num_classes)
    if (local_id == 0) {
        float m = input[offset];
        for (uint i = 1; i < num_classes; i++) {
            m = max(m, input[offset + i]);
        }
        max_val[batch_id] = m;
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Compute exp and sum
    if (local_id == 0) {
        float m = max_val[batch_id];
        float s = 0.0f;
        for (uint i = 0; i < num_classes; i++) {
            float e = exp(input[offset + i] - m);
            output[offset + i] = e;
            s += e;
        }
        sum_exp[batch_id] = s;
    }

    threadgroup_barrier(mem_flags::mem_device);

    // Normalize
    if (local_id == 0) {
        float s = sum_exp[batch_id];
        for (uint i = 0; i < num_classes; i++) {
            output[offset + i] /= s;
        }
    }
}

/// Argmax for categorical sampling
kernel void argmax_row(
    device const float* scores [[buffer(0)]],
    device int* indices [[buffer(1)]],
    constant uint& num_classes [[buffer(2)]],
    uint batch_id [[thread_position_in_grid]]
) {
    uint offset = batch_id * num_classes;

    int best_idx = 0;
    float best_val = scores[offset];

    for (uint i = 1; i < num_classes; i++) {
        if (scores[offset + i] > best_val) {
            best_val = scores[offset + i];
            best_idx = int(i);
        }
    }

    indices[batch_id] = best_idx;
}

// ============================================================================
// GAE Computation
// ============================================================================

/// Compute GAE for a single environment (sequential due to data dependency)
/// Each thread handles one environment across all timesteps
kernel void gae_compute(
    device const float* rewards [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const float* dones [[buffer(2)]],
    device float* advantages [[buffer(3)]],
    device float* returns [[buffer(4)]],
    device const float* last_values [[buffer(5)]],
    constant uint& num_steps [[buffer(6)]],
    constant uint& num_envs [[buffer(7)]],
    constant float& gamma [[buffer(8)]],
    constant float& gae_lambda [[buffer(9)]],
    uint env_id [[thread_position_in_grid]]
) {
    float last_gae = 0.0f;
    float next_value = last_values[env_id];

    // Backward pass through time
    for (int step = int(num_steps) - 1; step >= 0; step--) {
        uint idx = uint(step) * num_envs + env_id;

        float mask = 1.0f - dones[idx];
        float delta = rewards[idx] + gamma * next_value * mask - values[idx];
        last_gae = delta + gamma * gae_lambda * mask * last_gae;

        advantages[idx] = last_gae;
        returns[idx] = last_gae + values[idx];
        next_value = values[idx];
    }
}

// ============================================================================
// Batch Normalization Running Stats
// ============================================================================

/// Update running mean and variance (Welford's algorithm)
kernel void update_running_stats(
    device float* running_mean [[buffer(0)]],
    device float* running_var [[buffer(1)]],
    device const float* batch_mean [[buffer(2)]],
    device const float* batch_var [[buffer(3)]],
    constant float& momentum [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    running_mean[id] = (1.0f - momentum) * running_mean[id] + momentum * batch_mean[id];
    running_var[id] = (1.0f - momentum) * running_var[id] + momentum * batch_var[id];
}

/// Normalize observations using running stats
kernel void normalize_obs(
    device const float* obs [[buffer(0)]],
    device float* normalized [[buffer(1)]],
    device const float* mean [[buffer(2)]],
    device const float* var [[buffer(3)]],
    constant float& clip_range [[buffer(4)]],
    uint id [[thread_position_in_grid]]
) {
    float std = sqrt(var[id % 1024] + EPSILON);  // Assuming max obs_dim of 1024
    float norm = (obs[id] - mean[id % 1024]) / std;
    normalized[id] = clamp(norm, -clip_range, clip_range);
}
"#;

/// Metal compute context for GPU operations.
#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct MetalContext {
    /// Metal device handle.
    device: Device,
    /// Command queue for GPU operations.
    queue: CommandQueue,
    /// Compiled shader library.
    library: Library,
    /// Pipeline states for each kernel.
    pipelines: MetalPipelines,
}

/// Pre-compiled compute pipeline states.
#[cfg(all(target_os = "macos", feature = "metal"))]
struct MetalPipelines {
    gaussian_log_prob: ComputePipelineState,
    gaussian_entropy: ComputePipelineState,
    ppo_policy_loss: ComputePipelineState,
    ppo_value_loss: ComputePipelineState,
    ppo_combined_loss: ComputePipelineState,
    softmax_row: ComputePipelineState,
    argmax_row: ComputePipelineState,
    gae_compute: ComputePipelineState,
    normalize_obs: ComputePipelineState,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalContext {
    /// Create a new Metal context with compiled shaders.
    pub fn new() -> Result<Self> {
        let device = Device::system_default().ok_or(MetalError::DeviceNotAvailable)?;

        let queue = device.new_command_queue();

        // Compile shaders from source
        let library = device
            .new_library_with_source(METAL_SHADERS, &::metal::CompileOptions::new())
            .map_err(|e| MetalError::ShaderCompilationFailed(e.to_string()))?;

        // Create pipeline states for each kernel
        let pipelines = Self::create_pipelines(&device, &library)?;

        Ok(Self {
            device,
            queue,
            library,
            pipelines,
        })
    }

    /// Create compute pipelines for all kernels.
    fn create_pipelines(device: &Device, library: &Library) -> Result<MetalPipelines> {
        let create_pipeline = |name: &str| -> Result<ComputePipelineState> {
            let function = library
                .get_function(name, None)
                .map_err(|e| MetalError::PipelineCreationFailed(name.to_string(), e.to_string()))?;

            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| MetalError::PipelineCreationFailed(name.to_string(), e.to_string()))
        };

        Ok(MetalPipelines {
            gaussian_log_prob: create_pipeline("gaussian_log_prob")?,
            gaussian_entropy: create_pipeline("gaussian_entropy")?,
            ppo_policy_loss: create_pipeline("ppo_policy_loss")?,
            ppo_value_loss: create_pipeline("ppo_value_loss")?,
            ppo_combined_loss: create_pipeline("ppo_combined_loss")?,
            softmax_row: create_pipeline("softmax_row")?,
            argmax_row: create_pipeline("argmax_row")?,
            gae_compute: create_pipeline("gae_compute")?,
            normalize_obs: create_pipeline("normalize_obs")?,
        })
    }

    /// Create a Metal buffer from a slice.
    fn create_buffer<T>(&self, data: &[T]) -> Result<Buffer> {
        let size = std::mem::size_of_val(data) as u64;
        let buffer = self.device.new_buffer_with_data(
            data.as_ptr() as *const _,
            size,
            MTLResourceOptions::StorageModeShared,
        );
        Ok(buffer)
    }

    /// Create an empty Metal buffer.
    fn create_empty_buffer(&self, size: usize) -> Result<Buffer> {
        let buffer = self
            .device
            .new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        Ok(buffer)
    }

    /// Read data from a Metal buffer.
    fn read_buffer<T: Clone>(&self, buffer: &Buffer, count: usize) -> Vec<T> {
        let ptr = buffer.contents() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, count) };
        slice.to_vec()
    }

    /// Compute Gaussian log probabilities on GPU.
    ///
    /// # Arguments
    ///
    /// * `actions` - Sampled actions
    /// * `means` - Distribution means
    /// * `log_stds` - Log standard deviations
    ///
    /// # Returns
    ///
    /// Log probabilities for each action
    pub fn gaussian_log_prob(
        &self,
        actions: &[f32],
        means: &[f32],
        log_stds: &[f32],
    ) -> Result<Vec<f32>> {
        let n = actions.len();
        if means.len() != n || log_stds.len() != n {
            return Err(MetalError::InvalidBufferSize {
                expected: n,
                actual: means.len().min(log_stds.len()),
            });
        }

        let actions_buf = self.create_buffer(actions)?;
        let means_buf = self.create_buffer(means)?;
        let log_stds_buf = self.create_buffer(log_stds)?;
        let output_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.gaussian_log_prob);
        encoder.set_buffer(0, Some(&actions_buf), 0);
        encoder.set_buffer(1, Some(&means_buf), 0);
        encoder.set_buffer(2, Some(&log_stds_buf), 0);
        encoder.set_buffer(3, Some(&output_buf), 0);

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n as u64 + 255) / 256, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(self.read_buffer(&output_buf, n))
    }

    /// Compute Gaussian entropy on GPU.
    pub fn gaussian_entropy(&self, log_stds: &[f32]) -> Result<Vec<f32>> {
        let n = log_stds.len();

        let log_stds_buf = self.create_buffer(log_stds)?;
        let output_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.gaussian_entropy);
        encoder.set_buffer(0, Some(&log_stds_buf), 0);
        encoder.set_buffer(1, Some(&output_buf), 0);

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n as u64 + 255) / 256, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(self.read_buffer(&output_buf, n))
    }

    /// Compute PPO policy loss on GPU.
    ///
    /// Returns (losses, clip_fractions) for each sample.
    pub fn ppo_policy_loss(
        &self,
        log_probs: &[f32],
        old_log_probs: &[f32],
        advantages: &[f32],
        clip_range: f32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = log_probs.len();
        if old_log_probs.len() != n || advantages.len() != n {
            return Err(MetalError::InvalidBufferSize {
                expected: n,
                actual: old_log_probs.len().min(advantages.len()),
            });
        }

        let log_probs_buf = self.create_buffer(log_probs)?;
        let old_log_probs_buf = self.create_buffer(old_log_probs)?;
        let advantages_buf = self.create_buffer(advantages)?;
        let losses_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;
        let clip_fracs_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;
        let clip_range_buf = self.create_buffer(&[clip_range])?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.ppo_policy_loss);
        encoder.set_buffer(0, Some(&log_probs_buf), 0);
        encoder.set_buffer(1, Some(&old_log_probs_buf), 0);
        encoder.set_buffer(2, Some(&advantages_buf), 0);
        encoder.set_buffer(3, Some(&losses_buf), 0);
        encoder.set_buffer(4, Some(&clip_fracs_buf), 0);
        encoder.set_buffer(5, Some(&clip_range_buf), 0);

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n as u64 + 255) / 256, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((
            self.read_buffer(&losses_buf, n),
            self.read_buffer(&clip_fracs_buf, n),
        ))
    }

    /// Compute PPO value loss on GPU.
    pub fn ppo_value_loss(
        &self,
        values: &[f32],
        old_values: &[f32],
        returns: &[f32],
        clip_range: f32,
        use_clipping: bool,
    ) -> Result<Vec<f32>> {
        let n = values.len();
        if old_values.len() != n || returns.len() != n {
            return Err(MetalError::InvalidBufferSize {
                expected: n,
                actual: old_values.len().min(returns.len()),
            });
        }

        let values_buf = self.create_buffer(values)?;
        let old_values_buf = self.create_buffer(old_values)?;
        let returns_buf = self.create_buffer(returns)?;
        let losses_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;
        let clip_range_buf = self.create_buffer(&[clip_range])?;
        let use_clip_buf = self.create_buffer(&[if use_clipping { 1i32 } else { 0i32 }])?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.ppo_value_loss);
        encoder.set_buffer(0, Some(&values_buf), 0);
        encoder.set_buffer(1, Some(&old_values_buf), 0);
        encoder.set_buffer(2, Some(&returns_buf), 0);
        encoder.set_buffer(3, Some(&losses_buf), 0);
        encoder.set_buffer(4, Some(&clip_range_buf), 0);
        encoder.set_buffer(5, Some(&use_clip_buf), 0);

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n as u64 + 255) / 256, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(self.read_buffer(&losses_buf, n))
    }

    /// Compute full PPO loss (policy + value + entropy) on GPU.
    ///
    /// # Arguments
    ///
    /// * `log_probs` - Current policy log probabilities
    /// * `old_log_probs` - Old policy log probabilities
    /// * `advantages` - Normalized advantages
    /// * `values` - Current value predictions
    /// * `returns` - Computed returns
    /// * `entropy` - Policy entropy
    /// * `clip_range` - PPO clipping range
    /// * `vf_coef` - Value function loss coefficient
    /// * `ent_coef` - Entropy bonus coefficient
    ///
    /// # Returns
    ///
    /// Tuple of (policy_losses, value_losses, total_losses)
    pub fn ppo_loss(
        &self,
        log_probs: &[f32],
        old_log_probs: &[f32],
        advantages: &[f32],
        values: &[f32],
        returns: &[f32],
        entropy: &[f32],
        clip_range: f32,
        vf_coef: f32,
        ent_coef: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        let n = log_probs.len();

        let log_probs_buf = self.create_buffer(log_probs)?;
        let old_log_probs_buf = self.create_buffer(old_log_probs)?;
        let advantages_buf = self.create_buffer(advantages)?;
        let values_buf = self.create_buffer(values)?;
        let returns_buf = self.create_buffer(returns)?;
        let entropy_buf = self.create_buffer(entropy)?;

        let policy_losses_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;
        let value_losses_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;
        let total_losses_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;

        let clip_range_buf = self.create_buffer(&[clip_range])?;
        let vf_coef_buf = self.create_buffer(&[vf_coef])?;
        let ent_coef_buf = self.create_buffer(&[ent_coef])?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.ppo_combined_loss);
        encoder.set_buffer(0, Some(&log_probs_buf), 0);
        encoder.set_buffer(1, Some(&old_log_probs_buf), 0);
        encoder.set_buffer(2, Some(&advantages_buf), 0);
        encoder.set_buffer(3, Some(&values_buf), 0);
        encoder.set_buffer(4, Some(&returns_buf), 0);
        encoder.set_buffer(5, Some(&entropy_buf), 0);
        encoder.set_buffer(6, Some(&policy_losses_buf), 0);
        encoder.set_buffer(7, Some(&value_losses_buf), 0);
        encoder.set_buffer(8, Some(&total_losses_buf), 0);
        encoder.set_buffer(9, Some(&clip_range_buf), 0);
        encoder.set_buffer(10, Some(&vf_coef_buf), 0);
        encoder.set_buffer(11, Some(&ent_coef_buf), 0);

        let thread_group_size = MTLSize::new(256, 1, 1);
        let thread_groups = MTLSize::new((n as u64 + 255) / 256, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((
            self.read_buffer(&policy_losses_buf, n),
            self.read_buffer(&value_losses_buf, n),
            self.read_buffer(&total_losses_buf, n),
        ))
    }

    /// Compute GAE on GPU.
    ///
    /// Note: This is parallelized across environments, not timesteps,
    /// due to the sequential nature of GAE computation.
    pub fn compute_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        dones: &[f32],
        num_steps: usize,
        num_envs: usize,
        gamma: f32,
        gae_lambda: f32,
        last_values: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let n = num_steps * num_envs;

        let rewards_buf = self.create_buffer(rewards)?;
        let values_buf = self.create_buffer(values)?;
        let dones_buf = self.create_buffer(dones)?;
        let last_values_buf = self.create_buffer(last_values)?;

        let advantages_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;
        let returns_buf = self.create_empty_buffer(n * std::mem::size_of::<f32>())?;

        let num_steps_buf = self.create_buffer(&[num_steps as u32])?;
        let num_envs_buf = self.create_buffer(&[num_envs as u32])?;
        let gamma_buf = self.create_buffer(&[gamma])?;
        let gae_lambda_buf = self.create_buffer(&[gae_lambda])?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.gae_compute);
        encoder.set_buffer(0, Some(&rewards_buf), 0);
        encoder.set_buffer(1, Some(&values_buf), 0);
        encoder.set_buffer(2, Some(&dones_buf), 0);
        encoder.set_buffer(3, Some(&advantages_buf), 0);
        encoder.set_buffer(4, Some(&returns_buf), 0);
        encoder.set_buffer(5, Some(&last_values_buf), 0);
        encoder.set_buffer(6, Some(&num_steps_buf), 0);
        encoder.set_buffer(7, Some(&num_envs_buf), 0);
        encoder.set_buffer(8, Some(&gamma_buf), 0);
        encoder.set_buffer(9, Some(&gae_lambda_buf), 0);

        // One thread per environment
        let thread_group_size = MTLSize::new(64.min(num_envs as u64), 1, 1);
        let thread_groups = MTLSize::new((num_envs as u64 + 63) / 64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok((
            self.read_buffer(&advantages_buf, n),
            self.read_buffer(&returns_buf, n),
        ))
    }

    /// Compute argmax for categorical sampling.
    pub fn argmax(
        &self,
        scores: &[f32],
        batch_size: usize,
        num_classes: usize,
    ) -> Result<Vec<i32>> {
        let scores_buf = self.create_buffer(scores)?;
        let indices_buf = self.create_empty_buffer(batch_size * std::mem::size_of::<i32>())?;
        let num_classes_buf = self.create_buffer(&[num_classes as u32])?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();

        encoder.set_compute_pipeline_state(&self.pipelines.argmax_row);
        encoder.set_buffer(0, Some(&scores_buf), 0);
        encoder.set_buffer(1, Some(&indices_buf), 0);
        encoder.set_buffer(2, Some(&num_classes_buf), 0);

        let thread_group_size = MTLSize::new(64.min(batch_size as u64), 1, 1);
        let thread_groups = MTLSize::new((batch_size as u64 + 63) / 64, 1, 1);
        encoder.dispatch_thread_groups(thread_groups, thread_group_size);
        encoder.end_encoding();

        command_buffer.commit();
        command_buffer.wait_until_completed();

        Ok(self.read_buffer(&indices_buf, batch_size))
    }

    /// Get Metal device name.
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Check if Metal is available.
    pub fn is_available() -> bool {
        Device::system_default().is_some()
    }
}

// ============================================================================
// Non-macOS Stub Implementation
// ============================================================================

/// Stub Metal context for non-macOS platforms or when metal feature is disabled.
#[cfg(not(all(target_os = "macos", feature = "metal")))]
pub struct MetalContext;

#[cfg(not(all(target_os = "macos", feature = "metal")))]
impl MetalContext {
    /// Create a new Metal context (stub - always fails on non-macOS).
    pub fn new() -> Result<Self> {
        Err(MetalError::DeviceNotAvailable)
    }

    /// Check if Metal is available.
    pub fn is_available() -> bool {
        false
    }

    /// Compute Gaussian log probabilities (stub).
    pub fn gaussian_log_prob(
        &self,
        _actions: &[f32],
        _means: &[f32],
        _log_stds: &[f32],
    ) -> Result<Vec<f32>> {
        Err(MetalError::DeviceNotAvailable)
    }

    /// Compute PPO loss (stub).
    pub fn ppo_loss(
        &self,
        _log_probs: &[f32],
        _old_log_probs: &[f32],
        _advantages: &[f32],
        _values: &[f32],
        _returns: &[f32],
        _entropy: &[f32],
        _clip_range: f32,
        _vf_coef: f32,
        _ent_coef: f32,
    ) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        Err(MetalError::DeviceNotAvailable)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(all(test, target_os = "macos", feature = "metal"))]
mod tests {
    use super::*;

    #[test]
    fn test_metal_available() {
        assert!(MetalContext::is_available());
    }

    #[test]
    fn test_gaussian_log_prob() {
        if let Ok(ctx) = MetalContext::new() {
            let actions = vec![0.0f32, 0.5, 1.0, -0.5];
            let means = vec![0.0f32, 0.0, 0.0, 0.0];
            let log_stds = vec![0.0f32, 0.0, 0.0, 0.0]; // std = 1.0

            let log_probs = ctx.gaussian_log_prob(&actions, &means, &log_stds).unwrap();

            assert_eq!(log_probs.len(), 4);
            // log_prob at mean should be highest
            assert!(log_probs[0] > log_probs[2]);
        }
    }

    #[test]
    fn test_ppo_policy_loss() {
        if let Ok(ctx) = MetalContext::new() {
            let log_probs = vec![-1.0f32, -1.5, -2.0];
            let old_log_probs = vec![-1.0f32, -1.0, -1.0];
            let advantages = vec![1.0f32, -1.0, 0.5];

            let (losses, clip_fracs) = ctx
                .ppo_policy_loss(&log_probs, &old_log_probs, &advantages, 0.2)
                .unwrap();

            assert_eq!(losses.len(), 3);
            assert_eq!(clip_fracs.len(), 3);
        }
    }

    #[test]
    fn test_gae_compute() {
        if let Ok(ctx) = MetalContext::new() {
            let num_steps = 128;
            let num_envs = 4;

            let rewards = vec![1.0f32; num_steps * num_envs];
            let values = vec![0.5f32; num_steps * num_envs];
            let dones = vec![0.0f32; num_steps * num_envs];
            let last_values = vec![0.5f32; num_envs];

            let (advantages, returns) = ctx
                .compute_gae(
                    &rewards,
                    &values,
                    &dones,
                    num_steps,
                    num_envs,
                    0.99,
                    0.95,
                    &last_values,
                )
                .unwrap();

            assert_eq!(advantages.len(), num_steps * num_envs);
            assert_eq!(returns.len(), num_steps * num_envs);
        }
    }
}
