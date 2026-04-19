//! Experience buffers for reinforcement learning.
//!
//! This module provides various buffer implementations for RL algorithms:
//!
//! - **RolloutBuffer**: For on-policy algorithms (PPO, A2C). Stores complete rollouts
//!   and computes GAE advantages.
//! - **ReplayBuffer**: For off-policy algorithms (DQN, DDPG, TD3, SAC). Ring buffer
//!   with uniform or prioritized sampling.
//! - **NStepReplayBuffer**: Extends ReplayBuffer with n-step return computation for
//!   improved sample efficiency.
//! - **HERBuffer**: Hindsight Experience Replay for goal-conditioned RL. Implements
//!   goal relabeling strategies (final, future, episode, random).
//! - **MmapReplayBuffer**: Memory-mapped buffer for large-scale storage (100M+
//!   transitions) without RAM pressure.
//!
//! # Segment Trees
//!
//! The `segment_tree` module provides efficient data structures for prioritized
//! experience replay:
//! - **SumTree**: O(log n) proportional sampling based on priorities.
//! - **MinTree**: O(log n) minimum priority tracking for importance sampling.

mod her;
mod mmap;
mod nstep;
mod replay;
mod segment_tree;

pub use her::{HERBatch, HERBuffer, HERConfig, HERStrategy, RewardFn};
pub use mmap::{MmapBufferConfig, MmapBufferStats, MmapReplayBuffer};
pub use nstep::{NStepConfig, NStepReplayBuffer};
pub use replay::{ReplayBatch, ReplayBuffer, ReplayBufferConfig, Transition};
pub use segment_tree::{MinTree, SumTree};

use crate::core::{Device, OctaneError, Result};
use candle_core::{DType, Tensor};
use rand::seq::SliceRandom;

/// A single batch of rollout data for mini-batch training.
#[derive(Debug)]
pub struct RolloutBatch {
    /// Observations [batch_size, ...obs_shape].
    pub observations: Tensor,
    /// Actions [batch_size, action_dim].
    pub actions: Tensor,
    /// Old log probabilities [batch_size].
    pub old_log_probs: Tensor,
    /// Advantages [batch_size] (normalized).
    pub advantages: Tensor,
    /// Returns (value targets) [batch_size].
    pub returns: Tensor,
    /// Old values [batch_size] (for value clipping).
    pub old_values: Tensor,
}

/// Pre-allocated rollout buffer for on-policy algorithms.
///
/// Stores transitions from vectorized environments and computes GAE advantages.
/// Uses flat `Vec<f32>` storage to avoid intermediate tensor allocations -
/// data is collected directly into pre-allocated vectors and converted to
/// tensors only once when the buffer is full.
///
/// # Layout
///
/// All tensors have shape `[buffer_size, num_envs, ...]` where:
/// - `buffer_size`: Number of steps to collect before each training update
/// - `num_envs`: Number of parallel environments
///
/// When sampling batches, these are flattened to `[buffer_size * num_envs, ...]`.
///
/// # Performance
///
/// This design eliminates the double-copy problem of `Vec<Tensor>` + `stack()`:
/// - Old: input → clone() → Vec<Tensor> → stack() → final Tensor (2 copies)
/// - New: input → extend() → Vec<f32> → from_vec() → final Tensor (1 copy)
pub struct RolloutBuffer {
    /// Number of steps per rollout.
    buffer_size: usize,
    /// Number of parallel environments.
    num_envs: usize,
    /// Observation shape (single env, single step).
    obs_shape: Vec<usize>,
    /// Total observation size (product of obs_shape).
    obs_size: usize,
    /// Action dimension.
    action_dim: usize,
    /// Device for tensor operations.
    device: Device,

    // Flat storage vectors (pre-allocated, single contiguous block)
    /// Observations collected during rollout [buffer_size * num_envs * obs_size].
    obs_flat: Vec<f32>,
    /// Actions collected during rollout [buffer_size * num_envs * action_dim].
    actions_flat: Vec<f32>,
    /// Rewards collected during rollout [buffer_size * num_envs].
    rewards_flat: Vec<f32>,
    /// Termination flags collected during rollout [buffer_size * num_envs].
    /// True episode endings (goal reached, failure, etc.) - bootstrap value should be 0.
    terminated_flat: Vec<f32>,
    /// Truncation flags collected during rollout [buffer_size * num_envs].
    /// Time-limit cutoffs - bootstrap value should be used (episode was cut short).
    truncated_flat: Vec<f32>,
    /// Value estimates collected during rollout [buffer_size * num_envs].
    values_flat: Vec<f32>,
    /// Log probabilities collected during rollout [buffer_size * num_envs].
    log_probs_flat: Vec<f32>,

    // Stacked tensors (built after rollout is complete)
    /// Observations [buffer_size, num_envs, ...obs_shape].
    observations: Option<Tensor>,
    /// Actions [buffer_size, num_envs, action_dim].
    actions: Option<Tensor>,
    /// Rewards [buffer_size, num_envs].
    rewards: Option<Tensor>,
    /// Termination flags [buffer_size, num_envs].
    terminated: Option<Tensor>,
    /// Truncation flags [buffer_size, num_envs].
    truncated: Option<Tensor>,
    /// Value estimates [buffer_size, num_envs].
    values: Option<Tensor>,
    /// Log probabilities [buffer_size, num_envs].
    log_probs: Option<Tensor>,

    // Computed after rollout
    /// Advantages [buffer_size, num_envs] (computed via GAE).
    advantages: Option<Tensor>,
    /// Returns [buffer_size, num_envs] (discounted rewards).
    returns: Option<Tensor>,

    /// Current position in buffer.
    pos: usize,
    /// Whether buffer is full and ready for training.
    full: bool,
}

impl RolloutBuffer {
    /// Create a new rollout buffer with pre-allocated storage.
    ///
    /// # Arguments
    ///
    /// * `buffer_size` - Number of steps to collect per environment
    /// * `num_envs` - Number of parallel environments
    /// * `obs_shape` - Shape of a single observation (e.g., `[84, 84, 4]` for Atari)
    /// * `action_dim` - Dimension of action vector (1 for discrete, N for continuous)
    /// * `device` - Device to allocate tensors on
    ///
    /// # Returns
    ///
    /// A new `RolloutBuffer` ready for collecting transitions.
    pub fn new(
        buffer_size: usize,
        num_envs: usize,
        obs_shape: &[usize],
        action_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        if buffer_size == 0 {
            return Err(OctaneError::InvalidConfig(
                "Buffer size must be positive".to_string(),
            ));
        }
        if num_envs == 0 {
            return Err(OctaneError::InvalidConfig(
                "Number of environments must be positive".to_string(),
            ));
        }

        // Compute flat sizes for pre-allocation
        let obs_size: usize = obs_shape.iter().product();
        let obs_capacity = buffer_size * num_envs * obs_size;
        let actions_capacity = buffer_size * num_envs * action_dim;
        let scalar_capacity = buffer_size * num_envs;

        Ok(Self {
            buffer_size,
            num_envs,
            obs_shape: obs_shape.to_vec(),
            obs_size,
            action_dim,
            device: *device,
            // Pre-allocate flat vectors with exact capacity (single allocation)
            obs_flat: Vec::with_capacity(obs_capacity),
            actions_flat: Vec::with_capacity(actions_capacity),
            rewards_flat: Vec::with_capacity(scalar_capacity),
            terminated_flat: Vec::with_capacity(scalar_capacity),
            truncated_flat: Vec::with_capacity(scalar_capacity),
            values_flat: Vec::with_capacity(scalar_capacity),
            log_probs_flat: Vec::with_capacity(scalar_capacity),
            // Stacked tensors (built later)
            observations: None,
            actions: None,
            rewards: None,
            terminated: None,
            truncated: None,
            values: None,
            log_probs: None,
            // Computed after rollout
            advantages: None,
            returns: None,
            pos: 0,
            full: false,
        })
    }

    /// Add a transition to the buffer.
    ///
    /// # Arguments
    ///
    /// * `obs` - Observations [num_envs, ...obs_shape]
    /// * `action` - Actions [num_envs, action_dim]
    /// * `reward` - Rewards [num_envs]
    /// * `terminated` - Termination flags [num_envs] (true episode endings)
    /// * `truncated` - Truncation flags [num_envs] (time-limit cutoffs)
    /// * `value` - Value estimates [num_envs]
    /// * `log_prob` - Log probabilities [num_envs]
    ///
    /// # Returns
    ///
    /// `Ok(())` on success, error if buffer is already full.
    ///
    /// # Note
    ///
    /// Distinguishing between termination and truncation is critical for correct
    /// value estimation. Terminations (goal/failure) should zero the bootstrap,
    /// while truncations (time limits) should bootstrap the value.
    pub fn add(
        &mut self,
        obs: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        terminated: &Tensor,
        truncated: &Tensor,
        value: &Tensor,
        log_prob: &Tensor,
    ) -> Result<()> {
        if self.full {
            return Err(OctaneError::Buffer(
                "Buffer is full. Call reset() before adding more transitions.".to_string(),
            ));
        }

        // Validate shapes
        self.validate_obs_shape(obs)?;
        self.validate_action_shape(action)?;
        self.validate_scalar_shape(reward)?;
        self.validate_scalar_shape(terminated)?;
        self.validate_scalar_shape(truncated)?;
        self.validate_scalar_shape(value)?;
        self.validate_scalar_shape(log_prob)?;

        // Flatten tensors and extend into pre-allocated vectors (single copy)
        let obs_data: Vec<f32> = obs.flatten_all()?.to_vec1()?;
        let action_data: Vec<f32> = action.flatten_all()?.to_vec1()?;
        let reward_data: Vec<f32> = reward.to_vec1()?;
        let terminated_data: Vec<f32> = terminated.to_vec1()?;
        let truncated_data: Vec<f32> = truncated.to_vec1()?;
        let value_data: Vec<f32> = value.to_vec1()?;
        let log_prob_data: Vec<f32> = log_prob.to_vec1()?;

        self.obs_flat.extend_from_slice(&obs_data);
        self.actions_flat.extend_from_slice(&action_data);
        self.rewards_flat.extend_from_slice(&reward_data);
        self.terminated_flat.extend_from_slice(&terminated_data);
        self.truncated_flat.extend_from_slice(&truncated_data);
        self.values_flat.extend_from_slice(&value_data);
        self.log_probs_flat.extend_from_slice(&log_prob_data);

        self.pos += 1;
        if self.pos >= self.buffer_size {
            self.full = true;
            // Build tensors from flat vectors
            self.finalize_collection()?;
        }

        Ok(())
    }

    /// Build tensors from flat vectors (zero-copy when possible).
    fn finalize_collection(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // Build observation shape: [buffer_size, num_envs, ...obs_shape]
        let mut obs_tensor_shape = vec![self.buffer_size, self.num_envs];
        obs_tensor_shape.extend_from_slice(&self.obs_shape);

        // Build action shape: [buffer_size, num_envs, action_dim]
        let actions_shape = vec![self.buffer_size, self.num_envs, self.action_dim];

        // Scalar shape: [buffer_size, num_envs]
        let scalar_shape = vec![self.buffer_size, self.num_envs];

        // Create tensors from flat vectors (moves data, no copy if same device)
        self.observations = Some(Tensor::from_vec(
            std::mem::take(&mut self.obs_flat),
            obs_tensor_shape.as_slice(),
            &candle_device,
        )?);
        self.actions = Some(Tensor::from_vec(
            std::mem::take(&mut self.actions_flat),
            actions_shape.as_slice(),
            &candle_device,
        )?);
        self.rewards = Some(Tensor::from_vec(
            std::mem::take(&mut self.rewards_flat),
            scalar_shape.as_slice(),
            &candle_device,
        )?);
        self.terminated = Some(Tensor::from_vec(
            std::mem::take(&mut self.terminated_flat),
            scalar_shape.as_slice(),
            &candle_device,
        )?);
        self.truncated = Some(Tensor::from_vec(
            std::mem::take(&mut self.truncated_flat),
            scalar_shape.as_slice(),
            &candle_device,
        )?);
        self.values = Some(Tensor::from_vec(
            std::mem::take(&mut self.values_flat),
            scalar_shape.as_slice(),
            &candle_device,
        )?);
        self.log_probs = Some(Tensor::from_vec(
            std::mem::take(&mut self.log_probs_flat),
            scalar_shape.as_slice(),
            &candle_device,
        )?);

        Ok(())
    }

    /// Compute returns and advantages using Generalized Advantage Estimation (GAE).
    ///
    /// This implements the GAE formula with correct truncation handling:
    /// ```text
    /// delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated) - V(s_t)
    /// A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
    /// ```
    ///
    /// **Critical: Truncation vs Termination**
    /// - `terminated`: True episode ending (goal/failure). Bootstrap value = 0.
    /// - `truncated`: Time-limit cutoff. Bootstrap value = V(s').
    ///
    /// This fixes the value function collapse bug where time-limit truncations
    /// were incorrectly treated as terminal states.
    ///
    /// # Arguments
    ///
    /// * `last_values` - Value estimates for the state after the last step [num_envs]
    /// * `gamma` - Discount factor (typically 0.99)
    /// * `gae_lambda` - GAE lambda parameter (typically 0.95)
    pub fn compute_returns_and_advantages(
        &mut self,
        last_values: &Tensor,
        gamma: f32,
        gae_lambda: f32,
    ) -> Result<()> {
        if !self.full {
            return Err(OctaneError::Buffer(
                "Buffer not full. Collect more transitions before computing advantages."
                    .to_string(),
            ));
        }

        let rewards = self
            .rewards
            .as_ref()
            .ok_or_else(|| OctaneError::Buffer("Rewards not available".to_string()))?;
        let terminated = self
            .terminated
            .as_ref()
            .ok_or_else(|| OctaneError::Buffer("Terminated flags not available".to_string()))?;
        let truncated = self
            .truncated
            .as_ref()
            .ok_or_else(|| OctaneError::Buffer("Truncated flags not available".to_string()))?;
        let values = self
            .values
            .as_ref()
            .ok_or_else(|| OctaneError::Buffer("Values not available".to_string()))?;

        let candle_device = self.device.to_candle()?;

        // Initialize advantage accumulator [num_envs]
        let mut last_gae_lam = Tensor::zeros(&[self.num_envs], DType::F32, &candle_device)?;

        // Pre-allocate advantages vector
        let mut advantages_vec: Vec<Tensor> = Vec::with_capacity(self.buffer_size);

        // Compute GAE in reverse order
        for step in (0..self.buffer_size).rev() {
            // Get tensors for this step
            let reward = rewards.get(step)?; // [num_envs]
            let value = values.get(step)?; // [num_envs]
            let term = terminated.get(step)?; // [num_envs]
            let trunc = truncated.get(step)?; // [num_envs]

            // Get next value: either from buffer or from last_values
            let next_value = if step == self.buffer_size - 1 {
                last_values.clone()
            } else {
                values.get(step + 1)?
            };

            // Compute done = terminated OR truncated for GAE propagation reset
            let done = term.maximum(&trunc)?;

            // **CRITICAL FIX**: Only mask bootstrap value for TERMINATIONS, not truncations
            // not_terminated = 1 - terminated (NOT 1 - done!)
            // This ensures we bootstrap value for truncated episodes
            let ones = Tensor::ones(&[self.num_envs], DType::F32, &candle_device)?;
            let not_terminated = ones.sub(&term)?;

            // delta = reward + gamma * next_value * (1 - terminated) - value
            // Truncations still bootstrap the value (episode was cut short artificially)
            let discounted_next_value = (&next_value * gamma as f64)?;
            let masked_next_value = (&discounted_next_value * &not_terminated)?;
            let delta = ((&reward + &masked_next_value)? - &value)?;

            // For GAE propagation, we reset at episode boundaries (terminated OR truncated)
            // We use the CURRENT step's done to reset the accumulator for the NEXT iteration
            // (which computes the previous timestep)
            let ones_again = Tensor::ones(&[self.num_envs], DType::F32, &candle_device)?;
            let not_done = ones_again.sub(&done)?;

            // A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
            // The done mask ensures GAE doesn't propagate across episode boundaries
            let gae_discount = (gamma * gae_lambda) as f64;
            let discounted_gae = (&last_gae_lam * gae_discount)?;
            let masked_gae = (&discounted_gae * &not_done)?;
            last_gae_lam = (&delta + &masked_gae)?;

            advantages_vec.push(last_gae_lam.clone());
        }

        // Reverse to get correct order
        advantages_vec.reverse();

        // Stack advantages [buffer_size, num_envs]
        let advantages = Tensor::stack(&advantages_vec, 0)?;

        // Returns = advantages + values
        let returns = advantages.add(values)?;

        self.advantages = Some(advantages);
        self.returns = Some(returns);

        Ok(())
    }

    /// Get randomized mini-batches for training.
    ///
    /// Flattens the buffer from `[buffer_size, num_envs, ...]` to
    /// `[buffer_size * num_envs, ...]` and creates shuffled mini-batches.
    ///
    /// Advantages are normalized within each batch for training stability.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Size of each mini-batch
    ///
    /// # Returns
    ///
    /// Vector of `RolloutBatch` structs for training.
    pub fn get_batches(&self, batch_size: usize) -> Result<Vec<RolloutBatch>> {
        if self.advantages.is_none() || self.returns.is_none() {
            return Err(OctaneError::Buffer(
                "Must call compute_returns_and_advantages before get_batches".to_string(),
            ));
        }

        let total_size = self.buffer_size * self.num_envs;

        if batch_size == 0 || batch_size > total_size {
            return Err(OctaneError::InvalidConfig(format!(
                "Batch size must be between 1 and {} (buffer_size * num_envs)",
                total_size
            )));
        }

        let observations = self.observations.as_ref().unwrap();
        let actions = self.actions.as_ref().unwrap();
        let log_probs = self.log_probs.as_ref().unwrap();
        let values = self.values.as_ref().unwrap();
        let advantages = self.advantages.as_ref().unwrap();
        let returns = self.returns.as_ref().unwrap();

        // Flatten all tensors to [total_size, ...]
        let flat_obs = self.flatten_buffer(observations)?;
        let flat_actions = self.flatten_buffer(actions)?;
        let flat_log_probs = self.flatten_scalar_buffer(log_probs)?;
        let flat_values = self.flatten_scalar_buffer(values)?;
        let flat_advantages = self.flatten_scalar_buffer(advantages)?;
        let flat_returns = self.flatten_scalar_buffer(returns)?;

        // Generate shuffled indices
        let mut indices: Vec<usize> = (0..total_size).collect();
        let mut rng = rand::thread_rng();
        indices.shuffle(&mut rng);

        // Create batches
        let num_batches = total_size / batch_size;
        let mut batches = Vec::with_capacity(num_batches);

        let candle_device = self.device.to_candle()?;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let batch_indices: Vec<u32> = indices[start..start + batch_size]
                .iter()
                .map(|&i| i as u32)
                .collect();

            // Create index tensor
            let idx_tensor = Tensor::from_slice(&batch_indices, (batch_size,), &candle_device)?;

            // Index into flattened tensors
            let batch_obs = flat_obs.index_select(&idx_tensor, 0)?;
            let batch_actions = flat_actions.index_select(&idx_tensor, 0)?;
            let batch_log_probs = flat_log_probs.index_select(&idx_tensor, 0)?;
            let batch_values = flat_values.index_select(&idx_tensor, 0)?;
            let batch_advantages = flat_advantages.index_select(&idx_tensor, 0)?;
            let batch_returns = flat_returns.index_select(&idx_tensor, 0)?;

            // Normalize advantages for training stability
            let normalized_advantages = self.normalize_advantages(&batch_advantages)?;

            batches.push(RolloutBatch {
                observations: batch_obs,
                actions: batch_actions,
                old_log_probs: batch_log_probs,
                advantages: normalized_advantages,
                returns: batch_returns,
                old_values: batch_values,
            });
        }

        Ok(batches)
    }

    /// Reset the buffer for a new rollout.
    ///
    /// This clears all stored data and resets the position counter.
    /// Re-allocates flat vectors with the same capacity for the next rollout.
    pub fn reset(&mut self) {
        self.pos = 0;
        self.full = false;

        // Re-allocate flat vectors (finalize_collection took ownership via mem::take)
        let obs_capacity = self.buffer_size * self.num_envs * self.obs_size;
        let actions_capacity = self.buffer_size * self.num_envs * self.action_dim;
        let scalar_capacity = self.buffer_size * self.num_envs;

        // Only re-allocate if vectors were consumed
        if self.obs_flat.capacity() == 0 {
            self.obs_flat = Vec::with_capacity(obs_capacity);
        } else {
            self.obs_flat.clear();
        }
        if self.actions_flat.capacity() == 0 {
            self.actions_flat = Vec::with_capacity(actions_capacity);
        } else {
            self.actions_flat.clear();
        }
        if self.rewards_flat.capacity() == 0 {
            self.rewards_flat = Vec::with_capacity(scalar_capacity);
        } else {
            self.rewards_flat.clear();
        }
        if self.terminated_flat.capacity() == 0 {
            self.terminated_flat = Vec::with_capacity(scalar_capacity);
        } else {
            self.terminated_flat.clear();
        }
        if self.truncated_flat.capacity() == 0 {
            self.truncated_flat = Vec::with_capacity(scalar_capacity);
        } else {
            self.truncated_flat.clear();
        }
        if self.values_flat.capacity() == 0 {
            self.values_flat = Vec::with_capacity(scalar_capacity);
        } else {
            self.values_flat.clear();
        }
        if self.log_probs_flat.capacity() == 0 {
            self.log_probs_flat = Vec::with_capacity(scalar_capacity);
        } else {
            self.log_probs_flat.clear();
        }

        // Clear stacked tensors
        self.observations = None;
        self.actions = None;
        self.rewards = None;
        self.terminated = None;
        self.truncated = None;
        self.values = None;
        self.log_probs = None;
        // Clear computed values
        self.advantages = None;
        self.returns = None;
    }

    /// Check if buffer is full and ready for training.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.full
    }

    /// Get current buffer position.
    #[inline]
    pub fn position(&self) -> usize {
        self.pos
    }

    /// Get buffer size (steps per environment).
    #[inline]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get number of environments.
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get total number of samples (buffer_size * num_envs).
    #[inline]
    pub fn total_size(&self) -> usize {
        self.buffer_size * self.num_envs
    }

    /// Get the device this buffer is allocated on.
    #[inline]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get computed advantages (if available).
    pub fn advantages(&self) -> Option<&Tensor> {
        self.advantages.as_ref()
    }

    /// Get computed returns (if available).
    pub fn returns(&self) -> Option<&Tensor> {
        self.returns.as_ref()
    }

    // =========================================================================
    // Private helper methods
    // =========================================================================

    /// Validate observation tensor shape.
    fn validate_obs_shape(&self, obs: &Tensor) -> Result<()> {
        let mut expected = vec![self.num_envs];
        expected.extend_from_slice(&self.obs_shape);

        if obs.dims() != expected.as_slice() {
            return Err(OctaneError::ShapeMismatch {
                expected,
                got: obs.dims().to_vec(),
            });
        }
        Ok(())
    }

    /// Validate action tensor shape.
    fn validate_action_shape(&self, action: &Tensor) -> Result<()> {
        let expected = vec![self.num_envs, self.action_dim];
        if action.dims() != expected.as_slice() {
            return Err(OctaneError::ShapeMismatch {
                expected,
                got: action.dims().to_vec(),
            });
        }
        Ok(())
    }

    /// Validate scalar (per-env) tensor shape.
    fn validate_scalar_shape(&self, tensor: &Tensor) -> Result<()> {
        let expected = vec![self.num_envs];
        if tensor.dims() != expected.as_slice() {
            return Err(OctaneError::ShapeMismatch {
                expected,
                got: tensor.dims().to_vec(),
            });
        }
        Ok(())
    }

    /// Flatten buffer tensor from [buffer_size, num_envs, ...] to [total, ...].
    fn flatten_buffer(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.dims();
        if dims.len() < 2 {
            return Err(OctaneError::Buffer(
                "Tensor must have at least 2 dimensions".to_string(),
            ));
        }

        let total = dims[0] * dims[1];
        let mut new_shape = vec![total];
        new_shape.extend_from_slice(&dims[2..]);

        Ok(tensor.reshape(new_shape)?)
    }

    /// Flatten scalar buffer from [buffer_size, num_envs] to [total].
    fn flatten_scalar_buffer(&self, tensor: &Tensor) -> Result<Tensor> {
        let dims = tensor.dims();
        if dims.len() != 2 {
            return Err(OctaneError::Buffer(format!(
                "Scalar buffer must have 2 dimensions, got {:?}",
                dims
            )));
        }

        let total = dims[0] * dims[1];
        Ok(tensor.reshape((total,))?)
    }

    /// Normalize advantages to zero mean and unit variance.
    fn normalize_advantages(&self, advantages: &Tensor) -> Result<Tensor> {
        let mean = advantages.mean_all()?;
        let centered = advantages.broadcast_sub(&mean)?;

        // Compute variance manually: var = mean((x - mean)^2)
        let squared = centered.sqr()?;
        let variance = squared.mean_all()?;
        let std = variance.sqrt()?;

        // Add small epsilon for numerical stability
        let eps = 1e-8;
        let std_eps = (std + eps)?;

        Ok(centered.broadcast_div(&std_eps)?)
    }
}

/// Configuration for the rollout buffer.
#[derive(Debug, Clone)]
pub struct RolloutBufferConfig {
    /// Number of steps per rollout.
    pub buffer_size: usize,
    /// Discount factor (gamma).
    pub gamma: f32,
    /// GAE lambda parameter.
    pub gae_lambda: f32,
}

impl Default for RolloutBufferConfig {
    fn default() -> Self {
        Self {
            buffer_size: 2048,
            gamma: 0.99,
            gae_lambda: 0.95,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_device() -> Device {
        Device::Cpu
    }

    #[test]
    fn test_buffer_creation() -> Result<()> {
        let device = make_device();
        let buffer = RolloutBuffer::new(128, 4, &[8], 2, &device)?;

        assert_eq!(buffer.buffer_size(), 128);
        assert_eq!(buffer.num_envs(), 4);
        assert_eq!(buffer.total_size(), 512);
        assert!(!buffer.is_full());
        assert_eq!(buffer.position(), 0);

        Ok(())
    }

    #[test]
    fn test_buffer_add() -> Result<()> {
        let device = make_device();
        let candle_device = device.to_candle()?;
        let mut buffer = RolloutBuffer::new(4, 2, &[3], 1, &device)?;

        for i in 0..4 {
            let obs = Tensor::ones(&[2, 3], DType::F32, &candle_device)?;
            let action = Tensor::zeros(&[2, 1], DType::F32, &candle_device)?;
            let reward = Tensor::from_slice(&[1.0f32, 2.0], (2,), &candle_device)?;
            let terminated = Tensor::zeros((2,), DType::F32, &candle_device)?;
            let truncated = Tensor::zeros((2,), DType::F32, &candle_device)?;
            let value = Tensor::from_slice(&[0.5f32, 0.6], (2,), &candle_device)?;
            let log_prob = Tensor::from_slice(&[-0.5f32, -0.6], (2,), &candle_device)?;

            buffer.add(
                &obs,
                &action,
                &reward,
                &terminated,
                &truncated,
                &value,
                &log_prob,
            )?;
            assert_eq!(buffer.position(), i + 1);
        }

        assert!(buffer.is_full());

        Ok(())
    }

    #[test]
    fn test_gae_computation() -> Result<()> {
        let device = make_device();
        let candle_device = device.to_candle()?;
        let mut buffer = RolloutBuffer::new(3, 2, &[4], 1, &device)?;

        // Add some transitions
        for _ in 0..3 {
            let obs = Tensor::ones(&[2, 4], DType::F32, &candle_device)?;
            let action = Tensor::zeros(&[2, 1], DType::F32, &candle_device)?;
            let reward = Tensor::from_slice(&[1.0f32, 1.0], (2,), &candle_device)?;
            let terminated = Tensor::zeros((2,), DType::F32, &candle_device)?;
            let truncated = Tensor::zeros((2,), DType::F32, &candle_device)?;
            let value = Tensor::from_slice(&[0.5f32, 0.5], (2,), &candle_device)?;
            let log_prob = Tensor::from_slice(&[-1.0f32, -1.0], (2,), &candle_device)?;

            buffer.add(
                &obs,
                &action,
                &reward,
                &terminated,
                &truncated,
                &value,
                &log_prob,
            )?;
        }

        let last_values = Tensor::from_slice(&[0.5f32, 0.5], (2,), &candle_device)?;

        buffer.compute_returns_and_advantages(&last_values, 0.99, 0.95)?;

        assert!(buffer.advantages().is_some());
        assert!(buffer.returns().is_some());

        let advantages = buffer.advantages().unwrap();
        let returns = buffer.returns().unwrap();

        assert_eq!(advantages.dims(), &[3, 2]);
        assert_eq!(returns.dims(), &[3, 2]);

        Ok(())
    }

    #[test]
    fn test_batch_sampling() -> Result<()> {
        let device = make_device();
        let candle_device = device.to_candle()?;
        let mut buffer = RolloutBuffer::new(8, 4, &[6], 2, &device)?;

        // Fill the buffer
        for _ in 0..8 {
            let obs = Tensor::ones(&[4, 6], DType::F32, &candle_device)?;
            let action = Tensor::zeros(&[4, 2], DType::F32, &candle_device)?;
            let reward = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], (4,), &candle_device)?;
            let terminated = Tensor::zeros((4,), DType::F32, &candle_device)?;
            let truncated = Tensor::zeros((4,), DType::F32, &candle_device)?;
            let value = Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], (4,), &candle_device)?;
            let log_prob = Tensor::from_slice(&[-1.0f32, -1.0, -1.0, -1.0], (4,), &candle_device)?;

            buffer.add(
                &obs,
                &action,
                &reward,
                &terminated,
                &truncated,
                &value,
                &log_prob,
            )?;
        }

        let last_values = Tensor::from_slice(&[0.5f32, 0.5, 0.5, 0.5], (4,), &candle_device)?;

        buffer.compute_returns_and_advantages(&last_values, 0.99, 0.95)?;

        // Get batches of size 8 (total = 32, so 4 batches)
        let batches = buffer.get_batches(8)?;

        assert_eq!(batches.len(), 4);
        for batch in &batches {
            assert_eq!(batch.observations.dims(), &[8, 6]);
            assert_eq!(batch.actions.dims(), &[8, 2]);
            assert_eq!(batch.old_log_probs.dims(), &[8]);
            assert_eq!(batch.advantages.dims(), &[8]);
            assert_eq!(batch.returns.dims(), &[8]);
            assert_eq!(batch.old_values.dims(), &[8]);
        }

        Ok(())
    }

    #[test]
    fn test_buffer_reset() -> Result<()> {
        let device = make_device();
        let candle_device = device.to_candle()?;
        let mut buffer = RolloutBuffer::new(2, 2, &[4], 1, &device)?;

        // Fill the buffer
        for _ in 0..2 {
            let obs = Tensor::ones(&[2, 4], DType::F32, &candle_device)?;
            let action = Tensor::zeros(&[2, 1], DType::F32, &candle_device)?;
            let reward = Tensor::ones((2,), DType::F32, &candle_device)?;
            let terminated = Tensor::zeros((2,), DType::F32, &candle_device)?;
            let truncated = Tensor::zeros((2,), DType::F32, &candle_device)?;
            let value = Tensor::ones((2,), DType::F32, &candle_device)?;
            let log_prob = Tensor::ones((2,), DType::F32, &candle_device)?;

            buffer.add(
                &obs,
                &action,
                &reward,
                &terminated,
                &truncated,
                &value,
                &log_prob,
            )?;
        }

        assert!(buffer.is_full());

        buffer.reset();

        assert!(!buffer.is_full());
        assert_eq!(buffer.position(), 0);
        assert!(buffer.advantages().is_none());
        assert!(buffer.returns().is_none());

        Ok(())
    }

    #[test]
    fn test_episode_boundaries() -> Result<()> {
        // Test that GAE properly handles episode boundaries (terminated=1)
        let device = make_device();
        let candle_device = device.to_candle()?;
        let mut buffer = RolloutBuffer::new(4, 1, &[2], 1, &device)?;

        let obs = Tensor::ones(&[1, 2], DType::F32, &candle_device)?;
        let action = Tensor::zeros(&[1, 1], DType::F32, &candle_device)?;
        let log_prob = Tensor::from_slice(&[-1.0f32], (1,), &candle_device)?;
        let no_truncation = Tensor::zeros((1,), DType::F32, &candle_device)?;

        // Step 0: reward=1, value=0.5, not terminated
        buffer.add(
            &obs,
            &action,
            &Tensor::from_slice(&[1.0f32], (1,), &candle_device)?,
            &Tensor::from_slice(&[0.0f32], (1,), &candle_device)?, // not terminated
            &no_truncation,
            &Tensor::from_slice(&[0.5f32], (1,), &candle_device)?,
            &log_prob,
        )?;

        // Step 1: reward=2, value=0.5, TERMINATED (episode ends)
        buffer.add(
            &obs,
            &action,
            &Tensor::from_slice(&[2.0f32], (1,), &candle_device)?,
            &Tensor::from_slice(&[1.0f32], (1,), &candle_device)?, // terminated!
            &no_truncation,
            &Tensor::from_slice(&[0.5f32], (1,), &candle_device)?,
            &log_prob,
        )?;

        // Step 2: reward=1, value=0.5, not terminated (new episode)
        buffer.add(
            &obs,
            &action,
            &Tensor::from_slice(&[1.0f32], (1,), &candle_device)?,
            &Tensor::from_slice(&[0.0f32], (1,), &candle_device)?,
            &no_truncation,
            &Tensor::from_slice(&[0.5f32], (1,), &candle_device)?,
            &log_prob,
        )?;

        // Step 3: reward=1, value=0.5, not terminated
        buffer.add(
            &obs,
            &action,
            &Tensor::from_slice(&[1.0f32], (1,), &candle_device)?,
            &Tensor::from_slice(&[0.0f32], (1,), &candle_device)?,
            &no_truncation,
            &Tensor::from_slice(&[0.5f32], (1,), &candle_device)?,
            &log_prob,
        )?;

        let last_values = Tensor::from_slice(&[0.5f32], (1,), &candle_device)?;

        buffer.compute_returns_and_advantages(&last_values, 0.99, 0.95)?;

        // Verify that advantages were computed
        assert!(buffer.advantages().is_some());

        Ok(())
    }

    #[test]
    fn test_gae_values_correct() -> Result<()> {
        // Test GAE computation with known values
        let device = make_device();
        let candle_device = device.to_candle()?;
        let mut buffer = RolloutBuffer::new(2, 1, &[1], 1, &device)?;

        let obs = Tensor::ones(&[1, 1], DType::F32, &candle_device)?;
        let action = Tensor::zeros(&[1, 1], DType::F32, &candle_device)?;
        let log_prob = Tensor::from_slice(&[-1.0f32], (1,), &candle_device)?;
        let no_terminated = Tensor::zeros((1,), DType::F32, &candle_device)?;
        let no_truncated = Tensor::zeros((1,), DType::F32, &candle_device)?;

        // Step 0: reward=1.0, value=0.0
        buffer.add(
            &obs,
            &action,
            &Tensor::from_slice(&[1.0f32], (1,), &candle_device)?,
            &no_terminated,
            &no_truncated,
            &Tensor::from_slice(&[0.0f32], (1,), &candle_device)?,
            &log_prob,
        )?;

        // Step 1: reward=1.0, value=0.0
        buffer.add(
            &obs,
            &action,
            &Tensor::from_slice(&[1.0f32], (1,), &candle_device)?,
            &no_terminated,
            &no_truncated,
            &Tensor::from_slice(&[0.0f32], (1,), &candle_device)?,
            &log_prob,
        )?;

        let last_values = Tensor::from_slice(&[0.0f32], (1,), &candle_device)?;

        // gamma=1.0, lambda=1.0 for simple calculation
        // With values=0 everywhere:
        // Step 1: delta = 1 + 1*0 - 0 = 1, A = 1
        // Step 0: delta = 1 + 1*0 - 0 = 1, A = 1 + 1*1*1 = 2
        buffer.compute_returns_and_advantages(&last_values, 1.0, 1.0)?;

        let advantages = buffer.advantages().unwrap();
        let adv_vec: Vec<f32> = advantages.flatten_all()?.to_vec1()?;

        // Check that step 0 has higher advantage than step 1
        assert!(
            adv_vec[0] > adv_vec[1],
            "GAE should accumulate: {} > {}",
            adv_vec[0],
            adv_vec[1]
        );

        Ok(())
    }
}
