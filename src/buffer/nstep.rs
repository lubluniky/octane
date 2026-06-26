//! N-step returns replay buffer for improved sample efficiency.
//!
//! This module provides an n-step variant of the replay buffer that computes
//! multi-step returns, which can accelerate learning and reduce variance.
//!
//! # N-step Returns
//!
//! Instead of using single-step TD targets:
//! ```text
//! Q(s, a) <- r + gamma * Q(s', a')
//! ```
//!
//! N-step returns use multi-step targets:
//! ```text
//! Q(s, a) <- r_0 + gamma*r_1 + gamma^2*r_2 + ... + gamma^n*Q(s_n, a_n)
//! ```
//!
//! This provides a balance between high-variance Monte Carlo returns and
//! high-bias single-step TD returns.

use crate::buffer::{ReplayBatch, ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, OctaneError, Result};
use candle_core::Tensor;
use std::collections::VecDeque;

// Unused import warning will be silenced when Tensor API is used

/// Configuration for the n-step replay buffer.
#[derive(Debug, Clone)]
pub struct NStepConfig {
    /// Number of steps for n-step returns (default: 3).
    pub n_step: usize,
    /// Discount factor (default: 0.99).
    pub gamma: f32,
    /// Base replay buffer configuration.
    pub buffer_config: ReplayBufferConfig,
}

impl Default for NStepConfig {
    fn default() -> Self {
        Self {
            n_step: 3,
            gamma: 0.99,
            buffer_config: ReplayBufferConfig::default(),
        }
    }
}

impl NStepConfig {
    /// Create a new n-step config with default settings.
    pub fn new(obs_dim: usize, action_dim: usize) -> Self {
        Self {
            buffer_config: ReplayBufferConfig::new(obs_dim, action_dim),
            ..Default::default()
        }
    }

    /// Set the number of steps.
    pub fn n_step(mut self, n: usize) -> Self {
        self.n_step = n;
        self
    }

    /// Set the discount factor.
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Set the buffer capacity.
    pub fn capacity(mut self, cap: usize) -> Self {
        self.buffer_config.capacity = cap;
        self
    }

    /// Enable prioritized replay.
    pub fn prioritized(mut self, enabled: bool) -> Self {
        self.buffer_config.prioritized = enabled;
        self
    }
}

/// Temporary storage for a single transition before n-step computation.
#[derive(Debug, Clone)]
struct NStepTransition {
    /// Initial observation.
    obs: Vec<f32>,
    /// Action taken.
    action: Vec<f32>,
    /// Immediate reward.
    reward: f32,
    /// Whether episode terminated at this step.
    done: bool,
}

/// N-step replay buffer that computes multi-step returns.
///
/// This buffer maintains a sliding window of recent transitions and computes
/// n-step returns when storing completed n-step sequences. It properly handles
/// episode boundaries by truncating the n-step return at terminal states.
///
/// # Example
///
/// ```ignore
/// use octane::buffer::{NStepReplayBuffer, NStepConfig};
///
/// let config = NStepConfig::new(4, 2).n_step(3).gamma(0.99);
/// let mut buffer = NStepReplayBuffer::new(config, Device::Cpu)?;
///
/// // Add transitions (automatically computes n-step returns)
/// buffer.add(&obs, &action, reward, &next_obs, done);
///
/// // Sample batches with n-step returns
/// let batch = buffer.sample(32)?;
/// ```
pub struct NStepReplayBuffer {
    /// Underlying replay buffer.
    inner: ReplayBuffer,
    /// Configuration.
    config: NStepConfig,
    /// Device for tensor operations.
    device: Device,
    /// Sliding window of recent transitions.
    n_step_buffer: VecDeque<NStepTransition>,
    /// Discount powers: [1, gamma, gamma^2, ..., gamma^(n-1)].
    gamma_powers: Vec<f32>,
    /// The most recent observation (for next_obs in n-step).
    last_obs: Option<Vec<f32>>,
    #[cfg(test)]
    /// Debug capture of stored n-step returns for deterministic testing.
    debug_returns: Vec<f32>,
    #[cfg(test)]
    /// Debug capture of stored done flags for deterministic testing.
    debug_dones: Vec<bool>,
}

impl NStepReplayBuffer {
    /// Create a new n-step replay buffer.
    ///
    /// # Arguments
    ///
    /// * `config` - N-step buffer configuration
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// A new `NStepReplayBuffer` ready for collecting transitions.
    pub fn new(config: NStepConfig, device: Device) -> Result<Self> {
        if config.n_step == 0 {
            return Err(OctaneError::InvalidConfig(
                "n_step must be at least 1".to_string(),
            ));
        }
        if config.gamma <= 0.0 || config.gamma > 1.0 {
            return Err(OctaneError::InvalidConfig(
                "gamma must be in (0, 1]".to_string(),
            ));
        }

        let inner = ReplayBuffer::new(config.buffer_config.clone(), device)?;

        // Precompute gamma powers for efficiency
        let gamma_powers: Vec<f32> = (0..config.n_step)
            .map(|i| config.gamma.powi(i as i32))
            .collect();

        let n_step_capacity = config.n_step;

        Ok(Self {
            inner,
            config,
            device,
            n_step_buffer: VecDeque::with_capacity(n_step_capacity),
            gamma_powers,
            last_obs: None,
            #[cfg(test)]
            debug_returns: Vec::new(),
            #[cfg(test)]
            debug_dones: Vec::new(),
        })
    }

    /// Add a transition to the buffer.
    ///
    /// The transition is first added to the n-step sliding window. When the
    /// window is full (or an episode ends), the n-step return is computed
    /// and the complete transition is added to the main replay buffer.
    ///
    /// # Arguments
    ///
    /// * `obs` - Current observation
    /// * `action` - Action taken
    /// * `reward` - Reward received
    /// * `next_obs` - Next observation
    /// * `done` - Whether episode terminated
    pub fn add(&mut self, obs: &[f32], action: &[f32], reward: f32, next_obs: &[f32], done: bool) {
        // Store the transition in the n-step buffer
        self.n_step_buffer.push_back(NStepTransition {
            obs: obs.to_vec(),
            action: action.to_vec(),
            reward,
            done,
        });

        // Update last observation for n-step next_obs
        self.last_obs = Some(next_obs.to_vec());

        // If n-step buffer is full, compute n-step return and store
        if self.n_step_buffer.len() == self.config.n_step {
            self.store_nstep_transition(next_obs, done);
        }

        // Handle episode boundaries: flush remaining transitions with shorter horizons
        if done {
            self.flush_episode_end(next_obs);
        }
    }

    /// Compute and store an n-step transition.
    fn store_nstep_transition(&mut self, final_next_obs: &[f32], final_done: bool) {
        let first = match self.n_step_buffer.front() {
            Some(t) => t,
            None => return,
        };

        // Compute n-step return
        let mut n_step_return = 0.0;
        let mut gamma_discount = 1.0;
        let mut encountered_done = false;

        for transition in self.n_step_buffer.iter() {
            n_step_return += gamma_discount * transition.reward;

            if transition.done {
                encountered_done = true;
                break;
            }
            gamma_discount *= self.config.gamma;
        }

        // Always bootstrap from the provided next_obs (the terminal observation
        // when the episode ended within the window), matching flush_episode_end.
        // The previous branch stored a pre-terminal *state* as next_obs; `done`
        // masks it in the bootstrap target, but the inconsistency was a latent
        // correctness trap for any consumer that reads next_obs unconditionally.
        let n_step_next_obs = final_next_obs.to_vec();
        let n_step_done = encountered_done || final_done;

        // Store in the underlying buffer
        self.inner.add(
            &first.obs,
            &first.action,
            n_step_return,
            &n_step_next_obs,
            n_step_done,
        );

        #[cfg(test)]
        {
            self.debug_returns.push(n_step_return);
            self.debug_dones.push(n_step_done);
        }

        // Remove the oldest transition
        self.n_step_buffer.pop_front();
    }

    /// Flush remaining transitions at episode end.
    fn flush_episode_end(&mut self, terminal_obs: &[f32]) {
        while !self.n_step_buffer.is_empty() {
            let first = self.n_step_buffer.front().unwrap().clone();

            // Compute return for remaining steps
            let mut n_step_return = 0.0;
            let mut gamma_discount = 1.0;

            for transition in self.n_step_buffer.iter() {
                n_step_return += gamma_discount * transition.reward;
                if transition.done {
                    break;
                }
                gamma_discount *= self.config.gamma;
            }

            // Store with terminal observation
            self.inner
                .add(&first.obs, &first.action, n_step_return, terminal_obs, true);

            #[cfg(test)]
            {
                self.debug_returns.push(n_step_return);
                self.debug_dones.push(true);
            }

            self.n_step_buffer.pop_front();
        }
    }

    /// Add a transition using Tensor inputs.
    pub fn add_tensor(
        &mut self,
        obs: &Tensor,
        action: &Tensor,
        reward: f32,
        next_obs: &Tensor,
        done: bool,
    ) -> Result<()> {
        let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;
        let action_vec: Vec<f32> = action.flatten_all()?.to_vec1()?;
        let next_obs_vec: Vec<f32> = next_obs.flatten_all()?.to_vec1()?;

        self.add(&obs_vec, &action_vec, reward, &next_obs_vec, done);
        Ok(())
    }

    /// Sample a batch of transitions.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - Number of transitions to sample
    ///
    /// # Returns
    ///
    /// A batch of n-step transitions.
    pub fn sample(&mut self, batch_size: usize) -> Result<ReplayBatch> {
        self.inner.sample(batch_size)
    }

    /// Update priorities for sampled transitions (PER).
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        self.inner.update_priorities(indices, td_errors);
    }

    /// Get number of stored transitions.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Get buffer capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    /// Check if buffer can provide a batch of given size.
    #[inline]
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.inner.can_sample(batch_size)
    }

    /// Get the n-step value.
    #[inline]
    pub fn n_step(&self) -> usize {
        self.config.n_step
    }

    /// Get the discount factor.
    #[inline]
    pub fn gamma(&self) -> f32 {
        self.config.gamma
    }

    /// Get the gamma^n discount factor for bootstrapping.
    ///
    /// This is useful for computing the target value:
    /// `target = n_step_reward + gamma^n * V(next_obs)`
    #[inline]
    pub fn gamma_n(&self) -> f32 {
        self.config.gamma.powi(self.config.n_step as i32)
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.inner.clear();
        self.n_step_buffer.clear();
        self.last_obs = None;
        #[cfg(test)]
        {
            self.debug_returns.clear();
            self.debug_dones.clear();
        }
    }

    /// Set random seed for reproducibility.
    pub fn seed(&mut self, seed: u64) {
        self.inner.seed(seed);
    }

    /// Get access to the underlying replay buffer.
    pub fn inner(&self) -> &ReplayBuffer {
        &self.inner
    }

    /// Get mutable access to the underlying replay buffer.
    pub fn inner_mut(&mut self) -> &mut ReplayBuffer {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nstep_basic() -> Result<()> {
        let config = NStepConfig::new(4, 2).n_step(3).gamma(0.99);
        let buffer = NStepReplayBuffer::new(config, Device::Cpu)?;

        assert!(buffer.is_empty());
        assert_eq!(buffer.n_step(), 3);
        assert!((buffer.gamma() - 0.99).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_nstep_return_computation() -> Result<()> {
        let config = NStepConfig::new(2, 1)
            .n_step(3)
            .gamma(1.0) // gamma=1 for easy verification
            .capacity(100);
        let mut buffer = NStepReplayBuffer::new(config, Device::Cpu)?;

        // Add 5 transitions with unique rewards so the sliding window is
        // observable without relying on random sampling.
        for i in 0..5 {
            let obs = vec![i as f32, 0.0];
            let action = vec![0.0];
            let next_obs = vec![(i + 1) as f32, 0.0];
            buffer.add(&obs, &action, (i + 1) as f32, &next_obs, false);
        }

        assert_eq!(buffer.len(), 3);

        #[cfg(test)]
        assert_eq!(buffer.debug_returns, vec![6.0, 9.0, 12.0]);

        Ok(())
    }

    #[test]
    fn test_nstep_episode_boundary() -> Result<()> {
        let config = NStepConfig::new(2, 1).n_step(3).gamma(1.0).capacity(100);
        let mut buffer = NStepReplayBuffer::new(config, Device::Cpu)?;

        // Add transitions: 2 normal, then terminal
        buffer.add(&[0.0, 0.0], &[0.0], 1.0, &[1.0, 0.0], false);
        buffer.add(&[1.0, 0.0], &[0.0], 1.0, &[2.0, 0.0], false);
        buffer.add(&[2.0, 0.0], &[0.0], 1.0, &[3.0, 0.0], true); // Terminal

        // All transitions should be flushed
        // First: 1 + 1 + 1 = 3
        // Second: 1 + 1 = 2
        // Third: 1
        assert_eq!(buffer.len(), 3);

        #[cfg(test)]
        assert_eq!(buffer.debug_returns, vec![3.0, 2.0, 1.0]);

        Ok(())
    }

    #[test]
    fn test_nstep_gamma_discount() -> Result<()> {
        let config = NStepConfig::new(2, 1)
            .n_step(3)
            .gamma(0.9) // 10% discount
            .capacity(100);
        let mut buffer = NStepReplayBuffer::new(config, Device::Cpu)?;

        // Add 3 transitions with reward=1.0 each
        for i in 0..3 {
            let obs = vec![i as f32, 0.0];
            let next_obs = vec![(i + 1) as f32, 0.0];
            buffer.add(&obs, &[0.0], 1.0, &next_obs, i == 2);
        }

        #[cfg(test)]
        assert_eq!(buffer.debug_returns.len(), 3);
        #[cfg(test)]
        assert!((buffer.debug_returns[0] - 2.71).abs() < 0.01);
        #[cfg(test)]
        assert!((buffer.debug_returns[1] - 1.9).abs() < 0.01);
        #[cfg(test)]
        assert!((buffer.debug_returns[2] - 1.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_nstep_gamma_n() -> Result<()> {
        let config = NStepConfig::new(2, 1).n_step(3).gamma(0.99);
        let buffer = NStepReplayBuffer::new(config, Device::Cpu)?;

        let expected = 0.99_f32.powi(3);
        assert!((buffer.gamma_n() - expected).abs() < 1e-6);

        Ok(())
    }

    #[test]
    fn test_nstep_with_per() -> Result<()> {
        let config = NStepConfig::new(4, 2)
            .n_step(3)
            .gamma(0.99)
            .capacity(100)
            .prioritized(true);
        let mut buffer = NStepReplayBuffer::new(config, Device::Cpu)?;

        // Add transitions
        for i in 0..20 {
            let obs = vec![i as f32; 4];
            let action = vec![0.0, 1.0];
            let next_obs = vec![(i + 1) as f32; 4];
            buffer.add(&obs, &action, 1.0, &next_obs, i == 19);
        }

        assert!(!buffer.is_empty());

        // Sample with priorities
        let batch = buffer.sample(8)?;
        assert!(batch.weights.is_some());

        // Update priorities
        let td_errors: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect();
        buffer.update_priorities(&batch.indices, &td_errors);

        Ok(())
    }

    #[test]
    fn test_nstep_clear() -> Result<()> {
        let config = NStepConfig::new(2, 1).n_step(3).capacity(100);
        let mut buffer = NStepReplayBuffer::new(config, Device::Cpu)?;

        for i in 0..10 {
            buffer.add(&[i as f32, 0.0], &[0.0], 1.0, &[0.0, 0.0], false);
        }

        assert!(!buffer.is_empty());

        buffer.clear();
        assert!(buffer.is_empty());

        Ok(())
    }

    #[test]
    fn test_invalid_config() {
        let config = NStepConfig::new(2, 1).n_step(0);
        assert!(NStepReplayBuffer::new(config, Device::Cpu).is_err());

        let config = NStepConfig::new(2, 1).gamma(0.0);
        assert!(NStepReplayBuffer::new(config, Device::Cpu).is_err());

        let config = NStepConfig::new(2, 1).gamma(1.5);
        assert!(NStepReplayBuffer::new(config, Device::Cpu).is_err());
    }
}
