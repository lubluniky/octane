//! Rollout buffer for storing experience data during training.
//!
//! The rollout buffer stores transitions collected from the environment
//! and computes returns and advantages for policy gradient updates.

use crate::core::{Device, OctaneError, Result};
use candle_core::Tensor;

/// Sample from the rollout buffer for training.
#[derive(Debug)]
pub struct RolloutSample {
    /// Observations [batch_size, obs_dim]
    pub observations: Tensor,
    /// Actions [batch_size, act_dim] or [batch_size, 1] for discrete
    pub actions: Tensor,
    /// Old log probabilities [batch_size]
    pub log_probs: Tensor,
    /// Advantages [batch_size]
    pub advantages: Tensor,
    /// Returns (discounted cumulative rewards) [batch_size]
    pub returns: Tensor,
    /// Values [batch_size]
    pub values: Tensor,
}

/// Rollout buffer for on-policy algorithms (PPO, A2C).
///
/// Stores experience data and computes GAE advantages.
pub struct RolloutBuffer {
    /// Number of steps per rollout.
    n_steps: usize,
    /// Number of parallel environments.
    num_envs: usize,
    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    act_dim: usize,
    /// Device for tensor operations.
    device: Device,

    /// Stored observations [n_steps, num_envs, obs_dim]
    observations: Vec<Tensor>,
    /// Stored actions [n_steps, num_envs, act_dim]
    actions: Vec<Tensor>,
    /// Stored rewards [n_steps, num_envs]
    rewards: Vec<Tensor>,
    /// Stored done flags [n_steps, num_envs]
    dones: Vec<Tensor>,
    /// Stored values [n_steps, num_envs]
    values: Vec<Tensor>,
    /// Stored log probabilities [n_steps, num_envs]
    log_probs: Vec<Tensor>,

    /// Computed advantages [n_steps, num_envs]
    advantages: Option<Tensor>,
    /// Computed returns [n_steps, num_envs]
    returns: Option<Tensor>,

    /// Current position in buffer.
    pos: usize,
    /// Whether the buffer is full.
    full: bool,
}

impl RolloutBuffer {
    /// Create a new rollout buffer.
    pub fn new(
        n_steps: usize,
        num_envs: usize,
        obs_dim: usize,
        act_dim: usize,
        device: Device,
    ) -> Result<Self> {
        Ok(Self {
            n_steps,
            num_envs,
            obs_dim,
            act_dim,
            device,
            observations: Vec::with_capacity(n_steps),
            actions: Vec::with_capacity(n_steps),
            rewards: Vec::with_capacity(n_steps),
            dones: Vec::with_capacity(n_steps),
            values: Vec::with_capacity(n_steps),
            log_probs: Vec::with_capacity(n_steps),
            advantages: None,
            returns: None,
            pos: 0,
            full: false,
        })
    }

    /// Reset the buffer for a new rollout.
    pub fn reset(&mut self) {
        self.observations.clear();
        self.actions.clear();
        self.rewards.clear();
        self.dones.clear();
        self.values.clear();
        self.log_probs.clear();
        self.advantages = None;
        self.returns = None;
        self.pos = 0;
        self.full = false;
    }

    /// Add a transition to the buffer.
    pub fn add(
        &mut self,
        obs: &Tensor,
        action: &Tensor,
        reward: &Tensor,
        done: &Tensor,
        value: &Tensor,
        log_prob: &Tensor,
    ) -> Result<()> {
        if self.pos >= self.n_steps {
            return Err(OctaneError::Buffer(
                "Buffer overflow: too many transitions added".to_string(),
            ));
        }

        self.observations.push(obs.clone());
        self.actions.push(action.clone());
        self.rewards.push(reward.clone());
        self.dones.push(done.clone());
        self.values.push(value.clone());
        self.log_probs.push(log_prob.clone());

        self.pos += 1;
        if self.pos >= self.n_steps {
            self.full = true;
        }

        Ok(())
    }

    /// Compute returns and advantages using Generalized Advantage Estimation (GAE).
    ///
    /// GAE computes advantages as:
    /// A_t = delta_t + (gamma * lambda) * delta_{t+1} + ... + (gamma * lambda)^{T-t+1} * delta_{T-1}
    ///
    /// where delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
    pub fn compute_returns_and_advantages(
        &mut self,
        last_values: &Tensor,
        gamma: f32,
        gae_lambda: f32,
    ) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // Convert stored tensors to vectors for processing
        let mut rewards_vec: Vec<Vec<f32>> = Vec::with_capacity(self.n_steps);
        let mut values_vec: Vec<Vec<f32>> = Vec::with_capacity(self.n_steps);
        let mut dones_vec: Vec<Vec<f32>> = Vec::with_capacity(self.n_steps);

        for step in 0..self.pos {
            rewards_vec.push(self.rewards[step].to_vec1()?);
            values_vec.push(self.values[step].to_vec1()?);
            dones_vec.push(self.dones[step].to_vec1()?);
        }

        let last_vals: Vec<f32> = last_values.to_vec1()?;

        // Compute advantages using GAE
        let mut advantages_data = vec![vec![0.0f32; self.num_envs]; self.pos];
        let mut returns_data = vec![vec![0.0f32; self.num_envs]; self.pos];

        for env_idx in 0..self.num_envs {
            let mut last_gae = 0.0f32;
            let mut next_value = last_vals[env_idx];
            let mut next_non_terminal = 1.0f32;

            // Work backwards through time
            for t in (0..self.pos).rev() {
                let current_value = values_vec[t][env_idx];
                let reward = rewards_vec[t][env_idx];
                let done = dones_vec[t][env_idx];

                // TD error: delta = r + gamma * V(s') * (1 - done) - V(s)
                let delta = reward + gamma * next_value * next_non_terminal - current_value;

                // GAE: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
                last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae;

                advantages_data[t][env_idx] = last_gae;
                returns_data[t][env_idx] = last_gae + current_value;

                next_value = current_value;
                next_non_terminal = 1.0 - done;
            }
        }

        // Convert to tensors [n_steps * num_envs]
        let advantages_flat: Vec<f32> = advantages_data.into_iter().flatten().collect();
        let returns_flat: Vec<f32> = returns_data.into_iter().flatten().collect();

        self.advantages = Some(Tensor::from_slice(
            &advantages_flat,
            &[self.pos, self.num_envs],
            &candle_device,
        )?);

        self.returns = Some(Tensor::from_slice(
            &returns_flat,
            &[self.pos, self.num_envs],
            &candle_device,
        )?);

        Ok(())
    }

    /// Get all data from the buffer as a single sample.
    pub fn get_all(&self) -> Result<RolloutSample> {
        if !self.full && self.pos == 0 {
            return Err(OctaneError::Buffer("Buffer is empty".to_string()));
        }

        let advantages = self.advantages.as_ref().ok_or_else(|| {
            OctaneError::Buffer(
                "Advantages not computed. Call compute_returns_and_advantages first.".to_string(),
            )
        })?;

        let returns = self.returns.as_ref().ok_or_else(|| {
            OctaneError::Buffer(
                "Returns not computed. Call compute_returns_and_advantages first.".to_string(),
            )
        })?;

        // Stack all tensors along time dimension
        let observations = Tensor::stack(&self.observations[..self.pos], 0)?;
        let actions = Tensor::stack(&self.actions[..self.pos], 0)?;
        let values = Tensor::stack(&self.values[..self.pos], 0)?;
        let log_probs = Tensor::stack(&self.log_probs[..self.pos], 0)?;

        // Reshape to [n_steps * num_envs, ...]
        let total_samples = self.pos * self.num_envs;

        let observations = observations.reshape(&[total_samples, self.obs_dim])?;
        let actions = if self.obs_dim == self.act_dim {
            actions.reshape(&[total_samples, self.act_dim])?
        } else {
            // Discrete actions: [n_steps, num_envs, 1] -> [total_samples, 1]
            actions.reshape(&[total_samples, actions.dim(2)?])?
        };
        let values = values.flatten_all()?;
        let log_probs = log_probs.flatten_all()?;
        let advantages = advantages.flatten_all()?;
        let returns = returns.flatten_all()?;

        Ok(RolloutSample {
            observations,
            actions,
            log_probs,
            advantages,
            returns,
            values,
        })
    }

    /// Get data in batches for minibatch training.
    pub fn get_batches(&self, batch_size: usize) -> Result<Vec<RolloutSample>> {
        let all_data = self.get_all()?;
        let n_samples = all_data.observations.dim(0)?;
        let n_batches = n_samples.div_ceil(batch_size);

        let mut batches = Vec::with_capacity(n_batches);

        for i in 0..n_batches {
            let start = i * batch_size;
            let end = (start + batch_size).min(n_samples);

            let obs_batch = all_data.observations.narrow(0, start, end - start)?;
            let act_batch = all_data.actions.narrow(0, start, end - start)?;
            let lp_batch = all_data.log_probs.narrow(0, start, end - start)?;
            let adv_batch = all_data.advantages.narrow(0, start, end - start)?;
            let ret_batch = all_data.returns.narrow(0, start, end - start)?;
            let val_batch = all_data.values.narrow(0, start, end - start)?;

            batches.push(RolloutSample {
                observations: obs_batch,
                actions: act_batch,
                log_probs: lp_batch,
                advantages: adv_batch,
                returns: ret_batch,
                values: val_batch,
            });
        }

        Ok(batches)
    }

    /// Get the number of stored transitions.
    pub fn size(&self) -> usize {
        self.pos * self.num_envs
    }

    /// Check if the buffer is full.
    pub fn is_full(&self) -> bool {
        self.full
    }

    /// Get buffer capacity in transitions.
    pub fn capacity(&self) -> usize {
        self.n_steps * self.num_envs
    }
}

/// Generator for shuffled minibatch indices.
pub struct BatchSampler {
    /// All indices.
    indices: Vec<usize>,
    /// Batch size.
    batch_size: usize,
    /// Current position.
    pos: usize,
}

impl BatchSampler {
    /// Create a new batch sampler.
    pub fn new(n_samples: usize, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        Self {
            indices,
            batch_size,
            pos: 0,
        }
    }

    /// Get the next batch of indices.
    pub fn next_batch(&mut self) -> Option<&[usize]> {
        if self.pos >= self.indices.len() {
            return None;
        }

        let start = self.pos;
        let end = (start + self.batch_size).min(self.indices.len());
        self.pos = end;

        Some(&self.indices[start..end])
    }

    /// Reset the sampler for a new epoch.
    pub fn reset(&mut self, shuffle: bool) {
        self.pos = 0;
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get number of batches.
    pub fn n_batches(&self) -> usize {
        self.indices.len().div_ceil(self.batch_size)
    }
}

impl Iterator for BatchSampler {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        self.next_batch().map(|s| s.to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_sampler() {
        let mut sampler = BatchSampler::new(10, 3, false);
        assert_eq!(sampler.n_batches(), 4);

        let batch1 = sampler.next_batch().unwrap();
        assert_eq!(batch1, &[0, 1, 2]);

        let batch2 = sampler.next_batch().unwrap();
        assert_eq!(batch2, &[3, 4, 5]);

        let batch3 = sampler.next_batch().unwrap();
        assert_eq!(batch3, &[6, 7, 8]);

        let batch4 = sampler.next_batch().unwrap();
        assert_eq!(batch4, &[9]);

        assert!(sampler.next_batch().is_none());
    }

    #[test]
    fn test_rollout_buffer_creation() {
        let buffer = RolloutBuffer::new(10, 4, 8, 2, Device::Cpu).unwrap();
        assert_eq!(buffer.capacity(), 40);
        assert_eq!(buffer.size(), 0);
        assert!(!buffer.is_full());
    }
}
