//! Replay buffer for off-policy algorithms (DQN, DDPG, TD3, SAC).
//!
//! This module provides efficient, ring-buffer based storage for experience replay.
//! Supports uniform random sampling and optional prioritized experience replay (PER).

use crate::core::{Device, Result, OctaneError};
use candle_core::Tensor;
use rand::prelude::*;

/// A single transition stored in the replay buffer.
#[derive(Debug, Clone)]
pub struct Transition {
    /// Current observation.
    pub obs: Vec<f32>,
    /// Action taken.
    pub action: Vec<f32>,
    /// Reward received.
    pub reward: f32,
    /// Next observation.
    pub next_obs: Vec<f32>,
    /// Whether episode terminated.
    pub done: bool,
}

/// Batch of transitions for training.
#[derive(Debug)]
pub struct ReplayBatch {
    /// Observations [batch_size, obs_dim].
    pub observations: Tensor,
    /// Actions [batch_size, action_dim].
    pub actions: Tensor,
    /// Rewards [batch_size].
    pub rewards: Tensor,
    /// Next observations [batch_size, obs_dim].
    pub next_observations: Tensor,
    /// Done flags [batch_size].
    pub dones: Tensor,
    /// Indices of sampled transitions (for PER updates).
    pub indices: Vec<usize>,
    /// Importance sampling weights (for PER).
    pub weights: Option<Tensor>,
}

/// Configuration for replay buffer.
#[derive(Debug, Clone)]
pub struct ReplayBufferConfig {
    /// Maximum buffer capacity.
    pub capacity: usize,
    /// Observation dimension.
    pub obs_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Whether to use prioritized experience replay.
    pub prioritized: bool,
    /// PER alpha parameter (prioritization exponent).
    pub alpha: f32,
    /// PER beta parameter (importance sampling).
    pub beta: f32,
    /// PER beta annealing rate.
    pub beta_annealing: f32,
    /// Minimum priority for PER.
    pub min_priority: f32,
}

impl Default for ReplayBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 1_000_000,
            obs_dim: 4,
            action_dim: 1,
            prioritized: false,
            alpha: 0.6,
            beta: 0.4,
            beta_annealing: 1e-6,
            min_priority: 1e-6,
        }
    }
}

impl ReplayBufferConfig {
    /// Create a new config with given dimensions.
    pub fn new(obs_dim: usize, action_dim: usize) -> Self {
        Self {
            obs_dim,
            action_dim,
            ..Default::default()
        }
    }

    /// Set buffer capacity.
    pub fn capacity(mut self, cap: usize) -> Self {
        self.capacity = cap;
        self
    }

    /// Enable prioritized experience replay.
    pub fn prioritized(mut self, enabled: bool) -> Self {
        self.prioritized = enabled;
        self
    }

    /// Set PER alpha (prioritization exponent).
    pub fn alpha(mut self, a: f32) -> Self {
        self.alpha = a;
        self
    }

    /// Set PER beta (importance sampling).
    pub fn beta(mut self, b: f32) -> Self {
        self.beta = b;
        self
    }
}

/// Efficient ring-buffer based replay buffer for off-policy RL.
///
/// Stores transitions in contiguous arrays for cache efficiency.
/// Supports both uniform random sampling and prioritized experience replay.
pub struct ReplayBuffer {
    /// Configuration.
    config: ReplayBufferConfig,
    /// Device for tensor creation.
    device: Device,

    // Storage arrays (SoA layout for efficiency)
    /// Observations [capacity, obs_dim].
    observations: Vec<f32>,
    /// Actions [capacity, action_dim].
    actions: Vec<f32>,
    /// Rewards [capacity].
    rewards: Vec<f32>,
    /// Next observations [capacity, obs_dim].
    next_observations: Vec<f32>,
    /// Done flags [capacity].
    dones: Vec<f32>,

    /// Current write position (ring buffer index).
    position: usize,
    /// Number of valid transitions stored.
    size: usize,

    // Prioritized replay data
    /// Priorities for each transition.
    priorities: Option<Vec<f32>>,
    /// Sum tree for efficient priority sampling.
    sum_tree: Option<SumTree>,
    /// Current beta for importance sampling.
    current_beta: f32,

    /// Random number generator.
    rng: StdRng,
}

impl ReplayBuffer {
    /// Create a new replay buffer.
    pub fn new(config: ReplayBufferConfig, device: Device) -> Result<Self> {
        let capacity = config.capacity;
        let obs_dim = config.obs_dim;
        let action_dim = config.action_dim;

        if capacity == 0 {
            return Err(OctaneError::InvalidConfig("Capacity must be positive".into()));
        }

        let (priorities, sum_tree) = if config.prioritized {
            (
                Some(vec![1.0; capacity]),
                Some(SumTree::new(capacity)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            observations: vec![0.0; capacity * obs_dim],
            actions: vec![0.0; capacity * action_dim],
            rewards: vec![0.0; capacity],
            next_observations: vec![0.0; capacity * obs_dim],
            dones: vec![0.0; capacity],
            position: 0,
            size: 0,
            priorities,
            sum_tree,
            current_beta: config.beta,
            rng: StdRng::from_entropy(),
            config,
            device,
        })
    }

    /// Add a transition to the buffer.
    #[inline]
    pub fn add(
        &mut self,
        obs: &[f32],
        action: &[f32],
        reward: f32,
        next_obs: &[f32],
        done: bool,
    ) {
        let idx = self.position;
        let obs_dim = self.config.obs_dim;
        let action_dim = self.config.action_dim;

        // Copy observation
        let obs_start = idx * obs_dim;
        self.observations[obs_start..obs_start + obs_dim].copy_from_slice(obs);

        // Copy action
        let action_start = idx * action_dim;
        self.actions[action_start..action_start + action_dim].copy_from_slice(action);

        // Copy scalar values
        self.rewards[idx] = reward;
        self.dones[idx] = if done { 1.0 } else { 0.0 };

        // Copy next observation
        self.next_observations[obs_start..obs_start + obs_dim].copy_from_slice(next_obs);

        // Set max priority for new transitions (PER)
        if let Some(ref mut priorities) = self.priorities {
            let max_priority = priorities[..self.size.max(1)]
                .iter()
                .cloned()
                .fold(1.0f32, f32::max);
            priorities[idx] = max_priority;

            if let Some(ref mut tree) = self.sum_tree {
                tree.update(idx, max_priority.powf(self.config.alpha));
            }
        }

        // Update position and size
        self.position = (self.position + 1) % self.config.capacity;
        self.size = (self.size + 1).min(self.config.capacity);
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

    /// Sample a batch of transitions uniformly at random.
    pub fn sample(&mut self, batch_size: usize) -> Result<ReplayBatch> {
        if self.size < batch_size {
            return Err(OctaneError::Buffer(format!(
                "Not enough samples: {} < {}",
                self.size, batch_size
            )));
        }

        if self.config.prioritized {
            self.sample_prioritized(batch_size)
        } else {
            self.sample_uniform(batch_size)
        }
    }

    /// Sample uniformly at random.
    fn sample_uniform(&mut self, batch_size: usize) -> Result<ReplayBatch> {
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.size))
            .collect();

        self.get_batch(&indices, None)
    }

    /// Sample according to priorities (PER).
    fn sample_prioritized(&mut self, batch_size: usize) -> Result<ReplayBatch> {
        let tree = self.sum_tree.as_ref().unwrap();
        let total = tree.total();
        let segment = total / batch_size as f32;

        let mut indices = Vec::with_capacity(batch_size);
        let mut priorities = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let lower = segment * i as f32;
            let upper = segment * (i + 1) as f32;
            let value = self.rng.gen_range(lower..upper);
            let (idx, priority) = tree.get(value);
            indices.push(idx);
            priorities.push(priority);
        }

        // Compute importance sampling weights
        let max_weight = (self.size as f32 * tree.min_priority() / total)
            .powf(-self.current_beta);

        let weights: Vec<f32> = priorities
            .iter()
            .map(|&p| {
                let prob = p / total;
                let weight = (self.size as f32 * prob).powf(-self.current_beta);
                weight / max_weight
            })
            .collect();

        // Anneal beta
        self.current_beta = (self.current_beta + self.config.beta_annealing).min(1.0);

        let candle_device = self.device.to_candle()?;
        let weights_tensor = Tensor::from_slice(&weights, (batch_size,), &candle_device)?;

        self.get_batch(&indices, Some(weights_tensor))
    }

    /// Get batch from specific indices.
    fn get_batch(&self, indices: &[usize], weights: Option<Tensor>) -> Result<ReplayBatch> {
        let batch_size = indices.len();
        let obs_dim = self.config.obs_dim;
        let action_dim = self.config.action_dim;

        let mut obs_batch = Vec::with_capacity(batch_size * obs_dim);
        let mut action_batch = Vec::with_capacity(batch_size * action_dim);
        let mut reward_batch = Vec::with_capacity(batch_size);
        let mut next_obs_batch = Vec::with_capacity(batch_size * obs_dim);
        let mut done_batch = Vec::with_capacity(batch_size);

        for &idx in indices {
            let obs_start = idx * obs_dim;
            obs_batch.extend_from_slice(&self.observations[obs_start..obs_start + obs_dim]);

            let action_start = idx * action_dim;
            action_batch.extend_from_slice(&self.actions[action_start..action_start + action_dim]);

            reward_batch.push(self.rewards[idx]);

            next_obs_batch.extend_from_slice(&self.next_observations[obs_start..obs_start + obs_dim]);

            done_batch.push(self.dones[idx]);
        }

        let candle_device = self.device.to_candle()?;

        Ok(ReplayBatch {
            observations: Tensor::from_slice(&obs_batch, (batch_size, obs_dim), &candle_device)?,
            actions: Tensor::from_slice(&action_batch, (batch_size, action_dim), &candle_device)?,
            rewards: Tensor::from_slice(&reward_batch, (batch_size,), &candle_device)?,
            next_observations: Tensor::from_slice(&next_obs_batch, (batch_size, obs_dim), &candle_device)?,
            dones: Tensor::from_slice(&done_batch, (batch_size,), &candle_device)?,
            indices: indices.to_vec(),
            weights,
        })
    }

    /// Update priorities for sampled transitions (PER).
    pub fn update_priorities(&mut self, indices: &[usize], td_errors: &[f32]) {
        if let (Some(ref mut priorities), Some(ref mut tree)) =
            (&mut self.priorities, &mut self.sum_tree)
        {
            for (&idx, &td_error) in indices.iter().zip(td_errors.iter()) {
                let priority = (td_error.abs() + self.config.min_priority)
                    .powf(self.config.alpha);
                priorities[idx] = td_error.abs() + self.config.min_priority;
                tree.update(idx, priority);
            }
        }
    }

    /// Get number of stored transitions.
    #[inline]
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Get buffer capacity.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.config.capacity
    }

    /// Check if buffer can provide a batch of given size.
    #[inline]
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.size >= batch_size
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.position = 0;
        self.size = 0;

        if let Some(ref mut tree) = self.sum_tree {
            *tree = SumTree::new(self.config.capacity);
        }
        if let Some(ref mut priorities) = self.priorities {
            priorities.fill(1.0);
        }
    }

    /// Set random seed for reproducibility.
    pub fn seed(&mut self, seed: u64) {
        self.rng = StdRng::seed_from_u64(seed);
    }
}

/// Sum tree for efficient prioritized sampling.
///
/// Provides O(log n) sampling and O(log n) priority updates.
struct SumTree {
    /// Tree capacity (number of leaf nodes).
    capacity: usize,
    /// Tree array (size = 2 * capacity - 1).
    tree: Vec<f32>,
}

impl SumTree {
    /// Create a new sum tree.
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            tree: vec![0.0; 2 * capacity - 1],
        }
    }

    /// Update priority at leaf index.
    fn update(&mut self, idx: usize, priority: f32) {
        let tree_idx = idx + self.capacity - 1;
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;

        // Propagate change up the tree
        let mut parent = tree_idx;
        while parent > 0 {
            parent = (parent - 1) / 2;
            self.tree[parent] += change;
        }
    }

    /// Get leaf index and priority for a given value.
    fn get(&self, value: f32) -> (usize, f32) {
        let mut idx = 0;
        let mut value = value;

        while idx < self.capacity - 1 {
            let left = 2 * idx + 1;
            let right = left + 1;

            if value <= self.tree[left] {
                idx = left;
            } else {
                value -= self.tree[left];
                idx = right;
            }
        }

        let leaf_idx = idx - (self.capacity - 1);
        (leaf_idx, self.tree[idx])
    }

    /// Get total sum of priorities.
    #[inline]
    fn total(&self) -> f32 {
        self.tree[0]
    }

    /// Get minimum priority in the tree.
    fn min_priority(&self) -> f32 {
        self.tree[self.capacity - 1..]
            .iter()
            .cloned()
            .filter(|&p| p > 0.0)
            .fold(f32::MAX, f32::min)
            .max(1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer_basic() -> Result<()> {
        let config = ReplayBufferConfig::new(4, 2).capacity(100);
        let mut buffer = ReplayBuffer::new(config, Device::Cpu)?;

        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 100);

        // Add some transitions
        for i in 0..50 {
            let obs = vec![i as f32; 4];
            let action = vec![0.0, 1.0];
            let reward = i as f32 * 0.1;
            let next_obs = vec![(i + 1) as f32; 4];
            let done = i % 10 == 9;

            buffer.add(&obs, &action, reward, &next_obs, done);
        }

        assert_eq!(buffer.len(), 50);
        assert!(buffer.can_sample(32));

        // Sample a batch
        let batch = buffer.sample(32)?;
        assert_eq!(batch.observations.dims(), &[32, 4]);
        assert_eq!(batch.actions.dims(), &[32, 2]);
        assert_eq!(batch.rewards.dims(), &[32]);
        assert_eq!(batch.dones.dims(), &[32]);

        Ok(())
    }

    #[test]
    fn test_replay_buffer_overflow() -> Result<()> {
        let config = ReplayBufferConfig::new(2, 1).capacity(10);
        let mut buffer = ReplayBuffer::new(config, Device::Cpu)?;

        // Add more than capacity
        for i in 0..25 {
            buffer.add(&[i as f32, i as f32], &[0.0], 1.0, &[0.0, 0.0], false);
        }

        assert_eq!(buffer.len(), 10); // Should cap at capacity

        Ok(())
    }

    #[test]
    fn test_prioritized_replay() -> Result<()> {
        let config = ReplayBufferConfig::new(4, 2)
            .capacity(100)
            .prioritized(true);
        let mut buffer = ReplayBuffer::new(config, Device::Cpu)?;

        // Add transitions
        for i in 0..50 {
            let obs = vec![i as f32; 4];
            let action = vec![0.0, 1.0];
            buffer.add(&obs, &action, 1.0, &obs, false);
        }

        // Sample with priorities
        let batch = buffer.sample(16)?;
        assert!(batch.weights.is_some());

        // Update priorities
        let td_errors: Vec<f32> = (0..16).map(|i| (i as f32 + 1.0) * 0.1).collect();
        buffer.update_priorities(&batch.indices, &td_errors);

        Ok(())
    }

    #[test]
    fn test_sum_tree() {
        let mut tree = SumTree::new(4);

        tree.update(0, 1.0);
        tree.update(1, 2.0);
        tree.update(2, 3.0);
        tree.update(3, 4.0);

        assert!((tree.total() - 10.0).abs() < 1e-6);

        // Test sampling
        let (idx, _) = tree.get(0.5);
        assert_eq!(idx, 0);

        let (idx, _) = tree.get(2.5);
        assert_eq!(idx, 1);
    }
}
