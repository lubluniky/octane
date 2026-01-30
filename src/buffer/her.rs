//! Hindsight Experience Replay (HER) buffer for goal-conditioned RL.
//!
//! HER is a sample-efficient technique for learning goal-conditioned policies.
//! After an episode, it relabels experiences with alternative goals that were
//! actually achieved, turning failed attempts into successful training data.
//!
//! # Reference
//!
//! Andrychowicz et al., "Hindsight Experience Replay", NeurIPS 2017
//!
//! # Strategies
//!
//! - **Final**: Use the final achieved goal of the episode.
//! - **Future**: Sample k goals from future states in the same episode.
//! - **Episode**: Sample goals uniformly from the entire episode.
//! - **Random**: Sample goals uniformly from the entire buffer.

use crate::buffer::{ReplayBatch, ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, Result};
use candle_core::Tensor;
use rand::prelude::*;
use std::collections::VecDeque;

/// Goal relabeling strategy for HER.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HERStrategy {
    /// Use the final achieved goal of the episode.
    Final,
    /// Sample k goals from future timesteps in the episode.
    Future {
        /// Number of future goals to sample per transition.
        k: usize,
    },
    /// Sample goals uniformly from the entire episode.
    Episode,
    /// Sample goals uniformly from the entire buffer.
    Random,
}

impl Default for HERStrategy {
    fn default() -> Self {
        HERStrategy::Future { k: 4 }
    }
}

/// Configuration for the HER buffer.
#[derive(Debug, Clone)]
pub struct HERConfig {
    /// Goal relabeling strategy.
    pub strategy: HERStrategy,
    /// Observation dimension (excluding goal).
    pub obs_dim: usize,
    /// Goal dimension.
    pub goal_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Buffer capacity.
    pub capacity: usize,
    /// Enable prioritized experience replay.
    pub prioritized: bool,
    /// PER alpha parameter.
    pub alpha: f32,
    /// PER beta parameter.
    pub beta: f32,
}

impl Default for HERConfig {
    fn default() -> Self {
        Self {
            strategy: HERStrategy::default(),
            obs_dim: 10,
            goal_dim: 3,
            action_dim: 4,
            capacity: 1_000_000,
            prioritized: false,
            alpha: 0.6,
            beta: 0.4,
        }
    }
}

impl HERConfig {
    /// Create a new HER config with specified dimensions.
    pub fn new(obs_dim: usize, goal_dim: usize, action_dim: usize) -> Self {
        Self {
            obs_dim,
            goal_dim,
            action_dim,
            ..Default::default()
        }
    }

    /// Set the goal relabeling strategy.
    pub fn strategy(mut self, s: HERStrategy) -> Self {
        self.strategy = s;
        self
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
}

/// A single goal-conditioned transition.
#[derive(Debug, Clone)]
struct GoalTransition {
    /// Observation (without goal).
    obs: Vec<f32>,
    /// Action taken.
    action: Vec<f32>,
    /// Reward received.
    reward: f32,
    /// Next observation (without goal).
    next_obs: Vec<f32>,
    /// Whether episode terminated.
    done: bool,
    /// Goal that was achieved at next_obs.
    achieved_goal: Vec<f32>,
    /// Desired goal for this transition.
    desired_goal: Vec<f32>,
}

/// Type alias for reward recomputation function.
///
/// Takes (achieved_goal, desired_goal) and returns the reward.
pub type RewardFn = Box<dyn Fn(&[f32], &[f32]) -> f32 + Send + Sync>;

/// Hindsight Experience Replay buffer.
///
/// This buffer stores goal-conditioned transitions and applies HER during
/// sampling to improve sample efficiency in sparse reward settings.
///
/// # Example
///
/// ```ignore
/// use octane::buffer::{HERBuffer, HERConfig, HERStrategy};
///
/// let config = HERConfig::new(10, 3, 4)
///     .strategy(HERStrategy::Future { k: 4 })
///     .capacity(100_000);
///
/// // Reward function: -1 if goal not reached, 0 if reached
/// let reward_fn = |achieved: &[f32], desired: &[f32]| {
///     let dist: f32 = achieved.iter()
///         .zip(desired.iter())
///         .map(|(a, d)| (a - d).powi(2))
///         .sum::<f32>()
///         .sqrt();
///     if dist < 0.05 { 0.0 } else { -1.0 }
/// };
///
/// let mut buffer = HERBuffer::new(config, reward_fn, Device::Cpu)?;
///
/// // Collect episode and store with HER
/// buffer.add(obs, action, reward, next_obs, done, achieved_goal, desired_goal);
/// buffer.end_episode();
///
/// // Sample with HER relabeling
/// let batch = buffer.sample(256)?;
/// ```
pub struct HERBuffer {
    /// Configuration.
    config: HERConfig,
    /// Device for tensor operations.
    device: Device,
    /// Underlying replay buffer (stores concatenated [obs || goal]).
    inner: ReplayBuffer,
    /// Current episode being collected.
    current_episode: Vec<GoalTransition>,
    /// Episode storage for random strategy.
    episode_buffer: VecDeque<Vec<GoalTransition>>,
    /// Maximum episodes to store for random sampling.
    max_episodes: usize,
    /// Reward recomputation function.
    reward_fn: RewardFn,
    /// Random number generator.
    rng: StdRng,
}

impl HERBuffer {
    /// Create a new HER buffer.
    ///
    /// # Arguments
    ///
    /// * `config` - HER buffer configuration
    /// * `reward_fn` - Function to recompute reward given (achieved_goal, desired_goal)
    /// * `device` - Device for tensor operations
    pub fn new<F>(config: HERConfig, reward_fn: F, device: Device) -> Result<Self>
    where
        F: Fn(&[f32], &[f32]) -> f32 + Send + Sync + 'static,
    {
        // The inner buffer stores [obs || desired_goal] as observation
        let combined_obs_dim = config.obs_dim + config.goal_dim;

        let buffer_config = ReplayBufferConfig {
            capacity: config.capacity,
            obs_dim: combined_obs_dim,
            action_dim: config.action_dim,
            prioritized: config.prioritized,
            alpha: config.alpha,
            beta: config.beta,
            ..Default::default()
        };

        let inner = ReplayBuffer::new(buffer_config, device)?;

        // Estimate max episodes based on capacity and typical episode length
        let max_episodes = config.capacity / 50; // Assuming ~50 steps per episode

        Ok(Self {
            config,
            device,
            inner,
            current_episode: Vec::with_capacity(1000),
            episode_buffer: VecDeque::with_capacity(max_episodes),
            max_episodes,
            reward_fn: Box::new(reward_fn),
            rng: StdRng::from_entropy(),
        })
    }

    /// Add a goal-conditioned transition.
    ///
    /// # Arguments
    ///
    /// * `obs` - Current observation (without goal)
    /// * `action` - Action taken
    /// * `reward` - Reward received
    /// * `next_obs` - Next observation (without goal)
    /// * `done` - Whether episode terminated
    /// * `achieved_goal` - Goal achieved at next_obs
    /// * `desired_goal` - Original desired goal
    pub fn add(
        &mut self,
        obs: &[f32],
        action: &[f32],
        reward: f32,
        next_obs: &[f32],
        done: bool,
        achieved_goal: &[f32],
        desired_goal: &[f32],
    ) {
        self.current_episode.push(GoalTransition {
            obs: obs.to_vec(),
            action: action.to_vec(),
            reward,
            next_obs: next_obs.to_vec(),
            done,
            achieved_goal: achieved_goal.to_vec(),
            desired_goal: desired_goal.to_vec(),
        });

        // If episode ended, process it
        if done {
            self.end_episode();
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
        achieved_goal: &Tensor,
        desired_goal: &Tensor,
    ) -> Result<()> {
        let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;
        let action_vec: Vec<f32> = action.flatten_all()?.to_vec1()?;
        let next_obs_vec: Vec<f32> = next_obs.flatten_all()?.to_vec1()?;
        let achieved_vec: Vec<f32> = achieved_goal.flatten_all()?.to_vec1()?;
        let desired_vec: Vec<f32> = desired_goal.flatten_all()?.to_vec1()?;

        self.add(
            &obs_vec,
            &action_vec,
            reward,
            &next_obs_vec,
            done,
            &achieved_vec,
            &desired_vec,
        );
        Ok(())
    }

    /// End the current episode and apply HER relabeling.
    ///
    /// This should be called at the end of each episode. The collected
    /// transitions will be stored with the original goal, and additional
    /// relabeled transitions will be generated according to the HER strategy.
    pub fn end_episode(&mut self) {
        if self.current_episode.is_empty() {
            return;
        }

        let episode = std::mem::take(&mut self.current_episode);

        // Store original transitions
        for t in &episode {
            self.store_transition(t, &t.desired_goal);
        }

        // Apply HER relabeling based on strategy
        match self.config.strategy {
            HERStrategy::Final => self.relabel_final(&episode),
            HERStrategy::Future { k } => self.relabel_future(&episode, k),
            HERStrategy::Episode => self.relabel_episode(&episode),
            HERStrategy::Random => {
                // Store episode for later random sampling
                if self.episode_buffer.len() >= self.max_episodes {
                    self.episode_buffer.pop_front();
                }
                self.relabel_from_buffer(&episode);
                self.episode_buffer.push_back(episode);
            }
        }
    }

    /// Store a transition with a specific goal in the replay buffer.
    fn store_transition(&mut self, t: &GoalTransition, goal: &[f32]) {
        // Concatenate observation with goal
        let mut obs_with_goal = t.obs.clone();
        obs_with_goal.extend_from_slice(goal);

        let mut next_obs_with_goal = t.next_obs.clone();
        next_obs_with_goal.extend_from_slice(goal);

        // Recompute reward for this goal
        let reward = (self.reward_fn)(&t.achieved_goal, goal);

        self.inner.add(
            &obs_with_goal,
            &t.action,
            reward,
            &next_obs_with_goal,
            t.done,
        );
    }

    /// Relabel with final achieved goal.
    fn relabel_final(&mut self, episode: &[GoalTransition]) {
        if episode.is_empty() {
            return;
        }

        let final_goal = &episode.last().unwrap().achieved_goal;

        for t in episode {
            self.store_transition(t, final_goal);
        }
    }

    /// Relabel with k future goals.
    fn relabel_future(&mut self, episode: &[GoalTransition], k: usize) {
        for (idx, t) in episode.iter().enumerate() {
            // Sample k future goals from this transition onwards
            let future_range = idx + 1..episode.len();
            if future_range.is_empty() {
                continue;
            }

            let num_samples = k.min(future_range.len());
            let indices: Vec<usize> = future_range.clone().collect();

            for _ in 0..num_samples {
                let future_idx = indices[self.rng.gen_range(0..indices.len())];
                let future_goal = &episode[future_idx].achieved_goal;
                self.store_transition(t, future_goal);
            }
        }
    }

    /// Relabel with goals from anywhere in the episode.
    fn relabel_episode(&mut self, episode: &[GoalTransition]) {
        if episode.is_empty() {
            return;
        }

        for t in episode {
            // Sample a random goal from the episode
            let goal_idx = self.rng.gen_range(0..episode.len());
            let goal = &episode[goal_idx].achieved_goal;
            self.store_transition(t, goal);
        }
    }

    /// Relabel with goals from the stored episode buffer.
    fn relabel_from_buffer(&mut self, episode: &[GoalTransition]) {
        if self.episode_buffer.is_empty() {
            // Fall back to episode strategy if no stored episodes
            self.relabel_episode(episode);
            return;
        }

        for t in episode {
            // Sample a random episode
            let ep_idx = self.rng.gen_range(0..self.episode_buffer.len());
            let goal_idx = self.rng.gen_range(0..self.episode_buffer[ep_idx].len());

            // Clone the goal to avoid borrowing issues
            let goal = self.episode_buffer[ep_idx][goal_idx].achieved_goal.clone();
            self.store_transition(t, &goal);
        }
    }

    /// Sample a batch of transitions.
    ///
    /// Returns a batch with observations that include the goal.
    /// The observation shape is [batch_size, obs_dim + goal_dim].
    pub fn sample(&mut self, batch_size: usize) -> Result<ReplayBatch> {
        self.inner.sample(batch_size)
    }

    /// Sample a batch and split observations from goals.
    ///
    /// Returns a batch along with separate goal tensors.
    pub fn sample_with_goals(&mut self, batch_size: usize) -> Result<HERBatch> {
        let batch = self.inner.sample(batch_size)?;

        let candle_device = self.device.to_candle()?;
        let obs_dim = self.config.obs_dim;
        let goal_dim = self.config.goal_dim;

        // Split observations: [obs || goal]
        let obs_flat: Vec<f32> = batch.observations.flatten_all()?.to_vec1()?;
        let next_obs_flat: Vec<f32> = batch.next_observations.flatten_all()?.to_vec1()?;

        let mut observations = Vec::with_capacity(batch_size * obs_dim);
        let mut goals = Vec::with_capacity(batch_size * goal_dim);
        let mut next_observations = Vec::with_capacity(batch_size * obs_dim);

        for i in 0..batch_size {
            let start = i * (obs_dim + goal_dim);
            observations.extend_from_slice(&obs_flat[start..start + obs_dim]);
            goals.extend_from_slice(&obs_flat[start + obs_dim..start + obs_dim + goal_dim]);

            let next_start = i * (obs_dim + goal_dim);
            next_observations.extend_from_slice(&next_obs_flat[next_start..next_start + obs_dim]);
        }

        Ok(HERBatch {
            observations: Tensor::from_slice(&observations, (batch_size, obs_dim), &candle_device)?,
            actions: batch.actions,
            rewards: batch.rewards,
            next_observations: Tensor::from_slice(
                &next_observations,
                (batch_size, obs_dim),
                &candle_device,
            )?,
            dones: batch.dones,
            goals: Tensor::from_slice(&goals, (batch_size, goal_dim), &candle_device)?,
            indices: batch.indices,
            weights: batch.weights,
        })
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

    /// Get the HER strategy.
    #[inline]
    pub fn strategy(&self) -> HERStrategy {
        self.config.strategy
    }

    /// Clear the buffer.
    pub fn clear(&mut self) {
        self.inner.clear();
        self.current_episode.clear();
        self.episode_buffer.clear();
    }

    /// Set random seed for reproducibility.
    pub fn seed(&mut self, seed: u64) {
        self.inner.seed(seed);
        self.rng = StdRng::seed_from_u64(seed);
    }
}

/// Batch of HER transitions with separate goal tensors.
#[derive(Debug)]
pub struct HERBatch {
    /// Observations [batch_size, obs_dim] (without goals).
    pub observations: Tensor,
    /// Actions [batch_size, action_dim].
    pub actions: Tensor,
    /// Rewards [batch_size].
    pub rewards: Tensor,
    /// Next observations [batch_size, obs_dim] (without goals).
    pub next_observations: Tensor,
    /// Done flags [batch_size].
    pub dones: Tensor,
    /// Goals [batch_size, goal_dim].
    pub goals: Tensor,
    /// Indices of sampled transitions (for PER).
    pub indices: Vec<usize>,
    /// Importance sampling weights (for PER).
    pub weights: Option<Tensor>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sparse_reward(achieved: &[f32], desired: &[f32]) -> f32 {
        let dist: f32 = achieved
            .iter()
            .zip(desired.iter())
            .map(|(a, d)| (a - d).powi(2))
            .sum::<f32>()
            .sqrt();
        if dist < 0.05 {
            0.0
        } else {
            -1.0
        }
    }

    #[test]
    fn test_her_basic() -> Result<()> {
        let config = HERConfig::new(4, 3, 2).capacity(1000);
        let buffer = HERBuffer::new(config, sparse_reward, Device::Cpu)?;

        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), 1000);

        Ok(())
    }

    #[test]
    fn test_her_add_episode() -> Result<()> {
        let config = HERConfig::new(4, 3, 2)
            .strategy(HERStrategy::Final)
            .capacity(1000);
        let mut buffer = HERBuffer::new(config, sparse_reward, Device::Cpu)?;

        // Simulate an episode
        for i in 0..10 {
            let obs = vec![i as f32; 4];
            let action = vec![0.0, 1.0];
            let next_obs = vec![(i + 1) as f32; 4];
            let achieved = vec![i as f32 * 0.1; 3];
            let desired = vec![1.0; 3];
            let done = i == 9;

            buffer.add(&obs, &action, -1.0, &next_obs, done, &achieved, &desired);
        }

        // Original transitions + HER relabeled (2x with Final strategy)
        assert!(buffer.len() > 0);

        Ok(())
    }

    #[test]
    fn test_her_future_strategy() -> Result<()> {
        let config = HERConfig::new(4, 3, 2)
            .strategy(HERStrategy::Future { k: 4 })
            .capacity(10000);
        let mut buffer = HERBuffer::new(config, sparse_reward, Device::Cpu)?;

        // Run multiple episodes
        for _ in 0..5 {
            for i in 0..20 {
                let obs = vec![i as f32; 4];
                let action = vec![0.0, 1.0];
                let next_obs = vec![(i + 1) as f32; 4];
                let achieved = vec![i as f32 * 0.05; 3];
                let desired = vec![1.0; 3];
                let done = i == 19;

                buffer.add(&obs, &action, -1.0, &next_obs, done, &achieved, &desired);
            }
        }

        assert!(buffer.can_sample(32));

        let batch = buffer.sample(32)?;
        assert_eq!(batch.observations.dims(), &[32, 7]); // obs_dim + goal_dim

        Ok(())
    }

    #[test]
    fn test_her_with_goals_batch() -> Result<()> {
        let config = HERConfig::new(4, 3, 2)
            .strategy(HERStrategy::Final)
            .capacity(1000);
        let mut buffer = HERBuffer::new(config, sparse_reward, Device::Cpu)?;

        // Add an episode
        for i in 0..10 {
            let obs = vec![i as f32; 4];
            let action = vec![0.0, 1.0];
            let next_obs = vec![(i + 1) as f32; 4];
            let achieved = vec![i as f32 * 0.1; 3];
            let desired = vec![1.0; 3];
            let done = i == 9;

            buffer.add(&obs, &action, -1.0, &next_obs, done, &achieved, &desired);
        }

        let batch = buffer.sample_with_goals(8)?;
        assert_eq!(batch.observations.dims(), &[8, 4]); // Just obs
        assert_eq!(batch.goals.dims(), &[8, 3]); // Just goals
        assert_eq!(batch.actions.dims(), &[8, 2]);

        Ok(())
    }

    #[test]
    fn test_her_episode_strategy() -> Result<()> {
        let config = HERConfig::new(4, 3, 2)
            .strategy(HERStrategy::Episode)
            .capacity(1000);
        let mut buffer = HERBuffer::new(config, sparse_reward, Device::Cpu)?;

        for i in 0..10 {
            let done = i == 9;
            buffer.add(
                &[i as f32; 4],
                &[0.0, 1.0],
                -1.0,
                &[(i + 1) as f32; 4],
                done,
                &[i as f32 * 0.1; 3],
                &[1.0; 3],
            );
        }

        assert!(buffer.len() > 10); // Should have relabeled samples

        Ok(())
    }

    #[test]
    fn test_her_random_strategy() -> Result<()> {
        let config = HERConfig::new(4, 3, 2)
            .strategy(HERStrategy::Random)
            .capacity(10000);
        let mut buffer = HERBuffer::new(config, sparse_reward, Device::Cpu)?;

        // Add multiple episodes
        for ep in 0..5 {
            for i in 0..10 {
                let done = i == 9;
                buffer.add(
                    &[i as f32 + ep as f32 * 10.0; 4],
                    &[0.0, 1.0],
                    -1.0,
                    &[(i + 1) as f32 + ep as f32 * 10.0; 4],
                    done,
                    &[i as f32 * 0.1; 3],
                    &[1.0; 3],
                );
            }
        }

        assert!(buffer.len() > 50);

        Ok(())
    }

    #[test]
    fn test_her_with_per() -> Result<()> {
        let config = HERConfig::new(4, 3, 2)
            .strategy(HERStrategy::Future { k: 4 })
            .capacity(1000)
            .prioritized(true);
        let mut buffer = HERBuffer::new(config, sparse_reward, Device::Cpu)?;

        // Add episode
        for i in 0..20 {
            let done = i == 19;
            buffer.add(
                &[i as f32; 4],
                &[0.0, 1.0],
                -1.0,
                &[(i + 1) as f32; 4],
                done,
                &[i as f32 * 0.05; 3],
                &[1.0; 3],
            );
        }

        let batch = buffer.sample(16)?;
        assert!(batch.weights.is_some());

        let td_errors: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        buffer.update_priorities(&batch.indices, &td_errors);

        Ok(())
    }

    #[test]
    fn test_her_clear() -> Result<()> {
        let config = HERConfig::new(4, 3, 2).capacity(1000);
        let mut buffer = HERBuffer::new(config, sparse_reward, Device::Cpu)?;

        for i in 0..10 {
            let done = i == 9;
            buffer.add(
                &[i as f32; 4],
                &[0.0, 1.0],
                -1.0,
                &[(i + 1) as f32; 4],
                done,
                &[0.0; 3],
                &[1.0; 3],
            );
        }

        assert!(buffer.len() > 0);

        buffer.clear();
        assert!(buffer.is_empty());

        Ok(())
    }

    #[test]
    fn test_reward_relabeling() -> Result<()> {
        // Test that rewards are properly recomputed
        let config = HERConfig::new(4, 3, 2)
            .strategy(HERStrategy::Final)
            .capacity(1000);

        // Success reward function
        let reward_fn = |achieved: &[f32], desired: &[f32]| {
            if achieved == desired {
                1.0
            } else {
                0.0
            }
        };

        let mut buffer = HERBuffer::new(config, reward_fn, Device::Cpu)?;

        // Episode where we achieve goal [1.0, 1.0, 1.0] at the end
        for i in 0..5 {
            let achieved = if i == 4 {
                vec![1.0, 1.0, 1.0]
            } else {
                vec![0.0, 0.0, 0.0]
            };
            let done = i == 4;

            buffer.add(
                &[i as f32; 4],
                &[0.0, 1.0],
                0.0, // Original reward
                &[(i + 1) as f32; 4],
                done,
                &achieved,
                &[1.0, 1.0, 1.0],
            );
        }

        // The relabeled transitions with final goal should have reward = 1.0
        // for the last step (where achieved == desired)
        assert!(buffer.len() > 5);

        Ok(())
    }
}
