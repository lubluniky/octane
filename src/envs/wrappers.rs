//! Standard environment wrappers for preprocessing and monitoring.
//!
//! This module provides composable wrappers that modify environment
//! behavior, following the decorator pattern common in Gym/Gymnasium.
//!
//! # Example
//! ```ignore
//! use octane::envs::{TradingEnv, TimeLimit, NormalizeObservation};
//!
//! let env = TradingEnv::new(data)?;
//! let env = TimeLimit::new(env, 1000);
//! let env = NormalizeObservation::new(env);
//! ```

use crate::core::{Device, OctaneError, Result};
use crate::envs::{BoxSpace, Environment, ObsType, Space, StepInfo, StepResult};
use candle_core::Tensor;
use std::collections::VecDeque;

/// Wrapper that stacks the last N observations.
///
/// Useful for environments where temporal information is important,
/// such as Atari games or time-series prediction.
#[derive(Clone)]
pub struct FrameStack<E: Environment> {
    /// Wrapped environment.
    env: E,
    /// Number of frames to stack.
    n_stack: usize,
    /// Frame buffer (most recent frame at the end).
    frames: VecDeque<Tensor>,
    /// Stacked observation space.
    obs_space: BoxSpace,
}

impl<E: Environment<ObsSpace = BoxSpace>> FrameStack<E> {
    /// Create a new frame stacking wrapper.
    ///
    /// # Arguments
    /// * `env` - The environment to wrap.
    /// * `n_stack` - Number of frames to stack.
    pub fn new(env: E, n_stack: usize) -> Self {
        let base_shape = env.observation_space().shape();

        // Stacked shape: [n_stack * base_flat_dim] (flattened)
        // Or for image-like: [n_stack, ...base_shape]
        let stacked_shape = if base_shape.len() == 1 {
            vec![n_stack * base_shape[0]]
        } else {
            let mut shape = vec![n_stack];
            shape.extend_from_slice(base_shape);
            shape
        };

        let _flat_dim: usize = stacked_shape.iter().product();
        let obs_space = BoxSpace::unbounded(stacked_shape);

        // Initialize bounds based on original space
        let base_space = env.observation_space();
        let base_low = &base_space.low;
        let base_high = &base_space.high;

        let stacked_low: Vec<f32> = (0..n_stack)
            .flat_map(|_| base_low.iter().copied())
            .collect();
        let stacked_high: Vec<f32> = (0..n_stack)
            .flat_map(|_| base_high.iter().copied())
            .collect();

        let obs_space = BoxSpace {
            low: stacked_low,
            high: stacked_high,
            shape: obs_space.shape,
        };

        Self {
            env,
            n_stack,
            frames: VecDeque::with_capacity(n_stack),
            obs_space,
        }
    }

    /// Stack the current frames into a single observation.
    fn stack_frames(&self, _device: &Device) -> Result<Tensor> {
        if self.frames.is_empty() {
            return Err(OctaneError::Environment("No frames to stack".to_string()));
        }

        // Collect all frames
        let frames: Vec<&Tensor> = self.frames.iter().collect();

        // Concatenate along first dimension
        Tensor::cat(&frames, 0).map_err(Into::into)
    }
}

impl<E: Environment<ObsSpace = BoxSpace> + Clone> Environment for FrameStack<E> {
    type ObsSpace = BoxSpace;
    type ActSpace = E::ActSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActSpace {
        self.env.action_space()
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        let obs = self.env.reset(device)?;

        // Fill buffer with copies of the initial observation
        self.frames.clear();
        for _ in 0..self.n_stack {
            self.frames.push_back(obs.clone());
        }

        self.stack_frames(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let result = self.env.step(action, device)?;

        // Update frame buffer
        if self.frames.len() >= self.n_stack {
            self.frames.pop_front();
        }
        self.frames.push_back(result.observation);

        let stacked_obs = self.stack_frames(device)?;

        Ok(StepResult {
            observation: stacked_obs,
            reward: result.reward,
            terminated: result.terminated,
            truncated: result.truncated,
            info: result.info,
        })
    }

    fn render(&self) -> Result<()> {
        self.env.render()
    }

    fn close(&mut self) -> Result<()> {
        self.env.close()
    }

    fn name(&self) -> &str {
        self.env.name()
    }
}

/// Wrapper that truncates episodes after a maximum number of steps.
#[derive(Clone)]
pub struct TimeLimit<E: Environment> {
    /// Wrapped environment.
    env: E,
    /// Maximum steps per episode.
    max_steps: usize,
    /// Current step count.
    current_step: usize,
}

impl<E: Environment> TimeLimit<E> {
    /// Create a new time limit wrapper.
    ///
    /// # Arguments
    /// * `env` - The environment to wrap.
    /// * `max_steps` - Maximum steps before truncation.
    pub fn new(env: E, max_steps: usize) -> Self {
        Self {
            env,
            max_steps,
            current_step: 0,
        }
    }

    /// Get remaining steps in the current episode.
    pub fn remaining_steps(&self) -> usize {
        self.max_steps.saturating_sub(self.current_step)
    }
}

impl<E: Environment + Clone> Environment for TimeLimit<E> {
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> &Self::ActSpace {
        self.env.action_space()
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        self.current_step = 0;
        self.env.reset(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        self.current_step += 1;
        let mut result = self.env.step(action, device)?;

        // Check time limit
        if self.current_step >= self.max_steps && !result.terminated {
            result.truncated = true;

            // Add step info if not present
            if result.info.is_none() {
                result.info = Some(StepInfo::default());
            }
            if let Some(ref mut info) = result.info {
                info.extra.insert("TimeLimit.truncated".to_string(), 1.0);
            }
        }

        Ok(result)
    }

    fn render(&self) -> Result<()> {
        self.env.render()
    }

    fn close(&mut self) -> Result<()> {
        self.env.close()
    }

    fn name(&self) -> &str {
        self.env.name()
    }
}

/// Running statistics for online mean/variance estimation.
#[derive(Clone, Debug)]
pub struct RunningMeanStd {
    /// Running mean.
    mean: Vec<f32>,
    /// Running variance.
    var: Vec<f32>,
    /// Sample count.
    count: f64,
    /// Dimension.
    dim: usize,
    /// Epsilon for numerical stability.
    epsilon: f32,
}

impl RunningMeanStd {
    /// Create new running statistics tracker.
    pub fn new(dim: usize) -> Self {
        Self {
            mean: vec![0.0; dim],
            var: vec![1.0; dim],
            count: 1e-4, // Small initial count for stability
            dim,
            epsilon: 1e-8,
        }
    }

    /// Update statistics with a new batch of data.
    pub fn update(&mut self, data: &[f32]) {
        let n = data.len() / self.dim;
        if n == 0 {
            return;
        }

        // Compute batch statistics
        let mut batch_mean = vec![0.0f32; self.dim];
        let mut batch_var = vec![0.0f32; self.dim];

        for i in 0..n {
            for j in 0..self.dim {
                batch_mean[j] += data[i * self.dim + j];
            }
        }
        for j in 0..self.dim {
            batch_mean[j] /= n as f32;
        }

        for i in 0..n {
            for j in 0..self.dim {
                let diff = data[i * self.dim + j] - batch_mean[j];
                batch_var[j] += diff * diff;
            }
        }
        for j in 0..self.dim {
            batch_var[j] /= n as f32;
        }

        // Welford's online algorithm for combining statistics
        let batch_count = n as f64;
        let total_count = self.count + batch_count;

        let delta: Vec<f32> = batch_mean
            .iter()
            .zip(&self.mean)
            .map(|(b, m)| b - m)
            .collect();

        let m_a = self
            .var
            .iter()
            .map(|v| v * self.count as f32)
            .collect::<Vec<_>>();
        let m_b = batch_var
            .iter()
            .map(|v| v * batch_count as f32)
            .collect::<Vec<_>>();

        for j in 0..self.dim {
            let m2 = m_a[j]
                + m_b[j]
                + delta[j] * delta[j] * (self.count * batch_count / total_count) as f32;
            self.var[j] = m2 / total_count as f32;
            self.mean[j] += delta[j] * (batch_count / total_count) as f32;
        }

        self.count = total_count;
    }

    /// Normalize data using current statistics.
    pub fn normalize(&self, data: &[f32]) -> Vec<f32> {
        data.iter()
            .enumerate()
            .map(|(i, &x)| {
                let j = i % self.dim;
                (x - self.mean[j]) / (self.var[j].sqrt() + self.epsilon)
            })
            .collect()
    }

    /// Get current mean.
    pub fn mean(&self) -> &[f32] {
        &self.mean
    }

    /// Get current standard deviation.
    pub fn std(&self) -> Vec<f32> {
        self.var.iter().map(|v| v.sqrt()).collect()
    }
}

/// Wrapper that normalizes observations using running mean/std.
#[derive(Clone)]
pub struct NormalizeObservation<E: Environment> {
    /// Wrapped environment.
    env: E,
    /// Running statistics.
    obs_rms: RunningMeanStd,
    /// Clipping range for normalized observations.
    clip_obs: f32,
    /// Whether to update statistics (disable during evaluation).
    training: bool,
}

impl<E: Environment<ObsSpace = BoxSpace>> NormalizeObservation<E> {
    /// Create a new observation normalization wrapper.
    pub fn new(env: E) -> Self {
        let dim = env.observation_space().flat_dim();
        Self {
            env,
            obs_rms: RunningMeanStd::new(dim),
            clip_obs: 10.0,
            training: true,
        }
    }

    /// Set the observation clipping range.
    pub fn clip_obs(mut self, clip: f32) -> Self {
        self.clip_obs = clip;
        self
    }

    /// Enable or disable training mode (statistics updates).
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Get the running statistics.
    pub fn obs_rms(&self) -> &RunningMeanStd {
        &self.obs_rms
    }

    /// Normalize an observation tensor.
    fn normalize_obs(&mut self, obs: &Tensor, device: &Device) -> Result<Tensor> {
        let data: Vec<f32> = obs.flatten_all()?.to_vec1()?;

        if self.training {
            self.obs_rms.update(&data);
        }

        let normalized = self.obs_rms.normalize(&data);
        let clipped: Vec<f32> = normalized
            .into_iter()
            .map(|x| x.clamp(-self.clip_obs, self.clip_obs))
            .collect();

        let shape = self.env.observation_space().shape();
        let candle_device = device.to_candle()?;
        Tensor::from_slice(&clipped, shape, &candle_device).map_err(Into::into)
    }
}

impl<E: Environment<ObsSpace = BoxSpace> + Clone> Environment for NormalizeObservation<E> {
    type ObsSpace = BoxSpace;
    type ActSpace = E::ActSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> &Self::ActSpace {
        self.env.action_space()
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        let obs = self.env.reset(device)?;
        self.normalize_obs(&obs, device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let mut result = self.env.step(action, device)?;
        result.observation = self.normalize_obs(&result.observation, device)?;
        Ok(result)
    }

    fn render(&self) -> Result<()> {
        self.env.render()
    }

    fn close(&mut self) -> Result<()> {
        self.env.close()
    }

    fn name(&self) -> &str {
        self.env.name()
    }
}

/// Wrapper that normalizes rewards using running mean/std.
#[derive(Clone)]
pub struct NormalizeReward<E: Environment> {
    /// Wrapped environment.
    env: E,
    /// Running statistics for returns (discounted sum of rewards).
    return_rms: RunningMeanStd,
    /// Discount factor for return calculation.
    gamma: f32,
    /// Current discounted return estimate.
    returns: f32,
    /// Clipping range for normalized rewards.
    clip_reward: f32,
    /// Whether to update statistics.
    training: bool,
    /// Epsilon for numerical stability.
    epsilon: f32,
}

impl<E: Environment> NormalizeReward<E> {
    /// Create a new reward normalization wrapper.
    pub fn new(env: E) -> Self {
        Self {
            env,
            return_rms: RunningMeanStd::new(1),
            gamma: 0.99,
            returns: 0.0,
            clip_reward: 10.0,
            training: true,
            epsilon: 1e-8,
        }
    }

    /// Set the discount factor.
    pub fn gamma(mut self, gamma: f32) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the reward clipping range.
    pub fn clip_reward(mut self, clip: f32) -> Self {
        self.clip_reward = clip;
        self
    }

    /// Enable or disable training mode.
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    /// Normalize a reward value.
    fn normalize_reward(&mut self, reward: f32, done: bool) -> f32 {
        // Update return estimate
        self.returns = self.returns * self.gamma + reward;

        if self.training {
            self.return_rms.update(&[self.returns]);
        }

        // Normalize by return std
        let std = self.return_rms.std()[0];
        let normalized = reward / (std + self.epsilon);

        // Reset return on done
        if done {
            self.returns = 0.0;
        }

        normalized.clamp(-self.clip_reward, self.clip_reward)
    }
}

impl<E: Environment + Clone> Environment for NormalizeReward<E> {
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> &Self::ActSpace {
        self.env.action_space()
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        self.returns = 0.0;
        self.env.reset(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let mut result = self.env.step(action, device)?;
        result.reward = self.normalize_reward(result.reward, result.done());
        Ok(result)
    }

    fn render(&self) -> Result<()> {
        self.env.render()
    }

    fn close(&mut self) -> Result<()> {
        self.env.close()
    }

    fn name(&self) -> &str {
        self.env.name()
    }
}

/// Wrapper that clips continuous actions to space bounds.
#[derive(Clone)]
pub struct ClipAction<E: Environment> {
    /// Wrapped environment.
    env: E,
}

impl<E: Environment<ActSpace = BoxSpace>> ClipAction<E> {
    /// Create a new action clipping wrapper.
    pub fn new(env: E) -> Self {
        Self { env }
    }

    /// Clip action tensor to space bounds.
    fn clip_action(&self, action: &Tensor, device: &Device) -> Result<Tensor> {
        let action_space = self.env.action_space();
        let data: Vec<f32> = action.flatten_all()?.to_vec1()?;

        let clipped: Vec<f32> = data
            .iter()
            .enumerate()
            .map(|(i, &x)| x.clamp(action_space.low[i], action_space.high[i]))
            .collect();

        let shape = action_space.shape();
        let candle_device = device.to_candle()?;
        Tensor::from_slice(&clipped, shape, &candle_device).map_err(Into::into)
    }
}

impl<E: Environment<ActSpace = BoxSpace> + Clone> Environment for ClipAction<E> {
    type ObsSpace = E::ObsSpace;
    type ActSpace = BoxSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> &Self::ActSpace {
        self.env.action_space()
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        self.env.reset(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let clipped = self.clip_action(action, device)?;
        self.env.step(&clipped, device)
    }

    fn render(&self) -> Result<()> {
        self.env.render()
    }

    fn close(&mut self) -> Result<()> {
        self.env.close()
    }

    fn name(&self) -> &str {
        self.env.name()
    }
}

/// Episode statistics tracked by RecordEpisodeStatistics.
#[derive(Debug, Clone, Default)]
pub struct EpisodeStats {
    /// Total return for the episode.
    pub episode_return: f32,
    /// Episode length in steps.
    pub episode_length: usize,
    /// Time taken (if available).
    pub time_elapsed: Option<f32>,
}

/// Wrapper that records and tracks episode statistics.
#[derive(Clone)]
pub struct RecordEpisodeStatistics<E: Environment> {
    /// Wrapped environment.
    env: E,
    /// Current episode return.
    episode_return: f32,
    /// Current episode length.
    episode_length: usize,
    /// History of completed episodes.
    episode_history: VecDeque<EpisodeStats>,
    /// Maximum history size.
    history_size: usize,
    /// Start time of current episode.
    start_time: std::time::Instant,
}

impl<E: Environment> RecordEpisodeStatistics<E> {
    /// Create a new episode statistics wrapper.
    pub fn new(env: E) -> Self {
        Self::with_history(env, 100)
    }

    /// Create with custom history size.
    pub fn with_history(env: E, history_size: usize) -> Self {
        Self {
            env,
            episode_return: 0.0,
            episode_length: 0,
            episode_history: VecDeque::with_capacity(history_size),
            history_size,
            start_time: std::time::Instant::now(),
        }
    }

    /// Get the most recent episode statistics.
    pub fn last_episode(&self) -> Option<&EpisodeStats> {
        self.episode_history.back()
    }

    /// Get all recorded episode statistics.
    pub fn episode_history(&self) -> &VecDeque<EpisodeStats> {
        &self.episode_history
    }

    /// Get the mean return over recorded episodes.
    pub fn mean_return(&self) -> Option<f32> {
        if self.episode_history.is_empty() {
            return None;
        }
        let sum: f32 = self.episode_history.iter().map(|e| e.episode_return).sum();
        Some(sum / self.episode_history.len() as f32)
    }

    /// Get the mean episode length over recorded episodes.
    pub fn mean_length(&self) -> Option<f32> {
        if self.episode_history.is_empty() {
            return None;
        }
        let sum: usize = self.episode_history.iter().map(|e| e.episode_length).sum();
        Some(sum as f32 / self.episode_history.len() as f32)
    }

    /// Get number of completed episodes.
    pub fn episode_count(&self) -> usize {
        self.episode_history.len()
    }

    /// Clear episode history.
    pub fn clear_history(&mut self) {
        self.episode_history.clear();
    }
}

impl<E: Environment + Clone> Environment for RecordEpisodeStatistics<E> {
    type ObsSpace = E::ObsSpace;
    type ActSpace = E::ActSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        self.env.observation_space()
    }

    fn action_space(&self) -> &Self::ActSpace {
        self.env.action_space()
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        self.episode_return = 0.0;
        self.episode_length = 0;
        self.start_time = std::time::Instant::now();
        self.env.reset(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let mut result = self.env.step(action, device)?;

        self.episode_return += result.reward;
        self.episode_length += 1;

        // Record episode stats if done
        if result.done() {
            let elapsed = self.start_time.elapsed().as_secs_f32();

            let stats = EpisodeStats {
                episode_return: self.episode_return,
                episode_length: self.episode_length,
                time_elapsed: Some(elapsed),
            };

            // Add to history, removing oldest if at capacity
            if self.episode_history.len() >= self.history_size {
                self.episode_history.pop_front();
            }
            self.episode_history.push_back(stats);

            // Update step info with episode stats
            if result.info.is_none() {
                result.info = Some(StepInfo::default());
            }
            if let Some(ref mut info) = result.info {
                info.episode_return = Some(self.episode_return);
                info.episode_length = Some(self.episode_length);
                info.extra.insert("time_elapsed".to_string(), elapsed);
            }
        }

        Ok(result)
    }

    fn render(&self) -> Result<()> {
        self.env.render()
    }

    fn close(&mut self) -> Result<()> {
        self.env.close()
    }

    fn name(&self) -> &str {
        self.env.name()
    }
}

/// Convenience type for chaining multiple wrappers.
///
/// # Example
/// ```ignore
/// let env = TradingEnv::new(data)?;
/// let wrapped = WrappedEnv::new(env)
///     .with_time_limit(1000)
///     .with_normalized_obs()
///     .with_stats_recording()
///     .build();
/// ```
pub struct WrappedEnv<E> {
    env: E,
}

impl<E> WrappedEnv<E> {
    /// Start building a wrapped environment.
    pub fn new(env: E) -> Self {
        Self { env }
    }

    /// Get the wrapped environment.
    pub fn build(self) -> E {
        self.env
    }
}

impl<E: Environment + Clone> WrappedEnv<E> {
    /// Add time limit wrapper.
    pub fn with_time_limit(self, max_steps: usize) -> WrappedEnv<TimeLimit<E>> {
        WrappedEnv {
            env: TimeLimit::new(self.env, max_steps),
        }
    }

    /// Add reward normalization wrapper.
    pub fn with_normalized_reward(self) -> WrappedEnv<NormalizeReward<E>> {
        WrappedEnv {
            env: NormalizeReward::new(self.env),
        }
    }

    /// Add episode statistics recording wrapper.
    pub fn with_stats_recording(self) -> WrappedEnv<RecordEpisodeStatistics<E>> {
        WrappedEnv {
            env: RecordEpisodeStatistics::new(self.env),
        }
    }
}

impl<E: Environment<ObsSpace = BoxSpace> + Clone> WrappedEnv<E> {
    /// Add observation normalization wrapper.
    pub fn with_normalized_obs(self) -> WrappedEnv<NormalizeObservation<E>> {
        WrappedEnv {
            env: NormalizeObservation::new(self.env),
        }
    }

    /// Add frame stacking wrapper.
    pub fn with_frame_stack(self, n_stack: usize) -> WrappedEnv<FrameStack<E>> {
        WrappedEnv {
            env: FrameStack::new(self.env, n_stack),
        }
    }
}

impl<E: Environment<ActSpace = BoxSpace> + Clone> WrappedEnv<E> {
    /// Add action clipping wrapper.
    pub fn with_clipped_actions(self) -> WrappedEnv<ClipAction<E>> {
        WrappedEnv {
            env: ClipAction::new(self.env),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_mean_std() {
        let mut rms = RunningMeanStd::new(2);

        // Update with some data
        rms.update(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Mean should be around [3, 4]
        let mean = rms.mean();
        assert!((mean[0] - 3.0).abs() < 0.5);
        assert!((mean[1] - 4.0).abs() < 0.5);
    }

    #[test]
    fn test_episode_stats() {
        // Test that EpisodeStats correctly tracks values
        let stats = EpisodeStats {
            episode_return: 100.0,
            episode_length: 50,
            time_elapsed: Some(5.0),
        };

        assert_eq!(stats.episode_return, 100.0);
        assert_eq!(stats.episode_length, 50);
        assert_eq!(stats.time_elapsed, Some(5.0));
    }
}
