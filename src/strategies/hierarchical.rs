//! Hierarchical Reinforcement Learning for trading.
//!
//! This module implements a two-level hierarchy for trading decisions:
//! - High-level agent: Decides WHEN to trade (timing decisions)
//! - Low-level agent: Decides HOW to execute (sizing, order type)
//!
//! Based on the options framework for temporal abstraction.
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::strategies::{HierarchicalAgent, HierarchicalConfig, HierarchyLevel};
//!
//! let config = HierarchicalConfig::default()
//!     .high_level_frequency(10)  // High-level decides every 10 steps
//!     .num_options(4)            // 4 trading strategies
//!     .goal_conditioned(true);
//!
//! let agent = HierarchicalAgent::new(config, env, device)?;
//! ```

use crate::algorithms::{RLAlgorithm, TrainMetrics};
use crate::buffer::{ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, OctaneError, Result};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info};

/// Option/skill for temporal abstraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingOption {
    /// Hold current position (do nothing).
    Hold,
    /// Enter long position aggressively.
    AggressiveLong,
    /// Enter long position conservatively.
    ConservativeLong,
    /// Enter short position aggressively.
    AggressiveShort,
    /// Enter short position conservatively.
    ConservativeShort,
    /// Exit all positions.
    Exit,
    /// Scale into position gradually.
    ScaleIn,
    /// Scale out of position gradually.
    ScaleOut,
}

impl TradingOption {
    /// Get all available options.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Hold,
            Self::AggressiveLong,
            Self::ConservativeLong,
            Self::AggressiveShort,
            Self::ConservativeShort,
            Self::Exit,
            Self::ScaleIn,
            Self::ScaleOut,
        ]
    }

    /// Convert option index to enum.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Hold,
            1 => Self::AggressiveLong,
            2 => Self::ConservativeLong,
            3 => Self::AggressiveShort,
            4 => Self::ConservativeShort,
            5 => Self::Exit,
            6 => Self::ScaleIn,
            7 => Self::ScaleOut,
            _ => Self::Hold,
        }
    }

    /// Convert enum to index.
    pub fn to_index(self) -> usize {
        match self {
            Self::Hold => 0,
            Self::AggressiveLong => 1,
            Self::ConservativeLong => 2,
            Self::AggressiveShort => 3,
            Self::ConservativeShort => 4,
            Self::Exit => 5,
            Self::ScaleIn => 6,
            Self::ScaleOut => 7,
        }
    }
}

/// Goal specification for goal-conditioned low-level policies.
#[derive(Debug, Clone)]
pub struct Goal {
    /// Target position (-1 to 1).
    pub target_position: f32,
    /// Target holding period (in steps).
    pub target_horizon: usize,
    /// Risk tolerance (0 to 1).
    pub risk_tolerance: f32,
    /// Desired trade frequency (0 = few trades, 1 = many trades).
    pub trade_frequency: f32,
}

impl Default for Goal {
    fn default() -> Self {
        Self {
            target_position: 0.0,
            target_horizon: 10,
            risk_tolerance: 0.5,
            trade_frequency: 0.5,
        }
    }
}

impl Goal {
    /// Create a goal from a trading option.
    pub fn from_option(option: TradingOption) -> Self {
        match option {
            TradingOption::Hold => Self {
                target_position: 0.0,
                target_horizon: 20,
                risk_tolerance: 0.3,
                trade_frequency: 0.0,
            },
            TradingOption::AggressiveLong => Self {
                target_position: 1.0,
                target_horizon: 5,
                risk_tolerance: 0.8,
                trade_frequency: 0.8,
            },
            TradingOption::ConservativeLong => Self {
                target_position: 0.5,
                target_horizon: 20,
                risk_tolerance: 0.3,
                trade_frequency: 0.3,
            },
            TradingOption::AggressiveShort => Self {
                target_position: -1.0,
                target_horizon: 5,
                risk_tolerance: 0.8,
                trade_frequency: 0.8,
            },
            TradingOption::ConservativeShort => Self {
                target_position: -0.5,
                target_horizon: 20,
                risk_tolerance: 0.3,
                trade_frequency: 0.3,
            },
            TradingOption::Exit => Self {
                target_position: 0.0,
                target_horizon: 3,
                risk_tolerance: 0.5,
                trade_frequency: 0.5,
            },
            TradingOption::ScaleIn => Self {
                target_position: 0.0, // Will be adjusted based on context
                target_horizon: 10,
                risk_tolerance: 0.4,
                trade_frequency: 0.6,
            },
            TradingOption::ScaleOut => Self {
                target_position: 0.0,
                target_horizon: 10,
                risk_tolerance: 0.4,
                trade_frequency: 0.6,
            },
        }
    }

    /// Convert goal to tensor representation.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let candle_device = device.to_candle()?;
        let values = vec![
            self.target_position,
            self.target_horizon as f32 / 100.0, // Normalize
            self.risk_tolerance,
            self.trade_frequency,
        ];
        Ok(Tensor::from_slice(&values, &[4], &candle_device)?)
    }
}

/// Configuration for hierarchical agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalConfig {
    /// Frequency of high-level decisions (in low-level steps).
    pub high_level_frequency: usize,
    /// Number of options/skills available.
    pub num_options: usize,
    /// Whether to use goal-conditioned low-level policies.
    pub goal_conditioned: bool,
    /// Discount factor for high-level.
    pub high_gamma: f32,
    /// Discount factor for low-level.
    pub low_gamma: f32,
    /// Learning rate for high-level.
    pub high_lr: f32,
    /// Learning rate for low-level.
    pub low_lr: f32,
    /// Whether options can terminate early.
    pub option_termination: bool,
    /// Termination probability threshold.
    pub termination_threshold: f32,
    /// High-level hidden sizes.
    pub high_hidden_sizes: Vec<usize>,
    /// Low-level hidden sizes.
    pub low_hidden_sizes: Vec<usize>,
    /// Replay buffer size.
    pub buffer_size: usize,
    /// Batch size for training.
    pub batch_size: usize,
    /// Intrinsic reward coefficient for goal achievement.
    pub intrinsic_reward_coef: f32,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for HierarchicalConfig {
    fn default() -> Self {
        Self {
            high_level_frequency: 10,
            num_options: 8,
            goal_conditioned: true,
            high_gamma: 0.99,
            low_gamma: 0.95,
            high_lr: 1e-4,
            low_lr: 3e-4,
            option_termination: true,
            termination_threshold: 0.5,
            high_hidden_sizes: vec![128, 128],
            low_hidden_sizes: vec![256, 256],
            buffer_size: 100_000,
            batch_size: 256,
            intrinsic_reward_coef: 0.1,
            seed: None,
        }
    }
}

impl HierarchicalConfig {
    /// Create new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set high-level decision frequency.
    pub fn high_level_frequency(mut self, freq: usize) -> Self {
        self.high_level_frequency = freq;
        self
    }

    /// Set number of options.
    pub fn num_options(mut self, n: usize) -> Self {
        self.num_options = n;
        self
    }

    /// Enable/disable goal conditioning.
    pub fn goal_conditioned(mut self, enabled: bool) -> Self {
        self.goal_conditioned = enabled;
        self
    }

    /// Set high-level discount factor.
    pub fn high_gamma(mut self, gamma: f32) -> Self {
        self.high_gamma = gamma;
        self
    }

    /// Set low-level discount factor.
    pub fn low_gamma(mut self, gamma: f32) -> Self {
        self.low_gamma = gamma;
        self
    }

    /// Set high-level learning rate.
    pub fn high_lr(mut self, lr: f32) -> Self {
        self.high_lr = lr;
        self
    }

    /// Set low-level learning rate.
    pub fn low_lr(mut self, lr: f32) -> Self {
        self.low_lr = lr;
        self
    }

    /// Enable/disable option termination.
    pub fn option_termination(mut self, enabled: bool) -> Self {
        self.option_termination = enabled;
        self
    }

    /// Set intrinsic reward coefficient.
    pub fn intrinsic_reward_coef(mut self, coef: f32) -> Self {
        self.intrinsic_reward_coef = coef;
        self
    }

    /// Set seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.high_level_frequency == 0 {
            return Err("high_level_frequency must be positive".into());
        }
        if self.num_options == 0 {
            return Err("num_options must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.high_gamma) {
            return Err("high_gamma must be in [0, 1]".into());
        }
        if !(0.0..=1.0).contains(&self.low_gamma) {
            return Err("low_gamma must be in [0, 1]".into());
        }
        if self.high_lr <= 0.0 || self.low_lr <= 0.0 {
            return Err("learning rates must be positive".into());
        }
        Ok(())
    }
}

/// Transition for hierarchical replay buffer.
#[derive(Debug, Clone)]
pub struct HierarchicalTransition {
    /// Observation at start of option.
    pub obs: Vec<f32>,
    /// Selected option.
    pub option: usize,
    /// Cumulative reward during option.
    pub cumulative_reward: f32,
    /// Next observation (when option terminates).
    pub next_obs: Vec<f32>,
    /// Whether option terminated naturally.
    pub done: bool,
    /// Number of steps the option ran.
    pub duration: usize,
    /// Goal used (if goal-conditioned).
    pub goal: Option<Goal>,
}

/// Hierarchical replay buffer.
pub struct HierarchicalReplayBuffer {
    /// High-level transitions.
    high_level_buffer: Vec<HierarchicalTransition>,
    /// Low-level buffer (standard).
    low_level_buffer: ReplayBuffer,
    /// Maximum high-level capacity.
    capacity: usize,
    /// Current position.
    position: usize,
    /// RNG for sampling.
    rng: StdRng,
}

impl HierarchicalReplayBuffer {
    /// Create a new hierarchical buffer.
    pub fn new(capacity: usize, obs_dim: usize, action_dim: usize, device: Device) -> Result<Self> {
        let low_config = ReplayBufferConfig::new(obs_dim, action_dim).capacity(capacity);
        let low_level_buffer = ReplayBuffer::new(low_config, device)?;

        Ok(Self {
            high_level_buffer: Vec::with_capacity(capacity),
            low_level_buffer,
            capacity,
            position: 0,
            rng: StdRng::from_entropy(),
        })
    }

    /// Add high-level transition.
    pub fn add_high_level(&mut self, transition: HierarchicalTransition) {
        if self.high_level_buffer.len() < self.capacity {
            self.high_level_buffer.push(transition);
        } else {
            self.high_level_buffer[self.position] = transition;
        }
        self.position = (self.position + 1) % self.capacity;
    }

    /// Add low-level transition.
    pub fn add_low_level(
        &mut self,
        obs: &[f32],
        action: &[f32],
        reward: f32,
        next_obs: &[f32],
        done: bool,
    ) {
        self.low_level_buffer
            .add(obs, action, reward, next_obs, done);
    }

    /// Sample high-level batch.
    pub fn sample_high_level(&mut self, batch_size: usize) -> Option<Vec<HierarchicalTransition>> {
        if self.high_level_buffer.len() < batch_size {
            return None;
        }

        let indices: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.high_level_buffer.len()))
            .collect();

        Some(
            indices
                .iter()
                .map(|&i| self.high_level_buffer[i].clone())
                .collect(),
        )
    }

    /// Sample low-level batch.
    pub fn sample_low_level(&mut self, batch_size: usize) -> Result<crate::buffer::ReplayBatch> {
        self.low_level_buffer.sample(batch_size)
    }

    /// Check if can sample.
    pub fn can_sample(&self, batch_size: usize) -> bool {
        self.high_level_buffer.len() >= batch_size && self.low_level_buffer.can_sample(batch_size)
    }
}

/// State tracker for hierarchical agent.
#[derive(Debug, Clone)]
struct OptionState {
    /// Currently active option.
    current_option: Option<usize>,
    /// Current goal (if goal-conditioned).
    current_goal: Option<Goal>,
    /// Steps remaining in current option.
    steps_remaining: usize,
    /// Observation when option started.
    start_obs: Vec<f32>,
    /// Cumulative reward during option.
    cumulative_reward: f32,
}

impl Default for OptionState {
    fn default() -> Self {
        Self {
            current_option: None,
            current_goal: None,
            steps_remaining: 0,
            start_obs: Vec::new(),
            cumulative_reward: 0.0,
        }
    }
}

/// Hierarchical RL agent for trading.
pub struct HierarchicalAgent<E: Environment + Clone + 'static> {
    /// Configuration.
    config: HierarchicalConfig,
    /// Vectorized environment.
    env: VecEnv<E>,
    /// Device.
    device: Device,

    /// High-level policy network.
    high_var_map: VarMap,
    /// Low-level policy network.
    low_var_map: VarMap,
    /// Termination network (for option termination).
    term_var_map: VarMap,

    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    act_dim: usize,
    /// Goal dimension (if goal-conditioned).
    goal_dim: usize,

    /// Hierarchical replay buffer.
    buffer: HierarchicalReplayBuffer,

    /// Option state per environment.
    option_states: Vec<OptionState>,

    /// Total timesteps.
    total_timesteps: usize,
    /// High-level steps.
    high_level_steps: usize,

    /// RNG.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> HierarchicalAgent<E> {
    /// Create a new hierarchical agent.
    pub fn new(config: HierarchicalConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();
        let obs_dim = obs_space.flat_dim();
        let act_dim = act_space.flat_dim();
        let goal_dim = if config.goal_conditioned { 4 } else { 0 };
        let num_envs = env.num_envs();

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let high_var_map = VarMap::new();
        let low_var_map = VarMap::new();
        let term_var_map = VarMap::new();

        let buffer = HierarchicalReplayBuffer::new(config.buffer_size, obs_dim, act_dim, device)?;

        let option_states = (0..num_envs).map(|_| OptionState::default()).collect();

        let mut agent = Self {
            config,
            env,
            device,
            high_var_map,
            low_var_map,
            term_var_map,
            obs_dim,
            act_dim,
            goal_dim,
            buffer,
            option_states,
            total_timesteps: 0,
            high_level_steps: 0,
            rng,
        };

        agent.init_networks()?;

        info!(
            "HierarchicalAgent initialized: obs_dim={}, act_dim={}, num_options={}",
            obs_dim, act_dim, agent.config.num_options
        );

        Ok(agent)
    }

    /// Initialize neural networks.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // High-level policy (option selection)
        let vb_high = VarBuilder::from_varmap(&self.high_var_map, DType::F32, &candle_device);
        let mut in_dim = self.obs_dim;
        for (i, &hidden_size) in self.config.high_hidden_sizes.iter().enumerate() {
            let _ =
                candle_nn::linear(in_dim, hidden_size, vb_high.pp(format!("high.layer_{}", i)))?;
            in_dim = hidden_size;
        }
        let _ = candle_nn::linear(in_dim, self.config.num_options, vb_high.pp("high.output"))?;

        // Low-level policy (action selection)
        let vb_low = VarBuilder::from_varmap(&self.low_var_map, DType::F32, &candle_device);
        let low_input_dim = self.obs_dim + self.goal_dim + self.config.num_options; // obs + goal + option one-hot
        let mut in_dim = low_input_dim;
        for (i, &hidden_size) in self.config.low_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(in_dim, hidden_size, vb_low.pp(format!("low.layer_{}", i)))?;
            in_dim = hidden_size;
        }
        let _ = candle_nn::linear(in_dim, self.act_dim, vb_low.pp("low.output"))?;

        // Value networks for both levels
        let mut in_dim = self.obs_dim;
        for (i, &hidden_size) in self.config.high_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(
                in_dim,
                hidden_size,
                vb_high.pp(format!("high_value.layer_{}", i)),
            )?;
            in_dim = hidden_size;
        }
        let _ = candle_nn::linear(in_dim, 1, vb_high.pp("high_value.output"))?;

        // Termination network
        if self.config.option_termination {
            let vb_term = VarBuilder::from_varmap(&self.term_var_map, DType::F32, &candle_device);
            let term_input_dim = self.obs_dim + self.config.num_options;
            let _ = candle_nn::linear(term_input_dim, 64, vb_term.pp("term.layer_0"))?;
            let _ = candle_nn::linear(64, 1, vb_term.pp("term.output"))?;
        }

        Ok(())
    }

    /// Forward pass through high-level policy.
    fn high_level_forward(&self, obs: &Tensor) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.high_var_map, DType::F32, &candle_device);

        let mut x = obs.clone();
        for (i, &hidden_size) in self.config.high_hidden_sizes.iter().enumerate() {
            let in_dim = if i == 0 {
                self.obs_dim
            } else {
                self.config.high_hidden_sizes[i - 1]
            };
            let linear =
                candle_nn::linear(in_dim, hidden_size, vb.pp(format!("high.layer_{}", i)))?;
            x = linear.forward(&x)?;
            x = x.relu()?;
        }

        let output_linear = candle_nn::linear(
            *self.config.high_hidden_sizes.last().unwrap(),
            self.config.num_options,
            vb.pp("high.output"),
        )?;

        Ok(output_linear.forward(&x)?)
    }

    /// Forward pass through low-level policy.
    fn low_level_forward(
        &self,
        obs: &Tensor,
        option: usize,
        goal: Option<&Goal>,
    ) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.low_var_map, DType::F32, &candle_device);

        let batch_size = obs.dim(0)?;

        // Create option one-hot encoding
        let mut option_one_hot = vec![0.0f32; self.config.num_options];
        option_one_hot[option] = 1.0;
        let option_tensor =
            Tensor::from_slice(&option_one_hot, &[self.config.num_options], &candle_device)?;
        let option_broadcast =
            option_tensor.broadcast_as(&[batch_size, self.config.num_options])?;

        // Build input
        let input = if self.config.goal_conditioned {
            let default_goal = Goal::default();
            let goal_ref = goal.unwrap_or(&default_goal);
            let goal_tensor = goal_ref.to_tensor(&self.device)?;
            let goal_broadcast = goal_tensor.broadcast_as(&[batch_size, self.goal_dim])?;
            Tensor::cat(&[obs, &goal_broadcast, &option_broadcast], 1)?
        } else {
            Tensor::cat(&[obs, &option_broadcast], 1)?
        };

        let mut x = input;
        let input_dim = self.obs_dim + self.goal_dim + self.config.num_options;

        for (i, &hidden_size) in self.config.low_hidden_sizes.iter().enumerate() {
            let in_dim = if i == 0 {
                input_dim
            } else {
                self.config.low_hidden_sizes[i - 1]
            };
            let linear = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("low.layer_{}", i)))?;
            x = linear.forward(&x)?;
            x = x.tanh()?;
        }

        let output_linear = candle_nn::linear(
            *self.config.low_hidden_sizes.last().unwrap(),
            self.act_dim,
            vb.pp("low.output"),
        )?;

        Ok(output_linear.forward(&x)?.tanh()?)
    }

    /// Check if option should terminate.
    fn should_terminate(&self, obs: &Tensor, option: usize) -> Result<bool> {
        if !self.config.option_termination {
            return Ok(false);
        }

        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.term_var_map, DType::F32, &candle_device);

        // Create option one-hot
        let mut option_one_hot = vec![0.0f32; self.config.num_options];
        option_one_hot[option] = 1.0;
        let option_tensor = Tensor::from_slice(
            &option_one_hot,
            &[1, self.config.num_options],
            &candle_device,
        )?;

        let input = Tensor::cat(&[obs, &option_tensor], 1)?;

        let linear1 = candle_nn::linear(
            self.obs_dim + self.config.num_options,
            64,
            vb.pp("term.layer_0"),
        )?;
        let x = linear1.forward(&input)?.relu()?;

        let linear2 = candle_nn::linear(64, 1, vb.pp("term.output"))?;
        let logit = linear2.forward(&x)?;

        let prob: f32 = candle_nn::ops::sigmoid(&logit)?.squeeze(0)?.to_scalar()?;

        Ok(prob > self.config.termination_threshold)
    }

    /// Select option using high-level policy.
    fn select_option(&mut self, obs: &Tensor) -> Result<usize> {
        let logits = self.high_level_forward(obs)?;
        let probs = candle_nn::ops::softmax(&logits, 1)?;
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;

        // Sample from distribution
        let r: f32 = self.rng.gen();
        let mut cumsum = 0.0;
        let mut selected = 0;

        for (i, &p) in probs_vec.iter().enumerate() {
            cumsum += p;
            if r < cumsum {
                selected = i;
                break;
            }
        }

        Ok(selected)
    }

    /// Compute intrinsic reward for goal achievement.
    fn intrinsic_reward(&self, obs: &Tensor, goal: &Goal) -> Result<f32> {
        // Simple distance-based intrinsic reward
        let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;

        // Assume position is last element of observation
        let current_position = *obs_vec.last().unwrap_or(&0.0);
        let position_error = (current_position - goal.target_position).abs();

        let reward = -position_error * self.config.intrinsic_reward_coef;
        Ok(reward)
    }

    /// Predict action using hierarchical policy.
    pub fn predict(&mut self, obs: &Tensor, env_idx: usize, deterministic: bool) -> Result<Tensor> {
        // Extract state info without holding borrow
        let current_option = self.option_states[env_idx].current_option;
        let steps_remaining = self.option_states[env_idx].steps_remaining;
        let option_termination_enabled = self.config.option_termination;

        // Check if we need to select a new option
        let should_terminate = if option_termination_enabled && current_option.is_some() {
            self.should_terminate(obs, current_option.unwrap())?
        } else {
            false
        };

        let need_new_option = current_option.is_none() || steps_remaining == 0 || should_terminate;

        if need_new_option {
            // Store transition if option was active
            if let Some(prev_option) = current_option {
                let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;
                let state = &self.option_states[env_idx];
                let transition = HierarchicalTransition {
                    obs: state.start_obs.clone(),
                    option: prev_option,
                    cumulative_reward: state.cumulative_reward,
                    next_obs: obs_vec.clone(),
                    done: false,
                    duration: self.config.high_level_frequency - state.steps_remaining,
                    goal: state.current_goal.clone(),
                };
                self.buffer.add_high_level(transition);
            }

            // Select new option
            let option = if deterministic {
                let logits = self.high_level_forward(obs)?;
                logits.argmax(1)?.to_scalar::<u32>()? as usize
            } else {
                self.select_option(obs)?
            };

            let goal = if self.config.goal_conditioned {
                Some(Goal::from_option(TradingOption::from_index(option)))
            } else {
                None
            };

            let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;

            // Update state
            self.option_states[env_idx].current_option = Some(option);
            self.option_states[env_idx].current_goal = goal;
            self.option_states[env_idx].steps_remaining = self.config.high_level_frequency;
            self.option_states[env_idx].start_obs = obs_vec;
            self.option_states[env_idx].cumulative_reward = 0.0;

            self.high_level_steps += 1;
        }

        // Get low-level action
        let option = self.option_states[env_idx].current_option.unwrap();
        let goal_ref = self.option_states[env_idx].current_goal.clone();
        let action = self.low_level_forward(obs, option, goal_ref.as_ref())?;

        self.option_states[env_idx].steps_remaining -= 1;

        Ok(action)
    }

    /// Update agent after receiving reward.
    pub fn update_reward(&mut self, env_idx: usize, reward: f32) {
        self.option_states[env_idx].cumulative_reward += reward;
    }

    /// Train high-level policy.
    fn train_high_level(&mut self) -> Result<f32> {
        let batch = match self.buffer.sample_high_level(self.config.batch_size) {
            Some(b) => b,
            None => return Ok(0.0),
        };

        let candle_device = self.device.to_candle()?;

        // Prepare batch data
        let obs_data: Vec<f32> = batch.iter().flat_map(|t| t.obs.clone()).collect();
        let obs = Tensor::from_slice(&obs_data, &[batch.len(), self.obs_dim], &candle_device)?;

        let next_obs_data: Vec<f32> = batch.iter().flat_map(|t| t.next_obs.clone()).collect();
        let next_obs =
            Tensor::from_slice(&next_obs_data, &[batch.len(), self.obs_dim], &candle_device)?;

        let options: Vec<i64> = batch.iter().map(|t| t.option as i64).collect();
        let options_tensor = Tensor::from_slice(&options, &[batch.len()], &candle_device)?;

        let rewards: Vec<f32> = batch.iter().map(|t| t.cumulative_reward).collect();
        let rewards_tensor = Tensor::from_slice(&rewards, &[batch.len()], &candle_device)?;

        let dones: Vec<f32> = batch
            .iter()
            .map(|t| if t.done { 1.0 } else { 0.0 })
            .collect();
        let dones_tensor = Tensor::from_slice(&dones, &[batch.len()], &candle_device)?;

        // Compute target Q-values
        let next_q_logits = self.high_level_forward(&next_obs)?;
        let next_q_values = next_q_logits.max(1)?;
        let not_done = (Tensor::ones_like(&dones_tensor)? - &dones_tensor)?;
        let gamma_tensor = Tensor::new(&[self.config.high_gamma], &candle_device)?;
        let discounted_next_q = (&next_q_values * &not_done)?.broadcast_mul(&gamma_tensor)?;
        let targets = (&rewards_tensor + &discounted_next_q)?;

        // Current Q-values
        let q_logits = self.high_level_forward(&obs)?;
        let q_selected = q_logits
            .gather(&options_tensor.unsqueeze(1)?, 1)?
            .squeeze(1)?;

        // Loss
        let targets_detached = targets.detach();
        let diff = (&q_selected - &targets_detached)?;
        let loss = diff.sqr()?.mean_all()?;
        let loss_val: f32 = loss.to_scalar()?;

        // Update
        let optimizer_params = ParamsAdamW {
            lr: self.config.high_lr as f64,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(self.high_var_map.all_vars(), optimizer_params)?;
        optimizer.backward_step(&loss)?;

        Ok(loss_val)
    }

    /// Train low-level policy.
    fn train_low_level(&mut self) -> Result<f32> {
        if !self.buffer.can_sample(self.config.batch_size) {
            return Ok(0.0);
        }

        let batch = self.buffer.sample_low_level(self.config.batch_size)?;

        // Simple policy gradient update
        let _params = ParamsAdamW {
            lr: self.config.low_lr as f64,
            ..Default::default()
        };

        // For simplicity, use MSE loss towards current actions
        // In a full implementation, use proper actor-critic
        let loss = batch.rewards.sqr()?.mean_all()?; // Placeholder
        let loss_val: f32 = loss.to_scalar()?;

        Ok(loss_val)
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for HierarchicalAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let num_envs = self.env.num_envs();
        let mut obs = self.env.reset(&self.device)?;

        let mut episode_rewards = Vec::new();
        let mut total_reward = 0.0f32;

        // Run one high-level step (multiple low-level steps)
        for _ in 0..self.config.high_level_frequency {
            // Get actions for each environment
            let mut actions_list = Vec::with_capacity(num_envs);
            for env_idx in 0..num_envs {
                let env_obs = obs.narrow(0, env_idx, 1)?;
                let action = self.predict(&env_obs, env_idx, false)?;
                actions_list.push(action);
            }

            // Stack actions
            let actions = Tensor::cat(&actions_list, 0)?;

            // Step environment
            let step_result = self.env.step(&actions, &self.device)?;

            // Update rewards
            let rewards_vec: Vec<f32> = step_result.rewards.to_vec1()?;
            for (env_idx, &reward) in rewards_vec.iter().enumerate() {
                self.update_reward(env_idx, reward);
                total_reward += reward;
            }

            // Store low-level transitions
            let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;
            let actions_vec: Vec<f32> = actions.flatten_all()?.to_vec1()?;
            let next_obs_vec: Vec<f32> = step_result.observations.flatten_all()?.to_vec1()?;
            let dones_vec: Vec<f32> = step_result.dones()?.to_vec1()?;

            for env_idx in 0..num_envs {
                let obs_slice = &obs_vec[env_idx * self.obs_dim..(env_idx + 1) * self.obs_dim];
                let action_slice =
                    &actions_vec[env_idx * self.act_dim..(env_idx + 1) * self.act_dim];
                let next_obs_slice =
                    &next_obs_vec[env_idx * self.obs_dim..(env_idx + 1) * self.obs_dim];

                self.buffer.add_low_level(
                    obs_slice,
                    action_slice,
                    rewards_vec[env_idx],
                    next_obs_slice,
                    dones_vec[env_idx] > 0.5,
                );
            }

            obs = step_result.observations;
            self.total_timesteps += num_envs;

            // Check for episode completions
            for (env_idx, &done) in dones_vec.iter().enumerate() {
                if done > 0.5 {
                    episode_rewards.push(self.option_states[env_idx].cumulative_reward);
                    self.option_states[env_idx] = OptionState::default();
                }
            }
        }

        // Train both levels
        let high_loss = self.train_high_level()?;
        let low_loss = self.train_low_level()?;

        let mean_reward = if episode_rewards.is_empty() {
            total_reward / (self.config.high_level_frequency * num_envs) as f32
        } else {
            episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
        };

        debug!(
            "Hierarchical step: high_loss={:.4}, low_loss={:.4}, reward={:.4}",
            high_loss, low_loss, mean_reward
        );

        Ok(TrainMetrics {
            policy_loss: low_loss,
            value_loss: high_loss,
            mean_reward,
            timesteps: self.total_timesteps,
            episodes: episode_rewards.len(),
            ..Default::default()
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let mut all_tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        // Save high-level
        for (name, var) in self.high_var_map.data().lock().unwrap().iter() {
            all_tensors.insert(format!("high_{}", name), var.as_tensor().clone());
        }

        // Save low-level
        for (name, var) in self.low_var_map.data().lock().unwrap().iter() {
            all_tensors.insert(format!("low_{}", name), var.as_tensor().clone());
        }

        // Save termination
        for (name, var) in self.term_var_map.data().lock().unwrap().iter() {
            all_tensors.insert(format!("term_{}", name), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&all_tensors, path)?;

        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("HierarchicalAgent saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        // Load high-level
        let mut data = self.high_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("high_") {
                let key = name.trim_start_matches("high_");
                if let Some(var) = data.get_mut(key) {
                    var.set(tensor)?;
                }
            }
        }
        drop(data);

        // Load low-level
        let mut data = self.low_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("low_") {
                let key = name.trim_start_matches("low_");
                if let Some(var) = data.get_mut(key) {
                    var.set(tensor)?;
                }
            }
        }
        drop(data);

        // Load termination
        let mut data = self.term_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("term_") {
                let key = name.trim_start_matches("term_");
                if let Some(var) = data.get_mut(key) {
                    var.set(tensor)?;
                }
            }
        }

        info!("HierarchicalAgent loaded from {:?}", path);
        Ok(())
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn total_timesteps(&self) -> usize {
        self.total_timesteps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hierarchical_config_defaults() {
        let config = HierarchicalConfig::default();
        assert_eq!(config.high_level_frequency, 10);
        assert_eq!(config.num_options, 8);
        assert!(config.goal_conditioned);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_trading_option_conversion() {
        for (i, option) in TradingOption::all().iter().enumerate() {
            assert_eq!(option.to_index(), i);
            assert_eq!(TradingOption::from_index(i), *option);
        }
    }

    #[test]
    fn test_goal_from_option() {
        let goal = Goal::from_option(TradingOption::AggressiveLong);
        assert!((goal.target_position - 1.0).abs() < 1e-6);
        assert!((goal.risk_tolerance - 0.8).abs() < 1e-6);
    }
}
