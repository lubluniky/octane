//! Meta-learning for fast adaptation to market regime changes.
//!
//! This module implements meta-learning strategies for trading agents that can
//! quickly adapt to new market conditions:
//!
//! - [`AdaptiveAgent`] - Agent with fast adaptation capabilities
//! - [`ContextEncoder`] - Encodes recent market data into context vector
//! - [`RegimeAwarePolicy`] - Policy that conditions on detected regime
//! - MAML-style inner loop adaptation
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::strategies::{AdaptiveAgent, MetaLearningConfig, AdaptationStrategy};
//!
//! let config = MetaLearningConfig::default()
//!     .context_window(50)
//!     .adaptation_steps(3)
//!     .inner_lr(0.01);
//!
//! let agent = AdaptiveAgent::new(config, env, device)?;
//! ```

use crate::algorithms::{RLAlgorithm, TrainMetrics};
use crate::buffer::{ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, OctaneError, Result};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::Path;
use tracing::{debug, info};

/// Market regime types for regime-aware policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MarketRegime {
    /// Trending upward market.
    Bullish,
    /// Trending downward market.
    Bearish,
    /// Sideways/ranging market.
    #[default]
    Sideways,
    /// High volatility regime.
    HighVolatility,
    /// Low volatility regime.
    LowVolatility,
    /// Mean-reverting regime.
    MeanReverting,
    /// Momentum regime.
    Momentum,
    /// Unknown/transitional regime.
    Unknown,
}

impl MarketRegime {
    /// Get all regime types.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Bullish,
            Self::Bearish,
            Self::Sideways,
            Self::HighVolatility,
            Self::LowVolatility,
            Self::MeanReverting,
            Self::Momentum,
            Self::Unknown,
        ]
    }

    /// Convert to one-hot encoding.
    pub fn to_one_hot(&self) -> Vec<f32> {
        let mut one_hot = vec![0.0f32; 8];
        one_hot[self.to_index()] = 1.0;
        one_hot
    }

    /// Convert to index.
    pub fn to_index(&self) -> usize {
        match self {
            Self::Bullish => 0,
            Self::Bearish => 1,
            Self::Sideways => 2,
            Self::HighVolatility => 3,
            Self::LowVolatility => 4,
            Self::MeanReverting => 5,
            Self::Momentum => 6,
            Self::Unknown => 7,
        }
    }

    /// Create from index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Bullish,
            1 => Self::Bearish,
            2 => Self::Sideways,
            3 => Self::HighVolatility,
            4 => Self::LowVolatility,
            5 => Self::MeanReverting,
            6 => Self::Momentum,
            _ => Self::Unknown,
        }
    }
}

/// Adaptation strategy for meta-learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AdaptationStrategy {
    /// MAML-style gradient-based adaptation.
    #[default]
    MAML,
    /// Context-based adaptation (encode recent experience).
    ContextBased,
    /// Regime-aware policy selection.
    RegimeAware,
    /// Bayesian online adaptation.
    Bayesian,
    /// Ensemble-based task inference.
    EnsembleInference,
}

/// Configuration for meta-learning agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Adaptation strategy to use.
    pub adaptation_strategy: AdaptationStrategy,
    /// Window size for context encoding.
    pub context_window: usize,
    /// Number of inner loop adaptation steps (MAML).
    pub adaptation_steps: usize,
    /// Inner loop learning rate (MAML).
    pub inner_lr: f32,
    /// Outer loop (meta) learning rate.
    pub outer_lr: f32,
    /// Context encoder hidden sizes.
    pub context_hidden_sizes: Vec<usize>,
    /// Context embedding dimension.
    pub context_dim: usize,
    /// Policy hidden sizes.
    pub policy_hidden_sizes: Vec<usize>,
    /// Number of tasks for meta-training.
    pub num_tasks: usize,
    /// Samples per task for adaptation.
    pub samples_per_task: usize,
    /// Discount factor.
    pub gamma: f32,
    /// Whether to use first-order MAML (faster).
    pub first_order_maml: bool,
    /// Replay buffer size.
    pub buffer_size: usize,
    /// Batch size.
    pub batch_size: usize,
    /// How often to detect regime (in steps).
    pub regime_detection_frequency: usize,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            adaptation_strategy: AdaptationStrategy::ContextBased,
            context_window: 50,
            adaptation_steps: 3,
            inner_lr: 0.01,
            outer_lr: 3e-4,
            context_hidden_sizes: vec![128, 64],
            context_dim: 32,
            policy_hidden_sizes: vec![256, 256],
            num_tasks: 10,
            samples_per_task: 100,
            gamma: 0.99,
            first_order_maml: true,
            buffer_size: 100_000,
            batch_size: 256,
            regime_detection_frequency: 100,
            seed: None,
        }
    }
}

impl MetaLearningConfig {
    /// Create new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set adaptation strategy.
    pub fn adaptation_strategy(mut self, strategy: AdaptationStrategy) -> Self {
        self.adaptation_strategy = strategy;
        self
    }

    /// Set context window size.
    pub fn context_window(mut self, size: usize) -> Self {
        self.context_window = size;
        self
    }

    /// Set number of adaptation steps.
    pub fn adaptation_steps(mut self, steps: usize) -> Self {
        self.adaptation_steps = steps;
        self
    }

    /// Set inner loop learning rate.
    pub fn inner_lr(mut self, lr: f32) -> Self {
        self.inner_lr = lr;
        self
    }

    /// Set outer loop learning rate.
    pub fn outer_lr(mut self, lr: f32) -> Self {
        self.outer_lr = lr;
        self
    }

    /// Set context dimension.
    pub fn context_dim(mut self, dim: usize) -> Self {
        self.context_dim = dim;
        self
    }

    /// Set number of meta-training tasks.
    pub fn num_tasks(mut self, n: usize) -> Self {
        self.num_tasks = n;
        self
    }

    /// Enable/disable first-order MAML.
    pub fn first_order_maml(mut self, enabled: bool) -> Self {
        self.first_order_maml = enabled;
        self
    }

    /// Set seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.context_window == 0 {
            return Err("context_window must be positive".into());
        }
        if self.inner_lr <= 0.0 || self.outer_lr <= 0.0 {
            return Err("learning rates must be positive".into());
        }
        if self.context_dim == 0 {
            return Err("context_dim must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".into());
        }
        Ok(())
    }
}

/// Context for adaptation - recent experience.
#[derive(Debug, Clone)]
pub struct AdaptationContext {
    /// Recent observations.
    pub observations: VecDeque<Vec<f32>>,
    /// Recent actions.
    pub actions: VecDeque<Vec<f32>>,
    /// Recent rewards.
    pub rewards: VecDeque<f32>,
    /// Detected regime.
    pub detected_regime: MarketRegime,
    /// Regime confidence.
    pub regime_confidence: f32,
    /// Context embedding (computed).
    pub embedding: Option<Vec<f32>>,
}

impl AdaptationContext {
    /// Create new context with given window size.
    pub fn new(window_size: usize) -> Self {
        Self {
            observations: VecDeque::with_capacity(window_size),
            actions: VecDeque::with_capacity(window_size),
            rewards: VecDeque::with_capacity(window_size),
            detected_regime: MarketRegime::Unknown,
            regime_confidence: 0.0,
            embedding: None,
        }
    }

    /// Add experience to context.
    pub fn add(&mut self, obs: Vec<f32>, action: Vec<f32>, reward: f32, window_size: usize) {
        if self.observations.len() >= window_size {
            self.observations.pop_front();
            self.actions.pop_front();
            self.rewards.pop_front();
        }
        self.observations.push_back(obs);
        self.actions.push_back(action);
        self.rewards.push_back(reward);
    }

    /// Get recent statistics for regime detection.
    pub fn compute_statistics(&self) -> ContextStatistics {
        if self.rewards.is_empty() {
            return ContextStatistics::default();
        }

        let n = self.rewards.len() as f32;

        // Reward statistics
        let mean_reward: f32 = self.rewards.iter().sum::<f32>() / n;
        let reward_var: f32 = self
            .rewards
            .iter()
            .map(|r| (r - mean_reward).powi(2))
            .sum::<f32>()
            / n;

        // Return statistics
        let returns: Vec<f32> = self
            .rewards
            .iter()
            .zip(self.rewards.iter().skip(1))
            .map(|(r1, r2)| r2 - r1)
            .collect();

        let mean_return = if returns.is_empty() {
            0.0
        } else {
            returns.iter().sum::<f32>() / returns.len() as f32
        };

        // Trend detection (simple linear regression slope)
        let trend = if self.rewards.len() >= 2 {
            let x_mean = (self.rewards.len() - 1) as f32 / 2.0;
            let mut num = 0.0f32;
            let mut den = 0.0f32;
            for (i, &r) in self.rewards.iter().enumerate() {
                let x = i as f32 - x_mean;
                num += x * (r - mean_reward);
                den += x * x;
            }
            if den > 0.0 { num / den } else { 0.0 }
        } else {
            0.0
        };

        // Volatility (standard deviation)
        let volatility = reward_var.sqrt();

        // Autocorrelation (lag 1)
        let autocorr = if self.rewards.len() >= 2 {
            let mut num = 0.0f32;
            let den = reward_var * (self.rewards.len() - 1) as f32;
            for i in 0..(self.rewards.len() - 1) {
                let r1 = self.rewards[i] - mean_reward;
                let r2 = self.rewards[i + 1] - mean_reward;
                num += r1 * r2;
            }
            if den > 0.0 { num / den } else { 0.0 }
        } else {
            0.0
        };

        ContextStatistics {
            mean_reward,
            reward_variance: reward_var,
            mean_return,
            trend,
            volatility,
            autocorrelation: autocorr,
        }
    }
}

/// Statistics computed from context for regime detection.
#[derive(Debug, Clone, Default)]
pub struct ContextStatistics {
    /// Mean reward.
    pub mean_reward: f32,
    /// Reward variance.
    pub reward_variance: f32,
    /// Mean return (change in reward).
    pub mean_return: f32,
    /// Trend (slope).
    pub trend: f32,
    /// Volatility.
    pub volatility: f32,
    /// Autocorrelation.
    pub autocorrelation: f32,
}

impl ContextStatistics {
    /// Convert to tensor.
    pub fn to_tensor(&self, device: &Device) -> Result<Tensor> {
        let candle_device = device.to_candle()?;
        let values = vec![
            self.mean_reward,
            self.reward_variance,
            self.mean_return,
            self.trend,
            self.volatility,
            self.autocorrelation,
        ];
        Ok(Tensor::from_slice(&values, &[6], &candle_device)?)
    }
}

/// Task representation for meta-training.
#[derive(Debug, Clone)]
pub struct Task {
    /// Task ID.
    pub id: usize,
    /// Regime associated with this task.
    pub regime: MarketRegime,
    /// Support set (for adaptation).
    pub support_obs: Vec<Vec<f32>>,
    pub support_actions: Vec<Vec<f32>>,
    pub support_rewards: Vec<f32>,
    /// Query set (for evaluation).
    pub query_obs: Vec<Vec<f32>>,
    pub query_actions: Vec<Vec<f32>>,
    pub query_rewards: Vec<f32>,
}

impl Task {
    /// Create empty task.
    pub fn new(id: usize, regime: MarketRegime) -> Self {
        Self {
            id,
            regime,
            support_obs: Vec::new(),
            support_actions: Vec::new(),
            support_rewards: Vec::new(),
            query_obs: Vec::new(),
            query_actions: Vec::new(),
            query_rewards: Vec::new(),
        }
    }
}

/// Adaptive agent with meta-learning capabilities.
pub struct AdaptiveAgent<E: Environment + Clone + 'static> {
    /// Configuration.
    config: MetaLearningConfig,
    /// Vectorized environment.
    env: VecEnv<E>,
    /// Device.
    device: Device,

    /// Context encoder network.
    context_encoder_var_map: VarMap,
    /// Policy network.
    policy_var_map: VarMap,
    /// Value network.
    value_var_map: VarMap,
    /// Regime classifier.
    regime_classifier_var_map: VarMap,

    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    act_dim: usize,
    /// Whether actions are discrete.
    is_discrete: bool,

    /// Adaptation context per environment.
    contexts: Vec<AdaptationContext>,

    /// Replay buffer.
    buffer: ReplayBuffer,

    /// Current detected regime.
    current_regime: MarketRegime,

    /// Total timesteps.
    total_timesteps: usize,

    /// Regime-specific performance tracking.
    regime_performance: Vec<(MarketRegime, f32, usize)>,

    /// RNG.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> AdaptiveAgent<E> {
    /// Create a new adaptive agent.
    pub fn new(config: MetaLearningConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();
        let obs_dim = obs_space.flat_dim();
        let act_dim = act_space.flat_dim();
        let is_discrete = act_space.shape() == [1];
        let num_envs = env.num_envs();

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let context_encoder_var_map = VarMap::new();
        let policy_var_map = VarMap::new();
        let value_var_map = VarMap::new();
        let regime_classifier_var_map = VarMap::new();

        let buffer_config = ReplayBufferConfig::new(obs_dim, act_dim).capacity(config.buffer_size);
        let buffer = ReplayBuffer::new(buffer_config, device)?;

        let contexts = (0..num_envs)
            .map(|_| AdaptationContext::new(config.context_window))
            .collect();

        let regime_performance = MarketRegime::all()
            .into_iter()
            .map(|r| (r, 0.0f32, 0usize))
            .collect();

        let mut agent = Self {
            config,
            env,
            device,
            context_encoder_var_map,
            policy_var_map,
            value_var_map,
            regime_classifier_var_map,
            obs_dim,
            act_dim,
            is_discrete,
            contexts,
            buffer,
            current_regime: MarketRegime::Unknown,
            total_timesteps: 0,
            regime_performance,
            rng,
        };

        agent.init_networks()?;

        info!(
            "AdaptiveAgent initialized: strategy={:?}, context_window={}",
            agent.config.adaptation_strategy, agent.config.context_window
        );

        Ok(agent)
    }

    /// Initialize neural networks.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // Context encoder: maps recent experience to context embedding
        let vb_enc =
            VarBuilder::from_varmap(&self.context_encoder_var_map, DType::F32, &candle_device);
        // Input: flattened context (obs + action + reward for each timestep)
        let context_input_dim =
            (self.obs_dim + self.act_dim + 1) * self.config.context_window;
        let mut in_dim = context_input_dim.max(1); // Ensure at least 1

        for (i, &hidden_size) in self.config.context_hidden_sizes.iter().enumerate() {
            let _ =
                candle_nn::linear(in_dim, hidden_size, vb_enc.pp(format!("encoder.layer_{}", i)))?;
            in_dim = hidden_size;
        }
        let _ = candle_nn::linear(in_dim, self.config.context_dim, vb_enc.pp("encoder.output"))?;

        // Policy network: takes obs + context embedding
        let vb_pol = VarBuilder::from_varmap(&self.policy_var_map, DType::F32, &candle_device);
        let policy_input_dim = self.obs_dim + self.config.context_dim + 8; // +8 for regime one-hot
        let mut in_dim = policy_input_dim;

        for (i, &hidden_size) in self.config.policy_hidden_sizes.iter().enumerate() {
            let _ =
                candle_nn::linear(in_dim, hidden_size, vb_pol.pp(format!("policy.layer_{}", i)))?;
            in_dim = hidden_size;
        }
        let _ = candle_nn::linear(in_dim, self.act_dim, vb_pol.pp("policy.output"))?;

        // Value network
        let vb_val = VarBuilder::from_varmap(&self.value_var_map, DType::F32, &candle_device);
        let mut in_dim = policy_input_dim;

        for (i, &hidden_size) in self.config.policy_hidden_sizes.iter().enumerate() {
            let _ =
                candle_nn::linear(in_dim, hidden_size, vb_val.pp(format!("value.layer_{}", i)))?;
            in_dim = hidden_size;
        }
        let _ = candle_nn::linear(in_dim, 1, vb_val.pp("value.output"))?;

        // Regime classifier
        let vb_reg =
            VarBuilder::from_varmap(&self.regime_classifier_var_map, DType::F32, &candle_device);
        let _ = candle_nn::linear(self.config.context_dim, 64, vb_reg.pp("regime.layer_0"))?;
        let _ = candle_nn::linear(64, 8, vb_reg.pp("regime.output"))?; // 8 regimes

        Ok(())
    }

    /// Encode context into embedding.
    fn encode_context(&self, context: &AdaptationContext) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(
            &self.context_encoder_var_map,
            DType::F32,
            &candle_device,
        );

        // Flatten context into single vector
        let context_input_dim =
            (self.obs_dim + self.act_dim + 1) * self.config.context_window;

        let mut input_vec = vec![0.0f32; context_input_dim];
        let step_dim = self.obs_dim + self.act_dim + 1;

        for (i, ((obs, action), &reward)) in context
            .observations
            .iter()
            .zip(context.actions.iter())
            .zip(context.rewards.iter())
            .enumerate()
        {
            let start = i * step_dim;
            for (j, &o) in obs.iter().enumerate().take(self.obs_dim) {
                if start + j < input_vec.len() {
                    input_vec[start + j] = o;
                }
            }
            for (j, &a) in action.iter().enumerate().take(self.act_dim) {
                if start + self.obs_dim + j < input_vec.len() {
                    input_vec[start + self.obs_dim + j] = a;
                }
            }
            if start + self.obs_dim + self.act_dim < input_vec.len() {
                input_vec[start + self.obs_dim + self.act_dim] = reward;
            }
        }

        let input = Tensor::from_slice(&input_vec, &[1, context_input_dim], &candle_device)?;

        let mut x = input;
        let mut in_dim = context_input_dim;

        for (i, &hidden_size) in self.config.context_hidden_sizes.iter().enumerate() {
            let linear =
                candle_nn::linear(in_dim, hidden_size, vb.pp(format!("encoder.layer_{}", i)))?;
            x = linear.forward(&x)?;
            x = x.relu()?;
            in_dim = hidden_size;
        }

        let output_linear =
            candle_nn::linear(in_dim, self.config.context_dim, vb.pp("encoder.output"))?;

        Ok(output_linear.forward(&x)?)
    }

    /// Detect current market regime.
    fn detect_regime(&self, context_embedding: &Tensor) -> Result<(MarketRegime, f32)> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(
            &self.regime_classifier_var_map,
            DType::F32,
            &candle_device,
        );

        let linear1 = candle_nn::linear(self.config.context_dim, 64, vb.pp("regime.layer_0"))?;
        let x = linear1.forward(context_embedding)?.relu()?;

        let linear2 = candle_nn::linear(64, 8, vb.pp("regime.output"))?;
        let logits = linear2.forward(&x)?;

        let probs = candle_nn::ops::softmax(&logits, 1)?;
        let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;

        let (max_idx, &max_prob) = probs_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((7, &0.0));

        Ok((MarketRegime::from_index(max_idx), max_prob))
    }

    /// Forward pass through policy network.
    fn policy_forward(
        &self,
        obs: &Tensor,
        context_embedding: &Tensor,
        regime: MarketRegime,
    ) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.policy_var_map, DType::F32, &candle_device);

        let batch_size = obs.dim(0)?;

        // Broadcast context embedding
        let context_broadcast = context_embedding.broadcast_as(&[batch_size, self.config.context_dim])?;

        // Regime one-hot
        let regime_one_hot = regime.to_one_hot();
        let regime_tensor = Tensor::from_slice(&regime_one_hot, &[8], &candle_device)?;
        let regime_broadcast = regime_tensor.broadcast_as(&[batch_size, 8])?;

        // Concatenate inputs
        let input = Tensor::cat(&[obs, &context_broadcast, &regime_broadcast], 1)?;

        let mut x = input;
        let policy_input_dim = self.obs_dim + self.config.context_dim + 8;
        let mut in_dim = policy_input_dim;

        for (i, &hidden_size) in self.config.policy_hidden_sizes.iter().enumerate() {
            let linear =
                candle_nn::linear(in_dim, hidden_size, vb.pp(format!("policy.layer_{}", i)))?;
            x = linear.forward(&x)?;
            x = x.tanh()?;
            in_dim = hidden_size;
        }

        let output_linear = candle_nn::linear(in_dim, self.act_dim, vb.pp("policy.output"))?;

        let output = output_linear.forward(&x)?;

        if self.is_discrete {
            Ok(output)
        } else {
            Ok(output.tanh()?)
        }
    }

    /// MAML-style inner loop adaptation.
    fn adapt_policy(&mut self, context: &AdaptationContext) -> Result<()> {
        if context.observations.len() < self.config.samples_per_task {
            return Ok(());
        }

        // Simplified adaptation: just update based on recent rewards
        // Full MAML would require second-order gradients

        if self.config.first_order_maml {
            // First-order approximation: just do gradient descent
            let context_embedding = self.encode_context(context)?;
            let (regime, _) = self.detect_regime(&context_embedding)?;

            // Sample from context
            let candle_device = self.device.to_candle()?;
            let n_samples = context.observations.len().min(self.config.samples_per_task);

            let obs_data: Vec<f32> = context
                .observations
                .iter()
                .take(n_samples)
                .flat_map(|o| o.clone())
                .collect();
            let obs =
                Tensor::from_slice(&obs_data, &[n_samples, self.obs_dim], &candle_device)?;

            let rewards: Vec<f32> = context.rewards.iter().take(n_samples).cloned().collect();
            let rewards_tensor = Tensor::from_slice(&rewards, &[n_samples], &candle_device)?;

            // Inner loop adaptation steps
            for _ in 0..self.config.adaptation_steps {
                let actions = self.policy_forward(&obs, &context_embedding, regime)?;

                // Simple policy gradient loss
                let values_approx = actions.mean(1)?;
                let advantage = (&rewards_tensor - &values_approx)?;
                let loss = advantage.sqr()?.mean_all()?;

                // Update with inner learning rate
                let params = ParamsAdamW {
                    lr: self.config.inner_lr as f64,
                    ..Default::default()
                };
                let mut optimizer = AdamW::new(self.policy_var_map.all_vars(), params)?;
                optimizer.backward_step(&loss)?;
            }
        }

        Ok(())
    }

    /// Predict action with adaptation.
    pub fn predict(&mut self, obs: &Tensor, env_idx: usize, deterministic: bool) -> Result<Tensor> {
        let context = &self.contexts[env_idx];

        // Encode context
        let context_embedding = self.encode_context(context)?;

        // Detect regime (periodically)
        let (regime, _confidence) = if self.total_timesteps % self.config.regime_detection_frequency == 0 {
            let (r, c) = self.detect_regime(&context_embedding)?;
            self.current_regime = r;
            (r, c)
        } else {
            (self.current_regime, 0.5)
        };

        // Get action from policy
        let action_logits = self.policy_forward(obs, &context_embedding, regime)?;

        if deterministic || self.is_discrete {
            if self.is_discrete {
                Ok(action_logits.argmax(1)?.to_dtype(DType::F32)?)
            } else {
                Ok(action_logits)
            }
        } else {
            // Add exploration noise
            let noise = Tensor::randn_like(&action_logits, 0.0, 0.1)?;
            Ok((action_logits + noise)?.tanh()?)
        }
    }

    /// Update context after action.
    pub fn update_context(
        &mut self,
        env_idx: usize,
        obs: Vec<f32>,
        action: Vec<f32>,
        reward: f32,
    ) {
        self.contexts[env_idx].add(obs, action, reward, self.config.context_window);
    }

    /// Get current detected regime.
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get regime-specific performance statistics.
    pub fn regime_performance(&self) -> &[(MarketRegime, f32, usize)] {
        &self.regime_performance
    }

    /// Train meta-learning update.
    fn meta_train_step(&mut self) -> Result<f32> {
        if !self.buffer.can_sample(self.config.batch_size) {
            return Ok(0.0);
        }

        let batch = self.buffer.sample(self.config.batch_size)?;

        // Compute loss
        let context = &self.contexts[0]; // Use first env's context
        let context_embedding = self.encode_context(context)?;
        let (regime, _) = self.detect_regime(&context_embedding)?;

        let actions = self.policy_forward(&batch.observations, &context_embedding, regime)?;

        // Policy gradient loss (simplified)
        let not_done = (Tensor::ones_like(&batch.dones)? - &batch.dones)?;
        let returns = (&batch.rewards + (&not_done * self.config.gamma as f64)?)?;

        let loss = (actions.sqr()?.mean(1)? - returns)?.sqr()?.mean_all()?;
        let loss_val: f32 = loss.to_scalar()?;

        // Outer loop update
        let params = ParamsAdamW {
            lr: self.config.outer_lr as f64,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(self.policy_var_map.all_vars(), params)?;
        optimizer.backward_step(&loss)?;

        // Also update context encoder
        let mut enc_optimizer = AdamW::new(
            self.context_encoder_var_map.all_vars(),
            ParamsAdamW {
                lr: self.config.outer_lr as f64,
                ..Default::default()
            },
        )?;
        enc_optimizer.backward_step(&loss)?;

        Ok(loss_val)
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for AdaptiveAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let num_envs = self.env.num_envs();
        let obs = self.env.reset(&self.device)?;

        let mut total_reward = 0.0f32;
        let mut episode_rewards = Vec::new();

        // Collect experience
        for _ in 0..self.config.samples_per_task {
            // Get actions
            let mut actions_list = Vec::with_capacity(num_envs);
            for env_idx in 0..num_envs {
                let env_obs = obs.narrow(0, env_idx, 1)?;
                let action = self.predict(&env_obs, env_idx, false)?;
                actions_list.push(action);
            }
            let actions = Tensor::cat(&actions_list, 0)?;

            // Step environment
            let step_result = self.env.step(&actions, &self.device)?;

            // Update contexts and buffer
            let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;
            let actions_vec: Vec<f32> = actions.flatten_all()?.to_vec1()?;
            let next_obs_vec: Vec<f32> = step_result.observations.flatten_all()?.to_vec1()?;
            let rewards_vec: Vec<f32> = step_result.rewards.to_vec1()?;
            let dones_vec: Vec<f32> = step_result.dones()?.to_vec1()?;

            for env_idx in 0..num_envs {
                let obs_slice = obs_vec[env_idx * self.obs_dim..(env_idx + 1) * self.obs_dim].to_vec();
                let action_slice = actions_vec[env_idx * self.act_dim..(env_idx + 1) * self.act_dim].to_vec();
                let reward = rewards_vec[env_idx];

                self.update_context(env_idx, obs_slice.clone(), action_slice.clone(), reward);

                self.buffer.add(
                    &obs_slice,
                    &action_slice,
                    reward,
                    &next_obs_vec[env_idx * self.obs_dim..(env_idx + 1) * self.obs_dim],
                    dones_vec[env_idx] > 0.5,
                );

                total_reward += reward;

                if dones_vec[env_idx] > 0.5 {
                    episode_rewards.push(reward);
                    self.contexts[env_idx] = AdaptationContext::new(self.config.context_window);
                }
            }

            self.total_timesteps += num_envs;
        }

        // Adaptation step
        for env_idx in 0..num_envs {
            let context = self.contexts[env_idx].clone();
            self.adapt_policy(&context)?;
        }

        // Meta-training step
        let meta_loss = self.meta_train_step()?;

        // Update regime performance
        for (regime, perf, count) in self.regime_performance.iter_mut() {
            if *regime == self.current_regime {
                *perf = (*perf * *count as f32 + total_reward) / (*count + 1) as f32;
                *count += 1;
            }
        }

        let mean_reward = total_reward / (self.config.samples_per_task * num_envs) as f32;

        debug!(
            "Meta step: regime={:?}, loss={:.4}, reward={:.4}",
            self.current_regime, meta_loss, mean_reward
        );

        Ok(TrainMetrics {
            policy_loss: meta_loss,
            mean_reward,
            timesteps: self.total_timesteps,
            episodes: episode_rewards.len(),
            ..Default::default()
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let mut all_tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        // Save all networks
        for (name, var) in self.context_encoder_var_map.data().lock().unwrap().iter() {
            all_tensors.insert(format!("encoder_{}", name), var.as_tensor().clone());
        }
        for (name, var) in self.policy_var_map.data().lock().unwrap().iter() {
            all_tensors.insert(format!("policy_{}", name), var.as_tensor().clone());
        }
        for (name, var) in self.value_var_map.data().lock().unwrap().iter() {
            all_tensors.insert(format!("value_{}", name), var.as_tensor().clone());
        }
        for (name, var) in self.regime_classifier_var_map.data().lock().unwrap().iter() {
            all_tensors.insert(format!("regime_{}", name), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&all_tensors, path)?;

        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("AdaptiveAgent saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        // Load encoder
        let mut data = self.context_encoder_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("encoder_") {
                let key = name.trim_start_matches("encoder_");
                if let Some(var) = data.get_mut(key) {
                    var.set(tensor)?;
                }
            }
        }
        drop(data);

        // Load policy
        let mut data = self.policy_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("policy_") {
                let key = name.trim_start_matches("policy_");
                if let Some(var) = data.get_mut(key) {
                    var.set(tensor)?;
                }
            }
        }
        drop(data);

        // Load value
        let mut data = self.value_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("value_") {
                let key = name.trim_start_matches("value_");
                if let Some(var) = data.get_mut(key) {
                    var.set(tensor)?;
                }
            }
        }
        drop(data);

        // Load regime classifier
        let mut data = self.regime_classifier_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("regime_") {
                let key = name.trim_start_matches("regime_");
                if let Some(var) = data.get_mut(key) {
                    var.set(tensor)?;
                }
            }
        }

        info!("AdaptiveAgent loaded from {:?}", path);
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
    fn test_meta_learning_config_defaults() {
        let config = MetaLearningConfig::default();
        assert_eq!(config.context_window, 50);
        assert_eq!(config.adaptation_steps, 3);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_market_regime_conversion() {
        for (i, regime) in MarketRegime::all().iter().enumerate() {
            assert_eq!(regime.to_index(), i);
            assert_eq!(MarketRegime::from_index(i), *regime);
        }
    }

    #[test]
    fn test_context_statistics() {
        let mut context = AdaptationContext::new(10);
        context.add(vec![1.0], vec![0.0], 1.0, 10);
        context.add(vec![2.0], vec![0.0], 2.0, 10);
        context.add(vec![3.0], vec![0.0], 3.0, 10);

        let stats = context.compute_statistics();
        assert!((stats.mean_reward - 2.0).abs() < 1e-6);
    }
}
