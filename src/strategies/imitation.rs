//! Imitation Learning for trading from expert demonstrations.
//!
//! This module provides imitation learning algorithms to learn from
//! profitable trading strategies:
//!
//! - [`ImitationAgent`] - Agent that learns from demonstrations
//! - [`BehavioralCloning`] - Supervised learning from state-action pairs
//! - [`DAgger`] - Dataset Aggregation for iterative improvement
//! - [`DemoReplayBuffer`] - Buffer for storing expert demonstrations
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::strategies::{ImitationAgent, ImitationConfig, ImitationMethod};
//!
//! let config = ImitationConfig::default()
//!     .method(ImitationMethod::DAgger)
//!     .demo_buffer_size(10000)
//!     .mixture_ratio(0.5);
//!
//! let agent = ImitationAgent::new(config, env, device)?;
//! agent.load_demonstrations("expert_demos.safetensors")?;
//! ```

use crate::algorithms::{RLAlgorithm, TrainMetrics};
use crate::core::{Device, OctaneError, Result};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info};

/// Imitation learning method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ImitationMethod {
    /// Behavioral Cloning: supervised learning from demonstrations.
    #[default]
    BehavioralCloning,
    /// Dataset Aggregation: iteratively collect and add to dataset.
    DAgger,
    /// Generative Adversarial Imitation Learning (simplified).
    GAIL,
    /// Inverse Reinforcement Learning (simplified).
    IRL,
}

/// Loss function for imitation learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ImitationLoss {
    /// Mean Squared Error (for continuous actions).
    #[default]
    MSE,
    /// Cross-entropy (for discrete actions).
    CrossEntropy,
    /// Huber loss (robust to outliers).
    Huber,
    /// L1 loss.
    L1,
    /// Negative log-likelihood.
    NLL,
}

/// Expert trajectory demonstration.
#[derive(Debug, Clone)]
pub struct Demonstration {
    /// Sequence of observations.
    pub observations: Vec<Vec<f32>>,
    /// Sequence of actions.
    pub actions: Vec<Vec<f32>>,
    /// Sequence of rewards (optional, for reward shaping).
    pub rewards: Option<Vec<f32>>,
    /// Episode return.
    pub episode_return: f32,
    /// Metadata (e.g., strategy name, timestamp).
    pub metadata: DemoMetadata,
}

/// Metadata for demonstrations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DemoMetadata {
    /// Strategy/expert name.
    pub strategy_name: String,
    /// Timestamp when recorded.
    pub timestamp: u64,
    /// Market conditions during recording.
    pub market_regime: String,
    /// Quality score (if available).
    pub quality_score: f32,
}

impl Demonstration {
    /// Create a new demonstration.
    pub fn new(episode_return: f32) -> Self {
        Self {
            observations: Vec::new(),
            actions: Vec::new(),
            rewards: None,
            episode_return,
            metadata: DemoMetadata::default(),
        }
    }

    /// Add a transition to the demonstration.
    pub fn add(&mut self, obs: Vec<f32>, action: Vec<f32>, reward: Option<f32>) {
        self.observations.push(obs);
        self.actions.push(action);
        if let Some(r) = reward {
            if self.rewards.is_none() {
                self.rewards = Some(Vec::new());
            }
            self.rewards.as_mut().unwrap().push(r);
        }
    }

    /// Get the length of the demonstration.
    pub fn len(&self) -> usize {
        self.observations.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.observations.is_empty()
    }

    /// Get quality (episode return).
    pub fn quality(&self) -> f32 {
        self.episode_return
    }
}

/// Buffer for storing expert demonstrations.
pub struct DemoReplayBuffer {
    /// Stored demonstrations.
    demonstrations: Vec<Demonstration>,
    /// Flattened observation data for efficient sampling.
    obs_data: Vec<f32>,
    /// Flattened action data.
    action_data: Vec<f32>,
    /// Optional reward data.
    reward_data: Vec<f32>,
    /// Quality weights for sampling.
    quality_weights: Vec<f32>,
    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    action_dim: usize,
    /// Total number of transitions.
    total_transitions: usize,
    /// Maximum capacity.
    capacity: usize,
    /// RNG.
    rng: StdRng,
}

impl DemoReplayBuffer {
    /// Create a new demonstration buffer.
    pub fn new(obs_dim: usize, action_dim: usize, capacity: usize) -> Self {
        Self {
            demonstrations: Vec::new(),
            obs_data: Vec::new(),
            action_data: Vec::new(),
            reward_data: Vec::new(),
            quality_weights: Vec::new(),
            obs_dim,
            action_dim,
            total_transitions: 0,
            capacity,
            rng: StdRng::from_entropy(),
        }
    }

    /// Add a demonstration to the buffer.
    pub fn add_demonstration(&mut self, demo: Demonstration) {
        // Check capacity
        while self.total_transitions + demo.len() > self.capacity && !self.demonstrations.is_empty()
        {
            // Remove oldest demonstration
            let removed = self.demonstrations.remove(0);
            self.total_transitions -= removed.len();

            // Rebuild flattened data (expensive but simple)
            self.rebuild_flattened_data();
        }

        let quality = demo.quality();
        let demo_len = demo.len();

        // Add to flattened data
        for obs in &demo.observations {
            self.obs_data.extend(obs);
        }
        for action in &demo.actions {
            self.action_data.extend(action);
        }
        if let Some(rewards) = &demo.rewards {
            self.reward_data.extend(rewards);
        } else {
            self.reward_data.extend(vec![0.0; demo_len]);
        }

        // Quality weights (higher return = higher weight)
        let weight = (quality + 100.0).max(1.0); // Shift to positive
        for _ in 0..demo_len {
            self.quality_weights.push(weight);
        }

        self.total_transitions += demo_len;
        self.demonstrations.push(demo);
    }

    /// Rebuild flattened data from demonstrations.
    fn rebuild_flattened_data(&mut self) {
        self.obs_data.clear();
        self.action_data.clear();
        self.reward_data.clear();
        self.quality_weights.clear();
        self.total_transitions = 0;

        for demo in &self.demonstrations {
            let quality = demo.quality();
            let weight = (quality + 100.0).max(1.0);

            for obs in &demo.observations {
                self.obs_data.extend(obs);
            }
            for action in &demo.actions {
                self.action_data.extend(action);
            }
            if let Some(rewards) = &demo.rewards {
                self.reward_data.extend(rewards);
            } else {
                self.reward_data.extend(vec![0.0; demo.len()]);
            }
            for _ in 0..demo.len() {
                self.quality_weights.push(weight);
            }
            self.total_transitions += demo.len();
        }
    }

    /// Sample a batch uniformly.
    pub fn sample_uniform(&mut self, batch_size: usize, device: &Device) -> Result<DemoBatch> {
        if self.total_transitions < batch_size {
            return Err(OctaneError::Buffer(format!(
                "Not enough samples: {} < {}",
                self.total_transitions, batch_size
            )));
        }

        let indices: Vec<usize> = (0..batch_size)
            .map(|_| self.rng.gen_range(0..self.total_transitions))
            .collect();

        self.get_batch(&indices, device)
    }

    /// Sample a batch weighted by quality.
    pub fn sample_weighted(&mut self, batch_size: usize, device: &Device) -> Result<DemoBatch> {
        if self.total_transitions < batch_size {
            return Err(OctaneError::Buffer(format!(
                "Not enough samples: {} < {}",
                self.total_transitions, batch_size
            )));
        }

        // Normalize weights
        let total_weight: f32 = self.quality_weights.iter().sum();
        let probs: Vec<f32> = self
            .quality_weights
            .iter()
            .map(|w| w / total_weight)
            .collect();

        // Cumulative distribution
        let mut cumsum = Vec::with_capacity(self.total_transitions);
        let mut sum = 0.0f32;
        for p in probs {
            sum += p;
            cumsum.push(sum);
        }

        // Sample indices
        let indices: Vec<usize> = (0..batch_size)
            .map(|_| {
                let r: f32 = self.rng.gen();
                cumsum
                    .iter()
                    .position(|&c| r < c)
                    .unwrap_or(self.total_transitions - 1)
            })
            .collect();

        self.get_batch(&indices, device)
    }

    /// Get batch from indices.
    fn get_batch(&self, indices: &[usize], device: &Device) -> Result<DemoBatch> {
        let batch_size = indices.len();

        let mut obs_batch = Vec::with_capacity(batch_size * self.obs_dim);
        let mut action_batch = Vec::with_capacity(batch_size * self.action_dim);
        let mut reward_batch = Vec::with_capacity(batch_size);

        for &idx in indices {
            let obs_start = idx * self.obs_dim;
            obs_batch.extend_from_slice(&self.obs_data[obs_start..obs_start + self.obs_dim]);

            let action_start = idx * self.action_dim;
            action_batch
                .extend_from_slice(&self.action_data[action_start..action_start + self.action_dim]);

            if idx < self.reward_data.len() {
                reward_batch.push(self.reward_data[idx]);
            } else {
                reward_batch.push(0.0);
            }
        }

        let candle_device = device.to_candle()?;

        Ok(DemoBatch {
            observations: Tensor::from_slice(
                &obs_batch,
                &[batch_size, self.obs_dim],
                &candle_device,
            )?,
            actions: Tensor::from_slice(
                &action_batch,
                &[batch_size, self.action_dim],
                &candle_device,
            )?,
            rewards: Tensor::from_slice(&reward_batch, &[batch_size], &candle_device)?,
        })
    }

    /// Get number of stored transitions.
    pub fn len(&self) -> usize {
        self.total_transitions
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.total_transitions == 0
    }

    /// Get number of demonstrations.
    pub fn num_demonstrations(&self) -> usize {
        self.demonstrations.len()
    }

    /// Get average episode return across demonstrations.
    pub fn average_return(&self) -> f32 {
        if self.demonstrations.is_empty() {
            0.0
        } else {
            self.demonstrations
                .iter()
                .map(|d| d.episode_return)
                .sum::<f32>()
                / self.demonstrations.len() as f32
        }
    }

    /// Filter demonstrations by quality threshold.
    pub fn filter_by_quality(&mut self, min_return: f32) {
        self.demonstrations
            .retain(|d| d.episode_return >= min_return);
        self.rebuild_flattened_data();
    }

    /// Save demonstrations to file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let candle_device = candle_core::Device::Cpu;

        let obs_tensor = Tensor::from_slice(
            &self.obs_data,
            &[self.total_transitions, self.obs_dim],
            &candle_device,
        )?;
        let action_tensor = Tensor::from_slice(
            &self.action_data,
            &[self.total_transitions, self.action_dim],
            &candle_device,
        )?;
        let reward_tensor =
            Tensor::from_slice(&self.reward_data, &[self.total_transitions], &candle_device)?;

        let mut tensors = std::collections::HashMap::new();
        tensors.insert("observations".to_string(), obs_tensor);
        tensors.insert("actions".to_string(), action_tensor);
        tensors.insert("rewards".to_string(), reward_tensor);

        candle_core::safetensors::save(&tensors, path)?;

        // Save metadata
        let metadata: Vec<DemoMetadata> = self
            .demonstrations
            .iter()
            .map(|d| d.metadata.clone())
            .collect();
        let metadata_path = path.with_extension("json");
        let metadata_json = serde_json::to_string_pretty(&metadata)?;
        std::fs::write(metadata_path, metadata_json)?;

        info!(
            "Saved {} demonstrations ({} transitions) to {:?}",
            self.demonstrations.len(),
            self.total_transitions,
            path
        );

        Ok(())
    }

    /// Load demonstrations from file.
    pub fn load(&mut self, path: &Path, device: &Device) -> Result<()> {
        let candle_device = device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        let obs_tensor = tensors
            .get("observations")
            .ok_or_else(|| OctaneError::Buffer("Missing observations".into()))?;
        let action_tensor = tensors
            .get("actions")
            .ok_or_else(|| OctaneError::Buffer("Missing actions".into()))?;
        let reward_tensor = tensors
            .get("rewards")
            .ok_or_else(|| OctaneError::Buffer("Missing rewards".into()))?;

        self.obs_data = obs_tensor.flatten_all()?.to_vec1()?;
        self.action_data = action_tensor.flatten_all()?.to_vec1()?;
        self.reward_data = reward_tensor.flatten_all()?.to_vec1()?;

        self.total_transitions = obs_tensor.dim(0)?;
        self.quality_weights = vec![1.0; self.total_transitions];

        // Create a single demonstration for the loaded data
        let mut demo = Demonstration::new(self.reward_data.iter().sum());
        for i in 0..self.total_transitions {
            let obs = self.obs_data[i * self.obs_dim..(i + 1) * self.obs_dim].to_vec();
            let action = self.action_data[i * self.action_dim..(i + 1) * self.action_dim].to_vec();
            let reward = self.reward_data[i];
            demo.add(obs, action, Some(reward));
        }
        self.demonstrations = vec![demo];

        info!(
            "Loaded {} transitions from {:?}",
            self.total_transitions, path
        );

        Ok(())
    }
}

/// Batch of demonstration data.
#[derive(Debug)]
pub struct DemoBatch {
    /// Observations [batch_size, obs_dim].
    pub observations: Tensor,
    /// Actions [batch_size, action_dim].
    pub actions: Tensor,
    /// Rewards [batch_size].
    pub rewards: Tensor,
}

/// Configuration for imitation learning agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImitationConfig {
    /// Imitation learning method.
    pub method: ImitationMethod,
    /// Loss function.
    pub loss_function: ImitationLoss,
    /// Learning rate.
    pub learning_rate: f32,
    /// Demonstration buffer size.
    pub demo_buffer_size: usize,
    /// Batch size for training.
    pub batch_size: usize,
    /// Number of epochs for pre-training.
    pub pretrain_epochs: usize,
    /// DAgger mixture ratio (probability of using expert vs learner).
    pub mixture_ratio: f32,
    /// DAgger iterations.
    pub dagger_iterations: usize,
    /// Policy hidden sizes.
    pub policy_hidden_sizes: Vec<usize>,
    /// Whether to use weighted sampling by quality.
    pub weighted_sampling: bool,
    /// Minimum demonstration quality to use.
    pub min_demo_quality: f32,
    /// Regularization coefficient.
    pub regularization: f32,
    /// Huber loss delta (if using Huber).
    pub huber_delta: f32,
    /// Label smoothing (for cross-entropy).
    pub label_smoothing: f32,
    /// Fine-tuning learning rate (after pre-training).
    pub finetune_lr: f32,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for ImitationConfig {
    fn default() -> Self {
        Self {
            method: ImitationMethod::BehavioralCloning,
            loss_function: ImitationLoss::MSE,
            learning_rate: 1e-3,
            demo_buffer_size: 100_000,
            batch_size: 256,
            pretrain_epochs: 100,
            mixture_ratio: 0.5,
            dagger_iterations: 10,
            policy_hidden_sizes: vec![256, 256],
            weighted_sampling: true,
            min_demo_quality: 0.0,
            regularization: 1e-5,
            huber_delta: 1.0,
            label_smoothing: 0.0,
            finetune_lr: 1e-4,
            seed: None,
        }
    }
}

impl ImitationConfig {
    /// Create new config with defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set imitation method.
    pub fn method(mut self, method: ImitationMethod) -> Self {
        self.method = method;
        self
    }

    /// Set loss function.
    pub fn loss_function(mut self, loss: ImitationLoss) -> Self {
        self.loss_function = loss;
        self
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set demonstration buffer size.
    pub fn demo_buffer_size(mut self, size: usize) -> Self {
        self.demo_buffer_size = size;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set number of pre-training epochs.
    pub fn pretrain_epochs(mut self, epochs: usize) -> Self {
        self.pretrain_epochs = epochs;
        self
    }

    /// Set DAgger mixture ratio.
    pub fn mixture_ratio(mut self, ratio: f32) -> Self {
        self.mixture_ratio = ratio;
        self
    }

    /// Set DAgger iterations.
    pub fn dagger_iterations(mut self, iters: usize) -> Self {
        self.dagger_iterations = iters;
        self
    }

    /// Enable/disable weighted sampling.
    pub fn weighted_sampling(mut self, enabled: bool) -> Self {
        self.weighted_sampling = enabled;
        self
    }

    /// Set minimum demonstration quality.
    pub fn min_demo_quality(mut self, quality: f32) -> Self {
        self.min_demo_quality = quality;
        self
    }

    /// Set seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".into());
        }
        if self.demo_buffer_size == 0 {
            return Err("demo_buffer_size must be positive".into());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.mixture_ratio) {
            return Err("mixture_ratio must be in [0, 1]".into());
        }
        Ok(())
    }
}

/// Expert policy trait for DAgger.
pub trait ExpertPolicy: Send + Sync {
    /// Get expert action for observation.
    fn get_action(&self, obs: &Tensor) -> Result<Tensor>;

    /// Get expert action probability/confidence.
    fn get_confidence(&self, obs: &Tensor) -> Result<f32>;
}

/// Simple rule-based expert for demonstration.
pub struct RuleBasedExpert {
    /// Strategy name.
    name: String,
    /// Action rules based on observation features.
    rules: Vec<(Box<dyn Fn(&[f32]) -> bool + Send + Sync>, Vec<f32>)>,
    /// Default action.
    default_action: Vec<f32>,
    /// Device.
    device: Device,
}

impl RuleBasedExpert {
    /// Create a momentum-following expert.
    pub fn momentum_following(_obs_dim: usize, action_dim: usize, device: Device) -> Self {
        let default_action = vec![0.0; action_dim];

        let rules: Vec<(Box<dyn Fn(&[f32]) -> bool + Send + Sync>, Vec<f32>)> = vec![
            // If RSI > 70 (overbought), go short
            (
                Box::new(|obs: &[f32]| obs.get(6).is_some_and(|&rsi| rsi > 70.0)),
                vec![-1.0],
            ),
            // If RSI < 30 (oversold), go long
            (
                Box::new(|obs: &[f32]| obs.get(6).is_some_and(|&rsi| rsi < 30.0)),
                vec![1.0],
            ),
            // If SMA ratio > 1.02, trend following long
            (
                Box::new(|obs: &[f32]| obs.get(5).is_some_and(|&sma| sma > 1.02)),
                vec![0.5],
            ),
            // If SMA ratio < 0.98, trend following short
            (
                Box::new(|obs: &[f32]| obs.get(5).is_some_and(|&sma| sma < 0.98)),
                vec![-0.5],
            ),
        ];

        Self {
            name: "MomentumFollowing".to_string(),
            rules,
            default_action,
            device,
        }
    }

    /// Create a mean-reversion expert.
    pub fn mean_reversion(_obs_dim: usize, action_dim: usize, device: Device) -> Self {
        let default_action = vec![0.0; action_dim];

        let rules: Vec<(Box<dyn Fn(&[f32]) -> bool + Send + Sync>, Vec<f32>)> = vec![
            // If RSI > 80, fade the move (short)
            (
                Box::new(|obs: &[f32]| obs.get(6).is_some_and(|&rsi| rsi > 80.0)),
                vec![-0.8],
            ),
            // If RSI < 20, fade the move (long)
            (
                Box::new(|obs: &[f32]| obs.get(6).is_some_and(|&rsi| rsi < 20.0)),
                vec![0.8],
            ),
            // If volatility high and price extended, mean revert
            (
                Box::new(|obs: &[f32]| {
                    let vol = obs.get(7).unwrap_or(&0.0);
                    let sma = obs.get(5).unwrap_or(&1.0);
                    *vol > 0.03 && (sma - 1.0).abs() > 0.02
                }),
                vec![0.0], // Exit
            ),
        ];

        Self {
            name: "MeanReversion".to_string(),
            rules,
            default_action,
            device,
        }
    }
}

impl ExpertPolicy for RuleBasedExpert {
    fn get_action(&self, obs: &Tensor) -> Result<Tensor> {
        let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;
        let batch_size = obs.dim(0)?;
        let obs_dim = obs_vec.len() / batch_size;

        let candle_device = self.device.to_candle()?;
        let mut actions = Vec::with_capacity(batch_size * self.default_action.len());

        for b in 0..batch_size {
            let start = b * obs_dim;
            let end = start + obs_dim;
            let obs_slice = &obs_vec[start..end];

            let mut action = self.default_action.clone();
            for (condition, rule_action) in &self.rules {
                if condition(obs_slice) {
                    action = rule_action.clone();
                    break;
                }
            }
            actions.extend(action);
        }

        let action_dim = self.default_action.len();
        Ok(Tensor::from_slice(
            &actions,
            &[batch_size, action_dim],
            &candle_device,
        )?)
    }

    fn get_confidence(&self, _obs: &Tensor) -> Result<f32> {
        // Rule-based expert has fixed confidence
        Ok(0.8)
    }
}

/// Imitation learning agent.
pub struct ImitationAgent<E: Environment + Clone + 'static> {
    /// Configuration.
    config: ImitationConfig,
    /// Vectorized environment.
    env: VecEnv<E>,
    /// Device.
    device: Device,

    /// Policy network.
    policy_var_map: VarMap,

    /// Persistent optimizer (recreating it each step would reset Adam's
    /// moment estimates, degrading every supervised update to ~sign-SGD).
    policy_optimizer: AdamW,

    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    act_dim: usize,
    /// Whether actions are discrete.
    is_discrete: bool,

    /// Demonstration buffer.
    demo_buffer: DemoReplayBuffer,

    /// Expert policy (for DAgger).
    expert: Option<Box<dyn ExpertPolicy>>,

    /// Training phase.
    phase: TrainingPhase,

    /// Total timesteps.
    total_timesteps: usize,

    /// Pre-training epochs completed.
    pretrain_epochs_completed: usize,

    /// DAgger iterations completed.
    dagger_iterations_completed: usize,

    /// RNG.
    rng: StdRng,
}

/// Training phase for imitation agent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrainingPhase {
    /// Pre-training from demonstrations.
    PreTraining,
    /// DAgger aggregation phase.
    DAgger,
    /// Fine-tuning with RL.
    FineTuning,
}

impl<E: Environment + Clone + 'static> ImitationAgent<E> {
    /// Create a new imitation learning agent.
    pub fn new(config: ImitationConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();
        let obs_dim = obs_space.flat_dim();
        let act_dim = act_space.flat_dim();
        let is_discrete = act_space.shape() == [1];

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let policy_var_map = VarMap::new();
        let demo_buffer = DemoReplayBuffer::new(obs_dim, act_dim, config.demo_buffer_size);

        let mut agent = Self {
            config,
            env,
            device,
            policy_var_map,
            // Placeholder; rebound to the real policy vars after init_networks().
            policy_optimizer: AdamW::new(Vec::new(), ParamsAdamW::default())?,
            obs_dim,
            act_dim,
            is_discrete,
            demo_buffer,
            expert: None,
            phase: TrainingPhase::PreTraining,
            total_timesteps: 0,
            pretrain_epochs_completed: 0,
            dagger_iterations_completed: 0,
            rng,
        };

        agent.init_networks()?;
        agent.policy_optimizer = AdamW::new(
            agent.policy_var_map.all_vars(),
            ParamsAdamW {
                lr: agent.config.learning_rate as f64,
                ..Default::default()
            },
        )?;

        info!(
            "ImitationAgent initialized: method={:?}, obs_dim={}, act_dim={}",
            agent.config.method, obs_dim, act_dim
        );

        Ok(agent)
    }

    /// Initialize neural networks.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.policy_var_map, DType::F32, &candle_device);

        let mut in_dim = self.obs_dim;
        for (i, &hidden_size) in self.config.policy_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("policy.layer_{}", i)))?;
            in_dim = hidden_size;
        }
        let _ = candle_nn::linear(in_dim, self.act_dim, vb.pp("policy.output"))?;

        Ok(())
    }

    /// Forward pass through policy.
    fn policy_forward(&self, obs: &Tensor) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.policy_var_map, DType::F32, &candle_device);

        let mut x = obs.clone();
        for (i, &hidden_size) in self.config.policy_hidden_sizes.iter().enumerate() {
            let in_dim = if i == 0 {
                self.obs_dim
            } else {
                self.config.policy_hidden_sizes[i - 1]
            };
            let linear =
                candle_nn::linear(in_dim, hidden_size, vb.pp(format!("policy.layer_{}", i)))?;
            x = linear.forward(&x)?;
            x = x.relu()?;
        }

        let output_linear = candle_nn::linear(
            *self.config.policy_hidden_sizes.last().unwrap(),
            self.act_dim,
            vb.pp("policy.output"),
        )?;

        let output = output_linear.forward(&x)?;

        if self.is_discrete {
            Ok(output)
        } else {
            Ok(output.tanh()?)
        }
    }

    /// Compute imitation loss.
    fn compute_loss(&self, predicted: &Tensor, target: &Tensor) -> Result<Tensor> {
        match self.config.loss_function {
            ImitationLoss::MSE => {
                let diff = (predicted - target)?;
                Ok(diff.sqr()?.mean_all()?)
            }
            ImitationLoss::CrossEntropy => {
                // For discrete actions
                let log_probs = candle_nn::ops::log_softmax(predicted, 1)?;
                let target_indices = target.to_dtype(DType::I64)?;
                let nll = log_probs.gather(&target_indices, 1)?.neg()?.mean_all()?;
                Ok(nll)
            }
            ImitationLoss::Huber => {
                let diff = (predicted - target)?.abs()?;
                let delta = self.config.huber_delta;
                let candle_device = self.device.to_candle()?;

                // Huber loss: 0.5 * x^2 if |x| <= delta, else delta * (|x| - 0.5 * delta)
                let quadratic = (diff.clamp(0.0, delta)?.sqr()? * 0.5)?;
                let half_delta = Tensor::new(&[delta / 2.0], &candle_device)?;
                let delta_tensor = Tensor::new(&[delta], &candle_device)?;
                let diff_minus_half = diff.broadcast_sub(&half_delta)?;
                let clamped = diff_minus_half.clamp(0.0, f32::INFINITY)?;
                let linear = clamped.broadcast_mul(&delta_tensor)?;
                Ok((&quadratic + &linear)?.mean_all()?)
            }
            ImitationLoss::L1 => Ok((predicted - target)?.abs()?.mean_all()?),
            ImitationLoss::NLL => {
                // Assume Gaussian output
                let diff = (predicted - target)?;
                let candle_device = self.device.to_candle()?;
                let const_term =
                    Tensor::new(&[0.5 * (2.0 * std::f32::consts::PI).ln()], &candle_device)?;
                Ok(((diff.sqr()? * 0.5)?.broadcast_add(&const_term))?.mean_all()?)
            }
        }
    }

    /// Add demonstrations to buffer.
    pub fn add_demonstration(&mut self, demo: Demonstration) {
        if demo.episode_return >= self.config.min_demo_quality {
            self.demo_buffer.add_demonstration(demo);
        }
    }

    /// Load demonstrations from file.
    pub fn load_demonstrations(&mut self, path: &Path) -> Result<()> {
        self.demo_buffer.load(path, &self.device)?;
        self.demo_buffer
            .filter_by_quality(self.config.min_demo_quality);
        Ok(())
    }

    /// Set expert policy for DAgger.
    pub fn set_expert(&mut self, expert: Box<dyn ExpertPolicy>) {
        self.expert = Some(expert);
    }

    /// Pre-training step (behavioral cloning).
    fn pretrain_step(&mut self) -> Result<f32> {
        if self.demo_buffer.is_empty() {
            return Ok(0.0);
        }

        let batch = if self.config.weighted_sampling {
            self.demo_buffer
                .sample_weighted(self.config.batch_size, &self.device)?
        } else {
            self.demo_buffer
                .sample_uniform(self.config.batch_size, &self.device)?
        };

        // Forward pass
        let predicted = self.policy_forward(&batch.observations)?;

        // Compute loss
        let loss = self.compute_loss(&predicted, &batch.actions)?;

        // Add regularization
        let candle_device = self.device.to_candle()?;
        let mut reg_loss = Tensor::zeros(&[], DType::F32, &candle_device)?;
        if self.config.regularization > 0.0 {
            let data = self.policy_var_map.data().lock().unwrap();
            for (_, var) in data.iter() {
                let tensor = var.as_tensor();
                reg_loss = (&reg_loss + tensor.sqr()?.sum_all()?)?;
            }
            let reg_coef = Tensor::new(&[self.config.regularization], &candle_device)?;
            reg_loss = reg_loss.broadcast_mul(&reg_coef)?;
        }

        let total_loss = (&loss + &reg_loss)?;
        let loss_val: f32 = total_loss.to_scalar()?;

        // Reuse the persistent optimizer so Adam momentum accumulates.
        self.policy_optimizer.backward_step(&total_loss)?;

        Ok(loss_val)
    }

    /// DAgger step.
    fn dagger_step(&mut self) -> Result<f32> {
        let expert = match &self.expert {
            Some(e) => e,
            None => {
                return Err(OctaneError::InvalidConfig(
                    "No expert set for DAgger".into(),
                ))
            }
        };

        let num_envs = self.env.num_envs();
        let mut obs = self.env.reset(&self.device)?;

        let mut new_demo = Demonstration::new(0.0);
        let mut total_reward = 0.0f32;

        // Collect trajectory mixing expert and learner
        for _ in 0..100 {
            // Get both expert and learner actions
            let expert_action = expert.get_action(&obs)?;
            let learner_action = self.policy_forward(&obs)?;

            // Mix actions based on ratio
            let use_expert: bool = self.rng.gen::<f32>() < self.config.mixture_ratio;
            let action = if use_expert {
                expert_action.clone()
            } else {
                learner_action
            };

            // Step environment
            let step_result = self.env.step(&action, &self.device)?;

            // Store with expert label (always)
            let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;
            let expert_action_vec: Vec<f32> = expert_action.flatten_all()?.to_vec1()?;
            let rewards_vec: Vec<f32> = step_result.rewards.to_vec1()?;

            for env_idx in 0..num_envs {
                new_demo.add(
                    obs_vec[env_idx * self.obs_dim..(env_idx + 1) * self.obs_dim].to_vec(),
                    expert_action_vec[env_idx * self.act_dim..(env_idx + 1) * self.act_dim]
                        .to_vec(),
                    Some(rewards_vec[env_idx]),
                );
                total_reward += rewards_vec[env_idx];
            }

            obs = step_result.observations;
            self.total_timesteps += num_envs;
        }

        // Update demonstration return
        new_demo.episode_return = total_reward;

        // Add to buffer
        self.demo_buffer.add_demonstration(new_demo);

        // Train on aggregated dataset
        let loss = self.pretrain_step()?;

        Ok(loss)
    }

    /// Predict action.
    pub fn predict(&mut self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        let action = self.policy_forward(obs)?;

        if deterministic || self.is_discrete {
            if self.is_discrete {
                Ok(action.argmax(1)?.to_dtype(DType::F32)?)
            } else {
                Ok(action)
            }
        } else {
            // Add exploration noise during training
            let noise_scale = 0.1
                * (1.0
                    - self.pretrain_epochs_completed as f32 / self.config.pretrain_epochs as f32)
                    .max(0.01);
            let noise = Tensor::randn_like(&action, 0.0, noise_scale as f64)?;
            Ok((action + noise)?.tanh()?)
        }
    }

    /// Get current training phase.
    pub fn phase(&self) -> TrainingPhase {
        self.phase
    }

    /// Get demonstration buffer statistics.
    pub fn demo_stats(&self) -> (usize, usize, f32) {
        (
            self.demo_buffer.num_demonstrations(),
            self.demo_buffer.len(),
            self.demo_buffer.average_return(),
        )
    }

    /// Pre-train from demonstrations for specified epochs.
    pub fn pretrain(&mut self) -> Result<f32> {
        self.phase = TrainingPhase::PreTraining;
        let mut total_loss = 0.0f32;

        for epoch in 0..self.config.pretrain_epochs {
            let num_batches = (self.demo_buffer.len() / self.config.batch_size).max(1);
            let mut epoch_loss = 0.0f32;

            for _ in 0..num_batches {
                let loss = self.pretrain_step()?;
                epoch_loss += loss;
            }

            epoch_loss /= num_batches as f32;
            total_loss += epoch_loss;
            self.pretrain_epochs_completed += 1;

            if (epoch + 1) % 10 == 0 {
                info!(
                    "Pre-train epoch {}/{}: loss={:.4}",
                    epoch + 1,
                    self.config.pretrain_epochs,
                    epoch_loss
                );
            }
        }

        Ok(total_loss / self.config.pretrain_epochs as f32)
    }

    /// Run DAgger iterations.
    pub fn run_dagger(&mut self) -> Result<f32> {
        self.phase = TrainingPhase::DAgger;
        let mut total_loss = 0.0f32;

        for iter in 0..self.config.dagger_iterations {
            let loss = self.dagger_step()?;
            total_loss += loss;
            self.dagger_iterations_completed += 1;

            info!(
                "DAgger iteration {}/{}: loss={:.4}, demos={}",
                iter + 1,
                self.config.dagger_iterations,
                loss,
                self.demo_buffer.num_demonstrations()
            );

            // Decay mixture ratio
            self.config.mixture_ratio *= 0.9;
            self.config.mixture_ratio = self.config.mixture_ratio.max(0.1);
        }

        Ok(total_loss / self.config.dagger_iterations as f32)
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for ImitationAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let loss = match self.config.method {
            ImitationMethod::BehavioralCloning => self.pretrain_step()?,
            ImitationMethod::DAgger => {
                if self.pretrain_epochs_completed < self.config.pretrain_epochs {
                    self.pretrain_step()?
                } else {
                    self.dagger_step()?
                }
            }
            ImitationMethod::GAIL | ImitationMethod::IRL => {
                // Simplified: fall back to behavioral cloning
                self.pretrain_step()?
            }
        };

        let (num_demos, _num_transitions, avg_return) = self.demo_stats();

        debug!(
            "Imitation step: phase={:?}, loss={:.4}, demos={}, avg_return={:.2}",
            self.phase, loss, num_demos, avg_return
        );

        Ok(TrainMetrics {
            policy_loss: loss,
            mean_reward: avg_return,
            timesteps: self.total_timesteps,
            episodes: num_demos,
            ..Default::default()
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        for (name, var) in self.policy_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&tensors, path)?;

        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        // Save demonstrations separately
        let demo_path = path.with_file_name(format!(
            "{}_demos.safetensors",
            path.file_stem().unwrap_or_default().to_string_lossy()
        ));
        self.demo_buffer.save(&demo_path)?;

        info!("ImitationAgent saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        let mut data = self.policy_var_map.data().lock().unwrap();
        for (name, tensor) in tensors {
            if let Some(var) = data.get_mut(&name) {
                var.set(&tensor)?;
            }
        }

        // Try to load demonstrations
        let demo_path = path.with_file_name(format!(
            "{}_demos.safetensors",
            path.file_stem().unwrap_or_default().to_string_lossy()
        ));
        if demo_path.exists() {
            drop(data);
            self.demo_buffer.load(&demo_path, &self.device)?;
        }

        info!("ImitationAgent loaded from {:?}", path);
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
    fn test_imitation_config_defaults() {
        let config = ImitationConfig::default();
        assert_eq!(config.method, ImitationMethod::BehavioralCloning);
        assert_eq!(config.loss_function, ImitationLoss::MSE);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_demonstration_creation() {
        let mut demo = Demonstration::new(100.0);
        demo.add(vec![1.0, 2.0], vec![0.5], Some(1.0));
        demo.add(vec![2.0, 3.0], vec![-0.5], Some(2.0));

        assert_eq!(demo.len(), 2);
        assert!((demo.quality() - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_demo_buffer_creation() {
        let buffer = DemoReplayBuffer::new(4, 2, 1000);
        assert!(buffer.is_empty());
        assert_eq!(buffer.len(), 0);
    }

    #[test]
    fn test_demo_buffer_add() {
        let mut buffer = DemoReplayBuffer::new(4, 2, 1000);

        let mut demo = Demonstration::new(50.0);
        demo.add(vec![1.0, 2.0, 3.0, 4.0], vec![0.5, -0.5], Some(1.0));
        demo.add(vec![2.0, 3.0, 4.0, 5.0], vec![-0.5, 0.5], Some(2.0));

        buffer.add_demonstration(demo);

        assert_eq!(buffer.len(), 2);
        assert_eq!(buffer.num_demonstrations(), 1);
    }
}
