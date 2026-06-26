//! Phasic Policy Gradient (PPG) algorithm implementation.
//!
//! PPG is an extension of PPO that decouples policy and value function training
//! into separate phases to improve sample efficiency. It introduces an auxiliary
//! phase that trains the value function on stored rollouts while using a
//! behavioral cloning loss to prevent policy drift.
//!
//! Key features:
//! - Separate policy and value training phases
//! - Auxiliary phase for value function refinement
//! - Behavioral cloning loss to prevent policy drift during auxiliary updates
//! - Stores multiple rollouts for auxiliary training
//!
//! Reference: Cobbe et al., "Phasic Policy Gradient" (2021)
//! https://arxiv.org/abs/2009.04416

use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::rollout::RolloutBuffer;
use crate::algorithms::traits::RLAlgorithm;
use crate::core::{Device, OctaneError, Result};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::{debug, info};

/// Configuration for Phasic Policy Gradient (PPG) algorithm.
///
/// PPG extends PPO with separate policy and value training phases.
/// The auxiliary phase trains the value function on stored rollouts
/// while preventing policy drift using behavioral cloning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPGConfig {
    /// Learning rate for optimizer.
    /// Default: 5e-4
    pub learning_rate: f32,

    /// Number of steps to collect before each policy update.
    /// Default: 256
    pub n_steps: usize,

    /// Minibatch size for gradient updates.
    /// Default: 8
    pub batch_size: usize,

    /// Number of policy training epochs per rollout.
    /// Default: 1
    pub policy_epochs: usize,

    /// Number of value training epochs per rollout (during policy phase).
    /// Default: 1
    pub value_epochs: usize,

    /// Number of auxiliary training epochs.
    /// Default: 6
    pub aux_epochs: usize,

    /// Behavioral cloning coefficient to prevent policy drift during aux phase.
    /// Higher values = more conservative policy updates during aux phase.
    /// Default: 1.0
    pub beta_clone: f32,

    /// Number of rollouts to store for auxiliary training.
    /// Auxiliary phase is triggered after this many rollouts.
    /// Default: 32
    pub num_aux_rollouts: usize,

    /// Discount factor for future rewards.
    /// Default: 0.99
    pub gamma: f32,

    /// GAE lambda parameter for advantage estimation.
    /// Default: 0.95
    pub gae_lambda: f32,

    /// Clipping range for PPO surrogate objective.
    /// Default: 0.2
    pub clip_range: f32,

    /// Value function loss coefficient.
    /// Default: 0.5
    pub vf_coef: f32,

    /// Entropy bonus coefficient for exploration.
    /// Default: 0.01
    pub ent_coef: f32,

    /// Maximum gradient norm for clipping.
    /// Default: 0.5
    pub max_grad_norm: f32,

    /// Whether to normalize advantages.
    /// Default: true
    pub normalize_advantage: bool,

    /// Target KL divergence for early stopping (optional).
    /// Default: None (no early stopping)
    pub target_kl: Option<f32>,

    /// Hidden layer sizes for policy and value networks.
    /// Default: [64, 64]
    pub hidden_sizes: Vec<usize>,

    /// Random seed for reproducibility.
    /// Default: None (use system entropy)
    pub seed: Option<u64>,
}

impl Default for PPGConfig {
    fn default() -> Self {
        Self {
            learning_rate: 5e-4,
            n_steps: 256,
            batch_size: 8,
            policy_epochs: 1,
            value_epochs: 1,
            aux_epochs: 6,
            beta_clone: 1.0,
            num_aux_rollouts: 32,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_range: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.01,
            max_grad_norm: 0.5,
            normalize_advantage: true,
            target_kl: None,
            hidden_sizes: vec![64, 64],
            seed: None,
        }
    }
}

impl PPGConfig {
    /// Create a new PPG config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style setter for learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder-style setter for n_steps.
    pub fn n_steps(mut self, n: usize) -> Self {
        self.n_steps = n;
        self
    }

    /// Builder-style setter for batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Builder-style setter for policy epochs.
    pub fn policy_epochs(mut self, n: usize) -> Self {
        self.policy_epochs = n;
        self
    }

    /// Builder-style setter for value epochs.
    pub fn value_epochs(mut self, n: usize) -> Self {
        self.value_epochs = n;
        self
    }

    /// Builder-style setter for auxiliary epochs.
    pub fn aux_epochs(mut self, n: usize) -> Self {
        self.aux_epochs = n;
        self
    }

    /// Builder-style setter for behavioral cloning coefficient.
    pub fn beta_clone(mut self, beta: f32) -> Self {
        self.beta_clone = beta;
        self
    }

    /// Builder-style setter for number of auxiliary rollouts.
    pub fn num_aux_rollouts(mut self, n: usize) -> Self {
        self.num_aux_rollouts = n;
        self
    }

    /// Builder-style setter for gamma.
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Builder-style setter for GAE lambda.
    pub fn gae_lambda(mut self, l: f32) -> Self {
        self.gae_lambda = l;
        self
    }

    /// Builder-style setter for clip range.
    pub fn clip_range(mut self, c: f32) -> Self {
        self.clip_range = c;
        self
    }

    /// Builder-style setter for value function coefficient.
    pub fn vf_coef(mut self, c: f32) -> Self {
        self.vf_coef = c;
        self
    }

    /// Builder-style setter for entropy coefficient.
    pub fn ent_coef(mut self, c: f32) -> Self {
        self.ent_coef = c;
        self
    }

    /// Builder-style setter for hidden layer sizes.
    pub fn hidden_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.hidden_sizes = sizes;
        self
    }

    /// Builder-style setter for target KL divergence.
    pub fn target_kl(mut self, kl: Option<f32>) -> Self {
        self.target_kl = kl;
        self
    }

    /// Builder-style setter for seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".to_string());
        }
        if self.n_steps == 0 {
            return Err("n_steps must be positive".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".to_string());
        }
        if self.policy_epochs == 0 {
            return Err("policy_epochs must be positive".to_string());
        }
        if self.aux_epochs == 0 {
            return Err("aux_epochs must be positive".to_string());
        }
        if self.num_aux_rollouts == 0 {
            return Err("num_aux_rollouts must be positive".to_string());
        }
        if self.beta_clone < 0.0 {
            return Err("beta_clone must be non-negative".to_string());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.gae_lambda) {
            return Err("gae_lambda must be in [0, 1]".to_string());
        }
        if self.clip_range <= 0.0 {
            return Err("clip_range must be positive".to_string());
        }
        if self.hidden_sizes.is_empty() {
            return Err("hidden_sizes cannot be empty".to_string());
        }
        Ok(())
    }
}

/// Stored rollout data for auxiliary training.
#[derive(Debug)]
struct AuxRollout {
    /// Observations from this rollout.
    observations: Tensor,
    /// Returns (discounted cumulative rewards).
    returns: Tensor,
    /// Old policy logits/mean for behavioral cloning.
    old_policy_output: Tensor,
}

/// PPG Agent for training and inference.
///
/// Implements the Phasic Policy Gradient algorithm which separates
/// policy and value function training into distinct phases.
pub struct PPGAgent<E: Environment + Clone + 'static> {
    /// Algorithm configuration.
    config: PPGConfig,

    /// Vectorized environment.
    env: VecEnv<E>,

    /// Device for tensor operations.
    device: Device,

    /// Variable map for policy network parameters.
    policy_var_map: VarMap,

    /// Variable map for value network parameters.
    value_var_map: VarMap,

    /// Observation dimension.
    obs_dim: usize,

    /// Action dimension (for continuous) or number of actions (for discrete).
    act_dim: usize,

    /// Whether the action space is discrete.
    is_discrete: bool,

    /// Persistent optimizers (Adam moment state must survive across updates
    /// and across the policy/value and auxiliary phases).
    policy_optimizer: AdamW,
    value_optimizer: AdamW,

    /// Stored rollouts for auxiliary training.
    aux_rollouts: Vec<AuxRollout>,

    /// Number of rollouts collected since last auxiliary phase.
    rollouts_since_aux: usize,

    /// Total timesteps trained so far.
    total_timesteps: usize,

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> PPGAgent<E> {
    /// Create a new PPG agent.
    ///
    /// # Arguments
    ///
    /// * `config` - PPG configuration
    /// * `env` - Vectorized environment
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// A new PPG agent ready for training.
    pub fn new(config: PPGConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();

        let obs_dim = obs_space.flat_dim();
        let act_dim = act_space.flat_dim();
        // A continuous 1-D action space also has shape [1], so detect by the
        // space's own continuity flag (a `DiscreteSpace` is non-continuous).
        let is_discrete = !act_space.is_continuous();

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        let policy_var_map = VarMap::new();
        let value_var_map = VarMap::new();

        let opt_params = ParamsAdamW {
            lr: config.learning_rate as f64,
            weight_decay: 0.0,
            ..Default::default()
        };

        let mut agent = Self {
            config: config.clone(),
            env,
            device,
            policy_var_map,
            value_var_map,
            obs_dim,
            act_dim,
            is_discrete,
            policy_optimizer: AdamW::new(Vec::new(), opt_params.clone())?,
            value_optimizer: AdamW::new(Vec::new(), opt_params.clone())?,
            aux_rollouts: Vec::with_capacity(config.num_aux_rollouts),
            rollouts_since_aux: 0,
            total_timesteps: 0,
            rng,
        };

        agent.init_networks()?;

        // Bind optimizers to the populated variables (incl. the now-trainable
        // continuous log_std registered in the policy var map).
        agent.policy_optimizer = AdamW::new(agent.policy_var_map.all_vars(), opt_params.clone())?;
        agent.value_optimizer = AdamW::new(agent.value_var_map.all_vars(), opt_params)?;

        info!(
            "PPG Agent initialized: obs_dim={}, act_dim={}, discrete={}, aux_rollouts={}",
            obs_dim, act_dim, is_discrete, config.num_aux_rollouts
        );

        Ok(agent)
    }

    /// Initialize neural network weights for policy and value networks.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // Build policy network layers
        let vb_policy = VarBuilder::from_varmap(&self.policy_var_map, DType::F32, &candle_device);
        let mut in_dim = self.obs_dim;

        for (i, &hidden_size) in self.config.hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(
                in_dim,
                hidden_size,
                vb_policy.pp(format!("policy.layer_{}", i)),
            )?;
            in_dim = hidden_size;
        }

        // Output layer for policy (action logits or mean)
        let _ = candle_nn::linear(in_dim, self.act_dim, vb_policy.pp("policy.output"))?;

        // Build value network layers (separate from policy)
        let vb_value = VarBuilder::from_varmap(&self.value_var_map, DType::F32, &candle_device);
        in_dim = self.obs_dim;

        for (i, &hidden_size) in self.config.hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(
                in_dim,
                hidden_size,
                vb_value.pp(format!("value.layer_{}", i)),
            )?;
            in_dim = hidden_size;
        }

        // Output layer for value (single scalar)
        let _ = candle_nn::linear(in_dim, 1, vb_value.pp("value.output"))?;

        // Register the trainable log_std parameter for continuous actions in the
        // policy var map so the optimizer updates it and checkpoints persist it
        // (previously a fixed zeros tensor => std=1 forever, no gradient).
        if !self.is_discrete {
            let _ = self.log_std_tensor()?;
        }

        Ok(())
    }

    /// Fetch the trainable continuous-action log_std parameter from the policy
    /// var map. Re-fetched on each use so it reflects the latest optimizer step.
    fn log_std_tensor(&self) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.policy_var_map, DType::F32, &candle_device);
        let log_std =
            vb.get_with_hints(self.act_dim, "policy.log_std", candle_nn::Init::Const(0.0))?;
        Ok(log_std)
    }

    /// Forward pass through policy network.
    fn policy_forward(&self, obs: &Tensor) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.policy_var_map, DType::F32, &candle_device);

        let mut x = obs.clone();
        let num_layers = self.config.hidden_sizes.len();

        for i in 0..num_layers {
            let in_dim = if i == 0 {
                self.obs_dim
            } else {
                self.config.hidden_sizes[i - 1]
            };
            let linear = candle_nn::linear(
                in_dim,
                self.config.hidden_sizes[i],
                vb.pp(format!("policy.layer_{}", i)),
            )?;
            x = linear.forward(&x)?;
            x = x.tanh()?;
        }

        let output_linear = candle_nn::linear(
            self.config.hidden_sizes[num_layers - 1],
            self.act_dim,
            vb.pp("policy.output"),
        )?;

        Ok(output_linear.forward(&x)?)
    }

    /// Forward pass through value network.
    fn value_forward(&self, obs: &Tensor) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.value_var_map, DType::F32, &candle_device);

        let mut x = obs.clone();
        let num_layers = self.config.hidden_sizes.len();

        for i in 0..num_layers {
            let in_dim = if i == 0 {
                self.obs_dim
            } else {
                self.config.hidden_sizes[i - 1]
            };
            let linear = candle_nn::linear(
                in_dim,
                self.config.hidden_sizes[i],
                vb.pp(format!("value.layer_{}", i)),
            )?;
            x = linear.forward(&x)?;
            x = x.tanh()?;
        }

        let output_linear = candle_nn::linear(
            self.config.hidden_sizes[num_layers - 1],
            1,
            vb.pp("value.output"),
        )?;

        Ok(output_linear.forward(&x)?)
    }

    /// Sample actions from the policy distribution.
    fn sample_action(&mut self, obs: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits_or_mean = self.policy_forward(obs)?;

        if self.is_discrete {
            // Categorical distribution
            let probs = candle_nn::ops::softmax(&logits_or_mean, 1)?;
            let log_probs = candle_nn::ops::log_softmax(&logits_or_mean, 1)?;

            // Materialize probs and log_probs to host ONCE (previously the
            // per-action log-prob was fetched per environment inside the loop,
            // one GPU->CPU sync per env per step).
            let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;
            let log_probs_vec: Vec<f32> = log_probs.flatten_all()?.to_vec1()?;
            let batch_size = obs.dim(0)?;
            let num_actions = self.act_dim;

            let mut actions = Vec::with_capacity(batch_size);
            let mut action_log_probs = Vec::with_capacity(batch_size);

            for b in 0..batch_size {
                let start = b * num_actions;
                let end = start + num_actions;
                let action_probs = &probs_vec[start..end];

                // Sample action
                let r: f32 = self.rng.gen();
                let mut cumsum = 0.0;
                let mut action = 0usize;
                for (i, &p) in action_probs.iter().enumerate() {
                    cumsum += p;
                    if r < cumsum {
                        action = i;
                        break;
                    }
                }

                actions.push(action as f32);

                // Index the already-materialized log_probs (no per-env sync).
                action_log_probs.push(log_probs_vec[start + action]);
            }

            let candle_device = self.device.to_candle()?;
            let actions_tensor = Tensor::from_slice(&actions, &[batch_size, 1], &candle_device)?;
            let log_probs_tensor =
                Tensor::from_slice(&action_log_probs, &[batch_size], &candle_device)?;

            Ok((actions_tensor, log_probs_tensor))
        } else {
            // Gaussian distribution for continuous actions
            let mean = logits_or_mean;
            let log_std = self.log_std_tensor()?;
            let std = log_std.exp()?;

            // Sample from Gaussian: action = mean + std * noise
            let noise = Tensor::randn_like(&mean, 0.0, 1.0)?;
            let actions = (&mean + noise.broadcast_mul(&std)?)?;

            // Compute log probability
            let diff = (&actions - &mean)?;
            let normalized = diff.broadcast_div(&std)?;
            let candle_device = self.device.to_candle()?;
            let log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
            let log_2pi_tensor = Tensor::new(&[log_2pi], &candle_device)?;
            let log_prob_per_dim = (normalized.sqr()? * (-0.5))?
                .broadcast_sub(&log_std)?
                .broadcast_sub(&log_2pi_tensor)?;

            // Sum over action dimensions
            let log_probs = log_prob_per_dim.sum(1)?;

            Ok((actions, log_probs))
        }
    }

    /// Compute log probability of actions under current policy.
    fn evaluate_actions(&self, obs: &Tensor, actions: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let logits_or_mean = self.policy_forward(obs)?;
        let values = self.value_forward(obs)?.squeeze(1)?;

        if self.is_discrete {
            let log_probs = candle_nn::ops::log_softmax(&logits_or_mean, 1)?;
            let probs = candle_nn::ops::softmax(&logits_or_mean, 1)?;

            // Extract log probs for taken actions
            let actions_i64 = actions.squeeze(1)?.to_dtype(DType::I64)?;
            let action_log_probs = log_probs
                .gather(&actions_i64.unsqueeze(1)?, 1)?
                .squeeze(1)?;

            // Entropy: -sum(p * log(p))
            let entropy = (probs.neg()? * &log_probs)?.sum(1)?;

            Ok((action_log_probs, values, entropy))
        } else {
            let mean = logits_or_mean;
            let log_std = self.log_std_tensor()?;
            let std = log_std.exp()?;

            // Log probability for Gaussian
            let candle_device = self.device.to_candle()?;
            let log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
            let log_2pi_tensor = Tensor::new(&[log_2pi], &candle_device)?;
            let diff = (actions - &mean)?;
            let normalized = diff.broadcast_div(&std)?;
            let log_prob_per_dim = (normalized.sqr()? * (-0.5))?
                .broadcast_sub(&log_std)?
                .broadcast_sub(&log_2pi_tensor)?;
            let log_probs = log_prob_per_dim.sum(1)?;

            // Entropy for Gaussian
            let entropy_const = 0.5 * (1.0 + (2.0 * std::f32::consts::PI).ln());
            let entropy_const_tensor = Tensor::new(&[entropy_const], &candle_device)?;
            let entropy_per_dim = log_std.broadcast_add(&entropy_const_tensor)?;
            let entropy = entropy_per_dim.sum(0)?.broadcast_as(&[obs.dim(0)?])?;

            Ok((log_probs, values, entropy))
        }
    }

    /// Collect rollout data from environment.
    fn collect_rollout(&mut self) -> Result<(RolloutBuffer, Vec<f32>, Vec<usize>)> {
        let num_envs = self.env.num_envs();
        let n_steps = self.config.n_steps;

        let mut buffer = RolloutBuffer::new(
            n_steps,
            num_envs,
            self.obs_dim,
            // Discrete actions are stored as a single index per env.
            if self.is_discrete { 1 } else { self.act_dim },
            self.device,
        )?;

        // Reset environments and get initial observations
        let mut obs = self.env.reset(&self.device)?;

        let mut episode_rewards: Vec<f32> = Vec::new();
        let mut episode_lengths: Vec<usize> = Vec::new();
        let mut current_rewards = vec![0.0f32; num_envs];
        let mut current_lengths = vec![0usize; num_envs];

        for _step in 0..n_steps {
            // Get actions and log probs
            let (actions, log_probs) = self.sample_action(&obs)?;

            // Get values
            let values = self.value_forward(&obs)?.squeeze(1)?;

            // Step environment
            let step_result = self.env.step(&actions, &self.device)?;

            // Store transition with separate terminated/truncated signals
            // This enables correct GAE calculation that bootstraps value for truncations
            buffer.add(
                &obs,
                &actions,
                &step_result.rewards,
                &step_result.terminated,
                &step_result.truncated,
                &values,
                &log_probs,
            )?;

            // Track episode statistics
            let rewards_vec: Vec<f32> = step_result.rewards.to_vec1()?;
            let dones_vec: Vec<f32> = step_result.dones()?.to_vec1()?;

            for i in 0..num_envs {
                current_rewards[i] += rewards_vec[i];
                current_lengths[i] += 1;

                if dones_vec[i] > 0.5 {
                    episode_rewards.push(current_rewards[i]);
                    episode_lengths.push(current_lengths[i]);
                    current_rewards[i] = 0.0;
                    current_lengths[i] = 0;
                }
            }

            obs = step_result.observations;
        }

        // Compute last value for GAE
        let last_values = self.value_forward(&obs)?.squeeze(1)?;
        buffer.compute_returns_and_advantages(
            &last_values,
            self.config.gamma,
            self.config.gae_lambda,
        )?;

        self.total_timesteps += n_steps * num_envs;

        Ok((buffer, episode_rewards, episode_lengths))
    }

    /// Perform the policy phase update (similar to PPO).
    fn policy_phase_update(&mut self, buffer: &RolloutBuffer) -> Result<TrainMetrics> {
        let samples = buffer.get_all()?;
        let n_samples = samples.observations.dim(0)?;

        // Flatten for batch processing
        let obs_flat = samples.observations.reshape(&[n_samples, self.obs_dim])?;
        let actions_flat = if self.is_discrete {
            samples.actions.reshape(&[n_samples, 1])?
        } else {
            samples.actions.reshape(&[n_samples, self.act_dim])?
        };
        let old_log_probs = samples.log_probs.flatten_all()?;
        let advantages = samples.advantages.flatten_all()?;
        let returns = samples.returns.flatten_all()?;

        // Normalize advantages
        let advantages = if self.config.normalize_advantage {
            let mean = advantages.mean_all()?;
            let var = advantages.var(0)?;
            let std = (var + 1e-8)?.sqrt()?;
            ((advantages - mean)? / std)?
        } else {
            advantages
        };

        // Persistent optimizers are reused across phases (Adam state survives).

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy_loss = 0.0f32;
        let mut total_approx_kl = 0.0f32;
        let mut total_clip_fraction = 0.0f32;
        let mut n_updates = 0usize;

        // Generate batch indices
        let n_batches = n_samples.div_ceil(self.config.batch_size);

        // Policy phase: train policy for policy_epochs
        let mut continue_training = true;
        for _epoch in 0..self.config.policy_epochs {
            if !continue_training {
                break;
            }
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut self.rng);

            for batch_idx in 0..n_batches {
                let start = batch_idx * self.config.batch_size;
                let end = (start + self.config.batch_size).min(n_samples);
                let batch_indices: Vec<i64> =
                    indices[start..end].iter().map(|&i| i as i64).collect();

                let candle_device = self.device.to_candle()?;
                let idx_tensor =
                    Tensor::from_slice(&batch_indices, &[batch_indices.len()], &candle_device)?;

                // Get batch data
                let batch_obs = obs_flat.index_select(&idx_tensor, 0)?;
                let batch_actions = actions_flat.index_select(&idx_tensor, 0)?;
                let batch_old_log_probs = old_log_probs.index_select(&idx_tensor, 0)?;
                let batch_advantages = advantages.index_select(&idx_tensor, 0)?;

                // Evaluate actions under current policy
                let (new_log_probs, _, entropy) =
                    self.evaluate_actions(&batch_obs, &batch_actions)?;

                // PPO clipped surrogate objective
                let log_ratio = (&new_log_probs - &batch_old_log_probs)?;
                let ratio = log_ratio.exp()?;

                // Surrogate losses
                let surr1 = (&ratio * &batch_advantages)?;
                let ratio_clipped =
                    ratio.clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range)?;
                let surr2 = (&ratio_clipped * &batch_advantages)?;

                // Take minimum of surr1 and surr2
                let surr_min = surr1.minimum(&surr2)?;
                let policy_loss = surr_min.neg()?.mean_all()?;

                // Entropy loss (negative because we want to maximize entropy)
                let entropy_loss = entropy.neg()?.mean_all()?;

                // Total policy loss
                let total_policy = (&policy_loss + (&entropy_loss * self.config.ent_coef as f64)?)?;

                // Backward pass for policy
                self.policy_optimizer.backward_step(&total_policy)?;

                // Collect metrics
                let policy_loss_val: f32 = policy_loss.to_scalar()?;
                let entropy_val: f32 = entropy.mean_all()?.to_scalar()?;

                total_policy_loss += policy_loss_val;
                total_entropy_loss += entropy_val;

                // Approximate KL divergence
                let approx_kl: f32 = log_ratio.sqr()?.mean_all()?.to_scalar::<f32>()? * 0.5;
                total_approx_kl += approx_kl;

                // Clip fraction
                let ratio_vec: Vec<f32> = ratio.flatten_all()?.to_vec1()?;
                let clip_frac = ratio_vec
                    .iter()
                    .filter(|&&r| (r - 1.0).abs() > self.config.clip_range)
                    .count() as f32
                    / ratio_vec.len() as f32;
                total_clip_fraction += clip_frac;

                n_updates += 1;

                // Early stopping based on KL divergence. Flag the outer epoch
                // loop to stop as well; a bare `break` only ends the current
                // epoch's minibatches and lets later epochs resume updating.
                if let Some(target_kl) = self.config.target_kl {
                    if approx_kl > target_kl {
                        debug!(
                            "Early stopping policy phase at epoch {} due to KL={}",
                            _epoch, approx_kl
                        );
                        continue_training = false;
                        break;
                    }
                }
            }
        }

        // Value phase: train value function for value_epochs
        for _epoch in 0..self.config.value_epochs {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut self.rng);

            for batch_idx in 0..n_batches {
                let start = batch_idx * self.config.batch_size;
                let end = (start + self.config.batch_size).min(n_samples);
                let batch_indices: Vec<i64> =
                    indices[start..end].iter().map(|&i| i as i64).collect();

                let candle_device = self.device.to_candle()?;
                let idx_tensor =
                    Tensor::from_slice(&batch_indices, &[batch_indices.len()], &candle_device)?;

                let batch_obs = obs_flat.index_select(&idx_tensor, 0)?;
                let batch_returns = returns.index_select(&idx_tensor, 0)?;

                // Value loss (MSE)
                let values = self.value_forward(&batch_obs)?.squeeze(1)?;
                let value_diff = (&values - &batch_returns)?;
                let value_loss = value_diff.sqr()?.mean_all()?;

                // Backward pass for value
                self.value_optimizer.backward_step(&value_loss)?;

                let value_loss_val: f32 = value_loss.to_scalar()?;
                total_value_loss += value_loss_val;
            }
        }

        let n_updates_f = n_updates.max(1) as f32;
        let n_value_updates = (n_batches * self.config.value_epochs) as f32;

        Ok(TrainMetrics {
            policy_loss: total_policy_loss / n_updates_f,
            value_loss: total_value_loss / n_value_updates.max(1.0),
            entropy: total_entropy_loss / n_updates_f,
            approx_kl: total_approx_kl / n_updates_f,
            clip_fraction: total_clip_fraction / n_updates_f,
            explained_variance: 0.0,
            learning_rate: self.config.learning_rate,
            timesteps: self.total_timesteps,
            episodes: 0,
            mean_reward: 0.0,
            std_reward: 0.0,
        })
    }

    /// Store rollout data for auxiliary training.
    fn store_aux_rollout(&mut self, buffer: &RolloutBuffer) -> Result<()> {
        let samples = buffer.get_all()?;
        let n_samples = samples.observations.dim(0)?;

        let obs_flat = samples.observations.reshape(&[n_samples, self.obs_dim])?;
        let returns = samples.returns.flatten_all()?;

        // Store current policy output for behavioral cloning
        let policy_output = self.policy_forward(&obs_flat)?.detach();

        self.aux_rollouts.push(AuxRollout {
            observations: obs_flat.detach(),
            returns: returns.detach(),
            old_policy_output: policy_output,
        });

        self.rollouts_since_aux += 1;

        Ok(())
    }

    /// Perform the auxiliary phase update.
    ///
    /// This trains the value function on stored rollouts while using
    /// a behavioral cloning loss to prevent policy drift.
    fn auxiliary_phase_update(&mut self) -> Result<()> {
        if self.aux_rollouts.is_empty() {
            return Ok(());
        }

        info!(
            "Starting auxiliary phase with {} stored rollouts",
            self.aux_rollouts.len()
        );

        // Persistent optimizers reused from the policy/value phase.

        for epoch in 0..self.config.aux_epochs {
            let mut total_aux_value_loss = 0.0f32;
            let mut total_bc_loss = 0.0f32;
            let mut n_batches = 0usize;

            // Iterate over stored rollouts
            for aux_rollout in &self.aux_rollouts {
                let n_samples = aux_rollout.observations.dim(0)?;
                let batches = n_samples.div_ceil(self.config.batch_size);

                let mut indices: Vec<usize> = (0..n_samples).collect();
                indices.shuffle(&mut self.rng);

                for batch_idx in 0..batches {
                    let start = batch_idx * self.config.batch_size;
                    let end = (start + self.config.batch_size).min(n_samples);
                    let batch_indices: Vec<i64> =
                        indices[start..end].iter().map(|&i| i as i64).collect();

                    let candle_device = self.device.to_candle()?;
                    let idx_tensor =
                        Tensor::from_slice(&batch_indices, &[batch_indices.len()], &candle_device)?;

                    let batch_obs = aux_rollout.observations.index_select(&idx_tensor, 0)?;
                    let batch_returns = aux_rollout.returns.index_select(&idx_tensor, 0)?;
                    let batch_old_policy =
                        aux_rollout.old_policy_output.index_select(&idx_tensor, 0)?;

                    // Value loss (MSE)
                    let values = self.value_forward(&batch_obs)?.squeeze(1)?;
                    let value_diff = (&values - &batch_returns)?;
                    let aux_value_loss = value_diff.sqr()?.mean_all()?;

                    // Behavioral cloning loss to prevent policy drift
                    let current_policy_output = self.policy_forward(&batch_obs)?;
                    let bc_diff = (&current_policy_output - &batch_old_policy)?;
                    let bc_loss = bc_diff.sqr()?.mean_all()?;

                    // Combined auxiliary loss
                    let aux_policy_loss = (&bc_loss * self.config.beta_clone as f64)?;
                    let aux_total_loss = (&aux_value_loss * self.config.vf_coef as f64)?;

                    // Update value network
                    self.value_optimizer.backward_step(&aux_total_loss)?;

                    // Update policy network with BC loss
                    if self.config.beta_clone > 0.0 {
                        self.policy_optimizer.backward_step(&aux_policy_loss)?;
                    }

                    total_aux_value_loss += aux_value_loss.to_scalar::<f32>()?;
                    total_bc_loss += bc_loss.to_scalar::<f32>()?;
                    n_batches += 1;
                }
            }

            if n_batches > 0 {
                debug!(
                    "Aux epoch {}/{}: value_loss={:.4}, bc_loss={:.4}",
                    epoch + 1,
                    self.config.aux_epochs,
                    total_aux_value_loss / n_batches as f32,
                    total_bc_loss / n_batches as f32
                );
            }
        }

        // Clear stored rollouts
        self.aux_rollouts.clear();
        self.rollouts_since_aux = 0;

        info!("Auxiliary phase completed");

        Ok(())
    }

    /// Train the agent for a given number of timesteps.
    ///
    /// # Arguments
    ///
    /// * `total_timesteps` - Total number of environment steps to train for
    /// * `callback` - Callback function called with training metrics after each update
    pub fn train<F>(&mut self, total_timesteps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!("Starting PPG training for {} timesteps", total_timesteps);

        let num_envs = self.env.num_envs();
        let n_steps = self.config.n_steps;
        let rollout_timesteps = n_steps * num_envs;
        let n_iterations = total_timesteps.div_ceil(rollout_timesteps);

        for iteration in 0..n_iterations {
            // Collect rollout
            let (buffer, episode_rewards, _episode_lengths) = self.collect_rollout()?;

            // Store rollout for auxiliary training
            self.store_aux_rollout(&buffer)?;

            // Policy phase update
            let mut metrics = self.policy_phase_update(&buffer)?;

            // Check if it's time for auxiliary phase
            if self.rollouts_since_aux >= self.config.num_aux_rollouts {
                self.auxiliary_phase_update()?;
            }

            // Update metrics with episode info
            if !episode_rewards.is_empty() {
                metrics.mean_reward =
                    episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
                let mean = metrics.mean_reward;
                metrics.std_reward = (episode_rewards
                    .iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f32>()
                    / episode_rewards.len() as f32)
                    .sqrt();
                metrics.episodes = episode_rewards.len();
            }

            metrics.timesteps = self.total_timesteps;

            if iteration % 10 == 0 {
                info!(
                    "Iteration {}/{}: timesteps={}, mean_reward={:.2}, policy_loss={:.4}, value_loss={:.4}",
                    iteration + 1,
                    n_iterations,
                    self.total_timesteps,
                    metrics.mean_reward,
                    metrics.policy_loss,
                    metrics.value_loss
                );
            }

            callback(&metrics);

            if self.total_timesteps >= total_timesteps {
                break;
            }
        }

        // Final auxiliary phase if there are stored rollouts
        if !self.aux_rollouts.is_empty() {
            self.auxiliary_phase_update()?;
        }

        info!(
            "PPG training completed: {} total timesteps",
            self.total_timesteps
        );
        Ok(())
    }

    /// Predict action for given observation (inference mode).
    ///
    /// # Arguments
    ///
    /// * `obs` - Observation tensor
    /// * `deterministic` - Whether to use deterministic (argmax/mean) or stochastic actions
    pub fn predict(&mut self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        let logits_or_mean = self.policy_forward(obs)?;

        if self.is_discrete {
            if deterministic {
                Ok(logits_or_mean
                    .argmax(1)?
                    .unsqueeze(1)?
                    .to_dtype(DType::F32)?)
            } else {
                let (action, _) = self.sample_action(obs)?;
                Ok(action)
            }
        } else if deterministic {
            Ok(logits_or_mean)
        } else {
            let (action, _) = self.sample_action(obs)?;
            Ok(action)
        }
    }

    /// Get the current value estimate for observations.
    pub fn get_value(&self, obs: &Tensor) -> Result<Tensor> {
        self.value_forward(obs)
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for PPGAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let (buffer, episode_rewards, _) = self.collect_rollout()?;

        // Store for auxiliary training
        self.store_aux_rollout(&buffer)?;

        // Policy phase
        let mut metrics = self.policy_phase_update(&buffer)?;

        // Check if auxiliary phase is needed
        if self.rollouts_since_aux >= self.config.num_aux_rollouts {
            self.auxiliary_phase_update()?;
        }

        if !episode_rewards.is_empty() {
            metrics.mean_reward =
                episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
            metrics.episodes = episode_rewards.len();
        }

        Ok(metrics)
    }

    fn save(&self, path: &Path) -> Result<()> {
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        // Save policy weights
        for (name, var) in self.policy_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        // Save value weights
        for (name, var) in self.value_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&tensors, path)?;

        // Save config separately
        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("PPG model saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        // Load policy weights
        let mut policy_data = self.policy_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("policy") {
                if let Some(var) = policy_data.get_mut(name) {
                    var.set(tensor)?;
                }
            }
        }
        drop(policy_data);

        // Load value weights
        let mut value_data = self.value_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("value") {
                if let Some(var) = value_data.get_mut(name) {
                    var.set(tensor)?;
                }
            }
        }

        info!("PPG model loaded from {:?}", path);
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
    fn test_ppg_config_defaults() {
        let config = PPGConfig::default();
        assert!((config.learning_rate - 5e-4).abs() < 1e-8);
        assert_eq!(config.policy_epochs, 1);
        assert_eq!(config.aux_epochs, 6);
        assert_eq!(config.num_aux_rollouts, 32);
        assert!((config.beta_clone - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_ppg_config_builder() {
        let config = PPGConfig::new()
            .learning_rate(1e-3)
            .policy_epochs(2)
            .aux_epochs(8)
            .num_aux_rollouts(16)
            .beta_clone(0.5);

        assert!((config.learning_rate - 1e-3).abs() < 1e-8);
        assert_eq!(config.policy_epochs, 2);
        assert_eq!(config.aux_epochs, 8);
        assert_eq!(config.num_aux_rollouts, 16);
        assert!((config.beta_clone - 0.5).abs() < 1e-8);
    }

    #[test]
    fn test_ppg_config_validation() {
        let config = PPGConfig::default();
        assert!(config.validate().is_ok());

        let invalid = PPGConfig::default().learning_rate(-0.1);
        assert!(invalid.validate().is_err());

        let invalid_aux = PPGConfig::default().num_aux_rollouts(0);
        assert!(invalid_aux.validate().is_err());

        let invalid_beta = PPGConfig::default().beta_clone(-1.0);
        assert!(invalid_beta.validate().is_err());
    }
}
