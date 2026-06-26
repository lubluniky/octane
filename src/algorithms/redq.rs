//! Randomized Ensembled Double Q-learning (REDQ) algorithm implementation.
//!
//! REDQ is an off-policy algorithm that achieves high sample efficiency through:
//! - An ensemble of Q-networks for uncertainty quantification
//! - Random subset selection for target Q computation (reduces overestimation)
//! - High update-to-data (UTD) ratio for sample efficiency
//!
//! Key features:
//! - Ensemble of N Q-networks (default: 10)
//! - Randomly sample M networks (default: 2) for target computation
//! - High UTD ratio (default: 20 gradient steps per environment step)
//! - Works with continuous action spaces
//!
//! Reference: Chen et al., "Randomized Ensembled Double Q-Learning" (2021)
//! https://arxiv.org/abs/2101.05982

use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::traits::RLAlgorithm;
use crate::buffer::{ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, OctaneError, Result};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tracing::info;

/// Configuration for Randomized Ensembled Double Q-learning (REDQ) algorithm.
///
/// REDQ achieves high sample efficiency through ensemble Q-learning
/// with random subset selection and high update-to-data ratio.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct REDQConfig {
    /// Learning rate for all networks.
    /// Default: 3e-4
    pub learning_rate: f32,

    /// Number of Q-networks in the ensemble.
    /// Default: 10
    pub ensemble_size: usize,

    /// Number of Q-networks to sample for target computation.
    /// Must be <= ensemble_size.
    /// Default: 2
    pub num_q_samples: usize,

    /// Number of gradient updates per environment step (UTD ratio).
    /// Higher values improve sample efficiency but increase computation.
    /// Default: 20
    pub utd_ratio: usize,

    /// Replay buffer size.
    /// Default: 1_000_000
    pub buffer_size: usize,

    /// Number of timesteps before learning starts.
    /// Default: 5000
    pub learning_starts: usize,

    /// Minibatch size for gradient updates.
    /// Default: 256
    pub batch_size: usize,

    /// Discount factor for future rewards.
    /// Default: 0.99
    pub gamma: f32,

    /// Soft update coefficient for target networks.
    /// Default: 0.005
    pub tau: f32,

    /// Initial entropy coefficient (alpha).
    /// Default: 0.2
    pub ent_coef: f32,

    /// Automatically tune entropy coefficient.
    /// Default: true
    pub auto_entropy_tuning: bool,

    /// Target entropy for automatic tuning (if auto_entropy_tuning).
    /// Default: None (uses -dim(action))
    pub target_entropy: Option<f32>,

    /// Policy network hidden layer sizes.
    /// Default: [256, 256]
    pub policy_hidden_sizes: Vec<usize>,

    /// Q-network hidden layer sizes.
    /// Default: [256, 256]
    pub q_hidden_sizes: Vec<usize>,

    /// Random seed for reproducibility.
    /// Default: None
    pub seed: Option<u64>,
}

impl Default for REDQConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            ensemble_size: 10,
            num_q_samples: 2,
            utd_ratio: 20,
            buffer_size: 1_000_000,
            learning_starts: 5000,
            batch_size: 256,
            gamma: 0.99,
            tau: 0.005,
            ent_coef: 0.2,
            auto_entropy_tuning: true,
            target_entropy: None,
            policy_hidden_sizes: vec![256, 256],
            q_hidden_sizes: vec![256, 256],
            seed: None,
        }
    }
}

impl REDQConfig {
    /// Create a new REDQ config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style setter for learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder-style setter for ensemble size.
    pub fn ensemble_size(mut self, n: usize) -> Self {
        self.ensemble_size = n;
        self
    }

    /// Builder-style setter for number of Q samples.
    pub fn num_q_samples(mut self, n: usize) -> Self {
        self.num_q_samples = n;
        self
    }

    /// Builder-style setter for UTD ratio.
    pub fn utd_ratio(mut self, ratio: usize) -> Self {
        self.utd_ratio = ratio;
        self
    }

    /// Builder-style setter for buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Builder-style setter for learning starts.
    pub fn learning_starts(mut self, n: usize) -> Self {
        self.learning_starts = n;
        self
    }

    /// Builder-style setter for batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Builder-style setter for gamma.
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Builder-style setter for tau.
    pub fn tau(mut self, t: f32) -> Self {
        self.tau = t;
        self
    }

    /// Builder-style setter for entropy coefficient.
    pub fn ent_coef(mut self, c: f32) -> Self {
        self.ent_coef = c;
        self
    }

    /// Builder-style setter for automatic entropy tuning.
    pub fn auto_entropy_tuning(mut self, enabled: bool) -> Self {
        self.auto_entropy_tuning = enabled;
        self
    }

    /// Builder-style setter for target entropy.
    pub fn target_entropy(mut self, entropy: f32) -> Self {
        self.target_entropy = Some(entropy);
        self
    }

    /// Builder-style setter for policy hidden sizes.
    pub fn policy_hidden_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.policy_hidden_sizes = sizes;
        self
    }

    /// Builder-style setter for Q hidden sizes.
    pub fn q_hidden_sizes(mut self, sizes: Vec<usize>) -> Self {
        self.q_hidden_sizes = sizes;
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
        if self.ensemble_size == 0 {
            return Err("ensemble_size must be positive".to_string());
        }
        if self.num_q_samples == 0 {
            return Err("num_q_samples must be positive".to_string());
        }
        if self.num_q_samples > self.ensemble_size {
            return Err("num_q_samples cannot exceed ensemble_size".to_string());
        }
        if self.utd_ratio == 0 {
            return Err("utd_ratio must be positive".to_string());
        }
        if self.buffer_size == 0 {
            return Err("buffer_size must be positive".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".to_string());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.tau) {
            return Err("tau must be in [0, 1]".to_string());
        }
        if self.policy_hidden_sizes.is_empty() {
            return Err("policy_hidden_sizes cannot be empty".to_string());
        }
        if self.q_hidden_sizes.is_empty() {
            return Err("q_hidden_sizes cannot be empty".to_string());
        }
        Ok(())
    }
}

/// REDQ Agent for continuous action spaces.
///
/// Implements Randomized Ensembled Double Q-Learning with:
/// - Ensemble of Q-networks for uncertainty estimation
/// - Random subset selection to reduce overestimation bias
/// - High UTD ratio for sample efficiency
pub struct REDQAgent<E: Environment + Clone + 'static> {
    /// Algorithm configuration.
    config: REDQConfig,

    /// Vectorized environment.
    env: VecEnv<E>,

    /// Device for tensor operations.
    device: Device,

    /// Policy (actor) network var_map.
    policy_var_map: VarMap,

    /// Ensemble of Q-network var_maps.
    q_var_maps: Vec<VarMap>,

    /// Ensemble of target Q-network var_maps.
    target_q_var_maps: Vec<VarMap>,

    /// Observation dimension.
    obs_dim: usize,

    /// Action dimension.
    action_dim: usize,

    /// Experience replay buffer.
    replay_buffer: ReplayBuffer,

    /// Log alpha (entropy coefficient).
    log_alpha: Tensor,

    /// Target entropy for automatic tuning.
    target_entropy: f32,

    /// Current alpha value.
    alpha: f32,

    /// Total timesteps trained.
    total_timesteps: usize,

    /// Total gradient updates performed.
    total_updates: usize,

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> REDQAgent<E> {
    /// Create a new REDQ agent.
    ///
    /// # Arguments
    ///
    /// * `config` - REDQ configuration
    /// * `env` - Vectorized environment
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// A new REDQ agent ready for training.
    pub fn new(config: REDQConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();

        let obs_dim = obs_space.flat_dim();
        let action_dim = act_space.flat_dim();

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Create replay buffer
        let buffer_config =
            ReplayBufferConfig::new(obs_dim, action_dim).capacity(config.buffer_size);
        let replay_buffer = ReplayBuffer::new(buffer_config, device)?;

        // Target entropy: -dim(A) by default
        let target_entropy = config.target_entropy.unwrap_or(-(action_dim as f32));

        // Initialize log_alpha
        let candle_device = device.to_candle()?;
        let initial_alpha = config.ent_coef.ln();
        let log_alpha = Tensor::new(&[initial_alpha], &candle_device)?;

        // Create VarMaps for policy and Q-ensemble
        let policy_var_map = VarMap::new();
        let q_var_maps: Vec<VarMap> = (0..config.ensemble_size).map(|_| VarMap::new()).collect();
        let target_q_var_maps: Vec<VarMap> =
            (0..config.ensemble_size).map(|_| VarMap::new()).collect();

        let mut agent = Self {
            config: config.clone(),
            env,
            device,
            policy_var_map,
            q_var_maps,
            target_q_var_maps,
            obs_dim,
            action_dim,
            replay_buffer,
            log_alpha,
            target_entropy,
            alpha: config.ent_coef,
            total_timesteps: 0,
            total_updates: 0,
            rng,
        };

        agent.init_networks()?;

        info!(
            "REDQ Agent initialized: obs_dim={}, action_dim={}, ensemble_size={}, utd_ratio={}",
            obs_dim, action_dim, config.ensemble_size, config.utd_ratio
        );

        Ok(agent)
    }

    /// Initialize all networks (policy and Q-ensemble).
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // Initialize policy network
        self.init_policy_network(&candle_device)?;

        // Initialize Q-network ensemble
        for i in 0..self.config.ensemble_size {
            self.init_q_network(&self.q_var_maps[i], &format!("q{}", i), &candle_device)?;
            self.init_q_network(
                &self.target_q_var_maps[i],
                &format!("target_q{}", i),
                &candle_device,
            )?;
        }

        // Hard copy to targets
        self.hard_update_targets()?;

        Ok(())
    }

    /// Initialize policy network (outputs mean and log_std).
    fn init_policy_network(&self, candle_device: &candle_core::Device) -> Result<()> {
        let vb = VarBuilder::from_varmap(&self.policy_var_map, DType::F32, candle_device);

        let mut in_dim = self.obs_dim;
        for (i, &hidden_size) in self.config.policy_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("policy.layer_{}", i)))?;
            in_dim = hidden_size;
        }

        // Mean and log_std outputs
        let _ = candle_nn::linear(in_dim, self.action_dim, vb.pp("policy.mean"))?;
        let _ = candle_nn::linear(in_dim, self.action_dim, vb.pp("policy.log_std"))?;

        Ok(())
    }

    /// Initialize a Q-network (takes obs and action as input).
    fn init_q_network(
        &self,
        var_map: &VarMap,
        prefix: &str,
        candle_device: &candle_core::Device,
    ) -> Result<()> {
        let vb = VarBuilder::from_varmap(var_map, DType::F32, candle_device);

        let mut in_dim = self.obs_dim + self.action_dim;
        for (i, &hidden_size) in self.config.q_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(
                in_dim,
                hidden_size,
                vb.pp(format!("{}.layer_{}", prefix, i)),
            )?;
            in_dim = hidden_size;
        }

        // Q-value output
        let _ = candle_nn::linear(in_dim, 1, vb.pp(format!("{}.output", prefix)))?;

        Ok(())
    }

    /// Forward pass through policy network.
    fn policy_forward(&self, obs: &Tensor) -> Result<(Tensor, Tensor)> {
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

        let last_dim = *self.config.policy_hidden_sizes.last().unwrap();
        let mean_linear = candle_nn::linear(last_dim, self.action_dim, vb.pp("policy.mean"))?;
        let log_std_linear = candle_nn::linear(last_dim, self.action_dim, vb.pp("policy.log_std"))?;

        let mean = mean_linear.forward(&x)?;
        let log_std = log_std_linear.forward(&x)?;

        // Clamp log_std for numerical stability
        let log_std = log_std.clamp(-20.0, 2.0)?;

        Ok((mean, log_std))
    }

    /// Sample action using reparameterization trick (SAC-style).
    fn sample_action(&self, obs: &Tensor, deterministic: bool) -> Result<(Tensor, Tensor)> {
        let (mean, log_std) = self.policy_forward(obs)?;

        if deterministic {
            let action = mean.tanh()?;
            let log_prob = Tensor::zeros_like(&mean.narrow(1, 0, 1)?.squeeze(1)?)?;
            return Ok((action, log_prob));
        }

        let std = log_std.exp()?;

        // Reparameterization: action = tanh(mean + std * noise)
        let noise = Tensor::randn_like(&mean, 0.0, 1.0)?;
        let x_t = (&mean + &noise * &std)?;
        let action = x_t.tanh()?;

        // Log probability with tanh squashing correction
        let log_prob_gaussian = self.gaussian_log_prob(&x_t, &mean, &log_std)?;
        let tanh_correction = (Tensor::ones_like(&action)? - action.sqr()?)?
            .clamp(1e-6, 1.0)?
            .log()?
            .sum(1)?;
        let log_prob = (log_prob_gaussian - tanh_correction)?;

        Ok((action, log_prob))
    }

    /// Compute Gaussian log probability.
    fn gaussian_log_prob(&self, x: &Tensor, mean: &Tensor, log_std: &Tensor) -> Result<Tensor> {
        let diff = (x - mean)?;
        let std = log_std.exp()?;
        let normalized = diff.broadcast_div(&std)?;

        let log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
        let candle_device = self.device.to_candle()?;
        let log_2pi_tensor = Tensor::new(&[log_2pi], &candle_device)?;

        let log_prob_per_dim = (normalized.sqr()? * (-0.5))?
            .broadcast_sub(log_std)?
            .broadcast_sub(&log_2pi_tensor)?;

        Ok(log_prob_per_dim.sum(1)?)
    }

    /// Forward pass through a Q-network.
    fn q_forward(
        &self,
        obs: &Tensor,
        action: &Tensor,
        var_map: &VarMap,
        prefix: &str,
    ) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(var_map, DType::F32, &candle_device);

        // Concatenate obs and action
        let x = Tensor::cat(&[obs, action], 1)?;

        let mut h = x;
        for (i, &hidden_size) in self.config.q_hidden_sizes.iter().enumerate() {
            let in_dim = if i == 0 {
                self.obs_dim + self.action_dim
            } else {
                self.config.q_hidden_sizes[i - 1]
            };
            let linear = candle_nn::linear(
                in_dim,
                hidden_size,
                vb.pp(format!("{}.layer_{}", prefix, i)),
            )?;
            h = linear.forward(&h)?;
            h = h.relu()?;
        }

        let output_linear = candle_nn::linear(
            *self.config.q_hidden_sizes.last().unwrap(),
            1,
            vb.pp(format!("{}.output", prefix)),
        )?;

        Ok(output_linear.forward(&h)?.squeeze(1)?)
    }

    /// Compute Q-values from all ensemble members.
    fn ensemble_q_values(&self, obs: &Tensor, action: &Tensor) -> Result<Vec<Tensor>> {
        let mut q_values = Vec::with_capacity(self.config.ensemble_size);
        for i in 0..self.config.ensemble_size {
            let q = self.q_forward(obs, action, &self.q_var_maps[i], &format!("q{}", i))?;
            q_values.push(q);
        }
        Ok(q_values)
    }

    /// Compute target Q-values from a random subset of ensemble members.
    fn random_subset_target_q(&mut self, obs: &Tensor, action: &Tensor) -> Result<Tensor> {
        // Randomly sample num_q_samples indices from the ensemble
        let mut indices: Vec<usize> = (0..self.config.ensemble_size).collect();
        indices.shuffle(&mut self.rng);
        let selected_indices = &indices[..self.config.num_q_samples];

        // Compute Q-values from selected target networks
        let mut q_values = Vec::with_capacity(self.config.num_q_samples);
        for &i in selected_indices {
            let q = self.q_forward(
                obs,
                action,
                &self.target_q_var_maps[i],
                &format!("target_q{}", i),
            )?;
            q_values.push(q);
        }

        // Take minimum across selected Q-networks
        let mut min_q = q_values[0].clone();
        for q in &q_values[1..] {
            min_q = min_q.minimum(q)?;
        }

        Ok(min_q)
    }

    /// Hard update: copy all Q-networks to their targets.
    fn hard_update_targets(&mut self) -> Result<()> {
        for i in 0..self.config.ensemble_size {
            Self::copy_weights(
                &self.q_var_maps[i],
                &self.target_q_var_maps[i],
                &format!("q{}", i),
                &format!("target_q{}", i),
            )?;
        }
        Ok(())
    }

    /// Soft update: polyak averaging for all target networks.
    fn soft_update_targets(&mut self) -> Result<()> {
        let tau = self.config.tau;
        for i in 0..self.config.ensemble_size {
            Self::polyak_update(
                &self.q_var_maps[i],
                &self.target_q_var_maps[i],
                &format!("q{}", i),
                &format!("target_q{}", i),
                tau,
            )?;
        }
        Ok(())
    }

    /// Copy weights from source to target var_map.
    fn copy_weights(src: &VarMap, dst: &VarMap, src_prefix: &str, dst_prefix: &str) -> Result<()> {
        let src_data = src.data().lock().unwrap();
        let mut dst_data = dst.data().lock().unwrap();

        for (name, var) in src_data.iter() {
            let dst_name = name.replace(src_prefix, dst_prefix);
            if let Some(dst_var) = dst_data.get_mut(&dst_name) {
                dst_var.set(var.as_tensor())?;
            }
        }

        Ok(())
    }

    /// Polyak averaging update.
    fn polyak_update(
        src: &VarMap,
        dst: &VarMap,
        src_prefix: &str,
        dst_prefix: &str,
        tau: f32,
    ) -> Result<()> {
        let src_data = src.data().lock().unwrap();
        let mut dst_data = dst.data().lock().unwrap();

        for (name, src_var) in src_data.iter() {
            let dst_name = name.replace(src_prefix, dst_prefix);
            if let Some(dst_var) = dst_data.get_mut(&dst_name) {
                let new_val = ((src_var.as_tensor() * tau as f64)?
                    + (dst_var.as_tensor() * (1.0 - tau) as f64)?)?;
                dst_var.set(&new_val)?;
            }
        }

        Ok(())
    }

    /// Perform a single training update.
    fn update(&mut self) -> Result<(f32, f32, f32, f32)> {
        if !self.replay_buffer.can_sample(self.config.batch_size) {
            return Ok((0.0, 0.0, 0.0, 0.0));
        }

        let batch = self.replay_buffer.sample(self.config.batch_size)?;

        // ========== Compute Target Q-value ==========
        // Sample next actions from current policy
        let (next_actions, next_log_probs) = self.sample_action(&batch.next_observations, false)?;

        // Compute target Q using random subset of target networks
        let target_q = self.random_subset_target_q(&batch.next_observations, &next_actions)?;

        // Target: r + gamma * (1 - done) * (min_Q - alpha * log_pi)
        let not_done = (Tensor::ones_like(&batch.dones)? - &batch.dones)?;
        let entropy_term = (&next_log_probs * self.alpha as f64)?;
        let soft_q_target = ((&target_q - &entropy_term)? * self.config.gamma as f64)?;
        let td_target = (&batch.rewards + &soft_q_target * &not_done)?.detach();

        // ========== Update Q-networks ==========
        let mut total_q_loss = 0.0f32;
        let params = ParamsAdamW {
            lr: self.config.learning_rate as f64,
            ..Default::default()
        };

        // Update each Q-network in the ensemble
        for i in 0..self.config.ensemble_size {
            let current_q = self.q_forward(
                &batch.observations,
                &batch.actions,
                &self.q_var_maps[i],
                &format!("q{}", i),
            )?;
            let q_loss = (&current_q - &td_target)?.sqr()?.mean_all()?;

            let mut q_optimizer = AdamW::new(self.q_var_maps[i].all_vars(), params.clone())?;
            q_optimizer.backward_step(&q_loss)?;

            total_q_loss += q_loss.to_scalar::<f32>()?;
        }

        let avg_q_loss = total_q_loss / self.config.ensemble_size as f32;

        // ========== Update Policy ==========
        let (new_actions, new_log_probs) = self.sample_action(&batch.observations, false)?;

        // Use all Q-networks for the policy update, taking the MEAN across the
        // ensemble (REDQ uses the in-target min only for the critic target; the
        // actor is trained against the ensemble mean to avoid over-pessimism).
        let q_values = self.ensemble_q_values(&batch.observations, &new_actions)?;
        let mut mean_q = q_values[0].clone();
        for q in &q_values[1..] {
            mean_q = (mean_q + q)?;
        }
        let mean_q = (mean_q / q_values.len() as f64)?;

        // Policy loss: maximize Q - alpha * log_pi
        let entropy_term = (&new_log_probs * self.alpha as f64)?;
        let policy_loss = (&entropy_term - &mean_q)?.mean_all()?;

        let mut policy_optimizer = AdamW::new(self.policy_var_map.all_vars(), params.clone())?;
        policy_optimizer.backward_step(&policy_loss)?;

        // ========== Update Alpha (if auto-tuning) ==========
        let alpha_loss_val = if self.config.auto_entropy_tuning {
            let candle_device = self.device.to_candle()?;
            let log_pi_detached = new_log_probs.detach();

            // Alpha loss gradient: log_alpha -= lr * (mean(-log_pi) - target_entropy) * alpha.
            // When entropy is below target (diff < 0), alpha increases.
            let neg_log_pi = log_pi_detached.neg()?;
            let diff = neg_log_pi.mean_all()?.to_scalar::<f32>()? - self.target_entropy;

            // Manual gradient step for alpha.
            let alpha_grad = diff * self.alpha;
            let new_log_alpha =
                (self.log_alpha.to_scalar::<f32>()? - self.config.learning_rate * alpha_grad)
                    .clamp(-10.0, 10.0);
            self.log_alpha = Tensor::new(&[new_log_alpha], &candle_device)?;
            self.alpha = new_log_alpha.exp();

            (self.alpha * diff).abs()
        } else {
            0.0
        };

        // Soft update targets
        self.soft_update_targets()?;

        let policy_loss_val: f32 = policy_loss.to_scalar()?;
        self.total_updates += 1;

        Ok((avg_q_loss, policy_loss_val, self.alpha, alpha_loss_val))
    }

    /// Train the agent for a given number of timesteps.
    ///
    /// # Arguments
    ///
    /// * `total_timesteps` - Total number of environment steps to train for
    /// * `callback` - Callback function called with training metrics
    pub fn train<F>(&mut self, total_timesteps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!(
            "Starting REDQ training for {} timesteps with UTD ratio {}",
            total_timesteps, self.config.utd_ratio
        );

        let num_envs = self.env.num_envs();
        let mut obs = self.env.reset(&self.device)?;

        let mut episode_rewards: Vec<f32> = Vec::new();
        let mut current_rewards = vec![0.0f32; num_envs];

        let mut total_q_loss = 0.0f32;
        let mut total_policy_loss = 0.0f32;
        let mut update_count = 0usize;

        while self.total_timesteps < total_timesteps {
            // Select actions
            let (actions, _) = self.sample_action(&obs, false)?;

            // Step environment
            let step_result = self.env.step(&actions, &self.device)?;

            // Store transitions
            let obs_vec: Vec<f32> = obs.flatten_all()?.to_vec1()?;
            let action_vec: Vec<f32> = actions.flatten_all()?.to_vec1()?;
            let next_obs_vec: Vec<f32> = step_result.observations.flatten_all()?.to_vec1()?;
            let rewards_vec: Vec<f32> = step_result.rewards.to_vec1()?;
            let dones_vec: Vec<f32> = step_result.dones()?.to_vec1()?;

            let obs_per_env = self.obs_dim;
            let act_per_env = self.action_dim;

            for i in 0..num_envs {
                self.replay_buffer.add(
                    &obs_vec[i * obs_per_env..(i + 1) * obs_per_env],
                    &action_vec[i * act_per_env..(i + 1) * act_per_env],
                    rewards_vec[i],
                    &next_obs_vec[i * obs_per_env..(i + 1) * obs_per_env],
                    dones_vec[i] > 0.5,
                );

                current_rewards[i] += rewards_vec[i];

                if dones_vec[i] > 0.5 {
                    episode_rewards.push(current_rewards[i]);
                    current_rewards[i] = 0.0;
                }
            }

            obs = step_result.observations;
            self.total_timesteps += num_envs;

            // High UTD ratio: perform multiple gradient updates per environment step
            if self.total_timesteps >= self.config.learning_starts {
                for _ in 0..self.config.utd_ratio {
                    let (q_loss, policy_loss, _, _) = self.update()?;
                    total_q_loss += q_loss;
                    total_policy_loss += policy_loss;
                    update_count += 1;
                }
            }

            // Logging
            if self.total_timesteps % 10000 < num_envs {
                let metrics = TrainMetrics {
                    timesteps: self.total_timesteps,
                    episodes: episode_rewards.len(),
                    mean_reward: if episode_rewards.is_empty() {
                        0.0
                    } else {
                        episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32
                    },
                    std_reward: 0.0,
                    policy_loss: if update_count > 0 {
                        total_policy_loss / update_count as f32
                    } else {
                        0.0
                    },
                    value_loss: if update_count > 0 {
                        total_q_loss / update_count as f32
                    } else {
                        0.0
                    },
                    entropy: self.alpha,
                    approx_kl: 0.0,
                    clip_fraction: 0.0,
                    explained_variance: 0.0,
                    learning_rate: self.config.learning_rate,
                };

                info!(
                    "Step {}: reward={:.2}, alpha={:.4}, q_loss={:.4}, pi_loss={:.4}, updates={}",
                    self.total_timesteps,
                    metrics.mean_reward,
                    self.alpha,
                    metrics.value_loss,
                    metrics.policy_loss,
                    self.total_updates
                );

                callback(&metrics);

                episode_rewards.clear();
                total_q_loss = 0.0;
                total_policy_loss = 0.0;
                update_count = 0;
            }
        }

        info!(
            "REDQ training completed: {} timesteps, {} gradient updates",
            self.total_timesteps, self.total_updates
        );
        Ok(())
    }

    /// Predict action for given observation.
    ///
    /// # Arguments
    ///
    /// * `obs` - Observation tensor
    /// * `deterministic` - Whether to use deterministic (mean) or stochastic actions
    pub fn predict(&self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        let (action, _) = self.sample_action(obs, deterministic)?;
        Ok(action)
    }

    /// Get Q-value estimates from the ensemble.
    ///
    /// Returns all Q-values from the ensemble for the given state-action pair.
    pub fn get_ensemble_q_values(&self, obs: &Tensor, action: &Tensor) -> Result<Vec<f32>> {
        let q_values = self.ensemble_q_values(obs, action)?;
        let mut results = Vec::with_capacity(q_values.len());
        for q in q_values {
            results.push(q.mean_all()?.to_scalar()?);
        }
        Ok(results)
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for REDQAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        // Perform UTD ratio number of gradient updates
        let mut total_q_loss = 0.0f32;
        let mut total_policy_loss = 0.0f32;

        for _ in 0..self.config.utd_ratio {
            let (q_loss, policy_loss, _, _) = self.update()?;
            total_q_loss += q_loss;
            total_policy_loss += policy_loss;
        }

        let n_updates = self.config.utd_ratio as f32;

        Ok(TrainMetrics {
            policy_loss: total_policy_loss / n_updates,
            value_loss: total_q_loss / n_updates,
            entropy: self.alpha,
            timesteps: self.total_timesteps,
            ..Default::default()
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        // Save policy weights
        for (name, var) in self.policy_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        // Save Q-ensemble weights
        for (i, var_map) in self.q_var_maps.iter().enumerate() {
            for (name, var) in var_map.data().lock().unwrap().iter() {
                tensors.insert(format!("ensemble_{}.{}", i, name), var.as_tensor().clone());
            }
        }

        candle_core::safetensors::save(&tensors, path)?;

        // Save config separately
        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("REDQ model saved to {:?}", path);
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

        // Load Q-ensemble weights
        for (i, var_map) in self.q_var_maps.iter().enumerate() {
            let prefix = format!("ensemble_{}.", i);
            let mut data = var_map.data().lock().unwrap();
            for (name, tensor) in &tensors {
                if name.starts_with(&prefix) {
                    let var_name = name.strip_prefix(&prefix).unwrap();
                    if let Some(var) = data.get_mut(var_name) {
                        var.set(tensor)?;
                    }
                }
            }
        }

        // Hard copy to targets
        self.hard_update_targets()?;

        info!("REDQ model loaded from {:?}", path);
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
    fn test_redq_config_defaults() {
        let config = REDQConfig::default();
        assert_eq!(config.ensemble_size, 10);
        assert_eq!(config.num_q_samples, 2);
        assert_eq!(config.utd_ratio, 20);
        assert!((config.gamma - 0.99).abs() < 1e-6);
        assert!(config.auto_entropy_tuning);
    }

    #[test]
    fn test_redq_config_builder() {
        let config = REDQConfig::new()
            .ensemble_size(5)
            .num_q_samples(3)
            .utd_ratio(10)
            .learning_rate(1e-3)
            .batch_size(128);

        assert_eq!(config.ensemble_size, 5);
        assert_eq!(config.num_q_samples, 3);
        assert_eq!(config.utd_ratio, 10);
        assert!((config.learning_rate - 1e-3).abs() < 1e-8);
        assert_eq!(config.batch_size, 128);
    }

    #[test]
    fn test_redq_config_validation() {
        let config = REDQConfig::default();
        assert!(config.validate().is_ok());

        let invalid = REDQConfig::default().ensemble_size(0);
        assert!(invalid.validate().is_err());

        let invalid_samples = REDQConfig::default().ensemble_size(5).num_q_samples(10);
        assert!(invalid_samples.validate().is_err());

        let invalid_utd = REDQConfig::default().utd_ratio(0);
        assert!(invalid_utd.validate().is_err());

        let invalid_gamma = REDQConfig::default().gamma(1.5);
        assert!(invalid_gamma.validate().is_err());
    }
}
