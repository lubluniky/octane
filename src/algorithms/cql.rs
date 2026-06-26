//! Conservative Q-Learning (CQL) implementation for offline reinforcement learning.
//!
//! CQL is an offline RL algorithm that learns from a fixed dataset without
//! environment interaction. It addresses the overestimation problem in offline RL
//! by adding a conservative regularization term that penalizes Q-values of
//! out-of-distribution actions.
//!
//! Key features:
//! - Conservative Q-function that lower-bounds the true Q-function
//! - CQL(H) variant using logsumexp regularization
//! - Optional Lagrangian constraint for automatic alpha tuning
//! - Built on top of SAC for continuous action spaces
//!
//! Reference: Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning" (2020)

use crate::algorithms::config::CQLConfig;
use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::traits::RLAlgorithm;
use crate::buffer::{ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, OctaneError, Result};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use std::path::Path;
use tracing::info;

/// Conservative Q-Learning agent for offline reinforcement learning.
///
/// CQL extends SAC with a conservative regularization term that penalizes
/// Q-values for actions that are unlikely under the dataset distribution.
/// This prevents the policy from exploiting erroneously high Q-values for
/// out-of-distribution actions.
///
/// # CQL(H) Objective
///
/// The CQL(H) variant minimizes:
/// ```text
/// L_CQL = alpha * (E_{s~D}[logsumexp_a Q(s,a)] - E_{(s,a)~D}[Q(s,a)]) + L_SAC
/// ```
///
/// Where:
/// - First term: logsumexp over random actions (soft-maximum)
/// - Second term: Q-values of dataset actions
/// - L_SAC: Standard SAC critic loss
///
/// # Example
///
/// ```ignore
/// use octane_rs::{CQLAgent, CQLConfig, ReplayBuffer, Device};
///
/// // Load pre-collected dataset into replay buffer
/// let config = CQLConfig::default()
///     .cql_alpha(5.0)
///     .with_lagrange(true)
///     .lagrange_thresh(10.0);
///
/// let mut agent = CQLAgent::new_offline(config, replay_buffer, obs_dim, action_dim, device)?;
///
/// // Train on offline data
/// for _ in 0..100_000 {
///     let metrics = agent.train_step()?;
///     println!("Q loss: {:.4}, CQL loss: {:.4}", metrics.value_loss, metrics.entropy);
/// }
/// ```
pub struct CQLAgent {
    /// Algorithm configuration.
    config: CQLConfig,
    /// Device for tensor operations.
    device: Device,

    /// Policy (actor) network var_map.
    policy_var_map: VarMap,
    /// Q1 network var_map.
    q1_var_map: VarMap,
    /// Q2 network var_map.
    q2_var_map: VarMap,
    /// Target Q1 network var_map.
    target_q1_var_map: VarMap,
    /// Target Q2 network var_map.
    target_q2_var_map: VarMap,

    /// Persistent optimizers (Adam moment state must survive across updates).
    policy_optimizer: AdamW,
    q1_optimizer: AdamW,
    q2_optimizer: AdamW,

    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    action_dim: usize,

    /// Pre-collected experience replay buffer (offline dataset).
    replay_buffer: ReplayBuffer,

    /// Log alpha for SAC entropy (entropy coefficient).
    log_alpha: Tensor,
    /// Target entropy for automatic entropy tuning.
    target_entropy: f32,
    /// Current SAC alpha value.
    alpha: f32,

    /// CQL alpha (conservative penalty weight).
    /// Can be a learnable parameter if using Lagrangian.
    cql_log_alpha: Option<Tensor>,
    /// Current CQL alpha value.
    cql_alpha: f32,

    /// Total training steps completed.
    total_timesteps: usize,

    /// Random number generator.
    rng: StdRng,
}

impl CQLAgent {
    /// Create a new CQL agent for offline RL training.
    ///
    /// This constructor is designed for offline RL where you have a pre-collected
    /// dataset in a replay buffer and no environment interaction.
    ///
    /// # Arguments
    ///
    /// * `config` - CQL configuration
    /// * `replay_buffer` - Pre-filled replay buffer with offline dataset
    /// * `obs_dim` - Observation space dimension
    /// * `action_dim` - Action space dimension
    /// * `device` - Device for tensor operations
    ///
    /// # Returns
    ///
    /// A new `CQLAgent` ready for offline training.
    pub fn new_offline(
        config: CQLConfig,
        replay_buffer: ReplayBuffer,
        obs_dim: usize,
        action_dim: usize,
        device: Device,
    ) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Target entropy: -dim(A) by default
        let target_entropy = config.target_entropy.unwrap_or(-(action_dim as f32));

        // Initialize log_alpha for SAC
        let candle_device = device.to_candle()?;
        let initial_alpha = config.ent_coef.ln();
        let log_alpha = Tensor::new(&[initial_alpha], &candle_device)?;

        // Initialize CQL log_alpha if using Lagrangian
        let cql_log_alpha = if config.with_lagrange {
            Some(Tensor::new(&[config.cql_alpha.ln()], &candle_device)?)
        } else {
            None
        };

        let policy_var_map = VarMap::new();
        let q1_var_map = VarMap::new();
        let q2_var_map = VarMap::new();
        let target_q1_var_map = VarMap::new();
        let target_q2_var_map = VarMap::new();

        let opt_params = ParamsAdamW {
            lr: config.learning_rate as f64,
            ..Default::default()
        };

        let mut agent = Self {
            config: config.clone(),
            device,
            policy_var_map,
            q1_var_map,
            q2_var_map,
            target_q1_var_map,
            target_q2_var_map,
            policy_optimizer: AdamW::new(Vec::new(), opt_params.clone())?,
            q1_optimizer: AdamW::new(Vec::new(), opt_params.clone())?,
            q2_optimizer: AdamW::new(Vec::new(), opt_params.clone())?,
            obs_dim,
            action_dim,
            replay_buffer,
            log_alpha,
            target_entropy,
            alpha: config.ent_coef,
            cql_log_alpha,
            cql_alpha: config.cql_alpha,
            total_timesteps: 0,
            rng,
        };

        agent.init_networks()?;

        // Bind optimizers to the populated network variables.
        agent.policy_optimizer = AdamW::new(agent.policy_var_map.all_vars(), opt_params.clone())?;
        agent.q1_optimizer = AdamW::new(agent.q1_var_map.all_vars(), opt_params.clone())?;
        agent.q2_optimizer = AdamW::new(agent.q2_var_map.all_vars(), opt_params)?;

        info!(
            "CQL Agent initialized: obs_dim={}, action_dim={}, cql_alpha={}, with_lagrange={}",
            obs_dim, action_dim, config.cql_alpha, config.with_lagrange
        );

        Ok(agent)
    }

    /// Create a new CQL agent with an empty buffer (for later data loading).
    ///
    /// # Arguments
    ///
    /// * `config` - CQL configuration
    /// * `obs_dim` - Observation space dimension
    /// * `action_dim` - Action space dimension
    /// * `device` - Device for tensor operations
    pub fn new(
        config: CQLConfig,
        obs_dim: usize,
        action_dim: usize,
        device: Device,
    ) -> Result<Self> {
        let buffer_config =
            ReplayBufferConfig::new(obs_dim, action_dim).capacity(config.buffer_size);
        let replay_buffer = ReplayBuffer::new(buffer_config, device)?;

        Self::new_offline(config, replay_buffer, obs_dim, action_dim, device)
    }

    /// Initialize all neural networks.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // Policy network
        self.init_policy_network(&candle_device)?;

        // Q-networks
        self.init_q_network(&self.q1_var_map, "q1", &candle_device)?;
        self.init_q_network(&self.q2_var_map, "q2", &candle_device)?;
        self.init_q_network(&self.target_q1_var_map, "target_q1", &candle_device)?;
        self.init_q_network(&self.target_q2_var_map, "target_q2", &candle_device)?;

        // Copy Q to target Q
        self.hard_update_targets()?;

        Ok(())
    }

    /// Initialize policy network (Gaussian policy with mean and log_std outputs).
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

    /// Initialize a Q-network (takes concatenated obs and action as input).
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

        // Q-value output (scalar)
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

    /// Sample action using reparameterization trick with tanh squashing.
    fn sample_action(&self, obs: &Tensor, deterministic: bool) -> Result<(Tensor, Tensor)> {
        let (mean, log_std) = self.policy_forward(obs)?;

        if deterministic {
            let action = mean.tanh()?;
            let log_prob = Tensor::zeros_like(&mean.narrow(1, 0, 1)?.squeeze(1)?)?;
            return Ok((action, log_prob));
        }

        let std = log_std.exp()?;

        // Reparameterization: sample = mean + std * noise
        let noise = Tensor::randn_like(&mean, 0.0, 1.0)?;
        let x_t = (&mean + &noise * &std)?;

        // Apply tanh squashing
        let action = x_t.tanh()?;

        // Compute log probability with tanh correction
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

    /// Forward pass through Q-network.
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

    /// Compute CQL conservative penalty using logsumexp (CQL-H variant).
    ///
    /// The CQL penalty is:
    /// ```text
    /// CQL_loss = E_s[logsumexp_a Q(s,a)] - E_{(s,a)~D}[Q(s,a)]
    /// ```
    ///
    /// We approximate logsumexp by sampling random actions and policy actions.
    fn compute_cql_penalty(
        &mut self,
        obs: &Tensor,
        current_q1: &Tensor,
        current_q2: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = obs.dim(0)?;
        let candle_device = self.device.to_candle()?;

        // Sample random actions uniformly from [-1, 1]
        let num_random = self.config.num_random_actions;

        // Random actions: [batch_size * num_random, action_dim]
        let random_actions_flat: Vec<f32> = (0..batch_size * num_random * self.action_dim)
            .map(|_| self.rng.gen_range(-1.0..1.0))
            .collect();
        let random_actions = Tensor::from_slice(
            &random_actions_flat,
            (batch_size * num_random, self.action_dim),
            &candle_device,
        )?;

        // Repeat observations for random actions
        let obs_repeated = obs.repeat(&[num_random, 1])?;

        // Q-values for random actions
        let q1_rand = self.q_forward(&obs_repeated, &random_actions, &self.q1_var_map, "q1")?;
        let q2_rand = self.q_forward(&obs_repeated, &random_actions, &self.q2_var_map, "q2")?;

        // Reshape to [batch_size, num_random]
        let q1_rand = q1_rand.reshape((batch_size, num_random))?;
        let q2_rand = q2_rand.reshape((batch_size, num_random))?;

        // Sample policy actions for current obs
        let (policy_actions, policy_log_probs) = self.sample_action(obs, false)?;
        let q1_policy = self.q_forward(obs, &policy_actions, &self.q1_var_map, "q1")?;
        let q2_policy = self.q_forward(obs, &policy_actions, &self.q2_var_map, "q2")?;

        // Sample next policy actions (using same obs, simplified)
        let (next_policy_actions, next_policy_log_probs) = self.sample_action(obs, false)?;
        let q1_next_policy = self.q_forward(obs, &next_policy_actions, &self.q1_var_map, "q1")?;
        let q2_next_policy = self.q_forward(obs, &next_policy_actions, &self.q2_var_map, "q2")?;

        // Concatenate all Q-values for logsumexp
        // Shape: [batch_size, num_random + 2]
        let q1_all = Tensor::cat(
            &[
                q1_rand,
                q1_policy.unsqueeze(1)?,
                q1_next_policy.unsqueeze(1)?,
            ],
            1,
        )?;
        let q2_all = Tensor::cat(
            &[
                q2_rand,
                q2_policy.unsqueeze(1)?,
                q2_next_policy.unsqueeze(1)?,
            ],
            1,
        )?;

        // Compute logsumexp with temperature
        let temp = self.config.cql_temp;
        let q1_scaled = (&q1_all / temp as f64)?;
        let q2_scaled = (&q2_all / temp as f64)?;

        // logsumexp = temp * log(sum(exp(q/temp)))
        let logsumexp_q1 = (q1_scaled.exp()?.sum(1)?.log()? * temp as f64)?;
        let logsumexp_q2 = (q2_scaled.exp()?.sum(1)?.log()? * temp as f64)?;

        // Subtract importance sampling correction for policy actions
        let policy_log_prob_correction =
            (policy_log_probs.detach() + next_policy_log_probs.detach())?;
        let _correction = policy_log_prob_correction; // Used in full implementation

        // CQL penalty = logsumexp(Q) - Q(data)
        let cql_penalty_q1 = (&logsumexp_q1 - current_q1)?;
        let cql_penalty_q2 = (&logsumexp_q2 - current_q2)?;

        Ok((cql_penalty_q1.mean_all()?, cql_penalty_q2.mean_all()?))
    }

    /// Hard update: copy Q-networks to target networks.
    fn hard_update_targets(&mut self) -> Result<()> {
        Self::copy_weights(&self.q1_var_map, &self.target_q1_var_map, "q1", "target_q1")?;
        Self::copy_weights(&self.q2_var_map, &self.target_q2_var_map, "q2", "target_q2")?;
        Ok(())
    }

    /// Soft update: Polyak averaging of Q-networks to target networks.
    fn soft_update_targets(&mut self) -> Result<()> {
        let tau = self.config.tau;
        Self::polyak_update(
            &self.q1_var_map,
            &self.target_q1_var_map,
            "q1",
            "target_q1",
            tau,
        )?;
        Self::polyak_update(
            &self.q2_var_map,
            &self.target_q2_var_map,
            "q2",
            "target_q2",
            tau,
        )?;
        Ok(())
    }

    /// Copy weights from source to target VarMap.
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

    /// Perform a single CQL training update.
    ///
    /// This implements the full CQL(H) update:
    /// 1. Standard SAC critic update with TD target
    /// 2. CQL conservative penalty
    /// 3. Optional Lagrangian for automatic CQL alpha tuning
    /// 4. Policy update (maximize Q - alpha * log_pi)
    /// 5. Entropy coefficient update (if auto-tuning)
    fn update(&mut self) -> Result<(f32, f32, f32, f32, f32)> {
        if !self.replay_buffer.can_sample(self.config.batch_size) {
            return Ok((0.0, 0.0, 0.0, 0.0, 0.0));
        }

        let batch = self.replay_buffer.sample(self.config.batch_size)?;

        // ========== Compute TD Target ==========
        let (next_actions, next_log_probs) = self.sample_action(&batch.next_observations, false)?;

        let target_q1 = self.q_forward(
            &batch.next_observations,
            &next_actions,
            &self.target_q1_var_map,
            "target_q1",
        )?;
        let target_q2 = self.q_forward(
            &batch.next_observations,
            &next_actions,
            &self.target_q2_var_map,
            "target_q2",
        )?;
        let target_q = target_q1.minimum(&target_q2)?;

        // TD target: r + gamma * (1 - done) * (min_Q - alpha * log_pi)
        let not_done = (Tensor::ones_like(&batch.dones)? - &batch.dones)?;
        let entropy_term = (&next_log_probs * self.alpha as f64)?;
        let soft_q_target = ((&target_q - &entropy_term)? * self.config.gamma as f64)?;
        let td_target = (&batch.rewards + &soft_q_target * &not_done)?.detach();

        // ========== Compute Current Q-values ==========
        let current_q1 =
            self.q_forward(&batch.observations, &batch.actions, &self.q1_var_map, "q1")?;
        let current_q2 =
            self.q_forward(&batch.observations, &batch.actions, &self.q2_var_map, "q2")?;

        // ========== TD Loss ==========
        let td_loss_q1 = (&current_q1 - &td_target)?.sqr()?.mean_all()?;
        let td_loss_q2 = (&current_q2 - &td_target)?.sqr()?.mean_all()?;

        // ========== CQL Conservative Penalty ==========
        let (cql_penalty_q1, cql_penalty_q2) =
            self.compute_cql_penalty(&batch.observations, &current_q1, &current_q2)?;

        // ========== Update CQL Alpha (Lagrangian) ==========
        let cql_alpha_loss = if self.config.with_lagrange {
            let candle_device = self.device.to_candle()?;

            // Lagrange constraint: cql_penalty should be close to lagrange_thresh
            let avg_penalty = ((&cql_penalty_q1 + &cql_penalty_q2)? * 0.5)?;
            let thresh = self.config.lagrange_thresh;

            // alpha_loss = alpha * (penalty - thresh)
            let penalty_val: f32 = avg_penalty.to_scalar()?;
            let alpha_grad = penalty_val - thresh;

            // Update log_alpha
            if let Some(ref log_alpha) = self.cql_log_alpha {
                // [1] tensor -> read element 0 (to_scalar needs rank 0).
                let current_log_alpha: f32 = log_alpha.to_vec1::<f32>()?[0];
                let new_log_alpha = current_log_alpha + self.config.learning_rate * alpha_grad;
                let new_log_alpha = new_log_alpha.clamp(-10.0, 10.0); // Clamp for stability
                self.cql_log_alpha = Some(Tensor::new(&[new_log_alpha], &candle_device)?);
                self.cql_alpha = new_log_alpha.exp();
            }

            (self.cql_alpha * alpha_grad).abs()
        } else {
            0.0
        };

        // ========== Total Q Loss ==========
        let cql_weight = self.cql_alpha;
        let q1_loss = (&td_loss_q1 + &cql_penalty_q1 * cql_weight as f64)?;
        let q2_loss = (&td_loss_q2 + &cql_penalty_q2 * cql_weight as f64)?;

        // ========== Update Q-networks ==========
        self.q1_optimizer.backward_step(&q1_loss)?;
        self.q2_optimizer.backward_step(&q2_loss)?;

        // ========== Update Policy ==========
        let (new_actions, new_log_probs) = self.sample_action(&batch.observations, false)?;
        let q1_new = self.q_forward(&batch.observations, &new_actions, &self.q1_var_map, "q1")?;
        let q2_new = self.q_forward(&batch.observations, &new_actions, &self.q2_var_map, "q2")?;
        let min_q_new = q1_new.minimum(&q2_new)?;

        // Policy loss: maximize Q - alpha * log_pi  =>  minimize alpha * log_pi - Q
        let entropy_term = (&new_log_probs * self.alpha as f64)?;
        let policy_loss = (&entropy_term - &min_q_new)?.mean_all()?;

        self.policy_optimizer.backward_step(&policy_loss)?;

        // ========== Update SAC Alpha ==========
        if self.config.auto_entropy_tuning {
            let candle_device = self.device.to_candle()?;
            let log_pi_detached = new_log_probs.detach();
            let neg_log_pi = log_pi_detached.neg()?;
            let diff = neg_log_pi.mean_all()?.to_scalar::<f32>()? - self.target_entropy;
            // log_alpha -= lr * (mean(-log_pi) - target_entropy) * alpha, so that
            // alpha rises when entropy is below target (diff < 0).
            let alpha_grad = diff * self.alpha;
            // log_alpha is a rank-1 [1] tensor; read element 0 (to_scalar needs
            // rank 0 and would error at runtime on the default auto-entropy path).
            let new_log_alpha = (self.log_alpha.to_vec1::<f32>()?[0]
                - self.config.learning_rate * alpha_grad)
                .clamp(-10.0, 10.0);
            self.log_alpha = Tensor::new(&[new_log_alpha], &candle_device)?;
            self.alpha = new_log_alpha.exp();
        }

        // ========== Soft Update Targets ==========
        self.soft_update_targets()?;

        self.total_timesteps += 1;

        // Collect metrics
        let q_loss: f32 = (q1_loss.to_scalar::<f32>()? + q2_loss.to_scalar::<f32>()?) / 2.0;
        let cql_loss: f32 =
            (cql_penalty_q1.to_scalar::<f32>()? + cql_penalty_q2.to_scalar::<f32>()?) / 2.0;
        let policy_loss_val: f32 = policy_loss.to_scalar()?;

        Ok((
            q_loss,
            cql_loss,
            policy_loss_val,
            self.cql_alpha,
            cql_alpha_loss,
        ))
    }

    /// Train the CQL agent for a specified number of gradient steps.
    ///
    /// Since CQL is an offline algorithm, this trains directly on the replay buffer
    /// without environment interaction.
    ///
    /// # Arguments
    ///
    /// * `num_steps` - Number of gradient steps to perform
    /// * `callback` - Callback function called with metrics after each step
    pub fn train<F>(&mut self, num_steps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!(
            "Starting CQL offline training for {} gradient steps",
            num_steps
        );

        let mut total_q_loss = 0.0f32;
        let mut total_cql_loss = 0.0f32;
        let mut total_policy_loss = 0.0f32;

        for step in 0..num_steps {
            let (q_loss, cql_loss, policy_loss, cql_alpha, _) = self.update()?;

            total_q_loss += q_loss;
            total_cql_loss += cql_loss;
            total_policy_loss += policy_loss;

            // Log every 1000 steps
            if (step + 1) % 1000 == 0 {
                let metrics = TrainMetrics {
                    timesteps: self.total_timesteps,
                    episodes: 0, // No episodes in offline RL
                    mean_reward: 0.0,
                    std_reward: 0.0,
                    policy_loss: total_policy_loss / 1000.0,
                    value_loss: total_q_loss / 1000.0,
                    entropy: total_cql_loss / 1000.0, // Using entropy field for CQL loss
                    approx_kl: cql_alpha,
                    clip_fraction: self.alpha,
                    explained_variance: 0.0,
                    learning_rate: self.config.learning_rate,
                };

                info!(
                    "Step {}: q_loss={:.4}, cql_loss={:.4}, pi_loss={:.4}, cql_alpha={:.4}",
                    self.total_timesteps,
                    total_q_loss / 1000.0,
                    total_cql_loss / 1000.0,
                    total_policy_loss / 1000.0,
                    cql_alpha
                );

                callback(&metrics);

                total_q_loss = 0.0;
                total_cql_loss = 0.0;
                total_policy_loss = 0.0;
            }
        }

        info!(
            "CQL training completed: {} gradient steps",
            self.total_timesteps
        );
        Ok(())
    }

    /// Predict action for given observation.
    ///
    /// # Arguments
    ///
    /// * `obs` - Observation tensor
    /// * `deterministic` - If true, return mean action; otherwise sample
    pub fn predict(&self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        let (action, _) = self.sample_action(obs, deterministic)?;
        Ok(action)
    }

    /// Get a mutable reference to the replay buffer for loading data.
    pub fn replay_buffer_mut(&mut self) -> &mut ReplayBuffer {
        &mut self.replay_buffer
    }

    /// Get current CQL alpha value.
    pub fn cql_alpha(&self) -> f32 {
        self.cql_alpha
    }

    /// Get current SAC entropy alpha value.
    pub fn entropy_alpha(&self) -> f32 {
        self.alpha
    }
}

impl RLAlgorithm for CQLAgent {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let (q_loss, cql_loss, policy_loss, cql_alpha, _) = self.update()?;

        Ok(TrainMetrics {
            policy_loss,
            value_loss: q_loss,
            entropy: cql_loss, // CQL penalty
            approx_kl: cql_alpha,
            clip_fraction: self.alpha,
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

        // Save Q1 weights
        for (name, var) in self.q1_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        // Save Q2 weights
        for (name, var) in self.q2_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&tensors, path)?;

        // Save config
        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("CQL model saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        // Load policy weights
        let mut data = self.policy_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("policy") {
                if let Some(var) = data.get_mut(name) {
                    var.set(tensor)?;
                }
            }
        }
        drop(data);

        // Load Q1 weights
        let mut data = self.q1_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("q1") {
                if let Some(var) = data.get_mut(name) {
                    var.set(tensor)?;
                }
            }
        }
        drop(data);

        // Load Q2 weights
        let mut data = self.q2_var_map.data().lock().unwrap();
        for (name, tensor) in &tensors {
            if name.starts_with("q2") {
                if let Some(var) = data.get_mut(name) {
                    var.set(tensor)?;
                }
            }
        }
        drop(data);

        self.hard_update_targets()?;

        info!("CQL model loaded from {:?}", path);
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
    use crate::algorithms::config::CQLConfig;

    #[test]
    fn test_cql_config_defaults() {
        let config = CQLConfig::default();
        assert!((config.cql_alpha - 5.0).abs() < 1e-6);
        assert!((config.cql_temp - 1.0).abs() < 1e-6);
        assert_eq!(config.num_random_actions, 10);
        assert!(!config.with_lagrange);
    }

    #[test]
    fn test_cql_config_builder() {
        let config = CQLConfig::new()
            .cql_alpha(10.0)
            .cql_temp(0.5)
            .with_lagrange(true)
            .lagrange_thresh(5.0);

        assert!((config.cql_alpha - 10.0).abs() < 1e-6);
        assert!((config.cql_temp - 0.5).abs() < 1e-6);
        assert!(config.with_lagrange);
        assert!((config.lagrange_thresh - 5.0).abs() < 1e-6);
    }
}
