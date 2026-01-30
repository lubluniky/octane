//! Implicit Quantile Networks (IQN) implementation for distributional reinforcement learning.
//!
//! IQN is a distributional RL algorithm that learns the full distribution of returns
//! rather than just the expected value. It uses implicit quantile functions to
//! represent arbitrary return distributions without discretization.
//!
//! Key features:
//! - Learns return distributions via quantile regression
//! - Risk-sensitive action selection (CVaR, mean, optimistic)
//! - Cosine embedding for quantile inputs
//! - Quantile Huber loss for robust training
//!
//! Reference: Dabney et al., "Implicit Quantile Networks for Distributional RL" (2018)

use crate::algorithms::config::{IQNConfig, RiskMeasure};
use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::traits::RLAlgorithm;
use crate::buffer::{ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, OctaneError, Result};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use std::path::Path;
use tracing::info;

/// Implicit Quantile Network agent for distributional RL.
///
/// IQN learns the quantile function Z_tau(s, a) which gives the tau-th quantile
/// of the return distribution for state-action pair (s, a). The network takes
/// both the state and sampled quantile fractions as input.
///
/// # Architecture
///
/// The IQN consists of:
/// 1. State encoder: MLP that processes observations
/// 2. Quantile embedding: Cosine features for quantile fractions
/// 3. Combination layer: Element-wise multiplication of state and quantile embeddings
/// 4. Output layers: Q-value for each action at each quantile
///
/// # Risk-Sensitive Action Selection
///
/// IQN supports multiple risk measures:
/// - **Mean**: Average over quantiles (risk-neutral)
/// - **CVaR**: Average over lower quantiles (risk-averse)
/// - **Optimistic**: Average over upper quantiles (risk-seeking)
///
/// # Example
///
/// ```ignore
/// use octane_rs::{IQNAgent, IQNConfig, VecEnv, Device};
///
/// let config = IQNConfig::default()
///     .num_quantiles(64)
///     .embedding_dim(64)
///     .kappa(1.0);
///
/// let env = MyDiscreteEnv::new();
/// let vec_env = VecEnv::new(vec![env], 8);
/// let mut agent = IQNAgent::new(config, vec_env, Device::Cpu)?;
///
/// agent.train(1_000_000, |metrics| {
///     println!("Step {}: reward={:.2}", metrics.timesteps, metrics.mean_reward);
/// })?;
/// ```
pub struct IQNAgent<E: Environment + Clone + 'static> {
    /// Algorithm configuration.
    config: IQNConfig,
    /// Vectorized environment.
    env: VecEnv<E>,
    /// Device for tensor operations.
    device: Device,

    /// Online network var_map.
    online_var_map: VarMap,
    /// Target network var_map.
    target_var_map: VarMap,

    /// Observation dimension.
    obs_dim: usize,
    /// Number of discrete actions.
    num_actions: usize,
    /// Hidden layer sizes.
    hidden_sizes: Vec<usize>,

    /// Experience replay buffer.
    replay_buffer: ReplayBuffer,

    /// Current exploration rate (epsilon).
    epsilon: f32,
    /// Current learning rate.
    current_lr: f32,
    /// Total timesteps trained.
    total_timesteps: usize,
    /// Steps since last target network update.
    steps_since_target_update: usize,

    /// Risk measure for action selection.
    risk_measure: RiskMeasure,

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> IQNAgent<E> {
    /// Create a new IQN agent.
    ///
    /// # Arguments
    ///
    /// * `config` - IQN configuration
    /// * `env` - Vectorized environment (must have discrete action space)
    /// * `device` - Device for tensor operations
    pub fn new(config: IQNConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();

        let obs_dim = obs_space.flat_dim();
        let num_actions = act_space.flat_dim();

        // Verify discrete action space
        if act_space.shape() != [1] && act_space.shape().len() != 1 {
            return Err(OctaneError::InvalidConfig(
                "IQN requires discrete action space".into(),
            ));
        }

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Create replay buffer
        let buffer_config = ReplayBufferConfig::new(obs_dim, 1)
            .capacity(config.buffer_size)
            .prioritized(config.prioritized_replay);

        let replay_buffer = ReplayBuffer::new(buffer_config, device)?;

        let online_var_map = VarMap::new();
        let target_var_map = VarMap::new();
        let hidden_sizes = vec![256, 256];

        let mut agent = Self {
            config: config.clone(),
            env,
            device,
            online_var_map,
            target_var_map,
            obs_dim,
            num_actions,
            hidden_sizes,
            replay_buffer,
            epsilon: config.epsilon_start,
            current_lr: config.learning_rate,
            total_timesteps: 0,
            steps_since_target_update: 0,
            risk_measure: config.risk_measure,
            rng,
        };

        agent.init_networks()?;

        info!(
            "IQN Agent initialized: obs_dim={}, num_actions={}, num_quantiles={}, risk={:?}",
            obs_dim, num_actions, config.num_quantiles, config.risk_measure
        );

        Ok(agent)
    }

    /// Initialize online and target networks.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        self.init_iqn_network(&self.online_var_map, "online", &candle_device)?;
        self.init_iqn_network(&self.target_var_map, "target", &candle_device)?;

        // Copy online to target
        self.hard_update_target()?;

        Ok(())
    }

    /// Initialize an IQN network.
    ///
    /// The network has three parts:
    /// 1. State encoder (psi): obs -> embedding
    /// 2. Quantile embedding (phi): tau -> embedding using cosine features
    /// 3. Output network: combined -> Q-values per action
    fn init_iqn_network(
        &self,
        var_map: &VarMap,
        prefix: &str,
        candle_device: &candle_core::Device,
    ) -> Result<()> {
        let vb = VarBuilder::from_varmap(var_map, DType::F32, candle_device);
        let embed_dim = self.config.embedding_dim;

        // State encoder (psi network)
        let mut in_dim = self.obs_dim;
        for (i, &hidden_size) in self.hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(
                in_dim,
                hidden_size,
                vb.pp(format!("{}.psi.layer_{}", prefix, i)),
            )?;
            in_dim = hidden_size;
        }

        // Quantile embedding layer (linear layer after cosine features)
        // Input: embed_dim cosine features, output: hidden dimension
        let _ = candle_nn::linear(
            embed_dim,
            *self.hidden_sizes.last().unwrap(),
            vb.pp(format!("{}.phi", prefix)),
        )?;

        // Output network (after combining state and quantile embeddings)
        let combined_dim = *self.hidden_sizes.last().unwrap();
        let _ = candle_nn::linear(
            combined_dim,
            combined_dim,
            vb.pp(format!("{}.combine", prefix)),
        )?;
        let _ = candle_nn::linear(
            combined_dim,
            self.num_actions,
            vb.pp(format!("{}.output", prefix)),
        )?;

        Ok(())
    }

    /// Compute cosine embedding for quantile fractions.
    ///
    /// Uses the cosine basis: cos(pi * i * tau) for i = 0, 1, ..., n-1
    ///
    /// # Arguments
    ///
    /// * `taus` - Quantile fractions [batch_size, num_quantiles]
    ///
    /// # Returns
    ///
    /// Cosine embeddings [batch_size, num_quantiles, embedding_dim]
    fn cosine_embedding(&self, taus: &Tensor) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let embed_dim = self.config.embedding_dim;

        // Create i values: [0, 1, 2, ..., embed_dim-1]
        let i_values: Vec<f32> = (0..embed_dim).map(|i| i as f32).collect();
        let i_tensor = Tensor::from_slice(&i_values, (1, 1, embed_dim), &candle_device)?;

        // Expand taus: [batch, num_quantiles] -> [batch, num_quantiles, 1]
        let taus_expanded = taus.unsqueeze(2)?;

        // cos(pi * i * tau): [batch, num_quantiles, embed_dim]
        let pi = std::f32::consts::PI;
        let angles = (taus_expanded.broadcast_mul(&i_tensor)? * pi as f64)?;
        let cos_features = angles.cos()?;

        Ok(cos_features)
    }

    /// Sample uniform quantile fractions.
    fn sample_taus(&mut self, batch_size: usize, num_quantiles: usize) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let tau_samples: Vec<f32> = (0..batch_size * num_quantiles)
            .map(|_| self.rng.gen_range(0.0..1.0))
            .collect();
        Ok(Tensor::from_slice(&tau_samples, (batch_size, num_quantiles), &candle_device)?)
    }

    /// Forward pass through IQN network (immutable version).
    ///
    /// # Arguments
    ///
    /// * `obs` - Observations [batch_size, obs_dim]
    /// * `num_quantiles` - Number of quantile samples
    /// * `var_map` - Network weights
    /// * `prefix` - Network prefix ("online" or "target")
    /// * `taus` - Pre-specified quantile fractions
    ///
    /// # Returns
    ///
    /// * Q-values at each quantile: [batch_size, num_quantiles, num_actions]
    fn iqn_forward_with_taus(
        &self,
        obs: &Tensor,
        num_quantiles: usize,
        var_map: &VarMap,
        prefix: &str,
        taus: &Tensor,
    ) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(var_map, DType::F32, &candle_device);
        let batch_size = obs.dim(0)?;

        // ========== State Encoder (psi) ==========
        let mut state_embed = obs.clone();
        for (i, &hidden_size) in self.hidden_sizes.iter().enumerate() {
            let in_dim = if i == 0 {
                self.obs_dim
            } else {
                self.hidden_sizes[i - 1]
            };
            let linear = candle_nn::linear(
                in_dim,
                hidden_size,
                vb.pp(format!("{}.psi.layer_{}", prefix, i)),
            )?;
            state_embed = linear.forward(&state_embed)?;
            state_embed = state_embed.relu()?;
        }
        // state_embed: [batch_size, hidden_dim]

        // ========== Quantile Embedding (phi) ==========
        // Cosine embedding: [batch, num_quantiles, embed_dim]
        let cos_features = self.cosine_embedding(taus)?;

        // Linear projection: [batch, num_quantiles, hidden_dim]
        let phi_linear = candle_nn::linear(
            self.config.embedding_dim,
            *self.hidden_sizes.last().unwrap(),
            vb.pp(format!("{}.phi", prefix)),
        )?;
        // Reshape for linear: [batch * num_quantiles, embed_dim]
        let cos_flat =
            cos_features.reshape((batch_size * num_quantiles, self.config.embedding_dim))?;
        let quantile_embed = phi_linear.forward(&cos_flat)?;
        let quantile_embed = quantile_embed.relu()?;
        // Reshape back: [batch, num_quantiles, hidden_dim]
        let quantile_embed = quantile_embed.reshape((
            batch_size,
            num_quantiles,
            *self.hidden_sizes.last().unwrap(),
        ))?;

        // ========== Combine State and Quantile Embeddings ==========
        // Expand state embedding: [batch, 1, hidden_dim] -> [batch, num_quantiles, hidden_dim]
        let state_expanded = state_embed.unsqueeze(1)?.broadcast_as((
            batch_size,
            num_quantiles,
            *self.hidden_sizes.last().unwrap(),
        ))?;

        // Element-wise multiplication
        let combined = (state_expanded * &quantile_embed)?;

        // ========== Output Network ==========
        let hidden_dim = *self.hidden_sizes.last().unwrap();

        // Flatten for linear layers
        let combined_flat = combined.reshape((batch_size * num_quantiles, hidden_dim))?;

        let combine_linear =
            candle_nn::linear(hidden_dim, hidden_dim, vb.pp(format!("{}.combine", prefix)))?;
        let h = combine_linear.forward(&combined_flat)?;
        let h = h.relu()?;

        let output_linear = candle_nn::linear(
            hidden_dim,
            self.num_actions,
            vb.pp(format!("{}.output", prefix)),
        )?;
        let q_values = output_linear.forward(&h)?;

        // Reshape to [batch, num_quantiles, num_actions]
        let q_values = q_values.reshape((batch_size, num_quantiles, self.num_actions))?;

        Ok(q_values)
    }

    /// Compute Q-values using specified risk measure.
    ///
    /// # Arguments
    ///
    /// * `quantile_values` - Q-values at each quantile [batch, num_quantiles, num_actions]
    /// * `taus` - Quantile fractions [batch, num_quantiles]
    ///
    /// # Returns
    ///
    /// Risk-adjusted Q-values [batch, num_actions]
    fn compute_risk_q_values(&self, quantile_values: &Tensor, taus: &Tensor) -> Result<Tensor> {
        match self.risk_measure {
            RiskMeasure::Mean => {
                // Simple mean over quantiles
                Ok(quantile_values.mean(1)?)
            }
            RiskMeasure::CVaR(alpha) => {
                // CVaR: average over quantiles <= alpha
                let candle_device = self.device.to_candle()?;
                let _batch_size = quantile_values.dim(0)?;
                let num_quantiles = quantile_values.dim(1)?;

                // Create mask for taus <= alpha
                let alpha_tensor = Tensor::new(&[alpha], &candle_device)?;
                let mask = taus.broadcast_le(&alpha_tensor)?;

                // Count valid quantiles per batch element
                let mask_f32 = mask.to_dtype(DType::F32)?;
                let num_valid = mask_f32.sum(1)?.clamp(1.0, num_quantiles as f64)?;

                // Mask and sum quantile values
                let mask_expanded = mask_f32.unsqueeze(2)?;
                let masked_values = (quantile_values * &mask_expanded)?;
                let sum_values = masked_values.sum(1)?;

                // Average by number of valid quantiles
                Ok(sum_values.broadcast_div(&num_valid.unsqueeze(1)?)?)
            }
            RiskMeasure::Optimistic(alpha) => {
                // Optimistic: average over quantiles >= alpha
                let candle_device = self.device.to_candle()?;
                let _batch_size = quantile_values.dim(0)?;
                let num_quantiles = quantile_values.dim(1)?;

                let alpha_tensor = Tensor::new(&[alpha], &candle_device)?;
                let mask = taus.broadcast_ge(&alpha_tensor)?;

                let mask_f32 = mask.to_dtype(DType::F32)?;
                let num_valid = mask_f32.sum(1)?.clamp(1.0, num_quantiles as f64)?;

                let mask_expanded = mask_f32.unsqueeze(2)?;
                let masked_values = (quantile_values * &mask_expanded)?;
                let sum_values = masked_values.sum(1)?;

                Ok(sum_values.broadcast_div(&num_valid.unsqueeze(1)?)?)
            }
        }
    }

    /// Select action using epsilon-greedy policy with risk-adjusted Q-values.
    fn select_action(&mut self, obs: &Tensor, training: bool) -> Result<Tensor> {
        let batch_size = obs.dim(0)?;
        let candle_device = self.device.to_candle()?;

        if training && self.rng.gen::<f32>() < self.epsilon {
            // Random action
            let actions: Vec<f32> = (0..batch_size)
                .map(|_| self.rng.gen_range(0..self.num_actions) as f32)
                .collect();
            Ok(Tensor::from_slice(
                &actions,
                (batch_size, 1),
                &candle_device,
            )?)
        } else {
            // Greedy action based on risk-adjusted Q-values
            let num_quantiles = self.config.num_quantiles_policy;
            let taus = self.sample_taus(batch_size, num_quantiles)?;
            let quantile_values = self.iqn_forward_with_taus(
                obs,
                num_quantiles,
                &self.online_var_map,
                "online",
                &taus,
            )?;

            let q_values = self.compute_risk_q_values(&quantile_values, &taus)?;
            let actions = q_values.argmax(1)?.unsqueeze(1)?.to_dtype(DType::F32)?;
            Ok(actions)
        }
    }

    /// Compute quantile Huber loss for IQN training.
    ///
    /// The quantile Huber loss is:
    /// ```text
    /// rho_tau(u) = |tau - I(u < 0)| * L_kappa(u)
    /// L_kappa(u) = 0.5 * u^2  if |u| <= kappa
    ///            = kappa * (|u| - 0.5 * kappa)  otherwise
    /// ```
    fn quantile_huber_loss(&self, td_errors: &Tensor, taus: &Tensor) -> Result<Tensor> {
        let kappa = self.config.kappa;

        // Huber loss
        let abs_errors = td_errors.abs()?;
        let quadratic = (abs_errors.clamp(0.0, kappa)?.sqr()? * 0.5)?;
        let linear = ((&abs_errors - kappa as f64)?.clamp(0.0, f64::MAX)? * kappa as f64)?;
        let huber = (&quadratic + &linear)?;

        // Quantile weight: |tau - I(delta < 0)|
        let negative_mask = td_errors.lt(0.0)?.to_dtype(DType::F32)?;
        let tau_weight = (taus - &negative_mask)?.abs()?;

        // Quantile Huber loss
        let loss = (tau_weight * huber)?;

        Ok(loss.mean_all()?)
    }

    /// Hard update: copy online network to target network.
    fn hard_update_target(&mut self) -> Result<()> {
        let online_data = self.online_var_map.data().lock().unwrap();
        let mut target_data = self.target_var_map.data().lock().unwrap();

        for (name, online_var) in online_data.iter() {
            let target_name = name.replace("online", "target");
            if let Some(target_var) = target_data.get_mut(&target_name) {
                target_var.set(online_var.as_tensor())?;
            }
        }

        Ok(())
    }

    /// Soft update: Polyak averaging of online to target network.
    fn soft_update_target(&mut self) -> Result<()> {
        let tau = self.config.tau;
        let online_data = self.online_var_map.data().lock().unwrap();
        let mut target_data = self.target_var_map.data().lock().unwrap();

        for (name, online_var) in online_data.iter() {
            let target_name = name.replace("online", "target");
            if let Some(target_var) = target_data.get_mut(&target_name) {
                let new_val = ((online_var.as_tensor() * tau as f64)?
                    + (target_var.as_tensor() * (1.0 - tau) as f64)?)?;
                target_var.set(&new_val)?;
            }
        }

        Ok(())
    }

    /// Perform a single training update.
    fn update(&mut self) -> Result<(f32, f32)> {
        if !self.replay_buffer.can_sample(self.config.batch_size) {
            return Ok((0.0, 0.0));
        }

        let batch = self.replay_buffer.sample(self.config.batch_size)?;
        let batch_size = batch.observations.dim(0)?;
        let _candle_device = self.device.to_candle()?;

        let num_quantiles = self.config.num_quantiles;
        let num_quantiles_target = self.config.num_quantiles_target;
        let num_quantiles_policy = self.config.num_quantiles_policy;

        // Sample taus for all forward passes
        let target_taus = self.sample_taus(batch_size, num_quantiles_target)?;
        let online_taus = self.sample_taus(batch_size, num_quantiles_policy)?;
        let current_taus = self.sample_taus(batch_size, num_quantiles)?;

        // ========== Compute Target Q-values ==========
        // Get target quantile values
        let target_quantile_values = self.iqn_forward_with_taus(
            &batch.next_observations,
            num_quantiles_target,
            &self.target_var_map,
            "target",
            &target_taus,
        )?;

        // Use online network to select best action (Double DQN style)
        let online_next_values = self.iqn_forward_with_taus(
            &batch.next_observations,
            num_quantiles_policy,
            &self.online_var_map,
            "online",
            &online_taus,
        )?;
        let next_q_values = self.compute_risk_q_values(&online_next_values, &online_taus)?;
        let best_actions = next_q_values.argmax(1)?;

        // Gather target values for best actions: [batch, num_quantiles_target]
        let best_actions_expanded = best_actions
            .unsqueeze(1)?
            .unsqueeze(2)?
            .broadcast_as((batch_size, num_quantiles_target, 1))?
            .to_dtype(DType::I64)?;

        let target_values_selected = target_quantile_values
            .gather(&best_actions_expanded, 2)?
            .squeeze(2)?;

        // Compute TD target: r + gamma * (1 - done) * Z_target
        let not_done = (Tensor::ones_like(&batch.dones)? - &batch.dones)?;
        let not_done_expanded = not_done
            .unsqueeze(1)?
            .broadcast_as((batch_size, num_quantiles_target))?;
        let rewards_expanded = batch
            .rewards
            .unsqueeze(1)?
            .broadcast_as((batch_size, num_quantiles_target))?;

        let td_targets = (&rewards_expanded
            + (&target_values_selected * self.config.gamma as f64)? * &not_done_expanded)?
            .detach();

        // ========== Compute Current Quantile Values ==========
        let current_quantile_values = self.iqn_forward_with_taus(
            &batch.observations,
            num_quantiles,
            &self.online_var_map,
            "online",
            &current_taus,
        )?;

        // Gather values for taken actions: [batch, num_quantiles]
        let actions_i64 = batch.actions.squeeze(1)?.to_dtype(DType::I64)?;
        let actions_expanded =
            actions_i64
                .unsqueeze(1)?
                .unsqueeze(2)?
                .broadcast_as((batch_size, num_quantiles, 1))?;

        let current_values_selected = current_quantile_values
            .gather(&actions_expanded, 2)?
            .squeeze(2)?;

        // ========== Compute Quantile Huber Loss ==========
        // TD errors: [batch, num_quantiles, num_quantiles_target]
        // Each online quantile compared against each target quantile
        let current_expanded = current_values_selected.unsqueeze(2)?;
        let target_expanded = td_targets.unsqueeze(1)?;
        let td_errors = current_expanded.broadcast_sub(&target_expanded)?;

        // Expand taus for loss computation: [batch, num_quantiles, 1]
        let taus_expanded = current_taus.unsqueeze(2)?;

        // Compute loss
        let loss =
            self.quantile_huber_loss(&td_errors, &taus_expanded.broadcast_as(td_errors.dims())?)?;

        // ========== Backward Pass ==========
        let params = ParamsAdamW {
            lr: self.current_lr as f64,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(self.online_var_map.all_vars(), params)?;
        optimizer.backward_step(&loss)?;

        // Update priorities if using PER
        if self.config.prioritized_replay {
            let mean_td_errors: Vec<f32> = td_errors.abs()?.mean((1, 2))?.to_vec1()?;
            self.replay_buffer
                .update_priorities(&batch.indices, &mean_td_errors);
        }

        let loss_val: f32 = loss.to_scalar()?;
        let mean_q: f32 = current_values_selected.mean_all()?.to_scalar()?;

        Ok((loss_val, mean_q))
    }

    /// Update exploration rate (epsilon decay).
    fn update_epsilon(&mut self) {
        self.epsilon = (self.epsilon - self.config.epsilon_decay).max(self.config.epsilon_end);
    }

    /// Collect experience and train the agent.
    ///
    /// # Arguments
    ///
    /// * `total_timesteps` - Total number of timesteps to train
    /// * `callback` - Callback function called with metrics periodically
    pub fn train<F>(&mut self, total_timesteps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!("Starting IQN training for {} timesteps", total_timesteps);

        let num_envs = self.env.num_envs();
        let mut obs = self.env.reset(&self.device)?;

        let mut episode_rewards: Vec<f32> = Vec::new();
        let mut episode_lengths: Vec<usize> = Vec::new();
        let mut current_rewards = vec![0.0f32; num_envs];
        let mut current_lengths = vec![0usize; num_envs];

        let mut total_loss = 0.0f32;
        let mut total_q = 0.0f32;
        let mut update_count = 0usize;

        while self.total_timesteps < total_timesteps {
            // Select actions
            let actions = self.select_action(&obs, true)?;

            // Step environment
            let step_result = self.env.step(&actions, &self.device)?;

            // Store transitions
            let obs_vec: Vec<Vec<f32>> = self.tensor_to_batch_vecs(&obs)?;
            let action_vec: Vec<Vec<f32>> = self.tensor_to_batch_vecs(&actions)?;
            let next_obs_vec: Vec<Vec<f32>> =
                self.tensor_to_batch_vecs(&step_result.observations)?;
            let rewards_vec: Vec<f32> = step_result.rewards.to_vec1()?;
            let dones_vec: Vec<f32> = step_result.dones()?.to_vec1()?;

            for i in 0..num_envs {
                self.replay_buffer.add(
                    &obs_vec[i],
                    &action_vec[i],
                    rewards_vec[i],
                    &next_obs_vec[i],
                    dones_vec[i] > 0.5,
                );

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
            self.total_timesteps += num_envs;

            // Training updates
            if self.total_timesteps >= self.config.learning_starts
                && self.total_timesteps % self.config.train_freq == 0
            {
                for _ in 0..self.config.gradient_steps {
                    let (loss, mean_q) = self.update()?;
                    total_loss += loss;
                    total_q += mean_q;
                    update_count += 1;
                }
            }

            // Update target network
            self.steps_since_target_update += num_envs;
            if self.steps_since_target_update >= self.config.target_update_interval {
                if self.config.tau < 1.0 {
                    self.soft_update_target()?;
                } else {
                    self.hard_update_target()?;
                }
                self.steps_since_target_update = 0;
            }

            // Update epsilon
            self.update_epsilon();

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
                        total_loss / update_count as f32
                    } else {
                        0.0
                    },
                    value_loss: if update_count > 0 {
                        total_q / update_count as f32
                    } else {
                        0.0
                    },
                    entropy: self.epsilon,
                    approx_kl: 0.0,
                    clip_fraction: 0.0,
                    explained_variance: 0.0,
                    learning_rate: self.current_lr,
                };

                info!(
                    "Step {}: reward={:.2}, epsilon={:.3}, loss={:.4}, mean_q={:.2}",
                    self.total_timesteps,
                    metrics.mean_reward,
                    self.epsilon,
                    metrics.policy_loss,
                    metrics.value_loss
                );

                callback(&metrics);

                // Reset accumulators
                episode_rewards.clear();
                total_loss = 0.0;
                total_q = 0.0;
                update_count = 0;
            }
        }

        info!("IQN training completed: {} timesteps", self.total_timesteps);
        Ok(())
    }

    /// Convert batch tensor to Vec<Vec<f32>>.
    fn tensor_to_batch_vecs(&self, tensor: &Tensor) -> Result<Vec<Vec<f32>>> {
        let batch_size = tensor.dim(0)?;
        let flat: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        let dim = flat.len() / batch_size;

        Ok((0..batch_size)
            .map(|i| flat[i * dim..(i + 1) * dim].to_vec())
            .collect())
    }

    /// Predict action for given observation (inference).
    ///
    /// # Arguments
    ///
    /// * `obs` - Observation tensor
    /// * `deterministic` - If true, always take greedy action
    pub fn predict(&mut self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        let batch_size = obs.dim(0)?;
        let num_quantiles = self.config.num_quantiles_policy;
        let taus = self.sample_taus(batch_size, num_quantiles)?;

        if deterministic {
            let quantile_values = self.iqn_forward_with_taus(
                obs,
                num_quantiles,
                &self.online_var_map,
                "online",
                &taus,
            )?;
            let q_values = self.compute_risk_q_values(&quantile_values, &taus)?;
            Ok(q_values.argmax(1)?.unsqueeze(1)?.to_dtype(DType::F32)?)
        } else {
            self.select_action(obs, false)
        }
    }

    /// Set the risk measure for action selection.
    pub fn set_risk_measure(&mut self, risk_measure: RiskMeasure) {
        self.risk_measure = risk_measure;
    }

    /// Get current risk measure.
    pub fn risk_measure(&self) -> RiskMeasure {
        self.risk_measure
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for IQNAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let (loss, mean_q) = self.update()?;

        Ok(TrainMetrics {
            policy_loss: loss,
            value_loss: mean_q,
            entropy: self.epsilon,
            timesteps: self.total_timesteps,
            ..Default::default()
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let data = self.online_var_map.data().lock().unwrap();
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        for (name, var) in data.iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&tensors, path)?;

        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("IQN model saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        let mut data = self.online_var_map.data().lock().unwrap();
        for (name, tensor) in tensors {
            if let Some(var) = data.get_mut(&name) {
                var.set(&tensor)?;
            }
        }
        drop(data);

        self.hard_update_target()?;

        info!("IQN model loaded from {:?}", path);
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
    use crate::algorithms::config::IQNConfig;

    #[test]
    fn test_iqn_config_defaults() {
        let config = IQNConfig::default();
        assert_eq!(config.num_quantiles, 64);
        assert_eq!(config.embedding_dim, 64);
        assert!((config.kappa - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_iqn_config_builder() {
        let config = IQNConfig::new()
            .num_quantiles(32)
            .embedding_dim(128)
            .kappa(0.5);

        assert_eq!(config.num_quantiles, 32);
        assert_eq!(config.embedding_dim, 128);
        assert!((config.kappa - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_risk_measure_default() {
        let risk = RiskMeasure::default();
        assert_eq!(risk, RiskMeasure::Mean);
    }
}
