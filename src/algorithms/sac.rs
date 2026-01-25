//! Soft Actor-Critic (SAC) implementation.
//!
//! SAC is an off-policy algorithm for continuous action spaces that uses:
//! - Maximum entropy reinforcement learning
//! - Twin Q-networks for reduced overestimation
//! - Automatic temperature (entropy coefficient) tuning
//! - Reparameterization trick for policy gradient
//!
//! Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL" (2018)
//! Reference: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications" (2019)

use crate::algorithms::config::SACConfig;
use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::traits::RLAlgorithm;
use crate::buffer::{ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, Result, RocketError};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use std::path::Path;
use tracing::info;

/// SAC Agent for continuous action spaces.
pub struct SACAgent<E: Environment + Clone + 'static> {
    /// Algorithm configuration.
    config: SACConfig,
    /// Vectorized environment.
    env: VecEnv<E>,
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

    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    action_dim: usize,
    /// Action scale (for tanh squashing).
    action_scale: f32,
    /// Action bias.
    action_bias: f32,

    /// Experience replay buffer.
    replay_buffer: ReplayBuffer,

    /// Log alpha (entropy coefficient).
    log_alpha: Tensor,
    /// Target entropy.
    target_entropy: f32,
    /// Current alpha value.
    alpha: f32,

    /// Total timesteps trained.
    total_timesteps: usize,

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> SACAgent<E> {
    /// Create a new SAC agent.
    pub fn new(config: SACConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(RocketError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();

        let obs_dim = obs_space.flat_dim();
        let action_dim = act_space.flat_dim();

        // Assume normalized action space [-1, 1]
        let action_scale = 1.0;
        let action_bias = 0.0;

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Create replay buffer
        let buffer_config = ReplayBufferConfig::new(obs_dim, action_dim)
            .capacity(config.buffer_size);
        let replay_buffer = ReplayBuffer::new(buffer_config, device)?;

        // Target entropy: -dim(A) by default
        let target_entropy = config.target_entropy.unwrap_or(-(action_dim as f32));

        // Initialize log_alpha
        let candle_device = device.to_candle()?;
        let initial_alpha = config.ent_coef.ln();
        let log_alpha = Tensor::new(&[initial_alpha], &candle_device)?;

        let policy_var_map = VarMap::new();
        let q1_var_map = VarMap::new();
        let q2_var_map = VarMap::new();
        let target_q1_var_map = VarMap::new();
        let target_q2_var_map = VarMap::new();

        let mut agent = Self {
            config: config.clone(),
            env,
            device,
            policy_var_map,
            q1_var_map,
            q2_var_map,
            target_q1_var_map,
            target_q2_var_map,
            obs_dim,
            action_dim,
            action_scale,
            action_bias,
            replay_buffer,
            log_alpha,
            target_entropy,
            alpha: config.ent_coef,
            total_timesteps: 0,
            rng,
        };

        agent.init_networks()?;

        info!(
            "SAC Agent initialized: obs_dim={}, action_dim={}, auto_entropy={}",
            obs_dim, action_dim, config.auto_entropy_tuning
        );

        Ok(agent)
    }

    /// Initialize all networks.
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

    /// Initialize policy network (outputs mean and log_std).
    fn init_policy_network(&self, candle_device: &candle_core::Device) -> Result<()> {
        let vb = VarBuilder::from_varmap(&self.policy_var_map, DType::F32, candle_device);

        let mut in_dim = self.obs_dim;
        for (i, &hidden_size) in self.config.policy_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("policy.layer_{}", i)))?;
            in_dim = hidden_size;
        }

        // Mean output
        let _ = candle_nn::linear(in_dim, self.action_dim, vb.pp("policy.mean"))?;
        // Log std output
        let _ = candle_nn::linear(in_dim, self.action_dim, vb.pp("policy.log_std"))?;

        Ok(())
    }

    /// Initialize Q-network (takes obs and action as input).
    fn init_q_network(
        &self,
        var_map: &VarMap,
        prefix: &str,
        candle_device: &candle_core::Device,
    ) -> Result<()> {
        let vb = VarBuilder::from_varmap(var_map, DType::F32, candle_device);

        let mut in_dim = self.obs_dim + self.action_dim;
        for (i, &hidden_size) in self.config.q_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("{}.layer_{}", prefix, i)))?;
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
            let in_dim = if i == 0 { self.obs_dim } else { self.config.policy_hidden_sizes[i - 1] };
            let linear = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("policy.layer_{}", i)))?;
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

    /// Sample action using reparameterization trick.
    fn sample_action(&self, obs: &Tensor, deterministic: bool) -> Result<(Tensor, Tensor)> {
        let (mean, log_std) = self.policy_forward(obs)?;

        if deterministic {
            // Deterministic action (tanh of mean)
            let action = mean.tanh()?;
            let log_prob = Tensor::zeros_like(&mean.narrow(1, 0, 1)?.squeeze(1)?)?;
            return Ok((action, log_prob));
        }

        let std = log_std.exp()?;

        // Sample from Gaussian using reparameterization
        let noise = Tensor::randn_like(&mean, 0.0, 1.0)?;
        let x_t = (&mean + &noise * &std)?; // Pre-tanh action

        // Apply tanh squashing
        let action = x_t.tanh()?;

        // Compute log probability with correction for tanh squashing
        // log_prob = sum(log(N(x|mu,sigma))) - sum(log(1 - tanh(x)^2))
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

        // log_prob = -0.5 * (x - mu)^2 / sigma^2 - log(sigma) - 0.5 * log(2*pi)
        let log_prob_per_dim = (normalized.sqr()? * (-0.5))?
            .broadcast_sub(log_std)?
            .broadcast_sub(&log_2pi_tensor)?;

        Ok(log_prob_per_dim.sum(1)?)
    }

    /// Forward pass through Q-network.
    fn q_forward(&self, obs: &Tensor, action: &Tensor, var_map: &VarMap, prefix: &str) -> Result<Tensor> {
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
            let linear = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("{}.layer_{}", prefix, i)))?;
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

    /// Hard update: copy Q-networks to target networks.
    fn hard_update_targets(&mut self) -> Result<()> {
        Self::copy_weights(&self.q1_var_map, &self.target_q1_var_map, "q1", "target_q1")?;
        Self::copy_weights(&self.q2_var_map, &self.target_q2_var_map, "q2", "target_q2")?;
        Ok(())
    }

    /// Soft update: polyak averaging.
    fn soft_update_targets(&mut self) -> Result<()> {
        let tau = self.config.tau;
        Self::polyak_update(&self.q1_var_map, &self.target_q1_var_map, "q1", "target_q1", tau)?;
        Self::polyak_update(&self.q2_var_map, &self.target_q2_var_map, "q2", "target_q2", tau)?;
        Ok(())
    }

    /// Copy weights from source to target var_map.
    fn copy_weights(
        src: &VarMap,
        dst: &VarMap,
        src_prefix: &str,
        dst_prefix: &str,
    ) -> Result<()> {
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

        // ========== Update Q-networks ==========
        // Compute target Q-value
        let (next_actions, next_log_probs) = self.sample_action(&batch.next_observations, false)?;

        let target_q1 = self.q_forward(&batch.next_observations, &next_actions, &self.target_q1_var_map, "target_q1")?;
        let target_q2 = self.q_forward(&batch.next_observations, &next_actions, &self.target_q2_var_map, "target_q2")?;
        let target_q = target_q1.minimum(&target_q2)?;

        // Target: r + gamma * (1 - done) * (min_Q - alpha * log_pi)
        let not_done = (Tensor::ones_like(&batch.dones)? - &batch.dones)?;
        let entropy_term = (&next_log_probs * self.alpha as f64)?;
        let soft_q_target = ((&target_q - &entropy_term)? * self.config.gamma as f64)?;
        let td_target = (&batch.rewards + &soft_q_target * &not_done)?.detach();

        // Q1 loss
        let current_q1 = self.q_forward(&batch.observations, &batch.actions, &self.q1_var_map, "q1")?;
        let q1_loss = (&current_q1 - &td_target)?.sqr()?.mean_all()?;

        // Q2 loss
        let current_q2 = self.q_forward(&batch.observations, &batch.actions, &self.q2_var_map, "q2")?;
        let q2_loss = (&current_q2 - &td_target)?.sqr()?.mean_all()?;

        // Update Q1
        let params = ParamsAdamW {
            lr: self.config.learning_rate as f64,
            ..Default::default()
        };
        let mut q1_optimizer = AdamW::new(self.q1_var_map.all_vars(), params.clone())?;
        q1_optimizer.backward_step(&q1_loss)?;

        // Update Q2
        let mut q2_optimizer = AdamW::new(self.q2_var_map.all_vars(), params.clone())?;
        q2_optimizer.backward_step(&q2_loss)?;

        // ========== Update Policy ==========
        let (new_actions, new_log_probs) = self.sample_action(&batch.observations, false)?;
        let q1_new = self.q_forward(&batch.observations, &new_actions, &self.q1_var_map, "q1")?;
        let q2_new = self.q_forward(&batch.observations, &new_actions, &self.q2_var_map, "q2")?;
        let min_q_new = q1_new.minimum(&q2_new)?;

        // Policy loss: maximize Q - alpha * log_pi
        let entropy_term = (&new_log_probs * self.alpha as f64)?;
        let policy_loss = (&entropy_term - &min_q_new)?.mean_all()?;

        let mut policy_optimizer = AdamW::new(self.policy_var_map.all_vars(), params.clone())?;
        policy_optimizer.backward_step(&policy_loss)?;

        // ========== Update Alpha (if auto-tuning) ==========
        let alpha_loss_val = if self.config.auto_entropy_tuning {
            let candle_device = self.device.to_candle()?;
            let log_pi_detached = new_log_probs.detach();

            // Compute alpha loss: alpha * (-log_pi - target_entropy)
            let neg_log_pi = log_pi_detached.neg()?;
            let diff = neg_log_pi.mean_all()?.to_scalar::<f32>()? - self.target_entropy;

            // Simple gradient step for alpha (manual update since log_alpha is not in VarMap)
            let alpha_grad = -diff * self.alpha;
            let new_log_alpha = self.log_alpha.to_scalar::<f32>()? - self.config.learning_rate * alpha_grad;
            self.log_alpha = Tensor::new(&[new_log_alpha], &candle_device)?;
            self.alpha = new_log_alpha.exp();

            (self.alpha * diff).abs()
        } else {
            0.0
        };

        // Soft update targets
        self.soft_update_targets()?;

        let q_loss: f32 = (q1_loss.to_scalar::<f32>()? + q2_loss.to_scalar::<f32>()?) / 2.0;
        let policy_loss_val: f32 = policy_loss.to_scalar()?;

        Ok((q_loss, policy_loss_val, self.alpha, alpha_loss_val))
    }

    /// Train the agent.
    pub fn train<F>(&mut self, total_timesteps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!("Starting SAC training for {} timesteps", total_timesteps);

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

            // Training updates
            if self.total_timesteps >= self.config.learning_starts {
                for _ in 0..self.config.gradient_steps {
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
                    policy_loss: if update_count > 0 { total_policy_loss / update_count as f32 } else { 0.0 },
                    value_loss: if update_count > 0 { total_q_loss / update_count as f32 } else { 0.0 },
                    entropy: self.alpha,
                    approx_kl: 0.0,
                    clip_fraction: 0.0,
                    explained_variance: 0.0,
                    learning_rate: self.config.learning_rate,
                };

                info!(
                    "Step {}: reward={:.2}, alpha={:.4}, q_loss={:.4}, pi_loss={:.4}",
                    self.total_timesteps,
                    metrics.mean_reward,
                    self.alpha,
                    metrics.value_loss,
                    metrics.policy_loss
                );

                callback(&metrics);

                episode_rewards.clear();
                total_q_loss = 0.0;
                total_policy_loss = 0.0;
                update_count = 0;
            }
        }

        info!("SAC training completed: {} timesteps", self.total_timesteps);
        Ok(())
    }

    /// Predict action for given observation.
    pub fn predict(&self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        let (action, _) = self.sample_action(obs, deterministic)?;
        Ok(action)
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for SACAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let (q_loss, policy_loss, alpha, _) = self.update()?;

        Ok(TrainMetrics {
            policy_loss,
            value_loss: q_loss,
            entropy: alpha,
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

        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("SAC model saved to {:?}", path);
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

        info!("SAC model loaded from {:?}", path);
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
    fn test_sac_config_defaults() {
        let config = SACConfig::default();
        assert!(config.auto_entropy_tuning);
        assert!((config.gamma - 0.99).abs() < 1e-6);
        assert!((config.tau - 0.005).abs() < 1e-6);
    }
}
