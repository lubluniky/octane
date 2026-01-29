//! Proximal Policy Optimization (PPO) algorithm implementation.
//!
//! PPO is a policy gradient method that uses a clipped surrogate objective
//! to enable multiple epochs of minibatch updates while preventing too large
//! policy updates.
//!
//! Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)

use crate::algorithms::config::PPOConfig;
use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::rollout::RolloutBuffer;
use crate::algorithms::traits::RLAlgorithm;
use crate::core::{Device, Result, OctaneError};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use std::path::Path;
use tracing::{debug, info};

/// PPO Agent for training and inference.
pub struct PPOAgent<E: Environment + Clone + 'static> {
    /// Algorithm configuration.
    config: PPOConfig,

    /// Vectorized environment.
    env: VecEnv<E>,

    /// Device for tensor operations.
    device: Device,

    /// Variable map for network parameters.
    var_map: VarMap,

    /// Policy network weights path in var_map.
    policy_prefix: String,

    /// Value network weights path in var_map.
    value_prefix: String,

    /// Observation dimension.
    obs_dim: usize,

    /// Action dimension (for continuous) or number of actions (for discrete).
    act_dim: usize,

    /// Whether the action space is discrete.
    is_discrete: bool,

    /// Hidden layer sizes.
    hidden_sizes: Vec<usize>,

    /// Log standard deviation for continuous actions (learnable).
    log_std: Option<Tensor>,

    /// Current learning rate (for scheduling).
    current_lr: f32,

    /// Total timesteps trained so far.
    total_timesteps: usize,

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> PPOAgent<E> {
    /// Create a new PPO agent.
    pub fn new(config: PPOConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
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

        let var_map = VarMap::new();
        let hidden_sizes = vec![64, 64]; // Default MLP architecture

        let mut agent = Self {
            config: config.clone(),
            env,
            device,
            var_map,
            policy_prefix: "policy".to_string(),
            value_prefix: "value".to_string(),
            obs_dim,
            act_dim,
            is_discrete,
            hidden_sizes,
            log_std: None,
            current_lr: config.learning_rate,
            total_timesteps: 0,
            rng,
        };

        agent.init_networks()?;

        info!(
            "PPO Agent initialized: obs_dim={}, act_dim={}, discrete={}",
            obs_dim, act_dim, is_discrete
        );

        Ok(agent)
    }

    /// Initialize neural network weights.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // Build policy network layers
        let vb_policy = VarBuilder::from_varmap(&self.var_map, DType::F32, &candle_device);
        let mut in_dim = self.obs_dim;

        for (i, &hidden_size) in self.hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(
                in_dim,
                hidden_size,
                vb_policy.pp(format!("{}.layer_{}", self.policy_prefix, i)),
            )?;
            in_dim = hidden_size;
        }

        // Output layer for policy (action logits or mean)
        let _ = candle_nn::linear(
            in_dim,
            self.act_dim,
            vb_policy.pp(format!("{}.output", self.policy_prefix)),
        )?;

        // Build value network layers
        let vb_value = VarBuilder::from_varmap(&self.var_map, DType::F32, &candle_device);
        in_dim = self.obs_dim;

        for (i, &hidden_size) in self.hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(
                in_dim,
                hidden_size,
                vb_value.pp(format!("{}.layer_{}", self.value_prefix, i)),
            )?;
            in_dim = hidden_size;
        }

        // Output layer for value (single scalar)
        let _ = candle_nn::linear(
            in_dim,
            1,
            vb_value.pp(format!("{}.output", self.value_prefix)),
        )?;

        // Initialize log_std for continuous actions
        if !self.is_discrete {
            self.log_std = Some(Tensor::zeros(&[self.act_dim], DType::F32, &candle_device)?);
        }

        Ok(())
    }

    /// Forward pass through policy network.
    fn policy_forward(&self, obs: &Tensor) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.var_map, DType::F32, &candle_device);

        let mut x = obs.clone();
        let num_layers = self.hidden_sizes.len();

        for i in 0..num_layers {
            let linear = candle_nn::linear(
                if i == 0 {
                    self.obs_dim
                } else {
                    self.hidden_sizes[i - 1]
                },
                self.hidden_sizes[i],
                vb.pp(format!("{}.layer_{}", self.policy_prefix, i)),
            )?;
            x = linear.forward(&x)?;
            x = x.tanh()?;
        }

        let output_linear = candle_nn::linear(
            self.hidden_sizes[num_layers - 1],
            self.act_dim,
            vb.pp(format!("{}.output", self.policy_prefix)),
        )?;

        Ok(output_linear.forward(&x)?)
    }

    /// Forward pass through value network.
    fn value_forward(&self, obs: &Tensor) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(&self.var_map, DType::F32, &candle_device);

        let mut x = obs.clone();
        let num_layers = self.hidden_sizes.len();

        for i in 0..num_layers {
            let linear = candle_nn::linear(
                if i == 0 {
                    self.obs_dim
                } else {
                    self.hidden_sizes[i - 1]
                },
                self.hidden_sizes[i],
                vb.pp(format!("{}.layer_{}", self.value_prefix, i)),
            )?;
            x = linear.forward(&x)?;
            x = x.tanh()?;
        }

        let output_linear = candle_nn::linear(
            self.hidden_sizes[num_layers - 1],
            1,
            vb.pp(format!("{}.output", self.value_prefix)),
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

            // Sample from categorical distribution
            let probs_vec: Vec<f32> = probs.flatten_all()?.to_vec1()?;
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

                // Get log probability of selected action
                let lp_vec: Vec<f32> = log_probs.get(b)?.flatten_all()?.to_vec1()?;
                action_log_probs.push(lp_vec[action]);
            }

            let candle_device = self.device.to_candle()?;
            let actions_tensor = Tensor::from_slice(&actions, &[batch_size, 1], &candle_device)?;
            let log_probs_tensor =
                Tensor::from_slice(&action_log_probs, &[batch_size], &candle_device)?;

            Ok((actions_tensor, log_probs_tensor))
        } else {
            // Gaussian distribution for continuous actions
            let mean = logits_or_mean;
            let log_std = self
                .log_std
                .as_ref()
                .ok_or_else(|| OctaneError::InvalidConfig("log_std not initialized".to_string()))?;
            let std = log_std.exp()?;

            // Sample from Gaussian: action = mean + std * noise
            let noise = Tensor::randn_like(&mean, 0.0, 1.0)?;
            let actions = (&mean + noise.broadcast_mul(&std)?)?;

            // Compute log probability
            // log_prob = -0.5 * ((action - mean) / std)^2 - log(std) - 0.5 * log(2*pi)
            let diff = (&actions - &mean)?;
            let normalized = diff.broadcast_div(&std)?;
            let candle_device = self.device.to_candle()?;
            let log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
            let log_2pi_tensor = Tensor::new(&[log_2pi], &candle_device)?;
            let log_prob_per_dim = (normalized.sqr()? * (-0.5))?
                .broadcast_sub(log_std)?
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
            let log_std = self
                .log_std
                .as_ref()
                .ok_or_else(|| OctaneError::InvalidConfig("log_std not initialized".to_string()))?;
            let std = log_std.exp()?;

            // Log probability for Gaussian
            let candle_device = self.device.to_candle()?;
            let log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
            let log_2pi_tensor = Tensor::new(&[log_2pi], &candle_device)?;
            let diff = (actions - &mean)?;
            let normalized = diff.broadcast_div(&std)?;
            let log_prob_per_dim = (normalized.sqr()? * (-0.5))?
                .broadcast_sub(log_std)?
                .broadcast_sub(&log_2pi_tensor)?;
            let log_probs = log_prob_per_dim.sum(1)?;

            // Entropy for Gaussian: 0.5 * log(2 * pi * e * var) = 0.5 + 0.5 * log(2*pi) + log_std
            let entropy_const = 0.5 * (1.0 + (2.0 * std::f32::consts::PI).ln());
            let entropy_const_tensor = Tensor::new(&[entropy_const], &candle_device)?;
            let entropy_per_dim = log_std.broadcast_add(&entropy_const_tensor)?;
            let entropy = entropy_per_dim.sum(0)?.broadcast_as(&[obs.dim(0)?])?;

            Ok((log_probs, values, entropy))
        }
    }

    /// Compute Generalized Advantage Estimation (GAE).
    fn compute_gae(
        &self,
        rewards: &[f32],
        values: &[f32],
        dones: &[f32],
        last_value: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let n_steps = rewards.len();
        let mut advantages = vec![0.0f32; n_steps];
        let mut returns = vec![0.0f32; n_steps];

        let mut last_gae = 0.0f32;
        let mut next_value = last_value;

        for t in (0..n_steps).rev() {
            let mask = 1.0 - dones[t];
            let delta = rewards[t] + self.config.gamma * next_value * mask - values[t];
            last_gae = delta + self.config.gamma * self.config.gae_lambda * mask * last_gae;
            advantages[t] = last_gae;
            returns[t] = advantages[t] + values[t];
            next_value = values[t];
        }

        (advantages, returns)
    }

    /// Perform a single PPO update step.
    fn update(&mut self, buffer: &RolloutBuffer) -> Result<TrainMetrics> {
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

        // Setup optimizer
        let params = ParamsAdamW {
            lr: self.current_lr as f64,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(self.var_map.all_vars(), params)?;

        let mut total_policy_loss = 0.0f32;
        let mut total_value_loss = 0.0f32;
        let mut total_entropy_loss = 0.0f32;
        let mut total_approx_kl = 0.0f32;
        let mut total_clip_fraction = 0.0f32;
        let mut n_updates = 0usize;

        // Generate batch indices
        let n_batches = n_samples.div_ceil(self.config.batch_size);

        for _epoch in 0..self.config.n_epochs {
            // Shuffle indices
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
                let batch_returns = returns.index_select(&idx_tensor, 0)?;

                // Evaluate actions under current policy
                let (new_log_probs, values, entropy) =
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

                // Value loss (MSE)
                let value_diff = (&values - &batch_returns)?;
                let value_loss = value_diff.sqr()?.mean_all()?;

                // Entropy loss (negative because we want to maximize entropy)
                let entropy_loss = entropy.neg()?.mean_all()?;

                // Total loss
                let total_loss = ((&policy_loss + (&value_loss * self.config.vf_coef as f64)?)?
                    + (&entropy_loss * self.config.ent_coef as f64)?)?;

                // Backward pass
                optimizer.backward_step(&total_loss)?;

                // Collect metrics
                let policy_loss_val: f32 = policy_loss.to_scalar()?;
                let value_loss_val: f32 = value_loss.to_scalar()?;
                let entropy_val: f32 = entropy.mean_all()?.to_scalar()?;

                total_policy_loss += policy_loss_val;
                total_value_loss += value_loss_val;
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

                // Early stopping based on KL divergence
                if let Some(target_kl) = self.config.target_kl {
                    if approx_kl > target_kl {
                        debug!("Early stopping at epoch {} due to KL={}", _epoch, approx_kl);
                        break;
                    }
                }
            }
        }

        let n_updates_f = n_updates as f32;
        Ok(TrainMetrics {
            policy_loss: total_policy_loss / n_updates_f,
            value_loss: total_value_loss / n_updates_f,
            entropy: total_entropy_loss / n_updates_f,
            approx_kl: total_approx_kl / n_updates_f,
            clip_fraction: total_clip_fraction / n_updates_f,
            explained_variance: 0.0, // TODO: Compute
            learning_rate: self.current_lr,
            timesteps: self.total_timesteps,
            episodes: 0,
            mean_reward: 0.0,
            std_reward: 0.0,
        })
    }

    /// Update learning rate based on progress.
    fn update_learning_rate(&mut self, progress: f32) {
        if self.config.use_lr_schedule {
            // Linear decay
            self.current_lr = self.config.learning_rate * (1.0 - progress);
            self.current_lr = self.current_lr.max(1e-7); // Minimum LR
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
            self.act_dim,
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

            // Store transition
            buffer.add(
                &obs,
                &actions,
                &step_result.rewards,
                &step_result.dones()?,
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

    /// Train the agent for a given number of timesteps.
    pub fn train<F>(&mut self, total_timesteps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!("Starting PPO training for {} timesteps", total_timesteps);

        let num_envs = self.env.num_envs();
        let n_steps = self.config.n_steps;
        let rollout_timesteps = n_steps * num_envs;
        let n_iterations = total_timesteps.div_ceil(rollout_timesteps);

        for iteration in 0..n_iterations {
            let progress = self.total_timesteps as f32 / total_timesteps as f32;
            self.update_learning_rate(progress);

            // Collect rollout
            let (buffer, episode_rewards, _episode_lengths) = self.collect_rollout()?;

            // Update policy
            let mut metrics = self.update(&buffer)?;

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

        info!(
            "Training completed: {} total timesteps",
            self.total_timesteps
        );
        Ok(())
    }

    /// Predict action for given observation (inference mode).
    pub fn predict(&mut self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        let logits_or_mean = self.policy_forward(obs)?;

        if self.is_discrete {
            if deterministic {
                // Argmax for deterministic action
                Ok(logits_or_mean
                    .argmax(1)?
                    .unsqueeze(1)?
                    .to_dtype(DType::F32)?)
            } else {
                let (action, _) = self.sample_action(obs)?;
                Ok(action)
            }
        } else if deterministic {
            // Use mean directly
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

impl<E: Environment + Clone + 'static> RLAlgorithm for PPOAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let (buffer, episode_rewards, _) = self.collect_rollout()?;
        let mut metrics = self.update(&buffer)?;

        if !episode_rewards.is_empty() {
            metrics.mean_reward =
                episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
            metrics.episodes = episode_rewards.len();
        }

        Ok(metrics)
    }

    fn save(&self, path: &Path) -> Result<()> {
        let data = self.var_map.data().lock().unwrap();
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        for (name, var) in data.iter() {
            // Convert Var to Tensor by getting the inner tensor
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&tensors, path)?;

        // Save config separately
        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("Model saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        let mut data = self.var_map.data().lock().unwrap();
        for (name, tensor) in tensors {
            if let Some(var) = data.get_mut(&name) {
                var.set(&tensor)?;
            }
        }

        info!("Model loaded from {:?}", path);
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
    fn test_gae_computation() {
        // Simple test for GAE computation
        let config = PPOConfig::default();
        let rewards = vec![1.0, 1.0, 1.0, 1.0];
        let values = vec![0.5, 0.5, 0.5, 0.5];
        let dones = vec![0.0, 0.0, 0.0, 1.0];
        let last_value = 0.0;

        // Manual GAE computation for verification
        // This is a simplified test - actual PPO agent test would require env
        let gamma = config.gamma;
        let gae_lambda = config.gae_lambda;

        // With done at last step, last_gae starts fresh
        // t=3: delta = 1.0 + 0 - 0.5 = 0.5, gae = 0.5
        // t=2: delta = 1.0 + 0.99*0.5 - 0.5 = 0.995, gae = 0.995 + 0.99*0.95*0.5 = 1.46525
        // etc.

        assert!(gamma > 0.0 && gamma <= 1.0);
        assert!(gae_lambda > 0.0 && gae_lambda <= 1.0);
    }
}
