//! Advantage Actor-Critic (A2C) algorithm implementation.
//!
//! A2C is a synchronous, deterministic variant of A3C (Asynchronous Advantage Actor-Critic).
//! It uses the advantage function to reduce variance in policy gradient estimation.
//!
//! A2C serves as a simpler baseline compared to PPO, without the clipped objective,
//! but still benefits from the actor-critic architecture and advantage estimation.
//!
//! Reference: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)

use crate::algorithms::config::A2CConfig;
use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::rollout::RolloutBuffer;
use crate::algorithms::traits::RLAlgorithm;
use crate::core::{Device, Result, RocketError};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{Optimizer, VarBuilder, VarMap, AdamW, ParamsAdamW};
use rand::prelude::*;
use std::path::Path;
use tracing::info;

/// A2C Agent for training and inference.
pub struct A2CAgent<E: Environment + Clone + 'static> {
    /// Algorithm configuration.
    config: A2CConfig,

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

    /// Total timesteps trained so far.
    total_timesteps: usize,

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> A2CAgent<E> {
    /// Create a new A2C agent.
    pub fn new(config: A2CConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(RocketError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();

        let obs_dim = obs_space.flat_dim();
        let act_dim = act_space.flat_dim();
        let is_discrete = act_space.shape() == &[1];

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
            total_timesteps: 0,
            rng,
        };

        agent.init_networks()?;

        info!(
            "A2C Agent initialized: obs_dim={}, act_dim={}, discrete={}",
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

        // Build value network layers (shared feature extractor in A2C)
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
                if i == 0 { self.obs_dim } else { self.hidden_sizes[i - 1] },
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
                if i == 0 { self.obs_dim } else { self.hidden_sizes[i - 1] },
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
            let log_probs_tensor = Tensor::from_slice(&action_log_probs, &[batch_size], &candle_device)?;

            Ok((actions_tensor, log_probs_tensor))
        } else {
            // Gaussian distribution for continuous actions
            let mean = logits_or_mean;
            let log_std = self.log_std.as_ref().ok_or_else(|| {
                RocketError::InvalidConfig("log_std not initialized".to_string())
            })?;
            let std = log_std.exp()?;

            // Sample from Gaussian: action = mean + std * noise
            let noise = Tensor::randn_like(&mean, 0.0, 1.0)?;
            let actions = (&mean + noise.broadcast_mul(&std)?)?;

            // Compute log probability
            let candle_device = self.device.to_candle()?;
            let log_2pi = 0.5 * (2.0 * std::f32::consts::PI).ln();
            let log_2pi_tensor = Tensor::new(&[log_2pi], &candle_device)?;
            let diff = (&actions - &mean)?;
            let normalized = diff.broadcast_div(&std)?;
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
            let action_log_probs = log_probs.gather(&actions_i64.unsqueeze(1)?, 1)?.squeeze(1)?;

            // Entropy: -sum(p * log(p))
            let entropy = (probs.neg()? * &log_probs)?.sum(1)?;

            Ok((action_log_probs, values, entropy))
        } else {
            let mean = logits_or_mean;
            let log_std = self.log_std.as_ref().ok_or_else(|| {
                RocketError::InvalidConfig("log_std not initialized".to_string())
            })?;
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

            // Entropy for Gaussian
            let entropy_const = 0.5 * (1.0 + (2.0 * std::f32::consts::PI).ln());
            let entropy_const_tensor = Tensor::new(&[entropy_const], &candle_device)?;
            let entropy_per_dim = log_std.broadcast_add(&entropy_const_tensor)?;
            let entropy = entropy_per_dim.sum(0)?.broadcast_as(&[obs.dim(0)?])?;

            Ok((log_probs, values, entropy))
        }
    }

    /// Perform a single A2C update step.
    ///
    /// Unlike PPO, A2C performs a single update without clipping.
    /// The loss function is:
    /// L = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
    ///
    /// Where:
    /// - policy_loss = -log_prob * advantages
    /// - value_loss = (returns - values)^2
    /// - entropy_loss = -entropy
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

        // Setup optimizer (RMSprop is traditionally used for A2C, but AdamW works well too)
        let params = ParamsAdamW {
            lr: self.config.learning_rate as f64,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(self.var_map.all_vars(), params)?;

        // Evaluate actions under current policy
        let (log_probs, values, entropy) = self.evaluate_actions(&obs_flat, &actions_flat)?;

        // Policy loss: -log_prob * advantages (vanilla policy gradient)
        let policy_loss = (log_probs.neg()? * &advantages)?.mean_all()?;

        // Value loss: MSE between predicted and actual returns
        let value_diff = (&values - &returns)?;
        let value_loss = value_diff.sqr()?.mean_all()?;

        // Entropy loss (negative to maximize entropy)
        let entropy_mean = entropy.mean_all()?;
        let entropy_loss = entropy_mean.neg()?;

        // Total loss
        let total_loss = ((&policy_loss + (&value_loss * self.config.vf_coef as f64)?)?
            + (&entropy_loss * self.config.ent_coef as f64)?)?;

        // Backward pass with gradient clipping
        optimizer.backward_step(&total_loss)?;

        // Collect metrics
        let policy_loss_val: f32 = policy_loss.to_scalar()?;
        let value_loss_val: f32 = value_loss.to_scalar()?;
        let entropy_val: f32 = entropy_mean.to_scalar()?;

        Ok(TrainMetrics {
            policy_loss: policy_loss_val,
            value_loss: value_loss_val,
            entropy: entropy_val,
            approx_kl: 0.0,      // Not computed for A2C
            clip_fraction: 0.0,  // Not applicable to A2C
            explained_variance: 0.0,
            learning_rate: self.config.learning_rate,
            timesteps: self.total_timesteps,
            episodes: 0,
            mean_reward: 0.0,
            std_reward: 0.0,
        })
    }

    /// Collect rollout data from environment.
    fn collect_rollout(&mut self) -> Result<(RolloutBuffer, Vec<f32>, Vec<usize>)> {
        let num_envs = self.env.num_envs();
        let n_steps = self.config.n_steps;

        let mut buffer = RolloutBuffer::new(n_steps, num_envs, self.obs_dim, self.act_dim, self.device.clone())?;

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
        buffer.compute_returns_and_advantages(&last_values, self.config.gamma, self.config.gae_lambda)?;

        self.total_timesteps += n_steps * num_envs;

        Ok((buffer, episode_rewards, episode_lengths))
    }

    /// Train the agent for a given number of timesteps.
    pub fn train<F>(&mut self, total_timesteps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!("Starting A2C training for {} timesteps", total_timesteps);

        let num_envs = self.env.num_envs();
        let n_steps = self.config.n_steps;
        let rollout_timesteps = n_steps * num_envs;
        let n_iterations = (total_timesteps + rollout_timesteps - 1) / rollout_timesteps;

        for iteration in 0..n_iterations {
            // Collect rollout
            let (buffer, episode_rewards, _) = self.collect_rollout()?;

            // Update policy (single update per rollout, unlike PPO's multiple epochs)
            let mut metrics = self.update(&buffer)?;

            // Update metrics with episode info
            if !episode_rewards.is_empty() {
                metrics.mean_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
                let mean = metrics.mean_reward;
                metrics.std_reward = (episode_rewards.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f32>() / episode_rewards.len() as f32)
                    .sqrt();
                metrics.episodes = episode_rewards.len();
            }

            metrics.timesteps = self.total_timesteps;

            if iteration % 100 == 0 {
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

        info!("Training completed: {} total timesteps", self.total_timesteps);
        Ok(())
    }

    /// Predict action for given observation (inference mode).
    pub fn predict(&mut self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        let logits_or_mean = self.policy_forward(obs)?;

        if self.is_discrete {
            if deterministic {
                // Argmax for deterministic action
                Ok(logits_or_mean.argmax(1)?.unsqueeze(1)?.to_dtype(DType::F32)?)
            } else {
                let (action, _) = self.sample_action(obs)?;
                Ok(action)
            }
        } else {
            if deterministic {
                // Use mean directly
                Ok(logits_or_mean)
            } else {
                let (action, _) = self.sample_action(obs)?;
                Ok(action)
            }
        }
    }

    /// Get the current value estimate for observations.
    pub fn get_value(&self, obs: &Tensor) -> Result<Tensor> {
        self.value_forward(obs)
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for A2CAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let (buffer, episode_rewards, _) = self.collect_rollout()?;
        let mut metrics = self.update(&buffer)?;

        if !episode_rewards.is_empty() {
            metrics.mean_reward = episode_rewards.iter().sum::<f32>() / episode_rewards.len() as f32;
            metrics.episodes = episode_rewards.len();
        }

        Ok(metrics)
    }

    fn save(&self, path: &Path) -> Result<()> {
        let data = self.var_map.data().lock().unwrap();
        let mut tensors: std::collections::HashMap<String, Tensor> = std::collections::HashMap::new();

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
    fn test_a2c_config_validation() {
        let config = A2CConfig::default();
        assert!(config.validate().is_ok());

        let invalid = A2CConfig::default().learning_rate(-0.1);
        assert!(invalid.validate().is_err());
    }
}
