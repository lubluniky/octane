//! Deep Q-Network (DQN) and Double DQN implementation.
//!
//! DQN is an off-policy algorithm for discrete action spaces that uses:
//! - Experience replay for sample efficiency
//! - Target network for training stability
//! - Epsilon-greedy exploration
//!
//! Double DQN improvement reduces overestimation bias by decoupling
//! action selection from action evaluation.
//!
//! Reference: Mnih et al., "Human-level control through deep reinforcement learning" (2015)
//! Reference: Van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2016)

use crate::algorithms::config::DQNConfig;
use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::traits::RLAlgorithm;
use crate::buffer::{ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, Result, OctaneError};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use std::path::Path;
use tracing::info;

/// DQN Agent for discrete action spaces.
pub struct DQNAgent<E: Environment + Clone + 'static> {
    /// Algorithm configuration.
    config: DQNConfig,
    /// Vectorized environment.
    env: VecEnv<E>,
    /// Device for tensor operations.
    device: Device,

    /// Q-network variable map.
    q_var_map: VarMap,
    /// Target Q-network variable map.
    target_var_map: VarMap,

    /// Observation dimension.
    obs_dim: usize,
    /// Number of actions (discrete).
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

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> DQNAgent<E> {
    /// Create a new DQN agent.
    pub fn new(config: DQNConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();

        let obs_dim = obs_space.flat_dim();
        let num_actions = act_space.flat_dim();

        // Verify discrete action space
        if act_space.shape() != [1] && act_space.shape().len() != 1 {
            return Err(OctaneError::InvalidConfig(
                "DQN requires discrete action space".into(),
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

        let q_var_map = VarMap::new();
        let target_var_map = VarMap::new();
        let hidden_sizes = vec![256, 256];

        let mut agent = Self {
            config: config.clone(),
            env,
            device,
            q_var_map,
            target_var_map,
            obs_dim,
            num_actions,
            hidden_sizes,
            replay_buffer,
            epsilon: config.epsilon_start,
            current_lr: config.learning_rate,
            total_timesteps: 0,
            steps_since_target_update: 0,
            rng,
        };

        agent.init_networks()?;

        info!(
            "DQN Agent initialized: obs_dim={}, num_actions={}, double_dqn={}",
            obs_dim, num_actions, config.double_dqn
        );

        Ok(agent)
    }

    /// Initialize Q-network and target network.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // Initialize Q-network
        self.init_q_network(&self.q_var_map, "q", &candle_device)?;
        // Initialize target network with same architecture
        self.init_q_network(&self.target_var_map, "target", &candle_device)?;

        // Copy weights from Q to target
        self.hard_update_target()?;

        Ok(())
    }

    /// Initialize a Q-network in the given var_map.
    fn init_q_network(
        &self,
        var_map: &VarMap,
        prefix: &str,
        candle_device: &candle_core::Device,
    ) -> Result<()> {
        let vb = VarBuilder::from_varmap(var_map, DType::F32, candle_device);

        let mut in_dim = self.obs_dim;
        for (i, &hidden_size) in self.hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("{}.layer_{}", prefix, i)))?;
            in_dim = hidden_size;
        }

        // Output layer: Q-values for each action
        let _ = candle_nn::linear(in_dim, self.num_actions, vb.pp(format!("{}.output", prefix)))?;

        Ok(())
    }

    /// Forward pass through Q-network.
    fn q_forward(&self, obs: &Tensor, var_map: &VarMap) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(var_map, DType::F32, &candle_device);

        let prefix = if std::ptr::eq(var_map, &self.q_var_map) {
            "q"
        } else {
            "target"
        };

        let mut x = obs.clone();

        for (i, &hidden_size) in self.hidden_sizes.iter().enumerate() {
            let in_dim = if i == 0 { self.obs_dim } else { self.hidden_sizes[i - 1] };
            let linear = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("{}.layer_{}", prefix, i)))?;
            x = linear.forward(&x)?;
            x = x.relu()?;
        }

        let output_linear = candle_nn::linear(
            *self.hidden_sizes.last().unwrap(),
            self.num_actions,
            vb.pp(format!("{}.output", prefix)),
        )?;

        Ok(output_linear.forward(&x)?)
    }

    /// Hard update: copy Q-network weights to target network.
    fn hard_update_target(&mut self) -> Result<()> {
        let q_data = self.q_var_map.data().lock().unwrap();
        let mut target_data = self.target_var_map.data().lock().unwrap();

        for (name, q_var) in q_data.iter() {
            let target_name = name.replace("q.", "target.");
            if let Some(target_var) = target_data.get_mut(&target_name) {
                target_var.set(q_var.as_tensor())?;
            }
        }

        Ok(())
    }

    /// Soft update: polyak averaging of Q-network to target network.
    fn soft_update_target(&mut self) -> Result<()> {
        let tau = self.config.tau;
        let q_data = self.q_var_map.data().lock().unwrap();
        let mut target_data = self.target_var_map.data().lock().unwrap();

        for (name, q_var) in q_data.iter() {
            let target_name = name.replace("q.", "target.");
            if let Some(target_var) = target_data.get_mut(&target_name) {
                let q_tensor = q_var.as_tensor();
                let target_tensor = target_var.as_tensor();
                let new_tensor = ((q_tensor * tau as f64)? + (target_tensor * (1.0 - tau) as f64)?)?;
                target_var.set(&new_tensor)?;
            }
        }

        Ok(())
    }

    /// Select action using epsilon-greedy policy.
    fn select_action(&mut self, obs: &Tensor, training: bool) -> Result<Tensor> {
        let batch_size = obs.dim(0)?;
        let candle_device = self.device.to_candle()?;

        if training && self.rng.gen::<f32>() < self.epsilon {
            // Random action
            let actions: Vec<f32> = (0..batch_size)
                .map(|_| self.rng.gen_range(0..self.num_actions) as f32)
                .collect();
            Ok(Tensor::from_slice(&actions, (batch_size, 1), &candle_device)?)
        } else {
            // Greedy action from Q-network
            let q_values = self.q_forward(obs, &self.q_var_map)?;
            let actions = q_values.argmax(1)?.unsqueeze(1)?.to_dtype(DType::F32)?;
            Ok(actions)
        }
    }

    /// Compute TD target using DQN or Double DQN.
    fn compute_td_target(&self, batch: &crate::buffer::ReplayBatch) -> Result<Tensor> {
        let _candle_device = self.device.to_candle()?;

        // Get next Q-values from target network
        let next_q_target = self.q_forward(&batch.next_observations, &self.target_var_map)?;

        let next_q_values = if self.config.double_dqn {
            // Double DQN: select action using Q-network, evaluate with target
            let next_q_online = self.q_forward(&batch.next_observations, &self.q_var_map)?;
            let best_actions = next_q_online.argmax(1)?;

            // Gather Q-values for selected actions
            next_q_target.gather(&best_actions.unsqueeze(1)?, 1)?.squeeze(1)?
        } else {
            // Standard DQN: use max Q from target
            next_q_target.max(1)?
        };

        // TD target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        let not_done = (Tensor::ones_like(&batch.dones)? - &batch.dones)?;
        let td_target = (&batch.rewards + (&next_q_values * self.config.gamma as f64)? * &not_done)?;

        Ok(td_target)
    }

    /// Perform a single gradient update.
    fn update(&mut self) -> Result<(f32, f32)> {
        if !self.replay_buffer.can_sample(self.config.batch_size) {
            return Ok((0.0, 0.0));
        }

        let batch = self.replay_buffer.sample(self.config.batch_size)?;

        // Compute current Q-values
        let q_values = self.q_forward(&batch.observations, &self.q_var_map)?;
        let actions_i64 = batch.actions.squeeze(1)?.to_dtype(DType::I64)?;
        let current_q = q_values.gather(&actions_i64.unsqueeze(1)?, 1)?.squeeze(1)?;

        // Compute TD target (no gradient)
        let td_target = self.compute_td_target(&batch)?.detach();

        // Compute loss (Huber loss for stability)
        let td_error = (&current_q - &td_target)?;
        let loss = if self.config.use_huber_loss {
            // Huber loss: smooth L1
            let abs_error = td_error.abs()?;
            let quadratic = abs_error.clamp(0.0, 1.0)?.sqr()?;
            let linear = (abs_error - &quadratic.sqrt()?)?;
            ((quadratic * 0.5)? + &linear)?.mean_all()?
        } else {
            // MSE loss
            td_error.sqr()?.mean_all()?
        };

        // Backward pass
        let params = ParamsAdamW {
            lr: self.current_lr as f64,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut optimizer = AdamW::new(self.q_var_map.all_vars(), params)?;
        optimizer.backward_step(&loss)?;

        // Update priorities if using PER
        if self.config.prioritized_replay {
            let td_errors: Vec<f32> = td_error.abs()?.to_vec1()?;
            self.replay_buffer.update_priorities(&batch.indices, &td_errors);
        }

        let loss_val: f32 = loss.to_scalar()?;
        let mean_q: f32 = current_q.mean_all()?.to_scalar()?;

        Ok((loss_val, mean_q))
    }

    /// Update exploration rate (epsilon decay).
    fn update_epsilon(&mut self) {
        self.epsilon = (self.epsilon - self.config.epsilon_decay)
            .max(self.config.epsilon_end);
    }

    /// Collect experience and train.
    pub fn train<F>(&mut self, total_timesteps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!("Starting DQN training for {} timesteps", total_timesteps);

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
            let next_obs_vec: Vec<Vec<f32>> = self.tensor_to_batch_vecs(&step_result.observations)?;
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
                    policy_loss: if update_count > 0 { total_loss / update_count as f32 } else { 0.0 },
                    value_loss: if update_count > 0 { total_q / update_count as f32 } else { 0.0 },
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

        info!("DQN training completed: {} timesteps", self.total_timesteps);
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
    pub fn predict(&mut self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        if deterministic {
            let q_values = self.q_forward(obs, &self.q_var_map)?;
            Ok(q_values.argmax(1)?.unsqueeze(1)?.to_dtype(DType::F32)?)
        } else {
            self.select_action(obs, false)
        }
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for DQNAgent<E> {
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
        let data = self.q_var_map.data().lock().unwrap();
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        for (name, var) in data.iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&tensors, path)?;

        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("DQN model saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        let mut data = self.q_var_map.data().lock().unwrap();
        for (name, tensor) in tensors {
            if let Some(var) = data.get_mut(&name) {
                var.set(&tensor)?;
            }
        }
        drop(data);

        self.hard_update_target()?;

        info!("DQN model loaded from {:?}", path);
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
    fn test_dqn_config_defaults() {
        let config = DQNConfig::default();
        assert!(config.double_dqn);
        assert!((config.gamma - 0.99).abs() < 1e-6);
        assert!(config.buffer_size > 0);
    }
}
