//! Deep Deterministic Policy Gradient (DDPG) implementation.
//!
//! DDPG is an off-policy algorithm for continuous action spaces that combines:
//! - Deterministic policy gradient
//! - Experience replay
//! - Target networks for stability
//! - Exploration via action noise (Gaussian or Ornstein-Uhlenbeck)
//!
//! Reference: Lillicrap et al., "Continuous control with deep reinforcement learning" (2015)

use crate::algorithms::config::{DDPGConfig, NoiseType};
use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::traits::RLAlgorithm;
use crate::buffer::{ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, Result, OctaneError};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use rand::prelude::*;
use rand_distr::Normal;
use std::path::Path;
use tracing::info;

/// Ornstein-Uhlenbeck noise process for exploration.
struct OUNoise {
    mu: Vec<f32>,
    theta: f32,
    sigma: f32,
    state: Vec<f32>,
    rng: StdRng,
}

impl OUNoise {
    fn new(size: usize, mu: f32, theta: f32, sigma: f32, seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        Self {
            mu: vec![mu; size],
            theta,
            sigma,
            state: vec![mu; size],
            rng,
        }
    }

    fn sample(&mut self) -> Vec<f32> {
        let normal = Normal::new(0.0, 1.0).unwrap();
        for i in 0..self.state.len() {
            let noise: f32 = self.rng.sample(normal);
            self.state[i] += self.theta * (self.mu[i] - self.state[i])
                + self.sigma * noise;
        }
        self.state.clone()
    }

    fn reset(&mut self) {
        self.state = self.mu.clone();
    }
}

/// DDPG Agent for continuous action spaces.
pub struct DDPGAgent<E: Environment + Clone + 'static> {
    /// Algorithm configuration.
    config: DDPGConfig,
    /// Vectorized environment.
    env: VecEnv<E>,
    /// Device for tensor operations.
    device: Device,

    /// Actor network var_map.
    actor_var_map: VarMap,
    /// Target actor network var_map.
    target_actor_var_map: VarMap,
    /// Critic network var_map.
    critic_var_map: VarMap,
    /// Target critic network var_map.
    target_critic_var_map: VarMap,

    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    action_dim: usize,
    /// Action scale.
    action_scale: f32,

    /// Experience replay buffer.
    replay_buffer: ReplayBuffer,

    /// Exploration noise (OU or Gaussian).
    ou_noise: Option<OUNoise>,

    /// Total timesteps trained.
    total_timesteps: usize,

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> DDPGAgent<E> {
    /// Create a new DDPG agent.
    pub fn new(config: DDPGConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();

        let obs_dim = obs_space.flat_dim();
        let action_dim = act_space.flat_dim();
        let action_scale = 1.0;

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Create replay buffer
        let buffer_config = ReplayBufferConfig::new(obs_dim, action_dim)
            .capacity(config.buffer_size);
        let replay_buffer = ReplayBuffer::new(buffer_config, device)?;

        // Create OU noise if configured
        let ou_noise = match config.noise_type {
            NoiseType::OrnsteinUhlenbeck => Some(OUNoise::new(
                action_dim,
                0.0,
                0.15,
                config.noise_std,
                config.seed,
            )),
            NoiseType::Gaussian => None,
        };

        let actor_var_map = VarMap::new();
        let target_actor_var_map = VarMap::new();
        let critic_var_map = VarMap::new();
        let target_critic_var_map = VarMap::new();

        let mut agent = Self {
            config: config.clone(),
            env,
            device,
            actor_var_map,
            target_actor_var_map,
            critic_var_map,
            target_critic_var_map,
            obs_dim,
            action_dim,
            action_scale,
            replay_buffer,
            ou_noise,
            total_timesteps: 0,
            rng,
        };

        agent.init_networks()?;

        info!(
            "DDPG Agent initialized: obs_dim={}, action_dim={}, noise={:?}",
            obs_dim, action_dim, config.noise_type
        );

        Ok(agent)
    }

    /// Initialize all networks.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        self.init_actor_network(&self.actor_var_map, "actor", &candle_device)?;
        self.init_actor_network(&self.target_actor_var_map, "target_actor", &candle_device)?;
        self.init_critic_network(&self.critic_var_map, "critic", &candle_device)?;
        self.init_critic_network(&self.target_critic_var_map, "target_critic", &candle_device)?;

        self.hard_update_targets()?;

        Ok(())
    }

    /// Initialize actor network.
    fn init_actor_network(
        &self,
        var_map: &VarMap,
        prefix: &str,
        candle_device: &candle_core::Device,
    ) -> Result<()> {
        let vb = VarBuilder::from_varmap(var_map, DType::F32, candle_device);

        let mut in_dim = self.obs_dim;
        for (i, &hidden_size) in self.config.actor_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("{}.layer_{}", prefix, i)))?;
            in_dim = hidden_size;
        }

        let _ = candle_nn::linear(in_dim, self.action_dim, vb.pp(format!("{}.output", prefix)))?;

        Ok(())
    }

    /// Initialize critic network.
    fn init_critic_network(
        &self,
        var_map: &VarMap,
        prefix: &str,
        candle_device: &candle_core::Device,
    ) -> Result<()> {
        let vb = VarBuilder::from_varmap(var_map, DType::F32, candle_device);

        let mut in_dim = self.obs_dim + self.action_dim;
        for (i, &hidden_size) in self.config.critic_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("{}.layer_{}", prefix, i)))?;
            in_dim = hidden_size;
        }

        let _ = candle_nn::linear(in_dim, 1, vb.pp(format!("{}.output", prefix)))?;

        Ok(())
    }

    /// Forward pass through actor network.
    fn actor_forward(&self, obs: &Tensor, var_map: &VarMap, prefix: &str) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(var_map, DType::F32, &candle_device);

        let mut x = obs.clone();

        for (i, &hidden_size) in self.config.actor_hidden_sizes.iter().enumerate() {
            let in_dim = if i == 0 { self.obs_dim } else { self.config.actor_hidden_sizes[i - 1] };
            let linear = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("{}.layer_{}", prefix, i)))?;
            x = linear.forward(&x)?;
            x = x.relu()?;
        }

        let output_linear = candle_nn::linear(
            *self.config.actor_hidden_sizes.last().unwrap(),
            self.action_dim,
            vb.pp(format!("{}.output", prefix)),
        )?;

        let action = output_linear.forward(&x)?.tanh()?;
        Ok((action * self.action_scale as f64)?)
    }

    /// Forward pass through critic network.
    fn critic_forward(
        &self,
        obs: &Tensor,
        action: &Tensor,
        var_map: &VarMap,
        prefix: &str,
    ) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(var_map, DType::F32, &candle_device);

        let x = Tensor::cat(&[obs, action], 1)?;

        let mut h = x;
        for (i, &hidden_size) in self.config.critic_hidden_sizes.iter().enumerate() {
            let in_dim = if i == 0 {
                self.obs_dim + self.action_dim
            } else {
                self.config.critic_hidden_sizes[i - 1]
            };
            let linear = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("{}.layer_{}", prefix, i)))?;
            h = linear.forward(&h)?;
            h = h.relu()?;
        }

        let output_linear = candle_nn::linear(
            *self.config.critic_hidden_sizes.last().unwrap(),
            1,
            vb.pp(format!("{}.output", prefix)),
        )?;

        Ok(output_linear.forward(&h)?.squeeze(1)?)
    }

    /// Select action with exploration noise.
    fn select_action(&mut self, obs: &Tensor, training: bool) -> Result<Tensor> {
        let action = self.actor_forward(obs, &self.actor_var_map, "actor")?;

        if training {
            let batch_size = action.dim(0)?;
            let candle_device = self.device.to_candle()?;

            let noise = if let Some(ref mut ou) = self.ou_noise {
                // OU noise
                let mut noise_vec = Vec::with_capacity(batch_size * self.action_dim);
                for _ in 0..batch_size {
                    noise_vec.extend(ou.sample());
                }
                Tensor::from_slice(&noise_vec, (batch_size, self.action_dim), &candle_device)?
            } else {
                // Gaussian noise
                Tensor::randn_like(&action, 0.0, self.config.noise_std as f64)?
            };

            let noisy_action = (&action + &noise)?;
            Ok(noisy_action.clamp(-self.action_scale, self.action_scale)?)
        } else {
            Ok(action)
        }
    }

    /// Hard update targets.
    fn hard_update_targets(&mut self) -> Result<()> {
        Self::copy_weights(&self.actor_var_map, &self.target_actor_var_map, "actor", "target_actor")?;
        Self::copy_weights(&self.critic_var_map, &self.target_critic_var_map, "critic", "target_critic")?;
        Ok(())
    }

    /// Soft update targets.
    fn soft_update_targets(&mut self) -> Result<()> {
        let tau = self.config.tau;
        Self::polyak_update(&self.actor_var_map, &self.target_actor_var_map, "actor", "target_actor", tau)?;
        Self::polyak_update(&self.critic_var_map, &self.target_critic_var_map, "critic", "target_critic", tau)?;
        Ok(())
    }

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
    fn update(&mut self) -> Result<(f32, f32)> {
        if !self.replay_buffer.can_sample(self.config.batch_size) {
            return Ok((0.0, 0.0));
        }

        let batch = self.replay_buffer.sample(self.config.batch_size)?;

        // ========== Update Critic ==========
        let target_actions = self.actor_forward(&batch.next_observations, &self.target_actor_var_map, "target_actor")?;
        let target_q = self.critic_forward(&batch.next_observations, &target_actions, &self.target_critic_var_map, "target_critic")?;

        let not_done = (Tensor::ones_like(&batch.dones)? - &batch.dones)?;
        let td_target = (&batch.rewards + (&target_q * self.config.gamma as f64)? * &not_done)?.detach();

        let current_q = self.critic_forward(&batch.observations, &batch.actions, &self.critic_var_map, "critic")?;
        let critic_loss = (&current_q - &td_target)?.sqr()?.mean_all()?;

        let critic_params = ParamsAdamW {
            lr: self.config.critic_lr as f64,
            ..Default::default()
        };
        let mut critic_optimizer = AdamW::new(self.critic_var_map.all_vars(), critic_params)?;
        critic_optimizer.backward_step(&critic_loss)?;

        // ========== Update Actor ==========
        let actor_actions = self.actor_forward(&batch.observations, &self.actor_var_map, "actor")?;
        let actor_q = self.critic_forward(&batch.observations, &actor_actions, &self.critic_var_map, "critic")?;
        let actor_loss = actor_q.neg()?.mean_all()?;

        let actor_params = ParamsAdamW {
            lr: self.config.actor_lr as f64,
            ..Default::default()
        };
        let mut actor_optimizer = AdamW::new(self.actor_var_map.all_vars(), actor_params)?;
        actor_optimizer.backward_step(&actor_loss)?;

        // Soft update targets
        self.soft_update_targets()?;

        Ok((critic_loss.to_scalar()?, actor_loss.to_scalar()?))
    }

    /// Train the agent.
    pub fn train<F>(&mut self, total_timesteps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!("Starting DDPG training for {} timesteps", total_timesteps);

        let num_envs = self.env.num_envs();
        let mut obs = self.env.reset(&self.device)?;

        let mut episode_rewards: Vec<f32> = Vec::new();
        let mut current_rewards = vec![0.0f32; num_envs];

        let mut total_critic_loss = 0.0f32;
        let mut total_actor_loss = 0.0f32;
        let mut update_count = 0usize;

        while self.total_timesteps < total_timesteps {
            let actions = self.select_action(&obs, true)?;
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
                    // Reset OU noise on episode end
                    if let Some(ref mut ou) = self.ou_noise {
                        ou.reset();
                    }
                }
            }

            obs = step_result.observations;
            self.total_timesteps += num_envs;

            // Training updates
            if self.total_timesteps >= self.config.learning_starts {
                for _ in 0..self.config.gradient_steps {
                    let (critic_loss, actor_loss) = self.update()?;
                    total_critic_loss += critic_loss;
                    total_actor_loss += actor_loss;
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
                    policy_loss: if update_count > 0 { total_actor_loss / update_count as f32 } else { 0.0 },
                    value_loss: if update_count > 0 { total_critic_loss / update_count as f32 } else { 0.0 },
                    entropy: 0.0,
                    approx_kl: 0.0,
                    clip_fraction: 0.0,
                    explained_variance: 0.0,
                    learning_rate: self.config.actor_lr,
                };

                info!(
                    "Step {}: reward={:.2}, critic_loss={:.4}, actor_loss={:.4}",
                    self.total_timesteps,
                    metrics.mean_reward,
                    metrics.value_loss,
                    metrics.policy_loss
                );

                callback(&metrics);

                episode_rewards.clear();
                total_critic_loss = 0.0;
                total_actor_loss = 0.0;
                update_count = 0;
            }
        }

        info!("DDPG training completed: {} timesteps", self.total_timesteps);
        Ok(())
    }

    /// Predict action for given observation.
    pub fn predict(&mut self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        if deterministic {
            self.actor_forward(obs, &self.actor_var_map, "actor")
        } else {
            self.select_action(obs, true)
        }
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for DDPGAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let (critic_loss, actor_loss) = self.update()?;

        Ok(TrainMetrics {
            policy_loss: actor_loss,
            value_loss: critic_loss,
            timesteps: self.total_timesteps,
            ..Default::default()
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let mut tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        for (name, var) in self.actor_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }
        for (name, var) in self.critic_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&tensors, path)?;

        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("DDPG model saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        for var_map in [&self.actor_var_map, &self.critic_var_map] {
            let mut data = var_map.data().lock().unwrap();
            for (name, tensor) in &tensors {
                if let Some(var) = data.get_mut(name) {
                    var.set(tensor)?;
                }
            }
        }

        self.hard_update_targets()?;

        info!("DDPG model loaded from {:?}", path);
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
    fn test_ddpg_config_defaults() {
        let config = DDPGConfig::default();
        assert!((config.gamma - 0.99).abs() < 1e-6);
        assert!((config.tau - 0.005).abs() < 1e-6);
        assert_eq!(config.noise_type, NoiseType::Gaussian);
    }
}
