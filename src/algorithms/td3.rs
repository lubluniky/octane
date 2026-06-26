//! Twin Delayed DDPG (TD3) implementation.
//!
//! TD3 is an off-policy algorithm for continuous action spaces that improves upon DDPG with:
//! - Twin Q-networks to reduce overestimation bias
//! - Delayed policy updates (update policy less frequently than critics)
//! - Target policy smoothing (add noise to target actions)
//!
//! Reference: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (2018)

use crate::algorithms::config::TD3Config;
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

/// TD3 Agent for continuous action spaces.
pub struct TD3Agent<E: Environment + Clone + 'static> {
    /// Algorithm configuration.
    config: TD3Config,
    /// Vectorized environment.
    env: VecEnv<E>,
    /// Device for tensor operations.
    device: Device,

    /// Actor network var_map.
    actor_var_map: VarMap,
    /// Target actor network var_map.
    target_actor_var_map: VarMap,
    /// Critic1 network var_map.
    critic1_var_map: VarMap,
    /// Critic2 network var_map.
    critic2_var_map: VarMap,
    /// Target critic1 network var_map.
    target_critic1_var_map: VarMap,
    /// Target critic2 network var_map.
    target_critic2_var_map: VarMap,

    /// Persistent optimizers (Adam moment state must survive across updates).
    actor_optimizer: AdamW,
    critic1_optimizer: AdamW,
    critic2_optimizer: AdamW,

    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    action_dim: usize,
    /// Action scale (max action magnitude).
    action_scale: f32,

    /// Experience replay buffer.
    replay_buffer: ReplayBuffer,

    /// Total timesteps trained.
    total_timesteps: usize,
    /// Total gradient updates performed.
    gradient_steps_done: usize,

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> TD3Agent<E> {
    /// Create a new TD3 agent.
    pub fn new(config: TD3Config, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();

        let obs_dim = obs_space.flat_dim();
        let action_dim = act_space.flat_dim();
        let action_scale = 1.0; // Assume normalized action space [-1, 1]

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Create replay buffer
        let buffer_config =
            ReplayBufferConfig::new(obs_dim, action_dim).capacity(config.buffer_size);
        let replay_buffer = ReplayBuffer::new(buffer_config, device)?;

        let actor_var_map = VarMap::new();
        let target_actor_var_map = VarMap::new();
        let critic1_var_map = VarMap::new();
        let critic2_var_map = VarMap::new();
        let target_critic1_var_map = VarMap::new();
        let target_critic2_var_map = VarMap::new();

        let opt_params = ParamsAdamW {
            lr: config.learning_rate as f64,
            ..Default::default()
        };

        let mut agent = Self {
            config: config.clone(),
            env,
            device,
            actor_var_map,
            target_actor_var_map,
            critic1_var_map,
            critic2_var_map,
            target_critic1_var_map,
            target_critic2_var_map,
            actor_optimizer: AdamW::new(Vec::new(), opt_params.clone())?,
            critic1_optimizer: AdamW::new(Vec::new(), opt_params.clone())?,
            critic2_optimizer: AdamW::new(Vec::new(), opt_params.clone())?,
            obs_dim,
            action_dim,
            action_scale,
            replay_buffer,
            total_timesteps: 0,
            gradient_steps_done: 0,
            rng,
        };

        agent.init_networks()?;

        // Bind optimizers to the populated network variables.
        agent.actor_optimizer = AdamW::new(agent.actor_var_map.all_vars(), opt_params.clone())?;
        agent.critic1_optimizer = AdamW::new(agent.critic1_var_map.all_vars(), opt_params.clone())?;
        agent.critic2_optimizer = AdamW::new(agent.critic2_var_map.all_vars(), opt_params)?;

        info!(
            "TD3 Agent initialized: obs_dim={}, action_dim={}, policy_delay={}",
            obs_dim, action_dim, config.policy_delay
        );

        Ok(agent)
    }

    /// Initialize all networks.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        // Actor networks
        self.init_actor_network(&self.actor_var_map, "actor", &candle_device)?;
        self.init_actor_network(&self.target_actor_var_map, "target_actor", &candle_device)?;

        // Critic networks
        self.init_critic_network(&self.critic1_var_map, "critic1", &candle_device)?;
        self.init_critic_network(&self.critic2_var_map, "critic2", &candle_device)?;
        self.init_critic_network(
            &self.target_critic1_var_map,
            "target_critic1",
            &candle_device,
        )?;
        self.init_critic_network(
            &self.target_critic2_var_map,
            "target_critic2",
            &candle_device,
        )?;

        // Copy weights to targets
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
        for (i, &hidden_size) in self.config.policy_hidden_sizes.iter().enumerate() {
            let _ = candle_nn::linear(
                in_dim,
                hidden_size,
                vb.pp(format!("{}.layer_{}", prefix, i)),
            )?;
            in_dim = hidden_size;
        }

        // Output layer with tanh activation (bounded actions)
        let _ = candle_nn::linear(in_dim, self.action_dim, vb.pp(format!("{}.output", prefix)))?;

        Ok(())
    }

    /// Initialize critic network (Q-function).
    fn init_critic_network(
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

    /// Forward pass through actor network.
    fn actor_forward(&self, obs: &Tensor, var_map: &VarMap, prefix: &str) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(var_map, DType::F32, &candle_device);

        let mut x = obs.clone();

        for (i, &hidden_size) in self.config.policy_hidden_sizes.iter().enumerate() {
            let in_dim = if i == 0 {
                self.obs_dim
            } else {
                self.config.policy_hidden_sizes[i - 1]
            };
            let linear = candle_nn::linear(
                in_dim,
                hidden_size,
                vb.pp(format!("{}.layer_{}", prefix, i)),
            )?;
            x = linear.forward(&x)?;
            x = x.relu()?;
        }

        let output_linear = candle_nn::linear(
            *self.config.policy_hidden_sizes.last().unwrap(),
            self.action_dim,
            vb.pp(format!("{}.output", prefix)),
        )?;

        // Apply tanh to bound actions to [-1, 1]
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

    /// Select action with exploration noise.
    fn select_action(&mut self, obs: &Tensor, training: bool) -> Result<Tensor> {
        let action = self.actor_forward(obs, &self.actor_var_map, "actor")?;

        if training {
            // Add exploration noise
            let noise = Tensor::randn_like(&action, 0.0, self.config.exploration_noise as f64)?;
            let noisy_action = (&action + &noise)?;
            // Clip to action bounds
            Ok(noisy_action.clamp(-self.action_scale, self.action_scale)?)
        } else {
            Ok(action)
        }
    }

    /// Hard update: copy networks to targets.
    fn hard_update_targets(&mut self) -> Result<()> {
        Self::copy_weights(
            &self.actor_var_map,
            &self.target_actor_var_map,
            "actor",
            "target_actor",
        )?;
        Self::copy_weights(
            &self.critic1_var_map,
            &self.target_critic1_var_map,
            "critic1",
            "target_critic1",
        )?;
        Self::copy_weights(
            &self.critic2_var_map,
            &self.target_critic2_var_map,
            "critic2",
            "target_critic2",
        )?;
        Ok(())
    }

    /// Soft update: polyak averaging.
    fn soft_update_targets(&mut self) -> Result<()> {
        let tau = self.config.tau;
        Self::polyak_update(
            &self.actor_var_map,
            &self.target_actor_var_map,
            "actor",
            "target_actor",
            tau,
        )?;
        Self::polyak_update(
            &self.critic1_var_map,
            &self.target_critic1_var_map,
            "critic1",
            "target_critic1",
            tau,
        )?;
        Self::polyak_update(
            &self.critic2_var_map,
            &self.target_critic2_var_map,
            "critic2",
            "target_critic2",
            tau,
        )?;
        Ok(())
    }

    /// Copy weights from source to target.
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

    /// Polyak averaging.
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

        // ========== Compute Target Q-value ==========
        // Target policy smoothing: add clipped noise to target actions
        let target_actions = self.actor_forward(
            &batch.next_observations,
            &self.target_actor_var_map,
            "target_actor",
        )?;

        let noise =
            Tensor::randn_like(&target_actions, 0.0, self.config.target_policy_noise as f64)?;
        let noise = noise.clamp(
            -self.config.target_noise_clip,
            self.config.target_noise_clip,
        )?;
        let smoothed_target_actions =
            ((&target_actions + &noise)?).clamp(-self.action_scale, self.action_scale)?;

        // Twin Q-values from target critics
        let target_q1 = self.critic_forward(
            &batch.next_observations,
            &smoothed_target_actions,
            &self.target_critic1_var_map,
            "target_critic1",
        )?;
        let target_q2 = self.critic_forward(
            &batch.next_observations,
            &smoothed_target_actions,
            &self.target_critic2_var_map,
            "target_critic2",
        )?;

        // Take minimum of twin Q-values
        let target_q = target_q1.minimum(&target_q2)?;

        // TD target: r + gamma * (1 - done) * min_Q_target
        let not_done = (Tensor::ones_like(&batch.dones)? - &batch.dones)?;
        let td_target =
            (&batch.rewards + (&target_q * self.config.gamma as f64)? * &not_done)?.detach();

        // ========== Update Critics ==========
        let current_q1 = self.critic_forward(
            &batch.observations,
            &batch.actions,
            &self.critic1_var_map,
            "critic1",
        )?;
        let critic1_loss = (&current_q1 - &td_target)?.sqr()?.mean_all()?;

        let current_q2 = self.critic_forward(
            &batch.observations,
            &batch.actions,
            &self.critic2_var_map,
            "critic2",
        )?;
        let critic2_loss = (&current_q2 - &td_target)?.sqr()?.mean_all()?;

        self.critic1_optimizer.backward_step(&critic1_loss)?;
        self.critic2_optimizer.backward_step(&critic2_loss)?;

        let critic_loss_val =
            (critic1_loss.to_scalar::<f32>()? + critic2_loss.to_scalar::<f32>()?) / 2.0;

        // ========== Delayed Policy Update ==========
        let mut actor_loss_val = 0.0f32;

        self.gradient_steps_done += 1;

        if self
            .gradient_steps_done
            .is_multiple_of(self.config.policy_delay)
        {
            // Update actor to maximize Q1
            let actor_actions =
                self.actor_forward(&batch.observations, &self.actor_var_map, "actor")?;
            let actor_q = self.critic_forward(
                &batch.observations,
                &actor_actions,
                &self.critic1_var_map,
                "critic1",
            )?;
            let actor_loss = actor_q.neg()?.mean_all()?;

            self.actor_optimizer.backward_step(&actor_loss)?;

            actor_loss_val = actor_loss.to_scalar()?;

            // Soft update targets
            self.soft_update_targets()?;
        }

        Ok((critic_loss_val, actor_loss_val))
    }

    /// Train the agent.
    pub fn train<F>(&mut self, total_timesteps: usize, mut callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        info!("Starting TD3 training for {} timesteps", total_timesteps);

        let num_envs = self.env.num_envs();
        let mut obs = self.env.reset(&self.device)?;

        let mut episode_rewards: Vec<f32> = Vec::new();
        let mut current_rewards = vec![0.0f32; num_envs];

        let mut total_critic_loss = 0.0f32;
        let mut total_actor_loss = 0.0f32;
        let mut update_count = 0usize;

        while self.total_timesteps < total_timesteps {
            // Select actions
            let actions = self.select_action(&obs, true)?;

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
                    policy_loss: if update_count > 0 {
                        total_actor_loss / update_count as f32
                    } else {
                        0.0
                    },
                    value_loss: if update_count > 0 {
                        total_critic_loss / update_count as f32
                    } else {
                        0.0
                    },
                    entropy: 0.0,
                    approx_kl: 0.0,
                    clip_fraction: 0.0,
                    explained_variance: 0.0,
                    learning_rate: self.config.learning_rate,
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

        info!("TD3 training completed: {} timesteps", self.total_timesteps);
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

impl<E: Environment + Clone + 'static> RLAlgorithm for TD3Agent<E> {
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
        for (name, var) in self.critic1_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }
        for (name, var) in self.critic2_var_map.data().lock().unwrap().iter() {
            tensors.insert(name.clone(), var.as_tensor().clone());
        }

        candle_core::safetensors::save(&tensors, path)?;

        let config_path = path.with_extension("json");
        let config_json = serde_json::to_string_pretty(&self.config)?;
        std::fs::write(config_path, config_json)?;

        info!("TD3 model saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        for var_map in [
            &self.actor_var_map,
            &self.critic1_var_map,
            &self.critic2_var_map,
        ] {
            let mut data = var_map.data().lock().unwrap();
            for (name, tensor) in &tensors {
                if let Some(var) = data.get_mut(name) {
                    var.set(tensor)?;
                }
            }
        }

        self.hard_update_targets()?;

        info!("TD3 model loaded from {:?}", path);
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
    use crate::envs::{BoxSpace, StepResult};

    #[test]
    fn test_td3_config_defaults() {
        let config = TD3Config::default();
        assert_eq!(config.policy_delay, 2);
        assert!((config.gamma - 0.99).abs() < 1e-6);
        assert!((config.target_policy_noise - 0.2).abs() < 1e-6);
    }

    #[derive(Clone)]
    struct ContinuousTestEnv {
        obs: BoxSpace,
        act: BoxSpace,
    }

    impl Environment for ContinuousTestEnv {
        type ObsSpace = BoxSpace;
        type ActSpace = BoxSpace;
        fn observation_space(&self) -> &BoxSpace {
            &self.obs
        }
        fn action_space(&self) -> &BoxSpace {
            &self.act
        }
        fn reset(&mut self, device: &Device) -> Result<Tensor> {
            Ok(Tensor::zeros(
                self.obs.shape(),
                DType::F32,
                &device.to_candle()?,
            )?)
        }
        fn step(&mut self, _action: &Tensor, device: &Device) -> Result<StepResult> {
            let cd = device.to_candle()?;
            Ok(StepResult {
                observation: Tensor::zeros(self.obs.shape(), DType::F32, &cd)?,
                reward: 1.0,
                terminated: false,
                truncated: false,
                info: None,
            })
        }
    }

    // Smoke test: TD3 update path (twin critics + delayed actor) must run
    // end-to-end with the persistent optimizers introduced for Adam state.
    #[test]
    fn test_td3_training_runs_with_persistent_optimizers() {
        let device = Device::Cpu;
        let env = ContinuousTestEnv {
            obs: BoxSpace::symmetric(1.0, vec![3]),
            act: BoxSpace::symmetric(1.0, vec![2]),
        };
        let mut config = TD3Config::default();
        config.batch_size = 8;
        config.buffer_size = 256;
        config.learning_starts = 8;
        let mut agent = TD3Agent::new(config, VecEnv::new(vec![env], 1), device).unwrap();
        agent.train(40, |_| {}).unwrap();
    }
}
