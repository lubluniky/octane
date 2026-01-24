//! Unified Agent interface for RL algorithms.
//!
//! The Agent struct provides a high-level interface for training and inference,
//! abstracting over the underlying algorithm (PPO, A2C, etc.).

use crate::algorithms::a2c::A2CAgent;
use crate::algorithms::config::{A2CConfig, PPOConfig};
use crate::algorithms::metrics::TrainMetrics;
use crate::algorithms::ppo::PPOAgent;
use crate::algorithms::traits::RLAlgorithm;
use crate::core::{Device, Result};
use crate::envs::{Environment, VecEnv};
use candle_core::Tensor;
use std::path::Path;
use tracing::info;

/// Algorithm type for agent configuration.
#[derive(Debug, Clone)]
pub enum AlgorithmConfig {
    /// Proximal Policy Optimization.
    PPO(PPOConfig),
    /// Advantage Actor-Critic.
    A2C(A2CConfig),
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        AlgorithmConfig::PPO(PPOConfig::default())
    }
}

impl From<PPOConfig> for AlgorithmConfig {
    fn from(config: PPOConfig) -> Self {
        AlgorithmConfig::PPO(config)
    }
}

impl From<A2CConfig> for AlgorithmConfig {
    fn from(config: A2CConfig) -> Self {
        AlgorithmConfig::A2C(config)
    }
}

/// Internal algorithm implementation wrapper.
enum AgentImpl<E: Environment + Clone + 'static> {
    PPO(PPOAgent<E>),
    A2C(A2CAgent<E>),
}

/// High-level agent interface for training and inference.
///
/// # Example
///
/// ```ignore
/// use rocket_rs::{Agent, PPOConfig, VecEnv, Device};
///
/// // Create environment
/// let env = MyEnv::new();
/// let vec_env = VecEnv::new(vec![env], 8);
///
/// // Create and train agent
/// let config = PPOConfig::default();
/// let mut agent = Agent::new(config, vec_env, Device::Cpu)?;
///
/// agent.train(100_000, |metrics| {
///     println!("Step {}: reward = {:.2}", metrics.timesteps, metrics.mean_reward);
/// })?;
///
/// // Save trained model
/// agent.save("model.safetensors")?;
/// ```
pub struct Agent<E: Environment + Clone + 'static> {
    /// Internal algorithm implementation.
    inner: AgentImpl<E>,
    /// Device for tensor operations.
    device: Device,
}

impl<E: Environment + Clone + 'static> Agent<E> {
    /// Create a new agent with the specified algorithm configuration.
    ///
    /// # Arguments
    /// * `config` - Algorithm configuration (PPOConfig or A2CConfig)
    /// * `env` - Vectorized environment
    /// * `device` - Compute device (CPU, Metal, CUDA)
    ///
    /// # Example
    /// ```ignore
    /// let agent = Agent::new(PPOConfig::default(), vec_env, Device::Cpu)?;
    /// ```
    pub fn new<C: Into<AlgorithmConfig>>(
        config: C,
        env: VecEnv<E>,
        device: Device,
    ) -> Result<Self> {
        let config = config.into();

        let inner = match config {
            AlgorithmConfig::PPO(ppo_config) => {
                info!("Creating PPO agent");
                AgentImpl::PPO(PPOAgent::new(ppo_config, env, device)?)
            }
            AlgorithmConfig::A2C(a2c_config) => {
                info!("Creating A2C agent");
                AgentImpl::A2C(A2CAgent::new(a2c_config, env, device)?)
            }
        };

        Ok(Self { inner, device })
    }

    /// Create a new PPO agent with default configuration.
    pub fn ppo(env: VecEnv<E>, device: Device) -> Result<Self> {
        Self::new(PPOConfig::default(), env, device)
    }

    /// Create a new A2C agent with default configuration.
    pub fn a2c(env: VecEnv<E>, device: Device) -> Result<Self> {
        Self::new(A2CConfig::default(), env, device)
    }

    /// Train the agent for a specified number of timesteps.
    ///
    /// # Arguments
    /// * `total_timesteps` - Total environment steps to train for
    /// * `callback` - Function called after each update with training metrics
    ///
    /// # Example
    /// ```ignore
    /// agent.train(100_000, |metrics| {
    ///     if metrics.timesteps % 10_000 == 0 {
    ///         println!("Progress: {} steps, reward: {:.2}",
    ///             metrics.timesteps, metrics.mean_reward);
    ///     }
    /// })?;
    /// ```
    pub fn train<F>(&mut self, total_timesteps: usize, callback: F) -> Result<()>
    where
        F: FnMut(&TrainMetrics),
    {
        match &mut self.inner {
            AgentImpl::PPO(agent) => agent.train(total_timesteps, callback),
            AgentImpl::A2C(agent) => agent.train(total_timesteps, callback),
        }
    }

    /// Perform a single training step.
    ///
    /// Useful for custom training loops.
    pub fn train_step(&mut self) -> Result<TrainMetrics> {
        match &mut self.inner {
            AgentImpl::PPO(agent) => agent.train_step(),
            AgentImpl::A2C(agent) => agent.train_step(),
        }
    }

    /// Predict action for given observation.
    ///
    /// # Arguments
    /// * `obs` - Observation tensor [batch_size, obs_dim]
    /// * `deterministic` - Use deterministic (greedy) policy if true
    ///
    /// # Returns
    /// Action tensor [batch_size, act_dim]
    pub fn predict(&mut self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        match &mut self.inner {
            AgentImpl::PPO(agent) => agent.predict(obs, deterministic),
            AgentImpl::A2C(agent) => agent.predict(obs, deterministic),
        }
    }

    /// Get value estimate for observations.
    ///
    /// # Arguments
    /// * `obs` - Observation tensor [batch_size, obs_dim]
    ///
    /// # Returns
    /// Value estimates [batch_size, 1]
    pub fn get_value(&self, obs: &Tensor) -> Result<Tensor> {
        match &self.inner {
            AgentImpl::PPO(agent) => agent.get_value(obs),
            AgentImpl::A2C(agent) => agent.get_value(obs),
        }
    }

    /// Save the agent to disk.
    ///
    /// Saves both model weights (safetensors) and configuration (JSON).
    ///
    /// # Arguments
    /// * `path` - Path for the model file (e.g., "model.safetensors")
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let path = path.as_ref();
        match &self.inner {
            AgentImpl::PPO(agent) => agent.save(path),
            AgentImpl::A2C(agent) => agent.save(path),
        }
    }

    /// Load agent weights from disk.
    ///
    /// # Arguments
    /// * `path` - Path to the model file
    pub fn load<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        let path = path.as_ref();
        match &mut self.inner {
            AgentImpl::PPO(agent) => agent.load(path),
            AgentImpl::A2C(agent) => agent.load(path),
        }
    }

    /// Get the device being used.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get total timesteps trained.
    pub fn total_timesteps(&self) -> usize {
        match &self.inner {
            AgentImpl::PPO(agent) => agent.total_timesteps(),
            AgentImpl::A2C(agent) => agent.total_timesteps(),
        }
    }

    /// Get the algorithm name.
    pub fn algorithm_name(&self) -> &'static str {
        match &self.inner {
            AgentImpl::PPO(_) => "PPO",
            AgentImpl::A2C(_) => "A2C",
        }
    }
}

/// Builder for configuring and creating agents.
pub struct AgentBuilder<E: Environment + Clone + 'static> {
    config: AlgorithmConfig,
    device: Device,
    _phantom: std::marker::PhantomData<E>,
}

impl<E: Environment + Clone + 'static> AgentBuilder<E> {
    /// Create a new agent builder with default PPO configuration.
    pub fn new() -> Self {
        Self {
            config: AlgorithmConfig::default(),
            device: Device::Cpu,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Use PPO algorithm with custom configuration.
    pub fn ppo(mut self, config: PPOConfig) -> Self {
        self.config = AlgorithmConfig::PPO(config);
        self
    }

    /// Use A2C algorithm with custom configuration.
    pub fn a2c(mut self, config: A2CConfig) -> Self {
        self.config = AlgorithmConfig::A2C(config);
        self
    }

    /// Set the compute device.
    pub fn device(mut self, device: Device) -> Self {
        self.device = device;
        self
    }

    /// Build the agent with the specified environment.
    pub fn build(self, env: VecEnv<E>) -> Result<Agent<E>> {
        Agent::new(self.config, env, self.device)
    }
}

impl<E: Environment + Clone + 'static> Default for AgentBuilder<E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_algorithm_config_from() {
        let ppo: AlgorithmConfig = PPOConfig::default().into();
        matches!(ppo, AlgorithmConfig::PPO(_));

        let a2c: AlgorithmConfig = A2CConfig::default().into();
        matches!(a2c, AlgorithmConfig::A2C(_));
    }
}
