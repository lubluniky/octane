//! Reinforcement Learning algorithms module.
//!
//! This module provides implementations of state-of-the-art RL algorithms:
//!
//! - **PPO** (Proximal Policy Optimization): A robust policy gradient algorithm
//!   that uses a clipped surrogate objective to enable stable training with
//!   multiple epochs of minibatch updates.
//!
//! - **A2C** (Advantage Actor-Critic): A simpler synchronous actor-critic
//!   algorithm that serves as a strong baseline.
//!
//! # Quick Start
//!
//! ```ignore
//! use rocket_rs::{Agent, PPOConfig, VecEnv, Device};
//!
//! // Create vectorized environment
//! let env = MyEnv::new();
//! let vec_env = VecEnv::new(vec![env], 8);
//!
//! // Create agent with PPO
//! let config = PPOConfig::default()
//!     .learning_rate(3e-4)
//!     .n_steps(2048)
//!     .batch_size(64);
//!
//! let mut agent = Agent::new(config, vec_env, Device::Cpu)?;
//!
//! // Train
//! agent.train(1_000_000, |metrics| {
//!     println!("Step {}: reward = {:.2}", metrics.timesteps, metrics.mean_reward);
//! })?;
//!
//! // Save
//! agent.save("ppo_model.safetensors")?;
//! ```
//!
//! # Algorithm Selection
//!
//! - Use **PPO** for most tasks. It's more sample efficient and stable than A2C.
//! - Use **A2C** when you need faster wall-clock time and have plenty of samples,
//!   or as a simpler debugging baseline.
//!
//! # Key Concepts
//!
//! ## Generalized Advantage Estimation (GAE)
//!
//! Both algorithms use GAE for computing advantages:
//!
//! ```text
//! A_t = delta_t + (gamma * lambda) * A_{t+1}
//! where delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
//! ```
//!
//! ## PPO Clipped Objective
//!
//! PPO uses a clipped surrogate objective to prevent too large policy updates:
//!
//! ```text
//! L_CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]
//! where r_t = pi(a_t|s_t) / pi_old(a_t|s_t)
//! ```
//!
//! ## Total Loss
//!
//! The total loss combines policy, value, and entropy:
//!
//! ```text
//! L = L_policy + c1 * L_value - c2 * H[pi]
//! ```

mod a2c;
mod agent;
mod config;
mod metrics;
mod ppo;
mod rollout;
mod traits;

// Re-exports for public API
pub use a2c::A2CAgent;
pub use agent::{Agent, AgentBuilder, AlgorithmConfig};
pub use config::{A2CConfig, Activation, NetworkConfig, PPOConfig};
pub use metrics::{ProgressTracker, RewardStats, TrainMetrics};
pub use ppo::PPOAgent;
pub use rollout::{BatchSampler, RolloutBuffer, RolloutSample};
pub use traits::{
    ActorCritic, CallbackList, EarlyStoppingCallback, LoggingCallback, NoOpCallback, Policy,
    PolicyDistribution, RLAlgorithm, TrainCallback, ValueFunction,
};

/// Prelude for convenient imports.
pub mod prelude {
    pub use super::{
        A2CConfig, Agent, AgentBuilder, PPOConfig, RLAlgorithm, TrainCallback, TrainMetrics,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_public_exports() {
        // Verify all public types are accessible
        let _ = PPOConfig::default();
        let _ = A2CConfig::default();
        let _ = TrainMetrics::default();
        let _ = RewardStats::new(100);
    }

    #[test]
    fn test_config_builders() {
        let ppo = PPOConfig::new()
            .learning_rate(1e-4)
            .n_steps(1024)
            .batch_size(32)
            .n_epochs(5)
            .gamma(0.995)
            .gae_lambda(0.98)
            .clip_range(0.1)
            .vf_coef(0.25)
            .ent_coef(0.005);

        assert!(ppo.validate().is_ok());
        assert!((ppo.learning_rate - 1e-4).abs() < 1e-8);
        assert_eq!(ppo.n_steps, 1024);

        let a2c = A2CConfig::new()
            .learning_rate(5e-4)
            .n_steps(10)
            .gamma(0.99)
            .gae_lambda(0.95);

        assert!(a2c.validate().is_ok());
        assert_eq!(a2c.n_steps, 10);
    }
}
