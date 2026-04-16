//! Octane Trading Example
//!
//! Demonstrates the ideal user API for training a PPO agent
//! on a vectorized trading environment.
//!
//! Run with: cargo run --example trading_ppo --release

use octane_rs::algorithms::{PPOAgent, PPOConfig, RLAlgorithm};
use octane_rs::core::Device;
use octane_rs::envs::{MarketData, TradingEnv};
use octane_rs::prelude::*;
use std::path::Path;

fn main() -> octane_rs::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();

    // ==========================================================
    // IDEAL USER API (as specified in requirements)
    // ==========================================================

    // 1. Configure device (Apple Silicon M4 Metal or CUDA)
    #[cfg(feature = "metal")]
    let device = Device::m4_metal();

    #[cfg(feature = "cuda")]
    let device = Device::cuda(0);

    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let device = Device::cpu();

    println!("Using device: {device}");

    // 2. Create market data (synthetic for demo)
    let data = MarketData::synthetic(10000, 42);
    println!("Loaded {} timesteps of market data", data.len());

    // 3. Create trading environment
    let env = TradingEnv::new(data)?;
    println!(
        "Environment: {} (obs: {:?}, act: {:?})",
        env.name(),
        env.observation_space().shape(),
        env.action_space().shape()
    );

    // 4. Vectorize for parallel simulation (128 parallel envs)
    let num_envs = 128;
    let vec_env = env.make_vectorized(num_envs);
    println!("Vectorized to {num_envs} parallel environments");

    // 5. Configure PPO with optimal hyperparameters for trading
    let config = PPOConfig::default()
        .learning_rate(3e-4)
        .n_steps(2048)
        .batch_size(64)
        .n_epochs(10)
        .gamma(0.99)
        .gae_lambda(0.95)
        .clip_range(0.2)
        .vf_coef(0.5)
        .ent_coef(0.01)
        .max_grad_norm(0.5);

    println!("PPO Config: {config:?}");

    // 6. Create agent with environment and device
    let mut agent = PPOAgent::new(config, vec_env, device)?;

    // 7. Train with callback for metrics
    let total_timesteps = 100_000;
    println!("\nStarting training for {total_timesteps} timesteps...\n");

    agent.train(total_timesteps, |metrics| {
        println!(
            "Step {:>8} | Reward: {:>8.2} ± {:>6.2} | Policy Loss: {:>8.4} | Value Loss: {:>8.4} | Entropy: {:>6.4}",
            metrics.timesteps,
            metrics.mean_reward,
            metrics.std_reward,
            metrics.policy_loss,
            metrics.value_loss,
            metrics.entropy,
        );
    })?;

    // 8. Save trained model
    agent.save(Path::new("trading_ppo_model.safetensors"))?;
    println!("\nModel saved to trading_ppo_model.safetensors");

    Ok(())
}
