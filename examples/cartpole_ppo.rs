//! PPO on the native CartPole-v1 environment (no Python, no gym).
//!
//! Demonstrates that the engine is **not** trading-specific: the exact same
//! `PPOAgent` that trades also balances a pole. Discrete action space.
//!
//! Run with: `cargo run --example cartpole_ppo --release`

use octane_rs::algorithms::{PPOAgent, PPOConfig};
use octane_rs::core::Device;
use octane_rs::envs::CartPole;
use octane_rs::prelude::*;

fn main() -> octane_rs::Result<()> {
    tracing_subscriber::fmt::init();
    let device = Device::cpu();

    let env = CartPole::new();
    println!(
        "Env: {} (obs: {:?}, actions: {})",
        env.name(),
        env.observation_space().shape(),
        env.action_space().n
    );

    // 16 parallel envs; each clone reseeds from entropy (decorrelated).
    let vec_env = env.make_vectorized(16);

    let config = PPOConfig::default()
        .learning_rate(3e-4)
        .n_steps(512)
        .batch_size(64)
        .n_epochs(10)
        .gamma(0.99)
        .gae_lambda(0.95)
        .clip_range(0.2)
        .ent_coef(0.0);

    let mut agent = PPOAgent::new(config, vec_env, device)?;

    let total_timesteps = 100_000;
    println!("\nTraining PPO on CartPole for {total_timesteps} timesteps...\n");
    agent.train(total_timesteps, |m| {
        println!(
            "step {:>7} | mean_reward {:>7.2} ± {:>5.2} | policy_loss {:>7.4} | entropy {:>6.4}",
            m.timesteps, m.mean_reward, m.std_reward, m.policy_loss, m.entropy,
        );
    })?;

    println!("\nDone. A solved CartPole sustains ~500 reward per episode.");
    Ok(())
}
