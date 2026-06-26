//! SAC on the native Pendulum-v1 environment (continuous control, no gym).
//!
//! The same off-policy `SACAgent` used for continuous trading swings up a
//! pendulum. Continuous action space (torque in [-2, 2]).
//!
//! Run with: `cargo run --example pendulum_sac --release`

use octane_rs::algorithms::{SACAgent, SACConfig};
use octane_rs::core::Device;
use octane_rs::envs::Pendulum;
use octane_rs::prelude::*;

fn main() -> octane_rs::Result<()> {
    tracing_subscriber::fmt::init();
    let device = Device::cpu();

    let env = Pendulum::new();
    println!(
        "Env: {} (obs: {:?}, act: {:?})",
        env.name(),
        env.observation_space().shape(),
        env.action_space().shape()
    );

    let vec_env = env.make_vectorized(1);

    let config = SACConfig {
        learning_rate: 3e-4,
        buffer_size: 50_000,
        learning_starts: 1_000,
        batch_size: 256,
        gamma: 0.99,
        tau: 0.005,
        policy_hidden_sizes: vec![256, 256],
        q_hidden_sizes: vec![256, 256],
        auto_entropy_tuning: true,
        ..Default::default()
    };

    let mut agent = SACAgent::new(config, vec_env, device)?;

    let total_timesteps = 20_000;
    println!("\nTraining SAC on Pendulum for {total_timesteps} timesteps...\n");
    agent.train(total_timesteps, |m| {
        println!(
            "step {:>7} | mean_reward {:>8.2} | policy_loss {:>8.4} | q_loss {:>8.4}",
            m.timesteps, m.mean_reward, m.policy_loss, m.value_loss,
        );
    })?;

    println!("\nDone. Pendulum reward approaches 0 (negative cost) as it learns to balance.");
    Ok(())
}
