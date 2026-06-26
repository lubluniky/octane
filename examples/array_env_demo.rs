//! Train on an arbitrary dataset via `ArrayEnv` — the "any data" path.
//!
//! No trading assumptions: we synthesize a feature matrix and a target that is
//! a fixed linear function of the features, then train PPO to predict it. The
//! reward is `-MSE(action, target)`, so it climbs toward 0 as the agent learns
//! the mapping. Swap in your own `Vec<f32>` matrix to use real data.
//!
//! Run with: `cargo run --example array_env_demo --release`

use octane_rs::algorithms::{PPOAgent, PPOConfig};
use octane_rs::core::Device;
use octane_rs::envs::{ArrayEnv, ArrayReward};
use octane_rs::prelude::*;

fn main() -> octane_rs::Result<()> {
    tracing_subscriber::fmt::init();
    let device = Device::cpu();

    // Synthesize [n_rows, obs_dim] features and a linear target.
    let n_rows = 4096;
    let obs_dim = 3;
    let weights = [0.5_f32, -0.3, 0.2];
    let mut data = Vec::with_capacity(n_rows * obs_dim);
    let mut targets = Vec::with_capacity(n_rows);
    for t in 0..n_rows {
        let x = t as f32 * 0.01;
        let feats = [x.sin(), (2.0 * x).cos(), (0.5 * x).sin()];
        let y: f32 = feats.iter().zip(&weights).map(|(f, w)| f * w).sum();
        data.extend_from_slice(&feats);
        targets.push(y);
    }

    let env = ArrayEnv::new(
        data,
        obs_dim,
        ArrayReward::Regression {
            targets,
            target_dim: 1,
        },
    )?
    .with_episode_len(64)
    .with_random_start(true);

    println!(
        "Env: {} (obs_dim: {}, act_dim: {})",
        env.name(),
        env.obs_dim(),
        env.act_dim()
    );

    let vec_env = env.make_vectorized(16);

    let config = PPOConfig::default()
        .learning_rate(3e-4)
        .n_steps(256)
        .batch_size(64)
        .n_epochs(10)
        .ent_coef(0.0);

    let mut agent = PPOAgent::new(config, vec_env, device)?;

    let total_timesteps = 100_000;
    println!("\nTraining PPO to regress the target for {total_timesteps} timesteps...\n");
    agent.train(total_timesteps, |m| {
        println!(
            "step {:>7} | mean_reward (-MSE) {:>9.4} | policy_loss {:>8.4}",
            m.timesteps, m.mean_reward, m.policy_loss,
        );
    })?;

    println!("\nDone. Reward (-MSE) climbs toward 0 as the linear mapping is learned.");
    Ok(())
}
