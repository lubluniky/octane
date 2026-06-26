//! Live web dashboard on top of a real PPO training run.
//!
//! This is the "on top of the engine" proof: it starts the [`DashboardServer`]
//! and runs a real (small) PPO training on a vectorized trading environment,
//! pushing live [`TrainMetrics`] into the shared dashboard state from the
//! training callback. Open the printed URL in your browser to watch progress,
//! CPU/GPU/RAM load, the engine profiler, and reward/loss charts update live.
//!
//! Run with:
//!   cargo run --example web_dashboard --release
//!   cargo run --example web_dashboard --release --features metal   # Apple GPU

use octane_rs::algorithms::{PPOAgent, PPOConfig};
use octane_rs::core::Device;
use octane_rs::envs::{MarketData, TradingEnv};
use octane_rs::prelude::*;
use octane_rs::web::{
    spawn_system_monitor, DashboardConfig, DashboardMode, DashboardServer, DashboardState,
};
use std::time::{Duration, Instant};

fn main() -> octane_rs::Result<()> {
    tracing_subscriber::fmt::init();

    // 1. Device selection (Metal / CUDA / CPU).
    #[cfg(feature = "metal")]
    let device = Device::m4_metal();
    #[cfg(all(not(feature = "metal"), feature = "cuda"))]
    let device = Device::cuda(0);
    #[cfg(not(any(feature = "metal", feature = "cuda")))]
    let device = Device::cpu();

    // 2. Build a vectorized synthetic trading environment.
    let data = MarketData::synthetic(10_000, 42);
    let env = TradingEnv::new(data)?;
    let env_name = env.name().to_string();
    let num_envs = 64usize;
    let vec_env = env.make_vectorized(num_envs);

    // 3. Configure PPO.
    let total_timesteps = 200_000usize;
    let config = PPOConfig::default()
        .learning_rate(3e-4)
        .n_steps(512)
        .batch_size(64)
        .n_epochs(4)
        .gamma(0.99)
        .gae_lambda(0.95)
        .clip_range(0.2)
        .ent_coef(0.01);
    let mut agent = PPOAgent::new(config, vec_env, device)?;

    // 4. Wire up the dashboard state.
    let state = DashboardState::new(DashboardMode::Live);
    state.set_run_info(
        "PPO",
        "ppo-live",
        env_name,
        format!("{device}"),
        num_envs,
        total_timesteps,
    );

    // 5. Start the HTTP server + system monitor in the background.
    let server = DashboardServer::new(state.clone(), DashboardConfig::default());
    let url = server.url();
    server.spawn()?;
    spawn_system_monitor(state.clone(), Duration::from_secs(1));

    println!("\n  ▸ Octane dashboard live at  {url}");
    println!("  ▸ Training {total_timesteps} timesteps on {device} — open the URL to watch.\n");
    state.set_active(true);
    state.push_log(format!("[INFO] Live PPO training started on {device}"));

    // 6. Train, pushing metrics + an instantaneous steps/sec estimate.
    let mut last_ts = 0usize;
    let mut last_clock = Instant::now();
    let result = agent.train(total_timesteps, |m| {
        let dt = last_clock.elapsed().as_secs_f32();
        let sps = if dt > 0.0 && m.timesteps >= last_ts {
            (m.timesteps - last_ts) as f32 / dt
        } else {
            0.0
        };
        last_ts = m.timesteps;
        last_clock = Instant::now();
        state.record_metrics(m, sps);
    });

    // Keep the dashboard alive whatever happens to training, so system load and
    // collected metrics remain inspectable.
    state.set_active(false);
    match result {
        Ok(()) => {
            state.push_log("[INFO] Training complete — dashboard stays live (Ctrl-C to exit)");
            println!("\n  ✓ Training complete. Dashboard still live at {url} (Ctrl-C to exit).\n");
        }
        Err(e) => {
            state.push_log(format!("[WARN] Training stopped: {e}"));
            eprintln!(
                "\n  ! Training error: {e}\n  Dashboard stays live at {url} (Ctrl-C to exit).\n"
            );
        }
    }

    // 7. Keep the server alive so results remain inspectable.
    loop {
        std::thread::sleep(Duration::from_secs(3600));
    }
}
