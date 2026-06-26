//! Web dashboard for live training visualization.
//!
//! A zero-dependency (std-only HTTP) browser dashboard that runs on localhost
//! on top of the Octane engine. It surfaces:
//!
//! - **Training progress**: timesteps, episodes, progress %, ETA, steps/sec,
//!   reward (mean ± std), losses, entropy, KL, clip fraction, explained variance.
//! - **System load**: real CPU (per-core) and RAM via [`sysinfo`], best-effort
//!   GPU utilization and memory (Apple Silicon `ioreg` / NVIDIA `nvidia-smi`).
//! - **Engine internals**: the hierarchical [`crate::profiling`] breakdown
//!   (rollout / update / forward / backward / env_step …) and a live log feed.
//!
//! # Embedding in your own training loop
//!
//! ```ignore
//! use octane_rs::web::{DashboardState, DashboardServer, DashboardConfig, DashboardMode, spawn_system_monitor};
//! use std::time::Duration;
//!
//! let state = DashboardState::new(DashboardMode::Live);
//! state.set_run_info("PPO", "run-1", "TradingEnv", "Metal", 64, 100_000);
//!
//! // Start the server + system monitor in the background.
//! let server = DashboardServer::new(state.clone(), DashboardConfig::default());
//! println!("Dashboard: {}", server.url());
//! server.spawn()?;
//! spawn_system_monitor(state.clone(), Duration::from_secs(1));
//!
//! // In your training callback, push metrics:
//! agent.train(100_000, |m| state.record_metrics(m, sps))?;
//! # Ok::<(), octane_rs::OctaneError>(())
//! ```

mod assets;
mod server;
mod state;
mod system;

pub use server::{DashboardConfig, DashboardServer};
pub use state::{DashboardMode, DashboardState, History, ProfRow, SystemHistory};
pub use system::{SystemMonitor, SystemSnapshot};

use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Spawn a background thread that refreshes system + profiler metrics into
/// `state` on the given interval. Returns the thread handle.
///
/// The thread runs for the lifetime of the process; the returned handle can be
/// dropped if you don't need to join it.
pub fn spawn_system_monitor(state: DashboardState, interval: Duration) -> JoinHandle<()> {
    thread::spawn(move || {
        let mut monitor = SystemMonitor::new();
        loop {
            let snapshot = monitor.refresh();
            state.set_system(snapshot);
            state.set_profiling_from_global();
            thread::sleep(interval);
        }
    })
}
