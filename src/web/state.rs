//! Shared, thread-safe dashboard state.
//!
//! [`DashboardState`] is a cheap-to-clone handle around an `Arc<Mutex<…>>`.
//! Producers (a training loop, a log tailer, a system monitor) push data in;
//! the HTTP server reads a JSON snapshot out. All locking is internal and
//! non-panicking: if the mutex is ever poisoned the update is silently dropped
//! rather than crashing the process (important under `panic = "abort"`).

use crate::algorithms::{ProgressTracker, TrainMetrics};
use crate::profiling::global_profiler;
use crate::web::system::SystemSnapshot;
use serde::Serialize;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// Maximum number of training-metric points retained for charts.
const HISTORY_CAP: usize = 1024;
/// Maximum number of system-load samples retained for charts.
const SYS_HISTORY_CAP: usize = 600;
/// Maximum number of recent log lines retained.
const LOG_CAP: usize = 400;

/// Data-source mode for the dashboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DashboardMode {
    /// Synthetic, always-alive simulated training data.
    Demo,
    /// Tailing an on-disk training run (`metrics.jsonl`).
    Tail,
    /// Live, embedded training pushing metrics directly.
    Live,
}

impl DashboardMode {
    /// Lowercase string identifier for the JSON payload.
    pub fn as_str(&self) -> &'static str {
        match self {
            DashboardMode::Demo => "demo",
            DashboardMode::Tail => "tail",
            DashboardMode::Live => "live",
        }
    }
}

/// One row of the profiler breakdown.
#[derive(Debug, Clone, Serialize)]
pub struct ProfRow {
    /// Scope name (e.g. "rollout", "update").
    pub name: String,
    /// Total accumulated time, in milliseconds.
    pub total_ms: f64,
    /// Number of times the scope was entered.
    pub count: u64,
    /// Average time per call, in milliseconds.
    pub avg_ms: f64,
    /// Share of total profiled time, in percent.
    pub pct: f64,
}

/// Parallel ring buffers of training metrics over time.
#[derive(Debug, Clone, Default, Serialize)]
pub struct History {
    /// Cumulative timesteps at each sample.
    pub timesteps: Vec<f64>,
    /// Mean episode reward.
    pub mean_reward: Vec<f64>,
    /// Standard deviation of episode reward.
    pub std_reward: Vec<f64>,
    /// Policy loss.
    pub policy_loss: Vec<f64>,
    /// Value loss.
    pub value_loss: Vec<f64>,
    /// Policy entropy.
    pub entropy: Vec<f64>,
    /// Approximate KL divergence.
    pub approx_kl: Vec<f64>,
    /// Clip fraction (PPO).
    pub clip_fraction: Vec<f64>,
    /// Explained variance of the value function.
    pub explained_variance: Vec<f64>,
    /// Learning rate.
    pub learning_rate: Vec<f64>,
    /// Steps per second at each sample.
    pub sps: Vec<f64>,
}

impl History {
    fn push(&mut self, m: &TrainMetrics, sps: f32) {
        self.timesteps.push(m.timesteps as f64);
        self.mean_reward.push(m.mean_reward as f64);
        self.std_reward.push(m.std_reward as f64);
        self.policy_loss.push(m.policy_loss as f64);
        self.value_loss.push(m.value_loss as f64);
        self.entropy.push(m.entropy as f64);
        self.approx_kl.push(m.approx_kl as f64);
        self.clip_fraction.push(m.clip_fraction as f64);
        self.explained_variance.push(m.explained_variance as f64);
        self.learning_rate.push(m.learning_rate as f64);
        self.sps.push(sps as f64);
        self.trim();
    }

    fn trim(&mut self) {
        for v in [
            &mut self.timesteps,
            &mut self.mean_reward,
            &mut self.std_reward,
            &mut self.policy_loss,
            &mut self.value_loss,
            &mut self.entropy,
            &mut self.approx_kl,
            &mut self.clip_fraction,
            &mut self.explained_variance,
            &mut self.learning_rate,
            &mut self.sps,
        ] {
            if v.len() > HISTORY_CAP {
                let excess = v.len() - HISTORY_CAP;
                v.drain(0..excess);
            }
        }
    }
}

/// Time series of system load for sparklines.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SystemHistory {
    /// Relative time of each sample, in seconds since dashboard start.
    pub t: Vec<f64>,
    /// CPU usage percent.
    pub cpu: Vec<f64>,
    /// GPU usage percent (NaN -> null when unavailable).
    pub gpu: Vec<f64>,
    /// Memory usage percent.
    pub mem_pct: Vec<f64>,
    /// GPU memory usage percent.
    pub gpu_mem_pct: Vec<f64>,
}

impl SystemHistory {
    fn push(&mut self, t: f64, cpu: f64, gpu: f64, mem_pct: f64, gpu_mem_pct: f64) {
        self.t.push(t);
        self.cpu.push(cpu);
        self.gpu.push(gpu);
        self.mem_pct.push(mem_pct);
        self.gpu_mem_pct.push(gpu_mem_pct);
        for v in [
            &mut self.t,
            &mut self.cpu,
            &mut self.gpu,
            &mut self.mem_pct,
            &mut self.gpu_mem_pct,
        ] {
            if v.len() > SYS_HISTORY_CAP {
                let excess = v.len() - SYS_HISTORY_CAP;
                v.drain(0..excess);
            }
        }
    }
}

/// Static metadata describing the run being visualized.
#[derive(Debug, Clone, Default)]
struct RunMeta {
    algorithm: String,
    run_id: String,
    env_name: String,
    device: String,
    num_envs: usize,
    total_timesteps: usize,
}

/// Interior mutable state guarded by a mutex.
struct Inner {
    mode: DashboardMode,
    meta: RunMeta,
    progress: ProgressTracker,
    latest: TrainMetrics,
    last_sps: f32,
    history: History,
    system: SystemSnapshot,
    system_history: SystemHistory,
    profiling: Vec<ProfRow>,
    logs: VecDeque<String>,
    active: bool,
    paused: bool,
    has_metrics: bool,
    started: Instant,
}

/// A cheap-to-clone handle to the shared dashboard state.
#[derive(Clone)]
pub struct DashboardState {
    inner: Arc<Mutex<Inner>>,
}

impl DashboardState {
    /// Create a new dashboard state for the given data-source mode.
    pub fn new(mode: DashboardMode) -> Self {
        let inner = Inner {
            mode,
            meta: RunMeta::default(),
            progress: ProgressTracker::new(0),
            latest: TrainMetrics::default(),
            last_sps: 0.0,
            history: History::default(),
            system: SystemSnapshot::default(),
            system_history: SystemHistory::default(),
            profiling: Vec::new(),
            logs: VecDeque::new(),
            active: false,
            paused: false,
            has_metrics: false,
            started: Instant::now(),
        };
        Self {
            inner: Arc::new(Mutex::new(inner)),
        }
    }

    /// Set static run metadata (algorithm, environment, device, etc.).
    ///
    /// Call once before training starts. This (re)initializes the progress
    /// tracker with the total timestep target.
    #[allow(clippy::too_many_arguments)]
    pub fn set_run_info(
        &self,
        algorithm: impl Into<String>,
        run_id: impl Into<String>,
        env_name: impl Into<String>,
        device: impl Into<String>,
        num_envs: usize,
        total_timesteps: usize,
    ) {
        if let Ok(mut g) = self.inner.lock() {
            g.meta = RunMeta {
                algorithm: algorithm.into(),
                run_id: run_id.into(),
                env_name: env_name.into(),
                device: device.into(),
                num_envs,
                total_timesteps,
            };
            g.progress = ProgressTracker::new(total_timesteps);
        }
    }

    /// Record a training-metrics sample with the current steps-per-second.
    pub fn record_metrics(&self, metrics: &TrainMetrics, sps: f32) {
        if let Ok(mut g) = self.inner.lock() {
            g.latest = metrics.clone();
            g.last_sps = sps;
            g.has_metrics = true;
            g.active = true;
            g.progress.update(metrics);
            g.history.push(metrics, sps);
        }
    }

    /// Replace the current system snapshot and append to the load history.
    pub fn set_system(&self, snap: SystemSnapshot) {
        if let Ok(mut g) = self.inner.lock() {
            let t = g.started.elapsed().as_secs_f64();
            let gpu = if snap.gpu_usage_available {
                snap.gpu_usage as f64
            } else {
                f64::NAN
            };
            let gpu_mem_pct = if snap.gpu_mem_total_mb > 0.0 {
                snap.gpu_mem_used_mb / snap.gpu_mem_total_mb * 100.0
            } else {
                0.0
            };
            g.system_history.push(
                t,
                snap.cpu_usage as f64,
                gpu,
                snap.mem_used_pct as f64,
                gpu_mem_pct,
            );
            g.system = snap;
        }
    }

    /// Replace the profiler breakdown rows.
    pub fn set_profiling(&self, rows: Vec<ProfRow>) {
        if let Ok(mut g) = self.inner.lock() {
            g.profiling = rows;
        }
    }

    /// Refresh the profiler breakdown from the global profiler.
    pub fn set_profiling_from_global(&self) {
        let report = global_profiler().report();
        let total = report.total_time;
        let rows: Vec<ProfRow> = report
            .entries
            .iter()
            .map(|s| ProfRow {
                name: s.name.clone(),
                total_ms: s.total_time.as_secs_f64() * 1000.0,
                count: s.call_count,
                avg_ms: s.avg_time().as_secs_f64() * 1000.0,
                pct: s.percentage(total),
            })
            .collect();
        self.set_profiling(rows);
    }

    /// Append a log line (bounded ring buffer).
    pub fn push_log(&self, line: impl Into<String>) {
        if let Ok(mut g) = self.inner.lock() {
            g.logs.push_back(line.into());
            while g.logs.len() > LOG_CAP {
                g.logs.pop_front();
            }
        }
    }

    /// Set whether training is currently active.
    pub fn set_active(&self, active: bool) {
        if let Ok(mut g) = self.inner.lock() {
            g.active = active;
        }
    }

    /// Set whether training is paused.
    pub fn set_paused(&self, paused: bool) {
        if let Ok(mut g) = self.inner.lock() {
            g.paused = paused;
        }
    }

    /// Serialize just the system snapshot as JSON.
    pub fn system_json(&self) -> String {
        match self.inner.lock() {
            Ok(g) => serde_json::to_string(&g.system).unwrap_or_else(|_| "{}".to_string()),
            Err(_) => "{}".to_string(),
        }
    }

    /// Serialize the full dashboard state as a JSON document.
    pub fn to_json(&self) -> String {
        match self.inner.lock() {
            Ok(g) => {
                // When the total is unknown (e.g. tail mode with no info.json),
                // report indeterminate progress (0 / null ETA) instead of a fake
                // 100%. Otherwise clamp to [0,1] and keep ETA non-negative even if
                // timesteps overshoot the target on the final sample.
                let total_known = g.meta.total_timesteps > 0;
                let progress = if total_known {
                    g.progress.progress().clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let eta = if total_known {
                    g.progress.eta_secs().max(0.0)
                } else {
                    f64::INFINITY // serializes to JSON null
                };
                // NEG_INFINITY (no episode yet) serializes to null; the frontend
                // gates best_reward on has_data anyway.
                let best_reward = g.progress.best_reward();
                let sps = if g.last_sps > 0.0 {
                    g.last_sps
                } else {
                    g.progress.fps()
                };
                let logs: Vec<&str> = g.logs.iter().map(|s| s.as_str()).collect();

                let dto = StateDto {
                    server: ServerDto {
                        version: env!("CARGO_PKG_VERSION"),
                        mode: g.mode.as_str(),
                        uptime_secs: g.started.elapsed().as_secs_f64(),
                    },
                    training: TrainingDto {
                        active: g.active,
                        paused: g.paused,
                        algorithm: &g.meta.algorithm,
                        run_id: &g.meta.run_id,
                        env_name: &g.meta.env_name,
                        device: &g.meta.device,
                        num_envs: g.meta.num_envs,
                        progress,
                        timesteps: g.latest.timesteps,
                        total_timesteps: g.meta.total_timesteps,
                        episodes: g.latest.episodes,
                        elapsed_secs: g.progress.elapsed_secs(),
                        eta_secs: eta,
                        sps,
                        best_reward,
                        has_data: g.has_metrics,
                        latest: &g.latest,
                    },
                    system: &g.system,
                    history: &g.history,
                    system_history: &g.system_history,
                    profiling: &g.profiling,
                    logs,
                };
                serde_json::to_string(&dto).unwrap_or_else(|_| "{}".to_string())
            }
            Err(_) => "{}".to_string(),
        }
    }
}

/// Top-level serialized payload.
#[derive(Serialize)]
struct StateDto<'a> {
    server: ServerDto<'a>,
    training: TrainingDto<'a>,
    system: &'a SystemSnapshot,
    history: &'a History,
    system_history: &'a SystemHistory,
    profiling: &'a [ProfRow],
    logs: Vec<&'a str>,
}

/// Server/runtime metadata block.
#[derive(Serialize)]
struct ServerDto<'a> {
    version: &'a str,
    mode: &'a str,
    uptime_secs: f64,
}

/// Training status and latest-metrics block.
#[derive(Serialize)]
struct TrainingDto<'a> {
    active: bool,
    paused: bool,
    algorithm: &'a str,
    run_id: &'a str,
    env_name: &'a str,
    device: &'a str,
    num_envs: usize,
    progress: f32,
    timesteps: usize,
    total_timesteps: usize,
    episodes: usize,
    elapsed_secs: f64,
    eta_secs: f64,
    sps: f32,
    best_reward: f32,
    has_data: bool,
    latest: &'a TrainMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_and_serializes() {
        let state = DashboardState::new(DashboardMode::Live);
        state.set_run_info("PPO", "run-1", "TradingEnv", "CPU", 64, 1000);
        let m = TrainMetrics {
            timesteps: 500,
            mean_reward: 12.5,
            ..Default::default()
        };
        state.record_metrics(&m, 15000.0);
        let json = state.to_json();
        assert!(json.contains("\"algorithm\":\"PPO\""));
        assert!(json.contains("\"timesteps\":500"));
        assert!(json.contains("\"mode\":\"live\""));
        // progress = 500 / 1000
        assert!(json.contains("\"progress\":0.5"));
    }

    #[test]
    fn history_is_bounded() {
        let state = DashboardState::new(DashboardMode::Demo);
        for i in 0..(HISTORY_CAP + 50) {
            let m = TrainMetrics {
                timesteps: i,
                ..Default::default()
            };
            state.record_metrics(&m, 1.0);
        }
        let len = state
            .inner
            .lock()
            .map(|g| g.history.timesteps.len())
            .unwrap_or(0);
        assert_eq!(len, HISTORY_CAP);
    }
}
