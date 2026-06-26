//! Octane Web Dashboard
//!
//! A localhost browser dashboard for the Octane RL engine: live training
//! progress, CPU/GPU/RAM load, the engine profiler breakdown, and a log feed.
//!
//! Usage:
//!   cargo run --bin octane-web                 # synthetic demo (always alive)
//!   cargo run --bin octane-web -- --port 9000  # custom port
//!   cargo run --bin octane-web -- --logdir runs/   # tail a real training run
//!   cargo run --bin octane-web -- --open       # open the browser automatically
//!
//! To stream a *real* training run, point `--logdir` at the directory you passed
//! to `TrainingLogger`, or embed `DashboardServer` directly (see
//! `examples/web_dashboard.rs`).

use octane_rs::logging::{TrainingLogReader, TrainingRunInfo};
use octane_rs::profiling::global_profiler;
use octane_rs::web::{
    spawn_system_monitor, DashboardConfig, DashboardMode, DashboardServer, DashboardState,
};
use octane_rs::TrainMetrics;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::{Duration, Instant};

/// Parsed command-line options.
struct Args {
    host: String,
    port: u16,
    logdir: Option<PathBuf>,
    open: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 7878,
            logdir: None,
            open: false,
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--host" => {
                if let Some(v) = it.next() {
                    args.host = v;
                }
            }
            "--port" => {
                if let Some(v) = it.next() {
                    if let Ok(p) = v.parse() {
                        args.port = p;
                    }
                }
            }
            "--logdir" => {
                if let Some(v) = it.next() {
                    args.logdir = Some(PathBuf::from(v));
                }
            }
            "--open" => args.open = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ => {}
        }
    }
    args
}

fn print_help() {
    println!(
        "Octane Web Dashboard\n\n\
         OPTIONS:\n\
         \x20 --host <HOST>     Bind host (default 127.0.0.1)\n\
         \x20 --port <PORT>     Bind port (default 7878)\n\
         \x20 --logdir <DIR>    Tail the latest training run under DIR (metrics.jsonl)\n\
         \x20 --open            Open the dashboard in your browser\n\
         \x20 --help            Show this help"
    );
}

fn main() {
    let args = parse_args();

    let mode = if args.logdir.is_some() {
        DashboardMode::Tail
    } else {
        DashboardMode::Demo
    };
    let state = DashboardState::new(mode);

    // Background: real system + profiler metrics every second.
    spawn_system_monitor(state.clone(), Duration::from_secs(1));

    // Background: training data source.
    match &args.logdir {
        Some(dir) => start_tailer(state.clone(), dir.clone()),
        None => start_demo(state.clone()),
    }

    let config = DashboardConfig {
        host: args.host.clone(),
        port: args.port,
    };
    let server = DashboardServer::new(state, config);
    let url = server.url();

    println!("\n  ╔══════════════════════════════════════════════╗");
    println!("  ║   OCTANE · Live Web Dashboard                  ║");
    println!("  ╚══════════════════════════════════════════════╝\n");
    println!("  ▸ Mode:   {}", mode_label(mode));
    println!("  ▸ Serving at  {url}");
    println!("  ▸ Press Ctrl-C to stop.\n");

    if !is_loopback(&args.host) {
        eprintln!(
            "  ⚠ WARNING: bound to non-loopback host '{}'. The dashboard is unauthenticated and\n    \
             exposes system + training telemetry to other machines on the network.\n",
            args.host
        );
    }

    if args.open {
        open_browser(&url);
    }

    if let Err(e) = server.serve() {
        eprintln!(
            "  ✗ Failed to start server on {}:{} — {e}",
            args.host, args.port
        );
        eprintln!("    (is the port already in use? try --port <N>)");
        std::process::exit(1);
    }
}

fn mode_label(mode: DashboardMode) -> &'static str {
    match mode {
        DashboardMode::Demo => "DEMO (synthetic data)",
        DashboardMode::Tail => "TAIL (live log run)",
        DashboardMode::Live => "LIVE (embedded training)",
    }
}

/// Whether `host` is a loopback address (safe, local-only bind).
fn is_loopback(host: &str) -> bool {
    host == "localhost"
        || host == "::1"
        || host
            .parse::<std::net::IpAddr>()
            .map(|ip| ip.is_loopback())
            .unwrap_or(false)
}

/// Open `url` in the system browser (best-effort).
fn open_browser(url: &str) {
    let cmd = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(target_os = "windows") {
        "explorer"
    } else {
        "xdg-open"
    };
    let _ = Command::new(cmd).arg(url).spawn();
}

// ---------------------------------------------------------------------------
// Demo data source — a smooth, always-alive synthetic training simulation.
// Strictly separated from real data: only used in Demo mode.
// ---------------------------------------------------------------------------

fn start_demo(state: DashboardState) {
    thread::spawn(move || {
        let mut run = 0u64;
        loop {
            run += 1;
            run_demo_episode(&state, run);
        }
    });
}

fn run_demo_episode(state: &DashboardState, run: u64) {
    use rand::Rng;
    let total: usize = 1_000_000;
    let num_envs = 128usize;
    let rollout = num_envs * 64; // 8192 steps / iteration
    state.set_run_info(
        "PPO",
        format!("demo-{run:03}"),
        "TradingEnv (synthetic)",
        "Metal (Apple Silicon)",
        num_envs,
        total,
    );
    state.set_active(true);
    state.push_log(format!("[INFO] Starting synthetic PPO run demo-{run:03}"));

    let mut rng = rand::thread_rng();
    let mut timesteps = 0usize;
    let mut episodes = 0usize;
    let mut frame = 0u64;

    while timesteps < total {
        frame += 1;
        timesteps = (timesteps + rollout).min(total); // never overshoot the target
        episodes += rng.gen_range(2..6);

        // smooth improving reward with noise (logistic ramp from -180 -> ~260)
        let p = timesteps as f64 / total as f64;
        let base = -180.0 + 440.0 * (1.0 / (1.0 + (-6.0 * (p - 0.35)).exp()));
        let noise = rng.gen_range(-22.0..22.0);
        let mean_reward = (base + noise) as f32;
        let std_reward = (45.0 * (1.0 - p) + 8.0) as f32;

        let decay = (1.0 - 0.8 * p) as f32;
        let m = TrainMetrics {
            policy_loss: 0.5 * decay + rng.gen_range(-0.03..0.03),
            value_loss: 0.85 * decay + rng.gen_range(-0.05..0.05),
            entropy: (0.95 * (1.0 - 0.6 * p) as f32) + rng.gen_range(-0.02..0.02),
            approx_kl: (0.008 + 0.01 * rng.gen_range(0.0f32..1.0)).abs(),
            clip_fraction: 0.12 + rng.gen_range(-0.03..0.04),
            explained_variance: (p as f32 * 0.9 + rng.gen_range(-0.04..0.04)).clamp(0.0, 0.99),
            learning_rate: 3e-4 * (1.0 - p as f32 * 0.9),
            timesteps,
            episodes,
            mean_reward,
            std_reward,
        };
        let sps = 28000.0 + rng.gen_range(-3500.0..3500.0);
        state.record_metrics(&m, sps);

        // Synthetic engine profiler breakdown so the panel is lively in demo mode.
        let prof = global_profiler();
        let ms = |v: f64| Duration::from_micros((v.max(0.0) * 1000.0) as u64);
        prof.record("env_step", ms(3.0 + rng.gen_range(-0.4..0.4)));
        prof.record("forward", ms(4.2 + rng.gen_range(-0.5..0.5)));
        prof.record("backward", ms(5.1 + rng.gen_range(-0.6..0.6)));
        prof.record("compute_gae", ms(1.1 + rng.gen_range(-0.2..0.2)));
        prof.record("rollout", ms(8.0 + rng.gen_range(-0.8..0.8)));
        prof.record("update", ms(11.0 + rng.gen_range(-1.0..1.0)));

        if frame.is_multiple_of(12) {
            state.push_log(format!(
                "[TRAIN] step {timesteps:>9} | reward {mean_reward:>7.1} ± {std_reward:>5.1} | ploss {:.3} | ev {:.2}",
                m.policy_loss, m.explained_variance
            ));
        }

        thread::sleep(Duration::from_millis(220));
    }

    state.push_log(format!(
        "[INFO] Run demo-{run:03} reached {total} timesteps — restarting"
    ));
    state.set_active(false);
    thread::sleep(Duration::from_millis(1500));
}

// ---------------------------------------------------------------------------
// Tail data source — read a real on-disk run produced by `TrainingLogger`.
// ---------------------------------------------------------------------------

fn start_tailer(state: DashboardState, dir: PathBuf) {
    thread::spawn(move || {
        // Wait for a run to appear.
        let run_dir = loop {
            match find_latest_run(&dir) {
                Some(d) => break d,
                None => {
                    state.push_log(format!(
                        "[INFO] No runs found under {} yet — waiting…",
                        dir.display()
                    ));
                    thread::sleep(Duration::from_secs(2));
                }
            }
        };

        let mut reader = match TrainingLogReader::new(&run_dir) {
            Ok(r) => r,
            Err(e) => {
                state.push_log(format!(
                    "[WARN] Failed to open run {}: {e}",
                    run_dir.display()
                ));
                return;
            }
        };

        if let Some(info) = reader.run_info().cloned() {
            apply_run_info(&state, &info);
            state.push_log(format!(
                "[INFO] Tailing run '{}' ({} on {})",
                info.run_id, info.algorithm, info.device
            ));
        }
        state.set_active(true);

        // sps estimation across polls
        let mut last_ts = 0usize;
        let mut last_clock = Instant::now();
        // The reader caches info.json once, so it can't observe a run finishing.
        // Fall back to an idle timeout: no new entries for ~10s => mark inactive.
        let mut idle_polls = 0u32;
        const IDLE_LIMIT: u32 = 20;

        loop {
            let new_count = match reader.read_new() {
                Ok(entries) => {
                    let n = entries.len();
                    for entry in entries {
                        let mut m = TrainMetrics {
                            policy_loss: entry.policy_loss,
                            value_loss: entry.value_loss,
                            entropy: entry.entropy,
                            learning_rate: entry.learning_rate,
                            timesteps: entry.timestep,
                            episodes: entry.episode,
                            mean_reward: entry.mean_reward,
                            std_reward: entry.std_reward,
                            ..Default::default()
                        };
                        m.approx_kl = *entry.extra.get("approx_kl").unwrap_or(&0.0);
                        m.clip_fraction = *entry.extra.get("clip_fraction").unwrap_or(&0.0);
                        m.explained_variance =
                            *entry.extra.get("explained_variance").unwrap_or(&0.0);

                        let mut sps = entry.steps_per_second;
                        if sps <= 0.0 {
                            let dt = last_clock.elapsed().as_secs_f32();
                            if dt > 0.0 && entry.timestep >= last_ts {
                                sps = (entry.timestep - last_ts) as f32 / dt;
                            }
                        }
                        last_ts = entry.timestep;
                        last_clock = Instant::now();
                        state.record_metrics(&m, sps);
                    }
                    n
                }
                Err(e) => {
                    state.push_log(format!("[WARN] Read error: {e}"));
                    0
                }
            };

            if reader.is_complete() {
                state.set_active(false);
            } else if new_count > 0 {
                idle_polls = 0;
            } else {
                idle_polls = idle_polls.saturating_add(1);
                if idle_polls == IDLE_LIMIT {
                    state.set_active(false);
                    state.push_log("[INFO] No new metrics for 10s — marking run idle");
                }
            }
            thread::sleep(Duration::from_millis(500));
        }
    });
}

fn apply_run_info(state: &DashboardState, info: &TrainingRunInfo) {
    state.set_run_info(
        info.algorithm.clone(),
        info.run_id.clone(),
        info.environment.clone(),
        info.device.clone(),
        0,
        info.total_timesteps,
    );
}

/// Find the most-recently-modified run directory (one containing `metrics.jsonl`)
/// directly under `root`, or `root` itself if it is a run directory.
fn find_latest_run(root: &Path) -> Option<PathBuf> {
    if root.join("metrics.jsonl").is_file() {
        return Some(root.to_path_buf());
    }
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    let entries = std::fs::read_dir(root).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        let metrics = path.join("metrics.jsonl");
        if metrics.is_file() {
            let modified = metrics
                .metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::UNIX_EPOCH);
            match &best {
                Some((t, _)) if *t >= modified => {}
                _ => best = Some((modified, path)),
            }
        }
    }
    best.map(|(_, p)| p)
}
