//! Application state and logic for the Rocket TUI

use std::time::{Duration, Instant};

/// Application mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppMode {
    /// Demo mode with simulated training data
    Demo,
    /// Live training mode (future)
    Training,
    /// Benchmark display mode
    Benchmark,
}

impl Default for AppMode {
    fn default() -> Self {
        Self::Demo
    }
}

/// Navigation tabs
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tab {
    /// Main dashboard with metrics overview
    Dashboard,
    /// Training progress and charts
    Training,
    /// Environment visualization
    Environment,
    /// Performance benchmarks
    Benchmarks,
    /// Configuration settings
    Settings,
}

impl Tab {
    /// Get all tabs in order
    pub fn all() -> &'static [Tab] {
        &[
            Tab::Dashboard,
            Tab::Training,
            Tab::Environment,
            Tab::Benchmarks,
            Tab::Settings,
        ]
    }

    /// Get tab display name
    pub fn name(&self) -> &'static str {
        match self {
            Tab::Dashboard => "Dashboard",
            Tab::Training => "Training",
            Tab::Environment => "Environment",
            Tab::Benchmarks => "Benchmarks",
            Tab::Settings => "Settings",
        }
    }

    /// Get tab index
    pub fn index(&self) -> usize {
        match self {
            Tab::Dashboard => 0,
            Tab::Training => 1,
            Tab::Environment => 2,
            Tab::Benchmarks => 3,
            Tab::Settings => 4,
        }
    }
}

/// Training metrics for display
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// Current episode number
    pub episode: u64,
    /// Total timesteps completed
    pub timesteps: u64,
    /// Current episode reward
    pub episode_reward: f64,
    /// Average reward over last 100 episodes
    pub avg_reward: f64,
    /// Policy loss
    pub policy_loss: f64,
    /// Value loss
    pub value_loss: f64,
    /// Entropy bonus
    pub entropy: f64,
    /// Learning rate
    pub learning_rate: f64,
    /// Steps per second
    pub steps_per_second: f64,
    /// Explained variance
    pub explained_variance: f64,
    /// KL divergence
    pub kl_divergence: f64,
    /// Clip fraction
    pub clip_fraction: f64,
}

/// Reward history entry
#[derive(Debug, Clone)]
pub struct RewardEntry {
    pub episode: u64,
    pub reward: f64,
}

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub name: String,
    pub value: f64,
    pub unit: String,
    pub comparison: Option<f64>, // vs Python baseline
}

/// Application state
pub struct App {
    /// Current application mode
    pub mode: AppMode,
    /// Current active tab
    pub tab: Tab,
    /// Whether to show splash screen
    pub show_splash: bool,
    /// Splash screen start time
    splash_start: Instant,
    /// Splash screen duration
    splash_duration: Duration,
    /// Whether the app should quit
    pub should_quit: bool,
    /// Current training metrics
    pub metrics: TrainingMetrics,
    /// Reward history for plotting
    pub reward_history: Vec<RewardEntry>,
    /// Loss history for plotting
    pub loss_history: Vec<(u64, f64)>,
    /// Benchmark results
    pub benchmarks: Vec<BenchmarkResult>,
    /// Animation frame counter
    pub frame: u64,
    /// Demo tick counter
    demo_tick: u64,
    /// Selected list index (for navigable lists)
    pub selected_index: usize,
    /// Scroll offset for logs
    pub log_scroll: u16,
    /// Training logs
    pub logs: Vec<String>,
    /// Is training paused
    pub paused: bool,
    /// Show help overlay
    pub show_help: bool,
}

impl Default for App {
    fn default() -> Self {
        Self::new(AppMode::Demo)
    }
}

impl App {
    /// Create a new application with the specified mode
    pub fn new(mode: AppMode) -> Self {
        let mut app = Self {
            mode,
            tab: Tab::Dashboard,
            show_splash: true,
            splash_start: Instant::now(),
            splash_duration: Duration::from_secs(2),
            should_quit: false,
            metrics: TrainingMetrics::default(),
            reward_history: Vec::new(),
            loss_history: Vec::new(),
            benchmarks: Vec::new(),
            frame: 0,
            demo_tick: 0,
            selected_index: 0,
            log_scroll: 0,
            logs: Vec::new(),
            paused: false,
            show_help: false,
        };

        // Initialize with demo data if in demo mode
        if mode == AppMode::Demo {
            app.init_demo_data();
        }

        app.init_benchmarks();
        app
    }

    /// Initialize demo training data
    fn init_demo_data(&mut self) {
        self.metrics = TrainingMetrics {
            episode: 0,
            timesteps: 0,
            episode_reward: 0.0,
            avg_reward: 0.0,
            policy_loss: 0.5,
            value_loss: 0.8,
            entropy: 0.6,
            learning_rate: 3e-4,
            steps_per_second: 0.0,
            explained_variance: 0.0,
            kl_divergence: 0.01,
            clip_fraction: 0.1,
        };

        self.logs.push("[INFO] Rocket-RS TUI initialized".to_string());
        self.logs.push("[INFO] Running in demo mode".to_string());
        self.logs.push("[INFO] Press 'h' for help".to_string());
    }

    /// Initialize benchmark data
    fn init_benchmarks(&mut self) {
        self.benchmarks = vec![
            BenchmarkResult {
                name: "PPO Forward Pass".to_string(),
                value: 0.42,
                unit: "ms".to_string(),
                comparison: Some(2.1),
            },
            BenchmarkResult {
                name: "Environment Step".to_string(),
                value: 0.08,
                unit: "ms".to_string(),
                comparison: Some(0.45),
            },
            BenchmarkResult {
                name: "Rollout Collection".to_string(),
                value: 12.5,
                unit: "ms".to_string(),
                comparison: Some(89.3),
            },
            BenchmarkResult {
                name: "GAE Computation".to_string(),
                value: 0.15,
                unit: "ms".to_string(),
                comparison: Some(1.2),
            },
            BenchmarkResult {
                name: "Batch Update".to_string(),
                value: 8.3,
                unit: "ms".to_string(),
                comparison: Some(45.6),
            },
            BenchmarkResult {
                name: "Memory Usage".to_string(),
                value: 128.0,
                unit: "MB".to_string(),
                comparison: Some(512.0),
            },
        ];
    }

    /// Check if splash screen should still be shown
    pub fn update_splash(&mut self) {
        if self.show_splash && self.splash_start.elapsed() >= self.splash_duration {
            self.show_splash = false;
            self.logs.push("[INFO] Starting dashboard...".to_string());
        }
    }

    /// Update on tick (called at 60 FPS)
    pub fn on_tick(&mut self) {
        self.frame = self.frame.wrapping_add(1);
        self.update_splash();

        if self.mode == AppMode::Demo && !self.paused && !self.show_splash {
            self.update_demo();
        }
    }

    /// Update demo simulation
    fn update_demo(&mut self) {
        self.demo_tick += 1;

        // Update every ~10 frames for smoother animation
        if self.demo_tick % 10 != 0 {
            return;
        }

        // Simulate training progress
        self.metrics.timesteps += 256;

        // Episode completion every ~50 ticks
        if self.demo_tick % 50 == 0 {
            self.metrics.episode += 1;

            // Simulate improving rewards with some noise
            let base_reward = -200.0 + (self.metrics.episode as f64 * 2.5).min(400.0);
            let noise = (self.frame as f64 * 0.1).sin() * 20.0;
            self.metrics.episode_reward = base_reward + noise;

            self.reward_history.push(RewardEntry {
                episode: self.metrics.episode,
                reward: self.metrics.episode_reward,
            });

            // Keep history bounded
            if self.reward_history.len() > 200 {
                self.reward_history.remove(0);
            }

            // Update average reward
            if self.reward_history.len() >= 10 {
                let recent: f64 = self.reward_history
                    .iter()
                    .rev()
                    .take(100)
                    .map(|e| e.reward)
                    .sum::<f64>()
                    / self.reward_history.len().min(100) as f64;
                self.metrics.avg_reward = recent;
            }

            // Log milestone episodes
            if self.metrics.episode % 50 == 0 {
                self.logs.push(format!(
                    "[TRAIN] Episode {} | Reward: {:.1} | Avg: {:.1}",
                    self.metrics.episode,
                    self.metrics.episode_reward,
                    self.metrics.avg_reward
                ));
            }
        }

        // Update losses with decay
        let progress = (self.metrics.episode as f64 / 500.0).min(1.0);
        self.metrics.policy_loss = 0.5 * (1.0 - progress * 0.8) + (self.frame as f64 * 0.05).sin() * 0.05;
        self.metrics.value_loss = 0.8 * (1.0 - progress * 0.7) + (self.frame as f64 * 0.03).cos() * 0.08;
        self.metrics.entropy = 0.6 * (1.0 - progress * 0.5);
        self.metrics.explained_variance = progress * 0.85 + (self.frame as f64 * 0.02).sin() * 0.05;
        self.metrics.kl_divergence = 0.01 + (self.frame as f64 * 0.04).sin().abs() * 0.005;
        self.metrics.clip_fraction = 0.1 + (self.frame as f64 * 0.03).cos() * 0.02;

        // Update steps per second
        self.metrics.steps_per_second = 15000.0 + (self.frame as f64 * 0.1).sin() * 1000.0;

        // Record loss history
        if self.demo_tick % 100 == 0 {
            self.loss_history.push((self.metrics.timesteps, self.metrics.policy_loss));
            if self.loss_history.len() > 100 {
                self.loss_history.remove(0);
            }
        }
    }

    /// Navigate to next tab
    pub fn next_tab(&mut self) {
        let tabs = Tab::all();
        let current_idx = self.tab.index();
        let next_idx = (current_idx + 1) % tabs.len();
        self.tab = tabs[next_idx];
    }

    /// Navigate to previous tab
    pub fn prev_tab(&mut self) {
        let tabs = Tab::all();
        let current_idx = self.tab.index();
        let prev_idx = if current_idx == 0 {
            tabs.len() - 1
        } else {
            current_idx - 1
        };
        self.tab = tabs[prev_idx];
    }

    /// Select a specific tab
    pub fn select_tab(&mut self, tab: Tab) {
        self.tab = tab;
    }

    /// Toggle pause state
    pub fn toggle_pause(&mut self) {
        self.paused = !self.paused;
        let status = if self.paused { "paused" } else { "resumed" };
        self.logs.push(format!("[INFO] Training {}", status));
    }

    /// Toggle help overlay
    pub fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    /// Scroll logs up
    pub fn scroll_logs_up(&mut self) {
        self.log_scroll = self.log_scroll.saturating_sub(1);
    }

    /// Scroll logs down
    pub fn scroll_logs_down(&mut self) {
        let max_scroll = self.logs.len().saturating_sub(5) as u16;
        self.log_scroll = (self.log_scroll + 1).min(max_scroll);
    }

    /// Select next item in list
    pub fn select_next(&mut self) {
        self.selected_index = self.selected_index.saturating_add(1);
    }

    /// Select previous item in list
    pub fn select_prev(&mut self) {
        self.selected_index = self.selected_index.saturating_sub(1);
    }

    /// Request application quit
    pub fn quit(&mut self) {
        self.should_quit = true;
    }

    /// Skip splash screen
    pub fn skip_splash(&mut self) {
        self.show_splash = false;
    }

    /// Get speedup ratio for a benchmark (Rust vs Python)
    pub fn get_speedup(&self, benchmark: &BenchmarkResult) -> Option<f64> {
        benchmark.comparison.map(|py| py / benchmark.value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_creation() {
        let app = App::new(AppMode::Demo);
        assert_eq!(app.mode, AppMode::Demo);
        assert!(app.show_splash);
        assert!(!app.should_quit);
    }

    #[test]
    fn test_tab_navigation() {
        let mut app = App::new(AppMode::Demo);
        assert_eq!(app.tab, Tab::Dashboard);

        app.next_tab();
        assert_eq!(app.tab, Tab::Training);

        app.prev_tab();
        assert_eq!(app.tab, Tab::Dashboard);
    }

    #[test]
    fn test_pause_toggle() {
        let mut app = App::new(AppMode::Demo);
        assert!(!app.paused);

        app.toggle_pause();
        assert!(app.paused);

        app.toggle_pause();
        assert!(!app.paused);
    }
}
