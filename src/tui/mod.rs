//! Terminal User Interface for Octane.
//!
//! A professional TUI built with Ratatui for visualizing training,
//! benchmarks, and metrics in real-time.

pub mod screens;
pub mod theme;
pub mod widgets;

pub use theme::Theme;

use crate::algorithms::TrainMetrics;
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Tabs},
    Frame, Terminal,
};
use std::{
    io::{self, Stdout},
    process::Command,
    time::{Duration, Instant},
};
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

use screens::Screen;

/// Application mode - determines data source for metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AppMode {
    /// Demo mode with simulated data.
    #[default]
    Demo,
    /// Benchmark mode with real system metrics.
    Benchmark,
    /// Training mode with live model training.
    Training,
}

impl AppMode {
    /// Get display name for the mode.
    pub fn name(&self) -> &'static str {
        match self {
            AppMode::Demo => "DEMO",
            AppMode::Benchmark => "BENCHMARK",
            AppMode::Training => "TRAINING",
        }
    }

    /// Get color for the mode indicator.
    pub fn color(&self) -> Color {
        match self {
            AppMode::Demo => Color::Yellow,
            AppMode::Benchmark => Color::Cyan,
            AppMode::Training => Color::Green,
        }
    }

    /// Cycle to next mode.
    pub fn next(&self) -> Self {
        match self {
            AppMode::Demo => AppMode::Benchmark,
            AppMode::Benchmark => AppMode::Training,
            AppMode::Training => AppMode::Demo,
        }
    }
}

/// Application state for the TUI.
pub struct App {
    /// Current active screen/tab.
    pub current_screen: Screen,
    /// Current application mode.
    pub mode: AppMode,
    /// Whether the application should quit.
    pub should_quit: bool,
    /// Animation tick counter for visual effects.
    pub tick: u64,
    /// Training metrics history for visualization.
    pub metrics_history: Vec<TrainMetrics>,
    /// Current training metrics.
    pub current_metrics: TrainMetrics,
    /// Benchmark data: (name, rust_value, python_value).
    pub benchmark_data: Vec<(String, f64, f64)>,
    /// System metrics: CPU usage, GPU usage, memory.
    pub system_metrics: SystemMetrics,
    /// Recent rewards for sparkline.
    pub reward_history: Vec<u64>,
    /// Training is active.
    pub training_active: bool,
    /// Steps per second.
    pub steps_per_second: f64,
    /// System info for real metrics.
    sys: System,
    /// Last CPU refresh time.
    last_cpu_refresh: Instant,
}

/// System resource metrics.
#[derive(Debug, Clone, Default)]
pub struct SystemMetrics {
    /// CPU usage percentage (0-100).
    pub cpu_usage: f32,
    /// GPU usage percentage (0-100).
    pub gpu_usage: f32,
    /// Memory usage in MB.
    pub memory_used_mb: f64,
    /// Total memory in MB.
    pub memory_total_mb: f64,
    /// GPU memory usage in MB.
    pub gpu_memory_used_mb: f64,
    /// GPU memory total in MB.
    pub gpu_memory_total_mb: f64,
    /// Active environments count.
    pub active_envs: usize,
    /// Tensor operations per second.
    pub tensor_ops_per_sec: f64,
    /// GPU name.
    pub gpu_name: String,
    /// CPU name.
    pub cpu_name: String,
    /// Number of CPU cores.
    pub cpu_cores: usize,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Create a new App instance with default state.
    pub fn new() -> Self {
        let refresh_kind = RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());
        let mut sys = System::new_with_specifics(refresh_kind);
        sys.refresh_all();

        // Get initial system info
        let cpu_name = sys
            .cpus()
            .first()
            .map(|c| c.brand().to_string())
            .unwrap_or_else(|| "Unknown CPU".to_string());
        let cpu_cores = sys.cpus().len();

        // Get GPU info (macOS specific)
        let (gpu_name, gpu_memory_total_mb) = get_gpu_info();

        let total_memory_mb = sys.total_memory() as f64 / 1024.0 / 1024.0;

        Self {
            current_screen: Screen::Dashboard,
            mode: AppMode::Demo,
            should_quit: false,
            tick: 0,
            metrics_history: Vec::new(),
            current_metrics: TrainMetrics::default(),
            benchmark_data: vec![
                ("PPO Training".to_string(), 15000.0, 3200.0),
                ("Env Step".to_string(), 850000.0, 125000.0),
                ("Buffer Sample".to_string(), 2500000.0, 450000.0),
                ("Network Forward".to_string(), 180000.0, 42000.0),
                ("GAE Compute".to_string(), 920000.0, 180000.0),
            ],
            system_metrics: SystemMetrics {
                cpu_usage: 0.0,
                gpu_usage: 0.0,
                memory_used_mb: 0.0,
                memory_total_mb: total_memory_mb,
                gpu_memory_used_mb: 0.0,
                gpu_memory_total_mb,
                active_envs: 16,
                tensor_ops_per_sec: 0.0,
                gpu_name,
                cpu_name,
                cpu_cores,
            },
            reward_history: vec![10, 15, 20, 18, 25, 30, 28, 35, 40, 38, 45, 50, 48, 55, 60],
            training_active: false,
            steps_per_second: 0.0,
            sys,
            last_cpu_refresh: Instant::now(),
        }
    }

    /// Refresh real system metrics.
    pub fn refresh_system_metrics(&mut self) {
        // Only refresh CPU every 500ms (it needs time to measure)
        if self.last_cpu_refresh.elapsed() >= Duration::from_millis(500) {
            self.sys.refresh_cpu_usage();
            self.last_cpu_refresh = Instant::now();
        }
        self.sys.refresh_memory();

        // Calculate average CPU usage across all cores
        let cpu_usage: f32 = if self.sys.cpus().is_empty() {
            0.0
        } else {
            self.sys.cpus().iter().map(|c| c.cpu_usage()).sum::<f32>()
                / self.sys.cpus().len() as f32
        };

        let memory_used_mb = self.sys.used_memory() as f64 / 1024.0 / 1024.0;
        let memory_total_mb = self.sys.total_memory() as f64 / 1024.0 / 1024.0;

        // Get GPU metrics (macOS specific)
        let (gpu_usage, gpu_memory_used_mb) = get_gpu_usage();

        self.system_metrics.cpu_usage = cpu_usage;
        self.system_metrics.gpu_usage = gpu_usage;
        self.system_metrics.memory_used_mb = memory_used_mb;
        self.system_metrics.memory_total_mb = memory_total_mb;
        self.system_metrics.gpu_memory_used_mb = gpu_memory_used_mb;
    }

    /// Update training metrics.
    pub fn update_metrics(&mut self, metrics: TrainMetrics) {
        self.current_metrics = metrics.clone();
        self.metrics_history.push(metrics);

        // Keep history bounded
        if self.metrics_history.len() > 1000 {
            self.metrics_history.remove(0);
        }

        // Update reward history for sparkline
        let reward_scaled = ((self.current_metrics.mean_reward + 100.0).max(0.0) * 10.0) as u64;
        self.reward_history.push(reward_scaled);
        if self.reward_history.len() > 100 {
            self.reward_history.remove(0);
        }
    }

    /// Update system metrics.
    pub fn update_system_metrics(&mut self, metrics: SystemMetrics) {
        self.system_metrics = metrics;
    }

    /// Advance to next screen/tab.
    pub fn next_screen(&mut self) {
        self.current_screen = match self.current_screen {
            Screen::Dashboard => Screen::Training,
            Screen::Training => Screen::Benchmark,
            Screen::Benchmark => Screen::About,
            Screen::About => Screen::Dashboard,
        };
    }

    /// Go to previous screen/tab.
    pub fn prev_screen(&mut self) {
        self.current_screen = match self.current_screen {
            Screen::Dashboard => Screen::About,
            Screen::Training => Screen::Dashboard,
            Screen::Benchmark => Screen::Training,
            Screen::About => Screen::Benchmark,
        };
    }

    /// Handle keyboard input.
    pub fn handle_key(&mut self, key: KeyCode) {
        match key {
            KeyCode::Char('q') | KeyCode::Esc => self.should_quit = true,
            KeyCode::Tab | KeyCode::Right => self.next_screen(),
            KeyCode::BackTab | KeyCode::Left => self.prev_screen(),
            KeyCode::Char('1') => self.current_screen = Screen::Dashboard,
            KeyCode::Char('2') => self.current_screen = Screen::Training,
            KeyCode::Char('3') => self.current_screen = Screen::Benchmark,
            KeyCode::Char('4') => self.current_screen = Screen::About,
            KeyCode::Char(' ') => self.training_active = !self.training_active,
            KeyCode::Char('m') | KeyCode::Char('M') => self.mode = self.mode.next(),
            KeyCode::Char('d') | KeyCode::Char('D') => self.mode = AppMode::Demo,
            KeyCode::Char('b') | KeyCode::Char('B') => self.mode = AppMode::Benchmark,
            KeyCode::Char('t') | KeyCode::Char('T') => self.mode = AppMode::Training,
            _ => {}
        }
    }

    /// Increment animation tick.
    pub fn tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }

    /// Get color cycling based on tick for animations.
    pub fn cycle_color(&self, offset: u64) -> Color {
        let hue = ((self.tick + offset) % 360) as f64;
        hsl_to_rgb(hue, 0.7, 0.6)
    }

    /// Get pulsing intensity (0.0 - 1.0) for animations.
    pub fn pulse(&self) -> f64 {
        let phase = (self.tick as f64 * 0.1).sin();
        (phase + 1.0) / 2.0
    }
}

/// Get GPU info for macOS (Apple Silicon).
#[cfg(target_os = "macos")]
fn get_gpu_info() -> (String, f64) {
    // Try to get GPU info using system_profiler
    let output = Command::new("system_profiler")
        .args(["SPDisplaysDataType", "-json"])
        .output();

    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            // Parse GPU name
            if let Some(start) = text.find("\"sppci_model\"") {
                if let Some(name_start) = text[start..].find(": \"") {
                    let name_begin = start + name_start + 3;
                    if let Some(name_end) = text[name_begin..].find('"') {
                        let gpu_name = text[name_begin..name_begin + name_end].to_string();

                        // For Apple Silicon, estimate unified memory (shared with CPU)
                        // M1: 8-16GB, M1 Pro/Max: 16-64GB, M2: 8-24GB, M3: 8-128GB
                        let gpu_memory = if gpu_name.contains("M3 Max") {
                            96.0 * 1024.0
                        } else if gpu_name.contains("M3 Pro") || gpu_name.contains("M2 Max") {
                            36.0 * 1024.0
                        } else if gpu_name.contains("M2 Pro") || gpu_name.contains("M1 Max") {
                            32.0 * 1024.0
                        } else if gpu_name.contains("M1 Pro") {
                            16.0 * 1024.0
                        } else {
                            8.0 * 1024.0 // Base M1/M2/M3
                        };

                        return (gpu_name, gpu_memory);
                    }
                }
            }
        }
    }

    ("Apple GPU".to_string(), 8192.0)
}

#[cfg(not(target_os = "macos"))]
fn get_gpu_info() -> (String, f64) {
    // For NVIDIA GPUs, could use nvidia-smi
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            let parts: Vec<&str> = text.trim().split(", ").collect();
            if parts.len() >= 2 {
                let name = parts[0].to_string();
                let memory = parts[1].parse::<f64>().unwrap_or(8192.0);
                return (name, memory);
            }
        }
    }

    ("Unknown GPU".to_string(), 8192.0)
}

/// Get current GPU usage for macOS.
#[cfg(target_os = "macos")]
fn get_gpu_usage() -> (f32, f64) {
    // Use powermetrics or ioreg for GPU usage (requires sudo for powermetrics)
    // Fallback to estimating from GPU processes
    let output = Command::new("ioreg")
        .args(["-r", "-c", "IOAccelerator"])
        .output();

    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            // Look for GPU utilization
            if let Some(start) = text.find("PerformanceStatistics") {
                let section = &text[start..];

                // Try to find GPU utilization percentage
                if let Some(util_start) = section.find("GPU Activity") {
                    if let Some(eq_pos) = section[util_start..].find(" = ") {
                        let num_start = util_start + eq_pos + 3;
                        if let Some(num_end) = section[num_start..].find('\n') {
                            if let Ok(usage) = section[num_start..num_start + num_end]
                                .trim()
                                .parse::<f32>()
                            {
                                return (usage, 0.0);
                            }
                        }
                    }
                }

                // Try device utilization
                if let Some(util_start) = section.find("Device Utilization") {
                    if let Some(eq_pos) = section[util_start..].find(" = ") {
                        let num_start = util_start + eq_pos + 3;
                        if let Some(num_end) = section[num_start..].find('\n') {
                            let value_str = section[num_start..num_start + num_end].trim();
                            // Value might be like "42 %" or just "42"
                            let cleaned = value_str.replace('%', "").trim().to_string();
                            if let Ok(usage) = cleaned.parse::<f32>() {
                                return (usage, 0.0);
                            }
                        }
                    }
                }
            }
        }
    }

    // Fallback: estimate from running Metal processes
    let output = Command::new("ps").args(["aux"]).output();

    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            // Count GPU-related processes
            let gpu_processes = text
                .lines()
                .filter(|l| l.contains("Metal") || l.contains("GPU") || l.contains("WindowServer"))
                .count();

            // Rough estimate based on process count
            let estimated_usage = (gpu_processes as f32 * 5.0).min(95.0);
            return (estimated_usage, 0.0);
        }
    }

    (0.0, 0.0)
}

#[cfg(not(target_os = "macos"))]
fn get_gpu_usage() -> (f32, f64) {
    // For NVIDIA GPUs
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            let parts: Vec<&str> = text.trim().split(", ").collect();
            if parts.len() >= 2 {
                let usage = parts[0].parse::<f32>().unwrap_or(0.0);
                let memory = parts[1].parse::<f64>().unwrap_or(0.0);
                return (usage, memory);
            }
        }
    }

    (0.0, 0.0)
}

/// Convert HSL to RGB Color.
fn hsl_to_rgb(h: f64, s: f64, l: f64) -> Color {
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = l - c / 2.0;

    let (r, g, b) = match h as u32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    Color::Rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

/// TUI runner that manages the terminal and render loop.
pub struct Tui {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    tick_rate: Duration,
}

impl Tui {
    /// Create a new TUI instance.
    pub fn new(tick_rate: Duration) -> io::Result<Self> {
        let backend = CrosstermBackend::new(io::stdout());
        let terminal = Terminal::new(backend)?;
        Ok(Self {
            terminal,
            tick_rate,
        })
    }

    /// Initialize the terminal for TUI mode.
    pub fn enter(&mut self) -> io::Result<()> {
        enable_raw_mode()?;
        execute!(io::stdout(), EnterAlternateScreen, EnableMouseCapture)?;
        self.terminal.hide_cursor()?;
        self.terminal.clear()?;
        Ok(())
    }

    /// Restore the terminal to normal mode.
    pub fn exit(&mut self) -> io::Result<()> {
        disable_raw_mode()?;
        execute!(io::stdout(), LeaveAlternateScreen, DisableMouseCapture)?;
        self.terminal.show_cursor()?;
        Ok(())
    }

    /// Run the main TUI loop.
    pub fn run(&mut self, app: &mut App) -> io::Result<()> {
        self.enter()?;

        let mut last_tick = Instant::now();

        loop {
            // Render
            self.terminal.draw(|frame| {
                render(frame, app);
            })?;

            // Handle events with timeout
            let timeout = self
                .tick_rate
                .checked_sub(last_tick.elapsed())
                .unwrap_or_else(|| Duration::from_secs(0));

            if event::poll(timeout)? {
                if let Event::Key(key) = event::read()? {
                    if key.kind == KeyEventKind::Press {
                        app.handle_key(key.code);
                    }
                }
            }

            // Check for quit
            if app.should_quit {
                break;
            }

            // Tick update
            if last_tick.elapsed() >= self.tick_rate {
                app.tick();
                last_tick = Instant::now();

                // Update based on mode
                match app.mode {
                    AppMode::Demo => {
                        if app.training_active {
                            simulate_training_step(app);
                        }
                        // In demo mode, simulate system metrics too
                        simulate_system_metrics(app);
                    }
                    AppMode::Benchmark => {
                        // Real system metrics
                        app.refresh_system_metrics();
                    }
                    AppMode::Training => {
                        // Real training + real metrics
                        app.refresh_system_metrics();
                        if app.training_active {
                            // TODO: Hook into real training loop
                            simulate_training_step(app);
                        }
                    }
                }
            }
        }

        self.exit()?;
        Ok(())
    }
}

/// Simulate system metrics for demo mode.
fn simulate_system_metrics(app: &mut App) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Smooth random walk for demo metrics
    let cpu_delta: f32 = rng.gen_range(-5.0..5.0);
    let gpu_delta: f32 = rng.gen_range(-3.0..3.0);

    app.system_metrics.cpu_usage = (app.system_metrics.cpu_usage + cpu_delta).clamp(15.0, 85.0);
    app.system_metrics.gpu_usage = (app.system_metrics.gpu_usage + gpu_delta).clamp(30.0, 95.0);

    // Memory varies less
    let mem_delta: f64 = rng.gen_range(-100.0..100.0);
    app.system_metrics.memory_used_mb = (app.system_metrics.memory_used_mb + mem_delta)
        .clamp(1024.0, app.system_metrics.memory_total_mb * 0.9);

    app.system_metrics.gpu_memory_used_mb = app.system_metrics.gpu_memory_total_mb
        * (app.system_metrics.gpu_usage as f64 / 100.0)
        * rng.gen_range(0.8..1.2);

    app.system_metrics.tensor_ops_per_sec = rng.gen_range(800_000.0..2_500_000.0);
}

/// Simulate a training step for demo purposes.
fn simulate_training_step(app: &mut App) {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let reward_delta: f32 = rng.gen_range(-5.0..10.0);
    let new_reward = (app.current_metrics.mean_reward + reward_delta * 0.1).clamp(-100.0, 500.0);

    let metrics = TrainMetrics {
        policy_loss: (app.current_metrics.policy_loss * 0.99 + rng.gen_range(0.0..0.1)).max(0.001),
        value_loss: (app.current_metrics.value_loss * 0.99 + rng.gen_range(0.0..0.05)).max(0.001),
        entropy: (app.current_metrics.entropy * 0.999 + rng.gen_range(0.0..0.01)).clamp(0.1, 2.0),
        approx_kl: rng.gen_range(0.001..0.02),
        clip_fraction: rng.gen_range(0.05..0.15),
        explained_variance: rng.gen_range(0.7..0.95),
        learning_rate: 0.0003,
        timesteps: app.current_metrics.timesteps + rng.gen_range(128..512),
        episodes: app.current_metrics.episodes + if rng.gen_bool(0.1) { 1 } else { 0 },
        mean_reward: new_reward,
        std_reward: rng.gen_range(5.0..20.0),
    };

    app.update_metrics(metrics);
    app.steps_per_second = rng.gen_range(8000.0..15000.0);
}

/// Main render function that draws the entire UI.
fn render(frame: &mut Frame<'_>, app: &App) {
    let area = frame.area();

    // Main layout: header, content, footer
    let main_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header with tabs
            Constraint::Min(10),   // Main content
            Constraint::Length(3), // Footer
        ])
        .split(area);

    // Render header with tabs
    render_header(frame, app, main_layout[0]);

    // Render current screen content
    match app.current_screen {
        Screen::Dashboard => screens::render_dashboard(frame, app, main_layout[1]),
        Screen::Training => screens::render_training(frame, app, main_layout[1]),
        Screen::Benchmark => screens::render_benchmark(frame, app, main_layout[1]),
        Screen::About => screens::render_about(frame, app, main_layout[1]),
    }

    // Render footer
    render_footer(frame, app, main_layout[2]);
}

/// Render the header with navigation tabs.
fn render_header(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let titles: Vec<Line<'_>> = Screen::all()
        .iter()
        .map(|s| {
            let (icon, name) = s.title();
            Line::from(vec![
                Span::styled(icon, Style::default().fg(Color::Cyan)),
                Span::raw(" "),
                Span::raw(name),
            ])
        })
        .collect();

    let selected = app.current_screen.index();

    // Add mode indicator to title
    let mode_indicator = format!(" {} | {} ", app.mode.name(), "ROCKET");

    let tabs = Tabs::new(titles)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(100, 100, 120)))
                .title(Span::styled(
                    mode_indicator,
                    Style::default()
                        .fg(app.mode.color())
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .select(selected)
        .style(Style::default().fg(Color::Gray))
        .highlight_style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD)
                .add_modifier(Modifier::UNDERLINED),
        )
        .divider(Span::styled(" | ", Style::default().fg(Color::DarkGray)));

    frame.render_widget(tabs, area);
}

/// Render the footer with help and status.
fn render_footer(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let mode_span = Span::styled(
        format!(" {} ", app.mode.name()),
        Style::default()
            .fg(Color::Black)
            .bg(app.mode.color())
            .add_modifier(Modifier::BOLD),
    );

    let status = if app.training_active {
        Span::styled(
            " ACTIVE ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::Green)
                .add_modifier(Modifier::BOLD),
        )
    } else {
        Span::styled(
            " PAUSED ",
            Style::default()
                .fg(Color::Black)
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD),
        )
    };

    let help = Line::from(vec![
        mode_span,
        Span::raw(" "),
        status,
        Span::raw("  "),
        Span::styled("[m]", Style::default().fg(Color::Magenta)),
        Span::raw(" Mode  "),
        Span::styled("[Tab]", Style::default().fg(Color::Cyan)),
        Span::raw(" Screen  "),
        Span::styled("[Space]", Style::default().fg(Color::Cyan)),
        Span::raw(" Train  "),
        Span::styled("[q]", Style::default().fg(Color::Red)),
        Span::raw(" Quit"),
    ]);

    let footer = ratatui::widgets::Paragraph::new(help)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(100, 100, 120))),
        )
        .style(Style::default().fg(Color::Gray));

    frame.render_widget(footer, area);
}

/// Run the TUI application.
pub fn run_tui() -> io::Result<()> {
    let mut app = App::new();
    let mut tui = Tui::new(Duration::from_millis(100))?;
    tui.run(&mut app)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_app_creation() {
        let app = App::new();
        assert!(!app.should_quit);
        assert_eq!(app.current_screen, Screen::Dashboard);
        assert_eq!(app.mode, AppMode::Demo);
    }

    #[test]
    fn test_screen_navigation() {
        let mut app = App::new();
        assert_eq!(app.current_screen, Screen::Dashboard);

        app.next_screen();
        assert_eq!(app.current_screen, Screen::Training);

        app.next_screen();
        assert_eq!(app.current_screen, Screen::Benchmark);

        app.prev_screen();
        assert_eq!(app.current_screen, Screen::Training);
    }

    #[test]
    fn test_mode_switching() {
        let mut app = App::new();
        assert_eq!(app.mode, AppMode::Demo);

        app.handle_key(KeyCode::Char('m'));
        assert_eq!(app.mode, AppMode::Benchmark);

        app.handle_key(KeyCode::Char('m'));
        assert_eq!(app.mode, AppMode::Training);

        app.handle_key(KeyCode::Char('d'));
        assert_eq!(app.mode, AppMode::Demo);
    }

    #[test]
    fn test_key_handling() {
        let mut app = App::new();

        app.handle_key(KeyCode::Char('q'));
        assert!(app.should_quit);

        app.should_quit = false;
        app.handle_key(KeyCode::Char('2'));
        assert_eq!(app.current_screen, Screen::Training);
    }
}
