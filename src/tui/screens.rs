//! Screen definitions and rendering for the TUI.

use super::widgets::{benchmark, logo, metrics, training};
use super::App;
use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

/// Available screens/tabs in the TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Screen {
    /// Dashboard overview.
    #[default]
    Dashboard,
    /// Live training visualization.
    Training,
    /// Benchmark comparisons.
    Benchmark,
    /// About information.
    About,
}

impl Screen {
    /// Get all available screens.
    pub fn all() -> &'static [Screen] {
        &[
            Screen::Dashboard,
            Screen::Training,
            Screen::Benchmark,
            Screen::About,
        ]
    }

    /// Get the title and icon for this screen.
    pub fn title(&self) -> (&'static str, &'static str) {
        match self {
            Screen::Dashboard => ("[D]", "Dashboard"),
            Screen::Training => ("[T]", "Training"),
            Screen::Benchmark => ("[B]", "Benchmark"),
            Screen::About => ("[A]", "About"),
        }
    }

    /// Get the index of this screen.
    pub fn index(&self) -> usize {
        match self {
            Screen::Dashboard => 0,
            Screen::Training => 1,
            Screen::Benchmark => 2,
            Screen::About => 3,
        }
    }
}

/// Render the dashboard overview screen.
pub fn render_dashboard(frame: &mut Frame<'_>, app: &App, area: Rect) {
    // Layout: top row (logo + quick stats), bottom row (metrics)
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(12), Constraint::Min(10)])
        .split(area);

    // Top row: Logo on left, quick stats on right
    let top_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(layout[0]);

    // Render mini logo
    render_mini_logo(frame, app, top_layout[0]);

    // Render quick stats
    render_quick_stats(frame, app, top_layout[1]);

    // Bottom row: Split into three panels
    let bottom_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(layout[1]);

    // Metrics panels
    metrics::render_cpu_gauge(frame, app, bottom_layout[0]);
    metrics::render_gpu_gauge(frame, app, bottom_layout[1]);
    metrics::render_memory_gauge(frame, app, bottom_layout[2]);
}

/// Render a mini version of the logo for dashboard.
fn render_mini_logo(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let logo_lines = ["██████╗  ██████╗  ██████╗██╗  ██╗███████╗████████╗",
        "██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝██╔════╝╚══██╔══╝",
        "██████╔╝██║   ██║██║     █████╔╝ █████╗     ██║   ",
        "██╔══██╗██║   ██║██║     ██╔═██╗ ██╔══╝     ██║   ",
        "██║  ██║╚██████╔╝╚██████╗██║  ██╗███████╗   ██║   ",
        "╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝   "];

    let pulse = app.pulse();
    let base_color = if app.training_active {
        Color::Rgb(
            (100.0 + 155.0 * pulse) as u8,
            (200.0 + 55.0 * pulse) as u8,
            (100.0 + 50.0 * pulse) as u8,
        )
    } else {
        Color::Rgb(
            (200.0 + 55.0 * pulse) as u8,
            (150.0 + 50.0 * pulse) as u8,
            (50.0 + 50.0 * pulse) as u8,
        )
    };

    let lines: Vec<Line<'_>> = logo_lines
        .iter()
        .map(|line| Line::from(Span::styled(*line, Style::default().fg(base_color))))
        .collect();

    let logo = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " RocketRL ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .alignment(Alignment::Center);

    frame.render_widget(logo, area);
}

/// Render quick statistics panel.
fn render_quick_stats(frame: &mut Frame<'_>, app: &App, area: Rect) {
    use super::AppMode;

    // Truncate CPU/GPU names for display
    let cpu_short = if app.system_metrics.cpu_name.len() > 25 {
        format!("{}...", &app.system_metrics.cpu_name[..22])
    } else {
        app.system_metrics.cpu_name.clone()
    };
    let gpu_short = if app.system_metrics.gpu_name.len() > 25 {
        format!("{}...", &app.system_metrics.gpu_name[..22])
    } else {
        app.system_metrics.gpu_name.clone()
    };

    let mode_color = app.mode.color();
    let mode_indicator = match app.mode {
        AppMode::Demo => "(simulated)",
        AppMode::Benchmark => "(real)",
        AppMode::Training => "(live)",
    };

    let stats = vec![
        // Hardware info
        Line::from(vec![
            Span::styled("CPU: ", Style::default().fg(Color::Gray)),
            Span::styled(cpu_short, Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            Span::styled("GPU: ", Style::default().fg(Color::Gray)),
            Span::styled(gpu_short, Style::default().fg(Color::Cyan)),
        ]),
        Line::from(vec![
            Span::styled("Cores: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.system_metrics.cpu_cores),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled("  ", Style::default()),
            Span::styled(mode_indicator, Style::default().fg(mode_color)),
        ]),
        Line::from(""),
        // Training stats
        Line::from(vec![
            Span::styled("Episodes: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.current_metrics.episodes),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled("  Steps/s: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.0}", app.steps_per_second),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Reward: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>7.2}", app.current_metrics.mean_reward),
                Style::default()
                    .fg(if app.current_metrics.mean_reward > 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    })
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(vec![
            Span::styled("Policy: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.5}", app.current_metrics.policy_loss),
                Style::default().fg(Color::Yellow),
            ),
            Span::styled("  Value: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:.5}", app.current_metrics.value_loss),
                Style::default().fg(Color::Yellow),
            ),
        ]),
    ];

    let stats_widget = Paragraph::new(stats)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " System & Stats ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(stats_widget, area);
}

/// Render the training visualization screen.
pub fn render_training(frame: &mut Frame<'_>, app: &App, area: Rect) {
    // Layout: reward graph on top, loss/metrics on bottom
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);

    // Top: Reward sparkline and episode info
    let top_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(layout[0]);

    training::render_reward_sparkline(frame, app, top_layout[0]);
    training::render_episode_info(frame, app, top_layout[1]);

    // Bottom: Loss gauges and steps
    let bottom_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(34),
            Constraint::Percentage(33),
        ])
        .split(layout[1]);

    training::render_policy_loss_gauge(frame, app, bottom_layout[0]);
    training::render_value_loss_gauge(frame, app, bottom_layout[1]);
    training::render_training_stats(frame, app, bottom_layout[2]);
}

/// Render the benchmark comparison screen.
pub fn render_benchmark(frame: &mut Frame<'_>, app: &App, area: Rect) {
    // Layout: comparison bars on left, throughput on right
    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(area);

    benchmark::render_comparison_bars(frame, app, layout[0]);

    // Right side: throughput and speedup
    let right_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(layout[1]);

    benchmark::render_throughput_gauge(frame, app, right_layout[0]);
    benchmark::render_speedup_indicators(frame, app, right_layout[1]);
}

/// Render the about screen.
pub fn render_about(frame: &mut Frame<'_>, app: &App, area: Rect) {
    // Layout: logo on top, info on bottom
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(14), Constraint::Min(10)])
        .split(area);

    // Render the big logo
    logo::render_logo(frame, app, layout[0]);

    // Info section
    let info_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(layout[1]);

    render_about_info(frame, app, info_layout[0]);
    render_about_features(frame, app, info_layout[1]);
}

/// Render about info panel.
fn render_about_info(frame: &mut Frame<'_>, _app: &App, area: Rect) {
    let info = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled(
                "RocketRL",
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" - High-Performance RL Library"),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Version: ", Style::default().fg(Color::Gray)),
            Span::styled("0.2.0", Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled("License: ", Style::default().fg(Color::Gray)),
            Span::styled("GPL-2.0", Style::default().fg(Color::Green)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Backend: ", Style::default().fg(Color::Gray)),
            Span::styled("Candle (HuggingFace)", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(vec![
            Span::styled("Platforms: ", Style::default().fg(Color::Gray)),
            Span::styled("CPU, Metal, CUDA", Style::default().fg(Color::Yellow)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "github.com/rocketrl/rocketrl",
            Style::default()
                .fg(Color::Blue)
                .add_modifier(Modifier::UNDERLINED),
        )),
    ];

    let info_widget = Paragraph::new(info)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " About ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(info_widget, area);
}

/// Render features panel.
fn render_about_features(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let check = Span::styled("[x]", Style::default().fg(Color::Green));
    let features = vec![
        Line::from(""),
        Line::from(vec![
            check.clone(),
            Span::raw(" "),
            Span::styled("PPO & A2C Algorithms", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            check.clone(),
            Span::raw(" "),
            Span::styled("Vectorized Environments", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            check.clone(),
            Span::raw(" "),
            Span::styled("LSTM/GRU Networks", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            check.clone(),
            Span::raw(" "),
            Span::styled(
                "GPU Acceleration (Metal/CUDA)",
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(vec![
            check.clone(),
            Span::raw(" "),
            Span::styled("Trading Environment", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            check.clone(),
            Span::raw(" "),
            Span::styled("Zero-Copy Tensor Ops", Style::default().fg(Color::White)),
        ]),
        Line::from(vec![
            check.clone(),
            Span::raw(" "),
            Span::styled(
                "GAE for Advantage Estimation",
                Style::default().fg(Color::White),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Active Envs: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{}", app.system_metrics.active_envs),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    let features_widget = Paragraph::new(features)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " Features ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .wrap(Wrap { trim: true });

    frame.render_widget(features_widget, area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_screen_all() {
        let all = Screen::all();
        assert_eq!(all.len(), 4);
    }

    #[test]
    fn test_screen_index() {
        assert_eq!(Screen::Dashboard.index(), 0);
        assert_eq!(Screen::Training.index(), 1);
        assert_eq!(Screen::Benchmark.index(), 2);
        assert_eq!(Screen::About.index(), 3);
    }
}
