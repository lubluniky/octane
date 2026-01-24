//! UI rendering for the Rocket TUI
//!
//! Handles all visual rendering using ratatui widgets.

use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Bar, BarChart, BarGroup, Block, Borders, Chart, Dataset, Gauge, GraphType,
        List, ListItem, Padding, Paragraph, Row, Sparkline, Table, Tabs, Wrap,
    },
    Frame,
};

use super::app::{App, Tab};

// Theme colors - Rocket-inspired dark theme with orange/red accents
const ROCKET_ORANGE: Color = Color::Rgb(255, 107, 53);
const ROCKET_RED: Color = Color::Rgb(220, 50, 47);
const ROCKET_YELLOW: Color = Color::Rgb(255, 193, 7);
const ROCKET_GREEN: Color = Color::Rgb(46, 204, 113);
const ROCKET_BLUE: Color = Color::Rgb(52, 152, 219);
const ROCKET_PURPLE: Color = Color::Rgb(155, 89, 182);
const DARK_BG: Color = Color::Rgb(18, 18, 24);
const DARK_SURFACE: Color = Color::Rgb(28, 28, 36);
const DARK_BORDER: Color = Color::Rgb(48, 48, 60);
const TEXT_PRIMARY: Color = Color::Rgb(230, 230, 240);
const TEXT_SECONDARY: Color = Color::Rgb(150, 150, 165);
const TEXT_DIM: Color = Color::Rgb(100, 100, 115);

/// ASCII art logo for splash screen
const ROCKET_LOGO: &str = r#"
    ██████╗  ██████╗  ██████╗██╗  ██╗███████╗████████╗
    ██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝██╔════╝╚══██╔══╝
    ██████╔╝██║   ██║██║     █████╔╝ █████╗     ██║
    ██╔══██╗██║   ██║██║     ██╔═██╗ ██╔══╝     ██║
    ██║  ██║╚██████╔╝╚██████╗██║  ██╗███████╗   ██║
    ╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝
"#;

const ROCKET_SUBTITLE: &str = "High-Performance Reinforcement Learning for Rust";

/// Main draw function - entry point for rendering
pub fn draw(frame: &mut Frame, app: &App) {
    // Clear with dark background
    let area = frame.area();

    if app.show_splash {
        draw_splash(frame, area, app);
    } else if app.show_help {
        draw_main_with_help(frame, area, app);
    } else {
        draw_main(frame, area, app);
    }
}

/// Draw splash screen
fn draw_splash(frame: &mut Frame, area: Rect, app: &App) {
    let block = Block::default()
        .style(Style::default().bg(DARK_BG));
    frame.render_widget(block, area);

    // Center the logo vertically
    let vertical_center = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(30),
            Constraint::Min(12),
            Constraint::Percentage(30),
        ])
        .split(area);

    // Animated gradient effect based on frame
    let gradient_colors = [
        ROCKET_ORANGE,
        ROCKET_RED,
        ROCKET_YELLOW,
        ROCKET_ORANGE,
    ];
    let color_idx = (app.frame / 15) as usize % gradient_colors.len();
    let logo_color = gradient_colors[color_idx];

    // Logo
    let logo = Paragraph::new(ROCKET_LOGO)
        .style(Style::default().fg(logo_color).add_modifier(Modifier::BOLD))
        .alignment(Alignment::Center);
    frame.render_widget(logo, vertical_center[1]);

    // Subtitle below logo
    let subtitle_area = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(8),
            Constraint::Length(2),
            Constraint::Length(1),
            Constraint::Min(0),
        ])
        .split(vertical_center[1]);

    let subtitle = Paragraph::new(ROCKET_SUBTITLE)
        .style(Style::default().fg(TEXT_SECONDARY))
        .alignment(Alignment::Center);
    frame.render_widget(subtitle, subtitle_area[2]);

    // Loading animation
    let dots = ".".repeat(((app.frame / 20) % 4) as usize);
    let loading = Paragraph::new(format!("Initializing{}", dots))
        .style(Style::default().fg(TEXT_DIM))
        .alignment(Alignment::Center);

    let loading_area = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);
    frame.render_widget(loading, loading_area[1]);

    // Progress bar
    let progress = app.frame as f64 / 120.0; // ~2 seconds at 60fps
    let progress_area = centered_rect(40, 1, vertical_center[2]);
    let gauge = Gauge::default()
        .gauge_style(Style::default().fg(ROCKET_ORANGE).bg(DARK_SURFACE))
        .ratio(progress.min(1.0))
        .label("");
    frame.render_widget(gauge, progress_area);
}

/// Draw main interface
fn draw_main(frame: &mut Frame, area: Rect, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3), // Header
            Constraint::Min(0),    // Content
            Constraint::Length(3), // Footer
        ])
        .split(area);

    draw_header(frame, layout[0], app);
    draw_content(frame, layout[1], app);
    draw_footer(frame, layout[2], app);
}

/// Draw main interface with help overlay
fn draw_main_with_help(frame: &mut Frame, area: Rect, app: &App) {
    draw_main(frame, area, app);

    // Overlay help
    let help_area = centered_rect(60, 70, area);
    draw_help_overlay(frame, help_area);
}

/// Draw header with tabs
fn draw_header(frame: &mut Frame, area: Rect, app: &App) {
    let header_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Length(20), // Logo
            Constraint::Min(0),     // Tabs
            Constraint::Length(20), // Status
        ])
        .split(area);

    // Logo/title
    let title = Paragraph::new(" ROCKET")
        .style(Style::default()
            .fg(ROCKET_ORANGE)
            .add_modifier(Modifier::BOLD))
        .block(Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(DARK_BORDER)));
    frame.render_widget(title, header_layout[0]);

    // Tabs
    let tab_titles: Vec<Line> = Tab::all()
        .iter()
        .enumerate()
        .map(|(i, tab)| {
            let style = if *tab == app.tab {
                Style::default().fg(ROCKET_ORANGE).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(TEXT_SECONDARY)
            };
            Line::from(vec![
                Span::styled(format!("[{}] ", i + 1), Style::default().fg(TEXT_DIM)),
                Span::styled(tab.name(), style),
            ])
        })
        .collect();

    let tabs = Tabs::new(tab_titles)
        .select(app.tab.index())
        .highlight_style(Style::default().fg(ROCKET_ORANGE))
        .divider(Span::styled(" | ", Style::default().fg(DARK_BORDER)))
        .block(Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(DARK_BORDER)));
    frame.render_widget(tabs, header_layout[1]);

    // Status indicator
    let status_text = if app.paused {
        Span::styled(" PAUSED ", Style::default().fg(ROCKET_YELLOW).add_modifier(Modifier::BOLD))
    } else {
        Span::styled(" RUNNING ", Style::default().fg(ROCKET_GREEN).add_modifier(Modifier::BOLD))
    };
    let status = Paragraph::new(status_text)
        .alignment(Alignment::Right)
        .block(Block::default()
            .borders(Borders::BOTTOM)
            .border_style(Style::default().fg(DARK_BORDER)));
    frame.render_widget(status, header_layout[2]);
}

/// Draw main content area based on current tab
fn draw_content(frame: &mut Frame, area: Rect, app: &App) {
    match app.tab {
        Tab::Dashboard => draw_dashboard(frame, area, app),
        Tab::Training => draw_training(frame, area, app),
        Tab::Environment => draw_environment(frame, area, app),
        Tab::Benchmarks => draw_benchmarks(frame, area, app),
        Tab::Settings => draw_settings(frame, area, app),
    }
}

/// Draw dashboard tab
fn draw_dashboard(frame: &mut Frame, area: Rect, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(7), // Metrics cards
            Constraint::Min(10),   // Charts
            Constraint::Length(8), // Logs
        ])
        .margin(1)
        .split(area);

    // Metrics cards row
    draw_metrics_cards(frame, layout[0], app);

    // Charts area
    let charts_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(layout[1]);

    draw_reward_chart(frame, charts_layout[0], app);
    draw_loss_sparklines(frame, charts_layout[1], app);

    // Logs
    draw_logs(frame, layout[2], app);
}

/// Draw metrics cards
fn draw_metrics_cards(frame: &mut Frame, area: Rect, app: &App) {
    let cards_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
            Constraint::Percentage(20),
        ])
        .split(area);

    let metrics = [
        ("Episode", format!("{}", app.metrics.episode), ROCKET_BLUE),
        ("Timesteps", format!("{:.1}K", app.metrics.timesteps as f64 / 1000.0), ROCKET_PURPLE),
        ("Reward", format!("{:.1}", app.metrics.episode_reward),
            if app.metrics.episode_reward > 0.0 { ROCKET_GREEN } else { ROCKET_RED }),
        ("Avg Reward", format!("{:.1}", app.metrics.avg_reward),
            if app.metrics.avg_reward > 0.0 { ROCKET_GREEN } else { ROCKET_RED }),
        ("Steps/s", format!("{:.0}", app.metrics.steps_per_second), ROCKET_ORANGE),
    ];

    for (i, (label, value, color)) in metrics.iter().enumerate() {
        let card = Paragraph::new(vec![
            Line::from(Span::styled(*label, Style::default().fg(TEXT_SECONDARY))),
            Line::from(""),
            Line::from(Span::styled(
                value.clone(),
                Style::default().fg(*color).add_modifier(Modifier::BOLD),
            )),
        ])
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(DARK_BORDER))
            .style(Style::default().bg(DARK_SURFACE)));
        frame.render_widget(card, cards_layout[i]);
    }
}

/// Draw reward chart
fn draw_reward_chart(frame: &mut Frame, area: Rect, app: &App) {
    if app.reward_history.is_empty() {
        let placeholder = Paragraph::new("Waiting for training data...")
            .style(Style::default().fg(TEXT_DIM))
            .alignment(Alignment::Center)
            .block(Block::default()
                .title(" Reward History ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(DARK_BORDER)));
        frame.render_widget(placeholder, area);
        return;
    }

    let data: Vec<(f64, f64)> = app.reward_history
        .iter()
        .map(|e| (e.episode as f64, e.reward))
        .collect();

    let min_reward = data.iter().map(|(_, r)| *r).fold(f64::INFINITY, f64::min);
    let max_reward = data.iter().map(|(_, r)| *r).fold(f64::NEG_INFINITY, f64::max);
    let max_episode = data.last().map(|(e, _)| *e).unwrap_or(1.0);

    let datasets = vec![
        Dataset::default()
            .name("Reward")
            .marker(symbols::Marker::Braille)
            .graph_type(GraphType::Line)
            .style(Style::default().fg(ROCKET_ORANGE))
            .data(&data),
    ];

    let chart = Chart::new(datasets)
        .block(Block::default()
            .title(" Reward History ")
            .title_style(Style::default().fg(TEXT_PRIMARY))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(DARK_BORDER))
            .style(Style::default().bg(DARK_SURFACE)))
        .x_axis(Axis::default()
            .title("Episode")
            .style(Style::default().fg(TEXT_DIM))
            .bounds([0.0, max_episode])
            .labels(vec![
                Span::raw("0"),
                Span::raw(format!("{:.0}", max_episode)),
            ]))
        .y_axis(Axis::default()
            .title("Reward")
            .style(Style::default().fg(TEXT_DIM))
            .bounds([min_reward - 10.0, max_reward + 10.0])
            .labels(vec![
                Span::raw(format!("{:.0}", min_reward)),
                Span::raw(format!("{:.0}", max_reward)),
            ]));

    frame.render_widget(chart, area);
}

/// Draw loss sparklines
fn draw_loss_sparklines(frame: &mut Frame, area: Rect, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(area);

    // Policy loss
    let policy_data: Vec<u64> = (0..50)
        .map(|i| {
            let phase = (app.frame + i * 3) as f64 * 0.1;
            ((app.metrics.policy_loss + phase.sin() * 0.1) * 100.0) as u64
        })
        .collect();

    let policy_sparkline = Sparkline::default()
        .block(Block::default()
            .title(format!(" Policy Loss: {:.4} ", app.metrics.policy_loss))
            .title_style(Style::default().fg(TEXT_PRIMARY))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(DARK_BORDER))
            .style(Style::default().bg(DARK_SURFACE)))
        .data(&policy_data)
        .style(Style::default().fg(ROCKET_BLUE));
    frame.render_widget(policy_sparkline, layout[0]);

    // Value loss
    let value_data: Vec<u64> = (0..50)
        .map(|i| {
            let phase = (app.frame + i * 2) as f64 * 0.08;
            ((app.metrics.value_loss + phase.cos() * 0.15) * 100.0) as u64
        })
        .collect();

    let value_sparkline = Sparkline::default()
        .block(Block::default()
            .title(format!(" Value Loss: {:.4} ", app.metrics.value_loss))
            .title_style(Style::default().fg(TEXT_PRIMARY))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(DARK_BORDER))
            .style(Style::default().bg(DARK_SURFACE)))
        .data(&value_data)
        .style(Style::default().fg(ROCKET_PURPLE));
    frame.render_widget(value_sparkline, layout[1]);
}

/// Draw logs panel
fn draw_logs(frame: &mut Frame, area: Rect, app: &App) {
    let logs: Vec<ListItem> = app.logs
        .iter()
        .rev()
        .take(10)
        .map(|log| {
            let style = if log.contains("[ERROR]") {
                Style::default().fg(ROCKET_RED)
            } else if log.contains("[WARN]") {
                Style::default().fg(ROCKET_YELLOW)
            } else if log.contains("[TRAIN]") {
                Style::default().fg(ROCKET_GREEN)
            } else {
                Style::default().fg(TEXT_SECONDARY)
            };
            ListItem::new(Span::styled(log.clone(), style))
        })
        .collect();

    let logs_widget = List::new(logs)
        .block(Block::default()
            .title(" Logs ")
            .title_style(Style::default().fg(TEXT_PRIMARY))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(DARK_BORDER))
            .style(Style::default().bg(DARK_SURFACE)));
    frame.render_widget(logs_widget, area);
}

/// Draw training tab
fn draw_training(frame: &mut Frame, area: Rect, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .margin(1)
        .split(area);

    // Left: Detailed metrics
    let metrics_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Length(3),
            Constraint::Min(0),
        ])
        .split(layout[0]);

    let metrics_data = [
        ("Learning Rate", format!("{:.2e}", app.metrics.learning_rate)),
        ("Entropy", format!("{:.4}", app.metrics.entropy)),
        ("Explained Variance", format!("{:.4}", app.metrics.explained_variance)),
        ("KL Divergence", format!("{:.6}", app.metrics.kl_divergence)),
        ("Clip Fraction", format!("{:.4}", app.metrics.clip_fraction)),
    ];

    for (i, (label, value)) in metrics_data.iter().enumerate() {
        let metric = Paragraph::new(Line::from(vec![
            Span::styled(format!("{}: ", label), Style::default().fg(TEXT_SECONDARY)),
            Span::styled(value.clone(), Style::default().fg(ROCKET_ORANGE)),
        ]))
        .block(Block::default()
            .borders(Borders::ALL)
            .border_style(Style::default().fg(DARK_BORDER))
            .style(Style::default().bg(DARK_SURFACE)));
        frame.render_widget(metric, metrics_layout[i]);
    }

    // Right: Training progress
    draw_reward_chart(frame, layout[1], app);
}

/// Draw environment tab
fn draw_environment(frame: &mut Frame, area: Rect, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(10),
            Constraint::Min(0),
        ])
        .margin(1)
        .split(area);

    // Environment info
    let env_info = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Environment: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("TradingEnv", Style::default().fg(ROCKET_ORANGE)),
        ]),
        Line::from(vec![
            Span::styled("Observation Space: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("Box(128,)", Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(vec![
            Span::styled("Action Space: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("Box(3,)", Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(vec![
            Span::styled("Num Envs: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("8", Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Episode Length: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("2000 steps", Style::default().fg(TEXT_PRIMARY)),
        ]),
    ])
    .block(Block::default()
        .title(" Environment Configuration ")
        .title_style(Style::default().fg(TEXT_PRIMARY))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DARK_BORDER))
        .style(Style::default().bg(DARK_SURFACE)));
    frame.render_widget(env_info, layout[0]);

    // Simulated environment state
    let state_viz = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "Environment State Visualization",
            Style::default().fg(TEXT_DIM),
        )),
        Line::from(""),
        Line::from(Span::styled(
            format!("Current Position: {}", if app.frame % 60 < 30 { "LONG" } else { "SHORT" }),
            Style::default().fg(if app.frame % 60 < 30 { ROCKET_GREEN } else { ROCKET_RED }),
        )),
        Line::from(Span::styled(
            format!("Portfolio Value: ${:.2}", 10000.0 + (app.metrics.avg_reward * 10.0)),
            Style::default().fg(TEXT_PRIMARY),
        )),
    ])
    .alignment(Alignment::Center)
    .block(Block::default()
        .title(" State ")
        .title_style(Style::default().fg(TEXT_PRIMARY))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DARK_BORDER))
        .style(Style::default().bg(DARK_SURFACE)));
    frame.render_widget(state_viz, layout[1]);
}

/// Draw benchmarks tab
fn draw_benchmarks(frame: &mut Frame, area: Rect, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(12)])
        .margin(1)
        .split(area);

    // Benchmark table
    let header = Row::new(vec!["Benchmark", "Rust", "Python", "Speedup"])
        .style(Style::default().fg(ROCKET_ORANGE).add_modifier(Modifier::BOLD))
        .bottom_margin(1);

    let rows: Vec<Row> = app.benchmarks
        .iter()
        .map(|b| {
            let speedup = app.get_speedup(b);
            let speedup_str = speedup
                .map(|s| format!("{:.1}x", s))
                .unwrap_or_else(|| "-".to_string());
            let speedup_style = if speedup.unwrap_or(0.0) > 1.0 {
                Style::default().fg(ROCKET_GREEN)
            } else {
                Style::default().fg(ROCKET_RED)
            };

            Row::new(vec![
                Span::styled(&b.name, Style::default().fg(TEXT_PRIMARY)),
                Span::styled(format!("{:.2} {}", b.value, b.unit), Style::default().fg(ROCKET_BLUE)),
                Span::styled(
                    b.comparison.map(|v| format!("{:.2} {}", v, b.unit)).unwrap_or("-".to_string()),
                    Style::default().fg(TEXT_SECONDARY),
                ),
                Span::styled(speedup_str, speedup_style),
            ])
        })
        .collect();

    let table = Table::new(rows, [
        Constraint::Percentage(35),
        Constraint::Percentage(20),
        Constraint::Percentage(20),
        Constraint::Percentage(25),
    ])
    .header(header)
    .block(Block::default()
        .title(" Performance Benchmarks: Rust vs Python ")
        .title_style(Style::default().fg(TEXT_PRIMARY))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DARK_BORDER))
        .style(Style::default().bg(DARK_SURFACE)));
    frame.render_widget(table, layout[0]);

    // Speedup bar chart
    let speedups: Vec<(&str, u64)> = app.benchmarks
        .iter()
        .filter_map(|b| {
            app.get_speedup(b).map(|s| {
                let short_name = b.name.split_whitespace().next().unwrap_or(&b.name);
                (short_name, (s * 10.0) as u64)
            })
        })
        .collect();

    let bars: Vec<Bar> = speedups
        .iter()
        .map(|(name, value)| {
            Bar::default()
                .value(*value)
                .label(Line::from(*name))
                .style(Style::default().fg(ROCKET_ORANGE))
        })
        .collect();

    let barchart = BarChart::default()
        .block(Block::default()
            .title(" Speedup (10x scale) ")
            .title_style(Style::default().fg(TEXT_PRIMARY))
            .borders(Borders::ALL)
            .border_style(Style::default().fg(DARK_BORDER))
            .style(Style::default().bg(DARK_SURFACE)))
        .data(BarGroup::default().bars(&bars))
        .bar_width(8)
        .bar_gap(2)
        .bar_style(Style::default().fg(ROCKET_ORANGE))
        .value_style(Style::default().fg(TEXT_PRIMARY));
    frame.render_widget(barchart, layout[1]);
}

/// Draw settings tab
fn draw_settings(frame: &mut Frame, area: Rect, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(12),
            Constraint::Min(0),
        ])
        .margin(1)
        .split(area);

    let mode_str = match app.mode {
        super::app::AppMode::Demo => "Demo (Simulated)",
        super::app::AppMode::Training => "Training",
        super::app::AppMode::Benchmark => "Benchmark",
    };

    let settings = Paragraph::new(vec![
        Line::from(vec![
            Span::styled("Mode: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(mode_str, Style::default().fg(ROCKET_ORANGE)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Algorithm: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("PPO", Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(vec![
            Span::styled("Network: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("MLP [256, 256]", Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(vec![
            Span::styled("Device: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("CPU (Metal available)", Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Tick Rate: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled("60 FPS", Style::default().fg(TEXT_PRIMARY)),
        ]),
        Line::from(vec![
            Span::styled("Frame: ", Style::default().fg(TEXT_SECONDARY)),
            Span::styled(format!("{}", app.frame), Style::default().fg(TEXT_DIM)),
        ]),
    ])
    .block(Block::default()
        .title(" Settings ")
        .title_style(Style::default().fg(TEXT_PRIMARY))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DARK_BORDER))
        .padding(Padding::horizontal(1))
        .style(Style::default().bg(DARK_SURFACE)));
    frame.render_widget(settings, layout[0]);

    let about = Paragraph::new(vec![
        Line::from(""),
        Line::from(Span::styled(
            "Rocket-RS",
            Style::default().fg(ROCKET_ORANGE).add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "High-Performance Reinforcement Learning for Rust",
            Style::default().fg(TEXT_SECONDARY),
        )),
        Line::from(""),
        Line::from(Span::styled(
            "Built with Candle, Ratatui, and Crossterm",
            Style::default().fg(TEXT_DIM),
        )),
    ])
    .alignment(Alignment::Center)
    .block(Block::default()
        .title(" About ")
        .title_style(Style::default().fg(TEXT_PRIMARY))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(DARK_BORDER))
        .style(Style::default().bg(DARK_SURFACE)));
    frame.render_widget(about, layout[1]);
}

/// Draw footer with help text
fn draw_footer(frame: &mut Frame, area: Rect, app: &App) {
    let help_text = if app.paused {
        "[Space] Resume | [Tab] Switch Tab | [1-5] Go to Tab | [h] Help | [q] Quit"
    } else {
        "[Space] Pause | [Tab] Switch Tab | [1-5] Go to Tab | [h] Help | [q] Quit"
    };

    let footer = Paragraph::new(help_text)
        .style(Style::default().fg(TEXT_DIM))
        .alignment(Alignment::Center)
        .block(Block::default()
            .borders(Borders::TOP)
            .border_style(Style::default().fg(DARK_BORDER)));
    frame.render_widget(footer, area);
}

/// Draw help overlay
fn draw_help_overlay(frame: &mut Frame, area: Rect) {
    let block = Block::default()
        .title(" Keyboard Shortcuts ")
        .title_style(Style::default().fg(ROCKET_ORANGE).add_modifier(Modifier::BOLD))
        .borders(Borders::ALL)
        .border_style(Style::default().fg(ROCKET_ORANGE))
        .style(Style::default().bg(DARK_BG));
    frame.render_widget(block, area);

    let inner = Block::default().inner(area);
    let help_content = Paragraph::new(vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Navigation", Style::default().fg(ROCKET_ORANGE).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  Tab / Shift+Tab    ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("Next/Previous tab", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("  1-5                ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("Jump to specific tab", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("  j/k or Up/Down     ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("Scroll content", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Controls", Style::default().fg(ROCKET_ORANGE).add_modifier(Modifier::BOLD)),
        ]),
        Line::from(vec![
            Span::styled("  Space / p          ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("Pause/Resume training", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("  h / ?              ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("Toggle this help", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(vec![
            Span::styled("  q / Esc / Ctrl+C   ", Style::default().fg(TEXT_PRIMARY)),
            Span::styled("Quit application", Style::default().fg(TEXT_SECONDARY)),
        ]),
        Line::from(""),
        Line::from(Span::styled(
            "Press any key to close",
            Style::default().fg(TEXT_DIM),
        )),
    ])
    .alignment(Alignment::Left)
    .wrap(Wrap { trim: false });
    frame.render_widget(help_content, inner);
}

/// Helper function to create a centered rectangle
fn centered_rect(percent_x: u16, percent_y: u16, r: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(r);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}
