//! Training visualization widgets for the TUI.
//!
//! Provides real-time visualization of training progress including
//! reward graphs, loss gauges, and episode counters.

use crate::tui::App;
use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Gauge, Paragraph, Sparkline},
    Frame,
};

/// Render the reward history as a sparkline graph.
pub fn render_reward_sparkline(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let title = if app.training_active {
        Span::styled(
            " Reward History (LIVE) ",
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD | Modifier::SLOW_BLINK),
        )
    } else {
        Span::styled(
            " Reward History ",
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        )
    };

    let sparkline = Sparkline::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(title),
        )
        .data(&app.reward_history)
        .style(Style::default().fg(get_reward_color(app)));

    frame.render_widget(sparkline, area);
}

/// Get color for reward visualization based on trend.
fn get_reward_color(app: &App) -> Color {
    if app.reward_history.len() < 2 {
        return Color::Cyan;
    }

    let recent = &app.reward_history[app.reward_history.len().saturating_sub(5)..];
    if recent.is_empty() {
        return Color::Cyan;
    }

    let avg_recent: u64 = recent.iter().sum::<u64>() / recent.len() as u64;
    let avg_old: u64 =
        app.reward_history.iter().take(5).sum::<u64>() / app.reward_history.len().min(5) as u64;

    if avg_recent > avg_old + 5 {
        Color::Green
    } else if avg_recent + 5 < avg_old {
        Color::Red
    } else {
        Color::Cyan
    }
}

/// Render episode information panel.
pub fn render_episode_info(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let metrics = &app.current_metrics;

    // Calculate reward trend arrow
    let (trend_icon, trend_color) = if app.reward_history.len() >= 2 {
        let last = *app.reward_history.last().unwrap_or(&0);
        let prev = *app
            .reward_history
            .get(app.reward_history.len() - 2)
            .unwrap_or(&0);
        if last > prev + 2 {
            ("^", Color::Green)
        } else if last + 2 < prev {
            ("v", Color::Red)
        } else {
            ("-", Color::Yellow)
        }
    } else {
        ("-", Color::Gray)
    };

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Episode: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>6}", metrics.episodes),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Mean Reward: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>8.2}", metrics.mean_reward),
                Style::default()
                    .fg(if metrics.mean_reward > 0.0 {
                        Color::Green
                    } else {
                        Color::Red
                    })
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" "),
            Span::styled(trend_icon, Style::default().fg(trend_color)),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Std Reward: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>8.2}", metrics.std_reward),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Best: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!(
                    "{:>8.2}",
                    app.metrics_history
                        .iter()
                        .map(|m| m.mean_reward)
                        .fold(f32::NEG_INFINITY, f32::max)
                        .max(metrics.mean_reward)
                ),
                Style::default()
                    .fg(Color::Yellow)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
    ];

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " Episode Info ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .alignment(Alignment::Left);

    frame.render_widget(widget, area);
}

/// Render the policy loss gauge.
pub fn render_policy_loss_gauge(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let loss = app.current_metrics.policy_loss;

    // Normalize loss to 0-100 range (assuming loss is typically 0-1)
    let ratio = (loss * 100.0).clamp(0.0, 100.0) / 100.0;

    let color = loss_to_color(loss, 0.5);

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " Policy Loss ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .gauge_style(Style::default().fg(color).bg(Color::Rgb(30, 30, 40)))
        .ratio(ratio as f64)
        .label(Span::styled(
            format!("{:.6}", loss),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    frame.render_widget(gauge, area);
}

/// Render the value loss gauge.
pub fn render_value_loss_gauge(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let loss = app.current_metrics.value_loss;

    // Normalize loss to 0-100 range
    let ratio = (loss * 100.0).clamp(0.0, 100.0) / 100.0;

    let color = loss_to_color(loss, 0.3);

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " Value Loss ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .gauge_style(Style::default().fg(color).bg(Color::Rgb(30, 30, 40)))
        .ratio(ratio as f64)
        .label(Span::styled(
            format!("{:.6}", loss),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    frame.render_widget(gauge, area);
}

/// Convert loss value to color (green = low, red = high).
fn loss_to_color(loss: f32, threshold: f32) -> Color {
    if loss < threshold * 0.3 {
        Color::Green
    } else if loss < threshold * 0.6 {
        Color::Yellow
    } else if loss < threshold {
        Color::Rgb(255, 165, 0) // Orange
    } else {
        Color::Red
    }
}

/// Render training statistics panel.
pub fn render_training_stats(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let metrics = &app.current_metrics;

    let lines = vec![
        Line::from(""),
        Line::from(vec![
            Span::styled("Timesteps: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>10}", format_number(metrics.timesteps)),
                Style::default()
                    .fg(Color::Green)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Steps/sec: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>10.0}", app.steps_per_second),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Entropy: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>10.4}", metrics.entropy),
                Style::default().fg(Color::Magenta),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("KL Div: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>10.6}", metrics.approx_kl),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
        Line::from(""),
        Line::from(vec![
            Span::styled("Clip Frac: ", Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>10.2}%", metrics.clip_fraction * 100.0),
                Style::default().fg(Color::DarkGray),
            ),
        ]),
    ];

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " Training Stats ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .alignment(Alignment::Left);

    frame.render_widget(widget, area);
}

/// Format large numbers with separators.
fn format_number(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Render a progress bar for training completion.
pub fn render_training_progress(frame: &mut Frame<'_>, app: &App, area: Rect, total_timesteps: usize) {
    let progress = if total_timesteps > 0 {
        app.current_metrics.timesteps as f64 / total_timesteps as f64
    } else {
        0.0
    };

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " Training Progress ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .gauge_style(Style::default().fg(Color::Green).bg(Color::Rgb(30, 30, 40)))
        .ratio(progress.clamp(0.0, 1.0))
        .label(Span::styled(
            format!(
                "{:.1}% ({}/{})",
                progress * 100.0,
                format_number(app.current_metrics.timesteps),
                format_number(total_timesteps)
            ),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    frame.render_widget(gauge, area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(1000000), "1,000,000");
        assert_eq!(format_number(123), "123");
    }

    #[test]
    fn test_loss_to_color() {
        assert!(matches!(loss_to_color(0.01, 0.5), Color::Green));
        assert!(matches!(loss_to_color(0.2, 0.5), Color::Yellow));
        assert!(matches!(loss_to_color(1.0, 0.5), Color::Red));
    }
}
