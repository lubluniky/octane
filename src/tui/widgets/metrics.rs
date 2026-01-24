//! System metrics visualization widgets for the TUI.
//!
//! Provides gauges for CPU, GPU, and memory usage monitoring.

use crate::tui::App;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    text::Span,
    widgets::{Block, Borders, Gauge},
    Frame,
};

/// Render the CPU usage gauge with hardware name.
pub fn render_cpu_gauge(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let cpu = app.system_metrics.cpu_usage;
    let ratio = (cpu / 100.0).clamp(0.0, 1.0) as f64;

    let color = usage_to_color(cpu);

    // Truncate CPU name for title
    let cpu_name = &app.system_metrics.cpu_name;
    let title = if cpu_name.len() > 30 {
        format!(" {} ({} cores) ", &cpu_name[..27], app.system_metrics.cpu_cores)
    } else if !cpu_name.is_empty() && cpu_name != "Unknown CPU" {
        format!(" {} ({} cores) ", cpu_name, app.system_metrics.cpu_cores)
    } else {
        format!(" CPU ({} cores) ", app.system_metrics.cpu_cores)
    };

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    title,
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .gauge_style(Style::default().fg(color).bg(Color::Rgb(30, 30, 40)))
        .ratio(ratio)
        .label(Span::styled(
            format!("{:.1}%", cpu),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    frame.render_widget(gauge, area);
}

/// Render the GPU usage gauge with hardware name.
pub fn render_gpu_gauge(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let gpu = app.system_metrics.gpu_usage;
    let ratio = (gpu / 100.0).clamp(0.0, 1.0) as f64;

    let color = usage_to_color(gpu);

    // Build title with GPU name and memory info
    let gpu_name = &app.system_metrics.gpu_name;
    let title = if app.system_metrics.gpu_memory_total_mb > 0.0 {
        let mem_gb = app.system_metrics.gpu_memory_total_mb / 1024.0;
        if gpu_name.len() > 20 {
            format!(" {} ({:.0}GB) ", &gpu_name[..17], mem_gb)
        } else if !gpu_name.is_empty() {
            format!(" {} ({:.0}GB) ", gpu_name, mem_gb)
        } else {
            format!(" GPU ({:.0}GB) ", mem_gb)
        }
    } else {
        format!(" {} ", gpu_name)
    };

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    title,
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .gauge_style(Style::default().fg(color).bg(Color::Rgb(30, 30, 40)))
        .ratio(ratio)
        .label(Span::styled(
            format!("{:.1}%", gpu),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    frame.render_widget(gauge, area);
}

/// Render the memory usage gauge.
pub fn render_memory_gauge(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let used = app.system_metrics.memory_used_mb;
    let total = app.system_metrics.memory_total_mb;
    let ratio = if total > 0.0 {
        (used / total).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let usage_percent = ratio * 100.0;
    let color = usage_to_color(usage_percent as f32);

    // Show memory in GB for readability
    let used_gb = used / 1024.0;
    let total_gb = total / 1024.0;

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    format!(" Memory ({:.1}/{:.1} GB) ", used_gb, total_gb),
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .gauge_style(Style::default().fg(color).bg(Color::Rgb(30, 30, 40)))
        .ratio(ratio)
        .label(Span::styled(
            format!("{:.1}%", usage_percent),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    frame.render_widget(gauge, area);
}

/// Convert usage percentage to color (green = low, red = high).
fn usage_to_color(usage: f32) -> Color {
    if usage < 50.0 {
        Color::Green
    } else if usage < 70.0 {
        Color::Yellow
    } else if usage < 85.0 {
        Color::Rgb(255, 165, 0) // Orange
    } else {
        Color::Red
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_usage_to_color() {
        assert!(matches!(usage_to_color(30.0), Color::Green));
        assert!(matches!(usage_to_color(60.0), Color::Yellow));
        assert!(matches!(usage_to_color(80.0), Color::Rgb(255, 165, 0)));
        assert!(matches!(usage_to_color(95.0), Color::Red));
    }
}
