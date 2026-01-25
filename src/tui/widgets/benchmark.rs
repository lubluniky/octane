//! Benchmark visualization widgets for the TUI.
//!
//! Provides comparison charts and throughput displays for
//! Rust vs Python performance benchmarks.

use crate::tui::App;
use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{BarChart, Block, Borders, Gauge, Paragraph},
    Frame,
};

/// Render benchmark comparison bars (Rust vs Python).
pub fn render_comparison_bars(frame: &mut Frame<'_>, app: &App, area: Rect) {
    // Prepare bar chart data
    let data: Vec<(&str, u64)> = app
        .benchmark_data
        .iter()
        .map(|(name, rust_val, _)| (name.as_str(), *rust_val as u64))
        .collect();

    let bar_chart = BarChart::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " Rust Performance (ops/sec) ",
                    Style::default()
                        .fg(Color::Cyan)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .data(&data)
        .bar_width(9)
        .bar_gap(2)
        .bar_style(Style::default().fg(Color::Green))
        .value_style(
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        );

    frame.render_widget(bar_chart, area);
}

/// Render throughput gauge showing overall performance.
pub fn render_throughput_gauge(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let tensor_ops = app.system_metrics.tensor_ops_per_sec;

    // Normalize to 0-1 range (assuming max ~5M ops/sec)
    let ratio = (tensor_ops / 5_000_000.0).clamp(0.0, 1.0);

    let color = if ratio > 0.7 {
        Color::Green
    } else if ratio > 0.4 {
        Color::Yellow
    } else {
        Color::Red
    };

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " Tensor Throughput ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .gauge_style(Style::default().fg(color).bg(Color::Rgb(30, 30, 40)))
        .ratio(ratio)
        .label(Span::styled(
            format!("{:.2}M ops/sec", tensor_ops / 1_000_000.0),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ));

    frame.render_widget(gauge, area);
}

/// Render speedup indicators comparing Rust vs Python.
pub fn render_speedup_indicators(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let mut lines: Vec<Line<'_>> = vec![Line::from("")];

    for (name, rust_val, python_val) in &app.benchmark_data {
        let speedup = if *python_val > 0.0 {
            rust_val / python_val
        } else {
            0.0
        };

        let speedup_color = if speedup > 5.0 {
            Color::Green
        } else if speedup > 2.0 {
            Color::Yellow
        } else if speedup > 1.0 {
            Color::Cyan
        } else {
            Color::Red
        };

        lines.push(Line::from(vec![
            Span::styled(format!("{:<15}", name), Style::default().fg(Color::Gray)),
            Span::styled(
                format!("{:>6.1}x", speedup),
                Style::default()
                    .fg(speedup_color)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::raw(" faster"),
        ]));
    }

    // Add summary line
    let total_rust: f64 = app.benchmark_data.iter().map(|(_, r, _)| r).sum();
    let total_python: f64 = app.benchmark_data.iter().map(|(_, _, p)| p).sum();
    let avg_speedup = if total_python > 0.0 {
        total_rust / total_python
    } else {
        0.0
    };

    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("Average: ", Style::default().fg(Color::White)),
        Span::styled(
            format!("{:.1}x", avg_speedup),
            Style::default()
                .fg(Color::Green)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" speedup", Style::default().fg(Color::White)),
    ]));

    let widget = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::Rgb(80, 80, 100)))
                .title(Span::styled(
                    " Rust vs Python ",
                    Style::default()
                        .fg(Color::Rgb(255, 165, 0))
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .alignment(Alignment::Left);

    frame.render_widget(widget, area);
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_speedup_calculation() {
        let rust_val: f64 = 1000.0;
        let python_val: f64 = 200.0;
        let speedup = rust_val / python_val;
        assert!((speedup - 5.0).abs() < 0.001);
    }
}
