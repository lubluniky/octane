//! Enhanced chart widgets for the TUI.
//!
//! Provides advanced visualization components including sparklines,
//! multi-line charts, progress bars, and live metric displays.

use crate::tui::theme::Theme;
use ratatui::{
    layout::Rect,
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{
        Axis, Block, Borders, Chart, Dataset, Gauge, GraphType, Sparkline, Paragraph,
    },
    Frame,
};

/// Multi-metric sparkline widget showing multiple metrics in one view
pub struct MultiSparkline<'a> {
    /// Block wrapper for the sparkline
    block: Option<Block<'a>>,
    /// Data series (label, data, color)
    series: Vec<(&'a str, &'a [u64], Color)>,
    /// Maximum value for scaling (None = auto)
    max_value: Option<u64>,
}

impl<'a> MultiSparkline<'a> {
    pub fn new() -> Self {
        Self {
            block: None,
            series: Vec::new(),
            max_value: None,
        }
    }

    pub fn block(mut self, block: Block<'a>) -> Self {
        self.block = Some(block);
        self
    }

    pub fn add_series(mut self, label: &'a str, data: &'a [u64], color: Color) -> Self {
        self.series.push((label, data, color));
        self
    }

    pub fn max_value(mut self, max: u64) -> Self {
        self.max_value = Some(max);
        self
    }

    pub fn render(self, frame: &mut Frame<'_>, area: Rect) {
        if self.series.is_empty() {
            return;
        }

        // Calculate layout for each sparkline
        let height_per_series = (area.height.saturating_sub(2)) / self.series.len() as u16;

        let inner = self.block.as_ref().map_or(area, |b| b.inner(area));

        // Render block if present
        if let Some(block) = self.block {
            frame.render_widget(block, area);
        }

        // Render each series
        for (i, (label, data, color)) in self.series.iter().enumerate() {
            let y = inner.y + (i as u16 * height_per_series);
            let series_area = Rect {
                x: inner.x,
                y,
                width: inner.width,
                height: height_per_series,
            };

            let sparkline = Sparkline::default()
                .data(data)
                .style(Style::default().fg(*color))
                .block(
                    Block::default()
                        .title(Span::styled(
                            format!(" {} ", label),
                            Style::default()
                                .fg(Theme::TEXT_SECONDARY)
                                .add_modifier(Modifier::ITALIC),
                        ))
                        .borders(Borders::NONE),
                );

            frame.render_widget(sparkline, series_area);
        }
    }
}

/// Live metric display with trend indicator
pub struct LiveMetric<'a> {
    label: &'a str,
    value: f64,
    format: MetricFormat,
    trend: Option<TrendDirection>,
    color: Option<Color>,
}

#[derive(Clone, Copy)]
pub enum MetricFormat {
    Integer,
    Float1,
    Float2,
    Float3,
    Scientific,
    Percentage,
    Throughput, // K, M, B suffixes
}

#[derive(Clone, Copy)]
pub enum TrendDirection {
    Up,
    Down,
    Neutral,
}

impl<'a> LiveMetric<'a> {
    pub fn new(label: &'a str, value: f64) -> Self {
        Self {
            label,
            value,
            format: MetricFormat::Float2,
            trend: None,
            color: None,
        }
    }

    pub fn format(mut self, format: MetricFormat) -> Self {
        self.format = format;
        self
    }

    pub fn trend(mut self, trend: TrendDirection) -> Self {
        self.trend = Some(trend);
        self
    }

    pub fn color(mut self, color: Color) -> Self {
        self.color = Some(color);
        self
    }

    fn format_value(&self) -> String {
        match self.format {
            MetricFormat::Integer => format!("{}", self.value as i64),
            MetricFormat::Float1 => format!("{:.1}", self.value),
            MetricFormat::Float2 => format!("{:.2}", self.value),
            MetricFormat::Float3 => format!("{:.3}", self.value),
            MetricFormat::Scientific => format!("{:.2e}", self.value),
            MetricFormat::Percentage => format!("{:.1}%", self.value),
            MetricFormat::Throughput => format_throughput(self.value),
        }
    }

    pub fn render(self, frame: &mut Frame<'_>, area: Rect) {
        let trend_icon = match self.trend {
            Some(TrendDirection::Up) => "▲",
            Some(TrendDirection::Down) => "▼",
            Some(TrendDirection::Neutral) => "━",
            None => "",
        };

        let trend_color = match self.trend {
            Some(TrendDirection::Up) => Theme::SUCCESS,
            Some(TrendDirection::Down) => Theme::ERROR,
            Some(TrendDirection::Neutral) => Theme::WARNING,
            None => Theme::TEXT_TERTIARY,
        };

        let value_color = self.color.unwrap_or(Theme::TEXT_PRIMARY);

        let lines = vec![
            Line::from(Span::styled(
                self.label,
                Style::default().fg(Theme::TEXT_SECONDARY),
            )),
            Line::from(""),
            Line::from(vec![
                Span::styled(
                    self.format_value(),
                    Style::default()
                        .fg(value_color)
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(" "),
                Span::styled(trend_icon, Style::default().fg(trend_color)),
            ]),
        ];

        let widget = Paragraph::new(lines)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Theme::BORDER_DEFAULT))
                    .style(Style::default().bg(Theme::BG_TERTIARY)),
            )
            .alignment(ratatui::layout::Alignment::Center);

        frame.render_widget(widget, area);
    }
}

/// Enhanced progress bar with gradient and time estimation
pub struct EnhancedProgress<'a> {
    label: &'a str,
    current: u64,
    total: u64,
    show_eta: bool,
    start_time: Option<std::time::Instant>,
    use_gradient: bool,
}

impl<'a> EnhancedProgress<'a> {
    pub fn new(label: &'a str, current: u64, total: u64) -> Self {
        Self {
            label,
            current,
            total,
            show_eta: false,
            start_time: None,
            use_gradient: true,
        }
    }

    pub fn with_eta(mut self, start_time: std::time::Instant) -> Self {
        self.show_eta = true;
        self.start_time = Some(start_time);
        self
    }

    pub fn gradient(mut self, use_gradient: bool) -> Self {
        self.use_gradient = use_gradient;
        self
    }

    fn calculate_eta(&self) -> Option<String> {
        if !self.show_eta || self.start_time.is_none() || self.current == 0 {
            return None;
        }

        let elapsed = self.start_time?.elapsed().as_secs_f64();
        let progress = self.current as f64 / self.total as f64;

        if progress > 0.0 && progress < 1.0 {
            let total_time = elapsed / progress;
            let remaining = total_time - elapsed;

            Some(format_duration(remaining))
        } else {
            None
        }
    }

    pub fn render(self, frame: &mut Frame<'_>, area: Rect) {
        let ratio = if self.total > 0 {
            (self.current as f64 / self.total as f64).clamp(0.0, 1.0)
        } else {
            0.0
        };

        let color = if self.use_gradient {
            Theme::gradient_progress(ratio)
        } else {
            Theme::BRAND_PRIMARY
        };

        let mut label_text = format!(
            "{:.1}% ({}/{})",
            ratio * 100.0,
            format_number(self.current),
            format_number(self.total)
        );

        if let Some(eta) = self.calculate_eta() {
            label_text.push_str(&format!(" | ETA: {}", eta));
        }

        let gauge = Gauge::default()
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Theme::BORDER_DEFAULT))
                    .title(Span::styled(
                        format!(" {} ", self.label),
                        Style::default()
                            .fg(Theme::TEXT_PRIMARY)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .style(Style::default().bg(Theme::BG_TERTIARY)),
            )
            .gauge_style(Style::default().fg(color).bg(Theme::BG_SECONDARY))
            .ratio(ratio)
            .label(Span::styled(
                label_text,
                Style::default()
                    .fg(Theme::TEXT_PRIMARY)
                    .add_modifier(Modifier::BOLD),
            ));

        frame.render_widget(gauge, area);
    }
}

/// Real-time line chart with multiple datasets
pub struct RealtimeChart<'a> {
    title: &'a str,
    datasets: Vec<(&'a str, Vec<(f64, f64)>, Color)>,
    x_bounds: Option<[f64; 2]>,
    y_bounds: Option<[f64; 2]>,
    x_labels: Option<Vec<String>>,
    y_labels: Option<Vec<String>>,
}

impl<'a> RealtimeChart<'a> {
    pub fn new(title: &'a str) -> Self {
        Self {
            title,
            datasets: Vec::new(),
            x_bounds: None,
            y_bounds: None,
            x_labels: None,
            y_labels: None,
        }
    }

    pub fn add_dataset(
        mut self,
        label: &'a str,
        data: Vec<(f64, f64)>,
        color: Color,
    ) -> Self {
        self.datasets.push((label, data, color));
        self
    }

    pub fn x_bounds(mut self, bounds: [f64; 2]) -> Self {
        self.x_bounds = Some(bounds);
        self
    }

    pub fn y_bounds(mut self, bounds: [f64; 2]) -> Self {
        self.y_bounds = Some(bounds);
        self
    }

    pub fn render(self, frame: &mut Frame<'_>, area: Rect) {
        if self.datasets.is_empty() {
            return;
        }

        // Calculate bounds if not provided
        let (x_min, x_max, y_min, y_max) = if self.x_bounds.is_none() || self.y_bounds.is_none()
        {
            let mut x_min = f64::INFINITY;
            let mut x_max = f64::NEG_INFINITY;
            let mut y_min = f64::INFINITY;
            let mut y_max = f64::NEG_INFINITY;

            for (_, data, _) in &self.datasets {
                for (x, y) in data {
                    x_min = x_min.min(*x);
                    x_max = x_max.max(*x);
                    y_min = y_min.min(*y);
                    y_max = y_max.max(*y);
                }
            }

            // Add 10% padding
            let x_range = x_max - x_min;
            let y_range = y_max - y_min;
            (
                x_min - x_range * 0.05,
                x_max + x_range * 0.05,
                y_min - y_range * 0.1,
                y_max + y_range * 0.1,
            )
        } else {
            let [x_min, x_max] = self.x_bounds.unwrap();
            let [y_min, y_max] = self.y_bounds.unwrap();
            (x_min, x_max, y_min, y_max)
        };

        // Create datasets
        let datasets: Vec<Dataset> = self
            .datasets
            .iter()
            .map(|(label, data, color)| {
                Dataset::default()
                    .name(*label)
                    .marker(symbols::Marker::Braille)
                    .graph_type(GraphType::Line)
                    .style(Style::default().fg(*color))
                    .data(data)
            })
            .collect();

        let chart = Chart::new(datasets)
            .block(
                Block::default()
                    .title(Span::styled(
                        format!(" {} ", self.title),
                        Style::default()
                            .fg(Theme::TEXT_PRIMARY)
                            .add_modifier(Modifier::BOLD),
                    ))
                    .borders(Borders::ALL)
                    .border_style(Style::default().fg(Theme::BORDER_DEFAULT))
                    .style(Style::default().bg(Theme::BG_TERTIARY)),
            )
            .x_axis(
                Axis::default()
                    .style(Style::default().fg(Theme::TEXT_TERTIARY))
                    .bounds([x_min, x_max])
                    .labels(vec![
                        Span::raw(format!("{:.0}", x_min)),
                        Span::raw(format!("{:.0}", x_max)),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .style(Style::default().fg(Theme::TEXT_TERTIARY))
                    .bounds([y_min, y_max])
                    .labels(vec![
                        Span::raw(format!("{:.2}", y_min)),
                        Span::raw(format!("{:.2}", y_max)),
                    ]),
            );

        frame.render_widget(chart, area);
    }
}

// === Helper Functions ===

/// Format large numbers with K, M, B suffixes
fn format_throughput(value: f64) -> String {
    if value >= 1_000_000_000.0 {
        format!("{:.1}B", value / 1_000_000_000.0)
    } else if value >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if value >= 1_000.0 {
        format!("{:.1}K", value / 1_000.0)
    } else {
        format!("{:.0}", value)
    }
}

/// Format number with thousands separators
fn format_number(n: u64) -> String {
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

/// Format duration in human-readable form
fn format_duration(seconds: f64) -> String {
    if seconds < 60.0 {
        format!("{:.0}s", seconds)
    } else if seconds < 3600.0 {
        format!("{:.0}m {:.0}s", seconds / 60.0, seconds % 60.0)
    } else {
        let hours = (seconds / 3600.0).floor();
        let minutes = ((seconds % 3600.0) / 60.0).floor();
        format!("{:.0}h {:.0}m", hours, minutes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_throughput() {
        assert_eq!(format_throughput(500.0), "500");
        assert_eq!(format_throughput(1_500.0), "1.5K");
        assert_eq!(format_throughput(1_500_000.0), "1.5M");
        assert_eq!(format_throughput(1_500_000_000.0), "1.5B");
    }

    #[test]
    fn test_format_number() {
        assert_eq!(format_number(123), "123");
        assert_eq!(format_number(1_234), "1,234");
        assert_eq!(format_number(1_234_567), "1,234,567");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(45.0), "45s");
        assert_eq!(format_duration(125.0), "2m 5s");
        assert_eq!(format_duration(3665.0), "1h 1m");
    }
}
