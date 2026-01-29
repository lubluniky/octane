//! Color theme for Octane TUI.
//!
//! Professional dark theme with consistent color scheme.

use ratatui::style::{Color, Modifier, Style};

/// Octane color palette - professional dark theme.
pub struct Theme;

impl Theme {
    // ===== Background Colors =====
    /// Main background color.
    pub const BG_PRIMARY: Color = Color::Rgb(15, 17, 26);
    /// Secondary background (cards, panels).
    pub const BG_SECONDARY: Color = Color::Rgb(22, 25, 37);
    /// Tertiary background (hover, selected).
    pub const BG_TERTIARY: Color = Color::Rgb(30, 34, 50);
    /// Surface color for elevated elements.
    pub const BG_SURFACE: Color = Color::Rgb(38, 42, 60);

    // ===== Text Colors =====
    /// Primary text color.
    pub const TEXT_PRIMARY: Color = Color::Rgb(230, 235, 245);
    /// Secondary text (dimmed).
    pub const TEXT_SECONDARY: Color = Color::Rgb(160, 170, 190);
    /// Muted text (hints, placeholders).
    pub const TEXT_MUTED: Color = Color::Rgb(100, 110, 130);

    // ===== Accent Colors =====
    /// Primary accent (brand color - rocket orange).
    pub const ACCENT_PRIMARY: Color = Color::Rgb(255, 140, 50);
    /// Secondary accent (cyan).
    pub const ACCENT_SECONDARY: Color = Color::Rgb(80, 200, 255);
    /// Tertiary accent (purple).
    pub const ACCENT_TERTIARY: Color = Color::Rgb(180, 130, 255);

    // ===== Semantic Colors =====
    /// Success color (green).
    pub const SUCCESS: Color = Color::Rgb(80, 220, 130);
    /// Warning color (yellow).
    pub const WARNING: Color = Color::Rgb(255, 200, 70);
    /// Error color (red).
    pub const ERROR: Color = Color::Rgb(255, 90, 90);
    /// Info color (blue).
    pub const INFO: Color = Color::Rgb(80, 170, 255);

    // ===== Chart Colors =====
    /// Reward chart color.
    pub const CHART_REWARD: Color = Color::Rgb(80, 220, 130);
    /// Loss chart color.
    pub const CHART_LOSS: Color = Color::Rgb(255, 140, 50);
    /// Entropy chart color.
    pub const CHART_ENTROPY: Color = Color::Rgb(180, 130, 255);
    /// Value chart color.
    pub const CHART_VALUE: Color = Color::Rgb(80, 200, 255);

    // ===== Styles =====

    /// Default style.
    pub fn default() -> Style {
        Style::default()
            .fg(Self::TEXT_PRIMARY)
            .bg(Self::BG_PRIMARY)
    }

    /// Title style.
    pub fn title() -> Style {
        Style::default()
            .fg(Self::ACCENT_PRIMARY)
            .add_modifier(Modifier::BOLD)
    }

    /// Subtitle style.
    pub fn subtitle() -> Style {
        Style::default()
            .fg(Self::TEXT_SECONDARY)
    }

    /// Highlighted text.
    pub fn highlight() -> Style {
        Style::default()
            .fg(Self::TEXT_PRIMARY)
            .bg(Self::BG_TERTIARY)
    }

    /// Selected item.
    pub fn selected() -> Style {
        Style::default()
            .fg(Self::BG_PRIMARY)
            .bg(Self::ACCENT_PRIMARY)
            .add_modifier(Modifier::BOLD)
    }

    /// Active tab style.
    pub fn tab_active() -> Style {
        Style::default()
            .fg(Self::ACCENT_PRIMARY)
            .add_modifier(Modifier::BOLD)
    }

    /// Inactive tab style.
    pub fn tab_inactive() -> Style {
        Style::default()
            .fg(Self::TEXT_MUTED)
    }

    /// Border style.
    pub fn border() -> Style {
        Style::default()
            .fg(Self::BG_SURFACE)
    }

    /// Border style (focused).
    pub fn border_focused() -> Style {
        Style::default()
            .fg(Self::ACCENT_PRIMARY)
    }

    /// Label style.
    pub fn label() -> Style {
        Style::default()
            .fg(Self::TEXT_SECONDARY)
    }

    /// Value style (metrics).
    pub fn value() -> Style {
        Style::default()
            .fg(Self::TEXT_PRIMARY)
            .add_modifier(Modifier::BOLD)
    }

    /// Positive value style.
    pub fn value_positive() -> Style {
        Style::default()
            .fg(Self::SUCCESS)
            .add_modifier(Modifier::BOLD)
    }

    /// Negative value style.
    pub fn value_negative() -> Style {
        Style::default()
            .fg(Self::ERROR)
            .add_modifier(Modifier::BOLD)
    }

    /// Progress bar style (filled).
    pub fn progress_filled() -> Style {
        Style::default()
            .fg(Self::ACCENT_PRIMARY)
    }

    /// Progress bar style (empty).
    pub fn progress_empty() -> Style {
        Style::default()
            .fg(Self::BG_SURFACE)
    }

    /// Log level styles.
    pub fn log_info() -> Style {
        Style::default().fg(Self::INFO)
    }

    /// Warning log style.
    pub fn log_warn() -> Style {
        Style::default().fg(Self::WARNING)
    }

    /// Error log style.
    pub fn log_error() -> Style {
        Style::default().fg(Self::ERROR)
    }

    /// Debug log style.
    pub fn log_debug() -> Style {
        Style::default().fg(Self::TEXT_MUTED)
    }

    /// Sparkline style.
    pub fn sparkline(positive: bool) -> Style {
        if positive {
            Style::default().fg(Self::SUCCESS)
        } else {
            Style::default().fg(Self::ERROR)
        }
    }

    /// Status indicator styles.
    pub fn status_running() -> Style {
        Style::default()
            .fg(Self::SUCCESS)
            .add_modifier(Modifier::BOLD)
    }

    /// Paused status style.
    pub fn status_paused() -> Style {
        Style::default()
            .fg(Self::WARNING)
            .add_modifier(Modifier::BOLD)
    }

    /// Complete status style.
    pub fn status_complete() -> Style {
        Style::default()
            .fg(Self::ACCENT_SECONDARY)
            .add_modifier(Modifier::BOLD)
    }

    /// Error status style.
    pub fn status_error() -> Style {
        Style::default()
            .fg(Self::ERROR)
            .add_modifier(Modifier::BOLD)
    }
}

/// Format a number with appropriate precision and color.
pub fn format_metric(value: f64, precision: usize) -> String {
    if value.abs() >= 1_000_000.0 {
        format!("{:.1}M", value / 1_000_000.0)
    } else if value.abs() >= 1_000.0 {
        format!("{:.1}K", value / 1_000.0)
    } else {
        format!("{:.prec$}", value, prec = precision)
    }
}

/// Format duration in human-readable format.
pub fn format_duration(seconds: u64) -> String {
    if seconds < 60 {
        format!("{}s", seconds)
    } else if seconds < 3600 {
        format!("{}m {}s", seconds / 60, seconds % 60)
    } else {
        format!("{}h {}m", seconds / 3600, (seconds % 3600) / 60)
    }
}

/// Create a styled block with the theme.
pub fn themed_block(title: &str, focused: bool) -> ratatui::widgets::Block<'_> {
    use ratatui::widgets::{Block, Borders};

    let border_style = if focused {
        Theme::border_focused()
    } else {
        Theme::border()
    };

    Block::default()
        .title(title)
        .title_style(Theme::title())
        .borders(Borders::ALL)
        .border_style(border_style)
        .style(Style::default().bg(Theme::BG_SECONDARY))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_metric() {
        assert_eq!(format_metric(1234567.0, 2), "1.2M");
        assert_eq!(format_metric(12345.0, 2), "12.3K");
        assert_eq!(format_metric(123.456, 2), "123.46");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(format_duration(45), "45s");
        assert_eq!(format_duration(125), "2m 5s");
        assert_eq!(format_duration(7265), "2h 1m");
    }
}
