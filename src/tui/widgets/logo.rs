//! ASCII art logo widget for RocketRL.

use crate::tui::App;
use ratatui::{
    layout::{Alignment, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph},
    Frame,
};

/// The main ROCKET ASCII art logo.
const ROCKET_LOGO: &[&str] = &[
    r"██████╗  ██████╗  ██████╗██╗  ██╗███████╗████████╗",
    r"██╔══██╗██╔═══██╗██╔════╝██║ ██╔╝██╔════╝╚══██╔══╝",
    r"██████╔╝██║   ██║██║     █████╔╝ █████╗     ██║   ",
    r"██╔══██╗██║   ██║██║     ██╔═██╗ ██╔══╝     ██║   ",
    r"██║  ██║╚██████╔╝╚██████╗██║  ██╗███████╗   ██║   ",
    r"╚═╝  ╚═╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝   ╚═╝   ",
];

/// Tagline displayed below the logo.
const TAGLINE: &str = "High-Performance Reinforcement Learning for Rust";

/// Render the big ASCII logo with color cycling animation.
pub fn render_logo(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let mut lines: Vec<Line<'_>> = Vec::new();

    // Add empty line for top padding
    lines.push(Line::from(""));

    // Create color gradient for each row of the logo
    for (i, logo_line) in ROCKET_LOGO.iter().enumerate() {
        let color = get_gradient_color(i, ROCKET_LOGO.len(), app.tick);
        lines.push(Line::from(Span::styled(
            *logo_line,
            Style::default().fg(color).add_modifier(Modifier::BOLD),
        )));
    }

    // Add spacing
    lines.push(Line::from(""));

    // Add tagline with pulsing effect
    let pulse = app.pulse();
    let tagline_color = Color::Rgb(
        (150.0 + 105.0 * pulse) as u8,
        (150.0 + 105.0 * pulse) as u8,
        (200.0 + 55.0 * pulse) as u8,
    );

    lines.push(Line::from(Span::styled(
        TAGLINE,
        Style::default()
            .fg(tagline_color)
            .add_modifier(Modifier::ITALIC),
    )));

    // Add version info
    lines.push(Line::from(""));
    lines.push(Line::from(vec![
        Span::styled("v0.1.0", Style::default().fg(Color::DarkGray)),
        Span::raw("  |  "),
        Span::styled("Powered by Candle", Style::default().fg(Color::DarkGray)),
        Span::raw("  |  "),
        Span::styled("Metal + CUDA", Style::default().fg(Color::DarkGray)),
    ]));

    let logo = Paragraph::new(lines)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(get_border_color(app.tick)))
                .title(Span::styled(
                    " ROCKET-RS ",
                    Style::default()
                        .fg(Color::Rgb(255, 165, 0))
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .alignment(Alignment::Center);

    frame.render_widget(logo, area);
}

/// Get a gradient color based on row position and animation tick.
fn get_gradient_color(row: usize, total_rows: usize, tick: u64) -> Color {
    // Create a gradient from orange/red to cyan
    let progress = row as f64 / total_rows as f64;
    let phase = ((tick as f64 * 0.05) + progress * 2.0).sin() * 0.5 + 0.5;

    // Interpolate between colors based on position and animation
    let r = (255.0 * (1.0 - progress * 0.6) * (0.7 + 0.3 * phase)) as u8;
    let g = (100.0 + 155.0 * progress * (0.8 + 0.2 * phase)) as u8;
    let b = (50.0 + 205.0 * progress) as u8;

    Color::Rgb(r, g, b)
}

/// Get animated border color.
fn get_border_color(tick: u64) -> Color {
    let phase = (tick as f64 * 0.03).sin() * 0.5 + 0.5;
    Color::Rgb(
        (80.0 + 40.0 * phase) as u8,
        (80.0 + 40.0 * phase) as u8,
        (120.0 + 30.0 * phase) as u8,
    )
}

/// Render a compact version of the logo for smaller spaces.
pub fn render_compact_logo(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let compact = vec![
        Line::from(Span::styled(
            "ROCKET",
            Style::default()
                .fg(Color::Rgb(255, 165, 0))
                .add_modifier(Modifier::BOLD),
        )),
        Line::from(Span::styled(
            "Reinforcement Learning",
            Style::default().fg(Color::Cyan),
        )),
    ];

    let logo = Paragraph::new(compact)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(get_border_color(app.tick))),
        )
        .alignment(Alignment::Center);

    frame.render_widget(logo, area);
}

/// Rocket flame animation characters.
const FLAMES: &[&str] = &["*", "^", ".", "o", "O", "*"];

/// Render animated rocket flames (for fun).
pub fn render_rocket_animation(frame: &mut Frame<'_>, app: &App, area: Rect) {
    let flame_idx = (app.tick as usize / 2) % FLAMES.len();
    let flame_char = FLAMES[flame_idx];

    let rocket = vec![
        Line::from(Span::styled(
            "    /\\    ",
            Style::default().fg(Color::White),
        )),
        Line::from(Span::styled(
            "   /  \\   ",
            Style::default().fg(Color::White),
        )),
        Line::from(Span::styled("  |    |  ", Style::default().fg(Color::Gray))),
        Line::from(Span::styled("  |    |  ", Style::default().fg(Color::Gray))),
        Line::from(Span::styled("  | RL |  ", Style::default().fg(Color::Cyan))),
        Line::from(Span::styled("  |    |  ", Style::default().fg(Color::Gray))),
        Line::from(Span::styled(
            " /|    |\\ ",
            Style::default().fg(Color::Gray),
        )),
        Line::from(Span::styled(
            "/_|____|_\\",
            Style::default().fg(Color::Gray),
        )),
        Line::from(Span::styled(
            format!("   {}{}{}", flame_char, flame_char, flame_char),
            Style::default().fg(Color::Rgb(255, 100, 0)),
        )),
        Line::from(Span::styled(
            format!("    {}{}    ", flame_char, flame_char),
            Style::default().fg(Color::Yellow),
        )),
    ];

    let widget = Paragraph::new(rocket)
        .block(Block::default().borders(Borders::NONE))
        .alignment(Alignment::Center);

    frame.render_widget(widget, area);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logo_lines() {
        assert_eq!(ROCKET_LOGO.len(), 6);
        // Verify all lines have similar width
        let first_len = ROCKET_LOGO[0].chars().count();
        for line in ROCKET_LOGO {
            // Allow some variance for alignment
            assert!(line.chars().count() <= first_len + 5);
        }
    }

    #[test]
    fn test_gradient_color() {
        let color = get_gradient_color(0, 6, 0);
        match color {
            Color::Rgb(r, _, _) => assert!(r > 200), // First row should be reddish
            _ => panic!("Expected RGB color"),
        }
    }
}
