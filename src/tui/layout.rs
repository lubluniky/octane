//! Responsive layout system for the TUI.
//!
//! Provides adaptive layouts that adjust to terminal size and
//! optimize space usage.

use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
};

/// Minimum terminal dimensions for proper rendering
pub const MIN_WIDTH: u16 = 80;
pub const MIN_HEIGHT: u16 = 24;

/// Terminal size category for responsive layouts
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalSize {
    /// Small terminal (< 100x30)
    Small,
    /// Medium terminal (100x30 to 160x40)
    Medium,
    /// Large terminal (> 160x40)
    Large,
}

impl TerminalSize {
    /// Determine size category from dimensions
    pub fn from_dimensions(width: u16, height: u16) -> Self {
        if width < 100 || height < 30 {
            TerminalSize::Small
        } else if width < 160 || height < 40 {
            TerminalSize::Medium
        } else {
            TerminalSize::Large
        }
    }

    /// Check if terminal meets minimum requirements
    pub fn is_sufficient(width: u16, height: u16) -> bool {
        width >= MIN_WIDTH && height >= MIN_HEIGHT
    }
}

/// Responsive layout builder
pub struct ResponsiveLayout {
    size: TerminalSize,
    area: Rect,
}

impl ResponsiveLayout {
    pub fn new(area: Rect) -> Self {
        let size = TerminalSize::from_dimensions(area.width, area.height);
        Self { size, area }
    }

    /// Get the main app layout (header, content, footer)
    pub fn main_layout(&self) -> [Rect; 3] {
        let header_height = match self.size {
            TerminalSize::Small => 3,
            TerminalSize::Medium | TerminalSize::Large => 3,
        };

        let footer_height = match self.size {
            TerminalSize::Small => 2,
            TerminalSize::Medium | TerminalSize::Large => 3,
        };

        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(header_height),
                Constraint::Min(10),
                Constraint::Length(footer_height),
            ])
            .split(self.area);

        [chunks[0], chunks[1], chunks[2]]
    }

    /// Get dashboard layout
    pub fn dashboard_layout(&self, area: Rect) -> DashboardLayout {
        match self.size {
            TerminalSize::Small => self.dashboard_small(area),
            TerminalSize::Medium => self.dashboard_medium(area),
            TerminalSize::Large => self.dashboard_large(area),
        }
    }

    /// Small terminal dashboard (stacked vertically)
    fn dashboard_small(&self, area: Rect) -> DashboardLayout {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(8),  // Logo & status
                Constraint::Length(6),  // Metrics cards (stacked)
                Constraint::Min(10),    // Main chart
                Constraint::Length(6),  // Logs
            ])
            .split(area);

        // Split metrics into 2 rows
        let metrics = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[1]);

        let top_metrics = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(metrics[0]);

        let bottom_metrics = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(50),
                Constraint::Percentage(50),
            ])
            .split(metrics[1]);

        DashboardLayout {
            header: chunks[0],
            metrics: vec![
                top_metrics[0],
                top_metrics[1],
                bottom_metrics[0],
                bottom_metrics[1],
            ],
            main_chart: chunks[2],
            side_charts: None,
            logs: chunks[3],
            gauges: None,
        }
    }

    /// Medium terminal dashboard (hybrid layout)
    fn dashboard_medium(&self, area: Rect) -> DashboardLayout {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(10), // Logo & metrics
                Constraint::Min(15),    // Charts area
                Constraint::Length(8),  // Gauges & logs
            ])
            .split(area);

        // Top: Logo left, metrics right
        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(40), Constraint::Percentage(60)])
            .split(chunks[0]);

        // Metrics in 2x3 grid
        let metrics_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(top[1]);

        let metrics_top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(metrics_rows[0]);

        let metrics_bottom = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(metrics_rows[1]);

        // Middle: Main chart left, side charts right
        let middle = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
            .split(chunks[1]);

        let side_charts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(middle[1]);

        // Bottom: Gauges left, logs right
        let bottom = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(chunks[2]);

        let gauges = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(bottom[0]);

        DashboardLayout {
            header: top[0],
            metrics: vec![
                metrics_top[0],
                metrics_top[1],
                metrics_top[2],
                metrics_bottom[0],
                metrics_bottom[1],
                metrics_bottom[2],
            ],
            main_chart: middle[0],
            side_charts: Some(vec![side_charts[0], side_charts[1]]),
            logs: bottom[1],
            gauges: Some(vec![gauges[0], gauges[1], gauges[2]]),
        }
    }

    /// Large terminal dashboard (full layout)
    fn dashboard_large(&self, area: Rect) -> DashboardLayout {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(12), // Logo & metrics
                Constraint::Min(20),    // Charts area
                Constraint::Length(10), // Gauges & logs
            ])
            .split(area);

        // Top: Logo left, metrics right (single row)
        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(30), Constraint::Percentage(70)])
            .split(chunks[0]);

        let metrics = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(16),
                Constraint::Percentage(17),
                Constraint::Percentage(17),
                Constraint::Percentage(17),
                Constraint::Percentage(17),
                Constraint::Percentage(16),
            ])
            .split(top[1]);

        // Middle: Main chart (60%), side charts (40%)
        let middle = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(chunks[1]);

        let side_charts = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(middle[1]);

        // Bottom: Gauges (60%), logs (40%)
        let bottom = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(chunks[2]);

        let gauges = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(bottom[0]);

        DashboardLayout {
            header: top[0],
            metrics: vec![
                metrics[0],
                metrics[1],
                metrics[2],
                metrics[3],
                metrics[4],
                metrics[5],
            ],
            main_chart: middle[0],
            side_charts: Some(vec![side_charts[0], side_charts[1], side_charts[2]]),
            logs: bottom[1],
            gauges: Some(vec![gauges[0], gauges[1], gauges[2]]),
        }
    }

    /// Get training screen layout
    pub fn training_layout(&self, area: Rect) -> TrainingLayout {
        match self.size {
            TerminalSize::Small => self.training_small(area),
            TerminalSize::Medium => self.training_medium(area),
            TerminalSize::Large => self.training_large(area),
        }
    }

    fn training_small(&self, area: Rect) -> TrainingLayout {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(4),  // Progress bar
                Constraint::Min(12),    // Main chart
                Constraint::Length(8),  // Metrics
            ])
            .split(area);

        let metrics = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(chunks[2]);

        TrainingLayout {
            progress: Some(chunks[0]),
            main_chart: chunks[1],
            loss_charts: vec![metrics[0], metrics[1]],
            stats: None,
        }
    }

    fn training_medium(&self, area: Rect) -> TrainingLayout {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(4),  // Progress bar
                Constraint::Min(15),    // Charts area
                Constraint::Length(8),  // Stats
            ])
            .split(area);

        let middle = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(65), Constraint::Percentage(35)])
            .split(chunks[1]);

        let side = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(middle[1]);

        TrainingLayout {
            progress: Some(chunks[0]),
            main_chart: middle[0],
            loss_charts: vec![side[0], side[1]],
            stats: Some(chunks[2]),
        }
    }

    fn training_large(&self, area: Rect) -> TrainingLayout {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([
                Constraint::Length(4),  // Progress bar
                Constraint::Min(20),    // Charts area
                Constraint::Length(10), // Stats
            ])
            .split(area);

        let middle = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(chunks[1]);

        let side = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Percentage(33),
                Constraint::Percentage(34),
                Constraint::Percentage(33),
            ])
            .split(middle[1]);

        TrainingLayout {
            progress: Some(chunks[0]),
            main_chart: middle[0],
            loss_charts: vec![side[0], side[1], side[2]],
            stats: Some(chunks[2]),
        }
    }

    /// Get benchmark screen layout
    pub fn benchmark_layout(&self, area: Rect) -> BenchmarkLayout {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .margin(1)
            .constraints([Constraint::Min(15), Constraint::Length(10)])
            .split(area);

        let top = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(chunks[0]);

        BenchmarkLayout {
            comparison_table: top[0],
            speedup_chart: top[1],
            detailed_metrics: chunks[1],
        }
    }
}

/// Dashboard layout structure
pub struct DashboardLayout {
    pub header: Rect,
    pub metrics: Vec<Rect>,
    pub main_chart: Rect,
    pub side_charts: Option<Vec<Rect>>,
    pub logs: Rect,
    pub gauges: Option<Vec<Rect>>,
}

/// Training screen layout structure
pub struct TrainingLayout {
    pub progress: Option<Rect>,
    pub main_chart: Rect,
    pub loss_charts: Vec<Rect>,
    pub stats: Option<Rect>,
}

/// Benchmark screen layout structure
pub struct BenchmarkLayout {
    pub comparison_table: Rect,
    pub speedup_chart: Rect,
    pub detailed_metrics: Rect,
}

/// Create a centered rect within the given area
pub fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let popup_layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);

    Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(popup_layout[1])[1]
}

/// Create a popup rect that overlays the given area
pub fn popup_rect(area: Rect, percent_x: u16, percent_y: u16) -> Rect {
    centered_rect(percent_x.min(90), percent_y.min(90), area)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_terminal_size_categorization() {
        assert_eq!(TerminalSize::from_dimensions(80, 24), TerminalSize::Small);
        assert_eq!(TerminalSize::from_dimensions(120, 35), TerminalSize::Medium);
        assert_eq!(TerminalSize::from_dimensions(200, 50), TerminalSize::Large);
    }

    #[test]
    fn test_minimum_size() {
        assert!(TerminalSize::is_sufficient(80, 24));
        assert!(!TerminalSize::is_sufficient(70, 20));
    }

    #[test]
    fn test_main_layout() {
        let area = Rect::new(0, 0, 100, 30);
        let layout = ResponsiveLayout::new(area);
        let [header, content, footer] = layout.main_layout();

        assert!(header.height >= 2);
        assert!(footer.height >= 2);
        assert!(content.height >= 10);
    }

    #[test]
    fn test_dashboard_layouts() {
        // Small
        let small = ResponsiveLayout::new(Rect::new(0, 0, 90, 28));
        let dash = small.dashboard_layout(small.area);
        assert_eq!(dash.metrics.len(), 4);

        // Medium
        let medium = ResponsiveLayout::new(Rect::new(0, 0, 120, 35));
        let dash = medium.dashboard_layout(medium.area);
        assert_eq!(dash.metrics.len(), 6);
        assert!(dash.gauges.is_some());

        // Large
        let large = ResponsiveLayout::new(Rect::new(0, 0, 180, 50));
        let dash = large.dashboard_layout(large.area);
        assert_eq!(dash.metrics.len(), 6);
        assert!(dash.side_charts.is_some());
    }
}
