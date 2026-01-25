# Quick Start: TUI Improvements

## What Was Created

Five new modules to enhance the RocketRL TUI:

1. **`src/tui/theme.rs`** - Professional color palette
2. **`src/tui/widgets/charts.rs`** - Enhanced chart components
3. **`src/tui/log_reader.rs`** - Log file monitoring
4. **`src/tui/input.rs`** - Advanced keyboard navigation
5. **`src/tui/layout.rs`** - Responsive layout system

## Quick Integration Examples

### 1. Using the Theme

```rust
use crate::tui::theme::Theme;

// Replace hard-coded colors
let block = Block::default()
    .borders(Borders::ALL)
    .border_style(Style::default().fg(Theme::BORDER_DEFAULT))
    .style(Style::default().bg(Theme::BG_TERTIARY));

// Use semantic colors
let status_color = if training_active {
    Theme::STATUS_TRAINING  // Green
} else {
    Theme::STATUS_PAUSED    // Gray
};

// Gradient for progress
let progress_color = Theme::gradient_progress(0.75);  // 75% complete

// Performance-based coloring
let cpu_color = Theme::performance_color(cpu_usage, 0.0, 100.0);
```

### 2. Enhanced Charts

```rust
use crate::tui::widgets::charts::{
    LiveMetric, EnhancedProgress, MultiSparkline, RealtimeChart,
    MetricFormat, TrendDirection
};

// Live metric card
LiveMetric::new("Steps/sec", 15234.5)
    .format(MetricFormat::Throughput)  // Shows "15.2K"
    .trend(TrendDirection::Up)
    .color(Theme::SUCCESS)
    .render(frame, area);

// Progress bar with ETA
EnhancedProgress::new("Training", current, total)
    .with_eta(start_time)
    .gradient(true)
    .render(frame, area);

// Multi-metric sparkline
MultiSparkline::new()
    .block(Block::default().title(" Losses "))
    .add_series("Policy", &policy_data, Theme::CHART_PRIMARY)
    .add_series("Value", &value_data, Theme::CHART_SECONDARY)
    .render(frame, area);

// Real-time chart
RealtimeChart::new("Reward History")
    .add_dataset("Reward", reward_points, Theme::CHART_PRIMARY)
    .add_dataset("Average", avg_points, Theme::CHART_SECONDARY)
    .render(frame, area);
```

### 3. Log File Reading

```rust
use crate::tui::log_reader::{LogReader, LogLevel, MultiLogReader};

// Single file
let mut reader = LogReader::new("training.log")?;
reader.set_level_filter(Some(LogLevel::Info));
reader.read_all()?;

// In update loop
if reader.check_for_updates()? {
    let recent_logs = reader.recent_entries(50);
    for entry in recent_logs {
        println!("[{}] {}", entry.level.as_str(), entry.message);
    }
}

// Multiple files
let mut multi = MultiLogReader::new();
multi.add_file("training".to_string(), "train.log")?;
multi.add_file("validation".to_string(), "val.log")?;

// Auto-discover logs
let log_files = discover_log_files("./logs")?;
```

### 4. Keyboard Navigation

```rust
use crate::tui::input::{KeyBindings, InputAction};

let bindings = KeyBindings::new();

// In event loop
match event::read()? {
    Event::Key(key) => {
        match bindings.get_action(key) {
            InputAction::Quit => app.should_quit = true,
            InputAction::NextTab => app.next_tab(),
            InputAction::PrevTab => app.prev_tab(),
            InputAction::TogglePause => app.toggle_pause(),
            InputAction::ScrollUp => app.scroll_offset -= 1,
            InputAction::ScrollDown => app.scroll_offset += 1,
            InputAction::JumpTab1 => app.current_tab = 0,
            InputAction::ToggleHelp => app.show_help = !app.show_help,
            _ => {}
        }
    }
}

// Custom bindings
let mut custom_bindings = KeyBindings::new();
custom_bindings.bind(
    KeyCode::Char('x'),
    KeyModifiers::CONTROL,
    InputAction::ClearLogs
);

// Generate help text
let help = bindings.help_text();
// Returns: Vec<("q, Esc", "Quit"), ("Tab, l", "Next tab"), ...>
```

### 5. Responsive Layouts

```rust
use crate::tui::layout::{ResponsiveLayout, TerminalSize};

// Check terminal size
if !TerminalSize::is_sufficient(width, height) {
    render_warning(frame, "Terminal too small. Minimum 80x24");
    return;
}

// Create responsive layout
let layout = ResponsiveLayout::new(frame.area());
let [header, content, footer] = layout.main_layout();

// Dashboard with automatic adaptation
let dash = layout.dashboard_layout(content);

render_logo(frame, dash.header);

// Metrics (4 in small, 6 in medium/large)
for (i, area) in dash.metrics.iter().enumerate() {
    render_metric_card(frame, *area, &metrics[i]);
}

render_chart(frame, dash.main_chart);

// Optional side charts (only in medium/large)
if let Some(side_charts) = dash.side_charts {
    for (i, area) in side_charts.iter().enumerate() {
        render_side_chart(frame, *area, i);
    }
}

// Optional gauges (only in medium/large)
if let Some(gauges) = dash.gauges {
    render_cpu_gauge(frame, gauges[0]);
    render_gpu_gauge(frame, gauges[1]);
    render_mem_gauge(frame, gauges[2]);
}

render_logs(frame, dash.logs);
```

## Complete Example: Enhanced Dashboard

```rust
use crate::tui::theme::Theme;
use crate::tui::layout::ResponsiveLayout;
use crate::tui::widgets::charts::{LiveMetric, EnhancedProgress, MetricFormat};

fn render_dashboard(frame: &mut Frame, app: &App, area: Rect) {
    let layout = ResponsiveLayout::new(area);
    let dash = layout.dashboard_layout(area);

    // Logo section
    render_logo(frame, app, dash.header);

    // Metric cards
    let metrics = [
        ("Episode", app.episode as f64, MetricFormat::Integer),
        ("Timesteps", app.timesteps as f64, MetricFormat::Throughput),
        ("Reward", app.reward, MetricFormat::Float2),
        ("Steps/s", app.steps_per_sec, MetricFormat::Throughput),
    ];

    for (i, (label, value, format)) in metrics.iter().enumerate() {
        if i < dash.metrics.len() {
            LiveMetric::new(label, *value)
                .format(*format)
                .color(Theme::BRAND_PRIMARY)
                .render(frame, dash.metrics[i]);
        }
    }

    // Main chart
    use crate::tui::widgets::charts::RealtimeChart;

    let reward_data: Vec<(f64, f64)> = app.reward_history
        .iter()
        .enumerate()
        .map(|(i, r)| (i as f64, *r))
        .collect();

    RealtimeChart::new("Reward History")
        .add_dataset("Reward", reward_data, Theme::CHART_PRIMARY)
        .render(frame, dash.main_chart);

    // Side charts (if available)
    if let Some(side_charts) = dash.side_charts {
        use crate::tui::widgets::charts::MultiSparkline;

        MultiSparkline::new()
            .block(Block::default()
                .title(" Losses ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::BORDER_DEFAULT)))
            .add_series("Policy", &app.policy_loss_history, Theme::CHART_PRIMARY)
            .add_series("Value", &app.value_loss_history, Theme::CHART_SECONDARY)
            .render(frame, side_charts[0]);
    }

    // Gauges (if available)
    if let Some(gauges) = dash.gauges {
        use crate::tui::theme::WidgetColors;

        let cpu_pct = app.system_metrics.cpu_usage;
        let cpu_color = WidgetColors::gauge_color(
            cpu_pct,
            crate::tui::theme::GaugeType::Cpu
        );

        let gauge = Gauge::default()
            .block(Block::default()
                .title(format!(" CPU ({} cores) ", app.cpu_cores))
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Theme::BORDER_DEFAULT))
                .style(Style::default().bg(Theme::BG_TERTIARY)))
            .gauge_style(Style::default()
                .fg(cpu_color)
                .bg(Theme::BG_SECONDARY))
            .ratio((cpu_pct / 100.0) as f64)
            .label(format!("{:.1}%", cpu_pct));

        frame.render_widget(gauge, gauges[0]);
    }

    // Logs
    render_logs(frame, app, dash.logs);
}
```

## Adding to Existing Code

### Step 1: Module Declaration

Add to `src/tui/mod.rs`:
```rust
pub mod theme;
pub mod layout;
pub mod input;
pub mod log_reader;

// Update widgets
pub mod widgets {
    pub mod charts;  // NEW
    pub mod benchmark;
    pub mod logo;
    pub mod metrics;
    pub mod training;
}
```

### Step 2: Update App State

Add to `src/tui/app.rs` or `src/tui/mod.rs`:
```rust
use crate::tui::{
    log_reader::MultiLogReader,
    input::KeyBindings,
};

pub struct App {
    // ... existing fields ...

    // Add these
    pub log_reader: Option<MultiLogReader>,
    pub key_bindings: KeyBindings,
}

impl App {
    pub fn new() -> Self {
        Self {
            // ... existing initialization ...
            log_reader: None,
            key_bindings: KeyBindings::new(),
        }
    }

    pub fn init_logs(&mut self, path: &str) -> io::Result<()> {
        let mut reader = MultiLogReader::new();
        reader.add_file("training".to_string(), path)?;
        self.log_reader = Some(reader);
        Ok(())
    }
}
```

### Step 3: Update Main Loop

In `src/bin/rocket_tui.rs` or main TUI function:
```rust
use crate::tui::input::{InputAction};

// Initialize
let mut app = App::new();
app.init_logs("training.log").ok();

loop {
    // Render
    terminal.draw(|f| render(f, &app))?;

    // Handle events
    if event::poll(Duration::from_millis(16))? {
        if let Event::Key(key) = event::read()? {
            match app.key_bindings.get_action(key) {
                InputAction::Quit => break,
                InputAction::NextTab => app.next_tab(),
                InputAction::TogglePause => app.toggle_pause(),
                // ... handle other actions ...
                _ => {}
            }
        }
    }

    // Update logs
    if let Some(reader) = &mut app.log_reader {
        reader.check_for_updates()?;
    }

    // Update app state
    app.tick();
}
```

## Color Migration Quick Reference

| Old Color | New Theme Constant |
|-----------|-------------------|
| `Color::Rgb(100, 100, 120)` | `Theme::BORDER_DEFAULT` |
| `Color::Rgb(80, 80, 100)` | `Theme::BORDER_SUBTLE` |
| `Color::Cyan` | `Theme::BRAND_ACCENT` |
| `Color::Green` | `Theme::SUCCESS` |
| `Color::Red` | `Theme::ERROR` |
| `Color::Yellow` | `Theme::WARNING` |
| `Color::White` | `Theme::TEXT_PRIMARY` |
| `Color::Gray` | `Theme::TEXT_SECONDARY` |
| `Color::DarkGray` | `Theme::TEXT_TERTIARY` |
| `Color::Rgb(255, 165, 0)` | `Theme::BRAND_PRIMARY` |

## Testing the Improvements

### 1. Test Theme
```bash
cargo run --bin rocket-tui
# Check that all widgets use consistent colors
# Verify text is readable
```

### 2. Test Charts
```bash
# Create test data and verify:
# - LiveMetric shows correct format
# - Progress bar shows ETA
# - Charts render without panics
```

### 3. Test Log Reader
```bash
# Create a test log file
echo "[INFO] Test log entry" >> test.log

# In TUI, verify:
# - Logs are read
# - New lines appear when file is updated
# - Filters work
```

### 4. Test Navigation
```bash
# Try all key combinations
# Verify vim-style navigation works
# Check help overlay (F1 or ?)
```

### 5. Test Responsive Layout
```bash
# Resize terminal to different sizes
# Verify layout adapts smoothly
# Check minimum size warning
```

## Common Issues & Solutions

### Issue: Colors not visible
**Solution**: Ensure terminal supports 256 colors or true color

### Issue: Charts not rendering
**Solution**: Check that data vectors are not empty

### Issue: Logs not updating
**Solution**: Verify file path is correct and file exists

### Issue: Keys not working
**Solution**: Check for key binding conflicts

### Issue: Layout broken
**Solution**: Verify terminal size meets minimums (80x24)

## Next Steps

1. **Integrate theme** - Replace all hard-coded colors
2. **Add one chart type** - Start with LiveMetric or EnhancedProgress
3. **Enable log reading** - Point to an existing log file
4. **Update key handling** - Replace existing key match statements
5. **Apply responsive layout** - Start with dashboard screen

## Support

For questions or issues:
- Check `TUI_IMPROVEMENTS.md` for detailed documentation
- Review test cases in each module
- See existing widget implementations for patterns
