# RocketRL TUI Improvements Plan

## Executive Summary

This document outlines comprehensive improvements to the RocketRL Terminal User Interface (TUI), focusing on five key areas: consistent color theming, enhanced chart rendering, log file monitoring, keyboard navigation, and responsive layouts.

## Current State Analysis

### Architecture
- **Two implementations** exist that need reconciliation:
  - Primary: `src/tui/mod.rs` with 4 screens
  - Secondary: `src/tui/app.rs` + `ui.rs` with 5 tabs
- **Dependencies**: ratatui 0.29, crossterm 0.28, sysinfo 0.32
- **Issues**: Inconsistent colors, limited charts, no log reading, basic navigation

## Implementation Plan

### 1. Consistent Color Scheme ✓

**File**: `src/tui/theme.rs` (NEW)

**Professional Color Palette**:
```rust
// Brand Colors
BRAND_PRIMARY:   RGB(255, 107, 53)  // Rocket Orange
BRAND_SECONDARY: RGB(220, 47, 47)   // Fire Red
BRAND_ACCENT:    RGB(52, 152, 219)  // Electric Blue

// Backgrounds (3-tier depth)
BG_PRIMARY:   RGB(16, 16, 20)   // Darkest
BG_SECONDARY: RGB(24, 24, 30)   // Elevated
BG_TERTIARY:  RGB(32, 32, 40)   // Cards/panels

// Borders (3 levels)
BORDER_DEFAULT: RGB(60, 60, 70)
BORDER_ACTIVE:  RGB(100, 100, 120)
BORDER_SUBTLE:  RGB(40, 40, 48)

// Text (4 emphasis levels)
TEXT_PRIMARY:   RGB(240, 240, 250)  // High emphasis
TEXT_SECONDARY: RGB(180, 180, 195)  // Medium
TEXT_TERTIARY:  RGB(120, 120, 135)  // Low
TEXT_DISABLED:  RGB(80, 80, 90)     // Disabled

// Semantic Colors
SUCCESS: RGB(46, 204, 113)   // Green
WARNING: RGB(255, 193, 7)    // Yellow
ERROR:   RGB(231, 76, 60)    // Red
INFO:    RGB(52, 152, 219)   // Blue
```

**Features**:
- Gradient helpers for smooth color transitions
- Performance color mapping (high = green, low = red)
- HSL color space support for animations
- Widget-specific color palettes (CPU, GPU, Memory)

**Usage Example**:
```rust
use crate::tui::theme::Theme;

let gauge = Gauge::default()
    .gauge_style(Style::default()
        .fg(Theme::SUCCESS)
        .bg(Theme::BG_SECONDARY))
    .block(Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Theme::BORDER_DEFAULT)));
```

### 2. Enhanced Chart Rendering ✓

**File**: `src/tui/widgets/charts.rs` (NEW)

**New Components**:

#### MultiSparkline
Shows multiple metrics in a single compact widget:
```rust
MultiSparkline::new()
    .block(Block::default().title("Losses"))
    .add_series("Policy", &policy_data, Theme::CHART_PRIMARY)
    .add_series("Value", &value_data, Theme::CHART_SECONDARY)
    .add_series("Entropy", &entropy_data, Theme::CHART_TERTIARY)
    .render(frame, area);
```

#### LiveMetric
Real-time metric display with trend indicators:
```rust
LiveMetric::new("Steps/sec", 15000.0)
    .format(MetricFormat::Throughput)
    .trend(TrendDirection::Up)
    .color(Theme::SUCCESS)
    .render(frame, area);
```

**Metric Formats**:
- Integer, Float1/2/3, Scientific
- Percentage
- Throughput (K, M, B suffixes)

#### EnhancedProgress
Progress bar with gradient and ETA:
```rust
EnhancedProgress::new("Training", current_step, total_steps)
    .with_eta(start_time)
    .gradient(true)
    .render(frame, area);
```

#### RealtimeChart
Multi-dataset line chart with auto-scaling:
```rust
RealtimeChart::new("Reward History")
    .add_dataset("Mean", reward_data, Theme::CHART_PRIMARY)
    .add_dataset("Moving Avg", ma_data, Theme::CHART_SECONDARY)
    .y_bounds([min_reward, max_reward])
    .render(frame, area);
```

**Helper Functions**:
- `format_throughput()`: 1500000 → "1.5M"
- `format_number()`: 1234567 → "1,234,567"
- `format_duration()`: 3665s → "1h 1m"

### 3. Log File Reading Capability ✓

**File**: `src/tui/log_reader.rs` (NEW)

**Features**:

#### LogReader
Tail and parse log files:
```rust
let mut reader = LogReader::new("training.log")?;
reader.set_level_filter(Some(LogLevel::Info));
reader.set_search_filter(Some("episode".to_string()));

// Initial read
reader.read_all()?;

// Periodic updates (in main loop)
if reader.check_for_updates()? {
    // New lines available
    let recent = reader.recent_entries(10);
}
```

**Log Parsing**:
- Detects levels: TRACE, DEBUG, INFO, WARN, ERROR, TRAIN
- Filters by level (shows level + higher severity)
- Search filter (case-insensitive)
- Bounded buffer (default 10,000 lines)

#### MultiLogReader
Monitor multiple log files:
```rust
let mut multi_reader = MultiLogReader::new();
multi_reader.add_file("training".to_string(), "train.log")?;
multi_reader.add_file("benchmark".to_string(), "bench.log")?;

// Check all files
multi_reader.check_for_updates()?;

// Get merged entries
let all_logs = multi_reader.all_entries();
```

**Auto-discovery**:
```rust
let log_files = discover_log_files("./logs")?;
```

**File Handling**:
- Detects file rotation/truncation
- Efficient tailing (only reads new lines)
- Handles file modifications
- Automatic buffer management

### 4. Keyboard Navigation and Usability ✓

**File**: `src/tui/input.rs` (NEW)

**Enhanced Key Bindings**:

| Category | Action | Keys |
|----------|--------|------|
| **Navigation** | Quit | `q`, `Esc`, `Ctrl+C` |
| | Next Tab | `Tab`, `→`, `l` |
| | Prev Tab | `Shift+Tab`, `←`, `h` |
| | Jump to Tab | `1-5` |
| **Scrolling** | Up/Down | `j`/`k`, `↑`/`↓` |
| | Page Up/Down | `Ctrl+u`/`Ctrl+d`, `PgUp`/`PgDn` |
| | Top/Bottom | `g`/`G`, `Home`/`End` |
| **Training** | Pause/Resume | `Space` |
| | Reset | `r` |
| | Save Checkpoint | `Ctrl+S` |
| | Load Checkpoint | `Ctrl+O` |
| **View** | Toggle Help | `?`, `F1` |
| | Refresh | `Ctrl+R`, `F5` |
| | Toggle Fullscreen | `f` |
| | Cycle Theme | `Ctrl+T` |
| **Modes** | Demo | `d` |
| | Training | `t` |
| | Benchmark | `b` |
| | Cycle | `m` |
| **Logs** | Clear | `c` |
| | Export | `Ctrl+E` |
| | Search | `/` |
| | Filter | `Ctrl+F` |

**Usage**:
```rust
let bindings = KeyBindings::new();

match event {
    Event::Key(key) => {
        let action = bindings.get_action(key);
        match action {
            InputAction::Quit => app.should_quit = true,
            InputAction::NextTab => app.next_tab(),
            InputAction::TogglePause => app.toggle_pause(),
            // ...
        }
    }
}
```

**Custom Bindings**:
```rust
let mut bindings = KeyBindings::new();
bindings.bind(KeyCode::Char('x'), KeyModifiers::CONTROL, InputAction::ClearLogs);
```

**Help System**:
```rust
let help_text = bindings.help_text();
// Returns: Vec<(String, String)> of (key combo, description)
```

### 5. Responsive Layout System ✓

**File**: `src/tui/layout.rs` (NEW)

**Terminal Size Categories**:
- **Small**: < 100x30 (minimal, stacked layout)
- **Medium**: 100x30 to 160x40 (hybrid layout)
- **Large**: > 160x40 (full feature layout)

**Adaptive Layouts**:

#### Dashboard Layout

**Small Terminal** (80x24):
```
┌─────────────────────┐
│ Logo & Status       │
├─────────────────────┤
│ Metric1   Metric2   │
│ Metric3   Metric4   │
├─────────────────────┤
│                     │
│   Main Chart        │
│                     │
├─────────────────────┤
│ Logs                │
└─────────────────────┘
```

**Medium Terminal** (120x35):
```
┌───────────┬─────────────────────────┐
│           │ Metrics (2x3 grid)      │
│   Logo    ├─────────┬─────────┬─────┤
│           │ M1  M2  │  M3  M4 │ M5  │
├───────────┴─────────┴─────────┴─────┤
│ Main Chart         │  Side Charts   │
│                    │                 │
├────────────────────┼─────────────────┤
│ CPU  GPU  Memory   │     Logs        │
└────────────────────┴─────────────────┘
```

**Large Terminal** (180x50):
```
┌──────┬──────────────────────────────────────┐
│      │ M1   M2   M3   M4   M5   M6         │
│ Logo ├─────────────────────┬────────────────┤
│      │                     │  Chart 1       │
├──────┤   Main Chart        ├────────────────┤
│      │                     │  Chart 2       │
│      │                     ├────────────────┤
│      │                     │  Chart 3       │
├──────┴─────────────────────┴────────────────┤
│ CPU  GPU  Memory           │     Logs       │
└────────────────────────────┴────────────────┘
```

**Usage**:
```rust
let layout = ResponsiveLayout::new(frame.area());

match app.current_screen {
    Screen::Dashboard => {
        let dash_layout = layout.dashboard_layout(content_area);

        render_logo(frame, dash_layout.header);

        for (i, area) in dash_layout.metrics.iter().enumerate() {
            render_metric(frame, *area, &metrics[i]);
        }

        render_chart(frame, dash_layout.main_chart);

        if let Some(side_charts) = dash_layout.side_charts {
            // Only render if terminal is large enough
            for area in side_charts {
                render_side_chart(frame, area);
            }
        }
    }
}
```

**Helper Functions**:
```rust
// Create centered popup
let popup = centered_rect(60, 40, frame.area());

// Check minimum size
if !TerminalSize::is_sufficient(width, height) {
    render_size_warning(frame);
    return;
}
```

## Integration Guide

### Step 1: Add Module Declarations

**In `src/tui/mod.rs`**, add:
```rust
pub mod theme;
pub mod layout;
pub mod input;
pub mod log_reader;

// Update widgets module
pub mod widgets {
    pub mod charts;
    pub mod benchmark;
    pub mod logo;
    pub mod metrics;
    pub mod training;

    pub use charts::*;
}
```

### Step 2: Update Existing Screens

**Replace hard-coded colors**:
```rust
// Before
Style::default().fg(Color::Rgb(100, 100, 120))

// After
use crate::tui::theme::Theme;
Style::default().fg(Theme::BORDER_DEFAULT)
```

**Use responsive layouts**:
```rust
// Before
let chunks = Layout::default()
    .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
    .split(area);

// After
use crate::tui::layout::ResponsiveLayout;
let layout = ResponsiveLayout::new(area);
let dash = layout.dashboard_layout(area);
```

### Step 3: Add Log Reading to App State

**In `App` struct**:
```rust
use crate::tui::log_reader::{MultiLogReader, LogEntry};

pub struct App {
    // ... existing fields ...
    pub log_reader: Option<MultiLogReader>,
    pub log_entries: Vec<LogEntry>,
}

impl App {
    pub fn init_log_reader(&mut self, log_path: &str) -> io::Result<()> {
        let mut reader = MultiLogReader::new();
        reader.add_file("training".to_string(), log_path)?;
        self.log_reader = Some(reader);
        Ok(())
    }

    pub fn update_logs(&mut self) -> io::Result<()> {
        if let Some(reader) = &mut self.log_reader {
            if reader.check_for_updates()? {
                self.log_entries = reader.all_entries()
                    .into_iter()
                    .map(|(_, entry)| entry.clone())
                    .collect();
            }
        }
        Ok(())
    }
}
```

### Step 4: Replace Key Handling

**In main event loop**:
```rust
use crate::tui::input::{KeyBindings, InputAction};

let bindings = KeyBindings::new();

loop {
    // ... render ...

    if let Event::Key(key) = event::read()? {
        match bindings.get_action(key) {
            InputAction::Quit => break,
            InputAction::NextTab => app.next_tab(),
            InputAction::TogglePause => app.toggle_pause(),
            InputAction::ToggleHelp => app.show_help = !app.show_help,
            InputAction::Refresh => {
                app.refresh_system_metrics();
                app.update_logs()?;
            }
            InputAction::ClearLogs => app.log_entries.clear(),
            // ... handle other actions ...
            InputAction::None => {}
        }
    }
}
```

### Step 5: Use Enhanced Charts

**Replace basic sparklines**:
```rust
use crate::tui::widgets::charts::MultiSparkline;

// Before
let sparkline = Sparkline::default()
    .data(&data)
    .style(Style::default().fg(Color::Green));

// After
MultiSparkline::new()
    .block(Block::default().title(" Training Metrics "))
    .add_series("Policy Loss", &policy_data, Theme::CHART_PRIMARY)
    .add_series("Value Loss", &value_data, Theme::CHART_SECONDARY)
    .add_series("Entropy", &entropy_data, Theme::CHART_TERTIARY)
    .render(frame, area);
```

**Add progress bars**:
```rust
use crate::tui::widgets::charts::EnhancedProgress;

EnhancedProgress::new("Training Progress", current_steps, total_steps)
    .with_eta(training_start_time)
    .render(frame, progress_area);
```

## Testing Checklist

- [ ] Theme applies consistently across all screens
- [ ] Charts render correctly at different terminal sizes
- [ ] Log reader handles file rotation
- [ ] All key bindings work as documented
- [ ] Layouts adapt to small/medium/large terminals
- [ ] No panics on resize
- [ ] Help overlay displays correctly
- [ ] Colors are readable in dark terminals
- [ ] Performance is smooth at 60 FPS

## Performance Considerations

1. **Log Reading**: Limit buffer to 10,000 lines (configurable)
2. **Chart Data**: Keep max 1,000 points per series
3. **Render Rate**: 60 FPS for smooth animations
4. **File Watching**: Check logs every 100ms (configurable)

## Future Enhancements

1. **Mouse Support**: Click to select tabs, scroll logs
2. **Custom Themes**: User-configurable color schemes
3. **Export**: Save charts as images or data files
4. **Notifications**: Toast messages for events
5. **Split Panes**: Multiple views simultaneously
6. **Command Palette**: Vim-style `:` command mode
7. **Plugin System**: Custom widgets and screens
8. **Remote Monitoring**: Connect to remote training processes

## Dependencies

```toml
[dependencies]
ratatui = "0.29"
crossterm = "0.28"
sysinfo = "0.32"

[dev-dependencies]
tempfile = "3.0"  # For log reader tests
```

## Migration Path

### Phase 1: Foundation (Week 1)
- [ ] Add theme module
- [ ] Update all existing widgets to use theme
- [ ] Test color consistency

### Phase 2: Enhanced Widgets (Week 2)
- [ ] Add charts module
- [ ] Integrate enhanced charts into screens
- [ ] Add progress bars

### Phase 3: Log Reading (Week 3)
- [ ] Add log_reader module
- [ ] Integrate into app state
- [ ] Add log viewing screen

### Phase 4: Input & Layout (Week 4)
- [ ] Add input module
- [ ] Replace existing key handling
- [ ] Add responsive layouts
- [ ] Test on different terminal sizes

### Phase 5: Polish & Testing (Week 5)
- [ ] Add help overlay
- [ ] Performance optimization
- [ ] Documentation
- [ ] User testing

## Conclusion

These improvements will transform the RocketRL TUI into a professional, production-grade monitoring interface with:

- **Consistent visual design** through unified theming
- **Rich data visualization** with advanced charts
- **Real-time monitoring** via log file reading
- **Intuitive navigation** with vim-style keybindings
- **Adaptive interface** that works on any terminal size

The modular design allows incremental adoption - you can integrate features one at a time without breaking existing functionality.
