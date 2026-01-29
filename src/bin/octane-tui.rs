//! Octane-RS TUI Application
//!
//! Terminal user interface for monitoring and controlling
//! reinforcement learning training with Octane-RS.
//!
//! # Usage
//!
//! ```bash
//! # Run in demo mode (default)
//! cargo run --bin octane-tui
//!
//! # Run in demo mode explicitly
//! cargo run --bin octane-tui -- --demo
//!
//! # Run in benchmark mode
//! cargo run --bin octane-tui -- --benchmark
//! ```

use std::io::{self, stdout, Stdout};
use std::panic;
use std::time::Duration;

use anyhow::{Context, Result};
use crossterm::{
    event::{DisableMouseCapture, EnableMouseCapture},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};

// Import TUI modules from the library
use octane_rs::tui::{
    app::{App, AppMode, Tab},
    event::{Event, EventHandler, KeyAction},
    ui::draw,
};

/// Terminal type alias for cleaner code
type Tui = Terminal<CrosstermBackend<Stdout>>;

/// Initialize the terminal for TUI rendering
fn init_terminal() -> Result<Tui> {
    enable_raw_mode().context("Failed to enable raw mode")?;
    let mut stdout = stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)
        .context("Failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend).context("Failed to create terminal")?;
    Ok(terminal)
}

/// Restore the terminal to its original state
fn restore_terminal(terminal: &mut Tui) -> Result<()> {
    disable_raw_mode().context("Failed to disable raw mode")?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )
    .context("Failed to leave alternate screen")?;
    terminal.show_cursor().context("Failed to show cursor")?;
    Ok(())
}

/// Install custom panic hook that restores terminal before panicking
fn install_panic_hook() {
    let original_hook = panic::take_hook();
    panic::set_hook(Box::new(move |panic_info| {
        // Attempt to restore terminal
        let _ = disable_raw_mode();
        let _ = execute!(stdout(), LeaveAlternateScreen, DisableMouseCapture);

        // Call the original panic handler
        original_hook(panic_info);
    }));
}

/// Parse command line arguments
fn parse_args() -> AppMode {
    let args: Vec<String> = std::env::args().collect();

    for arg in args.iter().skip(1) {
        match arg.as_str() {
            "--demo" | "-d" => return AppMode::Demo,
            "--train" | "-t" => return AppMode::Training,
            "--benchmark" | "-b" => return AppMode::Benchmark,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            "--version" | "-v" => {
                println!("octane-tui {}", env!("CARGO_PKG_VERSION"));
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", arg);
                print_help();
                std::process::exit(1);
            }
        }
    }

    // Default to demo mode
    AppMode::Demo
}

/// Print help message
fn print_help() {
    println!(
        r#"
Octane-RS TUI - Terminal UI for Reinforcement Learning

USAGE:
    octane-tui [OPTIONS]

OPTIONS:
    -d, --demo        Run in demo mode with simulated data (default)
    -t, --train       Run in training mode (requires configuration)
    -b, --benchmark   Run and display benchmarks
    -h, --help        Print this help message
    -v, --version     Print version information

KEYBOARD SHORTCUTS:
    Tab / Shift+Tab   Navigate between tabs
    1-5               Jump to specific tab
    Space / p         Pause/Resume
    h / ?             Show help overlay
    q / Esc           Quit

For more information, visit: https://github.com/lubluniky/octane-rs
"#
    );
}

/// Main application loop
fn run_app(terminal: &mut Tui, mut app: App, events: EventHandler) -> Result<()> {
    loop {
        // Draw the UI
        terminal.draw(|frame| draw(frame, &app))?;

        // Handle events
        match events.next()? {
            Event::Tick => {
                app.on_tick();
            }
            Event::Key(key) => {
                let action = KeyAction::from(key);

                // Handle help overlay separately (closes on any key)
                if app.show_help && action != KeyAction::None {
                    app.toggle_help();
                    continue;
                }

                // Handle splash screen (skip on any key)
                if app.show_splash {
                    match action {
                        KeyAction::Quit => app.quit(),
                        KeyAction::SkipSplash | _ => app.skip_splash(),
                    }
                    continue;
                }

                // Normal key handling
                match action {
                    KeyAction::Quit => app.quit(),
                    KeyAction::NextTab => app.next_tab(),
                    KeyAction::PrevTab => app.prev_tab(),
                    KeyAction::Dashboard => app.select_tab(Tab::Dashboard),
                    KeyAction::Training => app.select_tab(Tab::Training),
                    KeyAction::Environment => app.select_tab(Tab::Environment),
                    KeyAction::Benchmarks => app.select_tab(Tab::Benchmarks),
                    KeyAction::Settings => app.select_tab(Tab::Settings),
                    KeyAction::TogglePause => app.toggle_pause(),
                    KeyAction::ToggleHelp => app.toggle_help(),
                    KeyAction::ScrollUp => app.scroll_logs_up(),
                    KeyAction::ScrollDown => app.scroll_logs_down(),
                    KeyAction::SelectNext => app.select_next(),
                    KeyAction::SelectPrev => app.select_prev(),
                    _ => {}
                }
            }
            Event::Resize(_, _) => {
                // Terminal will automatically redraw on next iteration
            }
            Event::Mouse(_) => {
                // Mouse events not handled yet
            }
        }

        // Check if we should quit
        if app.should_quit {
            return Ok(());
        }
    }
}

fn main() -> Result<()> {
    // Install panic hook for clean terminal restoration
    install_panic_hook();

    // Parse command line arguments
    let mode = parse_args();

    // Initialize terminal
    let mut terminal = init_terminal()?;

    // Create application state
    let app = App::new(mode);

    // Create event handler with 60 FPS tick rate (~16ms)
    let tick_rate = Duration::from_millis(16);
    let events = EventHandler::new(tick_rate);

    // Run the application
    let result = run_app(&mut terminal, app, events);

    // Restore terminal (even if app errored)
    restore_terminal(&mut terminal)?;

    // Return the result
    result
}
