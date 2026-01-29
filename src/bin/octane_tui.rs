//! Octane Terminal User Interface
//!
//! A beautiful TUI for visualizing training, benchmarks, and performance.
//!
//! Run with: cargo run --bin octane-tui

use octane_rs::tui::run_tui;
use std::process;

fn main() {
    // Set up panic hook to restore terminal on panic
    let default_panic = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // Attempt to restore terminal
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(
            std::io::stdout(),
            crossterm::terminal::LeaveAlternateScreen,
            crossterm::event::DisableMouseCapture
        );
        // Call default panic handler
        default_panic(info);
    }));

    // Print startup message
    println!("\n  OCTANE-RS TUI");
    println!("  High-Performance Reinforcement Learning\n");
    println!("  Starting terminal interface...\n");

    // Run the TUI
    if let Err(e) = run_tui() {
        eprintln!("Error running TUI: {}", e);
        process::exit(1);
    }

    println!("\n  Thanks for using Octane!\n");
}
