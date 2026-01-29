//! Widget modules for the TUI.
//!
//! This module exports all custom widgets used in the Octane-RS TUI.

pub mod benchmark;
pub mod logo;
pub mod metrics;
pub mod training;

// Re-export commonly used widgets
pub use benchmark::*;
pub use logo::*;
pub use metrics::*;
pub use training::*;
