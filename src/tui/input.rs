//! Enhanced input handling and keyboard navigation for the TUI.
//!
//! Provides comprehensive keyboard shortcuts, vim-style navigation,
//! and contextual help.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use std::collections::HashMap;

/// Input action that can be performed in the TUI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InputAction {
    // === Navigation ===
    /// Quit the application
    Quit,
    /// Move to next tab
    NextTab,
    /// Move to previous tab
    PrevTab,
    /// Jump to tab 1 (Dashboard)
    JumpTab1,
    /// Jump to tab 2 (Training)
    JumpTab2,
    /// Jump to tab 3 (Environment/Benchmark)
    JumpTab3,
    /// Jump to tab 4 (Benchmarks/About)
    JumpTab4,
    /// Jump to tab 5 (Settings)
    JumpTab5,

    // === Scrolling ===
    /// Scroll up one line
    ScrollUp,
    /// Scroll down one line
    ScrollDown,
    /// Scroll left
    ScrollLeft,
    /// Scroll right
    ScrollRight,
    /// Page up
    PageUp,
    /// Page down
    PageDown,
    /// Scroll to top
    ScrollTop,
    /// Scroll to bottom
    ScrollBottom,

    // === List Navigation ===
    /// Select previous item
    SelectPrev,
    /// Select next item
    SelectNext,
    /// Confirm selection
    Select,

    // === Training Controls ===
    /// Toggle training pause/resume
    TogglePause,
    /// Reset training
    ResetTraining,
    /// Save checkpoint
    SaveCheckpoint,
    /// Load checkpoint
    LoadCheckpoint,

    // === View Controls ===
    /// Toggle help overlay
    ToggleHelp,
    /// Toggle fullscreen mode
    ToggleFullscreen,
    /// Refresh data
    Refresh,
    /// Cycle color theme
    CycleTheme,
    /// Toggle dark/light mode
    ToggleDarkMode,

    // === Mode Switching ===
    /// Switch to demo mode
    DemoMode,
    /// Switch to training mode
    TrainingMode,
    /// Switch to benchmark mode
    BenchmarkMode,
    /// Cycle through modes
    CycleMode,

    // === Log Controls ===
    /// Clear logs
    ClearLogs,
    /// Export logs
    ExportLogs,
    /// Filter logs
    FilterLogs,
    /// Search logs
    SearchLogs,

    // === Other ===
    /// No action
    None,
    /// Cancel current operation
    Cancel,
    /// Enter command mode
    CommandMode,
}

/// Key binding configuration
pub struct KeyBindings {
    /// Map of key combinations to actions
    bindings: HashMap<(KeyCode, KeyModifiers), InputAction>,
}

impl Default for KeyBindings {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyBindings {
    /// Create default key bindings
    pub fn new() -> Self {
        let mut bindings = HashMap::new();

        // === Quit ===
        bindings.insert((KeyCode::Char('q'), KeyModifiers::NONE), InputAction::Quit);
        bindings.insert((KeyCode::Esc, KeyModifiers::NONE), InputAction::Quit);
        bindings.insert(
            (KeyCode::Char('c'), KeyModifiers::CONTROL),
            InputAction::Quit,
        );

        // === Tab Navigation ===
        bindings.insert((KeyCode::Tab, KeyModifiers::NONE), InputAction::NextTab);
        bindings.insert((KeyCode::BackTab, KeyModifiers::SHIFT), InputAction::PrevTab);
        bindings.insert((KeyCode::Right, KeyModifiers::NONE), InputAction::NextTab);
        bindings.insert((KeyCode::Left, KeyModifiers::NONE), InputAction::PrevTab);

        // Vim-style tab navigation
        bindings.insert((KeyCode::Char('l'), KeyModifiers::NONE), InputAction::NextTab);
        bindings.insert((KeyCode::Char('h'), KeyModifiers::NONE), InputAction::PrevTab);

        // Direct tab jumps
        bindings.insert((KeyCode::Char('1'), KeyModifiers::NONE), InputAction::JumpTab1);
        bindings.insert((KeyCode::Char('2'), KeyModifiers::NONE), InputAction::JumpTab2);
        bindings.insert((KeyCode::Char('3'), KeyModifiers::NONE), InputAction::JumpTab3);
        bindings.insert((KeyCode::Char('4'), KeyModifiers::NONE), InputAction::JumpTab4);
        bindings.insert((KeyCode::Char('5'), KeyModifiers::NONE), InputAction::JumpTab5);

        // === Scrolling (Vim-style) ===
        bindings.insert((KeyCode::Char('j'), KeyModifiers::NONE), InputAction::ScrollDown);
        bindings.insert((KeyCode::Char('k'), KeyModifiers::NONE), InputAction::ScrollUp);
        bindings.insert((KeyCode::Down, KeyModifiers::NONE), InputAction::ScrollDown);
        bindings.insert((KeyCode::Up, KeyModifiers::NONE), InputAction::ScrollUp);

        // Page navigation
        bindings.insert(
            (KeyCode::Char('d'), KeyModifiers::CONTROL),
            InputAction::PageDown,
        );
        bindings.insert(
            (KeyCode::Char('u'), KeyModifiers::CONTROL),
            InputAction::PageUp,
        );
        bindings.insert((KeyCode::PageDown, KeyModifiers::NONE), InputAction::PageDown);
        bindings.insert((KeyCode::PageUp, KeyModifiers::NONE), InputAction::PageUp);

        // Top/Bottom
        bindings.insert((KeyCode::Char('g'), KeyModifiers::NONE), InputAction::ScrollTop);
        bindings.insert(
            (KeyCode::Char('G'), KeyModifiers::SHIFT),
            InputAction::ScrollBottom,
        );
        bindings.insert((KeyCode::Home, KeyModifiers::NONE), InputAction::ScrollTop);
        bindings.insert((KeyCode::End, KeyModifiers::NONE), InputAction::ScrollBottom);

        // === List Navigation ===
        bindings.insert((KeyCode::Enter, KeyModifiers::NONE), InputAction::Select);
        bindings.insert(
            (KeyCode::Char('n'), KeyModifiers::NONE),
            InputAction::SelectNext,
        );
        bindings.insert(
            (KeyCode::Char('p'), KeyModifiers::NONE),
            InputAction::SelectPrev,
        );

        // === Training Controls ===
        bindings.insert(
            (KeyCode::Char(' '), KeyModifiers::NONE),
            InputAction::TogglePause,
        );
        bindings.insert(
            (KeyCode::Char('r'), KeyModifiers::NONE),
            InputAction::ResetTraining,
        );
        bindings.insert(
            (KeyCode::Char('s'), KeyModifiers::CONTROL),
            InputAction::SaveCheckpoint,
        );
        bindings.insert(
            (KeyCode::Char('o'), KeyModifiers::CONTROL),
            InputAction::LoadCheckpoint,
        );

        // === View Controls ===
        bindings.insert(
            (KeyCode::Char('?'), KeyModifiers::SHIFT),
            InputAction::ToggleHelp,
        );
        bindings.insert(
            (KeyCode::F(1), KeyModifiers::NONE),
            InputAction::ToggleHelp,
        );
        bindings.insert(
            (KeyCode::Char('f'), KeyModifiers::NONE),
            InputAction::ToggleFullscreen,
        );
        bindings.insert(
            (KeyCode::Char('r'), KeyModifiers::CONTROL),
            InputAction::Refresh,
        );
        bindings.insert((KeyCode::F(5), KeyModifiers::NONE), InputAction::Refresh);
        bindings.insert(
            (KeyCode::Char('t'), KeyModifiers::CONTROL),
            InputAction::CycleTheme,
        );

        // === Mode Switching ===
        bindings.insert((KeyCode::Char('d'), KeyModifiers::NONE), InputAction::DemoMode);
        bindings.insert((KeyCode::Char('t'), KeyModifiers::NONE), InputAction::TrainingMode);
        bindings.insert((KeyCode::Char('b'), KeyModifiers::NONE), InputAction::BenchmarkMode);
        bindings.insert((KeyCode::Char('m'), KeyModifiers::NONE), InputAction::CycleMode);

        // === Log Controls ===
        bindings.insert(
            (KeyCode::Char('c'), KeyModifiers::NONE),
            InputAction::ClearLogs,
        );
        bindings.insert(
            (KeyCode::Char('e'), KeyModifiers::CONTROL),
            InputAction::ExportLogs,
        );
        bindings.insert(
            (KeyCode::Char('f'), KeyModifiers::CONTROL),
            InputAction::FilterLogs,
        );
        bindings.insert(
            (KeyCode::Char('/'), KeyModifiers::NONE),
            InputAction::SearchLogs,
        );

        // === Other ===
        bindings.insert(
            (KeyCode::Char(':'), KeyModifiers::NONE),
            InputAction::CommandMode,
        );

        Self { bindings }
    }

    /// Get action for key event
    pub fn get_action(&self, key: KeyEvent) -> InputAction {
        self.bindings
            .get(&(key.code, key.modifiers))
            .copied()
            .unwrap_or(InputAction::None)
    }

    /// Add or override a binding
    pub fn bind(&mut self, key: KeyCode, modifiers: KeyModifiers, action: InputAction) {
        self.bindings.insert((key, modifiers), action);
    }

    /// Remove a binding
    pub fn unbind(&mut self, key: KeyCode, modifiers: KeyModifiers) {
        self.bindings.remove(&(key, modifiers));
    }

    /// Get all bindings for an action
    pub fn bindings_for_action(&self, action: InputAction) -> Vec<(KeyCode, KeyModifiers)> {
        self.bindings
            .iter()
            .filter(|(_, a)| **a == action)
            .map(|(k, _)| *k)
            .collect()
    }

    /// Get help text for all bindings
    pub fn help_text(&self) -> Vec<(String, String)> {
        let mut help = Vec::new();

        // Group by category
        help.push(("Navigation".to_string(), "".to_string()));
        self.add_help_for_action(&mut help, InputAction::Quit, "Quit application");
        self.add_help_for_action(&mut help, InputAction::NextTab, "Next tab");
        self.add_help_for_action(&mut help, InputAction::PrevTab, "Previous tab");
        self.add_help_for_action(&mut help, InputAction::JumpTab1, "Jump to Dashboard");

        help.push(("".to_string(), "".to_string())); // Separator

        help.push(("Scrolling".to_string(), "".to_string()));
        self.add_help_for_action(&mut help, InputAction::ScrollUp, "Scroll up");
        self.add_help_for_action(&mut help, InputAction::ScrollDown, "Scroll down");
        self.add_help_for_action(&mut help, InputAction::PageUp, "Page up");
        self.add_help_for_action(&mut help, InputAction::PageDown, "Page down");
        self.add_help_for_action(&mut help, InputAction::ScrollTop, "Scroll to top");
        self.add_help_for_action(&mut help, InputAction::ScrollBottom, "Scroll to bottom");

        help.push(("".to_string(), "".to_string()));

        help.push(("Training".to_string(), "".to_string()));
        self.add_help_for_action(&mut help, InputAction::TogglePause, "Pause/Resume");
        self.add_help_for_action(&mut help, InputAction::ResetTraining, "Reset training");
        self.add_help_for_action(&mut help, InputAction::SaveCheckpoint, "Save checkpoint");

        help.push(("".to_string(), "".to_string()));

        help.push(("View".to_string(), "".to_string()));
        self.add_help_for_action(&mut help, InputAction::ToggleHelp, "Toggle help");
        self.add_help_for_action(&mut help, InputAction::Refresh, "Refresh data");
        self.add_help_for_action(&mut help, InputAction::CycleTheme, "Cycle theme");

        help.push(("".to_string(), "".to_string()));

        help.push(("Modes".to_string(), "".to_string()));
        self.add_help_for_action(&mut help, InputAction::DemoMode, "Demo mode");
        self.add_help_for_action(&mut help, InputAction::TrainingMode, "Training mode");
        self.add_help_for_action(&mut help, InputAction::BenchmarkMode, "Benchmark mode");

        help
    }

    fn add_help_for_action(&self, help: &mut Vec<(String, String)>, action: InputAction, desc: &str) {
        let bindings = self.bindings_for_action(action);
        if !bindings.is_empty() {
            let key_str = Self::format_key_bindings(&bindings);
            help.push((key_str, desc.to_string()));
        }
    }

    fn format_key_bindings(bindings: &[(KeyCode, KeyModifiers)]) -> String {
        bindings
            .iter()
            .take(3) // Show max 3 bindings
            .map(|(code, mods)| Self::format_key(*code, *mods))
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn format_key(code: KeyCode, mods: KeyModifiers) -> String {
        let mut result = String::new();

        if mods.contains(KeyModifiers::CONTROL) {
            result.push_str("Ctrl+");
        }
        if mods.contains(KeyModifiers::SHIFT) {
            result.push_str("Shift+");
        }
        if mods.contains(KeyModifiers::ALT) {
            result.push_str("Alt+");
        }

        match code {
            KeyCode::Char(c) => result.push(c),
            KeyCode::F(n) => result.push_str(&format!("F{}", n)),
            KeyCode::Enter => result.push_str("Enter"),
            KeyCode::Tab => result.push_str("Tab"),
            KeyCode::BackTab => result.push_str("BackTab"),
            KeyCode::Esc => result.push_str("Esc"),
            KeyCode::Up => result.push_str("↑"),
            KeyCode::Down => result.push_str("↓"),
            KeyCode::Left => result.push_str("←"),
            KeyCode::Right => result.push_str("→"),
            KeyCode::PageUp => result.push_str("PgUp"),
            KeyCode::PageDown => result.push_str("PgDn"),
            KeyCode::Home => result.push_str("Home"),
            KeyCode::End => result.push_str("End"),
            _ => result.push_str(&format!("{:?}", code)),
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_bindings() {
        let bindings = KeyBindings::new();

        // Test quit binding
        let quit_key = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(bindings.get_action(quit_key), InputAction::Quit);

        // Test tab navigation
        let next_tab = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(bindings.get_action(next_tab), InputAction::NextTab);
    }

    #[test]
    fn test_custom_binding() {
        let mut bindings = KeyBindings::new();

        // Add custom binding
        bindings.bind(
            KeyCode::Char('x'),
            KeyModifiers::CONTROL,
            InputAction::ClearLogs,
        );

        let key = KeyEvent::new(KeyCode::Char('x'), KeyModifiers::CONTROL);
        assert_eq!(bindings.get_action(key), InputAction::ClearLogs);
    }

    #[test]
    fn test_bindings_for_action() {
        let bindings = KeyBindings::new();
        let quit_bindings = bindings.bindings_for_action(InputAction::Quit);

        // Should have multiple quit bindings
        assert!(quit_bindings.len() >= 2);
    }

    #[test]
    fn test_format_key() {
        assert_eq!(
            KeyBindings::format_key(KeyCode::Char('q'), KeyModifiers::NONE),
            "q"
        );
        assert_eq!(
            KeyBindings::format_key(KeyCode::Char('c'), KeyModifiers::CONTROL),
            "Ctrl+c"
        );
        assert_eq!(
            KeyBindings::format_key(KeyCode::F(1), KeyModifiers::NONE),
            "F1"
        );
    }
}
