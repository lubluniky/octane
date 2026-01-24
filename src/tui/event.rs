//! Event handling for the Rocket TUI
//!
//! Handles keyboard input and tick events for animations.

use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};

use crossterm::event::{self, Event as CrosstermEvent, KeyCode, KeyEvent, KeyModifiers};

/// TUI events
#[derive(Debug, Clone)]
pub enum Event {
    /// Terminal tick for animations
    Tick,
    /// Keyboard input
    Key(KeyEvent),
    /// Mouse event (future)
    Mouse(crossterm::event::MouseEvent),
    /// Terminal resize
    Resize(u16, u16),
}

/// Event handler that manages the event loop
pub struct EventHandler {
    /// Event sender
    #[allow(dead_code)]
    sender: mpsc::Sender<Event>,
    /// Event receiver
    receiver: mpsc::Receiver<Event>,
    /// Event handler thread
    #[allow(dead_code)]
    handler: thread::JoinHandle<()>,
}

impl EventHandler {
    /// Create a new event handler with the specified tick rate
    pub fn new(tick_rate: Duration) -> Self {
        let (sender, receiver) = mpsc::channel();
        let handler = {
            let sender = sender.clone();
            thread::spawn(move || {
                let mut last_tick = Instant::now();
                loop {
                    // Calculate timeout until next tick
                    let timeout = tick_rate
                        .checked_sub(last_tick.elapsed())
                        .unwrap_or(Duration::ZERO);

                    // Poll for events with timeout
                    if event::poll(timeout).expect("Failed to poll events") {
                        match event::read().expect("Failed to read event") {
                            CrosstermEvent::Key(key) => {
                                // Check for quit signal (Ctrl+C)
                                if key.code == KeyCode::Char('c')
                                    && key.modifiers.contains(KeyModifiers::CONTROL)
                                {
                                    // Send the key event then exit
                                    let _ = sender.send(Event::Key(key));
                                    return;
                                }
                                if sender.send(Event::Key(key)).is_err() {
                                    return;
                                }
                            }
                            CrosstermEvent::Mouse(mouse) => {
                                if sender.send(Event::Mouse(mouse)).is_err() {
                                    return;
                                }
                            }
                            CrosstermEvent::Resize(width, height) => {
                                if sender.send(Event::Resize(width, height)).is_err() {
                                    return;
                                }
                            }
                            _ => {}
                        }
                    }

                    // Send tick event if interval has passed
                    if last_tick.elapsed() >= tick_rate {
                        if sender.send(Event::Tick).is_err() {
                            return;
                        }
                        last_tick = Instant::now();
                    }
                }
            })
        };

        Self {
            sender,
            receiver,
            handler,
        }
    }

    /// Receive the next event (blocking)
    pub fn next(&self) -> Result<Event, mpsc::RecvError> {
        self.receiver.recv()
    }
}

/// Key action for the application
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Quit the application
    Quit,
    /// Navigate to next tab
    NextTab,
    /// Navigate to previous tab
    PrevTab,
    /// Select dashboard tab
    Dashboard,
    /// Select training tab
    Training,
    /// Select environment tab
    Environment,
    /// Select benchmarks tab
    Benchmarks,
    /// Select settings tab
    Settings,
    /// Toggle pause
    TogglePause,
    /// Toggle help overlay
    ToggleHelp,
    /// Scroll up
    ScrollUp,
    /// Scroll down
    ScrollDown,
    /// Select next item
    SelectNext,
    /// Select previous item
    SelectPrev,
    /// Skip splash screen
    SkipSplash,
    /// No action
    None,
}

impl From<KeyEvent> for KeyAction {
    fn from(key: KeyEvent) -> Self {
        match key.code {
            // Quit
            KeyCode::Char('q') => KeyAction::Quit,
            KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => KeyAction::Quit,
            KeyCode::Esc => KeyAction::Quit,

            // Tab navigation
            KeyCode::Tab => KeyAction::NextTab,
            KeyCode::BackTab => KeyAction::PrevTab,
            KeyCode::Right | KeyCode::Char('l') => KeyAction::NextTab,
            KeyCode::Left | KeyCode::Char('h') if !key.modifiers.is_empty() => KeyAction::PrevTab,

            // Direct tab selection
            KeyCode::Char('1') => KeyAction::Dashboard,
            KeyCode::Char('2') => KeyAction::Training,
            KeyCode::Char('3') => KeyAction::Environment,
            KeyCode::Char('4') => KeyAction::Benchmarks,
            KeyCode::Char('5') => KeyAction::Settings,

            // Controls
            KeyCode::Char(' ') => KeyAction::TogglePause,
            KeyCode::Char('p') => KeyAction::TogglePause,
            KeyCode::Char('h') => KeyAction::ToggleHelp,
            KeyCode::Char('?') => KeyAction::ToggleHelp,

            // Scrolling
            KeyCode::Up | KeyCode::Char('k') => KeyAction::ScrollUp,
            KeyCode::Down | KeyCode::Char('j') => KeyAction::ScrollDown,
            KeyCode::PageUp => KeyAction::ScrollUp,
            KeyCode::PageDown => KeyAction::ScrollDown,

            // Selection
            KeyCode::Enter => KeyAction::SelectNext,

            // Splash skip
            KeyCode::Char(_) => KeyAction::SkipSplash,

            _ => KeyAction::None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_action_quit() {
        let key = KeyEvent::new(KeyCode::Char('q'), KeyModifiers::NONE);
        assert_eq!(KeyAction::from(key), KeyAction::Quit);
    }

    #[test]
    fn test_key_action_tab() {
        let key = KeyEvent::new(KeyCode::Tab, KeyModifiers::NONE);
        assert_eq!(KeyAction::from(key), KeyAction::NextTab);
    }

    #[test]
    fn test_key_action_number() {
        let key = KeyEvent::new(KeyCode::Char('1'), KeyModifiers::NONE);
        assert_eq!(KeyAction::from(key), KeyAction::Dashboard);
    }
}
