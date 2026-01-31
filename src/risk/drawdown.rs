//! Drawdown control and monitoring for trading RL.
//!
//! This module provides real-time drawdown tracking and dynamic risk management,
//! including:
//!
//! - Real-time drawdown tracking
//! - Maximum drawdown limits with early stopping
//! - Dynamic risk scaling based on drawdown
//! - Drawdown recovery mode
//! - Peak equity tracking
//! - Underwater curve calculation
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::risk::{
//!     DrawdownController, DrawdownConfig, DrawdownEvent,
//! };
//!
//! let mut controller = DrawdownController::new(DrawdownConfig::default()
//!     .max_drawdown(0.20)
//!     .recovery_threshold(0.10)
//!     .scaling_method(RiskScaling::Linear));
//!
//! // Update with equity
//! let event = controller.update(10000.0);
//! match event {
//!     Some(DrawdownEvent::MaxDrawdownBreached) => {
//!         // Stop trading or reduce risk
//!     }
//!     Some(DrawdownEvent::RecoveryMode) => {
//!         // Enter recovery mode
//!     }
//!     _ => {}
//! }
//!
//! // Get current risk scale
//! let risk_scale = controller.risk_scale();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Events triggered by drawdown controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawdownEvent {
    /// Maximum drawdown limit breached.
    MaxDrawdownBreached,
    /// Entered drawdown recovery mode.
    RecoveryMode,
    /// Exited recovery mode (recovered).
    RecoveryComplete,
    /// New peak equity reached.
    NewPeak,
    /// Warning threshold crossed.
    WarningThreshold,
}

/// Risk scaling method based on drawdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskScaling {
    /// No scaling (constant risk).
    None,
    /// Linear reduction as drawdown increases.
    Linear,
    /// Exponential reduction.
    Exponential,
    /// Step function (discrete levels).
    Step,
    /// Custom sigmoid-based scaling.
    Sigmoid,
}

impl Default for RiskScaling {
    fn default() -> Self {
        RiskScaling::Linear
    }
}

/// Current state of the drawdown controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DrawdownState {
    /// Normal trading.
    Normal,
    /// In recovery mode (reduced risk).
    Recovery,
    /// Stopped due to max drawdown breach.
    Stopped,
}

impl Default for DrawdownState {
    fn default() -> Self {
        DrawdownState::Normal
    }
}

/// Configuration for drawdown controller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawdownConfig {
    /// Maximum allowed drawdown (fraction, e.g., 0.20 = 20%).
    pub max_drawdown: f64,
    /// Warning threshold (fraction).
    pub warning_threshold: f64,
    /// Drawdown level to trigger recovery mode.
    pub recovery_threshold: f64,
    /// Drawdown level to exit recovery mode.
    pub recovery_exit_threshold: f64,
    /// Risk scaling method.
    pub scaling_method: RiskScaling,
    /// Minimum risk scale in recovery mode.
    pub min_risk_scale: f64,
    /// Step levels for step scaling [drawdown_threshold, risk_scale].
    pub step_levels: Vec<(f64, f64)>,
    /// Whether to track underwater curve.
    pub track_underwater: bool,
    /// Maximum underwater curve history.
    pub underwater_history_size: usize,
    /// Enable automatic early stopping.
    pub enable_early_stop: bool,
}

impl Default for DrawdownConfig {
    fn default() -> Self {
        Self {
            max_drawdown: 0.20,        // 20% max drawdown
            warning_threshold: 0.10,   // 10% warning
            recovery_threshold: 0.15,  // Enter recovery at 15%
            recovery_exit_threshold: 0.05, // Exit recovery when DD < 5%
            scaling_method: RiskScaling::Linear,
            min_risk_scale: 0.25,
            step_levels: vec![
                (0.05, 0.9),  // 5% DD -> 90% risk
                (0.10, 0.7),  // 10% DD -> 70% risk
                (0.15, 0.5),  // 15% DD -> 50% risk
                (0.20, 0.0),  // 20% DD -> 0% risk (stop)
            ],
            track_underwater: true,
            underwater_history_size: 1000,
            enable_early_stop: true,
        }
    }
}

impl DrawdownConfig {
    /// Create a new drawdown config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum drawdown.
    pub fn max_drawdown(mut self, max: f64) -> Self {
        self.max_drawdown = max;
        self
    }

    /// Set warning threshold.
    pub fn warning_threshold(mut self, thresh: f64) -> Self {
        self.warning_threshold = thresh;
        self
    }

    /// Set recovery threshold.
    pub fn recovery_threshold(mut self, thresh: f64) -> Self {
        self.recovery_threshold = thresh;
        self
    }

    /// Set recovery exit threshold.
    pub fn recovery_exit_threshold(mut self, thresh: f64) -> Self {
        self.recovery_exit_threshold = thresh;
        self
    }

    /// Set scaling method.
    pub fn scaling_method(mut self, method: RiskScaling) -> Self {
        self.scaling_method = method;
        self
    }

    /// Set minimum risk scale.
    pub fn min_risk_scale(mut self, min: f64) -> Self {
        self.min_risk_scale = min;
        self
    }

    /// Set step levels.
    pub fn step_levels(mut self, levels: Vec<(f64, f64)>) -> Self {
        self.step_levels = levels;
        self
    }

    /// Enable/disable underwater tracking.
    pub fn track_underwater(mut self, enabled: bool) -> Self {
        self.track_underwater = enabled;
        self
    }

    /// Enable/disable early stopping.
    pub fn enable_early_stop(mut self, enabled: bool) -> Self {
        self.enable_early_stop = enabled;
        self
    }
}

/// Underwater curve data point.
#[derive(Debug, Clone, Copy)]
pub struct UnderwaterPoint {
    /// Timestamp or step index.
    pub step: usize,
    /// Equity value.
    pub equity: f64,
    /// Peak equity at this point.
    pub peak: f64,
    /// Drawdown at this point.
    pub drawdown: f64,
}

/// Underwater curve for visualization.
#[derive(Debug, Clone)]
pub struct UnderwaterCurve {
    /// Data points.
    points: VecDeque<UnderwaterPoint>,
    /// Maximum size.
    max_size: usize,
}

impl UnderwaterCurve {
    /// Create a new underwater curve.
    pub fn new(max_size: usize) -> Self {
        Self {
            points: VecDeque::with_capacity(max_size),
            max_size,
        }
    }

    /// Add a point.
    pub fn push(&mut self, point: UnderwaterPoint) {
        if self.points.len() >= self.max_size {
            self.points.pop_front();
        }
        self.points.push_back(point);
    }

    /// Get all points.
    pub fn points(&self) -> &VecDeque<UnderwaterPoint> {
        &self.points
    }

    /// Get drawdown values only.
    pub fn drawdowns(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.drawdown).collect()
    }

    /// Get equity values only.
    pub fn equities(&self) -> Vec<f64> {
        self.points.iter().map(|p| p.equity).collect()
    }

    /// Clear the curve.
    pub fn clear(&mut self) {
        self.points.clear();
    }

    /// Get length.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Drawdown controller for real-time drawdown management.
pub struct DrawdownController {
    /// Configuration.
    config: DrawdownConfig,
    /// Current equity.
    equity: f64,
    /// Peak equity (high water mark).
    peak_equity: f64,
    /// Current drawdown (fraction).
    current_drawdown: f64,
    /// Maximum drawdown observed.
    max_drawdown_observed: f64,
    /// Current state.
    state: DrawdownState,
    /// Current risk scale.
    risk_scale: f64,
    /// Step counter.
    step: usize,
    /// Underwater curve.
    underwater_curve: Option<UnderwaterCurve>,
    /// Number of steps in current drawdown.
    drawdown_duration: usize,
    /// Longest drawdown duration.
    max_drawdown_duration: usize,
    /// Initial equity.
    initial_equity: f64,
}

impl DrawdownController {
    /// Create a new drawdown controller.
    pub fn new(config: DrawdownConfig) -> Self {
        let underwater_curve = if config.track_underwater {
            Some(UnderwaterCurve::new(config.underwater_history_size))
        } else {
            None
        };

        Self {
            config,
            equity: 0.0,
            peak_equity: 0.0,
            current_drawdown: 0.0,
            max_drawdown_observed: 0.0,
            state: DrawdownState::Normal,
            risk_scale: 1.0,
            step: 0,
            underwater_curve,
            drawdown_duration: 0,
            max_drawdown_duration: 0,
            initial_equity: 0.0,
        }
    }

    /// Initialize with starting equity.
    pub fn init(&mut self, initial_equity: f64) {
        self.equity = initial_equity;
        self.peak_equity = initial_equity;
        self.initial_equity = initial_equity;
        self.current_drawdown = 0.0;
        self.max_drawdown_observed = 0.0;
        self.state = DrawdownState::Normal;
        self.risk_scale = 1.0;
        self.step = 0;
        self.drawdown_duration = 0;
        self.max_drawdown_duration = 0;
        if let Some(ref mut curve) = self.underwater_curve {
            curve.clear();
        }
    }

    /// Update with new equity value.
    pub fn update(&mut self, new_equity: f64) -> Option<DrawdownEvent> {
        self.equity = new_equity;
        self.step += 1;

        let old_drawdown = self.current_drawdown;
        let old_peak = self.peak_equity;
        let old_state = self.state;

        // Update peak
        let new_peak = new_equity > self.peak_equity;
        if new_peak {
            self.peak_equity = new_equity;
            self.drawdown_duration = 0;
        } else {
            self.drawdown_duration += 1;
            self.max_drawdown_duration = self.max_drawdown_duration.max(self.drawdown_duration);
        }

        // Calculate drawdown
        self.current_drawdown = if self.peak_equity > 0.0 {
            (self.peak_equity - self.equity) / self.peak_equity
        } else {
            0.0
        };

        // Update max observed
        self.max_drawdown_observed = self.max_drawdown_observed.max(self.current_drawdown);

        // Update risk scale
        self.risk_scale = self.calculate_risk_scale();

        // Track underwater curve
        if let Some(ref mut curve) = self.underwater_curve {
            curve.push(UnderwaterPoint {
                step: self.step,
                equity: self.equity,
                peak: self.peak_equity,
                drawdown: self.current_drawdown,
            });
        }

        // Determine state transitions and events
        let event = self.determine_event(old_drawdown, old_peak, old_state, new_peak);

        event
    }

    /// Determine what event (if any) occurred.
    fn determine_event(
        &mut self,
        old_drawdown: f64,
        _old_peak: f64,
        old_state: DrawdownState,
        new_peak: bool,
    ) -> Option<DrawdownEvent> {
        // Check max drawdown breach
        if self.config.enable_early_stop && self.current_drawdown >= self.config.max_drawdown {
            self.state = DrawdownState::Stopped;
            return Some(DrawdownEvent::MaxDrawdownBreached);
        }

        // Check recovery mode transitions
        match old_state {
            DrawdownState::Normal => {
                if self.current_drawdown >= self.config.recovery_threshold {
                    self.state = DrawdownState::Recovery;
                    return Some(DrawdownEvent::RecoveryMode);
                }
                if old_drawdown < self.config.warning_threshold
                    && self.current_drawdown >= self.config.warning_threshold
                {
                    return Some(DrawdownEvent::WarningThreshold);
                }
            }
            DrawdownState::Recovery => {
                if self.current_drawdown < self.config.recovery_exit_threshold {
                    self.state = DrawdownState::Normal;
                    return Some(DrawdownEvent::RecoveryComplete);
                }
            }
            DrawdownState::Stopped => {
                // Stay stopped
            }
        }

        // Check new peak
        if new_peak && old_state == DrawdownState::Normal {
            return Some(DrawdownEvent::NewPeak);
        }

        None
    }

    /// Calculate risk scale based on current drawdown.
    fn calculate_risk_scale(&self) -> f64 {
        if self.state == DrawdownState::Stopped {
            return 0.0;
        }

        match self.config.scaling_method {
            RiskScaling::None => 1.0,
            RiskScaling::Linear => {
                // Linear reduction from 1.0 at 0% DD to min_risk_scale at max_drawdown
                let ratio = self.current_drawdown / self.config.max_drawdown;
                let scale = 1.0 - ratio * (1.0 - self.config.min_risk_scale);
                scale.clamp(self.config.min_risk_scale, 1.0)
            }
            RiskScaling::Exponential => {
                // Exponential reduction
                let ratio = self.current_drawdown / self.config.max_drawdown;
                let scale = (-ratio * 3.0).exp(); // Decay factor
                scale.clamp(self.config.min_risk_scale, 1.0)
            }
            RiskScaling::Step => {
                // Find appropriate step level
                let mut scale = 1.0;
                for (threshold, level_scale) in &self.config.step_levels {
                    if self.current_drawdown >= *threshold {
                        scale = *level_scale;
                    }
                }
                scale.max(self.config.min_risk_scale)
            }
            RiskScaling::Sigmoid => {
                // Sigmoid function centered at half of max drawdown
                let midpoint = self.config.max_drawdown / 2.0;
                let steepness = 10.0 / self.config.max_drawdown;
                let x = self.current_drawdown - midpoint;
                let sigmoid = 1.0 / (1.0 + (steepness * x).exp());
                let scale = sigmoid * (1.0 - self.config.min_risk_scale) + self.config.min_risk_scale;
                scale.clamp(self.config.min_risk_scale, 1.0)
            }
        }
    }

    /// Get current drawdown.
    pub fn current_drawdown(&self) -> f64 {
        self.current_drawdown
    }

    /// Get maximum observed drawdown.
    pub fn max_drawdown_observed(&self) -> f64 {
        self.max_drawdown_observed
    }

    /// Get current risk scale.
    pub fn risk_scale(&self) -> f64 {
        self.risk_scale
    }

    /// Get current state.
    pub fn state(&self) -> DrawdownState {
        self.state
    }

    /// Check if trading should stop.
    pub fn should_stop(&self) -> bool {
        self.state == DrawdownState::Stopped
    }

    /// Check if in recovery mode.
    pub fn in_recovery(&self) -> bool {
        self.state == DrawdownState::Recovery
    }

    /// Get peak equity.
    pub fn peak_equity(&self) -> f64 {
        self.peak_equity
    }

    /// Get current equity.
    pub fn equity(&self) -> f64 {
        self.equity
    }

    /// Get initial equity.
    pub fn initial_equity(&self) -> f64 {
        self.initial_equity
    }

    /// Get total return.
    pub fn total_return(&self) -> f64 {
        if self.initial_equity > 0.0 {
            (self.equity - self.initial_equity) / self.initial_equity
        } else {
            0.0
        }
    }

    /// Get drawdown duration.
    pub fn drawdown_duration(&self) -> usize {
        self.drawdown_duration
    }

    /// Get max drawdown duration.
    pub fn max_drawdown_duration(&self) -> usize {
        self.max_drawdown_duration
    }

    /// Get underwater curve.
    pub fn underwater_curve(&self) -> Option<&UnderwaterCurve> {
        self.underwater_curve.as_ref()
    }

    /// Get the amount needed to recover from drawdown.
    pub fn recovery_amount(&self) -> f64 {
        self.peak_equity - self.equity
    }

    /// Get the percentage gain needed to recover.
    pub fn recovery_gain_needed(&self) -> f64 {
        if self.equity > 0.0 {
            (self.peak_equity - self.equity) / self.equity
        } else {
            0.0
        }
    }

    /// Scale a position size by current risk scale.
    pub fn scale_position(&self, position: f64) -> f64 {
        position * self.risk_scale
    }

    /// Reset the controller.
    pub fn reset(&mut self) {
        self.equity = 0.0;
        self.peak_equity = 0.0;
        self.initial_equity = 0.0;
        self.current_drawdown = 0.0;
        self.max_drawdown_observed = 0.0;
        self.state = DrawdownState::Normal;
        self.risk_scale = 1.0;
        self.step = 0;
        self.drawdown_duration = 0;
        self.max_drawdown_duration = 0;
        if let Some(ref mut curve) = self.underwater_curve {
            curve.clear();
        }
    }

    /// Get a summary of current state.
    pub fn summary(&self) -> DrawdownSummary {
        DrawdownSummary {
            current_drawdown: self.current_drawdown,
            max_drawdown: self.max_drawdown_observed,
            risk_scale: self.risk_scale,
            state: self.state,
            drawdown_duration: self.drawdown_duration,
            max_drawdown_duration: self.max_drawdown_duration,
            recovery_amount: self.recovery_amount(),
            recovery_gain_needed: self.recovery_gain_needed(),
            total_return: self.total_return(),
        }
    }
}

/// Summary of drawdown state.
#[derive(Debug, Clone)]
pub struct DrawdownSummary {
    /// Current drawdown.
    pub current_drawdown: f64,
    /// Maximum drawdown observed.
    pub max_drawdown: f64,
    /// Current risk scale.
    pub risk_scale: f64,
    /// Current state.
    pub state: DrawdownState,
    /// Duration of current drawdown.
    pub drawdown_duration: usize,
    /// Maximum drawdown duration.
    pub max_drawdown_duration: usize,
    /// Amount needed to recover.
    pub recovery_amount: f64,
    /// Percentage gain needed to recover.
    pub recovery_gain_needed: f64,
    /// Total return from initial.
    pub total_return: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drawdown_calculation() {
        let config = DrawdownConfig::default().max_drawdown(0.20);
        let mut controller = DrawdownController::new(config);
        controller.init(10000.0);

        // No drawdown initially
        assert!((controller.current_drawdown() - 0.0).abs() < 1e-10);

        // 10% loss
        controller.update(9000.0);
        assert!((controller.current_drawdown() - 0.10).abs() < 1e-10);

        // Further loss to 8000 (20% from peak)
        controller.update(8000.0);
        assert!((controller.current_drawdown() - 0.20).abs() < 1e-10);
    }

    #[test]
    fn test_peak_tracking() {
        let config = DrawdownConfig::default();
        let mut controller = DrawdownController::new(config);
        controller.init(10000.0);

        // New peak
        let event = controller.update(11000.0);
        assert_eq!(event, Some(DrawdownEvent::NewPeak));
        assert!((controller.peak_equity() - 11000.0).abs() < 1e-10);

        // Drawdown from new peak
        controller.update(10000.0);
        assert!((controller.current_drawdown() - (1000.0 / 11000.0)).abs() < 1e-10);
    }

    #[test]
    fn test_max_drawdown_breach() {
        let config = DrawdownConfig::default()
            .max_drawdown(0.20)
            .enable_early_stop(true);
        let mut controller = DrawdownController::new(config);
        controller.init(10000.0);

        // 20% loss
        let event = controller.update(8000.0);
        assert_eq!(event, Some(DrawdownEvent::MaxDrawdownBreached));
        assert!(controller.should_stop());
        assert_eq!(controller.state(), DrawdownState::Stopped);
    }

    #[test]
    fn test_recovery_mode() {
        let config = DrawdownConfig::default()
            .max_drawdown(0.25)
            .recovery_threshold(0.15)
            .recovery_exit_threshold(0.05);
        let mut controller = DrawdownController::new(config);
        controller.init(10000.0);

        // 15% loss -> recovery mode
        let event = controller.update(8500.0);
        assert_eq!(event, Some(DrawdownEvent::RecoveryMode));
        assert!(controller.in_recovery());

        // Recover back above 95% of peak (below 5% drawdown threshold)
        // At peak of 10000, we need equity > 9500 to be below 5% drawdown
        let event = controller.update(9600.0);
        // Drawdown is now 4% which is below 5% threshold
        assert!(controller.current_drawdown() < 0.05);
        assert_eq!(event, Some(DrawdownEvent::RecoveryComplete));
        assert!(!controller.in_recovery());
    }

    #[test]
    fn test_linear_risk_scaling() {
        let config = DrawdownConfig::default()
            .max_drawdown(0.20)
            .min_risk_scale(0.0)
            .scaling_method(RiskScaling::Linear);
        let mut controller = DrawdownController::new(config);
        controller.init(10000.0);

        // 10% drawdown -> 50% risk
        controller.update(9000.0);
        assert!((controller.risk_scale() - 0.5).abs() < 0.01);

        // 20% drawdown -> 0% risk
        controller.update(8000.0);
        assert!((controller.risk_scale() - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_step_risk_scaling() {
        let config = DrawdownConfig::default()
            .scaling_method(RiskScaling::Step)
            .step_levels(vec![
                (0.05, 0.8),
                (0.10, 0.5),
                (0.15, 0.2),
            ]);
        let mut controller = DrawdownController::new(config);
        controller.init(10000.0);

        // 3% drawdown -> full risk
        controller.update(9700.0);
        assert!((controller.risk_scale() - 1.0).abs() < 0.01);

        // 7% drawdown -> 80% risk
        controller.update(9300.0);
        assert!((controller.risk_scale() - 0.8).abs() < 0.01);

        // 12% drawdown -> 50% risk
        controller.update(8800.0);
        assert!((controller.risk_scale() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_underwater_curve() {
        let config = DrawdownConfig::default().track_underwater(true);
        let mut controller = DrawdownController::new(config);
        controller.init(10000.0);

        controller.update(9500.0);
        controller.update(9000.0);
        controller.update(9500.0);
        controller.update(10000.0);

        let curve = controller.underwater_curve().unwrap();
        assert_eq!(curve.len(), 4);

        let drawdowns = curve.drawdowns();
        assert!((drawdowns[0] - 0.05).abs() < 0.01);
        assert!((drawdowns[1] - 0.10).abs() < 0.01);
    }

    #[test]
    fn test_recovery_calculations() {
        let config = DrawdownConfig::default();
        let mut controller = DrawdownController::new(config);
        controller.init(10000.0);

        // 20% loss
        controller.update(8000.0);

        assert!((controller.recovery_amount() - 2000.0).abs() < 1e-10);
        assert!((controller.recovery_gain_needed() - 0.25).abs() < 1e-10); // Need 25% gain
    }

    #[test]
    fn test_position_scaling() {
        let config = DrawdownConfig::default()
            .max_drawdown(0.20)
            .min_risk_scale(0.0)
            .scaling_method(RiskScaling::Linear);
        let mut controller = DrawdownController::new(config);
        controller.init(10000.0);

        // 10% drawdown
        controller.update(9000.0);

        // Scale a position
        let original_position = 1000.0;
        let scaled = controller.scale_position(original_position);
        assert!((scaled - 500.0).abs() < 1.0); // Should be about 50%
    }
}
