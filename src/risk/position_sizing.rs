//! Kelly criterion and position sizing strategies for trading RL.
//!
//! This module provides various position sizing methods including:
//!
//! - Full Kelly criterion
//! - Fractional Kelly (half-Kelly, quarter-Kelly)
//! - Optimal-f calculation
//! - Volatility-based sizing (ATR)
//! - Fixed fractional sizing
//! - Anti-martingale sizing
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::risk::{
//!     PositionSizer, PositionSizingConfig, SizingMethod, KellyCalculator,
//! };
//!
//! // Create a position sizer using half-Kelly
//! let sizer = PositionSizer::new(PositionSizingConfig::default()
//!     .method(SizingMethod::HalfKelly)
//!     .max_position(0.25));
//!
//! // Calculate position size
//! let size = sizer.calculate_size(win_rate, avg_win, avg_loss, current_equity);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Position sizing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum SizingMethod {
    /// Full Kelly criterion.
    FullKelly,
    /// Half Kelly (more conservative).
    #[default]
    HalfKelly,
    /// Quarter Kelly (very conservative).
    QuarterKelly,
    /// Custom fractional Kelly.
    FractionalKelly,
    /// Optimal-f calculation.
    OptimalF,
    /// Volatility-based sizing (ATR).
    Volatility,
    /// Fixed fractional of equity.
    FixedFractional,
    /// Anti-martingale (increase after wins).
    AntiMartingale,
    /// Constant position size.
    Constant,
}


/// Result of Kelly criterion calculation.
#[derive(Debug, Clone)]
pub struct KellyResult {
    /// Optimal Kelly fraction.
    pub kelly_fraction: f64,
    /// Expected edge (expected return per bet).
    pub edge: f64,
    /// Expected growth rate (log return).
    pub growth_rate: f64,
    /// Win rate used in calculation.
    pub win_rate: f64,
    /// Win/loss ratio used.
    pub win_loss_ratio: f64,
}

/// Kelly criterion calculator.
pub struct KellyCalculator {
    /// Historical trade outcomes (true = win, false = loss).
    outcomes: VecDeque<bool>,
    /// Historical win amounts.
    win_amounts: VecDeque<f64>,
    /// Historical loss amounts.
    loss_amounts: VecDeque<f64>,
    /// Window size for rolling calculation.
    window_size: usize,
}

impl KellyCalculator {
    /// Create a new Kelly calculator.
    pub fn new(window_size: usize) -> Self {
        Self {
            outcomes: VecDeque::with_capacity(window_size),
            win_amounts: VecDeque::with_capacity(window_size),
            loss_amounts: VecDeque::with_capacity(window_size),
            window_size,
        }
    }

    /// Record a trade outcome.
    pub fn record_trade(&mut self, is_win: bool, amount: f64) {
        // Remove oldest if at capacity
        if self.outcomes.len() >= self.window_size {
            self.outcomes.pop_front();
        }
        self.outcomes.push_back(is_win);

        if is_win {
            if self.win_amounts.len() >= self.window_size {
                self.win_amounts.pop_front();
            }
            self.win_amounts.push_back(amount);
        } else {
            if self.loss_amounts.len() >= self.window_size {
                self.loss_amounts.pop_front();
            }
            self.loss_amounts.push_back(amount.abs());
        }
    }

    /// Calculate the win rate.
    pub fn win_rate(&self) -> f64 {
        if self.outcomes.is_empty() {
            return 0.5; // Default to 50%
        }
        let wins = self.outcomes.iter().filter(|&&o| o).count();
        wins as f64 / self.outcomes.len() as f64
    }

    /// Calculate average win amount.
    pub fn avg_win(&self) -> f64 {
        if self.win_amounts.is_empty() {
            return 0.0;
        }
        self.win_amounts.iter().sum::<f64>() / self.win_amounts.len() as f64
    }

    /// Calculate average loss amount.
    pub fn avg_loss(&self) -> f64 {
        if self.loss_amounts.is_empty() {
            return 0.0;
        }
        self.loss_amounts.iter().sum::<f64>() / self.loss_amounts.len() as f64
    }

    /// Calculate the Kelly fraction.
    ///
    /// Kelly formula: f* = (p * b - q) / b
    /// where:
    /// - p = probability of winning
    /// - q = probability of losing (1 - p)
    /// - b = win/loss ratio (avg_win / avg_loss)
    pub fn calculate(&self) -> KellyResult {
        let p = self.win_rate();
        let q = 1.0 - p;
        let avg_win = self.avg_win();
        let avg_loss = self.avg_loss();

        // Handle edge cases
        if avg_loss <= 0.0 || avg_win <= 0.0 {
            return KellyResult {
                kelly_fraction: 0.0,
                edge: 0.0,
                growth_rate: 0.0,
                win_rate: p,
                win_loss_ratio: 0.0,
            };
        }

        let b = avg_win / avg_loss;
        let kelly = (p * b - q) / b;

        // Expected edge
        let edge = p * avg_win - q * avg_loss;

        // Expected growth rate (approximation)
        let growth_rate = if kelly > 0.0 && kelly < 1.0 {
            p * (1.0 + kelly * b).ln() + q * (1.0 - kelly).ln()
        } else {
            0.0
        };

        KellyResult {
            kelly_fraction: kelly.max(0.0), // Never negative
            edge,
            growth_rate,
            win_rate: p,
            win_loss_ratio: b,
        }
    }

    /// Calculate Kelly fraction from explicit parameters.
    pub fn calculate_from_params(win_rate: f64, avg_win: f64, avg_loss: f64) -> KellyResult {
        let p = win_rate;
        let q = 1.0 - p;

        if avg_loss <= 0.0 || avg_win <= 0.0 {
            return KellyResult {
                kelly_fraction: 0.0,
                edge: 0.0,
                growth_rate: 0.0,
                win_rate: p,
                win_loss_ratio: 0.0,
            };
        }

        let b = avg_win / avg_loss;
        let kelly = (p * b - q) / b;
        let edge = p * avg_win - q * avg_loss;

        let growth_rate = if kelly > 0.0 && kelly < 1.0 {
            p * (1.0 + kelly * b).ln() + q * (1.0 - kelly).ln()
        } else {
            0.0
        };

        KellyResult {
            kelly_fraction: kelly.max(0.0),
            edge,
            growth_rate,
            win_rate: p,
            win_loss_ratio: b,
        }
    }

    /// Get number of recorded trades.
    pub fn trade_count(&self) -> usize {
        self.outcomes.len()
    }

    /// Reset the calculator.
    pub fn reset(&mut self) {
        self.outcomes.clear();
        self.win_amounts.clear();
        self.loss_amounts.clear();
    }
}

/// Volatility-based position sizer using ATR.
pub struct VolatilitySizer {
    /// ATR values.
    atr_values: VecDeque<f64>,
    /// Window size for ATR calculation.
    window_size: usize,
    /// Risk per trade as fraction of equity.
    risk_per_trade: f64,
}

impl VolatilitySizer {
    /// Create a new volatility sizer.
    pub fn new(window_size: usize, risk_per_trade: f64) -> Self {
        Self {
            atr_values: VecDeque::with_capacity(window_size),
            window_size,
            risk_per_trade,
        }
    }

    /// Update with new ATR value.
    pub fn update_atr(&mut self, high: f64, low: f64, prev_close: f64) {
        // True Range calculation
        let tr = (high - low)
            .max((high - prev_close).abs())
            .max((low - prev_close).abs());

        if self.atr_values.len() >= self.window_size {
            self.atr_values.pop_front();
        }
        self.atr_values.push_back(tr);
    }

    /// Get current ATR.
    pub fn atr(&self) -> f64 {
        if self.atr_values.is_empty() {
            return 0.0;
        }
        self.atr_values.iter().sum::<f64>() / self.atr_values.len() as f64
    }

    /// Calculate position size based on ATR.
    ///
    /// Position size = (Equity * Risk%) / (ATR * ATR_multiplier)
    pub fn calculate_size(&self, equity: f64, price: f64, atr_multiplier: f64) -> f64 {
        let atr = self.atr();
        if atr <= 0.0 || price <= 0.0 {
            return 0.0;
        }

        let risk_amount = equity * self.risk_per_trade;
        let stop_distance = atr * atr_multiplier;
        let shares = risk_amount / stop_distance;

        // Return as fraction of equity
        (shares * price) / equity
    }

    /// Reset the sizer.
    pub fn reset(&mut self) {
        self.atr_values.clear();
    }
}

/// Configuration for position sizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizingConfig {
    /// Sizing method to use.
    pub method: SizingMethod,
    /// Kelly fraction (for FractionalKelly method).
    pub kelly_fraction: f64,
    /// Maximum position size (as fraction of equity).
    pub max_position: f64,
    /// Minimum position size.
    pub min_position: f64,
    /// Fixed position size (for Constant method).
    pub fixed_size: f64,
    /// Fixed fractional percentage (for FixedFractional method).
    pub fixed_fraction: f64,
    /// Risk per trade (for Volatility method).
    pub risk_per_trade: f64,
    /// ATR multiplier (for Volatility method).
    pub atr_multiplier: f64,
    /// Window size for calculations.
    pub window_size: usize,
    /// Anti-martingale step size.
    pub anti_martingale_step: f64,
    /// Base position for anti-martingale.
    pub anti_martingale_base: f64,
}

impl Default for PositionSizingConfig {
    fn default() -> Self {
        Self {
            method: SizingMethod::HalfKelly,
            kelly_fraction: 0.5, // Half-Kelly
            max_position: 1.0,
            min_position: 0.0,
            fixed_size: 0.1,
            fixed_fraction: 0.02, // 2% per trade
            risk_per_trade: 0.01, // 1% risk
            atr_multiplier: 2.0,
            window_size: 100,
            anti_martingale_step: 0.25,
            anti_martingale_base: 0.1,
        }
    }
}

impl PositionSizingConfig {
    /// Create a new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set sizing method.
    pub fn method(mut self, m: SizingMethod) -> Self {
        self.method = m;
        self
    }

    /// Set Kelly fraction.
    pub fn kelly_fraction(mut self, f: f64) -> Self {
        self.kelly_fraction = f;
        self
    }

    /// Set maximum position.
    pub fn max_position(mut self, max: f64) -> Self {
        self.max_position = max;
        self
    }

    /// Set minimum position.
    pub fn min_position(mut self, min: f64) -> Self {
        self.min_position = min;
        self
    }

    /// Set fixed size.
    pub fn fixed_size(mut self, size: f64) -> Self {
        self.fixed_size = size;
        self
    }

    /// Set fixed fraction.
    pub fn fixed_fraction(mut self, fraction: f64) -> Self {
        self.fixed_fraction = fraction;
        self
    }

    /// Set risk per trade.
    pub fn risk_per_trade(mut self, risk: f64) -> Self {
        self.risk_per_trade = risk;
        self
    }

    /// Set ATR multiplier.
    pub fn atr_multiplier(mut self, mult: f64) -> Self {
        self.atr_multiplier = mult;
        self
    }

    /// Set window size.
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set anti-martingale parameters.
    pub fn anti_martingale(mut self, base: f64, step: f64) -> Self {
        self.anti_martingale_base = base;
        self.anti_martingale_step = step;
        self
    }
}

/// Position sizer that implements various sizing strategies.
pub struct PositionSizer {
    /// Configuration.
    config: PositionSizingConfig,
    /// Kelly calculator.
    kelly: KellyCalculator,
    /// Volatility sizer.
    vol_sizer: VolatilitySizer,
    /// Consecutive wins (for anti-martingale).
    consecutive_wins: usize,
    /// Last trade outcome.
    last_was_win: Option<bool>,
}

impl PositionSizer {
    /// Create a new position sizer.
    pub fn new(config: PositionSizingConfig) -> Self {
        let kelly = KellyCalculator::new(config.window_size);
        let vol_sizer = VolatilitySizer::new(config.window_size, config.risk_per_trade);

        Self {
            config,
            kelly,
            vol_sizer,
            consecutive_wins: 0,
            last_was_win: None,
        }
    }

    /// Record a trade outcome.
    pub fn record_trade(&mut self, is_win: bool, amount: f64) {
        self.kelly.record_trade(is_win, amount);

        // Track consecutive wins for anti-martingale
        if is_win {
            if self.last_was_win == Some(true) {
                self.consecutive_wins += 1;
            } else {
                self.consecutive_wins = 1;
            }
        } else {
            self.consecutive_wins = 0;
        }
        self.last_was_win = Some(is_win);
    }

    /// Update volatility data.
    pub fn update_volatility(&mut self, high: f64, low: f64, prev_close: f64) {
        self.vol_sizer.update_atr(high, low, prev_close);
    }

    /// Calculate position size.
    pub fn calculate_size(&self, equity: f64, price: f64) -> f64 {
        let raw_size = match self.config.method {
            SizingMethod::FullKelly => {
                let result = self.kelly.calculate();
                result.kelly_fraction
            }
            SizingMethod::HalfKelly => {
                let result = self.kelly.calculate();
                result.kelly_fraction * 0.5
            }
            SizingMethod::QuarterKelly => {
                let result = self.kelly.calculate();
                result.kelly_fraction * 0.25
            }
            SizingMethod::FractionalKelly => {
                let result = self.kelly.calculate();
                result.kelly_fraction * self.config.kelly_fraction
            }
            SizingMethod::OptimalF => {
                // Optimal-f is similar to Kelly but considers worst loss
                let result = self.kelly.calculate();
                // Conservative: use half of Kelly
                result.kelly_fraction * 0.5
            }
            SizingMethod::Volatility => {
                self.vol_sizer
                    .calculate_size(equity, price, self.config.atr_multiplier)
            }
            SizingMethod::FixedFractional => self.config.fixed_fraction,
            SizingMethod::AntiMartingale => {
                let base = self.config.anti_martingale_base;
                let step = self.config.anti_martingale_step;
                base + (self.consecutive_wins as f64 * step)
            }
            SizingMethod::Constant => self.config.fixed_size,
        };

        // Apply limits
        raw_size.clamp(self.config.min_position, self.config.max_position)
    }

    /// Calculate position size with explicit parameters.
    pub fn calculate_size_with_params(
        &self,
        win_rate: f64,
        avg_win: f64,
        avg_loss: f64,
    ) -> f64 {
        let kelly_result = KellyCalculator::calculate_from_params(win_rate, avg_win, avg_loss);

        let raw_size = match self.config.method {
            SizingMethod::FullKelly => kelly_result.kelly_fraction,
            SizingMethod::HalfKelly => kelly_result.kelly_fraction * 0.5,
            SizingMethod::QuarterKelly => kelly_result.kelly_fraction * 0.25,
            SizingMethod::FractionalKelly => {
                kelly_result.kelly_fraction * self.config.kelly_fraction
            }
            SizingMethod::OptimalF => kelly_result.kelly_fraction * 0.5,
            SizingMethod::FixedFractional => self.config.fixed_fraction,
            SizingMethod::Constant => self.config.fixed_size,
            _ => kelly_result.kelly_fraction * 0.5, // Default to half-Kelly
        };

        raw_size.clamp(self.config.min_position, self.config.max_position)
    }

    /// Get current Kelly result.
    pub fn kelly_result(&self) -> KellyResult {
        self.kelly.calculate()
    }

    /// Get current ATR.
    pub fn current_atr(&self) -> f64 {
        self.vol_sizer.atr()
    }

    /// Get trade count.
    pub fn trade_count(&self) -> usize {
        self.kelly.trade_count()
    }

    /// Get consecutive wins.
    pub fn consecutive_wins(&self) -> usize {
        self.consecutive_wins
    }

    /// Reset the sizer state.
    pub fn reset(&mut self) {
        self.kelly.reset();
        self.vol_sizer.reset();
        self.consecutive_wins = 0;
        self.last_was_win = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_calculation() {
        let mut kelly = KellyCalculator::new(100);

        // Record some trades: 60% win rate, 2:1 win/loss ratio
        for i in 0..100 {
            if i % 10 < 6 {
                // 60% wins
                kelly.record_trade(true, 200.0);
            } else {
                kelly.record_trade(false, 100.0);
            }
        }

        let result = kelly.calculate();
        assert!((result.win_rate - 0.6).abs() < 0.01);
        assert!((result.win_loss_ratio - 2.0).abs() < 0.01);
        assert!(result.kelly_fraction > 0.0);
        assert!(result.edge > 0.0);
    }

    #[test]
    fn test_kelly_from_params() {
        // 60% win rate, avg win = 200, avg loss = 100
        let result = KellyCalculator::calculate_from_params(0.6, 200.0, 100.0);

        // Kelly = (0.6 * 2 - 0.4) / 2 = (1.2 - 0.4) / 2 = 0.4
        assert!((result.kelly_fraction - 0.4).abs() < 0.01);
        assert!(result.edge > 0.0);
    }

    #[test]
    fn test_position_sizer_half_kelly() {
        let config = PositionSizingConfig::default()
            .method(SizingMethod::HalfKelly)
            .max_position(0.5);
        let mut sizer = PositionSizer::new(config);

        // Record some trades
        for i in 0..50 {
            if i % 10 < 6 {
                sizer.record_trade(true, 200.0);
            } else {
                sizer.record_trade(false, 100.0);
            }
        }

        let size = sizer.calculate_size(100000.0, 100.0);
        assert!(size > 0.0);
        assert!(size <= 0.5);
    }

    #[test]
    fn test_volatility_sizer() {
        let mut sizer = VolatilitySizer::new(14, 0.01);

        // Add some volatility data
        for i in 0..20 {
            let base = 100.0;
            let high = base + (i as f64 * 0.5);
            let low = base - (i as f64 * 0.3);
            let prev_close = base;
            sizer.update_atr(high, low, prev_close);
        }

        let atr = sizer.atr();
        assert!(atr > 0.0);

        let size = sizer.calculate_size(100000.0, 100.0, 2.0);
        assert!(size > 0.0);
    }

    #[test]
    fn test_anti_martingale() {
        let config = PositionSizingConfig::default()
            .method(SizingMethod::AntiMartingale)
            .anti_martingale(0.1, 0.05)
            .max_position(0.5);
        let mut sizer = PositionSizer::new(config);

        // Initial size
        let size1 = sizer.calculate_size(100000.0, 100.0);
        assert!((size1 - 0.1).abs() < 0.01);

        // After a win
        sizer.record_trade(true, 100.0);
        let size2 = sizer.calculate_size(100000.0, 100.0);
        assert!((size2 - 0.15).abs() < 0.01);

        // After two wins
        sizer.record_trade(true, 100.0);
        let size3 = sizer.calculate_size(100000.0, 100.0);
        assert!((size3 - 0.20).abs() < 0.01);

        // After a loss - reset
        sizer.record_trade(false, 50.0);
        let size4 = sizer.calculate_size(100000.0, 100.0);
        assert!((size4 - 0.1).abs() < 0.01);
    }

    #[test]
    fn test_fixed_fractional() {
        let config = PositionSizingConfig::default()
            .method(SizingMethod::FixedFractional)
            .fixed_fraction(0.02);
        let sizer = PositionSizer::new(config);

        let size = sizer.calculate_size(100000.0, 100.0);
        assert!((size - 0.02).abs() < 1e-10);
    }
}
