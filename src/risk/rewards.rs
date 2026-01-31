//! Risk-adjusted reward shaping for trading RL.
//!
//! This module provides reward transformation functions that incorporate risk
//! metrics into the RL training signal, including:
//!
//! - Sharpe ratio reward shaping
//! - Sortino ratio (downside deviation)
//! - Calmar ratio (return / max drawdown)
//! - Risk parity objectives
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::risk::{
//!     SharpeRewardShaper, SortinoRewardShaper, RewardShaperConfig,
//! };
//!
//! // Create a Sharpe-based reward shaper
//! let mut shaper = SharpeRewardShaper::new(RewardShaperConfig::default()
//!     .window_size(252)
//!     .risk_free_rate(0.02)
//!     .annualization_factor(252.0));
//!
//! // Transform raw reward
//! let raw_reward = 0.01;  // 1% return
//! let shaped_reward = shaper.shape_reward(raw_reward);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Trait for reward shaping transformations.
pub trait RewardShaper: Send + Sync {
    /// Transform a raw reward into a risk-adjusted reward.
    fn shape_reward(&mut self, reward: f64) -> f64;

    /// Get the current risk metric value.
    fn risk_metric(&self) -> f64;

    /// Reset the shaper state (e.g., at episode start).
    fn reset(&mut self);

    /// Get the name of the risk metric.
    fn metric_name(&self) -> &str;
}

/// Rolling statistics calculator.
#[derive(Debug, Clone)]
pub struct RollingStats {
    /// Window size for calculations.
    window_size: usize,
    /// Stored values.
    values: VecDeque<f64>,
    /// Running sum.
    sum: f64,
    /// Running sum of squares.
    sum_sq: f64,
    /// Running sum of negative values squared (for downside deviation).
    sum_neg_sq: f64,
}

impl RollingStats {
    /// Create a new rolling stats calculator.
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: VecDeque::with_capacity(window_size),
            sum: 0.0,
            sum_sq: 0.0,
            sum_neg_sq: 0.0,
        }
    }

    /// Add a new value.
    pub fn push(&mut self, value: f64) {
        // Remove oldest value if at capacity
        if self.values.len() >= self.window_size {
            if let Some(old) = self.values.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
                if old < 0.0 {
                    self.sum_neg_sq -= old * old;
                }
            }
        }

        // Add new value
        self.values.push_back(value);
        self.sum += value;
        self.sum_sq += value * value;
        if value < 0.0 {
            self.sum_neg_sq += value * value;
        }
    }

    /// Get the mean.
    pub fn mean(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f64
        }
    }

    /// Get the variance.
    pub fn variance(&self) -> f64 {
        let n = self.values.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        let mean = self.sum / n;
        (self.sum_sq / n) - (mean * mean)
    }

    /// Get the standard deviation.
    pub fn std_dev(&self) -> f64 {
        self.variance().max(0.0).sqrt()
    }

    /// Get the downside deviation (standard deviation of negative values only).
    pub fn downside_deviation(&self) -> f64 {
        let n = self.values.iter().filter(|&&v| v < 0.0).count() as f64;
        if n < 1.0 {
            return 0.0;
        }
        (self.sum_neg_sq / n).max(0.0).sqrt()
    }

    /// Get the count of values.
    pub fn count(&self) -> usize {
        self.values.len()
    }

    /// Check if we have enough data.
    pub fn is_ready(&self) -> bool {
        self.values.len() >= 2
    }

    /// Reset the statistics.
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.sum_neg_sq = 0.0;
    }

    /// Get the sum of values.
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Get minimum value.
    pub fn min(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    }

    /// Get maximum value.
    pub fn max(&self) -> f64 {
        self.values
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max)
    }
}

/// Configuration for reward shapers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardShaperConfig {
    /// Window size for rolling calculations.
    pub window_size: usize,
    /// Risk-free rate (annual).
    pub risk_free_rate: f64,
    /// Annualization factor (e.g., 252 for daily data).
    pub annualization_factor: f64,
    /// Minimum standard deviation to prevent division by zero.
    pub min_std: f64,
    /// Scaling factor for shaped rewards.
    pub scale: f64,
    /// Target volatility for risk parity.
    pub target_volatility: f64,
    /// Blend factor between raw reward and shaped reward (0 = raw, 1 = shaped).
    pub blend_factor: f64,
}

impl Default for RewardShaperConfig {
    fn default() -> Self {
        Self {
            window_size: 252, // ~1 year of daily data
            risk_free_rate: 0.02,
            annualization_factor: 252.0,
            min_std: 1e-8,
            scale: 1.0,
            target_volatility: 0.15, // 15% annualized volatility
            blend_factor: 1.0,
        }
    }
}

impl RewardShaperConfig {
    /// Create a new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set window size.
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set risk-free rate.
    pub fn risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Set annualization factor.
    pub fn annualization_factor(mut self, factor: f64) -> Self {
        self.annualization_factor = factor;
        self
    }

    /// Set minimum standard deviation.
    pub fn min_std(mut self, min: f64) -> Self {
        self.min_std = min;
        self
    }

    /// Set scale factor.
    pub fn scale(mut self, s: f64) -> Self {
        self.scale = s;
        self
    }

    /// Set target volatility.
    pub fn target_volatility(mut self, vol: f64) -> Self {
        self.target_volatility = vol;
        self
    }

    /// Set blend factor.
    pub fn blend_factor(mut self, blend: f64) -> Self {
        self.blend_factor = blend.clamp(0.0, 1.0);
        self
    }
}

/// Sharpe ratio reward shaper.
///
/// Transforms rewards to encourage high risk-adjusted returns.
/// Sharpe = (E[R] - Rf) / std(R)
pub struct SharpeRewardShaper {
    /// Configuration.
    config: RewardShaperConfig,
    /// Rolling statistics for returns.
    stats: RollingStats,
    /// Current Sharpe ratio.
    current_sharpe: f64,
}

impl SharpeRewardShaper {
    /// Create a new Sharpe reward shaper.
    pub fn new(config: RewardShaperConfig) -> Self {
        let stats = RollingStats::new(config.window_size);
        Self {
            config,
            stats,
            current_sharpe: 0.0,
        }
    }

    /// Compute the Sharpe ratio from current statistics.
    fn compute_sharpe(&self) -> f64 {
        if !self.stats.is_ready() {
            return 0.0;
        }

        let mean_return = self.stats.mean();
        let std_return = self.stats.std_dev().max(self.config.min_std);

        // Convert risk-free rate to per-period
        let rf_period = self.config.risk_free_rate / self.config.annualization_factor;

        // Compute Sharpe ratio
        let sharpe = (mean_return - rf_period) / std_return;

        // Annualize
        sharpe * self.config.annualization_factor.sqrt()
    }
}

impl RewardShaper for SharpeRewardShaper {
    fn shape_reward(&mut self, reward: f64) -> f64 {
        self.stats.push(reward);
        self.current_sharpe = self.compute_sharpe();

        // Blend raw reward with Sharpe-based shaping
        let shaped = if self.stats.is_ready() {
            let std = self.stats.std_dev().max(self.config.min_std);
            reward / std // Risk-adjusted return
        } else {
            reward
        };

        // Apply blend and scale
        let blended = reward * (1.0 - self.config.blend_factor)
            + shaped * self.config.blend_factor;

        blended * self.config.scale
    }

    fn risk_metric(&self) -> f64 {
        self.current_sharpe
    }

    fn reset(&mut self) {
        self.stats.reset();
        self.current_sharpe = 0.0;
    }

    fn metric_name(&self) -> &str {
        "sharpe_ratio"
    }
}

/// Sortino ratio reward shaper.
///
/// Like Sharpe but only penalizes downside volatility.
/// Sortino = (E[R] - Rf) / downside_deviation(R)
pub struct SortinoRewardShaper {
    /// Configuration.
    config: RewardShaperConfig,
    /// Rolling statistics for returns.
    stats: RollingStats,
    /// Current Sortino ratio.
    current_sortino: f64,
}

impl SortinoRewardShaper {
    /// Create a new Sortino reward shaper.
    pub fn new(config: RewardShaperConfig) -> Self {
        let stats = RollingStats::new(config.window_size);
        Self {
            config,
            stats,
            current_sortino: 0.0,
        }
    }

    /// Compute the Sortino ratio from current statistics.
    fn compute_sortino(&self) -> f64 {
        if !self.stats.is_ready() {
            return 0.0;
        }

        let mean_return = self.stats.mean();
        let downside_dev = self.stats.downside_deviation().max(self.config.min_std);

        // Convert risk-free rate to per-period
        let rf_period = self.config.risk_free_rate / self.config.annualization_factor;

        // Compute Sortino ratio
        let sortino = (mean_return - rf_period) / downside_dev;

        // Annualize
        sortino * self.config.annualization_factor.sqrt()
    }
}

impl RewardShaper for SortinoRewardShaper {
    fn shape_reward(&mut self, reward: f64) -> f64 {
        self.stats.push(reward);
        self.current_sortino = self.compute_sortino();

        // Sortino-style shaping: only penalize downside
        let shaped = if self.stats.is_ready() {
            let downside_dev = self.stats.downside_deviation().max(self.config.min_std);
            if reward >= 0.0 {
                reward // Don't penalize positive returns
            } else {
                reward / downside_dev // Risk-adjusted negative return
            }
        } else {
            reward
        };

        // Apply blend and scale
        let blended = reward * (1.0 - self.config.blend_factor)
            + shaped * self.config.blend_factor;

        blended * self.config.scale
    }

    fn risk_metric(&self) -> f64 {
        self.current_sortino
    }

    fn reset(&mut self) {
        self.stats.reset();
        self.current_sortino = 0.0;
    }

    fn metric_name(&self) -> &str {
        "sortino_ratio"
    }
}

/// Calmar ratio reward shaper.
///
/// Adjusts rewards based on return relative to maximum drawdown.
/// Calmar = Annual Return / Max Drawdown
pub struct CalmarRewardShaper {
    /// Configuration.
    config: RewardShaperConfig,
    /// Rolling statistics for returns.
    stats: RollingStats,
    /// Peak cumulative return.
    peak_cum_return: f64,
    /// Current cumulative return.
    cum_return: f64,
    /// Maximum drawdown.
    max_drawdown: f64,
    /// Current Calmar ratio.
    current_calmar: f64,
}

impl CalmarRewardShaper {
    /// Create a new Calmar reward shaper.
    pub fn new(config: RewardShaperConfig) -> Self {
        let stats = RollingStats::new(config.window_size);
        Self {
            config,
            stats,
            peak_cum_return: 0.0,
            cum_return: 0.0,
            max_drawdown: 0.0,
            current_calmar: 0.0,
        }
    }

    /// Compute the Calmar ratio from current statistics.
    fn compute_calmar(&self) -> f64 {
        if self.max_drawdown < self.config.min_std {
            return 0.0;
        }

        let annual_return = self.stats.mean() * self.config.annualization_factor;
        annual_return / self.max_drawdown
    }

    /// Update drawdown tracking.
    fn update_drawdown(&mut self, reward: f64) {
        self.cum_return += reward;
        self.peak_cum_return = self.peak_cum_return.max(self.cum_return);

        if self.peak_cum_return > 0.0 {
            let current_dd = (self.peak_cum_return - self.cum_return) / self.peak_cum_return;
            self.max_drawdown = self.max_drawdown.max(current_dd);
        }
    }

    /// Get current drawdown.
    pub fn current_drawdown(&self) -> f64 {
        if self.peak_cum_return > 0.0 {
            (self.peak_cum_return - self.cum_return) / self.peak_cum_return
        } else {
            0.0
        }
    }

    /// Get maximum drawdown.
    pub fn max_drawdown(&self) -> f64 {
        self.max_drawdown
    }
}

impl RewardShaper for CalmarRewardShaper {
    fn shape_reward(&mut self, reward: f64) -> f64 {
        self.stats.push(reward);
        self.update_drawdown(reward);
        self.current_calmar = self.compute_calmar();

        // Calmar-style shaping: penalize rewards that increase drawdown risk
        let shaped = if self.max_drawdown > self.config.min_std {
            // Reduce reward proportionally to drawdown risk
            let dd_penalty = 1.0 - self.current_drawdown();
            reward * dd_penalty
        } else {
            reward
        };

        // Apply blend and scale
        let blended = reward * (1.0 - self.config.blend_factor)
            + shaped * self.config.blend_factor;

        blended * self.config.scale
    }

    fn risk_metric(&self) -> f64 {
        self.current_calmar
    }

    fn reset(&mut self) {
        self.stats.reset();
        self.peak_cum_return = 0.0;
        self.cum_return = 0.0;
        self.max_drawdown = 0.0;
        self.current_calmar = 0.0;
    }

    fn metric_name(&self) -> &str {
        "calmar_ratio"
    }
}

/// Risk parity reward shaper.
///
/// Scales rewards to target a specific volatility level, encouraging
/// consistent risk-taking.
pub struct RiskParityShaper {
    /// Configuration.
    config: RewardShaperConfig,
    /// Rolling statistics for returns.
    stats: RollingStats,
    /// Current volatility.
    current_volatility: f64,
}

impl RiskParityShaper {
    /// Create a new risk parity shaper.
    pub fn new(config: RewardShaperConfig) -> Self {
        let stats = RollingStats::new(config.window_size);
        Self {
            config,
            stats,
            current_volatility: 0.0,
        }
    }

    /// Compute current annualized volatility.
    fn compute_volatility(&self) -> f64 {
        if !self.stats.is_ready() {
            return 0.0;
        }
        self.stats.std_dev() * self.config.annualization_factor.sqrt()
    }

    /// Get the volatility scaling factor.
    pub fn vol_scale(&self) -> f64 {
        if self.current_volatility < self.config.min_std {
            return 1.0;
        }
        self.config.target_volatility / self.current_volatility
    }
}

impl RewardShaper for RiskParityShaper {
    fn shape_reward(&mut self, reward: f64) -> f64 {
        self.stats.push(reward);
        self.current_volatility = self.compute_volatility();

        // Scale reward to target volatility
        let vol_scale = self.vol_scale();
        let shaped = reward * vol_scale;

        // Apply blend and scale
        let blended = reward * (1.0 - self.config.blend_factor)
            + shaped * self.config.blend_factor;

        blended * self.config.scale
    }

    fn risk_metric(&self) -> f64 {
        self.current_volatility
    }

    fn reset(&mut self) {
        self.stats.reset();
        self.current_volatility = 0.0;
    }

    fn metric_name(&self) -> &str {
        "volatility"
    }
}

/// Composite reward shaper that combines multiple shapers.
pub struct CompositeRewardShaper {
    /// List of shapers with their weights.
    shapers: Vec<(Box<dyn RewardShaper>, f64)>,
}

impl CompositeRewardShaper {
    /// Create a new composite shaper.
    pub fn new() -> Self {
        Self {
            shapers: Vec::new(),
        }
    }

    /// Add a shaper with a weight.
    pub fn add_shaper(&mut self, shaper: Box<dyn RewardShaper>, weight: f64) {
        self.shapers.push((shaper, weight));
    }
}

impl Default for CompositeRewardShaper {
    fn default() -> Self {
        Self::new()
    }
}

impl RewardShaper for CompositeRewardShaper {
    fn shape_reward(&mut self, reward: f64) -> f64 {
        if self.shapers.is_empty() {
            return reward;
        }

        let total_weight: f64 = self.shapers.iter().map(|(_, w)| w).sum();
        if total_weight < 1e-10 {
            return reward;
        }

        let weighted_sum: f64 = self
            .shapers
            .iter_mut()
            .map(|(shaper, weight)| shaper.shape_reward(reward) * *weight)
            .sum();

        weighted_sum / total_weight
    }

    fn risk_metric(&self) -> f64 {
        // Return weighted average of risk metrics
        if self.shapers.is_empty() {
            return 0.0;
        }

        let total_weight: f64 = self.shapers.iter().map(|(_, w)| w).sum();
        if total_weight < 1e-10 {
            return 0.0;
        }

        let weighted_sum: f64 = self
            .shapers
            .iter()
            .map(|(shaper, weight)| shaper.risk_metric() * weight)
            .sum();

        weighted_sum / total_weight
    }

    fn reset(&mut self) {
        for (shaper, _) in &mut self.shapers {
            shaper.reset();
        }
    }

    fn metric_name(&self) -> &str {
        "composite"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rolling_stats() {
        let mut stats = RollingStats::new(5);

        stats.push(1.0);
        stats.push(2.0);
        stats.push(3.0);
        stats.push(4.0);
        stats.push(5.0);

        assert!((stats.mean() - 3.0).abs() < 1e-10);
        assert!(stats.is_ready());

        // Add more values - should roll
        stats.push(6.0);
        assert!((stats.mean() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sharpe_shaper() {
        let config = RewardShaperConfig::default()
            .window_size(10)
            .risk_free_rate(0.0);
        let mut shaper = SharpeRewardShaper::new(config);

        // Add some returns
        for _ in 0..10 {
            shaper.shape_reward(0.01);
        }

        // Should have positive Sharpe
        assert!(shaper.risk_metric() > 0.0);
    }

    #[test]
    fn test_sortino_shaper() {
        let config = RewardShaperConfig::default()
            .window_size(10)
            .risk_free_rate(0.0);
        let mut shaper = SortinoRewardShaper::new(config);

        // Add mixed returns
        for i in 0..10 {
            if i % 2 == 0 {
                shaper.shape_reward(0.02);
            } else {
                shaper.shape_reward(-0.01);
            }
        }

        // Should compute Sortino
        let sortino = shaper.risk_metric();
        assert!(sortino.is_finite());
    }

    #[test]
    fn test_calmar_shaper() {
        let config = RewardShaperConfig::default().window_size(10);
        let mut shaper = CalmarRewardShaper::new(config);

        // Add returns with a drawdown
        for _ in 0..5 {
            shaper.shape_reward(0.02);
        }
        for _ in 0..3 {
            shaper.shape_reward(-0.01);
        }

        assert!(shaper.max_drawdown() > 0.0);
    }

    #[test]
    fn test_risk_parity_shaper() {
        let config = RewardShaperConfig::default()
            .window_size(10)
            .target_volatility(0.15);
        let mut shaper = RiskParityShaper::new(config);

        // Add some returns
        for i in 0..10 {
            let r = if i % 2 == 0 { 0.02 } else { -0.01 };
            shaper.shape_reward(r);
        }

        // Should compute volatility
        assert!(shaper.current_volatility > 0.0);
    }

    #[test]
    fn test_composite_shaper() {
        let config = RewardShaperConfig::default().window_size(5);
        let mut composite = CompositeRewardShaper::new();

        composite.add_shaper(
            Box::new(SharpeRewardShaper::new(config.clone())),
            1.0,
        );
        composite.add_shaper(
            Box::new(SortinoRewardShaper::new(config)),
            1.0,
        );

        // Should work with composite
        for _ in 0..10 {
            let shaped = composite.shape_reward(0.01);
            assert!(shaped.is_finite());
        }
    }
}
