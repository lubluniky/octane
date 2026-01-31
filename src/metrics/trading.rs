//! Trading performance metrics for algorithmic trading strategies.
//!
//! This module provides comprehensive financial metrics including:
//! - Risk-adjusted returns (Sharpe, Sortino, Calmar, Information, Treynor ratios)
//! - Drawdown analysis (max drawdown, recovery factor, Ulcer index)
//! - Trade statistics (win rate, profit factor, expectancy)
//! - Risk metrics (VaR, CVaR)
//!
//! All metrics support:
//! - Streaming/online computation for efficiency
//! - Rolling window calculations
//! - Annualization with configurable periods

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Configuration for trading metrics calculation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Trading periods per year for annualization (252 for daily, 12 for monthly, etc.).
    pub periods_per_year: f64,
    /// Risk-free rate for Sharpe ratio (annualized).
    pub risk_free_rate: f64,
    /// Benchmark returns for Information and Treynor ratios.
    pub benchmark_returns: Option<Vec<f64>>,
    /// VaR confidence level (e.g., 0.95 for 95% VaR).
    pub var_confidence: f64,
    /// Rolling window size for windowed metrics (0 = cumulative).
    pub rolling_window: usize,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            periods_per_year: 252.0, // Daily trading
            risk_free_rate: 0.02,    // 2% annual risk-free rate
            benchmark_returns: None,
            var_confidence: 0.95,
            rolling_window: 0, // Cumulative by default
        }
    }
}

impl MetricsConfig {
    /// Create a new config with specified annualization period.
    pub fn new(periods_per_year: f64) -> Self {
        Self {
            periods_per_year,
            ..Default::default()
        }
    }

    /// Set risk-free rate.
    pub fn risk_free_rate(mut self, rate: f64) -> Self {
        self.risk_free_rate = rate;
        self
    }

    /// Set benchmark returns.
    pub fn benchmark_returns(mut self, returns: Vec<f64>) -> Self {
        self.benchmark_returns = Some(returns);
        self
    }

    /// Set VaR confidence level.
    pub fn var_confidence(mut self, confidence: f64) -> Self {
        self.var_confidence = confidence;
        self
    }

    /// Set rolling window size.
    pub fn rolling_window(mut self, window: usize) -> Self {
        self.rolling_window = window;
        self
    }
}

/// Comprehensive trading metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    /// Sharpe ratio (annualized).
    pub sharpe_ratio: f64,
    /// Sortino ratio (annualized).
    pub sortino_ratio: f64,
    /// Calmar ratio (annualized return / max drawdown).
    pub calmar_ratio: f64,
    /// Maximum drawdown (absolute).
    pub max_drawdown: f64,
    /// Maximum drawdown (percentage).
    pub max_drawdown_pct: f64,
    /// Win rate (fraction of profitable trades).
    pub win_rate: f64,
    /// Profit factor (gross profit / gross loss).
    pub profit_factor: f64,
    /// Expectancy (average profit per trade).
    pub expectancy: f64,
    /// Value at Risk (historical).
    pub var_historical: f64,
    /// Value at Risk (parametric/Gaussian).
    pub var_parametric: f64,
    /// Conditional VaR (CVaR / Expected Shortfall).
    pub cvar: f64,
    /// Average win.
    pub avg_win: f64,
    /// Average loss.
    pub avg_loss: f64,
    /// Risk-reward ratio.
    pub risk_reward_ratio: f64,
    /// Recovery factor (total return / max drawdown).
    pub recovery_factor: f64,
    /// Ulcer index (downside volatility).
    pub ulcer_index: f64,
    /// Information ratio (excess return / tracking error).
    pub information_ratio: Option<f64>,
    /// Treynor ratio (excess return / beta).
    pub treynor_ratio: Option<f64>,
    /// Total return.
    pub total_return: f64,
    /// Annualized return.
    pub annualized_return: f64,
    /// Annualized volatility.
    pub annualized_volatility: f64,
    /// Number of trades.
    pub num_trades: usize,
    /// Number of winning trades.
    pub num_wins: usize,
    /// Number of losing trades.
    pub num_losses: usize,
}

impl Default for TradingMetrics {
    fn default() -> Self {
        Self {
            sharpe_ratio: 0.0,
            sortino_ratio: 0.0,
            calmar_ratio: 0.0,
            max_drawdown: 0.0,
            max_drawdown_pct: 0.0,
            win_rate: 0.0,
            profit_factor: 0.0,
            expectancy: 0.0,
            var_historical: 0.0,
            var_parametric: 0.0,
            cvar: 0.0,
            avg_win: 0.0,
            avg_loss: 0.0,
            risk_reward_ratio: 0.0,
            recovery_factor: 0.0,
            ulcer_index: 0.0,
            information_ratio: None,
            treynor_ratio: None,
            total_return: 0.0,
            annualized_return: 0.0,
            annualized_volatility: 0.0,
            num_trades: 0,
            num_wins: 0,
            num_losses: 0,
        }
    }
}

/// Online/streaming trading metrics calculator.
///
/// Computes metrics incrementally as new returns/trades arrive,
/// avoiding the need to store and reprocess all historical data.
pub struct MetricsCalculator {
    config: MetricsConfig,

    // Return statistics
    returns: VecDeque<f64>,
    returns_sum: f64,
    returns_sq_sum: f64,
    downside_returns_sq_sum: f64,
    n_returns: usize,

    // Equity curve tracking
    equity_curve: VecDeque<f64>,
    peak_equity: f64,
    current_drawdown: f64,
    max_drawdown: f64,
    drawdown_squared_sum: f64,

    // Trade statistics
    trades: VecDeque<f64>,
    gross_profit: f64,
    gross_loss: f64,
    num_wins: usize,
    num_losses: usize,

    // Benchmark tracking (optional)
    benchmark_idx: usize,
}

impl MetricsCalculator {
    /// Create a new metrics calculator.
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            returns: VecDeque::new(),
            returns_sum: 0.0,
            returns_sq_sum: 0.0,
            downside_returns_sq_sum: 0.0,
            n_returns: 0,
            equity_curve: VecDeque::new(),
            peak_equity: 0.0,
            current_drawdown: 0.0,
            max_drawdown: 0.0,
            drawdown_squared_sum: 0.0,
            trades: VecDeque::new(),
            gross_profit: 0.0,
            gross_loss: 0.0,
            num_wins: 0,
            num_losses: 0,
            benchmark_idx: 0,
            config,
        }
    }

    /// Add a new return observation.
    pub fn add_return(&mut self, ret: f64) {
        // Handle rolling window
        if self.config.rolling_window > 0 && self.returns.len() >= self.config.rolling_window {
            if let Some(old_ret) = self.returns.pop_front() {
                self.returns_sum -= old_ret;
                self.returns_sq_sum -= old_ret * old_ret;
                if old_ret < 0.0 {
                    self.downside_returns_sq_sum -= old_ret * old_ret;
                }
                self.n_returns -= 1;
            }
        }

        // Add new return
        self.returns.push_back(ret);
        self.returns_sum += ret;
        self.returns_sq_sum += ret * ret;
        if ret < 0.0 {
            self.downside_returns_sq_sum += ret * ret;
        }
        self.n_returns += 1;

        // Update equity curve
        let new_equity = if self.equity_curve.is_empty() {
            1.0 + ret
        } else {
            self.equity_curve.back().unwrap() * (1.0 + ret)
        };

        // Handle rolling window for equity curve
        if self.config.rolling_window > 0 && self.equity_curve.len() >= self.config.rolling_window {
            self.equity_curve.pop_front();
            // Recalculate drawdown stats when windowing
            self.recalculate_drawdown();
        }

        self.equity_curve.push_back(new_equity);

        // Update drawdown
        if new_equity > self.peak_equity {
            self.peak_equity = new_equity;
            self.current_drawdown = 0.0;
        } else {
            self.current_drawdown = self.peak_equity - new_equity;
            if self.current_drawdown > self.max_drawdown {
                self.max_drawdown = self.current_drawdown;
            }
        }

        // Update Ulcer Index (squared drawdown percentage)
        let dd_pct = if self.peak_equity > 0.0 {
            (self.peak_equity - new_equity) / self.peak_equity
        } else {
            0.0
        };
        self.drawdown_squared_sum += dd_pct * dd_pct;
    }

    /// Add a completed trade result.
    pub fn add_trade(&mut self, pnl: f64) {
        // Handle rolling window
        if self.config.rolling_window > 0 && self.trades.len() >= self.config.rolling_window {
            if let Some(old_trade) = self.trades.pop_front() {
                if old_trade > 0.0 {
                    self.gross_profit -= old_trade;
                    self.num_wins -= 1;
                } else {
                    self.gross_loss -= old_trade.abs();
                    self.num_losses -= 1;
                }
            }
        }

        self.trades.push_back(pnl);
        if pnl > 0.0 {
            self.gross_profit += pnl;
            self.num_wins += 1;
        } else {
            self.gross_loss += pnl.abs();
            self.num_losses += 1;
        }
    }

    /// Recalculate drawdown statistics (used when windowing).
    fn recalculate_drawdown(&mut self) {
        self.peak_equity = 0.0;
        self.max_drawdown = 0.0;
        self.current_drawdown = 0.0;
        self.drawdown_squared_sum = 0.0;

        for &equity in &self.equity_curve {
            if equity > self.peak_equity {
                self.peak_equity = equity;
                self.current_drawdown = 0.0;
            } else {
                self.current_drawdown = self.peak_equity - equity;
                if self.current_drawdown > self.max_drawdown {
                    self.max_drawdown = self.current_drawdown;
                }
            }

            let dd_pct = if self.peak_equity > 0.0 {
                (self.peak_equity - equity) / self.peak_equity
            } else {
                0.0
            };
            self.drawdown_squared_sum += dd_pct * dd_pct;
        }
    }

    /// Calculate mean return.
    fn mean_return(&self) -> f64 {
        if self.n_returns > 0 {
            self.returns_sum / self.n_returns as f64
        } else {
            0.0
        }
    }

    /// Calculate return variance.
    fn return_variance(&self) -> f64 {
        if self.n_returns > 1 {
            let mean = self.mean_return();
            (self.returns_sq_sum - self.n_returns as f64 * mean * mean) / (self.n_returns - 1) as f64
        } else {
            0.0
        }
    }

    /// Calculate return standard deviation.
    fn return_std(&self) -> f64 {
        self.return_variance().sqrt()
    }

    /// Calculate downside deviation (for Sortino ratio).
    fn downside_deviation(&self) -> f64 {
        if self.n_returns > 1 {
            (self.downside_returns_sq_sum / (self.n_returns - 1) as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Calculate Sharpe ratio (annualized).
    pub fn sharpe_ratio(&self) -> f64 {
        if self.n_returns == 0 {
            return 0.0;
        }

        let mean_ret = self.mean_return();
        let std_ret = self.return_std();

        if std_ret == 0.0 {
            return 0.0;
        }

        let risk_free_per_period = self.config.risk_free_rate / self.config.periods_per_year;
        let excess_return = mean_ret - risk_free_per_period;

        (excess_return / std_ret) * self.config.periods_per_year.sqrt()
    }

    /// Calculate Sortino ratio (annualized).
    pub fn sortino_ratio(&self) -> f64 {
        if self.n_returns == 0 {
            return 0.0;
        }

        let mean_ret = self.mean_return();
        let downside_dev = self.downside_deviation();

        if downside_dev == 0.0 {
            return 0.0;
        }

        let risk_free_per_period = self.config.risk_free_rate / self.config.periods_per_year;
        let excess_return = mean_ret - risk_free_per_period;

        (excess_return / downside_dev) * self.config.periods_per_year.sqrt()
    }

    /// Calculate Calmar ratio (annualized return / max drawdown).
    pub fn calmar_ratio(&self) -> f64 {
        if self.max_drawdown == 0.0 || self.n_returns == 0 {
            return 0.0;
        }

        let total_return = self.total_return();
        let annualized_return = (1.0 + total_return).powf(self.config.periods_per_year / self.n_returns as f64) - 1.0;

        let max_dd_pct = if self.peak_equity > 0.0 {
            self.max_drawdown / self.peak_equity
        } else {
            0.0
        };

        if max_dd_pct == 0.0 {
            0.0
        } else {
            annualized_return / max_dd_pct
        }
    }

    /// Calculate total return.
    pub fn total_return(&self) -> f64 {
        if let Some(&final_equity) = self.equity_curve.back() {
            final_equity - 1.0
        } else {
            0.0
        }
    }

    /// Calculate annualized return.
    pub fn annualized_return(&self) -> f64 {
        if self.n_returns == 0 {
            return 0.0;
        }

        let total_ret = self.total_return();
        (1.0 + total_ret).powf(self.config.periods_per_year / self.n_returns as f64) - 1.0
    }

    /// Calculate annualized volatility.
    pub fn annualized_volatility(&self) -> f64 {
        self.return_std() * self.config.periods_per_year.sqrt()
    }

    /// Calculate win rate.
    pub fn win_rate(&self) -> f64 {
        let total_trades = self.num_wins + self.num_losses;
        if total_trades > 0 {
            self.num_wins as f64 / total_trades as f64
        } else {
            0.0
        }
    }

    /// Calculate profit factor.
    pub fn profit_factor(&self) -> f64 {
        if self.gross_loss > 0.0 {
            self.gross_profit / self.gross_loss
        } else if self.gross_profit > 0.0 {
            f64::INFINITY
        } else {
            0.0
        }
    }

    /// Calculate expectancy (average profit per trade).
    pub fn expectancy(&self) -> f64 {
        if !self.trades.is_empty() {
            self.trades.iter().sum::<f64>() / self.trades.len() as f64
        } else {
            0.0
        }
    }

    /// Calculate average win.
    pub fn avg_win(&self) -> f64 {
        if self.num_wins > 0 {
            self.gross_profit / self.num_wins as f64
        } else {
            0.0
        }
    }

    /// Calculate average loss.
    pub fn avg_loss(&self) -> f64 {
        if self.num_losses > 0 {
            self.gross_loss / self.num_losses as f64
        } else {
            0.0
        }
    }

    /// Calculate risk-reward ratio.
    pub fn risk_reward_ratio(&self) -> f64 {
        let avg_loss = self.avg_loss();
        if avg_loss > 0.0 {
            self.avg_win() / avg_loss
        } else {
            0.0
        }
    }

    /// Calculate recovery factor (total return / max drawdown).
    pub fn recovery_factor(&self) -> f64 {
        let max_dd_pct = if self.peak_equity > 0.0 {
            self.max_drawdown / self.peak_equity
        } else {
            0.0
        };

        if max_dd_pct > 0.0 {
            self.total_return() / max_dd_pct
        } else {
            0.0
        }
    }

    /// Calculate Ulcer Index (downside volatility measure).
    pub fn ulcer_index(&self) -> f64 {
        if self.n_returns > 0 {
            (self.drawdown_squared_sum / self.n_returns as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Calculate Value at Risk (historical method).
    pub fn var_historical(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let mut sorted_returns: Vec<f64> = self.returns.iter().copied().collect();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let idx = ((1.0 - self.config.var_confidence) * sorted_returns.len() as f64).floor() as usize;
        -sorted_returns.get(idx).copied().unwrap_or(0.0)
    }

    /// Calculate Value at Risk (parametric/Gaussian method).
    pub fn var_parametric(&self) -> f64 {
        if self.n_returns == 0 {
            return 0.0;
        }

        let mean_ret = self.mean_return();
        let std_ret = self.return_std();

        // Z-score for confidence level (approximation)
        let z_score = match self.config.var_confidence {
            x if x >= 0.99 => 2.326,
            x if x >= 0.95 => 1.645,
            x if x >= 0.90 => 1.282,
            _ => 1.645,
        };

        -(mean_ret - z_score * std_ret)
    }

    /// Calculate Conditional VaR (CVaR / Expected Shortfall).
    pub fn cvar(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let mut sorted_returns: Vec<f64> = self.returns.iter().copied().collect();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let cutoff_idx = ((1.0 - self.config.var_confidence) * sorted_returns.len() as f64).ceil() as usize;

        if cutoff_idx == 0 {
            return 0.0;
        }

        let tail_sum: f64 = sorted_returns.iter().take(cutoff_idx).sum();
        -tail_sum / cutoff_idx as f64
    }

    /// Calculate Information Ratio (requires benchmark returns).
    pub fn information_ratio(&self) -> Option<f64> {
        let benchmark = self.config.benchmark_returns.as_ref()?;

        if self.returns.len() != benchmark.len() || self.returns.is_empty() {
            return None;
        }

        let mean_ret = self.mean_return();
        let mean_bench: f64 = benchmark.iter().sum::<f64>() / benchmark.len() as f64;

        let excess_return = mean_ret - mean_bench;

        // Calculate tracking error
        let tracking_errors: Vec<f64> = self.returns.iter()
            .zip(benchmark.iter())
            .map(|(r, b)| r - b)
            .collect();

        let te_mean: f64 = tracking_errors.iter().sum::<f64>() / tracking_errors.len() as f64;
        let te_var: f64 = tracking_errors.iter()
            .map(|te| (te - te_mean).powi(2))
            .sum::<f64>() / (tracking_errors.len() - 1) as f64;

        let tracking_error = te_var.sqrt();

        if tracking_error > 0.0 {
            Some((excess_return / tracking_error) * self.config.periods_per_year.sqrt())
        } else {
            None
        }
    }

    /// Calculate Treynor Ratio (requires benchmark returns for beta calculation).
    pub fn treynor_ratio(&self) -> Option<f64> {
        let benchmark = self.config.benchmark_returns.as_ref()?;

        if self.returns.len() != benchmark.len() || self.returns.is_empty() {
            return None;
        }

        // Calculate beta
        let mean_ret = self.mean_return();
        let mean_bench: f64 = benchmark.iter().sum::<f64>() / benchmark.len() as f64;

        let covariance: f64 = self.returns.iter()
            .zip(benchmark.iter())
            .map(|(r, b)| (r - mean_ret) * (b - mean_bench))
            .sum::<f64>() / (self.returns.len() - 1) as f64;

        let bench_var: f64 = benchmark.iter()
            .map(|b| (b - mean_bench).powi(2))
            .sum::<f64>() / (benchmark.len() - 1) as f64;

        if bench_var == 0.0 {
            return None;
        }

        let beta = covariance / bench_var;

        if beta == 0.0 {
            return None;
        }

        let risk_free_per_period = self.config.risk_free_rate / self.config.periods_per_year;
        let excess_return = mean_ret - risk_free_per_period;

        Some((excess_return * self.config.periods_per_year) / beta)
    }

    /// Get comprehensive metrics snapshot.
    pub fn compute_metrics(&self) -> TradingMetrics {
        let max_dd_pct = if self.peak_equity > 0.0 {
            self.max_drawdown / self.peak_equity
        } else {
            0.0
        };

        TradingMetrics {
            sharpe_ratio: self.sharpe_ratio(),
            sortino_ratio: self.sortino_ratio(),
            calmar_ratio: self.calmar_ratio(),
            max_drawdown: self.max_drawdown,
            max_drawdown_pct: max_dd_pct,
            win_rate: self.win_rate(),
            profit_factor: self.profit_factor(),
            expectancy: self.expectancy(),
            var_historical: self.var_historical(),
            var_parametric: self.var_parametric(),
            cvar: self.cvar(),
            avg_win: self.avg_win(),
            avg_loss: self.avg_loss(),
            risk_reward_ratio: self.risk_reward_ratio(),
            recovery_factor: self.recovery_factor(),
            ulcer_index: self.ulcer_index(),
            information_ratio: self.information_ratio(),
            treynor_ratio: self.treynor_ratio(),
            total_return: self.total_return(),
            annualized_return: self.annualized_return(),
            annualized_volatility: self.annualized_volatility(),
            num_trades: self.trades.len(),
            num_wins: self.num_wins,
            num_losses: self.num_losses,
        }
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.returns.clear();
        self.returns_sum = 0.0;
        self.returns_sq_sum = 0.0;
        self.downside_returns_sq_sum = 0.0;
        self.n_returns = 0;
        self.equity_curve.clear();
        self.peak_equity = 0.0;
        self.current_drawdown = 0.0;
        self.max_drawdown = 0.0;
        self.drawdown_squared_sum = 0.0;
        self.trades.clear();
        self.gross_profit = 0.0;
        self.gross_loss = 0.0;
        self.num_wins = 0;
        self.num_losses = 0;
        self.benchmark_idx = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_calculator_basic() {
        let config = MetricsConfig::default();
        let mut calc = MetricsCalculator::new(config);

        // Add some returns
        calc.add_return(0.01); // +1%
        calc.add_return(0.02); // +2%
        calc.add_return(-0.01); // -1%
        calc.add_return(0.015); // +1.5%

        assert_eq!(calc.n_returns, 4);
        assert!(calc.mean_return() > 0.0);
        assert!(calc.sharpe_ratio() != 0.0);
    }

    #[test]
    fn test_trade_statistics() {
        let config = MetricsConfig::default();
        let mut calc = MetricsCalculator::new(config);

        calc.add_trade(100.0); // Win
        calc.add_trade(-50.0); // Loss
        calc.add_trade(150.0); // Win
        calc.add_trade(-30.0); // Loss

        assert_eq!(calc.num_wins, 2);
        assert_eq!(calc.num_losses, 2);
        assert_eq!(calc.win_rate(), 0.5);
        assert_eq!(calc.gross_profit, 250.0);
        assert_eq!(calc.gross_loss, 80.0);
        assert!(calc.profit_factor() > 3.0);
    }

    #[test]
    fn test_drawdown_tracking() {
        let config = MetricsConfig::default();
        let mut calc = MetricsCalculator::new(config);

        calc.add_return(0.10); // Up 10%
        calc.add_return(0.05); // Up 5%
        calc.add_return(-0.15); // Down 15%
        calc.add_return(0.05); // Up 5%

        assert!(calc.max_drawdown > 0.0);
        let metrics = calc.compute_metrics();
        assert!(metrics.max_drawdown_pct > 0.0);
        assert!(metrics.max_drawdown_pct < 1.0);
    }

    #[test]
    fn test_rolling_window() {
        let config = MetricsConfig::default().rolling_window(3);
        let mut calc = MetricsCalculator::new(config);

        calc.add_return(0.01);
        calc.add_return(0.02);
        calc.add_return(0.03);
        assert_eq!(calc.returns.len(), 3);

        calc.add_return(0.04);
        assert_eq!(calc.returns.len(), 3); // Should still be 3
        assert_eq!(calc.returns[0], 0.02); // First element removed
    }

    #[test]
    fn test_var_cvar() {
        let config = MetricsConfig::default().var_confidence(0.95);
        let mut calc = MetricsCalculator::new(config);

        // Add returns with some tail risk
        // 10 negative returns, 90 positive returns
        for i in 0..100 {
            let ret = if i < 10 { -0.05 } else { 0.01 };
            calc.add_return(ret);
        }

        let var = calc.var_historical();
        let cvar = calc.cvar();

        assert!(var > 0.0, "VaR should be positive, got {}", var);
        // Use epsilon for floating point comparison
        assert!(cvar >= var - 1e-10, "CVaR ({}) should be >= VaR ({})", cvar, var);
    }
}
