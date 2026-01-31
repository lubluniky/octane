//! Walk-Forward Optimization for RL trading strategies.
//!
//! Walk-forward optimization is a technique used to validate trading strategies
//! by dividing historical data into multiple in-sample/out-of-sample segments.
//! The strategy is optimized on each in-sample period and then tested on the
//! subsequent out-of-sample period.
//!
//! # Features
//!
//! - Rolling and anchored walk-forward methods
//! - Configurable window sizes and step sizes
//! - Multiple optimization objectives
//! - Overfitting detection via in-sample vs out-of-sample comparison
//! - Parallel execution of optimization runs
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::backtesting::{WalkForwardConfig, WalkForwardOptimizer, WalkForwardObjective};
//!
//! let config = WalkForwardConfig::new()
//!     .in_sample_size(252)      // 1 year in-sample
//!     .out_of_sample_size(63)   // 3 months out-of-sample
//!     .step_size(63)            // Step forward by 3 months
//!     .anchored(false)          // Rolling window
//!     .objective(WalkForwardObjective::SharpeRatio);
//!
//! let optimizer = WalkForwardOptimizer::new(config, data);
//! let result = optimizer.run(|params, train_data| {
//!     // Train strategy with params on train_data
//!     // Return trained model
//! }, |model, test_data| {
//!     // Evaluate model on test_data
//!     // Return performance metrics
//! })?;
//!
//! println!("Aggregated OOS Sharpe: {:.3}", result.aggregated_sharpe());
//! println!("Overfitting ratio: {:.3}", result.overfitting_ratio());
//! ```

use crate::core::{OctaneError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Optimization objective for walk-forward analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WalkForwardObjective {
    /// Maximize Sharpe ratio.
    #[default]
    SharpeRatio,
    /// Maximize Sortino ratio.
    SortinoRatio,
    /// Maximize total return.
    TotalReturn,
    /// Maximize Calmar ratio (return / max drawdown).
    CalmarRatio,
    /// Minimize maximum drawdown.
    MinDrawdown,
    /// Maximize risk-adjusted return (return / volatility).
    RiskAdjustedReturn,
    /// Maximize profit factor.
    ProfitFactor,
    /// Custom objective (user-defined).
    Custom,
}

/// Configuration for walk-forward optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardConfig {
    /// Number of periods for in-sample (training) data.
    pub in_sample_size: usize,

    /// Number of periods for out-of-sample (testing) data.
    pub out_of_sample_size: usize,

    /// Step size for rolling forward (in periods).
    /// If None, defaults to out_of_sample_size.
    pub step_size: Option<usize>,

    /// Whether to use anchored walk-forward (fixed start) vs rolling.
    pub anchored: bool,

    /// Minimum number of splits required.
    pub min_splits: usize,

    /// Maximum number of splits (optional).
    pub max_splits: Option<usize>,

    /// Optimization objective.
    pub objective: WalkForwardObjective,

    /// Whether to run optimization in parallel.
    pub parallel: bool,

    /// Number of threads for parallel execution.
    pub n_jobs: Option<usize>,

    /// Minimum out-of-sample performance threshold (prune poor params).
    pub min_oos_performance: Option<f64>,

    /// Maximum acceptable overfitting ratio (IS/OOS degradation).
    pub max_overfitting_ratio: Option<f64>,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Warmup period at the start of each in-sample window.
    pub warmup_periods: usize,

    /// Gap between in-sample and out-of-sample (prevent leakage).
    pub gap_periods: usize,
}

impl Default for WalkForwardConfig {
    fn default() -> Self {
        Self {
            in_sample_size: 252,        // 1 year of daily data
            out_of_sample_size: 63,     // 3 months
            step_size: None,            // Defaults to out_of_sample_size
            anchored: false,            // Rolling window by default
            min_splits: 3,              // At least 3 OOS periods
            max_splits: None,
            objective: WalkForwardObjective::SharpeRatio,
            parallel: true,
            n_jobs: None,
            min_oos_performance: None,
            max_overfitting_ratio: None,
            seed: None,
            warmup_periods: 0,
            gap_periods: 0,
        }
    }
}

impl WalkForwardConfig {
    /// Create a new walk-forward configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the in-sample (training) window size.
    pub fn in_sample_size(mut self, size: usize) -> Self {
        self.in_sample_size = size;
        self
    }

    /// Set the out-of-sample (testing) window size.
    pub fn out_of_sample_size(mut self, size: usize) -> Self {
        self.out_of_sample_size = size;
        self
    }

    /// Set the step size for rolling forward.
    pub fn step_size(mut self, size: usize) -> Self {
        self.step_size = Some(size);
        self
    }

    /// Set whether to use anchored walk-forward.
    pub fn anchored(mut self, anchored: bool) -> Self {
        self.anchored = anchored;
        self
    }

    /// Set the minimum number of splits.
    pub fn min_splits(mut self, n: usize) -> Self {
        self.min_splits = n;
        self
    }

    /// Set the maximum number of splits.
    pub fn max_splits(mut self, n: usize) -> Self {
        self.max_splits = Some(n);
        self
    }

    /// Set the optimization objective.
    pub fn objective(mut self, obj: WalkForwardObjective) -> Self {
        self.objective = obj;
        self
    }

    /// Enable or disable parallel execution.
    pub fn parallel(mut self, enabled: bool) -> Self {
        self.parallel = enabled;
        self
    }

    /// Set the number of parallel jobs.
    pub fn n_jobs(mut self, n: usize) -> Self {
        self.n_jobs = Some(n);
        self
    }

    /// Set minimum out-of-sample performance threshold.
    pub fn min_oos_performance(mut self, threshold: f64) -> Self {
        self.min_oos_performance = Some(threshold);
        self
    }

    /// Set maximum acceptable overfitting ratio.
    pub fn max_overfitting_ratio(mut self, ratio: f64) -> Self {
        self.max_overfitting_ratio = Some(ratio);
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set warmup periods at the start of in-sample.
    pub fn warmup_periods(mut self, periods: usize) -> Self {
        self.warmup_periods = periods;
        self
    }

    /// Set gap periods between in-sample and out-of-sample.
    pub fn gap_periods(mut self, periods: usize) -> Self {
        self.gap_periods = periods;
        self
    }

    /// Get the effective step size.
    pub fn effective_step_size(&self) -> usize {
        self.step_size.unwrap_or(self.out_of_sample_size)
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.in_sample_size == 0 {
            return Err(OctaneError::InvalidConfig(
                "in_sample_size must be positive".into(),
            ));
        }
        if self.out_of_sample_size == 0 {
            return Err(OctaneError::InvalidConfig(
                "out_of_sample_size must be positive".into(),
            ));
        }
        if self.min_splits == 0 {
            return Err(OctaneError::InvalidConfig(
                "min_splits must be positive".into(),
            ));
        }
        if let Some(max) = self.max_splits {
            if max < self.min_splits {
                return Err(OctaneError::InvalidConfig(
                    "max_splits must be >= min_splits".into(),
                ));
            }
        }
        if let Some(ratio) = self.max_overfitting_ratio {
            if ratio <= 0.0 {
                return Err(OctaneError::InvalidConfig(
                    "max_overfitting_ratio must be positive".into(),
                ));
            }
        }
        Ok(())
    }

    /// Calculate the number of splits for a given data length.
    pub fn calculate_splits(&self, data_length: usize) -> Result<usize> {
        self.validate()?;

        let step = self.effective_step_size();
        let min_required = self.in_sample_size + self.out_of_sample_size + self.gap_periods;

        if data_length < min_required {
            return Err(OctaneError::InvalidConfig(format!(
                "Data length {} too short for walk-forward (need at least {})",
                data_length, min_required
            )));
        }

        let mut n_splits = 0;
        let mut current_end = self.in_sample_size + self.gap_periods + self.out_of_sample_size;

        while current_end <= data_length {
            n_splits += 1;
            current_end += step;

            if let Some(max) = self.max_splits {
                if n_splits >= max {
                    break;
                }
            }
        }

        if n_splits < self.min_splits {
            return Err(OctaneError::InvalidConfig(format!(
                "Data length {} only allows {} splits (need at least {})",
                data_length, n_splits, self.min_splits
            )));
        }

        Ok(n_splits)
    }
}

/// A single split in walk-forward optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardSplit {
    /// Split index (0-based).
    pub index: usize,

    /// Start index of in-sample data.
    pub in_sample_start: usize,

    /// End index of in-sample data (exclusive).
    pub in_sample_end: usize,

    /// Start index of out-of-sample data.
    pub out_of_sample_start: usize,

    /// End index of out-of-sample data (exclusive).
    pub out_of_sample_end: usize,
}

impl WalkForwardSplit {
    /// Get the in-sample data range.
    pub fn in_sample_range(&self) -> std::ops::Range<usize> {
        self.in_sample_start..self.in_sample_end
    }

    /// Get the out-of-sample data range.
    pub fn out_of_sample_range(&self) -> std::ops::Range<usize> {
        self.out_of_sample_start..self.out_of_sample_end
    }

    /// Get the in-sample size.
    pub fn in_sample_size(&self) -> usize {
        self.in_sample_end - self.in_sample_start
    }

    /// Get the out-of-sample size.
    pub fn out_of_sample_size(&self) -> usize {
        self.out_of_sample_end - self.out_of_sample_start
    }
}

/// Performance metrics for a single split.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitPerformance {
    /// Split index.
    pub split_index: usize,

    /// In-sample total return.
    pub in_sample_return: f64,

    /// Out-of-sample total return.
    pub out_of_sample_return: f64,

    /// In-sample Sharpe ratio.
    pub in_sample_sharpe: f64,

    /// Out-of-sample Sharpe ratio.
    pub out_of_sample_sharpe: f64,

    /// In-sample maximum drawdown.
    pub in_sample_max_drawdown: f64,

    /// Out-of-sample maximum drawdown.
    pub out_of_sample_max_drawdown: f64,

    /// In-sample volatility.
    pub in_sample_volatility: f64,

    /// Out-of-sample volatility.
    pub out_of_sample_volatility: f64,

    /// In-sample Sortino ratio.
    pub in_sample_sortino: f64,

    /// Out-of-sample Sortino ratio.
    pub out_of_sample_sortino: f64,

    /// In-sample profit factor.
    pub in_sample_profit_factor: f64,

    /// Out-of-sample profit factor.
    pub out_of_sample_profit_factor: f64,

    /// Number of trades in-sample.
    pub in_sample_trades: usize,

    /// Number of trades out-of-sample.
    pub out_of_sample_trades: usize,

    /// Optimization objective value (in-sample).
    pub in_sample_objective: f64,

    /// Optimization objective value (out-of-sample).
    pub out_of_sample_objective: f64,

    /// Best parameters found during optimization.
    pub best_params: HashMap<String, f64>,

    /// Additional custom metrics.
    pub custom_metrics: HashMap<String, f64>,
}

impl SplitPerformance {
    /// Create a new split performance record.
    pub fn new(split_index: usize) -> Self {
        Self {
            split_index,
            ..Default::default()
        }
    }

    /// Calculate the overfitting ratio for this split.
    /// Values > 1 indicate degradation in OOS vs IS.
    pub fn overfitting_ratio(&self) -> f64 {
        if self.out_of_sample_objective.abs() < 1e-10 {
            if self.in_sample_objective.abs() < 1e-10 {
                1.0
            } else {
                f64::INFINITY
            }
        } else {
            self.in_sample_objective / self.out_of_sample_objective
        }
    }

    /// Calculate the degradation percentage.
    pub fn degradation_pct(&self) -> f64 {
        if self.in_sample_objective.abs() < 1e-10 {
            0.0
        } else {
            (1.0 - self.out_of_sample_objective / self.in_sample_objective) * 100.0
        }
    }

    /// Check if this split shows signs of overfitting.
    pub fn is_overfit(&self, threshold: f64) -> bool {
        self.overfitting_ratio() > threshold
    }
}

/// Complete result from walk-forward optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardResult {
    /// Configuration used.
    pub config: WalkForwardConfig,

    /// Number of splits.
    pub n_splits: usize,

    /// Results for each split.
    pub split_results: Vec<SplitPerformance>,

    /// Total data length.
    pub data_length: usize,

    /// Execution time in seconds.
    pub execution_time_secs: f64,

    /// Timestamp of when analysis was run.
    pub timestamp: u64,
}

impl WalkForwardResult {
    /// Create a new walk-forward result.
    pub fn new(config: WalkForwardConfig, n_splits: usize, data_length: usize) -> Self {
        Self {
            config,
            n_splits,
            split_results: Vec::with_capacity(n_splits),
            data_length,
            execution_time_secs: 0.0,
            timestamp: current_timestamp(),
        }
    }

    /// Add a split result.
    pub fn add_split(&mut self, performance: SplitPerformance) {
        self.split_results.push(performance);
    }

    /// Get aggregated out-of-sample Sharpe ratio.
    pub fn aggregated_sharpe(&self) -> f64 {
        if self.split_results.is_empty() {
            return 0.0;
        }
        self.split_results
            .iter()
            .map(|s| s.out_of_sample_sharpe)
            .sum::<f64>()
            / self.split_results.len() as f64
    }

    /// Get aggregated out-of-sample return.
    pub fn aggregated_return(&self) -> f64 {
        if self.split_results.is_empty() {
            return 0.0;
        }
        // Compound the returns
        self.split_results
            .iter()
            .map(|s| 1.0 + s.out_of_sample_return)
            .product::<f64>()
            - 1.0
    }

    /// Get aggregated out-of-sample maximum drawdown.
    pub fn aggregated_max_drawdown(&self) -> f64 {
        if self.split_results.is_empty() {
            return 0.0;
        }
        self.split_results
            .iter()
            .map(|s| s.out_of_sample_max_drawdown)
            .fold(f64::NEG_INFINITY, f64::max)
    }

    /// Calculate the average overfitting ratio across all splits.
    pub fn overfitting_ratio(&self) -> f64 {
        if self.split_results.is_empty() {
            return 1.0;
        }
        let ratios: Vec<f64> = self
            .split_results
            .iter()
            .map(|s| s.overfitting_ratio())
            .filter(|&r| r.is_finite())
            .collect();

        if ratios.is_empty() {
            1.0
        } else {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        }
    }

    /// Calculate the median overfitting ratio (more robust to outliers).
    pub fn median_overfitting_ratio(&self) -> f64 {
        if self.split_results.is_empty() {
            return 1.0;
        }
        let mut ratios: Vec<f64> = self
            .split_results
            .iter()
            .map(|s| s.overfitting_ratio())
            .filter(|&r| r.is_finite())
            .collect();

        if ratios.is_empty() {
            return 1.0;
        }

        ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = ratios.len() / 2;
        if ratios.len() % 2 == 0 {
            (ratios[mid - 1] + ratios[mid]) / 2.0
        } else {
            ratios[mid]
        }
    }

    /// Get the percentage of splits that show overfitting.
    pub fn overfitting_pct(&self, threshold: f64) -> f64 {
        if self.split_results.is_empty() {
            return 0.0;
        }
        let overfit_count = self
            .split_results
            .iter()
            .filter(|s| s.is_overfit(threshold))
            .count();
        overfit_count as f64 / self.split_results.len() as f64 * 100.0
    }

    /// Get the best split by OOS performance.
    pub fn best_split(&self) -> Option<&SplitPerformance> {
        self.split_results
            .iter()
            .max_by(|a, b| {
                a.out_of_sample_objective
                    .partial_cmp(&b.out_of_sample_objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Get the worst split by OOS performance.
    pub fn worst_split(&self) -> Option<&SplitPerformance> {
        self.split_results
            .iter()
            .min_by(|a, b| {
                a.out_of_sample_objective
                    .partial_cmp(&b.out_of_sample_objective)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    /// Check if the strategy passes walk-forward validation.
    pub fn is_valid(&self) -> bool {
        // Check minimum OOS performance if configured
        if let Some(min_perf) = self.config.min_oos_performance {
            let avg_oos = self
                .split_results
                .iter()
                .map(|s| s.out_of_sample_objective)
                .sum::<f64>()
                / self.split_results.len() as f64;
            if avg_oos < min_perf {
                return false;
            }
        }

        // Check overfitting ratio if configured
        if let Some(max_ratio) = self.config.max_overfitting_ratio {
            if self.overfitting_ratio() > max_ratio {
                return false;
            }
        }

        // Ensure positive OOS performance in majority of splits
        let positive_oos = self
            .split_results
            .iter()
            .filter(|s| s.out_of_sample_objective > 0.0)
            .count();
        positive_oos > self.split_results.len() / 2
    }

    /// Get a summary of the walk-forward analysis.
    pub fn summary(&self) -> WalkForwardSummary {
        let oos_returns: Vec<f64> = self
            .split_results
            .iter()
            .map(|s| s.out_of_sample_return)
            .collect();
        let oos_sharpes: Vec<f64> = self
            .split_results
            .iter()
            .map(|s| s.out_of_sample_sharpe)
            .collect();

        WalkForwardSummary {
            n_splits: self.n_splits,
            aggregated_return: self.aggregated_return(),
            aggregated_sharpe: self.aggregated_sharpe(),
            aggregated_max_drawdown: self.aggregated_max_drawdown(),
            overfitting_ratio: self.overfitting_ratio(),
            median_overfitting_ratio: self.median_overfitting_ratio(),
            oos_return_mean: mean(&oos_returns),
            oos_return_std: std_dev(&oos_returns),
            oos_sharpe_mean: mean(&oos_sharpes),
            oos_sharpe_std: std_dev(&oos_sharpes),
            positive_oos_pct: self
                .split_results
                .iter()
                .filter(|s| s.out_of_sample_return > 0.0)
                .count() as f64
                / self.split_results.len() as f64
                * 100.0,
            is_valid: self.is_valid(),
        }
    }
}

/// Summary statistics from walk-forward optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalkForwardSummary {
    /// Number of splits.
    pub n_splits: usize,
    /// Aggregated OOS return.
    pub aggregated_return: f64,
    /// Aggregated OOS Sharpe ratio.
    pub aggregated_sharpe: f64,
    /// Aggregated OOS max drawdown.
    pub aggregated_max_drawdown: f64,
    /// Average overfitting ratio.
    pub overfitting_ratio: f64,
    /// Median overfitting ratio.
    pub median_overfitting_ratio: f64,
    /// Mean OOS return.
    pub oos_return_mean: f64,
    /// Standard deviation of OOS returns.
    pub oos_return_std: f64,
    /// Mean OOS Sharpe ratio.
    pub oos_sharpe_mean: f64,
    /// Standard deviation of OOS Sharpe ratios.
    pub oos_sharpe_std: f64,
    /// Percentage of splits with positive OOS returns.
    pub positive_oos_pct: f64,
    /// Whether the strategy passes validation.
    pub is_valid: bool,
}

impl fmt::Display for WalkForwardSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Walk-Forward Optimization Summary")?;
        writeln!(f, "=================================")?;
        writeln!(f, "Splits: {}", self.n_splits)?;
        writeln!(f, "Aggregated Return: {:.2}%", self.aggregated_return * 100.0)?;
        writeln!(f, "Aggregated Sharpe: {:.3}", self.aggregated_sharpe)?;
        writeln!(
            f,
            "Aggregated Max DD: {:.2}%",
            self.aggregated_max_drawdown * 100.0
        )?;
        writeln!(f, "Overfitting Ratio: {:.3}", self.overfitting_ratio)?;
        writeln!(
            f,
            "Median Overfitting: {:.3}",
            self.median_overfitting_ratio
        )?;
        writeln!(
            f,
            "OOS Return: {:.2}% +/- {:.2}%",
            self.oos_return_mean * 100.0,
            self.oos_return_std * 100.0
        )?;
        writeln!(
            f,
            "OOS Sharpe: {:.3} +/- {:.3}",
            self.oos_sharpe_mean, self.oos_sharpe_std
        )?;
        writeln!(f, "Positive OOS %: {:.1}%", self.positive_oos_pct)?;
        writeln!(
            f,
            "Validation: {}",
            if self.is_valid { "PASSED" } else { "FAILED" }
        )
    }
}

/// Walk-forward optimizer for trading strategy validation.
pub struct WalkForwardOptimizer {
    /// Configuration.
    config: WalkForwardConfig,

    /// Data length.
    data_length: usize,

    /// Computed splits.
    splits: Vec<WalkForwardSplit>,
}

impl WalkForwardOptimizer {
    /// Create a new walk-forward optimizer.
    pub fn new(config: WalkForwardConfig, data_length: usize) -> Result<Self> {
        config.validate()?;
        let n_splits = config.calculate_splits(data_length)?;
        let splits = Self::compute_splits(&config, data_length, n_splits);

        Ok(Self {
            config,
            data_length,
            splits,
        })
    }

    /// Compute the splits based on configuration.
    fn compute_splits(
        config: &WalkForwardConfig,
        data_length: usize,
        n_splits: usize,
    ) -> Vec<WalkForwardSplit> {
        let step = config.effective_step_size();
        let mut splits = Vec::with_capacity(n_splits);

        for i in 0..n_splits {
            let (is_start, is_end) = if config.anchored {
                // Anchored: in-sample always starts at 0
                let is_end = config.in_sample_size + i * step;
                (0, is_end)
            } else {
                // Rolling: in-sample window moves forward
                let is_start = i * step;
                let is_end = is_start + config.in_sample_size;
                (is_start, is_end)
            };

            let oos_start = is_end + config.gap_periods;
            let oos_end = (oos_start + config.out_of_sample_size).min(data_length);

            splits.push(WalkForwardSplit {
                index: i,
                in_sample_start: is_start,
                in_sample_end: is_end,
                out_of_sample_start: oos_start,
                out_of_sample_end: oos_end,
            });
        }

        splits
    }

    /// Get the configuration.
    pub fn config(&self) -> &WalkForwardConfig {
        &self.config
    }

    /// Get the splits.
    pub fn splits(&self) -> &[WalkForwardSplit] {
        &self.splits
    }

    /// Get the number of splits.
    pub fn n_splits(&self) -> usize {
        self.splits.len()
    }

    /// Run walk-forward optimization with provided callbacks.
    ///
    /// # Arguments
    ///
    /// * `optimize_fn` - Function that optimizes parameters on in-sample data.
    ///   Takes (split_index, in_sample_range) and returns optimized params + IS metrics.
    /// * `evaluate_fn` - Function that evaluates parameters on out-of-sample data.
    ///   Takes (split_index, params, out_of_sample_range) and returns OOS metrics.
    ///
    /// # Returns
    ///
    /// A `WalkForwardResult` containing all split results and aggregated metrics.
    pub fn run<F, G>(
        &self,
        optimize_fn: F,
        evaluate_fn: G,
    ) -> Result<WalkForwardResult>
    where
        F: Fn(usize, std::ops::Range<usize>) -> Result<(HashMap<String, f64>, SplitMetrics)>
            + Sync
            + Send,
        G: Fn(usize, &HashMap<String, f64>, std::ops::Range<usize>) -> Result<SplitMetrics>
            + Sync
            + Send,
    {
        let start_time = std::time::Instant::now();

        let split_results: Vec<Result<SplitPerformance>> = if self.config.parallel {
            self.splits
                .par_iter()
                .map(|split| {
                    self.run_split(split, &optimize_fn, &evaluate_fn)
                })
                .collect()
        } else {
            self.splits
                .iter()
                .map(|split| {
                    self.run_split(split, &optimize_fn, &evaluate_fn)
                })
                .collect()
        };

        let mut result =
            WalkForwardResult::new(self.config.clone(), self.splits.len(), self.data_length);

        for split_result in split_results {
            result.add_split(split_result?);
        }

        result.execution_time_secs = start_time.elapsed().as_secs_f64();

        Ok(result)
    }

    /// Run a single split.
    fn run_split<F, G>(
        &self,
        split: &WalkForwardSplit,
        optimize_fn: &F,
        evaluate_fn: &G,
    ) -> Result<SplitPerformance>
    where
        F: Fn(usize, std::ops::Range<usize>) -> Result<(HashMap<String, f64>, SplitMetrics)>,
        G: Fn(usize, &HashMap<String, f64>, std::ops::Range<usize>) -> Result<SplitMetrics>,
    {
        // Run optimization on in-sample data
        let (best_params, is_metrics) =
            optimize_fn(split.index, split.in_sample_range())?;

        // Evaluate on out-of-sample data
        let oos_metrics = evaluate_fn(split.index, &best_params, split.out_of_sample_range())?;

        // Compute objective values
        let is_objective = self.compute_objective(&is_metrics);
        let oos_objective = self.compute_objective(&oos_metrics);

        let mut performance = SplitPerformance::new(split.index);
        performance.in_sample_return = is_metrics.total_return;
        performance.out_of_sample_return = oos_metrics.total_return;
        performance.in_sample_sharpe = is_metrics.sharpe_ratio;
        performance.out_of_sample_sharpe = oos_metrics.sharpe_ratio;
        performance.in_sample_max_drawdown = is_metrics.max_drawdown;
        performance.out_of_sample_max_drawdown = oos_metrics.max_drawdown;
        performance.in_sample_volatility = is_metrics.volatility;
        performance.out_of_sample_volatility = oos_metrics.volatility;
        performance.in_sample_sortino = is_metrics.sortino_ratio;
        performance.out_of_sample_sortino = oos_metrics.sortino_ratio;
        performance.in_sample_profit_factor = is_metrics.profit_factor;
        performance.out_of_sample_profit_factor = oos_metrics.profit_factor;
        performance.in_sample_trades = is_metrics.n_trades;
        performance.out_of_sample_trades = oos_metrics.n_trades;
        performance.in_sample_objective = is_objective;
        performance.out_of_sample_objective = oos_objective;
        performance.best_params = best_params;
        performance.custom_metrics = oos_metrics.custom_metrics;

        Ok(performance)
    }

    /// Compute the objective value based on configuration.
    fn compute_objective(&self, metrics: &SplitMetrics) -> f64 {
        match self.config.objective {
            WalkForwardObjective::SharpeRatio => metrics.sharpe_ratio,
            WalkForwardObjective::SortinoRatio => metrics.sortino_ratio,
            WalkForwardObjective::TotalReturn => metrics.total_return,
            WalkForwardObjective::CalmarRatio => {
                if metrics.max_drawdown.abs() < 1e-10 {
                    0.0
                } else {
                    metrics.total_return / metrics.max_drawdown.abs()
                }
            }
            WalkForwardObjective::MinDrawdown => -metrics.max_drawdown.abs(),
            WalkForwardObjective::RiskAdjustedReturn => {
                if metrics.volatility.abs() < 1e-10 {
                    0.0
                } else {
                    metrics.total_return / metrics.volatility
                }
            }
            WalkForwardObjective::ProfitFactor => metrics.profit_factor,
            WalkForwardObjective::Custom => metrics.custom_objective.unwrap_or(0.0),
        }
    }
}

/// Metrics for a single split (in-sample or out-of-sample).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SplitMetrics {
    /// Total return.
    pub total_return: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Sortino ratio.
    pub sortino_ratio: f64,
    /// Maximum drawdown.
    pub max_drawdown: f64,
    /// Volatility (annualized std of returns).
    pub volatility: f64,
    /// Profit factor.
    pub profit_factor: f64,
    /// Number of trades.
    pub n_trades: usize,
    /// Win rate.
    pub win_rate: f64,
    /// Average win.
    pub avg_win: f64,
    /// Average loss.
    pub avg_loss: f64,
    /// Custom objective value.
    pub custom_objective: Option<f64>,
    /// Additional custom metrics.
    pub custom_metrics: HashMap<String, f64>,
}

impl SplitMetrics {
    /// Create new empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set total return.
    pub fn with_total_return(mut self, ret: f64) -> Self {
        self.total_return = ret;
        self
    }

    /// Set Sharpe ratio.
    pub fn with_sharpe(mut self, sharpe: f64) -> Self {
        self.sharpe_ratio = sharpe;
        self
    }

    /// Set Sortino ratio.
    pub fn with_sortino(mut self, sortino: f64) -> Self {
        self.sortino_ratio = sortino;
        self
    }

    /// Set maximum drawdown.
    pub fn with_max_drawdown(mut self, dd: f64) -> Self {
        self.max_drawdown = dd;
        self
    }

    /// Set volatility.
    pub fn with_volatility(mut self, vol: f64) -> Self {
        self.volatility = vol;
        self
    }

    /// Set profit factor.
    pub fn with_profit_factor(mut self, pf: f64) -> Self {
        self.profit_factor = pf;
        self
    }

    /// Set number of trades.
    pub fn with_n_trades(mut self, n: usize) -> Self {
        self.n_trades = n;
        self
    }

    /// Set custom objective.
    pub fn with_custom_objective(mut self, obj: f64) -> Self {
        self.custom_objective = Some(obj);
        self
    }

    /// Add a custom metric.
    pub fn with_custom_metric(mut self, name: impl Into<String>, value: f64) -> Self {
        self.custom_metrics.insert(name.into(), value);
        self
    }
}

// Helper functions
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn std_dev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let m = mean(values);
    let variance = values.iter().map(|&x| (x - m).powi(2)).sum::<f64>() / (values.len() - 1) as f64;
    variance.sqrt()
}

fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_walk_forward_config_defaults() {
        let config = WalkForwardConfig::new();
        assert_eq!(config.in_sample_size, 252);
        assert_eq!(config.out_of_sample_size, 63);
        assert!(!config.anchored);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_walk_forward_config_builder() {
        let config = WalkForwardConfig::new()
            .in_sample_size(200)
            .out_of_sample_size(50)
            .step_size(25)
            .anchored(true)
            .min_splits(5)
            .objective(WalkForwardObjective::CalmarRatio);

        assert_eq!(config.in_sample_size, 200);
        assert_eq!(config.out_of_sample_size, 50);
        assert_eq!(config.effective_step_size(), 25);
        assert!(config.anchored);
        assert_eq!(config.min_splits, 5);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_walk_forward_config_validation() {
        let invalid = WalkForwardConfig::new().in_sample_size(0);
        assert!(invalid.validate().is_err());

        let invalid = WalkForwardConfig::new().out_of_sample_size(0);
        assert!(invalid.validate().is_err());

        let invalid = WalkForwardConfig::new().min_splits(0);
        assert!(invalid.validate().is_err());

        let invalid = WalkForwardConfig::new().min_splits(10).max_splits(5);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_calculate_splits() {
        let config = WalkForwardConfig::new()
            .in_sample_size(100)
            .out_of_sample_size(25)
            .step_size(25)
            .min_splits(2);

        // 100 IS + 25 OOS = 125 minimum
        // With step 25, each additional split needs 25 more
        // 150 periods: (150 - 125) / 25 + 1 = 2 splits
        let n_splits = config.calculate_splits(150).unwrap();
        assert_eq!(n_splits, 2);

        // 200 periods: (200 - 125) / 25 + 1 = 4 splits
        let n_splits = config.calculate_splits(200).unwrap();
        assert_eq!(n_splits, 4);

        // Too short
        assert!(config.calculate_splits(100).is_err());
    }

    #[test]
    fn test_optimizer_splits_rolling() {
        let config = WalkForwardConfig::new()
            .in_sample_size(100)
            .out_of_sample_size(25)
            .step_size(25)
            .anchored(false)
            .min_splits(2);

        let optimizer = WalkForwardOptimizer::new(config, 200).unwrap();
        let splits = optimizer.splits();

        // First split: IS [0, 100), OOS [100, 125)
        assert_eq!(splits[0].in_sample_start, 0);
        assert_eq!(splits[0].in_sample_end, 100);
        assert_eq!(splits[0].out_of_sample_start, 100);
        assert_eq!(splits[0].out_of_sample_end, 125);

        // Second split: IS [25, 125), OOS [125, 150)
        assert_eq!(splits[1].in_sample_start, 25);
        assert_eq!(splits[1].in_sample_end, 125);
        assert_eq!(splits[1].out_of_sample_start, 125);
        assert_eq!(splits[1].out_of_sample_end, 150);
    }

    #[test]
    fn test_optimizer_splits_anchored() {
        let config = WalkForwardConfig::new()
            .in_sample_size(100)
            .out_of_sample_size(25)
            .step_size(25)
            .anchored(true)
            .min_splits(2);

        let optimizer = WalkForwardOptimizer::new(config, 200).unwrap();
        let splits = optimizer.splits();

        // First split: IS [0, 100), OOS [100, 125)
        assert_eq!(splits[0].in_sample_start, 0);
        assert_eq!(splits[0].in_sample_end, 100);

        // Second split: IS [0, 125), OOS [125, 150) - anchored starts at 0
        assert_eq!(splits[1].in_sample_start, 0);
        assert_eq!(splits[1].in_sample_end, 125);
    }

    #[test]
    fn test_split_performance() {
        let mut perf = SplitPerformance::new(0);
        perf.in_sample_objective = 2.0;
        perf.out_of_sample_objective = 1.5;

        let ratio = perf.overfitting_ratio();
        assert!((ratio - 1.333).abs() < 0.01);

        let degradation = perf.degradation_pct();
        assert!((degradation - 25.0).abs() < 0.1);

        assert!(!perf.is_overfit(1.5));
        assert!(perf.is_overfit(1.2));
    }

    #[test]
    fn test_walk_forward_result() {
        let config = WalkForwardConfig::new();
        let mut result = WalkForwardResult::new(config, 3, 1000);

        let mut perf1 = SplitPerformance::new(0);
        perf1.out_of_sample_return = 0.05;
        perf1.out_of_sample_sharpe = 1.0;
        perf1.in_sample_objective = 1.5;
        perf1.out_of_sample_objective = 1.2;

        let mut perf2 = SplitPerformance::new(1);
        perf2.out_of_sample_return = 0.03;
        perf2.out_of_sample_sharpe = 0.8;
        perf2.in_sample_objective = 1.6;
        perf2.out_of_sample_objective = 1.0;

        result.add_split(perf1);
        result.add_split(perf2);

        let agg_sharpe = result.aggregated_sharpe();
        assert!((agg_sharpe - 0.9).abs() < 0.01);

        let overfitting = result.overfitting_ratio();
        assert!(overfitting > 1.0); // IS > OOS indicates overfitting
    }

    #[test]
    fn test_split_metrics_builder() {
        let metrics = SplitMetrics::new()
            .with_total_return(0.15)
            .with_sharpe(1.5)
            .with_max_drawdown(-0.1)
            .with_volatility(0.2)
            .with_n_trades(50)
            .with_custom_metric("win_rate", 0.55);

        assert!((metrics.total_return - 0.15).abs() < 1e-10);
        assert!((metrics.sharpe_ratio - 1.5).abs() < 1e-10);
        assert!((metrics.max_drawdown - (-0.1)).abs() < 1e-10);
        assert_eq!(metrics.n_trades, 50);
        assert!((metrics.custom_metrics.get("win_rate").unwrap() - 0.55).abs() < 1e-10);
    }
}
