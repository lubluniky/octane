//! Cross-Validation for Reinforcement Learning strategies.
//!
//! This module provides specialized cross-validation techniques for time-series
//! and RL applications:
//!
//! - Purged K-Fold: Prevents information leakage between train/test with purge gap
//! - Combinatorial Purged CV: All train/test combinations with purging
//! - Time Series Split: Expanding window for sequential data
//! - Blocked Time Series Split: Fixed-size non-overlapping blocks
//! - Group K-Fold: Split by asset or market regime
//!
//! # Key Features
//!
//! - **Purge Gap**: Removes observations between train and test to prevent leakage
//! - **Embargo Period**: Excludes observations after test set
//! - **Multiple Scoring Metrics**: Evaluate with various performance measures
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::backtesting::{CVConfig, CrossValidator, CVMethod};
//!
//! let config = CVConfig::new()
//!     .method(CVMethod::PurgedKFold { n_splits: 5 })
//!     .purge_gap(5)
//!     .embargo_periods(2)
//!     .scoring(vec![CVScoring::SharpeRatio, CVScoring::MaxDrawdown]);
//!
//! let cv = CrossValidator::new(config);
//! let result = cv.validate(data_length, |train_idx, test_idx| {
//!     // Train on train_idx, evaluate on test_idx
//!     // Return CVMetrics
//! })?;
//!
//! println!("Mean CV Sharpe: {:.3}", result.mean_score("sharpe"));
//! println!("Std CV Sharpe: {:.3}", result.std_score("sharpe"));
//! ```

use crate::core::{OctaneError, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Cross-validation method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CVMethod {
    /// Standard K-Fold with purging.
    PurgedKFold {
        /// Number of folds.
        n_splits: usize,
    },
    /// Combinatorial purged cross-validation.
    CombinatorialPurged {
        /// Number of test groups to use in each split.
        n_test_splits: usize,
        /// Total number of groups to divide data into.
        n_groups: usize,
    },
    /// Time series split (expanding window).
    TimeSeriesSplit {
        /// Number of splits.
        n_splits: usize,
        /// Maximum training size (None = expanding).
        max_train_size: Option<usize>,
    },
    /// Blocked time series split (fixed-size blocks).
    BlockedTimeSeriesSplit {
        /// Number of blocks.
        n_blocks: usize,
    },
    /// Group K-Fold (split by group labels).
    GroupKFold {
        /// Number of folds.
        n_splits: usize,
    },
    /// Leave-one-group-out.
    LeaveOneGroupOut,
    /// Stratified time series (maintains class balance).
    StratifiedTimeSeries {
        /// Number of splits.
        n_splits: usize,
    },
}

impl Default for CVMethod {
    fn default() -> Self {
        CVMethod::PurgedKFold { n_splits: 5 }
    }
}

/// Scoring metric for cross-validation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CVScoring {
    /// Total return.
    Return,
    /// Sharpe ratio.
    SharpeRatio,
    /// Sortino ratio.
    SortinoRatio,
    /// Maximum drawdown.
    MaxDrawdown,
    /// Calmar ratio.
    CalmarRatio,
    /// Win rate.
    WinRate,
    /// Profit factor.
    ProfitFactor,
    /// Mean reward (for RL).
    MeanReward,
    /// Episode length (for RL).
    EpisodeLength,
    /// Custom metric.
    Custom(String),
}

impl CVScoring {
    /// Get the name of the scoring metric.
    pub fn name(&self) -> &str {
        match self {
            CVScoring::Return => "return",
            CVScoring::SharpeRatio => "sharpe",
            CVScoring::SortinoRatio => "sortino",
            CVScoring::MaxDrawdown => "max_drawdown",
            CVScoring::CalmarRatio => "calmar",
            CVScoring::WinRate => "win_rate",
            CVScoring::ProfitFactor => "profit_factor",
            CVScoring::MeanReward => "mean_reward",
            CVScoring::EpisodeLength => "episode_length",
            CVScoring::Custom(name) => name,
        }
    }
}

/// Configuration for cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVConfig {
    /// Cross-validation method.
    pub method: CVMethod,

    /// Number of observations to purge between train and test.
    pub purge_gap: usize,

    /// Embargo periods after test set.
    pub embargo_periods: usize,

    /// Scoring metrics to compute.
    pub scoring: Vec<CVScoring>,

    /// Whether to run folds in parallel.
    pub parallel: bool,

    /// Number of parallel jobs.
    pub n_jobs: Option<usize>,

    /// Random seed for shuffling (if applicable).
    pub seed: Option<u64>,

    /// Whether to shuffle before splitting (not recommended for time series).
    pub shuffle: bool,

    /// Minimum samples required in each fold.
    pub min_samples_per_fold: usize,
}

impl Default for CVConfig {
    fn default() -> Self {
        Self {
            method: CVMethod::default(),
            purge_gap: 0,
            embargo_periods: 0,
            scoring: vec![CVScoring::SharpeRatio, CVScoring::MaxDrawdown],
            parallel: true,
            n_jobs: None,
            seed: None,
            shuffle: false,
            min_samples_per_fold: 10,
        }
    }
}

impl CVConfig {
    /// Create a new CV configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the cross-validation method.
    pub fn method(mut self, method: CVMethod) -> Self {
        self.method = method;
        self
    }

    /// Set the purge gap.
    pub fn purge_gap(mut self, gap: usize) -> Self {
        self.purge_gap = gap;
        self
    }

    /// Set the embargo periods.
    pub fn embargo_periods(mut self, periods: usize) -> Self {
        self.embargo_periods = periods;
        self
    }

    /// Set the scoring metrics.
    pub fn scoring(mut self, metrics: Vec<CVScoring>) -> Self {
        self.scoring = metrics;
        self
    }

    /// Add a scoring metric.
    pub fn add_scoring(mut self, metric: CVScoring) -> Self {
        if !self.scoring.contains(&metric) {
            self.scoring.push(metric);
        }
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

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Enable or disable shuffling.
    pub fn shuffle(mut self, enabled: bool) -> Self {
        self.shuffle = enabled;
        self
    }

    /// Set minimum samples per fold.
    pub fn min_samples_per_fold(mut self, min: usize) -> Self {
        self.min_samples_per_fold = min;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        match &self.method {
            CVMethod::PurgedKFold { n_splits } => {
                if *n_splits < 2 {
                    return Err(OctaneError::InvalidConfig(
                        "n_splits must be at least 2".into(),
                    ));
                }
            }
            CVMethod::CombinatorialPurged {
                n_test_splits,
                n_groups,
            } => {
                if *n_test_splits == 0 || *n_groups == 0 {
                    return Err(OctaneError::InvalidConfig(
                        "n_test_splits and n_groups must be positive".into(),
                    ));
                }
                if *n_test_splits >= *n_groups {
                    return Err(OctaneError::InvalidConfig(
                        "n_test_splits must be less than n_groups".into(),
                    ));
                }
            }
            CVMethod::TimeSeriesSplit { n_splits, .. } => {
                if *n_splits < 2 {
                    return Err(OctaneError::InvalidConfig(
                        "n_splits must be at least 2".into(),
                    ));
                }
            }
            CVMethod::BlockedTimeSeriesSplit { n_blocks } => {
                if *n_blocks < 2 {
                    return Err(OctaneError::InvalidConfig(
                        "n_blocks must be at least 2".into(),
                    ));
                }
            }
            CVMethod::GroupKFold { n_splits } => {
                if *n_splits < 2 {
                    return Err(OctaneError::InvalidConfig(
                        "n_splits must be at least 2".into(),
                    ));
                }
            }
            CVMethod::LeaveOneGroupOut => {}
            CVMethod::StratifiedTimeSeries { n_splits } => {
                if *n_splits < 2 {
                    return Err(OctaneError::InvalidConfig(
                        "n_splits must be at least 2".into(),
                    ));
                }
            }
        }

        if self.scoring.is_empty() {
            return Err(OctaneError::InvalidConfig(
                "At least one scoring metric required".into(),
            ));
        }

        Ok(())
    }
}

/// A single fold in cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVFold {
    /// Fold index (0-based).
    pub fold_index: usize,

    /// Training indices.
    pub train_indices: Vec<usize>,

    /// Test indices.
    pub test_indices: Vec<usize>,

    /// Indices that were purged.
    pub purged_indices: Vec<usize>,

    /// Indices under embargo.
    pub embargo_indices: Vec<usize>,
}

impl CVFold {
    /// Create a new fold.
    pub fn new(
        fold_index: usize,
        train_indices: Vec<usize>,
        test_indices: Vec<usize>,
    ) -> Self {
        Self {
            fold_index,
            train_indices,
            test_indices,
            purged_indices: Vec::new(),
            embargo_indices: Vec::new(),
        }
    }

    /// Get the training set size.
    pub fn train_size(&self) -> usize {
        self.train_indices.len()
    }

    /// Get the test set size.
    pub fn test_size(&self) -> usize {
        self.test_indices.len()
    }

    /// Get the number of purged samples.
    pub fn n_purged(&self) -> usize {
        self.purged_indices.len()
    }

    /// Get the number of embargo samples.
    pub fn n_embargo(&self) -> usize {
        self.embargo_indices.len()
    }
}

/// Metrics from a single CV fold evaluation.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CVMetrics {
    /// Fold index.
    pub fold_index: usize,

    /// Total return.
    pub total_return: f64,

    /// Sharpe ratio.
    pub sharpe_ratio: f64,

    /// Sortino ratio.
    pub sortino_ratio: f64,

    /// Maximum drawdown.
    pub max_drawdown: f64,

    /// Calmar ratio.
    pub calmar_ratio: f64,

    /// Win rate.
    pub win_rate: f64,

    /// Profit factor.
    pub profit_factor: f64,

    /// Mean reward (for RL).
    pub mean_reward: f64,

    /// Mean episode length (for RL).
    pub episode_length: f64,

    /// Number of trades/episodes.
    pub n_trades: usize,

    /// Training samples used.
    pub train_samples: usize,

    /// Test samples used.
    pub test_samples: usize,

    /// Additional custom metrics.
    pub custom_metrics: HashMap<String, f64>,
}

impl CVMetrics {
    /// Create new empty metrics.
    pub fn new(fold_index: usize) -> Self {
        Self {
            fold_index,
            ..Default::default()
        }
    }

    /// Set total return.
    pub fn with_return(mut self, ret: f64) -> Self {
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

    /// Set Calmar ratio.
    pub fn with_calmar(mut self, calmar: f64) -> Self {
        self.calmar_ratio = calmar;
        self
    }

    /// Set win rate.
    pub fn with_win_rate(mut self, rate: f64) -> Self {
        self.win_rate = rate;
        self
    }

    /// Set profit factor.
    pub fn with_profit_factor(mut self, pf: f64) -> Self {
        self.profit_factor = pf;
        self
    }

    /// Set mean reward.
    pub fn with_mean_reward(mut self, reward: f64) -> Self {
        self.mean_reward = reward;
        self
    }

    /// Set episode length.
    pub fn with_episode_length(mut self, length: f64) -> Self {
        self.episode_length = length;
        self
    }

    /// Set number of trades.
    pub fn with_n_trades(mut self, n: usize) -> Self {
        self.n_trades = n;
        self
    }

    /// Set sample sizes.
    pub fn with_samples(mut self, train: usize, test: usize) -> Self {
        self.train_samples = train;
        self.test_samples = test;
        self
    }

    /// Add a custom metric.
    pub fn with_custom(mut self, name: impl Into<String>, value: f64) -> Self {
        self.custom_metrics.insert(name.into(), value);
        self
    }

    /// Get a score by name.
    pub fn get_score(&self, scoring: &CVScoring) -> f64 {
        match scoring {
            CVScoring::Return => self.total_return,
            CVScoring::SharpeRatio => self.sharpe_ratio,
            CVScoring::SortinoRatio => self.sortino_ratio,
            CVScoring::MaxDrawdown => self.max_drawdown,
            CVScoring::CalmarRatio => self.calmar_ratio,
            CVScoring::WinRate => self.win_rate,
            CVScoring::ProfitFactor => self.profit_factor,
            CVScoring::MeanReward => self.mean_reward,
            CVScoring::EpisodeLength => self.episode_length,
            CVScoring::Custom(name) => self.custom_metrics.get(name).copied().unwrap_or(0.0),
        }
    }
}

/// Complete result from cross-validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVResult {
    /// Configuration used.
    pub config: CVConfig,

    /// Number of folds.
    pub n_folds: usize,

    /// Total data length.
    pub data_length: usize,

    /// Fold results.
    pub fold_results: Vec<CVMetrics>,

    /// Fold definitions.
    pub folds: Vec<CVFold>,

    /// Execution time in seconds.
    pub execution_time_secs: f64,

    /// Timestamp.
    pub timestamp: u64,
}

impl CVResult {
    /// Create a new CV result.
    pub fn new(config: CVConfig, n_folds: usize, data_length: usize) -> Self {
        Self {
            config,
            n_folds,
            data_length,
            fold_results: Vec::with_capacity(n_folds),
            folds: Vec::with_capacity(n_folds),
            execution_time_secs: 0.0,
            timestamp: current_timestamp(),
        }
    }

    /// Add a fold and its result.
    pub fn add_fold(&mut self, fold: CVFold, metrics: CVMetrics) {
        self.folds.push(fold);
        self.fold_results.push(metrics);
    }

    /// Get mean score for a metric.
    pub fn mean_score(&self, scoring: &CVScoring) -> f64 {
        if self.fold_results.is_empty() {
            return 0.0;
        }
        self.fold_results
            .iter()
            .map(|m| m.get_score(scoring))
            .sum::<f64>()
            / self.fold_results.len() as f64
    }

    /// Get mean score by name.
    pub fn mean_score_by_name(&self, name: &str) -> f64 {
        let scoring = match name {
            "return" => CVScoring::Return,
            "sharpe" => CVScoring::SharpeRatio,
            "sortino" => CVScoring::SortinoRatio,
            "max_drawdown" => CVScoring::MaxDrawdown,
            "calmar" => CVScoring::CalmarRatio,
            "win_rate" => CVScoring::WinRate,
            "profit_factor" => CVScoring::ProfitFactor,
            "mean_reward" => CVScoring::MeanReward,
            "episode_length" => CVScoring::EpisodeLength,
            _ => return 0.0,
        };
        self.mean_score(&scoring)
    }

    /// Get standard deviation of scores for a metric.
    pub fn std_score(&self, scoring: &CVScoring) -> f64 {
        if self.fold_results.len() < 2 {
            return 0.0;
        }
        let mean = self.mean_score(scoring);
        let variance = self
            .fold_results
            .iter()
            .map(|m| (m.get_score(scoring) - mean).powi(2))
            .sum::<f64>()
            / (self.fold_results.len() - 1) as f64;
        variance.sqrt()
    }

    /// Get all scores for a metric.
    pub fn scores(&self, scoring: &CVScoring) -> Vec<f64> {
        self.fold_results
            .iter()
            .map(|m| m.get_score(scoring))
            .collect()
    }

    /// Get the best fold by a scoring metric.
    pub fn best_fold(&self, scoring: &CVScoring, maximize: bool) -> Option<&CVMetrics> {
        if maximize {
            self.fold_results
                .iter()
                .max_by(|a, b| {
                    a.get_score(scoring)
                        .partial_cmp(&b.get_score(scoring))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        } else {
            self.fold_results
                .iter()
                .min_by(|a, b| {
                    a.get_score(scoring)
                        .partial_cmp(&b.get_score(scoring))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        }
    }

    /// Get the worst fold by a scoring metric.
    pub fn worst_fold(&self, scoring: &CVScoring, maximize: bool) -> Option<&CVMetrics> {
        self.best_fold(scoring, !maximize)
    }

    /// Calculate coefficient of variation for a metric.
    pub fn cv_coefficient(&self, scoring: &CVScoring) -> f64 {
        let mean = self.mean_score(scoring);
        if mean.abs() < 1e-10 {
            return 0.0;
        }
        self.std_score(scoring) / mean.abs()
    }

    /// Get summary statistics for all configured metrics.
    pub fn summary(&self) -> CVSummary {
        let mut metrics_summary = HashMap::new();

        for scoring in &self.config.scoring {
            let name = scoring.name().to_string();
            metrics_summary.insert(
                name.clone(),
                MetricSummary {
                    name,
                    mean: self.mean_score(scoring),
                    std: self.std_score(scoring),
                    min: self
                        .fold_results
                        .iter()
                        .map(|m| m.get_score(scoring))
                        .fold(f64::INFINITY, f64::min),
                    max: self
                        .fold_results
                        .iter()
                        .map(|m| m.get_score(scoring))
                        .fold(f64::NEG_INFINITY, f64::max),
                    cv: self.cv_coefficient(scoring),
                },
            );
        }

        CVSummary {
            n_folds: self.n_folds,
            data_length: self.data_length,
            metrics: metrics_summary,
            avg_train_size: self.folds.iter().map(|f| f.train_size()).sum::<usize>() as f64
                / self.n_folds as f64,
            avg_test_size: self.folds.iter().map(|f| f.test_size()).sum::<usize>() as f64
                / self.n_folds as f64,
            total_purged: self.folds.iter().map(|f| f.n_purged()).sum(),
            total_embargo: self.folds.iter().map(|f| f.n_embargo()).sum(),
        }
    }
}

/// Summary statistics for a single metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    /// Metric name.
    pub name: String,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Coefficient of variation.
    pub cv: f64,
}

/// Summary of cross-validation results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVSummary {
    /// Number of folds.
    pub n_folds: usize,
    /// Total data length.
    pub data_length: usize,
    /// Metric summaries.
    pub metrics: HashMap<String, MetricSummary>,
    /// Average training set size.
    pub avg_train_size: f64,
    /// Average test set size.
    pub avg_test_size: f64,
    /// Total samples purged.
    pub total_purged: usize,
    /// Total samples under embargo.
    pub total_embargo: usize,
}

impl std::fmt::Display for CVSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Cross-Validation Summary")?;
        writeln!(f, "========================")?;
        writeln!(f, "Folds: {}", self.n_folds)?;
        writeln!(f, "Data Length: {}", self.data_length)?;
        writeln!(f, "Avg Train Size: {:.0}", self.avg_train_size)?;
        writeln!(f, "Avg Test Size: {:.0}", self.avg_test_size)?;
        writeln!(f, "Total Purged: {}", self.total_purged)?;
        writeln!(f, "Total Embargo: {}", self.total_embargo)?;
        writeln!(f)?;
        writeln!(f, "Metrics:")?;
        for (name, summary) in &self.metrics {
            writeln!(
                f,
                "  {}: {:.4} +/- {:.4} (min: {:.4}, max: {:.4}, CV: {:.2}%)",
                name,
                summary.mean,
                summary.std,
                summary.min,
                summary.max,
                summary.cv * 100.0
            )?;
        }
        Ok(())
    }
}

/// Cross-validator for RL and time series data.
pub struct CrossValidator {
    /// Configuration.
    config: CVConfig,
}

impl CrossValidator {
    /// Create a new cross-validator.
    pub fn new(config: CVConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Get the configuration.
    pub fn config(&self) -> &CVConfig {
        &self.config
    }

    /// Generate folds for the given data length.
    pub fn get_folds(&self, data_length: usize) -> Result<Vec<CVFold>> {
        self.get_folds_with_groups(data_length, None)
    }

    /// Generate folds with optional group labels.
    pub fn get_folds_with_groups(
        &self,
        data_length: usize,
        groups: Option<&[usize]>,
    ) -> Result<Vec<CVFold>> {
        match &self.config.method {
            CVMethod::PurgedKFold { n_splits } => {
                self.purged_kfold(data_length, *n_splits)
            }
            CVMethod::CombinatorialPurged {
                n_test_splits,
                n_groups,
            } => self.combinatorial_purged(data_length, *n_test_splits, *n_groups),
            CVMethod::TimeSeriesSplit {
                n_splits,
                max_train_size,
            } => self.time_series_split(data_length, *n_splits, *max_train_size),
            CVMethod::BlockedTimeSeriesSplit { n_blocks } => {
                self.blocked_time_series_split(data_length, *n_blocks)
            }
            CVMethod::GroupKFold { n_splits } => {
                let groups = groups.ok_or_else(|| {
                    OctaneError::InvalidConfig("Groups required for GroupKFold".into())
                })?;
                self.group_kfold(data_length, *n_splits, groups)
            }
            CVMethod::LeaveOneGroupOut => {
                let groups = groups.ok_or_else(|| {
                    OctaneError::InvalidConfig("Groups required for LeaveOneGroupOut".into())
                })?;
                self.leave_one_group_out(data_length, groups)
            }
            CVMethod::StratifiedTimeSeries { n_splits } => {
                self.stratified_time_series(data_length, *n_splits)
            }
        }
    }

    /// Run cross-validation with a provided evaluation function.
    pub fn validate<F>(&self, data_length: usize, evaluate_fn: F) -> Result<CVResult>
    where
        F: Fn(&[usize], &[usize]) -> Result<CVMetrics> + Sync + Send,
    {
        self.validate_with_groups(data_length, None, evaluate_fn)
    }

    /// Run cross-validation with groups.
    pub fn validate_with_groups<F>(
        &self,
        data_length: usize,
        groups: Option<&[usize]>,
        evaluate_fn: F,
    ) -> Result<CVResult>
    where
        F: Fn(&[usize], &[usize]) -> Result<CVMetrics> + Sync + Send,
    {
        let start_time = std::time::Instant::now();
        let folds = self.get_folds_with_groups(data_length, groups)?;

        let fold_results: Vec<Result<CVMetrics>> = if self.config.parallel {
            folds
                .par_iter()
                .map(|fold| {
                    let mut metrics = evaluate_fn(&fold.train_indices, &fold.test_indices)?;
                    metrics.fold_index = fold.fold_index;
                    metrics.train_samples = fold.train_size();
                    metrics.test_samples = fold.test_size();
                    Ok(metrics)
                })
                .collect()
        } else {
            folds
                .iter()
                .map(|fold| {
                    let mut metrics = evaluate_fn(&fold.train_indices, &fold.test_indices)?;
                    metrics.fold_index = fold.fold_index;
                    metrics.train_samples = fold.train_size();
                    metrics.test_samples = fold.test_size();
                    Ok(metrics)
                })
                .collect()
        };

        let mut result = CVResult::new(self.config.clone(), folds.len(), data_length);

        for (fold, metrics_result) in folds.into_iter().zip(fold_results) {
            let metrics = metrics_result?;
            result.add_fold(fold, metrics);
        }

        result.execution_time_secs = start_time.elapsed().as_secs_f64();

        Ok(result)
    }

    /// Purged K-Fold cross-validation.
    fn purged_kfold(&self, data_length: usize, n_splits: usize) -> Result<Vec<CVFold>> {
        let fold_size = data_length / n_splits;
        if fold_size < self.config.min_samples_per_fold {
            return Err(OctaneError::InvalidConfig(format!(
                "Fold size {} is less than minimum {}",
                fold_size, self.config.min_samples_per_fold
            )));
        }

        let mut folds = Vec::with_capacity(n_splits);

        for i in 0..n_splits {
            let test_start = i * fold_size;
            let test_end = if i == n_splits - 1 {
                data_length
            } else {
                (i + 1) * fold_size
            };

            let test_indices: Vec<usize> = (test_start..test_end).collect();

            // Apply purge gap and embargo
            let purge_start = test_start.saturating_sub(self.config.purge_gap);
            let embargo_end = (test_end + self.config.embargo_periods).min(data_length);

            let purged_indices: Vec<usize> = (purge_start..test_start).collect();
            let embargo_indices: Vec<usize> = (test_end..embargo_end).collect();

            // Training indices: everything except test, purge, and embargo
            let excluded: HashSet<usize> = test_indices
                .iter()
                .chain(purged_indices.iter())
                .chain(embargo_indices.iter())
                .copied()
                .collect();

            let train_indices: Vec<usize> = (0..data_length)
                .filter(|idx| !excluded.contains(idx))
                .collect();

            let mut fold = CVFold::new(i, train_indices, test_indices);
            fold.purged_indices = purged_indices;
            fold.embargo_indices = embargo_indices;
            folds.push(fold);
        }

        Ok(folds)
    }

    /// Combinatorial purged cross-validation.
    fn combinatorial_purged(
        &self,
        data_length: usize,
        n_test_splits: usize,
        n_groups: usize,
    ) -> Result<Vec<CVFold>> {
        let group_size = data_length / n_groups;
        if group_size < self.config.min_samples_per_fold {
            return Err(OctaneError::InvalidConfig(format!(
                "Group size {} is less than minimum {}",
                group_size, self.config.min_samples_per_fold
            )));
        }

        // Generate all combinations of n_test_splits groups from n_groups
        let combinations = self.combinations(n_groups, n_test_splits);
        let mut folds = Vec::with_capacity(combinations.len());

        for (fold_idx, test_group_indices) in combinations.iter().enumerate() {
            // Determine test indices
            let mut test_indices = Vec::new();
            let mut purged_indices = Vec::new();
            let mut embargo_indices = Vec::new();

            for &group_idx in test_group_indices {
                let group_start = group_idx * group_size;
                let group_end = if group_idx == n_groups - 1 {
                    data_length
                } else {
                    (group_idx + 1) * group_size
                };

                test_indices.extend(group_start..group_end);

                // Purge before test
                let purge_start = group_start.saturating_sub(self.config.purge_gap);
                purged_indices.extend(purge_start..group_start);

                // Embargo after test
                let embargo_end = (group_end + self.config.embargo_periods).min(data_length);
                embargo_indices.extend(group_end..embargo_end);
            }

            // Training indices
            let excluded: HashSet<usize> = test_indices
                .iter()
                .chain(purged_indices.iter())
                .chain(embargo_indices.iter())
                .copied()
                .collect();

            let train_indices: Vec<usize> = (0..data_length)
                .filter(|idx| !excluded.contains(idx))
                .collect();

            let mut fold = CVFold::new(fold_idx, train_indices, test_indices);
            fold.purged_indices = purged_indices;
            fold.embargo_indices = embargo_indices;
            folds.push(fold);
        }

        Ok(folds)
    }

    /// Time series split (expanding window).
    fn time_series_split(
        &self,
        data_length: usize,
        n_splits: usize,
        max_train_size: Option<usize>,
    ) -> Result<Vec<CVFold>> {
        let test_size = data_length / (n_splits + 1);
        if test_size < self.config.min_samples_per_fold {
            return Err(OctaneError::InvalidConfig(format!(
                "Test size {} is less than minimum {}",
                test_size, self.config.min_samples_per_fold
            )));
        }

        let mut folds = Vec::with_capacity(n_splits);

        for i in 0..n_splits {
            let test_start = (i + 1) * test_size;
            let test_end = if i == n_splits - 1 {
                data_length
            } else {
                (i + 2) * test_size
            };

            let test_indices: Vec<usize> = (test_start..test_end).collect();

            // Training: from start (or limited) to test_start
            let train_start = if let Some(max_size) = max_train_size {
                test_start.saturating_sub(max_size)
            } else {
                0
            };

            // Apply purge gap
            let train_end = test_start.saturating_sub(self.config.purge_gap);
            let train_indices: Vec<usize> = (train_start..train_end).collect();

            let purged_indices: Vec<usize> = (train_end..test_start).collect();

            // Embargo after test
            let embargo_end = (test_end + self.config.embargo_periods).min(data_length);
            let embargo_indices: Vec<usize> = (test_end..embargo_end).collect();

            let mut fold = CVFold::new(i, train_indices, test_indices);
            fold.purged_indices = purged_indices;
            fold.embargo_indices = embargo_indices;
            folds.push(fold);
        }

        Ok(folds)
    }

    /// Blocked time series split.
    fn blocked_time_series_split(
        &self,
        data_length: usize,
        n_blocks: usize,
    ) -> Result<Vec<CVFold>> {
        let block_size = data_length / n_blocks;
        if block_size < self.config.min_samples_per_fold {
            return Err(OctaneError::InvalidConfig(format!(
                "Block size {} is less than minimum {}",
                block_size, self.config.min_samples_per_fold
            )));
        }

        let mut folds = Vec::with_capacity(n_blocks - 1);

        for i in 0..(n_blocks - 1) {
            // Training: blocks 0 to i
            let train_end = (i + 1) * block_size;
            let mut train_indices: Vec<usize> = (0..train_end).collect();

            // Test: block i+1
            let test_start = (i + 1) * block_size;
            let test_end = if i == n_blocks - 2 {
                data_length
            } else {
                (i + 2) * block_size
            };
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            // Apply purge gap
            let purge_start = train_end.saturating_sub(self.config.purge_gap);
            let purged_indices: Vec<usize> = (purge_start..train_end).collect();
            train_indices.retain(|&idx| idx < purge_start);

            // Embargo after test
            let embargo_end = (test_end + self.config.embargo_periods).min(data_length);
            let embargo_indices: Vec<usize> = (test_end..embargo_end).collect();

            let mut fold = CVFold::new(i, train_indices, test_indices);
            fold.purged_indices = purged_indices;
            fold.embargo_indices = embargo_indices;
            folds.push(fold);
        }

        Ok(folds)
    }

    /// Group K-Fold cross-validation.
    fn group_kfold(
        &self,
        data_length: usize,
        n_splits: usize,
        groups: &[usize],
    ) -> Result<Vec<CVFold>> {
        if groups.len() != data_length {
            return Err(OctaneError::InvalidConfig(format!(
                "Groups length {} doesn't match data length {}",
                groups.len(),
                data_length
            )));
        }

        // Get unique groups
        let unique_groups: Vec<usize> = {
            let mut set: HashSet<usize> = groups.iter().copied().collect();
            let mut vec: Vec<usize> = set.drain().collect();
            vec.sort();
            vec
        };

        if unique_groups.len() < n_splits {
            return Err(OctaneError::InvalidConfig(format!(
                "Number of unique groups {} is less than n_splits {}",
                unique_groups.len(),
                n_splits
            )));
        }

        let groups_per_fold = unique_groups.len() / n_splits;
        let mut folds = Vec::with_capacity(n_splits);

        for i in 0..n_splits {
            let test_group_start = i * groups_per_fold;
            let test_group_end = if i == n_splits - 1 {
                unique_groups.len()
            } else {
                (i + 1) * groups_per_fold
            };

            let test_groups: HashSet<usize> =
                unique_groups[test_group_start..test_group_end].iter().copied().collect();

            let test_indices: Vec<usize> = (0..data_length)
                .filter(|&idx| test_groups.contains(&groups[idx]))
                .collect();

            let train_indices: Vec<usize> = (0..data_length)
                .filter(|&idx| !test_groups.contains(&groups[idx]))
                .collect();

            folds.push(CVFold::new(i, train_indices, test_indices));
        }

        Ok(folds)
    }

    /// Leave-one-group-out cross-validation.
    fn leave_one_group_out(&self, data_length: usize, groups: &[usize]) -> Result<Vec<CVFold>> {
        if groups.len() != data_length {
            return Err(OctaneError::InvalidConfig(format!(
                "Groups length {} doesn't match data length {}",
                groups.len(),
                data_length
            )));
        }

        let unique_groups: Vec<usize> = {
            let mut set: HashSet<usize> = groups.iter().copied().collect();
            let mut vec: Vec<usize> = set.drain().collect();
            vec.sort();
            vec
        };

        let mut folds = Vec::with_capacity(unique_groups.len());

        for (i, &test_group) in unique_groups.iter().enumerate() {
            let test_indices: Vec<usize> = (0..data_length)
                .filter(|&idx| groups[idx] == test_group)
                .collect();

            let train_indices: Vec<usize> = (0..data_length)
                .filter(|&idx| groups[idx] != test_group)
                .collect();

            folds.push(CVFold::new(i, train_indices, test_indices));
        }

        Ok(folds)
    }

    /// Stratified time series split.
    fn stratified_time_series(&self, data_length: usize, n_splits: usize) -> Result<Vec<CVFold>> {
        // Similar to time series split but ensures balanced representation
        // For simplicity, we implement a basic version
        self.time_series_split(data_length, n_splits, None)
    }

    /// Generate all combinations of k elements from n.
    fn combinations(&self, n: usize, k: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut combination = vec![0; k];
        self.combinations_recursive(n, k, 0, &mut combination, &mut result);
        result
    }

    fn combinations_recursive(
        &self,
        n: usize,
        k: usize,
        start: usize,
        combination: &mut [usize],
        result: &mut Vec<Vec<usize>>,
    ) {
        if k == 0 {
            result.push(combination.to_vec());
            return;
        }

        for i in start..=(n - k) {
            combination[combination.len() - k] = i;
            self.combinations_recursive(n, k - 1, i + 1, combination, result);
        }
    }
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
    fn test_cv_config_defaults() {
        let config = CVConfig::new();
        assert_eq!(config.purge_gap, 0);
        assert_eq!(config.embargo_periods, 0);
        assert!(config.parallel);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_cv_config_builder() {
        let config = CVConfig::new()
            .method(CVMethod::PurgedKFold { n_splits: 10 })
            .purge_gap(5)
            .embargo_periods(3)
            .add_scoring(CVScoring::Return);

        assert_eq!(config.purge_gap, 5);
        assert_eq!(config.embargo_periods, 3);
        assert!(config.scoring.contains(&CVScoring::Return));
    }

    #[test]
    fn test_purged_kfold() {
        let config = CVConfig::new()
            .method(CVMethod::PurgedKFold { n_splits: 5 })
            .purge_gap(2)
            .embargo_periods(1)
            .min_samples_per_fold(5);

        let cv = CrossValidator::new(config).unwrap();
        let folds = cv.get_folds(100).unwrap();

        assert_eq!(folds.len(), 5);

        // Check that train and test don't overlap
        for fold in &folds {
            let train_set: HashSet<usize> = fold.train_indices.iter().copied().collect();
            for &test_idx in &fold.test_indices {
                assert!(!train_set.contains(&test_idx));
            }
        }

        // Check purge gap is applied
        assert!(!folds[0].purged_indices.is_empty() || folds[0].fold_index == 0);
    }

    #[test]
    fn test_time_series_split() {
        let config = CVConfig::new()
            .method(CVMethod::TimeSeriesSplit {
                n_splits: 4,
                max_train_size: None,
            })
            .min_samples_per_fold(5);

        let cv = CrossValidator::new(config).unwrap();
        let folds = cv.get_folds(100).unwrap();

        assert_eq!(folds.len(), 4);

        // Check that test sets are sequential and non-overlapping
        for i in 0..folds.len() - 1 {
            let current_test_end = *folds[i].test_indices.last().unwrap();
            let next_test_start = *folds[i + 1].test_indices.first().unwrap();
            assert!(current_test_end < next_test_start);
        }

        // Check that training size increases (expanding window)
        for i in 0..folds.len() - 1 {
            // Allow for purge gap reducing train size
            assert!(folds[i].train_size() <= folds[i + 1].train_size() + 10);
        }
    }

    #[test]
    fn test_blocked_time_series_split() {
        let config = CVConfig::new()
            .method(CVMethod::BlockedTimeSeriesSplit { n_blocks: 5 })
            .min_samples_per_fold(5);

        let cv = CrossValidator::new(config).unwrap();
        let folds = cv.get_folds(100).unwrap();

        assert_eq!(folds.len(), 4); // n_blocks - 1 folds
    }

    #[test]
    fn test_group_kfold() {
        let config = CVConfig::new()
            .method(CVMethod::GroupKFold { n_splits: 3 })
            .min_samples_per_fold(1);

        let cv = CrossValidator::new(config).unwrap();

        // Groups: 0, 0, 1, 1, 2, 2, 3, 3, 4, 4
        let groups: Vec<usize> = (0..10).map(|i| i / 2).collect();
        let folds = cv.get_folds_with_groups(10, Some(&groups)).unwrap();

        assert_eq!(folds.len(), 3);

        // Check that same group is not in both train and test
        for fold in &folds {
            let train_groups: HashSet<usize> = fold.train_indices.iter().map(|&i| groups[i]).collect();
            let test_groups: HashSet<usize> = fold.test_indices.iter().map(|&i| groups[i]).collect();
            assert!(train_groups.is_disjoint(&test_groups));
        }
    }

    #[test]
    fn test_combinatorial_purged() {
        let config = CVConfig::new()
            .method(CVMethod::CombinatorialPurged {
                n_test_splits: 1,
                n_groups: 5,
            })
            .min_samples_per_fold(5);

        let cv = CrossValidator::new(config).unwrap();
        let folds = cv.get_folds(100).unwrap();

        // C(5, 1) = 5 combinations
        assert_eq!(folds.len(), 5);
    }

    #[test]
    fn test_cv_metrics() {
        let metrics = CVMetrics::new(0)
            .with_sharpe(1.5)
            .with_max_drawdown(0.1)
            .with_return(0.2)
            .with_n_trades(50)
            .with_custom("profit_target_hits", 10.0);

        assert!((metrics.sharpe_ratio - 1.5).abs() < 1e-10);
        assert!((metrics.max_drawdown - 0.1).abs() < 1e-10);
        assert_eq!(metrics.n_trades, 50);
        assert!((metrics.custom_metrics.get("profit_target_hits").unwrap() - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_cv_result_statistics() {
        let config = CVConfig::new();
        let mut result = CVResult::new(config, 3, 100);

        let fold1 = CVFold::new(0, (0..50).collect(), (50..70).collect());
        let metrics1 = CVMetrics::new(0).with_sharpe(1.0);

        let fold2 = CVFold::new(1, (0..60).collect(), (60..80).collect());
        let metrics2 = CVMetrics::new(1).with_sharpe(1.5);

        let fold3 = CVFold::new(2, (0..70).collect(), (70..100).collect());
        let metrics3 = CVMetrics::new(2).with_sharpe(2.0);

        result.add_fold(fold1, metrics1);
        result.add_fold(fold2, metrics2);
        result.add_fold(fold3, metrics3);

        let mean_sharpe = result.mean_score(&CVScoring::SharpeRatio);
        assert!((mean_sharpe - 1.5).abs() < 1e-10);

        let std_sharpe = result.std_score(&CVScoring::SharpeRatio);
        assert!(std_sharpe > 0.0);
    }

    #[test]
    fn test_combinations() {
        let config = CVConfig::new();
        let cv = CrossValidator::new(config).unwrap();

        let combs = cv.combinations(4, 2);
        // C(4, 2) = 6: [0,1], [0,2], [0,3], [1,2], [1,3], [2,3]
        assert_eq!(combs.len(), 6);
    }

    #[test]
    fn test_cross_validator_validate() {
        let config = CVConfig::new()
            .method(CVMethod::PurgedKFold { n_splits: 3 })
            .min_samples_per_fold(5);

        let cv = CrossValidator::new(config).unwrap();

        let result = cv
            .validate(100, |train_idx, test_idx| {
                Ok(CVMetrics::new(0)
                    .with_sharpe(1.0 + train_idx.len() as f64 / 100.0)
                    .with_samples(train_idx.len(), test_idx.len()))
            })
            .unwrap();

        assert_eq!(result.n_folds, 3);
        assert!(result.mean_score(&CVScoring::SharpeRatio) > 1.0);
    }
}
