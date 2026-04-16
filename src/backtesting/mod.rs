//! Backtesting and validation infrastructure for RL trading strategies.
//!
//! This module provides comprehensive tools for validating and stress-testing
//! reinforcement learning strategies before deployment:
//!
//! - [`walk_forward`] - Walk-forward optimization with rolling/anchored windows
//! - [`monte_carlo`] - Monte Carlo simulation for confidence intervals and stress testing
//! - [`cross_validation`] - Specialized CV techniques for time series and RL
//!
//! # Key Features
//!
//! - **Information Leakage Prevention**: Purge gaps and embargo periods prevent
//!   data leakage between training and testing periods
//! - **Overfitting Detection**: Compare in-sample vs out-of-sample performance
//! - **Statistical Robustness**: Monte Carlo confidence intervals for all metrics
//! - **Stress Testing**: Flash crash, volatility spike, and liquidity crisis scenarios
//!
//! # Example: Walk-Forward Optimization
//!
//! ```ignore
//! use octane_rs::backtesting::{WalkForwardConfig, WalkForwardOptimizer, WalkForwardObjective};
//!
//! let config = WalkForwardConfig::new()
//!     .in_sample_size(252)      // 1 year training
//!     .out_of_sample_size(63)   // 3 months testing
//!     .step_size(63)            // Roll forward quarterly
//!     .objective(WalkForwardObjective::SharpeRatio);
//!
//! let optimizer = WalkForwardOptimizer::new(config, data_length)?;
//!
//! let result = optimizer.run(
//!     |split_idx, train_range| {
//!         // Optimize strategy on training data
//!         Ok((best_params, train_metrics))
//!     },
//!     |split_idx, params, test_range| {
//!         // Evaluate on test data
//!         Ok(test_metrics)
//!     },
//! )?;
//!
//! println!("Aggregated OOS Sharpe: {:.3}", result.aggregated_sharpe());
//! println!("Overfitting Ratio: {:.3}", result.overfitting_ratio());
//! ```
//!
//! # Example: Monte Carlo Confidence Intervals
//!
//! ```ignore
//! use octane_rs::backtesting::{MonteCarloConfig, MonteCarloSimulator, ConfidenceLevel};
//!
//! let config = MonteCarloConfig::new()
//!     .n_simulations(10000)
//!     .confidence_level(ConfidenceLevel::P95)
//!     .seed(42);
//!
//! let mut simulator = MonteCarloSimulator::new(config)?;
//!
//! // Bootstrap analysis of trade returns
//! let trades = vec![0.02, -0.01, 0.03, -0.015, 0.025];
//! let result = simulator.bootstrap_trades(&trades)?;
//!
//! println!("Mean Return: {:.2}%", result.mean_return * 100.0);
//! println!("95% CI: [{:.2}%, {:.2}%]",
//!          result.return_ci.0 * 100.0,
//!          result.return_ci.1 * 100.0);
//! println!("Prob Loss: {:.1}%", result.prob_loss * 100.0);
//! ```
//!
//! # Example: Purged Cross-Validation
//!
//! ```ignore
//! use octane_rs::backtesting::{CVConfig, CrossValidator, CVMethod, CVScoring};
//!
//! let config = CVConfig::new()
//!     .method(CVMethod::PurgedKFold { n_splits: 5 })
//!     .purge_gap(5)        // 5-period gap between train/test
//!     .embargo_periods(2)  // 2-period embargo after test
//!     .scoring(vec![CVScoring::SharpeRatio, CVScoring::MaxDrawdown]);
//!
//! let cv = CrossValidator::new(config)?;
//!
//! let result = cv.validate(data_length, |train_idx, test_idx| {
//!     // Train and evaluate
//!     Ok(metrics)
//! })?;
//!
//! println!("{}", result.summary());
//! ```

mod cross_validation;
mod monte_carlo;
mod walk_forward;

// Walk-forward optimization exports
pub use walk_forward::{
    SplitMetrics, SplitPerformance, WalkForwardConfig, WalkForwardObjective, WalkForwardOptimizer,
    WalkForwardResult, WalkForwardSplit, WalkForwardSummary,
};

// Monte Carlo simulation exports
pub use monte_carlo::{
    BootstrapResult, ConfidenceLevel, MonteCarloConfig, MonteCarloResult, MonteCarloSimulator,
    PerturbationResult, PriceModel, PricePathResult, StressScenario, StressTestResult,
};

// Cross-validation exports
pub use cross_validation::{
    CVConfig, CVFold, CVMethod, CVMetrics, CVResult, CVScoring, CVSummary, CrossValidator,
    MetricSummary,
};
