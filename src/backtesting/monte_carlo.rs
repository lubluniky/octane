//! Monte Carlo Simulation for trading strategy analysis.
//!
//! This module provides Monte Carlo simulation capabilities for:
//!
//! - Trade sequence randomization (bootstrap)
//! - Synthetic price path generation
//! - Parameter perturbation analysis
//! - Confidence intervals for metrics
//! - Stress testing scenarios (flash crash, volatility spikes, etc.)
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::backtesting::{
//!     MonteCarloConfig, MonteCarloSimulator, StressScenario, ConfidenceLevel
//! };
//!
//! let config = MonteCarloConfig::new()
//!     .n_simulations(10000)
//!     .confidence_level(ConfidenceLevel::P95)
//!     .bootstrap_trades(true)
//!     .seed(42);
//!
//! let simulator = MonteCarloSimulator::new(config);
//!
//! // Bootstrap analysis of trade sequence
//! let trades = vec![100.0, -50.0, 75.0, -25.0, 150.0, -80.0];
//! let result = simulator.bootstrap_trades(&trades)?;
//! println!("95% CI for total return: [{:.2}, {:.2}]",
//!          result.return_ci.0, result.return_ci.1);
//!
//! // Stress testing
//! let stress_result = simulator.stress_test(StressScenario::FlashCrash { drop_pct: 0.1 })?;
//! ```

use crate::core::{OctaneError, Result};
use rand::prelude::*;
use rand_distr::{Distribution, Normal, Uniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Confidence level for interval estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum ConfidenceLevel {
    /// 90% confidence.
    P90,
    /// 95% confidence (default).
    #[default]
    P95,
    /// 99% confidence.
    P99,
    /// 99.5% confidence.
    P995,
    /// 99.9% confidence.
    P999,
}

impl ConfidenceLevel {
    /// Get the percentile bounds for this confidence level.
    pub fn percentiles(&self) -> (f64, f64) {
        match self {
            ConfidenceLevel::P90 => (0.05, 0.95),
            ConfidenceLevel::P95 => (0.025, 0.975),
            ConfidenceLevel::P99 => (0.005, 0.995),
            ConfidenceLevel::P995 => (0.0025, 0.9975),
            ConfidenceLevel::P999 => (0.0005, 0.9995),
        }
    }

    /// Get the confidence percentage.
    pub fn percentage(&self) -> f64 {
        match self {
            ConfidenceLevel::P90 => 90.0,
            ConfidenceLevel::P95 => 95.0,
            ConfidenceLevel::P99 => 99.0,
            ConfidenceLevel::P995 => 99.5,
            ConfidenceLevel::P999 => 99.9,
        }
    }
}

/// Price path generation model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PriceModel {
    /// Geometric Brownian Motion (standard log-normal).
    GBM {
        /// Annualized drift (expected return).
        drift: f64,
        /// Annualized volatility.
        volatility: f64,
    },
    /// GBM with stochastic volatility (Heston model).
    Heston {
        /// Long-term variance.
        theta: f64,
        /// Mean reversion speed.
        kappa: f64,
        /// Volatility of volatility.
        sigma: f64,
        /// Correlation between price and volatility.
        rho: f64,
        /// Initial variance.
        v0: f64,
    },
    /// Jump-diffusion (Merton model).
    JumpDiffusion {
        /// Base drift.
        drift: f64,
        /// Base volatility.
        volatility: f64,
        /// Jump intensity (expected jumps per year).
        jump_intensity: f64,
        /// Mean jump size (log space).
        jump_mean: f64,
        /// Jump size volatility.
        jump_std: f64,
    },
    /// Mean-reverting (Ornstein-Uhlenbeck).
    MeanReverting {
        /// Mean reversion level.
        mean_level: f64,
        /// Mean reversion speed.
        mean_reversion: f64,
        /// Volatility.
        volatility: f64,
    },
    /// Bootstrap from historical returns.
    Bootstrap,
}

impl Default for PriceModel {
    fn default() -> Self {
        PriceModel::GBM {
            drift: 0.05,
            volatility: 0.2,
        }
    }
}

/// Stress testing scenario.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StressScenario {
    /// Flash crash: sudden market drop.
    FlashCrash {
        /// Percentage drop (e.g., 0.1 = 10%).
        drop_pct: f64,
    },
    /// Volatility spike: sudden increase in volatility.
    VolatilitySpike {
        /// Multiplier for volatility (e.g., 3.0 = 3x normal).
        multiplier: f64,
        /// Duration in periods.
        duration: usize,
    },
    /// Correlation breakdown: correlations change dramatically.
    CorrelationBreakdown {
        /// New correlation level (-1 to 1).
        new_correlation: f64,
    },
    /// Liquidity crisis: increased slippage and spread.
    LiquidityCrisis {
        /// Spread multiplier.
        spread_multiplier: f64,
        /// Slippage multiplier.
        slippage_multiplier: f64,
    },
    /// Regime change: market characteristics shift.
    RegimeChange {
        /// New drift.
        new_drift: f64,
        /// New volatility.
        new_volatility: f64,
    },
    /// Gap event: overnight/weekend gap.
    GapEvent {
        /// Gap size (percentage).
        gap_pct: f64,
    },
    /// Custom scenario with user-defined parameters.
    Custom {
        /// Scenario name.
        name: String,
        /// Parameters for the scenario.
        params: HashMap<String, f64>,
    },
}

/// Configuration for Monte Carlo simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloConfig {
    /// Number of simulations to run.
    pub n_simulations: usize,

    /// Confidence level for intervals.
    pub confidence_level: ConfidenceLevel,

    /// Whether to bootstrap trade sequences.
    pub bootstrap_trades: bool,

    /// Whether to generate synthetic price paths.
    pub synthetic_prices: bool,

    /// Price generation model.
    pub price_model: PriceModel,

    /// Number of periods for synthetic paths.
    pub path_length: usize,

    /// Time step (in years, e.g., 1/252 for daily).
    pub dt: f64,

    /// Initial price for synthetic paths.
    pub initial_price: f64,

    /// Whether to run simulations in parallel.
    pub parallel: bool,

    /// Number of threads for parallel execution.
    pub n_jobs: Option<usize>,

    /// Random seed for reproducibility.
    pub seed: Option<u64>,

    /// Parameter perturbation range (fraction).
    pub perturbation_range: f64,

    /// List of stress scenarios to test.
    pub stress_scenarios: Vec<StressScenario>,

    /// Whether to calculate VaR and CVaR.
    pub calculate_var: bool,

    /// VaR confidence levels.
    pub var_levels: Vec<f64>,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            n_simulations: 10000,
            confidence_level: ConfidenceLevel::P95,
            bootstrap_trades: true,
            synthetic_prices: false,
            price_model: PriceModel::default(),
            path_length: 252,
            dt: 1.0 / 252.0,
            initial_price: 100.0,
            parallel: true,
            n_jobs: None,
            seed: None,
            perturbation_range: 0.1,
            stress_scenarios: Vec::new(),
            calculate_var: true,
            var_levels: vec![0.95, 0.99],
        }
    }
}

impl MonteCarloConfig {
    /// Create a new Monte Carlo configuration with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of simulations.
    pub fn n_simulations(mut self, n: usize) -> Self {
        self.n_simulations = n;
        self
    }

    /// Set the confidence level.
    pub fn confidence_level(mut self, level: ConfidenceLevel) -> Self {
        self.confidence_level = level;
        self
    }

    /// Enable or disable trade bootstrapping.
    pub fn bootstrap_trades(mut self, enabled: bool) -> Self {
        self.bootstrap_trades = enabled;
        self
    }

    /// Enable or disable synthetic price generation.
    pub fn synthetic_prices(mut self, enabled: bool) -> Self {
        self.synthetic_prices = enabled;
        self
    }

    /// Set the price generation model.
    pub fn price_model(mut self, model: PriceModel) -> Self {
        self.price_model = model;
        self
    }

    /// Set the path length for synthetic prices.
    pub fn path_length(mut self, length: usize) -> Self {
        self.path_length = length;
        self
    }

    /// Set the time step (in years).
    pub fn dt(mut self, dt: f64) -> Self {
        self.dt = dt;
        self
    }

    /// Set the initial price.
    pub fn initial_price(mut self, price: f64) -> Self {
        self.initial_price = price;
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

    /// Set the parameter perturbation range.
    pub fn perturbation_range(mut self, range: f64) -> Self {
        self.perturbation_range = range;
        self
    }

    /// Add a stress scenario.
    pub fn add_stress_scenario(mut self, scenario: StressScenario) -> Self {
        self.stress_scenarios.push(scenario);
        self
    }

    /// Set stress scenarios.
    pub fn stress_scenarios(mut self, scenarios: Vec<StressScenario>) -> Self {
        self.stress_scenarios = scenarios;
        self
    }

    /// Enable or disable VaR calculation.
    pub fn calculate_var(mut self, enabled: bool) -> Self {
        self.calculate_var = enabled;
        self
    }

    /// Set VaR confidence levels.
    pub fn var_levels(mut self, levels: Vec<f64>) -> Self {
        self.var_levels = levels;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<()> {
        if self.n_simulations == 0 {
            return Err(OctaneError::InvalidConfig(
                "n_simulations must be positive".into(),
            ));
        }
        if self.path_length == 0 && self.synthetic_prices {
            return Err(OctaneError::InvalidConfig(
                "path_length must be positive for synthetic prices".into(),
            ));
        }
        if self.dt <= 0.0 {
            return Err(OctaneError::InvalidConfig("dt must be positive".into()));
        }
        if self.initial_price <= 0.0 {
            return Err(OctaneError::InvalidConfig(
                "initial_price must be positive".into(),
            ));
        }
        if self.perturbation_range < 0.0 || self.perturbation_range > 1.0 {
            return Err(OctaneError::InvalidConfig(
                "perturbation_range must be in [0, 1]".into(),
            ));
        }
        for &level in &self.var_levels {
            if !(0.0..=1.0).contains(&level) {
                return Err(OctaneError::InvalidConfig(
                    "VaR levels must be in [0, 1]".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Result from bootstrap analysis of trades.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BootstrapResult {
    /// Number of simulations run.
    pub n_simulations: usize,

    /// Mean total return across simulations.
    pub mean_return: f64,

    /// Standard deviation of returns.
    pub std_return: f64,

    /// Confidence interval for total return.
    pub return_ci: (f64, f64),

    /// Mean Sharpe ratio.
    pub mean_sharpe: f64,

    /// Confidence interval for Sharpe ratio.
    pub sharpe_ci: (f64, f64),

    /// Mean maximum drawdown.
    pub mean_max_drawdown: f64,

    /// Confidence interval for max drawdown.
    pub max_drawdown_ci: (f64, f64),

    /// Distribution of returns (histogram bins).
    pub return_distribution: Vec<f64>,

    /// Distribution of max drawdowns.
    pub drawdown_distribution: Vec<f64>,

    /// Probability of negative return.
    pub prob_loss: f64,

    /// Value at Risk (various levels).
    pub var: HashMap<String, f64>,

    /// Conditional Value at Risk (expected shortfall).
    pub cvar: HashMap<String, f64>,

    /// Confidence level used.
    pub confidence_level: ConfidenceLevel,
}

impl BootstrapResult {
    /// Get the skewness of the return distribution.
    pub fn skewness(&self) -> f64 {
        if self.return_distribution.len() < 3 {
            return 0.0;
        }
        let n = self.return_distribution.len() as f64;
        let mean = self.mean_return;
        let std = self.std_return;
        if std.abs() < 1e-10 {
            return 0.0;
        }
        self.return_distribution
            .iter()
            .map(|&x| ((x - mean) / std).powi(3))
            .sum::<f64>()
            / n
    }

    /// Get the kurtosis of the return distribution.
    pub fn kurtosis(&self) -> f64 {
        if self.return_distribution.len() < 4 {
            return 0.0;
        }
        let n = self.return_distribution.len() as f64;
        let mean = self.mean_return;
        let std = self.std_return;
        if std.abs() < 1e-10 {
            return 0.0;
        }
        self.return_distribution
            .iter()
            .map(|&x| ((x - mean) / std).powi(4))
            .sum::<f64>()
            / n
            - 3.0 // Excess kurtosis
    }
}

/// Result from synthetic price path simulation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PricePathResult {
    /// Number of simulations run.
    pub n_simulations: usize,

    /// Mean final price.
    pub mean_final_price: f64,

    /// Confidence interval for final price.
    pub final_price_ci: (f64, f64),

    /// Mean total return.
    pub mean_return: f64,

    /// Confidence interval for return.
    pub return_ci: (f64, f64),

    /// Mean realized volatility.
    pub mean_realized_vol: f64,

    /// Mean maximum drawdown.
    pub mean_max_drawdown: f64,

    /// Distribution of final returns.
    pub return_distribution: Vec<f64>,

    /// Sample paths (if stored).
    pub sample_paths: Option<Vec<Vec<f64>>>,

    /// Price model used.
    pub price_model: PriceModel,
}

/// Result from stress testing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressTestResult {
    /// Scenario tested.
    pub scenario: StressScenario,

    /// Return under stress.
    pub stressed_return: f64,

    /// Max drawdown under stress.
    pub stressed_max_drawdown: f64,

    /// Comparison to base case return.
    pub return_delta: f64,

    /// Comparison to base case drawdown.
    pub drawdown_delta: f64,

    /// Recovery time (periods to recover, if applicable).
    pub recovery_time: Option<usize>,

    /// Number of simulations.
    pub n_simulations: usize,

    /// Additional metrics.
    pub metrics: HashMap<String, f64>,
}

/// Result from parameter perturbation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationResult {
    /// Original parameter values.
    pub original_params: HashMap<String, f64>,

    /// Sensitivity of return to each parameter.
    pub return_sensitivity: HashMap<String, f64>,

    /// Sensitivity of Sharpe to each parameter.
    pub sharpe_sensitivity: HashMap<String, f64>,

    /// Sensitivity of drawdown to each parameter.
    pub drawdown_sensitivity: HashMap<String, f64>,

    /// Most sensitive parameter (by return).
    pub most_sensitive: String,

    /// Number of simulations per parameter.
    pub n_simulations: usize,
}

/// Complete Monte Carlo simulation result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonteCarloResult {
    /// Configuration used.
    pub config: MonteCarloConfig,

    /// Bootstrap results (if run).
    pub bootstrap: Option<BootstrapResult>,

    /// Price path results (if run).
    pub price_paths: Option<PricePathResult>,

    /// Stress test results.
    pub stress_tests: Vec<StressTestResult>,

    /// Perturbation analysis results (if run).
    pub perturbation: Option<PerturbationResult>,

    /// Execution time in seconds.
    pub execution_time_secs: f64,

    /// Timestamp.
    pub timestamp: u64,
}

/// Monte Carlo simulator for trading strategy analysis.
pub struct MonteCarloSimulator {
    /// Configuration.
    config: MonteCarloConfig,

    /// Random number generator.
    rng: StdRng,
}

impl MonteCarloSimulator {
    /// Create a new Monte Carlo simulator.
    pub fn new(config: MonteCarloConfig) -> Result<Self> {
        config.validate()?;
        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };
        Ok(Self { config, rng })
    }

    /// Get the configuration.
    pub fn config(&self) -> &MonteCarloConfig {
        &self.config
    }

    /// Run bootstrap analysis on a sequence of trade returns.
    pub fn bootstrap_trades(&mut self, trades: &[f64]) -> Result<BootstrapResult> {
        if trades.is_empty() {
            return Err(OctaneError::InvalidConfig("No trades provided".into()));
        }

        let n = self.config.n_simulations;
        let n_trades = trades.len();

        // Generate bootstrap samples
        let seeds: Vec<u64> = (0..n).map(|_| self.rng.gen()).collect();

        let results: Vec<(f64, f64, f64)> = if self.config.parallel {
            seeds
                .par_iter()
                .map(|&seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    self.bootstrap_single(trades, n_trades, &mut rng)
                })
                .collect()
        } else {
            seeds
                .iter()
                .map(|&seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    self.bootstrap_single(trades, n_trades, &mut rng)
                })
                .collect()
        };

        let returns: Vec<f64> = results.iter().map(|r| r.0).collect();
        let sharpes: Vec<f64> = results.iter().map(|r| r.1).collect();
        let drawdowns: Vec<f64> = results.iter().map(|r| r.2).collect();

        // Calculate statistics
        let mean_return = mean(&returns);
        let std_return = std_dev(&returns);
        let return_ci = self.confidence_interval(&returns);
        let sharpe_ci = self.confidence_interval(&sharpes);
        let drawdown_ci = self.confidence_interval(&drawdowns);

        let prob_loss = returns.iter().filter(|&&r| r < 0.0).count() as f64 / n as f64;

        // Calculate VaR and CVaR
        let mut var = HashMap::new();
        let mut cvar = HashMap::new();
        if self.config.calculate_var {
            let mut sorted_returns = returns.clone();
            sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            for &level in &self.config.var_levels {
                let var_idx = ((1.0 - level) * n as f64).floor() as usize;
                let var_value = sorted_returns.get(var_idx).copied().unwrap_or(0.0);
                var.insert(format!("VaR_{:.0}", level * 100.0), var_value);

                // CVaR (Expected Shortfall)
                let cvar_value = if var_idx > 0 {
                    sorted_returns[..var_idx].iter().sum::<f64>() / var_idx as f64
                } else {
                    var_value
                };
                cvar.insert(format!("CVaR_{:.0}", level * 100.0), cvar_value);
            }
        }

        Ok(BootstrapResult {
            n_simulations: n,
            mean_return,
            std_return,
            return_ci,
            mean_sharpe: mean(&sharpes),
            sharpe_ci,
            mean_max_drawdown: mean(&drawdowns),
            max_drawdown_ci: drawdown_ci,
            return_distribution: returns,
            drawdown_distribution: drawdowns,
            prob_loss,
            var,
            cvar,
            confidence_level: self.config.confidence_level,
        })
    }

    /// Run a single bootstrap iteration.
    fn bootstrap_single(
        &self,
        trades: &[f64],
        n_trades: usize,
        rng: &mut StdRng,
    ) -> (f64, f64, f64) {
        let uniform = Uniform::new(0, n_trades);
        let mut sampled_trades = Vec::with_capacity(n_trades);
        for _ in 0..n_trades {
            let idx = uniform.sample(rng);
            sampled_trades.push(trades[idx]);
        }

        let total_return: f64 = sampled_trades.iter().sum();
        let mean_trade = total_return / n_trades as f64;
        let std_trade = std_dev(&sampled_trades);
        let sharpe = if std_trade.abs() < 1e-10 {
            0.0
        } else {
            mean_trade / std_trade * (252.0_f64).sqrt() // Annualized
        };

        let max_dd = calculate_max_drawdown(&sampled_trades);

        (total_return, sharpe, max_dd)
    }

    /// Generate synthetic price paths.
    pub fn generate_price_paths(&mut self) -> Result<PricePathResult> {
        let n = self.config.n_simulations;

        let seeds: Vec<u64> = (0..n).map(|_| self.rng.gen()).collect();

        let paths: Vec<Vec<f64>> = if self.config.parallel {
            seeds
                .par_iter()
                .map(|&seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    self.generate_single_path(&mut rng)
                })
                .collect()
        } else {
            seeds
                .iter()
                .map(|&seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    self.generate_single_path(&mut rng)
                })
                .collect()
        };

        let final_prices: Vec<f64> = paths.iter().map(|p| *p.last().unwrap_or(&0.0)).collect();
        let returns: Vec<f64> = paths
            .iter()
            .map(|p| {
                let first = *p.first().unwrap_or(&1.0);
                let last = *p.last().unwrap_or(&1.0);
                (last - first) / first
            })
            .collect();
        let realized_vols: Vec<f64> = paths.iter().map(|p| calculate_realized_vol(p)).collect();
        let max_dds: Vec<f64> = paths.iter().map(|p| calculate_price_drawdown(p)).collect();

        Ok(PricePathResult {
            n_simulations: n,
            mean_final_price: mean(&final_prices),
            final_price_ci: self.confidence_interval(&final_prices),
            mean_return: mean(&returns),
            return_ci: self.confidence_interval(&returns),
            mean_realized_vol: mean(&realized_vols),
            mean_max_drawdown: mean(&max_dds),
            return_distribution: returns,
            sample_paths: if n <= 100 { Some(paths) } else { None },
            price_model: self.config.price_model.clone(),
        })
    }

    /// Generate a single price path.
    fn generate_single_path(&self, rng: &mut StdRng) -> Vec<f64> {
        let dt = self.config.dt;
        let path_len = self.config.path_length;
        let mut path = Vec::with_capacity(path_len + 1);
        path.push(self.config.initial_price);

        let normal = Normal::new(0.0, 1.0).unwrap();

        match &self.config.price_model {
            PriceModel::GBM { drift, volatility } => {
                let drift_term = (*drift - 0.5 * volatility.powi(2)) * dt;
                let vol_term = volatility * dt.sqrt();

                for _ in 0..path_len {
                    let z: f64 = normal.sample(rng);
                    let last = *path.last().unwrap();
                    let next = last * (drift_term + vol_term * z).exp();
                    path.push(next);
                }
            }
            PriceModel::Heston {
                theta,
                kappa,
                sigma,
                rho,
                v0,
            } => {
                let mut v = *v0;
                for _ in 0..path_len {
                    let z1: f64 = normal.sample(rng);
                    let z2: f64 = normal.sample(rng);
                    let z_v = z1;
                    let z_s = *rho * z1 + (1.0 - rho.powi(2)).sqrt() * z2;

                    let last = *path.last().unwrap();
                    let sqrt_v = v.max(0.0).sqrt();
                    let price_drift = -0.5 * v * dt;
                    let price_diff = sqrt_v * dt.sqrt() * z_s;
                    let next = last * (price_drift + price_diff).exp();
                    path.push(next);

                    // Update variance
                    v = v + kappa * (theta - v) * dt + sigma * sqrt_v * dt.sqrt() * z_v;
                    v = v.max(0.0);
                }
            }
            PriceModel::JumpDiffusion {
                drift,
                volatility,
                jump_intensity,
                jump_mean,
                jump_std,
            } => {
                let poisson_rate = jump_intensity * dt;
                let drift_term = (*drift - 0.5 * volatility.powi(2)) * dt;
                let vol_term = volatility * dt.sqrt();
                let jump_normal = Normal::new(*jump_mean, *jump_std).unwrap();

                for _ in 0..path_len {
                    let z: f64 = normal.sample(rng);
                    let last = *path.last().unwrap();

                    // Poisson jumps (approximation)
                    let n_jumps = if rng.gen::<f64>() < poisson_rate {
                        1
                    } else {
                        0
                    };
                    let jump = if n_jumps > 0 {
                        jump_normal.sample(rng).exp()
                    } else {
                        1.0
                    };

                    let next = last * (drift_term + vol_term * z).exp() * jump;
                    path.push(next);
                }
            }
            PriceModel::MeanReverting {
                mean_level,
                mean_reversion,
                volatility,
            } => {
                for _ in 0..path_len {
                    let z: f64 = normal.sample(rng);
                    let last = *path.last().unwrap();
                    let drift = mean_reversion * (mean_level - last) * dt;
                    let diff = volatility * dt.sqrt() * z;
                    let next = (last + drift + diff).max(0.01); // Ensure positive
                    path.push(next);
                }
            }
            PriceModel::Bootstrap => {
                // Bootstrap requires historical returns - use GBM as fallback
                let drift_term = (0.05 - 0.5 * 0.2_f64.powi(2)) * dt;
                let vol_term = 0.2 * dt.sqrt();
                for _ in 0..path_len {
                    let z: f64 = normal.sample(rng);
                    let last = *path.last().unwrap();
                    let next = last * (drift_term + vol_term * z).exp();
                    path.push(next);
                }
            }
        }

        path
    }

    /// Run stress test for a given scenario.
    pub fn stress_test<F>(
        &mut self,
        scenario: &StressScenario,
        base_return: f64,
        base_drawdown: f64,
        evaluate_fn: F,
    ) -> Result<StressTestResult>
    where
        F: Fn(&StressScenario, &mut StdRng) -> (f64, f64, Option<usize>) + Sync + Send,
    {
        let n = self.config.n_simulations;
        let seeds: Vec<u64> = (0..n).map(|_| self.rng.gen()).collect();

        let results: Vec<(f64, f64, Option<usize>)> = if self.config.parallel {
            seeds
                .par_iter()
                .map(|&seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    evaluate_fn(scenario, &mut rng)
                })
                .collect()
        } else {
            seeds
                .iter()
                .map(|&seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    evaluate_fn(scenario, &mut rng)
                })
                .collect()
        };

        let returns: Vec<f64> = results.iter().map(|r| r.0).collect();
        let drawdowns: Vec<f64> = results.iter().map(|r| r.1).collect();
        let recovery_times: Vec<usize> = results.iter().filter_map(|r| r.2).collect();

        let stressed_return = mean(&returns);
        let stressed_max_drawdown = mean(&drawdowns);
        let recovery_time = if recovery_times.is_empty() {
            None
        } else {
            Some(mean_usize(&recovery_times))
        };

        Ok(StressTestResult {
            scenario: scenario.clone(),
            stressed_return,
            stressed_max_drawdown,
            return_delta: stressed_return - base_return,
            drawdown_delta: stressed_max_drawdown - base_drawdown,
            recovery_time,
            n_simulations: n,
            metrics: HashMap::new(),
        })
    }

    /// Run parameter perturbation analysis.
    pub fn perturbation_analysis<F>(
        &mut self,
        original_params: &HashMap<String, f64>,
        evaluate_fn: F,
    ) -> Result<PerturbationResult>
    where
        F: Fn(&HashMap<String, f64>) -> (f64, f64, f64) + Sync + Send,
    {
        let range = self.config.perturbation_range;

        let mut return_sensitivity = HashMap::new();
        let mut sharpe_sensitivity = HashMap::new();
        let mut drawdown_sensitivity = HashMap::new();

        // Evaluate base case
        let (_base_return, _base_sharpe, _base_drawdown) = evaluate_fn(original_params);

        for (param_name, &base_value) in original_params {
            // Perturb parameter up
            let mut params_up = original_params.clone();
            params_up.insert(param_name.clone(), base_value * (1.0 + range));
            let (ret_up, sharpe_up, dd_up) = evaluate_fn(&params_up);

            // Perturb parameter down
            let mut params_down = original_params.clone();
            params_down.insert(param_name.clone(), base_value * (1.0 - range));
            let (ret_down, sharpe_down, dd_down) = evaluate_fn(&params_down);

            // Calculate sensitivities (central difference)
            let delta_param = 2.0 * range * base_value;
            if delta_param.abs() > 1e-10 {
                return_sensitivity.insert(param_name.clone(), (ret_up - ret_down) / delta_param);
                sharpe_sensitivity
                    .insert(param_name.clone(), (sharpe_up - sharpe_down) / delta_param);
                drawdown_sensitivity.insert(param_name.clone(), (dd_up - dd_down) / delta_param);
            }
        }

        // Find most sensitive parameter
        let most_sensitive = return_sensitivity
            .iter()
            .max_by(|a, b| {
                a.1.abs()
                    .partial_cmp(&b.1.abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(k, _)| k.clone())
            .unwrap_or_default();

        Ok(PerturbationResult {
            original_params: original_params.clone(),
            return_sensitivity,
            sharpe_sensitivity,
            drawdown_sensitivity,
            most_sensitive,
            n_simulations: self.config.n_simulations,
        })
    }

    /// Calculate confidence interval for a set of values.
    fn confidence_interval(&self, values: &[f64]) -> (f64, f64) {
        if values.is_empty() {
            return (0.0, 0.0);
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let (lower_pct, upper_pct) = self.config.confidence_level.percentiles();
        let lower_idx = (lower_pct * values.len() as f64).floor() as usize;
        let upper_idx = (upper_pct * values.len() as f64).ceil() as usize;

        let lower = sorted.get(lower_idx).copied().unwrap_or(0.0);
        let upper = sorted
            .get(upper_idx.min(sorted.len() - 1))
            .copied()
            .unwrap_or(0.0);

        (lower, upper)
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

fn mean_usize(values: &[usize]) -> usize {
    if values.is_empty() {
        0
    } else {
        values.iter().sum::<usize>() / values.len()
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

fn calculate_max_drawdown(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }

    let mut equity: f64 = 1.0;
    let mut peak: f64 = 1.0;
    let mut max_dd: f64 = 0.0;

    for &ret in returns {
        equity *= 1.0 + ret;
        peak = peak.max(equity);
        let dd = (peak - equity) / peak;
        max_dd = max_dd.max(dd);
    }

    max_dd
}

fn calculate_price_drawdown(prices: &[f64]) -> f64 {
    if prices.is_empty() {
        return 0.0;
    }

    let mut peak: f64 = prices[0];
    let mut max_dd: f64 = 0.0;

    for &price in prices {
        peak = peak.max(price);
        let dd = (peak - price) / peak;
        max_dd = max_dd.max(dd);
    }

    max_dd
}

fn calculate_realized_vol(prices: &[f64]) -> f64 {
    if prices.len() < 2 {
        return 0.0;
    }

    let returns: Vec<f64> = prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect();

    std_dev(&returns) * (252.0_f64).sqrt() // Annualize
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
    fn test_monte_carlo_config_defaults() {
        let config = MonteCarloConfig::new();
        assert_eq!(config.n_simulations, 10000);
        assert_eq!(config.confidence_level, ConfidenceLevel::P95);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_monte_carlo_config_builder() {
        let config = MonteCarloConfig::new()
            .n_simulations(5000)
            .confidence_level(ConfidenceLevel::P99)
            .seed(42)
            .perturbation_range(0.2);

        assert_eq!(config.n_simulations, 5000);
        assert_eq!(config.confidence_level, ConfidenceLevel::P99);
        assert_eq!(config.seed, Some(42));
        assert!((config.perturbation_range - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_bootstrap_trades() {
        let config = MonteCarloConfig::new().n_simulations(1000).seed(42);
        let mut simulator = MonteCarloSimulator::new(config).unwrap();

        let trades = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.008, -0.003, 0.012];
        let result = simulator.bootstrap_trades(&trades).unwrap();

        assert_eq!(result.n_simulations, 1000);
        assert!(result.mean_return > 0.0); // Expect positive mean with positive trades
        assert!(result.return_ci.0 < result.mean_return);
        assert!(result.return_ci.1 > result.mean_return);
    }

    #[test]
    fn test_price_path_generation_gbm() {
        let config = MonteCarloConfig::new()
            .n_simulations(100)
            .synthetic_prices(true)
            .price_model(PriceModel::GBM {
                drift: 0.1,
                volatility: 0.2,
            })
            .path_length(100)
            .seed(42);

        let mut simulator = MonteCarloSimulator::new(config).unwrap();
        let result = simulator.generate_price_paths().unwrap();

        assert_eq!(result.n_simulations, 100);
        assert!(result.mean_final_price > 0.0);
        assert!(result.sample_paths.is_some());
    }

    #[test]
    fn test_price_path_generation_heston() {
        let config = MonteCarloConfig::new()
            .n_simulations(50)
            .price_model(PriceModel::Heston {
                theta: 0.04,
                kappa: 2.0,
                sigma: 0.3,
                rho: -0.7,
                v0: 0.04,
            })
            .path_length(50)
            .seed(42);

        let mut simulator = MonteCarloSimulator::new(config).unwrap();
        let result = simulator.generate_price_paths().unwrap();

        assert_eq!(result.n_simulations, 50);
        assert!(result.mean_final_price > 0.0);
    }

    #[test]
    fn test_confidence_levels() {
        assert_eq!(ConfidenceLevel::P90.percentiles(), (0.05, 0.95));
        assert_eq!(ConfidenceLevel::P95.percentiles(), (0.025, 0.975));
        assert_eq!(ConfidenceLevel::P99.percentiles(), (0.005, 0.995));
    }

    #[test]
    fn test_max_drawdown() {
        let returns = vec![0.1, -0.2, 0.15, -0.05, 0.1];
        let dd = calculate_max_drawdown(&returns);
        assert!(dd > 0.0);
        assert!(dd <= 1.0);
    }

    #[test]
    fn test_price_drawdown() {
        let prices = vec![100.0, 110.0, 105.0, 95.0, 100.0, 115.0];
        let dd = calculate_price_drawdown(&prices);
        // Max drawdown occurs from peak 110 to trough 95
        let expected_dd = (110.0 - 95.0) / 110.0;
        assert!((dd - expected_dd).abs() < 0.01);
    }

    #[test]
    fn test_realized_vol() {
        let prices: Vec<f64> = (0..100).map(|i| 100.0 + (i as f64 * 0.01)).collect();
        let vol = calculate_realized_vol(&prices);
        assert!(vol >= 0.0);
    }

    #[test]
    fn test_var_cvar_calculation() {
        let config = MonteCarloConfig::new()
            .n_simulations(1000)
            .calculate_var(true)
            .var_levels(vec![0.95, 0.99])
            .seed(42);

        let mut simulator = MonteCarloSimulator::new(config).unwrap();
        let trades = vec![0.02, -0.03, 0.01, -0.02, 0.015, -0.01, 0.005, -0.015];
        let result = simulator.bootstrap_trades(&trades).unwrap();

        assert!(result.var.contains_key("VaR_95"));
        assert!(result.var.contains_key("VaR_99"));
        assert!(result.cvar.contains_key("CVaR_95"));
        assert!(result.cvar.contains_key("CVaR_99"));
    }

    #[test]
    fn test_bootstrap_result_statistics() {
        let result = BootstrapResult {
            n_simulations: 1000,
            mean_return: 0.05,
            std_return: 0.15,
            return_ci: (0.01, 0.09),
            mean_sharpe: 0.8,
            sharpe_ci: (0.5, 1.1),
            mean_max_drawdown: 0.12,
            max_drawdown_ci: (0.05, 0.2),
            return_distribution: vec![0.02, 0.05, 0.08, 0.03, 0.06],
            drawdown_distribution: vec![0.1, 0.12, 0.15, 0.08, 0.11],
            prob_loss: 0.2,
            var: HashMap::new(),
            cvar: HashMap::new(),
            confidence_level: ConfidenceLevel::P95,
        };

        // Test skewness and kurtosis calculations
        let skew = result.skewness();
        let kurt = result.kurtosis();
        assert!(skew.is_finite());
        assert!(kurt.is_finite());
    }
}
