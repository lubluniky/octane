//! Multi-asset portfolio trading environment.
//!
//! Features:
//! - Portfolio of N assets traded simultaneously
//! - Correlation matrix between assets for realistic simulation
//! - Rebalancing actions
//! - Cross-asset position limits
//! - Portfolio-level metrics (Sharpe, max drawdown, etc.)

use crate::core::{Device, OctaneError, Result};
use crate::envs::{BoxSpace, Environment, ObsType, StepInfo, StepResult};
use crate::trading::{AdvancedMarketData, CommissionModel, OrderBook, OrderSide, SlippageModel};
use candle_core::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Portfolio action for multi-asset environment.
#[derive(Debug, Clone)]
pub struct PortfolioAction {
    /// Target weights for each asset (should sum to <= 1.0).
    pub target_weights: Vec<f32>,
    /// Whether to rebalance.
    pub rebalance: bool,
}

impl PortfolioAction {
    /// Create from weight vector.
    pub fn from_weights(weights: Vec<f32>) -> Self {
        Self {
            target_weights: weights,
            rebalance: true,
        }
    }

    /// Normalize weights to sum to 1.0 (for long-only).
    pub fn normalize(&mut self) {
        let sum: f32 = self.target_weights.iter().map(|w| w.max(0.0)).sum();
        if sum > 0.0 {
            for w in &mut self.target_weights {
                *w = (*w).max(0.0) / sum;
            }
        }
    }
}

/// Portfolio state tracking.
#[derive(Debug, Clone, Default)]
pub struct PortfolioState {
    /// Current weights for each asset.
    pub weights: Vec<f32>,
    /// Current positions (in units) for each asset.
    pub positions: Vec<f32>,
    /// Entry prices for each position.
    pub entry_prices: Vec<f32>,
    /// Unrealized PnL per asset.
    pub unrealized_pnl: Vec<f32>,
    /// Realized PnL per asset.
    pub realized_pnl: Vec<f32>,
    /// Total portfolio value.
    pub total_value: f32,
    /// Cash balance.
    pub cash: f32,
}

impl PortfolioState {
    /// Create new portfolio state for N assets.
    pub fn new(num_assets: usize) -> Self {
        Self {
            weights: vec![0.0; num_assets],
            positions: vec![0.0; num_assets],
            entry_prices: vec![0.0; num_assets],
            unrealized_pnl: vec![0.0; num_assets],
            realized_pnl: vec![0.0; num_assets],
            total_value: 0.0,
            cash: 0.0,
        }
    }

    /// Update portfolio values based on current prices.
    pub fn update_values(&mut self, prices: &[f32]) {
        let mut total_position_value = 0.0;

        for i in 0..self.positions.len() {
            let position_value = self.positions[i] * prices[i];
            total_position_value += position_value;

            // Update unrealized PnL
            if self.positions[i].abs() > 0.0001 {
                self.unrealized_pnl[i] = (prices[i] - self.entry_prices[i]) * self.positions[i];
            } else {
                self.unrealized_pnl[i] = 0.0;
            }
        }

        self.total_value = self.cash + total_position_value;

        // Update weights
        if self.total_value > 0.0 {
            for i in 0..self.positions.len() {
                self.weights[i] = (self.positions[i] * prices[i]) / self.total_value;
            }
        }
    }
}

/// Portfolio-level performance metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PortfolioMetrics {
    /// Total return.
    pub total_return: f32,
    /// Annualized return.
    pub annualized_return: f32,
    /// Sharpe ratio.
    pub sharpe_ratio: f32,
    /// Sortino ratio.
    pub sortino_ratio: f32,
    /// Maximum drawdown.
    pub max_drawdown: f32,
    /// Calmar ratio.
    pub calmar_ratio: f32,
    /// Win rate.
    pub win_rate: f32,
    /// Profit factor.
    pub profit_factor: f32,
    /// Number of trades.
    pub num_trades: usize,
    /// Average trade return.
    pub avg_trade_return: f32,
    /// Portfolio turnover.
    pub turnover: f32,
}

impl PortfolioMetrics {
    /// Compute metrics from return history.
    pub fn compute(returns: &[f32], risk_free_rate: f32, periods_per_year: f32) -> Self {
        if returns.is_empty() {
            return Self::default();
        }

        let n = returns.len() as f32;

        // Total return (cumulative)
        let total_return: f32 = returns.iter().map(|r| 1.0 + r).product::<f32>() - 1.0;

        // Mean return
        let mean_return: f32 = returns.iter().sum::<f32>() / n;

        // Annualized return
        let annualized_return = (1.0 + mean_return).powf(periods_per_year) - 1.0;

        // Standard deviation
        let variance: f32 = returns.iter().map(|r| (r - mean_return).powi(2)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        // Downside deviation (for Sortino)
        let downside_returns: Vec<f32> = returns.iter().filter(|&&r| r < 0.0).copied().collect();
        let downside_variance: f32 = if downside_returns.is_empty() {
            0.0
        } else {
            downside_returns.iter().map(|r| r.powi(2)).sum::<f32>() / downside_returns.len() as f32
        };
        let downside_dev = downside_variance.sqrt();

        // Sharpe ratio (annualized)
        let excess_return = mean_return - risk_free_rate / periods_per_year;
        let sharpe_ratio = if std_dev > 0.0 {
            excess_return / std_dev * periods_per_year.sqrt()
        } else {
            0.0
        };

        // Sortino ratio
        let sortino_ratio = if downside_dev > 0.0 {
            excess_return / downside_dev * periods_per_year.sqrt()
        } else {
            0.0
        };

        // Maximum drawdown
        let mut peak = 1.0f32;
        let mut max_dd = 0.0f32;
        let mut cumulative = 1.0f32;

        for &r in returns {
            cumulative *= 1.0 + r;
            peak = peak.max(cumulative);
            let dd = (peak - cumulative) / peak;
            max_dd = max_dd.max(dd);
        }

        // Calmar ratio
        let calmar_ratio = if max_dd > 0.0 {
            annualized_return / max_dd
        } else {
            0.0
        };

        // Win rate
        let wins = returns.iter().filter(|&&r| r > 0.0).count();
        let win_rate = wins as f32 / n;

        // Profit factor
        let gross_profit: f32 = returns.iter().filter(|&&r| r > 0.0).sum();
        let gross_loss: f32 = returns.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();
        let profit_factor = if gross_loss > 0.0 {
            gross_profit / gross_loss
        } else if gross_profit > 0.0 {
            f32::INFINITY
        } else {
            0.0
        };

        Self {
            total_return,
            annualized_return,
            sharpe_ratio,
            sortino_ratio,
            max_drawdown: max_dd,
            calmar_ratio,
            win_rate,
            profit_factor,
            num_trades: 0, // Updated externally
            avg_trade_return: mean_return,
            turnover: 0.0, // Updated externally
        }
    }
}

/// Configuration for multi-asset environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiAssetConfig {
    /// Number of assets in portfolio.
    pub num_assets: usize,
    /// Asset names/symbols.
    pub asset_names: Vec<String>,
    /// Initial cash balance.
    pub initial_balance: f32,
    /// Slippage model.
    pub slippage_model: SlippageModel,
    /// Commission model.
    pub commission_model: CommissionModel,
    /// Maximum weight per asset.
    pub max_weight_per_asset: f32,
    /// Maximum total gross exposure (sum of absolute weights).
    pub max_gross_exposure: f32,
    /// Maximum total net exposure (sum of weights).
    pub max_net_exposure: f32,
    /// Allow short positions.
    pub allow_short: bool,
    /// Lookback window for observations.
    pub lookback: usize,
    /// Episode length.
    pub episode_length: usize,
    /// Order book depth per asset.
    pub orderbook_depth: usize,
    /// Rebalancing threshold (minimum weight change to trigger trade).
    pub rebalance_threshold: f32,
    /// Correlation matrix between assets (flattened, row-major).
    pub correlation_matrix: Option<Vec<f32>>,
    /// Risk-free rate for Sharpe calculation.
    pub risk_free_rate: f32,
    /// Trading periods per year (252 for daily, etc.).
    pub periods_per_year: f32,
}

impl Default for MultiAssetConfig {
    fn default() -> Self {
        Self {
            num_assets: 4,
            asset_names: vec![
                "ASSET_A".into(),
                "ASSET_B".into(),
                "ASSET_C".into(),
                "ASSET_D".into(),
            ],
            initial_balance: 100000.0,
            slippage_model: SlippageModel::default(),
            commission_model: CommissionModel::default(),
            max_weight_per_asset: 0.4,
            max_gross_exposure: 1.5,
            max_net_exposure: 1.0,
            allow_short: true,
            lookback: 20,
            episode_length: 252,
            orderbook_depth: 3,
            rebalance_threshold: 0.01,
            correlation_matrix: None,
            risk_free_rate: 0.02,
            periods_per_year: 252.0,
        }
    }
}

impl MultiAssetConfig {
    /// Set number of assets.
    pub fn num_assets(mut self, n: usize) -> Self {
        self.num_assets = n;
        self
    }

    /// Set asset names.
    pub fn asset_names(mut self, names: Vec<String>) -> Self {
        self.asset_names = names;
        self.num_assets = self.asset_names.len();
        self
    }

    /// Set initial balance.
    pub fn initial_balance(mut self, balance: f32) -> Self {
        self.initial_balance = balance;
        self
    }

    /// Set slippage model.
    pub fn slippage_model(mut self, model: SlippageModel) -> Self {
        self.slippage_model = model;
        self
    }

    /// Set commission model.
    pub fn commission_model(mut self, model: CommissionModel) -> Self {
        self.commission_model = model;
        self
    }

    /// Set maximum weight per asset.
    pub fn max_weight_per_asset(mut self, max: f32) -> Self {
        self.max_weight_per_asset = max;
        self
    }

    /// Set maximum gross exposure.
    pub fn max_gross_exposure(mut self, max: f32) -> Self {
        self.max_gross_exposure = max;
        self
    }

    /// Set correlation matrix.
    pub fn correlation_matrix(mut self, matrix: Vec<f32>) -> Self {
        self.correlation_matrix = Some(matrix);
        self
    }

    /// Enable or disable short selling.
    pub fn allow_short(mut self, allow: bool) -> Self {
        self.allow_short = allow;
        self
    }

    /// Set lookback window.
    pub fn lookback(mut self, lookback: usize) -> Self {
        self.lookback = lookback;
        self
    }

    /// Set episode length.
    pub fn episode_length(mut self, length: usize) -> Self {
        self.episode_length = length;
        self
    }
}

/// Market data for multiple assets.
#[derive(Debug, Clone)]
pub struct MultiAssetMarketData {
    /// Market data per asset.
    pub assets: Vec<AdvancedMarketData>,
    /// Asset names.
    pub asset_names: Vec<String>,
    /// Correlation matrix (num_assets x num_assets).
    pub correlation_matrix: Vec<Vec<f32>>,
}

impl MultiAssetMarketData {
    /// Create synthetic correlated market data.
    pub fn synthetic_correlated(
        num_assets: usize,
        timesteps: usize,
        correlation: f32,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut assets = Vec::with_capacity(num_assets);
        let asset_names: Vec<String> = (0..num_assets).map(|i| format!("ASSET_{}", i)).collect();

        // Generate base random walk
        let mut base_returns = Vec::with_capacity(timesteps);
        for _ in 0..timesteps {
            base_returns.push(rng.gen::<f32>() * 0.04 - 0.02);
        }

        // Generate correlated asset data
        for i in 0..num_assets {
            let mut price = 100.0 + (i as f32) * 50.0; // Different starting prices
            let mut prices = Vec::with_capacity(timesteps);
            let mut timestamps = Vec::with_capacity(timesteps);
            let mut volumes = Vec::with_capacity(timesteps);

            let base_timestamp = 1704067200000u64;

            for t in 0..timesteps {
                // Correlated returns: mix of base and idiosyncratic
                let idio_return = rng.gen::<f32>() * 0.04 - 0.02;
                let returns = correlation * base_returns[t] + (1.0 - correlation) * idio_return;
                price *= 1.0 + returns;

                let volatility = 0.02 + rng.gen::<f32>() * 0.02;
                let open = price * (1.0 + rng.gen::<f32>() * 0.01 - 0.005);
                let high = price * (1.0 + rng.gen::<f32>() * 0.02);
                let low = price * (1.0 - rng.gen::<f32>() * 0.02);
                let close = price;
                let volume = (rng.gen::<f32>() * 1000.0 + 500.0) * (1.0 + i as f32 * 0.5);

                let sma_ratio = 1.0 + rng.gen::<f32>() * 0.1 - 0.05;
                let rsi = rng.gen::<f32>() * 100.0;

                prices.push(vec![
                    open, high, low, close, volume, sma_ratio, rsi, volatility,
                ]);
                timestamps.push(base_timestamp + (t as u64) * 60000);
                volumes.push(volume);
            }

            assets.push(AdvancedMarketData {
                prices,
                feature_names: vec![
                    "open".into(),
                    "high".into(),
                    "low".into(),
                    "close".into(),
                    "volume".into(),
                    "sma_ratio".into(),
                    "rsi".into(),
                    "volatility".into(),
                ],
                timestamps,
                volumes,
            });
        }

        // Build correlation matrix
        let mut correlation_matrix = vec![vec![0.0; num_assets]; num_assets];
        for i in 0..num_assets {
            for j in 0..num_assets {
                if i == j {
                    correlation_matrix[i][j] = 1.0;
                } else {
                    correlation_matrix[i][j] = correlation;
                }
            }
        }

        Self {
            assets,
            asset_names,
            correlation_matrix,
        }
    }

    /// Number of timesteps.
    pub fn len(&self) -> usize {
        self.assets.first().map(|a| a.len()).unwrap_or(0)
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.assets.is_empty() || self.assets[0].is_empty()
    }

    /// Number of assets.
    pub fn num_assets(&self) -> usize {
        self.assets.len()
    }
}

/// Multi-asset portfolio trading environment.
#[derive(Clone)]
pub struct MultiAssetEnv {
    /// Market data.
    data: MultiAssetMarketData,
    /// Configuration.
    config: MultiAssetConfig,
    /// Current timestep.
    current_step: usize,
    /// Start index.
    start_idx: usize,
    /// Portfolio state.
    portfolio: PortfolioState,
    /// Order books per asset.
    order_books: Vec<OrderBook>,
    /// Return history for metrics calculation.
    return_history: Vec<f32>,
    /// Initial portfolio value.
    initial_value: f32,
    /// Peak portfolio value (for drawdown).
    peak_value: f32,
    /// Total turnover.
    total_turnover: f32,
    /// Number of trades.
    num_trades: usize,
    /// Observation space.
    obs_space: BoxSpace,
    /// Action space.
    act_space: BoxSpace,
    /// Random number generator.
    rng: StdRng,
}

impl MultiAssetEnv {
    /// Create a new multi-asset environment.
    pub fn new(data: MultiAssetMarketData) -> Result<Self> {
        let config = MultiAssetConfig::default().num_assets(data.num_assets());
        Self::with_config(data, config)
    }

    /// Create with custom configuration.
    pub fn with_config(data: MultiAssetMarketData, config: MultiAssetConfig) -> Result<Self> {
        if data.num_assets() != config.num_assets {
            return Err(OctaneError::InvalidConfig(format!(
                "Data has {} assets but config expects {}",
                data.num_assets(),
                config.num_assets
            )));
        }

        if data.len() < config.lookback + config.episode_length {
            return Err(OctaneError::InvalidConfig(format!(
                "Data length {} too short for lookback {} + episode_length {}",
                data.len(),
                config.lookback,
                config.episode_length
            )));
        }

        let num_assets = config.num_assets;

        // Observation space:
        // Per asset: lookback * features + orderbook
        // Portfolio: weights + cash_ratio + metrics
        let features_per_asset = 8; // OHLCV + indicators
        let orderbook_per_asset = config.orderbook_depth * 4;
        let asset_obs_dim = config.lookback * features_per_asset + orderbook_per_asset;
        let portfolio_obs_dim = num_assets * 2 + 5; // weights, unrealized_pnl, cash, metrics

        let obs_dim = num_assets * asset_obs_dim + portfolio_obs_dim;
        let obs_space = BoxSpace::unbounded(vec![obs_dim]);

        // Action space: target weight per asset
        let act_dim = num_assets;
        let act_space = if config.allow_short {
            BoxSpace::symmetric(1.0, vec![act_dim])
        } else {
            BoxSpace::new(vec![0.0; act_dim], vec![1.0; act_dim], vec![act_dim])?
        };

        let order_books: Vec<OrderBook> = (0..num_assets)
            .map(|_| OrderBook::new(config.orderbook_depth))
            .collect();

        Ok(Self {
            data,
            config,
            current_step: 0,
            start_idx: 0,
            portfolio: PortfolioState::new(num_assets),
            order_books,
            return_history: Vec::new(),
            initial_value: 0.0,
            peak_value: 0.0,
            total_turnover: 0.0,
            num_trades: 0,
            obs_space,
            act_space,
            rng: StdRng::from_entropy(),
        })
    }

    /// Get current prices for all assets.
    fn current_prices(&self) -> Vec<f32> {
        let idx = (self.start_idx + self.current_step).min(self.data.len() - 1);
        self.data
            .assets
            .iter()
            .map(|a| a.prices[idx][3])
            .collect()
    }

    /// Get current volatilities.
    fn current_volatilities(&self) -> Vec<f32> {
        let idx = (self.start_idx + self.current_step).min(self.data.len() - 1);
        self.data
            .assets
            .iter()
            .map(|a| a.prices[idx][7])
            .collect()
    }

    /// Execute rebalancing trades.
    fn execute_rebalance(&mut self, target_weights: &[f32]) {
        let prices = self.current_prices();
        let total_value = self.portfolio.total_value;

        for i in 0..self.config.num_assets {
            let target_value = target_weights[i] * total_value;
            let current_value = self.portfolio.positions[i] * prices[i];
            let value_delta = target_value - current_value;

            // Check rebalance threshold
            if (value_delta / total_value).abs() < self.config.rebalance_threshold {
                continue;
            }

            let quantity_delta = value_delta / prices[i];
            let side = if quantity_delta > 0.0 {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            };

            // Calculate slippage and commission
            let slippage = self
                .config
                .slippage_model
                .calculate(quantity_delta.abs(), prices[i], side);
            let execution_price = prices[i] + slippage;

            let is_maker = false; // Market orders are taker
            let commission = self
                .config
                .commission_model
                .calculate(quantity_delta.abs(), execution_price, is_maker, 0.0);

            // Update position
            let old_position = self.portfolio.positions[i];
            self.portfolio.positions[i] += quantity_delta;

            // Update entry price
            if old_position.abs() < 0.0001 && quantity_delta.abs() > 0.0001 {
                self.portfolio.entry_prices[i] = execution_price;
            } else if old_position.signum() != self.portfolio.positions[i].signum() {
                // Position flip - realize PnL
                let realized =
                    (execution_price - self.portfolio.entry_prices[i]) * old_position.abs();
                self.portfolio.realized_pnl[i] += realized;
                self.portfolio.entry_prices[i] = execution_price;
            }

            // Update cash
            self.portfolio.cash -= quantity_delta * execution_price + commission;

            // Track turnover and trades
            self.total_turnover += (quantity_delta * execution_price).abs() / total_value;
            self.num_trades += 1;
        }

        // Update portfolio values
        self.portfolio.update_values(&prices);
    }

    /// Constrain weights to respect position limits.
    fn constrain_weights(&self, weights: &mut [f32]) {
        // Clip individual weights
        for w in weights.iter_mut() {
            *w = w.clamp(
                if self.config.allow_short {
                    -self.config.max_weight_per_asset
                } else {
                    0.0
                },
                self.config.max_weight_per_asset,
            );
        }

        // Check gross exposure
        let gross_exposure: f32 = weights.iter().map(|w| w.abs()).sum();
        if gross_exposure > self.config.max_gross_exposure {
            let scale = self.config.max_gross_exposure / gross_exposure;
            for w in weights.iter_mut() {
                *w *= scale;
            }
        }

        // Check net exposure
        let net_exposure: f32 = weights.iter().sum();
        if net_exposure.abs() > self.config.max_net_exposure {
            let adjustment = (net_exposure - net_exposure.signum() * self.config.max_net_exposure)
                / weights.len() as f32;
            for w in weights.iter_mut() {
                *w -= adjustment;
            }
        }
    }

    /// Build observation tensor.
    fn build_observation(&self, device: &Device) -> Result<Tensor> {
        let mut obs = Vec::new();
        let lookback = self.config.lookback;
        let prices = self.current_prices();

        // Per-asset observations
        for (asset_idx, asset_data) in self.data.assets.iter().enumerate() {
            let base_price = prices[asset_idx];

            // Historical features
            let start = self.start_idx + self.current_step.saturating_sub(lookback);
            let end = (self.start_idx + self.current_step).min(asset_data.len());

            for i in start..end {
                for (j, &val) in asset_data.prices[i].iter().enumerate() {
                    let normalized = match j {
                        0..=3 => (val - base_price) / base_price,
                        4 => val / 1000.0,
                        5 => val - 1.0,
                        6 => (val - 50.0) / 50.0,
                        _ => val,
                    };
                    obs.push(normalized);
                }
            }

            // Pad if needed
            let features_per_step = asset_data.num_features();
            while obs.len()
                < (asset_idx + 1) * lookback * features_per_step
                    + asset_idx * self.config.orderbook_depth * 4
            {
                obs.insert(
                    obs.len() - (end - start) * features_per_step,
                    0.0,
                );
            }

            // Order book
            let book_obs = self.order_books[asset_idx].to_observation();
            for (i, val) in book_obs.into_iter().enumerate() {
                let normalized = if i % 2 == 0 {
                    (val - base_price) / base_price
                } else {
                    val / 1000.0
                };
                obs.push(normalized);
            }
        }

        // Portfolio state
        for &w in &self.portfolio.weights {
            obs.push(w);
        }
        for &pnl in &self.portfolio.unrealized_pnl {
            obs.push(pnl / self.config.initial_balance);
        }
        obs.push(self.portfolio.cash / self.config.initial_balance);

        // Portfolio metrics
        let current_return =
            (self.portfolio.total_value - self.initial_value) / self.initial_value;
        let drawdown = (self.peak_value - self.portfolio.total_value) / self.peak_value;
        obs.push(current_return);
        obs.push(drawdown);
        obs.push(self.total_turnover);
        obs.push(self.num_trades as f32 / 100.0);

        let candle_device = device.to_candle()?;
        Ok(Tensor::from_slice(&obs, &[obs.len()], &candle_device)?)
    }
}

impl Environment for MultiAssetEnv {
    type ObsSpace = BoxSpace;
    type ActSpace = BoxSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActSpace {
        &self.act_space
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        let max_start = self.data.len() - self.config.lookback - self.config.episode_length;
        self.start_idx = self.rng.gen_range(0..=max_start);

        self.current_step = self.config.lookback;
        self.portfolio = PortfolioState::new(self.config.num_assets);
        self.portfolio.cash = self.config.initial_balance;
        self.portfolio.total_value = self.config.initial_balance;
        self.return_history.clear();
        self.initial_value = self.config.initial_balance;
        self.peak_value = self.config.initial_balance;
        self.total_turnover = 0.0;
        self.num_trades = 0;

        // Initialize order books
        let prices = self.current_prices();
        let vols = self.current_volatilities();
        for (i, book) in self.order_books.iter_mut().enumerate() {
            book.update(prices[i], 10.0, vols[i], &mut self.rng);
        }

        self.build_observation(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let action_vec: Vec<f32> = action.flatten_all()?.to_vec1()?;

        // Constrain target weights
        let mut target_weights = action_vec.clone();
        self.constrain_weights(&mut target_weights);

        let value_before = self.portfolio.total_value;

        // Execute rebalancing
        self.execute_rebalance(&target_weights);

        // Move to next step
        self.current_step += 1;

        // Update order books
        let new_prices = self.current_prices();
        let new_vols = self.current_volatilities();
        for (i, book) in self.order_books.iter_mut().enumerate() {
            book.update(new_prices[i], 10.0, new_vols[i], &mut self.rng);
        }

        // Update portfolio values
        self.portfolio.update_values(&new_prices);

        // Calculate return and update history
        let step_return = (self.portfolio.total_value - value_before) / value_before;
        self.return_history.push(step_return);

        // Update peak value
        self.peak_value = self.peak_value.max(self.portfolio.total_value);

        // Calculate reward (risk-adjusted return)
        let reward = step_return;

        // Check termination
        let episode_done = self.current_step >= self.config.lookback + self.config.episode_length;
        let bankrupt = self.portfolio.total_value < self.config.initial_balance * 0.5;

        let observation = self.build_observation(device)?;

        let info = if episode_done || bankrupt {
            let metrics = PortfolioMetrics::compute(
                &self.return_history,
                self.config.risk_free_rate,
                self.config.periods_per_year,
            );

            let mut extra = HashMap::new();
            extra.insert("total_return".into(), metrics.total_return * 100.0);
            extra.insert("sharpe_ratio".into(), metrics.sharpe_ratio);
            extra.insert("max_drawdown".into(), metrics.max_drawdown * 100.0);
            extra.insert("sortino_ratio".into(), metrics.sortino_ratio);
            extra.insert("win_rate".into(), metrics.win_rate * 100.0);
            extra.insert("num_trades".into(), self.num_trades as f32);
            extra.insert("turnover".into(), self.total_turnover);
            extra.insert("final_value".into(), self.portfolio.total_value);

            Some(StepInfo {
                episode_return: Some(metrics.total_return),
                episode_length: Some(self.current_step - self.config.lookback),
                extra,
            })
        } else {
            None
        };

        Ok(StepResult {
            observation,
            reward,
            terminated: bankrupt,
            truncated: episode_done && !bankrupt,
            info,
        })
    }

    fn name(&self) -> &str {
        "MultiAssetEnv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_portfolio_metrics() {
        let returns = vec![0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.005, 0.015];
        let metrics = PortfolioMetrics::compute(&returns, 0.02, 252.0);

        assert!(metrics.total_return != 0.0);
        assert!(metrics.sharpe_ratio.is_finite());
        assert!(metrics.max_drawdown >= 0.0 && metrics.max_drawdown <= 1.0);
    }

    #[test]
    fn test_multi_asset_data() {
        let data = MultiAssetMarketData::synthetic_correlated(4, 1000, 0.5, 42);
        assert_eq!(data.num_assets(), 4);
        assert_eq!(data.len(), 1000);
    }

    #[test]
    fn test_env_creation() {
        let data = MultiAssetMarketData::synthetic_correlated(3, 500, 0.5, 42);
        let config = MultiAssetConfig::default()
            .num_assets(3)
            .episode_length(100)
            .lookback(10);
        let env = MultiAssetEnv::with_config(data, config);
        assert!(env.is_ok());
    }
}
