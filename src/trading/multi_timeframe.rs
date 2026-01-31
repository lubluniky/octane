//! Multi-timeframe trading environment.
//!
//! Features:
//! - Agent observes multiple timeframes (1m, 5m, 1h, 1d, etc.)
//! - Hierarchical observation space combining all timeframes
//! - Timeframe alignment and synchronization
//! - Configurable timeframe combinations

use crate::core::{Device, OctaneError, Result};
use crate::envs::{BoxSpace, Environment, ObsType, StepInfo, StepResult};
use crate::trading::{
    AdvancedMarketData, CommissionModel, OrderBook, OrderSide, PositionType, SlippageModel,
};
use candle_core::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported timeframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Timeframe {
    /// 1 minute.
    M1,
    /// 5 minutes.
    M5,
    /// 15 minutes.
    M15,
    /// 30 minutes.
    M30,
    /// 1 hour.
    H1,
    /// 4 hours.
    H4,
    /// 1 day.
    D1,
    /// 1 week.
    W1,
}

impl Timeframe {
    /// Get the number of base periods (M1) in this timeframe.
    pub fn periods(&self) -> usize {
        match self {
            Timeframe::M1 => 1,
            Timeframe::M5 => 5,
            Timeframe::M15 => 15,
            Timeframe::M30 => 30,
            Timeframe::H1 => 60,
            Timeframe::H4 => 240,
            Timeframe::D1 => 1440,
            Timeframe::W1 => 10080,
        }
    }

    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            Timeframe::M1 => "1m",
            Timeframe::M5 => "5m",
            Timeframe::M15 => "15m",
            Timeframe::M30 => "30m",
            Timeframe::H1 => "1h",
            Timeframe::H4 => "4h",
            Timeframe::D1 => "1d",
            Timeframe::W1 => "1w",
        }
    }
}

/// Data aggregated to a specific timeframe.
#[derive(Debug, Clone)]
pub struct TimeframeData {
    /// Timeframe.
    pub timeframe: Timeframe,
    /// Aggregated OHLCV data.
    pub bars: Vec<TimeframeBar>,
}

/// A single bar at a specific timeframe.
#[derive(Debug, Clone, Default)]
pub struct TimeframeBar {
    /// Open price.
    pub open: f32,
    /// High price.
    pub high: f32,
    /// Low price.
    pub low: f32,
    /// Close price.
    pub close: f32,
    /// Volume.
    pub volume: f32,
    /// Additional features (technical indicators).
    pub features: Vec<f32>,
    /// Timestamp (start of bar).
    pub timestamp: u64,
}

impl TimeframeBar {
    /// Convert to feature vector.
    pub fn to_features(&self) -> Vec<f32> {
        let mut features = vec![
            self.open, self.high, self.low, self.close, self.volume,
        ];
        features.extend(&self.features);
        features
    }

    /// Number of features.
    pub fn num_features(&self) -> usize {
        5 + self.features.len()
    }
}

impl TimeframeData {
    /// Aggregate base (M1) data to this timeframe.
    pub fn aggregate_from_base(
        base_data: &AdvancedMarketData,
        timeframe: Timeframe,
    ) -> Result<Self> {
        let period = timeframe.periods();

        if base_data.len() < period {
            return Err(OctaneError::Environment(format!(
                "Not enough data to aggregate to {}",
                timeframe.name()
            )));
        }

        let num_bars = base_data.len() / period;
        let mut bars = Vec::with_capacity(num_bars);

        for i in 0..num_bars {
            let start = i * period;
            let end = start + period;

            let open = base_data.prices[start][0];
            let mut high = f32::NEG_INFINITY;
            let mut low = f32::INFINITY;
            let mut volume = 0.0;

            for j in start..end {
                high = high.max(base_data.prices[j][1]);
                low = low.min(base_data.prices[j][2]);
                volume += base_data.prices[j][4];
            }

            let close = base_data.prices[end - 1][3];

            // Calculate additional features (SMA ratio, RSI, volatility)
            let returns: Vec<f32> = (start + 1..end)
                .map(|j| {
                    (base_data.prices[j][3] - base_data.prices[j - 1][3])
                        / base_data.prices[j - 1][3]
                })
                .collect();

            let avg_return = if returns.is_empty() {
                0.0
            } else {
                returns.iter().sum::<f32>() / returns.len() as f32
            };

            let volatility = if returns.len() > 1 {
                let var: f32 =
                    returns.iter().map(|r| (r - avg_return).powi(2)).sum::<f32>() / returns.len() as f32;
                var.sqrt()
            } else {
                0.0
            };

            // Simplified RSI calculation
            let gains: f32 = returns.iter().filter(|&&r| r > 0.0).sum();
            let losses: f32 = returns.iter().filter(|&&r| r < 0.0).map(|r| -r).sum();
            let rsi = if losses > 0.0 {
                100.0 - 100.0 / (1.0 + gains / losses)
            } else if gains > 0.0 {
                100.0
            } else {
                50.0
            };

            // SMA ratio (current close vs average close in window)
            let avg_close: f32 = (start..end)
                .map(|j| base_data.prices[j][3])
                .sum::<f32>()
                / period as f32;
            let sma_ratio = close / avg_close;

            bars.push(TimeframeBar {
                open,
                high,
                low,
                close,
                volume,
                features: vec![sma_ratio, rsi, volatility],
                timestamp: base_data.timestamps[start],
            });
        }

        Ok(Self { timeframe, bars })
    }

    /// Get bar at index (or None if out of bounds).
    pub fn get(&self, idx: usize) -> Option<&TimeframeBar> {
        self.bars.get(idx)
    }

    /// Number of bars.
    pub fn len(&self) -> usize {
        self.bars.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.bars.is_empty()
    }
}

/// Synchronizes multiple timeframes to ensure alignment.
#[derive(Debug, Clone)]
pub struct TimeframeSynchronizer {
    /// Base timeframe (smallest).
    base_timeframe: Timeframe,
    /// All timeframes.
    timeframes: Vec<Timeframe>,
    /// Mapping from base index to higher timeframe indices.
    index_mapping: HashMap<Timeframe, Vec<usize>>,
}

impl TimeframeSynchronizer {
    /// Create new synchronizer.
    pub fn new(timeframes: Vec<Timeframe>) -> Self {
        let base_timeframe = *timeframes.iter().min_by_key(|t| t.periods()).unwrap();
        Self {
            base_timeframe,
            timeframes,
            index_mapping: HashMap::new(),
        }
    }

    /// Build index mapping for given data length.
    pub fn build_mapping(&mut self, base_length: usize) {
        self.index_mapping.clear();

        for &tf in &self.timeframes {
            let period = tf.periods();
            let tf_indices: Vec<usize> = (0..base_length).map(|i| i / period).collect();
            self.index_mapping.insert(tf, tf_indices);
        }
    }

    /// Get the index in a specific timeframe data for a given base index.
    pub fn get_index(&self, base_idx: usize, timeframe: Timeframe) -> usize {
        self.index_mapping
            .get(&timeframe)
            .and_then(|m| m.get(base_idx).copied())
            .unwrap_or(0)
    }

    /// Check if a bar boundary for a timeframe.
    pub fn is_bar_boundary(&self, base_idx: usize, timeframe: Timeframe) -> bool {
        base_idx % timeframe.periods() == 0
    }
}

/// Configuration for multi-timeframe environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiTimeframeConfig {
    /// Timeframes to observe (from smallest to largest).
    pub timeframes: Vec<Timeframe>,
    /// Lookback bars per timeframe.
    pub lookback_per_timeframe: HashMap<Timeframe, usize>,
    /// Initial cash balance.
    pub initial_balance: f32,
    /// Slippage model.
    pub slippage_model: SlippageModel,
    /// Commission model.
    pub commission_model: CommissionModel,
    /// Maximum position size.
    pub max_position: f32,
    /// Episode length (in base timeframe steps).
    pub episode_length: usize,
    /// Order book depth.
    pub orderbook_depth: usize,
    /// Allow short positions.
    pub allow_short: bool,
    /// Execution timeframe (which timeframe triggers trades).
    pub execution_timeframe: Timeframe,
}

impl Default for MultiTimeframeConfig {
    fn default() -> Self {
        let mut lookback = HashMap::new();
        lookback.insert(Timeframe::M1, 60);
        lookback.insert(Timeframe::M5, 24);
        lookback.insert(Timeframe::H1, 24);
        lookback.insert(Timeframe::D1, 20);

        Self {
            timeframes: vec![Timeframe::M1, Timeframe::M5, Timeframe::H1, Timeframe::D1],
            lookback_per_timeframe: lookback,
            initial_balance: 10000.0,
            slippage_model: SlippageModel::default(),
            commission_model: CommissionModel::default(),
            max_position: 1.0,
            episode_length: 1440, // 1 day of 1-minute bars
            orderbook_depth: 5,
            allow_short: true,
            execution_timeframe: Timeframe::M1,
        }
    }
}

impl MultiTimeframeConfig {
    /// Set timeframes.
    pub fn timeframes(mut self, timeframes: Vec<Timeframe>) -> Self {
        self.timeframes = timeframes;
        self
    }

    /// Set lookback for a specific timeframe.
    pub fn lookback(mut self, timeframe: Timeframe, lookback: usize) -> Self {
        self.lookback_per_timeframe.insert(timeframe, lookback);
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

    /// Set maximum position.
    pub fn max_position(mut self, pos: f32) -> Self {
        self.max_position = pos;
        self
    }

    /// Set episode length.
    pub fn episode_length(mut self, length: usize) -> Self {
        self.episode_length = length;
        self
    }

    /// Set execution timeframe.
    pub fn execution_timeframe(mut self, tf: Timeframe) -> Self {
        self.execution_timeframe = tf;
        self
    }

    /// Enable or disable short selling.
    pub fn allow_short(mut self, allow: bool) -> Self {
        self.allow_short = allow;
        self
    }

    /// Get total observation dimension.
    pub fn observation_dim(&self, features_per_bar: usize, orderbook_depth: usize) -> usize {
        let mut dim = 0;

        // Per timeframe: lookback * features
        for tf in &self.timeframes {
            let lookback = self.lookback_per_timeframe.get(tf).copied().unwrap_or(20);
            dim += lookback * features_per_bar;
        }

        // Order book
        dim += orderbook_depth * 4;

        // Position state
        dim += 4;

        dim
    }
}

/// Position state for the multi-timeframe environment.
#[derive(Debug, Clone, Default)]
struct MTFPositionState {
    position: f32,
    position_type: PositionType,
    entry_price: f32,
    unrealized_pnl: f32,
    realized_pnl: f32,
}

/// Multi-timeframe trading environment.
#[derive(Clone)]
pub struct MultiTimeframeEnv {
    /// Base (M1) market data.
    base_data: AdvancedMarketData,
    /// Aggregated timeframe data.
    timeframe_data: HashMap<Timeframe, TimeframeData>,
    /// Configuration.
    config: MultiTimeframeConfig,
    /// Timeframe synchronizer.
    synchronizer: TimeframeSynchronizer,
    /// Current step (in base timeframe).
    current_step: usize,
    /// Start index.
    start_idx: usize,
    /// Cash balance.
    balance: f32,
    /// Position state.
    position_state: MTFPositionState,
    /// Order book.
    order_book: OrderBook,
    /// Initial portfolio value.
    initial_value: f32,
    /// Minimum start index (based on largest timeframe lookback).
    min_start_idx: usize,
    /// Observation space.
    obs_space: BoxSpace,
    /// Action space.
    act_space: BoxSpace,
    /// Random number generator.
    rng: StdRng,
}

impl MultiTimeframeEnv {
    /// Create new multi-timeframe environment.
    pub fn new(base_data: AdvancedMarketData) -> Result<Self> {
        Self::with_config(base_data, MultiTimeframeConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(
        base_data: AdvancedMarketData,
        config: MultiTimeframeConfig,
    ) -> Result<Self> {
        // Aggregate data for all timeframes
        let mut timeframe_data = HashMap::new();
        for &tf in &config.timeframes {
            let data = TimeframeData::aggregate_from_base(&base_data, tf)?;
            timeframe_data.insert(tf, data);
        }

        // Build synchronizer
        let mut synchronizer = TimeframeSynchronizer::new(config.timeframes.clone());
        synchronizer.build_mapping(base_data.len());

        // Calculate minimum start index (need enough history for all timeframes)
        let mut min_start_idx = 0;
        for tf in &config.timeframes {
            let lookback = config.lookback_per_timeframe.get(tf).copied().unwrap_or(20);
            let min_for_tf = lookback * tf.periods();
            min_start_idx = min_start_idx.max(min_for_tf);
        }

        // Validate data length
        if base_data.len() < min_start_idx + config.episode_length {
            return Err(OctaneError::InvalidConfig(format!(
                "Data length {} too short for min_start {} + episode_length {}",
                base_data.len(),
                min_start_idx,
                config.episode_length
            )));
        }

        // Calculate observation dimension
        let features_per_bar = 8; // OHLCV + 3 indicators
        let obs_dim = config.observation_dim(features_per_bar, config.orderbook_depth);
        let obs_space = BoxSpace::unbounded(vec![obs_dim]);

        // Action space
        let act_space = BoxSpace::symmetric(1.0, vec![1]);

        let orderbook_depth = config.orderbook_depth;
        Ok(Self {
            base_data,
            timeframe_data,
            config,
            synchronizer,
            current_step: 0,
            start_idx: 0,
            balance: 0.0,
            position_state: MTFPositionState::default(),
            order_book: OrderBook::new(orderbook_depth),
            initial_value: 0.0,
            min_start_idx,
            obs_space,
            act_space,
            rng: StdRng::from_entropy(),
        })
    }

    /// Get current base price.
    fn current_price(&self) -> f32 {
        let idx = (self.start_idx + self.current_step).min(self.base_data.len() - 1);
        self.base_data.prices[idx][3]
    }

    /// Get current volatility.
    fn current_volatility(&self) -> f32 {
        let idx = (self.start_idx + self.current_step).min(self.base_data.len() - 1);
        self.base_data.prices[idx][7]
    }

    /// Calculate portfolio value.
    fn portfolio_value(&self) -> f32 {
        let price = self.current_price();
        self.balance
            + self.position_state.position * price * self.config.initial_balance
            + self.position_state.unrealized_pnl
    }

    /// Execute a trade.
    fn execute_trade(&mut self, target_position: f32) {
        let price = self.current_price();
        let position_delta = target_position - self.position_state.position;

        if position_delta.abs() < 0.01 {
            return;
        }

        let side = if position_delta > 0.0 {
            OrderSide::Buy
        } else {
            OrderSide::Sell
        };

        let quantity = position_delta.abs() * self.config.initial_balance;

        // Calculate slippage
        let slippage = self
            .config
            .slippage_model
            .calculate(quantity, price, side);
        let execution_price = price + slippage;

        // Calculate commission
        let commission = self
            .config
            .commission_model
            .calculate(quantity, execution_price, false, 0.0);

        // Update position
        let old_position = self.position_state.position;
        self.position_state.position = target_position;

        // Update position type
        self.position_state.position_type = if target_position > 0.01 {
            PositionType::Long
        } else if target_position < -0.01 {
            PositionType::Short
        } else {
            PositionType::Flat
        };

        // Handle entry price and PnL
        if old_position.abs() < 0.01 && target_position.abs() >= 0.01 {
            self.position_state.entry_price = execution_price;
        } else if old_position.signum() != target_position.signum() && target_position.abs() >= 0.01
        {
            let realized = (execution_price - self.position_state.entry_price) * old_position.abs();
            self.position_state.realized_pnl += realized;
            self.position_state.entry_price = execution_price;
        }

        self.balance -= commission;
    }

    /// Build hierarchical observation from all timeframes.
    fn build_observation(&self, device: &Device) -> Result<Tensor> {
        let mut obs = Vec::new();
        let base_idx = self.start_idx + self.current_step;
        let base_price = self.current_price();

        // Observations from each timeframe
        for tf in &self.config.timeframes {
            let lookback = self.config.lookback_per_timeframe.get(tf).copied().unwrap_or(20);
            let tf_data = self.timeframe_data.get(tf).unwrap();
            let current_tf_idx = self.synchronizer.get_index(base_idx, *tf);

            // Get lookback bars
            let start_tf_idx = current_tf_idx.saturating_sub(lookback);
            let end_tf_idx = current_tf_idx;

            for i in start_tf_idx..end_tf_idx {
                if let Some(bar) = tf_data.get(i) {
                    // Normalize features
                    obs.push((bar.open - base_price) / base_price);
                    obs.push((bar.high - base_price) / base_price);
                    obs.push((bar.low - base_price) / base_price);
                    obs.push((bar.close - base_price) / base_price);
                    obs.push(bar.volume / 1000.0);

                    // Additional features
                    for &f in &bar.features {
                        obs.push(f);
                    }
                } else {
                    // Pad with zeros
                    let features_per_bar = 5 + tf_data
                        .bars
                        .first()
                        .map(|b| b.features.len())
                        .unwrap_or(3);
                    for _ in 0..features_per_bar {
                        obs.push(0.0);
                    }
                }
            }

            // Pad if not enough history
            let features_per_bar = 5 + tf_data
                .bars
                .first()
                .map(|b| b.features.len())
                .unwrap_or(3);
            let expected_len_for_tf = lookback * features_per_bar;
            while obs.len() < expected_len_for_tf {
                obs.insert(0, 0.0);
            }
        }

        // Order book observation
        let book_obs = self.order_book.to_observation();
        for (i, val) in book_obs.into_iter().enumerate() {
            let normalized = if i % 2 == 0 {
                (val - base_price) / base_price
            } else {
                val / 1000.0
            };
            obs.push(normalized);
        }

        // Position state
        obs.push(self.position_state.position);
        obs.push(self.position_state.unrealized_pnl / self.config.initial_balance);
        obs.push(self.position_state.realized_pnl / self.config.initial_balance);
        obs.push(self.portfolio_value() / self.config.initial_balance - 1.0);

        let candle_device = device.to_candle()?;
        Ok(Tensor::from_slice(&obs, &[obs.len()], &candle_device)?)
    }
}

impl Environment for MultiTimeframeEnv {
    type ObsSpace = BoxSpace;
    type ActSpace = BoxSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActSpace {
        &self.act_space
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        // Random start (respecting minimum for all timeframes)
        let max_start = self.base_data.len() - self.min_start_idx - self.config.episode_length;
        self.start_idx = self.min_start_idx + self.rng.gen_range(0..=max_start);

        self.current_step = 0;
        self.balance = self.config.initial_balance;
        self.position_state = MTFPositionState::default();
        self.initial_value = self.config.initial_balance;

        // Initialize order book
        let price = self.current_price();
        let volatility = self.current_volatility();
        self.order_book.update(price, 10.0, volatility, &mut self.rng);

        self.build_observation(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let action_vec: Vec<f32> = action.flatten_all()?.to_vec1()?;
        let mut target_position = action_vec[0].clamp(-self.config.max_position, self.config.max_position);

        // Enforce short constraint
        if !self.config.allow_short && target_position < 0.0 {
            target_position = 0.0;
        }

        let _price_before = self.current_price();
        let value_before = self.portfolio_value();

        // Only execute on execution timeframe boundaries
        let base_idx = self.start_idx + self.current_step;
        if self.synchronizer.is_bar_boundary(base_idx, self.config.execution_timeframe) {
            self.execute_trade(target_position);
        }

        // Move to next step
        self.current_step += 1;

        // Update order book
        let new_price = self.current_price();
        let volatility = self.current_volatility();
        self.order_book.update(new_price, 10.0, volatility, &mut self.rng);

        // Update unrealized PnL
        if self.position_state.position.abs() > 0.01 {
            let pnl_direction = if self.position_state.position > 0.0 {
                1.0
            } else {
                -1.0
            };
            self.position_state.unrealized_pnl =
                (new_price - self.position_state.entry_price)
                    * self.position_state.position.abs()
                    * self.config.initial_balance
                    * pnl_direction;
        } else {
            self.position_state.unrealized_pnl = 0.0;
        }

        // Calculate reward
        let value_after = self.portfolio_value();
        let reward = (value_after - value_before) / self.config.initial_balance;

        // Check termination
        let episode_done = self.current_step >= self.config.episode_length;
        let bankrupt = value_after < self.config.initial_balance * 0.5;

        let observation = self.build_observation(device)?;

        let info = if episode_done || bankrupt {
            let total_return = (value_after - self.initial_value) / self.initial_value;
            let mut extra = HashMap::new();
            extra.insert("final_balance".into(), value_after);
            extra.insert("total_return_pct".into(), total_return * 100.0);
            extra.insert("realized_pnl".into(), self.position_state.realized_pnl);
            extra.insert("unrealized_pnl".into(), self.position_state.unrealized_pnl);

            Some(StepInfo {
                episode_return: Some(total_return),
                episode_length: Some(self.current_step),
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
        "MultiTimeframeEnv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timeframe_periods() {
        assert_eq!(Timeframe::M1.periods(), 1);
        assert_eq!(Timeframe::M5.periods(), 5);
        assert_eq!(Timeframe::H1.periods(), 60);
        assert_eq!(Timeframe::D1.periods(), 1440);
    }

    #[test]
    fn test_timeframe_aggregation() {
        let base_data = AdvancedMarketData::synthetic(1000, 42);
        let m5_data = TimeframeData::aggregate_from_base(&base_data, Timeframe::M5);
        assert!(m5_data.is_ok());

        let m5_data = m5_data.unwrap();
        assert_eq!(m5_data.len(), 200); // 1000 / 5
    }

    #[test]
    fn test_synchronizer() {
        let mut sync = TimeframeSynchronizer::new(vec![Timeframe::M1, Timeframe::M5, Timeframe::H1]);
        sync.build_mapping(1000);

        assert_eq!(sync.get_index(0, Timeframe::M1), 0);
        assert_eq!(sync.get_index(5, Timeframe::M5), 1);
        assert_eq!(sync.get_index(60, Timeframe::H1), 1);
    }

    #[test]
    fn test_env_creation() {
        let base_data = AdvancedMarketData::synthetic(5000, 42);
        let config = MultiTimeframeConfig::default()
            .episode_length(500)
            .timeframes(vec![Timeframe::M1, Timeframe::M5]);
        let env = MultiTimeframeEnv::with_config(base_data, config);
        assert!(env.is_ok());
    }
}
