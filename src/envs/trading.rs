//! Example trading environment for algorithmic trading RL.

use crate::core::{Device, OctaneError, Result};
use crate::envs::{BoxSpace, Environment, ObsType, StepInfo, StepResult};
use candle_core::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Market data for trading simulation.
#[derive(Debug, Clone)]
pub struct MarketData {
    /// OHLCV prices: [timesteps, features]
    pub prices: Vec<Vec<f32>>,
    /// Feature names for debugging.
    pub feature_names: Vec<String>,
}

impl MarketData {
    /// Create synthetic market data for testing.
    pub fn synthetic(timesteps: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut price = 100.0f32;
        let mut prices = Vec::with_capacity(timesteps);

        for _ in 0..timesteps {
            // Random walk with drift
            let returns = rng.gen::<f32>() * 0.04 - 0.02; // -2% to +2%
            price *= 1.0 + returns;

            // OHLCV-like features
            let open = price * (1.0 + rng.gen::<f32>() * 0.01 - 0.005);
            let high = price * (1.0 + rng.gen::<f32>() * 0.02);
            let low = price * (1.0 - rng.gen::<f32>() * 0.02);
            let close = price;
            let volume = rng.gen::<f32>() * 1000.0 + 500.0;

            // Technical indicators (simplified)
            let sma_ratio = 1.0 + rng.gen::<f32>() * 0.1 - 0.05;
            let rsi = rng.gen::<f32>() * 100.0;
            let volatility = rng.gen::<f32>() * 0.05;

            prices.push(vec![
                open, high, low, close, volume, sma_ratio, rsi, volatility,
            ]);
        }

        Self {
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
        }
    }

    /// Number of timesteps in the data.
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }

    /// Number of features per timestep.
    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }
}

/// Trading environment configuration.
#[derive(Debug, Clone)]
pub struct TradingEnvConfig {
    /// Initial cash balance.
    pub initial_balance: f32,
    /// Transaction cost as fraction of trade value.
    pub transaction_cost: f32,
    /// Maximum position size (as fraction of portfolio).
    pub max_position: f32,
    /// Lookback window for observations.
    pub lookback: usize,
    /// Episode length (0 = use all data).
    pub episode_length: usize,
}

impl Default for TradingEnvConfig {
    fn default() -> Self {
        Self {
            initial_balance: 10000.0,
            transaction_cost: 0.001,
            max_position: 1.0,
            lookback: 20,
            episode_length: 252, // ~1 trading year
        }
    }
}

/// Trading environment for RL-based algorithmic trading.
#[derive(Clone)]
pub struct TradingEnv {
    /// Market data.
    data: MarketData,
    /// Configuration.
    config: TradingEnvConfig,
    /// Current timestep in episode.
    current_step: usize,
    /// Start index in data for this episode.
    start_idx: usize,
    /// Current cash balance.
    balance: f32,
    /// Current position (-1 to 1, short to long).
    position: f32,
    /// Entry price for current position.
    entry_price: f32,
    /// Total portfolio value at start of episode.
    initial_portfolio_value: f32,
    /// Observation space.
    obs_space: BoxSpace,
    /// Action space (continuous: target position).
    act_space: BoxSpace,
    /// Random number generator.
    rng: StdRng,
}

impl TradingEnv {
    /// Create a new trading environment.
    pub fn new(data: MarketData) -> Result<Self> {
        Self::with_config(data, TradingEnvConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(data: MarketData, config: TradingEnvConfig) -> Result<Self> {
        if data.len() < config.lookback + config.episode_length {
            return Err(OctaneError::InvalidConfig(format!(
                "Data length {} too short for lookback {} + episode_length {}",
                data.len(),
                config.lookback,
                config.episode_length
            )));
        }

        // Observation: [lookback * features + position + balance_ratio]
        let obs_dim = config.lookback * data.num_features() + 2;
        let obs_space = BoxSpace::unbounded(vec![obs_dim]);

        // Action: target position [-1, 1]
        let act_space = BoxSpace::symmetric(1.0, vec![1]);

        Ok(Self {
            data,
            config,
            current_step: 0,
            start_idx: 0,
            balance: 0.0,
            position: 0.0,
            entry_price: 0.0,
            initial_portfolio_value: 0.0,
            obs_space,
            act_space,
            rng: StdRng::from_entropy(),
        })
    }

    /// Get current price (close).
    fn current_price(&self) -> f32 {
        let idx = (self.start_idx + self.current_step).min(self.data.len() - 1);
        self.data.prices[idx][3] // close price
    }

    /// Calculate portfolio value.
    fn portfolio_value(&self) -> f32 {
        let price = self.current_price();
        self.balance + self.position * price * self.config.initial_balance
    }

    /// Build observation tensor.
    fn build_observation(&self, device: &Device) -> Result<Tensor> {
        let lookback = self.config.lookback;
        let num_features = self.data.num_features();

        let mut obs = Vec::with_capacity(lookback * num_features + 2);

        // Historical features (normalized)
        let start = self.start_idx + self.current_step.saturating_sub(lookback);
        let end = (self.start_idx + self.current_step).min(self.data.len());

        for i in start..end {
            for (j, &val) in self.data.prices[i].iter().enumerate() {
                // Simple normalization (could be improved)
                let normalized = match j {
                    0..=3 => (val - 100.0) / 100.0, // prices
                    4 => (val - 750.0) / 250.0,     // volume
                    5 => val - 1.0,                 // sma_ratio
                    6 => (val - 50.0) / 50.0,       // rsi
                    _ => val,                       // others
                };
                obs.push(normalized);
            }
        }

        // Pad if not enough history
        while obs.len() < lookback * num_features {
            obs.insert(0, 0.0);
        }

        // Current position and balance ratio
        obs.push(self.position);
        obs.push(self.portfolio_value() / self.config.initial_balance - 1.0);

        let candle_device = device.to_candle()?;
        Ok(Tensor::from_slice(&obs, &[obs.len()], &candle_device)?)
    }
}

impl Environment for TradingEnv {
    type ObsSpace = BoxSpace;
    type ActSpace = BoxSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActSpace {
        &self.act_space
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        // Random start point
        let max_start = self.data.len() - self.config.lookback - self.config.episode_length;
        self.start_idx = self.rng.gen_range(0..=max_start);

        self.current_step = self.config.lookback;
        self.balance = self.config.initial_balance;
        self.position = 0.0;
        self.entry_price = 0.0;
        self.initial_portfolio_value = self.balance;

        self.build_observation(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let action_vec: Vec<f32> = action.flatten_all()?.to_vec1()?;
        let target_position =
            action_vec[0].clamp(-self.config.max_position, self.config.max_position);

        let price = self.current_price();

        // Execute trade
        let position_delta = target_position - self.position;
        if position_delta.abs() > 0.01 {
            // Transaction cost
            let trade_value = position_delta.abs() * self.config.initial_balance;
            let cost = trade_value * self.config.transaction_cost;
            self.balance -= cost;

            // Update position
            if self.position.abs() < 0.01 && target_position.abs() > 0.01 {
                self.entry_price = price;
            }
            self.position = target_position;
        }

        // Move to next step
        self.current_step += 1;
        let new_price = self.current_price();

        // Calculate reward (portfolio return)
        let portfolio_before = self.balance + self.position * price * self.config.initial_balance;
        let portfolio_after =
            self.balance + self.position * new_price * self.config.initial_balance;
        let reward = (portfolio_after - portfolio_before) / self.config.initial_balance;

        // Check termination
        let episode_done = self.current_step >= self.config.lookback + self.config.episode_length;
        let bankrupt = portfolio_after < self.config.initial_balance * 0.5;

        let observation = self.build_observation(device)?;

        let info = if episode_done || bankrupt {
            let total_return =
                (portfolio_after - self.initial_portfolio_value) / self.initial_portfolio_value;
            Some(StepInfo {
                episode_return: Some(total_return),
                episode_length: Some(self.current_step - self.config.lookback),
                extra: {
                    let mut m = std::collections::HashMap::new();
                    m.insert("final_balance".into(), portfolio_after);
                    m.insert("total_return_pct".into(), total_return * 100.0);
                    m
                },
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
        "TradingEnv"
    }
}
