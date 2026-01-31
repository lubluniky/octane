//! Advanced trading environment with realistic market microstructure.
//!
//! Features:
//! - Order book simulation with bid/ask spread and depth levels
//! - Multiple slippage models (linear, square-root, Almgren-Chriss)
//! - Latency simulation for order execution delays
//! - Partial fill support
//! - Comprehensive commission models
//! - Multiple order types (market, limit, stop-loss, take-profit)

use crate::core::{Device, OctaneError, Result};
use crate::envs::{BoxSpace, DiscreteSpace, Environment, ObsType, Space, StepInfo, StepResult};
use candle_core::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Order side (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderSide {
    /// Buy order.
    Buy,
    /// Sell order.
    Sell,
}

/// Order type.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OrderType {
    /// Market order - execute immediately at best available price.
    Market,
    /// Limit order - execute only at specified price or better.
    Limit {
        /// The limit price for the order.
        price: f32,
    },
    /// Stop-loss order - becomes market order when price falls below trigger.
    StopLoss {
        /// The price that triggers the stop-loss order.
        trigger_price: f32,
    },
    /// Take-profit order - becomes market order when price rises above trigger.
    TakeProfit {
        /// The price that triggers the take-profit order.
        trigger_price: f32,
    },
}

/// Order status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Order is pending execution.
    Pending,
    /// Order is partially filled.
    PartiallyFilled,
    /// Order is completely filled.
    Filled,
    /// Order was cancelled.
    Cancelled,
    /// Order was rejected.
    Rejected,
}

/// Position type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PositionType {
    /// Long position (holding asset).
    Long,
    /// Short position (borrowed and sold asset).
    Short,
    /// No position.
    #[default]
    Flat,
}

/// An order in the trading system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID.
    pub id: u64,
    /// Order side.
    pub side: OrderSide,
    /// Order type.
    pub order_type: OrderType,
    /// Requested quantity.
    pub quantity: f32,
    /// Filled quantity.
    pub filled_quantity: f32,
    /// Average fill price.
    pub avg_fill_price: f32,
    /// Order status.
    pub status: OrderStatus,
    /// Timestamp when order was created.
    pub created_at: usize,
    /// Timestamp when order should be executed (for latency simulation).
    pub execute_at: usize,
}

impl Order {
    /// Create a new market order.
    pub fn market(id: u64, side: OrderSide, quantity: f32, created_at: usize) -> Self {
        Self {
            id,
            side,
            order_type: OrderType::Market,
            quantity,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            status: OrderStatus::Pending,
            created_at,
            execute_at: created_at,
        }
    }

    /// Create a new limit order.
    pub fn limit(id: u64, side: OrderSide, quantity: f32, price: f32, created_at: usize) -> Self {
        Self {
            id,
            side,
            order_type: OrderType::Limit { price },
            quantity,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            status: OrderStatus::Pending,
            created_at,
            execute_at: created_at,
        }
    }

    /// Create a stop-loss order.
    pub fn stop_loss(
        id: u64,
        side: OrderSide,
        quantity: f32,
        trigger_price: f32,
        created_at: usize,
    ) -> Self {
        Self {
            id,
            side,
            order_type: OrderType::StopLoss { trigger_price },
            quantity,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            status: OrderStatus::Pending,
            created_at,
            execute_at: created_at,
        }
    }

    /// Create a take-profit order.
    pub fn take_profit(
        id: u64,
        side: OrderSide,
        quantity: f32,
        trigger_price: f32,
        created_at: usize,
    ) -> Self {
        Self {
            id,
            side,
            order_type: OrderType::TakeProfit { trigger_price },
            quantity,
            filled_quantity: 0.0,
            avg_fill_price: 0.0,
            status: OrderStatus::Pending,
            created_at,
            execute_at: created_at,
        }
    }

    /// Check if order is complete (filled or cancelled/rejected).
    pub fn is_complete(&self) -> bool {
        matches!(
            self.status,
            OrderStatus::Filled | OrderStatus::Cancelled | OrderStatus::Rejected
        )
    }

    /// Remaining quantity to fill.
    pub fn remaining_quantity(&self) -> f32 {
        self.quantity - self.filled_quantity
    }
}

/// A single level in the order book.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price at this level.
    pub price: f32,
    /// Total quantity available at this level.
    pub quantity: f32,
    /// Number of orders at this level.
    pub order_count: usize,
}

/// Simulated order book with bid/ask levels.
#[derive(Debug, Clone)]
pub struct OrderBook {
    /// Bid levels (sorted by price descending).
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending).
    pub asks: Vec<OrderBookLevel>,
    /// Number of depth levels to maintain.
    pub depth: usize,
}

impl OrderBook {
    /// Create a new order book with specified depth.
    pub fn new(depth: usize) -> Self {
        Self {
            bids: Vec::with_capacity(depth),
            asks: Vec::with_capacity(depth),
            depth,
        }
    }

    /// Update order book from market data.
    pub fn update(&mut self, mid_price: f32, spread_bps: f32, volatility: f32, rng: &mut StdRng) {
        self.bids.clear();
        self.asks.clear();

        let spread = mid_price * spread_bps / 10000.0;
        let best_bid = mid_price - spread / 2.0;
        let best_ask = mid_price + spread / 2.0;

        // Generate bid levels
        let mut bid_price = best_bid;
        for i in 0..self.depth {
            let depth_factor = 1.0 + i as f32 * 0.5;
            let quantity = (100.0 + rng.gen::<f32>() * 200.0) * depth_factor;
            let order_count = (5 + rng.gen_range(0..10)) as usize;

            self.bids.push(OrderBookLevel {
                price: bid_price,
                quantity,
                order_count,
            });

            // Spread increases with depth based on volatility
            bid_price -= spread * (1.0 + volatility * rng.gen::<f32>());
        }

        // Generate ask levels
        let mut ask_price = best_ask;
        for i in 0..self.depth {
            let depth_factor = 1.0 + i as f32 * 0.5;
            let quantity = (100.0 + rng.gen::<f32>() * 200.0) * depth_factor;
            let order_count = (5 + rng.gen_range(0..10)) as usize;

            self.asks.push(OrderBookLevel {
                price: ask_price,
                quantity,
                order_count,
            });

            ask_price += spread * (1.0 + volatility * rng.gen::<f32>());
        }
    }

    /// Get best bid price.
    pub fn best_bid(&self) -> Option<f32> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price.
    pub fn best_ask(&self) -> Option<f32> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price.
    pub fn mid_price(&self) -> Option<f32> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get spread in basis points.
    pub fn spread_bps(&self) -> Option<f32> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => {
                let mid = (bid + ask) / 2.0;
                Some((ask - bid) / mid * 10000.0)
            }
            _ => None,
        }
    }

    /// Calculate total available liquidity on one side up to a price level.
    pub fn liquidity_to_price(&self, side: OrderSide, price: f32) -> f32 {
        match side {
            OrderSide::Buy => self
                .asks
                .iter()
                .filter(|l| l.price <= price)
                .map(|l| l.quantity)
                .sum(),
            OrderSide::Sell => self
                .bids
                .iter()
                .filter(|l| l.price >= price)
                .map(|l| l.quantity)
                .sum(),
        }
    }

    /// Flatten order book into observation vector.
    pub fn to_observation(&self) -> Vec<f32> {
        let mut obs = Vec::with_capacity(self.depth * 4);

        // Bid levels: price, quantity pairs (normalized)
        for level in &self.bids {
            obs.push(level.price);
            obs.push(level.quantity);
        }
        // Pad if needed
        while obs.len() < self.depth * 2 {
            obs.push(0.0);
        }

        // Ask levels: price, quantity pairs
        for level in &self.asks {
            obs.push(level.price);
            obs.push(level.quantity);
        }
        while obs.len() < self.depth * 4 {
            obs.push(0.0);
        }

        obs
    }
}

/// Slippage model for realistic trade execution.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SlippageModel {
    /// No slippage (ideal execution).
    None,
    /// Linear slippage: slippage = impact_factor * quantity.
    Linear {
        /// Impact factor (price impact per unit traded).
        impact_factor: f32,
    },
    /// Square-root slippage: slippage = impact_factor * sqrt(quantity).
    /// More realistic for larger trades.
    SquareRoot {
        /// Impact factor.
        impact_factor: f32,
    },
    /// Almgren-Chriss model: temporary + permanent impact.
    AlmgrenChriss {
        /// Temporary impact coefficient (eta).
        eta: f32,
        /// Permanent impact coefficient (gamma).
        gamma: f32,
        /// Daily volatility.
        volatility: f32,
        /// Daily volume.
        daily_volume: f32,
    },
}

impl Default for SlippageModel {
    fn default() -> Self {
        Self::Linear {
            impact_factor: 0.0001,
        }
    }
}

impl SlippageModel {
    /// Calculate slippage for a given trade.
    pub fn calculate(&self, quantity: f32, price: f32, side: OrderSide) -> f32 {
        let direction = match side {
            OrderSide::Buy => 1.0,
            OrderSide::Sell => -1.0,
        };

        match *self {
            Self::None => 0.0,
            Self::Linear { impact_factor } => impact_factor * quantity.abs() * price * direction,
            Self::SquareRoot { impact_factor } => {
                impact_factor * quantity.abs().sqrt() * price * direction
            }
            Self::AlmgrenChriss {
                eta,
                gamma,
                volatility,
                daily_volume,
            } => {
                let participation_rate = quantity.abs() / daily_volume;
                // Temporary impact
                let temp_impact = eta * volatility * participation_rate.sqrt();
                // Permanent impact
                let perm_impact = gamma * volatility * participation_rate;
                (temp_impact + perm_impact) * price * direction
            }
        }
    }
}

/// Commission model for trading fees.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommissionModel {
    /// No commission.
    None,
    /// Flat percentage commission.
    Percentage {
        /// Commission rate (e.g., 0.001 = 0.1%).
        rate: f32,
    },
    /// Maker/taker fee model.
    MakerTaker {
        /// Fee for maker orders (add liquidity).
        maker_rate: f32,
        /// Fee for taker orders (remove liquidity).
        taker_rate: f32,
    },
    /// Tiered fee structure based on volume.
    Tiered {
        /// Volume tiers and corresponding rates.
        /// Format: [(volume_threshold, rate), ...]
        tiers: Vec<(f32, f32)>,
    },
}

impl Default for CommissionModel {
    fn default() -> Self {
        Self::MakerTaker {
            maker_rate: 0.0002,
            taker_rate: 0.0005,
        }
    }
}

impl CommissionModel {
    /// Calculate commission for a trade.
    pub fn calculate(&self, quantity: f32, price: f32, is_maker: bool, volume_30d: f32) -> f32 {
        let trade_value = quantity.abs() * price;

        match self {
            Self::None => 0.0,
            Self::Percentage { rate } => trade_value * rate,
            Self::MakerTaker {
                maker_rate,
                taker_rate,
            } => {
                if is_maker {
                    trade_value * maker_rate
                } else {
                    trade_value * taker_rate
                }
            }
            Self::Tiered { tiers } => {
                let rate = tiers
                    .iter()
                    .filter(|(threshold, _)| volume_30d >= *threshold)
                    .map(|(_, rate)| *rate)
                    .last()
                    .unwrap_or(tiers.first().map(|(_, r)| *r).unwrap_or(0.001));
                trade_value * rate
            }
        }
    }
}

/// Configuration for the advanced trading environment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedTradingConfig {
    /// Initial cash balance.
    pub initial_balance: f32,
    /// Slippage model.
    pub slippage_model: SlippageModel,
    /// Commission model.
    pub commission_model: CommissionModel,
    /// Latency in milliseconds (0 = no latency).
    pub latency_ms: u32,
    /// Enable partial fills.
    pub enable_partial_fills: bool,
    /// Minimum fill ratio for partial fills.
    pub min_fill_ratio: f32,
    /// Maximum position size (as fraction of portfolio).
    pub max_position: f32,
    /// Lookback window for observations.
    pub lookback: usize,
    /// Episode length (0 = use all data).
    pub episode_length: usize,
    /// Order book depth levels.
    pub orderbook_depth: usize,
    /// Default spread in basis points.
    pub default_spread_bps: f32,
    /// Enable short selling.
    pub allow_short: bool,
    /// Margin requirement for short positions.
    pub short_margin_requirement: f32,
    /// Maximum leverage.
    pub max_leverage: f32,
    /// Action mode: continuous or discrete.
    pub discrete_actions: bool,
    /// Number of discrete action levels (if discrete_actions = true).
    pub num_action_levels: usize,
}

impl Default for AdvancedTradingConfig {
    fn default() -> Self {
        Self {
            initial_balance: 10000.0,
            slippage_model: SlippageModel::default(),
            commission_model: CommissionModel::default(),
            latency_ms: 0,
            enable_partial_fills: false,
            min_fill_ratio: 0.1,
            max_position: 1.0,
            lookback: 20,
            episode_length: 252,
            orderbook_depth: 5,
            default_spread_bps: 10.0,
            allow_short: true,
            short_margin_requirement: 1.5,
            max_leverage: 1.0,
            discrete_actions: false,
            num_action_levels: 11, // [-1.0, -0.8, ..., 0, ..., 0.8, 1.0]
        }
    }
}

impl AdvancedTradingConfig {
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

    /// Set latency in milliseconds.
    pub fn latency_ms(mut self, ms: u32) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Enable or disable partial fills.
    pub fn enable_partial_fills(mut self, enable: bool) -> Self {
        self.enable_partial_fills = enable;
        self
    }

    /// Set initial balance.
    pub fn initial_balance(mut self, balance: f32) -> Self {
        self.initial_balance = balance;
        self
    }

    /// Set maximum position.
    pub fn max_position(mut self, pos: f32) -> Self {
        self.max_position = pos;
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

    /// Set order book depth.
    pub fn orderbook_depth(mut self, depth: usize) -> Self {
        self.orderbook_depth = depth;
        self
    }

    /// Enable or disable short selling.
    pub fn allow_short(mut self, allow: bool) -> Self {
        self.allow_short = allow;
        self
    }

    /// Set discrete actions mode.
    pub fn discrete_actions(mut self, discrete: bool, num_levels: usize) -> Self {
        self.discrete_actions = discrete;
        self.num_action_levels = num_levels;
        self
    }
}

/// Market data for the advanced trading environment.
#[derive(Debug, Clone)]
pub struct AdvancedMarketData {
    /// OHLCV prices: [timesteps, features].
    pub prices: Vec<Vec<f32>>,
    /// Feature names.
    pub feature_names: Vec<String>,
    /// Timestamps (milliseconds since epoch).
    pub timestamps: Vec<u64>,
    /// Volume data for slippage calculation.
    pub volumes: Vec<f32>,
}

impl AdvancedMarketData {
    /// Create synthetic market data for testing.
    pub fn synthetic(timesteps: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut price = 100.0f32;
        let mut prices = Vec::with_capacity(timesteps);
        let mut timestamps = Vec::with_capacity(timesteps);
        let mut volumes = Vec::with_capacity(timesteps);

        let base_timestamp = 1704067200000u64; // 2024-01-01 00:00:00 UTC

        for i in 0..timesteps {
            // Random walk with drift and volatility clustering
            let volatility = 0.02 + rng.gen::<f32>() * 0.03;
            let returns = rng.gen::<f32>() * volatility * 2.0 - volatility;
            price *= 1.0 + returns;

            let open = price * (1.0 + rng.gen::<f32>() * 0.01 - 0.005);
            let high = price * (1.0 + rng.gen::<f32>() * 0.02);
            let low = price * (1.0 - rng.gen::<f32>() * 0.02);
            let close = price;
            let volume = rng.gen::<f32>() * 1000.0 + 500.0;

            // Technical indicators
            let sma_ratio = 1.0 + rng.gen::<f32>() * 0.1 - 0.05;
            let rsi = rng.gen::<f32>() * 100.0;
            let realized_vol = volatility;

            prices.push(vec![
                open,
                high,
                low,
                close,
                volume,
                sma_ratio,
                rsi,
                realized_vol,
            ]);
            timestamps.push(base_timestamp + (i as u64) * 60000); // 1-minute bars
            volumes.push(volume);
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
            timestamps,
            volumes,
        }
    }

    /// Number of timesteps.
    pub fn len(&self) -> usize {
        self.prices.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.prices.is_empty()
    }

    /// Number of features.
    pub fn num_features(&self) -> usize {
        self.feature_names.len()
    }
}

/// Internal state for tracking positions.
#[derive(Debug, Clone, Default)]
struct PositionState {
    /// Current position (-1 to 1).
    position: f32,
    /// Position type.
    position_type: PositionType,
    /// Entry price.
    entry_price: f32,
    /// Unrealized PnL.
    unrealized_pnl: f32,
    /// Realized PnL.
    realized_pnl: f32,
}

/// Advanced trading environment with realistic market simulation.
#[derive(Clone)]
pub struct AdvancedTradingEnv {
    /// Market data.
    data: AdvancedMarketData,
    /// Configuration.
    config: AdvancedTradingConfig,
    /// Current timestep in episode.
    current_step: usize,
    /// Start index in data.
    start_idx: usize,
    /// Current cash balance.
    balance: f32,
    /// Position state.
    position_state: PositionState,
    /// Order book.
    order_book: OrderBook,
    /// Pending orders (for latency simulation).
    pending_orders: VecDeque<Order>,
    /// Next order ID.
    next_order_id: u64,
    /// Initial portfolio value.
    initial_portfolio_value: f32,
    /// 30-day trading volume (for tiered fees).
    volume_30d: f32,
    /// Observation space.
    obs_space: BoxSpace,
    /// Action space (continuous).
    continuous_act_space: BoxSpace,
    /// Action space (discrete).
    discrete_act_space: DiscreteSpace,
    /// Random number generator.
    rng: StdRng,
}

impl AdvancedTradingEnv {
    /// Create a new advanced trading environment with default config.
    pub fn new(data: AdvancedMarketData) -> Result<Self> {
        Self::with_config(data, AdvancedTradingConfig::default())
    }

    /// Create with custom configuration.
    pub fn with_config(data: AdvancedMarketData, config: AdvancedTradingConfig) -> Result<Self> {
        if data.len() < config.lookback + config.episode_length {
            return Err(OctaneError::InvalidConfig(format!(
                "Data length {} too short for lookback {} + episode_length {}",
                data.len(),
                config.lookback,
                config.episode_length
            )));
        }

        // Observation: [lookback * features + orderbook + position_state]
        // Position state: position, unrealized_pnl, realized_pnl, balance_ratio
        let orderbook_dim = config.orderbook_depth * 4;
        let position_dim = 4;
        let obs_dim = config.lookback * data.num_features() + orderbook_dim + position_dim;
        let obs_space = BoxSpace::unbounded(vec![obs_dim]);

        // Action space
        let continuous_act_space = BoxSpace::symmetric(1.0, vec![1]);
        let discrete_act_space = DiscreteSpace::new(config.num_action_levels);

        let orderbook_depth = config.orderbook_depth;
        Ok(Self {
            data,
            config,
            current_step: 0,
            start_idx: 0,
            balance: 0.0,
            position_state: PositionState::default(),
            order_book: OrderBook::new(orderbook_depth),
            pending_orders: VecDeque::new(),
            next_order_id: 0,
            initial_portfolio_value: 0.0,
            volume_30d: 0.0,
            obs_space,
            continuous_act_space,
            discrete_act_space,
            rng: StdRng::from_entropy(),
        })
    }

    /// Get current price.
    fn current_price(&self) -> f32 {
        let idx = (self.start_idx + self.current_step).min(self.data.len() - 1);
        self.data.prices[idx][3]
    }

    /// Get current volatility.
    fn current_volatility(&self) -> f32 {
        let idx = (self.start_idx + self.current_step).min(self.data.len() - 1);
        self.data.prices[idx][7]
    }

    /// Calculate portfolio value.
    fn portfolio_value(&self) -> f32 {
        let price = self.current_price();
        let position_value = self.position_state.position * price * self.config.initial_balance;
        self.balance + position_value + self.position_state.unrealized_pnl
    }

    /// Process pending orders with latency simulation.
    fn process_pending_orders(&mut self) {
        let current_time = self.current_step;
        let mut orders_to_execute = Vec::new();

        // Find orders ready for execution
        while let Some(order) = self.pending_orders.front() {
            if order.execute_at <= current_time {
                orders_to_execute.push(self.pending_orders.pop_front().unwrap());
            } else {
                break;
            }
        }

        // Execute orders
        for mut order in orders_to_execute {
            self.execute_order(&mut order);
        }
    }

    /// Execute an order.
    fn execute_order(&mut self, order: &mut Order) {
        let price = self.current_price();

        // Check trigger conditions for stop/take-profit orders
        match order.order_type {
            OrderType::StopLoss { trigger_price } => {
                if price > trigger_price {
                    order.status = OrderStatus::Pending;
                    self.pending_orders.push_back(order.clone());
                    return;
                }
            }
            OrderType::TakeProfit { trigger_price } => {
                if price < trigger_price {
                    order.status = OrderStatus::Pending;
                    self.pending_orders.push_back(order.clone());
                    return;
                }
            }
            OrderType::Limit { price: limit_price } => {
                match order.side {
                    OrderSide::Buy if price > limit_price => {
                        // Price too high for buy limit
                        order.status = OrderStatus::Pending;
                        self.pending_orders.push_back(order.clone());
                        return;
                    }
                    OrderSide::Sell if price < limit_price => {
                        // Price too low for sell limit
                        order.status = OrderStatus::Pending;
                        self.pending_orders.push_back(order.clone());
                        return;
                    }
                    _ => {}
                }
            }
            OrderType::Market => {}
        }

        // Calculate fill quantity
        let mut fill_quantity = order.remaining_quantity();

        if self.config.enable_partial_fills {
            // Simulate partial fill based on available liquidity
            let available_liquidity = match order.side {
                OrderSide::Buy => self
                    .order_book
                    .asks
                    .iter()
                    .map(|l| l.quantity)
                    .sum::<f32>(),
                OrderSide::Sell => self
                    .order_book
                    .bids
                    .iter()
                    .map(|l| l.quantity)
                    .sum::<f32>(),
            };

            let max_fill = available_liquidity * self.rng.gen_range(0.5..1.0);
            fill_quantity = fill_quantity.min(max_fill);

            // Ensure minimum fill ratio
            if fill_quantity < order.quantity * self.config.min_fill_ratio {
                fill_quantity = order.quantity.min(available_liquidity);
            }
        }

        // Calculate execution price with slippage
        let slippage = self
            .config
            .slippage_model
            .calculate(fill_quantity, price, order.side);
        let execution_price = price + slippage;

        // Calculate commission
        let is_maker = matches!(order.order_type, OrderType::Limit { .. });
        let commission = self.config.commission_model.calculate(
            fill_quantity,
            execution_price,
            is_maker,
            self.volume_30d,
        );

        // Update order
        let old_filled = order.filled_quantity;
        order.filled_quantity += fill_quantity;
        order.avg_fill_price = if old_filled == 0.0 {
            execution_price
        } else {
            (order.avg_fill_price * old_filled + execution_price * fill_quantity)
                / order.filled_quantity
        };

        if order.filled_quantity >= order.quantity * 0.999 {
            order.status = OrderStatus::Filled;
        } else {
            order.status = OrderStatus::PartiallyFilled;
        }

        // Update position
        let position_delta = match order.side {
            OrderSide::Buy => fill_quantity / self.config.initial_balance,
            OrderSide::Sell => -fill_quantity / self.config.initial_balance,
        };

        let old_position = self.position_state.position;
        self.position_state.position = (self.position_state.position + position_delta)
            .clamp(-self.config.max_position, self.config.max_position);

        // Update position type
        self.position_state.position_type = if self.position_state.position > 0.01 {
            PositionType::Long
        } else if self.position_state.position < -0.01 {
            PositionType::Short
        } else {
            PositionType::Flat
        };

        // Update entry price
        if old_position.abs() < 0.01 && self.position_state.position.abs() >= 0.01 {
            self.position_state.entry_price = execution_price;
        } else if old_position.signum() != self.position_state.position.signum()
            && self.position_state.position.abs() >= 0.01
        {
            // Position flip - realize PnL and set new entry
            let realized =
                (execution_price - self.position_state.entry_price) * old_position.abs();
            self.position_state.realized_pnl += realized;
            self.position_state.entry_price = execution_price;
        }

        // Deduct commission from balance
        self.balance -= commission;

        // Update 30-day volume
        self.volume_30d += fill_quantity * execution_price;
    }

    /// Submit a new order.
    pub fn submit_order(&mut self, mut order: Order) -> u64 {
        order.id = self.next_order_id;
        self.next_order_id += 1;

        // Apply latency
        let latency_steps = (self.config.latency_ms as f32 / 60000.0).ceil() as usize; // Convert ms to steps
        order.execute_at = self.current_step + latency_steps;

        let order_id = order.id;
        self.pending_orders.push_back(order);

        order_id
    }

    /// Build observation tensor.
    fn build_observation(&self, device: &Device) -> Result<Tensor> {
        let lookback = self.config.lookback;
        let num_features = self.data.num_features();
        let orderbook_dim = self.config.orderbook_depth * 4;
        let position_dim = 4;

        let mut obs = Vec::with_capacity(lookback * num_features + orderbook_dim + position_dim);

        // Historical features (normalized)
        let start = self.start_idx + self.current_step.saturating_sub(lookback);
        let end = (self.start_idx + self.current_step).min(self.data.len());

        let base_price = self.current_price();

        for i in start..end {
            for (j, &val) in self.data.prices[i].iter().enumerate() {
                let normalized = match j {
                    0..=3 => (val - base_price) / base_price, // prices relative to current
                    4 => val / 1000.0,                        // volume normalized
                    5 => val - 1.0,                           // sma_ratio centered
                    6 => (val - 50.0) / 50.0,                 // rsi normalized
                    _ => val,
                };
                obs.push(normalized);
            }
        }

        // Pad if not enough history
        while obs.len() < lookback * num_features {
            obs.insert(0, 0.0);
        }

        // Order book observation (normalized)
        let orderbook_obs = self.order_book.to_observation();
        for (i, val) in orderbook_obs.into_iter().enumerate() {
            let normalized = if i % 2 == 0 {
                // Price levels - normalize relative to current price
                (val - base_price) / base_price
            } else {
                // Quantities - normalize
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

    /// Convert discrete action to continuous position target.
    fn discrete_to_continuous(&self, action_idx: usize) -> f32 {
        let n = self.config.num_action_levels;
        if n <= 1 {
            return 0.0;
        }
        let step = 2.0 / (n - 1) as f32;
        -1.0 + action_idx as f32 * step
    }
}

/// Wrapper for observation space that can be either Box or Discrete.
#[derive(Debug, Clone)]
pub enum TradingObsSpace {
    /// Box observation space.
    Box(BoxSpace),
}

impl Space for TradingObsSpace {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Box(s) => s.shape(),
        }
    }

    fn sample(&self, rng: &mut impl Rng, device: &Device) -> Result<Tensor> {
        match self {
            Self::Box(s) => s.sample(rng, device),
        }
    }

    fn contains(&self, tensor: &Tensor) -> Result<bool> {
        match self {
            Self::Box(s) => s.contains(tensor),
        }
    }
}

/// Wrapper for action space that can be either Box or Discrete.
#[derive(Debug, Clone)]
pub enum TradingActSpace {
    /// Continuous action space.
    Continuous(BoxSpace),
    /// Discrete action space.
    Discrete(DiscreteSpace),
}

impl Space for TradingActSpace {
    fn shape(&self) -> &[usize] {
        match self {
            Self::Continuous(s) => s.shape(),
            Self::Discrete(s) => s.shape(),
        }
    }

    fn flat_dim(&self) -> usize {
        match self {
            Self::Continuous(s) => s.flat_dim(),
            Self::Discrete(s) => s.flat_dim(),
        }
    }

    fn sample(&self, rng: &mut impl Rng, device: &Device) -> Result<Tensor> {
        match self {
            Self::Continuous(s) => s.sample(rng, device),
            Self::Discrete(s) => s.sample(rng, device),
        }
    }

    fn contains(&self, tensor: &Tensor) -> Result<bool> {
        match self {
            Self::Continuous(s) => s.contains(tensor),
            Self::Discrete(s) => s.contains(tensor),
        }
    }
}

impl Environment for AdvancedTradingEnv {
    type ObsSpace = BoxSpace;
    type ActSpace = BoxSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActSpace {
        if self.config.discrete_actions {
            // Return continuous space but interpret as discrete index
            &self.continuous_act_space
        } else {
            &self.continuous_act_space
        }
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        // Random start point
        let max_start = self.data.len() - self.config.lookback - self.config.episode_length;
        self.start_idx = self.rng.gen_range(0..=max_start);

        self.current_step = self.config.lookback;
        self.balance = self.config.initial_balance;
        self.position_state = PositionState::default();
        self.pending_orders.clear();
        self.next_order_id = 0;
        self.volume_30d = 0.0;
        self.initial_portfolio_value = self.balance;

        // Initialize order book
        let price = self.current_price();
        let volatility = self.current_volatility();
        self.order_book.update(
            price,
            self.config.default_spread_bps,
            volatility,
            &mut self.rng,
        );

        self.build_observation(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let action_vec: Vec<f32> = action.flatten_all()?.to_vec1()?;

        // Convert action to target position
        let target_position = if self.config.discrete_actions {
            let action_idx = action_vec[0] as usize;
            self.discrete_to_continuous(action_idx)
        } else {
            action_vec[0].clamp(-self.config.max_position, self.config.max_position)
        };

        // Enforce short selling constraints
        let target_position = if !self.config.allow_short && target_position < 0.0 {
            0.0
        } else {
            target_position
        };

        let price_before = self.current_price();

        // Process any pending orders
        self.process_pending_orders();

        // Create order for position change
        let position_delta = target_position - self.position_state.position;
        if position_delta.abs() > 0.01 {
            let side = if position_delta > 0.0 {
                OrderSide::Buy
            } else {
                OrderSide::Sell
            };
            let quantity = position_delta.abs() * self.config.initial_balance;

            let order = Order::market(self.next_order_id, side, quantity, self.current_step);
            self.submit_order(order);
        }

        // Process orders immediately if no latency
        if self.config.latency_ms == 0 {
            self.process_pending_orders();
        }

        // Move to next step
        self.current_step += 1;

        // Update order book
        let new_price = self.current_price();
        let volatility = self.current_volatility();
        self.order_book.update(
            new_price,
            self.config.default_spread_bps,
            volatility,
            &mut self.rng,
        );

        // Calculate unrealized PnL
        if self.position_state.position.abs() > 0.01 {
            let pnl_direction = if self.position_state.position > 0.0 {
                1.0
            } else {
                -1.0
            };
            self.position_state.unrealized_pnl = (new_price - self.position_state.entry_price)
                * self.position_state.position.abs()
                * self.config.initial_balance
                * pnl_direction;
        } else {
            self.position_state.unrealized_pnl = 0.0;
        }

        // Calculate reward (portfolio return)
        let portfolio_before =
            self.balance + self.position_state.position * price_before * self.config.initial_balance;
        let portfolio_after = self.portfolio_value();
        let reward = (portfolio_after - portfolio_before) / self.config.initial_balance;

        // Check termination
        let episode_done = self.current_step >= self.config.lookback + self.config.episode_length;
        let bankrupt = portfolio_after < self.config.initial_balance * 0.5;

        let observation = self.build_observation(device)?;

        let info = if episode_done || bankrupt {
            let total_return =
                (portfolio_after - self.initial_portfolio_value) / self.initial_portfolio_value;
            let mut extra = std::collections::HashMap::new();
            extra.insert("final_balance".into(), portfolio_after);
            extra.insert("total_return_pct".into(), total_return * 100.0);
            extra.insert("realized_pnl".into(), self.position_state.realized_pnl);
            extra.insert("unrealized_pnl".into(), self.position_state.unrealized_pnl);
            extra.insert("volume_30d".into(), self.volume_30d);

            Some(StepInfo {
                episode_return: Some(total_return),
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
        "AdvancedTradingEnv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_book() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut book = OrderBook::new(5);
        book.update(100.0, 10.0, 0.02, &mut rng);

        assert!(book.best_bid().is_some());
        assert!(book.best_ask().is_some());
        assert!(book.best_bid().unwrap() < book.best_ask().unwrap());
        assert!(book.mid_price().is_some());
    }

    #[test]
    fn test_slippage_models() {
        let linear = SlippageModel::Linear { impact_factor: 0.001 };
        let sqrt = SlippageModel::SquareRoot { impact_factor: 0.01 };

        let slip1 = linear.calculate(100.0, 100.0, OrderSide::Buy);
        let slip2 = sqrt.calculate(100.0, 100.0, OrderSide::Buy);

        assert!(slip1 > 0.0);
        assert!(slip2 > 0.0);
    }

    #[test]
    fn test_commission_models() {
        let pct = CommissionModel::Percentage { rate: 0.001 };
        let maker_taker = CommissionModel::MakerTaker {
            maker_rate: 0.0002,
            taker_rate: 0.0005,
        };

        let comm1 = pct.calculate(100.0, 100.0, false, 0.0);
        let comm2_maker = maker_taker.calculate(100.0, 100.0, true, 0.0);
        let comm2_taker = maker_taker.calculate(100.0, 100.0, false, 0.0);

        assert_eq!(comm1, 10.0);
        assert!(comm2_maker < comm2_taker);
    }

    #[test]
    fn test_env_creation() {
        let data = AdvancedMarketData::synthetic(1000, 42);
        let config = AdvancedTradingConfig::default()
            .episode_length(100)
            .lookback(10);
        let env = AdvancedTradingEnv::with_config(data, config);
        assert!(env.is_ok());
    }
}
