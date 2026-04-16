//! Paper trading engine for simulated order execution.
//!
//! Provides a realistic paper trading environment without using real money:
//! - Simulated order execution with configurable slippage
//! - Virtual balance tracking
//! - Order book simulation
//! - Latency simulation
//! - Full trade history logging
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::live::paper::{PaperTradingEngine, PaperTradingConfig};
//! use octane_rs::live::types::{Order, Side};
//!
//! let config = PaperTradingConfig::default()
//!     .initial_balance("USDT", 10000.0)
//!     .slippage_bps(5.0)
//!     .latency_ms(50);
//!
//! let mut engine = PaperTradingEngine::new(config);
//!
//! // Place a market order
//! let order = Order::market("BTCUSDT", Side::Buy, 0.1);
//! let filled_order = engine.execute_order(order, 50000.0)?;
//! ```

use crate::live::error::{LiveTradingError, Result};
use crate::live::types::{
    current_timestamp_ms, Balance, Order, OrderBook, OrderBookLevel, OrderStatus, OrderType,
    Position, Side, Trade,
};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Slippage model for paper trading.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SlippageModel {
    /// No slippage.
    None,
    /// Fixed slippage in basis points.
    Fixed {
        /// Slippage in basis points (1 bp = 0.01%).
        bps: f64,
    },
    /// Random slippage within a range.
    Random {
        /// Minimum slippage in basis points.
        min_bps: f64,
        /// Maximum slippage in basis points.
        max_bps: f64,
    },
    /// Volume-weighted slippage (larger orders have more slippage).
    VolumeWeighted {
        /// Base slippage in basis points.
        base_bps: f64,
        /// Impact factor (slippage increases with order size).
        impact_factor: f64,
    },
    /// Square-root slippage model (common in practice).
    SquareRoot {
        /// Impact coefficient.
        impact: f64,
        /// Daily volume for normalization.
        daily_volume: f64,
    },
}

impl Default for SlippageModel {
    fn default() -> Self {
        SlippageModel::Fixed { bps: 5.0 }
    }
}

/// Fill model for paper trading.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FillModel {
    /// Immediate fill at current price.
    Immediate,
    /// Partial fills based on available liquidity.
    Partial {
        /// Probability of full fill.
        fill_probability: f64,
        /// Minimum fill ratio.
        min_fill_ratio: f64,
    },
    /// Realistic fills using simulated order book.
    OrderBook {
        /// Depth levels to consider.
        depth_levels: usize,
    },
}

impl Default for FillModel {
    fn default() -> Self {
        FillModel::Immediate
    }
}

/// Configuration for paper trading engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaperTradingConfig {
    /// Initial balances by asset.
    pub initial_balances: HashMap<String, f64>,
    /// Slippage model.
    pub slippage_model: SlippageModel,
    /// Fill model.
    pub fill_model: FillModel,
    /// Simulated latency in milliseconds.
    pub latency_ms: u64,
    /// Commission rate (e.g., 0.001 for 0.1%).
    pub commission_rate: f64,
    /// Commission asset (empty = use quote asset).
    pub commission_asset: Option<String>,
    /// Whether to enable realistic spread simulation.
    pub simulate_spread: bool,
    /// Default spread in basis points.
    pub default_spread_bps: f64,
    /// Maximum position size per symbol.
    pub max_position_size: Option<f64>,
    /// Enable trade logging.
    pub enable_logging: bool,
    /// Log file path.
    pub log_path: Option<String>,
}

impl Default for PaperTradingConfig {
    fn default() -> Self {
        let mut initial_balances = HashMap::new();
        initial_balances.insert("USDT".to_string(), 10000.0);

        Self {
            initial_balances,
            slippage_model: SlippageModel::default(),
            fill_model: FillModel::default(),
            latency_ms: 50,
            commission_rate: 0.001,
            commission_asset: None,
            simulate_spread: true,
            default_spread_bps: 10.0,
            max_position_size: None,
            enable_logging: true,
            log_path: None,
        }
    }
}

impl PaperTradingConfig {
    /// Create a new config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set initial balance for an asset.
    pub fn initial_balance(mut self, asset: impl Into<String>, amount: f64) -> Self {
        self.initial_balances.insert(asset.into(), amount);
        self
    }

    /// Set slippage model.
    pub fn slippage_model(mut self, model: SlippageModel) -> Self {
        self.slippage_model = model;
        self
    }

    /// Set slippage in basis points (convenience method).
    pub fn slippage_bps(mut self, bps: f64) -> Self {
        self.slippage_model = SlippageModel::Fixed { bps };
        self
    }

    /// Set fill model.
    pub fn fill_model(mut self, model: FillModel) -> Self {
        self.fill_model = model;
        self
    }

    /// Set simulated latency.
    pub fn latency_ms(mut self, ms: u64) -> Self {
        self.latency_ms = ms;
        self
    }

    /// Set commission rate.
    pub fn commission_rate(mut self, rate: f64) -> Self {
        self.commission_rate = rate;
        self
    }

    /// Enable or disable spread simulation.
    pub fn simulate_spread(mut self, enabled: bool) -> Self {
        self.simulate_spread = enabled;
        self
    }

    /// Set default spread in basis points.
    pub fn default_spread_bps(mut self, bps: f64) -> Self {
        self.default_spread_bps = bps;
        self
    }

    /// Set maximum position size.
    pub fn max_position_size(mut self, size: f64) -> Self {
        self.max_position_size = Some(size);
        self
    }

    /// Enable logging.
    pub fn enable_logging(mut self, enabled: bool) -> Self {
        self.enable_logging = enabled;
        self
    }

    /// Set log file path.
    pub fn log_path(mut self, path: impl Into<String>) -> Self {
        self.log_path = Some(path.into());
        self
    }
}

/// Paper trading statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PaperTradingStats {
    /// Total number of orders placed.
    pub total_orders: u64,
    /// Number of filled orders.
    pub filled_orders: u64,
    /// Number of cancelled orders.
    pub cancelled_orders: u64,
    /// Number of rejected orders.
    pub rejected_orders: u64,
    /// Total volume traded (in quote currency).
    pub total_volume: f64,
    /// Total commission paid.
    pub total_commission: f64,
    /// Total slippage cost.
    pub total_slippage: f64,
    /// Realized PnL.
    pub realized_pnl: f64,
    /// Peak portfolio value.
    pub peak_value: f64,
    /// Current drawdown.
    pub current_drawdown: f64,
    /// Maximum drawdown.
    pub max_drawdown: f64,
    /// Win rate (winning trades / total trades).
    pub win_rate: f64,
    /// Number of winning trades.
    pub winning_trades: u64,
    /// Number of losing trades.
    pub losing_trades: u64,
}

/// Simulated order book for paper trading.
#[derive(Debug, Clone)]
pub struct SimulatedOrderBook {
    /// Symbol.
    symbol: String,
    /// Mid price.
    mid_price: f64,
    /// Spread in basis points.
    spread_bps: f64,
    /// Bid levels.
    bids: Vec<OrderBookLevel>,
    /// Ask levels.
    asks: Vec<OrderBookLevel>,
    /// Random generator for depth.
    rng: rand::rngs::StdRng,
}

impl SimulatedOrderBook {
    /// Create a new simulated order book.
    pub fn new(symbol: impl Into<String>, mid_price: f64, spread_bps: f64, seed: u64) -> Self {
        use rand::SeedableRng;
        let mut book = Self {
            symbol: symbol.into(),
            mid_price,
            spread_bps,
            bids: Vec::new(),
            asks: Vec::new(),
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        };
        book.regenerate_depth(10);
        book
    }

    /// Update mid price and regenerate depth.
    pub fn update_price(&mut self, mid_price: f64) {
        self.mid_price = mid_price;
        self.regenerate_depth(10);
    }

    /// Regenerate order book depth.
    fn regenerate_depth(&mut self, levels: usize) {
        self.bids.clear();
        self.asks.clear();

        let spread = self.mid_price * self.spread_bps / 10000.0;
        let best_bid = self.mid_price - spread / 2.0;
        let best_ask = self.mid_price + spread / 2.0;

        // Generate bid levels
        for i in 0..levels {
            let price_offset = (i as f64) * 0.0001 * self.mid_price;
            let price = best_bid - price_offset;
            let quantity = self.rng.gen_range(0.5..5.0) * (1.0 + i as f64 * 0.1);
            self.bids.push(OrderBookLevel { price, quantity });
        }

        // Generate ask levels
        for i in 0..levels {
            let price_offset = (i as f64) * 0.0001 * self.mid_price;
            let price = best_ask + price_offset;
            let quantity = self.rng.gen_range(0.5..5.0) * (1.0 + i as f64 * 0.1);
            self.asks.push(OrderBookLevel { price, quantity });
        }
    }

    /// Get best bid.
    pub fn best_bid(&self) -> f64 {
        self.bids.first().map(|l| l.price).unwrap_or(self.mid_price)
    }

    /// Get best ask.
    pub fn best_ask(&self) -> f64 {
        self.asks.first().map(|l| l.price).unwrap_or(self.mid_price)
    }

    /// Convert to OrderBook type.
    pub fn to_order_book(&self) -> OrderBook {
        OrderBook {
            symbol: self.symbol.clone(),
            bids: self.bids.clone(),
            asks: self.asks.clone(),
            last_update_id: current_timestamp_ms(),
            timestamp: current_timestamp_ms(),
        }
    }

    /// Simulate fill for a given order size.
    pub fn simulate_fill(&self, side: Side, quantity: f64) -> (f64, f64) {
        let levels = match side {
            Side::Buy => &self.asks,
            Side::Sell => &self.bids,
        };

        let mut remaining = quantity;
        let mut total_cost = 0.0;
        let mut filled = 0.0;

        for level in levels {
            if remaining <= 0.0 {
                break;
            }

            let fill_qty = remaining.min(level.quantity);
            total_cost += fill_qty * level.price;
            filled += fill_qty;
            remaining -= fill_qty;
        }

        let avg_price = if filled > 0.0 {
            total_cost / filled
        } else {
            0.0
        };
        (filled, avg_price)
    }
}

/// Paper trading engine.
#[derive(Debug)]
pub struct PaperTradingEngine {
    /// Configuration.
    config: PaperTradingConfig,
    /// Current balances.
    balances: HashMap<String, Balance>,
    /// Open orders.
    open_orders: HashMap<String, Order>,
    /// Order history.
    order_history: Vec<Order>,
    /// Trade history.
    trade_history: Vec<Trade>,
    /// Positions by symbol.
    positions: HashMap<String, Position>,
    /// Simulated order books by symbol.
    order_books: HashMap<String, SimulatedOrderBook>,
    /// Statistics.
    stats: PaperTradingStats,
    /// Random generator.
    rng: rand::rngs::StdRng,
    /// Trade counter for ID generation.
    trade_counter: u64,
}

impl PaperTradingEngine {
    /// Create a new paper trading engine.
    pub fn new(config: PaperTradingConfig) -> Self {
        use rand::SeedableRng;

        let mut balances = HashMap::new();
        for (asset, amount) in &config.initial_balances {
            balances.insert(
                asset.clone(),
                Balance {
                    asset: asset.clone(),
                    free: *amount,
                    locked: 0.0,
                },
            );
        }

        Self {
            config,
            balances,
            open_orders: HashMap::new(),
            order_history: Vec::new(),
            trade_history: Vec::new(),
            positions: HashMap::new(),
            order_books: HashMap::new(),
            stats: PaperTradingStats::default(),
            rng: rand::rngs::StdRng::from_entropy(),
            trade_counter: 0,
        }
    }

    /// Get current balances.
    pub fn balances(&self) -> &HashMap<String, Balance> {
        &self.balances
    }

    /// Get balance for a specific asset.
    pub fn balance(&self, asset: &str) -> Option<&Balance> {
        self.balances.get(asset)
    }

    /// Get positions.
    pub fn positions(&self) -> &HashMap<String, Position> {
        &self.positions
    }

    /// Get position for a symbol.
    pub fn position(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Get open orders.
    pub fn open_orders(&self) -> &HashMap<String, Order> {
        &self.open_orders
    }

    /// Get order history.
    pub fn order_history(&self) -> &[Order] {
        &self.order_history
    }

    /// Get trade history.
    pub fn trade_history(&self) -> &[Trade] {
        &self.trade_history
    }

    /// Get statistics.
    pub fn stats(&self) -> &PaperTradingStats {
        &self.stats
    }

    /// Update market price for a symbol.
    pub fn update_price(&mut self, symbol: &str, price: f64) {
        // Update or create order book
        if let Some(book) = self.order_books.get_mut(symbol) {
            book.update_price(price);
        } else {
            self.order_books.insert(
                symbol.to_string(),
                SimulatedOrderBook::new(
                    symbol,
                    price,
                    self.config.default_spread_bps,
                    self.rng.gen(),
                ),
            );
        }

        // Update position mark price
        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.mark_price = price;
            pos.unrealized_pnl = (price - pos.entry_price) * pos.size;
            pos.updated_at = current_timestamp_ms();
        }

        // Check and execute stop/limit orders
        let orders_to_check: Vec<_> = self
            .open_orders
            .iter()
            .filter(|(_, o)| o.symbol == symbol)
            .map(|(id, _)| id.clone())
            .collect();

        for order_id in orders_to_check {
            let _ = self.check_and_execute_order(&order_id, price);
        }
    }

    /// Place an order.
    pub fn place_order(&mut self, mut order: Order) -> Result<Order> {
        // Validate order
        self.validate_order(&order)?;

        // Lock funds for the order
        self.lock_funds_for_order(&order)?;

        // Update order status
        order.status = OrderStatus::Submitted;
        order.updated_at = current_timestamp_ms();
        self.stats.total_orders += 1;

        // For market orders, execute immediately
        if order.order_type == OrderType::Market {
            let price = self.get_execution_price(&order.symbol, order.side)?;
            order = self.execute_fill(&order, price)?;
        } else {
            // Add to open orders for limit/stop orders
            self.open_orders
                .insert(order.client_order_id.clone(), order.clone());
        }

        Ok(order)
    }

    /// Execute an order with a given price (for market orders or triggered orders).
    pub fn execute_order(&mut self, order: Order, price: f64) -> Result<Order> {
        let mut order = order;
        order.status = OrderStatus::Submitted;
        order.updated_at = current_timestamp_ms();
        self.stats.total_orders += 1;

        // Validate
        self.validate_order(&order)?;

        // Lock funds
        self.lock_funds_for_order(&order)?;

        // Execute
        self.execute_fill(&order, price)
    }

    /// Cancel an order.
    pub fn cancel_order(&mut self, order_id: &str) -> Result<Order> {
        let mut order = self
            .open_orders
            .remove(order_id)
            .ok_or_else(|| LiveTradingError::OrderNotFound(order_id.to_string()))?;

        // Unlock funds
        self.unlock_funds_for_order(&order);

        // Update status
        order.status = OrderStatus::Cancelled;
        order.updated_at = current_timestamp_ms();
        self.stats.cancelled_orders += 1;

        // Add to history
        self.order_history.push(order.clone());

        Ok(order)
    }

    /// Get simulated order book for a symbol.
    pub fn order_book(&self, symbol: &str) -> Option<OrderBook> {
        self.order_books.get(symbol).map(|b| b.to_order_book())
    }

    /// Calculate total portfolio value in quote currency.
    pub fn portfolio_value(&self, quote_asset: &str, prices: &HashMap<String, f64>) -> f64 {
        let mut total = 0.0;

        // Add balances
        for (asset, balance) in &self.balances {
            if asset == quote_asset {
                total += balance.total();
            } else if let Some(&price) = prices.get(&format!("{asset}{quote_asset}")) {
                total += balance.total() * price;
            }
        }

        // Add unrealized PnL from positions
        for pos in self.positions.values() {
            total += pos.unrealized_pnl;
        }

        total
    }

    /// Reset the engine to initial state.
    pub fn reset(&mut self) {
        self.balances.clear();
        for (asset, amount) in &self.config.initial_balances {
            self.balances.insert(
                asset.clone(),
                Balance {
                    asset: asset.clone(),
                    free: *amount,
                    locked: 0.0,
                },
            );
        }
        self.open_orders.clear();
        self.order_history.clear();
        self.trade_history.clear();
        self.positions.clear();
        self.order_books.clear();
        self.stats = PaperTradingStats::default();
        self.trade_counter = 0;
    }

    // --- Private helper methods ---

    fn validate_order(&self, order: &Order) -> Result<()> {
        if order.quantity <= 0.0 {
            return Err(LiveTradingError::Order("Quantity must be positive".into()));
        }

        if let Some(max_size) = self.config.max_position_size {
            if order.quantity > max_size {
                return Err(LiveTradingError::PositionLimitExceeded(format!(
                    "Order size {} exceeds max {}",
                    order.quantity, max_size
                )));
            }
        }

        if order.order_type == OrderType::Limit && order.price.is_none() {
            return Err(LiveTradingError::Order("Limit order requires price".into()));
        }

        Ok(())
    }

    fn lock_funds_for_order(&mut self, order: &Order) -> Result<()> {
        // Parse symbol to get base and quote assets
        let (base, quote) = self.parse_symbol(&order.symbol)?;

        match order.side {
            Side::Buy => {
                // Lock quote currency
                let price = order
                    .price
                    .or_else(|| self.get_execution_price(&order.symbol, order.side).ok())
                    .unwrap_or(0.0);
                let required = order.quantity * price * (1.0 + self.config.commission_rate);

                let balance = self.balances.get_mut(&quote).ok_or_else(|| {
                    LiveTradingError::InsufficientBalance {
                        required,
                        available: 0.0,
                    }
                })?;

                if balance.free < required {
                    return Err(LiveTradingError::InsufficientBalance {
                        required,
                        available: balance.free,
                    });
                }

                balance.free -= required;
                balance.locked += required;
            }
            Side::Sell => {
                // Lock base currency
                let balance = self.balances.get_mut(&base).ok_or_else(|| {
                    LiveTradingError::InsufficientBalance {
                        required: order.quantity,
                        available: 0.0,
                    }
                })?;

                if balance.free < order.quantity {
                    return Err(LiveTradingError::InsufficientBalance {
                        required: order.quantity,
                        available: balance.free,
                    });
                }

                balance.free -= order.quantity;
                balance.locked += order.quantity;
            }
        }

        Ok(())
    }

    fn unlock_funds_for_order(&mut self, order: &Order) {
        let Ok((base, quote)) = self.parse_symbol(&order.symbol) else {
            return;
        };

        match order.side {
            Side::Buy => {
                let price = order.price.unwrap_or(0.0);
                let locked =
                    order.remaining_quantity() * price * (1.0 + self.config.commission_rate);
                if let Some(balance) = self.balances.get_mut(&quote) {
                    balance.locked -= locked.min(balance.locked);
                    balance.free += locked.min(balance.locked);
                }
            }
            Side::Sell => {
                if let Some(balance) = self.balances.get_mut(&base) {
                    let locked = order.remaining_quantity();
                    balance.locked -= locked.min(balance.locked);
                    balance.free += locked.min(balance.locked);
                }
            }
        }
    }

    fn execute_fill(&mut self, order: &Order, base_price: f64) -> Result<Order> {
        let mut order = order.clone();

        // Calculate slippage
        let slippage = self.calculate_slippage(&order, base_price);
        let fill_price = match order.side {
            Side::Buy => base_price * (1.0 + slippage),
            Side::Sell => base_price * (1.0 - slippage),
        };

        // Determine fill quantity based on fill model
        let fill_quantity = match self.config.fill_model {
            FillModel::Immediate => order.remaining_quantity(),
            FillModel::Partial {
                fill_probability,
                min_fill_ratio,
            } => {
                if self.rng.gen::<f64>() < fill_probability {
                    order.remaining_quantity()
                } else {
                    let ratio = self.rng.gen_range(min_fill_ratio..1.0);
                    order.remaining_quantity() * ratio
                }
            }
            FillModel::OrderBook { .. } => {
                if let Some(book) = self.order_books.get(&order.symbol) {
                    let (filled, _) = book.simulate_fill(order.side, order.remaining_quantity());
                    filled
                } else {
                    order.remaining_quantity()
                }
            }
        };

        // Calculate commission
        let commission = fill_quantity * fill_price * self.config.commission_rate;

        // Update order
        order.filled_quantity += fill_quantity;
        let total_value = order.average_fill_price.unwrap_or(0.0)
            * (order.filled_quantity - fill_quantity)
            + fill_price * fill_quantity;
        order.average_fill_price = Some(total_value / order.filled_quantity);
        order.updated_at = current_timestamp_ms();

        if (order.filled_quantity - order.quantity).abs() < 1e-10 {
            order.status = OrderStatus::Filled;
            self.stats.filled_orders += 1;
        } else {
            order.status = OrderStatus::PartiallyFilled;
        }

        // Record trade
        self.trade_counter += 1;
        let trade = Trade {
            trade_id: format!("paper_{}", self.trade_counter),
            order_id: order.client_order_id.clone(),
            symbol: order.symbol.clone(),
            side: order.side,
            price: fill_price,
            quantity: fill_quantity,
            commission,
            commission_asset: self
                .config
                .commission_asset
                .clone()
                .unwrap_or_else(|| "USDT".to_string()),
            timestamp: current_timestamp_ms(),
            is_maker: order.order_type == OrderType::Limit,
        };
        self.trade_history.push(trade);

        // Update balances
        self.settle_trade(&order, fill_quantity, fill_price, commission)?;

        // Update position
        self.update_position(&order.symbol, order.side, fill_quantity, fill_price);

        // Update stats
        self.stats.total_volume += fill_quantity * fill_price;
        self.stats.total_commission += commission;
        self.stats.total_slippage += slippage.abs() * fill_quantity * base_price;

        // Remove from open orders if filled
        if order.status == OrderStatus::Filled {
            self.open_orders.remove(&order.client_order_id);
        }

        // Add to history
        self.order_history.push(order.clone());

        Ok(order)
    }

    fn calculate_slippage(&mut self, order: &Order, price: f64) -> f64 {
        match self.config.slippage_model {
            SlippageModel::None => 0.0,
            SlippageModel::Fixed { bps } => bps / 10000.0,
            SlippageModel::Random { min_bps, max_bps } => {
                self.rng.gen_range(min_bps..max_bps) / 10000.0
            }
            SlippageModel::VolumeWeighted {
                base_bps,
                impact_factor,
            } => {
                let size_factor = (order.quantity * price / 10000.0).sqrt();
                (base_bps + impact_factor * size_factor) / 10000.0
            }
            SlippageModel::SquareRoot {
                impact,
                daily_volume,
            } => {
                let participation = (order.quantity * price) / daily_volume;
                impact * participation.sqrt()
            }
        }
    }

    fn settle_trade(
        &mut self,
        order: &Order,
        quantity: f64,
        price: f64,
        commission: f64,
    ) -> Result<()> {
        let (base, quote) = self.parse_symbol(&order.symbol)?;
        let trade_value = quantity * price;

        match order.side {
            Side::Buy => {
                // Deduct quote currency (including commission)
                if let Some(balance) = self.balances.get_mut(&quote) {
                    let cost = trade_value * (1.0 + self.config.commission_rate);
                    balance.locked -= cost.min(balance.locked);
                }
                // Add base currency
                let base_clone = base.clone();
                let balance = self.balances.entry(base).or_insert(Balance {
                    asset: base_clone,
                    free: 0.0,
                    locked: 0.0,
                });
                balance.free += quantity;
            }
            Side::Sell => {
                // Deduct base currency
                if let Some(balance) = self.balances.get_mut(&base) {
                    balance.locked -= quantity.min(balance.locked);
                }
                // Add quote currency (minus commission)
                let quote_clone = quote.clone();
                let balance = self.balances.entry(quote).or_insert(Balance {
                    asset: quote_clone,
                    free: 0.0,
                    locked: 0.0,
                });
                balance.free += trade_value - commission;
            }
        }

        Ok(())
    }

    fn update_position(&mut self, symbol: &str, side: Side, quantity: f64, price: f64) {
        let position = self
            .positions
            .entry(symbol.to_string())
            .or_insert(Position::new(symbol));

        let old_size = position.size;
        let old_entry = position.entry_price;

        match side {
            Side::Buy => {
                if position.size >= 0.0 {
                    // Adding to long or opening long
                    let total_value = old_size * old_entry + quantity * price;
                    position.size += quantity;
                    position.entry_price = if position.size > 0.0 {
                        total_value / position.size
                    } else {
                        price
                    };
                } else {
                    // Closing short
                    let close_qty = quantity.min(-old_size);
                    let pnl = (old_entry - price) * close_qty;
                    position.realized_pnl += pnl;
                    position.size += quantity;

                    if position.size > 0.0 {
                        position.entry_price = price;
                    }

                    // Update stats
                    if pnl > 0.0 {
                        self.stats.winning_trades += 1;
                    } else {
                        self.stats.losing_trades += 1;
                    }
                    self.stats.realized_pnl += pnl;
                }
            }
            Side::Sell => {
                if position.size <= 0.0 {
                    // Adding to short or opening short
                    let total_value = (-old_size) * old_entry + quantity * price;
                    position.size -= quantity;
                    position.entry_price = if position.size < 0.0 {
                        total_value / (-position.size)
                    } else {
                        price
                    };
                } else {
                    // Closing long
                    let close_qty = quantity.min(old_size);
                    let pnl = (price - old_entry) * close_qty;
                    position.realized_pnl += pnl;
                    position.size -= quantity;

                    if position.size < 0.0 {
                        position.entry_price = price;
                    }

                    // Update stats
                    if pnl > 0.0 {
                        self.stats.winning_trades += 1;
                    } else {
                        self.stats.losing_trades += 1;
                    }
                    self.stats.realized_pnl += pnl;
                }
            }
        }

        position.mark_price = price;
        position.unrealized_pnl = (price - position.entry_price) * position.size;
        position.updated_at = current_timestamp_ms();

        // Update win rate
        let total_trades = self.stats.winning_trades + self.stats.losing_trades;
        if total_trades > 0 {
            self.stats.win_rate = self.stats.winning_trades as f64 / total_trades as f64;
        }
    }

    fn get_execution_price(&self, symbol: &str, side: Side) -> Result<f64> {
        if let Some(book) = self.order_books.get(symbol) {
            Ok(match side {
                Side::Buy => book.best_ask(),
                Side::Sell => book.best_bid(),
            })
        } else {
            Err(LiveTradingError::SymbolNotFound(symbol.to_string()))
        }
    }

    fn check_and_execute_order(&mut self, order_id: &str, price: f64) -> Result<bool> {
        let order = match self.open_orders.get(order_id) {
            Some(o) => o.clone(),
            None => return Ok(false),
        };

        let should_execute = match order.order_type {
            OrderType::Limit => match order.side {
                Side::Buy => order.price.map(|p| price <= p).unwrap_or(false),
                Side::Sell => order.price.map(|p| price >= p).unwrap_or(false),
            },
            OrderType::StopLoss => match order.side {
                Side::Buy => order.stop_price.map(|p| price >= p).unwrap_or(false),
                Side::Sell => order.stop_price.map(|p| price <= p).unwrap_or(false),
            },
            OrderType::TakeProfit => match order.side {
                Side::Buy => order.stop_price.map(|p| price <= p).unwrap_or(false),
                Side::Sell => order.stop_price.map(|p| price >= p).unwrap_or(false),
            },
            _ => false,
        };

        if should_execute {
            self.open_orders.remove(order_id);
            self.execute_fill(&order, price)?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn parse_symbol(&self, symbol: &str) -> Result<(String, String)> {
        // Common quote assets
        let quote_assets = ["USDT", "USDC", "BUSD", "BTC", "ETH", "BNB"];

        for quote in &quote_assets {
            if symbol.ends_with(quote) {
                let base = symbol[..symbol.len() - quote.len()].to_string();
                return Ok((base, quote.to_string()));
            }
        }

        // Default: assume last 4 chars are quote
        if symbol.len() > 4 {
            let base = symbol[..symbol.len() - 4].to_string();
            let quote = symbol[symbol.len() - 4..].to_string();
            Ok((base, quote))
        } else {
            Err(LiveTradingError::SymbolNotFound(format!(
                "Cannot parse symbol: {}",
                symbol
            )))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_paper_trading_config() {
        let config = PaperTradingConfig::default()
            .initial_balance("USDT", 10000.0)
            .slippage_bps(5.0)
            .latency_ms(50);

        assert_eq!(config.initial_balances.get("USDT"), Some(&10000.0));
        assert_eq!(config.latency_ms, 50);
    }

    #[test]
    fn test_paper_trading_engine() {
        let config = PaperTradingConfig::default()
            .initial_balance("USDT", 10000.0)
            .initial_balance("BTC", 1.0);

        let mut engine = PaperTradingEngine::new(config);

        // Update price
        engine.update_price("BTCUSDT", 50000.0);

        // Check balances
        assert_eq!(engine.balance("USDT").map(|b| b.free), Some(10000.0));
        assert_eq!(engine.balance("BTC").map(|b| b.free), Some(1.0));
    }

    #[test]
    fn test_market_order_execution() {
        let config = PaperTradingConfig::default()
            .initial_balance("USDT", 10000.0)
            .slippage_model(SlippageModel::None)
            .commission_rate(0.0);

        let mut engine = PaperTradingEngine::new(config);
        engine.update_price("BTCUSDT", 50000.0);

        // Place buy order
        let order = Order::market("BTCUSDT", Side::Buy, 0.1);
        let filled = engine.execute_order(order, 50000.0).unwrap();

        assert_eq!(filled.status, OrderStatus::Filled);
        assert!((filled.filled_quantity - 0.1).abs() < 1e-10);

        // Check balances updated
        assert!(engine.balance("BTC").unwrap().free > 0.0);
        assert!(engine.balance("USDT").unwrap().free < 10000.0);
    }

    #[test]
    fn test_position_tracking() {
        let config = PaperTradingConfig::default()
            .initial_balance("USDT", 50000.0)
            .slippage_model(SlippageModel::None)
            .commission_rate(0.0);

        let mut engine = PaperTradingEngine::new(config);
        engine.update_price("BTCUSDT", 50000.0);

        // Open long position
        let order = Order::market("BTCUSDT", Side::Buy, 0.5);
        engine.execute_order(order, 50000.0).unwrap();

        let pos = engine.position("BTCUSDT").unwrap();
        assert!((pos.size - 0.5).abs() < 1e-10);
        assert!((pos.entry_price - 50000.0).abs() < 1e-10);

        // Price goes up
        engine.update_price("BTCUSDT", 51000.0);
        let pos = engine.position("BTCUSDT").unwrap();
        assert!(pos.unrealized_pnl > 0.0);
    }

    #[test]
    fn test_slippage_model() {
        let config = PaperTradingConfig::default()
            .initial_balance("USDT", 100000.0)
            .slippage_model(SlippageModel::Fixed { bps: 10.0 })
            .commission_rate(0.0);

        let mut engine = PaperTradingEngine::new(config);
        engine.update_price("BTCUSDT", 50000.0);

        // Place buy order - should have slippage
        let order = Order::market("BTCUSDT", Side::Buy, 0.1);
        let filled = engine.execute_order(order, 50000.0).unwrap();

        // 10 bps = 0.1% slippage, so price should be 50000 * 1.001 = 50050
        let expected_price = 50000.0 * 1.001;
        assert!((filled.average_fill_price.unwrap() - expected_price).abs() < 0.01);
    }
}
