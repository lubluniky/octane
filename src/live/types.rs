//! Common types for live trading infrastructure.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Side of an order (buy or sell).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Side {
    /// Buy order.
    Buy,
    /// Sell order.
    Sell,
}

impl std::fmt::Display for Side {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Side::Buy => write!(f, "BUY"),
            Side::Sell => write!(f, "SELL"),
        }
    }
}

/// Type of order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderType {
    /// Market order - executes immediately at best available price.
    Market,
    /// Limit order - executes at specified price or better.
    Limit,
    /// Stop-loss order - triggers market order when price reaches stop.
    StopLoss,
    /// Stop-limit order - triggers limit order when price reaches stop.
    StopLimit,
    /// Take-profit order.
    TakeProfit,
    /// Trailing stop order.
    TrailingStop,
}

impl std::fmt::Display for OrderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OrderType::Market => write!(f, "MARKET"),
            OrderType::Limit => write!(f, "LIMIT"),
            OrderType::StopLoss => write!(f, "STOP_LOSS"),
            OrderType::StopLimit => write!(f, "STOP_LIMIT"),
            OrderType::TakeProfit => write!(f, "TAKE_PROFIT"),
            OrderType::TrailingStop => write!(f, "TRAILING_STOP"),
        }
    }
}

/// Status of an order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OrderStatus {
    /// Order is pending submission.
    Pending,
    /// Order has been submitted to exchange.
    Submitted,
    /// Order is partially filled.
    PartiallyFilled,
    /// Order is fully filled.
    Filled,
    /// Order has been cancelled.
    Cancelled,
    /// Order was rejected.
    Rejected,
    /// Order has expired.
    Expired,
}

impl OrderStatus {
    /// Returns true if order is in a terminal state.
    pub fn is_terminal(&self) -> bool {
        matches!(
            self,
            OrderStatus::Filled
                | OrderStatus::Cancelled
                | OrderStatus::Rejected
                | OrderStatus::Expired
        )
    }

    /// Returns true if order is still active.
    pub fn is_active(&self) -> bool {
        matches!(
            self,
            OrderStatus::Pending | OrderStatus::Submitted | OrderStatus::PartiallyFilled
        )
    }
}

/// Time in force for orders.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeInForce {
    /// Good till cancelled.
    GTC,
    /// Immediate or cancel - fill what you can, cancel rest.
    IOC,
    /// Fill or kill - fill entirely or cancel.
    FOK,
    /// Good till date.
    GTD(u64),
}

impl Default for TimeInForce {
    fn default() -> Self {
        TimeInForce::GTC
    }
}

/// Represents an order.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Order {
    /// Unique order ID (client-generated).
    pub client_order_id: String,
    /// Exchange order ID (assigned by exchange after submission).
    pub exchange_order_id: Option<String>,
    /// Trading symbol (e.g., "BTCUSDT").
    pub symbol: String,
    /// Order side.
    pub side: Side,
    /// Order type.
    pub order_type: OrderType,
    /// Order quantity.
    pub quantity: f64,
    /// Limit price (for limit orders).
    pub price: Option<f64>,
    /// Stop price (for stop orders).
    pub stop_price: Option<f64>,
    /// Time in force.
    pub time_in_force: TimeInForce,
    /// Current order status.
    pub status: OrderStatus,
    /// Filled quantity.
    pub filled_quantity: f64,
    /// Average fill price.
    pub average_fill_price: Option<f64>,
    /// Timestamp of order creation (Unix ms).
    pub created_at: u64,
    /// Timestamp of last update (Unix ms).
    pub updated_at: u64,
}

impl Order {
    /// Create a new market order.
    pub fn market(symbol: impl Into<String>, side: Side, quantity: f64) -> Self {
        let now = current_timestamp_ms();
        Self {
            client_order_id: generate_client_order_id(),
            exchange_order_id: None,
            symbol: symbol.into(),
            side,
            order_type: OrderType::Market,
            quantity,
            price: None,
            stop_price: None,
            time_in_force: TimeInForce::IOC,
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
            average_fill_price: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a new limit order.
    pub fn limit(symbol: impl Into<String>, side: Side, quantity: f64, price: f64) -> Self {
        let now = current_timestamp_ms();
        Self {
            client_order_id: generate_client_order_id(),
            exchange_order_id: None,
            symbol: symbol.into(),
            side,
            order_type: OrderType::Limit,
            quantity,
            price: Some(price),
            stop_price: None,
            time_in_force: TimeInForce::GTC,
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
            average_fill_price: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Create a stop-loss order.
    pub fn stop_loss(
        symbol: impl Into<String>,
        side: Side,
        quantity: f64,
        stop_price: f64,
    ) -> Self {
        let now = current_timestamp_ms();
        Self {
            client_order_id: generate_client_order_id(),
            exchange_order_id: None,
            symbol: symbol.into(),
            side,
            order_type: OrderType::StopLoss,
            quantity,
            price: None,
            stop_price: Some(stop_price),
            time_in_force: TimeInForce::GTC,
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
            average_fill_price: None,
            created_at: now,
            updated_at: now,
        }
    }

    /// Set time in force.
    pub fn with_time_in_force(mut self, tif: TimeInForce) -> Self {
        self.time_in_force = tif;
        self
    }

    /// Set custom client order ID.
    pub fn with_client_order_id(mut self, id: impl Into<String>) -> Self {
        self.client_order_id = id.into();
        self
    }

    /// Get remaining unfilled quantity.
    pub fn remaining_quantity(&self) -> f64 {
        self.quantity - self.filled_quantity
    }

    /// Check if order is fully filled.
    pub fn is_filled(&self) -> bool {
        self.status == OrderStatus::Filled
    }
}

/// Represents a trade/fill.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    /// Trade ID.
    pub trade_id: String,
    /// Associated order ID.
    pub order_id: String,
    /// Trading symbol.
    pub symbol: String,
    /// Trade side.
    pub side: Side,
    /// Executed price.
    pub price: f64,
    /// Executed quantity.
    pub quantity: f64,
    /// Commission paid.
    pub commission: f64,
    /// Commission asset.
    pub commission_asset: String,
    /// Timestamp (Unix ms).
    pub timestamp: u64,
    /// Whether this trade is the maker side.
    pub is_maker: bool,
}

/// Represents a position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    /// Symbol.
    pub symbol: String,
    /// Position size (positive for long, negative for short).
    pub size: f64,
    /// Average entry price.
    pub entry_price: f64,
    /// Current mark price.
    pub mark_price: f64,
    /// Unrealized PnL.
    pub unrealized_pnl: f64,
    /// Realized PnL.
    pub realized_pnl: f64,
    /// Liquidation price (for futures).
    pub liquidation_price: Option<f64>,
    /// Leverage (for futures).
    pub leverage: Option<f64>,
    /// Margin used (for futures).
    pub margin: Option<f64>,
    /// Last update timestamp.
    pub updated_at: u64,
}

impl Position {
    /// Create a new empty position.
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
            size: 0.0,
            entry_price: 0.0,
            mark_price: 0.0,
            unrealized_pnl: 0.0,
            realized_pnl: 0.0,
            liquidation_price: None,
            leverage: None,
            margin: None,
            updated_at: current_timestamp_ms(),
        }
    }

    /// Check if position is long.
    pub fn is_long(&self) -> bool {
        self.size > 0.0
    }

    /// Check if position is short.
    pub fn is_short(&self) -> bool {
        self.size < 0.0
    }

    /// Check if position is flat (no position).
    pub fn is_flat(&self) -> bool {
        self.size.abs() < 1e-10
    }

    /// Calculate notional value.
    pub fn notional_value(&self) -> f64 {
        self.size.abs() * self.mark_price
    }
}

/// Account balance for a single asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    /// Asset symbol (e.g., "BTC", "USDT").
    pub asset: String,
    /// Free/available balance.
    pub free: f64,
    /// Locked balance (in open orders).
    pub locked: f64,
}

impl Balance {
    /// Get total balance.
    pub fn total(&self) -> f64 {
        self.free + self.locked
    }
}

/// Order book level.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct OrderBookLevel {
    /// Price level.
    pub price: f64,
    /// Quantity at this level.
    pub quantity: f64,
}

/// Order book snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    /// Symbol.
    pub symbol: String,
    /// Bid levels (sorted by price descending).
    pub bids: Vec<OrderBookLevel>,
    /// Ask levels (sorted by price ascending).
    pub asks: Vec<OrderBookLevel>,
    /// Last update ID from exchange.
    pub last_update_id: u64,
    /// Timestamp (Unix ms).
    pub timestamp: u64,
}

impl OrderBook {
    /// Get best bid price.
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|l| l.price)
    }

    /// Get best ask price.
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|l| l.price)
    }

    /// Get mid price.
    pub fn mid_price(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some((bid + ask) / 2.0),
            _ => None,
        }
    }

    /// Get spread.
    pub fn spread(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(bid), Some(ask)) => Some(ask - bid),
            _ => None,
        }
    }

    /// Get spread as percentage of mid price.
    pub fn spread_pct(&self) -> Option<f64> {
        match (self.spread(), self.mid_price()) {
            (Some(spread), Some(mid)) if mid > 0.0 => Some(spread / mid * 100.0),
            _ => None,
        }
    }
}

/// OHLCV candle data.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Candle {
    /// Open timestamp (Unix ms).
    pub timestamp: u64,
    /// Open price.
    pub open: f64,
    /// High price.
    pub high: f64,
    /// Low price.
    pub low: f64,
    /// Close price.
    pub close: f64,
    /// Volume.
    pub volume: f64,
}

/// Market ticker data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker {
    /// Symbol.
    pub symbol: String,
    /// Last traded price.
    pub last_price: f64,
    /// 24h price change.
    pub price_change: f64,
    /// 24h price change percentage.
    pub price_change_pct: f64,
    /// 24h high.
    pub high_24h: f64,
    /// 24h low.
    pub low_24h: f64,
    /// 24h volume.
    pub volume_24h: f64,
    /// 24h quote volume.
    pub quote_volume_24h: f64,
    /// Best bid price.
    pub bid_price: f64,
    /// Best ask price.
    pub ask_price: f64,
    /// Timestamp (Unix ms).
    pub timestamp: u64,
}

/// Timeframe for candles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Interval {
    /// 1 minute.
    M1,
    /// 3 minutes.
    M3,
    /// 5 minutes.
    M5,
    /// 15 minutes.
    M15,
    /// 30 minutes.
    M30,
    /// 1 hour.
    H1,
    /// 2 hours.
    H2,
    /// 4 hours.
    H4,
    /// 6 hours.
    H6,
    /// 8 hours.
    H8,
    /// 12 hours.
    H12,
    /// 1 day.
    D1,
    /// 3 days.
    D3,
    /// 1 week.
    W1,
    /// 1 month.
    Mo1,
}

impl Interval {
    /// Convert to milliseconds.
    pub fn to_millis(&self) -> u64 {
        match self {
            Interval::M1 => 60_000,
            Interval::M3 => 180_000,
            Interval::M5 => 300_000,
            Interval::M15 => 900_000,
            Interval::M30 => 1_800_000,
            Interval::H1 => 3_600_000,
            Interval::H2 => 7_200_000,
            Interval::H4 => 14_400_000,
            Interval::H6 => 21_600_000,
            Interval::H8 => 28_800_000,
            Interval::H12 => 43_200_000,
            Interval::D1 => 86_400_000,
            Interval::D3 => 259_200_000,
            Interval::W1 => 604_800_000,
            Interval::Mo1 => 2_592_000_000,
        }
    }
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Interval::M1 => write!(f, "1m"),
            Interval::M3 => write!(f, "3m"),
            Interval::M5 => write!(f, "5m"),
            Interval::M15 => write!(f, "15m"),
            Interval::M30 => write!(f, "30m"),
            Interval::H1 => write!(f, "1h"),
            Interval::H2 => write!(f, "2h"),
            Interval::H4 => write!(f, "4h"),
            Interval::H6 => write!(f, "6h"),
            Interval::H8 => write!(f, "8h"),
            Interval::H12 => write!(f, "12h"),
            Interval::D1 => write!(f, "1d"),
            Interval::D3 => write!(f, "3d"),
            Interval::W1 => write!(f, "1w"),
            Interval::Mo1 => write!(f, "1M"),
        }
    }
}

/// Get current Unix timestamp in milliseconds.
pub fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

/// Generate a unique client order ID.
pub fn generate_client_order_id() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let timestamp = current_timestamp_ms();
    let random: u32 = rng.gen();
    format!("octane_{timestamp}_{random:08x}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_order_creation() {
        let order = Order::market("BTCUSDT", Side::Buy, 0.1);
        assert_eq!(order.symbol, "BTCUSDT");
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.quantity, 0.1);
        assert_eq!(order.status, OrderStatus::Pending);
    }

    #[test]
    fn test_order_book() {
        let book = OrderBook {
            symbol: "BTCUSDT".to_string(),
            bids: vec![
                OrderBookLevel {
                    price: 50000.0,
                    quantity: 1.0,
                },
                OrderBookLevel {
                    price: 49990.0,
                    quantity: 2.0,
                },
            ],
            asks: vec![
                OrderBookLevel {
                    price: 50010.0,
                    quantity: 1.5,
                },
                OrderBookLevel {
                    price: 50020.0,
                    quantity: 2.5,
                },
            ],
            last_update_id: 12345,
            timestamp: current_timestamp_ms(),
        };

        assert_eq!(book.best_bid(), Some(50000.0));
        assert_eq!(book.best_ask(), Some(50010.0));
        assert_eq!(book.mid_price(), Some(50005.0));
        assert_eq!(book.spread(), Some(10.0));
    }

    #[test]
    fn test_position() {
        let pos = Position {
            symbol: "BTCUSDT".to_string(),
            size: 1.0,
            entry_price: 50000.0,
            mark_price: 51000.0,
            unrealized_pnl: 1000.0,
            realized_pnl: 0.0,
            liquidation_price: None,
            leverage: None,
            margin: None,
            updated_at: current_timestamp_ms(),
        };

        assert!(pos.is_long());
        assert!(!pos.is_short());
        assert!(!pos.is_flat());
        assert_eq!(pos.notional_value(), 51000.0);
    }
}
