//! Exchange connectors for live trading.
//!
//! This module provides a trait-based abstraction for connecting to
//! cryptocurrency exchanges, with implementations for:
//! - Binance (Spot and Futures)
//! - Bybit
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::live::exchanges::{ExchangeConnector, BinanceConnector, BinanceConfig};
//!
//! let config = BinanceConfig::default()
//!     .api_key("your_api_key")
//!     .api_secret("your_api_secret")
//!     .testnet(true);
//!
//! let connector = BinanceConnector::new(config);
//! connector.connect().await?;
//!
//! let balances = connector.get_balances().await?;
//! ```

pub mod binance;
pub mod bybit;

pub use binance::{BinanceConfig, BinanceConnector, BinanceMarketType};
pub use bybit::{BybitAccountType, BybitCategory, BybitConfig, BybitConnector};

use crate::live::error::Result;
use crate::live::types::{
    Balance, Candle, Interval, Order, OrderBook, Position, Ticker, Trade,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Exchange connection status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConnectionStatus {
    /// Not connected.
    Disconnected,
    /// Currently connecting.
    Connecting,
    /// Connected and ready.
    Connected,
    /// Connection lost, attempting to reconnect.
    Reconnecting,
    /// Connection error.
    Error,
}

/// Exchange information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeInfo {
    /// Exchange name.
    pub name: String,
    /// Exchange identifier.
    pub id: String,
    /// API version.
    pub api_version: String,
    /// Server time (Unix ms).
    pub server_time: u64,
    /// Supported symbols.
    pub symbols: Vec<SymbolInfo>,
    /// Rate limits.
    pub rate_limits: Vec<RateLimit>,
}

/// Symbol/market information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolInfo {
    /// Symbol name (e.g., "BTCUSDT").
    pub symbol: String,
    /// Base asset (e.g., "BTC").
    pub base_asset: String,
    /// Quote asset (e.g., "USDT").
    pub quote_asset: String,
    /// Price precision (decimal places).
    pub price_precision: u8,
    /// Quantity precision (decimal places).
    pub quantity_precision: u8,
    /// Minimum order quantity.
    pub min_quantity: f64,
    /// Maximum order quantity.
    pub max_quantity: f64,
    /// Step size for quantity.
    pub step_size: f64,
    /// Minimum notional value.
    pub min_notional: f64,
    /// Whether trading is enabled.
    pub trading_enabled: bool,
}

/// Rate limit information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimit {
    /// Limit type.
    pub limit_type: RateLimitType,
    /// Interval in seconds.
    pub interval_seconds: u64,
    /// Maximum requests in interval.
    pub limit: u64,
}

/// Rate limit type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RateLimitType {
    /// Request weight limit.
    RequestWeight,
    /// Order rate limit.
    Orders,
    /// Raw requests limit.
    RawRequests,
}

/// Order book update from WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookUpdate {
    /// Symbol.
    pub symbol: String,
    /// Update ID.
    pub update_id: u64,
    /// Bid updates.
    pub bids: Vec<(f64, f64)>,
    /// Ask updates.
    pub asks: Vec<(f64, f64)>,
    /// Timestamp.
    pub timestamp: u64,
}

/// Trade update from WebSocket.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeUpdate {
    /// Symbol.
    pub symbol: String,
    /// Trade ID.
    pub trade_id: u64,
    /// Price.
    pub price: f64,
    /// Quantity.
    pub quantity: f64,
    /// Buyer is maker.
    pub is_buyer_maker: bool,
    /// Timestamp.
    pub timestamp: u64,
}

/// Callback for order book WebSocket events.
pub type OrderBookCallback = Box<dyn Fn(OrderBookUpdate) + Send + Sync>;
/// Callback for trade WebSocket events.
pub type TradeCallback = Box<dyn Fn(TradeUpdate) + Send + Sync>;
/// Callback for order updates WebSocket events.
pub type OrderCallback = Box<dyn Fn(Order) + Send + Sync>;
/// Callback for position updates WebSocket events.
pub type PositionCallback = Box<dyn Fn(Position) + Send + Sync>;

/// Trait for exchange connectors.
///
/// Provides a unified interface for interacting with cryptocurrency exchanges.
/// Implementations handle authentication, rate limiting, and error handling.
#[async_trait]
pub trait ExchangeConnector: Send + Sync {
    /// Get exchange name.
    fn name(&self) -> &str;

    /// Get current connection status.
    fn status(&self) -> ConnectionStatus;

    /// Connect to the exchange.
    async fn connect(&mut self) -> Result<()>;

    /// Disconnect from the exchange.
    async fn disconnect(&mut self) -> Result<()>;

    /// Ping the exchange to check connection.
    async fn ping(&self) -> Result<u64>;

    /// Get exchange information.
    async fn get_exchange_info(&self) -> Result<ExchangeInfo>;

    /// Get server time.
    async fn get_server_time(&self) -> Result<u64>;

    // --- Account endpoints ---

    /// Get account balances.
    async fn get_balances(&self) -> Result<Vec<Balance>>;

    /// Get balance for a specific asset.
    async fn get_balance(&self, asset: &str) -> Result<Balance>;

    /// Get all positions.
    async fn get_positions(&self) -> Result<Vec<Position>>;

    /// Get position for a specific symbol.
    async fn get_position(&self, symbol: &str) -> Result<Position>;

    // --- Order endpoints ---

    /// Place an order.
    async fn place_order(&self, order: Order) -> Result<Order>;

    /// Cancel an order.
    async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<Order>;

    /// Cancel all orders for a symbol.
    async fn cancel_all_orders(&self, symbol: &str) -> Result<Vec<Order>>;

    /// Get order status.
    async fn get_order(&self, symbol: &str, order_id: &str) -> Result<Order>;

    /// Get open orders.
    async fn get_open_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>>;

    /// Get order history.
    async fn get_order_history(
        &self,
        symbol: &str,
        start_time: Option<u64>,
        end_time: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<Order>>;

    /// Get trade history.
    async fn get_trade_history(
        &self,
        symbol: &str,
        start_time: Option<u64>,
        end_time: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<Trade>>;

    // --- Market data endpoints ---

    /// Get order book.
    async fn get_order_book(&self, symbol: &str, depth: Option<usize>) -> Result<OrderBook>;

    /// Get recent trades.
    async fn get_recent_trades(&self, symbol: &str, limit: Option<usize>) -> Result<Vec<Trade>>;

    /// Get ticker.
    async fn get_ticker(&self, symbol: &str) -> Result<Ticker>;

    /// Get all tickers.
    async fn get_all_tickers(&self) -> Result<Vec<Ticker>>;

    /// Get historical candles/klines.
    async fn get_candles(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: Option<u64>,
        end_time: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<Candle>>;

    // --- WebSocket subscriptions ---

    /// Subscribe to order book updates.
    async fn subscribe_orderbook(
        &mut self,
        symbols: &[&str],
        callback: OrderBookCallback,
    ) -> Result<()>;

    /// Subscribe to trade updates.
    async fn subscribe_trades(&mut self, symbols: &[&str], callback: TradeCallback) -> Result<()>;

    /// Subscribe to user order updates.
    async fn subscribe_orders(&mut self, callback: OrderCallback) -> Result<()>;

    /// Subscribe to position updates.
    async fn subscribe_positions(&mut self, callback: PositionCallback) -> Result<()>;

    /// Unsubscribe from all streams.
    async fn unsubscribe_all(&mut self) -> Result<()>;
}

/// Rate limiter for API requests.
#[derive(Debug)]
pub struct RateLimiter {
    /// Request weights by endpoint.
    weights: HashMap<String, u64>,
    /// Current weight used.
    current_weight: u64,
    /// Weight limit.
    weight_limit: u64,
    /// Order count.
    order_count: u64,
    /// Order limit.
    order_limit: u64,
    /// Last reset time.
    last_reset: std::time::Instant,
    /// Reset interval.
    reset_interval: std::time::Duration,
}

impl RateLimiter {
    /// Create a new rate limiter.
    pub fn new(weight_limit: u64, order_limit: u64, reset_seconds: u64) -> Self {
        Self {
            weights: HashMap::new(),
            current_weight: 0,
            weight_limit,
            order_count: 0,
            order_limit,
            last_reset: std::time::Instant::now(),
            reset_interval: std::time::Duration::from_secs(reset_seconds),
        }
    }

    /// Check if we can make a request with given weight.
    pub fn can_request(&self, weight: u64) -> bool {
        self.current_weight + weight <= self.weight_limit
    }

    /// Check if we can place an order.
    pub fn can_order(&self) -> bool {
        self.order_count < self.order_limit
    }

    /// Record a request.
    pub fn record_request(&mut self, weight: u64) {
        self.maybe_reset();
        self.current_weight += weight;
    }

    /// Record an order.
    pub fn record_order(&mut self) {
        self.maybe_reset();
        self.order_count += 1;
    }

    /// Get wait time in milliseconds until we can make a request.
    pub fn wait_time_ms(&self, weight: u64) -> u64 {
        if self.can_request(weight) {
            return 0;
        }
        let elapsed = self.last_reset.elapsed();
        if elapsed >= self.reset_interval {
            return 0;
        }
        (self.reset_interval - elapsed).as_millis() as u64
    }

    fn maybe_reset(&mut self) {
        if self.last_reset.elapsed() >= self.reset_interval {
            self.current_weight = 0;
            self.order_count = 0;
            self.last_reset = std::time::Instant::now();
        }
    }
}

/// Authentication credentials.
#[derive(Debug, Clone)]
pub struct ApiCredentials {
    /// API key.
    pub api_key: String,
    /// API secret.
    pub api_secret: String,
    /// Optional passphrase (for some exchanges).
    pub passphrase: Option<String>,
}

impl ApiCredentials {
    /// Create new credentials.
    pub fn new(api_key: impl Into<String>, api_secret: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            api_secret: api_secret.into(),
            passphrase: None,
        }
    }

    /// Set passphrase.
    pub fn with_passphrase(mut self, passphrase: impl Into<String>) -> Self {
        self.passphrase = Some(passphrase.into());
        self
    }
}

/// HMAC-SHA256 signature helper.
pub fn hmac_sha256(secret: &str, message: &str) -> String {
    use std::fmt::Write;

    // Simple HMAC-SHA256 implementation placeholder
    // In production, use a proper crypto library like `hmac` + `sha2`
    let mut result = String::new();
    for byte in secret.as_bytes().iter().zip(message.as_bytes().iter().cycle()) {
        write!(&mut result, "{:02x}", byte.0 ^ byte.1).unwrap();
    }

    // Pad to expected length
    while result.len() < 64 {
        result.push('0');
    }
    result.truncate(64);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(100, 10, 60);

        assert!(limiter.can_request(50));
        limiter.record_request(50);
        assert!(limiter.can_request(50));
        limiter.record_request(50);
        assert!(!limiter.can_request(1));
    }

    #[test]
    fn test_api_credentials() {
        let creds = ApiCredentials::new("key", "secret").with_passphrase("pass");
        assert_eq!(creds.api_key, "key");
        assert_eq!(creds.api_secret, "secret");
        assert_eq!(creds.passphrase, Some("pass".to_string()));
    }
}
