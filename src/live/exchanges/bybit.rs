//! Bybit exchange connector.
//!
//! Supports Bybit's unified API for spot and derivatives trading.
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::live::exchanges::{BybitConnector, BybitConfig};
//!
//! let config = BybitConfig::default()
//!     .api_key("your_api_key")
//!     .api_secret("your_api_secret")
//!     .testnet(true);
//!
//! let mut connector = BybitConnector::new(config);
//! connector.connect().await?;
//! ```

use crate::live::error::{LiveTradingError, Result};
use crate::live::exchanges::{
    hmac_sha256, ApiCredentials, ConnectionStatus, ExchangeConnector, ExchangeInfo,
    OrderBookCallback, OrderCallback, PositionCallback, RateLimit, RateLimitType, RateLimiter,
    SymbolInfo, TradeCallback,
};
use crate::live::types::{
    current_timestamp_ms, Balance, Candle, Interval, Order, OrderBook, OrderStatus, OrderType,
    Position, Side, Ticker, TimeInForce, Trade,
};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Bybit account type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BybitAccountType {
    /// Unified Trading Account.
    Unified,
    /// Standard (non-unified) account.
    Standard,
    /// Spot account.
    Spot,
}

impl Default for BybitAccountType {
    fn default() -> Self {
        BybitAccountType::Unified
    }
}

/// Bybit category (market type).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BybitCategory {
    /// Spot trading.
    Spot,
    /// Linear perpetuals (USDT).
    Linear,
    /// Inverse perpetuals.
    Inverse,
    /// Options.
    Option,
}

impl Default for BybitCategory {
    fn default() -> Self {
        BybitCategory::Linear
    }
}

impl BybitCategory {
    fn to_string(&self) -> &'static str {
        match self {
            BybitCategory::Spot => "spot",
            BybitCategory::Linear => "linear",
            BybitCategory::Inverse => "inverse",
            BybitCategory::Option => "option",
        }
    }
}

/// Bybit connector configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitConfig {
    /// API credentials.
    #[serde(skip)]
    pub credentials: Option<ApiCredentials>,
    /// Account type.
    pub account_type: BybitAccountType,
    /// Default category for trading.
    pub default_category: BybitCategory,
    /// Use testnet.
    pub testnet: bool,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Receive window for signed requests.
    pub recv_window: u64,
    /// Enable WebSocket streams.
    pub enable_websocket: bool,
    /// WebSocket ping interval in seconds.
    pub ws_ping_interval: u64,
    /// Auto-reconnect on disconnect.
    pub auto_reconnect: bool,
    /// Maximum reconnection attempts.
    pub max_reconnect_attempts: u32,
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            credentials: None,
            account_type: BybitAccountType::Unified,
            default_category: BybitCategory::Linear,
            testnet: false,
            timeout_ms: 5000,
            recv_window: 5000,
            enable_websocket: true,
            ws_ping_interval: 20,
            auto_reconnect: true,
            max_reconnect_attempts: 5,
        }
    }
}

impl BybitConfig {
    /// Create new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set API key.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        let secret = self
            .credentials
            .as_ref()
            .map(|c| c.api_secret.clone())
            .unwrap_or_default();
        self.credentials = Some(ApiCredentials::new(key, secret));
        self
    }

    /// Set API secret.
    pub fn api_secret(mut self, secret: impl Into<String>) -> Self {
        let key = self
            .credentials
            .as_ref()
            .map(|c| c.api_key.clone())
            .unwrap_or_default();
        self.credentials = Some(ApiCredentials::new(key, secret));
        self
    }

    /// Set account type.
    pub fn account_type(mut self, account_type: BybitAccountType) -> Self {
        self.account_type = account_type;
        self
    }

    /// Set default category.
    pub fn default_category(mut self, category: BybitCategory) -> Self {
        self.default_category = category;
        self
    }

    /// Enable testnet.
    pub fn testnet(mut self, enabled: bool) -> Self {
        self.testnet = enabled;
        self
    }

    /// Set request timeout.
    pub fn timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Set receive window.
    pub fn recv_window(mut self, ms: u64) -> Self {
        self.recv_window = ms;
        self
    }

    /// Enable WebSocket.
    pub fn enable_websocket(mut self, enabled: bool) -> Self {
        self.enable_websocket = enabled;
        self
    }

    /// Set auto-reconnect.
    pub fn auto_reconnect(mut self, enabled: bool) -> Self {
        self.auto_reconnect = enabled;
        self
    }
}

/// Internal state for the connector.
struct BybitState {
    /// Connection status.
    status: ConnectionStatus,
    /// Rate limiter.
    rate_limiter: RateLimiter,
    /// Cached exchange info.
    exchange_info: Option<ExchangeInfo>,
    /// Symbol info cache.
    symbol_info: HashMap<String, SymbolInfo>,
}

/// Bybit exchange connector.
pub struct BybitConnector {
    /// Configuration.
    config: BybitConfig,
    /// Internal state.
    state: Arc<RwLock<BybitState>>,
}

impl BybitConnector {
    /// Create a new Bybit connector.
    pub fn new(config: BybitConfig) -> Self {
        Self {
            config,
            state: Arc::new(RwLock::new(BybitState {
                status: ConnectionStatus::Disconnected,
                rate_limiter: RateLimiter::new(120, 100, 60),
                exchange_info: None,
                symbol_info: HashMap::new(),
            })),
        }
    }

    /// Get the REST API base URL.
    fn rest_base_url(&self) -> &str {
        if self.config.testnet {
            "https://api-testnet.bybit.com"
        } else {
            "https://api.bybit.com"
        }
    }

    /// Get the WebSocket base URL.
    fn ws_base_url(&self) -> &str {
        if self.config.testnet {
            match self.config.default_category {
                BybitCategory::Spot => "wss://stream-testnet.bybit.com/v5/public/spot",
                BybitCategory::Linear => "wss://stream-testnet.bybit.com/v5/public/linear",
                BybitCategory::Inverse => "wss://stream-testnet.bybit.com/v5/public/inverse",
                BybitCategory::Option => "wss://stream-testnet.bybit.com/v5/public/option",
            }
        } else {
            match self.config.default_category {
                BybitCategory::Spot => "wss://stream.bybit.com/v5/public/spot",
                BybitCategory::Linear => "wss://stream.bybit.com/v5/public/linear",
                BybitCategory::Inverse => "wss://stream.bybit.com/v5/public/inverse",
                BybitCategory::Option => "wss://stream.bybit.com/v5/public/option",
            }
        }
    }

    /// Get private WebSocket URL.
    fn ws_private_url(&self) -> &str {
        if self.config.testnet {
            "wss://stream-testnet.bybit.com/v5/private"
        } else {
            "wss://stream.bybit.com/v5/private"
        }
    }

    /// Sign a request for Bybit V5 API.
    fn sign_request(&self, timestamp: u64, params: &str) -> Result<String> {
        let credentials = self.config.credentials.as_ref().ok_or_else(|| {
            LiveTradingError::Authentication("API credentials not configured".into())
        })?;

        let recv_window = self.config.recv_window;
        let sign_str = format!(
            "{}{}{}{}",
            timestamp, credentials.api_key, recv_window, params
        );

        Ok(hmac_sha256(&credentials.api_secret, &sign_str))
    }

    /// Convert interval to Bybit string.
    fn interval_to_string(interval: Interval) -> &'static str {
        match interval {
            Interval::M1 => "1",
            Interval::M3 => "3",
            Interval::M5 => "5",
            Interval::M15 => "15",
            Interval::M30 => "30",
            Interval::H1 => "60",
            Interval::H2 => "120",
            Interval::H4 => "240",
            Interval::H6 => "360",
            Interval::H12 => "720",
            Interval::D1 => "D",
            Interval::W1 => "W",
            Interval::Mo1 => "M",
            _ => "60", // Default to 1 hour for unsupported
        }
    }

    /// Convert order type to Bybit string.
    fn order_type_to_string(order_type: OrderType) -> &'static str {
        match order_type {
            OrderType::Market => "Market",
            OrderType::Limit => "Limit",
            OrderType::StopLoss | OrderType::StopLimit => "Stop",
            OrderType::TakeProfit => "TakeProfit",
            OrderType::TrailingStop => "TrailingStop",
        }
    }

    /// Convert time in force to Bybit string.
    fn tif_to_string(tif: TimeInForce) -> &'static str {
        match tif {
            TimeInForce::GTC => "GTC",
            TimeInForce::IOC => "IOC",
            TimeInForce::FOK => "FOK",
            TimeInForce::GTD(_) => "GTC",
        }
    }

    /// Parse order status from Bybit string.
    fn parse_order_status(status: &str) -> OrderStatus {
        match status {
            "New" => OrderStatus::Submitted,
            "PartiallyFilled" => OrderStatus::PartiallyFilled,
            "Filled" => OrderStatus::Filled,
            "Cancelled" => OrderStatus::Cancelled,
            "Rejected" => OrderStatus::Rejected,
            "Expired" => OrderStatus::Expired,
            _ => OrderStatus::Pending,
        }
    }

    // Simulated API call helpers

    async fn api_get(
        &self,
        _endpoint: &str,
        _params: &[(&str, &str)],
    ) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"retCode": 0, "result": {}}))
    }

    async fn api_post(
        &self,
        _endpoint: &str,
        _body: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"retCode": 0, "result": {}}))
    }

    async fn api_get_signed(
        &self,
        _endpoint: &str,
        _params: &[(&str, &str)],
    ) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"retCode": 0, "result": {}}))
    }

    async fn api_post_signed(
        &self,
        _endpoint: &str,
        _body: &serde_json::Value,
    ) -> Result<serde_json::Value> {
        Ok(serde_json::json!({"retCode": 0, "result": {}}))
    }
}

#[async_trait]
impl ExchangeConnector for BybitConnector {
    fn name(&self) -> &str {
        match self.config.default_category {
            BybitCategory::Spot => "Bybit Spot",
            BybitCategory::Linear => "Bybit Linear Perpetuals",
            BybitCategory::Inverse => "Bybit Inverse Perpetuals",
            BybitCategory::Option => "Bybit Options",
        }
    }

    fn status(&self) -> ConnectionStatus {
        ConnectionStatus::Disconnected
    }

    async fn connect(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        state.status = ConnectionStatus::Connecting;

        // In production:
        // 1. Test connection with server time endpoint
        // 2. Get exchange info
        // 3. Connect to WebSocket if enabled

        state.status = ConnectionStatus::Connected;
        Ok(())
    }

    async fn disconnect(&mut self) -> Result<()> {
        let mut state = self.state.write().await;
        state.status = ConnectionStatus::Disconnected;
        Ok(())
    }

    async fn ping(&self) -> Result<u64> {
        let start = std::time::Instant::now();
        let _response = self.api_get("/v5/market/time", &[]).await?;
        Ok(start.elapsed().as_millis() as u64)
    }

    async fn get_exchange_info(&self) -> Result<ExchangeInfo> {
        let category = self.config.default_category.to_string();
        let _response = self
            .api_get("/v5/market/instruments-info", &[("category", category)])
            .await?;

        Ok(ExchangeInfo {
            name: self.name().to_string(),
            id: "bybit".to_string(),
            api_version: "v5".to_string(),
            server_time: current_timestamp_ms(),
            symbols: Vec::new(),
            rate_limits: vec![RateLimit {
                limit_type: RateLimitType::RequestWeight,
                interval_seconds: 60,
                limit: 120,
            }],
        })
    }

    async fn get_server_time(&self) -> Result<u64> {
        let _response = self.api_get("/v5/market/time", &[]).await?;
        Ok(current_timestamp_ms())
    }

    async fn get_balances(&self) -> Result<Vec<Balance>> {
        let account_type = match self.config.account_type {
            BybitAccountType::Unified => "UNIFIED",
            BybitAccountType::Standard => "CONTRACT",
            BybitAccountType::Spot => "SPOT",
        };

        let _response = self
            .api_get_signed(
                "/v5/account/wallet-balance",
                &[("accountType", account_type)],
            )
            .await?;

        Ok(Vec::new())
    }

    async fn get_balance(&self, asset: &str) -> Result<Balance> {
        let balances = self.get_balances().await?;
        balances
            .into_iter()
            .find(|b| b.asset == asset)
            .ok_or_else(|| LiveTradingError::SymbolNotFound(asset.to_string()))
    }

    async fn get_positions(&self) -> Result<Vec<Position>> {
        let category = self.config.default_category.to_string();
        let _response = self
            .api_get_signed("/v5/position/list", &[("category", category)])
            .await?;

        Ok(Vec::new())
    }

    async fn get_position(&self, symbol: &str) -> Result<Position> {
        let category = self.config.default_category.to_string();
        let _response = self
            .api_get_signed(
                "/v5/position/list",
                &[("category", category), ("symbol", symbol)],
            )
            .await?;

        Ok(Position::new(symbol))
    }

    async fn place_order(&self, order: Order) -> Result<Order> {
        let category = self.config.default_category.to_string();
        let side = match order.side {
            Side::Buy => "Buy",
            Side::Sell => "Sell",
        };
        let order_type = Self::order_type_to_string(order.order_type);
        let tif = Self::tif_to_string(order.time_in_force);

        let mut body = serde_json::json!({
            "category": category,
            "symbol": order.symbol,
            "side": side,
            "orderType": order_type,
            "qty": order.quantity.to_string(),
            "timeInForce": tif,
        });

        if let Some(price) = order.price {
            body["price"] = serde_json::json!(price.to_string());
        }

        if let Some(stop_price) = order.stop_price {
            body["triggerPrice"] = serde_json::json!(stop_price.to_string());
        }

        let _response = self.api_post_signed("/v5/order/create", &body).await?;

        let mut result = order;
        result.status = OrderStatus::Submitted;
        result.updated_at = current_timestamp_ms();
        Ok(result)
    }

    async fn cancel_order(&self, symbol: &str, order_id: &str) -> Result<Order> {
        let category = self.config.default_category.to_string();
        let body = serde_json::json!({
            "category": category,
            "symbol": symbol,
            "orderLinkId": order_id,
        });

        let _response = self.api_post_signed("/v5/order/cancel", &body).await?;

        Ok(Order::market(symbol, Side::Buy, 0.0))
    }

    async fn cancel_all_orders(&self, symbol: &str) -> Result<Vec<Order>> {
        let category = self.config.default_category.to_string();
        let body = serde_json::json!({
            "category": category,
            "symbol": symbol,
        });

        let _response = self.api_post_signed("/v5/order/cancel-all", &body).await?;

        Ok(Vec::new())
    }

    async fn get_order(&self, symbol: &str, order_id: &str) -> Result<Order> {
        let category = self.config.default_category.to_string();
        let _response = self
            .api_get_signed(
                "/v5/order/realtime",
                &[
                    ("category", category),
                    ("symbol", symbol),
                    ("orderLinkId", order_id),
                ],
            )
            .await?;

        Ok(Order::market(symbol, Side::Buy, 0.0))
    }

    async fn get_open_orders(&self, symbol: Option<&str>) -> Result<Vec<Order>> {
        let category = self.config.default_category.to_string();
        let symbol_str = symbol.map(|s| s.to_string());

        let mut params: Vec<(&str, &str)> = vec![("category", &category)];
        if let Some(ref sym) = symbol_str {
            params.push(("symbol", sym));
        }

        let _response = self.api_get_signed("/v5/order/realtime", &params).await?;

        Ok(Vec::new())
    }

    async fn get_order_history(
        &self,
        symbol: &str,
        start_time: Option<u64>,
        end_time: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<Order>> {
        let category = self.config.default_category.to_string();
        let symbol_str = symbol.to_string();
        let start_str = start_time.map(|s| s.to_string());
        let end_str = end_time.map(|e| e.to_string());
        let limit_str = limit.map(|l| l.to_string());

        let mut params: Vec<(&str, &str)> = vec![("category", &category), ("symbol", &symbol_str)];
        if let Some(ref start) = start_str {
            params.push(("startTime", start));
        }
        if let Some(ref end) = end_str {
            params.push(("endTime", end));
        }
        if let Some(ref lim) = limit_str {
            params.push(("limit", lim));
        }

        let _response = self.api_get_signed("/v5/order/history", &params).await?;

        Ok(Vec::new())
    }

    async fn get_trade_history(
        &self,
        symbol: &str,
        start_time: Option<u64>,
        end_time: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<Trade>> {
        let category = self.config.default_category.to_string();
        let symbol_str = symbol.to_string();
        let start_str = start_time.map(|s| s.to_string());
        let end_str = end_time.map(|e| e.to_string());
        let limit_str = limit.map(|l| l.to_string());

        let mut params: Vec<(&str, &str)> = vec![("category", &category), ("symbol", &symbol_str)];
        if let Some(ref start) = start_str {
            params.push(("startTime", start));
        }
        if let Some(ref end) = end_str {
            params.push(("endTime", end));
        }
        if let Some(ref lim) = limit_str {
            params.push(("limit", lim));
        }

        let _response = self.api_get_signed("/v5/execution/list", &params).await?;

        Ok(Vec::new())
    }

    async fn get_order_book(&self, symbol: &str, depth: Option<usize>) -> Result<OrderBook> {
        let category = self.config.default_category.to_string();
        let depth_str = depth.unwrap_or(50).to_string();
        let _response = self
            .api_get(
                "/v5/market/orderbook",
                &[
                    ("category", &category),
                    ("symbol", symbol),
                    ("limit", &depth_str),
                ],
            )
            .await?;

        Ok(OrderBook {
            symbol: symbol.to_string(),
            bids: Vec::new(),
            asks: Vec::new(),
            last_update_id: 0,
            timestamp: current_timestamp_ms(),
        })
    }

    async fn get_recent_trades(&self, symbol: &str, limit: Option<usize>) -> Result<Vec<Trade>> {
        let category = self.config.default_category.to_string();
        let limit_str = limit.unwrap_or(60).to_string();
        let _response = self
            .api_get(
                "/v5/market/recent-trade",
                &[
                    ("category", &category),
                    ("symbol", symbol),
                    ("limit", &limit_str),
                ],
            )
            .await?;

        Ok(Vec::new())
    }

    async fn get_ticker(&self, symbol: &str) -> Result<Ticker> {
        let category = self.config.default_category.to_string();
        let _response = self
            .api_get(
                "/v5/market/tickers",
                &[("category", &category), ("symbol", symbol)],
            )
            .await?;

        Ok(Ticker {
            symbol: symbol.to_string(),
            last_price: 0.0,
            price_change: 0.0,
            price_change_pct: 0.0,
            high_24h: 0.0,
            low_24h: 0.0,
            volume_24h: 0.0,
            quote_volume_24h: 0.0,
            bid_price: 0.0,
            ask_price: 0.0,
            timestamp: current_timestamp_ms(),
        })
    }

    async fn get_all_tickers(&self) -> Result<Vec<Ticker>> {
        let category = self.config.default_category.to_string();
        let _response = self
            .api_get("/v5/market/tickers", &[("category", &category)])
            .await?;

        Ok(Vec::new())
    }

    async fn get_candles(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: Option<u64>,
        end_time: Option<u64>,
        limit: Option<usize>,
    ) -> Result<Vec<Candle>> {
        let category = self.config.default_category.to_string();
        let interval_str = Self::interval_to_string(interval).to_string();
        let symbol_str = symbol.to_string();
        let start_str = start_time.map(|s| s.to_string());
        let end_str = end_time.map(|e| e.to_string());
        let limit_str = limit.map(|l| l.to_string());

        let mut params: Vec<(&str, &str)> = vec![
            ("category", &category),
            ("symbol", &symbol_str),
            ("interval", &interval_str),
        ];
        if let Some(ref start) = start_str {
            params.push(("start", start));
        }
        if let Some(ref end) = end_str {
            params.push(("end", end));
        }
        if let Some(ref lim) = limit_str {
            params.push(("limit", lim));
        }

        let _response = self.api_get("/v5/market/kline", &params).await?;

        Ok(Vec::new())
    }

    async fn subscribe_orderbook(
        &mut self,
        symbols: &[&str],
        _callback: OrderBookCallback,
    ) -> Result<()> {
        // Build subscription message
        let topics: Vec<String> = symbols
            .iter()
            .map(|s| format!("orderbook.50.{}", s))
            .collect();

        let _subscribe_msg = serde_json::json!({
            "op": "subscribe",
            "args": topics,
        });

        // In production: use tokio-tungstenite to connect and handle messages

        Ok(())
    }

    async fn subscribe_trades(&mut self, symbols: &[&str], _callback: TradeCallback) -> Result<()> {
        let topics: Vec<String> = symbols
            .iter()
            .map(|s| format!("publicTrade.{}", s))
            .collect();

        let _subscribe_msg = serde_json::json!({
            "op": "subscribe",
            "args": topics,
        });

        Ok(())
    }

    async fn subscribe_orders(&mut self, _callback: OrderCallback) -> Result<()> {
        // Connect to private WebSocket
        let _ws_url = self.ws_private_url();

        // Authenticate
        let credentials = self.config.credentials.as_ref().ok_or_else(|| {
            LiveTradingError::Authentication("API credentials required for private stream".into())
        })?;

        let expires = current_timestamp_ms() + 10000;
        let sign_str = format!("GET/realtime{}", expires);
        let signature = hmac_sha256(&credentials.api_secret, &sign_str);

        let _auth_msg = serde_json::json!({
            "op": "auth",
            "args": [credentials.api_key, expires, signature],
        });

        // Subscribe to order updates
        let _subscribe_msg = serde_json::json!({
            "op": "subscribe",
            "args": ["order"],
        });

        Ok(())
    }

    async fn subscribe_positions(&mut self, _callback: PositionCallback) -> Result<()> {
        // Subscribe to position updates via private WebSocket
        let _subscribe_msg = serde_json::json!({
            "op": "subscribe",
            "args": ["position"],
        });

        Ok(())
    }

    async fn unsubscribe_all(&mut self) -> Result<()> {
        // Close all WebSocket connections
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bybit_config() {
        let config = BybitConfig::default()
            .api_key("test_key")
            .api_secret("test_secret")
            .default_category(BybitCategory::Linear)
            .testnet(true);

        assert_eq!(config.default_category, BybitCategory::Linear);
        assert!(config.testnet);
        assert!(config.credentials.is_some());
    }

    #[test]
    fn test_interval_to_string() {
        assert_eq!(BybitConnector::interval_to_string(Interval::M1), "1");
        assert_eq!(BybitConnector::interval_to_string(Interval::H1), "60");
        assert_eq!(BybitConnector::interval_to_string(Interval::D1), "D");
    }

    #[test]
    fn test_base_urls() {
        let config = BybitConfig::default();
        let connector = BybitConnector::new(config);
        assert!(connector.rest_base_url().contains("api.bybit.com"));

        let testnet_config = BybitConfig::default().testnet(true);
        let testnet = BybitConnector::new(testnet_config);
        assert!(testnet.rest_base_url().contains("testnet"));
    }
}
