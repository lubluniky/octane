//! Live trading infrastructure for Octane.
//!
//! This module provides comprehensive live trading capabilities:
//!
//! - **Paper Trading** - Simulated trading for strategy testing without real money
//! - **Exchange Connectors** - Unified interface for connecting to exchanges (Binance, Bybit)
//! - **Execution Engine** - Smart order routing and execution algorithms (TWAP, VWAP, Iceberg)
//! - **Monitoring** - Real-time P&L, risk metrics, alerts, and health monitoring
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        RL Agent / Strategy                       │
//! └─────────────────────────────────────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                       Execution Engine                           │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │     TWAP     │  │     VWAP     │  │   Iceberg    │   ...    │
//! │  └──────────────┘  └──────────────┘  └──────────────┘          │
//! └─────────────────────────────────────────────────────────────────┘
//!                                  │
//!                    ┌─────────────┴─────────────┐
//!                    ▼                           ▼
//! ┌────────────────────────────┐  ┌────────────────────────────────┐
//! │     Paper Trading Engine   │  │     Exchange Connectors        │
//! │  ┌──────────────────────┐  │  │  ┌──────────┐  ┌──────────┐   │
//! │  │ Virtual Balance      │  │  │  │ Binance  │  │  Bybit   │   │
//! │  │ Slippage Simulation  │  │  │  └──────────┘  └──────────┘   │
//! │  │ Order Book Sim       │  │  │                                │
//! │  └──────────────────────┘  │  │         REST + WebSocket       │
//! └────────────────────────────┘  └────────────────────────────────┘
//!                                  │
//!                                  ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                          Monitor                                 │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │   P&L        │  │    Risk      │  │   Alerts     │   ...    │
//! │  └──────────────┘  └──────────────┘  └──────────────┘          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example: Paper Trading
//!
//! ```ignore
//! use octane_rs::live::{
//!     PaperTradingEngine, PaperTradingConfig, SlippageModel,
//!     Order, Side,
//! };
//!
//! // Create paper trading engine
//! let config = PaperTradingConfig::default()
//!     .initial_balance("USDT", 10000.0)
//!     .slippage_model(SlippageModel::Fixed { bps: 5.0 })
//!     .commission_rate(0.001);
//!
//! let mut engine = PaperTradingEngine::new(config);
//!
//! // Update market price
//! engine.update_price("BTCUSDT", 50000.0);
//!
//! // Place and execute a market order
//! let order = Order::market("BTCUSDT", Side::Buy, 0.1);
//! let filled = engine.execute_order(order, 50000.0)?;
//!
//! println!("Filled at: {}", filled.average_fill_price.unwrap());
//! ```
//!
//! # Example: Live Trading with Binance
//!
//! ```ignore
//! use octane_rs::live::{
//!     BinanceConnector, BinanceConfig, BinanceMarketType,
//!     ExchangeConnector, ExecutionEngine, ExecutionConfig,
//!     Monitor, MonitorConfig, AlertType, AlertConfig, AlertSeverity,
//!     Order, Side,
//! };
//!
//! // Configure exchange connector
//! let exchange_config = BinanceConfig::default()
//!     .api_key("your_api_key")
//!     .api_secret("your_api_secret")
//!     .market_type(BinanceMarketType::Futures)
//!     .testnet(true);  // Use testnet first!
//!
//! let mut connector = BinanceConnector::new(exchange_config);
//! connector.connect().await?;
//!
//! // Configure execution engine
//! let exec_config = ExecutionConfig::default()
//!     .max_slippage_bps(10.0)
//!     .smart_routing(true);
//!
//! let engine = ExecutionEngine::new(exec_config);
//!
//! // Configure monitoring
//! let monitor_config = MonitorConfig::default()
//!     .initial_capital(10000.0)
//!     .enable_alerts(true);
//!
//! let mut monitor = Monitor::new(monitor_config);
//!
//! // Add risk alerts
//! monitor.add_alert(AlertConfig::new(AlertType::MaxDrawdown { threshold: 0.05 })
//!     .severity(AlertSeverity::Critical)
//!     .halt_on_trigger(true)).await;
//!
//! // Start monitoring
//! monitor.start().await?;
//!
//! // Execute a TWAP order
//! use octane_rs::live::TWAPParams;
//! use std::time::Duration;
//!
//! let params = TWAPParams::new(Duration::from_secs(300), 10)
//!     .randomize(true);
//!
//! let result = engine.execute_twap(
//!     &connector,
//!     "BTCUSDT",
//!     Side::Buy,
//!     0.1,
//!     params
//! ).await?;
//!
//! println!("Filled {} at avg price {}", result.filled_quantity, result.average_price);
//! ```
//!
//! # Feature Flags
//!
//! The live trading module requires the `distributed` feature flag for async tokio support:
//!
//! ```toml
//! [dependencies]
//! octane-rs = { version = "0.1", features = ["distributed"] }
//! ```

pub mod error;
pub mod exchanges;
pub mod execution;
pub mod monitor;
pub mod paper;
pub mod types;

// Re-export main types for convenience
pub use error::{LiveTradingError, Result};
pub use exchanges::{
    ApiCredentials, BinanceConfig, BinanceConnector, BinanceMarketType, BybitAccountType,
    BybitCategory, BybitConfig, BybitConnector, ConnectionStatus, ExchangeConnector,
    ExchangeInfo, OrderBookCallback, OrderBookUpdate, OrderCallback, PositionCallback,
    RateLimit, RateLimiter, RateLimitType, SymbolInfo, TradeCallback, TradeUpdate,
};
pub use execution::{
    ExecutionAlgorithm, ExecutionConfig, ExecutionEngine, ExecutionQualityMetrics,
    ExecutionRequest, ExecutionResult, ExecutionStatus, IcebergParams, TWAPParams, Urgency,
    VWAPParams,
};
pub use monitor::{
    AlertConfig, AlertNotification, AlertSeverity, AlertType, ComponentHealth, HealthStatus,
    Monitor, MonitorConfig, MonitorEvent, PnLSnapshot, RiskMetrics, SystemHealth, TradingStats,
};
pub use paper::{
    FillModel, PaperTradingConfig, PaperTradingEngine, PaperTradingStats, SimulatedOrderBook,
    SlippageModel,
};
pub use types::{
    Balance, Candle, Interval, Order, OrderBook, OrderBookLevel, OrderStatus, OrderType, Position,
    Side, Ticker, TimeInForce, Trade, current_timestamp_ms, generate_client_order_id,
};

/// Prelude for live trading - import all commonly used types.
pub mod prelude {
    pub use super::error::{LiveTradingError, Result};
    pub use super::exchanges::{
        BinanceConfig, BinanceConnector, BinanceMarketType, BybitConfig, BybitConnector,
        ConnectionStatus, ExchangeConnector,
    };
    pub use super::execution::{
        ExecutionAlgorithm, ExecutionConfig, ExecutionEngine, ExecutionRequest, ExecutionResult,
        IcebergParams, TWAPParams, VWAPParams,
    };
    pub use super::monitor::{
        AlertConfig, AlertSeverity, AlertType, Monitor, MonitorConfig, MonitorEvent,
    };
    pub use super::paper::{PaperTradingConfig, PaperTradingEngine, SlippageModel};
    pub use super::types::{
        Balance, Candle, Interval, Order, OrderBook, OrderStatus, OrderType, Position, Side,
        Ticker, TimeInForce, Trade,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prelude_imports() {
        // Verify that all types are accessible through prelude
        use prelude::*;

        let _side = Side::Buy;
        let _status = OrderStatus::Pending;
        let _algo = ExecutionAlgorithm::TWAP;
    }

    #[test]
    fn test_order_creation() {
        let order = Order::market("BTCUSDT", Side::Buy, 0.1);
        assert_eq!(order.symbol, "BTCUSDT");
        assert_eq!(order.side, Side::Buy);
        assert_eq!(order.quantity, 0.1);
        assert_eq!(order.order_type, OrderType::Market);
    }

    #[test]
    fn test_paper_trading() {
        let config = PaperTradingConfig::default()
            .initial_balance("USDT", 10000.0)
            .slippage_model(SlippageModel::None);

        let mut engine = PaperTradingEngine::new(config);
        engine.update_price("BTCUSDT", 50000.0);

        let order = Order::market("BTCUSDT", Side::Buy, 0.1);
        let result = engine.execute_order(order, 50000.0);

        assert!(result.is_ok());
        let filled = result.unwrap();
        assert_eq!(filled.status, OrderStatus::Filled);
    }
}
