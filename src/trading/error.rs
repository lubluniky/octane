//! Trading-specific error types.

use thiserror::Error;

/// Errors specific to trading environments.
#[derive(Error, Debug)]
pub enum TradingError {
    /// Invalid order parameters.
    #[error("Invalid order: {0}")]
    InvalidOrder(String),

    /// Insufficient balance for trade.
    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance {
        /// Required amount for the trade.
        required: f32,
        /// Available balance.
        available: f32,
    },

    /// Position limit exceeded.
    #[error("Position limit exceeded: {0}")]
    PositionLimitExceeded(String),

    /// Invalid market data.
    #[error("Invalid market data: {0}")]
    InvalidMarketData(String),

    /// Order book error.
    #[error("Order book error: {0}")]
    OrderBookError(String),

    /// Regime detection error.
    #[error("Regime detection error: {0}")]
    RegimeError(String),

    /// Timeframe synchronization error.
    #[error("Timeframe sync error: {0}")]
    TimeframeSyncError(String),

    /// Asset not found in portfolio.
    #[error("Asset not found: {0}")]
    AssetNotFound(String),

    /// Invalid correlation matrix.
    #[error("Invalid correlation matrix: {0}")]
    InvalidCorrelationMatrix(String),
}
