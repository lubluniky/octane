//! Error types for live trading infrastructure.

use thiserror::Error;

/// Result type alias for live trading operations.
pub type Result<T> = std::result::Result<T, LiveTradingError>;

/// Errors that can occur in live trading operations.
#[derive(Error, Debug)]
pub enum LiveTradingError {
    /// Connection to exchange failed.
    #[error("Connection error: {0}")]
    Connection(String),

    /// Authentication failed.
    #[error("Authentication error: {0}")]
    Authentication(String),

    /// Order placement failed.
    #[error("Order error: {0}")]
    Order(String),

    /// Order not found.
    #[error("Order not found: {0}")]
    OrderNotFound(String),

    /// Insufficient balance for trade.
    #[error("Insufficient balance: required {required}, available {available}")]
    InsufficientBalance {
        /// Required amount for the trade.
        required: f64,
        /// Available balance.
        available: f64,
    },

    /// Position limit exceeded.
    #[error("Position limit exceeded: {0}")]
    PositionLimitExceeded(String),

    /// Rate limit exceeded.
    #[error("Rate limit exceeded: retry after {retry_after_ms}ms")]
    RateLimitExceeded {
        /// Milliseconds to wait before retry.
        retry_after_ms: u64,
    },

    /// WebSocket error.
    #[error("WebSocket error: {0}")]
    WebSocket(String),

    /// API error from exchange.
    #[error("API error: {code} - {message}")]
    ApiError {
        /// Error code from exchange.
        code: i32,
        /// Error message from exchange.
        message: String,
    },

    /// Invalid configuration.
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Paper trading specific error.
    #[error("Paper trading error: {0}")]
    PaperTrading(String),

    /// Execution error.
    #[error("Execution error: {0}")]
    Execution(String),

    /// Monitoring error.
    #[error("Monitoring error: {0}")]
    Monitoring(String),

    /// Risk limit breached.
    #[error("Risk limit breached: {0}")]
    RiskLimitBreached(String),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Timeout error.
    #[error("Timeout: {0}")]
    Timeout(String),

    /// Market closed or unavailable.
    #[error("Market unavailable: {0}")]
    MarketUnavailable(String),

    /// Symbol not found.
    #[error("Symbol not found: {0}")]
    SymbolNotFound(String),
}

impl From<serde_json::Error> for LiveTradingError {
    fn from(err: serde_json::Error) -> Self {
        LiveTradingError::Serialization(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = LiveTradingError::InsufficientBalance {
            required: 1000.0,
            available: 500.0,
        };
        assert!(err.to_string().contains("Insufficient balance"));
    }
}
