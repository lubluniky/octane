//! Error types for Octane.

use thiserror::Error;

/// Result type alias for Octane operations.
pub type Result<T> = std::result::Result<T, OctaneError>;

/// Errors that can occur in Octane operations.
#[derive(Error, Debug)]
pub enum OctaneError {
    /// Tensor operation failed.
    #[error("Tensor error: {0}")]
    Tensor(#[from] candle_core::Error),

    /// Shape mismatch in tensor operations.
    #[error("Shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Expected shape dimensions
        expected: Vec<usize>,
        /// Actual shape dimensions received
        got: Vec<usize>,
    },

    /// Invalid configuration parameter.
    #[error("Invalid config: {0}")]
    InvalidConfig(String),

    /// Environment error.
    #[error("Environment error: {0}")]
    Environment(String),

    /// Buffer overflow or underflow.
    #[error("Buffer error: {0}")]
    Buffer(String),

    /// Serialization/deserialization error.
    #[error("Serialization error: {0}")]
    Serialization(String),

    /// Device not available.
    #[error("Device not available: {0}")]
    DeviceUnavailable(String),

    /// Numerical instability (NaN, Inf).
    #[error("Numerical instability: {0}")]
    NumericalInstability(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

impl From<serde_json::Error> for OctaneError {
    fn from(err: serde_json::Error) -> Self {
        OctaneError::Serialization(err.to_string())
    }
}

// bincode 2/3 split the single `bincode::Error` into separate encode/decode
// error types, so convert each into our serialization variant.
impl From<bincode::error::EncodeError> for OctaneError {
    fn from(err: bincode::error::EncodeError) -> Self {
        OctaneError::Serialization(err.to_string())
    }
}

impl From<bincode::error::DecodeError> for OctaneError {
    fn from(err: bincode::error::DecodeError) -> Self {
        OctaneError::Serialization(err.to_string())
    }
}
