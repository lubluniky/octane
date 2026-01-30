//! Core module: Device management, error handling, tensor backend abstractions.

mod device;
mod error;
pub mod precision;
mod tensor;

pub use device::Device;
pub use error::{OctaneError, Result};
pub use precision::{AutocastContext, GradScaler, MixedPrecisionConfig, Precision};
pub use tensor::TensorBackend;
