//! Core module: Device management, error handling, tensor backend abstractions.

mod device;
mod error;
mod tensor;

pub use device::Device;
pub use error::{RocketError, Result};
pub use tensor::TensorBackend;
