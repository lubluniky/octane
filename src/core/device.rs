//! Device abstraction for CPU/Metal/CUDA backends.

use candle_core::Device as CandleDevice;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Compute device for tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Device {
    /// CPU backend (always available)
    Cpu,
    /// Apple Metal backend (M1/M2/M3/M4)
    #[cfg(feature = "metal")]
    Metal,
    /// NVIDIA CUDA backend
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

impl Device {
    /// Create a Metal device for Apple Silicon.
    #[cfg(feature = "metal")]
    pub fn m4_metal() -> Self {
        Device::Metal
    }

    /// Create a CUDA device with the specified GPU ordinal.
    #[cfg(feature = "cuda")]
    pub fn cuda(ordinal: usize) -> Self {
        Device::Cuda(ordinal)
    }

    /// Create a CPU device.
    pub fn cpu() -> Self {
        Device::Cpu
    }

    /// Convert to Candle's device type.
    pub fn to_candle(&self) -> candle_core::Result<CandleDevice> {
        match self {
            Device::Cpu => Ok(CandleDevice::Cpu),
            #[cfg(feature = "metal")]
            Device::Metal => CandleDevice::new_metal(0),
            #[cfg(feature = "cuda")]
            Device::Cuda(ordinal) => CandleDevice::new_cuda(*ordinal),
        }
    }

    /// Check if this device is GPU-accelerated.
    pub fn is_gpu(&self) -> bool {
        match self {
            Device::Cpu => false,
            #[cfg(feature = "metal")]
            Device::Metal => true,
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => true,
        }
    }
}

impl Default for Device {
    fn default() -> Self {
        Device::Cpu
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Cpu => write!(f, "CPU"),
            #[cfg(feature = "metal")]
            Device::Metal => write!(f, "Metal (Apple Silicon)"),
            #[cfg(feature = "cuda")]
            Device::Cuda(ord) => write!(f, "CUDA:{}", ord),
        }
    }
}
