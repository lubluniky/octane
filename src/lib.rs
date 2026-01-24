//! # RocketRL
//!
//! High-performance Reinforcement Learning library for Rust.
//! Optimized for Apple Silicon (Metal) and NVIDIA GPUs (CUDA).
//!
//! ## Features
//! - Vectorized environments for massive parallelization
//! - PPO and A2C algorithms with GAE
//! - LSTM/GRU support for time-series (trading)
//! - Zero-copy tensor operations via Candle

#![forbid(unsafe_code)]
#![warn(missing_docs, rust_2018_idioms)]
#![allow(dead_code)]  // Allow unused code during development

pub mod core;
pub mod envs;
pub mod algorithms;
pub mod distributions;
pub mod networks;
pub mod buffer;
pub mod tui;

// Re-exports for ergonomic API
pub use crate::core::{Device, TensorBackend, RocketError, Result};
pub use crate::envs::{Environment, VecEnv, Space, ObsType, ActionType, TradingEnv, TradingEnvConfig, MarketData};
pub use crate::algorithms::{Agent, PPOConfig, A2CConfig, TrainMetrics, PPOAgent, A2CAgent};
pub use crate::distributions::{Distribution, Categorical, DiagGaussian, SquashedGaussian};
pub use crate::networks::{ActorCritic, MLP, LSTM, GRU};
pub use crate::buffer::{RolloutBuffer, RolloutBatch, RolloutBufferConfig};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::core::{Device, TensorBackend, Result};
    pub use crate::envs::{Environment, VecEnv, Space};
    pub use crate::algorithms::{Agent, PPOConfig, A2CConfig, TrainMetrics};
    pub use crate::distributions::Distribution;
    pub use crate::networks::ActorCritic;
    pub use crate::buffer::{RolloutBuffer, RolloutBatch};
}
