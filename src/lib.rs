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
#![allow(dead_code)] // Allow unused code during development

pub mod algorithms;
pub mod buffer;
pub mod core;
pub mod distributions;
pub mod envs;
pub mod networks;
pub mod tui;

// Re-exports for ergonomic API
pub use crate::algorithms::{A2CAgent, A2CConfig, Agent, PPOAgent, PPOConfig, TrainMetrics};
pub use crate::buffer::{RolloutBatch, RolloutBuffer, RolloutBufferConfig};
pub use crate::core::{Device, Result, RocketError, TensorBackend};
pub use crate::distributions::{Categorical, DiagGaussian, Distribution, SquashedGaussian};
pub use crate::envs::{
    ActionType, Environment, MarketData, ObsType, Space, TradingEnv, TradingEnvConfig, VecEnv,
};
pub use crate::networks::{ActorCritic, GRU, LSTM, MLP};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::algorithms::{A2CConfig, Agent, PPOConfig, TrainMetrics};
    pub use crate::buffer::{RolloutBatch, RolloutBuffer};
    pub use crate::core::{Device, Result, TensorBackend};
    pub use crate::distributions::Distribution;
    pub use crate::envs::{Environment, Space, VecEnv};
    pub use crate::networks::ActorCritic;
}
