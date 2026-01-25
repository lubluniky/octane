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
pub mod logging;
pub mod networks;
pub mod tui;

// Re-exports for ergonomic API
pub use crate::algorithms::{
    A2CAgent, A2CConfig, Agent, DDPGAgent, DDPGConfig, DQNAgent, DQNConfig, NoiseType, PPOAgent,
    PPOConfig, SACAgent, SACConfig, TD3Agent, TD3Config, TrainMetrics,
};
pub use crate::buffer::{
    ReplayBatch, ReplayBuffer, ReplayBufferConfig, RolloutBatch, RolloutBuffer, RolloutBufferConfig,
};
pub use crate::core::{Device, Result, RocketError, TensorBackend};
pub use crate::distributions::{Categorical, DiagGaussian, Distribution, SquashedGaussian};
pub use crate::envs::{
    ActionType, Environment, MarketData, ObsType, Space, TradingEnv, TradingEnvConfig, VecEnv,
};
pub use crate::logging::{
    list_training_runs, TrainingLogEntry, TrainingLogReader, TrainingLogger, TrainingRunInfo,
};
pub use crate::networks::{ActorCritic, GRU, LSTM, MLP};

/// Prelude module for convenient imports.
///
/// Use with: `use rocket_rs::prelude::*;`
pub mod prelude {
    // Core types
    pub use crate::core::{Device, Result, RocketError};

    // Algorithms
    pub use crate::algorithms::{
        A2CAgent, A2CConfig, DDPGAgent, DDPGConfig, DQNAgent, DQNConfig, PPOAgent, PPOConfig,
        SACAgent, SACConfig, TD3Agent, TD3Config, TrainMetrics,
    };

    // Buffers
    pub use crate::buffer::{ReplayBuffer, ReplayBufferConfig, RolloutBuffer};

    // Environments
    pub use crate::envs::{Environment, Space, TradingEnv, TradingEnvConfig, VecEnv};

    // Networks
    pub use crate::networks::{ActorCritic, MLP};

    // Distributions
    pub use crate::distributions::{Categorical, DiagGaussian, Distribution};

    // Logging
    pub use crate::logging::{TrainingLogEntry, TrainingLogReader, TrainingLogger, TrainingRunInfo};
}
