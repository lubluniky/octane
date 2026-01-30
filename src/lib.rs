//! # Octane
//!
//! High-performance Reinforcement Learning library for Rust.
//! Optimized for Apple Silicon (Metal) and NVIDIA GPUs (CUDA).
//!
//! ## Features
//! - Vectorized environments for massive parallelization
//! - PPO and A2C algorithms with GAE
//! - LSTM/GRU support for time-series (trading)
//! - Zero-copy tensor operations via Candle
//! - Distributed training support
//! - Mixed precision training (FP16/BF16)
//! - Checkpointing and training resumption
//! - Hyperparameter tuning

#![warn(missing_docs, rust_2018_idioms)]
#![allow(dead_code)] // Allow unused code during development

pub mod algorithms;
pub mod buffer;
pub mod checkpoint;
pub mod core;
pub mod distributed;
pub mod distributions;
pub mod envs;
pub mod logging;
pub mod networks;
pub mod profiling;
pub mod simd;
pub mod tui;
pub mod tuning;

// Re-exports for ergonomic API
pub use crate::algorithms::{
    A2CAgent, A2CConfig, Agent, CQLAgent, CQLConfig, DDPGAgent, DDPGConfig, DQNAgent, DQNConfig,
    IQNAgent, IQNConfig, NoiseType, PPGAgent, PPGConfig, PPOAgent, PPOConfig, REDQAgent,
    REDQConfig, RiskMeasure, SACAgent, SACConfig, TD3Agent, TD3Config, TrainMetrics,
};
pub use crate::buffer::{
    ReplayBatch, ReplayBuffer, ReplayBufferConfig, RolloutBatch, RolloutBuffer, RolloutBufferConfig,
};
pub use crate::checkpoint::{
    BestMetric, Checkpoint, CheckpointInfo, CheckpointManager, OptimizerState, TrainingResumer,
};
pub use crate::core::{
    AutocastContext, Device, GradScaler, MixedPrecisionConfig, OctaneError, Precision, Result,
    TensorBackend,
};
pub use crate::distributed::{
    DistributedBackend, DistributedConfig, DistributedCoordinator, DistributedStats,
    GradientAggregator, ReduceOp, SyncMode, WorkerMessage, WorkerPool, WorkerState,
};
pub use crate::distributions::{Categorical, DiagGaussian, Distribution, SquashedGaussian};
pub use crate::envs::{
    ActionType, Environment, MarketData, ObsType, Space, TradingEnv, TradingEnvConfig, VecEnv,
};
pub use crate::logging::{
    list_training_runs, CompositeLogger, ConfigValue, HistogramData, ImageData, MetricAggregator,
    MetricBuffer, MetricLogger, NullLogger, ResumeMode, TensorBoardWriter, TrainingLogEntry,
    TrainingLogReader, TrainingLogger, TrainingRunInfo, VideoData, WandbConfig, WandbLogger,
};
pub use crate::networks::{
    ActorCritic, AttentionActorCritic, DecisionTransformer, LayerNorm, RMSNorm, TransformerEncoder,
    GRU, LSTM, MLP,
};
pub use crate::profiling::{
    global_profiler, timed, PerfCounters, ProfileReport, ProfileScope, Profiler, ScopeStats,
    SimdStats,
};
pub use crate::tuning::{
    CategoricalParam, FloatParam, GridConfig, GridSearch, HyperparameterSpace, IntParam,
    OptimizationDirection, ParamValue, RandomSearch, Sampler, Study, Trial, TrialState,
};

/// Prelude module for convenient imports.
///
/// Use with: `use octane_rs::prelude::*;`
pub mod prelude {
    // Core types
    pub use crate::core::{Device, GradScaler, OctaneError, Precision, Result};

    // Algorithms
    pub use crate::algorithms::{
        A2CAgent, A2CConfig, CQLAgent, CQLConfig, DDPGAgent, DDPGConfig, DQNAgent, DQNConfig,
        IQNAgent, IQNConfig, PPGAgent, PPGConfig, PPOAgent, PPOConfig, REDQAgent, REDQConfig,
        RiskMeasure, SACAgent, SACConfig, TD3Agent, TD3Config, TrainMetrics,
    };

    // Buffers
    pub use crate::buffer::{ReplayBuffer, ReplayBufferConfig, RolloutBuffer};

    // Checkpointing
    pub use crate::checkpoint::{Checkpoint, CheckpointManager, TrainingResumer};

    // Distributed training
    pub use crate::distributed::{DistributedConfig, DistributedCoordinator, GradientAggregator};

    // Environments
    pub use crate::envs::{Environment, Space, TradingEnv, TradingEnvConfig, VecEnv};

    // Networks
    pub use crate::networks::{
        ActorCritic, AttentionActorCritic, DecisionTransformer, LayerNorm, MLP, RMSNorm,
        TransformerEncoder,
    };

    // Distributions
    pub use crate::distributions::{Categorical, DiagGaussian, Distribution};

    // Logging
    pub use crate::logging::{
        CompositeLogger, MetricLogger, TensorBoardWriter, TrainingLogEntry, TrainingLogReader,
        TrainingLogger, TrainingRunInfo, WandbConfig, WandbLogger,
    };

    // Profiling
    pub use crate::profiling::{global_profiler, timed, ProfileReport, ProfileScope, Profiler};

    // Hyperparameter tuning
    pub use crate::tuning::{HyperparameterSpace, RandomSearch, Study, Trial};
}
