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
//! - Live trading infrastructure (paper trading, exchange connectors, execution algorithms)

#![warn(missing_docs, rust_2018_idioms)]
#![allow(dead_code)] // Allow unused code during development
#![allow(
    clippy::if_same_then_else,
    clippy::needless_range_loop,
    clippy::only_used_in_recursion,
    clippy::same_item_push,
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::uninlined_format_args,
    clippy::upper_case_acronyms
)]

pub mod algorithms;
pub mod backtesting;
pub mod buffer;
pub mod checkpoint;
pub mod core;
pub mod distributed;
pub mod distributions;
pub mod envs;
#[cfg(feature = "distributed")]
pub mod live;
pub mod logging;
pub mod metrics;
pub mod networks;
pub mod profiling;
pub mod risk;
pub mod simd;
pub mod strategies;
pub mod trading;
pub mod tui;
pub mod tuning;

// Re-exports for ergonomic API
pub use crate::algorithms::{
    A2CAgent, A2CConfig, Agent, CQLAgent, CQLConfig, DDPGAgent, DDPGConfig, DQNAgent, DQNConfig,
    IQNAgent, IQNConfig, NoiseType, PPGAgent, PPGConfig, PPOAgent, PPOConfig, REDQAgent,
    REDQConfig, RiskMeasure, SACAgent, SACConfig, TD3Agent, TD3Config, TrainMetrics,
};
pub use crate::backtesting::{
    BootstrapResult, CVConfig, CVFold, CVMethod, CVMetrics, CVResult, CVScoring, CVSummary,
    ConfidenceLevel, CrossValidator, MetricSummary, MonteCarloConfig, MonteCarloResult,
    MonteCarloSimulator, PerturbationResult, PriceModel, PricePathResult, SplitMetrics,
    SplitPerformance, StressScenario, StressTestResult, WalkForwardConfig, WalkForwardObjective,
    WalkForwardOptimizer, WalkForwardResult, WalkForwardSplit, WalkForwardSummary,
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
pub use crate::metrics::{
    AttributionAnalyzer, AttributionConfig, AttributionEntry, AttributionReport, Direction,
    JournalConfig, JournalStats, MarketRegime, MetricsCalculator, MetricsConfig, TimeOfDay,
    TimePeriod, TradeDirection, TradeEntry, TradeJournal, TradingMetrics,
};
pub use crate::networks::{
    ActorCritic, AttentionActorCritic, DecisionTransformer, LayerNorm, RMSNorm, TransformerEncoder,
    GRU, LSTM, MLP,
};
pub use crate::profiling::{
    global_profiler, timed, PerfCounters, ProfileReport, ProfileScope, Profiler, ScopeStats,
    SimdStats,
};
pub use crate::risk::{
    ActionMask, BoxConstraint, CalmarRewardShaper, Constraint, ConstraintConfig, ConstraintManager,
    ConstraintResult, ConstraintType, DrawdownConfig, DrawdownController, DrawdownEvent,
    DrawdownState, EqualityConstraint, InequalityConstraint, KellyCalculator, KellyResult,
    LagrangianRelaxation, PositionSizer, PositionSizingConfig, ProjectionResult, RewardShaper,
    RewardShaperConfig, RiskParityShaper, RiskScaling, RollingStats, SharpeRewardShaper,
    SizingMethod, SortinoRewardShaper, UnderwaterCurve, VolatilitySizer,
};
pub use crate::strategies::{
    AdaptationContext, AdaptationStrategy, AdaptiveAgent, AgentPerformance, DemoBatch,
    DemoMetadata, DemoReplayBuffer, Demonstration, DiversityMetrics, EnsembleAgent, EnsembleConfig,
    ExpertPolicy, Goal, HierarchicalAgent, HierarchicalConfig, HierarchicalReplayBuffer,
    HierarchicalTransition, ImitationAgent, ImitationConfig, ImitationLoss, ImitationMethod,
    MarketRegime as StrategyMarketRegime, MetaLearningConfig, RuleBasedExpert, Task, TradingOption,
    TrainingPhase, VotingStrategy, WeightAdaptation,
};
pub use crate::trading::{
    AdvancedTradingConfig, AdvancedTradingEnv, CommissionModel, GarchParams, HmmParams,
    MarketRegime as TradingMarketRegime, MultiAssetConfig, MultiAssetEnv, MultiTimeframeConfig,
    MultiTimeframeEnv, Order, OrderBook, OrderBookLevel, OrderSide, OrderStatus, OrderType,
    PortfolioAction, PortfolioMetrics, PortfolioState, PositionType, RegimeCallback, RegimeConfig,
    RegimeDetector, RegimeObservation, RegimeTransition, SlippageModel, Timeframe, TimeframeData,
    TimeframeSynchronizer, TradingError,
};
pub use crate::tuning::{
    CategoricalParam, FloatParam, GridConfig, GridSearch, HyperparameterSpace, IntParam,
    OptimizationDirection, ParamValue, RandomSearch, Sampler, Study, Trial, TrialState,
};

// SIMD-optimized operations
pub use crate::simd::{
    best_simd_width, compute_gae_simd, compute_gae_simd_inplace, is_neon_available,
    normalize_advantages_simd, simd_features_info,
};

// Live trading infrastructure (requires distributed feature)
#[cfg(feature = "distributed")]
pub use crate::live::{
    // Monitoring
    AlertConfig,
    AlertNotification,
    AlertSeverity,
    AlertType,
    // Exchange connectors
    ApiCredentials,
    // Types
    Balance,
    BinanceConfig,
    BinanceConnector,
    BinanceMarketType,
    BybitAccountType,
    BybitCategory,
    BybitConfig,
    BybitConnector,
    Candle,
    ComponentHealth,
    ConnectionStatus,
    ExchangeConnector,
    ExchangeInfo,
    // Execution
    ExecutionAlgorithm,
    ExecutionConfig,
    ExecutionEngine,
    ExecutionQualityMetrics,
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    // Paper trading
    FillModel,
    HealthStatus,
    IcebergParams,
    Interval,
    // Error types
    LiveTradingError,
    Monitor,
    MonitorConfig,
    MonitorEvent,
    Order as LiveOrder,
    OrderBook as LiveOrderBook,
    OrderBookLevel as LiveOrderBookLevel,
    OrderStatus as LiveOrderStatus,
    OrderType as LiveOrderType,
    PaperTradingConfig,
    PaperTradingEngine,
    PaperTradingStats,
    PnLSnapshot,
    Position,
    RateLimit,
    RateLimitType,
    RateLimiter,
    RiskMetrics,
    Side,
    SimulatedOrderBook,
    SlippageModel as LiveSlippageModel,
    SymbolInfo,
    SystemHealth,
    TWAPParams,
    Ticker,
    TimeInForce,
    Trade,
    TradingStats,
    Urgency,
    VWAPParams,
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
        ActorCritic, AttentionActorCritic, DecisionTransformer, LayerNorm, RMSNorm,
        TransformerEncoder, MLP,
    };

    // Distributions
    pub use crate::distributions::{Categorical, DiagGaussian, Distribution};

    // Logging
    pub use crate::logging::{
        CompositeLogger, MetricLogger, TensorBoardWriter, TrainingLogEntry, TrainingLogReader,
        TrainingLogger, TrainingRunInfo, WandbConfig, WandbLogger,
    };

    // Metrics
    pub use crate::metrics::{
        AttributionAnalyzer, AttributionConfig, JournalConfig, MetricsCalculator, MetricsConfig,
        TradeJournal, TradingMetrics,
    };

    // Profiling
    pub use crate::profiling::{global_profiler, timed, ProfileReport, ProfileScope, Profiler};

    // Hyperparameter tuning
    pub use crate::tuning::{HyperparameterSpace, RandomSearch, Study, Trial};

    // Backtesting and validation
    pub use crate::backtesting::{
        CVConfig, CVMethod, CVResult, CVScoring, ConfidenceLevel, CrossValidator, MonteCarloConfig,
        MonteCarloSimulator, WalkForwardConfig, WalkForwardObjective, WalkForwardOptimizer,
        WalkForwardResult,
    };

    // Risk management
    pub use crate::risk::{
        CalmarRewardShaper, ConstraintConfig, ConstraintManager, DrawdownConfig,
        DrawdownController, KellyCalculator, PositionSizer, PositionSizingConfig, RewardShaper,
        RewardShaperConfig, RiskScaling, SharpeRewardShaper, SizingMethod, SortinoRewardShaper,
    };

    // Advanced trading environments
    pub use crate::trading::{
        AdvancedTradingConfig, AdvancedTradingEnv, CommissionModel, MultiAssetConfig,
        MultiAssetEnv, MultiTimeframeConfig, MultiTimeframeEnv, OrderSide, OrderType,
        PortfolioMetrics, PositionType, RegimeConfig, RegimeDetector, RegimeObservation,
        SlippageModel, Timeframe,
    };

    // Advanced RL strategies
    pub use crate::strategies::{
        AdaptiveAgent, DemoReplayBuffer, Demonstration, DiversityMetrics, EnsembleAgent,
        EnsembleConfig, HierarchicalAgent, HierarchicalConfig, ImitationAgent, ImitationConfig,
        ImitationMethod, MetaLearningConfig, TradingOption, VotingStrategy,
    };

    // Live trading (requires distributed feature)
    #[cfg(feature = "distributed")]
    pub use crate::live::{
        BinanceConfig, BinanceConnector, BybitConfig, BybitConnector, ExecutionAlgorithm,
        ExecutionConfig, ExecutionEngine, ExecutionRequest, Monitor, MonitorConfig,
        PaperTradingConfig, PaperTradingEngine, Side, TWAPParams, VWAPParams,
    };
}
