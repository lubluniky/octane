//! Risk management module for algorithmic trading RL.
//!
//! This module provides comprehensive risk management tools for trading-focused
//! reinforcement learning, including:
//!
//! - **Constraints**: Safe RL constraints on position size, drawdown, and exposure
//! - **Rewards**: Risk-adjusted reward shaping (Sharpe, Sortino, Calmar ratios)
//! - **Position Sizing**: Kelly criterion and related position sizing strategies
//! - **Drawdown Control**: Real-time drawdown tracking and dynamic risk scaling
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::risk::{
//!     DrawdownController, DrawdownConfig,
//!     PositionSizer, PositionSizingConfig, SizingMethod,
//!     RewardShaper, SharpeRewardShaper, RewardShaperConfig,
//!     ConstraintManager, ConstraintConfig, Constraint,
//! };
//!
//! // Create a drawdown controller
//! let controller = DrawdownController::new(DrawdownConfig::default()
//!     .max_drawdown(0.20)
//!     .recovery_threshold(0.10));
//!
//! // Create a position sizer using Kelly criterion
//! let sizer = PositionSizer::new(PositionSizingConfig::default()
//!     .method(SizingMethod::HalfKelly)
//!     .max_position(0.25));
//!
//! // Create a Sharpe-based reward shaper
//! let shaper = SharpeRewardShaper::new(RewardShaperConfig::default()
//!     .window_size(252)
//!     .risk_free_rate(0.02));
//! ```

pub mod constraints;
pub mod drawdown;
pub mod position_sizing;
pub mod rewards;

// Re-exports for ergonomic API
pub use constraints::{
    ActionMask, BoxConstraint, Constraint, ConstraintConfig, ConstraintManager, ConstraintResult,
    ConstraintType, EqualityConstraint, InequalityConstraint, LagrangianRelaxation,
    ProjectionResult,
};
pub use drawdown::{
    DrawdownConfig, DrawdownController, DrawdownEvent, DrawdownState, RiskScaling, UnderwaterCurve,
};
pub use position_sizing::{
    KellyCalculator, KellyResult, PositionSizer, PositionSizingConfig, SizingMethod,
    VolatilitySizer,
};
pub use rewards::{
    CalmarRewardShaper, RewardShaper, RewardShaperConfig, RiskParityShaper, RollingStats,
    SharpeRewardShaper, SortinoRewardShaper,
};
