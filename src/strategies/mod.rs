//! Advanced RL Strategies for Trading.
//!
//! This module provides sophisticated reinforcement learning strategies
//! specifically designed for algorithmic trading applications:
//!
//! ## Ensemble Agents
//!
//! Multiple agents voting on actions for robust decision-making.
//!
//! - [`EnsembleAgent`] - Combines multiple RL agents
//! - [`VotingStrategy`] - Majority, weighted average, stacking, boosting
//! - [`DiversityMetrics`] - Measures agent disagreement
//!
//! ## Hierarchical RL
//!
//! Two-level hierarchy for temporal abstraction in trading.
//!
//! - [`HierarchicalAgent`] - High-level timing, low-level execution
//! - [`TradingOption`] - Available trading strategies/skills
//! - [`Goal`] - Goal specification for goal-conditioned policies
//!
//! ## Meta-Learning
//!
//! Fast adaptation to changing market regimes.
//!
//! - [`AdaptiveAgent`] - Quick adaptation capabilities
//! - [`MarketRegime`] - Market regime types
//! - [`AdaptationContext`] - Recent experience for context-based adaptation
//!
//! ## Imitation Learning
//!
//! Learning from expert trading demonstrations.
//!
//! - [`ImitationAgent`] - Behavioral cloning and DAgger
//! - [`DemoReplayBuffer`] - Storage for expert demonstrations
//! - [`ExpertPolicy`] - Trait for expert policies
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::strategies::{
//!     EnsembleAgent, EnsembleConfig, VotingStrategy,
//!     HierarchicalAgent, HierarchicalConfig,
//!     AdaptiveAgent, MetaLearningConfig,
//!     ImitationAgent, ImitationConfig,
//! };
//!
//! // Create an ensemble agent
//! let ensemble_config = EnsembleConfig::default()
//!     .num_agents(5)
//!     .voting_strategy(VotingStrategy::WeightedAverage);
//! let ensemble = EnsembleAgent::new(ensemble_config, env.clone(), device)?;
//!
//! // Create a hierarchical agent
//! let hier_config = HierarchicalConfig::default()
//!     .high_level_frequency(10)
//!     .goal_conditioned(true);
//! let hierarchical = HierarchicalAgent::new(hier_config, env.clone(), device)?;
//!
//! // Create an adaptive agent
//! let meta_config = MetaLearningConfig::default()
//!     .context_window(50)
//!     .adaptation_steps(3);
//! let adaptive = AdaptiveAgent::new(meta_config, env.clone(), device)?;
//!
//! // Create an imitation agent
//! let imitation_config = ImitationConfig::default()
//!     .method(ImitationMethod::DAgger)
//!     .pretrain_epochs(100);
//! let imitation = ImitationAgent::new(imitation_config, env.clone(), device)?;
//! ```

pub mod ensemble;
pub mod hierarchical;
pub mod imitation;
pub mod meta;

// Re-exports for convenience
pub use ensemble::{
    AgentPerformance, DiversityMetrics, EnsembleAgent, EnsembleConfig, VotingStrategy,
    WeightAdaptation,
};

pub use hierarchical::{
    Goal, HierarchicalAgent, HierarchicalConfig, HierarchicalReplayBuffer, HierarchicalTransition,
    TradingOption,
};

pub use meta::{
    AdaptationContext, AdaptationStrategy, AdaptiveAgent, ContextStatistics, MarketRegime,
    MetaLearningConfig, Task,
};

pub use imitation::{
    DemoBatch, DemoMetadata, DemoReplayBuffer, Demonstration, ExpertPolicy, ImitationAgent,
    ImitationConfig, ImitationLoss, ImitationMethod, RuleBasedExpert, TrainingPhase,
};
