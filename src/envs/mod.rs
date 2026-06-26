//! Environment module: Gym-like trait interfaces with VecEnv support.
//!
//! This module provides:
//! - Core `Environment` trait for single-agent RL
//! - `VecEnv` for parallel environment execution
//! - Environment wrappers for preprocessing and monitoring
//! - Multi-agent environment support (CTDE)
//! - Python Gym compatibility (requires "gym" feature)

mod array_env;
mod classic_control;
mod multiagent;
mod space;
mod trading;
mod traits;
mod vecenv;
mod wrappers;

#[cfg(feature = "gym")]
mod gym;

// Core traits and types
pub use array_env::{ArrayEnv, ArrayReward};
pub use classic_control::{CartPole, Pendulum};
pub use space::{BoxSpace, DiscreteSpace, Space};
pub use trading::{MarketData, TradingEnv, TradingEnvConfig};
pub use traits::{ActionType, Environment, ObsType, StepInfo, StepResult};
pub use vecenv::{VecEnv, VecEnvConfig, VecStepResult};

// Environment wrappers
pub use wrappers::{
    ClipAction, EpisodeStats, FrameStack, NormalizeObservation, NormalizeReward,
    RecordEpisodeStatistics, RunningMeanStd, TimeLimit, WrappedEnv,
};

// Multi-agent support
pub use multiagent::utils as multiagent_utils;
pub use multiagent::{
    AgentId, CentralizedCritic, CommunicationConfig, JointAction, MultiAgentDone, MultiAgentEnv,
    MultiAgentInfo, MultiAgentObs, MultiAgentReward, MultiAgentSpace, MultiAgentStepResult,
    ParameterSharingConfig, SingleAgentAdapter,
};

// Python Gym compatibility (requires "gym" feature)
#[cfg(feature = "gym")]
pub use gym::{make_vec_gym_envs, GymEnv, GymEnvConfig, GymSpace};
