//! Environment module: Gym-like trait interfaces with VecEnv support.

mod space;
mod traits;
mod vecenv;
mod trading;

pub use space::{Space, BoxSpace, DiscreteSpace};
pub use traits::{Environment, ObsType, ActionType, StepResult, StepInfo};
pub use vecenv::{VecEnv, VecEnvConfig};
pub use trading::{TradingEnv, TradingEnvConfig, MarketData};
