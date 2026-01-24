//! Environment module: Gym-like trait interfaces with VecEnv support.

mod space;
mod trading;
mod traits;
mod vecenv;

pub use space::{BoxSpace, DiscreteSpace, Space};
pub use trading::{MarketData, TradingEnv, TradingEnvConfig};
pub use traits::{ActionType, Environment, ObsType, StepInfo, StepResult};
pub use vecenv::{VecEnv, VecEnvConfig};
