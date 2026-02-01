//! Core environment traits (Gym-like interface).

use crate::core::{Device, Result};
use crate::envs::Space;
use candle_core::Tensor;

/// Type alias for observations.
pub type ObsType = Tensor;

/// Type alias for actions.
pub type ActionType = Tensor;

/// Result of a single environment step.
#[derive(Debug)]
pub struct StepResult {
    /// New observation after taking action.
    pub observation: ObsType,
    /// Reward received from this transition.
    pub reward: f32,
    /// Whether the episode has terminated (goal reached or failure).
    pub terminated: bool,
    /// Whether the episode was truncated (time limit).
    pub truncated: bool,
    /// Additional info (optional).
    pub info: Option<StepInfo>,
}

impl StepResult {
    /// Check if episode is done (terminated OR truncated).
    #[inline]
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }
}

/// Additional step information.
#[derive(Debug, Clone, Default)]
pub struct StepInfo {
    /// Episode return if episode just ended.
    pub episode_return: Option<f32>,
    /// Episode length if episode just ended.
    pub episode_length: Option<usize>,
    /// Custom key-value pairs.
    pub extra: std::collections::HashMap<String, f32>,
}

/// Core environment trait (single environment instance).
pub trait Environment: Send + Sync + 'static {
    /// Observation space type.
    type ObsSpace: Space;
    /// Action space type.
    type ActSpace: Space;

    /// Get the observation space.
    fn observation_space(&self) -> &Self::ObsSpace;

    /// Get the action space.
    fn action_space(&self) -> &Self::ActSpace;

    /// Reset the environment and return initial observation.
    fn reset(&mut self, device: &Device) -> Result<ObsType>;

    /// Take a step with the given action.
    fn step(&mut self, action: &ActionType, device: &Device) -> Result<StepResult>;

    /// Render the environment (optional).
    fn render(&self) -> Result<()> {
        Ok(())
    }

    /// Close the environment and release resources.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// Get environment name/identifier.
    fn name(&self) -> &str {
        "Environment"
    }

    /// Create a vectorized version of this environment.
    fn make_vectorized(self, num_envs: usize) -> crate::envs::VecEnv<Self>
    where
        Self: Sized + Clone,
    {
        #[cfg(feature = "distributed")]
        {
            crate::envs::VecEnv::new_async(vec![self], num_envs)
        }
        #[cfg(not(feature = "distributed"))]
        {
            crate::envs::VecEnv::new(vec![self], num_envs)
        }
    }
}
