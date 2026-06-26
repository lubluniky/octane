//! Python Gym compatibility layer using PyO3.
//!
//! This module provides a wrapper around OpenAI Gym environments
//! for seamless integration with Octane's RL algorithms.
//!
//! # Example
//! ```ignore
//! use octane::envs::{GymEnv, Environment};
//! use octane::core::Device;
//!
//! let env = GymEnv::make("CartPole-v1")?;
//! let device = Device::cpu();
//! let obs = env.reset(&device)?;
//! ```

use crate::core::{Device, OctaneError, Result};
use crate::envs::{BoxSpace, DiscreteSpace, Environment, ObsType, Space, StepInfo, StepResult};
use candle_core::Tensor;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;

/// Space type enumeration for Gym environments.
#[derive(Debug, Clone)]
pub enum GymSpace {
    /// Continuous box space.
    Box(BoxSpace),
    /// Discrete action space.
    Discrete(DiscreteSpace),
}

impl Space for GymSpace {
    fn shape(&self) -> &[usize] {
        match self {
            GymSpace::Box(s) => s.shape(),
            GymSpace::Discrete(s) => s.shape(),
        }
    }

    fn flat_dim(&self) -> usize {
        match self {
            GymSpace::Box(s) => s.flat_dim(),
            GymSpace::Discrete(s) => s.flat_dim(),
        }
    }

    fn sample(&self, rng: &mut impl rand::Rng, device: &Device) -> Result<Tensor> {
        match self {
            GymSpace::Box(s) => s.sample(rng, device),
            GymSpace::Discrete(s) => s.sample(rng, device),
        }
    }

    fn contains(&self, tensor: &Tensor) -> Result<bool> {
        match self {
            GymSpace::Box(s) => s.contains(tensor),
            GymSpace::Discrete(s) => s.contains(tensor),
        }
    }
}

/// Configuration for Gym environment wrapper.
#[derive(Debug, Clone)]
pub struct GymEnvConfig {
    /// Whether to render the environment.
    pub render_mode: Option<String>,
    /// Additional keyword arguments for make().
    pub kwargs: HashMap<String, String>,
}

impl Default for GymEnvConfig {
    fn default() -> Self {
        Self {
            render_mode: None,
            kwargs: HashMap::new(),
        }
    }
}

impl GymEnvConfig {
    /// Set render mode (e.g., "human", "rgb_array").
    pub fn render_mode(mut self, mode: &str) -> Self {
        self.render_mode = Some(mode.to_string());
        self
    }

    /// Add a keyword argument.
    pub fn kwarg(mut self, key: &str, value: &str) -> Self {
        self.kwargs.insert(key.to_string(), value.to_string());
        self
    }
}

/// Python Gym environment wrapper.
///
/// Wraps a Python Gym environment using PyO3 for seamless
/// interoperability with Rust-based RL algorithms.
pub struct GymEnv {
    /// Python environment object.
    env: Py<PyAny>,
    /// Environment name/id.
    env_id: String,
    /// Observation space.
    obs_space: GymSpace,
    /// Action space.
    act_space: GymSpace,
    /// Whether actions are discrete.
    discrete_actions: bool,
    /// Current episode step count.
    step_count: usize,
    /// Accumulated episode reward.
    episode_reward: f32,
}

impl GymEnv {
    /// Create a Gym environment by ID.
    ///
    /// # Arguments
    /// * `env_id` - The Gym environment ID (e.g., "CartPole-v1", "MountainCar-v0")
    ///
    /// # Returns
    /// A new GymEnv instance wrapping the Python environment.
    pub fn make(env_id: &str) -> Result<Self> {
        Self::make_with_config(env_id, GymEnvConfig::default())
    }

    /// Create a Gym environment with custom configuration.
    pub fn make_with_config(env_id: &str, config: GymEnvConfig) -> Result<Self> {
        Python::attach(|py| {
            // Import gymnasium (or gym as fallback)
            let gym = py
                .import("gymnasium")
                .or_else(|_| py.import("gym"))
                .map_err(|e| {
                    OctaneError::Environment(format!(
                        "Failed to import gymnasium/gym: {}. \
                         Install with: pip install gymnasium",
                        e
                    ))
                })?;

            // Build kwargs for make()
            let kwargs = PyDict::new(py);
            if let Some(ref mode) = config.render_mode {
                kwargs.set_item("render_mode", mode).map_err(|e| {
                    OctaneError::Environment(format!("Failed to set render_mode: {}", e))
                })?;
            }
            for (key, value) in &config.kwargs {
                kwargs.set_item(key, value).map_err(|e| {
                    OctaneError::Environment(format!("Failed to set kwarg {}: {}", key, e))
                })?;
            }

            // Create environment
            let env = gym
                .call_method("make", (env_id,), Some(&kwargs))
                .map_err(|e| {
                    OctaneError::Environment(format!("Failed to create env '{}': {}", env_id, e))
                })?;

            // Extract observation space
            let obs_space_obj = env
                .getattr("observation_space")
                .map_err(|e| OctaneError::Environment(format!("No observation_space: {}", e)))?;
            let obs_space = Self::extract_space(&obs_space_obj)?;

            // Extract action space
            let action_space_obj = env
                .getattr("action_space")
                .map_err(|e| OctaneError::Environment(format!("No action_space: {}", e)))?;
            let act_space = Self::extract_space(&action_space_obj)?;

            let discrete_actions = matches!(act_space, GymSpace::Discrete(_));

            Ok(Self {
                env: env.unbind(),
                env_id: env_id.to_string(),
                obs_space,
                act_space,
                discrete_actions,
                step_count: 0,
                episode_reward: 0.0,
            })
        })
    }

    /// Extract space information from a Python space object.
    fn extract_space(space_obj: &Bound<'_, PyAny>) -> Result<GymSpace> {
        let space_type = space_obj
            .get_type()
            .name()
            .map_err(|e| OctaneError::Environment(format!("Failed to get space type: {}", e)))?
            .to_string();

        match space_type.as_str() {
            "Box" => {
                // Extract shape
                let shape_obj = space_obj
                    .getattr("shape")
                    .map_err(|e| OctaneError::Environment(format!("No shape attr: {}", e)))?;
                let shape: Vec<usize> = shape_obj.extract().map_err(|e| {
                    OctaneError::Environment(format!("Failed to extract shape: {}", e))
                })?;

                // Extract bounds
                let low_obj = space_obj
                    .getattr("low")
                    .map_err(|e| OctaneError::Environment(format!("No low attr: {}", e)))?;
                let high_obj = space_obj
                    .getattr("high")
                    .map_err(|e| OctaneError::Environment(format!("No high attr: {}", e)))?;

                // Flatten and convert to Vec<f32>
                let low: Vec<f32> = Self::numpy_to_vec(low_obj)?;
                let high: Vec<f32> = Self::numpy_to_vec(high_obj)?;

                Ok(GymSpace::Box(BoxSpace { low, high, shape }))
            }
            "Discrete" => {
                let n: usize = space_obj
                    .getattr("n")
                    .map_err(|e| OctaneError::Environment(format!("No n attr: {}", e)))?
                    .extract()
                    .map_err(|e| OctaneError::Environment(format!("Failed to extract n: {}", e)))?;

                Ok(GymSpace::Discrete(DiscreteSpace::new(n)))
            }
            other => Err(OctaneError::Environment(format!(
                "Unsupported space type: {}. Supported: Box, Discrete",
                other
            ))),
        }
    }

    /// Convert a numpy array to Vec<f32>.
    fn numpy_to_vec(obj: Bound<'_, PyAny>) -> Result<Vec<f32>> {
        // Try to flatten and convert
        let flat = obj
            .call_method0("flatten")
            .map_err(|e| OctaneError::Environment(format!("Failed to flatten array: {}", e)))?;

        let list = flat
            .call_method0("tolist")
            .map_err(|e| OctaneError::Environment(format!("Failed to convert to list: {}", e)))?;

        list.extract::<Vec<f32>>()
            .or_else(|_| {
                // Try converting from f64
                list.extract::<Vec<f64>>()
                    .map(|v| v.into_iter().map(|x| x as f32).collect())
            })
            .map_err(|e| OctaneError::Environment(format!("Failed to extract vec: {}", e)))
    }

    /// Convert a Python observation to a Candle Tensor.
    fn obs_to_tensor(&self, obs: &Bound<'_, PyAny>, device: &Device) -> Result<Tensor> {
        let flat = obs
            .call_method0("flatten")
            .map_err(|e| OctaneError::Environment(format!("Failed to flatten obs: {}", e)))?;

        let data: Vec<f32> = flat
            .call_method0("tolist")
            .map_err(|e| OctaneError::Environment(format!("Failed to convert obs: {}", e)))?
            .extract::<Vec<f32>>()
            .or_else(|_| {
                flat.call_method0("tolist")
                    .unwrap()
                    .extract::<Vec<f64>>()
                    .map(|v| v.into_iter().map(|x| x as f32).collect())
            })
            .map_err(|e| OctaneError::Environment(format!("Failed to extract obs: {}", e)))?;

        let shape = self.obs_space.shape();
        let candle_device = device.to_candle()?;
        Tensor::from_slice(&data, shape, &candle_device).map_err(Into::into)
    }

    /// Convert a Candle Tensor action to Python format.
    fn tensor_to_action<'py>(&self, action: &Tensor, py: Python<'py>) -> Result<Bound<'py, PyAny>> {
        if self.discrete_actions {
            // Discrete action: extract single integer
            let action_val: Vec<f32> = action.flatten_all()?.to_vec1()?;
            let action_int = action_val[0] as i64;
            Ok(action_int.into_pyobject(py).unwrap().into_any())
        } else {
            // Continuous action: convert to numpy array
            let action_vec: Vec<f32> = action.flatten_all()?.to_vec1()?;
            let np = py
                .import("numpy")
                .map_err(|e| OctaneError::Environment(format!("Failed to import numpy: {}", e)))?;

            let array = np
                .call_method1("array", (action_vec,))
                .map_err(|e| OctaneError::Environment(format!("Failed to create array: {}", e)))?;

            let shape = self.act_space.shape();
            let reshaped = array
                .call_method1("reshape", (shape,))
                .map_err(|e| OctaneError::Environment(format!("Failed to reshape: {}", e)))?;

            Ok(reshaped.into_any())
        }
    }

    /// Get the environment ID.
    pub fn env_id(&self) -> &str {
        &self.env_id
    }

    /// Check if actions are discrete.
    pub fn is_discrete(&self) -> bool {
        self.discrete_actions
    }
}

impl Environment for GymEnv {
    type ObsSpace = GymSpace;
    type ActSpace = GymSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActSpace {
        &self.act_space
    }

    fn reset(&mut self, device: &Device) -> Result<ObsType> {
        self.step_count = 0;
        self.episode_reward = 0.0;

        Python::attach(|py| {
            let env = self.env.bind(py);

            // Call reset() - returns (obs, info) in gymnasium, just obs in old gym
            let result = env
                .call_method0("reset")
                .map_err(|e| OctaneError::Environment(format!("reset() failed: {}", e)))?;

            // Handle both gymnasium (tuple) and old gym (single value) return formats
            let obs = if result.is_instance_of::<PyTuple>() {
                result
                    .get_item(0)
                    .map_err(|e| OctaneError::Environment(format!("Failed to get obs: {}", e)))?
            } else {
                result
            };

            self.obs_to_tensor(&obs, device)
        })
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        Python::attach(|py| {
            let env = self.env.bind(py);
            let py_action = self.tensor_to_action(action, py)?;

            // Call step(action)
            let result = env
                .call_method1("step", (py_action,))
                .map_err(|e| OctaneError::Environment(format!("step() failed: {}", e)))?;

            // Gymnasium returns: (obs, reward, terminated, truncated, info)
            // Old gym returns: (obs, reward, done, info)
            let tuple_len = result
                .len()
                .map_err(|e| OctaneError::Environment(format!("Invalid step result: {}", e)))?;

            let (obs, reward, terminated, truncated) = if tuple_len == 5 {
                // Gymnasium format
                let obs = result
                    .get_item(0)
                    .map_err(|e| OctaneError::Environment(format!("Failed to get obs: {}", e)))?;
                let reward: f32 = result
                    .get_item(1)
                    .map_err(|e| OctaneError::Environment(format!("Failed to get reward: {}", e)))?
                    .extract()
                    .or_else(|_| {
                        result
                            .get_item(1)
                            .unwrap()
                            .extract::<f64>()
                            .map(|r| r as f32)
                    })
                    .map_err(|e| {
                        OctaneError::Environment(format!("Failed to extract reward: {}", e))
                    })?;
                let terminated: bool = result
                    .get_item(2)
                    .map_err(|e| {
                        OctaneError::Environment(format!("Failed to get terminated: {}", e))
                    })?
                    .extract()
                    .map_err(|e| {
                        OctaneError::Environment(format!("Failed to extract terminated: {}", e))
                    })?;
                let truncated: bool = result
                    .get_item(3)
                    .map_err(|e| {
                        OctaneError::Environment(format!("Failed to get truncated: {}", e))
                    })?
                    .extract()
                    .map_err(|e| {
                        OctaneError::Environment(format!("Failed to extract truncated: {}", e))
                    })?;

                (obs, reward, terminated, truncated)
            } else if tuple_len == 4 {
                // Old gym format
                let obs = result
                    .get_item(0)
                    .map_err(|e| OctaneError::Environment(format!("Failed to get obs: {}", e)))?;
                let reward: f32 = result
                    .get_item(1)
                    .map_err(|e| OctaneError::Environment(format!("Failed to get reward: {}", e)))?
                    .extract()
                    .or_else(|_| {
                        result
                            .get_item(1)
                            .unwrap()
                            .extract::<f64>()
                            .map(|r| r as f32)
                    })
                    .map_err(|e| {
                        OctaneError::Environment(format!("Failed to extract reward: {}", e))
                    })?;
                let done: bool = result
                    .get_item(2)
                    .map_err(|e| OctaneError::Environment(format!("Failed to get done: {}", e)))?
                    .extract()
                    .map_err(|e| {
                        OctaneError::Environment(format!("Failed to extract done: {}", e))
                    })?;

                // Old gym doesn't distinguish terminated vs truncated
                (obs, reward, done, false)
            } else {
                return Err(OctaneError::Environment(format!(
                    "Unexpected step result tuple length: {}",
                    tuple_len
                )));
            };

            self.step_count += 1;
            self.episode_reward += reward;

            let observation = self.obs_to_tensor(&obs, device)?;

            let info = if terminated || truncated {
                Some(StepInfo {
                    episode_return: Some(self.episode_reward),
                    episode_length: Some(self.step_count),
                    extra: HashMap::new(),
                })
            } else {
                None
            };

            Ok(StepResult {
                observation,
                reward,
                terminated,
                truncated,
                info,
            })
        })
    }

    fn render(&self) -> Result<()> {
        Python::attach(|py| {
            let env = self.env.bind(py);
            env.call_method0("render")
                .map_err(|e| OctaneError::Environment(format!("render() failed: {}", e)))?;
            Ok(())
        })
    }

    fn close(&mut self) -> Result<()> {
        Python::attach(|py| {
            let env = self.env.bind(py);
            env.call_method0("close")
                .map_err(|e| OctaneError::Environment(format!("close() failed: {}", e)))?;
            Ok(())
        })
    }

    fn name(&self) -> &str {
        &self.env_id
    }
}

// GymEnv cannot be Clone due to Python GIL requirements,
// but it is Send + Sync because Py<PyAny> is thread-safe
unsafe impl Send for GymEnv {}
unsafe impl Sync for GymEnv {}

/// Create multiple Gym environments for vectorization.
///
/// This creates independent Python environment instances that can
/// be used with VecEnv for parallel training.
pub fn make_vec_gym_envs(env_id: &str, num_envs: usize) -> Result<Vec<GymEnv>> {
    let mut envs = Vec::with_capacity(num_envs);
    for _ in 0..num_envs {
        envs.push(GymEnv::make(env_id)?);
    }
    Ok(envs)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Note: These tests require Python with gymnasium installed
    // Run with: cargo test --features gym -- --ignored

    #[test]
    #[ignore]
    fn test_gym_cartpole() {
        let mut env = GymEnv::make("CartPole-v1").expect("Failed to create CartPole");
        let device = Device::cpu();

        let obs = env.reset(&device).expect("Reset failed");
        assert_eq!(obs.dims(), &[4]);

        // Take a random action
        let action = Tensor::from_slice(&[0.0f32], &[1], &device.to_candle().unwrap()).unwrap();
        let result = env.step(&action, &device).expect("Step failed");
        assert_eq!(result.observation.dims(), &[4]);
    }

    #[test]
    #[ignore]
    fn test_gym_continuous() {
        let mut env = GymEnv::make("Pendulum-v1").expect("Failed to create Pendulum");
        let device = Device::cpu();

        let obs = env.reset(&device).expect("Reset failed");
        assert_eq!(obs.dims(), &[3]);

        // Take a continuous action
        let action = Tensor::from_slice(&[0.5f32], &[1], &device.to_candle().unwrap()).unwrap();
        let result = env.step(&action, &device).expect("Step failed");
        assert_eq!(result.observation.dims(), &[3]);
    }
}
