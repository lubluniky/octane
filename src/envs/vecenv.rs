//! Vectorized environment implementation for massive parallelization.

use crate::core::{Device, OctaneError, Result};
use crate::envs::{Environment, ObsType, StepInfo, StepResult};
use candle_core::Tensor;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Configuration for vectorized environments.
#[derive(Debug, Clone)]
pub struct VecEnvConfig {
    /// Number of parallel environments.
    pub num_envs: usize,
    /// Whether to auto-reset on done.
    pub auto_reset: bool,
}

impl Default for VecEnvConfig {
    fn default() -> Self {
        Self {
            num_envs: 1,
            auto_reset: true,
        }
    }
}

/// Vectorized step result for batch processing.
#[derive(Debug)]
pub struct VecStepResult {
    /// Batched observations [num_envs, ...obs_shape].
    pub observations: Tensor,
    /// Rewards for each env [num_envs].
    pub rewards: Tensor,
    /// Termination flags [num_envs].
    pub terminated: Tensor,
    /// Truncation flags [num_envs].
    pub truncated: Tensor,
    /// Info for each environment.
    pub infos: Vec<Option<StepInfo>>,
}

impl VecStepResult {
    /// Get done mask (terminated OR truncated).
    pub fn dones(&self) -> Result<Tensor> {
        // Logical OR of terminated and truncated
        let t = self.terminated.to_dtype(candle_core::DType::F32)?;
        let tr = self.truncated.to_dtype(candle_core::DType::F32)?;
        let sum = (&t + &tr)?;
        Ok(sum.ge(1.0)?.to_dtype(candle_core::DType::F32)?)
    }
}

/// Vectorized environment wrapper for running multiple envs in parallel.
pub struct VecEnv<E: Environment + Clone> {
    /// Individual environments wrapped in Arc<Mutex> for parallel access.
    envs: Vec<Arc<Mutex<E>>>,
    /// Number of environments.
    num_envs: usize,
    /// Configuration.
    config: VecEnvConfig,
    /// Cached observation space.
    obs_space: E::ObsSpace,
    /// Cached action space.
    act_space: E::ActSpace,
}

impl<E: Environment + Clone + 'static> VecEnv<E> {
    /// Create a new vectorized environment.
    pub fn new(template_envs: Vec<E>, num_envs: usize) -> Self {
        let obs_space = template_envs[0].observation_space().clone();
        let act_space = template_envs[0].action_space().clone();

        // Clone environments to reach num_envs total
        let mut envs: Vec<Arc<Mutex<E>>> = Vec::with_capacity(num_envs);
        for i in 0..num_envs {
            let env = template_envs[i % template_envs.len()].clone();
            envs.push(Arc::new(Mutex::new(env)));
        }

        Self {
            envs,
            num_envs,
            config: VecEnvConfig {
                num_envs,
                auto_reset: true,
            },
            obs_space,
            act_space,
        }
    }

    /// Number of parallel environments.
    #[inline]
    pub fn num_envs(&self) -> usize {
        self.num_envs
    }

    /// Get observation space (single env).
    pub fn observation_space(&self) -> &E::ObsSpace {
        &self.obs_space
    }

    /// Get action space (single env).
    pub fn action_space(&self) -> &E::ActSpace {
        &self.act_space
    }

    /// Reset all environments in parallel.
    pub fn reset(&mut self, device: &Device) -> Result<Tensor> {
        let observations: Vec<Result<ObsType>> = self
            .envs
            .par_iter()
            .map(|env| {
                let mut env = env
                    .lock()
                    .map_err(|e| OctaneError::Environment(format!("Lock poisoned: {}", e)))?;
                env.reset(device)
            })
            .collect();

        // Check for errors and stack observations
        let obs_vec: Vec<Tensor> = observations.into_iter().collect::<Result<Vec<_>>>()?;

        Tensor::stack(&obs_vec, 0).map_err(Into::into)
    }

    /// Step all environments in parallel.
    pub fn step(&mut self, actions: &Tensor, device: &Device) -> Result<VecStepResult> {
        let num_envs = self.num_envs;

        // Split actions for each environment
        let action_list: Vec<Tensor> = (0..num_envs)
            .map(|i| actions.get(i))
            .collect::<candle_core::Result<Vec<_>>>()?;

        // Parallel step
        let results: Vec<Result<(StepResult, Option<ObsType>)>> = self
            .envs
            .par_iter()
            .zip(action_list.par_iter())
            .map(|(env, action)| {
                let mut env = env
                    .lock()
                    .map_err(|e| OctaneError::Environment(format!("Lock poisoned: {}", e)))?;
                let result = env.step(action, device)?;

                // Auto-reset if done
                let new_obs = if result.done() && self.config.auto_reset {
                    Some(env.reset(device)?)
                } else {
                    None
                };

                Ok((result, new_obs))
            })
            .collect();

        // Collect results
        let mut obs_vec = Vec::with_capacity(num_envs);
        let mut rewards = Vec::with_capacity(num_envs);
        let mut terminated = Vec::with_capacity(num_envs);
        let mut truncated = Vec::with_capacity(num_envs);
        let mut infos = Vec::with_capacity(num_envs);

        for result in results {
            let (step_result, auto_reset_obs) = result?;

            // Use auto-reset obs if available, otherwise use step obs
            let obs = auto_reset_obs.unwrap_or(step_result.observation);
            obs_vec.push(obs);
            rewards.push(step_result.reward);
            terminated.push(if step_result.terminated { 1.0f32 } else { 0.0 });
            truncated.push(if step_result.truncated { 1.0f32 } else { 0.0 });
            infos.push(step_result.info);
        }

        // Stack into batched tensors
        let candle_device = device.to_candle()?;
        let observations = Tensor::stack(&obs_vec, 0)?;
        let rewards = Tensor::from_slice(&rewards, &[num_envs], &candle_device)?;
        let terminated = Tensor::from_slice(&terminated, &[num_envs], &candle_device)?;
        let truncated = Tensor::from_slice(&truncated, &[num_envs], &candle_device)?;

        Ok(VecStepResult {
            observations,
            rewards,
            terminated,
            truncated,
            infos,
        })
    }

    /// Step with async processing (useful for I/O bound envs).
    #[cfg(feature = "tokio")]
    pub async fn step_async(&mut self, actions: &Tensor, device: &Device) -> Result<VecStepResult> {
        // For I/O bound environments, use tokio for concurrent stepping
        use tokio::task;

        let num_envs = self.num_envs;
        let action_list: Vec<Tensor> = (0..num_envs)
            .map(|i| actions.get(i))
            .collect::<candle_core::Result<Vec<_>>>()?;

        let mut handles = Vec::with_capacity(num_envs);
        let device_clone = *device;

        for (env, action) in self.envs.iter().zip(action_list.into_iter()) {
            let env = Arc::clone(env);
            let action = action.clone();
            let device = device_clone;

            handles.push(task::spawn_blocking(move || {
                let mut env = env.lock().unwrap();
                env.step(&action, &device)
            }));
        }

        // Collect results (simplified - full impl would mirror sync version)
        let mut obs_vec = Vec::with_capacity(num_envs);
        let mut rewards = Vec::with_capacity(num_envs);
        let mut terminated = Vec::with_capacity(num_envs);
        let mut truncated = Vec::with_capacity(num_envs);
        let mut infos = Vec::with_capacity(num_envs);

        for handle in handles {
            let result = handle
                .await
                .map_err(|e| OctaneError::Environment(format!("Task join error: {}", e)))??;

            obs_vec.push(result.observation);
            rewards.push(result.reward);
            terminated.push(if result.terminated { 1.0f32 } else { 0.0 });
            truncated.push(if result.truncated { 1.0f32 } else { 0.0 });
            infos.push(result.info);
        }

        let candle_device = device.to_candle()?;
        Ok(VecStepResult {
            observations: Tensor::stack(&obs_vec, 0)?,
            rewards: Tensor::from_slice(&rewards, &[num_envs], &candle_device)?,
            terminated: Tensor::from_slice(&terminated, &[num_envs], &candle_device)?,
            truncated: Tensor::from_slice(&truncated, &[num_envs], &candle_device)?,
            infos,
        })
    }

    /// Close all environments.
    pub fn close(&mut self) -> Result<()> {
        for env in &self.envs {
            let mut env = env
                .lock()
                .map_err(|e| OctaneError::Environment(format!("Lock poisoned: {}", e)))?;
            env.close()?;
        }
        Ok(())
    }
}

// Note: VecEnv is Send + Sync when E is Send + Sync (via Arc<Mutex<E>>)
