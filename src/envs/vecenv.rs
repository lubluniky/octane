//! Vectorized environment implementation for massive parallelization.

use crate::core::{Device, Result};
#[cfg(feature = "distributed")]
use crate::core::OctaneError;
use crate::envs::{Environment, ObsType, StepInfo, StepResult};
use candle_core::Tensor;
use rayon::prelude::*;
#[cfg(feature = "distributed")]
use std::thread::{self, JoinHandle};

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

/// Message sent to environment workers.
#[cfg(feature = "distributed")]
enum WorkerCommand {
    /// Step the environment with the given action and device.
    Step(Tensor, Device, bool),
    /// Reset the environment.
    Reset(Device),
    /// Shutdown the worker thread.
    Shutdown,
}

/// Result from an environment worker.
#[cfg(feature = "distributed")]
enum WorkerResponse {
    /// Step result with optional auto-reset observation.
    Step(Result<(StepResult, Option<ObsType>)>),
    /// Reset result.
    Reset(Result<ObsType>),
}

/// A persistent worker that owns an environment and runs on a dedicated thread.
#[cfg(feature = "distributed")]
struct EnvWorker {
    /// Channel to send commands to the worker.
    cmd_tx: crossbeam::channel::Sender<WorkerCommand>,
    /// Channel to receive responses from the worker.
    resp_rx: crossbeam::channel::Receiver<WorkerResponse>,
    /// Handle to the worker thread.
    handle: Option<JoinHandle<()>>,
}

#[cfg(feature = "distributed")]
impl EnvWorker {
    /// Create a new worker that owns the given environment.
    fn new<E: Environment + Clone + 'static>(mut env: E) -> Self {
        let (cmd_tx, cmd_rx) = crossbeam::channel::bounded::<WorkerCommand>(1);
        let (resp_tx, resp_rx) = crossbeam::channel::bounded::<WorkerResponse>(1);

        let handle = thread::spawn(move || {
            while let Ok(cmd) = cmd_rx.recv() {
                match cmd {
                    WorkerCommand::Step(action, device, auto_reset) => {
                        let result = env.step(&action, &device);
                        let response = match result {
                            Ok(step_result) => {
                                let new_obs = if step_result.done() && auto_reset {
                                    match env.reset(&device) {
                                        Ok(obs) => Some(obs),
                                        Err(e) => {
                                            let _ = resp_tx.send(WorkerResponse::Step(Err(e)));
                                            continue;
                                        }
                                    }
                                } else {
                                    None
                                };
                                WorkerResponse::Step(Ok((step_result, new_obs)))
                            }
                            Err(e) => WorkerResponse::Step(Err(e)),
                        };
                        if resp_tx.send(response).is_err() {
                            break;
                        }
                    }
                    WorkerCommand::Reset(device) => {
                        let result = env.reset(&device);
                        if resp_tx.send(WorkerResponse::Reset(result)).is_err() {
                            break;
                        }
                    }
                    WorkerCommand::Shutdown => {
                        let _ = env.close();
                        break;
                    }
                }
            }
        });

        Self {
            cmd_tx,
            resp_rx,
            handle: Some(handle),
        }
    }

    /// Send a step command to the worker (non-blocking).
    fn send_step(&self, action: Tensor, device: Device, auto_reset: bool) -> Result<()> {
        self.cmd_tx
            .send(WorkerCommand::Step(action, device, auto_reset))
            .map_err(|_| OctaneError::Environment("Worker channel disconnected".into()))
    }

    /// Receive the step result (blocking).
    fn recv_step(&self) -> Result<(StepResult, Option<ObsType>)> {
        match self.resp_rx.recv() {
            Ok(WorkerResponse::Step(result)) => result,
            Ok(_) => Err(OctaneError::Environment("Unexpected response type".into())),
            Err(_) => Err(OctaneError::Environment("Worker channel disconnected".into())),
        }
    }

    /// Send a reset command to the worker (non-blocking).
    fn send_reset(&self, device: Device) -> Result<()> {
        self.cmd_tx
            .send(WorkerCommand::Reset(device))
            .map_err(|_| OctaneError::Environment("Worker channel disconnected".into()))
    }

    /// Receive the reset result (blocking).
    fn recv_reset(&self) -> Result<ObsType> {
        match self.resp_rx.recv() {
            Ok(WorkerResponse::Reset(result)) => result,
            Ok(_) => Err(OctaneError::Environment("Unexpected response type".into())),
            Err(_) => Err(OctaneError::Environment("Worker channel disconnected".into())),
        }
    }

    /// Shutdown the worker thread gracefully.
    fn shutdown(&mut self) {
        let _ = self.cmd_tx.send(WorkerCommand::Shutdown);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

#[cfg(feature = "distributed")]
impl Drop for EnvWorker {
    fn drop(&mut self) {
        self.shutdown();
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
///
/// Uses `par_iter_mut()` for lock-free parallel stepping - each thread gets
/// exclusive `&mut` access to its environment without synchronization overhead.
/// This is significantly faster on ARM (Apple Silicon M-series) where atomic
/// operations and mutex locks have higher overhead than on x86.
///
/// When the `distributed` feature is enabled and `new_async()` is used, spawns
/// a persistent worker pool where each environment runs on its own dedicated
/// thread. This eliminates the overhead of wrapping environments in `Arc<Mutex>`
/// on every step, providing much better performance for I/O-bound environments.
pub struct VecEnv<E: Environment + Clone> {
    /// Individual environments - no Arc/Mutex needed since par_iter_mut()
    /// guarantees exclusive access per thread.
    /// Note: When using the persistent worker pool, this will be empty.
    envs: Vec<E>,
    /// Number of environments.
    num_envs: usize,
    /// Configuration.
    config: VecEnvConfig,
    /// Cached observation space.
    obs_space: E::ObsSpace,
    /// Cached action space.
    act_space: E::ActSpace,
    /// Persistent worker pool for async stepping (distributed feature only).
    /// Each worker owns its environment and runs on a dedicated thread.
    #[cfg(feature = "distributed")]
    workers: Option<Vec<EnvWorker>>,
}

impl<E: Environment + Clone + 'static> VecEnv<E> {
    /// Create a new vectorized environment for synchronous stepping.
    ///
    /// Uses Rayon's `par_iter_mut()` for parallel stepping, which is optimal
    /// for CPU-bound environments where the step time is predictable.
    pub fn new(template_envs: Vec<E>, num_envs: usize) -> Self {
        let obs_space = template_envs[0].observation_space().clone();
        let act_space = template_envs[0].action_space().clone();

        // Clone environments to reach num_envs total
        let envs: Vec<E> = (0..num_envs)
            .map(|i| template_envs[i % template_envs.len()].clone())
            .collect();

        Self {
            envs,
            num_envs,
            config: VecEnvConfig {
                num_envs,
                auto_reset: true,
            },
            obs_space,
            act_space,
            #[cfg(feature = "distributed")]
            workers: None,
        }
    }

    /// Create a new vectorized environment with persistent worker threads.
    ///
    /// Each environment runs on its own dedicated thread, communicating via
    /// crossbeam channels. This eliminates the overhead of creating `Arc<Mutex>`
    /// wrappers on every step, providing much better performance for I/O-bound
    /// or latency-sensitive environments (e.g., network simulators, trading envs).
    ///
    /// Use `step_async()` with this mode for optimal performance.
    #[cfg(feature = "distributed")]
    pub fn new_async(template_envs: Vec<E>, num_envs: usize) -> Self {
        let obs_space = template_envs[0].observation_space().clone();
        let act_space = template_envs[0].action_space().clone();

        // Create environments and spawn workers
        let workers: Vec<EnvWorker> = (0..num_envs)
            .map(|i| {
                let env = template_envs[i % template_envs.len()].clone();
                EnvWorker::new(env)
            })
            .collect();

        Self {
            envs: Vec::new(), // Environments are owned by workers
            num_envs,
            config: VecEnvConfig {
                num_envs,
                auto_reset: true,
            },
            obs_space,
            act_space,
            workers: Some(workers),
        }
    }

    /// Check if this VecEnv is using the persistent worker pool.
    #[cfg(feature = "distributed")]
    #[inline]
    pub fn is_async(&self) -> bool {
        self.workers.is_some()
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
    ///
    /// When using the persistent worker pool (`new_async()`), sends reset commands
    /// to all workers in parallel and collects responses.
    pub fn reset(&mut self, device: &Device) -> Result<Tensor> {
        #[cfg(feature = "distributed")]
        if let Some(ref workers) = self.workers {
            // Use the worker pool: send all reset commands in parallel
            for worker in workers {
                worker.send_reset(*device)?;
            }

            // Collect all responses
            let mut obs_vec = Vec::with_capacity(self.num_envs);
            for worker in workers {
                let obs = worker.recv_reset()?;
                obs_vec.push(obs);
            }

            return Tensor::stack(&obs_vec, 0).map_err(Into::into);
        }

        // Standard mode: use rayon parallel iteration
        let observations: Vec<Result<ObsType>> = self
            .envs
            .par_iter_mut()
            .map(|env| env.reset(device))
            .collect();

        // Check for errors and stack observations
        let obs_vec: Vec<Tensor> = observations.into_iter().collect::<Result<Vec<_>>>()?;

        Tensor::stack(&obs_vec, 0).map_err(Into::into)
    }

    /// Step all environments in parallel.
    pub fn step(&mut self, actions: &Tensor, device: &Device) -> Result<VecStepResult> {
        let num_envs = self.num_envs;
        let auto_reset = self.config.auto_reset;

        // Split actions for each environment
        let action_list: Vec<Tensor> = (0..num_envs)
            .map(|i| actions.get(i))
            .collect::<candle_core::Result<Vec<_>>>()?;

        // Parallel step - lock-free with par_iter_mut()
        let results: Vec<Result<(StepResult, Option<ObsType>)>> = self
            .envs
            .par_iter_mut()
            .zip(action_list.par_iter())
            .map(|(env, action)| {
                let result = env.step(action, device)?;

                // Auto-reset if done
                let new_obs = if result.done() && auto_reset {
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

    /// Step with async processing using the persistent worker pool.
    ///
    /// **IMPORTANT**: For optimal performance, create the VecEnv using `new_async()`
    /// which spawns dedicated worker threads. This eliminates the overhead of
    /// wrapping environments in `Arc<Mutex>` on every step.
    ///
    /// When using the worker pool:
    /// - Sends step commands to all workers in parallel via crossbeam channels
    /// - Workers process steps on their dedicated threads (no locking needed)
    /// - Collects results as they complete
    ///
    /// When NOT using the worker pool (fallback for VecEnv created with `new()`):
    /// - Falls back to the synchronous `step()` method wrapped in spawn_blocking
    #[cfg(feature = "distributed")]
    pub async fn step_async(&mut self, actions: &Tensor, device: &Device) -> Result<VecStepResult> {
        let num_envs = self.num_envs;
        let auto_reset = self.config.auto_reset;

        // Split actions for each environment
        let action_list: Vec<Tensor> = (0..num_envs)
            .map(|i| actions.get(i))
            .collect::<candle_core::Result<Vec<_>>>()?;

        // FAST PATH: Use persistent worker pool if available
        if let Some(ref workers) = self.workers {
            // Send all step commands in parallel (non-blocking sends)
            for (worker, action) in workers.iter().zip(action_list.into_iter()) {
                worker.send_step(action, *device, auto_reset)?;
            }

            // Collect results (blocking receives, but workers run in parallel)
            let mut obs_vec = Vec::with_capacity(num_envs);
            let mut rewards = Vec::with_capacity(num_envs);
            let mut terminated = Vec::with_capacity(num_envs);
            let mut truncated = Vec::with_capacity(num_envs);
            let mut infos = Vec::with_capacity(num_envs);

            for worker in workers {
                let (step_result, auto_reset_obs) = worker.recv_step()?;

                let obs = auto_reset_obs.unwrap_or(step_result.observation);
                obs_vec.push(obs);
                rewards.push(step_result.reward);
                terminated.push(if step_result.terminated { 1.0f32 } else { 0.0 });
                truncated.push(if step_result.truncated { 1.0f32 } else { 0.0 });
                infos.push(step_result.info);
            }

            let candle_device = device.to_candle()?;
            return Ok(VecStepResult {
                observations: Tensor::stack(&obs_vec, 0)?,
                rewards: Tensor::from_slice(&rewards, &[num_envs], &candle_device)?,
                terminated: Tensor::from_slice(&terminated, &[num_envs], &candle_device)?,
                truncated: Tensor::from_slice(&truncated, &[num_envs], &candle_device)?,
                infos,
            });
        }

        // SLOW PATH (fallback): VecEnv was created with new() instead of new_async()
        // Use Rayon parallel iteration in a spawn_blocking to not block the async runtime
        //
        // NOTE: For optimal performance, use new_async() which spawns persistent workers.
        // This fallback still works but involves moving environments in and out of the closure.
        use tokio::task;

        let envs = std::mem::take(&mut self.envs);
        let device_clone = *device;

        let (returned_envs, results): (Vec<E>, Vec<Result<(StepResult, Option<ObsType>)>>) =
            task::spawn_blocking(move || {
                // Process in parallel, collecting both results and environments
                let processed: Vec<(E, Result<(StepResult, Option<ObsType>)>)> = envs
                    .into_par_iter()
                    .zip(action_list.into_par_iter())
                    .map(|(mut env, action)| {
                        let result = (|| {
                            let step_result = env.step(&action, &device_clone)?;
                            let new_obs = if step_result.done() && auto_reset {
                                Some(env.reset(&device_clone)?)
                            } else {
                                None
                            };
                            Ok((step_result, new_obs))
                        })();
                        (env, result)
                    })
                    .collect();

                // Separate environments from results
                processed.into_iter().unzip()
            })
            .await
            .map_err(|e| OctaneError::Environment(format!("Task join error: {}", e)))?;

        self.envs = returned_envs;

        // Collect results
        let mut obs_vec = Vec::with_capacity(num_envs);
        let mut rewards = Vec::with_capacity(num_envs);
        let mut terminated = Vec::with_capacity(num_envs);
        let mut truncated = Vec::with_capacity(num_envs);
        let mut infos = Vec::with_capacity(num_envs);

        for result in results {
            let (step_result, auto_reset_obs) = result?;

            let obs = auto_reset_obs.unwrap_or(step_result.observation);
            obs_vec.push(obs);
            rewards.push(step_result.reward);
            terminated.push(if step_result.terminated { 1.0f32 } else { 0.0 });
            truncated.push(if step_result.truncated { 1.0f32 } else { 0.0 });
            infos.push(step_result.info);
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
    ///
    /// When using the persistent worker pool, sends shutdown commands to all
    /// workers and waits for them to terminate gracefully.
    pub fn close(&mut self) -> Result<()> {
        #[cfg(feature = "distributed")]
        if let Some(ref mut workers) = self.workers {
            // Shutdown all workers - the Drop impl handles joining threads
            for worker in workers.iter_mut() {
                worker.shutdown();
            }
            self.workers = None;
            return Ok(());
        }

        // Standard mode: close environments directly
        for env in &mut self.envs {
            env.close()?;
        }
        Ok(())
    }
}

// Note: VecEnv is Send + Sync when E is Send + Sync
