//! Distributed training infrastructure for Octane.
//!
//! This module provides distributed training capabilities including:
//! - Distributed configuration for multi-worker training
//! - Worker pools for async environment rollouts
//! - Gradient aggregation for distributed SGD
//!
//! # Architecture
//!
//! The distributed training follows a parameter server architecture:
//! - Workers collect rollouts and compute gradients
//! - Gradients are aggregated (averaged) across workers
//! - Parameters are synchronized after each update
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::distributed::{DistributedConfig, WorkerPool, GradientAggregator};
//!
//! let config = DistributedConfig::new(4, 0); // 4 workers, this is rank 0
//! let pool = WorkerPool::new(config.clone())?;
//! let aggregator = GradientAggregator::new(config)?;
//! ```

use crate::core::{Device, OctaneError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Backend protocol for distributed communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DistributedBackend {
    /// TCP-based communication (simple, low-latency for small clusters).
    #[default]
    Tcp,
    /// gRPC-based communication (better for large clusters, requires tonic).
    #[cfg(feature = "grpc")]
    Grpc,
}

/// Configuration for distributed training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    /// Total number of workers in the distributed setup.
    pub world_size: usize,

    /// Rank of this worker (0 to world_size - 1).
    pub rank: usize,

    /// Communication backend.
    pub backend: DistributedBackend,

    /// Address of the parameter server (for centralized aggregation).
    pub master_addr: String,

    /// Port for communication.
    pub master_port: u16,

    /// Timeout for communication operations in milliseconds.
    pub timeout_ms: u64,

    /// Whether this worker is the master (rank 0).
    pub is_master: bool,

    /// Enable gradient compression for bandwidth reduction.
    pub gradient_compression: bool,

    /// Compression threshold (gradients smaller than this won't be compressed).
    pub compression_threshold: f32,

    /// Synchronization mode.
    pub sync_mode: SyncMode,
}

/// Synchronization mode for distributed training.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum SyncMode {
    /// Synchronous SGD - wait for all workers before update.
    #[default]
    Synchronous,
    /// Asynchronous SGD - update as gradients arrive.
    Asynchronous,
    /// Bounded staleness - allow up to N stale updates.
    BoundedStaleness(usize),
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            backend: DistributedBackend::default(),
            master_addr: "127.0.0.1".to_string(),
            master_port: 29500,
            timeout_ms: 30000,
            is_master: true,
            gradient_compression: false,
            compression_threshold: 1e-4,
            sync_mode: SyncMode::default(),
        }
    }
}

impl DistributedConfig {
    /// Create a new distributed config.
    pub fn new(world_size: usize, rank: usize) -> Self {
        Self {
            world_size,
            rank,
            is_master: rank == 0,
            ..Default::default()
        }
    }

    /// Builder-style setter for backend.
    pub fn backend(mut self, backend: DistributedBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Builder-style setter for master address.
    pub fn master_addr(mut self, addr: impl Into<String>) -> Self {
        self.master_addr = addr.into();
        self
    }

    /// Builder-style setter for master port.
    pub fn master_port(mut self, port: u16) -> Self {
        self.master_port = port;
        self
    }

    /// Builder-style setter for timeout.
    pub fn timeout_ms(mut self, timeout: u64) -> Self {
        self.timeout_ms = timeout;
        self
    }

    /// Builder-style setter for gradient compression.
    pub fn gradient_compression(mut self, enabled: bool) -> Self {
        self.gradient_compression = enabled;
        self
    }

    /// Builder-style setter for sync mode.
    pub fn sync_mode(mut self, mode: SyncMode) -> Self {
        self.sync_mode = mode;
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<()> {
        if self.world_size == 0 {
            return Err(OctaneError::InvalidConfig(
                "world_size must be positive".into(),
            ));
        }
        if self.rank >= self.world_size {
            return Err(OctaneError::InvalidConfig(format!(
                "rank {} must be less than world_size {}",
                self.rank, self.world_size
            )));
        }
        if self.master_port == 0 {
            return Err(OctaneError::InvalidConfig(
                "master_port must be non-zero".into(),
            ));
        }
        Ok(())
    }

    /// Get the master endpoint URL.
    pub fn master_endpoint(&self) -> String {
        format!("{}:{}", self.master_addr, self.master_port)
    }

    /// Check if this is a single-worker setup (no distribution).
    pub fn is_single_worker(&self) -> bool {
        self.world_size == 1
    }
}

/// Message types for worker communication.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WorkerMessage {
    /// Gradients from a worker.
    Gradients {
        /// Worker rank.
        rank: usize,
        /// Gradient data keyed by parameter name.
        gradients: HashMap<String, Vec<f32>>,
        /// Step number for ordering.
        step: usize,
    },
    /// Aggregated parameters from master.
    Parameters {
        /// Parameter data keyed by name.
        parameters: HashMap<String, Vec<f32>>,
        /// Global step number.
        global_step: usize,
    },
    /// Worker registration.
    Register {
        /// Worker rank.
        rank: usize,
        /// Worker address for callbacks.
        addr: String,
    },
    /// Acknowledgment.
    Ack {
        /// Acknowledged step.
        step: usize,
    },
    /// Shutdown signal.
    Shutdown,
}

/// Worker pool for distributed environment rollouts.
///
/// Manages a pool of workers that can asynchronously collect rollouts
/// from environments. Each worker runs in its own thread/task.
#[derive(Debug)]
pub struct WorkerPool {
    /// Configuration.
    config: DistributedConfig,

    /// Worker states.
    workers: Vec<WorkerState>,

    /// Device for tensor operations.
    device: Device,
}

/// State of a single worker.
#[derive(Debug, Clone)]
pub struct WorkerState {
    /// Worker rank.
    pub rank: usize,
    /// Whether worker is active.
    pub active: bool,
    /// Last heartbeat timestamp.
    pub last_heartbeat: std::time::Instant,
    /// Number of completed rollouts.
    pub rollouts_completed: usize,
    /// Total timesteps collected.
    pub timesteps_collected: usize,
}

impl WorkerPool {
    /// Create a new worker pool.
    pub fn new(config: DistributedConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let workers = (0..config.world_size)
            .map(|rank| WorkerState {
                rank,
                active: rank == config.rank, // Only this worker is initially active
                last_heartbeat: std::time::Instant::now(),
                rollouts_completed: 0,
                timesteps_collected: 0,
            })
            .collect();

        Ok(Self {
            config,
            workers,
            device,
        })
    }

    /// Get the number of workers.
    pub fn num_workers(&self) -> usize {
        self.config.world_size
    }

    /// Get this worker's rank.
    pub fn rank(&self) -> usize {
        self.config.rank
    }

    /// Check if this is the master worker.
    pub fn is_master(&self) -> bool {
        self.config.is_master
    }

    /// Get the current device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get worker states.
    pub fn workers(&self) -> &[WorkerState] {
        &self.workers
    }

    /// Update worker heartbeat.
    pub fn update_heartbeat(&mut self, rank: usize) {
        if let Some(worker) = self.workers.get_mut(rank) {
            worker.last_heartbeat = std::time::Instant::now();
            worker.active = true;
        }
    }

    /// Record completed rollout for a worker.
    pub fn record_rollout(&mut self, rank: usize, timesteps: usize) {
        if let Some(worker) = self.workers.get_mut(rank) {
            worker.rollouts_completed += 1;
            worker.timesteps_collected += timesteps;
        }
    }

    /// Get total timesteps collected across all workers.
    pub fn total_timesteps(&self) -> usize {
        self.workers.iter().map(|w| w.timesteps_collected).sum()
    }

    /// Get number of active workers.
    pub fn active_workers(&self) -> usize {
        let timeout = std::time::Duration::from_secs(60);
        self.workers
            .iter()
            .filter(|w| w.active && w.last_heartbeat.elapsed() < timeout)
            .count()
    }

    /// Check if all workers are healthy.
    pub fn all_healthy(&self) -> bool {
        self.active_workers() == self.config.world_size
    }
}

/// Gradient aggregator for distributed SGD.
///
/// Collects gradients from multiple workers and aggregates them
/// (typically by averaging) before applying to the model.
#[derive(Debug)]
pub struct GradientAggregator {
    /// Configuration.
    config: DistributedConfig,

    /// Accumulated gradients per parameter.
    accumulated_gradients: HashMap<String, Vec<f32>>,

    /// Number of gradients accumulated.
    num_accumulated: usize,

    /// Current global step.
    global_step: usize,

    /// Gradient norms for monitoring.
    gradient_norms: Vec<f32>,
}

impl GradientAggregator {
    /// Create a new gradient aggregator.
    pub fn new(config: DistributedConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            config,
            accumulated_gradients: HashMap::new(),
            num_accumulated: 0,
            global_step: 0,
            gradient_norms: Vec::new(),
        })
    }

    /// Add gradients from a worker.
    pub fn add_gradients(&mut self, gradients: HashMap<String, Vec<f32>>) -> Result<()> {
        for (name, grad) in gradients {
            // Optionally compress gradients
            let grad = if self.config.gradient_compression {
                self.compress_gradient(&grad)
            } else {
                grad
            };

            // Accumulate
            self.accumulated_gradients
                .entry(name)
                .and_modify(|acc| {
                    for (a, g) in acc.iter_mut().zip(grad.iter()) {
                        *a += g;
                    }
                })
                .or_insert(grad);
        }

        self.num_accumulated += 1;
        Ok(())
    }

    /// Check if ready to aggregate (all workers have contributed).
    pub fn is_ready(&self) -> bool {
        match self.config.sync_mode {
            SyncMode::Synchronous => self.num_accumulated >= self.config.world_size,
            SyncMode::Asynchronous => self.num_accumulated > 0,
            SyncMode::BoundedStaleness(max_stale) => {
                self.num_accumulated >= self.config.world_size.saturating_sub(max_stale)
            }
        }
    }

    /// Aggregate and return averaged gradients.
    pub fn aggregate(&mut self) -> Result<HashMap<String, Vec<f32>>> {
        if self.num_accumulated == 0 {
            return Err(OctaneError::Buffer("No gradients to aggregate".into()));
        }

        let scale = 1.0 / self.num_accumulated as f32;
        let mut result = HashMap::new();

        for (name, grad) in self.accumulated_gradients.drain() {
            let averaged: Vec<f32> = grad.iter().map(|g| g * scale).collect();

            // Track gradient norm
            let norm: f32 = averaged.iter().map(|g| g * g).sum::<f32>().sqrt();
            self.gradient_norms.push(norm);

            result.insert(name, averaged);
        }

        self.num_accumulated = 0;
        self.global_step += 1;

        Ok(result)
    }

    /// Compress gradient using sparsification (keep top-k by magnitude).
    fn compress_gradient(&self, gradient: &[f32]) -> Vec<f32> {
        let threshold = self.config.compression_threshold;
        gradient
            .iter()
            .map(|&g| if g.abs() > threshold { g } else { 0.0 })
            .collect()
    }

    /// Get the current global step.
    pub fn global_step(&self) -> usize {
        self.global_step
    }

    /// Get the number of accumulated gradients.
    pub fn num_accumulated(&self) -> usize {
        self.num_accumulated
    }

    /// Get mean gradient norm (for monitoring).
    pub fn mean_gradient_norm(&self) -> f32 {
        if self.gradient_norms.is_empty() {
            0.0
        } else {
            self.gradient_norms.iter().sum::<f32>() / self.gradient_norms.len() as f32
        }
    }

    /// Reset accumulated gradients without incrementing step.
    pub fn reset(&mut self) {
        self.accumulated_gradients.clear();
        self.num_accumulated = 0;
    }

    /// Clear gradient norm history.
    pub fn clear_history(&mut self) {
        self.gradient_norms.clear();
    }
}

/// Distributed training coordinator.
///
/// Coordinates distributed training by managing worker registration,
/// gradient collection, and parameter distribution.
#[derive(Debug)]
pub struct DistributedCoordinator {
    /// Configuration.
    config: DistributedConfig,

    /// Gradient aggregator.
    aggregator: GradientAggregator,

    /// Worker pool.
    pool: WorkerPool,

    /// Current parameters (master only).
    parameters: HashMap<String, Vec<f32>>,

    /// Training statistics.
    stats: DistributedStats,
}

/// Statistics for distributed training.
#[derive(Debug, Clone, Default)]
pub struct DistributedStats {
    /// Total gradient updates.
    pub total_updates: usize,
    /// Total communication rounds.
    pub comm_rounds: usize,
    /// Average sync time in milliseconds.
    pub avg_sync_time_ms: f64,
    /// Number of stale updates (async mode).
    pub stale_updates: usize,
    /// Total bytes transferred.
    pub bytes_transferred: usize,
}

impl DistributedCoordinator {
    /// Create a new coordinator.
    pub fn new(config: DistributedConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let aggregator = GradientAggregator::new(config.clone())?;
        let pool = WorkerPool::new(config.clone(), device)?;

        Ok(Self {
            config,
            aggregator,
            pool,
            parameters: HashMap::new(),
            stats: DistributedStats::default(),
        })
    }

    /// Initialize with model parameters.
    pub fn init_parameters(&mut self, parameters: HashMap<String, Vec<f32>>) {
        self.parameters = parameters;
    }

    /// Submit gradients from local computation.
    pub fn submit_gradients(&mut self, gradients: HashMap<String, Vec<f32>>) -> Result<()> {
        self.aggregator.add_gradients(gradients)?;
        self.pool.update_heartbeat(self.config.rank);
        Ok(())
    }

    /// Check if ready to perform parameter update.
    pub fn ready_for_update(&self) -> bool {
        self.aggregator.is_ready()
    }

    /// Perform parameter update (returns updated parameters).
    pub fn update(&mut self, learning_rate: f32) -> Result<HashMap<String, Vec<f32>>> {
        let gradients = self.aggregator.aggregate()?;

        // Apply gradients to parameters
        for (name, grad) in &gradients {
            if let Some(param) = self.parameters.get_mut(name) {
                for (p, g) in param.iter_mut().zip(grad.iter()) {
                    *p -= learning_rate * g;
                }
            }
        }

        self.stats.total_updates += 1;
        self.stats.comm_rounds += 1;

        Ok(self.parameters.clone())
    }

    /// Get current parameters.
    pub fn parameters(&self) -> &HashMap<String, Vec<f32>> {
        &self.parameters
    }

    /// Get statistics.
    pub fn stats(&self) -> &DistributedStats {
        &self.stats
    }

    /// Get the global step.
    pub fn global_step(&self) -> usize {
        self.aggregator.global_step()
    }

    /// Get worker pool reference.
    pub fn pool(&self) -> &WorkerPool {
        &self.pool
    }

    /// Get mutable worker pool reference.
    pub fn pool_mut(&mut self) -> &mut WorkerPool {
        &mut self.pool
    }
}

/// All-reduce operation types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum all values.
    Sum,
    /// Average all values.
    Mean,
    /// Take maximum value.
    Max,
    /// Take minimum value.
    Min,
}

/// Perform all-reduce operation on a tensor (single-worker fallback).
///
/// In a full distributed implementation, this would communicate with other workers.
/// For single-worker mode, this is a no-op that returns the input.
pub fn all_reduce(data: &mut [f32], op: ReduceOp, config: &DistributedConfig) -> Result<()> {
    if config.is_single_worker() {
        // No-op for single worker
        return Ok(());
    }

    // For multi-worker, this would need actual communication.
    // This is a placeholder that simulates the operation.
    match op {
        ReduceOp::Sum | ReduceOp::Mean => {
            // In actual distributed setup, sum across workers
            // Then divide by world_size for mean
            if op == ReduceOp::Mean {
                let scale = 1.0 / config.world_size as f32;
                for d in data.iter_mut() {
                    *d *= scale;
                }
            }
        }
        ReduceOp::Max | ReduceOp::Min => {
            // In actual distributed setup, find global max/min
        }
    }

    Ok(())
}

/// Broadcast tensor from root to all workers (single-worker fallback).
pub fn broadcast(_data: &mut [f32], root: usize, config: &DistributedConfig) -> Result<()> {
    if config.is_single_worker() {
        return Ok(());
    }

    if config.rank != root {
        // Would receive data from root
    }
    // Root worker keeps its data

    Ok(())
}

/// Barrier synchronization across all workers.
pub fn barrier(config: &DistributedConfig) -> Result<()> {
    if config.is_single_worker() {
        return Ok(());
    }

    // In actual distributed setup, this would block until all workers reach this point
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distributed_config_defaults() {
        let config = DistributedConfig::default();
        assert_eq!(config.world_size, 1);
        assert_eq!(config.rank, 0);
        assert!(config.is_master);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_distributed_config_new() {
        let config = DistributedConfig::new(4, 2);
        assert_eq!(config.world_size, 4);
        assert_eq!(config.rank, 2);
        assert!(!config.is_master);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_distributed_config_validation() {
        let invalid = DistributedConfig::new(4, 5);
        assert!(invalid.validate().is_err());

        let zero_world = DistributedConfig {
            world_size: 0,
            ..Default::default()
        };
        assert!(zero_world.validate().is_err());
    }

    #[test]
    fn test_gradient_aggregator_sync() {
        let config = DistributedConfig::new(2, 0);
        let mut aggregator = GradientAggregator::new(config).unwrap();

        // Add gradients from worker 0
        let mut grads1 = HashMap::new();
        grads1.insert("layer1".to_string(), vec![1.0, 2.0, 3.0]);
        aggregator.add_gradients(grads1).unwrap();
        assert!(!aggregator.is_ready());

        // Add gradients from worker 1
        let mut grads2 = HashMap::new();
        grads2.insert("layer1".to_string(), vec![3.0, 4.0, 5.0]);
        aggregator.add_gradients(grads2).unwrap();
        assert!(aggregator.is_ready());

        // Aggregate
        let result = aggregator.aggregate().unwrap();
        let averaged = result.get("layer1").unwrap();

        // Should be average: [2.0, 3.0, 4.0]
        assert!((averaged[0] - 2.0).abs() < 1e-6);
        assert!((averaged[1] - 3.0).abs() < 1e-6);
        assert!((averaged[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_gradient_aggregator_async() {
        let config = DistributedConfig::new(4, 0).sync_mode(SyncMode::Asynchronous);
        let mut aggregator = GradientAggregator::new(config).unwrap();

        // Single gradient is enough in async mode
        let mut grads = HashMap::new();
        grads.insert("layer1".to_string(), vec![1.0, 2.0]);
        aggregator.add_gradients(grads).unwrap();

        assert!(aggregator.is_ready());
    }

    #[test]
    fn test_worker_pool() {
        let config = DistributedConfig::new(4, 0);
        let mut pool = WorkerPool::new(config, Device::Cpu).unwrap();

        assert_eq!(pool.num_workers(), 4);
        assert_eq!(pool.rank(), 0);
        assert!(pool.is_master());

        pool.record_rollout(0, 1000);
        assert_eq!(pool.total_timesteps(), 1000);
    }

    #[test]
    fn test_distributed_coordinator() {
        let config = DistributedConfig::new(1, 0);
        let mut coord = DistributedCoordinator::new(config, Device::Cpu).unwrap();

        // Initialize parameters
        let mut params = HashMap::new();
        params.insert("w".to_string(), vec![1.0, 2.0, 3.0]);
        coord.init_parameters(params);

        // Submit gradients
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), vec![0.1, 0.2, 0.3]);
        coord.submit_gradients(grads).unwrap();

        // Update
        let new_params = coord.update(1.0).unwrap();
        let w = new_params.get("w").unwrap();

        // w -= lr * grad = [1.0 - 0.1, 2.0 - 0.2, 3.0 - 0.3]
        assert!((w[0] - 0.9).abs() < 1e-6);
        assert!((w[1] - 1.8).abs() < 1e-6);
        assert!((w[2] - 2.7).abs() < 1e-6);
    }

    #[test]
    fn test_all_reduce_single_worker() {
        let config = DistributedConfig::default();
        let mut data = vec![1.0, 2.0, 3.0];

        all_reduce(&mut data, ReduceOp::Sum, &config).unwrap();

        // Single worker: no-op
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
    }
}
