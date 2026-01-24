//! Traits for RL algorithms.

use crate::algorithms::metrics::TrainMetrics;
use crate::core::{Device, Result};
use std::path::Path;

/// Core trait for reinforcement learning algorithms.
///
/// This trait defines the common interface for all RL algorithms,
/// enabling generic training loops and algorithm comparison.
pub trait RLAlgorithm {
    /// Perform a single training step (collect + update).
    ///
    /// Returns metrics from the training step.
    fn train_step(&mut self) -> Result<TrainMetrics>;

    /// Save the model to disk.
    fn save(&self, path: &Path) -> Result<()>;

    /// Load the model from disk.
    fn load(&mut self, path: &Path) -> Result<()>;

    /// Get the device being used.
    fn device(&self) -> &Device;

    /// Get total timesteps trained.
    fn total_timesteps(&self) -> usize;
}

/// Trait for policy inference.
pub trait Policy: Send + Sync {
    /// Predict action for a given observation.
    ///
    /// # Arguments
    /// * `obs` - Observation tensor [batch_size, obs_dim]
    /// * `deterministic` - Whether to use deterministic (argmax/mean) or stochastic actions
    ///
    /// # Returns
    /// Action tensor [batch_size, act_dim]
    fn predict(
        &self,
        obs: &candle_core::Tensor,
        deterministic: bool,
    ) -> Result<candle_core::Tensor>;

    /// Get action distribution parameters for a given observation.
    ///
    /// For discrete actions, returns logits.
    /// For continuous actions, returns (mean, log_std).
    fn get_distribution(&self, obs: &candle_core::Tensor) -> Result<PolicyDistribution>;
}

/// Policy distribution parameters.
#[derive(Debug)]
pub enum PolicyDistribution {
    /// Categorical distribution for discrete actions.
    Categorical {
        /// Logits [batch_size, num_actions]
        logits: candle_core::Tensor,
    },
    /// Diagonal Gaussian for continuous actions.
    DiagGaussian {
        /// Mean [batch_size, act_dim]
        mean: candle_core::Tensor,
        /// Log standard deviation [act_dim] or [batch_size, act_dim]
        log_std: candle_core::Tensor,
    },
}

/// Trait for value function estimation.
pub trait ValueFunction: Send + Sync {
    /// Estimate value of observations.
    ///
    /// # Arguments
    /// * `obs` - Observation tensor [batch_size, obs_dim]
    ///
    /// # Returns
    /// Value estimates [batch_size]
    fn value(&self, obs: &candle_core::Tensor) -> Result<candle_core::Tensor>;
}

/// Trait for actor-critic networks.
pub trait ActorCritic: Policy + ValueFunction {
    /// Forward pass returning both policy distribution and value.
    fn forward(
        &self,
        obs: &candle_core::Tensor,
    ) -> Result<(PolicyDistribution, candle_core::Tensor)>;

    /// Evaluate log probability and entropy for given observations and actions.
    fn evaluate(
        &self,
        obs: &candle_core::Tensor,
        actions: &candle_core::Tensor,
    ) -> Result<(
        candle_core::Tensor,
        candle_core::Tensor,
        candle_core::Tensor,
    )>;
}

/// Callback trait for training hooks.
pub trait TrainCallback {
    /// Called at the start of training.
    fn on_train_start(&mut self) {}

    /// Called at the end of training.
    fn on_train_end(&mut self) {}

    /// Called after each rollout collection.
    fn on_rollout_end(&mut self, _metrics: &TrainMetrics) {}

    /// Called after each policy update.
    fn on_step(&mut self, _metrics: &TrainMetrics) {}

    /// Called when an episode ends.
    fn on_episode_end(&mut self, _episode_reward: f32, _episode_length: usize) {}
}

/// No-op callback implementation.
#[derive(Debug, Default)]
pub struct NoOpCallback;

impl TrainCallback for NoOpCallback {}

/// Callback that logs metrics to console.
#[derive(Debug)]
pub struct LoggingCallback {
    /// Log every N steps.
    log_interval: usize,
    /// Steps since last log.
    steps_since_log: usize,
}

impl LoggingCallback {
    /// Create a new logging callback.
    pub fn new(log_interval: usize) -> Self {
        Self {
            log_interval,
            steps_since_log: 0,
        }
    }
}

impl TrainCallback for LoggingCallback {
    fn on_step(&mut self, metrics: &TrainMetrics) {
        self.steps_since_log += 1;
        if self.steps_since_log >= self.log_interval {
            tracing::info!("{}", metrics);
            self.steps_since_log = 0;
        }
    }

    fn on_train_end(&mut self) {
        tracing::info!("Training completed!");
    }
}

/// Callback for early stopping based on reward.
#[derive(Debug)]
pub struct EarlyStoppingCallback {
    /// Target reward to stop training.
    target_reward: f32,
    /// Number of consecutive episodes above target.
    patience: usize,
    /// Current consecutive count.
    consecutive_count: usize,
    /// Whether stopping condition was met.
    should_stop: bool,
}

impl EarlyStoppingCallback {
    /// Create a new early stopping callback.
    pub fn new(target_reward: f32, patience: usize) -> Self {
        Self {
            target_reward,
            patience,
            consecutive_count: 0,
            should_stop: false,
        }
    }

    /// Check if training should stop.
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }
}

impl TrainCallback for EarlyStoppingCallback {
    fn on_step(&mut self, metrics: &TrainMetrics) {
        if metrics.mean_reward >= self.target_reward {
            self.consecutive_count += 1;
            if self.consecutive_count >= self.patience {
                self.should_stop = true;
                tracing::info!(
                    "Early stopping: target reward {:.2} reached for {} consecutive steps",
                    self.target_reward,
                    self.patience
                );
            }
        } else {
            self.consecutive_count = 0;
        }
    }
}

/// Composite callback that runs multiple callbacks.
pub struct CallbackList {
    callbacks: Vec<Box<dyn TrainCallback>>,
}

impl CallbackList {
    /// Create a new callback list.
    pub fn new() -> Self {
        Self {
            callbacks: Vec::new(),
        }
    }

    /// Add a callback to the list.
    pub fn add<C: TrainCallback + 'static>(&mut self, callback: C) {
        self.callbacks.push(Box::new(callback));
    }
}

impl Default for CallbackList {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainCallback for CallbackList {
    fn on_train_start(&mut self) {
        for cb in &mut self.callbacks {
            cb.on_train_start();
        }
    }

    fn on_train_end(&mut self) {
        for cb in &mut self.callbacks {
            cb.on_train_end();
        }
    }

    fn on_rollout_end(&mut self, metrics: &TrainMetrics) {
        for cb in &mut self.callbacks {
            cb.on_rollout_end(metrics);
        }
    }

    fn on_step(&mut self, metrics: &TrainMetrics) {
        for cb in &mut self.callbacks {
            cb.on_step(metrics);
        }
    }

    fn on_episode_end(&mut self, episode_reward: f32, episode_length: usize) {
        for cb in &mut self.callbacks {
            cb.on_episode_end(episode_reward, episode_length);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping() {
        let mut callback = EarlyStoppingCallback::new(100.0, 3);

        let metrics_low = TrainMetrics {
            mean_reward: 50.0,
            ..Default::default()
        };
        let metrics_high = TrainMetrics {
            mean_reward: 120.0,
            ..Default::default()
        };

        callback.on_step(&metrics_low);
        assert!(!callback.should_stop());

        callback.on_step(&metrics_high);
        callback.on_step(&metrics_high);
        assert!(!callback.should_stop());

        callback.on_step(&metrics_high);
        assert!(callback.should_stop());
    }

    #[test]
    fn test_callback_list() {
        let mut list = CallbackList::new();
        list.add(NoOpCallback);
        list.add(LoggingCallback::new(10));

        let metrics = TrainMetrics::default();
        list.on_step(&metrics);
        list.on_train_end();
    }
}
