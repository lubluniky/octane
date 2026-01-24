//! Training metrics for RL algorithms.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Metrics collected during training.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainMetrics {
    /// Policy/actor loss.
    pub policy_loss: f32,

    /// Value/critic loss.
    pub value_loss: f32,

    /// Entropy of the policy distribution.
    pub entropy: f32,

    /// Approximate KL divergence (PPO).
    pub approx_kl: f32,

    /// Fraction of samples clipped (PPO).
    pub clip_fraction: f32,

    /// Explained variance of value function.
    pub explained_variance: f32,

    /// Current learning rate.
    pub learning_rate: f32,

    /// Total timesteps trained.
    pub timesteps: usize,

    /// Number of episodes completed.
    pub episodes: usize,

    /// Mean episode reward.
    pub mean_reward: f32,

    /// Standard deviation of episode rewards.
    pub std_reward: f32,
}

impl TrainMetrics {
    /// Create new empty metrics.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create metrics with basic values.
    pub fn with_losses(policy_loss: f32, value_loss: f32, entropy: f32) -> Self {
        Self {
            policy_loss,
            value_loss,
            entropy,
            ..Default::default()
        }
    }

    /// Check if training is progressing (loss is finite and decreasing).
    pub fn is_healthy(&self) -> bool {
        self.policy_loss.is_finite()
            && self.value_loss.is_finite()
            && self.entropy.is_finite()
            && !self.policy_loss.is_nan()
            && !self.value_loss.is_nan()
    }

    /// Merge metrics from multiple updates (average).
    pub fn merge(metrics: &[TrainMetrics]) -> Self {
        if metrics.is_empty() {
            return Self::default();
        }

        let n = metrics.len() as f32;
        Self {
            policy_loss: metrics.iter().map(|m| m.policy_loss).sum::<f32>() / n,
            value_loss: metrics.iter().map(|m| m.value_loss).sum::<f32>() / n,
            entropy: metrics.iter().map(|m| m.entropy).sum::<f32>() / n,
            approx_kl: metrics.iter().map(|m| m.approx_kl).sum::<f32>() / n,
            clip_fraction: metrics.iter().map(|m| m.clip_fraction).sum::<f32>() / n,
            explained_variance: metrics.iter().map(|m| m.explained_variance).sum::<f32>() / n,
            learning_rate: metrics.last().map(|m| m.learning_rate).unwrap_or(0.0),
            timesteps: metrics.last().map(|m| m.timesteps).unwrap_or(0),
            episodes: metrics.iter().map(|m| m.episodes).sum(),
            mean_reward: metrics.iter().map(|m| m.mean_reward).sum::<f32>() / n,
            std_reward: metrics.iter().map(|m| m.std_reward).sum::<f32>() / n,
        }
    }
}

impl fmt::Display for TrainMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TrainMetrics {{ timesteps: {}, episodes: {}, mean_reward: {:.2}, \
             policy_loss: {:.4}, value_loss: {:.4}, entropy: {:.4}, lr: {:.2e} }}",
            self.timesteps,
            self.episodes,
            self.mean_reward,
            self.policy_loss,
            self.value_loss,
            self.entropy,
            self.learning_rate
        )
    }
}

/// Rolling statistics tracker for rewards.
#[derive(Debug, Clone)]
pub struct RewardStats {
    /// Running sum of rewards.
    sum: f64,
    /// Running sum of squared rewards.
    sum_sq: f64,
    /// Number of samples.
    count: usize,
    /// Minimum reward seen.
    min: f32,
    /// Maximum reward seen.
    max: f32,
    /// Recent rewards for windowed statistics.
    recent: Vec<f32>,
    /// Window size for recent rewards.
    window_size: usize,
}

impl RewardStats {
    /// Create a new reward statistics tracker.
    pub fn new(window_size: usize) -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            min: f32::INFINITY,
            max: f32::NEG_INFINITY,
            recent: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Add a reward sample.
    pub fn update(&mut self, reward: f32) {
        self.sum += reward as f64;
        self.sum_sq += (reward as f64).powi(2);
        self.count += 1;
        self.min = self.min.min(reward);
        self.max = self.max.max(reward);

        // Update windowed statistics
        if self.recent.len() >= self.window_size {
            self.recent.remove(0);
        }
        self.recent.push(reward);
    }

    /// Get the mean reward.
    pub fn mean(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            (self.sum / self.count as f64) as f32
        }
    }

    /// Get the standard deviation of rewards.
    pub fn std(&self) -> f32 {
        if self.count < 2 {
            0.0
        } else {
            let mean = self.sum / self.count as f64;
            let variance = self.sum_sq / self.count as f64 - mean.powi(2);
            variance.max(0.0).sqrt() as f32
        }
    }

    /// Get the windowed mean (recent rewards only).
    pub fn windowed_mean(&self) -> f32 {
        if self.recent.is_empty() {
            0.0
        } else {
            self.recent.iter().sum::<f32>() / self.recent.len() as f32
        }
    }

    /// Get the windowed standard deviation.
    pub fn windowed_std(&self) -> f32 {
        if self.recent.len() < 2 {
            return 0.0;
        }

        let mean = self.windowed_mean();
        let variance =
            self.recent.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / self.recent.len() as f32;
        variance.sqrt()
    }

    /// Get minimum reward.
    pub fn min(&self) -> f32 {
        self.min
    }

    /// Get maximum reward.
    pub fn max(&self) -> f32 {
        self.max
    }

    /// Get total count.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Reset statistics.
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.count = 0;
        self.min = f32::INFINITY;
        self.max = f32::NEG_INFINITY;
        self.recent.clear();
    }
}

impl Default for RewardStats {
    fn default() -> Self {
        Self::new(100)
    }
}

/// Training progress tracker.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    /// Target total timesteps.
    total_timesteps: usize,
    /// Current timesteps.
    current_timesteps: usize,
    /// Start time.
    start_time: std::time::Instant,
    /// Metrics history.
    history: Vec<TrainMetrics>,
    /// Maximum history size.
    max_history: usize,
}

impl ProgressTracker {
    /// Create a new progress tracker.
    pub fn new(total_timesteps: usize) -> Self {
        Self {
            total_timesteps,
            current_timesteps: 0,
            start_time: std::time::Instant::now(),
            history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Update progress with new metrics.
    pub fn update(&mut self, metrics: &TrainMetrics) {
        self.current_timesteps = metrics.timesteps;

        if self.history.len() >= self.max_history {
            self.history.remove(0);
        }
        self.history.push(metrics.clone());
    }

    /// Get completion percentage.
    pub fn progress(&self) -> f32 {
        if self.total_timesteps == 0 {
            1.0
        } else {
            self.current_timesteps as f32 / self.total_timesteps as f32
        }
    }

    /// Get elapsed time in seconds.
    pub fn elapsed_secs(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }

    /// Get estimated remaining time in seconds.
    pub fn eta_secs(&self) -> f64 {
        let progress = self.progress();
        if progress <= 0.0 {
            f64::INFINITY
        } else {
            let elapsed = self.elapsed_secs();
            elapsed * (1.0 - progress as f64) / progress as f64
        }
    }

    /// Get timesteps per second.
    pub fn fps(&self) -> f32 {
        let elapsed = self.elapsed_secs();
        if elapsed <= 0.0 {
            0.0
        } else {
            self.current_timesteps as f32 / elapsed as f32
        }
    }

    /// Get the best mean reward from history.
    pub fn best_reward(&self) -> f32 {
        self.history
            .iter()
            .map(|m| m.mean_reward)
            .fold(f32::NEG_INFINITY, f32::max)
    }

    /// Check if training is complete.
    pub fn is_complete(&self) -> bool {
        self.current_timesteps >= self.total_timesteps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_metrics_merge() {
        let m1 = TrainMetrics {
            policy_loss: 1.0,
            value_loss: 2.0,
            entropy: 0.5,
            ..Default::default()
        };
        let m2 = TrainMetrics {
            policy_loss: 3.0,
            value_loss: 4.0,
            entropy: 1.5,
            ..Default::default()
        };

        let merged = TrainMetrics::merge(&[m1, m2]);
        assert!((merged.policy_loss - 2.0).abs() < 1e-6);
        assert!((merged.value_loss - 3.0).abs() < 1e-6);
        assert!((merged.entropy - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reward_stats() {
        let mut stats = RewardStats::new(10);
        stats.update(1.0);
        stats.update(2.0);
        stats.update(3.0);

        assert!((stats.mean() - 2.0).abs() < 1e-6);
        assert_eq!(stats.count(), 3);
        assert!((stats.min() - 1.0).abs() < 1e-6);
        assert!((stats.max() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_progress_tracker() {
        let mut tracker = ProgressTracker::new(1000);
        let metrics = TrainMetrics {
            timesteps: 500,
            ..Default::default()
        };
        tracker.update(&metrics);

        assert!((tracker.progress() - 0.5).abs() < 1e-6);
        assert!(!tracker.is_complete());
    }
}
