//! Probability distributions for reinforcement learning.
//!
//! This module provides probability distributions for sampling actions
//! and computing log probabilities during policy optimization.
//!
//! ## Supported Distributions
//!
//! - [`Categorical`]: For discrete action spaces (DQN, discrete PPO)
//! - [`DiagGaussian`]: For continuous action spaces (continuous PPO, SAC)
//!
//! ## Example
//!
//! ```ignore
//! use rocket_rs::distributions::{Distribution, Categorical, DiagGaussian};
//! use candle_core::Tensor;
//!
//! // Discrete actions with Categorical
//! let logits = Tensor::randn(0.0, 1.0, &[32, 4], &device)?; // batch=32, actions=4
//! let categorical = Categorical::new(logits)?;
//! let actions = categorical.sample()?;
//! let log_probs = categorical.log_prob(&actions)?;
//!
//! // Continuous actions with DiagGaussian
//! let mean = Tensor::zeros(&[32, 2], DType::F32, &device)?;
//! let log_std = Tensor::zeros(&[32, 2], DType::F32, &device)?;
//! let gaussian = DiagGaussian::new(mean, log_std)?;
//! let actions = gaussian.sample()?;
//! let log_probs = gaussian.log_prob(&actions)?;
//! ```

mod categorical;
mod gaussian;

pub use categorical::Categorical;
pub use gaussian::{DiagGaussian, SquashedGaussian};

use crate::core::Result;
use candle_core::Tensor;

/// Trait for probability distributions used in RL.
///
/// All distributions support batched operations, where the first dimension
/// is typically the batch size (number of parallel environments).
pub trait Distribution {
    /// Sample actions from the distribution.
    ///
    /// Returns a tensor of sampled actions with shape depending on the
    /// distribution type:
    /// - Categorical: `[batch_size]` (discrete action indices)
    /// - DiagGaussian: `[batch_size, action_dim]` (continuous action vectors)
    fn sample(&self) -> Result<Tensor>;

    /// Compute log probability of given actions under this distribution.
    ///
    /// # Arguments
    /// * `actions` - Tensor of actions to evaluate
    ///
    /// # Returns
    /// Log probabilities with shape `[batch_size]`
    fn log_prob(&self, actions: &Tensor) -> Result<Tensor>;

    /// Compute the entropy of the distribution.
    ///
    /// Entropy measures the randomness/uncertainty of the distribution.
    /// Higher entropy = more exploration.
    ///
    /// # Returns
    /// Entropy values with shape `[batch_size]`
    fn entropy(&self) -> Result<Tensor>;

    /// Get the deterministic action (mode of the distribution).
    ///
    /// Used during evaluation/inference when we want deterministic behavior:
    /// - Categorical: argmax of logits
    /// - DiagGaussian: the mean
    ///
    /// # Returns
    /// Deterministic actions with same shape as `sample()`
    fn mode(&self) -> Result<Tensor>;
}

/// Constants for numerical stability.
pub mod constants {
    /// Small epsilon for numerical stability in log operations.
    pub const LOG_STD_MIN: f32 = -20.0;
    /// Maximum log standard deviation to prevent explosion.
    pub const LOG_STD_MAX: f32 = 2.0;
    /// Small epsilon to prevent division by zero.
    pub const EPSILON: f32 = 1e-8;
    /// Log of 2*pi for Gaussian log probability computation.
    pub const LOG_2PI: f64 = 1.8378770664093453; // ln(2 * pi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_categorical_basic() -> Result<()> {
        let device = Device::Cpu;
        // Batch of 4, with 3 possible actions
        let logits = Tensor::from_slice(
            &[
                1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 2.0, 2.0, 2.0,
            ],
            &[4, 3],
            &device,
        )?;

        let dist = Categorical::new(logits)?;

        // Sample should return [batch_size] shape
        let samples = dist.sample()?;
        assert_eq!(samples.dims(), &[4]);

        // Log prob should return [batch_size] shape
        let log_probs = dist.log_prob(&samples)?;
        assert_eq!(log_probs.dims(), &[4]);

        // Entropy should return [batch_size] shape
        let entropy = dist.entropy()?;
        assert_eq!(entropy.dims(), &[4]);

        // Mode should return [batch_size] shape
        let mode = dist.mode()?;
        assert_eq!(mode.dims(), &[4]);

        Ok(())
    }

    #[test]
    fn test_gaussian_basic() -> Result<()> {
        let device = Device::Cpu;
        // Batch of 4, with 2-dim continuous actions
        let mean = Tensor::zeros(&[4, 2], DType::F32, &device)?;
        let log_std = Tensor::zeros(&[4, 2], DType::F32, &device)?;

        let dist = DiagGaussian::new(mean, log_std)?;

        // Sample should return [batch_size, action_dim] shape
        let samples = dist.sample()?;
        assert_eq!(samples.dims(), &[4, 2]);

        // Log prob should return [batch_size] shape
        let log_probs = dist.log_prob(&samples)?;
        assert_eq!(log_probs.dims(), &[4]);

        // Entropy should return [batch_size] shape
        let entropy = dist.entropy()?;
        assert_eq!(entropy.dims(), &[4]);

        // Mode should return [batch_size, action_dim] shape
        let mode = dist.mode()?;
        assert_eq!(mode.dims(), &[4, 2]);

        Ok(())
    }

    #[test]
    fn test_categorical_log_prob_values() -> Result<()> {
        let device = Device::Cpu;
        // Simple case: uniform distribution over 2 actions
        let logits = Tensor::zeros(&[1, 2], DType::F32, &device)?;
        let dist = Categorical::new(logits)?;

        let action = Tensor::from_slice(&[0u32], &[1], &device)?;
        let log_prob = dist.log_prob(&action)?;
        let log_prob_val: f32 = log_prob.to_vec1()?[0];

        // For uniform over 2 actions, log_prob should be -ln(2) = -0.693...
        assert!((log_prob_val - (-0.693)).abs() < 0.01);

        Ok(())
    }

    #[test]
    fn test_gaussian_log_prob_values() -> Result<()> {
        let device = Device::Cpu;
        // Standard normal: mean=0, std=1
        let mean = Tensor::zeros(&[1, 1], DType::F32, &device)?;
        let log_std = Tensor::zeros(&[1, 1], DType::F32, &device)?; // log(1) = 0

        let dist = DiagGaussian::new(mean, log_std)?;

        // Sample at mean (x=0)
        let action = Tensor::zeros(&[1, 1], DType::F32, &device)?;
        let log_prob = dist.log_prob(&action)?;
        let log_prob_val: f32 = log_prob.to_vec1()?[0];

        // log_prob at mean of standard normal: -0.5 * log(2*pi) = -0.9189...
        assert!((log_prob_val - (-0.9189)).abs() < 0.01);

        Ok(())
    }
}
