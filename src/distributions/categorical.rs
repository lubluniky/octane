//! Categorical distribution for discrete action spaces.
//!
//! Used when the action space is finite and discrete (e.g., left/right/up/down).

use candle_core::Tensor;
use candle_nn::ops::log_softmax;
use rand::Rng;

use crate::core::{OctaneError, Result};
use crate::distributions::Distribution;

/// Categorical distribution for discrete action spaces.
///
/// Given logits (unnormalized log probabilities), this distribution samples
/// discrete action indices and computes their log probabilities.
///
/// # Example
///
/// ```ignore
/// let logits = Tensor::randn(0.0, 1.0, &[batch_size, num_actions], &device)?;
/// let dist = Categorical::new(logits)?;
/// let actions = dist.sample()?;  // [batch_size] tensor of action indices
/// let log_probs = dist.log_prob(&actions)?;  // [batch_size] tensor
/// ```
#[derive(Debug, Clone)]
pub struct Categorical {
    /// Raw logits (unnormalized log probabilities), shape: [batch_size, num_actions]
    logits: Tensor,
    /// Log probabilities (log_softmax of logits), shape: [batch_size, num_actions]
    log_probs: Tensor,
    /// Probabilities (softmax of logits), shape: [batch_size, num_actions]
    probs: Tensor,
}

impl Categorical {
    /// Create a new Categorical distribution from logits.
    ///
    /// # Arguments
    /// * `logits` - Unnormalized log probabilities with shape `[batch_size, num_actions]`
    ///
    /// # Returns
    /// A new Categorical distribution
    pub fn new(logits: Tensor) -> Result<Self> {
        // Validate shape
        if logits.dims().len() != 2 {
            return Err(OctaneError::ShapeMismatch {
                expected: vec![0, 0], // placeholder
                got: logits.dims().to_vec(),
            });
        }

        // Compute log_softmax for numerical stability
        let log_probs = log_softmax(&logits, 1)?;

        // Compute probabilities from log_probs for sampling
        let probs = log_probs.exp()?;

        Ok(Self {
            logits,
            log_probs,
            probs,
        })
    }

    /// Create a Categorical distribution from pre-computed probabilities.
    ///
    /// # Arguments
    /// * `probs` - Normalized probabilities with shape `[batch_size, num_actions]`
    ///
    /// # Returns
    /// A new Categorical distribution
    pub fn from_probs(probs: Tensor) -> Result<Self> {
        if probs.dims().len() != 2 {
            return Err(OctaneError::ShapeMismatch {
                expected: vec![0, 0],
                got: probs.dims().to_vec(),
            });
        }

        // Compute log probs with numerical stability
        let eps = Tensor::new(&[super::constants::EPSILON], probs.device())?;
        let probs_safe = probs.broadcast_add(&eps)?;
        let log_probs = probs_safe.log()?;
        let logits = log_probs.clone();

        Ok(Self {
            logits,
            log_probs,
            probs,
        })
    }

    /// Get the number of actions (categories).
    pub fn num_actions(&self) -> usize {
        self.logits.dims()[1]
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.logits.dims()[0]
    }

    /// Get the probabilities tensor.
    pub fn probs(&self) -> &Tensor {
        &self.probs
    }

    /// Get the log probabilities tensor.
    pub fn log_probs_tensor(&self) -> &Tensor {
        &self.log_probs
    }

    /// Get the logits tensor.
    pub fn logits(&self) -> &Tensor {
        &self.logits
    }

    /// Sample using the Gumbel-max trick for numerical stability.
    ///
    /// Gumbel-max: argmax(logits + gumbel_noise) is equivalent to sampling
    /// from the categorical distribution defined by softmax(logits).
    fn sample_gumbel_max(&self) -> Result<Tensor> {
        let batch_size = self.batch_size();
        let num_actions = self.num_actions();
        let device = self.logits.device();

        // Generate uniform random numbers
        let mut rng = rand::thread_rng();
        let uniform: Vec<f32> = (0..batch_size * num_actions)
            .map(|_| rng.gen::<f32>())
            .collect();

        let u = Tensor::from_slice(&uniform, &[batch_size, num_actions], device)?;

        // Gumbel noise: -log(-log(u))
        // Clamp u to avoid log(0)
        let u_clamped = u.clamp(super::constants::EPSILON, 1.0 - super::constants::EPSILON)?;
        let neg_log_u = u_clamped.log()?.neg()?;
        let gumbel = neg_log_u.log()?.neg()?;

        // Add Gumbel noise to logits and take argmax
        let perturbed = self.logits.broadcast_add(&gumbel)?;
        let samples = perturbed.argmax(1)?;

        Ok(samples)
    }

    /// Alternative sampling using inverse CDF (cumulative distribution function).
    /// This is more straightforward but potentially less numerically stable.
    #[allow(dead_code)]
    fn sample_inverse_cdf(&self) -> Result<Tensor> {
        let batch_size = self.batch_size();
        let num_actions = self.num_actions();
        let device = self.logits.device();

        // Get probabilities as a flat vector
        let probs_vec: Vec<f32> = self.probs.flatten_all()?.to_vec1()?;

        // Sample for each batch element
        let mut rng = rand::thread_rng();
        let mut samples = Vec::with_capacity(batch_size);

        for b in 0..batch_size {
            let u: f32 = rng.gen();
            let mut cumsum = 0.0f32;
            let mut selected = num_actions - 1; // Default to last action

            for a in 0..num_actions {
                cumsum += probs_vec[b * num_actions + a];
                if u < cumsum {
                    selected = a;
                    break;
                }
            }
            samples.push(selected as u32);
        }

        let samples_tensor = Tensor::from_slice(&samples, &[batch_size], device)?;
        Ok(samples_tensor)
    }
}

impl Distribution for Categorical {
    fn sample(&self) -> Result<Tensor> {
        self.sample_gumbel_max()
    }

    fn log_prob(&self, actions: &Tensor) -> Result<Tensor> {
        let batch_size = self.batch_size();
        let device = self.logits.device();

        // Actions should be [batch_size] of integer indices
        if actions.dims() != [batch_size] {
            return Err(OctaneError::ShapeMismatch {
                expected: vec![batch_size],
                got: actions.dims().to_vec(),
            });
        }

        // Gather log_probs at action indices
        // log_probs shape: [batch_size, num_actions]
        // actions shape: [batch_size]
        // We need to extract log_probs[b, actions[b]] for each b

        // Convert actions to indices for gathering
        let actions_u32: Vec<u32> = actions.to_vec1()?;
        let log_probs_flat: Vec<f32> = self.log_probs.flatten_all()?.to_vec1()?;
        let num_actions = self.num_actions();

        let mut selected_log_probs = Vec::with_capacity(batch_size);
        for (b, &action) in actions_u32.iter().enumerate() {
            let idx = b * num_actions + (action as usize);
            selected_log_probs.push(log_probs_flat[idx]);
        }

        let result = Tensor::from_slice(&selected_log_probs, &[batch_size], device)?;
        Ok(result)
    }

    fn entropy(&self) -> Result<Tensor> {
        // Entropy of categorical: -sum(p * log(p))
        // Using log_probs for numerical stability: -sum(exp(log_p) * log_p)
        let neg_entropy = (&self.probs * &self.log_probs)?;
        let entropy = neg_entropy.sum(1)?.neg()?;
        Ok(entropy)
    }

    fn mode(&self) -> Result<Tensor> {
        // Mode is the argmax of logits (or probs, same result)
        let mode = self.logits.argmax(1)?;
        Ok(mode)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_categorical_creation() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 0.0, 0.0, 0.0], &[2, 3], &device)?;

        let dist = Categorical::new(logits)?;

        assert_eq!(dist.batch_size(), 2);
        assert_eq!(dist.num_actions(), 3);

        Ok(())
    }

    #[test]
    fn test_categorical_probs_sum_to_one() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::randn(0.0f32, 1.0, &[10, 5], &device)?;

        let dist = Categorical::new(logits)?;
        let probs_sum = dist.probs().sum(1)?;
        let probs_sum_vec: Vec<f32> = probs_sum.to_vec1()?;

        for sum in probs_sum_vec {
            assert!((sum - 1.0).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_categorical_entropy_bounds() -> Result<()> {
        let device = candle_core::Device::Cpu;

        // Uniform distribution should have maximum entropy
        let uniform_logits = Tensor::zeros(&[1, 4], DType::F32, &device)?;
        let uniform_dist = Categorical::new(uniform_logits)?;
        let uniform_entropy: f32 = uniform_dist.entropy()?.to_vec1()?[0];

        // Max entropy for 4 actions is ln(4) = 1.386...
        let max_entropy = (4.0f32).ln();
        assert!((uniform_entropy - max_entropy).abs() < 1e-4);

        // Concentrated distribution should have low entropy
        let concentrated_logits = Tensor::from_slice(&[100.0f32, 0.0, 0.0, 0.0], &[1, 4], &device)?;
        let concentrated_dist = Categorical::new(concentrated_logits)?;
        let concentrated_entropy: f32 = concentrated_dist.entropy()?.to_vec1()?[0];

        // Should be close to 0
        assert!(concentrated_entropy < 0.1);

        Ok(())
    }

    #[test]
    fn test_categorical_mode() -> Result<()> {
        let device = candle_core::Device::Cpu;

        // Clear preference for action 2
        let logits = Tensor::from_slice(&[0.0f32, 0.0, 10.0, 0.0], &[1, 4], &device)?;
        let dist = Categorical::new(logits)?;

        let mode: u32 = dist.mode()?.to_vec1()?[0];
        assert_eq!(mode, 2);

        Ok(())
    }

    #[test]
    fn test_categorical_log_prob_negative() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::randn(0.0f32, 1.0, &[5, 3], &device)?;

        let dist = Categorical::new(logits)?;
        let samples = dist.sample()?;
        let log_probs: Vec<f32> = dist.log_prob(&samples)?.to_vec1()?;

        // Log probabilities should always be <= 0
        for lp in log_probs {
            assert!(lp <= 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_sample_gumbel_vs_inverse_cdf() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let logits = Tensor::from_slice(&[1.0f32, 2.0, 3.0], &[1, 3], &device)?;

        let dist = Categorical::new(logits)?;

        // Both methods should produce valid samples
        let gumbel_samples = dist.sample_gumbel_max()?;
        let cdf_samples = dist.sample_inverse_cdf()?;

        assert_eq!(gumbel_samples.dims(), &[1]);
        assert_eq!(cdf_samples.dims(), &[1]);

        // Samples should be valid action indices
        let gumbel_val: u32 = gumbel_samples.to_vec1()?[0];
        let cdf_val: u32 = cdf_samples.to_vec1()?[0];

        assert!(gumbel_val < 3);
        assert!(cdf_val < 3);

        Ok(())
    }
}
