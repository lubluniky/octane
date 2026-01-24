//! Diagonal Gaussian distribution for continuous action spaces.
//!
//! Used when the action space is continuous (e.g., joint angles, velocities).

use candle_core::Tensor;
use rand::Rng;
use rand_distr::StandardNormal;

use crate::core::{Result, RocketError};
use crate::distributions::{constants, Distribution};

/// Diagonal Gaussian (Normal) distribution for continuous action spaces.
///
/// This distribution parameterizes each action dimension independently with
/// its own mean and standard deviation (diagonal covariance matrix).
///
/// Uses the reparameterization trick for sampling:
/// `sample = mean + std * noise` where `noise ~ N(0, 1)`
///
/// This enables gradient flow through the sampling operation, which is
/// essential for policy gradient methods.
///
/// # Example
///
/// ```ignore
/// // Network outputs mean and log_std
/// let mean = network.forward_mean(&obs)?;      // [batch_size, action_dim]
/// let log_std = network.forward_log_std(&obs)?; // [batch_size, action_dim]
///
/// let dist = DiagGaussian::new(mean, log_std)?;
/// let actions = dist.sample()?;  // [batch_size, action_dim]
/// let log_probs = dist.log_prob(&actions)?;  // [batch_size]
/// ```
#[derive(Debug, Clone)]
pub struct DiagGaussian {
    /// Mean of the distribution, shape: [batch_size, action_dim]
    mean: Tensor,
    /// Log standard deviation (clamped for stability), shape: [batch_size, action_dim]
    log_std: Tensor,
    /// Standard deviation (exp of clamped log_std), shape: [batch_size, action_dim]
    std: Tensor,
    /// Variance (std^2), shape: [batch_size, action_dim]
    variance: Tensor,
}

impl DiagGaussian {
    /// Create a new DiagGaussian distribution.
    ///
    /// # Arguments
    /// * `mean` - Mean of the distribution with shape `[batch_size, action_dim]`
    /// * `log_std` - Log standard deviation with shape `[batch_size, action_dim]`
    ///
    /// # Returns
    /// A new DiagGaussian distribution with clamped log_std for numerical stability
    ///
    /// # Notes
    /// - `log_std` is clamped to `[LOG_STD_MIN, LOG_STD_MAX]` for stability
    /// - LOG_STD_MIN = -20 prevents std from becoming too small (vanishing gradients)
    /// - LOG_STD_MAX = 2 prevents std from becoming too large (exploding outputs)
    pub fn new(mean: Tensor, log_std: Tensor) -> Result<Self> {
        // Validate shapes match
        if mean.dims() != log_std.dims() {
            return Err(RocketError::ShapeMismatch {
                expected: mean.dims().to_vec(),
                got: log_std.dims().to_vec(),
            });
        }

        // Validate 2D shape [batch_size, action_dim]
        if mean.dims().len() != 2 {
            return Err(RocketError::ShapeMismatch {
                expected: vec![0, 0], // placeholder for 2D
                got: mean.dims().to_vec(),
            });
        }

        // Clamp log_std for numerical stability
        let log_std_clamped = log_std.clamp(constants::LOG_STD_MIN, constants::LOG_STD_MAX)?;

        // Compute std and variance
        let std = log_std_clamped.exp()?;
        let variance = std.sqr()?;

        Ok(Self {
            mean,
            log_std: log_std_clamped,
            std,
            variance,
        })
    }

    /// Create a DiagGaussian with a fixed (learnable) log_std parameter.
    ///
    /// This is useful when log_std is a separate learnable parameter rather
    /// than a network output. The log_std is broadcast across the batch.
    ///
    /// # Arguments
    /// * `mean` - Mean with shape `[batch_size, action_dim]`
    /// * `log_std` - Single log_std with shape `[action_dim]` or `[1, action_dim]`
    pub fn with_fixed_std(mean: Tensor, log_std: Tensor) -> Result<Self> {
        let batch_size = mean.dims()[0];
        let action_dim = mean.dims()[1];

        // Broadcast log_std to match mean shape
        let log_std_broadcast = if log_std.dims().len() == 1 {
            // [action_dim] -> [1, action_dim] -> [batch_size, action_dim]
            let expanded = log_std.unsqueeze(0)?;
            expanded.broadcast_as(&[batch_size, action_dim])?
        } else if log_std.dims() == [1, action_dim] {
            log_std.broadcast_as(&[batch_size, action_dim])?
        } else {
            log_std
        };

        Self::new(mean, log_std_broadcast)
    }

    /// Get the mean tensor.
    pub fn mean(&self) -> &Tensor {
        &self.mean
    }

    /// Get the (clamped) log standard deviation tensor.
    pub fn log_std(&self) -> &Tensor {
        &self.log_std
    }

    /// Get the standard deviation tensor.
    pub fn std(&self) -> &Tensor {
        &self.std
    }

    /// Get the variance tensor.
    pub fn variance(&self) -> &Tensor {
        &self.variance
    }

    /// Get the batch size.
    pub fn batch_size(&self) -> usize {
        self.mean.dims()[0]
    }

    /// Get the action dimension.
    pub fn action_dim(&self) -> usize {
        self.mean.dims()[1]
    }

    /// Sample from standard normal distribution.
    fn sample_standard_normal(&self) -> Result<Tensor> {
        let batch_size = self.batch_size();
        let action_dim = self.action_dim();
        let device = self.mean.device();

        // Generate standard normal samples
        let mut rng = rand::thread_rng();
        let samples: Vec<f32> = (0..batch_size * action_dim)
            .map(|_| rng.sample(StandardNormal))
            .collect();

        let noise = Tensor::from_slice(&samples, &[batch_size, action_dim], device)?;
        Ok(noise)
    }

    /// Compute log probability for a single action dimension.
    /// Formula: -0.5 * ((x - mean) / std)^2 - log(std) - 0.5 * log(2*pi)
    fn log_prob_single_dim(
        x: &Tensor,
        mean: &Tensor,
        std: &Tensor,
        log_std: &Tensor,
    ) -> Result<Tensor> {
        // z = (x - mean) / std
        let diff = x.sub(mean)?;
        let z = diff.broadcast_div(std)?;

        // -0.5 * z^2
        let z_sq = z.sqr()?;
        let neg_half_z_sq = (z_sq * (-0.5))?;

        // -log(std)
        let neg_log_std = log_std.neg()?;

        // -0.5 * log(2*pi)
        let log_2pi_term = -0.5 * constants::LOG_2PI;

        // Sum: -0.5 * z^2 - log(std) - 0.5 * log(2*pi)
        let log_prob = (neg_half_z_sq + neg_log_std)?;
        let log_prob = (log_prob + log_2pi_term)?;

        Ok(log_prob)
    }
}

impl Distribution for DiagGaussian {
    fn sample(&self) -> Result<Tensor> {
        // Reparameterization trick: sample = mean + std * noise
        let noise = self.sample_standard_normal()?;
        let scaled_noise = self.std.mul(&noise)?;
        let sample = self.mean.add(&scaled_noise)?;
        Ok(sample)
    }

    fn log_prob(&self, actions: &Tensor) -> Result<Tensor> {
        let batch_size = self.batch_size();
        let action_dim = self.action_dim();

        // Validate action shape
        if actions.dims() != [batch_size, action_dim] {
            return Err(RocketError::ShapeMismatch {
                expected: vec![batch_size, action_dim],
                got: actions.dims().to_vec(),
            });
        }

        // Compute log prob for each dimension
        let log_prob_per_dim =
            Self::log_prob_single_dim(actions, &self.mean, &self.std, &self.log_std)?;

        // Sum across action dimensions to get total log prob
        // Independent dimensions -> sum of log probs
        let log_prob = log_prob_per_dim.sum(1)?;

        Ok(log_prob)
    }

    fn entropy(&self) -> Result<Tensor> {
        // Entropy of multivariate Gaussian with diagonal covariance:
        // H = 0.5 * action_dim * (1 + log(2*pi)) + sum(log(std))
        // Per dimension: 0.5 * (1 + log(2*pi)) + log(std)

        let action_dim = self.action_dim() as f64;

        // Sum of log_std across dimensions
        let sum_log_std = self.log_std.sum(1)?;

        // 0.5 * action_dim * (1 + log(2*pi))
        let constant_term = 0.5 * action_dim * (1.0 + constants::LOG_2PI);

        // Total entropy
        let entropy = (sum_log_std + constant_term)?;

        Ok(entropy)
    }

    fn mode(&self) -> Result<Tensor> {
        // Mode of Gaussian is the mean
        Ok(self.mean.clone())
    }
}

/// Squashed Gaussian distribution for bounded continuous actions.
///
/// Applies tanh squashing to bound actions to [-1, 1], commonly used in SAC.
/// The log probability is adjusted for the squashing transformation.
#[derive(Debug, Clone)]
pub struct SquashedGaussian {
    /// The underlying unsquashed Gaussian distribution.
    base_dist: DiagGaussian,
}

impl SquashedGaussian {
    /// Create a new SquashedGaussian distribution.
    ///
    /// # Arguments
    /// * `mean` - Mean of the underlying Gaussian
    /// * `log_std` - Log standard deviation of the underlying Gaussian
    pub fn new(mean: Tensor, log_std: Tensor) -> Result<Self> {
        let base_dist = DiagGaussian::new(mean, log_std)?;
        Ok(Self { base_dist })
    }

    /// Get the underlying DiagGaussian distribution.
    pub fn base_distribution(&self) -> &DiagGaussian {
        &self.base_dist
    }

    /// Sample and return both the squashed action and the unsquashed sample.
    ///
    /// This is useful when you need the unsquashed sample for log_prob computation.
    pub fn sample_with_unsquashed(&self) -> Result<(Tensor, Tensor)> {
        let unsquashed = self.base_dist.sample()?;
        let squashed = unsquashed.tanh()?;
        Ok((squashed, unsquashed))
    }

    /// Compute log probability with the Jacobian correction for tanh squashing.
    ///
    /// # Arguments
    /// * `actions` - Squashed actions in [-1, 1]
    /// * `unsquashed` - Optional unsquashed samples (for numerical stability)
    pub fn log_prob_with_correction(
        &self,
        actions: &Tensor,
        unsquashed: Option<&Tensor>,
    ) -> Result<Tensor> {
        // If unsquashed is not provided, compute it via atanh
        let unsquashed_actions = match unsquashed {
            Some(u) => u.clone(),
            None => {
                // atanh(x) = 0.5 * ln((1+x)/(1-x))
                // Clamp to avoid numerical issues at boundaries
                let clamped = actions.clamp(-0.999, 0.999)?;
                // Manual atanh: 0.5 * ln((1+x)/(1-x))
                let one_plus_x = (1.0 + &clamped)?;
                let one_minus_x = (1.0 - &clamped)?;
                let ratio = one_plus_x.div(&one_minus_x)?;
                let ln_ratio = ratio.log()?;
                (ln_ratio * 0.5)?
            }
        };

        // Log prob of the unsquashed action
        let log_prob_unsquashed = self.base_dist.log_prob(&unsquashed_actions)?;

        // Jacobian correction: -sum(log(1 - tanh(u)^2))
        // = -sum(log(1 - actions^2))
        let actions_sq = actions.sqr()?;
        let one_minus_sq = (1.0 - &actions_sq)?;
        // Add epsilon for numerical stability
        let one_minus_sq_safe = (one_minus_sq + constants::EPSILON as f64)?;
        let log_jacobian = one_minus_sq_safe.log()?.sum(1)?;

        // Final log prob = log_prob_unsquashed - log_jacobian
        let log_prob = log_prob_unsquashed.sub(&log_jacobian)?;

        Ok(log_prob)
    }
}

impl Distribution for SquashedGaussian {
    fn sample(&self) -> Result<Tensor> {
        let (squashed, _) = self.sample_with_unsquashed()?;
        Ok(squashed)
    }

    fn log_prob(&self, actions: &Tensor) -> Result<Tensor> {
        self.log_prob_with_correction(actions, None)
    }

    fn entropy(&self) -> Result<Tensor> {
        // Entropy of squashed Gaussian is complex; use base entropy as approximation
        // For exact entropy, would need numerical integration
        self.base_dist.entropy()
    }

    fn mode(&self) -> Result<Tensor> {
        // Mode is tanh(mean)
        self.base_dist.mean().tanh().map_err(|e| e.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;

    #[test]
    fn test_diag_gaussian_creation() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean = Tensor::zeros(&[4, 2], DType::F32, &device)?;
        let log_std = Tensor::zeros(&[4, 2], DType::F32, &device)?;

        let dist = DiagGaussian::new(mean, log_std)?;

        assert_eq!(dist.batch_size(), 4);
        assert_eq!(dist.action_dim(), 2);

        Ok(())
    }

    #[test]
    fn test_diag_gaussian_sample_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean = Tensor::zeros(&[8, 3], DType::F32, &device)?;
        let log_std = Tensor::zeros(&[8, 3], DType::F32, &device)?;

        let dist = DiagGaussian::new(mean, log_std)?;
        let samples = dist.sample()?;

        assert_eq!(samples.dims(), &[8, 3]);

        Ok(())
    }

    #[test]
    fn test_diag_gaussian_log_std_clamping() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean = Tensor::zeros(&[1, 2], DType::F32, &device)?;

        // Very extreme log_std values
        let log_std = Tensor::from_slice(&[-100.0f32, 100.0], &[1, 2], &device)?;

        let dist = DiagGaussian::new(mean, log_std)?;
        let clamped_log_std: Vec<f32> = dist.log_std().flatten_all()?.to_vec1()?;

        // Should be clamped to [LOG_STD_MIN, LOG_STD_MAX]
        assert!((clamped_log_std[0] - constants::LOG_STD_MIN).abs() < 1e-5);
        assert!((clamped_log_std[1] - constants::LOG_STD_MAX).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_diag_gaussian_log_prob_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean = Tensor::zeros(&[5, 4], DType::F32, &device)?;
        let log_std = Tensor::zeros(&[5, 4], DType::F32, &device)?;

        let dist = DiagGaussian::new(mean, log_std)?;
        let actions = Tensor::randn(0.0f32, 1.0, &[5, 4], &device)?;
        let log_probs = dist.log_prob(&actions)?;

        // Log probs should be [batch_size]
        assert_eq!(log_probs.dims(), &[5]);

        Ok(())
    }

    #[test]
    fn test_diag_gaussian_entropy_shape() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean = Tensor::zeros(&[3, 2], DType::F32, &device)?;
        let log_std = Tensor::zeros(&[3, 2], DType::F32, &device)?;

        let dist = DiagGaussian::new(mean, log_std)?;
        let entropy = dist.entropy()?;

        assert_eq!(entropy.dims(), &[3]);

        Ok(())
    }

    #[test]
    fn test_diag_gaussian_entropy_increases_with_std() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean = Tensor::zeros(&[1, 2], DType::F32, &device)?;

        // Low variance
        let log_std_low = Tensor::from_slice(&[-1.0f32, -1.0], &[1, 2], &device)?;
        let dist_low = DiagGaussian::new(mean.clone(), log_std_low)?;
        let entropy_low: f32 = dist_low.entropy()?.to_vec1()?[0];

        // High variance
        let log_std_high = Tensor::from_slice(&[1.0f32, 1.0], &[1, 2], &device)?;
        let dist_high = DiagGaussian::new(mean, log_std_high)?;
        let entropy_high: f32 = dist_high.entropy()?.to_vec1()?[0];

        // Higher std should have higher entropy
        assert!(entropy_high > entropy_low);

        Ok(())
    }

    #[test]
    fn test_diag_gaussian_mode() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mean = Tensor::from_slice(&mean_data, &[2, 2], &device)?;
        let log_std = Tensor::zeros(&[2, 2], DType::F32, &device)?;

        let dist = DiagGaussian::new(mean.clone(), log_std)?;
        let mode = dist.mode()?;

        // Mode should equal mean
        let mode_vec: Vec<f32> = mode.flatten_all()?.to_vec1()?;
        for (i, &m) in mode_vec.iter().enumerate() {
            assert!((m - mean_data[i]).abs() < 1e-5);
        }

        Ok(())
    }

    #[test]
    fn test_diag_gaussian_standard_normal_log_prob() -> Result<()> {
        let device = candle_core::Device::Cpu;

        // Standard normal: mean=0, std=1
        let mean = Tensor::zeros(&[1, 1], DType::F32, &device)?;
        let log_std = Tensor::zeros(&[1, 1], DType::F32, &device)?;

        let dist = DiagGaussian::new(mean, log_std)?;

        // Log prob at x=0 should be -0.5 * log(2*pi) = -0.9189...
        let x = Tensor::zeros(&[1, 1], DType::F32, &device)?;
        let log_prob: f32 = dist.log_prob(&x)?.to_vec1()?[0];
        let expected = -0.5 * constants::LOG_2PI as f32;
        assert!((log_prob - expected).abs() < 1e-4);

        // Log prob at x=1 should be -0.5 * 1^2 - 0.5 * log(2*pi) = -1.4189...
        let x = Tensor::from_slice(&[1.0f32], &[1, 1], &device)?;
        let log_prob: f32 = dist.log_prob(&x)?.to_vec1()?[0];
        let expected = -0.5 - 0.5 * constants::LOG_2PI as f32;
        assert!((log_prob - expected).abs() < 1e-4);

        Ok(())
    }

    #[test]
    fn test_with_fixed_std() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean = Tensor::zeros(&[4, 3], DType::F32, &device)?;
        let log_std = Tensor::from_slice(&[0.0f32, 0.5, -0.5], &[3], &device)?;

        let dist = DiagGaussian::with_fixed_std(mean, log_std)?;

        assert_eq!(dist.batch_size(), 4);
        assert_eq!(dist.action_dim(), 3);
        assert_eq!(dist.log_std().dims(), &[4, 3]);

        Ok(())
    }

    #[test]
    fn test_squashed_gaussian_bounds() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean = Tensor::randn(0.0f32, 2.0, &[100, 4], &device)?;
        let log_std = Tensor::zeros(&[100, 4], DType::F32, &device)?;

        let dist = SquashedGaussian::new(mean, log_std)?;
        let samples = dist.sample()?;
        let samples_vec: Vec<f32> = samples.flatten_all()?.to_vec1()?;

        // All samples should be in (-1, 1)
        for &s in &samples_vec {
            assert!(s > -1.0 && s < 1.0, "Sample {} out of bounds", s);
        }

        Ok(())
    }

    #[test]
    fn test_squashed_gaussian_mode() -> Result<()> {
        let device = candle_core::Device::Cpu;
        let mean = Tensor::from_slice(&[0.0f32, 1.0, -1.0, 2.0], &[2, 2], &device)?;
        let log_std = Tensor::zeros(&[2, 2], DType::F32, &device)?;

        let dist = SquashedGaussian::new(mean, log_std)?;
        let mode = dist.mode()?;
        let mode_vec: Vec<f32> = mode.flatten_all()?.to_vec1()?;

        // Mode should be tanh(mean)
        let expected = vec![
            0.0f32.tanh(),
            1.0f32.tanh(),
            (-1.0f32).tanh(),
            2.0f32.tanh(),
        ];
        for (i, &m) in mode_vec.iter().enumerate() {
            assert!((m - expected[i]).abs() < 1e-5);
        }

        Ok(())
    }
}
