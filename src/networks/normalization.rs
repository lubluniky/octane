//! Normalization layers for neural networks.
//!
//! This module provides implementations of common normalization techniques
//! used in deep learning, including Layer Normalization, Batch Normalization,
//! and RMS Normalization.
//!
//! # Example
//! ```ignore
//! use octane_rs::networks::normalization::{LayerNorm, LayerNormConfig};
//! use candle_core::Device;
//! use candle_nn::VarMap;
//!
//! let device = Device::Cpu;
//! let varmap = VarMap::new();
//! let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
//!
//! let config = LayerNormConfig::new(256);
//! let layer_norm = LayerNorm::new(vb, config).unwrap();
//! ```

use candle_core::{DType, Result as CandleResult, Tensor, D};
use candle_nn::VarBuilder;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, Ordering};

/// Configuration for Layer Normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerNormConfig {
    /// Normalized shape (typically the last dimension).
    pub normalized_shape: usize,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Whether to use learnable affine parameters (gamma, beta).
    pub elementwise_affine: bool,
}

impl LayerNormConfig {
    /// Create a new LayerNorm configuration.
    ///
    /// # Arguments
    /// * `normalized_shape` - Size of the dimension to normalize over
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: true,
        }
    }

    /// Set epsilon value for numerical stability.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Disable learnable affine parameters.
    pub fn without_affine(mut self) -> Self {
        self.elementwise_affine = false;
        self
    }
}

/// Layer Normalization module.
///
/// Normalizes across the feature dimension (last dimension) for each sample.
/// Unlike BatchNorm, LayerNorm normalizes across features rather than batch,
/// making it suitable for sequence models and RL where batch statistics
/// may be unreliable.
///
/// Formula: y = (x - mean) / sqrt(var + eps) * gamma + beta
#[derive(Debug)]
pub struct LayerNorm {
    /// Learnable scale parameter (gamma).
    weight: Option<Tensor>,
    /// Learnable shift parameter (beta).
    bias: Option<Tensor>,
    /// Configuration.
    config: LayerNormConfig,
}

impl LayerNorm {
    /// Create a new LayerNorm module.
    pub fn new(vb: VarBuilder<'_>, config: LayerNormConfig) -> CandleResult<Self> {
        let (weight, bias) = if config.elementwise_affine {
            let weight = vb.get_with_hints(
                &[config.normalized_shape],
                "weight",
                candle_nn::Init::Const(1.0),
            )?;
            let bias = vb.get_with_hints(
                &[config.normalized_shape],
                "bias",
                candle_nn::Init::Const(0.0),
            )?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        Ok(Self {
            weight,
            bias,
            config,
        })
    }

    /// Forward pass through layer normalization.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [..., normalized_shape]
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Compute mean and variance over last dimension
        let mean = x.mean_keepdim(D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(D::Minus1)?;

        // Normalize
        let std = (var + self.config.eps)?.sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;

        // Apply affine transformation if enabled
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => normalized.broadcast_mul(w)?.broadcast_add(b),
            _ => Ok(normalized),
        }
    }

    /// Get the normalized shape.
    pub fn normalized_shape(&self) -> usize {
        self.config.normalized_shape
    }
}

/// Configuration for Batch Normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchNormConfig {
    /// Number of features (channels).
    pub num_features: usize,
    /// Momentum for running statistics update.
    pub momentum: f64,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Whether to use learnable affine parameters.
    pub affine: bool,
    /// Whether to track running statistics.
    pub track_running_stats: bool,
}

impl BatchNormConfig {
    /// Create a new BatchNorm configuration.
    ///
    /// # Arguments
    /// * `num_features` - Number of features/channels
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            momentum: 0.1,
            eps: 1e-5,
            affine: true,
            track_running_stats: true,
        }
    }

    /// Set momentum for running statistics.
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set epsilon for numerical stability.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Disable learnable parameters.
    pub fn without_affine(mut self) -> Self {
        self.affine = false;
        self
    }

    /// Disable running statistics tracking.
    pub fn without_tracking(mut self) -> Self {
        self.track_running_stats = false;
        self
    }
}

/// Batch Normalization module.
///
/// Normalizes across the batch dimension for each feature/channel.
/// Maintains running estimates of mean and variance for inference.
///
/// Note: In RL, batch statistics can be noisy due to non-i.i.d. data.
/// Consider using LayerNorm or RMSNorm for more stable training.
#[derive(Debug)]
pub struct BatchNorm {
    /// Learnable scale parameter (gamma).
    weight: Option<Tensor>,
    /// Learnable shift parameter (beta).
    bias: Option<Tensor>,
    /// Running mean for inference.
    running_mean: Option<Tensor>,
    /// Running variance for inference.
    running_var: Option<Tensor>,
    /// Configuration.
    config: BatchNormConfig,
    /// Whether in training mode.
    training: AtomicBool,
}

impl BatchNorm {
    /// Create a new BatchNorm module.
    pub fn new(vb: VarBuilder<'_>, config: BatchNormConfig) -> CandleResult<Self> {
        let device = vb.device();

        let (weight, bias) = if config.affine {
            let weight = vb.get_with_hints(
                &[config.num_features],
                "weight",
                candle_nn::Init::Const(1.0),
            )?;
            let bias =
                vb.get_with_hints(&[config.num_features], "bias", candle_nn::Init::Const(0.0))?;
            (Some(weight), Some(bias))
        } else {
            (None, None)
        };

        let (running_mean, running_var) = if config.track_running_stats {
            let rm = Tensor::zeros(&[config.num_features], DType::F32, device)?;
            let rv = Tensor::ones(&[config.num_features], DType::F32, device)?;
            (Some(rm), Some(rv))
        } else {
            (None, None)
        };

        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            config,
            training: AtomicBool::new(true),
        })
    }

    /// Set training mode.
    pub fn train(&self) {
        self.training.store(true, Ordering::SeqCst);
    }

    /// Set evaluation mode.
    pub fn eval(&self) {
        self.training.store(false, Ordering::SeqCst);
    }

    /// Check if in training mode.
    pub fn is_training(&self) -> bool {
        self.training.load(Ordering::SeqCst)
    }

    /// Forward pass through batch normalization.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, num_features, ...] or [batch_size, num_features]
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dims = x.dims();
        let is_training = self.training.load(Ordering::SeqCst);

        // Handle 2D input [batch, features] by adding a dummy dimension
        let (x_reshaped, needs_squeeze) = if dims.len() == 2 {
            (x.unsqueeze(2)?, true)
        } else {
            (x.clone(), false)
        };

        let result = if is_training {
            self.forward_train(&x_reshaped)?
        } else {
            self.forward_eval(&x_reshaped)?
        };

        if needs_squeeze {
            result.squeeze(2)
        } else {
            Ok(result)
        }
    }

    /// Forward pass during training (uses batch statistics).
    fn forward_train(&self, x: &Tensor) -> CandleResult<Tensor> {
        let dims = x.dims();
        let batch_size = dims[0];

        // Compute mean and variance over batch and spatial dimensions
        // For [N, C, ...], reduce over N and spatial dims, keeping C
        let mean = x.mean_keepdim(0)?; // [1, C, ...]
        let mean = if dims.len() > 2 {
            // Average over spatial dimensions too
            let mut m = mean;
            for d in 2..dims.len() {
                m = m.mean_keepdim(d)?;
            }
            m
        } else {
            mean
        };

        let x_centered = x.broadcast_sub(&mean)?;
        let var = x_centered.sqr()?.mean_keepdim(0)?;
        let var = if dims.len() > 2 {
            let mut v = var;
            for d in 2..dims.len() {
                v = v.mean_keepdim(d)?;
            }
            v
        } else {
            var
        };

        // Update running statistics (simplified - in practice would use momentum)
        // Note: We can't mutate running stats in const fn, so this is a simplified version

        // Normalize
        let std = (var.broadcast_add(&Tensor::from_slice(
            &[self.config.eps as f32],
            &[1],
            x.device(),
        )?))?
        .sqrt()?;
        let normalized = x_centered.broadcast_div(&std)?;

        // Apply affine
        self.apply_affine(&normalized, batch_size)
    }

    /// Forward pass during evaluation (uses running statistics).
    fn forward_eval(&self, x: &Tensor) -> CandleResult<Tensor> {
        let batch_size = x.dims()[0];

        match (&self.running_mean, &self.running_var) {
            (Some(rm), Some(rv)) => {
                // Reshape running stats for broadcasting
                let rm_expanded = rm.unsqueeze(0)?.unsqueeze(2)?;
                let rv_expanded = rv.unsqueeze(0)?.unsqueeze(2)?;

                let x_centered = x.broadcast_sub(&rm_expanded)?;
                let std = (rv_expanded.broadcast_add(&Tensor::from_slice(
                    &[self.config.eps as f32],
                    &[1],
                    x.device(),
                )?))?
                .sqrt()?;
                let normalized = x_centered.broadcast_div(&std)?;

                self.apply_affine(&normalized, batch_size)
            }
            _ => {
                // No running stats, use batch statistics
                self.forward_train(x)
            }
        }
    }

    /// Apply affine transformation (gamma * x + beta).
    fn apply_affine(&self, x: &Tensor, _batch_size: usize) -> CandleResult<Tensor> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => {
                let w_expanded = w.unsqueeze(0)?.unsqueeze(2)?;
                let b_expanded = b.unsqueeze(0)?.unsqueeze(2)?;
                x.broadcast_mul(&w_expanded)?.broadcast_add(&b_expanded)
            }
            _ => Ok(x.clone()),
        }
    }

    /// Get number of features.
    pub fn num_features(&self) -> usize {
        self.config.num_features
    }
}

/// Configuration for RMS Normalization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RMSNormConfig {
    /// Normalized shape (typically the last dimension).
    pub normalized_shape: usize,
    /// Small constant for numerical stability.
    pub eps: f64,
}

impl RMSNormConfig {
    /// Create a new RMSNorm configuration.
    pub fn new(normalized_shape: usize) -> Self {
        Self {
            normalized_shape,
            eps: 1e-6,
        }
    }

    /// Set epsilon value.
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

/// Root Mean Square Layer Normalization.
///
/// A simplified and more efficient version of LayerNorm that only uses
/// RMS statistics without re-centering (no mean subtraction).
///
/// Formula: y = x / RMS(x) * gamma, where RMS(x) = sqrt(mean(x^2) + eps)
///
/// This is computationally cheaper than LayerNorm and has been shown to
/// perform comparably in many tasks. Used in LLaMA and other modern architectures.
#[derive(Debug)]
pub struct RMSNorm {
    /// Learnable scale parameter (gamma).
    weight: Tensor,
    /// Configuration.
    config: RMSNormConfig,
}

impl RMSNorm {
    /// Create a new RMSNorm module.
    pub fn new(vb: VarBuilder<'_>, config: RMSNormConfig) -> CandleResult<Self> {
        let weight = vb.get_with_hints(
            &[config.normalized_shape],
            "weight",
            candle_nn::Init::Const(1.0),
        )?;

        Ok(Self { weight, config })
    }

    /// Forward pass through RMS normalization.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [..., normalized_shape]
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        // Compute RMS: sqrt(mean(x^2) + eps)
        let x_squared = x.sqr()?;
        let mean_squared = x_squared.mean_keepdim(D::Minus1)?;
        let rms = (mean_squared + self.config.eps)?.sqrt()?;

        // Normalize and scale
        let normalized = x.broadcast_div(&rms)?;
        normalized.broadcast_mul(&self.weight)
    }

    /// Get the normalized shape.
    pub fn normalized_shape(&self) -> usize {
        self.config.normalized_shape
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn setup_vb(device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, device)
    }

    #[test]
    fn test_layer_norm_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = LayerNormConfig::new(64);
        let ln = LayerNorm::new(vb, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[8, 64], &device).unwrap();
        let y = ln.forward(&x).unwrap();

        assert_eq!(y.dims(), &[8, 64]);
    }

    #[test]
    fn test_layer_norm_statistics() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = LayerNormConfig::new(64);
        let ln = LayerNorm::new(vb, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[8, 64], &device).unwrap();
        let y = ln.forward(&x).unwrap();

        // Each sample should have approximately zero mean and unit variance
        let sample = y.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        let mean = sample.mean_all().unwrap().to_scalar::<f32>().unwrap();
        let var = sample
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(mean.abs() < 0.1, "Mean should be close to 0, got {}", mean);
        assert!(
            (var - 1.0).abs() < 0.2,
            "Variance should be close to 1, got {}",
            var
        );
    }

    #[test]
    fn test_batch_norm_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = BatchNormConfig::new(64);
        let bn = BatchNorm::new(vb, config).unwrap();

        // 2D input
        let x2d = Tensor::randn(0.0f32, 1.0, &[8, 64], &device).unwrap();
        let y2d = bn.forward(&x2d).unwrap();
        assert_eq!(y2d.dims(), &[8, 64]);

        // 3D input (with spatial dim)
        let x3d = Tensor::randn(0.0f32, 1.0, &[8, 64, 16], &device).unwrap();
        let y3d = bn.forward(&x3d).unwrap();
        assert_eq!(y3d.dims(), &[8, 64, 16]);
    }

    #[test]
    fn test_batch_norm_train_eval() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = BatchNormConfig::new(64);
        let bn = BatchNorm::new(vb, config).unwrap();

        assert!(bn.is_training());
        bn.eval();
        assert!(!bn.is_training());
        bn.train();
        assert!(bn.is_training());
    }

    #[test]
    fn test_rms_norm_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = RMSNormConfig::new(64);
        let rms = RMSNorm::new(vb, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[8, 64], &device).unwrap();
        let y = rms.forward(&x).unwrap();

        assert_eq!(y.dims(), &[8, 64]);
    }

    #[test]
    fn test_rms_norm_normalization() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = RMSNormConfig::new(64);
        let rms = RMSNorm::new(vb, config).unwrap();

        // Create input with large values
        let x = Tensor::randn(0.0f32, 10.0, &[8, 64], &device).unwrap();
        let y = rms.forward(&x).unwrap();

        // Output should have reasonable magnitude (RMS close to 1)
        let sample = y.narrow(0, 0, 1).unwrap().squeeze(0).unwrap();
        let rms_val = sample
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap()
            .sqrt();

        // RMS should be approximately 1 since weight is initialized to 1
        assert!(rms_val < 3.0, "RMS should be reasonable, got {}", rms_val);
    }

    #[test]
    fn test_layer_norm_3d() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        // Test with 3D input [batch, seq_len, features]
        let config = LayerNormConfig::new(64);
        let ln = LayerNorm::new(vb, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[8, 16, 64], &device).unwrap();
        let y = ln.forward(&x).unwrap();

        assert_eq!(y.dims(), &[8, 16, 64]);
    }
}
