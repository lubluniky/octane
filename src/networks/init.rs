//! Weight initialization utilities for neural networks.
//!
//! Provides standard initialization methods commonly used in reinforcement learning,
//! including orthogonal initialization (recommended for policy gradients) and
//! various Xavier/Kaiming variants.
//!
//! # Example
//! ```ignore
//! use octane_rs::networks::init::{orthogonal_init, InitMethod};
//! use candle_core::{Device, DType, Tensor};
//!
//! let device = Device::Cpu;
//! let weight = Tensor::randn(0.0f32, 1.0, &[64, 32], &device).unwrap();
//! let initialized = orthogonal_init(&weight, 1.0).unwrap();
//! ```

use candle_core::{DType, Device, Result as CandleResult, Tensor};
use serde::{Deserialize, Serialize};

/// Weight initialization method.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum InitMethod {
    /// Orthogonal initialization with specified gain (default for RL).
    #[default]
    Orthogonal,
    /// Xavier/Glorot uniform initialization.
    XavierUniform,
    /// Xavier/Glorot normal initialization.
    XavierNormal,
    /// Kaiming/He uniform initialization (for ReLU networks).
    KaimingUniform,
    /// Kaiming/He normal initialization (for ReLU networks).
    KaimingNormal,
    /// Zero initialization (typically for biases).
    Zeros,
    /// Constant value initialization.
    Constant(f64),
}

impl InitMethod {
    /// Apply this initialization method to create a new tensor.
    ///
    /// # Arguments
    /// * `shape` - Shape of the tensor to create
    /// * `dtype` - Data type of the tensor
    /// * `device` - Device to create tensor on
    /// * `gain` - Gain factor (used by orthogonal, xavier)
    pub fn init(
        &self,
        shape: &[usize],
        dtype: DType,
        device: &Device,
        gain: f64,
    ) -> CandleResult<Tensor> {
        match self {
            InitMethod::Orthogonal => {
                let random = Tensor::randn(0.0f64, 1.0, shape, device)?;
                let result = orthogonal_init_impl(&random, gain)?;
                result.to_dtype(dtype)
            }
            InitMethod::XavierUniform => xavier_uniform_tensor(shape, dtype, device, gain),
            InitMethod::XavierNormal => xavier_normal_tensor(shape, dtype, device, gain),
            InitMethod::KaimingUniform => kaiming_uniform_tensor(shape, dtype, device),
            InitMethod::KaimingNormal => kaiming_normal_tensor(shape, dtype, device),
            InitMethod::Zeros => Tensor::zeros(shape, dtype, device),
            InitMethod::Constant(val) => Tensor::ones(shape, dtype, device).and_then(|t| t * *val),
        }
    }
}

/// Orthogonal initialization for weight tensors.
///
/// Orthogonal initialization helps maintain gradient magnitudes during training,
/// which is especially important for deep networks and recurrent architectures.
/// This is the recommended initialization for RL policy networks.
///
/// # Arguments
/// * `tensor` - The tensor to use as shape reference (will be replaced)
/// * `gain` - Scaling factor (1.0 for linear, sqrt(2) for ReLU, 0.01 for output layers)
///
/// # Returns
/// A new tensor with orthogonal initialization
///
/// # Example
/// ```ignore
/// let weight = Tensor::randn(0.0f32, 1.0, &[256, 64], &device)?;
/// let ortho_weight = orthogonal_init(&weight, 1.0)?;
/// ```
pub fn orthogonal_init(tensor: &Tensor, gain: f64) -> CandleResult<Tensor> {
    let dtype = tensor.dtype();
    let result = orthogonal_init_impl(tensor, gain)?;
    result.to_dtype(dtype)
}

/// Internal implementation of orthogonal initialization.
fn orthogonal_init_impl(tensor: &Tensor, gain: f64) -> CandleResult<Tensor> {
    let shape = tensor.dims();
    let device = tensor.device();

    if shape.len() < 2 {
        // For 1D tensors, just return scaled random values
        return Tensor::randn(0.0f64, gain, shape, device);
    }

    // Compute rows and cols for QR decomposition
    let rows = shape[0];
    let cols: usize = shape[1..].iter().product();

    // Create random matrix
    let random = if rows >= cols {
        Tensor::randn(0.0f64, 1.0, &[rows, cols], device)?
    } else {
        Tensor::randn(0.0f64, 1.0, &[cols, rows], device)?
    };

    // Perform QR decomposition via Gram-Schmidt orthogonalization
    // This is a simplified version - for large matrices, proper QR would be better
    let q = gram_schmidt_orthogonalize(&random)?;

    // Extract the appropriate part and scale by gain
    let q = if rows >= cols {
        q.narrow(1, 0, cols)?.contiguous()?
    } else {
        q.t()?.narrow(1, 0, rows)?.contiguous()?
    };

    let q_scaled = (q * gain)?;

    // Reshape to original shape
    q_scaled.reshape(shape)
}

/// Gram-Schmidt orthogonalization for a matrix.
///
/// Produces an orthonormal matrix from the input.
fn gram_schmidt_orthogonalize(a: &Tensor) -> CandleResult<Tensor> {
    let dims = a.dims();
    let (rows, cols) = (dims[0], dims[1]);
    let device = a.device();
    let dtype = a.dtype();

    let mut q_columns: Vec<Tensor> = Vec::with_capacity(cols);

    for j in 0..cols {
        // Get column j
        let mut v = a.narrow(1, j, 1)?.squeeze(1)?;

        // Subtract projections onto previous orthonormal vectors
        for q_col in &q_columns {
            // dot product: sum(v * q_col)
            let dot = (&v * q_col)?.sum_all()?;
            let proj = (q_col * &dot.broadcast_as(&[rows])?)?;
            v = (&v - &proj)?;
        }

        // Normalize
        let norm = v.sqr()?.sum_all()?.sqrt()?;
        let norm_scalar = norm.to_scalar::<f64>()?;

        // Avoid division by zero
        let v_normalized = if norm_scalar > 1e-10 {
            (&v / &norm.broadcast_as(&[rows])?)?
        } else {
            // If the vector is nearly zero, use a random unit vector
            let random = Tensor::randn(0.0f64, 1.0, &[rows], device)?;
            let random_norm = random.sqr()?.sum_all()?.sqrt()?;
            (&random / &random_norm.broadcast_as(&[rows])?)?
        };

        q_columns.push(v_normalized);
    }

    // Stack columns back into matrix
    let q_unsqueezed: Vec<Tensor> = q_columns
        .into_iter()
        .map(|c| c.unsqueeze(1))
        .collect::<CandleResult<Vec<_>>>()?;

    let q_refs: Vec<&Tensor> = q_unsqueezed.iter().collect();
    let q = Tensor::cat(&q_refs, 1)?;

    q.to_dtype(dtype)
}

/// Xavier/Glorot uniform initialization.
///
/// Samples from U(-a, a) where a = gain * sqrt(6 / (fan_in + fan_out))
///
/// Best for tanh and sigmoid activations.
///
/// # Arguments
/// * `tensor` - Tensor to initialize (shape reference)
/// * `gain` - Scaling factor (default 1.0)
pub fn xavier_uniform(tensor: &Tensor, gain: f64) -> CandleResult<Tensor> {
    let shape = tensor.dims();
    let dtype = tensor.dtype();
    let device = tensor.device();
    xavier_uniform_tensor(shape, dtype, device, gain)
}

fn xavier_uniform_tensor(
    shape: &[usize],
    dtype: DType,
    device: &Device,
    gain: f64,
) -> CandleResult<Tensor> {
    let (fan_in, fan_out) = compute_fan_in_out(shape);
    let std = gain * (6.0 / (fan_in + fan_out) as f64).sqrt();

    // Uniform distribution U(-std, std)
    let random = Tensor::rand(-std as f32, std as f32, shape, device)?;
    random.to_dtype(dtype)
}

/// Xavier/Glorot normal initialization.
///
/// Samples from N(0, std^2) where std = gain * sqrt(2 / (fan_in + fan_out))
///
/// Best for tanh and sigmoid activations.
///
/// # Arguments
/// * `tensor` - Tensor to initialize (shape reference)
/// * `gain` - Scaling factor (default 1.0)
pub fn xavier_normal(tensor: &Tensor, gain: f64) -> CandleResult<Tensor> {
    let shape = tensor.dims();
    let dtype = tensor.dtype();
    let device = tensor.device();
    xavier_normal_tensor(shape, dtype, device, gain)
}

fn xavier_normal_tensor(
    shape: &[usize],
    dtype: DType,
    device: &Device,
    gain: f64,
) -> CandleResult<Tensor> {
    let (fan_in, fan_out) = compute_fan_in_out(shape);
    let std = gain * (2.0 / (fan_in + fan_out) as f64).sqrt();

    let random = Tensor::randn(0.0f32, std as f32, shape, device)?;
    random.to_dtype(dtype)
}

/// Kaiming/He uniform initialization.
///
/// Samples from U(-bound, bound) where bound = sqrt(6 / fan_in)
///
/// Designed for ReLU and variants. Use mode="fan_in" by default.
///
/// # Arguments
/// * `tensor` - Tensor to initialize (shape reference)
pub fn kaiming_uniform(tensor: &Tensor) -> CandleResult<Tensor> {
    let shape = tensor.dims();
    let dtype = tensor.dtype();
    let device = tensor.device();
    kaiming_uniform_tensor(shape, dtype, device)
}

fn kaiming_uniform_tensor(shape: &[usize], dtype: DType, device: &Device) -> CandleResult<Tensor> {
    let (fan_in, _) = compute_fan_in_out(shape);
    // For ReLU, gain = sqrt(2)
    let gain = std::f64::consts::SQRT_2;
    let std = gain / (fan_in as f64).sqrt();
    let bound = std * (3.0f64).sqrt(); // uniform bound

    let random = Tensor::rand(-bound as f32, bound as f32, shape, device)?;
    random.to_dtype(dtype)
}

/// Kaiming/He normal initialization.
///
/// Samples from N(0, std^2) where std = sqrt(2 / fan_in)
///
/// Designed for ReLU and variants. Use mode="fan_in" by default.
///
/// # Arguments
/// * `tensor` - Tensor to initialize (shape reference)
pub fn kaiming_normal(tensor: &Tensor) -> CandleResult<Tensor> {
    let shape = tensor.dims();
    let dtype = tensor.dtype();
    let device = tensor.device();
    kaiming_normal_tensor(shape, dtype, device)
}

fn kaiming_normal_tensor(shape: &[usize], dtype: DType, device: &Device) -> CandleResult<Tensor> {
    let (fan_in, _) = compute_fan_in_out(shape);
    // For ReLU, gain = sqrt(2)
    let gain = std::f64::consts::SQRT_2;
    let std = gain / (fan_in as f64).sqrt();

    let random = Tensor::randn(0.0f32, std as f32, shape, device)?;
    random.to_dtype(dtype)
}

/// Compute fan_in and fan_out for a weight tensor.
///
/// For 2D tensors (Linear): fan_in = cols, fan_out = rows
/// For 4D tensors (Conv2d): fan_in = c_in * kH * kW, fan_out = c_out * kH * kW
fn compute_fan_in_out(shape: &[usize]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (shape[0], shape[0]),
        2 => (shape[1], shape[0]), // Linear: [out_features, in_features]
        _ => {
            // Convolutional: [out_channels, in_channels, ...]
            let receptive_field_size: usize = shape[2..].iter().product();
            let fan_in = shape[1] * receptive_field_size;
            let fan_out = shape[0] * receptive_field_size;
            (fan_in, fan_out)
        }
    }
}

/// Calculate gain for activation functions.
///
/// Returns the recommended gain value for weight initialization
/// based on the activation function used after the layer.
pub fn calculate_gain(activation: &str) -> f64 {
    match activation.to_lowercase().as_str() {
        "linear" | "identity" | "none" => 1.0,
        "sigmoid" => 1.0,
        "tanh" => 5.0 / 3.0,                // approximately 1.6667
        "relu" => std::f64::consts::SQRT_2, // sqrt(2)
        "leaky_relu" => {
            // For negative_slope = 0.01
            (2.0 / (1.0 + 0.01_f64.powi(2))).sqrt()
        }
        "selu" => 0.75, // 3/4, preserves variance for SELU
        _ => 1.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orthogonal_init_shape() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 1.0, &[64, 32], &device).unwrap();
        let ortho = orthogonal_init(&weight, 1.0).unwrap();

        assert_eq!(ortho.dims(), &[64, 32]);
    }

    #[test]
    fn test_orthogonal_init_orthogonality() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 1.0, &[32, 32], &device).unwrap();
        let ortho = orthogonal_init(&weight, 1.0).unwrap();

        // For a square orthogonal matrix Q, Q^T * Q should be approximately I
        let qt = ortho.t().unwrap();
        let qtq = qt.matmul(&ortho).unwrap();

        // Check diagonal is approximately 1
        let diag_sum: f32 = qtq
            .to_vec2::<f32>()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, row)| row[i])
            .sum();

        // Should be close to 32 (sum of 1s on diagonal)
        assert!((diag_sum - 32.0).abs() < 1.0);
    }

    #[test]
    fn test_xavier_uniform_range() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 1.0, &[256, 128], &device).unwrap();
        let xavier = xavier_uniform(&weight, 1.0).unwrap();

        let (fan_in, fan_out) = compute_fan_in_out(&[256, 128]);
        let expected_bound = (6.0 / (fan_in + fan_out) as f64).sqrt() as f32;

        let max_val = xavier
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let min_val = xavier
            .min(0)
            .unwrap()
            .min(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();

        assert!(max_val <= expected_bound * 1.1); // Allow small tolerance
        assert!(min_val >= -expected_bound * 1.1);
    }

    #[test]
    fn test_kaiming_normal_stats() {
        let device = Device::Cpu;
        let weight = Tensor::randn(0.0f32, 1.0, &[512, 256], &device).unwrap();
        let kaiming = kaiming_normal(&weight).unwrap();

        let (fan_in, _) = compute_fan_in_out(&[512, 256]);
        let expected_std = std::f64::consts::SQRT_2 / (fan_in as f64).sqrt();

        // Check mean is close to 0
        let mean = kaiming.mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(mean.abs() < 0.1);

        // Check std is approximately correct (allow 20% tolerance due to randomness)
        let variance = kaiming
            .sqr()
            .unwrap()
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        let actual_std = variance.sqrt();
        assert!((actual_std - expected_std as f32).abs() < expected_std as f32 * 0.3);
    }

    #[test]
    fn test_compute_fan_in_out() {
        // Linear layer
        assert_eq!(compute_fan_in_out(&[256, 128]), (128, 256));

        // Conv2d layer [out_ch, in_ch, kH, kW]
        let (fan_in, fan_out) = compute_fan_in_out(&[64, 32, 3, 3]);
        assert_eq!(fan_in, 32 * 9); // in_channels * kernel_size
        assert_eq!(fan_out, 64 * 9); // out_channels * kernel_size
    }

    #[test]
    fn test_calculate_gain() {
        assert!((calculate_gain("tanh") - 5.0 / 3.0).abs() < 1e-6);
        assert!((calculate_gain("relu") - std::f64::consts::SQRT_2).abs() < 1e-6);
        assert!((calculate_gain("linear") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_init_method() {
        let device = Device::Cpu;
        let shape = [64, 32];

        let ortho = InitMethod::Orthogonal
            .init(&shape, DType::F32, &device, 1.0)
            .unwrap();
        assert_eq!(ortho.dims(), &[64, 32]);

        let zeros = InitMethod::Zeros
            .init(&shape, DType::F32, &device, 1.0)
            .unwrap();
        let sum = zeros.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((sum - 0.0).abs() < 1e-6);

        let const_val = InitMethod::Constant(0.5)
            .init(&shape, DType::F32, &device, 1.0)
            .unwrap();
        let mean = const_val.mean_all().unwrap().to_scalar::<f32>().unwrap();
        assert!((mean - 0.5).abs() < 1e-6);
    }
}
