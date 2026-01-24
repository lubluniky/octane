//! Tensor backend abstraction over Candle.

use crate::core::{Device, Result};
use candle_core::{DType, Tensor};

/// Tensor backend trait for generic tensor operations.
pub trait TensorBackend: Sized {
    /// Create a new tensor filled with zeros.
    fn zeros(shape: &[usize], device: &Device) -> Result<Self>;

    /// Create a new tensor filled with ones.
    fn ones(shape: &[usize], device: &Device) -> Result<Self>;

    /// Create a tensor from a slice of f32 values.
    fn from_slice(data: &[f32], shape: &[usize], device: &Device) -> Result<Self>;

    /// Get the shape of this tensor.
    fn shape(&self) -> &[usize];

    /// Move tensor to a different device.
    fn to_device(&self, device: &Device) -> Result<Self>;

    /// Convert to f32 Vec (for debugging/logging).
    fn to_vec(&self) -> Result<Vec<f32>>;
}

impl TensorBackend for Tensor {
    fn zeros(shape: &[usize], device: &Device) -> Result<Self> {
        let candle_device = device.to_candle()?;
        Ok(Tensor::zeros(shape, DType::F32, &candle_device)?)
    }

    fn ones(shape: &[usize], device: &Device) -> Result<Self> {
        let candle_device = device.to_candle()?;
        Ok(Tensor::ones(shape, DType::F32, &candle_device)?)
    }

    fn from_slice(data: &[f32], shape: &[usize], device: &Device) -> Result<Self> {
        let candle_device = device.to_candle()?;
        let tensor = Tensor::from_slice(data, shape, &candle_device)?;
        Ok(tensor)
    }

    fn shape(&self) -> &[usize] {
        self.dims()
    }

    fn to_device(&self, device: &Device) -> Result<Self> {
        let candle_device = device.to_candle()?;
        Ok(self.to_device(&candle_device)?)
    }

    fn to_vec(&self) -> Result<Vec<f32>> {
        Ok(self.flatten_all()?.to_vec1()?)
    }
}

/// Utility functions for tensor operations.
pub mod ops {
    use super::*;

    /// Compute softmax along the last dimension.
    pub fn softmax(tensor: &Tensor, dim: usize) -> Result<Tensor> {
        Ok(candle_nn::ops::softmax(tensor, dim)?)
    }

    /// Compute log-softmax along the last dimension.
    pub fn log_softmax(tensor: &Tensor, dim: usize) -> Result<Tensor> {
        Ok(candle_nn::ops::log_softmax(tensor, dim)?)
    }

    /// Clamp tensor values between min and max.
    pub fn clamp(tensor: &Tensor, min: f32, max: f32) -> Result<Tensor> {
        Ok(tensor.clamp(min, max)?)
    }

    /// Compute mean along specified dimensions.
    pub fn mean(tensor: &Tensor, dims: &[usize]) -> Result<Tensor> {
        let mut result = tensor.clone();
        for &dim in dims.iter().rev() {
            result = result.mean(dim)?;
        }
        Ok(result)
    }

    /// Standard deviation along dimension.
    pub fn std(tensor: &Tensor, dim: usize) -> Result<Tensor> {
        let mean = tensor.mean(dim)?;
        let diff = tensor.broadcast_sub(&mean)?;
        let sq = diff.sqr()?;
        let var = sq.mean(dim)?;
        Ok(var.sqrt()?)
    }

    /// Normalize tensor to zero mean, unit variance.
    pub fn normalize(tensor: &Tensor, dim: usize, eps: f32) -> Result<Tensor> {
        let mean = tensor.mean(dim)?;
        let std = std(tensor, dim)?;
        let std_eps = (std + eps as f64)?;
        let centered = tensor.broadcast_sub(&mean)?;
        Ok(centered.broadcast_div(&std_eps)?)
    }
}
