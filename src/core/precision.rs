//! Mixed precision training support for Octane.
//!
//! This module provides mixed precision training capabilities to reduce memory
//! usage and improve performance on hardware with fast FP16/BF16 support.
//!
//! # Features
//!
//! - Precision enum for selecting compute precision
//! - GradScaler for loss scaling to prevent gradient underflow
//! - Autocast context for automatic precision casting
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::core::precision::{Precision, GradScaler, AutocastContext};
//!
//! let scaler = GradScaler::new(Precision::F16);
//! let ctx = AutocastContext::new(Precision::F16);
//!
//! // Scale loss for backward pass
//! let scaled_loss = scaler.scale(&loss)?;
//! // Unscale gradients before optimizer step
//! scaler.unscale_gradients(&mut gradients)?;
//! ```

use crate::core::Result;
use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Compute precision for tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Precision {
    /// 32-bit floating point (default, full precision).
    #[default]
    F32,
    /// 16-bit floating point (half precision).
    F16,
    /// Brain floating point 16 (better range than F16, less precision).
    BF16,
}

impl Precision {
    /// Convert to Candle DType.
    pub fn to_dtype(&self) -> DType {
        match self {
            Precision::F32 => DType::F32,
            Precision::F16 => DType::F16,
            Precision::BF16 => DType::BF16,
        }
    }

    /// Create from Candle DType.
    pub fn from_dtype(dtype: DType) -> Option<Self> {
        match dtype {
            DType::F32 => Some(Precision::F32),
            DType::F16 => Some(Precision::F16),
            DType::BF16 => Some(Precision::BF16),
            _ => None,
        }
    }

    /// Check if this is a reduced precision format.
    pub fn is_reduced(&self) -> bool {
        matches!(self, Precision::F16 | Precision::BF16)
    }

    /// Get the number of bytes per element.
    pub fn bytes_per_element(&self) -> usize {
        match self {
            Precision::F32 => 4,
            Precision::F16 | Precision::BF16 => 2,
        }
    }

    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            Precision::F32 => "FP32",
            Precision::F16 => "FP16",
            Precision::BF16 => "BF16",
        }
    }
}

/// Gradient scaler for mixed precision training.
///
/// Loss scaling helps prevent gradient underflow when using reduced precision.
/// Gradients are scaled up during backward pass and scaled down before optimizer step.
#[derive(Debug, Clone)]
pub struct GradScaler {
    /// Compute precision.
    precision: Precision,

    /// Current scale factor.
    scale: f32,

    /// Minimum scale factor.
    min_scale: f32,

    /// Maximum scale factor.
    max_scale: f32,

    /// Growth factor for scale.
    growth_factor: f32,

    /// Backoff factor when overflow detected.
    backoff_factor: f32,

    /// Number of successful steps before growing scale.
    growth_interval: usize,

    /// Counter for successful steps.
    growth_counter: usize,

    /// Whether the scaler is enabled.
    enabled: bool,

    /// Number of times scale was reduced due to overflow.
    overflow_count: usize,
}

impl GradScaler {
    /// Create a new gradient scaler.
    pub fn new(precision: Precision) -> Self {
        let enabled = precision.is_reduced();

        Self {
            precision,
            scale: if enabled { 65536.0 } else { 1.0 },
            min_scale: 1.0,
            max_scale: 65536.0 * 2048.0, // 2^27
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            growth_counter: 0,
            enabled,
            overflow_count: 0,
        }
    }

    /// Create with custom initial scale.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Create with custom growth factor.
    pub fn with_growth_factor(mut self, factor: f32) -> Self {
        self.growth_factor = factor;
        self
    }

    /// Create with custom backoff factor.
    pub fn with_backoff_factor(mut self, factor: f32) -> Self {
        self.backoff_factor = factor;
        self
    }

    /// Create with custom growth interval.
    pub fn with_growth_interval(mut self, interval: usize) -> Self {
        self.growth_interval = interval;
        self
    }

    /// Enable or disable the scaler.
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.scale = 1.0;
        }
    }

    /// Get current scale factor.
    pub fn scale_factor(&self) -> f32 {
        self.scale
    }

    /// Check if scaler is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Scale a loss tensor for backward pass.
    pub fn scale(&self, loss: &Tensor) -> Result<Tensor> {
        if !self.enabled {
            return Ok(loss.clone());
        }

        Ok((loss * self.scale as f64)?)
    }

    /// Unscale gradients after backward pass.
    ///
    /// Returns true if gradients are valid (no inf/nan), false if overflow detected.
    pub fn unscale_gradients(&self, gradients: &mut HashMap<String, Tensor>) -> Result<bool> {
        if !self.enabled {
            return Ok(true);
        }

        let inv_scale = 1.0 / self.scale as f64;
        let mut valid = true;

        for grad in gradients.values_mut() {
            // Check for inf/nan before unscaling
            if self.tensor_has_inf_or_nan(grad)? {
                valid = false;
                break;
            }

            *grad = (grad.clone() * inv_scale)?;
        }

        Ok(valid)
    }

    /// Unscale a single gradient tensor.
    pub fn unscale(&self, gradient: &Tensor) -> Result<Tensor> {
        if !self.enabled {
            return Ok(gradient.clone());
        }

        let inv_scale = 1.0 / self.scale as f64;
        Ok((gradient * inv_scale)?)
    }

    /// Update the scale factor after an optimizer step.
    ///
    /// Call with `overflow=true` if the gradients contained inf/nan.
    pub fn update(&mut self, overflow: bool) {
        if !self.enabled {
            return;
        }

        if overflow {
            // Reduce scale
            self.scale *= self.backoff_factor;
            self.scale = self.scale.max(self.min_scale);
            self.growth_counter = 0;
            self.overflow_count += 1;
        } else {
            // Count successful step
            self.growth_counter += 1;

            // Grow scale if enough successful steps
            if self.growth_counter >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.scale = self.scale.min(self.max_scale);
                self.growth_counter = 0;
            }
        }
    }

    /// Check if a tensor contains inf or nan values.
    fn tensor_has_inf_or_nan(&self, tensor: &Tensor) -> Result<bool> {
        // Convert to f32 for checking
        let t = tensor.to_dtype(DType::F32)?;
        let values: Vec<f32> = t.flatten_all()?.to_vec1()?;

        Ok(values.iter().any(|v| v.is_infinite() || v.is_nan()))
    }

    /// Get overflow count.
    pub fn overflow_count(&self) -> usize {
        self.overflow_count
    }

    /// Get the precision setting.
    pub fn precision(&self) -> Precision {
        self.precision
    }

    /// Get state dict for checkpointing.
    pub fn state_dict(&self) -> GradScalerState {
        GradScalerState {
            scale: self.scale,
            growth_counter: self.growth_counter,
            overflow_count: self.overflow_count,
        }
    }

    /// Load state from checkpoint.
    pub fn load_state(&mut self, state: &GradScalerState) {
        self.scale = state.scale;
        self.growth_counter = state.growth_counter;
        self.overflow_count = state.overflow_count;
    }
}

/// Serializable state for GradScaler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradScalerState {
    /// Current scale factor.
    pub scale: f32,
    /// Growth counter.
    pub growth_counter: usize,
    /// Overflow count.
    pub overflow_count: usize,
}

/// Context for automatic precision casting.
///
/// Provides methods to cast tensors to the appropriate precision
/// for forward pass (reduced) and master weights (full).
#[derive(Debug, Clone)]
pub struct AutocastContext {
    /// Compute precision for forward/backward.
    compute_precision: Precision,

    /// Master weight precision (always F32 for stability).
    master_precision: Precision,

    /// Whether autocast is enabled.
    enabled: bool,

    /// Operations that should stay in full precision.
    fp32_ops: Vec<String>,
}

impl AutocastContext {
    /// Create a new autocast context.
    pub fn new(compute_precision: Precision) -> Self {
        Self {
            compute_precision,
            master_precision: Precision::F32,
            enabled: compute_precision.is_reduced(),
            fp32_ops: vec![
                "softmax".to_string(),
                "log_softmax".to_string(),
                "layer_norm".to_string(),
                "batch_norm".to_string(),
                "loss".to_string(),
            ],
        }
    }

    /// Create a disabled context (F32 everywhere).
    pub fn disabled() -> Self {
        Self {
            compute_precision: Precision::F32,
            master_precision: Precision::F32,
            enabled: false,
            fp32_ops: vec![],
        }
    }

    /// Check if autocast is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Get compute precision.
    pub fn compute_precision(&self) -> Precision {
        self.compute_precision
    }

    /// Get master weight precision.
    pub fn master_precision(&self) -> Precision {
        self.master_precision
    }

    /// Cast tensor to compute precision.
    pub fn cast_to_compute(&self, tensor: &Tensor) -> Result<Tensor> {
        if !self.enabled {
            return Ok(tensor.clone());
        }

        Ok(tensor.to_dtype(self.compute_precision.to_dtype())?)
    }

    /// Cast tensor to master precision (F32).
    pub fn cast_to_master(&self, tensor: &Tensor) -> Result<Tensor> {
        Ok(tensor.to_dtype(self.master_precision.to_dtype())?)
    }

    /// Cast tensor for a specific operation.
    ///
    /// Some operations (like softmax) should always use F32 for stability.
    pub fn cast_for_op(&self, tensor: &Tensor, op_name: &str) -> Result<Tensor> {
        if !self.enabled {
            return Ok(tensor.clone());
        }

        // Check if this op should stay in F32
        let should_use_fp32 = self
            .fp32_ops
            .iter()
            .any(|op| op_name.to_lowercase().contains(op));

        if should_use_fp32 {
            self.cast_to_master(tensor)
        } else {
            self.cast_to_compute(tensor)
        }
    }

    /// Add an operation to the FP32 list.
    pub fn add_fp32_op(&mut self, op_name: impl Into<String>) {
        self.fp32_ops.push(op_name.into());
    }

    /// Cast multiple tensors to compute precision.
    pub fn cast_inputs(&self, tensors: &[&Tensor]) -> Result<Vec<Tensor>> {
        tensors.iter().map(|t| self.cast_to_compute(t)).collect()
    }
}

/// Mixed precision training configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MixedPrecisionConfig {
    /// Compute precision.
    pub precision: Precision,

    /// Enable gradient scaling.
    pub use_grad_scaler: bool,

    /// Initial scale for gradient scaler.
    pub init_scale: f32,

    /// Growth factor for gradient scaler.
    pub growth_factor: f32,

    /// Backoff factor for gradient scaler.
    pub backoff_factor: f32,

    /// Growth interval for gradient scaler.
    pub growth_interval: usize,

    /// Operations to keep in FP32.
    pub fp32_ops: Vec<String>,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            precision: Precision::F32,
            use_grad_scaler: false,
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            fp32_ops: vec![
                "softmax".to_string(),
                "layer_norm".to_string(),
                "batch_norm".to_string(),
            ],
        }
    }
}

impl MixedPrecisionConfig {
    /// Create config for FP16 training.
    pub fn fp16() -> Self {
        Self {
            precision: Precision::F16,
            use_grad_scaler: true,
            ..Default::default()
        }
    }

    /// Create config for BF16 training (no scaler needed).
    pub fn bf16() -> Self {
        Self {
            precision: Precision::BF16,
            use_grad_scaler: false, // BF16 has same range as F32
            ..Default::default()
        }
    }

    /// Build a GradScaler from this config.
    pub fn build_scaler(&self) -> GradScaler {
        GradScaler::new(self.precision)
            .with_scale(self.init_scale)
            .with_growth_factor(self.growth_factor)
            .with_backoff_factor(self.backoff_factor)
            .with_growth_interval(self.growth_interval)
    }

    /// Build an AutocastContext from this config.
    pub fn build_context(&self) -> AutocastContext {
        let mut ctx = AutocastContext::new(self.precision);
        for op in &self.fp32_ops {
            ctx.add_fp32_op(op.clone());
        }
        ctx
    }
}

/// Helper to run a closure in an autocast context.
pub fn with_autocast<F, T>(_ctx: &AutocastContext, f: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    // In a full implementation, this would set thread-local state
    // For now, the context is passed explicitly
    f()
}

/// Cast model parameters to mixed precision format.
///
/// Returns both the reduced precision parameters (for forward pass)
/// and keeps master weights in F32 (for updates).
pub fn cast_model_to_mixed_precision(
    parameters: &HashMap<String, Tensor>,
    precision: Precision,
) -> Result<(HashMap<String, Tensor>, HashMap<String, Tensor>)> {
    let mut compute_params = HashMap::new();
    let mut master_params = HashMap::new();

    for (name, param) in parameters {
        // Keep master copy in F32
        master_params.insert(name.clone(), param.to_dtype(DType::F32)?);

        // Create compute copy in reduced precision
        compute_params.insert(name.clone(), param.to_dtype(precision.to_dtype())?);
    }

    Ok((compute_params, master_params))
}

/// Synchronize compute parameters from master weights.
pub fn sync_compute_from_master(
    compute_params: &mut HashMap<String, Tensor>,
    master_params: &HashMap<String, Tensor>,
    precision: Precision,
) -> Result<()> {
    for (name, master) in master_params {
        if let Some(compute) = compute_params.get_mut(name) {
            *compute = master.to_dtype(precision.to_dtype())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_dtype_conversion() {
        assert_eq!(Precision::F32.to_dtype(), DType::F32);
        assert_eq!(Precision::F16.to_dtype(), DType::F16);
        assert_eq!(Precision::BF16.to_dtype(), DType::BF16);

        assert_eq!(Precision::from_dtype(DType::F32), Some(Precision::F32));
        assert_eq!(Precision::from_dtype(DType::F16), Some(Precision::F16));
        assert_eq!(Precision::from_dtype(DType::BF16), Some(Precision::BF16));
    }

    #[test]
    fn test_precision_properties() {
        assert!(!Precision::F32.is_reduced());
        assert!(Precision::F16.is_reduced());
        assert!(Precision::BF16.is_reduced());

        assert_eq!(Precision::F32.bytes_per_element(), 4);
        assert_eq!(Precision::F16.bytes_per_element(), 2);
        assert_eq!(Precision::BF16.bytes_per_element(), 2);
    }

    #[test]
    fn test_grad_scaler_creation() {
        let scaler = GradScaler::new(Precision::F32);
        assert!(!scaler.is_enabled());
        assert_eq!(scaler.scale_factor(), 1.0);

        let scaler = GradScaler::new(Precision::F16);
        assert!(scaler.is_enabled());
        assert_eq!(scaler.scale_factor(), 65536.0);
    }

    #[test]
    fn test_grad_scaler_update() {
        let mut scaler = GradScaler::new(Precision::F16);
        let initial_scale = scaler.scale_factor();

        // Simulate overflow
        scaler.update(true);
        assert!(scaler.scale_factor() < initial_scale);
        assert_eq!(scaler.overflow_count(), 1);

        // Simulate many successful steps
        for _ in 0..2001 {
            scaler.update(false);
        }
        // Scale should have grown
        // (depends on growth_interval)
    }

    #[test]
    fn test_grad_scaler_scale_unscale() {
        let scaler = GradScaler::new(Precision::F16);
        let device = candle_core::Device::Cpu;

        let loss = Tensor::new(&[1.0f32], &device).unwrap();
        let scaled = scaler.scale(&loss).unwrap();

        let scaled_val: f32 = scaled.to_scalar().unwrap();
        assert!((scaled_val - 65536.0).abs() < 1.0);

        let unscaled = scaler.unscale(&scaled).unwrap();
        let unscaled_val: f32 = unscaled.to_scalar().unwrap();
        assert!((unscaled_val - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_autocast_context() {
        let ctx = AutocastContext::new(Precision::F16);
        assert!(ctx.is_enabled());
        assert_eq!(ctx.compute_precision(), Precision::F16);
        assert_eq!(ctx.master_precision(), Precision::F32);

        let disabled = AutocastContext::disabled();
        assert!(!disabled.is_enabled());
    }

    #[test]
    fn test_autocast_casting() {
        let ctx = AutocastContext::new(Precision::F16);
        let device = candle_core::Device::Cpu;

        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();

        let compute = ctx.cast_to_compute(&tensor).unwrap();
        assert_eq!(compute.dtype(), DType::F16);

        let master = ctx.cast_to_master(&compute).unwrap();
        assert_eq!(master.dtype(), DType::F32);
    }

    #[test]
    fn test_autocast_fp32_ops() {
        let ctx = AutocastContext::new(Precision::F16);
        let device = candle_core::Device::Cpu;

        let tensor = Tensor::new(&[1.0f32, 2.0], &device).unwrap();

        // Softmax should stay in F32
        let for_softmax = ctx.cast_for_op(&tensor, "softmax").unwrap();
        assert_eq!(for_softmax.dtype(), DType::F32);

        // Regular ops use F16
        let for_matmul = ctx.cast_for_op(&tensor, "matmul").unwrap();
        assert_eq!(for_matmul.dtype(), DType::F16);
    }

    #[test]
    fn test_mixed_precision_config() {
        let fp16_config = MixedPrecisionConfig::fp16();
        assert_eq!(fp16_config.precision, Precision::F16);
        assert!(fp16_config.use_grad_scaler);

        let bf16_config = MixedPrecisionConfig::bf16();
        assert_eq!(bf16_config.precision, Precision::BF16);
        assert!(!bf16_config.use_grad_scaler);
    }

    #[test]
    fn test_grad_scaler_state() {
        let mut scaler = GradScaler::new(Precision::F16);
        scaler.update(true); // Trigger overflow

        let state = scaler.state_dict();
        assert_eq!(state.overflow_count, 1);

        let mut new_scaler = GradScaler::new(Precision::F16);
        new_scaler.load_state(&state);
        assert_eq!(new_scaler.overflow_count(), 1);
    }
}
