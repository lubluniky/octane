//! Action and observation space definitions.

use crate::core::{Device, Result, RocketError};
use candle_core::Tensor;
use rand::Rng;
use serde::{Deserialize, Serialize};

/// Trait for action/observation spaces.
pub trait Space: Clone + Send + Sync {
    /// Shape of a single sample from this space.
    fn shape(&self) -> &[usize];

    /// Total number of elements in a single sample.
    fn flat_dim(&self) -> usize {
        self.shape().iter().product()
    }

    /// Sample a random element from this space.
    fn sample(&self, rng: &mut impl Rng, device: &Device) -> Result<Tensor>;

    /// Check if a tensor is a valid member of this space.
    fn contains(&self, tensor: &Tensor) -> Result<bool>;
}

/// Continuous box space with bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxSpace {
    /// Lower bounds for each dimension.
    pub low: Vec<f32>,
    /// Upper bounds for each dimension.
    pub high: Vec<f32>,
    /// Shape of the space.
    pub shape: Vec<usize>,
}

impl BoxSpace {
    /// Create a new box space.
    pub fn new(low: Vec<f32>, high: Vec<f32>, shape: Vec<usize>) -> Result<Self> {
        if low.len() != high.len() {
            return Err(RocketError::InvalidConfig(
                "Low and high bounds must have same length".to_string(),
            ));
        }
        let flat_dim: usize = shape.iter().product();
        if low.len() != flat_dim {
            return Err(RocketError::InvalidConfig(format!(
                "Bounds length {} doesn't match shape {:?}",
                low.len(),
                shape
            )));
        }
        Ok(Self { low, high, shape })
    }

    /// Create a symmetric box space centered at zero.
    pub fn symmetric(bound: f32, shape: Vec<usize>) -> Self {
        let flat_dim: usize = shape.iter().product();
        Self {
            low: vec![-bound; flat_dim],
            high: vec![bound; flat_dim],
            shape,
        }
    }

    /// Create an unbounded box space (using large values).
    pub fn unbounded(shape: Vec<usize>) -> Self {
        let flat_dim: usize = shape.iter().product();
        Self {
            low: vec![f32::NEG_INFINITY; flat_dim],
            high: vec![f32::INFINITY; flat_dim],
            shape,
        }
    }
}

impl Space for BoxSpace {
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn sample(&self, rng: &mut impl Rng, device: &Device) -> Result<Tensor> {
        let data: Vec<f32> = self
            .low
            .iter()
            .zip(&self.high)
            .map(|(&lo, &hi)| {
                if lo.is_infinite() || hi.is_infinite() {
                    rng.gen::<f32>() * 2.0 - 1.0 // Standard normal-ish for unbounded
                } else {
                    rng.gen::<f32>() * (hi - lo) + lo
                }
            })
            .collect();

        let candle_device = device.to_candle()?;
        Ok(Tensor::from_slice(
            &data,
            self.shape.as_slice(),
            &candle_device,
        )?)
    }

    fn contains(&self, tensor: &Tensor) -> Result<bool> {
        if tensor.dims() != self.shape.as_slice() {
            return Ok(false);
        }
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        Ok(data
            .iter()
            .zip(self.low.iter().zip(&self.high))
            .all(|(&val, (&lo, &hi))| val >= lo && val <= hi))
    }
}

/// Discrete space with n possible values [0, n).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscreteSpace {
    /// Number of discrete actions.
    pub n: usize,
}

impl DiscreteSpace {
    /// Create a new discrete space with n actions.
    pub fn new(n: usize) -> Self {
        Self { n }
    }
}

impl Space for DiscreteSpace {
    fn shape(&self) -> &[usize] {
        &[1]
    }

    fn flat_dim(&self) -> usize {
        self.n
    }

    fn sample(&self, rng: &mut impl Rng, device: &Device) -> Result<Tensor> {
        let action = rng.gen_range(0..self.n) as f32;
        let candle_device = device.to_candle()?;
        Ok(Tensor::from_slice(&[action], &[1], &candle_device)?)
    }

    fn contains(&self, tensor: &Tensor) -> Result<bool> {
        let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
        if data.len() != 1 {
            return Ok(false);
        }
        let val = data[0] as usize;
        Ok(val < self.n)
    }
}
