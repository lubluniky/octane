//! Multi-Layer Perceptron (MLP) implementation.
//!
//! Configurable feedforward neural network with support for various
//! activation functions and layer configurations.

use candle_core::{Tensor, Result as CandleResult};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use candle_core::{DType, Device};

/// Activation function for MLP layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// Rectified Linear Unit: max(0, x)
    ReLU,
    /// Hyperbolic tangent: tanh(x)
    Tanh,
    /// Sigmoid: 1 / (1 + exp(-x))
    Sigmoid,
    /// Gaussian Error Linear Unit
    GELU,
    /// Scaled Exponential Linear Unit (self-normalizing)
    SELU,
    /// Leaky ReLU with alpha=0.01
    LeakyReLU,
    /// No activation (identity)
    None,
}

impl Activation {
    /// Apply the activation function to a tensor.
    pub fn apply(&self, x: &Tensor) -> CandleResult<Tensor> {
        match self {
            Activation::ReLU => x.relu(),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => candle_nn::ops::sigmoid(x),
            Activation::GELU => x.gelu_erf(),
            Activation::SELU => {
                // SELU: scale * (max(0,x) + min(0, alpha * (exp(x) - 1)))
                // alpha = 1.6732632423543772848170429916717
                // scale = 1.0507009873554804934193349852946
                let alpha = 1.6732632423543772f64;
                let scale = 1.0507009873554805f64;
                let positive = x.relu()?;
                let negative = ((x.exp()? - 1.0)? * alpha)?.minimum(&Tensor::zeros_like(x)?)?;
                (positive + negative)? * scale
            }
            Activation::LeakyReLU => {
                // LeakyReLU: max(0, x) + 0.01 * min(0, x)
                let positive = x.relu()?;
                let negative = x.minimum(&Tensor::zeros_like(x)?)? * 0.01;
                positive + negative?
            }
            Activation::None => Ok(x.clone()),
        }
    }
}

impl Default for Activation {
    fn default() -> Self {
        Activation::ReLU
    }
}

/// Configuration for MLP construction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLPConfig {
    /// Input dimension.
    pub input_dim: usize,
    /// Hidden layer dimensions.
    pub hidden_dims: Vec<usize>,
    /// Output dimension.
    pub output_dim: usize,
    /// Activation function for hidden layers.
    pub activation: Activation,
    /// Whether to apply activation to the output layer.
    pub output_activation: Activation,
}

impl MLPConfig {
    /// Create a new MLP configuration.
    pub fn new(input_dim: usize, hidden_dims: Vec<usize>, output_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dims,
            output_dim,
            activation: Activation::ReLU,
            output_activation: Activation::None,
        }
    }

    /// Set the hidden layer activation function.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Set the output layer activation function.
    pub fn with_output_activation(mut self, activation: Activation) -> Self {
        self.output_activation = activation;
        self
    }

    /// Create a standard RL policy network configuration.
    pub fn policy_network(obs_dim: usize, action_dim: usize) -> Self {
        Self::new(obs_dim, vec![256, 256], action_dim)
            .with_activation(Activation::Tanh)
    }

    /// Create a standard RL value network configuration.
    pub fn value_network(obs_dim: usize) -> Self {
        Self::new(obs_dim, vec![256, 256], 1)
            .with_activation(Activation::Tanh)
    }
}

/// Multi-Layer Perceptron neural network.
///
/// A feedforward network with configurable hidden layers and activations.
/// Commonly used as the backbone for actor and critic networks in RL.
#[derive(Debug)]
pub struct MLP {
    layers: Vec<Linear>,
    activation: Activation,
    output_activation: Activation,
}

impl MLP {
    /// Create a new MLP from configuration.
    ///
    /// # Arguments
    /// * `vb` - Variable builder for weight initialization
    /// * `config` - MLP configuration
    ///
    /// # Returns
    /// A new MLP instance.
    pub fn new(vb: VarBuilder<'_>, config: MLPConfig) -> CandleResult<Self> {
        let mut layers = Vec::new();
        let mut in_dim = config.input_dim;

        // Hidden layers
        for (i, &out_dim) in config.hidden_dims.iter().enumerate() {
            let layer = candle_nn::linear(in_dim, out_dim, vb.pp(format!("layer_{}", i)))?;
            layers.push(layer);
            in_dim = out_dim;
        }

        // Output layer
        let output_layer = candle_nn::linear(
            in_dim,
            config.output_dim,
            vb.pp(format!("layer_{}", config.hidden_dims.len())),
        )?;
        layers.push(output_layer);

        Ok(Self {
            layers,
            activation: config.activation,
            output_activation: config.output_activation,
        })
    }

    /// Create a simple MLP with default ReLU activations.
    ///
    /// # Arguments
    /// * `vb` - Variable builder
    /// * `input_dim` - Input feature dimension
    /// * `hidden_dims` - Hidden layer dimensions
    /// * `output_dim` - Output dimension
    pub fn simple(
        vb: VarBuilder<'_>,
        input_dim: usize,
        hidden_dims: Vec<usize>,
        output_dim: usize,
    ) -> CandleResult<Self> {
        let config = MLPConfig::new(input_dim, hidden_dims, output_dim);
        Self::new(vb, config)
    }

    /// Get the number of layers (including output).
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get the output dimension.
    pub fn output_dim(&self) -> usize {
        self.layers.last().map(|l| l.weight().dims()[0]).unwrap_or(0)
    }
}

impl Module for MLP {
    /// Forward pass through the MLP.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, input_dim]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, output_dim]
    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let num_layers = self.layers.len();
        let mut output = x.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            output = layer.forward(&output)?;

            // Apply activation (hidden layers use self.activation, output uses output_activation)
            if i < num_layers - 1 {
                output = self.activation.apply(&output)?;
            } else {
                output = self.output_activation.apply(&output)?;
            }
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_mlp_forward() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = MLPConfig::new(4, vec![32, 32], 2);
        let mlp = MLP::new(vb, config).unwrap();

        let input = Tensor::randn(0.0f32, 1.0, &[8, 4], &device).unwrap();
        let output = mlp.forward(&input).unwrap();

        assert_eq!(output.dims(), &[8, 2]);
    }

    #[test]
    fn test_mlp_activations() {
        let device = Device::Cpu;
        let x = Tensor::from_slice(&[-1.0f32, 0.0, 1.0], &[3], &device).unwrap();

        // Test ReLU
        let relu_out = Activation::ReLU.apply(&x).unwrap();
        let relu_data: Vec<f32> = relu_out.to_vec1().unwrap();
        assert_eq!(relu_data[0], 0.0);
        assert_eq!(relu_data[1], 0.0);
        assert_eq!(relu_data[2], 1.0);

        // Test Tanh (should be between -1 and 1)
        let tanh_out = Activation::Tanh.apply(&x).unwrap();
        let tanh_data: Vec<f32> = tanh_out.to_vec1().unwrap();
        assert!(tanh_data[0] < 0.0);
        assert_eq!(tanh_data[1], 0.0);
        assert!(tanh_data[2] > 0.0);
    }
}
