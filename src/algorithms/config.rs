//! Configuration structs for RL algorithms.

use serde::{Deserialize, Serialize};

/// Configuration for Proximal Policy Optimization (PPO) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PPOConfig {
    /// Learning rate for optimizer.
    /// Default: 3e-4
    pub learning_rate: f32,

    /// Number of steps to collect before each policy update.
    /// Default: 2048
    pub n_steps: usize,

    /// Minibatch size for gradient updates.
    /// Default: 64
    pub batch_size: usize,

    /// Number of epochs for each policy update.
    /// Default: 10
    pub n_epochs: usize,

    /// Discount factor for future rewards.
    /// Default: 0.99
    pub gamma: f32,

    /// GAE lambda parameter for advantage estimation.
    /// Default: 0.95
    pub gae_lambda: f32,

    /// Clipping range for surrogate objective.
    /// Default: 0.2
    pub clip_range: f32,

    /// Value function loss coefficient.
    /// Default: 0.5
    pub vf_coef: f32,

    /// Entropy bonus coefficient for exploration.
    /// Default: 0.01
    pub ent_coef: f32,

    /// Maximum gradient norm for clipping.
    /// Default: 0.5
    pub max_grad_norm: f32,

    /// Whether to normalize advantages.
    /// Default: true
    pub normalize_advantage: bool,

    /// Target KL divergence for early stopping (optional).
    /// Default: None (no early stopping)
    pub target_kl: Option<f32>,

    /// Use linear learning rate schedule.
    /// Default: true
    pub use_lr_schedule: bool,

    /// Random seed for reproducibility.
    /// Default: None (use system entropy)
    pub seed: Option<u64>,
}

impl Default for PPOConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            n_steps: 2048,
            batch_size: 64,
            n_epochs: 10,
            gamma: 0.99,
            gae_lambda: 0.95,
            clip_range: 0.2,
            vf_coef: 0.5,
            ent_coef: 0.01,
            max_grad_norm: 0.5,
            normalize_advantage: true,
            target_kl: None,
            use_lr_schedule: true,
            seed: None,
        }
    }
}

impl PPOConfig {
    /// Create a new PPO config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style setter for learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder-style setter for n_steps.
    pub fn n_steps(mut self, n: usize) -> Self {
        self.n_steps = n;
        self
    }

    /// Builder-style setter for batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Builder-style setter for n_epochs.
    pub fn n_epochs(mut self, n: usize) -> Self {
        self.n_epochs = n;
        self
    }

    /// Builder-style setter for gamma.
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Builder-style setter for GAE lambda.
    pub fn gae_lambda(mut self, l: f32) -> Self {
        self.gae_lambda = l;
        self
    }

    /// Builder-style setter for clip range.
    pub fn clip_range(mut self, c: f32) -> Self {
        self.clip_range = c;
        self
    }

    /// Builder-style setter for value function coefficient.
    pub fn vf_coef(mut self, c: f32) -> Self {
        self.vf_coef = c;
        self
    }

    /// Builder-style setter for entropy coefficient.
    pub fn ent_coef(mut self, c: f32) -> Self {
        self.ent_coef = c;
        self
    }

    /// Builder-style setter for max gradient norm.
    pub fn max_grad_norm(mut self, n: f32) -> Self {
        self.max_grad_norm = n;
        self
    }

    /// Builder-style setter for target KL divergence.
    pub fn target_kl(mut self, kl: Option<f32>) -> Self {
        self.target_kl = kl;
        self
    }

    /// Builder-style setter for seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".to_string());
        }
        if self.n_steps == 0 {
            return Err("n_steps must be positive".to_string());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".to_string());
        }
        if self.batch_size > self.n_steps {
            return Err("batch_size cannot exceed n_steps".to_string());
        }
        if self.n_epochs == 0 {
            return Err("n_epochs must be positive".to_string());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.gae_lambda) {
            return Err("gae_lambda must be in [0, 1]".to_string());
        }
        if self.clip_range <= 0.0 {
            return Err("clip_range must be positive".to_string());
        }
        Ok(())
    }
}

/// Configuration for Advantage Actor-Critic (A2C) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2CConfig {
    /// Learning rate for optimizer.
    /// Default: 7e-4
    pub learning_rate: f32,

    /// Number of steps to collect before each update.
    /// Default: 5
    pub n_steps: usize,

    /// Discount factor for future rewards.
    /// Default: 0.99
    pub gamma: f32,

    /// GAE lambda parameter for advantage estimation.
    /// Default: 1.0 (no GAE, pure Monte Carlo)
    pub gae_lambda: f32,

    /// Value function loss coefficient.
    /// Default: 0.5
    pub vf_coef: f32,

    /// Entropy bonus coefficient for exploration.
    /// Default: 0.01
    pub ent_coef: f32,

    /// Maximum gradient norm for clipping.
    /// Default: 0.5
    pub max_grad_norm: f32,

    /// Whether to normalize advantages.
    /// Default: false
    pub normalize_advantage: bool,

    /// RMSprop epsilon for numerical stability.
    /// Default: 1e-5
    pub rms_prop_eps: f32,

    /// Random seed for reproducibility.
    /// Default: None
    pub seed: Option<u64>,
}

impl Default for A2CConfig {
    fn default() -> Self {
        Self {
            learning_rate: 7e-4,
            n_steps: 5,
            gamma: 0.99,
            gae_lambda: 1.0,
            vf_coef: 0.5,
            ent_coef: 0.01,
            max_grad_norm: 0.5,
            normalize_advantage: false,
            rms_prop_eps: 1e-5,
            seed: None,
        }
    }
}

impl A2CConfig {
    /// Create a new A2C config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder-style setter for learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Builder-style setter for n_steps.
    pub fn n_steps(mut self, n: usize) -> Self {
        self.n_steps = n;
        self
    }

    /// Builder-style setter for gamma.
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Builder-style setter for GAE lambda.
    pub fn gae_lambda(mut self, l: f32) -> Self {
        self.gae_lambda = l;
        self
    }

    /// Builder-style setter for value function coefficient.
    pub fn vf_coef(mut self, c: f32) -> Self {
        self.vf_coef = c;
        self
    }

    /// Builder-style setter for entropy coefficient.
    pub fn ent_coef(mut self, c: f32) -> Self {
        self.ent_coef = c;
        self
    }

    /// Builder-style setter for max gradient norm.
    pub fn max_grad_norm(mut self, n: f32) -> Self {
        self.max_grad_norm = n;
        self
    }

    /// Builder-style setter for seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".to_string());
        }
        if self.n_steps == 0 {
            return Err("n_steps must be positive".to_string());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".to_string());
        }
        if !(0.0..=1.0).contains(&self.gae_lambda) {
            return Err("gae_lambda must be in [0, 1]".to_string());
        }
        Ok(())
    }
}

/// Network architecture configuration shared by algorithms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Hidden layer sizes for the policy network.
    pub policy_layers: Vec<usize>,

    /// Hidden layer sizes for the value network.
    /// If None, shares layers with policy.
    pub value_layers: Option<Vec<usize>>,

    /// Activation function.
    pub activation: Activation,

    /// Whether to use orthogonal initialization.
    pub ortho_init: bool,

    /// Initial log standard deviation for continuous actions.
    pub log_std_init: f32,
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            policy_layers: vec![64, 64],
            value_layers: None, // Shared
            activation: Activation::Tanh,
            ortho_init: true,
            log_std_init: 0.0,
        }
    }
}

/// Activation function types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    /// Hyperbolic tangent.
    Tanh,
    /// Rectified linear unit.
    ReLU,
    /// Exponential linear unit.
    ELU,
    /// Leaky ReLU.
    LeakyReLU,
    /// Gaussian Error Linear Unit.
    GELU,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ppo_config_defaults() {
        let config = PPOConfig::default();
        assert!((config.learning_rate - 3e-4).abs() < 1e-8);
        assert_eq!(config.n_steps, 2048);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.n_epochs, 10);
        assert!((config.gamma - 0.99).abs() < 1e-8);
        assert!((config.clip_range - 0.2).abs() < 1e-8);
    }

    #[test]
    fn test_ppo_config_builder() {
        let config = PPOConfig::new()
            .learning_rate(1e-3)
            .n_steps(1024)
            .batch_size(32)
            .gamma(0.95);

        assert!((config.learning_rate - 1e-3).abs() < 1e-8);
        assert_eq!(config.n_steps, 1024);
        assert_eq!(config.batch_size, 32);
        assert!((config.gamma - 0.95).abs() < 1e-8);
    }

    #[test]
    fn test_ppo_config_validation() {
        let config = PPOConfig::default();
        assert!(config.validate().is_ok());

        let invalid = PPOConfig::default().learning_rate(-0.1);
        assert!(invalid.validate().is_err());

        let invalid_batch = PPOConfig::default().batch_size(10000);
        assert!(invalid_batch.validate().is_err());
    }

    #[test]
    fn test_a2c_config_defaults() {
        let config = A2CConfig::default();
        assert!((config.learning_rate - 7e-4).abs() < 1e-8);
        assert_eq!(config.n_steps, 5);
        assert!((config.gae_lambda - 1.0).abs() < 1e-8);
    }
}
