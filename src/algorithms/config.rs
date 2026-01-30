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

/// Configuration for Deep Q-Network (DQN) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DQNConfig {
    /// Learning rate for optimizer.
    pub learning_rate: f32,
    /// Replay buffer size.
    pub buffer_size: usize,
    /// Number of timesteps before learning starts.
    pub learning_starts: usize,
    /// Minibatch size.
    pub batch_size: usize,
    /// Discount factor.
    pub gamma: f32,
    /// Soft update coefficient (tau=1.0 for hard update).
    pub tau: f32,
    /// Target network update interval.
    pub target_update_interval: usize,
    /// Training frequency (update every N steps).
    pub train_freq: usize,
    /// Gradient steps per update.
    pub gradient_steps: usize,
    /// Initial exploration rate.
    pub epsilon_start: f32,
    /// Final exploration rate.
    pub epsilon_end: f32,
    /// Exploration decay per step.
    pub epsilon_decay: f32,
    /// Use Double DQN.
    pub double_dqn: bool,
    /// Use prioritized experience replay.
    pub prioritized_replay: bool,
    /// Use Huber loss instead of MSE.
    pub use_huber_loss: bool,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            buffer_size: 1_000_000,
            learning_starts: 50_000,
            batch_size: 32,
            gamma: 0.99,
            tau: 1.0,
            target_update_interval: 10_000,
            train_freq: 4,
            gradient_steps: 1,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay: 1e-5,
            double_dqn: true,
            prioritized_replay: false,
            use_huber_loss: true,
            seed: None,
        }
    }
}

impl DQNConfig {
    /// Create new DQN config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set gamma.
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Enable/disable Double DQN.
    pub fn double_dqn(mut self, enabled: bool) -> Self {
        self.double_dqn = enabled;
        self
    }

    /// Enable prioritized replay.
    pub fn prioritized_replay(mut self, enabled: bool) -> Self {
        self.prioritized_replay = enabled;
        self
    }

    /// Set seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".into());
        }
        if self.buffer_size == 0 {
            return Err("buffer_size must be positive".into());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".into());
        }
        Ok(())
    }
}

/// Configuration for Soft Actor-Critic (SAC) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SACConfig {
    /// Learning rate for all networks.
    pub learning_rate: f32,
    /// Replay buffer size.
    pub buffer_size: usize,
    /// Number of timesteps before learning starts.
    pub learning_starts: usize,
    /// Minibatch size.
    pub batch_size: usize,
    /// Discount factor.
    pub gamma: f32,
    /// Soft update coefficient.
    pub tau: f32,
    /// Training frequency.
    pub train_freq: usize,
    /// Gradient steps per update.
    pub gradient_steps: usize,
    /// Initial entropy coefficient (alpha).
    pub ent_coef: f32,
    /// Automatically tune entropy coefficient.
    pub auto_entropy_tuning: bool,
    /// Target entropy (if auto_entropy_tuning).
    pub target_entropy: Option<f32>,
    /// Policy network hidden sizes.
    pub policy_hidden_sizes: Vec<usize>,
    /// Q-network hidden sizes.
    pub q_hidden_sizes: Vec<usize>,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for SACConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            buffer_size: 1_000_000,
            learning_starts: 10_000,
            batch_size: 256,
            gamma: 0.99,
            tau: 0.005,
            train_freq: 1,
            gradient_steps: 1,
            ent_coef: 0.2,
            auto_entropy_tuning: true,
            target_entropy: None,
            policy_hidden_sizes: vec![256, 256],
            q_hidden_sizes: vec![256, 256],
            seed: None,
        }
    }
}

impl SACConfig {
    /// Create new SAC config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set gamma.
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Set tau.
    pub fn tau(mut self, t: f32) -> Self {
        self.tau = t;
        self
    }

    /// Set entropy coefficient.
    pub fn ent_coef(mut self, c: f32) -> Self {
        self.ent_coef = c;
        self
    }

    /// Enable/disable automatic entropy tuning.
    pub fn auto_entropy_tuning(mut self, enabled: bool) -> Self {
        self.auto_entropy_tuning = enabled;
        self
    }

    /// Set seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".into());
        }
        if self.buffer_size == 0 {
            return Err("buffer_size must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".into());
        }
        if !(0.0..=1.0).contains(&self.tau) {
            return Err("tau must be in [0, 1]".into());
        }
        Ok(())
    }
}

/// Configuration for Twin Delayed DDPG (TD3) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TD3Config {
    /// Learning rate for all networks.
    pub learning_rate: f32,
    /// Replay buffer size.
    pub buffer_size: usize,
    /// Number of timesteps before learning starts.
    pub learning_starts: usize,
    /// Minibatch size.
    pub batch_size: usize,
    /// Discount factor.
    pub gamma: f32,
    /// Soft update coefficient.
    pub tau: f32,
    /// Training frequency.
    pub train_freq: usize,
    /// Gradient steps per update.
    pub gradient_steps: usize,
    /// Policy update delay (update policy every N critic updates).
    pub policy_delay: usize,
    /// Target policy noise standard deviation.
    pub target_policy_noise: f32,
    /// Target policy noise clipping.
    pub target_noise_clip: f32,
    /// Exploration noise standard deviation.
    pub exploration_noise: f32,
    /// Policy network hidden sizes.
    pub policy_hidden_sizes: Vec<usize>,
    /// Q-network hidden sizes.
    pub q_hidden_sizes: Vec<usize>,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for TD3Config {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            buffer_size: 1_000_000,
            learning_starts: 10_000,
            batch_size: 256,
            gamma: 0.99,
            tau: 0.005,
            train_freq: 1,
            gradient_steps: 1,
            policy_delay: 2,
            target_policy_noise: 0.2,
            target_noise_clip: 0.5,
            exploration_noise: 0.1,
            policy_hidden_sizes: vec![256, 256],
            q_hidden_sizes: vec![256, 256],
            seed: None,
        }
    }
}

impl TD3Config {
    /// Create new TD3 config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set gamma.
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Set tau.
    pub fn tau(mut self, t: f32) -> Self {
        self.tau = t;
        self
    }

    /// Set policy delay.
    pub fn policy_delay(mut self, d: usize) -> Self {
        self.policy_delay = d;
        self
    }

    /// Set exploration noise.
    pub fn exploration_noise(mut self, n: f32) -> Self {
        self.exploration_noise = n;
        self
    }

    /// Set seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".into());
        }
        if self.buffer_size == 0 {
            return Err("buffer_size must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".into());
        }
        if self.policy_delay == 0 {
            return Err("policy_delay must be positive".into());
        }
        Ok(())
    }
}

/// Configuration for Deep Deterministic Policy Gradient (DDPG) algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DDPGConfig {
    /// Learning rate for actor network.
    pub actor_lr: f32,
    /// Learning rate for critic network.
    pub critic_lr: f32,
    /// Replay buffer size.
    pub buffer_size: usize,
    /// Number of timesteps before learning starts.
    pub learning_starts: usize,
    /// Minibatch size.
    pub batch_size: usize,
    /// Discount factor.
    pub gamma: f32,
    /// Soft update coefficient.
    pub tau: f32,
    /// Training frequency.
    pub train_freq: usize,
    /// Gradient steps per update.
    pub gradient_steps: usize,
    /// Exploration noise type.
    pub noise_type: NoiseType,
    /// Exploration noise standard deviation (for Gaussian).
    pub noise_std: f32,
    /// Actor network hidden sizes.
    pub actor_hidden_sizes: Vec<usize>,
    /// Critic network hidden sizes.
    pub critic_hidden_sizes: Vec<usize>,
    /// Random seed.
    pub seed: Option<u64>,
}

/// Type of exploration noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    /// Gaussian noise.
    Gaussian,
    /// Ornstein-Uhlenbeck process.
    OrnsteinUhlenbeck,
}

impl Default for DDPGConfig {
    fn default() -> Self {
        Self {
            actor_lr: 1e-4,
            critic_lr: 1e-3,
            buffer_size: 1_000_000,
            learning_starts: 10_000,
            batch_size: 256,
            gamma: 0.99,
            tau: 0.005,
            train_freq: 1,
            gradient_steps: 1,
            noise_type: NoiseType::Gaussian,
            noise_std: 0.1,
            actor_hidden_sizes: vec![256, 256],
            critic_hidden_sizes: vec![256, 256],
            seed: None,
        }
    }
}

impl DDPGConfig {
    /// Create new DDPG config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set actor learning rate.
    pub fn actor_lr(mut self, lr: f32) -> Self {
        self.actor_lr = lr;
        self
    }

    /// Set critic learning rate.
    pub fn critic_lr(mut self, lr: f32) -> Self {
        self.critic_lr = lr;
        self
    }

    /// Set buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set gamma.
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Set tau.
    pub fn tau(mut self, t: f32) -> Self {
        self.tau = t;
        self
    }

    /// Set noise type.
    pub fn noise_type(mut self, t: NoiseType) -> Self {
        self.noise_type = t;
        self
    }

    /// Set noise std.
    pub fn noise_std(mut self, s: f32) -> Self {
        self.noise_std = s;
        self
    }

    /// Set seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.actor_lr <= 0.0 || self.critic_lr <= 0.0 {
            return Err("learning rates must be positive".into());
        }
        if self.buffer_size == 0 {
            return Err("buffer_size must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".into());
        }
        Ok(())
    }
}

/// Configuration for Conservative Q-Learning (CQL) algorithm.
///
/// CQL is an offline RL algorithm that learns from a fixed dataset without
/// environment interaction. It uses a conservative regularization term to
/// prevent overestimation of Q-values for out-of-distribution actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CQLConfig {
    /// Learning rate for all networks.
    /// Default: 3e-4
    pub learning_rate: f32,
    /// Replay buffer size.
    /// Default: 1_000_000
    pub buffer_size: usize,
    /// Minibatch size.
    /// Default: 256
    pub batch_size: usize,
    /// Discount factor.
    /// Default: 0.99
    pub gamma: f32,
    /// Soft update coefficient for target networks.
    /// Default: 0.005
    pub tau: f32,
    /// Initial entropy coefficient (SAC alpha).
    /// Default: 0.2
    pub ent_coef: f32,
    /// Automatically tune SAC entropy coefficient.
    /// Default: true
    pub auto_entropy_tuning: bool,
    /// Target entropy for automatic entropy tuning.
    /// Default: -dim(A)
    pub target_entropy: Option<f32>,
    /// CQL alpha: weight for conservative Q penalty.
    /// Higher values make Q-values more conservative.
    /// Default: 5.0
    pub cql_alpha: f32,
    /// Temperature for logsumexp in CQL penalty.
    /// Default: 1.0
    pub cql_temp: f32,
    /// Number of random actions to sample for CQL penalty.
    /// Default: 10
    pub num_random_actions: usize,
    /// Use Lagrangian constraint for automatic CQL alpha tuning.
    /// Default: false
    pub with_lagrange: bool,
    /// Target value for Lagrangian constraint.
    /// CQL will adjust alpha to keep penalty near this value.
    /// Default: 10.0
    pub lagrange_thresh: f32,
    /// Policy network hidden sizes.
    /// Default: [256, 256]
    pub policy_hidden_sizes: Vec<usize>,
    /// Q-network hidden sizes.
    /// Default: [256, 256]
    pub q_hidden_sizes: Vec<usize>,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for CQLConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            buffer_size: 1_000_000,
            batch_size: 256,
            gamma: 0.99,
            tau: 0.005,
            ent_coef: 0.2,
            auto_entropy_tuning: true,
            target_entropy: None,
            cql_alpha: 5.0,
            cql_temp: 1.0,
            num_random_actions: 10,
            with_lagrange: false,
            lagrange_thresh: 10.0,
            policy_hidden_sizes: vec![256, 256],
            q_hidden_sizes: vec![256, 256],
            seed: None,
        }
    }
}

impl CQLConfig {
    /// Create new CQL config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set gamma (discount factor).
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Set tau (soft update coefficient).
    pub fn tau(mut self, t: f32) -> Self {
        self.tau = t;
        self
    }

    /// Set SAC entropy coefficient.
    pub fn ent_coef(mut self, c: f32) -> Self {
        self.ent_coef = c;
        self
    }

    /// Set CQL alpha (conservative penalty weight).
    pub fn cql_alpha(mut self, alpha: f32) -> Self {
        self.cql_alpha = alpha;
        self
    }

    /// Set CQL temperature.
    pub fn cql_temp(mut self, temp: f32) -> Self {
        self.cql_temp = temp;
        self
    }

    /// Set number of random actions for CQL penalty.
    pub fn num_random_actions(mut self, n: usize) -> Self {
        self.num_random_actions = n;
        self
    }

    /// Enable/disable Lagrangian constraint.
    pub fn with_lagrange(mut self, enabled: bool) -> Self {
        self.with_lagrange = enabled;
        self
    }

    /// Set Lagrangian threshold.
    pub fn lagrange_thresh(mut self, thresh: f32) -> Self {
        self.lagrange_thresh = thresh;
        self
    }

    /// Set seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".into());
        }
        if self.buffer_size == 0 {
            return Err("buffer_size must be positive".into());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".into());
        }
        if !(0.0..=1.0).contains(&self.tau) {
            return Err("tau must be in [0, 1]".into());
        }
        if self.cql_alpha < 0.0 {
            return Err("cql_alpha must be non-negative".into());
        }
        if self.cql_temp <= 0.0 {
            return Err("cql_temp must be positive".into());
        }
        if self.num_random_actions == 0 {
            return Err("num_random_actions must be positive".into());
        }
        Ok(())
    }
}

/// Risk measure for action selection in IQN.
///
/// Different risk measures allow for different policies:
/// - Mean: Risk-neutral (standard expected value)
/// - CVaR: Risk-averse (focuses on worst-case outcomes)
/// - Optimistic: Risk-seeking (focuses on best-case outcomes)
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RiskMeasure {
    /// Expected value (mean of quantiles). Risk-neutral policy.
    Mean,
    /// Conditional Value at Risk. Risk-averse policy.
    /// The parameter is the quantile level (e.g., 0.25 for CVaR_0.25).
    CVaR(f32),
    /// Upper quantile expectation. Risk-seeking policy.
    /// The parameter is the lower quantile to start from (e.g., 0.75).
    Optimistic(f32),
}

impl Default for RiskMeasure {
    fn default() -> Self {
        RiskMeasure::Mean
    }
}

/// Configuration for Implicit Quantile Networks (IQN) algorithm.
///
/// IQN is a distributional RL algorithm that learns the full distribution
/// of returns rather than just the expected value. It uses implicit quantile
/// functions with cosine embeddings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IQNConfig {
    /// Learning rate for optimizer.
    /// Default: 1e-4
    pub learning_rate: f32,
    /// Replay buffer size.
    /// Default: 1_000_000
    pub buffer_size: usize,
    /// Number of timesteps before learning starts.
    /// Default: 50_000
    pub learning_starts: usize,
    /// Minibatch size.
    /// Default: 32
    pub batch_size: usize,
    /// Discount factor.
    /// Default: 0.99
    pub gamma: f32,
    /// Soft update coefficient (tau=1.0 for hard update).
    /// Default: 1.0
    pub tau: f32,
    /// Target network update interval.
    /// Default: 10_000
    pub target_update_interval: usize,
    /// Training frequency (update every N steps).
    /// Default: 4
    pub train_freq: usize,
    /// Gradient steps per update.
    /// Default: 1
    pub gradient_steps: usize,
    /// Initial exploration rate.
    /// Default: 1.0
    pub epsilon_start: f32,
    /// Final exploration rate.
    /// Default: 0.05
    pub epsilon_end: f32,
    /// Exploration decay per step.
    /// Default: 1e-5
    pub epsilon_decay: f32,
    /// Number of quantile samples for training.
    /// Default: 64
    pub num_quantiles: usize,
    /// Number of quantile samples for target.
    /// Default: 64
    pub num_quantiles_target: usize,
    /// Number of quantile samples for policy (action selection).
    /// Default: 32
    pub num_quantiles_policy: usize,
    /// Embedding dimension for cosine features.
    /// Default: 64
    pub embedding_dim: usize,
    /// Huber loss threshold (kappa).
    /// Default: 1.0
    pub kappa: f32,
    /// Risk measure for action selection.
    /// Default: Mean
    pub risk_measure: RiskMeasure,
    /// Use prioritized experience replay.
    /// Default: false
    pub prioritized_replay: bool,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for IQNConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-4,
            buffer_size: 1_000_000,
            learning_starts: 50_000,
            batch_size: 32,
            gamma: 0.99,
            tau: 1.0,
            target_update_interval: 10_000,
            train_freq: 4,
            gradient_steps: 1,
            epsilon_start: 1.0,
            epsilon_end: 0.05,
            epsilon_decay: 1e-5,
            num_quantiles: 64,
            num_quantiles_target: 64,
            num_quantiles_policy: 32,
            embedding_dim: 64,
            kappa: 1.0,
            risk_measure: RiskMeasure::Mean,
            prioritized_replay: false,
            seed: None,
        }
    }
}

impl IQNConfig {
    /// Create new IQN config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set learning rate.
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set buffer size.
    pub fn buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set learning starts.
    pub fn learning_starts(mut self, starts: usize) -> Self {
        self.learning_starts = starts;
        self
    }

    /// Set batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set gamma (discount factor).
    pub fn gamma(mut self, g: f32) -> Self {
        self.gamma = g;
        self
    }

    /// Set tau (soft update coefficient).
    pub fn tau(mut self, t: f32) -> Self {
        self.tau = t;
        self
    }

    /// Set number of quantiles for training.
    pub fn num_quantiles(mut self, n: usize) -> Self {
        self.num_quantiles = n;
        self
    }

    /// Set number of quantiles for target computation.
    pub fn num_quantiles_target(mut self, n: usize) -> Self {
        self.num_quantiles_target = n;
        self
    }

    /// Set number of quantiles for policy/action selection.
    pub fn num_quantiles_policy(mut self, n: usize) -> Self {
        self.num_quantiles_policy = n;
        self
    }

    /// Set embedding dimension.
    pub fn embedding_dim(mut self, dim: usize) -> Self {
        self.embedding_dim = dim;
        self
    }

    /// Set kappa (Huber loss threshold).
    pub fn kappa(mut self, k: f32) -> Self {
        self.kappa = k;
        self
    }

    /// Set risk measure for action selection.
    pub fn risk_measure(mut self, measure: RiskMeasure) -> Self {
        self.risk_measure = measure;
        self
    }

    /// Enable prioritized replay.
    pub fn prioritized_replay(mut self, enabled: bool) -> Self {
        self.prioritized_replay = enabled;
        self
    }

    /// Set seed.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.learning_rate <= 0.0 {
            return Err("learning_rate must be positive".into());
        }
        if self.buffer_size == 0 {
            return Err("buffer_size must be positive".into());
        }
        if self.batch_size == 0 {
            return Err("batch_size must be positive".into());
        }
        if !(0.0..=1.0).contains(&self.gamma) {
            return Err("gamma must be in [0, 1]".into());
        }
        if self.num_quantiles == 0 {
            return Err("num_quantiles must be positive".into());
        }
        if self.num_quantiles_target == 0 {
            return Err("num_quantiles_target must be positive".into());
        }
        if self.num_quantiles_policy == 0 {
            return Err("num_quantiles_policy must be positive".into());
        }
        if self.embedding_dim == 0 {
            return Err("embedding_dim must be positive".into());
        }
        if self.kappa <= 0.0 {
            return Err("kappa must be positive".into());
        }
        Ok(())
    }
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
