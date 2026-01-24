//! Actor-Critic network architecture for policy gradient methods.
//!
//! This module provides a combined actor (policy) and critic (value) network
//! that supports both discrete and continuous action spaces, with optional
//! recurrent layers for time-series data like trading.

use candle_core::{Device, Result as CandleResult, Tensor};
use candle_nn::{Module, VarBuilder};
use serde::{Deserialize, Serialize};

#[cfg(test)]
use candle_core::DType;

use super::mlp::{Activation, MLPConfig, MLP};
use super::rnn::{GRUState, LSTMState, RNNConfig, GRU, LSTM};

/// Action space specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionSpace {
    /// Discrete actions with n possible choices (Categorical distribution).
    Discrete {
        /// Number of possible actions.
        n: usize,
    },
    /// Continuous actions (Gaussian distribution).
    Continuous {
        /// Dimension of the action vector.
        dim: usize,
        /// Whether to learn the standard deviation.
        learnable_std: bool,
        /// Initial log standard deviation (if learnable_std is true).
        init_log_std: f32,
    },
}

impl ActionSpace {
    /// Create a discrete action space.
    pub fn discrete(n: usize) -> Self {
        ActionSpace::Discrete { n }
    }

    /// Create a continuous action space with learnable std.
    pub fn continuous(dim: usize) -> Self {
        ActionSpace::Continuous {
            dim,
            learnable_std: true,
            init_log_std: 0.0,
        }
    }

    /// Create a continuous action space with fixed std.
    pub fn continuous_fixed_std(dim: usize) -> Self {
        ActionSpace::Continuous {
            dim,
            learnable_std: false,
            init_log_std: 0.0,
        }
    }

    /// Get the output dimension needed for the action head.
    pub fn action_dim(&self) -> usize {
        match self {
            ActionSpace::Discrete { n } => *n,
            ActionSpace::Continuous {
                dim, learnable_std, ..
            } => {
                if *learnable_std {
                    dim * 2
                } else {
                    *dim
                }
            }
        }
    }
}

/// Type of recurrent backbone.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[derive(Default)]
pub enum RecurrentType {
    /// No recurrent layer (feedforward only).
    #[default]
    None,
    /// LSTM recurrent layer.
    LSTM,
    /// GRU recurrent layer.
    GRU,
}


/// Recurrent state for ActorCritic networks.
#[derive(Debug, Clone)]
pub enum RecurrentState {
    /// No recurrent state.
    None,
    /// LSTM state (hidden and cell).
    LSTM(LSTMState),
    /// GRU state (hidden only).
    GRU(GRUState),
}

impl RecurrentState {
    /// Check if this is a non-recurrent state.
    pub fn is_none(&self) -> bool {
        matches!(self, RecurrentState::None)
    }

    /// Detach state from computation graph.
    pub fn detach(&self) -> Self {
        match self {
            RecurrentState::None => RecurrentState::None,
            RecurrentState::LSTM(state) => RecurrentState::LSTM(state.detach()),
            RecurrentState::GRU(state) => RecurrentState::GRU(state.detach()),
        }
    }
}

/// Configuration for ActorCritic networks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorCriticConfig {
    /// Observation (input) dimension.
    pub obs_dim: usize,
    /// Action space specification.
    pub action_space: ActionSpace,
    /// Hidden layer dimensions for shared backbone.
    pub hidden_dims: Vec<usize>,
    /// Activation function for hidden layers.
    pub activation: Activation,
    /// Type of recurrent layer (if any).
    pub recurrent_type: RecurrentType,
    /// Hidden dimension for recurrent layer (if used).
    pub recurrent_hidden_dim: usize,
    /// Whether actor and critic share the same backbone.
    pub shared_backbone: bool,
    /// Hidden dimensions for actor head (after backbone).
    pub actor_head_dims: Vec<usize>,
    /// Hidden dimensions for critic head (after backbone).
    pub critic_head_dims: Vec<usize>,
    /// Whether to orthogonally initialize weights.
    pub ortho_init: bool,
}

impl ActorCriticConfig {
    /// Create a configuration for discrete action space.
    pub fn discrete(obs_dim: usize, num_actions: usize) -> Self {
        Self {
            obs_dim,
            action_space: ActionSpace::discrete(num_actions),
            hidden_dims: vec![256, 256],
            activation: Activation::Tanh,
            recurrent_type: RecurrentType::None,
            recurrent_hidden_dim: 256,
            shared_backbone: true,
            actor_head_dims: vec![],
            critic_head_dims: vec![],
            ortho_init: true,
        }
    }

    /// Create a configuration for continuous action space.
    pub fn continuous(obs_dim: usize, action_dim: usize) -> Self {
        Self {
            obs_dim,
            action_space: ActionSpace::continuous(action_dim),
            hidden_dims: vec![256, 256],
            activation: Activation::Tanh,
            recurrent_type: RecurrentType::None,
            recurrent_hidden_dim: 256,
            shared_backbone: true,
            actor_head_dims: vec![],
            critic_head_dims: vec![],
            ortho_init: true,
        }
    }

    /// Add LSTM recurrent layer.
    pub fn with_lstm(mut self, hidden_dim: usize) -> Self {
        self.recurrent_type = RecurrentType::LSTM;
        self.recurrent_hidden_dim = hidden_dim;
        self
    }

    /// Add GRU recurrent layer.
    pub fn with_gru(mut self, hidden_dim: usize) -> Self {
        self.recurrent_type = RecurrentType::GRU;
        self.recurrent_hidden_dim = hidden_dim;
        self
    }

    /// Set hidden layer dimensions.
    pub fn with_hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    /// Set activation function.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Disable shared backbone (separate networks for actor/critic).
    pub fn with_separate_networks(mut self) -> Self {
        self.shared_backbone = false;
        self
    }

    /// Set actor head hidden dimensions.
    pub fn with_actor_head(mut self, dims: Vec<usize>) -> Self {
        self.actor_head_dims = dims;
        self
    }

    /// Set critic head hidden dimensions.
    pub fn with_critic_head(mut self, dims: Vec<usize>) -> Self {
        self.critic_head_dims = dims;
        self
    }
}

/// Recurrent backbone enum for internal use.
#[derive(Debug)]
enum RecurrentBackbone {
    None,
    LSTM(LSTM),
    GRU(GRU),
}

/// Actor-Critic network combining policy and value estimation.
///
/// This architecture supports:
/// - Discrete actions (outputs logits for Categorical distribution)
/// - Continuous actions (outputs mean and optionally log_std for Gaussian)
/// - Optional LSTM/GRU backbone for time-series data
/// - Shared or separate backbones for actor and critic
#[derive(Debug)]
pub struct ActorCritic {
    /// Shared feature extractor (if shared_backbone is true).
    shared_backbone: Option<MLP>,
    /// Recurrent layer (if any).
    recurrent: RecurrentBackbone,
    /// Actor backbone (if not sharing).
    actor_backbone: Option<MLP>,
    /// Critic backbone (if not sharing).
    critic_backbone: Option<MLP>,
    /// Actor head (policy network).
    actor_head: MLP,
    /// Critic head (value network).
    critic_head: MLP,
    /// Log standard deviation for continuous actions (if learnable).
    log_std: Option<Tensor>,
    /// Configuration.
    config: ActorCriticConfig,
}

impl ActorCritic {
    /// Create a new ActorCritic network.
    pub fn new(vb: VarBuilder<'_>, config: ActorCriticConfig) -> CandleResult<Self> {
        let device = vb.device();

        // Determine backbone output dimension
        let backbone_out_dim = if config.recurrent_type != RecurrentType::None {
            config.recurrent_hidden_dim
        } else if !config.hidden_dims.is_empty() {
            *config.hidden_dims.last().unwrap()
        } else {
            config.obs_dim
        };

        // Build recurrent layer if needed
        let recurrent = match config.recurrent_type {
            RecurrentType::None => RecurrentBackbone::None,
            RecurrentType::LSTM => {
                let rnn_input_dim = if !config.hidden_dims.is_empty() {
                    *config.hidden_dims.last().unwrap()
                } else {
                    config.obs_dim
                };
                let rnn_config = RNNConfig::new(rnn_input_dim, config.recurrent_hidden_dim);
                RecurrentBackbone::LSTM(LSTM::new(vb.pp("recurrent"), rnn_config)?)
            }
            RecurrentType::GRU => {
                let rnn_input_dim = if !config.hidden_dims.is_empty() {
                    *config.hidden_dims.last().unwrap()
                } else {
                    config.obs_dim
                };
                let rnn_config = RNNConfig::new(rnn_input_dim, config.recurrent_hidden_dim);
                RecurrentBackbone::GRU(GRU::new(vb.pp("recurrent"), rnn_config)?)
            }
        };

        // Build shared or separate backbones
        let (shared_backbone, actor_backbone, critic_backbone) = if config.shared_backbone {
            let backbone = if !config.hidden_dims.is_empty() {
                let mlp_config = MLPConfig::new(
                    config.obs_dim,
                    config.hidden_dims[..config.hidden_dims.len() - 1].to_vec(),
                    *config.hidden_dims.last().unwrap(),
                )
                .with_activation(config.activation)
                .with_output_activation(config.activation);
                Some(MLP::new(vb.pp("shared_backbone"), mlp_config)?)
            } else {
                None
            };
            (backbone, None, None)
        } else {
            let actor_mlp_config =
                MLPConfig::new(config.obs_dim, config.hidden_dims.clone(), backbone_out_dim)
                    .with_activation(config.activation)
                    .with_output_activation(config.activation);
            let actor_backbone = MLP::new(vb.pp("actor_backbone"), actor_mlp_config)?;

            let critic_mlp_config =
                MLPConfig::new(config.obs_dim, config.hidden_dims.clone(), backbone_out_dim)
                    .with_activation(config.activation)
                    .with_output_activation(config.activation);
            let critic_backbone = MLP::new(vb.pp("critic_backbone"), critic_mlp_config)?;

            (None, Some(actor_backbone), Some(critic_backbone))
        };

        // Build actor head
        let actor_input_dim = if !config.actor_head_dims.is_empty() {
            backbone_out_dim
        } else {
            backbone_out_dim
        };
        let actor_output_dim = config.action_space.action_dim();
        let actor_head_config = if !config.actor_head_dims.is_empty() {
            MLPConfig::new(
                actor_input_dim,
                config.actor_head_dims.clone(),
                actor_output_dim,
            )
            .with_activation(config.activation)
        } else {
            MLPConfig::new(actor_input_dim, vec![], actor_output_dim)
        };
        let actor_head = MLP::new(vb.pp("actor_head"), actor_head_config)?;

        // Build critic head
        let critic_input_dim = if !config.critic_head_dims.is_empty() {
            backbone_out_dim
        } else {
            backbone_out_dim
        };
        let critic_head_config = if !config.critic_head_dims.is_empty() {
            MLPConfig::new(critic_input_dim, config.critic_head_dims.clone(), 1)
                .with_activation(config.activation)
        } else {
            MLPConfig::new(critic_input_dim, vec![], 1)
        };
        let critic_head = MLP::new(vb.pp("critic_head"), critic_head_config)?;

        // Create learnable log_std for continuous actions if needed
        let log_std = match &config.action_space {
            ActionSpace::Continuous {
                dim,
                learnable_std: true,
                init_log_std,
            } => Some(vb.get_with_hints(
                &[*dim],
                "log_std",
                candle_nn::Init::Const((*init_log_std) as f64),
            )?),
            _ => None,
        };

        Ok(Self {
            shared_backbone,
            recurrent,
            actor_backbone,
            critic_backbone,
            actor_head,
            critic_head,
            log_std,
            config,
        })
    }

    /// Get the configuration.
    pub fn config(&self) -> &ActorCriticConfig {
        &self.config
    }

    /// Initialize recurrent state for given batch size.
    pub fn init_recurrent_state(
        &self,
        batch_size: usize,
        device: &Device,
    ) -> CandleResult<RecurrentState> {
        match &self.recurrent {
            RecurrentBackbone::None => Ok(RecurrentState::None),
            RecurrentBackbone::LSTM(lstm) => {
                Ok(RecurrentState::LSTM(lstm.init_state(batch_size, device)?))
            }
            RecurrentBackbone::GRU(gru) => {
                Ok(RecurrentState::GRU(gru.init_state(batch_size, device)?))
            }
        }
    }

    /// Forward pass through the network.
    ///
    /// # Arguments
    /// * `obs` - Observation tensor of shape [batch_size, obs_dim]
    /// * `state` - Optional recurrent state
    ///
    /// # Returns
    /// Tuple of (action_output, value, new_state) where:
    /// - For discrete: action_output is logits [batch_size, num_actions]
    /// - For continuous: action_output is (mean, log_std) each [batch_size, action_dim]
    /// - value is [batch_size, 1]
    pub fn forward(
        &self,
        obs: &Tensor,
        state: Option<&RecurrentState>,
    ) -> CandleResult<(Tensor, Tensor, RecurrentState)> {
        // Process through backbone
        let (actor_features, critic_features, new_state) = self.extract_features(obs, state)?;

        // Actor head
        let action_output = self.actor_head.forward(&actor_features)?;

        // Critic head
        let value = self.critic_head.forward(&critic_features)?;

        Ok((action_output, value, new_state))
    }

    /// Forward pass returning action distribution parameters.
    ///
    /// For discrete actions: returns (logits, value, state)
    /// For continuous actions: returns (mean, value, state) - use get_log_std() separately
    pub fn forward_actor_critic(
        &self,
        obs: &Tensor,
        state: Option<&RecurrentState>,
    ) -> CandleResult<(Tensor, Tensor, RecurrentState)> {
        self.forward(obs, state)
    }

    /// Get the log standard deviation for continuous actions.
    ///
    /// For continuous action spaces with learnable std, this returns the log_std tensor.
    /// For fixed std or discrete actions, returns None.
    pub fn get_log_std(&self) -> Option<&Tensor> {
        self.log_std.as_ref()
    }

    /// Forward pass for actor only (policy network).
    ///
    /// # Arguments
    /// * `obs` - Observation tensor
    /// * `state` - Optional recurrent state
    ///
    /// # Returns
    /// (action_output, new_state)
    pub fn forward_actor(
        &self,
        obs: &Tensor,
        state: Option<&RecurrentState>,
    ) -> CandleResult<(Tensor, RecurrentState)> {
        let (actor_features, _, new_state) = self.extract_features(obs, state)?;
        let action_output = self.actor_head.forward(&actor_features)?;
        Ok((action_output, new_state))
    }

    /// Forward pass for critic only (value network).
    ///
    /// # Arguments
    /// * `obs` - Observation tensor
    /// * `state` - Optional recurrent state
    ///
    /// # Returns
    /// (value, new_state)
    pub fn forward_critic(
        &self,
        obs: &Tensor,
        state: Option<&RecurrentState>,
    ) -> CandleResult<(Tensor, RecurrentState)> {
        let (_, critic_features, new_state) = self.extract_features(obs, state)?;
        let value = self.critic_head.forward(&critic_features)?;
        Ok((value, new_state))
    }

    /// Extract features from observation through backbone and recurrent layers.
    fn extract_features(
        &self,
        obs: &Tensor,
        state: Option<&RecurrentState>,
    ) -> CandleResult<(Tensor, Tensor, RecurrentState)> {
        let batch_size = obs.dims()[0];
        let device = obs.device();

        // Get default state if not provided
        let default_state;
        let state = match state {
            Some(s) => s,
            None => {
                default_state = self.init_recurrent_state(batch_size, device)?;
                &default_state
            }
        };

        if self.config.shared_backbone {
            // Shared backbone path
            let features = match &self.shared_backbone {
                Some(backbone) => backbone.forward(obs)?,
                None => obs.clone(),
            };

            // Process through recurrent layer
            let (features, new_state) = match (&self.recurrent, state) {
                (RecurrentBackbone::None, _) => (features, RecurrentState::None),
                (RecurrentBackbone::LSTM(lstm), RecurrentState::LSTM(lstm_state)) => {
                    let (out, new_lstm_state) = lstm.forward_step(&features, lstm_state)?;
                    (out, RecurrentState::LSTM(new_lstm_state))
                }
                (RecurrentBackbone::GRU(gru), RecurrentState::GRU(gru_state)) => {
                    let (out, new_gru_state) = gru.forward_step(&features, gru_state)?;
                    (out, RecurrentState::GRU(new_gru_state))
                }
                (RecurrentBackbone::LSTM(lstm), _) => {
                    let init_state = lstm.init_state(batch_size, device)?;
                    let (out, new_lstm_state) = lstm.forward_step(&features, &init_state)?;
                    (out, RecurrentState::LSTM(new_lstm_state))
                }
                (RecurrentBackbone::GRU(gru), _) => {
                    let init_state = gru.init_state(batch_size, device)?;
                    let (out, new_gru_state) = gru.forward_step(&features, &init_state)?;
                    (out, RecurrentState::GRU(new_gru_state))
                }
            };

            Ok((features.clone(), features, new_state))
        } else {
            // Separate backbone path
            let actor_features = match &self.actor_backbone {
                Some(backbone) => backbone.forward(obs)?,
                None => obs.clone(),
            };
            let critic_features = match &self.critic_backbone {
                Some(backbone) => backbone.forward(obs)?,
                None => obs.clone(),
            };

            // Note: For separate backbones, we don't use recurrent layers
            // (would need separate recurrent states for actor and critic)
            Ok((actor_features, critic_features, RecurrentState::None))
        }
    }

    /// Check if this network has a recurrent component.
    pub fn is_recurrent(&self) -> bool {
        !matches!(self.recurrent, RecurrentBackbone::None)
    }

    /// Get the action space configuration.
    pub fn action_space(&self) -> &ActionSpace {
        &self.config.action_space
    }

    /// Get the observation dimension.
    pub fn obs_dim(&self) -> usize {
        self.config.obs_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_actor_critic_discrete() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = ActorCriticConfig::discrete(64, 4);
        let ac = ActorCritic::new(vb, config).unwrap();

        let obs = Tensor::randn(0.0f32, 1.0, &[8, 64], &device).unwrap();
        let (action_logits, value, _) = ac.forward(&obs, None).unwrap();

        assert_eq!(action_logits.dims(), &[8, 4]);
        assert_eq!(value.dims(), &[8, 1]);
    }

    #[test]
    fn test_actor_critic_continuous() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = ActorCriticConfig::continuous(64, 2);
        let ac = ActorCritic::new(vb, config).unwrap();

        let obs = Tensor::randn(0.0f32, 1.0, &[8, 64], &device).unwrap();
        let (action_mean, value, _) = ac.forward(&obs, None).unwrap();

        // Continuous with learnable_std outputs mean and log_std concatenated
        assert_eq!(action_mean.dims(), &[8, 4]); // 2 * action_dim for mean + log_std
        assert_eq!(value.dims(), &[8, 1]);
    }

    #[test]
    fn test_actor_critic_with_lstm() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = ActorCriticConfig::discrete(64, 4)
            .with_lstm(128)
            .with_hidden_dims(vec![256]);
        let ac = ActorCritic::new(vb, config).unwrap();

        assert!(ac.is_recurrent());

        let batch_size = 8;
        let obs = Tensor::randn(0.0f32, 1.0, &[batch_size, 64], &device).unwrap();

        // First forward pass (no state)
        let (logits1, value1, state1) = ac.forward(&obs, None).unwrap();
        assert_eq!(logits1.dims(), &[batch_size, 4]);
        assert_eq!(value1.dims(), &[batch_size, 1]);
        assert!(!state1.is_none());

        // Second forward pass (with state)
        let (logits2, value2, _state2) = ac.forward(&obs, Some(&state1)).unwrap();
        assert_eq!(logits2.dims(), &[batch_size, 4]);
        assert_eq!(value2.dims(), &[batch_size, 1]);
    }

    #[test]
    fn test_actor_critic_with_gru() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = ActorCriticConfig::discrete(64, 4)
            .with_gru(128)
            .with_hidden_dims(vec![256]);
        let ac = ActorCritic::new(vb, config).unwrap();

        assert!(ac.is_recurrent());

        let batch_size = 8;
        let obs = Tensor::randn(0.0f32, 1.0, &[batch_size, 64], &device).unwrap();

        let (logits, value, state) = ac.forward(&obs, None).unwrap();
        assert_eq!(logits.dims(), &[batch_size, 4]);
        assert_eq!(value.dims(), &[batch_size, 1]);

        match state {
            RecurrentState::GRU(_) => (),
            _ => panic!("Expected GRU state"),
        }
    }

    #[test]
    fn test_actor_critic_separate_networks() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = ActorCriticConfig::discrete(64, 4)
            .with_separate_networks()
            .with_hidden_dims(vec![128, 128]);
        let ac = ActorCritic::new(vb, config).unwrap();

        let obs = Tensor::randn(0.0f32, 1.0, &[8, 64], &device).unwrap();

        // Test forward_actor
        let (logits, _) = ac.forward_actor(&obs, None).unwrap();
        assert_eq!(logits.dims(), &[8, 4]);

        // Test forward_critic
        let (value, _) = ac.forward_critic(&obs, None).unwrap();
        assert_eq!(value.dims(), &[8, 1]);
    }

    #[test]
    fn test_action_space_dim() {
        assert_eq!(ActionSpace::discrete(10).action_dim(), 10);
        assert_eq!(ActionSpace::continuous(4).action_dim(), 8); // mean + log_std
        assert_eq!(ActionSpace::continuous_fixed_std(4).action_dim(), 4); // mean only
    }
}
