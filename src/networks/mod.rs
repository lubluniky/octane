//! Neural network modules for Octane.
//!
//! This module provides configurable neural network architectures for
//! reinforcement learning, including MLPs, recurrent networks (LSTM/GRU),
//! Transformers, attention mechanisms, and the combined ActorCritic architecture.
//!
//! # Feature Flags
//! - `metal`: Enable Apple Silicon GPU acceleration via Metal
//! - `cuda`: Enable NVIDIA GPU acceleration via CUDA
//!
//! # Modules
//! - [`mlp`]: Multi-Layer Perceptron implementations
//! - [`rnn`]: Recurrent networks (LSTM, GRU)
//! - [`actor_critic`]: Combined actor-critic architectures
//! - [`transformer`]: Transformer encoder for sequence modeling
//! - [`attention`]: Attention mechanisms (self-attention, cross-attention)
//! - [`normalization`]: Normalization layers (LayerNorm, BatchNorm, RMSNorm)
//! - [`init`]: Weight initialization utilities
//!
//! # Example
//! ```ignore
//! use octane_rs::networks::{MLP, ActorCritic, ActorCriticConfig};
//! use candle_core::Device;
//! use candle_nn::VarMap;
//!
//! let device = Device::Cpu;
//! let varmap = VarMap::new();
//! let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
//!
//! let config = ActorCriticConfig::discrete(64, 4);
//! let actor_critic = ActorCritic::new(vb, config).unwrap();
//! ```

mod actor_critic;
pub mod attention;
pub mod init;
mod mlp;
pub mod normalization;
mod rnn;
pub mod transformer;

// Actor-Critic exports
pub use actor_critic::{ActionSpace, ActorCritic, ActorCriticConfig, RecurrentState};

// MLP exports
pub use mlp::{Activation, MLPConfig, MLP};

// RNN exports
pub use rnn::{GRUState, LSTMState, RNNConfig, GRU, LSTM};

// Transformer exports
pub use transformer::{
    DecisionTransformer, DecisionTransformerConfig, MultiHeadAttention, PositionalEncoding,
    TransformerConfig, TransformerEncoder, TransformerEncoderLayer,
};

// Attention exports
pub use attention::{
    AttentionActorCritic, AttentionActorCriticConfig, CrossAttention, CrossAttentionConfig,
    SelfAttention, SelfAttentionConfig,
};

// Normalization exports
pub use normalization::{
    BatchNorm, BatchNormConfig, LayerNorm, LayerNormConfig, RMSNorm, RMSNormConfig,
};

// Initialization exports
pub use init::{
    calculate_gain, kaiming_normal, kaiming_uniform, orthogonal_init, xavier_normal,
    xavier_uniform, InitMethod,
};
