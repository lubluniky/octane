//! Neural network modules for RocketRL.
//!
//! This module provides configurable neural network architectures for
//! reinforcement learning, including MLPs, recurrent networks (LSTM/GRU),
//! and the combined ActorCritic architecture.
//!
//! # Feature Flags
//! - `metal`: Enable Apple Silicon GPU acceleration via Metal
//! - `cuda`: Enable NVIDIA GPU acceleration via CUDA
//!
//! # Example
//! ```ignore
//! use rocket_rs::networks::{MLP, ActorCritic, ActorCriticConfig};
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
mod mlp;
mod rnn;

pub use actor_critic::{ActionSpace, ActorCritic, ActorCriticConfig, RecurrentState};
pub use mlp::{Activation, MLPConfig, MLP};
pub use rnn::{GRUState, LSTMState, RNNConfig, GRU, LSTM};
