//! Attention mechanisms for neural networks.
//!
//! This module provides various attention implementations for use in RL,
//! including self-attention, cross-attention, and an attention-augmented
//! actor-critic architecture.
//!
//! # Example
//! ```ignore
//! use octane_rs::networks::attention::{SelfAttention, SelfAttentionConfig};
//! use candle_core::Device;
//! use candle_nn::VarMap;
//!
//! let device = Device::Cpu;
//! let varmap = VarMap::new();
//! let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
//!
//! let config = SelfAttentionConfig::new(256, 4);
//! let attention = SelfAttention::new(vb, config).unwrap();
//! ```

use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

use super::mlp::{Activation, MLPConfig, MLP};
use super::normalization::{LayerNorm, LayerNormConfig};

/// Configuration for Self-Attention module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelfAttentionConfig {
    /// Input/output dimension (d_model).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dropout probability (set to 0.0 for deterministic inference).
    pub dropout: f64,
    /// Whether to use a causal mask (for autoregressive models).
    pub causal: bool,
    /// Whether to use pre-normalization (recommended for stability).
    pub pre_norm: bool,
}

impl SelfAttentionConfig {
    /// Create a new SelfAttention configuration.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `num_heads` - Number of attention heads (d_model must be divisible by num_heads)
    pub fn new(d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model.is_multiple_of(num_heads),
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );

        Self {
            d_model,
            num_heads,
            dropout: 0.0,
            causal: false,
            pre_norm: true,
        }
    }

    /// Enable causal masking for autoregressive models.
    pub fn with_causal(mut self) -> Self {
        self.causal = true;
        self
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Disable pre-normalization.
    pub fn without_pre_norm(mut self) -> Self {
        self.pre_norm = false;
        self
    }
}

/// Self-Attention module.
///
/// Implements multi-head self-attention with optional causal masking.
/// This is the core building block for Transformer architectures.
///
/// The attention mechanism allows the model to focus on different parts
/// of the input sequence when making predictions, which is valuable for
/// capturing long-range dependencies in RL observation sequences.
#[derive(Debug)]
pub struct SelfAttention {
    /// Query projection.
    wq: Linear,
    /// Key projection.
    wk: Linear,
    /// Value projection.
    wv: Linear,
    /// Output projection.
    wo: Linear,
    /// Layer normalization (if pre_norm).
    layer_norm: Option<LayerNorm>,
    /// Configuration.
    config: SelfAttentionConfig,
    /// Head dimension (d_model / num_heads).
    head_dim: usize,
    /// Scaling factor for attention scores.
    scale: f64,
}

impl SelfAttention {
    /// Create a new SelfAttention module.
    pub fn new(vb: VarBuilder<'_>, config: SelfAttentionConfig) -> CandleResult<Self> {
        let d_model = config.d_model;
        let head_dim = d_model / config.num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let wq = candle_nn::linear(d_model, d_model, vb.pp("wq"))?;
        let wk = candle_nn::linear(d_model, d_model, vb.pp("wk"))?;
        let wv = candle_nn::linear(d_model, d_model, vb.pp("wv"))?;
        let wo = candle_nn::linear(d_model, d_model, vb.pp("wo"))?;

        let layer_norm = if config.pre_norm {
            Some(LayerNorm::new(vb.pp("ln"), LayerNormConfig::new(d_model))?)
        } else {
            None
        };

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            layer_norm,
            config,
            head_dim,
            scale,
        })
    }

    /// Forward pass through self-attention.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, d_model]
    /// * `mask` - Optional attention mask of shape [batch_size, seq_len, seq_len]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, seq_len, d_model]
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let residual = x.clone();

        // Pre-normalization
        let x = match &self.layer_norm {
            Some(ln) => ln.forward(x)?,
            None => x.clone(),
        };

        let (batch_size, seq_len, _) = x.dims3()?;

        // Project to Q, K, V
        let q = self.wq.forward(&x)?;
        let k = self.wk.forward(&x)?;
        let v = self.wv.forward(&x)?;

        // Reshape for multi-head attention: [batch, seq, heads, head_dim]
        let q = q
            .reshape(&[batch_size, seq_len, self.config.num_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?; // [batch, heads, seq, head_dim]
        let k = k
            .reshape(&[batch_size, seq_len, self.config.num_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape(&[batch_size, seq_len, self.config.num_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?;

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_scores = (q.matmul(&k_t)? * self.scale)?;

        // Apply causal mask if needed
        let attn_scores = if self.config.causal {
            let causal_mask = create_causal_mask(seq_len, x.device(), x.dtype())?;
            let causal_mask = causal_mask.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, seq, seq]
            attn_scores.broadcast_add(&causal_mask)?
        } else {
            attn_scores
        };

        // Apply optional external mask
        let attn_scores = match mask {
            Some(m) => {
                let m = m.unsqueeze(1)?; // [batch, 1, seq, seq] for broadcasting
                attn_scores.broadcast_add(&m)?
            }
            None => attn_scores,
        };

        // Softmax over last dimension
        let attn_weights = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, d_model]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape(&[
            batch_size,
            seq_len,
            self.config.d_model,
        ])?;

        // Output projection
        let output = self.wo.forward(&attn_output)?;

        // Residual connection
        output + residual
    }

    /// Get the model dimension.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Get the number of heads.
    pub fn num_heads(&self) -> usize {
        self.config.num_heads
    }
}

/// Create a causal mask for autoregressive attention.
///
/// Returns a lower-triangular mask where positions that should not be attended to
/// have value -inf (or a large negative number).
fn create_causal_mask(seq_len: usize, device: &Device, dtype: DType) -> CandleResult<Tensor> {
    // Create a matrix where mask[i][j] = -inf if j > i, else 0
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }

    let mask = Tensor::from_slice(&mask_data, &[seq_len, seq_len], device)?;
    mask.to_dtype(dtype)
}

/// Configuration for Cross-Attention module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossAttentionConfig {
    /// Query input dimension.
    pub d_query: usize,
    /// Key/Value input dimension.
    pub d_kv: usize,
    /// Internal model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dropout probability.
    pub dropout: f64,
}

impl CrossAttentionConfig {
    /// Create a new CrossAttention configuration.
    ///
    /// # Arguments
    /// * `d_query` - Dimension of query input
    /// * `d_kv` - Dimension of key/value input
    /// * `d_model` - Internal model dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of attention heads
    pub fn new(d_query: usize, d_kv: usize, d_model: usize, num_heads: usize) -> Self {
        assert!(
            d_model.is_multiple_of(num_heads),
            "d_model must be divisible by num_heads"
        );

        Self {
            d_query,
            d_kv,
            d_model,
            num_heads,
            dropout: 0.0,
        }
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }
}

/// Cross-Attention module.
///
/// Allows one sequence to attend to another sequence.
/// This is useful for multi-agent scenarios where agents need to
/// attend to observations from other agents.
#[derive(Debug)]
pub struct CrossAttention {
    /// Query projection.
    wq: Linear,
    /// Key projection.
    wk: Linear,
    /// Value projection.
    wv: Linear,
    /// Output projection.
    wo: Linear,
    /// Layer normalization for query.
    ln_q: LayerNorm,
    /// Layer normalization for key/value.
    ln_kv: LayerNorm,
    /// Configuration.
    config: CrossAttentionConfig,
    /// Head dimension.
    head_dim: usize,
    /// Scaling factor.
    scale: f64,
}

impl CrossAttention {
    /// Create a new CrossAttention module.
    pub fn new(vb: VarBuilder<'_>, config: CrossAttentionConfig) -> CandleResult<Self> {
        let d_model = config.d_model;
        let head_dim = d_model / config.num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let wq = candle_nn::linear(config.d_query, d_model, vb.pp("wq"))?;
        let wk = candle_nn::linear(config.d_kv, d_model, vb.pp("wk"))?;
        let wv = candle_nn::linear(config.d_kv, d_model, vb.pp("wv"))?;
        let wo = candle_nn::linear(d_model, config.d_query, vb.pp("wo"))?;

        let ln_q = LayerNorm::new(vb.pp("ln_q"), LayerNormConfig::new(config.d_query))?;
        let ln_kv = LayerNorm::new(vb.pp("ln_kv"), LayerNormConfig::new(config.d_kv))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            ln_q,
            ln_kv,
            config,
            head_dim,
            scale,
        })
    }

    /// Forward pass through cross-attention.
    ///
    /// # Arguments
    /// * `query` - Query tensor of shape [batch_size, query_len, d_query]
    /// * `key_value` - Key/Value tensor of shape [batch_size, kv_len, d_kv]
    /// * `mask` - Optional attention mask of shape [batch_size, query_len, kv_len]
    ///
    /// # Returns
    /// Output tensor of shape [batch_size, query_len, d_query]
    pub fn forward(
        &self,
        query: &Tensor,
        key_value: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let residual = query.clone();

        // Pre-normalization
        let query = self.ln_q.forward(query)?;
        let key_value = self.ln_kv.forward(key_value)?;

        let (batch_size, query_len, _) = query.dims3()?;
        let kv_len = key_value.dims()[1];

        // Project Q from query, K/V from key_value
        let q = self.wq.forward(&query)?;
        let k = self.wk.forward(&key_value)?;
        let v = self.wv.forward(&key_value)?;

        // Reshape for multi-head attention
        let q = q
            .reshape(&[batch_size, query_len, self.config.num_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape(&[batch_size, kv_len, self.config.num_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape(&[batch_size, kv_len, self.config.num_heads, self.head_dim])?
            .transpose(1, 2)?
            .contiguous()?;

        // Compute attention: Q @ K^T / sqrt(d_k)
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let attn_scores = (q.matmul(&k_t)? * self.scale)?;

        // Apply mask if provided
        let attn_scores = match mask {
            Some(m) => {
                let m = m.unsqueeze(1)?;
                attn_scores.broadcast_add(&m)?
            }
            None => attn_scores,
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;

        // Apply to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape(&[
            batch_size,
            query_len,
            self.config.d_model,
        ])?;

        // Output projection
        let output = self.wo.forward(&attn_output)?;

        // Residual connection
        output + residual
    }

    /// Get the query dimension.
    pub fn d_query(&self) -> usize {
        self.config.d_query
    }

    /// Get the key/value dimension.
    pub fn d_kv(&self) -> usize {
        self.config.d_kv
    }
}

/// Configuration for Attention-augmented ActorCritic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionActorCriticConfig {
    /// Observation dimension.
    pub obs_dim: usize,
    /// Action dimension (for continuous) or number of actions (for discrete).
    pub action_dim: usize,
    /// Whether action space is discrete.
    pub discrete: bool,
    /// Hidden dimensions for MLP backbone.
    pub hidden_dims: Vec<usize>,
    /// Attention model dimension.
    pub attention_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Sequence length for attention (e.g., observation history).
    pub seq_len: usize,
    /// Activation function.
    pub activation: Activation,
}

impl AttentionActorCriticConfig {
    /// Create configuration for discrete action space.
    pub fn discrete(obs_dim: usize, num_actions: usize, seq_len: usize) -> Self {
        Self {
            obs_dim,
            action_dim: num_actions,
            discrete: true,
            hidden_dims: vec![256],
            attention_dim: 128,
            num_heads: 4,
            seq_len,
            activation: Activation::ReLU,
        }
    }

    /// Create configuration for continuous action space.
    pub fn continuous(obs_dim: usize, action_dim: usize, seq_len: usize) -> Self {
        Self {
            obs_dim,
            action_dim,
            discrete: false,
            hidden_dims: vec![256],
            attention_dim: 128,
            num_heads: 4,
            seq_len,
            activation: Activation::ReLU,
        }
    }

    /// Set hidden dimensions.
    pub fn with_hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.hidden_dims = dims;
        self
    }

    /// Set attention dimension.
    pub fn with_attention_dim(mut self, dim: usize) -> Self {
        self.attention_dim = dim;
        self
    }

    /// Set number of attention heads.
    pub fn with_num_heads(mut self, heads: usize) -> Self {
        self.num_heads = heads;
        self
    }

    /// Set activation function.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }
}

/// Attention-augmented Actor-Critic network.
///
/// Combines an MLP backbone with self-attention to process sequences
/// of observations. This architecture can capture temporal dependencies
/// in the observation history, making it suitable for partially observable
/// environments.
///
/// Architecture:
/// 1. MLP feature extractor: obs -> features
/// 2. Self-attention over sequence of features
/// 3. Pooling (mean or last) to get sequence representation
/// 4. Actor head: representation -> action logits/mean
/// 5. Critic head: representation -> value
#[derive(Debug)]
pub struct AttentionActorCritic {
    /// Feature extractor MLP.
    feature_extractor: MLP,
    /// Self-attention layer.
    attention: SelfAttention,
    /// Actor head.
    actor_head: MLP,
    /// Critic head.
    critic_head: MLP,
    /// Log std for continuous actions (if applicable).
    log_std: Option<Tensor>,
    /// Configuration.
    config: AttentionActorCriticConfig,
}

impl AttentionActorCritic {
    /// Create a new AttentionActorCritic network.
    pub fn new(vb: VarBuilder<'_>, config: AttentionActorCriticConfig) -> CandleResult<Self> {
        // Feature extractor: obs_dim -> attention_dim
        let feature_config = MLPConfig::new(
            config.obs_dim,
            config.hidden_dims.clone(),
            config.attention_dim,
        )
        .with_activation(config.activation)
        .with_output_activation(config.activation);
        let feature_extractor = MLP::new(vb.pp("feature"), feature_config)?;

        // Self-attention layer
        let attn_config = SelfAttentionConfig::new(config.attention_dim, config.num_heads);
        let attention = SelfAttention::new(vb.pp("attention"), attn_config)?;

        // Actor head
        let actor_out_dim = if config.discrete {
            config.action_dim
        } else {
            config.action_dim // Mean only, log_std is separate
        };
        let actor_config = MLPConfig::new(config.attention_dim, vec![128], actor_out_dim)
            .with_activation(config.activation);
        let actor_head = MLP::new(vb.pp("actor"), actor_config)?;

        // Critic head
        let critic_config =
            MLPConfig::new(config.attention_dim, vec![128], 1).with_activation(config.activation);
        let critic_head = MLP::new(vb.pp("critic"), critic_config)?;

        // Log std for continuous actions
        let log_std = if !config.discrete {
            Some(vb.get_with_hints(&[config.action_dim], "log_std", candle_nn::Init::Const(0.0))?)
        } else {
            None
        };

        Ok(Self {
            feature_extractor,
            attention,
            actor_head,
            critic_head,
            log_std,
            config,
        })
    }

    /// Forward pass through the network.
    ///
    /// # Arguments
    /// * `obs_seq` - Observation sequence of shape [batch_size, seq_len, obs_dim]
    ///
    /// # Returns
    /// Tuple of (action_output, value) where:
    /// - For discrete: action_output is logits [batch_size, num_actions]
    /// - For continuous: action_output is mean [batch_size, action_dim]
    /// - value is [batch_size, 1]
    pub fn forward(&self, obs_seq: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        let (batch_size, seq_len, _) = obs_seq.dims3()?;

        // Extract features from each observation
        // Reshape to [batch*seq, obs_dim] for batch processing
        let obs_flat = obs_seq.reshape(&[batch_size * seq_len, self.config.obs_dim])?;
        let features_flat = self.feature_extractor.forward(&obs_flat)?;
        let features = features_flat.reshape(&[batch_size, seq_len, self.config.attention_dim])?;

        // Apply self-attention
        let attended = self.attention.forward(&features, None)?;

        // Pool over sequence (use last position)
        let pooled = attended.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        // Actor and critic heads
        let action_output = self.actor_head.forward(&pooled)?;
        let value = self.critic_head.forward(&pooled)?;

        Ok((action_output, value))
    }

    /// Get log standard deviation for continuous actions.
    pub fn get_log_std(&self) -> Option<&Tensor> {
        self.log_std.as_ref()
    }

    /// Check if this network is for discrete actions.
    pub fn is_discrete(&self) -> bool {
        self.config.discrete
    }

    /// Get the expected sequence length.
    pub fn seq_len(&self) -> usize {
        self.config.seq_len
    }

    /// Get the observation dimension.
    pub fn obs_dim(&self) -> usize {
        self.config.obs_dim
    }

    /// Get the action dimension.
    pub fn action_dim(&self) -> usize {
        self.config.action_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    fn setup_vb(device: &Device) -> VarBuilder<'static> {
        let varmap = VarMap::new();
        VarBuilder::from_varmap(&varmap, DType::F32, device)
    }

    #[test]
    fn test_self_attention_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = SelfAttentionConfig::new(64, 4);
        let attn = SelfAttention::new(vb, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[8, 16, 64], &device).unwrap();
        let y = attn.forward(&x, None).unwrap();

        assert_eq!(y.dims(), &[8, 16, 64]);
    }

    #[test]
    fn test_self_attention_causal() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = SelfAttentionConfig::new(64, 4).with_causal();
        let attn = SelfAttention::new(vb, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[4, 8, 64], &device).unwrap();
        let y = attn.forward(&x, None).unwrap();

        assert_eq!(y.dims(), &[4, 8, 64]);
    }

    #[test]
    fn test_cross_attention_shape() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = CrossAttentionConfig::new(64, 128, 64, 4);
        let attn = CrossAttention::new(vb, config).unwrap();

        let query = Tensor::randn(0.0f32, 1.0, &[4, 8, 64], &device).unwrap();
        let key_value = Tensor::randn(0.0f32, 1.0, &[4, 16, 128], &device).unwrap();

        let y = attn.forward(&query, &key_value, None).unwrap();

        assert_eq!(y.dims(), &[4, 8, 64]);
    }

    #[test]
    fn test_attention_actor_critic_discrete() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = AttentionActorCriticConfig::discrete(32, 4, 8);
        let ac = AttentionActorCritic::new(vb, config).unwrap();

        let obs_seq = Tensor::randn(0.0f32, 1.0, &[4, 8, 32], &device).unwrap();
        let (action_logits, value) = ac.forward(&obs_seq).unwrap();

        assert_eq!(action_logits.dims(), &[4, 4]); // [batch, num_actions]
        assert_eq!(value.dims(), &[4, 1]); // [batch, 1]
    }

    #[test]
    fn test_attention_actor_critic_continuous() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = AttentionActorCriticConfig::continuous(32, 2, 8);
        let ac = AttentionActorCritic::new(vb, config).unwrap();

        let obs_seq = Tensor::randn(0.0f32, 1.0, &[4, 8, 32], &device).unwrap();
        let (action_mean, value) = ac.forward(&obs_seq).unwrap();

        assert_eq!(action_mean.dims(), &[4, 2]); // [batch, action_dim]
        assert_eq!(value.dims(), &[4, 1]);
        assert!(ac.get_log_std().is_some());
    }

    #[test]
    fn test_causal_mask() {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, &device, DType::F32).unwrap();

        let mask_data: Vec<Vec<f32>> = mask.to_vec2().unwrap();

        // Check that upper triangle is -inf and lower triangle + diagonal is 0
        for i in 0..4 {
            for j in 0..4 {
                if j > i {
                    assert!(mask_data[i][j].is_infinite() && mask_data[i][j] < 0.0);
                } else {
                    assert!((mask_data[i][j] - 0.0).abs() < 1e-6);
                }
            }
        }
    }
}
