//! Transformer architecture for sequence modeling in RL.
//!
//! This module provides a Transformer encoder implementation suitable for
//! reinforcement learning tasks, particularly for processing observation
//! sequences in partially observable environments or for Decision Transformer
//! style architectures.
//!
//! # Example
//! ```ignore
//! use octane_rs::networks::transformer::{TransformerEncoder, TransformerConfig};
//! use candle_core::Device;
//! use candle_nn::VarMap;
//!
//! let device = Device::Cpu;
//! let varmap = VarMap::new();
//! let vb = candle_nn::VarBuilder::from_varmap(&varmap, candle_core::DType::F32, &device);
//!
//! let config = TransformerConfig::new(256, 4, 4);
//! let transformer = TransformerEncoder::new(vb, config).unwrap();
//! ```

use candle_core::{Device, Result as CandleResult, Tensor, D};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

use super::mlp::Activation;
use super::normalization::{LayerNorm, LayerNormConfig};

/// Configuration for Transformer architecture.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformerConfig {
    /// Model dimension (embedding size).
    pub d_model: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Feed-forward network hidden dimension (typically 4 * d_model).
    pub d_ff: usize,
    /// Dropout probability.
    pub dropout: f64,
    /// Maximum sequence length for positional encoding.
    pub max_seq_len: usize,
    /// Whether to use causal (autoregressive) masking.
    pub causal: bool,
    /// Activation function for feed-forward network.
    pub activation: Activation,
    /// Whether to use pre-normalization (more stable training).
    pub pre_norm: bool,
}

impl TransformerConfig {
    /// Create a new Transformer configuration.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `num_heads` - Number of attention heads (d_model must be divisible by num_heads)
    /// * `num_layers` - Number of transformer encoder layers
    pub fn new(d_model: usize, num_heads: usize, num_layers: usize) -> Self {
        assert!(
            d_model % num_heads == 0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );

        Self {
            d_model,
            num_heads,
            num_layers,
            d_ff: d_model * 4,
            dropout: 0.0,
            max_seq_len: 1024,
            causal: false,
            activation: Activation::GELU,
            pre_norm: true,
        }
    }

    /// Set feed-forward dimension.
    pub fn with_d_ff(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
        self
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f64) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set maximum sequence length.
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = max_seq_len;
        self
    }

    /// Enable causal masking for autoregressive models.
    pub fn with_causal(mut self) -> Self {
        self.causal = true;
        self
    }

    /// Set activation function.
    pub fn with_activation(mut self, activation: Activation) -> Self {
        self.activation = activation;
        self
    }

    /// Use post-normalization instead of pre-normalization.
    pub fn with_post_norm(mut self) -> Self {
        self.pre_norm = false;
        self
    }
}

/// Multi-Head Attention module.
///
/// Implements scaled dot-product attention with multiple parallel attention heads.
/// This allows the model to jointly attend to information from different
/// representation subspaces.
#[derive(Debug)]
pub struct MultiHeadAttention {
    /// Query projection.
    wq: Linear,
    /// Key projection.
    wk: Linear,
    /// Value projection.
    wv: Linear,
    /// Output projection.
    wo: Linear,
    /// Number of attention heads.
    num_heads: usize,
    /// Head dimension (d_model / num_heads).
    head_dim: usize,
    /// Scaling factor for attention scores.
    scale: f64,
}

impl MultiHeadAttention {
    /// Create a new MultiHeadAttention module.
    pub fn new(vb: VarBuilder<'_>, d_model: usize, num_heads: usize) -> CandleResult<Self> {
        assert!(d_model % num_heads == 0);
        let head_dim = d_model / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        let wq = candle_nn::linear(d_model, d_model, vb.pp("wq"))?;
        let wk = candle_nn::linear(d_model, d_model, vb.pp("wk"))?;
        let wv = candle_nn::linear(d_model, d_model, vb.pp("wv"))?;
        let wo = candle_nn::linear(d_model, d_model, vb.pp("wo"))?;

        Ok(Self {
            wq,
            wk,
            wv,
            wo,
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Forward pass through multi-head attention.
    ///
    /// # Arguments
    /// * `query` - Query tensor [batch_size, query_len, d_model]
    /// * `key` - Key tensor [batch_size, key_len, d_model]
    /// * `value` - Value tensor [batch_size, key_len, d_model]
    /// * `mask` - Optional attention mask [batch_size, query_len, key_len] or [query_len, key_len]
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> CandleResult<Tensor> {
        let (batch_size, query_len, _) = query.dims3()?;
        let key_len = key.dims()[1];

        // Project Q, K, V
        let q = self.wq.forward(query)?;
        let k = self.wk.forward(key)?;
        let v = self.wv.forward(value)?;

        // Reshape for multi-head: [batch, seq, d_model] -> [batch, heads, seq, head_dim]
        let q = q
            .reshape(&[batch_size, query_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k = k
            .reshape(&[batch_size, key_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v = v
            .reshape(&[batch_size, key_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let attn_scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;

        // Apply mask if provided
        let attn_scores = match mask {
            Some(m) => {
                // Handle different mask dimensions
                let m = if m.dims().len() == 2 {
                    m.unsqueeze(0)?.unsqueeze(0)? // [1, 1, q_len, k_len]
                } else if m.dims().len() == 3 {
                    m.unsqueeze(1)? // [batch, 1, q_len, k_len]
                } else {
                    m.clone()
                };
                attn_scores.broadcast_add(&m)?
            }
            None => attn_scores,
        };

        // Softmax
        let attn_weights = candle_nn::ops::softmax(&attn_scores, D::Minus1)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, d_model]
        let d_model = self.num_heads * self.head_dim;
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape(&[batch_size, query_len, d_model])?;

        // Output projection
        self.wo.forward(&attn_output)
    }
}

/// Position-wise Feed-Forward Network.
///
/// Two linear transformations with an activation in between:
/// FFN(x) = activation(x @ W1 + b1) @ W2 + b2
#[derive(Debug)]
struct FeedForward {
    linear1: Linear,
    linear2: Linear,
    activation: Activation,
}

impl FeedForward {
    fn new(
        vb: VarBuilder<'_>,
        d_model: usize,
        d_ff: usize,
        activation: Activation,
    ) -> CandleResult<Self> {
        let linear1 = candle_nn::linear(d_model, d_ff, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(d_ff, d_model, vb.pp("linear2"))?;

        Ok(Self {
            linear1,
            linear2,
            activation,
        })
    }

    fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let h = self.linear1.forward(x)?;
        let h = self.activation.apply(&h)?;
        self.linear2.forward(&h)
    }
}

/// Single Transformer Encoder Layer.
///
/// Consists of:
/// 1. Multi-head self-attention with residual connection
/// 2. Position-wise feed-forward network with residual connection
/// Both sub-layers have layer normalization (pre or post).
#[derive(Debug)]
pub struct TransformerEncoderLayer {
    /// Self-attention.
    self_attn: MultiHeadAttention,
    /// Feed-forward network.
    ff: FeedForward,
    /// Layer norm for attention.
    norm1: LayerNorm,
    /// Layer norm for feed-forward.
    norm2: LayerNorm,
    /// Whether to use pre-normalization.
    pre_norm: bool,
}

impl TransformerEncoderLayer {
    /// Create a new transformer encoder layer.
    pub fn new(
        vb: VarBuilder<'_>,
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        activation: Activation,
        pre_norm: bool,
    ) -> CandleResult<Self> {
        let self_attn = MultiHeadAttention::new(vb.pp("self_attn"), d_model, num_heads)?;
        let ff = FeedForward::new(vb.pp("ff"), d_model, d_ff, activation)?;
        let norm1 = LayerNorm::new(vb.pp("norm1"), LayerNormConfig::new(d_model))?;
        let norm2 = LayerNorm::new(vb.pp("norm2"), LayerNormConfig::new(d_model))?;

        Ok(Self {
            self_attn,
            ff,
            norm1,
            norm2,
            pre_norm,
        })
    }

    /// Forward pass through the encoder layer.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, d_model]
    /// * `mask` - Optional attention mask
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        if self.pre_norm {
            // Pre-normalization: norm -> sublayer -> residual
            let normed = self.norm1.forward(x)?;
            let attn_out = self.self_attn.forward(&normed, &normed, &normed, mask)?;
            let x = (x + &attn_out)?;

            let normed = self.norm2.forward(&x)?;
            let ff_out = self.ff.forward(&normed)?;
            x + ff_out
        } else {
            // Post-normalization: sublayer -> residual -> norm
            let attn_out = self.self_attn.forward(x, x, x, mask)?;
            let x = self.norm1.forward(&(x + &attn_out)?)?;

            let ff_out = self.ff.forward(&x)?;
            self.norm2.forward(&(x + ff_out)?)
        }
    }
}

/// Sinusoidal Positional Encoding.
///
/// Adds positional information to the input embeddings using sine and cosine
/// functions of different frequencies. This allows the model to learn to
/// attend by relative positions.
#[derive(Debug)]
pub struct PositionalEncoding {
    /// Precomputed positional encodings [max_seq_len, d_model].
    encoding: Tensor,
}

impl PositionalEncoding {
    /// Create positional encoding.
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `max_seq_len` - Maximum sequence length
    /// * `device` - Device to create tensor on
    pub fn new(d_model: usize, max_seq_len: usize, device: &Device) -> CandleResult<Self> {
        let mut pe_data = vec![0.0f32; max_seq_len * d_model];

        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let div_term = (10000.0f64).powf((2.0 * (i / 2) as f64) / d_model as f64);
                let value = if i % 2 == 0 {
                    (pos as f64 / div_term).sin() as f32
                } else {
                    (pos as f64 / div_term).cos() as f32
                };
                pe_data[pos * d_model + i] = value;
            }
        }

        let encoding = Tensor::from_slice(&pe_data, &[max_seq_len, d_model], device)?;

        Ok(Self { encoding })
    }

    /// Add positional encoding to input.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, d_model]
    pub fn forward(&self, x: &Tensor) -> CandleResult<Tensor> {
        let seq_len = x.dims()[1];

        // Get positional encoding for this sequence length
        let pe = self.encoding.narrow(0, 0, seq_len)?;

        // Add to input (broadcasting over batch dimension)
        x.broadcast_add(&pe)
    }
}

/// Transformer Encoder.
///
/// A stack of transformer encoder layers with positional encoding.
/// Suitable for processing observation sequences in RL.
///
/// # Architecture
/// 1. Input projection (if input_dim != d_model)
/// 2. Positional encoding
/// 3. Stack of encoder layers
/// 4. Optional final layer normalization
#[derive(Debug)]
pub struct TransformerEncoder {
    /// Input projection (optional).
    input_proj: Option<Linear>,
    /// Positional encoding.
    pos_encoding: PositionalEncoding,
    /// Stack of encoder layers.
    layers: Vec<TransformerEncoderLayer>,
    /// Final layer normalization (for pre-norm architecture).
    final_norm: Option<LayerNorm>,
    /// Causal mask (if using causal attention).
    causal_mask: Option<Tensor>,
    /// Configuration.
    config: TransformerConfig,
}

impl TransformerEncoder {
    /// Create a new Transformer encoder.
    ///
    /// # Arguments
    /// * `vb` - Variable builder for weight initialization
    /// * `config` - Transformer configuration
    pub fn new(vb: VarBuilder<'_>, config: TransformerConfig) -> CandleResult<Self> {
        Self::with_input_dim(vb, config.d_model, config)
    }

    /// Create a new Transformer encoder with custom input dimension.
    ///
    /// # Arguments
    /// * `vb` - Variable builder for weight initialization
    /// * `input_dim` - Input feature dimension (will be projected to d_model)
    /// * `config` - Transformer configuration
    pub fn with_input_dim(
        vb: VarBuilder<'_>,
        input_dim: usize,
        config: TransformerConfig,
    ) -> CandleResult<Self> {
        let device = vb.device();

        // Input projection if dimensions don't match
        let input_proj = if input_dim != config.d_model {
            Some(candle_nn::linear(
                input_dim,
                config.d_model,
                vb.pp("input_proj"),
            )?)
        } else {
            None
        };

        // Positional encoding
        let pos_encoding = PositionalEncoding::new(config.d_model, config.max_seq_len, device)?;

        // Encoder layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let layer = TransformerEncoderLayer::new(
                vb.pp(format!("layer_{}", i)),
                config.d_model,
                config.num_heads,
                config.d_ff,
                config.activation,
                config.pre_norm,
            )?;
            layers.push(layer);
        }

        // Final normalization for pre-norm architecture
        let final_norm = if config.pre_norm {
            Some(LayerNorm::new(
                vb.pp("final_norm"),
                LayerNormConfig::new(config.d_model),
            )?)
        } else {
            None
        };

        // Causal mask
        let causal_mask = if config.causal {
            Some(create_causal_mask(config.max_seq_len, device)?)
        } else {
            None
        };

        Ok(Self {
            input_proj,
            pos_encoding,
            layers,
            final_norm,
            causal_mask,
            config,
        })
    }

    /// Forward pass through the transformer encoder.
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, input_dim]
    /// * `mask` - Optional additional attention mask
    ///
    /// # Returns
    /// Output tensor [batch_size, seq_len, d_model]
    pub fn forward(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let seq_len = x.dims()[1];

        // Input projection
        let mut h = match &self.input_proj {
            Some(proj) => proj.forward(x)?,
            None => x.clone(),
        };

        // Add positional encoding
        h = self.pos_encoding.forward(&h)?;

        // Prepare mask
        let effective_mask = match (&self.causal_mask, mask) {
            (Some(causal), Some(user_mask)) => {
                // Combine causal mask with user mask
                let causal_slice = causal.narrow(0, 0, seq_len)?.narrow(1, 0, seq_len)?;
                Some(causal_slice.broadcast_add(user_mask)?)
            }
            (Some(causal), None) => {
                let causal_slice = causal.narrow(0, 0, seq_len)?.narrow(1, 0, seq_len)?;
                Some(causal_slice)
            }
            (None, Some(user_mask)) => Some(user_mask.clone()),
            (None, None) => None,
        };

        // Pass through encoder layers
        for layer in &self.layers {
            h = layer.forward(&h, effective_mask.as_ref())?;
        }

        // Final normalization
        match &self.final_norm {
            Some(norm) => norm.forward(&h),
            None => Ok(h),
        }
    }

    /// Get output from the last position (for classification/RL).
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, input_dim]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Output tensor [batch_size, d_model]
    pub fn forward_last(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let h = self.forward(x, mask)?;
        let seq_len = h.dims()[1];
        h.narrow(1, seq_len - 1, 1)?.squeeze(1)
    }

    /// Get mean-pooled output (for sequence representation).
    ///
    /// # Arguments
    /// * `x` - Input tensor [batch_size, seq_len, input_dim]
    /// * `mask` - Optional attention mask
    ///
    /// # Returns
    /// Output tensor [batch_size, d_model]
    pub fn forward_mean(&self, x: &Tensor, mask: Option<&Tensor>) -> CandleResult<Tensor> {
        let h = self.forward(x, mask)?;
        h.mean(1)
    }

    /// Get the model dimension.
    pub fn d_model(&self) -> usize {
        self.config.d_model
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Check if using causal attention.
    pub fn is_causal(&self) -> bool {
        self.config.causal
    }
}

/// Create a causal attention mask.
fn create_causal_mask(max_seq_len: usize, device: &Device) -> CandleResult<Tensor> {
    let mut mask_data = vec![0.0f32; max_seq_len * max_seq_len];
    for i in 0..max_seq_len {
        for j in (i + 1)..max_seq_len {
            mask_data[i * max_seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Tensor::from_slice(&mask_data, &[max_seq_len, max_seq_len], device)
}

/// Decision Transformer configuration.
///
/// Specialized configuration for Decision Transformer style RL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTransformerConfig {
    /// State/observation dimension.
    pub state_dim: usize,
    /// Action dimension.
    pub action_dim: usize,
    /// Hidden dimension.
    pub hidden_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Maximum episode length.
    pub max_ep_len: usize,
    /// Context length (number of timesteps to consider).
    pub context_len: usize,
    /// Dropout probability.
    pub dropout: f64,
}

impl DecisionTransformerConfig {
    /// Create a new Decision Transformer configuration.
    pub fn new(state_dim: usize, action_dim: usize) -> Self {
        Self {
            state_dim,
            action_dim,
            hidden_dim: 128,
            num_heads: 4,
            num_layers: 3,
            max_ep_len: 1000,
            context_len: 20,
            dropout: 0.1,
        }
    }

    /// Set hidden dimension.
    pub fn with_hidden_dim(mut self, dim: usize) -> Self {
        self.hidden_dim = dim;
        self
    }

    /// Set number of layers.
    pub fn with_num_layers(mut self, n: usize) -> Self {
        self.num_layers = n;
        self
    }

    /// Set context length.
    pub fn with_context_len(mut self, len: usize) -> Self {
        self.context_len = len;
        self
    }
}

/// Decision Transformer for offline RL.
///
/// Implements the Decision Transformer architecture which frames RL as
/// sequence modeling. It takes (return-to-go, state, action) sequences
/// and predicts the next action conditioned on desired returns.
#[derive(Debug)]
pub struct DecisionTransformer {
    /// State embedding.
    state_embed: Linear,
    /// Action embedding.
    action_embed: Linear,
    /// Return embedding.
    return_embed: Linear,
    /// Timestep embedding.
    timestep_embed: Tensor,
    /// Transformer backbone.
    transformer: TransformerEncoder,
    /// Action prediction head.
    action_head: Linear,
    /// Configuration.
    config: DecisionTransformerConfig,
}

impl DecisionTransformer {
    /// Create a new Decision Transformer.
    pub fn new(vb: VarBuilder<'_>, config: DecisionTransformerConfig) -> CandleResult<Self> {
        let hidden_dim = config.hidden_dim;

        // Embeddings
        let state_embed = candle_nn::linear(config.state_dim, hidden_dim, vb.pp("state_embed"))?;
        let action_embed = candle_nn::linear(config.action_dim, hidden_dim, vb.pp("action_embed"))?;
        let return_embed = candle_nn::linear(1, hidden_dim, vb.pp("return_embed"))?;

        // Learnable timestep embedding
        let timestep_embed = vb.get_with_hints(
            &[config.max_ep_len, hidden_dim],
            "timestep_embed",
            candle_nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        )?;

        // Transformer backbone
        let transformer_config =
            TransformerConfig::new(hidden_dim, config.num_heads, config.num_layers)
                .with_causal()
                .with_max_seq_len(config.context_len * 3) // 3 tokens per timestep
                .with_dropout(config.dropout);
        let transformer = TransformerEncoder::new(vb.pp("transformer"), transformer_config)?;

        // Action prediction head
        let action_head = candle_nn::linear(hidden_dim, config.action_dim, vb.pp("action_head"))?;

        Ok(Self {
            state_embed,
            action_embed,
            return_embed,
            timestep_embed,
            transformer,
            action_head,
            config,
        })
    }

    /// Forward pass for training.
    ///
    /// # Arguments
    /// * `states` - State sequence [batch_size, seq_len, state_dim]
    /// * `actions` - Action sequence [batch_size, seq_len, action_dim]
    /// * `returns_to_go` - Return-to-go sequence [batch_size, seq_len, 1]
    /// * `timesteps` - Timestep indices [batch_size, seq_len]
    ///
    /// # Returns
    /// Predicted actions [batch_size, seq_len, action_dim]
    pub fn forward(
        &self,
        states: &Tensor,
        actions: &Tensor,
        returns_to_go: &Tensor,
        timesteps: &Tensor,
    ) -> CandleResult<Tensor> {
        let (_, seq_len, _) = states.dims3()?;

        // Embed inputs
        let state_emb = self.state_embed.forward(states)?;
        let action_emb = self.action_embed.forward(actions)?;
        let return_emb = self.return_embed.forward(returns_to_go)?;

        // Add timestep embeddings
        let timestep_emb = self.get_timestep_embeddings(timesteps)?;
        let state_emb = (&state_emb + &timestep_emb)?;
        let action_emb = (&action_emb + &timestep_emb)?;
        let return_emb = (&return_emb + &timestep_emb)?;

        // Interleave: [R1, S1, A1, R2, S2, A2, ...]
        // Shape: [batch, seq_len * 3, hidden_dim]
        let tokens = self.interleave_tokens(&return_emb, &state_emb, &action_emb)?;

        // Pass through transformer
        let output = self.transformer.forward(&tokens, None)?;

        // Extract state positions (index 1, 4, 7, ... in the interleaved sequence)
        // These correspond to positions after seeing (R, S) pairs
        let state_outputs = self.extract_state_outputs(&output, seq_len)?;

        // Predict actions
        self.action_head.forward(&state_outputs)
    }

    /// Get timestep embeddings.
    fn get_timestep_embeddings(&self, timesteps: &Tensor) -> CandleResult<Tensor> {
        let (batch_size, seq_len) = timesteps.dims2()?;
        let hidden_dim = self.config.hidden_dim;

        // Flatten timesteps for indexing
        let timesteps_flat = timesteps.flatten_all()?;

        // Index into timestep embedding table
        let emb_flat = self.timestep_embed.index_select(&timesteps_flat, 0)?;

        // Reshape back
        emb_flat.reshape(&[batch_size, seq_len, hidden_dim])
    }

    /// Interleave (return, state, action) tokens.
    fn interleave_tokens(
        &self,
        returns: &Tensor,
        states: &Tensor,
        actions: &Tensor,
    ) -> CandleResult<Tensor> {
        let (batch_size, seq_len, hidden_dim) = returns.dims3()?;

        // Stack along a new dimension: [batch, seq, 3, hidden]
        let stacked = Tensor::stack(&[returns, states, actions], 2)?;

        // Reshape to interleave: [batch, seq * 3, hidden]
        stacked.reshape(&[batch_size, seq_len * 3, hidden_dim])
    }

    /// Extract outputs corresponding to state positions.
    fn extract_state_outputs(&self, output: &Tensor, seq_len: usize) -> CandleResult<Tensor> {
        // State positions are at indices 1, 4, 7, ... (after R, before A)
        // We want to predict action after seeing (R, S)
        let indices: Vec<u32> = (0..seq_len).map(|i| (i * 3 + 1) as u32).collect();
        let indices_tensor = Tensor::from_slice(&indices, &[seq_len], output.device())?;

        // Index select along sequence dimension
        output.index_select(&indices_tensor, 1)
    }

    /// Predict action for a single timestep (inference).
    ///
    /// # Arguments
    /// * `states` - State history [1, context_len, state_dim]
    /// * `actions` - Action history [1, context_len, action_dim]
    /// * `returns_to_go` - Return-to-go history [1, context_len, 1]
    /// * `timesteps` - Timestep indices [1, context_len]
    ///
    /// # Returns
    /// Predicted action [1, action_dim]
    pub fn predict_action(
        &self,
        states: &Tensor,
        actions: &Tensor,
        returns_to_go: &Tensor,
        timesteps: &Tensor,
    ) -> CandleResult<Tensor> {
        let predicted = self.forward(states, actions, returns_to_go, timesteps)?;
        // Return the last predicted action
        let seq_len = predicted.dims()[1];
        predicted.narrow(1, seq_len - 1, 1)?.squeeze(1)
    }

    /// Get the action dimension.
    pub fn action_dim(&self) -> usize {
        self.config.action_dim
    }

    /// Get the state dimension.
    pub fn state_dim(&self) -> usize {
        self.config.state_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_multi_head_attention() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let mha = MultiHeadAttention::new(vb, 64, 4).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[4, 8, 64], &device).unwrap();
        let output = mha.forward(&x, &x, &x, None).unwrap();

        assert_eq!(output.dims(), &[4, 8, 64]);
    }

    #[test]
    fn test_positional_encoding() {
        let device = Device::Cpu;
        let pe = PositionalEncoding::new(64, 100, &device).unwrap();

        let x = Tensor::zeros(&[4, 16, 64], DType::F32, &device).unwrap();
        let output = pe.forward(&x).unwrap();

        assert_eq!(output.dims(), &[4, 16, 64]);

        // Check that positional encoding adds non-zero values
        let sum = output.sum_all().unwrap().to_scalar::<f32>().unwrap();
        assert!(sum.abs() > 0.0);
    }

    #[test]
    fn test_transformer_encoder() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = TransformerConfig::new(64, 4, 2);
        let encoder = TransformerEncoder::new(vb, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[4, 16, 64], &device).unwrap();
        let output = encoder.forward(&x, None).unwrap();

        assert_eq!(output.dims(), &[4, 16, 64]);
    }

    #[test]
    fn test_transformer_encoder_causal() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = TransformerConfig::new(64, 4, 2).with_causal();
        let encoder = TransformerEncoder::new(vb, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[4, 8, 64], &device).unwrap();
        let output = encoder.forward(&x, None).unwrap();

        assert_eq!(output.dims(), &[4, 8, 64]);
        assert!(encoder.is_causal());
    }

    #[test]
    fn test_transformer_encoder_with_input_projection() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = TransformerConfig::new(64, 4, 2);
        let encoder = TransformerEncoder::with_input_dim(vb, 32, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[4, 16, 32], &device).unwrap();
        let output = encoder.forward(&x, None).unwrap();

        assert_eq!(output.dims(), &[4, 16, 64]); // Output is d_model
    }

    #[test]
    fn test_transformer_encoder_pooling() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = TransformerConfig::new(64, 4, 2);
        let encoder = TransformerEncoder::new(vb, config).unwrap();

        let x = Tensor::randn(0.0f32, 1.0, &[4, 16, 64], &device).unwrap();

        let last_output = encoder.forward_last(&x, None).unwrap();
        assert_eq!(last_output.dims(), &[4, 64]);

        let mean_output = encoder.forward_mean(&x, None).unwrap();
        assert_eq!(mean_output.dims(), &[4, 64]);
    }

    #[test]
    fn test_decision_transformer() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = DecisionTransformerConfig::new(8, 2)
            .with_hidden_dim(32)
            .with_num_layers(2)
            .with_context_len(10);
        let dt = DecisionTransformer::new(vb, config).unwrap();

        let batch_size = 4;
        let seq_len = 10;

        let states = Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, 8], &device).unwrap();
        let actions = Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, 2], &device).unwrap();
        let returns = Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, 1], &device).unwrap();
        let timesteps = Tensor::zeros(&[batch_size, seq_len], DType::U32, &device).unwrap();

        let predicted = dt.forward(&states, &actions, &returns, &timesteps).unwrap();

        assert_eq!(predicted.dims(), &[batch_size, seq_len, 2]);
    }

    #[test]
    fn test_decision_transformer_predict() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = DecisionTransformerConfig::new(8, 2)
            .with_hidden_dim(32)
            .with_num_layers(1)
            .with_context_len(5);
        let dt = DecisionTransformer::new(vb, config).unwrap();

        let states = Tensor::randn(0.0f32, 1.0, &[1, 5, 8], &device).unwrap();
        let actions = Tensor::randn(0.0f32, 1.0, &[1, 5, 2], &device).unwrap();
        let returns = Tensor::randn(0.0f32, 1.0, &[1, 5, 1], &device).unwrap();
        let timesteps = Tensor::zeros(&[1, 5], DType::U32, &device).unwrap();

        let action = dt
            .predict_action(&states, &actions, &returns, &timesteps)
            .unwrap();

        assert_eq!(action.dims(), &[1, 2]);
    }
}
