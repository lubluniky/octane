//! Recurrent Neural Network implementations: LSTM and GRU.
//!
//! These modules are essential for time-series data like trading,
//! where historical context influences current decisions.

use candle_core::{DType, Device, Tensor, Result as CandleResult, D, IndexOp};
use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

/// Configuration for recurrent layers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RNNConfig {
    /// Input feature dimension.
    pub input_dim: usize,
    /// Hidden state dimension.
    pub hidden_dim: usize,
    /// Number of stacked RNN layers.
    pub num_layers: usize,
    /// Dropout probability between layers (0.0 = no dropout).
    pub dropout: f32,
}

impl RNNConfig {
    /// Create a new RNN configuration.
    pub fn new(input_dim: usize, hidden_dim: usize) -> Self {
        Self {
            input_dim,
            hidden_dim,
            num_layers: 1,
            dropout: 0.0,
        }
    }

    /// Set the number of stacked layers.
    pub fn with_num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }
}

/// LSTM hidden state (h, c).
#[derive(Debug, Clone)]
pub struct LSTMState {
    /// Hidden state tensor of shape [num_layers, batch_size, hidden_dim].
    pub h: Tensor,
    /// Cell state tensor of shape [num_layers, batch_size, hidden_dim].
    pub c: Tensor,
}

impl LSTMState {
    /// Create a new LSTM state with given tensors.
    pub fn new(h: Tensor, c: Tensor) -> Self {
        Self { h, c }
    }

    /// Create a zero-initialized LSTM state.
    pub fn zeros(
        num_layers: usize,
        batch_size: usize,
        hidden_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Self> {
        let h = Tensor::zeros(&[num_layers, batch_size, hidden_dim], dtype, device)?;
        let c = Tensor::zeros(&[num_layers, batch_size, hidden_dim], dtype, device)?;
        Ok(Self { h, c })
    }

    /// Detach state from computation graph (for truncated BPTT).
    pub fn detach(&self) -> Self {
        Self {
            h: self.h.detach(),
            c: self.c.detach(),
        }
    }

    /// Get batch size from state.
    pub fn batch_size(&self) -> usize {
        self.h.dims()[1]
    }

    /// Get hidden dimension from state.
    pub fn hidden_dim(&self) -> usize {
        self.h.dims()[2]
    }
}

/// Single LSTM layer implementation.
#[derive(Debug)]
struct LSTMCell {
    /// Combined input-hidden weight matrix for gates [4*hidden_dim, input_dim + hidden_dim].
    weight_ih: Linear,
    weight_hh: Linear,
    #[allow(dead_code)]
    hidden_dim: usize,
}

impl LSTMCell {
    fn new(input_dim: usize, hidden_dim: usize, vb: VarBuilder<'_>) -> CandleResult<Self> {
        // LSTM has 4 gates: input, forget, cell, output
        // Each gate needs weights for input and hidden
        let weight_ih = candle_nn::linear(input_dim, 4 * hidden_dim, vb.pp("ih"))?;
        let weight_hh = candle_nn::linear(hidden_dim, 4 * hidden_dim, vb.pp("hh"))?;

        Ok(Self {
            weight_ih,
            weight_hh,
            hidden_dim,
        })
    }

    fn forward(&self, x: &Tensor, h: &Tensor, c: &Tensor) -> CandleResult<(Tensor, Tensor)> {
        // x: [batch_size, input_dim]
        // h: [batch_size, hidden_dim]
        // c: [batch_size, hidden_dim]

        // Compute gates
        let gates = (self.weight_ih.forward(x)? + self.weight_hh.forward(h)?)?;

        // Split into 4 gates
        let chunks = gates.chunk(4, D::Minus1)?;
        let i_gate = candle_nn::ops::sigmoid(&chunks[0])?; // Input gate
        let f_gate = candle_nn::ops::sigmoid(&chunks[1])?; // Forget gate
        let g_gate = chunks[2].tanh()?;                     // Cell gate (candidate)
        let o_gate = candle_nn::ops::sigmoid(&chunks[3])?; // Output gate

        // Update cell state: c_new = f * c + i * g
        let c_new = ((&f_gate * c)? + (&i_gate * &g_gate)?)?;

        // Update hidden state: h_new = o * tanh(c_new)
        let h_new = (&o_gate * c_new.tanh()?)?;

        Ok((h_new, c_new))
    }
}

/// Long Short-Term Memory (LSTM) network.
///
/// LSTM is effective for capturing long-term dependencies in sequential data,
/// making it ideal for trading strategies that need to remember market patterns.
#[derive(Debug)]
pub struct LSTM {
    cells: Vec<LSTMCell>,
    config: RNNConfig,
}

impl LSTM {
    /// Create a new LSTM network.
    pub fn new(vb: VarBuilder<'_>, config: RNNConfig) -> CandleResult<Self> {
        let mut cells = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let layer_input_dim = if i == 0 {
                config.input_dim
            } else {
                config.hidden_dim
            };
            let cell = LSTMCell::new(layer_input_dim, config.hidden_dim, vb.pp(format!("layer_{}", i)))?;
            cells.push(cell);
        }

        Ok(Self { cells, config })
    }

    /// Get the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Get the output dimension (same as hidden_dim).
    pub fn output_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Create initial zero state for given batch size.
    pub fn init_state(&self, batch_size: usize, device: &Device) -> CandleResult<LSTMState> {
        LSTMState::zeros(
            self.config.num_layers,
            batch_size,
            self.config.hidden_dim,
            DType::F32,
            device,
        )
    }

    /// Forward pass for a single timestep.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, input_dim]
    /// * `state` - Previous LSTM state
    ///
    /// # Returns
    /// Tuple of (output, new_state) where output has shape [batch_size, hidden_dim]
    pub fn forward_step(&self, x: &Tensor, state: &LSTMState) -> CandleResult<(Tensor, LSTMState)> {
        let batch_size = x.dims()[0];
        let mut h_states = Vec::with_capacity(self.config.num_layers);
        let mut c_states = Vec::with_capacity(self.config.num_layers);
        let mut layer_input = x.clone();

        for (i, cell) in self.cells.iter().enumerate() {
            // Extract h and c for this layer
            let h_i = state.h.i(i)?;
            let c_i = state.c.i(i)?;

            let (h_new, c_new) = cell.forward(&layer_input, &h_i, &c_i)?;

            // Store states
            h_states.push(h_new.unsqueeze(0)?);
            c_states.push(c_new.unsqueeze(0)?);

            // Output of this layer becomes input to next layer
            layer_input = h_states.last().unwrap().squeeze(0)?;
        }

        // Stack states back into [num_layers, batch_size, hidden_dim]
        let h = Tensor::cat(&h_states, 0)?;
        let c = Tensor::cat(&c_states, 0)?;

        let output = layer_input;
        let new_state = LSTMState::new(h, c);

        Ok((output, new_state))
    }

    /// Forward pass for a sequence.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, input_dim]
    /// * `state` - Optional initial state (zeros if None)
    ///
    /// # Returns
    /// Tuple of (outputs, final_state) where outputs has shape [batch_size, seq_len, hidden_dim]
    pub fn forward_sequence(
        &self,
        x: &Tensor,
        state: Option<&LSTMState>,
    ) -> CandleResult<(Tensor, LSTMState)> {
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let device = x.device();

        let mut current_state = match state {
            Some(s) => s.clone(),
            None => self.init_state(batch_size, device)?,
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = x.i((.., t, ..))?;
            let (out, new_state) = self.forward_step(&x_t, &current_state)?;
            outputs.push(out.unsqueeze(1)?);
            current_state = new_state;
        }

        let all_outputs = Tensor::cat(&outputs, 1)?;
        Ok((all_outputs, current_state))
    }
}

/// GRU hidden state.
#[derive(Debug, Clone)]
pub struct GRUState {
    /// Hidden state tensor of shape [num_layers, batch_size, hidden_dim].
    pub h: Tensor,
}

impl GRUState {
    /// Create a new GRU state.
    pub fn new(h: Tensor) -> Self {
        Self { h }
    }

    /// Create a zero-initialized GRU state.
    pub fn zeros(
        num_layers: usize,
        batch_size: usize,
        hidden_dim: usize,
        dtype: DType,
        device: &Device,
    ) -> CandleResult<Self> {
        let h = Tensor::zeros(&[num_layers, batch_size, hidden_dim], dtype, device)?;
        Ok(Self { h })
    }

    /// Detach state from computation graph.
    pub fn detach(&self) -> Self {
        Self {
            h: self.h.detach(),
        }
    }

    /// Get batch size from state.
    pub fn batch_size(&self) -> usize {
        self.h.dims()[1]
    }

    /// Get hidden dimension from state.
    pub fn hidden_dim(&self) -> usize {
        self.h.dims()[2]
    }
}

/// Single GRU cell implementation.
#[derive(Debug)]
struct GRUCell {
    weight_ih: Linear,
    weight_hh: Linear,
    #[allow(dead_code)]
    hidden_dim: usize,
}

impl GRUCell {
    fn new(input_dim: usize, hidden_dim: usize, vb: VarBuilder<'_>) -> CandleResult<Self> {
        // GRU has 3 gates: reset, update, new
        let weight_ih = candle_nn::linear(input_dim, 3 * hidden_dim, vb.pp("ih"))?;
        let weight_hh = candle_nn::linear(hidden_dim, 3 * hidden_dim, vb.pp("hh"))?;

        Ok(Self {
            weight_ih,
            weight_hh,
            hidden_dim,
        })
    }

    fn forward(&self, x: &Tensor, h: &Tensor) -> CandleResult<Tensor> {
        // x: [batch_size, input_dim]
        // h: [batch_size, hidden_dim]

        let gi = self.weight_ih.forward(x)?;
        let gh = self.weight_hh.forward(h)?;

        let gi_chunks = gi.chunk(3, D::Minus1)?;
        let gh_chunks = gh.chunk(3, D::Minus1)?;

        // Reset gate
        let r = candle_nn::ops::sigmoid(&(&gi_chunks[0] + &gh_chunks[0])?)?;
        // Update gate
        let z = candle_nn::ops::sigmoid(&(&gi_chunks[1] + &gh_chunks[1])?)?;
        // New gate (candidate hidden state)
        let n = (&gi_chunks[2] + (&r * &gh_chunks[2])?)?.tanh()?;

        // h_new = (1 - z) * n + z * h
        let one_minus_z = (Tensor::ones_like(&z)? - &z)?;
        let h_new = ((&one_minus_z * &n)? + (&z * h)?)?;

        Ok(h_new)
    }
}

/// Gated Recurrent Unit (GRU) network.
///
/// GRU is a lighter alternative to LSTM with fewer parameters,
/// often achieving comparable performance with faster training.
#[derive(Debug)]
pub struct GRU {
    cells: Vec<GRUCell>,
    config: RNNConfig,
}

impl GRU {
    /// Create a new GRU network.
    pub fn new(vb: VarBuilder<'_>, config: RNNConfig) -> CandleResult<Self> {
        let mut cells = Vec::with_capacity(config.num_layers);

        for i in 0..config.num_layers {
            let layer_input_dim = if i == 0 {
                config.input_dim
            } else {
                config.hidden_dim
            };
            let cell = GRUCell::new(layer_input_dim, config.hidden_dim, vb.pp(format!("layer_{}", i)))?;
            cells.push(cell);
        }

        Ok(Self { cells, config })
    }

    /// Get the hidden dimension.
    pub fn hidden_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Get the number of layers.
    pub fn num_layers(&self) -> usize {
        self.config.num_layers
    }

    /// Get the output dimension (same as hidden_dim).
    pub fn output_dim(&self) -> usize {
        self.config.hidden_dim
    }

    /// Create initial zero state for given batch size.
    pub fn init_state(&self, batch_size: usize, device: &Device) -> CandleResult<GRUState> {
        GRUState::zeros(
            self.config.num_layers,
            batch_size,
            self.config.hidden_dim,
            DType::F32,
            device,
        )
    }

    /// Forward pass for a single timestep.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, input_dim]
    /// * `state` - Previous GRU state
    ///
    /// # Returns
    /// Tuple of (output, new_state) where output has shape [batch_size, hidden_dim]
    pub fn forward_step(&self, x: &Tensor, state: &GRUState) -> CandleResult<(Tensor, GRUState)> {
        let mut h_states = Vec::with_capacity(self.config.num_layers);
        let mut layer_input = x.clone();

        for (i, cell) in self.cells.iter().enumerate() {
            let h_i = state.h.i(i)?;
            let h_new = cell.forward(&layer_input, &h_i)?;

            h_states.push(h_new.unsqueeze(0)?);
            layer_input = h_states.last().unwrap().squeeze(0)?;
        }

        let h = Tensor::cat(&h_states, 0)?;
        let output = layer_input;
        let new_state = GRUState::new(h);

        Ok((output, new_state))
    }

    /// Forward pass for a sequence.
    ///
    /// # Arguments
    /// * `x` - Input tensor of shape [batch_size, seq_len, input_dim]
    /// * `state` - Optional initial state (zeros if None)
    ///
    /// # Returns
    /// Tuple of (outputs, final_state) where outputs has shape [batch_size, seq_len, hidden_dim]
    pub fn forward_sequence(
        &self,
        x: &Tensor,
        state: Option<&GRUState>,
    ) -> CandleResult<(Tensor, GRUState)> {
        let dims = x.dims();
        let batch_size = dims[0];
        let seq_len = dims[1];
        let device = x.device();

        let mut current_state = match state {
            Some(s) => s.clone(),
            None => self.init_state(batch_size, device)?,
        };

        let mut outputs = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let x_t = x.i((.., t, ..))?;
            let (out, new_state) = self.forward_step(&x_t, &current_state)?;
            outputs.push(out.unsqueeze(1)?);
            current_state = new_state;
        }

        let all_outputs = Tensor::cat(&outputs, 1)?;
        Ok((all_outputs, current_state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::VarMap;

    #[test]
    fn test_lstm_forward_step() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = RNNConfig::new(32, 64);
        let lstm = LSTM::new(vb, config).unwrap();

        let batch_size = 4;
        let input = Tensor::randn(0.0f32, 1.0, &[batch_size, 32], &device).unwrap();
        let state = lstm.init_state(batch_size, &device).unwrap();

        let (output, new_state) = lstm.forward_step(&input, &state).unwrap();

        assert_eq!(output.dims(), &[batch_size, 64]);
        assert_eq!(new_state.h.dims(), &[1, batch_size, 64]);
        assert_eq!(new_state.c.dims(), &[1, batch_size, 64]);
    }

    #[test]
    fn test_lstm_forward_sequence() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = RNNConfig::new(32, 64);
        let lstm = LSTM::new(vb, config).unwrap();

        let batch_size = 4;
        let seq_len = 10;
        let input = Tensor::randn(0.0f32, 1.0, &[batch_size, seq_len, 32], &device).unwrap();

        let (outputs, final_state) = lstm.forward_sequence(&input, None).unwrap();

        assert_eq!(outputs.dims(), &[batch_size, seq_len, 64]);
        assert_eq!(final_state.h.dims(), &[1, batch_size, 64]);
    }

    #[test]
    fn test_gru_forward_step() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = RNNConfig::new(32, 64);
        let gru = GRU::new(vb, config).unwrap();

        let batch_size = 4;
        let input = Tensor::randn(0.0f32, 1.0, &[batch_size, 32], &device).unwrap();
        let state = gru.init_state(batch_size, &device).unwrap();

        let (output, new_state) = gru.forward_step(&input, &state).unwrap();

        assert_eq!(output.dims(), &[batch_size, 64]);
        assert_eq!(new_state.h.dims(), &[1, batch_size, 64]);
    }

    #[test]
    fn test_multi_layer_lstm() {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = RNNConfig::new(32, 64).with_num_layers(2);
        let lstm = LSTM::new(vb, config).unwrap();

        let batch_size = 4;
        let input = Tensor::randn(0.0f32, 1.0, &[batch_size, 32], &device).unwrap();
        let state = lstm.init_state(batch_size, &device).unwrap();

        let (output, new_state) = lstm.forward_step(&input, &state).unwrap();

        assert_eq!(output.dims(), &[batch_size, 64]);
        assert_eq!(new_state.h.dims(), &[2, batch_size, 64]);
        assert_eq!(new_state.c.dims(), &[2, batch_size, 64]);
    }
}
