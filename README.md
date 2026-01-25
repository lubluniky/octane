# Rocket-RS

<div align="center">

<img src="benchmarks/charts/hero_chart.png" alt="RocketRL Performance" width="800"/>

**High-Performance Reinforcement Learning Library for Rust**

*Blazingly fast RL for algorithmic trading and beyond*

[![CI](https://github.com/lubluniky/rocket-rs/workflows/CI/badge.svg)](https://github.com/lubluniky/rocket-rs/actions)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](#license)
[![Candle](https://img.shields.io/badge/backend-Candle-green.svg)](https://github.com/huggingface/candle)
[![Crates.io](https://img.shields.io/crates/v/rocket-rs.svg)](https://crates.io/crates/rocket-rs)

</div>

---

## Why RocketRL?

Python's Global Interpreter Lock (GIL) and dynamic typing create fundamental performance bottlenecks for RL training. RocketRL eliminates these constraints with pure Rust:

| Metric | Python (Gymnasium) | RocketRL (Rust) | Speedup |
|--------|-------------------|-----------------|---------|
| Single Env Step | 2.84 μs | 0.227 μs | **12.5x** |
| Environment Reset | 1.06 μs | 0.222 μs | **4.8x** |
| VecEnv (128 parallel) | 437 μs | 123 μs | **3.6x** |
| VecEnv (1024 parallel) | 3549 μs | 585 μs | **6.1x** |
| Memory Usage | 100% | ~35% | **~3x less** |

## Features

- **Zero-Cost Abstractions** - Rust's ownership model eliminates runtime overhead
- **Vectorized Environments** - Run 1000s of parallel simulations on CPU
- **GPU Acceleration** - Native Metal (M1-M4) and CUDA (H100/H200) support
- **Production-Ready** - Memory-safe, thread-safe, no garbage collection pauses
- **Complete Algorithms** - PPO, A2C, SAC, TD3, DDPG, DQN with Double DQN
- **Time-Series Ready** - LSTM/GRU networks for trading and sequential decisions
- **Training Logging** - JSON-based logging for background training with TUI monitoring

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Benchmarks](#benchmarks)
- [API Reference](#api-reference)
- [Examples](#examples)
- [GPU Acceleration](#gpu-acceleration)
- [Algorithms](#algorithms)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)

---

## Installation

Add Rocket-RS to your `Cargo.toml`:

```toml
[dependencies]
rocket-rs = "0.1"

# For Apple Silicon (M1/M2/M3/M4)
# rocket-rs = { version = "0.1", features = ["metal"] }

# For NVIDIA GPUs
# rocket-rs = { version = "0.1", features = ["cuda"] }
```

### Build from Source

```bash
git clone https://github.com/rocketrl/rocket-rs
cd rocket-rs

# CPU only
cargo build --release

# Apple Silicon with Metal
cargo build --release --features metal

# NVIDIA GPU with CUDA
cargo build --release --features cuda
```

### System Requirements

| Platform | Minimum | Recommended |
|----------|---------|-------------|
| **Rust** | 1.75+ | 1.80+ |
| **macOS** | 12.0+ (Metal) | 14.0+ (M4) |
| **Linux** | CUDA 11.8+ | CUDA 12.0+ |
| **Memory** | 4 GB | 16+ GB |

---

## Quick Start

```rust
use rocket_rs::prelude::*;
use rocket_rs::envs::{TradingEnv, MarketData};
use rocket_rs::algorithms::{PPOConfig, PPOAgent, RLAlgorithm};
use std::path::Path;

fn main() -> rocket_rs::Result<()> {
    // 1. Select device
    let device = Device::cpu();  // or Device::m4_metal() / Device::cuda(0)

    // 2. Create environment
    let data = MarketData::synthetic(10000, 42);
    let env = TradingEnv::new(data)?;

    // 3. Vectorize for parallel simulation
    let vec_env = env.make_vectorized(128);  // 128 parallel envs

    // 4. Configure PPO
    let config = PPOConfig::default()
        .learning_rate(3e-4)
        .n_steps(2048)
        .batch_size(64)
        .n_epochs(10)
        .gamma(0.99)
        .gae_lambda(0.95)
        .clip_range(0.2);

    // 5. Create and train agent
    let mut agent = PPOAgent::new(config, vec_env, device)?;

    agent.train(1_000_000, |metrics| {
        println!(
            "Step {} | Reward: {:.2} | Loss: {:.4}",
            metrics.timesteps,
            metrics.mean_reward,
            metrics.policy_loss
        );
    })?;

    // 6. Save model
    agent.save(Path::new("trading_agent.safetensors"))?;

    Ok(())
}
```

---

## Architecture

```
rocket-rs/
├── src/
│   ├── core/              # Device, Error, Tensor abstractions
│   │   ├── device.rs      # CPU/Metal/CUDA device management
│   │   ├── error.rs       # RocketError types
│   │   └── tensor.rs      # TensorBackend trait over Candle
│   │
│   ├── envs/              # Gym-like environment interface
│   │   ├── traits.rs      # Environment trait
│   │   ├── space.rs       # BoxSpace, DiscreteSpace
│   │   ├── vecenv.rs      # Vectorized environments (Rayon)
│   │   └── trading.rs     # Example trading environment
│   │
│   ├── networks/          # Neural network architectures
│   │   ├── mlp.rs         # Multi-Layer Perceptron
│   │   ├── rnn.rs         # LSTM and GRU
│   │   └── actor_critic.rs # Combined policy-value network
│   │
│   ├── distributions/     # Action distributions
│   │   ├── categorical.rs # Discrete actions
│   │   └── gaussian.rs    # Continuous actions (DiagGaussian)
│   │
│   ├── buffer/            # Experience storage
│   │   └── mod.rs         # RolloutBuffer with GAE
│   │
│   └── algorithms/        # RL algorithms
│       ├── ppo.rs         # Proximal Policy Optimization
│       ├── a2c.rs         # Advantage Actor-Critic
│       └── traits.rs      # RLAlgorithm trait
│
├── examples/
│   └── trading_ppo.rs     # Complete trading example
│
└── benches/               # Criterion benchmarks
    ├── env_benchmark.rs
    └── ppo_benchmark.rs
```

---

## Benchmarks

All benchmarks run on Apple M4 Max, comparing RocketRL against Python (NumPy/Gymnasium).

### Environment Step Performance

<img src="benchmarks/charts/env_comparison.png" alt="Environment Comparison" width="800"/>

**Key findings (Apple M4 Max, Jan 2025):**
- Single environment step: **12.5x faster** (227 ns vs 2.84 μs)
- VecEnv with 128 parallel envs: **3.6x faster** (123 μs vs 437 μs)
- VecEnv with 1024 parallel envs: **6.1x faster** (585 μs vs 3549 μs)

### Throughput Scaling

<img src="benchmarks/charts/throughput.png" alt="Throughput Scaling" width="700"/>

RocketRL achieves **1.7M+ environment steps per second** with 1024 parallel environments.

### Tensor Operations

<img src="benchmarks/charts/tensor_ops.png" alt="Tensor Operations" width="800"/>

### Speedup Summary

<img src="benchmarks/charts/speedup_summary.png" alt="Speedup Summary" width="700"/>

### PPO Algorithm Performance

<img src="benchmarks/charts/ppo_operations.png" alt="PPO Operations" width="800"/>

| Operation | Batch 64 | Batch 256 | Batch 1024 |
|-----------|----------|-----------|------------|
| PPO Loss | 1.10 μs | 1.83 μs | 4.08 μs |
| MLP Forward | 19.7 μs | 563 μs | 3.51 ms |
| Advantage Norm | 1.90 μs | 6.17 μs | 24.3 μs |

### Run Benchmarks Yourself

```bash
# Python baseline
pip install numpy torch matplotlib
python benchmarks/python_baseline.py

# Rust benchmarks
cargo bench

# Generate visualization
python benchmarks/visualize.py
```

---

## API Reference

### Core Types

#### Device

```rust
pub enum Device {
    Cpu,
    #[cfg(feature = "metal")]
    Metal,
    #[cfg(feature = "cuda")]
    Cuda(usize),
}

// Usage
let cpu = Device::cpu();
let metal = Device::m4_metal();  // requires "metal" feature
let cuda = Device::cuda(0);      // requires "cuda" feature
```

#### Environment Trait

```rust
pub trait Environment: Send + Sync + 'static {
    type ObsSpace: Space;
    type ActSpace: Space;

    fn observation_space(&self) -> &Self::ObsSpace;
    fn action_space(&self) -> &Self::ActSpace;
    fn reset(&mut self, device: &Device) -> Result<Tensor>;
    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult>;
}
```

#### VecEnv

```rust
// Create 128 parallel environments
let vec_env = env.make_vectorized(128);

// Step all environments at once
let observations = vec_env.reset(&device)?;
let result = vec_env.step(&actions, &device)?;

// Access batched results
println!("Observations: {:?}", result.observations.shape());  // [128, obs_dim]
println!("Rewards: {:?}", result.rewards.shape());            // [128]
println!("Dones: {:?}", result.dones()?.shape());             // [128]
```

### Configuration

#### PPOConfig

```rust
let config = PPOConfig::default()
    .learning_rate(3e-4)      // Optimizer learning rate
    .n_steps(2048)            // Steps per rollout
    .batch_size(64)           // Minibatch size
    .n_epochs(10)             // Epochs per update
    .gamma(0.99)              // Discount factor
    .gae_lambda(0.95)         // GAE lambda
    .clip_range(0.2)          // PPO clipping
    .vf_coef(0.5)             // Value loss coefficient
    .ent_coef(0.01)           // Entropy bonus
    .max_grad_norm(0.5);      // Gradient clipping
```

#### A2CConfig

```rust
let config = A2CConfig::default()
    .learning_rate(7e-4)
    .n_steps(5)
    .gamma(0.99)
    .gae_lambda(1.0)          // 1.0 = no GAE (Monte Carlo)
    .vf_coef(0.5)
    .ent_coef(0.01);
```

### Neural Networks

#### MLP

```rust
use rocket_rs::networks::{MLP, MLPConfig, Activation};

let config = MLPConfig {
    input_dim: 64,
    hidden_dims: vec![256, 256],
    output_dim: 4,
    activation: Activation::ReLU,
};

let mlp = MLP::new(&var_builder, config)?;
let output = mlp.forward(&input)?;
```

#### LSTM

```rust
use rocket_rs::networks::{LSTM, LSTMState, RNNConfig};

let config = RNNConfig {
    input_dim: 64,
    hidden_dim: 128,
    num_layers: 2,
    dropout: 0.0,
};

let lstm = LSTM::new(&var_builder, config)?;
let (output, new_state) = lstm.forward_step(&input, &state)?;
```

#### ActorCritic

```rust
use rocket_rs::networks::{ActorCritic, ActorCriticConfig, ActionSpace};

let config = ActorCriticConfig::continuous(obs_dim, action_dim)
    .with_hidden_dims(vec![256, 256])
    .with_lstm(128);  // Add LSTM layer

let ac = ActorCritic::new(&var_builder, config)?;
let (action_params, value, new_state) = ac.forward(&obs, state)?;
```

### Distributions

```rust
use rocket_rs::distributions::{Categorical, DiagGaussian, Distribution};

// Discrete actions
let categorical = Categorical::from_logits(logits)?;
let action = categorical.sample()?;
let log_prob = categorical.log_prob(&action)?;
let entropy = categorical.entropy()?;

// Continuous actions
let gaussian = DiagGaussian::new(mean, log_std)?;
let action = gaussian.sample()?;
let log_prob = gaussian.log_prob(&action)?;
```

---

## Examples

### Custom Environment

```rust
use rocket_rs::envs::{Environment, BoxSpace, StepResult, StepInfo};
use rocket_rs::core::{Device, Result};
use candle_core::Tensor;

#[derive(Clone)]
pub struct MyEnv {
    state: Vec<f32>,
    obs_space: BoxSpace,
    act_space: BoxSpace,
}

impl Environment for MyEnv {
    type ObsSpace = BoxSpace;
    type ActSpace = BoxSpace;

    fn observation_space(&self) -> &Self::ObsSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &Self::ActSpace {
        &self.act_space
    }

    fn reset(&mut self, device: &Device) -> Result<Tensor> {
        self.state = vec![0.0; 4];
        let candle_dev = device.to_candle()?;
        Ok(Tensor::from_slice(&self.state, &[4], &candle_dev)?)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let action_vec: Vec<f32> = action.to_vec1()?;

        // Update state based on action
        for (s, a) in self.state.iter_mut().zip(&action_vec) {
            *s += a;
        }

        let reward = -self.state.iter().map(|x| x.powi(2)).sum::<f32>();
        let done = self.state.iter().any(|x| x.abs() > 10.0);

        let candle_dev = device.to_candle()?;
        let obs = Tensor::from_slice(&self.state, &[4], &candle_dev)?;

        Ok(StepResult {
            observation: obs,
            reward,
            terminated: done,
            truncated: false,
            info: None,
        })
    }
}
```

### Training with Callbacks

```rust
use rocket_rs::algorithms::{PPOAgent, PPOConfig, RLAlgorithm, TrainMetrics};

let mut agent = PPOAgent::new(config, vec_env, device)?;

// Simple callback
agent.train(1_000_000, |m: &TrainMetrics| {
    if m.timesteps % 10_000 == 0 {
        println!(
            "[{:>7}] reward={:>7.2} | policy_loss={:.4} | value_loss={:.4} | entropy={:.4}",
            m.timesteps, m.mean_reward, m.policy_loss, m.value_loss, m.entropy
        );
    }
})?;
```

### Model Saving and Loading

```rust
use std::path::Path;
use rocket_rs::algorithms::RLAlgorithm;

// Save
agent.save(Path::new("models/ppo_trading.safetensors"))?;

// Load
let agent = PPOAgent::load(Path::new("models/ppo_trading.safetensors"), vec_env, device)?;
```

---

## GPU Acceleration

### Apple Silicon (Metal)

RocketRL leverages Metal Performance Shaders for M1/M2/M3/M4 chips:

```rust
// Enable Metal
let device = Device::m4_metal();

// All operations automatically use Metal
let agent = PPOAgent::new(config, vec_env, device)?;
agent.train(1_000_000, |_| {})?;
```

Build with Metal support:

```bash
cargo build --release --features metal
```

### NVIDIA CUDA

For H100/H200 and other NVIDIA GPUs:

```rust
// Select CUDA device
let device = Device::cuda(0);  // First GPU
let device = Device::cuda(1);  // Second GPU (multi-GPU)
```

Build with CUDA support:

```bash
cargo build --release --features cuda
```

### Multi-GPU Training

```rust
// Coming soon: distributed training across multiple GPUs
let devices = vec![Device::cuda(0), Device::cuda(1)];
```

---

## Algorithms

RocketRL provides 6 state-of-the-art RL algorithms:

### On-Policy Algorithms

#### PPO (Proximal Policy Optimization)

The default algorithm, recommended for most use cases.

```rust
let config = PPOConfig::default()
    .learning_rate(3e-4)
    .n_steps(2048)
    .clip_range(0.2);
let agent = PPOAgent::new(config, vec_env, device)?;
```

#### A2C (Advantage Actor-Critic)

Simpler synchronous actor-critic, good baseline for fast environments.

```rust
let config = A2CConfig::default()
    .learning_rate(7e-4)
    .n_steps(5);
let agent = A2CAgent::new(config, vec_env, device)?;
```

### Off-Policy Algorithms

#### SAC (Soft Actor-Critic)

Maximum entropy RL for continuous control. Excellent sample efficiency.

```rust
let config = SACConfig::default()
    .learning_rate(3e-4)
    .auto_entropy_tuning(true);
let agent = SACAgent::new(config, vec_env, device)?;
```

#### TD3 (Twin Delayed DDPG)

Improved DDPG with twin critics and delayed policy updates.

```rust
let config = TD3Config::default()
    .learning_rate(3e-4)
    .policy_delay(2);
let agent = TD3Agent::new(config, vec_env, device)?;
```

#### DDPG (Deep Deterministic Policy Gradient)

Classic off-policy algorithm for continuous control.

```rust
let config = DDPGConfig::default()
    .actor_lr(1e-4)
    .critic_lr(1e-3)
    .noise_type(NoiseType::OrnsteinUhlenbeck);
let agent = DDPGAgent::new(config, vec_env, device)?;
```

#### DQN (Deep Q-Network) with Double DQN

For discrete action spaces with experience replay.

```rust
let config = DQNConfig::default()
    .double_dqn(true)
    .prioritized_replay(true);
let agent = DQNAgent::new(config, vec_env, device)?;
```

### Algorithm Comparison

| Algorithm | Action Space | Sample Efficiency | Stability | Use Case |
|-----------|-------------|-------------------|-----------|----------|
| **PPO** | Both | Medium | High | General purpose |
| **A2C** | Both | Low | Medium | Fast envs |
| **SAC** | Continuous | High | High | Robotics, trading |
| **TD3** | Continuous | High | High | Continuous control |
| **DDPG** | Continuous | Medium | Medium | Baseline |
| **DQN** | Discrete | High | High | Games, discrete tasks |

---

## Advanced Usage

### Custom Network Architectures

```rust
use rocket_rs::networks::{ActorCriticConfig, Activation, RecurrentType};

// Custom architecture with GRU
let config = ActorCriticConfig::continuous(obs_dim, act_dim)
    .with_hidden_dims(vec![512, 512, 256])
    .with_activation(Activation::GELU)
    .with_gru(256)
    .with_separate_networks(true);  // Separate actor/critic backbones
```

### Observation Normalization

```rust
// Coming soon: running mean/std normalization
let vec_env = env.make_vectorized(128)
    .with_obs_normalization(true)
    .with_reward_normalization(true);
```

### Learning Rate Scheduling

```rust
let config = PPOConfig::default()
    .learning_rate(3e-4)
    .use_lr_schedule(true);  // Linear decay to 0
```

---

## Comparison with Other Libraries

| Feature | RocketRL | Stable-Baselines3 | RLlib | CleanRL |
|---------|----------|-------------------|-------|---------|
| Language | Rust | Python | Python | Python |
| Performance | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Memory Safety | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| Metal Support | ✅ Native | ❌ | ❌ | ❌ |
| CUDA Support | ✅ | ✅ | ✅ | ✅ |
| VecEnv | ✅ Native | ✅ | ✅ | ✅ |
| Algorithms | 6 (PPO, A2C, SAC, TD3, DDPG, DQN) | 10+ | 20+ | 10+ |
| Replay Buffer | ✅ PER | ✅ | ✅ | ✅ |
| Training TUI | ✅ | ❌ | ❌ | ❌ |
| Documentation | Good | Excellent | Good | Good |

---

## Roadmap

- [x] **v0.2**: Off-policy algorithms (SAC, TD3, DDPG, DQN) ✅
- [x] **v0.2**: Prioritized Experience Replay ✅
- [x] **v0.2**: Training logging and TUI monitoring ✅
- [ ] **v0.3**: Multi-GPU distributed training
- [ ] **v0.4**: Model-based RL (Dreamer, World Models)
- [ ] **v0.5**: Offline RL (CQL, IQL, Decision Transformer)
- [ ] **v1.0**: Stable API, comprehensive docs, Python bindings

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

```bash
# Run tests
cargo test

# Run benchmarks
cargo bench

# Format code
cargo fmt

# Check lints
cargo clippy
```

---

## Citation

If you use RocketRL in your research, please cite:

```bibtex
@software{rocketrl2025,
  title = {RocketRL: High-Performance Reinforcement Learning in Rust},
  author = {RocketRL Contributors},
  year = {2025},
  url = {https://github.com/rocketrl/rocket-rs}
}
```

---

## License

Rocket-RS is licensed under the GNU General Public License v2.0.

See [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with Rust for maximum performance**

[Documentation](https://docs.rs/rocket-rs) · [Examples](examples/) · [Benchmarks](benchmarks/)

</div>
