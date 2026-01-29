# Octane

<div align="center">

**High-Performance Reinforcement Learning**

*Blazingly fast RL library written in Rust*

[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-GPL--2.0-blue.svg)](LICENSE)
[![Apple Silicon](https://img.shields.io/badge/Apple%20Silicon-optimized-black.svg)](https://developer.apple.com/metal/)

</div>

---

## Features

- **1000x faster than Python SB3** - Pure Rust eliminates GIL and dynamic typing overhead
- **ARM NEON SIMD optimizations** - Vectorized operations for Apple Silicon
- **Metal GPU support** - Native acceleration for M1/M2/M3/M4 chips
- **Parallel environments (VecEnv)** - Run thousands of simulations concurrently with Rayon
- **Complete algorithm suite** - PPO, SAC, TD3, DQN, A2C implementations

---

## Benchmark Results

All benchmarks performed on Apple M4 Max comparing Octane vs Python Stable-Baselines3.

| Steps | Octane | SB3 (Python) | Speedup |
|-------|--------|--------------|---------|
| 500K | ~0.6s | ~600s | **1000x** |
| 5M | ~5.6s | ~6000s | **1071x** |

### Throughput

| Metric | Octane | SB3 (Python) |
|--------|--------|--------------|
| FPS | 800,000 - 1,800,000 | ~833 |

---

## Quick Start

Add Octane to your project:

```bash
cargo add octane-rs
```

Basic usage example:

```rust
use octane_rs::prelude::*;
use octane_rs::envs::TradingEnv;
use octane_rs::algorithms::{PPOConfig, PPOAgent, RLAlgorithm};

fn main() -> octane_rs::Result<()> {
    // Select device (CPU, Metal, or CUDA)
    let device = Device::cpu();

    // Create and vectorize environment
    let env = TradingEnv::default();
    let vec_env = env.make_vectorized(128);

    // Configure PPO algorithm
    let config = PPOConfig::default()
        .learning_rate(3e-4)
        .n_steps(2048)
        .batch_size(64)
        .gamma(0.99);

    // Create agent and train
    let mut agent = PPOAgent::new(config, vec_env, device)?;

    agent.train(1_000_000, |metrics| {
        println!("Step {} | Reward: {:.2}", metrics.timesteps, metrics.mean_reward);
    })?;

    Ok(())
}
```

---

## Installation

### Feature Flags

```toml
[dependencies]
# Default (CPU only)
octane-rs = "0.1"

# Apple Silicon GPU (Metal)
octane-rs = { version = "0.1", features = ["metal"] }

# ARM NEON SIMD optimizations
octane-rs = { version = "0.1", features = ["simd"] }

# NVIDIA GPU (CUDA)
octane-rs = { version = "0.1", features = ["cuda"] }

# Full (all features)
octane-rs = { version = "0.1", features = ["full"] }
```

### Build from Source

```bash
git clone https://github.com/octane-rs/octane
cd octane

# CPU only
cargo build --release

# Apple Silicon with Metal
cargo build --release --features metal

# NVIDIA GPU with CUDA
cargo build --release --features cuda
```

---

## Architecture

```
octane/
├── src/
│   ├── core/           # Device abstraction (CPU/Metal/CUDA), error types
│   ├── envs/           # Gym-like Environment trait, VecEnv, TradingEnv
│   ├── networks/       # MLP, LSTM, GRU, ActorCritic architectures
│   ├── distributions/  # Categorical, DiagGaussian, SquashedGaussian
│   ├── buffer/         # RolloutBuffer (on-policy), ReplayBuffer (off-policy)
│   ├── algorithms/     # PPO, A2C, SAC, TD3, DDPG, DQN
│   ├── logging/        # Training metrics and TUI monitoring
│   └── tui/            # Terminal UI for visualization
```

### Key Components

| Module | Description |
|--------|-------------|
| `core` | Device management, error handling, tensor backend over Candle |
| `envs` | Environment trait with `reset()` / `step()`, vectorized environments |
| `networks` | Neural network architectures with configurable layers |
| `algorithms` | RL algorithms implementing `RLAlgorithm` trait |
| `buffer` | Experience storage with GAE and optional PER |

---

## Performance

### Environment Step Comparison

<!-- Performance chart placeholder -->
![Environment Performance](benchmarks/charts/env_comparison.png)

### Throughput Scaling

<!-- Throughput chart placeholder -->
![Throughput Scaling](benchmarks/charts/throughput.png)

### Speedup Summary

<!-- Speedup summary chart placeholder -->
![Speedup Summary](benchmarks/charts/speedup_summary.png)

---

## Algorithms

| Algorithm | Type | Action Space | Best For |
|-----------|------|--------------|----------|
| **PPO** | On-policy | Discrete/Continuous | General purpose |
| **A2C** | On-policy | Discrete/Continuous | Fast environments |
| **SAC** | Off-policy | Continuous | Sample efficiency |
| **TD3** | Off-policy | Continuous | Continuous control |
| **DQN** | Off-policy | Discrete | Games, discrete tasks |

---

## License

Octane is licensed under the [GNU General Public License v2.0](LICENSE).

---

<div align="center">

**Built with Rust for maximum performance**

</div>
