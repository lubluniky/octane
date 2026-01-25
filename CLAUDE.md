# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RocketRL is a high-performance reinforcement learning library written in Rust, using Candle (HuggingFace's tensor library) as the backend. It's optimized for Apple Silicon (Metal) and NVIDIA GPUs (CUDA), targeting algorithmic trading and other RL applications.

## Build Commands

```bash
# Build (CPU only)
cargo build --release

# Build with Apple Silicon Metal support
cargo build --release --features metal

# Build with NVIDIA CUDA support
cargo build --release --features cuda

# Run tests
cargo test

# Run a specific test
cargo test test_name

# Run benchmarks
cargo bench

# Run specific benchmark
cargo bench env_benchmark

# Linting and formatting
cargo fmt
cargo clippy -- -D warnings
```

## Architecture

### Module Structure

- **`core/`** - Device abstraction (CPU/Metal/CUDA), error types (`RocketError`), and tensor backend trait over Candle
- **`envs/`** - Gym-like environment interface with `Environment` trait, `VecEnv` for parallel simulation via Rayon, and `TradingEnv` example
- **`networks/`** - Neural network architectures: `MLP`, `LSTM`, `GRU`, and combined `ActorCritic` policy-value networks
- **`distributions/`** - Action distributions: `Categorical` (discrete), `DiagGaussian` and `SquashedGaussian` (continuous)
- **`buffer/`** - Experience storage:
  - `RolloutBuffer` - For on-policy algorithms (PPO, A2C) with GAE
  - `ReplayBuffer` - For off-policy algorithms (SAC, TD3, DDPG, DQN) with optional PER
- **`algorithms/`** - RL algorithms with `RLAlgorithm` trait:
  - On-policy: `PPOAgent`, `A2CAgent`
  - Off-policy: `SACAgent`, `TD3Agent`, `DDPGAgent`, `DQNAgent`
- **`logging/`** - Training logging system:
  - `TrainingLogger` - Writes JSON-lines metrics for background training
  - `TrainingLogReader` - Reads logs for TUI monitoring
- **`tui/`** - Terminal UI using ratatui:
  - `theme.rs` - Professional dark color scheme
  - `screens.rs` - Dashboard, Training, Benchmark, About screens

### Key Traits

- **`Environment`** (`src/envs/traits.rs`) - Core environment interface requiring `reset()` and `step()` methods. Returns `StepResult` with observation, reward, terminated/truncated flags.

- **`RLAlgorithm`** (`src/algorithms/traits.rs`) - Common RL algorithm interface with `train_step()`, `save()`, `load()`.

- **`Policy`/`ValueFunction`/`ActorCritic`** (`src/algorithms/traits.rs`) - Traits for neural network inference and evaluation.

- **`Space`** (`src/envs/space.rs`) - Defines observation/action spaces (`BoxSpace`, `DiscreteSpace`).

### Parallelization

`VecEnv` uses Rayon for parallel environment stepping. Environments must implement `Clone` + `Send` + `Sync` to be vectorized.

### Feature Flags

- `cpu` (default) - CPU-only build
- `metal` - Apple Silicon GPU via Metal Performance Shaders
- `cuda` - NVIDIA GPU via CUDA
- `full` - Both metal and cuda

## Code Style

- Uses `thiserror` for error types with `RocketError` as the main error enum
- Uses builder pattern for configs (e.g., `PPOConfig::default().learning_rate(3e-4).n_steps(2048)`)
- All public APIs require documentation comments (`///`)
- `#![forbid(unsafe_code)]` enforced at crate level
- Tensor operations use Candle types (`candle_core::Tensor`)

## Binary

`rocket-tui` - Terminal UI binary for training visualization (`src/bin/rocket_tui.rs`)
