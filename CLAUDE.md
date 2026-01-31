# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Octane is a high-performance reinforcement learning library written in Rust, using Candle (HuggingFace's tensor library) as the backend. It's optimized for Apple Silicon (Metal) and NVIDIA GPUs (CUDA), achieving up to 1000x speedup over Python Stable-Baselines3.

## Build Commands

```bash
# Build (CPU only)
cargo build --release

# Build with Apple Silicon Metal + SIMD
cargo build --release --features metal,simd

# Build with NVIDIA CUDA support
cargo build --release --features cuda

# Build with x86_64 SIMD (AVX2/AVX-512)
cargo build --release --features avx2
cargo build --release --features avx512

# Run tests
cargo test

# Run a specific test
cargo test test_name

# Run benchmarks
cargo bench

# Run specific benchmark
cargo bench env_benchmark
cargo bench ppo_benchmark
cargo bench gpu_benchmark

# Linting and formatting
cargo fmt
cargo clippy -- -D warnings
```

## Architecture

### Module Structure

- **`core/`** - Device abstraction (CPU/Metal/CUDA), error types (`OctaneError`), precision (`Precision`, `GradScaler`), and tensor utilities
- **`envs/`** - Gym-like environment interface with `Environment` trait, `VecEnv` for parallel simulation, wrappers (`FrameStack`, `NormalizeObservation`, etc.), and multi-agent support
- **`networks/`** - Neural network architectures:
  - `MLP`, `LSTM`, `GRU`, `ActorCritic`
  - `TransformerEncoder`, `DecisionTransformer`, `AttentionActorCritic`
  - Normalization: `LayerNorm`, `RMSNorm`, `BatchNorm`
  - Weight init: orthogonal, xavier, kaiming
- **`distributions/`** - Action distributions: `Categorical` (discrete), `DiagGaussian` and `SquashedGaussian` (continuous)
- **`buffer/`** - Experience storage:
  - `RolloutBuffer` - For on-policy algorithms (PPO, A2C, PPG) with GAE
  - `ReplayBuffer` - For off-policy algorithms with optional PER via `SegmentTree`
  - `HERBuffer` - Hindsight Experience Replay (goal-conditioned)
  - `NStepBuffer` - N-step returns
  - `MmapReplayBuffer` - Memory-mapped for 100M+ transitions
- **`algorithms/`** - RL algorithms with `RLAlgorithm` trait:
  - On-policy: `PPOAgent`, `A2CAgent`, `PPGAgent`
  - Off-policy continuous: `SACAgent`, `TD3Agent`, `DDPGAgent`, `REDQAgent`, `CQLAgent`
  - Off-policy discrete: `DQNAgent`, `IQNAgent`
- **`simd/`** - Cross-platform SIMD optimizations:
  - ARM NEON (Apple Silicon): GAE, Gaussian sampling, softmax
  - AVX2/AVX-512 (x86_64): GAE, TD-error, log-prob
- **`distributed/`** - Multi-worker training with gradient aggregation and sync modes
- **`checkpoint/`** - Atomic saves, best model tracking, training resumption
- **`logging/`** - TensorBoard (pure Rust), Weights & Biases integration
- **`profiling/`** - Hierarchical timing with `ProfileScope` RAII guards
- **`tuning/`** - Hyperparameter optimization: `RandomSearch`, `GridSearch`, `Study`
- **`tui/`** - Terminal UI using ratatui for training visualization

### Trading-Specific Modules

- **`trading/`** - Advanced trading environments:
  - `env.rs` - Order book simulation, slippage models (Linear, SquareRoot, Almgren-Chriss), latency, partial fills, commission models
  - `multi_asset.rs` - Portfolio of N assets, correlations, rebalancing, cross-asset limits
  - `multi_timeframe.rs` - M1/M5/M15/H1/D1/W1 support, timeframe synchronization
  - `regime.rs` - HMM-based regime detection, GARCH volatility, 6 market regimes
- **`risk/`** - Risk management:
  - `constraints.rs` - Hard constraints, action masking, Lagrangian relaxation
  - `rewards.rs` - Sharpe, Sortino, Calmar, Risk Parity reward shaping
  - `position_sizing.rs` - Kelly criterion, fractional Kelly, ATR-based, anti-martingale
  - `drawdown.rs` - Real-time tracking, max drawdown limits, recovery mode, dynamic scaling
- **`metrics/`** - Trading analytics:
  - `trading.rs` - Sharpe, Sortino, Calmar, VaR, CVaR, Win Rate, Profit Factor, streaming updates
  - `journal.rs` - Trade logging, feature attribution, JSON/CSV export
  - `attribution.rs` - P&L breakdown by time, asset, regime, direction
- **`backtesting/`** - Validation infrastructure:
  - `walk_forward.rs` - Rolling/anchored WFO, overfitting detection
  - `monte_carlo.rs` - Bootstrap, GBM/Heston/Jump-diffusion, stress tests
  - `cross_validation.rs` - Purged K-Fold, embargo periods, combinatorial CV
- **`live/`** - Live trading (requires `distributed` feature):
  - `paper.rs` - Paper trading engine with slippage/fill simulation
  - `exchanges/` - Binance, Bybit connectors (REST + WebSocket)
  - `execution.rs` - TWAP, VWAP, Iceberg execution algorithms
  - `monitor.rs` - Real-time P&L, risk metrics, alerts
- **`strategies/`** - Advanced RL strategies:
  - `ensemble.rs` - Multi-agent voting (Majority, Stacking, Boosting), diversity metrics
  - `hierarchical.rs` - Two-level RL (timing + execution), trading options
  - `meta.rs` - MAML-style adaptation, regime-aware policies
  - `imitation.rs` - Behavioral Cloning, DAgger, demo replay

### Key Traits

- **`Environment`** (`src/envs/traits.rs`) - Core environment interface requiring `reset()` and `step()` methods. Returns `StepResult` with observation, reward, terminated/truncated flags.

- **`RLAlgorithm`** (`src/algorithms/traits.rs`) - Common RL algorithm interface with `train_step()`, `save()`, `load()`.

- **`Policy`/`ValueFunction`/`ActorCritic`** (`src/algorithms/traits.rs`) - Traits for neural network inference and evaluation.

- **`Distribution`** (`src/distributions/mod.rs`) - Action distribution interface with `sample()`, `log_prob()`, `entropy()`, `mode()`.

- **`Space`** (`src/envs/space.rs`) - Defines observation/action spaces (`BoxSpace`, `DiscreteSpace`).

### Parallelization

`VecEnv` uses Rayon for parallel environment stepping. Environments must implement `Clone` + `Send` + `Sync` to be vectorized.

### Feature Flags

- `cpu` (default) - CPU-only build
- `metal` - Apple Silicon GPU via Metal Performance Shaders
- `cuda` - NVIDIA GPU via CUDA
- `simd` - ARM NEON optimizations (Apple Silicon)
- `avx2` - x86_64 AVX2 optimizations
- `avx512` - x86_64 AVX-512 optimizations (implies avx2)
- `gym` - Python Gymnasium compatibility via PyO3
- `wandb` - Weights & Biases integration via PyO3
- `distributed` - Multi-worker training with Tokio/gRPC
- `half` - FP16/BF16 mixed precision support
- `full` - metal + cuda + simd + avx2 + half

## Code Style

- Uses `thiserror` for error types with `OctaneError` as the main error enum
- Uses builder pattern for configs (e.g., `PPOConfig::default().learning_rate(3e-4).n_steps(2048)`)
- All public APIs require documentation comments (`///`)
- `#![forbid(unsafe_code)]` enforced at crate level (except SIMD module)
- Tensor operations use Candle types (`candle_core::Tensor`)
- `Result<T>` alias for `std::result::Result<T, OctaneError>`

## Binary

`octane-tui` - Terminal UI binary for training visualization (`src/bin/octane_tui.rs`)
