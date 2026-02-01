# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2025-02-01

### Added

#### Performance Optimizations
- **RolloutBuffer Flat Storage** - Replaced `Vec<Tensor>` with flat `Vec<f32>` storage for all buffer fields (`observations`, `actions`, `rewards`, `dones`, `values`, `log_probs`, `advantages`, `returns`). Eliminates per-step tensor allocations and improves cache locality.
- **VecEnv Persistent Worker Pool** - Implemented `EnvWorker` struct with dedicated threads and `crossbeam::channel` communication. Removes `Arc<Mutex>` wrapping from hot path, enabling true parallel execution.
- **SIMD GAE Computation** - Added `simd/gae.rs` with vectorized Generalized Advantage Estimation:
  - ARM NEON: Processes 4 environments per iteration using 128-bit registers with FMA (`vfmaq_f32`)
  - x86_64 AVX2: Processes 8 environments per iteration using 256-bit registers with FMA (`_mm256_fmadd_ps`)
  - Inverted loop order (time-outer, env-inner) for optimal cache access patterns
- **Correct Truncation Handling** - Separated `terminated` and `truncated` signals in `RolloutBuffer` and `VecEnv`. Fixed GAE calculation to bootstrap value estimates on truncation instead of treating as terminal (zero value). Prevents value function collapse at episode time limits.

### Changed
- **RolloutBuffer API** - `add()` method now accepts separate `terminated` and `truncated` tensors instead of combined `dones`
- **Algorithms Updated** - PPO, A2C, PPG now use corrected truncation handling
- **Buffer Module** - `buffer/mod.rs` updated with separate `terminated_flat`/`truncated_flat` storage

### Performance Improvements
- VecEnv worker pool: **Removed per-step locking overhead**
- RolloutBuffer: **Zero tensor allocations during rollout collection**
- GAE computation: **~4x speedup (NEON)**, **~8x speedup (AVX2)** for high env counts
- Metal GPU MatMul 128x128: **30x speedup** (157µs → 5.2µs)
- Policy inference batch 512: **7.9x speedup** (1.03ms → 130µs)

### Fixed
- Value function collapse bug when episodes truncate at time limit
- Incorrect GAE values near episode boundaries
- Memory allocation overhead in `step_async()` hot path

### Files Added
- `src/simd/gae.rs` - SIMD-optimized GAE computation with NEON/AVX2 support

### Files Modified
- `src/algorithms/rollout.rs` - Flat storage, SIMD GAE integration
- `src/algorithms/ppo.rs` - Separated terminated/truncated handling
- `src/algorithms/a2c.rs` - Separated terminated/truncated handling
- `src/algorithms/ppg.rs` - Separated terminated/truncated handling
- `src/buffer/mod.rs` - Separated terminated/truncated storage
- `src/envs/vecenv.rs` - Persistent worker pool implementation
- `src/simd/mod.rs` - GAE module export

## [0.3.0] - 2025-01-30

### Added

#### New Algorithms
- **PPG (Phasic Policy Gradient)** - Decoupled policy and value function training phases
- **REDQ (Randomized Ensemble Double Q-Learning)** - 10 Q-network ensemble with UTD=20
- **CQL (Conservative Q-Learning)** - Offline RL with conservative Q-value penalties
- **IQN (Implicit Quantile Networks)** - Distributional RL with risk-sensitive policies (CVaR, Wang, CPW)

#### Advanced Experience Replay
- **HER (Hindsight Experience Replay)** - Goal-conditioned learning with Final/Future/Episode/Random strategies
- **N-step Returns Buffer** - Configurable multi-step TD targets
- **Memory-Mapped Buffers** - Handle 100M+ transitions with minimal RAM usage
- **Segment Tree PER** - O(log n) prioritized sampling with SumTree/MinTree

#### Neural Network Architectures
- **TransformerEncoder** - Multi-head self-attention layers
- **DecisionTransformer** - Transformer architecture for offline RL
- **AttentionActorCritic** - Attention-based policy and value networks
- **LayerNorm/RMSNorm/BatchNorm** - Modern normalization layers
- **Weight Initialization** - Orthogonal, Xavier, Kaiming initializers

#### x86_64 SIMD Optimizations (~3,300 LOC)
- **AVX2/AVX-512 support** - Vectorized operations for Intel/AMD processors
- `x86.rs` - Gaussian sampling, softmax, gather/scatter, GAE computation
- `td_error.rs` - SIMD TD-error for SAC, TD3, DQN, PER weight updates
- `log_prob.rs` - Vectorized Gaussian/SquashedGaussian log-probability

#### Environment Features
- **Gym/Gymnasium Wrapper** - Python environment integration via PyO3
- **Multi-Agent Support** - CTDE (Centralized Training, Decentralized Execution)
- **Environment Wrappers** - FrameStack, TimeLimit, NormalizeObservation, NormalizeReward, ClipAction

#### Infrastructure
- **Distributed Training** - Multi-worker gradient aggregation with gRPC backend
- **Mixed Precision** - FP16/BF16 with GradScaler and AutocastContext
- **Checkpointing** - Atomic saves, best model tracking, TrainingResumer
- **Hyperparameter Tuning** - RandomSearch, GridSearch, Study with trial management

#### Observability
- **TensorBoard Export** - Pure Rust TFRecord writer (no Python dependency)
- **Weights & Biases** - Full W&B integration via PyO3
- **Profiling System** - Hierarchical timing with ProfileScope, SIMD stats

### Changed
- Codebase grew from ~20,000 to ~42,000 lines of code
- Expanded algorithm suite from 6 to 10 algorithms
- Updated README with comprehensive documentation
- Added `avx2`, `avx512`, `gym`, `wandb`, `distributed`, `half` feature flags

### New Feature Flags
```toml
avx2 = []           # AVX2 SIMD optimizations (x86_64)
avx512 = ["avx2"]   # AVX-512 SIMD optimizations (x86_64)
gym = ["dep:pyo3"]  # Python Gym/Gymnasium compatibility
wandb = ["dep:pyo3"] # Weights & Biases integration
distributed = ["dep:tokio", "dep:tonic", "dep:prost"]
grpc = ["distributed"]
half = ["dep:half"] # FP16/BF16 support
```

### Files Added (24 new files)
- `src/algorithms/ppg.rs` - Phasic Policy Gradient
- `src/algorithms/redq.rs` - Randomized Ensemble Double Q
- `src/algorithms/cql.rs` - Conservative Q-Learning
- `src/algorithms/iqn.rs` - Implicit Quantile Networks
- `src/buffer/her.rs` - Hindsight Experience Replay
- `src/buffer/nstep.rs` - N-step returns buffer
- `src/buffer/mmap.rs` - Memory-mapped buffer
- `src/buffer/segment_tree.rs` - SumTree/MinTree for PER
- `src/checkpoint.rs` - Checkpointing and training resumption
- `src/core/precision.rs` - Mixed precision training
- `src/envs/gym.rs` - Python Gym wrapper
- `src/envs/multiagent.rs` - Multi-agent environments
- `src/envs/wrappers.rs` - Environment wrappers
- `src/logging/metrics.rs` - MetricLogger trait
- `src/logging/tensorboard.rs` - TensorBoard writer
- `src/logging/wandb.rs` - W&B integration
- `src/networks/transformer.rs` - Transformer architectures
- `src/networks/attention.rs` - Multi-head attention
- `src/networks/normalization.rs` - Normalization layers
- `src/networks/init.rs` - Weight initialization
- `src/simd/x86.rs` - AVX2/AVX-512 operations
- `src/simd/td_error.rs` - SIMD TD-error computation
- `src/simd/log_prob.rs` - SIMD log probability
- `src/tuning.rs` - Hyperparameter optimization
- `src/distributed/mod.rs` - Distributed training
- `src/profiling/mod.rs` - Performance profiling

## [0.2.0] - 2025-01-29

### Added
- ARM NEON SIMD optimizations for Apple Silicon M4
  - `gae_neon.c` - Vectorized GAE computation (4x speedup)
  - `gaussian_neon.c` - Box-Muller sampling with xoroshiro128+ RNG
  - `categorical_neon.c` - Gumbel-max trick, SIMD softmax
  - `buffer_ops_neon.c` - Batch gather/scatter operations
- Metal compute shaders for GPU acceleration
  - Fused Gaussian log_prob kernel
  - PPO loss computation kernel
  - Categorical sampling on GPU
- Rust FFI bindings for all SIMD operations
- New `simd` feature flag for NEON optimizations
- Benchmark suite for performance testing

### Changed
- Project renamed from RocketRL to Octane
- Performance improved from 17K FPS to 800K-1.8M FPS
- Updated documentation with new benchmarks

### Performance
- Environment stepping: 1000x faster than Python SB3
- GAE computation: 4-5x faster with NEON
- Gaussian sampling: 5x faster with vectorized Box-Muller

## [0.1.0] - 2025-01-24

### Added
- Initial release
- PPO, A2C, SAC, TD3, DQN, DDPG algorithms
- VecEnv for parallel simulation
- Trading environment example
- TUI for training visualization
- Metal and CUDA support via Candle

[Unreleased]: https://github.com/lubluniky/octane-rs/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/lubluniky/octane-rs/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/lubluniky/octane-rs/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/lubluniky/octane-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lubluniky/octane-rs/releases/tag/v0.1.0
