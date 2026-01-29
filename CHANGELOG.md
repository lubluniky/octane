# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/lubluniky/rocket-rs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/lubluniky/rocket-rs/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/lubluniky/rocket-rs/releases/tag/v0.1.0
