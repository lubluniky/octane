# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial release of Rocket-RS
- PPO (Proximal Policy Optimization) implementation
- A2C (Advantage Actor-Critic) implementation
- Vectorized environment support
- GPU acceleration via Metal (Apple Silicon) and CUDA (NVIDIA)
- Neural network architectures: MLP, LSTM, GRU
- Trading environment for algorithmic trading
- Terminal UI (TUI) for monitoring training
- Comprehensive benchmarks

### Features
- **Zero-Cost Abstractions** - Rust ownership model eliminates runtime overhead
- **Vectorized Environments** - Run 1000s of parallel simulations on CPU
- **GPU Acceleration** - Native Metal (M1-M4) and CUDA support
- **Production-Ready** - Memory-safe, thread-safe, no garbage collection
- **Complete Algorithms** - PPO and A2C with GAE
- **Time-Series Ready** - LSTM/GRU networks for sequential decisions

### Performance
- 12.5x faster environment steps vs Python Gymnasium
- 4.8x faster environment resets
- 5.9x faster vectorized environments (1024 parallel)
- ~3x less memory usage

## [0.1.0] - 2025-01-24

### Added
- Initial public release
- Core reinforcement learning infrastructure
- Full documentation and examples
- CI/CD pipeline with GitHub Actions
- Comprehensive test suite
- Performance benchmarks

[Unreleased]: https://github.com/lubluniky/rocket-rs/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/lubluniky/rocket-rs/releases/tag/v0.1.0
