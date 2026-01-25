# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2025-01-25

### Added

#### New Algorithms
- **SAC (Soft Actor-Critic)** - Maximum entropy RL for continuous control
  - Twin Q-networks for reduced overestimation
  - Automatic entropy coefficient tuning
  - Reparameterization trick for policy gradient
- **TD3 (Twin Delayed DDPG)** - Improved DDPG with:
  - Twin critics for reduced overestimation
  - Delayed policy updates
  - Target policy smoothing with clipped noise
- **DDPG (Deep Deterministic Policy Gradient)** - Classic off-policy continuous control
  - Ornstein-Uhlenbeck noise for exploration
  - Gaussian noise option
  - Soft target updates
- **DQN (Deep Q-Network)** - For discrete action spaces
  - Double DQN for reduced overestimation
  - Prioritized Experience Replay (PER)
  - Huber loss for stability
  - Epsilon-greedy exploration with decay

#### Experience Buffers
- **ReplayBuffer** - Efficient ring buffer for off-policy algorithms
  - O(1) insertion and sampling
  - Prioritized Experience Replay with sum tree
  - Importance sampling weights with beta annealing
  - Configurable capacity and batch sizes

#### Training Infrastructure
- **TrainingLogger** - JSON-lines based logging system
  - Structured metrics output for background processes
  - Run metadata (algorithm, config, timestamps)
  - Steps per second computation
- **TrainingLogReader** - Real-time log reading for TUI
  - Incremental file reading
  - Progress tracking
  - Run discovery and listing

#### TUI Improvements
- **Theme System** - Professional dark color scheme
  - Consistent color palette (orange accent, cyan secondary)
  - Semantic colors for success/warning/error
  - Chart-specific colors for different metrics
- **Improved Usability**
  - Better chart rendering
  - Log file reading from background training
  - Status indicators for training state

#### API Improvements
- Expanded prelude module with all algorithm types
- Consistent builder pattern for all configs
- Better error messages and validation

### Changed
- Updated lib.rs exports to include all new algorithms
- Improved algorithm module organization
- Better documentation for all public types

### Performance
- Efficient sum tree for PER (O(log n) operations)
- Memory-efficient SoA layout for ReplayBuffer
- Optimized tensor operations in all algorithms

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
