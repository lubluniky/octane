# Contributing to RocketRL

Thank you for your interest in contributing to RocketRL! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Benchmarking](#benchmarking)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/rocket-rs.git
   cd rocket-rs
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/rocketrl/rocket-rs.git
   ```

## Development Setup

### Prerequisites

- Rust 1.75+ (1.80+ recommended)
- For GPU development:
  - **macOS**: Xcode Command Line Tools for Metal support
  - **Linux**: CUDA 11.8+ for NVIDIA GPU support

### Build

```bash
# CPU only
cargo build

# With Metal (Apple Silicon)
cargo build --features metal

# With CUDA (NVIDIA)
cargo build --features cuda

# Release build
cargo build --release
```

### Run Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_name

# Run with output
cargo test -- --nocapture
```

### Run Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench env_benchmark
```

## How to Contribute

### Types of Contributions

- **Bug Fixes**: Fix issues and improve stability
- **New Features**: Add new algorithms, environments, or utilities
- **Documentation**: Improve README, API docs, or examples
- **Performance**: Optimize existing code or add benchmarks
- **Tests**: Increase test coverage

### Workflow

1. **Create a branch** for your changes:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes** following our [coding standards](#coding-standards)

3. **Test your changes**:
   ```bash
   cargo test
   cargo clippy -- -D warnings
   cargo fmt --check
   ```

4. **Commit your changes**:
   ```bash
   git add .
   git commit -m "feat: add new feature" # or "fix: resolve bug"
   ```

5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request** on GitHub

## Coding Standards

### Rust Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)
- Use `cargo fmt` to format code (this is enforced in CI)
- Use `cargo clippy` to catch common mistakes
- Run `cargo clippy -- -D warnings` to ensure no warnings

### Documentation

- All public APIs must have documentation comments (`///`)
- Include examples in doc comments where appropriate
- Use `cargo doc --open` to preview documentation

Example:
```rust
/// Computes the advantage estimates using GAE (Generalized Advantage Estimation).
///
/// # Arguments
///
/// * `rewards` - Tensor of shape [batch_size] containing rewards
/// * `values` - Tensor of shape [batch_size] containing value estimates
/// * `gamma` - Discount factor (typically 0.99)
/// * `lambda` - GAE lambda parameter (typically 0.95)
///
/// # Returns
///
/// Tensor of shape [batch_size] containing advantage estimates
///
/// # Example
///
/// ```
/// use rocket_rs::algorithms::gae;
/// let advantages = gae(&rewards, &values, 0.99, 0.95)?;
/// ```
pub fn gae(
    rewards: &Tensor,
    values: &Tensor,
    gamma: f64,
    lambda: f64,
) -> Result<Tensor> {
    // implementation
}
```

### Error Handling

- Use `Result<T, RocketError>` for fallible operations
- Use `thiserror` for custom error types
- Provide meaningful error messages

### Code Organization

- Keep functions focused and small
- Use modules to organize related functionality
- Prefer composition over inheritance
- Use traits for polymorphism

## Testing

### Unit Tests

- Place tests in the same file as the code, in a `tests` module
- Test edge cases and error conditions
- Use descriptive test names

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_observation_space_creation() {
        let space = ObservationSpace::new(vec![4]);
        assert_eq!(space.shape(), &[4]);
    }

    #[test]
    fn test_invalid_action_returns_error() {
        let mut env = MyEnv::new();
        let result = env.step(&Tensor::new(&[-999.0], &Device::Cpu).unwrap());
        assert!(result.is_err());
    }
}
```

### Integration Tests

- Place integration tests in `tests/` directory
- Test realistic workflows
- Ensure GPU features work correctly

### Performance Tests

- Add benchmarks for performance-critical code
- Use `criterion` for benchmarking
- Document performance expectations

## Benchmarking

When adding new features that affect performance:

1. Add benchmarks in `benches/` directory
2. Compare before and after performance
3. Include benchmark results in PR description

```bash
# Run benchmarks and save baseline
cargo bench --bench env_benchmark -- --save-baseline main

# Make changes, then compare
cargo bench --bench env_benchmark -- --baseline main
```

## Pull Request Process

### Before Submitting

- [ ] Code builds without errors: `cargo build`
- [ ] All tests pass: `cargo test`
- [ ] No clippy warnings: `cargo clippy -- -D warnings`
- [ ] Code is formatted: `cargo fmt`
- [ ] Documentation is updated
- [ ] CHANGELOG.md is updated (if applicable)

### PR Guidelines

1. **Title**: Use conventional commits format
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation
   - `perf:` for performance improvements
   - `refactor:` for code refactoring
   - `test:` for adding tests
   - `chore:` for maintenance tasks

2. **Description**: 
   - Explain what changes you made and why
   - Reference related issues (e.g., "Closes #123")
   - Include benchmark results for performance changes
   - Add screenshots/examples if applicable

3. **Size**: Keep PRs focused and reasonably sized
   - Large changes should be discussed in an issue first
   - Consider breaking large changes into smaller PRs

### Review Process

- Maintainers will review your PR
- Address feedback and requested changes
- Once approved, a maintainer will merge your PR

## Issue Guidelines

### Bug Reports

When reporting bugs, include:

- **Description**: Clear description of the bug
- **Steps to Reproduce**: Minimal code to reproduce the issue
- **Expected Behavior**: What you expected to happen
- **Actual Behavior**: What actually happened
- **Environment**: 
  - OS and version
  - Rust version (`rustc --version`)
  - RocketRL version
  - GPU info (if relevant)
- **Logs/Errors**: Full error messages or stack traces

### Feature Requests

When requesting features:

- **Use Case**: Describe the problem you're trying to solve
- **Proposed Solution**: Outline your suggested approach
- **Alternatives**: Other solutions you've considered
- **Additional Context**: Any other relevant information

### Questions

- Check existing issues and documentation first
- Use GitHub Discussions for general questions
- Use Issues for specific problems or bugs

## GPU Development

### Metal (Apple Silicon)

- Test on real hardware when possible
- Ensure fallback to CPU works
- Use `#[cfg(feature = "metal")]` for Metal-specific code

### CUDA (NVIDIA)

- Test with different CUDA versions if possible
- Handle CUDA initialization errors gracefully
- Use `#[cfg(feature = "cuda")]` for CUDA-specific code

## Performance Considerations

- Profile before optimizing
- Prefer readability unless performance is critical
- Document performance trade-offs
- Add benchmarks for performance-critical paths

## Documentation

### API Documentation

```bash
# Generate and view documentation
cargo doc --open --no-deps
```

### Examples

- Add examples to `examples/` directory
- Examples should be self-contained and runnable
- Include comments explaining key concepts

## Getting Help

- **Documentation**: Check the [README](README.md) and API docs
- **GitHub Discussions**: For questions and ideas
- **Issues**: For bugs and feature requests

## Recognition

Contributors will be recognized in:
- The repository's contributor graph
- Release notes for significant contributions
- The project README (for major contributions)

Thank you for contributing to RocketRL! 🚀