# Rocket-RS Quick Start Guide

## Installation

### From crates.io (when published)

```toml
[dependencies]
rocket-rs = "0.1"
```

### From source

```bash
git clone https://github.com/lubluniky/rocket-rs.git
cd rocket-rs
cargo build --release
```

## Your First RL Agent (5 minutes)

```rust
use rocket_rs::prelude::*;
use rocket_rs::envs::{TradingEnv, MarketData};
use rocket_rs::algorithms::{PPOConfig, PPOAgent, RLAlgorithm};
use rocket_rs::core::Device;

fn main() -> rocket_rs::Result<()> {
    // 1. Create your environment
    let data = MarketData::random(1000);
    let env = TradingEnv::new(data)?;
    let vec_env = VecEnv::new(vec![env], 8); // 8 parallel environments

    // 2. Configure PPO algorithm
    let config = PPOConfig {
        learning_rate: 3e-4,
        n_steps: 2048,
        batch_size: 64,
        ..PPOConfig::default()
    };

    // 3. Create agent (automatically selects Metal on Apple Silicon)
    let device = Device::new_default()?;
    let mut agent = PPOAgent::new(config, vec_env, device)?;

    // 4. Train!
    agent.train(100_000, |metrics| {
        println!("Step {}: reward = {:.2}", 
                 metrics.timesteps, 
                 metrics.mean_reward);
    })?;

    // 5. Save trained model
    agent.save("trading_agent.safetensors")?;

    Ok(())
}
```

## Run the Example

```bash
# CPU only
cargo run --example trading_ppo --release

# Apple Silicon with Metal
cargo run --example trading_ppo --release --features metal

# NVIDIA GPU with CUDA
cargo run --example trading_ppo --release --features cuda
```

## Monitor Training with TUI

```bash
# Launch the terminal UI
cargo run --bin rocket-tui --release

# Or in benchmark mode
cargo run --bin rocket-tui --release -- --benchmark
```

## Next Steps

- 📖 Read the [full documentation](README.md)
- 🎯 Check out [examples/](examples/)
- 🚀 Run [benchmarks](benchmarks/)
- 💬 Join discussions on [GitHub](https://github.com/lubluniky/rocket-rs/discussions)

## Common Use Cases

### Algorithmic Trading
```rust
let env = TradingEnv::new(market_data)?;
// Train with PPO for policy-based trading
```

### Custom Environments
```rust
impl Environment for MyEnv {
    type ObsSpace = Box<[f32]>;
    type ActSpace = i64;
    
    fn step(&mut self, action: &Tensor, device: &Device) 
        -> Result<(Tensor, Tensor, Tensor, Vec<bool>)> {
        // Your environment logic
    }
}
```

### GPU Acceleration
```rust
// Automatically uses Metal on M1-M4 Macs
let device = Device::new_default()?;

// Or explicitly choose
let device = Device::Metal(0);
let device = Device::Cuda(0);
```

## Performance Tips

1. **Vectorize**: Use `VecEnv` with 8-32 parallel environments
2. **GPU**: Enable Metal/CUDA for 5-10x speedup on large networks
3. **Batch Size**: Increase `batch_size` if you have GPU memory
4. **Release Mode**: Always use `--release` for training

## Getting Help

- 🐛 [Report bugs](https://github.com/lubluniky/rocket-rs/issues/new?template=bug_report.md)
- 💡 [Request features](https://github.com/lubluniky/rocket-rs/issues/new?template=feature_request.md)
- 📚 [Read contributing guide](CONTRIBUTING.md)

---

**Happy training! 🚀**
