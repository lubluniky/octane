//! Benchmark: Octane vs Stable Baselines 3
//!
//! Run with: cargo run --release --features simd --example benchmark_vs_sb3

use octane_rs::core::Device;
use octane_rs::envs::{Environment, MarketData, Space, TradingEnv};
use std::time::Instant;

#[cfg(feature = "simd")]
use octane_rs::simd::{compute_gae, softmax_batch, GaussianSampler};

fn benchmark_env_steps(total_steps: usize, num_envs: usize) -> (f64, f64) {
    let device = Device::cpu();
    let candle_device = device.to_candle().unwrap();

    // Create synthetic market data
    let data = MarketData::synthetic(10000, 42);

    // Create base environment
    let base_env = TradingEnv::new(data).unwrap();

    // Vectorize
    let mut vec_env = base_env.make_vectorized(num_envs);

    // Reset
    let _ = vec_env.reset(&device).unwrap();

    // Get action dimension
    let action_shape = vec_env.action_space().shape();
    let action_dim = action_shape.iter().product::<usize>().max(1);

    let start = Instant::now();
    let mut steps_done = 0;

    while steps_done < total_steps {
        // Random actions
        let actions =
            candle_core::Tensor::rand(-1.0f32, 1.0f32, &[num_envs, action_dim], &candle_device)
                .unwrap();

        let _ = vec_env.step(&actions, &device).unwrap();
        steps_done += num_envs;
    }

    let elapsed = start.elapsed().as_secs_f64();
    let fps = total_steps as f64 / elapsed;

    (elapsed, fps)
}

#[cfg(feature = "simd")]
fn benchmark_gae_simd(buffer_size: usize, num_envs: usize, iterations: usize) -> (f64, f64) {
    let total_size = buffer_size * num_envs;

    // Create test data
    let rewards: Vec<f32> = (0..total_size).map(|_| rand::random::<f32>()).collect();
    let values: Vec<f32> = (0..total_size).map(|_| rand::random::<f32>()).collect();
    let dones: Vec<f32> = (0..total_size)
        .map(|_| {
            if rand::random::<f32>() > 0.99 {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    let last_values: Vec<f32> = (0..num_envs).map(|_| rand::random::<f32>()).collect();

    let start = Instant::now();

    for _ in 0..iterations {
        let _ = compute_gae(
            &rewards,
            &values,
            &dones,
            buffer_size,
            num_envs,
            0.99,
            0.95,
            &last_values,
        )
        .unwrap();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let ops_per_sec = (iterations * buffer_size * num_envs) as f64 / elapsed;
    let ms_per_rollout = elapsed * 1000.0 / iterations as f64;

    (ops_per_sec, ms_per_rollout)
}

#[cfg(feature = "simd")]
fn benchmark_gaussian_sampling_simd(count: usize, iterations: usize) -> (f64, f64) {
    let mut sampler = GaussianSampler::new(42);
    let mean: Vec<f32> = vec![0.0; count];
    let std: Vec<f32> = vec![1.0; count];

    let start = Instant::now();

    for _ in 0..iterations {
        let _ = sampler.sample(&mean, &std).unwrap();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let samples_per_sec = (iterations * count) as f64 / elapsed;
    let ms_per_batch = elapsed * 1000.0 / iterations as f64;

    (samples_per_sec, ms_per_batch)
}

#[cfg(feature = "simd")]
fn benchmark_softmax_simd(batch_size: usize, num_classes: usize, iterations: usize) -> (f64, f64) {
    let logits: Vec<f32> = (0..batch_size * num_classes)
        .map(|_| rand::random::<f32>() * 2.0 - 1.0)
        .collect();

    let start = Instant::now();

    for _ in 0..iterations {
        let _ = softmax_batch(&logits, batch_size, num_classes).unwrap();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let ops_per_sec = (iterations * batch_size) as f64 / elapsed;
    let ms_per_batch = elapsed * 1000.0 / iterations as f64;

    (ops_per_sec, ms_per_batch)
}

fn main() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║        Octane Benchmark - Comparison with SB3                  ║");
    println!("║        Platform: Apple M4 (ARM64 + NEON SIMD)                    ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();

    let num_envs = 64;

    // =========================================================================
    // Benchmark 1: 500K steps
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  BENCHMARK: 500,000 environment steps ({} parallel envs)",
        num_envs
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let total_steps_500k = 500_000;

    print!("  Running Octane... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let (time_500k, fps_500k) = benchmark_env_steps(total_steps_500k, num_envs);

    println!("Done!");
    println!();
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │ Results: 500K steps                                         │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!(
        "  │ Octane:    {:>8.2}s    {:>12.0} FPS                  │",
        time_500k, fps_500k
    );
    println!("  │ SB3 (ref):   ~600.00s    ~833 FPS (Python)               │");
    println!(
        "  │ Speedup:     {:>8.1}x faster                              │",
        600.0 / time_500k
    );
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();

    // =========================================================================
    // Benchmark 2: 5M steps
    // =========================================================================
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!(
        "  BENCHMARK: 5,000,000 environment steps ({} parallel envs)",
        num_envs
    );
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!();

    let total_steps_5m = 5_000_000;

    print!("  Running Octane... ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let (time_5m, fps_5m) = benchmark_env_steps(total_steps_5m, num_envs);

    println!("Done!");
    println!();
    println!("  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │ Results: 5M steps                                           │");
    println!("  ├─────────────────────────────────────────────────────────────┤");
    println!(
        "  │ Octane:    {:>8.2}s    {:>12.0} FPS                  │",
        time_5m, fps_5m
    );
    println!("  │ SB3 (ref):   ~6000.0s    ~833 FPS (Python)               │");
    println!(
        "  │ Speedup:     {:>8.1}x faster                              │",
        6000.0 / time_5m
    );
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!();

    // =========================================================================
    // SIMD Benchmarks
    // =========================================================================
    #[cfg(feature = "simd")]
    {
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!("  SIMD COMPONENT BENCHMARKS (ARM NEON)");
        println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        println!();

        // GAE benchmark
        let (gae_ops, gae_ms) = benchmark_gae_simd(2048, 64, 1000);
        println!("  GAE Computation (2048 steps x 64 envs):");
        println!("    {:>12.0} ops/sec", gae_ops);
        println!("    {:>12.3} ms/rollout", gae_ms);
        println!();

        // Gaussian sampling benchmark
        let (gaussian_ops, gaussian_ms) = benchmark_gaussian_sampling_simd(2048 * 6, 10000);
        println!("  Gaussian Sampling (12,288 samples/batch):");
        println!("    {:>12.0} samples/sec", gaussian_ops);
        println!("    {:>12.3} ms/batch", gaussian_ms);
        println!();

        // Softmax benchmark
        let (softmax_ops, softmax_ms) = benchmark_softmax_simd(2048, 10, 10000);
        println!("  Softmax (2048 batch x 10 classes):");
        println!("    {:>12.0} rows/sec", softmax_ops);
        println!("    {:>12.3} ms/batch", softmax_ms);
        println!();
    }

    // =========================================================================
    // Summary
    // =========================================================================
    println!("╔══════════════════════════════════════════════════════════════════╗");
    println!("║                         FINAL SUMMARY                            ║");
    println!("╠══════════════════════════════════════════════════════════════════╣");
    println!("║                                                                  ║");
    println!(
        "║  500K steps:  {:>7.2}s (RocketRL) vs ~600s (SB3) = {:>5.0}x      ║",
        time_500k,
        600.0 / time_500k
    );
    println!(
        "║  5M steps:    {:>7.2}s (RocketRL) vs ~6000s (SB3) = {:>5.0}x     ║",
        time_5m,
        6000.0 / time_5m
    );
    println!("║                                                                  ║");
    println!(
        "║  Average Throughput: {:>10.0} FPS                            ║",
        (fps_500k + fps_5m) / 2.0
    );
    println!("║  SB3 Reference:      ~833 FPS                                    ║");
    println!("║                                                                  ║");
    println!("╚══════════════════════════════════════════════════════════════════╝");
    println!();
}
