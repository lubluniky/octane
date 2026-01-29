//! PPO Algorithm benchmarks
//!
//! Run with: cargo bench --bench ppo_benchmark

use candle_core::Tensor;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use octane_rs::core::Device;

fn benchmark_ppo_loss_computation(c: &mut Criterion) {
    let device = Device::cpu();
    let candle_device = device.to_candle().unwrap();

    let mut group = c.benchmark_group("ppo_loss");

    for batch_size in [64, 256, 1024].iter() {
        let log_probs_new = Tensor::randn(0.0f32, 1.0, &[*batch_size], &candle_device).unwrap();
        let log_probs_old = Tensor::randn(0.0f32, 1.0, &[*batch_size], &candle_device).unwrap();
        let advantages = Tensor::randn(0.0f32, 1.0, &[*batch_size], &candle_device).unwrap();
        let clip_range = 0.2f32;

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    let ratio = (&log_probs_new - &log_probs_old).unwrap().exp().unwrap();
                    let surr1 = (&ratio * &advantages).unwrap();
                    let ratio_clamped = ratio.clamp(1.0 - clip_range, 1.0 + clip_range).unwrap();
                    let surr2 = (&ratio_clamped * &advantages).unwrap();
                    let min_surr = surr1.minimum(&surr2).unwrap();
                    let policy_loss = min_surr.neg().unwrap().mean_all().unwrap();
                    black_box(policy_loss)
                })
            },
        );
    }

    group.finish();
}

fn benchmark_forward_pass(c: &mut Criterion) {
    let device = Device::cpu();
    let candle_device = device.to_candle().unwrap();

    let mut group = c.benchmark_group("forward_pass");

    for (batch_size, hidden_dim) in [(32, 64), (128, 256), (512, 512)].iter() {
        let input = Tensor::randn(0.0f32, 1.0, &[*batch_size, 162], &candle_device).unwrap();
        let w1 = Tensor::randn(0.0f32, 0.1, &[162, *hidden_dim], &candle_device).unwrap();
        let w2 = Tensor::randn(0.0f32, 0.1, &[*hidden_dim, *hidden_dim], &candle_device).unwrap();
        let w3 = Tensor::randn(0.0f32, 0.1, &[*hidden_dim, 1], &candle_device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("mlp", format!("{}x{}", batch_size, hidden_dim)),
            &(*batch_size, *hidden_dim),
            |b, _| {
                b.iter(|| {
                    let h1 = input.matmul(&w1).unwrap().tanh().unwrap();
                    let h2 = h1.matmul(&w2).unwrap().tanh().unwrap();
                    let out = h2.matmul(&w3).unwrap();
                    black_box(out)
                })
            },
        );
    }

    group.finish();
}

fn benchmark_advantage_normalization(c: &mut Criterion) {
    let device = Device::cpu();
    let candle_device = device.to_candle().unwrap();

    let mut group = c.benchmark_group("advantage_norm");

    for size in [1024, 4096, 16384].iter() {
        let advantages = Tensor::randn(0.0f64, 1.0, &[*size], &candle_device).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, _| {
            b.iter(|| {
                let mean = advantages.mean_all().unwrap();
                let mean_val: f64 = mean.to_scalar().unwrap();
                let centered = (&advantages - mean_val).unwrap();
                let var = centered.sqr().unwrap().mean_all().unwrap();
                let var_val: f64 = var.to_scalar().unwrap();
                let std = (var_val + 1e-8).sqrt();
                let normalized = (&centered / std).unwrap();
                black_box(normalized)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_ppo_loss_computation,
    benchmark_forward_pass,
    benchmark_advantage_normalization,
);

criterion_main!(benches);
