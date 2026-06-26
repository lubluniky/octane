//! Environment benchmarks for Octane
//!
//! Run with: cargo bench --bench env_benchmark

use candle_core::Tensor;
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use octane_rs::core::Device;
use octane_rs::envs::{Environment, MarketData, TradingEnv};
use std::hint::black_box;

fn benchmark_single_env_step(c: &mut Criterion) {
    let device = Device::cpu();
    let data = MarketData::synthetic(10000, 42);
    let mut env = TradingEnv::new(data).unwrap();
    let _ = env.reset(&device).unwrap();

    let candle_device = device.to_candle().unwrap();
    let action = Tensor::zeros(&[1], candle_core::DType::F32, &candle_device).unwrap();

    c.bench_function("single_env_step", |b| {
        b.iter(|| {
            let result = env.step(black_box(&action), &device).unwrap();
            if result.done() {
                let _ = env.reset(&device);
            }
            black_box(result)
        })
    });
}

fn benchmark_vecenv_step(c: &mut Criterion) {
    let device = Device::cpu();

    let mut group = c.benchmark_group("vecenv_step");

    for num_envs in [1, 8, 32, 128, 512, 1024].iter() {
        let data = MarketData::synthetic(10000, 42);
        let env = TradingEnv::new(data).unwrap();
        let mut vec_env = env.make_vectorized(*num_envs);
        let _ = vec_env.reset(&device).unwrap();

        let candle_device = device.to_candle().unwrap();
        let actions =
            Tensor::zeros(&[*num_envs, 1], candle_core::DType::F32, &candle_device).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(num_envs), num_envs, |b, _| {
            b.iter(|| black_box(vec_env.step(&actions, &device).unwrap()))
        });
    }

    group.finish();
}

#[cfg(feature = "distributed")]
fn benchmark_vecenv_step_async(c: &mut Criterion) {
    let device = Device::cpu();

    let mut group = c.benchmark_group("vecenv_step_async");

    for num_envs in [1, 8, 32, 128, 512, 1024].iter() {
        let data = MarketData::synthetic(10000, 42);
        let template_env = TradingEnv::new(data).unwrap();
        let mut vec_env = VecEnv::new_async(vec![template_env], *num_envs);
        let _ = vec_env.reset(&device).unwrap();

        let candle_device = device.to_candle().unwrap();
        let actions =
            Tensor::zeros(&[*num_envs, 1], candle_core::DType::F32, &candle_device).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(num_envs), num_envs, |b, _| {
            b.iter(|| black_box(vec_env.step(&actions, &device).unwrap()))
        });
    }

    group.finish();
}

fn benchmark_env_reset(c: &mut Criterion) {
    let device = Device::cpu();
    let data = MarketData::synthetic(10000, 42);
    let mut env = TradingEnv::new(data).unwrap();

    c.bench_function("env_reset", |b| {
        b.iter(|| black_box(env.reset(&device).unwrap()))
    });
}

fn benchmark_tensor_ops(c: &mut Criterion) {
    let device = Device::cpu();
    let candle_device = device.to_candle().unwrap();

    let mut group = c.benchmark_group("tensor_ops");

    for size in [64, 256, 1024].iter() {
        let a = Tensor::randn(0.0f32, 1.0, &[*size, *size], &candle_device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0, &[*size, *size], &candle_device).unwrap();

        group.bench_with_input(BenchmarkId::new("matmul", size), size, |bench, _| {
            bench.iter(|| black_box(a.matmul(&b).unwrap()))
        });
    }

    for batch_size in [32, 128, 512].iter() {
        let logits = Tensor::randn(0.0f32, 1.0, &[*batch_size, 64], &candle_device).unwrap();

        group.bench_with_input(
            BenchmarkId::new("softmax", batch_size),
            batch_size,
            |bench, _| bench.iter(|| black_box(candle_nn::ops::softmax(&logits, 1).unwrap())),
        );
    }

    group.finish();
}

#[cfg(not(feature = "distributed"))]
criterion_group!(
    benches,
    benchmark_single_env_step,
    benchmark_vecenv_step,
    benchmark_env_reset,
    benchmark_tensor_ops,
);

#[cfg(feature = "distributed")]
criterion_group!(
    benches,
    benchmark_single_env_step,
    benchmark_vecenv_step,
    benchmark_vecenv_step_async,
    benchmark_env_reset,
    benchmark_tensor_ops,
);

criterion_main!(benches);
