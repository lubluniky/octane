//! CPU vs Metal GPU Benchmarks for Octane
//!
//! Run with: cargo bench --bench gpu_benchmark --features metal

use candle_core::{DType, Tensor};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

fn get_devices() -> Vec<(&'static str, candle_core::Device)> {
    let devices = vec![("CPU", candle_core::Device::Cpu)];

    #[cfg(feature = "metal")]
    {
        if let Ok(metal) = candle_core::Device::new_metal(0) {
            devices.push(("Metal", metal));
        }
    }

    devices
}

fn benchmark_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("matmul_comparison");
    group.sample_size(50);

    let devices = get_devices();

    for size in [128, 512, 1024, 2048].iter() {
        for (name, device) in &devices {
            let a = Tensor::randn(0.0f32, 1.0, &[*size, *size], device).unwrap();
            let b = Tensor::randn(0.0f32, 1.0, &[*size, *size], device).unwrap();

            // Warmup
            let _ = a.matmul(&b).unwrap();

            group.bench_with_input(BenchmarkId::new(*name, size), size, |bench, _| {
                bench.iter(|| {
                    let result = a.matmul(&b).unwrap();
                    black_box(result)
                })
            });
        }
    }

    group.finish();
}

fn benchmark_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("softmax_comparison");
    group.sample_size(50);

    let devices = get_devices();

    for (batch, features) in [(64, 512), (256, 1024), (512, 2048), (1024, 4096)].iter() {
        for (name, device) in &devices {
            let logits = Tensor::randn(0.0f32, 1.0, &[*batch, *features], device).unwrap();

            // Warmup
            let _ = candle_nn::ops::softmax(&logits, 1).unwrap();

            group.bench_with_input(
                BenchmarkId::new(*name, format!("{batch}x{features}")),
                &(*batch, *features),
                |bench, _| {
                    bench.iter(|| {
                        let result = candle_nn::ops::softmax(&logits, 1).unwrap();
                        black_box(result)
                    })
                },
            );
        }
    }

    group.finish();
}

fn benchmark_mlp_forward(c: &mut Criterion) {
    let mut group = c.benchmark_group("mlp_forward_comparison");
    group.sample_size(50);

    let devices = get_devices();

    // Simulate MLP forward pass: input -> hidden -> output
    let batch_size = 256;
    let input_dim = 128;
    let hidden_dim = 512;
    let output_dim = 64;

    for (name, device) in &devices {
        let input = Tensor::randn(0.0f32, 1.0, &[batch_size, input_dim], device).unwrap();
        let w1 = Tensor::randn(0.0f32, 0.1, &[input_dim, hidden_dim], device).unwrap();
        let b1 = Tensor::zeros(&[hidden_dim], DType::F32, device).unwrap();
        let w2 = Tensor::randn(0.0f32, 0.1, &[hidden_dim, output_dim], device).unwrap();
        let b2 = Tensor::zeros(&[output_dim], DType::F32, device).unwrap();

        // Warmup
        let h = input.matmul(&w1).unwrap().broadcast_add(&b1).unwrap();
        let h = h.relu().unwrap();
        let _ = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();

        group.bench_function(BenchmarkId::new(*name, "256x128->512->64"), |bench| {
            bench.iter(|| {
                let h = input.matmul(&w1).unwrap().broadcast_add(&b1).unwrap();
                let h = h.relu().unwrap();
                let output = h.matmul(&w2).unwrap().broadcast_add(&b2).unwrap();
                black_box(output)
            })
        });
    }

    group.finish();
}

fn benchmark_gae_tensors(c: &mut Criterion) {
    let mut group = c.benchmark_group("gae_tensor_ops");
    group.sample_size(50);

    let devices = get_devices();
    let n_steps = 2048;
    let n_envs = 16;

    for (name, device) in &devices {
        let rewards = Tensor::randn(0.0f32, 1.0, &[n_steps, n_envs], device).unwrap();
        let values = Tensor::randn(0.0f32, 1.0, &[n_steps, n_envs], device).unwrap();
        let dones = Tensor::zeros(&[n_steps, n_envs], DType::F32, device).unwrap();
        let gamma = 0.99f32;

        // Warmup
        let next_values = values.narrow(0, 1, n_steps - 1).unwrap();
        let current_values = values.narrow(0, 0, n_steps - 1).unwrap();
        let current_rewards = rewards.narrow(0, 0, n_steps - 1).unwrap();
        let current_dones = dones.narrow(0, 0, n_steps - 1).unwrap();
        let not_dones = (1.0 - &current_dones).unwrap();
        let _ = ((&current_rewards
            + &((&next_values * gamma as f64).unwrap() * &not_dones).unwrap())
            .unwrap()
            - &current_values)
            .unwrap();

        group.bench_function(
            BenchmarkId::new(*name, format!("{n_steps}x{n_envs}")),
            |bench| {
                bench.iter(|| {
                    let next_values = values.narrow(0, 1, n_steps - 1).unwrap();
                    let current_values = values.narrow(0, 0, n_steps - 1).unwrap();
                    let current_rewards = rewards.narrow(0, 0, n_steps - 1).unwrap();
                    let current_dones = dones.narrow(0, 0, n_steps - 1).unwrap();

                    let not_dones = (1.0 - &current_dones).unwrap();
                    let td_target = (&current_rewards
                        + &((&next_values * gamma as f64).unwrap() * &not_dones).unwrap())
                        .unwrap();
                    let advantages = (&td_target - &current_values).unwrap();

                    black_box(advantages)
                })
            },
        );
    }

    group.finish();
}

fn benchmark_large_batch_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_batch_inference");
    group.sample_size(30);

    let devices = get_devices();

    // Simulate policy network inference with large batch
    for batch_size in [512, 1024, 2048, 4096].iter() {
        let obs_dim = 256;
        let hidden1 = 512;
        let hidden2 = 256;
        let action_dim = 32;

        for (name, device) in &devices {
            let obs = Tensor::randn(0.0f32, 1.0, &[*batch_size, obs_dim], device).unwrap();
            let w1 = Tensor::randn(0.0f32, 0.05, &[obs_dim, hidden1], device).unwrap();
            let w2 = Tensor::randn(0.0f32, 0.05, &[hidden1, hidden2], device).unwrap();
            let w3 = Tensor::randn(0.0f32, 0.05, &[hidden2, action_dim], device).unwrap();

            // Warmup
            let h1 = obs.matmul(&w1).unwrap().relu().unwrap();
            let h2 = h1.matmul(&w2).unwrap().relu().unwrap();
            let _ = h2.matmul(&w3).unwrap();

            group.bench_with_input(
                BenchmarkId::new(*name, batch_size),
                batch_size,
                |bench, _| {
                    bench.iter(|| {
                        let h1 = obs.matmul(&w1).unwrap().relu().unwrap();
                        let h2 = h1.matmul(&w2).unwrap().relu().unwrap();
                        let output = h2.matmul(&w3).unwrap();
                        black_box(output)
                    })
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_matmul,
    benchmark_softmax,
    benchmark_mlp_forward,
    benchmark_gae_tensors,
    benchmark_large_batch_inference,
);

criterion_main!(benches);
