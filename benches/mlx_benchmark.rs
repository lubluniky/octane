//! Experimental MLX side-path micro-benchmark.
//!
//! Compares a fused `matmul + tanh` forward with **both-direction marshaling**
//! (host -> framework -> host) in `mlx-rs` versus Candle, across a range of
//! batch sizes. The marshaling is the point: a Candle-resident environment loop
//! must cross the framework boundary every step, so this measures the real
//! side-path cost, not just the matmul. See `docs/MLX.md`.
//!
//! Run with:
//!   cargo bench --bench mlx_benchmark --features mlx          # candle on CPU
//!   cargo bench --bench mlx_benchmark --features "mlx,metal"  # candle on Metal

fn main() {
    #[cfg(not(feature = "mlx"))]
    {
        eprintln!(
            "mlx_benchmark requires --features mlx (optionally with metal). See docs/MLX.md."
        );
    }
    #[cfg(feature = "mlx")]
    run();
}

#[cfg(feature = "mlx")]
fn run() {
    use candle_core::Tensor;
    use octane_rs::mlx_experiment::fused_mlp_roundtrip;
    use std::time::Instant;

    let cols = 64usize;
    let out = 64usize;
    let iters = 200u32;

    #[cfg(feature = "metal")]
    let (dev, dev_name) = (
        candle_core::Device::new_metal(0).expect("metal device"),
        "metal",
    );
    #[cfg(not(feature = "metal"))]
    let (dev, dev_name) = (candle_core::Device::Cpu, "cpu");

    println!(
        "MLX side-path micro-bench (candle backend: {dev_name}); per-call us, lower is better"
    );
    println!(
        "{:>9} | {:>16} | {:>18}",
        "num_envs", "mlx roundtrip", "candle roundtrip"
    );

    for &n in &[8usize, 256, 1024, 4096] {
        let input: Vec<f32> = (0..n * cols).map(|i| (i as f32 * 0.001).sin()).collect();
        let weight: Vec<f32> = (0..cols * out)
            .map(|i| (i as f32 * 0.002).cos() * 0.01)
            .collect();

        // Warm up + time the MLX round-trip (host -> MLX -> host).
        let _ = fused_mlp_roundtrip(&input, n as i32, cols as i32, &weight, out as i32);
        let t = Instant::now();
        for _ in 0..iters {
            let _ = fused_mlp_roundtrip(&input, n as i32, cols as i32, &weight, out as i32);
        }
        let mlx_us = t.elapsed().as_micros() as f64 / iters as f64;

        // Candle round-trip with the same both-direction marshaling.
        let x = Tensor::from_slice(&input, &[n, cols], &dev).unwrap();
        let w = Tensor::from_slice(&weight, &[cols, out], &dev).unwrap();
        let _ = x.matmul(&w).unwrap().tanh().unwrap();
        let t = Instant::now();
        for _ in 0..iters {
            let y = x.matmul(&w).unwrap().tanh().unwrap();
            let _v: Vec<f32> = y.flatten_all().unwrap().to_vec1().unwrap();
        }
        let candle_us = t.elapsed().as_micros() as f64 / iters as f64;

        println!("{n:>9} | {mlx_us:>13.2} us | {candle_us:>15.2} us");
    }
}
