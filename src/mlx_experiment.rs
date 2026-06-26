//! Experimental MLX side-path, gated behind the `mlx` feature.
//!
//! This is intentionally tiny: the review's conclusion (see `docs/MLX.md`) is
//! that a marshaled MLX side-path mostly does not pay versus the free Candle
//! recoveries, and that a real training win needs a full backend migration that
//! owns the parameters and optimizer. This module exists so the integration
//! point is real and the micro-benchmark (`benches/mlx_benchmark.rs`) has
//! something to call when the `mlx` feature and the MLX toolchain are present.
#![cfg(feature = "mlx")]

use mlx_rs::Array;

/// Marshal a host buffer into MLX, run a fused matmul + tanh, and read it back.
///
/// This deliberately includes both-direction marshaling so a benchmark measures
/// the *side-path* cost (host -> MLX -> host), not just the matmul: that round
/// trip is exactly what makes a Candle-resident environment loop unattractive
/// for an MLX side-path.
pub fn fused_mlp_roundtrip(
    input: &[f32],
    rows: i32,
    cols: i32,
    weight: &[f32],
    out_cols: i32,
) -> Vec<f32> {
    let x = Array::from_slice(input, &[rows, cols]);
    let w = Array::from_slice(weight, &[cols, out_cols]);
    let y = x.matmul(&w).expect("matmul");
    let y = mlx_rs::ops::tanh(&y).expect("tanh");
    y.eval().expect("eval");
    y.as_slice::<f32>().to_vec()
}
