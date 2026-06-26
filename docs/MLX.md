# MLX optimization analysis (Apple Silicon)

## TL;DR

Octane's tensor layer is **Candle**. MLX is a *competing* array framework, not a
Candle backend you can bolt on — there is no `candle-mlx`. So "MLX optimization"
has three honest framings:

| Option | What it is | Verdict |
|---|---|---|
| (a) Optimize the existing Apple path | Faster Metal kernels / `MetalContext` / NEON, no new deps | Real, low-risk — pursued in the perf pass |
| (b) `mlx-rs` side-path | Run a few hot loops in MLX *parallel* to Candle, paying f32 marshaling crossing the frameworks | Experimental; **mostly does not pay** (see below) |
| (c) Full backend migration | Replace Candle with MLX everywhere | Weeks of work; out of scope for one pass |

This document records the side-path (b) analysis and a runnable micro-benchmark
so the conclusion is grounded in numbers, not hand-waving. The honest finding is
that **most of the apparent MLX win is recoverable for free inside Candle**, and
the part that isn't requires owning the parameters (full migration), not a
marshaled side-path.

## Where a marshaled MLX side-path could plausibly help — and why it mostly doesn't

The hot regions, ranked, with the marshaling reality:

1. **Rollout policy/value forward** (`ppo.rs::policy_forward`, called per step in
   `collect_rollout`). Weights are static across a rollout, so MLX could marshal
   weights once and run forward-only passes, fusing `linear+tanh` into fewer
   dispatches. **Two caveats gut the win:**
   - Much of the apparent speedup is just "stop rebuilding the `Linear` module
     via `format!` every call" — a **free Candle fix** (build modules once).
     Do that first or the comparison is dishonest.
   - The environment runs CPU-side, so observations/actions cross the boundary
     **every step**; the forward can't stay resident. Payoff only appears at
     large `num_envs × hidden`.

2. **PPO/SAC update loop** — the only compute-dense region, but **not** a viable
   marshaled side-path: to compute gradients MLX must *own* the parameters and
   optimizer state. As a side-path you would round-trip all weights+grads every
   minibatch (PPO ≈ `10 × n_batches`/update; SAC every step), dwarfing the matmul
   savings on `[256,256]` nets. MLX pays off in *training* only via a full
   backend migration that owns params + optimizer.

3. **Discrete categorical sampling** (`ppo.rs` per-step CPU inverse-CDF loop) —
   **recoverable in Candle**, skip MLX. The repo already has an on-tensor
   `Categorical::sample_gumbel_max`; route discrete sampling through it.

4. **Gaussian log-prob / PPO loss reduction as marshaled MLX kernels** — the
   anti-pattern. `simd/metal.rs` already shows the cost: per-call buffer alloc +
   `commit()` + `wait_until_completed()` + `Vec<f32>` round-trip, and **no
   autodiff** (returns `Vec<f32>`, can't participate in `backward()`). Keep these
   inside the Candle autodiff graph.

5. **Standalone MLX Gaussian sampler** — `randn + axpy` is memory-bound;
   marshaling exceeds the arithmetic. Not worth it.

## The free Candle recoveries (do these instead of/ before MLX)

These capture most of the realistically-achievable Apple-Silicon speedup with
**zero** new dependencies and full autodiff:

- **Build network modules once** at agent construction instead of rebuilding
  `Linear`/`Sequential` from the `VarMap` (with `format!` path strings + HashMap
  lookups + device re-derivation) on *every* forward. (Review P1 #2.)
- **Hoist per-step GPU→CPU syncs** out of the discrete sampler and grad-norm
  computation. (Done in the perf pass — `log_probs` is materialized once.)
- **Cache the candle `Device`** on each agent rather than calling `to_candle()`
  per forward (which re-derives `new_metal(0)` under the `metal` feature).

## Running the experiment

A feature-gated micro-benchmark lives behind the optional `mlx` Cargo feature so
the default build never touches MLX. On an Apple-Silicon machine with the MLX
toolchain available:

```bash
cargo bench --bench mlx_benchmark --features "metal,mlx"
```

It compares a fused `matmul + tanh` forward in `mlx-rs` against Candle across
`num_envs ∈ {8, 256, 1024, 4096}`, **including both-direction marshaling**, so
the reported numbers reflect the real side-path cost, not just the matmul.

### Measured results (per-call µs, lower is better)

Run on this machine with `--features mlx` (Candle on **CPU**, `[64→64]` layer):

| num_envs | mlx-rs round-trip | candle round-trip | winner |
|---:|---:|---:|---|
| 8 | 239 µs | **2.2 µs** | candle ×109 |
| 256 | 237 µs | **93 µs** | candle ×2.5 |
| 1024 | 243 µs | 233 µs | ~parity |
| 4096 | **335 µs** | 642 µs | mlx ×1.9 |

This **empirically confirms the side-path is the wrong shape for the inner
loop**: `mlx-rs` carries a roughly fixed ~240 µs marshal+dispatch+eval cost per
call, so it only wins once the batch is large enough (≈4096) to amortize it.
RL rollouts step the (CPU-side) environment at small `num_envs`, exactly where
the round trip dominates — so a marshaled MLX side-path *loses* in the regime
that matters. The conclusion holds even before comparing against candle-*metal*
(which would narrow Candle's large-batch gap further).

If `mlx-rs` cannot be built in your environment (it requires the MLX C++/Metal
toolchain), the benchmark and feature are simply not compiled; the analysis above
still stands and the free Candle recoveries are where the guaranteed wins are.

## Recommendation

1. Land the **free Candle recoveries** (build-once networks, cached device,
   sync hoisting) — guaranteed, autodiff-preserving, no new deps.
2. Treat `mlx-rs` as an **experiment**, gated behind the `mlx` feature, measured
   by the micro-benchmark before committing any inference path to it.
3. Only consider a **full MLX backend migration** if profiling shows the
   training matmuls dominate after (1) — and budget it as a multi-week project,
   since it also has to re-own the optimizer state (which would, as a side
   effect, be a different way to fix the SAC/TD3/… optimizer lifecycle).
