# Regression / Leak / Correctness Audit

Adversarial multi-agent audit of octane-rs: 8 subsystem finders → per-finding
adversarial verification (refute-by-default) → dedup + synthesis. 29 raw
findings → 16 confirmed → **9 deduplicated issues**. No P0 survived adversarial
correction (the four original P0s were downgraded but their defects affirmed).

All 9 are **fixed** in this pass. Full suite green (375 tests), clippy clean,
`--features simd/python/gym/wandb/distributed` all compile.

| # | Sev | Area | Issue | Fix |
|---|-----|------|-------|-----|
| 1 | P1 | `envs/trading.rs` | `TradingEnv` derived `Clone` copied the `StdRng` state, so every `VecEnv` replica drew the same episode-window start and traversed identical market data — silently collapsing vectorization diversity. The lone env still using `derive(Clone)`. | Manual `Clone` reseeding `rng` from entropy, mirroring `ArrayEnv`/`CartPole`/`Pendulum`. |
| 2 | P2 | `live/execution.rs` | VWAP `params.duration / profile.len()` panics (Duration ÷ 0) when an explicit empty `volume_profile` is supplied. | Fall back to the uniform default when the profile is empty. |
| 3 | P2 | `live/execution.rs` | Passive execution submitted a `0.0` limit price when none was given (rests unfilled / rejected). | Fetch the current mid price via `get_mid_price` when `limit_price` is `None`. |
| 4 | P2 | `strategies/{meta,hierarchical,imitation}.rs` | `AdamW` recreated inside the train loop every step → Adam's moment estimates reset each step (≈ sign-SGD), degrading convergence. Systemic across BC, the hierarchical high-level update, and the meta outer loop. | Persistent optimizer struct fields, initialized once after `init_networks()`, reused. The MAML **inner**-loop fresh optimizer is left as-is (correct fast-weights). |
| 5 | P3 | `risk/drawdown.rs` | `calculate_risk_scale` divides by `max_drawdown` in the Linear/Exponential/Sigmoid branches; `max_drawdown = 0` at equity peak gives 0/0 → NaN that survives `clamp()`. | One shared early guard: `if max_drawdown <= 0.0 { return 1.0 }`. |
| 6 | P3 | `algorithms/{ppo,a2c}.rs` | Discrete inverse-CDF sampler defaulted `action = 0`; when `r` lands in the f32 softmax residual (probs sum < 1) the loop never breaks and action 0 absorbs the residual mass — a silent bias toward action 0. | Default to the **last** index (standard inverse-CDF convention). Both copy-pasted sites. |
| 7 | P3 | `simd/log_prob.rs` | Squashed-Gaussian SIMD log-prob: the scalar paths computed `gaussian_lp + log(1−tanh²)` while the AVX2/NEON vector bodies computed the SAC-correct `gaussian_lp − log(1−tanh²)`. The scalar sign was wrong (and a unit test enshrined the wrong direction `squashed < gaussian`). Exported-only, currently unused. | Aligned all 5 scalar sites with the vector bodies; corrected the test to assert `squashed > gaussian`. |
| 8 | P3 | `buffer/nstep.rs` | On in-window termination the n-step buffer stored a **pre-terminal state** as `next_obs` (inconsistent with `flush_episode_end`). `done` masks it in the target, but a latent trap for consumers reading `next_obs` unconditionally. | Always bootstrap from the provided `next_obs`; `done = encountered_done \|\| final_done`. |
| 9 | P3 | `metrics/trading.rs` | `downside_deviation` lacked the `.max(0.0)` guard before `sqrt` that `return_std` already has — float cancellation could NaN-poison Sortino. | Added `.max(0.0)` before `sqrt`. |

## Method notes

- **Adversarial verification.** Every finding was re-checked by an independent
  agent prompted to *refute* it (default to refuted on weak evidence). The four
  original P0 ratings (TradingEnv RNG, drawdown NaN, optimizer recreation,
  discrete sampling) were all downgraded by the verifiers while affirming the
  defect — none were dropped, none stayed P0.
- **Dedup clusters.** Collapsed: two TradingEnv-Clone reports (one file:line);
  three drawdown div-by-zero branches (one guard); the identical PPO/A2C
  discrete sampler; and four optimizer-recreation sites into one systemic issue.
- **Not changed.** The MAML inner-loop optimizer (meta.rs `adapt_policy`) is
  intentionally recreated per task adaptation — correct fast-weights semantics,
  not the anti-pattern.
