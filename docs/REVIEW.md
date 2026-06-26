# Octane-rs Massive Review

Synthesis of 123 verified findings. Priority is by **impact on training correctness and throughput**, not by count. The single highest-ROI fix (optimizer recreation) silently degrades *nine* agents to sign-SGD and should be done first.

## Executive summary

- **Adam is broken in every agent except PPO.** SAC/TD3/DDPG/REDQ/DQN/IQN/CQL/A2C/PPG all reconstruct `AdamW` inside `update()`, zeroing moment buffers and the step counter every step → updates collapse to ~`lr·sign(g)`. One pattern fix (store optimizers as fields) repairs all of them. *(findings 13, 20, 35)*
- **Entropy-temperature auto-tuning has an inverted sign in SAC, REDQ, and CQL** — default-on, drives `alpha` away from the target-entropy fixed point (collapse/divergence). One-line sign flip per agent. *(14, 15, 22)*
- **On-policy agents hard-reset every rollout and never train `log_std`.** PPO/A2C/PPG reset all envs at the start of each rollout (A2C every 5 steps), so they can't learn horizons longer than `n_steps`; continuous A2C/PPG also use a fixed, non-trainable `std=1`. *(2, 33, 32)*
- **Truncation is treated as termination everywhere**, cutting the value bootstrap at time limits; the VecEnv layer also destroys the terminal observation, so even the rollout buffer's "truncation-aware" GAE can't get the right `s'`. Systematic negative value bias on fixed-horizon (incl. trading) episodes. *(17, 25, 68, 84, 75)*
- **PER and Mmap buffers are algorithmically O(n)/O(n²)** at default capacities (1e6), making warmup/sampling the dominant cost; a well-tested `MinTree` already exists but is left unused. *(47, 48, 49, 50)*
- **Trading reward double-counts P&L** (the headline use case): `portfolio_value()` adds lifetime `unrealized_pnl` on top of an already marked-to-market position, biasing the agent to hold winners forever. *(91, 92)*

---

## P0 — Correctness bugs (RL math / numerical)

Ordered by training impact. Items grouped by shared root cause.

1. **AdamW recreated every gradient step → moments/step-count reset → degenerates to sign-SGD.** `sac.rs@431-453`, `td3.rs@436-471`, `ddpg.rs@429-452`, `redq.rs@689-727` (×ensemble×utd = worst), `dqn.rs@322-328`, `iqn.rs@632-638`, `cql.rs@632-654` (3 optimizers), `a2c.rs@385-390`, `ppg.rs@719-725,906-912`. **Fix:** construct each optimizer once at agent `new()` and store as a struct field; reuse across `update()` so `m,v,t` persist. *(13, 20, 35)*

2. **Entropy temperature (alpha) auto-tune sign inverted** — pushes alpha away from target entropy; default-on. `sac.rs@456-471`, `redq.rs@730-745`, `cql.rs@656-667`. **Fix:** `alpha_grad = diff * alpha` (i.e. `log_alpha -= lr*(mean(-log_pi)-target_entropy)*alpha`); also clamp `log_alpha` to ~[-10,10] before `exp()`. Note: CQL's own Lagrange update 40 lines up uses the correct sign, so the file self-contradicts. *(14, 15, 22)*

3. **Truncation treated as termination; terminal obs discarded.** Root cause at VecEnv: `vecenv.rs@375,423,488,554` replace the terminal obs with the reset obs and `VecStepResult` has no terminal-obs field. Consumers: off-policy `sac.rs@415-418`, `td3.rs@415`, `ddpg.rs@417`, `redq.rs@682`, `dqn.rs@285,380-388`, `iqn.rs@583,695`. **Fix:** add `terminal_observations: Vec<Option<Tensor>>` to `VecStepResult`; propagate `terminated`/`truncated` separately into `ReplayBuffer::add`; mask bootstrap with `(1-terminated)` only and bootstrap `V(s_terminal)` on truncation. *(17, 25, 68; latent siblings 84 Metal, 75 simd)*

4. **On-policy hard-reset destroys episode continuity.** `ppo.rs@581`, `a2c.rs@442`, `ppg.rs@631` call `env.reset()` at the start of every rollout. **Fix:** store `last_obs` on the agent; reset once lazily on the first rollout, then carry obs across rollouts (rely on VecEnv auto-reset for per-env dones). A2C with `n_steps=5` currently can never exceed the 5-step horizon. *(2, 33)*

5. **Continuous `log_std` is a fixed, non-trainable constant.** `a2c.rs@157`, `ppg.rs@427` create `log_std` as a plain `Tensor::zeros` (not a `Var`, not in `VarMap`) → `std=1` forever, no gradient, not saved. **Fix:** register via `vb.get_with_hints(&[act_dim], "policy.log_std", init)` using `NetworkConfig.log_std_init`. Cripples all continuous A2C/PPG runs. *(32)*

6. **IQN quantile-Huber TD-error sign inverted → learned quantiles mirrored (τ↔1−τ).** `iqn.rs@620-622` use `td_errors = current - target`, complementing the pinball indicator; CVaR/risk selection becomes risk-*seeking* (breaks the headline risk-sensitive feature). **Fix:** `td_errors = target - current` (or flip the indicator); add a test that τ=0.9 recovers a high quantile. *(21)*

7. **CQL conservative penalty mixes Q across different states + unstable logsumexp.** `cql.rs@423-431,445-469`: `obs.repeat` block-tiles so each row's logsumexp aggregates Q from `num_random` *different* states (corrupt for batch>1); `cql.rs@462-469` logsumexp has no max-subtraction (f32 overflow for Q>~88). **Fix:** tile interleaved via `obs.unsqueeze(1).broadcast_as(...)` so each state repeats consecutively; subtract per-row max before `exp`. *(23, 24)*

8. **REDQ policy update uses MIN over the full ensemble instead of MEAN.** `redq.rs@715-724` (comment even says "take mean"). Over-pessimistic, biases the actor. **Fix:** mean the ensemble Q's before `policy_loss = (alpha*log_pi - mean_q).mean()`. (Random-subset min for the *target* at 679 is correct.) *(16)*

9. **PPO/PPG KL early-stop only breaks the minibatch loop, not the epoch loop** → after tripping `target_kl` the agent still runs all remaining epochs. `ppo.rs@536-542` (default `n_epochs=10`, so this is live), `ppg.rs@807-816`. **Fix:** `continue_training` flag (or labeled break) that exits both loops. *(1, 39)*

10. **A2C silently skips gradient clipping** despite the comment and `max_grad_norm=0.5`. `a2c.rs@410-411` only call `backward_step`. **Fix:** use the manual `backward()` → `clip_gradients` → `step_with_grads` path PPO already has (`ppo.rs@508-509`). PPG also ignores `max_grad_norm`. *(34)*

11. **Trading reward double-counts position P&L.** `trading/env.rs@797-801,1198-1217`: `portfolio_value()` adds full `unrealized_pnl` on top of the marked position while `portfolio_before` omits it → reward re-adds the entire gain-from-entry every step (`multi_timeframe.rs@511-517` is a bounded 2× of the true step return). `multi_asset.rs@97` is the correct reference (debits cash, never adds upnl). **Fix:** drop the `+ unrealized_pnl` term (position is already marked) or adopt the cash-debit model; compute `before`/`after` with the same function. *(91, 92)*

12. **`Categorical::log_prob` detaches autograd (host round-trip) → zero gradient.** `categorical.rs@188-218` gathers via `to_vec1` + `Tensor::from_slice`, breaking the graph. Public distribution API only — internal PPO/PPG/A2C reimplement `gather` and are unaffected. **Fix:** `self.log_probs.gather(&actions.to_dtype(I64)?.unsqueeze(1)?,1)?.squeeze(1)`. *(42)*

13. **Orthogonal init is silently ignored, and the impl is broken for dim-reducing layers.** `ortho_init=true` is never read (`actor_critic.rs@256-379`); every layer is Kaiming. And `init.rs@110-130` errors (numel mismatch) for any `out<in` weight (the common case) via a wrong `narrow`+`reshape`. **Fix:** drop the `narrow` in the `rows<cols` branch (`q.t()?.contiguous()` is already correct); then actually wire `orthogonal_init` into Linear weights with activation gains. Affects PPO reproducibility / SB3 parity. *(54, 55)*

14. **FrameStack produces wrong shape for multi-dim obs.** `wrappers.rs@47-53,85-95`: declares `[n_stack, …]` but `cat(dim 0)` yields `[n_stack*C, H, W]` (Atari `[4,84,84]` → `[336,84]`). **Fix:** use `Tensor::stack(&frames,0)` for `base_shape.len()>1`, keep `cat` for the 1D branch. *(67)*

15. **BatchNorm running stats are never updated → eval mode doesn't normalize.** `normalization.rs@322-323` (momentum update skipped), read at `339-364`. Train uses live batch stats; eval subtracts mean=0/divides by ~1 → distribution shift. **Fix:** update running stats via interior mutability in `forward_train`; until then steer users to LayerNorm/RMSNorm. *(62)*

16. **Epsilon-greedy uses one RNG draw for the whole vectorized batch** → all envs explore or exploit together, collapsing replay diversity. `dqn.rs@245-261`, `iqn.rs@443-468`. **Fix:** per-row Bernoulli(ε); compute batch argmax then overwrite the exploring rows. *(26)*

17. **DecisionTransformer double-encodes position** (learned timestep embeddings + unconditional sinusoidal PE over interleaved tokens, which conflict). `transformer.rs@748-758,512-518`. **Fix:** add `use_positional_encoding` flag and disable PE for the DT backbone. *(63)*

18. **Separate-backbone + recurrent silently drops recurrence** while `is_recurrent()` reports true (allocates unused LSTM/GRU params). `actor_critic.rs@269-289,537-557`. **Fix:** reject the combo at construction or actually use per-head recurrent state. *(56)*

19. **Metrics NaN poisoning: `sqrt` of a possibly-negative variance.** `metrics/trading.rs@349-361` (one-pass variance via subtraction can go slightly negative). One NaN propagates into Sharpe/Sortino/vol/VaR. **Fix:** `variance().max(0.0).sqrt()` (sibling `rewards.rs@117,126` already clamps). Related: `powf` on negative base for returns <−100% (`@434-441,407-409`). *(98, 102)*

20. **DQN Huber loss computes |u| via `sqrt(sqr())` → inf/NaN gradient at td_error=0.** `dqn.rs@310-315`. **Fix:** `(abs_error - 1.0).clamp(0.0, MAX)` as IQN already does (`iqn.rs@485`). *(31)*

21. **LATENT (public API, not in active training path): Metal kernels corrupt/OOB.** `metal.rs@gae_compute` over-dispatch aliases valid output columns when `num_envs%64≠0` (e.g. 96/100/200) → silent advantage corruption; all element-wise/argmax kernels lack `id>=n` bounds guards (OOB writes); `normalize_obs` hardcodes `%1024`. `MetalContext` is re-exported but never called internally. **Fix:** add `if (id>=n) return;` to every kernel (or `dispatch_threads`), pass `obs_dim`; do this **before** wiring MetalContext into any path. *(80, 81, 85, 87)*

---

## P1 — Performance bottlenecks (ranked by expected impact)

1. **PER buffer is O(n)/O(n²); Mmap LRU is O(cache_size) per access.** `replay.rs@213-223` rescans all priorities on every `add` (filling 1e6 = O(n²)); `replay.rs@291,472-479` scans all leaves for min-priority every `sample`; `mmap.rs@381-405` uses `Vec::retain`+`insert(0)` (cache_size=1e5) per touched transition. **Fix:** track running `max_priority` scalar; back min with the existing `segment_tree::MinTree`; replace Vec-LRU with an O(1) LRU. **Bench:** new buffer benches — fill 1M prioritized buffer (add); `sample(256)` at capacity 1e6; mmap `sample(256)`. *(47, 48, 49, 50)*

2. **Networks rebuilt from VarMap on every forward (`format!` allocs + HashMap lookups + device re-derive).** Pervasive: `ppo.rs@207-266,347-392` (also `to_candle()` inside the minibatch loop @466), `sac.rs@203-321`, `td3.rs@200-269`, `ddpg.rs@229-296`, `redq.rs@446-570` (×ensemble×utd ≈ 200 rebuilds/step), `dqn.rs@169-203`, `iqn.rs@292-377`, `cql.rs@354-390`. **Fix:** build `Linear`/`Sequential` modules once at `new()`, cache the candle `Device` on the struct, precompute `log_2pi`. **Bench:** `gpu_benchmark`/`ppo_benchmark` `update()` throughput, esp. REDQ default `ensemble=10, utd=20`. *(5, 7, 18, 28)*

3. **Per-step GPU→CPU syncs in rollout/update hot loops.** PPO discrete sampler does a `to_vec1` *per env* inside the batch loop (`ppo.rs@305`; same `a2c.rs@305`, `ppg.rs`); off-policy copies 5 tensors/step via `flatten_all().to_vec1()` (`sac.rs@510-514` + td3/ddpg/redq); `clip_gradients` pulls per-Var grad norm to scalar (`ppo.rs@33`); `clip_fraction` transfers full ratio every minibatch (`ppo.rs@526-532`). **Fix:** hoist `log_probs` to host once and index; batch device→host via existing `ReplayBuffer::add_tensor` (`replay.rs@231`); accumulate grad-norm on-device, read once; compute clip-fraction on-device. **Bench:** `env_benchmark`/`gpu_benchmark` on Metal with large `num_envs`. *(3, 19, 36, 4, 12)*

4. **`buffer/mod.rs` GAE allocates two `Tensor::ones` per timestep + thousands of tiny kernel launches.** `mod.rs@391-440` (ones @411,423). **Fix:** hoist the ones tensors; better, compute on flat `Vec<f32>` like `algorithms/rollout.rs` or call `simd::gae::compute_gae_simd_inplace`. **Bench:** micro-bench `compute_returns_and_advantages` at `buffer_size=2048, num_envs=64` vs the scalar rollout path. *(10)*

5. **Streaming metrics aren't streaming.** `metrics/trading.rs@249-253,303-327` runs full O(window) `recalculate_drawdown` every `add_return`; `@531-582` clone+full-sort the return deque twice per `compute_metrics` (VaR + CVaR). **Fix:** incremental drawdown (monotonic-deque running peak); single `select_nth_unstable` for VaR/CVaR. **Bench:** `add_return` throughput at `rolling_window=252`; `compute_metrics()` at n=1000. *(99, 100)*

6. **SelfAttention rebuilds the causal mask (host alloc + H2D + dtype cast) every forward.** `attention.rs@186-189,238-249` (TransformerEncoder already precomputes once). **Fix:** precompute in `new()` to `max_seq_len`, narrow in forward. **Bench:** `SelfAttention::forward` causal=true, seq_len=128, Metal. *(61)*

7. **Lower-impact allocation churn.** `Vec::remove(0)` ring buffers (`metrics.rs@150-153,260-263` → `VecDeque`); `NStepReplayBuffer.last_obs` dead per-step `to_vec()` (`nstep.rs@206`); Mmap element-wise f32 serialize vs `bytemuck::cast_slice` (`mmap.rs@270-294,341-369`); trading `current_prices/volatilities` alloc ×3/step (`multi_asset.rs@585-594`); vecenv `normalize_obs` round-trip + `std()` per-step alloc (`wrappers.rs@388-404,341-343`); FrameStack full re-concat per step (`wrappers.rs@122-140`); distribution host-RNG `from_slice` per sample (`gaussian.rs@149-162`, `categorical.rs@122-146`); SELU/LeakyReLU `zeros_like` per forward (`mlp.rs@41-56`); attention redundant first `.contiguous()` on K (`attention.rs@172-182`); BatchNorm eps tensor per forward (`normalization.rs@326-354`). *(38, 51, 52, 93, 71, 73, 72, 44, 45, 59, 65, 66)*

---

## P2 — RL mechanism improvements

- **PPO value clipping not implemented; `old_values` stored but unused; `explained_variance` hardcoded 0.** `ppo.rs@495-497,553`, `rollout.rs` materializes `values` for nothing. **Fix:** SB3-style clipped value loss behind `clip_range_vf`, or drop the wasted tensor; compute EV. (A2C/PPG EV also always 0: `a2c.rs@424`, `ppg.rs@860`.) *(6, 41)*
- **CQL importance-sampling correction computed then discarded** (assigned to `_correction`; uniform-action `-d·ln2` term absent) → biased conservative lower bound. `cql.rs@443-474`. **Fix:** subtract per-sample log-densities before logsumexp. *(27)*
- **PPG auxiliary phase deviates from the paper** (no auxiliary value head on the policy; BC target snapshotted at collection time across 32 stale policies, dragging the improved policy back). `ppg.rs@869-988`. **Fix:** snapshot pi_old from the current net at the start of the aux phase; add an aux value head or document as simplified. *(37)*
- **DQN/IQN Polyak soft update gated behind `target_update_interval`** (default 10k) → target effectively frozen with `tau=0.005`. `dqn.rs@417-426`, `iqn.rs@732-741`. **Fix:** soft-update every step; let the interval gate only hard updates. *(29)*
- **Epsilon decayed once per loop iteration, not per timestep** → schedule silently scales with `num_envs`. `dqn.rs@344-346`, `iqn.rs@654-656`. **Fix:** decay `×num_envs`, or compute ε from `total_timesteps`. *(30)*
- **Normalize wrappers keep per-env running stats when vectorized** (vs shared SB3 `VecNormalize`) → inconsistent normalization across the batch, num_envs× slower warmup. `wrappers.rs@347-405` via `make_vectorized`. **Fix:** apply normalization as a VecEnv-level wrapper with one shared `RunningMeanStd`. *(70)*
- **`regime_persistence` has inverted semantics** (counts toward a change, resets while stable → ~0 throughout any stable regime). `regime.rs@426-448`. **Fix:** separate stability counter (time-in-regime) from a consecutive-candidate counter. *(94)*
- **`SquashedGaussian::entropy` returns the unsquashed (unbounded) Gaussian entropy.** `gaussian.rs@340-344`. **Fix:** Monte-Carlo `-log_prob`, or subtract `E[Σ log(1-tanh²u)]`; steer SAC callers to `-log_prob`. *(46)*
- **Convention inconsistencies across metrics modules:** Sortino downside-dev divides by count-of-negatives vs total-N (`rewards.rs@121-127` vs `trading.rs@355-361`); biased `/n` vs unbiased `/(n-1)` variance (`rewards.rs@105-113`); calmar/recovery mix full-history return with window drawdown (`trading.rs@401-431`). **Fix:** pick one convention crate-wide. *(101, 104, 103)*

---

## PyO3 / uv binding surface

**Foundational blockers (must clear in order):**
1. **Not built as an extension module.** `Cargo.toml@1-25,66-67` has no `[lib] crate-type`; pyo3 is gated under `gym`/`wandb` only, no `extension-module`/`abi3`, no `#[pymodule]`. **Fix:** `[lib] crate-type=["cdylib","lib"]`; a `python` feature with `pyo3={version="0.24",features=["extension-module","abi3-py39"]}` + `numpy`; `pyproject.toml` (maturin); a `#[pymodule]`. *(107)*
2. **All agents are generic `XAgent<E: Environment+Clone+'static>`** — `#[pyclass]` needs concrete, non-generic, Send, 'static. `ppo.rs@94`, `sac.rs@25`, … (only `CQLAgent` `cql.rs@65` is concrete). **Fix:** pick ONE concrete bridge env, emit newtype wrappers `#[pyclass] PyPPO(PPOAgent<BridgeEnv>)`; wrap `CQLAgent` directly. *(108)*
3. **Python-env bridge is impossible under current VecEnv design.** `Clone` is a *struct-level* bound (`vecenv.rs@204`) and `new` reaches `num_envs` by **cloning a template** (`@228-235`); `GymEnv` is non-Clone (`gym.rs@493`), and a `Py<PyAny>` "clone" is a refcount bump → N "parallel" envs share ONE Python object (data races). **Fix:** relax bound to `VecEnv<E: Environment>` (keep `Clone` on `fn new` only), add non-cloning `from_envs(Vec<E>)`, build N independent `PyEnv` objects; drop `E: Clone` from agents that only construct. *(109)*
4. **`candle::Tensor` saturates the public I/O surface; no `numpy` crate.** `traits.rs@8-64`, `ppo.rs@707 predict`, `VecStepResult`. Foreign type → can't `#[pyclass]`; current `gym.rs` marshals via Python `.tolist()` (unusably slow). **Fix:** add `numpy`; zero-copy `tensor_from_numpy`/`tensor_to_numpy` at the FFI edge only. *(110)*
5. **No `From<OctaneError> for PyErr`** (`error.rs@9-63`). **Fix:** feature-gated impl mapping `InvalidConfig/ShapeMismatch→PyValueError`, `Io→PyIOError`, rest→`PyRuntimeError`. *(111)*
6. **GIL/threading:** `train()` runs for minutes; a `PyEnv` per-step re-enters Python and fights `allow_threads`, killing the parallelism premise. `ppo.rs@643-704`, `sac.rs@486-597`. **Fix:** run `.train()` over **native** envs inside `py.allow_threads(...)`, re-acquiring the GIL only to fire an optional callback. *(112)*

**Per-type wrapping strategy (low effort once blockers clear):**
- **Device** (`device.rs@9-37`): POD, maps to a small pyclass enum; expose `cpu()/metal()/cuda(i)` staticmethods returning a clear PyErr when the cfg feature is absent. *(114)*
- **Configs** (`config.rs`, all `Clone+Serialize`): `#[pyclass]` with `#[new](**kwargs)` over `Default`, plus `from_json/to_json` reusing serde. *(115)*
- **Step results** (`VecStepResult` `vecenv.rs@168-191`): `#[pyclass] PyVecStepResult` with numpy arrays + `list[dict]` infos via the numpy helpers. *(116)*
- **Buffers** (`replay.rs`, `RolloutBuffer`): keep internal to wrapped agents (hold `Device`, non-Send-ish `StdRng`/`SumTree`, emit Tensor batches). *(117)*
- **Send check:** add `fn _assert_send<T:Send>(){}` instantiated on each wrapper and compile under `--features metal` (agents hold `VarMap`/`Tensor`/`AdamW`/`StdRng` — Send by design, but verify). *(118)*

**Recommended architecture (113):** wrap the **native Rust trading envs** (`AdvancedTradingEnv`/`TradingEnv`/`MultiAssetEnv`, all `Send+Sync+Clone`), construct from numpy OHLCV once at `__init__`, monomorphize agents over them, run `.learn()` entirely in Rust under `allow_threads`, marshal numpy only at construction and `predict()`. Keep the Gym per-step bridge behind a `gym` extra as a documented slow/serial fallback.

---

## MLX side-path candidates (ranked, honest payoff)

1. **Rollout policy+value forward — #1 candidate, but conditional and partly recoverable in Candle.** `ppo.rs@207-266`, called per-step in `collect_rollout@588-633`. Weights are static across a rollout, so MLX could marshal weights once and run `n_steps` forward-only passes, fusing `linear+tanh` into far fewer dispatches. **Two honest caveats:** (a) much of the apparent win is just "stop rebuilding `Linear` via `format!` every call" — do that **free Candle fix first** (P1 #2) so the comparison is honest; (b) the env runs CPU-side, so obs/actions cross the boundary every step — the forward can't stay resident. Payoff only materializes at large `num_envs`×`hidden`. **Bench:** MLX fused MLP vs candle-metal across `num_envs∈{8,256,1024,4096}`, including both-direction marshaling. *(119)*
2. **PPO/SAC update loop — the only compute-dense region, but NOT a viable marshaled side-path.** `ppo.rs@455-543`, `sac.rs@389-483`. To compute gradients MLX must *own* the parameters; as a side-path you'd round-trip all weights+grads every minibatch (PPO ~10×n_batches, SAC every step), dwarfing the matmul savings on [256,256] nets. **Conclusion:** MLX pays off in training only via a **full backend migration** that owns params + optimizer (which would also fix the SAC per-step optimizer recreation), not a cheap experiment. *(123)*
3. **Discrete categorical sampling — recoverable in Candle, skip MLX.** `ppo.rs@272-314` does a per-step CPU inverse-CDF loop with a per-element `to_vec1`. The repo already has on-tensor `Categorical::sample_gumbel_max` (`categorical.rs@122-146`); route PPO through it. Only consider `mlx random::categorical` if the forward is already MLX-resident. *(120)*
4. **Do NOT port Gaussian log-prob / PPO loss-reduction as marshaled MLX kernels.** `metal.rs@461-697` is the anti-pattern: per-call buffer alloc + `commit()` + `wait_until_completed()` + `Vec<f32>` round-trip, and no autodiff path (returns `Vec<f32>`, can't participate in `backward()`). Keep these inside the autodiff graph. *(121)*
5. **Do NOT build a standalone MLX Gaussian sampler.** `gaussian.rs@149-201` — `randn`+axpy is memory-bound; marshaling exceeds the arithmetic. Only fuse sampling into an on-device forward if MLX is adopted. *(122)*

---

## P3 — Cleanups / lower priority

- **Dead/duplicated code:** unused `PPOAgent::compute_gae` that would conflate terminated/truncated if wired (`ppo.rs@394-419`); two distinct public `RolloutBuffer` types, the slower one re-exported under the prelude name (`buffer/mod.rs@76` vs `algorithms/rollout.rs@58`, `lib.rs@65,220`); duplicate private `SumTree` shadowing the unused `segment_tree::MinTree` (`replay.rs@414-480`); dead no-op `actor_input_dim`/`critic_input_dim` if/else (`actor_critic.rs@323-346`); `NStep.last_obs` never read (`nstep.rs@128`); A2C dead `rms_prop_eps` + misleading RMSprop comment (`config.rs@251-253`). *(8, 9, 50, 60, 51, 41)*
- **Silent no-op config:** dropout fields/builders throughout transformer/attention but no `Dropout` ever instantiated (`transformer.rs@41,87`, `attention.rs@36,263`). Wire it or remove. *(64)*
- **Misleading docs:** `rollout.rs@24-32` claims SIMD GAE needs the `simd` cargo feature, but it's gated on `target_feature=neon` (always-on for aarch64). *(11)*
- **Numerical edge cases (mostly latent/defaults-safe):** `RunningMeanStd` Welford combine in f32 freezes variance after ~16M samples (`wrappers.rs@303-318` → accumulate in f64); RNN `init_state` hardcodes `DType::F32`, breaks `half` (`rnn.rs@191-199,413-421`); HMM forward can underflow to all-zero and freeze (`regime.rs@200-207,453-484` → log-space); `GarchParams::unconditional_variance` no stationarity guard (`regime.rs@243-245`); `profit_factor` returns `Infinity` → serde_json serializes `null`, breaking round-trip (`trading.rs@459-467`); categorical inverse-CDF biases toward action 0 on underflow (`a2c.rs@248-257` → default to last index); `ReplayBuffer::clear` doesn't reset `current_beta` (`replay.rs@393`); SquashedGaussian inconsistent atanh/Jacobian clamps (`gaussian.rs@291-327`); BoxSpace unbounded sample is uniform-not-normal and ignores one-sided finite bounds (`space.rs@90-110`); trading `build_observation` uses `insert(0,..)` padding, multi_timeframe pad guard can't pad per-timeframe (`multi_timeframe.rs@616-621`). *(69, 57, 95, 96, 106, 40, 53, 43, 74, 97)*
- **SIMD AVX2/Metal accuracy & robustness (latent — public, not in active path):** AVX2 log-prob mixes approximate `fast_exp/log/tanh` body with exact libm tail (disagrees with NEON/scalar) (`log_prob.rs@447-635`); `fast_sincos` Taylor over full `[-π,π]` distorts Box-Muller near ±π (`x86.rs@362-407`); `+avx2` build with failed runtime detection silently leaves zeros (`gae.rs@440-524`); naive f32 reductions in `normalize_advantages_simd` (`gae.rs@628-646`); Metal `create_buffer` never checks nil (dead `BufferCreationFailed`) (`metal.rs@425-441`); serial `softmax_row` with no-op barriers (`metal.rs@212-256`); per-call Metal buffer alloc + blocking sync + triple host round-trip (`metal.rs@425-441,494-495,…`); whole `MetalContext` is dead-but-public — mark experimental/`doc(hidden)` until the OOB/`obs_dim` bugs (P0 #21) are fixed. *(76, 77, 78, 79, 89, 88, 82, 83, 90)*
---

## Implementation status (this branch)

### ✅ Fixed & tested (committed)
- **Optimizer persistence (P0 #1)** — all 9 agents (SAC/TD3/DDPG/REDQ/DQN/IQN/CQL/A2C/PPG): AdamW now built once and stored, Adam state survives updates.
- **Entropy-temperature sign (P0 #2)** — SAC/REDQ/CQL alpha auto-tune un-inverted + log_alpha clamp.
- **IQN quantile TD-error sign (P0 #6)**, **REDQ mean-vs-min actor (P0 #8)**, **DQN Huber NaN-grad (P0 #20)**.
- **PPO/PPG KL early-stop double-break (P0 #9)**, **A2C gradient clipping (P0 #10)**.
- **A2C & PPG trainable continuous log_std (P0 #5)**.
- **Categorical::log_prob autograd (P0 #12)**, **metrics sqrt(neg variance) clamp (P0 #19)**.
- **Latent runtime bugs found by new smoke tests:** SAC/REDQ/CQL `log_alpha.to_scalar()` on `[1]` tensor; IQN `gather` on non-contiguous tensors. Both broke default training paths at runtime.
- Tests: 7 end-to-end training smoke tests + 2 targeted regression tests added (359 total, green).

### ⏳ Deferred (documented roadmap — higher blast radius / needs data-flow reconciliation)
- **Truncation/terminal-obs in VecEnv (P0 #3)** — requires changing `VecStepResult` (+7 consumers); the `rollout.rs` GAE already handles truncation correctly, the claim is that VecEnv discards terminal obs upstream. Needs data-flow trace before touching.
- **On-policy hard-reset every rollout (P0 #4)** — persist `last_obs` across rollouts (PPO/A2C/PPG).
- **Network build-once (P1 #2)** — another all-9-agent invasive refactor; gate behind the new smoke tests.
- Orthogonal init wiring (#13), FrameStack multi-dim shape (#14), BatchNorm running stats (#15), epsilon-greedy per-row (#16), DQN/IQN soft-update cadence (#29), CQL conservative logsumexp/Q-mixing (#7), PPG grad-clip (#34), and the remaining P2/P3 items.
