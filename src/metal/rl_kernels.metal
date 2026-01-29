//
// rl_kernels.metal
// VortexRL - High-Performance Metal Compute Shaders for Reinforcement Learning
//
// Optimized for Apple M4 GPU architecture with:
// - Fused operations to minimize memory bandwidth
// - Warp-level (SIMD group) primitives for parallel reductions
// - Efficient threadgroup memory usage
// - FP32 precision for numerical stability
//
// Copyright (c) 2024-2026 VortexRL Project
// SPDX-License-Identifier: GPL-2.0
//

#include <metal_stdlib>
#include <metal_math>
#include <metal_simdgroup>

using namespace metal;

// =============================================================================
// Constants
// =============================================================================

constant float LOG_2PI = 1.8378770664093453f;       // ln(2 * pi)
constant float HALF_LOG_2PI = 0.9189385332046727f;  // 0.5 * ln(2 * pi)
constant float EPSILON = 1e-8f;                     // Numerical stability
constant float LOG_STD_MIN = -20.0f;                // Minimum log std for clamping
constant float LOG_STD_MAX = 2.0f;                  // Maximum log std for clamping

// Threadgroup sizes optimized for M4 GPU
constant uint SIMD_SIZE = 32;                       // Apple Silicon SIMD width
constant uint THREADGROUP_SIZE = 256;               // Optimal for M4


// =============================================================================
// 1. Fused Gaussian Log Probability
// =============================================================================
//
// Computes: log_prob = -0.5 * ((x - mean) / std)^2 - log(std) - 0.5 * log(2*pi)
//
// This fused kernel eliminates intermediate memory allocations and reduces
// memory bandwidth by computing everything in a single pass.
//
// Input shapes (all [batch_size] or [batch_size * action_dim] when flattened):
//   x:        sampled actions
//   mean:     distribution mean
//   std:      distribution standard deviation (NOT log_std)
// Output:
//   log_prob: log probability for each element
//
// For multi-dimensional actions, sum the results across action dimensions.

kernel void gaussian_log_prob(
    device const float* x        [[buffer(0)]],
    device const float* mean     [[buffer(1)]],
    device const float* std      [[buffer(2)]],
    device float* log_prob       [[buffer(3)]],
    constant uint& count         [[buffer(4)]],
    uint id                      [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float x_val = x[id];
    float mean_val = mean[id];
    float std_val = std[id];

    // Clamp std for numerical stability
    std_val = max(std_val, EPSILON);

    // Compute normalized difference: z = (x - mean) / std
    float z = (x_val - mean_val) / std_val;

    // Fused log probability computation:
    // -0.5 * z^2 - log(std) - 0.5 * log(2*pi)
    float log_std = log(std_val);
    float log_p = -0.5f * z * z - log_std - HALF_LOG_2PI;

    log_prob[id] = log_p;
}


// =============================================================================
// 1b. Fused Gaussian Log Probability with Log-Std Input
// =============================================================================
//
// Same as above but takes log_std directly (common in RL networks).
// Avoids exp() followed by log() for efficiency.

kernel void gaussian_log_prob_log_std(
    device const float* x           [[buffer(0)]],
    device const float* mean        [[buffer(1)]],
    device const float* log_std     [[buffer(2)]],
    device float* log_prob          [[buffer(3)]],
    constant uint& count            [[buffer(4)]],
    uint id                         [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float x_val = x[id];
    float mean_val = mean[id];
    float log_std_val = log_std[id];

    // Clamp log_std for numerical stability
    log_std_val = clamp(log_std_val, LOG_STD_MIN, LOG_STD_MAX);

    // std = exp(log_std)
    float std_val = exp(log_std_val);

    // Compute normalized difference: z = (x - mean) / std
    float z = (x_val - mean_val) / std_val;

    // Fused log probability:
    // -0.5 * z^2 - log_std - 0.5 * log(2*pi)
    float log_p = -0.5f * z * z - log_std_val - HALF_LOG_2PI;

    log_prob[id] = log_p;
}


// =============================================================================
// 1c. Batched Gaussian Log Probability with Reduction
// =============================================================================
//
// Computes log probability for batched multi-dimensional actions and reduces
// across action dimensions in a single kernel.
//
// Input shapes:
//   x, mean, std: [batch_size, action_dim] (row-major)
// Output:
//   log_prob: [batch_size] (sum across action_dim)

kernel void gaussian_log_prob_batched(
    device const float* x           [[buffer(0)]],
    device const float* mean        [[buffer(1)]],
    device const float* log_std     [[buffer(2)]],
    device float* log_prob          [[buffer(3)]],
    constant uint& batch_size       [[buffer(4)]],
    constant uint& action_dim       [[buffer(5)]],
    uint batch_id                   [[thread_position_in_grid]]
) {
    if (batch_id >= batch_size) return;

    float sum_log_prob = 0.0f;
    uint base_idx = batch_id * action_dim;

    for (uint d = 0; d < action_dim; d++) {
        uint idx = base_idx + d;

        float x_val = x[idx];
        float mean_val = mean[idx];
        float log_std_val = clamp(log_std[idx], LOG_STD_MIN, LOG_STD_MAX);
        float std_val = exp(log_std_val);

        float z = (x_val - mean_val) / std_val;
        float log_p = -0.5f * z * z - log_std_val - HALF_LOG_2PI;

        sum_log_prob += log_p;
    }

    log_prob[batch_id] = sum_log_prob;
}


// =============================================================================
// 2. PPO Clipped Surrogate Loss
// =============================================================================
//
// Computes the PPO clipped objective per sample:
//   ratio = exp(new_log_prob - old_log_prob)
//   surr1 = ratio * advantage
//   surr2 = clip(ratio, 1-eps, 1+eps) * advantage
//   loss = -min(surr1, surr2)
//
// The final loss should be averaged across the batch on the host.

kernel void ppo_clip_loss(
    device const float* old_log_probs [[buffer(0)]],
    device const float* new_log_probs [[buffer(1)]],
    device const float* advantages    [[buffer(2)]],
    device float* loss                [[buffer(3)]],
    constant float& clip_eps          [[buffer(4)]],
    constant uint& count              [[buffer(5)]],
    uint id                           [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float old_lp = old_log_probs[id];
    float new_lp = new_log_probs[id];
    float adv = advantages[id];

    // Compute probability ratio using log subtraction for stability
    float log_ratio = new_lp - old_lp;

    // Clamp log_ratio to prevent exp overflow
    log_ratio = clamp(log_ratio, -20.0f, 20.0f);
    float ratio = exp(log_ratio);

    // Unclipped surrogate
    float surr1 = ratio * adv;

    // Clipped surrogate
    float ratio_clipped = clamp(ratio, 1.0f - clip_eps, 1.0f + clip_eps);
    float surr2 = ratio_clipped * adv;

    // PPO objective: maximize min(surr1, surr2)
    // Loss to minimize: -min(surr1, surr2)
    float obj = min(surr1, surr2);
    loss[id] = -obj;
}


// =============================================================================
// 2b. PPO Clipped Loss with Metrics
// =============================================================================
//
// Extended version that also computes:
// - Approximate KL divergence: 0.5 * (log_ratio)^2
// - Clip fraction: ratio of samples where clipping occurred

kernel void ppo_clip_loss_with_metrics(
    device const float* old_log_probs [[buffer(0)]],
    device const float* new_log_probs [[buffer(1)]],
    device const float* advantages    [[buffer(2)]],
    device float* loss                [[buffer(3)]],
    device float* approx_kl           [[buffer(4)]],
    device float* clipped             [[buffer(5)]],  // 1.0 if clipped, 0.0 otherwise
    constant float& clip_eps          [[buffer(6)]],
    constant uint& count              [[buffer(7)]],
    uint id                           [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float old_lp = old_log_probs[id];
    float new_lp = new_log_probs[id];
    float adv = advantages[id];

    float log_ratio = new_lp - old_lp;
    log_ratio = clamp(log_ratio, -20.0f, 20.0f);
    float ratio = exp(log_ratio);

    // Surrogate objectives
    float surr1 = ratio * adv;
    float ratio_clipped = clamp(ratio, 1.0f - clip_eps, 1.0f + clip_eps);
    float surr2 = ratio_clipped * adv;

    // Loss
    loss[id] = -min(surr1, surr2);

    // Approximate KL divergence: 0.5 * (new_lp - old_lp)^2
    approx_kl[id] = 0.5f * log_ratio * log_ratio;

    // Clip indicator
    float abs_ratio_minus_one = abs(ratio - 1.0f);
    clipped[id] = abs_ratio_minus_one > clip_eps ? 1.0f : 0.0f;
}


// =============================================================================
// 3. Categorical Log Probability Gather
// =============================================================================
//
// Gathers log probabilities for selected discrete actions.
// This avoids CPU<->GPU data transfer for action indexing.
//
// Input:
//   log_probs: [batch_size, num_actions] - log softmax of action logits
//   actions:   [batch_size] - selected action indices (uint)
// Output:
//   output:    [batch_size] - log probability of each selected action

kernel void categorical_log_prob_gather(
    device const float* log_probs    [[buffer(0)]],
    device const uint* actions       [[buffer(1)]],
    device float* output             [[buffer(2)]],
    constant uint& num_actions       [[buffer(3)]],
    constant uint& batch_size        [[buffer(4)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= batch_size) return;

    uint action = actions[id];

    // Bounds check for safety
    action = min(action, num_actions - 1);

    // Gather: output[id] = log_probs[id, action]
    uint idx = id * num_actions + action;
    output[id] = log_probs[idx];
}


// =============================================================================
// 3b. Categorical Entropy
// =============================================================================
//
// Computes entropy of categorical distribution: H = -sum(p * log(p))
// Using log_probs for numerical stability: H = -sum(exp(log_p) * log_p)
//
// Input:
//   log_probs: [batch_size, num_actions]
// Output:
//   entropy: [batch_size]

kernel void categorical_entropy(
    device const float* log_probs    [[buffer(0)]],
    device float* entropy            [[buffer(1)]],
    constant uint& num_actions       [[buffer(2)]],
    constant uint& batch_size        [[buffer(3)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= batch_size) return;

    float ent = 0.0f;
    uint base_idx = id * num_actions;

    for (uint a = 0; a < num_actions; a++) {
        float log_p = log_probs[base_idx + a];
        float p = exp(log_p);
        // Avoid -inf * 0 when p is very small
        if (p > EPSILON) {
            ent -= p * log_p;
        }
    }

    entropy[id] = ent;
}


// =============================================================================
// 4. Parallel Advantage Normalization
// =============================================================================
//
// Two-pass parallel reduction to compute mean and std, then normalize.
// Uses SIMD group operations for efficient warp-level reduction on M4.
//
// Pass 1: Compute partial sums and squared sums in threadgroups
// Pass 2: Apply normalization: (adv - mean) / (std + eps)

// Helper: SIMD group reduction for sum
inline float simd_sum(float val) {
    return simd_sum(val);
}

// Pass 1: Compute statistics (mean, variance) using parallel reduction
kernel void compute_advantage_stats(
    device const float* advantages   [[buffer(0)]],
    device float* partial_sums       [[buffer(1)]],  // [num_threadgroups]
    device float* partial_sq_sums    [[buffer(2)]],  // [num_threadgroups]
    constant uint& count             [[buffer(3)]],
    uint id                          [[thread_position_in_grid]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint gid                         [[threadgroup_position_in_grid]],
    uint tg_size                     [[threads_per_threadgroup]],
    threadgroup float* shared_sum    [[threadgroup(0)]],
    threadgroup float* shared_sq_sum [[threadgroup(1)]]
) {
    // Load data with bounds check
    float val = (id < count) ? advantages[id] : 0.0f;
    float val_sq = val * val;

    // Store to threadgroup memory
    shared_sum[tid] = val;
    shared_sq_sum[tid] = val_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Parallel reduction within threadgroup
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // First thread writes partial results
    if (tid == 0) {
        partial_sums[gid] = shared_sum[0];
        partial_sq_sums[gid] = shared_sq_sum[0];
    }
}

// Final reduction and output mean/std
kernel void finalize_advantage_stats(
    device const float* partial_sums    [[buffer(0)]],
    device const float* partial_sq_sums [[buffer(1)]],
    device float* mean_std              [[buffer(2)]],  // [mean, std]
    constant uint& num_partials         [[buffer(3)]],
    constant uint& total_count          [[buffer(4)]],
    uint tid                            [[thread_index_in_threadgroup]],
    threadgroup float* shared_sum       [[threadgroup(0)]],
    threadgroup float* shared_sq_sum    [[threadgroup(1)]]
) {
    // Load partial sums
    float sum = (tid < num_partials) ? partial_sums[tid] : 0.0f;
    float sq_sum = (tid < num_partials) ? partial_sq_sums[tid] : 0.0f;

    shared_sum[tid] = sum;
    shared_sq_sum[tid] = sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction (assuming num_partials <= threadgroup size)
    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride && tid + stride < THREADGROUP_SIZE) {
            shared_sum[tid] += shared_sum[tid + stride];
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute and output mean/std
    if (tid == 0) {
        float n = float(total_count);
        float mean = shared_sum[0] / n;
        float variance = (shared_sq_sum[0] / n) - (mean * mean);
        float std = sqrt(max(variance, 0.0f) + EPSILON);

        mean_std[0] = mean;
        mean_std[1] = std;
    }
}

// Pass 2: Apply normalization
kernel void normalize_advantages(
    device float* advantages         [[buffer(0)]],
    device const float* mean_std     [[buffer(1)]],  // [mean, std]
    constant uint& count             [[buffer(2)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float mean = mean_std[0];
    float std = mean_std[1];

    // Normalize: (adv - mean) / std
    advantages[id] = (advantages[id] - mean) / std;
}


// =============================================================================
// 4b. Single-Pass Advantage Normalization (for small batches)
// =============================================================================
//
// For small batch sizes that fit in a single threadgroup, compute stats and
// normalize in one kernel launch. More efficient for typical RL batch sizes.

kernel void normalize_advantages_single_pass(
    device float* advantages         [[buffer(0)]],
    constant uint& count             [[buffer(1)]],
    uint tid                         [[thread_index_in_threadgroup]],
    uint tg_size                     [[threads_per_threadgroup]],
    threadgroup float* shared_data   [[threadgroup(0)]]
) {
    // Phase 1: Load data
    float val = (tid < count) ? advantages[tid] : 0.0f;
    shared_data[tid] = val;
    shared_data[tid + tg_size] = val * val;  // Store squared values in second half
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Parallel reduction for sum and sum of squares
    for (uint stride = tg_size / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
            shared_data[tid + tg_size] += shared_data[tid + tg_size + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compute mean and std (all threads read the same values)
    float n = float(count);
    float mean = shared_data[0] / n;
    float variance = (shared_data[tg_size] / n) - (mean * mean);
    float std = sqrt(max(variance, 0.0f) + EPSILON);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Normalize and write back
    if (tid < count) {
        advantages[tid] = (val - mean) / std;
    }
}


// =============================================================================
// 5. Value Function Loss (MSE)
// =============================================================================
//
// Computes MSE loss: loss = (predicted - target)^2
// Can also compute Huber loss for robustness to outliers.

kernel void value_loss_mse(
    device const float* predicted    [[buffer(0)]],
    device const float* target       [[buffer(1)]],
    device float* loss               [[buffer(2)]],
    constant uint& count             [[buffer(3)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float diff = predicted[id] - target[id];
    loss[id] = diff * diff;
}

// Huber loss (smooth L1) - more robust to outliers
kernel void value_loss_huber(
    device const float* predicted    [[buffer(0)]],
    device const float* target       [[buffer(1)]],
    device float* loss               [[buffer(2)]],
    constant float& delta            [[buffer(3)]],  // Huber threshold (typically 1.0)
    constant uint& count             [[buffer(4)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float diff = predicted[id] - target[id];
    float abs_diff = abs(diff);

    // Huber: 0.5 * x^2 if |x| <= delta, else delta * (|x| - 0.5 * delta)
    if (abs_diff <= delta) {
        loss[id] = 0.5f * diff * diff;
    } else {
        loss[id] = delta * (abs_diff - 0.5f * delta);
    }
}


// =============================================================================
// 6. Generalized Advantage Estimation (GAE)
// =============================================================================
//
// Computes GAE in a parallelizable manner. Due to the sequential nature of GAE,
// this uses a parallel scan algorithm for efficiency.
//
// GAE formula:
//   delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
//   A_t = sum_{l=0}^{T-t} (gamma * lambda)^l * delta_{t+l}

// First pass: Compute TD errors (deltas)
kernel void gae_compute_deltas(
    device const float* rewards      [[buffer(0)]],
    device const float* values       [[buffer(1)]],
    device const float* next_values  [[buffer(2)]],  // values[t+1], last is bootstrap
    device const float* dones        [[buffer(3)]],
    device float* deltas             [[buffer(4)]],
    constant float& gamma            [[buffer(5)]],
    constant uint& count             [[buffer(6)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float r = rewards[id];
    float v = values[id];
    float v_next = next_values[id];
    float not_done = 1.0f - dones[id];

    // TD error: delta = r + gamma * V(s') * (1-done) - V(s)
    deltas[id] = r + gamma * v_next * not_done - v;
}

// Sequential GAE computation (for small sequences)
// Should be called with a single thread for correctness
kernel void gae_compute_advantages_sequential(
    device const float* deltas       [[buffer(0)]],
    device const float* dones        [[buffer(1)]],
    device float* advantages         [[buffer(2)]],
    constant float& gamma            [[buffer(3)]],
    constant float& gae_lambda       [[buffer(4)]],
    constant uint& count             [[buffer(5)]]
) {
    float gae = 0.0f;
    float gamma_lambda = gamma * gae_lambda;

    // Backward pass: compute GAE from end to beginning
    for (int t = int(count) - 1; t >= 0; t--) {
        float not_done = 1.0f - dones[t];
        gae = deltas[t] + gamma_lambda * not_done * gae;
        advantages[t] = gae;
    }
}


// =============================================================================
// 7. Action Clamping / Squashing
// =============================================================================
//
// For bounded continuous action spaces, apply tanh squashing
// and compute the log determinant of the Jacobian for log prob correction.

kernel void tanh_squash_actions(
    device const float* actions_raw  [[buffer(0)]],
    device float* actions_squashed   [[buffer(1)]],
    device float* log_det_jacobian   [[buffer(2)]],  // Can be nullptr if not needed
    constant uint& count             [[buffer(3)]],
    constant bool& compute_jacobian  [[buffer(4)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float x = actions_raw[id];
    float y = tanh(x);
    actions_squashed[id] = y;

    if (compute_jacobian) {
        // log |det(d(tanh(x))/dx)| = log(1 - tanh(x)^2)
        // = log(1 - y^2) but we need to be numerically stable
        float one_minus_y_sq = 1.0f - y * y;
        // Clamp to prevent log(0)
        one_minus_y_sq = max(one_minus_y_sq, EPSILON);
        log_det_jacobian[id] = log(one_minus_y_sq);
    }
}


// =============================================================================
// 8. Replay Buffer Sampling (Uniform)
// =============================================================================
//
// Efficiently gather samples from replay buffer using random indices.
// The RNG is expected to generate indices on CPU and pass them to GPU.

kernel void gather_replay_samples(
    device const float* buffer       [[buffer(0)]],  // [buffer_size, feature_dim]
    device const uint* indices       [[buffer(1)]],  // [batch_size]
    device float* output             [[buffer(2)]],  // [batch_size, feature_dim]
    constant uint& buffer_size       [[buffer(3)]],
    constant uint& feature_dim       [[buffer(4)]],
    constant uint& batch_size        [[buffer(5)]],
    uint2 gid                        [[thread_position_in_grid]]
) {
    uint batch_idx = gid.x;
    uint feat_idx = gid.y;

    if (batch_idx >= batch_size || feat_idx >= feature_dim) return;

    uint buffer_idx = indices[batch_idx];
    buffer_idx = min(buffer_idx, buffer_size - 1);  // Safety clamp

    uint src_idx = buffer_idx * feature_dim + feat_idx;
    uint dst_idx = batch_idx * feature_dim + feat_idx;

    output[dst_idx] = buffer[src_idx];
}


// =============================================================================
// 9. Softmax with Temperature (for exploration)
// =============================================================================
//
// Computes softmax with temperature parameter for controlling exploration:
// softmax(x / temperature)

kernel void softmax_with_temperature(
    device const float* logits       [[buffer(0)]],
    device float* probs              [[buffer(1)]],
    constant float& temperature      [[buffer(2)]],
    constant uint& batch_size        [[buffer(3)]],
    constant uint& num_actions       [[buffer(4)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= batch_size) return;

    uint base = id * num_actions;
    float inv_temp = 1.0f / max(temperature, EPSILON);

    // Find max for numerical stability
    float max_logit = logits[base];
    for (uint a = 1; a < num_actions; a++) {
        max_logit = max(max_logit, logits[base + a]);
    }

    // Compute exp sum
    float exp_sum = 0.0f;
    for (uint a = 0; a < num_actions; a++) {
        float scaled = (logits[base + a] - max_logit) * inv_temp;
        exp_sum += exp(scaled);
    }

    // Compute probabilities
    float inv_sum = 1.0f / exp_sum;
    for (uint a = 0; a < num_actions; a++) {
        float scaled = (logits[base + a] - max_logit) * inv_temp;
        probs[base + a] = exp(scaled) * inv_sum;
    }
}


// =============================================================================
// 10. Reward Normalization (Running Statistics)
// =============================================================================
//
// Updates running mean and variance for reward normalization.
// Uses Welford's online algorithm for numerical stability.

kernel void update_reward_stats(
    device const float* rewards      [[buffer(0)]],
    device float* running_mean       [[buffer(1)]],  // Single float
    device float* running_var        [[buffer(2)]],  // Single float
    device uint* count_ptr           [[buffer(3)]],  // Running count
    constant uint& batch_size        [[buffer(4)]],
    uint tid                         [[thread_index_in_threadgroup]],
    threadgroup float* shared_data   [[threadgroup(0)]]
) {
    // Compute batch statistics first
    float local_sum = 0.0f;
    float local_sq_sum = 0.0f;

    // Each thread processes multiple elements
    uint elements_per_thread = (batch_size + THREADGROUP_SIZE - 1) / THREADGROUP_SIZE;
    uint start = tid * elements_per_thread;
    uint end = min(start + elements_per_thread, batch_size);

    for (uint i = start; i < end; i++) {
        float r = rewards[i];
        local_sum += r;
        local_sq_sum += r * r;
    }

    shared_data[tid] = local_sum;
    shared_data[tid + THREADGROUP_SIZE] = local_sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce
    for (uint stride = THREADGROUP_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_data[tid] += shared_data[tid + stride];
            shared_data[tid + THREADGROUP_SIZE] += shared_data[tid + THREADGROUP_SIZE + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Update running statistics (only thread 0)
    if (tid == 0) {
        float batch_sum = shared_data[0];
        float batch_sq_sum = shared_data[THREADGROUP_SIZE];
        float batch_mean = batch_sum / float(batch_size);
        float batch_var = (batch_sq_sum / float(batch_size)) - (batch_mean * batch_mean);

        // Combine with running statistics using parallel algorithm
        uint old_count = count_ptr[0];
        uint new_count = old_count + batch_size;

        float old_mean = running_mean[0];
        float old_var = running_var[0];

        // Delta between means
        float delta = batch_mean - old_mean;

        // Update mean
        float new_mean = old_mean + delta * float(batch_size) / float(new_count);

        // Update variance using parallel algorithm
        float new_var;
        if (old_count == 0) {
            new_var = batch_var;
        } else {
            float m_a = old_var * float(old_count);
            float m_b = batch_var * float(batch_size);
            new_var = (m_a + m_b + delta * delta * float(old_count) * float(batch_size) / float(new_count)) / float(new_count);
        }

        running_mean[0] = new_mean;
        running_var[0] = max(new_var, EPSILON);
        count_ptr[0] = new_count;
    }
}

// Apply reward normalization
kernel void normalize_rewards(
    device float* rewards            [[buffer(0)]],
    device const float* running_mean [[buffer(1)]],
    device const float* running_var  [[buffer(2)]],
    constant uint& count             [[buffer(3)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= count) return;

    float mean = running_mean[0];
    float std = sqrt(running_var[0]);

    rewards[id] = (rewards[id] - mean) / std;
}


// =============================================================================
// 11. Entropy Bonus Computation
// =============================================================================
//
// Computes entropy bonus for continuous (Gaussian) policies.
// Entropy of Gaussian: H = 0.5 * log(2 * pi * e * var) per dimension
//                     = 0.5 + 0.5 * log(2*pi) + log(std) per dimension

kernel void gaussian_entropy(
    device const float* log_std      [[buffer(0)]],  // [batch_size, action_dim]
    device float* entropy            [[buffer(1)]],  // [batch_size]
    constant uint& batch_size        [[buffer(2)]],
    constant uint& action_dim        [[buffer(3)]],
    uint id                          [[thread_position_in_grid]]
) {
    if (id >= batch_size) return;

    float ent = 0.0f;
    float const_term = 0.5f + HALF_LOG_2PI;  // 0.5 * (1 + log(2*pi))

    uint base = id * action_dim;
    for (uint d = 0; d < action_dim; d++) {
        float log_s = clamp(log_std[base + d], LOG_STD_MIN, LOG_STD_MAX);
        ent += const_term + log_s;
    }

    entropy[id] = ent;
}


// =============================================================================
// 12. Combined PPO Loss Kernel
// =============================================================================
//
// Computes all PPO losses in a single kernel for maximum efficiency:
// - Policy loss (clipped surrogate)
// - Value loss (MSE or Huber)
// - Entropy bonus
// Returns individual losses for separate coefficient weighting on host.

kernel void ppo_combined_loss(
    device const float* old_log_probs [[buffer(0)]],
    device const float* new_log_probs [[buffer(1)]],
    device const float* advantages    [[buffer(2)]],
    device const float* values        [[buffer(3)]],
    device const float* returns       [[buffer(4)]],
    device const float* entropy       [[buffer(5)]],
    device float* policy_loss         [[buffer(6)]],
    device float* value_loss          [[buffer(7)]],
    device float* entropy_loss        [[buffer(8)]],
    constant float& clip_eps          [[buffer(9)]],
    constant float& vf_clip_eps       [[buffer(10)]],  // Value function clipping (0 to disable)
    constant uint& count              [[buffer(11)]],
    uint id                           [[thread_position_in_grid]]
) {
    if (id >= count) return;

    // Policy loss (clipped surrogate)
    float log_ratio = new_log_probs[id] - old_log_probs[id];
    log_ratio = clamp(log_ratio, -20.0f, 20.0f);
    float ratio = exp(log_ratio);

    float adv = advantages[id];
    float surr1 = ratio * adv;
    float ratio_clipped = clamp(ratio, 1.0f - clip_eps, 1.0f + clip_eps);
    float surr2 = ratio_clipped * adv;
    policy_loss[id] = -min(surr1, surr2);

    // Value loss (optionally clipped)
    float v_pred = values[id];
    float v_target = returns[id];
    float v_diff = v_pred - v_target;

    if (vf_clip_eps > 0.0f) {
        // Clipped value loss (PPO2 style)
        float v_clipped = clamp(v_pred, v_target - vf_clip_eps, v_target + vf_clip_eps);
        float v_loss1 = v_diff * v_diff;
        float v_loss2 = (v_clipped - v_target) * (v_clipped - v_target);
        value_loss[id] = max(v_loss1, v_loss2);
    } else {
        value_loss[id] = v_diff * v_diff;
    }

    // Entropy loss (negative because we maximize entropy)
    entropy_loss[id] = -entropy[id];
}
