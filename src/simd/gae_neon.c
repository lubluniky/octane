/**
 * @file gae_neon.c
 * @brief Generalized Advantage Estimation (GAE) with ARM NEON for Apple M4
 *
 * High-performance implementation of GAE computation using ARM NEON SIMD
 * instructions optimized for Apple Silicon (M4).
 *
 * GAE Formula:
 *   delta_t = reward_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
 *   A_t = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
 *
 * Features:
 * - Processes 4 environments in parallel using float32x4_t
 * - Fused multiply-add operations (vfmaq_f32)
 * - Handles arbitrary num_envs with vectorized loop + scalar remainder
 * - Row-major layout: [buffer_size, num_envs]
 *
 * @copyright 2024 RocketRL Project
 * @license GPL-2.0
 */

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>

/**
 * @brief Compute GAE advantages using ARM NEON SIMD
 *
 * Computes Generalized Advantage Estimation for on-policy RL algorithms
 * (PPO, A2C) using vectorized NEON operations.
 *
 * The computation iterates backwards through time due to the recursive
 * dependency in the GAE formula, processing 4 environments in parallel.
 *
 * Memory Layout (row-major [buffer_size, num_envs]):
 *   rewards[t][e]    = rewards[t * num_envs + e]
 *   values[t][e]     = values[t * num_envs + e]
 *   dones[t][e]      = dones[t * num_envs + e]
 *   advantages[t][e] = advantages[t * num_envs + e]
 *
 * @param rewards       Reward buffer [buffer_size, num_envs]
 * @param values        Value estimates [buffer_size, num_envs]
 * @param dones         Episode termination flags [buffer_size, num_envs], 1.0 if done
 * @param advantages    Output advantage buffer [buffer_size, num_envs]
 * @param buffer_size   Number of timesteps in rollout
 * @param num_envs      Number of parallel environments
 * @param gamma         Discount factor (typically 0.99)
 * @param gae_lambda    GAE lambda parameter (typically 0.95)
 * @param last_values   Bootstrap values for final step [num_envs]
 */
void gae_compute_neon(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    size_t buffer_size,
    size_t num_envs,
    float gamma,
    float gae_lambda,
    const float* last_values
) {
    /* Early exit for empty buffer */
    if (buffer_size == 0 || num_envs == 0) {
        return;
    }

    /* Broadcast scalars to NEON vectors */
    const float32x4_t v_gamma = vdupq_n_f32(gamma);
    const float32x4_t v_gae_lambda = vdupq_n_f32(gae_lambda);
    const float32x4_t v_one = vdupq_n_f32(1.0f);
    const float32x4_t v_gamma_lambda = vdupq_n_f32(gamma * gae_lambda);

    /* Number of complete 4-wide groups */
    const size_t num_vec = num_envs / 4;
    const size_t remainder = num_envs % 4;

    /*
     * Reverse iteration through timesteps.
     * GAE has temporal dependency: A_t depends on A_{t+1}
     * We start from the last timestep and work backwards.
     */
    for (size_t t = buffer_size; t > 0; --t) {
        const size_t curr_idx = t - 1;
        const size_t row_offset = curr_idx * num_envs;

        /* Pointers for current timestep */
        const float* reward_row = rewards + row_offset;
        const float* value_row = values + row_offset;
        const float* done_row = dones + row_offset;
        float* adv_row = advantages + row_offset;

        /* Next values: either bootstrap (last step) or values[t+1] */
        const float* next_value_row;
        const float* next_adv_row;

        if (t == buffer_size) {
            /* Last timestep: use bootstrap values, advantage starts at 0 */
            next_value_row = last_values;
            next_adv_row = NULL;  /* Will use zero */
        } else {
            next_value_row = values + (t * num_envs);
            next_adv_row = advantages + (t * num_envs);
        }

        /* Process 4 environments at a time with NEON */
        size_t e = 0;
        for (size_t vec_idx = 0; vec_idx < num_vec; ++vec_idx, e += 4) {
            /* Load current timestep data */
            float32x4_t v_reward = vld1q_f32(reward_row + e);
            float32x4_t v_value = vld1q_f32(value_row + e);
            float32x4_t v_done = vld1q_f32(done_row + e);

            /* Load next values */
            float32x4_t v_next_value = vld1q_f32(next_value_row + e);

            /* Compute not_done = 1 - done */
            float32x4_t v_not_done = vsubq_f32(v_one, v_done);

            /*
             * Compute TD residual (delta):
             * delta = reward + gamma * next_value * not_done - value
             *
             * Step by step:
             * 1. next_value * not_done (zeroes out if episode ended)
             * 2. gamma * (result)
             * 3. reward + (result)
             * 4. (result) - value
             */
            float32x4_t v_next_discounted = vmulq_f32(v_next_value, v_not_done);
            float32x4_t v_gamma_next = vmulq_f32(v_gamma, v_next_discounted);
            float32x4_t v_reward_plus = vaddq_f32(v_reward, v_gamma_next);
            float32x4_t v_delta = vsubq_f32(v_reward_plus, v_value);

            /*
             * Compute advantage:
             * advantage = delta + gamma * lambda * not_done * prev_advantage
             *
             * For last timestep, prev_advantage = 0, so advantage = delta
             */
            float32x4_t v_advantage;
            if (next_adv_row != NULL) {
                float32x4_t v_prev_adv = vld1q_f32(next_adv_row + e);

                /* gamma * lambda * not_done * prev_advantage */
                float32x4_t v_not_done_adv = vmulq_f32(v_not_done, v_prev_adv);

                /* advantage = delta + gamma_lambda * not_done * prev_adv */
                /* Using fused multiply-add: delta + gamma_lambda * not_done_adv */
                v_advantage = vfmaq_f32(v_delta, v_gamma_lambda, v_not_done_adv);
            } else {
                /* Last timestep: advantage = delta */
                v_advantage = v_delta;
            }

            /* Store computed advantages */
            vst1q_f32(adv_row + e, v_advantage);
        }

        /* Handle remaining environments (scalar fallback) */
        for (size_t r = 0; r < remainder; ++r, ++e) {
            float reward = reward_row[e];
            float value = value_row[e];
            float done = done_row[e];
            float next_value = next_value_row[e];
            float not_done = 1.0f - done;

            /* TD residual */
            float delta = reward + gamma * next_value * not_done - value;

            /* Advantage */
            float advantage;
            if (next_adv_row != NULL) {
                float prev_adv = next_adv_row[e];
                advantage = delta + gamma * gae_lambda * not_done * prev_adv;
            } else {
                advantage = delta;
            }

            adv_row[e] = advantage;
        }
    }
}

/**
 * @brief Compute GAE advantages with returns calculation
 *
 * Extended version that also computes returns (value targets) in addition
 * to advantages. Returns are computed as: return = advantage + value
 *
 * This is useful for algorithms that need both advantages (for policy loss)
 * and returns (for value function loss).
 *
 * @param rewards       Reward buffer [buffer_size, num_envs]
 * @param values        Value estimates [buffer_size, num_envs]
 * @param dones         Episode termination flags [buffer_size, num_envs]
 * @param advantages    Output advantage buffer [buffer_size, num_envs]
 * @param returns       Output returns buffer [buffer_size, num_envs]
 * @param buffer_size   Number of timesteps
 * @param num_envs      Number of environments
 * @param gamma         Discount factor
 * @param gae_lambda    GAE lambda
 * @param last_values   Bootstrap values [num_envs]
 */
void gae_compute_with_returns_neon(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    float* returns,
    size_t buffer_size,
    size_t num_envs,
    float gamma,
    float gae_lambda,
    const float* last_values
) {
    /* Early exit for empty buffer */
    if (buffer_size == 0 || num_envs == 0) {
        return;
    }

    /* Broadcast scalars to NEON vectors */
    const float32x4_t v_gamma = vdupq_n_f32(gamma);
    const float32x4_t v_one = vdupq_n_f32(1.0f);
    const float32x4_t v_gamma_lambda = vdupq_n_f32(gamma * gae_lambda);

    const size_t num_vec = num_envs / 4;
    const size_t remainder = num_envs % 4;

    /* Reverse iteration through timesteps */
    for (size_t t = buffer_size; t > 0; --t) {
        const size_t curr_idx = t - 1;
        const size_t row_offset = curr_idx * num_envs;

        const float* reward_row = rewards + row_offset;
        const float* value_row = values + row_offset;
        const float* done_row = dones + row_offset;
        float* adv_row = advantages + row_offset;
        float* ret_row = returns + row_offset;

        const float* next_value_row;
        const float* next_adv_row;

        if (t == buffer_size) {
            next_value_row = last_values;
            next_adv_row = NULL;
        } else {
            next_value_row = values + (t * num_envs);
            next_adv_row = advantages + (t * num_envs);
        }

        /* NEON vectorized loop */
        size_t e = 0;
        for (size_t vec_idx = 0; vec_idx < num_vec; ++vec_idx, e += 4) {
            float32x4_t v_reward = vld1q_f32(reward_row + e);
            float32x4_t v_value = vld1q_f32(value_row + e);
            float32x4_t v_done = vld1q_f32(done_row + e);
            float32x4_t v_next_value = vld1q_f32(next_value_row + e);

            float32x4_t v_not_done = vsubq_f32(v_one, v_done);

            /* TD residual */
            float32x4_t v_next_discounted = vmulq_f32(v_next_value, v_not_done);
            float32x4_t v_gamma_next = vmulq_f32(v_gamma, v_next_discounted);
            float32x4_t v_reward_plus = vaddq_f32(v_reward, v_gamma_next);
            float32x4_t v_delta = vsubq_f32(v_reward_plus, v_value);

            /* Advantage */
            float32x4_t v_advantage;
            if (next_adv_row != NULL) {
                float32x4_t v_prev_adv = vld1q_f32(next_adv_row + e);
                float32x4_t v_not_done_adv = vmulq_f32(v_not_done, v_prev_adv);
                v_advantage = vfmaq_f32(v_delta, v_gamma_lambda, v_not_done_adv);
            } else {
                v_advantage = v_delta;
            }

            /* Returns = advantage + value */
            float32x4_t v_returns = vaddq_f32(v_advantage, v_value);

            vst1q_f32(adv_row + e, v_advantage);
            vst1q_f32(ret_row + e, v_returns);
        }

        /* Scalar remainder */
        for (size_t r = 0; r < remainder; ++r, ++e) {
            float reward = reward_row[e];
            float value = value_row[e];
            float done = done_row[e];
            float next_value = next_value_row[e];
            float not_done = 1.0f - done;

            float delta = reward + gamma * next_value * not_done - value;

            float advantage;
            if (next_adv_row != NULL) {
                float prev_adv = next_adv_row[e];
                advantage = delta + gamma * gae_lambda * not_done * prev_adv;
            } else {
                advantage = delta;
            }

            adv_row[e] = advantage;
            ret_row[e] = advantage + value;
        }
    }
}

/**
 * @brief Normalize advantages in-place using NEON
 *
 * Computes: advantage = (advantage - mean) / (std + eps)
 *
 * Normalization helps stabilize policy gradient training by keeping
 * advantage magnitudes consistent across batches.
 *
 * @param advantages    Advantage buffer to normalize in-place [count]
 * @param count         Total number of elements (buffer_size * num_envs)
 * @param epsilon       Small constant for numerical stability (typically 1e-8)
 */
void gae_normalize_neon(
    float* advantages,
    size_t count,
    float epsilon
) {
    if (count == 0) {
        return;
    }

    const size_t num_vec = count / 4;
    const size_t remainder = count % 4;

    /* First pass: compute mean */
    float32x4_t v_sum = vdupq_n_f32(0.0f);
    float scalar_sum = 0.0f;

    size_t i = 0;
    for (size_t vec_idx = 0; vec_idx < num_vec; ++vec_idx, i += 4) {
        float32x4_t v_adv = vld1q_f32(advantages + i);
        v_sum = vaddq_f32(v_sum, v_adv);
    }

    /* Reduce NEON sum */
    float sum = vgetq_lane_f32(v_sum, 0) + vgetq_lane_f32(v_sum, 1) +
                vgetq_lane_f32(v_sum, 2) + vgetq_lane_f32(v_sum, 3);

    /* Add scalar remainder */
    for (size_t r = 0; r < remainder; ++r, ++i) {
        scalar_sum += advantages[i];
    }
    sum += scalar_sum;

    float mean = sum / (float)count;

    /* Second pass: compute variance */
    float32x4_t v_mean = vdupq_n_f32(mean);
    float32x4_t v_var_sum = vdupq_n_f32(0.0f);
    float scalar_var_sum = 0.0f;

    i = 0;
    for (size_t vec_idx = 0; vec_idx < num_vec; ++vec_idx, i += 4) {
        float32x4_t v_adv = vld1q_f32(advantages + i);
        float32x4_t v_diff = vsubq_f32(v_adv, v_mean);
        /* Accumulate diff^2 using fused multiply-add */
        v_var_sum = vfmaq_f32(v_var_sum, v_diff, v_diff);
    }

    /* Reduce variance sum */
    float var_sum = vgetq_lane_f32(v_var_sum, 0) + vgetq_lane_f32(v_var_sum, 1) +
                    vgetq_lane_f32(v_var_sum, 2) + vgetq_lane_f32(v_var_sum, 3);

    /* Add scalar remainder */
    for (size_t r = 0; r < remainder; ++r, ++i) {
        float diff = advantages[i] - mean;
        scalar_var_sum += diff * diff;
    }
    var_sum += scalar_var_sum;

    /* Standard deviation with epsilon for numerical stability */
    float variance = var_sum / (float)count;
    float std = sqrtf(variance) + epsilon;
    float inv_std = 1.0f / std;

    /* Third pass: normalize in place */
    float32x4_t v_inv_std = vdupq_n_f32(inv_std);

    i = 0;
    for (size_t vec_idx = 0; vec_idx < num_vec; ++vec_idx, i += 4) {
        float32x4_t v_adv = vld1q_f32(advantages + i);
        float32x4_t v_centered = vsubq_f32(v_adv, v_mean);
        float32x4_t v_normalized = vmulq_f32(v_centered, v_inv_std);
        vst1q_f32(advantages + i, v_normalized);
    }

    /* Scalar remainder */
    for (size_t r = 0; r < remainder; ++r, ++i) {
        advantages[i] = (advantages[i] - mean) * inv_std;
    }
}

/**
 * @brief Compute GAE with normalization in a single pass
 *
 * Combines GAE computation and normalization for better cache efficiency.
 * After computing all advantages, normalizes them before returning.
 *
 * @param rewards       Reward buffer [buffer_size, num_envs]
 * @param values        Value estimates [buffer_size, num_envs]
 * @param dones         Episode termination flags [buffer_size, num_envs]
 * @param advantages    Output normalized advantage buffer [buffer_size, num_envs]
 * @param buffer_size   Number of timesteps
 * @param num_envs      Number of environments
 * @param gamma         Discount factor
 * @param gae_lambda    GAE lambda
 * @param last_values   Bootstrap values [num_envs]
 * @param epsilon       Normalization epsilon (typically 1e-8)
 */
void gae_compute_normalized_neon(
    const float* rewards,
    const float* values,
    const float* dones,
    float* advantages,
    size_t buffer_size,
    size_t num_envs,
    float gamma,
    float gae_lambda,
    const float* last_values,
    float epsilon
) {
    /* First compute raw GAE */
    gae_compute_neon(
        rewards, values, dones, advantages,
        buffer_size, num_envs, gamma, gae_lambda, last_values
    );

    /* Then normalize */
    gae_normalize_neon(advantages, buffer_size * num_envs, epsilon);
}
