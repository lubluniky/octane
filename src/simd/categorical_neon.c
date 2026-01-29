// src/simd/categorical_neon.c
// Categorical distribution implementation using ARM NEON for Apple M4
// Part of RocketRL - High-performance reinforcement learning library

#include <arm_neon.h>
#include <stddef.h>
#include <stdint.h>
#include <math.h>
#include <float.h>

// ============================================================================
// RNG State for Gumbel sampling
// ============================================================================

typedef struct {
    uint64_t state[4];  // xoshiro256** state
} rng_state_t;

// xoshiro256** - fast, high-quality PRNG
static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static inline uint64_t rng_next(rng_state_t* rng) {
    const uint64_t result = rotl(rng->state[1] * 5, 7) * 9;
    const uint64_t t = rng->state[1] << 17;

    rng->state[2] ^= rng->state[0];
    rng->state[3] ^= rng->state[1];
    rng->state[1] ^= rng->state[2];
    rng->state[0] ^= rng->state[3];

    rng->state[2] ^= t;
    rng->state[3] = rotl(rng->state[3], 45);

    return result;
}

// Generate uniform float in (0, 1)
static inline float rng_uniform(rng_state_t* rng) {
    uint64_t x = rng_next(rng);
    // Use upper 23 bits for mantissa, ensuring (0, 1) range
    return (float)((x >> 40) + 1) * 0x1.0p-24f;
}

// Generate 4 uniform floats using NEON
static inline float32x4_t rng_uniform_x4(rng_state_t* rng) {
    float uniforms[4];
    uniforms[0] = rng_uniform(rng);
    uniforms[1] = rng_uniform(rng);
    uniforms[2] = rng_uniform(rng);
    uniforms[3] = rng_uniform(rng);
    return vld1q_f32(uniforms);
}

// ============================================================================
// Fast Exponential Approximation using NEON
// ============================================================================

// Fast exp approximation: valid for x in [-87, 88]
// Uses polynomial approximation: exp(x) ≈ 2^(x * log2(e))
// Accuracy: ~1e-5 relative error
static inline float32x4_t fast_exp_neon(float32x4_t x) {
    const float32x4_t log2e = vdupq_n_f32(1.4426950408889634f);   // log2(e)
    const float32x4_t ln2 = vdupq_n_f32(0.6931471805599453f);     // ln(2)
    const float32x4_t one = vdupq_n_f32(1.0f);
    const float32x4_t half = vdupq_n_f32(0.5f);

    // Clamp input to valid range
    const float32x4_t max_val = vdupq_n_f32(88.0f);
    const float32x4_t min_val = vdupq_n_f32(-87.0f);
    x = vminq_f32(x, max_val);
    x = vmaxq_f32(x, min_val);

    // x * log2(e) = n + f, where n is integer and f in [-0.5, 0.5]
    float32x4_t z = vmulq_f32(x, log2e);

    // Round to nearest integer
    float32x4_t n = vrndnq_f32(z);  // ARM NEON round to nearest
    float32x4_t f = vsubq_f32(z, n);

    // Convert back: f_ln = f * ln(2)
    float32x4_t f_ln = vmulq_f32(f, ln2);

    // Polynomial approximation for exp(f_ln) where f_ln in [-0.5*ln2, 0.5*ln2]
    // exp(x) ≈ 1 + x + x²/2 + x³/6 + x⁴/24 + x⁵/120
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.16666666666666666f);  // 1/6
    const float32x4_t c4 = vdupq_n_f32(0.041666666666666664f); // 1/24
    const float32x4_t c5 = vdupq_n_f32(0.008333333333333333f); // 1/120

    float32x4_t f2 = vmulq_f32(f_ln, f_ln);
    float32x4_t f3 = vmulq_f32(f2, f_ln);
    float32x4_t f4 = vmulq_f32(f2, f2);
    float32x4_t f5 = vmulq_f32(f4, f_ln);

    // Horner's method for better numerical stability
    float32x4_t exp_f = vaddq_f32(c1, f_ln);
    exp_f = vaddq_f32(exp_f, vmulq_f32(c2, f2));
    exp_f = vaddq_f32(exp_f, vmulq_f32(c3, f3));
    exp_f = vaddq_f32(exp_f, vmulq_f32(c4, f4));
    exp_f = vaddq_f32(exp_f, vmulq_f32(c5, f5));

    // Reconstruct: exp(x) = exp(f_ln) * 2^n
    // Use integer manipulation for 2^n
    int32x4_t n_int = vcvtq_s32_f32(n);
    n_int = vaddq_s32(n_int, vdupq_n_s32(127));  // Add IEEE754 exponent bias
    n_int = vshlq_n_s32(n_int, 23);               // Shift to exponent position
    float32x4_t pow2n = vreinterpretq_f32_s32(n_int);

    return vmulq_f32(exp_f, pow2n);
}

// Fast natural log approximation
static inline float32x4_t fast_log_neon(float32x4_t x) {
    const float32x4_t ln2 = vdupq_n_f32(0.6931471805599453f);
    const float32x4_t one = vdupq_n_f32(1.0f);

    // Extract exponent and mantissa
    int32x4_t xi = vreinterpretq_s32_f32(x);
    int32x4_t exp_bits = vshrq_n_s32(xi, 23);
    exp_bits = vsubq_s32(exp_bits, vdupq_n_s32(127));
    float32x4_t e = vcvtq_f32_s32(exp_bits);

    // Normalize mantissa to [1, 2)
    int32x4_t mantissa_bits = vandq_s32(xi, vdupq_n_s32(0x007FFFFF));
    mantissa_bits = vorrq_s32(mantissa_bits, vdupq_n_s32(0x3F800000));
    float32x4_t m = vreinterpretq_f32_s32(mantissa_bits);

    // log(m) where m in [1, 2) using polynomial
    // log(m) ≈ (m-1) - (m-1)²/2 + (m-1)³/3 - ...
    float32x4_t f = vsubq_f32(m, one);
    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t f3 = vmulq_f32(f2, f);
    float32x4_t f4 = vmulq_f32(f2, f2);

    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(-0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.33333333f);
    const float32x4_t c4 = vdupq_n_f32(-0.25f);

    float32x4_t log_m = vmulq_f32(c1, f);
    log_m = vaddq_f32(log_m, vmulq_f32(c2, f2));
    log_m = vaddq_f32(log_m, vmulq_f32(c3, f3));
    log_m = vaddq_f32(log_m, vmulq_f32(c4, f4));

    // log(x) = log(m) + e * ln(2)
    return vaddq_f32(log_m, vmulq_f32(e, ln2));
}

// ============================================================================
// Horizontal Reduction Operations
// ============================================================================

// Horizontal maximum of 4 floats
static inline float hmax_f32(float32x4_t v) {
    return vmaxvq_f32(v);
}

// Horizontal sum of 4 floats
static inline float hsum_f32(float32x4_t v) {
    return vaddvq_f32(v);
}

// Find maximum value in array
static inline float find_max_neon(const float* data, size_t n) {
    float32x4_t max_vec = vdupq_n_f32(-FLT_MAX);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        max_vec = vmaxq_f32(max_vec, v);
    }

    float max_val = hmax_f32(max_vec);

    // Handle remaining elements
    for (; i < n; i++) {
        if (data[i] > max_val) {
            max_val = data[i];
        }
    }

    return max_val;
}

// Sum array elements
static inline float sum_neon(const float* data, size_t n) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t v = vld1q_f32(data + i);
        sum_vec = vaddq_f32(sum_vec, v);
    }

    float sum = hsum_f32(sum_vec);

    // Handle remaining elements
    for (; i < n; i++) {
        sum += data[i];
    }

    return sum;
}

// ============================================================================
// Softmax Implementation
// ============================================================================

// Softmax: exp(x - max) / sum(exp(x - max))
// Numerically stable implementation
void softmax_neon(
    const float* logits,    // [batch_size, num_actions]
    float* probs,           // [batch_size, num_actions]
    size_t batch_size,
    size_t num_actions
) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* row_logits = logits + b * num_actions;
        float* row_probs = probs + b * num_actions;

        // Step 1: Find max for numerical stability
        float max_val = find_max_neon(row_logits, num_actions);
        float32x4_t max_vec = vdupq_n_f32(max_val);

        // Step 2: Compute exp(x - max) and accumulate sum
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        size_t i = 0;
        for (; i + 4 <= num_actions; i += 4) {
            float32x4_t x = vld1q_f32(row_logits + i);
            float32x4_t x_shifted = vsubq_f32(x, max_vec);
            float32x4_t exp_x = fast_exp_neon(x_shifted);
            vst1q_f32(row_probs + i, exp_x);
            sum_vec = vaddq_f32(sum_vec, exp_x);
        }

        // Handle remaining elements
        float sum = hsum_f32(sum_vec);
        for (; i < num_actions; i++) {
            float exp_val = expf(row_logits[i] - max_val);
            row_probs[i] = exp_val;
            sum += exp_val;
        }

        // Step 3: Normalize by sum
        float inv_sum = 1.0f / sum;
        float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);

        i = 0;
        for (; i + 4 <= num_actions; i += 4) {
            float32x4_t p = vld1q_f32(row_probs + i);
            p = vmulq_f32(p, inv_sum_vec);
            vst1q_f32(row_probs + i, p);
        }

        for (; i < num_actions; i++) {
            row_probs[i] *= inv_sum;
        }
    }
}

// ============================================================================
// Log-Softmax Implementation
// ============================================================================

// Log-softmax: x - max - log(sum(exp(x - max)))
// More numerically stable than log(softmax(x))
void log_softmax_neon(
    const float* logits,
    float* log_probs,
    size_t batch_size,
    size_t num_actions
) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* row_logits = logits + b * num_actions;
        float* row_log_probs = log_probs + b * num_actions;

        // Step 1: Find max for numerical stability
        float max_val = find_max_neon(row_logits, num_actions);
        float32x4_t max_vec = vdupq_n_f32(max_val);

        // Step 2: Compute sum(exp(x - max))
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        size_t i = 0;
        for (; i + 4 <= num_actions; i += 4) {
            float32x4_t x = vld1q_f32(row_logits + i);
            float32x4_t x_shifted = vsubq_f32(x, max_vec);
            float32x4_t exp_x = fast_exp_neon(x_shifted);
            sum_vec = vaddq_f32(sum_vec, exp_x);
        }

        float sum = hsum_f32(sum_vec);
        for (; i < num_actions; i++) {
            sum += expf(row_logits[i] - max_val);
        }

        // Step 3: Compute log(sum) and subtract from shifted logits
        float log_sum = logf(sum);
        float offset = max_val + log_sum;
        float32x4_t offset_vec = vdupq_n_f32(offset);

        i = 0;
        for (; i + 4 <= num_actions; i += 4) {
            float32x4_t x = vld1q_f32(row_logits + i);
            float32x4_t log_p = vsubq_f32(x, offset_vec);
            vst1q_f32(row_log_probs + i, log_p);
        }

        for (; i < num_actions; i++) {
            row_log_probs[i] = row_logits[i] - offset;
        }
    }
}

// ============================================================================
// Categorical Sampling using Gumbel-Max Trick
// ============================================================================

// Generate Gumbel noise: -log(-log(uniform))
static inline float32x4_t gumbel_noise_neon(rng_state_t* rng) {
    float32x4_t u = rng_uniform_x4(rng);

    // Clamp to avoid log(0)
    const float32x4_t eps = vdupq_n_f32(1e-20f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    u = vmaxq_f32(u, eps);
    u = vminq_f32(u, vsubq_f32(one, eps));

    // -log(-log(u))
    float32x4_t neg_log_u = vnegq_f32(fast_log_neon(u));
    float32x4_t gumbel = vnegq_f32(fast_log_neon(neg_log_u));

    return gumbel;
}

// Gumbel-max sampling: argmax(logits + gumbel_noise)
// This is equivalent to sampling from categorical distribution
// but is more vectorization-friendly
void categorical_sample_gumbel_neon(
    const float* logits,
    uint32_t* actions,
    size_t batch_size,
    size_t num_actions,
    rng_state_t* rng
) {
    // Temporary buffer for perturbed logits
    // For large num_actions, consider heap allocation
    float perturbed[4];

    for (size_t b = 0; b < batch_size; b++) {
        const float* row_logits = logits + b * num_actions;

        float max_val = -FLT_MAX;
        uint32_t max_idx = 0;

        size_t i = 0;

        // Process 4 elements at a time
        for (; i + 4 <= num_actions; i += 4) {
            float32x4_t x = vld1q_f32(row_logits + i);
            float32x4_t gumbel = gumbel_noise_neon(rng);
            float32x4_t perturbed_vec = vaddq_f32(x, gumbel);

            // Store to find argmax
            vst1q_f32(perturbed, perturbed_vec);

            for (int j = 0; j < 4; j++) {
                if (perturbed[j] > max_val) {
                    max_val = perturbed[j];
                    max_idx = (uint32_t)(i + j);
                }
            }
        }

        // Handle remaining elements
        for (; i < num_actions; i++) {
            float u = rng_uniform(rng);
            u = fmaxf(u, 1e-20f);
            u = fminf(u, 1.0f - 1e-20f);
            float gumbel = -logf(-logf(u));
            float perturbed_val = row_logits[i] + gumbel;

            if (perturbed_val > max_val) {
                max_val = perturbed_val;
                max_idx = (uint32_t)i;
            }
        }

        actions[b] = max_idx;
    }
}

// ============================================================================
// Batch Sampling with Inverse CDF (Alternative Method)
// ============================================================================

// Standard categorical sampling using inverse CDF
// More accurate but less vectorizable
void categorical_sample_icdf_neon(
    const float* probs,     // Already normalized probabilities
    uint32_t* actions,
    size_t batch_size,
    size_t num_actions,
    rng_state_t* rng
) {
    for (size_t b = 0; b < batch_size; b++) {
        const float* row_probs = probs + b * num_actions;
        float u = rng_uniform(rng);

        // Binary search would be O(log n) but linear scan
        // is often faster for small num_actions due to cache
        float cumsum = 0.0f;
        uint32_t action = (uint32_t)(num_actions - 1);  // Default to last action

        // Use NEON for cumulative sum computation
        float32x4_t cumsum_vec = vdupq_n_f32(0.0f);
        float32x4_t u_vec = vdupq_n_f32(u);

        size_t i = 0;
        for (; i + 4 <= num_actions; i += 4) {
            float32x4_t p = vld1q_f32(row_probs + i);

            // Prefix sum within vector
            float p0 = vgetq_lane_f32(p, 0);
            float p1 = vgetq_lane_f32(p, 1);
            float p2 = vgetq_lane_f32(p, 2);
            float p3 = vgetq_lane_f32(p, 3);

            float c0 = cumsum + p0;
            float c1 = c0 + p1;
            float c2 = c1 + p2;
            float c3 = c2 + p3;

            // Check each cumsum against u
            if (cumsum < u && u <= c0) {
                action = (uint32_t)i;
                goto done;
            }
            if (c0 < u && u <= c1) {
                action = (uint32_t)(i + 1);
                goto done;
            }
            if (c1 < u && u <= c2) {
                action = (uint32_t)(i + 2);
                goto done;
            }
            if (c2 < u && u <= c3) {
                action = (uint32_t)(i + 3);
                goto done;
            }

            cumsum = c3;
        }

        // Handle remaining elements
        for (; i < num_actions; i++) {
            cumsum += row_probs[i];
            if (u <= cumsum) {
                action = (uint32_t)i;
                break;
            }
        }

    done:
        actions[b] = action;
    }
}

// ============================================================================
// Entropy Computation
// ============================================================================

// Compute entropy: -sum(p * log(p))
void categorical_entropy_neon(
    const float* probs,
    float* entropy,
    size_t batch_size,
    size_t num_actions
) {
    const float32x4_t eps = vdupq_n_f32(1e-10f);
    const float32x4_t neg_one = vdupq_n_f32(-1.0f);

    for (size_t b = 0; b < batch_size; b++) {
        const float* row_probs = probs + b * num_actions;
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        size_t i = 0;
        for (; i + 4 <= num_actions; i += 4) {
            float32x4_t p = vld1q_f32(row_probs + i);
            // Add epsilon to avoid log(0)
            float32x4_t p_safe = vmaxq_f32(p, eps);
            float32x4_t log_p = fast_log_neon(p_safe);
            float32x4_t contrib = vmulq_f32(p, log_p);
            sum_vec = vaddq_f32(sum_vec, contrib);
        }

        float sum = hsum_f32(sum_vec);

        // Handle remaining elements
        for (; i < num_actions; i++) {
            float p = row_probs[i];
            if (p > 1e-10f) {
                sum += p * logf(p);
            }
        }

        entropy[b] = -sum;
    }
}

// ============================================================================
// Log Probability of Actions
// ============================================================================

// Compute log probability of selected actions
void categorical_log_prob_neon(
    const float* log_probs,     // [batch_size, num_actions] from log_softmax
    const uint32_t* actions,    // [batch_size]
    float* action_log_probs,    // [batch_size]
    size_t batch_size,
    size_t num_actions
) {
    // This is a gather operation, limited vectorization possible
    // Process 4 batches at a time if possible

    size_t b = 0;
    for (; b + 4 <= batch_size; b += 4) {
        action_log_probs[b] = log_probs[b * num_actions + actions[b]];
        action_log_probs[b + 1] = log_probs[(b + 1) * num_actions + actions[b + 1]];
        action_log_probs[b + 2] = log_probs[(b + 2) * num_actions + actions[b + 2]];
        action_log_probs[b + 3] = log_probs[(b + 3) * num_actions + actions[b + 3]];
    }

    for (; b < batch_size; b++) {
        action_log_probs[b] = log_probs[b * num_actions + actions[b]];
    }
}

// ============================================================================
// KL Divergence
// ============================================================================

// Compute KL divergence: sum(p * log(p / q))
void categorical_kl_divergence_neon(
    const float* p_probs,
    const float* q_probs,
    float* kl_div,
    size_t batch_size,
    size_t num_actions
) {
    const float32x4_t eps = vdupq_n_f32(1e-10f);

    for (size_t b = 0; b < batch_size; b++) {
        const float* p_row = p_probs + b * num_actions;
        const float* q_row = q_probs + b * num_actions;
        float32x4_t sum_vec = vdupq_n_f32(0.0f);

        size_t i = 0;
        for (; i + 4 <= num_actions; i += 4) {
            float32x4_t p = vld1q_f32(p_row + i);
            float32x4_t q = vld1q_f32(q_row + i);

            // Add epsilon for numerical stability
            float32x4_t p_safe = vmaxq_f32(p, eps);
            float32x4_t q_safe = vmaxq_f32(q, eps);

            // log(p / q) = log(p) - log(q)
            float32x4_t log_p = fast_log_neon(p_safe);
            float32x4_t log_q = fast_log_neon(q_safe);
            float32x4_t log_ratio = vsubq_f32(log_p, log_q);

            // p * log(p / q)
            float32x4_t contrib = vmulq_f32(p, log_ratio);
            sum_vec = vaddq_f32(sum_vec, contrib);
        }

        float sum = hsum_f32(sum_vec);

        // Handle remaining elements
        for (; i < num_actions; i++) {
            float p = p_row[i];
            float q = q_row[i];
            if (p > 1e-10f) {
                sum += p * logf(fmaxf(p, 1e-10f) / fmaxf(q, 1e-10f));
            }
        }

        kl_div[b] = sum;
    }
}

// ============================================================================
// RNG Initialization
// ============================================================================

// Initialize RNG state with seed
void rng_init(rng_state_t* rng, uint64_t seed) {
    // SplitMix64 for seeding
    uint64_t z = seed;
    for (int i = 0; i < 4; i++) {
        z += 0x9e3779b97f4a7c15;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        rng->state[i] = z ^ (z >> 31);
    }
}

// ============================================================================
// Batch Operations for Training
// ============================================================================

// Combined forward pass: logits -> log_probs, sample actions, compute entropy
void categorical_forward_neon(
    const float* logits,
    float* log_probs,
    uint32_t* actions,
    float* entropy,
    size_t batch_size,
    size_t num_actions,
    rng_state_t* rng,
    int deterministic  // If true, use argmax instead of sampling
) {
    // Temporary buffer for probabilities (needed for entropy)
    // For production, this should be passed in or use arena allocation
    float* probs = (float*)__builtin_alloca(batch_size * num_actions * sizeof(float));

    // Compute softmax and log-softmax
    softmax_neon(logits, probs, batch_size, num_actions);
    log_softmax_neon(logits, log_probs, batch_size, num_actions);

    // Sample or argmax
    if (deterministic) {
        // Argmax
        for (size_t b = 0; b < batch_size; b++) {
            const float* row = logits + b * num_actions;
            float max_val = -FLT_MAX;
            uint32_t max_idx = 0;

            for (size_t i = 0; i < num_actions; i++) {
                if (row[i] > max_val) {
                    max_val = row[i];
                    max_idx = (uint32_t)i;
                }
            }
            actions[b] = max_idx;
        }
    } else {
        categorical_sample_gumbel_neon(logits, actions, batch_size, num_actions, rng);
    }

    // Compute entropy
    categorical_entropy_neon(probs, entropy, batch_size, num_actions);
}
