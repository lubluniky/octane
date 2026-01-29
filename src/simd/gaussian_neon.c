/**
 * @file gaussian_neon.c
 * @brief High-performance Gaussian sampling with ARM NEON for Apple M4
 *
 * Implementation of vectorized Gaussian sampling using Box-Muller transform
 * optimized for ARM NEON SIMD instructions on Apple Silicon (M1/M2/M3/M4).
 *
 * Techniques used:
 * - Vectorized xoroshiro128+ PRNG (4 parallel streams)
 * - Fast polynomial approximations for log/exp/sin/cos
 * - Newton-Raphson refinement for rsqrt
 * - Branchless NEON operations throughout
 *
 * @copyright 2024 RocketRL Project
 * @license GPL-2.0
 */

#include "gaussian_neon.h"
#include <arm_neon.h>
#include <math.h>

/* ============================================================================
 * Constants
 * ============================================================================ */

/* Mathematical constants */
static const float TWO_PI = 6.283185307179586476925286766559f;
static const float LOG_2PI = 1.8378770664093454835606594728112f;
static const float NEG_TWO = -2.0f;

/* Epsilon to avoid log(0) */
static const float EPSILON = 1.175494351e-38f;  /* FLT_MIN */

/* Coefficients for log2 approximation (degree 5 polynomial) */
static const float LOG2_C0 = -1.7417939f;
static const float LOG2_C1 = 2.8212026f;
static const float LOG2_C2 = -1.4699568f;
static const float LOG2_C3 = 0.44717955f;
static const float LOG2_C4 = -0.056570851f;
static const float LN2 = 0.69314718055994530942f;

/* Coefficients for sin approximation (Bhaskara I) - high accuracy */
static const float SIN_A = 16.0f;
static const float SIN_B = 5.0f;
static const float SIN_C = 4.0f;

/* ============================================================================
 * SplitMix64 for seed expansion
 * ============================================================================ */

static inline uint64_t splitmix64(uint64_t* state) {
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

/* ============================================================================
 * RNG Initialization
 * ============================================================================ */

void init_rng_state(uint64_t seed, uint64_t* state) {
    /* Initialize 4 independent xoroshiro128+ states using SplitMix64 */
    uint64_t sm_state = seed;

    for (int i = 0; i < 4; i++) {
        state[i] = splitmix64(&sm_state);       /* s0 for stream i */
        state[i + 4] = splitmix64(&sm_state);   /* s1 for stream i */
    }
}

/* ============================================================================
 * Vectorized xoroshiro128+ RNG
 * ============================================================================ */

/**
 * @brief Generate 4 parallel uint64 random numbers using xoroshiro128+
 *
 * State layout in rng_state[8]:
 *   [0..3] = s0 for 4 streams
 *   [4..7] = s1 for 4 streams
 */
static inline void xoroshiro128plus_next_4x(uint64_t* rng_state,
                                             uint64x2_t* out0,
                                             uint64x2_t* out1) {
    /* Load state */
    uint64x2_t s0_lo = vld1q_u64(&rng_state[0]);  /* s0[0], s0[1] */
    uint64x2_t s0_hi = vld1q_u64(&rng_state[2]);  /* s0[2], s0[3] */
    uint64x2_t s1_lo = vld1q_u64(&rng_state[4]);  /* s1[0], s1[1] */
    uint64x2_t s1_hi = vld1q_u64(&rng_state[6]);  /* s1[2], s1[3] */

    /* result = s0 + s1 */
    *out0 = vaddq_u64(s0_lo, s1_lo);
    *out1 = vaddq_u64(s0_hi, s1_hi);

    /* s1 ^= s0 */
    s1_lo = veorq_u64(s1_lo, s0_lo);
    s1_hi = veorq_u64(s1_hi, s0_hi);

    /* s0 = rotl(s0, 24) ^ s1 ^ (s1 << 16) */
    /* rotl(x, 24) = (x << 24) | (x >> 40) */
    uint64x2_t rotl24_lo = vorrq_u64(vshlq_n_u64(s0_lo, 24),
                                      vshrq_n_u64(s0_lo, 40));
    uint64x2_t rotl24_hi = vorrq_u64(vshlq_n_u64(s0_hi, 24),
                                      vshrq_n_u64(s0_hi, 40));

    s0_lo = veorq_u64(veorq_u64(rotl24_lo, s1_lo), vshlq_n_u64(s1_lo, 16));
    s0_hi = veorq_u64(veorq_u64(rotl24_hi, s1_hi), vshlq_n_u64(s1_hi, 16));

    /* s1 = rotl(s1, 37) */
    /* rotl(x, 37) = (x << 37) | (x >> 27) */
    s1_lo = vorrq_u64(vshlq_n_u64(s1_lo, 37), vshrq_n_u64(s1_lo, 27));
    s1_hi = vorrq_u64(vshlq_n_u64(s1_hi, 37), vshrq_n_u64(s1_hi, 27));

    /* Store updated state */
    vst1q_u64(&rng_state[0], s0_lo);
    vst1q_u64(&rng_state[2], s0_hi);
    vst1q_u64(&rng_state[4], s1_lo);
    vst1q_u64(&rng_state[6], s1_hi);
}

/**
 * @brief Convert 4 uint64 to 4 floats in (0, 1)
 *
 * Uses the upper 23 bits of each uint64 to create floats.
 * Result is in range (0, 1) exclusive of 0.
 */
static inline float32x4_t uint64_to_uniform_f32(uint64x2_t u64_lo,
                                                  uint64x2_t u64_hi) {
    /* Extract upper 32 bits from each uint64 */
    uint32x2_t u32_lo = vget_high_u32(vreinterpretq_u32_u64(u64_lo));
    uint32x2_t u32_hi = vget_high_u32(vreinterpretq_u32_u64(u64_hi));
    uint32x4_t u32 = vcombine_u32(u32_lo, u32_hi);

    /* Shift right by 9 bits to get 23-bit mantissa portion */
    u32 = vshrq_n_u32(u32, 9);

    /* OR with 1.0f exponent (0x3f800000) to get float in [1, 2) */
    uint32x4_t one_exp = vdupq_n_u32(0x3f800000);
    u32 = vorrq_u32(u32, one_exp);

    /* Convert bit pattern to float and subtract 1 to get [0, 1) */
    float32x4_t f = vreinterpretq_f32_u32(u32);
    f = vsubq_f32(f, vdupq_n_f32(1.0f));

    /* Add small epsilon to avoid exactly 0 */
    f = vaddq_f32(f, vdupq_n_f32(EPSILON));

    return f;
}

/* ============================================================================
 * Fast Math Approximations (NEON vectorized)
 * ============================================================================ */

/**
 * @brief Fast natural logarithm approximation using polynomial
 *
 * Uses the identity: ln(x) = ln(2) * log2(x)
 * log2(x) is approximated with a polynomial on the mantissa.
 * Max relative error: ~1e-4
 */
static inline float32x4_t fast_log_neon(float32x4_t x) {
    /* Extract exponent: floor(log2(x)) */
    int32x4_t xi = vreinterpretq_s32_f32(x);
    int32x4_t exp_bits = vshrq_n_s32(xi, 23);
    exp_bits = vsubq_s32(exp_bits, vdupq_n_s32(127));
    float32x4_t exp_f = vcvtq_f32_s32(exp_bits);

    /* Extract mantissa: set exponent to 127 (value in [1, 2)) */
    int32x4_t mantissa_mask = vdupq_n_s32(0x007fffff);
    int32x4_t exp_one = vdupq_n_s32(0x3f800000);
    int32x4_t mantissa_bits = vorrq_s32(vandq_s32(xi, mantissa_mask), exp_one);
    float32x4_t m = vreinterpretq_f32_s32(mantissa_bits);

    /* Polynomial approximation of log2(m) for m in [1, 2)
     * log2(m) = c0 + m*(c1 + m*(c2 + m*(c3 + m*c4))) */
    float32x4_t p = vdupq_n_f32(LOG2_C4);
    p = vfmaq_f32(vdupq_n_f32(LOG2_C3), m, p);
    p = vfmaq_f32(vdupq_n_f32(LOG2_C2), m, p);
    p = vfmaq_f32(vdupq_n_f32(LOG2_C1), m, p);
    p = vfmaq_f32(vdupq_n_f32(LOG2_C0), m, p);

    /* log2(x) = exp_f + log2(m) */
    float32x4_t log2_x = vaddq_f32(exp_f, p);

    /* ln(x) = ln(2) * log2(x) */
    return vmulq_f32(log2_x, vdupq_n_f32(LN2));
}

/**
 * @brief Fast reciprocal square root with Newton-Raphson refinement
 *
 * Uses NEON vrsqrteq_f32 as initial estimate, then 2 N-R iterations
 * for high precision: x_new = x * (3 - a * x^2) / 2
 */
static inline float32x4_t fast_rsqrt_neon(float32x4_t x) {
    float32x4_t est = vrsqrteq_f32(x);

    /* Newton-Raphson iteration 1 */
    float32x4_t est2 = vmulq_f32(est, est);
    float32x4_t step = vrsqrtsq_f32(x, est2);
    est = vmulq_f32(est, step);

    /* Newton-Raphson iteration 2 for higher precision */
    est2 = vmulq_f32(est, est);
    step = vrsqrtsq_f32(x, est2);
    est = vmulq_f32(est, step);

    return est;
}

/**
 * @brief Fast square root using rsqrt: sqrt(x) = x * rsqrt(x)
 */
static inline float32x4_t fast_sqrt_neon(float32x4_t x) {
    /* Handle zero specially to avoid inf * 0 = nan */
    uint32x4_t zero_mask = vceqzq_f32(x);

    float32x4_t rsqrt = fast_rsqrt_neon(x);
    float32x4_t result = vmulq_f32(x, rsqrt);

    /* If x == 0, result should be 0 */
    result = vbslq_f32(zero_mask, vdupq_n_f32(0.0f), result);

    return result;
}

/**
 * @brief Fast sine approximation using Bhaskara I formula
 *
 * Valid for x in [0, pi]
 * sin(x) approx (16x(pi-x)) / (5*pi^2 - 4x(pi-x))
 * Extended to full range using symmetry.
 */
static inline float32x4_t fast_sin_neon(float32x4_t x) {
    /* Reduce x to [0, 2*pi] */
    float32x4_t two_pi = vdupq_n_f32(TWO_PI);
    float32x4_t inv_two_pi = vdupq_n_f32(1.0f / TWO_PI);

    /* x = x - floor(x / 2pi) * 2pi */
    float32x4_t n = vmulq_f32(x, inv_two_pi);
    n = vrndmq_f32(n);  /* floor */
    x = vfmsq_f32(x, n, two_pi);

    /* For x in [pi, 2*pi], use sin(x) = -sin(x - pi) */
    float32x4_t pi = vdupq_n_f32(3.14159265358979323846f);
    uint32x4_t gt_pi = vcgtq_f32(x, pi);
    float32x4_t x_adj = vbslq_f32(gt_pi, vsubq_f32(x, pi), x);

    /* Bhaskara I formula: sin(x) = 16x(pi-x) / (5pi^2 - 4x(pi-x)) */
    float32x4_t pi_minus_x = vsubq_f32(pi, x_adj);
    float32x4_t x_pi_x = vmulq_f32(x_adj, pi_minus_x);

    float32x4_t numer = vmulq_f32(vdupq_n_f32(SIN_A), x_pi_x);
    float32x4_t denom = vfmsq_f32(vdupq_n_f32(SIN_B * 9.8696044f), /* 5*pi^2 */
                                   vdupq_n_f32(SIN_C), x_pi_x);

    float32x4_t sin_val = vdivq_f32(numer, denom);

    /* Negate if x was in [pi, 2*pi] */
    sin_val = vbslq_f32(gt_pi, vnegq_f32(sin_val), sin_val);

    return sin_val;
}

/**
 * @brief Fast cosine: cos(x) = sin(x + pi/2)
 */
static inline float32x4_t fast_cos_neon(float32x4_t x) {
    float32x4_t half_pi = vdupq_n_f32(1.5707963267948966f);
    return fast_sin_neon(vaddq_f32(x, half_pi));
}

/* ============================================================================
 * Box-Muller Transform (NEON vectorized)
 * ============================================================================ */

/**
 * @brief Generate 8 standard normal samples using Box-Muller
 *
 * Box-Muller transform:
 *   z0 = sqrt(-2 * ln(u1)) * cos(2 * pi * u2)
 *   z1 = sqrt(-2 * ln(u1)) * sin(2 * pi * u2)
 *
 * Generates 2 normal samples per uniform pair, so 4 uniforms -> 4 normals.
 * Called twice to generate 8 samples.
 */
static inline void box_muller_neon(uint64_t* rng_state,
                                    float32x4_t* out0,
                                    float32x4_t* out1) {
    /* Generate 8 uniform randoms (for 2 sets of Box-Muller) */
    uint64x2_t r0_lo, r0_hi, r1_lo, r1_hi;

    xoroshiro128plus_next_4x(rng_state, &r0_lo, &r0_hi);
    float32x4_t u1 = uint64_to_uniform_f32(r0_lo, r0_hi);

    xoroshiro128plus_next_4x(rng_state, &r1_lo, &r1_hi);
    float32x4_t u2 = uint64_to_uniform_f32(r1_lo, r1_hi);

    /* r = sqrt(-2 * ln(u1)) */
    float32x4_t log_u1 = fast_log_neon(u1);
    float32x4_t neg_two_log = vmulq_f32(log_u1, vdupq_n_f32(NEG_TWO));
    float32x4_t r = fast_sqrt_neon(neg_two_log);

    /* theta = 2 * pi * u2 */
    float32x4_t theta = vmulq_f32(u2, vdupq_n_f32(TWO_PI));

    /* z0 = r * cos(theta), z1 = r * sin(theta) */
    *out0 = vmulq_f32(r, fast_cos_neon(theta));
    *out1 = vmulq_f32(r, fast_sin_neon(theta));
}

/* ============================================================================
 * Public API Implementation
 * ============================================================================ */

void sample_standard_normal_neon(float* output, size_t count, uint64_t* rng_state) {
    size_t i = 0;

    /* Process 8 samples at a time (2 NEON vectors) */
    while (i + 8 <= count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);

        vst1q_f32(&output[i], z0);
        vst1q_f32(&output[i + 4], z1);
        i += 8;
    }

    /* Process 4 samples */
    if (i + 4 <= count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);

        vst1q_f32(&output[i], z0);
        i += 4;

        /* Store remaining from z1 if needed */
        size_t remaining = count - i;
        if (remaining > 0) {
            float temp[4];
            vst1q_f32(temp, z1);
            for (size_t j = 0; j < remaining; j++) {
                output[i + j] = temp[j];
            }
            i = count;
        }
    }

    /* Handle tail (< 4 samples) */
    if (i < count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);

        float temp[4];
        vst1q_f32(temp, z0);

        size_t remaining = count - i;
        for (size_t j = 0; j < remaining; j++) {
            output[i + j] = temp[j];
        }
    }
}

void sample_gaussian_neon(const float* mean, const float* std,
                          float* output, size_t count, uint64_t* rng_state) {
    size_t i = 0;

    /* Process 8 samples at a time */
    while (i + 8 <= count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);

        /* Load mean and std */
        float32x4_t m0 = vld1q_f32(&mean[i]);
        float32x4_t m1 = vld1q_f32(&mean[i + 4]);
        float32x4_t s0 = vld1q_f32(&std[i]);
        float32x4_t s1 = vld1q_f32(&std[i + 4]);

        /* output = mean + std * z */
        float32x4_t out0 = vfmaq_f32(m0, s0, z0);
        float32x4_t out1 = vfmaq_f32(m1, s1, z1);

        vst1q_f32(&output[i], out0);
        vst1q_f32(&output[i + 4], out1);
        i += 8;
    }

    /* Process 4 samples */
    if (i + 4 <= count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);

        float32x4_t m0 = vld1q_f32(&mean[i]);
        float32x4_t s0 = vld1q_f32(&std[i]);
        float32x4_t out0 = vfmaq_f32(m0, s0, z0);
        vst1q_f32(&output[i], out0);
        i += 4;

        /* Store remaining from z1 */
        size_t remaining = count - i;
        if (remaining > 0) {
            float z_temp[4];
            vst1q_f32(z_temp, z1);
            for (size_t j = 0; j < remaining; j++) {
                output[i + j] = mean[i + j] + std[i + j] * z_temp[j];
            }
            i = count;
        }
    }

    /* Handle tail */
    if (i < count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);

        float z_temp[4];
        vst1q_f32(z_temp, z0);

        size_t remaining = count - i;
        for (size_t j = 0; j < remaining; j++) {
            output[i + j] = mean[i + j] + std[i + j] * z_temp[j];
        }
    }
}

void sample_gaussian_batch_neon(const float* mean, const float* std,
                                float* output, size_t batch_size,
                                size_t action_dim, uint64_t* rng_state) {
    /* Batch sampling is equivalent to flat sampling for row-major layout */
    size_t total = batch_size * action_dim;
    sample_gaussian_neon(mean, std, output, total, rng_state);
}

void sample_gaussian_with_logprob_neon(const float* mean, const float* std,
                                        float* output, float* log_prob,
                                        size_t count, uint64_t* rng_state) {
    /* Constants for log probability calculation */
    /* log_prob = -0.5 * (log(2*pi) + 2*log(std) + z^2) */
    float32x4_t neg_half = vdupq_n_f32(-0.5f);
    float32x4_t log_2pi = vdupq_n_f32(LOG_2PI);
    float32x4_t two = vdupq_n_f32(2.0f);

    size_t i = 0;

    /* Process 8 samples at a time */
    while (i + 8 <= count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);

        /* Load mean and std */
        float32x4_t m0 = vld1q_f32(&mean[i]);
        float32x4_t m1 = vld1q_f32(&mean[i + 4]);
        float32x4_t s0 = vld1q_f32(&std[i]);
        float32x4_t s1 = vld1q_f32(&std[i + 4]);

        /* output = mean + std * z */
        float32x4_t out0 = vfmaq_f32(m0, s0, z0);
        float32x4_t out1 = vfmaq_f32(m1, s1, z1);
        vst1q_f32(&output[i], out0);
        vst1q_f32(&output[i + 4], out1);

        /* log_prob = -0.5 * (log(2*pi) + 2*log(std) + z^2) */
        float32x4_t log_s0 = fast_log_neon(s0);
        float32x4_t log_s1 = fast_log_neon(s1);
        float32x4_t z0_sq = vmulq_f32(z0, z0);
        float32x4_t z1_sq = vmulq_f32(z1, z1);

        /* log_2pi + 2*log_s + z^2 */
        float32x4_t lp0 = vfmaq_f32(log_2pi, two, log_s0);
        float32x4_t lp1 = vfmaq_f32(log_2pi, two, log_s1);
        lp0 = vaddq_f32(lp0, z0_sq);
        lp1 = vaddq_f32(lp1, z1_sq);

        /* Multiply by -0.5 */
        lp0 = vmulq_f32(neg_half, lp0);
        lp1 = vmulq_f32(neg_half, lp1);

        vst1q_f32(&log_prob[i], lp0);
        vst1q_f32(&log_prob[i + 4], lp1);

        i += 8;
    }

    /* Process 4 samples */
    if (i + 4 <= count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);

        float32x4_t m0 = vld1q_f32(&mean[i]);
        float32x4_t s0 = vld1q_f32(&std[i]);
        float32x4_t out0 = vfmaq_f32(m0, s0, z0);
        vst1q_f32(&output[i], out0);

        float32x4_t log_s0 = fast_log_neon(s0);
        float32x4_t z0_sq = vmulq_f32(z0, z0);
        float32x4_t lp0 = vfmaq_f32(log_2pi, two, log_s0);
        lp0 = vaddq_f32(lp0, z0_sq);
        lp0 = vmulq_f32(neg_half, lp0);
        vst1q_f32(&log_prob[i], lp0);

        i += 4;

        /* Process remaining from z1 */
        size_t remaining = count - i;
        if (remaining > 0) {
            float z_temp[4], lp_temp[4];
            vst1q_f32(z_temp, z1);

            float32x4_t s1 = vdupq_n_f32(0.0f);
            float s1_arr[4] = {0};
            for (size_t j = 0; j < remaining; j++) {
                s1_arr[j] = std[i + j];
            }
            s1 = vld1q_f32(s1_arr);

            float32x4_t log_s1 = fast_log_neon(s1);
            float32x4_t z1_sq = vmulq_f32(z1, z1);
            float32x4_t lp1 = vfmaq_f32(log_2pi, two, log_s1);
            lp1 = vaddq_f32(lp1, z1_sq);
            lp1 = vmulq_f32(neg_half, lp1);
            vst1q_f32(lp_temp, lp1);

            for (size_t j = 0; j < remaining; j++) {
                output[i + j] = mean[i + j] + std[i + j] * z_temp[j];
                log_prob[i + j] = lp_temp[j];
            }
            i = count;
        }
    }

    /* Handle tail (< 4 samples) */
    if (i < count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);

        float z_temp[4];
        vst1q_f32(z_temp, z0);

        size_t remaining = count - i;
        for (size_t j = 0; j < remaining; j++) {
            float z = z_temp[j];
            float s = std[i + j];
            output[i + j] = mean[i + j] + s * z;
            log_prob[i + j] = -0.5f * (LOG_2PI + 2.0f * logf(s) + z * z);
        }
    }
}

/* ============================================================================
 * Additional Utility Functions
 * ============================================================================ */

/**
 * @brief Benchmark helper: generate many samples without storing
 *
 * Useful for measuring raw generation throughput.
 */
void benchmark_gaussian_throughput(size_t count, uint64_t* rng_state) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    size_t i = 0;
    while (i + 8 <= count) {
        float32x4_t z0, z1;
        box_muller_neon(rng_state, &z0, &z1);
        sum = vaddq_f32(sum, z0);
        sum = vaddq_f32(sum, z1);
        i += 8;
    }

    /* Prevent optimization by using the result */
    volatile float discard = vgetq_lane_f32(sum, 0);
    (void)discard;
}
