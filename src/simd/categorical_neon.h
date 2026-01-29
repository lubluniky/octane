/**
 * @file categorical_neon.h
 * @brief ARM NEON optimized Categorical distribution for Apple M4
 */

#ifndef CATEGORICAL_NEON_H
#define CATEGORICAL_NEON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* RNG state for xoshiro256** */
typedef struct {
    uint64_t s[4];
} xoshiro256_state_t;

/**
 * Initialize RNG state from seed
 */
void xoshiro256_init(xoshiro256_state_t* state, uint64_t seed);

/**
 * Numerically stable softmax using NEON
 *
 * @param logits     Input logits [batch_size, num_actions]
 * @param probs      Output probabilities [batch_size, num_actions]
 * @param batch_size Number of samples
 * @param num_actions Number of actions
 */
void softmax_neon(
    const float* logits,
    float* probs,
    size_t batch_size,
    size_t num_actions
);

/**
 * Log-softmax: more numerically stable than log(softmax(x))
 */
void log_softmax_neon(
    const float* logits,
    float* log_probs,
    size_t batch_size,
    size_t num_actions
);

/**
 * Sample from categorical distribution using Gumbel-max trick
 *
 * @param logits     Input logits [batch_size, num_actions]
 * @param actions    Output sampled actions [batch_size]
 * @param batch_size Number of samples
 * @param num_actions Number of actions
 * @param rng        RNG state
 */
void categorical_sample_gumbel_neon(
    const float* logits,
    uint32_t* actions,
    size_t batch_size,
    size_t num_actions,
    xoshiro256_state_t* rng
);

/**
 * Sample using inverse CDF method
 */
void categorical_sample_icdf_neon(
    const float* probs,
    const float* uniform_samples,
    uint32_t* actions,
    size_t batch_size,
    size_t num_actions
);

/**
 * Compute log probability of actions
 */
void categorical_log_prob_neon(
    const float* log_probs,
    const uint32_t* actions,
    float* output,
    size_t batch_size,
    size_t num_actions
);

/**
 * Compute entropy of categorical distribution
 */
void categorical_entropy_neon(
    const float* probs,
    const float* log_probs,
    float* entropy,
    size_t batch_size,
    size_t num_actions
);

/**
 * Compute KL divergence between two distributions
 */
void categorical_kl_divergence_neon(
    const float* p_probs,
    const float* p_log_probs,
    const float* q_log_probs,
    float* kl_div,
    size_t batch_size,
    size_t num_actions
);

/**
 * Combined forward pass: logits -> probs, log_probs, sample, entropy
 */
void categorical_forward_neon(
    const float* logits,
    float* probs,
    float* log_probs,
    uint32_t* actions,
    float* action_log_probs,
    float* entropy,
    size_t batch_size,
    size_t num_actions,
    xoshiro256_state_t* rng
);

#ifdef __cplusplus
}
#endif

#endif /* CATEGORICAL_NEON_H */
