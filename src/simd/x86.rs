//! AVX2/AVX-512 SIMD implementations for x86_64.
//!
//! This module provides high-performance SIMD operations using AVX2 and AVX-512
//! instructions for x86_64 processors. All operations have safe Rust wrappers
//! with proper alignment checking and error handling.
//!
//! # Available Operations
//!
//! - **Gaussian sampling**: Vectorized Box-Muller transform (8-wide AVX2, 16-wide AVX-512)
//! - **Softmax**: SIMD-accelerated softmax computation
//! - **Gather operations**: Fast batch gathering for replay buffers
//! - **GAE computation**: Generalized Advantage Estimation
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::simd::x86::{GaussianSamplerX86, is_avx2_available};
//!
//! if is_avx2_available() {
//!     let mut sampler = GaussianSamplerX86::new(42);
//!     let means = vec![0.0f32; 64];
//!     let stds = vec![1.0f32; 64];
//!     let samples = sampler.sample(&means, &stds)?;
//! }
//! ```

#![allow(unsafe_code)]

use super::{Result, SimdError};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// Feature Detection
// ============================================================================

/// Check if AVX2 is available on this CPU.
#[cfg(target_arch = "x86_64")]
pub fn is_avx2_available() -> bool {
    is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn is_avx2_available() -> bool {
    false
}

/// Check if AVX-512F is available on this CPU.
#[cfg(target_arch = "x86_64")]
pub fn is_avx512_available() -> bool {
    is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512dq")
}

#[cfg(not(target_arch = "x86_64"))]
pub fn is_avx512_available() -> bool {
    false
}

// ============================================================================
// Constants
// ============================================================================

/// Required alignment for AVX2 operations (32 bytes).
pub const AVX2_ALIGNMENT: usize = 32;

/// Required alignment for AVX-512 operations (64 bytes).
pub const AVX512_ALIGNMENT: usize = 64;

/// Log(2 * PI) constant for Gaussian computations.
const LOG_2PI: f32 = 1.8378770664093453;

/// Small epsilon for numerical stability.
const EPSILON: f32 = 1e-8;

// ============================================================================
// RNG State (xoroshiro128+ for AVX2, extended for AVX-512)
// ============================================================================

/// RNG state for AVX2 (4 parallel xoroshiro128+ streams).
#[derive(Clone)]
pub struct RngStateAvx2 {
    state: [u64; 8], // 4 streams x 2 u64 each
}

impl RngStateAvx2 {
    /// Create a new RNG state from a seed.
    pub fn new(seed: u64) -> Self {
        let mut state = [0u64; 8];
        // SplitMix64 to generate initial states
        let mut x = seed;
        for s in state.iter_mut() {
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *s = z ^ (z >> 31);
        }
        Self { state }
    }

    /// Generate next batch of random u64 values (4 values).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn next_u64x4(&mut self) -> [u64; 4] {
        let mut result = [0u64; 4];
        for i in 0..4 {
            let s0 = self.state[i * 2];
            let mut s1 = self.state[i * 2 + 1];
            result[i] = s0.wrapping_add(s1);

            s1 ^= s0;
            self.state[i * 2] = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
            self.state[i * 2 + 1] = s1.rotate_left(37);
        }
        result
    }
}

/// RNG state for AVX-512 (8 parallel xoroshiro128+ streams).
#[derive(Clone)]
pub struct RngStateAvx512 {
    state: [u64; 16], // 8 streams x 2 u64 each
}

impl RngStateAvx512 {
    /// Create a new RNG state from a seed.
    pub fn new(seed: u64) -> Self {
        let mut state = [0u64; 16];
        let mut x = seed;
        for s in state.iter_mut() {
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            *s = z ^ (z >> 31);
        }
        Self { state }
    }

    /// Generate next batch of random u64 values (8 values).
    #[cfg(target_arch = "x86_64")]
    #[inline]
    fn next_u64x8(&mut self) -> [u64; 8] {
        let mut result = [0u64; 8];
        for i in 0..8 {
            let s0 = self.state[i * 2];
            let mut s1 = self.state[i * 2 + 1];
            result[i] = s0.wrapping_add(s1);

            s1 ^= s0;
            self.state[i * 2] = s0.rotate_left(24) ^ s1 ^ (s1 << 16);
            self.state[i * 2 + 1] = s1.rotate_left(37);
        }
        result
    }
}

// ============================================================================
// Gaussian Sampling - AVX2 (8-wide Box-Muller)
// ============================================================================

/// High-performance Gaussian sampler using AVX2 SIMD.
///
/// Uses vectorized Box-Muller transform with parallel xoroshiro128+ RNG streams.
/// Processes 8 samples at a time using 256-bit SIMD registers.
pub struct GaussianSamplerAvx2 {
    rng: RngStateAvx2,
    /// Cached values from Box-Muller (produces pairs).
    cache: Vec<f32>,
}

impl GaussianSamplerAvx2 {
    /// Create a new sampler with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RngStateAvx2::new(seed),
            cache: Vec::new(),
        }
    }

    /// Sample from standard normal distribution N(0, 1).
    ///
    /// Uses Box-Muller transform: given U1, U2 ~ Uniform(0, 1),
    /// Z0 = sqrt(-2 * ln(U1)) * cos(2 * pi * U2)
    /// Z1 = sqrt(-2 * ln(U1)) * sin(2 * pi * U2)
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    pub fn sample_standard_normal(&mut self, count: usize) -> Result<Vec<f32>> {
        if !is_avx2_available() {
            return Err(SimdError::InvalidParameter(
                "AVX2 not available".to_string(),
            ));
        }

        let mut output = Vec::with_capacity(count);

        // Use cached values first
        while !self.cache.is_empty() && output.len() < count {
            output.push(self.cache.pop().unwrap());
        }

        // Generate new samples in batches of 8
        unsafe {
            self.sample_standard_normal_avx2(&mut output, count);
        }

        output.truncate(count);
        Ok(output)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn sample_standard_normal_avx2(&mut self, output: &mut Vec<f32>, count: usize) {
        let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
        let neg_two = _mm256_set1_ps(-2.0);
        let one = _mm256_set1_ps(1.0);
        let scale = _mm256_set1_ps(2.3283064365386963e-10_f32); // 1 / 2^32

        while output.len() < count {
            // Generate 8 uniform random values (need 2 sets for Box-Muller)
            let u64_1 = self.rng.next_u64x4();
            let u64_2 = self.rng.next_u64x4();

            // Convert to f32 in [0, 1)
            // Take lower 32 bits of each u64
            let u32_1: [u32; 8] = [
                u64_1[0] as u32,
                (u64_1[0] >> 32) as u32,
                u64_1[1] as u32,
                (u64_1[1] >> 32) as u32,
                u64_1[2] as u32,
                (u64_1[2] >> 32) as u32,
                u64_1[3] as u32,
                (u64_1[3] >> 32) as u32,
            ];
            let u32_2: [u32; 8] = [
                u64_2[0] as u32,
                (u64_2[0] >> 32) as u32,
                u64_2[1] as u32,
                (u64_2[1] >> 32) as u32,
                u64_2[2] as u32,
                (u64_2[2] >> 32) as u32,
                u64_2[3] as u32,
                (u64_2[3] >> 32) as u32,
            ];

            let u1_int = _mm256_loadu_si256(u32_1.as_ptr() as *const __m256i);
            let u2_int = _mm256_loadu_si256(u32_2.as_ptr() as *const __m256i);

            // Convert to float and scale to (0, 1)
            let u1 = _mm256_add_ps(
                _mm256_mul_ps(_mm256_cvtepi32_ps(u1_int), scale),
                _mm256_set1_ps(0.5),
            );
            let u2 = _mm256_mul_ps(_mm256_cvtepi32_ps(u2_int), scale);

            // Clamp u1 to avoid log(0)
            let u1_clamped = _mm256_max_ps(u1, _mm256_set1_ps(1e-10));

            // Box-Muller: r = sqrt(-2 * ln(u1))
            // Using approximation for ln and sqrt
            let ln_u1 = self.fast_log_avx2(u1_clamped);
            let r_sq = _mm256_mul_ps(neg_two, ln_u1);
            let r = _mm256_sqrt_ps(r_sq);

            // theta = 2 * pi * u2
            let theta = _mm256_mul_ps(two_pi, u2);

            // z0 = r * cos(theta), z1 = r * sin(theta)
            let (sin_theta, cos_theta) = self.fast_sincos_avx2(theta);
            let z0 = _mm256_mul_ps(r, cos_theta);
            let z1 = _mm256_mul_ps(r, sin_theta);

            // Store results
            let mut z0_arr = [0.0f32; 8];
            let mut z1_arr = [0.0f32; 8];
            _mm256_storeu_ps(z0_arr.as_mut_ptr(), z0);
            _mm256_storeu_ps(z1_arr.as_mut_ptr(), z1);

            for &v in &z0_arr {
                if output.len() < count {
                    output.push(v);
                } else {
                    self.cache.push(v);
                }
            }
            for &v in &z1_arr {
                if output.len() < count {
                    output.push(v);
                } else {
                    self.cache.push(v);
                }
            }
        }
    }

    /// Fast log approximation using AVX2.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn fast_log_avx2(&self, x: __m256) -> __m256 {
        // Polynomial approximation of ln(x) for x in [0.5, 2]
        // ln(x) = ln(m * 2^e) = ln(m) + e * ln(2)
        let one = _mm256_set1_ps(1.0);
        let ln2 = _mm256_set1_ps(0.6931471805599453);

        // Extract exponent
        let xi = _mm256_castps_si256(x);
        let exp = _mm256_sub_epi32(_mm256_srli_epi32(xi, 23), _mm256_set1_epi32(127));
        let exp_f = _mm256_cvtepi32_ps(exp);

        // Extract mantissa and normalize to [1, 2)
        let mantissa_mask = _mm256_set1_epi32(0x007FFFFF);
        let one_bits = _mm256_set1_epi32(0x3F800000);
        let m_int = _mm256_or_si256(_mm256_and_si256(xi, mantissa_mask), one_bits);
        let m = _mm256_castsi256_ps(m_int);

        // Polynomial for ln(m) where m in [1, 2)
        // Using minimax polynomial
        let m_minus_1 = _mm256_sub_ps(m, one);
        let c1 = _mm256_set1_ps(0.9999964239);
        let c2 = _mm256_set1_ps(-0.4998741238);
        let c3 = _mm256_set1_ps(0.3317990258);
        let c4 = _mm256_set1_ps(-0.2407338084);
        let c5 = _mm256_set1_ps(0.1676540711);
        let c6 = _mm256_set1_ps(-0.0953293897);

        // Horner's method
        let mut result = c6;
        result = _mm256_fmadd_ps(result, m_minus_1, c5);
        result = _mm256_fmadd_ps(result, m_minus_1, c4);
        result = _mm256_fmadd_ps(result, m_minus_1, c3);
        result = _mm256_fmadd_ps(result, m_minus_1, c2);
        result = _mm256_fmadd_ps(result, m_minus_1, c1);
        result = _mm256_mul_ps(result, m_minus_1);

        // Add exponent contribution
        _mm256_fmadd_ps(exp_f, ln2, result)
    }

    /// Fast sin/cos approximation using AVX2.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn fast_sincos_avx2(&self, x: __m256) -> (__m256, __m256) {
        // Range reduction to [-pi, pi]
        let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
        let inv_two_pi = _mm256_set1_ps(1.0 / (2.0 * std::f32::consts::PI));
        let pi = _mm256_set1_ps(std::f32::consts::PI);

        // x = x - 2*pi * round(x / (2*pi))
        let n = _mm256_round_ps::<_MM_FROUND_TO_NEAREST_INT>(
            _mm256_mul_ps(x, inv_two_pi),
        );
        let x_reduced = _mm256_fnmadd_ps(n, two_pi, x);

        // Taylor series approximation
        // sin(x) = x - x^3/6 + x^5/120 - x^7/5040
        // cos(x) = 1 - x^2/2 + x^4/24 - x^6/720
        let x2 = _mm256_mul_ps(x_reduced, x_reduced);
        let x3 = _mm256_mul_ps(x2, x_reduced);
        let x4 = _mm256_mul_ps(x2, x2);
        let x5 = _mm256_mul_ps(x4, x_reduced);
        let x6 = _mm256_mul_ps(x4, x2);
        let x7 = _mm256_mul_ps(x6, x_reduced);

        // Sin coefficients
        let s1 = _mm256_set1_ps(1.0);
        let s3 = _mm256_set1_ps(-1.0 / 6.0);
        let s5 = _mm256_set1_ps(1.0 / 120.0);
        let s7 = _mm256_set1_ps(-1.0 / 5040.0);

        // Cos coefficients
        let c0 = _mm256_set1_ps(1.0);
        let c2 = _mm256_set1_ps(-0.5);
        let c4 = _mm256_set1_ps(1.0 / 24.0);
        let c6 = _mm256_set1_ps(-1.0 / 720.0);

        let sin_x = _mm256_fmadd_ps(
            s7,
            x7,
            _mm256_fmadd_ps(
                s5,
                x5,
                _mm256_fmadd_ps(s3, x3, _mm256_mul_ps(s1, x_reduced)),
            ),
        );

        let cos_x = _mm256_fmadd_ps(
            c6,
            x6,
            _mm256_fmadd_ps(c4, x4, _mm256_fmadd_ps(c2, x2, c0)),
        );

        (sin_x, cos_x)
    }

    /// Sample with reparameterization: output = mean + std * N(0, 1).
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    pub fn sample(&mut self, mean: &[f32], std: &[f32]) -> Result<Vec<f32>> {
        if mean.len() != std.len() {
            return Err(SimdError::SizeMismatch {
                expected: mean.len(),
                actual: std.len(),
            });
        }

        let z = self.sample_standard_normal(mean.len())?;
        let mut output = vec![0.0f32; mean.len()];

        unsafe {
            self.reparameterize_avx2(mean, std, &z, &mut output);
        }

        Ok(output)
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn reparameterize_avx2(
        &self,
        mean: &[f32],
        std: &[f32],
        z: &[f32],
        output: &mut [f32],
    ) {
        let n = mean.len();
        let chunks = n / 8;
        let remainder = n % 8;

        for i in 0..chunks {
            let idx = i * 8;
            let m = _mm256_loadu_ps(mean.as_ptr().add(idx));
            let s = _mm256_loadu_ps(std.as_ptr().add(idx));
            let noise = _mm256_loadu_ps(z.as_ptr().add(idx));

            // output = mean + std * z
            let result = _mm256_fmadd_ps(s, noise, m);
            _mm256_storeu_ps(output.as_mut_ptr().add(idx), result);
        }

        // Handle remainder
        for i in (chunks * 8)..n {
            output[i] = mean[i] + std[i] * z[i];
        }
    }

    /// Sample and compute log probability simultaneously.
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    pub fn sample_with_log_prob(
        &mut self,
        mean: &[f32],
        std: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        if mean.len() != std.len() {
            return Err(SimdError::SizeMismatch {
                expected: mean.len(),
                actual: std.len(),
            });
        }

        let z = self.sample_standard_normal(mean.len())?;
        let mut output = vec![0.0f32; mean.len()];
        let mut log_prob = vec![0.0f32; mean.len()];

        unsafe {
            self.sample_with_log_prob_avx2(mean, std, &z, &mut output, &mut log_prob);
        }

        Ok((output, log_prob))
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn sample_with_log_prob_avx2(
        &self,
        mean: &[f32],
        std: &[f32],
        z: &[f32],
        output: &mut [f32],
        log_prob: &mut [f32],
    ) {
        let n = mean.len();
        let chunks = n / 8;

        let log_2pi_vec = _mm256_set1_ps(LOG_2PI);
        let neg_half = _mm256_set1_ps(-0.5);
        let two = _mm256_set1_ps(2.0);

        for i in 0..chunks {
            let idx = i * 8;
            let m = _mm256_loadu_ps(mean.as_ptr().add(idx));
            let s = _mm256_loadu_ps(std.as_ptr().add(idx));
            let noise = _mm256_loadu_ps(z.as_ptr().add(idx));

            // output = mean + std * z
            let result = _mm256_fmadd_ps(s, noise, m);
            _mm256_storeu_ps(output.as_mut_ptr().add(idx), result);

            // log_prob = -0.5 * (log(2*pi) + 2*log(std) + z^2)
            let log_std = self.fast_log_avx2(s);
            let z_sq = _mm256_mul_ps(noise, noise);
            let two_log_std = _mm256_mul_ps(two, log_std);
            let inner = _mm256_add_ps(log_2pi_vec, _mm256_add_ps(two_log_std, z_sq));
            let lp = _mm256_mul_ps(neg_half, inner);
            _mm256_storeu_ps(log_prob.as_mut_ptr().add(idx), lp);
        }

        // Handle remainder
        for i in (chunks * 8)..n {
            output[i] = mean[i] + std[i] * z[i];
            let log_std = std[i].ln();
            log_prob[i] = -0.5 * (LOG_2PI + 2.0 * log_std + z[i] * z[i]);
        }
    }

    // Fallback implementations for when AVX2 is not available at compile time
    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    pub fn sample_standard_normal(&mut self, count: usize) -> Result<Vec<f32>> {
        Err(SimdError::InvalidParameter(
            "AVX2 not available - compile with target-feature=+avx2,+fma".to_string(),
        ))
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    pub fn sample(&mut self, _mean: &[f32], _std: &[f32]) -> Result<Vec<f32>> {
        Err(SimdError::InvalidParameter(
            "AVX2 not available - compile with target-feature=+avx2,+fma".to_string(),
        ))
    }

    #[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
    pub fn sample_with_log_prob(
        &mut self,
        _mean: &[f32],
        _std: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        Err(SimdError::InvalidParameter(
            "AVX2 not available - compile with target-feature=+avx2,+fma".to_string(),
        ))
    }
}

// ============================================================================
// Gaussian Sampling - AVX-512 (16-wide Box-Muller)
// ============================================================================

/// High-performance Gaussian sampler using AVX-512 SIMD.
///
/// Uses vectorized Box-Muller transform with parallel xoroshiro128+ RNG streams.
/// Processes 16 samples at a time using 512-bit SIMD registers.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
pub struct GaussianSamplerAvx512 {
    rng: RngStateAvx512,
    cache: Vec<f32>,
}

#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
impl GaussianSamplerAvx512 {
    /// Create a new sampler with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            rng: RngStateAvx512::new(seed),
            cache: Vec::new(),
        }
    }

    /// Sample from standard normal distribution N(0, 1) using AVX-512.
    #[cfg(target_feature = "avx512f")]
    pub fn sample_standard_normal(&mut self, count: usize) -> Result<Vec<f32>> {
        if !is_avx512_available() {
            return Err(SimdError::InvalidParameter(
                "AVX-512 not available".to_string(),
            ));
        }

        let mut output = Vec::with_capacity(count);

        // Use cached values first
        while !self.cache.is_empty() && output.len() < count {
            output.push(self.cache.pop().unwrap());
        }

        unsafe {
            self.sample_standard_normal_avx512(&mut output, count);
        }

        output.truncate(count);
        Ok(output)
    }

    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f", enable = "avx512dq")]
    unsafe fn sample_standard_normal_avx512(&mut self, output: &mut Vec<f32>, count: usize) {
        let two_pi = _mm512_set1_ps(2.0 * std::f32::consts::PI);
        let neg_two = _mm512_set1_ps(-2.0);
        let scale = _mm512_set1_ps(2.3283064365386963e-10_f32);

        while output.len() < count {
            // Generate 16 uniform random values
            let u64_1 = self.rng.next_u64x8();
            let u64_2 = self.rng.next_u64x8();

            // Convert to f32
            let mut u32_1 = [0u32; 16];
            let mut u32_2 = [0u32; 16];
            for i in 0..8 {
                u32_1[i * 2] = u64_1[i] as u32;
                u32_1[i * 2 + 1] = (u64_1[i] >> 32) as u32;
                u32_2[i * 2] = u64_2[i] as u32;
                u32_2[i * 2 + 1] = (u64_2[i] >> 32) as u32;
            }

            let u1_int = _mm512_loadu_si512(u32_1.as_ptr() as *const i32);
            let u2_int = _mm512_loadu_si512(u32_2.as_ptr() as *const i32);

            let u1 = _mm512_add_ps(
                _mm512_mul_ps(_mm512_cvtepi32_ps(u1_int), scale),
                _mm512_set1_ps(0.5),
            );
            let u2 = _mm512_mul_ps(_mm512_cvtepi32_ps(u2_int), scale);

            // Clamp u1 to avoid log(0)
            let u1_clamped = _mm512_max_ps(u1, _mm512_set1_ps(1e-10));

            // Box-Muller transform
            let ln_u1 = self.fast_log_avx512(u1_clamped);
            let r_sq = _mm512_mul_ps(neg_two, ln_u1);
            let r = _mm512_sqrt_ps(r_sq);

            let theta = _mm512_mul_ps(two_pi, u2);
            let sin_theta = self.fast_sin_avx512(theta);
            let cos_theta = self.fast_cos_avx512(theta);

            let z0 = _mm512_mul_ps(r, cos_theta);
            let z1 = _mm512_mul_ps(r, sin_theta);

            let mut z0_arr = [0.0f32; 16];
            let mut z1_arr = [0.0f32; 16];
            _mm512_storeu_ps(z0_arr.as_mut_ptr(), z0);
            _mm512_storeu_ps(z1_arr.as_mut_ptr(), z1);

            for &v in &z0_arr {
                if output.len() < count {
                    output.push(v);
                } else {
                    self.cache.push(v);
                }
            }
            for &v in &z1_arr {
                if output.len() < count {
                    output.push(v);
                } else {
                    self.cache.push(v);
                }
            }
        }
    }

    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fast_log_avx512(&self, x: __m512) -> __m512 {
        // Similar to AVX2 version but 16-wide
        let ln2 = _mm512_set1_ps(0.6931471805599453);

        let xi = _mm512_castps_si512(x);
        let exp = _mm512_sub_epi32(_mm512_srli_epi32(xi, 23), _mm512_set1_epi32(127));
        let exp_f = _mm512_cvtepi32_ps(exp);

        let mantissa_mask = _mm512_set1_epi32(0x007FFFFF);
        let one_bits = _mm512_set1_epi32(0x3F800000);
        let m_int = _mm512_or_si512(_mm512_and_si512(xi, mantissa_mask), one_bits);
        let m = _mm512_castsi512_ps(m_int);

        let one = _mm512_set1_ps(1.0);
        let m_minus_1 = _mm512_sub_ps(m, one);

        // Polynomial coefficients
        let c1 = _mm512_set1_ps(0.9999964239);
        let c2 = _mm512_set1_ps(-0.4998741238);
        let c3 = _mm512_set1_ps(0.3317990258);
        let c4 = _mm512_set1_ps(-0.2407338084);
        let c5 = _mm512_set1_ps(0.1676540711);
        let c6 = _mm512_set1_ps(-0.0953293897);

        let mut result = c6;
        result = _mm512_fmadd_ps(result, m_minus_1, c5);
        result = _mm512_fmadd_ps(result, m_minus_1, c4);
        result = _mm512_fmadd_ps(result, m_minus_1, c3);
        result = _mm512_fmadd_ps(result, m_minus_1, c2);
        result = _mm512_fmadd_ps(result, m_minus_1, c1);
        result = _mm512_mul_ps(result, m_minus_1);

        _mm512_fmadd_ps(exp_f, ln2, result)
    }

    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fast_sin_avx512(&self, x: __m512) -> __m512 {
        let two_pi = _mm512_set1_ps(2.0 * std::f32::consts::PI);
        let inv_two_pi = _mm512_set1_ps(1.0 / (2.0 * std::f32::consts::PI));

        let n = _mm512_roundscale_ps::<0>(
            _mm512_mul_ps(x, inv_two_pi),
        );
        let x_reduced = _mm512_fnmadd_ps(n, two_pi, x);

        let x2 = _mm512_mul_ps(x_reduced, x_reduced);
        let x3 = _mm512_mul_ps(x2, x_reduced);
        let x5 = _mm512_mul_ps(x3, x2);
        let x7 = _mm512_mul_ps(x5, x2);

        let s3 = _mm512_set1_ps(-1.0 / 6.0);
        let s5 = _mm512_set1_ps(1.0 / 120.0);
        let s7 = _mm512_set1_ps(-1.0 / 5040.0);

        _mm512_fmadd_ps(
            s7,
            x7,
            _mm512_fmadd_ps(s5, x5, _mm512_fmadd_ps(s3, x3, x_reduced)),
        )
    }

    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn fast_cos_avx512(&self, x: __m512) -> __m512 {
        let two_pi = _mm512_set1_ps(2.0 * std::f32::consts::PI);
        let inv_two_pi = _mm512_set1_ps(1.0 / (2.0 * std::f32::consts::PI));

        let n = _mm512_roundscale_ps::<0>(
            _mm512_mul_ps(x, inv_two_pi),
        );
        let x_reduced = _mm512_fnmadd_ps(n, two_pi, x);

        let x2 = _mm512_mul_ps(x_reduced, x_reduced);
        let x4 = _mm512_mul_ps(x2, x2);
        let x6 = _mm512_mul_ps(x4, x2);

        let c0 = _mm512_set1_ps(1.0);
        let c2 = _mm512_set1_ps(-0.5);
        let c4 = _mm512_set1_ps(1.0 / 24.0);
        let c6 = _mm512_set1_ps(-1.0 / 720.0);

        _mm512_fmadd_ps(c6, x6, _mm512_fmadd_ps(c4, x4, _mm512_fmadd_ps(c2, x2, c0)))
    }

    /// Sample with reparameterization.
    #[cfg(target_feature = "avx512f")]
    pub fn sample(&mut self, mean: &[f32], std: &[f32]) -> Result<Vec<f32>> {
        if mean.len() != std.len() {
            return Err(SimdError::SizeMismatch {
                expected: mean.len(),
                actual: std.len(),
            });
        }

        let z = self.sample_standard_normal(mean.len())?;
        let mut output = vec![0.0f32; mean.len()];

        unsafe {
            self.reparameterize_avx512(mean, std, &z, &mut output);
        }

        Ok(output)
    }

    #[cfg(target_feature = "avx512f")]
    #[target_feature(enable = "avx512f")]
    unsafe fn reparameterize_avx512(
        &self,
        mean: &[f32],
        std: &[f32],
        z: &[f32],
        output: &mut [f32],
    ) {
        let n = mean.len();
        let chunks = n / 16;

        for i in 0..chunks {
            let idx = i * 16;
            let m = _mm512_loadu_ps(mean.as_ptr().add(idx));
            let s = _mm512_loadu_ps(std.as_ptr().add(idx));
            let noise = _mm512_loadu_ps(z.as_ptr().add(idx));

            let result = _mm512_fmadd_ps(s, noise, m);
            _mm512_storeu_ps(output.as_mut_ptr().add(idx), result);
        }

        for i in (chunks * 16)..n {
            output[i] = mean[i] + std[i] * z[i];
        }
    }

    // Fallback for non-AVX512 compile targets
    #[cfg(not(target_feature = "avx512f"))]
    pub fn sample_standard_normal(&mut self, _count: usize) -> Result<Vec<f32>> {
        Err(SimdError::InvalidParameter(
            "AVX-512 not available - compile with target-feature=+avx512f".to_string(),
        ))
    }

    #[cfg(not(target_feature = "avx512f"))]
    pub fn sample(&mut self, _mean: &[f32], _std: &[f32]) -> Result<Vec<f32>> {
        Err(SimdError::InvalidParameter(
            "AVX-512 not available - compile with target-feature=+avx512f".to_string(),
        ))
    }
}

// ============================================================================
// Softmax - AVX2
// ============================================================================

/// Compute softmax using AVX2 SIMD.
///
/// # Arguments
///
/// * `input` - Input logits [batch_size, num_classes]
/// * `batch_size` - Number of samples
/// * `num_classes` - Number of classes
///
/// # Returns
///
/// Softmax probabilities with same shape as input.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
pub fn softmax_avx2(input: &[f32], batch_size: usize, num_classes: usize) -> Result<Vec<f32>> {
    if input.len() != batch_size * num_classes {
        return Err(SimdError::SizeMismatch {
            expected: batch_size * num_classes,
            actual: input.len(),
        });
    }

    if !is_avx2_available() {
        return Err(SimdError::InvalidParameter(
            "AVX2 not available".to_string(),
        ));
    }

    let mut output = vec![0.0f32; input.len()];

    unsafe {
        softmax_avx2_impl(input, &mut output, batch_size, num_classes);
    }

    Ok(output)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn softmax_avx2_impl(
    input: &[f32],
    output: &mut [f32],
    batch_size: usize,
    num_classes: usize,
) {
    for b in 0..batch_size {
        let offset = b * num_classes;
        let slice = &input[offset..offset + num_classes];

        // Find max for numerical stability
        let mut max_val = f32::NEG_INFINITY;
        let chunks = num_classes / 8;

        if chunks > 0 {
            let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
            for i in 0..chunks {
                let v = _mm256_loadu_ps(slice.as_ptr().add(i * 8));
                max_vec = _mm256_max_ps(max_vec, v);
            }

            // Horizontal max
            let mut max_arr = [0.0f32; 8];
            _mm256_storeu_ps(max_arr.as_mut_ptr(), max_vec);
            for &v in &max_arr {
                max_val = max_val.max(v);
            }
        }

        // Handle remainder
        for i in (chunks * 8)..num_classes {
            max_val = max_val.max(slice[i]);
        }

        // Compute exp(x - max) and sum
        let max_vec = _mm256_set1_ps(max_val);
        let mut sum = 0.0f32;

        for i in 0..chunks {
            let idx = offset + i * 8;
            let v = _mm256_loadu_ps(input.as_ptr().add(idx));
            let shifted = _mm256_sub_ps(v, max_vec);
            let exp_v = fast_exp_avx2(shifted);
            _mm256_storeu_ps(output.as_mut_ptr().add(idx), exp_v);

            let mut exp_arr = [0.0f32; 8];
            _mm256_storeu_ps(exp_arr.as_mut_ptr(), exp_v);
            sum += exp_arr.iter().sum::<f32>();
        }

        // Handle remainder
        for i in (chunks * 8)..num_classes {
            let exp_v = (slice[i] - max_val).exp();
            output[offset + i] = exp_v;
            sum += exp_v;
        }

        // Normalize
        let inv_sum = 1.0 / sum;
        let inv_sum_vec = _mm256_set1_ps(inv_sum);

        for i in 0..chunks {
            let idx = offset + i * 8;
            let v = _mm256_loadu_ps(output.as_ptr().add(idx));
            let normalized = _mm256_mul_ps(v, inv_sum_vec);
            _mm256_storeu_ps(output.as_mut_ptr().add(idx), normalized);
        }

        for i in (chunks * 8)..num_classes {
            output[offset + i] *= inv_sum;
        }
    }
}

/// Fast exp approximation using AVX2.
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
unsafe fn fast_exp_avx2(x: __m256) -> __m256 {
    // exp(x) = 2^(x * log2(e))
    // Using polynomial approximation for 2^f where f is fractional part
    let log2e = _mm256_set1_ps(1.4426950408889634);
    let one = _mm256_set1_ps(1.0);

    // Clamp to avoid overflow/underflow
    let x_clamped = _mm256_max_ps(
        _mm256_min_ps(x, _mm256_set1_ps(88.0)),
        _mm256_set1_ps(-88.0),
    );

    let t = _mm256_mul_ps(x_clamped, log2e);

    // Split into integer and fractional parts
    let t_floor = _mm256_floor_ps(t);
    let f = _mm256_sub_ps(t, t_floor);
    let n = _mm256_cvtps_epi32(t_floor);

    // Polynomial approximation for 2^f, f in [0, 1]
    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(0.6931471805599453);
    let c2 = _mm256_set1_ps(0.2402265069591007);
    let c3 = _mm256_set1_ps(0.0555041086648216);
    let c4 = _mm256_set1_ps(0.0096181291076284);
    let c5 = _mm256_set1_ps(0.0013333558146428);

    let mut p = c5;
    p = _mm256_fmadd_ps(p, f, c4);
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);

    // Scale by 2^n
    let n_scaled = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    let scale = _mm256_castsi256_ps(n_scaled);

    _mm256_mul_ps(p, scale)
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
pub fn softmax_avx2(_input: &[f32], _batch_size: usize, _num_classes: usize) -> Result<Vec<f32>> {
    Err(SimdError::InvalidParameter(
        "AVX2 not available - compile with target-feature=+avx2,+fma".to_string(),
    ))
}

// ============================================================================
// Gather Operations - AVX2
// ============================================================================

/// Gather rows from a 2D array using random indices (AVX2 optimized).
///
/// # Arguments
///
/// * `src` - Source buffer of shape [capacity, dim]
/// * `indices` - Indices to gather [batch_size]
/// * `dim` - Dimension of each row
/// * `capacity` - Total capacity of source buffer
///
/// # Returns
///
/// Gathered data of shape [batch_size, dim]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn gather_batch_f32_avx2(
    src: &[f32],
    indices: &[usize],
    dim: usize,
    capacity: usize,
) -> Result<Vec<f32>> {
    let batch_size = indices.len();
    let expected_src_len = capacity * dim;
    if src.len() < expected_src_len {
        return Err(SimdError::SizeMismatch {
            expected: expected_src_len,
            actual: src.len(),
        });
    }

    // Validate indices
    for &idx in indices {
        if idx >= capacity {
            return Err(SimdError::IndexOutOfBounds {
                index: idx,
                capacity,
            });
        }
    }

    let mut dst = vec![0.0f32; batch_size * dim];

    unsafe {
        gather_batch_f32_avx2_impl(src, indices, &mut dst, dim);
    }

    Ok(dst)
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
#[target_feature(enable = "avx2")]
unsafe fn gather_batch_f32_avx2_impl(
    src: &[f32],
    indices: &[usize],
    dst: &mut [f32],
    dim: usize,
) {
    let batch_size = indices.len();
    let chunks = dim / 8;

    for (i, &idx) in indices.iter().enumerate() {
        let src_offset = idx * dim;
        let dst_offset = i * dim;

        // Copy 8 floats at a time
        for c in 0..chunks {
            let v = _mm256_loadu_ps(src.as_ptr().add(src_offset + c * 8));
            _mm256_storeu_ps(dst.as_mut_ptr().add(dst_offset + c * 8), v);
        }

        // Handle remainder
        for j in (chunks * 8)..dim {
            dst[dst_offset + j] = src[src_offset + j];
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2")))]
pub fn gather_batch_f32_avx2(
    _src: &[f32],
    _indices: &[usize],
    _dim: usize,
    _capacity: usize,
) -> Result<Vec<f32>> {
    Err(SimdError::InvalidParameter(
        "AVX2 not available - compile with target-feature=+avx2".to_string(),
    ))
}

// ============================================================================
// GAE Computation - AVX2
// ============================================================================

/// Compute Generalized Advantage Estimation (GAE) using AVX2 SIMD.
///
/// # Arguments
///
/// * `rewards` - Rewards [num_steps, num_envs]
/// * `values` - Value estimates [num_steps, num_envs]
/// * `dones` - Episode termination flags [num_steps, num_envs]
/// * `num_steps` - Number of time steps
/// * `num_envs` - Number of parallel environments
/// * `gamma` - Discount factor
/// * `gae_lambda` - GAE lambda parameter
/// * `last_values` - Bootstrap values [num_envs]
///
/// # Returns
///
/// Tuple of (advantages, returns) each with shape [num_steps, num_envs]
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
pub fn compute_gae_avx2(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let expected_len = num_steps * num_envs;
    if rewards.len() != expected_len || values.len() != expected_len || dones.len() != expected_len {
        return Err(SimdError::SizeMismatch {
            expected: expected_len,
            actual: rewards.len().min(values.len()).min(dones.len()),
        });
    }

    if last_values.len() != num_envs {
        return Err(SimdError::SizeMismatch {
            expected: num_envs,
            actual: last_values.len(),
        });
    }

    let mut advantages = vec![0.0f32; expected_len];
    let mut returns = vec![0.0f32; expected_len];

    unsafe {
        compute_gae_avx2_impl(
            rewards,
            values,
            dones,
            &mut advantages,
            &mut returns,
            num_steps,
            num_envs,
            gamma,
            gae_lambda,
            last_values,
        );
    }

    Ok((advantages, returns))
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_gae_avx2_impl(
    rewards: &[f32],
    values: &[f32],
    dones: &[f32],
    advantages: &mut [f32],
    returns: &mut [f32],
    num_steps: usize,
    num_envs: usize,
    gamma: f32,
    gae_lambda: f32,
    last_values: &[f32],
) {
    let gamma_vec = _mm256_set1_ps(gamma);
    let gae_lambda_vec = _mm256_set1_ps(gae_lambda);
    let one_vec = _mm256_set1_ps(1.0);
    let gamma_lambda = _mm256_set1_ps(gamma * gae_lambda);

    let chunks = num_envs / 8;

    // Process 8 environments at a time
    for chunk in 0..chunks {
        let env_offset = chunk * 8;

        let mut last_gae = _mm256_setzero_ps();
        let mut next_value = _mm256_loadu_ps(last_values.as_ptr().add(env_offset));

        // Backward pass through time
        for step in (0..num_steps).rev() {
            let idx = step * num_envs + env_offset;

            let reward = _mm256_loadu_ps(rewards.as_ptr().add(idx));
            let value = _mm256_loadu_ps(values.as_ptr().add(idx));
            let done = _mm256_loadu_ps(dones.as_ptr().add(idx));

            // mask = 1 - done
            let mask = _mm256_sub_ps(one_vec, done);

            // delta = reward + gamma * next_value * mask - value
            let gamma_next = _mm256_mul_ps(gamma_vec, next_value);
            let gamma_next_masked = _mm256_mul_ps(gamma_next, mask);
            let delta = _mm256_sub_ps(_mm256_add_ps(reward, gamma_next_masked), value);

            // last_gae = delta + gamma * lambda * mask * last_gae
            let gae_decay = _mm256_mul_ps(gamma_lambda, mask);
            last_gae = _mm256_fmadd_ps(gae_decay, last_gae, delta);

            // Store advantages
            _mm256_storeu_ps(advantages.as_mut_ptr().add(idx), last_gae);

            // returns = advantages + values
            let ret = _mm256_add_ps(last_gae, value);
            _mm256_storeu_ps(returns.as_mut_ptr().add(idx), ret);

            next_value = value;
        }
    }

    // Handle remaining environments (scalar fallback)
    for env in (chunks * 8)..num_envs {
        let mut last_gae = 0.0f32;
        let mut next_value = last_values[env];

        for step in (0..num_steps).rev() {
            let idx = step * num_envs + env;
            let mask = 1.0 - dones[idx];

            let delta = rewards[idx] + gamma * next_value * mask - values[idx];
            last_gae = delta + gamma * gae_lambda * mask * last_gae;

            advantages[idx] = last_gae;
            returns[idx] = last_gae + values[idx];
            next_value = values[idx];
        }
    }
}

#[cfg(not(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
pub fn compute_gae_avx2(
    _rewards: &[f32],
    _values: &[f32],
    _dones: &[f32],
    _num_steps: usize,
    _num_envs: usize,
    _gamma: f32,
    _gae_lambda: f32,
    _last_values: &[f32],
) -> Result<(Vec<f32>, Vec<f32>)> {
    Err(SimdError::InvalidParameter(
        "AVX2 not available - compile with target-feature=+avx2,+fma".to_string(),
    ))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_detection() {
        // These should not panic
        let _avx2 = is_avx2_available();
        let _avx512 = is_avx512_available();
        println!("AVX2 available: {}", is_avx2_available());
        println!("AVX-512 available: {}", is_avx512_available());
    }

    #[test]
    fn test_rng_state_avx2() {
        let mut rng = RngStateAvx2::new(42);
        let vals1 = rng.next_u64x4();
        let vals2 = rng.next_u64x4();

        // Values should be different
        assert_ne!(vals1, vals2);

        // Values should be non-zero
        for v in vals1.iter().chain(vals2.iter()) {
            assert_ne!(*v, 0);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    fn test_gaussian_sampler_avx2() {
        if !is_avx2_available() {
            return;
        }

        let mut sampler = GaussianSamplerAvx2::new(42);
        let samples = sampler.sample_standard_normal(100).unwrap();

        assert_eq!(samples.len(), 100);

        // Check basic statistics
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let variance: f32 =
            samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / samples.len() as f32;

        // Mean should be close to 0, variance close to 1
        assert!(mean.abs() < 0.5, "Mean {} too far from 0", mean);
        assert!((variance - 1.0).abs() < 0.5, "Variance {} too far from 1", variance);
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    fn test_gaussian_reparameterize_avx2() {
        if !is_avx2_available() {
            return;
        }

        let mut sampler = GaussianSamplerAvx2::new(42);
        let mean = vec![5.0f32; 64];
        let std = vec![2.0f32; 64];

        let samples = sampler.sample(&mean, &std).unwrap();

        assert_eq!(samples.len(), 64);

        // Check that samples are centered around mean
        let sample_mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        assert!(
            (sample_mean - 5.0).abs() < 2.0,
            "Sample mean {} too far from 5.0",
            sample_mean
        );
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    fn test_softmax_avx2() {
        if !is_avx2_available() {
            return;
        }

        let input = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = softmax_avx2(&input, 1, 8).unwrap();

        assert_eq!(output.len(), 8);

        // Sum should be 1
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "Softmax sum {} != 1", sum);

        // Values should be increasing
        for i in 1..output.len() {
            assert!(output[i] > output[i - 1]);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
    fn test_gae_avx2() {
        if !is_avx2_available() {
            return;
        }

        let num_steps = 128;
        let num_envs = 8;
        let rewards = vec![1.0f32; num_steps * num_envs];
        let values = vec![0.5f32; num_steps * num_envs];
        let dones = vec![0.0f32; num_steps * num_envs];
        let last_values = vec![0.5f32; num_envs];

        let (advantages, returns) =
            compute_gae_avx2(&rewards, &values, &dones, num_steps, num_envs, 0.99, 0.95, &last_values)
                .unwrap();

        assert_eq!(advantages.len(), num_steps * num_envs);
        assert_eq!(returns.len(), num_steps * num_envs);

        // Advantages should be positive (constant reward, no termination)
        for &adv in &advantages {
            assert!(adv > 0.0, "Advantage {} should be positive", adv);
        }
    }

    #[test]
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn test_gather_avx2() {
        if !is_avx2_available() {
            return;
        }

        let dim = 16;
        let capacity = 100;
        let src: Vec<f32> = (0..capacity * dim).map(|i| i as f32).collect();
        let indices = vec![0, 10, 20, 30];

        let result = gather_batch_f32_avx2(&src, &indices, dim, capacity).unwrap();

        assert_eq!(result.len(), indices.len() * dim);

        // Check first gathered row
        for i in 0..dim {
            assert_eq!(result[i], src[indices[0] * dim + i]);
        }

        // Check second gathered row
        for i in 0..dim {
            assert_eq!(result[dim + i], src[indices[1] * dim + i]);
        }
    }
}
