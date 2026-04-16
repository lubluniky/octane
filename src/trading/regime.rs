//! Market regime detection for trading environments.
//!
//! Features:
//! - Trend/Range/Volatile regime classification
//! - HMM-based regime detection
//! - Volatility clustering (GARCH-like)
//! - Regime as part of observation space
//! - Regime change callbacks

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Market regime types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[derive(Default)]
pub enum MarketRegime {
    /// Strong upward trend.
    BullTrend,
    /// Strong downward trend.
    BearTrend,
    /// Range-bound/sideways market.
    #[default]
    Range,
    /// High volatility regime.
    HighVolatility,
    /// Low volatility regime.
    LowVolatility,
    /// Transition/uncertain regime.
    Transition,
}

impl MarketRegime {
    /// Get regime index (for one-hot encoding).
    pub fn index(&self) -> usize {
        match self {
            MarketRegime::BullTrend => 0,
            MarketRegime::BearTrend => 1,
            MarketRegime::Range => 2,
            MarketRegime::HighVolatility => 3,
            MarketRegime::LowVolatility => 4,
            MarketRegime::Transition => 5,
        }
    }

    /// Number of regime types.
    pub fn count() -> usize {
        6
    }

    /// Convert from index.
    pub fn from_index(idx: usize) -> Self {
        match idx {
            0 => MarketRegime::BullTrend,
            1 => MarketRegime::BearTrend,
            2 => MarketRegime::Range,
            3 => MarketRegime::HighVolatility,
            4 => MarketRegime::LowVolatility,
            _ => MarketRegime::Transition,
        }
    }

    /// Get display name.
    pub fn name(&self) -> &'static str {
        match self {
            MarketRegime::BullTrend => "Bull Trend",
            MarketRegime::BearTrend => "Bear Trend",
            MarketRegime::Range => "Range",
            MarketRegime::HighVolatility => "High Vol",
            MarketRegime::LowVolatility => "Low Vol",
            MarketRegime::Transition => "Transition",
        }
    }

    /// Get one-hot encoding.
    pub fn to_one_hot(&self) -> Vec<f32> {
        let mut vec = vec![0.0; Self::count()];
        vec[self.index()] = 1.0;
        vec
    }
}


/// Regime transition event.
#[derive(Debug, Clone)]
pub struct RegimeTransition {
    /// Previous regime.
    pub from: MarketRegime,
    /// New regime.
    pub to: MarketRegime,
    /// Timestep when transition occurred.
    pub timestep: usize,
    /// Confidence in the new regime (0-1).
    pub confidence: f32,
}

/// Callback trait for regime change events.
pub trait RegimeCallback: Send + Sync {
    /// Called when regime changes.
    fn on_regime_change(&mut self, transition: &RegimeTransition);
}

/// Simple callback that logs transitions.
#[derive(Default, Clone)]
pub struct LoggingCallback {
    /// Recorded transitions.
    pub transitions: Vec<RegimeTransition>,
}

impl RegimeCallback for LoggingCallback {
    fn on_regime_change(&mut self, transition: &RegimeTransition) {
        self.transitions.push(transition.clone());
    }
}

/// Regime observation for environment state.
#[derive(Debug, Clone, Default)]
pub struct RegimeObservation {
    /// Current detected regime.
    pub current_regime: MarketRegime,
    /// Regime probabilities (from HMM).
    pub regime_probabilities: Vec<f32>,
    /// Estimated volatility.
    pub volatility: f32,
    /// Trend strength (-1 to 1).
    pub trend_strength: f32,
    /// Mean reversion indicator.
    pub mean_reversion: f32,
    /// Regime persistence (how long in current regime).
    pub regime_persistence: usize,
}

impl RegimeObservation {
    /// Convert to feature vector.
    pub fn to_features(&self) -> Vec<f32> {
        let mut features = self.regime_probabilities.clone();
        features.push(self.volatility);
        features.push(self.trend_strength);
        features.push(self.mean_reversion);
        features.push(self.regime_persistence as f32 / 100.0);
        features
    }

    /// Feature dimension.
    pub fn feature_dim() -> usize {
        MarketRegime::count() + 4
    }
}

/// Hidden Markov Model parameters for regime detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HmmParams {
    /// Number of hidden states.
    pub n_states: usize,
    /// Initial state probabilities.
    pub initial_probs: Vec<f32>,
    /// Transition probability matrix (row-major).
    pub transition_matrix: Vec<Vec<f32>>,
    /// Emission means for each state.
    pub emission_means: Vec<f32>,
    /// Emission standard deviations for each state.
    pub emission_stds: Vec<f32>,
}

impl Default for HmmParams {
    fn default() -> Self {
        // 3-state HMM: Low Vol, Normal, High Vol
        Self {
            n_states: 3,
            initial_probs: vec![0.33, 0.34, 0.33],
            transition_matrix: vec![
                vec![0.95, 0.04, 0.01], // Low vol stays low
                vec![0.02, 0.96, 0.02], // Normal stays normal
                vec![0.01, 0.04, 0.95], // High vol stays high
            ],
            emission_means: vec![0.005, 0.015, 0.035], // Volatility levels
            emission_stds: vec![0.002, 0.005, 0.015],
        }
    }
}

impl HmmParams {
    /// Create custom HMM parameters.
    pub fn new(
        n_states: usize,
        initial_probs: Vec<f32>,
        transition_matrix: Vec<Vec<f32>>,
        emission_means: Vec<f32>,
        emission_stds: Vec<f32>,
    ) -> Self {
        Self {
            n_states,
            initial_probs,
            transition_matrix,
            emission_means,
            emission_stds,
        }
    }

    /// Compute emission probability for observation in state.
    pub fn emission_prob(&self, state: usize, observation: f32) -> f32 {
        let mean = self.emission_means[state];
        let std = self.emission_stds[state];

        // Gaussian emission
        let z = (observation - mean) / std;
        (-0.5 * z * z).exp() / (std * (2.0 * std::f32::consts::PI).sqrt())
    }
}

/// GARCH-like parameters for volatility estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GarchParams {
    /// Long-run variance (omega).
    pub omega: f32,
    /// ARCH coefficient (alpha) - weight of past squared returns.
    pub alpha: f32,
    /// GARCH coefficient (beta) - weight of past variance.
    pub beta: f32,
}

impl Default for GarchParams {
    fn default() -> Self {
        Self {
            omega: 0.00001,
            alpha: 0.1,
            beta: 0.85,
        }
    }
}

impl GarchParams {
    /// Create custom GARCH parameters.
    pub fn new(omega: f32, alpha: f32, beta: f32) -> Self {
        Self { omega, alpha, beta }
    }

    /// Update variance estimate.
    pub fn update_variance(&self, prev_variance: f32, squared_return: f32) -> f32 {
        self.omega + self.alpha * squared_return + self.beta * prev_variance
    }

    /// Get unconditional variance.
    pub fn unconditional_variance(&self) -> f32 {
        self.omega / (1.0 - self.alpha - self.beta)
    }
}

/// Configuration for regime detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeConfig {
    /// Window size for regime detection.
    pub window_size: usize,
    /// HMM parameters.
    pub hmm_params: HmmParams,
    /// GARCH parameters.
    pub garch_params: GarchParams,
    /// Trend threshold (absolute return for trend detection).
    pub trend_threshold: f32,
    /// Volatility high threshold.
    pub volatility_high_threshold: f32,
    /// Volatility low threshold.
    pub volatility_low_threshold: f32,
    /// Minimum regime persistence (steps before allowing change).
    pub min_persistence: usize,
    /// Smoothing factor for probability updates.
    pub smoothing: f32,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            hmm_params: HmmParams::default(),
            garch_params: GarchParams::default(),
            trend_threshold: 0.02,
            volatility_high_threshold: 0.03,
            volatility_low_threshold: 0.01,
            min_persistence: 5,
            smoothing: 0.1,
        }
    }
}

impl RegimeConfig {
    /// Set window size.
    pub fn window_size(mut self, size: usize) -> Self {
        self.window_size = size;
        self
    }

    /// Set HMM parameters.
    pub fn hmm_params(mut self, params: HmmParams) -> Self {
        self.hmm_params = params;
        self
    }

    /// Set GARCH parameters.
    pub fn garch_params(mut self, params: GarchParams) -> Self {
        self.garch_params = params;
        self
    }

    /// Set trend threshold.
    pub fn trend_threshold(mut self, threshold: f32) -> Self {
        self.trend_threshold = threshold;
        self
    }

    /// Set volatility thresholds.
    pub fn volatility_thresholds(mut self, low: f32, high: f32) -> Self {
        self.volatility_low_threshold = low;
        self.volatility_high_threshold = high;
        self
    }

    /// Set minimum persistence.
    pub fn min_persistence(mut self, persistence: usize) -> Self {
        self.min_persistence = persistence;
        self
    }
}

/// Market regime detector.
#[derive(Clone)]
pub struct RegimeDetector {
    /// Configuration.
    config: RegimeConfig,
    /// Current regime.
    current_regime: MarketRegime,
    /// Regime probabilities.
    regime_probs: Vec<f32>,
    /// HMM state probabilities (filtered).
    hmm_state_probs: Vec<f32>,
    /// Current GARCH variance estimate.
    garch_variance: f32,
    /// Return history.
    returns: VecDeque<f32>,
    /// Price history.
    prices: VecDeque<f32>,
    /// Regime persistence counter.
    persistence: usize,
    /// Timestep counter.
    timestep: usize,
    /// Random number generator.
    rng: StdRng,
}

impl RegimeDetector {
    /// Create new regime detector.
    pub fn new(config: RegimeConfig) -> Self {
        let n_states = config.hmm_params.n_states;
        let n_regimes = MarketRegime::count();

        Self {
            config,
            current_regime: MarketRegime::Range,
            regime_probs: vec![1.0 / n_regimes as f32; n_regimes],
            hmm_state_probs: vec![1.0 / n_states as f32; n_states],
            garch_variance: 0.0001, // Initial variance
            returns: VecDeque::new(),
            prices: VecDeque::new(),
            persistence: 0,
            timestep: 0,
            rng: StdRng::from_entropy(),
        }
    }

    /// Create with default configuration.
    pub fn default_detector() -> Self {
        Self::new(RegimeConfig::default())
    }

    /// Reset detector state.
    pub fn reset(&mut self) {
        let n_states = self.config.hmm_params.n_states;
        let n_regimes = MarketRegime::count();

        self.current_regime = MarketRegime::Range;
        self.regime_probs = vec![1.0 / n_regimes as f32; n_regimes];
        self.hmm_state_probs = vec![1.0 / n_states as f32; n_states];
        self.garch_variance = 0.0001;
        self.returns.clear();
        self.prices.clear();
        self.persistence = 0;
        self.timestep = 0;
    }

    /// Update with new price and return regime observation.
    pub fn update(&mut self, price: f32) -> RegimeObservation {
        self.prices.push_back(price);

        // Calculate return
        if self.prices.len() > 1 {
            let prev_price = self.prices[self.prices.len() - 2];
            let ret = (price - prev_price) / prev_price;
            self.returns.push_back(ret);
        }

        // Maintain window size
        while self.prices.len() > self.config.window_size + 1 {
            self.prices.pop_front();
        }
        while self.returns.len() > self.config.window_size {
            self.returns.pop_front();
        }

        // Update GARCH variance
        if let Some(&last_return) = self.returns.back() {
            self.garch_variance = self
                .config
                .garch_params
                .update_variance(self.garch_variance, last_return * last_return);
        }

        // Update HMM state probabilities
        self.update_hmm();

        // Calculate indicators
        let volatility = self.garch_variance.sqrt();
        let trend_strength = self.calculate_trend_strength();
        let mean_reversion = self.calculate_mean_reversion();

        // Determine regime
        let new_regime = self.classify_regime(volatility, trend_strength);

        // Update regime with persistence check
        if new_regime != self.current_regime {
            self.persistence += 1;
            if self.persistence >= self.config.min_persistence {
                self.current_regime = new_regime;
                self.persistence = 0;
            }
        } else {
            self.persistence = 0;
        }

        // Update regime probabilities
        self.update_regime_probabilities(volatility, trend_strength);

        self.timestep += 1;

        RegimeObservation {
            current_regime: self.current_regime,
            regime_probabilities: self.regime_probs.clone(),
            volatility,
            trend_strength,
            mean_reversion,
            regime_persistence: self.persistence,
        }
    }

    /// Update HMM state probabilities using forward algorithm.
    fn update_hmm(&mut self) {
        if self.returns.is_empty() {
            return;
        }

        let obs = self.garch_variance.sqrt(); // Use volatility as observation
        let hmm = &self.config.hmm_params;

        // Compute emission probabilities
        let emissions: Vec<f32> = (0..hmm.n_states)
            .map(|s| hmm.emission_prob(s, obs))
            .collect();

        // Forward step: predict then update
        let mut new_probs = vec![0.0; hmm.n_states];

        for j in 0..hmm.n_states {
            let predict: f32 = (0..hmm.n_states)
                .map(|i| self.hmm_state_probs[i] * hmm.transition_matrix[i][j])
                .sum();
            new_probs[j] = predict * emissions[j];
        }

        // Normalize
        let sum: f32 = new_probs.iter().sum();
        if sum > 0.0 {
            for p in &mut new_probs {
                *p /= sum;
            }
            self.hmm_state_probs = new_probs;
        }
    }

    /// Calculate trend strength from returns.
    fn calculate_trend_strength(&self) -> f32 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let window = self.config.window_size.min(self.returns.len());
        let recent_returns: Vec<f32> = self
            .returns
            .iter()
            .rev()
            .take(window)
            .copied()
            .collect();

        // Simple trend: sum of returns
        let sum: f32 = recent_returns.iter().sum();

        // Normalize to [-1, 1]
        (sum / self.config.trend_threshold).clamp(-1.0, 1.0)
    }

    /// Calculate mean reversion indicator.
    fn calculate_mean_reversion(&self) -> f32 {
        if self.prices.len() < 2 {
            return 0.0;
        }

        let current_price = *self.prices.back().unwrap();
        let avg_price: f32 = self.prices.iter().sum::<f32>() / self.prices.len() as f32;

        // How far from mean (normalized)
        ((current_price - avg_price) / avg_price).clamp(-1.0, 1.0)
    }

    /// Classify current regime based on indicators.
    fn classify_regime(&self, volatility: f32, trend_strength: f32) -> MarketRegime {
        // Volatility-based classification
        if volatility > self.config.volatility_high_threshold {
            return MarketRegime::HighVolatility;
        }

        if volatility < self.config.volatility_low_threshold {
            return MarketRegime::LowVolatility;
        }

        // Trend-based classification
        if trend_strength > 0.5 {
            return MarketRegime::BullTrend;
        }

        if trend_strength < -0.5 {
            return MarketRegime::BearTrend;
        }

        // Check for transition using HMM uncertainty
        let max_prob = self
            .hmm_state_probs
            .iter()
            .fold(0.0f32, |a, &b| a.max(b));
        if max_prob < 0.5 {
            return MarketRegime::Transition;
        }

        MarketRegime::Range
    }

    /// Update regime probability distribution.
    fn update_regime_probabilities(&mut self, volatility: f32, trend_strength: f32) {
        let mut probs = vec![0.0; MarketRegime::count()];

        // Base probabilities from indicators
        let vol_high_prob =
            (volatility / self.config.volatility_high_threshold).clamp(0.0, 1.0);
        let vol_low_prob =
            (1.0 - volatility / self.config.volatility_low_threshold).clamp(0.0, 1.0);

        probs[MarketRegime::HighVolatility.index()] = vol_high_prob * 0.5;
        probs[MarketRegime::LowVolatility.index()] = vol_low_prob * 0.5;

        // Trend probabilities
        if trend_strength > 0.0 {
            probs[MarketRegime::BullTrend.index()] = trend_strength;
        } else {
            probs[MarketRegime::BearTrend.index()] = -trend_strength;
        }

        // Range probability (inverse of trend strength)
        probs[MarketRegime::Range.index()] = 1.0 - trend_strength.abs();

        // Transition probability based on HMM uncertainty
        let hmm_uncertainty = 1.0
            - self
                .hmm_state_probs
                .iter()
                .fold(0.0f32, |a, &b| a.max(b));
        probs[MarketRegime::Transition.index()] = hmm_uncertainty;

        // Normalize
        let sum: f32 = probs.iter().sum();
        if sum > 0.0 {
            for p in &mut probs {
                *p /= sum;
            }
        }

        // Apply smoothing
        let alpha = self.config.smoothing;
        for (i, p) in probs.iter().enumerate() {
            self.regime_probs[i] = (1.0 - alpha) * self.regime_probs[i] + alpha * p;
        }
    }

    /// Get current regime.
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get regime probabilities.
    pub fn regime_probabilities(&self) -> &[f32] {
        &self.regime_probs
    }

    /// Get current volatility estimate.
    pub fn volatility(&self) -> f32 {
        self.garch_variance.sqrt()
    }

    /// Get full observation.
    pub fn observation(&self) -> RegimeObservation {
        RegimeObservation {
            current_regime: self.current_regime,
            regime_probabilities: self.regime_probs.clone(),
            volatility: self.volatility(),
            trend_strength: self.calculate_trend_strength(),
            mean_reversion: self.calculate_mean_reversion(),
            regime_persistence: self.persistence,
        }
    }

    /// Generate synthetic regime sequence for testing.
    pub fn generate_synthetic_sequence(
        &mut self,
        length: usize,
        regime_duration_mean: usize,
    ) -> Vec<(f32, MarketRegime)> {
        let mut result = Vec::with_capacity(length);
        let mut price = 100.0f32;
        let mut current_regime = MarketRegime::Range;
        let mut regime_steps = 0;
        let mut regime_duration = regime_duration_mean;

        for _ in 0..length {
            // Check for regime change
            regime_steps += 1;
            if regime_steps >= regime_duration {
                // Transition to new regime
                let new_regime_idx = self.rng.gen_range(0..MarketRegime::count());
                current_regime = MarketRegime::from_index(new_regime_idx);
                regime_duration =
                    (regime_duration_mean as f32 * (0.5 + self.rng.gen::<f32>())) as usize;
                regime_steps = 0;
            }

            // Generate price based on regime
            let (drift, vol) = match current_regime {
                MarketRegime::BullTrend => (0.001, 0.01),
                MarketRegime::BearTrend => (-0.001, 0.01),
                MarketRegime::Range => (0.0, 0.005),
                MarketRegime::HighVolatility => (0.0, 0.03),
                MarketRegime::LowVolatility => (0.0, 0.002),
                MarketRegime::Transition => (0.0, 0.015),
            };

            let returns = drift + self.rng.gen::<f32>() * vol * 2.0 - vol;
            price *= 1.0 + returns;

            result.push((price, current_regime));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_indices() {
        for i in 0..MarketRegime::count() {
            let regime = MarketRegime::from_index(i);
            assert_eq!(regime.index(), i);
        }
    }

    #[test]
    fn test_one_hot_encoding() {
        let regime = MarketRegime::BullTrend;
        let one_hot = regime.to_one_hot();
        assert_eq!(one_hot.len(), MarketRegime::count());
        assert_eq!(one_hot[regime.index()], 1.0);
        assert_eq!(one_hot.iter().filter(|&&x| x == 1.0).count(), 1);
    }

    #[test]
    fn test_garch_variance() {
        let garch = GarchParams::default();
        let var1 = garch.update_variance(0.0001, 0.001);
        let var2 = garch.update_variance(var1, 0.002);

        assert!(var1 > 0.0);
        assert!(var2 > 0.0);
    }

    #[test]
    fn test_regime_detector() {
        let mut detector = RegimeDetector::default_detector();

        // Feed some prices
        let prices: Vec<f32> = (0..100)
            .map(|i| 100.0 + (i as f32 * 0.01).sin() * 5.0)
            .collect();

        for price in prices {
            let obs = detector.update(price);
            assert!(obs.volatility >= 0.0);
            assert!(obs.trend_strength >= -1.0 && obs.trend_strength <= 1.0);
        }
    }

    #[test]
    fn test_hmm_emission() {
        let hmm = HmmParams::default();
        let prob = hmm.emission_prob(0, 0.005);
        assert!(prob > 0.0);
        assert!(prob.is_finite());
    }

    #[test]
    fn test_synthetic_sequence() {
        let mut detector = RegimeDetector::default_detector();
        let sequence = detector.generate_synthetic_sequence(100, 20);

        assert_eq!(sequence.len(), 100);
        for (price, _regime) in &sequence {
            assert!(price.is_finite() && *price > 0.0);
        }
    }
}
