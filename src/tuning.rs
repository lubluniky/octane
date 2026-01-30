//! Hyperparameter tuning for Octane.
//!
//! This module provides hyperparameter tuning capabilities including:
//!
//! - Hyperparameter space definitions (uniform, log-uniform, categorical)
//! - Trial management with suggestion methods
//! - Grid search and random search samplers
//! - Optional Optuna integration via PyO3
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::tuning::{HyperparameterSpace, RandomSearch, Trial};
//!
//! let space = HyperparameterSpace::new()
//!     .add_float("learning_rate", 1e-5, 1e-2, true)  // log-uniform
//!     .add_float("gamma", 0.9, 0.999, false)          // uniform
//!     .add_int("batch_size", 32, 256)
//!     .add_categorical("activation", vec!["tanh", "relu"]);
//!
//! let sampler = RandomSearch::new(space, 42);
//!
//! for trial_id in 0..100 {
//!     let trial = sampler.suggest(trial_id);
//!     let lr = trial.get_float("learning_rate")?;
//!     let batch_size = trial.get_int("batch_size")?;
//!     // Train and evaluate...
//!     sampler.report(trial_id, reward);
//! }
//! ```

use crate::core::{OctaneError, Result};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Definition of a hyperparameter search space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyperparameterSpace {
    /// Float parameters.
    float_params: HashMap<String, FloatParam>,
    /// Integer parameters.
    int_params: HashMap<String, IntParam>,
    /// Categorical parameters.
    categorical_params: HashMap<String, CategoricalParam>,
    /// Parameter ordering for deterministic iteration.
    param_order: Vec<String>,
}

/// Float hyperparameter definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatParam {
    /// Parameter name.
    pub name: String,
    /// Lower bound.
    pub low: f64,
    /// Upper bound.
    pub high: f64,
    /// Use log-uniform sampling.
    pub log_scale: bool,
    /// Default value (optional).
    pub default: Option<f64>,
}

/// Integer hyperparameter definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntParam {
    /// Parameter name.
    pub name: String,
    /// Lower bound (inclusive).
    pub low: i64,
    /// Upper bound (inclusive).
    pub high: i64,
    /// Use log-uniform sampling.
    pub log_scale: bool,
    /// Step size (for grid search).
    pub step: Option<i64>,
    /// Default value (optional).
    pub default: Option<i64>,
}

/// Categorical hyperparameter definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalParam {
    /// Parameter name.
    pub name: String,
    /// Possible choices.
    pub choices: Vec<String>,
    /// Default value (optional).
    pub default: Option<String>,
}

impl HyperparameterSpace {
    /// Create a new empty hyperparameter space.
    pub fn new() -> Self {
        Self {
            float_params: HashMap::new(),
            int_params: HashMap::new(),
            categorical_params: HashMap::new(),
            param_order: Vec::new(),
        }
    }

    /// Add a float parameter with uniform sampling.
    pub fn add_float(
        mut self,
        name: impl Into<String>,
        low: f64,
        high: f64,
        log_scale: bool,
    ) -> Self {
        let name = name.into();
        self.param_order.push(name.clone());
        self.float_params.insert(
            name.clone(),
            FloatParam {
                name,
                low,
                high,
                log_scale,
                default: None,
            },
        );
        self
    }

    /// Add a float parameter with a default value.
    pub fn add_float_with_default(
        mut self,
        name: impl Into<String>,
        low: f64,
        high: f64,
        log_scale: bool,
        default: f64,
    ) -> Self {
        let name = name.into();
        self.param_order.push(name.clone());
        self.float_params.insert(
            name.clone(),
            FloatParam {
                name,
                low,
                high,
                log_scale,
                default: Some(default),
            },
        );
        self
    }

    /// Add an integer parameter.
    pub fn add_int(mut self, name: impl Into<String>, low: i64, high: i64) -> Self {
        let name = name.into();
        self.param_order.push(name.clone());
        self.int_params.insert(
            name.clone(),
            IntParam {
                name,
                low,
                high,
                log_scale: false,
                step: None,
                default: None,
            },
        );
        self
    }

    /// Add an integer parameter with log scale.
    pub fn add_int_log(mut self, name: impl Into<String>, low: i64, high: i64) -> Self {
        let name = name.into();
        self.param_order.push(name.clone());
        self.int_params.insert(
            name.clone(),
            IntParam {
                name,
                low,
                high,
                log_scale: true,
                step: None,
                default: None,
            },
        );
        self
    }

    /// Add an integer parameter with step size.
    pub fn add_int_with_step(
        mut self,
        name: impl Into<String>,
        low: i64,
        high: i64,
        step: i64,
    ) -> Self {
        let name = name.into();
        self.param_order.push(name.clone());
        self.int_params.insert(
            name.clone(),
            IntParam {
                name,
                low,
                high,
                log_scale: false,
                step: Some(step),
                default: None,
            },
        );
        self
    }

    /// Add a categorical parameter.
    pub fn add_categorical<S: Into<String>>(
        mut self,
        name: impl Into<String>,
        choices: Vec<S>,
    ) -> Self {
        let name = name.into();
        self.param_order.push(name.clone());
        self.categorical_params.insert(
            name.clone(),
            CategoricalParam {
                name,
                choices: choices.into_iter().map(|c| c.into()).collect(),
                default: None,
            },
        );
        self
    }

    /// Add a categorical parameter with default.
    pub fn add_categorical_with_default<S: Into<String>>(
        mut self,
        name: impl Into<String>,
        choices: Vec<S>,
        default: impl Into<String>,
    ) -> Self {
        let name = name.into();
        self.param_order.push(name.clone());
        self.categorical_params.insert(
            name.clone(),
            CategoricalParam {
                name,
                choices: choices.into_iter().map(|c| c.into()).collect(),
                default: Some(default.into()),
            },
        );
        self
    }

    /// Get all parameter names.
    pub fn param_names(&self) -> &[String] {
        &self.param_order
    }

    /// Get float parameter definition.
    pub fn get_float(&self, name: &str) -> Option<&FloatParam> {
        self.float_params.get(name)
    }

    /// Get int parameter definition.
    pub fn get_int(&self, name: &str) -> Option<&IntParam> {
        self.int_params.get(name)
    }

    /// Get categorical parameter definition.
    pub fn get_categorical(&self, name: &str) -> Option<&CategoricalParam> {
        self.categorical_params.get(name)
    }

    /// Calculate total number of configurations for grid search.
    pub fn grid_size(&self) -> usize {
        let mut size = 1usize;

        for param in self.categorical_params.values() {
            size = size.saturating_mul(param.choices.len());
        }

        // For continuous params, we'd need a discretization
        // This is mainly useful for pure categorical/int spaces
        for param in self.int_params.values() {
            let step = param.step.unwrap_or(1);
            let count = ((param.high - param.low) / step + 1) as usize;
            size = size.saturating_mul(count);
        }

        size
    }

    /// Check if space is valid.
    pub fn validate(&self) -> Result<()> {
        for param in self.float_params.values() {
            if param.low >= param.high {
                return Err(OctaneError::InvalidConfig(format!(
                    "Float param {}: low >= high",
                    param.name
                )));
            }
            if param.log_scale && param.low <= 0.0 {
                return Err(OctaneError::InvalidConfig(format!(
                    "Float param {} with log_scale must have low > 0",
                    param.name
                )));
            }
        }

        for param in self.int_params.values() {
            if param.low > param.high {
                return Err(OctaneError::InvalidConfig(format!(
                    "Int param {}: low > high",
                    param.name
                )));
            }
            if param.log_scale && param.low <= 0 {
                return Err(OctaneError::InvalidConfig(format!(
                    "Int param {} with log_scale must have low > 0",
                    param.name
                )));
            }
        }

        for param in self.categorical_params.values() {
            if param.choices.is_empty() {
                return Err(OctaneError::InvalidConfig(format!(
                    "Categorical param {} must have at least one choice",
                    param.name
                )));
            }
        }

        Ok(())
    }
}

impl Default for HyperparameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

/// A single trial with sampled hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trial {
    /// Unique trial identifier.
    pub trial_id: usize,

    /// Sampled float values.
    pub float_values: HashMap<String, f64>,

    /// Sampled integer values.
    pub int_values: HashMap<String, i64>,

    /// Sampled categorical values.
    pub categorical_values: HashMap<String, String>,

    /// Trial state.
    pub state: TrialState,

    /// Result value (e.g., reward or loss).
    pub value: Option<f64>,

    /// Intermediate values for pruning.
    pub intermediate_values: Vec<(usize, f64)>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,

    /// Start timestamp.
    pub start_time: Option<u64>,

    /// End timestamp.
    pub end_time: Option<u64>,
}

/// State of a trial.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum TrialState {
    /// Trial is running.
    #[default]
    Running,
    /// Trial completed successfully.
    Complete,
    /// Trial was pruned early.
    Pruned,
    /// Trial failed with an error.
    Failed,
}

impl Trial {
    /// Create a new trial.
    pub fn new(trial_id: usize) -> Self {
        Self {
            trial_id,
            float_values: HashMap::new(),
            int_values: HashMap::new(),
            categorical_values: HashMap::new(),
            state: TrialState::Running,
            value: None,
            intermediate_values: Vec::new(),
            metadata: HashMap::new(),
            start_time: Some(current_timestamp()),
            end_time: None,
        }
    }

    /// Get a float parameter value.
    pub fn get_float(&self, name: &str) -> Result<f64> {
        self.float_values
            .get(name)
            .copied()
            .ok_or_else(|| OctaneError::InvalidConfig(format!("Float param {} not found", name)))
    }

    /// Get an integer parameter value.
    pub fn get_int(&self, name: &str) -> Result<i64> {
        self.int_values
            .get(name)
            .copied()
            .ok_or_else(|| OctaneError::InvalidConfig(format!("Int param {} not found", name)))
    }

    /// Get a categorical parameter value.
    pub fn get_categorical(&self, name: &str) -> Result<&str> {
        self.categorical_values
            .get(name)
            .map(|s| s.as_str())
            .ok_or_else(|| {
                OctaneError::InvalidConfig(format!("Categorical param {} not found", name))
            })
    }

    /// Set a float parameter value.
    pub fn set_float(&mut self, name: impl Into<String>, value: f64) {
        self.float_values.insert(name.into(), value);
    }

    /// Set an integer parameter value.
    pub fn set_int(&mut self, name: impl Into<String>, value: i64) {
        self.int_values.insert(name.into(), value);
    }

    /// Set a categorical parameter value.
    pub fn set_categorical(&mut self, name: impl Into<String>, value: impl Into<String>) {
        self.categorical_values.insert(name.into(), value.into());
    }

    /// Report an intermediate value (for pruning).
    pub fn report_intermediate(&mut self, step: usize, value: f64) {
        self.intermediate_values.push((step, value));
    }

    /// Complete the trial with a final value.
    pub fn complete(&mut self, value: f64) {
        self.value = Some(value);
        self.state = TrialState::Complete;
        self.end_time = Some(current_timestamp());
    }

    /// Mark trial as pruned.
    pub fn prune(&mut self) {
        self.state = TrialState::Pruned;
        self.end_time = Some(current_timestamp());
    }

    /// Mark trial as failed.
    pub fn fail(&mut self) {
        self.state = TrialState::Failed;
        self.end_time = Some(current_timestamp());
    }

    /// Get trial duration in seconds.
    pub fn duration_secs(&self) -> Option<u64> {
        match (self.start_time, self.end_time) {
            (Some(start), Some(end)) => Some(end.saturating_sub(start)),
            _ => None,
        }
    }

    /// Convert to a simple parameter dictionary.
    pub fn to_params(&self) -> HashMap<String, ParamValue> {
        let mut params = HashMap::new();

        for (k, v) in &self.float_values {
            params.insert(k.clone(), ParamValue::Float(*v));
        }
        for (k, v) in &self.int_values {
            params.insert(k.clone(), ParamValue::Int(*v));
        }
        for (k, v) in &self.categorical_values {
            params.insert(k.clone(), ParamValue::Categorical(v.clone()));
        }

        params
    }
}

/// A parameter value (for type-erased access).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamValue {
    /// Float value.
    Float(f64),
    /// Integer value.
    Int(i64),
    /// Categorical value.
    Categorical(String),
}

impl ParamValue {
    /// Get as float.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParamValue::Float(v) => Some(*v),
            ParamValue::Int(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// Get as int.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParamValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Get as string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParamValue::Categorical(v) => Some(v),
            _ => None,
        }
    }
}

/// Sampler trait for hyperparameter search strategies.
pub trait Sampler: Send + Sync {
    /// Sample a new trial.
    fn sample(&mut self, trial_id: usize, space: &HyperparameterSpace) -> Trial;

    /// Report a trial result.
    fn report(&mut self, trial: &Trial);

    /// Get the best trial so far.
    fn best_trial(&self) -> Option<&Trial>;

    /// Get all completed trials.
    fn trials(&self) -> &[Trial];

    /// Check if a trial should be pruned.
    fn should_prune(&self, _trial: &Trial) -> bool {
        false
    }
}

/// Random search sampler.
#[derive(Debug)]
pub struct RandomSearch {
    /// Hyperparameter space.
    space: HyperparameterSpace,
    /// Random number generator.
    rng: StdRng,
    /// Completed trials.
    trials: Vec<Trial>,
    /// Best trial index.
    best_trial_idx: Option<usize>,
    /// Direction of optimization.
    direction: OptimizationDirection,
}

/// Direction of optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OptimizationDirection {
    /// Maximize the objective.
    #[default]
    Maximize,
    /// Minimize the objective.
    Minimize,
}

impl RandomSearch {
    /// Create a new random search sampler.
    pub fn new(space: HyperparameterSpace, seed: u64) -> Self {
        Self {
            space,
            rng: StdRng::seed_from_u64(seed),
            trials: Vec::new(),
            best_trial_idx: None,
            direction: OptimizationDirection::Maximize,
        }
    }

    /// Set optimization direction.
    pub fn direction(mut self, direction: OptimizationDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Get the hyperparameter space.
    pub fn space(&self) -> &HyperparameterSpace {
        &self.space
    }

    /// Suggest a new trial.
    pub fn suggest(&mut self, trial_id: usize) -> Trial {
        self.sample(trial_id, &self.space.clone())
    }

    fn sample_float(&mut self, param: &FloatParam) -> f64 {
        if param.log_scale {
            let log_low = param.low.ln();
            let log_high = param.high.ln();
            let log_val = self.rng.gen_range(log_low..log_high);
            log_val.exp()
        } else {
            self.rng.gen_range(param.low..param.high)
        }
    }

    fn sample_int(&mut self, param: &IntParam) -> i64 {
        if param.log_scale {
            let log_low = (param.low as f64).ln();
            let log_high = (param.high as f64).ln();
            let log_val = self.rng.gen_range(log_low..log_high);
            log_val.exp().round() as i64
        } else if let Some(step) = param.step {
            let n_steps = (param.high - param.low) / step;
            let step_idx = self.rng.gen_range(0..=n_steps);
            param.low + step_idx * step
        } else {
            self.rng.gen_range(param.low..=param.high)
        }
    }

    fn sample_categorical(&mut self, param: &CategoricalParam) -> String {
        let idx = self.rng.gen_range(0..param.choices.len());
        param.choices[idx].clone()
    }

    fn is_better(&self, a: f64, b: f64) -> bool {
        match self.direction {
            OptimizationDirection::Maximize => a > b,
            OptimizationDirection::Minimize => a < b,
        }
    }
}

impl Sampler for RandomSearch {
    fn sample(&mut self, trial_id: usize, space: &HyperparameterSpace) -> Trial {
        let mut trial = Trial::new(trial_id);

        for (name, param) in &space.float_params {
            let value = self.sample_float(param);
            trial.set_float(name, value);
        }

        for (name, param) in &space.int_params {
            let value = self.sample_int(param);
            trial.set_int(name, value);
        }

        for (name, param) in &space.categorical_params {
            let value = self.sample_categorical(param);
            trial.set_categorical(name, value);
        }

        trial
    }

    fn report(&mut self, trial: &Trial) {
        self.trials.push(trial.clone());

        if let Some(value) = trial.value {
            let is_best = match self.best_trial_idx {
                None => true,
                Some(idx) => {
                    let best_value = self.trials[idx].value.unwrap_or(f64::NEG_INFINITY);
                    self.is_better(value, best_value)
                }
            };

            if is_best {
                self.best_trial_idx = Some(self.trials.len() - 1);
            }
        }
    }

    fn best_trial(&self) -> Option<&Trial> {
        self.best_trial_idx.map(|idx| &self.trials[idx])
    }

    fn trials(&self) -> &[Trial] {
        &self.trials
    }
}

/// Grid search sampler.
#[derive(Debug)]
pub struct GridSearch {
    /// Hyperparameter space.
    space: HyperparameterSpace,
    /// Grid configuration.
    grid_config: GridConfig,
    /// Current grid index.
    current_idx: usize,
    /// Completed trials.
    trials: Vec<Trial>,
    /// Best trial index.
    best_trial_idx: Option<usize>,
    /// Direction of optimization.
    direction: OptimizationDirection,
}

/// Configuration for grid search discretization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Number of points for each float parameter.
    pub float_steps: HashMap<String, usize>,
    /// Default number of steps for float parameters.
    pub default_float_steps: usize,
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            float_steps: HashMap::new(),
            default_float_steps: 5,
        }
    }
}

impl GridSearch {
    /// Create a new grid search sampler.
    pub fn new(space: HyperparameterSpace) -> Self {
        Self {
            space,
            grid_config: GridConfig::default(),
            current_idx: 0,
            trials: Vec::new(),
            best_trial_idx: None,
            direction: OptimizationDirection::Maximize,
        }
    }

    /// Set grid configuration.
    pub fn grid_config(mut self, config: GridConfig) -> Self {
        self.grid_config = config;
        self
    }

    /// Set optimization direction.
    pub fn direction(mut self, direction: OptimizationDirection) -> Self {
        self.direction = direction;
        self
    }

    /// Get total number of configurations.
    pub fn total_configurations(&self) -> usize {
        let mut count = 1;

        for (name, _param) in &self.space.float_params {
            let steps = self
                .grid_config
                .float_steps
                .get(name)
                .copied()
                .unwrap_or(self.grid_config.default_float_steps);
            count *= steps;
        }

        for param in self.space.int_params.values() {
            let step = param.step.unwrap_or(1);
            count *= ((param.high - param.low) / step + 1) as usize;
        }

        for param in self.space.categorical_params.values() {
            count *= param.choices.len();
        }

        count
    }

    /// Get configuration at a specific index.
    #[allow(unused_assignments)]
    fn get_configuration(&self, idx: usize) -> Trial {
        let mut trial = Trial::new(idx);
        let mut divisor = 1;

        // Float parameters
        for (name, param) in &self.space.float_params {
            let steps = self
                .grid_config
                .float_steps
                .get(name)
                .copied()
                .unwrap_or(self.grid_config.default_float_steps);

            let step_idx = (idx / divisor) % steps;
            let value = if param.log_scale {
                let log_low = param.low.ln();
                let log_high = param.high.ln();
                let t = step_idx as f64 / (steps - 1).max(1) as f64;
                (log_low + t * (log_high - log_low)).exp()
            } else {
                let t = step_idx as f64 / (steps - 1).max(1) as f64;
                param.low + t * (param.high - param.low)
            };

            trial.set_float(name, value);
            divisor *= steps;
        }

        // Integer parameters
        for (name, param) in &self.space.int_params {
            let step = param.step.unwrap_or(1);
            let num_values = ((param.high - param.low) / step + 1) as usize;
            let step_idx = (idx / divisor) % num_values;
            let value = param.low + (step_idx as i64) * step;

            trial.set_int(name, value);
            divisor *= num_values;
        }

        // Categorical parameters
        for (name, param) in &self.space.categorical_params {
            let choice_idx = (idx / divisor) % param.choices.len();
            trial.set_categorical(name, param.choices[choice_idx].clone());
            divisor *= param.choices.len();
        }

        trial
    }

    fn is_better(&self, a: f64, b: f64) -> bool {
        match self.direction {
            OptimizationDirection::Maximize => a > b,
            OptimizationDirection::Minimize => a < b,
        }
    }
}

impl Sampler for GridSearch {
    fn sample(&mut self, trial_id: usize, _space: &HyperparameterSpace) -> Trial {
        let trial = self.get_configuration(self.current_idx);
        self.current_idx += 1;
        Trial { trial_id, ..trial }
    }

    fn report(&mut self, trial: &Trial) {
        self.trials.push(trial.clone());

        if let Some(value) = trial.value {
            let is_best = match self.best_trial_idx {
                None => true,
                Some(idx) => {
                    let best_value = self.trials[idx].value.unwrap_or(f64::NEG_INFINITY);
                    self.is_better(value, best_value)
                }
            };

            if is_best {
                self.best_trial_idx = Some(self.trials.len() - 1);
            }
        }
    }

    fn best_trial(&self) -> Option<&Trial> {
        self.best_trial_idx.map(|idx| &self.trials[idx])
    }

    fn trials(&self) -> &[Trial] {
        &self.trials
    }
}

/// Study for managing hyperparameter optimization.
#[derive(Debug)]
pub struct Study<S: Sampler> {
    /// Study name.
    name: String,
    /// Sampler.
    sampler: S,
    /// Hyperparameter space.
    space: HyperparameterSpace,
    /// Maximum number of trials.
    max_trials: Option<usize>,
    /// Current trial count.
    trial_count: usize,
}

impl<S: Sampler> Study<S> {
    /// Create a new study.
    pub fn new(name: impl Into<String>, space: HyperparameterSpace, sampler: S) -> Self {
        Self {
            name: name.into(),
            sampler,
            space,
            max_trials: None,
            trial_count: 0,
        }
    }

    /// Set maximum number of trials.
    pub fn max_trials(mut self, n: usize) -> Self {
        self.max_trials = Some(n);
        self
    }

    /// Get study name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Ask for a new trial.
    pub fn ask(&mut self) -> Option<Trial> {
        if let Some(max) = self.max_trials {
            if self.trial_count >= max {
                return None;
            }
        }

        let trial = self.sampler.sample(self.trial_count, &self.space);
        self.trial_count += 1;
        Some(trial)
    }

    /// Tell the study about a completed trial.
    pub fn tell(&mut self, trial: Trial) {
        self.sampler.report(&trial);
    }

    /// Run optimization with a given objective function.
    pub fn optimize<F>(&mut self, mut objective: F) -> Option<&Trial>
    where
        F: FnMut(&Trial) -> f64,
    {
        while let Some(mut trial) = self.ask() {
            let value = objective(&trial);
            trial.complete(value);
            self.tell(trial);
        }

        self.best_trial()
    }

    /// Get the best trial.
    pub fn best_trial(&self) -> Option<&Trial> {
        self.sampler.best_trial()
    }

    /// Get all trials.
    pub fn trials(&self) -> &[Trial] {
        self.sampler.trials()
    }

    /// Get number of completed trials.
    pub fn n_trials(&self) -> usize {
        self.sampler.trials().len()
    }
}

/// Helper function to get current timestamp.
fn current_timestamp() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Create a default PPO hyperparameter space.
pub fn ppo_default_space() -> HyperparameterSpace {
    HyperparameterSpace::new()
        .add_float("learning_rate", 1e-5, 1e-2, true)
        .add_float("gamma", 0.9, 0.9999, false)
        .add_float("gae_lambda", 0.8, 0.99, false)
        .add_float("clip_range", 0.1, 0.4, false)
        .add_float("ent_coef", 0.0, 0.1, false)
        .add_float("vf_coef", 0.25, 1.0, false)
        .add_int_with_step("n_steps", 128, 4096, 128)
        .add_int_with_step("batch_size", 32, 512, 32)
        .add_int("n_epochs", 3, 30)
}

/// Create a default SAC hyperparameter space.
pub fn sac_default_space() -> HyperparameterSpace {
    HyperparameterSpace::new()
        .add_float("learning_rate", 1e-5, 1e-2, true)
        .add_float("gamma", 0.9, 0.9999, false)
        .add_float("tau", 0.001, 0.1, true)
        .add_float("ent_coef", 0.01, 1.0, true)
        .add_int_with_step("batch_size", 64, 512, 64)
        .add_int_log("buffer_size", 10000, 1000000)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperparameter_space() {
        let space = HyperparameterSpace::new()
            .add_float("lr", 0.0001, 0.1, true)
            .add_int("batch_size", 32, 256)
            .add_categorical("activation", vec!["tanh", "relu"]);

        assert!(space.validate().is_ok());
        assert_eq!(space.param_names().len(), 3);
    }

    #[test]
    fn test_invalid_space() {
        let space = HyperparameterSpace::new().add_float("lr", 0.1, 0.01, false);
        assert!(space.validate().is_err());

        let space = HyperparameterSpace::new().add_float("lr", -0.1, 0.1, true);
        assert!(space.validate().is_err());

        let space = HyperparameterSpace::new().add_categorical::<String>("act", vec![]);
        assert!(space.validate().is_err());
    }

    #[test]
    fn test_trial() {
        let mut trial = Trial::new(0);
        trial.set_float("lr", 0.001);
        trial.set_int("batch_size", 64);
        trial.set_categorical("activation", "relu");

        assert!((trial.get_float("lr").unwrap() - 0.001).abs() < 1e-9);
        assert_eq!(trial.get_int("batch_size").unwrap(), 64);
        assert_eq!(trial.get_categorical("activation").unwrap(), "relu");

        trial.complete(100.0);
        assert_eq!(trial.value, Some(100.0));
        assert_eq!(trial.state, TrialState::Complete);
    }

    #[test]
    fn test_random_search() {
        let space = HyperparameterSpace::new()
            .add_float("lr", 0.0001, 0.1, true)
            .add_int("batch_size", 32, 256)
            .add_categorical("activation", vec!["tanh", "relu"]);

        let mut sampler = RandomSearch::new(space, 42);

        for i in 0..10 {
            let mut trial = sampler.suggest(i);

            let lr = trial.get_float("lr").unwrap();
            assert!(lr >= 0.0001 && lr <= 0.1);

            let batch_size = trial.get_int("batch_size").unwrap();
            assert!(batch_size >= 32 && batch_size <= 256);

            let activation = trial.get_categorical("activation").unwrap();
            assert!(activation == "tanh" || activation == "relu");

            trial.complete(lr * batch_size as f64);
            sampler.report(&trial);
        }

        assert_eq!(sampler.trials().len(), 10);
        assert!(sampler.best_trial().is_some());
    }

    #[test]
    fn test_grid_search() {
        let space = HyperparameterSpace::new()
            .add_categorical("a", vec!["x", "y"])
            .add_categorical("b", vec!["1", "2", "3"]);

        let mut sampler = GridSearch::new(space);
        assert_eq!(sampler.total_configurations(), 6);

        for i in 0..6 {
            let mut trial = sampler.sample(i, &HyperparameterSpace::new());
            trial.complete(i as f64);
            sampler.report(&trial);
        }

        assert_eq!(sampler.trials().len(), 6);
    }

    #[test]
    fn test_study() {
        let space = HyperparameterSpace::new()
            .add_float("x", -10.0, 10.0, false)
            .add_float("y", -10.0, 10.0, false);

        let sampler = RandomSearch::new(space.clone(), 42);
        let mut study = Study::new("test_study", space, sampler).max_trials(50);

        // Optimize x^2 + y^2 (minimize)
        let best = study.optimize(|trial| {
            let x = trial.get_float("x").unwrap();
            let y = trial.get_float("y").unwrap();
            -(x * x + y * y) // Negate because default is maximize
        });

        assert!(best.is_some());
        assert_eq!(study.n_trials(), 50);
    }

    #[test]
    fn test_param_value() {
        let float_val = ParamValue::Float(3.14);
        assert!((float_val.as_float().unwrap() - 3.14).abs() < 1e-9);

        let int_val = ParamValue::Int(42);
        assert_eq!(int_val.as_int().unwrap(), 42);

        let cat_val = ParamValue::Categorical("hello".to_string());
        assert_eq!(cat_val.as_str().unwrap(), "hello");
    }

    #[test]
    fn test_default_spaces() {
        let ppo_space = ppo_default_space();
        assert!(ppo_space.validate().is_ok());
        assert!(ppo_space.get_float("learning_rate").is_some());

        let sac_space = sac_default_space();
        assert!(sac_space.validate().is_ok());
        assert!(sac_space.get_float("tau").is_some());
    }

    #[test]
    fn test_optimization_direction() {
        let space = HyperparameterSpace::new().add_float("x", 0.0, 1.0, false);

        // Maximize
        let mut max_sampler = RandomSearch::new(space.clone(), 42);
        for i in 0..10 {
            let mut trial = max_sampler.suggest(i);
            let x = trial.get_float("x").unwrap();
            trial.complete(x);
            max_sampler.report(&trial);
        }
        let max_best = max_sampler.best_trial().unwrap().value.unwrap();

        // Minimize
        let mut min_sampler =
            RandomSearch::new(space, 42).direction(OptimizationDirection::Minimize);
        for i in 0..10 {
            let mut trial = min_sampler.suggest(i);
            let x = trial.get_float("x").unwrap();
            trial.complete(x);
            min_sampler.report(&trial);
        }
        let min_best = min_sampler.best_trial().unwrap().value.unwrap();

        // Max should find high value, min should find low value
        assert!(max_best > min_best);
    }
}
