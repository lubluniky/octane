//! Safe RL constraints for trading.
//!
//! This module provides constraint enforcement mechanisms for safe reinforcement
//! learning in trading environments, including:
//!
//! - Hard constraints on position size, drawdown, and exposure
//! - Action masking when constraints would be violated
//! - Lagrangian relaxation for soft constraint handling
//! - Constraint satisfaction layers that project actions to feasible sets
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::risk::{
//!     ConstraintManager, ConstraintConfig, Constraint,
//!     BoxConstraint, InequalityConstraint,
//! };
//!
//! let mut manager = ConstraintManager::new(ConstraintConfig::default());
//!
//! // Add position size constraint
//! manager.add_constraint(Constraint::Box(BoxConstraint::new(
//!     "position".into(),
//!     -1.0,  // min position (short)
//!     1.0,   // max position (long)
//! )));
//!
//! // Add max exposure constraint
//! manager.add_constraint(Constraint::Inequality(InequalityConstraint::new(
//!     "exposure".into(),
//!     0.5,  // max exposure
//! )));
//!
//! // Check and project action
//! let action = vec![1.5];  // Would exceed constraint
//! let result = manager.project_action(&action, &state);
//! assert!(result.projected[0] <= 1.0);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Errors specific to constraint operations.
#[derive(Debug, Clone, PartialEq)]
pub enum ConstraintError {
    /// Constraint violation detected.
    Violation {
        /// Name of the violated constraint.
        name: String,
        /// Current value.
        value: f64,
        /// Constraint bound.
        bound: f64,
    },
    /// Invalid constraint configuration.
    InvalidConfig(String),
    /// Infeasible constraint set (no solution exists).
    Infeasible(String),
}

impl std::fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConstraintError::Violation { name, value, bound } => {
                write!(
                    f,
                    "Constraint '{}' violated: value {} exceeds bound {}",
                    name, value, bound
                )
            }
            ConstraintError::InvalidConfig(msg) => write!(f, "Invalid constraint config: {}", msg),
            ConstraintError::Infeasible(msg) => write!(f, "Infeasible constraints: {}", msg),
        }
    }
}

impl std::error::Error for ConstraintError {}

/// Type of constraint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConstraintType {
    /// Equality constraint: g(x) = 0
    Equality,
    /// Inequality constraint: g(x) <= 0
    Inequality,
    /// Box constraint: lower <= x <= upper
    Box,
}

/// A generic constraint trait.
pub trait ConstraintTrait: Send + Sync {
    /// Get the constraint name.
    fn name(&self) -> &str;

    /// Get the constraint type.
    fn constraint_type(&self) -> ConstraintType;

    /// Evaluate the constraint violation.
    /// Returns 0 if satisfied, positive value for violation amount.
    fn evaluate(&self, value: f64) -> f64;

    /// Check if the constraint is satisfied.
    fn is_satisfied(&self, value: f64) -> bool {
        self.evaluate(value) <= 0.0
    }

    /// Project a value to satisfy the constraint.
    fn project(&self, value: f64) -> f64;

    /// Get the gradient of the constraint function (for Lagrangian methods).
    fn gradient(&self, value: f64) -> f64;
}

/// Box constraint: lower <= x <= upper.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoxConstraint {
    /// Constraint name.
    pub name: String,
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
}

impl BoxConstraint {
    /// Create a new box constraint.
    pub fn new(name: String, lower: f64, upper: f64) -> Self {
        Self { name, lower, upper }
    }

    /// Create a symmetric box constraint [-bound, bound].
    pub fn symmetric(name: String, bound: f64) -> Self {
        Self::new(name, -bound.abs(), bound.abs())
    }
}

impl ConstraintTrait for BoxConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn constraint_type(&self) -> ConstraintType {
        ConstraintType::Box
    }

    fn evaluate(&self, value: f64) -> f64 {
        if value < self.lower {
            self.lower - value
        } else if value > self.upper {
            value - self.upper
        } else {
            0.0
        }
    }

    fn project(&self, value: f64) -> f64 {
        value.clamp(self.lower, self.upper)
    }

    fn gradient(&self, value: f64) -> f64 {
        if value < self.lower {
            -1.0
        } else if value > self.upper {
            1.0
        } else {
            0.0
        }
    }
}

/// Inequality constraint: g(x) <= bound.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InequalityConstraint {
    /// Constraint name.
    pub name: String,
    /// Upper bound.
    pub bound: f64,
    /// Whether this is a greater-than constraint (g(x) >= bound).
    pub greater_than: bool,
}

impl InequalityConstraint {
    /// Create a new inequality constraint (x <= bound).
    pub fn new(name: String, bound: f64) -> Self {
        Self {
            name,
            bound,
            greater_than: false,
        }
    }

    /// Create a greater-than constraint (x >= bound).
    pub fn greater_than(name: String, bound: f64) -> Self {
        Self {
            name,
            bound,
            greater_than: true,
        }
    }
}

impl ConstraintTrait for InequalityConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn constraint_type(&self) -> ConstraintType {
        ConstraintType::Inequality
    }

    fn evaluate(&self, value: f64) -> f64 {
        if self.greater_than {
            // g(x) >= bound  =>  bound - g(x) <= 0
            (self.bound - value).max(0.0)
        } else {
            // g(x) <= bound  =>  g(x) - bound <= 0
            (value - self.bound).max(0.0)
        }
    }

    fn project(&self, value: f64) -> f64 {
        if self.greater_than {
            value.max(self.bound)
        } else {
            value.min(self.bound)
        }
    }

    fn gradient(&self, value: f64) -> f64 {
        if self.greater_than {
            if value < self.bound {
                -1.0
            } else {
                0.0
            }
        } else if value > self.bound {
            1.0
        } else {
            0.0
        }
    }
}

/// Equality constraint: g(x) = target.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EqualityConstraint {
    /// Constraint name.
    pub name: String,
    /// Target value.
    pub target: f64,
    /// Tolerance for equality (for numerical stability).
    pub tolerance: f64,
}

impl EqualityConstraint {
    /// Create a new equality constraint.
    pub fn new(name: String, target: f64) -> Self {
        Self {
            name,
            target,
            tolerance: 1e-6,
        }
    }

    /// Create with custom tolerance.
    pub fn with_tolerance(name: String, target: f64, tolerance: f64) -> Self {
        Self {
            name,
            target,
            tolerance,
        }
    }
}

impl ConstraintTrait for EqualityConstraint {
    fn name(&self) -> &str {
        &self.name
    }

    fn constraint_type(&self) -> ConstraintType {
        ConstraintType::Equality
    }

    fn evaluate(&self, value: f64) -> f64 {
        let diff = (value - self.target).abs();
        if diff <= self.tolerance {
            0.0
        } else {
            diff
        }
    }

    fn project(&self, _value: f64) -> f64 {
        self.target
    }

    fn gradient(&self, value: f64) -> f64 {
        if (value - self.target).abs() <= self.tolerance {
            0.0
        } else {
            (value - self.target).signum()
        }
    }
}

/// Enum wrapper for different constraint types.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Constraint {
    /// Box constraint.
    Box(BoxConstraint),
    /// Inequality constraint.
    Inequality(InequalityConstraint),
    /// Equality constraint.
    Equality(EqualityConstraint),
}

impl Constraint {
    /// Get the constraint name.
    pub fn name(&self) -> &str {
        match self {
            Constraint::Box(c) => &c.name,
            Constraint::Inequality(c) => &c.name,
            Constraint::Equality(c) => &c.name,
        }
    }

    /// Get the constraint type.
    pub fn constraint_type(&self) -> ConstraintType {
        match self {
            Constraint::Box(_) => ConstraintType::Box,
            Constraint::Inequality(_) => ConstraintType::Inequality,
            Constraint::Equality(_) => ConstraintType::Equality,
        }
    }

    /// Evaluate the constraint violation.
    pub fn evaluate(&self, value: f64) -> f64 {
        match self {
            Constraint::Box(c) => c.evaluate(value),
            Constraint::Inequality(c) => c.evaluate(value),
            Constraint::Equality(c) => c.evaluate(value),
        }
    }

    /// Check if the constraint is satisfied.
    pub fn is_satisfied(&self, value: f64) -> bool {
        self.evaluate(value) <= 0.0
    }

    /// Project a value to satisfy the constraint.
    pub fn project(&self, value: f64) -> f64 {
        match self {
            Constraint::Box(c) => c.project(value),
            Constraint::Inequality(c) => c.project(value),
            Constraint::Equality(c) => c.project(value),
        }
    }

    /// Get the gradient.
    pub fn gradient(&self, value: f64) -> f64 {
        match self {
            Constraint::Box(c) => c.gradient(value),
            Constraint::Inequality(c) => c.gradient(value),
            Constraint::Equality(c) => c.gradient(value),
        }
    }
}

/// Result of constraint checking.
#[derive(Debug, Clone)]
pub struct ConstraintResult {
    /// Whether all constraints are satisfied.
    pub satisfied: bool,
    /// Total violation amount (sum of all violations).
    pub total_violation: f64,
    /// Individual constraint violations.
    pub violations: HashMap<String, f64>,
    /// Names of violated constraints.
    pub violated_constraints: Vec<String>,
}

impl ConstraintResult {
    /// Create a satisfied result.
    pub fn satisfied() -> Self {
        Self {
            satisfied: true,
            total_violation: 0.0,
            violations: HashMap::new(),
            violated_constraints: Vec::new(),
        }
    }
}

/// Result of action projection.
#[derive(Debug, Clone)]
pub struct ProjectionResult {
    /// Original action.
    pub original: Vec<f64>,
    /// Projected action (satisfies constraints).
    pub projected: Vec<f64>,
    /// Whether projection was needed.
    pub was_projected: bool,
    /// Maximum projection distance.
    pub max_projection_distance: f64,
    /// Constraint check result after projection.
    pub constraint_result: ConstraintResult,
}

/// Action mask for constraint enforcement.
#[derive(Debug, Clone)]
pub struct ActionMask {
    /// Mask values (1.0 = allowed, 0.0 = forbidden).
    pub mask: Vec<f64>,
    /// Lower bounds for each action dimension.
    pub lower_bounds: Vec<f64>,
    /// Upper bounds for each action dimension.
    pub upper_bounds: Vec<f64>,
}

impl ActionMask {
    /// Create a new action mask.
    pub fn new(action_dim: usize) -> Self {
        Self {
            mask: vec![1.0; action_dim],
            lower_bounds: vec![f64::NEG_INFINITY; action_dim],
            upper_bounds: vec![f64::INFINITY; action_dim],
        }
    }

    /// Mask a specific action dimension.
    pub fn mask_action(&mut self, dim: usize) {
        if dim < self.mask.len() {
            self.mask[dim] = 0.0;
        }
    }

    /// Set bounds for a dimension.
    pub fn set_bounds(&mut self, dim: usize, lower: f64, upper: f64) {
        if dim < self.mask.len() {
            self.lower_bounds[dim] = lower;
            self.upper_bounds[dim] = upper;
        }
    }

    /// Apply mask to an action.
    pub fn apply(&self, action: &[f64], default_action: &[f64]) -> Vec<f64> {
        action
            .iter()
            .zip(self.mask.iter())
            .zip(default_action.iter())
            .zip(self.lower_bounds.iter().zip(self.upper_bounds.iter()))
            .map(
                |(((a, m), d), (lo, hi))| {
                    if *m > 0.5 {
                        a.clamp(*lo, *hi)
                    } else {
                        *d
                    }
                },
            )
            .collect()
    }

    /// Check if any action is masked.
    pub fn any_masked(&self) -> bool {
        self.mask.iter().any(|&m| m < 0.5)
    }
}

/// Configuration for constraint manager.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintConfig {
    /// Whether to use hard constraints (project actions).
    pub hard_constraints: bool,
    /// Whether to use soft constraints (Lagrangian penalty).
    pub soft_constraints: bool,
    /// Initial Lagrange multipliers.
    pub initial_multipliers: f64,
    /// Learning rate for Lagrange multiplier updates.
    pub multiplier_lr: f64,
    /// Maximum Lagrange multiplier value.
    pub max_multiplier: f64,
    /// Number of projection iterations.
    pub projection_iterations: usize,
    /// Projection step size.
    pub projection_step_size: f64,
    /// Tolerance for constraint satisfaction.
    pub tolerance: f64,
}

impl Default for ConstraintConfig {
    fn default() -> Self {
        Self {
            hard_constraints: true,
            soft_constraints: false,
            initial_multipliers: 0.0,
            multiplier_lr: 0.01,
            max_multiplier: 100.0,
            projection_iterations: 10,
            projection_step_size: 0.1,
            tolerance: 1e-6,
        }
    }
}

impl ConstraintConfig {
    /// Create a new constraint config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable hard constraints.
    pub fn hard_constraints(mut self, enabled: bool) -> Self {
        self.hard_constraints = enabled;
        self
    }

    /// Enable soft constraints.
    pub fn soft_constraints(mut self, enabled: bool) -> Self {
        self.soft_constraints = enabled;
        self
    }

    /// Set initial Lagrange multipliers.
    pub fn initial_multipliers(mut self, value: f64) -> Self {
        self.initial_multipliers = value;
        self
    }

    /// Set multiplier learning rate.
    pub fn multiplier_lr(mut self, lr: f64) -> Self {
        self.multiplier_lr = lr;
        self
    }

    /// Set maximum multiplier.
    pub fn max_multiplier(mut self, max: f64) -> Self {
        self.max_multiplier = max;
        self
    }

    /// Set projection iterations.
    pub fn projection_iterations(mut self, iters: usize) -> Self {
        self.projection_iterations = iters;
        self
    }

    /// Set tolerance.
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }
}

/// Lagrangian relaxation for soft constraint handling.
#[derive(Debug, Clone)]
pub struct LagrangianRelaxation {
    /// Lagrange multipliers for each constraint.
    multipliers: HashMap<String, f64>,
    /// Learning rate for multiplier updates.
    learning_rate: f64,
    /// Maximum multiplier value.
    max_multiplier: f64,
}

impl LagrangianRelaxation {
    /// Create a new Lagrangian relaxation handler.
    pub fn new(learning_rate: f64, max_multiplier: f64) -> Self {
        Self {
            multipliers: HashMap::new(),
            learning_rate,
            max_multiplier,
        }
    }

    /// Initialize multiplier for a constraint.
    pub fn init_multiplier(&mut self, name: &str, initial_value: f64) {
        self.multipliers.insert(name.to_string(), initial_value);
    }

    /// Get multiplier for a constraint.
    pub fn get_multiplier(&self, name: &str) -> f64 {
        self.multipliers.get(name).copied().unwrap_or(0.0)
    }

    /// Update multipliers based on constraint violations.
    ///
    /// Uses dual gradient ascent: lambda += lr * g(x)
    /// where g(x) is the constraint violation.
    pub fn update(&mut self, violations: &HashMap<String, f64>) {
        for (name, &violation) in violations {
            let multiplier = self.multipliers.entry(name.clone()).or_insert(0.0);
            *multiplier =
                (*multiplier + self.learning_rate * violation).clamp(0.0, self.max_multiplier);
        }
    }

    /// Compute the Lagrangian penalty term.
    ///
    /// Returns sum_i lambda_i * g_i(x) for all constraints.
    pub fn penalty(&self, violations: &HashMap<String, f64>) -> f64 {
        violations
            .iter()
            .map(|(name, &violation)| self.get_multiplier(name) * violation)
            .sum()
    }

    /// Get all multipliers.
    pub fn multipliers(&self) -> &HashMap<String, f64> {
        &self.multipliers
    }

    /// Reset all multipliers.
    pub fn reset(&mut self) {
        for value in self.multipliers.values_mut() {
            *value = 0.0;
        }
    }
}

/// Trading state for constraint evaluation.
#[derive(Debug, Clone, Default)]
pub struct TradingState {
    /// Current position (normalized, -1 to 1).
    pub position: f64,
    /// Current equity.
    pub equity: f64,
    /// Peak equity (for drawdown).
    pub peak_equity: f64,
    /// Current exposure (absolute position value * equity).
    pub exposure: f64,
    /// Number of open positions.
    pub num_positions: usize,
    /// Current drawdown.
    pub drawdown: f64,
    /// Daily returns for volatility calculation.
    pub daily_returns: Vec<f64>,
    /// Additional state variables.
    pub extra: HashMap<String, f64>,
}

impl TradingState {
    /// Create a new trading state.
    pub fn new(equity: f64) -> Self {
        Self {
            equity,
            peak_equity: equity,
            ..Default::default()
        }
    }

    /// Update drawdown based on current equity.
    pub fn update_drawdown(&mut self) {
        self.peak_equity = self.peak_equity.max(self.equity);
        self.drawdown = if self.peak_equity > 0.0 {
            (self.peak_equity - self.equity) / self.peak_equity
        } else {
            0.0
        };
    }
}

/// Constraint manager for safe RL.
pub struct ConstraintManager {
    /// Configuration.
    config: ConstraintConfig,
    /// Position constraints (keyed by dimension index).
    position_constraints: HashMap<usize, Constraint>,
    /// Global constraints (apply to derived values).
    global_constraints: Vec<(String, Constraint)>,
    /// Lagrangian relaxation handler.
    lagrangian: Option<LagrangianRelaxation>,
    /// Current action mask.
    action_mask: Option<ActionMask>,
}

impl ConstraintManager {
    /// Create a new constraint manager.
    pub fn new(config: ConstraintConfig) -> Self {
        let lagrangian = if config.soft_constraints {
            Some(LagrangianRelaxation::new(
                config.multiplier_lr,
                config.max_multiplier,
            ))
        } else {
            None
        };

        Self {
            config,
            position_constraints: HashMap::new(),
            global_constraints: Vec::new(),
            lagrangian,
            action_mask: None,
        }
    }

    /// Add a position constraint for a specific action dimension.
    pub fn add_position_constraint(&mut self, dim: usize, constraint: Constraint) {
        if let Some(ref mut lagrangian) = self.lagrangian {
            lagrangian.init_multiplier(constraint.name(), self.config.initial_multipliers);
        }
        self.position_constraints.insert(dim, constraint);
    }

    /// Add a global constraint.
    pub fn add_global_constraint(&mut self, value_name: String, constraint: Constraint) {
        if let Some(ref mut lagrangian) = self.lagrangian {
            lagrangian.init_multiplier(constraint.name(), self.config.initial_multipliers);
        }
        self.global_constraints.push((value_name, constraint));
    }

    /// Add a constraint (shorthand for common cases).
    pub fn add_constraint(&mut self, constraint: Constraint) {
        self.add_global_constraint(constraint.name().to_string(), constraint);
    }

    /// Add a maximum drawdown constraint.
    pub fn add_max_drawdown_constraint(&mut self, max_drawdown: f64) {
        self.add_global_constraint(
            "drawdown".to_string(),
            Constraint::Inequality(InequalityConstraint::new(
                "max_drawdown".to_string(),
                max_drawdown,
            )),
        );
    }

    /// Add a maximum position size constraint.
    pub fn add_max_position_constraint(&mut self, dim: usize, max_position: f64) {
        self.add_position_constraint(
            dim,
            Constraint::Box(BoxConstraint::symmetric(
                format!("position_{}", dim),
                max_position,
            )),
        );
    }

    /// Add a maximum exposure constraint.
    pub fn add_max_exposure_constraint(&mut self, max_exposure: f64) {
        self.add_global_constraint(
            "exposure".to_string(),
            Constraint::Inequality(InequalityConstraint::new(
                "max_exposure".to_string(),
                max_exposure,
            )),
        );
    }

    /// Check constraints for given action and state.
    pub fn check_constraints(&self, action: &[f64], state: &TradingState) -> ConstraintResult {
        let mut violations = HashMap::new();
        let mut violated_constraints = Vec::new();
        let mut total_violation = 0.0;

        // Check position constraints
        for (&dim, constraint) in &self.position_constraints {
            if dim < action.len() {
                let violation = constraint.evaluate(action[dim]);
                if violation > self.config.tolerance {
                    violations.insert(constraint.name().to_string(), violation);
                    violated_constraints.push(constraint.name().to_string());
                    total_violation += violation;
                }
            }
        }

        // Check global constraints
        for (value_name, constraint) in &self.global_constraints {
            let value = match value_name.as_str() {
                "drawdown" => state.drawdown,
                "exposure" => state.exposure,
                "position" => state.position,
                "equity" => state.equity,
                _ => state.extra.get(value_name).copied().unwrap_or(0.0),
            };

            let violation = constraint.evaluate(value);
            if violation > self.config.tolerance {
                violations.insert(constraint.name().to_string(), violation);
                violated_constraints.push(constraint.name().to_string());
                total_violation += violation;
            }
        }

        ConstraintResult {
            satisfied: violated_constraints.is_empty(),
            total_violation,
            violations,
            violated_constraints,
        }
    }

    /// Project action to satisfy constraints.
    pub fn project_action(&self, action: &[f64], _state: &TradingState) -> ProjectionResult {
        if !self.config.hard_constraints {
            return ProjectionResult {
                original: action.to_vec(),
                projected: action.to_vec(),
                was_projected: false,
                max_projection_distance: 0.0,
                constraint_result: ConstraintResult::satisfied(),
            };
        }

        let mut projected = action.to_vec();
        let mut max_distance: f64 = 0.0;

        // Project each dimension according to its constraint
        for (&dim, constraint) in &self.position_constraints {
            if dim < projected.len() {
                let original = projected[dim];
                projected[dim] = constraint.project(original);
                max_distance = max_distance.max((projected[dim] - original).abs());
            }
        }

        let was_projected = max_distance > self.config.tolerance;

        ProjectionResult {
            original: action.to_vec(),
            projected,
            was_projected,
            max_projection_distance: max_distance,
            constraint_result: ConstraintResult::satisfied(),
        }
    }

    /// Compute Lagrangian penalty for soft constraints.
    pub fn lagrangian_penalty(&self, action: &[f64], state: &TradingState) -> f64 {
        match &self.lagrangian {
            Some(lagrangian) => {
                let result = self.check_constraints(action, state);
                lagrangian.penalty(&result.violations)
            }
            None => 0.0,
        }
    }

    /// Update Lagrange multipliers based on constraint violations.
    pub fn update_lagrangian(&mut self, action: &[f64], state: &TradingState) {
        // First compute the result without borrowing lagrangian
        let result = self.check_constraints(action, state);
        // Then update the lagrangian
        if let Some(ref mut lagrangian) = self.lagrangian {
            lagrangian.update(&result.violations);
        }
    }

    /// Get the current action mask.
    pub fn get_action_mask(&self, action_dim: usize, state: &TradingState) -> ActionMask {
        let mut mask = ActionMask::new(action_dim);

        // Check if we're near drawdown limit - reduce allowed position sizes
        for (value_name, constraint) in &self.global_constraints {
            if value_name == "drawdown" {
                let violation = constraint.evaluate(state.drawdown);
                if violation > 0.0 {
                    // Near or at drawdown limit - mask aggressive positions
                    for dim in 0..action_dim {
                        // Only allow positions that would reduce exposure
                        if state.position > 0.0 {
                            mask.set_bounds(dim, -1.0, 0.0);
                        } else if state.position < 0.0 {
                            mask.set_bounds(dim, 0.0, 1.0);
                        } else {
                            mask.mask_action(dim);
                        }
                    }
                }
            }
        }

        // Apply position constraints to mask
        for (&dim, constraint) in &self.position_constraints {
            if dim < action_dim {
                if let Constraint::Box(box_constraint) = constraint {
                    mask.set_bounds(dim, box_constraint.lower, box_constraint.upper);
                }
            }
        }

        mask
    }

    /// Initialize action mask.
    pub fn init_action_mask(&mut self, action_dim: usize) {
        self.action_mask = Some(ActionMask::new(action_dim));
    }

    /// Get current Lagrange multipliers.
    pub fn get_multipliers(&self) -> Option<&HashMap<String, f64>> {
        self.lagrangian.as_ref().map(|l| l.multipliers())
    }

    /// Reset the constraint manager state.
    pub fn reset(&mut self) {
        if let Some(ref mut lagrangian) = self.lagrangian {
            lagrangian.reset();
        }
        self.action_mask = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_constraint() {
        let constraint = BoxConstraint::new("position".into(), -1.0, 1.0);

        assert!(constraint.is_satisfied(0.5));
        assert!(constraint.is_satisfied(-1.0));
        assert!(constraint.is_satisfied(1.0));
        assert!(!constraint.is_satisfied(1.5));
        assert!(!constraint.is_satisfied(-1.5));

        assert!((constraint.project(1.5) - 1.0).abs() < 1e-10);
        assert!((constraint.project(-1.5) - (-1.0)).abs() < 1e-10);
        assert!((constraint.project(0.5) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_inequality_constraint() {
        let constraint = InequalityConstraint::new("max_exposure".into(), 0.5);

        assert!(constraint.is_satisfied(0.3));
        assert!(constraint.is_satisfied(0.5));
        assert!(!constraint.is_satisfied(0.7));

        assert!((constraint.project(0.7) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_equality_constraint() {
        let constraint = EqualityConstraint::new("target".into(), 1.0);

        assert!(constraint.is_satisfied(1.0));
        assert!(!constraint.is_satisfied(1.1));

        assert!((constraint.project(0.5) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_constraint_manager() {
        let config = ConstraintConfig::default();
        let mut manager = ConstraintManager::new(config);

        manager.add_max_position_constraint(0, 1.0);
        manager.add_max_drawdown_constraint(0.2);

        let state = TradingState {
            drawdown: 0.1,
            ..Default::default()
        };

        let action = vec![0.5];
        let result = manager.check_constraints(&action, &state);
        assert!(result.satisfied);

        let action = vec![1.5];
        let result = manager.check_constraints(&action, &state);
        assert!(!result.satisfied);

        let projected = manager.project_action(&action, &state);
        assert!(projected.was_projected);
        assert!((projected.projected[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lagrangian_relaxation() {
        let mut lagrangian = LagrangianRelaxation::new(0.1, 10.0);
        lagrangian.init_multiplier("constraint1", 0.0);

        let mut violations = HashMap::new();
        violations.insert("constraint1".to_string(), 1.0);

        // Update multiplier
        lagrangian.update(&violations);
        assert!((lagrangian.get_multiplier("constraint1") - 0.1).abs() < 1e-10);

        // Compute penalty
        let penalty = lagrangian.penalty(&violations);
        assert!((penalty - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_action_mask() {
        let mut mask = ActionMask::new(2);
        mask.set_bounds(0, -0.5, 0.5);
        mask.mask_action(1);

        let action = vec![1.0, 1.0];
        let default = vec![0.0, 0.0];
        let masked = mask.apply(&action, &default);

        assert!((masked[0] - 0.5).abs() < 1e-10); // Clamped
        assert!((masked[1] - 0.0).abs() < 1e-10); // Masked to default
    }
}
