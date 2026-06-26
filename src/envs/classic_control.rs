//! Native classic-control environments (no Python / gym dependency).
//!
//! These are pure-Rust reimplementations of the canonical Gymnasium
//! `CartPole-v1` and `Pendulum-v1` environments. They exist to demonstrate
//! that the engine is **not** trading-specific: any [`Environment`] plugs into
//! every agent (`PPOAgent<E>`, `SACAgent<E>`, ...). The dynamics match the
//! Farama-Foundation reference implementations so learning curves are
//! comparable to Stable-Baselines3 on the same tasks.
//!
//! ## RNG decorrelation
//!
//! [`VecEnv::new`] replicates a single template env by `Clone`. A derived
//! `Clone` would copy the RNG *state*, so every parallel copy would draw the
//! same reset states and produce identical trajectories — silently defeating
//! vectorization. Both envs therefore implement `Clone` manually and reseed
//! each clone from OS entropy. Use [`CartPole::seeded`] / [`Pendulum::seeded`]
//! for a reproducible single-env run.

use crate::core::{Device, Result};
use crate::envs::{BoxSpace, DiscreteSpace, Environment, StepResult};
use candle_core::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Wrap an angle to the half-open interval `(-pi, pi]`.
#[inline]
fn angle_normalize(x: f32) -> f32 {
    use std::f32::consts::PI;
    ((x + PI).rem_euclid(2.0 * PI)) - PI
}

// ===========================================================================
// CartPole-v1
// ===========================================================================

/// Canonical `CartPole-v1`: balance a pole on a cart by pushing left/right.
///
/// * Observation: `[x, x_dot, theta, theta_dot]` (continuous, shape `[4]`).
/// * Action: discrete `{0 = push left, 1 = push right}`.
/// * Reward: `+1` per surviving step.
/// * Termination: `|x| > 2.4` or `|theta| > 0.2095 rad` (~12 deg).
/// * Truncation: after 500 steps (the `-v1` horizon).
pub struct CartPole {
    // Physical constants (Gymnasium reference values).
    gravity: f32,
    masspole: f32,
    total_mass: f32,
    length: f32, // actually half the pole's length
    polemass_length: f32,
    force_mag: f32,
    tau: f32, // seconds between state updates

    // Termination thresholds.
    x_threshold: f32,
    theta_threshold: f32,

    // Mutable episode state: [x, x_dot, theta, theta_dot].
    state: [f32; 4],
    steps: u32,
    max_steps: u32,

    obs_space: BoxSpace,
    act_space: DiscreteSpace,
    rng: StdRng,
}

impl CartPole {
    /// Create a CartPole with an entropy-seeded RNG.
    pub fn new() -> Self {
        Self::from_rng(StdRng::from_entropy())
    }

    /// Create a CartPole with a fixed seed (reproducible single-env runs).
    pub fn seeded(seed: u64) -> Self {
        Self::from_rng(StdRng::seed_from_u64(seed))
    }

    fn from_rng(rng: StdRng) -> Self {
        let masspole = 0.1_f32;
        let masscart = 1.0_f32;
        let length = 0.5_f32;
        let theta_threshold = 12.0_f32 * 2.0 * std::f32::consts::PI / 360.0;
        // Observation bounds match Gymnasium: position/angle padded x2, the
        // velocities are unbounded.
        let high = vec![4.8_f32, f32::INFINITY, 2.0 * theta_threshold, f32::INFINITY];
        let low = high.iter().map(|h| -h).collect();
        Self {
            gravity: 9.8,
            masspole,
            total_mass: masspole + masscart,
            length,
            polemass_length: masspole * length,
            force_mag: 10.0,
            tau: 0.02,
            x_threshold: 2.4,
            theta_threshold,
            state: [0.0; 4],
            steps: 0,
            max_steps: 500,
            obs_space: BoxSpace::new(low, high, vec![4]).expect("valid bounds"),
            act_space: DiscreteSpace::new(2),
            rng,
        }
    }

    fn obs(&self, device: &Device) -> Result<Tensor> {
        Ok(Tensor::from_slice(&self.state, &[4], &device.to_candle()?)?)
    }
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for CartPole {
    fn clone(&self) -> Self {
        // Reseed from entropy so VecEnv replicas are decorrelated (see module
        // docs). All other fields are copied as-is.
        Self {
            rng: StdRng::from_entropy(),
            obs_space: self.obs_space.clone(),
            act_space: self.act_space.clone(),
            ..*self
        }
    }
}

impl Environment for CartPole {
    type ObsSpace = BoxSpace;
    type ActSpace = DiscreteSpace;

    fn observation_space(&self) -> &BoxSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &DiscreteSpace {
        &self.act_space
    }

    fn reset(&mut self, device: &Device) -> Result<Tensor> {
        for s in self.state.iter_mut() {
            *s = self.rng.gen_range(-0.05_f32..0.05);
        }
        self.steps = 0;
        self.obs(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let a = action.flatten_all()?.to_vec1::<f32>()?[0];
        let force = if a > 0.5 {
            self.force_mag
        } else {
            -self.force_mag
        };

        let [x, x_dot, theta, theta_dot] = self.state;
        let costheta = theta.cos();
        let sintheta = theta.sin();

        // Reference dynamics (Gymnasium classic_control/cartpole.py).
        let temp =
            (force + self.polemass_length * theta_dot * theta_dot * sintheta) / self.total_mass;
        let thetaacc = (self.gravity * sintheta - costheta * temp)
            / (self.length * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass));
        let xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass;

        // Euler integration: position is advanced with the *old* velocity, then
        // the velocity is advanced (order matters and matches the reference).
        let new_x = x + self.tau * x_dot;
        let new_x_dot = x_dot + self.tau * xacc;
        let new_theta = theta + self.tau * theta_dot;
        let new_theta_dot = theta_dot + self.tau * thetaacc;
        self.state = [new_x, new_x_dot, new_theta, new_theta_dot];
        self.steps += 1;

        let terminated = new_x.abs() > self.x_threshold || new_theta.abs() > self.theta_threshold;
        let truncated = self.steps >= self.max_steps;

        Ok(StepResult {
            observation: self.obs(device)?,
            reward: 1.0,
            terminated,
            truncated,
            info: None,
        })
    }

    fn name(&self) -> &str {
        "CartPole-v1"
    }
}

// ===========================================================================
// Pendulum-v1
// ===========================================================================

/// Canonical `Pendulum-v1`: swing up and balance an inverted pendulum.
///
/// * Observation: `[cos(theta), sin(theta), theta_dot]` (shape `[3]`).
/// * Action: continuous torque `u in [-2, 2]` (shape `[1]`).
/// * Reward: `-(angle_normalize(theta)^2 + 0.1*theta_dot^2 + 0.001*u^2)`,
///   computed from the state *before* the update (a negative cost).
/// * No termination; truncates after 200 steps.
pub struct Pendulum {
    max_speed: f32,
    max_torque: f32,
    dt: f32,
    g: f32,
    m: f32,
    l: f32,

    // [theta, theta_dot].
    theta: f32,
    theta_dot: f32,
    steps: u32,
    max_steps: u32,

    obs_space: BoxSpace,
    act_space: BoxSpace,
    rng: StdRng,
}

impl Pendulum {
    /// Create a Pendulum with an entropy-seeded RNG.
    pub fn new() -> Self {
        Self::from_rng(StdRng::from_entropy())
    }

    /// Create a Pendulum with a fixed seed (reproducible single-env runs).
    pub fn seeded(seed: u64) -> Self {
        Self::from_rng(StdRng::seed_from_u64(seed))
    }

    fn from_rng(rng: StdRng) -> Self {
        let max_speed = 8.0_f32;
        Self {
            max_speed,
            max_torque: 2.0,
            dt: 0.05,
            g: 10.0,
            m: 1.0,
            l: 1.0,
            theta: 0.0,
            theta_dot: 0.0,
            steps: 0,
            max_steps: 200,
            obs_space: BoxSpace::new(
                vec![-1.0, -1.0, -max_speed],
                vec![1.0, 1.0, max_speed],
                vec![3],
            )
            .expect("valid bounds"),
            act_space: BoxSpace::new(vec![-2.0], vec![2.0], vec![1]).expect("valid bounds"),
            rng,
        }
    }

    fn obs(&self, device: &Device) -> Result<Tensor> {
        let data = [self.theta.cos(), self.theta.sin(), self.theta_dot];
        Ok(Tensor::from_slice(&data, &[3], &device.to_candle()?)?)
    }
}

impl Default for Pendulum {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Pendulum {
    fn clone(&self) -> Self {
        Self {
            rng: StdRng::from_entropy(),
            obs_space: self.obs_space.clone(),
            act_space: self.act_space.clone(),
            ..*self
        }
    }
}

impl Environment for Pendulum {
    type ObsSpace = BoxSpace;
    type ActSpace = BoxSpace;

    fn observation_space(&self) -> &BoxSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &BoxSpace {
        &self.act_space
    }

    fn reset(&mut self, device: &Device) -> Result<Tensor> {
        use std::f32::consts::PI;
        self.theta = self.rng.gen_range(-PI..PI);
        self.theta_dot = self.rng.gen_range(-1.0_f32..1.0);
        self.steps = 0;
        self.obs(device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let u = action.flatten_all()?.to_vec1::<f32>()?[0].clamp(-self.max_torque, self.max_torque);

        // Cost is evaluated on the state *before* integration.
        let cost = angle_normalize(self.theta).powi(2)
            + 0.1 * self.theta_dot * self.theta_dot
            + 0.001 * u * u;

        let new_theta_dot = (self.theta_dot
            + (3.0 * self.g / (2.0 * self.l) * self.theta.sin()
                + 3.0 / (self.m * self.l * self.l) * u)
                * self.dt)
            .clamp(-self.max_speed, self.max_speed);
        let new_theta = self.theta + new_theta_dot * self.dt;

        self.theta = new_theta;
        self.theta_dot = new_theta_dot;
        self.steps += 1;

        Ok(StepResult {
            observation: self.obs(device)?,
            reward: -cost,
            terminated: false,
            truncated: self.steps >= self.max_steps,
            info: None,
        })
    }

    fn name(&self) -> &str {
        "Pendulum-v1"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn cartpole_known_value_transition() {
        // From the origin with a rightward push (action = 1) the reference
        // dynamics give a specific next state. These constants were derived by
        // hand from the Gymnasium formulas (total_mass = 1.1).
        let mut env = CartPole::seeded(0);
        env.state = [0.0, 0.0, 0.0, 0.0];
        let action = Tensor::from_slice(&[1.0_f32], &[1], &cpu().to_candle().unwrap()).unwrap();
        let r = env.step(&action, &cpu()).unwrap();
        let next: Vec<f32> = r.observation.to_vec1().unwrap();
        // x stays 0 (old x_dot was 0); x_dot = tau*xacc; theta stays 0;
        // theta_dot = tau*thetaacc.
        assert!((next[0] - 0.0).abs() < 1e-5, "x = {}", next[0]);
        assert!((next[1] - 0.195122).abs() < 1e-4, "x_dot = {}", next[1]);
        assert!((next[2] - 0.0).abs() < 1e-5, "theta = {}", next[2]);
        assert!(
            (next[3] - (-0.292651)).abs() < 1e-4,
            "theta_dot = {}",
            next[3]
        );
        assert!((r.reward - 1.0).abs() < 1e-6);
        assert!(!r.terminated);
    }

    #[test]
    fn cartpole_terminates_out_of_bounds() {
        let mut env = CartPole::seeded(0);
        env.state = [0.0, 0.0, 0.3, 0.0]; // |theta| 0.3 > 0.2095
        let action = Tensor::from_slice(&[1.0_f32], &[1], &cpu().to_candle().unwrap()).unwrap();
        let r = env.step(&action, &cpu()).unwrap();
        assert!(
            r.terminated,
            "should terminate when pole angle exceeds threshold"
        );
    }

    #[test]
    fn cartpole_reset_within_init_bounds() {
        let mut env = CartPole::seeded(42);
        let obs = env.reset(&cpu()).unwrap();
        let v: Vec<f32> = obs.to_vec1().unwrap();
        assert_eq!(v.len(), 4);
        assert!(
            v.iter().all(|&x| x.abs() <= 0.05),
            "init state {v:?} out of [-0.05,0.05]"
        );
    }

    #[test]
    fn pendulum_known_value_transition() {
        // theta = 0, theta_dot = 0, u = +2:
        //   new_theta_dot = 0 + (3*10/(2*1)*sin(0) + 3/(1*1)*2)*0.05 = 6*0.05 = 0.3
        //   new_theta     = 0 + 0.3*0.05 = 0.015
        //   reward        = -(0 + 0 + 0.001*4) = -0.004
        let mut env = Pendulum::seeded(0);
        env.theta = 0.0;
        env.theta_dot = 0.0;
        let action = Tensor::from_slice(&[2.0_f32], &[1], &cpu().to_candle().unwrap()).unwrap();
        let r = env.step(&action, &cpu()).unwrap();
        assert!(
            (env.theta_dot - 0.3).abs() < 1e-5,
            "theta_dot = {}",
            env.theta_dot
        );
        assert!((env.theta - 0.015).abs() < 1e-5, "theta = {}", env.theta);
        assert!((r.reward - (-0.004)).abs() < 1e-5, "reward = {}", r.reward);
        let obs: Vec<f32> = r.observation.to_vec1().unwrap();
        assert!((obs[0] - env.theta.cos()).abs() < 1e-6);
        assert!((obs[1] - env.theta.sin()).abs() < 1e-6);
    }

    #[test]
    fn pendulum_torque_is_clipped() {
        // A huge torque must be clamped to max_torque = 2 before integration.
        let mut env = Pendulum::seeded(0);
        env.theta = 0.0;
        env.theta_dot = 0.0;
        let action = Tensor::from_slice(&[100.0_f32], &[1], &cpu().to_candle().unwrap()).unwrap();
        env.step(&action, &cpu()).unwrap();
        // With u clamped to 2: new_theta_dot = 0.3 (not 15).
        assert!(
            (env.theta_dot - 0.3).abs() < 1e-5,
            "torque not clipped: {}",
            env.theta_dot
        );
    }

    #[test]
    fn angle_normalize_wraps() {
        use std::f32::consts::PI;
        // The range is [-pi, pi); test away from the +-pi boundary, which is
        // numerically ambiguous in f32.
        assert!(angle_normalize(0.0).abs() < 1e-6);
        assert!(
            angle_normalize(2.0 * PI).abs() < 1e-4,
            "2pi should wrap to 0"
        );
        assert!((angle_normalize(PI / 2.0) - PI / 2.0).abs() < 1e-6);
        assert!(
            (angle_normalize(2.0 * PI + PI / 2.0) - PI / 2.0).abs() < 1e-4,
            "2pi + pi/2 should wrap to pi/2"
        );
    }

    #[test]
    fn clones_are_decorrelated() {
        // Two clones reset from independent RNGs must (almost surely) differ.
        let template = CartPole::new();
        let mut a = template.clone();
        let mut b = template.clone();
        let oa: Vec<f32> = a.reset(&cpu()).unwrap().to_vec1().unwrap();
        let ob: Vec<f32> = b.reset(&cpu()).unwrap().to_vec1().unwrap();
        assert!(
            oa != ob,
            "clones produced identical reset states: RNG not decorrelated"
        );
    }
}
