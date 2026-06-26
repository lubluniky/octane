//! Generic dataset-driven environment — the "any data" entry point.
//!
//! [`ArrayEnv`] turns an arbitrary numeric matrix into an RL task without any
//! trading assumptions. You hand it a feature matrix `[T, obs_dim]` once; the
//! per-step loop then runs entirely in Rust (no Python in the hot path, so the
//! native speed advantage is preserved). Each row is an observation; the agent
//! emits a continuous action and is scored by an [`ArrayReward`].
//!
//! Two reward framings are supported because "any data" means different things:
//!
//! * [`ArrayReward::Regression`] — `reward = -MSE(action, target_row)`. A
//!   sequential-prediction / contextual setup: learn to predict a target
//!   vector from each feature row. The action does not alter the (fixed) data
//!   sequence — this is a contextual problem, not a controlled MDP.
//! * [`ArrayReward::Weighted`] — `reward = dot(action, returns_row)`. The
//!   action is a weight vector applied to the next-step quantities (e.g.
//!   portfolio weights over asset returns); the realized P&L is the reward.
//!
//! Both are genuine RL signals (reward depends on the action). Pick the framing
//! that matches your data; feed normalized features for stable learning.

use crate::core::{Device, OctaneError, Result};
use crate::envs::{BoxSpace, Environment, StepResult};
use candle_core::Tensor;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// How a row + action is converted into a scalar reward.
#[derive(Debug, Clone)]
pub enum ArrayReward {
    /// `reward = -mean((action - target_row)^2)`. `targets` is a flat
    /// `[T * target_dim]` matrix; the action dimension equals `target_dim`.
    Regression {
        /// Flattened `[T, target_dim]` target matrix.
        targets: Vec<f32>,
        /// Width of each target row (= action dimension).
        target_dim: usize,
    },
    /// `reward = dot(action, returns_row)`. `returns` is a flat
    /// `[T * n_assets]` matrix; the action dimension equals `n_assets`.
    Weighted {
        /// Flattened `[T, n_assets]` returns matrix.
        returns: Vec<f32>,
        /// Number of assets (= action dimension).
        n_assets: usize,
    },
}

impl ArrayReward {
    fn action_dim(&self) -> usize {
        match self {
            ArrayReward::Regression { target_dim, .. } => *target_dim,
            ArrayReward::Weighted { n_assets, .. } => *n_assets,
        }
    }

    /// Number of rows implied by the reward matrix (for length validation).
    fn rows(&self) -> usize {
        match self {
            ArrayReward::Regression {
                targets,
                target_dim,
            } => targets.len() / (*target_dim).max(1),
            ArrayReward::Weighted { returns, n_assets } => returns.len() / (*n_assets).max(1),
        }
    }

    fn reward(&self, row: usize, action: &[f32]) -> f32 {
        match self {
            ArrayReward::Regression {
                targets,
                target_dim,
            } => {
                let base = row * target_dim;
                let mut sse = 0.0_f32;
                for i in 0..*target_dim {
                    let diff = action[i] - targets[base + i];
                    sse += diff * diff;
                }
                -(sse / *target_dim as f32)
            }
            ArrayReward::Weighted { returns, n_assets } => {
                let base = row * n_assets;
                let mut dot = 0.0_f32;
                for i in 0..*n_assets {
                    dot += action[i] * returns[base + i];
                }
                dot
            }
        }
    }
}

/// A generic environment over a fixed `[T, obs_dim]` feature matrix.
pub struct ArrayEnv {
    data: Vec<f32>, // flat [n_rows * obs_dim]
    obs_dim: usize,
    n_rows: usize,
    reward: ArrayReward,
    act_dim: usize,

    cursor: usize,
    start: usize,
    episode_len: usize,
    random_start: bool,

    obs_space: BoxSpace,
    act_space: BoxSpace,
    rng: StdRng,
}

impl ArrayEnv {
    /// Build an `ArrayEnv` from a flat `[n_rows * obs_dim]` feature matrix and
    /// a reward source. Returns an error if the shapes are inconsistent.
    pub fn new(data: Vec<f32>, obs_dim: usize, reward: ArrayReward) -> Result<Self> {
        if obs_dim == 0 {
            return Err(OctaneError::InvalidConfig("obs_dim must be > 0".into()));
        }
        if !data.len().is_multiple_of(obs_dim) {
            return Err(OctaneError::InvalidConfig(format!(
                "data length {} is not a multiple of obs_dim {}",
                data.len(),
                obs_dim
            )));
        }
        let n_rows = data.len() / obs_dim;
        if n_rows < 2 {
            return Err(OctaneError::InvalidConfig(
                "ArrayEnv needs at least 2 rows".into(),
            ));
        }
        if reward.rows() != n_rows {
            return Err(OctaneError::InvalidConfig(format!(
                "reward matrix implies {} rows but data has {}",
                reward.rows(),
                n_rows
            )));
        }
        let act_dim = reward.action_dim();
        // Continuous actions: predictions are unbounded; portfolio weights live
        // in a bounded box but we keep a symmetric unit box as a sane default.
        let act_space = match &reward {
            ArrayReward::Regression { .. } => BoxSpace::symmetric(f32::INFINITY, vec![act_dim]),
            ArrayReward::Weighted { .. } => BoxSpace::symmetric(1.0, vec![act_dim]),
        };
        Ok(Self {
            data,
            obs_dim,
            n_rows,
            reward,
            act_dim,
            cursor: 0,
            start: 0,
            episode_len: n_rows,
            random_start: false,
            obs_space: BoxSpace::unbounded(vec![obs_dim]),
            act_space,
            rng: StdRng::from_entropy(),
        })
    }

    /// Use a fixed RNG seed (only relevant with [`Self::with_random_start`]).
    pub fn seeded(mut self, seed: u64) -> Self {
        self.rng = StdRng::seed_from_u64(seed);
        self
    }

    /// Cap each episode to `len` rows (default: the full dataset).
    pub fn with_episode_len(mut self, len: usize) -> Self {
        self.episode_len = len.clamp(1, self.n_rows);
        self
    }

    /// Start each episode at a random row (decorrelates VecEnv replicas over a
    /// shared dataset). Default is sequential from row 0.
    pub fn with_random_start(mut self, yes: bool) -> Self {
        self.random_start = yes;
        self
    }

    /// Number of feature columns (observation dimension).
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Action dimension implied by the reward source.
    pub fn act_dim(&self) -> usize {
        self.act_dim
    }

    fn obs(&self, row: usize, device: &Device) -> Result<Tensor> {
        let base = row * self.obs_dim;
        let slice = &self.data[base..base + self.obs_dim];
        Ok(Tensor::from_slice(
            slice,
            &[self.obs_dim],
            &device.to_candle()?,
        )?)
    }

    /// Upper bound (exclusive) of the current episode's row window. `start` is
    /// the most recent reset position, so this respects `episode_len` caps even
    /// with random starts.
    fn episode_end(&self) -> usize {
        (self.start + self.episode_len).min(self.n_rows)
    }
}

impl Clone for ArrayEnv {
    fn clone(&self) -> Self {
        Self {
            data: self.data.clone(),
            reward: self.reward.clone(),
            obs_space: self.obs_space.clone(),
            act_space: self.act_space.clone(),
            rng: StdRng::from_entropy(),
            ..*self
        }
    }
}

impl Environment for ArrayEnv {
    type ObsSpace = BoxSpace;
    type ActSpace = BoxSpace;

    fn observation_space(&self) -> &BoxSpace {
        &self.obs_space
    }

    fn action_space(&self) -> &BoxSpace {
        &self.act_space
    }

    fn reset(&mut self, device: &Device) -> Result<Tensor> {
        // Leave room for at least one transition within the episode window.
        let max_start = self.n_rows.saturating_sub(1);
        self.start = if self.random_start && max_start > 0 {
            self.rng.gen_range(0..=max_start.min(self.n_rows - 1))
        } else {
            0
        };
        self.cursor = self.start;
        self.obs(self.cursor, device)
    }

    fn step(&mut self, action: &Tensor, device: &Device) -> Result<StepResult> {
        let act: Vec<f32> = action.flatten_all()?.to_vec1::<f32>()?;
        if act.len() < self.act_dim {
            return Err(OctaneError::InvalidConfig(format!(
                "action has {} elements, expected {}",
                act.len(),
                self.act_dim
            )));
        }
        let row = self.cursor;
        let reward = self.reward.reward(row, &act);

        self.cursor += 1;
        let truncated = self.cursor >= self.episode_end();
        let obs_row = self.cursor.min(self.n_rows - 1);

        Ok(StepResult {
            observation: self.obs(obs_row, device)?,
            reward,
            terminated: false,
            truncated,
            info: None,
        })
    }

    fn name(&self) -> &str {
        "ArrayEnv"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu() -> Device {
        Device::Cpu
    }

    #[test]
    fn regression_reward_is_negative_mse() {
        // 2 rows, obs_dim 1; targets [2.0, 3.0].
        let env = ArrayEnv::new(
            vec![0.0, 1.0],
            1,
            ArrayReward::Regression {
                targets: vec![2.0, 3.0],
                target_dim: 1,
            },
        )
        .unwrap();
        let mut env = env;
        let o0: Vec<f32> = env.reset(&cpu()).unwrap().to_vec1().unwrap();
        assert_eq!(o0, vec![0.0]);

        // Perfect prediction at row 0 -> reward 0.
        let a = Tensor::from_slice(&[2.0_f32], &[1], &cpu().to_candle().unwrap()).unwrap();
        let r0 = env.step(&a, &cpu()).unwrap();
        assert!((r0.reward - 0.0).abs() < 1e-6, "reward {}", r0.reward);
        let o1: Vec<f32> = r0.observation.to_vec1().unwrap();
        assert_eq!(o1, vec![1.0]);

        // Prediction 0 at row 1, target 3 -> mse = 9 -> reward -9.
        let a = Tensor::from_slice(&[0.0_f32], &[1], &cpu().to_candle().unwrap()).unwrap();
        let r1 = env.step(&a, &cpu()).unwrap();
        assert!((r1.reward - (-9.0)).abs() < 1e-6, "reward {}", r1.reward);
        assert!(r1.truncated, "episode should end after the last row");
    }

    #[test]
    fn weighted_reward_is_dot_product() {
        // 2 rows, obs_dim 2, returns row0 = [0.1, -0.2].
        let env = ArrayEnv::new(
            vec![1.0, 2.0, 3.0, 4.0],
            2,
            ArrayReward::Weighted {
                returns: vec![0.1, -0.2, 0.05, 0.05],
                n_assets: 2,
            },
        )
        .unwrap();
        let mut env = env;
        env.reset(&cpu()).unwrap();
        // action = [1, 0] -> dot = 0.1.
        let a = Tensor::from_slice(&[1.0_f32, 0.0], &[2], &cpu().to_candle().unwrap()).unwrap();
        let r = env.step(&a, &cpu()).unwrap();
        assert!((r.reward - 0.1).abs() < 1e-6, "reward {}", r.reward);
    }

    #[test]
    fn rejects_inconsistent_shapes() {
        // data 3 elems, obs_dim 2 -> not a multiple.
        assert!(ArrayEnv::new(
            vec![0.0, 1.0, 2.0],
            2,
            ArrayReward::Regression {
                targets: vec![0.0],
                target_dim: 1
            }
        )
        .is_err());
        // reward rows mismatch.
        assert!(ArrayEnv::new(
            vec![0.0, 1.0],
            1,
            ArrayReward::Regression {
                targets: vec![0.0, 0.0, 0.0],
                target_dim: 1
            }
        )
        .is_err());
    }

    #[test]
    fn action_dim_follows_reward() {
        let env = ArrayEnv::new(
            vec![0.0, 1.0, 2.0, 3.0],
            1,
            ArrayReward::Weighted {
                returns: vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                n_assets: 2,
            },
        )
        .unwrap();
        assert_eq!(env.act_dim(), 2);
        assert_eq!(env.obs_dim(), 1);
    }
}
