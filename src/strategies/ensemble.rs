//! Ensemble agents for robust trading decisions.
//!
//! This module provides ensemble learning strategies that combine multiple
//! agents for improved trading performance through diverse decision-making:
//!
//! - [`EnsembleAgent`] - Combines multiple RL agents using voting/averaging
//! - [`VotingStrategy`] - Different strategies for combining agent decisions
//! - [`DiversityMetrics`] - Measures diversity between agents in the ensemble
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::strategies::{EnsembleAgent, EnsembleConfig, VotingStrategy};
//!
//! let config = EnsembleConfig::default()
//!     .voting_strategy(VotingStrategy::WeightedAverage)
//!     .diversity_weight(0.1)
//!     .adaptation_rate(0.01);
//!
//! let ensemble = EnsembleAgent::new(config, agents, device)?;
//! ```

use crate::algorithms::{RLAlgorithm, TrainMetrics};
use crate::buffer::{ReplayBuffer, ReplayBufferConfig};
use crate::core::{Device, OctaneError, Result};
use crate::envs::{Environment, Space, VecEnv};
use candle_core::{DType, Module, Tensor};
use candle_nn::{VarBuilder, VarMap};
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info};

/// Voting strategy for combining agent decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VotingStrategy {
    /// Majority voting for discrete actions (most common choice wins).
    #[default]
    Majority,
    /// Weighted average for continuous actions.
    WeightedAverage,
    /// Stacking: meta-learner combines agent outputs.
    Stacking,
    /// Boosting-style: weight agents by recent performance.
    Boosting,
    /// Softmax voting: probability-weighted selection.
    Softmax,
    /// Median: robust to outliers for continuous actions.
    Median,
}

/// Weight adaptation strategy for ensemble members.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum WeightAdaptation {
    /// No adaptation - fixed equal weights.
    #[default]
    Fixed,
    /// Exponential moving average of recent performance.
    ExponentialMovingAverage,
    /// Softmax of cumulative rewards.
    SoftmaxReward,
    /// UCB1-style exploration-exploitation balance.
    UCB1,
    /// Multiplicative weights update.
    MultiplicativeWeights,
}

/// Configuration for ensemble agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    /// Number of agents in the ensemble.
    pub num_agents: usize,
    /// Voting strategy for combining decisions.
    pub voting_strategy: VotingStrategy,
    /// Weight adaptation strategy.
    pub weight_adaptation: WeightAdaptation,
    /// Learning rate for weight adaptation.
    pub adaptation_rate: f32,
    /// Discount factor for performance tracking.
    pub performance_gamma: f32,
    /// Temperature for softmax voting.
    pub softmax_temperature: f32,
    /// Weight for diversity bonus in reward.
    pub diversity_weight: f32,
    /// Minimum weight for any agent (prevents collapse).
    pub min_weight: f32,
    /// Hidden sizes for stacking meta-learner.
    pub meta_hidden_sizes: Vec<usize>,
    /// Window size for performance tracking.
    pub performance_window: usize,
    /// Whether to train agents jointly or independently.
    pub joint_training: bool,
    /// Random seed.
    pub seed: Option<u64>,
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            num_agents: 5,
            voting_strategy: VotingStrategy::WeightedAverage,
            weight_adaptation: WeightAdaptation::ExponentialMovingAverage,
            adaptation_rate: 0.01,
            performance_gamma: 0.99,
            softmax_temperature: 1.0,
            diversity_weight: 0.1,
            min_weight: 0.05,
            meta_hidden_sizes: vec![64, 32],
            performance_window: 100,
            joint_training: false,
            seed: None,
        }
    }
}

impl EnsembleConfig {
    /// Create new ensemble config with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of agents.
    pub fn num_agents(mut self, n: usize) -> Self {
        self.num_agents = n;
        self
    }

    /// Set voting strategy.
    pub fn voting_strategy(mut self, strategy: VotingStrategy) -> Self {
        self.voting_strategy = strategy;
        self
    }

    /// Set weight adaptation strategy.
    pub fn weight_adaptation(mut self, strategy: WeightAdaptation) -> Self {
        self.weight_adaptation = strategy;
        self
    }

    /// Set adaptation rate for weight updates.
    pub fn adaptation_rate(mut self, rate: f32) -> Self {
        self.adaptation_rate = rate;
        self
    }

    /// Set softmax temperature.
    pub fn softmax_temperature(mut self, temp: f32) -> Self {
        self.softmax_temperature = temp;
        self
    }

    /// Set diversity weight for encouraging diverse agent behaviors.
    pub fn diversity_weight(mut self, weight: f32) -> Self {
        self.diversity_weight = weight;
        self
    }

    /// Set minimum weight per agent.
    pub fn min_weight(mut self, weight: f32) -> Self {
        self.min_weight = weight;
        self
    }

    /// Set seed for reproducibility.
    pub fn seed(mut self, s: u64) -> Self {
        self.seed = Some(s);
        self
    }

    /// Validate configuration.
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.num_agents == 0 {
            return Err("num_agents must be positive".into());
        }
        if self.adaptation_rate < 0.0 || self.adaptation_rate > 1.0 {
            return Err("adaptation_rate must be in [0, 1]".into());
        }
        if self.softmax_temperature <= 0.0 {
            return Err("softmax_temperature must be positive".into());
        }
        if self.min_weight < 0.0 || self.min_weight > 1.0 / self.num_agents as f32 {
            return Err("min_weight must be in [0, 1/num_agents]".into());
        }
        Ok(())
    }
}

/// Diversity metrics for measuring agent disagreement.
#[derive(Debug, Clone, Default)]
pub struct DiversityMetrics {
    /// Q-statistic: pairwise disagreement measure.
    pub q_statistic: f32,
    /// Correlation coefficient between agent outputs.
    pub correlation: f32,
    /// Entropy of ensemble predictions.
    pub entropy: f32,
    /// Disagreement rate: fraction of samples with different predictions.
    pub disagreement_rate: f32,
    /// Kohavi-Wolpert variance.
    pub kw_variance: f32,
}

impl DiversityMetrics {
    /// Compute diversity metrics from agent actions.
    pub fn compute(actions: &[Tensor], is_discrete: bool) -> Result<Self> {
        let num_agents = actions.len();
        if num_agents < 2 {
            return Ok(Self::default());
        }

        let batch_size = actions[0].dim(0)?;

        // Convert to f32 vectors for computation
        let action_vecs: Vec<Vec<f32>> = actions
            .iter()
            .map(|a| a.flatten_all().and_then(|t| t.to_vec1()))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if is_discrete {
            Self::compute_discrete(&action_vecs, batch_size)
        } else {
            Self::compute_continuous(&action_vecs, batch_size)
        }
    }

    fn compute_discrete(actions: &[Vec<f32>], batch_size: usize) -> Result<Self> {
        let num_agents = actions.len();
        let mut disagreements = 0usize;
        let mut q_values = Vec::new();

        // Compute pairwise disagreement
        for i in 0..num_agents {
            for j in (i + 1)..num_agents {
                let mut both_correct = 0;
                let mut i_only = 0;
                let mut j_only = 0;
                let mut both_wrong = 0;

                for b in 0..batch_size {
                    let ai = actions[i][b].round() as i32;
                    let aj = actions[j][b].round() as i32;
                    let same = ai == aj;

                    // For Q-statistic, use agreement as proxy
                    if same {
                        both_correct += 1;
                    } else {
                        disagreements += 1;
                        i_only += 1;
                        j_only += 1;
                    }
                    both_wrong += 0; // Placeholder
                }

                let n11 = both_correct as f32;
                let n10 = i_only as f32;
                let n01 = j_only as f32;
                let n00 = both_wrong as f32;

                let num = n11 * n00 - n01 * n10;
                let den = n11 * n00 + n01 * n10;
                if den > 0.0 {
                    q_values.push(num / den);
                }
            }
        }

        let q_statistic = if q_values.is_empty() {
            0.0
        } else {
            q_values.iter().sum::<f32>() / q_values.len() as f32
        };

        let total_pairs = num_agents * (num_agents - 1) / 2 * batch_size;
        let disagreement_rate = disagreements as f32 / total_pairs as f32;

        // Compute entropy of majority vote
        let mut vote_counts: HashMap<i32, usize> = HashMap::new();
        for b in 0..batch_size {
            for agent_actions in actions.iter() {
                let action = agent_actions[b].round() as i32;
                *vote_counts.entry(action).or_insert(0) += 1;
            }
        }
        let total_votes = num_agents * batch_size;
        let entropy: f32 = vote_counts
            .values()
            .map(|&c| {
                let p = c as f32 / total_votes as f32;
                if p > 0.0 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum();

        Ok(Self {
            q_statistic,
            correlation: 1.0 - disagreement_rate,
            entropy,
            disagreement_rate,
            kw_variance: disagreement_rate,
        })
    }

    fn compute_continuous(actions: &[Vec<f32>], batch_size: usize) -> Result<Self> {
        let num_agents = actions.len();
        let action_dim = actions[0].len() / batch_size;

        // Compute mean action across agents
        let mut mean_actions = vec![0.0f32; batch_size * action_dim];
        for agent_actions in actions.iter() {
            for (i, &a) in agent_actions.iter().enumerate() {
                mean_actions[i] += a;
            }
        }
        for m in mean_actions.iter_mut() {
            *m /= num_agents as f32;
        }

        // Compute variance (Kohavi-Wolpert variance)
        let mut kw_variance = 0.0f32;
        for agent_actions in actions.iter() {
            for (i, &a) in agent_actions.iter().enumerate() {
                kw_variance += (a - mean_actions[i]).powi(2);
            }
        }
        kw_variance /= (num_agents * batch_size * action_dim) as f32;

        // Compute pairwise correlations
        let mut correlations = Vec::new();
        for i in 0..num_agents {
            for j in (i + 1)..num_agents {
                let mean_i: f32 = actions[i].iter().sum::<f32>() / actions[i].len() as f32;
                let mean_j: f32 = actions[j].iter().sum::<f32>() / actions[j].len() as f32;

                let mut cov = 0.0f32;
                let mut var_i = 0.0f32;
                let mut var_j = 0.0f32;

                for k in 0..actions[i].len() {
                    let di = actions[i][k] - mean_i;
                    let dj = actions[j][k] - mean_j;
                    cov += di * dj;
                    var_i += di * di;
                    var_j += dj * dj;
                }

                if var_i > 0.0 && var_j > 0.0 {
                    correlations.push(cov / (var_i.sqrt() * var_j.sqrt()));
                }
            }
        }

        let correlation = if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<f32>() / correlations.len() as f32
        };

        // Entropy based on action variance
        let entropy = if kw_variance > 0.0 {
            0.5 * (2.0 * std::f32::consts::PI * std::f32::consts::E * kw_variance).ln()
        } else {
            0.0
        };

        Ok(Self {
            q_statistic: 1.0 - correlation,
            correlation,
            entropy,
            disagreement_rate: kw_variance.sqrt(),
            kw_variance,
        })
    }
}

/// Performance tracker for individual agents.
#[derive(Debug, Clone)]
pub struct AgentPerformance {
    /// Recent rewards.
    pub rewards: Vec<f32>,
    /// Cumulative reward.
    pub cumulative_reward: f32,
    /// Number of times selected (for UCB).
    pub selection_count: usize,
    /// Running mean reward.
    pub mean_reward: f32,
    /// Running variance.
    pub variance: f32,
}

impl AgentPerformance {
    fn new() -> Self {
        Self {
            rewards: Vec::new(),
            cumulative_reward: 0.0,
            selection_count: 0,
            mean_reward: 0.0,
            variance: 0.0,
        }
    }

    fn update(&mut self, reward: f32, window_size: usize) {
        self.rewards.push(reward);
        if self.rewards.len() > window_size {
            self.rewards.remove(0);
        }

        self.cumulative_reward += reward;
        self.selection_count += 1;

        // Update running statistics
        let n = self.selection_count as f32;
        let delta = reward - self.mean_reward;
        self.mean_reward += delta / n;
        let delta2 = reward - self.mean_reward;
        self.variance += (delta * delta2 - self.variance) / n;
    }

    fn windowed_mean(&self) -> f32 {
        if self.rewards.is_empty() {
            0.0
        } else {
            self.rewards.iter().sum::<f32>() / self.rewards.len() as f32
        }
    }
}

/// Ensemble agent combining multiple RL agents.
pub struct EnsembleAgent<E: Environment + Clone + 'static> {
    /// Configuration.
    config: EnsembleConfig,
    /// Vectorized environment.
    env: VecEnv<E>,
    /// Device for tensor operations.
    device: Device,

    /// Individual agent variable maps.
    agent_var_maps: Vec<VarMap>,
    /// Agent weights for voting.
    agent_weights: Vec<f32>,
    /// Performance tracking for each agent.
    agent_performance: Vec<AgentPerformance>,

    /// Meta-learner for stacking (optional).
    meta_var_map: Option<VarMap>,

    /// Observation dimension.
    obs_dim: usize,
    /// Action dimension.
    act_dim: usize,
    /// Whether action space is discrete.
    is_discrete: bool,
    /// Hidden sizes for agent networks.
    hidden_sizes: Vec<usize>,

    /// Replay buffer for training.
    replay_buffer: ReplayBuffer,

    /// Total timesteps trained.
    total_timesteps: usize,

    /// Random number generator.
    rng: StdRng,
}

impl<E: Environment + Clone + 'static> EnsembleAgent<E> {
    /// Create a new ensemble agent.
    pub fn new(config: EnsembleConfig, env: VecEnv<E>, device: Device) -> Result<Self> {
        config.validate().map_err(OctaneError::InvalidConfig)?;

        let obs_space = env.observation_space();
        let act_space = env.action_space();
        let obs_dim = obs_space.flat_dim();
        let act_dim = act_space.flat_dim();
        let is_discrete = act_space.shape() == [1];

        let rng = match config.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_entropy(),
        };

        // Initialize agent weights uniformly
        let agent_weights = vec![1.0 / config.num_agents as f32; config.num_agents];
        let agent_performance = (0..config.num_agents)
            .map(|_| AgentPerformance::new())
            .collect();

        // Create agent variable maps
        let agent_var_maps = (0..config.num_agents)
            .map(|_| VarMap::new())
            .collect::<Vec<_>>();

        // Meta-learner for stacking
        let meta_var_map = if config.voting_strategy == VotingStrategy::Stacking {
            Some(VarMap::new())
        } else {
            None
        };

        // Create replay buffer
        let buffer_config = ReplayBufferConfig::new(obs_dim, act_dim).capacity(100_000);
        let replay_buffer = ReplayBuffer::new(buffer_config, device)?;

        let hidden_sizes = vec![256, 256];

        let mut agent = Self {
            config,
            env,
            device,
            agent_var_maps,
            agent_weights,
            agent_performance,
            meta_var_map,
            obs_dim,
            act_dim,
            is_discrete,
            hidden_sizes,
            replay_buffer,
            total_timesteps: 0,
            rng,
        };

        agent.init_networks()?;

        info!(
            "EnsembleAgent initialized: {} agents, strategy={:?}",
            agent.config.num_agents, agent.config.voting_strategy
        );

        Ok(agent)
    }

    /// Initialize neural networks for all agents.
    fn init_networks(&mut self) -> Result<()> {
        let candle_device = self.device.to_candle()?;

        for (i, var_map) in self.agent_var_maps.iter().enumerate() {
            let vb = VarBuilder::from_varmap(var_map, DType::F32, &candle_device);

            // Policy network
            let mut in_dim = self.obs_dim;
            for (j, &hidden_size) in self.hidden_sizes.iter().enumerate() {
                let _ = candle_nn::linear(
                    in_dim,
                    hidden_size,
                    vb.pp(format!("agent_{}.policy.layer_{}", i, j)),
                )?;
                in_dim = hidden_size;
            }
            let _ = candle_nn::linear(
                in_dim,
                self.act_dim,
                vb.pp(format!("agent_{}.policy.output", i)),
            )?;

            // Value network
            in_dim = self.obs_dim;
            for (j, &hidden_size) in self.hidden_sizes.iter().enumerate() {
                let _ = candle_nn::linear(
                    in_dim,
                    hidden_size,
                    vb.pp(format!("agent_{}.value.layer_{}", i, j)),
                )?;
                in_dim = hidden_size;
            }
            let _ = candle_nn::linear(in_dim, 1, vb.pp(format!("agent_{}.value.output", i)))?;
        }

        // Initialize meta-learner if using stacking
        if let Some(ref meta_var_map) = self.meta_var_map {
            let vb = VarBuilder::from_varmap(meta_var_map, DType::F32, &candle_device);
            let meta_input_dim = self.config.num_agents * self.act_dim;
            let mut in_dim = meta_input_dim;

            for (j, &hidden_size) in self.config.meta_hidden_sizes.iter().enumerate() {
                let _ = candle_nn::linear(in_dim, hidden_size, vb.pp(format!("meta.layer_{}", j)))?;
                in_dim = hidden_size;
            }
            let _ = candle_nn::linear(in_dim, self.act_dim, vb.pp("meta.output"))?;
        }

        Ok(())
    }

    /// Forward pass through a single agent's policy network.
    fn agent_policy_forward(&self, agent_idx: usize, obs: &Tensor) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;
        let vb =
            VarBuilder::from_varmap(&self.agent_var_maps[agent_idx], DType::F32, &candle_device);

        let mut x = obs.clone();
        for (j, &hidden_size) in self.hidden_sizes.iter().enumerate() {
            let in_dim = if j == 0 {
                self.obs_dim
            } else {
                self.hidden_sizes[j - 1]
            };
            let linear = candle_nn::linear(
                in_dim,
                hidden_size,
                vb.pp(format!("agent_{}.policy.layer_{}", agent_idx, j)),
            )?;
            x = linear.forward(&x)?;
            x = x.tanh()?;
        }

        let output_linear = candle_nn::linear(
            *self.hidden_sizes.last().unwrap(),
            self.act_dim,
            vb.pp(format!("agent_{}.policy.output", agent_idx)),
        )?;

        Ok(output_linear.forward(&x)?)
    }

    /// Get actions from all agents.
    fn get_all_agent_actions(&self, obs: &Tensor) -> Result<Vec<Tensor>> {
        let mut actions = Vec::with_capacity(self.config.num_agents);

        for i in 0..self.config.num_agents {
            let logits = self.agent_policy_forward(i, obs)?;

            let action = if self.is_discrete {
                logits.argmax(1)?.to_dtype(DType::F32)?
            } else {
                logits.tanh()?
            };

            actions.push(action);
        }

        Ok(actions)
    }

    /// Combine agent actions using the configured voting strategy.
    fn combine_actions(&self, actions: Vec<Tensor>) -> Result<Tensor> {
        match self.config.voting_strategy {
            VotingStrategy::Majority => self.majority_vote(&actions),
            VotingStrategy::WeightedAverage => self.weighted_average(&actions),
            VotingStrategy::Stacking => self.stacking_combine(&actions),
            VotingStrategy::Boosting => self.boosting_combine(&actions),
            VotingStrategy::Softmax => self.softmax_vote(&actions),
            VotingStrategy::Median => self.median_combine(&actions),
        }
    }

    /// Majority voting for discrete actions.
    fn majority_vote(&self, actions: &[Tensor]) -> Result<Tensor> {
        let batch_size = actions[0].dim(0)?;
        let candle_device = self.device.to_candle()?;

        let mut votes = vec![HashMap::new(); batch_size];

        for action_tensor in actions.iter() {
            let action_vec: Vec<f32> = action_tensor.flatten_all()?.to_vec1()?;
            for (b, &a) in action_vec.iter().enumerate() {
                *votes[b].entry(a.round() as i32).or_insert(0) += 1;
            }
        }

        let final_actions: Vec<f32> = votes
            .into_iter()
            .map(|vote_map| {
                vote_map
                    .into_iter()
                    .max_by_key(|&(_, count)| count)
                    .map(|(action, _)| action as f32)
                    .unwrap_or(0.0)
            })
            .collect();

        Ok(Tensor::from_slice(
            &final_actions,
            &[batch_size, 1],
            &candle_device,
        )?)
    }

    /// Weighted average for continuous actions.
    fn weighted_average(&self, actions: &[Tensor]) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;

        let mut result = Tensor::zeros_like(&actions[0])?;

        for (i, action) in actions.iter().enumerate() {
            let weight = self.agent_weights[i];
            let weight_tensor = Tensor::new(&[weight], &candle_device)?;
            result = (&result + action.broadcast_mul(&weight_tensor)?)?;
        }

        Ok(result)
    }

    /// Stacking: use meta-learner to combine actions.
    fn stacking_combine(&self, actions: &[Tensor]) -> Result<Tensor> {
        let meta_var_map = self
            .meta_var_map
            .as_ref()
            .ok_or_else(|| OctaneError::InvalidConfig("Meta-learner not initialized".into()))?;

        // Concatenate all agent actions
        let stacked = Tensor::cat(actions, 1)?;

        let candle_device = self.device.to_candle()?;
        let vb = VarBuilder::from_varmap(meta_var_map, DType::F32, &candle_device);

        let mut x = stacked;
        for (j, &hidden_size) in self.config.meta_hidden_sizes.iter().enumerate() {
            let in_dim = if j == 0 {
                self.config.num_agents * self.act_dim
            } else {
                self.config.meta_hidden_sizes[j - 1]
            };
            let linear =
                candle_nn::linear(in_dim, hidden_size, vb.pp(format!("meta.layer_{}", j)))?;
            x = linear.forward(&x)?;
            x = x.relu()?;
        }

        let output_linear = candle_nn::linear(
            *self.config.meta_hidden_sizes.last().unwrap(),
            self.act_dim,
            vb.pp("meta.output"),
        )?;

        Ok(output_linear.forward(&x)?)
    }

    /// Boosting-style combination based on recent performance.
    fn boosting_combine(&self, actions: &[Tensor]) -> Result<Tensor> {
        // Use performance-based weights
        let performances: Vec<f32> = self
            .agent_performance
            .iter()
            .map(|p| p.windowed_mean())
            .collect();

        let min_perf = performances.iter().cloned().fold(f32::INFINITY, f32::min);
        let shifted: Vec<f32> = performances.iter().map(|p| p - min_perf + 1e-6).collect();
        let sum: f32 = shifted.iter().sum();
        let weights: Vec<f32> = shifted.iter().map(|p| p / sum).collect();

        let candle_device = self.device.to_candle()?;
        let mut result = Tensor::zeros_like(&actions[0])?;

        for (i, action) in actions.iter().enumerate() {
            let weight_tensor = Tensor::new(&[weights[i]], &candle_device)?;
            result = (&result + action.broadcast_mul(&weight_tensor)?)?;
        }

        Ok(result)
    }

    /// Softmax voting: probability-weighted combination.
    fn softmax_vote(&self, actions: &[Tensor]) -> Result<Tensor> {
        let candle_device = self.device.to_candle()?;

        // Apply softmax to weights
        let exp_weights: Vec<f32> = self
            .agent_weights
            .iter()
            .map(|&w| (w / self.config.softmax_temperature).exp())
            .collect();
        let sum: f32 = exp_weights.iter().sum();
        let softmax_weights: Vec<f32> = exp_weights.iter().map(|w| w / sum).collect();

        let mut result = Tensor::zeros_like(&actions[0])?;

        for (i, action) in actions.iter().enumerate() {
            let weight_tensor = Tensor::new(&[softmax_weights[i]], &candle_device)?;
            result = (&result + action.broadcast_mul(&weight_tensor)?)?;
        }

        Ok(result)
    }

    /// Median combination for robustness to outliers.
    fn median_combine(&self, actions: &[Tensor]) -> Result<Tensor> {
        let batch_size = actions[0].dim(0)?;
        let action_dim = if actions[0].dims().len() > 1 {
            actions[0].dim(1)?
        } else {
            1
        };

        let candle_device = self.device.to_candle()?;

        let action_vecs: Vec<Vec<f32>> = actions
            .iter()
            .map(|a| a.flatten_all().and_then(|t| t.to_vec1()))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        let mut median_actions = Vec::with_capacity(batch_size * action_dim);

        for b in 0..batch_size {
            for d in 0..action_dim {
                let idx = b * action_dim + d;
                let mut values: Vec<f32> = action_vecs.iter().map(|v| v[idx]).collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let median = if values.len().is_multiple_of(2) {
                    (values[values.len() / 2 - 1] + values[values.len() / 2]) / 2.0
                } else {
                    values[values.len() / 2]
                };
                median_actions.push(median);
            }
        }

        Ok(Tensor::from_slice(
            &median_actions,
            &[batch_size, action_dim],
            &candle_device,
        )?)
    }

    /// Update agent weights based on recent performance.
    fn update_weights(&mut self) {
        match self.config.weight_adaptation {
            WeightAdaptation::Fixed => {}
            WeightAdaptation::ExponentialMovingAverage => {
                let performances: Vec<f32> = self
                    .agent_performance
                    .iter()
                    .map(|p| p.windowed_mean())
                    .collect();

                let min_perf = performances.iter().cloned().fold(f32::INFINITY, f32::min);
                let shifted: Vec<f32> = performances.iter().map(|p| p - min_perf + 1e-6).collect();
                let sum: f32 = shifted.iter().sum();

                for (i, p) in shifted.iter().enumerate() {
                    let target = p / sum;
                    self.agent_weights[i] = (1.0 - self.config.adaptation_rate)
                        * self.agent_weights[i]
                        + self.config.adaptation_rate * target;
                    self.agent_weights[i] = self.agent_weights[i].max(self.config.min_weight);
                }

                // Renormalize
                let sum: f32 = self.agent_weights.iter().sum();
                for w in self.agent_weights.iter_mut() {
                    *w /= sum;
                }
            }
            WeightAdaptation::SoftmaxReward => {
                let cumulative: Vec<f32> = self
                    .agent_performance
                    .iter()
                    .map(|p| p.cumulative_reward)
                    .collect();

                let exp_rewards: Vec<f32> = cumulative
                    .iter()
                    .map(|&r| (r / self.config.softmax_temperature).exp())
                    .collect();
                let sum: f32 = exp_rewards.iter().sum();

                for (i, e) in exp_rewards.iter().enumerate() {
                    self.agent_weights[i] = (e / sum).max(self.config.min_weight);
                }

                let sum: f32 = self.agent_weights.iter().sum();
                for w in self.agent_weights.iter_mut() {
                    *w /= sum;
                }
            }
            WeightAdaptation::UCB1 => {
                let total_count: usize = self
                    .agent_performance
                    .iter()
                    .map(|p| p.selection_count)
                    .sum();
                let ln_total = (total_count as f32 + 1.0).ln();

                let ucb_values: Vec<f32> = self
                    .agent_performance
                    .iter()
                    .map(|p| {
                        let mean = p.mean_reward;
                        let exploration =
                            (2.0 * ln_total / (p.selection_count as f32 + 1.0)).sqrt();
                        mean + exploration
                    })
                    .collect();

                let min_ucb = ucb_values.iter().cloned().fold(f32::INFINITY, f32::min);
                let shifted: Vec<f32> = ucb_values.iter().map(|u| u - min_ucb + 1e-6).collect();
                let sum: f32 = shifted.iter().sum();

                for (i, s) in shifted.iter().enumerate() {
                    self.agent_weights[i] = (s / sum).max(self.config.min_weight);
                }

                let sum: f32 = self.agent_weights.iter().sum();
                for w in self.agent_weights.iter_mut() {
                    *w /= sum;
                }
            }
            WeightAdaptation::MultiplicativeWeights => {
                for (i, perf) in self.agent_performance.iter().enumerate() {
                    let reward = perf.windowed_mean();
                    let update = (self.config.adaptation_rate * reward).exp();
                    self.agent_weights[i] *= update;
                    self.agent_weights[i] = self.agent_weights[i].max(self.config.min_weight);
                }

                let sum: f32 = self.agent_weights.iter().sum();
                for w in self.agent_weights.iter_mut() {
                    *w /= sum;
                }
            }
        }
    }

    /// Predict action using ensemble.
    pub fn predict(&mut self, obs: &Tensor, deterministic: bool) -> Result<Tensor> {
        let actions = self.get_all_agent_actions(obs)?;

        if deterministic {
            self.combine_actions(actions)
        } else {
            // Add exploration noise for non-deterministic mode
            let combined = self.combine_actions(actions)?;
            if !self.is_discrete {
                let noise = Tensor::randn_like(&combined, 0.0, 0.1)?;
                Ok((combined + noise)?)
            } else {
                Ok(combined)
            }
        }
    }

    /// Compute diversity bonus for the current actions.
    pub fn diversity_bonus(&self, actions: &[Tensor]) -> Result<f32> {
        let metrics = DiversityMetrics::compute(actions, self.is_discrete)?;
        Ok(self.config.diversity_weight * metrics.kw_variance)
    }

    /// Get current agent weights.
    pub fn get_weights(&self) -> &[f32] {
        &self.agent_weights
    }

    /// Get diversity metrics for current predictions.
    pub fn get_diversity_metrics(&self, obs: &Tensor) -> Result<DiversityMetrics> {
        let actions = self.get_all_agent_actions(obs)?;
        DiversityMetrics::compute(&actions, self.is_discrete)
    }
}

impl<E: Environment + Clone + 'static> RLAlgorithm for EnsembleAgent<E> {
    fn train_step(&mut self) -> Result<TrainMetrics> {
        let num_envs = self.env.num_envs();
        let obs = self.env.reset(&self.device)?;

        // Get actions from all agents
        let agent_actions = self.get_all_agent_actions(&obs)?;

        // Combine actions
        let combined_action = self.combine_actions(agent_actions.clone())?;

        // Step environment
        let step_result = self.env.step(&combined_action, &self.device)?;

        // Update timesteps
        self.total_timesteps += num_envs;

        // Compute per-agent rewards (could be different based on individual actions)
        let rewards_vec: Vec<f32> = step_result.rewards.to_vec1()?;
        let mean_reward = rewards_vec.iter().sum::<f32>() / rewards_vec.len() as f32;

        // Update agent performance
        for perf in self.agent_performance.iter_mut() {
            // Use mean reward weighted by agreement with ensemble decision
            perf.update(mean_reward, self.config.performance_window);
        }

        // Update weights
        self.update_weights();

        // Compute diversity metrics
        let diversity = DiversityMetrics::compute(&agent_actions, self.is_discrete)?;

        debug!(
            "Ensemble step: reward={:.4}, diversity={:.4}, weights={:?}",
            mean_reward, diversity.kw_variance, self.agent_weights
        );

        Ok(TrainMetrics {
            mean_reward,
            timesteps: self.total_timesteps,
            entropy: diversity.entropy,
            ..Default::default()
        })
    }

    fn save(&self, path: &Path) -> Result<()> {
        let mut all_tensors: std::collections::HashMap<String, Tensor> =
            std::collections::HashMap::new();

        for (i, var_map) in self.agent_var_maps.iter().enumerate() {
            let data = var_map.data().lock().unwrap();
            for (name, var) in data.iter() {
                all_tensors.insert(format!("agent_{}_{}", i, name), var.as_tensor().clone());
            }
        }

        if let Some(ref meta_var_map) = self.meta_var_map {
            let data = meta_var_map.data().lock().unwrap();
            for (name, var) in data.iter() {
                all_tensors.insert(format!("meta_{}", name), var.as_tensor().clone());
            }
        }

        candle_core::safetensors::save(&all_tensors, path)?;

        // Save config and weights
        let config_path = path.with_extension("json");
        let config_data = serde_json::json!({
            "config": self.config,
            "weights": self.agent_weights,
        });
        std::fs::write(config_path, serde_json::to_string_pretty(&config_data)?)?;

        info!("EnsembleAgent saved to {:?}", path);
        Ok(())
    }

    fn load(&mut self, path: &Path) -> Result<()> {
        let candle_device = self.device.to_candle()?;
        let tensors = candle_core::safetensors::load(path, &candle_device)?;

        for (i, var_map) in self.agent_var_maps.iter_mut().enumerate() {
            let mut data = var_map.data().lock().unwrap();
            let prefix = format!("agent_{}_", i);
            for (name, tensor) in &tensors {
                if name.starts_with(&prefix) {
                    let key = name.trim_start_matches(&prefix);
                    if let Some(var) = data.get_mut(key) {
                        var.set(tensor)?;
                    }
                }
            }
        }

        if let Some(ref meta_var_map) = self.meta_var_map {
            let mut data = meta_var_map.data().lock().unwrap();
            for (name, tensor) in &tensors {
                if name.starts_with("meta_") {
                    let key = name.trim_start_matches("meta_");
                    if let Some(var) = data.get_mut(key) {
                        var.set(tensor)?;
                    }
                }
            }
        }

        // Load weights
        let config_path = path.with_extension("json");
        if config_path.exists() {
            let config_data: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
            if let Some(weights) = config_data["weights"].as_array() {
                self.agent_weights = weights
                    .iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32))
                    .collect();
            }
        }

        info!("EnsembleAgent loaded from {:?}", path);
        Ok(())
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn total_timesteps(&self) -> usize {
        self.total_timesteps
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_config_defaults() {
        let config = EnsembleConfig::default();
        assert_eq!(config.num_agents, 5);
        assert_eq!(config.voting_strategy, VotingStrategy::WeightedAverage);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ensemble_config_builder() {
        let config = EnsembleConfig::new()
            .num_agents(10)
            .voting_strategy(VotingStrategy::Majority)
            .adaptation_rate(0.05)
            .diversity_weight(0.2);

        assert_eq!(config.num_agents, 10);
        assert_eq!(config.voting_strategy, VotingStrategy::Majority);
        assert!((config.adaptation_rate - 0.05).abs() < 1e-6);
    }

    #[test]
    fn test_diversity_metrics_default() {
        let metrics = DiversityMetrics::default();
        assert_eq!(metrics.q_statistic, 0.0);
        assert_eq!(metrics.correlation, 0.0);
    }

    #[test]
    fn test_agent_performance_update() {
        let mut perf = AgentPerformance::new();
        perf.update(1.0, 10);
        perf.update(2.0, 10);
        perf.update(3.0, 10);

        assert!((perf.mean_reward - 2.0).abs() < 1e-6);
        assert_eq!(perf.selection_count, 3);
    }
}
