//! Multi-agent environment support for cooperative and competitive RL.
//!
//! This module provides traits and utilities for multi-agent reinforcement
//! learning, supporting both decentralized execution and centralized training
//! (CTDE) paradigms.
//!
//! # Key Concepts
//!
//! - **AgentId**: Unique identifier for each agent in the environment.
//! - **MultiAgentEnv**: Extension of Environment for multi-agent scenarios.
//! - **CentralizedCritic**: Trait for value functions that see all agents' observations.
//! - **JointAction**: Combined actions from all agents.
//!
//! # Example
//! ```ignore
//! use octane::envs::{MultiAgentEnv, AgentId};
//!
//! struct TwoPlayerGame { ... }
//!
//! impl MultiAgentEnv for TwoPlayerGame {
//!     fn agents(&self) -> &[AgentId] { &self.agent_ids }
//!     // ...
//! }
//! ```

use crate::core::{Device, OctaneError, Result};
use crate::envs::{BoxSpace, StepInfo};
use candle_core::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Unique identifier for an agent in a multi-agent environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub usize);

impl AgentId {
    /// Create a new agent ID.
    pub fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the underlying ID value.
    pub fn value(&self) -> usize {
        self.0
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Agent_{}", self.0)
    }
}

impl From<usize> for AgentId {
    fn from(id: usize) -> Self {
        Self(id)
    }
}

/// Per-agent observations in a multi-agent environment.
#[derive(Debug)]
pub struct MultiAgentObs {
    /// Observations for each agent.
    observations: HashMap<AgentId, Tensor>,
    /// Optional global state (for centralized training).
    global_state: Option<Tensor>,
}

impl MultiAgentObs {
    /// Create a new multi-agent observation.
    pub fn new(observations: HashMap<AgentId, Tensor>) -> Self {
        Self {
            observations,
            global_state: None,
        }
    }

    /// Create with global state.
    pub fn with_global_state(mut self, state: Tensor) -> Self {
        self.global_state = Some(state);
        self
    }

    /// Get observation for a specific agent.
    pub fn get(&self, agent_id: &AgentId) -> Option<&Tensor> {
        self.observations.get(agent_id)
    }

    /// Get all observations.
    pub fn observations(&self) -> &HashMap<AgentId, Tensor> {
        &self.observations
    }

    /// Get global state (if available).
    pub fn global_state(&self) -> Option<&Tensor> {
        self.global_state.as_ref()
    }

    /// Number of agents with observations.
    pub fn num_agents(&self) -> usize {
        self.observations.len()
    }

    /// Stack all observations into a single tensor.
    /// Returns tensor of shape [num_agents, ...obs_shape].
    pub fn stacked(&self, agent_order: &[AgentId]) -> Result<Tensor> {
        let obs_vec: Vec<&Tensor> = agent_order
            .iter()
            .filter_map(|id| self.observations.get(id))
            .collect();

        if obs_vec.is_empty() {
            return Err(OctaneError::Environment(
                "No observations to stack".to_string(),
            ));
        }

        Tensor::stack(&obs_vec, 0).map_err(Into::into)
    }
}

/// Per-agent rewards in a multi-agent environment.
#[derive(Debug, Clone)]
pub struct MultiAgentReward {
    /// Rewards for each agent.
    rewards: HashMap<AgentId, f32>,
    /// Optional team reward (shared by all agents).
    team_reward: Option<f32>,
}

impl MultiAgentReward {
    /// Create a new multi-agent reward.
    pub fn new(rewards: HashMap<AgentId, f32>) -> Self {
        Self {
            rewards,
            team_reward: None,
        }
    }

    /// Create with team reward.
    pub fn with_team_reward(mut self, reward: f32) -> Self {
        self.team_reward = Some(reward);
        self
    }

    /// Create uniform reward for all agents.
    pub fn uniform(agents: &[AgentId], reward: f32) -> Self {
        let rewards = agents.iter().map(|&id| (id, reward)).collect();
        Self {
            rewards,
            team_reward: None,
        }
    }

    /// Get reward for a specific agent.
    pub fn get(&self, agent_id: &AgentId) -> Option<f32> {
        self.rewards.get(agent_id).copied()
    }

    /// Get all rewards.
    pub fn rewards(&self) -> &HashMap<AgentId, f32> {
        &self.rewards
    }

    /// Get team reward (if available).
    pub fn team_reward(&self) -> Option<f32> {
        self.team_reward
    }

    /// Get total reward (sum of all agent rewards + team reward).
    pub fn total(&self) -> f32 {
        let individual: f32 = self.rewards.values().sum();
        individual + self.team_reward.unwrap_or(0.0)
    }

    /// Get mean reward across agents.
    pub fn mean(&self) -> f32 {
        if self.rewards.is_empty() {
            return 0.0;
        }
        let sum: f32 = self.rewards.values().sum();
        sum / self.rewards.len() as f32
    }
}

/// Per-agent done flags in a multi-agent environment.
#[derive(Debug, Clone)]
pub struct MultiAgentDone {
    /// Termination flags for each agent.
    terminated: HashMap<AgentId, bool>,
    /// Truncation flags for each agent.
    truncated: HashMap<AgentId, bool>,
    /// Whether the entire environment episode is done.
    env_done: bool,
}

impl MultiAgentDone {
    /// Create a new multi-agent done status.
    pub fn new(terminated: HashMap<AgentId, bool>, truncated: HashMap<AgentId, bool>) -> Self {
        let env_done = terminated.values().any(|&t| t) || truncated.values().any(|&t| t);
        Self {
            terminated,
            truncated,
            env_done,
        }
    }

    /// Create with environment done flag.
    pub fn with_env_done(mut self, done: bool) -> Self {
        self.env_done = done;
        self
    }

    /// Check if a specific agent is done.
    pub fn is_done(&self, agent_id: &AgentId) -> bool {
        let terminated = self.terminated.get(agent_id).copied().unwrap_or(false);
        let truncated = self.truncated.get(agent_id).copied().unwrap_or(false);
        terminated || truncated
    }

    /// Check if a specific agent terminated.
    pub fn is_terminated(&self, agent_id: &AgentId) -> bool {
        self.terminated.get(agent_id).copied().unwrap_or(false)
    }

    /// Check if a specific agent was truncated.
    pub fn is_truncated(&self, agent_id: &AgentId) -> bool {
        self.truncated.get(agent_id).copied().unwrap_or(false)
    }

    /// Check if the entire environment is done.
    pub fn env_done(&self) -> bool {
        self.env_done
    }

    /// Check if all agents are done.
    pub fn all_done(&self) -> bool {
        self.terminated.values().all(|&t| t) || self.truncated.values().all(|&t| t)
    }

    /// Check if any agent is done.
    pub fn any_done(&self) -> bool {
        self.terminated.values().any(|&t| t) || self.truncated.values().any(|&t| t)
    }

    /// Get list of active (not done) agents.
    pub fn active_agents(&self) -> Vec<AgentId> {
        self.terminated
            .keys()
            .filter(|id| !self.is_done(id))
            .copied()
            .collect()
    }
}

/// Joint action from all agents.
#[derive(Debug)]
pub struct JointAction {
    /// Actions for each agent.
    actions: HashMap<AgentId, Tensor>,
}

impl JointAction {
    /// Create a new joint action.
    pub fn new(actions: HashMap<AgentId, Tensor>) -> Self {
        Self { actions }
    }

    /// Get action for a specific agent.
    pub fn get(&self, agent_id: &AgentId) -> Option<&Tensor> {
        self.actions.get(agent_id)
    }

    /// Get all actions.
    pub fn actions(&self) -> &HashMap<AgentId, Tensor> {
        &self.actions
    }

    /// Stack all actions into a single tensor.
    pub fn stacked(&self, agent_order: &[AgentId]) -> Result<Tensor> {
        let action_vec: Vec<&Tensor> = agent_order
            .iter()
            .filter_map(|id| self.actions.get(id))
            .collect();

        if action_vec.is_empty() {
            return Err(OctaneError::Environment("No actions to stack".to_string()));
        }

        Tensor::stack(&action_vec, 0).map_err(Into::into)
    }

    /// Create from a stacked tensor and agent order.
    pub fn from_stacked(stacked: &Tensor, agent_order: &[AgentId]) -> Result<Self> {
        let num_agents = agent_order.len();
        let mut actions = HashMap::with_capacity(num_agents);

        for (i, agent_id) in agent_order.iter().enumerate() {
            let action = stacked.get(i)?;
            actions.insert(*agent_id, action);
        }

        Ok(Self { actions })
    }
}

/// Result of a multi-agent environment step.
#[derive(Debug)]
pub struct MultiAgentStepResult {
    /// Per-agent observations.
    pub observations: MultiAgentObs,
    /// Per-agent rewards.
    pub rewards: MultiAgentReward,
    /// Per-agent done flags.
    pub dones: MultiAgentDone,
    /// Additional info.
    pub info: Option<MultiAgentInfo>,
}

/// Additional info for multi-agent step.
#[derive(Debug, Clone, Default)]
pub struct MultiAgentInfo {
    /// Per-agent step info.
    pub agent_info: HashMap<AgentId, StepInfo>,
    /// Global/shared info.
    pub extra: HashMap<String, f32>,
}

impl MultiAgentInfo {
    /// Create new multi-agent info.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add info for a specific agent.
    pub fn with_agent_info(mut self, agent_id: AgentId, info: StepInfo) -> Self {
        self.agent_info.insert(agent_id, info);
        self
    }

    /// Add global info.
    pub fn with_extra(mut self, key: &str, value: f32) -> Self {
        self.extra.insert(key.to_string(), value);
        self
    }
}

/// Space type enumeration for multi-agent environments.
#[derive(Debug, Clone)]
pub enum MultiAgentSpace {
    /// Homogeneous: all agents share the same space.
    Homogeneous(BoxSpace),
    /// Heterogeneous: each agent has a different space.
    Heterogeneous(HashMap<AgentId, BoxSpace>),
}

impl MultiAgentSpace {
    /// Get space for a specific agent.
    pub fn get(&self, agent_id: &AgentId) -> Option<&BoxSpace> {
        match self {
            MultiAgentSpace::Homogeneous(space) => Some(space),
            MultiAgentSpace::Heterogeneous(spaces) => spaces.get(agent_id),
        }
    }

    /// Check if all agents share the same space.
    pub fn is_homogeneous(&self) -> bool {
        matches!(self, MultiAgentSpace::Homogeneous(_))
    }
}

/// Core trait for multi-agent environments.
///
/// This extends the single-agent Environment concept to support
/// multiple agents with potentially different observation and action spaces.
pub trait MultiAgentEnv: Send + Sync + 'static {
    /// Get list of all agent IDs.
    fn agents(&self) -> &[AgentId];

    /// Get number of agents.
    fn num_agents(&self) -> usize {
        self.agents().len()
    }

    /// Get observation space for an agent.
    fn observation_space(&self, agent_id: &AgentId) -> Option<&BoxSpace>;

    /// Get action space for an agent.
    fn action_space(&self, agent_id: &AgentId) -> Option<&BoxSpace>;

    /// Get the global observation space (for centralized training).
    fn global_state_space(&self) -> Option<&BoxSpace> {
        None
    }

    /// Reset the environment and return initial observations.
    fn reset(&mut self, device: &Device) -> Result<MultiAgentObs>;

    /// Take a step with joint actions from all agents.
    fn step(&mut self, actions: &JointAction, device: &Device) -> Result<MultiAgentStepResult>;

    /// Render the environment.
    fn render(&self) -> Result<()> {
        Ok(())
    }

    /// Close the environment.
    fn close(&mut self) -> Result<()> {
        Ok(())
    }

    /// Get environment name.
    fn name(&self) -> &str {
        "MultiAgentEnv"
    }

    /// Check if agents are cooperative (shared reward).
    fn is_cooperative(&self) -> bool {
        false
    }

    /// Check if environment is zero-sum (competitive).
    fn is_zero_sum(&self) -> bool {
        false
    }

    /// Get the global state (for centralized training).
    /// Returns None if CTDE is not supported.
    fn get_global_state(&self, _device: &Device) -> Result<Option<Tensor>> {
        Ok(None)
    }
}

/// Trait for centralized critics in CTDE (Centralized Training, Decentralized Execution).
///
/// A centralized critic has access to all agents' observations and actions
/// during training, but individual agents only use their local observations
/// for action selection during execution.
pub trait CentralizedCritic: Send + Sync {
    /// Evaluate the joint value function.
    ///
    /// # Arguments
    /// * `global_state` - The global state (all agents' observations concatenated or other representation).
    /// * `joint_actions` - Optional joint action for Q-value estimation.
    ///
    /// # Returns
    /// Value(s) for the given state (and actions if provided).
    fn value(&self, global_state: &Tensor, joint_actions: Option<&Tensor>) -> Result<Tensor>;

    /// Evaluate per-agent values with counterfactual baselines.
    ///
    /// Used in methods like COMA (Counterfactual Multi-Agent Policy Gradients).
    fn counterfactual_values(
        &self,
        global_state: &Tensor,
        joint_actions: &Tensor,
        agent_id: AgentId,
    ) -> Result<Tensor>;
}

/// Adapter to convert a multi-agent environment to a single-agent view.
///
/// Useful for training a single agent in a multi-agent environment while
/// using fixed policies for other agents.
pub struct SingleAgentAdapter<M: MultiAgentEnv> {
    /// The multi-agent environment.
    env: M,
    /// The agent being controlled.
    controlled_agent: AgentId,
    /// Observation space.
    obs_space: BoxSpace,
    /// Action space.
    act_space: BoxSpace,
}

impl<M: MultiAgentEnv> SingleAgentAdapter<M> {
    /// Create a single-agent view of a multi-agent environment.
    ///
    /// # Arguments
    /// * `env` - The multi-agent environment.
    /// * `agent_id` - The agent to control.
    pub fn new(env: M, agent_id: AgentId) -> Result<Self> {
        let obs_space = env
            .observation_space(&agent_id)
            .ok_or_else(|| OctaneError::Environment(format!("Agent {:?} not found", agent_id)))?
            .clone();

        let act_space = env
            .action_space(&agent_id)
            .ok_or_else(|| OctaneError::Environment(format!("Agent {:?} not found", agent_id)))?
            .clone();

        Ok(Self {
            env,
            controlled_agent: agent_id,
            obs_space,
            act_space,
        })
    }

    /// Get the controlled agent ID.
    pub fn controlled_agent(&self) -> AgentId {
        self.controlled_agent
    }

    /// Get reference to the underlying multi-agent environment.
    pub fn inner(&self) -> &M {
        &self.env
    }

    /// Get mutable reference to the underlying environment.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.env
    }
}

/// Parameter sharing configuration for multi-agent training.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSharingConfig {
    /// Whether to share policy parameters across agents.
    pub share_policy: bool,
    /// Whether to share value function parameters.
    pub share_value: bool,
    /// Agent groups that share parameters (if not full sharing).
    pub agent_groups: Option<Vec<Vec<AgentId>>>,
}

impl Default for ParameterSharingConfig {
    fn default() -> Self {
        Self {
            share_policy: true,
            share_value: true,
            agent_groups: None,
        }
    }
}

impl ParameterSharingConfig {
    /// Create config for no parameter sharing.
    pub fn independent() -> Self {
        Self {
            share_policy: false,
            share_value: false,
            agent_groups: None,
        }
    }

    /// Create config for full parameter sharing.
    pub fn full_sharing() -> Self {
        Self {
            share_policy: true,
            share_value: true,
            agent_groups: None,
        }
    }

    /// Create config for group-based sharing.
    pub fn group_sharing(groups: Vec<Vec<AgentId>>) -> Self {
        Self {
            share_policy: true,
            share_value: true,
            agent_groups: Some(groups),
        }
    }
}

/// Communication configuration for multi-agent environments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationConfig {
    /// Whether communication is enabled.
    pub enabled: bool,
    /// Size of the message vector.
    pub message_size: usize,
    /// Whether messages are discrete or continuous.
    pub discrete_messages: bool,
    /// Number of discrete message tokens (if discrete).
    pub vocab_size: Option<usize>,
    /// Whether communication is learned or fixed.
    pub learned: bool,
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            message_size: 0,
            discrete_messages: false,
            vocab_size: None,
            learned: true,
        }
    }
}

impl CommunicationConfig {
    /// Create config for continuous communication.
    pub fn continuous(message_size: usize) -> Self {
        Self {
            enabled: true,
            message_size,
            discrete_messages: false,
            vocab_size: None,
            learned: true,
        }
    }

    /// Create config for discrete communication.
    pub fn discrete(message_size: usize, vocab_size: usize) -> Self {
        Self {
            enabled: true,
            message_size,
            discrete_messages: true,
            vocab_size: Some(vocab_size),
            learned: true,
        }
    }
}

/// Helper functions for multi-agent training.
pub mod utils {
    use super::*;

    /// Stack observations from multiple agents into a batch tensor.
    pub fn stack_agent_obs(obs: &MultiAgentObs, agent_order: &[AgentId]) -> Result<Tensor> {
        obs.stacked(agent_order)
    }

    /// Split a batch tensor back into per-agent observations.
    pub fn split_agent_obs(
        stacked: &Tensor,
        agent_order: &[AgentId],
    ) -> Result<HashMap<AgentId, Tensor>> {
        let num_agents = agent_order.len();
        let mut obs_map = HashMap::with_capacity(num_agents);

        for (i, agent_id) in agent_order.iter().enumerate() {
            let obs = stacked.get(i)?;
            obs_map.insert(*agent_id, obs);
        }

        Ok(obs_map)
    }

    /// Compute advantage estimates for each agent.
    pub fn compute_multi_agent_advantages(
        rewards: &[MultiAgentReward],
        values: &HashMap<AgentId, Vec<f32>>,
        dones: &[MultiAgentDone],
        gamma: f32,
        gae_lambda: f32,
    ) -> HashMap<AgentId, Vec<f32>> {
        let mut advantages: HashMap<AgentId, Vec<f32>> = HashMap::new();

        for agent_id in values.keys() {
            let agent_values = &values[agent_id];
            let n_steps = rewards.len();
            let mut agent_advantages = vec![0.0f32; n_steps];
            let mut last_gae = 0.0f32;

            for t in (0..n_steps).rev() {
                let reward = rewards[t].get(agent_id).unwrap_or(0.0);
                let done = dones[t].is_done(agent_id);
                let not_done = if done { 0.0 } else { 1.0 };

                let next_value = if t + 1 < n_steps {
                    agent_values[t + 1]
                } else {
                    0.0
                };

                let delta = reward + gamma * next_value * not_done - agent_values[t];
                last_gae = delta + gamma * gae_lambda * not_done * last_gae;
                agent_advantages[t] = last_gae;
            }

            advantages.insert(*agent_id, agent_advantages);
        }

        advantages
    }

    /// Compute returns for each agent.
    pub fn compute_multi_agent_returns(
        advantages: &HashMap<AgentId, Vec<f32>>,
        values: &HashMap<AgentId, Vec<f32>>,
    ) -> HashMap<AgentId, Vec<f32>> {
        let mut returns = HashMap::new();

        for (agent_id, agent_advantages) in advantages {
            let agent_values = &values[agent_id];
            let agent_returns: Vec<f32> = agent_advantages
                .iter()
                .zip(agent_values.iter())
                .map(|(a, v)| a + v)
                .collect();
            returns.insert(*agent_id, agent_returns);
        }

        returns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_id() {
        let agent = AgentId::new(5);
        assert_eq!(agent.value(), 5);
        assert_eq!(format!("{}", agent), "Agent_5");

        let agent2: AgentId = 10.into();
        assert_eq!(agent2.value(), 10);
    }

    #[test]
    fn test_multi_agent_reward() {
        let mut rewards = HashMap::new();
        rewards.insert(AgentId::new(0), 1.0);
        rewards.insert(AgentId::new(1), 2.0);
        rewards.insert(AgentId::new(2), 3.0);

        let mar = MultiAgentReward::new(rewards);
        assert_eq!(mar.get(&AgentId::new(0)), Some(1.0));
        assert_eq!(mar.get(&AgentId::new(1)), Some(2.0));
        assert_eq!(mar.total(), 6.0);
        assert_eq!(mar.mean(), 2.0);
    }

    #[test]
    fn test_uniform_reward() {
        let agents = vec![AgentId::new(0), AgentId::new(1)];
        let mar = MultiAgentReward::uniform(&agents, 5.0);

        assert_eq!(mar.get(&AgentId::new(0)), Some(5.0));
        assert_eq!(mar.get(&AgentId::new(1)), Some(5.0));
        assert_eq!(mar.total(), 10.0);
    }

    #[test]
    fn test_multi_agent_done() {
        let mut terminated = HashMap::new();
        terminated.insert(AgentId::new(0), false);
        terminated.insert(AgentId::new(1), true);

        let mut truncated = HashMap::new();
        truncated.insert(AgentId::new(0), false);
        truncated.insert(AgentId::new(1), false);

        let dones = MultiAgentDone::new(terminated, truncated);

        assert!(!dones.is_done(&AgentId::new(0)));
        assert!(dones.is_done(&AgentId::new(1)));
        assert!(dones.any_done());
        assert!(!dones.all_done());
        assert_eq!(dones.active_agents(), vec![AgentId::new(0)]);
    }

    #[test]
    fn test_parameter_sharing_config() {
        let independent = ParameterSharingConfig::independent();
        assert!(!independent.share_policy);
        assert!(!independent.share_value);

        let full = ParameterSharingConfig::full_sharing();
        assert!(full.share_policy);
        assert!(full.share_value);
    }

    #[test]
    fn test_communication_config() {
        let continuous = CommunicationConfig::continuous(32);
        assert!(continuous.enabled);
        assert_eq!(continuous.message_size, 32);
        assert!(!continuous.discrete_messages);

        let discrete = CommunicationConfig::discrete(16, 100);
        assert!(discrete.enabled);
        assert_eq!(discrete.message_size, 16);
        assert!(discrete.discrete_messages);
        assert_eq!(discrete.vocab_size, Some(100));
    }
}
