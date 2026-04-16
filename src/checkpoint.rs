//! Checkpointing system for Octane training.
//!
//! This module provides a robust checkpointing system for saving and resuming
//! training runs. Features include:
//!
//! - Atomic saves (write to temp, then rename)
//! - Automatic checkpoint rotation (keep last N)
//! - Best model tracking
//! - Optimizer state preservation
//! - Training metrics history
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::checkpoint::{CheckpointManager, Checkpoint};
//!
//! let manager = CheckpointManager::new("./checkpoints")
//!     .save_interval(1000)
//!     .keep_last_n(5);
//!
//! // Save checkpoint
//! manager.save(&checkpoint)?;
//!
//! // Resume from latest
//! let checkpoint = manager.load_latest()?;
//! ```

use crate::algorithms::TrainMetrics;
use crate::core::{OctaneError, Result};
use candle_core::{Device as CandleDevice, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::{debug, info};

/// A training checkpoint containing all state needed to resume.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    /// Model weights keyed by parameter name.
    #[serde(skip)]
    pub model_weights: HashMap<String, Vec<f32>>,

    /// Shape information for model weights.
    pub weight_shapes: HashMap<String, Vec<usize>>,

    /// Optimizer state (momentum, adaptive learning rates, etc.).
    pub optimizer_state: OptimizerState,

    /// Total timesteps trained.
    pub timesteps: usize,

    /// Total episodes completed.
    pub episodes: usize,

    /// Current epoch/iteration.
    pub iteration: usize,

    /// Best reward achieved.
    pub best_reward: f32,

    /// Training metrics history.
    pub metrics_history: Vec<TrainMetrics>,

    /// Configuration used for training.
    pub config_json: String,

    /// Timestamp when checkpoint was created.
    pub timestamp: String,

    /// Random state for reproducibility.
    pub rng_state: Option<Vec<u8>>,

    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Optimizer state for checkpointing.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OptimizerState {
    /// First moment estimates (Adam).
    pub momentum: HashMap<String, Vec<f32>>,

    /// Second moment estimates (Adam).
    pub velocity: HashMap<String, Vec<f32>>,

    /// Current learning rate.
    pub learning_rate: f32,

    /// Step count for bias correction.
    pub step: usize,

    /// Beta1 parameter (Adam).
    pub beta1: f32,

    /// Beta2 parameter (Adam).
    pub beta2: f32,

    /// Epsilon for numerical stability.
    pub epsilon: f32,

    /// Weight decay coefficient.
    pub weight_decay: f32,
}

impl Checkpoint {
    /// Create a new checkpoint.
    pub fn new() -> Self {
        Self {
            model_weights: HashMap::new(),
            weight_shapes: HashMap::new(),
            optimizer_state: OptimizerState::default(),
            timesteps: 0,
            episodes: 0,
            iteration: 0,
            best_reward: f32::NEG_INFINITY,
            metrics_history: Vec::new(),
            config_json: String::new(),
            timestamp: chrono_timestamp(),
            rng_state: None,
            metadata: HashMap::new(),
        }
    }

    /// Set model weights from tensors.
    pub fn set_weights(&mut self, weights: &HashMap<String, Tensor>) -> Result<()> {
        for (name, tensor) in weights {
            let shape = tensor.dims().to_vec();
            let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;

            self.weight_shapes.insert(name.clone(), shape);
            self.model_weights.insert(name.clone(), data);
        }
        Ok(())
    }

    /// Get model weights as tensors.
    pub fn get_weights(&self, device: &CandleDevice) -> Result<HashMap<String, Tensor>> {
        let mut tensors = HashMap::new();

        for (name, data) in &self.model_weights {
            let shape = self
                .weight_shapes
                .get(name)
                .ok_or_else(|| OctaneError::Serialization(format!("Missing shape for {}", name)))?;

            let tensor = Tensor::from_slice(data, shape.as_slice(), device)?;
            tensors.insert(name.clone(), tensor);
        }

        Ok(tensors)
    }

    /// Set training configuration.
    pub fn set_config<T: Serialize>(&mut self, config: &T) -> Result<()> {
        self.config_json = serde_json::to_string_pretty(config)?;
        Ok(())
    }

    /// Get training configuration.
    pub fn get_config<T: for<'de> Deserialize<'de>>(&self) -> Result<T> {
        Ok(serde_json::from_str(&self.config_json)?)
    }

    /// Add metadata entry.
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    /// Update timestamp to current time.
    pub fn update_timestamp(&mut self) {
        self.timestamp = chrono_timestamp();
    }

    /// Get the latest metrics.
    pub fn latest_metrics(&self) -> Option<&TrainMetrics> {
        self.metrics_history.last()
    }

    /// Get mean reward from recent history.
    pub fn recent_mean_reward(&self, window: usize) -> f32 {
        if self.metrics_history.is_empty() {
            return 0.0;
        }

        let start = self.metrics_history.len().saturating_sub(window);
        let recent = &self.metrics_history[start..];
        recent.iter().map(|m| m.mean_reward).sum::<f32>() / recent.len() as f32
    }
}

impl Default for Checkpoint {
    fn default() -> Self {
        Self::new()
    }
}

/// Checkpoint manager for saving and loading checkpoints.
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    /// Directory to store checkpoints.
    checkpoint_dir: PathBuf,

    /// Save checkpoint every N timesteps.
    save_interval: usize,

    /// Keep only the last N checkpoints.
    keep_last_n: usize,

    /// Track best model separately.
    track_best: bool,

    /// Metric to use for best model (higher is better).
    best_metric: BestMetric,

    /// Current best metric value.
    best_value: f32,

    /// Last save timestep.
    last_save_timesteps: usize,

    /// Checkpoint filename prefix.
    prefix: String,
}

/// Metric to track for best model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BestMetric {
    /// Mean episode reward.
    #[default]
    MeanReward,
    /// Minimum loss.
    MinLoss,
    /// Custom metric from metadata.
    Custom,
}

impl CheckpointManager {
    /// Create a new checkpoint manager.
    pub fn new(checkpoint_dir: impl AsRef<Path>) -> Self {
        Self {
            checkpoint_dir: checkpoint_dir.as_ref().to_path_buf(),
            save_interval: 10000,
            keep_last_n: 5,
            track_best: true,
            best_metric: BestMetric::default(),
            best_value: f32::NEG_INFINITY,
            last_save_timesteps: 0,
            prefix: "checkpoint".to_string(),
        }
    }

    /// Builder-style setter for save interval.
    pub fn save_interval(mut self, interval: usize) -> Self {
        self.save_interval = interval;
        self
    }

    /// Builder-style setter for keep_last_n.
    pub fn keep_last_n(mut self, n: usize) -> Self {
        self.keep_last_n = n;
        self
    }

    /// Builder-style setter for track_best.
    pub fn track_best(mut self, track: bool) -> Self {
        self.track_best = track;
        self
    }

    /// Builder-style setter for best_metric.
    pub fn best_metric(mut self, metric: BestMetric) -> Self {
        self.best_metric = metric;
        self
    }

    /// Builder-style setter for prefix.
    pub fn prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Ensure checkpoint directory exists.
    pub fn init(&self) -> Result<()> {
        fs::create_dir_all(&self.checkpoint_dir)?;
        Ok(())
    }

    /// Check if it's time to save a checkpoint.
    pub fn should_save(&self, timesteps: usize) -> bool {
        timesteps >= self.last_save_timesteps + self.save_interval
    }

    /// Save a checkpoint atomically.
    pub fn save(&mut self, checkpoint: &Checkpoint) -> Result<PathBuf> {
        self.init()?;

        let filename = format!("{}_{:010}.safetensors", self.prefix, checkpoint.timesteps);
        let final_path = self.checkpoint_dir.join(&filename);
        let temp_path = self.checkpoint_dir.join(format!(".{}.tmp", filename));

        // Save weights using safetensors
        self.save_weights(&temp_path, checkpoint)?;

        // Save metadata
        let meta_path = final_path.with_extension("json");
        let temp_meta_path = temp_path.with_extension("json");
        self.save_metadata(&temp_meta_path, checkpoint)?;

        // Atomic rename
        fs::rename(&temp_path, &final_path)?;
        fs::rename(&temp_meta_path, &meta_path)?;

        self.last_save_timesteps = checkpoint.timesteps;
        info!(
            "Saved checkpoint to {:?} (timesteps: {})",
            final_path, checkpoint.timesteps
        );

        // Track best model
        if self.track_best {
            self.update_best(checkpoint)?;
        }

        // Cleanup old checkpoints
        self.cleanup_old_checkpoints()?;

        Ok(final_path)
    }

    /// Save weights using safetensors format.
    fn save_weights(&self, path: &Path, checkpoint: &Checkpoint) -> Result<()> {
        let device = CandleDevice::Cpu;
        let mut tensors: HashMap<String, Tensor> = HashMap::new();

        for (name, data) in &checkpoint.model_weights {
            let shape = checkpoint
                .weight_shapes
                .get(name)
                .ok_or_else(|| OctaneError::Serialization(format!("Missing shape for {}", name)))?;
            let tensor = Tensor::from_slice(data, shape.as_slice(), &device)?;
            tensors.insert(name.clone(), tensor);
        }

        candle_core::safetensors::save(&tensors, path)?;
        Ok(())
    }

    /// Save metadata as JSON.
    fn save_metadata(&self, path: &Path, checkpoint: &Checkpoint) -> Result<()> {
        // Create a serializable version without the weights (they're in safetensors)
        let meta = CheckpointMetadata {
            weight_shapes: checkpoint.weight_shapes.clone(),
            optimizer_state: checkpoint.optimizer_state.clone(),
            timesteps: checkpoint.timesteps,
            episodes: checkpoint.episodes,
            iteration: checkpoint.iteration,
            best_reward: if checkpoint.best_reward.is_finite() {
                Some(checkpoint.best_reward)
            } else {
                None
            },
            metrics_history: checkpoint.metrics_history.clone(),
            config_json: checkpoint.config_json.clone(),
            timestamp: checkpoint.timestamp.clone(),
            metadata: checkpoint.metadata.clone(),
        };

        let json = serde_json::to_string_pretty(&meta)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Update best model if this checkpoint is better.
    fn update_best(&mut self, checkpoint: &Checkpoint) -> Result<()> {
        let current_value = match self.best_metric {
            BestMetric::MeanReward => checkpoint.recent_mean_reward(10),
            BestMetric::MinLoss => {
                // For loss, lower is better, so negate
                -checkpoint
                    .latest_metrics()
                    .map(|m| m.policy_loss)
                    .unwrap_or(f32::INFINITY)
            }
            BestMetric::Custom => checkpoint
                .metadata
                .get("custom_metric")
                .and_then(|s| s.parse().ok())
                .unwrap_or(f32::NEG_INFINITY),
        };

        if current_value > self.best_value {
            self.best_value = current_value;

            // Copy to best model location
            let best_path = self
                .checkpoint_dir
                .join(format!("{}_best.safetensors", self.prefix));
            let best_meta = self
                .checkpoint_dir
                .join(format!("{}_best.json", self.prefix));

            self.save_weights(&best_path, checkpoint)?;
            self.save_metadata(&best_meta, checkpoint)?;

            info!(
                "New best model saved (metric: {:.4}, timesteps: {})",
                current_value, checkpoint.timesteps
            );
        }

        Ok(())
    }

    /// Remove old checkpoints, keeping only the most recent.
    fn cleanup_old_checkpoints(&self) -> Result<()> {
        if self.keep_last_n == 0 {
            return Ok(());
        }

        let mut checkpoints: Vec<(PathBuf, u64)> = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            // Skip best model and temp files
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.contains("_best") || name.starts_with('.') {
                    continue;
                }

                if name.starts_with(&self.prefix) && name.ends_with(".safetensors") {
                    // Extract timestep from filename
                    if let Some(ts) = self.extract_timestep(name) {
                        checkpoints.push((path, ts));
                    }
                }
            }
        }

        // Sort by timestep (descending)
        checkpoints.sort_by(|a, b| b.1.cmp(&a.1));

        // Remove old checkpoints
        for (path, _) in checkpoints.iter().skip(self.keep_last_n) {
            debug!("Removing old checkpoint: {:?}", path);
            fs::remove_file(path)?;

            // Also remove metadata file
            let meta_path = path.with_extension("json");
            if meta_path.exists() {
                fs::remove_file(meta_path)?;
            }
        }

        Ok(())
    }

    /// Extract timestep from checkpoint filename.
    fn extract_timestep(&self, filename: &str) -> Option<u64> {
        let prefix_len = self.prefix.len() + 1; // +1 for underscore
        let suffix_len = ".safetensors".len();

        if filename.len() > prefix_len + suffix_len {
            let ts_str = &filename[prefix_len..filename.len() - suffix_len];
            ts_str.parse().ok()
        } else {
            None
        }
    }

    /// Load the latest checkpoint.
    pub fn load_latest(&self) -> Result<Checkpoint> {
        let mut latest: Option<(PathBuf, u64)> = None;

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(&self.prefix)
                    && name.ends_with(".safetensors")
                    && !name.contains("_best")
                {
                    if let Some(ts) = self.extract_timestep(name) {
                        if latest.is_none() || ts > latest.as_ref().unwrap().1 {
                            latest = Some((path, ts));
                        }
                    }
                }
            }
        }

        match latest {
            Some((path, _)) => self.load(&path),
            None => Err(OctaneError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No checkpoint found",
            ))),
        }
    }

    /// Load the best checkpoint.
    pub fn load_best(&self) -> Result<Checkpoint> {
        let best_path = self
            .checkpoint_dir
            .join(format!("{}_best.safetensors", self.prefix));
        if best_path.exists() {
            self.load(&best_path)
        } else {
            Err(OctaneError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "No best checkpoint found",
            )))
        }
    }

    /// Load a specific checkpoint.
    pub fn load(&self, path: &Path) -> Result<Checkpoint> {
        let device = CandleDevice::Cpu;

        // Load weights
        let tensors = candle_core::safetensors::load(path, &device)?;

        // Load metadata
        let meta_path = path.with_extension("json");
        let meta_json = fs::read_to_string(&meta_path)?;
        let meta: CheckpointMetadata = serde_json::from_str(&meta_json)?;

        // Convert tensors to Vec<f32>
        let mut model_weights = HashMap::new();
        for (name, tensor) in tensors {
            let data: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
            model_weights.insert(name, data);
        }

        let checkpoint = Checkpoint {
            model_weights,
            weight_shapes: meta.weight_shapes,
            optimizer_state: meta.optimizer_state,
            timesteps: meta.timesteps,
            episodes: meta.episodes,
            iteration: meta.iteration,
            best_reward: meta.best_reward.unwrap_or(f32::NEG_INFINITY),
            metrics_history: meta.metrics_history,
            config_json: meta.config_json,
            timestamp: meta.timestamp,
            rng_state: None,
            metadata: meta.metadata,
        };

        info!(
            "Loaded checkpoint from {:?} (timesteps: {})",
            path, checkpoint.timesteps
        );
        Ok(checkpoint)
    }

    /// Load checkpoint at specific timestep.
    pub fn load_at_timestep(&self, timesteps: usize) -> Result<Checkpoint> {
        let filename = format!("{}_{:010}.safetensors", self.prefix, timesteps);
        let path = self.checkpoint_dir.join(filename);

        if path.exists() {
            self.load(&path)
        } else {
            Err(OctaneError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Checkpoint at timestep {} not found", timesteps),
            )))
        }
    }

    /// List all available checkpoints.
    pub fn list_checkpoints(&self) -> Result<Vec<CheckpointInfo>> {
        let mut checkpoints = Vec::new();

        for entry in fs::read_dir(&self.checkpoint_dir)? {
            let entry = entry?;
            let path = entry.path();

            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if name.starts_with(&self.prefix)
                    && name.ends_with(".safetensors")
                    && !name.contains("_best")
                    && !name.starts_with('.')
                {
                    if let Some(ts) = self.extract_timestep(name) {
                        let meta_path = path.with_extension("json");
                        let timestamp = if meta_path.exists() {
                            fs::read_to_string(&meta_path)
                                .ok()
                                .and_then(|s| serde_json::from_str::<CheckpointMetadata>(&s).ok())
                                .map(|m| m.timestamp)
                        } else {
                            None
                        };

                        checkpoints.push(CheckpointInfo {
                            path: path.clone(),
                            timesteps: ts as usize,
                            timestamp,
                            is_best: false,
                        });
                    }
                }
            }
        }

        // Check for best model
        let best_path = self
            .checkpoint_dir
            .join(format!("{}_best.safetensors", self.prefix));
        if best_path.exists() {
            checkpoints.push(CheckpointInfo {
                path: best_path,
                timesteps: 0, // Unknown
                timestamp: None,
                is_best: true,
            });
        }

        // Sort by timesteps
        checkpoints.sort_by_key(|c| std::cmp::Reverse(c.timesteps));

        Ok(checkpoints)
    }

    /// Get checkpoint directory.
    pub fn checkpoint_dir(&self) -> &Path {
        &self.checkpoint_dir
    }

    /// Get the best metric value so far.
    pub fn best_value(&self) -> f32 {
        self.best_value
    }
}

/// Metadata stored alongside checkpoint weights.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointMetadata {
    weight_shapes: HashMap<String, Vec<usize>>,
    optimizer_state: OptimizerState,
    timesteps: usize,
    episodes: usize,
    iteration: usize,
    #[serde(default)]
    best_reward: Option<f32>,
    metrics_history: Vec<TrainMetrics>,
    config_json: String,
    timestamp: String,
    metadata: HashMap<String, String>,
}

/// Information about an available checkpoint.
#[derive(Debug, Clone)]
pub struct CheckpointInfo {
    /// Path to checkpoint file.
    pub path: PathBuf,
    /// Timesteps at which checkpoint was saved.
    pub timesteps: usize,
    /// Timestamp when saved.
    pub timestamp: Option<String>,
    /// Whether this is the best model.
    pub is_best: bool,
}

/// Get current timestamp as ISO 8601 string.
fn chrono_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = duration.as_secs();
    let nanos = duration.subsec_nanos();

    // Simple ISO 8601 format
    format!("{}.{:09}", secs, nanos)
}

/// Resume training from a checkpoint.
pub struct TrainingResumer {
    /// Checkpoint manager.
    manager: CheckpointManager,
    /// Loaded checkpoint.
    checkpoint: Option<Checkpoint>,
}

impl TrainingResumer {
    /// Create a new training resumer.
    pub fn new(checkpoint_dir: impl AsRef<Path>) -> Self {
        Self {
            manager: CheckpointManager::new(checkpoint_dir),
            checkpoint: None,
        }
    }

    /// Attempt to resume from latest checkpoint.
    pub fn try_resume(&mut self) -> Result<bool> {
        match self.manager.load_latest() {
            Ok(checkpoint) => {
                info!(
                    "Resuming training from checkpoint (timesteps: {})",
                    checkpoint.timesteps
                );
                self.checkpoint = Some(checkpoint);
                Ok(true)
            }
            Err(OctaneError::Io(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                info!("No checkpoint found, starting fresh training");
                Ok(false)
            }
            Err(e) => Err(e),
        }
    }

    /// Get the loaded checkpoint.
    pub fn checkpoint(&self) -> Option<&Checkpoint> {
        self.checkpoint.as_ref()
    }

    /// Take the checkpoint (consume resumer).
    pub fn take_checkpoint(self) -> Option<Checkpoint> {
        self.checkpoint
    }

    /// Get starting timesteps (0 if no checkpoint).
    pub fn starting_timesteps(&self) -> usize {
        self.checkpoint.as_ref().map(|c| c.timesteps).unwrap_or(0)
    }

    /// Get checkpoint manager.
    pub fn manager(&self) -> &CheckpointManager {
        &self.manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_checkpoint_creation() {
        let checkpoint = Checkpoint::new();
        assert_eq!(checkpoint.timesteps, 0);
        assert!(checkpoint.model_weights.is_empty());
    }

    #[test]
    fn test_checkpoint_weights() {
        let mut checkpoint = Checkpoint::new();
        let device = CandleDevice::Cpu;

        let mut weights = HashMap::new();
        let tensor = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).unwrap();
        weights.insert("layer1".to_string(), tensor);

        checkpoint.set_weights(&weights).unwrap();

        assert!(checkpoint.model_weights.contains_key("layer1"));
        assert_eq!(checkpoint.weight_shapes.get("layer1").unwrap(), &vec![2, 2]);

        let loaded = checkpoint.get_weights(&device).unwrap();
        let loaded_tensor = loaded.get("layer1").unwrap();
        assert_eq!(loaded_tensor.dims(), &[2, 2]);
    }

    #[test]
    fn test_checkpoint_manager_save_load() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path())
            .save_interval(100)
            .keep_last_n(3);

        let mut checkpoint = Checkpoint::new();
        checkpoint.timesteps = 1000;

        let device = CandleDevice::Cpu;
        let mut weights = HashMap::new();
        weights.insert(
            "w".to_string(),
            Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap(),
        );
        checkpoint.set_weights(&weights).unwrap();

        // Save
        let path = manager.save(&checkpoint).unwrap();
        assert!(path.exists());

        // Load
        let loaded = manager.load(&path).unwrap();
        assert_eq!(loaded.timesteps, 1000);

        let loaded_weights = loaded.get_weights(&device).unwrap();
        let w: Vec<f32> = loaded_weights
            .get("w")
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_eq!(w, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_checkpoint_manager_latest() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path());

        let weights: HashMap<String, Tensor> = HashMap::new();

        // Save multiple checkpoints
        for ts in [100, 200, 300] {
            let mut checkpoint = Checkpoint::new();
            checkpoint.timesteps = ts;
            checkpoint.set_weights(&weights).unwrap();
            manager.save(&checkpoint).unwrap();
        }

        // Load latest
        let latest = manager.load_latest().unwrap();
        assert_eq!(latest.timesteps, 300);
    }

    #[test]
    fn test_checkpoint_manager_cleanup() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path()).keep_last_n(2);

        let weights: HashMap<String, Tensor> = HashMap::new();

        // Save 5 checkpoints
        for ts in [100, 200, 300, 400, 500] {
            let mut checkpoint = Checkpoint::new();
            checkpoint.timesteps = ts;
            checkpoint.set_weights(&weights).unwrap();
            manager.save(&checkpoint).unwrap();
        }

        // Should only have 2 checkpoints (plus best)
        let checkpoints = manager.list_checkpoints().unwrap();
        let regular: Vec<_> = checkpoints.iter().filter(|c| !c.is_best).collect();
        assert_eq!(regular.len(), 2);

        // Should be the most recent
        assert!(regular.iter().any(|c| c.timesteps == 500));
        assert!(regular.iter().any(|c| c.timesteps == 400));
    }

    #[test]
    fn test_optimizer_state() {
        let mut state = OptimizerState {
            learning_rate: 0.001,
            step: 100,
            ..Default::default()
        };
        state.momentum.insert("w".to_string(), vec![0.1, 0.2]);

        let json = serde_json::to_string(&state).unwrap();
        let loaded: OptimizerState = serde_json::from_str(&json).unwrap();

        assert_eq!(loaded.learning_rate, 0.001);
        assert_eq!(loaded.step, 100);
    }

    #[test]
    fn test_recent_mean_reward() {
        let mut checkpoint = Checkpoint::new();

        for i in 0..10 {
            checkpoint.metrics_history.push(TrainMetrics {
                mean_reward: i as f32,
                ..Default::default()
            });
        }

        let mean = checkpoint.recent_mean_reward(5);
        // Last 5 rewards: 5, 6, 7, 8, 9 -> mean = 7
        assert!((mean - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_training_resumer() {
        let temp_dir = TempDir::new().unwrap();
        let mut resumer = TrainingResumer::new(temp_dir.path());

        // No checkpoint exists
        assert!(!resumer.try_resume().unwrap());
        assert_eq!(resumer.starting_timesteps(), 0);

        // Save a checkpoint
        let mut manager = CheckpointManager::new(temp_dir.path());
        let mut checkpoint = Checkpoint::new();
        checkpoint.timesteps = 5000;
        let weights: HashMap<String, Tensor> = HashMap::new();
        checkpoint.set_weights(&weights).unwrap();
        manager.save(&checkpoint).unwrap();

        // Now resume should work
        let mut resumer = TrainingResumer::new(temp_dir.path());
        assert!(resumer.try_resume().unwrap());
        assert_eq!(resumer.starting_timesteps(), 5000);
    }

    #[test]
    fn test_checkpoint_best_reward_non_finite_roundtrip() {
        let temp_dir = TempDir::new().unwrap();
        let mut manager = CheckpointManager::new(temp_dir.path());

        let mut checkpoint = Checkpoint::new();
        checkpoint.timesteps = 7;
        checkpoint.best_reward = f32::NEG_INFINITY;

        let path = manager.save(&checkpoint).unwrap();
        let loaded = manager.load(&path).unwrap();

        assert_eq!(loaded.best_reward, f32::NEG_INFINITY);
    }
}
