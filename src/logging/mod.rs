//! Training logging system for RocketRL.
//!
//! This module provides a JSON-based logging system that allows:
//! - Background training processes to write structured logs
//! - TUI to read and display training progress in real-time
//! - Post-training analysis and visualization

use crate::algorithms::TrainMetrics;
use crate::core::Result;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// A single log entry for training progress.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingLogEntry {
    /// Unix timestamp in milliseconds.
    pub timestamp: u64,
    /// Training timestep.
    pub timestep: usize,
    /// Episode number.
    pub episode: usize,
    /// Mean reward over recent episodes.
    pub mean_reward: f32,
    /// Standard deviation of rewards.
    pub std_reward: f32,
    /// Policy loss.
    pub policy_loss: f32,
    /// Value loss.
    pub value_loss: f32,
    /// Entropy.
    pub entropy: f32,
    /// Learning rate.
    pub learning_rate: f32,
    /// Steps per second.
    pub steps_per_second: f32,
    /// Additional metrics (algorithm-specific).
    #[serde(default)]
    pub extra: std::collections::HashMap<String, f32>,
}

impl TrainingLogEntry {
    /// Create from TrainMetrics.
    pub fn from_metrics(metrics: &TrainMetrics) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let mut extra = std::collections::HashMap::new();
        extra.insert("approx_kl".to_string(), metrics.approx_kl);
        extra.insert("clip_fraction".to_string(), metrics.clip_fraction);
        extra.insert("explained_variance".to_string(), metrics.explained_variance);

        Self {
            timestamp,
            timestep: metrics.timesteps,
            episode: metrics.episodes,
            mean_reward: metrics.mean_reward,
            std_reward: metrics.std_reward,
            policy_loss: metrics.policy_loss,
            value_loss: metrics.value_loss,
            entropy: metrics.entropy,
            learning_rate: metrics.learning_rate,
            steps_per_second: 0.0, // Computed externally
            extra,
        }
    }
}

/// Training run metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingRunInfo {
    /// Unique run ID.
    pub run_id: String,
    /// Algorithm name (PPO, SAC, etc.).
    pub algorithm: String,
    /// Environment name.
    pub environment: String,
    /// Total timesteps target.
    pub total_timesteps: usize,
    /// Start timestamp.
    pub start_time: u64,
    /// End timestamp (if finished).
    pub end_time: Option<u64>,
    /// Device used (CPU, Metal, CUDA).
    pub device: String,
    /// Configuration as JSON.
    pub config: String,
}

/// Training logger that writes to a JSON-lines file.
pub struct TrainingLogger {
    /// Log file path.
    log_path: PathBuf,
    /// Writer for log entries.
    writer: BufWriter<File>,
    /// Run info.
    run_info: TrainingRunInfo,
    /// Last timestep for computing steps/second.
    last_timestep: usize,
    /// Last timestamp for computing steps/second.
    last_time: u64,
}

impl TrainingLogger {
    /// Create a new training logger.
    ///
    /// Creates a log directory structure:
    /// ```text
    /// logs/
    ///   {run_id}/
    ///     info.json      - Run metadata
    ///     metrics.jsonl  - Training metrics (JSON lines)
    /// ```
    pub fn new(
        log_dir: impl AsRef<Path>,
        algorithm: &str,
        environment: &str,
        total_timesteps: usize,
        device: &str,
        config: &str,
    ) -> Result<Self> {
        let run_id = format!(
            "{}_{}_{}",
            algorithm.to_lowercase(),
            chrono_timestamp(),
            random_suffix()
        );

        let run_dir = log_dir.as_ref().join(&run_id);
        std::fs::create_dir_all(&run_dir)?;

        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let run_info = TrainingRunInfo {
            run_id: run_id.clone(),
            algorithm: algorithm.to_string(),
            environment: environment.to_string(),
            total_timesteps,
            start_time,
            end_time: None,
            device: device.to_string(),
            config: config.to_string(),
        };

        // Write info file
        let info_path = run_dir.join("info.json");
        let info_file = File::create(&info_path)?;
        serde_json::to_writer_pretty(info_file, &run_info)?;

        // Create metrics file
        let metrics_path = run_dir.join("metrics.jsonl");
        let metrics_file = File::create(&metrics_path)?;
        let writer = BufWriter::new(metrics_file);

        Ok(Self {
            log_path: metrics_path,
            writer,
            run_info,
            last_timestep: 0,
            last_time: start_time,
        })
    }

    /// Log training metrics.
    pub fn log(&mut self, metrics: &TrainMetrics) -> Result<()> {
        let mut entry = TrainingLogEntry::from_metrics(metrics);

        // Compute steps per second
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let elapsed_ms = now.saturating_sub(self.last_time);
        if elapsed_ms > 0 {
            let steps = metrics.timesteps.saturating_sub(self.last_timestep);
            entry.steps_per_second = (steps as f32 * 1000.0) / elapsed_ms as f32;
        }

        self.last_timestep = metrics.timesteps;
        self.last_time = now;

        // Write as JSON line
        let line = serde_json::to_string(&entry)?;
        writeln!(self.writer, "{}", line)?;
        self.writer.flush()?;

        Ok(())
    }

    /// Finalize the log (called when training ends).
    pub fn finalize(&mut self) -> Result<()> {
        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        self.run_info.end_time = Some(end_time);

        // Update info file
        let info_path = self.log_path.parent().unwrap().join("info.json");
        let info_file = File::create(&info_path)?;
        serde_json::to_writer_pretty(info_file, &self.run_info)?;

        self.writer.flush()?;
        Ok(())
    }

    /// Get the run ID.
    pub fn run_id(&self) -> &str {
        &self.run_info.run_id
    }

    /// Get the log directory path.
    pub fn log_dir(&self) -> &Path {
        self.log_path.parent().unwrap()
    }
}

/// Training log reader for TUI.
pub struct TrainingLogReader {
    /// Log file path.
    log_path: PathBuf,
    /// Current read position.
    position: u64,
    /// Cached entries.
    entries: Vec<TrainingLogEntry>,
    /// Run info.
    run_info: Option<TrainingRunInfo>,
}

impl TrainingLogReader {
    /// Create a reader for a training run.
    pub fn new(run_dir: impl AsRef<Path>) -> Result<Self> {
        let run_dir = run_dir.as_ref();

        // Read run info
        let info_path = run_dir.join("info.json");
        let run_info = if info_path.exists() {
            let file = File::open(&info_path)?;
            Some(serde_json::from_reader(file)?)
        } else {
            None
        };

        let log_path = run_dir.join("metrics.jsonl");

        Ok(Self {
            log_path,
            position: 0,
            entries: Vec::new(),
            run_info,
        })
    }

    /// Read new entries since last read.
    pub fn read_new(&mut self) -> Result<&[TrainingLogEntry]> {
        if !self.log_path.exists() {
            return Ok(&[]);
        }

        let file = File::open(&self.log_path)?;
        let mut reader = BufReader::new(file);

        // Seek to last position
        reader.seek(SeekFrom::Start(self.position))?;

        let start_idx = self.entries.len();

        for line in reader.lines() {
            let line = line?;
            if line.is_empty() {
                continue;
            }
            match serde_json::from_str::<TrainingLogEntry>(&line) {
                Ok(entry) => self.entries.push(entry),
                Err(_) => continue, // Skip malformed lines
            }
        }

        // Update position
        let file = File::open(&self.log_path)?;
        self.position = file.metadata()?.len();

        Ok(&self.entries[start_idx..])
    }

    /// Get all entries.
    pub fn entries(&self) -> &[TrainingLogEntry] {
        &self.entries
    }

    /// Get the latest entry.
    pub fn latest(&self) -> Option<&TrainingLogEntry> {
        self.entries.last()
    }

    /// Get run info.
    pub fn run_info(&self) -> Option<&TrainingRunInfo> {
        self.run_info.as_ref()
    }

    /// Check if training is complete.
    pub fn is_complete(&self) -> bool {
        self.run_info
            .as_ref()
            .map(|info| info.end_time.is_some())
            .unwrap_or(false)
    }

    /// Get training progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        match (&self.run_info, self.latest()) {
            (Some(info), Some(entry)) => {
                entry.timestep as f32 / info.total_timesteps as f32
            }
            _ => 0.0,
        }
    }
}

/// List available training runs in a log directory.
pub fn list_training_runs(log_dir: impl AsRef<Path>) -> Result<Vec<TrainingRunInfo>> {
    let log_dir = log_dir.as_ref();
    let mut runs = Vec::new();

    if !log_dir.exists() {
        return Ok(runs);
    }

    for entry in std::fs::read_dir(log_dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_dir() {
            let info_path = path.join("info.json");
            if info_path.exists() {
                match File::open(&info_path) {
                    Ok(file) => {
                        if let Ok(info) = serde_json::from_reader::<_, TrainingRunInfo>(file) {
                            runs.push(info);
                        }
                    }
                    Err(_) => continue,
                }
            }
        }
    }

    // Sort by start time (newest first)
    runs.sort_by(|a, b| b.start_time.cmp(&a.start_time));

    Ok(runs)
}

/// Generate a timestamp string.
fn chrono_timestamp() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    format!("{}", now)
}

/// Generate a random suffix.
fn random_suffix() -> String {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    format!("{:04x}", rng.gen::<u16>())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn test_training_logger() -> Result<()> {
        let temp_dir = TempDir::new()?;
        let mut logger = TrainingLogger::new(
            temp_dir.path(),
            "PPO",
            "TradingEnv",
            1000000,
            "CPU",
            "{}",
        )?;

        // Log some metrics
        let metrics = TrainMetrics {
            timesteps: 1000,
            episodes: 10,
            mean_reward: 100.0,
            policy_loss: 0.5,
            value_loss: 0.3,
            entropy: 0.1,
            learning_rate: 0.0003,
            ..Default::default()
        };

        logger.log(&metrics)?;
        logger.finalize()?;

        // Read back
        let mut reader = TrainingLogReader::new(logger.log_dir())?;
        reader.read_new()?;

        assert_eq!(reader.entries().len(), 1);
        assert_eq!(reader.entries()[0].timestep, 1000);

        Ok(())
    }

    #[test]
    fn test_list_runs() -> Result<()> {
        let temp_dir = TempDir::new()?;

        // Create a run
        let mut logger = TrainingLogger::new(
            temp_dir.path(),
            "SAC",
            "TestEnv",
            100000,
            "Metal",
            "{}",
        )?;
        logger.finalize()?;

        let runs = list_training_runs(temp_dir.path())?;
        assert_eq!(runs.len(), 1);
        assert_eq!(runs[0].algorithm, "SAC");

        Ok(())
    }
}
