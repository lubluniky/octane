//! Weights & Biases (W&B) integration for experiment tracking.
//!
//! This module provides integration with Weights & Biases for logging
//! training metrics, hyperparameters, and artifacts.
//!
//! # Requirements
//!
//! - Enable the `wandb` feature in Cargo.toml
//! - Install the wandb Python package: `pip install wandb`
//! - Authenticate: `wandb login`
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::logging::{WandbLogger, WandbConfig, MetricLogger};
//!
//! let config = WandbConfig::new("my-project")
//!     .entity("my-team")
//!     .run_name("ppo-trading-v1")
//!     .tags(&["ppo", "trading"])
//!     .config("learning_rate", 0.0003)
//!     .config("n_steps", 2048);
//!
//! let mut logger = WandbLogger::init(config)?;
//! logger.log_scalar("train/loss", 0.5, 1000)?;
//! logger.log_scalar("train/reward", 100.0, 1000)?;
//! logger.finish()?;
//! ```

use crate::core::{OctaneError, Result};
use crate::logging::metrics::{HistogramData, ImageData, MetricLogger, VideoData};
use std::collections::HashMap;

#[cfg(feature = "wandb")]
use std::path::Path;

#[cfg(feature = "wandb")]
use pyo3::prelude::*;
#[cfg(feature = "wandb")]
use pyo3::types::{PyDict, PyList};
// pyo3 0.23+ removed ToPyObject::to_object; into_py_any() is the replacement
// that yields an owned Py<PyAny> directly.
#[cfg(feature = "wandb")]
use pyo3::IntoPyObjectExt;

/// Configuration for Weights & Biases runs.
#[derive(Debug, Clone)]
pub struct WandbConfig {
    /// W&B project name.
    pub project: String,
    /// W&B entity (team or username). None uses default.
    pub entity: Option<String>,
    /// Run name. None generates a random name.
    pub run_name: Option<String>,
    /// Run ID for resuming. None creates new run.
    pub run_id: Option<String>,
    /// Tags for the run.
    pub tags: Vec<String>,
    /// Notes/description for the run.
    pub notes: Option<String>,
    /// Hyperparameters and config.
    pub config: HashMap<String, ConfigValue>,
    /// Run group for grouping related runs.
    pub group: Option<String>,
    /// Job type (e.g., "train", "eval").
    pub job_type: Option<String>,
    /// Mode: "online", "offline", or "disabled".
    pub mode: String,
    /// Directory for local logging.
    pub dir: Option<String>,
    /// Whether to resume a previous run.
    pub resume: ResumeMode,
    /// Whether to reinitialize if already initialized.
    pub reinit: bool,
}

/// Resume mode for W&B runs.
#[derive(Debug, Clone, Default)]
pub enum ResumeMode {
    /// Never resume, always start fresh.
    #[default]
    Never,
    /// Resume if run_id exists, otherwise start new.
    Allow,
    /// Must resume existing run_id.
    Must,
    /// Auto-detect based on run_id.
    Auto,
}

impl ResumeMode {
    fn as_str(&self) -> &str {
        match self {
            ResumeMode::Never => "never",
            ResumeMode::Allow => "allow",
            ResumeMode::Must => "must",
            ResumeMode::Auto => "auto",
        }
    }
}

/// Configuration value types.
#[derive(Debug, Clone)]
pub enum ConfigValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Float value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// List of values.
    List(Vec<ConfigValue>),
    /// Nested dictionary.
    Dict(HashMap<String, ConfigValue>),
}

impl From<&str> for ConfigValue {
    fn from(s: &str) -> Self {
        ConfigValue::String(s.to_string())
    }
}

impl From<String> for ConfigValue {
    fn from(s: String) -> Self {
        ConfigValue::String(s)
    }
}

impl From<i64> for ConfigValue {
    fn from(v: i64) -> Self {
        ConfigValue::Int(v)
    }
}

impl From<i32> for ConfigValue {
    fn from(v: i32) -> Self {
        ConfigValue::Int(v as i64)
    }
}

impl From<usize> for ConfigValue {
    fn from(v: usize) -> Self {
        ConfigValue::Int(v as i64)
    }
}

impl From<f64> for ConfigValue {
    fn from(v: f64) -> Self {
        ConfigValue::Float(v)
    }
}

impl From<f32> for ConfigValue {
    fn from(v: f32) -> Self {
        ConfigValue::Float(v as f64)
    }
}

impl From<bool> for ConfigValue {
    fn from(v: bool) -> Self {
        ConfigValue::Bool(v)
    }
}

impl WandbConfig {
    /// Create a new W&B configuration with project name.
    pub fn new(project: impl Into<String>) -> Self {
        Self {
            project: project.into(),
            entity: None,
            run_name: None,
            run_id: None,
            tags: Vec::new(),
            notes: None,
            config: HashMap::new(),
            group: None,
            job_type: None,
            mode: "online".to_string(),
            dir: None,
            resume: ResumeMode::Never,
            reinit: false,
        }
    }

    /// Set the entity (team or username).
    pub fn entity(mut self, entity: impl Into<String>) -> Self {
        self.entity = Some(entity.into());
        self
    }

    /// Set the run name.
    pub fn run_name(mut self, name: impl Into<String>) -> Self {
        self.run_name = Some(name.into());
        self
    }

    /// Set the run ID (for resuming).
    pub fn run_id(mut self, id: impl Into<String>) -> Self {
        self.run_id = Some(id.into());
        self
    }

    /// Set tags for the run.
    pub fn tags(mut self, tags: &[&str]) -> Self {
        self.tags = tags.iter().map(|s| s.to_string()).collect();
        self
    }

    /// Add a single tag.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tags.push(tag.into());
        self
    }

    /// Set notes/description.
    pub fn notes(mut self, notes: impl Into<String>) -> Self {
        self.notes = Some(notes.into());
        self
    }

    /// Add a configuration value.
    pub fn config(mut self, key: impl Into<String>, value: impl Into<ConfigValue>) -> Self {
        self.config.insert(key.into(), value.into());
        self
    }

    /// Add multiple configuration values.
    pub fn configs(mut self, configs: HashMap<String, ConfigValue>) -> Self {
        self.config.extend(configs);
        self
    }

    /// Set run group.
    pub fn group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    /// Set job type.
    pub fn job_type(mut self, job_type: impl Into<String>) -> Self {
        self.job_type = Some(job_type.into());
        self
    }

    /// Set mode ("online", "offline", or "disabled").
    pub fn mode(mut self, mode: impl Into<String>) -> Self {
        self.mode = mode.into();
        self
    }

    /// Run in offline mode.
    pub fn offline(mut self) -> Self {
        self.mode = "offline".to_string();
        self
    }

    /// Set local logging directory.
    pub fn dir(mut self, dir: impl Into<String>) -> Self {
        self.dir = Some(dir.into());
        self
    }

    /// Set resume mode.
    pub fn resume(mut self, mode: ResumeMode) -> Self {
        self.resume = mode;
        self
    }

    /// Allow reinitialization.
    pub fn reinit(mut self) -> Self {
        self.reinit = true;
        self
    }
}

/// Weights & Biases logger.
///
/// Provides logging capabilities to W&B for experiment tracking.
#[cfg(feature = "wandb")]
pub struct WandbLogger {
    /// Run object from wandb.init().
    run: Py<PyAny>,
    /// Configuration.
    config: WandbConfig,
    /// Whether the run is active.
    active: bool,
}

#[cfg(feature = "wandb")]
impl WandbLogger {
    /// Initialize a new W&B run.
    pub fn init(config: WandbConfig) -> Result<Self> {
        Python::attach(|py| {
            let wandb = py.import("wandb").map_err(|e| {
                OctaneError::Environment(format!(
                    "Failed to import wandb. Install with `pip install wandb`: {}",
                    e
                ))
            })?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("project", &config.project)?;

            if let Some(ref entity) = config.entity {
                kwargs.set_item("entity", entity)?;
            }
            if let Some(ref name) = config.run_name {
                kwargs.set_item("name", name)?;
            }
            if let Some(ref id) = config.run_id {
                kwargs.set_item("id", id)?;
            }
            if !config.tags.is_empty() {
                let tags = PyList::new(py, &config.tags)?;
                kwargs.set_item("tags", tags)?;
            }
            if let Some(ref notes) = config.notes {
                kwargs.set_item("notes", notes)?;
            }
            if !config.config.is_empty() {
                let config_dict = config_to_pydict(py, &config.config)?;
                kwargs.set_item("config", config_dict)?;
            }
            if let Some(ref group) = config.group {
                kwargs.set_item("group", group)?;
            }
            if let Some(ref job_type) = config.job_type {
                kwargs.set_item("job_type", job_type)?;
            }
            kwargs.set_item("mode", &config.mode)?;
            if let Some(ref dir) = config.dir {
                kwargs.set_item("dir", dir)?;
            }
            kwargs.set_item("resume", config.resume.as_str())?;
            kwargs.set_item("reinit", config.reinit)?;

            let run = wandb.call_method("init", (), Some(&kwargs))?;

            Ok(Self {
                run: run.into(),
                config,
                active: true,
            })
        })
    }

    /// Get the run URL.
    pub fn url(&self) -> Result<String> {
        Python::attach(|py| {
            let run = self.run.bind(py);
            let url = run.getattr("url")?.extract::<String>()?;
            Ok(url)
        })
    }

    /// Get the run ID.
    pub fn run_id(&self) -> Result<String> {
        Python::attach(|py| {
            let run = self.run.bind(py);
            let id = run.getattr("id")?.extract::<String>()?;
            Ok(id)
        })
    }

    /// Get the run name.
    pub fn name(&self) -> Result<String> {
        Python::attach(|py| {
            let run = self.run.bind(py);
            let name = run.getattr("name")?.extract::<String>()?;
            Ok(name)
        })
    }

    /// Update the run configuration.
    pub fn update_config(&mut self, updates: HashMap<String, ConfigValue>) -> Result<()> {
        Python::attach(|py| {
            let run = self.run.bind(py);
            let config = run.getattr("config")?;
            let update_dict = config_to_pydict(py, &updates)?;
            config.call_method("update", (update_dict,), None)?;
            Ok(())
        })
    }

    /// Log a summary value (final metrics).
    pub fn log_summary(&mut self, key: &str, value: impl Into<ConfigValue>) -> Result<()> {
        Python::attach(|py| {
            let run = self.run.bind(py);
            let summary = run.getattr("summary")?;
            let val = config_value_to_py(py, &value.into())?;
            summary.set_item(key, val)?;
            Ok(())
        })
    }

    /// Save a model artifact.
    pub fn save_model(&mut self, path: impl AsRef<Path>, name: &str) -> Result<()> {
        Python::attach(|py| {
            let wandb = py.import("wandb")?;
            let artifact = wandb.call_method("Artifact", (name, "model"), None)?;
            artifact.call_method(
                "add_file",
                (path.as_ref().to_string_lossy().to_string(),),
                None,
            )?;

            let run = self.run.bind(py);
            run.call_method("log_artifact", (artifact,), None)?;
            Ok(())
        })
    }

    /// Mark the run as finished.
    pub fn finish(&mut self) -> Result<()> {
        if !self.active {
            return Ok(());
        }

        // Annotate the closure's error type: now that both PyErr and
        // OctaneError implement `From<PyErr>`, the `?`-then-`?` chain is
        // otherwise ambiguous.
        Python::attach(|py| -> Result<()> {
            let run = self.run.bind(py);
            run.call_method("finish", (), None)?;
            Ok(())
        })?;

        self.active = false;
        Ok(())
    }

    /// Define custom metrics with specific step keys.
    pub fn define_metric(&mut self, name: &str, step_metric: Option<&str>) -> Result<()> {
        Python::attach(|py| {
            let wandb = py.import("wandb")?;
            let kwargs = PyDict::new(py);
            if let Some(step) = step_metric {
                kwargs.set_item("step_metric", step)?;
            }
            wandb.call_method("define_metric", (name,), Some(&kwargs))?;
            Ok(())
        })
    }
}

#[cfg(feature = "wandb")]
impl MetricLogger for WandbLogger {
    fn log_scalar(&mut self, tag: &str, value: f64, step: u64) -> Result<()> {
        Python::attach(|py| {
            let run = self.run.bind(py);
            let log_dict = PyDict::new(py);
            log_dict.set_item(tag, value)?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("step", step)?;

            run.call_method("log", (log_dict,), Some(&kwargs))?;
            Ok(())
        })
    }

    fn log_scalars(&mut self, scalars: &[(&str, f64)], step: u64) -> Result<()> {
        Python::attach(|py| {
            let run = self.run.bind(py);
            let log_dict = PyDict::new(py);
            for (tag, value) in scalars {
                log_dict.set_item(*tag, *value)?;
            }

            let kwargs = PyDict::new(py);
            kwargs.set_item("step", step)?;

            run.call_method("log", (log_dict,), Some(&kwargs))?;
            Ok(())
        })
    }

    fn log_histogram(&mut self, tag: &str, data: &HistogramData, step: u64) -> Result<()> {
        Python::attach(|py| {
            let wandb = py.import("wandb")?;
            let run = self.run.bind(py);

            let values = PyList::new(py, &data.values)?;
            let histogram = wandb.call_method("Histogram", (values,), None)?;

            let log_dict = PyDict::new(py);
            log_dict.set_item(tag, histogram)?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("step", step)?;

            run.call_method("log", (log_dict,), Some(&kwargs))?;
            Ok(())
        })
    }

    fn log_video(&mut self, tag: &str, video: &VideoData, step: u64) -> Result<()> {
        Python::attach(|py| {
            let wandb = py.import("wandb")?;
            let numpy = py.import("numpy")?;
            let run = self.run.bind(py);

            // Convert frames to numpy array [T, H, W, C]
            let num_frames = video.frames.len();
            let frame_size = (video.height * video.width * 4) as usize;

            let mut flat_data: Vec<u8> = Vec::with_capacity(num_frames * frame_size);
            for frame in &video.frames {
                flat_data.extend_from_slice(frame);
            }

            let shape = vec![num_frames, video.height as usize, video.width as usize, 4];
            let np_array = numpy
                .call_method("array", (flat_data,), None)?
                .call_method("reshape", (shape,), None)?;

            // wandb.Video expects [T, C, H, W] so we need to transpose
            let transposed = np_array.call_method("transpose", ((0, 3, 1, 2),), None)?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("fps", video.fps as i32)?;

            let video_obj = wandb.call_method("Video", (transposed,), Some(&kwargs))?;

            let log_dict = PyDict::new(py);
            log_dict.set_item(tag, video_obj)?;

            let step_kwargs = PyDict::new(py);
            step_kwargs.set_item("step", step)?;

            run.call_method("log", (log_dict,), Some(&step_kwargs))?;
            Ok(())
        })
    }

    fn log_image(&mut self, tag: &str, image: &ImageData, step: u64) -> Result<()> {
        Python::attach(|py| {
            let wandb = py.import("wandb")?;
            let numpy = py.import("numpy")?;
            let run = self.run.bind(py);

            let shape = vec![
                image.height as usize,
                image.width as usize,
                image.channels as usize,
            ];
            let np_array = numpy
                .call_method("array", (image.data.clone(),), None)?
                .call_method("reshape", (shape,), None)?;

            let image_obj = wandb.call_method("Image", (np_array,), None)?;

            let log_dict = PyDict::new(py);
            log_dict.set_item(tag, image_obj)?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("step", step)?;

            run.call_method("log", (log_dict,), Some(&kwargs))?;
            Ok(())
        })
    }

    fn log_text(&mut self, tag: &str, text: &str, step: u64) -> Result<()> {
        Python::attach(|py| {
            let wandb = py.import("wandb")?;
            let run = self.run.bind(py);

            // Create a simple HTML table for text
            let html = format!("<pre>{}</pre>", text);
            let html_obj = wandb.call_method("Html", (html,), None)?;

            let log_dict = PyDict::new(py);
            log_dict.set_item(tag, html_obj)?;

            let kwargs = PyDict::new(py);
            kwargs.set_item("step", step)?;

            run.call_method("log", (log_dict,), Some(&kwargs))?;
            Ok(())
        })
    }

    fn flush(&mut self) -> Result<()> {
        // W&B handles flushing automatically
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.finish()
    }
}

#[cfg(feature = "wandb")]
impl Drop for WandbLogger {
    fn drop(&mut self) {
        if self.active {
            let _ = self.finish();
        }
    }
}

#[cfg(feature = "wandb")]
fn config_to_pydict<'py>(
    py: Python<'py>,
    config: &HashMap<String, ConfigValue>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    for (key, value) in config {
        let py_value = config_value_to_py(py, value)?;
        dict.set_item(key, py_value)?;
    }
    Ok(dict)
}

#[cfg(feature = "wandb")]
fn config_value_to_py(py: Python<'_>, value: &ConfigValue) -> PyResult<Py<PyAny>> {
    match value {
        ConfigValue::String(s) => s.into_py_any(py),
        ConfigValue::Int(i) => i.into_py_any(py),
        ConfigValue::Float(f) => f.into_py_any(py),
        ConfigValue::Bool(b) => b.into_py_any(py),
        ConfigValue::List(items) => {
            let list = PyList::empty(py);
            for item in items {
                let py_item = config_value_to_py(py, item)?;
                list.append(py_item)?;
            }
            list.into_py_any(py)
        }
        ConfigValue::Dict(d) => {
            let dict = config_to_pydict(py, d)?;
            dict.into_py_any(py)
        }
    }
}

// Stub implementation when wandb feature is not enabled
#[cfg(not(feature = "wandb"))]
/// Stub W&B logger returned when the `wandb` feature is disabled.
pub struct WandbLogger {
    _private: (),
}

#[cfg(not(feature = "wandb"))]
impl WandbLogger {
    /// Initialize a new W&B run (stub - requires wandb feature).
    pub fn init(_config: WandbConfig) -> Result<Self> {
        Err(OctaneError::Environment(
            "W&B support requires the 'wandb' feature. Enable it in Cargo.toml: \
            octane-rs = { version = \"*\", features = [\"wandb\"] }"
                .to_string(),
        ))
    }
}

#[cfg(not(feature = "wandb"))]
impl MetricLogger for WandbLogger {
    fn log_scalar(&mut self, _tag: &str, _value: f64, _step: u64) -> Result<()> {
        Err(OctaneError::Environment(
            "W&B feature not enabled".to_string(),
        ))
    }

    fn log_histogram(&mut self, _tag: &str, _data: &HistogramData, _step: u64) -> Result<()> {
        Err(OctaneError::Environment(
            "W&B feature not enabled".to_string(),
        ))
    }

    fn log_video(&mut self, _tag: &str, _video: &VideoData, _step: u64) -> Result<()> {
        Err(OctaneError::Environment(
            "W&B feature not enabled".to_string(),
        ))
    }

    fn log_image(&mut self, _tag: &str, _image: &ImageData, _step: u64) -> Result<()> {
        Err(OctaneError::Environment(
            "W&B feature not enabled".to_string(),
        ))
    }

    fn log_text(&mut self, _tag: &str, _text: &str, _step: u64) -> Result<()> {
        Err(OctaneError::Environment(
            "W&B feature not enabled".to_string(),
        ))
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wandb_config_builder() {
        let config = WandbConfig::new("test-project")
            .entity("my-team")
            .run_name("test-run")
            .tags(&["ppo", "trading"])
            .config("learning_rate", 0.0003_f64)
            .config("n_steps", 2048_i32)
            .config("use_gae", true)
            .offline();

        assert_eq!(config.project, "test-project");
        assert_eq!(config.entity, Some("my-team".to_string()));
        assert_eq!(config.run_name, Some("test-run".to_string()));
        assert_eq!(config.tags, vec!["ppo", "trading"]);
        assert_eq!(config.mode, "offline");
        assert!(config.config.contains_key("learning_rate"));
        assert!(config.config.contains_key("n_steps"));
        assert!(config.config.contains_key("use_gae"));
    }

    #[test]
    fn test_config_value_conversions() {
        let _s: ConfigValue = "test".into();
        let _i: ConfigValue = 42i64.into();
        let _f: ConfigValue = std::f64::consts::PI.into();
        let _b: ConfigValue = true.into();
    }

    #[test]
    #[cfg(not(feature = "wandb"))]
    fn test_wandb_init_without_feature() {
        let config = WandbConfig::new("test");
        let result = WandbLogger::init(config);
        assert!(result.is_err());
    }
}
