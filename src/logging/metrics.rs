//! Unified metrics interface for logging backends.
//!
//! This module provides a common trait for logging metrics to various backends
//! (TensorBoard, Weights & Biases, JSON files, etc.) with support for scalars,
//! histograms, and videos.
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::logging::{MetricLogger, CompositeLogger, MetricBuffer};
//!
//! // Create a composite logger with multiple backends
//! let mut logger = CompositeLogger::new();
//! logger.add_backend(Box::new(TensorBoardWriter::new("logs/tb")?));
//!
//! // Log metrics
//! logger.log_scalar("train/loss", 0.5, 1000)?;
//! logger.log_histogram("weights/layer1", &weights, 1000)?;
//!
//! // Use a buffer for batched logging
//! let mut buffer = MetricBuffer::new(100);
//! buffer.add_scalar("train/reward", 100.0, 500);
//! buffer.flush(&mut logger)?;
//! ```

use crate::core::Result;
use std::collections::HashMap;

/// A single metric value with metadata.
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Scalar value (loss, reward, etc.).
    Scalar(f64),
    /// Histogram data (weights, activations, etc.).
    Histogram(HistogramData),
    /// Video frames (environment renders, etc.).
    Video(VideoData),
    /// Text annotation.
    Text(String),
    /// Image data.
    Image(ImageData),
}

/// Histogram data for distribution logging.
#[derive(Debug, Clone)]
pub struct HistogramData {
    /// Raw values for the histogram.
    pub values: Vec<f32>,
    /// Optional pre-computed bin edges.
    pub bin_edges: Option<Vec<f32>>,
    /// Optional pre-computed bin counts.
    pub bin_counts: Option<Vec<u64>>,
}

impl HistogramData {
    /// Create histogram from raw values.
    pub fn from_values(values: Vec<f32>) -> Self {
        Self {
            values,
            bin_edges: None,
            bin_counts: None,
        }
    }

    /// Create histogram from pre-computed bins.
    pub fn from_bins(bin_edges: Vec<f32>, bin_counts: Vec<u64>) -> Self {
        Self {
            values: Vec::new(),
            bin_edges: Some(bin_edges),
            bin_counts: Some(bin_counts),
        }
    }

    /// Compute histogram statistics.
    pub fn compute_stats(&self) -> HistogramStats {
        if self.values.is_empty() {
            return HistogramStats::default();
        }

        let n = self.values.len() as f64;
        let sum: f64 = self.values.iter().map(|&v| v as f64).sum();
        let mean = sum / n;

        let sum_sq: f64 = self.values.iter().map(|&v| (v as f64).powi(2)).sum();
        let variance = (sum_sq / n) - mean.powi(2);
        let std = variance.max(0.0).sqrt();

        let min = self.values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self
            .values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        HistogramStats {
            min: min as f64,
            max: max as f64,
            mean,
            std,
            count: self.values.len(),
            sum,
            sum_squares: sum_sq,
        }
    }

    /// Compute bins using Freedman-Diaconis rule.
    pub fn compute_bins(&self, num_bins: usize) -> (Vec<f32>, Vec<u64>) {
        if self.values.is_empty() {
            return (vec![0.0; num_bins + 1], vec![0; num_bins]);
        }

        let min = self.values.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = self
            .values
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);

        let bin_width = (max - min) / num_bins as f32;
        let bin_edges: Vec<f32> = (0..=num_bins).map(|i| min + i as f32 * bin_width).collect();

        let mut bin_counts = vec![0u64; num_bins];
        for &value in &self.values {
            let bin_idx = if value >= max {
                num_bins - 1
            } else {
                ((value - min) / bin_width).floor() as usize
            };
            let bin_idx = bin_idx.min(num_bins - 1);
            bin_counts[bin_idx] += 1;
        }

        (bin_edges, bin_counts)
    }
}

/// Pre-computed histogram statistics.
#[derive(Debug, Clone, Default)]
pub struct HistogramStats {
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std: f64,
    /// Number of values.
    pub count: usize,
    /// Sum of values.
    pub sum: f64,
    /// Sum of squared values.
    pub sum_squares: f64,
}

/// Video data for episode recordings.
#[derive(Debug, Clone)]
pub struct VideoData {
    /// Frames as RGBA byte arrays [num_frames, height, width, 4].
    pub frames: Vec<Vec<u8>>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Frames per second.
    pub fps: f32,
}

impl VideoData {
    /// Create video data from frames.
    pub fn new(frames: Vec<Vec<u8>>, width: u32, height: u32, fps: f32) -> Self {
        Self {
            frames,
            width,
            height,
            fps,
        }
    }

    /// Number of frames.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// Duration in seconds.
    pub fn duration(&self) -> f32 {
        self.frames.len() as f32 / self.fps
    }
}

/// Image data for single frame logging.
#[derive(Debug, Clone)]
pub struct ImageData {
    /// Pixel data (RGB or RGBA).
    pub data: Vec<u8>,
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Number of channels (3 for RGB, 4 for RGBA).
    pub channels: u8,
}

impl ImageData {
    /// Create RGB image.
    pub fn rgb(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
            channels: 3,
        }
    }

    /// Create RGBA image.
    pub fn rgba(data: Vec<u8>, width: u32, height: u32) -> Self {
        Self {
            data,
            width,
            height,
            channels: 4,
        }
    }
}

/// Trait for metric logging backends.
///
/// Implementations should handle logging to specific backends like TensorBoard,
/// Weights & Biases, or file-based logging.
pub trait MetricLogger: Send {
    /// Log a scalar value.
    ///
    /// # Arguments
    /// * `tag` - Metric name (e.g., "train/loss", "eval/reward")
    /// * `value` - Scalar value
    /// * `step` - Global step number
    fn log_scalar(&mut self, tag: &str, value: f64, step: u64) -> Result<()>;

    /// Log multiple scalar values at once.
    ///
    /// Default implementation calls `log_scalar` for each value.
    fn log_scalars(&mut self, scalars: &[(&str, f64)], step: u64) -> Result<()> {
        for (tag, value) in scalars {
            self.log_scalar(tag, *value, step)?;
        }
        Ok(())
    }

    /// Log a histogram.
    ///
    /// # Arguments
    /// * `tag` - Metric name
    /// * `data` - Histogram data
    /// * `step` - Global step number
    fn log_histogram(&mut self, tag: &str, data: &HistogramData, step: u64) -> Result<()>;

    /// Log video frames.
    ///
    /// # Arguments
    /// * `tag` - Video name
    /// * `video` - Video data
    /// * `step` - Global step number
    fn log_video(&mut self, tag: &str, video: &VideoData, step: u64) -> Result<()>;

    /// Log an image.
    ///
    /// # Arguments
    /// * `tag` - Image name
    /// * `image` - Image data
    /// * `step` - Global step number
    fn log_image(&mut self, tag: &str, image: &ImageData, step: u64) -> Result<()>;

    /// Log text.
    ///
    /// # Arguments
    /// * `tag` - Text name
    /// * `text` - Text content
    /// * `step` - Global step number
    fn log_text(&mut self, tag: &str, text: &str, step: u64) -> Result<()>;

    /// Flush any buffered data to storage.
    fn flush(&mut self) -> Result<()>;

    /// Close the logger and release resources.
    fn close(&mut self) -> Result<()> {
        self.flush()
    }
}

/// Composite logger that fans out to multiple backends.
///
/// Useful for logging to both TensorBoard and Weights & Biases simultaneously.
pub struct CompositeLogger {
    backends: Vec<Box<dyn MetricLogger>>,
}

impl CompositeLogger {
    /// Create an empty composite logger.
    pub fn new() -> Self {
        Self {
            backends: Vec::new(),
        }
    }

    /// Add a logging backend.
    pub fn add_backend(&mut self, backend: Box<dyn MetricLogger>) {
        self.backends.push(backend);
    }

    /// Create with initial backends.
    pub fn with_backends(backends: Vec<Box<dyn MetricLogger>>) -> Self {
        Self { backends }
    }

    /// Number of backends.
    pub fn num_backends(&self) -> usize {
        self.backends.len()
    }
}

impl Default for CompositeLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricLogger for CompositeLogger {
    fn log_scalar(&mut self, tag: &str, value: f64, step: u64) -> Result<()> {
        for backend in &mut self.backends {
            backend.log_scalar(tag, value, step)?;
        }
        Ok(())
    }

    fn log_scalars(&mut self, scalars: &[(&str, f64)], step: u64) -> Result<()> {
        for backend in &mut self.backends {
            backend.log_scalars(scalars, step)?;
        }
        Ok(())
    }

    fn log_histogram(&mut self, tag: &str, data: &HistogramData, step: u64) -> Result<()> {
        for backend in &mut self.backends {
            backend.log_histogram(tag, data, step)?;
        }
        Ok(())
    }

    fn log_video(&mut self, tag: &str, video: &VideoData, step: u64) -> Result<()> {
        for backend in &mut self.backends {
            backend.log_video(tag, video, step)?;
        }
        Ok(())
    }

    fn log_image(&mut self, tag: &str, image: &ImageData, step: u64) -> Result<()> {
        for backend in &mut self.backends {
            backend.log_image(tag, image, step)?;
        }
        Ok(())
    }

    fn log_text(&mut self, tag: &str, text: &str, step: u64) -> Result<()> {
        for backend in &mut self.backends {
            backend.log_text(tag, text, step)?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        for backend in &mut self.backends {
            backend.flush()?;
        }
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        for backend in &mut self.backends {
            backend.close()?;
        }
        Ok(())
    }
}

/// Buffered metric entry.
#[derive(Debug, Clone)]
struct BufferedMetric {
    tag: String,
    value: MetricValue,
    step: u64,
}

/// Buffer for batched metric logging.
///
/// Accumulates metrics and flushes them in batches for efficiency.
pub struct MetricBuffer {
    buffer: Vec<BufferedMetric>,
    capacity: usize,
    auto_flush: bool,
}

impl MetricBuffer {
    /// Create a new metric buffer with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Vec::with_capacity(capacity),
            capacity,
            auto_flush: true,
        }
    }

    /// Disable auto-flush (manual flush only).
    pub fn without_auto_flush(mut self) -> Self {
        self.auto_flush = false;
        self
    }

    /// Add a scalar metric.
    pub fn add_scalar(&mut self, tag: &str, value: f64, step: u64) {
        self.buffer.push(BufferedMetric {
            tag: tag.to_string(),
            value: MetricValue::Scalar(value),
            step,
        });
    }

    /// Add a histogram metric.
    pub fn add_histogram(&mut self, tag: &str, values: Vec<f32>, step: u64) {
        self.buffer.push(BufferedMetric {
            tag: tag.to_string(),
            value: MetricValue::Histogram(HistogramData::from_values(values)),
            step,
        });
    }

    /// Add a text metric.
    pub fn add_text(&mut self, tag: &str, text: String, step: u64) {
        self.buffer.push(BufferedMetric {
            tag: tag.to_string(),
            value: MetricValue::Text(text),
            step,
        });
    }

    /// Check if buffer should be flushed.
    pub fn should_flush(&self) -> bool {
        self.buffer.len() >= self.capacity
    }

    /// Number of buffered metrics.
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Flush buffered metrics to a logger.
    pub fn flush(&mut self, logger: &mut dyn MetricLogger) -> Result<()> {
        for metric in self.buffer.drain(..) {
            match metric.value {
                MetricValue::Scalar(v) => {
                    logger.log_scalar(&metric.tag, v, metric.step)?;
                }
                MetricValue::Histogram(data) => {
                    logger.log_histogram(&metric.tag, &data, metric.step)?;
                }
                MetricValue::Text(text) => {
                    logger.log_text(&metric.tag, &text, metric.step)?;
                }
                MetricValue::Video(video) => {
                    logger.log_video(&metric.tag, &video, metric.step)?;
                }
                MetricValue::Image(image) => {
                    logger.log_image(&metric.tag, &image, metric.step)?;
                }
            }
        }
        Ok(())
    }

    /// Clear the buffer without flushing.
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// A no-op logger for testing or when logging is disabled.
pub struct NullLogger;

impl MetricLogger for NullLogger {
    fn log_scalar(&mut self, _tag: &str, _value: f64, _step: u64) -> Result<()> {
        Ok(())
    }

    fn log_histogram(&mut self, _tag: &str, _data: &HistogramData, _step: u64) -> Result<()> {
        Ok(())
    }

    fn log_video(&mut self, _tag: &str, _video: &VideoData, _step: u64) -> Result<()> {
        Ok(())
    }

    fn log_image(&mut self, _tag: &str, _image: &ImageData, _step: u64) -> Result<()> {
        Ok(())
    }

    fn log_text(&mut self, _tag: &str, _text: &str, _step: u64) -> Result<()> {
        Ok(())
    }

    fn flush(&mut self) -> Result<()> {
        Ok(())
    }
}

/// Metric aggregator for computing statistics over multiple values.
#[derive(Debug, Clone, Default)]
pub struct MetricAggregator {
    metrics: HashMap<String, Vec<f64>>,
}

impl MetricAggregator {
    /// Create a new aggregator.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a value for a metric.
    pub fn add(&mut self, tag: &str, value: f64) {
        self.metrics.entry(tag.to_string()).or_default().push(value);
    }

    /// Get the mean for a metric.
    pub fn mean(&self, tag: &str) -> Option<f64> {
        self.metrics.get(tag).map(|values| {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            }
        })
    }

    /// Get the standard deviation for a metric.
    pub fn std(&self, tag: &str) -> Option<f64> {
        self.metrics.get(tag).map(|values| {
            if values.len() < 2 {
                return 0.0;
            }
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            variance.sqrt()
        })
    }

    /// Get the minimum value for a metric.
    pub fn min(&self, tag: &str) -> Option<f64> {
        self.metrics
            .get(tag)
            .and_then(|v| v.iter().cloned().reduce(f64::min))
    }

    /// Get the maximum value for a metric.
    pub fn max(&self, tag: &str) -> Option<f64> {
        self.metrics
            .get(tag)
            .and_then(|v| v.iter().cloned().reduce(f64::max))
    }

    /// Get all values for a metric.
    pub fn values(&self, tag: &str) -> Option<&[f64]> {
        self.metrics.get(tag).map(|v| v.as_slice())
    }

    /// Get all metric tags.
    pub fn tags(&self) -> impl Iterator<Item = &str> {
        self.metrics.keys().map(|s| s.as_str())
    }

    /// Clear all aggregated values.
    pub fn clear(&mut self) {
        self.metrics.clear();
    }

    /// Clear values for a specific metric.
    pub fn clear_tag(&mut self, tag: &str) {
        self.metrics.remove(tag);
    }

    /// Log all aggregated means to a logger and clear.
    pub fn log_means_and_clear(&mut self, logger: &mut dyn MetricLogger, step: u64) -> Result<()> {
        let scalars: Vec<_> = self
            .metrics
            .iter()
            .map(|(tag, values)| {
                let mean = if values.is_empty() {
                    0.0
                } else {
                    values.iter().sum::<f64>() / values.len() as f64
                };
                (tag.as_str(), mean)
            })
            .collect();

        logger.log_scalars(&scalars, step)?;
        self.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_data() {
        let data = HistogramData::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let stats = data.compute_stats();

        assert!((stats.mean - 3.0).abs() < 1e-6);
        assert!((stats.min - 1.0).abs() < 1e-6);
        assert!((stats.max - 5.0).abs() < 1e-6);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_histogram_bins() {
        let data = HistogramData::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (edges, counts) = data.compute_bins(5);

        assert_eq!(edges.len(), 6); // 5 bins = 6 edges
        assert_eq!(counts.len(), 5);
        assert_eq!(counts.iter().sum::<u64>(), 5); // All values binned
    }

    #[test]
    fn test_composite_logger() {
        let mut logger = CompositeLogger::new();
        logger.add_backend(Box::new(NullLogger));
        logger.add_backend(Box::new(NullLogger));

        assert_eq!(logger.num_backends(), 2);

        logger.log_scalar("test", 1.0, 0).unwrap();
        logger.flush().unwrap();
    }

    #[test]
    fn test_metric_buffer() {
        let mut buffer = MetricBuffer::new(10);
        buffer.add_scalar("loss", 0.5, 100);
        buffer.add_scalar("reward", 100.0, 100);

        assert_eq!(buffer.len(), 2);
        assert!(!buffer.should_flush());

        let mut logger = NullLogger;
        buffer.flush(&mut logger).unwrap();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_metric_aggregator() {
        let mut agg = MetricAggregator::new();
        agg.add("reward", 100.0);
        agg.add("reward", 200.0);
        agg.add("reward", 300.0);

        assert!((agg.mean("reward").unwrap() - 200.0).abs() < 1e-6);
        assert!((agg.min("reward").unwrap() - 100.0).abs() < 1e-6);
        assert!((agg.max("reward").unwrap() - 300.0).abs() < 1e-6);
    }
}
