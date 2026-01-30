//! TensorBoard logging backend using TFRecord format.
//!
//! This module provides a pure Rust implementation of TensorBoard event file writing,
//! compatible with `tensorboard --logdir`.
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::logging::{TensorBoardWriter, MetricLogger};
//!
//! let mut writer = TensorBoardWriter::new("logs/tensorboard")?;
//! writer.log_scalar("train/loss", 0.5, 1000)?;
//! writer.log_scalar("train/reward", 100.0, 1000)?;
//! writer.flush()?;
//! ```
//!
//! Then run: `tensorboard --logdir logs/tensorboard`

use crate::core::Result;
use crate::logging::metrics::{HistogramData, ImageData, MetricLogger, VideoData};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

/// CRC32C lookup table for TFRecord format.
const CRC_TABLE: [u32; 256] = generate_crc_table();

/// Generate CRC32C table at compile time.
const fn generate_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0x82F63B78;
            } else {
                crc >>= 1;
            }
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Compute CRC32C checksum.
fn crc32c(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC_TABLE[index];
    }
    crc ^ 0xFFFFFFFF
}

/// Mask CRC for TFRecord format.
fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c(data);
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282ead8)
}

/// TensorBoard event writer using TFRecord format.
///
/// Writes events to files that can be read by TensorBoard.
/// Each log directory contains event files named `events.out.tfevents.{timestamp}.{hostname}`.
pub struct TensorBoardWriter {
    /// Log directory path.
    log_dir: PathBuf,
    /// Event file writer.
    writer: BufWriter<File>,
    /// Current wall time.
    wall_time: f64,
    /// Hostname for file naming.
    hostname: String,
    /// File index for multiple files.
    file_index: u32,
}

impl TensorBoardWriter {
    /// Create a new TensorBoard writer.
    ///
    /// Creates the log directory if it doesn't exist and starts a new event file.
    pub fn new(log_dir: impl AsRef<Path>) -> Result<Self> {
        let log_dir = log_dir.as_ref().to_path_buf();
        fs::create_dir_all(&log_dir)?;

        let hostname = hostname::get()
            .map(|h| h.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "localhost".to_string());

        let wall_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(0.0);

        let filename = format!("events.out.tfevents.{:.0}.{}", wall_time, hostname);
        let filepath = log_dir.join(&filename);
        let file = File::create(&filepath)?;
        let writer = BufWriter::new(file);

        let mut tb_writer = Self {
            log_dir,
            writer,
            wall_time,
            hostname,
            file_index: 0,
        };

        // Write file_version event
        tb_writer.write_file_version()?;

        Ok(tb_writer)
    }

    /// Create with a custom subdirectory name.
    pub fn with_name(log_dir: impl AsRef<Path>, name: &str) -> Result<Self> {
        let log_dir = log_dir.as_ref().join(name);
        Self::new(log_dir)
    }

    /// Get the log directory path.
    pub fn log_dir(&self) -> &Path {
        &self.log_dir
    }

    /// Write file version event.
    fn write_file_version(&mut self) -> Result<()> {
        let event = Event {
            wall_time: self.wall_time,
            step: 0,
            value: EventValue::FileVersion("brain.Event:2".to_string()),
        };
        self.write_event(&event)
    }

    /// Write a TFRecord to the file.
    fn write_tfrecord(&mut self, data: &[u8]) -> Result<()> {
        // TFRecord format:
        // uint64    length
        // uint32    masked_crc32_of_length
        // byte      data[length]
        // uint32    masked_crc32_of_data

        let length = data.len() as u64;
        let length_bytes = length.to_le_bytes();
        let length_crc = masked_crc32c(&length_bytes);
        let data_crc = masked_crc32c(data);

        self.writer.write_all(&length_bytes)?;
        self.writer.write_all(&length_crc.to_le_bytes())?;
        self.writer.write_all(data)?;
        self.writer.write_all(&data_crc.to_le_bytes())?;

        Ok(())
    }

    /// Write an event to the file.
    fn write_event(&mut self, event: &Event) -> Result<()> {
        let serialized = event.serialize();
        self.write_tfrecord(&serialized)
    }

    /// Add a graph definition (optional, for network visualization).
    pub fn add_graph(&mut self, graph_def: &[u8]) -> Result<()> {
        let event = Event {
            wall_time: self.current_wall_time(),
            step: 0,
            value: EventValue::GraphDef(graph_def.to_vec()),
        };
        self.write_event(&event)
    }

    /// Get current wall time.
    fn current_wall_time(&self) -> f64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs_f64())
            .unwrap_or(self.wall_time)
    }
}

impl MetricLogger for TensorBoardWriter {
    fn log_scalar(&mut self, tag: &str, value: f64, step: u64) -> Result<()> {
        let event = Event {
            wall_time: self.current_wall_time(),
            step: step as i64,
            value: EventValue::Summary(vec![SummaryValue::Scalar {
                tag: tag.to_string(),
                value: value as f32,
            }]),
        };
        self.write_event(&event)
    }

    fn log_histogram(&mut self, tag: &str, data: &HistogramData, step: u64) -> Result<()> {
        let stats = data.compute_stats();
        let (bucket_limits, bucket_counts) =
            if data.bin_edges.is_some() && data.bin_counts.is_some() {
                (
                    data.bin_edges.clone().unwrap(),
                    data.bin_counts.clone().unwrap(),
                )
            } else {
                data.compute_bins(30)
            };

        let histogram = HistogramProto {
            min: stats.min,
            max: stats.max,
            num: stats.count as f64,
            sum: stats.sum,
            sum_squares: stats.sum_squares,
            bucket_limits: bucket_limits.iter().map(|&v| v as f64).collect(),
            bucket_counts: bucket_counts.iter().map(|&v| v as f64).collect(),
        };

        let event = Event {
            wall_time: self.current_wall_time(),
            step: step as i64,
            value: EventValue::Summary(vec![SummaryValue::Histogram {
                tag: tag.to_string(),
                histogram,
            }]),
        };
        self.write_event(&event)
    }

    fn log_video(&mut self, tag: &str, video: &VideoData, step: u64) -> Result<()> {
        // TensorBoard doesn't natively support video in the same way as images.
        // We log the first frame as an image with a note about video.
        if let Some(first_frame) = video.frames.first() {
            let image = ImageData::rgba(first_frame.clone(), video.width, video.height);
            self.log_image(&format!("{}/frame_0", tag), &image, step)?;
        }

        // Log video metadata as text
        let metadata = format!(
            "Video: {} frames, {}x{}, {:.1} fps, {:.2}s duration",
            video.num_frames(),
            video.width,
            video.height,
            video.fps,
            video.duration()
        );
        self.log_text(&format!("{}/metadata", tag), &metadata, step)
    }

    fn log_image(&mut self, tag: &str, image: &ImageData, step: u64) -> Result<()> {
        let encoded = encode_png(image)?;
        let event = Event {
            wall_time: self.current_wall_time(),
            step: step as i64,
            value: EventValue::Summary(vec![SummaryValue::Image {
                tag: tag.to_string(),
                height: image.height,
                width: image.width,
                colorspace: image.channels as u32,
                encoded_image: encoded,
            }]),
        };
        self.write_event(&event)
    }

    fn log_text(&mut self, tag: &str, text: &str, step: u64) -> Result<()> {
        let event = Event {
            wall_time: self.current_wall_time(),
            step: step as i64,
            value: EventValue::Summary(vec![SummaryValue::Text {
                tag: tag.to_string(),
                text: text.to_string(),
            }]),
        };
        self.write_event(&event)
    }

    fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }

    fn close(&mut self) -> Result<()> {
        self.flush()
    }
}

/// Internal event representation.
struct Event {
    wall_time: f64,
    step: i64,
    value: EventValue,
}

/// Event value types.
enum EventValue {
    FileVersion(String),
    GraphDef(Vec<u8>),
    Summary(Vec<SummaryValue>),
}

/// Summary value types.
enum SummaryValue {
    Scalar {
        tag: String,
        value: f32,
    },
    Histogram {
        tag: String,
        histogram: HistogramProto,
    },
    Image {
        tag: String,
        height: u32,
        width: u32,
        colorspace: u32,
        encoded_image: Vec<u8>,
    },
    Text {
        tag: String,
        text: String,
    },
}

/// Histogram protocol buffer representation.
struct HistogramProto {
    min: f64,
    max: f64,
    num: f64,
    sum: f64,
    sum_squares: f64,
    bucket_limits: Vec<f64>,
    bucket_counts: Vec<f64>,
}

impl Event {
    /// Serialize event to protocol buffer wire format.
    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // wall_time (field 1, double)
        buf.push(0x09); // field 1, wire type 1 (64-bit)
        buf.extend_from_slice(&self.wall_time.to_le_bytes());

        // step (field 2, int64)
        buf.push(0x10); // field 2, wire type 0 (varint)
        encode_varint(&mut buf, self.step as u64);

        // value
        match &self.value {
            EventValue::FileVersion(version) => {
                // file_version (field 3, string)
                buf.push(0x1a); // field 3, wire type 2 (length-delimited)
                encode_varint(&mut buf, version.len() as u64);
                buf.extend_from_slice(version.as_bytes());
            }
            EventValue::GraphDef(graph_def) => {
                // graph_def (field 4, bytes)
                buf.push(0x22); // field 4, wire type 2
                encode_varint(&mut buf, graph_def.len() as u64);
                buf.extend_from_slice(graph_def);
            }
            EventValue::Summary(values) => {
                // summary (field 5, message)
                let summary_bytes = serialize_summary(values);
                buf.push(0x2a); // field 5, wire type 2
                encode_varint(&mut buf, summary_bytes.len() as u64);
                buf.extend_from_slice(&summary_bytes);
            }
        }

        buf
    }
}

/// Serialize summary to protocol buffer wire format.
fn serialize_summary(values: &[SummaryValue]) -> Vec<u8> {
    let mut buf = Vec::new();

    for value in values {
        // value (field 1, repeated message)
        let value_bytes = serialize_summary_value(value);
        buf.push(0x0a); // field 1, wire type 2
        encode_varint(&mut buf, value_bytes.len() as u64);
        buf.extend_from_slice(&value_bytes);
    }

    buf
}

/// Serialize a single summary value.
fn serialize_summary_value(value: &SummaryValue) -> Vec<u8> {
    let mut buf = Vec::new();

    match value {
        SummaryValue::Scalar { tag, value } => {
            // tag (field 1, string)
            buf.push(0x0a);
            encode_varint(&mut buf, tag.len() as u64);
            buf.extend_from_slice(tag.as_bytes());

            // simple_value (field 2, float)
            buf.push(0x15); // field 2, wire type 5 (32-bit)
            buf.extend_from_slice(&value.to_le_bytes());
        }
        SummaryValue::Histogram { tag, histogram } => {
            // tag (field 1, string)
            buf.push(0x0a);
            encode_varint(&mut buf, tag.len() as u64);
            buf.extend_from_slice(tag.as_bytes());

            // histo (field 4, message)
            let histo_bytes = serialize_histogram(histogram);
            buf.push(0x22); // field 4, wire type 2
            encode_varint(&mut buf, histo_bytes.len() as u64);
            buf.extend_from_slice(&histo_bytes);
        }
        SummaryValue::Image {
            tag,
            height,
            width,
            colorspace,
            encoded_image,
        } => {
            // tag (field 1, string)
            buf.push(0x0a);
            encode_varint(&mut buf, tag.len() as u64);
            buf.extend_from_slice(tag.as_bytes());

            // image (field 3, message)
            let image_bytes = serialize_image(*height, *width, *colorspace, encoded_image);
            buf.push(0x1a); // field 3, wire type 2
            encode_varint(&mut buf, image_bytes.len() as u64);
            buf.extend_from_slice(&image_bytes);
        }
        SummaryValue::Text { tag, text } => {
            // tag (field 1, string)
            buf.push(0x0a);
            encode_varint(&mut buf, tag.len() as u64);
            buf.extend_from_slice(tag.as_bytes());

            // metadata (field 9, message) - mark as text plugin
            let metadata = serialize_text_metadata();
            buf.push(0x4a); // field 9, wire type 2
            encode_varint(&mut buf, metadata.len() as u64);
            buf.extend_from_slice(&metadata);

            // tensor (field 8, message) - text content as tensor
            let tensor_bytes = serialize_text_tensor(text);
            buf.push(0x42); // field 8, wire type 2
            encode_varint(&mut buf, tensor_bytes.len() as u64);
            buf.extend_from_slice(&tensor_bytes);
        }
    }

    buf
}

/// Serialize histogram to protobuf.
fn serialize_histogram(histogram: &HistogramProto) -> Vec<u8> {
    let mut buf = Vec::new();

    // min (field 1, double)
    buf.push(0x09);
    buf.extend_from_slice(&histogram.min.to_le_bytes());

    // max (field 2, double)
    buf.push(0x11);
    buf.extend_from_slice(&histogram.max.to_le_bytes());

    // num (field 3, double)
    buf.push(0x19);
    buf.extend_from_slice(&histogram.num.to_le_bytes());

    // sum (field 4, double)
    buf.push(0x21);
    buf.extend_from_slice(&histogram.sum.to_le_bytes());

    // sum_squares (field 5, double)
    buf.push(0x29);
    buf.extend_from_slice(&histogram.sum_squares.to_le_bytes());

    // bucket_limit (field 6, repeated double) - packed
    if !histogram.bucket_limits.is_empty() {
        buf.push(0x32); // field 6, wire type 2 (packed)
        let packed: Vec<u8> = histogram
            .bucket_limits
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        encode_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    // bucket (field 7, repeated double) - packed
    if !histogram.bucket_counts.is_empty() {
        buf.push(0x3a); // field 7, wire type 2 (packed)
        let packed: Vec<u8> = histogram
            .bucket_counts
            .iter()
            .flat_map(|&v| v.to_le_bytes())
            .collect();
        encode_varint(&mut buf, packed.len() as u64);
        buf.extend_from_slice(&packed);
    }

    buf
}

/// Serialize image to protobuf.
fn serialize_image(height: u32, width: u32, colorspace: u32, encoded: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();

    // height (field 1, int32)
    buf.push(0x08);
    encode_varint(&mut buf, height as u64);

    // width (field 2, int32)
    buf.push(0x10);
    encode_varint(&mut buf, width as u64);

    // colorspace (field 3, int32)
    buf.push(0x18);
    encode_varint(&mut buf, colorspace as u64);

    // encoded_image_string (field 4, bytes)
    buf.push(0x22);
    encode_varint(&mut buf, encoded.len() as u64);
    buf.extend_from_slice(encoded);

    buf
}

/// Serialize text metadata for TensorBoard text plugin.
fn serialize_text_metadata() -> Vec<u8> {
    let mut buf = Vec::new();

    // plugin_data (field 1, message)
    let plugin_name = "text";
    let mut plugin_data = Vec::new();
    plugin_data.push(0x0a); // plugin_name field 1
    encode_varint(&mut plugin_data, plugin_name.len() as u64);
    plugin_data.extend_from_slice(plugin_name.as_bytes());

    buf.push(0x0a);
    encode_varint(&mut buf, plugin_data.len() as u64);
    buf.extend_from_slice(&plugin_data);

    buf
}

/// Serialize text as tensor for TensorBoard.
fn serialize_text_tensor(text: &str) -> Vec<u8> {
    let mut buf = Vec::new();

    // dtype (field 1, DT_STRING = 7)
    buf.push(0x08);
    encode_varint(&mut buf, 7);

    // string_val (field 8, repeated bytes)
    buf.push(0x42);
    encode_varint(&mut buf, text.len() as u64);
    buf.extend_from_slice(text.as_bytes());

    buf
}

/// Encode a varint (variable-length integer).
fn encode_varint(buf: &mut Vec<u8>, mut value: u64) {
    loop {
        let byte = (value & 0x7F) as u8;
        value >>= 7;
        if value == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// Simple PNG encoder for images.
fn encode_png(image: &ImageData) -> Result<Vec<u8>> {
    // Minimal PNG encoder - for production, consider using the `png` crate.
    // This is a simplified implementation for basic image logging.

    let mut buf = Vec::new();

    // PNG signature
    buf.extend_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);

    // IHDR chunk
    let color_type = if image.channels == 4 { 6 } else { 2 }; // RGBA or RGB
    let ihdr_data = [
        ((image.width >> 24) & 0xFF) as u8,
        ((image.width >> 16) & 0xFF) as u8,
        ((image.width >> 8) & 0xFF) as u8,
        (image.width & 0xFF) as u8,
        ((image.height >> 24) & 0xFF) as u8,
        ((image.height >> 16) & 0xFF) as u8,
        ((image.height >> 8) & 0xFF) as u8,
        (image.height & 0xFF) as u8,
        8, // bit depth
        color_type,
        0, // compression
        0, // filter
        0, // interlace
    ];
    write_png_chunk(&mut buf, b"IHDR", &ihdr_data);

    // IDAT chunk (uncompressed for simplicity - uses zlib store)
    let raw_data = prepare_raw_image_data(image);
    let compressed = zlib_store(&raw_data);
    write_png_chunk(&mut buf, b"IDAT", &compressed);

    // IEND chunk
    write_png_chunk(&mut buf, b"IEND", &[]);

    Ok(buf)
}

/// Prepare raw image data with filter bytes.
fn prepare_raw_image_data(image: &ImageData) -> Vec<u8> {
    let row_size = image.width as usize * image.channels as usize;
    let mut raw = Vec::with_capacity((row_size + 1) * image.height as usize);

    for y in 0..image.height as usize {
        raw.push(0); // No filter
        let start = y * row_size;
        let end = start + row_size;
        if end <= image.data.len() {
            raw.extend_from_slice(&image.data[start..end]);
        } else {
            // Pad with zeros if data is incomplete
            raw.extend_from_slice(&image.data[start..]);
            raw.resize(raw.len() + (end - image.data.len()), 0);
        }
    }

    raw
}

/// Simple zlib store (no compression).
fn zlib_store(data: &[u8]) -> Vec<u8> {
    let mut buf = Vec::new();

    // zlib header (no compression)
    buf.push(0x78); // CMF
    buf.push(0x01); // FLG

    // Store blocks
    let max_block = 65535;
    let mut offset = 0;

    while offset < data.len() {
        let remaining = data.len() - offset;
        let block_size = remaining.min(max_block);
        let is_final = offset + block_size >= data.len();

        buf.push(if is_final { 0x01 } else { 0x00 }); // BFINAL + BTYPE (store)
        buf.push((block_size & 0xFF) as u8);
        buf.push(((block_size >> 8) & 0xFF) as u8);
        buf.push((!block_size & 0xFF) as u8);
        buf.push(((!block_size >> 8) & 0xFF) as u8);

        buf.extend_from_slice(&data[offset..offset + block_size]);
        offset += block_size;
    }

    // Adler-32 checksum
    let adler = adler32(data);
    buf.push(((adler >> 24) & 0xFF) as u8);
    buf.push(((adler >> 16) & 0xFF) as u8);
    buf.push(((adler >> 8) & 0xFF) as u8);
    buf.push((adler & 0xFF) as u8);

    buf
}

/// Compute Adler-32 checksum.
fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;

    for &byte in data {
        a = (a + byte as u32) % 65521;
        b = (b + a) % 65521;
    }

    (b << 16) | a
}

/// Write a PNG chunk.
fn write_png_chunk(buf: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    let length = data.len() as u32;
    buf.extend_from_slice(&length.to_be_bytes());
    buf.extend_from_slice(chunk_type);
    buf.extend_from_slice(data);

    // CRC32 of type + data
    let mut crc_data = Vec::with_capacity(4 + data.len());
    crc_data.extend_from_slice(chunk_type);
    crc_data.extend_from_slice(data);
    let crc = png_crc32(&crc_data);
    buf.extend_from_slice(&crc.to_be_bytes());
}

/// PNG CRC32 (different polynomial than CRC32C).
fn png_crc32(data: &[u8]) -> u32 {
    static PNG_CRC_TABLE: [u32; 256] = {
        let mut table = [0u32; 256];
        let mut i = 0;
        while i < 256 {
            let mut crc = i as u32;
            let mut j = 0;
            while j < 8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
                j += 1;
            }
            table[i] = crc;
            i += 1;
        }
        table
    };

    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ PNG_CRC_TABLE[index];
    }
    crc ^ 0xFFFFFFFF
}

/// Get system hostname.
mod hostname {
    use std::ffi::OsString;

    pub fn get() -> std::io::Result<OsString> {
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let mut buf = vec![0u8; 256];
            let ret = unsafe { libc::gethostname(buf.as_mut_ptr() as *mut i8, buf.len()) };
            if ret == 0 {
                let len = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
                buf.truncate(len);
                Ok(OsString::from_vec(buf))
            } else {
                Err(std::io::Error::last_os_error())
            }
        }

        #[cfg(windows)]
        {
            use std::env;
            env::var_os("COMPUTERNAME")
                .or_else(|| env::var_os("HOSTNAME"))
                .ok_or_else(|| {
                    std::io::Error::new(std::io::ErrorKind::NotFound, "hostname not found")
                })
        }

        #[cfg(not(any(unix, windows)))]
        {
            Ok(OsString::from("localhost"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_tensorboard_writer_scalar() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = TensorBoardWriter::new(temp_dir.path()).unwrap();

        writer.log_scalar("train/loss", 0.5, 100).unwrap();
        writer.log_scalar("train/reward", 100.0, 100).unwrap();
        writer.flush().unwrap();

        // Check that event file was created
        let entries: Vec<_> = std::fs::read_dir(temp_dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(entries.len(), 1);
        assert!(entries[0]
            .file_name()
            .to_string_lossy()
            .contains("tfevents"));
    }

    #[test]
    fn test_tensorboard_writer_histogram() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = TensorBoardWriter::new(temp_dir.path()).unwrap();

        let data = HistogramData::from_values(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        writer.log_histogram("weights/layer1", &data, 100).unwrap();
        writer.flush().unwrap();
    }

    #[test]
    fn test_tensorboard_writer_text() {
        let temp_dir = TempDir::new().unwrap();
        let mut writer = TensorBoardWriter::new(temp_dir.path()).unwrap();

        writer
            .log_text("config", "learning_rate: 0.001", 0)
            .unwrap();
        writer.flush().unwrap();
    }

    #[test]
    fn test_crc32c() {
        // Test vector from RFC 3720
        let data = b"123456789";
        let crc = crc32c(data);
        assert_eq!(crc, 0xE3069283);
    }

    #[test]
    fn test_varint_encoding() {
        let mut buf = Vec::new();
        encode_varint(&mut buf, 300);
        assert_eq!(buf, vec![0xAC, 0x02]);

        buf.clear();
        encode_varint(&mut buf, 0);
        assert_eq!(buf, vec![0x00]);

        buf.clear();
        encode_varint(&mut buf, 127);
        assert_eq!(buf, vec![0x7F]);
    }
}
