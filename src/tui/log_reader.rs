//! Log file reader for monitoring training processes.
//!
//! Provides capabilities to read, parse, and tail log files from
//! background training processes.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

/// Maximum number of log lines to keep in memory
const MAX_LOG_LINES: usize = 10000;

/// Log entry with metadata
#[derive(Debug, Clone)]
pub struct LogEntry {
    /// Timestamp (if parsed from log)
    pub timestamp: Option<SystemTime>,
    /// Log level (INFO, WARN, ERROR, etc.)
    pub level: LogLevel,
    /// Log message
    pub message: String,
    /// Raw line from file
    pub raw: String,
}

/// Log level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
    Train,  // Special level for training metrics
    Unknown,
}

impl LogLevel {
    /// Parse log level from string
    pub fn from_str(s: &str) -> Self {
        match s.to_uppercase().as_str() {
            "TRACE" => LogLevel::Trace,
            "DEBUG" => LogLevel::Debug,
            "INFO" => LogLevel::Info,
            "WARN" | "WARNING" => LogLevel::Warn,
            "ERROR" => LogLevel::Error,
            "TRAIN" => LogLevel::Train,
            _ => LogLevel::Unknown,
        }
    }

    /// Get display string
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Trace => "TRACE",
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
            LogLevel::Train => "TRAIN",
            LogLevel::Unknown => "UNKNOWN",
        }
    }
}

/// Log file reader with tailing capability
pub struct LogReader {
    /// Path to the log file
    path: PathBuf,
    /// File handle
    file: BufReader<File>,
    /// Current file position
    position: u64,
    /// Last modified time
    last_modified: Option<SystemTime>,
    /// Buffer of log entries
    entries: VecDeque<LogEntry>,
    /// Maximum entries to keep
    max_entries: usize,
    /// Filter by log level
    level_filter: Option<LogLevel>,
    /// Search filter
    search_filter: Option<String>,
}

impl LogReader {
    /// Create a new log reader for the specified file
    pub fn new<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let path = path.as_ref().to_path_buf();
        let file = File::open(&path)?;
        let metadata = file.metadata()?;
        let last_modified = metadata.modified().ok();

        Ok(Self {
            path,
            file: BufReader::new(file),
            position: 0,
            last_modified,
            entries: VecDeque::new(),
            max_entries: MAX_LOG_LINES,
            level_filter: None,
            search_filter: None,
        })
    }

    /// Set log level filter (only show logs at this level or higher)
    pub fn set_level_filter(&mut self, level: Option<LogLevel>) {
        self.level_filter = level;
    }

    /// Set search filter (case-insensitive substring match)
    pub fn set_search_filter(&mut self, filter: Option<String>) {
        self.search_filter = filter;
    }

    /// Read all existing lines from the file
    pub fn read_all(&mut self) -> io::Result<()> {
        self.file.seek(SeekFrom::Start(0))?;
        self.position = 0;
        self.entries.clear();

        let mut line = String::new();
        loop {
            line.clear();
            let bytes_read = self.file.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }

            self.position += bytes_read as u64;

            if let Some(entry) = Self::parse_line(&line) {
                if self.should_include(&entry) {
                    self.entries.push_back(entry);
                    if self.entries.len() > self.max_entries {
                        self.entries.pop_front();
                    }
                }
            }
        }

        Ok(())
    }

    /// Check for new lines (tail mode)
    pub fn check_for_updates(&mut self) -> io::Result<bool> {
        // Check if file was modified
        let metadata = std::fs::metadata(&self.path)?;
        let current_modified = metadata.modified().ok();

        if current_modified == self.last_modified {
            return Ok(false);
        }

        self.last_modified = current_modified;

        // Reopen file if it was rotated/truncated
        let current_size = metadata.len();
        if current_size < self.position {
            // File was truncated or rotated
            self.file = BufReader::new(File::open(&self.path)?);
            self.position = 0;
            self.entries.clear();
            return self.read_all().map(|_| true);
        }

        // Read new lines
        let mut new_lines = false;
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = self.file.read_line(&mut line)?;
            if bytes_read == 0 {
                break;
            }

            self.position += bytes_read as u64;
            new_lines = true;

            if let Some(entry) = Self::parse_line(&line) {
                if self.should_include(&entry) {
                    self.entries.push_back(entry);
                    if self.entries.len() > self.max_entries {
                        self.entries.pop_front();
                    }
                }
            }
        }

        Ok(new_lines)
    }

    /// Parse a log line into a LogEntry
    fn parse_line(line: &str) -> Option<LogEntry> {
        let line = line.trim();
        if line.is_empty() {
            return None;
        }

        // Try to parse format: [LEVEL] message
        // or: timestamp [LEVEL] message
        let level = if let Some(start) = line.find('[') {
            if let Some(end) = line.find(']') {
                let level_str = &line[start + 1..end];
                LogLevel::from_str(level_str)
            } else {
                LogLevel::Unknown
            }
        } else {
            LogLevel::Unknown
        };

        Some(LogEntry {
            timestamp: None, // Could parse timestamp if format is known
            level,
            message: line.to_string(),
            raw: line.to_string(),
        })
    }

    /// Check if entry should be included based on filters
    fn should_include(&self, entry: &LogEntry) -> bool {
        // Level filter
        if let Some(filter_level) = self.level_filter {
            if !Self::level_matches(entry.level, filter_level) {
                return false;
            }
        }

        // Search filter
        if let Some(ref search) = self.search_filter {
            if !entry.message.to_lowercase().contains(&search.to_lowercase()) {
                return false;
            }
        }

        true
    }

    /// Check if log level matches filter (includes higher severity)
    fn level_matches(entry_level: LogLevel, filter_level: LogLevel) -> bool {
        let entry_severity = Self::level_severity(entry_level);
        let filter_severity = Self::level_severity(filter_level);
        entry_severity >= filter_severity
    }

    /// Get numeric severity for log level
    fn level_severity(level: LogLevel) -> u8 {
        match level {
            LogLevel::Trace => 0,
            LogLevel::Debug => 1,
            LogLevel::Info => 2,
            LogLevel::Train => 2,
            LogLevel::Warn => 3,
            LogLevel::Error => 4,
            LogLevel::Unknown => 0,
        }
    }

    /// Get all log entries
    pub fn entries(&self) -> &VecDeque<LogEntry> {
        &self.entries
    }

    /// Get recent log entries (last N)
    pub fn recent_entries(&self, count: usize) -> Vec<&LogEntry> {
        self.entries.iter().rev().take(count).collect()
    }

    /// Get entries by level
    pub fn entries_by_level(&self, level: LogLevel) -> Vec<&LogEntry> {
        self.entries
            .iter()
            .filter(|e| e.level == level)
            .collect()
    }

    /// Clear all entries
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get file path
    pub fn path(&self) -> &Path {
        &self.path
    }
}

/// Multi-file log reader for monitoring multiple log files
pub struct MultiLogReader {
    readers: Vec<(String, LogReader)>,
}

impl MultiLogReader {
    pub fn new() -> Self {
        Self {
            readers: Vec::new(),
        }
    }

    /// Add a log file to monitor
    pub fn add_file<P: AsRef<Path>>(&mut self, name: String, path: P) -> io::Result<()> {
        let mut reader = LogReader::new(path)?;
        reader.read_all()?;
        self.readers.push((name, reader));
        Ok(())
    }

    /// Check all files for updates
    pub fn check_for_updates(&mut self) -> io::Result<bool> {
        let mut any_updated = false;
        for (_, reader) in &mut self.readers {
            if reader.check_for_updates()? {
                any_updated = true;
            }
        }
        Ok(any_updated)
    }

    /// Get all entries from all files, merged and sorted
    pub fn all_entries(&self) -> Vec<(&str, &LogEntry)> {
        let mut entries = Vec::new();
        for (name, reader) in &self.readers {
            for entry in reader.entries() {
                entries.push((name.as_str(), entry));
            }
        }
        // Could sort by timestamp if available
        entries
    }

    /// Get entries from a specific file
    pub fn entries_for(&self, name: &str) -> Option<&VecDeque<LogEntry>> {
        self.readers
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, r)| r.entries())
    }

    /// Get list of monitored files
    pub fn files(&self) -> Vec<&str> {
        self.readers.iter().map(|(n, _)| n.as_str()).collect()
    }
}

impl Default for MultiLogReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Auto-discover log files in a directory
pub fn discover_log_files<P: AsRef<Path>>(dir: P) -> io::Result<Vec<PathBuf>> {
    let mut log_files = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.is_file() {
            if let Some(ext) = path.extension() {
                if ext == "log" || ext == "txt" {
                    log_files.push(path);
                }
            }
        }
    }

    // Sort by modification time (newest first)
    log_files.sort_by(|a, b| {
        let a_modified = std::fs::metadata(a).ok().and_then(|m| m.modified().ok());
        let b_modified = std::fs::metadata(b).ok().and_then(|m| m.modified().ok());
        b_modified.cmp(&a_modified)
    });

    Ok(log_files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_log_level_parsing() {
        assert_eq!(LogLevel::from_str("INFO"), LogLevel::Info);
        assert_eq!(LogLevel::from_str("warn"), LogLevel::Warn);
        assert_eq!(LogLevel::from_str("ERROR"), LogLevel::Error);
        assert_eq!(LogLevel::from_str("unknown"), LogLevel::Unknown);
    }

    #[test]
    fn test_log_reader() -> io::Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "[INFO] Test log line 1")?;
        writeln!(temp_file, "[WARN] Test log line 2")?;
        writeln!(temp_file, "[ERROR] Test log line 3")?;
        temp_file.flush()?;

        let mut reader = LogReader::new(temp_file.path())?;
        reader.read_all()?;

        assert_eq!(reader.entries().len(), 3);
        assert_eq!(reader.entries()[0].level, LogLevel::Info);
        assert_eq!(reader.entries()[1].level, LogLevel::Warn);
        assert_eq!(reader.entries()[2].level, LogLevel::Error);

        Ok(())
    }

    #[test]
    fn test_level_filter() -> io::Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "[DEBUG] Debug message")?;
        writeln!(temp_file, "[INFO] Info message")?;
        writeln!(temp_file, "[WARN] Warning message")?;
        writeln!(temp_file, "[ERROR] Error message")?;
        temp_file.flush()?;

        let mut reader = LogReader::new(temp_file.path())?;
        reader.set_level_filter(Some(LogLevel::Warn));
        reader.read_all()?;

        // Should only have WARN and ERROR
        assert_eq!(reader.entries().len(), 2);
        assert!(reader.entries().iter().all(|e|
            e.level == LogLevel::Warn || e.level == LogLevel::Error
        ));

        Ok(())
    }

    #[test]
    fn test_search_filter() -> io::Result<()> {
        let mut temp_file = NamedTempFile::new()?;
        writeln!(temp_file, "[INFO] Training episode 1")?;
        writeln!(temp_file, "[INFO] Something else")?;
        writeln!(temp_file, "[INFO] Training episode 2")?;
        temp_file.flush()?;

        let mut reader = LogReader::new(temp_file.path())?;
        reader.set_search_filter(Some("Training".to_string()));
        reader.read_all()?;

        // Should only have lines with "Training"
        assert_eq!(reader.entries().len(), 2);

        Ok(())
    }
}
