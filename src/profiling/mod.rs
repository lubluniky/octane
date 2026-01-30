//! Profiling system for performance analysis.
//!
//! This module provides hierarchical timing and profiling capabilities for
//! identifying performance bottlenecks in RL training.
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::profiling::{Profiler, profile_scope};
//!
//! // Using the global profiler
//! {
//!     let _guard = profile_scope!("train_step");
//!
//!     {
//!         let _guard = profile_scope!("rollout");
//!         // Collect rollout...
//!     }
//!
//!     {
//!         let _guard = profile_scope!("update");
//!         // Update policy...
//!     }
//! }
//!
//! // Print profile report
//! println!("{}", Profiler::global().report());
//! ```
//!
//! # Thread Safety
//!
//! The profiler uses thread-local storage for timing data and a global mutex
//! for aggregation, making it safe to use from multiple threads.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Global profiler instance.
static GLOBAL_PROFILER: OnceLock<Profiler> = OnceLock::new();

/// Get or create the global profiler instance.
pub fn global_profiler() -> &'static Profiler {
    GLOBAL_PROFILER.get_or_init(Profiler::new)
}

/// RAII guard for timing a scope.
///
/// When dropped, records the elapsed time to the profiler.
pub struct ProfileScope {
    name: &'static str,
    start: Instant,
    profiler: &'static Profiler,
}

impl ProfileScope {
    /// Create a new profile scope.
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            start: Instant::now(),
            profiler: global_profiler(),
        }
    }

    /// Create a profile scope with a specific profiler.
    pub fn with_profiler(name: &'static str, profiler: &'static Profiler) -> Self {
        Self {
            name,
            start: Instant::now(),
            profiler,
        }
    }
}

impl Drop for ProfileScope {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        self.profiler.record(self.name, elapsed);
    }
}

/// Create a profile scope for the current block.
///
/// # Example
///
/// ```ignore
/// {
///     let _guard = profile_scope!("my_function");
///     // Code to profile...
/// }
/// ```
#[macro_export]
macro_rules! profile_scope {
    ($name:expr) => {
        $crate::profiling::ProfileScope::new($name)
    };
}

/// Timing statistics for a single scope.
#[derive(Debug, Clone)]
pub struct ScopeStats {
    /// Scope name.
    pub name: String,
    /// Total accumulated time.
    pub total_time: Duration,
    /// Number of calls.
    pub call_count: u64,
    /// Minimum call duration.
    pub min_time: Duration,
    /// Maximum call duration.
    pub max_time: Duration,
    /// Sum of squared durations (for variance calculation).
    sum_squared_nanos: u128,
}

impl ScopeStats {
    /// Create new stats for a scope.
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            total_time: Duration::ZERO,
            call_count: 0,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            sum_squared_nanos: 0,
        }
    }

    /// Record a timing sample.
    fn record(&mut self, duration: Duration) {
        self.total_time += duration;
        self.call_count += 1;
        self.min_time = self.min_time.min(duration);
        self.max_time = self.max_time.max(duration);
        let nanos = duration.as_nanos();
        self.sum_squared_nanos += nanos * nanos;
    }

    /// Get average call duration.
    pub fn avg_time(&self) -> Duration {
        if self.call_count == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.call_count as u32
        }
    }

    /// Get standard deviation of call durations.
    pub fn std_time(&self) -> Duration {
        if self.call_count < 2 {
            return Duration::ZERO;
        }

        let n = self.call_count as f64;
        let mean_nanos = self.total_time.as_nanos() as f64 / n;
        let variance = (self.sum_squared_nanos as f64 / n) - (mean_nanos * mean_nanos);
        let std_nanos = variance.max(0.0).sqrt();

        Duration::from_nanos(std_nanos as u64)
    }

    /// Get percentage of total profiled time.
    pub fn percentage(&self, total: Duration) -> f64 {
        if total.is_zero() {
            0.0
        } else {
            (self.total_time.as_secs_f64() / total.as_secs_f64()) * 100.0
        }
    }

    /// Merge another stats into this one.
    fn merge(&mut self, other: &ScopeStats) {
        self.total_time += other.total_time;
        self.call_count += other.call_count;
        self.min_time = self.min_time.min(other.min_time);
        self.max_time = self.max_time.max(other.max_time);
        self.sum_squared_nanos += other.sum_squared_nanos;
    }
}

impl Default for ScopeStats {
    fn default() -> Self {
        Self::new("")
    }
}

/// Thread-local profiler data.
struct ThreadLocalData {
    stats: HashMap<&'static str, ScopeStats>,
}

impl ThreadLocalData {
    fn new() -> Self {
        Self {
            stats: HashMap::new(),
        }
    }

    fn record(&mut self, name: &'static str, duration: Duration) {
        self.stats
            .entry(name)
            .or_insert_with(|| ScopeStats::new(name))
            .record(duration);
    }
}

/// Profiler for collecting timing statistics.
///
/// Thread-safe profiler that collects hierarchical timing data
/// across multiple scopes.
pub struct Profiler {
    /// Aggregated stats from all threads.
    global_stats: Mutex<HashMap<String, ScopeStats>>,
    /// Whether profiling is enabled.
    enabled: Mutex<bool>,
    /// Start time for calculating uptime.
    start_time: Instant,
}

impl Profiler {
    /// Create a new profiler.
    pub fn new() -> Self {
        Self {
            global_stats: Mutex::new(HashMap::new()),
            enabled: Mutex::new(true),
            start_time: Instant::now(),
        }
    }

    /// Get the global profiler instance.
    pub fn global() -> &'static Profiler {
        global_profiler()
    }

    /// Enable profiling.
    pub fn enable(&self) {
        *self.enabled.lock().unwrap() = true;
    }

    /// Disable profiling.
    pub fn disable(&self) {
        *self.enabled.lock().unwrap() = false;
    }

    /// Check if profiling is enabled.
    pub fn is_enabled(&self) -> bool {
        *self.enabled.lock().unwrap()
    }

    /// Record a timing sample.
    pub fn record(&self, name: &'static str, duration: Duration) {
        if !self.is_enabled() {
            return;
        }

        let mut stats = self.global_stats.lock().unwrap();
        stats
            .entry(name.to_string())
            .or_insert_with(|| ScopeStats::new(name))
            .record(duration);
    }

    /// Get stats for a specific scope.
    pub fn get_stats(&self, name: &str) -> Option<ScopeStats> {
        let stats = self.global_stats.lock().unwrap();
        stats.get(name).cloned()
    }

    /// Get all scope stats.
    pub fn all_stats(&self) -> Vec<ScopeStats> {
        let stats = self.global_stats.lock().unwrap();
        stats.values().cloned().collect()
    }

    /// Get total profiled time.
    pub fn total_time(&self) -> Duration {
        let stats = self.global_stats.lock().unwrap();
        stats.values().map(|s| s.total_time).sum()
    }

    /// Get profiler uptime.
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Reset all statistics.
    pub fn reset(&self) {
        let mut stats = self.global_stats.lock().unwrap();
        stats.clear();
    }

    /// Generate a profile report.
    pub fn report(&self) -> ProfileReport {
        let stats = self.global_stats.lock().unwrap();
        let total_time = stats.values().map(|s| s.total_time).sum();

        let mut entries: Vec<_> = stats.values().cloned().collect();
        entries.sort_by(|a, b| b.total_time.cmp(&a.total_time));

        ProfileReport {
            entries,
            total_time,
            uptime: self.uptime(),
        }
    }

    /// Export stats as JSON.
    pub fn to_json(&self) -> String {
        let stats = self.global_stats.lock().unwrap();
        let mut result = String::from("{\n");

        for (i, (name, stat)) in stats.iter().enumerate() {
            if i > 0 {
                result.push_str(",\n");
            }
            result.push_str(&format!(
                "  \"{}\": {{\n    \"total_ms\": {:.3},\n    \"count\": {},\n    \"avg_ms\": {:.3},\n    \"min_ms\": {:.3},\n    \"max_ms\": {:.3}\n  }}",
                name,
                stat.total_time.as_secs_f64() * 1000.0,
                stat.call_count,
                stat.avg_time().as_secs_f64() * 1000.0,
                stat.min_time.as_secs_f64() * 1000.0,
                stat.max_time.as_secs_f64() * 1000.0
            ));
        }

        result.push_str("\n}");
        result
    }
}

impl Default for Profiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Profile report with formatted output.
#[derive(Debug, Clone)]
pub struct ProfileReport {
    /// Scope statistics sorted by total time.
    pub entries: Vec<ScopeStats>,
    /// Total profiled time.
    pub total_time: Duration,
    /// Profiler uptime.
    pub uptime: Duration,
}

impl ProfileReport {
    /// Get the top N scopes by total time.
    pub fn top(&self, n: usize) -> &[ScopeStats] {
        &self.entries[..n.min(self.entries.len())]
    }

    /// Filter scopes by name prefix.
    pub fn filter_prefix(&self, prefix: &str) -> Vec<&ScopeStats> {
        self.entries
            .iter()
            .filter(|s| s.name.starts_with(prefix))
            .collect()
    }
}

impl std::fmt::Display for ProfileReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Profile Report")?;
        writeln!(f, "==============")?;
        writeln!(
            f,
            "Total profiled time: {:.3}s",
            self.total_time.as_secs_f64()
        )?;
        writeln!(f, "Profiler uptime: {:.3}s", self.uptime.as_secs_f64())?;
        writeln!(f)?;

        writeln!(
            f,
            "{:<30} {:>10} {:>10} {:>10} {:>10} {:>10} {:>8}",
            "Scope", "Total", "Count", "Avg", "Min", "Max", "%"
        )?;
        writeln!(f, "{}", "-".repeat(98))?;

        for stat in &self.entries {
            writeln!(
                f,
                "{:<30} {:>10.3}ms {:>10} {:>10.3}ms {:>10.3}ms {:>10.3}ms {:>7.1}%",
                truncate_name(&stat.name, 30),
                stat.total_time.as_secs_f64() * 1000.0,
                stat.call_count,
                stat.avg_time().as_secs_f64() * 1000.0,
                stat.min_time.as_secs_f64() * 1000.0,
                stat.max_time.as_secs_f64() * 1000.0,
                stat.percentage(self.total_time)
            )?;
        }

        Ok(())
    }
}

/// Truncate a name to fit in a column.
fn truncate_name(name: &str, max_len: usize) -> String {
    if name.len() <= max_len {
        name.to_string()
    } else {
        format!("...{}", &name[name.len() - max_len + 3..])
    }
}

/// Timed execution wrapper.
///
/// Executes a closure and records its duration to the profiler.
pub fn timed<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let _guard = ProfileScope::new(name);
    f()
}

/// Profile a function with custom profiler.
pub fn timed_with<T>(name: &'static str, profiler: &'static Profiler, f: impl FnOnce() -> T) -> T {
    let _guard = ProfileScope::with_profiler(name, profiler);
    f()
}

/// Integration points for RL algorithms.
pub mod integration {
    use super::*;

    /// Common scope names for RL algorithms.
    pub mod scopes {
        /// Rollout collection scope.
        pub const ROLLOUT: &str = "rollout";
        /// Policy update scope.
        pub const UPDATE: &str = "update";
        /// Forward pass scope.
        pub const FORWARD: &str = "forward";
        /// Backward pass scope.
        pub const BACKWARD: &str = "backward";
        /// Environment step scope.
        pub const ENV_STEP: &str = "env_step";
        /// Environment reset scope.
        pub const ENV_RESET: &str = "env_reset";
        /// Action sampling scope.
        pub const SAMPLE_ACTION: &str = "sample_action";
        /// GAE computation scope.
        pub const COMPUTE_GAE: &str = "compute_gae";
        /// Batch processing scope.
        pub const BATCH_PROCESS: &str = "batch_process";
        /// Model save scope.
        pub const MODEL_SAVE: &str = "model_save";
        /// Model load scope.
        pub const MODEL_LOAD: &str = "model_load";
    }

    /// Profile a rollout collection.
    pub fn profile_rollout<T>(f: impl FnOnce() -> T) -> T {
        timed(scopes::ROLLOUT, f)
    }

    /// Profile a policy update.
    pub fn profile_update<T>(f: impl FnOnce() -> T) -> T {
        timed(scopes::UPDATE, f)
    }

    /// Profile a forward pass.
    pub fn profile_forward<T>(f: impl FnOnce() -> T) -> T {
        timed(scopes::FORWARD, f)
    }

    /// Profile a backward pass.
    pub fn profile_backward<T>(f: impl FnOnce() -> T) -> T {
        timed(scopes::BACKWARD, f)
    }

    /// Profile an environment step.
    pub fn profile_env_step<T>(f: impl FnOnce() -> T) -> T {
        timed(scopes::ENV_STEP, f)
    }
}

/// Performance counters for detailed analysis.
#[derive(Debug, Clone, Default)]
pub struct PerfCounters {
    /// Number of forward passes.
    pub forward_passes: u64,
    /// Number of backward passes.
    pub backward_passes: u64,
    /// Number of environment steps.
    pub env_steps: u64,
    /// Total samples processed.
    pub samples_processed: u64,
    /// Total bytes transferred to/from GPU.
    pub gpu_bytes_transferred: u64,
}

impl PerfCounters {
    /// Create new performance counters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Reset all counters.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Compute samples per second.
    pub fn samples_per_second(&self, elapsed: Duration) -> f64 {
        if elapsed.is_zero() {
            0.0
        } else {
            self.samples_processed as f64 / elapsed.as_secs_f64()
        }
    }

    /// Compute environment steps per second.
    pub fn steps_per_second(&self, elapsed: Duration) -> f64 {
        if elapsed.is_zero() {
            0.0
        } else {
            self.env_steps as f64 / elapsed.as_secs_f64()
        }
    }
}

/// SIMD statistics integration from the simd module.
#[derive(Debug, Clone, Default)]
pub struct SimdStats {
    /// Total SIMD operations performed.
    pub simd_operations: u64,
    /// Cycles spent in SIMD operations (if available).
    pub simd_cycles: u64,
    /// Bytes processed by SIMD operations.
    pub simd_bytes: u64,
    /// Whether NEON is being used.
    pub using_neon: bool,
}

impl SimdStats {
    /// Create new SIMD stats.
    pub fn new() -> Self {
        Self {
            using_neon: crate::simd::is_neon_available(),
            ..Default::default()
        }
    }

    /// Record a SIMD operation.
    pub fn record_operation(&mut self, bytes: u64, cycles: u64) {
        self.simd_operations += 1;
        self.simd_bytes += bytes;
        self.simd_cycles += cycles;
    }

    /// Compute effective bandwidth in GB/s.
    pub fn bandwidth_gbps(&self, elapsed: Duration) -> f64 {
        if elapsed.is_zero() {
            0.0
        } else {
            (self.simd_bytes as f64) / elapsed.as_secs_f64() / 1e9
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_profile_scope() {
        let profiler = Profiler::new();
        let profiler_ref: &'static Profiler = Box::leak(Box::new(profiler));

        {
            let _guard = ProfileScope::with_profiler("test_scope", profiler_ref);
            thread::sleep(Duration::from_millis(10));
        }

        let stats = profiler_ref.get_stats("test_scope").unwrap();
        assert_eq!(stats.call_count, 1);
        assert!(stats.total_time >= Duration::from_millis(10));
    }

    #[test]
    fn test_scope_stats() {
        let mut stats = ScopeStats::new("test");

        stats.record(Duration::from_millis(10));
        stats.record(Duration::from_millis(20));
        stats.record(Duration::from_millis(30));

        assert_eq!(stats.call_count, 3);
        assert_eq!(stats.total_time, Duration::from_millis(60));
        assert_eq!(stats.min_time, Duration::from_millis(10));
        assert_eq!(stats.max_time, Duration::from_millis(30));
        assert_eq!(stats.avg_time(), Duration::from_millis(20));
    }

    #[test]
    fn test_profiler_enable_disable() {
        let profiler = Profiler::new();

        profiler.record("scope1", Duration::from_millis(10));
        assert!(profiler.get_stats("scope1").is_some());

        profiler.disable();
        profiler.record("scope2", Duration::from_millis(10));
        assert!(profiler.get_stats("scope2").is_none());

        profiler.enable();
        profiler.record("scope3", Duration::from_millis(10));
        assert!(profiler.get_stats("scope3").is_some());
    }

    #[test]
    fn test_timed() {
        let profiler = Profiler::new();
        let profiler_ref: &'static Profiler = Box::leak(Box::new(profiler));

        let result = timed_with("compute", profiler_ref, || {
            thread::sleep(Duration::from_millis(5));
            42
        });

        assert_eq!(result, 42);
        let stats = profiler_ref.get_stats("compute").unwrap();
        assert_eq!(stats.call_count, 1);
    }

    #[test]
    fn test_profile_report() {
        let profiler = Profiler::new();

        profiler.record("slow", Duration::from_millis(100));
        profiler.record("fast", Duration::from_millis(10));
        profiler.record("medium", Duration::from_millis(50));

        let report = profiler.report();
        assert_eq!(report.entries.len(), 3);
        assert_eq!(report.entries[0].name, "slow"); // Sorted by total time
    }

    #[test]
    fn test_perf_counters() {
        let mut counters = PerfCounters::new();
        counters.samples_processed = 10000;
        counters.env_steps = 5000;

        let elapsed = Duration::from_secs(10);
        assert!((counters.samples_per_second(elapsed) - 1000.0).abs() < 0.001);
        assert!((counters.steps_per_second(elapsed) - 500.0).abs() < 0.001);
    }
}
