//! System resource monitoring for the Octane web dashboard.
//!
//! Collects CPU, memory and (best-effort) GPU telemetry using [`sysinfo`]
//! plus platform-specific probes (`ioreg` on macOS, `nvidia-smi` elsewhere).
//!
//! All values are honest. GPU utilization is only reported when a real
//! measurement is available; otherwise [`SystemSnapshot::gpu_usage_available`]
//! is `false` and the frontend renders it as "N/A" rather than a fabricated
//! number. CPU and RAM are always real (sourced from `sysinfo`).

use serde::Serialize;
use std::process::Command;
use sysinfo::{
    get_current_pid, CpuRefreshKind, MemoryRefreshKind, Pid, ProcessesToUpdate, RefreshKind, System,
};

/// A point-in-time snapshot of system resource usage.
///
/// Serialized directly into the dashboard JSON payload.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SystemSnapshot {
    /// Average CPU usage across all cores, in percent (0-100).
    pub cpu_usage: f32,
    /// Per-core CPU usage, in percent (0-100).
    pub cpu_per_core: Vec<f32>,
    /// Human-readable CPU model name.
    pub cpu_name: String,
    /// Number of logical CPU cores.
    pub cpu_cores: usize,
    /// System load average over [1, 5, 15] minutes.
    pub load_avg: [f64; 3],
    /// Used physical memory, in megabytes.
    pub mem_used_mb: f64,
    /// Total physical memory, in megabytes.
    pub mem_total_mb: f64,
    /// Used physical memory, in percent (0-100).
    pub mem_used_pct: f32,
    /// Used swap, in megabytes.
    pub swap_used_mb: f64,
    /// Total swap, in megabytes.
    pub swap_total_mb: f64,
    /// Resident memory of the current process, in megabytes.
    pub process_rss_mb: f64,
    /// Human-readable GPU model name.
    pub gpu_name: String,
    /// GPU utilization, in percent (0-100). Meaningful only when
    /// [`SystemSnapshot::gpu_usage_available`] is `true`.
    pub gpu_usage: f32,
    /// Whether a real GPU utilization measurement was available.
    pub gpu_usage_available: bool,
    /// GPU/accelerator in-use memory, in megabytes.
    pub gpu_mem_used_mb: f64,
    /// Total GPU memory, in megabytes (unified system memory on Apple Silicon).
    pub gpu_mem_total_mb: f64,
    /// GPU renderer-pipeline utilization, in percent (Apple Silicon only).
    pub gpu_renderer_util: f32,
    /// GPU tiler-pipeline utilization, in percent (Apple Silicon only).
    pub gpu_tiler_util: f32,
    /// Whether ARM NEON SIMD acceleration is available in this build.
    pub neon: bool,
    /// Operating system identifier (e.g. "macos", "linux").
    pub os: String,
    /// CPU architecture identifier (e.g. "aarch64", "x86_64").
    pub arch: String,
}

/// Best-effort runtime GPU telemetry parsed from platform probes.
#[derive(Debug, Clone, Default)]
struct GpuRuntime {
    util: Option<f32>,
    renderer: Option<f32>,
    tiler: Option<f32>,
    mem_used_mb: Option<f64>,
}

/// Live system monitor that holds a reusable [`sysinfo::System`] handle.
///
/// Create one per polling thread and call [`SystemMonitor::refresh`] on an
/// interval (1 second is a good default; CPU deltas need a measurement window).
pub struct SystemMonitor {
    sys: System,
    pid: Option<Pid>,
    cpu_name: String,
    cpu_cores: usize,
    gpu_name: String,
    /// Whether GPU memory is unified with system RAM (Apple Silicon).
    apple_unified: bool,
    /// Cached discrete-GPU total memory in MB (non-Apple platforms).
    gpu_mem_total_mb: f64,
}

impl SystemMonitor {
    /// Create a new monitor and probe static hardware information.
    pub fn new() -> Self {
        let refresh_kind = RefreshKind::new()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything());
        let mut sys = System::new_with_specifics(refresh_kind);
        sys.refresh_cpu_all();
        sys.refresh_memory();

        let cpu_cores = sys.cpus().len();
        let cpu_name = sys
            .cpus()
            .first()
            .map(|c| c.brand().trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(detect_cpu_name);

        let gpu_name = detect_gpu_name();
        let apple_unified = cfg!(target_os = "macos");
        let gpu_mem_total_mb = if apple_unified {
            0.0 // filled from unified system memory on each refresh
        } else {
            detect_gpu_total_mb()
        };

        Self {
            sys,
            pid: get_current_pid().ok(),
            cpu_name,
            cpu_cores,
            gpu_name,
            apple_unified,
            gpu_mem_total_mb,
        }
    }

    /// Refresh and return the latest system snapshot.
    pub fn refresh(&mut self) -> SystemSnapshot {
        self.sys.refresh_cpu_usage();
        self.sys.refresh_memory();
        if let Some(pid) = self.pid {
            self.sys
                .refresh_processes(ProcessesToUpdate::Some(&[pid]), true);
        }

        let cpu_per_core: Vec<f32> = self.sys.cpus().iter().map(|c| c.cpu_usage()).collect();
        let cpu_usage = if cpu_per_core.is_empty() {
            0.0
        } else {
            cpu_per_core.iter().sum::<f32>() / cpu_per_core.len() as f32
        };

        let mem_used_mb = bytes_to_mb(self.sys.used_memory());
        let mem_total_mb = bytes_to_mb(self.sys.total_memory());
        let mem_used_pct = if mem_total_mb > 0.0 {
            (mem_used_mb / mem_total_mb * 100.0) as f32
        } else {
            0.0
        };

        let swap_used_mb = bytes_to_mb(self.sys.used_swap());
        let swap_total_mb = bytes_to_mb(self.sys.total_swap());

        let process_rss_mb = self
            .pid
            .and_then(|p| self.sys.process(p))
            .map(|p| bytes_to_mb(p.memory()))
            .unwrap_or(0.0);

        let la = System::load_average();
        let gpu = read_gpu_runtime();
        let gpu_mem_total_mb = if self.apple_unified {
            mem_total_mb
        } else {
            self.gpu_mem_total_mb
        };

        SystemSnapshot {
            cpu_usage,
            cpu_per_core,
            cpu_name: self.cpu_name.clone(),
            cpu_cores: self.cpu_cores,
            load_avg: [la.one, la.five, la.fifteen],
            mem_used_mb,
            mem_total_mb,
            mem_used_pct,
            swap_used_mb,
            swap_total_mb,
            process_rss_mb,
            gpu_name: self.gpu_name.clone(),
            gpu_usage: gpu.util.unwrap_or(0.0),
            gpu_usage_available: gpu.util.is_some(),
            gpu_mem_used_mb: gpu.mem_used_mb.unwrap_or(0.0),
            gpu_mem_total_mb,
            gpu_renderer_util: gpu.renderer.unwrap_or(0.0),
            gpu_tiler_util: gpu.tiler.unwrap_or(0.0),
            neon: crate::simd::is_neon_available(),
            os: std::env::consts::OS.to_string(),
            arch: std::env::consts::ARCH.to_string(),
        }
    }

    /// The detected CPU model name.
    pub fn cpu_name(&self) -> &str {
        &self.cpu_name
    }

    /// The detected GPU model name.
    pub fn gpu_name(&self) -> &str {
        &self.gpu_name
    }
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Convert a byte count to megabytes.
fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / 1024.0 / 1024.0
}

// ---------------------------------------------------------------------------
// macOS GPU probing (Apple Silicon).
// ---------------------------------------------------------------------------

/// Parse a numeric value following `key_with_eq` in `ioreg` output.
///
/// `ioreg` prints accelerator stats as `"<name>"=<value>` (no surrounding
/// spaces), so the key passed here should include the trailing `"=` to
/// disambiguate prefixes such as `"In use system memory (driver)"`.
#[cfg(target_os = "macos")]
fn ioreg_value(text: &str, key_with_eq: &str) -> Option<f64> {
    let idx = text.find(key_with_eq)?;
    let after = &text[idx + key_with_eq.len()..];
    let end = after
        .find(|c: char| !(c.is_ascii_digit() || c == '.' || c == '-'))
        .unwrap_or(after.len());
    if end == 0 {
        return None;
    }
    after[..end].parse::<f64>().ok()
}

/// Read live GPU utilization and memory from `ioreg` (no sudo required).
#[cfg(target_os = "macos")]
fn read_gpu_runtime() -> GpuRuntime {
    let mut rt = GpuRuntime::default();
    let output = Command::new("ioreg")
        .args(["-r", "-d", "1", "-w", "0", "-c", "IOAccelerator"])
        .output();

    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            rt.util = ioreg_value(&text, "\"Device Utilization %\"=").map(|v| v as f32);
            rt.renderer = ioreg_value(&text, "\"Renderer Utilization %\"=").map(|v| v as f32);
            rt.tiler = ioreg_value(&text, "\"Tiler Utilization %\"=").map(|v| v as f32);
            rt.mem_used_mb =
                ioreg_value(&text, "\"In use system memory\"=").map(|v| v / 1024.0 / 1024.0);
        }
    }
    rt
}

/// Detect the Apple GPU model name via `system_profiler`.
#[cfg(target_os = "macos")]
fn detect_gpu_name() -> String {
    let output = Command::new("system_profiler")
        .args(["SPDisplaysDataType"])
        .output();
    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            for line in text.lines() {
                if let Some(rest) = line.trim().strip_prefix("Chipset Model:") {
                    let name = rest.trim();
                    if !name.is_empty() {
                        return name.to_string();
                    }
                }
            }
        }
    }
    "Apple GPU".to_string()
}

/// Fallback CPU name via `sysctl` when `sysinfo` returns an empty brand.
#[cfg(target_os = "macos")]
fn detect_cpu_name() -> String {
    let output = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output();
    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            let name = text.trim();
            if !name.is_empty() {
                return name.to_string();
            }
        }
    }
    "Unknown CPU".to_string()
}

/// Apple Silicon uses unified memory, so discrete total is unused.
#[cfg(target_os = "macos")]
fn detect_gpu_total_mb() -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Non-macOS GPU probing (NVIDIA via nvidia-smi).
// ---------------------------------------------------------------------------

/// Read live GPU utilization and memory from `nvidia-smi`.
#[cfg(not(target_os = "macos"))]
fn read_gpu_runtime() -> GpuRuntime {
    let mut rt = GpuRuntime::default();
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            let line = text.lines().next().unwrap_or("");
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 2 {
                rt.util = parts[0].parse::<f32>().ok();
                rt.mem_used_mb = parts[1].parse::<f64>().ok();
            }
        }
    }
    rt
}

/// Detect the NVIDIA GPU model name via `nvidia-smi`.
#[cfg(not(target_os = "macos"))]
fn detect_gpu_name() -> String {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output();
    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            let name = text.lines().next().unwrap_or("").trim();
            if !name.is_empty() {
                return name.to_string();
            }
        }
    }
    "CPU (no GPU)".to_string()
}

/// CPU name fallback on non-macOS platforms.
#[cfg(not(target_os = "macos"))]
fn detect_cpu_name() -> String {
    "Unknown CPU".to_string()
}

/// Detect total discrete-GPU memory in MB via `nvidia-smi`.
#[cfg(not(target_os = "macos"))]
fn detect_gpu_total_mb() -> f64 {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        .output();
    if let Ok(output) = output {
        if let Ok(text) = String::from_utf8(output.stdout) {
            if let Some(v) = text.lines().next().and_then(|l| l.trim().parse::<f64>().ok()) {
                return v;
            }
        }
    }
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_has_real_cpu_and_mem() {
        let mut mon = SystemMonitor::new();
        let snap = mon.refresh();
        assert!(snap.cpu_cores >= 1);
        assert!(snap.mem_total_mb > 0.0);
        assert_eq!(snap.cpu_per_core.len(), snap.cpu_cores);
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn ioreg_parser_handles_real_format() {
        let sample = r#"{"Tiler Utilization %"=21,"Device Utilization %"=42,"In use system memory (driver)"=0,"In use system memory"=830930944}"#;
        assert_eq!(ioreg_value(sample, "\"Device Utilization %\"="), Some(42.0));
        assert_eq!(ioreg_value(sample, "\"Tiler Utilization %\"="), Some(21.0));
        // Must skip the "(driver)" variant and read the real in-use value.
        assert_eq!(
            ioreg_value(sample, "\"In use system memory\"="),
            Some(830930944.0)
        );
    }
}
