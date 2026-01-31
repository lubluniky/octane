//! Python bindings for Octane using PyO3
//!
//! This module provides Python bindings for the core Octane functionality,
//! enabling use from Python while maintaining Rust performance.

use pyo3::prelude::*;

/// Device abstraction for CPU/GPU selection
#[pyclass]
#[derive(Clone)]
pub struct PyDevice {
    inner: String,
}

#[pymethods]
impl PyDevice {
    #[new]
    #[pyo3(signature = (device_type="cpu"))]
    fn new(device_type: &str) -> PyResult<Self> {
        Ok(Self {
            inner: device_type.to_string(),
        })
    }

    fn __repr__(&self) -> String {
        format!("Device('{}')", self.inner)
    }

    #[staticmethod]
    fn cpu() -> Self {
        Self {
            inner: "cpu".to_string(),
        }
    }

    #[staticmethod]
    fn metal() -> Self {
        Self {
            inner: "metal".to_string(),
        }
    }

    #[staticmethod]
    fn cuda(device_id: usize) -> Self {
        Self {
            inner: format!("cuda:{}", device_id),
        }
    }
}

/// Trading metrics calculator
#[pyclass]
pub struct PyTradingMetrics {
    returns: Vec<f64>,
    window_size: usize,
}

#[pymethods]
impl PyTradingMetrics {
    #[new]
    #[pyo3(signature = (window_size=252))]
    fn new(window_size: usize) -> Self {
        Self {
            returns: Vec::new(),
            window_size,
        }
    }

    /// Add a return observation
    fn add_return(&mut self, ret: f64) {
        self.returns.push(ret);
        if self.returns.len() > self.window_size {
            self.returns.remove(0);
        }
    }

    /// Calculate Sharpe ratio (annualized)
    fn sharpe_ratio(&self, risk_free_rate: Option<f64>) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        let rf = risk_free_rate.unwrap_or(0.0);
        let mean: f64 = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let excess = mean - rf / 252.0;
        let variance: f64 = self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>()
            / self.returns.len() as f64;
        let std = variance.sqrt();
        if std == 0.0 {
            return 0.0;
        }
        (excess / std) * (252.0_f64).sqrt()
    }

    /// Calculate maximum drawdown
    fn max_drawdown(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        let mut equity = 1.0;
        let mut peak = 1.0;
        let mut max_dd = 0.0;

        for ret in &self.returns {
            equity *= 1.0 + ret;
            peak = peak.max(equity);
            let dd = (peak - equity) / peak;
            max_dd = max_dd.max(dd);
        }
        max_dd
    }

    /// Calculate win rate
    fn win_rate(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        let wins = self.returns.iter().filter(|&&r| r > 0.0).count();
        wins as f64 / self.returns.len() as f64
    }

    /// Calculate Value at Risk (historical)
    #[pyo3(signature = (confidence=0.95))]
    fn var(&self, confidence: f64) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        let mut sorted = self.returns.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((1.0 - confidence) * sorted.len() as f64).floor() as usize;
        -sorted[index.min(sorted.len() - 1)]
    }
}

/// Drawdown controller for risk management
#[pyclass]
pub struct PyDrawdownController {
    max_drawdown: f64,
    recovery_threshold: f64,
    recovery_risk_factor: f64,
    peak_equity: f64,
    current_equity: f64,
    in_recovery: bool,
}

#[pymethods]
impl PyDrawdownController {
    #[new]
    #[pyo3(signature = (max_drawdown=0.2, recovery_threshold=0.1, recovery_risk_factor=0.5))]
    fn new(max_drawdown: f64, recovery_threshold: f64, recovery_risk_factor: f64) -> Self {
        Self {
            max_drawdown,
            recovery_threshold,
            recovery_risk_factor,
            peak_equity: 1.0,
            current_equity: 1.0,
            in_recovery: false,
        }
    }

    /// Update equity and check drawdown limits
    fn update(&mut self, equity: f64) -> PyResult<bool> {
        self.current_equity = equity;
        self.peak_equity = self.peak_equity.max(equity);

        let dd = self.current_drawdown();

        // Check if we should enter recovery mode
        if dd >= self.recovery_threshold && !self.in_recovery {
            self.in_recovery = true;
        }

        // Check if we exceeded max drawdown (should stop)
        if dd >= self.max_drawdown {
            return Ok(true); // Should stop trading
        }

        // Check if we recovered
        if self.in_recovery && dd < self.recovery_threshold * 0.5 {
            self.in_recovery = false;
        }

        Ok(false)
    }

    /// Get current drawdown
    fn current_drawdown(&self) -> f64 {
        if self.peak_equity == 0.0 {
            return 0.0;
        }
        (self.peak_equity - self.current_equity) / self.peak_equity
    }

    /// Get position scale factor based on current state
    fn position_scale(&self) -> f64 {
        if self.in_recovery {
            self.recovery_risk_factor
        } else {
            1.0
        }
    }

    /// Check if in recovery mode
    fn is_recovering(&self) -> bool {
        self.in_recovery
    }

    /// Reset controller state
    fn reset(&mut self) {
        self.peak_equity = 1.0;
        self.current_equity = 1.0;
        self.in_recovery = false;
    }
}

/// Position sizer using Kelly criterion
#[pyclass]
pub struct PyPositionSizer {
    method: String,
    kelly_fraction: f64,
    max_position: f64,
}

#[pymethods]
impl PyPositionSizer {
    #[new]
    #[pyo3(signature = (method="half_kelly", max_position=1.0))]
    fn new(method: &str, max_position: f64) -> Self {
        let kelly_fraction = match method {
            "full_kelly" => 1.0,
            "half_kelly" => 0.5,
            "quarter_kelly" => 0.25,
            _ => 0.5,
        };
        Self {
            method: method.to_string(),
            kelly_fraction,
            max_position,
        }
    }

    /// Calculate position size based on win rate and win/loss ratio
    fn calculate(&self, win_rate: f64, avg_win: f64, avg_loss: f64) -> f64 {
        if avg_loss == 0.0 || win_rate <= 0.0 || win_rate >= 1.0 {
            return 0.0;
        }

        let win_loss_ratio = avg_win / avg_loss.abs();
        let kelly = win_rate - (1.0 - win_rate) / win_loss_ratio;

        let position = kelly * self.kelly_fraction;
        position.max(0.0).min(self.max_position)
    }

    fn __repr__(&self) -> String {
        format!(
            "PositionSizer(method='{}', kelly_fraction={}, max_position={})",
            self.method, self.kelly_fraction, self.max_position
        )
    }
}

/// Octane Python module
#[pymodule]
fn octane_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    m.add_class::<PyTradingMetrics>()?;
    m.add_class::<PyDrawdownController>()?;
    m.add_class::<PyPositionSizer>()?;

    // Add version
    m.add("__version__", "0.4.0")?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device() {
        let cpu = PyDevice::cpu();
        assert_eq!(cpu.inner, "cpu");

        let metal = PyDevice::metal();
        assert_eq!(metal.inner, "metal");

        let cuda = PyDevice::cuda(0);
        assert_eq!(cuda.inner, "cuda:0");
    }

    #[test]
    fn test_trading_metrics() {
        let mut metrics = PyTradingMetrics::new(252);
        for i in 0..100 {
            let ret = if i % 3 == 0 { -0.01 } else { 0.02 };
            metrics.add_return(ret);
        }

        let sharpe = metrics.sharpe_ratio(None);
        assert!(sharpe > 0.0);

        let win_rate = metrics.win_rate();
        assert!((win_rate - 0.67).abs() < 0.1);
    }

    #[test]
    fn test_drawdown_controller() {
        let mut dd = PyDrawdownController::new(0.2, 0.1, 0.5);

        // Simulate drawdown
        assert!(!dd.update(0.95).unwrap()); // 5% DD
        assert!(!dd.is_recovering());

        assert!(!dd.update(0.88).unwrap()); // 12% DD, should enter recovery
        assert!(dd.is_recovering());
        assert_eq!(dd.position_scale(), 0.5);

        assert!(dd.update(0.75).unwrap()); // 25% DD, exceeds max
    }

    #[test]
    fn test_position_sizer() {
        let sizer = PyPositionSizer::new("half_kelly", 1.0);
        let position = sizer.calculate(0.6, 0.02, 0.01);
        assert!(position > 0.0);
        assert!(position <= 1.0);
    }
}
