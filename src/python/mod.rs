//! Python bindings for octane-rs, built as an extension module with maturin/uv.
//!
//! The bindings wrap the **native** Rust trading environment and monomorphize
//! the agents over it (`PPOAgent<TradingEnv>`, `SACAgent<TradingEnv>`). This is
//! the architecture the binding surface review recommended: a Python-defined
//! environment cannot be vectorized safely (the `VecEnv` clones a template, and
//! a `Py<PyAny>` "clone" is a refcount bump, so N "parallel" envs would share
//! one Python object). Market data crosses the FFI boundary once via numpy at
//! construction; training runs entirely in Rust with the GIL released.
#![cfg(feature = "python")]
#![allow(clippy::useless_conversion)]

use candle_core::Tensor;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::algorithms::traits::RLAlgorithm;
use crate::algorithms::{PPOAgent, PPOConfig, SACAgent, SACConfig};
use crate::core::{Device, OctaneError};
use crate::envs::{Environment, MarketData, Space, TradingEnv, TradingEnvConfig, VecEnv};
use crate::metrics::{MetricsCalculator, MetricsConfig};

/// Convert the crate error type into the most appropriate Python exception.
impl From<OctaneError> for PyErr {
    fn from(err: OctaneError) -> PyErr {
        match err {
            OctaneError::InvalidConfig(_)
            | OctaneError::ShapeMismatch { .. }
            | OctaneError::Environment(_) => PyValueError::new_err(err.to_string()),
            OctaneError::Io(_) => PyIOError::new_err(err.to_string()),
            other => PyRuntimeError::new_err(other.to_string()),
        }
    }
}

/// Build a 2-D candle tensor from a contiguous numpy `f32` array.
fn tensor_from_numpy(arr: &PyReadonlyArray2<'_, f32>, device: &Device) -> PyResult<Tensor> {
    let view = arr.as_array();
    let shape = view.shape();
    let (rows, cols) = (shape[0], shape[1]);
    let slice = arr
        .as_slice()
        .map_err(|_| PyValueError::new_err("observation array must be C-contiguous"))?;
    let cd = device.to_candle().map_err(OctaneError::from)?;
    Ok(Tensor::from_slice(slice, &[rows, cols], &cd).map_err(OctaneError::from)?)
}

/// Convert a 2-D candle tensor back into a numpy array.
fn numpy_from_tensor<'py>(py: Python<'py>, t: &Tensor) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let t = t
        .to_dtype(candle_core::DType::F32)
        .map_err(OctaneError::from)?;
    let data: Vec<Vec<f32>> = t.to_vec2::<f32>().map_err(OctaneError::from)?;
    PyArray2::from_vec2(py, &data).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Compute device for tensor operations.
#[pyclass(name = "Device")]
#[derive(Clone)]
pub struct PyDevice {
    inner: Device,
}

#[pymethods]
impl PyDevice {
    /// CPU device (always available).
    #[staticmethod]
    fn cpu() -> Self {
        Self {
            inner: Device::cpu(),
        }
    }

    /// Apple Metal device. Errors if the crate was built without the `metal` feature.
    #[staticmethod]
    fn metal() -> PyResult<Self> {
        #[cfg(feature = "metal")]
        {
            Ok(Self {
                inner: Device::m4_metal(),
            })
        }
        #[cfg(not(feature = "metal"))]
        {
            Err(PyRuntimeError::new_err(
                "octane-rs was built without the `metal` feature",
            ))
        }
    }

    /// CUDA device. Errors if the crate was built without the `cuda` feature.
    #[staticmethod]
    fn cuda(ordinal: usize) -> PyResult<Self> {
        #[cfg(feature = "cuda")]
        {
            Ok(Self {
                inner: Device::cuda(ordinal),
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = ordinal;
            Err(PyRuntimeError::new_err(
                "octane-rs was built without the `cuda` feature",
            ))
        }
    }

    fn is_gpu(&self) -> bool {
        self.inner.is_gpu()
    }

    fn __repr__(&self) -> String {
        format!("Device('{}')", self.inner)
    }
}

impl Default for PyDevice {
    fn default() -> Self {
        Self::cpu()
    }
}

/// OHLCV(+features) market data, constructed once from a numpy array.
#[pyclass(name = "MarketData")]
#[derive(Clone)]
pub struct PyMarketData {
    inner: MarketData,
}

#[pymethods]
impl PyMarketData {
    /// Build market data from a `[timesteps, features]` numpy array.
    ///
    /// `feature_names` is optional and only used for debugging.
    #[new]
    #[pyo3(signature = (prices, feature_names=None))]
    fn new(
        prices: PyReadonlyArray2<'_, f32>,
        feature_names: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let arr = prices.as_array();
        let prices: Vec<Vec<f32>> = arr.outer_iter().map(|row| row.to_vec()).collect();
        if prices.is_empty() {
            return Err(PyValueError::new_err("market data is empty"));
        }
        let n_features = prices[0].len();
        let feature_names = feature_names
            .unwrap_or_else(|| (0..n_features).map(|i| format!("f{i}")).collect::<Vec<_>>());
        Ok(Self {
            inner: MarketData {
                prices,
                feature_names,
            },
        })
    }

    /// Synthetic random-walk market data for quick experiments.
    #[staticmethod]
    fn synthetic(timesteps: usize, seed: u64) -> Self {
        Self {
            inner: MarketData::synthetic(timesteps, seed),
        }
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn num_features(&self) -> usize {
        self.inner.num_features()
    }
}

/// Native trading environment over a fixed market-data series.
#[pyclass(name = "TradingEnv")]
#[derive(Clone)]
pub struct PyTradingEnv {
    inner: TradingEnv,
    obs_dim: usize,
    act_dim: usize,
}

#[pymethods]
impl PyTradingEnv {
    #[new]
    #[pyo3(signature = (
        data,
        initial_balance=10000.0,
        transaction_cost=0.001,
        max_position=1.0,
        lookback=20,
        episode_length=252,
    ))]
    fn new(
        data: &PyMarketData,
        initial_balance: f32,
        transaction_cost: f32,
        max_position: f32,
        lookback: usize,
        episode_length: usize,
    ) -> PyResult<Self> {
        let config = TradingEnvConfig {
            initial_balance,
            transaction_cost,
            max_position,
            lookback,
            episode_length,
        };
        let env = TradingEnv::with_config(data.inner.clone(), config)?;
        let obs_dim = env.observation_space().flat_dim();
        let act_dim = env.action_space().flat_dim();
        Ok(Self {
            inner: env,
            obs_dim,
            act_dim,
        })
    }

    /// Observation dimension.
    #[getter]
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Action dimension.
    #[getter]
    fn act_dim(&self) -> usize {
        self.act_dim
    }
}

/// Proximal Policy Optimization agent over the native trading environment.
#[pyclass(name = "PPO")]
pub struct PyPPO {
    agent: PPOAgent<TradingEnv>,
    device: Device,
    obs_dim: usize,
    act_dim: usize,
}

#[pymethods]
impl PyPPO {
    #[new]
    #[pyo3(signature = (
        env,
        num_envs=8,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        hidden_sizes=vec![256, 256],
        seed=None,
        device=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &PyTradingEnv,
        num_envs: usize,
        learning_rate: f32,
        n_steps: usize,
        batch_size: usize,
        n_epochs: usize,
        gamma: f32,
        hidden_sizes: Vec<usize>,
        seed: Option<u64>,
        device: Option<PyDevice>,
    ) -> PyResult<Self> {
        let device = device.unwrap_or_default().inner;
        let mut config = PPOConfig::default()
            .learning_rate(learning_rate)
            .n_steps(n_steps)
            .batch_size(batch_size)
            .n_epochs(n_epochs)
            .gamma(gamma)
            .hidden_sizes(hidden_sizes);
        if let Some(s) = seed {
            config = config.seed(s);
        }
        let vec_env = VecEnv::new(vec![env.inner.clone()], num_envs);
        let agent = PPOAgent::new(config, vec_env, device)?;
        Ok(Self {
            agent,
            device,
            obs_dim: env.obs_dim,
            act_dim: env.act_dim,
        })
    }

    /// Train for `total_timesteps`. The GIL is released for the whole run.
    fn learn(&mut self, py: Python<'_>, total_timesteps: usize) -> PyResult<()> {
        py.allow_threads(|| self.agent.train(total_timesteps, |_| {}))?;
        Ok(())
    }

    /// Predict actions for a batch of observations `[batch, obs_dim]`.
    #[pyo3(signature = (observations, deterministic=true))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        observations: PyReadonlyArray2<'py, f32>,
        deterministic: bool,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if observations.as_array().shape()[1] != self.obs_dim {
            return Err(PyValueError::new_err(format!(
                "expected observations with {} features, got {}",
                self.obs_dim,
                observations.as_array().shape()[1]
            )));
        }
        let obs = tensor_from_numpy(&observations, &self.device)?;
        let actions = self.agent.predict(&obs, deterministic)?;
        numpy_from_tensor(py, &actions)
    }

    /// Save the policy to a safetensors file.
    fn save(&self, path: &str) -> PyResult<()> {
        self.agent.save(std::path::Path::new(path))?;
        Ok(())
    }

    #[getter]
    fn act_dim(&self) -> usize {
        self.act_dim
    }
}

/// Soft Actor-Critic agent over the native trading environment.
#[pyclass(name = "SAC")]
pub struct PySAC {
    agent: SACAgent<TradingEnv>,
    device: Device,
    obs_dim: usize,
}

#[pymethods]
impl PySAC {
    #[new]
    #[pyo3(signature = (
        env,
        num_envs=1,
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=1_000_000,
        gamma=0.99,
        seed=None,
        device=None,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        env: &PyTradingEnv,
        num_envs: usize,
        learning_rate: f32,
        batch_size: usize,
        buffer_size: usize,
        gamma: f32,
        seed: Option<u64>,
        device: Option<PyDevice>,
    ) -> PyResult<Self> {
        let device = device.unwrap_or_default().inner;
        let mut config = SACConfig::default()
            .learning_rate(learning_rate)
            .batch_size(batch_size)
            .buffer_size(buffer_size)
            .gamma(gamma);
        if let Some(s) = seed {
            config = config.seed(s);
        }
        let vec_env = VecEnv::new(vec![env.inner.clone()], num_envs);
        let agent = SACAgent::new(config, vec_env, device)?;
        Ok(Self {
            agent,
            device,
            obs_dim: env.obs_dim,
        })
    }

    /// Train for `total_timesteps`, with the GIL released.
    fn learn(&mut self, py: Python<'_>, total_timesteps: usize) -> PyResult<()> {
        py.allow_threads(|| self.agent.train(total_timesteps, |_| {}))?;
        Ok(())
    }

    /// Predict actions for a batch of observations `[batch, obs_dim]`.
    #[pyo3(signature = (observations, deterministic=true))]
    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        observations: PyReadonlyArray2<'py, f32>,
        deterministic: bool,
    ) -> PyResult<Bound<'py, PyArray2<f32>>> {
        if observations.as_array().shape()[1] != self.obs_dim {
            return Err(PyValueError::new_err(format!(
                "expected observations with {} features, got {}",
                self.obs_dim,
                observations.as_array().shape()[1]
            )));
        }
        let obs = tensor_from_numpy(&observations, &self.device)?;
        let actions = self.agent.predict(&obs, deterministic)?;
        numpy_from_tensor(py, &actions)
    }
}

/// Streaming trading-performance metrics (wraps the native calculator).
#[pyclass(name = "TradingMetrics")]
pub struct PyTradingMetrics {
    inner: MetricsCalculator,
}

#[pymethods]
impl PyTradingMetrics {
    #[new]
    #[pyo3(signature = (rolling_window=252))]
    fn new(rolling_window: usize) -> Self {
        let config = MetricsConfig {
            rolling_window,
            ..Default::default()
        };
        Self {
            inner: MetricsCalculator::new(config),
        }
    }

    /// Feed all returns from a 1-D numpy array, then read metrics.
    fn add_returns(&mut self, returns: PyReadonlyArray1<'_, f64>) -> PyResult<()> {
        for &r in returns
            .as_slice()
            .map_err(|_| PyValueError::new_err("returns array must be C-contiguous"))?
        {
            self.inner.add_return(r);
        }
        Ok(())
    }

    fn add_return(&mut self, ret: f64) {
        self.inner.add_return(ret);
    }

    fn sharpe_ratio(&self) -> f64 {
        self.inner.sharpe_ratio()
    }

    fn sortino_ratio(&self) -> f64 {
        self.inner.sortino_ratio()
    }

    fn calmar_ratio(&self) -> f64 {
        self.inner.calmar_ratio()
    }

    fn win_rate(&self) -> f64 {
        self.inner.win_rate()
    }
}

/// Crate version string.
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// Whether the extension was built with Metal support.
#[pyfunction]
fn metal_available() -> bool {
    cfg!(feature = "metal")
}

/// Whether the extension was built with CUDA support.
#[pyfunction]
fn cuda_available() -> bool {
    cfg!(feature = "cuda")
}

/// The compiled extension module (`octane.octane_rs`).
#[pymodule]
fn octane_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_class::<PyDevice>()?;
    m.add_class::<PyMarketData>()?;
    m.add_class::<PyTradingEnv>()?;
    m.add_class::<PyPPO>()?;
    m.add_class::<PySAC>()?;
    m.add_class::<PyTradingMetrics>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(metal_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    Ok(())
}
