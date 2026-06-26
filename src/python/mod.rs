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
use crate::envs::{
    ArrayEnv, ArrayReward, CartPole, Environment, MarketData, Pendulum, Space, TradingEnv,
    TradingEnvConfig, VecEnv,
};
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
#[pyclass(name = "Device", from_py_object)]
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
#[pyclass(name = "MarketData", from_py_object)]
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
#[pyclass(name = "TradingEnv", from_py_object)]
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

/// Native `CartPole-v1` — the canonical discrete control benchmark.
#[pyclass(name = "CartPole", from_py_object)]
#[derive(Clone)]
pub struct PyCartPole {
    inner: CartPole,
    obs_dim: usize,
    act_dim: usize,
}

#[pymethods]
impl PyCartPole {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        let inner = match seed {
            Some(s) => CartPole::seeded(s),
            None => CartPole::new(),
        };
        let obs_dim = inner.observation_space().flat_dim();
        let act_dim = inner.action_space().flat_dim();
        Self {
            inner,
            obs_dim,
            act_dim,
        }
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    #[getter]
    fn act_dim(&self) -> usize {
        self.act_dim
    }
}

/// Native `Pendulum-v1` — the canonical continuous control benchmark.
#[pyclass(name = "Pendulum", from_py_object)]
#[derive(Clone)]
pub struct PyPendulum {
    inner: Pendulum,
    obs_dim: usize,
    act_dim: usize,
}

#[pymethods]
impl PyPendulum {
    #[new]
    #[pyo3(signature = (seed=None))]
    fn new(seed: Option<u64>) -> Self {
        let inner = match seed {
            Some(s) => Pendulum::seeded(s),
            None => Pendulum::new(),
        };
        let obs_dim = inner.observation_space().flat_dim();
        let act_dim = inner.action_space().flat_dim();
        Self {
            inner,
            obs_dim,
            act_dim,
        }
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    #[getter]
    fn act_dim(&self) -> usize {
        self.act_dim
    }
}

/// Generic dataset environment over an arbitrary `[T, obs_dim]` numpy matrix.
///
/// `reward_kind="regression"` scores `-MSE(action, targets_row)`;
/// `reward_kind="weighted"` scores `dot(action, returns_row)`.
#[pyclass(name = "ArrayEnv", from_py_object)]
#[derive(Clone)]
pub struct PyArrayEnv {
    inner: ArrayEnv,
    obs_dim: usize,
    act_dim: usize,
}

#[pymethods]
impl PyArrayEnv {
    #[new]
    #[pyo3(signature = (
        data,
        reward_kind="regression",
        targets=None,
        returns=None,
        episode_len=None,
        random_start=false,
    ))]
    fn new(
        data: PyReadonlyArray2<'_, f32>,
        reward_kind: &str,
        targets: Option<PyReadonlyArray2<'_, f32>>,
        returns: Option<PyReadonlyArray2<'_, f32>>,
        episode_len: Option<usize>,
        random_start: bool,
    ) -> PyResult<Self> {
        let obs_dim = data.as_array().shape()[1];
        let flat: Vec<f32> = data
            .as_slice()
            .map_err(|_| PyValueError::new_err("data array must be C-contiguous"))?
            .to_vec();

        let reward = match reward_kind {
            "regression" => {
                let t = targets.ok_or_else(|| {
                    PyValueError::new_err("reward_kind='regression' requires `targets`")
                })?;
                let target_dim = t.as_array().shape()[1];
                let tflat: Vec<f32> = t
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("targets array must be C-contiguous"))?
                    .to_vec();
                ArrayReward::Regression {
                    targets: tflat,
                    target_dim,
                }
            }
            "weighted" => {
                let r = returns.ok_or_else(|| {
                    PyValueError::new_err("reward_kind='weighted' requires `returns`")
                })?;
                let n_assets = r.as_array().shape()[1];
                let rflat: Vec<f32> = r
                    .as_slice()
                    .map_err(|_| PyValueError::new_err("returns array must be C-contiguous"))?
                    .to_vec();
                ArrayReward::Weighted {
                    returns: rflat,
                    n_assets,
                }
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown reward_kind '{other}', expected 'regression' or 'weighted'"
                )))
            }
        };

        let mut env = ArrayEnv::new(flat, obs_dim, reward)?;
        if let Some(l) = episode_len {
            env = env.with_episode_len(l);
        }
        env = env.with_random_start(random_start);
        let act_dim = env.act_dim();
        Ok(Self {
            inner: env,
            obs_dim,
            act_dim,
        })
    }

    #[getter]
    fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    #[getter]
    fn act_dim(&self) -> usize {
        self.act_dim
    }
}

/// Monomorphized PPO agents behind one Python `PPO` class. PPO handles both
/// discrete (CartPole) and continuous (Pendulum/ArrayEnv/Trading) action spaces.
enum PpoBackend {
    Trading(PPOAgent<TradingEnv>),
    CartPole(PPOAgent<CartPole>),
    Pendulum(PPOAgent<Pendulum>),
    Array(PPOAgent<ArrayEnv>),
}

impl PpoBackend {
    fn train(&mut self, n: usize) -> crate::core::Result<()> {
        match self {
            PpoBackend::Trading(a) => a.train(n, |_| {}),
            PpoBackend::CartPole(a) => a.train(n, |_| {}),
            PpoBackend::Pendulum(a) => a.train(n, |_| {}),
            PpoBackend::Array(a) => a.train(n, |_| {}),
        }
    }

    fn predict(&mut self, obs: &Tensor, deterministic: bool) -> crate::core::Result<Tensor> {
        match self {
            PpoBackend::Trading(a) => a.predict(obs, deterministic),
            PpoBackend::CartPole(a) => a.predict(obs, deterministic),
            PpoBackend::Pendulum(a) => a.predict(obs, deterministic),
            PpoBackend::Array(a) => a.predict(obs, deterministic),
        }
    }

    fn save(&self, path: &std::path::Path) -> crate::core::Result<()> {
        match self {
            PpoBackend::Trading(a) => a.save(path),
            PpoBackend::CartPole(a) => a.save(path),
            PpoBackend::Pendulum(a) => a.save(path),
            PpoBackend::Array(a) => a.save(path),
        }
    }
}

/// Proximal Policy Optimization. Accepts any native env: `TradingEnv`,
/// `CartPole`, `Pendulum`, or `ArrayEnv`.
#[pyclass(name = "PPO")]
pub struct PyPPO {
    backend: PpoBackend,
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
        env: &Bound<'_, PyAny>,
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

        // Dispatch on the concrete env type so one `PPO` class drives every
        // monomorphization. Each `extract` clones the wrapped native env out.
        if let Ok(e) = env.extract::<PyTradingEnv>() {
            let agent =
                PPOAgent::new(config, VecEnv::new(vec![e.inner.clone()], num_envs), device)?;
            Ok(Self {
                backend: PpoBackend::Trading(agent),
                device,
                obs_dim: e.obs_dim,
                act_dim: e.act_dim,
            })
        } else if let Ok(e) = env.extract::<PyCartPole>() {
            let agent =
                PPOAgent::new(config, VecEnv::new(vec![e.inner.clone()], num_envs), device)?;
            Ok(Self {
                backend: PpoBackend::CartPole(agent),
                device,
                obs_dim: e.obs_dim,
                act_dim: e.act_dim,
            })
        } else if let Ok(e) = env.extract::<PyPendulum>() {
            let agent =
                PPOAgent::new(config, VecEnv::new(vec![e.inner.clone()], num_envs), device)?;
            Ok(Self {
                backend: PpoBackend::Pendulum(agent),
                device,
                obs_dim: e.obs_dim,
                act_dim: e.act_dim,
            })
        } else if let Ok(e) = env.extract::<PyArrayEnv>() {
            let agent =
                PPOAgent::new(config, VecEnv::new(vec![e.inner.clone()], num_envs), device)?;
            Ok(Self {
                backend: PpoBackend::Array(agent),
                device,
                obs_dim: e.obs_dim,
                act_dim: e.act_dim,
            })
        } else {
            Err(PyValueError::new_err(
                "unsupported env: expected TradingEnv, CartPole, Pendulum, or ArrayEnv",
            ))
        }
    }

    /// Train for `total_timesteps`. The GIL is released for the whole run.
    fn learn(&mut self, py: Python<'_>, total_timesteps: usize) -> PyResult<()> {
        py.detach(|| self.backend.train(total_timesteps))?;
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
        let actions = self.backend.predict(&obs, deterministic)?;
        numpy_from_tensor(py, &actions)
    }

    /// Save the policy to a safetensors file.
    fn save(&self, path: &str) -> PyResult<()> {
        self.backend.save(std::path::Path::new(path))?;
        Ok(())
    }

    #[getter]
    fn act_dim(&self) -> usize {
        self.act_dim
    }
}

/// Monomorphized SAC agents behind one Python `SAC` class. SAC is
/// continuous-only, so a discrete env (CartPole) is rejected at construction.
enum SacBackend {
    Trading(SACAgent<TradingEnv>),
    Pendulum(SACAgent<Pendulum>),
    Array(SACAgent<ArrayEnv>),
}

impl SacBackend {
    fn train(&mut self, n: usize) -> crate::core::Result<()> {
        match self {
            SacBackend::Trading(a) => a.train(n, |_| {}),
            SacBackend::Pendulum(a) => a.train(n, |_| {}),
            SacBackend::Array(a) => a.train(n, |_| {}),
        }
    }

    fn predict(&self, obs: &Tensor, deterministic: bool) -> crate::core::Result<Tensor> {
        match self {
            SacBackend::Trading(a) => a.predict(obs, deterministic),
            SacBackend::Pendulum(a) => a.predict(obs, deterministic),
            SacBackend::Array(a) => a.predict(obs, deterministic),
        }
    }
}

/// Soft Actor-Critic (continuous control). Accepts `TradingEnv`, `Pendulum`,
/// or `ArrayEnv`; a discrete env raises an error.
#[pyclass(name = "SAC")]
pub struct PySAC {
    backend: SacBackend,
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
        env: &Bound<'_, PyAny>,
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

        if let Ok(e) = env.extract::<PyTradingEnv>() {
            let agent =
                SACAgent::new(config, VecEnv::new(vec![e.inner.clone()], num_envs), device)?;
            Ok(Self {
                backend: SacBackend::Trading(agent),
                device,
                obs_dim: e.obs_dim,
            })
        } else if let Ok(e) = env.extract::<PyPendulum>() {
            let agent =
                SACAgent::new(config, VecEnv::new(vec![e.inner.clone()], num_envs), device)?;
            Ok(Self {
                backend: SacBackend::Pendulum(agent),
                device,
                obs_dim: e.obs_dim,
            })
        } else if let Ok(e) = env.extract::<PyArrayEnv>() {
            let agent =
                SACAgent::new(config, VecEnv::new(vec![e.inner.clone()], num_envs), device)?;
            Ok(Self {
                backend: SacBackend::Array(agent),
                device,
                obs_dim: e.obs_dim,
            })
        } else if env.extract::<PyCartPole>().is_ok() {
            Err(PyValueError::new_err(
                "SAC is continuous-only; CartPole is discrete. Use PPO for CartPole.",
            ))
        } else {
            Err(PyValueError::new_err(
                "unsupported env: expected TradingEnv, Pendulum, or ArrayEnv",
            ))
        }
    }

    /// Train for `total_timesteps`, with the GIL released.
    fn learn(&mut self, py: Python<'_>, total_timesteps: usize) -> PyResult<()> {
        py.detach(|| self.backend.train(total_timesteps))?;
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
        let actions = self.backend.predict(&obs, deterministic)?;
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
    m.add_class::<PyCartPole>()?;
    m.add_class::<PyPendulum>()?;
    m.add_class::<PyArrayEnv>()?;
    m.add_class::<PyPPO>()?;
    m.add_class::<PySAC>()?;
    m.add_class::<PyTradingMetrics>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(metal_available, m)?)?;
    m.add_function(wrap_pyfunction!(cuda_available, m)?)?;
    Ok(())
}
