"""Octane — high-performance reinforcement learning for trading.

Rust-powered RL agents with a thin Python API. Agents are monomorphized over the
native Rust trading environment, so training runs entirely in Rust (with the GIL
released) and only market data / observations cross the boundary as numpy arrays.

Example
-------
>>> import numpy as np
>>> import octane
>>> data = octane.MarketData.synthetic(timesteps=2000, seed=0)
>>> env = octane.TradingEnv(data, lookback=20, episode_length=252)
>>> agent = octane.PPO(env, num_envs=16, n_steps=512, device=octane.Device.cpu())
>>> agent.learn(total_timesteps=50_000)          # runs in Rust, GIL released
>>> obs = np.zeros((4, env.obs_dim), dtype=np.float32)
>>> actions = agent.predict(obs)                 # -> np.ndarray [4, act_dim]
"""

from .octane_rs import (  # type: ignore[attr-defined]
    __version__,
    ArrayEnv,
    CartPole,
    Device,
    MarketData,
    PPO,
    Pendulum,
    SAC,
    TradingEnv,
    TradingMetrics,
    cuda_available,
    metal_available,
    version,
)

__all__ = [
    "__version__",
    "Device",
    "MarketData",
    "TradingEnv",
    "CartPole",
    "Pendulum",
    "ArrayEnv",
    "PPO",
    "SAC",
    "TradingMetrics",
    "version",
    "metal_available",
    "cuda_available",
]
