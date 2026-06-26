from typing import Optional, Sequence, Union

import numpy as np
import numpy.typing as npt

__version__: str

def version() -> str: ...
def metal_available() -> bool: ...
def cuda_available() -> bool: ...

class Device:
    @staticmethod
    def cpu() -> "Device": ...
    @staticmethod
    def metal() -> "Device": ...
    @staticmethod
    def cuda(ordinal: int) -> "Device": ...
    def is_gpu(self) -> bool: ...

class MarketData:
    def __init__(
        self,
        prices: npt.NDArray[np.float32],
        feature_names: Optional[Sequence[str]] = ...,
    ) -> None: ...
    @staticmethod
    def synthetic(timesteps: int, seed: int) -> "MarketData": ...
    def __len__(self) -> int: ...
    def num_features(self) -> int: ...

class TradingEnv:
    def __init__(
        self,
        data: MarketData,
        initial_balance: float = ...,
        transaction_cost: float = ...,
        max_position: float = ...,
        lookback: int = ...,
        episode_length: int = ...,
    ) -> None: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def act_dim(self) -> int: ...

class CartPole:
    """Native CartPole-v1 (discrete control). Use with PPO."""

    def __init__(self, seed: Optional[int] = ...) -> None: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def act_dim(self) -> int: ...

class Pendulum:
    """Native Pendulum-v1 (continuous control). Use with PPO or SAC."""

    def __init__(self, seed: Optional[int] = ...) -> None: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def act_dim(self) -> int: ...

class ArrayEnv:
    """Generic dataset env over an arbitrary [T, obs_dim] numpy matrix.

    reward_kind='regression' scores -MSE(action, targets_row) and requires
    `targets`; reward_kind='weighted' scores dot(action, returns_row) and
    requires `returns`.
    """

    def __init__(
        self,
        data: npt.NDArray[np.float32],
        reward_kind: str = ...,
        targets: Optional[npt.NDArray[np.float32]] = ...,
        returns: Optional[npt.NDArray[np.float32]] = ...,
        episode_len: Optional[int] = ...,
        random_start: bool = ...,
    ) -> None: ...
    @property
    def obs_dim(self) -> int: ...
    @property
    def act_dim(self) -> int: ...

# Any native environment accepted by the agents.
Env = Union[TradingEnv, CartPole, Pendulum, ArrayEnv]

class PPO:
    def __init__(
        self,
        env: Env,
        num_envs: int = ...,
        learning_rate: float = ...,
        n_steps: int = ...,
        batch_size: int = ...,
        n_epochs: int = ...,
        gamma: float = ...,
        hidden_sizes: Sequence[int] = ...,
        seed: Optional[int] = ...,
        device: Optional[Device] = ...,
    ) -> None: ...
    def learn(self, total_timesteps: int) -> None: ...
    def predict(
        self, observations: npt.NDArray[np.float32], deterministic: bool = ...
    ) -> npt.NDArray[np.float32]: ...
    def save(self, path: str) -> None: ...
    @property
    def act_dim(self) -> int: ...

class SAC:
    def __init__(
        self,
        env: Env,
        num_envs: int = ...,
        learning_rate: float = ...,
        batch_size: int = ...,
        buffer_size: int = ...,
        gamma: float = ...,
        seed: Optional[int] = ...,
        device: Optional[Device] = ...,
    ) -> None: ...
    def learn(self, total_timesteps: int) -> None: ...
    def predict(
        self, observations: npt.NDArray[np.float32], deterministic: bool = ...
    ) -> npt.NDArray[np.float32]: ...

class TradingMetrics:
    def __init__(self, rolling_window: int = ...) -> None: ...
    def add_returns(self, returns: npt.NDArray[np.float64]) -> None: ...
    def add_return(self, ret: float) -> None: ...
    def sharpe_ratio(self) -> float: ...
    def sortino_ratio(self) -> float: ...
    def calmar_ratio(self) -> float: ...
    def win_rate(self) -> float: ...
