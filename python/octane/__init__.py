# Octane: High-Performance Reinforcement Learning for Trading
# Python bindings for the Rust octane-rs library

from .octane_rs import *

__version__ = "0.4.0"
__all__ = [
    # Core
    "Device",
    # Trading Environments
    "TradingEnv",
    "MultiAssetEnv",
    "MultiTimeframeEnv",
    "RegimeDetector",
    # Risk Management
    "DrawdownController",
    "PositionSizer",
    "ConstraintManager",
    # Metrics
    "TradingMetrics",
    "TradeJournal",
    # Backtesting
    "WalkForwardOptimizer",
    "MonteCarloSimulator",
    "CrossValidator",
    # Algorithms
    "PPOAgent",
    "SACAgent",
    "TD3Agent",
    "DQNAgent",
]
