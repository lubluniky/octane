//! Advanced trading environments for algorithmic trading RL.
//!
//! This module provides sophisticated trading environments with realistic
//! market microstructure simulation:
//!
//! - [`AdvancedTradingEnv`] - Full-featured trading environment with order book,
//!   slippage models, latency simulation, and partial fills
//! - [`MultiAssetEnv`] - Portfolio trading across multiple correlated assets
//! - [`MultiTimeframeEnv`] - Hierarchical observations across multiple timeframes
//! - [`RegimeDetector`] - Market regime detection (trend/range/volatile)
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::trading::{AdvancedTradingEnv, AdvancedTradingConfig, SlippageModel};
//! use octane_rs::envs::MarketData;
//!
//! let data = MarketData::synthetic(5000, 42);
//! let config = AdvancedTradingConfig::default()
//!     .slippage_model(SlippageModel::SquareRoot { impact_factor: 0.1 })
//!     .latency_ms(50)
//!     .enable_partial_fills(true);
//!
//! let env = AdvancedTradingEnv::with_config(data, config)?;
//! ```

mod env;
mod error;
mod multi_asset;
mod multi_timeframe;
mod regime;

pub use env::{
    AdvancedMarketData, AdvancedTradingConfig, AdvancedTradingEnv, CommissionModel, Order,
    OrderBook, OrderBookLevel, OrderSide, OrderStatus, OrderType, PositionType, SlippageModel,
};
pub use error::TradingError;
pub use multi_asset::{
    MultiAssetConfig, MultiAssetEnv, PortfolioAction, PortfolioMetrics, PortfolioState,
};
pub use multi_timeframe::{
    MultiTimeframeConfig, MultiTimeframeEnv, Timeframe, TimeframeData, TimeframeSynchronizer,
};
pub use regime::{
    GarchParams, HmmParams, MarketRegime, RegimeCallback, RegimeConfig, RegimeDetector,
    RegimeObservation, RegimeTransition,
};
