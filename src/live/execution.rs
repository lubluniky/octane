//! Execution engine for smart order routing and execution algorithms.
//!
//! Provides sophisticated order execution capabilities:
//! - Smart order routing
//! - Execution algorithms (TWAP, VWAP, Iceberg)
//! - Order management system (OMS)
//! - Fill tracking and reconciliation
//! - Execution quality metrics
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::live::execution::{ExecutionEngine, ExecutionConfig, ExecutionAlgorithm};
//! use octane_rs::live::types::{Side, Order};
//!
//! let config = ExecutionConfig::default()
//!     .max_slippage_bps(10.0)
//!     .default_algorithm(ExecutionAlgorithm::TWAP);
//!
//! let engine = ExecutionEngine::new(config);
//!
//! // Execute a large order using TWAP
//! let params = TWAPParams::new(Duration::from_secs(300), 10);
//! engine.execute_twap("BTCUSDT", Side::Buy, 1.0, params).await?;
//! ```

use crate::live::error::{LiveTradingError, Result};
use crate::live::exchanges::ExchangeConnector;
use crate::live::types::{current_timestamp_ms, generate_client_order_id, Order, Side};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, RwLock};

/// Execution algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionAlgorithm {
    /// Direct market order.
    Market,
    /// Direct limit order.
    Limit,
    /// Time-Weighted Average Price.
    TWAP,
    /// Volume-Weighted Average Price.
    VWAP,
    /// Iceberg (hidden size) orders.
    Iceberg,
    /// Implementation shortfall (adaptive).
    IS,
    /// Aggressive execution (take liquidity).
    Aggressive,
    /// Passive execution (provide liquidity).
    Passive,
    /// Percentage of Volume.
    POV,
}

impl Default for ExecutionAlgorithm {
    fn default() -> Self {
        ExecutionAlgorithm::Market
    }
}

/// Urgency level for execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Urgency {
    /// Low urgency - prioritize price over speed.
    Low,
    /// Medium urgency - balanced approach.
    Medium,
    /// High urgency - prioritize speed over price.
    High,
    /// Critical - execute immediately at any price.
    Critical,
}

impl Default for Urgency {
    fn default() -> Self {
        Urgency::Medium
    }
}

/// Configuration for execution engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Maximum slippage tolerance in basis points.
    pub max_slippage_bps: f64,
    /// Default execution algorithm.
    pub default_algorithm: ExecutionAlgorithm,
    /// Default urgency level.
    pub default_urgency: Urgency,
    /// Order retry count on failure.
    pub max_retries: u32,
    /// Retry delay in milliseconds.
    pub retry_delay_ms: u64,
    /// Enable smart order routing.
    pub smart_routing: bool,
    /// Minimum order size.
    pub min_order_size: f64,
    /// Maximum order size per slice.
    pub max_slice_size: f64,
    /// Enable order logging.
    pub enable_logging: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            max_slippage_bps: 10.0,
            default_algorithm: ExecutionAlgorithm::Market,
            default_urgency: Urgency::Medium,
            max_retries: 3,
            retry_delay_ms: 1000,
            smart_routing: true,
            min_order_size: 0.0001,
            max_slice_size: 10.0,
            enable_logging: true,
        }
    }
}

impl ExecutionConfig {
    /// Create new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set maximum slippage.
    pub fn max_slippage_bps(mut self, bps: f64) -> Self {
        self.max_slippage_bps = bps;
        self
    }

    /// Set default algorithm.
    pub fn default_algorithm(mut self, algo: ExecutionAlgorithm) -> Self {
        self.default_algorithm = algo;
        self
    }

    /// Set default urgency.
    pub fn default_urgency(mut self, urgency: Urgency) -> Self {
        self.default_urgency = urgency;
        self
    }

    /// Set max retries.
    pub fn max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }

    /// Enable smart routing.
    pub fn smart_routing(mut self, enabled: bool) -> Self {
        self.smart_routing = enabled;
        self
    }

    /// Set max slice size.
    pub fn max_slice_size(mut self, size: f64) -> Self {
        self.max_slice_size = size;
        self
    }
}

/// Parameters for TWAP execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TWAPParams {
    /// Total duration for execution.
    pub duration: Duration,
    /// Number of slices.
    pub num_slices: usize,
    /// Randomize slice timing.
    pub randomize: bool,
    /// Maximum price deviation to allow.
    pub max_deviation_bps: f64,
    /// Use limit orders instead of market.
    pub use_limit_orders: bool,
    /// Limit order offset from mid price (bps).
    pub limit_offset_bps: f64,
}

impl TWAPParams {
    /// Create new TWAP parameters.
    pub fn new(duration: Duration, num_slices: usize) -> Self {
        Self {
            duration,
            num_slices,
            randomize: true,
            max_deviation_bps: 50.0,
            use_limit_orders: false,
            limit_offset_bps: 0.0,
        }
    }

    /// Set randomization.
    pub fn randomize(mut self, enabled: bool) -> Self {
        self.randomize = enabled;
        self
    }

    /// Set max deviation.
    pub fn max_deviation_bps(mut self, bps: f64) -> Self {
        self.max_deviation_bps = bps;
        self
    }

    /// Use limit orders.
    pub fn use_limit_orders(mut self, enabled: bool, offset_bps: f64) -> Self {
        self.use_limit_orders = enabled;
        self.limit_offset_bps = offset_bps;
        self
    }
}

/// Parameters for VWAP execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VWAPParams {
    /// Total duration for execution.
    pub duration: Duration,
    /// Historical volume profile to match.
    pub volume_profile: Option<Vec<f64>>,
    /// Participation rate (0.0-1.0).
    pub participation_rate: f64,
    /// Maximum deviation from VWAP target.
    pub max_deviation_bps: f64,
}

impl VWAPParams {
    /// Create new VWAP parameters.
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            volume_profile: None,
            participation_rate: 0.1,
            max_deviation_bps: 50.0,
        }
    }

    /// Set volume profile.
    pub fn volume_profile(mut self, profile: Vec<f64>) -> Self {
        self.volume_profile = Some(profile);
        self
    }

    /// Set participation rate.
    pub fn participation_rate(mut self, rate: f64) -> Self {
        self.participation_rate = rate.clamp(0.0, 1.0);
        self
    }
}

/// Parameters for Iceberg orders.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IcebergParams {
    /// Visible (display) quantity.
    pub display_qty: f64,
    /// Price for limit orders.
    pub price: Option<f64>,
    /// Variance in display quantity (randomization).
    pub qty_variance: f64,
    /// Delay between refills in milliseconds.
    pub refill_delay_ms: u64,
}

impl IcebergParams {
    /// Create new Iceberg parameters.
    pub fn new(display_qty: f64) -> Self {
        Self {
            display_qty,
            price: None,
            qty_variance: 0.1,
            refill_delay_ms: 100,
        }
    }

    /// Set limit price.
    pub fn price(mut self, price: f64) -> Self {
        self.price = Some(price);
        self
    }

    /// Set quantity variance.
    pub fn qty_variance(mut self, variance: f64) -> Self {
        self.qty_variance = variance.clamp(0.0, 0.5);
        self
    }
}

/// Execution order request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequest {
    /// Request ID.
    pub request_id: String,
    /// Symbol.
    pub symbol: String,
    /// Side.
    pub side: Side,
    /// Total quantity to execute.
    pub quantity: f64,
    /// Execution algorithm.
    pub algorithm: ExecutionAlgorithm,
    /// Urgency level.
    pub urgency: Urgency,
    /// Limit price (optional).
    pub limit_price: Option<f64>,
    /// Maximum slippage tolerance.
    pub max_slippage_bps: Option<f64>,
    /// Request timestamp.
    pub created_at: u64,
}

impl ExecutionRequest {
    /// Create a new execution request.
    pub fn new(symbol: impl Into<String>, side: Side, quantity: f64) -> Self {
        Self {
            request_id: generate_client_order_id(),
            symbol: symbol.into(),
            side,
            quantity,
            algorithm: ExecutionAlgorithm::default(),
            urgency: Urgency::default(),
            limit_price: None,
            max_slippage_bps: None,
            created_at: current_timestamp_ms(),
        }
    }

    /// Set algorithm.
    pub fn algorithm(mut self, algo: ExecutionAlgorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Set urgency.
    pub fn urgency(mut self, urgency: Urgency) -> Self {
        self.urgency = urgency;
        self
    }

    /// Set limit price.
    pub fn limit_price(mut self, price: f64) -> Self {
        self.limit_price = Some(price);
        self
    }

    /// Set max slippage.
    pub fn max_slippage_bps(mut self, bps: f64) -> Self {
        self.max_slippage_bps = Some(bps);
        self
    }
}

/// Status of an execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionStatus {
    /// Execution is pending.
    Pending,
    /// Execution is in progress.
    InProgress,
    /// Execution is paused.
    Paused,
    /// Execution completed successfully.
    Completed,
    /// Execution was cancelled.
    Cancelled,
    /// Execution failed.
    Failed,
}

/// Execution result/status.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Request ID.
    pub request_id: String,
    /// Execution status.
    pub status: ExecutionStatus,
    /// Total quantity filled.
    pub filled_quantity: f64,
    /// Average fill price.
    pub average_price: f64,
    /// Number of child orders.
    pub num_orders: u32,
    /// Number of fills.
    pub num_fills: u32,
    /// Total commission paid.
    pub total_commission: f64,
    /// Execution start time.
    pub start_time: u64,
    /// Execution end time.
    pub end_time: Option<u64>,
    /// Slippage from arrival price (bps).
    pub slippage_bps: f64,
    /// Implementation shortfall (bps).
    pub implementation_shortfall_bps: f64,
    /// Child order IDs.
    pub child_orders: Vec<String>,
    /// Error message if failed.
    pub error: Option<String>,
}

impl ExecutionResult {
    /// Create a new pending result.
    fn new(request_id: impl Into<String>) -> Self {
        Self {
            request_id: request_id.into(),
            status: ExecutionStatus::Pending,
            filled_quantity: 0.0,
            average_price: 0.0,
            num_orders: 0,
            num_fills: 0,
            total_commission: 0.0,
            start_time: current_timestamp_ms(),
            end_time: None,
            slippage_bps: 0.0,
            implementation_shortfall_bps: 0.0,
            child_orders: Vec::new(),
            error: None,
        }
    }
}

/// Execution quality metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExecutionQualityMetrics {
    /// Total number of executions.
    pub total_executions: u64,
    /// Number of completed executions.
    pub completed_executions: u64,
    /// Number of failed executions.
    pub failed_executions: u64,
    /// Average slippage (bps).
    pub avg_slippage_bps: f64,
    /// Average implementation shortfall (bps).
    pub avg_is_bps: f64,
    /// Average fill rate (filled / requested).
    pub avg_fill_rate: f64,
    /// Average execution time (ms).
    pub avg_execution_time_ms: f64,
    /// Total volume executed.
    pub total_volume: f64,
    /// Total commission paid.
    pub total_commission: f64,
    /// VWAP performance (bps vs market VWAP).
    pub vwap_performance_bps: f64,
}

/// Order management system state.
#[derive(Debug, Default)]
struct OMSState {
    /// Active executions.
    active_executions: HashMap<String, ExecutionResult>,
    /// Completed executions.
    completed_executions: Vec<ExecutionResult>,
    /// Child orders by parent execution ID.
    child_orders: HashMap<String, Vec<Order>>,
    /// Quality metrics.
    metrics: ExecutionQualityMetrics,
}

/// Execution engine for smart order routing and algorithms.
pub struct ExecutionEngine {
    /// Configuration.
    config: ExecutionConfig,
    /// OMS state.
    state: Arc<RwLock<OMSState>>,
    /// Cancellation sender.
    cancel_tx: mpsc::Sender<String>,
    /// Cancellation receiver.
    cancel_rx: Option<mpsc::Receiver<String>>,
}

impl ExecutionEngine {
    /// Create a new execution engine.
    pub fn new(config: ExecutionConfig) -> Self {
        let (cancel_tx, cancel_rx) = mpsc::channel(100);
        Self {
            config,
            state: Arc::new(RwLock::new(OMSState::default())),
            cancel_tx,
            cancel_rx: Some(cancel_rx),
        }
    }

    /// Get current configuration.
    pub fn config(&self) -> &ExecutionConfig {
        &self.config
    }

    /// Get execution metrics.
    pub async fn metrics(&self) -> ExecutionQualityMetrics {
        self.state.read().await.metrics.clone()
    }

    /// Get active executions.
    pub async fn active_executions(&self) -> Vec<ExecutionResult> {
        self.state
            .read()
            .await
            .active_executions
            .values()
            .cloned()
            .collect()
    }

    /// Get execution result by ID.
    pub async fn get_execution(&self, request_id: &str) -> Option<ExecutionResult> {
        let state = self.state.read().await;
        state
            .active_executions
            .get(request_id)
            .cloned()
            .or_else(|| {
                state
                    .completed_executions
                    .iter()
                    .find(|e| e.request_id == request_id)
                    .cloned()
            })
    }

    /// Cancel an active execution.
    pub async fn cancel_execution(&self, request_id: &str) -> Result<()> {
        self.cancel_tx
            .send(request_id.to_string())
            .await
            .map_err(|e| LiveTradingError::Execution(e.to_string()))
    }

    /// Execute a market order.
    pub async fn execute_market<C: ExchangeConnector>(
        &self,
        connector: &C,
        symbol: &str,
        side: Side,
        quantity: f64,
    ) -> Result<ExecutionResult> {
        let request =
            ExecutionRequest::new(symbol, side, quantity).algorithm(ExecutionAlgorithm::Market);

        self.execute(connector, request).await
    }

    /// Execute a limit order.
    pub async fn execute_limit<C: ExchangeConnector>(
        &self,
        connector: &C,
        symbol: &str,
        side: Side,
        quantity: f64,
        price: f64,
    ) -> Result<ExecutionResult> {
        let request = ExecutionRequest::new(symbol, side, quantity)
            .algorithm(ExecutionAlgorithm::Limit)
            .limit_price(price);

        self.execute(connector, request).await
    }

    /// Execute using TWAP algorithm.
    pub async fn execute_twap<C: ExchangeConnector>(
        &self,
        connector: &C,
        symbol: &str,
        side: Side,
        quantity: f64,
        params: TWAPParams,
    ) -> Result<ExecutionResult> {
        let request_id = generate_client_order_id();
        let mut result = ExecutionResult::new(&request_id);
        result.status = ExecutionStatus::InProgress;

        // Store in active executions
        {
            let mut state = self.state.write().await;
            state
                .active_executions
                .insert(request_id.clone(), result.clone());
        }

        // Calculate slice parameters
        let slice_qty = quantity / params.num_slices as f64;
        let slice_interval = params.duration / params.num_slices as u32;

        let mut total_filled = 0.0;
        let mut total_value = 0.0;

        // Get arrival price for IS calculation
        let arrival_price = self.get_mid_price(connector, symbol).await?;

        for i in 0..params.num_slices {
            // Check for cancellation
            if self.is_cancelled(&request_id).await {
                break;
            }

            // Calculate this slice's quantity (with optional randomization)
            let qty = if params.randomize {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let variance = rng.gen_range(-0.2..0.2);
                slice_qty * (1.0 + variance)
            } else {
                slice_qty
            };

            // Place order
            let order = if params.use_limit_orders {
                let mid = self.get_mid_price(connector, symbol).await?;
                let offset = mid * params.limit_offset_bps / 10000.0;
                let price = match side {
                    Side::Buy => mid - offset,
                    Side::Sell => mid + offset,
                };
                Order::limit(symbol, side, qty, price)
            } else {
                Order::market(symbol, side, qty)
            };

            match connector.place_order(order.clone()).await {
                Ok(filled_order) => {
                    if let Some(avg_price) = filled_order.average_fill_price {
                        total_filled += filled_order.filled_quantity;
                        total_value += filled_order.filled_quantity * avg_price;
                    }

                    // Update result
                    let mut state = self.state.write().await;
                    if let Some(exec) = state.active_executions.get_mut(&request_id) {
                        exec.filled_quantity = total_filled;
                        exec.average_price = if total_filled > 0.0 {
                            total_value / total_filled
                        } else {
                            0.0
                        };
                        exec.num_orders += 1;
                        exec.child_orders.push(filled_order.client_order_id.clone());
                    }
                }
                Err(e) => {
                    tracing::warn!("TWAP slice {} failed: {}", i, e);
                }
            }

            // Wait for next slice (unless last)
            if i < params.num_slices - 1 {
                tokio::time::sleep(slice_interval).await;
            }
        }

        // Finalize result
        let mut state = self.state.write().await;
        if let Some(mut exec) = state.active_executions.remove(&request_id) {
            exec.status = if exec.filled_quantity >= quantity * 0.99 {
                ExecutionStatus::Completed
            } else if exec.filled_quantity > 0.0 {
                ExecutionStatus::Completed // Partial fill
            } else {
                ExecutionStatus::Failed
            };
            exec.end_time = Some(current_timestamp_ms());

            // Calculate slippage and IS
            if exec.average_price > 0.0 && arrival_price > 0.0 {
                let price_diff = match side {
                    Side::Buy => exec.average_price - arrival_price,
                    Side::Sell => arrival_price - exec.average_price,
                };
                exec.slippage_bps = (price_diff / arrival_price) * 10000.0;
                exec.implementation_shortfall_bps = exec.slippage_bps;
            }

            // Update metrics
            state.metrics.total_executions += 1;
            if exec.status == ExecutionStatus::Completed {
                state.metrics.completed_executions += 1;
            } else {
                state.metrics.failed_executions += 1;
            }

            state.completed_executions.push(exec.clone());
            return Ok(exec);
        }

        Err(LiveTradingError::Execution("Execution not found".into()))
    }

    /// Execute using VWAP algorithm.
    pub async fn execute_vwap<C: ExchangeConnector>(
        &self,
        connector: &C,
        symbol: &str,
        side: Side,
        quantity: f64,
        params: VWAPParams,
    ) -> Result<ExecutionResult> {
        let request_id = generate_client_order_id();
        let mut result = ExecutionResult::new(&request_id);
        result.status = ExecutionStatus::InProgress;

        // Store in active executions
        {
            let mut state = self.state.write().await;
            state
                .active_executions
                .insert(request_id.clone(), result.clone());
        }

        // Get volume profile or use default
        let profile = params.volume_profile.unwrap_or_else(|| {
            // Default to uniform distribution
            vec![1.0; 10]
        });

        // An explicitly-supplied empty profile would divide-by-zero below
        // (Duration / 0 panics); fall back to the uniform default.
        let profile = if profile.is_empty() {
            vec![1.0; 10]
        } else {
            profile
        };

        let total_weight: f64 = profile.iter().sum();
        let slice_duration = params.duration / profile.len() as u32;

        let mut total_filled = 0.0;
        let mut total_value = 0.0;
        let arrival_price = self.get_mid_price(connector, symbol).await?;

        for (i, &weight) in profile.iter().enumerate() {
            if self.is_cancelled(&request_id).await {
                break;
            }

            // Calculate target quantity for this slice based on volume profile
            let target_qty = quantity * (weight / total_weight);

            // Get current market volume and adjust
            let slice_qty = target_qty.min(quantity - total_filled);

            if slice_qty < self.config.min_order_size {
                continue;
            }

            let order = Order::market(symbol, side, slice_qty);

            match connector.place_order(order).await {
                Ok(filled_order) => {
                    if let Some(avg_price) = filled_order.average_fill_price {
                        total_filled += filled_order.filled_quantity;
                        total_value += filled_order.filled_quantity * avg_price;
                    }

                    let mut state = self.state.write().await;
                    if let Some(exec) = state.active_executions.get_mut(&request_id) {
                        exec.filled_quantity = total_filled;
                        exec.average_price = if total_filled > 0.0 {
                            total_value / total_filled
                        } else {
                            0.0
                        };
                        exec.num_orders += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!("VWAP slice {} failed: {}", i, e);
                }
            }

            if i < profile.len() - 1 {
                tokio::time::sleep(slice_duration).await;
            }
        }

        // Finalize
        self.finalize_execution(&request_id, quantity, side, arrival_price)
            .await
    }

    /// Execute using Iceberg algorithm.
    pub async fn execute_iceberg<C: ExchangeConnector>(
        &self,
        connector: &C,
        symbol: &str,
        side: Side,
        quantity: f64,
        params: IcebergParams,
    ) -> Result<ExecutionResult> {
        let request_id = generate_client_order_id();
        let mut result = ExecutionResult::new(&request_id);
        result.status = ExecutionStatus::InProgress;

        {
            let mut state = self.state.write().await;
            state
                .active_executions
                .insert(request_id.clone(), result.clone());
        }

        let arrival_price = self.get_mid_price(connector, symbol).await?;
        let mut remaining = quantity;
        let mut total_filled = 0.0;
        let mut total_value = 0.0;

        while remaining > self.config.min_order_size {
            if self.is_cancelled(&request_id).await {
                break;
            }

            // Calculate display quantity with variance
            let display = if params.qty_variance > 0.0 {
                use rand::Rng;
                let mut rng = rand::thread_rng();
                let variance = rng.gen_range(-params.qty_variance..params.qty_variance);
                params.display_qty * (1.0 + variance)
            } else {
                params.display_qty
            };

            let slice_qty = display.min(remaining);

            let order = if let Some(price) = params.price {
                Order::limit(symbol, side, slice_qty, price)
            } else {
                Order::market(symbol, side, slice_qty)
            };

            match connector.place_order(order).await {
                Ok(filled_order) => {
                    let filled = filled_order.filled_quantity;
                    if let Some(avg_price) = filled_order.average_fill_price {
                        total_filled += filled;
                        total_value += filled * avg_price;
                        remaining -= filled;
                    }

                    let mut state = self.state.write().await;
                    if let Some(exec) = state.active_executions.get_mut(&request_id) {
                        exec.filled_quantity = total_filled;
                        exec.average_price = if total_filled > 0.0 {
                            total_value / total_filled
                        } else {
                            0.0
                        };
                        exec.num_orders += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!("Iceberg slice failed: {}", e);
                    // Continue trying
                }
            }

            // Delay between refills
            tokio::time::sleep(Duration::from_millis(params.refill_delay_ms)).await;
        }

        self.finalize_execution(&request_id, quantity, side, arrival_price)
            .await
    }

    /// Execute a generic request.
    pub async fn execute<C: ExchangeConnector>(
        &self,
        connector: &C,
        request: ExecutionRequest,
    ) -> Result<ExecutionResult> {
        match request.algorithm {
            ExecutionAlgorithm::Market => self.execute_market_internal(connector, &request).await,
            ExecutionAlgorithm::Limit => self.execute_limit_internal(connector, &request).await,
            ExecutionAlgorithm::TWAP => {
                let params = TWAPParams::new(Duration::from_secs(300), 10);
                self.execute_twap(
                    connector,
                    &request.symbol,
                    request.side,
                    request.quantity,
                    params,
                )
                .await
            }
            ExecutionAlgorithm::VWAP => {
                let params = VWAPParams::new(Duration::from_secs(300));
                self.execute_vwap(
                    connector,
                    &request.symbol,
                    request.side,
                    request.quantity,
                    params,
                )
                .await
            }
            ExecutionAlgorithm::Iceberg => {
                let display_qty = request.quantity / 10.0;
                let params = IcebergParams::new(display_qty);
                self.execute_iceberg(
                    connector,
                    &request.symbol,
                    request.side,
                    request.quantity,
                    params,
                )
                .await
            }
            ExecutionAlgorithm::Aggressive => {
                // Aggressive: use market orders with higher size
                self.execute_market_internal(connector, &request).await
            }
            ExecutionAlgorithm::Passive => {
                // Passive: rest a limit order at the current mid price. Falling
                // back to 0.0 when no limit was supplied either rests unfilled
                // (paper) or is rejected (Binance/Bybit) — fetch the real mid.
                let price = match request.limit_price {
                    Some(p) => p,
                    None => self.get_mid_price(connector, &request.symbol).await?,
                };
                self.execute_limit_internal(connector, &request.limit_price(price))
                    .await
            }
            _ => {
                // Default to market
                self.execute_market_internal(connector, &request).await
            }
        }
    }

    // --- Private helper methods ---

    async fn execute_market_internal<C: ExchangeConnector>(
        &self,
        connector: &C,
        request: &ExecutionRequest,
    ) -> Result<ExecutionResult> {
        let mut result = ExecutionResult::new(&request.request_id);
        result.status = ExecutionStatus::InProgress;

        let arrival_price = self.get_mid_price(connector, &request.symbol).await?;

        let order = Order::market(&request.symbol, request.side, request.quantity);

        match connector.place_order(order).await {
            Ok(filled_order) => {
                result.filled_quantity = filled_order.filled_quantity;
                result.average_price = filled_order.average_fill_price.unwrap_or(0.0);
                result.num_orders = 1;
                result.num_fills = 1;
                result.status = ExecutionStatus::Completed;
                result.end_time = Some(current_timestamp_ms());
                result.child_orders.push(filled_order.client_order_id);

                // Calculate slippage
                if result.average_price > 0.0 && arrival_price > 0.0 {
                    let price_diff = match request.side {
                        Side::Buy => result.average_price - arrival_price,
                        Side::Sell => arrival_price - result.average_price,
                    };
                    result.slippage_bps = (price_diff / arrival_price) * 10000.0;
                }
            }
            Err(e) => {
                result.status = ExecutionStatus::Failed;
                result.error = Some(e.to_string());
                result.end_time = Some(current_timestamp_ms());
            }
        }

        // Update metrics
        let mut state = self.state.write().await;
        state.metrics.total_executions += 1;
        if result.status == ExecutionStatus::Completed {
            state.metrics.completed_executions += 1;
        } else {
            state.metrics.failed_executions += 1;
        }
        state.completed_executions.push(result.clone());

        Ok(result)
    }

    async fn execute_limit_internal<C: ExchangeConnector>(
        &self,
        connector: &C,
        request: &ExecutionRequest,
    ) -> Result<ExecutionResult> {
        let mut result = ExecutionResult::new(&request.request_id);
        result.status = ExecutionStatus::InProgress;

        let price = request
            .limit_price
            .ok_or_else(|| LiveTradingError::Execution("Limit price required".into()))?;

        let order = Order::limit(&request.symbol, request.side, request.quantity, price);

        match connector.place_order(order).await {
            Ok(placed_order) => {
                result.num_orders = 1;
                result
                    .child_orders
                    .push(placed_order.client_order_id.clone());

                // For limit orders, we'd need to wait for fills
                // This is simplified - in production would monitor order status
                result.status = ExecutionStatus::InProgress;
            }
            Err(e) => {
                result.status = ExecutionStatus::Failed;
                result.error = Some(e.to_string());
                result.end_time = Some(current_timestamp_ms());
            }
        }

        let mut state = self.state.write().await;
        state
            .active_executions
            .insert(request.request_id.clone(), result.clone());

        Ok(result)
    }

    async fn get_mid_price<C: ExchangeConnector>(
        &self,
        connector: &C,
        symbol: &str,
    ) -> Result<f64> {
        let book = connector.get_order_book(symbol, Some(5)).await?;
        book.mid_price()
            .ok_or_else(|| LiveTradingError::Execution("No mid price available".into()))
    }

    async fn is_cancelled(&self, request_id: &str) -> bool {
        let state = self.state.read().await;
        state
            .active_executions
            .get(request_id)
            .map(|e| e.status == ExecutionStatus::Cancelled)
            .unwrap_or(false)
    }

    async fn finalize_execution(
        &self,
        request_id: &str,
        target_quantity: f64,
        side: Side,
        arrival_price: f64,
    ) -> Result<ExecutionResult> {
        let mut state = self.state.write().await;

        if let Some(mut exec) = state.active_executions.remove(request_id) {
            exec.status = if exec.filled_quantity >= target_quantity * 0.99 {
                ExecutionStatus::Completed
            } else if exec.filled_quantity > 0.0 {
                ExecutionStatus::Completed
            } else {
                ExecutionStatus::Failed
            };
            exec.end_time = Some(current_timestamp_ms());

            // Calculate slippage and IS
            if exec.average_price > 0.0 && arrival_price > 0.0 {
                let price_diff = match side {
                    Side::Buy => exec.average_price - arrival_price,
                    Side::Sell => arrival_price - exec.average_price,
                };
                exec.slippage_bps = (price_diff / arrival_price) * 10000.0;
                exec.implementation_shortfall_bps = exec.slippage_bps;
            }

            // Update metrics
            state.metrics.total_executions += 1;
            if exec.status == ExecutionStatus::Completed {
                state.metrics.completed_executions += 1;
            } else {
                state.metrics.failed_executions += 1;
            }

            // Update running averages
            let n = state.metrics.total_executions as f64;
            state.metrics.avg_slippage_bps =
                (state.metrics.avg_slippage_bps * (n - 1.0) + exec.slippage_bps) / n;
            state.metrics.avg_is_bps =
                (state.metrics.avg_is_bps * (n - 1.0) + exec.implementation_shortfall_bps) / n;

            if target_quantity > 0.0 {
                let fill_rate = exec.filled_quantity / target_quantity;
                state.metrics.avg_fill_rate =
                    (state.metrics.avg_fill_rate * (n - 1.0) + fill_rate) / n;
            }

            let exec_time = exec.end_time.unwrap_or(0) - exec.start_time;
            state.metrics.avg_execution_time_ms =
                (state.metrics.avg_execution_time_ms * (n - 1.0) + exec_time as f64) / n;

            state.metrics.total_volume += exec.filled_quantity * exec.average_price;
            state.metrics.total_commission += exec.total_commission;

            state.completed_executions.push(exec.clone());
            return Ok(exec);
        }

        Err(LiveTradingError::Execution("Execution not found".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execution_config() {
        let config = ExecutionConfig::default()
            .max_slippage_bps(20.0)
            .default_algorithm(ExecutionAlgorithm::TWAP)
            .max_retries(5);

        assert_eq!(config.max_slippage_bps, 20.0);
        assert_eq!(config.default_algorithm, ExecutionAlgorithm::TWAP);
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_twap_params() {
        let params = TWAPParams::new(Duration::from_secs(300), 10)
            .randomize(false)
            .max_deviation_bps(30.0);

        assert_eq!(params.num_slices, 10);
        assert!(!params.randomize);
        assert_eq!(params.max_deviation_bps, 30.0);
    }

    #[test]
    fn test_execution_request() {
        let request = ExecutionRequest::new("BTCUSDT", Side::Buy, 1.0)
            .algorithm(ExecutionAlgorithm::VWAP)
            .urgency(Urgency::High)
            .max_slippage_bps(15.0);

        assert_eq!(request.symbol, "BTCUSDT");
        assert_eq!(request.side, Side::Buy);
        assert_eq!(request.algorithm, ExecutionAlgorithm::VWAP);
        assert_eq!(request.urgency, Urgency::High);
    }

    #[test]
    fn test_iceberg_params() {
        let params = IcebergParams::new(0.1).price(50000.0).qty_variance(0.2);

        assert_eq!(params.display_qty, 0.1);
        assert_eq!(params.price, Some(50000.0));
        assert_eq!(params.qty_variance, 0.2);
    }
}
