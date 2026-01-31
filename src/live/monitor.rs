//! Live trading monitoring and alerting system.
//!
//! Provides real-time monitoring of:
//! - P&L tracking
//! - Position monitoring
//! - Risk metric updates
//! - Alert system for drawdown and exposure limits
//! - Heartbeat/health checks
//! - Integration with TUI dashboard
//!
//! # Example
//!
//! ```ignore
//! use octane_rs::live::monitor::{Monitor, MonitorConfig, AlertConfig};
//!
//! let config = MonitorConfig::default()
//!     .update_interval_ms(100)
//!     .enable_alerts(true);
//!
//! let mut monitor = Monitor::new(config);
//!
//! // Add alert for drawdown
//! monitor.add_alert(Alert::MaxDrawdown { threshold: 0.05 });
//!
//! // Start monitoring
//! monitor.start().await?;
//! ```

use crate::live::error::{LiveTradingError, Result};
use crate::live::types::{Balance, Order, Position, Trade, current_timestamp_ms};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{broadcast, mpsc, RwLock};

/// Alert severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert.
    Info,
    /// Warning - requires attention.
    Warning,
    /// Critical - immediate action needed.
    Critical,
    /// Emergency - trading should be halted.
    Emergency,
}

/// Type of alert condition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// Maximum drawdown exceeded.
    MaxDrawdown {
        /// Drawdown threshold (0.0-1.0).
        threshold: f64,
    },
    /// Maximum exposure exceeded.
    MaxExposure {
        /// Exposure threshold in quote currency.
        threshold: f64,
    },
    /// Maximum position size exceeded.
    MaxPositionSize {
        /// Symbol (None for any symbol).
        symbol: Option<String>,
        /// Size threshold.
        threshold: f64,
    },
    /// Daily loss limit exceeded.
    DailyLossLimit {
        /// Loss threshold in quote currency.
        threshold: f64,
    },
    /// Position held too long.
    PositionDuration {
        /// Maximum duration in seconds.
        max_duration_secs: u64,
    },
    /// Large adverse price move.
    AdversePriceMove {
        /// Price move threshold in percentage.
        threshold_pct: f64,
    },
    /// Connection lost.
    ConnectionLost,
    /// High latency detected.
    HighLatency {
        /// Latency threshold in milliseconds.
        threshold_ms: u64,
    },
    /// Order fill rate too low.
    LowFillRate {
        /// Minimum fill rate threshold.
        threshold: f64,
    },
    /// Custom alert with condition function.
    Custom {
        /// Alert name.
        name: String,
        /// Description.
        description: String,
    },
}

/// An alert notification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertNotification {
    /// Alert ID.
    pub id: String,
    /// Alert type.
    pub alert_type: AlertType,
    /// Severity level.
    pub severity: AlertSeverity,
    /// Alert message.
    pub message: String,
    /// Current value that triggered the alert.
    pub current_value: f64,
    /// Threshold value.
    pub threshold_value: f64,
    /// Timestamp.
    pub timestamp: u64,
    /// Whether the alert is acknowledged.
    pub acknowledged: bool,
}

/// P&L snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PnLSnapshot {
    /// Timestamp.
    pub timestamp: u64,
    /// Realized P&L.
    pub realized_pnl: f64,
    /// Unrealized P&L.
    pub unrealized_pnl: f64,
    /// Total P&L.
    pub total_pnl: f64,
    /// P&L percentage.
    pub pnl_pct: f64,
    /// Daily P&L.
    pub daily_pnl: f64,
    /// Daily P&L percentage.
    pub daily_pnl_pct: f64,
    /// Current drawdown.
    pub drawdown: f64,
    /// Maximum drawdown.
    pub max_drawdown: f64,
    /// Peak portfolio value.
    pub peak_value: f64,
    /// Current portfolio value.
    pub portfolio_value: f64,
}

/// Risk metrics snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RiskMetrics {
    /// Timestamp.
    pub timestamp: u64,
    /// Total exposure (sum of absolute position values).
    pub total_exposure: f64,
    /// Net exposure (sum of signed position values).
    pub net_exposure: f64,
    /// Gross exposure.
    pub gross_exposure: f64,
    /// Long exposure.
    pub long_exposure: f64,
    /// Short exposure.
    pub short_exposure: f64,
    /// Number of open positions.
    pub num_positions: usize,
    /// Leverage ratio.
    pub leverage: f64,
    /// Value at Risk (VaR) estimate.
    pub var_95: f64,
    /// Expected shortfall / CVaR.
    pub cvar_95: f64,
    /// Sharpe ratio (rolling).
    pub sharpe_ratio: f64,
    /// Sortino ratio (rolling).
    pub sortino_ratio: f64,
    /// Win rate.
    pub win_rate: f64,
    /// Average win.
    pub avg_win: f64,
    /// Average loss.
    pub avg_loss: f64,
    /// Profit factor.
    pub profit_factor: f64,
}

/// Health check status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System is healthy.
    Healthy,
    /// System is degraded but operational.
    Degraded,
    /// System is unhealthy.
    Unhealthy,
    /// Health check failed.
    Unknown,
}

/// Component health.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    /// Component name.
    pub name: String,
    /// Health status.
    pub status: HealthStatus,
    /// Last heartbeat timestamp.
    pub last_heartbeat: u64,
    /// Latency in milliseconds.
    pub latency_ms: Option<u64>,
    /// Error message if unhealthy.
    pub error: Option<String>,
}

/// System health snapshot.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Overall status.
    pub status: HealthStatus,
    /// Component health statuses.
    pub components: Vec<ComponentHealth>,
    /// Last update timestamp.
    pub timestamp: u64,
    /// Uptime in seconds.
    pub uptime_secs: u64,
}

/// Trading statistics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TradingStats {
    /// Total number of trades.
    pub total_trades: u64,
    /// Winning trades.
    pub winning_trades: u64,
    /// Losing trades.
    pub losing_trades: u64,
    /// Total volume traded.
    pub total_volume: f64,
    /// Total commission paid.
    pub total_commission: f64,
    /// Average trade size.
    pub avg_trade_size: f64,
    /// Average holding period (seconds).
    pub avg_holding_period_secs: f64,
    /// Best trade P&L.
    pub best_trade: f64,
    /// Worst trade P&L.
    pub worst_trade: f64,
    /// Total slippage.
    pub total_slippage: f64,
    /// Average slippage per trade.
    pub avg_slippage: f64,
    /// Average winning trade P&L.
    pub avg_win: f64,
    /// Average losing trade P&L.
    pub avg_loss: f64,
}

/// Monitor event for streaming.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MonitorEvent {
    /// P&L update.
    PnLUpdate(PnLSnapshot),
    /// Risk metrics update.
    RiskUpdate(RiskMetrics),
    /// Position update.
    PositionUpdate(Position),
    /// Trade executed.
    TradeExecuted(Trade),
    /// Order update.
    OrderUpdate(Order),
    /// Alert triggered.
    AlertTriggered(AlertNotification),
    /// Health update.
    HealthUpdate(SystemHealth),
    /// Stats update.
    StatsUpdate(TradingStats),
}

/// Configuration for the monitor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// Update interval in milliseconds.
    pub update_interval_ms: u64,
    /// P&L update interval in milliseconds.
    pub pnl_update_interval_ms: u64,
    /// Risk update interval in milliseconds.
    pub risk_update_interval_ms: u64,
    /// Health check interval in milliseconds.
    pub health_check_interval_ms: u64,
    /// Enable alerts.
    pub enable_alerts: bool,
    /// Enable event streaming.
    pub enable_streaming: bool,
    /// Event buffer size.
    pub event_buffer_size: usize,
    /// Enable logging.
    pub enable_logging: bool,
    /// Log file path.
    pub log_path: Option<String>,
    /// Initial capital for P&L calculations.
    pub initial_capital: f64,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            update_interval_ms: 100,
            pnl_update_interval_ms: 1000,
            risk_update_interval_ms: 5000,
            health_check_interval_ms: 10000,
            enable_alerts: true,
            enable_streaming: true,
            event_buffer_size: 1000,
            enable_logging: true,
            log_path: None,
            initial_capital: 10000.0,
        }
    }
}

impl MonitorConfig {
    /// Create new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set update interval.
    pub fn update_interval_ms(mut self, ms: u64) -> Self {
        self.update_interval_ms = ms;
        self
    }

    /// Set P&L update interval.
    pub fn pnl_update_interval_ms(mut self, ms: u64) -> Self {
        self.pnl_update_interval_ms = ms;
        self
    }

    /// Set risk update interval.
    pub fn risk_update_interval_ms(mut self, ms: u64) -> Self {
        self.risk_update_interval_ms = ms;
        self
    }

    /// Enable alerts.
    pub fn enable_alerts(mut self, enabled: bool) -> Self {
        self.enable_alerts = enabled;
        self
    }

    /// Enable streaming.
    pub fn enable_streaming(mut self, enabled: bool) -> Self {
        self.enable_streaming = enabled;
        self
    }

    /// Set initial capital.
    pub fn initial_capital(mut self, capital: f64) -> Self {
        self.initial_capital = capital;
        self
    }

    /// Set log path.
    pub fn log_path(mut self, path: impl Into<String>) -> Self {
        self.log_path = Some(path.into());
        self
    }
}

/// Alert configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfig {
    /// Alert type.
    pub alert_type: AlertType,
    /// Severity level.
    pub severity: AlertSeverity,
    /// Cooldown between alerts (seconds).
    pub cooldown_secs: u64,
    /// Whether to auto-halt trading on trigger.
    pub halt_on_trigger: bool,
    /// Whether the alert is enabled.
    pub enabled: bool,
}

impl AlertConfig {
    /// Create new alert config.
    pub fn new(alert_type: AlertType) -> Self {
        Self {
            alert_type,
            severity: AlertSeverity::Warning,
            cooldown_secs: 60,
            halt_on_trigger: false,
            enabled: true,
        }
    }

    /// Set severity.
    pub fn severity(mut self, severity: AlertSeverity) -> Self {
        self.severity = severity;
        self
    }

    /// Set cooldown.
    pub fn cooldown_secs(mut self, secs: u64) -> Self {
        self.cooldown_secs = secs;
        self
    }

    /// Set halt on trigger.
    pub fn halt_on_trigger(mut self, halt: bool) -> Self {
        self.halt_on_trigger = halt;
        self
    }
}

/// Internal monitor state.
struct MonitorState {
    /// Current P&L snapshot.
    pnl: PnLSnapshot,
    /// Current risk metrics.
    risk: RiskMetrics,
    /// Current system health.
    health: SystemHealth,
    /// Trading statistics.
    stats: TradingStats,
    /// Positions by symbol.
    positions: HashMap<String, Position>,
    /// Balances by asset.
    balances: HashMap<String, Balance>,
    /// Open orders.
    open_orders: HashMap<String, Order>,
    /// Recent trades.
    recent_trades: Vec<Trade>,
    /// Alert configurations.
    alerts: Vec<AlertConfig>,
    /// Active (unacknowledged) alerts.
    active_alerts: Vec<AlertNotification>,
    /// Alert cooldowns (alert ID -> last trigger time).
    alert_cooldowns: HashMap<String, u64>,
    /// Trading halted flag.
    trading_halted: bool,
    /// Start time.
    start_time: u64,
    /// Daily start values for P&L.
    daily_start_value: f64,
    /// Daily start time.
    daily_start_time: u64,
}

impl Default for MonitorState {
    fn default() -> Self {
        let now = current_timestamp_ms();
        Self {
            pnl: PnLSnapshot::default(),
            risk: RiskMetrics::default(),
            health: SystemHealth {
                status: HealthStatus::Unknown,
                components: Vec::new(),
                timestamp: now,
                uptime_secs: 0,
            },
            stats: TradingStats::default(),
            positions: HashMap::new(),
            balances: HashMap::new(),
            open_orders: HashMap::new(),
            recent_trades: Vec::new(),
            alerts: Vec::new(),
            active_alerts: Vec::new(),
            alert_cooldowns: HashMap::new(),
            trading_halted: false,
            start_time: now,
            daily_start_value: 0.0,
            daily_start_time: now,
        }
    }
}

/// Live trading monitor.
pub struct Monitor {
    /// Configuration.
    config: MonitorConfig,
    /// Internal state.
    state: Arc<RwLock<MonitorState>>,
    /// Event broadcaster.
    event_tx: broadcast::Sender<MonitorEvent>,
    /// Shutdown signal.
    shutdown_tx: Option<mpsc::Sender<()>>,
    /// Running flag.
    running: Arc<RwLock<bool>>,
}

impl Monitor {
    /// Create a new monitor.
    pub fn new(config: MonitorConfig) -> Self {
        let (event_tx, _) = broadcast::channel(config.event_buffer_size);

        Self {
            config,
            state: Arc::new(RwLock::new(MonitorState::default())),
            event_tx,
            shutdown_tx: None,
            running: Arc::new(RwLock::new(false)),
        }
    }

    /// Get the event receiver for streaming.
    pub fn subscribe(&self) -> broadcast::Receiver<MonitorEvent> {
        self.event_tx.subscribe()
    }

    /// Start the monitor.
    pub async fn start(&mut self) -> Result<()> {
        let mut running = self.running.write().await;
        if *running {
            return Err(LiveTradingError::Monitoring("Monitor already running".into()));
        }
        *running = true;
        drop(running);

        let (shutdown_tx, mut shutdown_rx) = mpsc::channel::<()>(1);
        self.shutdown_tx = Some(shutdown_tx);

        // Initialize daily tracking
        {
            let mut state = self.state.write().await;
            state.daily_start_time = current_timestamp_ms();
            state.daily_start_value = self.config.initial_capital;
        }

        // Spawn monitoring tasks
        let state = Arc::clone(&self.state);
        let config = self.config.clone();
        let event_tx = self.event_tx.clone();
        let running = Arc::clone(&self.running);

        tokio::spawn(async move {
            let mut pnl_interval =
                tokio::time::interval(Duration::from_millis(config.pnl_update_interval_ms));
            let mut risk_interval =
                tokio::time::interval(Duration::from_millis(config.risk_update_interval_ms));
            let mut health_interval =
                tokio::time::interval(Duration::from_millis(config.health_check_interval_ms));

            loop {
                tokio::select! {
                    _ = shutdown_rx.recv() => {
                        break;
                    }
                    _ = pnl_interval.tick() => {
                        let s = state.read().await;
                        if config.enable_streaming {
                            let _ = event_tx.send(MonitorEvent::PnLUpdate(s.pnl.clone()));
                        }
                    }
                    _ = risk_interval.tick() => {
                        let s = state.read().await;
                        if config.enable_streaming {
                            let _ = event_tx.send(MonitorEvent::RiskUpdate(s.risk.clone()));
                        }
                    }
                    _ = health_interval.tick() => {
                        let s = state.read().await;
                        if config.enable_streaming {
                            let _ = event_tx.send(MonitorEvent::HealthUpdate(s.health.clone()));
                        }
                    }
                }

                // Check if still running
                if !*running.read().await {
                    break;
                }
            }
        });

        Ok(())
    }

    /// Stop the monitor.
    pub async fn stop(&mut self) -> Result<()> {
        *self.running.write().await = false;

        if let Some(tx) = self.shutdown_tx.take() {
            let _ = tx.send(()).await;
        }

        Ok(())
    }

    /// Check if monitor is running.
    pub async fn is_running(&self) -> bool {
        *self.running.read().await
    }

    /// Check if trading is halted.
    pub async fn is_trading_halted(&self) -> bool {
        self.state.read().await.trading_halted
    }

    /// Halt trading.
    pub async fn halt_trading(&self, reason: &str) {
        let mut state = self.state.write().await;
        state.trading_halted = true;

        let alert = AlertNotification {
            id: format!("halt_{}", current_timestamp_ms()),
            alert_type: AlertType::Custom {
                name: "Trading Halted".to_string(),
                description: reason.to_string(),
            },
            severity: AlertSeverity::Emergency,
            message: format!("Trading halted: {}", reason),
            current_value: 0.0,
            threshold_value: 0.0,
            timestamp: current_timestamp_ms(),
            acknowledged: false,
        };

        state.active_alerts.push(alert.clone());

        if self.config.enable_streaming {
            let _ = self.event_tx.send(MonitorEvent::AlertTriggered(alert));
        }
    }

    /// Resume trading.
    pub async fn resume_trading(&self) {
        let mut state = self.state.write().await;
        state.trading_halted = false;
    }

    /// Add an alert configuration.
    pub async fn add_alert(&self, alert_config: AlertConfig) {
        let mut state = self.state.write().await;
        state.alerts.push(alert_config);
    }

    /// Remove an alert by index.
    pub async fn remove_alert(&self, index: usize) {
        let mut state = self.state.write().await;
        if index < state.alerts.len() {
            state.alerts.remove(index);
        }
    }

    /// Get active alerts.
    pub async fn active_alerts(&self) -> Vec<AlertNotification> {
        self.state.read().await.active_alerts.clone()
    }

    /// Acknowledge an alert.
    pub async fn acknowledge_alert(&self, alert_id: &str) {
        let mut state = self.state.write().await;
        for alert in &mut state.active_alerts {
            if alert.id == alert_id {
                alert.acknowledged = true;
            }
        }
    }

    /// Clear acknowledged alerts.
    pub async fn clear_acknowledged_alerts(&self) {
        let mut state = self.state.write().await;
        state.active_alerts.retain(|a| !a.acknowledged);
    }

    /// Get current P&L snapshot.
    pub async fn pnl(&self) -> PnLSnapshot {
        self.state.read().await.pnl.clone()
    }

    /// Get current risk metrics.
    pub async fn risk_metrics(&self) -> RiskMetrics {
        self.state.read().await.risk.clone()
    }

    /// Get current system health.
    pub async fn health(&self) -> SystemHealth {
        self.state.read().await.health.clone()
    }

    /// Get trading statistics.
    pub async fn stats(&self) -> TradingStats {
        self.state.read().await.stats.clone()
    }

    /// Get positions.
    pub async fn positions(&self) -> HashMap<String, Position> {
        self.state.read().await.positions.clone()
    }

    /// Get balances.
    pub async fn balances(&self) -> HashMap<String, Balance> {
        self.state.read().await.balances.clone()
    }

    /// Update position.
    pub async fn update_position(&self, position: Position) {
        let mut state = self.state.write().await;
        state.positions.insert(position.symbol.clone(), position.clone());

        // Recalculate metrics
        self.recalculate_pnl(&mut state);
        self.recalculate_risk(&mut state);

        // Check alerts
        let alerts = self.check_alerts(&state);
        for alert in alerts {
            state.active_alerts.push(alert.clone());
            if self.config.enable_streaming {
                let _ = self.event_tx.send(MonitorEvent::AlertTriggered(alert));
            }
        }

        if self.config.enable_streaming {
            let _ = self.event_tx.send(MonitorEvent::PositionUpdate(position));
        }
    }

    /// Update balance.
    pub async fn update_balance(&self, balance: Balance) {
        let mut state = self.state.write().await;
        state.balances.insert(balance.asset.clone(), balance);

        self.recalculate_pnl(&mut state);
    }

    /// Record a trade.
    pub async fn record_trade(&self, trade: Trade) {
        let mut state = self.state.write().await;

        // Update stats
        state.stats.total_trades += 1;
        state.stats.total_volume += trade.quantity * trade.price;
        state.stats.total_commission += trade.commission;

        // Calculate P&L (simplified)
        // In production, would need to track cost basis properly

        // Keep recent trades (last 100)
        state.recent_trades.push(trade.clone());
        if state.recent_trades.len() > 100 {
            state.recent_trades.remove(0);
        }

        if self.config.enable_streaming {
            let _ = self.event_tx.send(MonitorEvent::TradeExecuted(trade));
        }
    }

    /// Update order.
    pub async fn update_order(&self, order: Order) {
        let mut state = self.state.write().await;

        if order.status.is_terminal() {
            state.open_orders.remove(&order.client_order_id);
        } else {
            state.open_orders.insert(order.client_order_id.clone(), order.clone());
        }

        if self.config.enable_streaming {
            let _ = self.event_tx.send(MonitorEvent::OrderUpdate(order));
        }
    }

    /// Update component health.
    pub async fn update_component_health(&self, component: ComponentHealth) {
        let mut state = self.state.write().await;

        // Update or add component
        if let Some(existing) = state
            .health
            .components
            .iter_mut()
            .find(|c| c.name == component.name)
        {
            *existing = component;
        } else {
            state.health.components.push(component);
        }

        // Recalculate overall health
        state.health.status = self.calculate_overall_health(&state.health.components);
        state.health.timestamp = current_timestamp_ms();
        state.health.uptime_secs = (current_timestamp_ms() - state.start_time) / 1000;
    }

    /// Record heartbeat for a component.
    pub async fn heartbeat(&self, component_name: &str, latency_ms: Option<u64>) {
        let component = ComponentHealth {
            name: component_name.to_string(),
            status: HealthStatus::Healthy,
            last_heartbeat: current_timestamp_ms(),
            latency_ms,
            error: None,
        };

        self.update_component_health(component).await;
    }

    // --- Private helper methods ---

    fn recalculate_pnl(&self, state: &mut MonitorState) {
        let now = current_timestamp_ms();

        // Calculate total portfolio value
        let mut portfolio_value = 0.0;

        // Add balances (simplified - assumes USDT as quote)
        for balance in state.balances.values() {
            if balance.asset == "USDT" || balance.asset == "USD" {
                portfolio_value += balance.total();
            }
            // Would need price lookup for other assets
        }

        // Add unrealized P&L from positions
        let mut unrealized_pnl = 0.0;
        for pos in state.positions.values() {
            unrealized_pnl += pos.unrealized_pnl;
        }
        portfolio_value += unrealized_pnl;

        // Calculate realized P&L from trades
        let realized_pnl: f64 = state.positions.values().map(|p| p.realized_pnl).sum();

        let total_pnl = realized_pnl + unrealized_pnl;
        let pnl_pct = if self.config.initial_capital > 0.0 {
            total_pnl / self.config.initial_capital * 100.0
        } else {
            0.0
        };

        // Calculate daily P&L
        let daily_pnl = portfolio_value - state.daily_start_value;
        let daily_pnl_pct = if state.daily_start_value > 0.0 {
            daily_pnl / state.daily_start_value * 100.0
        } else {
            0.0
        };

        // Update peak and drawdown
        if portfolio_value > state.pnl.peak_value {
            state.pnl.peak_value = portfolio_value;
        }

        let drawdown = if state.pnl.peak_value > 0.0 {
            (state.pnl.peak_value - portfolio_value) / state.pnl.peak_value
        } else {
            0.0
        };

        if drawdown > state.pnl.max_drawdown {
            state.pnl.max_drawdown = drawdown;
        }

        state.pnl = PnLSnapshot {
            timestamp: now,
            realized_pnl,
            unrealized_pnl,
            total_pnl,
            pnl_pct,
            daily_pnl,
            daily_pnl_pct,
            drawdown,
            max_drawdown: state.pnl.max_drawdown,
            peak_value: state.pnl.peak_value.max(portfolio_value),
            portfolio_value,
        };
    }

    fn recalculate_risk(&self, state: &mut MonitorState) {
        let now = current_timestamp_ms();

        let mut total_exposure = 0.0;
        let mut net_exposure = 0.0;
        let mut long_exposure = 0.0;
        let mut short_exposure = 0.0;

        for pos in state.positions.values() {
            let notional = pos.notional_value();
            total_exposure += notional;

            if pos.is_long() {
                long_exposure += notional;
                net_exposure += notional;
            } else if pos.is_short() {
                short_exposure += notional;
                net_exposure -= notional;
            }
        }

        let gross_exposure = long_exposure + short_exposure;
        let leverage = if state.pnl.portfolio_value > 0.0 {
            gross_exposure / state.pnl.portfolio_value
        } else {
            0.0
        };

        // Calculate win rate and profit factor
        let total_trades = state.stats.winning_trades + state.stats.losing_trades;
        let win_rate = if total_trades > 0 {
            state.stats.winning_trades as f64 / total_trades as f64
        } else {
            0.0
        };

        let profit_factor = if state.stats.avg_loss.abs() > 0.0 {
            state.stats.avg_win / state.stats.avg_loss.abs()
        } else {
            0.0
        };

        state.risk = RiskMetrics {
            timestamp: now,
            total_exposure,
            net_exposure,
            gross_exposure,
            long_exposure,
            short_exposure,
            num_positions: state.positions.len(),
            leverage,
            var_95: 0.0,      // Would need historical data to calculate
            cvar_95: 0.0,     // Would need historical data to calculate
            sharpe_ratio: 0.0, // Would need historical returns
            sortino_ratio: 0.0,
            win_rate,
            avg_win: state.stats.avg_win,
            avg_loss: state.stats.avg_loss,
            profit_factor,
        };
    }

    fn check_alerts(&self, state: &MonitorState) -> Vec<AlertNotification> {
        let mut triggered = Vec::new();
        let now = current_timestamp_ms();

        for alert_config in &state.alerts {
            if !alert_config.enabled {
                continue;
            }

            // Check cooldown
            let alert_id = format!("{:?}", alert_config.alert_type);
            if let Some(&last_trigger) = state.alert_cooldowns.get(&alert_id) {
                if now - last_trigger < alert_config.cooldown_secs * 1000 {
                    continue;
                }
            }

            let (should_trigger, current_value, threshold_value, message) =
                self.evaluate_alert(state, &alert_config.alert_type);

            if should_trigger {
                triggered.push(AlertNotification {
                    id: format!("{}_{}", alert_id, now),
                    alert_type: alert_config.alert_type.clone(),
                    severity: alert_config.severity,
                    message,
                    current_value,
                    threshold_value,
                    timestamp: now,
                    acknowledged: false,
                });
            }
        }

        triggered
    }

    fn evaluate_alert(
        &self,
        state: &MonitorState,
        alert_type: &AlertType,
    ) -> (bool, f64, f64, String) {
        match alert_type {
            AlertType::MaxDrawdown { threshold } => {
                let current = state.pnl.drawdown;
                let triggered = current > *threshold;
                (
                    triggered,
                    current,
                    *threshold,
                    format!(
                        "Drawdown {:.2}% exceeds threshold {:.2}%",
                        current * 100.0,
                        threshold * 100.0
                    ),
                )
            }
            AlertType::MaxExposure { threshold } => {
                let current = state.risk.total_exposure;
                let triggered = current > *threshold;
                (
                    triggered,
                    current,
                    *threshold,
                    format!(
                        "Exposure ${:.2} exceeds threshold ${:.2}",
                        current, threshold
                    ),
                )
            }
            AlertType::MaxPositionSize { symbol, threshold } => {
                let current = if let Some(sym) = symbol {
                    state
                        .positions
                        .get(sym)
                        .map(|p| p.size.abs())
                        .unwrap_or(0.0)
                } else {
                    state
                        .positions
                        .values()
                        .map(|p| p.size.abs())
                        .fold(0.0, f64::max)
                };
                let triggered = current > *threshold;
                (
                    triggered,
                    current,
                    *threshold,
                    format!("Position size {} exceeds threshold {}", current, threshold),
                )
            }
            AlertType::DailyLossLimit { threshold } => {
                let current = -state.pnl.daily_pnl;
                let triggered = current > *threshold;
                (
                    triggered,
                    current,
                    *threshold,
                    format!(
                        "Daily loss ${:.2} exceeds limit ${:.2}",
                        current, threshold
                    ),
                )
            }
            AlertType::HighLatency { threshold_ms } => {
                let max_latency = state
                    .health
                    .components
                    .iter()
                    .filter_map(|c| c.latency_ms)
                    .max()
                    .unwrap_or(0);
                let triggered = max_latency > *threshold_ms;
                (
                    triggered,
                    max_latency as f64,
                    *threshold_ms as f64,
                    format!("High latency detected: {}ms", max_latency),
                )
            }
            AlertType::ConnectionLost => {
                let any_unhealthy = state
                    .health
                    .components
                    .iter()
                    .any(|c| c.status == HealthStatus::Unhealthy);
                (
                    any_unhealthy,
                    if any_unhealthy { 1.0 } else { 0.0 },
                    0.0,
                    "Connection lost to one or more components".to_string(),
                )
            }
            _ => (false, 0.0, 0.0, String::new()),
        }
    }

    fn calculate_overall_health(&self, components: &[ComponentHealth]) -> HealthStatus {
        if components.is_empty() {
            return HealthStatus::Unknown;
        }

        let unhealthy_count = components
            .iter()
            .filter(|c| c.status == HealthStatus::Unhealthy)
            .count();

        let degraded_count = components
            .iter()
            .filter(|c| c.status == HealthStatus::Degraded)
            .count();

        if unhealthy_count > 0 {
            HealthStatus::Unhealthy
        } else if degraded_count > 0 {
            HealthStatus::Degraded
        } else {
            HealthStatus::Healthy
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monitor_config() {
        let config = MonitorConfig::default()
            .update_interval_ms(50)
            .enable_alerts(true)
            .initial_capital(50000.0);

        assert_eq!(config.update_interval_ms, 50);
        assert!(config.enable_alerts);
        assert_eq!(config.initial_capital, 50000.0);
    }

    #[test]
    fn test_alert_config() {
        let alert = AlertConfig::new(AlertType::MaxDrawdown { threshold: 0.05 })
            .severity(AlertSeverity::Critical)
            .cooldown_secs(120)
            .halt_on_trigger(true);

        assert_eq!(alert.severity, AlertSeverity::Critical);
        assert_eq!(alert.cooldown_secs, 120);
        assert!(alert.halt_on_trigger);
    }

    #[tokio::test]
    async fn test_monitor_creation() {
        let config = MonitorConfig::default();
        let monitor = Monitor::new(config);

        assert!(!monitor.is_running().await);
        assert!(!monitor.is_trading_halted().await);
    }

    #[tokio::test]
    async fn test_position_update() {
        let config = MonitorConfig::default().initial_capital(10000.0);
        let monitor = Monitor::new(config);

        let position = Position {
            symbol: "BTCUSDT".to_string(),
            size: 0.1,
            entry_price: 50000.0,
            mark_price: 51000.0,
            unrealized_pnl: 100.0,
            realized_pnl: 0.0,
            liquidation_price: None,
            leverage: None,
            margin: None,
            updated_at: current_timestamp_ms(),
        };

        monitor.update_position(position).await;

        let positions = monitor.positions().await;
        assert!(positions.contains_key("BTCUSDT"));
    }

    #[tokio::test]
    async fn test_health_tracking() {
        let config = MonitorConfig::default();
        let monitor = Monitor::new(config);

        let component = ComponentHealth {
            name: "exchange".to_string(),
            status: HealthStatus::Healthy,
            last_heartbeat: current_timestamp_ms(),
            latency_ms: Some(50),
            error: None,
        };

        monitor.update_component_health(component).await;

        let health = monitor.health().await;
        assert_eq!(health.components.len(), 1);
        assert_eq!(health.components[0].name, "exchange");
    }
}
