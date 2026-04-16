//! Trading metrics and analytics module.
//!
//! Provides comprehensive financial analysis tools including:
//! - **Trading Metrics** (`trading`): Risk-adjusted returns, drawdown analysis, VaR/CVaR
//! - **Trade Journal** (`journal`): Automatic trade logging with feature attribution
//! - **Performance Attribution** (`attribution`): P&L breakdown by multiple dimensions
//!
//! # Examples
//!
//! ## Computing Trading Metrics
//!
//! ```rust
//! use octane_rs::metrics::{MetricsConfig, MetricsCalculator};
//!
//! let config = MetricsConfig::new(252.0) // Daily trading
//!     .risk_free_rate(0.02)
//!     .var_confidence(0.95);
//!
//! let mut calc = MetricsCalculator::new(config);
//!
//! // Add returns
//! calc.add_return(0.01);  // +1%
//! calc.add_return(0.02);  // +2%
//! calc.add_return(-0.01); // -1%
//!
//! // Add completed trades
//! calc.add_trade(100.0);  // Win
//! calc.add_trade(-50.0);  // Loss
//!
//! // Get comprehensive metrics
//! let metrics = calc.compute_metrics();
//! println!("Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
//! println!("Win Rate: {:.2}%", metrics.win_rate * 100.0);
//! println!("Max Drawdown: {:.2}%", metrics.max_drawdown_pct * 100.0);
//! ```
//!
//! ## Using Trade Journal
//!
//! ```rust
//! use octane_rs::metrics::{JournalConfig, TradeJournal, TradeDirection};
//!
//! let config = JournalConfig::default()
//!     .auto_tag_trades(true)
//!     .track_feature_attribution(true);
//!
//! let mut journal = TradeJournal::new(config)
//!     .with_output_path("trades.json".to_string());
//!
//! // Open a trade
//! let trade_id = journal.open_trade(
//!     "AAPL".to_string(),
//!     TradeDirection::Long,
//!     1640000000, // timestamp
//!     150.0,      // price
//!     10.0,       // size
//! );
//!
//! // Close the trade
//! journal.close_trade(trade_id, 1640003600, 155.0);
//!
//! // Get statistics
//! let stats = journal.aggregate_stats();
//! println!("Total P&L: ${:.2}", stats.total_pnl);
//!
//! // Export to CSV
//! journal.export_csv("trades.csv").unwrap();
//! ```
//!
//! ## Performance Attribution
//!
//! ```rust
//! use octane_rs::metrics::{
//!     AttributionConfig, AttributionAnalyzer, Direction, MarketRegime
//! };
//!
//! let config = AttributionConfig::default()
//!     .enable_asset_attribution(true)
//!     .enable_regime_attribution(true);
//!
//! let mut analyzer = AttributionAnalyzer::new(config);
//!
//! // Add trades with metadata
//! analyzer.add_trade(
//!     100.0,                          // P&L
//!     1640000000,                     // timestamp
//!     Some("AAPL"),                   // symbol
//!     Some(Direction::Long),          // direction
//!     Some(MarketRegime::Trending),   // regime
//!     3600,                           // duration
//!     None,                           // factor exposures
//! );
//!
//! // Finalize and get report
//! analyzer.finalize_report();
//! let report = analyzer.get_report();
//!
//! println!("Total P&L: ${:.2}", report.total_pnl);
//! println!("Top contributors:");
//! for (dim, pnl) in report.top_contributors(5) {
//!     println!("  {}: ${:.2}", dim, pnl);
//! }
//! ```

pub mod attribution;
pub mod journal;
pub mod trading;

// Re-export main types for convenience
pub use attribution::{
    AttributionAnalyzer, AttributionConfig, AttributionEntry, AttributionReport, Direction,
    MarketRegime, TimeOfDay, TimePeriod,
};
pub use journal::{JournalConfig, JournalStats, TradeDirection, TradeEntry, TradeJournal};
pub use trading::{MetricsCalculator, MetricsConfig, TradingMetrics};
