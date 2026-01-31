# Trading Metrics Module

Comprehensive financial metrics and analytics for algorithmic trading strategies in Octane.

## Overview

The metrics module provides three main components:

1. **Trading Metrics** (`trading.rs`) - Risk-adjusted returns and statistical analysis
2. **Trade Journal** (`journal.rs`) - Automatic trade logging with feature attribution
3. **Performance Attribution** (`attribution.rs`) - P&L breakdown by multiple dimensions

## Features

### Trading Metrics

Comprehensive financial metrics including:

- **Risk-Adjusted Returns**
  - Sharpe Ratio (annualized)
  - Sortino Ratio (downside deviation)
  - Calmar Ratio (return / max drawdown)
  - Information Ratio (vs benchmark)
  - Treynor Ratio (beta-adjusted)

- **Drawdown Analysis**
  - Maximum Drawdown (absolute and percentage)
  - Recovery Factor
  - Ulcer Index (downside volatility)

- **Trade Statistics**
  - Win Rate
  - Profit Factor (gross profit / gross loss)
  - Expectancy (average profit per trade)
  - Average Win / Average Loss
  - Risk-Reward Ratio

- **Risk Metrics**
  - Value at Risk (VaR) - Historical and Parametric
  - Conditional VaR (CVaR / Expected Shortfall)

All metrics support:
- **Streaming/online computation** for efficiency
- **Rolling window calculations**
- **Annualization** with configurable periods

### Trade Journal

Automatic logging of all trades with:
- Entry/exit timestamps, prices, and sizes
- P&L tracking per trade
- Feature attribution for decision analysis
- Automatic and manual trade tagging
- Market regime tracking
- Export to JSON/CSV
- Trade replay capability
- Filtering by tags, time, symbol

### Performance Attribution

Detailed P&L breakdown by:
- **Time Period**: Hourly, daily, weekly, monthly, quarterly, yearly
- **Asset/Symbol**: Per-instrument analysis
- **Market Regime**: Trending, ranging, volatile, quiet, custom
- **Trade Direction**: Long vs short performance
- **Time of Day**: Open, morning, midday, afternoon, close
- **Factor Exposure**: Custom factor attribution

## Usage Examples

### Computing Trading Metrics

```rust
use octane_rs::metrics::{MetricsConfig, MetricsCalculator};

// Configure for daily trading
let config = MetricsConfig::new(252.0) // 252 trading days per year
    .risk_free_rate(0.02)              // 2% annual
    .var_confidence(0.95)              // 95% VaR
    .rolling_window(0);                // Cumulative (not rolling)

let mut calc = MetricsCalculator::new(config);

// Add returns
calc.add_return(0.01);  // +1% return
calc.add_return(0.02);  // +2% return
calc.add_return(-0.01); // -1% return

// Add completed trades
calc.add_trade(100.0);  // Win
calc.add_trade(-50.0);  // Loss

// Get comprehensive metrics
let metrics = calc.compute_metrics();

println!("Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
println!("Win Rate: {:.2}%", metrics.win_rate * 100.0);
println!("Max Drawdown: {:.2}%", metrics.max_drawdown_pct * 100.0);
```

### Using Trade Journal

```rust
use octane_rs::metrics::{JournalConfig, TradeJournal, TradeDirection};

let config = JournalConfig::default()
    .auto_tag_trades(true)
    .track_feature_attribution(true);

let mut journal = TradeJournal::new(config)
    .with_output_path("trades.json".to_string());

// Open a trade
let trade_id = journal.open_trade(
    "AAPL".to_string(),
    TradeDirection::Long,
    1640000000, // Unix timestamp
    150.0,      // Entry price
    10.0,       // Position size
);

// Close the trade
journal.close_trade(trade_id, 1640003600, 155.0);

// Get statistics
let stats = journal.aggregate_stats();
println!("Total P&L: ${:.2}", stats.total_pnl);
println!("Win Rate: {:.1}%", stats.win_rate * 100.0);

// Export
journal.export_json("trades.json").unwrap();
journal.export_csv("trades.csv").unwrap();

// Filter trades
let wins = journal.filter_by_tags(&["win".to_string()]);
let recent = journal.filter_by_time(start_time, end_time);
```

### Performance Attribution

```rust
use octane_rs::metrics::{
    AttributionConfig, AttributionAnalyzer,
    Direction, MarketRegime
};

let config = AttributionConfig::default()
    .enable_asset_attribution(true)
    .enable_regime_attribution(true);

let mut analyzer = AttributionAnalyzer::new(config);

// Add trades with metadata
analyzer.add_trade(
    100.0,                          // P&L
    1640000000,                     // Timestamp
    Some("AAPL"),                   // Symbol
    Some(Direction::Long),          // Direction
    Some(MarketRegime::Trending),   // Market regime
    3600,                           // Duration (seconds)
    None,                           // Factor exposures
);

// Finalize and get report
analyzer.finalize_report();
let report = analyzer.get_report();

println!("Total P&L: ${:.2}", report.total_pnl);

// Analyze by dimension
for (asset, entry) in &report.asset_attribution {
    println!("{}: ${:.2} ({:.1}% of total)",
        asset, entry.pnl, entry.contribution_pct);
}

// Find top/worst contributors
let top = report.top_contributors(5);
let worst = report.worst_contributors(5);
```

### Rolling Window Metrics

```rust
// 30-day rolling window
let config = MetricsConfig::new(252.0)
    .rolling_window(30);

let mut calc = MetricsCalculator::new(config);

// Metrics are computed over the last 30 observations
for return_val in returns {
    calc.add_return(return_val);

    // Only uses last 30 returns
    let metrics = calc.compute_metrics();
    println!("30-day Sharpe: {:.2}", metrics.sharpe_ratio);
}
```

### Feature Attribution

```rust
use std::collections::HashMap;

// Open trade with features
let trade_id = journal.open_trade(...);

// Get trade for attribution
if let Some(trade) = journal.get_trade_mut(trade_id) {
    // Set entry features
    let features = vec![0.5, 0.3, -0.2, 0.8];
    let names = vec![
        "momentum".to_string(),
        "mean_reversion".to_string(),
        "volatility".to_string(),
        "volume".to_string(),
    ];
    trade.set_entry_features(features, names);

    // Set feature importance/attribution
    let mut attribution = HashMap::new();
    attribution.insert("momentum".to_string(), 0.6);
    attribution.insert("mean_reversion".to_string(), 0.3);
    attribution.insert("volatility".to_string(), 0.1);
    trade.set_feature_attribution(attribution);
}
```

### Factor Attribution

```rust
use std::collections::HashMap;

let config = AttributionConfig::default()
    .factor_names(vec![
        "momentum".to_string(),
        "value".to_string(),
        "quality".to_string(),
    ]);

let mut analyzer = AttributionAnalyzer::new(config);

// Add trade with factor exposures
let mut exposures = HashMap::new();
exposures.insert("momentum".to_string(), 0.7);
exposures.insert("value".to_string(), 0.2);
exposures.insert("quality".to_string(), 0.1);

analyzer.add_trade(
    100.0,      // P&L
    timestamp,
    None,
    None,
    None,
    duration,
    Some(exposures),
);

let report = analyzer.get_report();
// Factor attribution shows P&L contribution weighted by exposure
for (factor, contribution) in &report.factor_attribution {
    println!("{}: ${:.2}", factor, contribution);
}
```

## Design Principles

1. **Streaming Computation**: All metrics support online/incremental updates without storing full history
2. **Memory Efficiency**: Ring buffers for rolling windows, SoA (Structure of Arrays) layout
3. **Numerical Stability**: Welford's algorithm for variance, careful handling of edge cases
4. **Zero-Copy**: Minimal allocations, efficient tensor conversions
5. **Type Safety**: Strong typing for directions, regimes, time periods
6. **Serialization**: Full serde support for all types

## Implementation Details

### Online Algorithms

- **Mean/Variance**: Welford's online algorithm for numerically stable computation
- **Drawdown**: Single-pass computation with peak tracking
- **VaR/CVaR**: Efficient sorted insertion or histogram-based approximation
- **Sharpe/Sortino**: Incremental computation with downside deviation tracking

### Memory Usage

- Fixed memory overhead per metric (independent of history length)
- Rolling windows use ring buffers (VecDeque) with fixed capacity
- SoA layout minimizes cache misses

### Performance

- O(1) metric updates
- O(log n) for VaR/CVaR with sorted returns
- O(n log n) for finalization (sorting for percentiles)
- Parallelization opportunities in attribution analysis

## Integration with Octane

The metrics module integrates seamlessly with:

- **Environments**: Automatic reward tracking from `TradingEnv`
- **Algorithms**: Built-in metrics in all RL agents
- **Logging**: Export to TensorBoard, W&B via `MetricLogger`
- **Checkpointing**: Serialize/deserialize metrics state
- **TUI**: Real-time metrics visualization

## Testing

Comprehensive test coverage includes:

- Unit tests for each metric calculation
- Integration tests with realistic trading scenarios
- Edge case handling (empty data, zero variance, etc.)
- Numerical stability tests
- Serialization round-trip tests

Run tests with:
```bash
cargo test --lib metrics::
```

## Examples

See `examples/trading_metrics_demo.rs` for a complete demonstration:

```bash
cargo run --example trading_metrics_demo
```

## References

- Sharpe, W. F. (1966). "Mutual Fund Performance"
- Sortino, F. A. (1994). "Downside Risk"
- Young, T. W. (1991). "Calmar Ratio: A Smoother Tool"
- Martin, P. (1987). "The Ulcer Index"
- Artzner et al. (1999). "Coherent Measures of Risk"
