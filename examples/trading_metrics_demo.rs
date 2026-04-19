//! Trading Metrics Demo
//!
//! Demonstrates the usage of Octane's trading metrics module for analyzing
//! trading strategy performance.
//!
//! Run with: `cargo run --example trading_metrics_demo`

use octane_rs::metrics::{
    AttributionAnalyzer, AttributionConfig, Direction, JournalConfig, MarketRegime,
    MetricsCalculator, MetricsConfig, TradeDirection, TradeJournal,
};

fn main() {
    println!("=== Octane Trading Metrics Demo ===\n");

    // Part 1: Computing comprehensive trading metrics
    demo_trading_metrics();

    println!("\n{}\n", "=".repeat(60));

    // Part 2: Using trade journal
    demo_trade_journal();

    println!("\n{}\n", "=".repeat(60));

    // Part 3: Performance attribution
    demo_attribution();
}

fn demo_trading_metrics() {
    println!("Part 1: Trading Metrics Calculation");
    println!("{}", "-".repeat(60));

    // Configure metrics for daily trading
    let config = MetricsConfig::new(252.0) // 252 trading days per year
        .risk_free_rate(0.02) // 2% annual risk-free rate
        .var_confidence(0.95) // 95% confidence level for VaR
        .rolling_window(0); // 0 = cumulative (not rolling)

    let mut calc = MetricsCalculator::new(config);

    // Simulate a trading strategy with returns
    println!("\nSimulating trading returns...");
    let returns = [
        0.02, 0.01, -0.01, 0.03, 0.02, // Week 1
        -0.02, 0.01, 0.02, 0.01, -0.01, // Week 2
        0.04, 0.02, -0.03, 0.01, 0.02, // Week 3
        0.01, -0.01, 0.02, 0.03, 0.01, // Week 4
    ];

    for (i, &ret) in returns.iter().enumerate() {
        calc.add_return(ret);
        if (i + 1) % 5 == 0 {
            println!(
                "  Week {}: Cumulative return = {:.2}%",
                (i + 1) / 5,
                calc.total_return() * 100.0
            );
        }
    }

    // Simulate completed trades
    println!("\nSimulating completed trades...");
    let trades = vec![100.0, -50.0, 150.0, -30.0, 200.0, 75.0, -40.0, 120.0];
    for &pnl in &trades {
        calc.add_trade(pnl);
    }

    // Get comprehensive metrics
    let metrics = calc.compute_metrics();

    println!("\n=== Performance Metrics ===");
    println!("Total Return:        {:.2}%", metrics.total_return * 100.0);
    println!(
        "Annualized Return:   {:.2}%",
        metrics.annualized_return * 100.0
    );
    println!(
        "Annualized Vol:      {:.2}%",
        metrics.annualized_volatility * 100.0
    );
    println!("Sharpe Ratio:        {:.2}", metrics.sharpe_ratio);
    println!("Sortino Ratio:       {:.2}", metrics.sortino_ratio);
    println!("Calmar Ratio:        {:.2}", metrics.calmar_ratio);

    println!("\n=== Risk Metrics ===");
    println!(
        "Max Drawdown:        {:.2}%",
        metrics.max_drawdown_pct * 100.0
    );
    println!(
        "VaR (95%):           {:.2}%",
        metrics.var_historical * 100.0
    );
    println!("CVaR (95%):          {:.2}%", metrics.cvar * 100.0);
    println!("Ulcer Index:         {:.2}", metrics.ulcer_index);

    println!("\n=== Trade Statistics ===");
    println!("Total Trades:        {}", metrics.num_trades);
    println!("Win Rate:            {:.1}%", metrics.win_rate * 100.0);
    println!("Profit Factor:       {:.2}", metrics.profit_factor);
    println!("Avg Win:             ${:.2}", metrics.avg_win);
    println!("Avg Loss:            ${:.2}", metrics.avg_loss);
    println!("Expectancy:          ${:.2}", metrics.expectancy);
}

fn demo_trade_journal() {
    println!("Part 2: Trade Journal");
    println!("{}", "-".repeat(60));

    let config = JournalConfig::default()
        .auto_tag_trades(true)
        .track_feature_attribution(true)
        .auto_flush_interval(0); // Disable auto-flush for demo

    let mut journal = TradeJournal::new(config);

    println!("\nOpening trades...");

    // Trade 1: Winning long trade
    let trade1 = journal.open_trade(
        "AAPL".to_string(),
        TradeDirection::Long,
        1640000000, // 2021-12-20 10:13:20 UTC
        150.0,
        10.0,
    );
    println!("  Opened trade {trade1} - AAPL Long @ $150.00");

    // Trade 2: Losing short trade
    let trade2 = journal.open_trade(
        "GOOGL".to_string(),
        TradeDirection::Short,
        1640003600,
        2800.0,
        5.0,
    );
    println!("  Opened trade {trade2} - GOOGL Short @ $2800.00");

    // Trade 3: Another winning long
    let trade3 = journal.open_trade(
        "AAPL".to_string(),
        TradeDirection::Long,
        1640007200,
        152.0,
        10.0,
    );
    println!("  Opened trade {trade3} - AAPL Long @ $152.00");

    println!("\nClosing trades...");

    // Close trade 1 - win
    if let Some(pnl) = journal.close_trade(trade1, 1640010800, 155.0) {
        println!("  Closed trade {trade1} - P&L: ${pnl:.2}");
    }

    // Close trade 2 - loss
    if let Some(pnl) = journal.close_trade(trade2, 1640014400, 2820.0) {
        println!("  Closed trade {trade2} - P&L: ${pnl:.2}");
    }

    // Close trade 3 - win
    if let Some(pnl) = journal.close_trade(trade3, 1640018000, 158.0) {
        println!("  Closed trade {trade3} - P&L: ${pnl:.2}");
    }

    // Get statistics
    let stats = journal.aggregate_stats();

    println!("\n=== Journal Statistics ===");
    println!("Total Trades:        {}", stats.total_trades);
    println!("Winning Trades:      {}", stats.num_wins);
    println!("Losing Trades:       {}", stats.num_losses);
    println!("Win Rate:            {:.1}%", stats.win_rate * 100.0);
    println!("Total P&L:           ${:.2}", stats.total_pnl);
    println!("Avg Win:             ${:.2}", stats.avg_win);
    println!("Avg Loss:            ${:.2}", stats.avg_loss);

    // Filter by tags
    println!("\n=== Filtered by Tags ===");
    let wins = journal.filter_by_tags(&["win".to_string()]);
    println!("Winning trades:      {}", wins.len());

    let aapl_trades = journal
        .get_all_trades()
        .iter()
        .filter(|t| t.symbol == "AAPL")
        .count();
    println!("AAPL trades:         {aapl_trades}");

    // Try exporting (commented out since we're in example)
    // journal.export_json("trades.json").unwrap();
    // journal.export_csv("trades.csv").unwrap();
    println!("\nTo export, use:");
    println!("  journal.export_json(\"trades.json\")");
    println!("  journal.export_csv(\"trades.csv\")");
}

fn demo_attribution() {
    println!("Part 3: Performance Attribution");
    println!("{}", "-".repeat(60));

    let config = AttributionConfig::default()
        .enable_asset_attribution(true)
        .enable_regime_attribution(true);

    let mut analyzer = AttributionAnalyzer::new(config);

    println!("\nAdding trades for attribution analysis...");

    // Trending market - long AAPL - win
    analyzer.add_trade(
        100.0,
        1640000000,
        Some("AAPL"),
        Some(Direction::Long),
        Some(MarketRegime::Trending),
        3600,
        None,
    );
    println!("  AAPL Long in Trending market: +$100");

    // Trending market - long GOOGL - win
    analyzer.add_trade(
        150.0,
        1640003600,
        Some("GOOGL"),
        Some(Direction::Long),
        Some(MarketRegime::Trending),
        7200,
        None,
    );
    println!("  GOOGL Long in Trending market: +$150");

    // Ranging market - short AAPL - loss
    analyzer.add_trade(
        -50.0,
        1640007200,
        Some("AAPL"),
        Some(Direction::Short),
        Some(MarketRegime::Ranging),
        1800,
        None,
    );
    println!("  AAPL Short in Ranging market: -$50");

    // Volatile market - long MSFT - small win
    analyzer.add_trade(
        30.0,
        1640010800,
        Some("MSFT"),
        Some(Direction::Long),
        Some(MarketRegime::Volatile),
        5400,
        None,
    );
    println!("  MSFT Long in Volatile market: +$30");

    // Trending market - long AAPL - win
    analyzer.add_trade(
        75.0,
        1640014400,
        Some("AAPL"),
        Some(Direction::Long),
        Some(MarketRegime::Trending),
        3600,
        None,
    );
    println!("  AAPL Long in Trending market: +$75");

    // Finalize and get report
    analyzer.finalize_report();
    let report = analyzer.get_report();

    println!("\n=== Attribution Report ===");
    println!("Total P&L:           ${:.2}", report.total_pnl);

    println!("\n--- By Asset ---");
    let mut assets: Vec<_> = report.asset_attribution.iter().collect();
    assets.sort_by(|a, b| b.1.pnl.partial_cmp(&a.1.pnl).unwrap());
    for (asset, entry) in assets {
        println!(
            "  {:<8} ${:>8.2} ({:>5.1}%) - {} trades",
            asset, entry.pnl, entry.contribution_pct, entry.num_trades
        );
    }

    println!("\n--- By Direction ---");
    for (dir, entry) in &report.direction_attribution {
        println!(
            "  {:<8} ${:>8.2} ({:>5.1}%) - {} trades, {:.1}% win rate",
            dir,
            entry.pnl,
            entry.contribution_pct,
            entry.num_trades,
            entry.win_rate * 100.0
        );
    }

    println!("\n--- By Market Regime ---");
    let mut regimes: Vec<_> = report.regime_attribution.iter().collect();
    regimes.sort_by(|a, b| b.1.pnl.partial_cmp(&a.1.pnl).unwrap());
    for (regime, entry) in regimes {
        println!(
            "  {:<12} ${:>8.2} ({:>5.1}%) - {} trades",
            regime, entry.pnl, entry.contribution_pct, entry.num_trades
        );
    }

    println!("\n--- Top Contributors ---");
    for (dim, pnl) in report.top_contributors(3) {
        println!("  {dim} - ${pnl:.2}");
    }
}
