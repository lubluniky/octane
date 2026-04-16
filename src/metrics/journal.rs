//! Trade journal for logging and analyzing individual trades.
//!
//! Provides automatic logging of all trades with:
//! - Entry/exit timestamps, prices, and sizes
//! - P&L tracking per trade
//! - Feature attribution for decision analysis
//! - Trade tagging system
//! - Export to JSON/CSV
//! - Trade replay capability

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

/// A single trade entry in the journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEntry {
    /// Unique trade identifier.
    pub trade_id: u64,
    /// Asset/symbol traded.
    pub symbol: String,
    /// Trade direction.
    pub direction: TradeDirection,
    /// Entry timestamp (Unix timestamp).
    pub entry_time: i64,
    /// Entry price.
    pub entry_price: f64,
    /// Position size.
    pub size: f64,
    /// Exit timestamp (Unix timestamp).
    pub exit_time: Option<i64>,
    /// Exit price.
    pub exit_price: Option<f64>,
    /// Profit/Loss (absolute).
    pub pnl: Option<f64>,
    /// Profit/Loss (percentage).
    pub pnl_pct: Option<f64>,
    /// Commission/fees paid.
    pub commission: f64,
    /// Trade duration in seconds.
    pub duration_secs: Option<i64>,
    /// Observation features at entry.
    pub entry_features: Vec<f64>,
    /// Feature names for attribution.
    pub feature_names: Vec<String>,
    /// Feature importance/attribution scores.
    pub feature_attribution: HashMap<String, f64>,
    /// User-defined tags.
    pub tags: Vec<String>,
    /// Free-form notes.
    pub notes: String,
    /// Market regime at entry (e.g., "trending", "ranging", "volatile").
    pub market_regime: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

/// Trade direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeDirection {
    /// Long position (buy).
    Long,
    /// Short position (sell).
    Short,
}

impl TradeEntry {
    /// Create a new trade entry.
    pub fn new(
        trade_id: u64,
        symbol: String,
        direction: TradeDirection,
        entry_time: i64,
        entry_price: f64,
        size: f64,
    ) -> Self {
        Self {
            trade_id,
            symbol,
            direction,
            entry_time,
            entry_price,
            size,
            exit_time: None,
            exit_price: None,
            pnl: None,
            pnl_pct: None,
            commission: 0.0,
            duration_secs: None,
            entry_features: Vec::new(),
            feature_names: Vec::new(),
            feature_attribution: HashMap::new(),
            tags: Vec::new(),
            notes: String::new(),
            market_regime: None,
            metadata: HashMap::new(),
        }
    }

    /// Close the trade with exit information.
    pub fn close(&mut self, exit_time: i64, exit_price: f64) {
        self.exit_time = Some(exit_time);
        self.exit_price = Some(exit_price);
        self.duration_secs = Some(exit_time - self.entry_time);

        // Calculate P&L
        let price_change = match self.direction {
            TradeDirection::Long => exit_price - self.entry_price,
            TradeDirection::Short => self.entry_price - exit_price,
        };

        let gross_pnl = price_change * self.size;
        let net_pnl = gross_pnl - self.commission;

        self.pnl = Some(net_pnl);
        self.pnl_pct = Some(price_change / self.entry_price);
    }

    /// Check if trade is closed.
    pub fn is_closed(&self) -> bool {
        self.exit_time.is_some()
    }

    /// Add a tag to the trade.
    pub fn add_tag(&mut self, tag: String) {
        if !self.tags.contains(&tag) {
            self.tags.push(tag);
        }
    }

    /// Set feature attribution scores.
    pub fn set_feature_attribution(&mut self, attribution: HashMap<String, f64>) {
        self.feature_attribution = attribution;
    }

    /// Set entry features.
    pub fn set_entry_features(&mut self, features: Vec<f64>, names: Vec<String>) {
        self.entry_features = features;
        self.feature_names = names;
    }

    /// Set market regime.
    pub fn set_market_regime(&mut self, regime: String) {
        self.market_regime = Some(regime);
    }

    /// Add metadata entry.
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

/// Configuration for trade journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalConfig {
    /// Maximum number of trades to keep in memory.
    pub max_trades_in_memory: usize,
    /// Automatically flush to disk every N trades.
    pub auto_flush_interval: usize,
    /// Enable feature attribution tracking.
    pub track_feature_attribution: bool,
    /// Automatically tag trades based on outcomes.
    pub auto_tag_trades: bool,
}

impl Default for JournalConfig {
    fn default() -> Self {
        Self {
            max_trades_in_memory: 10000,
            auto_flush_interval: 100,
            track_feature_attribution: true,
            auto_tag_trades: true,
        }
    }
}

impl JournalConfig {
    /// Create new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set max trades in memory.
    pub fn max_trades_in_memory(mut self, max: usize) -> Self {
        self.max_trades_in_memory = max;
        self
    }

    /// Set auto flush interval.
    pub fn auto_flush_interval(mut self, interval: usize) -> Self {
        self.auto_flush_interval = interval;
        self
    }

    /// Enable/disable feature attribution tracking.
    pub fn track_feature_attribution(mut self, enabled: bool) -> Self {
        self.track_feature_attribution = enabled;
        self
    }

    /// Enable/disable auto tagging.
    pub fn auto_tag_trades(mut self, enabled: bool) -> Self {
        self.auto_tag_trades = enabled;
        self
    }
}

/// Trade journal for logging and analyzing trades.
pub struct TradeJournal {
    config: JournalConfig,
    trades: Vec<TradeEntry>,
    open_trades: HashMap<u64, usize>, // trade_id -> index in trades
    next_trade_id: u64,
    total_closed_trades: usize,
    output_path: Option<String>,
}

impl TradeJournal {
    /// Create a new trade journal.
    pub fn new(config: JournalConfig) -> Self {
        Self {
            config,
            trades: Vec::new(),
            open_trades: HashMap::new(),
            next_trade_id: 0,
            total_closed_trades: 0,
            output_path: None,
        }
    }

    /// Set output path for auto-flushing.
    pub fn with_output_path(mut self, path: String) -> Self {
        self.output_path = Some(path);
        self
    }

    /// Open a new trade.
    pub fn open_trade(
        &mut self,
        symbol: String,
        direction: TradeDirection,
        entry_time: i64,
        entry_price: f64,
        size: f64,
    ) -> u64 {
        let trade_id = self.next_trade_id;
        self.next_trade_id += 1;

        let mut trade = TradeEntry::new(trade_id, symbol, direction, entry_time, entry_price, size);

        // Auto-tag on entry
        if self.config.auto_tag_trades {
            match direction {
                TradeDirection::Long => trade.add_tag("long".to_string()),
                TradeDirection::Short => trade.add_tag("short".to_string()),
            }
        }

        let idx = self.trades.len();
        self.trades.push(trade);
        self.open_trades.insert(trade_id, idx);

        trade_id
    }

    /// Close an open trade.
    pub fn close_trade(&mut self, trade_id: u64, exit_time: i64, exit_price: f64) -> Option<f64> {
        if let Some(&idx) = self.open_trades.get(&trade_id) {
            let pnl = {
                let trade = &mut self.trades[idx];
                trade.close(exit_time, exit_price);

                // Auto-tag based on outcome
                if self.config.auto_tag_trades {
                    if let Some(pnl) = trade.pnl {
                        if pnl > 0.0 {
                            trade.add_tag("win".to_string());
                        } else if pnl < 0.0 {
                            trade.add_tag("loss".to_string());
                        } else {
                            trade.add_tag("breakeven".to_string());
                        }

                        // Tag large wins/losses
                        if let Some(pnl_pct) = trade.pnl_pct {
                            if pnl_pct.abs() > 0.05 {
                                trade.add_tag("large_move".to_string());
                            }
                        }
                    }
                }

                trade.pnl
            };

            self.open_trades.remove(&trade_id);
            self.total_closed_trades += 1;

            // Auto-flush if needed
            if self.config.auto_flush_interval > 0
                && self
                    .total_closed_trades
                    .is_multiple_of(self.config.auto_flush_interval)
            {
                let _ = self.flush_to_disk();
            }

            pnl
        } else {
            None
        }
    }

    /// Get a trade by ID.
    pub fn get_trade(&self, trade_id: u64) -> Option<&TradeEntry> {
        self.trades.iter().find(|t| t.trade_id == trade_id)
    }

    /// Get a mutable trade by ID.
    pub fn get_trade_mut(&mut self, trade_id: u64) -> Option<&mut TradeEntry> {
        self.trades.iter_mut().find(|t| t.trade_id == trade_id)
    }

    /// Get all trades.
    pub fn get_all_trades(&self) -> &[TradeEntry] {
        &self.trades
    }

    /// Get only closed trades.
    pub fn get_closed_trades(&self) -> Vec<&TradeEntry> {
        self.trades.iter().filter(|t| t.is_closed()).collect()
    }

    /// Get only open trades.
    pub fn get_open_trades(&self) -> Vec<&TradeEntry> {
        self.trades.iter().filter(|t| !t.is_closed()).collect()
    }

    /// Filter trades by tags.
    pub fn filter_by_tags(&self, tags: &[String]) -> Vec<&TradeEntry> {
        self.trades
            .iter()
            .filter(|t| tags.iter().any(|tag| t.tags.contains(tag)))
            .collect()
    }

    /// Filter trades by time range.
    pub fn filter_by_time(&self, start_time: i64, end_time: i64) -> Vec<&TradeEntry> {
        self.trades
            .iter()
            .filter(|t| t.entry_time >= start_time && t.entry_time <= end_time)
            .collect()
    }

    /// Calculate aggregate statistics.
    pub fn aggregate_stats(&self) -> JournalStats {
        let closed_trades = self.get_closed_trades();

        let total_pnl: f64 = closed_trades.iter().filter_map(|t| t.pnl).sum();

        let num_wins = closed_trades
            .iter()
            .filter(|t| t.pnl.is_some_and(|p| p > 0.0))
            .count();

        let num_losses = closed_trades
            .iter()
            .filter(|t| t.pnl.is_some_and(|p| p < 0.0))
            .count();

        let avg_win = if num_wins > 0 {
            closed_trades
                .iter()
                .filter_map(|t| t.pnl)
                .filter(|&p| p > 0.0)
                .sum::<f64>()
                / num_wins as f64
        } else {
            0.0
        };

        let avg_loss = if num_losses > 0 {
            closed_trades
                .iter()
                .filter_map(|t| t.pnl)
                .filter(|&p| p < 0.0)
                .sum::<f64>()
                / num_losses as f64
        } else {
            0.0
        };

        let avg_duration = if !closed_trades.is_empty() {
            closed_trades
                .iter()
                .filter_map(|t| t.duration_secs)
                .sum::<i64>() as f64
                / closed_trades.len() as f64
        } else {
            0.0
        };

        JournalStats {
            total_trades: closed_trades.len(),
            num_wins,
            num_losses,
            total_pnl,
            avg_win,
            avg_loss,
            avg_duration_secs: avg_duration,
            win_rate: if !closed_trades.is_empty() {
                num_wins as f64 / closed_trades.len() as f64
            } else {
                0.0
            },
        }
    }

    /// Export trades to JSON file.
    pub fn export_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.trades)?;
        Ok(())
    }

    /// Export trades to CSV file.
    pub fn export_csv<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writeln!(
            writer,
            "trade_id,symbol,direction,entry_time,entry_price,size,exit_time,exit_price,pnl,pnl_pct,commission,duration_secs,tags"
        )?;

        // Write trades
        for trade in &self.trades {
            writeln!(
                writer,
                "{},{},{:?},{},{},{},{},{},{},{},{},{},\"{}\"",
                trade.trade_id,
                trade.symbol,
                trade.direction,
                trade.entry_time,
                trade.entry_price,
                trade.size,
                trade.exit_time.map_or("".to_string(), |t| t.to_string()),
                trade.exit_price.map_or("".to_string(), |p| p.to_string()),
                trade.pnl.map_or("".to_string(), |p| p.to_string()),
                trade.pnl_pct.map_or("".to_string(), |p| p.to_string()),
                trade.commission,
                trade
                    .duration_secs
                    .map_or("".to_string(), |d| d.to_string()),
                trade.tags.join(";")
            )?;
        }

        Ok(())
    }

    /// Import trades from JSON file.
    pub fn import_json<P: AsRef<Path>>(&mut self, path: P) -> std::io::Result<()> {
        let file = File::open(path)?;
        let trades: Vec<TradeEntry> = serde_json::from_reader(file)?;

        for trade in trades {
            if trade.trade_id >= self.next_trade_id {
                self.next_trade_id = trade.trade_id + 1;
            }

            if !trade.is_closed() {
                let idx = self.trades.len();
                self.open_trades.insert(trade.trade_id, idx);
            } else {
                self.total_closed_trades += 1;
            }

            self.trades.push(trade);
        }

        Ok(())
    }

    /// Flush in-memory trades to disk (if output path is set).
    fn flush_to_disk(&self) -> std::io::Result<()> {
        if let Some(ref path) = self.output_path {
            self.export_json(path)?;
        }
        Ok(())
    }

    /// Clear all trades.
    pub fn clear(&mut self) {
        self.trades.clear();
        self.open_trades.clear();
        self.next_trade_id = 0;
        self.total_closed_trades = 0;
    }

    /// Get number of trades in journal.
    pub fn len(&self) -> usize {
        self.trades.len()
    }

    /// Check if journal is empty.
    pub fn is_empty(&self) -> bool {
        self.trades.is_empty()
    }
}

/// Aggregate statistics from the journal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalStats {
    /// Total number of closed trades.
    pub total_trades: usize,
    /// Number of winning trades.
    pub num_wins: usize,
    /// Number of losing trades.
    pub num_losses: usize,
    /// Total P&L.
    pub total_pnl: f64,
    /// Average win size.
    pub avg_win: f64,
    /// Average loss size.
    pub avg_loss: f64,
    /// Average trade duration in seconds.
    pub avg_duration_secs: f64,
    /// Win rate.
    pub win_rate: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trade_journal_basic() {
        let config = JournalConfig::default();
        let mut journal = TradeJournal::new(config);

        let trade_id =
            journal.open_trade("AAPL".to_string(), TradeDirection::Long, 1000, 150.0, 10.0);

        assert_eq!(journal.get_open_trades().len(), 1);
        assert_eq!(journal.get_closed_trades().len(), 0);

        let pnl = journal.close_trade(trade_id, 2000, 155.0);
        assert!(pnl.is_some());
        assert!(pnl.unwrap() > 0.0);

        assert_eq!(journal.get_open_trades().len(), 0);
        assert_eq!(journal.get_closed_trades().len(), 1);
    }

    #[test]
    fn test_trade_pnl_calculation() {
        let config = JournalConfig::default();
        let mut journal = TradeJournal::new(config);

        // Long trade
        let long_id =
            journal.open_trade("AAPL".to_string(), TradeDirection::Long, 1000, 100.0, 10.0);
        journal.close_trade(long_id, 2000, 110.0);

        let long_trade = journal.get_trade(long_id).unwrap();
        assert_eq!(long_trade.pnl.unwrap(), 100.0); // (110-100)*10

        // Short trade
        let short_id =
            journal.open_trade("AAPL".to_string(), TradeDirection::Short, 3000, 110.0, 10.0);
        journal.close_trade(short_id, 4000, 105.0);

        let short_trade = journal.get_trade(short_id).unwrap();
        assert_eq!(short_trade.pnl.unwrap(), 50.0); // (110-105)*10
    }

    #[test]
    fn test_auto_tagging() {
        let config = JournalConfig::default().auto_tag_trades(true);
        let mut journal = TradeJournal::new(config);

        let trade_id =
            journal.open_trade("AAPL".to_string(), TradeDirection::Long, 1000, 100.0, 10.0);

        let trade = journal.get_trade(trade_id).unwrap();
        assert!(trade.tags.contains(&"long".to_string()));

        journal.close_trade(trade_id, 2000, 110.0);

        let trade = journal.get_trade(trade_id).unwrap();
        assert!(trade.tags.contains(&"win".to_string()));
    }

    #[test]
    fn test_journal_stats() {
        let config = JournalConfig::default();
        let mut journal = TradeJournal::new(config);

        // Add some trades
        let id1 = journal.open_trade("AAPL".to_string(), TradeDirection::Long, 1000, 100.0, 10.0);
        journal.close_trade(id1, 2000, 110.0);

        let id2 = journal.open_trade("AAPL".to_string(), TradeDirection::Long, 3000, 110.0, 10.0);
        journal.close_trade(id2, 4000, 105.0);

        let stats = journal.aggregate_stats();
        assert_eq!(stats.total_trades, 2);
        assert_eq!(stats.num_wins, 1);
        assert_eq!(stats.num_losses, 1);
        assert!(stats.total_pnl > 0.0);
        assert_eq!(stats.win_rate, 0.5);
    }

    #[test]
    fn test_filter_by_tags() {
        let config = JournalConfig::default().auto_tag_trades(true);
        let mut journal = TradeJournal::new(config);

        let id1 = journal.open_trade("AAPL".to_string(), TradeDirection::Long, 1000, 100.0, 10.0);
        journal.close_trade(id1, 2000, 110.0);

        let id2 = journal.open_trade("AAPL".to_string(), TradeDirection::Long, 3000, 110.0, 10.0);
        journal.close_trade(id2, 4000, 105.0);

        let wins = journal.filter_by_tags(&["win".to_string()]);
        assert_eq!(wins.len(), 1);

        let losses = journal.filter_by_tags(&["loss".to_string()]);
        assert_eq!(losses.len(), 1);
    }
}
