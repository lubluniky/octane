//! Performance attribution for analyzing P&L breakdown.
//!
//! Provides detailed analysis of profit/loss by:
//! - Time period (hourly, daily, weekly, monthly)
//! - Asset/symbol
//! - Market regime
//! - Trade direction (long/short)
//! - Time of day
//! - Factor exposure
//! - Contribution analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Time period granularity for attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimePeriod {
    /// Hourly breakdown.
    Hourly,
    /// Daily breakdown.
    Daily,
    /// Weekly breakdown.
    Weekly,
    /// Monthly breakdown.
    Monthly,
    /// Quarterly breakdown.
    Quarterly,
    /// Yearly breakdown.
    Yearly,
}

/// Market regime classification.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Trending upward.
    Trending,
    /// Ranging/sideways.
    Ranging,
    /// High volatility.
    Volatile,
    /// Low volatility.
    Quiet,
    /// Custom regime.
    Custom(String),
}

/// Trade direction for attribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// Long positions.
    Long,
    /// Short positions.
    Short,
}

/// Time of day classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TimeOfDay {
    /// Market open (first hour).
    Open,
    /// Morning session.
    Morning,
    /// Midday.
    Midday,
    /// Afternoon.
    Afternoon,
    /// Market close (last hour).
    Close,
    /// After hours.
    AfterHours,
}

impl TimeOfDay {
    /// Classify time of day from hour (0-23).
    pub fn from_hour(hour: u32) -> Self {
        match hour {
            9 => Self::Open,
            10..=11 => Self::Morning,
            12..=13 => Self::Midday,
            14..=15 => Self::Afternoon,
            16 => Self::Close,
            _ => Self::AfterHours,
        }
    }
}

/// Configuration for attribution analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionConfig {
    /// Enable time period attribution.
    pub enable_time_attribution: bool,
    /// Enable asset attribution.
    pub enable_asset_attribution: bool,
    /// Enable regime attribution.
    pub enable_regime_attribution: bool,
    /// Enable direction attribution.
    pub enable_direction_attribution: bool,
    /// Enable time of day attribution.
    pub enable_time_of_day_attribution: bool,
    /// Enable factor attribution.
    pub enable_factor_attribution: bool,
    /// Factor names for analysis.
    pub factor_names: Vec<String>,
}

impl Default for AttributionConfig {
    fn default() -> Self {
        Self {
            enable_time_attribution: true,
            enable_asset_attribution: true,
            enable_regime_attribution: true,
            enable_direction_attribution: true,
            enable_time_of_day_attribution: true,
            enable_factor_attribution: false,
            factor_names: Vec::new(),
        }
    }
}

impl AttributionConfig {
    /// Create new config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable time attribution.
    pub fn enable_time_attribution(mut self, enabled: bool) -> Self {
        self.enable_time_attribution = enabled;
        self
    }

    /// Enable asset attribution.
    pub fn enable_asset_attribution(mut self, enabled: bool) -> Self {
        self.enable_asset_attribution = enabled;
        self
    }

    /// Enable regime attribution.
    pub fn enable_regime_attribution(mut self, enabled: bool) -> Self {
        self.enable_regime_attribution = enabled;
        self
    }

    /// Enable factor attribution.
    pub fn enable_factor_attribution(mut self, enabled: bool) -> Self {
        self.enable_factor_attribution = enabled;
        self
    }

    /// Set factor names.
    pub fn factor_names(mut self, names: Vec<String>) -> Self {
        self.enable_factor_attribution = !names.is_empty();
        self.factor_names = names;
        self
    }
}

/// Attribution entry for a specific dimension.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionEntry {
    /// P&L for this dimension.
    pub pnl: f64,
    /// Number of trades.
    pub num_trades: usize,
    /// Win rate.
    pub win_rate: f64,
    /// Average trade P&L.
    pub avg_trade_pnl: f64,
    /// Contribution to total P&L (percentage).
    pub contribution_pct: f64,
    /// Exposure time (seconds).
    pub exposure_time_secs: i64,
}

impl AttributionEntry {
    /// Create a new attribution entry.
    pub fn new() -> Self {
        Self {
            pnl: 0.0,
            num_trades: 0,
            win_rate: 0.0,
            avg_trade_pnl: 0.0,
            contribution_pct: 0.0,
            exposure_time_secs: 0,
        }
    }

    /// Add a trade to this entry.
    pub fn add_trade(&mut self, pnl: f64, duration_secs: i64) {
        self.pnl += pnl;
        self.num_trades += 1;
        self.exposure_time_secs += duration_secs;

        if self.num_trades > 0 {
            self.avg_trade_pnl = self.pnl / self.num_trades as f64;
        }
    }

    /// Update win rate.
    pub fn update_win_rate(&mut self, num_wins: usize) {
        if self.num_trades > 0 {
            self.win_rate = num_wins as f64 / self.num_trades as f64;
        }
    }

    /// Update contribution percentage.
    pub fn update_contribution(&mut self, total_pnl: f64) {
        if total_pnl != 0.0 {
            self.contribution_pct = (self.pnl / total_pnl) * 100.0;
        }
    }
}

impl Default for AttributionEntry {
    fn default() -> Self {
        Self::new()
    }
}

/// Comprehensive attribution report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionReport {
    /// Total P&L.
    pub total_pnl: f64,
    /// Time period attribution.
    pub time_attribution: HashMap<String, AttributionEntry>,
    /// Asset attribution.
    pub asset_attribution: HashMap<String, AttributionEntry>,
    /// Regime attribution.
    pub regime_attribution: HashMap<String, AttributionEntry>,
    /// Direction attribution.
    pub direction_attribution: HashMap<String, AttributionEntry>,
    /// Time of day attribution.
    pub time_of_day_attribution: HashMap<String, AttributionEntry>,
    /// Factor attribution.
    pub factor_attribution: HashMap<String, f64>,
}

impl AttributionReport {
    /// Create a new empty report.
    pub fn new() -> Self {
        Self {
            total_pnl: 0.0,
            time_attribution: HashMap::new(),
            asset_attribution: HashMap::new(),
            regime_attribution: HashMap::new(),
            direction_attribution: HashMap::new(),
            time_of_day_attribution: HashMap::new(),
            factor_attribution: HashMap::new(),
        }
    }

    /// Get top contributors by P&L.
    pub fn top_contributors(&self, n: usize) -> Vec<(String, f64)> {
        let mut all_contributions: Vec<(String, f64)> = Vec::new();

        for (key, entry) in &self.asset_attribution {
            all_contributions.push((format!("asset:{}", key), entry.pnl));
        }

        for (key, entry) in &self.regime_attribution {
            all_contributions.push((format!("regime:{}", key), entry.pnl));
        }

        for (key, entry) in &self.direction_attribution {
            all_contributions.push((format!("direction:{}", key), entry.pnl));
        }

        all_contributions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all_contributions.into_iter().take(n).collect()
    }

    /// Get worst contributors by P&L.
    pub fn worst_contributors(&self, n: usize) -> Vec<(String, f64)> {
        let mut all_contributions: Vec<(String, f64)> = Vec::new();

        for (key, entry) in &self.asset_attribution {
            all_contributions.push((format!("asset:{}", key), entry.pnl));
        }

        for (key, entry) in &self.regime_attribution {
            all_contributions.push((format!("regime:{}", key), entry.pnl));
        }

        for (key, entry) in &self.direction_attribution {
            all_contributions.push((format!("direction:{}", key), entry.pnl));
        }

        all_contributions.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        all_contributions.into_iter().take(n).collect()
    }
}

impl Default for AttributionReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance attribution analyzer.
pub struct AttributionAnalyzer {
    config: AttributionConfig,
    report: AttributionReport,
}

impl AttributionAnalyzer {
    /// Create a new attribution analyzer.
    pub fn new(config: AttributionConfig) -> Self {
        Self {
            config,
            report: AttributionReport::new(),
        }
    }

    /// Add a trade for attribution.
    pub fn add_trade(
        &mut self,
        pnl: f64,
        timestamp: i64,
        symbol: Option<&str>,
        direction: Option<Direction>,
        regime: Option<MarketRegime>,
        duration_secs: i64,
        factor_exposures: Option<HashMap<String, f64>>,
    ) {
        self.report.total_pnl += pnl;

        // Time attribution
        if self.config.enable_time_attribution {
            self.add_time_attribution(pnl, timestamp, duration_secs);
        }

        // Asset attribution
        if self.config.enable_asset_attribution {
            if let Some(sym) = symbol {
                self.add_asset_attribution(pnl, sym, duration_secs);
            }
        }

        // Direction attribution
        if self.config.enable_direction_attribution {
            if let Some(dir) = direction {
                self.add_direction_attribution(pnl, dir, duration_secs);
            }
        }

        // Regime attribution
        if self.config.enable_regime_attribution {
            if let Some(reg) = regime {
                self.add_regime_attribution(pnl, reg, duration_secs);
            }
        }

        // Time of day attribution
        if self.config.enable_time_of_day_attribution {
            self.add_time_of_day_attribution(pnl, timestamp, duration_secs);
        }

        // Factor attribution
        if self.config.enable_factor_attribution {
            if let Some(exposures) = factor_exposures {
                self.add_factor_attribution(pnl, exposures);
            }
        }
    }

    /// Add time period attribution.
    fn add_time_attribution(&mut self, pnl: f64, timestamp: i64, duration_secs: i64) {
        // Convert timestamp to date string (simplified - assumes UTC)
        let days = timestamp / 86400;
        let date_key = format!("day_{}", days);

        let entry = self.report.time_attribution
            .entry(date_key)
            .or_insert_with(AttributionEntry::new);

        entry.add_trade(pnl, duration_secs);
    }

    /// Add asset attribution.
    fn add_asset_attribution(&mut self, pnl: f64, symbol: &str, duration_secs: i64) {
        let entry = self.report.asset_attribution
            .entry(symbol.to_string())
            .or_insert_with(AttributionEntry::new);

        entry.add_trade(pnl, duration_secs);
    }

    /// Add direction attribution.
    fn add_direction_attribution(&mut self, pnl: f64, direction: Direction, duration_secs: i64) {
        let key = match direction {
            Direction::Long => "long",
            Direction::Short => "short",
        };

        let entry = self.report.direction_attribution
            .entry(key.to_string())
            .or_insert_with(AttributionEntry::new);

        entry.add_trade(pnl, duration_secs);
    }

    /// Add regime attribution.
    fn add_regime_attribution(&mut self, pnl: f64, regime: MarketRegime, duration_secs: i64) {
        let key = match &regime {
            MarketRegime::Trending => "trending".to_string(),
            MarketRegime::Ranging => "ranging".to_string(),
            MarketRegime::Volatile => "volatile".to_string(),
            MarketRegime::Quiet => "quiet".to_string(),
            MarketRegime::Custom(s) => s.clone(),
        };

        let entry = self.report.regime_attribution
            .entry(key)
            .or_insert_with(AttributionEntry::new);

        entry.add_trade(pnl, duration_secs);
    }

    /// Add time of day attribution.
    fn add_time_of_day_attribution(&mut self, pnl: f64, timestamp: i64, duration_secs: i64) {
        let hour = ((timestamp % 86400) / 3600) as u32;
        let time_of_day = TimeOfDay::from_hour(hour);

        let key = format!("{:?}", time_of_day);
        let entry = self.report.time_of_day_attribution
            .entry(key)
            .or_insert_with(AttributionEntry::new);

        entry.add_trade(pnl, duration_secs);
    }

    /// Add factor attribution.
    fn add_factor_attribution(&mut self, pnl: f64, exposures: HashMap<String, f64>) {
        for (factor, exposure) in exposures {
            let contribution = self.report.factor_attribution
                .entry(factor)
                .or_insert(0.0);

            *contribution += pnl * exposure;
        }
    }

    /// Finalize report with contribution percentages.
    pub fn finalize_report(&mut self) {
        let total_pnl = self.report.total_pnl;

        // Update contributions for all dimensions
        for entry in self.report.time_attribution.values_mut() {
            entry.update_contribution(total_pnl);
        }

        for entry in self.report.asset_attribution.values_mut() {
            entry.update_contribution(total_pnl);
        }

        for entry in self.report.regime_attribution.values_mut() {
            entry.update_contribution(total_pnl);
        }

        for entry in self.report.direction_attribution.values_mut() {
            entry.update_contribution(total_pnl);
        }

        for entry in self.report.time_of_day_attribution.values_mut() {
            entry.update_contribution(total_pnl);
        }
    }

    /// Get the attribution report.
    pub fn get_report(&self) -> &AttributionReport {
        &self.report
    }

    /// Get mutable attribution report.
    pub fn get_report_mut(&mut self) -> &mut AttributionReport {
        &mut self.report
    }

    /// Reset the analyzer.
    pub fn reset(&mut self) {
        self.report = AttributionReport::new();
    }

    /// Export report to JSON.
    pub fn export_json(&self) -> serde_json::Result<String> {
        serde_json::to_string_pretty(&self.report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attribution_analyzer_basic() {
        let config = AttributionConfig::default();
        let mut analyzer = AttributionAnalyzer::new(config);

        analyzer.add_trade(
            100.0,
            1000,
            Some("AAPL"),
            Some(Direction::Long),
            Some(MarketRegime::Trending),
            3600,
            None,
        );

        analyzer.add_trade(
            -50.0,
            2000,
            Some("AAPL"),
            Some(Direction::Short),
            Some(MarketRegime::Ranging),
            1800,
            None,
        );

        analyzer.finalize_report();

        let report = analyzer.get_report();
        assert_eq!(report.total_pnl, 50.0);
        assert!(report.asset_attribution.contains_key("AAPL"));
        assert!(report.direction_attribution.contains_key("long"));
        assert!(report.direction_attribution.contains_key("short"));
    }

    #[test]
    fn test_asset_attribution() {
        let config = AttributionConfig::default();
        let mut analyzer = AttributionAnalyzer::new(config);

        analyzer.add_trade(100.0, 1000, Some("AAPL"), None, None, 3600, None);
        analyzer.add_trade(50.0, 2000, Some("AAPL"), None, None, 1800, None);
        analyzer.add_trade(-20.0, 3000, Some("GOOGL"), None, None, 1200, None);

        analyzer.finalize_report();

        let report = analyzer.get_report();
        let aapl_entry = report.asset_attribution.get("AAPL").unwrap();
        assert_eq!(aapl_entry.pnl, 150.0);
        assert_eq!(aapl_entry.num_trades, 2);

        let googl_entry = report.asset_attribution.get("GOOGL").unwrap();
        assert_eq!(googl_entry.pnl, -20.0);
        assert_eq!(googl_entry.num_trades, 1);
    }

    #[test]
    fn test_direction_attribution() {
        let config = AttributionConfig::default();
        let mut analyzer = AttributionAnalyzer::new(config);

        analyzer.add_trade(100.0, 1000, None, Some(Direction::Long), None, 3600, None);
        analyzer.add_trade(50.0, 2000, None, Some(Direction::Long), None, 1800, None);
        analyzer.add_trade(-30.0, 3000, None, Some(Direction::Short), None, 1200, None);

        analyzer.finalize_report();

        let report = analyzer.get_report();
        let long_entry = report.direction_attribution.get("long").unwrap();
        assert_eq!(long_entry.pnl, 150.0);
        assert_eq!(long_entry.num_trades, 2);

        let short_entry = report.direction_attribution.get("short").unwrap();
        assert_eq!(short_entry.pnl, -30.0);
        assert_eq!(short_entry.num_trades, 1);
    }

    #[test]
    fn test_factor_attribution() {
        let config = AttributionConfig::default().factor_names(vec![
            "momentum".to_string(),
            "mean_reversion".to_string(),
        ]);
        let mut analyzer = AttributionAnalyzer::new(config);

        let mut exposures = HashMap::new();
        exposures.insert("momentum".to_string(), 0.8);
        exposures.insert("mean_reversion".to_string(), 0.2);

        analyzer.add_trade(100.0, 1000, None, None, None, 3600, Some(exposures));

        let report = analyzer.get_report();
        assert_eq!(*report.factor_attribution.get("momentum").unwrap(), 80.0);
        assert_eq!(*report.factor_attribution.get("mean_reversion").unwrap(), 20.0);
    }

    #[test]
    fn test_top_contributors() {
        let config = AttributionConfig::default();
        let mut analyzer = AttributionAnalyzer::new(config);

        analyzer.add_trade(100.0, 1000, Some("AAPL"), None, None, 3600, None);
        analyzer.add_trade(50.0, 2000, Some("GOOGL"), None, None, 1800, None);
        analyzer.add_trade(-20.0, 3000, Some("MSFT"), None, None, 1200, None);

        analyzer.finalize_report();

        let report = analyzer.get_report();
        let top = report.top_contributors(2);

        assert_eq!(top.len(), 2);
        assert_eq!(top[0].0, "asset:AAPL");
        assert_eq!(top[0].1, 100.0);
    }
}
