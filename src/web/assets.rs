//! Embedded static frontend assets.
//!
//! The dashboard is a single self-contained HTML file (inline CSS + JS, no CDN)
//! compiled into the binary so the server has zero filesystem dependencies and
//! works fully offline on localhost.

/// The dashboard single-page app, served at `/`.
pub const INDEX_HTML: &str = include_str!("static/index.html");
