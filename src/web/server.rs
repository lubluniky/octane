//! A tiny, dependency-free HTTP server for the dashboard.
//!
//! Built on `std::net` with a thread-per-connection model — appropriate for a
//! localhost dashboard with a handful of pollers. The server only handles
//! `GET`, binds to a loopback address by default, and never panics on
//! malformed input (every parse step degrades to an error response).

use crate::web::assets::INDEX_HTML;
use crate::web::state::DashboardState;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Per-connection read/write timeout (frees threads held by stalled clients).
const IO_TIMEOUT: Duration = Duration::from_secs(15);
/// Maximum bytes accepted for the request line.
const MAX_REQUEST_LINE: u64 = 16 * 1024;
/// Maximum bytes accepted for a single header line.
const MAX_HEADER_LINE: u64 = 16 * 1024;
/// Maximum number of header lines accepted before giving up.
const MAX_HEADERS: usize = 100;
/// Backoff after a failed `accept()` to avoid busy-spinning on fd exhaustion.
const ACCEPT_BACKOFF: Duration = Duration::from_millis(20);

/// Bind/listen configuration for [`DashboardServer`].
#[derive(Debug, Clone)]
pub struct DashboardConfig {
    /// Host/interface to bind. Defaults to loopback `127.0.0.1`.
    pub host: String,
    /// TCP port to listen on. Defaults to `7878`.
    pub port: u16,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 7878,
        }
    }
}

/// The dashboard HTTP server, owning a clone of the shared state.
pub struct DashboardServer {
    state: DashboardState,
    config: DashboardConfig,
}

impl DashboardServer {
    /// Create a server bound to the given state and configuration.
    pub fn new(state: DashboardState, config: DashboardConfig) -> Self {
        Self { state, config }
    }

    /// The browser URL this server will be reachable at.
    pub fn url(&self) -> String {
        format!("http://{}:{}/", self.config.host, self.config.port)
    }

    /// Run the accept loop on the current thread (blocking).
    pub fn serve(self) -> std::io::Result<()> {
        let listener = TcpListener::bind((self.config.host.as_str(), self.config.port))?;
        for stream in listener.incoming() {
            match stream {
                Ok(stream) => spawn_handler(stream, self.state.clone()),
                Err(_) => thread::sleep(ACCEPT_BACKOFF),
            }
        }
        Ok(())
    }

    /// Bind immediately, then run the accept loop on a background thread.
    ///
    /// Returns the join handle once the socket is bound, so the caller can
    /// surface bind errors (e.g. port in use) synchronously.
    pub fn spawn(self) -> std::io::Result<JoinHandle<()>> {
        let listener = TcpListener::bind((self.config.host.as_str(), self.config.port))?;
        thread::Builder::new()
            .name("octane-web-accept".to_string())
            .spawn(move || {
                for stream in listener.incoming() {
                    match stream {
                        Ok(stream) => spawn_handler(stream, self.state.clone()),
                        Err(_) => thread::sleep(ACCEPT_BACKOFF),
                    }
                }
            })
    }
}

/// Spawn a per-connection handler thread, recovering from thread-creation
/// failure by dropping the connection rather than panicking (which under
/// `panic = "abort"` would take down the whole server).
fn spawn_handler(stream: TcpStream, state: DashboardState) {
    let spawned = thread::Builder::new()
        .name("octane-web-conn".to_string())
        .spawn(move || {
            let _ = handle_connection(stream, &state);
        });
    // On Err the closure (and its stream) is dropped, closing the connection.
    let _ = spawned;
}

/// Handle a single HTTP connection: parse the request line, route, respond.
fn handle_connection(mut stream: TcpStream, state: &DashboardState) -> std::io::Result<()> {
    // Bound how long a stalled client can hold this thread.
    let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
    let mut reader = BufReader::new(stream.try_clone()?);

    // Read the request line with a hard byte cap (a newline-less flood would
    // otherwise grow this String without bound).
    let mut request_line = String::new();
    if reader
        .by_ref()
        .take(MAX_REQUEST_LINE)
        .read_line(&mut request_line)?
        == 0
    {
        return Ok(());
    }

    // Drain the remaining headers — bounded in both per-line length and count.
    for _ in 0..MAX_HEADERS {
        let mut line = String::new();
        let n = reader.by_ref().take(MAX_HEADER_LINE).read_line(&mut line)?;
        if n == 0 || line == "\r\n" || line == "\n" {
            break;
        }
    }

    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("/");
    let path = target.split('?').next().unwrap_or("/");

    if method != "GET" {
        return write_response(
            &mut stream,
            405,
            "Method Not Allowed",
            "text/plain; charset=utf-8",
            b"405 Method Not Allowed",
        );
    }

    match path {
        "/" | "/index.html" => write_response(
            &mut stream,
            200,
            "OK",
            "text/html; charset=utf-8",
            INDEX_HTML.as_bytes(),
        ),
        "/api/state" => {
            let body = state.to_json();
            write_response(
                &mut stream,
                200,
                "OK",
                "application/json; charset=utf-8",
                body.as_bytes(),
            )
        }
        "/api/system" => {
            let body = state.system_json();
            write_response(
                &mut stream,
                200,
                "OK",
                "application/json; charset=utf-8",
                body.as_bytes(),
            )
        }
        "/healthz" => write_response(&mut stream, 200, "OK", "text/plain; charset=utf-8", b"ok"),
        "/favicon.ico" => write_response(&mut stream, 204, "No Content", "image/x-icon", b""),
        _ => write_response(
            &mut stream,
            404,
            "Not Found",
            "text/plain; charset=utf-8",
            b"404 Not Found",
        ),
    }
}

/// Write a complete HTTP/1.1 response and close the connection.
fn write_response(
    stream: &mut TcpStream,
    code: u16,
    status: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    // 204/304 responses must not carry Content-Length (RFC 7230 §3.3.2).
    let content_length = if code == 204 || code == 304 {
        String::new()
    } else {
        format!("Content-Length: {}\r\n", body.len())
    };
    let header = format!(
        "HTTP/1.1 {code} {status}\r\n\
         Content-Type: {content_type}\r\n\
         {content_length}\
         Cache-Control: no-store\r\n\
         Connection: close\r\n\r\n",
    );
    stream.write_all(header.as_bytes())?;
    if !body.is_empty() {
        stream.write_all(body)?;
    }
    stream.flush()
}
