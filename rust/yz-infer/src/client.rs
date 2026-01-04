//! `InferenceClient`: background-IO client for the v1 Rust↔Python inference protocol.
//!
//! Design goals (PRD E5S2):
//! - tickets + routing by `request_id`
//! - caps for in-flight + outbound queue (backpressure)
//! - latency histogram for observability
//! - support both TCP and Unix Domain Sockets (UDS)

use std::collections::HashMap;
use std::io;
use std::io::Write as _;
use std::net::{Shutdown, TcpStream, ToSocketAddrs};
use std::os::unix::net::UnixStream;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::codec::{decode_response_v1, encode_request_v1_into, DecodeError};
use crate::frame::{read_frame, write_frame, FrameError};
use crate::protocol::{InferRequestV1, InferResponseV1};

// region agent log
static DBG_TX: OnceLock<mpsc::Sender<String>> = OnceLock::new();
fn dbg_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("YZ_DEBUG_LOG").as_deref(), Ok("1" | "true" | "yes")))
}

fn dbg_start_writer_thread() -> Option<mpsc::Sender<String>> {
    let (tx, rx) = mpsc::channel::<String>();
    thread::Builder::new()
        .name("yz-infer-debuglog".to_string())
        .spawn(move || {
            let path = std::path::Path::new("/Users/andreashornqvist/code/yA0tzy/.cursor/debug.log");
            let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path);
            // Buffer + periodic flush: reduces per-log syscalls dramatically.
            let mut buf = f.as_mut().ok().map(|ff| std::io::BufWriter::new(ff));
            let mut n_since_flush: u32 = 0;
            let mut last_flush = Instant::now();
            while let Ok(line) = rx.recv() {
                if let Some(bw) = buf.as_mut() {
                    let _ = bw.write_all(line.as_bytes());
                    let _ = bw.write_all(b"\n");
                    n_since_flush += 1;
                    if n_since_flush >= 1024 || last_flush.elapsed() >= Duration::from_millis(250) {
                        let _ = bw.flush();
                        n_since_flush = 0;
                        last_flush = Instant::now();
                    }
                } else {
                    // If opening failed earlier, keep dropping logs silently.
                }
            }
        })
        .ok()?;
    Some(tx)
}

fn dbg_log(hypothesis_id: &str, location: &str, message: &str, data: serde_json::Value) {
    if !dbg_enabled() {
        return;
    }
    let payload = serde_json::json!({
        "timestamp": (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64),
        "sessionId": "debug-session",
        "runId": "pre-fix",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    });
    if let Ok(line) = serde_json::to_string(&payload) {
        let tx = DBG_TX
            .get_or_init(|| dbg_start_writer_thread().unwrap_or_else(|| tx_dropper()))
            .clone();
        let _ = tx.send(line);
    }
}

fn tx_dropper() -> mpsc::Sender<String> {
    // If we cannot spawn/open the writer, keep a sender that drops everything.
    let (tx, _rx) = mpsc::channel::<String>();
    tx
}
static CLIENT_SUBMIT_COUNTER: AtomicU64 = AtomicU64::new(0);
// Sample mask for very hot-path logs (1/1024).
static DBG_SAMPLE_MASK: u64 = 0x3FF;
// endregion agent log

#[derive(Debug, Error)]
pub enum ClientError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("frame error: {0}")]
    Frame(#[from] FrameError),
    #[error("decode error: {0}")]
    Decode(#[from] DecodeError),
    #[error("backpressure: {0}")]
    Backpressure(&'static str),
    #[error("client disconnected")]
    Disconnected,
    #[error("request timed out")]
    Timeout,
}

#[derive(Debug, Clone)]
pub struct ClientOptions {
    /// Maximum number of pending in-flight requests across this client connection.
    pub max_inflight_total: usize,
    /// Bounded outbound queue capacity in number of frames.
    pub max_outbound_queue: usize,
    /// Request id starting value (useful for tests).
    pub request_id_start: u64,
}

impl Default for ClientOptions {
    fn default() -> Self {
        Self {
            // Reduced from 4096 to prevent flooding the inference server.
            // With N workers, total system inflight = N * max_inflight_total.
            // Server healthy capacity is ~50-150 requests at target RTT.
            // Default 64 means 10 workers = 640 max system inflight (reasonable headroom).
            max_inflight_total: 64,
            max_outbound_queue: 256,
            request_id_start: 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct LatencySummary {
    pub count: u64,
    pub min_us: u64,
    pub max_us: u64,
    pub mean_us: f64,
    /// Approximate percentiles (computed from buckets).
    pub p50_us: u64,
    pub p95_us: u64,
}

#[derive(Debug, Clone)]
pub struct LatencyHistogramSnapshot {
    /// Log2 buckets over microseconds. Bucket `b` covers `[2^b, 2^(b+1))` us.
    pub buckets: Vec<u64>,
    pub summary: LatencySummary,
}

#[derive(Debug, Clone)]
pub struct ClientStatsSnapshot {
    pub inflight: usize,
    pub sent: u64,
    pub received: u64,
    pub errors: u64,
    pub latency_us: LatencyHistogramSnapshot,
}

#[derive(Debug)]
struct PendingEntry {
    start: Instant,
    tx: mpsc::Sender<Result<InferResponseV1, ClientError>>,
}

#[derive(Debug)]
pub struct Ticket {
    pub request_id: u64,
    rx: mpsc::Receiver<Result<InferResponseV1, ClientError>>,
}

impl Ticket {
    pub fn recv(&self) -> Result<InferResponseV1, ClientError> {
        match self.rx.recv() {
            Ok(r) => r,
            Err(_) => Err(ClientError::Disconnected),
        }
    }

    pub fn recv_timeout(&self, timeout: Duration) -> Result<InferResponseV1, ClientError> {
        match self.rx.recv_timeout(timeout) {
            Ok(r) => r,
            Err(mpsc::RecvTimeoutError::Timeout) => Err(ClientError::Timeout),
            Err(mpsc::RecvTimeoutError::Disconnected) => Err(ClientError::Disconnected),
        }
    }

    pub fn try_recv(&self) -> Result<Option<InferResponseV1>, ClientError> {
        match self.rx.try_recv() {
            Ok(r) => Ok(Some(r?)),
            Err(mpsc::TryRecvError::Empty) => Ok(None),
            Err(mpsc::TryRecvError::Disconnected) => Err(ClientError::Disconnected),
        }
    }
}

#[derive(Debug)]
pub struct InferenceClient {
    next_request_id: AtomicU64,
    opts: ClientOptions,

    inflight: Arc<AtomicUsize>,
    pending: Arc<Mutex<HashMap<u64, PendingEntry>>>,

    /// Coalesced "some response arrived" signal used by outer loops to avoid
    /// fixed-interval polling sleeps (which can add large response-consumption latency).
    ///
    /// Bounded to 1 so multiple responses collapse into a single wake-up token.
    progress_rx: Mutex<mpsc::Receiver<()>>,

    outbound_tx: Option<mpsc::SyncSender<Vec<u8>>>,
    pool_tx: mpsc::SyncSender<Vec<u8>>,
    pool_rx: Mutex<mpsc::Receiver<Vec<u8>>>,
    shutdown_stream: Stream,
    shutdown: Arc<AtomicBool>,
    reader_handle: Option<JoinHandle<()>>,
    writer_handle: Option<JoinHandle<()>>,

    stats: Arc<Mutex<Stats>>,
}

impl InferenceClient {
    pub fn connect_tcp<A: ToSocketAddrs>(
        addr: A,
        opts: ClientOptions,
    ) -> Result<Self, ClientError> {
        let stream = TcpStream::connect(addr)?;
        stream.set_nodelay(true).ok();
        Self::from_stream(Stream::Tcp(stream), opts)
    }

    pub fn connect_uds(
        path: impl AsRef<std::path::Path>,
        opts: ClientOptions,
    ) -> Result<Self, ClientError> {
        let stream = UnixStream::connect(path)?;
        Self::from_stream(Stream::Uds(stream), opts)
    }

    fn from_stream(stream: Stream, opts: ClientOptions) -> Result<Self, ClientError> {
        let pending = Arc::new(Mutex::new(HashMap::<u64, PendingEntry>::new()));
        let stats = Arc::new(Mutex::new(Stats::default()));
        let shutdown = Arc::new(AtomicBool::new(false));
        let inflight = Arc::new(AtomicUsize::new(0));

        let (outbound_tx, outbound_rx) = mpsc::sync_channel::<Vec<u8>>(opts.max_outbound_queue);
        let (pool_tx, pool_rx) = mpsc::sync_channel::<Vec<u8>>(opts.max_outbound_queue);
        let (progress_tx, progress_rx) = mpsc::sync_channel::<()>(1);

        let reader_stream = stream.try_clone()?;
        let writer_stream = stream.try_clone()?;

        let pending_r = Arc::clone(&pending);
        let pending_w = Arc::clone(&pending);
        let shutdown_r = Arc::clone(&shutdown);
        let shutdown_w = Arc::clone(&shutdown);
        let stats_r = Arc::clone(&stats);
        let stats_w = Arc::clone(&stats);
        let inflight_r = Arc::clone(&inflight);
        let inflight_w = Arc::clone(&inflight);
        let progress_tx_r = progress_tx.clone();
        let progress_tx_w = progress_tx.clone();

        let reader_handle = thread::spawn(move || {
            reader_loop(
                reader_stream,
                pending_r,
                inflight_r,
                shutdown_r,
                stats_r,
                progress_tx_r,
            );
        });
        let pool_tx_writer = pool_tx.clone();
        let writer_handle = thread::spawn(move || {
            writer_loop(
                writer_stream,
                outbound_rx,
                pool_tx_writer,
                pending_w,
                inflight_w,
                shutdown_w,
                stats_w,
                progress_tx_w,
            );
        });

        Ok(Self {
            next_request_id: AtomicU64::new(opts.request_id_start),
            opts,
            inflight,
            pending,
            progress_rx: Mutex::new(progress_rx),
            outbound_tx: Some(outbound_tx),
            pool_tx,
            pool_rx: Mutex::new(pool_rx),
            shutdown_stream: stream,
            shutdown,
            reader_handle: Some(reader_handle),
            writer_handle: Some(writer_handle),
            stats,
        })
    }

    /// Wait until at least one response (or a disconnect/error) has been observed, or timeout.
    ///
    /// This is used by scheduler loops to avoid fixed sleeps that can add large *extra* latency
    /// between "response arrived" and "response consumed".
    pub fn wait_for_progress(&self, timeout: Duration) {
        let rx = self.progress_rx.lock().unwrap();
        let _ = rx.recv_timeout(timeout);
    }

    /// Submit a request for evaluation.
    ///
    /// Note: the client overwrites `req.request_id` with a unique id and returns it in the ticket.
    pub fn submit(&self, mut req: InferRequestV1) -> Result<Ticket, ClientError> {
        let t0 = Instant::now();
        // Backpressure: inflight cap.
        if !self.try_acquire_inflight() {
            self.stats_lock().on_error();
            return Err(ClientError::Backpressure("max_inflight_total exceeded"));
        }

        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        req.request_id = request_id;

        let t_ch0 = Instant::now();
        let (tx, rx) = mpsc::channel::<Result<InferResponseV1, ClientError>>();
        let ch_ms = t_ch0.elapsed().as_secs_f64() * 1000.0;
        {
            let t_ins0 = Instant::now();
            let mut g = self.pending.lock().unwrap();
            g.insert(
                request_id,
                PendingEntry {
                    start: Instant::now(),
                    tx,
                },
            );
            let ins_ms = t_ins0.elapsed().as_secs_f64() * 1000.0;
            // region agent log
            // If insert itself is slow, log it; otherwise rely on total breakdown below.
            if ins_ms >= 1.0 {
                dbg_log(
                    "H_client_breakdown",
                    "rust/yz-infer/src/client.rs:InferenceClient::submit",
                    "slow pending.insert",
                    serde_json::json!({ "insert_ms": ins_ms, "pending_len": g.len() }),
                );
            }
            // endregion agent log
        }

        let (mut payload, reused) = match self.pool_rx.lock().unwrap().try_recv() {
            Ok(mut b) => {
                b.clear();
                (b, true)
            }
            Err(_) => (Vec::with_capacity(512), false),
        };

        let t_enc0 = Instant::now();
        encode_request_v1_into(&mut payload, &req);
        let enc_ms = t_enc0.elapsed().as_secs_f64() * 1000.0;

        let tx = self.outbound_tx.as_ref().ok_or(ClientError::Disconnected)?;
        let t_send0 = Instant::now();
        let r = tx.try_send(payload);
        let send_ms = t_send0.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // region agent log
        // Outlier-only, but captures the full breakdown so we can optimize with confidence.
        if total_ms >= 1.0 {
            dbg_log(
                "H_client_breakdown",
                "rust/yz-infer/src/client.rs:InferenceClient::submit",
                "submit breakdown",
                serde_json::json!({
                    "total_ms": total_ms,
                    "channel_ms": ch_ms,
                    "encode_ms": enc_ms,
                    "try_send_ms": send_ms,
                    "reused_buf": reused,
                    "features_len": req.features.len(),
                    "legal_len": req.legal_mask.len(),
                }),
            );
        }
        // endregion agent log

        match r {
            Ok(()) => {
                self.stats_lock().on_sent();
                // region agent log
                if dbg_enabled() {
                    let n = CLIENT_SUBMIT_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
                    if enc_ms >= 0.5 || (n % 50_000 == 0) {
                        dbg_log(
                            "H_codec",
                            "rust/yz-infer/src/client.rs:InferenceClient::submit",
                            "encode_request_v1 timing",
                            serde_json::json!({
                                "n": n,
                                "encode_ms": enc_ms,
                                "features_len": req.features.len(),
                                "legal_len": req.legal_mask.len(),
                                "reused_buf": reused,
                            }),
                        );
                    }
                }
                // endregion agent log
                Ok(Ticket { request_id, rx })
            }
            Err(mpsc::TrySendError::Full(payload)) => {
                // Return buffer to pool best-effort to avoid allocator churn.
                let _ = self.pool_tx.try_send(payload);
                self.remove_pending_with_error(
                    request_id,
                    ClientError::Backpressure("outbound queue full"),
                );
                Err(ClientError::Backpressure("outbound queue full"))
            }
            Err(mpsc::TrySendError::Disconnected(payload)) => {
                let _ = self.pool_tx.try_send(payload);
                self.remove_pending_with_error(request_id, ClientError::Disconnected);
                Err(ClientError::Disconnected)
            }
        }
    }

    pub fn stats_snapshot(&self) -> ClientStatsSnapshot {
        let inflight = self.inflight.load(Ordering::Relaxed);
        let s = self.stats.lock().unwrap().snapshot();
        ClientStatsSnapshot {
            inflight,
            sent: s.sent,
            received: s.received,
            errors: s.errors,
            latency_us: s.latency_us,
        }
    }

    fn try_acquire_inflight(&self) -> bool {
        let cap = self.opts.max_inflight_total;
        let mut cur = self.inflight.load(Ordering::Relaxed);
        loop {
            if cur >= cap {
                return false;
            }
            match self.inflight.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(next) => cur = next,
            }
        }
    }

    fn release_inflight(&self) {
        self.inflight.fetch_sub(1, Ordering::SeqCst);
    }

    fn remove_pending_with_error(&self, request_id: u64, err: ClientError) {
        let entry = { self.pending.lock().unwrap().remove(&request_id) };
        if let Some(e) = entry {
            // best-effort send
            let _ = e.tx.send(Err(err));
        }
        self.release_inflight();
        self.stats_lock().on_error();
    }

    fn stats_lock(&self) -> std::sync::MutexGuard<'_, Stats> {
        self.stats.lock().unwrap()
    }
}

impl Drop for InferenceClient {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        self.shutdown_stream.shutdown();
        // Dropping outbound_tx closes writer loop.
        self.outbound_tx.take();
        // Best-effort join threads; ignore panics.
        self.reader_handle.take().map(|h| h.join().ok());
        self.writer_handle.take().map(|h| h.join().ok());
        // Drain any remaining pending with Disconnected.
        fail_all_pending(&self.pending, &self.inflight, || ClientError::Disconnected);
    }
}

#[derive(Debug)]
enum Stream {
    Tcp(TcpStream),
    Uds(UnixStream),
}

impl Stream {
    fn try_clone(&self) -> io::Result<Self> {
        match self {
            Stream::Tcp(s) => Ok(Stream::Tcp(s.try_clone()?)),
            Stream::Uds(s) => Ok(Stream::Uds(s.try_clone()?)),
        }
    }

    fn shutdown(&self) {
        match self {
            Stream::Tcp(s) => {
                let _ = s.shutdown(Shutdown::Both);
            }
            Stream::Uds(s) => {
                let _ = s.shutdown(Shutdown::Both);
            }
        }
    }
}

fn reader_loop(
    mut stream: Stream,
    pending: Arc<Mutex<HashMap<u64, PendingEntry>>>,
    inflight: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<Mutex<Stats>>,
    progress_tx: mpsc::SyncSender<()>,
) {
    loop {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        let payload = match read_frame(&mut stream_as_read(&mut stream)) {
            Ok(p) => p,
            Err(e) => {
                stats.lock().unwrap().on_error();
                stream.shutdown();
                let _ = e;
                let _ = progress_tx.try_send(());
                fail_all_pending(&pending, &inflight, || ClientError::Disconnected);
                break;
            }
        };
        let resp = match decode_response_v1(&payload) {
            Ok(r) => r,
            Err(e) => {
                stats.lock().unwrap().on_error();
                stream.shutdown();
                let _ = e;
                let _ = progress_tx.try_send(());
                fail_all_pending(&pending, &inflight, || ClientError::Disconnected);
                break;
            }
        };

        let (entry, pending_len_after) = {
            let mut g = pending.lock().unwrap();
            let e = g.remove(&resp.request_id);
            let len = g.len();
            (e, len)
        };
        if let Some(e) = entry {
            let dt = e.start.elapsed();
            stats.lock().unwrap().on_received(dt);
            inflight.fetch_sub(1, Ordering::SeqCst);
            // Wake any outer loop waiting for "some response arrived".
            let _ = progress_tx.try_send(());
            // region agent log
            // RTT log (very hot): sample heavily + keep only extreme outliers.
            // This measures *submit->response dispatch* on the Rust side.
            if dbg_enabled() {
                let dt_ms = dt.as_secs_f64() * 1000.0;
                let rid = resp.request_id;
                let sampled = (rid & DBG_SAMPLE_MASK) == 0;
                // NOTE: Logging itself has previously been a bottleneck. Keep volume low:
                // - always log only extreme tails
                // - otherwise log only sampled requests
                if dt_ms >= 200.0 || sampled {
                    dbg_log(
                        "H_rtt",
                        "rust/yz-infer/src/client.rs:reader_loop",
                        "slow request rtt",
                        serde_json::json!({
                            "request_id": rid,
                            "rtt_ms": dt_ms,
                            "sampled": sampled,
                            "pending_len_after": pending_len_after,
                            "inflight": inflight.load(Ordering::Relaxed),
                            "payload_len": payload.len(),
                        }),
                    );
                }
            }
            // endregion agent log
            let _ = e.tx.send(Ok(resp));
        } else {
            // Unknown request_id; ignore but count as error for observability.
            stats.lock().unwrap().on_error();
        }
    }
}

fn writer_loop(
    mut stream: Stream,
    outbound_rx: mpsc::Receiver<Vec<u8>>,
    pool_tx: mpsc::SyncSender<Vec<u8>>,
    pending: Arc<Mutex<HashMap<u64, PendingEntry>>>,
    inflight: Arc<AtomicUsize>,
    shutdown: Arc<AtomicBool>,
    stats: Arc<Mutex<Stats>>,
    progress_tx: mpsc::SyncSender<()>,
) {
    while let Ok(mut payload) = outbound_rx.recv() {
        if shutdown.load(Ordering::Relaxed) {
            break;
        }
        let t0 = Instant::now();
        if let Err(e) = write_frame(&mut stream_as_write(&mut stream), &payload) {
            stats.lock().unwrap().on_error();
            stream.shutdown();
            let _ = e;
            let _ = progress_tx.try_send(());
            fail_all_pending(&pending, &inflight, || ClientError::Disconnected);
            break;
        }
        // region agent log
        // Write log (hot): sample heavily + keep true outliers.
        if dbg_enabled() {
            let write_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let sampled = (CLIENT_SUBMIT_COUNTER.load(Ordering::Relaxed) & DBG_SAMPLE_MASK) == 0;
            if write_ms >= 10.0 || sampled {
                dbg_log(
                    "H_write_frame",
                    "rust/yz-infer/src/client.rs:writer_loop",
                    "slow write_frame",
                    serde_json::json!({
                        "write_ms": write_ms,
                        "sampled": sampled,
                        "payload_len": payload.len(),
                        "pending_len": pending.lock().unwrap().len(),
                        "inflight": inflight.load(Ordering::Relaxed),
                    }),
                );
            }
        }
        // endregion agent log
        payload.clear();
        let _ = pool_tx.try_send(payload);
    }
}

fn fail_all_pending<F: Fn() -> ClientError>(
    pending: &Arc<Mutex<HashMap<u64, PendingEntry>>>,
    inflight: &AtomicUsize,
    mk_err: F,
) {
    let entries = {
        let mut g = pending.lock().unwrap();
        let mut v = Vec::with_capacity(g.len());
        for (_, e) in g.drain() {
            v.push(e);
        }
        v
    };
    if !entries.is_empty() {
        let n = entries.len();
        let mut cur = inflight.load(Ordering::Relaxed);
        loop {
            let next = cur.saturating_sub(n);
            match inflight.compare_exchange_weak(cur, next, Ordering::SeqCst, Ordering::Relaxed) {
                Ok(_) => break,
                Err(v) => cur = v,
            }
        }
    }
    for e in entries {
        let _ = e.tx.send(Err(mk_err()));
    }
}

fn stream_as_read(s: &mut Stream) -> &mut dyn io::Read {
    match s {
        Stream::Tcp(t) => t,
        Stream::Uds(u) => u,
    }
}

fn stream_as_write(s: &mut Stream) -> &mut dyn io::Write {
    match s {
        Stream::Tcp(t) => t,
        Stream::Uds(u) => u,
    }
}

#[derive(Debug, Default)]
struct Stats {
    sent: u64,
    received: u64,
    errors: u64,
    latency: LatencyHistogram,
}

impl Stats {
    fn on_sent(&mut self) {
        self.sent += 1;
    }

    fn on_received(&mut self, dt: Duration) {
        self.received += 1;
        self.latency.record(dt);
    }

    fn on_error(&mut self) {
        self.errors += 1;
    }

    fn snapshot(&self) -> StatsSnapshot {
        StatsSnapshot {
            sent: self.sent,
            received: self.received,
            errors: self.errors,
            latency_us: self.latency.snapshot(),
        }
    }
}

#[derive(Debug, Clone)]
struct StatsSnapshot {
    sent: u64,
    received: u64,
    errors: u64,
    latency_us: LatencyHistogramSnapshot,
}

#[derive(Debug, Default)]
struct LatencyHistogram {
    // 0..=31 covers up to ~2^32 us. Bucket 0 is special-cased for [0,1)us.
    buckets: [u64; 32],
    count: u64,
    sum_us: u128,
    min_us: u64,
    max_us: u64,
}

impl LatencyHistogram {
    fn record(&mut self, dt: Duration) {
        let us = dt.as_micros().min(u128::from(u64::MAX)) as u64;
        let b = bucket_log2_us(us);
        self.buckets[b] += 1;
        self.count += 1;
        self.sum_us += us as u128;
        if self.count == 1 {
            self.min_us = us;
            self.max_us = us;
        } else {
            self.min_us = self.min_us.min(us);
            self.max_us = self.max_us.max(us);
        }
    }

    fn snapshot(&self) -> LatencyHistogramSnapshot {
        let count = self.count;
        let mean_us = if count == 0 {
            0.0
        } else {
            (self.sum_us as f64) / (count as f64)
        };
        let p50_us = self.approx_percentile_us(0.50);
        let p95_us = self.approx_percentile_us(0.95);

        LatencyHistogramSnapshot {
            buckets: self.buckets.to_vec(),
            summary: LatencySummary {
                count,
                min_us: if count == 0 { 0 } else { self.min_us },
                max_us: if count == 0 { 0 } else { self.max_us },
                mean_us,
                p50_us,
                p95_us,
            },
        }
    }

    fn approx_percentile_us(&self, q: f64) -> u64 {
        if self.count == 0 {
            return 0;
        }
        let target = ((self.count as f64) * q).ceil().max(1.0) as u64;
        let mut acc = 0u64;
        for (b, &n) in self.buckets.iter().enumerate() {
            acc += n;
            if acc >= target {
                // Return the upper bound of the bucket as a conservative estimate.
                return bucket_upper_bound_us(b);
            }
        }
        self.max_us
    }
}

fn bucket_log2_us(us: u64) -> usize {
    if us == 0 {
        return 0;
    }
    let lz = us.leading_zeros() as usize;
    let msb = 63usize.saturating_sub(lz);
    msb.min(31)
}

fn bucket_upper_bound_us(bucket: usize) -> u64 {
    if bucket == 0 {
        1
    } else if bucket >= 31 {
        u64::MAX
    } else {
        1u64 << (bucket + 1)
    }
}
