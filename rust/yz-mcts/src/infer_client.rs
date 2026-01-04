//! `yz-infer` integration boundary for `yz-mcts` (PRD E5S3).
//!
//! Responsible for:
//! - schema/action-space invariants
//! - encoding `GameState` -> features
//! - converting legal mask to protocol format
//! - submitting requests and receiving responses via `InferenceClient`

use std::path::Path;
use std::time::Duration;
use std::time::Instant;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use thiserror::Error;
use yz_core::{GameState, A};
use yz_features::{encode_state_v1, schema};
use yz_infer::protocol;
use yz_infer::{ClientError, ClientOptions, InferenceClient, Ticket};

// region agent log
fn dbg_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("YZ_DEBUG_LOG").as_deref(), Ok("1" | "true" | "yes")))
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
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("/Users/andreashornqvist/code/yA0tzy/.cursor/debug.log")
        {
            let _ = std::io::Write::write_all(&mut f, line.as_bytes());
            let _ = std::io::Write::write_all(&mut f, b"\n");
        }
    }
}
static SUBMIT_COUNTER: AtomicU64 = AtomicU64::new(0);
// endregion agent log

#[derive(Debug, Error)]
pub enum InferBackendError {
    #[error("protocol/schema mismatch: {0}")]
    Invariant(&'static str),
    #[error("client error: {0}")]
    Client(#[from] ClientError),
}

#[derive(Debug)]
pub struct InferBackend {
    client: InferenceClient,
    model_id: u32,
    feature_schema_id: u32,
}

impl InferBackend {
    pub fn connect_tcp(
        addr: impl std::net::ToSocketAddrs,
        model_id: u32,
        opts: ClientOptions,
    ) -> Result<Self, InferBackendError> {
        Self::check_invariants()?;
        Ok(Self {
            client: InferenceClient::connect_tcp(addr, opts)?,
            model_id,
            feature_schema_id: protocol::FEATURE_SCHEMA_ID_V1,
        })
    }

    #[cfg(unix)]
    pub fn connect_uds(
        path: impl AsRef<Path>,
        model_id: u32,
        opts: ClientOptions,
    ) -> Result<Self, InferBackendError> {
        Self::check_invariants()?;
        Ok(Self {
            client: InferenceClient::connect_uds(path, opts)?,
            model_id,
            feature_schema_id: protocol::FEATURE_SCHEMA_ID_V1,
        })
    }

    pub fn stats_snapshot(&self) -> yz_infer::ClientStatsSnapshot {
        self.client.stats_snapshot()
    }

    pub fn wait_for_progress(&self, timeout: Duration) {
        self.client.wait_for_progress(timeout);
    }

    pub fn submit(
        &self,
        state: &GameState,
        legal: &[bool; A],
    ) -> Result<Ticket, InferBackendError> {
        let t0 = Instant::now();
        let t_enc0 = Instant::now();
        let features = encode_state_v1(state);
        let enc_ms = t_enc0.elapsed().as_secs_f64() * 1000.0;

        let t_mask0 = Instant::now();
        let mut legal_mask = Vec::with_capacity(A);
        for &ok in legal.iter() {
            legal_mask.push(if ok { 1u8 } else { 0u8 });
        }
        let mask_ms = t_mask0.elapsed().as_secs_f64() * 1000.0;

        let t_req0 = Instant::now();
        let req = protocol::InferRequestV1 {
            request_id: 0, // overwritten by client
            model_id: self.model_id,
            feature_schema_id: self.feature_schema_id,
            features: features.to_vec(),
            legal_mask,
        };
        let t_submit0 = Instant::now();
        let out = self.client.submit(req)?;
        let req_build_ms = t_req0.elapsed().as_secs_f64() * 1000.0;
        let submit_ms = t_submit0.elapsed().as_secs_f64() * 1000.0;
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

        // region agent log
        if dbg_enabled() {
            let n = SUBMIT_COUNTER.fetch_add(1, Ordering::Relaxed) + 1;
            if total_ms >= 0.5 || (n % 50_000 == 0) {
                dbg_log(
                    "H_submit_alloc",
                    "rust/yz-mcts/src/infer_client.rs:InferBackend::submit",
                    "submit timing",
                    serde_json::json!({
                        "n": n,
                        "enc_ms": enc_ms,
                        "mask_ms": mask_ms,
                        "req_build_ms": req_build_ms,
                        "client_submit_ms": submit_ms,
                        "total_ms": total_ms,
                    }),
                );
            }
        }
        // endregion agent log

        Ok(out)
    }

    pub fn recv_timeout(
        &self,
        ticket: &Ticket,
        timeout: Duration,
    ) -> Result<protocol::InferResponseV1, InferBackendError> {
        Ok(ticket.recv_timeout(timeout)?)
    }

    fn check_invariants() -> Result<(), InferBackendError> {
        if protocol::ACTION_SPACE_A as usize != A {
            return Err(InferBackendError::Invariant("ACTION_SPACE_A != yz_core::A"));
        }
        if protocol::FEATURE_SCHEMA_ID_V1 != schema::FEATURE_SCHEMA_ID {
            return Err(InferBackendError::Invariant(
                "FEATURE_SCHEMA_ID_V1 != yz_features::schema::FEATURE_SCHEMA_ID",
            ));
        }
        if protocol::FEATURE_LEN_V1 as usize != schema::F {
            return Err(InferBackendError::Invariant(
                "FEATURE_LEN_V1 != yz_features::schema::F",
            ));
        }
        Ok(())
    }
}
