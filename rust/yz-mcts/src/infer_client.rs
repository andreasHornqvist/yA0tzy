//! `yz-infer` integration boundary for `yz-mcts` (PRD E5S3).
//!
//! Responsible for:
//! - schema/action-space invariants
//! - encoding `GameState` -> features
//! - converting legal mask to protocol format
//! - submitting requests and receiving responses via `InferenceClient`

use std::path::Path;
use std::time::Duration;

use thiserror::Error;
use yz_core::{GameState, A};
use yz_features::{encode_state_v1, schema};
use yz_infer::protocol;
use yz_infer::{ClientError, ClientOptions, InferenceClient, Ticket};

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

    pub fn submit(
        &self,
        state: &GameState,
        legal: &[bool; A],
    ) -> Result<Ticket, InferBackendError> {
        let features = encode_state_v1(state);

        let mut legal_mask = Vec::with_capacity(A);
        for &ok in legal.iter() {
            legal_mask.push(if ok { 1u8 } else { 0u8 });
        }

        let req = protocol::InferRequestV1 {
            request_id: 0, // overwritten by client
            model_id: self.model_id,
            feature_schema_id: self.feature_schema_id,
            features: features.to_vec(),
            legal_mask,
        };
        Ok(self.client.submit(req)?)
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
