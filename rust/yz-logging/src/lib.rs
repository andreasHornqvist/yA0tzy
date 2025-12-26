//! yz-logging: NDJSON events + metrics/tracing setup.
//!
//! v1 scope (PRD E7S3): append-only NDJSON logs for run post-mortems.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use serde::Serialize;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Minimal log schema versioning fields (PRD §10.3).
#[derive(Debug, Clone, Serialize)]
pub struct VersionInfoV1 {
    pub protocol_version: u32,
    pub feature_schema_id: u32,
    pub action_space_id: &'static str,
    pub ruleset_id: &'static str,
}

#[derive(Debug, Clone, Serialize)]
pub struct InferStatsV1 {
    pub inflight: u64,
    pub sent: u64,
    pub received: u64,
    pub errors: u64,
    pub latency_p50_us: u64,
    pub latency_p95_us: u64,
    pub latency_mean_us: f64,
}

#[derive(Debug, Clone, Serialize)]
pub struct IterationStatsEventV1 {
    pub event: &'static str,
    pub ts_ms: u64,
    pub v: VersionInfoV1,

    pub run_id: String,
    pub tick: u64,
    pub global_ply: u64,

    pub tasks: u64,
    pub completed_games: u64,

    pub steps: u64,
    pub would_block: u64,
    pub terminal: u64,

    pub infer: InferStatsV1,
}

#[derive(Debug, Clone, Serialize)]
pub struct PiSummaryV1 {
    pub entropy: f32,
    pub max_p: f32,
    pub argmax_a: u8,
}

#[derive(Debug, Clone, Serialize)]
pub struct MctsRootEventV1 {
    pub event: &'static str,
    pub ts_ms: u64,
    pub v: VersionInfoV1,

    pub run_id: String,
    pub global_ply: u64,

    pub game_id: u64,
    pub game_ply: u32,
    pub player_to_move: u8,
    pub rerolls_left: u8,
    pub dice: [u8; 5],

    pub chosen_action: u8,
    pub root_value: f32,
    pub fallbacks: u32,
    pub pending_count_max: u64,
    pub pending_collisions: u32,

    pub pi: PiSummaryV1,
}

#[derive(Debug)]
pub enum NdjsonError {
    Io(io::Error),
    Json(serde_json::Error),
}

impl From<io::Error> for NdjsonError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<serde_json::Error> for NdjsonError {
    fn from(e: serde_json::Error) -> Self {
        Self::Json(e)
    }
}

/// Append-only NDJSON writer.
///
/// Contract: each call writes exactly one JSON object followed by a newline.
pub struct NdjsonWriter {
    w: BufWriter<File>,
    lines_since_flush: u64,
    flush_every_lines: u64,
}

impl NdjsonWriter {
    /// Open a file for append. Creates it if it doesn't exist.
    pub fn open_append(path: impl AsRef<Path>) -> Result<Self, NdjsonError> {
        Self::open_append_with_flush(path, 0)
    }

    /// `flush_every_lines=0` disables periodic flushing.
    pub fn open_append_with_flush(
        path: impl AsRef<Path>,
        flush_every_lines: u64,
    ) -> Result<Self, NdjsonError> {
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .write(true)
            .open(path)?;
        Ok(Self {
            w: BufWriter::new(f),
            lines_since_flush: 0,
            flush_every_lines,
        })
    }

    pub fn write_event<T: Serialize>(&mut self, event: &T) -> Result<(), NdjsonError> {
        let mut buf = serde_json::to_vec(event)?;
        buf.push(b'\n');
        self.w.write_all(&buf)?;
        self.lines_since_flush += 1;
        if self.flush_every_lines > 0 && self.lines_since_flush >= self.flush_every_lines {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), NdjsonError> {
        self.w.flush()?;
        self.lines_since_flush = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    use serde_json::Value;

    fn read_ndjson_lenient(path: &Path) -> Vec<Value> {
        let s = fs::read_to_string(path).expect("read");
        let mut out = Vec::new();
        for line in s.lines() {
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                out.push(v);
            }
        }
        out
    }

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn writes_one_valid_json_object_per_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.ndjson");
        let mut w = NdjsonWriter::open_append(&path).unwrap();

        #[derive(Serialize)]
        struct E {
            event: &'static str,
            x: u32,
        }

        w.write_event(&E { event: "e", x: 1 }).unwrap();
        w.write_event(&E { event: "e", x: 2 }).unwrap();
        w.flush().unwrap();

        let vals = read_ndjson_lenient(&path);
        assert_eq!(vals.len(), 2);
        assert_eq!(vals[0]["x"], 1);
        assert_eq!(vals[1]["x"], 2);
    }

    #[test]
    fn lenient_reader_tolerates_trailing_partial_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("events.ndjson");

        {
            let mut w = NdjsonWriter::open_append(&path).unwrap();
            #[derive(Serialize)]
            struct E {
                event: &'static str,
                x: u32,
            }
            w.write_event(&E { event: "e", x: 1 }).unwrap();
            w.flush().unwrap();
        }

        // Simulate crash: append a partial JSON line (no newline, invalid JSON).
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(br#"{"event":"e","x":"#).unwrap();
        f.flush().unwrap();

        let vals = read_ndjson_lenient(&path);
        assert_eq!(vals.len(), 1);
        assert_eq!(vals[0]["x"], 1);
    }
}
