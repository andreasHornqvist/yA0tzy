//! yz-logging: NDJSON events + metrics/tracing setup.
//!
//! v1 scope (PRD E7S3): append-only NDJSON logs for run post-mortems.

use std::fs::{File, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Run manifest schema version (Epic E8.5).
pub const RUN_MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifestV1 {
    pub run_manifest_version: u32,

    pub run_id: String,
    pub created_ts_ms: u64,

    // Versioning (PRD §10.3).
    pub protocol_version: u32,
    pub feature_schema_id: u32,
    pub action_space_id: String,
    pub ruleset_id: String,

    // Hashes for reproducibility.
    pub git_hash: Option<String>,
    pub config_hash: Option<String>,
    pub config_snapshot: Option<String>,
    pub config_snapshot_hash: Option<String>,

    // Layout.
    pub replay_dir: String,
    pub logs_dir: String,
    pub models_dir: String,

    // Counters.
    pub selfplay_games_completed: u64,
    pub train_step: u64,

    // Artifacts.
    pub best_checkpoint: Option<String>,
    pub candidate_checkpoint: Option<String>,

    // Gating/promotion (Epic E8.5.2).
    pub promotion_decision: Option<String>, // "promote" | "reject"
    pub promotion_ts_ms: Option<u64>,
    pub gate_games: Option<u64>,
    pub gate_win_rate: Option<f64>,
    pub gate_seeds_hash: Option<String>,
    pub gate_oracle_match_rate_overall: Option<f64>,
    pub gate_oracle_match_rate_mark: Option<f64>,
    pub gate_oracle_match_rate_reroll: Option<f64>,
    pub gate_oracle_keepall_ignored: Option<u64>,

    // Controller/TUI (optional).
    pub controller_phase: Option<String>,  // "idle" | "selfplay" | "train" | "gate" | "done" | "error"
    pub controller_status: Option<String>, // human-readable status
    pub controller_last_ts_ms: Option<u64>,
    pub controller_error: Option<String>,
}

pub fn now_ms() -> u64 {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    d.as_millis() as u64
}

pub fn hash_config_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

pub fn try_git_hash() -> Option<String> {
    use std::process::Command;

    let out = Command::new("git").args(["rev-parse", "HEAD"]).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8(out.stdout).ok()?;
    let t = s.trim();
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

pub fn read_manifest(path: impl AsRef<Path>) -> Result<RunManifestV1, NdjsonError> {
    let bytes = std::fs::read(path)?;
    Ok(serde_json::from_slice::<RunManifestV1>(&bytes)?)
}

pub fn write_manifest_atomic(path: impl AsRef<Path>, m: &RunManifestV1) -> Result<(), NdjsonError> {
    let path = path.as_ref();
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(m)?;
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Write (or ensure) a run-local normalized config snapshot `config.yaml`.
///
/// Contract:
/// - Path is always `run_dir/config.yaml`
/// - If it already exists, it is not modified
/// - Writes are crash-safe via temp+rename
pub fn write_config_snapshot_atomic(
    run_dir: impl AsRef<Path>,
    cfg: &yz_core::Config,
) -> Result<(String, String), NdjsonError> {
    let run_dir = run_dir.as_ref();
    let rel = "config.yaml".to_string();
    let path = run_dir.join(&rel);
    let tmp = run_dir.join("config.yaml.tmp");

    if path.exists() {
        let bytes = std::fs::read(&path)?;
        return Ok((rel, blake3::hash(&bytes).to_hex().to_string()));
    }

    let s = serde_yaml::to_string(cfg)?;
    let bytes = s.into_bytes();
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &path)?;
    Ok((rel, blake3::hash(&bytes).to_hex().to_string()))
}

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

// -------------------------
// E10.5S2: Unified metrics stream (runs/<id>/logs/metrics.ndjson)
// -------------------------

#[derive(Debug, Clone, Serialize)]
pub struct MetricsSelfplayIterV1 {
    pub event: &'static str, // "selfplay_iter"
    pub ts_ms: u64,
    pub v: VersionInfoV1,
    pub run_id: String,
    pub git_hash: Option<String>,
    pub config_snapshot: Option<String>,

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
pub struct MetricsMctsRootSampleV1 {
    pub event: &'static str, // "mcts_root_sample"
    pub ts_ms: u64,
    pub v: VersionInfoV1,
    pub run_id: String,
    pub git_hash: Option<String>,
    pub config_snapshot: Option<String>,

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

#[derive(Debug, Clone, Serialize)]
pub struct MetricsGateSummaryV1 {
    pub event: &'static str, // "gate_summary"
    pub ts_ms: u64,
    pub v: VersionInfoV1,
    pub run_id: String,
    pub git_hash: Option<String>,
    pub config_snapshot: Option<String>,

    pub decision: String,
    pub games: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    pub win_rate: f64,

    pub mean_score_diff: f64,
    pub score_diff_se: f64,
    pub score_diff_ci95_low: f64,
    pub score_diff_ci95_high: f64,

    pub seeds_hash: String,

    pub oracle_match_rate_overall: f64,
    pub oracle_match_rate_mark: f64,
    pub oracle_match_rate_reroll: f64,
    pub oracle_keepall_ignored: u64,
}

#[derive(Debug)]
pub enum NdjsonError {
    Io(io::Error),
    Json(serde_json::Error),
    Yaml(serde_yaml::Error),
}

impl std::fmt::Display for NdjsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NdjsonError::Io(e) => write!(f, "io error: {e}"),
            NdjsonError::Json(e) => write!(f, "json error: {e}"),
            NdjsonError::Yaml(e) => write!(f, "yaml error: {e}"),
        }
    }
}

impl std::error::Error for NdjsonError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            NdjsonError::Io(e) => Some(e),
            NdjsonError::Json(e) => Some(e),
            NdjsonError::Yaml(e) => Some(e),
        }
    }
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

impl From<serde_yaml::Error> for NdjsonError {
    fn from(e: serde_yaml::Error) -> Self {
        Self::Yaml(e)
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

    #[test]
    fn metrics_gate_summary_serializes_as_one_object() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("metrics.ndjson");
        let mut w = NdjsonWriter::open_append(&path).unwrap();

        let ev = MetricsGateSummaryV1 {
            event: "gate_summary",
            ts_ms: now_ms(),
            v: VersionInfoV1 {
                protocol_version: 1,
                feature_schema_id: 1,
                action_space_id: "oracle_keepmask_v1",
                ruleset_id: "swedish_scandinavian_v1",
            },
            run_id: "r".to_string(),
            git_hash: None,
            config_snapshot: Some("config.yaml".to_string()),
            decision: "promote".to_string(),
            games: 2,
            wins: 1,
            losses: 1,
            draws: 0,
            win_rate: 0.5,
            mean_score_diff: 0.0,
            score_diff_se: 1.0,
            score_diff_ci95_low: -2.0,
            score_diff_ci95_high: 2.0,
            seeds_hash: "abc".to_string(),
            oracle_match_rate_overall: 0.1,
            oracle_match_rate_mark: 0.2,
            oracle_match_rate_reroll: 0.3,
            oracle_keepall_ignored: 4,
        };
        w.write_event(&ev).unwrap();
        w.flush().unwrap();

        let vals = read_ndjson_lenient(&path);
        assert_eq!(vals.len(), 1);
        assert_eq!(vals[0]["event"], "gate_summary");
        assert_eq!(vals[0]["games"], 2);
        assert_eq!(vals[0]["decision"], "promote");
        assert_eq!(vals[0]["oracle_keepall_ignored"], 4);
    }

    #[test]
    fn manifest_write_is_atomic_wrt_tmp_file() {
        let dir = tempfile::tempdir().unwrap();
        let run_json = dir.path().join("run.json");

        let mut m = RunManifestV1 {
            run_manifest_version: RUN_MANIFEST_VERSION,
            run_id: "r".to_string(),
            created_ts_ms: now_ms(),
            protocol_version: 1,
            feature_schema_id: 1,
            action_space_id: "oracle_keepmask_v1".to_string(),
            ruleset_id: "swedish_scandinavian_v1".to_string(),
            git_hash: None,
            config_hash: Some("abc".to_string()),
            config_snapshot: None,
            config_snapshot_hash: None,
            replay_dir: "replay".to_string(),
            logs_dir: "logs".to_string(),
            models_dir: "models".to_string(),
            selfplay_games_completed: 0,
            train_step: 0,
            best_checkpoint: None,
            candidate_checkpoint: None,
            promotion_decision: None,
            promotion_ts_ms: None,
            gate_games: None,
            gate_win_rate: None,
            gate_seeds_hash: None,
            gate_oracle_match_rate_overall: None,
            gate_oracle_match_rate_mark: None,
            gate_oracle_match_rate_reroll: None,
            gate_oracle_keepall_ignored: None,
            controller_phase: None,
            controller_status: None,
            controller_last_ts_ms: None,
            controller_error: None,
        };
        write_manifest_atomic(&run_json, &m).unwrap();

        // Simulate crash leaving a corrupt tmp file around; run.json must remain readable.
        let tmp = run_json.with_extension("json.tmp");
        fs::write(&tmp, b"{not valid json").unwrap();

        let got = read_manifest(&run_json).unwrap();
        assert_eq!(got.run_id, "r");

        // Update manifest and ensure it overwrites cleanly.
        m.selfplay_games_completed = 7;
        write_manifest_atomic(&run_json, &m).unwrap();
        let got2 = read_manifest(&run_json).unwrap();
        assert_eq!(got2.selfplay_games_completed, 7);
    }

    #[test]
    fn config_snapshot_writer_is_idempotent() {
        let dir = tempfile::tempdir().unwrap();

        let cfg = yz_core::Config::from_yaml(
            r#"
inference:
  bind: "unix:///tmp/test.sock"
  device: "cpu"
  max_batch: 16
  max_wait_us: 500
mcts:
  c_puct: 1.0
  budget_reroll: 10
  budget_mark: 10
  max_inflight_per_game: 2
selfplay:
  games_per_iteration: 1
  workers: 1
  threads_per_worker: 1
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 1
gating:
  games: 2
  win_rate_threshold: 0.5
  paired_seed_swap: false
"#,
        )
        .unwrap();

        let (rel1, h1) = write_config_snapshot_atomic(dir.path(), &cfg).unwrap();
        let (rel2, h2) = write_config_snapshot_atomic(dir.path(), &cfg).unwrap();

        assert_eq!(rel1, "config.yaml");
        assert_eq!(rel1, rel2);
        assert_eq!(h1, h2);
        assert!(dir.path().join("config.yaml").exists());
    }
}
