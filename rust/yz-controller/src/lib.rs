//! Iteration controller (TUI-driven).
//!
//! v1 scope:
//! - define a phase state machine
//! - write controller phase/status fields into `runs/<id>/run.json`

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use thiserror::Error;
use yz_logging::RunManifestV1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Idle,
    Selfplay,
    Train,
    Gate,
    Done,
    Error,
}

impl Phase {
    pub fn as_str(&self) -> &'static str {
        match self {
            Phase::Idle => "idle",
            Phase::Selfplay => "selfplay",
            Phase::Train => "train",
            Phase::Gate => "gate",
            Phase::Done => "done",
            Phase::Error => "error",
        }
    }
}

#[derive(Debug, Error)]
pub enum ControllerError {
    #[error("missing run.json at {0}")]
    MissingManifest(PathBuf),
    #[error("io/json error: {0}")]
    Io(#[from] yz_logging::NdjsonError),
    #[error("io error: {0}")]
    Fs(#[from] std::io::Error),
    #[error("replay error: {0}")]
    Replay(#[from] yz_replay::ReplayError),
}

/// Minimal controller handle (phase/status updates only, v1).
pub struct IterationController {
    run_dir: PathBuf,
    cancel: Arc<AtomicBool>,
}

impl IterationController {
    pub fn new(run_dir: impl AsRef<Path>) -> Self {
        Self {
            run_dir: run_dir.as_ref().to_path_buf(),
            cancel: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn request_cancel(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    pub fn set_phase(&self, phase: Phase, status: impl Into<String>) -> Result<(), ControllerError> {
        let run_json = self.run_dir.join("run.json");
        if !run_json.exists() {
            return Err(ControllerError::MissingManifest(run_json));
        }
        let mut m = yz_logging::read_manifest(&run_json)?;
        m.controller_phase = Some(phase.as_str().to_string());
        m.controller_status = Some(status.into());
        m.controller_last_ts_ms = Some(yz_logging::now_ms());
        if phase != Phase::Error {
            m.controller_error = None;
        }
        yz_logging::write_manifest_atomic(&run_json, &m)?;
        Ok(())
    }

    pub fn set_error(&self, msg: impl Into<String>) -> Result<(), ControllerError> {
        let msg = msg.into();
        self.set_phase(Phase::Error, &msg)?;
        let run_json = self.run_dir.join("run.json");
        let mut m = yz_logging::read_manifest(&run_json)?;
        m.controller_error = Some(msg);
        m.controller_last_ts_ms = Some(yz_logging::now_ms());
        yz_logging::write_manifest_atomic(&run_json, &m)?;
        Ok(())
    }
}

fn connect_infer_backend(endpoint: &str) -> yz_mcts::InferBackend {
    let opts = yz_infer::ClientOptions {
        max_inflight_total: 8192,
        max_outbound_queue: 8192,
        request_id_start: 1,
    };
    if let Some(rest) = endpoint.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            return yz_mcts::InferBackend::connect_uds(rest, 0, opts).unwrap();
        }
        #[cfg(not(unix))]
        {
            panic!("unix:// endpoints are only supported on unix");
        }
    }
    if let Some(rest) = endpoint.strip_prefix("tcp://") {
        return yz_mcts::InferBackend::connect_tcp(rest, 0, opts).unwrap();
    }
    panic!("Unsupported infer endpoint: {endpoint}");
}

fn ensure_run_layout(run_dir: &Path) -> Result<(), ControllerError> {
    std::fs::create_dir_all(run_dir.join("logs"))?;
    std::fs::create_dir_all(run_dir.join("models"))?;
    std::fs::create_dir_all(run_dir.join("replay"))?;
    Ok(())
}

fn ensure_manifest(run_dir: &Path, cfg: &yz_core::Config) -> Result<RunManifestV1, ControllerError> {
    ensure_run_layout(run_dir)?;
    let run_json = run_dir.join("run.json");

    let run_id = run_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("run")
        .to_string();

    let mut m = if run_json.exists() {
        yz_logging::read_manifest(&run_json)?
    } else {
        RunManifestV1 {
            run_manifest_version: yz_logging::RUN_MANIFEST_VERSION,
            run_id,
            created_ts_ms: yz_logging::now_ms(),
            protocol_version: yz_infer::protocol::PROTOCOL_VERSION,
            feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
            action_space_id: "oracle_keepmask_v1".to_string(),
            ruleset_id: "swedish_scandinavian_v1".to_string(),
            git_hash: yz_logging::try_git_hash(),
            config_hash: None,
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
            controller_phase: Some(Phase::Idle.as_str().to_string()),
            controller_status: Some("initialized".to_string()),
            controller_last_ts_ms: Some(yz_logging::now_ms()),
            controller_error: None,
        }
    };

    // Ensure config snapshot exists and record hashes.
    if m.config_snapshot.is_none() || !run_dir.join("config.yaml").exists() {
        let (rel, h) = yz_logging::write_config_snapshot_atomic(run_dir, cfg)?;
        m.config_snapshot = Some(rel);
        m.config_snapshot_hash = Some(h.clone());
        // For UI-generated configs, config_hash can be the same as snapshot hash.
        m.config_hash = Some(h);
    }

    yz_logging::write_manifest_atomic(&run_json, &m)?;
    Ok(m)
}

/// Run selfplay then gate in-process, using the same core logic as the CLI.
///
/// Notes:
/// - requires an inference server already running at `infer_endpoint`
/// - writes replay shards + metrics logs under `run_dir`
pub fn run_selfplay_then_gate(
    run_dir: impl AsRef<Path>,
    cfg: yz_core::Config,
    infer_endpoint: &str,
) -> Result<(), ControllerError> {
    let run_dir = run_dir.as_ref();
    let ctrl = IterationController::new(run_dir);
    let mut manifest = ensure_manifest(run_dir, &cfg)?;

    ctrl.set_phase(Phase::Selfplay, "starting selfplay")?;
    run_selfplay(run_dir, &cfg, infer_endpoint, &mut manifest)?;

    ctrl.set_phase(Phase::Gate, "starting gate")?;
    run_gate(run_dir, &cfg, infer_endpoint)?;

    ctrl.set_phase(Phase::Done, "done")?;
    Ok(())
}

/// Run a full iteration: selfplay → train → gate.
///
/// Training is currently implemented as a **subprocess** invocation of the Python trainer
/// (fallback mode). This keeps Rust orchestration in-process while avoiding a heavy PyO3 embed.
pub fn run_iteration(
    run_dir: impl AsRef<Path>,
    cfg: yz_core::Config,
    infer_endpoint: &str,
    python_exe: &str,
) -> Result<(), ControllerError> {
    let run_dir = run_dir.as_ref();
    let ctrl = IterationController::new(run_dir);
    let mut manifest = ensure_manifest(run_dir, &cfg)?;

    ctrl.set_phase(Phase::Selfplay, "starting selfplay")?;
    run_selfplay(run_dir, &cfg, infer_endpoint, &mut manifest)?;

    ctrl.set_phase(Phase::Train, "starting train")?;
    run_train_subprocess(run_dir, python_exe)?;

    ctrl.set_phase(Phase::Gate, "starting gate")?;
    run_gate(run_dir, &cfg, infer_endpoint)?;

    ctrl.set_phase(Phase::Done, "done")?;
    Ok(())
}

fn run_train_subprocess(run_dir: &Path, python_exe: &str) -> Result<(), ControllerError> {
    // We expect the standard repo layout: python package lives under ./python
    // and run dir is runs/<id>/ with replay/ and models/.
    let out_models = run_dir.join("models");
    let replay_dir = run_dir.join("replay");
    std::fs::create_dir_all(&out_models)?;

    // Minimal args. Trainer will update run.json and metrics.ndjson itself (E10.5S2).
    let status = std::process::Command::new(python_exe)
        .current_dir("python")
        .args([
            "-m",
            "yatzy_az",
            "train",
            "--replay",
            replay_dir.to_string_lossy().as_ref(),
            "--out",
            out_models.to_string_lossy().as_ref(),
            "--config",
            run_dir.join("config.yaml").to_string_lossy().as_ref(),
        ])
        .status()
        .map_err(ControllerError::Fs)?;
    if !status.success() {
        return Err(ControllerError::Fs(std::io::Error::new(
            std::io::ErrorKind::Other,
            format!("python train failed: {status}"),
        )));
    }
    Ok(())
}

fn run_selfplay(
    run_dir: &Path,
    cfg: &yz_core::Config,
    infer_endpoint: &str,
    manifest: &mut RunManifestV1,
) -> Result<(), ControllerError> {
    let replay_dir = run_dir.join("replay");
    let logs_dir = run_dir.join("logs");
    let run_json = run_dir.join("run.json");

    let backend = connect_infer_backend(infer_endpoint);
    let mut writer = yz_replay::ShardWriter::new(yz_replay::ShardWriterConfig {
        out_dir: replay_dir,
        max_samples_per_shard: 8192,
        git_hash: manifest.git_hash.clone(),
        config_hash: manifest.config_hash.clone(),
    })?;

    let parallel = cfg.selfplay.threads_per_worker.max(1) as usize;
    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations: cfg.mcts.budget_mark.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: cfg.mcts.dirichlet_epsilon,
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss: 1.0,
    };

    let mut tasks = Vec::new();
    for gid in 0..parallel as u64 {
        let mut ctx = yz_core::TurnContext::new_rng(0xC0FFEE ^ gid);
        let s = yz_core::initial_state(&mut ctx);
        tasks.push(yz_runtime::GameTask::new(
            gid,
            s,
            yz_mcts::ChanceMode::Rng {
                seed: 0xBADC0DE ^ gid,
            },
            mcts_cfg,
        ));
    }
    let mut sched = yz_runtime::Scheduler::new(tasks, 64);

    let v = yz_logging::VersionInfoV1 {
        protocol_version: yz_infer::protocol::PROTOCOL_VERSION,
        feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
        action_space_id: "oracle_keepmask_v1",
        ruleset_id: "swedish_scandinavian_v1",
    };
    let mut loggers = yz_runtime::RunLoggers {
        run_id: manifest.run_id.clone(),
        v,
        git_hash: manifest.git_hash.clone(),
        config_snapshot: manifest.config_snapshot.clone(),
        root_log_every_n: 50,
        iter: yz_logging::NdjsonWriter::open_append_with_flush(
            logs_dir.join("iteration_stats.ndjson"),
            100,
        )?,
        roots: yz_logging::NdjsonWriter::open_append_with_flush(
            logs_dir.join("mcts_roots.ndjson"),
            100,
        )?,
        metrics: yz_logging::NdjsonWriter::open_append_with_flush(
            logs_dir.join("metrics.ndjson"),
            100,
        )?,
    };

    let games = cfg.selfplay.games_per_iteration.max(1);
    let mut completed_games: u32 = 0;
    let mut next_game_id: u64 = parallel as u64;
    while completed_games < games {
        sched.tick_and_write(&backend, &mut writer, Some(&mut loggers))?;
        for t in sched.tasks_mut() {
            if yz_core::is_terminal(&t.state) {
                completed_games += 1;
                if completed_games % 10 == 0 || completed_games == games {
                    manifest.selfplay_games_completed = completed_games as u64;
                    let _ = yz_logging::write_manifest_atomic(&run_json, manifest);
                }
                if completed_games >= games {
                    break;
                }
                // Reset task for next game.
                let mut ctx = yz_core::TurnContext::new_rng(0xC0FFEE ^ next_game_id);
                let s = yz_core::initial_state(&mut ctx);
                *t = yz_runtime::GameTask::new(
                    next_game_id,
                    s,
                    yz_mcts::ChanceMode::Rng {
                        seed: 0xBADC0DE ^ next_game_id,
                    },
                    mcts_cfg,
                );
                next_game_id += 1;
            }
        }
    }

    writer.finish()?;
    let _ = loggers.iter.flush();
    let _ = loggers.roots.flush();
    let _ = loggers.metrics.flush();

    manifest.selfplay_games_completed = completed_games as u64;
    let _ = yz_logging::write_manifest_atomic(&run_json, manifest);
    Ok(())
}

fn run_gate(run_dir: &Path, cfg: &yz_core::Config, infer_endpoint: &str) -> Result<(), ControllerError> {
    let run_json = run_dir.join("run.json");
    let mut m = yz_logging::read_manifest(&run_json)?;

    let report = yz_eval::gate(
        cfg,
        yz_eval::GateOptions {
            infer_endpoint: infer_endpoint.to_string(),
            best_model_id: 0,
            cand_model_id: 1,
            client_opts: yz_infer::ClientOptions {
                max_inflight_total: 8192,
                max_outbound_queue: 8192,
                request_id_start: 1,
            },
            mcts_cfg: yz_mcts::MctsConfig {
                c_puct: cfg.mcts.c_puct,
                simulations: cfg.mcts.budget_mark.max(1),
                dirichlet_alpha: cfg.mcts.dirichlet_alpha,
                dirichlet_epsilon: 0.0,
                max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
                virtual_loss: 1.0,
            },
        },
    )
    .unwrap();

    let wr = report.win_rate();
    m.gate_games = Some(report.games as u64);
    m.gate_win_rate = Some(wr);
    m.gate_seeds_hash = Some(report.seeds_hash.clone());
    m.gate_oracle_match_rate_overall = Some(report.oracle_match_rate_overall);
    m.gate_oracle_match_rate_mark = Some(report.oracle_match_rate_mark);
    m.gate_oracle_match_rate_reroll = Some(report.oracle_match_rate_reroll);
    m.gate_oracle_keepall_ignored = Some(report.oracle_keepall_ignored);
    yz_logging::write_manifest_atomic(&run_json, &m)?;

    Ok(())
}


