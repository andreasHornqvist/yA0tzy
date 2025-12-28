//! Iteration controller (TUI-driven).
//!
//! v1 scope:
//! - define a phase state machine
//! - write controller phase/status fields into `runs/<id>/run.json`

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

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
    #[error("infer backend error: {0}")]
    InferBackend(#[from] yz_mcts::InferBackendError),
    #[error("gate error: {0}")]
    Gate(#[from] yz_eval::GateError),
    #[error("cancelled")]
    Cancelled,
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

fn repo_root_from_run_dir(run_dir: &Path) -> PathBuf {
    // Expected v1 layout: <repo>/runs/<run_id>/...
    // If the user passes a different layout, fall back to CWD-based resolution.
    run_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."))
}

fn python_project_dir_from_run_dir(run_dir: &Path) -> PathBuf {
    repo_root_from_run_dir(run_dir).join("python")
}

fn update_manifest_atomic(
    run_json: &Path,
    f: impl FnOnce(&mut RunManifestV1),
) -> Result<(), ControllerError> {
    let mut m = yz_logging::read_manifest(run_json)?;
    f(&mut m);
    yz_logging::write_manifest_atomic(run_json, &m)?;
    Ok(())
}

#[derive(Debug)]
pub struct IterationHandle {
    cancel: Arc<AtomicBool>,
    join: JoinHandle<Result<(), ControllerError>>,
}

impl IterationHandle {
    pub fn cancel_hard(&self) {
        self.cancel.store(true, Ordering::Relaxed);
    }

    pub fn is_finished(&self) -> bool {
        self.join.is_finished()
    }

    pub fn join(self) -> Result<(), ControllerError> {
        match self.join.join() {
            Ok(r) => r,
            Err(_) => Err(ControllerError::Fs(std::io::Error::new(
                std::io::ErrorKind::Other,
                "controller thread panicked",
            ))),
        }
    }
}

fn connect_infer_backend(endpoint: &str) -> Result<yz_mcts::InferBackend, ControllerError> {
    let opts = yz_infer::ClientOptions {
        max_inflight_total: 8192,
        max_outbound_queue: 8192,
        request_id_start: 1,
    };
    if let Some(rest) = endpoint.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            return Ok(yz_mcts::InferBackend::connect_uds(rest, 0, opts)?);
        }
        #[cfg(not(unix))]
        {
            panic!("unix:// endpoints are only supported on unix");
        }
    }
    if let Some(rest) = endpoint.strip_prefix("tcp://") {
        return Ok(yz_mcts::InferBackend::connect_tcp(rest, 0, opts)?);
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
            train_last_loss_total: None,
            train_last_loss_policy: None,
            train_last_loss_value: None,
            promotion_decision: None,
            promotion_ts_ms: None,
            gate_games: None,
            gate_win_rate: None,
            gate_draw_rate: None,
            gate_seeds_hash: None,
            gate_oracle_match_rate_overall: None,
            gate_oracle_match_rate_mark: None,
            gate_oracle_match_rate_reroll: None,
            gate_oracle_keepall_ignored: None,
            controller_phase: Some(Phase::Idle.as_str().to_string()),
            controller_status: Some("initialized".to_string()),
            controller_last_ts_ms: Some(yz_logging::now_ms()),
            controller_error: None,
            controller_iteration_idx: 0,
            iterations: Vec::new(),
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
    let iter_idx = manifest.controller_iteration_idx;
    begin_iteration(run_dir, &cfg, &mut manifest, iter_idx)?;
    run_selfplay(run_dir, &cfg, infer_endpoint, &mut manifest, &ctrl, iter_idx)?;

    ctrl.set_phase(Phase::Gate, "starting gate")?;
    run_gate(run_dir, &cfg, infer_endpoint, &ctrl, iter_idx)?;
    finalize_iteration(run_dir, &cfg, &mut manifest, iter_idx)?;

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
    let iter_idx = manifest.controller_iteration_idx;
    begin_iteration(run_dir, &cfg, &mut manifest, iter_idx)?;

    ctrl.set_phase(Phase::Selfplay, "starting selfplay")?;
    run_selfplay(run_dir, &cfg, infer_endpoint, &mut manifest, &ctrl, iter_idx)?;

    ctrl.set_phase(Phase::Train, "starting train")?;
    run_train_subprocess(run_dir, python_exe, &ctrl, iter_idx)?;

    ctrl.set_phase(Phase::Gate, "starting gate")?;
    refresh_train_stats_from_run_json(run_dir, &mut manifest, iter_idx)?;
    run_gate(run_dir, &cfg, infer_endpoint, &ctrl, iter_idx)?;
    finalize_iteration(run_dir, &cfg, &mut manifest, iter_idx)?;

    ctrl.set_phase(Phase::Done, "done")?;
    Ok(())
}

pub fn spawn_iteration(
    run_dir: impl AsRef<Path>,
    cfg: yz_core::Config,
    infer_endpoint: String,
    python_exe: String,
) -> IterationHandle {
    let run_dir = run_dir.as_ref().to_path_buf();
    let cancel = Arc::new(AtomicBool::new(false));
    let cancel2 = Arc::clone(&cancel);
    let join = std::thread::spawn(move || {
        // Create a controller bound to the shared cancel token.
        let ctrl = IterationController {
            run_dir: run_dir.clone(),
            cancel: cancel2,
        };
        // Ensure manifest/config snapshot exists even if cancelled early.
        let mut manifest = ensure_manifest(&run_dir, &cfg)?;

        let res: Result<(), ControllerError> = (|| {
            if ctrl.cancelled() {
                return Err(ControllerError::Cancelled);
            }

            let total_iters = cfg.controller.total_iterations.unwrap_or(1).max(1);

            for _ in 0..total_iters {
                if ctrl.cancelled() {
                    return Err(ControllerError::Cancelled);
                }

                let iter_idx = manifest.controller_iteration_idx;
                begin_iteration(&run_dir, &cfg, &mut manifest, iter_idx)?;

                ctrl.set_phase(
                    Phase::Selfplay,
                    format!(
                        "starting selfplay (iter {}/{} )",
                        iter_idx + 1,
                        total_iters
                    ),
                )?;
                run_selfplay(&run_dir, &cfg, &infer_endpoint, &mut manifest, &ctrl, iter_idx)?;
                if ctrl.cancelled() {
                    return Err(ControllerError::Cancelled);
                }

                ctrl.set_phase(
                    Phase::Train,
                    format!("starting train (iter {}/{})", iter_idx + 1, total_iters),
                )?;
                run_train_subprocess(&run_dir, &python_exe, &ctrl, iter_idx)?;
                if ctrl.cancelled() {
                    return Err(ControllerError::Cancelled);
                }
                // After training completes, pull the latest train stats from run.json (trainer updates it).
                refresh_train_stats_from_run_json(&run_dir, &mut manifest, iter_idx)?;

                ctrl.set_phase(
                    Phase::Gate,
                    format!("starting gate (iter {}/{})", iter_idx + 1, total_iters),
                )?;
                run_gate(&run_dir, &cfg, &infer_endpoint, &ctrl, iter_idx)?;

                finalize_iteration(&run_dir, &cfg, &mut manifest, iter_idx)?;
                manifest.controller_iteration_idx = manifest.controller_iteration_idx.saturating_add(1);
                yz_logging::write_manifest_atomic(run_dir.join("run.json"), &manifest)?;
            }

            ctrl.set_phase(Phase::Done, "done")?;
            Ok(())
        })();

        if let Err(e) = &res {
            let _ = ctrl.set_error(e.to_string());
        }
        res
    });
    IterationHandle { cancel, join }
}

fn begin_iteration(
    run_dir: &Path,
    cfg: &yz_core::Config,
    manifest: &mut RunManifestV1,
    iter_idx: u32,
) -> Result<(), ControllerError> {
    // Ensure the iteration entry exists.
    if manifest.iterations.iter().all(|it| it.idx != iter_idx) {
        let mut it = yz_logging::IterationSummaryV1::default();
        it.idx = iter_idx;
        it.started_ts_ms = yz_logging::now_ms();
        it.selfplay.games_target = cfg.selfplay.games_per_iteration.max(1) as u64;
        it.train.steps_target = cfg
            .training
            .steps_per_iteration
            .map(|x| x.max(1) as u64);
        // Gate schedule size can be clamped by seed set length; store configured target for now.
        it.gate.games_target = cfg.gating.games.max(1) as u64;
        manifest.iterations.push(it);
    }

    // Reset phase-local progress counters for the iteration.
    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.ended_ts_ms = None;
        it.promoted = None;
        it.promoted_model = None;
        it.promotion_reason = None;
        it.selfplay.games_completed = 0;
        it.gate.games_completed = 0;
        it.gate.win_rate = None;
        it.gate.draw_rate = None;
        it.oracle.match_rate_overall = None;
        it.oracle.match_rate_mark = None;
        it.oracle.match_rate_reroll = None;
        it.oracle.keepall_ignored = None;
        it.train.started_ts_ms = None;
        it.train.ended_ts_ms = None;
        it.train.steps_completed = None;
        it.train.last_loss_total = None;
        it.train.last_loss_policy = None;
        it.train.last_loss_value = None;
    }

    yz_logging::write_manifest_atomic(run_dir.join("run.json"), manifest)?;
    Ok(())
}

fn finalize_iteration(
    run_dir: &Path,
    cfg: &yz_core::Config,
    manifest: &mut RunManifestV1,
    iter_idx: u32,
) -> Result<(), ControllerError> {
    let wr = manifest.gate_win_rate;
    let threshold = cfg.gating.win_rate_threshold;
    let promoted = wr.map(|x| x >= threshold);
    let reason = wr.map(|x| format!("win_rate={x:.4} threshold={threshold:.4}"));

    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.ended_ts_ms = Some(yz_logging::now_ms());
        it.gate.win_rate = wr;
        it.gate.draw_rate = manifest.gate_draw_rate;
        it.oracle.match_rate_overall = manifest.gate_oracle_match_rate_overall;
        it.oracle.match_rate_mark = manifest.gate_oracle_match_rate_mark;
        it.oracle.match_rate_reroll = manifest.gate_oracle_match_rate_reroll;
        it.oracle.keepall_ignored = manifest.gate_oracle_keepall_ignored;
        it.promoted = promoted;
        it.promoted_model = promoted.map(|p| if p { "candidate".to_string() } else { "best".to_string() });
        it.promotion_reason = reason;
        it.train.steps_completed = Some(manifest.train_step);
    }

    yz_logging::write_manifest_atomic(run_dir.join("run.json"), manifest)?;
    Ok(())
}

fn refresh_train_stats_from_run_json(
    run_dir: &Path,
    manifest: &mut RunManifestV1,
    iter_idx: u32,
) -> Result<(), ControllerError> {
    let run_json = run_dir.join("run.json");
    let fresh = yz_logging::read_manifest(&run_json)?;
    *manifest = fresh;

    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.train.steps_completed = Some(manifest.train_step);
        it.train.last_loss_total = manifest.train_last_loss_total;
        it.train.last_loss_policy = manifest.train_last_loss_policy;
        it.train.last_loss_value = manifest.train_last_loss_value;
    }

    yz_logging::write_manifest_atomic(run_dir.join("run.json"), manifest)?;
    Ok(())
}

fn wait_child_cancellable(mut child: std::process::Child, ctrl: &IterationController) -> Result<(), ControllerError> {
    loop {
        if ctrl.cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            return Err(ControllerError::Cancelled);
        }
        match child.try_wait()? {
            Some(status) => {
                if status.success() {
                    return Ok(());
                }
                return Err(ControllerError::Fs(std::io::Error::new(
                    std::io::ErrorKind::Other,
                    format!("train subprocess failed: {status}"),
                )));
            }
            None => std::thread::sleep(Duration::from_millis(50)),
        }
    }
}

fn tail_file(path: &Path, max_bytes: usize) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    let start = bytes.len().saturating_sub(max_bytes);
    let s = String::from_utf8_lossy(&bytes[start..]).to_string();
    let t = s.trim();
    if t.is_empty() { None } else { Some(t.to_string()) }
}

fn build_train_command(
    run_dir: &Path,
    python_exe: &str,
) -> std::process::Command {
    // Preferred runner: `uv run python -m yatzy_az ...` if uv is available.
    // Fallback: invoke the provided python executable directly.
    let use_uv = std::process::Command::new("uv")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    // We expect the standard repo layout: python package lives under ./python.
    let py_dir = python_project_dir_from_run_dir(run_dir);
    let out_models = run_dir.join("models");
    let replay_dir = run_dir.join("replay");

    if use_uv {
        let mut cmd = std::process::Command::new("uv");
        cmd.current_dir(py_dir);
        cmd.args(["run", "python", "-m", "yatzy_az", "train"]);
        cmd.args([
            "--replay",
            replay_dir.to_string_lossy().as_ref(),
            "--out",
            out_models.to_string_lossy().as_ref(),
            "--config",
            run_dir.join("config.yaml").to_string_lossy().as_ref(),
        ]);
        cmd
    } else {
        let mut cmd = std::process::Command::new(python_exe);
        cmd.current_dir(py_dir);
        cmd.args(["-m", "yatzy_az", "train"]);
        cmd.args([
            "--replay",
            replay_dir.to_string_lossy().as_ref(),
            "--out",
            out_models.to_string_lossy().as_ref(),
            "--config",
            run_dir.join("config.yaml").to_string_lossy().as_ref(),
        ]);
        cmd
    }
}

fn run_train_subprocess(
    run_dir: &Path,
    python_exe: &str,
    ctrl: &IterationController,
    iter_idx: u32,
) -> Result<(), ControllerError> {
    // We expect the standard repo layout: python package lives under ./python
    // and run dir is runs/<id>/ with replay/ and models/.
    let out_models = run_dir.join("models");
    let _replay_dir = run_dir.join("replay");
    std::fs::create_dir_all(&out_models)?;

    let run_json = run_dir.join("run.json");
    let logs_dir = run_dir.join("logs");
    std::fs::create_dir_all(&logs_dir)?;

    // Mark train start in iteration summary (merge-safe with trainer updates).
    let started_ts = yz_logging::now_ms();
    update_manifest_atomic(&run_json, |m| {
        if let Some(it) = m.iterations.iter_mut().find(|it| it.idx == iter_idx) {
            it.train.started_ts_ms = Some(started_ts);
            it.train.ended_ts_ms = None;
        }
        m.controller_last_ts_ms = Some(started_ts);
    })?;

    ctrl.set_phase(Phase::Train, "running train")?;

    // Capture logs for debugging; do not spam the TUI.
    let stdout_path = logs_dir.join("train_stdout.log");
    let stderr_path = logs_dir.join("train_stderr.log");
    let stdout_f = std::fs::File::create(&stdout_path)?;
    let stderr_f = std::fs::File::create(&stderr_path)?;

    // Minimal args. Trainer will update run.json and metrics.ndjson itself (E10.5S2).
    let mut cmd = build_train_command(run_dir, python_exe);
    let child = cmd
        .stdout(std::process::Stdio::from(stdout_f))
        .stderr(std::process::Stdio::from(stderr_f))
        .spawn()
        .map_err(ControllerError::Fs)?;

    let res = wait_child_cancellable(child, ctrl);
    let ended_ts = yz_logging::now_ms();
    let _ = update_manifest_atomic(&run_json, |m| {
        if let Some(it) = m.iterations.iter_mut().find(|it| it.idx == iter_idx) {
            it.train.ended_ts_ms = Some(ended_ts);
        }
        m.controller_last_ts_ms = Some(ended_ts);
    });

    match res {
        Ok(()) => Ok(()),
        Err(e) => {
            // Include a short stderr tail for quick diagnosis.
            let tail = tail_file(&stderr_path, 8 * 1024);
            let mut msg = e.to_string();
            if let Some(t) = tail {
                msg.push_str("\n--- train_stderr_tail ---\n");
                msg.push_str(&t);
            }
            Err(ControllerError::Fs(std::io::Error::new(
                std::io::ErrorKind::Other,
                msg,
            )))
        }
    }
}

fn run_selfplay(
    run_dir: &Path,
    cfg: &yz_core::Config,
    infer_endpoint: &str,
    manifest: &mut RunManifestV1,
    ctrl: &IterationController,
    iter_idx: u32,
) -> Result<(), ControllerError> {
    let replay_dir = run_dir.join("replay");
    let logs_dir = run_dir.join("logs");
    let run_json = run_dir.join("run.json");

    let backend = connect_infer_backend(infer_endpoint)?;
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
    let base_total = manifest.selfplay_games_completed;
    let mut completed_games: u32 = 0;
    let mut next_game_id: u64 = parallel as u64;
    while completed_games < games {
        if ctrl.cancelled() {
            return Err(ControllerError::Cancelled);
        }
        sched.tick_and_write(&backend, &mut writer, Some(&mut loggers))?;
        for t in sched.tasks_mut() {
            if yz_core::is_terminal(&t.state) {
                completed_games += 1;
                if completed_games % 10 == 0 || completed_games == games {
                    let total = base_total + completed_games as u64;
                    manifest.selfplay_games_completed = total;
                    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
                        it.selfplay.games_completed = completed_games as u64;
                        it.selfplay.games_target = games as u64;
                    }
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

    let total = base_total + completed_games as u64;
    manifest.selfplay_games_completed = total;
    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.selfplay.games_completed = completed_games as u64;
        it.selfplay.games_target = games as u64;
    }
    let _ = yz_logging::write_manifest_atomic(&run_json, manifest);
    Ok(())
}

fn run_gate(
    run_dir: &Path,
    cfg: &yz_core::Config,
    infer_endpoint: &str,
    ctrl: &IterationController,
    iter_idx: u32,
) -> Result<(), ControllerError> {
    if ctrl.cancelled() {
        return Err(ControllerError::Cancelled);
    }
    let run_json = run_dir.join("run.json");
    let mut m = yz_logging::read_manifest(&run_json)?;

    struct ProgressSink {
        run_json: PathBuf,
        iter_idx: u32,
    }
    impl yz_eval::GateProgress for ProgressSink {
        fn on_game_completed(&mut self, completed: u32, total: u32) {
            let Ok(mut m) = yz_logging::read_manifest(&self.run_json) else {
                return;
            };
            if let Some(it) = m.iterations.iter_mut().find(|it| it.idx == self.iter_idx) {
                it.gate.games_completed = completed as u64;
                // Prefer the actual schedule length discovered by yz-eval.
                it.gate.games_target = total as u64;
            }
            let _ = yz_logging::write_manifest_atomic(&self.run_json, &m);
        }
    }

    let mut sink = ProgressSink {
        run_json: run_json.clone(),
        iter_idx,
    };

    let report = yz_eval::gate_with_progress(
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
        Some(&mut sink),
    )
    ?;

    let wr = report.win_rate();
    m.gate_games = Some(report.games as u64);
    m.gate_win_rate = Some(wr);
    m.gate_draw_rate = Some(report.draw_rate);
    m.gate_seeds_hash = Some(report.seeds_hash.clone());
    m.gate_oracle_match_rate_overall = Some(report.oracle_match_rate_overall);
    m.gate_oracle_match_rate_mark = Some(report.oracle_match_rate_mark);
    m.gate_oracle_match_rate_reroll = Some(report.oracle_match_rate_reroll);
    m.gate_oracle_keepall_ignored = Some(report.oracle_keepall_ignored);

    // Also update current iteration entry (final values).
    if let Some(it) = m.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.gate.games_target = report.games as u64;
        it.gate.games_completed = report.games as u64;
        it.gate.win_rate = Some(wr);
        it.gate.draw_rate = Some(report.draw_rate);
        it.oracle.match_rate_overall = Some(report.oracle_match_rate_overall);
        it.oracle.match_rate_mark = Some(report.oracle_match_rate_mark);
        it.oracle.match_rate_reroll = Some(report.oracle_match_rate_reroll);
        it.oracle.keepall_ignored = Some(report.oracle_keepall_ignored);
    }
    yz_logging::write_manifest_atomic(&run_json, &m)?;

    Ok(())
}

#[cfg(test)]
mod cancel_tests {
    use super::*;

    #[test]
    fn cancel_kills_child_process() {
        // This unit test exercises the cancellation kill loop without requiring torch/yatzy_az.
        let dir = tempfile::tempdir().unwrap();
        let ctrl = IterationController::new(dir.path());
        // Create a dummy run.json so set_error works if needed.
        std::fs::write(dir.path().join("run.json"), serde_json::to_string_pretty(&RunManifestV1 {
            run_manifest_version: yz_logging::RUN_MANIFEST_VERSION,
            run_id: "test".to_string(),
            created_ts_ms: yz_logging::now_ms(),
            protocol_version: 1,
            feature_schema_id: 1,
            action_space_id: "oracle_keepmask_v1".to_string(),
            ruleset_id: "swedish_scandinavian_v1".to_string(),
            git_hash: None,
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
            train_last_loss_total: None,
            train_last_loss_policy: None,
            train_last_loss_value: None,
            promotion_decision: None,
            promotion_ts_ms: None,
            gate_games: None,
            gate_win_rate: None,
            gate_draw_rate: None,
            gate_seeds_hash: None,
            gate_oracle_match_rate_overall: None,
            gate_oracle_match_rate_mark: None,
            gate_oracle_match_rate_reroll: None,
            gate_oracle_keepall_ignored: None,
            controller_phase: None,
            controller_status: None,
            controller_last_ts_ms: None,
            controller_error: None,
            controller_iteration_idx: 0,
            iterations: Vec::new(),
        }).unwrap()).unwrap();

        let child = std::process::Command::new("sleep").arg("10").spawn().unwrap();
        ctrl.request_cancel();
        let res = wait_child_cancellable(child, &ctrl);
        assert!(matches!(res, Err(ControllerError::Cancelled)));
    }

    #[test]
    fn begin_iteration_initializes_targets_and_resets_progress() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let mut cfg = yz_core::Config::default();
        cfg.selfplay.games_per_iteration = 7;
        cfg.gating.games = 10;
        cfg.training.steps_per_iteration = Some(123);

        let mut m = ensure_manifest(run_dir, &cfg).unwrap();
        assert!(m.iterations.is_empty());

        begin_iteration(run_dir, &cfg, &mut m, 0).unwrap();
        assert_eq!(m.controller_iteration_idx, 0);
        assert_eq!(m.iterations.len(), 1);
        let it = m.iterations.iter().find(|it| it.idx == 0).unwrap();
        assert_eq!(it.selfplay.games_target, 7);
        assert_eq!(it.selfplay.games_completed, 0);
        assert_eq!(it.gate.games_target, 10);
        assert_eq!(it.gate.games_completed, 0);
        assert_eq!(it.train.steps_target, Some(123));
        assert_eq!(it.train.steps_completed, None);
    }

    #[test]
    #[cfg(unix)]
    fn train_failure_sets_train_ended_ts_ms() {
        // Ensure run.json exists with an iteration entry, then simulate a failing "train" process.
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let mut cfg = yz_core::Config::default();
        cfg.selfplay.games_per_iteration = 1;
        cfg.gating.games = 1;
        cfg.training.steps_per_iteration = Some(10);

        let mut m = ensure_manifest(run_dir, &cfg).unwrap();
        begin_iteration(run_dir, &cfg, &mut m, 0).unwrap();
        yz_logging::write_manifest_atomic(run_dir.join("run.json"), &m).unwrap();

        let ctrl = IterationController::new(run_dir);

        // Build a command that fails immediately.
        let mut cmd = std::process::Command::new("sh");
        cmd.arg("-c").arg("echo boom 1>&2; exit 7");
        // Avoid relying on repo layout in tests (tempdir has no python/).

        // Reuse internal logic: mark started, run child, mark ended, attach stderr tail.
        let res = {
            let run_json = run_dir.join("run.json");
            let logs_dir = run_dir.join("logs");
            std::fs::create_dir_all(&logs_dir).unwrap();
            let stderr_path = logs_dir.join("train_stderr.log");
            let stdout_path = logs_dir.join("train_stdout.log");
            let stdout_f = std::fs::File::create(&stdout_path).unwrap();
            let stderr_f = std::fs::File::create(&stderr_path).unwrap();

            let started_ts = yz_logging::now_ms();
            update_manifest_atomic(&run_json, |m| {
                if let Some(it) = m.iterations.iter_mut().find(|it| it.idx == 0) {
                    it.train.started_ts_ms = Some(started_ts);
                    it.train.ended_ts_ms = None;
                }
            })
            .unwrap();

            let child = cmd
                .stdout(std::process::Stdio::from(stdout_f))
                .stderr(std::process::Stdio::from(stderr_f))
                .spawn()
                .unwrap();
            let res = wait_child_cancellable(child, &ctrl);

            let ended_ts = yz_logging::now_ms();
            update_manifest_atomic(&run_json, |m| {
                if let Some(it) = m.iterations.iter_mut().find(|it| it.idx == 0) {
                    it.train.ended_ts_ms = Some(ended_ts);
                }
            })
            .unwrap();
            res
        };

        assert!(res.is_err());

        let fresh = yz_logging::read_manifest(run_dir.join("run.json")).unwrap();
        let it = fresh.iterations.iter().find(|it| it.idx == 0).unwrap();
        assert!(it.train.started_ts_ms.is_some());
        assert!(it.train.ended_ts_ms.is_some());
    }
}


