//! Iteration controller (TUI-driven).
//!
//! v1 scope:
//! - define a phase state machine
//! - write controller phase/status fields into `runs/<id>/run.json`

use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
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
    #[error("model reload failed: {0}")]
    ReloadModel(String),
    #[error("infer-server failed: {0}")]
    InferServer(String),
}

#[derive(Debug, serde::Deserialize)]
struct ServerCapabilities {
    #[allow(dead_code)]
    version: String,
    hot_reload: bool,
    #[serde(default)]
    #[allow(dead_code)]
    pid: Option<u32>,
    #[serde(default)]
    bind: Option<String>,
    #[serde(default)]
    metrics_bind: Option<String>,
    #[serde(default)]
    device: Option<String>,
    #[serde(default)]
    max_batch: Option<u32>,
    #[serde(default)]
    max_wait_us: Option<u64>,
}

struct InferServerChild {
    child: Child,
}

impl Drop for InferServerChild {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
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

    pub fn set_phase(
        &self,
        phase: Phase,
        status: impl Into<String>,
    ) -> Result<(), ControllerError> {
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

fn get_server_capabilities(metrics_bind: &str) -> Result<ServerCapabilities, ControllerError> {
    let url = format!("http://{}/capabilities", metrics_bind);
    let resp = ureq::get(&url)
        .call()
        .map_err(|e| ControllerError::InferServer(format!("HTTP request failed: {e}")))?;
    if resp.status() != 200 {
        return Err(ControllerError::InferServer(format!(
            "capabilities returned status {}",
            resp.status()
        )));
    }
    resp.into_json()
        .map_err(|e| ControllerError::InferServer(format!("invalid capabilities JSON: {e}")))
}

fn shutdown_server(metrics_bind: &str) -> Result<(), ControllerError> {
    let url = format!("http://{}/shutdown", metrics_bind);
    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_string("{}")
        .map_err(|e| ControllerError::InferServer(format!("shutdown request failed: {e}")))?;
    if resp.status() != 200 {
        return Err(ControllerError::InferServer(format!(
            "shutdown returned status {}",
            resp.status()
        )));
    }
    Ok(())
}

fn build_infer_server_command(
    run_dir: &Path,
    cfg: &yz_core::Config,
    python_exe: &str,
) -> Command {
    // Preferred runner: `uv run python -m yatzy_az ...` if uv is available.
    let use_uv = Command::new("uv")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    let py_dir = python_project_dir_from_run_dir(run_dir);

    let bind = cfg.inference.bind.clone();
    let metrics_bind = cfg.inference.metrics_bind.clone();
    let device = cfg.inference.device.clone();
    let max_batch = cfg.inference.max_batch.to_string();
    let max_wait_us = cfg.inference.max_wait_us.to_string();

    if use_uv {
        let mut cmd = Command::new("uv");
        cmd.current_dir(py_dir);
        cmd.args(["run", "python", "-m", "yatzy_az", "infer-server"]);
        cmd.args(["--best", "dummy", "--cand", "dummy"]);
        cmd.args(["--bind", &bind]);
        cmd.args(["--metrics-bind", &metrics_bind]);
        cmd.args(["--device", &device]);
        cmd.args(["--max-batch", &max_batch]);
        cmd.args(["--max-wait-us", &max_wait_us]);
        cmd.args(["--print-stats-every-s", "0"]);
        cmd
    } else {
        let mut cmd = Command::new(python_exe);
        cmd.current_dir(py_dir);
        cmd.args(["-m", "yatzy_az", "infer-server"]);
        cmd.args(["--best", "dummy", "--cand", "dummy"]);
        cmd.args(["--bind", &bind]);
        cmd.args(["--metrics-bind", &metrics_bind]);
        cmd.args(["--device", &device]);
        cmd.args(["--max-batch", &max_batch]);
        cmd.args(["--max-wait-us", &max_wait_us]);
        cmd.args(["--print-stats-every-s", "0"]);
        cmd
    }
}

fn ensure_infer_server(
    run_dir: &Path,
    cfg: &yz_core::Config,
    python_exe: &str,
) -> Result<Option<InferServerChild>, ControllerError> {
    // If reachable, decide reuse vs restart. We restart if:
    // - server is an older capabilities version (pre-v2, no config fields)
    // - or server config doesn't match the desired config (TUI settings must win)
    if let Ok(caps) = get_server_capabilities(&cfg.inference.metrics_bind) {
        if caps.hot_reload {
            let caps_is_v2 = caps.version.trim() == "2";
            let cfg_matches = caps_is_v2
                && caps.device.as_deref() == Some(cfg.inference.device.as_str())
                && caps.bind.as_deref() == Some(cfg.inference.bind.as_str())
                && caps.metrics_bind.as_deref() == Some(cfg.inference.metrics_bind.as_str())
                && caps.max_batch == Some(cfg.inference.max_batch)
                && caps.max_wait_us == Some(cfg.inference.max_wait_us);
            if cfg_matches {
                return Ok(None);
            }

            // Try to shut down the existing server so we can restart with correct settings.
            let _ = shutdown_server(&cfg.inference.metrics_bind);
            // Give it a moment to release the port/socket.
            std::thread::sleep(Duration::from_millis(150));
        }
    }

    // Clean up UDS socket path if configured (server will recreate it).
    if let Some(rest) = cfg.inference.bind.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            let sock = Path::new(rest);
            let _ = std::fs::remove_file(sock);
        }
    }

    // Capture server logs to run-local file for debugging.
    let log_path = run_dir.join("logs").join("infer_server.log");
    let stdout = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let stderr = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;

    let mut cmd = build_infer_server_command(run_dir, cfg, python_exe);
    cmd.stdout(Stdio::from(stdout));
    cmd.stderr(Stdio::from(stderr));
    let mut child = cmd.spawn().map_err(|e| {
        ControllerError::InferServer(format!("failed to spawn infer-server: {e}"))
    })?;

    // Wait for readiness (capabilities endpoint).
    for _ in 0..200 {
        std::thread::sleep(Duration::from_millis(25));
        if let Ok(caps) = get_server_capabilities(&cfg.inference.metrics_bind) {
            if caps.hot_reload {
                return Ok(Some(InferServerChild { child }));
            }
        }
        if let Ok(Some(status)) = child.try_wait() {
            let tail = tail_file(&log_path, 16 * 1024).unwrap_or_default();
            return Err(ControllerError::InferServer(format!(
                "infer-server exited early: {status}\n{tail}"
            )));
        }
    }

    let tail = tail_file(&log_path, 16 * 1024).unwrap_or_default();
    Err(ControllerError::InferServer(format!(
        "infer-server did not become ready (metrics_bind={}); log tail:\n{}",
        cfg.inference.metrics_bind, tail
    )))
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
            Err(_) => Err(ControllerError::Fs(std::io::Error::other(
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

fn ensure_manifest(
    run_dir: &Path,
    cfg: &yz_core::Config,
) -> Result<RunManifestV1, ControllerError> {
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
            model_reloads: 0,
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
    // E13.2S4: Hot-reload best model before selfplay.
    manifest.model_reloads += reload_best_for_selfplay(run_dir, &cfg)?;
    run_selfplay(
        run_dir,
        &cfg,
        infer_endpoint,
        &mut manifest,
        &ctrl,
        iter_idx,
    )?;

    ctrl.set_phase(Phase::Gate, "starting gate")?;
    // E13.2S4: Hot-reload best + candidate models before gating.
    manifest.model_reloads += reload_models_for_gating(run_dir, &cfg)?;
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

    // E13.2S2: Ensure best.pt exists (bootstrap via model-init if missing).
    ensure_best_pt(run_dir, &cfg, python_exe)?;

    let iter_idx = manifest.controller_iteration_idx;
    begin_iteration(run_dir, &cfg, &mut manifest, iter_idx)?;

    ctrl.set_phase(Phase::Selfplay, "starting selfplay")?;
    // E13.2S4: Hot-reload best model before selfplay.
    manifest.model_reloads += reload_best_for_selfplay(run_dir, &cfg)?;
    run_selfplay(
        run_dir,
        &cfg,
        infer_endpoint,
        &mut manifest,
        &ctrl,
        iter_idx,
    )?;

    ctrl.set_phase(Phase::Train, "starting train")?;
    run_train_subprocess(run_dir, python_exe, &ctrl, iter_idx)?;

    ctrl.set_phase(Phase::Gate, "starting gate")?;
    refresh_train_stats_from_run_json(run_dir, &mut manifest, iter_idx)?;
    // E13.2S4: Hot-reload best + candidate models before gating.
    manifest.model_reloads += reload_models_for_gating(run_dir, &cfg)?;
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

        let total_iters = cfg.controller.total_iterations.unwrap_or(1).max(1);

        // Absolute semantics: controller_iteration_idx is the number of completed iterations so far.
        // If already complete, do a no-op and do NOT start subprocesses (infer-server / model-init).
        if manifest.controller_iteration_idx >= total_iters {
            ctrl.set_phase(
                Phase::Done,
                format!(
                    "done (completed {}/{})",
                    manifest.controller_iteration_idx, total_iters
                ),
            )?;
            return Ok(());
        }

        // Ensure inference server is running (owned by controller for this iteration).
        // If one is already running, we reuse it.
        let _infer_srv = ensure_infer_server(&run_dir, &cfg, &python_exe)?;

        // E13.2S2: Ensure best.pt exists (bootstrap via model-init if missing).
        ensure_best_pt(&run_dir, &cfg, &python_exe)?;

        let res: Result<(), ControllerError> = (|| {
            if ctrl.cancelled() {
                return Err(ControllerError::Cancelled);
            }

            while manifest.controller_iteration_idx < total_iters {
                if ctrl.cancelled() {
                    return Err(ControllerError::Cancelled);
                }

                let iter_idx = manifest.controller_iteration_idx;
                begin_iteration(&run_dir, &cfg, &mut manifest, iter_idx)?;

                // Update in-memory AND on-disk phase to stay in sync.
                manifest.controller_phase = Some(Phase::Selfplay.as_str().to_string());
                manifest.controller_status = Some(format!(
                    "starting selfplay (iter {}/{})",
                    iter_idx + 1,
                    total_iters
                ));
                manifest.controller_last_ts_ms = Some(yz_logging::now_ms());
                yz_logging::write_manifest_atomic(run_dir.join("run.json"), &manifest)?;

                // E13.2S4: Hot-reload best model before selfplay.
                manifest.model_reloads += reload_best_for_selfplay(&run_dir, &cfg)?;
                run_selfplay(
                    &run_dir,
                    &cfg,
                    &infer_endpoint,
                    &mut manifest,
                    &ctrl,
                    iter_idx,
                )?;
                if ctrl.cancelled() {
                    return Err(ControllerError::Cancelled);
                }

                // Update in-memory AND on-disk phase to stay in sync.
                manifest.controller_phase = Some(Phase::Train.as_str().to_string());
                manifest.controller_status = Some(format!(
                    "starting train (iter {}/{})",
                    iter_idx + 1,
                    total_iters
                ));
                manifest.controller_last_ts_ms = Some(yz_logging::now_ms());
                yz_logging::write_manifest_atomic(run_dir.join("run.json"), &manifest)?;

                run_train_subprocess(&run_dir, &python_exe, &ctrl, iter_idx)?;
                if ctrl.cancelled() {
                    return Err(ControllerError::Cancelled);
                }
                // After training completes, pull the latest train stats from run.json (trainer updates it).
                refresh_train_stats_from_run_json(&run_dir, &mut manifest, iter_idx)?;

                // Update in-memory AND on-disk phase to stay in sync.
                manifest.controller_phase = Some(Phase::Gate.as_str().to_string());
                manifest.controller_status = Some(format!(
                    "starting gate (iter {}/{})",
                    iter_idx + 1,
                    total_iters
                ));
                manifest.controller_last_ts_ms = Some(yz_logging::now_ms());
                yz_logging::write_manifest_atomic(run_dir.join("run.json"), &manifest)?;

                // E13.2S4: Hot-reload best + candidate models before gating.
                manifest.model_reloads += reload_models_for_gating(&run_dir, &cfg)?;
                run_gate(&run_dir, &cfg, &infer_endpoint, &ctrl, iter_idx)?;

                finalize_iteration(&run_dir, &cfg, &mut manifest, iter_idx)?;
                manifest.controller_iteration_idx =
                    manifest.controller_iteration_idx.saturating_add(1);
                yz_logging::write_manifest_atomic(run_dir.join("run.json"), &manifest)?;
            }

            // Update in-memory AND on-disk phase to stay in sync.
            manifest.controller_phase = Some(Phase::Done.as_str().to_string());
            manifest.controller_status = Some(format!(
                "done (completed {}/{})",
                manifest.controller_iteration_idx, total_iters
            ));
            manifest.controller_last_ts_ms = Some(yz_logging::now_ms());
            yz_logging::write_manifest_atomic(run_dir.join("run.json"), &manifest)?;
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
        let it = yz_logging::IterationSummaryV1 {
            idx: iter_idx,
            started_ts_ms: yz_logging::now_ms(),
            selfplay: yz_logging::IterSelfplaySummaryV1 {
                games_target: cfg.selfplay.games_per_iteration.max(1) as u64,
                ..Default::default()
            },
            train: yz_logging::IterTrainSummaryV1 {
                steps_target: cfg.training.steps_per_iteration.map(|x| x.max(1) as u64),
                ..Default::default()
            },
            gate: yz_logging::IterGateSummaryV1 {
                // Gate schedule size can be clamped by seed set length; store configured target for now.
                games_target: cfg.gating.games.max(1) as u64,
                ..Default::default()
            },
            ..Default::default()
        };
        manifest.iterations.push(it);
    }

    // Reset phase-local progress counters for the iteration.
    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.ended_ts_ms = None;
        it.promoted = None;
        it.promoted_model = None;
        it.promotion_reason = None;
        it.selfplay.started_ts_ms = None;
        it.selfplay.ended_ts_ms = None;
        it.selfplay.games_completed = 0;
        it.gate.started_ts_ms = None;
        it.gate.ended_ts_ms = None;
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

/// Copy a file atomically via temp + rename.
fn copy_atomic(src: &Path, dst: &Path) -> std::io::Result<()> {
    let tmp = dst.with_extension("pt.tmp");
    std::fs::copy(src, &tmp)?;
    std::fs::rename(&tmp, dst)?;
    Ok(())
}

/// Reload request payload (E13.2S4).
#[derive(serde::Serialize)]
struct ReloadRequest<'a> {
    model_id: &'a str,
    path: &'a str,
}

/// Reload response payload (E13.2S4).
#[derive(serde::Deserialize)]
struct ReloadResponse {
    ok: bool,
    #[serde(default)]
    error: Option<String>,
}

/// Call the inference server's `/reload` endpoint to hot-swap a model (E13.2S4).
///
/// * `metrics_bind` - HTTP bind address (e.g. "127.0.0.1:9100")
/// * `model_id` - "best" or "cand"
/// * `checkpoint_path` - absolute path to the checkpoint file
fn reload_model(
    metrics_bind: &str,
    model_id: &str,
    checkpoint_path: &Path,
) -> Result<(), ControllerError> {
    let url = format!("http://{}/reload", metrics_bind);
    let body = ReloadRequest {
        model_id,
        path: checkpoint_path.to_str().ok_or_else(|| {
            ControllerError::ReloadModel("path contains invalid UTF-8".to_string())
        })?,
    };

    let resp = ureq::post(&url)
        .set("Content-Type", "application/json")
        .send_json(&body)
        .map_err(|e| ControllerError::ReloadModel(format!("HTTP request failed: {e}")))?;

    let resp_body: ReloadResponse = resp
        .into_json()
        .map_err(|e| ControllerError::ReloadModel(format!("invalid response JSON: {e}")))?;

    if resp_body.ok {
        Ok(())
    } else {
        Err(ControllerError::ReloadModel(
            resp_body
                .error
                .unwrap_or_else(|| "unknown error".to_string()),
        ))
    }
}

/// Reload the "best" model before selfplay (E13.2S4).
/// Returns the number of reloads performed.
fn reload_best_for_selfplay(run_dir: &Path, cfg: &yz_core::Config) -> Result<u64, ControllerError> {
    // Must use absolute path since server runs from different directory.
    let run_dir_abs = run_dir
        .canonicalize()
        .unwrap_or_else(|_| run_dir.to_path_buf());
    let best_path = run_dir_abs.join("models").join("best.pt");
    if best_path.exists() {
        reload_model(&cfg.inference.metrics_bind, "best", &best_path)?;
        Ok(1)
    } else {
        Ok(0)
    }
}

/// Reload both "best" and "cand" models before gating (E13.2S4).
/// Returns the number of reloads performed.
fn reload_models_for_gating(run_dir: &Path, cfg: &yz_core::Config) -> Result<u64, ControllerError> {
    // Must use absolute path since server runs from different directory.
    let run_dir_abs = run_dir
        .canonicalize()
        .unwrap_or_else(|_| run_dir.to_path_buf());
    let best_path = run_dir_abs.join("models").join("best.pt");
    let cand_path = run_dir_abs.join("models").join("candidate.pt");

    let mut count = 0u64;
    if best_path.exists() {
        reload_model(&cfg.inference.metrics_bind, "best", &best_path)?;
        count += 1;
    }
    if cand_path.exists() {
        reload_model(&cfg.inference.metrics_bind, "cand", &cand_path)?;
        count += 1;
    }
    Ok(count)
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
    let reason = wr
        .map(|x| format!("win_rate={x:.4} threshold={threshold:.4}"))
        .unwrap_or_default();

    let candidate_path = run_dir.join("models").join("candidate.pt");
    let best_path = run_dir.join("models").join("best.pt");

    // E13.2S3: If promoted, copy candidate.pt → best.pt atomically.
    if promoted == Some(true) {
        if candidate_path.exists() {
            copy_atomic(&candidate_path, &best_path)?;
            manifest.best_checkpoint = Some("models/best.pt".to_string());
        } else {
            // Defensive: candidate doesn't exist. Log warning but don't fail.
            eprintln!(
                "warning: promoted but candidate.pt does not exist: {}",
                candidate_path.display()
            );
        }
    }

    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.ended_ts_ms = Some(yz_logging::now_ms());
        it.gate.win_rate = wr;
        it.gate.draw_rate = manifest.gate_draw_rate;
        it.oracle.match_rate_overall = manifest.gate_oracle_match_rate_overall;
        it.oracle.match_rate_mark = manifest.gate_oracle_match_rate_mark;
        it.oracle.match_rate_reroll = manifest.gate_oracle_match_rate_reroll;
        it.oracle.keepall_ignored = manifest.gate_oracle_keepall_ignored;
        it.promoted = promoted;
        it.promoted_model = promoted.map(|p| {
            if p {
                "candidate".to_string()
            } else {
                "best".to_string()
            }
        });
        it.promotion_reason = Some(reason.clone());
        it.train.steps_completed = Some(manifest.train_step);
    }

    yz_logging::write_manifest_atomic(run_dir.join("run.json"), manifest)?;

    // E13.2S3: Emit promotion metrics event.
    let metrics_path = run_dir.join("logs").join("metrics.ndjson");
    if let Ok(mut metrics) = yz_logging::NdjsonWriter::open_append_with_flush(&metrics_path, 1) {
        let ev = yz_logging::MetricsPromotionV1 {
            event: "promotion",
            ts_ms: yz_logging::now_ms(),
            v: yz_logging::VersionInfoV1 {
                protocol_version: yz_infer::protocol::PROTOCOL_VERSION,
                feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
                action_space_id: "oracle_keepmask_v1",
                ruleset_id: "swedish_scandinavian_v1",
            },
            run_id: manifest.run_id.clone(),
            git_hash: manifest.git_hash.clone(),
            config_snapshot: manifest.config_snapshot.clone(),
            iteration_idx: iter_idx,
            promoted: promoted.unwrap_or(false),
            win_rate: wr,
            threshold,
            reason,
            candidate_path: candidate_path.to_string_lossy().to_string(),
            best_path: best_path.to_string_lossy().to_string(),
        };
        let _ = metrics.write_event(&ev);
        let _ = metrics.flush();
    }

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

fn wait_child_cancellable(
    mut child: std::process::Child,
    ctrl: &IterationController,
) -> Result<(), ControllerError> {
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
                return Err(ControllerError::Fs(std::io::Error::other(format!(
                    "train subprocess failed: {status}"
                ))));
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
    if t.is_empty() {
        None
    } else {
        Some(t.to_string())
    }
}

fn build_train_command(run_dir: &Path, python_exe: &str) -> std::process::Command {
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
    // Must use absolute paths since command runs from py_dir, not cwd.
    let run_dir_abs = run_dir
        .canonicalize()
        .unwrap_or_else(|_| run_dir.to_path_buf());
    let out_models = run_dir_abs.join("models");
    let replay_dir = run_dir_abs.join("replay");
    let best_pt = run_dir_abs.join("models").join("best.pt");
    let config_yaml = run_dir_abs.join("config.yaml");

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
            config_yaml.to_string_lossy().as_ref(),
            "--best",
            best_pt.to_string_lossy().as_ref(),
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
            config_yaml.to_string_lossy().as_ref(),
            "--best",
            best_pt.to_string_lossy().as_ref(),
        ]);
        cmd
    }
}

fn build_model_init_command(
    run_dir: &Path,
    cfg: &yz_core::Config,
    python_exe: &str,
) -> std::process::Command {
    // Preferred runner: `uv run python -m yatzy_az ...` if uv is available.
    let use_uv = std::process::Command::new("uv")
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);

    let py_dir = python_project_dir_from_run_dir(run_dir);
    // Must use absolute path since command runs from py_dir, not cwd.
    let best_pt = run_dir
        .canonicalize()
        .unwrap_or_else(|_| run_dir.to_path_buf())
        .join("models")
        .join("best.pt");
    let hidden = cfg.model.hidden_dim.to_string();
    let blocks = cfg.model.num_blocks.to_string();

    if use_uv {
        let mut cmd = std::process::Command::new("uv");
        cmd.current_dir(py_dir);
        cmd.args(["run", "python", "-m", "yatzy_az", "model-init"]);
        cmd.args([
            "--out",
            best_pt.to_string_lossy().as_ref(),
            "--hidden",
            &hidden,
            "--blocks",
            &blocks,
        ]);
        cmd
    } else {
        let mut cmd = std::process::Command::new(python_exe);
        cmd.current_dir(py_dir);
        cmd.args(["-m", "yatzy_az", "model-init"]);
        cmd.args([
            "--out",
            best_pt.to_string_lossy().as_ref(),
            "--hidden",
            &hidden,
            "--blocks",
            &blocks,
        ]);
        cmd
    }
}

/// Ensure `models/best.pt` exists, invoking `model-init` if missing.
fn ensure_best_pt(
    run_dir: &Path,
    cfg: &yz_core::Config,
    python_exe: &str,
) -> Result<(), ControllerError> {
    let best_pt = run_dir.join("models").join("best.pt");
    if best_pt.exists() {
        return Ok(());
    }

    // Create models dir if needed.
    std::fs::create_dir_all(run_dir.join("models"))?;

    let mut cmd = build_model_init_command(run_dir, cfg, python_exe);
    // Capture output to avoid corrupting TUI display.
    let output = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(ControllerError::Fs(std::io::Error::other(format!(
            "model-init failed: {}\nstderr: {}",
            output.status, stderr
        ))));
    }

    // Verify the file was created.
    if !best_pt.exists() {
        return Err(ControllerError::Fs(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("model-init did not create {}", best_pt.display()),
        )));
    }

    Ok(())
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
            Err(ControllerError::Fs(std::io::Error::other(msg)))
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

    // Record per-iteration phase start (best-effort).
    {
        let ts = yz_logging::now_ms();
        if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
            if it.selfplay.started_ts_ms.is_none() {
                it.selfplay.started_ts_ms = Some(ts);
            }
            it.selfplay.ended_ts_ms = None;
        }
        let _ = yz_logging::write_manifest_atomic(&run_json, manifest);
    }

    let backend = connect_infer_backend(infer_endpoint)?;
    let mut writer = yz_replay::ShardWriter::new(yz_replay::ShardWriterConfig {
        out_dir: replay_dir,
        max_samples_per_shard: 8192,
        git_hash: manifest.git_hash.clone(),
        config_hash: manifest.config_hash.clone(),
    })?;

    // Number of concurrent game tasks. In v1 runtime we don't spawn OS processes; instead we run
    // multiple game state machines in a single scheduler. To preserve the intent of the config,
    // interpret (workers * threads_per_worker) as total parallel game slots.
    let parallel = (cfg.selfplay.workers.max(1) as usize) * (cfg.selfplay.threads_per_worker.max(1) as usize);
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
        // Avoid busy-spinning when all tasks are waiting on inference.
        // If there was no progress (no steps/terminals) this tick, sleep briefly to reduce CPU
        // contention with the Python server and to reduce log volume from tick-based stats.
        let before_steps = sched.stats().steps;
        let before_terminal = sched.stats().terminal;
        sched.tick_and_write(&backend, &mut writer, Some(&mut loggers))?;
        let after_steps = sched.stats().steps;
        let after_terminal = sched.stats().terminal;
        if after_steps == before_steps && after_terminal == before_terminal {
            std::thread::sleep(Duration::from_micros(200));
        }
        for t in sched.tasks_mut() {
            if yz_core::is_terminal(&t.state) {
                completed_games += 1;
                // Update progress every game (for small runs) or every 10 games (for large runs).
                let update_every = if games <= 20 { 1 } else { 10 };
                if completed_games.is_multiple_of(update_every) || completed_games == games {
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

    // E13.1S1: replay pruning (optional) + metrics event.
    if let Some(cap) = cfg.replay.capacity_shards {
        if cap > 0 {
            if let Ok(rep) = yz_replay::prune_shards_by_idx(&run_dir.join("replay"), cap as usize) {
                let ev = yz_logging::MetricsReplayPruneV1 {
                    event: "replay_prune",
                    ts_ms: yz_logging::now_ms(),
                    v: loggers.v.clone(),
                    run_id: loggers.run_id.clone(),
                    git_hash: loggers.git_hash.clone(),
                    config_snapshot: loggers.config_snapshot.clone(),
                    capacity_shards: cap,
                    before_shards: rep.before_shards as u32,
                    after_shards: rep.after_shards as u32,
                    deleted_shards: rep.deleted_shards as u32,
                    deleted_min_idx: rep.deleted_min_idx,
                    deleted_max_idx: rep.deleted_max_idx,
                };
                let _ = loggers.metrics.write_event(&ev);
            }
        }
    }
    let _ = loggers.iter.flush();
    let _ = loggers.roots.flush();
    let _ = loggers.metrics.flush();

    // Record per-iteration phase end (best-effort).
    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.selfplay.ended_ts_ms = Some(yz_logging::now_ms());
    }

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

    // Record per-iteration phase start (best-effort).
    if let Some(it) = m.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        if it.gate.started_ts_ms.is_none() {
            it.gate.started_ts_ms = Some(yz_logging::now_ms());
        }
        it.gate.ended_ts_ms = None;
    }
    let _ = yz_logging::write_manifest_atomic(&run_json, &m);

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
    )?;

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
        it.gate.ended_ts_ms = Some(yz_logging::now_ms());
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
        std::fs::write(
            dir.path().join("run.json"),
            serde_json::to_string_pretty(&RunManifestV1 {
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
                model_reloads: 0,
                controller_iteration_idx: 0,
                iterations: Vec::new(),
            })
            .unwrap(),
        )
        .unwrap();

        let child = std::process::Command::new("sleep")
            .arg("10")
            .spawn()
            .unwrap();
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

    #[test]
    fn loop_semantics_counts_remaining_iterations() {
        // Absolute semantics: total_iterations is a cap, so remaining = total - current.
        let total = 3u32;
        let mut current = 2u32;
        let mut ran = 0u32;
        while current < total {
            ran += 1;
            current += 1;
        }
        assert_eq!(ran, 1);
        assert_eq!(current, 3);
    }

    #[test]
    fn spawn_iteration_is_noop_when_already_complete() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let mut cfg = yz_core::Config::default();
        cfg.controller.total_iterations = Some(3);

        // Create manifest so spawn_iteration can read/write run.json.
        let mut m = ensure_manifest(run_dir, &cfg).unwrap();
        m.controller_iteration_idx = 3;
        yz_logging::write_manifest_atomic(run_dir.join("run.json"), &m).unwrap();

        // E13.2S2: ensure_best_pt is called early; provide a fake best.pt to skip model-init.
        std::fs::create_dir_all(run_dir.join("models")).unwrap();
        std::fs::write(run_dir.join("models").join("best.pt"), b"fake").unwrap();

        let h = spawn_iteration(
            run_dir,
            cfg,
            "unix:///tmp/does_not_matter.sock".to_string(),
            "python".to_string(),
        );
        let res = h.join();
        assert!(res.is_ok());

        let fresh = yz_logging::read_manifest(run_dir.join("run.json")).unwrap();
        assert_eq!(fresh.controller_iteration_idx, 3);
        assert_eq!(fresh.controller_phase.as_deref(), Some("done"));
    }

    #[test]
    fn build_train_command_includes_best_arg() {
        // E13.2S1: controller passes --best to trainer.
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let cmd = build_train_command(run_dir, "python");

        // Collect args as strings for easy searching.
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();
        // Find --best in args and verify the next element is the expected path.
        let idx = args.iter().position(|a| a == "--best");
        assert!(
            idx.is_some(),
            "command must include --best; got: {:?}",
            args
        );
        let best_path = &args[idx.unwrap() + 1];
        let expected_suffix = run_dir.join("models").join("best.pt");
        assert!(
            best_path.ends_with(expected_suffix.to_string_lossy().as_ref()),
            "--best value should point to models/best.pt: got {best_path:?}"
        );
    }

    #[test]
    fn build_model_init_command_includes_correct_args() {
        // E13.2S2: controller bootstraps best.pt via model-init.
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let mut cfg = yz_core::Config::default();
        cfg.model.hidden_dim = 128;
        cfg.model.num_blocks = 3;

        let cmd = build_model_init_command(run_dir, &cfg, "python");

        // Collect args as strings for easy searching.
        let args: Vec<String> = cmd
            .get_args()
            .map(|a| a.to_string_lossy().to_string())
            .collect();

        // Verify --out
        let out_idx = args.iter().position(|a| a == "--out");
        assert!(
            out_idx.is_some(),
            "command must include --out; got: {:?}",
            args
        );
        let out_path = &args[out_idx.unwrap() + 1];
        let expected_out = run_dir.join("models").join("best.pt");
        assert!(
            out_path.ends_with(expected_out.to_string_lossy().as_ref()),
            "--out value should point to models/best.pt: got {out_path:?}"
        );

        // Verify --hidden
        let hidden_idx = args.iter().position(|a| a == "--hidden");
        assert!(
            hidden_idx.is_some(),
            "command must include --hidden; got: {:?}",
            args
        );
        assert_eq!(args[hidden_idx.unwrap() + 1], "128");

        // Verify --blocks
        let blocks_idx = args.iter().position(|a| a == "--blocks");
        assert!(
            blocks_idx.is_some(),
            "command must include --blocks; got: {:?}",
            args
        );
        assert_eq!(args[blocks_idx.unwrap() + 1], "3");
    }

    #[test]
    fn promotion_copies_candidate_to_best() {
        // E13.2S3: when promoted, candidate.pt is atomically copied to best.pt.
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let cfg = yz_core::Config::default();

        // Set up run directory with manifest + models.
        let mut m = ensure_manifest(run_dir, &cfg).unwrap();
        std::fs::create_dir_all(run_dir.join("models")).unwrap();
        std::fs::create_dir_all(run_dir.join("logs")).unwrap();

        // Create fake candidate.pt and best.pt with distinct content.
        let candidate = run_dir.join("models").join("candidate.pt");
        let best = run_dir.join("models").join("best.pt");
        std::fs::write(&candidate, b"candidate-content").unwrap();
        std::fs::write(&best, b"original-best-content").unwrap();

        // Set up manifest to trigger promotion (win_rate >= threshold).
        m.gate_win_rate = Some(0.60);
        begin_iteration(run_dir, &cfg, &mut m, 0).unwrap();
        yz_logging::write_manifest_atomic(run_dir.join("run.json"), &m).unwrap();

        finalize_iteration(run_dir, &cfg, &mut m, 0).unwrap();

        // Verify best.pt now has candidate's content.
        let best_content = std::fs::read_to_string(&best).unwrap();
        assert_eq!(best_content, "candidate-content");

        // Verify manifest records promotion.
        let fresh = yz_logging::read_manifest(run_dir.join("run.json")).unwrap();
        let it = fresh.iterations.iter().find(|it| it.idx == 0).unwrap();
        assert_eq!(it.promoted, Some(true));
        assert_eq!(it.promoted_model.as_deref(), Some("candidate"));
    }

    #[test]
    fn no_promotion_leaves_best_unchanged() {
        // E13.2S3: when not promoted, best.pt is left unchanged.
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let cfg = yz_core::Config::default();

        // Set up run directory with manifest + models.
        let mut m = ensure_manifest(run_dir, &cfg).unwrap();
        std::fs::create_dir_all(run_dir.join("models")).unwrap();
        std::fs::create_dir_all(run_dir.join("logs")).unwrap();

        // Create fake candidate.pt and best.pt with distinct content.
        let candidate = run_dir.join("models").join("candidate.pt");
        let best = run_dir.join("models").join("best.pt");
        std::fs::write(&candidate, b"candidate-content").unwrap();
        std::fs::write(&best, b"original-best-content").unwrap();

        // Set up manifest to NOT trigger promotion (win_rate < threshold).
        m.gate_win_rate = Some(0.40);
        begin_iteration(run_dir, &cfg, &mut m, 0).unwrap();
        yz_logging::write_manifest_atomic(run_dir.join("run.json"), &m).unwrap();

        finalize_iteration(run_dir, &cfg, &mut m, 0).unwrap();

        // Verify best.pt still has original content.
        let best_content = std::fs::read_to_string(&best).unwrap();
        assert_eq!(best_content, "original-best-content");

        // Verify manifest records no promotion.
        let fresh = yz_logging::read_manifest(run_dir.join("run.json")).unwrap();
        let it = fresh.iterations.iter().find(|it| it.idx == 0).unwrap();
        assert_eq!(it.promoted, Some(false));
        assert_eq!(it.promoted_model.as_deref(), Some("best"));
    }

    #[test]
    fn reload_request_serializes_correctly() {
        // E13.2S4: verify ReloadRequest JSON format.
        let req = ReloadRequest {
            model_id: "best",
            path: "/tmp/models/best.pt",
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains(r#""model_id":"best""#));
        assert!(json.contains(r#""path":"/tmp/models/best.pt""#));
    }

    #[test]
    fn reload_response_deserializes_correctly() {
        // E13.2S4: verify ReloadResponse JSON parsing.
        let ok_json = r#"{"ok":true}"#;
        let resp: ReloadResponse = serde_json::from_str(ok_json).unwrap();
        assert!(resp.ok);
        assert!(resp.error.is_none());

        let err_json = r#"{"ok":false,"error":"file not found"}"#;
        let resp: ReloadResponse = serde_json::from_str(err_json).unwrap();
        assert!(!resp.ok);
        assert_eq!(resp.error.as_deref(), Some("file not found"));
    }
}
