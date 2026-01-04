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
        // We generally request a graceful shutdown via the metrics `/shutdown` endpoint first.
        // Avoid sending SIGKILL immediately: it makes Python's asyncio print noisy
        // "Task was destroyed but it is pending!" warnings which look like failures in the TUI.
        for _ in 0..200 {
            match self.child.try_wait() {
                Ok(Some(_status)) => return,
                Ok(None) => std::thread::sleep(Duration::from_millis(10)),
                Err(_) => break,
            }
        }
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

fn uv_available() -> bool {
    Command::new("uv")
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
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
    // Fallback: invoke the provided python executable directly.
    let use_uv = (python_exe == "python")
        && uv_available();

    let py_dir = python_project_dir_from_run_dir(run_dir);

    let bind = cfg.inference.bind.clone();
    let metrics_bind = cfg.inference.metrics_bind.clone();
    let device = cfg.inference.device.clone();
    let max_batch = cfg.inference.max_batch.to_string();
    let max_wait_us = cfg.inference.max_wait_us.to_string();
    let torch_threads = cfg.inference.torch_threads.map(|x| x.to_string());
    let torch_interop_threads = cfg.inference.torch_interop_threads.map(|x| x.to_string());
    let print_stats_every_s = if matches!(
        std::env::var("YZ_INFER_PRINT_STATS").as_deref(),
        Ok("1" | "true" | "yes")
    ) {
        "2.0"
    } else {
        "0"
    };

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
        if let Some(v) = torch_threads.as_deref() {
            cmd.args(["--torch-threads", v]);
        }
        if let Some(v) = torch_interop_threads.as_deref() {
            cmd.args(["--torch-interop-threads", v]);
        }
        cmd.args(["--print-stats-every-s", print_stats_every_s]);
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
        if let Some(v) = torch_threads.as_deref() {
            cmd.args(["--torch-threads", v]);
        }
        if let Some(v) = torch_interop_threads.as_deref() {
            cmd.args(["--torch-interop-threads", v]);
        }
        cmd.args(["--print-stats-every-s", print_stats_every_s]);
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
    // IMPORTANT: run_gate writes progress + final gate metrics into run.json using its own manifest.
    // Refresh the in-memory manifest before finalize_iteration to avoid clobbering gate results.
    manifest = yz_logging::read_manifest(run_dir.join("run.json"))?;
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
    run_train_subprocess(run_dir, &cfg, python_exe, &ctrl, iter_idx)?;

    ctrl.set_phase(Phase::Gate, "starting gate")?;
    refresh_train_stats_from_run_json(run_dir, &mut manifest, iter_idx)?;
    // E13.2S4: Hot-reload best + candidate models before gating.
    manifest.model_reloads += reload_models_for_gating(run_dir, &cfg)?;
    run_gate(run_dir, &cfg, infer_endpoint, &ctrl, iter_idx)?;
    // Refresh manifest after run_gate to avoid overwriting gate fields when finalizing.
    manifest = yz_logging::read_manifest(run_dir.join("run.json"))?;
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
        // Ensure we always attempt to stop the inference server on exit (success/cancel/error).
        // Per TUI semantics: always shut down the server at config.inference.metrics_bind.
        struct ShutdownGuard {
            metrics_bind: String,
            infer_srv: Option<InferServerChild>,
        }
        impl Drop for ShutdownGuard {
            fn drop(&mut self) {
                let _ = shutdown_server(&self.metrics_bind);
                // Drop infer server child last; if it was started by us, this guarantees teardown.
                // If the server was reused (no child), shutdown_server above still applies.
                let _ = self.infer_srv.take();
            }
        }

        // Create a controller bound to the shared cancel token.
        let ctrl = IterationController {
            run_dir: run_dir.clone(),
            cancel: cancel2,
        };
        let res: Result<(), ControllerError> = (|| {
            // Enforce uv-managed Python when using PATH python (TUI default).
            if python_exe == "python" && !uv_available() {
                return Err(ControllerError::InferServer(
                    "uv not found on PATH. Install uv (recommended) or run yz-controller with an explicit python_exe.".to_string(),
                ));
            }

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
            let infer_srv = ensure_infer_server(&run_dir, &cfg, &python_exe)?;
            let _shutdown_guard = ShutdownGuard {
                metrics_bind: cfg.inference.metrics_bind.clone(),
                infer_srv,
            };

            // E13.2S2: Ensure best.pt exists (bootstrap via model-init if missing).
            ensure_best_pt(&run_dir, &cfg, &python_exe)?;

            if ctrl.cancelled() {
                let _ = ctrl.set_phase(Phase::Selfplay, "shutting down...");
                return Err(ControllerError::Cancelled);
            }

            while manifest.controller_iteration_idx < total_iters {
                if ctrl.cancelled() {
                    let _ = ctrl.set_phase(Phase::Selfplay, "shutting down...");
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
                    let _ = ctrl.set_phase(Phase::Selfplay, "shutting down...");
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

                run_train_subprocess(&run_dir, &cfg, &python_exe, &ctrl, iter_idx)?;
                if ctrl.cancelled() {
                    let _ = ctrl.set_phase(Phase::Train, "shutting down...");
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

                // Refresh manifest after run_gate to avoid overwriting gate results in finalize_iteration.
                manifest = yz_logging::read_manifest(run_dir.join("run.json"))?;
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
        it.selfplay.first_game_started_ts_ms = None;
        it.selfplay.games_completed = 0;
        it.gate.started_ts_ms = None;
        it.gate.ended_ts_ms = None;
        it.gate.first_game_started_ts_ms = None;
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
    // region agent log
    fn dbg_enabled() -> bool {
        matches!(std::env::var("YZ_DEBUG_LOG").as_deref(), Ok("1" | "true" | "yes"))
    }
    fn dbg_emit(hypothesis_id: &str, location: &str, message: &str, data: serde_json::Value) {
        if !dbg_enabled() {
            return;
        }
        let payload = serde_json::json!({
            "timestamp": yz_logging::now_ms(),
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
    let t0 = std::time::Instant::now();
    dbg_emit(
        "H_reload",
        "rust/yz-controller/src/lib.rs:reload_model",
        "reload start",
        serde_json::json!({
            "metrics_bind": metrics_bind,
            "model_id": model_id,
            "checkpoint": checkpoint_path.display().to_string(),
        }),
    );
    // endregion agent log

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

    // region agent log
    dbg_emit(
        "H_reload",
        "rust/yz-controller/src/lib.rs:reload_model",
        "reload done",
        serde_json::json!({
            "model_id": model_id,
            "ok": resp_body.ok,
            "dt_ms": (t0.elapsed().as_secs_f64() * 1000.0),
            "error": resp_body.error,
        }),
    );
    // endregion agent log

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

fn build_train_command(
    run_dir: &Path,
    cfg: &yz_core::Config,
    python_exe: &str,
    iter_idx: u32,
) -> std::process::Command {
    // Preferred runner: `uv run python -m yatzy_az ...` if uv is available.
    // Fallback: invoke the provided python executable directly.
    let use_uv = (python_exe == "python")
        && uv_available();

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
    // Use per-iteration replay snapshots so pruning/new shards never break training.
    // (Trainer will create this snapshot if missing; it is used to freeze the dataset for that iteration.)
    let snapshot_path = run_dir_abs.join(format!("replay_snapshot_iter_{iter_idx:03}.json"));
    // IMPORTANT: pass model shape so the trainer can load best.pt even if config changes.
    // Without this, the trainer defaults (--hidden=256 --blocks=2) can mismatch a run-local best.pt.
    let hidden = cfg.model.hidden_dim.to_string();
    let blocks = cfg.model.num_blocks.to_string();
    let num_workers = cfg.training.dataloader_workers.to_string();
    let sample_mode = cfg.training.sample_mode.as_str();

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
            "--snapshot",
            snapshot_path.to_string_lossy().as_ref(),
            "--hidden",
            &hidden,
            "--blocks",
            &blocks,
            "--num-workers",
            &num_workers,
            "--sample-mode",
            sample_mode,
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
            "--snapshot",
            snapshot_path.to_string_lossy().as_ref(),
            "--hidden",
            &hidden,
            "--blocks",
            &blocks,
            "--num-workers",
            &num_workers,
            "--sample-mode",
            sample_mode,
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
    let use_uv = (python_exe == "python")
        && uv_available();

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
    cfg: &yz_core::Config,
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
    let mut cmd = build_train_command(run_dir, cfg, python_exe, iter_idx);
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
    let replay_workers_dir = run_dir.join("replay_workers");
    let logs_dir = run_dir.join("logs");
    let logs_workers_dir = run_dir.join("logs_workers");
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

    std::fs::create_dir_all(&replay_dir)?;
    std::fs::create_dir_all(&replay_workers_dir)?;
    std::fs::create_dir_all(&logs_dir)?;
    std::fs::create_dir_all(&logs_workers_dir)?;

    // Spawn OS processes for true CPU parallelism.
    let num_workers = cfg.selfplay.workers.max(1);
    let games_total = cfg.selfplay.games_per_iteration.max(1);
    let base_total = manifest.selfplay_games_completed;
    let seed_base = manifest.created_ts_ms ^ 0xC0FFEEu64;

    // Partition games across workers.
    let base = games_total / num_workers;
    let rem = games_total % num_workers;

    let exe = std::env::current_exe().map_err(ControllerError::Fs)?;
    let mut children: Vec<(u32, u32, Child)> = Vec::new(); // (wid, games_for_worker, child)
    for wid in 0..num_workers {
        let games_for = base + if wid < rem { 1 } else { 0 };
        if games_for == 0 {
            continue;
        }
        let mut cmd = Command::new(&exe);
        cmd.arg("selfplay-worker");
        cmd.arg("--run-dir").arg(run_dir);
        cmd.arg("--infer").arg(infer_endpoint);
        cmd.arg("--worker-id").arg(wid.to_string());
        cmd.arg("--num-workers").arg(num_workers.to_string());
        cmd.arg("--games").arg(games_for.to_string());
        cmd.arg("--seed-base").arg(seed_base.to_string());
        cmd.stdout(Stdio::inherit());
        cmd.stderr(Stdio::inherit());
        let child = cmd.spawn().map_err(ControllerError::Fs)?;
        children.push((wid, games_for, child));
    }

    fn read_worker_progress_sum(logs_workers_dir: &Path) -> (u32, Option<u64>) {
        #[derive(serde::Deserialize)]
        struct P {
            games_completed: u32,
            #[serde(default)]
            first_game_started_ts_ms: Option<u64>,
        }
        let mut sum = 0u32;
        let mut first_game_min: Option<u64> = None;
        if let Ok(rd) = std::fs::read_dir(logs_workers_dir) {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_dir() {
                    continue;
                }
                let f = p.join("progress.json");
                if let Ok(bytes) = std::fs::read(&f) {
                    if let Ok(pp) = serde_json::from_slice::<P>(&bytes) {
                        sum = sum.saturating_add(pp.games_completed);
                        if let Some(ts) = pp.first_game_started_ts_ms {
                            first_game_min = Some(match first_game_min {
                                Some(cur) => cur.min(ts),
                                None => ts,
                            });
                        }
                    }
                }
            }
        }
        (sum, first_game_min)
    }

    let mut completed_games: u32;
    let mut last_manifest_ts = std::time::Instant::now();
    // Poll loop so cancellation can be honored without blocking in wait().
    while !children.is_empty() {
        if ctrl.cancelled() {
            for (_, _, mut c) in children {
                let _ = c.kill();
            }
            return Err(ControllerError::Cancelled);
        }

        let mut failure: Option<String> = None;
        let mut i = 0usize;
        while i < children.len() {
            let (wid, _games_for, child) = &mut children[i];
            if let Some(status) = child.try_wait().map_err(ControllerError::Fs)? {
                if !status.success() {
                    failure = Some(format!("selfplay-worker {wid} failed with status {status}"));
                    break;
                }
                // Progress is driven by reading per-worker progress files, not by exit events.
                children.remove(i);
                continue;
            }
            i += 1;
        }
        if let Some(msg) = failure {
            // Kill all remaining workers best-effort.
            for (_, _, mut c) in children {
                let _ = c.kill();
            }
            return Err(ControllerError::Fs(std::io::Error::other(msg)));
        }

        // Live progress: periodically sum worker progress files and update run.json for the TUI.
        if last_manifest_ts.elapsed() >= Duration::from_millis(250) {
            let (sum, first_game_min) = read_worker_progress_sum(&logs_workers_dir);
            completed_games = sum.min(games_total);
            let total = base_total + completed_games as u64;
            manifest.selfplay_games_completed = total;
            if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
                it.selfplay.games_completed = completed_games as u64;
                it.selfplay.games_target = games_total as u64;
                if it.selfplay.first_game_started_ts_ms.is_none() {
                    it.selfplay.first_game_started_ts_ms = first_game_min;
                }
            }
            let _ = yz_logging::write_manifest_atomic(&run_json, manifest);
            last_manifest_ts = std::time::Instant::now();
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    // Final progress update (workers are done; ensure we report completion).
    completed_games = games_total;
    manifest.selfplay_games_completed = base_total + completed_games as u64;
    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.selfplay.games_completed = completed_games as u64;
        it.selfplay.games_target = games_total as u64;
        if it.selfplay.first_game_started_ts_ms.is_none() {
            // Best-effort: if not observed during polling, approximate as phase start.
            it.selfplay.first_game_started_ts_ms = it.selfplay.started_ts_ms;
        }
    }
    let _ = yz_logging::write_manifest_atomic(&run_json, manifest);

    // Merge replay shards from worker dirs into the canonical run replay dir.
    merge_replay_workers(&replay_workers_dir, &replay_dir)?;

    // E13.1S1: replay pruning (optional).
    if let Some(cap) = cfg.replay.capacity_shards {
        if cap > 0 {
            let _ = yz_replay::prune_shards_by_idx(&run_dir.join("replay"), cap as usize);
        }
    }

    // Record per-iteration phase end (best-effort).
    if let Some(it) = manifest.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.selfplay.ended_ts_ms = Some(yz_logging::now_ms());
    }

    // (already written above)
    Ok(())
}

fn parse_shard_idx_from_name(name: &str) -> Option<u64> {
    let rest = name.strip_prefix("shard_")?;
    let digits = rest
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        return None;
    }
    digits.parse::<u64>().ok()
}

fn next_shard_idx(dir: &Path) -> Result<u64, ControllerError> {
    if !dir.exists() {
        return Ok(0);
    }
    let mut max_idx: Option<u64> = None;
    for entry in std::fs::read_dir(dir).map_err(ControllerError::Fs)? {
        let e = entry.map_err(ControllerError::Fs)?;
        let p = e.path();
        let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !(name.ends_with(".safetensors") || name.ends_with(".meta.json")) {
            continue;
        }
        if let Some(idx) = parse_shard_idx_from_name(name) {
            max_idx = Some(max_idx.map(|m| m.max(idx)).unwrap_or(idx));
        }
    }
    Ok(max_idx.map(|m| m.saturating_add(1)).unwrap_or(0))
}

fn merge_replay_workers(replay_workers_dir: &Path, replay_dir: &Path) -> Result<(), ControllerError> {
    std::fs::create_dir_all(replay_dir).map_err(ControllerError::Fs)?;
    let mut out_idx = next_shard_idx(replay_dir)?;

    if !replay_workers_dir.exists() {
        return Ok(());
    }

    // Process worker dirs in sorted order for determinism.
    let mut worker_dirs: Vec<std::path::PathBuf> = Vec::new();
    for entry in std::fs::read_dir(replay_workers_dir).map_err(ControllerError::Fs)? {
        let e = entry.map_err(ControllerError::Fs)?;
        if e.path().is_dir() {
            worker_dirs.push(e.path());
        }
    }
    worker_dirs.sort();

    for wdir in worker_dirs {
        // Collect shard indices in this worker dir.
        let mut idxs: Vec<u64> = Vec::new();
        for entry in std::fs::read_dir(&wdir).map_err(ControllerError::Fs)? {
            let e = entry.map_err(ControllerError::Fs)?;
            let p = e.path();
            let Some(name) = p.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            if !(name.ends_with(".safetensors") || name.ends_with(".meta.json")) {
                continue;
            }
            if let Some(idx) = parse_shard_idx_from_name(name) {
                idxs.push(idx);
            }
        }
        idxs.sort();
        idxs.dedup();

        for idx in idxs {
            let st_src = wdir.join(format!("shard_{idx:06}.safetensors"));
            let meta_src = wdir.join(format!("shard_{idx:06}.meta.json"));
            if !(st_src.exists() && meta_src.exists()) {
                continue;
            }
            let st_dst = replay_dir.join(format!("shard_{out_idx:06}.safetensors"));
            let meta_dst = replay_dir.join(format!("shard_{out_idx:06}.meta.json"));
            std::fs::rename(&st_src, &st_dst).map_err(ControllerError::Fs)?;
            std::fs::rename(&meta_src, &meta_dst).map_err(ControllerError::Fs)?;
            out_idx += 1;
        }
    }
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

    // Plan schedule (paired seed swap, deterministic chance, argmax policy are handled in yz-eval).
    let plan = yz_eval::gate_plan(cfg)?;
    let total_games = plan.schedule.len().max(1) as u32;

    // Spawn OS processes (like self-play) for parallel gating.
    let gate_workers_dir = run_dir.join("gate_workers");
    let logs_gate_workers_dir = run_dir.join("logs_gate_workers");
    std::fs::create_dir_all(&gate_workers_dir)?;
    std::fs::create_dir_all(&logs_gate_workers_dir)?;

    // Partition schedule across workers.
    let num_workers = cfg.selfplay.workers.max(1) as usize;

    let mut per_worker: Vec<Vec<yz_eval::GameSpec>> = vec![Vec::new(); num_workers];
    if cfg.gating.paired_seed_swap {
        // Keep swap pairs together (2 games per seed) for nicer locality.
        let pairs: Vec<[yz_eval::GameSpec; 2]> = plan
            .schedule
            .chunks_exact(2)
            .map(|c| [c[0], c[1]])
            .collect();
        let pairs_total = pairs.len();
        let base = pairs_total / num_workers;
        let rem = pairs_total % num_workers;
        let mut idx = 0usize;
        for wid in 0..num_workers {
            let take = base + if wid < rem { 1 } else { 0 };
            for _ in 0..take {
                if idx >= pairs_total {
                    break;
                }
                per_worker[wid].push(pairs[idx][0]);
                per_worker[wid].push(pairs[idx][1]);
                idx += 1;
            }
        }
    } else {
        let games_total = plan.schedule.len();
        let base = games_total / num_workers;
        let rem = games_total % num_workers;
        let mut idx = 0usize;
        for wid in 0..num_workers {
            let take = base + if wid < rem { 1 } else { 0 };
            for _ in 0..take {
                if idx >= games_total {
                    break;
                }
                per_worker[wid].push(plan.schedule[idx]);
                idx += 1;
            }
        }
    }
    // Drop empty workers to avoid spawning useless processes.
    let per_worker: Vec<(u32, Vec<yz_eval::GameSpec>)> = per_worker
        .into_iter()
        .enumerate()
        .filter_map(|(wid, s)| {
            if s.is_empty() {
                None
            } else {
                Some((wid as u32, s))
            }
        })
        .collect();

    #[derive(serde::Serialize)]
    struct GameSpecJson {
        episode_seed: u64,
        swap: bool,
    }

    let exe = std::env::current_exe().map_err(ControllerError::Fs)?;
    let mut children: Vec<(u32, PathBuf, Child)> = Vec::new(); // (wid, result_path, child)
    for (wid, sched) in &per_worker {
        let sched_path = gate_workers_dir.join(format!("sched_{wid:03}.json"));
        let out_path = gate_workers_dir.join(format!("result_{wid:03}.json"));
        let items: Vec<GameSpecJson> = sched
            .iter()
            .map(|g| GameSpecJson {
                episode_seed: g.episode_seed,
                swap: g.swap,
            })
            .collect();
        let bytes =
            serde_json::to_vec(&items).map_err(|e| ControllerError::Fs(std::io::Error::other(e)))?;
        std::fs::write(&sched_path, bytes)?;

        let mut cmd = Command::new(&exe);
        cmd.arg("gate-worker");
        cmd.arg("--run-dir").arg(run_dir);
        cmd.arg("--infer").arg(infer_endpoint);
        cmd.arg("--worker-id").arg(wid.to_string());
        cmd.arg("--num-workers").arg((per_worker.len() as u32).to_string());
        cmd.arg("--best-id").arg("0");
        cmd.arg("--cand-id").arg("1");
        cmd.arg("--schedule-file").arg(&sched_path);
        cmd.arg("--out").arg(&out_path);
        cmd.stdout(Stdio::inherit());
        cmd.stderr(Stdio::inherit());
        let child = cmd.spawn().map_err(ControllerError::Fs)?;
        children.push((*wid, out_path, child));
    }

    fn read_gate_worker_progress_sum(logs_gate_workers_dir: &Path) -> (u32, Option<u64>) {
        #[derive(serde::Deserialize)]
        struct P {
            games_completed: u32,
            #[serde(default)]
            first_game_started_ts_ms: Option<u64>,
        }
        let mut sum = 0u32;
        let mut first_game_min: Option<u64> = None;
        if let Ok(rd) = std::fs::read_dir(logs_gate_workers_dir) {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_dir() {
                    continue;
                }
                let f = p.join("progress.json");
                if let Ok(bytes) = std::fs::read(&f) {
                    if let Ok(pp) = serde_json::from_slice::<P>(&bytes) {
                        sum = sum.saturating_add(pp.games_completed);
                        if let Some(ts) = pp.first_game_started_ts_ms {
                            first_game_min = Some(match first_game_min {
                                Some(cur) => cur.min(ts),
                                None => ts,
                            });
                        }
                    }
                }
            }
        }
        (sum, first_game_min)
    }

    let mut last_manifest_ts = std::time::Instant::now();
    while !children.is_empty() {
        if ctrl.cancelled() {
            for (_, _, mut c) in children {
                let _ = c.kill();
            }
            return Err(ControllerError::Cancelled);
        }

        let mut failure: Option<String> = None;
        let mut i = 0usize;
        while i < children.len() {
            let (wid, _out_path, child) = &mut children[i];
            if let Some(status) = child.try_wait().map_err(ControllerError::Fs)? {
                if !status.success() {
                    failure = Some(format!("gate-worker {wid} failed with status {status}"));
                    break;
                }
                children.remove(i);
                continue;
            }
            i += 1;
        }
        if let Some(msg) = failure {
            for (_, _, mut c) in children {
                let _ = c.kill();
            }
            return Err(ControllerError::Fs(std::io::Error::other(msg)));
        }

        // Live progress: sum worker progress and update run.json for the TUI.
        if last_manifest_ts.elapsed() >= Duration::from_millis(250) {
            let (sum, first_game_min) = read_gate_worker_progress_sum(&logs_gate_workers_dir);
            let completed = sum.min(total_games);
            if let Ok(mut mm) = yz_logging::read_manifest(&run_json) {
                if let Some(it) = mm.iterations.iter_mut().find(|it| it.idx == iter_idx) {
                    it.gate.games_completed = completed as u64;
                    it.gate.games_target = total_games as u64;
                    if it.gate.first_game_started_ts_ms.is_none() {
                        it.gate.first_game_started_ts_ms = first_game_min;
                    }
                }
                let _ = yz_logging::write_manifest_atomic(&run_json, &mm);
            }
            last_manifest_ts = std::time::Instant::now();
        }
        std::thread::sleep(Duration::from_millis(25));
    }

    // Aggregate results.
    #[derive(serde::Deserialize)]
    struct GateWorkerResult {
        games: u32,
        cand_wins: u32,
        cand_losses: u32,
        draws: u32,
        cand_score_diff_sum: i64,
        cand_score_diff_sumsq: f64,
        cand_score_sum: i64,
        best_score_sum: i64,
    }

    let mut partial = yz_eval::GatePartial::default();
    for (wid, _) in &per_worker {
        let path = gate_workers_dir.join(format!("result_{wid:03}.json"));
        let bytes = std::fs::read(&path)?;
        let r: GateWorkerResult =
            serde_json::from_slice(&bytes).map_err(|e| ControllerError::Fs(std::io::Error::other(e)))?;
        let mut p = yz_eval::GatePartial::default();
        p.games = r.games;
        p.cand_wins = r.cand_wins;
        p.cand_losses = r.cand_losses;
        p.draws = r.draws;
        p.cand_score_diff_sum = r.cand_score_diff_sum;
        p.cand_score_diff_sumsq = r.cand_score_diff_sumsq;
        p.cand_score_sum = r.cand_score_sum;
        p.best_score_sum = r.best_score_sum;
        partial.merge(&p);
    }

    let report = partial.into_report(yz_eval::GatePlan {
        schedule: Vec::new(),
        seeds: plan.seeds,
        seeds_hash: plan.seeds_hash,
        warnings: plan.warnings,
    });

    // Oracle diagnostics: compute once in controller after workers finish.
    // This avoids building the oracle DP table in every gate-worker process.
    let (oracle_overall, oracle_mark, oracle_reroll, oracle_keepall_ignored) = {
        let oracle = yz_oracle::oracle();
        let mut total: u64 = 0;
        let mut matched: u64 = 0;
        let mut total_mark: u64 = 0;
        let mut matched_mark: u64 = 0;
        let mut total_reroll: u64 = 0;
        let mut matched_reroll: u64 = 0;
        let mut keepall_ignored: u64 = 0;

        fn should_ignore(oa: yz_oracle::Action, rerolls_left: u8) -> bool {
            matches!(oa, yz_oracle::Action::KeepMask { mask: 31 } if rerolls_left > 0)
        }
        fn oracle_action_to_core(oa: yz_oracle::Action) -> yz_core::Action {
            match oa {
                yz_oracle::Action::Mark { cat } => yz_core::Action::Mark(cat),
                yz_oracle::Action::KeepMask { mask } => yz_core::Action::KeepMask(mask),
            }
        }

        for (wid, _) in &per_worker {
            let p = logs_gate_workers_dir
                .join(format!("worker_{wid:03}"))
                .join("oracle_diag.ndjson");
            let Ok(bytes) = std::fs::read(&p) else {
                continue;
            };
            for line in bytes.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                let Ok(ev) = serde_json::from_slice::<yz_eval::OracleDiagEvent>(line) else {
                    continue;
                };
                let chosen = yz_core::index_to_action(ev.chosen_action_idx);
                let (oa, _ev) =
                    oracle.best_action(ev.avail_mask, ev.upper_total_cap, ev.dice_sorted, ev.rerolls_left);
                if should_ignore(oa, ev.rerolls_left) {
                    keepall_ignored += 1;
                    continue;
                }
                let expected = oracle_action_to_core(oa);
                let is_match = expected == chosen;
                total += 1;
                if is_match {
                    matched += 1;
                }
                match chosen {
                    yz_core::Action::Mark(_) => {
                        total_mark += 1;
                        if is_match {
                            matched_mark += 1;
                        }
                    }
                    yz_core::Action::KeepMask(_) => {
                        total_reroll += 1;
                        if is_match {
                            matched_reroll += 1;
                        }
                    }
                }
            }
        }

        let overall = if total == 0 { 0.0 } else { matched as f64 / total as f64 };
        let mark = if total_mark == 0 {
            0.0
        } else {
            matched_mark as f64 / total_mark as f64
        };
        let reroll = if total_reroll == 0 {
            0.0
        } else {
            matched_reroll as f64 / total_reroll as f64
        };
        (overall, mark, reroll, keepall_ignored)
    };

    let wr = report.win_rate();
    m.gate_games = Some(report.games as u64);
    m.gate_win_rate = Some(wr);
    m.gate_draw_rate = Some(report.draw_rate);
    m.gate_seeds_hash = Some(report.seeds_hash.clone());
    m.gate_oracle_match_rate_overall = Some(oracle_overall);
    m.gate_oracle_match_rate_mark = Some(oracle_mark);
    m.gate_oracle_match_rate_reroll = Some(oracle_reroll);
    m.gate_oracle_keepall_ignored = Some(oracle_keepall_ignored);

    // Emit gate_summary metrics event (best-effort).
    {
        let decision = if wr + 1e-12 < cfg.gating.win_rate_threshold {
            "reject"
        } else {
            "promote"
        };
        let metrics_path = run_dir.join("logs").join("metrics.ndjson");
        if let Ok(mut metrics) = yz_logging::NdjsonWriter::open_append_with_flush(&metrics_path, 1)
        {
            let ev = yz_logging::MetricsGateSummaryV1 {
                event: "gate_summary",
                ts_ms: yz_logging::now_ms(),
                v: yz_logging::VersionInfoV1 {
                    protocol_version: m.protocol_version,
                    feature_schema_id: m.feature_schema_id,
                    action_space_id: "oracle_keepmask_v1",
                    ruleset_id: "swedish_scandinavian_v1",
                },
                run_id: m.run_id.clone(),
                git_hash: m.git_hash.clone(),
                config_snapshot: m.config_snapshot.clone(),
                decision: decision.to_string(),
                games: report.games,
                wins: report.cand_wins,
                losses: report.cand_losses,
                draws: report.draws,
                win_rate: wr,
                mean_score_diff: report.mean_score_diff(),
                mean_cand_score: Some(report.mean_cand_score()),
                mean_best_score: Some(report.mean_best_score()),
                score_diff_se: report.score_diff_se,
                score_diff_ci95_low: report.score_diff_ci95_low,
                score_diff_ci95_high: report.score_diff_ci95_high,
                seeds_hash: report.seeds_hash.clone(),
                oracle_match_rate_overall: oracle_overall,
                oracle_match_rate_mark: oracle_mark,
                oracle_match_rate_reroll: oracle_reroll,
                oracle_keepall_ignored: oracle_keepall_ignored,
            };
            let _ = metrics.write_event(&ev);
            let _ = metrics.flush();
        }
    }

    // Also update current iteration entry (final values).
    if let Some(it) = m.iterations.iter_mut().find(|it| it.idx == iter_idx) {
        it.gate.games_target = report.games as u64;
        it.gate.games_completed = report.games as u64;
        it.gate.win_rate = Some(wr);
        it.gate.draw_rate = Some(report.draw_rate);
        it.gate.mean_cand_score = Some(report.mean_cand_score());
        it.gate.mean_best_score = Some(report.mean_best_score());
        it.gate.ended_ts_ms = Some(yz_logging::now_ms());
        it.oracle.match_rate_overall = Some(oracle_overall);
        it.oracle.match_rate_mark = Some(oracle_mark);
        it.oracle.match_rate_reroll = Some(oracle_reroll);
        it.oracle.keepall_ignored = Some(oracle_keepall_ignored);
    }
    yz_logging::write_manifest_atomic(&run_json, &m)?;

    Ok(())
}

#[cfg(test)]
mod cancel_tests {
    use super::*;
    use std::fs;

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
    fn merge_replay_workers_renumbers_shards_without_collisions() {
        let dir = tempfile::tempdir().unwrap();
        let run_dir = dir.path();
        let replay_workers = run_dir.join("replay_workers");
        let replay = run_dir.join("replay");
        fs::create_dir_all(replay_workers.join("worker_000")).unwrap();
        fs::create_dir_all(replay_workers.join("worker_001")).unwrap();
        fs::create_dir_all(&replay).unwrap();

        // Create one shard per worker (both idx=0 in their local dirs).
        fs::write(
            replay_workers.join("worker_000").join("shard_000000.safetensors"),
            b"st0",
        )
        .unwrap();
        fs::write(
            replay_workers.join("worker_000").join("shard_000000.meta.json"),
            b"meta0",
        )
        .unwrap();

        fs::write(
            replay_workers.join("worker_001").join("shard_000000.safetensors"),
            b"st1",
        )
        .unwrap();
        fs::write(
            replay_workers.join("worker_001").join("shard_000000.meta.json"),
            b"meta1",
        )
        .unwrap();

        merge_replay_workers(&replay_workers, &replay).unwrap();

        // Expect two shards in the merged dir: idx 0 and 1.
        assert!(replay.join("shard_000000.safetensors").exists());
        assert!(replay.join("shard_000000.meta.json").exists());
        assert!(replay.join("shard_000001.safetensors").exists());
        assert!(replay.join("shard_000001.meta.json").exists());

        // Worker dirs should no longer contain those files (renamed/moved).
        assert!(!replay_workers
            .join("worker_000")
            .join("shard_000000.safetensors")
            .exists());
        assert!(!replay_workers
            .join("worker_001")
            .join("shard_000000.safetensors")
            .exists());
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
        let cfg = yz_core::Config::default();
        let cmd = build_train_command(run_dir, &cfg, "python", 0);

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
