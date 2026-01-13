//! Ratatui terminal UI (TUI) for configuring and monitoring runs.
//!
//! v1 scope:
//! - basic screen routing + key handling
//! - run picker (list/create)

mod config_io;
mod form;
mod validate;

use std::collections::HashMap;
use std::io;
use std::fs::File;
use std::io::{Read as _, Seek as _, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::{execute, ExecutableCommand};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Gauge, List, ListItem, Paragraph, Sparkline};
use ratatui::Terminal;

use crate::form::{EditMode, FieldId, FormState, Section, StepSize, ALL_FIELDS};
use yz_core::config::TemperatureSchedule;
use yz_logging::RunManifestV1;

#[derive(Debug, Clone, serde::Deserialize)]
struct ReplaySnapshotV1 {
    #[allow(dead_code)]
    snapshot_version: u32,
    #[allow(dead_code)]
    replay_dir: String,
    shards: Vec<ReplaySnapshotShardV1>,
    #[allow(dead_code)]
    total_samples: u64,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ReplaySnapshotShardV1 {
    safetensors: String,
    #[allow(dead_code)]
    meta: String,
    #[allow(dead_code)]
    num_samples: u64,
}

#[derive(Debug, Clone)]
struct ReplayShardView {
    #[allow(dead_code)]
    shard_id: u32,
    origin_iter: Option<u32>,
    num_samples: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Home,
    NamingRun, // Input mode for naming a new run
    ExtendingRun, // Input mode for naming an extended run (fork)
    Config,
    Dashboard,
    Search,
    System,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DashboardTab {
    Performance,
    Learning,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default, serde::Deserialize)]
struct LearnSummaryNdjsonV1 {
    #[allow(dead_code)]
    #[serde(default)]
    event: Option<String>,
    #[serde(default)]
    iter_idx: Option<u32>,

    // Dataset + progress
    #[serde(default)]
    total_samples: Option<u64>,
    #[serde(default)]
    batch_size: Option<u64>,
    #[serde(default)]
    steps_target: Option<u64>,
    #[serde(default)]
    steps_completed: Option<u64>,
    #[serde(default)]
    samples_s_mean: Option<f64>,

    // Replay staleness
    #[serde(default)]
    age_wall_s_p50: Option<f64>,
    #[serde(default)]
    age_wall_s_p95: Option<f64>,
    #[serde(default)]
    age_shard_idx_p50: Option<f64>,
    #[serde(default)]
    age_shard_idx_p95: Option<f64>,

    // Policy target spikiness
    #[serde(default)]
    pi_entropy_mean: Option<f64>,
    #[serde(default)]
    pi_entropy_p50: Option<f64>,
    #[serde(default)]
    pi_entropy_p95: Option<f64>,
    // Normalized entropy H(pi)/log(n_legal); helps compare across varying legal counts.
    #[serde(default)]
    pi_entropy_norm_p50: Option<f64>,
    #[serde(default)]
    pi_entropy_norm_p95: Option<f64>,
    #[serde(default)]
    pi_top1_p50: Option<f64>,
    #[serde(default)]
    pi_top1_p95: Option<f64>,
    // Split by inferred state type using the legal mask structure (oracle_keepmask_v1).
    #[serde(default)]
    pi_entropy_mean_mark: Option<f64>,
    #[serde(default)]
    pi_entropy_mean_reroll: Option<f64>,
    #[serde(default)]
    pi_top1_p95_mark: Option<f64>,
    #[serde(default)]
    pi_top1_p95_reroll: Option<f64>,
    #[serde(default)]
    pi_eff_actions_p50: Option<f64>,
    #[serde(default)]
    pi_eff_actions_p95: Option<f64>,
    #[serde(default)]
    pi_eff_actions_p50_mark: Option<f64>,
    #[serde(default)]
    pi_eff_actions_p50_reroll: Option<f64>,
    // Legal action count diagnostics (derived from legal mask).
    #[serde(default)]
    n_legal_p50: Option<f64>,
    #[serde(default)]
    n_legal_p95: Option<f64>,

    // Value predictions
    #[serde(default)]
    v_pred_mean: Option<f64>,
    #[serde(default)]
    v_pred_std: Option<f64>,
    #[serde(default)]
    v_pred_p05: Option<f64>,
    #[serde(default)]
    v_pred_p50: Option<f64>,
    #[serde(default)]
    v_pred_p95: Option<f64>,
    #[serde(default)]
    v_pred_sat_frac: Option<f64>,

    // Calibration
    #[serde(default)]
    ece: Option<f64>,

    // Policy alignment (optional; emitted by trainer when available)
    #[serde(default)]
    pi_model_entropy_mean: Option<f64>,
    #[serde(default)]
    pi_model_entropy_p50: Option<f64>,
    #[serde(default)]
    pi_model_entropy_p95: Option<f64>,
    #[serde(default)]
    pi_kl_mean: Option<f64>,
    #[serde(default)]
    pi_kl_p50: Option<f64>,
    #[serde(default)]
    pi_kl_p95: Option<f64>,
    #[serde(default)]
    pi_entropy_gap_mean: Option<f64>,

    // Legal-renormalized alignment (ignores model illegal mass).
    #[serde(default)]
    pi_model_entropy_legal_mean: Option<f64>,
    #[serde(default)]
    pi_model_entropy_legal_p50: Option<f64>,
    #[serde(default)]
    pi_model_entropy_legal_p95: Option<f64>,
    #[serde(default)]
    pi_kl_legal_mean: Option<f64>,
    #[serde(default)]
    pi_kl_legal_p50: Option<f64>,
    #[serde(default)]
    pi_kl_legal_p95: Option<f64>,
    #[serde(default)]
    p_top1_legal_p95: Option<f64>,

    // Illegal-mass diagnostics (fractions in [0,1]).
    // - pi_illegal_*: mass in the *replay target* pi on illegal actions (before trainer masking)
    // - p_illegal_*: mass in the *model policy* on illegal actions
    #[serde(default)]
    pi_illegal_mass_mean: Option<f64>,
    #[serde(default)]
    pi_illegal_mass_p95: Option<f64>,
    #[serde(default)]
    p_illegal_mass_mean: Option<f64>,
    #[serde(default)]
    p_illegal_mass_p95: Option<f64>,

    // Calibration bins (for a tiny reliability diagram).
    #[serde(default)]
    calibration_bins: Option<Vec<LearnCalibBinV1>>,

    // Training throughput
    #[serde(default)]
    step_ms_p50: Option<f64>,
    #[serde(default)]
    step_ms_p95: Option<f64>,
}

#[derive(Debug, Clone, Default, serde::Deserialize)]
struct LearnCalibBinV1 {
    #[serde(default)]
    count: u64,
    #[serde(default)]
    mean_pred: Option<f64>,
    #[serde(default)]
    mean_z: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default, serde::Deserialize)]
struct SelfplaySummaryNdjsonV1 {
    #[allow(dead_code)]
    #[serde(default)]
    event: Option<String>,
    #[serde(default)]
    iter_idx: Option<u32>,

    // Throughput / volume.
    #[serde(default)]
    moves_executed: Option<u64>,
    #[serde(default)]
    moves_s_mean: Option<f64>,

    // Visit-policy (MCTS pi) shape.
    #[serde(default)]
    pi_entropy_mean: Option<f64>,
    #[serde(default)]
    pi_entropy_p95: Option<f64>,
    #[serde(default)]
    pi_max_p_p95: Option<f64>,
    #[serde(default)]
    pi_eff_actions_p50: Option<f64>,
    #[serde(default)]
    pi_eff_actions_p95: Option<f64>,

    // Search stability + improvement.
    #[serde(default)]
    fallbacks_rate: Option<f64>,
    #[serde(default)]
    pending_collisions_per_move: Option<f64>,
    #[serde(default)]
    prior_kl_mean: Option<f64>,
    #[serde(default)]
    prior_argmax_overturn_rate: Option<f64>,
    #[serde(default)]
    noise_kl_mean: Option<f64>,
    #[serde(default)]
    noise_argmax_flip_rate: Option<f64>,

    // Search quality (optional; newer controller versions).
    #[serde(default)]
    visit_entropy_norm_p50: Option<f64>,
    #[serde(default)]
    late_eval_discard_frac: Option<f64>,
    #[serde(default)]
    delta_root_value_mean: Option<f64>,

    // Game stats.
    #[serde(default)]
    game_ply_p95: Option<f64>,
    #[serde(default)]
    score_diff_p95: Option<f64>,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Default, serde::Deserialize)]
struct InferSnapshotNdjsonV1 {
    #[allow(dead_code)]
    #[serde(default)]
    event: Option<String>,
    #[serde(default)]
    ts_ms: Option<u64>,
    #[serde(default)]
    iter_idx: Option<u32>,
    #[serde(default)]
    metrics_bind: Option<String>,

    // Server snapshot.
    #[serde(default)]
    queue_depth: Option<u64>,
    #[serde(default)]
    requests_s: Option<f64>,
    #[serde(default)]
    batches_s: Option<f64>,
    #[serde(default)]
    batch_size_mean: Option<f64>,
    #[serde(default)]
    batch_size_p50: Option<f64>,
    #[serde(default)]
    batch_size_p95: Option<f64>,
    #[serde(default)]
    underfill_frac_mean: Option<f64>,
    #[serde(default)]
    full_frac_mean: Option<f64>,
    #[serde(default)]
    flush_full_s: Option<f64>,
    #[serde(default)]
    flush_deadline_s: Option<f64>,

    #[serde(default)]
    queue_wait_us_p50: Option<f64>,
    #[serde(default)]
    queue_wait_us_p95: Option<f64>,
    #[serde(default)]
    build_ms_p50: Option<f64>,
    #[serde(default)]
    build_ms_p95: Option<f64>,
    #[serde(default)]
    forward_ms_p50: Option<f64>,
    #[serde(default)]
    forward_ms_p95: Option<f64>,
    #[serde(default)]
    post_ms_p50: Option<f64>,
    #[serde(default)]
    post_ms_p95: Option<f64>,

    // Worker snapshot (optional).
    #[serde(default)]
    workers_seen: Option<u32>,
    #[serde(default)]
    inflight_sum: Option<u64>,
    #[serde(default)]
    inflight_max: Option<u64>,
    #[serde(default)]
    rtt_p95_us_min: Option<u64>,
    #[serde(default)]
    rtt_p95_us_med: Option<u64>,
    #[serde(default)]
    rtt_p95_us_max: Option<u64>,
    #[serde(default)]
    would_block_frac: Option<f64>,
}

#[derive(Debug, Clone)]
struct SystemLiveSample {
    ts: Instant,
    // Server
    queue_depth: Option<u64>,
    requests_s: Option<f64>,
    batches_s: Option<f64>,
    batch_size_mean: Option<f64>,
    batch_size_p95: Option<f64>,
    underfill_frac_mean: Option<f64>,
    full_frac_mean: Option<f64>,
    flush_full_s: Option<f64>,
    flush_deadline_s: Option<f64>,
    queue_wait_us_p95: Option<f64>,
    forward_ms_p95: Option<f64>,
    // Workers (selfplay only)
    inflight_sum: Option<u64>,
    inflight_max: Option<u64>,
    rtt_p95_us_med: Option<u64>,
    would_block_frac: Option<f64>,
    // Parse/transport
    source: &'static str, // "live" | "playback"
}

#[derive(Debug)]
struct App {
    screen: Screen,
    status: String,
    runs_dir: PathBuf,
    runs: Vec<String>,
    selected: usize,

    active_run_id: Option<String>,
    extend_source_run_id: Option<String>,
    extend_include_replay: bool,
    extend_return_to: Screen,
    cfg: yz_core::Config,
    form: FormState,

    dashboard_manifest: Option<RunManifestV1>,
    dashboard_err: Option<String>,
    dashboard_planned_total_iterations: Option<u32>,
    dashboard_planned_loaded_for_run: Option<String>,
    dashboard_tab: DashboardTab,
    /// Scroll offset (in rows) for the dashboard Performance iteration list.
    dashboard_iter_scroll: usize,
    /// If true, keep the current iteration row visible (auto-follow). Any manual scroll disables it.
    dashboard_iter_follow: bool,

    /// Search/MCTS screen selection state.
    search_selected_iter: u32,
    /// If true, auto-follow the controller's current iteration.
    search_follow: bool,

    // Learning dashboard state (tail metrics.ndjson).
    learn_loaded_for_run: Option<String>,
    learn_metrics_offset: u64,
    learn_metrics_partial: String,
    learn_by_iter: HashMap<u32, LearnSummaryNdjsonV1>,
    selfplay_by_iter: HashMap<u32, SelfplaySummaryNdjsonV1>,
    infer_snapshots: Vec<InferSnapshotNdjsonV1>,
    infer_last_ts_ms: u64,

    // Replay buffer visualization (best-effort, cheap: small JSON snapshots + meta filenames).
    replay_loaded_for_run: Option<String>,
    replay_last_poll: Instant,
    replay_scanned_up_to_iter: i32,
    replay_max_snapshot_iter: Option<u32>,
    replay_origin_by_shard: HashMap<u32, u32>, // shard_id -> first snapshot iter where it appears
    replay_current: Vec<ReplayShardView>,
    replay_total_samples_current: Option<u64>,
    replay_err: Option<String>,

    // System/Inference screen live state (1 Hz polling).
    system_agent: ureq::Agent,
    system_last_poll: Instant,
    system_live: Option<SystemLiveSample>,
    system_live_err: Option<String>,
    system_series_queue_depth: Vec<u64>,
    system_series_rtt_p95_us: Vec<u64>,
    system_series_rps_x10: Vec<u64>,
    system_series_batch_mean_x100: Vec<u64>,
    system_series_would_block_x1000: Vec<u64>,
    system_prev_ts: Option<Instant>,
    system_prev_req_total: Option<f64>,
    system_prev_batches_total: Option<f64>,
    system_prev_flush_full_total: Option<f64>,
    system_prev_flush_deadline_total: Option<f64>,
    system_prev_steps_sum: Option<u64>,
    system_prev_wb_sum: Option<u64>,

    iter: Option<yz_controller::IterationHandle>,

    /// Shutdown state: set when user cancels (x) or quits (q) during an active run.
    shutdown_requested: bool,
    /// If true, exit the TUI after shutdown completes.
    shutdown_exit_after: bool,

    /// Input buffer for naming a new run.
    naming_input: String,
    /// Input buffer for naming an extended run.
    extend_input: String,
}

impl App {
    fn new(runs_dir: PathBuf) -> Self {
        let system_agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_millis(200))
            .timeout_read(Duration::from_millis(250))
            .timeout_write(Duration::from_millis(250))
            .build();
        Self {
            screen: Screen::Home,
            status: "q: quit | r: refresh | n: new run".to_string(),
            runs_dir,
            runs: Vec::new(),
            selected: 0,
            active_run_id: None,
            extend_source_run_id: None,
            extend_include_replay: false,
            extend_return_to: Screen::Home,
            cfg: crate::config_io::default_cfg_for_new_run(),
            form: FormState::default(),
            dashboard_manifest: None,
            dashboard_err: None,
            dashboard_planned_total_iterations: None,
            dashboard_planned_loaded_for_run: None,
            dashboard_tab: DashboardTab::Performance,
            dashboard_iter_scroll: 0,
            dashboard_iter_follow: true,
            search_selected_iter: 0,
            search_follow: true,
            learn_loaded_for_run: None,
            learn_metrics_offset: 0,
            learn_metrics_partial: String::new(),
            learn_by_iter: HashMap::new(),
            selfplay_by_iter: HashMap::new(),
            infer_snapshots: Vec::new(),
            infer_last_ts_ms: 0,

            replay_loaded_for_run: None,
            replay_last_poll: Instant::now(),
            replay_scanned_up_to_iter: -1,
            replay_max_snapshot_iter: None,
            replay_origin_by_shard: HashMap::new(),
            replay_current: Vec::new(),
            replay_total_samples_current: None,
            replay_err: None,

            system_agent,
            system_last_poll: Instant::now(),
            system_live: None,
            system_live_err: None,
            system_series_queue_depth: Vec::new(),
            system_series_rtt_p95_us: Vec::new(),
            system_series_rps_x10: Vec::new(),
            system_series_batch_mean_x100: Vec::new(),
            system_series_would_block_x1000: Vec::new(),
            system_prev_ts: None,
            system_prev_req_total: None,
            system_prev_batches_total: None,
            system_prev_flush_full_total: None,
            system_prev_flush_deadline_total: None,
            system_prev_steps_sum: None,
            system_prev_wb_sum: None,
            iter: None,
            shutdown_requested: false,
            shutdown_exit_after: false,
            naming_input: String::new(),
            extend_input: String::new(),
        }
    }

    fn refresh_runs(&mut self) {
        self.runs = list_runs(&self.runs_dir);
        if self.selected >= self.runs.len() {
            self.selected = 0;
        }
        if self.runs.is_empty() {
            self.status = format!(
                "No runs found. Press 'n' to create one in {}",
                self.runs_dir.display()
            );
        } else {
            self.status =
                "q: quit | r: refresh | n: new run | e: extend | Enter: open | ↑/↓ select | PgUp/PgDn | Home/End"
                    .to_string();
        }
    }

    fn create_run_with_name(&mut self, name: &str) -> io::Result<()> {
        std::fs::create_dir_all(&self.runs_dir)?;
        // Sanitize name for filesystem: replace invalid chars with underscore
        let sanitized: String = name
            .chars()
            .map(|c| {
                if c.is_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        // Use sanitized name if non-empty, otherwise fallback to timestamp
        let id = if sanitized.is_empty() {
            let ts = yz_logging::now_ms();
            format!("run_{ts}")
        } else {
            sanitized
        };
        // Ensure unique name by appending timestamp if directory exists
        let mut final_id = id.clone();
        if self.runs_dir.join(&final_id).exists() {
            let ts = yz_logging::now_ms();
            final_id = format!("{id}_{ts}");
        }
        let dir = self.runs_dir.join(&final_id);
        std::fs::create_dir_all(dir.join("logs"))?;
        std::fs::create_dir_all(dir.join("models"))?;
        std::fs::create_dir_all(dir.join("replay"))?;
        self.status = format!("Created {final_id}");
        self.refresh_runs();
        self.selected = self.runs.iter().position(|r| r == &final_id).unwrap_or(0);
        Ok(())
    }

    fn begin_extend_selected_run(&mut self, return_to: Screen) {
        if self.runs.is_empty() {
            return;
        }
        let src = self.runs[self.selected].clone();
        self.extend_source_run_id = Some(src);
        self.extend_include_replay = false;
        self.extend_input.clear();
        self.extend_return_to = return_to;
        self.screen = Screen::ExtendingRun;
        self.status = "Extend run: type new name | Space: toggle replay copy | Enter: confirm | Esc: cancel".to_string();
    }

    fn begin_extend_active_run(&mut self, return_to: Screen) {
        let Some(src) = self.active_run_id.clone() else {
            return;
        };
        self.extend_source_run_id = Some(src);
        self.extend_include_replay = false;
        self.extend_input.clear();
        self.extend_return_to = return_to;
        self.screen = Screen::ExtendingRun;
        self.status = "Extend run: type new name | Space: toggle replay copy | Enter: confirm | Esc: cancel".to_string();
    }

    fn extend_run_with_name(&mut self, name: &str) -> io::Result<String> {
        let Some(src_id) = self.extend_source_run_id.clone() else {
            return Err(io::Error::other("no source run selected"));
        };
        std::fs::create_dir_all(&self.runs_dir)?;

        let src_dir = self.runs_dir.join(&src_id);
        let src_run_json = src_dir.join("run.json");
        if !src_run_json.exists() {
            return Err(io::Error::other(format!(
                "source run has no run.json: {}",
                src_run_json.display()
            )));
        }
        let src_manifest = yz_logging::read_manifest(&src_run_json)
            .map_err(|e| io::Error::other(format!("failed to read source run.json: {e}")))?;

        // Sanitize + ensure unique id in runs/
        let sanitized: String = name
            .chars()
            .map(|c| if c.is_alphanumeric() || c == '_' || c == '-' { c } else { '_' })
            .collect();
        if sanitized.trim().is_empty() {
            return Err(io::Error::other("new run name must be non-empty"));
        }
        let base_id = sanitized;
        let mut final_id = base_id.clone();
        if self.runs_dir.join(&final_id).exists() {
            let ts = yz_logging::now_ms();
            final_id = format!("{base_id}_{ts}");
        }

        let dst_dir = self.runs_dir.join(&final_id);
        std::fs::create_dir_all(dst_dir.join("logs"))?;
        std::fs::create_dir_all(dst_dir.join("models"))?;
        std::fs::create_dir_all(dst_dir.join("replay"))?;

        // Copy config snapshot/draft.
        let src_cfg = src_dir.join("config.yaml");
        let src_draft = src_dir.join(crate::config_io::CONFIG_DRAFT_NAME);
        let dst_cfg = dst_dir.join("config.yaml");
        let dst_draft = dst_dir.join(crate::config_io::CONFIG_DRAFT_NAME);
        if src_cfg.exists() {
            std::fs::copy(&src_cfg, &dst_cfg)?;
        } else if src_draft.exists() {
            std::fs::copy(&src_draft, &dst_cfg)?;
        }
        if src_draft.exists() {
            let _ = std::fs::copy(&src_draft, &dst_draft);
        }

        // Copy models (best.pt at minimum).
        let src_best = src_dir.join("models").join("best.pt");
        let dst_best = dst_dir.join("models").join("best.pt");
        if src_best.exists() {
            let _ = std::fs::copy(&src_best, &dst_best);
        }

        // Optionally copy replay/.
        if self.extend_include_replay {
            let src_replay = src_dir.join("replay");
            let dst_replay = dst_dir.join("replay");
            if src_replay.exists() {
                copy_dir_recursive(&src_replay, &dst_replay)?;
            }
        }

        // Load the copied config. If the source run already completed all planned iterations,
        // bump total_iterations so the extended run is runnable immediately (common expectation).
        //
        // Without this, extending a completed run will show "done (completed N/N)" and pressing
        // 'g' won't start anything because controller_iteration_idx == total_iterations.
        let mut cfg = yz_core::Config::load(&dst_cfg).unwrap_or_default();
        let cur = src_manifest.controller_iteration_idx;
        let planned = cfg.controller.total_iterations.unwrap_or(cur.max(1));
        let mut bumped_total: Option<u32> = None;
        if planned <= cur {
            let new_total = cur.saturating_add(1).max(1);
            cfg.controller.total_iterations = Some(new_total);
            bumped_total = Some(new_total);
            // Keep draft (if present) consistent with the snapshot, since Config screen prefers draft.
            if dst_draft.exists() {
                let _ = crate::config_io::save_cfg_draft_atomic(&dst_dir, &cfg);
            }
        }

        // Normalize + hash into config.yaml, then write a fresh run.json that continues counting
        // from the source controller_iteration_idx.
        let (rel_cfg, cfg_hash) = yz_logging::write_config_snapshot_atomic(&dst_dir, &cfg)
            .map_err(|e| io::Error::other(format!("failed to write config snapshot: {e}")))?;

        let now = yz_logging::now_ms();
        let best_ckpt = if dst_best.exists() {
            Some("models/best.pt".to_string())
        } else {
            None
        };
        let mut m = RunManifestV1 {
            run_manifest_version: src_manifest.run_manifest_version,
            run_id: final_id.clone(),
            created_ts_ms: now,
            protocol_version: src_manifest.protocol_version,
            feature_schema_id: src_manifest.feature_schema_id,
            action_space_id: src_manifest.action_space_id.clone(),
            ruleset_id: src_manifest.ruleset_id.clone(),
            git_hash: yz_logging::try_git_hash().or(src_manifest.git_hash.clone()),
            config_hash: Some(cfg_hash.clone()),
            config_snapshot: Some(rel_cfg),
            config_snapshot_hash: Some(cfg_hash),
            replay_dir: src_manifest.replay_dir.clone(),
            logs_dir: src_manifest.logs_dir.clone(),
            models_dir: src_manifest.models_dir.clone(),
            selfplay_games_completed: 0,
            train_step: 0,
            best_checkpoint: best_ckpt,
            candidate_checkpoint: None,
            best_promoted_iter: src_manifest.best_promoted_iter,
            train_last_loss_total: None,
            train_last_loss_policy: None,
            train_last_loss_value: None,
            promotion_decision: None,
            promotion_ts_ms: None,
            gate_games: None,
            gate_win_rate: None,
            gate_draw_rate: None,
            gate_wins: None,
            gate_losses: None,
            gate_draws: None,
            gate_ci95_low: None,
            gate_ci95_high: None,
            gate_sprt: None,
            gate_seeds_hash: None,
            gate_oracle_match_rate_overall: None,
            gate_oracle_match_rate_mark: None,
            gate_oracle_match_rate_reroll: None,
            gate_oracle_keepall_ignored: None,
            controller_phase: Some("idle".to_string()),
            controller_status: Some(format!(
                "extended from {src_id} (continue at iter {}){}",
                src_manifest.controller_iteration_idx,
                bumped_total
                    .map(|t| format!(", bumped total_iterations to {t}"))
                    .unwrap_or_default()
            )),
            controller_last_ts_ms: Some(now),
            controller_error: None,
            model_reloads: 0,
            controller_iteration_idx: src_manifest.controller_iteration_idx,
            iterations: Vec::new(),
        };
        // If the source had a controller error, do not propagate it.
        m.controller_error = None;
        yz_logging::write_manifest_atomic(&dst_dir.join("run.json"), &m)
            .map_err(|e| io::Error::other(format!("failed to write run.json: {e}")))?;

        Ok(final_id)
    }

    fn enter_selected_run(&mut self) {
        if self.runs.is_empty() {
            return;
        }
        self.active_run_id = Some(self.runs[self.selected].clone());
        if let Some(run_dir) = self.run_dir() {
            let (cfg, msg) = crate::config_io::load_cfg_for_run(&run_dir);
            self.cfg = cfg;
            self.form = FormState::default();
            if let Some(msg) = msg {
                self.form.last_validation_error = None;
                self.status = msg;
            }
        } else {
            self.cfg = yz_core::Config::default();
            self.form = FormState::default();
        }
        self.screen = Screen::Config;
        self.status = config_help(&self.form);
    }

    fn run_dir(&self) -> Option<PathBuf> {
        self.active_run_id.as_ref().map(|id| self.runs_dir.join(id))
    }

    fn save_config_draft(&mut self) {
        let Some(run_dir) = self.run_dir() else {
            self.status = "No run selected".to_string();
            return;
        };
        if let Err(e) = std::fs::create_dir_all(&run_dir) {
            self.status = format!("save failed: {e}");
            return;
        }
        match crate::validate::validate_config(&self.cfg) {
            Ok(()) => {}
            Err(e) => {
                self.form.last_validation_error = Some(e.clone());
                self.status = format!("not saved (invalid): {e}");
                return;
            }
        }
        match crate::config_io::save_cfg_draft_atomic(&run_dir, &self.cfg) {
            Ok(()) => self.status = format!("Saved {}", crate::config_io::CONFIG_DRAFT_NAME),
            Err(e) => self.status = format!("save failed: {e}"),
        }
    }

    fn refresh_dashboard(&mut self) {
        let Some(run_dir) = self.run_dir() else {
            self.dashboard_manifest = None;
            self.dashboard_err = None;
            self.dashboard_planned_total_iterations = None;
            self.dashboard_planned_loaded_for_run = None;
            self.learn_loaded_for_run = None;
            self.learn_metrics_offset = 0;
            self.learn_metrics_partial.clear();
            self.learn_by_iter.clear();
            self.selfplay_by_iter.clear();
            return;
        };
        let run_json = run_dir.join("run.json");
        if !run_json.exists() {
            self.dashboard_manifest = None;
            self.dashboard_err = Some("run.json not found (start an iteration first)".to_string());
            return;
        }
        match yz_logging::read_manifest(&run_json) {
            Ok(m) => {
                self.dashboard_manifest = Some(m);
                self.dashboard_err = None;
            }
            Err(e) => {
                self.dashboard_manifest = None;
                self.dashboard_err = Some(format!("failed to read run.json: {e}"));
            }
        }

        // Keep Search screen selection pinned to the current iteration when in follow mode.
        if self.search_follow {
            if let Some(m) = &self.dashboard_manifest {
                self.search_selected_iter = m.controller_iteration_idx;
            }
        }

        // Load planned iterations (best-effort) once per active run.
        let rid = self.active_run_id.clone().unwrap_or_default();
        if self.dashboard_planned_loaded_for_run.as_deref() != Some(rid.as_str()) {
            self.dashboard_planned_loaded_for_run = Some(rid);
            self.dashboard_planned_total_iterations = None;
            let cfg_path = run_dir.join("config.yaml");
            if cfg_path.exists() {
                if let Ok(cfg) = yz_core::Config::load(&cfg_path) {
                    self.dashboard_planned_total_iterations = cfg.controller.total_iterations;
                }
            }
        }

        self.refresh_learning_metrics(&run_dir);
        self.refresh_replay_buffer(&run_dir);
    }

    fn parse_shard_id(name: &str) -> Option<u32> {
        // Accept "shard_000240.safetensors" or "shard_000240.meta.json"
        let s = name.strip_prefix("shard_")?;
        let digits: String = s.chars().take_while(|c| c.is_ascii_digit()).collect();
        if digits.is_empty() {
            return None;
        }
        digits.parse::<u32>().ok()
    }

    fn refresh_replay_buffer(&mut self, run_dir: &Path) {
        // Rate limit: scanning snapshot files is small but avoid doing it at 4Hz.
        if self.replay_last_poll.elapsed() < Duration::from_secs(2) {
            return;
        }
        self.replay_last_poll = Instant::now();

        let rid = self.active_run_id.clone().unwrap_or_default();
        if self.replay_loaded_for_run.as_deref() != Some(rid.as_str()) {
            self.replay_loaded_for_run = Some(rid);
            self.replay_scanned_up_to_iter = -1;
            self.replay_max_snapshot_iter = None;
            self.replay_origin_by_shard.clear();
            self.replay_current.clear();
            self.replay_total_samples_current = None;
            self.replay_err = None;
        }

        // 1) Discover available replay snapshots.
        let mut snapshot_iters: Vec<u32> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(run_dir) {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_file() {
                    continue;
                }
                let Some(name) = p.file_name().and_then(|x| x.to_str()) else {
                    continue;
                };
                // replay_snapshot_iter_021.json
                if !name.starts_with("replay_snapshot_iter_") || !name.ends_with(".json") {
                    continue;
                }
                let mid = name
                    .trim_end_matches(".json")
                    .trim_start_matches("replay_snapshot_iter_");
                if let Ok(i) = mid.parse::<u32>() {
                    snapshot_iters.push(i);
                }
            }
        }
        snapshot_iters.sort_unstable();
        let max_snapshot = snapshot_iters.iter().copied().max();
        self.replay_max_snapshot_iter = max_snapshot;

        // 2) Incrementally scan new snapshots to build shard -> origin iteration mapping.
        if let Some(max_i) = max_snapshot {
            let start = (self.replay_scanned_up_to_iter + 1).max(0) as u32;
            for i in start..=max_i {
                if !snapshot_iters.binary_search(&i).is_ok() {
                    continue;
                }
                let f = run_dir.join(format!("replay_snapshot_iter_{i:03}.json"));
                let bytes = match std::fs::read(&f) {
                    Ok(b) => b,
                    Err(_) => continue,
                };
                let snap = match serde_json::from_slice::<ReplaySnapshotV1>(&bytes) {
                    Ok(s) => s,
                    Err(_) => continue,
                };
                for sh in snap.shards {
                    if let Some(id) = Self::parse_shard_id(&sh.safetensors) {
                        self.replay_origin_by_shard.entry(id).or_insert(i);
                    }
                }
                self.replay_scanned_up_to_iter = i as i32;
            }
        }

        // 3) Current buffer shards are the meta files in replay/ (capacity_shards count).
        let replay_dir = run_dir.join("replay");
        #[derive(Debug, serde::Deserialize)]
        struct ShardMeta {
            #[serde(default)]
            num_samples: u64,
        }

        let mut shards: Vec<u32> = Vec::new();
        let mut samples_by_shard: HashMap<u32, u64> = HashMap::new();
        if let Ok(rd) = std::fs::read_dir(&replay_dir) {
            for e in rd.flatten() {
                let p = e.path();
                if !p.is_file() {
                    continue;
                }
                let Some(name) = p.file_name().and_then(|x| x.to_str()) else {
                    continue;
                };
                if !name.starts_with("shard_") || !name.ends_with(".meta.json") {
                    continue;
                }
                if let Some(id) = Self::parse_shard_id(name) {
                    shards.push(id);
                    if let Ok(bytes) = std::fs::read(&p) {
                        if let Ok(m) = serde_json::from_slice::<ShardMeta>(&bytes) {
                            samples_by_shard.insert(id, m.num_samples);
                        }
                    }
                }
            }
        } else {
            self.replay_err = Some("replay/ not found".to_string());
            self.replay_current.clear();
            self.replay_total_samples_current = None;
            return;
        }
        shards.sort_unstable();
        let mut total = 0u64;
        self.replay_current = shards
            .into_iter()
            .map(|id| {
                let ns = samples_by_shard.get(&id).copied();
                if let Some(v) = ns {
                    total = total.saturating_add(v);
                }
                ReplayShardView {
                    shard_id: id,
                    origin_iter: self.replay_origin_by_shard.get(&id).copied(),
                    num_samples: ns,
                }
            })
            .collect();
        self.replay_total_samples_current = Some(total);
        self.replay_err = None;
    }

    fn refresh_learning_metrics(&mut self, run_dir: &Path) {
        let rid = self.active_run_id.clone().unwrap_or_default();
        if self.learn_loaded_for_run.as_deref() != Some(rid.as_str()) {
            self.learn_loaded_for_run = Some(rid);
            self.learn_metrics_offset = 0;
            self.learn_metrics_partial.clear();
            self.learn_by_iter.clear();
            self.selfplay_by_iter.clear();
            self.infer_snapshots.clear();
            self.infer_last_ts_ms = 0;

            // Reset replay buffer cache when switching runs.
            self.replay_loaded_for_run = None;
            self.replay_scanned_up_to_iter = -1;
            self.replay_max_snapshot_iter = None;
            self.replay_origin_by_shard.clear();
            self.replay_current.clear();
            self.replay_total_samples_current = None;
            self.replay_err = None;

            // Reset search selection when switching runs.
            self.search_follow = true;
            self.search_selected_iter = 0;

            // Reset system live deltas/series when switching runs.
            self.system_live = None;
            self.system_live_err = None;
            self.system_series_queue_depth.clear();
            self.system_series_rtt_p95_us.clear();
            self.system_series_rps_x10.clear();
            self.system_series_batch_mean_x100.clear();
            self.system_series_would_block_x1000.clear();
            self.system_prev_ts = None;
            self.system_prev_req_total = None;
            self.system_prev_batches_total = None;
            self.system_prev_flush_full_total = None;
            self.system_prev_flush_deadline_total = None;
            self.system_prev_steps_sum = None;
            self.system_prev_wb_sum = None;
        }

        let metrics_path = run_dir.join("logs").join("metrics.ndjson");
        if !metrics_path.exists() {
            return;
        }
        let mut f = match File::open(&metrics_path) {
            Ok(x) => x,
            Err(_) => return,
        };
        if f.seek(SeekFrom::Start(self.learn_metrics_offset)).is_err() {
            // If the file was truncated/rotated, restart from 0.
            self.learn_metrics_offset = 0;
            self.learn_metrics_partial.clear();
            if f.seek(SeekFrom::Start(0)).is_err() {
                return;
            }
        }
        let mut buf = String::new();
        if f.read_to_string(&mut buf).is_err() {
            return;
        }
        self.learn_metrics_offset = self.learn_metrics_offset.saturating_add(buf.len() as u64);

        // Prepend any partial line from the previous refresh.
        let mut data = std::mem::take(&mut self.learn_metrics_partial);
        data.push_str(&buf);

        // Keep the last partial line (no trailing newline) for next time.
        let mut lines = data.split('\n').peekable();
        while let Some(line) = lines.next() {
            if lines.peek().is_none() && !data.ends_with('\n') {
                self.learn_metrics_partial = line.to_string();
                break;
            }
            let s = line.trim();
            if s.is_empty() {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(s) else {
                continue;
            };
            let ev = v.get("event").and_then(|x| x.as_str()).unwrap_or("");
            match ev {
                "learn_summary" => {
                    let Ok(ls) = serde_json::from_value::<LearnSummaryNdjsonV1>(v) else {
                        continue;
                    };
                    let Some(i) = ls.iter_idx else {
                        continue;
                    };
                    self.learn_by_iter.insert(i, ls);
                }
                "selfplay_summary" => {
                    let Ok(ss) = serde_json::from_value::<SelfplaySummaryNdjsonV1>(v) else {
                        continue;
                    };
                    let Some(i) = ss.iter_idx else {
                        continue;
                    };
                    self.selfplay_by_iter.insert(i, ss);
                }
                "infer_snapshot" => {
                    let Ok(ss) = serde_json::from_value::<InferSnapshotNdjsonV1>(v) else {
                        continue;
                    };
                    let Some(ts) = ss.ts_ms else {
                        continue;
                    };
                    // Avoid duplicates if we ever re-read from earlier offsets.
                    if ts <= self.infer_last_ts_ms {
                        continue;
                    }
                    self.infer_last_ts_ms = ts;
                    self.infer_snapshots.push(ss);
                    const MAX: usize = 6000; // ~8.3h at 5s cadence
                    if self.infer_snapshots.len() > MAX {
                        let drain = self.infer_snapshots.len() - MAX;
                        self.infer_snapshots.drain(0..drain);
                    }
                }
                _ => {}
            }
        }
    }

    fn enter_dashboard(&mut self) {
        if self.active_run_id.is_none() {
            return;
        }
        self.refresh_dashboard();
        self.screen = Screen::Dashboard;
        // Default behavior: follow the current iteration so the latest rows are visible.
        self.dashboard_iter_follow = true;
        if self.shutdown_requested {
            self.status = "cancelling… waiting for shutdown | r refresh".to_string();
        } else {
            self.status =
                "r refresh | l toggle learning/perf | d perf | s search | i system | e extend | x cancel | ↑/↓ scroll | PgUp/PgDn | Home/End | Esc back | q quit"
                    .to_string();
        }
    }

    fn enter_search(&mut self) {
        if self.active_run_id.is_none() {
            return;
        }
        self.refresh_dashboard();
        self.screen = Screen::Search;
        self.search_follow = true;
        if let Some(m) = &self.dashboard_manifest {
            self.search_selected_iter = m.controller_iteration_idx;
        } else if let Some(k) = self.selfplay_by_iter.keys().max().copied() {
            self.search_selected_iter = k;
        } else {
            self.search_selected_iter = 0;
        }
        self.status =
            "r refresh | d perf | l learning | i system | s search | ↑/↓ select | PgUp/PgDn | Home/End | Esc back | q quit"
                .to_string();
    }

    fn enter_system(&mut self) {
        if self.active_run_id.is_none() {
            return;
        }
        // Refresh once so we have run.json + any infer_snapshot playback immediately.
        self.refresh_dashboard();
        self.screen = Screen::System;
        self.status =
            "r refresh | d perf | l learning | s search | i system | Esc back | q quit".to_string();
        // Force immediate live poll on next tick.
        self.system_last_poll = Instant::now()
            .checked_sub(Duration::from_secs(10))
            .unwrap_or_else(Instant::now);
    }

    fn refresh_system_live(&mut self) {
        // Only poll when the system screen is active.
        if !matches!(self.screen, Screen::System) {
            return;
        }
        if self.system_last_poll.elapsed() < Duration::from_secs(1) {
            return;
        }
        self.system_last_poll = Instant::now();

        let Some(run_dir) = self.run_dir() else {
            return;
        };
        // Keep run.json + metrics playback reasonably fresh while on this screen.
        self.refresh_dashboard();

        let metrics_bind = self.cfg.inference.metrics_bind.clone();
        match fetch_infer_metrics_text(&self.system_agent, &metrics_bind) {
            Ok(text) => {
                let server = parse_infer_server_metrics_text(&text);
                let workers = read_selfplay_worker_progress_live(
                    &run_dir.join("logs_workers"),
                    &mut self.system_prev_steps_sum,
                    &mut self.system_prev_wb_sum,
                );
                let now = Instant::now();

                let (requests_s, batches_s, batch_size_mean, flush_full_s, flush_deadline_s) =
                    derive_rates(
                        now,
                        &mut self.system_prev_ts,
                        &mut self.system_prev_req_total,
                        &mut self.system_prev_batches_total,
                        &mut self.system_prev_flush_full_total,
                        &mut self.system_prev_flush_deadline_total,
                        server.requests_total,
                        server.batches_total,
                        server.flush_full_total,
                        server.flush_deadline_total,
                    );

                let live = SystemLiveSample {
                    ts: now,
                    queue_depth: server.queue_depth,
                    requests_s,
                    batches_s,
                    batch_size_mean,
                    batch_size_p95: server.batch_size_p95,
                    underfill_frac_mean: server.underfill_frac_mean,
                    full_frac_mean: server.full_frac_mean,
                    flush_full_s,
                    flush_deadline_s,
                    queue_wait_us_p95: server.queue_wait_us_p95,
                    forward_ms_p95: server.forward_ms_p95,
                    inflight_sum: workers.inflight_sum,
                    inflight_max: workers.inflight_max,
                    rtt_p95_us_med: workers.rtt_p95_us_med,
                    would_block_frac: workers.would_block_frac,
                    source: "live",
                };
                self.system_live = Some(live);
                self.system_live_err = None;
                self.push_system_series_from_current();
            }
            Err(e) => {
                self.system_live_err = Some(e);
                // Playback fallback: use last infer_snapshot (every ~5s) if available.
                if let Some(last) = self.infer_snapshots.last() {
                    let now = Instant::now();
                    self.system_live = Some(SystemLiveSample {
                        ts: now,
                        queue_depth: last.queue_depth,
                        requests_s: last.requests_s,
                        batches_s: last.batches_s,
                        batch_size_mean: last.batch_size_mean,
                        batch_size_p95: last.batch_size_p95,
                        underfill_frac_mean: last.underfill_frac_mean,
                        full_frac_mean: last.full_frac_mean,
                        flush_full_s: last.flush_full_s,
                        flush_deadline_s: last.flush_deadline_s,
                        queue_wait_us_p95: last.queue_wait_us_p95,
                        forward_ms_p95: last.forward_ms_p95,
                        inflight_sum: last.inflight_sum,
                        inflight_max: last.inflight_max,
                        rtt_p95_us_med: last.rtt_p95_us_med,
                        would_block_frac: last.would_block_frac,
                        source: "playback",
                    });
                }
            }
        }
    }

    fn push_system_series_from_current(&mut self) {
        const MAX: usize = 120; // ~2 minutes at 1Hz
        let Some(cur) = self.system_live.as_ref() else {
            return;
        };
        push_series(&mut self.system_series_queue_depth, cur.queue_depth.unwrap_or(0), MAX);
        push_series(
            &mut self.system_series_rtt_p95_us,
            cur.rtt_p95_us_med.unwrap_or(0),
            MAX,
        );
        push_series(
            &mut self.system_series_rps_x10,
            (cur.requests_s.unwrap_or(0.0).max(0.0) * 10.0) as u64,
            MAX,
        );
        push_series(
            &mut self.system_series_batch_mean_x100,
            (cur.batch_size_mean.unwrap_or(0.0).max(0.0) * 100.0) as u64,
            MAX,
        );
        push_series(
            &mut self.system_series_would_block_x1000,
            (cur.would_block_frac.unwrap_or(0.0).clamp(0.0, 1.0) * 1000.0) as u64,
            MAX,
        );
    }

    fn start_iteration(&mut self) {
        if self.iter.is_some() {
            self.status = "already running".to_string();
            return;
        }
        let Some(run_dir) = self.run_dir() else {
            self.status = "No run selected".to_string();
            return;
        };

        // Validate config.
        if let Err(e) = crate::validate::validate_config(&self.cfg) {
            self.form.last_validation_error = Some(e.clone());
            self.status = format!("cannot start (invalid): {e}");
            return;
        }

        // Best-effort save draft before starting.
        self.save_config_draft();

        // Spawn controller in background.
        let infer_endpoint = self.cfg.inference.bind.clone();
        let python_exe = "python".to_string();
        let handle =
            yz_controller::spawn_iteration(run_dir, self.cfg.clone(), infer_endpoint, python_exe);
        self.iter = Some(handle);
        self.enter_dashboard();
    }

    fn cancel_iteration_hard(&mut self) {
        if let Some(h) = &self.iter {
            h.cancel_hard();
            self.shutdown_requested = true;
            self.shutdown_exit_after = false;
            self.status = "cancelling… waiting for shutdown".to_string();
        } else {
            // If the run was started outside the TUI (e.g. `yz start-run`), request cancel via
            // run-local cancel.request so the controller can observe it.
            let Some(run_dir) = self.run_dir() else {
                self.status = "no active run".to_string();
                return;
            };
            let run_json = run_dir.join("run.json");
            if !run_json.exists() {
                self.status = "no active run".to_string();
                return;
            }
            match yz_logging::read_manifest(&run_json) {
                Ok(m) => {
                    let phase = m.controller_phase.as_deref().unwrap_or("");
                    let running = matches!(phase, "selfplay" | "train" | "gate");
                    if running {
                        let tmp = run_dir.join("cancel.request.tmp");
                        let p = run_dir.join("cancel.request");
                        if std::fs::write(&tmp, format!("ts_ms: {}\n", yz_logging::now_ms()))
                            .and_then(|_| std::fs::rename(&tmp, &p))
                            .is_ok()
                        {
                            self.shutdown_requested = true;
                            self.shutdown_exit_after = false;
                            self.enter_dashboard();
                            self.status = "cancel requested… (waiting for controller)".to_string();
                        } else {
                            self.status = "cancel request failed".to_string();
                        }
                    } else {
                        self.status = "no active run".to_string();
                    }
                }
                Err(_) => {
                    self.status = "no active run".to_string();
                }
            }
        }
    }

    fn request_quit_or_cancel(&mut self) -> bool {
        // Returns true if the UI should exit immediately.
        if self.iter.is_some() {
            // Always route to dashboard while shutting down so the user can see status.
            self.enter_dashboard();
            if let Some(h) = &self.iter {
                h.cancel_hard();
            }
            self.shutdown_requested = true;
            self.shutdown_exit_after = true;
            self.status = "cancelling… waiting for shutdown".to_string();
            false
        } else {
            // If the run is active but was started outside the TUI, request cancel and exit.
            let Some(run_dir) = self.run_dir() else {
                return true;
            };
            let run_json = run_dir.join("run.json");
            if !run_json.exists() {
                return true;
            }
            if let Ok(m) = yz_logging::read_manifest(&run_json) {
                let phase = m.controller_phase.as_deref().unwrap_or("");
                let running = matches!(phase, "selfplay" | "train" | "gate");
                if running {
                    let tmp = run_dir.join("cancel.request.tmp");
                    let p = run_dir.join("cancel.request");
                    let _ = std::fs::write(&tmp, format!("ts_ms: {}\n", yz_logging::now_ms()))
                        .and_then(|_| std::fs::rename(&tmp, &p));
                }
            }
            true
        }
    }
}

fn config_help(form: &FormState) -> String {
    match form.edit_mode {
        EditMode::View => {
            "↑/↓ select | Enter edit | ←/→ step | Shift+←/→ big step | Space toggle | Tab section | s save | g start | d dashboard | Esc back | q quit"
                .to_string()
        }
        EditMode::Editing => "type | Backspace | Enter commit | Esc cancel".to_string(),
    }
}

fn list_runs(runs_dir: &Path) -> Vec<String> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(runs_dir) else {
        return out;
    };
    for ent in rd.flatten() {
        let Ok(ft) = ent.file_type() else { continue };
        if !ft.is_dir() {
            continue;
        }
        let name = ent.file_name().to_string_lossy().to_string();
        if name.starts_with('.') {
            continue;
        }
        out.push(name);
    }
    out.sort();
    out
}

fn copy_dir_recursive(src: &Path, dst: &Path) -> io::Result<()> {
    if !src.exists() {
        return Ok(());
    }
    std::fs::create_dir_all(dst)?;
    for ent in std::fs::read_dir(src)? {
        let ent = ent?;
        let ft = ent.file_type()?;
        let name = ent.file_name();
        let src_p = ent.path();
        let dst_p = dst.join(name);
        if ft.is_dir() {
            copy_dir_recursive(&src_p, &dst_p)?;
        } else if ft.is_file() {
            // Best-effort overwrite (destination is a fresh run dir anyway).
            let _ = std::fs::copy(&src_p, &dst_p)?;
        }
    }
    Ok(())
}

pub fn run() -> io::Result<()> {
    // Terminal init.
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    stdout.execute(EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;
    terminal.clear()?;

    let mut app = App::new(PathBuf::from("runs"));
    app.refresh_runs();

    let tick_rate = Duration::from_millis(100);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| draw(f, &app))?;

        let timeout = tick_rate.saturating_sub(last_tick.elapsed());
        if event::poll(timeout)? {
            if let Event::Key(k) = event::read()? {
                if k.kind != KeyEventKind::Press {
                    continue;
                }
                match app.screen {
                    Screen::Home => match k.code {
                        KeyCode::Char('q') => {
                            if app.request_quit_or_cancel() {
                                break;
                            }
                        }
                        KeyCode::Char('r') => app.refresh_runs(),
                        KeyCode::Char('n') => {
                            app.naming_input.clear();
                            app.screen = Screen::NamingRun;
                            app.status = "Enter run name (Enter to confirm, Esc to cancel)".to_string();
                        }
                        KeyCode::Char('e') => app.begin_extend_selected_run(Screen::Home),
                        KeyCode::Enter => app.enter_selected_run(),
                        KeyCode::Up => {
                            if app.selected > 0 {
                                app.selected -= 1;
                            }
                        }
                        KeyCode::Down => {
                            if app.selected + 1 < app.runs.len() {
                                app.selected += 1;
                            }
                        }
                        KeyCode::PageUp => {
                            let step = 10usize;
                            app.selected = app.selected.saturating_sub(step);
                        }
                        KeyCode::PageDown => {
                            let step = 10usize;
                            if !app.runs.is_empty() {
                                app.selected = (app.selected + step).min(app.runs.len() - 1);
                            }
                        }
                        KeyCode::Home => {
                            app.selected = 0;
                        }
                        KeyCode::End => {
                            if !app.runs.is_empty() {
                                app.selected = app.runs.len() - 1;
                            }
                        }
                        _ => {}
                    },
                    Screen::NamingRun => match k.code {
                        KeyCode::Esc => {
                            app.screen = Screen::Home;
                            app.naming_input.clear();
                            app.refresh_runs();
                        }
                        KeyCode::Enter => {
                            let name = app.naming_input.clone();
                            app.screen = Screen::Home;
                            if let Err(e) = app.create_run_with_name(&name) {
                                app.status = format!("create failed: {e}");
                            }
                            app.naming_input.clear();
                        }
                        KeyCode::Backspace => {
                            app.naming_input.pop();
                        }
                        KeyCode::Char(c) => {
                            if !c.is_control() {
                                app.naming_input.push(c);
                            }
                        }
                        _ => {}
                    },
                    Screen::ExtendingRun => match k.code {
                        KeyCode::Esc => {
                            app.screen = app.extend_return_to;
                            app.extend_source_run_id = None;
                            app.extend_input.clear();
                            if matches!(app.screen, Screen::Home) {
                                app.refresh_runs();
                            } else if matches!(app.screen, Screen::Dashboard) {
                                app.enter_dashboard();
                            } else if matches!(app.screen, Screen::Config) {
                                app.status = config_help(&app.form);
                            }
                        }
                        KeyCode::Enter => {
                            let name = app.extend_input.clone();
                            match app.extend_run_with_name(&name) {
                                Ok(new_id) => {
                                    app.extend_source_run_id = None;
                                    app.extend_input.clear();
                                    app.active_run_id = Some(new_id.clone());
                                    if let Some(run_dir) = app.run_dir() {
                                        let (cfg, msg) = crate::config_io::load_cfg_for_run(&run_dir);
                                        app.cfg = cfg;
                                        app.form = FormState::default();
                                        if let Some(msg) = msg {
                                            app.status = msg;
                                        } else {
                                            app.status = "Extended run created".to_string();
                                        }
                                    }
                                    app.screen = Screen::Config;
                                    app.status = config_help(&app.form);
                                }
                                Err(e) => {
                                    app.status = format!("extend failed: {e}");
                                }
                            }
                        }
                        KeyCode::Backspace => {
                            app.extend_input.pop();
                        }
                        KeyCode::Char(' ') => {
                            app.extend_include_replay = !app.extend_include_replay;
                        }
                        KeyCode::Char(c) => {
                            if !c.is_control() {
                                app.extend_input.push(c);
                            }
                        }
                        _ => {}
                    },
                    Screen::Config => match k.code {
                        KeyCode::Char('q') => {
                            if app.request_quit_or_cancel() {
                                break;
                            }
                        }
                        KeyCode::Esc => {
                            app.screen = Screen::Home;
                            app.active_run_id = None;
                            app.refresh_runs();
                        }
                        KeyCode::Char('s') => app.save_config_draft(),
                        KeyCode::Char('d') => app.enter_dashboard(),
                        KeyCode::Char('i') => app.enter_system(),
                        KeyCode::Char('g') => app.start_iteration(),
                        _ => handle_config_key(&mut app, k),
                    },
                    Screen::Dashboard => match k.code {
                        KeyCode::Char('q') => {
                            if app.request_quit_or_cancel() {
                                break;
                            }
                        }
                        KeyCode::Esc => {
                            app.screen = Screen::Config;
                            app.status = config_help(&app.form);
                        }
                        KeyCode::Char('e') => app.begin_extend_active_run(Screen::Dashboard),
                        KeyCode::Char('l') => {
                            app.dashboard_tab = match app.dashboard_tab {
                                DashboardTab::Performance => DashboardTab::Learning,
                                DashboardTab::Learning => DashboardTab::Performance,
                            };
                            app.enter_dashboard();
                        }
                        // Convenience: always jump back to the Performance dashboard.
                        KeyCode::Char('d') => {
                            app.dashboard_tab = DashboardTab::Performance;
                            app.enter_dashboard();
                        }
                        KeyCode::Char('s') => app.enter_search(),
                        KeyCode::Char('i') => app.enter_system(),
                        KeyCode::Char('r') => app.refresh_dashboard(),
                        KeyCode::Char('x') => app.cancel_iteration_hard(),
                        KeyCode::Up => {
                            app.dashboard_iter_follow = false;
                            app.dashboard_iter_scroll = app.dashboard_iter_scroll.saturating_sub(1);
                        }
                        KeyCode::Down => {
                            app.dashboard_iter_follow = false;
                            app.dashboard_iter_scroll = app.dashboard_iter_scroll.saturating_add(1);
                        }
                        KeyCode::PageUp => {
                            app.dashboard_iter_follow = false;
                            app.dashboard_iter_scroll = app.dashboard_iter_scroll.saturating_sub(10);
                        }
                        KeyCode::PageDown => {
                            app.dashboard_iter_follow = false;
                            app.dashboard_iter_scroll = app.dashboard_iter_scroll.saturating_add(10);
                        }
                        KeyCode::Home => {
                            app.dashboard_iter_follow = false;
                            app.dashboard_iter_scroll = 0;
                        }
                        KeyCode::End => {
                            app.dashboard_iter_follow = true;
                        }
                        _ => {}
                    },
                    Screen::Search => match k.code {
                        KeyCode::Char('q') => {
                            if app.request_quit_or_cancel() {
                                break;
                            }
                        }
                        KeyCode::Esc => {
                            // Back to dashboard (keep the current tab).
                            app.enter_dashboard();
                        }
                        KeyCode::Char('d') => {
                            app.dashboard_tab = DashboardTab::Performance;
                            app.enter_dashboard();
                        }
                        KeyCode::Char('l') => {
                            app.dashboard_tab = DashboardTab::Learning;
                            app.enter_dashboard();
                        }
                        KeyCode::Char('i') => app.enter_system(),
                        KeyCode::Char('s') => app.enter_search(),
                        KeyCode::Char('r') => app.refresh_dashboard(),
                        KeyCode::Char('f') => {
                            app.search_follow = true;
                            if let Some(m) = &app.dashboard_manifest {
                                app.search_selected_iter = m.controller_iteration_idx;
                            }
                        }
                        KeyCode::Up => {
                            app.search_follow = false;
                            app.search_selected_iter = app.search_selected_iter.saturating_sub(1);
                        }
                        KeyCode::Down => {
                            app.search_follow = false;
                            app.search_selected_iter = app.search_selected_iter.saturating_add(1);
                        }
                        KeyCode::PageUp => {
                            app.search_follow = false;
                            app.search_selected_iter = app.search_selected_iter.saturating_sub(10);
                        }
                        KeyCode::PageDown => {
                            app.search_follow = false;
                            app.search_selected_iter = app.search_selected_iter.saturating_add(10);
                        }
                        KeyCode::Home => {
                            app.search_follow = false;
                            app.search_selected_iter = 0;
                        }
                        KeyCode::End => {
                            app.search_follow = true;
                            if let Some(m) = &app.dashboard_manifest {
                                app.search_selected_iter = m.controller_iteration_idx;
                            }
                        }
                        _ => {}
                    },
                    Screen::System => match k.code {
                        KeyCode::Char('q') => {
                            if app.request_quit_or_cancel() {
                                break;
                            }
                        }
                        KeyCode::Esc => {
                            // Back to dashboard (keep the current tab).
                            app.enter_dashboard();
                        }
                        KeyCode::Char('d') => {
                            app.dashboard_tab = DashboardTab::Performance;
                            app.enter_dashboard();
                        }
                        KeyCode::Char('l') => {
                            app.dashboard_tab = DashboardTab::Learning;
                            app.enter_dashboard();
                        }
                        KeyCode::Char('s') => app.enter_search(),
                        KeyCode::Char('r') => {
                            app.refresh_dashboard();
                            app.system_last_poll = Instant::now()
                                .checked_sub(Duration::from_secs(10))
                                .unwrap_or_else(Instant::now);
                            app.refresh_system_live();
                        }
                        _ => {}
                    },
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
            // If controller finished, join and clear.
            if let Some(h) = &app.iter {
                if h.is_finished() {
                    // Take ownership and join.
                    let h = app.iter.take().unwrap();
                    match h.join() {
                        Ok(()) => {
                            app.status =
                                "Completed | r refresh | g start | Esc back | q quit".to_string();
                        }
                        Err(e) => {
                            let msg = if matches!(e, yz_controller::ControllerError::Cancelled) {
                                "Cancelled"
                            } else {
                                "Failed"
                            };
                            // Surface a short error so failures are diagnosable without digging in logs.
                            let mut detail = e.to_string();
                            const MAX: usize = 180;
                            if detail.len() > MAX {
                                detail.truncate(MAX);
                                detail.push_str("…");
                            }
                            app.status = format!(
                                "{msg}: {detail} | r refresh | g start | Esc back | q quit"
                            );
                        }
                    }
                    app.refresh_dashboard();
                    if app.shutdown_exit_after {
                        break;
                    }
                    app.shutdown_requested = false;
                    app.shutdown_exit_after = false;
                }
            }
            if matches!(app.screen, Screen::Dashboard | Screen::Search) {
                app.refresh_dashboard();
                if app.shutdown_requested {
                    app.status = "cancelling… waiting for shutdown | r refresh".to_string();
                }
            }
            if matches!(app.screen, Screen::System) {
                app.refresh_system_live();
            }
            last_tick = Instant::now();
        }
    }

    // Terminal restore.
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    Ok(())
}

fn handle_config_key(app: &mut App, k: crossterm::event::KeyEvent) {
    // Keep status help fresh as mode changes.
    app.status = config_help(&app.form);

    fn is_field_visible(cfg: &yz_core::Config, f: FieldId) -> bool {
        // Keep in sync with `render_config_lines`.
        if matches!(f, FieldId::MctsTempT1 | FieldId::MctsTempCutoffTurn)
            && matches!(cfg.mcts.temperature_schedule, TemperatureSchedule::Constant { .. })
        {
            return false;
        }
        if f == FieldId::TrainingEpochs && cfg.training.steps_per_iteration.is_some() {
            return false;
        }
        if f == FieldId::TrainingStepsPerIteration && cfg.training.steps_per_iteration.is_none() {
            return false;
        }
        if f == FieldId::TrainingResetOptimizer && cfg.training.continuous_candidate_training {
            return false;
        }
        if f == FieldId::GatingGames && cfg.gating.katago.sprt {
            return false;
        }
        if matches!(
            f,
            FieldId::GatingKatagoSprtMinGames
                | FieldId::GatingKatagoSprtMaxGames
                | FieldId::GatingKatagoSprtAlpha
                | FieldId::GatingKatagoSprtBeta
                | FieldId::GatingKatagoSprtDelta
        ) && !cfg.gating.katago.sprt
        {
            return false;
        }
        true
    }

    fn ensure_selected_visible(app: &mut App) {
        if ALL_FIELDS.is_empty() {
            app.form.selected_idx = 0;
            return;
        }
        let mut sel = app.form.selected_idx.min(ALL_FIELDS.len().saturating_sub(1));
        if is_field_visible(&app.cfg, ALL_FIELDS[sel]) {
            app.form.selected_idx = sel;
            return;
        }
        // Prefer scanning forward so navigation feels natural.
        for i in sel..ALL_FIELDS.len() {
            if is_field_visible(&app.cfg, ALL_FIELDS[i]) {
                app.form.selected_idx = i;
                return;
            }
        }
        // Fallback: scan backward.
        while sel > 0 {
            sel -= 1;
            if is_field_visible(&app.cfg, ALL_FIELDS[sel]) {
                app.form.selected_idx = sel;
                return;
            }
        }
        app.form.selected_idx = 0;
    }

    fn step_selection(app: &mut App, dir: i32) {
        if ALL_FIELDS.is_empty() {
            app.form.selected_idx = 0;
            return;
        }
        let mut i = app.form.selected_idx as i32;
        loop {
            i += dir.signum();
            if i < 0 || i >= ALL_FIELDS.len() as i32 {
                break;
            }
            let idx = i as usize;
            if is_field_visible(&app.cfg, ALL_FIELDS[idx]) {
                app.form.selected_idx = idx;
                break;
            }
        }
    }

    ensure_selected_visible(app);
    let sel = app
        .form
        .selected_idx
        .min(ALL_FIELDS.len().saturating_sub(1));
    app.form.selected_idx = sel;
    let field = ALL_FIELDS[sel];

    match app.form.edit_mode {
        EditMode::Editing => match k.code {
            KeyCode::Esc => {
                app.form.edit_mode = EditMode::View;
                app.form.input_buf.clear();
            }
            KeyCode::Enter => commit_field_edit(app, field),
            KeyCode::Backspace => {
                app.form.input_buf.pop();
            }
            KeyCode::Char(' ') => toggle_or_cycle(app, field),
            KeyCode::Char(c) => {
                // Allow typing for all fields; per-field parsing/validation happens on commit.
                if !c.is_control() {
                    app.form.input_buf.push(c);
                }
            }
            KeyCode::Left => {
                // Allow stepping while editing by adjusting buffer to new value.
                step_field(app, field, -1, StepSize::from_mods(k.modifiers));
                app.form.input_buf = field_value_string(&app.cfg, field);
            }
            KeyCode::Right => {
                step_field(app, field, 1, StepSize::from_mods(k.modifiers));
                app.form.input_buf = field_value_string(&app.cfg, field);
            }
            KeyCode::Tab => jump_section(app, 1),
            KeyCode::BackTab => jump_section(app, -1),
            _ => {}
        },
        EditMode::View => match k.code {
            KeyCode::Up => {
                step_selection(app, -1);
            }
            KeyCode::Down => {
                step_selection(app, 1);
            }
            KeyCode::Tab => jump_section(app, 1),
            KeyCode::BackTab => jump_section(app, -1),
            KeyCode::Enter => {
                app.form.edit_mode = EditMode::Editing;
                app.form.input_buf = field_value_string(&app.cfg, field);
            }
            KeyCode::Left => step_field(app, field, -1, StepSize::from_mods(k.modifiers)),
            KeyCode::Right => step_field(app, field, 1, StepSize::from_mods(k.modifiers)),
            KeyCode::Char(' ') => toggle_or_cycle(app, field),
            _ => {}
        },
    }

    // Config changes can hide/show fields; never leave the cursor on a hidden field.
    ensure_selected_visible(app);
    app.status = config_help(&app.form);
}

fn jump_section(app: &mut App, dir: i32) {
    let sel = app
        .form
        .selected_idx
        .min(ALL_FIELDS.len().saturating_sub(1));
    let cur = ALL_FIELDS[sel].section();
    let cur_idx = Section::ALL.iter().position(|s| *s == cur).unwrap_or(0) as i32;
    let next_idx = (cur_idx + dir).rem_euclid(Section::ALL.len() as i32) as usize;
    let next = Section::ALL[next_idx];
    // Jump to the first *visible* field in that section.
    if let Some(pos) = ALL_FIELDS.iter().position(|f| f.section() == next) {
        app.form.selected_idx = pos;
        // If the first field is hidden under current config, scan forward within the section.
        for i in pos..ALL_FIELDS.len() {
            let f = ALL_FIELDS[i];
            if f.section() != next {
                break;
            }
            // Reuse the same visibility logic as the config renderer.
            let visible = !((matches!(f, FieldId::MctsTempT1 | FieldId::MctsTempCutoffTurn)
                && matches!(app.cfg.mcts.temperature_schedule, TemperatureSchedule::Constant { .. }))
                || (f == FieldId::TrainingEpochs && app.cfg.training.steps_per_iteration.is_some())
                || (f == FieldId::TrainingStepsPerIteration
                    && app.cfg.training.steps_per_iteration.is_none()));
            if visible {
                app.form.selected_idx = i;
                break;
            }
        }
    }
}

fn toggle_or_cycle(app: &mut App, field: FieldId) {
    fn list_seed_sets() -> Vec<String> {
        use std::path::PathBuf;
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../configs/seed_sets");
        let mut out: Vec<String> = Vec::new();
        if let Ok(rd) = std::fs::read_dir(&dir) {
            for e in rd.flatten() {
                let p = e.path();
                if p.extension().and_then(|s| s.to_str()) != Some("txt") {
                    continue;
                }
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    out.push(stem.to_string());
                }
            }
        }
        out.sort();
        out
    }

    match field {
        FieldId::GatingSeedSetId => {
            // Space-cycle over available seed set ids: None -> first -> ... -> None.
            let sets = list_seed_sets();
            if sets.is_empty() {
                // Fallback: toggle between None and dev_v1.
                app.cfg.gating.seed_set_id = match app.cfg.gating.seed_set_id.as_deref() {
                    Some(s) if !s.is_empty() => None,
                    _ => Some("dev_v1".to_string()),
                };
            } else {
                let cur = app.cfg.gating.seed_set_id.clone().unwrap_or_default();
                let idx = sets.iter().position(|s| *s == cur);
                app.cfg.gating.seed_set_id = match idx {
                    None => Some(sets[0].clone()),
                    Some(i) if i + 1 < sets.len() => Some(sets[i + 1].clone()),
                    _ => None,
                };
            }
        }
        FieldId::GatingPairedSeedSwap => {
            app.cfg.gating.paired_seed_swap = !app.cfg.gating.paired_seed_swap;
        }
        FieldId::GatingDeterministicChance => {
            app.cfg.gating.deterministic_chance = !app.cfg.gating.deterministic_chance;
        }
        FieldId::GatingKatagoSprt => {
            app.cfg.gating.katago.sprt = !app.cfg.gating.katago.sprt;
        }
        FieldId::InferDevice => {
            // Cycle: cpu → mps → cuda → cpu
            app.cfg.inference.device = match app.cfg.inference.device.as_str() {
                "cpu" => "mps".to_string(),
                "mps" => "cuda".to_string(),
                _ => "cpu".to_string(),
            };
        }
        FieldId::InferProtocolVersion => {
            // Toggle between v1 and v2.
            app.cfg.inference.protocol_version = if app.cfg.inference.protocol_version == 2 {
                1
            } else {
                2
            };
        }
        FieldId::InferLegalMaskBitset => {
            app.cfg.inference.legal_mask_bitset = !app.cfg.inference.legal_mask_bitset;
        }
        FieldId::InferDebugLog => {
            app.cfg.inference.debug_log = !app.cfg.inference.debug_log;
        }
        FieldId::InferPrintStats => {
            app.cfg.inference.print_stats = !app.cfg.inference.print_stats;
        }
        FieldId::MctsTempKind => {
            let t0 = match app.cfg.mcts.temperature_schedule {
                TemperatureSchedule::Constant { t0 } => t0,
                TemperatureSchedule::Step { t0, .. } => t0,
            };
            app.cfg.mcts.temperature_schedule = match app.cfg.mcts.temperature_schedule {
                TemperatureSchedule::Constant { .. } => TemperatureSchedule::Step {
                    t0,
                    t1: 0.1,
                    cutoff_turn: 10,
                },
                TemperatureSchedule::Step { .. } => TemperatureSchedule::Constant { t0 },
            };
        }
        FieldId::MctsVirtualLossMode => {
            // Cycle: q_penalty → n_virtual_only → off → q_penalty
            app.cfg.mcts.virtual_loss_mode = match app.cfg.mcts.virtual_loss_mode.as_str() {
                "q_penalty" => "n_virtual_only".to_string(),
                "n_virtual_only" => "off".to_string(),
                _ => "q_penalty".to_string(),
            };
        }
        FieldId::MctsKatagoExpansionLock => {
            app.cfg.mcts.katago.expansion_lock = !app.cfg.mcts.katago.expansion_lock;
        }
        FieldId::TrainingMode => {
            // Toggle between epochs mode (steps_per_iteration=None) and steps mode
            if app.cfg.training.steps_per_iteration.is_some() {
                // Currently in steps mode → switch to epochs mode
                app.cfg.training.steps_per_iteration = None;
            } else {
                // Currently in epochs mode → switch to steps mode
                // Default to 500 steps if not set
                app.cfg.training.steps_per_iteration = Some(500);
            }
        }
        FieldId::TrainingContinuousCandidateTraining => {
            app.cfg.training.continuous_candidate_training =
                !app.cfg.training.continuous_candidate_training;
        }
        FieldId::TrainingResetOptimizer => {
            app.cfg.training.reset_optimizer = !app.cfg.training.reset_optimizer;
        }
        FieldId::TrainingOptimizer => {
            // Space-cycle: adamw → adam → sgd → adamw
            app.cfg.training.optimizer = match app.cfg.training.optimizer.to_lowercase().as_str() {
                "adamw" => "adam".to_string(),
                "adam" => "sgd".to_string(),
                _ => "adamw".to_string(),
            };
        }
        FieldId::ModelKind => {
            // Cycle: residual → mlp → residual
            app.cfg.model.kind = match app.cfg.model.kind.as_str() {
                "residual" => "mlp".to_string(),
                _ => "residual".to_string(),
            };
        }
        _ => {}
    }
    // Best-effort validation; keep value but surface error if invalid.
    if let Err(e) = crate::validate::validate_config(&app.cfg) {
        app.form.last_validation_error = Some(e);
    } else {
        app.form.last_validation_error = None;
    }
}

fn commit_field_edit(app: &mut App, field: FieldId) {
    let buf = app.form.input_buf.trim().to_string();
    let mut next = app.cfg.clone();
    let res = apply_input_to_cfg(&mut next, field, &buf)
        .and_then(|()| crate::validate::validate_config(&next));
    match res {
        Ok(()) => {
            app.cfg = next;
            app.form.last_validation_error = None;
            app.form.edit_mode = EditMode::View;
            app.form.input_buf.clear();
        }
        Err(e) => {
            app.form.last_validation_error = Some(e);
        }
    }
}

fn apply_input_to_cfg(cfg: &mut yz_core::Config, field: FieldId, buf: &str) -> Result<(), String> {
    match field {
        FieldId::InferBind => {
            cfg.inference.bind = buf.to_string();
            Ok(())
        }
        FieldId::InferDevice => {
            if buf == "cpu" || buf == "cuda" || buf == "mps" {
                cfg.inference.device = buf.to_string();
                Ok(())
            } else {
                Err("inference.device must be cpu|cuda|mps".to_string())
            }
        }
        FieldId::InferProtocolVersion => {
            let v = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            if v == 1 || v == 2 {
                cfg.inference.protocol_version = v;
                Ok(())
            } else {
                Err("inference.protocol_version must be 1|2".to_string())
            }
        }
        FieldId::InferLegalMaskBitset => {
            let b = match buf.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "y" => true,
                "false" | "0" | "no" | "n" => false,
                _ => return Err("inference.legal_mask_bitset must be true|false".to_string()),
            };
            cfg.inference.legal_mask_bitset = b;
            Ok(())
        }
        FieldId::InferMaxBatch => {
            cfg.inference.max_batch = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::InferMaxWaitUs => {
            cfg.inference.max_wait_us =
                buf.parse::<u64>().map_err(|_| "invalid u64".to_string())?;
            Ok(())
        }
        FieldId::InferTorchThreads => {
            cfg.inference.torch_threads = if buf.trim().is_empty() {
                None
            } else {
                Some(buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?)
            };
            Ok(())
        }
        FieldId::InferTorchInteropThreads => {
            cfg.inference.torch_interop_threads = if buf.trim().is_empty() {
                None
            } else {
                Some(buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?)
            };
            Ok(())
        }
        FieldId::InferDebugLog => {
            let b = match buf.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "y" => true,
                "false" | "0" | "no" | "n" => false,
                _ => return Err("inference.debug_log must be true|false".to_string()),
            };
            cfg.inference.debug_log = b;
            Ok(())
        }
        FieldId::InferPrintStats => {
            let b = match buf.trim().to_ascii_lowercase().as_str() {
                "true" | "1" | "yes" | "y" => true,
                "false" | "0" | "no" | "n" => false,
                _ => return Err("inference.print_stats must be true|false".to_string()),
            };
            cfg.inference.print_stats = b;
            Ok(())
        }

        FieldId::MctsCPuct => {
            cfg.mcts.c_puct = buf.parse::<f32>().map_err(|_| "invalid f32".to_string())?;
            Ok(())
        }
        FieldId::MctsBudgetReroll => {
            cfg.mcts.budget_reroll = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::MctsBudgetMark => {
            cfg.mcts.budget_mark = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::MctsMaxInflightPerGame => {
            cfg.mcts.max_inflight_per_game =
                buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::MctsDirichletAlpha => {
            cfg.mcts.dirichlet_alpha = buf.parse::<f32>().map_err(|_| "invalid f32".to_string())?;
            Ok(())
        }
        FieldId::MctsDirichletEpsilon => {
            cfg.mcts.dirichlet_epsilon =
                buf.parse::<f32>().map_err(|_| "invalid f32".to_string())?;
            Ok(())
        }
        FieldId::MctsTempKind => {
            let t0 = match cfg.mcts.temperature_schedule {
                TemperatureSchedule::Constant { t0 } => t0,
                TemperatureSchedule::Step { t0, .. } => t0,
            };
            cfg.mcts.temperature_schedule = match buf {
                "constant" => TemperatureSchedule::Constant { t0 },
                "step" => TemperatureSchedule::Step {
                    t0,
                    t1: 0.1,
                    cutoff_turn: 10,
                },
                _ => return Err("temperature kind must be constant|step".to_string()),
            };
            Ok(())
        }
        FieldId::MctsTempT0 => {
            let v = buf.parse::<f32>().map_err(|_| "invalid f32".to_string())?;
            match &mut cfg.mcts.temperature_schedule {
                TemperatureSchedule::Constant { t0 } => *t0 = v,
                TemperatureSchedule::Step { t0, .. } => *t0 = v,
            }
            Ok(())
        }
        FieldId::MctsTempT1 => {
            let v = buf.parse::<f32>().map_err(|_| "invalid f32".to_string())?;
            match &mut cfg.mcts.temperature_schedule {
                TemperatureSchedule::Step { t1, .. } => {
                    *t1 = v;
                    Ok(())
                }
                _ => Err("t1 only applies when kind=step".to_string()),
            }
        }
        FieldId::MctsTempCutoffTurn => {
            let v = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            match &mut cfg.mcts.temperature_schedule {
                TemperatureSchedule::Step { cutoff_turn, .. } => {
                    *cutoff_turn = v;
                    Ok(())
                }
                _ => Err("cutoff_turn only applies when kind=step".to_string()),
            }
        }
        FieldId::MctsVirtualLossMode => {
            let s = buf.trim().to_lowercase();
            match s.as_str() {
                "q_penalty" | "n_virtual_only" | "off" => {
                    cfg.mcts.virtual_loss_mode = s;
                    Ok(())
                }
                _ => Err("virtual_loss_mode must be q_penalty|n_virtual_only|off".to_string()),
            }
        }
        FieldId::MctsVirtualLoss => {
            cfg.mcts.virtual_loss = buf.parse::<f32>().map_err(|_| "invalid f32".to_string())?;
            Ok(())
        }
        FieldId::MctsKatagoExpansionLock => {
            let s = buf.trim().to_lowercase();
            match s.as_str() {
                "true" | "1" | "yes" | "y" => {
                    cfg.mcts.katago.expansion_lock = true;
                    Ok(())
                }
                "false" | "0" | "no" | "n" => {
                    cfg.mcts.katago.expansion_lock = false;
                    Ok(())
                }
                _ => Err("expansion_lock must be true|false".to_string()),
            }
        }

        FieldId::SelfplayGamesPerIteration => {
            cfg.selfplay.games_per_iteration =
                buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::SelfplayWorkers => {
            cfg.selfplay.workers = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::SelfplayThreadsPerWorker => {
            cfg.selfplay.threads_per_worker =
                buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }

        FieldId::TrainingMode => {
            match buf.trim().to_lowercase().as_str() {
                "epochs" => {
                    cfg.training.steps_per_iteration = None;
                    Ok(())
                }
                "steps" => {
                    if cfg.training.steps_per_iteration.is_none() {
                        cfg.training.steps_per_iteration = Some(500);
                    }
                    Ok(())
                }
                _ => Err("expected 'epochs' or 'steps'".to_string()),
            }
        }
        FieldId::TrainingBatchSize => {
            cfg.training.batch_size = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::TrainingLearningRate => {
            cfg.training.learning_rate =
                buf.parse::<f64>().map_err(|_| "invalid f64".to_string())?;
            Ok(())
        }
        FieldId::TrainingContinuousCandidateTraining => {
            cfg.training.continuous_candidate_training = buf
                .parse::<bool>()
                .map_err(|_| "invalid bool".to_string())?;
            Ok(())
        }
        FieldId::TrainingResetOptimizer => {
            cfg.training.reset_optimizer = buf
                .parse::<bool>()
                .map_err(|_| "invalid bool".to_string())?;
            Ok(())
        }
        FieldId::TrainingOptimizer => {
            cfg.training.optimizer = buf.trim().to_string();
            Ok(())
        }
        FieldId::TrainingEpochs => {
            cfg.training.epochs = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::TrainingWeightDecay => {
            cfg.training.weight_decay =
                buf.parse::<f64>().map_err(|_| "invalid f64".to_string())?;
            Ok(())
        }
        FieldId::TrainingStepsPerIteration => {
            cfg.training.steps_per_iteration = if buf.trim().is_empty() {
                None
            } else {
                Some(buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?)
            };
            Ok(())
        }

        FieldId::GatingGames => {
            cfg.gating.games = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::GatingSeed => {
            cfg.gating.seed = buf.parse::<u64>().map_err(|_| "invalid u64".to_string())?;
            Ok(())
        }
        FieldId::GatingSeedSetId => {
            cfg.gating.seed_set_id = if buf.trim().is_empty() {
                None
            } else {
                Some(buf.trim().to_string())
            };
            Ok(())
        }
        FieldId::GatingWinRateThreshold => {
            cfg.gating.win_rate_threshold =
                buf.parse::<f64>().map_err(|_| "invalid f64".to_string())?;
            Ok(())
        }
        FieldId::GatingPairedSeedSwap => {
            cfg.gating.paired_seed_swap = buf
                .parse::<bool>()
                .map_err(|_| "invalid bool".to_string())?;
            Ok(())
        }
        FieldId::GatingDeterministicChance => {
            cfg.gating.deterministic_chance = buf
                .parse::<bool>()
                .map_err(|_| "invalid bool".to_string())?;
            Ok(())
        }
        FieldId::GatingKatagoSprt => {
            cfg.gating.katago.sprt = buf
                .parse::<bool>()
                .map_err(|_| "invalid bool".to_string())?;
            Ok(())
        }
        FieldId::GatingKatagoSprtMinGames => {
            cfg.gating.katago.sprt_min_games =
                buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::GatingKatagoSprtMaxGames => {
            cfg.gating.katago.sprt_max_games =
                buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::GatingKatagoSprtAlpha => {
            cfg.gating.katago.sprt_alpha =
                buf.parse::<f64>().map_err(|_| "invalid f64".to_string())?;
            Ok(())
        }
        FieldId::GatingKatagoSprtBeta => {
            cfg.gating.katago.sprt_beta =
                buf.parse::<f64>().map_err(|_| "invalid f64".to_string())?;
            Ok(())
        }
        FieldId::GatingKatagoSprtDelta => {
            cfg.gating.katago.sprt_delta =
                buf.parse::<f64>().map_err(|_| "invalid f64".to_string())?;
            Ok(())
        }

        FieldId::ReplayCapacityShards => {
            cfg.replay.capacity_shards = if buf.trim().is_empty() {
                None
            } else {
                Some(buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?)
            };
            Ok(())
        }

        FieldId::ControllerTotalIterations => {
            cfg.controller.total_iterations = if buf.trim().is_empty() {
                None
            } else {
                Some(buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?)
            };
            Ok(())
        }

        FieldId::ModelHiddenDim => {
            cfg.model.hidden_dim = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::ModelNumBlocks => {
            cfg.model.num_blocks = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::ModelKind => {
            let s = buf.trim().to_lowercase();
            match s.as_str() {
                "residual" | "mlp" => {
                    cfg.model.kind = s;
                    Ok(())
                }
                _ => Err("model.kind must be residual|mlp".to_string()),
            }
        }
    }
}

fn field_value_string(cfg: &yz_core::Config, field: FieldId) -> String {
    match field {
        FieldId::InferBind => cfg.inference.bind.clone(),
        FieldId::InferDevice => cfg.inference.device.clone(),
        FieldId::InferProtocolVersion => cfg.inference.protocol_version.to_string(),
        FieldId::InferLegalMaskBitset => cfg.inference.legal_mask_bitset.to_string(),
        FieldId::InferMaxBatch => cfg.inference.max_batch.to_string(),
        FieldId::InferMaxWaitUs => cfg.inference.max_wait_us.to_string(),
        FieldId::InferTorchThreads => cfg
            .inference
            .torch_threads
            .map(|x| x.to_string())
            .unwrap_or_default(),
        FieldId::InferTorchInteropThreads => cfg
            .inference
            .torch_interop_threads
            .map(|x| x.to_string())
            .unwrap_or_default(),
        FieldId::InferDebugLog => cfg.inference.debug_log.to_string(),
        FieldId::InferPrintStats => cfg.inference.print_stats.to_string(),

        FieldId::MctsCPuct => format!("{:.4}", cfg.mcts.c_puct),
        FieldId::MctsBudgetReroll => cfg.mcts.budget_reroll.to_string(),
        FieldId::MctsBudgetMark => cfg.mcts.budget_mark.to_string(),
        FieldId::MctsMaxInflightPerGame => cfg.mcts.max_inflight_per_game.to_string(),
        FieldId::MctsDirichletAlpha => format!("{:.4}", cfg.mcts.dirichlet_alpha),
        FieldId::MctsDirichletEpsilon => format!("{:.4}", cfg.mcts.dirichlet_epsilon),
        FieldId::MctsTempKind => match cfg.mcts.temperature_schedule {
            TemperatureSchedule::Constant { .. } => "constant".to_string(),
            TemperatureSchedule::Step { .. } => "step".to_string(),
        },
        FieldId::MctsTempT0 => match cfg.mcts.temperature_schedule {
            TemperatureSchedule::Constant { t0 } => format!("{:.4}", t0),
            TemperatureSchedule::Step { t0, .. } => format!("{:.4}", t0),
        },
        FieldId::MctsTempT1 => match cfg.mcts.temperature_schedule {
            TemperatureSchedule::Step { t1, .. } => format!("{:.4}", t1),
            _ => "(n/a)".to_string(),
        },
        FieldId::MctsTempCutoffTurn => match cfg.mcts.temperature_schedule {
            TemperatureSchedule::Step { cutoff_turn, .. } => cutoff_turn.to_string(),
            _ => "(n/a)".to_string(),
        },
        FieldId::MctsVirtualLossMode => cfg.mcts.virtual_loss_mode.clone(),
        FieldId::MctsVirtualLoss => format!("{:.3}", cfg.mcts.virtual_loss),
        FieldId::MctsKatagoExpansionLock => cfg.mcts.katago.expansion_lock.to_string(),

        FieldId::SelfplayGamesPerIteration => cfg.selfplay.games_per_iteration.to_string(),
        FieldId::SelfplayWorkers => cfg.selfplay.workers.to_string(),
        FieldId::SelfplayThreadsPerWorker => cfg.selfplay.threads_per_worker.to_string(),

        FieldId::TrainingMode => {
            if cfg.training.steps_per_iteration.is_some() {
                "steps".to_string()
            } else {
                "epochs".to_string()
            }
        }
        FieldId::TrainingBatchSize => cfg.training.batch_size.to_string(),
        FieldId::TrainingLearningRate => format!("{:.6}", cfg.training.learning_rate),
        FieldId::TrainingContinuousCandidateTraining => {
            cfg.training.continuous_candidate_training.to_string()
        }
        FieldId::TrainingResetOptimizer => cfg.training.reset_optimizer.to_string(),
        FieldId::TrainingOptimizer => cfg.training.optimizer.clone(),
        FieldId::TrainingEpochs => cfg.training.epochs.to_string(),
        FieldId::TrainingWeightDecay => format!("{:.6}", cfg.training.weight_decay),
        FieldId::TrainingStepsPerIteration => cfg
            .training
            .steps_per_iteration
            .map(|x| x.to_string())
            .unwrap_or_default(),

        FieldId::GatingGames => cfg.gating.games.to_string(),
        FieldId::GatingSeed => cfg.gating.seed.to_string(),
        FieldId::GatingSeedSetId => cfg.gating.seed_set_id.clone().unwrap_or_default(),
        FieldId::GatingWinRateThreshold => format!("{:.4}", cfg.gating.win_rate_threshold),
        FieldId::GatingPairedSeedSwap => cfg.gating.paired_seed_swap.to_string(),
        FieldId::GatingDeterministicChance => cfg.gating.deterministic_chance.to_string(),
        FieldId::GatingKatagoSprt => cfg.gating.katago.sprt.to_string(),
        FieldId::GatingKatagoSprtMinGames => cfg.gating.katago.sprt_min_games.to_string(),
        FieldId::GatingKatagoSprtMaxGames => cfg.gating.katago.sprt_max_games.to_string(),
        FieldId::GatingKatagoSprtAlpha => format!("{:.4}", cfg.gating.katago.sprt_alpha),
        FieldId::GatingKatagoSprtBeta => format!("{:.4}", cfg.gating.katago.sprt_beta),
        FieldId::GatingKatagoSprtDelta => format!("{:.4}", cfg.gating.katago.sprt_delta),

        FieldId::ReplayCapacityShards => cfg
            .replay
            .capacity_shards
            .map(|x| x.to_string())
            .unwrap_or_default(),

        FieldId::ControllerTotalIterations => cfg
            .controller
            .total_iterations
            .map(|x| x.to_string())
            .unwrap_or_default(),

        FieldId::ModelHiddenDim => cfg.model.hidden_dim.to_string(),
        FieldId::ModelNumBlocks => cfg.model.num_blocks.to_string(),
        FieldId::ModelKind => cfg.model.kind.clone(),
    }
}

fn step_field(app: &mut App, field: FieldId, dir: i32, step: StepSize) {
    let mut next = app.cfg.clone();
    let d = if dir >= 0 { 1.0 } else { -1.0 };
    let ok = match field {
        FieldId::MctsVirtualLossMode => {
            let opts = ["q_penalty", "n_virtual_only", "off"];
            let cur = next.mcts.virtual_loss_mode.as_str();
            let mut idx = opts.iter().position(|x| *x == cur).unwrap_or(0) as i32;
            idx = (idx + dir.signum()).rem_euclid(opts.len() as i32);
            next.mcts.virtual_loss_mode = opts[idx as usize].to_string();
            true
        }
        FieldId::MctsVirtualLoss => {
            let inc = if step == StepSize::Large { 1.0 } else { 0.1 };
            next.mcts.virtual_loss = (next.mcts.virtual_loss as f64 + d * inc).max(0.0) as f32;
            true
        }
        FieldId::InferMaxBatch => {
            let inc = if step == StepSize::Large { 8 } else { 1 };
            next.inference.max_batch = if dir >= 0 {
                next.inference.max_batch.saturating_add(inc)
            } else {
                next.inference.max_batch.saturating_sub(inc)
            };
            true
        }
        FieldId::InferMaxWaitUs => {
            let inc = if step == StepSize::Large {
                10_000
            } else {
                1_000
            };
            next.inference.max_wait_us = if dir >= 0 {
                next.inference.max_wait_us.saturating_add(inc)
            } else {
                next.inference.max_wait_us.saturating_sub(inc)
            };
            true
        }
        FieldId::MctsCPuct => {
            let inc = if step == StepSize::Large { 0.5 } else { 0.1 };
            next.mcts.c_puct = (next.mcts.c_puct as f64 + d * inc).max(0.0) as f32;
            true
        }
        FieldId::MctsBudgetReroll => {
            let inc = if step == StepSize::Large { 100 } else { 10 };
            next.mcts.budget_reroll = if dir >= 0 {
                next.mcts.budget_reroll.saturating_add(inc)
            } else {
                next.mcts.budget_reroll.saturating_sub(inc)
            };
            true
        }
        FieldId::MctsBudgetMark => {
            let inc = if step == StepSize::Large { 100 } else { 10 };
            next.mcts.budget_mark = if dir >= 0 {
                next.mcts.budget_mark.saturating_add(inc)
            } else {
                next.mcts.budget_mark.saturating_sub(inc)
            };
            true
        }
        FieldId::MctsMaxInflightPerGame => {
            let inc = if step == StepSize::Large { 4 } else { 1 };
            next.mcts.max_inflight_per_game = if dir >= 0 {
                next.mcts.max_inflight_per_game.saturating_add(inc)
            } else {
                next.mcts.max_inflight_per_game.saturating_sub(inc)
            };
            true
        }
        FieldId::MctsDirichletAlpha => {
            let inc = if step == StepSize::Large { 0.1 } else { 0.01 };
            next.mcts.dirichlet_alpha =
                (next.mcts.dirichlet_alpha as f64 + d * inc).max(0.0) as f32;
            true
        }
        FieldId::MctsDirichletEpsilon => {
            let inc = if step == StepSize::Large { 0.1 } else { 0.01 };
            next.mcts.dirichlet_epsilon =
                (next.mcts.dirichlet_epsilon as f64 + d * inc).clamp(0.0, 1.0) as f32;
            true
        }
        FieldId::MctsKatagoExpansionLock => {
            next.mcts.katago.expansion_lock = !next.mcts.katago.expansion_lock;
            true
        }
        FieldId::MctsTempT0 => {
            let inc = if step == StepSize::Large { 0.5 } else { 0.1 };
            match &mut next.mcts.temperature_schedule {
                TemperatureSchedule::Constant { t0 } => {
                    *t0 = (*t0 as f64 + d * inc).max(0.0) as f32;
                }
                TemperatureSchedule::Step { t0, .. } => {
                    *t0 = (*t0 as f64 + d * inc).max(0.0) as f32;
                }
            }
            true
        }
        FieldId::MctsTempT1 => {
            let inc = if step == StepSize::Large { 0.5 } else { 0.1 };
            if let TemperatureSchedule::Step { t1, .. } = &mut next.mcts.temperature_schedule {
                *t1 = (*t1 as f64 + d * inc).max(0.0) as f32;
                true
            } else {
                false
            }
        }
        FieldId::MctsTempCutoffTurn => {
            let inc = if step == StepSize::Large { 10 } else { 1 };
            if let TemperatureSchedule::Step { cutoff_turn, .. } =
                &mut next.mcts.temperature_schedule
            {
                *cutoff_turn = if dir >= 0 {
                    cutoff_turn.saturating_add(inc)
                } else {
                    cutoff_turn.saturating_sub(inc)
                };
                true
            } else {
                false
            }
        }
        FieldId::SelfplayGamesPerIteration => {
            let inc = if step == StepSize::Large { 50 } else { 1 };
            next.selfplay.games_per_iteration = if dir >= 0 {
                next.selfplay.games_per_iteration.saturating_add(inc)
            } else {
                next.selfplay.games_per_iteration.saturating_sub(inc)
            };
            true
        }
        FieldId::SelfplayWorkers => {
            let inc = if step == StepSize::Large { 4 } else { 1 };
            next.selfplay.workers = if dir >= 0 {
                next.selfplay.workers.saturating_add(inc)
            } else {
                next.selfplay.workers.saturating_sub(inc)
            };
            true
        }
        FieldId::SelfplayThreadsPerWorker => {
            let inc = if step == StepSize::Large { 4 } else { 1 };
            next.selfplay.threads_per_worker = if dir >= 0 {
                next.selfplay.threads_per_worker.saturating_add(inc)
            } else {
                next.selfplay.threads_per_worker.saturating_sub(inc)
            };
            true
        }
        FieldId::TrainingMode => {
            // Toggle between epochs and steps mode on left/right
            if next.training.steps_per_iteration.is_some() {
                next.training.steps_per_iteration = None;
            } else {
                next.training.steps_per_iteration = Some(500);
            }
            true
        }
        FieldId::TrainingBatchSize => {
            let inc = if step == StepSize::Large { 256 } else { 32 };
            next.training.batch_size = if dir >= 0 {
                next.training.batch_size.saturating_add(inc)
            } else {
                next.training.batch_size.saturating_sub(inc)
            };
            true
        }
        FieldId::TrainingLearningRate => {
            let inc = if step == StepSize::Large { 1e-3 } else { 1e-4 };
            next.training.learning_rate = (next.training.learning_rate + d * inc).max(1e-12);
            true
        }
        FieldId::TrainingWeightDecay => {
            let inc = if step == StepSize::Large { 1e-2 } else { 1e-3 };
            next.training.weight_decay = (next.training.weight_decay + d * inc).max(0.0);
            true
        }
        FieldId::TrainingEpochs => {
            let inc = if step == StepSize::Large { 10 } else { 1 };
            next.training.epochs = if dir >= 0 {
                next.training.epochs.saturating_add(inc)
            } else {
                next.training.epochs.saturating_sub(inc)
            };
            true
        }
        FieldId::TrainingStepsPerIteration => {
            let cur = next.training.steps_per_iteration.unwrap_or(0);
            let inc = if step == StepSize::Large { 200 } else { 25 };
            let v = if dir >= 0 {
                cur.saturating_add(inc)
            } else {
                cur.saturating_sub(inc)
            };
            next.training.steps_per_iteration = if v == 0 { None } else { Some(v) };
            true
        }
        FieldId::GatingGames => {
            let inc = if step == StepSize::Large { 50 } else { 1 };
            next.gating.games = if dir >= 0 {
                next.gating.games.saturating_add(inc)
            } else {
                next.gating.games.saturating_sub(inc)
            };
            true
        }
        FieldId::GatingSeed => {
            let inc = if step == StepSize::Large { 100 } else { 1 };
            next.gating.seed = if dir >= 0 {
                next.gating.seed.saturating_add(inc)
            } else {
                next.gating.seed.saturating_sub(inc)
            };
            true
        }
        FieldId::GatingWinRateThreshold => {
            let inc = if step == StepSize::Large { 0.05 } else { 0.01 };
            next.gating.win_rate_threshold =
                (next.gating.win_rate_threshold + d * inc).clamp(0.0, 1.0);
            true
        }
        FieldId::GatingKatagoSprtMinGames => {
            let inc = if step == StepSize::Large { 20 } else { 5 };
            next.gating.katago.sprt_min_games = if dir >= 0 {
                next.gating.katago.sprt_min_games.saturating_add(inc)
            } else {
                next.gating.katago.sprt_min_games.saturating_sub(inc)
            };
            true
        }
        FieldId::GatingKatagoSprtMaxGames => {
            let inc = if step == StepSize::Large { 100 } else { 20 };
            next.gating.katago.sprt_max_games = if dir >= 0 {
                next.gating.katago.sprt_max_games.saturating_add(inc)
            } else {
                next.gating.katago.sprt_max_games.saturating_sub(inc)
            };
            true
        }
        FieldId::GatingKatagoSprtAlpha => {
            let inc = if step == StepSize::Large { 0.05 } else { 0.01 };
            next.gating.katago.sprt_alpha = (next.gating.katago.sprt_alpha + d * inc).clamp(1e-6, 1.0);
            true
        }
        FieldId::GatingKatagoSprtBeta => {
            let inc = if step == StepSize::Large { 0.05 } else { 0.01 };
            next.gating.katago.sprt_beta = (next.gating.katago.sprt_beta + d * inc).clamp(1e-6, 1.0);
            true
        }
        FieldId::GatingKatagoSprtDelta => {
            let inc = if step == StepSize::Large { 0.02 } else { 0.005 };
            next.gating.katago.sprt_delta = (next.gating.katago.sprt_delta + d * inc).max(0.0);
            true
        }
        FieldId::ReplayCapacityShards => {
            let cur = next.replay.capacity_shards.unwrap_or(0);
            let inc = if step == StepSize::Large { 100 } else { 10 };
            let v = if dir >= 0 {
                cur.saturating_add(inc)
            } else {
                cur.saturating_sub(inc)
            };
            next.replay.capacity_shards = if v == 0 { None } else { Some(v) };
            true
        }
        FieldId::ControllerTotalIterations => {
            let cur = next.controller.total_iterations.unwrap_or(0);
            let inc = if step == StepSize::Large { 10 } else { 1 };
            let v = if dir >= 0 {
                cur.saturating_add(inc)
            } else {
                cur.saturating_sub(inc)
            };
            next.controller.total_iterations = if v == 0 { None } else { Some(v) };
            true
        }
        FieldId::ModelHiddenDim => {
            let inc = if step == StepSize::Large { 64 } else { 32 };
            next.model.hidden_dim = if dir >= 0 {
                next.model.hidden_dim.saturating_add(inc)
            } else {
                next.model.hidden_dim.saturating_sub(inc).max(32)
            };
            true
        }
        FieldId::ModelNumBlocks => {
            let inc = if step == StepSize::Large { 2 } else { 1 };
            next.model.num_blocks = if dir >= 0 {
                next.model.num_blocks.saturating_add(inc)
            } else {
                next.model.num_blocks.saturating_sub(inc).max(1)
            };
            true
        }
        FieldId::ModelKind => {
            let opts = ["residual", "mlp"];
            let cur = next.model.kind.as_str();
            let mut idx = opts.iter().position(|x| *x == cur).unwrap_or(0) as i32;
            idx = (idx + dir.signum()).rem_euclid(opts.len() as i32);
            next.model.kind = opts[idx as usize].to_string();
            true
        }
        _ => false,
    };

    if !ok {
        return;
    }
    if let Err(e) = crate::validate::validate_config(&next) {
        app.form.last_validation_error = Some(e);
        return;
    }
    app.cfg = next;
    app.form.last_validation_error = None;
}

fn draw(f: &mut ratatui::Frame, app: &App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(1), Constraint::Length(3)].as_ref())
        .split(f.area());

    match app.screen {
        Screen::Home => {
            let title = Line::from(vec![
                Span::styled("yA0tzy", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::raw("Runs"),
            ]);
            let items: Vec<ListItem> = if app.runs.is_empty() {
                vec![ListItem::new(Line::from("No runs yet"))]
            } else {
                let sel = app.selected.min(app.runs.len().saturating_sub(1));
                let view_h = chunks[0].height.saturating_sub(2).max(1) as usize;
                let mut start = sel.saturating_sub(view_h.saturating_sub(1));
                if start + view_h > app.runs.len() {
                    start = app.runs.len().saturating_sub(view_h);
                }
                app.runs
                    .iter()
                    .enumerate()
                    .skip(start)
                    .take(view_h)
                    .map(|(i, r)| {
                        let prefix = if i == sel { "> " } else { "  " };
                        ListItem::new(Line::from(format!("{prefix}{r}")))
                    })
                    .collect()
            };
            let list = List::new(items).block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(list, chunks[0]);
        }
        Screen::NamingRun => {
            let title = Line::from(vec![
                Span::styled("New Run", Style::default().add_modifier(Modifier::BOLD)),
            ]);
            let mut lines: Vec<Line> = Vec::new();
            lines.push(Line::from(""));
            lines.push(Line::from("  Enter a name for the new run:"));
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::raw("  ▸ "),
                Span::styled(
                    if app.naming_input.is_empty() {
                        "(leave empty for auto-generated name)".to_string()
                    } else {
                        app.naming_input.clone()
                    },
                    if app.naming_input.is_empty() {
                        Style::default().fg(Color::DarkGray)
                    } else {
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                    },
                ),
                Span::styled("█", Style::default().fg(Color::Cyan)),
            ]));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  Enter to confirm, Esc to cancel",
                Style::default().fg(Color::DarkGray),
            )));
            let p = Paragraph::new(lines).block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(p, chunks[0]);
        }
        Screen::ExtendingRun => {
            let src = app
                .extend_source_run_id
                .as_deref()
                .unwrap_or("<no source>");
            let title = Line::from(vec![Span::styled(
                "Extend Run",
                Style::default().add_modifier(Modifier::BOLD),
            )]);
            let mut lines: Vec<Line> = Vec::new();
            lines.push(Line::from(""));
            lines.push(Line::from(format!("  Source: {src}")));
            lines.push(Line::from(format!(
                "  Copy replay: {}  (Space to toggle)",
                if app.extend_include_replay { "yes" } else { "no" }
            )));
            lines.push(Line::from(""));
            lines.push(Line::from("  Enter a name for the new run:"));
            lines.push(Line::from(""));
            lines.push(Line::from(vec![
                Span::raw("  ▸ "),
                Span::styled(
                    if app.extend_input.is_empty() {
                        "(required)".to_string()
                    } else {
                        app.extend_input.clone()
                    },
                    if app.extend_input.is_empty() {
                        Style::default().fg(Color::DarkGray)
                    } else {
                        Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
                    },
                ),
                Span::styled("█", Style::default().fg(Color::Cyan)),
            ]));
            lines.push(Line::from(""));
            lines.push(Line::from(Span::styled(
                "  Enter to confirm, Esc to cancel",
                Style::default().fg(Color::DarkGray),
            )));
            let p = Paragraph::new(lines).block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(p, chunks[0]);
        }
        Screen::Config => {
            let rid = app.active_run_id.as_deref().unwrap_or("<no run>");
            let title = Line::from(vec![
                Span::styled("Config", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::raw(format!("run={rid}")),
            ]);
            let body = render_config_lines(app, chunks[0].height.saturating_sub(2) as usize);
            let p = Paragraph::new(body).block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(p, chunks[0]);
        }
        Screen::Dashboard => {
            let rid = app.active_run_id.as_deref().unwrap_or("<no run>");
            if matches!(app.dashboard_tab, DashboardTab::Learning) {
                draw_dashboard_learning(f, app, chunks[0]);
            } else {
            let title = Line::from(vec![
                Span::styled("Performance", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::raw(rid.to_string()),
            ]);

            let cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
                .split(chunks[0]);

            // Left: iteration history table.
            let mut left: Vec<Line> = Vec::new();
            match (&app.dashboard_manifest, &app.dashboard_err) {
                (Some(m), _) => {
                    // Visible height inside the left block (minus borders).
                    let left_view_h = cols[0].height.saturating_sub(2) as usize;

                    // Iteration-level AvgIter + ETA (completed iterations only).
                    if let Some(total_planned) = app.dashboard_planned_total_iterations {
                        let mut completed_durs_ms: Vec<u64> = Vec::new();
                        for it in &m.iterations {
                            if let Some(end) = it.ended_ts_ms {
                                completed_durs_ms.push(end.saturating_sub(it.started_ts_ms));
                            }
                        }
                        if !completed_durs_ms.is_empty() {
                            let sum: u64 = completed_durs_ms.iter().sum();
                            let avg_ms = sum / completed_durs_ms.len() as u64;
                            let avg_s = avg_ms / 1000;
                            let avg_m = avg_s / 60;
                            let avg_s_rem = avg_s % 60;
                            left.push(Line::from(format!(" AvgIter: {avg_m}m {avg_s_rem:02}s")));

                            let completed_iters = completed_durs_ms.len() as u32;
                            let remaining = total_planned.saturating_sub(completed_iters);
                            let eta_ms = avg_ms.saturating_mul(remaining as u64);
                            let eta_s = eta_ms / 1000;
                            let eta_m = eta_s / 60;
                            let eta_s_rem = eta_s % 60;
                            left.push(Line::from(format!(
                                " ETA: {eta_m}m {eta_s_rem:02}s (remaining {remaining}/{total_planned})"
                            )));
                            left.push(Line::from(""));
                        }
                    }

                    // Header row
                    // Keep header widths in sync with the row formatter below to avoid visual drift.
                    const W_ITER: usize = 3;
                    const W_DEC: usize = 8; // "Decision" is 8 chars
                    const W_GG: usize = 11; // "GatingGames" is 11 chars
                    const W_WR: usize = 6; // "WinRate" is 6 chars
                    const W_SCORE: usize = 11; // " 123.4/ 567.8"
                    const W_ORA: usize = 6; // "Oracle"
                    left.push(Line::from(Span::styled(
                        format!(
                            "  {iter:>W_ITER$}   {dec:>W_DEC$}   {gg:>W_GG$}   {wr:>W_WR$}   {sc:>W_SCORE$}   {ora:>W_ORA$}   {loss}",
                            iter = "Iter",
                            dec = "Decision",
                            gg = "GatingGames",
                            wr = "WinRate",
                            sc = "Score(c/b)",
                            ora = "Oracle",
                            loss = "Loss (t/p/v)",
                            W_ITER = W_ITER,
                            W_DEC = W_DEC,
                            W_GG = W_GG,
                            W_WR = W_WR,
                            W_SCORE = W_SCORE,
                            W_ORA = W_ORA,
                        ),
                        Style::default().fg(Color::DarkGray),
                    )));

                    let cur_idx = m.controller_iteration_idx;
                    if m.iterations.is_empty() {
                        left.push(Line::from(""));
                        left.push(Line::from(" (no iterations yet)"));
                    } else {
                        // Build iteration rows separately so we can slice for scrolling.
                        let mut iter_rows: Vec<Line> = Vec::new();
                        for it in &m.iterations {
                            // ASCII marker to avoid wide-char rendering differences across terminals.
                            let marker = if it.idx == cur_idx { ">" } else { " " };
                            let promo = it
                                .promoted
                                .map(|p| if p { "promote" } else { "reject" })
                                .unwrap_or("-");
                            let wr = it
                                .gate
                                .win_rate
                                .map(|x| format!("{x:.3}"))
                                .unwrap_or_else(|| "-".to_string());
                            let gating_games = if it.gate.games_target > 0 {
                                format!("{}/{}", it.gate.games_completed, it.gate.games_target)
                            } else {
                                "-".to_string()
                            };
                            let score_cb = match (it.gate.mean_cand_score, it.gate.mean_best_score) {
                                (Some(c), Some(b)) => format!("{c:>5.1}/{b:>5.1}"),
                                _ => "-".to_string(),
                            };
                            let oracle = it
                                .oracle
                                .match_rate_overall
                                .map(|x| format!("{x:.3}"))
                                .unwrap_or_else(|| "-".to_string());
                            let lt = it
                                .train
                                .last_loss_total
                                .map(|x| format!("{x:.3}"))
                                .unwrap_or_else(|| "-".to_string());
                            let lp = it
                                .train
                                .last_loss_policy
                                .map(|x| format!("{x:.3}"))
                                .unwrap_or_else(|| "-".to_string());
                            let lv = it
                                .train
                                .last_loss_value
                                .map(|x| format!("{x:.3}"))
                                .unwrap_or_else(|| "-".to_string());

                            let row_style = if it.idx == cur_idx {
                                Style::default().fg(Color::Cyan)
                            } else {
                                Style::default()
                            };
                            iter_rows.push(Line::from(Span::styled(
                                format!(
                                    "{marker} {:>W_ITER$}   {promo:<W_DEC$}   {gating_games:>W_GG$}   {wr:>W_WR$}   {score_cb:>W_SCORE$}   {oracle:>W_ORA$}   {lt:>5}/{lp:>5}/{lv:>5}",
                                    it.idx
                                ),
                                row_style,
                            )));
                        }

                        // Determine how many rows we can show (keep the lines already in `left` fixed).
                        let fixed_len = left.len();
                        let view_rows = left_view_h.saturating_sub(fixed_len).max(1);

                        // Compute effective scroll start.
                        let max_start = iter_rows.len().saturating_sub(view_rows);
                        let mut start = if app.dashboard_iter_follow {
                            // Keep current iteration visible (similar to config view).
                            let cur_pos = m
                                .iterations
                                .iter()
                                .position(|it| it.idx == cur_idx)
                                .unwrap_or(iter_rows.len().saturating_sub(1));
                            if cur_pos < view_rows {
                                0
                            } else {
                                cur_pos.saturating_sub(view_rows * 2 / 3)
                            }
                        } else {
                            app.dashboard_iter_scroll
                        };
                        start = start.min(max_start);
                        let end = (start + view_rows).min(iter_rows.len());

                        for line in &iter_rows[start..end] {
                            left.push(line.clone());
                        }
                    }
                }
                (_, Some(e)) => {
                    left.push(Line::from(format!(" (unavailable: {e})")));
                }
                _ => {
                    left.push(Line::from(" (no run selected)"));
                }
            }
            let left_p = Paragraph::new(left)
                .block(Block::default().title(title.clone()).borders(Borders::ALL));
            f.render_widget(left_p, cols[0]);

            // Right: phase view + progress bars.
            let mut right_lines: Vec<Line> = Vec::new();
            let mut gauge: Option<(f64, String)> = None;
            let mut phase_title = "Phase".to_string();
            if let Some(m) = &app.dashboard_manifest {
                let phase = m.controller_phase.as_deref().unwrap_or("?");
                let status = m.controller_status.as_deref().unwrap_or("");
                let is_cancelled = status == "cancelled"
                    || m.controller_error.as_deref() == Some("cancelled");
                // Set phase title for block
                phase_title = match phase {
                    "selfplay" => "Phase: Self-play".to_string(),
                    "train" => "Phase: Training".to_string(),
                    "gate" => "Phase: Gating".to_string(),
                    "idle" => "Phase: Idle".to_string(),
                    "error" if is_cancelled => "Phase: Cancelled".to_string(),
                    "error" => "Phase: Error".to_string(),
                    other => format!("Phase: {other}"),
                };
                // Show status/error message, but avoid duplicating "cancelled"
                if is_cancelled {
                    right_lines.push(Line::from("Run was cancelled."));
                } else {
                    if !status.is_empty() {
                        right_lines.push(Line::from(format!("{status}")));
                    }
                    if let Some(e) = m.controller_error.as_deref() {
                        if !e.is_empty() {
                            right_lines.push(Line::from(format!("error: {e}")));
                        }
                    }
                }
                if !right_lines.is_empty() {
                    right_lines.push(Line::from(""));
                }

                let cur_idx = m.controller_iteration_idx;
                let cur = m.iterations.iter().find(|it| it.idx == cur_idx);
                // Persisted provenance (controller writes these at train start / promotion).
                let best_iter = m
                    .best_promoted_iter
                    .or_else(|| {
                        m.iterations
                            .iter()
                            .filter_map(|it| it.promoted.and_then(|p| if p { Some(it.idx) } else { None }))
                            .max()
                    })
                    .unwrap_or(0);
                if let Some(it) = cur {
                    let opt = it.train.optimizer_kind.as_deref().unwrap_or("-");
                    let resumed = it.train.optimizer_resumed.unwrap_or(false);
                    let src = it
                        .train
                        .init_from
                        .as_deref()
                        .unwrap_or("-");
                    let src_iter = it
                        .train
                        .init_from_iter
                        .map(|x| x.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    right_lines.push(Line::from(format!(
                        "best=iter {best_iter} | train: {opt} ({}) from {src}:{src_iter}",
                        if resumed { "resumed" } else { "reset" }
                    )));
                    right_lines.push(Line::from(""));
                }
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_millis() as u64)
                    .unwrap_or(0);

                if let Some(it) = cur {
                    match phase {
                        "selfplay" => {
                            let done = it.selfplay.games_completed;
                            let tot = it.selfplay.games_target.max(1);
                            right_lines.push(Line::from(format!("{done} / {tot} games")));
                            gauge = Some((done as f64 / tot as f64, String::new()));

                            // Timing stats
                            if let Some(started) = it.selfplay.started_ts_ms {
                                if let Some(first) = it.selfplay.first_game_started_ts_ms {
                                    let setup_ms = first.saturating_sub(started);
                                    let s = setup_ms / 1000;
                                    let m2 = s / 60;
                                    let s2 = s % 60;
                                    right_lines.push(Line::from(format!("setup: {m2}m {s2:02}s")));
                                    let run_ms = now_ms.saturating_sub(first);
                                    let rs = run_ms / 1000;
                                    let rm = rs / 60;
                                    let rs2 = rs % 60;
                                    right_lines.push(Line::from(format!("running: {rm}m {rs2:02}s")));

                                    if done > 0 {
                                        let avg_ms = run_ms / done as u64;
                                        let remaining = tot.saturating_sub(done);
                                        let eta_ms = avg_ms * remaining as u64;
                                        let avg_s = avg_ms as f64 / 1000.0;
                                        let eta_s = eta_ms / 1000;
                                        let eta_m = eta_s / 60;
                                        let eta_s_rem = eta_s % 60;
                                        right_lines.push(Line::from(format!(
                                            "avg: {avg_s:.1}s/game  ETA: {eta_m}m {eta_s_rem}s"
                                        )));
                                    }
                                } else {
                                    let elapsed_ms = now_ms.saturating_sub(started);
                                    let s = elapsed_ms / 1000;
                                    let m2 = s / 60;
                                    let s2 = s % 60;
                                    right_lines.push(Line::from(format!(
                                        "setup: {m2}m {s2:02}s (waiting for first game)"
                                    )));
                                }
                            }
                        }
                        "gate" => {
                            let done = it.gate.games_completed;
                            let tot = it.gate.games_target.max(1);
                            right_lines.push(Line::from(format!("{done} / {tot} games")));
                            gauge = Some((done as f64 / tot as f64, String::new()));

                            // Gate diagnostics (if available; filled at end of gate, and for SPRT).
                            if let (Some(w), Some(l), Some(d)) =
                                (it.gate.wins, it.gate.losses, it.gate.draws)
                            {
                                let n = (w + l + d).max(1) as f64;
                                let wr = (w as f64 + 0.5 * d as f64) / n;
                                let ci = match (it.gate.win_ci95_low, it.gate.win_ci95_high) {
                                    (Some(lo), Some(hi)) => format!("  ci95=[{lo:.4},{hi:.4}]"),
                                    _ => String::new(),
                                };
                                right_lines.push(Line::from(format!(
                                    "W/D/L: {w}/{d}/{l}   win_rate={wr:.4}{ci}"
                                )));
                            }
                            if let Some(s) = it.gate.sprt.as_ref() {
                                let dec = s
                                    .decision
                                    .as_deref()
                                    .unwrap_or(if s.enabled { "continue" } else { "disabled" });
                                right_lines.push(Line::from(format!(
                                    "SPRT: {dec}  llr={:.3}  A={:.3}  B={:.3}",
                                    s.llr, s.bound_a, s.bound_b
                                )));
                                right_lines.push(Line::from(format!(
                                    "p0/p1={:.3}/{:.3}  min/max={}/{}  Δ={:.3}  α/β={:.3}/{:.3}",
                                    s.p0, s.p1, s.min_games, s.max_games, s.delta, s.alpha, s.beta
                                )));
                                if let Some(reason) = s.decision_reason.as_deref() {
                                    if !reason.is_empty() {
                                        right_lines.push(Line::from(format!("sprt_reason: {reason}")));
                                    }
                                }
                            }
                            if !right_lines.is_empty() {
                                right_lines.push(Line::from(""));
                            }

                            // Timing stats
                            if let Some(started) = it.gate.started_ts_ms {
                                if let Some(first) = it.gate.first_game_started_ts_ms {
                                    let setup_ms = first.saturating_sub(started);
                                    let s = setup_ms / 1000;
                                    let m2 = s / 60;
                                    let s2 = s % 60;
                                    right_lines.push(Line::from(format!("setup: {m2}m {s2:02}s")));
                                    let run_ms = now_ms.saturating_sub(first);
                                    let rs = run_ms / 1000;
                                    let rm = rs / 60;
                                    let rs2 = rs % 60;
                                    right_lines.push(Line::from(format!("running: {rm}m {rs2:02}s")));

                                    if done > 0 {
                                        let avg_ms = run_ms / done as u64;
                                        let remaining = tot.saturating_sub(done);
                                        let eta_ms = avg_ms * remaining as u64;
                                        let avg_s = avg_ms as f64 / 1000.0;
                                        let eta_s = eta_ms / 1000;
                                        let eta_m = eta_s / 60;
                                        let eta_s_rem = eta_s % 60;
                                        right_lines.push(Line::from(format!(
                                            "avg: {avg_s:.1}s/game  ETA: {eta_m}m {eta_s_rem}s"
                                        )));
                                    }
                                } else {
                                    let elapsed_ms = now_ms.saturating_sub(started);
                                    let s = elapsed_ms / 1000;
                                    let m2 = s / 60;
                                    let s2 = s % 60;
                                    right_lines.push(Line::from(format!(
                                        "setup: {m2}m {s2:02}s (waiting for first game)"
                                    )));
                                }
                            }
                        }
                        "train" => {
                            if let Some(tot) = it.train.steps_target {
                                let done = it.train.steps_completed.unwrap_or(0);
                                let tot = tot.max(1);
                                right_lines.push(Line::from(format!("{done} / {tot} steps")));
                                gauge = Some((done as f64 / tot as f64, String::new()));

                                // Timing stats
                                if let Some(started) = it.train.started_ts_ms {
                                    let elapsed_ms = now_ms.saturating_sub(started);
                                    if done > 0 {
                                        let avg_ms = elapsed_ms / done as u64;
                                        let remaining = tot.saturating_sub(done);
                                        let eta_ms = avg_ms * remaining as u64;
                                        let avg_s = avg_ms as f64 / 1000.0;
                                        let eta_s = eta_ms / 1000;
                                        let eta_m = eta_s / 60;
                                        let eta_s_rem = eta_s % 60;
                                        right_lines.push(Line::from(format!(
                                            "avg: {avg_s:.1}s/step  ETA: {eta_m}m {eta_s_rem}s"
                                        )));
                                    }
                                }
                            } else if let Some(done) = it.train.steps_completed {
                                right_lines.push(Line::from(format!("step: {done}")));
                            }

                            if let Some(v) = it.train.last_loss_total {
                                right_lines.push(Line::from(format!("loss_total: {v:.4}")));
                            }
                            if let Some(v) = it.train.last_loss_policy {
                                right_lines.push(Line::from(format!("loss_policy: {v:.4}")));
                            }
                            if let Some(v) = it.train.last_loss_value {
                                right_lines.push(Line::from(format!("loss_value: {v:.4}")));
                            }
                        }
                        _ => {}
                    }
                } else if phase == "idle" {
                    right_lines.push(Line::from("No iteration running."));
                } else if phase == "done" {
                    // Common case for extended runs: we intentionally do not copy the source run's
                    // per-iteration summaries, so the Performance table can be empty even though
                    // controller_iteration_idx > 0.
                    right_lines.push(Line::from(format!(
                        "Run complete. To keep going, set controller.total_iterations > {cur_idx} in Config and press 'g'."
                    )));
                } else {
                    right_lines.push(Line::from(format!(
                        "no iteration summary for idx={cur_idx}"
                    )));
                }
            } else if let Some(e) = &app.dashboard_err {
                right_lines.push(Line::from(format!("(unavailable: {e})")));
            } else {
                right_lines.push(Line::from("(no data)"));
            }

            let right_block = Block::default().title(phase_title).borders(Borders::ALL);
            if let Some((ratio, _label)) = gauge {
                let rows = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Min(5),
                        Constraint::Length(3),
                    ])
                    .split(cols[1]);
                let p = Paragraph::new(right_lines).block(right_block);
                f.render_widget(p, rows[0]);
                // Progress bar without label overlay - label is shown as text above
                let g = Gauge::default()
                    .block(Block::default().borders(Borders::ALL))
                    .ratio(ratio.clamp(0.0, 1.0))
                    .gauge_style(Style::default().fg(Color::Cyan));
                f.render_widget(g, rows[1]);
            } else {
                let p = Paragraph::new(right_lines).block(right_block);
                f.render_widget(p, cols[1]);
            }
            }
        }
        Screen::Search => {
            draw_search(f, app, chunks[0]);
        }
        Screen::System => {
            draw_system(f, app, chunks[0]);
        }
    }

    let help = Paragraph::new(app.status.clone())
        .block(Block::default().title("Commands").borders(Borders::ALL));
    f.render_widget(help, chunks[1]);
}

fn draw_dashboard_learning(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    let rid = app.active_run_id.as_deref().unwrap_or("<no run>");
    let title = Line::from(vec![
        Span::styled("Learning", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::raw(rid.to_string()),
    ]);

    // Layout: reduced sparkline area + full-width replay panel + existing table/details.
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Length(8),  // sparklines (reduced)
                Constraint::Length(12), // replay buffer (expanded)
                Constraint::Min(1),     // table + details
            ]
            .as_ref(),
        )
        .split(area);
    let spark_area = chunks[0];
    let replay_area = chunks[1];
    let bottom_area = chunks[2];

    // Sparklines (2 cols x 2 rows)
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(spark_area);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(top[0]);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(top[1]);

    let max_iter = app
        .dashboard_manifest
        .as_ref()
        .map(|m| m.iterations.len().max(1))
        .unwrap_or(1);

    fn carry_series(len: usize, mut get: impl FnMut(usize) -> Option<u64>, default: u64) -> Vec<u64> {
        let mut out = Vec::with_capacity(len);
        let mut last = default;
        for i in 0..len {
            if let Some(v) = get(i) {
                last = v;
            }
            out.push(last);
        }
        out
    }

    let stale_p95 = carry_series(
        max_iter,
        |i| {
            app.learn_by_iter
                .get(&(i as u32))
                .and_then(|x| x.age_wall_s_p95)
                .map(|s| (s.max(0.0) as u64).min(1_000_000))
        },
        0,
    );
    let top1_reroll_p95 = carry_series(
        max_iter,
        |i| {
            app.learn_by_iter
                .get(&(i as u32))
                .and_then(|x| x.pi_top1_p95_reroll.or(x.pi_top1_p95))
                .map(|v| (v.clamp(0.0, 1.0) * 1000.0) as u64)
        },
        0,
    );
    let ece = carry_series(
        max_iter,
        |i| {
            app.learn_by_iter
                .get(&(i as u32))
                .and_then(|x| x.ece)
                .map(|v| (v.clamp(0.0, 1.0) * 1000.0) as u64)
        },
        0,
    );
    let illegal_model = carry_series(
        max_iter,
        |i| {
            app.learn_by_iter
                .get(&(i as u32))
                .and_then(|x| x.p_illegal_mass_mean)
                .map(|v| (v.clamp(0.0, 1.0) * 1000.0) as u64)
        },
        0,
    );
    fn age_color(age_iters: u32) -> Color {
        // Coarse buckets, tuned for readability in terminals.
        match age_iters {
            0..=1 => Color::Green,
            2..=3 => Color::LightGreen,
            4..=7 => Color::Yellow,
            8..=15 => Color::LightRed,
            _ => Color::Red,
        }
    }

    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("staleness p95 (s)").borders(Borders::ALL))
            .data(&stale_p95),
        left[0],
    );
    f.render_widget(
        Sparkline::default()
            .block(
                Block::default()
                    .title("top1 p95 (target, reroll)")
                    .borders(Borders::ALL),
            )
            .data(&top1_reroll_p95),
        left[1],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("illegal mass (model)").borders(Borders::ALL))
            .data(&illegal_model),
        right[0],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("ECE").borders(Borders::ALL))
            .data(&ece),
        right[1],
    );

    // Replay buffer panel (full width): composition + config + derived training info.
    //
    // This must handle varying replay.capacity_shards. When there are too many shards to render
    // as one cell each, switch to a compact strip + histogram.
    let mut buf_lines: Vec<Line> = Vec::new();
    let buf_title = if let Some(max_i) = app.replay_max_snapshot_iter {
        format!("Replay buffer  snap≤{max_i}")
    } else {
        "Replay buffer".to_string()
    };
    if let Some(e) = &app.replay_err {
        buf_lines.push(Line::from(format!("(unavailable: {e})")));
    } else if app.replay_current.is_empty() {
        buf_lines.push(Line::from("(no shards yet)"));
    } else {
        let cur_iter = app.replay_max_snapshot_iter.unwrap_or(0);
        let inner_w = replay_area.width.saturating_sub(2) as usize;
        let inner_h = replay_area.height.saturating_sub(2) as usize;

        let shards = &app.replay_current;
        let n = shards.len();
        let cap = app
            .cfg
            .replay
            .capacity_shards
            .map(|x| x.to_string())
            .unwrap_or_else(|| "?".to_string());
        let total_samples_cur = app.replay_total_samples_current.unwrap_or(0);
        let avg_samples = if n > 0 {
            total_samples_cur as f64 / n as f64
        } else {
            0.0
        };

        // Training config + derived steps.
        let bs = app.cfg.training.batch_size.max(1) as u64;
        let epochs = app.cfg.training.epochs.max(1) as u64;
        let steps_mode = app.cfg.training.steps_per_iteration.is_some();
        let sample_mode = app.cfg.training.sample_mode.as_str();
        let steps_cfg = app.cfg.training.steps_per_iteration.unwrap_or(0) as u64;

        // Prefer total_samples from the active iteration's learn_summary; fall back to latest; then current buffer sum.
        let active_idx = app.dashboard_manifest.as_ref().map(|m| {
            let cur = m.controller_iteration_idx;
            if m.iterations.iter().any(|it| it.idx == cur) {
                cur
            } else {
                cur.saturating_sub(1)
            }
        });
        let total_samples_train = active_idx
            .and_then(|i| app.learn_by_iter.get(&i).and_then(|x| x.total_samples))
            .or_else(|| {
                app.learn_by_iter
                    .keys()
                    .max()
                    .and_then(|i| app.learn_by_iter.get(i).and_then(|x| x.total_samples))
            })
            .unwrap_or(total_samples_cur);
        let epoch_steps = (total_samples_train + bs - 1) / bs;
        let steps_derived = if steps_mode {
            steps_cfg
        } else {
            epochs.saturating_mul(epoch_steps)
        };

        buf_lines.push(Line::from(format!(
            "capacity_shards={cap} | shards_present={n} | samples≈{total_samples_cur} (avg≈{avg_samples:.0}/shard)"
        )));
        if steps_mode {
            buf_lines.push(Line::from(format!(
                "training: mode=steps  steps/iter={steps_cfg}  batch={bs}  sample_mode={sample_mode}"
            )));
        } else {
            buf_lines.push(Line::from(format!(
                "training: mode=epochs  epochs={epochs}  batch={bs}  sample_mode={sample_mode}"
            )));
        }
        buf_lines.push(Line::from(format!(
            "derived: total_samples≈{total_samples_train}  epoch_steps≈{epoch_steps}  steps_target≈{steps_derived}"
        )));

        // Composition histogram for CURRENT buffer: shards and samples by origin iter (top-K newest + old + unknown).
        let mut count_by_iter: HashMap<u32, u32> = HashMap::new();
        let mut samples_by_iter: HashMap<u32, u64> = HashMap::new();
        let mut unknown = 0u32;
        let mut unknown_samples = 0u64;
        for sh in shards {
            match sh.origin_iter {
                Some(o) => {
                    *count_by_iter.entry(o).or_insert(0) += 1;
                    if let Some(ns) = sh.num_samples {
                        *samples_by_iter.entry(o).or_insert(0) += ns;
                    }
                }
                None => {
                    unknown += 1;
                    unknown_samples = unknown_samples.saturating_add(sh.num_samples.unwrap_or(0));
                }
            }
        }
        let mut keys: Vec<u32> = count_by_iter.keys().copied().collect();
        keys.sort_unstable();
        let k = 10usize;
        let keep = keys.len().min(k);
        let newest: Vec<u32> = keys.into_iter().rev().take(keep).collect();
        let newest = {
            let mut v = newest;
            v.sort_unstable();
            v
        };
        let mut old_count = 0u32;
        let mut old_samples = 0u64;
        let mut old_min: Option<u32> = None;
        let mut old_max: Option<u32> = None;
        for (&it, &c) in &count_by_iter {
            if !newest.contains(&it) {
                old_count += c;
                old_samples = old_samples.saturating_add(samples_by_iter.get(&it).copied().unwrap_or(0));
                old_min = Some(match old_min {
                    Some(m) => m.min(it),
                    None => it,
                });
                old_max = Some(match old_max {
                    Some(m) => m.max(it),
                    None => it,
                });
            }
        }
        let mut hist1 = String::new();
        let mut hist2 = String::new();
        if old_count > 0 {
            let old_label = match (old_min, old_max) {
                (Some(a), Some(b)) if a == b => format!("{a}"),
                (Some(a), Some(b)) => format!("{a}-{b}"),
                _ => "old".to_string(),
            };
            hist1.push_str(&format!("{old_label}:{old_count}"));
            hist2.push_str(&format!("{old_label}:{old_samples}"));
        }
        for it in newest {
            let c = count_by_iter.get(&it).copied().unwrap_or(0);
            let s = samples_by_iter.get(&it).copied().unwrap_or(0);
            if !hist1.is_empty() {
                hist1.push_str("  ");
                hist2.push_str("  ");
            }
            hist1.push_str(&format!("{it}:{c}"));
            hist2.push_str(&format!("{it}:{s}"));
        }
        if unknown > 0 {
            if !hist1.is_empty() {
                hist1.push_str("  ");
                hist2.push_str("  ");
            }
            hist1.push_str(&format!("??:{unknown}"));
            hist2.push_str(&format!("??:{unknown_samples}"));
        }
        if hist1.len() > inner_w {
            hist1.truncate(inner_w.saturating_sub(1));
            hist1.push('…');
        }
        if hist2.len() > inner_w {
            hist2.truncate(inner_w.saturating_sub(1));
            hist2.push('…');
        }
        buf_lines.push(Line::from(Span::styled(
            "origin_shards (iter:count):",
            Style::default().fg(Color::DarkGray),
        )));
        buf_lines.push(Line::from(hist1));
        buf_lines.push(Line::from(Span::styled(
            "origin_samples (iter:samples):",
            Style::default().fg(Color::DarkGray),
        )));
        buf_lines.push(Line::from(hist2));

        // Age summary (in iterations) for current buffer.
        let mut ages: Vec<u32> = shards
            .iter()
            .filter_map(|sh| sh.origin_iter.map(|o| cur_iter.saturating_sub(o)))
            .collect();
        ages.sort_unstable();
        if !ages.is_empty() {
            let p50 = ages[ages.len() / 2];
            let idx95 = (ages.len().saturating_mul(95)) / 100;
            let p95 = ages[idx95.min(ages.len().saturating_sub(1))];
            buf_lines.push(Line::from(format!(
                "age(iters): p50={p50}  p95={p95}  (newer=greener)"
            )));
        }

        // Grid: show per-shard cells labeled as `i<iter>:<samples>` (colored by age).
        let header_lines = buf_lines.len();
        let remaining_h = inner_h.saturating_sub(header_lines).max(1);
        let cell_w = 10usize; // e.g. "i49:0575 "
        let cells_per_row = (inner_w / cell_w).max(1);
        let max_rows = remaining_h;
        let can_show_all = n <= cells_per_row.saturating_mul(max_rows);

        if can_show_all {
            let mut idx = 0usize;
            for _row in 0..max_rows {
                if idx >= n {
                    break;
                }
                let end = (idx + cells_per_row).min(n);
                let mut spans: Vec<Span> = Vec::new();
                for sh in &shards[idx..end] {
                    let cell = match (sh.origin_iter, sh.num_samples) {
                        (Some(o), Some(ns)) => format!("i{o}:{ns}"),
                        (Some(o), None) => format!("i{o}:?"),
                        (None, Some(ns)) => format!("i??:{ns}"),
                        (None, None) => "i??:?".to_string(),
                    };
                    let age = sh.origin_iter.map(|o| cur_iter.saturating_sub(o)).unwrap_or(1_000_000);
                    let mut s = cell;
                    // Pad/truncate to fixed width-1 then add a spacer.
                    let pad_w = cell_w.saturating_sub(1);
                    if s.len() > pad_w {
                        s.truncate(pad_w.saturating_sub(1));
                        s.push('…');
                    }
                    spans.push(Span::styled(
                        format!("{s:<pad_w$} ", pad_w = pad_w),
                        Style::default().fg(age_color(age)).add_modifier(Modifier::BOLD),
                    ));
                }
                buf_lines.push(Line::from(spans));
                idx = end;
            }
        } else {
            // Compact fallback: downsampled strip (color by age) + rely on histogram above.
            let strip_w = inner_w.saturating_sub(6).max(1);
            let mut spans: Vec<Span> = Vec::new();
            spans.push(Span::styled("buf: ", Style::default().fg(Color::DarkGray)));
            for i in 0..strip_w {
                let idx = (i as u64 * n as u64 / strip_w as u64) as usize;
                let idx = idx.min(n.saturating_sub(1));
                let sh = &shards[idx];
                let age = sh.origin_iter.map(|o| cur_iter.saturating_sub(o)).unwrap_or(1_000_000);
                spans.push(Span::styled("■", Style::default().fg(age_color(age))));
            }
            buf_lines.push(Line::from(spans));
            buf_lines.push(Line::from(format!(
                "grid omitted ({n} shards); see origin histogram above"
            )));
        }
    }
    let p =
        Paragraph::new(buf_lines).block(Block::default().title(buf_title).borders(Borders::ALL));
    f.render_widget(p, replay_area);

    // Bottom: table + details.
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)].as_ref())
        .split(bottom_area);
    let table_area = bottom[0];
    let detail_area = bottom[1];

    // Table (left)
    let mut lines: Vec<Line> = Vec::new();
    const W_IDX: usize = 4;
    const W_DEC: usize = 8;
    const W_WR: usize = 7;
    const W_STALE: usize = 7;
    const W_PI: usize = 5;
    const W_TOP1: usize = 5;
    const W_EFFA: usize = 5;
    const W_KL: usize = 5;
    const W_OVR: usize = 5;
    const W_VMEAN: usize = 5;
    const W_ECE: usize = 5;
    const W_ILL: usize = 5;

    lines.push(Line::from(Span::styled(
        format!(
            "  {idx:>W_IDX$}  {dec:<W_DEC$}  {wr:>W_WR$}  {stale:>W_STALE$}  {pi:>W_PI$}  {top1:>W_TOP1$}  {effa:>W_EFFA$}  {kl:>W_KL$}  {ovr:>W_OVR$}  {vmean:>W_VMEAN$}  {ece:>W_ECE$}  {ill:>W_ILL$}",
            idx = "Iter",
            dec = "Decision",
            wr = "WinRate",
            stale = "Stale95",
            pi = "PiEnt",
            top1 = "Top1%",
            effa = "EffA",
            kl = "KL",
            ovr = "Ovr%",
            vmean = "Vmean",
            ece = "ECE",
            ill = "Ill%",
        ),
        Style::default().fg(Color::DarkGray),
    )));

    match (&app.dashboard_manifest, &app.dashboard_err) {
        (Some(m), _) => {
            let cur_idx = m.controller_iteration_idx;
            let active_idx = if m.iterations.iter().any(|it| it.idx == cur_idx) {
                cur_idx
            } else {
                cur_idx.saturating_sub(1)
            };
            let mut rows: Vec<Line> = Vec::new();
            fn fmt_opt_f64(v: Option<f64>, width: usize, prec: usize) -> String {
                match v {
                    Some(x) if x.is_finite() => {
                        let s = format!("{x:.prec$}", prec = prec);
                        format!("{s:>width$}", width = width)
                    }
                    _ => format!("{:>width$}", "-", width = width),
                }
            }
            fn fmt_opt_pct(v: Option<f64>, width: usize) -> String {
                match v {
                    Some(x) if x.is_finite() => {
                        let p = (x.clamp(0.0, 1.0) * 100.0).round() as i64;
                        let s = format!("{p}%");
                        format!("{s:>width$}", width = width)
                    }
                    _ => format!("{:>width$}", "-", width = width),
                }
            }
            fn fmt_opt_u0(v: Option<f64>, width: usize) -> String {
                match v {
                    Some(x) if x.is_finite() => {
                        let s = format!("{:.0}", x);
                        format!("{s:>width$}", width = width)
                    }
                    _ => format!("{:>width$}", "-", width = width),
                }
            }
            for it in &m.iterations {
                // ASCII marker to avoid wide-char rendering differences across terminals.
                let marker = if it.idx == active_idx { ">" } else { " " };
                let promo = it
                    .promoted
                    .map(|p| if p { "promote" } else { "reject" })
                    .unwrap_or("-");
                let wr = fmt_opt_f64(it.gate.win_rate, W_WR, 3);

                let ls = app.learn_by_iter.get(&it.idx);
                let ss = app.selfplay_by_iter.get(&it.idx);
                let stale95 = fmt_opt_u0(ls.and_then(|x| x.age_wall_s_p95), W_STALE);
                let pi_ent = fmt_opt_f64(ls.and_then(|x| x.pi_entropy_mean), W_PI, 2);
                let top1 = fmt_opt_pct(ls.and_then(|x| x.pi_top1_p95), W_TOP1);
                let effa = fmt_opt_f64(ls.and_then(|x| x.pi_eff_actions_p50), W_EFFA, 1);
                let kl = fmt_opt_f64(ls.and_then(|x| x.pi_kl_mean), W_KL, 2);
                let ovr = fmt_opt_pct(ss.and_then(|x| x.prior_argmax_overturn_rate), W_OVR);
                let vmean = fmt_opt_f64(ls.and_then(|x| x.v_pred_mean), W_VMEAN, 2);
                let ece = fmt_opt_f64(ls.and_then(|x| x.ece), W_ECE, 3);
                let ill = fmt_opt_pct(ls.and_then(|x| x.p_illegal_mass_mean), W_ILL);

                // Light alerting: highlight rows that match common failure signatures.
                let mut warn_level: u8 = 0;
                if let Some(v) = ls.and_then(|x| x.ece) {
                    if v > 0.20 {
                        warn_level = warn_level.max(2);
                    } else if v > 0.10 {
                        warn_level = warn_level.max(1);
                    }
                }
                if let Some(v) = ls.and_then(|x| x.pi_kl_mean) {
                    if v > 2.0 {
                        warn_level = warn_level.max(2);
                    } else if v > 1.0 {
                        warn_level = warn_level.max(1);
                    }
                }
                if let Some(v) = ls.and_then(|x| x.pi_entropy_mean) {
                    if v < 0.30 {
                        warn_level = warn_level.max(1);
                    }
                }
                if let Some(v) = ss.and_then(|x| x.fallbacks_rate) {
                    if v > 0.10 {
                        warn_level = warn_level.max(2);
                    } else if v > 0.05 {
                        warn_level = warn_level.max(1);
                    }
                }
                if let Some(v) = ss.and_then(|x| x.prior_argmax_overturn_rate) {
                    if v < 0.02 {
                        warn_level = warn_level.max(1);
                    }
                }

                let mut row_style = if it.idx == active_idx {
                    Style::default().fg(Color::Cyan)
                } else {
                    Style::default()
                };
                if warn_level >= 2 {
                    row_style = row_style.fg(Color::Red).add_modifier(Modifier::BOLD);
                } else if warn_level == 1 {
                    row_style = row_style.fg(Color::Yellow);
                }
                rows.push(Line::from(Span::styled(
                    format!(
                        "{marker} {idx:>W_IDX$}  {dec:<W_DEC$}  {wr}  {stale:>W_STALE$}  {pi}  {top1}  {effa}  {kl}  {ovr}  {vmean}  {ece}  {ill}",
                        idx = it.idx,
                        dec = promo,
                        stale = stale95,
                        pi = pi_ent,
                        top1 = top1,
                        effa = effa,
                        kl = kl,
                        ovr = ovr,
                        vmean = vmean,
                        ece = ece,
                        ill = ill,
                    ),
                    row_style,
                )));
            }

            // Determine scroll window for rows.
            let view_h = table_area.height.saturating_sub(2) as usize;
            let max_start = rows.len().saturating_sub(view_h.saturating_sub(1));
            let start = if app.dashboard_iter_follow {
                max_start
            } else {
                app.dashboard_iter_scroll.min(max_start)
            };
            for line in rows.into_iter().skip(start).take(view_h.saturating_sub(1)) {
                lines.push(line);
            }

            // Details panel (right): show metrics for active_idx (best-effort).
            let mut detail: Vec<Line> = Vec::new();
            detail.push(Line::from(Span::styled(
                format!("iter {active_idx}"),
                Style::default().add_modifier(Modifier::BOLD),
            )));
            if let Some(ss) = app.selfplay_by_iter.get(&active_idx) {
                if let Some(v) = ss.moves_s_mean {
                    detail.push(Line::from(format!("selfplay: {v:.0} moves/s")));
                }
                if let Some(v) = ss.pi_max_p_p95 {
                    detail.push(Line::from(format!(
                        "max_p95_visit: {:.1}%",
                        v.clamp(0.0, 1.0) * 100.0
                    )));
                }
                if let Some(v) = ss.prior_argmax_overturn_rate {
                    detail.push(Line::from(format!("overturn: {:.0}%", v * 100.0)));
                }
                if let Some(v) = ss.prior_kl_mean {
                    detail.push(Line::from(format!("prior_KL: {v:.3}")));
                }
                if let Some(v) = ss.fallbacks_rate {
                    detail.push(Line::from(format!("fallbacks: {:.1}%", v * 100.0)));
                }
                if let Some(v) = ss.pending_collisions_per_move {
                    detail.push(Line::from(format!("pending_coll: {v:.3}/move")));
                }
            }
            if let Some(ls) = app.learn_by_iter.get(&active_idx) {
                if let Some(v) = ls.samples_s_mean {
                    detail.push(Line::from(format!("train: {v:.0} samples/s")));
                }
                if let Some(v) = ls.pi_entropy_mean {
                    detail.push(Line::from(format!("pi_ent_tgt: {v:.3}")));
                }
                if let Some(v) = ls.pi_top1_p95_reroll.or(ls.pi_top1_p95) {
                    detail.push(Line::from(format!(
                        "top1_p95_reroll_tgt: {:.1}%",
                        v.clamp(0.0, 1.0) * 100.0
                    )));
                }
                if let Some(v) = ls.pi_top1_p95_mark.or(ls.pi_top1_p95) {
                    detail.push(Line::from(format!(
                        "top1_p95_mark_tgt: {:.1}%",
                        v.clamp(0.0, 1.0) * 100.0
                    )));
                }
                if let Some(v) = ls.pi_entropy_mean_reroll {
                    detail.push(Line::from(format!("pi_ent_reroll_tgt: {v:.3}")));
                }
                if let Some(v) = ls.pi_entropy_mean_mark {
                    detail.push(Line::from(format!("pi_ent_mark_tgt: {v:.3}")));
                }
                if let Some(v) = ls.pi_eff_actions_p50 {
                    detail.push(Line::from(format!("effA_p50_tgt: {v:.1}")));
                }
                if let Some(v) = ls.pi_eff_actions_p50_reroll {
                    detail.push(Line::from(format!("effA_p50_reroll_tgt: {v:.1}")));
                }
                if let Some(v) = ls.pi_eff_actions_p50_mark {
                    detail.push(Line::from(format!("effA_p50_mark_tgt: {v:.1}")));
                }
                if let Some(v) = ls.pi_entropy_norm_p50 {
                    detail.push(Line::from(format!("Hnorm_p50_tgt: {v:.3}")));
                }
                if let Some(v) = ls.n_legal_p50 {
                    detail.push(Line::from(format!("n_legal_p50: {v:.0}")));
                }
                if let Some(v) = ls.pi_illegal_mass_mean {
                    detail.push(Line::from(format!(
                        "pi_illegal_tgt: {:.2}%",
                        v.clamp(0.0, 1.0) * 100.0
                    )));
                }
                if let Some(v) = ls.p_illegal_mass_mean {
                    detail.push(Line::from(format!(
                        "p_illegal_model: {:.2}%",
                        v.clamp(0.0, 1.0) * 100.0
                    )));
                }
                if let Some(v) = ls.pi_model_entropy_mean {
                    detail.push(Line::from(format!("pi_ent_model: {v:.3}")));
                }
                if let Some(v) = ls.pi_kl_mean {
                    detail.push(Line::from(format!("pi_KL: {v:.3}")));
                }
                if let Some(v) = ls.pi_model_entropy_legal_mean {
                    detail.push(Line::from(format!("pi_ent_model_legal: {v:.3}")));
                }
                if let Some(v) = ls.pi_kl_legal_mean {
                    detail.push(Line::from(format!("pi_KL_legal: {v:.3}")));
                }
                if let Some(v) = ls.p_top1_legal_p95 {
                    detail.push(Line::from(format!(
                        "p_top1_p95_legal: {:.1}%",
                        v.clamp(0.0, 1.0) * 100.0
                    )));
                }
                if let Some(v) = ls.pi_entropy_gap_mean {
                    detail.push(Line::from(format!("ent_gap: {v:.3}")));
                }
                if let Some(v) = ls.v_pred_mean {
                    detail.push(Line::from(format!("v_mean: {v:.3}")));
                }
                if let Some(v) = ls.ece {
                    detail.push(Line::from(format!("ECE: {v:.3}")));
                }

                // Alerts summary for current iteration (best-effort).
                let mut alerts: Vec<String> = Vec::new();
                if let Some(v) = ls.pi_kl_mean {
                    if v > 2.0 {
                        alerts.push(format!("KL spike ({v:.2})"));
                    }
                }
                if let Some(v) = ls.pi_entropy_mean {
                    if v < 0.30 {
                        alerts.push(format!("entropy low ({v:.2})"));
                    }
                }
                if let Some(v) = ls.ece {
                    if v > 0.20 {
                        alerts.push(format!("ECE high ({v:.2})"));
                    }
                }
                if let Some(ss) = app.selfplay_by_iter.get(&active_idx) {
                    if let Some(v) = ss.fallbacks_rate {
                        if v > 0.10 {
                            alerts.push(format!("fallbacks high ({:.0}%)", v * 100.0));
                        }
                    }
                    if let Some(v) = ss.prior_argmax_overturn_rate {
                        if v < 0.02 {
                            alerts.push("search not improving (overturn <2%)".to_string());
                        }
                    }
                }
                if !alerts.is_empty() {
                    detail.push(Line::from(""));
                    detail.push(Line::from(Span::styled(
                        "alerts:",
                        Style::default().fg(Color::DarkGray),
                    )));
                    for a in alerts {
                        detail.push(Line::from(Span::styled(a, Style::default().fg(Color::Yellow))));
                    }
                }

                // Tiny reliability diagram.
                if let Some(bins) = &ls.calibration_bins {
                    detail.push(Line::from(""));
                    detail.push(Line::from(Span::styled(
                        "calibration (pred vs z):",
                        Style::default().fg(Color::DarkGray),
                    )));
                    for (i, b) in bins.iter().enumerate() {
                        if b.count == 0 {
                            continue;
                        }
                        let mp = b.mean_pred.unwrap_or(0.0);
                        let mz = b.mean_z.unwrap_or(0.0);
                        let err = (mp - mz).abs();
                        let bar_w = 10usize;
                        let filled = (err.clamp(0.0, 1.0) * (bar_w as f64)).round() as usize;
                        let mut bar = String::new();
                        for _ in 0..filled.min(bar_w) {
                            bar.push('#');
                        }
                        for _ in filled.min(bar_w)..bar_w {
                            bar.push(' ');
                        }
                        detail.push(Line::from(format!(
                            "b{i}: n={:>4} pred={:+.2} z={:+.2} |{bar}|",
                            b.count, mp, mz
                        )));
                    }
                }
            }

            let p_details =
                Paragraph::new(detail).block(Block::default().title("Details").borders(Borders::ALL));
            f.render_widget(p_details, detail_area);
        }
        (_, Some(e)) => {
            lines.push(Line::from(format!(" (unavailable: {e})")));
            let p_details = Paragraph::new(vec![Line::from("(no data)")])
                .block(Block::default().title("Details").borders(Borders::ALL));
            f.render_widget(p_details, detail_area);
        }
        _ => {
            lines.push(Line::from(" (no run selected)"));
            let p_details = Paragraph::new(vec![Line::from("(no run selected)")])
                .block(Block::default().title("Details").borders(Borders::ALL));
            f.render_widget(p_details, detail_area);
        }
    }

    let p = Paragraph::new(lines).block(Block::default().title(title).borders(Borders::ALL));
    f.render_widget(p, table_area);
}

fn render_config_lines(app: &App, view_height: usize) -> Vec<Line<'static>> {
    // Build rows with optional field ids.
    #[derive(Clone)]
    enum Row {
        Header(&'static str),
        SubHeader(&'static str),
        Field(FieldId),
        Spacer,
        Error(String),
    }

    let mut rows: Vec<Row> = Vec::new();
    fn is_visible(cfg: &yz_core::Config, f: FieldId) -> bool {
        // Hide step-only fields when not applicable.
        if matches!(f, FieldId::MctsTempT1 | FieldId::MctsTempCutoffTurn)
            && matches!(cfg.mcts.temperature_schedule, TemperatureSchedule::Constant { .. })
        {
            return false;
        }
        // Hide epochs field when in steps mode.
        if f == FieldId::TrainingEpochs && cfg.training.steps_per_iteration.is_some() {
            return false;
        }
        // Hide steps_per_iteration field when in epochs mode.
        if f == FieldId::TrainingStepsPerIteration && cfg.training.steps_per_iteration.is_none() {
            return false;
        }
        // Continuous training: reset_optimizer is ignored, so hide it.
        if f == FieldId::TrainingResetOptimizer && cfg.training.continuous_candidate_training {
            return false;
        }
        // SPRT: show either fixed-games or sprt knobs.
        if f == FieldId::GatingGames && cfg.gating.katago.sprt {
            return false;
        }
        if matches!(
            f,
            FieldId::GatingKatagoSprtMinGames
                | FieldId::GatingKatagoSprtMaxGames
                | FieldId::GatingKatagoSprtAlpha
                | FieldId::GatingKatagoSprtBeta
                | FieldId::GatingKatagoSprtDelta
        ) && !cfg.gating.katago.sprt
        {
            return false;
        }
        true
    }

    // Top-level sections, with Pipeline subheaders.
    for sec in Section::ALL {
        rows.push(Row::Header(sec.title()));
        match sec {
            Section::Pipeline => {
                // Controller-level pipeline control.
                rows.push(Row::Field(FieldId::ControllerTotalIterations));
                rows.push(Row::Spacer);
                rows.push(Row::SubHeader("selfplay"));
                for f in [
                    FieldId::SelfplayGamesPerIteration,
                    FieldId::SelfplayWorkers,
                    FieldId::SelfplayThreadsPerWorker,
                ] {
                    if is_visible(&app.cfg, f) {
                        rows.push(Row::Field(f));
                    }
                }
                rows.push(Row::Spacer);
                rows.push(Row::SubHeader("training"));
                for f in [
                    FieldId::TrainingMode,
                    FieldId::TrainingBatchSize,
                    FieldId::TrainingLearningRate,
                    FieldId::TrainingContinuousCandidateTraining,
                    FieldId::TrainingResetOptimizer,
                    FieldId::TrainingOptimizer,
                    FieldId::TrainingEpochs,
                    FieldId::TrainingWeightDecay,
                    FieldId::TrainingStepsPerIteration,
                ] {
                    if is_visible(&app.cfg, f) {
                        rows.push(Row::Field(f));
                    }
                }
                rows.push(Row::Spacer);
                rows.push(Row::SubHeader("gating"));
                for f in [
                    FieldId::GatingKatagoSprt,
                    FieldId::GatingGames,
                    FieldId::GatingKatagoSprtMinGames,
                    FieldId::GatingKatagoSprtMaxGames,
                    FieldId::GatingKatagoSprtAlpha,
                    FieldId::GatingKatagoSprtBeta,
                    FieldId::GatingKatagoSprtDelta,
                    FieldId::GatingSeedSetId,
                    FieldId::GatingSeed,
                    FieldId::GatingPairedSeedSwap,
                    FieldId::GatingDeterministicChance,
                    FieldId::GatingWinRateThreshold,
                ] {
                    if is_visible(&app.cfg, f) {
                        rows.push(Row::Field(f));
                    }
                }
            }
            _ => {
                for f in ALL_FIELDS.iter().copied().filter(|f| f.section() == sec) {
                    if is_visible(&app.cfg, f) {
                        rows.push(Row::Field(f));
                    }
                }
            }
        }
        rows.push(Row::Spacer);
    }
    if let Some(e) = &app.form.last_validation_error {
        rows.push(Row::Error(format!("ERROR: {e}")));
    }

    // Find selected row index.
    let selected_field = ALL_FIELDS
        .get(app.form.selected_idx)
        .copied()
        .unwrap_or(FieldId::InferBind);
    let mut selected_row = 0usize;
    for (i, r) in rows.iter().enumerate() {
        if matches!(r, Row::Field(f) if *f == selected_field) {
            selected_row = i;
            break;
        }
    }

    // Scroll to keep selection visible.
    // Ensure the selected row is always within the visible window.
    let height = view_height.max(1);
    let start = if selected_row < height {
        0
    } else {
        // Keep selected row in the lower third of the view for better context
        selected_row.saturating_sub(height * 2 / 3)
    };
    let end = (start + height).min(rows.len());
    // Adjust start if we're near the end to fill the view
    let start = if end == rows.len() && rows.len() > height {
        rows.len().saturating_sub(height)
    } else {
        start
    };
    let slice = &rows[start..end];

    let mut out: Vec<Line<'static>> = Vec::new();
    for r in slice {
        match r {
            Row::Header(t) => {
                // Section headers in dimmed gray with bold
                out.push(Line::from(vec![Span::styled(
                    format!("─── {t} ───"),
                    Style::default()
                        .fg(Color::DarkGray)
                        .add_modifier(Modifier::BOLD),
                )]));
            }
            Row::SubHeader(t) => {
                out.push(Line::from(vec![Span::styled(
                    format!("  {t}"),
                    Style::default().fg(Color::DarkGray),
                )]));
            }
            Row::Spacer => out.push(Line::from("")),
            Row::Error(e) => out.push(Line::from(vec![Span::styled(
                e.clone(),
                Style::default().fg(Color::Red),
            )])),
            Row::Field(f) => {
                let is_sel = *f == selected_field;
                let label = f.label();
                let v = if app.form.edit_mode == EditMode::Editing && is_sel {
                    app.form.input_buf.clone()
                } else {
                    field_value_string(&app.cfg, *f)
                };
                let prefix = if is_sel { "▸ " } else { "  " };
                if is_sel {
                    // Selected field: cyan with bold
                    out.push(Line::from(vec![
                        Span::styled(
                            format!("{prefix}{label}"),
                            Style::default()
                                .fg(Color::Cyan)
                                .add_modifier(Modifier::BOLD),
                        ),
                        Span::styled(
                            " = ",
                            Style::default().fg(Color::DarkGray),
                        ),
                        Span::styled(
                            v,
                            Style::default()
                                .fg(Color::White)
                                .add_modifier(Modifier::BOLD),
                        ),
                    ]));
                } else {
                    // Non-selected field: dimmer styling
                    out.push(Line::from(vec![
                        Span::raw(format!("{prefix}{label}")),
                        Span::styled(" = ", Style::default().fg(Color::DarkGray)),
                        Span::styled(v, Style::default().fg(Color::Gray)),
                    ]));
                }
            }
        }
    }
    out
}

// -------------------------
// System / Inference screen helpers
// -------------------------

fn push_series(v: &mut Vec<u64>, x: u64, max: usize) {
    v.push(x);
    if v.len() > max {
        let drain = v.len() - max;
        v.drain(0..drain);
    }
}

#[derive(Debug, Default, Clone)]
struct ServerLiveAgg {
    queue_depth: Option<u64>,
    requests_total: Option<f64>,
    batches_total: Option<f64>,
    batch_size_p50: Option<f64>,
    batch_size_p95: Option<f64>,
    queue_wait_us_p95: Option<f64>,
    forward_ms_p95: Option<f64>,
    underfill_frac_mean: Option<f64>,
    full_frac_mean: Option<f64>,
    flush_full_total: Option<f64>,
    flush_deadline_total: Option<f64>,
}

fn quantile_from_cumulative(mut buckets: Vec<(u64, u64)>, total: u64, q: f64) -> Option<u64> {
    if total == 0 || buckets.is_empty() {
        return None;
    }
    buckets.sort_by_key(|(le, _)| *le);
    let qq = q.clamp(0.0, 1.0);
    let target = (qq * (total as f64)).ceil() as u64;
    for (le, cum) in buckets {
        if cum >= target {
            return Some(le);
        }
    }
    None
}

fn parse_prom_sample(line: &str) -> Option<(&str, Option<&str>, f64)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (lhs, rhs) = line.split_once(' ')?;
    let value = rhs.trim().parse::<f64>().ok()?;
    if let Some((name, rest)) = lhs.split_once('{') {
        let labels = rest.strip_suffix('}')?;
        Some((name, Some(labels), value))
    } else {
        Some((lhs, None, value))
    }
}

fn parse_label<'a>(labels: &'a str, key: &str) -> Option<&'a str> {
    for part in labels.split(',') {
        let (k, v) = part.split_once('=')?;
        if k.trim() != key {
            continue;
        }
        let vs = v.trim();
        return vs.strip_prefix('"')?.strip_suffix('"');
    }
    None
}

fn parse_infer_server_metrics_text(text: &str) -> ServerLiveAgg {
    let mut out = ServerLiveAgg::default();

    let mut batch_size_cum: Vec<(u64, u64)> = Vec::new();
    let mut batch_size_total: u64 = 0;
    let mut qwait_cum: Vec<(u64, u64)> = Vec::new();
    let mut qwait_total: u64 = 0;
    let mut forward_cum_us: Vec<(u64, u64)> = Vec::new();
    let mut forward_total: u64 = 0;

    let mut underfill_sum = 0.0f64;
    let mut underfill_n = 0u64;
    let mut full_sum = 0.0f64;
    let mut full_n = 0u64;

    let mut flush_full_total = 0.0f64;
    let mut flush_deadline_total = 0.0f64;

    for line in text.lines() {
        let Some((name, labels_opt, value)) = parse_prom_sample(line) else {
            continue;
        };
        match name {
            "yatzy_infer_queue_depth" => out.queue_depth = Some(value.max(0.0) as u64),
            "yatzy_infer_requests_total" => out.requests_total = Some(value),
            "yatzy_infer_batches_total" => out.batches_total = Some(value),
            "yatzy_infer_batch_underfill_frac" => {
                underfill_sum += value;
                underfill_n += 1;
            }
            "yatzy_infer_batch_full_frac" => {
                full_sum += value;
                full_n += 1;
            }
            "yatzy_infer_flush_reason_total" => {
                let Some(labels) = labels_opt else { continue };
                let Some(reason) = parse_label(labels, "reason") else { continue };
                match reason {
                    "full" => flush_full_total += value,
                    "deadline" => flush_deadline_total += value,
                    _ => {}
                }
            }
            "yatzy_infer_batch_size_bucket" => {
                let Some(labels) = labels_opt else { continue };
                let Some(le) = parse_label(labels, "le") else { continue };
                if le == "+Inf" {
                    batch_size_total = batch_size_total.max(value.max(0.0) as u64);
                } else if let Ok(x) = le.parse::<f64>() {
                    batch_size_cum.push((x.max(0.0) as u64, value.max(0.0) as u64));
                }
            }
            "yatzy_infer_batch_queue_wait_us_bucket" => {
                let Some(labels) = labels_opt else { continue };
                let Some(le) = parse_label(labels, "le") else { continue };
                if le == "+Inf" {
                    qwait_total = qwait_total.max(value.max(0.0) as u64);
                } else if let Ok(x) = le.parse::<f64>() {
                    qwait_cum.push((x.max(0.0) as u64, value.max(0.0) as u64));
                }
            }
            "yatzy_infer_batch_forward_ms_bucket" => {
                let Some(labels) = labels_opt else { continue };
                let Some(le) = parse_label(labels, "le") else { continue };
                if le == "+Inf" {
                    forward_total = forward_total.max(value.max(0.0) as u64);
                } else if let Ok(x) = le.parse::<f64>() {
                    // Store boundary in us for easy ordering, convert back later.
                    let us = (x.max(0.0) * 1000.0).round() as u64;
                    forward_cum_us.push((us, value.max(0.0) as u64));
                }
            }
            _ => {}
        }
    }

    out.batch_size_p50 = quantile_from_cumulative(batch_size_cum.clone(), batch_size_total, 0.50).map(|x| x as f64);
    out.batch_size_p95 = quantile_from_cumulative(batch_size_cum, batch_size_total, 0.95).map(|x| x as f64);
    out.queue_wait_us_p95 = quantile_from_cumulative(qwait_cum, qwait_total, 0.95).map(|x| x as f64);
    out.forward_ms_p95 = quantile_from_cumulative(forward_cum_us, forward_total, 0.95).map(|x| (x as f64) / 1000.0);

    out.underfill_frac_mean = if underfill_n > 0 {
        Some(underfill_sum / (underfill_n as f64))
    } else {
        None
    };
    out.full_frac_mean = if full_n > 0 {
        Some(full_sum / (full_n as f64))
    } else {
        None
    };
    out.flush_full_total = Some(flush_full_total);
    out.flush_deadline_total = Some(flush_deadline_total);
    out
}

fn fetch_infer_metrics_text(agent: &ureq::Agent, metrics_bind: &str) -> Result<String, String> {
    let url = format!("http://{}/metrics", metrics_bind);
    let resp = agent
        .get(&url)
        .call()
        .map_err(|e| format!("live /metrics failed: {e}"))?;
    if resp.status() != 200 {
        return Err(format!("live /metrics status {}", resp.status()));
    }
    resp.into_string()
        .map_err(|e| format!("live /metrics read failed: {e}"))
}

#[derive(Debug, Default, Clone)]
struct WorkerLiveAgg {
    inflight_sum: Option<u64>,
    inflight_max: Option<u64>,
    rtt_p95_us_med: Option<u64>,
    would_block_frac: Option<f64>,
}

fn read_selfplay_worker_progress_live(
    logs_workers_dir: &Path,
    prev_steps_sum: &mut Option<u64>,
    prev_wb_sum: &mut Option<u64>,
) -> WorkerLiveAgg {
    #[derive(serde::Deserialize)]
    struct P {
        #[serde(default)]
        infer_inflight: u64,
        #[serde(default)]
        infer_latency_p95_us: u64,
        #[serde(default)]
        sched_steps: u64,
        #[serde(default)]
        sched_would_block: u64,
    }
    let mut out = WorkerLiveAgg::default();
    let mut rtts: Vec<u64> = Vec::new();
    let mut steps_sum = 0u64;
    let mut wb_sum = 0u64;

    if let Ok(rd) = std::fs::read_dir(logs_workers_dir) {
        let mut inflight_sum = 0u64;
        let mut inflight_max = 0u64;
        for e in rd.flatten() {
            let p = e.path();
            if !p.is_dir() {
                continue;
            }
            let f = p.join("progress.json");
            let Ok(bytes) = std::fs::read(&f) else { continue };
            let Ok(pp) = serde_json::from_slice::<P>(&bytes) else { continue };
            inflight_sum = inflight_sum.saturating_add(pp.infer_inflight);
            inflight_max = inflight_max.max(pp.infer_inflight);
            rtts.push(pp.infer_latency_p95_us);
            steps_sum = steps_sum.saturating_add(pp.sched_steps);
            wb_sum = wb_sum.saturating_add(pp.sched_would_block);
        }
        if !rtts.is_empty() {
            out.inflight_sum = Some(inflight_sum);
            out.inflight_max = Some(inflight_max);
        }
    }

    if !rtts.is_empty() {
        rtts.sort();
        out.rtt_p95_us_med = Some(rtts[rtts.len() / 2]);
    }

    if let (Some(prev_s), Some(prev_w)) = (*prev_steps_sum, *prev_wb_sum) {
        let ds = steps_sum.saturating_sub(prev_s);
        let dw = wb_sum.saturating_sub(prev_w);
        let denom = (ds + dw) as f64;
        if denom > 0.0 {
            out.would_block_frac = Some((dw as f64) / denom);
        }
    }
    *prev_steps_sum = Some(steps_sum);
    *prev_wb_sum = Some(wb_sum);

    out
}

fn derive_rates(
    now: Instant,
    prev_ts: &mut Option<Instant>,
    prev_req_total: &mut Option<f64>,
    prev_batches_total: &mut Option<f64>,
    prev_flush_full_total: &mut Option<f64>,
    prev_flush_deadline_total: &mut Option<f64>,
    req_total: Option<f64>,
    batches_total: Option<f64>,
    flush_full_total: Option<f64>,
    flush_deadline_total: Option<f64>,
) -> (Option<f64>, Option<f64>, Option<f64>, Option<f64>, Option<f64>) {
    let mut rps = None;
    let mut bps = None;
    let mut mean = None;
    let mut full_s = None;
    let mut dead_s = None;

    if let Some(prev) = *prev_ts {
        let dt = (now - prev).as_secs_f64().max(1e-6);
        if let (Some(prev_r), Some(cur_r)) = (*prev_req_total, req_total) {
            let dr = (cur_r - prev_r).max(0.0);
            rps = Some(dr / dt);
        }
        if let (Some(prev_b), Some(cur_b)) = (*prev_batches_total, batches_total) {
            let db = (cur_b - prev_b).max(0.0);
            bps = Some(db / dt);
            if db > 0.0 {
                if let (Some(prev_r), Some(cur_r)) = (*prev_req_total, req_total) {
                    let dr = (cur_r - prev_r).max(0.0);
                    mean = Some(dr / db);
                }
            }
        }
        if let (Some(prev_f), Some(cur_f)) = (*prev_flush_full_total, flush_full_total) {
            full_s = Some((cur_f - prev_f).max(0.0) / dt);
        }
        if let (Some(prev_f), Some(cur_f)) = (*prev_flush_deadline_total, flush_deadline_total) {
            dead_s = Some((cur_f - prev_f).max(0.0) / dt);
        }
    }
    *prev_ts = Some(now);
    *prev_req_total = req_total;
    *prev_batches_total = batches_total;
    *prev_flush_full_total = flush_full_total;
    *prev_flush_deadline_total = flush_deadline_total;
    (rps, bps, mean, full_s, dead_s)
}

fn draw_search(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    let rid = app.active_run_id.as_deref().unwrap_or("<no run>");
    let title = Line::from(vec![
        Span::styled("Search / MCTS", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::raw(rid.to_string()),
    ]);

    let Some(m) = app.dashboard_manifest.as_ref() else {
        let p = Paragraph::new(vec![
            title,
            Line::from(""),
            Line::from(" (run.json not loaded yet)"),
            Line::from(""),
            Line::from("Tip: start an iteration, then press r to refresh."),
        ])
        .block(Block::default().borders(Borders::ALL));
        f.render_widget(p, area);
        return;
    };

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(18), Constraint::Min(1)].as_ref())
        .split(area);

    let max_iter = m.iterations.len().max(1);

    fn carry_series(
        len: usize,
        mut get: impl FnMut(usize) -> Option<u64>,
        default: u64,
    ) -> Vec<u64> {
        let mut out = Vec::with_capacity(len);
        let mut last = default;
        for i in 0..len {
            if let Some(v) = get(i) {
                last = v;
            }
            out.push(last);
        }
        out
    }

    let hnorm_p50 = carry_series(
        max_iter,
        |i| {
            app.selfplay_by_iter
                .get(&(i as u32))
                .and_then(|x| x.visit_entropy_norm_p50)
                .map(|v| (v.clamp(0.0, 1.0) * 1000.0) as u64)
        },
        0,
    );
    let late_discard = carry_series(
        max_iter,
        |i| {
            app.selfplay_by_iter
                .get(&(i as u32))
                .and_then(|x| x.late_eval_discard_frac)
                .map(|v| (v.clamp(0.0, 1.0) * 1000.0) as u64)
        },
        0,
    );
    let pending_coll = carry_series(
        max_iter,
        |i| {
            app.selfplay_by_iter
                .get(&(i as u32))
                .and_then(|x| x.pending_collisions_per_move)
                .map(|v| (v.clamp(0.0, 10.0) * 1000.0) as u64)
        },
        0,
    );
    let noise_flip = carry_series(
        max_iter,
        |i| {
            app.selfplay_by_iter
                .get(&(i as u32))
                .and_then(|x| x.noise_argmax_flip_rate)
                .map(|v| (v.clamp(0.0, 1.0) * 1000.0) as u64)
        },
        0,
    );
    let delta_v = carry_series(
        max_iter,
        |i| {
            app.selfplay_by_iter
                .get(&(i as u32))
                .and_then(|x| x.delta_root_value_mean)
                .map(|v| ((v.clamp(-1.0, 1.0) + 1.0) * 1000.0) as u64)
        },
        1000,
    );
    let top1_p95 = carry_series(
        max_iter,
        |i| {
            app.selfplay_by_iter
                .get(&(i as u32))
                .and_then(|x| x.pi_max_p_p95)
                .map(|v| (v.clamp(0.0, 1.0) * 1000.0) as u64)
        },
        0,
    );

    // Top: 2×3 sparklines
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(chunks[0]);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(top[0]);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(33),
            Constraint::Percentage(33),
            Constraint::Percentage(34),
        ])
        .split(top[1]);

    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("visit_entropy_norm_p50 x1000").borders(Borders::ALL))
            .data(&hnorm_p50),
        left[0],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("late_eval_discard_frac x1000").borders(Borders::ALL))
            .data(&late_discard),
        left[1],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("pending_collisions/move x1000").borders(Borders::ALL))
            .data(&pending_coll),
        left[2],
    );

    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("noise_argmax_flip_rate x1000").borders(Borders::ALL))
            .data(&noise_flip),
        right[0],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("delta_root_value_mean (+1)*1000").borders(Borders::ALL))
            .data(&delta_v),
        right[1],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("max_p95_visit x1000").borders(Borders::ALL))
            .data(&top1_p95),
        right[2],
    );

    // Bottom: table + details
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)].as_ref())
        .split(chunks[1]);

    fn fmt_opt_f64(v: Option<f64>, width: usize, prec: usize) -> String {
        match v {
            Some(x) if x.is_finite() => format!("{x:>width$.prec$}", width = width, prec = prec),
            _ => format!("{:>width$}", "-", width = width),
        }
    }
    fn fmt_opt_pct(v: Option<f64>, width: usize) -> String {
        match v {
            Some(x) if x.is_finite() => {
                let p = x.clamp(0.0, 1.0) * 100.0;
                format!("{p:>width$.1}", width = width)
            }
            _ => format!("{:>width$}", "-", width = width),
        }
    }

    let cur_idx = m.controller_iteration_idx;
    let mut sel = app.search_selected_iter;
    if max_iter > 0 {
        sel = sel.min((max_iter as u32).saturating_sub(1));
    }

    let mut lines: Vec<Line> = Vec::new();
    lines.push(title);
    lines.push(Line::from(""));

    // Header
    const W_IDX: usize = 4;
    const W_H: usize = 5;
    const W_LATE: usize = 6;
    const W_COLL: usize = 6;
    const W_FLIP: usize = 6;
    const W_DV: usize = 7;
    const W_TOP1: usize = 7;
    const W_OVR: usize = 6;
    const W_KL: usize = 5;
    lines.push(Line::from(Span::styled(
        format!(
            "  {sel:>1}   {cur:>1}  {idx:>W_IDX$}  {h:>W_H$}  {late:>W_LATE$}  {coll:>W_COLL$}  {flip:>W_FLIP$}  {dv:>W_DV$}  {top1:>W_TOP1$}  {ovr:>W_OVR$}  {kl:>W_KL$}",
            sel = "S",
            cur = "C",
            idx = "Iter",
            h = "Hn50",
            late = "late%",
            coll = "coll",
            flip = "flip%",
            dv = "ΔV",
            top1 = "top1%",
            ovr = "ovr%",
            kl = "KL",
            W_IDX = W_IDX,
            W_H = W_H,
            W_LATE = W_LATE,
            W_COLL = W_COLL,
            W_FLIP = W_FLIP,
            W_DV = W_DV,
            W_TOP1 = W_TOP1,
            W_OVR = W_OVR,
            W_KL = W_KL,
        ),
        Style::default().fg(Color::DarkGray),
    )));

    // Build all rows then slice for scrolling.
    let mut rows: Vec<Line> = Vec::new();
    for it in &m.iterations {
        let idx = it.idx;
        let ss = app.selfplay_by_iter.get(&idx);
        let h = fmt_opt_f64(ss.and_then(|x| x.visit_entropy_norm_p50), W_H, 3);
        let late = fmt_opt_pct(ss.and_then(|x| x.late_eval_discard_frac), W_LATE);
        let coll = fmt_opt_f64(ss.and_then(|x| x.pending_collisions_per_move), W_COLL, 3);
        let flip = fmt_opt_pct(ss.and_then(|x| x.noise_argmax_flip_rate), W_FLIP);
        let dv = fmt_opt_f64(ss.and_then(|x| x.delta_root_value_mean), W_DV, 3);
        let top1 = fmt_opt_pct(ss.and_then(|x| x.pi_max_p_p95), W_TOP1);
        let ovr = fmt_opt_pct(ss.and_then(|x| x.prior_argmax_overturn_rate), W_OVR);
        let kl = fmt_opt_f64(ss.and_then(|x| x.prior_kl_mean), W_KL, 2);

        // ASCII marker to avoid wide-char rendering differences across terminals.
        let sel_mark = if idx == sel { ">" } else { " " };
        let cur_mark = if idx == cur_idx { "*" } else { " " };

        // Light alerting.
        let mut warn = false;
        if let Some(v) = ss.and_then(|x| x.visit_entropy_norm_p50) {
            if v < 0.20 {
                warn = true;
            }
        }
        if let Some(v) = ss.and_then(|x| x.late_eval_discard_frac) {
            if v > 0.10 {
                warn = true;
            }
        }

        let mut style = if idx == sel {
            Style::default().fg(Color::Cyan)
        } else {
            Style::default()
        };
        if warn {
            style = style.fg(Color::Yellow);
        }

        rows.push(Line::from(Span::styled(
            format!(
                "  {sel_mark}   {cur_mark}  {idx:>W_IDX$}  {h}  {late}  {coll}  {flip}  {dv}  {top1}  {ovr}  {kl}",
            ),
            style,
        )));
    }

    let view_h = cols[0].height.saturating_sub(2) as usize;
    let view_rows = view_h.saturating_sub(lines.len()).max(1);
    let sel_pos = m
        .iterations
        .iter()
        .position(|it| it.idx == sel)
        .unwrap_or(rows.len().saturating_sub(1));
    let max_start = rows.len().saturating_sub(view_rows);
    let mut start = if sel_pos < view_rows {
        0
    } else {
        sel_pos.saturating_sub(view_rows * 2 / 3)
    };
    start = start.min(max_start);
    let end = (start + view_rows).min(rows.len());
    for line in &rows[start..end] {
        lines.push(line.clone());
    }

    let p_table = Paragraph::new(lines).block(Block::default().borders(Borders::ALL));
    f.render_widget(p_table, cols[0]);

    // Details panel
    let ss = app.selfplay_by_iter.get(&sel);
    let mut detail: Vec<Line> = Vec::new();
    detail.push(Line::from(Span::styled(
        format!("iter {sel}"),
        Style::default().add_modifier(Modifier::BOLD),
    )));
    if let Some(ss) = ss {
        if let Some(v) = ss.visit_entropy_norm_p50 {
            detail.push(Line::from(format!("Hnorm_p50: {v:.3}")));
        }
        if let Some(v) = ss.pi_max_p_p95 {
            detail.push(Line::from(format!("max_p95_visit: {:.1}%", v * 100.0)));
        }
        if let Some(v) = ss.delta_root_value_mean {
            detail.push(Line::from(format!("ΔV_mean: {v:+.3}")));
        }
        if let Some(v) = ss.late_eval_discard_frac {
            detail.push(Line::from(format!("late_eval_discard: {:.1}%", v * 100.0)));
        }
        if let Some(v) = ss.pending_collisions_per_move {
            detail.push(Line::from(format!("pending_coll/move: {v:.3}")));
        }
        if let Some(v) = ss.noise_argmax_flip_rate {
            detail.push(Line::from(format!("noise_flip: {:.1}%", v * 100.0)));
        }
        if let Some(v) = ss.prior_argmax_overturn_rate {
            detail.push(Line::from(format!("overturn: {:.1}%", v * 100.0)));
        }
        if let Some(v) = ss.prior_kl_mean {
            detail.push(Line::from(format!("prior_KL: {v:.3}")));
        }

        // Simple diagnosis hints.
        let collapse = ss
            .visit_entropy_norm_p50
            .is_some_and(|x| x < 0.20)
            || ss.pi_max_p_p95.is_some_and(|x| x > 0.85);
        let waste = ss.late_eval_discard_frac.is_some_and(|x| x > 0.10)
            || ss.pending_collisions_per_move.is_some_and(|x| x > 0.20);
        let noise_effective = ss.noise_argmax_flip_rate.is_some_and(|x| x > 0.05);
        let improving = ss.prior_argmax_overturn_rate.is_some_and(|x| x > 0.02)
            && ss.delta_root_value_mean.is_some_and(|x| x > 0.0);

        detail.push(Line::from(""));
        detail.push(Line::from(Span::styled(
            "diagnosis:",
            Style::default().fg(Color::DarkGray),
        )));
        detail.push(Line::from(format!(" collapse: {}", if collapse { "yes" } else { "no" })));
        detail.push(Line::from(format!(" waste: {}", if waste { "yes" } else { "no" })));
        detail.push(Line::from(format!(
            " noise_effective: {}",
            if noise_effective { "yes" } else { "no" }
        )));
        detail.push(Line::from(format!(
            " improving: {}",
            if improving { "yes" } else { "no" }
        )));
    } else {
        detail.push(Line::from(""));
        detail.push(Line::from("(no selfplay_summary for this iteration)"));
    }

    let p_details =
        Paragraph::new(detail).block(Block::default().title("Details").borders(Borders::ALL));
    f.render_widget(p_details, cols[1]);
}

fn draw_system(f: &mut ratatui::Frame, app: &App, area: ratatui::layout::Rect) {
    let rid = app.active_run_id.as_deref().unwrap_or("<no run>");
    let title = Line::from(vec![
        Span::styled("System", Style::default().add_modifier(Modifier::BOLD)),
        Span::raw("  "),
        Span::raw(rid.to_string()),
    ]);

    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(58), Constraint::Percentage(42)].as_ref())
        .split(area);

    let cur = app.system_live.as_ref();

    let mut left: Vec<Line> = Vec::new();
    left.push(Line::from(title));
    left.push(Line::from(""));

    // Phase/status from run.json (best-effort, cheap: already loaded by refresh_dashboard()).
    if let Some(m) = app.dashboard_manifest.as_ref() {
        let phase = m.controller_phase.as_deref().unwrap_or("-");
        let status = m.controller_status.as_deref().unwrap_or("-");
        let planned = app
            .dashboard_planned_total_iterations
            .map(|x| format!("{x}"))
            .unwrap_or_else(|| "?".to_string());
        left.push(Line::from(Span::styled(
            "Controller",
            Style::default().fg(Color::DarkGray),
        )));
        left.push(Line::from(format!(
            " phase={phase}  iter={}/{}",
            m.controller_iteration_idx, planned
        )));
        left.push(Line::from(format!(" status: {status}")));
        left.push(Line::from(""));
    }

    // Config knobs (high-signal).
    left.push(Line::from(Span::styled(
        "Config",
        Style::default().fg(Color::DarkGray),
    )));
    left.push(Line::from(format!(
        " infer: max_batch={} max_wait_us={} torch_threads={:?} interop={:?}",
        app.cfg.inference.max_batch,
        app.cfg.inference.max_wait_us,
        app.cfg.inference.torch_threads,
        app.cfg.inference.torch_interop_threads
    )));
    left.push(Line::from(format!(
        " selfplay: workers={} threads/worker={}  mcts: max_inflight/game={}",
        app.cfg.selfplay.workers,
        app.cfg.selfplay.threads_per_worker,
        app.cfg.mcts.max_inflight_per_game
    )));
    left.push(Line::from(""));

    // Live vs playback status.
    if let Some(err) = &app.system_live_err {
        left.push(Line::from(vec![
            Span::styled("live:", Style::default().fg(Color::Red)),
            Span::raw(" "),
            Span::raw(err.clone()),
        ]));
    } else if let Some(c) = cur {
        left.push(Line::from(format!("source: {}", c.source)));
    }
    left.push(Line::from(""));

    // Per-iteration averages (not required to be live).
    // We use existing per-iteration summaries (learn_summary + selfplay_summary + run.json timestamps).
    if let Some(m) = app.dashboard_manifest.as_ref() {
        let cur_idx = m.controller_iteration_idx;
        let active_idx = if m.iterations.iter().any(|it| it.idx == cur_idx) {
            cur_idx
        } else {
            cur_idx.saturating_sub(1)
        };
        let it = m.iterations.iter().find(|it| it.idx == active_idx);
        let moves_s = app
            .selfplay_by_iter
            .get(&active_idx)
            .and_then(|x| x.moves_s_mean);
        let samples_s = app
            .learn_by_iter
            .get(&active_idx)
            .and_then(|x| x.samples_s_mean);

        let games_s = it.and_then(|it| {
            let games = it.selfplay.games_completed as f64;
            let start = it
                .selfplay
                .first_game_started_ts_ms
                .or(it.selfplay.started_ts_ms)?;
            let end = it.selfplay.ended_ts_ms.unwrap_or_else(yz_logging::now_ms);
            let dt_s = ((end.saturating_sub(start)) as f64) / 1000.0;
            if dt_s > 0.0 && games >= 0.0 {
                Some(games / dt_s)
            } else {
                None
            }
        });

        left.push(Line::from(Span::styled(
            "Per-iteration averages",
            Style::default().fg(Color::DarkGray),
        )));
        left.push(Line::from(format!(
            " iter={}  games/s={}  moves/s={}  samples/s={}",
            active_idx,
            fmt_opt_f(games_s, 2),
            fmt_opt_f(moves_s, 0),
            fmt_opt_f(samples_s, 0),
        )));
        left.push(Line::from(""));

        // Seconds per game / step by phase (derived from run.json timestamps; no extra logging).
        left.push(Line::from(Span::styled(
            "Phase timings",
            Style::default().fg(Color::DarkGray),
        )));
        let now_ms = yz_logging::now_ms();
        let fmt_s = |ms: u64| -> String { format!("{:.2}s", (ms as f64) / 1000.0) };
        let fmt_s_per = |ms: u64, n: u64| -> String {
            if n == 0 {
                "-".to_string()
            } else {
                format!("{:.3}s", (ms as f64) / 1000.0 / (n as f64))
            }
        };

        if let Some(it2) = it {
            // Selfplay: split setup vs running using first_game_started when available.
            let sp_games = it2.selfplay.games_completed;
            if let Some(sp_start) = it2.selfplay.started_ts_ms {
                let sp_first = it2
                    .selfplay
                    .first_game_started_ts_ms
                    .unwrap_or(sp_start);
                let sp_end = it2.selfplay.ended_ts_ms.unwrap_or(now_ms);
                let setup_ms = sp_first.saturating_sub(sp_start);
                let run_ms = sp_end.saturating_sub(sp_first);
                let total_ms = sp_end.saturating_sub(sp_start);
                left.push(Line::from(format!(
                    " selfplay: setup={} run={} total={}  s/game(run)={}  s/game(total)={}",
                    fmt_s(setup_ms),
                    fmt_s(run_ms),
                    fmt_s(total_ms),
                    fmt_s_per(run_ms, sp_games),
                    fmt_s_per(total_ms, sp_games),
                )));
            }

            // Gate: per-game timing (gate may not record first_game_started in some paths).
            let g_games = it2.gate.games_completed;
            if let Some(g_start) = it2.gate.started_ts_ms {
                let g_first = it2.gate.first_game_started_ts_ms.unwrap_or(g_start);
                let g_end = it2.gate.ended_ts_ms.unwrap_or(now_ms);
                let setup_ms = g_first.saturating_sub(g_start);
                let run_ms = g_end.saturating_sub(g_first);
                let total_ms = g_end.saturating_sub(g_start);
                left.push(Line::from(format!(
                    " gate:    setup={} run={} total={}  s/game(run)={}  s/game(total)={}",
                    fmt_s(setup_ms),
                    fmt_s(run_ms),
                    fmt_s(total_ms),
                    fmt_s_per(run_ms, g_games),
                    fmt_s_per(total_ms, g_games),
                )));
            }

            // Train: seconds per step (not games, but useful phase speed signal).
            if let (Some(t_start), Some(steps_done)) =
                (it2.train.started_ts_ms, it2.train.steps_completed)
            {
                let t_end = it2.train.ended_ts_ms.unwrap_or(now_ms);
                let t_ms = t_end.saturating_sub(t_start);
                left.push(Line::from(format!(
                    " train:   total={}  s/step={}",
                    fmt_s(t_ms),
                    fmt_s_per(t_ms, steps_done),
                )));
            }
        } else {
            left.push(Line::from(" (no iteration timing yet)"));
        }
        left.push(Line::from(""));
    }

    fn fmt_opt_u64(v: Option<u64>) -> String {
        v.map(|x| x.to_string()).unwrap_or_else(|| "-".to_string())
    }
    fn fmt_opt_f(v: Option<f64>, prec: usize) -> String {
        match v {
            Some(x) if x.is_finite() => format!("{x:.prec$}", prec = prec),
            _ => "-".to_string(),
        }
    }
    fn fmt_pct(v: Option<f64>) -> String {
        match v {
            Some(x) if x.is_finite() => format!("{:.0}%", (x.clamp(0.0, 1.0) * 100.0)),
            _ => "-".to_string(),
        }
    }

    left.push(Line::from(Span::styled(
        "Live snapshot",
        Style::default().fg(Color::DarkGray),
    )));
    if let Some(c) = cur {
        let age_ms = c.ts.elapsed().as_millis() as u64;
        left.push(Line::from(format!(" age_ms={age_ms}")));
        left.push(Line::from(format!(
            " queue_depth={}  rps={}  bps={}  avg_batch={}",
            fmt_opt_u64(c.queue_depth),
            fmt_opt_f(c.requests_s, 1),
            fmt_opt_f(c.batches_s, 1),
            fmt_opt_f(c.batch_size_mean, 2),
        )));
        left.push(Line::from(format!(
            " batch_p95={}  underfill={}  full={}  would_block={}",
            fmt_opt_f(c.batch_size_p95, 1),
            fmt_pct(c.underfill_frac_mean),
            fmt_pct(c.full_frac_mean),
            fmt_pct(c.would_block_frac),
        )));
        left.push(Line::from(format!(
            " qwait_p95_us={}  fwd_p95_ms={}  flush(full/dead)/s={}/{}",
            fmt_opt_f(c.queue_wait_us_p95, 0),
            fmt_opt_f(c.forward_ms_p95, 2),
            fmt_opt_f(c.flush_full_s, 2),
            fmt_opt_f(c.flush_deadline_s, 2),
        )));
        left.push(Line::from(format!(
            " inflight(sum/max)={}/{}  rtt_p95_med_us={}",
            fmt_opt_u64(c.inflight_sum),
            fmt_opt_u64(c.inflight_max),
            fmt_opt_u64(c.rtt_p95_us_med),
        )));
    } else {
        left.push(Line::from(" (no data yet)"));
    }

    let left_block = Block::default().title("System / Inference").borders(Borders::ALL);
    f.render_widget(Paragraph::new(left).block(left_block), cols[0]);

    // Right: sparklines (prefer live series, fall back to playback samples).
    let spark_rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints(
            [
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
                Constraint::Length(4),
            ]
            .as_ref(),
        )
        .split(cols[1]);

    let series_q = if app.system_series_queue_depth.is_empty() {
        app.infer_snapshots
            .iter()
            .filter_map(|x| x.queue_depth)
            .collect::<Vec<u64>>()
    } else {
        app.system_series_queue_depth.clone()
    };
    let series_rtt = if app.system_series_rtt_p95_us.is_empty() {
        app.infer_snapshots
            .iter()
            .filter_map(|x| x.rtt_p95_us_med)
            .collect::<Vec<u64>>()
    } else {
        app.system_series_rtt_p95_us.clone()
    };
    let series_rps = if app.system_series_rps_x10.is_empty() {
        app.infer_snapshots
            .iter()
            .filter_map(|x| x.requests_s.map(|v| (v.max(0.0) * 10.0) as u64))
            .collect::<Vec<u64>>()
    } else {
        app.system_series_rps_x10.clone()
    };
    let series_batch = if app.system_series_batch_mean_x100.is_empty() {
        app.infer_snapshots
            .iter()
            .filter_map(|x| x.batch_size_mean.map(|v| (v.max(0.0) * 100.0) as u64))
            .collect::<Vec<u64>>()
    } else {
        app.system_series_batch_mean_x100.clone()
    };
    let series_wb = if app.system_series_would_block_x1000.is_empty() {
        app.infer_snapshots
            .iter()
            .filter_map(|x| x.would_block_frac.map(|v| (v.clamp(0.0, 1.0) * 1000.0) as u64))
            .collect::<Vec<u64>>()
    } else {
        app.system_series_would_block_x1000.clone()
    };

    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("queue_depth").borders(Borders::ALL))
            .data(&series_q),
        spark_rows[0],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("rtt_p95_med (us)").borders(Borders::ALL))
            .data(&series_rtt),
        spark_rows[1],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("rps x10").borders(Borders::ALL))
            .data(&series_rps),
        spark_rows[2],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("avg_batch x100").borders(Borders::ALL))
            .data(&series_batch),
        spark_rows[3],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("would_block x1000").borders(Borders::ALL))
            .data(&series_wb),
        spark_rows[4],
    );
}

#[cfg(test)]
mod learn_summary_parse_tests {
    use super::LearnSummaryNdjsonV1;

    #[test]
    fn parses_learn_summary_minimal() {
        let s = r#"{"event":"learn_summary","iter_idx":3,"age_wall_s_p95":12.0,"pi_entropy_mean":1.23,"v_pred_mean":-0.1,"ece":0.02,"samples_s_mean":456.0}"#;
        let v: LearnSummaryNdjsonV1 = serde_json::from_str(s).expect("parse");
        assert_eq!(v.iter_idx, Some(3));
        assert_eq!(v.age_wall_s_p95, Some(12.0));
        assert_eq!(v.pi_entropy_mean, Some(1.23));
        assert_eq!(v.v_pred_mean, Some(-0.1));
        assert_eq!(v.ece, Some(0.02));
        assert_eq!(v.samples_s_mean, Some(456.0));
    }
}

#[cfg(test)]
mod selfplay_summary_parse_tests {
    use super::SelfplaySummaryNdjsonV1;

    #[test]
    fn parses_selfplay_summary_minimal() {
        let s = r#"{"event":"selfplay_summary","iter_idx":2,"moves_executed":1234,"moves_s_mean":99.0,"pending_collisions_per_move":0.01}"#;
        let v: SelfplaySummaryNdjsonV1 = serde_json::from_str(s).expect("parse");
        assert_eq!(v.iter_idx, Some(2));
        assert_eq!(v.moves_executed, Some(1234));
        assert_eq!(v.pending_collisions_per_move, Some(0.01));
    }

    #[test]
    fn parses_selfplay_summary_with_search_quality_fields() {
        let s = r#"{"event":"selfplay_summary","iter_idx":3,"visit_entropy_norm_p50":0.42,"late_eval_discard_frac":0.07,"delta_root_value_mean":0.03}"#;
        let v: SelfplaySummaryNdjsonV1 = serde_json::from_str(s).expect("parse");
        assert_eq!(v.iter_idx, Some(3));
        assert_eq!(v.visit_entropy_norm_p50, Some(0.42));
        assert_eq!(v.late_eval_discard_frac, Some(0.07));
        assert_eq!(v.delta_root_value_mean, Some(0.03));
    }
}
