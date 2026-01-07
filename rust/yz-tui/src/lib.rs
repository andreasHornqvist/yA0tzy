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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Home,
    NamingRun, // Input mode for naming a new run
    Config,
    Dashboard,
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
    #[serde(default)]
    pi_top1_p50: Option<f64>,
    #[serde(default)]
    pi_top1_p95: Option<f64>,
    #[serde(default)]
    pi_eff_actions_p50: Option<f64>,
    #[serde(default)]
    pi_eff_actions_p95: Option<f64>,

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

    // Game stats.
    #[serde(default)]
    game_ply_p95: Option<f64>,
    #[serde(default)]
    score_diff_p95: Option<f64>,
}

#[derive(Debug)]
struct App {
    screen: Screen,
    status: String,
    runs_dir: PathBuf,
    runs: Vec<String>,
    selected: usize,

    active_run_id: Option<String>,
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

    // Learning dashboard state (tail metrics.ndjson).
    learn_loaded_for_run: Option<String>,
    learn_metrics_offset: u64,
    learn_metrics_partial: String,
    learn_by_iter: HashMap<u32, LearnSummaryNdjsonV1>,
    selfplay_by_iter: HashMap<u32, SelfplaySummaryNdjsonV1>,

    iter: Option<yz_controller::IterationHandle>,

    /// Shutdown state: set when user cancels (x) or quits (q) during an active run.
    shutdown_requested: bool,
    /// If true, exit the TUI after shutdown completes.
    shutdown_exit_after: bool,

    /// Input buffer for naming a new run.
    naming_input: String,
}

impl App {
    fn new(runs_dir: PathBuf) -> Self {
        Self {
            screen: Screen::Home,
            status: "q: quit | r: refresh | n: new run".to_string(),
            runs_dir,
            runs: Vec::new(),
            selected: 0,
            active_run_id: None,
            cfg: crate::config_io::default_cfg_for_new_run(),
            form: FormState::default(),
            dashboard_manifest: None,
            dashboard_err: None,
            dashboard_planned_total_iterations: None,
            dashboard_planned_loaded_for_run: None,
            dashboard_tab: DashboardTab::Performance,
            dashboard_iter_scroll: 0,
            dashboard_iter_follow: true,
            learn_loaded_for_run: None,
            learn_metrics_offset: 0,
            learn_metrics_partial: String::new(),
            learn_by_iter: HashMap::new(),
            selfplay_by_iter: HashMap::new(),
            iter: None,
            shutdown_requested: false,
            shutdown_exit_after: false,
            naming_input: String::new(),
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
                "q: quit | r: refresh | n: new run | Enter: open | ↑/↓ select | PgUp/PgDn | Home/End"
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
    }

    fn refresh_learning_metrics(&mut self, run_dir: &Path) {
        let rid = self.active_run_id.clone().unwrap_or_default();
        if self.learn_loaded_for_run.as_deref() != Some(rid.as_str()) {
            self.learn_loaded_for_run = Some(rid);
            self.learn_metrics_offset = 0;
            self.learn_metrics_partial.clear();
            self.learn_by_iter.clear();
            self.selfplay_by_iter.clear();
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
                "r refresh | l toggle learning/perf | d perf | x cancel | ↑/↓ scroll | PgUp/PgDn | Home/End | Esc back | q quit"
                    .to_string();
        }
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
                        KeyCode::Char('q') => {
                            if app.request_quit_or_cancel() {
                                break;
                            }
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
            if matches!(app.screen, Screen::Dashboard) {
                app.refresh_dashboard();
                if app.shutdown_requested {
                    app.status = "cancelling… waiting for shutdown | r refresh".to_string();
                }
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
                if app.form.selected_idx > 0 {
                    app.form.selected_idx -= 1;
                }
            }
            KeyCode::Down => {
                if app.form.selected_idx + 1 < ALL_FIELDS.len() {
                    app.form.selected_idx += 1;
                }
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
    if let Some(pos) = ALL_FIELDS.iter().position(|f| f.section() == next) {
        app.form.selected_idx = pos;
    }
}

fn toggle_or_cycle(app: &mut App, field: FieldId) {
    match field {
        FieldId::GatingPairedSeedSwap => {
            app.cfg.gating.paired_seed_swap = !app.cfg.gating.paired_seed_swap;
        }
        FieldId::GatingDeterministicChance => {
            app.cfg.gating.deterministic_chance = !app.cfg.gating.deterministic_chance;
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
                    cutoff_ply: 10,
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
                    cutoff_ply: 10,
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
        FieldId::MctsTempCutoffPly => {
            let v = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            match &mut cfg.mcts.temperature_schedule {
                TemperatureSchedule::Step { cutoff_ply, .. } => {
                    *cutoff_ply = v;
                    Ok(())
                }
                _ => Err("cutoff_ply only applies when kind=step".to_string()),
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
        FieldId::MctsTempCutoffPly => match cfg.mcts.temperature_schedule {
            TemperatureSchedule::Step { cutoff_ply, .. } => cutoff_ply.to_string(),
            _ => "(n/a)".to_string(),
        },
        FieldId::MctsVirtualLossMode => cfg.mcts.virtual_loss_mode.clone(),
        FieldId::MctsVirtualLoss => format!("{:.3}", cfg.mcts.virtual_loss),

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
        FieldId::MctsTempCutoffPly => {
            let inc = if step == StepSize::Large { 10 } else { 1 };
            if let TemperatureSchedule::Step { cutoff_ply, .. } =
                &mut next.mcts.temperature_schedule
            {
                *cutoff_ply = if dir >= 0 {
                    cutoff_ply.saturating_add(inc)
                } else {
                    cutoff_ply.saturating_sub(inc)
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
                    left.push(Line::from(Span::styled(
                        " Iter   Decision   WinRate   Score(c/b)   Oracle    Loss (t/p/v)",
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
                            let marker = if it.idx == cur_idx { "▸" } else { " " };
                            let promo = it
                                .promoted
                                .map(|p| if p { "promote" } else { "reject" })
                                .unwrap_or("-");
                            let wr = it
                                .gate
                                .win_rate
                                .map(|x| format!("{x:.3}"))
                                .unwrap_or_else(|| "-".to_string());
                            let score_cb = match (it.gate.mean_cand_score, it.gate.mean_best_score) {
                                (Some(c), Some(b)) => format!("{c:.1}/{b:.1}"),
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
                                    "{marker} {:>3}   {promo:>7}   {wr:>6}   {score_cb:>9}   {oracle:>6}   {lt:>5}/{lp:>5}/{lv:>5}",
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

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(9), Constraint::Min(1)].as_ref())
        .split(area);

    // Sparklines (2x2)
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)].as_ref())
        .split(chunks[0]);
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
    let top1_p95 = carry_series(
        max_iter,
        |i| {
            app.learn_by_iter
                .get(&(i as u32))
                .and_then(|x| x.pi_top1_p95)
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
    let overturn = carry_series(
        max_iter,
        |i| {
            app.selfplay_by_iter
                .get(&(i as u32))
                .and_then(|x| x.prior_argmax_overturn_rate)
                .map(|v| (v.clamp(0.0, 1.0) * 1000.0) as u64)
        },
        0,
    );

    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("staleness p95 (s)").borders(Borders::ALL))
            .data(&stale_p95),
        left[0],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("top1 p95 (target)").borders(Borders::ALL))
            .data(&top1_p95),
        left[1],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("ECE").borders(Borders::ALL))
            .data(&ece),
        right[0],
    );
    f.render_widget(
        Sparkline::default()
            .block(Block::default().title("overturn rate").borders(Borders::ALL))
            .data(&overturn),
        right[1],
    );

    // Bottom: table + details.
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)].as_ref())
        .split(chunks[1]);
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
    const W_KL: usize = 5;
    const W_OVR: usize = 5;
    const W_VMEAN: usize = 5;
    const W_ECE: usize = 5;
    const W_ILL: usize = 5;
    const W_SPS: usize = 6;
    const W_MPS: usize = 6;

    lines.push(Line::from(Span::styled(
        format!(
            "  {idx:>W_IDX$}  {dec:<W_DEC$}  {wr:>W_WR$}  {stale:>W_STALE$}  {pi:>W_PI$}  {top1:>W_TOP1$}  {kl:>W_KL$}  {ovr:>W_OVR$}  {vmean:>W_VMEAN$}  {ece:>W_ECE$}  {ill:>W_ILL$}  {sps:>W_SPS$}  {mps:>W_MPS$}",
            idx = "Iter",
            dec = "Decision",
            wr = "WinRate",
            stale = "Stale95",
            pi = "PiEnt",
            top1 = "Top1%",
            kl = "KL",
            ovr = "Ovr%",
            vmean = "Vmean",
            ece = "ECE",
            ill = "Ill%",
            sps = "Samp/s",
            mps = "Move/s",
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
                let kl = fmt_opt_f64(ls.and_then(|x| x.pi_kl_mean), W_KL, 2);
                let ovr = fmt_opt_pct(ss.and_then(|x| x.prior_argmax_overturn_rate), W_OVR);
                let vmean = fmt_opt_f64(ls.and_then(|x| x.v_pred_mean), W_VMEAN, 2);
                let ece = fmt_opt_f64(ls.and_then(|x| x.ece), W_ECE, 3);
                let ill = fmt_opt_pct(ls.and_then(|x| x.p_illegal_mass_mean), W_ILL);
                let sps = fmt_opt_u0(ls.and_then(|x| x.samples_s_mean), W_SPS);
                let mps = fmt_opt_u0(ss.and_then(|x| x.moves_s_mean), W_MPS);

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
                        "{marker} {idx:>W_IDX$}  {dec:<W_DEC$}  {wr}  {stale:>W_STALE$}  {pi}  {top1}  {kl}  {ovr}  {vmean}  {ece}  {ill}  {sps}  {mps}",
                        idx = it.idx,
                        dec = promo,
                        stale = stale95,
                        pi = pi_ent,
                        top1 = top1,
                        kl = kl,
                        ovr = ovr,
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
                if let Some(v) = ls.pi_top1_p95 {
                    detail.push(Line::from(format!("top1_p95_tgt: {:.1}%", v.clamp(0.0, 1.0) * 100.0)));
                }
                if let Some(v) = ls.pi_eff_actions_p50 {
                    detail.push(Line::from(format!("effA_p50_tgt: {v:.1}")));
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
        Field(FieldId),
        Spacer,
        Error(String),
    }

    let mut rows: Vec<Row> = Vec::new();
    for sec in Section::ALL {
        rows.push(Row::Header(sec.title()));
        for f in ALL_FIELDS.iter().copied().filter(|f| f.section() == sec) {
            // Hide step-only fields when not applicable.
            if matches!(f, FieldId::MctsTempT1 | FieldId::MctsTempCutoffPly)
                && matches!(
                    app.cfg.mcts.temperature_schedule,
                    TemperatureSchedule::Constant { .. }
                )
            {
                continue;
            }
            // Hide epochs field when in steps mode.
            if f == FieldId::TrainingEpochs && app.cfg.training.steps_per_iteration.is_some() {
                continue;
            }
            // Hide steps_per_iteration field when in epochs mode.
            if f == FieldId::TrainingStepsPerIteration && app.cfg.training.steps_per_iteration.is_none() {
                continue;
            }
            rows.push(Row::Field(f));
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

// (no tests)
