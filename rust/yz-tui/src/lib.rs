//! Ratatui terminal UI (TUI) for configuring and monitoring runs.
//!
//! v1 scope:
//! - basic screen routing + key handling
//! - run picker (list/create)

mod config_io;
mod form;
mod validate;

use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen};
use crossterm::{execute, ExecutableCommand};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph};
use ratatui::Terminal;

use crate::form::{EditMode, FieldId, FormState, Section, StepSize, ALL_FIELDS};
use yz_core::config::TemperatureSchedule;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Screen {
    Home,
    Config,
    Dashboard,
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

    metrics_tail: Vec<String>,
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
            cfg: yz_core::Config::default(),
            form: FormState::default(),
            metrics_tail: Vec::new(),
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
            self.status = "q: quit | r: refresh | n: new run | ↑/↓: select".to_string();
        }
    }

    fn create_run(&mut self) -> io::Result<()> {
        std::fs::create_dir_all(&self.runs_dir)?;
        let ts = yz_logging::now_ms();
        let id = format!("run_{ts}");
        let dir = self.runs_dir.join(&id);
        std::fs::create_dir_all(dir.join("logs"))?;
        std::fs::create_dir_all(dir.join("models"))?;
        std::fs::create_dir_all(dir.join("replay"))?;
        self.status = format!("Created {id}");
        self.refresh_runs();
        self.selected = self.runs.iter().position(|r| r == &id).unwrap_or(0);
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
        self.active_run_id
            .as_ref()
            .map(|id| self.runs_dir.join(id))
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

    fn refresh_metrics_tail(&mut self) {
        let Some(run_dir) = self.run_dir() else {
            self.metrics_tail.clear();
            return;
        };
        let metrics = run_dir.join("logs").join("metrics.ndjson");
        let mut out: Vec<String> = Vec::new();
        if let Ok(s) = std::fs::read_to_string(metrics) {
            let lines: Vec<&str> = s.lines().filter(|l| !l.trim().is_empty()).collect();
            let n = 20usize.min(lines.len());
            out.extend(lines[lines.len().saturating_sub(n)..].iter().map(|x| x.to_string()));
        }
        self.metrics_tail = out;
    }

    fn enter_dashboard(&mut self) {
        if self.active_run_id.is_none() {
            return;
        }
        self.refresh_metrics_tail();
        self.screen = Screen::Dashboard;
        self.status = "Dashboard: r refresh | Esc back | q quit".to_string();
    }
}

fn config_help(form: &FormState) -> String {
    match form.edit_mode {
        EditMode::View => {
            "Config: ↑/↓ select | Enter edit | ←/→ step | Shift+←/→ big step | Space toggle/cycle | Tab section | s save draft | d dashboard | Esc back | q quit"
                .to_string()
        }
        EditMode::Editing => "Editing: type | Backspace | Enter commit | Esc cancel".to_string(),
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
                        KeyCode::Char('q') => break,
                        KeyCode::Char('r') => app.refresh_runs(),
                        KeyCode::Char('n') => {
                            if let Err(e) = app.create_run() {
                                app.status = format!("create failed: {e}");
                            }
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
                        _ => {}
                    },
                    Screen::Config => match k.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Esc => {
                            app.screen = Screen::Home;
                            app.active_run_id = None;
                            app.refresh_runs();
                        }
                        KeyCode::Char('s') => app.save_config_draft(),
                        KeyCode::Char('d') => app.enter_dashboard(),
                        _ => handle_config_key(&mut app, k),
                    },
                    Screen::Dashboard => match k.code {
                        KeyCode::Char('q') => break,
                        KeyCode::Esc => {
                            app.screen = Screen::Config;
                            app.status = config_help(&app.form);
                        }
                        KeyCode::Char('r') => app.refresh_metrics_tail(),
                        _ => {}
                    },
                }
            }
        }
        if last_tick.elapsed() >= tick_rate {
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

    let sel = app.form.selected_idx.min(ALL_FIELDS.len().saturating_sub(1));
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
    let sel = app.form.selected_idx.min(ALL_FIELDS.len().saturating_sub(1));
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
            if app.cfg.inference.device == "cpu" {
                app.cfg.inference.device = "cuda".to_string();
            } else {
                app.cfg.inference.device = "cpu".to_string();
            }
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
        .and_then(|()| crate::validate::validate_config(&next).map_err(|e| e));
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

fn apply_input_to_cfg(
    cfg: &mut yz_core::Config,
    field: FieldId,
    buf: &str,
) -> Result<(), String> {
    match field {
        FieldId::InferBind => {
            cfg.inference.bind = buf.to_string();
            Ok(())
        }
        FieldId::InferDevice => {
            if buf == "cpu" || buf == "cuda" {
                cfg.inference.device = buf.to_string();
                Ok(())
            } else {
                Err("inference.device must be cpu|cuda".to_string())
            }
        }
        FieldId::InferMaxBatch => {
            cfg.inference.max_batch = buf.parse::<u32>().map_err(|_| "invalid u32".to_string())?;
            Ok(())
        }
        FieldId::InferMaxWaitUs => {
            cfg.inference.max_wait_us = buf.parse::<u64>().map_err(|_| "invalid u64".to_string())?;
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
            cfg.mcts.dirichlet_alpha =
                buf.parse::<f32>().map_err(|_| "invalid f32".to_string())?;
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
            cfg.gating.paired_seed_swap = buf.parse::<bool>().map_err(|_| "invalid bool".to_string())?;
            Ok(())
        }
        FieldId::GatingDeterministicChance => {
            cfg.gating.deterministic_chance = buf.parse::<bool>().map_err(|_| "invalid bool".to_string())?;
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
    }
}

fn field_value_string(cfg: &yz_core::Config, field: FieldId) -> String {
    match field {
        FieldId::InferBind => cfg.inference.bind.clone(),
        FieldId::InferDevice => cfg.inference.device.clone(),
        FieldId::InferMaxBatch => cfg.inference.max_batch.to_string(),
        FieldId::InferMaxWaitUs => cfg.inference.max_wait_us.to_string(),

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

        FieldId::SelfplayGamesPerIteration => cfg.selfplay.games_per_iteration.to_string(),
        FieldId::SelfplayWorkers => cfg.selfplay.workers.to_string(),
        FieldId::SelfplayThreadsPerWorker => cfg.selfplay.threads_per_worker.to_string(),

        FieldId::TrainingBatchSize => cfg.training.batch_size.to_string(),
        FieldId::TrainingLearningRate => format!("{:.6}", cfg.training.learning_rate),
        FieldId::TrainingEpochs => cfg.training.epochs.to_string(),
        FieldId::TrainingWeightDecay => format!("{:.6}", cfg.training.weight_decay),
        FieldId::TrainingStepsPerIteration => cfg
            .training
            .steps_per_iteration
            .map(|x| x.to_string())
            .unwrap_or_else(|| "".to_string()),

        FieldId::GatingGames => cfg.gating.games.to_string(),
        FieldId::GatingSeed => cfg.gating.seed.to_string(),
        FieldId::GatingSeedSetId => cfg
            .gating
            .seed_set_id
            .clone()
            .unwrap_or_else(|| "".to_string()),
        FieldId::GatingWinRateThreshold => format!("{:.4}", cfg.gating.win_rate_threshold),
        FieldId::GatingPairedSeedSwap => cfg.gating.paired_seed_swap.to_string(),
        FieldId::GatingDeterministicChance => cfg.gating.deterministic_chance.to_string(),

        FieldId::ReplayCapacityShards => cfg
            .replay
            .capacity_shards
            .map(|x| x.to_string())
            .unwrap_or_else(|| "".to_string()),

        FieldId::ControllerTotalIterations => cfg
            .controller
            .total_iterations
            .map(|x| x.to_string())
            .unwrap_or_else(|| "".to_string()),
    }
}

fn step_field(app: &mut App, field: FieldId, dir: i32, step: StepSize) {
    let mut next = app.cfg.clone();
    let d = if dir >= 0 { 1.0 } else { -1.0 };
    let ok = match field {
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
            let inc = if step == StepSize::Large { 10_000 } else { 1_000 };
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
            next.mcts.dirichlet_alpha = (next.mcts.dirichlet_alpha as f64 + d * inc).max(0.0) as f32;
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
            if let TemperatureSchedule::Step { t1, .. } = &mut next.mcts.temperature_schedule
            {
                *t1 = (*t1 as f64 + d * inc).max(0.0) as f32;
                true
            } else {
                false
            }
        }
        FieldId::MctsTempCutoffPly => {
            let inc = if step == StepSize::Large { 10 } else { 1 };
            if let TemperatureSchedule::Step { cutoff_ply, .. } = &mut next.mcts.temperature_schedule
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
            next.gating.win_rate_threshold = (next.gating.win_rate_threshold + d * inc).clamp(0.0, 1.0);
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
                Span::raw("TUI"),
            ]);
            let items: Vec<ListItem> = if app.runs.is_empty() {
                vec![ListItem::new(Line::from("No runs yet"))]
            } else {
                app.runs
                    .iter()
                    .enumerate()
                    .map(|(i, r)| {
                        let prefix = if i == app.selected { "> " } else { "  " };
                        ListItem::new(Line::from(format!("{prefix}{r}")))
                    })
                    .collect()
            };
            let list = List::new(items).block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(list, chunks[0]);
        }
        Screen::Config => {
            let rid = app.active_run_id.as_deref().unwrap_or("<no run>");
            let title = Line::from(vec![
                Span::styled("Config", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::raw(format!("run={rid}")),
            ]);
            let body = render_config_lines(app, chunks[0].height.saturating_sub(2) as usize);
            let p = Paragraph::new(body)
                .block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(p, chunks[0]);
        }
        Screen::Dashboard => {
            let rid = app.active_run_id.as_deref().unwrap_or("<no run>");
            let title = Line::from(vec![
                Span::styled("Dashboard", Style::default().add_modifier(Modifier::BOLD)),
                Span::raw("  "),
                Span::raw(format!("run={rid}")),
            ]);
            let mut body: Vec<Line> = Vec::new();
            body.push(Line::from("Tail: logs/metrics.ndjson (last 20 lines)"));
            body.push(Line::from(""));
            if app.metrics_tail.is_empty() {
                body.push(Line::from("(no metrics yet)"));
            } else {
                for l in &app.metrics_tail {
                    body.push(Line::from(l.clone()));
                }
            }
            let p = Paragraph::new(body)
                .block(Block::default().title(title).borders(Borders::ALL));
            f.render_widget(p, chunks[0]);
        }
    }

    let help = Paragraph::new(app.status.clone())
        .block(Block::default().title("Status").borders(Borders::ALL));
    f.render_widget(help, chunks[1]);
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
            if matches!(
                f,
                FieldId::MctsTempT1 | FieldId::MctsTempCutoffPly
            ) {
                if matches!(app.cfg.mcts.temperature_schedule, TemperatureSchedule::Constant { .. }) {
                    continue;
                }
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
    let height = view_height.max(1);
    let start = selected_row.saturating_sub(height / 2);
    let end = (start + height).min(rows.len());
    let slice = &rows[start..end];

    let mut out: Vec<Line<'static>> = Vec::new();
    for r in slice {
        match r {
            Row::Header(t) => out.push(Line::from(vec![Span::styled(
                *t,
                Style::default().add_modifier(Modifier::BOLD),
            )])),
            Row::Spacer => out.push(Line::from("")),
            Row::Error(e) => out.push(Line::from(e.clone())),
            Row::Field(f) => {
                let is_sel = *f == selected_field;
                let label = f.label();
                let v = if app.form.edit_mode == EditMode::Editing && is_sel {
                    app.form.input_buf.clone()
                } else {
                    field_value_string(&app.cfg, *f)
                };
                let prefix = if is_sel { "> " } else { "  " };
                let line = format!("{prefix}{label} = {v}");
                if is_sel {
                    out.push(Line::from(vec![Span::styled(
                        line,
                        Style::default().add_modifier(Modifier::BOLD),
                    )]));
                } else {
                    out.push(Line::from(line));
                }
            }
        }
    }
    out
}


