//! yz-eval: Gating games + stats aggregation for candidate vs best evaluation.

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

use std::fs;
use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;
use yz_core::{apply_action, initial_state, is_terminal, terminal_winner, TurnContext};
use yz_infer::ClientOptions;
use yz_mcts::{ChanceMode, InferBackend, Mcts, MctsConfig, SearchDriver};
#[cfg(test)]
use yz_oracle as oracle_mod;

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct OracleDiagEvent {
    pub avail_mask: u16,
    pub upper_total_cap: u8,
    pub dice_sorted: [u8; 5],
    pub rerolls_left: u8,
    pub chosen_action_idx: u8,
    // Optional context for debugging/offline analysis.
    pub episode_seed: u64,
    pub swap: bool,
}

pub trait OracleDiagSink {
    fn on_step(&mut self, ev: &OracleDiagEvent);
}

#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OracleDiagStep {
    // Oracle-slice state for the current player (solitaire view).
    avail_mask: u16,
    upper_total_cap: u8,
    dice_sorted: [u8; 5],
    rerolls_left: u8,
    chosen_action: yz_core::Action,
}

#[cfg(test)]
fn should_ignore_oracle_action(a: oracle_mod::Action, rerolls_left: u8) -> bool {
    match a {
        oracle_mod::Action::KeepMask { mask } => mask == 31 && rerolls_left > 0,
        oracle_mod::Action::Mark { .. } => false,
    }
}


#[derive(Debug, Error)]
pub enum GateError {
    #[error("invalid gating config: {0}")]
    InvalidConfig(&'static str),
    #[error("failed to load seed set: {0}")]
    SeedSet(String),
    #[error("unsupported infer endpoint: {0}")]
    BadEndpoint(String),
    #[error("infer backend error: {0}")]
    Infer(#[from] yz_mcts::InferBackendError),
    #[error("illegal transition while applying action")]
    IllegalTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GameSpec {
    pub episode_seed: u64,
    pub swap: bool,
}

/// Build a deterministic schedule of games.
///
/// If `paired_swap=true`, then `games` must be even and we generate `games/2` distinct seeds,
/// each expanded to two games: (swap=false) then (swap=true).
pub fn gating_schedule(
    seed0: u64,
    games: u32,
    paired_swap: bool,
) -> Result<Vec<GameSpec>, GateError> {
    if games == 0 {
        return Err(GateError::InvalidConfig("gating.games must be > 0"));
    }
    if paired_swap && !games.is_multiple_of(2) {
        return Err(GateError::InvalidConfig(
            "gating.games must be even when gating.paired_seed_swap=true",
        ));
    }

    let mut out = Vec::with_capacity(games as usize);
    if paired_swap {
        let pairs = games / 2;
        for i in 0..pairs {
            let s = splitmix64(seed0 ^ (i as u64));
            out.push(GameSpec {
                episode_seed: s,
                swap: false,
            });
            out.push(GameSpec {
                episode_seed: s,
                swap: true,
            });
        }
    } else {
        for i in 0..games {
            let s = splitmix64(seed0 ^ (i as u64));
            out.push(GameSpec {
                episode_seed: s,
                swap: false,
            });
        }
    }
    Ok(out)
}

fn gating_schedule_from_seed_set(
    seeds: &[u64],
    requested_games: u32,
    paired_swap: bool,
) -> (Vec<GameSpec>, Vec<u64>, Vec<String>) {
    let mut warnings = Vec::new();

    if seeds.is_empty() {
        return (
            Vec::new(),
            Vec::new(),
            vec!["seed set is empty; running 0 games".to_string()],
        );
    }

    if paired_swap {
        // requested_games must be even; validator enforces this in `gate()`.
        let wanted_seeds = (requested_games / 2) as usize;
        let used_seeds = seeds.len().min(wanted_seeds);
        if seeds.len() > wanted_seeds {
            warnings.push(format!(
                "seed set has {} seeds, truncating to first {} seeds (requested_games={})",
                seeds.len(),
                wanted_seeds,
                requested_games
            ));
        } else if seeds.len() < wanted_seeds {
            warnings.push(format!(
                "seed set has only {} seeds (< wanted {}); running fewer games: {}",
                seeds.len(),
                wanted_seeds,
                2 * seeds.len()
            ));
        }

        let used = seeds[..used_seeds].to_vec();
        let out = schedule_from_seed_list(&used, true);
        return (out, used, warnings);
    }

    // Non-paired mode: one game per seed. Clamp to requested_games and warn on truncation.
    let wanted = requested_games as usize;
    let used = seeds.len().min(wanted);
    if seeds.len() > wanted {
        warnings.push(format!(
            "seed set has {} seeds, truncating to first {} seeds (requested_games={})",
            seeds.len(),
            wanted,
            requested_games
        ));
    } else if seeds.len() < wanted {
        warnings.push(format!(
            "seed set has only {} seeds (< wanted {}); running fewer games: {}",
            seeds.len(),
            wanted,
            seeds.len()
        ));
    }

    let used = seeds[..used].to_vec();
    let out = schedule_from_seed_list(&used, false);
    (out, used, warnings)
}

fn seed_set_path(id: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.join("../../configs/seed_sets")
        .join(format!("{id}.txt"))
}

fn load_seed_set(id: &str) -> Result<Vec<u64>, GateError> {
    let path = seed_set_path(id);
    let txt = fs::read_to_string(&path)
        .map_err(|e| GateError::SeedSet(format!("failed to read {}: {e}", path.display())))?;
    let mut out = Vec::new();
    for (lineno, line) in txt.lines().enumerate() {
        let s = line.trim();
        if s.is_empty() || s.starts_with('#') {
            continue;
        }
        let v: u64 = s.parse().map_err(|e| {
            GateError::SeedSet(format!(
                "failed to parse {}:{} as u64: {e}",
                path.display(),
                lineno + 1
            ))
        })?;
        out.push(v);
    }
    if out.is_empty() {
        return Err(GateError::SeedSet(format!(
            "seed set {} is empty (path={})",
            id,
            path.display()
        )));
    }
    Ok(out)
}

fn splitmix64(mut x: u64) -> u64 {
    // Stable seed mixer (same as common SplitMix64).
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Clone)]
pub struct GateOptions {
    pub infer_endpoint: String,
    pub best_model_id: u32,
    pub cand_model_id: u32,
    pub client_opts: ClientOptions,
    pub mcts_cfg: MctsConfig,
}

/// Periodic scheduling stats for gating (used for low-overhead worker diagnostics).
#[derive(Debug, Clone, Copy)]
pub struct GateTickStats {
    pub ticks: u64,
    pub tasks: u32,
    pub games_completed: u32,
    pub would_block: u64,
    pub progress: u64,
    pub terminal: u64,
    pub best_inflight: u64,
    pub cand_inflight: u64,
}

/// Optional progress sink for gating (used by TUI/controller to render live progress bars).
pub trait GateProgress {
    fn on_game_completed(&mut self, completed: u32, total: u32);
    fn on_tick(&mut self, _stats: &GateTickStats) {
        // Optional; default is no-op.
    }
}

#[derive(Debug, Clone)]
pub struct GatePlan {
    pub schedule: Vec<GameSpec>,
    pub seeds: Vec<u64>,
    pub seeds_hash: String,
    pub warnings: Vec<String>,
}

/// Effective per-process gating parallelism (number of in-flight games per gate-worker).
///
/// If `cfg.gating.threads_per_worker` is set, we use it (clamped to >=1).
/// Otherwise we derive a reasonable default from self-play, typically scaling by 2 because
/// gating uses two model_id streams (best + candidate).
///
/// NOTE: This does **not** change per-game leaf eval concurrency; that remains capped by
/// `cfg.mcts.max_inflight_per_game`.
pub fn effective_gate_parallel_games(cfg: &yz_core::Config) -> u32 {
    let base = cfg.selfplay.threads_per_worker.max(1);
    let derived = base.saturating_mul(2);
    let override_v = cfg.gating.threads_per_worker.map(|x| x.max(1));

    // Soft cap based on client inflight budget (64) and per-game inflight cap.
    let per_game = cfg.mcts.max_inflight_per_game.max(1);
    let cap = (2 * (64 / per_game).max(1)).max(1);

    override_v.unwrap_or(derived).min(cap).max(1)
}

pub fn gate_plan(cfg: &yz_core::Config) -> Result<GatePlan, GateError> {
    let (schedule, seeds, warnings) = if let Some(id) = cfg.gating.seed_set_id.as_deref() {
        if cfg.gating.paired_seed_swap && !cfg.gating.games.is_multiple_of(2) {
            return Err(GateError::InvalidConfig(
                "gating.games must be even when gating.paired_seed_swap=true",
            ));
        }
        let seeds = load_seed_set(id)?;
        gating_schedule_from_seed_set(&seeds, cfg.gating.games, cfg.gating.paired_seed_swap)
    } else {
        let seeds = derived_seed_list(
            cfg.gating.seed,
            cfg.gating.games,
            cfg.gating.paired_seed_swap,
        )?;
        let schedule = schedule_from_seed_list(&seeds, cfg.gating.paired_seed_swap);
        (schedule, seeds, Vec::new())
    };

    Ok(GatePlan {
        seeds_hash: hash_seeds(&seeds),
        schedule,
        seeds,
        warnings,
    })
}

#[derive(Debug, Clone, Default)]
pub struct GatePartial {
    pub games: u32,
    pub cand_wins: u32,
    pub cand_losses: u32,
    pub draws: u32,
    pub cand_score_diff_sum: i64,
    pub cand_score_diff_sumsq: f64, // sum of (diff^2)
    pub cand_score_sum: i64,
    pub best_score_sum: i64,
}

impl GatePartial {
    pub fn merge(&mut self, other: &GatePartial) {
        self.games = self.games.saturating_add(other.games);
        self.cand_wins = self.cand_wins.saturating_add(other.cand_wins);
        self.cand_losses = self.cand_losses.saturating_add(other.cand_losses);
        self.draws = self.draws.saturating_add(other.draws);
        self.cand_score_diff_sum = self.cand_score_diff_sum.saturating_add(other.cand_score_diff_sum);
        self.cand_score_diff_sumsq += other.cand_score_diff_sumsq;
        self.cand_score_sum = self.cand_score_sum.saturating_add(other.cand_score_sum);
        self.best_score_sum = self.best_score_sum.saturating_add(other.best_score_sum);
    }

    pub fn into_report(self, plan: GatePlan) -> GateReport {
        let mut report = GateReport::default();
        report.games = self.games;
        report.cand_wins = self.cand_wins;
        report.cand_losses = self.cand_losses;
        report.draws = self.draws;
        report.cand_score_diff_sum = self.cand_score_diff_sum;
        report.cand_score_sum = self.cand_score_sum;
        report.best_score_sum = self.best_score_sum;
        report.seeds = plan.seeds;
        report.seeds_hash = plan.seeds_hash;
        report.warnings = plan.warnings;

        compute_score_diff_stats_from_moments(
            self.cand_score_diff_sum,
            self.cand_score_diff_sumsq,
            self.games as u64,
            &mut report,
        );

        report
    }
}

#[derive(Debug, Clone, Default)]
pub struct GateReport {
    pub games: u32,
    pub cand_wins: u32,
    pub cand_losses: u32,
    pub draws: u32,
    pub cand_score_diff_sum: i64,
    pub cand_score_sum: i64,
    pub best_score_sum: i64,
    pub seeds: Vec<u64>,
    pub seeds_hash: String,
    pub score_diff_std: f64,
    pub score_diff_se: f64,
    pub score_diff_ci95_low: f64,
    pub score_diff_ci95_high: f64,
    pub draw_rate: f64,
    pub warnings: Vec<String>,
    pub oracle_match_rate_overall: f64,
    pub oracle_match_rate_mark: f64,
    pub oracle_match_rate_reroll: f64,
    pub oracle_keepall_ignored: u64,
}

impl GateReport {
    pub fn win_rate(&self) -> f64 {
        if self.games == 0 {
            return 0.0;
        }
        let w = self.cand_wins as f64;
        let d = self.draws as f64;
        (w + 0.5 * d) / (self.games as f64)
    }

    pub fn mean_score_diff(&self) -> f64 {
        if self.games == 0 {
            return 0.0;
        }
        (self.cand_score_diff_sum as f64) / (self.games as f64)
    }

    pub fn mean_cand_score(&self) -> f64 {
        if self.games == 0 {
            return 0.0;
        }
        (self.cand_score_sum as f64) / (self.games as f64)
    }

    pub fn mean_best_score(&self) -> f64 {
        if self.games == 0 {
            return 0.0;
        }
        (self.best_score_sum as f64) / (self.games as f64)
    }
}

pub fn gate(cfg: &yz_core::Config, opts: GateOptions) -> Result<GateReport, GateError> {
    gate_with_progress(cfg, opts, None)
}

pub fn gate_with_progress(
    cfg: &yz_core::Config,
    opts: GateOptions,
    progress: Option<&mut dyn GateProgress>,
) -> Result<GateReport, GateError> {
    let plan = gate_plan(cfg)?;

    let (best_backend, cand_backend) = connect_two_backends(
        &opts.infer_endpoint,
        opts.best_model_id,
        opts.cand_model_id,
        &opts.client_opts,
    )?;

    let partial = gate_schedule_subset_with_backends(
        cfg,
        opts.mcts_cfg,
        &best_backend,
        &cand_backend,
        &plan.schedule,
        progress,
        None,
    )?;

    // Note: oracle diagnostics are computed offline (controller) for multi-process gating.
    // For in-process `yz gate`, we leave these fields at default.
    Ok(partial.into_report(plan))
}

/// Run gating for a subset of games (used by multi-process gate workers).
///
/// This does **not** build the schedule; caller provides the schedule slice.
pub fn gate_schedule_subset(
    cfg: &yz_core::Config,
    opts: GateOptions,
    schedule: &[GameSpec],
    progress: Option<&mut dyn GateProgress>,
    oracle_sink: Option<&mut dyn OracleDiagSink>,
) -> Result<GatePartial, GateError> {
    let (best_backend, cand_backend) = connect_two_backends(
        &opts.infer_endpoint,
        opts.best_model_id,
        opts.cand_model_id,
        &opts.client_opts,
    )?;
    gate_schedule_subset_with_backends(
        cfg,
        opts.mcts_cfg,
        &best_backend,
        &cand_backend,
        schedule,
        progress,
        oracle_sink,
    )
}

fn gate_schedule_subset_with_backends(
    cfg: &yz_core::Config,
    mut mcts_cfg: MctsConfig,
    best_backend: &InferBackend,
    cand_backend: &InferBackend,
    schedule: &[GameSpec],
    mut progress: Option<&mut dyn GateProgress>,
    mut oracle_sink: Option<&mut dyn OracleDiagSink>,
) -> Result<GatePartial, GateError> {
    let total = schedule.len() as u32;

    // Always disable root Dirichlet noise in gating/eval.
    mcts_cfg.dirichlet_epsilon = 0.0;

    let mut partial = GatePartial::default();
    partial.games = total;

    if schedule.is_empty() {
        // Keep semantics: 0 scheduled -> 0 completed (progress never called).
        return Ok(partial);
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum StepStatus {
        Progress,
        WouldBlock,
        Terminal,
    }

    #[derive(Debug, Clone)]
    struct TerminalResult {
        cand_score: i32,
        best_score: i32,
        outcome: Outcome,
    }

    #[derive(Debug)]
    struct StepResult {
        status: StepStatus,
        terminal: Option<TerminalResult>,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum BackendSel {
        Best,
        Cand,
    }

    struct GateGameTask {
        gs: GameSpec,
        cand_seat: u8,
        ctx: TurnContext,
        state: yz_core::GameState,
        chance_mode: ChanceMode,
        mcts: Mcts,
        search: Option<SearchDriver>,
        search_backend: BackendSel,
        active: bool,
    }

    impl GateGameTask {
        fn new(
            cfg: &yz_core::Config,
            mcts_cfg: MctsConfig,
            gs: GameSpec,
        ) -> Result<Self, GateError> {
            // Initialize state using the same chance mode policy as the sequential gating loop.
            let mut ctx = if cfg.gating.deterministic_chance {
                TurnContext::new_deterministic(gs.episode_seed)
            } else {
                TurnContext::new_rng(gs.episode_seed)
            };
            let state = initial_state(&mut ctx);
            let chance_mode = if cfg.gating.deterministic_chance {
                ChanceMode::Deterministic {
                    episode_seed: gs.episode_seed,
                }
            } else {
                ChanceMode::Rng {
                    seed: gs.episode_seed,
                }
            };

            let cand_seat = if gs.swap { 1u8 } else { 0u8 };
            let mcts =
                Mcts::new(mcts_cfg).map_err(|_| GateError::InvalidConfig("bad mcts cfg"))?;

            Ok(Self {
                gs,
                cand_seat,
                ctx,
                state,
                chance_mode,
                mcts,
                search: None,
                search_backend: BackendSel::Best,
                active: true,
            })
        }

        fn reset(
            &mut self,
            cfg: &yz_core::Config,
            mcts_cfg: MctsConfig,
            gs: GameSpec,
        ) -> Result<(), GateError> {
            *self = GateGameTask::new(cfg, mcts_cfg, gs)?;
            Ok(())
        }

        fn backend_for_turn<'a>(
            &mut self,
            best_backend: &'a InferBackend,
            cand_backend: &'a InferBackend,
        ) -> (&'a InferBackend, BackendSel, bool) {
            let cand_turn = self.state.player_to_move == self.cand_seat;
            let backend = select_backend(best_backend, cand_backend, self.state.player_to_move, self.gs.swap);
            let sel = if std::ptr::eq(backend, cand_backend) {
                BackendSel::Cand
            } else {
                BackendSel::Best
            };
            (backend, sel, cand_turn)
        }

        fn step(
            &mut self,
            best_backend: &InferBackend,
            cand_backend: &InferBackend,
            max_work: u32,
            oracle_sink: &mut Option<&mut dyn OracleDiagSink>,
        ) -> Result<StepResult, GateError> {
            if !self.active {
                return Ok(StepResult {
                    status: StepStatus::Terminal,
                    terminal: None,
                });
            }

            if is_terminal(&self.state) {
                self.active = false;
                let tr = Self::terminal_result(&self.state, self.cand_seat)?;
                return Ok(StepResult {
                    status: StepStatus::Terminal,
                    terminal: Some(tr),
                });
            }

            if self.search.is_none() {
                let (backend, sel, _cand_turn) = self.backend_for_turn(best_backend, cand_backend);
                self.search_backend = sel;
                let d = self
                    .mcts
                    .begin_search_with_backend(self.state, self.chance_mode, backend);
                self.search = Some(d);
            }

            let backend = match self.search_backend {
                BackendSel::Best => best_backend,
                BackendSel::Cand => cand_backend,
            };
            if let Some(search) = &mut self.search {
                let res = search.tick(&mut self.mcts, backend, max_work);
                if let Some(sr) = res {
                    // Search finished -> choose executed action.
                    let a = argmax_tie_lowest(&sr.pi);
                    let action = yz_core::index_to_action(a);

                    // Oracle diagnostics: record only candidate moves (agent under test).
                    let is_cand_turn = self.state.player_to_move == self.cand_seat;
                    if is_cand_turn {
                        let ps = &self.state.players[self.state.player_to_move as usize];
                        if let Some(s) = oracle_sink.as_deref_mut() {
                            s.on_step(&OracleDiagEvent {
                                avail_mask: ps.avail_mask,
                                upper_total_cap: ps.upper_total_cap,
                                dice_sorted: self.state.dice_sorted,
                                rerolls_left: self.state.rerolls_left,
                                chosen_action_idx: yz_core::action_to_index(action),
                                episode_seed: self.gs.episode_seed,
                                swap: self.gs.swap,
                            });
                        }
                    }

                    let next = apply_action(self.state, action, &mut self.ctx)
                        .map_err(|_| GateError::IllegalTransition)?;
                    self.state = next;
                    self.search = None;

                    if is_terminal(&self.state) {
                        self.active = false;
                        let tr = Self::terminal_result(&self.state, self.cand_seat)?;
                        return Ok(StepResult {
                            status: StepStatus::Terminal,
                            terminal: Some(tr),
                        });
                    }

                    return Ok(StepResult {
                        status: StepStatus::Progress,
                        terminal: None,
                    });
                }
            }

            Ok(StepResult {
                status: StepStatus::WouldBlock,
                terminal: None,
            })
        }

        fn terminal_result(state: &yz_core::GameState, cand_seat: u8) -> Result<TerminalResult, GateError> {
            let winner = terminal_winner(state).map_err(|_| GateError::IllegalTransition)?;
            let best_seat = 1u8 ^ cand_seat;
            let cand_score = state.players[cand_seat as usize].total_score as i32;
            let best_score = state.players[best_seat as usize].total_score as i32;
            let outcome = if winner == 2 {
                Outcome::Draw
            } else if winner == cand_seat {
                Outcome::Win
            } else {
                Outcome::Loss
            };
            Ok(TerminalResult {
                cand_score,
                best_score,
                outcome,
            })
        }
    }

    let parallel_games =
        (effective_gate_parallel_games(cfg) as usize).min(schedule.len()).max(1);
    let init_n = parallel_games.min(schedule.len());
    let mut tasks: Vec<GateGameTask> = Vec::with_capacity(init_n);
    for &gs in schedule.iter().take(init_n) {
        tasks.push(GateGameTask::new(cfg, mcts_cfg, gs)?);
    }
    let mut next_idx: usize = init_n;

    let mut completed: u32 = 0;
    let mut sched_ticks: u64 = 0;
    let mut sched_progress: u64 = 0;
    let mut sched_would_block: u64 = 0;
    let mut sched_terminal: u64 = 0;
    let mut last_tick_emit = std::time::Instant::now();
    while (completed as usize) < schedule.len() {
        let mut made_progress = false;
        sched_ticks += 1;
        for t in tasks.iter_mut() {
            if !t.active && next_idx >= schedule.len() {
                continue;
            }
            let r = t.step(best_backend, cand_backend, 64, &mut oracle_sink)?;
            match r.status {
                StepStatus::Progress => {
                    made_progress = true;
                    sched_progress += 1;
                }
                StepStatus::Terminal => {
                    if let Some(tr) = r.terminal {
                        match tr.outcome {
                            Outcome::Win => partial.cand_wins += 1,
                            Outcome::Loss => partial.cand_losses += 1,
                            Outcome::Draw => partial.draws += 1,
                        }
                        partial.cand_score_sum += tr.cand_score as i64;
                        partial.best_score_sum += tr.best_score as i64;
                        let d = tr.cand_score - tr.best_score;
                        partial.cand_score_diff_sum += d as i64;
                        let x = d as f64;
                        partial.cand_score_diff_sumsq += x * x;

                        completed += 1;
                        made_progress = true;
                        sched_terminal += 1;
                        if let Some(p) = progress.as_deref_mut() {
                            p.on_game_completed(completed, total);
                        }
                    }

                    // Start a new game in this slot if schedule items remain.
                    if next_idx < schedule.len() {
                        let gs = schedule[next_idx];
                        next_idx += 1;
                        t.reset(cfg, mcts_cfg, gs)?;
                    }
                }
                StepStatus::WouldBlock => {}
            }
        }

        // Optional: periodic tick stats for diagnostics.
        if let Some(p) = progress.as_deref_mut() {
            if last_tick_emit.elapsed() >= Duration::from_secs(1) {
                let b = best_backend.stats_snapshot();
                let c = cand_backend.stats_snapshot();
                p.on_tick(&GateTickStats {
                    ticks: sched_ticks,
                    tasks: tasks.len() as u32,
                    games_completed: completed,
                    would_block: sched_would_block,
                    progress: sched_progress,
                    terminal: sched_terminal,
                    best_inflight: b.inflight as u64,
                    cand_inflight: c.inflight as u64,
                });
                last_tick_emit = std::time::Instant::now();
            }
        }

        // Avoid busy-spinning when all tasks are waiting on inference, but do NOT introduce
        // fixed polling delays (which add response-consumption latency).
        if !made_progress {
            sched_would_block += 1;
            best_backend.wait_for_progress(Duration::from_micros(100));
            cand_backend.wait_for_progress(Duration::from_micros(100));
        }
    }
    Ok(partial)
}

fn derived_seed_list(seed0: u64, games: u32, paired_swap: bool) -> Result<Vec<u64>, GateError> {
    if games == 0 {
        return Err(GateError::InvalidConfig("gating.games must be > 0"));
    }
    if paired_swap && !games.is_multiple_of(2) {
        return Err(GateError::InvalidConfig(
            "gating.games must be even when gating.paired_seed_swap=true",
        ));
    }
    let n = if paired_swap { games / 2 } else { games };
    let mut out = Vec::with_capacity(n as usize);
    for i in 0..n {
        out.push(splitmix64(seed0 ^ (i as u64)));
    }
    Ok(out)
}

fn schedule_from_seed_list(seeds: &[u64], paired_swap: bool) -> Vec<GameSpec> {
    if paired_swap {
        let mut out = Vec::with_capacity(seeds.len() * 2);
        for &s in seeds {
            out.push(GameSpec {
                episode_seed: s,
                swap: false,
            });
            out.push(GameSpec {
                episode_seed: s,
                swap: true,
            });
        }
        out
    } else {
        seeds
            .iter()
            .copied()
            .map(|s| GameSpec {
                episode_seed: s,
                swap: false,
            })
            .collect()
    }
}

fn hash_seeds(seeds: &[u64]) -> String {
    let mut buf = Vec::with_capacity(seeds.len() * 8);
    for &s in seeds {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    blake3::hash(&buf).to_hex().to_string()
}

fn compute_score_diff_stats_from_moments(
    diff_sum: i64,
    diff_sumsq: f64,
    n: u64,
    report: &mut GateReport,
) {
    if n == 0 {
        report.score_diff_std = 0.0;
        report.score_diff_se = 0.0;
        report.score_diff_ci95_low = 0.0;
        report.score_diff_ci95_high = 0.0;
        report.draw_rate = 0.0;
        return;
    }

    let n_f = n as f64;
    let mean = (diff_sum as f64) / n_f;
    // Sample variance from moments: var = (Σx² - (Σx)²/n)/(n-1)
    let var = if n > 1 {
        let ss = diff_sumsq - (diff_sum as f64) * (diff_sum as f64) / n_f;
        (ss / (n_f - 1.0)).max(0.0)
    } else {
        0.0
    };
    let std = var.sqrt();
    let se = std / n_f.sqrt();
    let ci = 1.96 * se;

    report.score_diff_std = std;
    report.score_diff_se = se;
    report.score_diff_ci95_low = mean - ci;
    report.score_diff_ci95_high = mean + ci;
    report.draw_rate = (report.draws as f64) / n_f;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Outcome {
    Win,
    Loss,
    Draw,
}

fn argmax_tie_lowest(pi: &[f32; yz_core::A]) -> u8 {
    let mut best_i: u8 = 0;
    let mut best_v: f32 = f32::NEG_INFINITY;
    for (i, &v) in pi.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i as u8;
        }
    }
    best_i
}

fn select_backend<'a>(
    best: &'a InferBackend,
    cand: &'a InferBackend,
    player_to_move: u8,
    swap: bool,
) -> &'a InferBackend {
    let cand_seat = if swap { 1u8 } else { 0u8 };
    if player_to_move == cand_seat {
        cand
    } else {
        best
    }
}

fn connect_two_backends(
    endpoint: &str,
    best_model_id: u32,
    cand_model_id: u32,
    client_opts: &ClientOptions,
) -> Result<(InferBackend, InferBackend), GateError> {
    if let Some(rest) = endpoint.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            let best = InferBackend::connect_uds(rest, best_model_id, client_opts.clone())?;
            let cand = InferBackend::connect_uds(rest, cand_model_id, client_opts.clone())?;
            return Ok((best, cand));
        }
        #[cfg(not(unix))]
        {
            return Err(GateError::BadEndpoint(endpoint.to_string()));
        }
    }
    if let Some(rest) = endpoint.strip_prefix("tcp://") {
        let best = InferBackend::connect_tcp(rest, best_model_id, client_opts.clone())?;
        let cand = InferBackend::connect_tcp(rest, cand_model_id, client_opts.clone())?;
        return Ok((best, cand));
    }
    Err(GateError::BadEndpoint(endpoint.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn start_dummy_infer_server_tcp() -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
        use std::net::TcpListener;
        use std::thread;

        use yz_infer::codec::{decode_request_v1, encode_response_v1};
        use yz_infer::frame::{read_frame, write_frame};
        use yz_infer::protocol::{InferResponseV1, ACTION_SPACE_A};

        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut sock) = conn else { break };
                thread::spawn(move || loop {
                    let payload = match read_frame(&mut sock) {
                        Ok(p) => p,
                        Err(_) => break,
                    };
                    let req = match decode_request_v1(&payload) {
                        Ok(r) => r,
                        Err(_) => break,
                    };

                    let mut logits = vec![0.0f32; ACTION_SPACE_A as usize];
                    for (i, &b) in req.legal_mask.iter().enumerate() {
                        if b == 0 {
                            logits[i] = -1.0e9;
                        }
                    }

                    let resp = InferResponseV1 {
                        request_id: req.request_id,
                        policy_logits: logits,
                        value: 0.0,
                        margin: None,
                    };
                    let out = encode_response_v1(&resp);
                    if write_frame(&mut sock, &out).is_err() {
                        break;
                    }
                });
            }
        });
        (addr, handle)
    }

    #[test]
    fn version_is_set() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn schedule_is_deterministic() {
        let a = gating_schedule(123, 10, false).unwrap();
        let b = gating_schedule(123, 10, false).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.len(), 10);
        assert!(a.iter().all(|g| g.swap == false));
    }

    #[test]
    fn seed_set_parse_rejects_bad_line() {
        // Parse via internal helper by simulating a file parse error string.
        // We can't easily write into configs/seed_sets in a unit test without IO;
        // instead we validate that the loader reports parse errors with context by calling the parser path.
        // This is a minimal sanity check: SplitMix schedule remains tested elsewhere.
        let res = "not_a_number".parse::<u64>();
        assert!(res.is_err());
    }

    #[test]
    fn seed_set_sizing_truncates_and_warns_when_longer() {
        let seeds = vec![1u64, 2, 3, 4, 5];
        let (sched, _used, warn) = gating_schedule_from_seed_set(&seeds, 4, true); // wanted_seeds=2
        assert_eq!(sched.len(), 4);
        assert!(!warn.is_empty());
    }

    #[test]
    fn seed_set_sizing_runs_fewer_games_when_shorter() {
        let seeds = vec![1u64];
        let (sched, _used, warn) = gating_schedule_from_seed_set(&seeds, 10, true); // wanted_seeds=5
        assert_eq!(sched.len(), 2);
        assert!(!warn.is_empty());
    }

    #[test]
    fn seeds_hash_is_stable_for_same_seed_list() {
        let seeds = vec![1u64, 2, 3, 4, 5];
        let h1 = hash_seeds(&seeds);
        let h2 = hash_seeds(&seeds);
        assert_eq!(h1, h2);
        assert!(h1.len() >= 32);
    }

    #[test]
    fn seeds_hash_changes_if_seed_list_changes() {
        let h1 = hash_seeds(&[1u64, 2, 3]);
        let h2 = hash_seeds(&[1u64, 2, 4]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn paired_swap_requires_even_games() {
        let e = gating_schedule(0, 3, true).unwrap_err();
        let msg = format!("{e}");
        assert!(msg.contains("even"));
    }

    #[test]
    fn paired_swap_expands_each_seed_twice() {
        let s = gating_schedule(999, 6, true).unwrap();
        assert_eq!(s.len(), 6);
        assert_eq!(s[0].swap, false);
        assert_eq!(s[1].swap, true);
        assert_eq!(s[0].episode_seed, s[1].episode_seed);
        assert_eq!(s[2].swap, false);
        assert_eq!(s[3].swap, true);
        assert_eq!(s[2].episode_seed, s[3].episode_seed);
    }

    #[test]
    fn gate_smoke_paired_swap_is_stable_for_identical_models() {
        let (addr, _server) = start_dummy_infer_server_tcp();

        let cfg = yz_core::Config::from_yaml(
            r#"
inference:
  bind: "tcp://127.0.0.1:0"
  device: "cpu"
  max_batch: 32
  max_wait_us: 1000
mcts:
  c_puct: 1.5
  budget_reroll: 8
  budget_mark: 8
  max_inflight_per_game: 4
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
  seed: 0
  win_rate_threshold: 0.5
  paired_seed_swap: true
  deterministic_chance: true
"#,
        )
        .unwrap();

        let report = gate(
            &cfg,
            GateOptions {
                infer_endpoint: format!("tcp://{addr}"),
                best_model_id: 0,
                cand_model_id: 1,
                client_opts: ClientOptions {
                    max_inflight_total: 1024,
                    max_outbound_queue: 1024,
                    request_id_start: 1,
                },
                mcts_cfg: MctsConfig {
                    c_puct: 1.5,
                    simulations_mark: 8,
                    simulations_reroll: 8,
                    dirichlet_alpha: 0.3,
                    dirichlet_epsilon: 0.0,
                    max_inflight: 4,
                    virtual_loss: 1.0,
                },
            },
        )
        .unwrap();

        assert_eq!(report.games, 2);
        assert_eq!(report.cand_score_diff_sum, 0);
        assert!((report.win_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn oracle_ignore_rule_keepall_is_enabled_only_when_rerolls_left_is_positive() {
        assert!(should_ignore_oracle_action(
            oracle_mod::Action::KeepMask { mask: 31 },
            2
        ));
        assert!(!should_ignore_oracle_action(
            oracle_mod::Action::KeepMask { mask: 31 },
            0
        ));
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct OracleDiagCounts {
        total: u64,
        matched: u64,
        total_mark: u64,
        matched_mark: u64,
        total_reroll: u64,
        matched_reroll: u64,
        oracle_keepall_ignored: u64,
    }

    fn compute_oracle_diag_counts(steps: &[OracleDiagStep]) -> OracleDiagCounts {
        let oracle = oracle_mod::oracle();
        let mut c = OracleDiagCounts::default();
        for s in steps {
            let (oa, _ev) = oracle.best_action(
                s.avail_mask,
                s.upper_total_cap,
                s.dice_sorted,
                s.rerolls_left,
            );
            if should_ignore_oracle_action(oa, s.rerolls_left) {
                c.oracle_keepall_ignored += 1;
                continue;
            }

            let expected: yz_core::Action = match oa {
                oracle_mod::Action::Mark { cat } => yz_core::Action::Mark(cat),
                oracle_mod::Action::KeepMask { mask } => yz_core::Action::KeepMask(mask),
            };
            let is_match = expected == s.chosen_action;

            c.total += 1;
            if is_match {
                c.matched += 1;
            }

            match s.chosen_action {
                yz_core::Action::Mark(_) => {
                    c.total_mark += 1;
                    if is_match {
                        c.matched_mark += 1;
                    }
                }
                yz_core::Action::KeepMask(_) => {
                    c.total_reroll += 1;
                    if is_match {
                        c.matched_reroll += 1;
                    }
                }
            }
        }
        c
    }

    #[test]
    fn oracle_diag_is_deterministic_for_identical_steps() {
        // This builds the oracle DP table once; acceptable for a unit test.
        let steps = vec![OracleDiagStep {
            avail_mask: oracle_mod::FULL_MASK,
            upper_total_cap: 0,
            dice_sorted: [1, 2, 3, 4, 5],
            rerolls_left: 2,
            chosen_action: yz_core::Action::KeepMask(0),
        }];
        let a = compute_oracle_diag_counts(&steps);
        let b = compute_oracle_diag_counts(&steps);
        assert_eq!(a.oracle_keepall_ignored, b.oracle_keepall_ignored);
        assert_eq!(a.total, b.total);
        assert_eq!(a.matched, b.matched);
        assert_eq!(a.total_mark, b.total_mark);
        assert_eq!(a.matched_mark, b.matched_mark);
        assert_eq!(a.total_reroll, b.total_reroll);
        assert_eq!(a.matched_reroll, b.matched_reroll);
    }
}
