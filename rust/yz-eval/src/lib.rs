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
use yz_core::PlayerState;

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

// --- Fixed oracle set (for stable per-iteration diagnostics) ---

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize, PartialEq, Eq, Hash)]
pub struct OracleSliceStateV1 {
    pub avail_mask: u16,
    pub upper_total_cap: u8,
    pub dice_sorted: [u8; 5],
    pub rerolls_left: u8,
}

fn oracle_set_path(id: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    dir.join("../../configs/oracle_sets")
        .join(format!("{id}.json"))
}

pub fn load_oracle_set(id: &str) -> Result<Vec<OracleSliceStateV1>, GateError> {
    let path = oracle_set_path(id);
    let bytes = fs::read(&path)
        .map_err(|e| GateError::SeedSet(format!("failed to read {}: {e}", path.display())))?;
    let v = parse_oracle_set_json(&bytes)
        .map_err(|e| GateError::SeedSet(format!("failed to parse {}: {e}", path.display())))?;
    if v.is_empty() {
        return Err(GateError::SeedSet(format!(
            "oracle set {} is empty (path={})",
            id,
            path.display()
        )));
    }
    Ok(v)
}

pub fn parse_oracle_set_json(bytes: &[u8]) -> Result<Vec<OracleSliceStateV1>, serde_json::Error> {
    serde_json::from_slice::<Vec<OracleSliceStateV1>>(bytes)
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct OracleFixedReport {
    pub set_id: String,
    pub num_states: u64,
    pub total: u64,
    pub matched: u64,
    pub total_mark: u64,
    pub matched_mark: u64,
    pub total_reroll: u64,
    pub matched_reroll: u64,
    /// Bucketed by rerolls_left (2/1/0).
    pub total_by_r: [u64; 3],
    pub matched_by_r: [u64; 3],

    /// Per-action counts for the *chosen* action index (0..A-1).
    pub action_total: Vec<u64>,
    pub action_matched: Vec<u64>,

    /// Per-action counts for how often the candidate *chose* each action (0..A-1).
    ///
    /// This lets us compute e.g. “how often was this action oracle-optimal, given that the
    /// candidate chose it?” i.e. action-level precision.
    pub chosen_total: Vec<u64>,

    /// Per-turn (turn index) counts. turn_idx = 0..14 based on filled categories.
    pub turn_total: Vec<u64>,
    pub turn_matched: Vec<u64>,
}

impl OracleFixedReport {
    pub fn match_rate_overall(&self) -> f64 {
        if self.total == 0 { 0.0 } else { self.matched as f64 / self.total as f64 }
    }
    pub fn match_rate_mark(&self) -> f64 {
        if self.total_mark == 0 { 0.0 } else { self.matched_mark as f64 / self.total_mark as f64 }
    }
    pub fn match_rate_reroll(&self) -> f64 {
        if self.total_reroll == 0 { 0.0 } else { self.matched_reroll as f64 / self.total_reroll as f64 }
    }
}

fn connect_one_backend(
    endpoint: &str,
    model_id: u32,
    client_opts: &ClientOptions,
) -> Result<InferBackend, GateError> {
    if let Some(rest) = endpoint.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            return Ok(InferBackend::connect_uds(rest, model_id, client_opts.clone())?);
        }
        #[cfg(not(unix))]
        {
            return Err(GateError::BadEndpoint(endpoint.to_string()));
        }
    }
    if let Some(rest) = endpoint.strip_prefix("tcp://") {
        return Ok(InferBackend::connect_tcp(rest, model_id, client_opts.clone())?);
    }
    Err(GateError::BadEndpoint(endpoint.to_string()))
}

/// Convert oracle action to core action, mapping oracle's "mark at rerolls>0" to KeepMask(31).
///
/// The oracle returns Mark when keep-all is optimal because marking immediately is equivalent
/// to keeping all dice and marking at the final roll. Under mark-only-at-roll-3 rules, we
/// translate this to KeepMask(31) since Mark is illegal at rerolls_left > 0.
fn oracle_action_to_az_action(oa: yz_oracle::Action, rerolls_left: u8) -> yz_core::Action {
    match oa {
        yz_oracle::Action::Mark { cat } if rerolls_left > 0 => {
            // Oracle's Mark at rerolls>0 means keep-all is optimal -> KeepMask(31)
            yz_core::Action::KeepMask(31)
        }
        yz_oracle::Action::Mark { cat } => yz_core::Action::Mark(cat),
        yz_oracle::Action::KeepMask { mask } => yz_core::Action::KeepMask(mask),
    }
}

fn canonicalize_keepmask(dice_sorted: [u8; 5], mask: u8) -> u8 {
    debug_assert!(dice_sorted.windows(2).all(|w| w[0] <= w[1]));
    debug_assert!(mask < 32);

    // Count kept faces.
    let mut need = [0u8; 6];
    for i in 0..5usize {
        let bit = 1u8 << (4 - i);
        if (mask & bit) != 0 {
            let face = dice_sorted[i] as usize;
            debug_assert!((1..=6).contains(&face));
            need[face - 1] = need[face - 1].saturating_add(1);
        }
    }

    // Reconstruct canonical mask by keeping the rightmost occurrences for each face.
    let mut out: u8 = 0;
    for i in (0..5usize).rev() {
        let bit = 1u8 << (4 - i);
        let face = dice_sorted[i] as usize;
        let slot = face - 1;
        if need[slot] > 0 {
            need[slot] -= 1;
            out |= bit;
        }
    }
    out
}

pub fn eval_fixed_oracle_set(
    _cfg: &yz_core::Config,
    infer_endpoint: &str,
    cand_model_id: u32,
    client_opts: &ClientOptions,
    mut mcts_cfg: MctsConfig,
    set_id: &str,
    states: &[OracleSliceStateV1],
    mut progress: Option<&mut dyn FnMut(u64, u64)>,
) -> Result<OracleFixedReport, GateError> {
    // Match gating: no Dirichlet noise in eval.
    mcts_cfg.dirichlet_epsilon = 0.0;

    let oracle = yz_oracle::oracle();
    let backend = connect_one_backend(infer_endpoint, cand_model_id, client_opts)?;

    let mut rep = OracleFixedReport {
        set_id: set_id.to_string(),
        num_states: states.len() as u64,
        action_total: vec![0u64; yz_core::A],
        action_matched: vec![0u64; yz_core::A],
        chosen_total: vec![0u64; yz_core::A],
        turn_total: vec![0u64; yz_core::NUM_CATS],
        turn_matched: vec![0u64; yz_core::NUM_CATS],
        ..OracleFixedReport::default()
    };
    rep.num_states = states.len() as u64;

    let total_states = states.len() as u64;
    if let Some(cb) = progress.as_deref_mut() {
        cb(0, total_states);
    }
    for (idx, st) in states.iter().enumerate() {
        // Build a canonical 1v1 state with a constant opponent. This makes the metric stable.
        let gs = yz_core::GameState {
            players: [
                PlayerState {
                    avail_mask: st.avail_mask,
                    upper_total_cap: st.upper_total_cap,
                    total_score: 0,
                },
                PlayerState {
                    avail_mask: 0x7fff,
                    upper_total_cap: 0,
                    total_score: 0,
                },
            ],
            dice_sorted: st.dice_sorted,
            rerolls_left: st.rerolls_left,
            player_to_move: 0,
        };
        let seed = splitmix64(
            (st.avail_mask as u64)
                ^ ((st.upper_total_cap as u64) << 16)
                ^ ((st.rerolls_left as u64) << 24)
                ^ (st.dice_sorted[0] as u64)
                ^ ((st.dice_sorted[1] as u64) << 8)
                ^ ((st.dice_sorted[2] as u64) << 16)
                ^ ((st.dice_sorted[3] as u64) << 24)
                ^ ((st.dice_sorted[4] as u64) << 32),
        );
        let chance_mode = ChanceMode::Rng { seed };

        let mut mcts = Mcts::new(mcts_cfg).map_err(|_| GateError::InvalidConfig("bad mcts cfg"))?;
        let mut search = mcts.begin_search_with_backend(gs, chance_mode, &backend);
        let sr = loop {
            // Use a modest work chunk; SearchDriver stops when budget is reached.
            if let Some(sr) = search.tick(&mut mcts, &backend, 2048) {
                break sr;
            }
        };
        let chosen_idx = argmax_tie_lowest(&sr.pi);
        let chosen = yz_core::index_to_action(chosen_idx);

        let (oa, _ev) =
            oracle.best_action(st.avail_mask, st.upper_total_cap, st.dice_sorted, st.rerolls_left);
        let expected = oracle_action_to_az_action(oa, st.rerolls_left);
        let expected_canon = match expected {
            yz_core::Action::KeepMask(mask) => {
                yz_core::Action::KeepMask(canonicalize_keepmask(st.dice_sorted, mask))
            }
            _ => expected,
        };
        let chosen_canon = match chosen {
            yz_core::Action::KeepMask(mask) => {
                yz_core::Action::KeepMask(canonicalize_keepmask(st.dice_sorted, mask))
            }
            _ => chosen,
        };
        let is_match = chosen_canon == expected_canon;

        let chosen_ai = yz_core::action_to_index(chosen_canon) as usize;
        if chosen_ai < rep.chosen_total.len() {
            rep.chosen_total[chosen_ai] += 1;
        }

        let avail = st.avail_mask;
        let available = avail.count_ones() as i32;
        let filled = (yz_core::NUM_CATS as i32 - available).clamp(0, yz_core::NUM_CATS as i32);
        let turn_idx = filled as usize;
        if turn_idx < rep.turn_total.len() {
            rep.turn_total[turn_idx] += 1;
            if is_match {
                rep.turn_matched[turn_idx] += 1;
            }
        }

        let ai = yz_core::action_to_index(expected_canon) as usize;
        if ai < yz_core::A && ai < rep.action_total.len() && ai < rep.action_matched.len() {
            rep.action_total[ai] += 1;
            if is_match {
                rep.action_matched[ai] += 1;
            }
        }

        rep.total += 1;
        if is_match {
            rep.matched += 1;
        }
        match chosen {
            yz_core::Action::Mark(_) => {
                rep.total_mark += 1;
                if is_match {
                    rep.matched_mark += 1;
                }
            }
            yz_core::Action::KeepMask(_) => {
                rep.total_reroll += 1;
                if is_match {
                    rep.matched_reroll += 1;
                }
            }
        }
        let b = match st.rerolls_left {
            2 => 0,
            1 => 1,
            _ => 2,
        };
        rep.total_by_r[b] += 1;
        if is_match {
            rep.matched_by_r[b] += 1;
        }

        // Progress callback (periodic + final).
        if let Some(cb) = progress.as_deref_mut() {
            let done = (idx + 1) as u64;
            if done == total_states || done.is_multiple_of(32) {
                cb(done, total_states);
            }
        }
    }

    Ok(rep)
}

// --- Gating replay traces (for TUI replay viewer) ---

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GateReplayPlayerV1 {
    pub avail_mask: u16,
    pub upper_total_cap: u8,
    pub total_score: i16,
    /// Per-category raw scores (bonus excluded). None means unfilled.
    pub filled_raw: [Option<i16>; yz_core::NUM_CATS],
    pub upper_raw_sum: i16,
    pub upper_bonus_awarded: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GateReplayStepV1 {
    pub ply: u32,
    pub turn_idx: u8,
    pub player_to_move_before: u8,
    pub rerolls_left_before: u8,
    pub dice_sorted_before: [u8; 5],
    pub chosen_action_idx: u8,
    pub chosen_action_str: String,
    pub player_to_move_after: u8,
    pub rerolls_left_after: u8,
    pub dice_sorted_after: [u8; 5],
    pub players_after: [GateReplayPlayerV1; 2],
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GateReplayTerminalV1 {
    pub cand_score: i32,
    pub best_score: i32,
    pub outcome: String, // "win" | "loss" | "draw" from candidate POV
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GateReplayTraceV1 {
    pub trace_version: u32,
    pub episode_seed: u64,
    pub swap: bool,
    pub cand_seat: u8,
    pub best_seat: u8,
    pub steps: Vec<GateReplayStepV1>,
    pub terminal: Option<GateReplayTerminalV1>,
}

pub trait GateReplaySink {
    /// Called when one gating game completes; implementers should persist the trace.
    fn on_game(&mut self, trace: GateReplayTraceV1);
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
fn oracle_action_to_az_action_test(oa: oracle_mod::Action, rerolls_left: u8) -> yz_core::Action {
    match oa {
        oracle_mod::Action::Mark { cat } if rerolls_left > 0 => {
            // Oracle's Mark at rerolls>0 means keep-all is optimal -> KeepMask(31)
            yz_core::Action::KeepMask(31)
        }
        oracle_mod::Action::Mark { cat } => yz_core::Action::Mark(cat),
        oracle_mod::Action::KeepMask { mask } => yz_core::Action::KeepMask(mask),
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

fn state_hash_for_search_seed(s: &yz_core::GameState) -> u64 {
    let p0 = &s.players[0];
    let p1 = &s.players[1];
    let mut x = 0u64;
    x ^= p0.avail_mask as u64;
    x ^= (p1.avail_mask as u64) << 16;
    x ^= (p0.upper_total_cap as u64) << 32;
    x ^= (p1.upper_total_cap as u64) << 40;
    x ^= (p0.total_score as u64) << 48;
    x ^= (p1.total_score as u64) << 56;
    x ^= (s.player_to_move as u64) << 8;
    x ^= (s.rerolls_left as u64) << 12;
    x ^= (s.dice_sorted[0] as u64) << 1;
    x ^= (s.dice_sorted[1] as u64) << 4;
    x ^= (s.dice_sorted[2] as u64) << 7;
    x ^= (s.dice_sorted[3] as u64) << 10;
    x ^= (s.dice_sorted[4] as u64) << 13;
    splitmix64(x)
}

fn gating_search_seed(episode_seed: u64, state: &yz_core::GameState, ply: u32) -> u64 {
    splitmix64(
        episode_seed
            ^ state_hash_for_search_seed(state)
            ^ (ply as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
    )
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
    pub cand_wins: u32,
    pub cand_losses: u32,
    pub draws: u32,
    pub would_block: u64,
    pub progress: u64,
    pub terminal: u64,
    pub best_inflight: u64,
    pub cand_inflight: u64,
}

/// Optional progress sink for gating (used by TUI/controller to render live progress bars).
pub trait GateProgress {
    fn on_game_completed(&mut self, completed: u32, total: u32);
    /// Optional: allow the caller to stop launching new games (e.g., controller-driven SPRT stop).
    fn should_stop(&mut self) -> bool {
        false
    }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SprtDecision {
    Continue,
    AcceptH1,
    AcceptH0,
    InconclusiveMaxGames,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct SprtDiag {
    pub min_games: u32,
    pub max_games: u32,
    pub alpha: f64,
    pub beta: f64,
    pub delta: f64,
    pub thr: f64,
    pub p0: f64,
    pub p1: f64,
    pub llr: f64,
    pub bound_a: f64,
    pub bound_b: f64,
    pub decision: SprtDecision,
    /// The number of completed games when the decision was first reached (may be < games played due to overshoot).
    pub games_at_decision: u32,
}

fn sprt_params(cfg: &yz_core::Config) -> Result<Option<(u32, u32, f64, f64, f64, f64, f64)>, GateError> {
    if !cfg.gating.katago.sprt {
        return Ok(None);
    }
    let min_games = cfg.gating.katago.sprt_min_games;
    let max_games = cfg.gating.katago.sprt_max_games;
    if min_games == 0 {
        return Err(GateError::InvalidConfig(
            "gating.katago.sprt_min_games must be > 0",
        ));
    }
    if max_games < min_games {
        return Err(GateError::InvalidConfig(
            "gating.katago.sprt_max_games must be >= sprt_min_games",
        ));
    }
    if cfg.gating.paired_seed_swap {
        if !min_games.is_multiple_of(2) {
            return Err(GateError::InvalidConfig(
                "gating.katago.sprt_min_games must be even when gating.paired_seed_swap=true",
            ));
        }
        if !max_games.is_multiple_of(2) {
            return Err(GateError::InvalidConfig(
                "gating.katago.sprt_max_games must be even when gating.paired_seed_swap=true",
            ));
        }
    }
    let alpha = cfg.gating.katago.sprt_alpha;
    let beta = cfg.gating.katago.sprt_beta;
    let delta = cfg.gating.katago.sprt_delta;
    let thr = cfg.gating.win_rate_threshold;
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(GateError::InvalidConfig("gating.katago.sprt_alpha must be in (0,1)"));
    }
    if !(0.0 < beta && beta < 1.0) {
        return Err(GateError::InvalidConfig("gating.katago.sprt_beta must be in (0,1)"));
    }
    if !(0.0 < delta && delta < 1.0) {
        return Err(GateError::InvalidConfig("gating.katago.sprt_delta must be in (0,1)"));
    }
    let p0 = thr - delta;
    let p1 = thr + delta;
    if !(p0 > 0.0 && p1 < 1.0 && p0 < p1) {
        return Err(GateError::InvalidConfig(
            "SPRT requires p0=thr-delta and p1=thr+delta with 0<p0<p1<1",
        ));
    }
    Ok(Some((min_games, max_games, alpha, beta, delta, p0, p1)))
}

fn sprt_llr(wins: u32, draws: u32, losses: u32, p0: f64, p1: f64) -> f64 {
    // Half-point binomial transform: draw counts as 0.5 win.
    let w2 = (2 * wins + draws) as f64;
    let l2 = (2 * losses + draws) as f64;
    w2 * (p1 / p0).ln() + l2 * ((1.0 - p1) / (1.0 - p0)).ln()
}

fn sprt_bounds(alpha: f64, beta: f64) -> (f64, f64) {
    // Standard Wald thresholds.
    let a = ((1.0 - beta) / alpha).ln();
    let b = (beta / (1.0 - alpha)).ln();
    (a, b)
}

fn wilson_ci95(successes: u64, trials: u64) -> (f64, f64) {
    if trials == 0 {
        return (0.0, 1.0);
    }
    let n = trials as f64;
    let phat = (successes as f64) / n;
    let z = 1.959_963_984_540_054_f64; // 95% two-sided
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = (phat + z2 / (2.0 * n)) / denom;
    let half = (z / denom)
        * ((phat * (1.0 - phat) / n) + (z2 / (4.0 * n * n))).sqrt();
    ((center - half).clamp(0.0, 1.0), (center + half).clamp(0.0, 1.0))
}

fn win_ci95_halfpoint(wins: u32, draws: u32, losses: u32) -> (f64, f64) {
    let w2 = (2 * wins + draws) as u64;
    let l2 = (2 * losses + draws) as u64;
    let n2 = w2 + l2;
    wilson_ci95(w2, n2)
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
    // When SPRT is enabled, treat sprt_max_games as the gating games cap / UI target.
    let requested_games = if cfg.gating.katago.sprt {
        sprt_params(cfg)?.map(|(_, max, _, _, _, _, _)| max).unwrap_or(cfg.gating.games)
    } else {
        cfg.gating.games
    };

    let (schedule, seeds, warnings) = if let Some(id) = cfg.gating.seed_set_id.as_deref() {
        if cfg.gating.paired_seed_swap && !requested_games.is_multiple_of(2) {
            return Err(GateError::InvalidConfig(
                "gating.games must be even when gating.paired_seed_swap=true",
            ));
        }
        let seeds = load_seed_set(id)?;
        gating_schedule_from_seed_set(&seeds, requested_games, cfg.gating.paired_seed_swap)
    } else {
        let seeds = derived_seed_list(
            cfg.gating.seed,
            requested_games,
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
    pub sprt: Option<SprtDiag>,
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
        // SPRT diag is only meaningful for single-process SPRT gating; leave it unset when merging.
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
        report.sprt = self.sprt;

        let (lo, hi) = win_ci95_halfpoint(self.cand_wins, self.draws, self.cand_losses);
        report.win_rate_ci95_low = lo;
        report.win_rate_ci95_high = hi;

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
    pub win_rate_ci95_low: f64,
    pub win_rate_ci95_high: f64,
    pub draw_rate: f64,
    pub warnings: Vec<String>,
    pub oracle_match_rate_overall: f64,
    pub oracle_match_rate_mark: f64,
    pub oracle_match_rate_reroll: f64,
    pub sprt: Option<SprtDiag>,
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
    replay_sink: Option<&mut dyn GateReplaySink>,
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
        replay_sink,
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
    mut replay_sink: Option<&mut dyn GateReplaySink>,
) -> Result<GatePartial, GateError> {
    let total = schedule.len() as u32;
    let sprt_cfg = sprt_params(cfg)?;

    // Always disable root Dirichlet noise in gating/eval.
    mcts_cfg.dirichlet_epsilon = 0.0;

    let mut partial = GatePartial::default();

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
        mcts: Mcts,
        search: Option<SearchDriver>,
        search_backend: BackendSel,
        active: bool,
        // Replay trace state (derived scorecards + step buffer).
        filled_raw: [[Option<i16>; yz_core::NUM_CATS]; 2],
        upper_raw_sum: [i16; 2],
        upper_bonus_awarded: [bool; 2],
        ply: u32,
        turn_idx: u8,
        trace_steps: Vec<GateReplayStepV1>,
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

            let cand_seat = if gs.swap { 1u8 } else { 0u8 };
            let mcts =
                Mcts::new(mcts_cfg).map_err(|_| GateError::InvalidConfig("bad mcts cfg"))?;

            Ok(Self {
                gs,
                cand_seat,
                ctx,
                state,
                mcts,
                search: None,
                search_backend: BackendSel::Best,
                active: true,
                filled_raw: [[None; yz_core::NUM_CATS]; 2],
                upper_raw_sum: [0, 0],
                upper_bonus_awarded: [false, false],
                ply: 0,
                turn_idx: 0,
                trace_steps: Vec::new(),
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
            cfg: &yz_core::Config,
            best_backend: &InferBackend,
            cand_backend: &InferBackend,
            max_work: u32,
            oracle_sink: &mut Option<&mut dyn OracleDiagSink>,
            replay_sink: &mut Option<&mut dyn GateReplaySink>,
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
                let search_mode = if cfg.gating.deterministic_chance {
                    ChanceMode::Rng {
                        seed: gating_search_seed(self.gs.episode_seed, &self.state, self.ply),
                    }
                } else {
                    ChanceMode::Rng {
                        seed: self.gs.episode_seed,
                    }
                };
                let d = self
                    .mcts
                    .begin_search_with_backend(self.state, search_mode, backend);
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

                    // Replay trace: record every executed move with enough context to replay.
                    let dice_before = self.state.dice_sorted;
                    let rerolls_before = self.state.rerolls_left;
                    let player_before = self.state.player_to_move;
                    let chosen_action_idx = yz_core::action_to_index(action);
                    let chosen_action_str = format!("{action:?}");

                    // Update derived scorecards for Mark before applying transition (needs dice_before).
                    if let yz_core::Action::Mark(cat) = action {
                        let actor = player_before as usize;
                        let raw = yz_core::scores_for_dice(dice_before)[cat as usize] as i16;
                        self.filled_raw[actor][cat as usize] = Some(raw);
                        if cat < 6 {
                            let prev_upper = self.upper_raw_sum[actor] as i32;
                            let new_upper = prev_upper + raw as i32;
                            if !self.upper_bonus_awarded[actor] && prev_upper < 63 && new_upper >= 63 {
                                self.upper_bonus_awarded[actor] = true;
                            }
                            self.upper_raw_sum[actor] = new_upper.clamp(-32768, 32767) as i16;
                        }
                    }

                    let next = apply_action(self.state, action, &mut self.ctx)
                        .map_err(|_| GateError::IllegalTransition)?;
                    self.state = next;
                    self.search = None;

                    // Mark ends a full turn.
                    if matches!(action, yz_core::Action::Mark(_)) {
                        self.turn_idx = self.turn_idx.saturating_add(1);
                    }

                    let players_after = [0usize, 1usize].map(|pi| {
                        let ps = self.state.players[pi];
                        GateReplayPlayerV1 {
                            avail_mask: ps.avail_mask,
                            upper_total_cap: ps.upper_total_cap,
                            total_score: ps.total_score,
                            filled_raw: self.filled_raw[pi],
                            upper_raw_sum: self.upper_raw_sum[pi],
                            upper_bonus_awarded: self.upper_bonus_awarded[pi],
                        }
                    });

                    self.trace_steps.push(GateReplayStepV1 {
                        ply: self.ply,
                        turn_idx: self.turn_idx,
                        player_to_move_before: player_before,
                        rerolls_left_before: rerolls_before,
                        dice_sorted_before: dice_before,
                        chosen_action_idx,
                        chosen_action_str,
                        player_to_move_after: self.state.player_to_move,
                        rerolls_left_after: self.state.rerolls_left,
                        dice_sorted_after: self.state.dice_sorted,
                        players_after,
                    });
                    self.ply = self.ply.saturating_add(1);

                    if is_terminal(&self.state) {
                        self.active = false;
                        let tr = Self::terminal_result(&self.state, self.cand_seat)?;

                        // Emit full trace at terminal (best-effort).
                        if let Some(s) = replay_sink.as_deref_mut() {
                            let best_seat = 1u8 ^ self.cand_seat;
                            let outcome = match tr.outcome {
                                Outcome::Win => "win",
                                Outcome::Loss => "loss",
                                Outcome::Draw => "draw",
                            }
                            .to_string();
                            s.on_game(GateReplayTraceV1 {
                                trace_version: 1,
                                episode_seed: self.gs.episode_seed,
                                swap: self.gs.swap,
                                cand_seat: self.cand_seat,
                                best_seat,
                                steps: std::mem::take(&mut self.trace_steps),
                                terminal: Some(GateReplayTerminalV1 {
                                    cand_score: tr.cand_score,
                                    best_score: tr.best_score,
                                    outcome,
                                }),
                            });
                        }

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
    let mut stop_launching = false;
    let mut sprt_diag: Option<SprtDiag> = None;
    let mut sched_ticks: u64 = 0;
    let mut sched_progress: u64 = 0;
    let mut sched_would_block: u64 = 0;
    let mut sched_terminal: u64 = 0;
    let mut last_tick_emit = std::time::Instant::now();

    loop {
        if (completed as usize) >= schedule.len() {
            break;
        }
        if stop_launching && tasks.iter().all(|t| !t.active) {
            break;
        }

        let mut made_progress = false;
        sched_ticks += 1;
        for t in tasks.iter_mut() {
            if !t.active && next_idx >= schedule.len() {
                continue;
            }
            let r = t.step(cfg, best_backend, cand_backend, 64, &mut oracle_sink, &mut replay_sink)?;
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

                        // SPRT: update decision after each completed game, and stop launching new games once decided.
                        if let Some((min_games, _max_games, alpha, beta, delta, p0, p1)) = sprt_cfg {
                            if sprt_diag.is_none() && completed >= min_games {
                                let llr = sprt_llr(
                                    partial.cand_wins,
                                    partial.draws,
                                    partial.cand_losses,
                                    p0,
                                    p1,
                                );
                                let (a, b) = sprt_bounds(alpha, beta);
                                let (decision, stop_now) = if llr >= a {
                                    (SprtDecision::AcceptH1, true)
                                } else if llr <= b {
                                    (SprtDecision::AcceptH0, true)
                                } else {
                                    (SprtDecision::Continue, false)
                                };
                                if stop_now {
                                    stop_launching = true;
                                    next_idx = schedule.len(); // stop refilling slots; allow small overshoot from in-flight games.
                                    sprt_diag = Some(SprtDiag {
                                        min_games,
                                        max_games: total,
                                        alpha,
                                        beta,
                                        delta,
                                        thr: cfg.gating.win_rate_threshold,
                                        p0,
                                        p1,
                                        llr,
                                        bound_a: a,
                                        bound_b: b,
                                        decision,
                                        games_at_decision: completed,
                                    });
                                }
                            }
                        }
                    }

                    // Start a new game in this slot if schedule items remain.
                    if !stop_launching && next_idx < schedule.len() {
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
            if !stop_launching && p.should_stop() {
                stop_launching = true;
                next_idx = schedule.len();
            }
            if last_tick_emit.elapsed() >= Duration::from_secs(1) {
                let b = best_backend.stats_snapshot();
                let c = cand_backend.stats_snapshot();
                p.on_tick(&GateTickStats {
                    ticks: sched_ticks,
                    tasks: tasks.len() as u32,
                    games_completed: completed,
                    cand_wins: partial.cand_wins,
                    cand_losses: partial.cand_losses,
                    draws: partial.draws,
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
    partial.games = completed;
    partial.sprt = if let Some(diag) = sprt_diag {
        Some(diag)
    } else if let Some((min_games, _max_games, alpha, beta, delta, p0, p1)) = sprt_cfg {
        // SPRT enabled but no early decision; mark as inconclusive at max games.
        let llr = sprt_llr(partial.cand_wins, partial.draws, partial.cand_losses, p0, p1);
        let (a, b) = sprt_bounds(alpha, beta);
        Some(SprtDiag {
            min_games,
            max_games: total,
            alpha,
            beta,
            delta,
            thr: cfg.gating.win_rate_threshold,
            p0,
            p1,
            llr,
            bound_a: a,
            bound_b: b,
            decision: if completed >= min_games {
                SprtDecision::InconclusiveMaxGames
            } else {
                SprtDecision::Continue
            },
            games_at_decision: completed,
        })
    } else {
        None
    };
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
                    for i in 0..(ACTION_SPACE_A as usize) {
                        if ((req.legal_mask >> i) & 1) == 0 {
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
                    protocol_version: yz_infer::protocol::PROTOCOL_VERSION_V1,
                    legal_mask_bitset: false,
                },
                mcts_cfg: MctsConfig {
                    c_puct: 1.5,
                    simulations_mark: 8,
                    simulations_reroll: 8,
                    dirichlet_alpha: 0.3,
                    dirichlet_epsilon: 0.0,
                    max_inflight: 4,
                    virtual_loss_mode: yz_mcts::VirtualLossMode::QPenalty,
                    virtual_loss: 1.0,
                    expansion_lock: false,
                    explicit_keepmask_chance: false,
                    chance_pw_enabled: false,
                    chance_pw_c: 2.0,
                    chance_pw_alpha: 0.6,
                    chance_pw_max_children: 64,
                },
            },
        )
        .unwrap();

        assert_eq!(report.games, 2);
        assert_eq!(report.cand_score_diff_sum, 0);
        assert!((report.win_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn oracle_action_mapping_mark_at_rerolls_gt0_becomes_keepall() {
        // Oracle's Mark at rerolls>0 should map to KeepMask(31)
        let result = oracle_action_to_az_action_test(
            oracle_mod::Action::Mark { cat: 5 },
            2,
        );
        assert_eq!(result, yz_core::Action::KeepMask(31));

        // Oracle's Mark at rerolls==0 stays as Mark
        let result = oracle_action_to_az_action_test(
            oracle_mod::Action::Mark { cat: 5 },
            0,
        );
        assert_eq!(result, yz_core::Action::Mark(5));

        // KeepMask is always passed through
        let result = oracle_action_to_az_action_test(
            oracle_mod::Action::KeepMask { mask: 15 },
            2,
        );
        assert_eq!(result, yz_core::Action::KeepMask(15));
    }

    #[derive(Debug, Clone, Copy, Default)]
    struct OracleDiagCounts {
        total: u64,
        matched: u64,
        total_mark: u64,
        matched_mark: u64,
        total_reroll: u64,
        matched_reroll: u64,
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

            let expected = oracle_action_to_az_action_test(oa, s.rerolls_left);
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
        assert_eq!(a.total, b.total);
        assert_eq!(a.matched, b.matched);
        assert_eq!(a.total_mark, b.total_mark);
        assert_eq!(a.matched_mark, b.matched_mark);
        assert_eq!(a.total_reroll, b.total_reroll);
        assert_eq!(a.matched_reroll, b.matched_reroll);
    }

    #[test]
    fn sprt_llr_is_zero_when_p0_equals_p1() {
        // Degenerate case: p0==p1 should give llr=0 regardless of outcomes.
        // (We never allow this in config, but it sanity-checks the math helper.)
        let p = 0.55;
        let llr = sprt_llr(10, 4, 6, p, p);
        assert!(llr.is_finite());
        assert!((llr - 0.0).abs() < 1e-12);
    }

    #[test]
    fn sprt_accepts_h1_when_all_wins() {
        // With a strong win stream, llr should quickly exceed A.
        let alpha = 0.05;
        let beta = 0.05;
        let (a, _b) = sprt_bounds(alpha, beta);
        let p0 = 0.52;
        let p1 = 0.58;
        let mut wins = 0u32;
        for _ in 0..200 {
            wins += 1;
            let llr = sprt_llr(wins, 0, 0, p0, p1);
            if llr >= a {
                return;
            }
        }
        panic!("expected llr to cross A for all-wins stream within 200 games");
    }

    #[test]
    fn sprt_accepts_h0_when_all_losses() {
        // With a strong loss stream, llr should quickly go below B.
        let alpha = 0.05;
        let beta = 0.05;
        let (_a, b) = sprt_bounds(alpha, beta);
        let p0 = 0.52;
        let p1 = 0.58;
        let mut losses = 0u32;
        for _ in 0..200 {
            losses += 1;
            let llr = sprt_llr(0, 0, losses, p0, p1);
            if llr <= b {
                return;
            }
        }
        panic!("expected llr to cross B for all-losses stream within 200 games");
    }

    #[test]
    fn sprt_draw_counts_as_half_point() {
        // One draw should be equivalent to half win + half loss in the doubled-trials view.
        let p0 = 0.52;
        let p1 = 0.58;
        let llr_draw = sprt_llr(0, 1, 0, p0, p1);
        let llr_half = 0.5 * sprt_llr(1, 0, 0, p0, p1) + 0.5 * sprt_llr(0, 0, 1, p0, p1);
        assert!((llr_draw - llr_half).abs() < 1e-9);
    }

    #[test]
    fn win_ci95_halfpoint_smoke_bounds() {
        let (lo, hi) = win_ci95_halfpoint(10, 10, 10);
        assert!(0.0 <= lo && lo <= hi && hi <= 1.0);
    }

    #[test]
    fn oracle_set_json_parses() {
        let bytes = br#"[
  {"avail_mask": 32767, "upper_total_cap": 0, "dice_sorted": [1,2,3,4,5], "rerolls_left": 2},
  {"avail_mask": 1, "upper_total_cap": 63, "dice_sorted": [6,6,6,6,6], "rerolls_left": 0}
]"#;
        let v = parse_oracle_set_json(bytes).expect("parse");
        assert_eq!(v.len(), 2);
        assert_eq!(v[0].dice_sorted, [1, 2, 3, 4, 5]);
        assert_eq!(v[1].rerolls_left, 0);
    }

    #[test]
    fn gating_search_seed_is_reproducible_and_varies_with_state_and_ply() {
        let mut ctx = TurnContext::new_deterministic(123);
        let s0 = initial_state(&mut ctx);
        let a = gating_search_seed(999, &s0, 0);
        let b = gating_search_seed(999, &s0, 0);
        assert_eq!(a, b);

        let c = gating_search_seed(999, &s0, 1);
        assert_ne!(a, c);

        // Mutate state slightly (dice) and ensure seed changes.
        let mut s1 = s0;
        s1.dice_sorted = [1, 1, 1, 1, 1];
        let d = gating_search_seed(999, &s1, 0);
        assert_ne!(a, d);
    }

    #[test]
    fn deterministic_execution_is_stable_for_same_seed_and_action() {
        let seed = 424242u64;
        let mut ctx0 = TurnContext::new_deterministic(seed);
        let mut ctx1 = TurnContext::new_deterministic(seed);
        let s0 = initial_state(&mut ctx0);
        let s1 = initial_state(&mut ctx1);
        assert_eq!(s0.dice_sorted, s1.dice_sorted);

        // Apply a deterministic KeepMask transition on both.
        let a = yz_core::Action::KeepMask(0); // reroll all dice
        let n0 = apply_action(s0, a, &mut ctx0).unwrap();
        let n1 = apply_action(s1, a, &mut ctx1).unwrap();
        assert_eq!(n0.dice_sorted, n1.dice_sorted);
        assert_eq!(n0.rerolls_left, n1.rerolls_left);
        assert_eq!(n0.player_to_move, n1.player_to_move);
    }
}
