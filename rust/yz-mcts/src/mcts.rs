//! Core PUCT MCTS (single-threaded) with stochastic transitions via `yz-core` engine.

use crate::arena::Arena;
use crate::infer::Inference;
use crate::infer_client::InferBackend;
use crate::node::{Node, NodeId};
use crate::state_key::{state_key, StateKey};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use rustc_hash::FxHashMap;
use std::collections::VecDeque;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use thiserror::Error;
use yz_core::{index_to_action, GameState, LegalMask, A};

// region agent log
#[inline]
fn dbg_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| matches!(std::env::var("YZ_DEBUG_LOG").as_deref(), Ok("1" | "true" | "yes")))
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn dbg_log_line(line: String) {
    if !dbg_enabled() {
        return;
    }
    static DBG_TX: OnceLock<std::sync::mpsc::Sender<String>> = OnceLock::new();
    fn start() -> Option<std::sync::mpsc::Sender<String>> {
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let tx_outlier = tx.clone();
        std::thread::Builder::new()
            .name("yz-mcts-debuglog".to_string())
            .spawn(move || {
                let _ = std::fs::create_dir_all(".cursor");
                let path = std::path::Path::new(".cursor").join("debug.log");
                let mut f = std::fs::OpenOptions::new().create(true).append(true).open(path);
                let mut w = f.as_mut().ok().map(|ff| std::io::BufWriter::with_capacity(1 << 20, ff));
                let mut lines_since_flush: u32 = 0;
                let mut last_flush = std::time::Instant::now();

                while let Ok(first) = rx.recv() {
                    // Batch multiple lines per wakeup to reduce syscall pressure and file contention.
                    let mut buf = String::with_capacity(first.len() * 2 + 1);
                    buf.push_str(&first);
                    buf.push('\n');
                    for _ in 0..1023 {
                        match rx.try_recv() {
                            Ok(s) => {
                                buf.push_str(&s);
                                buf.push('\n');
                            }
                            Err(_) => break,
                        }
                    }

                    let t0 = std::time::Instant::now();
                    if let Some(bw) = w.as_mut() {
                        let _ = std::io::Write::write_all(bw, buf.as_bytes());
                        lines_since_flush = lines_since_flush.saturating_add(1);
                        if lines_since_flush >= 1024 || last_flush.elapsed().as_millis() >= 250 {
                            let _ = std::io::Write::flush(bw);
                            lines_since_flush = 0;
                            last_flush = std::time::Instant::now();
                        }
                    }
                    let dt_ms = t0.elapsed().as_secs_f64() * 1000.0;
                    if dt_ms >= 5.0 {
                        // If the debug log itself is slow, enqueue a report (do NOT immediately
                        // write another line synchronously; that amplifies stalls).
                        let _ = tx_outlier.send(format!(
                            "{{\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"pre-fix\",\"hypothesisId\":\"H_rust_logio\",\"location\":\"rust/yz-mcts/src/mcts.rs:dbg_log_line\",\"message\":\"slow debug.log write\",\"data\":{{\"dt_ms\":{:.3},\"bytes\":{}}}}}",
                            now_ms(),
                            dt_ms,
                            buf.len()
                        ));
                    }
                }
            })
            .ok()?;
        Some(tx)
    }
    let tx = DBG_TX.get_or_init(|| start().unwrap_or_else(|| {
        let (tx, _rx) = std::sync::mpsc::channel::<String>();
        tx
    }));
    let _ = tx.send(line);
}
static MCTS_TICK_COUNTER: AtomicU64 = AtomicU64::new(0);
// endregion agent log

#[derive(Clone, Copy)]
pub enum ChanceMode {
    /// Self-play style: sample dice with a PRNG. Determinism not required.
    Rng { seed: u64 },
    /// Eval/gating style: deterministic event-keyed chance stream (PRD §6.1).
    Deterministic { episode_seed: u64 },
}

// --- KeepMask symmetry canonicalization (self-play only) ---------------------
//
// Motivation: KeepMask actions are position masks over `dice_sorted`. When dice contain duplicates,
// multiple masks are semantically equivalent (keep the same multiset) but occupy different action
// indices. This inflates branching and injects policy-target noise. We prune redundant KeepMasks
// only in self-play (ChanceMode::Rng), preserving eval/gating behavior and the oracle action space.

const KEEP_MASK_ACTIONS_MASK: u64 = (1u64 << 32) - 1; // actions 0..31
const ALL_ACTIONS_MASK: u64 = (1u64 << 47) - 1; // actions 0..46
const MARK_ACTIONS_MASK: u64 = ALL_ACTIONS_MASK ^ KEEP_MASK_ACTIONS_MASK; // actions 32..46

#[inline]
fn dice_key(dice_sorted: [u8; 5]) -> u32 {
    // dice are in 1..=6, fit in 3 bits each.
    (dice_sorted[0] as u32)
        | ((dice_sorted[1] as u32) << 3)
        | ((dice_sorted[2] as u32) << 6)
        | ((dice_sorted[3] as u32) << 9)
        | ((dice_sorted[4] as u32) << 12)
}

#[inline]
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

fn allowed_canonical_keepmask_bits(dice_sorted: [u8; 5]) -> u32 {
    static MAP: OnceLock<FxHashMap<u32, u32>> = OnceLock::new();
    let m = MAP.get_or_init(|| {
        let mut out: FxHashMap<u32, u32> = FxHashMap::default();
        // Enumerate all sorted dice multisets (252 combos).
        for a in 1u8..=6 {
            for b in a..=6 {
                for c in b..=6 {
                    for d in c..=6 {
                        for e in d..=6 {
                            let dice = [a, b, c, d, e];
                            let mut bits: u32 = 0;
                            // KeepMask(31) is illegal (dominated); only consider 0..=30.
                            for mask in 0u8..=30u8 {
                                if canonicalize_keepmask(dice, mask) == mask {
                                    bits |= 1u32 << (mask as u32);
                                }
                            }
                            out.insert(dice_key(dice), bits);
                        }
                    }
                }
            }
        }
        out
    });
    *m.get(&dice_key(dice_sorted)).unwrap_or(&0)
}

/// Legal action mask, with KeepMask symmetry pruning in self-play mode only.
pub fn legal_action_mask_for_mode(state: &GameState, mode: ChanceMode) -> LegalMask {
    let base = yz_core::legal_action_mask(
        state.players[state.player_to_move as usize].avail_mask,
        state.rerolls_left,
    );
    if state.rerolls_left == 0 {
        return base;
    }
    if !matches!(mode, ChanceMode::Rng { .. }) {
        return base;
    }
    let allowed = allowed_canonical_keepmask_bits(state.dice_sorted) as u64;
    let keep = (base & KEEP_MASK_ACTIONS_MASK) & allowed;
    let marks = base & MARK_ACTIONS_MASK;
    keep | marks
}

#[derive(Clone, Copy)]
pub struct MctsConfig {
    pub c_puct: f32,
    /// Simulation budget for mark decisions (rerolls_left==0).
    pub simulations_mark: u32,
    /// Simulation budget for reroll/keep decisions (rerolls_left>0).
    pub simulations_reroll: u32,
    /// Root Dirichlet alpha (self-play only).
    pub dirichlet_alpha: f32,
    /// Root Dirichlet epsilon mix-in fraction (self-play only).
    pub dirichlet_epsilon: f32,
    /// Maximum in-flight leaf evaluations per search.
    pub max_inflight: usize,
    /// Virtual-loss/inflight scheme.
    pub virtual_loss_mode: VirtualLossMode,
    /// Virtual loss to apply while a leaf is pending.
    pub virtual_loss: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VirtualLossMode {
    /// Reserve visits and subtract virtual loss from Q while pending (current behavior).
    QPenalty,
    /// Reserve visits only (affects counts / selection), but do not subtract from Q/W.
    NVirtualOnly,
    /// No reservations at all.
    Off,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            simulations_mark: 64,
            simulations_reroll: 64,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            max_inflight: 8,
            virtual_loss_mode: VirtualLossMode::QPenalty,
            virtual_loss: 1.0,
        }
    }
}

#[derive(Debug, Error)]
pub enum MctsError {
    #[error("invalid config: {msg}")]
    InvalidConfig { msg: &'static str },
}

#[derive(Default, Clone)]
pub struct SearchStats {
    pub node_count: usize,
    pub expansions: u32,
    pub fallbacks: u32,
    pub pending_count_max: usize,
    pub pending_collisions: u32,
}

pub struct SearchResult {
    pub pi: [f32; A],
    pub root_value: f32,
    pub root_priors_raw: Option<[f32; A]>,
    pub root_priors_noisy: Option<[f32; A]>,
    pub fallbacks: u32,
    pub stats: SearchStats,
}

#[derive(Debug)]
struct PendingEvalClient {
    ticket: yz_infer::Ticket,
    leaf_node: NodeId,
    leaf_state: GameState,
    path: Vec<(NodeId, usize, u8, u8)>,
}

/// Incremental, non-blocking driver for one MCTS search (PRD E7S1).
///
/// Use `tick()` with a small budget to multiplex multiple games without blocking on inference.
pub struct SearchDriver {
    root_state: GameState,
    mode: ChanceMode,
    root_id: NodeId,
    target_sims: u32,
    // region agent log
    search_started: std::time::Instant,
    submitted: u32,
    applied: u32,
    terminals: u32,
    // endregion agent log

    // Root evaluation is async.
    root_ticket: Option<yz_infer::Ticket>,
    root_priors_raw: Option<[f32; A]>,
    root_priors_noisy: Option<[f32; A]>,

    completed: u32,
    sel_counter: u64,
    pending: VecDeque<PendingEvalClient>,
}

struct PendingEval {
    leaf_node: NodeId,
    leaf_state: GameState,
    path: Vec<(NodeId, usize, u8, u8)>,
}

enum LeafSelection {
    Terminal {
        path: Vec<(NodeId, usize, u8, u8)>,
        z: f32,
    },
    Pending(PendingEval),
}

pub struct Mcts {
    cfg: MctsConfig,
    arena: Arena,
    // Child mapping per parent: (parent_node_id, action_idx, state_key(child_state)) -> child_node_id.
    children: FxHashMap<(NodeId, u8, StateKey), NodeId>,
    stats: SearchStats,
    // If true, we force the returned root `pi` to uniform-over-legal as a guardrail.
    force_uniform_root_pi: bool,
}

impl Mcts {
    pub fn new(cfg: MctsConfig) -> Result<Self, MctsError> {
        if !(cfg.c_puct.is_finite() && cfg.c_puct > 0.0) {
            return Err(MctsError::InvalidConfig {
                msg: "c_puct must be finite and > 0",
            });
        }
        if cfg.simulations_mark == 0 || cfg.simulations_reroll == 0 {
            return Err(MctsError::InvalidConfig {
                msg: "simulations_mark and simulations_reroll must be > 0",
            });
        }
        if cfg.max_inflight == 0 {
            return Err(MctsError::InvalidConfig {
                msg: "max_inflight must be > 0",
            });
        }
        if !(cfg.virtual_loss.is_finite() && cfg.virtual_loss >= 0.0) {
            return Err(MctsError::InvalidConfig {
                msg: "virtual_loss must be finite and >= 0",
            });
        }
        Ok(Self {
            cfg,
            arena: Arena::new(),
            children: FxHashMap::default(),
            stats: SearchStats::default(),
            force_uniform_root_pi: false,
        })
    }

    pub fn run_search(
        &mut self,
        root_state: GameState,
        mode: ChanceMode,
        infer: &impl Inference,
    ) -> SearchResult {
        let target_sims = if root_state.rerolls_left > 0 {
            self.cfg.simulations_reroll
        } else {
            self.cfg.simulations_mark
        };
        self.reset_tree();
        self.stats = SearchStats::default();
        self.force_uniform_root_pi = false;

        let root_id = self.arena.push(Node::new(root_state.player_to_move));
        self.stats.node_count = self.arena.len();

        // Expand root immediately (priors available for PUCT).
        let (raw_priors, _v_root, used_fallback_priors) =
            self.expand_node(root_id, &root_state, mode, infer);
        if used_fallback_priors {
            self.force_uniform_root_pi = true;
        }

        let mut root_priors_raw: Option<[f32; A]> = None;
        let mut root_priors_noisy: Option<[f32; A]> = None;

        // Root Dirichlet noise is self-play only (RNG mode).
        if let ChanceMode::Rng { seed } = mode {
            // Use a deterministic PRNG derived from the episode seed for noise itself.
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xD1E7_C437_9E37_79B9u64);
            let legal = legal_action_mask_for_mode(&root_state, mode);
            if self.cfg.dirichlet_epsilon > 0.0 {
                let noisy = apply_root_dirichlet_noise(
                    &raw_priors,
                    legal,
                    self.cfg.dirichlet_alpha,
                    self.cfg.dirichlet_epsilon,
                    &mut rng,
                );
                root_priors_raw = Some(raw_priors);
                root_priors_noisy = Some(noisy);
                self.arena.get_mut(root_id).p = noisy;
            } else {
                root_priors_raw = Some(raw_priors);
                root_priors_noisy = Some(raw_priors);
            }
        }

        // In-flight leaf pipeline scaffolding (E4S4): enqueue leaf evals up to max_inflight,
        // then drain one (still synchronous stub inference).
        let mut completed: u32 = 0;
        let mut sel_counter: u64 = 0;
        let mut pending: std::collections::VecDeque<PendingEval> =
            std::collections::VecDeque::new();

        while completed < target_sims {
            while pending.len() < self.cfg.max_inflight
                && (completed as usize + pending.len()) < (target_sims as usize)
            {
                sel_counter += 1;
                let mut ctx = match mode {
                    ChanceMode::Deterministic { episode_seed } => {
                        yz_core::TurnContext::new_deterministic(episode_seed)
                    }
                    ChanceMode::Rng { seed } => {
                        let sel_seed = seed ^ sel_counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        yz_core::TurnContext::new_rng(sel_seed)
                    }
                };

                match self.select_leaf_and_reserve(root_id, root_state, &mut ctx, mode) {
                    LeafSelection::Terminal { path, z } => {
                        self.backup_with_virtual_loss(&path, z);
                        completed += 1;
                    }
                    LeafSelection::Pending(pe) => {
                        pending.push_back(pe);
                        self.stats.pending_count_max =
                            self.stats.pending_count_max.max(pending.len());
                    }
                }
            }

            if let Some(pe) = pending.pop_front() {
                let legal = legal_action_mask_for_mode(&pe.leaf_state, mode);
                let features = yz_features::encode_state_v1(&pe.leaf_state);
                let (logits, v) = infer.eval(&features, legal);
                let (priors, _used_fallback) = masked_softmax(&logits, legal, &mut self.stats);

                let leaf = self.arena.get_mut(pe.leaf_node);
                if !leaf.is_expanded {
                    leaf.is_expanded = true;
                    leaf.to_play = pe.leaf_state.player_to_move;
                    leaf.p = priors;
                }
                self.backup_with_virtual_loss(&pe.path, v.clamp(-1.0, 1.0));
                completed += 1;
            }
        }

        let (pi, root_value) = self.root_pi_value(root_id, &root_state, mode);

        SearchResult {
            pi,
            root_value,
            root_priors_raw,
            root_priors_noisy,
            fallbacks: self.stats.fallbacks,
            stats: self.stats.clone(),
        }
    }

    pub fn run_search_with_backend(
        &mut self,
        root_state: GameState,
        mode: ChanceMode,
        backend: &InferBackend,
    ) -> SearchResult {
        let mut driver = self.begin_search_with_backend(root_state, mode, backend);
        loop {
            if let Some(res) = driver.tick(self, backend, 1024) {
                return res;
            }
        }
    }

    /// Begin a non-blocking search driven by `SearchDriver::tick()`.
    pub fn begin_search_with_backend(
        &mut self,
        root_state: GameState,
        mode: ChanceMode,
        backend: &InferBackend,
    ) -> SearchDriver {
        let target_sims = if root_state.rerolls_left > 0 {
            self.cfg.simulations_reroll
        } else {
            self.cfg.simulations_mark
        };

        // region agent log
        if dbg_enabled() {
            dbg_log_line(format!(
                "{{\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"pre-fix\",\"hypothesisId\":\"H_budget\",\"location\":\"rust/yz-mcts/src/mcts.rs:begin_search_with_backend\",\"message\":\"begin_search\",\"data\":{{\"rerolls_left\":{},\"sim_mark\":{},\"sim_reroll\":{},\"target_sims\":{}}}}}",
                now_ms(),
                root_state.rerolls_left,
                self.cfg.simulations_mark,
                self.cfg.simulations_reroll,
                target_sims
            ));
        }
        // endregion agent log

        self.reset_tree();
        self.stats = SearchStats::default();
        self.force_uniform_root_pi = false;

        let root_id = self.arena.push(Node::new(root_state.player_to_move));
        self.stats.node_count = self.arena.len();

        let legal = legal_action_mask_for_mode(&root_state, mode);
        let root_ticket = backend.submit(&root_state, legal).ok(); // if submit fails, we'll fallback in tick()

        SearchDriver {
            root_state,
            mode,
            root_id,
            target_sims,
            // region agent log
            search_started: std::time::Instant::now(),
            submitted: if root_ticket.is_some() { 1 } else { 0 },
            applied: 0,
            terminals: 0,
            // endregion agent log
            root_ticket,
            root_priors_raw: None,
            root_priors_noisy: None,
            completed: 0,
            sel_counter: 0,
            pending: VecDeque::new(),
        }
    }

    fn reset_tree(&mut self) {
        self.arena = Arena::new();
        self.children.clear();
    }

    fn select_leaf_and_reserve(
        &mut self,
        root_id: NodeId,
        root_state: GameState,
        ctx: &mut yz_core::TurnContext,
        mode: ChanceMode,
    ) -> LeafSelection {
        let mut path: Vec<(NodeId, usize, u8, u8)> = Vec::new();
        let mut node_id = root_id;
        let mut state = root_state;

        loop {
            let node_to_play = self.arena.get(node_id).to_play;

            if yz_core::is_terminal(&state) {
                let z = yz_core::terminal_z_from_player_to_move(&state).unwrap_or(0.0);
                // Reserve path (even though terminal) to keep accounting consistent.
                self.apply_virtual_loss_path(&path);
                return LeafSelection::Terminal { path, z };
            }

            if !self.arena.get(node_id).is_expanded {
                // Leaf node to be evaluated.
                self.apply_virtual_loss_path(&path);
                let pe = PendingEval {
                    leaf_node: node_id,
                    leaf_state: state,
                    path,
                };
                return LeafSelection::Pending(pe);
            }

            let legal = legal_action_mask_for_mode(&state, mode);
            let a_idx = self.select_action(node_id, legal, mode) as usize;
            let action = index_to_action(a_idx as u8);
            let next_state = match yz_core::apply_action(state, action, ctx) {
                Ok(s2) => s2,
                Err(_) => {
                    self.stats.fallbacks += 1;
                    self.apply_virtual_loss_path(&path);
                    return LeafSelection::Terminal { path, z: 0.0 };
                }
            };

            let child_key = state_key(&next_state);
            let child_id = if let Some(&cid) = self.children.get(&(node_id, a_idx as u8, child_key))
            {
                cid
            } else {
                let cid = self.arena.push(Node::new(next_state.player_to_move));
                self.children.insert((node_id, a_idx as u8, child_key), cid);
                self.stats.node_count = self.arena.len();
                cid
            };

            path.push((node_id, a_idx, node_to_play, next_state.player_to_move));
            node_id = child_id;
            state = next_state;
        }
    }

    fn apply_virtual_loss_path(&mut self, path: &[(NodeId, usize, u8, u8)]) {
        if self.cfg.virtual_loss_mode == VirtualLossMode::Off {
            return;
        }
        if path.is_empty() {
            return;
        }
        for &(node_id, a_idx, _pt, _ct) in path {
            let n = self.arena.get_mut(node_id);
            n.vl_n[a_idx] = n.vl_n[a_idx].saturating_add(1);
            if self.cfg.virtual_loss_mode == VirtualLossMode::QPenalty {
                n.vl_w[a_idx] += self.cfg.virtual_loss;
            }
            n.vl_sum = n.vl_sum.saturating_add(1);
        }
    }

    fn remove_virtual_loss_path(&mut self, path: &[(NodeId, usize, u8, u8)]) {
        if self.cfg.virtual_loss_mode == VirtualLossMode::Off {
            return;
        }
        for &(node_id, a_idx, _pt, _ct) in path.iter().rev() {
            let n = self.arena.get_mut(node_id);
            n.vl_n[a_idx] = n.vl_n[a_idx].saturating_sub(1);
            if self.cfg.virtual_loss_mode == VirtualLossMode::QPenalty {
                n.vl_w[a_idx] -= self.cfg.virtual_loss;
            }
            n.vl_sum = n.vl_sum.saturating_sub(1);
        }
    }

    fn backup_with_virtual_loss(&mut self, path: &[(NodeId, usize, u8, u8)], v_leaf: f32) {
        if self.cfg.virtual_loss_mode != VirtualLossMode::Off {
            self.remove_virtual_loss_path(path);
        }
        self.backup(path, v_leaf);
    }

    fn expand_node(
        &mut self,
        node_id: NodeId,
        state: &GameState,
        mode: ChanceMode,
        infer: &impl Inference,
    ) -> ([f32; A], f32, bool) {
        let legal = legal_action_mask_for_mode(state, mode);
        let features = yz_features::encode_state_v1(state);
        let (logits, v) = infer.eval(&features, legal);

        let (priors, used_fallback) = masked_softmax(&logits, legal, &mut self.stats);

        let n = self.arena.get_mut(node_id);
        n.is_expanded = true;
        n.to_play = state.player_to_move;
        n.p = priors;

        self.stats.expansions += 1;
        (priors, v.clamp(-1.0, 1.0), used_fallback)
    }

    fn apply_infer_response(
        &mut self,
        leaf_node: NodeId,
        leaf_state: &GameState,
        path: &[(NodeId, usize, u8, u8)],
        mode: ChanceMode,
        resp: yz_infer::protocol::InferResponseV1,
    ) {
        let legal = legal_action_mask_for_mode(leaf_state, mode);
        let logits = vec_logits_to_array(&resp.policy_logits);
        let (priors, _used_fallback) = masked_softmax(&logits, legal, &mut self.stats);

        let leaf = self.arena.get_mut(leaf_node);
        if !leaf.is_expanded {
            leaf.is_expanded = true;
            leaf.to_play = leaf_state.player_to_move;
            leaf.p = priors;
            self.stats.expansions += 1;
        }

        self.backup_with_virtual_loss(path, resp.value.clamp(-1.0, 1.0));
    }

    fn select_action(&mut self, node_id: NodeId, legal: LegalMask, mode: ChanceMode) -> u8 {
        let n = self.arena.get(node_id);
        let use_virtual_counts = self.cfg.virtual_loss_mode != VirtualLossMode::Off;
        let use_q_penalty = self.cfg.virtual_loss_mode == VirtualLossMode::QPenalty;
        let n_sum_eff = if use_virtual_counts {
            n.n_sum.saturating_add(n.vl_sum)
        } else {
            n.n_sum
        };
        let sqrt_sum = (n_sum_eff as f32).sqrt();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_a: u8 = 0;

        for a in 0..A {
            if !legal_ok(legal, a) {
                continue;
            }
            let q = if use_q_penalty { n.q_eff(a, true) } else { n.q(a) };
            let n_eff = if use_virtual_counts {
                n.n[a].saturating_add(n.vl_n[a]) as f32
            } else {
                n.n[a] as f32
            };
            let u = self.cfg.c_puct * n.p[a] * sqrt_sum / (1.0 + n_eff);
            let score = q + u;

            if score > best_score {
                best_score = score;
                best_a = a as u8;
            } else if score == best_score {
                // Deterministic tie-break in eval mode.
                if matches!(mode, ChanceMode::Deterministic { .. }) && (a as u8) < best_a {
                    best_a = a as u8;
                }
            }
        }

        // Collision: chose an edge that already has a pending reservation.
        if use_virtual_counts && self.arena.get(node_id).vl_n[best_a as usize] > 0 {
            self.stats.pending_collisions += 1;
        }

        best_a
    }

    fn backup(&mut self, path: &[(NodeId, usize, u8, u8)], mut v_leaf: f32) {
        // v_leaf is from POV of the leaf state's player_to_move.
        for &(node_id, a_idx, parent_to_play, child_to_play) in path.iter().rev() {
            // Express value in the parent node's POV.
            let mut v_parent = v_leaf;
            if parent_to_play != child_to_play {
                v_parent = -v_parent;
            }

            let n = self.arena.get_mut(node_id);
            n.n[a_idx] += 1;
            n.w[a_idx] += v_parent;
            n.n_sum += 1;

            // For next step up, the leaf value POV becomes the parent's POV at this edge.
            v_leaf = v_parent;
        }
    }

    fn root_pi_value(&self, root_id: NodeId, state: &GameState, mode: ChanceMode) -> ([f32; A], f32) {
        let root = self.arena.get(root_id);
        let legal = legal_action_mask_for_mode(state, mode);

        if self.force_uniform_root_pi {
            // Guardrail: if priors were invalid at root (fallback), return uniform-over-legal `pi`.
            let pi = uniform_over_legal(legal);
            return (pi, 0.0);
        }

        let mut pi = [0.0f32; A];
        let mut sum = 0.0f32;
        for a in 0..A {
            if legal_ok(legal, a) {
                let v = root.n[a] as f32;
                pi[a] = v;
                sum += v;
            }
        }
        if sum <= 0.0 {
            // fallback uniform
            // count as a fallback event (pi degenerate)
            // (we can't mutate self.stats here; this is a pure read path)
            let mut cnt = 0usize;
            for a in 0..A {
                if legal_ok(legal, a) {
                    cnt += 1;
                }
            }
            if cnt > 0 {
                let u = 1.0 / (cnt as f32);
                for a in 0..A {
                    if legal_ok(legal, a) {
                        pi[a] = u;
                    }
                }
            }
        } else {
            for v in &mut pi {
                *v /= sum;
            }
        }

        // Root value estimate: mean Q over legal actions weighted by visits (or 0 if none).
        let mut v = 0.0f32;
        if sum > 0.0 {
            for a in 0..A {
                if legal_ok(legal, a) {
                    v += (root.n[a] as f32) * root.q(a);
                }
            }
            v /= sum;
        }

        (pi, v.clamp(-1.0, 1.0))
    }
}

/// Benchmark-only helper to measure the PUCT selection loop cost without running full searches.
///
/// This API is behind the `bench` feature and has **no stability guarantees**.
#[cfg(feature = "bench")]
pub fn bench_select_action_v1(
    cfg: &MctsConfig,
    node: &Node,
    legal: LegalMask,
    mode: ChanceMode,
) -> u8 {
    let use_virtual_counts = cfg.virtual_loss_mode != VirtualLossMode::Off;
    let use_q_penalty = cfg.virtual_loss_mode == VirtualLossMode::QPenalty;
    let n_sum_eff = if use_virtual_counts {
        node.n_sum.saturating_add(node.vl_sum)
    } else {
        node.n_sum
    };
    let sqrt_sum = (n_sum_eff as f32).sqrt();

    let mut best_score = f32::NEG_INFINITY;
    let mut best_a: u8 = 0;

    for a in 0..A {
        if !legal_ok(legal, a) {
            continue;
        }
        let q = if use_q_penalty { node.q_eff(a, true) } else { node.q(a) };
        let n_eff = if use_virtual_counts {
            node.n[a].saturating_add(node.vl_n[a]) as f32
        } else {
            node.n[a] as f32
        };
        let u = cfg.c_puct * node.p[a] * sqrt_sum / (1.0 + n_eff);
        let score = q + u;

        if score > best_score {
            best_score = score;
            best_a = a as u8;
        } else if score == best_score {
            if matches!(mode, ChanceMode::Deterministic { .. }) && (a as u8) < best_a {
                best_a = a as u8;
            }
        }
    }

    best_a
}

impl SearchDriver {
    /// Advance the search by up to `max_work` small operations.
    ///
    /// Returns `Some(SearchResult)` when the search is complete; otherwise `None`.
    pub fn tick(
        &mut self,
        mcts: &mut Mcts,
        backend: &InferBackend,
        max_work: u32,
    ) -> Option<SearchResult> {
        // region agent log
        let dbg = dbg_enabled();
        let tick_id = if dbg {
            MCTS_TICK_COUNTER.fetch_add(1, AtomicOrdering::Relaxed)
        } else {
            0
        };
        let t_tick0 = std::time::Instant::now();
        let mut n_submit: u32 = 0;
        let mut n_apply: u32 = 0;
        let mut n_term: u32 = 0;
        let mut ret_reason: &'static str = "work_exhausted";

        fn maybe_emit_slow_tick(
            tick_started: std::time::Instant,
            dbg: bool,
            tick_id: u64,
            max_work: u32,
            pending_len: usize,
            completed: u32,
            target_sims: u32,
            n_submit: u32,
            n_apply: u32,
            n_term: u32,
            ret_reason: &'static str,
        ) {
            let dt_ms = tick_started.elapsed().as_secs_f64() * 1000.0;
            if dt_ms < 2.0 {
                return;
            }
            if dbg {
                dbg_log_line(format!(
                    "{{\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"pre-fix\",\"hypothesisId\":\"H_mcts_tick\",\"location\":\"rust/yz-mcts/src/mcts.rs:SearchDriver::tick\",\"message\":\"slow tick\",\"data\":{{\"tick_id\":{},\"dt_ms\":{:.3},\"max_work\":{},\"pending\":{},\"completed\":{},\"target_sims\":{},\"n_submit\":{},\"n_apply\":{},\"n_term\":{},\"ret_reason\":\"{}\"}}}}",
                    now_ms(),
                    tick_id,
                    dt_ms,
                    max_work,
                    pending_len,
                    completed,
                    target_sims,
                    n_submit,
                    n_apply,
                    n_term,
                    ret_reason
                ));
            }
        }
        // endregion agent log

        for _ in 0..max_work {
            // Ensure root is expanded before doing any selections.
            if !mcts.arena.get(self.root_id).is_expanded {
                if !self.poll_root(mcts) {
                    // Root not ready yet; non-blocking.
                    // region agent log
                    if dbg {
                        ret_reason = "root_not_ready";
                    }
                    // endregion agent log
                    // region agent log
                    if dbg {
                        maybe_emit_slow_tick(
                            t_tick0,
                            dbg,
                            tick_id,
                            max_work,
                            self.pending.len(),
                            self.completed,
                            self.target_sims,
                            n_submit,
                            n_apply,
                            n_term,
                            ret_reason,
                        );
                    }
                    // endregion agent log
                    return None;
                }
                continue;
            }

            if self.completed >= self.target_sims {
                // region agent log
                if dbg {
                    ret_reason = "done";
                }
                // endregion agent log
                // region agent log
                if dbg {
                    maybe_emit_slow_tick(
                        t_tick0,
                            dbg,
                        tick_id,
                        max_work,
                        self.pending.len(),
                        self.completed,
                        self.target_sims,
                        n_submit,
                        n_apply,
                        n_term,
                        ret_reason,
                    );
                }
                // endregion agent log
                // region agent log
                if dbg {
                    let dur_ms = self.search_started.elapsed().as_secs_f64() * 1000.0;
                    dbg_log_line(format!(
                        "{{\"timestamp\":{},\"sessionId\":\"debug-session\",\"runId\":\"pre-fix\",\"hypothesisId\":\"H_search\",\"location\":\"rust/yz-mcts/src/mcts.rs:SearchDriver::tick\",\"message\":\"search_done\",\"data\":{{\"duration_ms\":{:.3},\"target_sims\":{},\"completed\":{},\"submitted\":{},\"applied\":{},\"terminals\":{},\"fallbacks\":{},\"expansions\":{},\"node_count\":{},\"pending_max\":{}}}}}",
                        now_ms(),
                        dur_ms,
                        self.target_sims,
                        self.completed,
                        self.submitted,
                        self.applied,
                        self.terminals,
                        mcts.stats.fallbacks,
                        mcts.stats.expansions,
                        mcts.stats.node_count,
                        mcts.stats.pending_count_max
                    ));
                }
                // endregion agent log
                return Some(self.finish(mcts));
            }

            // Enqueue one leaf selection if we are under inflight cap and budget.
            if self.pending.len() < mcts.cfg.max_inflight
                && (self.completed as usize + self.pending.len()) < (self.target_sims as usize)
            {
                self.sel_counter += 1;
                let mut ctx = match self.mode {
                    ChanceMode::Deterministic { episode_seed } => {
                        yz_core::TurnContext::new_deterministic(episode_seed)
                    }
                    ChanceMode::Rng { seed } => {
                        let sel_seed = seed ^ self.sel_counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        yz_core::TurnContext::new_rng(sel_seed)
                    }
                };

                match mcts.select_leaf_and_reserve(
                    self.root_id,
                    self.root_state,
                    &mut ctx,
                    self.mode,
                ) {
                    LeafSelection::Terminal { path, z } => {
                        mcts.backup_with_virtual_loss(&path, z);
                        self.completed += 1;
                        // region agent log
                        if dbg {
                            n_term += 1;
                            self.terminals += 1;
                        }
                        // endregion agent log
                    }
                    LeafSelection::Pending(pe) => {
                        let legal = legal_action_mask_for_mode(&pe.leaf_state, self.mode);
                        let ticket = match backend.submit(&pe.leaf_state, legal) {
                            Ok(t) => t,
                            Err(_) => {
                                mcts.backup_with_virtual_loss(&pe.path, 0.0);
                                self.completed += 1;
                                continue;
                            }
                        };
                        self.pending.push_back(PendingEvalClient {
                            ticket,
                            leaf_node: pe.leaf_node,
                            leaf_state: pe.leaf_state,
                            path: pe.path,
                        });
                        mcts.stats.pending_count_max =
                            mcts.stats.pending_count_max.max(self.pending.len());
                        // region agent log
                        if dbg {
                            n_submit += 1;
                            self.submitted += 1;
                        }
                        // endregion agent log
                    }
                }
                continue;
            }

            // Otherwise, try to drain a completed inference response without blocking.
            if self.pending.is_empty() {
                // Nothing to do (should be rare); yield to scheduler.
                // region agent log
                if dbg {
                    ret_reason = "no_pending";
                }
                // endregion agent log
                // region agent log
                if dbg {
                    maybe_emit_slow_tick(
                        t_tick0,
                        dbg,
                        tick_id,
                        max_work,
                        self.pending.len(),
                        self.completed,
                        self.target_sims,
                        n_submit,
                        n_apply,
                        n_term,
                        ret_reason,
                    );
                }
                // endregion agent log
                return None;
            }

            let mut processed = false;
            for _ in 0..self.pending.len() {
                let pe = self.pending.pop_front().expect("pending non-empty");
                match pe.ticket.try_recv() {
                    Ok(Some(resp)) => {
                        mcts.apply_infer_response(
                            pe.leaf_node,
                            &pe.leaf_state,
                            &pe.path,
                            self.mode,
                            resp,
                        );
                        self.completed += 1;
                        processed = true;
                        // region agent log
                        if dbg {
                            n_apply += 1;
                            self.applied += 1;
                        }
                        // endregion agent log
                        break;
                    }
                    Ok(None) => self.pending.push_back(pe),
                    Err(_) => {
                        mcts.backup_with_virtual_loss(&pe.path, 0.0);
                        self.completed += 1;
                        processed = true;
                        break;
                    }
                }
            }
            if !processed {
                // No response ready; non-blocking.
                // region agent log
                if dbg {
                    ret_reason = "no_resp_ready";
                }
                // endregion agent log
                // region agent log
                if dbg {
                    maybe_emit_slow_tick(
                        t_tick0,
                        dbg,
                        tick_id,
                        max_work,
                        self.pending.len(),
                        self.completed,
                        self.target_sims,
                        n_submit,
                        n_apply,
                        n_term,
                        ret_reason,
                    );
                }
                // endregion agent log
                return None;
            }
        }
        // region agent log
        if dbg {
            maybe_emit_slow_tick(
                t_tick0,
                dbg,
                tick_id,
                max_work,
                self.pending.len(),
                self.completed,
                self.target_sims,
                n_submit,
                n_apply,
                n_term,
                ret_reason,
            );
        }
        // endregion agent log
        None
    }

    fn poll_root(&mut self, mcts: &mut Mcts) -> bool {
        // If root submit failed, fallback to uniform priors and proceed.
        let legal = legal_action_mask_for_mode(&self.root_state, self.mode);

        let Some(ticket) = &self.root_ticket else {
            mcts.stats.fallbacks += 1;
            let priors = uniform_over_legal(legal);
            let n = mcts.arena.get_mut(self.root_id);
            n.is_expanded = true;
            n.to_play = self.root_state.player_to_move;
            n.p = priors;
            mcts.stats.expansions += 1;
            mcts.force_uniform_root_pi = true;
            self.root_priors_raw = Some(priors);
            self.root_priors_noisy = Some(priors);
            return true;
        };

        match ticket.try_recv() {
            Ok(Some(resp)) => {
                let logits = vec_logits_to_array(&resp.policy_logits);
                let (mut priors, used_fallback) = masked_softmax(&logits, legal, &mut mcts.stats);
                if used_fallback {
                    mcts.force_uniform_root_pi = true;
                }

                self.root_priors_raw = Some(priors);
                self.root_priors_noisy = Some(priors);

                // Apply root Dirichlet noise only in RNG mode.
                if let ChanceMode::Rng { seed } = self.mode {
                    if mcts.cfg.dirichlet_epsilon > 0.0 {
                        let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xD1E7_C437_9E37_79B9u64);
                        let noisy = apply_root_dirichlet_noise(
                            &priors,
                            legal,
                            mcts.cfg.dirichlet_alpha,
                            mcts.cfg.dirichlet_epsilon,
                            &mut rng,
                        );
                        self.root_priors_noisy = Some(noisy);
                        priors = noisy;
                    }
                }

                let n = mcts.arena.get_mut(self.root_id);
                n.is_expanded = true;
                n.to_play = self.root_state.player_to_move;
                n.p = priors;
                mcts.stats.expansions += 1;
                true
            }
            Ok(None) => false,
            Err(_) => {
                // Treat as fallback.
                mcts.stats.fallbacks += 1;
                let priors = uniform_over_legal(legal);
                let n = mcts.arena.get_mut(self.root_id);
                n.is_expanded = true;
                n.to_play = self.root_state.player_to_move;
                n.p = priors;
                mcts.stats.expansions += 1;
                mcts.force_uniform_root_pi = true;
                self.root_priors_raw = Some(priors);
                self.root_priors_noisy = Some(priors);
                true
            }
        }
    }

    fn finish(&self, mcts: &Mcts) -> SearchResult {
        let (pi, root_value) = mcts.root_pi_value(self.root_id, &self.root_state, self.mode);
        SearchResult {
            pi,
            root_value,
            root_priors_raw: self.root_priors_raw,
            root_priors_noisy: self.root_priors_noisy,
            fallbacks: mcts.stats.fallbacks,
            stats: mcts.stats.clone(),
        }
    }
}

fn vec_logits_to_array(v: &[f32]) -> [f32; A] {
    let mut out = [0.0f32; A];
    if v.len() == A {
        out.copy_from_slice(v);
        return out;
    }
    let n = v.len().min(A);
    out[..n].copy_from_slice(&v[..n]);
    out
}

#[inline]
fn legal_ok(legal: LegalMask, idx: usize) -> bool {
    ((legal >> idx) & 1) != 0
}

fn masked_softmax(
    logits: &[f32; A],
    legal: LegalMask,
    stats: &mut SearchStats,
) -> ([f32; A], bool) {
    let mut out = [0.0f32; A];

    let mut max = f32::NEG_INFINITY;
    for a in 0..A {
        if legal_ok(legal, a) && logits[a].is_finite() {
            max = max.max(logits[a]);
        }
    }
    if !max.is_finite() {
        stats.fallbacks += 1;
        return (uniform_over_legal(legal), true);
    }

    let mut sum = 0.0f32;
    for a in 0..A {
        if legal_ok(legal, a) {
            let z = (logits[a] - max).exp();
            if z.is_finite() {
                out[a] = z;
                sum += z;
            }
        }
    }

    if !(sum.is_finite() && sum > 0.0) {
        stats.fallbacks += 1;
        return (uniform_over_legal(legal), true);
    }

    for v in &mut out {
        *v /= sum;
    }
    (out, false)
}

fn uniform_over_legal(legal: LegalMask) -> [f32; A] {
    let mut out = [0.0f32; A];
    let mut cnt = 0usize;
    for i in 0..A {
        if legal_ok(legal, i) {
            cnt += 1;
        }
    }
    if cnt == 0 {
        return out;
    }
    let u = 1.0 / (cnt as f32);
    for i in 0..A {
        if legal_ok(legal, i) {
            out[i] = u;
        }
    }
    out
}

fn apply_root_dirichlet_noise(
    p_raw: &[f32; A],
    legal: LegalMask,
    alpha: f32,
    eps: f32,
    rng: &mut impl Rng,
) -> [f32; A] {
    if !(alpha.is_finite() && alpha > 0.0 && eps.is_finite() && (0.0..=1.0).contains(&eps)) {
        return *p_raw;
    }

    // Sample gamma(alpha, 1) for each legal action, then normalize -> Dirichlet.
    let gamma = Gamma::new(alpha as f64, 1.0).expect("alpha>0");
    let mut eta = [0.0f32; A];
    let mut sum = 0.0f64;
    for i in 0..A {
        if legal_ok(legal, i) {
            let x = gamma.sample(rng);
            eta[i] = x as f32;
            sum += x;
        }
    }
    if !(sum.is_finite() && sum > 0.0) {
        return *p_raw;
    }
    for v in &mut eta {
        *v = (*v as f64 / sum) as f32;
    }

    // Mix.
    let mut out = [0.0f32; A];
    for i in 0..A {
        if legal_ok(legal, i) {
            out[i] = (1.0 - eps) * p_raw[i] + eps * eta[i];
        } else {
            out[i] = 0.0;
        }
    }
    out
}

pub fn apply_temperature(pi_target: &[f32; A], legal: LegalMask, t: f32) -> [f32; A] {
    // Executed-move distribution only. Caller chooses how/when to sample.
    if !t.is_finite() || t < 0.0 {
        return uniform_over_legal(legal);
    }
    if t == 0.0 {
        // Greedy argmax with deterministic tie-break (lowest idx).
        let mut best = None::<(usize, f32)>;
        for i in 0..A {
            if !legal_ok(legal, i) {
                continue;
            }
            let v = pi_target[i];
            if let Some((bi, bv)) = best {
                if v > bv || (v == bv && i < bi) {
                    best = Some((i, v));
                }
            } else {
                best = Some((i, v));
            }
        }
        let mut out = [0.0f32; A];
        if let Some((i, _)) = best {
            out[i] = 1.0;
        }
        return out;
    }

    let inv_t = 1.0 / t;
    let mut out = [0.0f32; A];
    let mut sum = 0.0f32;
    for i in 0..A {
        if legal_ok(legal, i) {
            let v = pi_target[i].max(0.0);
            let w = v.powf(inv_t);
            out[i] = w;
            sum += w;
        }
    }
    if !(sum.is_finite() && sum > 0.0) {
        return uniform_over_legal(legal);
    }
    for v in &mut out {
        *v /= sum;
    }
    out
}
