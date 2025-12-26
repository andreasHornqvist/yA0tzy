//! Core PUCT MCTS (single-threaded) with stochastic transitions via `yz-core` engine.

use crate::arena::Arena;
use crate::infer::Inference;
use crate::node::{Node, NodeId};
use crate::state_key::{state_key, StateKey};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Gamma};
use rustc_hash::FxHashMap;
use thiserror::Error;
use yz_core::{index_to_action, legal_action_mask, GameState, A};

type Ticket = u64;

#[derive(Clone, Copy)]
pub enum ChanceMode {
    /// Self-play style: sample dice with a PRNG. Determinism not required.
    Rng { seed: u64 },
    /// Eval/gating style: deterministic event-keyed chance stream (PRD §6.1).
    Deterministic { episode_seed: u64 },
}

#[derive(Clone, Copy)]
pub struct MctsConfig {
    pub c_puct: f32,
    pub simulations: u32,
    /// Root Dirichlet alpha (self-play only).
    pub dirichlet_alpha: f32,
    /// Root Dirichlet epsilon mix-in fraction (self-play only).
    pub dirichlet_epsilon: f32,
    /// Maximum in-flight leaf evaluations per search.
    pub max_inflight: usize,
    /// Virtual loss to apply while a leaf is pending.
    pub virtual_loss: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            simulations: 64,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
            max_inflight: 8,
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

struct PendingEval {
    ticket: Ticket,
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
        if cfg.simulations == 0 {
            return Err(MctsError::InvalidConfig {
                msg: "simulations must be > 0",
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
        self.stats = SearchStats::default();
        self.force_uniform_root_pi = false;

        let root_id = self.arena.push(Node::new(root_state.player_to_move));
        self.stats.node_count = self.arena.len();

        // Expand root immediately (priors available for PUCT).
        let (raw_priors, _v_root, used_fallback_priors) =
            self.expand_node(root_id, &root_state, infer);
        if used_fallback_priors {
            self.force_uniform_root_pi = true;
        }

        let mut root_priors_raw: Option<[f32; A]> = None;
        let mut root_priors_noisy: Option<[f32; A]> = None;

        // Root Dirichlet noise is self-play only (RNG mode).
        if let ChanceMode::Rng { seed } = mode {
            // Use a deterministic PRNG derived from the episode seed for noise itself.
            let mut rng = ChaCha8Rng::seed_from_u64(seed ^ 0xD1E7_C437_9E37_79B9u64);
            let legal = legal_action_mask(
                root_state.players[root_state.player_to_move as usize].avail_mask,
                root_state.rerolls_left,
            );
            if self.cfg.dirichlet_epsilon > 0.0 {
                let noisy = apply_root_dirichlet_noise(
                    &raw_priors,
                    &legal,
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
        let mut ticket: Ticket = 0;
        let mut pending: std::collections::VecDeque<PendingEval> =
            std::collections::VecDeque::new();

        while completed < self.cfg.simulations {
            while pending.len() < self.cfg.max_inflight
                && (completed as usize + pending.len()) < (self.cfg.simulations as usize)
            {
                ticket += 1;
                let mut ctx = match mode {
                    ChanceMode::Deterministic { episode_seed } => {
                        yz_core::TurnContext::new_deterministic(episode_seed)
                    }
                    ChanceMode::Rng { seed } => {
                        let sel_seed = seed ^ ticket.wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        yz_core::TurnContext::new_rng(sel_seed)
                    }
                };

                match self.select_leaf_and_reserve(root_id, root_state, &mut ctx, mode) {
                    LeafSelection::Terminal { path, z } => {
                        self.backup_with_virtual_loss(&path, z);
                        completed += 1;
                    }
                    LeafSelection::Pending(mut pe) => {
                        pe.ticket = ticket;
                        pending.push_back(pe);
                        self.stats.pending_count_max =
                            self.stats.pending_count_max.max(pending.len());
                    }
                }
            }

            if let Some(pe) = pending.pop_front() {
                let legal = legal_action_mask(
                    pe.leaf_state.players[pe.leaf_state.player_to_move as usize].avail_mask,
                    pe.leaf_state.rerolls_left,
                );
                let features = yz_features::encode_state_v1(&pe.leaf_state);
                let (logits, v) = infer.eval(&features, &legal);
                let (priors, _used_fallback) = masked_softmax(&logits, &legal, &mut self.stats);

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

        let (pi, root_value) = self.root_pi_value(root_id, &root_state);

        SearchResult {
            pi,
            root_value,
            root_priors_raw,
            root_priors_noisy,
            fallbacks: self.stats.fallbacks,
            stats: self.stats.clone(),
        }
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
                    ticket: 0,
                    leaf_node: node_id,
                    leaf_state: state,
                    path,
                };
                return LeafSelection::Pending(pe);
            }

            let legal = legal_action_mask(
                state.players[state.player_to_move as usize].avail_mask,
                state.rerolls_left,
            );
            let a_idx = self.select_action(node_id, &legal, mode) as usize;
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
        if path.is_empty() {
            return;
        }
        for &(node_id, a_idx, _pt, _ct) in path {
            let n = self.arena.get_mut(node_id);
            n.vl_n[a_idx] = n.vl_n[a_idx].saturating_add(1);
            n.vl_w[a_idx] += self.cfg.virtual_loss;
            n.vl_sum = n.vl_sum.saturating_add(1);
        }
    }

    fn remove_virtual_loss_path(&mut self, path: &[(NodeId, usize, u8, u8)]) {
        for &(node_id, a_idx, _pt, _ct) in path.iter().rev() {
            let n = self.arena.get_mut(node_id);
            n.vl_n[a_idx] = n.vl_n[a_idx].saturating_sub(1);
            n.vl_w[a_idx] -= self.cfg.virtual_loss;
            n.vl_sum = n.vl_sum.saturating_sub(1);
        }
    }

    fn backup_with_virtual_loss(&mut self, path: &[(NodeId, usize, u8, u8)], v_leaf: f32) {
        self.remove_virtual_loss_path(path);
        self.backup(path, v_leaf);
    }

    fn expand_node(
        &mut self,
        node_id: NodeId,
        state: &GameState,
        infer: &impl Inference,
    ) -> ([f32; A], f32, bool) {
        let legal = legal_action_mask(
            state.players[state.player_to_move as usize].avail_mask,
            state.rerolls_left,
        );
        let features = yz_features::encode_state_v1(state);
        let (logits, v) = infer.eval(&features, &legal);

        let (priors, used_fallback) = masked_softmax(&logits, &legal, &mut self.stats);

        let n = self.arena.get_mut(node_id);
        n.is_expanded = true;
        n.to_play = state.player_to_move;
        n.p = priors;

        self.stats.expansions += 1;
        (priors, v.clamp(-1.0, 1.0), used_fallback)
    }

    fn select_action(&mut self, node_id: NodeId, legal: &[bool; A], mode: ChanceMode) -> u8 {
        let n = self.arena.get(node_id);
        let use_vl = self.cfg.virtual_loss > 0.0;
        let n_sum_eff = if use_vl {
            n.n_sum.saturating_add(n.vl_sum)
        } else {
            n.n_sum
        };
        let sqrt_sum = (n_sum_eff as f32).sqrt();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_a: u8 = 0;

        for (a, &ok) in legal.iter().enumerate() {
            if !ok {
                continue;
            }
            let q = n.q_eff(a, use_vl);
            let n_eff = if use_vl {
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
        if self.arena.get(node_id).vl_n[best_a as usize] > 0 {
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

    fn root_pi_value(&self, root_id: NodeId, state: &GameState) -> ([f32; A], f32) {
        let root = self.arena.get(root_id);
        let legal = legal_action_mask(
            state.players[state.player_to_move as usize].avail_mask,
            state.rerolls_left,
        );

        if self.force_uniform_root_pi {
            // Guardrail: if priors were invalid at root (fallback), return uniform-over-legal `pi`.
            let pi = uniform_over_legal(&legal);
            return (pi, 0.0);
        }

        let mut pi = [0.0f32; A];
        let mut sum = 0.0f32;
        for (a, &ok) in legal.iter().enumerate() {
            if ok {
                let v = root.n[a] as f32;
                pi[a] = v;
                sum += v;
            }
        }
        if sum <= 0.0 {
            // fallback uniform
            // count as a fallback event (pi degenerate)
            // (we can't mutate self.stats here; this is a pure read path)
            let cnt = legal.iter().filter(|&&ok| ok).count();
            if cnt > 0 {
                let u = 1.0 / (cnt as f32);
                for (a, &ok) in legal.iter().enumerate() {
                    if ok {
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
            for (a, &ok) in legal.iter().enumerate() {
                if ok {
                    v += (root.n[a] as f32) * root.q(a);
                }
            }
            v /= sum;
        }

        (pi, v.clamp(-1.0, 1.0))
    }
}

fn masked_softmax(
    logits: &[f32; A],
    legal: &[bool; A],
    stats: &mut SearchStats,
) -> ([f32; A], bool) {
    let mut out = [0.0f32; A];

    let mut max = f32::NEG_INFINITY;
    for (a, &ok) in legal.iter().enumerate() {
        if ok && logits[a].is_finite() {
            max = max.max(logits[a]);
        }
    }
    if !max.is_finite() {
        stats.fallbacks += 1;
        return (uniform_over_legal(legal), true);
    }

    let mut sum = 0.0f32;
    for a in 0..A {
        if legal[a] {
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

fn uniform_over_legal(legal: &[bool; A]) -> [f32; A] {
    let mut out = [0.0f32; A];
    let mut cnt = 0usize;
    for &ok in legal {
        if ok {
            cnt += 1;
        }
    }
    if cnt == 0 {
        return out;
    }
    let u = 1.0 / (cnt as f32);
    for (i, &ok) in legal.iter().enumerate() {
        if ok {
            out[i] = u;
        }
    }
    out
}

fn apply_root_dirichlet_noise(
    p_raw: &[f32; A],
    legal: &[bool; A],
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
    for (i, &ok) in legal.iter().enumerate() {
        if ok {
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
    for (i, &ok) in legal.iter().enumerate() {
        if ok {
            out[i] = (1.0 - eps) * p_raw[i] + eps * eta[i];
        } else {
            out[i] = 0.0;
        }
    }
    out
}

pub fn apply_temperature(pi_target: &[f32; A], legal: &[bool; A], t: f32) -> [f32; A] {
    // Executed-move distribution only. Caller chooses how/when to sample.
    if !t.is_finite() || t < 0.0 {
        return uniform_over_legal(legal);
    }
    if t == 0.0 {
        // Greedy argmax with deterministic tie-break (lowest idx).
        let mut best = None::<(usize, f32)>;
        for (i, &ok) in legal.iter().enumerate() {
            if !ok {
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
    for (i, &ok) in legal.iter().enumerate() {
        if ok {
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
