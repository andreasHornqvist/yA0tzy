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
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            c_puct: 1.5,
            simulations: 64,
            dirichlet_alpha: 0.3,
            dirichlet_epsilon: 0.25,
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
}

pub struct SearchResult {
    pub pi: [f32; A],
    pub root_value: f32,
    pub root_priors_raw: Option<[f32; A]>,
    pub root_priors_noisy: Option<[f32; A]>,
    pub stats: SearchStats,
}

pub struct Mcts {
    cfg: MctsConfig,
    arena: Arena,
    // Child mapping per parent: (parent_node_id, action_idx, state_key(child_state)) -> child_node_id.
    children: FxHashMap<(NodeId, u8, StateKey), NodeId>,
    stats: SearchStats,
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
        Ok(Self {
            cfg,
            arena: Arena::new(),
            children: FxHashMap::default(),
            stats: SearchStats::default(),
        })
    }

    pub fn run_search(
        &mut self,
        root_state: GameState,
        mode: ChanceMode,
        infer: &impl Inference,
    ) -> SearchResult {
        self.stats = SearchStats::default();

        let root_id = self.arena.push(Node::new(root_state.player_to_move));
        self.stats.node_count = self.arena.len();

        // Expand root immediately (priors available for PUCT).
        let (raw_priors, _v_root) = self.expand_node(root_id, &root_state, infer);

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

        for sim in 0..self.cfg.simulations {
            let mut ctx = match mode {
                ChanceMode::Deterministic { episode_seed } => {
                    yz_core::TurnContext::new_deterministic(episode_seed)
                }
                ChanceMode::Rng { seed } => {
                    // Derive a per-sim seed to decorrelate rollouts.
                    let sim_seed = seed ^ ((sim as u64).wrapping_mul(0x9E3779B97F4A7C15));
                    yz_core::TurnContext::new_rng(sim_seed)
                }
            };

            self.simulate(root_id, root_state, &mut ctx, infer, mode);
        }

        let (pi, root_value) = self.root_pi_value(root_id, &root_state);

        SearchResult {
            pi,
            root_value,
            root_priors_raw,
            root_priors_noisy,
            stats: self.stats.clone(),
        }
    }

    fn simulate(
        &mut self,
        root_id: NodeId,
        mut state: GameState,
        ctx: &mut yz_core::TurnContext,
        infer: &impl Inference,
        mode: ChanceMode,
    ) {
        // Path as (node_id, action_idx) so we can update stats on the way back.
        let mut path: Vec<(
            NodeId,
            usize,
            u8, /* parent to_play */
            u8, /* child to_play */
        )> = Vec::new();
        let mut node_id = root_id;

        loop {
            let node_to_play = self.arena.get(node_id).to_play;

            // Terminal leaf value.
            if yz_core::is_terminal(&state) {
                let z = yz_core::terminal_z_from_player_to_move(&state).unwrap_or(0.0);
                self.backup(&path, z);
                return;
            }

            // Expand if needed.
            if !self.arena.get(node_id).is_expanded {
                let (_priors, v) = self.expand_node(node_id, &state, infer);
                self.backup(&path, v);
                return;
            }

            // Select action by PUCT.
            let legal = legal_action_mask(
                state.players[state.player_to_move as usize].avail_mask,
                state.rerolls_left,
            );
            let a_idx = self.select_action(node_id, &legal, mode);

            // Apply action -> stochastic next state.
            let action = index_to_action(a_idx as u8);
            let next_state = match yz_core::apply_action(state, action, ctx) {
                Ok(s2) => s2,
                Err(_) => {
                    // Should not happen if legal mask is consistent; treat as fallback.
                    self.stats.fallbacks += 1;
                    let v = 0.0;
                    self.backup(&path, v);
                    return;
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

            path.push((
                node_id,
                a_idx as usize,
                node_to_play,
                next_state.player_to_move,
            ));

            node_id = child_id;
            state = next_state;
        }
    }

    fn expand_node(
        &mut self,
        node_id: NodeId,
        state: &GameState,
        infer: &impl Inference,
    ) -> ([f32; A], f32) {
        let legal = legal_action_mask(
            state.players[state.player_to_move as usize].avail_mask,
            state.rerolls_left,
        );
        let features = yz_features::encode_state_v1(state);
        let (logits, v) = infer.eval(&features, &legal);

        let priors = masked_softmax(&logits, &legal, &mut self.stats);

        let n = self.arena.get_mut(node_id);
        n.is_expanded = true;
        n.to_play = state.player_to_move;
        n.p = priors;

        self.stats.expansions += 1;
        (priors, v.clamp(-1.0, 1.0))
    }

    fn select_action(&self, node_id: NodeId, legal: &[bool; A], mode: ChanceMode) -> u8 {
        let n = self.arena.get(node_id);
        let sqrt_sum = (n.n_sum as f32).sqrt();

        let mut best_score = f32::NEG_INFINITY;
        let mut best_a: u8 = 0;

        for (a, &ok) in legal.iter().enumerate() {
            if !ok {
                continue;
            }
            let q = n.q(a);
            let u = self.cfg.c_puct * n.p[a] * sqrt_sum / (1.0 + (n.n[a] as f32));
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

fn masked_softmax(logits: &[f32; A], legal: &[bool; A], stats: &mut SearchStats) -> [f32; A] {
    let mut out = [0.0f32; A];

    let mut max = f32::NEG_INFINITY;
    for (a, &ok) in legal.iter().enumerate() {
        if ok && logits[a].is_finite() {
            max = max.max(logits[a]);
        }
    }
    if !max.is_finite() {
        stats.fallbacks += 1;
        return uniform_over_legal(legal);
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
        return uniform_over_legal(legal);
    }

    for v in &mut out {
        *v /= sum;
    }
    out
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
