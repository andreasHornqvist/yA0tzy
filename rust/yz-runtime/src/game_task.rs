use thiserror::Error;
use yz_core::{apply_action, index_to_action, is_terminal, Action, GameState};
use yz_features::schema::F;
use yz_mcts::{ChanceMode, InferBackend, Mcts, MctsConfig, SearchDriver};
use yz_replay::ReplaySample;
use yz_core::config::TemperatureSchedule;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;

#[derive(Debug, Error)]
pub enum GameTaskError {
    #[error("illegal transition while applying move")]
    IllegalTransition,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepStatus {
    Progress,
    WouldBlock,
    Terminal,
}

#[derive(Debug)]
pub struct StepResult {
    pub status: StepStatus,
    pub work_done: u32,
    /// Present only when a move was executed (i.e. a search finished and we applied one action).
    pub executed: Option<ExecutedMove>,
    /// If a full episode finished this step, the collected replay samples.
    pub completed_episode: Option<Vec<ReplaySample>>,
}

#[derive(Debug, Clone)]
pub struct SearchSummary {
    pub pi: [f32; yz_core::A],
    pub root_value: f32,
    pub delta_root_value: f32,
    pub root_priors_raw: Option<[f32; yz_core::A]>,
    pub root_priors_noisy: Option<[f32; yz_core::A]>,
    pub fallbacks: u32,
    pub pending_count_max: usize,
    pub pending_collisions: u32,
    pub leaf_eval_submitted: u32,
    pub leaf_eval_discarded: u32,
}

#[derive(Debug, Clone)]
pub struct ExecutedMove {
    pub game_id: u64,
    pub game_ply: u32,
    pub chosen_action: u8,
    pub player_to_move: u8,
    pub rerolls_left: u8,
    pub dice: [u8; 5],
    pub search: SearchSummary,
}

struct PendingSample {
    features: [f32; F],
    legal_mask: [u8; yz_core::A],
    pi: [f32; yz_core::A],
    pov_player: u8,
}

pub struct GameTask {
    pub game_id: u64,
    pub ply: u32,
    /// Scoring turn index (number of Mark actions taken so far).
    pub turn_idx: u32,
    pub state: GameState,
    pub mode: ChanceMode,
    pub temperature_schedule: TemperatureSchedule,

    mcts: Mcts,
    search: Option<SearchDriver>,
    traj: Vec<PendingSample>,
}

impl GameTask {
    pub fn new(
        game_id: u64,
        state: GameState,
        mode: ChanceMode,
        temperature_schedule: TemperatureSchedule,
        mcts_cfg: MctsConfig,
    ) -> Self {
        let mcts = Mcts::new(mcts_cfg).expect("valid mcts cfg");
        Self {
            game_id,
            ply: 0,
            turn_idx: 0,
            state,
            mode,
            temperature_schedule,
            mcts,
            search: None,
            traj: Vec::new(),
        }
    }

    fn temperature_for_turn(&self) -> f32 {
        match self.temperature_schedule {
            TemperatureSchedule::Constant { t0 } => t0,
            TemperatureSchedule::Step {
                t0,
                t1,
                cutoff_turn,
            } => {
                if self.turn_idx < cutoff_turn {
                    t0
                } else {
                    t1
                }
            }
        }
    }

    fn sample_exec_action(&self, exec_pi: &[f32; yz_core::A]) -> u8 {
        // Deterministic per-game/per-ply sampling: seed derived from game mode + ply.
        let base: u64 = match self.mode {
            ChanceMode::Deterministic { episode_seed } => episode_seed,
            ChanceMode::Rng { seed } => seed,
        };
        let seed = base ^ (self.ply as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1E7_C437_9E37_79B9;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let r: f32 = rng.gen::<f32>(); // [0,1)
        let mut acc = 0.0f32;
        for (i, &p) in exec_pi.iter().enumerate() {
            if p <= 0.0 {
                continue;
            }
            acc += p;
            if r <= acc {
                return i as u8;
            }
        }
        // Numeric edge case: fallback to argmax.
        let mut best_i: usize = 0;
        let mut best_v: f32 = f32::NEG_INFINITY;
        for (i, &p) in exec_pi.iter().enumerate() {
            if p > best_v {
                best_v = p;
                best_i = i;
            }
        }
        best_i as u8
    }

    pub fn is_terminal(&self) -> bool {
        is_terminal(&self.state)
    }

    /// Step this game forward by up to `max_work` small operations.
    ///
    /// This method is non-blocking: if it cannot make progress (e.g. waiting on inference),
    /// it returns `WouldBlock` quickly so a scheduler can run other games.
    pub fn step(
        &mut self,
        backend: &InferBackend,
        max_work: u32,
    ) -> Result<StepResult, GameTaskError> {
        if self.is_terminal() {
            return Ok(StepResult {
                status: StepStatus::Terminal,
                work_done: 0,
                executed: None,
                completed_episode: None,
            });
        }

        if self.search.is_none() {
            let d = self
                .mcts
                .begin_search_with_backend(self.state, self.mode, backend);
            self.search = Some(d);
        }

        let mut work_done = 0u32;
        if let Some(search) = &mut self.search {
            let res = search.tick(&mut self.mcts, backend, max_work);
            work_done = max_work;
            if let Some(sr) = res {
                // Choose executed action using temperature schedule (PRD §7.3).
                // NOTE: this does NOT change the stored replay `pi` target (visit distribution).
                let legal_b = yz_mcts::legal_action_mask_for_mode(&self.state, self.mode);
                let t = self.temperature_for_turn();
                let exec_pi = yz_mcts::apply_temperature(&sr.pi, legal_b, t);
                let chosen_action = if t == 0.0 {
                    // apply_temperature already returns a one-hot distribution in this case.
                    exec_pi
                        .iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .map(|(i, _)| i as u8)
                        .unwrap_or(0)
                } else {
                    self.sample_exec_action(&exec_pi)
                };
                let action = index_to_action(chosen_action);
                let pi = sr.pi;
                let search = SearchSummary {
                    pi,
                    root_value: sr.root_value,
                    delta_root_value: sr.delta_root_value,
                    root_priors_raw: sr.root_priors_raw,
                    root_priors_noisy: sr.root_priors_noisy,
                    fallbacks: sr.fallbacks,
                    pending_count_max: sr.stats.pending_count_max,
                    pending_collisions: sr.stats.pending_collisions,
                    leaf_eval_submitted: sr.leaf_eval_submitted,
                    leaf_eval_discarded: sr.leaf_eval_discarded,
                };

                let mut ctx = match self.mode {
                    ChanceMode::Deterministic { episode_seed } => {
                        yz_core::TurnContext::new_deterministic(episode_seed)
                    }
                    ChanceMode::Rng { seed } => {
                        // Derive a per-ply seed for executed moves.
                        let s = seed ^ (self.ply as u64).wrapping_mul(0xD1E7_C437_9E37_79B9);
                        yz_core::TurnContext::new_rng(s)
                    }
                };

                // Emit replay sample (features/legal/pi) before state transition, z assigned at terminal.
                let feats = yz_features::encode_state_v1(&self.state);
                let legal = yz_core::legal_mask_to_u8_array(legal_b);
                self.traj.push(PendingSample {
                    features: feats,
                    legal_mask: legal,
                    pi,
                    pov_player: self.state.player_to_move,
                });

                let executed = ExecutedMove {
                    game_id: self.game_id,
                    game_ply: self.ply,
                    chosen_action,
                    player_to_move: self.state.player_to_move,
                    rerolls_left: self.state.rerolls_left,
                    dice: self.state.dice_sorted,
                    search,
                };

                let next = apply_action(self.state, action, &mut ctx)
                    .map_err(|_| GameTaskError::IllegalTransition)?;
                self.state = next;
                if matches!(action, Action::Mark(_)) {
                    self.turn_idx += 1;
                }
                self.ply += 1;
                self.search = None;
                let mut out_episode = None;
                if is_terminal(&self.state) {
                    // Terminal z for each POV.
                    let z0 = yz_core::terminal_z_from_pov(&self.state, 0).unwrap_or(0.0);
                    let z1 = yz_core::terminal_z_from_pov(&self.state, 1).unwrap_or(0.0);
                    let mut samples = Vec::with_capacity(self.traj.len());
                    for ps in self.traj.drain(..) {
                        let z = if ps.pov_player == 0 { z0 } else { z1 };
                        samples.push(ReplaySample {
                            features: ps.features,
                            legal_mask: ps.legal_mask,
                            pi: ps.pi,
                            z,
                            z_margin: None,
                        });
                    }
                    out_episode = Some(samples);
                }

                return Ok(StepResult {
                    status: if out_episode.is_some() {
                        StepStatus::Terminal
                    } else {
                        StepStatus::Progress
                    },
                    work_done,
                    executed: Some(executed),
                    completed_episode: out_episode,
                });
            }
        }

        Ok(StepResult {
            status: StepStatus::WouldBlock,
            work_done,
            executed: None,
            completed_episode: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use yz_core::config::TemperatureSchedule;

    #[test]
    fn temperature_step_uses_turn_idx_not_ply() {
        let mut ctx = yz_core::TurnContext::new_deterministic(1);
        let s = yz_core::initial_state(&mut ctx);
        let mut t = GameTask::new(
            0,
            s,
            ChanceMode::Deterministic { episode_seed: 1 },
            TemperatureSchedule::Step {
                t0: 1.0,
                t1: 0.0,
                cutoff_turn: 1,
            },
            MctsConfig::default(),
        );

        // Before any marks: turn_idx=0 -> t0, regardless of ply.
        t.ply = 999;
        t.turn_idx = 0;
        assert_eq!(t.temperature_for_turn(), 1.0);

        // After one mark: turn_idx=1 -> t1.
        t.turn_idx = 1;
        assert_eq!(t.temperature_for_turn(), 0.0);
    }
}
