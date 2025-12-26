use thiserror::Error;
use yz_core::{apply_action, index_to_action, is_terminal, GameState};
use yz_mcts::{ChanceMode, InferBackend, Mcts, MctsConfig, SearchDriver};

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
}

pub struct GameTask {
    pub game_id: u64,
    pub ply: u32,
    pub state: GameState,
    pub mode: ChanceMode,

    mcts: Mcts,
    search: Option<SearchDriver>,
}

impl GameTask {
    pub fn new(game_id: u64, state: GameState, mode: ChanceMode, mcts_cfg: MctsConfig) -> Self {
        let mcts = Mcts::new(mcts_cfg).expect("valid mcts cfg");
        Self {
            game_id,
            ply: 0,
            state,
            mode,
            mcts,
            search: None,
        }
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
                // Choose executed action (v1: greedy argmax of pi).
                let (best_a, _best_p) = sr
                    .pi
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap();
                let action = index_to_action(best_a as u8);

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

                let next = apply_action(self.state, action, &mut ctx)
                    .map_err(|_| GameTaskError::IllegalTransition)?;
                self.state = next;
                self.ply += 1;
                self.search = None;
                return Ok(StepResult {
                    status: StepStatus::Progress,
                    work_done,
                });
            }
        }

        Ok(StepResult {
            status: StepStatus::WouldBlock,
            work_done,
        })
    }
}
