//! Game rules engine: state transitions for the oracle-compatible ruleset.
//!
//! This module is the single place that mutates `GameState` via rules.
//! (PRD Epic E3.5.1)

use crate::action::{action_to_index, avail_bit_for_cat, Action, NUM_CATS};
use crate::chance::{self, EventKey};
use crate::legal::legal_action_mask;
use crate::scoring::apply_mark_score;
use crate::state::{GameState, PlayerState};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_core::SeedableRng;
use thiserror::Error;

/// All categories available (oracle convention: 15 bits set).
pub const FULL_MASK: u16 = (1u16 << 15) - 1;

/// How dice are generated for transitions.
pub enum ChanceMode {
    /// Deterministic, event-keyed dice stream (PRD §6.1). Requires an episode seed.
    DeterministicEventKeyed { episode_seed: u64 },
    /// Pseudorandom dice stream backed by a small PRNG.
    Rng { rng: Box<ChaCha8Rng> },
}

impl ChanceMode {
    fn roll5(&mut self, key: EventKey) -> [u8; 5] {
        match self {
            ChanceMode::DeterministicEventKeyed { .. } => chance::roll5(key),
            ChanceMode::Rng { rng } => {
                let mut out = [0u8; 5];
                for o in &mut out {
                    *o = rng.gen_range(1..=6);
                }
                out
            }
        }
    }
}

/// Mutable transition context: chance mode + (future) any per-episode bookkeeping.
pub struct TurnContext {
    pub chance: ChanceMode,
}

impl TurnContext {
    pub fn new_deterministic(episode_seed: u64) -> Self {
        Self {
            chance: ChanceMode::DeterministicEventKeyed { episode_seed },
        }
    }

    pub fn new_rng(seed: u64) -> Self {
        Self {
            chance: ChanceMode::Rng {
                rng: Box::new(ChaCha8Rng::seed_from_u64(seed)),
            },
        }
    }
}

#[derive(Debug, Error)]
pub enum ApplyError {
    #[error("illegal action {action:?} in current state")]
    IllegalAction { action: Action },
    #[error("invalid state: {msg}")]
    InvalidState { msg: &'static str },
}

/// Create a canonical initial game state (player 0 to move, new turn).
pub fn initial_state(ctx: &mut TurnContext) -> GameState {
    let players = [
        PlayerState {
            avail_mask: FULL_MASK,
            upper_total_cap: 0,
            total_score: 0,
        },
        PlayerState {
            avail_mask: FULL_MASK,
            upper_total_cap: 0,
            total_score: 0,
        },
    ];

    let player_to_move = 0u8;
    let rerolls_left = 2u8;

    let dice_sorted = roll_fresh_turn_dice(ctx, &players, player_to_move, rerolls_left);

    GameState {
        players,
        dice_sorted,
        rerolls_left,
        player_to_move,
    }
}

/// Terminal when both players have filled all 15 categories.
pub fn is_terminal(s: &GameState) -> bool {
    s.players[0].avail_mask == 0 && s.players[1].avail_mask == 0
}

/// Apply an action to a state, producing the next state (or an error if illegal).
pub fn apply_action(
    mut state: GameState,
    action: Action,
    ctx: &mut TurnContext,
) -> Result<GameState, ApplyError> {
    validate_state(&state)?;

    let p = state.player_to_move as usize;
    let avail_mask = state.players[p].avail_mask;
    let legal = legal_action_mask(avail_mask, state.rerolls_left);
    let idx = action_to_index(action) as usize;
    if !legal[idx] {
        return Err(ApplyError::IllegalAction { action });
    }

    match action {
        Action::KeepMask(mask) => {
            // Legal mask already enforced mask<=30 and rerolls_left>0.
            let old_rerolls_left = state.rerolls_left;
            let new_roll_idx =
                3u8.checked_sub(old_rerolls_left)
                    .ok_or(ApplyError::InvalidState {
                        msg: "rerolls_left out of range for KeepMask",
                    })?;

            let next_dice = match &mut ctx.chance {
                ChanceMode::DeterministicEventKeyed { episode_seed } => {
                    let round_idx = round_idx_for_player(&state.players, state.player_to_move);
                    let key = EventKey {
                        episode_seed: *episode_seed,
                        player: state.player_to_move,
                        round_idx,
                        roll_idx: new_roll_idx,
                    };
                    chance::apply_keepmask(state.dice_sorted, mask, key)
                }
                ChanceMode::Rng { rng } => {
                    let mut next = state.dice_sorted;
                    for (i, die) in next.iter_mut().enumerate() {
                        let bit = 1u8 << (4 - i);
                        if (mask & bit) == 0 {
                            *die = rng.gen_range(1..=6);
                        }
                    }
                    next.sort();
                    next
                }
            };

            state.dice_sorted = next_dice;
            state.rerolls_left =
                state
                    .rerolls_left
                    .checked_sub(1)
                    .ok_or(ApplyError::InvalidState {
                        msg: "rerolls_left underflow on KeepMask",
                    })?;
            Ok(state)
        }
        Action::Mark(cat) => {
            // Update current player's board.
            let bit = avail_bit_for_cat(cat);
            state.players[p].avail_mask &= !bit;

            let (delta, new_upper) =
                apply_mark_score(state.dice_sorted, cat, state.players[p].upper_total_cap);
            state.players[p].upper_total_cap = new_upper;

            // Total score is stored as i16, but deltas are small (<=~100), so i32->i16 is safe.
            let new_total = (state.players[p].total_score as i32) + delta;
            state.players[p].total_score = new_total as i16;

            // Switch turn.
            state.player_to_move = 1u8.saturating_sub(state.player_to_move);
            state.rerolls_left = 2;

            // Roll fresh dice for next player (roll_idx=0).
            state.dice_sorted = roll_fresh_turn_dice(
                ctx,
                &state.players,
                state.player_to_move,
                state.rerolls_left,
            );

            Ok(state)
        }
    }
}

fn validate_state(s: &GameState) -> Result<(), ApplyError> {
    if s.player_to_move > 1 {
        return Err(ApplyError::InvalidState {
            msg: "player_to_move must be 0 or 1",
        });
    }
    if s.rerolls_left > 2 {
        return Err(ApplyError::InvalidState {
            msg: "rerolls_left must be in 0..=2",
        });
    }
    if !s.dice_sorted.windows(2).all(|w| w[0] <= w[1]) {
        return Err(ApplyError::InvalidState {
            msg: "dice_sorted must be sorted ascending",
        });
    }
    for &d in &s.dice_sorted {
        if !(1..=6).contains(&d) {
            return Err(ApplyError::InvalidState {
                msg: "dice values must be in 1..=6",
            });
        }
    }
    Ok(())
}

fn round_idx_for_player(players: &[PlayerState; 2], player: u8) -> u8 {
    let mask = players[player as usize].avail_mask;
    let avail = mask.count_ones() as u8;
    (NUM_CATS as u8).saturating_sub(avail)
}

fn current_roll_idx_from_rerolls_left(rerolls_left: u8) -> u8 {
    // Start-of-turn: rerolls_left=2 => roll_idx=0. After 1 reroll: rerolls_left=1 => roll_idx=1, etc.
    2u8.saturating_sub(rerolls_left)
}

fn roll_fresh_turn_dice(
    ctx: &mut TurnContext,
    players: &[PlayerState; 2],
    player_to_move: u8,
    rerolls_left: u8,
) -> [u8; 5] {
    let roll_idx = current_roll_idx_from_rerolls_left(rerolls_left);
    let round_idx = round_idx_for_player(players, player_to_move);

    let key = match &ctx.chance {
        ChanceMode::DeterministicEventKeyed { episode_seed } => EventKey {
            episode_seed: *episode_seed,
            player: player_to_move,
            round_idx,
            roll_idx,
        },
        ChanceMode::Rng { .. } => EventKey {
            // Unused in RNG mode, but we need a value to pass to the shared function.
            episode_seed: 0,
            player: player_to_move,
            round_idx,
            roll_idx,
        },
    };

    let mut dice = ctx.chance.roll5(key);
    dice.sort();
    dice
}
