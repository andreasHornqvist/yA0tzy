//! Swedish/Scandinavian Yatzy optimal-play oracle (Larsson & Sjöberg) in Rust.
//!
//! This crate exposes a **low-level** oracle API so you can plug it into any Yatzy implementation.
//! See `oracle::YatzyDP::best_action` for the full state/action specification.

// Vendored code - allow clippy warnings that would require significant refactoring
#![allow(clippy::needless_range_loop)]
#![allow(clippy::comparison_chain)]
#![allow(clippy::manual_contains)]
#![allow(clippy::new_without_default)]
#![allow(clippy::wrong_self_convention)]

pub mod eval;
pub mod game;
pub mod oracle;

pub use game::{CAT_NAMES, FULL_MASK, NUM_CATS};
pub use oracle::{Action, YatzyDP};
