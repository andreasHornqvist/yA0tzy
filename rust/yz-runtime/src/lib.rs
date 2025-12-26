//! Worker runtime / schedulers (PRD Epic E7).

pub mod game_task;
pub mod scheduler;

pub use game_task::{GameTask, GameTaskError, StepResult, StepStatus};
pub use scheduler::{Scheduler, SchedulerStats};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_nonempty() {
        assert!(!VERSION.is_empty());
    }
}

#[cfg(test)]
mod runtime_tests;
