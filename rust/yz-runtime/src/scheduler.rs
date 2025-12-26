use crate::game_task::{GameTask, StepStatus};
use yz_mcts::InferBackend;
use yz_replay::{ReplayError, ShardWriter};

#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    pub ticks: u64,
    pub steps: u64,
    pub terminal: u64,
    pub would_block: u64,
}

pub struct Scheduler {
    tasks: Vec<GameTask>,
    steps_per_tick: u32,
    stats: SchedulerStats,
}

impl Scheduler {
    pub fn new(tasks: Vec<GameTask>, steps_per_tick: u32) -> Self {
        Self {
            tasks,
            steps_per_tick,
            stats: SchedulerStats::default(),
        }
    }

    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    pub fn tasks(&self) -> &[GameTask] {
        &self.tasks
    }

    pub fn tasks_mut(&mut self) -> &mut [GameTask] {
        &mut self.tasks
    }

    /// Run one scheduler tick: round-robin over all tasks, giving each up to `steps_per_tick`.
    pub fn tick(&mut self, backend: &InferBackend) {
        self.stats.ticks += 1;
        for t in &mut self.tasks {
            let r = t.step(backend, self.steps_per_tick);
            match r {
                Ok(sr) => match sr.status {
                    StepStatus::Progress => {
                        self.stats.steps += 1;
                    }
                    StepStatus::WouldBlock => self.stats.would_block += 1,
                    StepStatus::Terminal => self.stats.terminal += 1,
                },
                Err(_) => {
                    // Treat errors as terminal for v1 runtime scaffolding.
                    self.stats.terminal += 1;
                }
            }
        }
    }

    /// Like `tick`, but also writes any completed episodes to the shard writer.
    pub fn tick_and_write(
        &mut self,
        backend: &InferBackend,
        writer: &mut ShardWriter,
    ) -> Result<(), ReplayError> {
        self.stats.ticks += 1;
        for t in &mut self.tasks {
            let r = t.step(backend, self.steps_per_tick);
            match r {
                Ok(sr) => {
                    if let Some(ep) = sr.completed_episode {
                        writer.extend(ep)?;
                        self.stats.terminal += 1;
                        continue;
                    }
                    match sr.status {
                        StepStatus::Progress => self.stats.steps += 1,
                        StepStatus::WouldBlock => self.stats.would_block += 1,
                        StepStatus::Terminal => self.stats.terminal += 1,
                    }
                }
                Err(_) => {
                    self.stats.terminal += 1;
                }
            }
        }
        Ok(())
    }
}
