use crate::game_task::{GameTask, StepStatus};
use yz_mcts::InferBackend;
use yz_replay::{ReplayError, ShardWriter};
use yz_logging::{InferStatsV1, IterationStatsEventV1, MctsRootEventV1, NdjsonWriter, PiSummaryV1};
use yz_logging::VersionInfoV1;

#[derive(Debug, Default, Clone)]
pub struct SchedulerStats {
    pub ticks: u64,
    pub steps: u64,
    pub terminal: u64,
    pub would_block: u64,
}

pub struct RunLoggers {
    pub run_id: String,
    pub v: VersionInfoV1,
    pub root_log_every_n: u64,
    pub iter: NdjsonWriter,
    pub roots: NdjsonWriter,
}

pub struct Scheduler {
    tasks: Vec<GameTask>,
    steps_per_tick: u32,
    stats: SchedulerStats,
    global_ply: u64,
    completed_games_total: u64,
}

impl Scheduler {
    pub fn new(tasks: Vec<GameTask>, steps_per_tick: u32) -> Self {
        Self {
            tasks,
            steps_per_tick,
            stats: SchedulerStats::default(),
            global_ply: 0,
            completed_games_total: 0,
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
        mut loggers: Option<&mut RunLoggers>,
    ) -> Result<(), ReplayError> {
        self.stats.ticks += 1;
        for t in &mut self.tasks {
            let r = t.step(backend, self.steps_per_tick);
            match r {
                Ok(sr) => {
                    if let Some(exec) = sr.executed {
                        self.global_ply += 1;
                        if let Some(lg) = loggers.as_deref_mut() {
                            if lg.root_log_every_n > 0
                                && (self.global_ply % lg.root_log_every_n == 0)
                            {
                                let ts_ms = now_ms();
                                let pi = summarize_pi(&exec.search.pi);
                                let ev = MctsRootEventV1 {
                                    event: "mcts_root_v1",
                                    ts_ms,
                                    v: lg.v.clone(),
                                    run_id: lg.run_id.clone(),
                                    global_ply: self.global_ply,
                                    game_id: exec.game_id,
                                    game_ply: exec.game_ply,
                                    player_to_move: exec.player_to_move,
                                    rerolls_left: exec.rerolls_left,
                                    dice: exec.dice,
                                    chosen_action: exec.chosen_action,
                                    root_value: exec.search.root_value,
                                    fallbacks: exec.search.fallbacks,
                                    pending_count_max: exec.search.pending_count_max as u64,
                                    pending_collisions: exec.search.pending_collisions,
                                    pi,
                                };
                                // Logging must not break replay writing; ignore errors in v1.
                                let _ = lg.roots.write_event(&ev);
                            }
                        }
                    }
                    if let Some(ep) = sr.completed_episode {
                        writer.extend(ep)?;
                        self.stats.terminal += 1;
                        self.completed_games_total += 1;
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
        if let Some(lg) = loggers.as_deref_mut() {
            let s = backend.stats_snapshot();
            let ts_ms = now_ms();
            let ev = IterationStatsEventV1 {
                event: "iteration_stats_v1",
                ts_ms,
                v: lg.v.clone(),
                run_id: lg.run_id.clone(),
                tick: self.stats.ticks,
                global_ply: self.global_ply,
                tasks: self.tasks.len() as u64,
                completed_games: self.completed_games_total,
                steps: self.stats.steps,
                would_block: self.stats.would_block,
                terminal: self.stats.terminal,
                infer: InferStatsV1 {
                    inflight: s.inflight as u64,
                    sent: s.sent,
                    received: s.received,
                    errors: s.errors,
                    latency_p50_us: s.latency_us.summary.p50_us,
                    latency_p95_us: s.latency_us.summary.p95_us,
                    latency_mean_us: s.latency_us.summary.mean_us,
                },
            };
            let _ = lg.iter.write_event(&ev);
        }
        Ok(())
    }
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    d.as_millis() as u64
}

fn summarize_pi(pi: &[f32; yz_core::A]) -> PiSummaryV1 {
    let mut max_p = -1.0f32;
    let mut argmax = 0usize;
    let mut ent = 0.0f32;
    for (i, &p) in pi.iter().enumerate() {
        if p > max_p {
            max_p = p;
            argmax = i;
        }
        if p > 0.0 && p.is_finite() {
            ent -= p * p.ln();
        }
    }
    PiSummaryV1 {
        entropy: ent,
        max_p: max_p.max(0.0),
        argmax_a: argmax as u8,
    }
}
