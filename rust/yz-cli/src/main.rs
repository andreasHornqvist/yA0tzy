//! yz: CLI binary for AlphaZero Yatzy.
//!
//! Subcommands (PRD section 13):
//! - oracle expected
//! - oracle sim
//! - selfplay
//! - gate
//! - oracle-eval
//! - bench
//! - profile

use std::env;
use std::path::PathBuf;
use std::process;
use std::process::Command;

/// Print the oracle's optimal expected score for a fresh game.
fn cmd_oracle_expected() {
    println!("Building oracle DP table...");
    let info = yz_oracle::get_expected_score();
    println!();
    println!("Optimal expected score: {:.4}", info.expected_score);
    println!("Build time: {:.2}s", info.build_time_secs);
}

fn cmd_oracle_sim(args: &[String]) {
    let mut games: usize = 10_000;
    let mut seed: u64 = 0;
    let mut no_hist = false;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz oracle sim

USAGE:
    yz oracle sim [--games N] [--seed S] [--no-hist]

OPTIONS:
    --games N    Number of games to simulate (default: 10000)
    --seed S     RNG seed (default: 0)
    --no-hist    Skip printing histogram
"#
                );
                return;
            }
            "--games" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --games");
                    process::exit(1);
                }
                games = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --games value: {}", args[i + 1]);
                    process::exit(1);
                });
                i += 2;
            }
            "--seed" => {
                if i + 1 >= args.len() {
                    eprintln!("Missing value for --seed");
                    process::exit(1);
                }
                seed = args[i + 1].parse().unwrap_or_else(|_| {
                    eprintln!("Invalid --seed value: {}", args[i + 1]);
                    process::exit(1);
                });
                i += 2;
            }
            "--no-hist" => {
                no_hist = true;
                i += 1;
            }
            other => {
                eprintln!("Unknown option for `yz oracle sim`: {}", other);
                eprintln!("Run `yz oracle sim --help` for usage.");
                process::exit(1);
            }
        }
    }

    println!("Building oracle DP table...");
    let _ = yz_oracle::oracle(); // build+cache
    println!("Running simulation...");

    let report = yz_oracle::simulate(games, seed);
    let s = report.summary;

    println!();
    println!("Evaluation:");
    println!("  - Games: {}", games);
    println!(
        "  - Score: mean={:.2}, median={}, std={:.2}, min={}, max={}",
        s.mean, s.median, s.std_dev, s.min, s.max
    );
    println!("  - Upper bonus rate: {:.1}%", report.bonus_rate * 100.0);

    if !no_hist {
        yz_oracle::print_histogram(&report.scores);
    }
}

fn print_help() {
    eprintln!(
        r#"yz - AlphaZero Yatzy CLI

USAGE:
    yz <COMMAND> [OPTIONS]

COMMANDS:
    oracle expected     Print oracle expected score (~248.44)
    oracle sim          Run oracle solitaire simulation
    selfplay            Run self-play with MCTS + inference
    selfplay-worker     Internal: run one self-play worker process (spawned by controller)
    gate                Gate candidate vs best model (paired seed + side swap)
    gate-worker         Internal: run one gating worker process (spawned by controller)
    oracle-eval         Evaluate models against oracle baseline
    bench               Run Criterion micro-benchmarks (wrapper around cargo bench)
    tui                 Terminal UI (Ratatui) for configuring + monitoring runs
    profile             Run with profiler hooks enabled

OPTIONS:
    -h, --help          Print this help message
    -V, --version       Print version

For more information, see the PRD or run `yz <COMMAND> --help`.
"#
    );
}

fn cmd_selfplay_worker(args: &[String]) {
    let mut run_dir: Option<String> = None;
    let mut infer: Option<String> = None;
    let mut worker_id: u32 = 0;
    let mut num_workers: u32 = 1;
    let mut games: u32 = 1;
    let mut seed_base: u64 = 0xC0FFEE;
    let mut max_samples_per_shard: usize = 8192;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz selfplay-worker

USAGE:
    yz selfplay-worker --run-dir runs/<id> --infer unix:///tmp/yatzy_infer.sock --worker-id W --num-workers N --games G [--seed-base S] [--max-samples-per-shard N]

OPTIONS:
    --run-dir DIR               Run directory (contains run.json + config.yaml) (required)
    --infer ENDPOINT            Inference endpoint (unix:///... or tcp://host:port) (required)
    --worker-id W               Worker id in [0, N) (required)
    --num-workers N             Total worker processes (required)
    --games G                   Games to complete in this worker (required)
    --seed-base S               Seed base for deterministic uniqueness (default: 0xC0FFEE)
    --max-samples-per-shard N   Samples per replay shard (default: 8192)
"#
                );
                return;
            }
            "--run-dir" => {
                run_dir = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--infer" => {
                infer = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--worker-id" => {
                worker_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --worker-id value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--num-workers" => {
                num_workers = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --num-workers value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--games" => {
                games = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --games value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--seed-base" => {
                seed_base = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --seed-base value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--max-samples-per-shard" => {
                max_samples_per_shard = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --max-samples-per-shard value");
                        process::exit(1);
                    });
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz selfplay-worker`: {other}");
                eprintln!("Run `yz selfplay-worker --help` for usage.");
                process::exit(1);
            }
        }
    }

    let run_dir = run_dir.unwrap_or_else(|| {
        eprintln!("Missing --run-dir");
        process::exit(1);
    });
    let infer = infer.unwrap_or_else(|| {
        eprintln!("Missing --infer");
        process::exit(1);
    });
    if num_workers < 1 {
        eprintln!("--num-workers must be >= 1");
        process::exit(1);
    }
    if worker_id >= num_workers {
        eprintln!("--worker-id must be in [0, num_workers)");
        process::exit(1);
    }
    if games < 1 {
        eprintln!("--games must be >= 1");
        process::exit(1);
    }

    let run_dir = PathBuf::from(run_dir);
    let cfg_path = run_dir.join("config.yaml");
    let cfg = yz_core::Config::load(&cfg_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config at {}: {e}", cfg_path.display());
        process::exit(1);
    });

    let backend = connect_infer_backend(&infer);

    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations_mark: cfg.mcts.budget_mark.max(1),
        simulations_reroll: cfg.mcts.budget_reroll.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: cfg.mcts.dirichlet_epsilon,
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss: 1.0,
    };

    // Replay output is worker-local to avoid collisions.
    let replay_dir = run_dir
        .join("replay_workers")
        .join(format!("worker_{worker_id:03}"));
    let manifest_path = run_dir.join("run.json");
    let manifest = yz_logging::read_manifest(&manifest_path).unwrap_or_else(|e| {
        eprintln!("Failed to read manifest at {}: {e}", manifest_path.display());
        process::exit(1);
    });

    let mut writer = yz_replay::ShardWriter::new(yz_replay::ShardWriterConfig {
        out_dir: replay_dir.clone(),
        max_samples_per_shard,
        git_hash: manifest.git_hash.clone(),
        config_hash: manifest.config_hash.clone(),
    })
    .unwrap_or_else(|e| {
        eprintln!("Failed to create shard writer at {}: {e}", replay_dir.display());
        process::exit(1);
    });

    // Worker stats log (per-process).
    let logs_dir = run_dir
        .join("logs_workers")
        .join(format!("worker_{worker_id:03}"));
    let _ = std::fs::create_dir_all(&logs_dir);
    let mut worker_log =
        yz_logging::NdjsonWriter::open_append_with_flush(logs_dir.join("worker_stats.ndjson"), 50)
            .ok();
    let progress_path = logs_dir.join("progress.json");

    #[derive(serde::Serialize)]
    struct WorkerStatsEvent<'a> {
        event: &'a str,
        ts_ms: u64,
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        games_completed: u32,
        wall_ms: u64,
    }

    let t_start = std::time::Instant::now();
    if let Some(w) = worker_log.as_mut() {
        let _ = w.write_event(&WorkerStatsEvent {
            event: "selfplay_worker_start",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target: games,
            games_completed: 0,
            wall_ms: 0,
        });
        // Ensure the start event is visible even if the worker is cancelled early.
        let _ = w.flush();
    }
    // Write initial progress atomically (for controller/TUI live progress).
    #[derive(serde::Serialize)]
    struct WorkerProgress {
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        games_completed: u32,
        pid: u32,
        sched_ticks: u64,
        sched_steps: u64,
        sched_would_block: u64,
        sched_terminal: u64,
        ts_ms: u64,
    }
    fn write_progress_atomic(path: &PathBuf, p: &WorkerProgress) {
        let tmp = path.with_extension("json.tmp");
        if let Ok(bytes) = serde_json::to_vec(p) {
            let _ = std::fs::write(&tmp, &bytes);
            let _ = std::fs::rename(&tmp, path);
        }
    }
    write_progress_atomic(
        &progress_path,
        &WorkerProgress {
            worker_id,
            num_workers,
            games_target: games,
            games_completed: 0,
            pid: std::process::id(),
            sched_ticks: 0,
            sched_steps: 0,
            sched_would_block: 0,
            sched_terminal: 0,
            ts_ms: yz_logging::now_ms(),
        },
    );

    // Each process runs `threads_per_worker` concurrent game tasks.
    let parallel_games = cfg.selfplay.threads_per_worker.max(1) as usize;
    let num_workers_u64 = num_workers as u64;
    let worker_id_u64 = worker_id as u64;

    let mut tasks = Vec::with_capacity(parallel_games);
    for slot in 0..parallel_games {
        let game_id = worker_id_u64 + (slot as u64) * num_workers_u64;
        let mut ctx = yz_core::TurnContext::new_rng(seed_base ^ (0xC0FFEE ^ game_id));
        let s = yz_core::initial_state(&mut ctx);
        tasks.push(yz_runtime::GameTask::new(
            game_id,
            s,
            yz_mcts::ChanceMode::Rng {
                seed: seed_base ^ (0xBADC0DE ^ game_id),
            },
            mcts_cfg,
        ));
    }
    let mut sched = yz_runtime::Scheduler::new(tasks, 64);

    let mut completed_games: u32 = 0;
    let mut next_seq: u64 = parallel_games as u64;
    let mut last_progress_write = std::time::Instant::now();
    while completed_games < games {
        let before_steps = sched.stats().steps;
        let before_terminal = sched.stats().terminal;
        sched.tick_and_write(&backend, &mut writer, None)
            .unwrap_or_else(|e| {
                eprintln!("worker {worker_id}: scheduler failed: {e}");
                process::exit(1);
            });
        let after_steps = sched.stats().steps;
        let after_terminal = sched.stats().terminal;
        // Avoid busy-spinning when all tasks are waiting on inference.
        // This yields CPU to the Python inference server and improves throughput.
        if after_steps == before_steps && after_terminal == before_terminal {
            std::thread::sleep(std::time::Duration::from_micros(200));
        }

        // Heartbeat progress even before games complete, so we can prove workers are running.
        // Rate-limit to avoid excessive filesystem churn.
        if last_progress_write.elapsed() >= std::time::Duration::from_millis(500) {
            let s = sched.stats();
            write_progress_atomic(
                &progress_path,
                &WorkerProgress {
                    worker_id,
                    num_workers,
                    games_target: games,
                    games_completed: completed_games,
                    pid: std::process::id(),
                    sched_ticks: s.ticks,
                    sched_steps: s.steps,
                    sched_would_block: s.would_block,
                    sched_terminal: s.terminal,
                    ts_ms: yz_logging::now_ms(),
                },
            );
            last_progress_write = std::time::Instant::now();
        }

        for t in sched.tasks_mut() {
            if yz_core::is_terminal(&t.state) {
                completed_games += 1;
                // Progress update: cheap atomic write. Rate-limit a bit.
                if completed_games == games || (completed_games % 5 == 0) {
                    write_progress_atomic(
                        &progress_path,
                        &WorkerProgress {
                            worker_id,
                            num_workers,
                            games_target: games,
                            games_completed: completed_games,
                            pid: std::process::id(),
                            // Note: we can't borrow `sched` here because we already hold a mutable
                            // borrow from `tasks_mut()`. These fields will be updated by the
                            // periodic heartbeat above.
                            sched_ticks: 0,
                            sched_steps: 0,
                            sched_would_block: 0,
                            sched_terminal: 0,
                            ts_ms: yz_logging::now_ms(),
                        },
                    );
                }
                if completed_games >= games {
                    break;
                }
                // Reset task for next game; ensure unique game_id across processes.
                let game_id = worker_id_u64 + next_seq * num_workers_u64;
                next_seq += 1;
                let mut ctx = yz_core::TurnContext::new_rng(seed_base ^ (0xC0FFEE ^ game_id));
                let s = yz_core::initial_state(&mut ctx);
                *t = yz_runtime::GameTask::new(
                    game_id,
                    s,
                    yz_mcts::ChanceMode::Rng {
                        seed: seed_base ^ (0xBADC0DE ^ game_id),
                    },
                    mcts_cfg,
                );
            }
        }
    }

    writer.finish().unwrap_or_else(|e| {
        eprintln!("worker {worker_id}: failed to finish replay writer: {e}");
        process::exit(1);
    });
    // Final progress.
    let s = sched.stats();
    write_progress_atomic(
        &progress_path,
        &WorkerProgress {
            worker_id,
            num_workers,
            games_target: games,
            games_completed: completed_games,
            pid: std::process::id(),
            sched_ticks: s.ticks,
            sched_steps: s.steps,
            sched_would_block: s.would_block,
            sched_terminal: s.terminal,
            ts_ms: yz_logging::now_ms(),
        },
    );

    if let Some(w) = worker_log.as_mut() {
        let _ = w.write_event(&WorkerStatsEvent {
            event: "selfplay_worker_done",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target: games,
            games_completed: completed_games,
            wall_ms: t_start.elapsed().as_millis() as u64,
        });
        let _ = w.flush();
    }
}

fn print_version() {
    println!("yz {}", env!("CARGO_PKG_VERSION"));
}

fn cmd_selfplay(args: &[String]) {
    let mut config_path: Option<String> = None;
    let mut infer: Option<String> = None;
    let mut out: Option<String> = None;
    let mut games: u32 = 10;
    let mut max_samples_per_shard: usize = 8192;
    let mut root_log_every: u64 = 50;
    let mut log_flush_every: u64 = 100;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz selfplay

USAGE:
    yz selfplay --config cfg.yaml --infer unix:///tmp/yatzy_infer.sock --out runs/<id>/ [--games N] [--max-samples-per-shard N]

OPTIONS:
    --config PATH               Path to YAML config (required)
    --infer ENDPOINT            Inference endpoint (unix:///... or tcp://host:port) (required)
    --out DIR                   Output directory (required)
    --games N                   Number of games to play (default: 10)
    --max-samples-per-shard N   Samples per replay shard (default: 8192)
    --root-log-every N          Log one MCTS root every N executed moves (default: 50)
    --log-flush-every N         Flush NDJSON logs every N lines (0 disables) (default: 100)
"#
                );
                return;
            }
            "--config" => {
                config_path = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--infer" => {
                infer = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--out" => {
                out = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--games" => {
                games = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --games value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--max-samples-per-shard" => {
                max_samples_per_shard = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --max-samples-per-shard value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--root-log-every" => {
                root_log_every =
                    args.get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Invalid --root-log-every value");
                            process::exit(1);
                        });
                i += 2;
            }
            "--log-flush-every" => {
                log_flush_every =
                    args.get(i + 1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or_else(|| {
                            eprintln!("Invalid --log-flush-every value");
                            process::exit(1);
                        });
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz selfplay`: {}", other);
                eprintln!("Run `yz selfplay --help` for usage.");
                process::exit(1);
            }
        }
    }

    let config_path = config_path.unwrap_or_else(|| {
        eprintln!("Missing --config");
        process::exit(1);
    });
    let infer = infer.unwrap_or_else(|| {
        eprintln!("Missing --infer");
        process::exit(1);
    });
    let out = out.unwrap_or_else(|| {
        eprintln!("Missing --out");
        process::exit(1);
    });

    let cfg = yz_core::Config::load(&config_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config: {e}");
        process::exit(1);
    });

    let replay_dir = PathBuf::from(&out).join("replay");
    let _ = yz_replay::cleanup_tmp_files(&replay_dir);

    let logs_dir = PathBuf::from(&out).join("logs");
    std::fs::create_dir_all(&logs_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create logs dir: {e}");
        process::exit(1);
    });

    let models_dir = PathBuf::from(&out).join("models");
    std::fs::create_dir_all(&models_dir).unwrap_or_else(|e| {
        eprintln!("Failed to create models dir: {e}");
        process::exit(1);
    });

    // E8.5.1: run manifest (runs/<id>/run.json).
    let run_json = PathBuf::from(&out).join("run.json");
    let config_bytes = std::fs::read(&config_path).unwrap_or_else(|e| {
        eprintln!("Failed to read config file: {e}");
        process::exit(1);
    });
    let config_hash = yz_logging::hash_config_bytes(&config_bytes);
    let run_id = PathBuf::from(&out)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or(&out)
        .to_string();
    let mut manifest = yz_logging::RunManifestV1 {
        run_manifest_version: yz_logging::RUN_MANIFEST_VERSION,
        run_id,
        created_ts_ms: yz_logging::now_ms(),
        protocol_version: yz_infer::protocol::PROTOCOL_VERSION,
        feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
        action_space_id: "oracle_keepmask_v1".to_string(),
        ruleset_id: "swedish_scandinavian_v1".to_string(),
        git_hash: yz_logging::try_git_hash(),
        config_hash: Some(config_hash),
        config_snapshot: None,
        config_snapshot_hash: None,
        replay_dir: "replay".to_string(),
        logs_dir: "logs".to_string(),
        models_dir: "models".to_string(),
        selfplay_games_completed: 0,
        train_step: 0,
        best_checkpoint: None,
        candidate_checkpoint: None,
        train_last_loss_total: None,
        train_last_loss_policy: None,
        train_last_loss_value: None,
        promotion_decision: None,
        promotion_ts_ms: None,
        gate_games: None,
        gate_win_rate: None,
        gate_draw_rate: None,
        gate_seeds_hash: None,
        gate_oracle_match_rate_overall: None,
        gate_oracle_match_rate_mark: None,
        gate_oracle_match_rate_reroll: None,
        gate_oracle_keepall_ignored: None,
        controller_phase: None,
        controller_status: None,
        controller_last_ts_ms: None,
        controller_error: None,
        model_reloads: 0,
        controller_iteration_idx: 0,
        iterations: Vec::new(),
    };
    // If a manifest already exists (resume), keep its created_ts_ms/run_id.
    if let Ok(existing) = yz_logging::read_manifest(&run_json) {
        manifest.created_ts_ms = existing.created_ts_ms;
        manifest.run_id = existing.run_id;
        manifest.train_step = existing.train_step;
        manifest.best_checkpoint = existing.best_checkpoint;
        manifest.candidate_checkpoint = existing.candidate_checkpoint;
        manifest.train_last_loss_total = existing.train_last_loss_total;
        manifest.train_last_loss_policy = existing.train_last_loss_policy;
        manifest.train_last_loss_value = existing.train_last_loss_value;
        manifest.promotion_decision = existing.promotion_decision;
        manifest.promotion_ts_ms = existing.promotion_ts_ms;
        manifest.gate_games = existing.gate_games;
        manifest.gate_win_rate = existing.gate_win_rate;
        manifest.gate_draw_rate = existing.gate_draw_rate;
        manifest.gate_seeds_hash = existing.gate_seeds_hash;
        manifest.gate_oracle_match_rate_overall = existing.gate_oracle_match_rate_overall;
        manifest.gate_oracle_match_rate_mark = existing.gate_oracle_match_rate_mark;
        manifest.gate_oracle_match_rate_reroll = existing.gate_oracle_match_rate_reroll;
        manifest.gate_oracle_keepall_ignored = existing.gate_oracle_keepall_ignored;
        manifest.config_snapshot = existing.config_snapshot;
        manifest.config_snapshot_hash = existing.config_snapshot_hash;
        manifest.controller_phase = existing.controller_phase;
        manifest.controller_status = existing.controller_status;
        manifest.controller_last_ts_ms = existing.controller_last_ts_ms;
        manifest.controller_error = existing.controller_error;
        manifest.controller_iteration_idx = existing.controller_iteration_idx;
        manifest.iterations = existing.iterations;
    }

    // E10.5S1: run-local config snapshot (normalized).
    if manifest.config_snapshot.is_none() || !PathBuf::from(&out).join("config.yaml").exists() {
        if let Ok((rel, h)) = yz_logging::write_config_snapshot_atomic(&out, &cfg) {
            manifest.config_snapshot = Some(rel);
            manifest.config_snapshot_hash = Some(h);
        }
    }
    yz_logging::write_manifest_atomic(&run_json, &manifest).unwrap_or_else(|e| {
        eprintln!("Failed to write run manifest: {e:?}");
        process::exit(1);
    });

    let backend = connect_infer_backend(&infer);
    let mut writer = yz_replay::ShardWriter::new(yz_replay::ShardWriterConfig {
        out_dir: replay_dir.clone(),
        max_samples_per_shard,
        git_hash: None,
        config_hash: None,
    })
    .unwrap_or_else(|e| {
        eprintln!("Failed to create shard writer: {e}");
        process::exit(1);
    });

    let parallel = cfg.selfplay.threads_per_worker.max(1) as usize;
    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations_mark: cfg.mcts.budget_mark.max(1),
        simulations_reroll: cfg.mcts.budget_reroll.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: cfg.mcts.dirichlet_epsilon,
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss: 1.0,
    };

    let mut tasks = Vec::new();
    for gid in 0..parallel as u64 {
        let mut ctx = yz_core::TurnContext::new_rng(0xC0FFEE ^ gid);
        let s = yz_core::initial_state(&mut ctx);
        tasks.push(yz_runtime::GameTask::new(
            gid,
            s,
            yz_mcts::ChanceMode::Rng {
                seed: 0xBADC0DE ^ gid,
            },
            mcts_cfg,
        ));
    }
    let mut sched = yz_runtime::Scheduler::new(tasks, 64);

    let run_id = out.clone();
    let v = yz_logging::VersionInfoV1 {
        protocol_version: yz_infer::protocol::PROTOCOL_VERSION,
        feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
        action_space_id: "oracle_keepmask_v1",
        ruleset_id: "swedish_scandinavian_v1",
    };
    let iter_log_path = logs_dir.join("iteration_stats.ndjson");
    let roots_log_path = logs_dir.join("mcts_roots.ndjson");
    let mut loggers = yz_runtime::RunLoggers {
        run_id,
        v,
        git_hash: manifest.git_hash.clone(),
        config_snapshot: manifest.config_snapshot.clone(),
        root_log_every_n: root_log_every,
        iter: yz_logging::NdjsonWriter::open_append_with_flush(iter_log_path, log_flush_every)
            .unwrap_or_else(|e| {
                eprintln!("Failed to create iteration log: {e:?}");
                process::exit(1);
            }),
        roots: yz_logging::NdjsonWriter::open_append_with_flush(roots_log_path, log_flush_every)
            .unwrap_or_else(|e| {
                eprintln!("Failed to create root log: {e:?}");
                process::exit(1);
            }),
        metrics: yz_logging::NdjsonWriter::open_append_with_flush(
            logs_dir.join("metrics.ndjson"),
            log_flush_every,
        )
        .unwrap_or_else(|e| {
            eprintln!("Failed to create metrics log: {e:?}");
            process::exit(1);
        }),
    };

    let mut completed_games: u32 = 0;
    let mut next_game_id: u64 = parallel as u64;
    while completed_games < games {
        sched
            .tick_and_write(&backend, &mut writer, Some(&mut loggers))
            .unwrap_or_else(|e| {
                eprintln!("Replay write error: {e}");
                process::exit(1);
            });

        for t in sched.tasks_mut() {
            if yz_core::is_terminal(&t.state) {
                completed_games += 1;
                if completed_games.is_multiple_of(10) || completed_games == games {
                    manifest.selfplay_games_completed = completed_games as u64;
                    let _ = yz_logging::write_manifest_atomic(&run_json, &manifest);
                }
                if completed_games >= games {
                    break;
                }
                // Reset task for next game.
                let mut ctx = yz_core::TurnContext::new_rng(0xC0FFEE ^ next_game_id);
                let s = yz_core::initial_state(&mut ctx);
                *t = yz_runtime::GameTask::new(
                    next_game_id,
                    s,
                    yz_mcts::ChanceMode::Rng {
                        seed: 0xBADC0DE ^ next_game_id,
                    },
                    mcts_cfg,
                );
                next_game_id += 1;
            }
        }
    }

    writer.finish().unwrap_or_else(|e| {
        eprintln!("Failed to flush writer: {e}");
        process::exit(1);
    });
    let _ = loggers.iter.flush();
    let _ = loggers.roots.flush();

    // E13.1S1: replay pruning (optional) + metrics event.
    if let Some(cap) = cfg.replay.capacity_shards {
        if cap > 0 {
            match yz_replay::prune_shards_by_idx(&replay_dir, cap as usize) {
                Ok(rep) => {
                    let ev = yz_logging::MetricsReplayPruneV1 {
                        event: "replay_prune",
                        ts_ms: yz_logging::now_ms(),
                        v: loggers.v.clone(),
                        run_id: loggers.run_id.clone(),
                        git_hash: loggers.git_hash.clone(),
                        config_snapshot: loggers.config_snapshot.clone(),
                        capacity_shards: cap,
                        before_shards: rep.before_shards as u32,
                        after_shards: rep.after_shards as u32,
                        deleted_shards: rep.deleted_shards as u32,
                        deleted_min_idx: rep.deleted_min_idx,
                        deleted_max_idx: rep.deleted_max_idx,
                    };
                    let _ = loggers.metrics.write_event(&ev);
                }
                Err(e) => eprintln!("Replay prune error: {e}"),
            }
        }
    }
    let _ = loggers.metrics.flush();

    // Final manifest update.
    manifest.selfplay_games_completed = completed_games as u64;
    let _ = yz_logging::write_manifest_atomic(&run_json, &manifest);

    println!("Self-play complete. Games={games} out={out}");
}

fn cmd_gate_worker(args: &[String]) {
    use std::path::PathBuf;

    let mut run_dir: Option<String> = None;
    let mut infer: Option<String> = None;
    let mut worker_id: u32 = 0;
    let mut num_workers: u32 = 1;
    let mut best_id: u32 = 0;
    let mut cand_id: u32 = 1;
    let mut schedule_file: Option<String> = None;
    let mut out_path: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz gate-worker

USAGE:
    yz gate-worker --run-dir runs/<id> --infer unix:///tmp/yatzy_infer.sock --worker-id W --num-workers N --best-id 0 --cand-id 1 --schedule-file PATH --out PATH

NOTES:
    - Internal command spawned by the controller for parallel gating.
    - Does NOT write replay shards; only writes results/progress.

OPTIONS:
    --run-dir DIR         Run directory (contains run.json + config.yaml) (required)
    --infer ENDPOINT      Inference endpoint (unix:///... or tcp://host:port) (required)
    --worker-id W         Worker id in [0, N) (required)
    --num-workers N       Total worker processes (required)
    --best-id N           model_id for best (default 0)
    --cand-id N           model_id for candidate (default 1)
    --schedule-file PATH  JSON schedule file for this worker (required)
    --out PATH            Output JSON result file path (required)
"#
                );
                return;
            }
            "--run-dir" => {
                run_dir = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--infer" => {
                infer = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--worker-id" => {
                worker_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --worker-id value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--num-workers" => {
                num_workers = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --num-workers value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--best-id" => {
                best_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(best_id);
                i += 2;
            }
            "--cand-id" => {
                cand_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(cand_id);
                i += 2;
            }
            "--schedule-file" => {
                schedule_file = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--out" => {
                out_path = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz gate-worker`: {other}");
                eprintln!("Run `yz gate-worker --help` for usage.");
                process::exit(1);
            }
        }
    }

    let run_dir = run_dir.unwrap_or_else(|| {
        eprintln!("Missing --run-dir");
        process::exit(1);
    });
    let infer = infer.unwrap_or_else(|| {
        eprintln!("Missing --infer");
        process::exit(1);
    });
    let schedule_file = schedule_file.unwrap_or_else(|| {
        eprintln!("Missing --schedule-file");
        process::exit(1);
    });
    let out_path = out_path.unwrap_or_else(|| {
        eprintln!("Missing --out");
        process::exit(1);
    });

    let run_dir = PathBuf::from(run_dir);
    let cfg_path = run_dir.join("config.yaml");
    let cfg = yz_core::Config::load(&cfg_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config.yaml: {e}");
        process::exit(1);
    });

    #[derive(serde::Deserialize)]
    struct GameSpecJson {
        episode_seed: u64,
        swap: bool,
    }
    let sched_bytes = std::fs::read(&schedule_file).unwrap_or_else(|e| {
        eprintln!("Failed to read schedule file: {e}");
        process::exit(1);
    });
    let sched: Vec<GameSpecJson> = serde_json::from_slice(&sched_bytes).unwrap_or_else(|e| {
        eprintln!("Failed to parse schedule JSON: {e}");
        process::exit(1);
    });
    let schedule: Vec<yz_eval::GameSpec> = sched
        .into_iter()
        .map(|s| yz_eval::GameSpec {
            episode_seed: s.episode_seed,
            swap: s.swap,
        })
        .collect();

    let games_target = schedule.len() as u32;

    // Bounded inflight to avoid flooding the inference server.
    let client_opts = yz_infer::ClientOptions {
        max_inflight_total: 64,
        max_outbound_queue: 256,
        request_id_start: 1,
    };
    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations_mark: cfg.mcts.budget_mark.max(1),
        simulations_reroll: cfg.mcts.budget_reroll.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: 0.0, // gating: no root noise
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss: 1.0,
    };

    // Progress + output directories (mirrors selfplay-worker layout, but under logs_gate_workers).
    let logs_dir = run_dir
        .join("logs_gate_workers")
        .join(format!("worker_{worker_id:03}"));
    let _ = std::fs::create_dir_all(&logs_dir);
    let progress_path = logs_dir.join("progress.json");

    #[derive(serde::Serialize)]
    struct GateWorkerProgress {
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        games_completed: u32,
        pid: u32,
        ts_ms: u64,
    }

    fn write_progress_atomic(path: &PathBuf, p: &GateWorkerProgress) {
        let tmp = path.with_extension("json.tmp");
        if let Ok(bytes) = serde_json::to_vec(p) {
            let _ = std::fs::write(&tmp, &bytes);
            let _ = std::fs::rename(&tmp, path);
        }
    }

    // Initial progress file for live aggregation.
    write_progress_atomic(
        &progress_path,
        &GateWorkerProgress {
            worker_id,
            num_workers,
            games_target,
            games_completed: 0,
            pid: std::process::id(),
            ts_ms: yz_logging::now_ms(),
        },
    );

    struct ProgressSink {
        progress_path: PathBuf,
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        worker_stats: Option<yz_logging::NdjsonWriter>,
        t_start: std::time::Instant,
        first_done_logged: bool,
    }
    impl yz_eval::GateProgress for ProgressSink {
        fn on_game_completed(&mut self, completed: u32, _total: u32) {
            write_progress_atomic(
                &self.progress_path,
                &GateWorkerProgress {
                    worker_id: self.worker_id,
                    num_workers: self.num_workers,
                    games_target: self.games_target,
                    games_completed: completed,
                    pid: std::process::id(),
                    ts_ms: yz_logging::now_ms(),
                },
            );

            if !self.first_done_logged && completed >= 1 {
                if let Some(w) = self.worker_stats.as_mut() {
                    let _ = w.write_event(&GateWorkerStatsEvent {
                        event: "gate_worker_first_game_done",
                        ts_ms: yz_logging::now_ms(),
                        worker_id: self.worker_id,
                        num_workers: self.num_workers,
                        games_target: self.games_target,
                        wall_ms: self.t_start.elapsed().as_millis() as u64,
                    });
                    let _ = w.flush();
                }
                self.first_done_logged = true;
            }
        }
    }

    // Oracle diag log (ndjson, best-effort).
    let mut oracle_log =
        yz_logging::NdjsonWriter::open_append_with_flush(logs_dir.join("oracle_diag.ndjson"), 200)
            .ok();
    // Worker timing stats (ndjson, best-effort).
    let mut worker_stats =
        yz_logging::NdjsonWriter::open_append_with_flush(logs_dir.join("worker_stats.ndjson"), 50)
            .ok();
    #[derive(serde::Serialize)]
    struct GateWorkerStatsEvent<'a> {
        event: &'a str,
        ts_ms: u64,
        worker_id: u32,
        num_workers: u32,
        games_target: u32,
        wall_ms: u64,
    }
    let t_start = std::time::Instant::now();
    if let Some(w) = worker_stats.as_mut() {
        let _ = w.write_event(&GateWorkerStatsEvent {
            event: "gate_worker_start",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target,
            wall_ms: 0,
        });
        let _ = w.flush();
    }

    struct OracleSink<'a> {
        w: &'a mut yz_logging::NdjsonWriter,
    }
    impl yz_eval::OracleDiagSink for OracleSink<'_> {
        fn on_step(&mut self, ev: &yz_eval::OracleDiagEvent) {
            // Best-effort; don't crash worker for logging.
            let _ = self.w.write_event(ev);
        }
    }
    let mut oracle_sink = oracle_log.as_mut().map(|w| OracleSink { w });

    if let Some(w) = worker_stats.as_mut() {
        let _ = w.write_event(&GateWorkerStatsEvent {
            event: "gate_worker_first_game_start",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target,
            wall_ms: t_start.elapsed().as_millis() as u64,
        });
        let _ = w.flush();
    }

    let mut sink = ProgressSink {
        progress_path: progress_path.clone(),
        worker_id,
        num_workers,
        games_target,
        worker_stats,
        t_start,
        first_done_logged: false,
    };

    let partial = yz_eval::gate_schedule_subset(
        &cfg,
        yz_eval::GateOptions {
            infer_endpoint: infer,
            best_model_id: best_id,
            cand_model_id: cand_id,
            client_opts,
            mcts_cfg,
        },
        &schedule,
        Some(&mut sink),
        oracle_sink
            .as_mut()
            .map(|s| s as &mut dyn yz_eval::OracleDiagSink),
    )
    .unwrap_or_else(|e| {
        eprintln!("gate-worker failed: {e}");
        process::exit(1);
    });

    // Ensure final progress is complete.
    write_progress_atomic(
        &progress_path,
        &GateWorkerProgress {
            worker_id,
            num_workers,
            games_target,
            games_completed: games_target,
            pid: std::process::id(),
            ts_ms: yz_logging::now_ms(),
        },
    );

    if let Some(w) = sink.worker_stats.as_mut() {
        let _ = w.write_event(&GateWorkerStatsEvent {
            event: "gate_worker_done",
            ts_ms: yz_logging::now_ms(),
            worker_id,
            num_workers,
            games_target,
            wall_ms: t_start.elapsed().as_millis() as u64,
        });
        let _ = w.flush();
    }

    #[derive(serde::Serialize, serde::Deserialize)]
    struct GateWorkerResult {
        worker_id: u32,
        games: u32,
        cand_wins: u32,
        cand_losses: u32,
        draws: u32,
        cand_score_diff_sum: i64,
        cand_score_diff_sumsq: f64,
    }

    let out_path = PathBuf::from(out_path);
    let tmp = out_path.with_extension("json.tmp");
    let res = GateWorkerResult {
        worker_id,
        games: partial.games,
        cand_wins: partial.cand_wins,
        cand_losses: partial.cand_losses,
        draws: partial.draws,
        cand_score_diff_sum: partial.cand_score_diff_sum,
        cand_score_diff_sumsq: partial.cand_score_diff_sumsq,
    };
    let bytes = serde_json::to_vec(&res).expect("serialize gate worker result");
    std::fs::write(&tmp, bytes).unwrap_or_else(|e| {
        eprintln!("Failed to write gate worker result: {e}");
        process::exit(1);
    });
    std::fs::rename(&tmp, &out_path).unwrap_or_else(|e| {
        eprintln!("Failed to rename gate worker result: {e}");
        process::exit(1);
    });
}

fn cmd_gate(args: &[String]) {
    let mut config_path: Option<String> = None;
    let mut infer: Option<String> = None;
    let mut best_id: u32 = 0;
    let mut cand_id: u32 = 1;
    let mut run_dir: Option<String> = None;
    let mut out_path: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz gate

USAGE:
    yz gate --config cfg.yaml [--infer unix:///tmp/yatzy_infer.sock] [--best-id 0] [--cand-id 1] [--run runs/<id>/]

NOTES:
    - Candidate vs best are selected by model_id routing on the inference server.
    - If gating.paired_seed_swap=true, gating.games must be even (two games per seed, side-swapped).

OPTIONS:
    --config PATH       Config YAML path
    --infer ENDPOINT    Override inference endpoint (unix:///... or tcp://host:port). Defaults to config.inference.bind
    --best-id N         model_id for best (default 0)
    --cand-id N         model_id for candidate (default 1)
    --run DIR           Optional run dir to update runs/<id>/run.json with gate stats
"#
                );
                return;
            }
            "--config" => {
                config_path = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--infer" => {
                infer = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--best-id" => {
                best_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(best_id);
                i += 2;
            }
            "--cand-id" => {
                cand_id = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(cand_id);
                i += 2;
            }
            "--run" => {
                run_dir = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--out" => {
                out_path = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz gate`: {other}");
                eprintln!("Run `yz gate --help` for usage.");
                process::exit(1);
            }
        }
    }

    let config_path = config_path.unwrap_or_else(|| {
        eprintln!("Missing --config");
        process::exit(1);
    });

    let cfg = yz_core::Config::load(&config_path).unwrap_or_else(|e| {
        eprintln!("Failed to load config: {e}");
        process::exit(1);
    });

    let infer_ep = infer.unwrap_or_else(|| cfg.inference.bind.clone());

    // Use bounded inflight to prevent flooding the inference server.
    // With N workers, system-wide max = N * 64. Server healthy capacity ~50-150.
    let client_opts = yz_infer::ClientOptions {
        max_inflight_total: 64,
        max_outbound_queue: 256,
        request_id_start: 1,
    };

    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations_mark: cfg.mcts.budget_mark.max(1),
        simulations_reroll: cfg.mcts.budget_reroll.max(1),
        dirichlet_alpha: cfg.mcts.dirichlet_alpha,
        dirichlet_epsilon: 0.0, // gating: no root noise
        max_inflight: cfg.mcts.max_inflight_per_game.max(1) as usize,
        virtual_loss: 1.0,
    };

    let report = yz_eval::gate(
        &cfg,
        yz_eval::GateOptions {
            infer_endpoint: infer_ep.clone(),
            best_model_id: best_id,
            cand_model_id: cand_id,
            client_opts,
            mcts_cfg,
        },
    )
    .unwrap_or_else(|e| {
        eprintln!("Gating failed: {e}");
        process::exit(1);
    });

    let wr = report.win_rate();
    let decision = if wr + 1e-12 < cfg.gating.win_rate_threshold {
        "reject"
    } else {
        "promote"
    };
    if let Some(id) = cfg.gating.seed_set_id.as_deref() {
        println!(
            "Seed source: seed_set_id={id} requested_games={}",
            cfg.gating.games
        );
    } else {
        println!(
            "Seed source: seed={} requested_games={}",
            cfg.gating.seed, cfg.gating.games
        );
    }
    for w in &report.warnings {
        eprintln!("warning: {w}");
    }
    println!(
        "Gating complete. decision={} games={} wins={} losses={} draws={} win_rate={:.4} mean_score_diff={:.2} se={:.3} ci95=[{:.3},{:.3}] seeds_hash={}",
        decision,
        report.games,
        report.cand_wins,
        report.cand_losses,
        report.draws,
        wr,
        report.mean_score_diff(),
        report.score_diff_se,
        report.score_diff_ci95_low,
        report.score_diff_ci95_high,
        report.seeds_hash
    );

    if let Some(run) = run_dir {
        let run_dir = PathBuf::from(run);
        let run_json = run_dir.join("run.json");
        match yz_logging::read_manifest(&run_json) {
            Ok(mut m) => {
                m.gate_games = Some(report.games as u64);
                m.gate_win_rate = Some(wr);
                m.gate_seeds_hash = Some(report.seeds_hash.clone());
                m.gate_oracle_match_rate_overall = Some(report.oracle_match_rate_overall);
                m.gate_oracle_match_rate_mark = Some(report.oracle_match_rate_mark);
                m.gate_oracle_match_rate_reroll = Some(report.oracle_match_rate_reroll);
                m.gate_oracle_keepall_ignored = Some(report.oracle_keepall_ignored);

                // E10.5S1: ensure run-local config snapshot exists (normalized).
                if m.config_snapshot.is_none() || !run_dir.join("config.yaml").exists() {
                    if let Ok((rel, h)) = yz_logging::write_config_snapshot_atomic(&run_dir, &cfg) {
                        m.config_snapshot = Some(rel);
                        m.config_snapshot_hash = Some(h);
                    }
                }

                let _ = yz_logging::write_manifest_atomic(&run_json, &m);

                // E10.5S2: unified metrics stream.
                let logs_dir = run_dir.join(&m.logs_dir);
                let _ = std::fs::create_dir_all(&logs_dir);
                let metrics_path = logs_dir.join("metrics.ndjson");
                if let Ok(mut w) = yz_logging::NdjsonWriter::open_append(metrics_path) {
                    let ev = yz_logging::MetricsGateSummaryV1 {
                        event: "gate_summary",
                        ts_ms: yz_logging::now_ms(),
                        v: yz_logging::VersionInfoV1 {
                            protocol_version: m.protocol_version,
                            feature_schema_id: m.feature_schema_id,
                            action_space_id: "oracle_keepmask_v1",
                            ruleset_id: "swedish_scandinavian_v1",
                        },
                        run_id: m.run_id.clone(),
                        git_hash: m.git_hash.clone(),
                        config_snapshot: m.config_snapshot.clone(),
                        decision: decision.to_string(),
                        games: report.games,
                        wins: report.cand_wins,
                        losses: report.cand_losses,
                        draws: report.draws,
                        win_rate: wr,
                        mean_score_diff: report.mean_score_diff(),
                        score_diff_se: report.score_diff_se,
                        score_diff_ci95_low: report.score_diff_ci95_low,
                        score_diff_ci95_high: report.score_diff_ci95_high,
                        seeds_hash: report.seeds_hash.clone(),
                        oracle_match_rate_overall: report.oracle_match_rate_overall,
                        oracle_match_rate_mark: report.oracle_match_rate_mark,
                        oracle_match_rate_reroll: report.oracle_match_rate_reroll,
                        oracle_keepall_ignored: report.oracle_keepall_ignored,
                    };
                    let _ = w.write_event(&ev);
                    let _ = w.flush();
                }
            }
            Err(e) => {
                eprintln!("Warning: failed to update run manifest: {e:?}");
            }
        }

        // Write gate_report.json (default under runs/<id>/ unless --out provided).
        let out = out_path
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(|| run_dir.join("gate_report.json"));
        let _ = write_gate_report_atomic(
            &out,
            &GateReportJson {
                decision: decision.to_string(),
                games: report.games,
                wins: report.cand_wins,
                losses: report.cand_losses,
                draws: report.draws,
                win_rate: wr,
                win_rate_threshold: cfg.gating.win_rate_threshold,
                mean_score_diff: report.mean_score_diff(),
                score_diff_se: report.score_diff_se,
                score_diff_ci95_low: report.score_diff_ci95_low,
                score_diff_ci95_high: report.score_diff_ci95_high,
                seeds_hash: report.seeds_hash.clone(),
                seed: cfg.gating.seed,
                seed_set_id: cfg.gating.seed_set_id.clone(),
                warnings: report.warnings.clone(),
                oracle_match_rate_overall: report.oracle_match_rate_overall,
                oracle_match_rate_mark: report.oracle_match_rate_mark,
                oracle_match_rate_reroll: report.oracle_match_rate_reroll,
                oracle_keepall_ignored: report.oracle_keepall_ignored,
            },
        );
    }

    if decision == "reject" {
        eprintln!(
            "Candidate rejected: win_rate {:.4} < threshold {:.4}",
            wr, cfg.gating.win_rate_threshold
        );
        process::exit(2);
    }
}

#[derive(serde::Serialize)]
struct GateReportJson {
    decision: String,
    games: u32,
    wins: u32,
    losses: u32,
    draws: u32,
    win_rate: f64,
    win_rate_threshold: f64,
    mean_score_diff: f64,
    score_diff_se: f64,
    score_diff_ci95_low: f64,
    score_diff_ci95_high: f64,
    seeds_hash: String,
    seed: u64,
    seed_set_id: Option<String>,
    warnings: Vec<String>,
    oracle_match_rate_overall: f64,
    oracle_match_rate_mark: f64,
    oracle_match_rate_reroll: f64,
    oracle_keepall_ignored: u64,
}

fn write_gate_report_atomic(path: &PathBuf, report: &GateReportJson) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(report).expect("serialize gate_report");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn connect_infer_backend(endpoint: &str) -> yz_mcts::InferBackend {
    // Use bounded inflight to prevent flooding the inference server.
    let opts = yz_infer::ClientOptions {
        max_inflight_total: 64,
        max_outbound_queue: 256,
        request_id_start: 1,
    };
    if let Some(rest) = endpoint.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            return yz_mcts::InferBackend::connect_uds(rest, 0, opts).unwrap();
        }
        #[cfg(not(unix))]
        {
            panic!("unix:// endpoints are only supported on unix");
        }
    }
    if let Some(rest) = endpoint.strip_prefix("tcp://") {
        return yz_mcts::InferBackend::connect_tcp(rest, 0, opts).unwrap();
    }
    panic!("Unsupported infer endpoint: {endpoint}");
}

fn cmd_iter_finalize(args: &[String]) {
    let mut run: Option<String> = None;
    let mut decision: Option<String> = None;

    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz iter finalize

USAGE:
    yz iter finalize --run runs/<id>/ --decision promote|reject

OPTIONS:
    --run DIR            Run directory (contains run.json, models/, replay/, logs/)
    --decision D         promote|reject
"#
                );
                return;
            }
            "--run" => {
                run = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--decision" => {
                decision = Some(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            other => {
                eprintln!("Unknown option for `yz iter finalize`: {}", other);
                eprintln!("Run `yz iter finalize --help` for usage.");
                process::exit(1);
            }
        }
    }

    let run = run.unwrap_or_else(|| {
        eprintln!("Missing --run");
        process::exit(1);
    });
    let decision = decision.unwrap_or_else(|| {
        eprintln!("Missing --decision");
        process::exit(1);
    });
    if decision != "promote" && decision != "reject" {
        eprintln!("Invalid --decision (expected promote|reject)");
        process::exit(1);
    }

    let run_dir = PathBuf::from(&run);
    let run_json = run_dir.join("run.json");
    let mut manifest = yz_logging::read_manifest(&run_json).unwrap_or_else(|e| {
        eprintln!("Failed to read run manifest: {e:?}");
        process::exit(1);
    });

    let models_dir = run_dir.join(&manifest.models_dir);
    let candidate_pt = models_dir.join("candidate.pt");
    if !candidate_pt.exists() {
        eprintln!("Missing candidate checkpoint: {}", candidate_pt.display());
        process::exit(1);
    }

    // Promote: copy candidate -> best atomically via temp+rename.
    if decision == "promote" {
        let best_pt = models_dir.join("best.pt");
        let tmp_best = models_dir.join("best.pt.tmp");
        let bytes = std::fs::read(&candidate_pt).unwrap_or_else(|e| {
            eprintln!("Failed to read candidate.pt: {e}");
            process::exit(1);
        });
        std::fs::write(&tmp_best, bytes).unwrap_or_else(|e| {
            eprintln!("Failed to write tmp best.pt: {e}");
            process::exit(1);
        });
        std::fs::rename(&tmp_best, &best_pt).unwrap_or_else(|e| {
            eprintln!("Failed to rename best.pt: {e}");
            process::exit(1);
        });

        // best.meta.json: derived from candidate.meta.json if present, else minimal.
        let cand_meta = models_dir.join("candidate.meta.json");
        let best_meta = models_dir.join("best.meta.json");
        if cand_meta.exists() {
            let tmp_meta = models_dir.join("best.meta.json.tmp");
            let meta_bytes = std::fs::read(&cand_meta).unwrap_or_else(|e| {
                eprintln!("Failed to read candidate.meta.json: {e}");
                process::exit(1);
            });
            std::fs::write(&tmp_meta, meta_bytes).unwrap_or_else(|e| {
                eprintln!("Failed to write tmp best.meta.json: {e}");
                process::exit(1);
            });
            std::fs::rename(&tmp_meta, &best_meta).unwrap_or_else(|e| {
                eprintln!("Failed to rename best.meta.json: {e}");
                process::exit(1);
            });
        } else {
            let tmp_meta = models_dir.join("best.meta.json.tmp");
            let meta = serde_json::json!({
                "protocol_version": manifest.protocol_version,
                "feature_schema_id": manifest.feature_schema_id,
                "action_space_id": manifest.action_space_id,
                "ruleset_id": manifest.ruleset_id,
            });
            std::fs::write(&tmp_meta, serde_json::to_vec_pretty(&meta).unwrap()).unwrap_or_else(
                |e| {
                    eprintln!("Failed to write tmp best.meta.json: {e}");
                    process::exit(1);
                },
            );
            std::fs::rename(&tmp_meta, &best_meta).unwrap_or_else(|e| {
                eprintln!("Failed to rename best.meta.json: {e}");
                process::exit(1);
            });
        }

        manifest.best_checkpoint = Some("models/best.pt".to_string());
    }

    manifest.promotion_decision = Some(decision.clone());
    manifest.promotion_ts_ms = Some(yz_logging::now_ms());
    manifest.candidate_checkpoint = Some("models/candidate.pt".to_string());

    yz_logging::write_manifest_atomic(&run_json, &manifest).unwrap_or_else(|e| {
        eprintln!("Failed to write run manifest: {e:?}");
        process::exit(1);
    });

    println!(
        "Finalize complete. decision={decision} run={}",
        run_dir.display()
    );
}

fn cmd_bench(args: &[String]) {
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            r#"yz bench

USAGE:
    yz bench [<cargo args>...]
    yz bench e2e -- [OPTIONS]

NOTES:
    - This is a thin wrapper around:
        cargo bench -p yz-bench <cargo args...>
    - E2E benchmark is a separate harness crate:
        cargo run -p yz-bench-e2e -- [OPTIONS]

EXAMPLES:
    yz bench
    yz bench --bench scoring
    yz bench --bench scoring -- --warm-up-time 0.5 --measurement-time 1.0
    yz bench e2e -- --seconds 10 --parallel 8 --simulations 64 --max-inflight 4 --chance deterministic
"#
        );
        return;
    }

    // Subcommand: yz bench e2e -- [opts]
    if args.first().map(|s| s.as_str()) == Some("e2e") {
        let mut cmd = Command::new("cargo");
        cmd.arg("run").arg("-p").arg("yz-bench-e2e").arg("--");
        cmd.args(&args[1..]);
        let status = cmd.status().unwrap_or_else(|e| {
            eprintln!("Failed to run e2e bench harness: {e}");
            eprintln!("Hint: ensure Rust tooling is installed and `cargo` is on PATH.");
            process::exit(1);
        });
        if !status.success() {
            process::exit(status.code().unwrap_or(1));
        }
        return;
    }

    let mut cmd = Command::new("cargo");
    cmd.arg("bench").arg("-p").arg("yz-bench");
    cmd.args(args);

    let status = cmd.status().unwrap_or_else(|e| {
        eprintln!("Failed to run cargo bench: {e}");
        eprintln!("Hint: ensure Rust tooling is installed and `cargo` is on PATH.");
        process::exit(1);
    });
    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }
}

fn cargo_flamegraph_available() -> bool {
    let out = Command::new("cargo")
        .args(["flamegraph", "--version"])
        .output();
    matches!(out, Ok(o) if o.status.success())
}

fn spawn_current_exe(args: &[String]) -> process::ExitStatus {
    let exe = env::current_exe().unwrap_or_else(|e| {
        eprintln!("Failed to locate current executable: {e}");
        process::exit(1);
    });
    Command::new(exe).args(args).status().unwrap_or_else(|e| {
        eprintln!("Failed to spawn yz: {e}");
        process::exit(1);
    })
}

fn cmd_profile(args: &[String]) {
    if args.is_empty() || args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            r#"yz profile

USAGE:
    yz profile <selfplay|gate|bench-e2e> -- <args...>

NOTES:
    - This command is a thin wrapper around `cargo flamegraph`.
    - If `cargo flamegraph` is not installed, we fall back to running the underlying command normally.

EXAMPLES:
    yz profile selfplay -- --help
    yz profile bench-e2e -- --seconds 10 --parallel 8 --simulations 64 --max-inflight 4 --chance deterministic

INSTALL:
    cargo install flamegraph
"#
        );
        return;
    }

    let target = args[0].as_str();
    let sep = args.iter().position(|s| s == "--");
    let (before, after) = match sep {
        Some(i) => (&args[1..i], &args[(i + 1)..]),
        None => {
            eprintln!("Missing `--` separator. See `yz profile --help`.");
            process::exit(2);
        }
    };
    if !before.is_empty() {
        eprintln!("Unexpected args before `--`: {before:?}");
        eprintln!("See `yz profile --help`.");
        process::exit(2);
    }

    // Underlying command args (when we fall back).
    let mut underlying: Vec<String> = Vec::new();
    match target {
        "selfplay" => {
            underlying.push("selfplay".to_string());
            underlying.extend_from_slice(after);
        }
        "gate" => {
            underlying.push("gate".to_string());
            underlying.extend_from_slice(after);
        }
        "bench-e2e" => {
            underlying.push("bench".to_string());
            underlying.push("e2e".to_string());
            underlying.push("--".to_string());
            underlying.extend_from_slice(after);
        }
        other => {
            eprintln!("Unknown profile target: {other}");
            eprintln!("See `yz profile --help`.");
            process::exit(2);
        }
    }

    if !cargo_flamegraph_available() {
        eprintln!("warning: `cargo flamegraph` not found (install with: cargo install flamegraph)");
        eprintln!("warning: running underlying command without profiling");
        let status = spawn_current_exe(&underlying);
        if !status.success() {
            process::exit(status.code().unwrap_or(1));
        }
        return;
    }

    // Profile path via cargo flamegraph.
    let mut cmd = Command::new("cargo");
    cmd.arg("flamegraph");
    match target {
        "selfplay" => {
            cmd.args(["--bin", "yz", "--"]);
            cmd.arg("selfplay");
            cmd.args(after);
        }
        "gate" => {
            cmd.args(["--bin", "yz", "--"]);
            cmd.arg("gate");
            cmd.args(after);
        }
        "bench-e2e" => {
            cmd.args(["-p", "yz-bench-e2e", "--bin", "yz-bench-e2e", "--"]);
            cmd.args(after);
        }
        _ => unreachable!(),
    }

    let status = cmd.status().unwrap_or_else(|e| {
        eprintln!("Failed to run cargo flamegraph: {e}");
        process::exit(1);
    });
    if !status.success() {
        process::exit(status.code().unwrap_or(1));
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        print_help();
        process::exit(0);
    }

    match args[1].as_str() {
        "-h" | "--help" | "help" => {
            print_help();
        }
        "-V" | "--version" => {
            print_version();
        }
        "oracle" => {
            if args.len() < 3 {
                eprintln!("Usage: yz oracle <expected|sim> [OPTIONS]");
                process::exit(1);
            }
            match args[2].as_str() {
                "expected" => {
                    cmd_oracle_expected();
                }
                "sim" => {
                    cmd_oracle_sim(&args[3..]);
                }
                _ => {
                    eprintln!("Unknown oracle subcommand: {}", args[2]);
                    process::exit(1);
                }
            }
        }
        "selfplay" => {
            cmd_selfplay(&args[2..]);
        }
        "selfplay-worker" => {
            cmd_selfplay_worker(&args[2..]);
        }
        "gate-worker" => {
            cmd_gate_worker(&args[2..]);
        }
        "iter" => {
            if args.len() < 3 {
                eprintln!("Usage: yz iter finalize [OPTIONS]");
                process::exit(1);
            }
            match args[2].as_str() {
                "finalize" => {
                    cmd_iter_finalize(&args[3..]);
                }
                other => {
                    eprintln!("Unknown iter subcommand: {other}");
                    eprintln!("Usage: yz iter finalize [OPTIONS]");
                    process::exit(1);
                }
            }
        }
        "gate" => {
            cmd_gate(&args[2..]);
        }
        "oracle-eval" => {
            println!("Oracle evaluation (not yet implemented)");
            println!("Usage: yz oracle-eval --config cfg.yaml --best ... --cand ...");
        }
        "bench" => {
            cmd_bench(&args[2..]);
        }
        "tui" => {
            if let Err(e) = yz_tui::run() {
                eprintln!("TUI failed: {e}");
                process::exit(1);
            }
        }
        "profile" => {
            cmd_profile(&args[2..]);
        }
        cmd => {
            eprintln!("Unknown command: {}", cmd);
            eprintln!("Run `yz --help` for usage.");
            process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cli_compiles() {
        // Basic sanity: the binary compiles and this test runs.
        assert!(true);
    }
}
