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
    gate                Gate candidate vs best model (paired seed + side swap)
    oracle-eval         Evaluate models against oracle baseline
    bench               Run micro-benchmarks
    profile             Run with profiler hooks enabled

OPTIONS:
    -h, --help          Print this help message
    -V, --version       Print version

For more information, see the PRD or run `yz <COMMAND> --help`.
"#
    );
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
                root_log_every = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| {
                        eprintln!("Invalid --root-log-every value");
                        process::exit(1);
                    });
                i += 2;
            }
            "--log-flush-every" => {
                log_flush_every = args
                    .get(i + 1)
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
        replay_dir: "replay".to_string(),
        logs_dir: "logs".to_string(),
        models_dir: "models".to_string(),
        selfplay_games_completed: 0,
        train_step: 0,
        best_checkpoint: None,
        candidate_checkpoint: None,
        promotion_decision: None,
        promotion_ts_ms: None,
        gate_games: None,
        gate_win_rate: None,
        gate_seeds_hash: None,
    };
    // If a manifest already exists (resume), keep its created_ts_ms/run_id.
    if let Ok(existing) = yz_logging::read_manifest(&run_json) {
        manifest.created_ts_ms = existing.created_ts_ms;
        manifest.run_id = existing.run_id;
        manifest.train_step = existing.train_step;
        manifest.best_checkpoint = existing.best_checkpoint;
        manifest.candidate_checkpoint = existing.candidate_checkpoint;
        manifest.promotion_decision = existing.promotion_decision;
        manifest.promotion_ts_ms = existing.promotion_ts_ms;
        manifest.gate_games = existing.gate_games;
        manifest.gate_win_rate = existing.gate_win_rate;
        manifest.gate_seeds_hash = existing.gate_seeds_hash;
    }
    yz_logging::write_manifest_atomic(&run_json, &manifest).unwrap_or_else(|e| {
        eprintln!("Failed to write run manifest: {e:?}");
        process::exit(1);
    });

    let backend = connect_infer_backend(&infer);
    let mut writer = yz_replay::ShardWriter::new(yz_replay::ShardWriterConfig {
        out_dir: replay_dir,
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
        simulations: cfg.mcts.budget_mark.max(1),
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
                if completed_games % 10 == 0 || completed_games == games {
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

    // Final manifest update.
    manifest.selfplay_games_completed = completed_games as u64;
    let _ = yz_logging::write_manifest_atomic(&run_json, &manifest);

    println!("Self-play complete. Games={games} out={out}");
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

    let client_opts = yz_infer::ClientOptions {
        max_inflight_total: 8192,
        max_outbound_queue: 8192,
        request_id_start: 1,
    };

    let mcts_cfg = yz_mcts::MctsConfig {
        c_puct: cfg.mcts.c_puct,
        simulations: cfg.mcts.budget_mark.max(1),
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
        println!("Seed source: seed_set_id={id} requested_games={}", cfg.gating.games);
    } else {
        println!("Seed source: seed={} requested_games={}", cfg.gating.seed, cfg.gating.games);
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
                let _ = yz_logging::write_manifest_atomic(&run_json, &m);
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
}

fn write_gate_report_atomic(path: &PathBuf, report: &GateReportJson) -> std::io::Result<()> {
    let tmp = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(report).expect("serialize gate_report");
    std::fs::write(&tmp, bytes)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

fn connect_infer_backend(endpoint: &str) -> yz_mcts::InferBackend {
    let opts = yz_infer::ClientOptions {
        max_inflight_total: 8192,
        max_outbound_queue: 8192,
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

    println!("Finalize complete. decision={decision} run={}", run_dir.display());
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
            println!("Benchmarks (not yet implemented)");
        }
        "profile" => {
            println!("Profiling (not yet implemented)");
            println!("Usage: yz profile selfplay ...");
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
