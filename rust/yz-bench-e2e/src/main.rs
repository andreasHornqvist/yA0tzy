use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

use yz_infer::codec::{decode_request_v1, encode_response_v1};
use yz_infer::frame::{read_frame, write_frame, FrameError};
use yz_infer::protocol::{InferResponseV1, ACTION_SPACE_A};
use yz_mcts::{ChanceMode, InferBackend, MctsConfig};
use yz_runtime::GameTask;

use yz_bench_e2e::E2eBenchJsonV1;

#[derive(Debug, Clone)]
struct Opts {
    seconds: Option<u64>,
    games: Option<u64>,
    parallel: u64,
    simulations: u32,
    max_inflight: u32,
    chance: String, // "deterministic" | "rng"
    server_delay_ms: u64,
}

fn parse_args() -> Opts {
    let mut seconds: Option<u64> = Some(10);
    let mut games: Option<u64> = None;
    let mut parallel: u64 = 8;
    let mut simulations: u32 = 64;
    let mut max_inflight: u32 = 4;
    let mut chance: String = "deterministic".to_string();
    let mut server_delay_ms: u64 = 0;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--help" | "-h" => {
                println!(
                    r#"yz-bench-e2e

USAGE:
  yz bench e2e -- [OPTIONS]

OPTIONS:
  --seconds T          Run for T seconds (default 10)
  --games N            Run until N games completed (overrides --seconds)
  --parallel P         Number of parallel games (default 8)
  --simulations S      MCTS simulations per move (default 64)
  --max-inflight K     Max inflight per search (default 4)
  --chance MODE        deterministic|rng (default deterministic)
  --server-delay-ms D  Dummy server per-request delay (default 0)
"#
                );
                std::process::exit(0);
            }
            "--seconds" => {
                seconds = Some(
                    args.get(i + 1)
                        .unwrap_or(&"10".to_string())
                        .parse()
                        .unwrap(),
                );
                i += 2;
            }
            "--games" => {
                games = Some(args.get(i + 1).unwrap_or(&"1".to_string()).parse().unwrap());
                seconds = None;
                i += 2;
            }
            "--parallel" => {
                parallel = args.get(i + 1).unwrap().parse().unwrap();
                i += 2;
            }
            "--simulations" => {
                simulations = args.get(i + 1).unwrap().parse().unwrap();
                i += 2;
            }
            "--max-inflight" => {
                max_inflight = args.get(i + 1).unwrap().parse().unwrap();
                i += 2;
            }
            "--chance" => {
                chance = args.get(i + 1).unwrap().to_string();
                i += 2;
            }
            "--server-delay-ms" => {
                server_delay_ms = args.get(i + 1).unwrap().parse().unwrap();
                i += 2;
            }
            other => {
                eprintln!("Unknown arg: {other}");
                eprintln!("Run with --help for usage.");
                std::process::exit(2);
            }
        }
    }

    Opts {
        seconds,
        games,
        parallel,
        simulations,
        max_inflight,
        chance,
        server_delay_ms,
    }
}

fn start_dummy_infer_server_tcp(delay_ms: u64) -> (SocketAddr, thread::JoinHandle<()>) {
    // For E2E bench we only need a single connection (one InferenceClient).
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("addr");
    let handle = thread::spawn(move || {
        let (sock, _peer) = listener.accept().expect("accept");
        handle_conn(sock, delay_ms);
    });
    (addr, handle)
}

fn handle_conn(mut sock: TcpStream, delay_ms: u64) {
    loop {
        let payload = match read_frame(&mut sock) {
            Ok(p) => p,
            // Important: avoid nonblocking/timeout here; partial reads would corrupt the stream.
            Err(FrameError::Io(_)) => break,
            Err(_) => break,
        };
        let req = match decode_request_v1(&payload) {
            Ok(r) => r,
            Err(_) => break,
        };
        if delay_ms > 0 {
            thread::sleep(Duration::from_millis(delay_ms));
        }
        let mut logits = vec![0.0f32; ACTION_SPACE_A as usize];
        for (i, &b) in req.legal_mask.iter().enumerate() {
            if b == 0 {
                logits[i] = -1.0e9;
            }
        }
        let resp = InferResponseV1 {
            request_id: req.request_id,
            policy_logits: logits,
            value: 0.0,
            margin: None,
        };
        let out = encode_response_v1(&resp);
        if write_frame(&mut sock, &out).is_err() {
            break;
        }
    }
}

fn main() {
    let opts = parse_args();

    let (addr, server) = start_dummy_infer_server_tcp(opts.server_delay_ms);
    let backend = InferBackend::connect_tcp(
        addr,
        0,
        yz_infer::ClientOptions {
            max_inflight_total: 8192,
            max_outbound_queue: 8192,
            request_id_start: 1,
        },
    )
    .expect("connect backend");

    let mode = match opts.chance.as_str() {
        "deterministic" => ChanceMode::Deterministic { episode_seed: 123 },
        "rng" => ChanceMode::Rng { seed: 123 },
        other => {
            eprintln!("invalid --chance: {other} (expected deterministic|rng)");
            std::process::exit(2);
        }
    };

    let mcts_cfg = MctsConfig {
        simulations_mark: opts.simulations,
        simulations_reroll: opts.simulations,
        max_inflight: opts.max_inflight as usize,
        ..MctsConfig::default()
    };

    let mut tasks: Vec<GameTask> = Vec::with_capacity(opts.parallel as usize);
    for i in 0..opts.parallel {
        let mut ctx = yz_core::TurnContext::new_rng(0xC0FFEE ^ i);
        let state = yz_core::initial_state(&mut ctx);
        let per_game_mode = match mode {
            ChanceMode::Deterministic { .. } => ChanceMode::Deterministic {
                episode_seed: 123 ^ i,
            },
            ChanceMode::Rng { .. } => ChanceMode::Rng { seed: 123 ^ i },
        };
        tasks.push(GameTask::new(i, state, per_game_mode, mcts_cfg));
    }

    let start_stats = backend.stats_snapshot();
    let t0 = Instant::now();
    let deadline = opts.seconds.map(|s| t0 + Duration::from_secs(s));

    let mut steps: u64 = 0;
    let mut would_block: u64 = 0;
    let mut terminal: u64 = 0;
    let mut games_completed: u64 = 0;
    let mut executed_moves: u64 = 0;

    let mut mcts_fallbacks: u64 = 0;
    let mut mcts_pending_collisions: u64 = 0;
    let mut mcts_pending_count_max: u64 = 0;

    let mut next_game_id: u64 = opts.parallel;
    while opts.games.map(|g| games_completed < g).unwrap_or(true)
        && deadline.map(|d| Instant::now() < d).unwrap_or(true)
    {
        for t in &mut tasks {
            let r = t.step(&backend, 32);
            match r {
                Ok(sr) => {
                    match sr.status {
                        yz_runtime::StepStatus::Progress => steps += 1,
                        yz_runtime::StepStatus::WouldBlock => would_block += 1,
                        yz_runtime::StepStatus::Terminal => terminal += 1,
                    }
                    if let Some(exec) = sr.executed {
                        executed_moves += 1;
                        mcts_fallbacks += exec.search.fallbacks as u64;
                        mcts_pending_collisions += exec.search.pending_collisions as u64;
                        mcts_pending_count_max =
                            mcts_pending_count_max.max(exec.search.pending_count_max as u64);
                    }
                    if sr.completed_episode.is_some() {
                        games_completed += 1;
                        // Reset the task fully (avoid carrying over traj/search/mcts state).
                        let gid = next_game_id;
                        next_game_id += 1;
                        let mut ctx = yz_core::TurnContext::new_rng(0xBADC0DE ^ gid);
                        let s = yz_core::initial_state(&mut ctx);
                        let new_mode = match mode {
                            ChanceMode::Deterministic { .. } => ChanceMode::Deterministic {
                                episode_seed: 123 ^ gid,
                            },
                            ChanceMode::Rng { .. } => ChanceMode::Rng { seed: 123 ^ gid },
                        };
                        *t = GameTask::new(gid, s, new_mode, mcts_cfg);
                    }
                }
                Err(_) => terminal += 1,
            }
        }
        // Small sleep reduces busy-wait when infer threads are catching up.
        thread::sleep(Duration::from_millis(1));
    }

    let wall = t0.elapsed();
    let end_stats = backend.stats_snapshot();

    let sent = end_stats.sent.saturating_sub(start_stats.sent);
    let received = end_stats.received.saturating_sub(start_stats.received);
    let errors = end_stats.errors.saturating_sub(start_stats.errors);

    let wall_s = wall.as_secs_f64().max(1e-9);
    let sims = (executed_moves as f64) * (opts.simulations as f64);
    let sims_per_sec = sims / wall_s;
    let evals_per_sec = (received as f64) / wall_s;

    let out = E2eBenchJsonV1 {
        event: "bench_e2e_v1".to_string(),
        wall_ms: wall.as_millis() as u64,
        games_completed,
        executed_moves,
        simulations_per_move: opts.simulations,
        sims_per_sec,
        evals_per_sec,
        infer_sent: sent,
        infer_received: received,
        infer_errors: errors,
        infer_inflight: end_stats.inflight as u64,
        infer_latency_p50_us: end_stats.latency_us.summary.p50_us,
        infer_latency_p95_us: end_stats.latency_us.summary.p95_us,
        infer_latency_mean_us: end_stats.latency_us.summary.mean_us,
        steps,
        would_block,
        terminal,
        mcts_fallbacks,
        mcts_pending_collisions,
        mcts_pending_count_max,
    };

    println!(
        "E2E bench: wall={:.2}s games={} moves={} sims/sec={:.0} evals/sec={:.0} inflight={} p95_us={}",
        wall_s,
        out.games_completed,
        out.executed_moves,
        out.sims_per_sec,
        out.evals_per_sec,
        out.infer_inflight,
        out.infer_latency_p95_us
    );
    println!("{}", serde_json::to_string(&out).expect("json"));

    drop(backend);
    let _ = server.join();
}
