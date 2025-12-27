use std::net::TcpListener;
use std::thread;
use std::time::Duration;

use yz_infer::codec::{decode_request_v1, encode_response_v1};
use yz_infer::frame::{read_frame, write_frame};
use yz_infer::protocol::{InferResponseV1, ACTION_SPACE_A};
use yz_mcts::{ChanceMode, InferBackend, MctsConfig};

use crate::{GameTask, Scheduler};

fn start_dummy_infer_server_tcp(delay_ms: u64) -> (std::net::SocketAddr, thread::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut sock, _peer) = listener.accept().unwrap();
        loop {
            let payload = match read_frame(&mut sock) {
                Ok(p) => p,
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
    });
    (addr, handle)
}

#[test]
fn scheduler_multiplexes_many_games_without_deadlock() {
    let (addr, server) = start_dummy_infer_server_tcp(1);
    let backend = InferBackend::connect_tcp(
        addr,
        0,
        yz_infer::ClientOptions {
            max_inflight_total: 4096,
            max_outbound_queue: 4096,
            request_id_start: 1,
        },
    )
    .unwrap();

    let mut tasks = Vec::new();
    for i in 0..8u64 {
        let mut ctx = yz_core::TurnContext::new_rng(123 ^ i);
        let state = yz_core::initial_state(&mut ctx);
        let mcts_cfg = MctsConfig {
            simulations: 32,
            max_inflight: 4,
            ..MctsConfig::default()
        };
        tasks.push(GameTask::new(
            i,
            state,
            ChanceMode::Rng { seed: 1234 ^ i },
            mcts_cfg,
        ));
    }

    let mut sched = Scheduler::new(tasks, 16);
    for _ in 0..500 {
        sched.tick(&backend);
        // Give the async IO threads + dummy server time to respond.
        thread::sleep(Duration::from_millis(1));
        if sched.tasks().iter().all(|t| t.ply > 0) {
            break;
        }
    }

    assert!(sched.tasks().iter().all(|t| t.ply > 0));
    assert!(sched.stats().steps > 0);

    drop(backend);
    server.join().unwrap();
}

#[test]
fn scheduler_writes_ndjson_iteration_and_root_logs() {
    let (addr, server) = start_dummy_infer_server_tcp(0);
    let backend = InferBackend::connect_tcp(
        addr,
        0,
        yz_infer::ClientOptions {
            max_inflight_total: 4096,
            max_outbound_queue: 4096,
            request_id_start: 1,
        },
    )
    .unwrap();

    let dir = tempfile::tempdir().unwrap();
    let out_dir = dir.path().join("run");
    let replay_dir = out_dir.join("replay");
    let logs_dir = out_dir.join("logs");
    std::fs::create_dir_all(&logs_dir).unwrap();

    let mut writer = yz_replay::ShardWriter::new(yz_replay::ShardWriterConfig {
        out_dir: replay_dir,
        max_samples_per_shard: 1024,
        git_hash: None,
        config_hash: None,
    })
    .unwrap();

    let mut tasks = Vec::new();
    for i in 0..2u64 {
        let mut ctx = yz_core::TurnContext::new_rng(123 ^ i);
        let state = yz_core::initial_state(&mut ctx);
        let mcts_cfg = MctsConfig {
            simulations: 16,
            max_inflight: 2,
            ..MctsConfig::default()
        };
        tasks.push(GameTask::new(
            i,
            state,
            ChanceMode::Rng { seed: 1234 ^ i },
            mcts_cfg,
        ));
    }
    let mut sched = Scheduler::new(tasks, 8);

    let mut loggers = crate::RunLoggers {
        run_id: "test".to_string(),
        v: yz_logging::VersionInfoV1 {
            protocol_version: yz_infer::protocol::PROTOCOL_VERSION,
            feature_schema_id: yz_features::schema::FEATURE_SCHEMA_ID,
            action_space_id: "oracle_keepmask_v1",
            ruleset_id: "swedish_scandinavian_v1",
        },
        git_hash: None,
        config_snapshot: Some("config.yaml".to_string()),
        root_log_every_n: 1,
        iter: yz_logging::NdjsonWriter::open_append(logs_dir.join("iteration_stats.ndjson"))
            .unwrap(),
        roots: yz_logging::NdjsonWriter::open_append(logs_dir.join("mcts_roots.ndjson")).unwrap(),
        metrics: yz_logging::NdjsonWriter::open_append(logs_dir.join("metrics.ndjson")).unwrap(),
    };

    for _ in 0..200 {
        sched.tick_and_write(&backend, &mut writer, Some(&mut loggers))
            .unwrap();
        thread::sleep(Duration::from_millis(1));
        if sched.tasks().iter().any(|t| t.ply > 0) {
            break;
        }
    }

    writer.finish().unwrap();
    loggers.iter.flush().unwrap();
    loggers.roots.flush().unwrap();
    loggers.metrics.flush().unwrap();

    let iter_path = logs_dir.join("iteration_stats.ndjson");
    let roots_path = logs_dir.join("mcts_roots.ndjson");
    let metrics_path = logs_dir.join("metrics.ndjson");
    assert!(iter_path.exists());
    assert!(roots_path.exists());
    assert!(metrics_path.exists());

    // Validate that all complete lines are valid JSON objects.
    for p in [iter_path, roots_path, metrics_path] {
        let s = std::fs::read_to_string(&p).unwrap();
        let mut ok = 0usize;
        for line in s.lines() {
            if line.trim().is_empty() {
                continue;
            }
            serde_json::from_str::<serde_json::Value>(line).unwrap();
            ok += 1;
        }
        assert!(ok > 0);
    }

    drop(backend);
    server.join().unwrap();
}
