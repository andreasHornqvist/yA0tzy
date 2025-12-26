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
