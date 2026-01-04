use std::net::{SocketAddr, TcpListener};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use crate::client::{ClientError, ClientOptions, InferenceClient};
use crate::codec::{decode_request_v1, encode_response_v1};
use crate::frame::{read_frame, write_frame};
use crate::protocol::{
    InferRequestV1, InferResponseV1, ACTION_SPACE_A, FEATURE_LEN_V1, FEATURE_SCHEMA_ID_V1,
};

fn mk_req(model_id: u32) -> InferRequestV1 {
    InferRequestV1 {
        request_id: 0,
        model_id,
        feature_schema_id: FEATURE_SCHEMA_ID_V1,
        features: vec![0.0; FEATURE_LEN_V1 as usize],
        legal_mask: vec![1u8; ACTION_SPACE_A as usize],
    }
}

fn start_tcp_reverse_server(n: usize) -> (SocketAddr, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        let (mut sock, _peer) = listener.accept().unwrap();
        let mut ids = Vec::with_capacity(n);
        for _ in 0..n {
            let payload = read_frame(&mut sock).unwrap();
            let req = decode_request_v1(&payload).unwrap();
            ids.push(req.request_id);
        }
        for request_id in ids.into_iter().rev() {
            let resp = InferResponseV1 {
                request_id,
                policy_logits: vec![0.0; ACTION_SPACE_A as usize],
                value: 0.0,
                margin: None,
            };
            let payload = encode_response_v1(&resp);
            write_frame(&mut sock, &payload).unwrap();
        }
    });
    (addr, handle)
}

#[test]
fn client_routes_out_of_order_tcp() {
    let n = 128usize;
    let (addr, server) = start_tcp_reverse_server(n);

    let client = InferenceClient::connect_tcp(
        addr,
        ClientOptions {
            max_inflight_total: 4096,
            max_outbound_queue: 4096,
            request_id_start: 1,
            protocol_version: crate::protocol::PROTOCOL_VERSION_V1,
        },
    )
    .unwrap();

    let mut tickets = Vec::with_capacity(n);
    for _ in 0..n {
        tickets.push(client.submit(mk_req(7)).unwrap());
    }

    for t in tickets {
        let resp = t.recv_timeout(Duration::from_secs(2)).unwrap();
        assert_eq!(resp.policy_logits.len(), ACTION_SPACE_A as usize);
        assert_eq!(resp.value, 0.0);
        assert!(resp.margin.is_none());
    }

    let stats = client.stats_snapshot();
    assert_eq!(stats.sent, n as u64);
    assert_eq!(stats.received, n as u64);
    assert_eq!(stats.latency_us.summary.count, n as u64);

    server.join().unwrap();
}

#[cfg(unix)]
mod uds {
    use super::*;
    use std::fs;
    use std::os::unix::net::UnixListener;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn tmp_sock_path(tag: &str) -> PathBuf {
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "yatzy_infer_{}_{}_{}.sock",
            tag,
            std::process::id(),
            ts
        ))
    }

    fn start_uds_reverse_server(path: PathBuf, n: usize) -> JoinHandle<()> {
        // Ensure no stale socket.
        let _ = fs::remove_file(&path);
        let listener = UnixListener::bind(&path).unwrap();
        thread::spawn(move || {
            let (mut sock, _addr) = listener.accept().unwrap();
            let mut ids = Vec::with_capacity(n);
            for _ in 0..n {
                let payload = read_frame(&mut sock).unwrap();
                let req = decode_request_v1(&payload).unwrap();
                ids.push(req.request_id);
            }
            for request_id in ids.into_iter().rev() {
                let resp = InferResponseV1 {
                    request_id,
                    policy_logits: vec![0.0; ACTION_SPACE_A as usize],
                    value: 0.0,
                    margin: None,
                };
                let payload = encode_response_v1(&resp);
                write_frame(&mut sock, &payload).unwrap();
            }
            // Cleanup socket file best-effort.
            let _ = fs::remove_file(&path);
        })
    }

    #[test]
    fn client_routes_out_of_order_uds() {
        let n = 128usize;
        let path = tmp_sock_path("reverse");
        let server = start_uds_reverse_server(path.clone(), n);

        let client = InferenceClient::connect_uds(
            &path,
            ClientOptions {
                max_inflight_total: 4096,
                max_outbound_queue: 4096,
                request_id_start: 1,
                protocol_version: crate::protocol::PROTOCOL_VERSION_V1,
            },
        )
        .unwrap();

        let mut tickets = Vec::with_capacity(n);
        for _ in 0..n {
            tickets.push(client.submit(mk_req(7)).unwrap());
        }

        for t in tickets {
            let _ = t.recv_timeout(Duration::from_secs(2)).unwrap();
        }

        let stats = client.stats_snapshot();
        assert_eq!(stats.sent, n as u64);
        assert_eq!(stats.received, n as u64);
        assert_eq!(stats.latency_us.summary.count, n as u64);

        server.join().unwrap();
    }
}

#[test]
fn client_backpressure_inflight_cap() {
    let n = 8usize;
    let (addr, server) = {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let handle = thread::spawn(move || {
            let (mut sock, _peer) = listener.accept().unwrap();
            let mut ids = Vec::with_capacity(n);
            for _ in 0..n {
                let payload = read_frame(&mut sock).unwrap();
                let req = decode_request_v1(&payload).unwrap();
                ids.push(req.request_id);
            }
            // Keep requests in-flight for a moment.
            thread::sleep(Duration::from_millis(200));
            for request_id in ids {
                let resp = InferResponseV1 {
                    request_id,
                    policy_logits: vec![0.0; ACTION_SPACE_A as usize],
                    value: 0.0,
                    margin: None,
                };
                let payload = encode_response_v1(&resp);
                write_frame(&mut sock, &payload).unwrap();
            }
        });
        (addr, handle)
    };

    let client = InferenceClient::connect_tcp(
        addr,
        ClientOptions {
            max_inflight_total: n,
            max_outbound_queue: 1024,
            request_id_start: 1,
            protocol_version: crate::protocol::PROTOCOL_VERSION_V1,
        },
    )
    .unwrap();

    let mut tickets = Vec::with_capacity(n);
    for _ in 0..n {
        tickets.push(client.submit(mk_req(7)).unwrap());
    }
    let err = client.submit(mk_req(7)).unwrap_err();
    match err {
        ClientError::Backpressure(_) => {}
        other => panic!("expected backpressure, got {other:?}"),
    }

    for t in tickets {
        let _ = t.recv_timeout(Duration::from_secs(2)).unwrap();
    }

    server.join().unwrap();
}

#[test]
fn client_handles_thousands_inflight_tcp() {
    let n = 2000usize;
    let (addr, server) = start_tcp_reverse_server(n);

    let client = InferenceClient::connect_tcp(
        addr,
        ClientOptions {
            max_inflight_total: n + 16,
            max_outbound_queue: n + 16,
            request_id_start: 1,
            protocol_version: crate::protocol::PROTOCOL_VERSION_V1,
        },
    )
    .unwrap();

    // Submit from multiple threads (contention + correctness).
    let (tx, rx) = std::sync::mpsc::channel();
    let threads = 4usize;
    let per = n / threads;
    thread::scope(|s| {
        let client = &client;
        for _ in 0..threads {
            let tx = tx.clone();
            let client = client;
            s.spawn(move || {
                let mut local = Vec::with_capacity(per);
                for _ in 0..per {
                    local.push(client.submit(mk_req(7)).unwrap());
                }
                tx.send(local).unwrap();
            });
        }
    });
    drop(tx);

    let mut tickets = Vec::with_capacity(n);
    for local in rx {
        tickets.extend(local);
    }
    assert_eq!(tickets.len(), n);

    for t in tickets {
        let _ = t.recv_timeout(Duration::from_secs(5)).unwrap();
    }

    let stats = client.stats_snapshot();
    assert_eq!(stats.sent, n as u64);
    assert_eq!(stats.received, n as u64);
    assert_eq!(stats.latency_us.summary.count, n as u64);

    server.join().unwrap();
}
