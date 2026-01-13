use std::fs;
use std::process::Command;

use serde_json::Value;

fn yz_bin() -> String {
    env!("CARGO_BIN_EXE_yz").to_string()
}

fn start_dummy_infer_server_tcp() -> (std::net::SocketAddr, std::thread::JoinHandle<()>) {
    use std::net::TcpListener;
    use std::thread;

    use yz_infer::codec::{decode_request_v1, encode_response_v1, encode_response_v2};
    use yz_infer::frame::{read_frame, write_frame};
    use yz_infer::protocol::{InferResponseV1, ACTION_SPACE_A};

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = thread::spawn(move || {
        for conn in listener.incoming() {
            let Ok(mut sock) = conn else { break };
            thread::spawn(move || loop {
                let payload = match read_frame(&mut sock) {
                    Ok(p) => p,
                    Err(_) => break,
                };
                // Test helper: accept both v1 and v2 requests.
                // Gate code should use v2 by default now, but other tests may still send v1.
                let version = if payload.len() >= 4 {
                    u32::from_le_bytes([payload[0], payload[1], payload[2], payload[3]])
                } else {
                    break;
                };
                let is_v2 = version == yz_infer::protocol::PROTOCOL_VERSION_V2;

                // For this test's purposes, decode v1 fully (includes legal mask).
                // For v2, decode a minimal subset: request_id + legal_mask.
                let (request_id, legal_mask) = if !is_v2 {
                    let req = match decode_request_v1(&payload) {
                        Ok(r) => r,
                        Err(_) => break,
                    };
                    (req.request_id, req.legal_mask)
                } else {
                    // v2 frame layout:
                    // u32 version, u8 kind, u8 flags, u16 reserved,
                    // u64 request_id, u32 model_id, u32 feature_schema_id,
                    // u32 features_byte_len, [features bytes],
                    // u32 legal_len, [legal bytes]
                    if payload.len() < 4 + 1 + 1 + 2 + 8 + 4 + 4 + 4 {
                        break;
                    }
                    // request_id starts at offset 8
                    let rid_off = 8;
                    let request_id = u64::from_le_bytes(
                        payload[rid_off..rid_off + 8].try_into().unwrap(),
                    );
                    let mut off = 4 + 1 + 1 + 2 + 8 + 4 + 4;
                    let features_byte_len =
                        u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
                    off += 4;
                    if payload.len() < off + features_byte_len + 4 {
                        break;
                    }
                    off += features_byte_len;
                    let legal_len = u32::from_le_bytes(payload[off..off + 4].try_into().unwrap()) as usize;
                    off += 4;
                    if payload.len() < off + legal_len {
                        break;
                    }
                    let legal = &payload[off..off + legal_len];
                    let mut legal_mask: yz_infer::protocol::LegalMask = 0;
                    if legal_len == yz_infer::protocol::LEGAL_MASK_BITSET_BYTES {
                        // LSB-first, 6 bytes
                        let mut b = [0u8; 8];
                        b[0..6].copy_from_slice(legal);
                        legal_mask = u64::from_le_bytes(b);
                    } else if legal_len == ACTION_SPACE_A as usize {
                        for (i, &b) in legal.iter().enumerate() {
                            if b == 1 {
                                legal_mask |= 1u64 << i;
                            }
                        }
                    } else {
                        break;
                    }
                    (request_id, legal_mask)
                };

                let mut logits = vec![0.0f32; ACTION_SPACE_A as usize];
                for i in 0..(ACTION_SPACE_A as usize) {
                    if ((legal_mask >> i) & 1) == 0 {
                        logits[i] = -1.0e9;
                    }
                }

                let resp = InferResponseV1 {
                    request_id,
                    policy_logits: logits,
                    value: 0.0,
                    margin: None,
                };
                let out = if is_v2 {
                    encode_response_v2(&resp)
                } else {
                    encode_response_v1(&resp)
                };
                if write_frame(&mut sock, &out).is_err() {
                    break;
                }
            });
        }
    });
    (addr, handle)
}

#[test]
fn gate_runs_and_updates_manifest() {
    let (addr, _server) = start_dummy_infer_server_tcp();

    let dir = tempfile::tempdir().unwrap();
    let run_dir = dir.path().join("run");
    fs::create_dir_all(&run_dir).unwrap();

    // Minimal run.json for updates.
    let run_json = run_dir.join("run.json");
    let m = yz_logging::RunManifestV1 {
        run_manifest_version: yz_logging::RUN_MANIFEST_VERSION,
        run_id: "run".to_string(),
        created_ts_ms: yz_logging::now_ms(),
        protocol_version: 2,
        feature_schema_id: 1,
        action_space_id: "oracle_keepmask_v1".to_string(),
        ruleset_id: "swedish_scandinavian_v1".to_string(),
        git_hash: None,
        config_hash: None,
        config_snapshot: None,
        config_snapshot_hash: None,
        replay_dir: "replay".to_string(),
        logs_dir: "logs".to_string(),
        models_dir: "models".to_string(),
        selfplay_games_completed: 0,
        train_step: 0,
        best_checkpoint: None,
        candidate_checkpoint: None,
        best_promoted_iter: None,
        train_last_loss_total: None,
        train_last_loss_policy: None,
        train_last_loss_value: None,
        promotion_decision: None,
        promotion_ts_ms: None,
        gate_games: None,
        gate_win_rate: None,
        gate_draw_rate: None,
        gate_wins: None,
        gate_losses: None,
        gate_draws: None,
        gate_ci95_low: None,
        gate_ci95_high: None,
        gate_sprt: None,
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
    yz_logging::write_manifest_atomic(&run_json, &m).unwrap();

    // Config file for gating.
    let cfg_path = dir.path().join("cfg.yaml");
    fs::write(
        &cfg_path,
        format!(
            r#"
inference:
  bind: "unix:///tmp/ignored.sock"
  device: "cpu"
  max_batch: 32
  max_wait_us: 1000
mcts:
  c_puct: 1.5
  budget_reroll: 8
  budget_mark: 8
  max_inflight_per_game: 4
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25
selfplay:
  games_per_iteration: 1
  workers: 1
  threads_per_worker: 1
training:
  batch_size: 32
  learning_rate: 0.001
  epochs: 1
gating:
  games: 2
  seed: 0
  seed_set_id: "dev_small_v1"
  win_rate_threshold: 0.5
  paired_seed_swap: true
  deterministic_chance: true
"#
        ),
    )
    .unwrap();

    let out = Command::new(yz_bin())
        .args([
            "gate",
            "--config",
            cfg_path.to_str().unwrap(),
            "--infer",
            &format!("tcp://{addr}"),
            "--run",
            run_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let got = yz_logging::read_manifest(&run_json).unwrap();
    assert_eq!(got.gate_games, Some(2));
    assert!((got.gate_win_rate.unwrap_or(0.0) - 0.5).abs() < 1e-9);
    assert!(got.gate_seeds_hash.as_deref().unwrap_or("").len() >= 32);
    assert!(
        (0.0..=1.0).contains(&got.gate_oracle_match_rate_overall.unwrap_or(0.0)),
        "gate_oracle_match_rate_overall missing or out of range"
    );
    assert_eq!(got.config_snapshot.as_deref(), Some("config.yaml"));
    assert!(got.config_snapshot_hash.as_deref().unwrap_or("").len() >= 32);
    assert!(run_dir.join("config.yaml").exists());

    let gate_report = run_dir.join("gate_report.json");
    assert!(gate_report.exists());

    let bytes = fs::read(&gate_report).unwrap();
    let v: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(v.get("oracle_match_rate_overall").is_some());
    assert!(v.get("oracle_match_rate_mark").is_some());
    assert!(v.get("oracle_match_rate_reroll").is_some());
    assert!(v.get("oracle_keepall_ignored").is_some());

    // E10.5S2: metrics.ndjson includes gate_summary.
    let metrics = run_dir.join("logs").join("metrics.ndjson");
    assert!(metrics.exists());
    let s = fs::read_to_string(metrics).unwrap();
    assert!(
        s.lines().any(|l| l.contains(r#""event":"gate_summary""#)),
        "expected gate_summary event in metrics.ndjson"
    );
}
