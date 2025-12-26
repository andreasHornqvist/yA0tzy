use std::fs;
use std::process::Command;

fn yz_bin() -> String {
    // Provided by Cargo for integration tests of binaries.
    env!("CARGO_BIN_EXE_yz").to_string()
}

#[test]
fn iter_finalize_promote_creates_best_and_updates_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let run_dir = dir.path().join("run");
    let models_dir = run_dir.join("models");
    fs::create_dir_all(&models_dir).unwrap();

    fs::write(models_dir.join("candidate.pt"), b"candidate-bytes").unwrap();
    fs::write(models_dir.join("candidate.meta.json"), b"{\"x\":1}\n").unwrap();

    let run_json = run_dir.join("run.json");
    let m = yz_logging::RunManifestV1 {
        run_manifest_version: yz_logging::RUN_MANIFEST_VERSION,
        run_id: "run".to_string(),
        created_ts_ms: yz_logging::now_ms(),
        protocol_version: 1,
        feature_schema_id: 1,
        action_space_id: "oracle_keepmask_v1".to_string(),
        ruleset_id: "swedish_scandinavian_v1".to_string(),
        git_hash: None,
        config_hash: None,
        replay_dir: "replay".to_string(),
        logs_dir: "logs".to_string(),
        models_dir: "models".to_string(),
        selfplay_games_completed: 0,
        train_step: 0,
        best_checkpoint: None,
        candidate_checkpoint: Some("models/candidate.pt".to_string()),
        promotion_decision: None,
        promotion_ts_ms: None,
        gate_games: None,
        gate_win_rate: None,
    };
    yz_logging::write_manifest_atomic(&run_json, &m).unwrap();

    let out = Command::new(yz_bin())
        .args([
            "iter",
            "finalize",
            "--run",
            run_dir.to_str().unwrap(),
            "--decision",
            "promote",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    let best_pt = models_dir.join("best.pt");
    assert!(best_pt.exists());
    assert_eq!(fs::read(best_pt).unwrap(), b"candidate-bytes");

    let best_meta = models_dir.join("best.meta.json");
    assert!(best_meta.exists());

    let got = yz_logging::read_manifest(&run_json).unwrap();
    assert_eq!(got.best_checkpoint.as_deref(), Some("models/best.pt"));
    assert_eq!(got.promotion_decision.as_deref(), Some("promote"));
    assert!(got.promotion_ts_ms.unwrap_or(0) > 0);
}

#[test]
fn iter_finalize_reject_does_not_create_best_but_updates_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let run_dir = dir.path().join("run");
    let models_dir = run_dir.join("models");
    fs::create_dir_all(&models_dir).unwrap();

    fs::write(models_dir.join("candidate.pt"), b"candidate-bytes").unwrap();

    let run_json = run_dir.join("run.json");
    let m = yz_logging::RunManifestV1 {
        run_manifest_version: yz_logging::RUN_MANIFEST_VERSION,
        run_id: "run".to_string(),
        created_ts_ms: yz_logging::now_ms(),
        protocol_version: 1,
        feature_schema_id: 1,
        action_space_id: "oracle_keepmask_v1".to_string(),
        ruleset_id: "swedish_scandinavian_v1".to_string(),
        git_hash: None,
        config_hash: None,
        replay_dir: "replay".to_string(),
        logs_dir: "logs".to_string(),
        models_dir: "models".to_string(),
        selfplay_games_completed: 0,
        train_step: 0,
        best_checkpoint: None,
        candidate_checkpoint: Some("models/candidate.pt".to_string()),
        promotion_decision: None,
        promotion_ts_ms: None,
        gate_games: None,
        gate_win_rate: None,
    };
    yz_logging::write_manifest_atomic(&run_json, &m).unwrap();

    let out = Command::new(yz_bin())
        .args([
            "iter",
            "finalize",
            "--run",
            run_dir.to_str().unwrap(),
            "--decision",
            "reject",
        ])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );

    assert!(!models_dir.join("best.pt").exists());
    let got = yz_logging::read_manifest(&run_json).unwrap();
    assert_eq!(got.promotion_decision.as_deref(), Some("reject"));
    assert!(got.promotion_ts_ms.unwrap_or(0) > 0);
}


