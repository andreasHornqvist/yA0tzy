use std::ffi::OsString;
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;

use yz_controller::spawn_iteration;

fn repo_root() -> PathBuf {
    // rust/yz-controller -> rust -> repo root
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .expect("failed to compute repo root")
}

fn pick_python_exe_with_deps(root: &Path) -> Option<OsString> {
    // Prefer explicit override.
    if let Ok(s) = std::env::var("YATZY_AZ_PYTHON") {
        let exe = OsString::from(s);
        if python_can_import(&exe) {
            return Some(exe);
        }
    }

    // Prefer repo venv.
    let venv = root.join("python").join(".venv").join("bin").join("python");
    if venv.exists() {
        let exe = venv.into_os_string();
        if python_can_import(&exe) {
            return Some(exe);
        }
    }

    // Fallback to python3 (avoid controller's special-casing for "python").
    let exe = OsString::from("python3");
    if python_can_import(&exe) {
        return Some(exe);
    }

    None
}

fn python_can_import(exe: &OsString) -> bool {
    // We need torch + safetensors for infer-server + trainer.
    Command::new(exe)
        .args(["-c", "import torch, safetensors"])
        .status()
        .map(|st| st.success())
        .unwrap_or(false)
}

fn free_local_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    l.local_addr().expect("local_addr").port()
}

#[test]
fn controller_two_iterations_e2e_opt_in() {
    if std::env::var("YZ_PY_E2E").ok().as_deref() != Some("1") {
        eprintln!("skipping (set YZ_PY_E2E=1 to enable)");
        return;
    }

    let root = repo_root();
    let python = match pick_python_exe_with_deps(&root) {
        Some(p) => p,
        None => {
            eprintln!("skipping (no python found that can import torch+safetensors)");
            return;
        }
    };

    // Use a unique run dir so sockets/logs don't collide with other tests or local runs.
    let td = tempfile::tempdir().expect("tempdir");
    let run_dir = td.path().join("run");

    // Pick unique endpoints to avoid collisions.
    let sock = run_dir.join("yatzy_infer.sock");
    let infer_endpoint = format!("unix://{}", sock.to_string_lossy());
    let metrics_port = free_local_port();
    let metrics_bind = format!("127.0.0.1:{metrics_port}");

    // Tiny config to keep E2E fast. We want to cover:
    // - selfplay writing replay
    // - trainer using random_indexed dataset
    // - gating completing
    let cfg_yaml = r#"
inference:
  bind: unix:///tmp/will_be_overridden
  device: cpu
  max_batch: 32
  max_wait_us: 1000
  torch_threads: 2
  torch_interop_threads: 1
  metrics_bind: 127.0.0.1:18080
mcts:
  c_puct: 1.5
  budget_reroll: 16
  budget_mark: 16
  max_inflight_per_game: 4
  dirichlet_alpha: 0.3
  dirichlet_epsilon: 0.25
  temperature_schedule:
    kind: constant
    t0: 1.0
selfplay:
  games_per_iteration: 10
  workers: 1
  threads_per_worker: 1
training:
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 0.0001
  epochs: 1
  steps_per_iteration: 5
  sample_mode: random_indexed
  dataloader_workers: 0
  cache_shards: 2
gating:
  games: 10
  seed: 0
  seed_set_id: dev_v1
  win_rate_threshold: 0.0
  paired_seed_swap: true
  deterministic_chance: true
  threads_per_worker: 1
replay:
  capacity_shards: 50
controller:
  total_iterations: 2
model:
  hidden_dim: 8
  num_blocks: 1
"#;

    let mut cfg: yz_core::Config = serde_yaml::from_str(cfg_yaml).expect("parse cfg");
    cfg.inference.bind = infer_endpoint.clone();
    cfg.inference.metrics_bind = metrics_bind;

    let handle = spawn_iteration(
        &run_dir,
        cfg,
        infer_endpoint,
        python.to_string_lossy().to_string(),
    );

    handle.join().expect("controller failed");

    let m = yz_logging::read_manifest(run_dir.join("run.json")).expect("read run.json");
    assert_eq!(m.controller_phase.as_deref(), Some("done"));
    assert_eq!(m.controller_iteration_idx, 2);
    assert!(
        m.iterations.len() >= 2,
        "expected >=2 iterations, got {}",
        m.iterations.len()
    );

    for (i, it) in m.iterations.iter().take(2).enumerate() {
        assert!(it.ended_ts_ms.is_some(), "iter {i} missing ended_ts_ms");

        assert_eq!(it.selfplay.games_target, 10);
        assert_eq!(it.selfplay.games_completed, 10);
        assert!(it.selfplay.ended_ts_ms.is_some(), "iter {i} selfplay not ended");

        assert!(it.train.ended_ts_ms.is_some(), "iter {i} train not ended");
        assert!(
            it.train.last_loss_total.is_some(),
            "iter {i} missing train last_loss_total"
        );

        assert_eq!(it.gate.games_target, 10);
        assert_eq!(it.gate.games_completed, 10);
        assert!(it.gate.ended_ts_ms.is_some(), "iter {i} gate not ended");
        assert!(it.gate.win_rate.is_some(), "iter {i} missing gate win_rate");
    }
}


