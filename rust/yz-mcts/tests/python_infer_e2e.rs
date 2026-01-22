use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use yz_infer::ClientOptions;
use yz_mcts::{ChanceMode, InferBackend, Mcts, MctsConfig};

fn repo_root() -> PathBuf {
    // rust/yz-mcts -> rust -> repo root
    let here = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    here.parent()
        .and_then(|p| p.parent())
        .map(|p| p.to_path_buf())
        .expect("failed to compute repo root")
}

fn pick_python_exe(root: &Path) -> OsString {
    if let Ok(s) = std::env::var("YATZY_AZ_PYTHON") {
        return OsString::from(s);
    }
    let venv = root.join("python").join(".venv").join("bin").join("python");
    if venv.exists() {
        return venv.into_os_string();
    }
    OsString::from("python")
}

fn wait_for_path(p: &Path, timeout: Duration) -> Result<(), String> {
    let t0 = Instant::now();
    while t0.elapsed() < timeout {
        if p.exists() {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(25));
    }
    Err(format!("timed out waiting for {}", p.display()))
}

struct ChildGuard {
    child: Child,
}

impl Drop for ChildGuard {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

#[test]
fn python_infer_server_real_checkpoint_e2e_opt_in() {
    if std::env::var("YZ_PY_E2E").ok().as_deref() != Some("1") {
        eprintln!("skipping (set YZ_PY_E2E=1 to enable)");
        return;
    }

    let root = repo_root();
    let python = pick_python_exe(&root);

    let td = tempfile::tempdir().expect("tempdir");
    let sock = td.path().join("yatzy_infer.sock");
    let best_pt = td.path().join("best.pt");
    let stderr_path = td.path().join("infer.stderr.txt");
    let stdout_path = td.path().join("infer.stdout.txt");

    // Ensure old socket is gone.
    let _ = fs::remove_file(&sock);

    // 1) Create a tiny checkpoint via model-init.
    let st = Command::new(&python)
        .current_dir(root.join("python"))
        .args([
            "-m",
            "yatzy_az",
            "model-init",
            "--out",
            best_pt.to_string_lossy().as_ref(),
            "--hidden",
            "8",
            "--blocks",
            "1",
        ])
        .status()
        .expect("spawn model-init");
    assert!(st.success(), "model-init failed: {st}");
    assert!(best_pt.exists(), "best.pt not created");

    // 2) Spawn Python infer-server on UDS.
    let stdout_f = fs::File::create(&stdout_path).expect("create stdout file");
    let stderr_f = fs::File::create(&stderr_path).expect("create stderr file");
    let child = Command::new(&python)
        .current_dir(root.join("python"))
        .args([
            "-m",
            "yatzy_az",
            "infer-server",
            "--bind",
            &format!("unix://{}", sock.to_string_lossy()),
            "--device",
            "cpu",
            "--best",
            &format!("path:{}", best_pt.to_string_lossy()),
            "--cand",
            "dummy",
            "--metrics-disable",
            "--print-stats-every-s",
            "0",
        ])
        .stdout(Stdio::from(stdout_f))
        .stderr(Stdio::from(stderr_f))
        .spawn()
        .expect("spawn infer-server");
    let _guard = ChildGuard { child };

    wait_for_path(&sock, Duration::from_secs(5)).unwrap_or_else(|e| {
        let stderr = fs::read_to_string(&stderr_path).unwrap_or_default();
        panic!("{e}\n---- infer-server stderr ----\n{stderr}");
    });

    // 3) Connect from Rust and run a short search.
    let backend = InferBackend::connect_uds(
        &sock,
        0,
        ClientOptions {
            max_inflight_total: 4096,
            max_outbound_queue: 4096,
            request_id_start: 1,
            protocol_version: yz_infer::protocol::PROTOCOL_VERSION_V1,
            legal_mask_bitset: false,
        },
    )
    .unwrap();

    let mut mcts = Mcts::new(MctsConfig {
        c_puct: 1.5,
        simulations_mark: 32,
        simulations_reroll: 32,
        dirichlet_alpha: 0.3,
        dirichlet_epsilon: 0.0,
        max_inflight: 8,
        virtual_loss_mode: yz_mcts::VirtualLossMode::QPenalty,
        virtual_loss: 1.0,
        expansion_lock: false,
        explicit_keepmask_chance: false,
        chance_pw_enabled: false,
        chance_pw_c: 2.0,
        chance_pw_alpha: 0.6,
        chance_pw_max_children: 64,
    })
    .unwrap();

    let mut ctx = yz_core::TurnContext::new_deterministic(123);
    let root_state = yz_core::initial_state(&mut ctx);
    let res = mcts.run_search_with_backend(
        root_state,
        ChanceMode::Deterministic { episode_seed: 123 },
        &backend,
    );

    // Validate pi over legal actions.
    let legal = yz_core::legal_action_mask(
        root_state.players[root_state.player_to_move as usize].avail_mask,
        root_state.rerolls_left,
    );
    let mut sum = 0.0f32;
    for a in 0..yz_core::A {
        if ((legal >> a) & 1) != 0 {
            assert!(res.pi[a].is_finite());
            assert!(res.pi[a] >= 0.0);
            sum += res.pi[a];
        } else {
            assert_eq!(res.pi[a], 0.0);
        }
    }
    assert!((sum - 1.0).abs() < 1e-5, "sum={sum}");
}
