use std::process::Command;

fn yz_bin() -> String {
    env!("CARGO_BIN_EXE_yz").to_string()
}

#[test]
fn bench_e2e_help_runs() {
    // Smoke: ensure the wrapper path is wired and the harness can print help.
    let out = Command::new(yz_bin())
        .args(["bench", "e2e", "--help"])
        .output()
        .unwrap();

    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("yz-bench-e2e"));
}


