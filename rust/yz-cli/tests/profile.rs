use std::process::Command;

fn yz_bin() -> String {
    env!("CARGO_BIN_EXE_yz").to_string()
}

#[test]
fn profile_help_runs() {
    let out = Command::new(yz_bin())
        .args(["profile", "--help"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let s = String::from_utf8_lossy(&out.stdout);
    assert!(s.contains("yz profile"));
}

#[test]
fn profile_selfplay_help_fallback_runs() {
    // Should print underlying help if flamegraph isn't installed (fallback).
    // Even if flamegraph is installed, this should still succeed.
    let out = Command::new(yz_bin())
        .args(["profile", "selfplay", "--", "--help"])
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}


