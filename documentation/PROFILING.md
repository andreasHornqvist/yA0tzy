# Profiling guide (E11S3)

This project is throughput-sensitive. When the pipeline “feels slow”, you typically want to answer:

- **Where is CPU time going?** (PUCT selection, state transitions, protocol encode/decode, IO)
- **Are we inference-bound?** (high `would_block`, high inflight, high latency)

## Prerequisites (Rust flamegraphs)

Install the flamegraph tool:

- `cargo install flamegraph`

Notes:
- On Linux you typically need `perf` available and permitted.
- On macOS, support depends on your environment; if flamegraph fails, you can still run the workload without profiling and use alternative profilers.\n+
## One-command profiling wrappers

The `yz` CLI provides wrappers that try to run `cargo flamegraph`. If flamegraph is not installed, `yz` will print a warning and run the command normally.

### Profile self-play

- `cargo run --bin yz -- profile selfplay -- --help`
- `cargo run --bin yz -- profile selfplay -- --config configs/local_cpu.yaml --infer unix:///tmp/yatzy_infer.sock --out runs/profile/ --games 10`

### Profile gating

- `cargo run --bin yz -- profile gate -- --config configs/local_cpu.yaml --infer tcp://127.0.0.1:1234 --run runs/profile_run/`

### Profile E2E benchmark harness

- `cargo run --bin yz -- profile bench-e2e -- --seconds 10 --parallel 8 --simulations 64 --max-inflight 4 --chance deterministic`

## What to look for in flamegraphs

- **MCTS selection/backup**: hotspots in `yz_mcts` (selection loop, backup, masked softmax)\n+- **Rules engine**: hotspots in `yz_core::apply_action` / chance rerolls\n+- **Protocol/IO**: hotspots in `yz_infer` frame/codec or socket read/write\n+- **Scheduler**: time spent in runtime stepping vs waiting (`would_block` events)\n+
When you’re unsure whether you’re compute-bound or inference-bound, run `yz bench e2e` first to see throughput and `would_block`/latency counters, then profile the slowest mode.\n+

