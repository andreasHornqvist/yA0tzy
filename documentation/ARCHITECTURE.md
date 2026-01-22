# yA0tzy Architecture Overview

> High-level guide for AI agents and contributors. Keep this document updated as the project evolves.

---

## Project Summary

**yA0tzy** is an AlphaZero-style training system for 2-player Swedish Yatzy. The architecture splits work between:

- **Rust**: High-performance game engine, MCTS, self-play, evaluation
- **Python**: Neural network training, batched inference server

---

## Repository Structure

```
yA0tzy/
├── rust/                     # Rust workspace (all performance-critical code)
│   ├── yz-core/              # Game rules, scoring, state representation
│   ├── yz-features/          # Feature encoding for neural network input
│   ├── yz-mcts/              # Monte Carlo Tree Search (PUCT algorithm)
│   ├── yz-runtime/           # Async schedulers, game task orchestration
│   ├── yz-infer/             # Socket protocol for Rust ↔ Python inference
│   ├── yz-replay/            # Replay buffer shards (safetensors format)
│   ├── yz-eval/              # Model gating (candidate vs best evaluation)
│   ├── yz-oracle/            # Adapter to swedish_yatzy_dp perfect-play oracle
│   ├── yz-logging/           # NDJSON event logging, metrics
│   ├── yz-controller/        # In-process iteration controller (phase/status in run.json)
│   ├── yz-tui/               # Ratatui terminal UI (run picker + config + dashboard)
│   ├── yz-bench/             # Criterion microbenches
│   ├── yz-bench-e2e/         # End-to-end perf benchmark harness
│   ├── yz-cli/               # `yz` binary (CLI entrypoint)
│   └── swedish_yatzy_dp/     # Vendored oracle (optimal-play DP solver)
│
├── python/                   # Python package
│   ├── pyproject.toml        # PEP 621 metadata (uv-compatible)
│   ├── .python-version       # Python version pin (3.12)
│   └── yatzy_az/             # Main package
│       ├── __init__.py
│       ├── __main__.py       # CLI entrypoint (python -m yatzy_az)
│       ├── model/            # Neural network architecture
│       ├── trainer/          # Training loop
│       └── server/           # Inference server (asyncio + batching + metrics)
│
├── configs/                  # YAML configuration files
│   └── local_cpu.yaml        # Example config for local CPU runs
│   └── seed_sets/            # Fixed dev seed sets for reproducible gating (E9.2)
│       └── dev_small_v1.txt
│
├── documentation/            # Project documentation
│   ├── prd.md                # Product Requirements Document
│   ├── ARCHITECTURE.md       # This file
│   └── PROFILING.md          # Profiling workflows and tips
│
├── runs/                     # Training run outputs (gitignored)
├── Cargo.toml                # Rust workspace definition
├── README.md                 # Quick start guide
└── .gitignore
```

---

## Tooling

| Tool | Purpose | Install |
|------|---------|---------|
| **Rust (stable)** | Compile Rust workspace | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| **uv** | Python package/env manager | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Python 3.12+** | Neural network training | via `uv` or system install |

### Common Commands

```bash
# Rust
cargo test --workspace          # Run all Rust tests
cargo build --release           # Release build
cargo run --bin yz -- --help    # Run CLI
cargo fmt --check               # Check formatting
cargo clippy --workspace -- -D warnings  # Lint

# Makefile shortcuts (recommended for local runs)
make run                        # Optimized default (release)
make run-dev                    # Debug build (slow, for debugging)

# Experiment CLI (small A/B runs)
#
# Start a run from a config file (creates runs/<name> or runs/<name>_<ts>):
cargo run -p yz-cli --bin yz -- start-run --run-name exp1 --config /tmp/cfg.yaml
#
# Run an existing run directory in the foreground and print per-iteration summaries:
cargo run -p yz-cli --bin yz -- controller --run-dir runs/exp1 --print-iter-table
#
# Cancel from another terminal:
# - open `yz tui`, select the run, and press `x` to request cancellation (writes runs/<id>/cancel.request)
# - `q` quits the TUI; if a run is currently active, the TUI will also request cancel before exiting
# - the controller polls for cancel.request and shuts down cleanly

# Opt-in Rust↔Python inference e2e test (E6.5S6)
YZ_PY_E2E=1 cargo test -p yz-mcts --test python_infer_e2e

# Python
cd python
uv sync --all-extras                 # Install all dependencies (including dev)
uv run python -m yatzy_az --help     # Run Python CLI
uv run ruff check .                  # Lint
uv run ruff format --check .         # Check formatting
uv run pytest                        # Run tests (future)

# Terminal UI
cargo run -p yz-cli --bin yz -- tui

# Learning dashboard (TUI)
# - From the Dashboard, press `l` to toggle Performance ⇄ Learning.
# - Learning reads `runs/<id>/logs/metrics.ndjson` and shows per-iteration:
#   - `learn_summary` (trainer): buffer staleness, policy target spikiness + alignment (KL/entropy gap),
#     value distribution + calibration bins (ECE + tiny reliability diagram), training throughput.
#   - `selfplay_summary` (controller): merged worker-local search stats (visit entropy/max-share,
#     fallbacks/pending collisions, prior-vs-visit “overturn” rate, self-play throughput).
# - Optional sampled per-move roots are written worker-locally under:
#   `runs/<id>/logs_workers/worker_*/mcts_root_sample.ndjson`
# - Worker-local merge inputs are written as:
#   `runs/<id>/logs_workers/worker_*/selfplay_worker_summary.json`
#   Controlled by `selfplay.root_sample_every_n` (0 disables; default 10).

# Inference server (Python)
cd python
uv run python -m yatzy_az infer-server --bind unix:///tmp/yatzy_infer.sock --device cpu --best dummy --cand dummy
# Apple GPU (MPS) inference (optional):
# - requires torch with MPS support (`torch.backends.mps.is_available()==true`)
# - yA0tzy hard-fails if PyTorch fallback-to-CPU is enabled (see PYTORCH_ENABLE_MPS_FALLBACK)
uv run python -m yatzy_az infer-server --bind unix:///tmp/yatzy_infer.sock --device mps --best dummy --cand dummy
# Real checkpoints (E6.5):
uv run python -m yatzy_az model-init --out runs/<id>/models/best.pt --hidden 256 --blocks 2
uv run python -m yatzy_az infer-server --bind unix:///tmp/yatzy_infer.sock --device cpu --best path:runs/<id>/models/best.pt --cand path:runs/<id>/models/candidate.pt
# Optional CPU perf stability knobs (E6.5S5):
uv run python -m yatzy_az infer-server --bind unix:///tmp/yatzy_infer.sock --device cpu --best path:runs/<id>/models/best.pt --cand path:runs/<id>/models/candidate.pt --torch-threads 1 --torch-interop-threads 1

# Notes:
# - Controller/TUI normally starts infer-server for you.
# - If infer-server fails to start due to Python environment, set:
#     YZ_PYTHON_EXE=/abs/path/to/python
# - If you explicitly want the controller to use `uv run`, set:
#     YZ_USE_UV=1

# Benchmarks
cargo bench -p yz-bench              # Run Criterion microbenches
cargo run --bin yz -- bench --bench scoring     # Run via yz wrapper (passes args to cargo bench)
cargo run --bin yz -- bench e2e -- --seconds 10 --parallel 8 --simulations 64 --max-inflight 4 --chance deterministic

# Profiling
cargo run --bin yz -- profile --help
# See documentation/PROFILING.md for more examples
```

### CI

GitHub Actions runs on every push/PR to `main`:
- **Rust**: fmt, clippy, test
- **Python**: ruff check, ruff format, config validation

See `.github/workflows/ci.yml` for details.

---

## Key Design Decisions

### Language Split
- **Rust** handles all latency-sensitive work: game simulation, MCTS, self-play workers
- **Python** handles GPU-bound work: neural network inference and training
- Communication via Unix domain sockets (or TCP) with a batched request/response protocol

### Stochastic MCTS (decision vs chance)
`yz-mcts` supports two equivalent ways to model dice randomness in search:
- **Implicit stochastic transitions** (baseline): dice are sampled as part of `step(s,a)` and children are keyed by the realized next state.
- **Explicit chance nodes for rerolls** (Story S1): KeepMask decisions transition to a `Chance(AfterState)` node, which samples a dice outcome and then transitions to the next decision state. (Fresh-roll chance after `Mark` remains implicit unless/ until extended.)
- **Progressive widening at chance nodes** (Story S2): chance nodes can cap stored outcome children via a power-law schedule and use transient (non-stored) leaf evaluations to keep sampling unbiased. Configured via `mcts.chance_pw.*`.

For gating/eval, played-game dice remain deterministic via `yz-core`’s event-keyed chance stream; search randomness can still be made reproducible via seeded PRNG.

### Inference batching is a first-class performance feature (do not regress it)
Most end-to-end throughput is determined by how efficiently the Python infer-server can form and execute large batches.

- **Infer-server**: `python/yatzy_az/server/` (asyncio + batching + Prometheus-style metrics)
- **Protocol + client**: `rust/yz-infer/` (length-delimited frames over UDS/TCP, background reader/writer threads)
- **MCTS integration**: `rust/yz-mcts/src/infer_client.rs` (encodes state → features, submits to `yz-infer`)

#### Batcher design (current)
The server’s batcher is intentionally structured to avoid “global batch then split” (which crushes effective batch sizes during gating):

- **Ingress**: a single asyncio queue receives requests (`Batcher.enqueue`).
- **Staging**: requests are staged into **per-model queues** (`_pending_by_model[model_id]`).
- **Draining**: after receiving one item, the batcher **opportunistically drains** already-queued items via `get_nowait()` (`_drain_nowait`) to reduce per-item scheduling overhead and form full batches when `max_wait_us` is small.
- **Flush policy**: flush a per-model batch when either:
  - **full**: that model has `len(queue) >= max_batch`, or
  - **deadline**: the oldest queued item for a model exceeds `max_wait_us`.

#### “Do not break batching again” checklist
If you touch the infer-server, batcher, protocol, or scheduler loops, use this list as a hard constraint:

- **Avoid accidental small-batch flushes**
  - Do not introduce per-item awaits in hot loops that prevent draining already-ready queue items.
  - Keep the “drain already queued items nowait” behavior (or an equivalent mechanism). Without it, the system can devolve into tiny batches even at high QPS.
  - Do not build a global batch and then split it by `model_id` after the fact (this creates many sub-batches).

- **Avoid fixed polling sleeps on the Rust side**
  - Do not add fixed sleeps that delay consuming already-arrived inference responses (this adds latency and reduces throughput).
  - Prefer event-driven wakeups: the `yz-infer` client exposes a “progress” signal so outer loops can wait for “some response arrived” without busy-spinning.

- **Validate batching with metrics (this catches regressions immediately)**
  - **Batch size health** (Prometheus endpoint): check `yatzy_infer_requests_total` vs `yatzy_infer_batches_total`.
    - Under steady load, `avg_batch_size ≈ requests_total / batches_total` should be “close” to `inference.max_batch` (or at least not single-digit).
  - **Histogram**: `yatzy_infer_batch_size_bucket{le="..."}`
    - You want most mass in the top bucket(s) near `max_batch` under load.
  - **Worker-side symptoms**: `runs/<id>/logs_workers/worker_*/progress.json`
    - Very high `sched_would_block` + high `infer_latency_*` typically means inference is the bottleneck.
    - Low batch sizes with low utilization is a batching regression.

### Controller + training orchestration (v1)
- **Controller**: in-process Rust (`rust/yz-controller`) invoked by the TUI (`rust/yz-tui`).
- **Training**: executed as a **Python subprocess** (`python -m yatzy_az train ...`), launched by the controller.
  - The subprocess **reads replay shards from disk** (`runs/<id>/replay/`) and writes checkpoints/metrics into the run directory.
  - There is **no tensor IPC** between Rust and Python for training (no per-batch serialization overhead).
  - Observability is via `runs/<id>/run.json` (phase/status + latest scalars) and `runs/<id>/logs/metrics.ndjson` (step events).
  - Controller passes `--best runs/<id>/models/best.pt` to the trainer (E13.2S1).
  - Controller auto-bootstraps `best.pt` via `model-init` if missing (E13.2S2), using `model.hidden_dim` and `model.num_blocks` from config.

### Configuration
- All runtime knobs live in YAML files under `configs/`
- Schema validation in both languages:
  - **Rust**: `yz_core::Config` with serde (`rust/yz-core/src/config.rs`)
  - **Python**: Pydantic models (`python/yatzy_az/config.py`)
- The same YAML file loads in both Rust and Python
- Config files are passed to both `yz` and `yatzy_az` CLIs

#### Inference device selection
`inference.device` supports:
- `cpu` (default)
- `mps` (Apple GPU via Metal; local Macs only)
- `cuda` (NVIDIA GPU)

When `inference.device=mps` we **fail fast** if:
- MPS is not available (`torch.backends.mps.is_available()==false`), or
- `PYTORCH_ENABLE_MPS_FALLBACK` is enabled (prevents silent CPU execution).

If infer-server fails to start, check:
- `runs/<id>/logs/infer_server.log` (log tail is surfaced in the TUI error message)
- `runs/<id>/run.json: controller_error`

### Data Flow
```
┌─────────────┐     UDS/TCP      ┌─────────────┐
│  yz selfplay│ ──────────────▶  │ yatzy_az    │
│  (Rust)     │ ◀──────────────  │ infer-server│
└─────────────┘   batched NN     └─────────────┘
      │           inference            │
      │                                │
      ▼                                ▼
  ┌─────────┐                    ┌─────────┐
  │ Replay  │                    │ PyTorch │
  │ Shards  │ ─────────────────▶ │ Trainer │
  └─────────┘   safetensors      └─────────┘
```

### Naming Conventions
- Rust crates: `yz-*` (kebab-case)
- Python package: `yatzy_az` (snake_case)
- Config files: `lowercase_with_underscores.yaml`

### Runtime environment variables (selected)
- **`YZ_DEBUG_LOG`**: enable debug logging across Rust/Python (off by default; can be expensive in hot paths).
- **`YZ_INFER_PRINT_STATS`**: make the controller start infer-server with periodic batch/rps prints (debug aid).
- **`YZ_PYTHON_EXE`**: override Python executable used by controller/TUI for infer-server and training.
- **`YZ_USE_UV`**: if set, controller uses `uv run python ...` when spawning Python subprocesses.
- **`PYTORCH_ENABLE_MPS_FALLBACK`**: PyTorch MPS fallback knob. **yA0tzy hard-fails when `inference.device=mps` and this is enabled**, because fallback can silently run ops on CPU.

---

## Where to Find Things

| Looking for... | Location |
|----------------|----------|
| Game rules & scoring | `rust/yz-core/` |
| Canonical GameState + POV swap | `rust/yz-core/src/state.rs` |
| Game rules engine (state transitions) | `rust/yz-core/src/engine.rs` |
| Config schema (Rust) | `rust/yz-core/src/config.rs` |
| Config schema (Python) | `python/yatzy_az/config.py` |
| Feature schema + encoder | `rust/yz-features/` |
| Oracle (optimal play solver) | `rust/swedish_yatzy_dp/` |
| Oracle adapter | `rust/yz-oracle/` |
| MCTS implementation (PUCT) | `rust/yz-mcts/` (`src/mcts.rs`) |
| MCTS inference integration (uses `yz-infer`) | `rust/yz-mcts/` (`src/infer_client.rs`) |
| Inference protocol (Rust↔Python) | `rust/yz-infer/` (`src/protocol.rs`, `src/codec.rs`, `src/frame.rs`) |
| Inference client (background IO, tickets, routing) | `rust/yz-infer/` (`src/client.rs`) |
| Replay shards (safetensors) | `rust/yz-replay/` (`src/writer.rs`) |
| NDJSON run logs (iteration + sampled roots) | `rust/yz-logging/` (`src/lib.rs`), outputs to `runs/<id>/logs/` |
| Unified metrics stream (E10.5S2+) | `runs/<id>/logs/metrics.ndjson` (written by `yz selfplay`, `yz gate --run`, and `python -m yatzy_az train`) |
| Run manifest (E8.5.x) | `runs/<id>/run.json` (written by `yz selfplay`, updated by `python -m yatzy_az train`, and finalized by `yz iter finalize`) |
| Live per-worker progress (scheduler + inference stats) | `runs/<id>/logs_workers/worker_<N>/progress.json` |
| Per-worker detailed events/timings | `runs/<id>/logs_workers/worker_<N>/worker_stats.ndjson` |
| Replay snapshot (E8.5.4) | `runs/<id>/replay_snapshot_iter_{iter:03}.json` (preferred; passed by controller, created by training; freezes shard list per iteration) |
| Python replay dataset loader (E8S1) | `python/yatzy_az/replay_dataset.py` |
| Neural network model | `python/yatzy_az/model/` |
| Training loop | `python/yatzy_az/trainer/` |
| Metrics consumer (JSON-only, W&B-friendly) | `python/yatzy_az/wandb_sync.py` (`python -m yatzy_az wandb-sync --run runs/<id>/`) |
| Python inference server (asyncio + batching + metrics) | `python/yatzy_az/server/` |
| Inference checkpoint contract (E6.5S1) | `python/yatzy_az/server/checkpoint.py` |
| Runtime scheduler + GameTask | `rust/yz-runtime/` (`src/game_task.rs`, `src/scheduler.rs`) |
| CLI commands (Rust) | `rust/yz-cli/src/main.rs` |
| Terminal UI (ratatui) | `rust/yz-tui/` |
| Iteration controller (phase/status + runners) | `rust/yz-controller/` |
| CLI commands (Python) | `python/yatzy_az/__main__.py` |
| Config examples | `configs/` |
| Full requirements | `documentation/prd.md` |

---

## Where to Put Things

| Adding... | Put it in |
|-----------|-----------|
| New game logic (actions/scoring/chance) | `rust/yz-core/` |
| New feature encoding | `rust/yz-features/` |
| MCTS improvements | `rust/yz-mcts/` |
| New model architecture | `python/yatzy_az/model/` |
| Training utilities | `python/yatzy_az/trainer/` |
| New CLI subcommand (Rust) | `rust/yz-cli/src/` |
| New CLI subcommand (Python) | `python/yatzy_az/__main__.py` |
| Config schema updates | `configs/` + Python validator |

---

## Development Milestones

See `documentation/prd.md` Section 14 for full roadmap.

| Epoch | Focus |
|-------|-------|
| **E0** | Skeleton, config, CI, oracle vendoring |
| **E1** | Core game logic, feature encoding, oracle integration |
| **E2** | MCTS, inference protocol, self-play |
| **E3** | Training loop, gating, iteration orchestration |
| **E4** | Logging, profiling, polish |

Current status: **E11 complete** (microbenches + e2e bench + profiling wrapper/docs) and **E10.5 complete** (run-local `config.yaml` snapshots + unified `logs/metrics.ndjson` emitted by selfplay/gate/train + JSON-only `yatzy_az wandb-sync` consumer). Note: per-seed gating results in `gate_report.json` are still optional and not implemented.

Terminal UI status: **Epic E13.1 complete** (replay pruning, controller iteration loop, epochs vs steps). **Epic E13.2 complete**: E13.2S1–E13.2S5 are all done (controller passes `--best`, auto-bootstrap `best.pt`, auto-promotion, model hot-reload, TUI preflight checks). Additional observability: System (`i`) and Search (`s`) screens, plus `infer_snapshot` playback in the TUI for inference/batching diagnostics.

### Automatic promotion (E13.2S3)
- After gating, `finalize_iteration` in `yz-controller` checks if `win_rate >= threshold`.
- If promoted, `candidate.pt` is atomically copied to `best.pt` (via temp file + rename).
- A `MetricsPromotionV1` event is emitted to `runs/<id>/logs/metrics.ndjson` for traceability.
- `run.json.iterations[].promoted` records the decision; `manifest.best_checkpoint` is updated on promotion.

### Model hot-reload (E13.2S4)
- The Python inference server exposes `POST /reload` on the metrics HTTP endpoint (default `127.0.0.1:18080`).
- Request body: `{"model_id": "best"|"cand", "path": "/abs/path/to/checkpoint.pt"}`.
- `Batcher.replace_model()` atomically swaps the model (GIL-safe dict assignment).
- Prometheus counter `yatzy_infer_model_reloads_total` tracks reload events.
- Rust controller calls:
  - `reload_best_for_selfplay()` before selfplay phase.
  - `reload_models_for_gating()` before gating phase (reloads both best + candidate).
- Config knob: `inference.metrics_bind` (default `127.0.0.1:18080`).

### TUI preflight checks (E13.2S5)
- Before starting an iteration, the TUI verifies the inference server supports hot-reload.
- Python server exposes `GET /capabilities` returning `{"version": "2", "hot_reload": true|false}`.
- Rust function `check_server_supports_hot_reload()` in `yz-tui` calls this endpoint and parses the response.
- If hot-reload is not supported, `start_iteration()` fails with a clear error message prompting the user to restart the server.
- Dashboard displays `model_reloads: N` counter from `run.json.model_reloads` (incremented by controller after each successful reload).

### Replay shard naming + retention (E13.1S1)
- Shards are stored as paired files under `runs/<id>/replay/`:
  - `shard_{idx:06}.safetensors`
  - `shard_{idx:06}.meta.json`
- `ShardWriter` resumes at `max_existing_idx + 1` to avoid overwriting shards when appending more data to an existing run.
- If `replay.capacity_shards` is set, we keep the **newest N shards by filename index** and delete older pairs.
- Pruning emits a `replay_prune` event into `runs/<id>/logs/metrics.ndjson` for auditability.

### TUI dashboard: source of truth (v1)
- Progress bars and iteration summaries are driven primarily by `runs/<id>/run.json`:
  - `controller_iteration_idx` selects the current entry in `iterations[]`
  - `iterations[].selfplay.games_completed/games_target` drives self-play progress
  - `iterations[].gate.games_completed/games_target` drives gating progress
  - training loss scalars are stored as the latest values in `run.json` (best-effort) and copied into `iterations[]` by the controller

### Controller loop semantics (E13.1S2)
- `controller.total_iterations` is an **absolute cap** per run directory.
  - `run.json.controller_iteration_idx` counts completed iterations.
  - Starting the controller when `controller_iteration_idx >= total_iterations` is a no-op (immediately `done`).
  - Otherwise, the controller runs remaining iterations until `controller_iteration_idx == total_iterations`.

### Training duration semantics (E13.1S3)
- Preferred: specify `training.steps_per_iteration` so each iteration has a **stable compute budget**.
- If `steps_per_iteration` is not set, training derives a deterministic step target from the frozen replay snapshot:
  - `steps_per_epoch = ceil(replay_snapshot.total_samples / training.batch_size)`
  - `steps_target = training.epochs * steps_per_epoch`
- Epochs-mode requires replay snapshot semantics (or explicit CLI `--steps`) for reproducibility.
- Trainer records `iterations[].train.steps_target` in `run.json` and emits a one-time `train_plan` event in `logs/metrics.ndjson`.

---

## Extending This Document

When making significant changes:

1. Update the **Repository Structure** section if adding new directories
2. Update **Where to Find/Put Things** tables for new modules
3. Add new **Tooling** entries if introducing dependencies
4. Update **Development Milestones** as epochs complete

Keep sections concise and scannable. Link to `prd.md` for detailed requirements.

