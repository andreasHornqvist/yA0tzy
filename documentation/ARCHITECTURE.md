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
│   ├── yz-cli/               # `yz` binary (CLI entrypoint)
│   └── swedish_yatzy_dp/     # Vendored oracle (optimal-play DP solver)
│
├── python/                   # Python package
│   ├── pyproject.toml        # PEP 621 metadata (uv-compatible)
│   ├── .python-version       # Python version pin (3.12)
│   └── yatzy_az/             # Main package
│       ├── __init__.py
│       ├── __main__.py       # CLI entrypoint (python -m yatzy_az)
│       ├── model/            # Neural network architecture (future)
│       ├── trainer/          # Training loop (future)
│       └── server/           # Inference server (future)
│
├── configs/                  # YAML configuration files
│   └── local_cpu.yaml        # Example config for local CPU runs
│
├── documentation/            # Project documentation
│   ├── prd.md                # Product Requirements Document
│   └── ARCHITECTURE.md       # This file
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

# Python
cd python
uv sync --all-extras                 # Install all dependencies (including dev)
uv run python -m yatzy_az --help     # Run Python CLI
uv run ruff check .                  # Lint
uv run ruff format --check .         # Check formatting
uv run pytest                        # Run tests (future)
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

### Configuration
- All runtime knobs live in YAML files under `configs/`
- Schema validation in both languages:
  - **Rust**: `yz_core::Config` with serde (`rust/yz-core/src/config.rs`)
  - **Python**: Pydantic models (`python/yatzy_az/config.py`)
- The same YAML file loads in both Rust and Python
- Config files are passed to both `yz` and `yatzy_az` CLIs

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
| Neural network model | `python/yatzy_az/model/` (future) |
| Training loop | `python/yatzy_az/trainer/` (future) |
| Python inference server (asyncio + batching + metrics) | `python/yatzy_az/infer_server/` |
| Runtime scheduler + GameTask | `rust/yz-runtime/` (`src/game_task.rs`, `src/scheduler.rs`) |
| CLI commands (Rust) | `rust/yz-cli/src/main.rs` |
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

Current status: **E7S2 complete** (`yz-replay` shard writer + runtime sample emission; `yz selfplay` writes replay shards incrementally)

---

## Extending This Document

When making significant changes:

1. Update the **Repository Structure** section if adding new directories
2. Update **Where to Find/Put Things** tables for new modules
3. Add new **Tooling** entries if introducing dependencies
4. Update **Development Milestones** as epochs complete

Keep sections concise and scannable. Link to `prd.md` for detailed requirements.

