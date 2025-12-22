# yA0tzy

High-throughput AlphaZero-style training for 2-player Yatzy.

## Prerequisites

- **Rust** (stable, latest) — install via [rustup](https://rustup.rs/)
- **Python 3.12+**
- **uv** — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Project Structure

```
yA0tzy/
├── rust/                 # Rust workspace
│   ├── yz-core/          # Rules, scoring, state, action mapping
│   ├── yz-features/      # Feature schema + encoding
│   ├── yz-mcts/          # PUCT, tree arena, async leaf pipeline
│   ├── yz-runtime/       # Schedulers, GameTask state machines
│   ├── yz-infer/         # Socket protocol + inference client
│   ├── yz-replay/        # Safetensors shards + readers
│   ├── yz-eval/          # Gating games + stats aggregation
│   ├── yz-oracle/        # Adapter to swedish_yatzy_dp oracle
│   ├── yz-logging/       # NDJSON events + metrics/tracing
│   └── yz-cli/           # `yz` binary
├── python/               # Python package
│   └── yatzy_az/         # Inference server, trainer, model
├── configs/              # YAML configuration files
└── prd.md                # Product requirements document
```

## Quickstart

### Rust

```bash
# Build and test the Rust workspace
cargo test --workspace

# Run the CLI
cargo run --bin yz -- --help
```

### Python

```bash
# Using uv (recommended)
cd python
uv run python -m yatzy_az --help

# Or with pip
cd python
pip install -e .
python -m yatzy_az --help
```

### Available Commands

**Rust CLI (`yz`)**:
- `yz oracle expected` — Print oracle expected score
- `yz oracle sim` — Run oracle solitaire simulation
- `yz selfplay` — Run self-play with MCTS + inference
- `yz gate` — Gate candidate vs best model
- `yz oracle-eval` — Evaluate models against oracle baseline
- `yz bench` — Run micro-benchmarks
- `yz profile` — Run with profiler hooks enabled

**Python CLI (`yatzy_az`)**:
- `python -m yatzy_az infer-server` — Run batched inference server
- `python -m yatzy_az train` — Train candidate model from replay
- `python -m yatzy_az controller` — Run one iteration end-to-end

## Configuration

Configuration files are in `configs/`. See `configs/local_cpu.yaml` for an example.

## Development

See [prd.md](prd.md) for the full product requirements document.
