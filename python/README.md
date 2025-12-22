# yatzy-az

Python components for AlphaZero-style Yatzy training.

## Components

- **Inference Server**: Batched PyTorch inference server (UDS/TCP)
- **Trainer**: Replay → candidate model training
- **Config**: Pydantic-validated configuration schema

## Installation

```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Usage

```bash
# Run CLI
python -m yatzy_az --help

# Load config
python -c "from yatzy_az.config import load_config; print(load_config('../configs/local_cpu.yaml'))"
```

## Configuration

See `configs/local_cpu.yaml` for an example configuration file.
The config schema is defined in `yatzy_az/config.py` and is compatible
with the Rust configuration loader in `yz-core`.

