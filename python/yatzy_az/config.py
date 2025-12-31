"""Unified configuration schema for yA0tzy.

This module defines the configuration structure that is shared between
Rust and Python components. The same YAML file should load in both.
"""

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class InferenceConfig(BaseModel):
    """Inference server configuration."""

    bind: str = Field(
        description='Bind address (e.g., "unix:///tmp/yatzy_infer.sock" or "tcp://host:port")'
    )
    device: str = Field(description='Device: "cpu" or "cuda"')
    max_batch: int = Field(description="Maximum batch size before flushing")
    max_wait_us: int = Field(
        description="Maximum wait time (microseconds) before flushing a partial batch"
    )
    torch_threads: int | None = Field(
        default=None, description="Optional: torch intra-op threads (CPU perf stability)"
    )
    torch_interop_threads: int | None = Field(
        default=None, description="Optional: torch inter-op threads (CPU perf stability)"
    )
    metrics_bind: str = Field(
        default="127.0.0.1:18080",
        description="Metrics/control HTTP bind address for hot-reloading models (E13.2S4)",
    )


class MctsConfig(BaseModel):
    """MCTS algorithm configuration."""

    c_puct: float = Field(description="PUCT exploration constant")
    budget_reroll: int = Field(description="Simulation budget for reroll decisions")
    budget_mark: int = Field(
        description="Simulation budget for mark (category selection) decisions"
    )
    max_inflight_per_game: int = Field(description="Maximum in-flight inference requests per game")
    dirichlet_alpha: float = Field(
        default=0.3, description="Dirichlet noise alpha (self-play only)"
    )
    dirichlet_epsilon: float = Field(
        default=0.25,
        description="Dirichlet noise epsilon - fraction of noise mixed into priors",
    )
    temperature_schedule: dict = Field(
        default_factory=lambda: {"kind": "constant", "t0": 1.0},
        description=(
            "Executed-move temperature schedule (does not affect replay pi targets). "
            'Example: {"kind":"step","t0":1.0,"t1":0.0,"cutoff_ply":10}'
        ),
    )


class SelfplayConfig(BaseModel):
    """Self-play configuration."""

    games_per_iteration: int = Field(
        description="Number of games to generate per training iteration"
    )
    workers: int = Field(description="Number of worker processes")
    threads_per_worker: int = Field(description="Number of threads per worker process")


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = Field(description="Mini-batch size for gradient updates")
    learning_rate: float = Field(description="Learning rate")
    epochs: int = Field(description="Number of epochs per training iteration")
    weight_decay: float = Field(
        default=0.0, description="Weight decay (L2) for AdamW/SGD (0 disables)"
    )
    steps_per_iteration: int | None = Field(
        default=None,
        description=(
            "Optional number of optimizer steps per iteration. "
            "If set, takes precedence over epochs."
        ),
    )


class ReplayConfig(BaseModel):
    """Replay retention configuration."""

    capacity_shards: int | None = Field(
        default=None,
        description=(
            "Keep at most N replay shards under runs/<id>/replay/ (prune older). "
            "If None, do not prune automatically."
        ),
    )


class ControllerConfig(BaseModel):
    """Iteration controller configuration."""

    total_iterations: int | None = Field(
        default=None,
        description=(
            "Optional number of full iterations to run (selfplay → train → gate). "
            "If None, controller runs until stopped."
        ),
    )


class ModelConfig(BaseModel):
    """Neural network model architecture configuration."""

    hidden_dim: int = Field(default=256, description="Hidden layer size for the neural network")
    num_blocks: int = Field(default=2, description="Number of residual blocks in the network")


class GatingConfig(BaseModel):
    """Gating (candidate vs best evaluation) configuration."""

    games: int = Field(description="Number of games to play for evaluation")
    seed: int = Field(
        default=0, description="Base seed for deterministic paired-seed scheduling in gating"
    )
    seed_set_id: str | None = Field(
        default="dev_v1",
        description=(
            "Optional fixed dev seed set id. If set, gating loads "
            "`configs/seed_sets/<id>.txt` and uses those seeds instead of deriving from `seed`."
        ),
    )
    win_rate_threshold: float = Field(
        default=0.55, description="Win rate threshold for promotion (0.55 = 55%)"
    )
    paired_seed_swap: bool = Field(
        description="Use paired seeds with side swap for reduced variance"
    )
    deterministic_chance: bool = Field(
        default=True,
        description=(
            "Use deterministic event-keyed chance stream for gating/eval (optional). "
            "Recommended for reproducible experiments; disable for more realistic variance."
        ),
    )


class Config(BaseModel):
    """Root configuration structure."""

    inference: InferenceConfig
    mcts: MctsConfig
    selfplay: SelfplayConfig
    training: TrainingConfig
    gating: GatingConfig
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def load_config(path: str | Path) -> Config:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated Config object.

    Raises:
        FileNotFoundError: If the config file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
        pydantic.ValidationError: If the config doesn't match the schema.
    """
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return Config.model_validate(data)


def load_config_from_string(yaml_str: str) -> Config:
    """Load configuration from a YAML string.

    Args:
        yaml_str: YAML configuration as a string.

    Returns:
        Validated Config object.

    Raises:
        yaml.YAMLError: If the YAML is malformed.
        pydantic.ValidationError: If the config doesn't match the schema.
    """
    data = yaml.safe_load(yaml_str)
    return Config.model_validate(data)


if __name__ == "__main__":
    # Quick test: load the local_cpu config
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        # Default to local_cpu.yaml relative to this file
        config_path = Path(__file__).parent.parent.parent / "configs" / "local_cpu.yaml"

    config = load_config(config_path)
    print(f"Loaded config from {config_path}")
    print(config.model_dump_json(indent=2))
