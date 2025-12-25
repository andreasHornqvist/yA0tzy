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


class GatingConfig(BaseModel):
    """Gating (candidate vs best evaluation) configuration."""

    games: int = Field(description="Number of games to play for evaluation")
    win_rate_threshold: float = Field(description="Win rate threshold for promotion (0.55 = 55%)")
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
