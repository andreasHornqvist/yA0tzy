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
    device: str = Field(description='Device: "cpu", "mps", or "cuda"')
    protocol_version: int = Field(
        default=2,
        description=(
            "Inference protocol version for Rust↔Python.\n"
            "- 1: v1 (legacy)\n"
            "- 2: v2 (packed f32 tensors)"
        ),
    )
    legal_mask_bitset: bool = Field(
        default=False,
        description=(
            "If true and protocol_version==2, encode legal_mask as a compact 6-byte bitset (A=47)."
        ),
    )
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
    debug_log: bool = Field(
        default=False,
        description=(
            "Enable debug logging across Rust/Python components for this run "
            "(maps to YZ_DEBUG_LOG=1 for spawned subprocesses)."
        ),
    )
    print_stats: bool = Field(
        default=False,
        description=(
            "Make infer-server print periodic throughput/batching stats "
            "(maps to YZ_INFER_PRINT_STATS=1 / --print-stats-every-s)."
        ),
    )
    metrics_bind: str = Field(
        default="127.0.0.1:18080",
        description="Metrics/control HTTP bind address for hot-reloading models (E13.2S4)",
    )


class MctsConfig(BaseModel):
    """MCTS algorithm configuration."""

    # Allow backward-compatible keys in YAML.
    model_config = {"populate_by_name": True}

    class ChancePwConfig(BaseModel):
        """Chance-node progressive widening knobs (Story S2)."""

        enabled: bool = Field(
            default=False,
            description=(
                "If true, cap stored outcome children at chance nodes using a widening schedule."
            ),
        )
        c: float = Field(
            default=2.0,
            description="Power-law scale for K(N)=ceil(c * N^alpha).",
        )
        alpha: float = Field(
            default=0.6,
            description="Power-law exponent for K(N)=ceil(c * N^alpha), typically in (0,1).",
        )
        max_children: int = Field(
            default=64,
            description="Hard cap on number of stored chance outcome children.",
        )

    class KataGoConfig(BaseModel):
        """KataGo-inspired parallel search knobs."""

        expansion_lock: bool = Field(
            default=False,
            description=(
                "If true, avoid selecting in-flight reserved edges when any non-pending legal edge exists "
                "(reduces parallel herding/collisions)."
            ),
        )

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
            'Example: {"kind":"step","t0":1.0,"t1":0.0,"cutoff_turn":10} (alias: cutoff_ply)'
        ),
    )
    virtual_loss_mode: str = Field(
        default="q_penalty",
        description=(
            "Virtual-loss/inflight scheme.\n"
            "- q_penalty: reserve visits and subtract virtual loss from Q while pending (default)\n"
            "- n_virtual_only: reserve visits only (AlphaYatzy-style)\n"
            "- off: no reservations"
        ),
    )
    virtual_loss: float = Field(
        default=1.0,
        description="Virtual loss magnitude (used when virtual_loss_mode != off).",
    )
    chance_nodes: bool = Field(
        default=False,
        validation_alias="explicit_keepmask_chance",
        description="If true, enable explicit chance nodes for rerolls (Story S1).",
    )
    chance_pw: ChancePwConfig = Field(
        default_factory=ChancePwConfig,
        description="Chance-node progressive widening knobs (Story S2).",
    )
    katago: KataGoConfig = Field(
        default_factory=KataGoConfig,
        description="KataGo-inspired parallel search knobs.",
    )


class SelfplayConfig(BaseModel):
    """Self-play configuration."""

    games_per_iteration: int = Field(
        description="Number of games to generate per training iteration"
    )
    workers: int = Field(description="Number of worker processes")
    threads_per_worker: int = Field(description="Number of threads per worker process")
    root_sample_every_n: int = Field(
        default=10,
        description=(
            "Emit a sampled root-search record every N executed moves (0 disables). "
            "Used for lightweight search diagnostics without large logs."
        ),
    )


class TrainingConfig(BaseModel):
    """Training configuration."""

    batch_size: int = Field(description="Mini-batch size for gradient updates")
    learning_rate: float = Field(description="Learning rate")
    optimizer: str = Field(
        default="adamw",
        description="Optimizer: adamw | adam | sgd",
    )
    continuous_candidate_training: bool = Field(
        default=False,
        description=(
            "If true, keep training the candidate continuously across iterations (even on gate rejects). "
            "Self-play still uses best until promotion."
        ),
    )
    reset_optimizer: bool = Field(
        default=True,
        description=(
            "If true, initialize candidate from best weights with a fresh optimizer each iteration. "
            "If false, resume from the previous iteration's candidate checkpoint (keeps optimizer state)."
        ),
    )
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
    sample_mode: str = Field(
        default="random_indexed",
        description=(
            "Replay sampling mode for training. "
            "sequential streams samples in shard order; random_indexed uses a global index + DataLoader shuffle."
        ),
    )
    dataloader_workers: int = Field(
        default=0,
        description="Torch DataLoader worker processes (0 disables multiprocessing).",
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
    kind: str = Field(
        default="residual",
        description=(
            "Model architecture kind.\n"
            "- residual: residual MLP with LayerNorm (default)\n"
            "- mlp: plain MLP (AlphaYatzy-style)"
        ),
    )


class GatingKatagoConfig(BaseModel):
    """KataGo-style gating options (sequential tests like SPRT)."""

    sprt: bool = Field(
        default=False,
        description=(
            "Enable win-rate SPRT (sequential probability ratio test) around "
            "`gating.win_rate_threshold`."
        ),
    )
    sprt_min_games: int = Field(
        default=40,
        description="Minimum number of games to play before SPRT can decide.",
    )
    sprt_max_games: int = Field(
        default=200,
        description=(
            "Maximum number of games to play when SPRT is enabled. "
            "With paired_seed_swap=true you need at least sprt_max_games/2 seeds."
        ),
    )
    sprt_alpha: float = Field(
        default=0.05,
        description="SPRT alpha (false promote) target.",
    )
    sprt_beta: float = Field(
        default=0.05,
        description="SPRT beta (false reject) target.",
    )
    sprt_delta: float = Field(
        default=0.03,
        description=(
            "SPRT indifference half-width: p0 = thr - delta, p1 = thr + delta "
            "(thr = gating.win_rate_threshold)."
        ),
    )


class GatingConfig(BaseModel):
    """Gating (candidate vs best evaluation) configuration."""

    games: int = Field(description="Number of games to play for evaluation")
    seed: int = Field(
        default=0, description="Base seed for deterministic paired-seed scheduling in gating"
    )
    seed_set_id: str | None = Field(
        default="dev_v2",
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
    katago: GatingKatagoConfig = Field(
        default_factory=GatingKatagoConfig,
        description="KataGo-style gating options (SPRT, etc).",
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
