"""CLI entrypoint for yatzy_az.

Usage:
    python -m yatzy_az --help
    python -m yatzy_az infer-server ...
    python -m yatzy_az train ...
"""

import argparse
import sys
from pathlib import Path

from . import __version__, wandb_sync
from .model.init import init_model_checkpoint
from .server import server as infer_server
from .trainer import train as train_mod


def cmd_infer_server(args: argparse.Namespace) -> int:
    """Run the batched inference server."""
    # Model/device args are placeholders for later (PyTorch integration).
    return infer_server.run_from_args(args)


def cmd_train(args: argparse.Namespace) -> int:
    """Run training from replay shards."""
    return train_mod.run_from_args(args)


def cmd_controller(args: argparse.Namespace) -> int:
    """Run one full iteration end-to-end (optional orchestration)."""
    print("Controller (not yet implemented)")
    return 0


def cmd_wandb_sync(args: argparse.Namespace) -> int:
    """Consume runs/<id>/logs/metrics.ndjson and emit W&B-friendly JSON."""
    return wandb_sync.run_from_args(args)


def cmd_model_init(args: argparse.Namespace) -> int:
    """Create a fresh best.pt checkpoint for a new run."""
    out = Path(args.out)
    init_model_checkpoint(out, hidden=int(args.hidden), blocks=int(args.blocks))
    print(f"wrote: {out}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="yatzy_az",
        description="AlphaZero-style training for 2-player Yatzy",
    )
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # infer-server
    p_infer = subparsers.add_parser(
        "infer-server",
        help="Run batched PyTorch inference server (UDS/TCP)",
    )
    p_infer.add_argument("--model", default="best.pt", help="Model checkpoint path")
    p_infer.add_argument("--bind", default="unix:///tmp/yatzy_infer.sock", help="Bind address")
    p_infer.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    infer_server.add_args(p_infer)
    p_infer.set_defaults(func=cmd_infer_server)

    # train
    p_train = subparsers.add_parser(
        "train",
        help="Train candidate model from replay shards",
    )
    train_mod.add_args(p_train)
    p_train.set_defaults(func=cmd_train)

    # controller (optional)
    p_ctrl = subparsers.add_parser(
        "controller",
        help="Run one iteration end-to-end (self-play + train + gate)",
    )
    p_ctrl.set_defaults(func=cmd_controller)

    # wandb-sync
    p_wandb = subparsers.add_parser(
        "wandb-sync",
        help="Read runs/<id>/logs/metrics.ndjson and emit W&B-friendly JSON (stdout)",
    )
    wandb_sync.add_args(p_wandb)
    p_wandb.set_defaults(func=cmd_wandb_sync)

    # model-init
    p_init = subparsers.add_parser(
        "model-init",
        help="Initialize a fresh best.pt checkpoint for a new run",
    )
    p_init.add_argument("--out", required=True, help="Output checkpoint path (e.g. runs/<id>/models/best.pt)")
    p_init.add_argument("--hidden", type=int, default=256, help="Hidden size")
    p_init.add_argument("--blocks", type=int, default=2, help="Number of residual blocks")
    p_init.add_argument(
        "--device",
        default="cpu",
        help="Initialization device (cpu/cuda). Note: checkpoint is saved on CPU.",
    )
    p_init.set_defaults(func=cmd_model_init)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
