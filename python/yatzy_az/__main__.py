"""CLI entrypoint for yatzy_az.

Usage:
    python -m yatzy_az --help
    python -m yatzy_az infer-server ...
    python -m yatzy_az train ...
"""

import argparse
import sys

from . import __version__
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

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
