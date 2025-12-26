"""Training entrypoints (PRD Epic E8).

E8S1 scope: a loader smoke test that iterates replay batches. No model/training yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True, help="Path to replay shards directory")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--num-workers", type=int, default=0, help="Torch DataLoader workers")
    p.add_argument("--steps", type=int, default=5, help="Number of batches to iterate")
    p.add_argument("--shuffle-shards", action="store_true", help="Shuffle shards per epoch")
    p.add_argument(
        "--no-repeat", action="store_true", help="Iterate one pass over shards then stop"
    )


def run_from_args(args: argparse.Namespace) -> int:
    try:
        from torch.utils.data import DataLoader
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "torch is required for `yatzy_az train` (E8S1). Install with the `train` extra."
        ) from e

    from .replay_dataset import ReplayIterableDataset

    ds = ReplayIterableDataset(
        Path(args.replay),
        shuffle_shards=bool(args.shuffle_shards),
        seed=0,
        repeat=not bool(args.no_repeat),
    )
    dl = DataLoader(ds, batch_size=int(args.batch_size), num_workers=int(args.num_workers))

    n = 0
    for batch in dl:
        n += 1
        x, legal, pi, z, z_margin = batch
        z_margin_shape = None if z_margin is None else tuple(z_margin.shape)
        print(
            f"batch {n}: x={tuple(x.shape)} legal={tuple(legal.shape)} pi={tuple(pi.shape)} "
            f"z={tuple(z.shape)} z_margin={z_margin_shape}"
        )
        if n >= int(args.steps):
            break

    print(f"done: batches={n}")
    return 0
