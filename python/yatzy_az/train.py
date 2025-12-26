"""Training entrypoints (PRD Epic E8).

E8S2 scope: policy+value model, losses, and a minimal training loop with checkpointing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True, help="Path to replay shards directory")
    p.add_argument("--out", required=True, help="Output directory for checkpoints")
    p.add_argument("--batch-size", type=int, default=256, help="Batch size")
    p.add_argument("--num-workers", type=int, default=0, help="Torch DataLoader workers")
    p.add_argument("--steps", type=int, default=200, help="Number of optimizer steps")
    p.add_argument("--shuffle-shards", action="store_true", help="Shuffle shards per epoch")
    p.add_argument(
        "--no-repeat", action="store_true", help="Iterate one pass over shards then stop"
    )
    p.add_argument("--device", default="cpu", help="Device: cpu/cuda")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--value-lambda", type=float, default=1.0, help="Weight for value loss")
    p.add_argument("--hidden", type=int, default=256, help="Hidden size")
    p.add_argument("--blocks", type=int, default=2, help="Number of residual blocks")
    p.add_argument("--log-every", type=int, default=25, help="Log scalars every N steps")


def run_from_args(args: argparse.Namespace) -> int:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "torch is required for `yatzy_az train` (E8S1). Install with the `train` extra."
        ) from e

    from .model import YatzyNet, YatzyNetConfig
    from .replay_dataset import (
        ACTION_SPACE_A,
        FEATURE_LEN,
        FEATURE_SCHEMA_ID,
        PROTOCOL_VERSION,
        RULESET_ID,
        ReplayIterableDataset,
    )

    ds = ReplayIterableDataset(
        Path(args.replay),
        shuffle_shards=bool(args.shuffle_shards),
        seed=int(args.seed),
        repeat=not bool(args.no_repeat),
    )
    dl = DataLoader(ds, batch_size=int(args.batch_size), num_workers=int(args.num_workers))

    device = torch.device(str(args.device))
    torch.manual_seed(int(args.seed))

    model = YatzyNet(YatzyNetConfig(hidden=int(args.hidden), blocks=int(args.blocks))).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    it = iter(dl)
    for step in range(1, int(args.steps) + 1):
        try:
            x, legal, pi, z, _z_margin = next(it)
        except StopIteration:
            it = iter(dl)
            x, legal, pi, z, _z_margin = next(it)

        x = x.to(device=device, dtype=torch.float32)
        legal = legal.to(device=device)
        pi = pi.to(device=device, dtype=torch.float32)
        z = z.to(device=device, dtype=torch.float32)

        # Safety: if pi has any mass on illegal, renormalize over legal to avoid NaNs.
        legal_f = legal.to(dtype=torch.float32)
        pi = pi * legal_f
        pi_sum = pi.sum(dim=1, keepdim=True).clamp_min(1e-12)
        pi = pi / pi_sum

        logits, v = model(x)
        if logits.shape[1] != ACTION_SPACE_A:
            raise RuntimeError(f"bad logits shape: {tuple(logits.shape)}")

        logp = F.log_softmax(logits, dim=1)
        loss_pi = -(pi * logp).sum(dim=1).mean()
        loss_v = F.mse_loss(v, z)
        loss = loss_pi + float(args.value_lambda) * loss_v

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if step % int(args.log_every) == 0 or step == 1:
            with torch.no_grad():
                p = torch.softmax(logits, dim=1)
                ent = -(p * torch.log(p.clamp_min(1e-12))).sum(dim=1).mean()
                print(
                    f"step={step} loss={loss.item():.4f} "
                    f"loss_pi={loss_pi.item():.4f} loss_v={loss_v.item():.4f} "
                    f"entropy={ent.item():.3f} v_mean={v.mean().item():.3f}"
                )

    ckpt_path = out_dir / "candidate.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "hidden": int(args.hidden),
                "blocks": int(args.blocks),
                "feature_len": FEATURE_LEN,
                "action_space_a": ACTION_SPACE_A,
            },
        },
        ckpt_path,
    )

    meta: dict[str, Any] = {
        "protocol_version": PROTOCOL_VERSION,
        "feature_schema_id": FEATURE_SCHEMA_ID,
        "feature_len": FEATURE_LEN,
        "ruleset_id": RULESET_ID,
        "action_space_a": ACTION_SPACE_A,
        "checkpoint": str(ckpt_path.name),
    }
    (out_dir / "candidate.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    print(f"saved: {ckpt_path}")
    return 0
