"""Training entrypoints (PRD Epic E8).

E8S2 scope: policy+value model, losses, and a minimal training loop with checkpointing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--replay", required=True, help="Path to replay shards directory")
    p.add_argument(
        "--config",
        default=None,
        help="Path to YAML config to snapshot into runs/<id>/config.yaml if missing (run layout only)",
    )
    p.add_argument(
        "--best", default=None, help="Path to best checkpoint (required unless --resume)"
    )
    p.add_argument("--resume", default=None, help="Path to candidate checkpoint to resume from")
    p.add_argument("--out", required=True, help="Output directory for checkpoints")
    p.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    p.add_argument("--num-workers", type=int, default=0, help="Torch DataLoader workers")
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of optimizer steps (overrides config training.steps_per_iteration)",
    )
    p.add_argument("--shuffle-shards", action="store_true", help="Shuffle shards per epoch")
    p.add_argument(
        "--no-repeat", action="store_true", help="Iterate one pass over shards then stop"
    )
    p.add_argument("--device", default="cpu", help="Device: cpu/cuda")
    p.add_argument("--seed", type=int, default=0, help="RNG seed")
    p.add_argument(
        "--lr", type=float, default=None, help="Adam/AdamW learning rate (overrides config)"
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="Weight decay (L2) for AdamW (overrides config)",
    )
    p.add_argument("--value-lambda", type=float, default=1.0, help="Weight for value loss")
    p.add_argument("--hidden", type=int, default=256, help="Hidden size")
    p.add_argument("--blocks", type=int, default=2, help="Number of residual blocks")
    p.add_argument("--log-every", type=int, default=25, help="Log scalars every N steps")
    p.add_argument(
        "--snapshot",
        default=None,
        help="Replay snapshot path (default: runs/<id>/replay_snapshot.json when run layout detected)",
    )
    p.add_argument(
        "--no-snapshot",
        action="store_true",
        help="Disable replay snapshot semantics (ad-hoc training only)",
    )


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    d = torch.load(path, map_location="cpu")
    if not isinstance(d, dict):
        raise RuntimeError(f"invalid checkpoint (expected dict): {path}")
    return d


def _init_model_and_opt(*, hidden: int, blocks: int, lr: float, weight_decay: float, device):
    import torch

    from ..model import YatzyNet, YatzyNetConfig

    model = YatzyNet(YatzyNetConfig(hidden=hidden, blocks=blocks)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=float(weight_decay))
    return model, opt


def init_from_best(
    *, best_path: Path, hidden: int, blocks: int, lr: float, weight_decay: float, device
):
    """Initialize candidate from best weights with a fresh optimizer (no state)."""
    d = _load_checkpoint(best_path)
    if "model" not in d:
        raise RuntimeError(f"best checkpoint missing 'model' key: {best_path}")

    model, opt = _init_model_and_opt(
        hidden=hidden, blocks=blocks, lr=lr, weight_decay=weight_decay, device=device
    )
    model.load_state_dict(d["model"])

    # Boundary requirement: starting from best must not reuse optimizer state.
    if len(opt.state) != 0:
        raise RuntimeError("optimizer state is not empty after creation (expected reset)")

    return model, opt, 0


def resume_candidate(
    *, candidate_path: Path, hidden: int, blocks: int, lr: float, weight_decay: float, device
):
    """Resume candidate training by loading model + optimizer state."""
    d = _load_checkpoint(candidate_path)
    if "model" not in d:
        raise RuntimeError(f"candidate checkpoint missing 'model' key: {candidate_path}")

    model, opt = _init_model_and_opt(
        hidden=hidden, blocks=blocks, lr=lr, weight_decay=weight_decay, device=device
    )
    model.load_state_dict(d["model"])
    if "optimizer" in d and d["optimizer"] is not None:
        opt.load_state_dict(d["optimizer"])

    meta = d.get("meta", {})
    train_step = int(meta.get("train_step", 0)) if isinstance(meta, dict) else 0
    return model, opt, train_step


def _ensure_run_config_snapshot(*, run_root: Path, config_path: Path | None) -> None:
    """Ensure `run_root/config.yaml` exists for run-layout training."""
    if not (run_root / "run.json").exists():
        return

    snap = run_root / "config.yaml"
    if snap.exists():
        return

    if config_path is None:
        raise RuntimeError(f"missing run config snapshot: {snap}. Provide --config to write it.")

    import yaml

    from ..config import load_config

    cfg = load_config(Path(config_path))
    tmp = run_root / "config.yaml.tmp"
    # normalized YAML (not preserving comments)
    tmp.write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
    tmp.replace(snap)

    # Best-effort: record snapshot reference in run.json if present.
    try:
        from ..run_manifest import load_manifest, save_manifest_atomic

        m = load_manifest(run_root)
        m["config_snapshot"] = snap.name
        save_manifest_atomic(run_root, m)
    except Exception:
        pass


def _append_metrics_train_step(
    *,
    run_root: Path,
    train_step: int,
    loss_total: float,
    loss_policy: float,
    loss_value: float,
    entropy: float,
    lr: float,
    throughput_steps_s: float,
) -> None:
    """Append a train_step metrics event to runs/<id>/logs/metrics.ndjson (best-effort)."""
    run_root = Path(run_root)
    if not (run_root / "run.json").exists():
        return

    logs_dir = run_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "metrics.ndjson"

    run_id = run_root.name
    git_hash = None
    config_snapshot = "config.yaml" if (run_root / "config.yaml").exists() else None
    try:
        from ..run_manifest import load_manifest

        m = load_manifest(run_root)
        run_id = str(m.get("run_id", run_id))
        git_hash = m.get("git_hash")
        config_snapshot = m.get("config_snapshot", config_snapshot)
    except Exception:
        pass

    from ..replay_dataset import FEATURE_SCHEMA_ID, PROTOCOL_VERSION, RULESET_ID

    ev = {
        "event": "train_step",
        "ts_ms": int(time.time() * 1000),
        "run_id": run_id,
        "v": {
            "protocol_version": int(PROTOCOL_VERSION),
            "feature_schema_id": int(FEATURE_SCHEMA_ID),
            "action_space_id": "oracle_keepmask_v1",
            "ruleset_id": str(RULESET_ID),
        },
        "git_hash": git_hash,
        "config_snapshot": config_snapshot,
        "train_step": int(train_step),
        "loss_total": float(loss_total),
        "loss_policy": float(loss_policy),
        "loss_value": float(loss_value),
        "entropy": float(entropy),
        "lr": float(lr),
        "throughput_steps_s": float(throughput_steps_s),
    }

    with metrics_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(ev) + "\n")


def run_from_args(args: argparse.Namespace) -> int:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "torch is required for `yatzy_az train` (E8). Install with the `train` extra."
        ) from e

    from ..replay_dataset import (
        ACTION_SPACE_A,
        FEATURE_LEN,
        FEATURE_SCHEMA_ID,
        PROTOCOL_VERSION,
        RULESET_ID,
        ReplayIterableDataset,
    )

    replay_dir = Path(args.replay)

    # E8.5.4: snapshot semantics (fixed shard list for the iteration).
    out_dir = Path(args.out)
    run_root = out_dir.parent

    # E10.5S1: ensure run-local config snapshot exists.
    _ensure_run_config_snapshot(run_root=run_root, config_path=args.config)
    # Prefer run-local config snapshot values when present, unless CLI overrides are provided.
    cfg = None
    cfg_path = run_root / "config.yaml"
    if cfg_path.exists():
        try:
            from ..config import load_config

            cfg = load_config(cfg_path)
        except Exception:
            cfg = None

    if cfg is not None:
        if args.batch_size is None:
            args.batch_size = int(cfg.training.batch_size)
        if args.lr is None:
            args.lr = float(cfg.training.learning_rate)
        if args.weight_decay is None:
            args.weight_decay = float(cfg.training.weight_decay)
        if args.steps is None:
            s = cfg.training.steps_per_iteration
            args.steps = int(s) if s is not None else None

    # Final defaults (backwards compatible when no config is present).
    if args.batch_size is None:
        args.batch_size = 256
    if args.lr is None:
        args.lr = 1e-3
    if args.weight_decay is None:
        args.weight_decay = 0.0
    if args.steps is None:
        # If config didn't specify steps_per_iteration, keep existing default behavior.
        args.steps = 200
    shard_files = None
    if not bool(args.no_snapshot):
        snapshot_path = (
            Path(args.snapshot) if args.snapshot is not None else run_root / "replay_snapshot.json"
        )
        from ..replay_snapshot import create_snapshot, load_snapshot, shard_filenames

        if snapshot_path.exists():
            snap = load_snapshot(snapshot_path)
        else:
            snap = create_snapshot(replay_dir=replay_dir, out_path=snapshot_path)

        shard_files = shard_filenames(snap)

        # Best-effort: record snapshot reference in run.json if present.
        try:
            from ..run_manifest import load_manifest, save_manifest_atomic

            m = load_manifest(run_root)
            m["replay_snapshot"] = snapshot_path.name
            save_manifest_atomic(run_root, m)
        except Exception:
            pass

    ds = ReplayIterableDataset(
        replay_dir,
        shard_files=shard_files,
        shuffle_shards=bool(args.shuffle_shards),
        seed=int(args.seed),
        repeat=not bool(args.no_repeat),
    )
    dl = DataLoader(ds, batch_size=int(args.batch_size), num_workers=int(args.num_workers))

    device = torch.device(str(args.device))
    torch.manual_seed(int(args.seed))

    if args.resume is not None and args.best is not None:
        raise RuntimeError("use either --resume or --best, not both")

    if args.resume is not None:
        model, opt, start_step = resume_candidate(
            candidate_path=Path(args.resume),
            hidden=int(args.hidden),
            blocks=int(args.blocks),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=device,
        )
    else:
        if args.best is None:
            raise RuntimeError("--best is required when starting a new iteration (unless --resume)")
        model, opt, start_step = init_from_best(
            best_path=Path(args.best),
            hidden=int(args.hidden),
            blocks=int(args.blocks),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            device=device,
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    it = iter(dl)
    last_log_t = time.perf_counter()
    last_log_step = start_step
    for step in range(start_step + 1, start_step + int(args.steps) + 1):
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
                now_t = time.perf_counter()
                dt = max(1e-9, now_t - last_log_t)
                steps_done = step - last_log_step
                throughput = float(steps_done) / float(dt)
                last_log_t = now_t
                last_log_step = step
                print(
                    f"step={step} loss={loss.item():.4f} "
                    f"loss_pi={loss_pi.item():.4f} loss_v={loss_v.item():.4f} "
                    f"entropy={ent.item():.3f} v_mean={v.mean().item():.3f}"
                )

                # E10.5S2: unified metrics stream (best-effort).
                _append_metrics_train_step(
                    run_root=run_root,
                    train_step=int(step),
                    loss_total=float(loss.item()),
                    loss_policy=float(loss_pi.item()),
                    loss_value=float(loss_v.item()),
                    entropy=float(ent.item()),
                    lr=float(args.lr),
                    throughput_steps_s=float(throughput),
                )

    ckpt_path = out_dir / "candidate.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "config": {
                "hidden": int(args.hidden),
                "blocks": int(args.blocks),
                "feature_len": FEATURE_LEN,
                "action_space_a": ACTION_SPACE_A,
            },
            "meta": {
                "protocol_version": PROTOCOL_VERSION,
                "feature_schema_id": FEATURE_SCHEMA_ID,
                "feature_len": FEATURE_LEN,
                "ruleset_id": RULESET_ID,
                "action_space_a": ACTION_SPACE_A,
                "train_step": step,
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
        "train_step": step,
    }
    (out_dir / "candidate.meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    # E8.5.1: if this is a standard run layout (runs/<id>/models), update runs/<id>/run.json.
    run_root = out_dir.parent
    try:
        from ..run_manifest import load_manifest, save_manifest_atomic

        m = load_manifest(run_root)
        m["train_step"] = int(step)
        m["candidate_checkpoint"] = str(Path("models") / ckpt_path.name)
        save_manifest_atomic(run_root, m)
    except Exception:
        # Best-effort; training can still run without a manifest.
        pass

    print(f"saved: {ckpt_path}")
    return 0
