from __future__ import annotations

import argparse


def test_apply_torch_thread_settings_sets_num_threads() -> None:
    import torch

    from yatzy_az.server.server import ServerConfig, _apply_torch_thread_settings

    prev = torch.get_num_threads()
    try:
        cfg = ServerConfig(
            bind="unix:///tmp/ignored.sock",
            device="cpu",
            max_batch=16,
            max_wait_us=1000,
            print_stats_every_s=0.0,
            best_id=0,
            cand_id=1,
            best_spec="dummy",
            cand_spec="dummy",
            metrics_bind="127.0.0.1:0",
            metrics_disable=True,
            torch_threads=1,
            torch_interop_threads=None,
        )
        _apply_torch_thread_settings(cfg)
        assert torch.get_num_threads() == 1
    finally:
        torch.set_num_threads(prev)


def test_add_args_parses_torch_thread_flags() -> None:
    from yatzy_az.server import server as infer_server

    p = argparse.ArgumentParser()
    p.add_argument("--bind", default="unix:///tmp/x.sock")
    p.add_argument("--device", default="cpu")
    infer_server.add_args(p)
    args = p.parse_args(["--torch-threads", "2"])
    assert args.torch_threads == 2


