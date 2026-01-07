"""Neural network model (PRD E8S2).

`YatzyNet` v1:
- Input: feature vector x ∈ R^45 (FeatureSchema v1)
- Outputs:
  - policy logits ∈ R^47 (oracle_keepmask_v1 action space)
  - value ∈ [-1, 1] (tanh)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

F: int = 45
A: int = 47


@dataclass(frozen=True)
class YatzyNetConfig:
    hidden: int = 256
    blocks: int = 2
    kind: str = "residual"  # residual|mlp


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc1(x)
        y = self.act(y)
        y = self.fc2(y)
        y = x + y
        y = self.act(y)
        return y


class YatzyNet(nn.Module):
    def __init__(self, cfg: YatzyNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        kind = str(cfg.kind).strip().lower()
        if kind not in {"residual", "mlp"}:
            raise ValueError(f"invalid model kind: {cfg.kind!r} (expected 'residual' or 'mlp')")

        self.kind = kind
        self.act = nn.ReLU()

        if kind == "residual":
            self.inp = nn.Linear(F, cfg.hidden)
            self.norm = nn.LayerNorm(cfg.hidden)
            self.blocks = nn.ModuleList([ResidualBlock(cfg.hidden) for _ in range(cfg.blocks)])
            self.mlp_layers = None
        else:
            # Plain MLP trunk (AlphaYatzy-style). `blocks` is interpreted as number of hidden layers.
            layers: list[nn.Module] = []
            in_dim = F
            n_layers = max(1, int(cfg.blocks))
            for _ in range(n_layers):
                layers.append(nn.Linear(in_dim, cfg.hidden))
                layers.append(nn.ReLU())
                in_dim = cfg.hidden
            self.mlp_layers = nn.Sequential(*layers)
            self.inp = None
            self.norm = None
            self.blocks = None

        self.policy = nn.Linear(cfg.hidden, A)
        self.value = nn.Linear(cfg.hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (policy_logits[B,A], value[B])."""
        if x.ndim != 2 or x.shape[1] != F:
            raise ValueError(f"expected x shape [B,{F}], got {tuple(x.shape)}")
        if self.kind == "residual":
            assert self.inp is not None and self.norm is not None and self.blocks is not None
            h = self.inp(x)
            h = self.norm(h)
            h = self.act(h)
            for b in self.blocks:
                h = b(h)
        else:
            assert self.mlp_layers is not None
            h = self.mlp_layers(x)
        logits = self.policy(h)
        v = torch.tanh(self.value(h).squeeze(-1))
        return logits, v
