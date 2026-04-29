from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class TiedSubstrateConfig:
    enable_tied_substrate: bool = True
    in_dim: int = 64
    out_dim: int = 64
    init_scale: float = 0.02


class TiedSubstrate(nn.Module):

    def __init__(self, cfg: TiedSubstrateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.W = nn.Parameter(
            torch.randn(cfg.out_dim, cfg.in_dim) * cfg.init_scale
        )

    def forward_a_to_b(
        self,
        x_a: Tensor,
        receptor_modulation: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.cfg.enable_tied_substrate:
            return torch.zeros(
                x_a.shape[0], self.cfg.out_dim,
                device=x_a.device, dtype=x_a.dtype,
            )
        if receptor_modulation is None:
            effective_W = self.W
        else:
            effective_W = self.W * receptor_modulation
        return torch.matmul(x_a, effective_W.t())

    def forward_b_to_a(
        self,
        x_b: Tensor,
        receptor_modulation: Optional[Tensor] = None,
    ) -> Tensor:
        if not self.cfg.enable_tied_substrate:
            return torch.zeros(
                x_b.shape[0], self.cfg.in_dim,
                device=x_b.device, dtype=x_b.dtype,
            )
        if receptor_modulation is None:
            effective_W = self.W
        else:
            effective_W = self.W * receptor_modulation
        return torch.matmul(x_b, effective_W)

    def serialize(self) -> dict:
        return {"W": self.W.detach().cpu().clone()}

    def restore(self, state: dict) -> None:
        saved_W = state["W"]
        if saved_W.shape != self.W.shape:
            raise ValueError(
                f"TiedSubstrate restore: shape mismatch. Saved {saved_W.shape}, "
                f"current {tuple(self.W.shape)}. Architecture has changed."
            )
        self.W.data = saved_W.to(self.W.device)
