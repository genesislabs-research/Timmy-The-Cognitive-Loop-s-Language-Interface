from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import torch
import torch.nn as nn
from torch import Tensor


@dataclass
class ArcuateConfig:
    enable_arcuate: bool = True
    n_segments: int = 64
    tau_arc_steps: int = 2
    identity_jitter: float = 0.0


class Arcuate(nn.Module):

    def __init__(self, cfg: ArcuateConfig) -> None:
        super().__init__()
        self.cfg = cfg
        w_init = torch.eye(cfg.n_segments)
        if cfg.identity_jitter > 0:
            w_init = w_init + torch.randn_like(w_init) * cfg.identity_jitter
        self.register_buffer("W_arc", w_init)
        self._delay_buffer: Deque[Tensor] = deque(
            [torch.zeros(1, cfg.n_segments) for _ in range(cfg.tau_arc_steps)],
            maxlen=cfg.tau_arc_steps,
        )

    def forward(self, segment: Tensor) -> Tensor:
        if not self.cfg.enable_arcuate:
            return torch.zeros(
                segment.shape[0], self.cfg.n_segments,
                device=segment.device, dtype=segment.dtype,
            )
        if self._delay_buffer[0].shape[0] != segment.shape[0]:
            B = segment.shape[0]
            self._delay_buffer = deque(
                [
                    torch.zeros(
                        B, self.cfg.n_segments,
                        device=segment.device, dtype=segment.dtype,
                    )
                    for _ in range(self.cfg.tau_arc_steps)
                ],
                maxlen=self.cfg.tau_arc_steps,
            )
        emerging = self._delay_buffer[0]
        self._delay_buffer.popleft()
        self._delay_buffer.append(segment)
        return torch.matmul(emerging, self.W_arc.t())

    def reset_state(self) -> None:
        for slot in self._delay_buffer:
            slot.zero_()

    def serialize(self) -> dict:
        return {
            "cold": {
                "W_arc": self.W_arc.detach().cpu().clone(),
            },
            "warm": {
                "delay_buffer": [
                    slot.detach().cpu().clone()
                    for slot in self._delay_buffer
                ],
            },
        }

    def restore(self, state: dict) -> None:
        cold = state["cold"]
        saved_w = cold["W_arc"].to(self.W_arc.device)
        if saved_w.shape != self.W_arc.shape:
            raise ValueError(
                f"Arcuate restore: W_arc shape mismatch. "
                f"Saved {saved_w.shape}, current {tuple(self.W_arc.shape)}."
            )
        self.W_arc = saved_w
        warm = state["warm"]
        saved_buffer = warm["delay_buffer"]
        if len(saved_buffer) != self.cfg.tau_arc_steps:
            raise ValueError(
                f"Arcuate restore: buffer length mismatch. "
                f"Saved {len(saved_buffer)}, current {self.cfg.tau_arc_steps}."
            )
        self._delay_buffer = deque(
            [slot.to(self.W_arc.device) for slot in saved_buffer],
            maxlen=self.cfg.tau_arc_steps,
        )
