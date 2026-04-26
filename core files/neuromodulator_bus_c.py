from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import Tensor


NEUTRAL_DEFAULTS: Dict[str, float] = {
    "DA": 0.0,
    "NE": 1.0,
    "ACh_inc": 0.5,
    "ACh_dec": 0.5,
    "5HT": 0.0,
}


@dataclass
class NeuromodulatorBusConfig:
    enable_neuromodulator_bus: bool = True
    device: str = "cpu"


class NeuromodulatorBus:

    def __init__(
        self,
        cfg: NeuromodulatorBusConfig,
        upstream_broadcast: Optional[Any] = None,
    ) -> None:
        self.cfg = cfg
        self.upstream = upstream_broadcast
        device = torch.device(cfg.device)
        self._scalars: Dict[str, Tensor] = {
            key: torch.tensor(value, device=device)
            for key, value in NEUTRAL_DEFAULTS.items()
        }

    def get(self, key: str) -> Tensor:
        if key not in NEUTRAL_DEFAULTS:
            raise KeyError(
                f"Unknown modulator key '{key}'. "
                f"Expected one of {sorted(NEUTRAL_DEFAULTS.keys())}."
            )
        if not self.cfg.enable_neuromodulator_bus:
            return torch.tensor(
                NEUTRAL_DEFAULTS[key],
                device=torch.device(self.cfg.device),
            )
        if self.upstream is not None and hasattr(self.upstream, "get"):
            try:
                return self.upstream.get(key)
            except (KeyError, AttributeError):
                pass
        return self._scalars[key]

    def set(self, key: str, value: Tensor) -> None:
        if key not in NEUTRAL_DEFAULTS:
            raise KeyError(
                f"Unknown modulator key '{key}'. "
                f"Expected one of {sorted(NEUTRAL_DEFAULTS.keys())}."
            )
        device = torch.device(self.cfg.device)
        if not isinstance(value, Tensor):
            value = torch.tensor(value, device=device)
        else:
            value = value.to(device)
        self._scalars[key] = value

    def reset(self) -> None:
        device = torch.device(self.cfg.device)
        for key, value in NEUTRAL_DEFAULTS.items():
            self._scalars[key] = torch.tensor(value, device=device)

    def serialize(self) -> Dict[str, float]:
        return {key: value.item() for key, value in self._scalars.items()}

    def restore(self, state: Dict[str, float]) -> None:
        device = torch.device(self.cfg.device)
        for key in NEUTRAL_DEFAULTS:
            if key in state:
                self._scalars[key] = torch.tensor(state[key], device=device)
            else:
                self._scalars[key] = torch.tensor(
                    NEUTRAL_DEFAULTS[key], device=device,
                )
