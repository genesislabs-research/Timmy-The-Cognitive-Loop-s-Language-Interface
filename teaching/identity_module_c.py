from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


DEFAULT_PRONOUN_ROUTING: Dict[str, str] = {
    "you": "SELF",
    "your": "SELF",
    "yours": "SELF",
    "yourself": "SELF",
    "i": "OTHER",
    "me": "OTHER",
    "my": "OTHER",
    "mine": "OTHER",
    "myself": "OTHER",
}


@dataclass
class IdentityModuleConfig:
    enable_identity_module: bool = True
    enable_self_episode_tagging: bool = True
    pronoun_routing: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_PRONOUN_ROUTING)
    )
    identity_boost_magnitude: float = 5.0
    self_activation_threshold: float = 1.0


class IdentityModule:

    def __init__(self, cfg: IdentityModuleConfig) -> None:
        self.cfg = cfg
        self._routing: Dict[str, str] = {
            k.lower(): v for k, v in cfg.pronoun_routing.items()
        }

    def is_pronoun(self, token: str) -> bool:
        if not self.cfg.enable_identity_module:
            return False
        return token.lower() in self._routing

    def route_pronoun(
        self,
        token: str,
        midmtg_region: Any,
    ) -> Optional[str]:
        if not self.cfg.enable_identity_module:
            return None
        normalized = token.lower()
        if normalized not in self._routing:
            return None
        target = self._routing[normalized]
        if target == "SELF":
            slot_name = "self_lemma"
        elif target == "OTHER":
            slot_name = "other_lemma"
        else:
            return None
        slot_idx = midmtg_region.identity_slot(slot_name)
        with torch.no_grad():
            midmtg_region.a_lemma[:, slot_idx] += (
                self.cfg.identity_boost_magnitude
            )
        return target

    def route_perceived_phrase(
        self,
        tokens: List[str],
        midmtg_region: Any,
    ) -> List[Tuple[str, str]]:
        if not self.cfg.enable_identity_module:
            return []
        routed: List[Tuple[str, str]] = []
        for tok in tokens:
            target = self.route_pronoun(tok, midmtg_region)
            if target is not None:
                routed.append((tok, target))
        return routed

    def check_self_active(self, midmtg_region: Any) -> bool:
        if not self.cfg.enable_identity_module:
            return False
        if not self.cfg.enable_self_episode_tagging:
            return False
        slot_idx = midmtg_region.identity_slot("self_lemma")
        activation = midmtg_region.a_lemma[:, slot_idx].mean().item()
        return activation >= self.cfg.self_activation_threshold

    def tag_episode(
        self,
        episode_metadata: Dict[str, Any],
        midmtg_region: Any,
    ) -> Dict[str, Any]:
        episode_metadata["self_relevant"] = self.check_self_active(midmtg_region)
        return episode_metadata

    def get_production_pronoun(
        self,
        midmtg_region: Any,
        register: str = "informal",
    ) -> Optional[str]:
        if not self.cfg.enable_identity_module:
            return None
        self_idx = midmtg_region.identity_slot("self_lemma")
        other_idx = midmtg_region.identity_slot("other_lemma")
        self_activation = midmtg_region.a_lemma[:, self_idx].mean().item()
        other_activation = midmtg_region.a_lemma[:, other_idx].mean().item()
        if (
            self_activation < self.cfg.self_activation_threshold
            and other_activation < self.cfg.self_activation_threshold
        ):
            return None
        if self_activation >= other_activation:
            return "i"
        else:
            return "you"

    def serialize(self) -> Dict[str, Any]:
        return {
            "pronoun_routing": dict(self._routing),
            "identity_boost_magnitude": self.cfg.identity_boost_magnitude,
            "self_activation_threshold": self.cfg.self_activation_threshold,
            "enable_identity_module": self.cfg.enable_identity_module,
            "enable_self_episode_tagging": (
                self.cfg.enable_self_episode_tagging
            ),
        }

    def restore(self, state: Dict[str, Any]) -> None:
        if "pronoun_routing" in state:
            self._routing = {
                k.lower(): v for k, v in state["pronoun_routing"].items()
            }
        if "identity_boost_magnitude" in state:
            self.cfg.identity_boost_magnitude = float(
                state["identity_boost_magnitude"]
            )
        if "self_activation_threshold" in state:
            self.cfg.self_activation_threshold = float(
                state["self_activation_threshold"]
            )
        if "enable_identity_module" in state:
            self.cfg.enable_identity_module = bool(
                state["enable_identity_module"]
            )
        if "enable_self_episode_tagging" in state:
            self.cfg.enable_self_episode_tagging = bool(
                state["enable_self_episode_tagging"]
            )
