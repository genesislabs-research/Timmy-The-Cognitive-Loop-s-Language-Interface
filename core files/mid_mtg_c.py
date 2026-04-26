from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from substrate.tied_substrate_c import TiedSubstrate, TiedSubstrateConfig


IDENTITY_LEMMA_SLOTS: Dict[str, int] = {
    "self_lemma": 0,
    "other_lemma": 1,
}

UNCERTAINTY_LEMMA_SLOTS: Dict[str, int] = {
    "i_dont_know": 2,
    "im_not_sure": 3,
    "maybe": 4,
    "i_think": 5,
    "probably": 6,
    "i_dont_remember": 7,
    "no_idea": 8,
    "cannot_say": 9,
}

QUESTION_LEMMA_SLOTS: Dict[str, int] = {
    "what": 10,
    "who": 11,
    "where": 12,
    "when": 13,
    "why": 14,
    "how": 15,
}

N_RESERVED_LEMMAS: int = (
    len(IDENTITY_LEMMA_SLOTS)
    + len(UNCERTAINTY_LEMMA_SLOTS)
    + len(QUESTION_LEMMA_SLOTS)
)


@dataclass
class MidMTGConfig:
    enable_mid_mtg: bool = True
    enable_persistence: bool = True
    enable_lateral_interference: bool = True
    enable_identity_routing: bool = True
    enable_uncertainty_lemmas: bool = True
    enable_question_lemmas: bool = True
    n_concepts: int = 1024
    n_lemmas: int = 512
    gamma_lemma: float = 0.95
    kappa_interfere: float = 0.1
    t_lemma_steps: int = 50
    confidence_floor: float = 1e-6


class MidMTG(nn.Module):

    def __init__(self, cfg: MidMTGConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.w_c_to_l = TiedSubstrate(TiedSubstrateConfig(
            in_dim=cfg.n_concepts,
            out_dim=cfg.n_lemmas,
        ))
        self.register_buffer(
            "a_lemma", torch.zeros(1, cfg.n_lemmas),
        )
        self.register_buffer(
            "_concept_embedding_for_sim",
            torch.randn(cfg.n_concepts, 32) * 0.1,
        )
        self.register_buffer(
            "_last_concept", torch.zeros(1, cfg.n_concepts),
        )
        self.register_buffer(
            "_steps_since_reset", torch.tensor(0, dtype=torch.long),
        )
        is_allocated = torch.zeros(cfg.n_lemmas, dtype=torch.bool)
        if cfg.enable_identity_routing:
            for slot_idx in IDENTITY_LEMMA_SLOTS.values():
                is_allocated[slot_idx] = True
        if cfg.enable_uncertainty_lemmas:
            for slot_idx in UNCERTAINTY_LEMMA_SLOTS.values():
                is_allocated[slot_idx] = True
        if cfg.enable_question_lemmas:
            for slot_idx in QUESTION_LEMMA_SLOTS.values():
                is_allocated[slot_idx] = True
        self.register_buffer("is_allocated", is_allocated)

    def identity_slot(self, name: str) -> int:
        if name not in IDENTITY_LEMMA_SLOTS:
            raise KeyError(
                f"Unknown identity slot '{name}'. "
                f"Expected one of {sorted(IDENTITY_LEMMA_SLOTS.keys())}."
            )
        return IDENTITY_LEMMA_SLOTS[name]

    def uncertainty_slot(self, name: str) -> int:
        if name not in UNCERTAINTY_LEMMA_SLOTS:
            raise KeyError(
                f"Unknown uncertainty slot '{name}'. "
                f"Expected one of {sorted(UNCERTAINTY_LEMMA_SLOTS.keys())}."
            )
        return UNCERTAINTY_LEMMA_SLOTS[name]

    def question_slot(self, name: str) -> int:
        if name not in QUESTION_LEMMA_SLOTS:
            raise KeyError(
                f"Unknown question slot '{name}'. "
                f"Expected one of {sorted(QUESTION_LEMMA_SLOTS.keys())}."
            )
        return QUESTION_LEMMA_SLOTS[name]

    def get_question_lemma_slots(self) -> Dict[str, int]:
        return dict(QUESTION_LEMMA_SLOTS)

    def forward_production(self, c_lex: Tensor) -> Tensor:
        if not self.cfg.enable_mid_mtg:
            return torch.zeros(
                c_lex.shape[0], self.cfg.n_lemmas,
                device=c_lex.device, dtype=c_lex.dtype,
            )
        B = c_lex.shape[0]
        if self.a_lemma.shape[0] != B:
            self.a_lemma = torch.zeros(
                B, self.cfg.n_lemmas,
                device=c_lex.device, dtype=c_lex.dtype,
            )
        if self.cfg.enable_persistence:
            decayed = self.a_lemma * self.cfg.gamma_lemma
        else:
            decayed = torch.zeros_like(self.a_lemma)
        new_input = self.w_c_to_l.forward_a_to_b(c_lex)
        a_new = decayed + new_input
        if self.cfg.enable_lateral_interference:
            interference = self._compute_lateral_interference(a_new, c_lex)
            a_new = a_new + interference
        self.a_lemma = a_new
        self._last_concept = c_lex.detach().clone()
        self._steps_since_reset = self._steps_since_reset + 1
        return self.a_lemma

    def _compute_lateral_interference(
        self,
        a: Tensor,
        c_lex: Tensor,
    ) -> Tensor:
        lemma_concept_signatures = self.w_c_to_l.W
        sim_space = lemma_concept_signatures @ self._concept_embedding_for_sim
        sim_norm = torch.nn.functional.normalize(sim_space, dim=1)
        sim_matrix = sim_norm @ sim_norm.t()
        n_lemmas = sim_matrix.shape[0]
        sim_matrix = sim_matrix - torch.eye(
            n_lemmas, device=sim_matrix.device, dtype=sim_matrix.dtype,
        )
        interference_strength = a @ sim_matrix.t()
        return -self.cfg.kappa_interfere * interference_strength

    def forward_comprehension(self, a_lemma_input: Tensor) -> Tensor:
        if not self.cfg.enable_mid_mtg:
            return torch.zeros(
                a_lemma_input.shape[0], self.cfg.n_concepts,
                device=a_lemma_input.device, dtype=a_lemma_input.dtype,
            )
        return self.w_c_to_l.forward_b_to_a(a_lemma_input)

    def select_lemma(self) -> Optional[Tensor]:
        if not self.cfg.enable_mid_mtg:
            return None
        if self._steps_since_reset.item() < self.cfg.t_lemma_steps:
            return None
        winner = self.a_lemma.argmax(dim=1)
        one_hot = torch.zeros_like(self.a_lemma)
        one_hot.scatter_(1, winner.unsqueeze(1), 1.0)
        return one_hot

    def get_lemma_confidence(self) -> Tensor:
        if not self.cfg.enable_mid_mtg:
            return torch.zeros(self.a_lemma.shape[0])
        abs_a = self.a_lemma.abs()
        peak_idx = abs_a.argmax(dim=1)
        peak_allocated = self.is_allocated[peak_idx]
        n_allocated = self.is_allocated.sum().item()
        if n_allocated == 0:
            return torch.zeros(
                abs_a.shape[0], device=abs_a.device, dtype=abs_a.dtype,
            )
        allocated_mask = self.is_allocated.unsqueeze(0).to(abs_a.dtype)
        allocated_sum = (abs_a * allocated_mask).sum(dim=1)
        avg_allocated = (allocated_sum / n_allocated).clamp(
            min=self.cfg.confidence_floor
        )
        masked_a = abs_a.masked_fill(~self.is_allocated.unsqueeze(0), -1.0)
        peak_allocated_value = masked_a.max(dim=1).values
        peak_allocated_value = peak_allocated_value.clamp(min=0.0)
        ratio = peak_allocated_value / avg_allocated
        target_ratio = 10.0
        confidence = (ratio / target_ratio).clamp(max=1.0)
        confidence = confidence * peak_allocated.to(confidence.dtype)
        return confidence

    def allocate_lemma(self, lemma_idx: int) -> None:
        if not (0 <= lemma_idx < self.cfg.n_lemmas):
            raise IndexError(
                f"Lemma index {lemma_idx} out of range "
                f"[0, {self.cfg.n_lemmas})."
            )
        self.is_allocated[lemma_idx] = True

    def reset_state(self) -> None:
        self.a_lemma.zero_()
        self._last_concept.zero_()
        self._steps_since_reset.zero_()

    def reset_for_selection(self) -> None:
        self._steps_since_reset.zero_()

    def serialize(self) -> dict:
        return {
            "cold": {
                "w_c_to_l": self.w_c_to_l.serialize(),
                "concept_embedding_for_sim": (
                    self._concept_embedding_for_sim.detach().cpu().clone()
                ),
                "is_allocated": self.is_allocated.detach().cpu().clone(),
            },
            "warm": {
                "a_lemma": self.a_lemma.detach().cpu().clone(),
                "last_concept": self._last_concept.detach().cpu().clone(),
                "steps_since_reset": self._steps_since_reset.item(),
            },
        }

    def restore(self, state: dict) -> None:
        cold = state["cold"]
        self.w_c_to_l.restore(cold["w_c_to_l"])
        sim_emb = cold["concept_embedding_for_sim"].to(
            self._concept_embedding_for_sim.device
        )
        if sim_emb.shape != self._concept_embedding_for_sim.shape:
            raise ValueError(
                f"MidMTG restore: concept_embedding_for_sim shape mismatch. "
                f"Saved {sim_emb.shape}, current "
                f"{tuple(self._concept_embedding_for_sim.shape)}."
            )
        self._concept_embedding_for_sim = sim_emb
        if "is_allocated" in cold:
            saved_alloc = cold["is_allocated"].to(self.is_allocated.device)
            if saved_alloc.shape != self.is_allocated.shape:
                raise ValueError(
                    f"MidMTG restore: is_allocated shape mismatch. "
                    f"Saved {saved_alloc.shape}, current "
                    f"{tuple(self.is_allocated.shape)}."
                )
            self.is_allocated = saved_alloc
        warm = state["warm"]
        a_lemma = warm["a_lemma"].to(self.a_lemma.device)
        last_concept = warm["last_concept"].to(self._last_concept.device)
        if a_lemma.shape[1] != self.cfg.n_lemmas:
            raise ValueError(
                f"MidMTG restore: a_lemma n_lemmas mismatch. "
                f"Saved {a_lemma.shape[1]}, current {self.cfg.n_lemmas}."
            )
        self.a_lemma = a_lemma
        self._last_concept = last_concept
        self._steps_since_reset = torch.tensor(
            warm["steps_since_reset"],
            dtype=torch.long,
            device=self._steps_since_reset.device,
        )
