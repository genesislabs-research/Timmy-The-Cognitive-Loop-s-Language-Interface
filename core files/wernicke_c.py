from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from substrate.tied_substrate_c import TiedSubstrate, TiedSubstrateConfig


SEGMENT_SILENCE: int = 0
SEGMENT_SYLLABLE_BOUNDARY: int = 1
SEGMENT_WORD_END: int = 2
N_RESERVED_SEGMENTS: int = 3


@dataclass
class WernickeConfig:
    enable_wernicke: bool = True
    enable_persistence: bool = True
    enable_spell_out: bool = True
    n_lemmas: int = 512
    d_phon: int = 512
    n_segments: int = 64
    d_decoder_hidden: int = 64
    tau_decay_steps: int = 30
    spell_out_max_steps: int = 32
    confidence_perturbation_scale: float = 0.05
    confidence_floor: float = 1e-6


class Wernicke(nn.Module):

    def __init__(self, cfg: WernickeConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.w_l_to_p = TiedSubstrate(TiedSubstrateConfig(
            in_dim=cfg.n_lemmas,
            out_dim=cfg.d_phon,
        ))
        self.spell_out_input_to_hidden = nn.Linear(
            cfg.d_phon + cfg.n_segments, cfg.d_decoder_hidden,
        )
        self.spell_out_hidden_to_hidden = nn.Linear(
            cfg.d_decoder_hidden, cfg.d_decoder_hidden,
        )
        self.spell_out_hidden_to_segment = nn.Linear(
            cfg.d_decoder_hidden, cfg.n_segments,
        )
        self.register_buffer(
            "a_l_percept", torch.zeros(1, cfg.n_lemmas),
        )
        self.register_buffer(
            "_spell_out_hidden", torch.zeros(1, cfg.d_decoder_hidden),
        )
        self.register_buffer(
            "_last_segment", torch.zeros(1, cfg.n_segments),
        )

    def retrieve_phonological_code(self, l_star: Tensor) -> Tensor:
        if not self.cfg.enable_wernicke:
            return torch.zeros(
                l_star.shape[0], self.cfg.d_phon,
                device=l_star.device, dtype=l_star.dtype,
            )
        return self.w_l_to_p.forward_a_to_b(l_star)

    def reset_spell_out_state(self, batch_size: int) -> None:
        device = self._spell_out_hidden.device
        dtype = self._spell_out_hidden.dtype
        self._spell_out_hidden = torch.zeros(
            batch_size, self.cfg.d_decoder_hidden,
            device=device, dtype=dtype,
        )
        self._last_segment = torch.zeros(
            batch_size, self.cfg.n_segments,
            device=device, dtype=dtype,
        )

    def emit_next_segment(self, phon_code: Tensor) -> Tensor:
        if not self.cfg.enable_wernicke or not self.cfg.enable_spell_out:
            return torch.zeros(
                phon_code.shape[0], self.cfg.n_segments,
                device=phon_code.device, dtype=phon_code.dtype,
            )
        B = phon_code.shape[0]
        if self._spell_out_hidden.shape[0] != B:
            self.reset_spell_out_state(B)
        decoder_input = torch.cat(
            [phon_code, self._last_segment], dim=1,
        )
        h_input = self.spell_out_input_to_hidden(decoder_input)
        h_recurrent = self.spell_out_hidden_to_hidden(self._spell_out_hidden)
        new_hidden = torch.tanh(h_input + h_recurrent)
        segment_logits = self.spell_out_hidden_to_segment(new_hidden)
        self._spell_out_hidden = new_hidden
        argmax_idx = segment_logits.argmax(dim=1)
        new_last_segment = torch.zeros_like(self._last_segment)
        new_last_segment.scatter_(1, argmax_idx.unsqueeze(1), 1.0)
        self._last_segment = new_last_segment
        return segment_logits

    def spell_out_word(
        self,
        phon_code: Tensor,
        max_steps: Optional[int] = None,
    ) -> Tensor:
        if max_steps is None:
            max_steps = self.cfg.spell_out_max_steps
        B = phon_code.shape[0]
        self.reset_spell_out_state(B)
        emitted_segments = []
        word_ended = torch.zeros(B, dtype=torch.bool, device=phon_code.device)
        for _ in range(max_steps):
            logits = self.emit_next_segment(phon_code)
            argmax_idx = logits.argmax(dim=1)
            emitted_segments.append(argmax_idx)
            word_ended = word_ended | (argmax_idx == SEGMENT_WORD_END)
            if word_ended.all():
                break
        return torch.stack(emitted_segments, dim=1)

    def perceive_phonological_code(self, phi_input: Tensor) -> Tensor:
        if not self.cfg.enable_wernicke:
            return torch.zeros(
                phi_input.shape[0], self.cfg.n_lemmas,
                device=phi_input.device, dtype=phi_input.dtype,
            )
        B = phi_input.shape[0]
        if self.a_l_percept.shape[0] != B:
            self.a_l_percept = torch.zeros(
                B, self.cfg.n_lemmas,
                device=phi_input.device, dtype=phi_input.dtype,
            )
        if self.cfg.enable_persistence:
            decay_factor = torch.exp(
                torch.tensor(-1.0 / self.cfg.tau_decay_steps)
            )
            decayed = self.a_l_percept * decay_factor
        else:
            decayed = torch.zeros_like(self.a_l_percept)
        new_input = self.w_l_to_p.forward_b_to_a(phi_input)
        self.a_l_percept = decayed + new_input
        return self.a_l_percept

    def get_phonological_confidence(self, l_star: Tensor) -> Tensor:
        if not self.cfg.enable_wernicke:
            return torch.zeros(l_star.shape[0])
        with torch.no_grad():
            phi_ref = self.retrieve_phonological_code(l_star)
            n_perturbations = 8
            perturbed_codes = []
            for _ in range(n_perturbations):
                noise = torch.randn_like(l_star) * (
                    self.cfg.confidence_perturbation_scale
                )
                phi_perturbed = self.retrieve_phonological_code(l_star + noise)
                perturbed_codes.append(phi_perturbed)
            stacked = torch.stack(perturbed_codes, dim=0)
            code_variance = stacked.var(dim=0).sum(dim=1)
            ref_norm_sq = (phi_ref ** 2).sum(dim=1).clamp(
                min=self.cfg.confidence_floor
            )
            relative_instability = code_variance / ref_norm_sq
            confidence = torch.exp(-relative_instability)
        return confidence

    def reset_state(self) -> None:
        self.a_l_percept.zero_()
        self._spell_out_hidden.zero_()
        self._last_segment.zero_()

    def serialize(self) -> dict:
        return {
            "cold": {
                "w_l_to_p": self.w_l_to_p.serialize(),
                "decoder_input_to_hidden": (
                    self.spell_out_input_to_hidden.state_dict()
                ),
                "decoder_hidden_to_hidden": (
                    self.spell_out_hidden_to_hidden.state_dict()
                ),
                "decoder_hidden_to_segment": (
                    self.spell_out_hidden_to_segment.state_dict()
                ),
            },
            "warm": {
                "a_l_percept": self.a_l_percept.detach().cpu().clone(),
                "spell_out_hidden": (
                    self._spell_out_hidden.detach().cpu().clone()
                ),
                "last_segment": self._last_segment.detach().cpu().clone(),
            },
        }

    def restore(self, state: dict) -> None:
        cold = state["cold"]
        self.w_l_to_p.restore(cold["w_l_to_p"])
        self.spell_out_input_to_hidden.load_state_dict(
            cold["decoder_input_to_hidden"]
        )
        self.spell_out_hidden_to_hidden.load_state_dict(
            cold["decoder_hidden_to_hidden"]
        )
        self.spell_out_hidden_to_segment.load_state_dict(
            cold["decoder_hidden_to_segment"]
        )
        warm = state["warm"]
        a_l_percept = warm["a_l_percept"].to(self.a_l_percept.device)
        if a_l_percept.shape[1] != self.cfg.n_lemmas:
            raise ValueError(
                f"Wernicke restore: a_l_percept n_lemmas mismatch. "
                f"Saved {a_l_percept.shape[1]}, current {self.cfg.n_lemmas}."
            )
        self.a_l_percept = a_l_percept
        self._spell_out_hidden = warm["spell_out_hidden"].to(
            self._spell_out_hidden.device
        )
        self._last_segment = warm["last_segment"].to(self._last_segment.device)
