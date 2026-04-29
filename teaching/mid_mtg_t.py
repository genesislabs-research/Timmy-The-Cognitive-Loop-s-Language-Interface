"""
mid_mtg_t.py
Mid-MTG: Lemma Stratum at the Kernel Boundary

BIOLOGICAL GROUNDING
====================
This file implements the lemma stratum in the v2 speech pathway.
The biological commitment is that the boundary between the
conceptual stratum and the speech pathway lives in mid-MTG (middle
temporal gyrus, BA21/BA37 anterior portion). This is the cortical
region that holds the lemma representation between conceptual
content and lexical phonological codes (Indefrey and Levelt 2004).

The kernel speaks in concept vectors at the kernel boundary.
mid-MTG receives those concept vectors, projects them through
W_C_to_L, integrates the resulting lemma drive into a persistent
lemma activation a_lemma with gamma_lemma decay, and emits the
selected lemma to Wernicke's for phonological-code retrieval. In
the comprehension direction, perceived lemma activations flow
backward through the same W_C_to_L matrix to reconstruct a concept
vector that the kernel boundary consumes.

This module is the refactored consumer of the LexicalSubstrate
parent. It does not construct its own W_C_to_L; it holds a
reference to the parent and reads through parent.tied_w_c_to_l for
both production and comprehension directions. The persistent
a_lemma buffer remains in this module because lemma activation is
state of the lemma stratum specifically, not part of the lexical
knowledge that is shared across mid-MTG, Wernicke's, and the
acquisition module.

The lateral interference mechanism (Equation 12.1's kappa_interfere
term) is a within-lemma-stratum dynamic that suppresses competing
lemma activations during settling. Without lateral interference,
multiple lemmas can fire simultaneously above threshold, producing
ambiguous production output. With lateral interference, settling
converges on a single dominant lemma per concept. The mechanism
runs entirely in this module; it does not touch the parent's
matrices or the acquisition module's status array.

Identity slot routing is the architectural commitment that the
self_lemma and other_lemma slots carry abstract first-person and
second-person semantics rather than being learnable like ordinary
lemmas. Their concept vectors are pre-allocated by the acquisition
module, and the production pathway routes through them when the
conceptual stratum's content is "self" or "other" respectively.
This module exposes a helper to read identity-slot indices but does
not implement the routing logic itself; the runtime's surface
production layer applies pronoun substitution at emission time.

Primary grounding papers:

Indefrey P, Levelt WJM (2004). "The spatial and temporal signatures
of word production components." Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001.

Hickok G, Poeppel D (2007). "The cortical organization of speech
processing." Nature Reviews Neuroscience, 8(5), 393-402.
DOI: 10.1038/nrn2113.

Schwartz MF, Kimberg DY, Walker GM, Faseyitan O, Brecher A, Dell GS,
Coslett HB (2009). "Anterior temporal involvement in semantic word
retrieval: voxel-based lesion-symptom mapping evidence from
aphasia." Brain, 132(12), 3411-3427.
DOI: 10.1093/brain/awp284. (Lesion evidence localizing the lemma
stratum to anterior temporal cortex including mid-MTG.)

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from substrate.lexical_substrate_t import LexicalSubstrate
from substrate.lemma_slots_t import (
    IDENTITY_LEMMA_SLOTS,
    SLOT_OTHER_LEMMA,
    SLOT_SELF_LEMMA,
)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class MidMTGConfig:
    """Configuration for the mid-MTG lemma stratum.

    Master flag is first per the Genesis Labs ablation flag standard.
    NOT a biological quantity.

    Attributes:
        enable_mid_mtg: master flag. False produces a module that
            returns zero activations and does not update internal
            state.
        enable_persistence: when False, a_lemma is reset between
            forward calls instead of decaying with gamma_lemma.
            Used for stateless tests.
        enable_lateral_interference: when False, the kappa_interfere
            term in Equation 12.1 is dropped. Used to verify that
            settling depends on the interference mechanism for
            single-lemma convergence.
        gamma_lemma: decay rate for the persistent lemma activation.
            Typical value 0.95 produces a settling timescale of
            roughly t_lemma_steps cycles. NOT a tightly-derived
            biological quantity; tune against settling-curve data.
        kappa_interfere: lateral interference scaling. Typical value
            0.1. NOT a biological quantity in this exact form.
        t_lemma_steps: nominal number of cycles for lemma settling.
            Used in diagnostics; the actual settling depends on
            gamma_lemma and the input drive.
    """

    enable_mid_mtg: bool = True
    enable_persistence: bool = True
    enable_lateral_interference: bool = True
    gamma_lemma: float = 0.95
    kappa_interfere: float = 0.1
    t_lemma_steps: int = 50


# =========================================================================
# MidMTG
# =========================================================================

class MidMTG(nn.Module):
    """Lemma stratum in mid-MTG, refactored to consume the parent
    substrate.

    BIOLOGICAL STRUCTURE: Anterior portion of the middle temporal
    gyrus and adjacent inferior temporal cortex. The lemma stratum
    that mediates between abstract conceptual content (kernel
    boundary, upstream conceptual cortex) and lexical phonological
    codes (Wernicke's area).

    BIOLOGICAL FUNCTION: Receives concept distributions from the
    kernel, drives lemma activations through the W_C_to_L
    projection, integrates lemma activations over time with decay
    and lateral interference, and exposes the selected lemma to
    Wernicke's for phonological-code retrieval. In the comprehension
    direction, drives concept reconstructions back to the kernel
    boundary through the transposed W_C_to_L.

    Reference: Equation 12.1 of the Broca's corpus. Architect's
    Phase 0 gap document, resolution 3 of the matrix-ownership
    question.

    ANATOMICAL INTERFACE (production input):
        Sending structure: Conceptual stratum at the kernel
            boundary (anterior temporal cortex and prefrontal
            cortex for high-level semantic content).
        Receiving structure: Lemma stratum in mid-MTG (this
            module).
        Connection: Concept-to-lemma projection W_C_to_L (held by
            the LexicalSubstrate parent).

    ANATOMICAL INTERFACE (production output):
        Sending structure: Lemma stratum in mid-MTG (this module).
        Receiving structure: Lexical phonological code store in
            Wernicke's (left posterior STG and pMTG).
        Connection: Lemma-to-phonological-code projection W_L_to_P
            (held by the parent, accessed by Wernicke's directly).

    ANATOMICAL INTERFACE (comprehension):
        Sending structure: Wernicke's lexical store.
        Receiving structure: Lemma stratum in mid-MTG (drive into
            a_lemma through the transposed W_L_to_P, then through
            this module's settling dynamics).
        Connection: same W_L_to_P matrix accessed in reverse
            direction.

    STATE: Persistent a_lemma activation buffer, decays at
    gamma_lemma per cycle. Serializes through this module's
    state_dict; the matrix lives in the parent's state_dict.
    """

    def __init__(
        self,
        cfg: MidMTGConfig,
        substrate: LexicalSubstrate,
    ) -> None:
        """Initialize mid-MTG against a parent substrate.

        Args:
            cfg: MidMTGConfig.
            substrate: a LexicalSubstrate instance whose
                tied_w_c_to_l wrapper this module reads through for
                both directions.
        """
        super().__init__()
        self.cfg = cfg
        self.substrate = substrate
        self.n_concepts = substrate.cfg.n_concepts
        self.n_lemmas = substrate.cfg.n_lemmas

        # Persistent lemma activation buffer. Shape (1, n_lemmas) so
        # the standard batch-dim-first conventions apply on forward
        # calls without unsqueezing in the hot path.
        # Reference: v2 spec Section 9 (state serialization).
        self.register_buffer(
            "a_lemma", torch.zeros(1, self.n_lemmas),
        )

    # =====================================================================
    # Production direction
    # =====================================================================

    def forward_production(self, concept: Tensor) -> Tensor:
        """Drive lemma activations from a concept distribution.

        Implements Equation 12.1 of the corpus:

            a_lemma(t+1) = gamma * a_lemma(t)
                         + (1 - gamma) * drive(t)
                         - kappa * lateral_interference(t)

        where drive(t) = W_C_to_L @ concept(t) read through the
        parent's tied_w_c_to_l.forward_a_to_b. Lateral interference
        is computed within the lemma stratum and suppresses competing
        lemmas.

        The persistent a_lemma updates in-place. Callers can read
        the post-update activations from self.a_lemma or from the
        return value; both are the same tensor.

        Args:
            concept: (B, n_concepts) concept distribution from the
                kernel boundary.

        Returns:
            (B, n_lemmas) post-update lemma activation.
        """
        if not self.cfg.enable_mid_mtg:
            return torch.zeros(
                concept.shape[0], self.n_lemmas,
                device=concept.device, dtype=concept.dtype,
            )

        # Drive through the parent's tied wrapper.
        drive = self.substrate.tied_w_c_to_l.forward_a_to_b(concept)

        # Lateral interference within the lemma stratum.
        if self.cfg.enable_lateral_interference:
            interference = self._lateral_interference(drive)
        else:
            interference = torch.zeros_like(drive)

        # Persistent integration (Equation 12.1).
        if self.cfg.enable_persistence:
            # Make sure the persistent buffer has the right batch dim.
            if self.a_lemma.shape[0] != concept.shape[0]:
                with torch.no_grad():
                    self.a_lemma = torch.zeros(
                        concept.shape[0], self.n_lemmas,
                        device=concept.device, dtype=concept.dtype,
                    )
            with torch.no_grad():
                self.a_lemma.copy_(
                    self.cfg.gamma_lemma * self.a_lemma
                    + (1.0 - self.cfg.gamma_lemma) * drive
                    - self.cfg.kappa_interfere * interference
                )
            return self.a_lemma.clone()

        # Stateless: just return the drive minus interference.
        return drive - self.cfg.kappa_interfere * interference

    def _lateral_interference(self, drive: Tensor) -> Tensor:
        """Compute lateral interference from competing lemma drives.

        The interference at each lemma is the sum of drives at all
        other lemmas above zero. This implements winner-take-most
        dynamics: the highest-driven lemma wins, others are
        suppressed in proportion to their drive.

        NOT a biological quantity in this exact form. Biology
        realizes lateral interference through inhibitory
        interneurons whose activation is roughly the average
        excitatory drive in a local region; this implementation is
        a tractable approximation.

        Args:
            drive: (B, n_lemmas) lemma drive from the projection.

        Returns:
            (B, n_lemmas) interference contribution to subtract.
        """
        # Clamp to non-negative drive before summing.
        positive = drive.clamp(min=0.0)
        # Total drive across all lemmas, per batch element.
        total = positive.sum(dim=1, keepdim=True)
        # Each lemma's interference is the total minus its own
        # contribution. This produces an interference signal
        # proportional to "how much other lemmas are active."
        return total - positive

    # =====================================================================
    # Comprehension direction
    # =====================================================================

    def forward_comprehension(
        self, lemma_activation: Tensor,
    ) -> Tensor:
        """Reconstruct a concept distribution from a lemma activation.

        The comprehension direction reads through tied_w_c_to_l in
        the reverse direction (forward_b_to_a), producing a concept
        vector that the kernel boundary consumes. This is how
        perceived speech reaches the kernel: Wernicke's drives lemma
        activations from perceived phonological codes, those
        activations flow into mid-MTG's a_lemma through the lemma
        stratum's normal dynamics, and the resulting activation is
        projected back to a concept distribution through this method.

        Args:
            lemma_activation: (B, n_lemmas) lemma activation. May
                be the persistent a_lemma or an externally-supplied
                activation.

        Returns:
            (B, n_concepts) reconstructed concept distribution.
        """
        if not self.cfg.enable_mid_mtg:
            return torch.zeros(
                lemma_activation.shape[0], self.n_concepts,
                device=lemma_activation.device,
                dtype=lemma_activation.dtype,
            )
        return self.substrate.tied_w_c_to_l.forward_b_to_a(
            lemma_activation,
        )

    # =====================================================================
    # Identity slot helpers
    # =====================================================================

    def identity_slot(self, role: str) -> int:
        """Return the slot index for a named identity role.

        Args:
            role: "self_lemma" or "other_lemma".

        Returns:
            slot index from the unified slot inventory.

        Raises:
            KeyError: if role is not a recognized identity name.
        """
        if role not in IDENTITY_LEMMA_SLOTS:
            raise KeyError(
                f"Unknown identity role '{role}'. "
                f"Expected one of {sorted(IDENTITY_LEMMA_SLOTS.keys())}."
            )
        return IDENTITY_LEMMA_SLOTS[role]

    def is_identity_active(
        self, threshold: float = 0.5,
    ) -> Optional[str]:
        """Return the name of the identity slot whose activation is
        above threshold, or None if neither is.

        Used by the runtime's surface production layer to apply
        pronoun substitution at emission time.

        Args:
            threshold: minimum activation magnitude to consider an
                identity slot active. NOT a biological quantity,
                training artifact only.

        Returns:
            "self_lemma", "other_lemma", or None.
        """
        with torch.no_grad():
            self_act = float(
                self.a_lemma[0, SLOT_SELF_LEMMA].abs().item()
            )
            other_act = float(
                self.a_lemma[0, SLOT_OTHER_LEMMA].abs().item()
            )
            if self_act < threshold and other_act < threshold:
                return None
            return (
                "self_lemma" if self_act >= other_act
                else "other_lemma"
            )

    # =====================================================================
    # State management
    # =====================================================================

    def reset_state(self) -> None:
        """Clear the persistent lemma activation buffer.

        Called between unrelated dialogue sessions and when the
        runtime needs to ensure the substrate enters production with
        no carryover from a previous turn.
        """
        with torch.no_grad():
            self.a_lemma.zero_()

    def get_diagnostic_state(self) -> Dict[str, float]:
        """Return a dict of internal norms and counters."""
        with torch.no_grad():
            return {
                "a_lemma_norm": float(self.a_lemma.norm().item()),
                "a_lemma_max": float(self.a_lemma.max().item()),
                "a_lemma_argmax": int(
                    self.a_lemma.argmax(dim=1)[0].item()
                ),
                "n_above_half": int(
                    (self.a_lemma.abs() > 0.5).sum().item()
                ),
            }
