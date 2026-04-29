"""
lemma_acquisition_t.py
Lemma Acquisition: Discrete Allocation, Provisional State, and
Confirmation-Gated Reinforcement Through the Shared Lexical Substrate

BIOLOGICAL GROUNDING
====================
This file implements the discrete-event mutation interface for the
v2 speech pathway's lexical knowledge. The biological commitment is
that lexical acquisition in human language is a discrete one-shot
event ("fast mapping" per Carey and Bartlett 1978) rather than a
gradient descent on a vocabulary distribution, that newly-acquired
bindings begin in a provisional state that does not consolidate
without confirmation (the synaptic-tagging-and-capture mechanism of
Frey and Morris 1997), and that confirmation is mediated by a phasic
dopamine event coincident with the active lemma (Schultz 1998).

This module is the refactored version under resolution 3 of the
matrix-ownership question raised in the Phase 0 gap document. The
W_C_to_L and W_L_to_P matrices are owned by the LexicalSubstrate
parent and accessed through dependency injection rather than
constructed independently in this module. The architectural
invariant the refactor enforces is that there is exactly one set of
matrix tensors in the speech substrate at any time, and every
consumer of those tensors (mid-MTG forward pass, Wernicke's forward
pass, this module's discrete-event mutations) sees the same state
on the very next forward call after any mutation.

The status array and allocation_time array remain in this module
because they are bookkeeping for the discrete-event mutation
machinery rather than part of the lexical knowledge itself. The
serialization protocol captures both the parent's matrices (through
the parent's state_dict) and this module's status and timing buffers
(through this module's state_dict), and the .soul checkpoint bundles
both so a restored substrate has the same allocations in the same
states with the same timing as the substrate that went to sleep.

The pre_allocate_reserved_slots method is the construction-time hook
that fills the seventeen reserved slot indices with their concept
vectors and phonological codes. Identity slots get empty
phonological codes because their bindings are acquired during the
cold-start dialogue (TIMMY for self_lemma). Wh-words and uncertainty
markers get phonological codes derived from their canonical text via
the supplied phonological-code function. The status array marks all
reserved slots as STATUS_CONFIRMED so they fire reliably from
cold-start without going through the provisional-to-confirmed
transition.

Primary grounding papers:

Carey S, Bartlett E (1978). "Acquiring a single new word."
Proceedings of the Stanford Child Language Conference, 15, 17-29.
DOI: {Conference proceedings, no DOI assigned.}

Frey U, Morris RGM (1997). "Synaptic tagging and long-term
potentiation." Nature, 385(6616), 533-536. DOI: 10.1038/385533a0.

Schultz W (1998). "Predictive reward signal of dopamine neurons."
Journal of Neurophysiology, 80(1), 1-27.
DOI: 10.1152/jn.1998.80.1.1.

Indefrey P, Levelt WJM (2004). "The spatial and temporal signatures
of word production components." Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001.

Florian RV (2007). "Reinforcement learning through modulation of
spike-timing-dependent synaptic plasticity." Neural Computation,
19(6), 1468-1502. DOI: 10.1162/neco.2007.19.6.1468.

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from substrate.lexical_substrate_t import LexicalSubstrate
from substrate.lemma_slots_t import (
    PHONOLOGICAL_TEXT_BY_NAME,
    RESERVED_SLOTS,
    SLOT_BY_NAME,
    SLOT_I_DONT_KNOW,
    SLOT_POLAR_QUESTION,
    STATUS_CONFIRMED,
    STATUS_PROVISIONAL,
    STATUS_UNALLOCATED,
)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class LemmaAcquisitionConfig:
    """Configuration for the lemma acquisition module.

    Master flag is first per the Genesis Labs ablation flag standard.
    NOT a biological quantity.

    Attributes:
        enable_lemma_acquisition: master flag. False produces a
            module whose mutation methods are no-ops and whose
            production selection always returns the i_dont_know
            slot.
        enable_provisional_state: when False, allocations transition
            directly to confirmed status, skipping the
            provisional-to-confirmed gating. Used for tests of the
            non-conversational acquisition paths.
        enable_polar_question_coactivation: when False, provisional
            lemmas do not fire the polar-question marker during
            production. Used to test that the substrate would emit
            assertions on provisional bindings if the marker were
            absent.
        enable_decay_timeout: when False, provisional rows persist
            indefinitely without confirmation. Used for tests that
            run longer than the timeout.
        enable_hebbian_reinforcement: when False, confirmed lemmas
            do not get Hebbian-reinforced on subsequent exposures.
        theta_novelty: cosine-similarity threshold below which a
            perceived phonological code is treated as novel. NOT a
            biological quantity, training artifact only. The
            substrate's actual novelty threshold is determined by
            the statistics of the phonological-code embedding space
            at runtime; 0.65 is a starting value that works for the
            substrate's default dimensionality.
        theta_production: minimum lemma activation below which the
            production pathway falls through to i_dont_know. NOT a
            biological quantity, training artifact only.
        theta_da: dopamine threshold below which a DA event does not
            fire the provisional-to-confirmed transition. The
            confirmation_detector module fires DA at amplitudes
            between 0.4 and 0.95; theta_da of 0.3 admits all of
            these as valid confirmations.
        theta_a: lemma-activation threshold for the confirmation
            transition. The lemma must be active above this level
            at the moment the DA event fires. NOT a biological
            quantity, training artifact only.
        timeout_seconds: decay-without-confirmation interval.
            Provisional rows that have not been confirmed within
            this window revert to STATUS_UNALLOCATED. The 180-second
            default is loosely motivated by working-memory
            consolidation literature; tune against multi-turn
            dialogue test data.
        learning_rate: Hebbian reinforcement step size. NOT a
            biological quantity, training artifact only.
    """

    enable_lemma_acquisition: bool = True
    enable_provisional_state: bool = True
    enable_polar_question_coactivation: bool = True
    enable_decay_timeout: bool = True
    enable_hebbian_reinforcement: bool = True
    theta_novelty: float = 0.65
    theta_production: float = 0.55
    theta_da: float = 0.3
    theta_a: float = 0.3
    timeout_seconds: float = 180.0
    learning_rate: float = 0.05


# =========================================================================
# LemmaAcquisitionModule
# =========================================================================

class LemmaAcquisitionModule(nn.Module):
    """Discrete allocation and confirmation-gated reinforcement.

    BIOLOGICAL STRUCTURE: The synaptic plasticity machinery of
    mid-MTG and Wernicke's that implements one-shot lexical
    acquisition. Models the cellular processes (synaptic tagging,
    dopamine-gated capture, Hebbian reinforcement) that underlie
    fast mapping in human language acquisition.

    BIOLOGICAL FUNCTION: Manages the eight architectural commitments
    that distinguish the v2 substrate from a token-prediction model:
    novelty-gated allocation, three-valued allocation status,
    matrices outside the optimizer, polar-question co-activation on
    provisional rows, polar_question as a structural slot, no
    Hebbian reinforcement on provisional rows, frame-bias
    application before allocation, and i_dont_know fall-through on
    production threshold failure.

    Reference: Equations 25.1 through 25.9 of the Broca's corpus
    Section 25. Architect's Phase 3 spec, item 3 (matrix ownership)
    and item 4 (architectural commitments).

    INTERFACE CONTRACT:
        Inputs:
            substrate: a LexicalSubstrate instance whose W_C_to_L
                and W_L_to_P buffers this module mutates through
                the parent's write_row, clear_row, and reinforce_row
                methods.
        Outputs:
            is_novel(phon): True if the phonological code does not
                match any allocated row above theta_novelty.
            find_free_slot(): index of the first unallocated slot,
                or -1 if none.
            allocate_row(concept, phon): writes a new row in
                provisional state and returns the slot index.
            confirm_row(slot): transitions a provisional row to
                confirmed.
            decay_row(slot): immediately decays a provisional row
                back to unallocated.
            decay_unconfirmed(): periodic sweep of provisional rows
                that have timed out.
            reinforce_row(slot, concept, phon): Hebbian
                reinforcement step on a confirmed row.
            select_lemma_for_production(concept): returns
                (slot_index, polar_question_flag) for the production
                pathway.
            pre_allocate_reserved_slots(phon_fn): construction-time
                hook that writes the seventeen reserved slots'
                concept vectors and phonological codes.

        State: status array (n_lemmas,) and allocation_time array
            (n_lemmas,). Both serialize through this module's
            state_dict. The matrices serialize through the parent's
            state_dict.
    """

    def __init__(
        self,
        cfg: LemmaAcquisitionConfig,
        substrate: LexicalSubstrate,
    ) -> None:
        """Initialize the acquisition module against a parent
        substrate.

        Args:
            cfg: LemmaAcquisitionConfig.
            substrate: a LexicalSubstrate instance. The module holds
                a reference to it and mutates its matrices through
                the parent's write_row, clear_row, and reinforce_row
                methods.
        """
        super().__init__()
        self.cfg = cfg
        self.substrate = substrate
        self.n_lemmas = substrate.cfg.n_lemmas
        self.n_concepts = substrate.cfg.n_concepts
        self.d_phon = substrate.cfg.d_phon

        # Status array. Three-valued per slot: STATUS_UNALLOCATED,
        # STATUS_PROVISIONAL, STATUS_CONFIRMED. Buffer rather than
        # parameter because the values are discrete and must not
        # appear in any gradient graph.
        self.register_buffer(
            "status",
            torch.zeros(self.n_lemmas, dtype=torch.long),
        )

        # Allocation time array. Wall-clock time at which each row
        # was allocated, used to enforce the decay-without-
        # confirmation timeout. Float32 buffer because torch
        # serializes float64 buffers inefficiently and we do not
        # need sub-microsecond precision.
        self.register_buffer(
            "allocation_time",
            torch.zeros(self.n_lemmas, dtype=torch.float32),
        )

    # =====================================================================
    # Pre-allocation of reserved slots
    # =====================================================================

    def pre_allocate_reserved_slots(
        self,
        phonological_code_fn: Optional[
            Callable[[str], Tensor]
        ] = None,
        identity_concept_fn: Optional[
            Callable[[str], Tensor]
        ] = None,
        wh_concept_fn: Optional[
            Callable[[str], Tensor]
        ] = None,
        uncertainty_concept_fn: Optional[
            Callable[[str], Tensor]
        ] = None,
    ) -> None:
        """Write the seventeen reserved slots at construction time.

        Each reserved slot gets its concept vector and phonological
        code written into the substrate's matrices, and its status
        set to STATUS_CONFIRMED so the slot fires reliably without
        going through the provisional-to-confirmed transition.

        The four optional functions let the caller supply
        deterministic concept vectors and phonological codes. When
        omitted, the method uses zero vectors and a deterministic
        hash-based fallback for phonological codes derived from the
        slot's canonical text. The fallback is sufficient for tests
        and for the cold-start dialogue, but production deployments
        should supply functions tied to the substrate's actual
        encoding.

        Args:
            phonological_code_fn: optional function mapping canonical
                text to a (d_phon,) tensor.
            identity_concept_fn: optional function mapping identity
                slot name ("self_lemma" or "other_lemma") to a
                (n_concepts,) tensor.
            wh_concept_fn: optional function mapping wh-word name to
                a (n_concepts,) tensor.
            uncertainty_concept_fn: optional function mapping
                uncertainty marker name to a (n_concepts,) tensor.
        """
        for name, slot_index in SLOT_BY_NAME.items():
            phon_text = PHONOLOGICAL_TEXT_BY_NAME.get(name, "")
            if phonological_code_fn is not None and phon_text:
                phon = phonological_code_fn(phon_text)
            else:
                phon = self._default_phonological_code(phon_text)

            concept = self._concept_for_slot(
                name, slot_index,
                identity_concept_fn,
                wh_concept_fn,
                uncertainty_concept_fn,
            )

            self.substrate.write_row(slot_index, concept, phon)
            with torch.no_grad():
                self.status[slot_index] = STATUS_CONFIRMED
                self.allocation_time[slot_index] = float(time.time())

    def _default_phonological_code(self, text: str) -> Tensor:
        """Deterministic hash-based phonological code for the
        construction-time stub.

        Maps a canonical text string to a (d_phon,) tensor by hashing
        the string and unpacking the hash into deterministic float
        values. Same text always produces the same code; different
        texts produce codes that are nearly orthogonal in
        expectation.

        NOT a biological quantity, engineering convenience only. The
        biology realizes phonological codes through distributed
        cortical activation patterns shaped by acoustic and
        articulatory experience; this stub stands in for that
        encoding during testing.
        """
        if not text:
            return torch.zeros(self.d_phon)
        # Use Python's hash to seed a deterministic generator. The
        # generator produces a unit-norm random vector whose entries
        # are gaussian. The same text always seeds the same vector.
        seed = abs(hash(text)) % (2 ** 31)
        gen = torch.Generator()
        gen.manual_seed(seed)
        v = torch.randn(self.d_phon, generator=gen)
        return v / (v.norm() + 1e-9)

    def _concept_for_slot(
        self,
        name: str,
        slot_index: int,
        identity_concept_fn: Optional[
            Callable[[str], Tensor]
        ],
        wh_concept_fn: Optional[
            Callable[[str], Tensor]
        ],
        uncertainty_concept_fn: Optional[
            Callable[[str], Tensor]
        ],
    ) -> Tensor:
        """Return the concept vector for a reserved slot.

        Routes to the appropriate caller-supplied function if
        available, otherwise returns a deterministic hash-based
        default. The defaults seed the concept vectors so that
        production fall-through to the i_dont_know slot produces a
        non-zero phonological code through the W_L_to_P forward pass.
        """
        if slot_index in {SLOT_BY_NAME[n] for n in (
            "self_lemma", "other_lemma",
        )}:
            if identity_concept_fn is not None:
                return identity_concept_fn(name)
            return self._default_concept_for_text(f"identity:{name}")

        wh_names = {
            "what", "who", "where", "when", "why", "how",
            "polar_question",
        }
        if name in wh_names:
            if wh_concept_fn is not None:
                return wh_concept_fn(name)
            return self._default_concept_for_text(f"question:{name}")

        if uncertainty_concept_fn is not None:
            return uncertainty_concept_fn(name)
        return self._default_concept_for_text(f"uncertainty:{name}")

    def _default_concept_for_text(self, text: str) -> Tensor:
        """Deterministic hash-based concept vector default.

        Same shape as _default_phonological_code but in the concept
        dimensionality.
        """
        seed = abs(hash(text)) % (2 ** 31)
        gen = torch.Generator()
        gen.manual_seed(seed)
        v = torch.randn(self.n_concepts, generator=gen)
        return v / (v.norm() + 1e-9)

    # =====================================================================
    # Novelty gate (Item 1)
    # =====================================================================

    def is_novel(self, phonological_code: Tensor) -> bool:
        """Return True if the phonological code does not match any
        allocated row above theta_novelty.

        The novelty comparison runs over the allocated set only.
        Unallocated rows are excluded so that an empty matrix does
        not produce a near-zero maximum that would falsely register
        everything as novel. The comparison is cosine similarity
        between the input and each allocated row.

        Args:
            phonological_code: (d_phon,) phonological code from
                Wernicke's spell-out.

        Returns:
            True if no allocated row matches above theta_novelty.
        """
        if not self.cfg.enable_lemma_acquisition:
            return False
        with torch.no_grad():
            allocated_mask = self.status > STATUS_UNALLOCATED
            if not allocated_mask.any():
                return True
            # W_L_to_P has shape (d_phon, n_lemmas). Each column is
            # a stored phonological code. Allocated columns are the
            # ones to compare against.
            stored = self.substrate.W_L_to_P  # (d_phon, n_lemmas)
            stored_allocated = stored[:, allocated_mask]
            # Cosine similarity between input and each stored
            # column.
            input_norm = phonological_code / (
                phonological_code.norm() + 1e-9
            )
            stored_norm = stored_allocated / (
                stored_allocated.norm(dim=0, keepdim=True) + 1e-9
            )
            similarities = input_norm @ stored_norm  # (n_allocated,)
            max_similarity = similarities.max().item()
            return max_similarity < self.cfg.theta_novelty

    # =====================================================================
    # Free slot finder
    # =====================================================================

    def find_free_slot(self) -> int:
        """Return the first unallocated slot index >= RESERVED_SLOTS,
        or -1 if no free slot is available.

        Skips the reserved-slot range because those slots are
        permanently allocated at construction. The deterministic
        lowest-index choice produces inspectable allocation order
        for tests.
        """
        if not self.cfg.enable_lemma_acquisition:
            return -1
        with torch.no_grad():
            for index in range(RESERVED_SLOTS, self.n_lemmas):
                if self.status[index].item() == STATUS_UNALLOCATED:
                    return index
            return -1

    # =====================================================================
    # Allocation
    # =====================================================================

    def allocate_row(
        self,
        concept_vector: Tensor,
        phonological_code: Tensor,
    ) -> int:
        """Allocate a new lemma row in provisional state.

        Writes through the parent's write_row method (atomic across
        both matrices), sets status to STATUS_PROVISIONAL, records
        the wall-clock allocation time. The row will participate in
        polar-question co-activation but will not be Hebbian-
        reinforced until confirmation.

        When enable_provisional_state is False, the row goes
        directly to STATUS_CONFIRMED.

        Args:
            concept_vector: (n_concepts,) concept distribution.
            phonological_code: (d_phon,) phonological code.

        Returns:
            slot index of the new row, or -1 if allocation failed.
        """
        if not self.cfg.enable_lemma_acquisition:
            return -1
        slot_index = self.find_free_slot()
        if slot_index < 0:
            return -1

        self.substrate.write_row(
            slot_index, concept_vector, phonological_code,
        )
        with torch.no_grad():
            if self.cfg.enable_provisional_state:
                self.status[slot_index] = STATUS_PROVISIONAL
            else:
                self.status[slot_index] = STATUS_CONFIRMED
            self.allocation_time[slot_index] = float(time.time())
        return slot_index

    # =====================================================================
    # Confirmation
    # =====================================================================

    def confirm_row(self, slot_index: int) -> None:
        """Transition a provisional row to confirmed.

        No-op if the row is not currently in STATUS_PROVISIONAL. The
        confirmation signal is fundamentally noisy (a stray "yes"
        in conversation should not crash the substrate), so silently
        ignoring out-of-state confirmation is correct.

        Args:
            slot_index: row to confirm.
        """
        if not self.cfg.enable_lemma_acquisition:
            return
        with torch.no_grad():
            if self.status[slot_index].item() == STATUS_PROVISIONAL:
                self.status[slot_index] = STATUS_CONFIRMED

    # =====================================================================
    # Decay
    # =====================================================================

    def decay_row(self, slot_index: int) -> None:
        """Immediately decay a provisional row back to unallocated.

        Used by the confirmation_detector when an explicit correction
        ("no, your name is not Timmy") arrives. Symmetric with
        confirm_row: same atomic operation through the parent, just
        zeroing the row instead of changing the status to confirmed.

        No-op if the row is not currently in STATUS_PROVISIONAL.

        Args:
            slot_index: row to decay.
        """
        if not self.cfg.enable_lemma_acquisition:
            return
        with torch.no_grad():
            if self.status[slot_index].item() == STATUS_PROVISIONAL:
                self.substrate.clear_row(slot_index)
                self.status[slot_index] = STATUS_UNALLOCATED
                self.allocation_time[slot_index] = 0.0

    def decay_unconfirmed(self) -> None:
        """Periodic sweep that decays timed-out provisional rows.

        Called by the runtime tick. Any row in STATUS_PROVISIONAL
        whose allocation_time is older than timeout_seconds is
        cleared and reverted to STATUS_UNALLOCATED. Reserved slots
        are never affected because they have STATUS_CONFIRMED.
        """
        if not self.cfg.enable_lemma_acquisition:
            return
        if not self.cfg.enable_decay_timeout:
            return
        now = time.time()
        with torch.no_grad():
            for index in range(RESERVED_SLOTS, self.n_lemmas):
                if (
                    self.status[index].item() != STATUS_PROVISIONAL
                ):
                    continue
                age = now - float(self.allocation_time[index].item())
                if age > self.cfg.timeout_seconds:
                    self.substrate.clear_row(index)
                    self.status[index] = STATUS_UNALLOCATED
                    self.allocation_time[index] = 0.0

    # =====================================================================
    # Reinforcement
    # =====================================================================

    def reinforce_row(
        self,
        slot_index: int,
        concept_vector: Tensor,
        phonological_code: Tensor,
    ) -> bool:
        """Hebbian reinforcement step on a confirmed row.

        Calls the parent's reinforce_row method, which applies the
        leaky-integrator delta-rule under torch.no_grad. Returns
        False if the row is not confirmed (provisional rows do not
        reinforce, per Item 6).

        Args:
            slot_index: row to reinforce.
            concept_vector: (n_concepts,) target.
            phonological_code: (d_phon,) target.

        Returns:
            True if reinforcement applied, False if the row was not
                in STATUS_CONFIRMED.
        """
        if not self.cfg.enable_lemma_acquisition:
            return False
        if not self.cfg.enable_hebbian_reinforcement:
            return False
        with torch.no_grad():
            if self.status[slot_index].item() != STATUS_CONFIRMED:
                return False
        self.substrate.reinforce_row(
            slot_index, concept_vector, phonological_code,
            learning_rate=self.cfg.learning_rate,
        )
        return True

    # =====================================================================
    # Production selection
    # =====================================================================

    def select_lemma_for_production(
        self, concept_vector: Tensor,
    ) -> Tuple[int, bool]:
        """Select a lemma for the production pathway.

        Computes the dot product of the concept vector against each
        allocated row of W_C_to_L, finds the highest-scoring
        allocated row, and returns its index plus a polar-question
        co-activation flag. If the highest score is below
        theta_production, falls through to i_dont_know with no
        polar-question flag.

        Args:
            concept_vector: (n_concepts,) concept distribution.

        Returns:
            (slot_index, polar_question_flag).
        """
        if not self.cfg.enable_lemma_acquisition:
            return SLOT_I_DONT_KNOW, False
        with torch.no_grad():
            allocated_mask = self.status > STATUS_UNALLOCATED
            if not allocated_mask.any():
                return SLOT_I_DONT_KNOW, False
            scores = self.substrate.W_C_to_L @ concept_vector
            scores_allocated = scores.clone()
            scores_allocated[~allocated_mask] = float("-inf")
            best_slot = int(scores_allocated.argmax().item())
            best_score = float(
                scores_allocated[best_slot].item()
            )
            if best_score < self.cfg.theta_production:
                return SLOT_I_DONT_KNOW, False
            polar_q = False
            if self.cfg.enable_polar_question_coactivation:
                polar_q = (
                    self.status[best_slot].item()
                    == STATUS_PROVISIONAL
                )
            return best_slot, polar_q

    # =====================================================================
    # Diagnostics
    # =====================================================================

    def get_diagnostic_state(self) -> dict:
        """Return a dict of internal counters for logging."""
        with torch.no_grad():
            n_provisional = int(
                (self.status == STATUS_PROVISIONAL).sum().item()
            )
            n_confirmed = int(
                (self.status == STATUS_CONFIRMED).sum().item()
            )
            n_unallocated = int(
                (self.status == STATUS_UNALLOCATED).sum().item()
            )
            return {
                "n_provisional": n_provisional,
                "n_confirmed": n_confirmed,
                "n_unallocated": n_unallocated,
                "n_total": self.n_lemmas,
                "n_reserved": RESERVED_SLOTS,
            }
