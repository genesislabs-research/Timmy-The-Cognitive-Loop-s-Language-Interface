"""
lexical_substrate_t.py
Parent Substrate Owning the Tied Lexical Matrices

BIOLOGICAL GROUNDING
====================
This file implements the single-source-of-truth parent for the two
lexical projection matrices that connect the conceptual stratum, the
lemma stratum, and the lexical phonological code stratum in the v2
speech pathway. The matrices model the synaptic projections between
mid-MTG (lemma stratum) and Wernicke's area (lexical phonological code
stratum), and between mid-MTG and upstream conceptual cortex
(conceptual stratum, kernel boundary).

The biological reality is that there is one set of synapses between
each pair of cortical regions. A given lemma in mid-MTG projects to a
specific population of neurons in Wernicke's through a particular set
of axons; that same population in Wernicke's projects back to mid-MTG
through axons whose terminal arbors share trophic and developmental
history with the forward-direction terminals. The forward and reverse
directions therefore use the same physical synapses with their weights
constrained to be transposes of each other for the duration of any
single processing window. This is the corpus's tied-weights commitment
in Appendix F. The same commitment applies to the projection between
mid-MTG and the conceptual stratum.

Because there is one set of synapses, there is exactly one matrix
W_C_to_L and exactly one matrix W_L_to_P in the substrate at any time.
This module owns those matrices. mid-MTG and Wernicke's do not
construct their own copies; they hold references to this parent and
read or mutate the parent's tensors. The discrete-event mutation
methods (allocate_row, confirm_row, reinforce_row, decay_unconfirmed)
operate on the same tensors that the differentiable forward pass uses
during perception and production.

The matrices are stored as register_buffer rather than as nn.Parameter
with requires_grad=False. The distinction is not cosmetic. Parameters
participate in the optimizer's state-tracking machinery, which means
their state at training time partially lives in the optimizer rather
than in the module. The Reconamics continuity frame in Appendix C
specifies that experience must persist through buffer state, not
optimizer state, because optimizer state is a swappable training
artifact that does not survive deployment cycles. Buffers persist
through the model state_dict and round-trip through the .soul
checkpoint without the optimizer needing to be reconstructed at restore
time.

The forward-direction reads use the tied_substrate.BufferTiedSubstrate
wrapper which provides forward_a_to_b and forward_b_to_a methods over
an external buffer reference. The wrapper holds no parameters of its
own; all storage lives in the parent. This is the dependency-injection
pattern that the architect specified for resolution three of the
matrix-ownership question raised in the Phase 0 gap document.

Primary grounding papers:

Indefrey P, Levelt WJM (2004). "The spatial and temporal signatures of
word production components." Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001

Levelt WJM, Roelofs A, Meyer AS (1999). "A theory of lexical access in
speech production." Behavioral and Brain Sciences, 22(1), 1-75.
DOI: 10.1017/S0140525X99001776

Price CJ (2011). "A generative model of speech production in Broca's
and Wernicke's areas." Frontiers in Psychology, 2:237.
DOI: 10.3389/fpsyg.2011.00237

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# =========================================================================
# BufferTiedSubstrate
# =========================================================================

class BufferTiedSubstrate(nn.Module):
    """Tied-weights wrapper over an externally-owned buffer tensor.

    BIOLOGICAL STRUCTURE: A cortico-cortical projection between two
    populations of neurons. Production direction reads forward through
    the connection; perception direction reads backward through the
    same connection with transposed roles.

    BIOLOGICAL FUNCTION: Provides the bidirectional read interface
    over a single matrix that models the projection. The wrapper holds
    no storage of its own. The matrix is owned by a parent module and
    passed by reference at construction. When the parent mutates the
    matrix through discrete allocation events, the next forward call
    through this wrapper sees the mutation immediately because there
    is no second copy to fall out of sync.

    Reference: Appendix F of the Broca's corpus, tied-weights idiom.

    INTERFACE CONTRACT:
        Inputs:
            external_W: (out_dim, in_dim) tensor reference, owned and
                registered as a buffer by the parent module.
            in_dim: input dimensionality, used for shape verification.
            out_dim: output dimensionality, used for shape verification.
        Outputs:
            forward_a_to_b(x): (B, in_dim) -> (B, out_dim).
                Production direction: x @ W.T.
            forward_b_to_a(x): (B, out_dim) -> (B, in_dim).
                Perception direction: x @ W.

        State: stateless. All storage lives in the parent's buffer.
            The wrapper holds only a reference, which is shared with
            the parent and is preserved by Python reference semantics
            across module loading and saving.
    """

    def __init__(
        self,
        external_W: Tensor,
        in_dim: int,
        out_dim: int,
    ) -> None:
        """Initialize the wrapper around an external matrix buffer.

        Args:
            external_W: (out_dim, in_dim) tensor reference.
            in_dim: input dimensionality of forward_a_to_b.
            out_dim: output dimensionality of forward_a_to_b.

        Raises:
            ValueError: if external_W shape does not match
                (out_dim, in_dim).
        """
        super().__init__()
        if external_W.shape != (out_dim, in_dim):
            raise ValueError(
                f"external_W has shape {tuple(external_W.shape)}, "
                f"expected ({out_dim}, {in_dim})."
            )
        # Hold a Python reference to the parent's buffer. The buffer
        # is registered on the parent module rather than on self, so
        # state_dict serialization through the parent captures the
        # storage exactly once. The reference is updated automatically
        # when the parent calls register_buffer or in-place mutates the
        # tensor.
        self._W_ref = external_W
        self.in_dim = in_dim
        self.out_dim = out_dim

    @property
    def W(self) -> Tensor:
        """Return the current matrix tensor.

        Reads through to the parent's buffer at every access. This
        ensures that any in-place mutation by the parent (allocate_row,
        confirm_row, reinforce_row, decay_unconfirmed) is visible on
        the very next forward call without explicit synchronization.

        Returns:
            (out_dim, in_dim) tensor.
        """
        return self._W_ref

    def forward_a_to_b(self, x: Tensor) -> Tensor:
        """Production direction: in_dim space to out_dim space.

        Computes x @ W.T where W has shape (out_dim, in_dim). The
        result has shape (B, out_dim).

        For W_C_to_L (shape n_lemmas by n_concepts), this is the
        concept-to-lemma projection used in production: a concept
        distribution drives lemma activation contributions.

        For W_L_to_P (shape d_phon by n_lemmas), this is the
        lemma-to-phonological-code projection used in production: a
        lemma activation produces a phonological code.

        Args:
            x: (B, in_dim) input.

        Returns:
            (B, out_dim) output.
        """
        return x @ self._W_ref.t()

    def forward_b_to_a(self, x: Tensor) -> Tensor:
        """Perception direction: out_dim space to in_dim space.

        Computes x @ W where W has shape (out_dim, in_dim). The result
        has shape (B, in_dim).

        For W_C_to_L, this is the lemma-to-concept reverse projection
        used in comprehension: an active lemma drives a reconstructed
        concept distribution back to the kernel boundary.

        For W_L_to_P, this is the phonological-code-to-lemma reverse
        projection used in comprehension: a perceived phonological code
        drives lemma activation back into mid-MTG.

        Args:
            x: (B, out_dim) input.

        Returns:
            (B, in_dim) output.
        """
        return x @ self._W_ref


# =========================================================================
# LexicalSubstrateConfig
# =========================================================================

@dataclass
class LexicalSubstrateConfig:
    """Configuration for the parent lexical substrate.

    The dimensions specified here propagate to mid-MTG, Wernicke's, and
    the lemma acquisition module through dependency injection. Changing
    them here is the only correct place to change them; the consumer
    modules read these values from the parent at construction time.

    Master flag is first per the Genesis Labs ablation flag standard.
    NOT a biological quantity.

    Attributes:
        enable_lexical_substrate: master flag for the whole module.
            False produces a substrate where forward calls return
            zeros and mutations silently no-op, which is the standard
            ablation behavior.
        n_concepts: dimensionality of the conceptual stratum vector.
            The kernel boundary speaks in this dimensionality, and the
            architect's reservation places frame-bias dimensions at
            1000-1015 and uncertainty-subspace dimensions at 1016-1023
            within this space.
        n_lemmas: total number of lemma slots in mid-MTG. The first
            RESERVED_SLOTS slots are pre-allocated for identity, wh-words,
            polar_question, and the eight uncertainty markers; the
            remainder are available for runtime allocation through
            allocate_row.
        d_phon: dimensionality of the lexical phonological code in
            Wernicke's. The phonological code is a distributed
            representation rather than a single segment; segmental
            spell-out happens downstream through the spell-out decoder.
    """

    enable_lexical_substrate: bool = True
    n_concepts: int = 1024
    n_lemmas: int = 512
    d_phon: int = 512


# =========================================================================
# LexicalSubstrate
# =========================================================================

class LexicalSubstrate(nn.Module):
    """Parent module owning the tied lexical matrices.

    BIOLOGICAL STRUCTURE: The full set of cortico-cortical projections
    between mid-MTG, Wernicke's lexical phonological code store, and
    the conceptual stratum upstream. Models a single physical set of
    synapses for each connection rather than independent forward and
    backward pathways.

    BIOLOGICAL FUNCTION: Holds the W_C_to_L and W_L_to_P matrices that
    encode lexical knowledge. Discrete acquisition events (Phase 3
    lemma allocation, confirmation, reinforcement, decay) mutate these
    matrices. The differentiable forward pass during perception and
    production reads them. mid-MTG and Wernicke's read and write the
    matrices through the parent rather than maintaining their own
    copies.

    Reference: Equations 12.1, 13.1, and 13.3 of the Broca's corpus.
    Appendix F tied-weights idiom. Phase 0 gap document, resolution 3
    of the matrix-ownership question.

    ANATOMICAL INTERFACE (W_C_to_L):
        Sending structure (production): Conceptual stratum at the
            kernel boundary.
        Receiving structure (production): Lemma stratum in mid-MTG.
        Connection: Concept-to-lemma projection W_C_to_L of shape
            (n_lemmas, n_concepts). The forward direction is
            forward_a_to_b producing lemma activations from concept
            distributions; the reverse direction is forward_b_to_a
            producing reconstructed concept distributions from lemma
            activations.

    ANATOMICAL INTERFACE (W_L_to_P):
        Sending structure (production): Lemma stratum in mid-MTG.
        Receiving structure (production): Lexical phonological code
            store in Wernicke's.
        Connection: Lemma-to-phonological-code projection W_L_to_P of
            shape (d_phon, n_lemmas). Forward direction produces
            phonological codes from lemma activations; reverse
            direction drives lemma activation from perceived
            phonological codes.

    STATE: Two register_buffer tensors W_C_to_L and W_L_to_P that
    serialize through the parent's state_dict. No optimizer state.
    No nn.Parameter for these matrices. The Reconamics commitment is
    that experience lives in buffer state.
    """

    def __init__(self, cfg: LexicalSubstrateConfig) -> None:
        """Initialize the parent substrate with zeroed matrices.

        At construction the matrices are all zero. Pre-allocation of
        identity lemmas, uncertainty markers, and question primes
        happens through the lemma_acquisition module's
        pre_allocate_reserved_slots method, called by the runtime after
        the substrate is constructed and before the first dialogue
        turn. The parent does not pre-allocate by itself because the
        slot inventory is owned by the coordination/lemma_slots module
        rather than by the substrate.

        Args:
            cfg: LexicalSubstrateConfig.
        """
        super().__init__()
        self.cfg = cfg

        # W_C_to_L: concept-to-lemma projection. Shape (n_lemmas,
        # n_concepts). Initialized to zero so the lemma-local novelty
        # gate at Wernicke's correctly classifies all phonological
        # codes as novel until allocation populates rows.
        # Reference: Equation 12.1 of the Broca's corpus.
        # NOT a biological quantity in this exact form. Biology
        # initializes synaptic projections through a developmental
        # process that produces small random weights with structured
        # priors; the v2 substrate uses zero initialization plus
        # discrete allocation events to model the tabula-rasa start
        # state for Phase 3 acquisition testing.
        self.register_buffer(
            "W_C_to_L",
            torch.zeros(cfg.n_lemmas, cfg.n_concepts),
        )

        # W_L_to_P: lemma-to-phonological-code projection. Shape
        # (d_phon, n_lemmas). Same zero initialization rationale as
        # above.
        # Reference: Equation 13.1 of the Broca's corpus.
        self.register_buffer(
            "W_L_to_P",
            torch.zeros(cfg.d_phon, cfg.n_lemmas),
        )

        # The TiedSubstrate wrappers. These hold references to the
        # buffers above. Forward calls through them read the parent's
        # tensors directly; there is no second copy.
        self.tied_w_c_to_l = BufferTiedSubstrate(
            external_W=self.W_C_to_L,
            in_dim=cfg.n_concepts,
            out_dim=cfg.n_lemmas,
        )
        self.tied_w_l_to_p = BufferTiedSubstrate(
            external_W=self.W_L_to_P,
            in_dim=cfg.n_lemmas,
            out_dim=cfg.d_phon,
        )

    # =====================================================================
    # Discrete-event mutation interface
    # =====================================================================
    #
    # These methods are the canonical mutation surface for the matrices.
    # They operate under torch.no_grad() because discrete allocation events
    # are not differentiable. Any code that needs to mutate W_C_to_L or
    # W_L_to_P calls through these methods rather than touching the buffers
    # directly. This preserves the architectural invariant that mutations
    # are atomic at the row level and that perception and production see
    # the post-mutation state on the very next forward call.

    def write_row(
        self,
        slot_index: int,
        concept_vector: Tensor,
        phonological_code: Tensor,
    ) -> None:
        """Write a new lemma row across both tied matrices atomically.

        The atomicity matters because partially-written rows produce
        ill-defined behavior in the comprehension direction: a row
        with concept content but no phonological code routes
        comprehension to a lemma that has no spoken form, which then
        tries to drive Wernicke's with a zero vector. Writing both
        rows under a single torch.no_grad() block ensures that the
        next forward call sees a well-formed lemma.

        Args:
            slot_index: row index to write.
            concept_vector: (n_concepts,) concept distribution to bind
                into W_C_to_L[slot_index, :].
            phonological_code: (d_phon,) phonological code to bind
                into W_L_to_P[:, slot_index].

        Raises:
            IndexError: if slot_index is outside [0, n_lemmas).
            ValueError: if either input has the wrong shape.
        """
        if not 0 <= slot_index < self.cfg.n_lemmas:
            raise IndexError(
                f"slot_index {slot_index} out of range "
                f"[0, {self.cfg.n_lemmas})."
            )
        if concept_vector.shape != (self.cfg.n_concepts,):
            raise ValueError(
                f"concept_vector has shape "
                f"{tuple(concept_vector.shape)}, "
                f"expected ({self.cfg.n_concepts},)."
            )
        if phonological_code.shape != (self.cfg.d_phon,):
            raise ValueError(
                f"phonological_code has shape "
                f"{tuple(phonological_code.shape)}, "
                f"expected ({self.cfg.d_phon},)."
            )
        with torch.no_grad():
            self.W_C_to_L[slot_index] = concept_vector
            self.W_L_to_P[:, slot_index] = phonological_code

    def clear_row(self, slot_index: int) -> None:
        """Zero a row across both tied matrices atomically.

        Used by decay_unconfirmed in the lemma_acquisition module when
        a provisional binding times out without confirmation. Also
        used by tests to reset between cases.

        Args:
            slot_index: row index to clear.

        Raises:
            IndexError: if slot_index is outside [0, n_lemmas).
        """
        if not 0 <= slot_index < self.cfg.n_lemmas:
            raise IndexError(
                f"slot_index {slot_index} out of range "
                f"[0, {self.cfg.n_lemmas})."
            )
        with torch.no_grad():
            self.W_C_to_L[slot_index].zero_()
            self.W_L_to_P[:, slot_index].zero_()

    def reinforce_row(
        self,
        slot_index: int,
        concept_vector: Tensor,
        phonological_code: Tensor,
        learning_rate: float = 0.05,
    ) -> None:
        """Apply a small Hebbian reinforcement step to a row.

        Reinforcement nudges the existing row toward the supplied
        concept and phonological code rather than overwriting it.
        Confirmation gating is enforced upstream by the
        lemma_acquisition module: this method does not check the
        row's status and will reinforce any row it is called on.
        Callers are expected to verify status before calling.

        The Hebbian rule is W += eta * (target - W) where target is
        the supplied vector. This is a leaky integrator that converges
        on the target as exposure repeats, with the learning_rate
        scalar controlling convergence speed.

        Args:
            slot_index: row index to reinforce.
            concept_vector: (n_concepts,) target concept distribution.
            phonological_code: (d_phon,) target phonological code.
            learning_rate: step size in (0, 1]. NOT a biological
                quantity, training artifact only. Default 0.05 is
                engineering judgment for substrate-typical activation
                magnitudes.

        Raises:
            IndexError: if slot_index is outside [0, n_lemmas).
            ValueError: if either input has the wrong shape, or if
                learning_rate is not in (0, 1].
        """
        if not 0 <= slot_index < self.cfg.n_lemmas:
            raise IndexError(
                f"slot_index {slot_index} out of range "
                f"[0, {self.cfg.n_lemmas})."
            )
        if concept_vector.shape != (self.cfg.n_concepts,):
            raise ValueError(
                f"concept_vector has shape "
                f"{tuple(concept_vector.shape)}, "
                f"expected ({self.cfg.n_concepts},)."
            )
        if phonological_code.shape != (self.cfg.d_phon,):
            raise ValueError(
                f"phonological_code has shape "
                f"{tuple(phonological_code.shape)}, "
                f"expected ({self.cfg.d_phon},)."
            )
        if not 0.0 < learning_rate <= 1.0:
            raise ValueError(
                f"learning_rate {learning_rate} not in (0, 1]."
            )
        with torch.no_grad():
            current_concept = self.W_C_to_L[slot_index]
            current_phon = self.W_L_to_P[:, slot_index]
            self.W_C_to_L[slot_index] = (
                current_concept
                + learning_rate * (concept_vector - current_concept)
            )
            self.W_L_to_P[:, slot_index] = (
                current_phon
                + learning_rate * (phonological_code - current_phon)
            )

    # =====================================================================
    # Read interface
    # =====================================================================

    def read_concept_row(self, slot_index: int) -> Tensor:
        """Return the concept-side row for a given lemma slot.

        Used by diagnostic code and by the comprehension direction
        when reconstructing the concept distribution from a single
        active lemma without going through the full forward_b_to_a
        path. The returned tensor is a view into the buffer; mutation
        of the returned tensor mutates the substrate.

        Args:
            slot_index: row index to read.

        Returns:
            (n_concepts,) tensor view into W_C_to_L[slot_index, :].

        Raises:
            IndexError: if slot_index is outside [0, n_lemmas).
        """
        if not 0 <= slot_index < self.cfg.n_lemmas:
            raise IndexError(
                f"slot_index {slot_index} out of range "
                f"[0, {self.cfg.n_lemmas})."
            )
        return self.W_C_to_L[slot_index]

    def read_phonological_row(self, slot_index: int) -> Tensor:
        """Return the phonological-side column for a given lemma slot.

        Symmetric with read_concept_row but for the W_L_to_P matrix.
        Note that the per-lemma data lives in a column rather than a
        row because of the (d_phon, n_lemmas) shape convention.

        Args:
            slot_index: lemma index to read.

        Returns:
            (d_phon,) tensor view into W_L_to_P[:, slot_index].

        Raises:
            IndexError: if slot_index is outside [0, n_lemmas).
        """
        if not 0 <= slot_index < self.cfg.n_lemmas:
            raise IndexError(
                f"slot_index {slot_index} out of range "
                f"[0, {self.cfg.n_lemmas})."
            )
        return self.W_L_to_P[:, slot_index]

    # =====================================================================
    # Forward-pass convenience methods
    # =====================================================================
    #
    # These delegate to the BufferTiedSubstrate wrappers. Consumers can call
    # either the wrapper directly through self.tied_w_c_to_l.forward_a_to_b
    # or through these convenience methods. Both paths read the same buffer.

    def forward_concept_to_lemma(self, concept: Tensor) -> Tensor:
        """Production direction through W_C_to_L.

        Args:
            concept: (B, n_concepts) concept distribution.

        Returns:
            (B, n_lemmas) lemma activation contributions.
        """
        if not self.cfg.enable_lexical_substrate:
            return torch.zeros(
                concept.shape[0],
                self.cfg.n_lemmas,
                device=concept.device,
                dtype=concept.dtype,
            )
        return self.tied_w_c_to_l.forward_a_to_b(concept)

    def forward_lemma_to_concept(self, lemma: Tensor) -> Tensor:
        """Comprehension direction through W_C_to_L (transposed read).

        Args:
            lemma: (B, n_lemmas) lemma activation.

        Returns:
            (B, n_concepts) reconstructed concept distribution.
        """
        if not self.cfg.enable_lexical_substrate:
            return torch.zeros(
                lemma.shape[0],
                self.cfg.n_concepts,
                device=lemma.device,
                dtype=lemma.dtype,
            )
        return self.tied_w_c_to_l.forward_b_to_a(lemma)

    def forward_lemma_to_phonological(self, lemma: Tensor) -> Tensor:
        """Production direction through W_L_to_P.

        Args:
            lemma: (B, n_lemmas) lemma activation.

        Returns:
            (B, d_phon) phonological code.
        """
        if not self.cfg.enable_lexical_substrate:
            return torch.zeros(
                lemma.shape[0],
                self.cfg.d_phon,
                device=lemma.device,
                dtype=lemma.dtype,
            )
        return self.tied_w_l_to_p.forward_a_to_b(lemma)

    def forward_phonological_to_lemma(self, phon: Tensor) -> Tensor:
        """Comprehension direction through W_L_to_P (transposed read).

        Args:
            phon: (B, d_phon) phonological code.

        Returns:
            (B, n_lemmas) lemma activation drive.
        """
        if not self.cfg.enable_lexical_substrate:
            return torch.zeros(
                phon.shape[0],
                self.cfg.n_lemmas,
                device=phon.device,
                dtype=phon.dtype,
            )
        return self.tied_w_l_to_p.forward_b_to_a(phon)

    # =====================================================================
    # Diagnostics
    # =====================================================================

    def get_diagnostic_state(self) -> dict:
        """Return a dict of internal norms and counters for logging.

        Useful for verifying that the substrate is in the expected
        state during integration tests and for monitoring drift during
        long-running sessions.

        Returns:
            dict with keys:
                W_C_to_L_norm: Frobenius norm of W_C_to_L.
                W_L_to_P_norm: Frobenius norm of W_L_to_P.
                W_C_to_L_n_nonzero_rows: count of rows with any
                    nonzero element in W_C_to_L.
                W_L_to_P_n_nonzero_cols: count of columns with any
                    nonzero element in W_L_to_P.
        """
        with torch.no_grad():
            c_l_nonzero_rows = int(
                (self.W_C_to_L.abs().sum(dim=1) > 0).sum().item()
            )
            l_p_nonzero_cols = int(
                (self.W_L_to_P.abs().sum(dim=0) > 0).sum().item()
            )
            return {
                "W_C_to_L_norm": float(
                    self.W_C_to_L.norm().item()
                ),
                "W_L_to_P_norm": float(
                    self.W_L_to_P.norm().item()
                ),
                "W_C_to_L_n_nonzero_rows": c_l_nonzero_rows,
                "W_L_to_P_n_nonzero_cols": l_p_nonzero_cols,
            }
