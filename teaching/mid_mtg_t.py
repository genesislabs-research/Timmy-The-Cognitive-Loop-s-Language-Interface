"""
mid_mtg_t.py
Mid-MTG: The Lemma Stratum

BIOLOGICAL GROUNDING
====================
This file models the mid section of the left middle temporal gyrus (mid-MTG),
the cortical substrate for lemma activation and selection in the language
production network. The lemma in classical psycholinguistic theory is the
abstract syntactic word: the unit that carries grammatical category, gender,
argument structure, and other syntactic features, but not yet phonological
form. Picture naming activates mid-MTG between approximately 150 and 225 ms
post stimulus onset, with selection complete by 250 ms. Comprehension
activates the same substrate roughly 150 to 175 ms later through the
reverse-direction projection from Wernicke's phonological code store.

The mid-MTG-as-lemma identification rests on three converging lines of
evidence. First, magnetoencephalographic and intracranial recordings show
mid-MTG activation at the lemma-selection latency window across multiple
languages and tasks (Indefrey and Levelt 2004 meta-analysis of 82 word
production experiments). Second, mid-MTG lesions produce semantic and
syntactic naming errors that respect grammatical structure even when
phonological output is intact, indicating the lesioned substrate carries
syntactic information rather than acoustic information (Damasio et al. 2004).
Third, mid-MTG activates similarly during picture naming and word listening
when the same lemma is targeted, supporting the bidirectional access pattern
that the v2 tied-substrate commitment formalizes.

The boundary with the Cognitive Kernel is at this stratum. The kernel hands
in a concept distribution c_lex of shape (B, n_concepts), the substrate
selects a lemma, and the kernel hands a reconstructed concept activation
back in the comprehension direction. The lemma vocabulary is
substrate-internal and grows through Phase 3 lexical acquisition. The concept
vocabulary is kernel-internal and the substrate does not modify it.

This file implements the three equations from Section 12 of the Broca's
corpus:

    a_l(t) = a_l(t-1) * gamma_lemma
             + sum_i W_C_to_L[l,i] * c_lex,i(t)                        (12.1)

    l*(t) = argmax_l a_l(t) when t >= t_lemma                          (12.2)

    delta_a_l_interfere(t) = -kappa_interfere * sum_{j != l}
                                 a_j(t) * sim(C_l, C_j)                (12.3)

Plus the pre-allocated identity and uncertainty lemma slot machinery from
Section 24.7.4 and Section 24a.5 of the v2 spec.

PLACEHOLDER NOTE
================
Equation 12.3 specifies sim as "a learned similarity kernel over concept
embeddings." The scaffold uses cosine similarity over a fixed random
projection of the concept distribution as a stand-in. This is honest about
the substitution: cosine similarity is what a learned kernel would converge
to in the limit of orthogonal concept embeddings, and the dynamics it
produces have the right qualitative shape (semantically similar lemmas
suppress each other through the lateral interference term). When the kernel
is trained, the cosine stand-in is replaced by the learned kernel without
changing the surrounding code. The substitution is labeled at the
implementation site and re-flagged in this docstring so it does not propagate
unflagged into downstream files.

Primary grounding papers:

Indefrey P, Levelt WJM (2004). "The spatial and temporal signatures of word
production components." Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001

Levelt WJM, Roelofs A, Meyer AS (1999). "A theory of lexical access in speech
production." Behavioral and Brain Sciences, 22(1), 1-75.
DOI: 10.1017/S0140525X99001776

Damasio H, Tranel D, Grabowski T, Adolphs R, Damasio A (2004). "Neural
systems behind word and concept retrieval." Cognition, 92(1-2), 179-229.
DOI: 10.1016/j.cognition.2002.07.001

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from substrate.tied_substrate_t import TiedSubstrate, TiedSubstrateConfig


# =========================================================================
# Reserved lemma slot identities
# =========================================================================

# Pre-allocated lemma slot indices for the identity module and uncertainty
# vocabulary. These slots are reserved at substrate construction; their
# phonological codes are filled in during Phase 3 acquisition through
# normal lexical learning.
#
# Reference: v2 Spec Section 24a.5 (identity module) and Section 24.7.4
# (uncertainty lemma reservation).

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

# Question-word lemma slots. Pre-allocated structural lemmas that the
# substrate uses to form questions when its epistemic state corresponds
# to medium confidence plus high curiosity. The slots exist from
# construction; their phonological codes are acquired through normal
# Phase 3 lexical acquisition during early sessions when the instructor
# demonstrates question-asking and the substrate observes the pattern.
# Reference: Genesis Teaching for Timmy specification, section "Why the
# substrate must be able to form questions."
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


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class MidMTGConfig:
    """Configuration for the mid-MTG region.

    The master flag follows the cognitive-loop ablation flag standard. Each
    sub-flag corresponds to a cited mechanism and can be ablated
    independently to verify that the mechanism contributes its expected
    behavior.

    NOT a biological quantity for the flags themselves. The biological
    quantities (gamma_lemma, kappa_interfere, t_lemma_steps) are labeled
    with their citations.

    Attributes:
        enable_mid_mtg: master flag.
        enable_persistence: enable a_l decay across timesteps.
        enable_lateral_interference: enable the kappa_interfere term.
        enable_identity_routing: reserve identity lemma slots.
        enable_uncertainty_lemmas: reserve uncertainty lemma slots.
        n_concepts: dimensionality of the concept distribution from the
            kernel boundary. The Boundary Contract Section 2 specifies
            that the kernel hands in c_lex of shape (B, n_concepts) and
            that this dimension is kernel-internal. The default of 1024
            matches the contract's recommended starting value; eventual
            integration with the kernel will replace this with whatever
            the kernel actually exposes.
        n_lemmas: maximum lemma slot count. Allocated up front to avoid
            dynamic tensor resizing during acquisition. The first
            N_RESERVED_LEMMAS slots are pre-allocated for identity and
            uncertainty; the rest are available for normal lexical
            acquisition.
        gamma_lemma: lemma activation persistence decay per cycle.
            Equation 12.1. From the Broca's corpus, lemma activation
            "peaks 150 to 225 ms post picture onset" and "selection
            completes at approximately 250 ms," giving a characteristic
            integration window of roughly 75 ms. With dt = 5 ms (15
            timesteps), gamma = 0.95 gives a half-life around 14 steps
            (70 ms), within the documented range.
        kappa_interfere: lateral interference strength. Equation 12.3.
            Tuned so that semantically similar lemmas suppress each other
            without preventing the target lemma from selecting.
        t_lemma_steps: number of integration timesteps before the
            argmax selection in equation 12.2. Approximately 250 ms /
            dt = 50 steps with dt = 5 ms.
        confidence_floor: floor on the lemma confidence signal.
            Numerical safety to avoid division by zero in the
            peak-to-average ratio computation.
    """

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


# =========================================================================
# Mid-MTG Region
# =========================================================================

class MidMTG(nn.Module):
    """The lemma stratum.

    BIOLOGICAL STRUCTURE: Mid-section of left middle temporal gyrus.

    BIOLOGICAL FUNCTION: Lemma activation and selection. Carries syntactic
    properties (word category, grammatical gender, argument structure)
    without phonological form. Bidirectional with the conceptual stratum
    upstream (production: concept activates lemma; comprehension: lemma
    activates concept).

    Reference: Indefrey P, Levelt WJM (2004). DOI: 10.1016/j.cognition.2002.06.001

    ANATOMICAL INTERFACE (production input):
        Sending structure: Conceptual stratum in distributed semantic
            cortex (kernel boundary).
        Receiving structure: Lemma stratum in mid-MTG (this module).
        Connection: Concept-to-lemma projection W_C_to_L.

    ANATOMICAL INTERFACE (production output):
        Sending structure: Lemma stratum in mid-MTG (this module).
        Receiving structure: Lexical phonological code store in
            Wernicke's (left posterior STG and pMTG).
        Connection: Lemma-to-phonological-code projection W_L_to_P
            (held by the Wernicke's region).

    ANATOMICAL INTERFACE (comprehension input):
        Sending structure: Wernicke's lexical phonological code store.
        Receiving structure: Lemma stratum in mid-MTG (this module).
        Connection: Phonological-code-to-lemma projection (the same
            W_L_to_P matrix, accessed in the reverse direction).

    ANATOMICAL INTERFACE (comprehension output):
        Sending structure: Lemma stratum in mid-MTG (this module).
        Receiving structure: Conceptual stratum upstream (kernel
            boundary).
        Connection: Lemma-to-concept projection (the same W_C_to_L
            matrix, accessed in the reverse direction).

    STATE: Persistent lemma activations a_l across timesteps. Decay is
    gamma_lemma per cycle. Serializes as a vector of shape (n_lemmas,).
    """

    def __init__(self, cfg: MidMTGConfig) -> None:
        """Initialize mid-MTG with the W_C_to_L tied substrate and the
        pre-allocated reserved lemma slots.

        Args:
            cfg: MidMTGConfig.
        """
        super().__init__()
        self.cfg = cfg

        # The concept-to-lemma tied substrate. W_C_to_L has shape
        # (n_lemmas, n_concepts). forward_a_to_b takes a concept
        # distribution and produces lemma activation contributions;
        # forward_b_to_a takes lemma activation and produces a
        # reconstructed concept distribution.
        # Reference: Equation 12.1 of the Broca's corpus.
        self.w_c_to_l = TiedSubstrate(TiedSubstrateConfig(
            in_dim=cfg.n_concepts,
            out_dim=cfg.n_lemmas,
        ))

        # Persistent lemma activation buffer. Initialized to zero on
        # construction. Updated each timestep through equation 12.1.
        # Cleared by reset_state() between unrelated sessions; preserved
        # across sessions through the .soul checkpoint.
        # Reference: v2 Spec Section 9 (state serialization).
        self.register_buffer(
            "a_lemma", torch.zeros(1, cfg.n_lemmas),
        )

        # Concept-distribution memory used for the lateral interference
        # similarity kernel. Holds the most recent concept input so that
        # similarity between currently-active lemmas can be computed.
        # PLACEHOLDER: a learned similarity kernel would use a separate
        # concept-embedding matrix. The scaffold uses a fixed random
        # projection of the concept distribution into a similarity space,
        # which gives cosine similarity in that space the right
        # qualitative shape.
        # Reference: Equation 12.3 of the Broca's corpus, with the
        # placeholder substitution flagged in this file's header docstring.
        self.register_buffer(
            "_concept_embedding_for_sim",
            torch.randn(cfg.n_concepts, 32) * 0.1,
        )

        # Last concept distribution received, held for similarity
        # computation. Default zero, meaning no incoming concept yet.
        self.register_buffer(
            "_last_concept", torch.zeros(1, cfg.n_concepts),
        )

        # Internal step counter for the t_lemma_steps gating in
        # equation 12.2. Resets to zero at the start of each new
        # selection episode (when reset_for_selection() is called).
        # NOT a biological quantity. Implementation accounting.
        self.register_buffer(
            "_steps_since_reset", torch.tensor(0, dtype=torch.long),
        )

        # Allocation flags for each lemma slot. A lemma is "allocated"
        # once its W_C_to_L row has been written to by acquisition or
        # by the construction-time reservation for identity and
        # uncertainty slots. The flag is read by get_lemma_confidence to
        # distinguish "I have no acquired lemma for this concept" (peak
        # falls on an unallocated slot, confidence floored) from "I have
        # acquired lemmas but my activation distribution is currently
        # noisy" (peak falls on an allocated slot, confidence reflects
        # the peak-to-average ratio over allocated slots only).
        # Reference: v2 Spec Section 24.2.3 ("I do not have a word for
        # this concept") and Section 24a.6 dependency 10 (lemma
        # allocation on novel codes).
        # NOT a biological quantity in this exact form. The biology has
        # implicit allocation through the presence or absence of
        # synaptic strength; the explicit flag is an implementation
        # convenience that lets the confidence signal read the
        # allocation status without thresholding the weight matrix.
        is_allocated = torch.zeros(cfg.n_lemmas, dtype=torch.bool)
        # Reserved slots are pre-allocated by construction. Their
        # phonological codes are not yet trained; that happens during
        # Phase 3 acquisition. But the slot itself is allocated, so a
        # peak landing here is a real signal that the substrate
        # recognizes the relevant identity, uncertainty, or
        # question-word concept.
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

    # ---------------------------------------------------------------
    # Reserved lemma slot accessors
    # ---------------------------------------------------------------

    def identity_slot(self, name: str) -> int:
        """Return the lemma slot index for an identity lemma.

        Args:
            name: one of the keys in IDENTITY_LEMMA_SLOTS
                ("self_lemma", "other_lemma").

        Returns:
            integer slot index in the lemma vocabulary.
        """
        if name not in IDENTITY_LEMMA_SLOTS:
            raise KeyError(
                f"Unknown identity slot '{name}'. "
                f"Expected one of {sorted(IDENTITY_LEMMA_SLOTS.keys())}."
            )
        return IDENTITY_LEMMA_SLOTS[name]

    def uncertainty_slot(self, name: str) -> int:
        """Return the lemma slot index for an uncertainty lemma.

        Args:
            name: one of the keys in UNCERTAINTY_LEMMA_SLOTS.

        Returns:
            integer slot index in the lemma vocabulary.
        """
        if name not in UNCERTAINTY_LEMMA_SLOTS:
            raise KeyError(
                f"Unknown uncertainty slot '{name}'. "
                f"Expected one of {sorted(UNCERTAINTY_LEMMA_SLOTS.keys())}."
            )
        return UNCERTAINTY_LEMMA_SLOTS[name]

    def question_slot(self, name: str) -> int:
        """Return the lemma slot index for a question-word lemma.

        Question words are pre-allocated structural lemmas the substrate
        uses to form questions when its epistemic state corresponds to
        medium confidence plus high curiosity. The slots exist from
        construction; their phonological codes are acquired through
        normal Phase 3 lexical acquisition.

        Reference: Genesis Teaching for Timmy specification.

        Args:
            name: one of the keys in QUESTION_LEMMA_SLOTS
                ("what", "who", "where", "when", "why", "how").

        Returns:
            integer slot index in the lemma vocabulary.
        """
        if name not in QUESTION_LEMMA_SLOTS:
            raise KeyError(
                f"Unknown question slot '{name}'. "
                f"Expected one of {sorted(QUESTION_LEMMA_SLOTS.keys())}."
            )
        return QUESTION_LEMMA_SLOTS[name]

    def get_question_lemma_slots(self) -> Dict[str, int]:
        """Return the mapping of question-word names to slot indices.

        Used by the production loop's question-formation trigger to
        identify which lemma slots correspond to question-words when
        biasing lemma activation during medium-confidence-plus-high-
        curiosity states.

        Returns:
            dict mapping question-word name to integer slot index.
        """
        return dict(QUESTION_LEMMA_SLOTS)

    # ---------------------------------------------------------------
    # Forward computation
    # ---------------------------------------------------------------

    def forward_production(self, c_lex: Tensor) -> Tensor:
        """Production direction: concept distribution to lemma activation.

        Implements equation 12.1 followed by lateral interference from
        equation 12.3. Updates the persistent activation buffer in place.
        Does not perform selection; that happens in select_lemma() once
        the integration window has elapsed.

        Args:
            c_lex: (B, n_concepts) concept distribution from the kernel
                boundary. Must be normalized along the concept dimension
                (the kernel is responsible for this normalization).

        Returns:
            (B, n_lemmas) updated lemma activation tensor.
        """
        if not self.cfg.enable_mid_mtg:
            return torch.zeros(
                c_lex.shape[0], self.cfg.n_lemmas,
                device=c_lex.device, dtype=c_lex.dtype,
            )

        # Make sure the persistent buffer matches the batch size of the
        # incoming input. If the batch size changes between calls, expand
        # or truncate the buffer to match. This is the same pattern the
        # cognitive-loop CorticalBuffer uses.
        # NOT a biological quantity. PyTorch shape-handling convention.
        B = c_lex.shape[0]
        if self.a_lemma.shape[0] != B:
            self.a_lemma = torch.zeros(
                B, self.cfg.n_lemmas,
                device=c_lex.device, dtype=c_lex.dtype,
            )

        # Equation 12.1: persistence decay plus new concept-driven input.
        # gamma_lemma carries previous activation forward; the
        # concept-to-lemma projection adds the new contribution.
        if self.cfg.enable_persistence:
            decayed = self.a_lemma * self.cfg.gamma_lemma
        else:
            decayed = torch.zeros_like(self.a_lemma)

        new_input = self.w_c_to_l.forward_a_to_b(c_lex)
        a_new = decayed + new_input

        # Equation 12.3: lateral interference. Each lemma's activation is
        # reduced by a weighted sum of other lemmas' activations,
        # weighted by their concept-space similarity.
        # PLACEHOLDER: cosine similarity over a fixed random projection
        # of the concept distribution. Replaced by a learned similarity
        # kernel during acquisition.
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
        """Compute the lateral interference term from equation 12.3.

        For each lemma l, the interference is the weighted sum of every
        other lemma's activation, weighted by the concept-space cosine
        similarity between the lemma pair. The result is negative
        (suppressive) and scaled by kappa_interfere.

        PLACEHOLDER: the similarity kernel is approximated by cosine
        similarity over a fixed random projection of the per-lemma
        concept reconstructions. The reconstructions are obtained by
        running each lemma's activation back through the tied substrate
        in the comprehension direction, which gives a concept-space
        signature for each currently-active lemma.

        Args:
            a: (B, n_lemmas) current lemma activations.
            c_lex: (B, n_concepts) most recent concept input.

        Returns:
            (B, n_lemmas) interference contributions, negative-valued.
        """
        # Project each lemma's identity into the similarity space. We use
        # the rows of the tied W_C_to_L matrix as concept-space signatures
        # for each lemma; W_C_to_L[l, :] is what concept distribution
        # would produce strong activation of lemma l, which is a
        # reasonable proxy for "the concept signature of lemma l."
        # NOT a biological quantity in this exact form. PLACEHOLDER for
        # the learned similarity kernel.
        lemma_concept_signatures = self.w_c_to_l.W  # (n_lemmas, n_concepts)
        sim_space = lemma_concept_signatures @ self._concept_embedding_for_sim
        # (n_lemmas, sim_dim)

        # Cosine similarity matrix between lemma signatures. Shape
        # (n_lemmas, n_lemmas).
        sim_norm = torch.nn.functional.normalize(sim_space, dim=1)
        sim_matrix = sim_norm @ sim_norm.t()

        # Zero out the diagonal so a lemma does not interfere with itself.
        n_lemmas = sim_matrix.shape[0]
        sim_matrix = sim_matrix - torch.eye(
            n_lemmas, device=sim_matrix.device, dtype=sim_matrix.dtype,
        )

        # Equation 12.3: interference = -kappa * sum_{j!=l} a_j * sim(C_l, C_j).
        # Implemented as a matrix-vector product where for each lemma l,
        # we multiply the row sim_matrix[l, :] by the activations a and
        # sum.
        # Shape gymnastics: a is (B, n_lemmas); sim_matrix is
        # (n_lemmas, n_lemmas). We want for each batch row b and each
        # lemma l, the sum over j of a[b, j] * sim_matrix[l, j]. That
        # is a @ sim_matrix.t().
        interference_strength = a @ sim_matrix.t()
        return -self.cfg.kappa_interfere * interference_strength

    def forward_comprehension(self, a_lemma_input: Tensor) -> Tensor:
        """Comprehension direction: lemma activation to concept distribution.

        Runs the tied substrate in the reverse direction. Used by the
        Cognitive Kernel boundary to receive the substrate's reconstructed
        conceptual interpretation of incoming language.

        Args:
            a_lemma_input: (B, n_lemmas) lemma activation, typically
                produced by Wernicke's perception direction from
                incoming phonological codes.

        Returns:
            (B, n_concepts) reconstructed concept distribution to hand
                back to the kernel.
        """
        if not self.cfg.enable_mid_mtg:
            return torch.zeros(
                a_lemma_input.shape[0], self.cfg.n_concepts,
                device=a_lemma_input.device, dtype=a_lemma_input.dtype,
            )
        return self.w_c_to_l.forward_b_to_a(a_lemma_input)

    def select_lemma(self) -> Optional[Tensor]:
        """Equation 12.2: select the winning lemma if the integration
        window has elapsed.

        Returns the argmax over the persistent lemma activation, expressed
        as a one-hot indicator of shape (B, n_lemmas), if at least
        t_lemma_steps integration ticks have happened since the last
        reset. Otherwise returns None to indicate selection is not yet
        available.

        Returns:
            (B, n_lemmas) one-hot tensor of selected lemmas, or None
                if selection is not yet available.
        """
        if not self.cfg.enable_mid_mtg:
            return None
        if self._steps_since_reset.item() < self.cfg.t_lemma_steps:
            return None

        winner = self.a_lemma.argmax(dim=1)
        one_hot = torch.zeros_like(self.a_lemma)
        one_hot.scatter_(1, winner.unsqueeze(1), 1.0)
        return one_hot

    # ---------------------------------------------------------------
    # Confidence signal
    # ---------------------------------------------------------------

    def get_lemma_confidence(self) -> Tensor:
        """Compute the lemma confidence signal from Section 24.2.3 of v2.

        The signal corresponds to "I do not have a word for this concept"
        when low. It is computed in two stages.

        First, identify the peak lemma (argmax over absolute activation).
        If the peak lemma is unallocated, the substrate has no acquired
        lemma matching the input and confidence is at the floor. This is
        the Phase 8 case from the cold-start dialogue: the substrate is
        asked "what is the world," has no lemma allocated for WORLD, and
        the peak lands on whatever random unallocated slot has the
        largest activation by chance, which is not a real lemma.

        Second, when the peak lemma is allocated, compute the
        peak-to-average ratio restricted to allocated slots and
        normalize against a target ratio. This is the Phase 7 case: the
        substrate has acquired lemma_513 for TIMMY, the peak lands on
        that allocated slot, and the activation distribution reflects
        whether the substrate has retrieved cleanly.

        Reference: v2 Spec Section 24.2.3.

        Returns:
            (B,) tensor of lemma confidence values in [0, 1].
        """
        if not self.cfg.enable_mid_mtg:
            return torch.zeros(self.a_lemma.shape[0])

        # Operate on the absolute value of activations.
        abs_a = self.a_lemma.abs()
        peak_idx = abs_a.argmax(dim=1)  # (B,)

        # Check whether each batch row's peak landed on an allocated
        # lemma. A peak on an unallocated slot floors confidence.
        peak_allocated = self.is_allocated[peak_idx]  # (B,) bool

        # Peak-to-average ratio restricted to allocated slots. If no
        # slots are allocated at all, the ratio is undefined and we
        # floor confidence; this is the cold-start case before any
        # reservation has happened.
        n_allocated = self.is_allocated.sum().item()
        if n_allocated == 0:
            return torch.zeros(
                abs_a.shape[0], device=abs_a.device, dtype=abs_a.dtype,
            )

        allocated_mask = self.is_allocated.unsqueeze(0).to(abs_a.dtype)
        # (1, n_lemmas)

        # Average over allocated slots only. Unallocated slots
        # contribute zero through the mask, and the divisor counts only
        # allocated slots.
        allocated_sum = (abs_a * allocated_mask).sum(dim=1)
        avg_allocated = (allocated_sum / n_allocated).clamp(
            min=self.cfg.confidence_floor
        )

        # Peak over allocated slots. Unallocated slots are masked to
        # negative infinity so they cannot win the max.
        masked_a = abs_a.masked_fill(~self.is_allocated.unsqueeze(0), -1.0)
        peak_allocated_value = masked_a.max(dim=1).values
        # If no allocated slot has a positive activation, the peak
        # value is -1.0 (the mask fill); clamp to zero so the ratio
        # is well-defined.
        peak_allocated_value = peak_allocated_value.clamp(min=0.0)

        ratio = peak_allocated_value / avg_allocated

        # Normalize to [0, 1] using a target ratio of 10.0 as the
        # confident-retrieval anchor.
        # NOT a biological quantity. Engineering normalization for the
        # epistemic monitor.
        target_ratio = 10.0
        confidence = (ratio / target_ratio).clamp(max=1.0)

        # Floor confidence to zero when the peak landed on an
        # unallocated slot.
        confidence = confidence * peak_allocated.to(confidence.dtype)

        return confidence

    def allocate_lemma(self, lemma_idx: int) -> None:
        """Mark a lemma slot as allocated.

        Called by the acquisition pipeline when a new lemma is allocated
        in response to a novel concept-phonology pairing (the rule from
        the Boundary Contract Section 2). This is also used in tests to
        simulate post-acquisition state without running the full
        pipeline.

        Args:
            lemma_idx: integer slot index in [0, n_lemmas).
        """
        if not (0 <= lemma_idx < self.cfg.n_lemmas):
            raise IndexError(
                f"Lemma index {lemma_idx} out of range "
                f"[0, {self.cfg.n_lemmas})."
            )
        self.is_allocated[lemma_idx] = True

    # ---------------------------------------------------------------
    # State management
    # ---------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset the persistent activation and step counter to zero.

        Called between unrelated sessions to prevent activation from one
        utterance from contaminating the next. The .soul checkpoint
        captures the activation state for cross-session continuity; this
        method is for within-session resets between utterances.
        """
        self.a_lemma.zero_()
        self._last_concept.zero_()
        self._steps_since_reset.zero_()

    def reset_for_selection(self) -> None:
        """Reset only the step counter, not the activation.

        Called at the start of each selection episode so that the
        t_lemma_steps gating works correctly. Activation is preserved
        because the production loop may want to pick up where a previous
        episode left off (for chained productions).
        """
        self._steps_since_reset.zero_()

    def serialize(self) -> dict:
        """Serialize mid-MTG state for the .soul checkpoint.

        COLD-tier: the W_C_to_L weight matrix, the similarity-kernel
            random projection, and the allocation flags (which slots
            have been written to during acquisition or reservation).
        WARM-tier: the persistent lemma activation, the last concept
            distribution, the step counter.

        Returns:
            dict with sub-dicts "cold" and "warm".
        """
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
        """Restore mid-MTG state from a .soul checkpoint.

        Args:
            state: dict from a previous serialize() call.
        """
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
