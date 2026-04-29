"""
tied_substrate_t.py
The Tied Substrate Primitive: One Matrix, Two Directions

BIOLOGICAL GROUNDING
====================
This file implements the keystone primitive of the v2 Broca's pathway. It does
not model a single brain region. It models a property that several regions of
the language network share: the substrate that maps representation A to
representation B in production is the same substrate that maps B back to A in
comprehension. The matrix is shared, accessed bidirectionally, and trained by
both directions simultaneously.

The empirical case for this property is strongest at three boundaries in the
Broca's corpus:

    1. Concept to lemma in mid-MTG. The mid-section of left middle temporal
       gyrus activates during both picture naming (production: concept to
       lemma) and word listening (comprehension: lemma activation drives
       conceptual reconstruction). The Indefrey and Levelt 2004 meta-analysis
       of 82 word production experiments together with comprehension data
       puts the same anatomical substrate on both sides.

    2. Lemma to phonological code in Wernicke's. Left posterior superior
       temporal gyrus activates during picture naming, word reading, word
       generation, and word listening, but not during pseudoword reading or
       pseudoword listening. The diagnostic identifies it specifically as the
       lexical phonological code substrate, accessed in both directions.

    3. Syllable structure in Broca's during reading. The cluster operator
       that builds syllables from segments in production is the same operator
       that runs in reverse for orthographic input during reading.

The architectural commitment is that these are not separate production and
perception modules with weights that happen to be aligned. They are single
matrices accessed bidirectionally. Building two matrices and trying to align
them through gradient flow is forbidden by the v2 spec because the alignment
is structural, not emergent.

The neuroscience behind the commitment is the picture-word interference
paradigm. When a participant names a picture while hearing a phonologically
related distracter word, the distracter facilitates rather than impairs naming
at SOA 0 to 150 ms. This falls out naturally if the perception of the
distracter pre-activates the phonological code that the production pathway is
about to retrieve. With separate production and perception substrates, the
facilitation has to be engineered through additional cross-pathway machinery.
With tied weights, it is the default behavior.

Primary grounding papers:

Indefrey P, Levelt WJM (2004). "The spatial and temporal signatures of word
production components." Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001

Levelt WJM, Roelofs A, Meyer AS (1999). "A theory of lexical access in speech
production." Behavioral and Brain Sciences, 22(1), 1-75.
DOI: 10.1017/S0140525X99001776

Hickok G, Poeppel D (2007). "The cortical organization of speech processing."
Nature Reviews Neuroscience, 8(5), 393-402. DOI: 10.1038/nrn2113

ENGINEERING NOTE
================
The tying must be mechanical, not emergent. Two nn.Linear layers with the
same shape are not tied; they are independent and will diverge during
training. Tying is achieved here by declaring a single nn.Parameter on the
module and referencing it in both forward methods, with the comprehension
direction using .t() to transpose without creating a new parameter.

The receptor modulation argument scales the effective weight by a
region-and-receptor-specific factor from the routing table of the main
corpus subsection 10a. This is how D1 versus D2 dopamine effects, M1
versus M2 acetylcholine effects, and the BA45 versus BA44 receptor-density
differences become local rather than global modulations. The modulation
multiplies the effective weight at forward time without altering the
underlying parameter.

Receptor scaling references:

Avery MC, Krichmar JL (2017). "Neuromodulatory systems and their
interactions: a review of models, theories, and experiments." Frontiers in
Neural Circuits, 11, 108. DOI: 10.3389/fncir.2017.00108

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class TiedSubstrateConfig:
    """Configuration for a TiedSubstrate instance.

    The master flag follows the cognitive-loop ablation flag standard.
    Setting enable_tied_substrate to False forces both forward methods to
    return the neutral value (zeros), which is the standard ablation
    behavior.

    NOT a biological quantity. Engineering convention for ablation studies.

    Reference: Pragmi-Cognitive-Loop_v1 README, ablation flag standard.

    Attributes:
        enable_tied_substrate: master flag for the whole module.
        in_dim: input dimensionality of the A-to-B direction.
        out_dim: output dimensionality of the A-to-B direction.
        init_scale: standard deviation for Gaussian initialization of W.
            NOT a biological quantity. Standard PyTorch initialization
            practice for linear layers.
    """

    enable_tied_substrate: bool = True
    in_dim: int = 64
    out_dim: int = 64
    init_scale: float = 0.02


# =========================================================================
# The TiedSubstrate Primitive
# =========================================================================

class TiedSubstrate(nn.Module):
    """One weight matrix, two forward directions.

    BIOLOGICAL STRUCTURE: A bidirectional cortical projection between two
    populations. The same axonal connection that carries production-direction
    activity from the source population to the target population can also
    carry comprehension-direction activity in the reverse direction (through
    feedback fibers running alongside the feedforward projection).

    BIOLOGICAL FUNCTION: Implements the architectural commitment that the
    substrate that maps concept to lemma in production is the same substrate
    that maps lemma to concept in comprehension. The shared matrix W is
    accessed in both directions: forward_a_to_b applies W, forward_b_to_a
    applies W transposed. Both directions backpropagate into the same
    parameter, which is what makes the tied-weights training objective work.

    Reference: Indefrey P, Levelt WJM (2004). DOI: 10.1016/j.cognition.2002.06.001

    ANATOMICAL INTERFACE (production direction):
        Sending structure: Source population in the A-dimensional space
            (e.g., conceptual stratum upstream of mid-MTG).
        Receiving structure: Target population in the B-dimensional space
            (e.g., lemma stratum in mid-MTG).
        Connection: Feedforward cortical projection.

    ANATOMICAL INTERFACE (comprehension direction):
        Sending structure: Target population in the B-dimensional space.
        Receiving structure: Source population in the A-dimensional space.
        Connection: Feedback cortical projection running alongside the
            feedforward projection. Same axonal substrate, reverse direction.

    STATE: The substrate itself is stateless except for the weight. Persistent
    state (membrane potentials, synaptic currents) lives in the regions that
    hold TiedSubstrate instances, not in the substrate itself. This separation
    is what allows the same TiedSubstrate to be reused at multiple boundaries
    in the architecture without coupling their dynamics.
    """

    def __init__(self, cfg: TiedSubstrateConfig) -> None:
        """Initialize the substrate with a single weight matrix.

        The weight matrix is shape (out_dim, in_dim) following PyTorch's
        nn.Linear convention. The forward_a_to_b direction applies W, the
        forward_b_to_a direction applies W.t(). Both directions reference
        the same nn.Parameter.

        Args:
            cfg: TiedSubstrateConfig with in_dim, out_dim, init_scale.
        """
        super().__init__()
        self.cfg = cfg

        # Single weight matrix. Declared once. Referenced in both forward
        # methods. The tying is mechanical: there is one nn.Parameter, and
        # any gradient that flows through either forward method updates
        # this same parameter.
        # Reference: PyTorch nn.Parameter sharing pattern.
        self.W = nn.Parameter(
            torch.randn(cfg.out_dim, cfg.in_dim) * cfg.init_scale
        )

    def forward_a_to_b(
        self,
        x_a: Tensor,
        receptor_modulation: Optional[Tensor] = None,
    ) -> Tensor:
        """Production direction: map A-space input to B-space output.

        Applies the weight matrix W to project from in_dim to out_dim.
        Optional receptor modulation scales the effective weight at forward
        time without altering the underlying parameter.

        Args:
            x_a: (B, in_dim) input tensor in A-space.
            receptor_modulation: optional scalar tensor modulating the
                effective weight. Default 1.0 (no modulation). Values
                typically range 0.3 to 2.0 with 1.0 representing
                substrate-wide average.

        Returns:
            (B, out_dim) output tensor in B-space.
        """
        if not self.cfg.enable_tied_substrate:
            return torch.zeros(
                x_a.shape[0], self.cfg.out_dim,
                device=x_a.device, dtype=x_a.dtype,
            )

        if receptor_modulation is None:
            effective_W = self.W
        else:
            # Receptor modulation scales the effective weight without
            # touching the parameter itself. The gradient flows through
            # the modulated form during backpropagation but updates the
            # underlying W only.
            # Reference: Avery MC, Krichmar JL (2017). DOI: 10.3389/fncir.2017.00108
            effective_W = self.W * receptor_modulation

        return torch.matmul(x_a, effective_W.t())

    def forward_b_to_a(
        self,
        x_b: Tensor,
        receptor_modulation: Optional[Tensor] = None,
    ) -> Tensor:
        """Comprehension direction: map B-space input to A-space output.

        Applies the transposed weight matrix W.t() to project from out_dim
        back to in_dim. The matrix is the same parameter as forward_a_to_b;
        only the direction of access differs. This is the mechanical tying
        commitment.

        Args:
            x_b: (B, out_dim) input tensor in B-space.
            receptor_modulation: optional scalar tensor modulating the
                effective weight. Default 1.0 (no modulation).

        Returns:
            (B, in_dim) output tensor in A-space.
        """
        if not self.cfg.enable_tied_substrate:
            return torch.zeros(
                x_b.shape[0], self.cfg.in_dim,
                device=x_b.device, dtype=x_b.dtype,
            )

        if receptor_modulation is None:
            effective_W = self.W
        else:
            effective_W = self.W * receptor_modulation

        # Comprehension direction uses the same matrix, accessed transposed.
        # No new parameter is created; .t() returns a view.
        return torch.matmul(x_b, effective_W)

    def serialize(self) -> dict:
        """Serialize the substrate state for the .soul checkpoint.

        The substrate is parameter-only; no warm or hot state. The COLD
        layer captures W. Returns a dict that the session_state writer
        can embed under this substrate's path in the checkpoint.

        Returns:
            dict with key "W" mapping to the weight tensor (CPU clone).
        """
        return {"W": self.W.detach().cpu().clone()}

    def restore(self, state: dict) -> None:
        """Restore the substrate state from a .soul checkpoint.

        Validates that the saved weight has the expected shape before
        loading. A shape mismatch indicates an architecture mismatch
        and is treated as a hard error rather than a silent partial load.

        Args:
            state: dict from a previous serialize() call.

        Raises:
            ValueError: if the saved weight shape does not match the
                current parameter shape.
        """
        saved_W = state["W"]
        if saved_W.shape != self.W.shape:
            raise ValueError(
                f"TiedSubstrate restore: shape mismatch. Saved {saved_W.shape}, "
                f"current {tuple(self.W.shape)}. Architecture has changed."
            )
        self.W.data = saved_W.to(self.W.device)
