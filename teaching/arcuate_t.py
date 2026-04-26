"""
arcuate_t.py
The Arcuate Fasciculus: Transport Tract from Wernicke's to Broca's

BIOLOGICAL GROUNDING
====================
This file models the arcuate fasciculus, the white-matter tract that
carries spelled-out phonological segments from Wernicke's area to Broca's
area in the language production pathway. It is bidirectional in principle,
but the production-direction load dominates during speech production. The
arcuate is the structure whose damage produces conduction aphasia: the
clinical picture of preserved comprehension, preserved spontaneous
production, but severe repetition deficit and phonemic paraphasias under
repetition. The dissociation works because the arcuate is the specific
substrate that supports the perception-to-production transit needed for
repetition, while the spontaneous production pathway from concept through
Wernicke's already has the segments by the time it reaches Broca's.

The conduction delay is small but not zero. The corpus puts it at 5 to 10
ms, on the order of one or two timesteps at the dt = 5 ms scaffold
default. Treating the conduction time as zero compresses the
syllabification dynamics in Broca's because the arcuate's output timing is
load-bearing for the gradient-order working memory buffer's segment
arrival pattern. The buffer expects segments to arrive at roughly 25 ms
intervals; the arcuate delay places the first segment of each spell-out
emission a few timesteps after Wernicke's emits it.

This file implements equation 14.1 from the Broca's corpus:

    s_Broca(t) = W_arc * s(t - tau_arc)                                (14.1)

ENGINEERING NOTE
================
The arcuate is a connection-spec module, not a substrate population. It
has no Appendix-D state-space neurons. It is a Linear layer wrapping a
deque of length tau_arc / dt. Two design constraints follow from the
spec.

The first constraint is that W_arc must be a frozen near-identity
projection. A fully learnable W_arc would let the substrate route around
the rest of the speech pathway by encoding everything into this one
matrix, which is the routing-around failure mode the v2 spec Section 4.3
explicitly forbids. The scaffold initializes W_arc to the identity matrix
and freezes it (gradient is detached). When biological topology
information becomes available (sparse near-identity with topographic
preservation), the initializer can be replaced without changing the
surrounding code.

The second constraint is that the delay buffer must be in-order and
lossless. Segments emitted at tick t must arrive downstream at exactly
tick t + tau_arc / dt, in the order they were emitted, with no segments
dropped or duplicated. The buffer is a fixed-length deque of zero tensors;
each tick pushes the new segment onto the back and pops the oldest segment
from the front. Conduction failure is not a normal-mode behavior.

Conduction aphasia is implemented by zeroing W_arc through the ablation
flag system (the disorder configuration from Appendix G of the corpus
sets W_arc to zeros), not by perturbing the buffer. The buffer represents
the structural transit; the matrix represents the topographic mapping;
clinical conduction aphasia is a matrix lesion, not a buffer disorder.

Primary grounding papers:

Indefrey P, Levelt WJM (2004). "The spatial and temporal signatures of
word production components." Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001

Catani M, Jones DK, Ffytche DH (2005). "Perisylvian language networks of
the human brain." Annals of Neurology, 57(1), 8-16.
DOI: 10.1002/ana.20319

Bernal B, Ardila A (2009). "The role of the arcuate fasciculus in
conduction aphasia." Brain, 132(9), 2309-2316.
DOI: 10.1093/brain/awp206

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional

import torch
import torch.nn as nn
from torch import Tensor


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class ArcuateConfig:
    """Configuration for the arcuate fasciculus.

    Attributes:
        enable_arcuate: master ablation flag. When False, downstream
            output is zeros, reproducing the conduction aphasia disorder
            configuration from Appendix G of the corpus.
        n_segments: dimensionality of the segment vector being
            transported. Must match Wernicke's n_segments.
        tau_arc_steps: conduction delay in timesteps. From Equation 14.2,
            tau_arc is approximately 5 to 10 ms. With dt = 5 ms (the
            scaffold runtime default), tau_arc_steps = 1 to 2 corresponds
            to the lower and upper ends of that biological range. Default
            is 2 steps (10 ms), the upper end which matches longer
            white-matter tracts in adults.
        identity_jitter: standard deviation of small random perturbation
            added to the identity matrix at initialization. Zero gives
            an exact identity (a clean test case); a small non-zero
            value approximates the fact that biological arcuate fibers
            are not strictly point-to-point. NOT a biological quantity
            in the sense of being measured; small enough not to alter
            the behavior in the scaffold.
    """

    enable_arcuate: bool = True
    n_segments: int = 64
    tau_arc_steps: int = 2
    identity_jitter: float = 0.0


# =========================================================================
# Arcuate Fasciculus
# =========================================================================

class Arcuate(nn.Module):
    """The arcuate fasciculus transport tract.

    BIOLOGICAL STRUCTURE: White-matter tract connecting left posterior
    superior temporal gyrus and posterior middle temporal gyrus
    (Wernicke's area) to left posterior inferior frontal gyrus (Broca's
    area).

    BIOLOGICAL FUNCTION: Carries spelled-out phonological segments from
    Wernicke's spell-out output to Broca's syllabification machinery.
    Bidirectional in principle but production-direction load is dominant
    during speech production. Repetition specifically depends on this
    pathway because repetition routes a perceived word's phonological
    code through the production-direction transit rather than building
    the production from the conceptual source.

    Reference: Catani M, Jones DK, Ffytche DH (2005). DOI: 10.1002/ana.20319

    ANATOMICAL INTERFACE (input):
        Sending structure: Lexical phonological code store in Wernicke's,
            specifically the spell-out output emitting segments at 25 ms
            intervals.
        Receiving structure: Arcuate fasciculus white matter (this module).
        Connection: Spell-out output to arcuate input.

    ANATOMICAL INTERFACE (output):
        Sending structure: Arcuate fasciculus white matter (this module).
        Receiving structure: Syllabification machinery in Broca's area
            (BA44 and BA45).
        Connection: Arcuate output to Broca's gradient-order buffer.

    STATE: Delay buffer holding segments in transit. Length tau_arc_steps.
    Serializes as a list of tensors that the runtime can replay on
    restore.

    DISORDER CONFIGURATIONS:
        Conduction aphasia: zero W_arc (set enable_arcuate to False, or
            replace W_arc with zeros directly). Predicted phenotype:
            preserved comprehension, preserved spontaneous production,
            severe repetition deficit, phonemic paraphasias under
            repetition.
        Reference: Broca's corpus Appendix G.
    """

    def __init__(self, cfg: ArcuateConfig) -> None:
        """Initialize the arcuate with a frozen near-identity projection
        and an empty delay buffer.

        Args:
            cfg: ArcuateConfig.
        """
        super().__init__()
        self.cfg = cfg

        # Frozen near-identity projection. Initialized to the identity
        # matrix plus optional small jitter, then frozen by setting
        # requires_grad to False. The matrix is registered as a buffer
        # rather than a parameter so optimizers do not pick it up.
        # Reference: Equation 14.1 of the Broca's corpus, with the
        # constraint from Section 4.3 that "W_arc must be constrained
        # as a sparse near-identity projection or frozen after
        # initialization with biologically motivated topology. A fully
        # learnable arcuate lets the substrate route around the rest of
        # the speech pathway."
        w_init = torch.eye(cfg.n_segments)
        if cfg.identity_jitter > 0:
            w_init = w_init + torch.randn_like(w_init) * cfg.identity_jitter
        self.register_buffer("W_arc", w_init)

        # Delay buffer. A list of n_segments-dim tensors of length
        # tau_arc_steps. Each tick pushes a new segment onto the back
        # and pops the oldest from the front. Initialized to zeros so
        # downstream sees silence for the first tau_arc_steps ticks
        # after construction.
        # NOT a biological quantity in the sense of being measured;
        # the buffer length tracks the conduction delay parameter.
        # Reference: v2 Spec Section 9 (state serialization). The buffer
        # contents are WARM-tier state that survives a .soul checkpoint.
        self._delay_buffer: Deque[Tensor] = deque(
            [torch.zeros(1, cfg.n_segments) for _ in range(cfg.tau_arc_steps)],
            maxlen=cfg.tau_arc_steps,
        )

    def forward(self, segment: Tensor) -> Tensor:
        """Push a segment into the delay buffer and emit the oldest one.

        This is the per-tick operation. The runtime calls forward once
        per tick with whatever segment Wernicke's spell-out emitted on
        that tick (which may be zeros if the spell-out is between
        emissions). The arcuate returns the segment that entered the
        buffer tau_arc_steps ticks ago.

        Args:
            segment: (B, n_segments) segment vector from Wernicke's
                spell-out. May be zeros when the spell-out is idle.

        Returns:
            (B, n_segments) segment vector arriving at Broca's on this
                tick, projected through the frozen W_arc. May be zeros
                during the initial transient before the buffer has
                filled up with real input.
        """
        if not self.cfg.enable_arcuate:
            return torch.zeros(
                segment.shape[0], self.cfg.n_segments,
                device=segment.device, dtype=segment.dtype,
            )

        # Adapt buffer batch dimension if the runtime changes batch size
        # mid-session. This is a recovery path; in normal operation the
        # batch size is fixed for a session.
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

        # Push the new segment onto the back (will become the oldest in
        # tau_arc_steps ticks). The deque's maxlen enforces fixed length
        # by automatically dropping the front element, but we want to
        # capture and return that dropped element, so we pop explicitly.
        emerging = self._delay_buffer[0]
        self._delay_buffer.popleft()
        self._delay_buffer.append(segment)

        # Apply the frozen W_arc projection. For an exact identity, this
        # is the same as returning emerging directly; the projection is
        # kept in place so the disorder configuration that lesions
        # W_arc can still attenuate the signal.
        return torch.matmul(emerging, self.W_arc.t())

    # ---------------------------------------------------------------
    # State management
    # ---------------------------------------------------------------

    def reset_state(self) -> None:
        """Clear the delay buffer.

        Called between unrelated sessions. Within-session resets are
        not normally needed because the buffer naturally clears
        itself within tau_arc_steps ticks of zero input.
        """
        for slot in self._delay_buffer:
            slot.zero_()

    def serialize(self) -> dict:
        """Serialize arcuate state for the .soul checkpoint.

        COLD-tier: the W_arc projection matrix (frozen, but captured for
            reproducibility and so that a checkpoint from a system with
            jittered initialization restores to the same matrix).
        WARM-tier: the delay buffer contents.

        Returns:
            dict with sub-dicts "cold" and "warm".
        """
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
        """Restore arcuate state from a .soul checkpoint.

        Args:
            state: dict from a previous serialize() call.

        Raises:
            ValueError: if the saved buffer length or W_arc shape does
                not match the current configuration.
        """
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
