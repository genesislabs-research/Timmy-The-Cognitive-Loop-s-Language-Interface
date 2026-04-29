"""
frame_recognizer_t.py
Teaching Frame Recognizer: Surface-Pattern Detector for Conversational
Frames That Drive Acetylcholine and the Conceptual Stratum's Frame Bias

BIOLOGICAL GROUNDING
====================
This file implements the surface-level pattern matcher that recognizes
teaching frames in the instructor's input phonological stream before
the input reaches mid-MTG. The biological commitment is that
encoding-mode acetylcholine rises when the conversational frame is
recognized as teaching-shaped, and that the recognized frame
contributes a top-down bias to the conceptual stratum so the concept
vector at the moment of allocation reflects the frame structure
rather than just the bare semantic content of the words.

The architect's Phase 3 spec specifies the inventory and behavior of
this recognizer in detail. The naming_self frame ("your name is X")
is the load-bearing case for the cold-start dialogue because it is
what fires the allocation gate when the substrate is told its name
for the first time. The other frames in the inventory are inexpensive
to add and they enable the immediately-following vocabulary stages
without further work.

The recognizer is rule-based rather than learned. The rationale is
the same as for the confirmation detector: rule-based is sufficient
for the small fixed inventory of cold-start frames, and a learned
classifier can replace this module without changing the architectural
shape of the equation in Section 25.8. The substitution is local.

The recognizer is structurally separate from the existing
FrameRecognizer in the lemma_acquisition module. The existing
recognizer compares concept-vector templates against the conceptual
stratum's current state, which is a post-mid-MTG operation that runs
after the input has been transduced into concepts. This recognizer
runs on the surface tokens before transduction, which is what the
architect's spec calls for: the recognized frame influences the
allocation event itself by biasing the concept vector, so the
recognition has to happen before the allocation gate fires.

Two outputs flow from a successful match. The first is a scalar ACh
amplitude written to the NeuromodulatorBus, which raises encoding
mode and lets the allocation gate at mid-MTG fire. The second is a
frame-specific bias vector that the runtime adds to the conceptual
stratum's input on the next forward pass. The bias is a one-hot
activation on a reserved dimension in the frame-bias subspace at
concept dimensions 1000 through 1015. Different frames activate
different dimensions so the substrate can distinguish a naming
allocation from a vocabulary allocation from a definition allocation
without any of them collapsing into the same lemma.

Primary grounding papers:

Goffman, E. (1974). "Frame Analysis: An Essay on the Organization of
Experience." Harvard University Press, Cambridge MA. (Foundational
work on interaction frames; predates DOI system.)

Hagoort, P. (2014). "Nodes and networks in the neural architecture
for language: Broca's region and beyond." Current Opinion in
Neurobiology, 28, 136-141. DOI: 10.1016/j.conb.2014.07.013.

Hasselmo, M.E. (2006). "The role of acetylcholine in learning and
memory." Current Opinion in Neurobiology, 16(6), 710-715.
DOI: 10.1016/j.conb.2006.09.002. (Encoding-mode acetylcholine
biases cortical processing toward feedforward sensory drive and
enhanced LTP, which is the architectural target of the ACh amplitude
emitted by this recognizer.)

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from coordination.confirmation_detector_t import tokenize


# =========================================================================
# Frame-bias subspace
# =========================================================================
#
# Reserved dimensions in the conceptual stratum where the frame-specific
# bias is delivered. The architect's reservation places frame-bias at
# concept-space dimensions 1000 through 1015 with one-hot activations
# per frame name. Five frame names are currently in use; the remaining
# dimensions are reserved for future frame additions.

FRAME_BIAS_SUBSPACE_START: int = 1000
FRAME_BIAS_SUBSPACE_END: int = 1015  # inclusive

FRAME_BIAS_DIMS: Dict[str, int] = {
    "naming_frame": 1000,
    "vocabulary_frame": 1001,
    "definition_frame": 1002,
    "correction_frame": 1003,
    "confirmation_frame": 1004,
}


# =========================================================================
# Wildcard token
# =========================================================================
#
# Patterns use the literal string "<X>" or "<Y>" or similar to indicate
# a wildcard slot that matches exactly one token. The matching code
# treats any token starting with "<" and ending with ">" as a wildcard,
# so additional wildcard names ("<NAME>", "<WORD>") work without
# changes if the patterns add them.

def _is_wildcard(token: str) -> bool:
    """Return True if a pattern token is a wildcard."""
    return (
        len(token) >= 2
        and token.startswith("<")
        and token.endswith(">")
    )


# =========================================================================
# Frame definition
# =========================================================================

@dataclass
class TeachingFrame:
    """One entry in the teaching frame inventory.

    Attributes:
        name: stable identifier for the frame.
        pattern: token list with optional wildcards. Wildcards match
            exactly one input token each.
        bias_dim_name: key into FRAME_BIAS_DIMS. The runtime applies a
            one-hot bias on the corresponding concept-space dimension
            when this frame fires.
        ach_amplitude: ACh scalar amplitude in [0, 1] to write to the
            NeuromodulatorBus. Higher amplitude means stronger
            encoding-mode biasing of the substrate's allocation gate.
    """

    name: str
    pattern: List[str]
    bias_dim_name: str
    ach_amplitude: float


TEACHING_FRAME_INVENTORY: List[TeachingFrame] = [
    TeachingFrame(
        name="vocabulary_intro",
        pattern=["the", "word", "for", "<X>", "is", "<Y>"],
        bias_dim_name="vocabulary_frame",
        ach_amplitude=0.90,
    ),
    TeachingFrame(
        name="correction",
        pattern=["no", "<X>", "is"],
        bias_dim_name="correction_frame",
        ach_amplitude=0.95,
    ),
    TeachingFrame(
        name="naming_self",
        pattern=["your", "name", "is", "<X>"],
        bias_dim_name="naming_frame",
        ach_amplitude=0.85,
    ),
    TeachingFrame(
        name="naming_other",
        pattern=["my", "name", "is", "<X>"],
        bias_dim_name="naming_frame",
        ach_amplitude=0.85,
    ),
    TeachingFrame(
        name="definition",
        pattern=["<X>", "means", "<Y>"],
        bias_dim_name="definition_frame",
        ach_amplitude=0.85,
    ),
    TeachingFrame(
        name="vocabulary_call",
        pattern=["this", "is", "a", "<X>"],
        bias_dim_name="vocabulary_frame",
        ach_amplitude=0.75,
    ),
    TeachingFrame(
        name="confirmation_pos",
        pattern=["yes", "<X>"],
        bias_dim_name="confirmation_frame",
        ach_amplitude=0.40,
    ),
]


# =========================================================================
# Recognition result
# =========================================================================

@dataclass
class FrameRecognitionResult:
    """Outcome of a single recognize_frame call.

    Attributes:
        recognized: True if any frame matched.
        frame_name: name of the matching frame, or None.
        bias_dim_name: bias dimension key, or None.
        bias_dim_index: concept-space index for the bias, or None.
        ach_amplitude: ACh scalar to fire, 0.0 if no match.
        wildcard_bindings: dict mapping wildcard token (e.g. "<X>")
            to the input token that filled it. Useful downstream for
            associating a novel phonological code with the recognized
            frame.
    """

    recognized: bool = False
    frame_name: Optional[str] = None
    bias_dim_name: Optional[str] = None
    bias_dim_index: Optional[int] = None
    ach_amplitude: float = 0.0
    wildcard_bindings: Dict[str, str] = field(default_factory=dict)


# =========================================================================
# Pattern matching
# =========================================================================

def _match_pattern_at(
    input_tokens: List[str],
    pattern: List[str],
    start: int,
) -> Optional[Dict[str, str]]:
    """Try to match `pattern` against `input_tokens[start:]`.

    Returns a dict of wildcard bindings on success, or None on failure.
    The match requires exactly len(pattern) tokens starting at `start`.
    Each non-wildcard pattern token must equal the corresponding input
    token. Each wildcard pattern token matches any input token and
    binds the input token under its wildcard name.

    Args:
        input_tokens: the instructor's input as a token list.
        pattern: the frame's token pattern.
        start: starting index in input_tokens.

    Returns:
        dict of {wildcard_name: input_token} on success, None on
            failure.
    """
    if start + len(pattern) > len(input_tokens):
        return None
    bindings: Dict[str, str] = {}
    for i, ptok in enumerate(pattern):
        itok = input_tokens[start + i]
        if _is_wildcard(ptok):
            # First binding wins for repeated wildcards.
            if ptok not in bindings:
                bindings[ptok] = itok
            else:
                # If the wildcard is repeated and the second
                # occurrence does not match the binding, fail.
                if bindings[ptok] != itok:
                    return None
        else:
            if itok != ptok:
                return None
    return bindings


def _match_frame(
    input_tokens: List[str],
    frame: TeachingFrame,
) -> Optional[Dict[str, str]]:
    """Try to match a frame's pattern at any position in the input.

    Returns the wildcard bindings on the first matching position, or
    None if no position matches.

    Args:
        input_tokens: tokenized input.
        frame: the frame to try.

    Returns:
        wildcard-bindings dict or None.
    """
    if not input_tokens or not frame.pattern:
        return None
    for start in range(len(input_tokens) - len(frame.pattern) + 1):
        bindings = _match_pattern_at(
            input_tokens, frame.pattern, start,
        )
        if bindings is not None:
            return bindings
    return None


# =========================================================================
# TeachingFrameRecognizerConfig
# =========================================================================

@dataclass
class TeachingFrameRecognizerConfig:
    """Configuration for the teaching frame recognizer.

    Master flag is first per the Genesis Labs ablation flag standard.
    NOT a biological quantity.

    Attributes:
        enable_frame_recognizer: master flag. False produces a
            recognizer that never matches any frame.
        enable_ach_emission: when False, the recognizer matches
            frames and produces bias vectors but does not write to
            the NeuromodulatorBus. Useful for testing the substrate's
            allocation behavior without ACh modulation.
        enable_bias_emission: when False, the recognizer matches
            frames and writes ACh but does not produce a bias
            vector. Useful for testing whether ACh alone is
            sufficient to fire the allocation gate.
        n_concepts: dimensionality of the conceptual stratum. The
            bias vector has this length, with a one-hot activation on
            the bias_dim_index for the matched frame.
        bias_amplitude: magnitude of the one-hot activation in the
            bias vector. Default 1.0 means the bias dimension takes
            unit activation when the frame is active. NOT a
            biological quantity, training artifact only.
    """

    enable_frame_recognizer: bool = True
    enable_ach_emission: bool = True
    enable_bias_emission: bool = True
    n_concepts: int = 1024
    bias_amplitude: float = 1.0


# =========================================================================
# TeachingFrameRecognizer
# =========================================================================

class TeachingFrameRecognizer:
    """Surface-pattern recognizer for teaching frames.

    BIOLOGICAL STRUCTURE: Sits between the instructor's input and the
    conceptual stratum at mid-MTG. Models the early-attention
    processing in Wernicke's area and adjacent posterior temporal
    cortex that detects conversational frame structure before the
    full lexical-conceptual transduction completes.

    BIOLOGICAL FUNCTION: Recognizes a small inventory of teaching
    frames in the input phonological stream and produces two outputs:
    a scalar ACh amplitude that the NeuromodulatorBus broadcasts to
    drive encoding mode at the allocation gate, and a frame-specific
    bias vector that the runtime adds to the conceptual stratum's
    input on the next forward pass. The bias dimension is one-hot
    in the frame-bias subspace at concept dimensions 1000-1015.

    Reference: Equation 25.8 of the Broca's corpus. Architect's
    Phase 3 spec, item 2.

    INTERFACE CONTRACT:
        Inputs:
            recognize_frame(input_text): called by the input
                processor on each instructor turn. Returns a
                FrameRecognitionResult and, on match, has already
                fired the ACh amplitude into the NeuromodulatorBus.
                The returned bias vector is for the runtime to add
                to the conceptual stratum.
            get_bias_vector(frame_name): convenience method that
                returns the bias vector for a given frame name
                without doing any pattern matching. Used by the
                runtime when it already knows which frame is active
                from a previous turn (frame persistence within a
                turn).

        State: stateless. The recognizer makes no assumptions about
            inter-turn frame persistence; the runtime is responsible
            for tracking which frame is active across the current
            turn.
    """

    def __init__(
        self,
        cfg: TeachingFrameRecognizerConfig,
        neuromodulator_bus: Optional[Any] = None,
    ) -> None:
        """Initialize the recognizer with optional bus reference.

        Args:
            cfg: TeachingFrameRecognizerConfig.
            neuromodulator_bus: optional NeuromodulatorBus instance
                whose set("ACh_inc", value) method fires the ACh
                amplitude. When None, the recognizer matches frames
                and returns results but does not write ACh.
        """
        self.cfg = cfg
        self.bus = neuromodulator_bus

    # ---------------------------------------------------------------
    # Bias-vector construction
    # ---------------------------------------------------------------

    def _make_bias_vector(self, bias_dim_name: str) -> torch.Tensor:
        """Construct a one-hot bias vector for a frame.

        Returns a (n_concepts,) tensor with a single non-zero entry
        at the bias dimension corresponding to bias_dim_name. Other
        dimensions are zero. The returned tensor is detached and on
        CPU; the runtime is responsible for moving it to the right
        device if needed.

        Args:
            bias_dim_name: key into FRAME_BIAS_DIMS.

        Returns:
            (n_concepts,) tensor.

        Raises:
            KeyError: if bias_dim_name is not in FRAME_BIAS_DIMS.
        """
        if bias_dim_name not in FRAME_BIAS_DIMS:
            raise KeyError(
                f"Unknown bias_dim_name '{bias_dim_name}'. "
                f"Expected one of {sorted(FRAME_BIAS_DIMS.keys())}."
            )
        bias = torch.zeros(self.cfg.n_concepts)
        bias[FRAME_BIAS_DIMS[bias_dim_name]] = self.cfg.bias_amplitude
        return bias

    def get_bias_vector(self, frame_name: str) -> torch.Tensor:
        """Return the bias vector for a frame without matching.

        Used by the runtime when it already knows which frame is
        active and just needs the bias vector to add to the
        conceptual stratum on subsequent turns within the same
        frame's persistence window.

        Args:
            frame_name: name of an entry in TEACHING_FRAME_INVENTORY.

        Returns:
            (n_concepts,) tensor with one-hot at the frame's bias
                dimension.

        Raises:
            KeyError: if frame_name is not in the inventory.
        """
        for frame in TEACHING_FRAME_INVENTORY:
            if frame.name == frame_name:
                return self._make_bias_vector(frame.bias_dim_name)
        raise KeyError(
            f"Unknown frame name '{frame_name}'. "
            f"Expected one of "
            f"{sorted(f.name for f in TEACHING_FRAME_INVENTORY)}."
        )

    # ---------------------------------------------------------------
    # Recognition entry point
    # ---------------------------------------------------------------

    def recognize_frame(
        self, input_text: str,
    ) -> FrameRecognitionResult:
        """Process an instructor input and recognize a teaching frame.

        The recognition sequence:
        1. Master flag check. If disabled, return a no-match result.
        2. Tokenize the input.
        3. Try each frame in TEACHING_FRAME_INVENTORY in order.
           The inventory is ordered most-specific first so that
           longer patterns ('the word for X is Y') match before
           shorter patterns ('this is a X') that they would also
           accidentally satisfy.
        4. On the first match, build the bias vector, fire ACh into
           the bus if enabled, and return the result.
        5. If no frame matches, return a no-match result.

        Args:
            input_text: raw instructor input string.

        Returns:
            FrameRecognitionResult.
        """
        if not self.cfg.enable_frame_recognizer:
            return FrameRecognitionResult()

        input_tokens = tokenize(input_text)
        if not input_tokens:
            return FrameRecognitionResult()

        for frame in TEACHING_FRAME_INVENTORY:
            bindings = _match_frame(input_tokens, frame)
            if bindings is None:
                continue

            if self.cfg.enable_ach_emission:
                self._fire_ach(frame.ach_amplitude)

            return FrameRecognitionResult(
                recognized=True,
                frame_name=frame.name,
                bias_dim_name=frame.bias_dim_name,
                bias_dim_index=FRAME_BIAS_DIMS[frame.bias_dim_name],
                ach_amplitude=frame.ach_amplitude,
                wildcard_bindings=dict(bindings),
            )

        return FrameRecognitionResult()

    def recognize_and_get_bias(
        self, input_text: str,
    ) -> tuple:
        """Convenience method returning (result, bias_vector).

        Equivalent to calling recognize_frame and then constructing
        the bias vector from the result. Returns a zero bias vector
        if no frame matches or if bias emission is disabled.

        Args:
            input_text: raw instructor input string.

        Returns:
            (FrameRecognitionResult, bias_vector). bias_vector is a
                (n_concepts,) tensor.
        """
        result = self.recognize_frame(input_text)
        if (
            not result.recognized
            or not self.cfg.enable_bias_emission
            or result.bias_dim_name is None
        ):
            return result, torch.zeros(self.cfg.n_concepts)
        bias = self._make_bias_vector(result.bias_dim_name)
        return result, bias

    # ---------------------------------------------------------------
    # Side-effect helper
    # ---------------------------------------------------------------

    def _fire_ach(self, amplitude: float) -> None:
        """Write an ACh_inc scalar to the NeuromodulatorBus.

        The incremental ACh pathway from nucleus basalis to neocortex
        is the one that drives encoding mode (Hasselmo 2006). The
        recognizer writes to ACh_inc rather than ACh_dec because the
        teaching-frame recognition signals that new content is
        arriving and should be encoded, not that retrieval should
        dominate.

        Args:
            amplitude: scalar in [0, 1] to write.
        """
        if self.bus is None:
            return
        if not hasattr(self.bus, "set"):
            return
        self.bus.set("ACh_inc", torch.tensor(float(amplitude)))
