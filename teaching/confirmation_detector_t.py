"""
confirmation_detector_t.py
Confirmation Detector: Pattern-Based Recognizer for Conversational
Confirmation and Correction Events

BIOLOGICAL GROUNDING
====================
This file implements the runtime hook that recognizes when a
conversational confirmation or correction event has occurred and
fires the corresponding dopamine event into the NeuromodulatorBus.
The biological commitment is that the provisional-to-confirmed
transition for a freshly-allocated lemma requires a phasic dopamine
signal coincident with the lemma being active above threshold, per
Equation 25.5 of the Broca's corpus and the Schultz 1998 reward
prediction error framing.

In adult conversation between two competent speakers the
confirmation event is rich: prosody, gaze, head nod, and verbal
content all carry confirmation information simultaneously. The v2
substrate operates over text input only, so the detector reduces the
verbal content to a small inventory of explicit confirmation patterns
plus the structural condition that a polar-question with a
provisional lemma was emitted in the immediately prior turn. The
restriction to immediately-prior-turn matters: without it, the
detector would fire on every "yes" in conversation, which would
reinforce arbitrary lemmas. The lemma-coincidence requirement is
what makes the confirmation specific.

The pronoun-flip handling deserves explicit treatment. When the
substrate emits a polar question about itself, it uses the
first-person pronouns MY/I/ME because the substrate is the speaker
in that turn. When the instructor confirms, the instructor refers to
the substrate using the second-person pronouns YOUR/YOU. The
"yes <repeat>" pattern from the architect's spec recognizes the
instructor's repetition of the substrate's emission in declarative
form, and the repetition has the pronouns flipped because the role
of speaker has flipped between the two turns. The detector handles
this by flipping pronouns in the substrate's recorded emission
before doing the substring match against the instructor's input.

The detector also handles the negative-polarity case (correction)
symmetrically. The instructor saying "no" or "wrong" on a
just-emitted provisional lemma fires a negative DA event and decays
the provisional row immediately rather than waiting for the
multi-minute timeout. This is the conversational correction
mechanism that lets a misallocated binding be undone in a single
turn.

Primary grounding papers:

Schultz, W. (1998). "Predictive reward signal of dopamine neurons."
Journal of Neurophysiology, 80(1), 1-27.
DOI: 10.1152/jn.1998.80.1.1.

Holroyd CB, Coles MGH (2002). "The neural basis of human error
processing: reinforcement learning, dopamine, and the error-related
negativity." Psychological Review, 109(4), 679-709.
DOI: 10.1037/0033-295X.109.4.679. (Negative-polarity dopamine
events on error feedback, the reverse-direction analog of
confirmation.)

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


# =========================================================================
# Pronoun flip table
# =========================================================================
#
# Maps each first-person pronoun (substrate's self-reference in production)
# to its second-person counterpart (instructor's reference to the substrate).
# The flip is applied to the substrate's recorded emission before substring
# matching against the instructor's input. The reverse direction is also
# included so the same table works for either direction of flip.

PRONOUN_FLIP: Dict[str, str] = {
    "my": "your",
    "your": "my",
    "mine": "yours",
    "yours": "mine",
    "i": "you",
    "you": "i",
    "me": "you",
    "myself": "yourself",
    "yourself": "myself",
}


def flip_pronouns(tokens: List[str]) -> List[str]:
    """Return a copy of tokens with first-person pronouns flipped to
    second-person and vice versa.

    Used to convert a substrate emission like ["my", "name", "is",
    "timmy"] into the form an instructor would use to confirm it,
    namely ["your", "name", "is", "timmy"].

    Args:
        tokens: lowercase token list.

    Returns:
        new list with pronouns flipped according to PRONOUN_FLIP.
    """
    return [PRONOUN_FLIP.get(t, t) for t in tokens]


def tokenize(text: str) -> List[str]:
    """Lowercase and split on whitespace, stripping punctuation.

    Minimal tokenization sufficient for the rule-based pattern
    matching. The substrate's actual phonological-code stream is what
    drives lemma activation; this tokenizer is only used for
    surface-level pattern recognition in the confirmation detector
    and the frame recognizer.

    Args:
        text: raw input string.

    Returns:
        list of lowercase tokens with leading/trailing punctuation
        removed from each.
    """
    if not text:
        return []
    out: List[str] = []
    for raw in text.strip().split():
        # Strip punctuation from both ends; do not split contractions.
        cleaned = raw.lower().strip(".,?!;:\"'")
        if cleaned:
            out.append(cleaned)
    return out


# =========================================================================
# Confirmation patterns
# =========================================================================

# Each pattern entry specifies the surface tokens that match, the DA
# amplitude to fire, and the polarity. Polarity "+1" is confirmation
# (transition provisional to confirmed); polarity "-1" is correction
# (decay provisional immediately). The "repeat" key is a special
# pattern that matches "yes" plus the substrate's flipped emission;
# the runtime check for it is in detect_confirmation rather than as a
# static token list.

POLARITY_POSITIVE: int = +1
POLARITY_NEGATIVE: int = -1


@dataclass
class ConfirmationPattern:
    """One entry in the confirmation pattern inventory.

    Attributes:
        tokens: token sequence that triggers this pattern. Special
            empty list means the pattern requires the "repeat"
            mechanism rather than a direct token match.
        da_amplitude: dopamine event amplitude in [0, 1] to fire on
            match.
        polarity: POLARITY_POSITIVE or POLARITY_NEGATIVE.
        is_repeat: if True, this pattern matches on substrate-emission
            repetition rather than on a literal token sequence.
    """

    tokens: List[str]
    da_amplitude: float
    polarity: int
    is_repeat: bool = False


# Inventory ordered from most-specific to least-specific. The detector
# scans in order and returns the first match. The "yes <repeat>" pattern
# is checked first because it has the highest amplitude and is the most
# distinctive confirmation form.
CONFIRMATION_PATTERN_INVENTORY: List[Tuple[str, ConfirmationPattern]] = [
    (
        "yes_repeat",
        ConfirmationPattern(
            tokens=[], da_amplitude=0.95,
            polarity=POLARITY_POSITIVE, is_repeat=True,
        ),
    ),
    (
        "thats_right",
        ConfirmationPattern(
            tokens=["that's", "right"], da_amplitude=0.8,
            polarity=POLARITY_POSITIVE,
        ),
    ),
    (
        "that_is_right",
        ConfirmationPattern(
            tokens=["that", "is", "right"], da_amplitude=0.8,
            polarity=POLARITY_POSITIVE,
        ),
    ),
    (
        "thats_wrong",
        ConfirmationPattern(
            tokens=["that's", "wrong"], da_amplitude=0.8,
            polarity=POLARITY_NEGATIVE,
        ),
    ),
    (
        "correct",
        ConfirmationPattern(
            tokens=["correct"], da_amplitude=0.8,
            polarity=POLARITY_POSITIVE,
        ),
    ),
    (
        "wrong",
        ConfirmationPattern(
            tokens=["wrong"], da_amplitude=0.7,
            polarity=POLARITY_NEGATIVE,
        ),
    ),
    (
        "right",
        ConfirmationPattern(
            tokens=["right"], da_amplitude=0.7,
            polarity=POLARITY_POSITIVE,
        ),
    ),
    (
        "yes",
        ConfirmationPattern(
            tokens=["yes"], da_amplitude=0.7,
            polarity=POLARITY_POSITIVE,
        ),
    ),
    (
        "no",
        ConfirmationPattern(
            tokens=["no"], da_amplitude=0.6,
            polarity=POLARITY_NEGATIVE,
        ),
    ),
]


# =========================================================================
# Substrate emission record
# =========================================================================

@dataclass
class SubstrateEmission:
    """Record of a single substrate-side production turn.

    The runtime records each emission the substrate produces so the
    confirmation detector can check the lemma-coincidence requirement
    and the substrate-repetition pattern. The record holds the surface
    token list (used for pronoun-flip matching), the lemma slot
    indices that fired, whether a polar-question marker was
    co-activated, and the slot index of any provisional lemma that
    was active.

    A single emission may have at most one provisional lemma active
    at a time during the cold-start dialogue, and at most one in
    practice during normal operation; the substrate does not emit
    multiple provisional bindings in a single turn because the
    polar-question prime co-activation interrupts the production loop
    after the first provisional lemma fires.

    Attributes:
        tokens: the substrate's surface output as a token list.
        lemma_ids: the slot indices that drove the emission, in
            order.
        polar_question: True if the polar-question marker was
            co-activated for this emission.
        provisional_lemma_id: slot index of the provisional lemma
            that drove the polar-question, or None if no provisional
            lemma was active.
    """

    tokens: List[str] = field(default_factory=list)
    lemma_ids: List[int] = field(default_factory=list)
    polar_question: bool = False
    provisional_lemma_id: Optional[int] = None

    def contains_polar_question(self) -> bool:
        """Return True if this emission was a polar question."""
        return self.polar_question

    def has_provisional_lemma(self) -> bool:
        """Return True if this emission had a provisional lemma."""
        return self.provisional_lemma_id is not None


# =========================================================================
# Detection result
# =========================================================================

@dataclass
class DetectionResult:
    """Outcome of a detect_confirmation call.

    Attributes:
        fired: True if a confirmation event fired.
        pattern_name: name of the matching pattern, or None if no
            match.
        da_amplitude: amplitude of the DA event fired, or 0.0 if no
            match.
        polarity: POLARITY_POSITIVE, POLARITY_NEGATIVE, or 0 if no
            match.
        target_lemma_id: slot index of the lemma that was confirmed
            or decayed, or None if no match.
    """

    fired: bool = False
    pattern_name: Optional[str] = None
    da_amplitude: float = 0.0
    polarity: int = 0
    target_lemma_id: Optional[int] = None


# =========================================================================
# ConfirmationDetectorConfig
# =========================================================================

@dataclass
class ConfirmationDetectorConfig:
    """Configuration for the confirmation detector.

    Master flag is first per the Genesis Labs ablation flag standard.
    NOT a biological quantity.

    Attributes:
        enable_confirmation_detector: master flag. False produces a
            detector that never fires, which is the standard
            ablation behavior.
        enable_positive_polarity: when False, positive confirmations
            do not fire. Used to test that the substrate continues
            asking for confirmation in the absence of positive
            signals.
        enable_negative_polarity: when False, corrections do not
            fire. Used to test the timeout-based decay path
            independently of the explicit-correction path.
        enable_repeat_pattern: when False, the high-amplitude "yes
            <repeat>" pattern is disabled and the detector falls
            back to the lower-amplitude bare "yes" pattern. Used
            to verify that the basic detector works without the
            repeat-recognition refinement.
    """

    enable_confirmation_detector: bool = True
    enable_positive_polarity: bool = True
    enable_negative_polarity: bool = True
    enable_repeat_pattern: bool = True


# =========================================================================
# ConfirmationDetector
# =========================================================================

class ConfirmationDetector:
    """Recognizer for conversational confirmation and correction.

    BIOLOGICAL STRUCTURE: Sits between the conversational input
    (Wernicke's perceptual stream) and the dopaminergic source (VTA
    in the cognitive loop's NeuromodulatorBroadcast). Models the
    cortical processing that interprets a verbal confirmation as a
    reward-prediction-error event for the just-emitted provisional
    lemma.

    BIOLOGICAL FUNCTION: Detects when an explicit confirmation or
    correction pattern has occurred in the immediately-prior
    conversational turn, fires a phasic dopamine event of the
    appropriate amplitude and polarity into the NeuromodulatorBus,
    and triggers the corresponding lemma-status transition through
    the lemma_acquisition module.

    Reference: Equations 25.5 (provisional-to-confirmed transition)
    and the corpus's three-factor Hebbian framing in Section 25.4.
    Architect's Phase 3 spec, item 3.

    INTERFACE CONTRACT:
        Inputs:
            record_emission(emission): called by the production loop
                after each substrate-side turn to register the
                emission for subsequent confirmation matching.
            detect_confirmation(input_text): called by the input
                processor on each instructor turn. Returns a
                DetectionResult and, if firing, has already called
                through to the NeuromodulatorBus and the lemma
                acquisition module to fire the DA event and the
                status transition.
            reset(): clears the recorded last-emission state. Used
                between unrelated dialogue sessions.

        State: holds the most recent SubstrateEmission for the
            lemma-coincidence check. The state lives in this module
            rather than in the substrate because it is detector-side
            bookkeeping, not part of the substrate's lemma-stratum
            state.
    """

    def __init__(
        self,
        cfg: ConfirmationDetectorConfig,
        neuromodulator_bus: Any,
        lemma_acquisition: Any,
    ) -> None:
        """Initialize the detector with references to the bus and the
        lemma acquisition module.

        Args:
            cfg: ConfirmationDetectorConfig.
            neuromodulator_bus: a NeuromodulatorBus instance whose
                set("DA", value) method fires the DA event.
            lemma_acquisition: a LemmaAcquisitionModule instance
                whose confirm_row and decay_row methods are called
                on positive and negative polarity matches
                respectively.
        """
        self.cfg = cfg
        self.bus = neuromodulator_bus
        self.lemma_acquisition = lemma_acquisition
        self._last_emission: Optional[SubstrateEmission] = None

    # ---------------------------------------------------------------
    # Recording substrate emissions
    # ---------------------------------------------------------------

    def record_emission(self, emission: SubstrateEmission) -> None:
        """Register a substrate emission as the most recent turn.

        Called by the production loop after each substrate-side turn.
        The runtime is responsible for assembling the SubstrateEmission
        from the production-loop state at the moment the emission is
        finalized, including the surface tokens, the lemma slot
        indices that drove the emission, the polar-question flag, and
        the provisional-lemma slot index if any.

        Args:
            emission: the SubstrateEmission to record. Replaces any
                previous recorded emission.
        """
        self._last_emission = emission

    def reset(self) -> None:
        """Clear the recorded last-emission state.

        Called between unrelated dialogue sessions and at substrate
        cold-start. After reset, the next detect_confirmation call
        cannot fire any pattern because the lemma-coincidence
        requirement is structurally unmet.
        """
        self._last_emission = None

    # ---------------------------------------------------------------
    # Pattern matching
    # ---------------------------------------------------------------

    def _match_repeat_pattern(self, input_tokens: List[str]) -> bool:
        """Return True if input matches "yes <flipped substrate
        emission>".

        The check requires the recorded last-emission to exist with
        non-empty tokens, requires the input to start with "yes",
        and requires the flipped substrate-emission tokens to appear
        as a contiguous subsequence of the input following "yes".

        The substrate emitted "my name is Timmy?", whose tokens are
        ["my", "name", "is", "timmy"]. After flipping: ["your",
        "name", "is", "timmy"]. The instructor's "yes your name is
        Timmy" tokenizes to ["yes", "your", "name", "is", "timmy"].
        The check verifies that the flipped tokens appear contiguous
        starting at index 1 of the input.

        Args:
            input_tokens: the instructor's input as a token list.

        Returns:
            True if the repeat pattern matches.
        """
        if not self.cfg.enable_repeat_pattern:
            return False
        if self._last_emission is None:
            return False
        substrate_tokens = self._last_emission.tokens
        if not substrate_tokens:
            return False
        if not input_tokens or input_tokens[0] != "yes":
            return False
        flipped = flip_pronouns(substrate_tokens)
        # Strip trailing question-mark token if it is present in the
        # substrate's emission; the instructor's confirmation will
        # not have it.
        if flipped and flipped[-1] in {"?", ""}:
            flipped = flipped[:-1]
        # Check contiguous subsequence match starting at input[1:].
        rest = input_tokens[1:]
        if len(rest) < len(flipped):
            return False
        return rest[: len(flipped)] == flipped

    def _match_static_pattern(
        self, input_tokens: List[str],
    ) -> Optional[Tuple[str, ConfirmationPattern]]:
        """Return the first matching static pattern, or None.

        Static patterns are token-list literals that match if the
        input contains them as a contiguous subsequence anywhere.

        Args:
            input_tokens: the instructor's input as a token list.

        Returns:
            (pattern_name, pattern) on match, None otherwise.
        """
        for name, pattern in CONFIRMATION_PATTERN_INVENTORY:
            if pattern.is_repeat:
                continue
            if not pattern.tokens:
                continue
            n = len(pattern.tokens)
            if n > len(input_tokens):
                continue
            for start in range(len(input_tokens) - n + 1):
                if input_tokens[start : start + n] == pattern.tokens:
                    return (name, pattern)
        return None

    # ---------------------------------------------------------------
    # Detection entry point
    # ---------------------------------------------------------------

    def detect_confirmation(self, input_text: str) -> DetectionResult:
        """Process an instructor input and fire confirmation if
        appropriate.

        The detection sequence:
        1. Master flag check. If disabled, return a no-fire result.
        2. Lemma-coincidence check. The recorded last-emission must
           exist, be a polar question, and have a provisional lemma.
           If not, return no-fire.
        3. Tokenize the input.
        4. Try the repeat pattern first (highest amplitude).
        5. Fall through to the static-pattern inventory.
        6. On match, check polarity-enable flags. If the matched
           polarity is disabled, treat as no-fire.
        7. Fire the DA event into the bus and call confirm_row or
           decay_row on the target lemma.
        8. Clear the recorded last-emission so the same confirmation
           cannot fire twice on the same provisional binding.

        Args:
            input_text: raw instructor input string.

        Returns:
            DetectionResult describing the outcome.
        """
        if not self.cfg.enable_confirmation_detector:
            return DetectionResult()

        if self._last_emission is None:
            return DetectionResult()
        if not self._last_emission.contains_polar_question():
            return DetectionResult()
        if not self._last_emission.has_provisional_lemma():
            return DetectionResult()

        input_tokens = tokenize(input_text)
        if not input_tokens:
            return DetectionResult()

        pattern_name: Optional[str] = None
        matched_pattern: Optional[ConfirmationPattern] = None

        if self._match_repeat_pattern(input_tokens):
            pattern_name = "yes_repeat"
            matched_pattern = next(
                (p for n, p in CONFIRMATION_PATTERN_INVENTORY
                 if n == "yes_repeat"),
                None,
            )

        if matched_pattern is None:
            static = self._match_static_pattern(input_tokens)
            if static is not None:
                pattern_name, matched_pattern = static

        if matched_pattern is None:
            return DetectionResult()

        # Polarity gating.
        if (
            matched_pattern.polarity == POLARITY_POSITIVE
            and not self.cfg.enable_positive_polarity
        ):
            return DetectionResult()
        if (
            matched_pattern.polarity == POLARITY_NEGATIVE
            and not self.cfg.enable_negative_polarity
        ):
            return DetectionResult()

        # Fire the DA event and trigger the lemma transition. The
        # amplitude is signed by polarity so that downstream consumers
        # of the DA scalar receive both magnitude and sign in a
        # single read.
        signed_amplitude = (
            matched_pattern.da_amplitude * matched_pattern.polarity
        )
        target_lemma_id = self._last_emission.provisional_lemma_id
        self._fire_dopamine(signed_amplitude)

        if matched_pattern.polarity == POLARITY_POSITIVE:
            self._confirm_target(target_lemma_id)
        else:
            self._decay_target(target_lemma_id)

        # Clear last-emission so a stray "yes" later in the same
        # session does not fire a second confirmation against the
        # same provisional lemma.
        self._last_emission = None

        return DetectionResult(
            fired=True,
            pattern_name=pattern_name,
            da_amplitude=matched_pattern.da_amplitude,
            polarity=matched_pattern.polarity,
            target_lemma_id=target_lemma_id,
        )

    # ---------------------------------------------------------------
    # Side-effect helpers
    # ---------------------------------------------------------------

    def _fire_dopamine(self, signed_amplitude: float) -> None:
        """Write a DA scalar to the NeuromodulatorBus.

        Args:
            signed_amplitude: amplitude in [-1, 1]; sign carries
                polarity.
        """
        if hasattr(self.bus, "set"):
            self.bus.set("DA", torch.tensor(float(signed_amplitude)))

    def _confirm_target(self, slot_index: Optional[int]) -> None:
        """Call confirm_row on the lemma_acquisition module."""
        if slot_index is None:
            return
        if hasattr(self.lemma_acquisition, "confirm_row"):
            self.lemma_acquisition.confirm_row(slot_index)

    def _decay_target(self, slot_index: Optional[int]) -> None:
        """Call decay_row on the lemma_acquisition module if it has
        one, else fall back to setting the row to STATUS_UNALLOCATED
        through whatever interface is exposed.
        """
        if slot_index is None:
            return
        if hasattr(self.lemma_acquisition, "decay_row"):
            self.lemma_acquisition.decay_row(slot_index)
        elif hasattr(self.lemma_acquisition, "decay_provisional"):
            self.lemma_acquisition.decay_provisional(slot_index)
