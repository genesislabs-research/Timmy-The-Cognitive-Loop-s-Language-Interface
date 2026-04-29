"""
test_confirmation_detector.py
Tests for the confirmation detector module.

Each test targets an architectural claim from the architect's Phase 3
spec. The cases cover the lemma-coincidence requirement, the
pronoun-flip handling for the repeat pattern, polarity routing,
amplitude correctness, and ablation behavior. Stub bus and stub
lemma-acquisition objects let the tests verify that the detector
fires the right side effects without depending on the full substrate.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from coordination.confirmation_detector_t import (
    POLARITY_NEGATIVE,
    POLARITY_POSITIVE,
    ConfirmationDetector,
    ConfirmationDetectorConfig,
    SubstrateEmission,
    flip_pronouns,
    tokenize,
)


# =========================================================================
# Stubs
# =========================================================================

class StubBus:
    """Minimal NeuromodulatorBus stand-in for tests.

    Records every set() call so tests can verify that the detector
    fires the right DA amplitude.
    """

    def __init__(self) -> None:
        self.set_calls: List[tuple] = []

    def set(self, key: str, value: torch.Tensor) -> None:
        self.set_calls.append((key, float(value.item())))


class StubLemmaAcquisition:
    """Minimal LemmaAcquisitionModule stand-in for tests.

    Records every confirm_row and decay_row call so tests can verify
    the detector triggers the right transitions on the right slots.
    """

    def __init__(self) -> None:
        self.confirm_calls: List[int] = []
        self.decay_calls: List[int] = []

    def confirm_row(self, slot_index: int) -> None:
        self.confirm_calls.append(slot_index)

    def decay_row(self, slot_index: int) -> None:
        self.decay_calls.append(slot_index)


def make_detector() -> tuple:
    """Construct a detector with stub dependencies for tests.

    Returns:
        (detector, bus_stub, lemma_acquisition_stub).
    """
    bus = StubBus()
    lemmas = StubLemmaAcquisition()
    cfg = ConfirmationDetectorConfig()
    detector = ConfirmationDetector(
        cfg=cfg,
        neuromodulator_bus=bus,
        lemma_acquisition=lemmas,
    )
    return detector, bus, lemmas


def naming_emission(
    provisional_slot: Optional[int] = 17,
    polar: bool = True,
    tokens: Optional[List[str]] = None,
) -> SubstrateEmission:
    """Construct a substrate emission for the cold-start naming
    dialogue's turn 3: 'my name is Timmy?' with provisional lemma 17."""
    if tokens is None:
        tokens = ["my", "name", "is", "timmy"]
    return SubstrateEmission(
        tokens=tokens,
        lemma_ids=[0, 17],  # self_lemma + Timmy lemma
        polar_question=polar,
        provisional_lemma_id=provisional_slot,
    )


# =========================================================================
# Tokenization and pronoun flip helpers
# =========================================================================

class TestHelpers:

    def test_tokenize_lowercases_and_strips_punctuation(self):
        assert tokenize("Yes, your name is Timmy.") == [
            "yes", "your", "name", "is", "timmy",
        ]
        assert tokenize("My name is Timmy?") == [
            "my", "name", "is", "timmy",
        ]
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_flip_pronouns_swaps_first_and_second_person(self):
        assert flip_pronouns(
            ["my", "name", "is", "timmy"]
        ) == ["your", "name", "is", "timmy"]
        assert flip_pronouns(
            ["your", "name", "is", "timmy"]
        ) == ["my", "name", "is", "timmy"]
        assert flip_pronouns(["i", "am", "tired"]) == [
            "you", "am", "tired",
        ]

    def test_flip_pronouns_passes_through_non_pronouns(self):
        assert flip_pronouns(
            ["the", "cat", "sat"]
        ) == ["the", "cat", "sat"]


# =========================================================================
# Lemma-coincidence requirement
# =========================================================================

class TestLemmaCoincidence:
    """Tests verifying that the detector only fires when a polar
    question with a provisional lemma was emitted in the immediately
    prior turn. This is the architectural commitment that prevents
    arbitrary 'yes' utterances from confirming arbitrary lemmas."""

    def test_does_not_fire_without_recorded_emission(self):
        """No emission recorded means no provisional lemma is
        pending. 'yes' cannot fire."""
        detector, bus, lemmas = make_detector()
        result = detector.detect_confirmation("yes")
        assert result.fired is False
        assert bus.set_calls == []
        assert lemmas.confirm_calls == []

    def test_does_not_fire_on_non_polar_emission(self):
        """If the substrate emitted a declarative (non-polar) turn,
        a 'yes' from the instructor is not a confirmation event."""
        detector, bus, lemmas = make_detector()
        emission = SubstrateEmission(
            tokens=["the", "cat", "sat"],
            lemma_ids=[42],
            polar_question=False,
            provisional_lemma_id=None,
        )
        detector.record_emission(emission)
        result = detector.detect_confirmation("yes")
        assert result.fired is False
        assert lemmas.confirm_calls == []

    def test_does_not_fire_when_no_provisional_lemma(self):
        """A polar question without a provisional lemma is a
        clarifying question, not a confirmation request. 'yes' here
        does not confirm anything."""
        detector, bus, lemmas = make_detector()
        emission = SubstrateEmission(
            tokens=["are", "you", "there"],
            lemma_ids=[8],  # polar_question marker only
            polar_question=True,
            provisional_lemma_id=None,
        )
        detector.record_emission(emission)
        result = detector.detect_confirmation("yes")
        assert result.fired is False

    def test_fires_after_correct_emission(self):
        """The base case: polar question with provisional lemma,
        followed by 'yes', fires the confirmation."""
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("yes")
        assert result.fired is True
        assert result.target_lemma_id == 17
        assert lemmas.confirm_calls == [17]


# =========================================================================
# Pattern matching and amplitude
# =========================================================================

class TestPatternMatching:

    def test_yes_repeat_matches_with_pronoun_flip(self):
        """Architect's spec: 'yes your name is Timmy' must match the
        repeat pattern when the substrate just emitted 'my name is
        Timmy?'. The pronoun flip from MY to YOUR happens because
        the role of speaker has flipped between the two turns."""
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("yes your name is Timmy")
        assert result.fired is True
        assert result.pattern_name == "yes_repeat"
        # The repeat pattern has the highest amplitude in the
        # inventory.
        assert abs(result.da_amplitude - 0.95) < 1e-6
        assert result.polarity == POLARITY_POSITIVE

    def test_bare_yes_matches_with_lower_amplitude(self):
        """Bare 'yes' matches but at a lower amplitude than the
        repeat pattern."""
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("yes")
        assert result.fired is True
        assert result.pattern_name == "yes"
        assert abs(result.da_amplitude - 0.7) < 1e-6

    def test_correct_matches_positive(self):
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("correct")
        assert result.fired is True
        assert result.pattern_name == "correct"
        assert result.polarity == POLARITY_POSITIVE
        assert abs(result.da_amplitude - 0.8) < 1e-6

    def test_thats_right_matches_positive(self):
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("That's right.")
        assert result.fired is True
        assert result.polarity == POLARITY_POSITIVE

    def test_no_matches_negative(self):
        """'no' is a correction that decays the provisional row
        immediately."""
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("no")
        assert result.fired is True
        assert result.pattern_name == "no"
        assert result.polarity == POLARITY_NEGATIVE
        assert lemmas.decay_calls == [17]
        assert lemmas.confirm_calls == []

    def test_wrong_matches_negative_with_higher_amplitude(self):
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("wrong")
        assert result.fired is True
        assert result.polarity == POLARITY_NEGATIVE
        assert abs(result.da_amplitude - 0.7) < 1e-6


# =========================================================================
# Side effects
# =========================================================================

class TestSideEffects:
    """Tests verifying that the detector fires the right DA event
    into the bus and the right transition into the lemma module."""

    def test_positive_match_fires_positive_da_and_confirms(self):
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        detector.detect_confirmation("yes your name is Timmy")
        assert len(bus.set_calls) == 1
        key, value = bus.set_calls[0]
        assert key == "DA"
        assert value > 0
        assert lemmas.confirm_calls == [17]
        assert lemmas.decay_calls == []

    def test_negative_match_fires_negative_da_and_decays(self):
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        detector.detect_confirmation("no")
        assert len(bus.set_calls) == 1
        key, value = bus.set_calls[0]
        assert key == "DA"
        assert value < 0
        assert lemmas.decay_calls == [17]
        assert lemmas.confirm_calls == []

    def test_signed_amplitude_carries_both_magnitude_and_polarity(self):
        """The DA scalar fired into the bus should be amplitude
        signed by polarity, so a single read by downstream consumers
        carries both."""
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        detector.detect_confirmation("yes your name is Timmy")
        _, pos_value = bus.set_calls[0]

        # Reset and try negative.
        detector.reset()
        bus.set_calls = []
        detector.record_emission(naming_emission(provisional_slot=17))
        detector.detect_confirmation("no")
        _, neg_value = bus.set_calls[0]

        assert pos_value > 0
        assert neg_value < 0


# =========================================================================
# Single-fire-per-emission
# =========================================================================

class TestSingleFire:
    """Tests verifying that one substrate emission produces at most
    one confirmation event. After a confirmation fires, the recorded
    emission is cleared so subsequent 'yes' utterances do not
    re-confirm the same lemma."""

    def test_second_yes_does_not_fire(self):
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        first = detector.detect_confirmation("yes")
        second = detector.detect_confirmation("yes")
        assert first.fired is True
        assert second.fired is False
        assert lemmas.confirm_calls == [17]  # only fired once

    def test_new_emission_resets_eligibility(self):
        """After a confirmation clears the last-emission state, a
        new emission re-enables the detector for the new provisional
        lemma."""
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        detector.detect_confirmation("yes")
        # Now record a new emission for a different provisional lemma.
        detector.record_emission(naming_emission(
            provisional_slot=18,
            tokens=["my", "name", "is", "robby"],
        ))
        result = detector.detect_confirmation("yes")
        assert result.fired is True
        assert lemmas.confirm_calls == [17, 18]


# =========================================================================
# Pronoun flip edge cases
# =========================================================================

class TestPronounFlipEdgeCases:

    def test_repeat_pattern_requires_matching_content(self):
        """The substrate emitted 'my name is Timmy?'. If the
        instructor says 'yes your name is Robby', the content does
        not match the substrate's emission and the repeat pattern
        does not fire. The bare 'yes' pattern still fires because
        a 'yes' is present at the start of the input."""
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("yes your name is Robby")
        assert result.fired is True
        # Falls back to bare 'yes' pattern, not 'yes_repeat'.
        assert result.pattern_name == "yes"

    def test_repeat_pattern_does_not_fire_without_yes_prefix(self):
        """The repeat pattern requires 'yes' as the first token. A
        bare repetition without 'yes' is not the architect's
        confirmation form."""
        detector, bus, lemmas = make_detector()
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("your name is Timmy")
        # No 'yes', no 'no', no static pattern present in the input;
        # the 'is' is in the input but is not a confirmation pattern.
        # No fire.
        assert result.fired is False


# =========================================================================
# Ablation
# =========================================================================

class TestAblation:

    def test_master_flag_disables_detection(self):
        bus = StubBus()
        lemmas = StubLemmaAcquisition()
        cfg = ConfirmationDetectorConfig(
            enable_confirmation_detector=False,
        )
        detector = ConfirmationDetector(
            cfg=cfg,
            neuromodulator_bus=bus,
            lemma_acquisition=lemmas,
        )
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("yes your name is Timmy")
        assert result.fired is False
        assert bus.set_calls == []
        assert lemmas.confirm_calls == []

    def test_disable_positive_polarity(self):
        bus = StubBus()
        lemmas = StubLemmaAcquisition()
        cfg = ConfirmationDetectorConfig(
            enable_positive_polarity=False,
        )
        detector = ConfirmationDetector(
            cfg=cfg,
            neuromodulator_bus=bus,
            lemma_acquisition=lemmas,
        )
        detector.record_emission(naming_emission(provisional_slot=17))
        # 'yes' is positive: should not fire.
        result_pos = detector.detect_confirmation("yes")
        assert result_pos.fired is False
        # 'no' is negative: should still fire.
        detector.record_emission(naming_emission(provisional_slot=17))
        result_neg = detector.detect_confirmation("no")
        assert result_neg.fired is True

    def test_disable_negative_polarity(self):
        bus = StubBus()
        lemmas = StubLemmaAcquisition()
        cfg = ConfirmationDetectorConfig(
            enable_negative_polarity=False,
        )
        detector = ConfirmationDetector(
            cfg=cfg,
            neuromodulator_bus=bus,
            lemma_acquisition=lemmas,
        )
        detector.record_emission(naming_emission(provisional_slot=17))
        result_neg = detector.detect_confirmation("no")
        assert result_neg.fired is False
        detector.record_emission(naming_emission(provisional_slot=17))
        result_pos = detector.detect_confirmation("yes")
        assert result_pos.fired is True

    def test_disable_repeat_pattern_falls_back_to_bare_yes(self):
        """With the repeat pattern disabled, 'yes your name is Timmy'
        no longer matches the high-amplitude pattern but still
        matches the bare 'yes'."""
        bus = StubBus()
        lemmas = StubLemmaAcquisition()
        cfg = ConfirmationDetectorConfig(enable_repeat_pattern=False)
        detector = ConfirmationDetector(
            cfg=cfg,
            neuromodulator_bus=bus,
            lemma_acquisition=lemmas,
        )
        detector.record_emission(naming_emission(provisional_slot=17))
        result = detector.detect_confirmation("yes your name is Timmy")
        assert result.fired is True
        assert result.pattern_name == "yes"
        assert abs(result.da_amplitude - 0.7) < 1e-6


# =========================================================================
# Cold-start dialogue end-to-end at the detector level
# =========================================================================

class TestColdStartDialogue:
    """Walks the cold-start naming dialogue's confirmation moment in
    isolation. Turn 3 is the substrate emitting 'my name is Timmy?',
    turn 4 is the instructor confirming with 'yes your name is
    Timmy'. Verifies that the detector fires correctly and triggers
    the lemma transition at turn 4."""

    def test_cold_start_dialogue_confirmation_moment(self):
        detector, bus, lemmas = make_detector()

        # Turn 3 (substrate side): record the polar-question
        # emission for the freshly-allocated Timmy lemma at slot 17.
        emission = SubstrateEmission(
            tokens=["my", "name", "is", "timmy"],
            lemma_ids=[0, 17],  # self_lemma plus Timmy lemma
            polar_question=True,
            provisional_lemma_id=17,
        )
        detector.record_emission(emission)

        # Turn 4 (instructor side): process the confirmation.
        result = detector.detect_confirmation(
            "yes your name is Timmy"
        )

        # Architectural assertions for turn 4.
        # 1. The repeat pattern matched at the highest amplitude.
        assert result.fired is True
        assert result.pattern_name == "yes_repeat"
        assert abs(result.da_amplitude - 0.95) < 1e-6
        assert result.polarity == POLARITY_POSITIVE

        # 2. The DA event fired into the bus with positive sign.
        assert len(bus.set_calls) == 1
        key, value = bus.set_calls[0]
        assert key == "DA"
        assert value > 0

        # 3. The Timmy lemma at slot 17 was confirmed (turn 4 is the
        #    moment of the provisional-to-confirmed transition).
        assert lemmas.confirm_calls == [17]
        assert lemmas.decay_calls == []

        # 4. Subsequent stray 'yes' does not re-confirm.
        result2 = detector.detect_confirmation("yes")
        assert result2.fired is False
        assert lemmas.confirm_calls == [17]  # still only one
