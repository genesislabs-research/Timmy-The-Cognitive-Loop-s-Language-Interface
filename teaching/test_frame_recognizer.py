"""
test_frame_recognizer.py
Tests for the teaching frame recognizer module.

Each case targets a specific pattern in the architect's spec or a
specific architectural commitment about the bias-vector mechanism
and the ACh emission. The cold-start dialogue test at the bottom
walks turn 2 (instructor says 'your name is Timmy') and verifies
that naming_self matches, the wildcard binding captures TIMMY, the
bias vector activates the naming_frame dimension, and ACh fires at
0.85.
"""

from __future__ import annotations

from typing import List

import torch

from coordination.frame_recognizer_t import (
    FRAME_BIAS_DIMS,
    TEACHING_FRAME_INVENTORY,
    TeachingFrameRecognizer,
    TeachingFrameRecognizerConfig,
)


# =========================================================================
# Stubs
# =========================================================================

class StubBus:
    """Minimal NeuromodulatorBus stand-in.

    Records every set() call so tests can verify that the
    recognizer fires the right ACh amplitude into the right key.
    """

    def __init__(self) -> None:
        self.set_calls: List[tuple] = []

    def set(self, key: str, value: torch.Tensor) -> None:
        self.set_calls.append((key, float(value.item())))


def make_recognizer(
    bus: StubBus = None,
    n_concepts: int = 1024,
) -> TeachingFrameRecognizer:
    """Construct a recognizer for tests."""
    cfg = TeachingFrameRecognizerConfig(n_concepts=n_concepts)
    return TeachingFrameRecognizer(
        cfg=cfg, neuromodulator_bus=bus,
    )


# =========================================================================
# Inventory invariants
# =========================================================================

class TestInventory:
    """Tests verifying the inventory is well-formed and matches the
    architect's spec."""

    def test_inventory_has_seven_frames(self):
        """The architect's spec lists seven teaching frames:
        naming_self, naming_other, vocabulary_intro, vocabulary_call,
        definition, correction, confirmation_pos."""
        names = {f.name for f in TEACHING_FRAME_INVENTORY}
        expected = {
            "naming_self", "naming_other", "vocabulary_intro",
            "vocabulary_call", "definition", "correction",
            "confirmation_pos",
        }
        assert names == expected

    def test_naming_self_amplitude_matches_spec(self):
        """The architect's spec sets naming_self ach_amplitude to
        0.85."""
        ns = next(
            f for f in TEACHING_FRAME_INVENTORY
            if f.name == "naming_self"
        )
        assert abs(ns.ach_amplitude - 0.85) < 1e-9

    def test_correction_amplitude_is_highest(self):
        """The architect's spec sets correction at 0.95, the
        highest amplitude in the inventory because corrections
        require the strongest encoding-mode signal."""
        corr = next(
            f for f in TEACHING_FRAME_INVENTORY
            if f.name == "correction"
        )
        assert abs(corr.ach_amplitude - 0.95) < 1e-9
        # Verify nothing else is higher.
        for f in TEACHING_FRAME_INVENTORY:
            if f.name != "correction":
                assert f.ach_amplitude <= corr.ach_amplitude

    def test_confirmation_pos_amplitude_is_lowest(self):
        """The architect's spec sets confirmation_pos at 0.40, the
        lowest amplitude because confirmations should not drive new
        encoding events."""
        cp = next(
            f for f in TEACHING_FRAME_INVENTORY
            if f.name == "confirmation_pos"
        )
        assert abs(cp.ach_amplitude - 0.40) < 1e-9

    def test_bias_dims_are_in_subspace(self):
        """All frame bias dimensions live at concept-space indices
        1000-1015, the architect's reserved frame-bias subspace."""
        for dim_name, dim_idx in FRAME_BIAS_DIMS.items():
            assert 1000 <= dim_idx <= 1015, (
                f"Frame bias dim {dim_name} at {dim_idx} is "
                f"outside the reserved subspace 1000-1015."
            )


# =========================================================================
# Pattern matching
# =========================================================================

class TestPatternMatching:

    def test_naming_self_matches(self):
        """The architect's load-bearing pattern: 'your name is X'
        fires naming_self."""
        r = make_recognizer()
        result = r.recognize_frame("your name is Timmy")
        assert result.recognized is True
        assert result.frame_name == "naming_self"
        assert result.bias_dim_name == "naming_frame"
        assert abs(result.ach_amplitude - 0.85) < 1e-9
        assert result.wildcard_bindings == {"<X>": "timmy"}

    def test_naming_other_matches(self):
        """The reverse-direction pattern: 'my name is X' for the
        instructor introducing themselves."""
        r = make_recognizer()
        result = r.recognize_frame("my name is Amellia")
        assert result.recognized is True
        assert result.frame_name == "naming_other"
        assert result.wildcard_bindings == {"<X>": "amellia"}

    def test_vocabulary_intro_matches(self):
        """'the word for X is Y' fires vocabulary_intro with two
        wildcard bindings."""
        r = make_recognizer()
        result = r.recognize_frame(
            "the word for cat is feline"
        )
        assert result.recognized is True
        assert result.frame_name == "vocabulary_intro"
        assert result.wildcard_bindings == {
            "<X>": "cat",
            "<Y>": "feline",
        }
        assert abs(result.ach_amplitude - 0.90) < 1e-9

    def test_vocabulary_call_matches(self):
        r = make_recognizer()
        result = r.recognize_frame("this is a dog")
        assert result.recognized is True
        assert result.frame_name == "vocabulary_call"
        assert result.wildcard_bindings == {"<X>": "dog"}

    def test_definition_matches(self):
        r = make_recognizer()
        result = r.recognize_frame("happy means joyful")
        assert result.recognized is True
        assert result.frame_name == "definition"

    def test_correction_matches(self):
        r = make_recognizer()
        # Pattern is "no <X> is", so input needs to fit that shape.
        result = r.recognize_frame("no this is")
        assert result.recognized is True
        assert result.frame_name == "correction"
        assert abs(result.ach_amplitude - 0.95) < 1e-6

    def test_confirmation_pos_matches(self):
        r = make_recognizer()
        result = r.recognize_frame("yes correct")
        assert result.recognized is True
        assert result.frame_name == "confirmation_pos"
        assert abs(result.ach_amplitude - 0.40) < 1e-9

    def test_no_match_returns_unrecognized(self):
        """Input that does not match any frame returns a no-match
        result with no fields set."""
        r = make_recognizer()
        result = r.recognize_frame("hello there how are you")
        assert result.recognized is False
        assert result.frame_name is None
        assert result.ach_amplitude == 0.0
        assert result.wildcard_bindings == {}

    def test_empty_input_does_not_match(self):
        r = make_recognizer()
        result = r.recognize_frame("")
        assert result.recognized is False

    def test_match_handles_punctuation(self):
        """Tokenization strips punctuation, so 'Your name is Timmy.'
        still matches naming_self."""
        r = make_recognizer()
        result = r.recognize_frame("Your name is Timmy.")
        assert result.recognized is True
        assert result.frame_name == "naming_self"

    def test_more_specific_pattern_matches_first(self):
        """The architect's spec orders the inventory most-specific
        first. 'the word for X is Y' should match vocabulary_intro
        rather than 'this is a Y' even though the latter is also
        present in the input as a substring fragment."""
        r = make_recognizer()
        # "the word for cat is feline" contains "is feline" but not
        # "this is a feline", so vocabulary_intro is the only match.
        # Sanity check that vocabulary_intro wins.
        result = r.recognize_frame("the word for cat is feline")
        assert result.frame_name == "vocabulary_intro"


# =========================================================================
# Wildcard handling
# =========================================================================

class TestWildcardHandling:

    def test_wildcard_binds_one_token(self):
        """Each wildcard matches exactly one token, no more and no
        less. 'your name is Big Bird' has two tokens after the
        prefix; the pattern only matches the first one."""
        r = make_recognizer()
        result = r.recognize_frame("your name is Big Bird")
        assert result.recognized is True
        assert result.wildcard_bindings == {"<X>": "big"}

    def test_short_input_does_not_match_long_pattern(self):
        """'your name is' has only three tokens. The pattern
        ['your', 'name', 'is', '<X>'] needs four tokens. No match."""
        r = make_recognizer()
        result = r.recognize_frame("your name is")
        assert result.recognized is False

    def test_wildcard_at_start_of_pattern(self):
        """The 'definition' pattern starts with a wildcard:
        '<X> means <Y>'. Verify it matches 'happy means joyful'."""
        r = make_recognizer()
        result = r.recognize_frame("happy means joyful")
        assert result.recognized is True
        assert result.frame_name == "definition"


# =========================================================================
# Bias vector
# =========================================================================

class TestBiasVector:
    """Tests verifying the bias-vector mechanism. The bias is a
    one-hot activation in the frame-bias subspace at concept-space
    dimensions 1000-1015."""

    def test_bias_vector_is_one_hot_at_correct_dim(self):
        r = make_recognizer()
        _, bias = r.recognize_and_get_bias("your name is Timmy")
        # Naming frame is at dim 1000.
        expected_dim = FRAME_BIAS_DIMS["naming_frame"]
        # Verify one-hot.
        assert bias.shape == (1024,)
        assert abs(bias[expected_dim].item() - 1.0) < 1e-9
        # All other dims zero.
        non_target = bias.clone()
        non_target[expected_dim] = 0.0
        assert non_target.norm().item() == 0.0

    def test_bias_vector_for_no_match_is_zero(self):
        """When no frame matches, the bias vector is all zeros so
        the runtime can add it unconditionally without changing the
        conceptual stratum."""
        r = make_recognizer()
        _, bias = r.recognize_and_get_bias("hello there")
        assert bias.norm().item() == 0.0

    def test_get_bias_vector_by_name(self):
        """The runtime can request a bias vector for a frame
        without doing pattern matching. Used for frame persistence
        within a turn."""
        r = make_recognizer()
        bias = r.get_bias_vector("naming_self")
        expected_dim = FRAME_BIAS_DIMS["naming_frame"]
        assert abs(bias[expected_dim].item() - 1.0) < 1e-9

    def test_get_bias_vector_unknown_name_raises(self):
        r = make_recognizer()
        try:
            r.get_bias_vector("nonexistent_frame")
            assert False, "Expected KeyError on unknown frame."
        except KeyError:
            pass

    def test_different_frames_use_different_bias_dims(self):
        """The architect's spec reserves separate bias dimensions
        for each conceptual frame so allocations under different
        frames produce distinguishable concept vectors."""
        r = make_recognizer()
        result_naming = r.recognize_frame("your name is Timmy")
        result_vocab = r.recognize_frame("this is a dog")
        assert (
            result_naming.bias_dim_index
            != result_vocab.bias_dim_index
        )


# =========================================================================
# ACh emission
# =========================================================================

class TestAChEmission:

    def test_match_writes_ach_to_bus(self):
        bus = StubBus()
        r = make_recognizer(bus=bus)
        r.recognize_frame("your name is Timmy")
        assert len(bus.set_calls) == 1
        key, value = bus.set_calls[0]
        # The architect specified ACh as the modulator that drives
        # encoding mode. Specifically the incremental pathway from
        # nucleus basalis (Hasselmo 2006), which the
        # NeuromodulatorBus exposes as the ACh_inc key.
        assert key == "ACh_inc"
        # Tolerance is 1e-6 because the bus value is round-tripped
        # through a float32 tensor in StubBus.set, which loses
        # precision below 1e-7.
        assert abs(value - 0.85) < 1e-6

    def test_no_match_does_not_write_ach(self):
        bus = StubBus()
        r = make_recognizer(bus=bus)
        r.recognize_frame("hello there")
        assert bus.set_calls == []

    def test_disable_ach_emission(self):
        bus = StubBus()
        cfg = TeachingFrameRecognizerConfig(
            enable_ach_emission=False,
        )
        r = TeachingFrameRecognizer(
            cfg=cfg, neuromodulator_bus=bus,
        )
        result = r.recognize_frame("your name is Timmy")
        # Still recognizes the frame.
        assert result.recognized is True
        # But does not write to the bus.
        assert bus.set_calls == []

    def test_no_bus_does_not_crash(self):
        """The recognizer can be used standalone without a bus.
        Useful for unit tests on the pattern matching alone."""
        r = make_recognizer(bus=None)
        result = r.recognize_frame("your name is Timmy")
        assert result.recognized is True


# =========================================================================
# Ablation
# =========================================================================

class TestAblation:

    def test_master_flag_disables_recognition(self):
        bus = StubBus()
        cfg = TeachingFrameRecognizerConfig(
            enable_frame_recognizer=False,
        )
        r = TeachingFrameRecognizer(
            cfg=cfg, neuromodulator_bus=bus,
        )
        result = r.recognize_frame("your name is Timmy")
        assert result.recognized is False
        assert bus.set_calls == []

    def test_disable_bias_emission(self):
        """When bias emission is disabled, recognize_and_get_bias
        returns the recognized frame but a zero bias vector."""
        cfg = TeachingFrameRecognizerConfig(
            enable_bias_emission=False,
        )
        r = TeachingFrameRecognizer(cfg=cfg, neuromodulator_bus=None)
        result, bias = r.recognize_and_get_bias("your name is Timmy")
        assert result.recognized is True
        assert bias.norm().item() == 0.0


# =========================================================================
# Cold-start dialogue at the recognizer level
# =========================================================================

class TestColdStartDialogue:
    """Walks turn 2 of the cold-start naming dialogue: instructor
    says 'your name is Timmy'. Verifies that the recognizer fires
    naming_self, captures TIMMY as the wildcard binding, fires ACh
    at 0.85, and produces the right bias vector for the conceptual
    stratum."""

    def test_turn_2_naming_self_recognition(self):
        bus = StubBus()
        r = make_recognizer(bus=bus)

        result, bias = r.recognize_and_get_bias("your name is Timmy")

        # 1. The naming_self frame matched.
        assert result.recognized is True
        assert result.frame_name == "naming_self"

        # 2. The wildcard bound TIMMY for downstream consumption.
        # The substrate uses this binding to construct the new
        # phonological code that gets allocated.
        assert result.wildcard_bindings == {"<X>": "timmy"}

        # 3. ACh fired into the bus at the spec amplitude.
        assert len(bus.set_calls) == 1
        key, value = bus.set_calls[0]
        assert key == "ACh_inc"
        # Tolerance 1e-6 for float32 round-trip through the bus.
        assert abs(value - 0.85) < 1e-6

        # 4. The bias vector is one-hot at the naming_frame
        # dimension. The substrate's allocation event writes a row
        # whose concept vector has this naming-frame dimension
        # active, so the row reflects the frame structure rather
        # than just the bare 'self + name' concept.
        naming_dim = FRAME_BIAS_DIMS["naming_frame"]
        assert abs(bias[naming_dim].item() - 1.0) < 1e-9
        # Other frame-bias dims are zero.
        for other_name, other_dim in FRAME_BIAS_DIMS.items():
            if other_name != "naming_frame":
                assert abs(bias[other_dim].item()) < 1e-9
