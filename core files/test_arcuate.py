"""
test_arcuate.py
Tests for the arcuate fasciculus transport module.

These tests verify:
    1. Conduction delay is exactly tau_arc_steps: a segment pushed at
       tick t emerges downstream at tick t + tau_arc_steps.
    2. Order is preserved: segments emerge in the same order they
       entered.
    3. Initial transient: for the first tau_arc_steps ticks after
       construction, downstream sees zeros.
    4. W_arc default is identity, so segments pass through unchanged.
    5. Conduction-aphasia ablation (enable_arcuate=False) zeros the
       output regardless of input.
    6. Round-trip serialization preserves buffer contents.
    7. Round-trip serialization preserves in-flight segments correctly.
    8. Buffer batch-size adaptation works when the runtime changes B.
"""

from __future__ import annotations

import pytest
import torch

from regions.arcuate_t import Arcuate, ArcuateConfig


# =========================================================================
# Conduction delay
# =========================================================================

class TestConductionDelay:

    def test_segment_emerges_after_tau_arc_steps(self):
        """A segment pushed at tick 0 should emerge at tick tau_arc_steps,
        not before and not after.
        """
        cfg = ArcuateConfig(n_segments=8, tau_arc_steps=3)
        arc = Arcuate(cfg)

        # Distinctive input segment.
        segment_in = torch.zeros(1, 8)
        segment_in[0, 5] = 1.0

        # Tick 0: push the segment, expect zeros out.
        out_0 = arc(segment_in)
        assert torch.allclose(out_0, torch.zeros_like(out_0)), (
            f"Tick 0 output should be zeros (initial transient), "
            f"got {out_0}."
        )

        # Ticks 1 and 2: still in transit, push zeros and expect zeros.
        zero_input = torch.zeros(1, 8)
        out_1 = arc(zero_input)
        assert torch.allclose(out_1, torch.zeros_like(out_1))
        out_2 = arc(zero_input)
        assert torch.allclose(out_2, torch.zeros_like(out_2))

        # Tick 3: the segment from tick 0 should emerge.
        out_3 = arc(zero_input)
        assert torch.allclose(out_3, segment_in, atol=1e-6), (
            f"Tick 3 output should match the segment pushed at tick 0, "
            f"got {out_3}."
        )

    def test_short_delay(self):
        """tau_arc_steps=1 (5 ms biological delay) emerges on the next tick."""
        cfg = ArcuateConfig(n_segments=8, tau_arc_steps=1)
        arc = Arcuate(cfg)

        seg = torch.zeros(1, 8)
        seg[0, 3] = 1.0

        _ = arc(seg)  # Tick 0: push, output is initial zero buffer slot.
        out_1 = arc(torch.zeros(1, 8))  # Tick 1: segment emerges.
        assert torch.allclose(out_1, seg, atol=1e-6)


# =========================================================================
# Order preservation
# =========================================================================

class TestOrderPreservation:

    def test_segments_emerge_in_input_order(self):
        """Push three distinct segments in sequence; they must emerge in
        the same order they entered.
        """
        cfg = ArcuateConfig(n_segments=8, tau_arc_steps=2)
        arc = Arcuate(cfg)

        s0 = torch.zeros(1, 8); s0[0, 0] = 1.0
        s1 = torch.zeros(1, 8); s1[0, 1] = 1.0
        s2 = torch.zeros(1, 8); s2[0, 2] = 1.0

        # Tick 0..2: push the three segments (initial transient zeros).
        _ = arc(s0)
        _ = arc(s1)
        out_2 = arc(s2)
        # First emergence is the segment from tick 0.
        assert torch.allclose(out_2, s0, atol=1e-6)

        out_3 = arc(torch.zeros(1, 8))
        assert torch.allclose(out_3, s1, atol=1e-6)

        out_4 = arc(torch.zeros(1, 8))
        assert torch.allclose(out_4, s2, atol=1e-6)


# =========================================================================
# Identity projection
# =========================================================================

class TestIdentityProjection:

    def test_default_w_arc_is_exact_identity(self):
        """With identity_jitter=0, W_arc should be the exact identity."""
        cfg = ArcuateConfig(n_segments=64, identity_jitter=0.0)
        arc = Arcuate(cfg)
        assert torch.allclose(
            arc.W_arc, torch.eye(64), atol=1e-9,
        )

    def test_jittered_w_arc_is_near_identity(self):
        """With small jitter, W_arc should be close to identity but not exact."""
        torch.manual_seed(0)
        cfg = ArcuateConfig(n_segments=8, identity_jitter=0.01)
        arc = Arcuate(cfg)
        assert not torch.allclose(arc.W_arc, torch.eye(8), atol=1e-6)
        # But the diagonal should still dominate.
        diag = arc.W_arc.diag()
        off_diag_max = (arc.W_arc - torch.diag(diag)).abs().max()
        assert diag.mean() > off_diag_max * 5.0, (
            "Diagonal should dominate the off-diagonal entries."
        )

    def test_w_arc_is_not_a_parameter(self):
        """W_arc must be a buffer, not a parameter, so optimizers do not
        pick it up. Making it learnable would let the substrate route
        around the rest of the speech pathway.
        """
        cfg = ArcuateConfig(n_segments=8)
        arc = Arcuate(cfg)
        param_names = {name for name, _ in arc.named_parameters()}
        assert "W_arc" not in param_names, (
            "W_arc must be a buffer (frozen), not a learnable parameter."
        )


# =========================================================================
# Conduction aphasia ablation
# =========================================================================

class TestConductionAphasiaAblation:

    def test_ablation_zeros_output(self):
        """Setting enable_arcuate=False reproduces conduction aphasia:
        the transport pathway is destroyed, downstream output is zero
        regardless of input.
        """
        cfg = ArcuateConfig(n_segments=8, enable_arcuate=False)
        arc = Arcuate(cfg)
        seg = torch.zeros(1, 8); seg[0, 3] = 1.0
        for _ in range(10):
            out = arc(seg)
            assert torch.allclose(out, torch.zeros_like(out))


# =========================================================================
# Serialization
# =========================================================================

class TestSerialization:

    def test_round_trip_preserves_buffer_contents(self):
        """Save with segments in flight, restore, verify the same
        segments emerge at the same ticks.
        """
        cfg = ArcuateConfig(n_segments=8, tau_arc_steps=3)
        original = Arcuate(cfg)

        s0 = torch.zeros(1, 8); s0[0, 0] = 1.0
        s1 = torch.zeros(1, 8); s1[0, 1] = 1.0
        # After construction, buffer is [z, z, z] front-to-back.
        # Tick 0: push s0, pop z. Buffer becomes [z, z, s0].
        # Tick 1: push s1, pop z. Buffer becomes [z, s0, s1].
        # The next two pops will yield z then s0 then s1.
        _ = original(s0)
        _ = original(s1)

        state = original.serialize()
        restored = Arcuate(cfg)
        restored.restore(state)

        # First pop on the restored arcuate is the remaining initial
        # zero from before s0 was pushed.
        out_initial = restored(torch.zeros(1, 8))
        assert torch.allclose(out_initial, torch.zeros_like(out_initial))

        # Then s0 emerges.
        out_a = restored(torch.zeros(1, 8))
        assert torch.allclose(out_a, s0, atol=1e-6), (
            "Restored arcuate should emit s0 after the initial zero pad."
        )

        # Then s1.
        out_b = restored(torch.zeros(1, 8))
        assert torch.allclose(out_b, s1, atol=1e-6)

    def test_round_trip_preserves_w_arc(self):
        """The W_arc matrix must survive a checkpoint cycle.
        """
        torch.manual_seed(0)
        cfg = ArcuateConfig(n_segments=8, identity_jitter=0.05)
        original = Arcuate(cfg)

        state = original.serialize()
        restored = Arcuate(cfg)
        # Restored has a fresh random init; restore must overwrite it
        # to match the original.
        restored.restore(state)
        assert torch.allclose(original.W_arc, restored.W_arc, atol=1e-9)

    def test_restore_with_buffer_length_mismatch_raises(self):
        cfg_short = ArcuateConfig(n_segments=8, tau_arc_steps=2)
        cfg_long = ArcuateConfig(n_segments=8, tau_arc_steps=5)

        short = Arcuate(cfg_short)
        long = Arcuate(cfg_long)
        state = short.serialize()
        with pytest.raises(ValueError, match="buffer length mismatch"):
            long.restore(state)


# =========================================================================
# Batch-size adaptation
# =========================================================================

class TestBatchSizeAdaptation:

    def test_buffer_adapts_to_input_batch_size(self):
        """If the runtime changes batch size between calls, the buffer
        should adapt without raising. This is the same recovery pattern
        the cognitive-loop CorticalBuffer uses.
        """
        cfg = ArcuateConfig(n_segments=8, tau_arc_steps=2)
        arc = Arcuate(cfg)

        # First call with batch size 1.
        seg_1 = torch.zeros(1, 8)
        out_1 = arc(seg_1)
        assert out_1.shape == (1, 8)

        # Then a call with batch size 4. Buffer reallocates.
        seg_4 = torch.zeros(4, 8)
        out_4 = arc(seg_4)
        assert out_4.shape == (4, 8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
