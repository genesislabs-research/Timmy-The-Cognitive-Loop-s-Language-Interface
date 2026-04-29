"""
test_tied_substrate.py
Tests for the TiedSubstrate primitive.

These tests verify the four test obligations from Section 3.6 of the
v2 Broca's Pathway Specification:

    (a) One matrix not two, by gradient identity check.
    (b) forward_a_to_b followed by forward_b_to_a returns to the input
        subspace within numerical precision when the matrix is full-rank.
    (c) Receptor modulation scales effective weights without altering the
        underlying parameter.
    (d) Serialization round-trips preserve direction symmetry.

The architectural commitment is that the tying is mechanical, not emergent.
Each test below is structured to fail loudly if the tying is broken in
that specific way, rather than to pass silently when the tying happens to
hold under benign conditions.
"""

from __future__ import annotations

import pytest
import torch

from substrate.tied_substrate_t import TiedSubstrate, TiedSubstrateConfig


# =========================================================================
# Test Obligation (a): One matrix, not two
# =========================================================================

class TestSingleMatrix:
    """The substrate must hold exactly one weight matrix accessed in both
    directions. Two independent matrices that happen to have the same shape
    do not satisfy the architectural commitment.
    """

    def test_single_parameter_in_module(self):
        """The substrate must declare exactly one nn.Parameter for the weight."""
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=16)
        sub = TiedSubstrate(cfg)
        params = list(sub.parameters())
        assert len(params) == 1, (
            f"TiedSubstrate must hold exactly one parameter, got {len(params)}. "
            f"Two matrices with the same shape are not tied."
        )

    def test_gradient_flows_to_same_parameter_from_both_directions(self):
        """A gradient computed through forward_a_to_b and a gradient computed
        through forward_b_to_a must update the same parameter. This is the
        defining property of mechanical tying.
        """
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=16)
        sub = TiedSubstrate(cfg)

        x_a = torch.randn(4, 8, requires_grad=False)
        x_b = torch.randn(4, 16, requires_grad=False)

        # Forward in A-to-B direction, compute a scalar loss, backward.
        y_b = sub.forward_a_to_b(x_a)
        loss_ab = y_b.sum()
        loss_ab.backward()
        grad_ab = sub.W.grad.clone()
        sub.W.grad.zero_()

        # Forward in B-to-A direction on a separate input, backward.
        y_a = sub.forward_b_to_a(x_b)
        loss_ba = y_a.sum()
        loss_ba.backward()
        grad_ba = sub.W.grad.clone()

        # Both gradients are nonzero (the parameter received gradient
        # contributions from both directions).
        assert not torch.allclose(grad_ab, torch.zeros_like(grad_ab)), (
            "A-to-B direction did not produce gradient on W."
        )
        assert not torch.allclose(grad_ba, torch.zeros_like(grad_ba)), (
            "B-to-A direction did not produce gradient on W."
        )

    def test_weight_change_in_one_direction_visible_in_other(self):
        """If the parameter is updated by training in one direction, the
        update must be visible to the other direction. Two independent
        matrices would fail this test.
        """
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=16)
        sub = TiedSubstrate(cfg)

        x_a = torch.randn(4, 8)
        y_b_before = sub.forward_a_to_b(x_a).clone()

        # Manually modify the parameter (simulating an optimizer step).
        with torch.no_grad():
            sub.W.add_(torch.ones_like(sub.W) * 0.1)

        # Forward in B-to-A direction with a known input. The change in W
        # must be visible here too because both directions reference the
        # same parameter.
        x_b = torch.randn(4, 16)
        y_a = sub.forward_b_to_a(x_b)

        # Construct the same forward by hand using the post-update W.
        expected_y_a = torch.matmul(x_b, sub.W)
        assert torch.allclose(y_a, expected_y_a, atol=1e-6), (
            "B-to-A direction did not see the weight update made through "
            "the A-to-B direction. The matrices are not tied."
        )

        # Also confirm the A-to-B direction sees the update.
        y_b_after = sub.forward_a_to_b(x_a)
        assert not torch.allclose(y_b_before, y_b_after, atol=1e-6), (
            "A-to-B direction output did not change after weight update."
        )


# =========================================================================
# Test Obligation (b): Round-trip preserves the input subspace
# =========================================================================

class TestRoundTripDirectionSymmetry:
    """When the matrix is full-rank, forward_a_to_b followed by
    forward_b_to_a should map x_a back into the A-space rather than into
    some unrelated subspace. With tied weights, the round-trip is W.t() @ W
    @ x_a, which projects onto the row space of W. For a square full-rank
    matrix this is the identity up to scaling.
    """

    def test_orthogonal_matrix_round_trip_is_identity(self):
        """When W is orthogonal, W.t() @ W = I, so forward_b_to_a composed
        with forward_a_to_b is the identity transformation. This is the
        cleanest test of the bidirectional access pattern: with an
        orthogonal weight, the round-trip exactly recovers the input.

        Orthogonal matrices arise naturally in tied-weight settings because
        they preserve norms in both directions, which is desirable for
        keeping production and perception activations on comparable scales.
        Using one here is a test convenience, not a biological commitment.
        """
        cfg = TiedSubstrateConfig(in_dim=16, out_dim=16)
        sub = TiedSubstrate(cfg)

        # Replace W with a random orthogonal matrix.
        # Reference: torch.linalg.qr produces an orthogonal Q from a random
        # Gaussian, the standard way to sample uniformly from the orthogonal
        # group up to sign.
        with torch.no_grad():
            Q, _ = torch.linalg.qr(torch.randn(16, 16))
            sub.W.copy_(Q)

        x_a = torch.randn(4, 16)
        y_b = sub.forward_a_to_b(x_a)
        x_a_recovered = sub.forward_b_to_a(y_b)

        assert torch.allclose(x_a, x_a_recovered, atol=1e-5), (
            f"Orthogonal-matrix round-trip should recover input exactly. "
            f"Max difference: {(x_a - x_a_recovered).abs().max().item():.6f}."
        )

    def test_rectangular_round_trip_still_well_defined(self):
        """For a rectangular matrix, the round-trip is not identity but
        must be well-defined and produce finite output of the correct shape.
        """
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=32)
        sub = TiedSubstrate(cfg)

        x_a = torch.randn(4, 8)
        y_b = sub.forward_a_to_b(x_a)
        x_a_recovered = sub.forward_b_to_a(y_b)

        assert x_a_recovered.shape == x_a.shape, (
            f"Round-trip changed shape: {x_a.shape} -> {x_a_recovered.shape}."
        )
        assert torch.isfinite(x_a_recovered).all(), (
            "Round-trip produced non-finite values."
        )


# =========================================================================
# Test Obligation (c): Receptor modulation does not alter the parameter
# =========================================================================

class TestReceptorModulation:
    """The receptor modulation argument scales the effective weight at
    forward time. The underlying parameter must not change. This is what
    makes region-specific receptor effects local rather than global.
    """

    def test_receptor_modulation_scales_output(self):
        """Doubling the receptor modulation should approximately double the
        output (linear scaling).
        """
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=16)
        sub = TiedSubstrate(cfg)

        x_a = torch.randn(4, 8)
        y_neutral = sub.forward_a_to_b(x_a)
        y_doubled = sub.forward_a_to_b(x_a, receptor_modulation=torch.tensor(2.0))

        assert torch.allclose(y_doubled, y_neutral * 2.0, atol=1e-6), (
            "Receptor modulation of 2.0 did not double the output."
        )

    def test_receptor_modulation_does_not_alter_parameter(self):
        """Calling forward with a receptor modulation must not change W."""
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=16)
        sub = TiedSubstrate(cfg)
        W_before = sub.W.detach().clone()

        x_a = torch.randn(4, 8)
        _ = sub.forward_a_to_b(x_a, receptor_modulation=torch.tensor(1.5))

        assert torch.allclose(sub.W, W_before, atol=1e-9), (
            "forward_a_to_b with receptor_modulation altered the W parameter."
        )

    def test_receptor_modulation_applies_to_both_directions(self):
        """Receptor modulation must scale forward_b_to_a output as well."""
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=16)
        sub = TiedSubstrate(cfg)

        x_b = torch.randn(4, 16)
        y_neutral = sub.forward_b_to_a(x_b)
        y_doubled = sub.forward_b_to_a(x_b, receptor_modulation=torch.tensor(2.0))

        assert torch.allclose(y_doubled, y_neutral * 2.0, atol=1e-6), (
            "Receptor modulation of 2.0 did not double B-to-A output."
        )


# =========================================================================
# Test Obligation (d): Serialization round-trips preserve direction symmetry
# =========================================================================

class TestSerialization:
    """Serializing and restoring the substrate must preserve all behavior.
    A round-trip serialize/restore should produce a substrate that gives
    bit-identical output to the original on the same inputs.
    """

    def test_serialize_restore_round_trip(self):
        """Save and restore the substrate, verify both directions produce
        the same output as the original on the same inputs.
        """
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=16)
        original = TiedSubstrate(cfg)
        # Train a bit to move weights off initialization.
        x_a_train = torch.randn(4, 8)
        y_b = original.forward_a_to_b(x_a_train)
        loss = y_b.sum()
        loss.backward()
        with torch.no_grad():
            original.W.sub_(original.W.grad * 0.01)

        state = original.serialize()

        # Restore into a fresh instance.
        restored = TiedSubstrate(cfg)
        restored.restore(state)

        # Both directions must produce identical output on the same inputs.
        x_a = torch.randn(4, 8)
        x_b = torch.randn(4, 16)

        assert torch.allclose(
            original.forward_a_to_b(x_a),
            restored.forward_a_to_b(x_a),
            atol=1e-9,
        ), "Serialization broke A-to-B direction."

        assert torch.allclose(
            original.forward_b_to_a(x_b),
            restored.forward_b_to_a(x_b),
            atol=1e-9,
        ), "Serialization broke B-to-A direction."

    def test_restore_with_shape_mismatch_raises(self):
        """Restoring a state with the wrong shape must raise ValueError,
        not silently load partial state.
        """
        cfg_small = TiedSubstrateConfig(in_dim=8, out_dim=16)
        cfg_large = TiedSubstrateConfig(in_dim=16, out_dim=32)

        small = TiedSubstrate(cfg_small)
        large = TiedSubstrate(cfg_large)

        state_small = small.serialize()

        with pytest.raises(ValueError, match="shape mismatch"):
            large.restore(state_small)


# =========================================================================
# Ablation flag behavior
# =========================================================================

class TestAblation:
    """When enable_tied_substrate is False, both directions must return
    zeros of the appropriate shape. This is the standard ablation behavior
    from the cognitive-loop ablation flag standard.
    """

    def test_ablation_returns_zeros_in_a_to_b(self):
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=16, enable_tied_substrate=False)
        sub = TiedSubstrate(cfg)
        x_a = torch.randn(4, 8)
        y_b = sub.forward_a_to_b(x_a)
        assert y_b.shape == (4, 16)
        assert torch.allclose(y_b, torch.zeros_like(y_b))

    def test_ablation_returns_zeros_in_b_to_a(self):
        cfg = TiedSubstrateConfig(in_dim=8, out_dim=16, enable_tied_substrate=False)
        sub = TiedSubstrate(cfg)
        x_b = torch.randn(4, 16)
        y_a = sub.forward_b_to_a(x_b)
        assert y_a.shape == (4, 8)
        assert torch.allclose(y_a, torch.zeros_like(y_a))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
