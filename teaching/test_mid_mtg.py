"""
test_mid_mtg.py
Tests for the mid-MTG lemma stratum.

These tests verify:
    1. Reserved identity and uncertainty lemma slots are present and
       correctly indexed.
    2. Production direction with a "trained" lemma (manually planted
       weight row) produces a sharp activation peak on the right slot
       and high lemma confidence.
    3. Production direction with a flat random concept distribution
       on an untrained substrate produces low lemma confidence.
    4. Comprehension direction recovers a concept distribution from a
       lemma activation pattern.
    5. The selection gate respects t_lemma_steps.
    6. Serialization round-trip preserves activation, weights, and
       step counter.
    7. Ablation flag forces neutral returns.
"""

from __future__ import annotations

import pytest
import torch

from regions.mid_mtg_t import (
    MidMTG,
    MidMTGConfig,
    IDENTITY_LEMMA_SLOTS,
    UNCERTAINTY_LEMMA_SLOTS,
    QUESTION_LEMMA_SLOTS,
    N_RESERVED_LEMMAS,
)


# =========================================================================
# Helpers
# =========================================================================

def _plant_lemma(midmtg: MidMTG, lemma_idx: int, concept_idx: int) -> None:
    """Manually set the W_C_to_L matrix so that concept concept_idx
    activates lemma lemma_idx strongly, and mark the lemma as allocated.
    This simulates what Phase 3 acquisition would produce after the
    substrate has learned the concept-lemma pairing.

    The scaffold uses this rather than running real acquisition because
    Phase 3 acquisition requires the full pipeline (Wernicke's, the
    perception loop, the loss) and is not in this file's test scope.

    The weight magnitude (20.0) is chosen to dominate lateral interference
    from the other pre-allocated slots over a typical t_lemma_steps
    integration window. A real acquisition pipeline would produce
    weights of similar magnitude after enough exposures; the magnitude
    is not a biological quantity, just a stand-in for "thoroughly
    trained."
    """
    with torch.no_grad():
        midmtg.w_c_to_l.W[lemma_idx, :] = 0.0
        midmtg.w_c_to_l.W[lemma_idx, concept_idx] = 20.0
    midmtg.allocate_lemma(lemma_idx)


# =========================================================================
# Reserved slot machinery
# =========================================================================

class TestReservedSlots:

    def test_identity_slots_present(self):
        """Both identity slots must be defined."""
        assert "self_lemma" in IDENTITY_LEMMA_SLOTS
        assert "other_lemma" in IDENTITY_LEMMA_SLOTS

    def test_uncertainty_slots_include_required_lemmas(self):
        """The Section 24.7.4 reservation specifies 5 to 10 slots; the
        scaffold uses 8 covering the most common uncertainty markers.
        """
        required = {"i_dont_know", "im_not_sure", "maybe", "i_think"}
        assert required.issubset(UNCERTAINTY_LEMMA_SLOTS.keys())
        assert 5 <= len(UNCERTAINTY_LEMMA_SLOTS) <= 10

    def test_no_slot_index_collision(self):
        """The reserved slot indices must be unique. Identity, uncertainty,
        and question slots together must occupy distinct integer indices.
        """
        all_indices = (
            list(IDENTITY_LEMMA_SLOTS.values())
            + list(UNCERTAINTY_LEMMA_SLOTS.values())
            + list(QUESTION_LEMMA_SLOTS.values())
        )
        assert len(all_indices) == len(set(all_indices)), (
            "Reserved lemma slot indices collide."
        )

    def test_reserved_lemmas_at_low_indices(self):
        """The first N_RESERVED_LEMMAS slots are reserved. New lemmas
        from acquisition should start at index N_RESERVED_LEMMAS.
        """
        all_indices = (
            list(IDENTITY_LEMMA_SLOTS.values())
            + list(UNCERTAINTY_LEMMA_SLOTS.values())
            + list(QUESTION_LEMMA_SLOTS.values())
        )
        assert max(all_indices) < N_RESERVED_LEMMAS

    def test_question_slots_present(self):
        """The six question-word slots specified in Genesis Teaching must
        all be defined.
        """
        for word in ("what", "who", "where", "when", "why", "how"):
            assert word in QUESTION_LEMMA_SLOTS

    def test_question_slot_lookup(self):
        cfg = MidMTGConfig()
        midmtg = MidMTG(cfg)
        for name, expected_idx in QUESTION_LEMMA_SLOTS.items():
            assert midmtg.question_slot(name) == expected_idx

    def test_question_slots_pre_allocated(self):
        """Question slots must be pre-allocated by construction so that
        a peak landing on a question lemma reads as a real signal rather
        than as an unallocated noise peak.
        """
        cfg = MidMTGConfig()
        midmtg = MidMTG(cfg)
        for slot_idx in QUESTION_LEMMA_SLOTS.values():
            assert midmtg.is_allocated[slot_idx].item(), (
                f"Question slot {slot_idx} should be pre-allocated."
            )

    def test_get_question_lemma_slots_returns_full_mapping(self):
        cfg = MidMTGConfig()
        midmtg = MidMTG(cfg)
        slots = midmtg.get_question_lemma_slots()
        assert slots == QUESTION_LEMMA_SLOTS

    def test_unknown_question_slot_raises(self):
        cfg = MidMTGConfig()
        midmtg = MidMTG(cfg)
        with pytest.raises(KeyError, match="Unknown question slot"):
            midmtg.question_slot("nonexistent")

    def test_identity_slot_lookup(self):
        """The identity_slot accessor must return the right index."""
        cfg = MidMTGConfig()
        midmtg = MidMTG(cfg)
        assert midmtg.identity_slot("self_lemma") == IDENTITY_LEMMA_SLOTS["self_lemma"]
        assert midmtg.identity_slot("other_lemma") == IDENTITY_LEMMA_SLOTS["other_lemma"]

    def test_unknown_identity_slot_raises(self):
        cfg = MidMTGConfig()
        midmtg = MidMTG(cfg)
        with pytest.raises(KeyError, match="Unknown identity slot"):
            midmtg.identity_slot("nonexistent")

    def test_unknown_uncertainty_slot_raises(self):
        cfg = MidMTGConfig()
        midmtg = MidMTG(cfg)
        with pytest.raises(KeyError, match="Unknown uncertainty slot"):
            midmtg.uncertainty_slot("nonexistent")


# =========================================================================
# Production direction
# =========================================================================

class TestProductionDirection:

    def test_trained_concept_produces_sharp_peak(self):
        """When the substrate has 'learned' a concept-lemma pair (we
        plant the weight by hand), feeding a one-hot concept distribution
        for that concept must produce a sharp activation peak on the
        target lemma after the integration window.
        """
        cfg = MidMTGConfig(n_concepts=64, n_lemmas=512, t_lemma_steps=5)
        midmtg = MidMTG(cfg)

        target_lemma = N_RESERVED_LEMMAS  # First non-reserved slot.
        target_concept = 17
        _plant_lemma(midmtg, target_lemma, target_concept)

        c_lex = torch.zeros(1, cfg.n_concepts)
        c_lex[0, target_concept] = 1.0

        # Run integration for t_lemma_steps ticks.
        midmtg.reset_for_selection()
        for _ in range(cfg.t_lemma_steps):
            midmtg.forward_production(c_lex)

        winner = midmtg.select_lemma()
        assert winner is not None, (
            "Selection should be available after t_lemma_steps integration."
        )
        winning_idx = winner.argmax(dim=1).item()
        assert winning_idx == target_lemma, (
            f"Expected lemma {target_lemma} to win, got {winning_idx}."
        )

    def test_trained_concept_produces_high_confidence(self):
        """A sharp peak corresponds to high lemma confidence."""
        cfg = MidMTGConfig(n_concepts=64, n_lemmas=512, t_lemma_steps=5)
        midmtg = MidMTG(cfg)

        target_lemma = N_RESERVED_LEMMAS
        target_concept = 17
        _plant_lemma(midmtg, target_lemma, target_concept)

        c_lex = torch.zeros(1, cfg.n_concepts)
        c_lex[0, target_concept] = 1.0

        midmtg.reset_for_selection()
        for _ in range(cfg.t_lemma_steps):
            midmtg.forward_production(c_lex)

        confidence = midmtg.get_lemma_confidence()
        assert confidence.item() > 0.3, (
            f"Trained concept should produce high lemma confidence, "
            f"got {confidence.item():.3f}."
        )

    def test_unallocated_concept_floors_confidence(self):
        """Phase 8 cold-start dialogue case: the substrate has acquired
        some lemmas but the current concept does not match any of them.
        The peak activation lands on an unallocated slot (whichever
        random initialization gives the largest noise activation), and
        confidence floors regardless of how flat or peaked the
        distribution looks.

        This is the architectural commitment that "I don't know what
        the world is" works correctly even after the substrate has
        learned its name. Lemma confidence reads allocation status, not
        just activation statistics.
        """
        torch.manual_seed(0)
        cfg = MidMTGConfig(n_concepts=64, n_lemmas=512, t_lemma_steps=5)
        midmtg = MidMTG(cfg)

        # Allocate one lemma for one concept (simulating having
        # acquired lemma_513 for TIMMY).
        learned_lemma = N_RESERVED_LEMMAS
        learned_concept = 17
        _plant_lemma(midmtg, learned_lemma, learned_concept)

        # Now feed a different concept (simulating the WORLD query
        # which has no acquired lemma).
        novel_concept = 42
        c_lex = torch.zeros(1, cfg.n_concepts)
        c_lex[0, novel_concept] = 1.0

        midmtg.reset_for_selection()
        for _ in range(cfg.t_lemma_steps):
            midmtg.forward_production(c_lex)

        confidence = midmtg.get_lemma_confidence().item()
        # The peak lands on an unallocated slot (the random
        # initialization gives some unallocated lemma the highest
        # activation by chance), so confidence floors to zero.
        assert confidence == pytest.approx(0.0, abs=1e-6), (
            f"Unallocated concept should floor confidence to zero, "
            f"got {confidence:.6f}."
        )

    def test_trained_confidence_above_unallocated_floor(self):
        """Confidence on a trained concept should be meaningfully above
        the floored confidence for an unallocated concept. This is the
        relative property mid-MTG must satisfy: the confidence signal
        is informative.
        """
        torch.manual_seed(0)
        cfg = MidMTGConfig(n_concepts=64, n_lemmas=512, t_lemma_steps=5)
        midmtg = MidMTG(cfg)

        learned_lemma = N_RESERVED_LEMMAS
        learned_concept = 17
        _plant_lemma(midmtg, learned_lemma, learned_concept)

        # Trained input.
        c_match = torch.zeros(1, cfg.n_concepts)
        c_match[0, learned_concept] = 1.0
        midmtg.reset_for_selection()
        for _ in range(cfg.t_lemma_steps):
            midmtg.forward_production(c_match)
        trained_conf = midmtg.get_lemma_confidence().item()

        midmtg.reset_state()

        # Unallocated input.
        c_novel = torch.zeros(1, cfg.n_concepts)
        c_novel[0, 42] = 1.0
        midmtg.reset_for_selection()
        for _ in range(cfg.t_lemma_steps):
            midmtg.forward_production(c_novel)
        novel_conf = midmtg.get_lemma_confidence().item()

        assert trained_conf > 0.2 and novel_conf < 0.05, (
            f"Trained concept should produce meaningful confidence "
            f"({trained_conf:.3f}) and unallocated concept should floor "
            f"({novel_conf:.3f})."
        )

    def test_persistence_carries_activation_across_ticks(self):
        """With persistence enabled, lemma activation at tick t depends
        on activation at tick t-1. Disabling persistence should make the
        activation purely a function of the current input.
        """
        cfg_with = MidMTGConfig(
            n_concepts=64, n_lemmas=512,
            enable_persistence=True, gamma_lemma=0.95,
        )
        cfg_without = MidMTGConfig(
            n_concepts=64, n_lemmas=512,
            enable_persistence=False,
        )
        midmtg_with = MidMTG(cfg_with)
        midmtg_without = MidMTG(cfg_without)

        # Match weights so the only difference is persistence.
        with torch.no_grad():
            midmtg_without.w_c_to_l.W.copy_(midmtg_with.w_c_to_l.W)

        c_lex = torch.randn(1, 64)
        c_lex = torch.softmax(c_lex, dim=1)

        midmtg_with.reset_for_selection()
        midmtg_without.reset_for_selection()

        # Run multiple ticks. The persistent version should accumulate
        # activation; the non-persistent version should not.
        for _ in range(10):
            a_with = midmtg_with.forward_production(c_lex).clone()
            a_without = midmtg_without.forward_production(c_lex).clone()

        # The persistent version's max activation should exceed the
        # non-persistent version's by a factor reflecting the
        # accumulation. Conservative check: the persistent version is
        # strictly larger in max magnitude.
        assert a_with.abs().max() > a_without.abs().max(), (
            "Persistence should accumulate activation across ticks."
        )


# =========================================================================
# Comprehension direction
# =========================================================================

class TestComprehensionDirection:

    def test_comprehension_recovers_concept_signature(self):
        """Feeding a one-hot lemma activation through the comprehension
        direction should recover the concept-space signature of that
        lemma (the corresponding row of W_C_to_L).
        """
        cfg = MidMTGConfig(n_concepts=64, n_lemmas=512)
        midmtg = MidMTG(cfg)

        target_lemma = N_RESERVED_LEMMAS
        target_concept = 17
        _plant_lemma(midmtg, target_lemma, target_concept)

        a_lemma = torch.zeros(1, cfg.n_lemmas)
        a_lemma[0, target_lemma] = 1.0

        c_recovered = midmtg.forward_comprehension(a_lemma)
        # The recovered concept distribution should peak at the planted
        # concept index.
        peak_idx = c_recovered.argmax(dim=1).item()
        assert peak_idx == target_concept, (
            f"Comprehension should recover concept {target_concept}, "
            f"got peak at {peak_idx}."
        )


# =========================================================================
# Selection gating
# =========================================================================

class TestSelectionGating:

    def test_selection_blocked_before_t_lemma(self):
        """select_lemma() must return None before t_lemma_steps integration."""
        cfg = MidMTGConfig(n_concepts=64, n_lemmas=512, t_lemma_steps=10)
        midmtg = MidMTG(cfg)

        c_lex = torch.zeros(1, 64)
        c_lex[0, 0] = 1.0

        midmtg.reset_for_selection()
        for _ in range(5):  # Fewer than t_lemma_steps.
            midmtg.forward_production(c_lex)

        assert midmtg.select_lemma() is None

    def test_selection_available_after_t_lemma(self):
        cfg = MidMTGConfig(n_concepts=64, n_lemmas=512, t_lemma_steps=10)
        midmtg = MidMTG(cfg)

        c_lex = torch.zeros(1, 64)
        c_lex[0, 0] = 1.0

        midmtg.reset_for_selection()
        for _ in range(10):
            midmtg.forward_production(c_lex)

        result = midmtg.select_lemma()
        assert result is not None
        assert result.shape == (1, 512)
        # Must be a valid one-hot vector.
        assert result.sum(dim=1).item() == pytest.approx(1.0)


# =========================================================================
# Serialization
# =========================================================================

class TestSerialization:

    def test_round_trip_preserves_state(self):
        """Save and restore mid-MTG, verify activation, weights, step
        counter, and similarity embedding all survive.
        """
        cfg = MidMTGConfig(n_concepts=64, n_lemmas=512, t_lemma_steps=5)
        original = MidMTG(cfg)

        target_lemma = N_RESERVED_LEMMAS
        _plant_lemma(original, target_lemma, 17)

        c_lex = torch.zeros(1, cfg.n_concepts)
        c_lex[0, 17] = 1.0
        original.reset_for_selection()
        for _ in range(3):  # Less than t_lemma_steps so step counter is mid-window.
            original.forward_production(c_lex)

        state = original.serialize()

        restored = MidMTG(cfg)
        restored.restore(state)

        # Weights match.
        assert torch.allclose(
            original.w_c_to_l.W, restored.w_c_to_l.W, atol=1e-9,
        )
        # Activation matches.
        assert torch.allclose(
            original.a_lemma, restored.a_lemma, atol=1e-9,
        )
        # Step counter matches.
        assert (
            original._steps_since_reset.item()
            == restored._steps_since_reset.item()
        )

    def test_restore_with_lemma_count_mismatch_raises(self):
        cfg_small = MidMTGConfig(n_concepts=64, n_lemmas=128)
        cfg_large = MidMTGConfig(n_concepts=64, n_lemmas=512)

        small = MidMTG(cfg_small)
        large = MidMTG(cfg_large)

        state_small = small.serialize()
        with pytest.raises((ValueError,)):
            large.restore(state_small)


# =========================================================================
# Ablation
# =========================================================================

class TestAblation:

    def test_ablation_returns_zeros_in_production(self):
        cfg = MidMTGConfig(enable_mid_mtg=False, n_concepts=64, n_lemmas=512)
        midmtg = MidMTG(cfg)
        c_lex = torch.randn(1, 64)
        out = midmtg.forward_production(c_lex)
        assert out.shape == (1, 512)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_ablation_returns_none_for_selection(self):
        cfg = MidMTGConfig(enable_mid_mtg=False, n_concepts=64, n_lemmas=512)
        midmtg = MidMTG(cfg)
        midmtg.reset_for_selection()
        c_lex = torch.randn(1, 64)
        for _ in range(50):
            midmtg.forward_production(c_lex)
        assert midmtg.select_lemma() is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
