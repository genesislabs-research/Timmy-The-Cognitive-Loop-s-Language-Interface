"""
test_lexical_substrate.py
Tests for the parent lexical substrate module.

Each test case targets a specific architectural claim from the gap
document and the Phase 3 build list. The cases cover ownership
semantics (single source of truth across the wrappers and the
mutation interface), buffer-versus-parameter discipline, atomicity
of row writes, decay of cleared rows, reinforcement convergence, and
state_dict round-trip preservation.

Run with pytest from the repository root.
"""

from __future__ import annotations

import torch

from substrate.lemma_slots_t import (
    RESERVED_SLOTS,
    SLOT_I_DONT_KNOW,
    SLOT_SELF_LEMMA,
    SLOT_POLAR_QUESTION,
    is_reserved,
    is_uncertainty_marker,
    is_identity_marker,
    name_for,
    slot_for,
)
from substrate.lexical_substrate_t import (
    BufferTiedSubstrate,
    LexicalSubstrate,
    LexicalSubstrateConfig,
)


# =========================================================================
# Helpers
# =========================================================================

def make_substrate(
    n_concepts: int = 1024,
    n_lemmas: int = 64,
    d_phon: int = 128,
) -> LexicalSubstrate:
    """Construct a small substrate for tests.

    The defaults keep n_lemmas small enough that exhaustive iteration
    over slots is cheap, while preserving n_concepts at its default
    1024 so the architect's frame-bias and uncertainty-subspace
    reservations at dimensions 1000-1023 are real.
    """
    cfg = LexicalSubstrateConfig(
        n_concepts=n_concepts,
        n_lemmas=n_lemmas,
        d_phon=d_phon,
    )
    return LexicalSubstrate(cfg)


def random_concept(n_concepts: int = 1024) -> torch.Tensor:
    """Return a random concept vector with unit norm."""
    v = torch.randn(n_concepts)
    return v / v.norm()


def random_phon(d_phon: int = 128) -> torch.Tensor:
    """Return a random phonological code with unit norm."""
    v = torch.randn(d_phon)
    return v / v.norm()


# =========================================================================
# Construction and initial state
# =========================================================================

class TestConstruction:
    """Tests verifying the substrate constructs cleanly and that its
    initial state matches the architectural commitments."""

    def test_matrices_are_buffers_not_parameters(self):
        """Architectural commitment: matrices are register_buffer
        rather than nn.Parameter. The Reconamics frame requires
        experience to live in buffer state, not optimizer state.
        """
        s = make_substrate()
        # Both matrices must appear in state_dict (buffers do, parameters
        # do too) but must not appear in parameters() (parameters do,
        # buffers do not).
        param_names = {name for name, _ in s.named_parameters()}
        buffer_names = {name for name, _ in s.named_buffers()}

        assert "W_C_to_L" not in param_names, (
            "W_C_to_L is a Parameter; should be a buffer."
        )
        assert "W_L_to_P" not in param_names, (
            "W_L_to_P is a Parameter; should be a buffer."
        )
        assert "W_C_to_L" in buffer_names, (
            "W_C_to_L is missing from buffers."
        )
        assert "W_L_to_P" in buffer_names, (
            "W_L_to_P is missing from buffers."
        )

    def test_matrices_initialize_to_zero(self):
        """Architectural commitment: zero initialization at cold-start.
        The lemma-local novelty gate at Wernicke's depends on this so
        that all phonological codes register as novel until allocation
        populates rows.
        """
        s = make_substrate()
        assert torch.all(s.W_C_to_L == 0), (
            "W_C_to_L not zero-initialized."
        )
        assert torch.all(s.W_L_to_P == 0), (
            "W_L_to_P not zero-initialized."
        )

    def test_matrix_shapes_match_config(self):
        """Architectural commitment: shapes are (n_lemmas, n_concepts)
        and (d_phon, n_lemmas) per the corpus tied-weights idiom in
        Appendix F.
        """
        s = make_substrate(n_concepts=256, n_lemmas=64, d_phon=128)
        assert s.W_C_to_L.shape == (64, 256), (
            f"W_C_to_L shape {tuple(s.W_C_to_L.shape)}, "
            f"expected (64, 256)."
        )
        assert s.W_L_to_P.shape == (128, 64), (
            f"W_L_to_P shape {tuple(s.W_L_to_P.shape)}, "
            f"expected (128, 64)."
        )

    def test_tied_substrate_wrappers_present(self):
        """The two BufferTiedSubstrate wrappers exist and reference
        the parent's buffers, not separate copies."""
        s = make_substrate()
        assert isinstance(s.tied_w_c_to_l, BufferTiedSubstrate)
        assert isinstance(s.tied_w_l_to_p, BufferTiedSubstrate)
        # The wrapper's W reference must be the same object as the
        # parent's buffer. data_ptr() identity is the strongest test.
        assert (
            s.tied_w_c_to_l.W.data_ptr() == s.W_C_to_L.data_ptr()
        ), "tied_w_c_to_l holds a copy rather than a reference."
        assert (
            s.tied_w_l_to_p.W.data_ptr() == s.W_L_to_P.data_ptr()
        ), "tied_w_l_to_p holds a copy rather than a reference."


# =========================================================================
# Single-source-of-truth invariant
# =========================================================================

class TestSingleSourceOfTruth:
    """Tests verifying that mutations through any of the three
    interfaces (parent, wrapper, direct buffer access) are
    immediately visible to all three. This is the architectural
    commitment that resolution-3 of the matrix ownership question
    enforces.
    """

    def test_write_row_visible_through_wrapper_forward(self):
        """A row written through the parent's write_row must be
        visible on the very next forward call through the
        BufferTiedSubstrate wrapper, with no synchronization step.
        """
        s = make_substrate(n_lemmas=32)
        slot = 17  # First non-reserved slot.

        concept = random_concept(s.cfg.n_concepts)
        phon = random_phon(s.cfg.d_phon)
        s.write_row(slot, concept, phon)

        # Forward concept-to-lemma with a one-hot at the written
        # concept direction. The result should activate the slot.
        # Build a concept input that is approximately the concept
        # vector itself, and check that the slot's lemma activation
        # is the highest in the output.
        concept_input = concept.unsqueeze(0)  # (1, n_concepts)
        lemma_out = s.forward_concept_to_lemma(concept_input)
        winning_slot = int(lemma_out.argmax(dim=1).item())
        assert winning_slot == slot, (
            f"Forward call did not see written row. Winning slot "
            f"{winning_slot}, wrote {slot}."
        )

    def test_write_row_visible_through_phonological_forward(self):
        """A row written through write_row must also be visible on
        the W_L_to_P side, exercised by forward_lemma_to_phonological.
        """
        s = make_substrate(n_lemmas=32)
        slot = 17

        concept = random_concept(s.cfg.n_concepts)
        phon = random_phon(s.cfg.d_phon)
        s.write_row(slot, concept, phon)

        # Drive the slot directly with a one-hot lemma input. The
        # phonological output should match phon up to numerical
        # precision because the matrix multiplies a one-hot input by
        # the column at index `slot`, which we just wrote to be phon.
        lemma_input = torch.zeros(1, s.cfg.n_lemmas)
        lemma_input[0, slot] = 1.0
        phon_out = s.forward_lemma_to_phonological(lemma_input)
        assert torch.allclose(phon_out[0], phon, atol=1e-6), (
            "Phonological output did not match the written row."
        )

    def test_write_row_visible_through_read(self):
        """The read_concept_row and read_phonological_row methods see
        the same content that the wrapper does."""
        s = make_substrate(n_lemmas=32)
        slot = 17

        concept = random_concept(s.cfg.n_concepts)
        phon = random_phon(s.cfg.d_phon)
        s.write_row(slot, concept, phon)

        assert torch.allclose(
            s.read_concept_row(slot), concept, atol=1e-6,
        )
        assert torch.allclose(
            s.read_phonological_row(slot), phon, atol=1e-6,
        )

    def test_clear_row_zeros_both_matrices(self):
        """clear_row must zero both the W_C_to_L row and the W_L_to_P
        column atomically. Used by decay_unconfirmed.
        """
        s = make_substrate(n_lemmas=32)
        slot = 17

        s.write_row(
            slot,
            random_concept(s.cfg.n_concepts),
            random_phon(s.cfg.d_phon),
        )
        # Sanity: row is non-zero before clearing.
        assert s.read_concept_row(slot).abs().sum() > 0
        assert s.read_phonological_row(slot).abs().sum() > 0

        s.clear_row(slot)
        assert torch.all(s.read_concept_row(slot) == 0), (
            "clear_row did not zero the W_C_to_L row."
        )
        assert torch.all(s.read_phonological_row(slot) == 0), (
            "clear_row did not zero the W_L_to_P column."
        )


# =========================================================================
# Reinforcement
# =========================================================================

class TestReinforcement:
    """Tests verifying the Hebbian leaky-integrator reinforcement
    rule. Reinforcement should pull the row toward the target without
    overshooting, and repeated reinforcement should converge."""

    def test_reinforce_moves_toward_target(self):
        """A single reinforcement step moves the row partway toward
        the target, with the distance decreasing by approximately
        learning_rate."""
        s = make_substrate(n_lemmas=32)
        slot = 17

        # Start from an existing row.
        initial_concept = random_concept(s.cfg.n_concepts)
        initial_phon = random_phon(s.cfg.d_phon)
        s.write_row(slot, initial_concept, initial_phon)

        # Reinforce toward a different target.
        target_concept = random_concept(s.cfg.n_concepts)
        target_phon = random_phon(s.cfg.d_phon)
        s.reinforce_row(
            slot, target_concept, target_phon, learning_rate=0.1,
        )

        # The row should have moved a small amount toward the target.
        # Specifically, new_W = old_W + 0.1 * (target - old_W) =
        # 0.9 * old_W + 0.1 * target.
        expected_concept = (
            0.9 * initial_concept + 0.1 * target_concept
        )
        expected_phon = 0.9 * initial_phon + 0.1 * target_phon
        assert torch.allclose(
            s.read_concept_row(slot), expected_concept, atol=1e-6,
        )
        assert torch.allclose(
            s.read_phonological_row(slot), expected_phon, atol=1e-6,
        )

    def test_reinforce_converges_with_repetition(self):
        """Many reinforcement steps with the same target converge
        the row close to the target. The leaky-integrator update
        rule has a geometric convergence rate."""
        s = make_substrate(n_lemmas=32)
        slot = 17

        s.write_row(
            slot,
            random_concept(s.cfg.n_concepts),
            random_phon(s.cfg.d_phon),
        )

        target_concept = random_concept(s.cfg.n_concepts)
        target_phon = random_phon(s.cfg.d_phon)

        for _ in range(100):
            s.reinforce_row(
                slot, target_concept, target_phon,
                learning_rate=0.1,
            )

        # After 100 steps with lr=0.1 the residual should be
        # 0.9^100 ≈ 2.66e-5 of the initial distance, which we treat
        # as effectively converged.
        concept_residual = (
            s.read_concept_row(slot) - target_concept
        ).norm().item()
        phon_residual = (
            s.read_phonological_row(slot) - target_phon
        ).norm().item()
        assert concept_residual < 1e-3, (
            f"Concept residual {concept_residual} did not converge."
        )
        assert phon_residual < 1e-3, (
            f"Phonological residual {phon_residual} did not converge."
        )


# =========================================================================
# Validation
# =========================================================================

class TestValidation:
    """Tests verifying that invalid inputs raise the right exceptions
    rather than silently corrupting state."""

    def test_write_row_rejects_wrong_concept_shape(self):
        """Wrong-shape concept_vector should raise rather than
        partial-write."""
        s = make_substrate()
        slot = 17

        bad_concept = torch.zeros(7)  # not (n_concepts,)
        good_phon = random_phon(s.cfg.d_phon)
        try:
            s.write_row(slot, bad_concept, good_phon)
            assert False, "Expected ValueError on bad concept shape."
        except ValueError:
            pass
        # The matrices should remain untouched.
        assert torch.all(s.W_C_to_L == 0)
        assert torch.all(s.W_L_to_P == 0)

    def test_write_row_rejects_wrong_phon_shape(self):
        """Wrong-shape phonological_code should raise."""
        s = make_substrate()
        slot = 17

        good_concept = random_concept(s.cfg.n_concepts)
        bad_phon = torch.zeros(7)
        try:
            s.write_row(slot, good_concept, bad_phon)
            assert False, "Expected ValueError on bad phon shape."
        except ValueError:
            pass

    def test_write_row_rejects_out_of_range_slot(self):
        """slot_index outside [0, n_lemmas) should raise IndexError."""
        s = make_substrate(n_lemmas=64)
        try:
            s.write_row(
                64,  # equals n_lemmas, out of range
                random_concept(s.cfg.n_concepts),
                random_phon(s.cfg.d_phon),
            )
            assert False, "Expected IndexError on out-of-range slot."
        except IndexError:
            pass

    def test_reinforce_rejects_bad_learning_rate(self):
        """learning_rate must be in (0, 1]."""
        s = make_substrate()
        slot = 17
        s.write_row(
            slot,
            random_concept(s.cfg.n_concepts),
            random_phon(s.cfg.d_phon),
        )

        for bad_lr in [0.0, -0.1, 1.1, 2.0]:
            try:
                s.reinforce_row(
                    slot,
                    random_concept(s.cfg.n_concepts),
                    random_phon(s.cfg.d_phon),
                    learning_rate=bad_lr,
                )
                assert False, (
                    f"Expected ValueError on lr={bad_lr}."
                )
            except ValueError:
                pass


# =========================================================================
# Ablation
# =========================================================================

class TestAblation:
    """Tests verifying the master flag's neutral-output behavior."""

    def test_disabled_substrate_returns_zero_forward(self):
        """When enable_lexical_substrate is False, forward calls
        return zero tensors of the right shape."""
        cfg = LexicalSubstrateConfig(enable_lexical_substrate=False)
        s = LexicalSubstrate(cfg)
        # Even if matrices contained data, forward should return zeros.
        s.W_C_to_L.fill_(1.0)
        s.W_L_to_P.fill_(1.0)

        concept = torch.randn(2, cfg.n_concepts)
        out = s.forward_concept_to_lemma(concept)
        assert torch.all(out == 0)
        assert out.shape == (2, cfg.n_lemmas)

        lemma = torch.randn(2, cfg.n_lemmas)
        out = s.forward_lemma_to_phonological(lemma)
        assert torch.all(out == 0)
        assert out.shape == (2, cfg.d_phon)


# =========================================================================
# Serialization round-trip
# =========================================================================

class TestSerialization:
    """Tests verifying that the substrate round-trips through
    state_dict without losing the matrix contents. This is the
    foundation of the .soul checkpoint protocol that Phase 1 builds
    on top of."""

    def test_state_dict_round_trip_preserves_matrices(self):
        """After save and load, the matrices must be bit-identical."""
        s_original = make_substrate(n_lemmas=32)
        # Write a few rows.
        for slot in [17, 19, 23]:
            s_original.write_row(
                slot,
                random_concept(s_original.cfg.n_concepts),
                random_phon(s_original.cfg.d_phon),
            )

        state = s_original.state_dict()

        s_restored = make_substrate(n_lemmas=32)
        s_restored.load_state_dict(state)

        assert torch.allclose(
            s_original.W_C_to_L, s_restored.W_C_to_L, atol=0.0,
        )
        assert torch.allclose(
            s_original.W_L_to_P, s_restored.W_L_to_P, atol=0.0,
        )

    def test_round_trip_preserves_forward_behavior(self):
        """A restored substrate must produce identical forward outputs
        for the same input as the original. This is the operational
        guarantee that the .soul checkpoint relies on."""
        s_original = make_substrate(n_lemmas=32)
        for slot in [17, 19, 23]:
            s_original.write_row(
                slot,
                random_concept(s_original.cfg.n_concepts),
                random_phon(s_original.cfg.d_phon),
            )

        state = s_original.state_dict()
        s_restored = make_substrate(n_lemmas=32)
        s_restored.load_state_dict(state)

        # The restored TiedSubstrate wrappers must point to the
        # restored buffers, not stale references. Test this by
        # running a forward call and comparing.
        concept = random_concept(s_original.cfg.n_concepts).unsqueeze(0)
        out_original = s_original.forward_concept_to_lemma(concept)
        out_restored = s_restored.forward_concept_to_lemma(concept)
        assert torch.allclose(out_original, out_restored, atol=1e-6)


# =========================================================================
# Slot inventory
# =========================================================================

class TestSlotInventory:
    """Tests verifying the unified slot inventory's invariants. The
    slot table is consumed by mid-MTG, the identity module, the
    uncertainty router, the frame recognizer, and the lemma
    acquisition module, so these invariants are load-bearing."""

    def test_reserved_slots_count_is_correct(self):
        """RESERVED_SLOTS = 17, which is 2 identity + 6 wh-words +
        1 polar_question + 8 uncertainty = 17."""
        assert RESERVED_SLOTS == 17

    def test_slot_indices_are_contiguous_from_zero(self):
        """The reserved slots occupy indices 0 through 16 with no
        gaps. Required for find_free_slot's loop boundary logic."""
        from substrate.lemma_slots_t import RESERVED_SLOT_INDICES
        assert RESERVED_SLOT_INDICES == frozenset(range(RESERVED_SLOTS))

    def test_self_lemma_is_slot_zero(self):
        """self_lemma is at index 0 by convention. The identity
        module's perception-side routing depends on this constant."""
        assert SLOT_SELF_LEMMA == 0
        assert slot_for("self_lemma") == 0

    def test_polar_question_is_slot_eight(self):
        """polar_question is at index 8. The lemma_acquisition
        module's polar-question co-activation reads this slot."""
        assert SLOT_POLAR_QUESTION == 8

    def test_i_dont_know_is_slot_nine(self):
        """i_dont_know is at index 9. The production fall-through
        routes here when no allocated lemma exceeds threshold."""
        assert SLOT_I_DONT_KNOW == 9
        assert slot_for("i_dont_know") == 9

    def test_is_reserved_returns_correct_classification(self):
        for slot in range(RESERVED_SLOTS):
            assert is_reserved(slot)
        for slot in [RESERVED_SLOTS, RESERVED_SLOTS + 1, 100]:
            assert not is_reserved(slot)

    def test_is_uncertainty_marker_classifies_uncertainty_slots(self):
        for slot in range(9, 17):  # uncertainty slots 9-16
            assert is_uncertainty_marker(slot)
        # Identity and question slots are not uncertainty.
        for slot in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            assert not is_uncertainty_marker(slot)

    def test_is_identity_marker_classifies_identity_slots(self):
        for slot in [0, 1]:
            assert is_identity_marker(slot)
        for slot in range(2, RESERVED_SLOTS):
            assert not is_identity_marker(slot)

    def test_name_for_returns_canonical_names(self):
        assert name_for(0) == "self_lemma"
        assert name_for(8) == "polar_question"
        assert name_for(9) == "i_dont_know"
        assert name_for(RESERVED_SLOTS) is None  # not reserved

    def test_slot_for_raises_on_unknown_name(self):
        try:
            slot_for("not_a_real_lemma")
            assert False, "Expected KeyError on unknown name."
        except KeyError:
            pass


# =========================================================================
# Diagnostics
# =========================================================================

class TestDiagnostics:
    """Tests verifying the diagnostic-state reporter produces useful
    output for monitoring and integration testing."""

    def test_diagnostic_state_reports_zero_norms_at_init(self):
        s = make_substrate()
        d = s.get_diagnostic_state()
        assert d["W_C_to_L_norm"] == 0.0
        assert d["W_L_to_P_norm"] == 0.0
        assert d["W_C_to_L_n_nonzero_rows"] == 0
        assert d["W_L_to_P_n_nonzero_cols"] == 0

    def test_diagnostic_state_reflects_writes(self):
        s = make_substrate(n_lemmas=32)
        s.write_row(
            17,
            random_concept(s.cfg.n_concepts),
            random_phon(s.cfg.d_phon),
        )
        s.write_row(
            19,
            random_concept(s.cfg.n_concepts),
            random_phon(s.cfg.d_phon),
        )
        d = s.get_diagnostic_state()
        assert d["W_C_to_L_n_nonzero_rows"] == 2
        assert d["W_L_to_P_n_nonzero_cols"] == 2
        assert d["W_C_to_L_norm"] > 0
        assert d["W_L_to_P_norm"] > 0
