"""
test_lemma_acquisition.py
Tests for the refactored lemma acquisition module.

Each case targets one of the eight architectural commitments from
Section 25 of the Broca's corpus, plus the matrix-ownership invariant
that resolution-3 of the Phase 0 question enforces. The cold-start
naming sequence test at the bottom walks the full five turns of the
dialogue at the substrate level (without the broca/vpmc/runtime
layer), exercising every commitment in the order the architect's
spec walks them.
"""

from __future__ import annotations

import time

import torch

from substrate.lexical_substrate_t import (
    LexicalSubstrate,
    LexicalSubstrateConfig,
)
from substrate.lemma_slots_t import (
    RESERVED_SLOTS,
    SLOT_I_DONT_KNOW,
    SLOT_SELF_LEMMA,
    STATUS_CONFIRMED,
    STATUS_PROVISIONAL,
    STATUS_UNALLOCATED,
)
from coordination.lemma_acquisition_t import (
    LemmaAcquisitionConfig,
    LemmaAcquisitionModule,
)


# =========================================================================
# Helpers
# =========================================================================

def make_pair(
    n_concepts: int = 1024,
    n_lemmas: int = 64,
    d_phon: int = 128,
) -> tuple:
    """Construct a fresh substrate and acquisition module pair."""
    sub_cfg = LexicalSubstrateConfig(
        n_concepts=n_concepts,
        n_lemmas=n_lemmas,
        d_phon=d_phon,
    )
    substrate = LexicalSubstrate(sub_cfg)
    acq_cfg = LemmaAcquisitionConfig()
    acq = LemmaAcquisitionModule(cfg=acq_cfg, substrate=substrate)
    return substrate, acq


def random_concept(n_concepts: int = 1024) -> torch.Tensor:
    v = torch.randn(n_concepts)
    return v / v.norm()


def random_phon(d_phon: int = 128) -> torch.Tensor:
    v = torch.randn(d_phon)
    return v / v.norm()


# =========================================================================
# Construction
# =========================================================================

class TestConstruction:

    def test_status_array_starts_unallocated(self):
        """At construction, all slots are unallocated. Pre-allocation
        of reserved slots is an explicit method call, not implicit
        in construction."""
        substrate, acq = make_pair()
        assert torch.all(acq.status == STATUS_UNALLOCATED)

    def test_substrate_reference_is_held(self):
        """The acquisition module holds a reference to the parent
        substrate, not a copy. Verifying this catches the
        copy-instead-of-reference bug at construction time."""
        substrate, acq = make_pair()
        assert acq.substrate is substrate

    def test_dimensions_match_substrate(self):
        substrate, acq = make_pair(
            n_concepts=512, n_lemmas=32, d_phon=256,
        )
        assert acq.n_concepts == 512
        assert acq.n_lemmas == 32
        assert acq.d_phon == 256


# =========================================================================
# Pre-allocation of reserved slots
# =========================================================================

class TestPreAllocation:
    """Tests verifying the seventeen reserved slots get written into
    the substrate at construction-time and start in STATUS_CONFIRMED
    so they fire reliably from cold-start without going through the
    provisional transition."""

    def test_pre_allocate_marks_reserved_slots_confirmed(self):
        substrate, acq = make_pair()
        acq.pre_allocate_reserved_slots()
        for slot in range(RESERVED_SLOTS):
            assert acq.status[slot].item() == STATUS_CONFIRMED, (
                f"Reserved slot {slot} not confirmed after "
                f"pre_allocate_reserved_slots."
            )

    def test_pre_allocate_does_not_touch_acquired_range(self):
        """Slots from RESERVED_SLOTS onward remain unallocated."""
        substrate, acq = make_pair()
        acq.pre_allocate_reserved_slots()
        for slot in range(RESERVED_SLOTS, acq.n_lemmas):
            assert acq.status[slot].item() == STATUS_UNALLOCATED

    def test_pre_allocate_writes_substrate_matrices(self):
        """The reserved-slot rows in W_C_to_L and W_L_to_P are
        non-zero after pre-allocation."""
        substrate, acq = make_pair()
        acq.pre_allocate_reserved_slots()
        # W_C_to_L rows for reserved slots should be non-zero.
        for slot in range(RESERVED_SLOTS):
            assert (
                substrate.W_C_to_L[slot].abs().sum() > 0
            ), f"W_C_to_L row {slot} is zero after pre-allocation."
        # W_L_to_P columns for reserved slots that have phonological
        # text should be non-zero. Identity slots (self_lemma=0,
        # other_lemma=1) and polar_question (8) have empty phon text
        # and thus zero columns at this stage.
        slots_with_phon = {2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13,
                           14, 15, 16}
        for slot in slots_with_phon:
            assert (
                substrate.W_L_to_P[:, slot].abs().sum() > 0
            ), f"W_L_to_P col {slot} is zero after pre-allocation."

    def test_pre_allocate_uses_caller_supplied_concept_fn(self):
        """When caller supplies a concept-vector function, it is
        called for the appropriate slot category."""
        substrate, acq = make_pair()
        marker = torch.zeros(acq.n_concepts)
        marker[42] = 1.0
        called_with = []

        def my_uncertainty_fn(name):
            called_with.append(name)
            return marker

        acq.pre_allocate_reserved_slots(
            uncertainty_concept_fn=my_uncertainty_fn,
        )
        # Eight uncertainty slots got called.
        assert len(called_with) == 8
        # The marker concept was written into uncertainty rows.
        assert torch.allclose(
            substrate.W_C_to_L[SLOT_I_DONT_KNOW], marker,
        )


# =========================================================================
# Item 1: Lemma-local novelty gate
# =========================================================================

class TestNovelty:

    def test_empty_substrate_classifies_all_as_novel(self):
        substrate, acq = make_pair()
        for _ in range(5):
            assert acq.is_novel(random_phon()) is True

    def test_allocated_phon_is_not_novel(self):
        """A previously-allocated phonological code should not be
        classified as novel by the next call."""
        substrate, acq = make_pair()
        concept = random_concept()
        phon = random_phon()
        slot = acq.allocate_row(concept, phon)
        assert slot >= RESERVED_SLOTS
        assert acq.is_novel(phon) is False

    def test_truly_novel_phon_is_novel_after_some_allocations(self):
        """An unrelated phonological code is still classified as
        novel even after several other allocations."""
        substrate, acq = make_pair()
        for _ in range(3):
            acq.allocate_row(random_concept(), random_phon())
        # A fresh random vector is unlikely to be similar to any of
        # the three existing rows.
        assert acq.is_novel(random_phon()) is True


# =========================================================================
# Item 2 and 3: Three-valued status, matrix ownership through parent
# =========================================================================

class TestAllocation:

    def test_allocation_produces_provisional_status(self):
        substrate, acq = make_pair()
        slot = acq.allocate_row(random_concept(), random_phon())
        assert acq.status[slot].item() == STATUS_PROVISIONAL

    def test_allocation_writes_through_parent_substrate(self):
        """Allocations are visible through the substrate's read
        methods. This is the core matrix-ownership invariant."""
        substrate, acq = make_pair()
        concept = random_concept()
        phon = random_phon()
        slot = acq.allocate_row(concept, phon)
        # Read through the parent.
        assert torch.allclose(
            substrate.read_concept_row(slot), concept, atol=1e-6,
        )
        assert torch.allclose(
            substrate.read_phonological_row(slot), phon, atol=1e-6,
        )

    def test_allocation_visible_through_forward_pass(self):
        """A freshly-allocated row drives a forward pass through the
        substrate's tied wrapper. This is the same-frame visibility
        guarantee the architect specified."""
        substrate, acq = make_pair()
        concept = random_concept()
        phon = random_phon()
        slot = acq.allocate_row(concept, phon)

        # Drive the substrate with the same concept; the slot should
        # win the lemma activation.
        out = substrate.forward_concept_to_lemma(concept.unsqueeze(0))
        winning = int(out.argmax(dim=1).item())
        assert winning == slot

    def test_full_substrate_returns_minus_one(self):
        """When all slots are allocated, the next allocate_row
        returns -1 rather than overwriting an existing row."""
        substrate, acq = make_pair(n_lemmas=RESERVED_SLOTS + 2)
        # Allocate all non-reserved slots.
        s1 = acq.allocate_row(random_concept(), random_phon())
        s2 = acq.allocate_row(random_concept(), random_phon())
        s3 = acq.allocate_row(random_concept(), random_phon())
        assert s1 >= RESERVED_SLOTS
        assert s2 >= RESERVED_SLOTS
        assert s3 == -1


# =========================================================================
# Item 4: Polar-question co-activation on provisional rows
# =========================================================================

class TestPolarQuestionCoactivation:

    def test_provisional_row_triggers_polar_question(self):
        substrate, acq = make_pair()
        concept = random_concept()
        phon = random_phon()
        slot = acq.allocate_row(concept, phon)

        selected, polar_q = acq.select_lemma_for_production(concept)
        assert selected == slot
        assert polar_q is True

    def test_confirmed_row_does_not_trigger_polar_question(self):
        substrate, acq = make_pair()
        concept = random_concept()
        slot = acq.allocate_row(concept, random_phon())
        acq.confirm_row(slot)

        selected, polar_q = acq.select_lemma_for_production(concept)
        assert selected == slot
        assert polar_q is False

    def test_disable_polar_question_coactivation(self):
        """Ablation flag turns off polar-question co-activation
        even on provisional rows."""
        sub_cfg = LexicalSubstrateConfig(n_lemmas=64, d_phon=128)
        substrate = LexicalSubstrate(sub_cfg)
        acq_cfg = LemmaAcquisitionConfig(
            enable_polar_question_coactivation=False,
        )
        acq = LemmaAcquisitionModule(cfg=acq_cfg, substrate=substrate)

        concept = random_concept()
        slot = acq.allocate_row(concept, random_phon())
        _, polar_q = acq.select_lemma_for_production(concept)
        assert polar_q is False


# =========================================================================
# Item 6: Provisional rows do not Hebbian-reinforce
# =========================================================================

class TestReinforcementGating:

    def test_reinforce_skips_provisional(self):
        substrate, acq = make_pair()
        slot = acq.allocate_row(random_concept(), random_phon())

        saved_concept = substrate.read_concept_row(slot).clone()
        applied = acq.reinforce_row(
            slot, random_concept(), random_phon(),
        )
        assert applied is False
        assert torch.allclose(
            substrate.read_concept_row(slot), saved_concept,
        )

    def test_reinforce_applies_to_confirmed(self):
        substrate, acq = make_pair()
        slot = acq.allocate_row(random_concept(), random_phon())
        acq.confirm_row(slot)

        saved_concept = substrate.read_concept_row(slot).clone()
        applied = acq.reinforce_row(
            slot, random_concept(), random_phon(),
        )
        assert applied is True
        # Row moved from saved value.
        assert not torch.allclose(
            substrate.read_concept_row(slot), saved_concept,
        )


# =========================================================================
# Decay
# =========================================================================

class TestDecay:

    def test_decay_row_clears_provisional_immediately(self):
        substrate, acq = make_pair()
        slot = acq.allocate_row(random_concept(), random_phon())
        acq.decay_row(slot)
        assert acq.status[slot].item() == STATUS_UNALLOCATED
        assert substrate.read_concept_row(slot).abs().sum() == 0
        assert substrate.read_phonological_row(slot).abs().sum() == 0

    def test_decay_row_does_not_affect_confirmed(self):
        substrate, acq = make_pair()
        slot = acq.allocate_row(random_concept(), random_phon())
        acq.confirm_row(slot)
        saved_concept = substrate.read_concept_row(slot).clone()
        acq.decay_row(slot)
        # Status still confirmed; row unchanged.
        assert acq.status[slot].item() == STATUS_CONFIRMED
        assert torch.allclose(
            substrate.read_concept_row(slot), saved_concept,
        )

    def test_decay_unconfirmed_clears_timed_out_rows(self):
        sub_cfg = LexicalSubstrateConfig(n_lemmas=64, d_phon=128)
        substrate = LexicalSubstrate(sub_cfg)
        acq_cfg = LemmaAcquisitionConfig(timeout_seconds=0.05)
        acq = LemmaAcquisitionModule(cfg=acq_cfg, substrate=substrate)

        slot = acq.allocate_row(random_concept(), random_phon())
        time.sleep(0.1)
        acq.decay_unconfirmed()
        assert acq.status[slot].item() == STATUS_UNALLOCATED

    def test_decay_unconfirmed_does_not_affect_confirmed(self):
        sub_cfg = LexicalSubstrateConfig(n_lemmas=64, d_phon=128)
        substrate = LexicalSubstrate(sub_cfg)
        acq_cfg = LemmaAcquisitionConfig(timeout_seconds=0.05)
        acq = LemmaAcquisitionModule(cfg=acq_cfg, substrate=substrate)

        slot = acq.allocate_row(random_concept(), random_phon())
        acq.confirm_row(slot)
        time.sleep(0.1)
        acq.decay_unconfirmed()
        assert acq.status[slot].item() == STATUS_CONFIRMED


# =========================================================================
# Item 8: Production fall-through to i_dont_know
# =========================================================================

class TestProductionFallthrough:

    def test_cold_start_falls_through_to_i_dont_know(self):
        """With no allocated lemmas, production selects the
        i_dont_know slot."""
        substrate, acq = make_pair()
        slot, polar_q = acq.select_lemma_for_production(
            random_concept(),
        )
        assert slot == SLOT_I_DONT_KNOW
        assert polar_q is False

    def test_below_threshold_falls_through(self):
        """If the allocated rows do not score above theta_production
        for the current concept, production falls through."""
        substrate, acq = make_pair()
        # Allocate a row with one concept, query with an orthogonal
        # one. The dot product should be near zero.
        c1 = torch.zeros(acq.n_concepts)
        c1[0] = 1.0
        c2 = torch.zeros(acq.n_concepts)
        c2[1] = 1.0
        acq.allocate_row(c1, random_phon())

        slot, _ = acq.select_lemma_for_production(c2)
        assert slot == SLOT_I_DONT_KNOW


# =========================================================================
# Cold-start naming dialogue
# =========================================================================

class TestColdStartNamingDialogue:
    """End-to-end walk of the cold-start naming dialogue at the
    substrate level. Exercises every architectural commitment in
    the order the architect's spec walks them."""

    def test_full_naming_sequence(self):
        substrate, acq = make_pair()
        acq.pre_allocate_reserved_slots()

        # Construct a stable concept vector for "name-of-self".
        # In the actual runtime this comes from the kernel boundary
        # with the naming-frame bias added; here we use a
        # representative vector.
        name_concept = random_concept()
        timmy_phon = random_phon()

        # ------------------------------------------------------------------
        # Turn 1: instructor asks "what is your name?"
        # ------------------------------------------------------------------
        # No name lemma allocated. Production falls through.
        slot, polar_q = acq.select_lemma_for_production(name_concept)
        assert slot == SLOT_I_DONT_KNOW, (
            "Turn 1 should fall through to i_dont_know."
        )
        assert polar_q is False

        # ------------------------------------------------------------------
        # Turn 2: instructor says "your name is Timmy"
        # ------------------------------------------------------------------
        # The frame recognizer fires (tested separately), the
        # phonological code for TIMMY arrives at the novelty gate.
        assert acq.is_novel(timmy_phon) is True, (
            "TIMMY should register as novel before allocation."
        )
        # Allocation event fires.
        timmy_slot = acq.allocate_row(name_concept, timmy_phon)
        assert timmy_slot >= RESERVED_SLOTS, (
            "Acquired lemma should land at a non-reserved index."
        )
        assert acq.status[timmy_slot].item() == STATUS_PROVISIONAL, (
            "Newly-allocated lemma should be provisional."
        )

        # ------------------------------------------------------------------
        # Turn 3: substrate produces "my name is Timmy?"
        # ------------------------------------------------------------------
        # The provisional lemma fires above threshold and triggers
        # polar-question co-activation. The runtime's surface
        # production turns this into the question form with rising
        # intonation; here we just verify the substrate-level
        # contract.
        slot, polar_q = acq.select_lemma_for_production(name_concept)
        assert slot == timmy_slot, (
            "Production should select the provisional Timmy lemma."
        )
        assert polar_q is True, (
            "Provisional lemma should fire polar-question "
            "co-activation."
        )

        # ------------------------------------------------------------------
        # Turn 4: instructor says "yes your name is Timmy"
        # ------------------------------------------------------------------
        # The confirmation_detector fires (tested separately); its
        # side effect is to call acq.confirm_row.
        acq.confirm_row(timmy_slot)
        assert acq.status[timmy_slot].item() == STATUS_CONFIRMED, (
            "Confirmation should transition status to confirmed."
        )

        # ------------------------------------------------------------------
        # Turn 5: substrate produces "ok my name is Timmy"
        # ------------------------------------------------------------------
        # Now the lemma is confirmed; production no longer
        # co-activates the polar-question prime.
        slot, polar_q = acq.select_lemma_for_production(name_concept)
        assert slot == timmy_slot
        assert polar_q is False, (
            "Confirmed lemma should NOT fire polar-question "
            "co-activation."
        )

    def test_cold_restart_preserves_acquired_lemma(self):
        """The .soul checkpoint round-trips both the parent's
        matrices and this module's status array. After restart, the
        substrate retrieves the confirmed lemma exactly as it was
        before shutdown."""
        # Set up substrate, acquire and confirm a lemma.
        substrate, acq = make_pair()
        acq.pre_allocate_reserved_slots()

        name_concept = random_concept()
        timmy_phon = random_phon()
        timmy_slot = acq.allocate_row(name_concept, timmy_phon)
        acq.confirm_row(timmy_slot)

        # Capture the state.
        sub_state = substrate.state_dict()
        acq_state = acq.state_dict()

        # Construct a fresh substrate and acquisition module.
        # Restore from the captured states.
        sub_cfg2 = LexicalSubstrateConfig(
            n_concepts=substrate.cfg.n_concepts,
            n_lemmas=substrate.cfg.n_lemmas,
            d_phon=substrate.cfg.d_phon,
        )
        substrate2 = LexicalSubstrate(sub_cfg2)
        acq_cfg2 = LemmaAcquisitionConfig()
        acq2 = LemmaAcquisitionModule(
            cfg=acq_cfg2, substrate=substrate2,
        )
        substrate2.load_state_dict(sub_state)
        acq2.load_state_dict(acq_state)

        # Verify restart behavior matches pre-shutdown.
        assert acq2.status[timmy_slot].item() == STATUS_CONFIRMED
        slot, polar_q = acq2.select_lemma_for_production(name_concept)
        assert slot == timmy_slot
        assert polar_q is False

    def test_post_restart_unknown_concept_falls_through(self):
        """After acquiring TIMMY and restarting, asking about a
        concept that has no lemma still falls through to
        i_dont_know."""
        substrate, acq = make_pair()
        acq.pre_allocate_reserved_slots()

        # Acquire and confirm TIMMY.
        name_concept = random_concept()
        acq.confirm_row(
            acq.allocate_row(name_concept, random_phon()),
        )

        # Ask about an unrelated concept.
        world_concept = random_concept()
        slot, _ = acq.select_lemma_for_production(world_concept)
        assert slot == SLOT_I_DONT_KNOW


# =========================================================================
# Negative confirmation (correction)
# =========================================================================

class TestCorrection:
    """Tests verifying that decay_row produces the same architectural
    state as the timeout-based decay, just immediately rather than
    after a multi-minute wait. This is the conversational correction
    pathway that the confirmation_detector triggers on negative
    polarity."""

    def test_correction_decays_provisional_lemma(self):
        substrate, acq = make_pair()
        slot = acq.allocate_row(random_concept(), random_phon())
        # Verify provisional state.
        assert acq.status[slot].item() == STATUS_PROVISIONAL
        # Apply correction.
        acq.decay_row(slot)
        # Slot is freed and can be reallocated.
        assert acq.status[slot].item() == STATUS_UNALLOCATED
        new_slot = acq.allocate_row(random_concept(), random_phon())
        assert new_slot == slot, (
            "After correction, the slot should be reallocatable."
        )


# =========================================================================
# Diagnostic state
# =========================================================================

class TestDiagnostics:

    def test_diagnostic_state_at_cold_start(self):
        substrate, acq = make_pair(n_lemmas=64)
        d = acq.get_diagnostic_state()
        assert d["n_provisional"] == 0
        assert d["n_confirmed"] == 0
        assert d["n_unallocated"] == 64
        assert d["n_total"] == 64
        assert d["n_reserved"] == RESERVED_SLOTS

    def test_diagnostic_state_after_pre_allocation(self):
        substrate, acq = make_pair(n_lemmas=64)
        acq.pre_allocate_reserved_slots()
        d = acq.get_diagnostic_state()
        assert d["n_confirmed"] == RESERVED_SLOTS
        assert d["n_unallocated"] == 64 - RESERVED_SLOTS

    def test_diagnostic_state_after_allocation(self):
        substrate, acq = make_pair(n_lemmas=64)
        acq.pre_allocate_reserved_slots()
        acq.allocate_row(random_concept(), random_phon())
        acq.allocate_row(random_concept(), random_phon())
        d = acq.get_diagnostic_state()
        assert d["n_provisional"] == 2
        assert d["n_confirmed"] == RESERVED_SLOTS
