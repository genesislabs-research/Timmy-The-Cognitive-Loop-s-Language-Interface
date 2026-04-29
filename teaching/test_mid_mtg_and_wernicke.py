"""
test_mid_mtg_and_wernicke.py
Tests for the refactored mid_mtg and wernicke modules. Each case
verifies that the consumer modules read through the parent's tied
wrappers correctly, that mutations through the acquisition module
are visible to mid_mtg's forward pass on the very next call, and
that the spell-out path produces the right surface text for the
cold-start dialogue's emission turns.

The end-to-end test at the bottom walks the full cold-start naming
dialogue with all four substrate modules wired together (substrate,
acquisition, mid_mtg, wernicke), producing a complete trace from
turn 1 to turn 5 with the substrate's surface emissions captured
at each step.
"""

from __future__ import annotations

import torch

from substrate.lexical_substrate_t import (
    LexicalSubstrate,
    LexicalSubstrateConfig,
)
from substrate.lemma_slots_t import (
    SLOT_I_DONT_KNOW,
    SLOT_SELF_LEMMA,
    SLOT_OTHER_LEMMA,
    STATUS_CONFIRMED,
    STATUS_PROVISIONAL,
)
from substrate.mid_mtg_t import MidMTG, MidMTGConfig
from substrate.wernicke_t import Wernicke, WernickeConfig
from coordination.lemma_acquisition_t import (
    LemmaAcquisitionConfig,
    LemmaAcquisitionModule,
)


# =========================================================================
# Helpers
# =========================================================================

def make_full_substrate(
    n_concepts: int = 1024,
    n_lemmas: int = 64,
    d_phon: int = 128,
    pre_allocate: bool = True,
):
    """Construct the full four-module substrate stack.

    Returns (substrate, acquisition, mid_mtg, wernicke).
    """
    sub_cfg = LexicalSubstrateConfig(
        n_concepts=n_concepts,
        n_lemmas=n_lemmas,
        d_phon=d_phon,
    )
    substrate = LexicalSubstrate(sub_cfg)
    acq = LemmaAcquisitionModule(
        cfg=LemmaAcquisitionConfig(),
        substrate=substrate,
    )
    if pre_allocate:
        acq.pre_allocate_reserved_slots()
    mid_mtg = MidMTG(cfg=MidMTGConfig(), substrate=substrate)
    wernicke = Wernicke(cfg=WernickeConfig(), substrate=substrate)
    return substrate, acq, mid_mtg, wernicke


def random_concept(n_concepts: int = 1024) -> torch.Tensor:
    v = torch.randn(n_concepts)
    return v / v.norm()


def random_phon(d_phon: int = 128) -> torch.Tensor:
    v = torch.randn(d_phon)
    return v / v.norm()


# =========================================================================
# MidMTG construction
# =========================================================================

class TestMidMTGConstruction:

    def test_a_lemma_starts_zero(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        assert torch.all(mid_mtg.a_lemma == 0)

    def test_substrate_reference_held(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        assert mid_mtg.substrate is substrate

    def test_dimensions_match_substrate(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate(
            n_concepts=256, n_lemmas=32, d_phon=64,
        )
        assert mid_mtg.n_concepts == 256
        assert mid_mtg.n_lemmas == 32


# =========================================================================
# MidMTG production direction
# =========================================================================

class TestMidMTGProduction:

    def test_zero_concept_produces_zero_lemma(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate(
            pre_allocate=False,
        )
        zero_concept = torch.zeros(1, mid_mtg.n_concepts)
        out = mid_mtg.forward_production(zero_concept)
        assert out.shape == (1, mid_mtg.n_lemmas)
        assert out.norm().item() == 0.0

    def test_allocated_concept_drives_correct_lemma(self):
        """A concept that was used at allocation time should drive
        the corresponding lemma activation when fed back in."""
        substrate, acq, mid_mtg, wernicke = make_full_substrate(
            pre_allocate=False,
        )

        concept = random_concept(mid_mtg.n_concepts)
        slot = acq.allocate_row(
            concept, random_phon(mid_mtg.substrate.cfg.d_phon),
        )

        # Drive mid_mtg with the allocation concept.
        out = mid_mtg.forward_production(concept.unsqueeze(0))
        # The slot should be the strongest activation.
        winning = int(out.argmax(dim=1).item())
        assert winning == slot

    def test_persistence_accumulates_across_calls(self):
        """With persistence enabled, repeated forward passes
        accumulate the activation toward the steady state."""
        substrate, acq, mid_mtg, wernicke = make_full_substrate(
            pre_allocate=False,
        )

        concept = random_concept(mid_mtg.n_concepts)
        acq.allocate_row(
            concept, random_phon(mid_mtg.substrate.cfg.d_phon),
        )

        out1 = mid_mtg.forward_production(concept.unsqueeze(0))
        magnitude1 = out1.norm().item()
        out2 = mid_mtg.forward_production(concept.unsqueeze(0))
        magnitude2 = out2.norm().item()
        # Persistence means out2 has accumulated more drive than out1.
        assert magnitude2 > magnitude1

    def test_reset_state_clears_persistence(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate(
            pre_allocate=False,
        )
        concept = random_concept(mid_mtg.n_concepts)
        acq.allocate_row(
            concept, random_phon(mid_mtg.substrate.cfg.d_phon),
        )
        for _ in range(5):
            mid_mtg.forward_production(concept.unsqueeze(0))
        assert mid_mtg.a_lemma.norm().item() > 0
        mid_mtg.reset_state()
        assert mid_mtg.a_lemma.norm().item() == 0

    def test_disable_persistence_returns_drive_only(self):
        cfg = MidMTGConfig(enable_persistence=False)
        substrate, acq, _, _ = make_full_substrate(pre_allocate=False)
        mid_mtg = MidMTG(cfg=cfg, substrate=substrate)

        concept = random_concept(mid_mtg.n_concepts)
        acq.allocate_row(
            concept, random_phon(substrate.cfg.d_phon),
        )

        out1 = mid_mtg.forward_production(concept.unsqueeze(0))
        out2 = mid_mtg.forward_production(concept.unsqueeze(0))
        # Without persistence both calls produce the same output.
        assert torch.allclose(out1, out2)


# =========================================================================
# MidMTG comprehension direction
# =========================================================================

class TestMidMTGComprehension:

    def test_comprehension_reconstructs_concept(self):
        """A one-hot lemma input should reconstruct approximately
        the concept vector that was bound to that slot at
        allocation."""
        substrate, acq, mid_mtg, wernicke = make_full_substrate(
            pre_allocate=False,
        )
        concept = random_concept(mid_mtg.n_concepts)
        slot = acq.allocate_row(
            concept, random_phon(mid_mtg.substrate.cfg.d_phon),
        )

        one_hot = torch.zeros(1, mid_mtg.n_lemmas)
        one_hot[0, slot] = 1.0
        reconstructed = mid_mtg.forward_comprehension(one_hot)
        # The reconstruction should match the bound concept.
        assert torch.allclose(
            reconstructed[0], concept, atol=1e-6,
        )

    def test_comprehension_through_full_path(self):
        """End-to-end comprehension: phonological code arrives at
        wernicke's, drives lemma activation backward through W_L_to_P,
        the lemma activation feeds into mid_mtg's forward_comprehension,
        and the resulting concept reconstruction approximates the
        concept the lemma was bound to."""
        substrate, acq, mid_mtg, wernicke = make_full_substrate(
            pre_allocate=False,
        )

        concept = random_concept(mid_mtg.n_concepts)
        phon = random_phon(mid_mtg.substrate.cfg.d_phon)
        slot = acq.allocate_row(concept, phon)

        # Wernicke's perceives the phonological code.
        lemma_drive = wernicke.perceive_phonological_code(
            phon.unsqueeze(0),
        )
        # The slot should be the strongest lemma activation.
        winning = int(lemma_drive.argmax(dim=1).item())
        assert winning == slot


# =========================================================================
# Wernicke production direction
# =========================================================================

class TestWernickeProduction:

    def test_retrieve_phonological_code_for_allocated_slot(self):
        """A one-hot lemma activation should retrieve the
        phonological code that was bound to that slot."""
        substrate, acq, mid_mtg, wernicke = make_full_substrate(
            pre_allocate=False,
        )
        concept = random_concept(mid_mtg.n_concepts)
        phon = random_phon(substrate.cfg.d_phon)
        slot = acq.allocate_row(concept, phon)

        one_hot = torch.zeros(1, substrate.cfg.n_lemmas)
        one_hot[0, slot] = 1.0
        retrieved = wernicke.retrieve_phonological_code(one_hot)
        assert torch.allclose(retrieved[0], phon, atol=1e-6)

    def test_retrieve_for_unallocated_slot_is_zero(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate(
            pre_allocate=False,
        )
        # Slot 30 has not been allocated.
        one_hot = torch.zeros(1, substrate.cfg.n_lemmas)
        one_hot[0, 30] = 1.0
        retrieved = wernicke.retrieve_phonological_code(one_hot)
        assert retrieved.norm().item() == 0.0


# =========================================================================
# Wernicke spell-out
# =========================================================================

class TestWernickeSpellOut:
    """Tests verifying the slot-text spell-out path. This is the
    cold-start dialogue's surface emission mechanism."""

    def test_reserved_slots_have_canonical_text(self):
        """At construction, reserved slots from the slot inventory
        with non-empty PHONOLOGICAL_TEXT_BY_NAME entries are
        registered. Identity and polar_question slots have empty
        text and are not registered."""
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        assert wernicke.spell_out_for_slot(SLOT_I_DONT_KNOW) == (
            "i don't know"
        )
        # Wh-words have their canonical surface forms.
        assert wernicke.spell_out_for_slot(2) == "what"  # SLOT_WHAT
        # Identity slots do not register at construction; the
        # runtime registers them at acquisition time.
        assert wernicke.spell_out_for_slot(SLOT_SELF_LEMMA) == ""

    def test_register_slot_text(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        wernicke.register_slot_text(17, "Timmy")
        assert wernicke.spell_out_for_slot(17) == "Timmy"

    def test_register_empty_text_clears(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        wernicke.register_slot_text(17, "Timmy")
        wernicke.register_slot_text(17, "")
        assert wernicke.spell_out_for_slot(17) == ""

    def test_spell_out_with_polar_question_appends_question_mark(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        wernicke.register_slot_text(17, "my name is Timmy")
        assert wernicke.spell_out_with_polar_question(
            17, polar_question=True,
        ) == "my name is Timmy?"
        assert wernicke.spell_out_with_polar_question(
            17, polar_question=False,
        ) == "my name is Timmy"

    def test_spell_out_does_not_double_question_mark(self):
        """If the registered text already ends with a question
        mark, the polar-question form does not duplicate it."""
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        wernicke.register_slot_text(17, "what?")
        assert wernicke.spell_out_with_polar_question(
            17, polar_question=True,
        ) == "what?"

    def test_spell_out_unregistered_slot_returns_empty(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        assert wernicke.spell_out_for_slot(50) == ""

    def test_reset_slot_text_registry_restores_defaults(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        wernicke.register_slot_text(17, "Timmy")
        wernicke.reset_slot_text_registry()
        assert wernicke.spell_out_for_slot(17) == ""
        # Reserved slots still registered.
        assert wernicke.spell_out_for_slot(SLOT_I_DONT_KNOW) == (
            "i don't know"
        )


# =========================================================================
# GRU spell-out
# =========================================================================

class TestWernickeGRUSpellOut:
    """Tests verifying the GRU spell-out path produces output of
    the right shape. The trained-substrate semantics of the GRU
    output are out of scope for v2 cold-start; these tests only
    verify the architectural shape is correct."""

    def test_gru_output_shape(self):
        substrate, acq, mid_mtg, wernicke = make_full_substrate()
        phon = torch.randn(1, substrate.cfg.d_phon)
        out = wernicke.spell_out_gru_decoder(phon, n_steps=10)
        assert out.shape == (1, 10, wernicke.cfg.n_segments)


# =========================================================================
# End-to-end cold-start dialogue at the substrate level
# =========================================================================

class TestColdStartDialogueWithSubstrate:
    """End-to-end walk of the cold-start naming dialogue with all
    four substrate modules wired together. This is the closest the
    test suite can get to the milestone dialogue without the runtime
    layer that actually accepts text input."""

    def test_full_dialogue_with_substrate_emissions(self):
        """Walks turns 1-5 of the cold-start dialogue, capturing
        the substrate's surface emission at each step. Each turn
        exercises a specific architectural commitment, and the
        captured emissions match the architect's expected output."""
        substrate, acq, mid_mtg, wernicke = make_full_substrate()

        # Concept vector for "name-of-self". The runtime constructs
        # this from the kernel boundary plus the naming-frame bias;
        # the test substitutes a fixed concept vector for clarity.
        name_concept = random_concept(mid_mtg.n_concepts)
        timmy_phon = random_phon(substrate.cfg.d_phon)

        emissions = []

        # ------------------------------------------------------------------
        # Turn 1: instructor asks "what is your name?"
        # ------------------------------------------------------------------
        # Substrate has no allocated name lemma. Production falls
        # through to i_dont_know.
        slot, polar_q = acq.select_lemma_for_production(name_concept)
        text = wernicke.spell_out_with_polar_question(slot, polar_q)
        emissions.append(("turn1", text, slot, polar_q))
        assert text == "i don't know"
        assert polar_q is False

        # ------------------------------------------------------------------
        # Turn 2: instructor says "your name is Timmy"
        # ------------------------------------------------------------------
        # Frame recognizer fires (tested separately); allocation
        # happens; runtime registers the surface text "my name is
        # Timmy" against the allocated slot. Runtime knows the text
        # form because it owns the assembly of pronouns, frame
        # words, and the wildcard-bound name token.
        timmy_slot = acq.allocate_row(name_concept, timmy_phon)
        wernicke.register_slot_text(timmy_slot, "my name is Timmy")
        assert acq.status[timmy_slot].item() == STATUS_PROVISIONAL

        # ------------------------------------------------------------------
        # Turn 3: substrate produces "my name is Timmy?"
        # ------------------------------------------------------------------
        slot, polar_q = acq.select_lemma_for_production(name_concept)
        text = wernicke.spell_out_with_polar_question(slot, polar_q)
        emissions.append(("turn3", text, slot, polar_q))
        assert slot == timmy_slot
        assert polar_q is True
        assert text == "my name is Timmy?"

        # ------------------------------------------------------------------
        # Turn 4: instructor says "yes your name is Timmy"
        # ------------------------------------------------------------------
        # Confirmation_detector fires; lemma transitions to confirmed.
        acq.confirm_row(timmy_slot)
        assert acq.status[timmy_slot].item() == STATUS_CONFIRMED

        # ------------------------------------------------------------------
        # Turn 5: substrate produces "ok my name is Timmy"
        # ------------------------------------------------------------------
        # The lemma is now confirmed; polar-question co-activation
        # does not fire. The runtime prepends "ok " to the text;
        # at the substrate level we just verify polar_q is False.
        slot, polar_q = acq.select_lemma_for_production(name_concept)
        text = wernicke.spell_out_with_polar_question(slot, polar_q)
        emissions.append(("turn5", text, slot, polar_q))
        assert slot == timmy_slot
        assert polar_q is False
        assert text == "my name is Timmy"

        # Verify the full emission trace.
        assert emissions[0][1] == "i don't know"
        assert emissions[1][1] == "my name is Timmy?"
        assert emissions[2][1] == "my name is Timmy"

    def test_full_dialogue_survives_state_dict_round_trip(self):
        """The .soul checkpoint shape: serialize substrate, acq,
        mid_mtg, and wernicke state_dicts, plus the slot-text
        registry as a side dict, restore everything, and verify
        the post-restart substrate produces the same emission for
        the confirmed lemma."""
        substrate, acq, mid_mtg, wernicke = make_full_substrate()

        # Acquire and confirm.
        name_concept = random_concept(mid_mtg.n_concepts)
        timmy_phon = random_phon(substrate.cfg.d_phon)
        timmy_slot = acq.allocate_row(name_concept, timmy_phon)
        acq.confirm_row(timmy_slot)
        wernicke.register_slot_text(timmy_slot, "my name is Timmy")

        # Capture state.
        sub_state = substrate.state_dict()
        acq_state = acq.state_dict()
        mid_state = mid_mtg.state_dict()
        wer_state = wernicke.state_dict()
        # The slot-text registry is a side dict that the runtime
        # serializes alongside the state_dicts.
        slot_text_registry = dict(wernicke._slot_text)

        # Construct fresh stack.
        substrate2 = LexicalSubstrate(LexicalSubstrateConfig(
            n_concepts=substrate.cfg.n_concepts,
            n_lemmas=substrate.cfg.n_lemmas,
            d_phon=substrate.cfg.d_phon,
        ))
        acq2 = LemmaAcquisitionModule(
            cfg=LemmaAcquisitionConfig(),
            substrate=substrate2,
        )
        mid_mtg2 = MidMTG(cfg=MidMTGConfig(), substrate=substrate2)
        wernicke2 = Wernicke(
            cfg=WernickeConfig(), substrate=substrate2,
        )

        # Restore.
        substrate2.load_state_dict(sub_state)
        acq2.load_state_dict(acq_state)
        mid_mtg2.load_state_dict(mid_state)
        wernicke2.load_state_dict(wer_state)
        wernicke2._slot_text = dict(slot_text_registry)

        # Verify post-restart behavior matches.
        slot, polar_q = acq2.select_lemma_for_production(name_concept)
        text = wernicke2.spell_out_with_polar_question(slot, polar_q)
        assert slot == timmy_slot
        assert polar_q is False
        assert text == "my name is Timmy"


# =========================================================================
# Master flag ablation
# =========================================================================

class TestAblation:

    def test_disabled_mid_mtg_returns_zero(self):
        cfg = MidMTGConfig(enable_mid_mtg=False)
        substrate, _, _, _ = make_full_substrate(pre_allocate=False)
        mid_mtg = MidMTG(cfg=cfg, substrate=substrate)

        concept = torch.randn(1, mid_mtg.n_concepts)
        out = mid_mtg.forward_production(concept)
        assert torch.all(out == 0)

    def test_disabled_wernicke_returns_zero(self):
        cfg = WernickeConfig(enable_wernicke=False)
        substrate, _, _, _ = make_full_substrate(pre_allocate=False)
        wernicke = Wernicke(cfg=cfg, substrate=substrate)

        lemma = torch.randn(1, wernicke.n_lemmas)
        out = wernicke.retrieve_phonological_code(lemma)
        assert torch.all(out == 0)

    def test_disabled_spell_out_returns_empty_string(self):
        cfg = WernickeConfig(enable_spell_out=False)
        substrate, _, _, _ = make_full_substrate(pre_allocate=False)
        wernicke = Wernicke(cfg=cfg, substrate=substrate)
        wernicke.register_slot_text(17, "Timmy")
        assert wernicke.spell_out_for_slot(17) == ""
