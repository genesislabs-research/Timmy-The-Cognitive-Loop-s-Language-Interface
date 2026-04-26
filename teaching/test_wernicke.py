"""
test_wernicke.py
Tests for the Wernicke's lexical phonological code store.

These tests verify:
    1. Production direction retrieves a phonological code from a selected
       lemma through W_L_to_P.
    2. Comprehension direction projects a perceived phonological code
       backward into perceptual lemma activation.
    3. Spell-out is incremental: emit_next_segment returns one segment
       per call, with hidden state carried across calls.
    4. spell_out_word terminates at SEGMENT_WORD_END or at the safety cap.
    5. Phonological confidence is meaningfully higher for a stable
       (trained) lemma-code mapping than for a wobbly random mapping.
    6. Persistence: perceptual activation decays between calls when
       enabled, and does not when disabled.
    7. Round-trip serialization preserves state.
    8. Ablation flags force neutral behavior.
"""

from __future__ import annotations

import pytest
import torch

from regions.wernicke_t import (
    Wernicke,
    WernickeConfig,
    SEGMENT_SILENCE,
    SEGMENT_SYLLABLE_BOUNDARY,
    SEGMENT_WORD_END,
    N_RESERVED_SEGMENTS,
)


# =========================================================================
# Helpers
# =========================================================================

def _plant_phonological_mapping(
    wer: Wernicke,
    lemma_idx: int,
    code_signature: torch.Tensor,
) -> None:
    """Manually set the W_L_to_P matrix so that lemma lemma_idx
    retrieves the given phonological code signature. Simulates Phase 3
    acquisition on a single lemma-code pairing.

    The W_L_to_P matrix has shape (d_phon, n_lemmas). Setting column
    lemma_idx to the desired code signature makes a one-hot at
    lemma_idx produce that signature in the production direction.
    """
    with torch.no_grad():
        wer.w_l_to_p.W[:, lemma_idx] = 0.0
        wer.w_l_to_p.W[:, lemma_idx] = code_signature


def _plant_decoder_to_emit(
    wer: Wernicke,
    target_segment: int,
) -> None:
    """Overwrite the spell-out decoder so the segment_logits always
    favor target_segment regardless of input. Used to verify that
    emit_next_segment respects the configured output and that the
    runtime correctly consumes argmax indices.

    Sets the hidden_to_segment projection bias so target_segment
    dominates and zeroes the other paths so input does not perturb the
    decision. This is a test convenience; trained decoders produce real
    distributions over the segment vocabulary.
    """
    with torch.no_grad():
        wer.spell_out_hidden_to_segment.weight.zero_()
        bias = torch.full(
            (wer.cfg.n_segments,), -10.0,
            device=wer.spell_out_hidden_to_segment.bias.device,
        )
        bias[target_segment] = 10.0
        wer.spell_out_hidden_to_segment.bias.copy_(bias)


# =========================================================================
# Reserved segment indices
# =========================================================================

class TestReservedSegments:

    def test_reserved_segment_indices_distinct(self):
        assert SEGMENT_SILENCE != SEGMENT_SYLLABLE_BOUNDARY
        assert SEGMENT_SILENCE != SEGMENT_WORD_END
        assert SEGMENT_SYLLABLE_BOUNDARY != SEGMENT_WORD_END
        assert N_RESERVED_SEGMENTS == 3

    def test_reserved_segments_at_low_indices(self):
        for idx in (SEGMENT_SILENCE, SEGMENT_SYLLABLE_BOUNDARY, SEGMENT_WORD_END):
            assert 0 <= idx < N_RESERVED_SEGMENTS


# =========================================================================
# Production direction
# =========================================================================

class TestProductionDirection:

    def test_retrieve_phonological_code_returns_planted_signature(self):
        """A one-hot lemma activation should retrieve the planted code."""
        cfg = WernickeConfig(n_lemmas=128, d_phon=128)
        wer = Wernicke(cfg)

        lemma_idx = 17
        code = torch.randn(cfg.d_phon)
        _plant_phonological_mapping(wer, lemma_idx, code)

        l_star = torch.zeros(1, cfg.n_lemmas)
        l_star[0, lemma_idx] = 1.0

        retrieved = wer.retrieve_phonological_code(l_star)
        assert torch.allclose(retrieved.squeeze(0), code, atol=1e-5), (
            "Production retrieval should match the planted code signature."
        )

    def test_retrieve_for_unmatched_lemma_yields_random_signature(self):
        """An untrained lemma slot retrieves whatever the random
        initialization produced. The retrieved code should not match
        the planted signature for a different lemma.
        """
        cfg = WernickeConfig(n_lemmas=128, d_phon=128)
        wer = Wernicke(cfg)

        planted_lemma = 17
        planted_code = torch.randn(cfg.d_phon)
        _plant_phonological_mapping(wer, planted_lemma, planted_code)

        # Different lemma, no planted code.
        l_other = torch.zeros(1, cfg.n_lemmas)
        l_other[0, 42] = 1.0

        other_code = wer.retrieve_phonological_code(l_other)
        # Cosine similarity between the planted code and the other
        # lemma's retrieval should be near zero.
        cos = torch.nn.functional.cosine_similarity(
            other_code, planted_code.unsqueeze(0), dim=1,
        )
        assert cos.abs().item() < 0.5, (
            f"Different lemmas should not retrieve the same code. "
            f"Cosine similarity {cos.item():.3f} was too high."
        )


# =========================================================================
# Spell-out
# =========================================================================

class TestSpellOut:

    def test_emit_next_segment_advances_hidden_state(self):
        """Each emit_next_segment call should update the decoder hidden
        state. Two consecutive calls with the same code input should
        produce different hidden states.
        """
        cfg = WernickeConfig(n_lemmas=128, d_phon=128)
        wer = Wernicke(cfg)
        wer.reset_spell_out_state(batch_size=1)

        phon_code = torch.randn(1, cfg.d_phon)
        h0 = wer._spell_out_hidden.clone()
        wer.emit_next_segment(phon_code)
        h1 = wer._spell_out_hidden.clone()
        wer.emit_next_segment(phon_code)
        h2 = wer._spell_out_hidden.clone()

        assert not torch.allclose(h0, h1)
        assert not torch.allclose(h1, h2)

    def test_emit_next_segment_returns_segment_logits(self):
        cfg = WernickeConfig(n_lemmas=128, d_phon=128, n_segments=64)
        wer = Wernicke(cfg)
        wer.reset_spell_out_state(batch_size=1)

        phon_code = torch.randn(1, cfg.d_phon)
        logits = wer.emit_next_segment(phon_code)
        assert logits.shape == (1, 64)

    def test_spell_out_word_terminates_on_word_end(self):
        """When the decoder is forced to emit SEGMENT_WORD_END, spell_out_word
        should terminate immediately rather than running to max_steps.
        """
        cfg = WernickeConfig(n_lemmas=128, d_phon=128, spell_out_max_steps=20)
        wer = Wernicke(cfg)
        _plant_decoder_to_emit(wer, SEGMENT_WORD_END)

        phon_code = torch.randn(1, cfg.d_phon)
        emitted = wer.spell_out_word(phon_code)
        # First emission is the word-end marker, so length is 1.
        assert emitted.shape[1] == 1
        assert emitted[0, 0].item() == SEGMENT_WORD_END

    def test_spell_out_word_respects_max_steps_safety_cap(self):
        """If the decoder never emits a word-end marker, spell_out_word
        must terminate at the safety cap rather than looping forever.
        """
        cfg = WernickeConfig(n_lemmas=128, d_phon=128, spell_out_max_steps=10)
        wer = Wernicke(cfg)
        # Force decoder to emit silence forever, never word_end.
        _plant_decoder_to_emit(wer, SEGMENT_SILENCE)

        phon_code = torch.randn(1, cfg.d_phon)
        emitted = wer.spell_out_word(phon_code)
        assert emitted.shape[1] == 10  # Hit the cap exactly.
        assert (emitted == SEGMENT_SILENCE).all()


# =========================================================================
# Comprehension direction
# =========================================================================

class TestComprehensionDirection:

    def test_perceive_drives_perceptual_lemma_activation(self):
        """A perceived phonological code should produce a non-zero
        perceptual lemma activation through the tied substrate's
        reverse direction.
        """
        cfg = WernickeConfig(n_lemmas=128, d_phon=128)
        wer = Wernicke(cfg)

        phi_input = torch.randn(1, cfg.d_phon)
        a_percept = wer.perceive_phonological_code(phi_input)

        assert a_percept.shape == (1, 128)
        assert not torch.allclose(a_percept, torch.zeros_like(a_percept))

    def test_perceive_with_planted_mapping_recovers_lemma(self):
        """When a code is planted for a lemma, perceiving that exact
        code should produce maximal activation for that lemma in the
        comprehension direction.
        """
        cfg = WernickeConfig(n_lemmas=128, d_phon=128, enable_persistence=False)
        wer = Wernicke(cfg)

        target_lemma = 17
        code = torch.randn(cfg.d_phon)
        _plant_phonological_mapping(wer, target_lemma, code)

        # Reset state so accumulated activation does not interfere.
        wer.reset_state()

        a_percept = wer.perceive_phonological_code(code.unsqueeze(0))
        peak_lemma = a_percept.argmax(dim=1).item()
        assert peak_lemma == target_lemma, (
            f"Perceiving the planted code should peak at lemma "
            f"{target_lemma}, got {peak_lemma}."
        )


# =========================================================================
# Persistence
# =========================================================================

class TestPersistence:

    def test_persistent_activation_decays_over_calls(self):
        """When persistence is enabled, perceptual activation from an
        earlier input should decay across subsequent calls with zero
        input.
        """
        cfg = WernickeConfig(
            n_lemmas=128, d_phon=128,
            enable_persistence=True, tau_decay_steps=10,
        )
        wer = Wernicke(cfg)

        # Initial input.
        phi = torch.randn(1, cfg.d_phon)
        a0 = wer.perceive_phonological_code(phi).clone()

        # Subsequent zero inputs.
        zero_input = torch.zeros(1, cfg.d_phon)
        for _ in range(20):
            wer.perceive_phonological_code(zero_input)

        a_after = wer.a_l_percept.clone()
        assert a_after.abs().max() < a0.abs().max(), (
            "Persistent activation should decay over zero-input calls."
        )

    def test_no_persistence_zeros_each_call(self):
        cfg = WernickeConfig(
            n_lemmas=128, d_phon=128, enable_persistence=False,
        )
        wer = Wernicke(cfg)

        phi = torch.randn(1, cfg.d_phon)
        a0 = wer.perceive_phonological_code(phi).clone()
        a1 = wer.perceive_phonological_code(torch.zeros(1, cfg.d_phon)).clone()

        # With persistence disabled, the second call's activation comes
        # only from the new (zero) input, so it should be zero (W_L_to_P
        # of zero is zero).
        assert torch.allclose(a1, torch.zeros_like(a1), atol=1e-6)


# =========================================================================
# Phonological confidence
# =========================================================================

class TestPhonologicalConfidence:

    def test_confidence_in_valid_range(self):
        cfg = WernickeConfig(n_lemmas=128, d_phon=128)
        wer = Wernicke(cfg)

        l_star = torch.zeros(1, cfg.n_lemmas)
        l_star[0, 17] = 1.0

        conf = wer.get_phonological_confidence(l_star).item()
        assert 0.0 <= conf <= 1.0

    def test_strong_signature_more_stable_than_weak(self):
        """A lemma column with a strong, clean code signature should
        produce higher phonological confidence than one with a weak,
        noisy signature. Strong signatures dominate the perturbation
        noise; weak signatures are swamped by it.
        """
        cfg = WernickeConfig(
            n_lemmas=128, d_phon=128,
            confidence_perturbation_scale=0.1,
        )
        wer = Wernicke(cfg)

        # Lemma 17: strong, clean signature (large magnitude).
        with torch.no_grad():
            wer.w_l_to_p.W.zero_()
            wer.w_l_to_p.W[:, 17] = torch.randn(cfg.d_phon) * 5.0
            # Lemma 18: weak signature (small magnitude).
            wer.w_l_to_p.W[:, 18] = torch.randn(cfg.d_phon) * 0.05

        l_strong = torch.zeros(1, cfg.n_lemmas); l_strong[0, 17] = 1.0
        l_weak = torch.zeros(1, cfg.n_lemmas); l_weak[0, 18] = 1.0

        conf_strong = wer.get_phonological_confidence(l_strong).item()
        conf_weak = wer.get_phonological_confidence(l_weak).item()

        assert conf_strong > conf_weak + 0.1, (
            f"Strong signature should produce higher phonological "
            f"confidence ({conf_strong:.3f}) than weak signature "
            f"({conf_weak:.3f})."
        )


# =========================================================================
# Serialization
# =========================================================================

class TestSerialization:

    def test_round_trip_preserves_state(self):
        cfg = WernickeConfig(n_lemmas=128, d_phon=128)
        original = Wernicke(cfg)

        # Drive some state.
        phi = torch.randn(1, cfg.d_phon)
        original.perceive_phonological_code(phi)
        original.reset_spell_out_state(batch_size=1)
        original.emit_next_segment(phi)

        state = original.serialize()

        restored = Wernicke(cfg)
        restored.restore(state)

        # Weights match.
        assert torch.allclose(
            original.w_l_to_p.W, restored.w_l_to_p.W, atol=1e-9,
        )
        # Perceptual activation matches.
        assert torch.allclose(
            original.a_l_percept, restored.a_l_percept, atol=1e-9,
        )
        # Spell-out hidden state matches.
        assert torch.allclose(
            original._spell_out_hidden,
            restored._spell_out_hidden,
            atol=1e-9,
        )
        # Decoder parameters match.
        assert torch.allclose(
            original.spell_out_hidden_to_segment.weight,
            restored.spell_out_hidden_to_segment.weight,
            atol=1e-9,
        )

    def test_round_trip_preserves_retrieval_behavior(self):
        """A restored Wernicke's must produce identical phonological
        codes for the same lemma input as the original.
        """
        cfg = WernickeConfig(n_lemmas=128, d_phon=128)
        original = Wernicke(cfg)
        _plant_phonological_mapping(
            original, 17, torch.randn(cfg.d_phon),
        )

        state = original.serialize()
        restored = Wernicke(cfg)
        restored.restore(state)

        l_star = torch.zeros(1, cfg.n_lemmas)
        l_star[0, 17] = 1.0

        assert torch.allclose(
            original.retrieve_phonological_code(l_star),
            restored.retrieve_phonological_code(l_star),
            atol=1e-9,
        )


# =========================================================================
# Ablation
# =========================================================================

class TestAblation:

    def test_master_ablation_returns_zeros_in_production(self):
        cfg = WernickeConfig(enable_wernicke=False, n_lemmas=128, d_phon=128)
        wer = Wernicke(cfg)
        l_star = torch.zeros(1, cfg.n_lemmas)
        l_star[0, 17] = 1.0
        out = wer.retrieve_phonological_code(l_star)
        assert out.shape == (1, 128)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_master_ablation_returns_zeros_in_comprehension(self):
        cfg = WernickeConfig(enable_wernicke=False, n_lemmas=128, d_phon=128)
        wer = Wernicke(cfg)
        phi = torch.randn(1, cfg.d_phon)
        a = wer.perceive_phonological_code(phi)
        assert a.shape == (1, 128)
        assert torch.allclose(a, torch.zeros_like(a))

    def test_spell_out_ablation_returns_zeros(self):
        """When spell-out is disabled, emit_next_segment should return
        zeros even though the rest of Wernicke's is enabled.
        """
        cfg = WernickeConfig(
            enable_wernicke=True, enable_spell_out=False,
            n_lemmas=128, d_phon=128, n_segments=64,
        )
        wer = Wernicke(cfg)
        phi = torch.randn(1, cfg.d_phon)
        logits = wer.emit_next_segment(phi)
        assert logits.shape == (1, 64)
        assert torch.allclose(logits, torch.zeros_like(logits))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
