"""
test_epistemic_monitor.py
Tests for the EpistemicMonitor.

These tests verify:
    1. Component readers handle missing upstream sources gracefully.
    2. Aggregation produces the right register at each confidence band.
    3. Maturity gating forces confidence to zero below the low threshold,
       passes through above the high threshold, and ramps in between.
    4. Question-formation trigger fires only when confidence is in the
       hedged band AND curiosity is above threshold.
    5. The cold-start dialogue Phase 1 case (everything floor-low,
       maturity at zero) produces the very_humble register.
    6. The cold-start dialogue Phase 7 case (substrate has acquired its
       name) produces a higher register on the matching query.
    7. Configuration serializes round-trip preserving all thresholds and
       weights.
    8. Ablation flag forces the very_humble register.
"""

from __future__ import annotations

import pytest
import torch

from coordination.epistemic_monitor_t import (
    EpistemicMonitor,
    EpistemicMonitorConfig,
    ConfidenceRegister,
    ConfidenceReport,
)
from coordination.neuromodulator_bus_t import (
    NeuromodulatorBus,
    NeuromodulatorBusConfig,
)
from regions.mid_mtg_t import MidMTG, MidMTGConfig, N_RESERVED_LEMMAS
from regions.wernicke_t import Wernicke, WernickeConfig


# =========================================================================
# Helpers
# =========================================================================

class FakeBus:
    """Bus that exposes global_maturity through the same get() interface
    the real NeuromodulatorBus uses.
    """

    def __init__(self, maturity: float) -> None:
        self._maturity = maturity

    def get(self, key: str) -> torch.Tensor:
        if key == "global_maturity":
            return torch.tensor(self._maturity)
        raise KeyError(key)


class FakeKernel:
    """Kernel stub that exposes a last_novelty attribute."""

    def __init__(self, novelty: float) -> None:
        self.last_novelty = torch.tensor(novelty)


class FakeWorldModel:
    """World model stub that exposes a last_ensemble_variance attribute."""

    def __init__(self, variance: float) -> None:
        self.last_ensemble_variance = torch.tensor(variance)


def _fresh_substrates():
    """Construct mid-MTG and Wernicke's for testing."""
    midmtg = MidMTG(MidMTGConfig(n_concepts=64, n_lemmas=128))
    wernicke = Wernicke(WernickeConfig(n_lemmas=128))
    return midmtg, wernicke


# =========================================================================
# Component readers with missing sources
# =========================================================================

class TestComponentReaders:
    """Each component reader must handle a missing upstream source by
    returning a neutral value rather than crashing.
    """

    def test_no_kernel_returns_neutral(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        ca3 = monitor._read_ca3_confidence(None)
        assert ca3 == 0.5

    def test_no_world_model_returns_neutral(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        wm = monitor._read_world_model_confidence(None)
        assert wm == 0.5

    def test_no_neuromod_bus_returns_zero_maturity(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        m = monitor._read_maturity(None)
        assert m == 0.0

    def test_kernel_with_high_novelty_gives_low_ca3_confidence(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        kernel = FakeKernel(novelty=0.9)
        ca3 = monitor._read_ca3_confidence(kernel)
        assert ca3 == pytest.approx(0.1, abs=0.01)

    def test_world_model_with_low_variance_gives_high_confidence(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        wm_stub = FakeWorldModel(variance=0.01)
        wm = monitor._read_world_model_confidence(wm_stub)
        # exp(-0.01) is approximately 0.99.
        assert wm > 0.95


# =========================================================================
# Maturity gating
# =========================================================================

class TestMaturityGating:

    def test_maturity_below_low_threshold_zeros_gain(self):
        cfg = EpistemicMonitorConfig(
            maturity_threshold_low=0.3,
            maturity_threshold_high=0.6,
        )
        monitor = EpistemicMonitor(cfg)
        gain = monitor._compute_maturity_gain(0.1)
        assert gain == 0.0

    def test_maturity_above_high_threshold_full_gain(self):
        cfg = EpistemicMonitorConfig(
            maturity_threshold_low=0.3,
            maturity_threshold_high=0.6,
        )
        monitor = EpistemicMonitor(cfg)
        gain = monitor._compute_maturity_gain(0.8)
        assert gain == 1.0

    def test_maturity_in_ramp_interpolates_linearly(self):
        cfg = EpistemicMonitorConfig(
            maturity_threshold_low=0.3,
            maturity_threshold_high=0.6,
        )
        monitor = EpistemicMonitor(cfg)
        # Midpoint at 0.45 should give gain 0.5.
        gain = monitor._compute_maturity_gain(0.45)
        assert gain == pytest.approx(0.5, abs=0.01)

    def test_maturity_gating_disabled_passes_through(self):
        cfg = EpistemicMonitorConfig(enable_maturity_gating=False)
        monitor = EpistemicMonitor(cfg)
        # Low maturity should still produce gain=1 when gating is off.
        assert monitor._compute_maturity_gain(0.0) == 1.0


# =========================================================================
# Register classification
# =========================================================================

class TestRegisterClassification:

    def test_high_aggregate_classifies_confident(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        assert monitor._classify_register(0.9) == ConfidenceRegister.CONFIDENT

    def test_medium_aggregate_classifies_hedged(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        assert monitor._classify_register(0.5) == ConfidenceRegister.HEDGED

    def test_low_aggregate_classifies_humble(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        assert monitor._classify_register(0.25) == ConfidenceRegister.HUMBLE

    def test_very_low_aggregate_classifies_very_humble(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        assert monitor._classify_register(0.05) == ConfidenceRegister.VERY_HUMBLE


# =========================================================================
# Cold-start dialogue cases
# =========================================================================

class TestColdStartDialogueCases:
    """Integration tests for the substrate's behavior at the dialogue
    phases that the epistemic monitor governs.
    """

    def test_cold_start_phase_1_produces_very_humble(self):
        """Phase 1 of the cold-start dialogue: maturity is zero, no
        kernel, no world model, fresh untrained substrates. The
        aggregate must floor in the very_humble band so the production
        loop emits "I don't know."
        """
        torch.manual_seed(0)
        midmtg, wernicke = _fresh_substrates()
        bus = FakeBus(maturity=0.0)
        monitor = EpistemicMonitor(EpistemicMonitorConfig())

        # Drive an empty lemma activation through Wernicke's confidence
        # measurement.
        lemma_zero = torch.zeros(1, midmtg.cfg.n_lemmas)

        report = monitor.compute_confidence(
            midmtg, wernicke, lemma_zero,
            kernel=None, world_model=None, neuromod_bus=bus,
        )
        assert report.register == ConfidenceRegister.VERY_HUMBLE
        assert report.aggregate < 0.15

    def test_phase_7_higher_with_acquired_lemma_and_maturity(self):
        """Phase 7 of the cold-start dialogue: the substrate has
        acquired its name, the maturity has nudged off zero. The
        aggregate must rise above the very_humble band, although not
        necessarily into confident given the still-low maturity.
        """
        torch.manual_seed(0)
        midmtg, wernicke = _fresh_substrates()

        # Plant an acquired lemma matching the input concept.
        target_lemma = N_RESERVED_LEMMAS
        target_concept = 17
        with torch.no_grad():
            midmtg.w_c_to_l.W[target_lemma, :] = 0.0
            midmtg.w_c_to_l.W[target_lemma, target_concept] = 20.0
        midmtg.allocate_lemma(target_lemma)

        c_lex = torch.zeros(1, midmtg.cfg.n_concepts)
        c_lex[0, target_concept] = 1.0
        midmtg.reset_for_selection()
        for _ in range(midmtg.cfg.t_lemma_steps):
            midmtg.forward_production(c_lex)

        # Bus reports moderate maturity (just above the low threshold).
        bus = FakeBus(maturity=0.4)
        monitor = EpistemicMonitor(EpistemicMonitorConfig())

        # Build a one-hot lemma indicator at the planted slot.
        lemma_one_hot = torch.zeros(1, midmtg.cfg.n_lemmas)
        lemma_one_hot[0, target_lemma] = 1.0

        report = monitor.compute_confidence(
            midmtg, wernicke, lemma_one_hot,
            kernel=FakeKernel(novelty=0.2),
            world_model=FakeWorldModel(variance=0.3),
            neuromod_bus=bus,
        )
        # Acquired lemma plus low novelty plus moderate maturity should
        # produce confidence above the very_humble floor.
        assert report.aggregate > 0.15, (
            f"Aggregate {report.aggregate:.3f} should rise above "
            f"very_humble after acquisition."
        )

    def test_phase_8_world_query_floors_again(self):
        """Phase 8 of the cold-start dialogue: substrate has acquired
        its name (so maturity is non-zero) but is asked about WORLD,
        which it has not been taught. Lemma confidence floors to zero
        because the peak lands on an unallocated slot, which drives
        the aggregate below the humble threshold.
        """
        torch.manual_seed(0)
        midmtg, wernicke = _fresh_substrates()

        # Allocate one lemma for one concept.
        target_lemma = N_RESERVED_LEMMAS
        target_concept = 17
        with torch.no_grad():
            midmtg.w_c_to_l.W[target_lemma, :] = 0.0
            midmtg.w_c_to_l.W[target_lemma, target_concept] = 20.0
        midmtg.allocate_lemma(target_lemma)

        # Now feed a different concept (the WORLD query).
        novel_concept = 42
        c_lex = torch.zeros(1, midmtg.cfg.n_concepts)
        c_lex[0, novel_concept] = 1.0
        midmtg.reset_for_selection()
        for _ in range(midmtg.cfg.t_lemma_steps):
            midmtg.forward_production(c_lex)

        bus = FakeBus(maturity=0.4)
        monitor = EpistemicMonitor(EpistemicMonitorConfig())

        # Use the activation peak as the lemma indicator (this is what
        # the runtime would do).
        peak_idx = midmtg.a_lemma.abs().argmax(dim=1)
        lemma_one_hot = torch.zeros(1, midmtg.cfg.n_lemmas)
        lemma_one_hot[0, peak_idx.item()] = 1.0

        report = monitor.compute_confidence(
            midmtg, wernicke, lemma_one_hot,
            kernel=FakeKernel(novelty=0.9),  # Novel input.
            world_model=FakeWorldModel(variance=2.0),  # Untrained region.
            neuromod_bus=bus,
        )
        # Aggregate should be in humble or very_humble.
        assert report.register in (
            ConfidenceRegister.HUMBLE,
            ConfidenceRegister.VERY_HUMBLE,
        ), f"Expected humble register, got {report.register}."
        assert report.lemma_confidence == pytest.approx(0.0, abs=1e-6)


# =========================================================================
# Question-formation trigger
# =========================================================================

class TestQuestionFormationTrigger:

    def _hedged_report(self) -> ConfidenceReport:
        return ConfidenceReport(
            aggregate=0.5,
            register=ConfidenceRegister.HEDGED,
            maturity_gain=1.0,
            raw_confidence=0.5,
        )

    def _confident_report(self) -> ConfidenceReport:
        return ConfidenceReport(
            aggregate=0.9,
            register=ConfidenceRegister.CONFIDENT,
            maturity_gain=1.0,
            raw_confidence=0.9,
        )

    def test_hedged_plus_high_curiosity_fires_trigger(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        report = self._hedged_report()
        assert monitor.should_form_question(report, curiosity_signal=0.8)

    def test_hedged_with_low_curiosity_does_not_fire(self):
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        report = self._hedged_report()
        assert not monitor.should_form_question(report, curiosity_signal=0.3)

    def test_confident_with_high_curiosity_does_not_fire(self):
        """Confident substrate has the answer; should not ask a question."""
        cfg = EpistemicMonitorConfig()
        monitor = EpistemicMonitor(cfg)
        report = self._confident_report()
        assert not monitor.should_form_question(report, curiosity_signal=0.9)


# =========================================================================
# Theo signal extension
# =========================================================================

class TestTheoSignalExtension:

    def test_theo_signals_ignored_when_disabled(self):
        """When enable_theo_signals is False, providing them must not
        affect the aggregate.
        """
        torch.manual_seed(0)
        midmtg, wernicke = _fresh_substrates()
        bus = FakeBus(maturity=0.7)
        cfg = EpistemicMonitorConfig(enable_theo_signals=False)
        monitor = EpistemicMonitor(cfg)
        lemma_one_hot = torch.zeros(1, midmtg.cfg.n_lemmas)
        lemma_one_hot[0, 0] = 1.0

        # Seed before each call so the phonological confidence
        # perturbation noise is deterministic across the two calls.
        torch.manual_seed(123)
        report_no_theo = monitor.compute_confidence(
            midmtg, wernicke, lemma_one_hot,
            neuromod_bus=bus,
        )
        torch.manual_seed(123)
        report_with_theo = monitor.compute_confidence(
            midmtg, wernicke, lemma_one_hot,
            neuromod_bus=bus,
            theo_signals={
                "engram_retrieval_confidence": 1.0,
                "crystallization_confidence": 1.0,
            },
        )
        # Aggregates must match (Theo signals ignored).
        assert report_no_theo.aggregate == pytest.approx(
            report_with_theo.aggregate, abs=1e-6
        )

    def test_theo_signals_used_when_enabled(self):
        torch.manual_seed(0)
        midmtg, wernicke = _fresh_substrates()
        bus = FakeBus(maturity=0.7)
        cfg = EpistemicMonitorConfig(
            enable_theo_signals=True,
            weight_engram=0.1,
            weight_crystallization=0.1,
        )
        monitor = EpistemicMonitor(cfg)
        lemma_one_hot = torch.zeros(1, midmtg.cfg.n_lemmas)
        lemma_one_hot[0, 0] = 1.0

        report_low = monitor.compute_confidence(
            midmtg, wernicke, lemma_one_hot,
            neuromod_bus=bus,
            theo_signals={
                "engram_retrieval_confidence": 0.0,
                "crystallization_confidence": 0.0,
            },
        )
        report_high = monitor.compute_confidence(
            midmtg, wernicke, lemma_one_hot,
            neuromod_bus=bus,
            theo_signals={
                "engram_retrieval_confidence": 1.0,
                "crystallization_confidence": 1.0,
            },
        )
        assert report_high.aggregate > report_low.aggregate


# =========================================================================
# Serialization
# =========================================================================

class TestSerialization:

    def test_round_trip_preserves_thresholds(self):
        cfg = EpistemicMonitorConfig(
            confidence_threshold_confident=0.85,
            confidence_threshold_hedged=0.45,
            confidence_threshold_humble=0.20,
            maturity_threshold_low=0.25,
            maturity_threshold_high=0.55,
        )
        original = EpistemicMonitor(cfg)
        state = original.serialize()

        restored = EpistemicMonitor(EpistemicMonitorConfig())
        restored.restore(state)

        assert restored.cfg.confidence_threshold_confident == pytest.approx(0.85)
        assert restored.cfg.confidence_threshold_hedged == pytest.approx(0.45)
        assert restored.cfg.confidence_threshold_humble == pytest.approx(0.20)
        assert restored.cfg.maturity_threshold_low == pytest.approx(0.25)
        assert restored.cfg.maturity_threshold_high == pytest.approx(0.55)


# =========================================================================
# Ablation
# =========================================================================

class TestAblation:

    def test_ablation_returns_very_humble(self):
        torch.manual_seed(0)
        midmtg, wernicke = _fresh_substrates()
        cfg = EpistemicMonitorConfig(enable_epistemic_monitor=False)
        monitor = EpistemicMonitor(cfg)
        lemma_one_hot = torch.zeros(1, midmtg.cfg.n_lemmas)

        report = monitor.compute_confidence(
            midmtg, wernicke, lemma_one_hot,
        )
        assert report.register == ConfidenceRegister.VERY_HUMBLE
        assert report.aggregate == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
