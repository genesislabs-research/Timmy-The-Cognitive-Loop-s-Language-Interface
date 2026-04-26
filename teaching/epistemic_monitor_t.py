"""
epistemic_monitor_t.py
The Epistemic Monitor: Aggregating Confidence Across the Substrate

BIOLOGICAL GROUNDING
====================
This file implements the substrate's continuous self-assessment of how
much it knows. It does not model a single brain region. It models a
property the substrate distributes across multiple regions: at any given
moment, the substrate's confidence in what it is about to say is a
weighted aggregate of the cleanliness of the underlying retrievals.

The neuroscience grounding for distributed confidence monitoring is the
medial prefrontal cortex and anterior cingulate complex. People with
lesions to those regions over-claim and confabulate. People with healthy
monitoring say "I don't know" because that is the most accurate thing
available, not because they are blocked from speaking. The architecture
treats the four confidence components from Section 24.2 as the
substrate's distributed monitoring substrate and aggregates them into a
single epistemic_confidence scalar that shapes the production loop's
register without blocking it.

The four confidence components, by section of the v2 spec:

    24.2.1 CA3 retrieval confidence. From the kernel's CA1 novelty
    signal. High novelty means the kernel has not stored an episode
    like the current input; low novelty means the kernel
    pattern-completes cleanly. ca3_confidence = 1 - novelty.

    24.2.2 World model ensemble agreement. From the
    WorldModelEnsemble. Five prediction heads vote on what comes next;
    agreement (low variance) means the ensemble has trained on
    similar inputs, disagreement (high variance) means the ensemble
    is uncertain about this region of coordinate space.

    24.2.3 Lemma peak sharpness. From mid-MTG's get_lemma_confidence.
    The peak-to-average ratio of the lemma activation, restricted to
    allocated slots. Sharp peak on an allocated slot means the
    substrate has acquired a lemma matching the input; flat or
    unallocated-peak means the substrate has no clear word for it.

    24.2.4 Phonological retrieval reliability. From Wernicke's
    get_phonological_confidence. Local stability of the phonological
    code projection under small lemma perturbations. Stable means the
    substrate can reliably produce the form; unstable means the
    retrieval is noisy.

The aggregation gates the raw weighted sum by global_maturity from the
NeuromodulatorBroadcast. Below maturity 0.3, the aggregate is forced
near zero because the substrate's own confidence assessments are not
yet trustworthy (the world model and lemma stratum have not learned
enough to produce reliable signals). Between 0.3 and 0.6, the gain
ramps linearly. Above 0.6, full confidence-trust. This is the
constitutional humility of an immature substrate: it cannot over-claim
because the architecture does not yet trust its own assessments.

For deployments where Theo is wired up, the aggregation extends to six
components by adding two Theo-side signals (engram retrieval confidence
and crystallization confidence). Theo is out of scope for v2 per the
Genesis Teaching for Timmy specification, so this file exposes the
six-component aggregation as an interface but defaults to the
four-component case when no Theo signals are provided.

Primary grounding papers:

Yeung N, Summerfield C (2012). "Metacognition in human decision-making:
confidence and error monitoring." Philosophical Transactions of the
Royal Society B, 367(1594), 1310-1321. DOI: 10.1098/rstb.2011.0416

Fleming SM, Dolan RJ (2012). "The neural basis of metacognitive ability."
Philosophical Transactions of the Royal Society B, 367(1594), 1338-1349.
DOI: 10.1098/rstb.2011.0417

Hensch TK (2005). "Critical period plasticity in local cortical
circuits." Nature Reviews Neuroscience, 6(11), 877-888.
DOI: 10.1038/nrn1787

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import torch
from torch import Tensor


# =========================================================================
# Confidence registers
# =========================================================================

class ConfidenceRegister(Enum):
    """The four register branches from v2 Spec Section 24.3.

    The production loop reads epistemic_confidence and dispatches to
    one of these four registers, which determines how the substrate
    speaks rather than whether it speaks.
    """

    CONFIDENT = "confident"  # epistemic_confidence >= 0.7
    HEDGED = "hedged"        # 0.4 <= epistemic_confidence < 0.7
    HUMBLE = "humble"        # 0.15 <= epistemic_confidence < 0.4
    VERY_HUMBLE = "very_humble"  # epistemic_confidence < 0.15


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class EpistemicMonitorConfig:
    """Configuration for the EpistemicMonitor.

    The four register threshold values are configurable per session and
    serializable, not hardcoded constants in module bodies. Section
    24.6 of the v2 spec specifies that the architecture must preserve
    the option to calibrate these thresholds from experience over time.

    Attributes:
        enable_epistemic_monitor: master flag.
        enable_maturity_gating: enable the global_maturity multiplier.
            When False, raw_confidence passes through unmodified.
        enable_theo_signals: enable the two-component Theo extension
            of the aggregation. When True and Theo signals are
            provided to compute_confidence, the aggregation is
            six-component instead of four-component.
        confidence_threshold_confident: lower bound of the confident
            register.
        confidence_threshold_hedged: lower bound of the hedged
            register.
        confidence_threshold_humble: lower bound of the humble
            register. Below this value, the register is very_humble.
        maturity_threshold_low: maturity_gain is zero below this value.
        maturity_threshold_high: maturity_gain is one above this value.
            Between low and high, gain ramps linearly.
        weight_ca3: weight of ca3_confidence in the raw aggregate.
        weight_world_model: weight of world_model_confidence.
        weight_lemma: weight of lemma_confidence.
        weight_phonological: weight of phonological_confidence.
        weight_engram: weight of Theo's engram_retrieval_confidence
            in the six-component case. Ignored when enable_theo_signals
            is False.
        weight_crystallization: weight of Theo's crystallization
            confidence. Ignored when enable_theo_signals is False.
    """

    enable_epistemic_monitor: bool = True
    enable_maturity_gating: bool = True
    enable_theo_signals: bool = False

    # Section 24.3 register thresholds.
    confidence_threshold_confident: float = 0.7
    confidence_threshold_hedged: float = 0.4
    confidence_threshold_humble: float = 0.15

    # Section 24.2.5 maturity gate.
    maturity_threshold_low: float = 0.3
    maturity_threshold_high: float = 0.6

    # Section 24.2.5 default equal weights (1/4 each in the four-
    # component case). Adjusted to 1/6 each if Theo signals are
    # enabled, but we keep the per-key defaults at 0.25 so the
    # four-component path matches the spec exactly without rescaling.
    weight_ca3: float = 0.25
    weight_world_model: float = 0.25
    weight_lemma: float = 0.25
    weight_phonological: float = 0.25

    # Theo extension weights, only used when enable_theo_signals is True.
    weight_engram: float = 0.0
    weight_crystallization: float = 0.0


# =========================================================================
# Result type
# =========================================================================

@dataclass
class ConfidenceReport:
    """The output of EpistemicMonitor.compute_confidence.

    Carries both the aggregate scalar and the individual components
    so the production loop and the chat interface can display the
    full vector for diagnostics. Section 24.2 of the v2 spec calls
    for the components to be preserved alongside the aggregate
    because the regions of disagreement between components are
    diagnostically interesting.
    """

    aggregate: float
    register: ConfidenceRegister
    maturity_gain: float
    raw_confidence: float

    # Component scalars, all in [0, 1].
    ca3_confidence: float = 0.0
    world_model_confidence: float = 0.0
    lemma_confidence: float = 0.0
    phonological_confidence: float = 0.0

    # Optional Theo components.
    engram_retrieval_confidence: float = 0.0
    crystallization_confidence: float = 0.0


# =========================================================================
# The EpistemicMonitor
# =========================================================================

class EpistemicMonitor:
    """Continuous self-assessment of substrate confidence.

    BIOLOGICAL STRUCTURE: A simplified placeholder for the metacognitive
    confidence signal that medial prefrontal cortex and anterior
    cingulate compute over distributed substrate activity.

    BIOLOGICAL FUNCTION: Aggregates four (or six) component confidence
    signals into a single epistemic_confidence scalar per tick, gated by
    global_maturity from the NeuromodulatorBroadcast. The aggregate
    determines the substrate's production register: confident, hedged,
    humble, or very humble. The architecture's commitment to honesty is
    that the register is the natural output of the aggregation, not a
    refusal layered on top.

    Reference: v2 Spec Section 24.

    INTERFACE CONTRACT:
        Inputs:
            compute_confidence(midmtg, wernicke, lemma_one_hot,
                kernel=None, world_model=None, neuromod_bus=None,
                theo_signals=None) - reads each component from its
                source and returns a ConfidenceReport.

        Outputs:
            ConfidenceReport with the aggregate scalar, the register
                branch, the maturity gain, and all individual
                components.

        State: stateless. The state is in the substrate regions,
            kernel, world model, and neuromodulator bus that this
            monitor reads.
    """

    def __init__(self, cfg: EpistemicMonitorConfig) -> None:
        """Initialize the monitor with the given configuration.

        Args:
            cfg: EpistemicMonitorConfig.
        """
        self.cfg = cfg

    # ---------------------------------------------------------------
    # Component readers
    # ---------------------------------------------------------------

    def _read_ca3_confidence(self, kernel: Optional[Any]) -> float:
        """Read the CA3 retrieval confidence from the kernel.

        The cognitive-loop CognitiveKernel exposes a novelty signal as
        part of its forward() output; the convention is that the
        kernel's most recent forward() call leaves a novelty scalar
        accessible through a property or a stored attribute. The
        scaffold reads the kernel's last_novelty attribute if present,
        defaulting to neutral (0.5) when no kernel is wired up or the
        kernel has not yet run.

        ca3_confidence = 1 - novelty per Section 24.2.1.

        Args:
            kernel: CognitiveKernel instance or None.

        Returns:
            ca3_confidence scalar in [0, 1].
        """
        if kernel is None:
            # No kernel wired up. Return neutral confidence so the
            # aggregate is not biased high or low by an unavailable
            # component.
            return 0.5
        # Duck-type the kernel's novelty interface. The CognitiveKernel
        # in the existing repo exposes novelty through its forward()
        # return value; we read whatever attribute the kernel uses to
        # cache its most recent value.
        novelty = getattr(kernel, "last_novelty", None)
        if novelty is None:
            return 0.5
        if isinstance(novelty, Tensor):
            novelty = novelty.mean().item()
        return float(max(0.0, min(1.0, 1.0 - novelty)))

    def _read_world_model_confidence(
        self,
        world_model: Optional[Any],
    ) -> float:
        """Read the world model ensemble agreement signal.

        The cognitive-loop WorldModelEnsemble exposes an
        ensemble_variance scalar after evaluating a coordinate; high
        variance means the ensemble disagrees, low variance means the
        ensemble has trained on similar inputs.

        world_model_confidence = 1 - normalize(variance) per
        Section 24.2.2.

        Args:
            world_model: WorldModelEnsemble instance or None.

        Returns:
            world_model_confidence scalar in [0, 1].
        """
        if world_model is None:
            return 0.5
        variance = getattr(world_model, "last_ensemble_variance", None)
        if variance is None:
            return 0.5
        if isinstance(variance, Tensor):
            variance = variance.mean().item()
        # Normalize variance to [0, 1] using a soft mapping. The
        # reference variance scale is set by typical ensemble values
        # observed during operation; the soft mapping uses
        # exp(-variance) which gives confidence near 1 for low
        # variance and near 0 for high variance.
        # NOT a biological quantity. Engineering normalization for the
        # epistemic monitor.
        import math
        return float(max(0.0, min(1.0, math.exp(-variance))))

    def _read_lemma_confidence(self, midmtg: Any) -> float:
        """Read the lemma peak sharpness from mid-MTG.

        Reads through mid-MTG's get_lemma_confidence accessor. Returns
        a scalar in [0, 1] per Section 24.2.3.

        Args:
            midmtg: MidMTG instance.

        Returns:
            lemma_confidence scalar in [0, 1].
        """
        signal = midmtg.get_lemma_confidence()
        if isinstance(signal, Tensor):
            signal = signal.mean().item()
        return float(max(0.0, min(1.0, signal)))

    def _read_phonological_confidence(
        self,
        wernicke: Any,
        lemma_one_hot: Tensor,
    ) -> float:
        """Read the phonological retrieval reliability from Wernicke's.

        Reads through Wernicke's get_phonological_confidence accessor,
        which requires the current selected lemma as input.

        Args:
            wernicke: Wernicke instance.
            lemma_one_hot: (B, n_lemmas) selected lemma indicator.

        Returns:
            phonological_confidence scalar in [0, 1].
        """
        signal = wernicke.get_phonological_confidence(lemma_one_hot)
        if isinstance(signal, Tensor):
            signal = signal.mean().item()
        return float(max(0.0, min(1.0, signal)))

    def _read_maturity(self, neuromod_bus: Optional[Any]) -> float:
        """Read the global maturity scalar from the neuromodulator bus.

        Genesis Teaching for Timmy specifies that global_maturity is a
        scalar in [0, 1] that the bus exposes alongside the four named
        modulators. The scaffold reads through the bus's get accessor
        if present, falling back to a fixed maturity of 0.0 when no
        bus is provided. The 0.0 fallback gives a constitutionally
        humble substrate, which is the correct conservative default.

        Args:
            neuromod_bus: NeuromodulatorBus instance or None.

        Returns:
            maturity scalar in [0, 1].
        """
        if neuromod_bus is None:
            return 0.0
        # The NeuromodulatorBus may not have global_maturity in its
        # default keys; attempt the read and fall back to 0.0 when
        # it raises KeyError.
        try:
            value = neuromod_bus.get("global_maturity")
            if isinstance(value, Tensor):
                value = value.item()
            return float(max(0.0, min(1.0, value)))
        except (KeyError, AttributeError):
            return 0.0

    def _compute_maturity_gain(self, maturity: float) -> float:
        """Compute the maturity_gain multiplier per Section 24.2.5.

        The maturity gain is zero below maturity_threshold_low and one
        above maturity_threshold_high, with a linear ramp between
        them. The gain multiplies raw_confidence to produce
        epistemic_confidence; below maturity_threshold_low the
        substrate is constitutionally humble.

        Args:
            maturity: scalar in [0, 1].

        Returns:
            maturity_gain scalar in [0, 1].
        """
        if not self.cfg.enable_maturity_gating:
            return 1.0
        low = self.cfg.maturity_threshold_low
        high = self.cfg.maturity_threshold_high
        if maturity <= low:
            return 0.0
        if maturity >= high:
            return 1.0
        return (maturity - low) / (high - low)

    def _classify_register(self, aggregate: float) -> ConfidenceRegister:
        """Map the aggregate confidence to one of the four registers.

        Args:
            aggregate: scalar in [0, 1].

        Returns:
            the ConfidenceRegister branch.
        """
        if aggregate >= self.cfg.confidence_threshold_confident:
            return ConfidenceRegister.CONFIDENT
        if aggregate >= self.cfg.confidence_threshold_hedged:
            return ConfidenceRegister.HEDGED
        if aggregate >= self.cfg.confidence_threshold_humble:
            return ConfidenceRegister.HUMBLE
        return ConfidenceRegister.VERY_HUMBLE

    # ---------------------------------------------------------------
    # Main aggregation
    # ---------------------------------------------------------------

    def compute_confidence(
        self,
        midmtg: Any,
        wernicke: Any,
        lemma_one_hot: Tensor,
        kernel: Optional[Any] = None,
        world_model: Optional[Any] = None,
        neuromod_bus: Optional[Any] = None,
        theo_signals: Optional[Dict[str, float]] = None,
    ) -> ConfidenceReport:
        """Compute the aggregated epistemic confidence and register.

        Reads each component from its source, weights them per the
        configuration, applies the maturity gate, and classifies the
        result into one of four register branches. Returns the full
        ConfidenceReport including individual components for
        diagnostics.

        Args:
            midmtg: MidMTG instance.
            wernicke: Wernicke instance.
            lemma_one_hot: (B, n_lemmas) selected lemma indicator,
                used by Wernicke's phonological confidence
                computation.
            kernel: optional CognitiveKernel for ca3_confidence.
            world_model: optional WorldModelEnsemble for
                world_model_confidence.
            neuromod_bus: optional NeuromodulatorBus for the maturity
                gain.
            theo_signals: optional dict with keys
                "engram_retrieval_confidence" and
                "crystallization_confidence", used when
                enable_theo_signals is True.

        Returns:
            ConfidenceReport with aggregate, register, maturity_gain,
                raw_confidence, and all individual components.
        """
        if not self.cfg.enable_epistemic_monitor:
            # Ablation: report neutral confidence in the humble band.
            # This forces the production loop into the humble register,
            # which is the safe default when monitoring is disabled.
            return ConfidenceReport(
                aggregate=0.0,
                register=ConfidenceRegister.VERY_HUMBLE,
                maturity_gain=0.0,
                raw_confidence=0.0,
            )

        # Read each component.
        ca3 = self._read_ca3_confidence(kernel)
        wm = self._read_world_model_confidence(world_model)
        lemma = self._read_lemma_confidence(midmtg)
        phon = self._read_phonological_confidence(wernicke, lemma_one_hot)

        # Theo extension components.
        engram = 0.0
        crystal = 0.0
        if self.cfg.enable_theo_signals and theo_signals is not None:
            engram = float(
                theo_signals.get("engram_retrieval_confidence", 0.0)
            )
            crystal = float(
                theo_signals.get("crystallization_confidence", 0.0)
            )

        # Weighted aggregate.
        raw = (
            self.cfg.weight_ca3 * ca3
            + self.cfg.weight_world_model * wm
            + self.cfg.weight_lemma * lemma
            + self.cfg.weight_phonological * phon
        )
        if self.cfg.enable_theo_signals:
            raw += (
                self.cfg.weight_engram * engram
                + self.cfg.weight_crystallization * crystal
            )

        # Maturity gate.
        maturity = self._read_maturity(neuromod_bus)
        gain = self._compute_maturity_gain(maturity)
        aggregate = max(0.0, min(1.0, gain * raw))

        # Register classification.
        register = self._classify_register(aggregate)

        return ConfidenceReport(
            aggregate=aggregate,
            register=register,
            maturity_gain=gain,
            raw_confidence=raw,
            ca3_confidence=ca3,
            world_model_confidence=wm,
            lemma_confidence=lemma,
            phonological_confidence=phon,
            engram_retrieval_confidence=engram,
            crystallization_confidence=crystal,
        )

    # ---------------------------------------------------------------
    # Question-formation trigger
    # ---------------------------------------------------------------

    def should_form_question(
        self,
        report: ConfidenceReport,
        curiosity_signal: float = 0.0,
        curiosity_threshold: float = 0.6,
    ) -> bool:
        """Return True when the substrate should bias toward forming a
        question rather than an assertion.

        Per the Genesis Teaching for Timmy specification, the
        production-direction trigger for question formation is the
        conjunction of two internal states: medium epistemic
        confidence (the hedged register) and a high curiosity signal
        for the active coordinate region. When both hold, the
        production loop biases the lemma activation in mid-MTG
        toward the question-word lemmas.

        The curiosity signal in v2 comes from the EpistemicSelector on
        the cognitive-loop side; the scaffold accepts it as an
        explicit argument here so the monitor can be tested in
        isolation. The runtime is responsible for reading the
        curiosity signal from the EpistemicSelector and passing it in.

        Args:
            report: the ConfidenceReport from compute_confidence.
            curiosity_signal: scalar in [0, 1] from the
                EpistemicSelector's curiosity output for the active
                coordinate region.
            curiosity_threshold: minimum curiosity signal level to
                trigger question formation. Default 0.6.

        Returns:
            True if the substrate should bias toward question
                formation, False otherwise.
        """
        if not self.cfg.enable_epistemic_monitor:
            return False
        # Hedged register is the medium-confidence band where
        # questions are appropriate; humble and very_humble are too
        # low (the substrate has nothing to ask about), confident is
        # too high (the substrate has the answer already).
        if report.register != ConfidenceRegister.HEDGED:
            return False
        return curiosity_signal >= curiosity_threshold

    # ---------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Serialize the monitor configuration for the .soul checkpoint.

        The monitor is stateless, so only the configuration is
        captured. The threshold and weight values are session-level
        configuration that should survive checkpoint cycles per
        Section 24.6 of the v2 spec.

        Returns:
            dict with all configurable fields.
        """
        return {
            "enable_epistemic_monitor": self.cfg.enable_epistemic_monitor,
            "enable_maturity_gating": self.cfg.enable_maturity_gating,
            "enable_theo_signals": self.cfg.enable_theo_signals,
            "confidence_threshold_confident": (
                self.cfg.confidence_threshold_confident
            ),
            "confidence_threshold_hedged": (
                self.cfg.confidence_threshold_hedged
            ),
            "confidence_threshold_humble": (
                self.cfg.confidence_threshold_humble
            ),
            "maturity_threshold_low": self.cfg.maturity_threshold_low,
            "maturity_threshold_high": self.cfg.maturity_threshold_high,
            "weight_ca3": self.cfg.weight_ca3,
            "weight_world_model": self.cfg.weight_world_model,
            "weight_lemma": self.cfg.weight_lemma,
            "weight_phonological": self.cfg.weight_phonological,
            "weight_engram": self.cfg.weight_engram,
            "weight_crystallization": self.cfg.weight_crystallization,
        }

    def restore(self, state: Dict[str, Any]) -> None:
        """Restore the monitor configuration from a .soul checkpoint.

        Tolerates missing keys by keeping the existing config value.
        Configuration mismatches are not structural errors; the
        monitor continues to function with whatever subset of fields
        were saved.

        Args:
            state: dict from a previous serialize() call.
        """
        for key, value in state.items():
            if hasattr(self.cfg, key):
                # Preserve the type by reading the existing value's type.
                existing = getattr(self.cfg, key)
                if isinstance(existing, bool):
                    setattr(self.cfg, key, bool(value))
                elif isinstance(existing, float):
                    setattr(self.cfg, key, float(value))
                else:
                    setattr(self.cfg, key, value)
