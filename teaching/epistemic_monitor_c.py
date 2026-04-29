from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import torch
from torch import Tensor


class ConfidenceRegister(Enum):
    CONFIDENT = "confident"
    HEDGED = "hedged"
    HUMBLE = "humble"
    VERY_HUMBLE = "very_humble"


@dataclass
class EpistemicMonitorConfig:
    enable_epistemic_monitor: bool = True
    enable_maturity_gating: bool = True
    enable_theo_signals: bool = False
    confidence_threshold_confident: float = 0.7
    confidence_threshold_hedged: float = 0.4
    confidence_threshold_humble: float = 0.15
    maturity_threshold_low: float = 0.3
    maturity_threshold_high: float = 0.6
    weight_ca3: float = 0.25
    weight_world_model: float = 0.25
    weight_lemma: float = 0.25
    weight_phonological: float = 0.25
    weight_engram: float = 0.0
    weight_crystallization: float = 0.0


@dataclass
class ConfidenceReport:
    aggregate: float
    register: ConfidenceRegister
    maturity_gain: float
    raw_confidence: float
    ca3_confidence: float = 0.0
    world_model_confidence: float = 0.0
    lemma_confidence: float = 0.0
    phonological_confidence: float = 0.0
    engram_retrieval_confidence: float = 0.0
    crystallization_confidence: float = 0.0


class EpistemicMonitor:

    def __init__(self, cfg: EpistemicMonitorConfig) -> None:
        self.cfg = cfg

    def _read_ca3_confidence(self, kernel: Optional[Any]) -> float:
        if kernel is None:
            return 0.5
        novelty = getattr(kernel, "last_novelty", None)
        if novelty is None:
            return 0.5
        if isinstance(novelty, Tensor):
            novelty = novelty.mean().item()
        return float(max(0.0, min(1.0, 1.0 - novelty)))

    def _read_world_model_confidence(self, world_model: Optional[Any]) -> float:
        if world_model is None:
            return 0.5
        variance = getattr(world_model, "last_ensemble_variance", None)
        if variance is None:
            return 0.5
        if isinstance(variance, Tensor):
            variance = variance.mean().item()
        return float(max(0.0, min(1.0, math.exp(-variance))))

    def _read_lemma_confidence(self, midmtg: Any) -> float:
        signal = midmtg.get_lemma_confidence()
        if isinstance(signal, Tensor):
            signal = signal.mean().item()
        return float(max(0.0, min(1.0, signal)))

    def _read_phonological_confidence(
        self,
        wernicke: Any,
        lemma_one_hot: Tensor,
    ) -> float:
        signal = wernicke.get_phonological_confidence(lemma_one_hot)
        if isinstance(signal, Tensor):
            signal = signal.mean().item()
        return float(max(0.0, min(1.0, signal)))

    def _read_maturity(self, neuromod_bus: Optional[Any]) -> float:
        if neuromod_bus is None:
            return 0.0
        try:
            value = neuromod_bus.get("global_maturity")
            if isinstance(value, Tensor):
                value = value.item()
            return float(max(0.0, min(1.0, value)))
        except (KeyError, AttributeError):
            return 0.0

    def _compute_maturity_gain(self, maturity: float) -> float:
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
        if aggregate >= self.cfg.confidence_threshold_confident:
            return ConfidenceRegister.CONFIDENT
        if aggregate >= self.cfg.confidence_threshold_hedged:
            return ConfidenceRegister.HEDGED
        if aggregate >= self.cfg.confidence_threshold_humble:
            return ConfidenceRegister.HUMBLE
        return ConfidenceRegister.VERY_HUMBLE

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
        if not self.cfg.enable_epistemic_monitor:
            return ConfidenceReport(
                aggregate=0.0,
                register=ConfidenceRegister.VERY_HUMBLE,
                maturity_gain=0.0,
                raw_confidence=0.0,
            )
        ca3 = self._read_ca3_confidence(kernel)
        wm = self._read_world_model_confidence(world_model)
        lemma = self._read_lemma_confidence(midmtg)
        phon = self._read_phonological_confidence(wernicke, lemma_one_hot)
        engram = 0.0
        crystal = 0.0
        if self.cfg.enable_theo_signals and theo_signals is not None:
            engram = float(
                theo_signals.get("engram_retrieval_confidence", 0.0)
            )
            crystal = float(
                theo_signals.get("crystallization_confidence", 0.0)
            )
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
        maturity = self._read_maturity(neuromod_bus)
        gain = self._compute_maturity_gain(maturity)
        aggregate = max(0.0, min(1.0, gain * raw))
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

    def should_form_question(
        self,
        report: ConfidenceReport,
        curiosity_signal: float = 0.0,
        curiosity_threshold: float = 0.6,
    ) -> bool:
        if not self.cfg.enable_epistemic_monitor:
            return False
        if report.register != ConfidenceRegister.HEDGED:
            return False
        return curiosity_signal >= curiosity_threshold

    def serialize(self) -> Dict[str, Any]:
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
        for key, value in state.items():
            if hasattr(self.cfg, key):
                existing = getattr(self.cfg, key)
                if isinstance(existing, bool):
                    setattr(self.cfg, key, bool(value))
                elif isinstance(existing, float):
                    setattr(self.cfg, key, float(value))
                else:
                    setattr(self.cfg, key, value)
