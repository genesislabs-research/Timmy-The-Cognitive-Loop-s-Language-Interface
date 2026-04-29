from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol

import torch


class TickConsumer(Protocol):
    def tick(self, dt_ms: float) -> None: ...
    def reset(self) -> None: ...
    def get_diagnostic_state(self) -> Dict[str, Any]: ...


@dataclass
class DecaySweepConfig:
    enable_decay_sweep: bool = True
    sweep_interval_ms: float = 100.0


class DecaySweepConsumer:

    def __init__(self, cfg: DecaySweepConfig, acquisition_module: Any) -> None:
        self.cfg = cfg
        self.acquisition_module = acquisition_module
        self._time_since_last_sweep_ms: float = 0.0
        self._n_sweeps_run: int = 0

    def tick(self, dt_ms: float) -> None:
        if not self.cfg.enable_decay_sweep:
            return
        self._time_since_last_sweep_ms += dt_ms
        if self._time_since_last_sweep_ms >= self.cfg.sweep_interval_ms:
            self.acquisition_module.decay_unconfirmed()
            self._time_since_last_sweep_ms = 0.0
            self._n_sweeps_run += 1

    def reset(self) -> None:
        self._time_since_last_sweep_ms = 0.0
        self._n_sweeps_run = 0

    def get_diagnostic_state(self) -> Dict[str, Any]:
        return {
            "consumer": "decay_sweep",
            "time_since_last_sweep_ms": self._time_since_last_sweep_ms,
            "n_sweeps_run": self._n_sweeps_run,
        }


@dataclass
class LemmaDecayConfig:
    enable_lemma_decay: bool = True
    decay_interval_ms: float = 50.0


class LemmaDecayConsumer:

    def __init__(
        self,
        cfg: LemmaDecayConfig,
        mid_mtg: Any,
        idle_flag_getter: Callable[[], bool],
    ) -> None:
        self.cfg = cfg
        self.mid_mtg = mid_mtg
        self._idle = idle_flag_getter
        self._time_since_last_decay_ms: float = 0.0
        self._n_decay_steps: int = 0

    def tick(self, dt_ms: float) -> None:
        if not self.cfg.enable_lemma_decay:
            return
        if not self._idle():
            return
        self._time_since_last_decay_ms += dt_ms
        if self._time_since_last_decay_ms >= self.cfg.decay_interval_ms:
            with torch.no_grad():
                gamma = self.mid_mtg.cfg.gamma_lemma
                self.mid_mtg.a_lemma.mul_(gamma)
            self._time_since_last_decay_ms = 0.0
            self._n_decay_steps += 1

    def reset(self) -> None:
        self._time_since_last_decay_ms = 0.0
        self._n_decay_steps = 0

    def get_diagnostic_state(self) -> Dict[str, Any]:
        return {
            "consumer": "lemma_decay",
            "time_since_last_decay_ms": self._time_since_last_decay_ms,
            "n_decay_steps": self._n_decay_steps,
        }


@dataclass
class SpellOutConfig:
    enable_spell_out: bool = True
    segment_interval_ms: float = 25.0


class SpellOutConsumer:

    def __init__(self, cfg: SpellOutConfig, wernicke: Any) -> None:
        self.cfg = cfg
        self.wernicke = wernicke
        self._pending: List = []
        self._time_since_last_emission_ms: float = 0.0
        self._n_segments_emitted: int = 0

    def enqueue_spell_out(
        self,
        slot_index: int,
        polar_question: bool,
        callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._pending.append((slot_index, polar_question, callback))

    def tick(self, dt_ms: float) -> None:
        if not self.cfg.enable_spell_out:
            return
        if not self._pending:
            return
        self._time_since_last_emission_ms += dt_ms
        if self._time_since_last_emission_ms < self.cfg.segment_interval_ms:
            return
        slot_index, polar_q, cb = self._pending.pop(0)
        if polar_q:
            text = self.wernicke.spell_out_with_polar_question(slot_index)
        else:
            text = self.wernicke.spell_out_for_slot(slot_index)
        if cb is not None:
            cb(text)
        self._time_since_last_emission_ms = 0.0
        self._n_segments_emitted += 1

    def reset(self) -> None:
        self._pending.clear()
        self._time_since_last_emission_ms = 0.0
        self._n_segments_emitted = 0

    def get_diagnostic_state(self) -> Dict[str, Any]:
        return {
            "consumer": "spell_out",
            "n_pending": len(self._pending),
            "n_segments_emitted": self._n_segments_emitted,
        }


@dataclass
class ArcuateTransportConfig:
    enable_arcuate_transport: bool = True
    dt_arcuate_ms: float = 5.0


class ArcuateTransportConsumer:

    def __init__(self, cfg: ArcuateTransportConfig, arcuate: Any) -> None:
        self.cfg = cfg
        self.arcuate = arcuate
        self._time_since_last_step_ms: float = 0.0
        self._n_steps: int = 0

    def tick(self, dt_ms: float) -> None:
        if not self.cfg.enable_arcuate_transport:
            return
        self._time_since_last_step_ms += dt_ms
        if self._time_since_last_step_ms < self.cfg.dt_arcuate_ms:
            return
        self._time_since_last_step_ms = 0.0
        self._n_steps += 1

    def reset(self) -> None:
        self._time_since_last_step_ms = 0.0
        self._n_steps = 0

    def get_diagnostic_state(self) -> Dict[str, Any]:
        return {
            "consumer": "arcuate_transport",
            "time_since_last_step_ms": self._time_since_last_step_ms,
            "n_steps": self._n_steps,
        }


@dataclass
class EventLoopConfig:
    enable_event_loop: bool = True
    dt_ms: float = 5.0
    max_ticks_per_run: int = 100_000


class EventLoop:

    def __init__(self, cfg: EventLoopConfig) -> None:
        self.cfg = cfg
        self._consumers: List[TickConsumer] = []
        self._tick_count: int = 0
        self._simulated_time_ms: float = 0.0

    def register_consumer(self, consumer: TickConsumer) -> None:
        self._consumers.append(consumer)

    def step(self) -> None:
        if not self.cfg.enable_event_loop:
            return
        for consumer in self._consumers:
            consumer.tick(self.cfg.dt_ms)
        self._tick_count += 1
        self._simulated_time_ms += self.cfg.dt_ms

    def run_for_ticks(self, n_ticks: int) -> int:
        if not self.cfg.enable_event_loop:
            return 0
        n_to_run = min(n_ticks, self.cfg.max_ticks_per_run)
        for _ in range(n_to_run):
            self.step()
        return n_to_run

    def run_for_ms(self, duration_ms: float) -> int:
        n_ticks = int(round(duration_ms / self.cfg.dt_ms))
        return self.run_for_ticks(n_ticks)

    def run_until(
        self,
        stop_condition: Callable[[], bool],
        max_ticks: Optional[int] = None,
    ) -> int:
        if not self.cfg.enable_event_loop:
            return 0
        bound = max_ticks if max_ticks is not None else self.cfg.max_ticks_per_run
        n = 0
        while n < bound:
            self.step()
            n += 1
            if stop_condition():
                break
        return n

    def reset(self) -> None:
        for consumer in self._consumers:
            consumer.reset()
        self._tick_count = 0
        self._simulated_time_ms = 0.0

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def simulated_time_ms(self) -> float:
        return self._simulated_time_ms

    def get_diagnostic_state(self) -> Dict[str, Any]:
        return {
            "tick_count": self._tick_count,
            "simulated_time_ms": self._simulated_time_ms,
            "n_consumers": len(self._consumers),
            "consumers": [
                c.get_diagnostic_state() for c in self._consumers
            ],
        }
