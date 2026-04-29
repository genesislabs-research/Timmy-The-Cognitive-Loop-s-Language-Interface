"""
event_loop_t.py
The Tick-Based Runtime: A Uniform Protocol for Modules with Different Rates

BIOLOGICAL GROUNDING
====================
This file implements the runtime that drives the v2 speech pathway. It does
not model a brain region. It models the property that biological neural
circuits run as concurrent processes with their own timescales rather than
as a single forward pass. The arcuate fasciculus has a 5 to 10 ms
conduction delay. Wernicke's spell-out emits one segment per 25 ms.
Mid-MTG's persistent activation decays by gamma_lemma each cycle.
Lemma acquisition's decay-without-confirmation sweep runs on a multi-second
timescale. The basal ganglia's syllable-rate release fires every 150 to
250 ms when speech is in progress. These rates are different because they
correspond to different biological circuits with different physical
substrates (white matter conduction, cortical integration, hippocampal
consolidation, basal ganglia disinhibition).

The architectural commitment is that all of these rates are expressible
through the same tick interface. The previous instance's review of this
file made the constraint explicit: every module with its own rate is a
tick consumer with its own internal counters, not a different kind of
object scheduled differently. The temptation is to special-case the
modules that need different timing, and the architectural answer is
that they should all be uniform consumers of the same protocol.

The implementation pattern is a TickConsumer adapter for each module.
The adapter owns the per-module timing counter and translates the
uniform tick(dt) call into the appropriate operation on its wrapped
module. Modules themselves remain unchanged because their natural
APIs (forward_production, retrieve_phonological_code, decay_unconfirmed)
are what their tests already exercise. The adapter is the seam between
the runtime's uniform tick protocol and the module's specific
operations.

The runtime tick is the unit of scheduling. Default dt = 5 ms of
simulated biological time per tick, which matches the arcuate delay
floor. With dt = 5 ms, Wernicke's spell-out emits a segment every 5
ticks, mid-MTG's lemma activation peaks around tick 30 to 45, the
arcuate's two-step delay corresponds to 1 to 2 ticks of conduction
latency, and the lemma acquisition's decay sweep can run every 10 to
20 ticks (50 to 100 ms of simulated time). The dt is configurable;
shorter dt resolves higher-frequency dynamics at the cost of more
compute, longer dt is faster but cannot capture cross-frequency
coupling at the gamma band.

The runtime does not own the modules. The runtime holds references to
TickConsumer adapters, which hold references to the modules. The
modules can be created, tested, serialized, and restored independently
of the runtime; the runtime is just the thing that drives them when a
session is in progress.

Primary grounding papers:

Friston K (2010). "The free-energy principle: a unified brain theory?"
Nature Reviews Neuroscience, 11(2), 127-138. DOI: 10.1038/nrn2787
(The active-inference framing under which the runtime tick functions
as the unit of perception-action coupling.)

Indefrey P, Levelt WJM (2004). "The spatial and temporal signatures of
word production components." Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001 (The 250 / 330 / 455 / 600 ms
timing landmarks the runtime tick density resolves.)

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

import torch
from torch import Tensor


# =========================================================================
# The TickConsumer protocol
# =========================================================================

class TickConsumer(Protocol):
    """The uniform interface every module-with-its-own-rate implements.

    A TickConsumer holds its own time counter and decides per-tick what
    operation, if any, to perform on its wrapped module. The runtime
    advances every consumer one dt per tick; the consumer translates
    that uniform advance into the appropriate per-module action.

    The protocol has three methods. tick(dt) advances internal time and
    runs whatever the consumer is responsible for during this tick.
    reset() returns the consumer to its initial state at the start of
    a new session. get_diagnostic_state() returns a dict the chat
    interface or runtime logger can display alongside the conversation.
    """

    def tick(self, dt_ms: float) -> None:
        """Advance internal time by dt_ms and run any due operations."""
        ...

    def reset(self) -> None:
        """Return the consumer to its initial state."""
        ...

    def get_diagnostic_state(self) -> Dict[str, Any]:
        """Return a dict of internal counters for logging or display."""
        ...


# =========================================================================
# Decay sweep consumer for the lemma acquisition module
# =========================================================================

@dataclass
class DecaySweepConfig:
    """Configuration for the lemma acquisition decay-sweep consumer.

    Attributes:
        enable_decay_sweep: master flag. When False, tick is a no-op.
        sweep_interval_ms: how often the consumer runs
            decay_unconfirmed on its wrapped module. Default 100 ms,
            which is fast enough that an unconfirmed allocation
            decays well before the conversation moves on but slow
            enough that the sweep does not contend for cycles with
            production-side dynamics.
    """

    enable_decay_sweep: bool = True
    sweep_interval_ms: float = 100.0


class DecaySweepConsumer:
    """Tick consumer that runs lemma_acquisition.decay_unconfirmed
    on a fixed interval.

    Reference: Genesis Labs Research, April 2026.

    Per-tick behavior: increments an internal time counter by dt_ms.
    When the counter exceeds sweep_interval_ms, calls
    decay_unconfirmed on the wrapped module and resets the counter.
    """

    def __init__(
        self,
        cfg: DecaySweepConfig,
        acquisition_module: Any,
    ) -> None:
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


# =========================================================================
# Lemma activation decay consumer for mid-MTG
# =========================================================================

@dataclass
class LemmaDecayConfig:
    """Configuration for the mid-MTG lemma decay consumer.

    Attributes:
        enable_lemma_decay: master flag.
        decay_interval_ms: how often gamma_lemma decay is applied
            outside of the production forward pass. The forward pass
            already applies gamma_lemma when forward_production is
            called; this consumer handles the case where production
            is idle and the persistent activation should still decay
            so a stale lemma does not hold the substrate's mid-MTG
            state forever. Default 50 ms.
    """

    enable_lemma_decay: bool = True
    decay_interval_ms: float = 50.0


class LemmaDecayConsumer:
    """Tick consumer that applies gamma_lemma decay to mid-MTG when
    no production is in progress.

    The decay term is applied only when the runtime knows production
    is idle (because the production pathway applies decay itself
    during forward_production). The consumer reads its production-idle
    status from a shared flag set by the production driver elsewhere
    in the runtime.
    """

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
            # Production is active; mid-MTG's own forward_production
            # is applying decay through gamma_lemma every call. Do
            # not double-decay.
            return
        self._time_since_last_decay_ms += dt_ms
        if self._time_since_last_decay_ms >= self.cfg.decay_interval_ms:
            with torch.no_grad():
                # gamma_lemma is held on the mid_mtg config. The
                # decay multiplies the persistent a_lemma by it.
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


# =========================================================================
# Spell-out consumer for Wernicke's
# =========================================================================

@dataclass
class SpellOutConfig:
    """Configuration for the Wernicke's spell-out consumer.

    The spell-out emits one phonological segment per
    segment_interval_ms during an active production. Default 25 ms
    matches the architectural commitment from Section 13 of the
    Broca's corpus: spell-out at 25 ms per segment.

    Attributes:
        enable_spell_out: master flag.
        segment_interval_ms: time between successive segment emissions.
            NOT a free parameter; pinned to 25 ms by the corpus.
    """

    enable_spell_out: bool = True
    segment_interval_ms: float = 25.0


class SpellOutConsumer:
    """Tick consumer that drives Wernicke's incremental spell-out at
    one segment per 25 ms.

    The consumer holds a queue of pending spell-out tasks; each task
    is a (slot_index, callback) pair where callback receives the
    emitted segment when the spell-out completes. The chat interface
    places tasks in the queue when production begins; this consumer
    drains them at the segment rate.

    Note: the v2 scaffold uses spell_out_for_slot which returns the
    full word text in one call. The 25 ms per segment timing is
    enforced here at the runtime level rather than inside Wernicke's,
    because the wernicke module is a pure-computation substrate and
    timing is a runtime concern. When the spell-out decoder upgrades
    to a sequential GRU per the spec, the per-segment emission will
    move into Wernicke's directly and this consumer becomes the
    driver that calls it once per segment.
    """

    def __init__(
        self,
        cfg: SpellOutConfig,
        wernicke: Any,
    ) -> None:
        self.cfg = cfg
        self.wernicke = wernicke
        # Each pending task: (slot_index, polar_question_flag, callback).
        self._pending: List = []
        self._time_since_last_emission_ms: float = 0.0
        self._n_segments_emitted: int = 0

    def enqueue_spell_out(
        self,
        slot_index: int,
        polar_question: bool,
        callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Schedule a spell-out for the given slot.

        Args:
            slot_index: the lemma slot to spell out.
            polar_question: whether to append the polar-question marker.
            callback: optional function called with the emitted text
                when the spell-out completes.
        """
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


# =========================================================================
# Arcuate transport consumer
# =========================================================================

@dataclass
class ArcuateTransportConfig:
    """Configuration for the arcuate-transport consumer.

    The arcuate has a 5 to 10 ms conduction delay. The transport
    consumer runs the arcuate's internal delay-buffer step on every
    runtime tick that is at least dt_arcuate_ms long. Default
    dt_arcuate_ms = 5 ms matches the conduction-delay floor.

    Attributes:
        enable_arcuate_transport: master flag.
        dt_arcuate_ms: minimum tick interval at which the arcuate
            advances its delay buffer.
    """

    enable_arcuate_transport: bool = True
    dt_arcuate_ms: float = 5.0


class ArcuateTransportConsumer:
    """Tick consumer that drives the arcuate's delay-buffer
    advancement.

    The arcuate is a pure transport module; on every advancement
    cycle it shifts its delay buffer by one position. The consumer
    holds the timing counter that controls when those shifts happen,
    keeping the per-module timing logic at the runtime level.
    """

    def __init__(
        self,
        cfg: ArcuateTransportConfig,
        arcuate: Any,
    ) -> None:
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
        # The arcuate is driven by external segments; the consumer
        # does not call forward() unconditionally. The consumer's
        # role is to advance the delay buffer's clock so that any
        # segments already in flight progress through the buffer at
        # the correct rate. Since the v2 arcuate's forward() takes
        # an input segment, this consumer just resets the timing
        # counter; segments are pushed through forward() by the
        # production driver elsewhere in the runtime, on the
        # arcuate clock.
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


# =========================================================================
# The EventLoop itself
# =========================================================================

@dataclass
class EventLoopConfig:
    """Configuration for the EventLoop runtime.

    Attributes:
        enable_event_loop: master flag.
        dt_ms: simulated biological time per runtime tick. Default
            5 ms, which matches the arcuate conduction-delay floor
            and is fast enough to capture cross-frequency coupling at
            the gamma band when needed.
        max_ticks_per_run: safety bound on the number of ticks per
            run() call. Prevents runaway loops when a stop condition
            never fires. Set to a generous default that is still
            small enough that a stuck loop terminates in finite time.
    """

    enable_event_loop: bool = True
    dt_ms: float = 5.0
    max_ticks_per_run: int = 100_000


class EventLoop:
    """The tick-based runtime scheduler.

    BIOLOGICAL FUNCTION: drives all of the v2 speech pathway's
    rate-different modules through a uniform tick interface. The
    runtime advances every registered TickConsumer one dt_ms per
    tick; each consumer decides per-tick what operation, if any, to
    perform on its wrapped module.

    INTERFACE CONTRACT:
        register_consumer(consumer): add a TickConsumer to the
            schedule. Order of registration determines order of tick
            calls within each tick, which matters when multiple
            consumers depend on each other's effects in the same
            tick.
        run_until(stop_condition): advance ticks until the
            stop_condition returns True or max_ticks_per_run is
            reached, whichever comes first.
        run_for_ticks(n_ticks): advance exactly n_ticks ticks.
        run_for_ms(duration_ms): advance enough ticks to cover
            duration_ms of simulated time.
        reset(): reset every registered consumer to initial state.

    State: list of registered consumers (in registration order), the
        current tick count, the current simulated time. The consumers
        own their own state; the runtime owns the global tick clock.
    """

    def __init__(self, cfg: EventLoopConfig) -> None:
        self.cfg = cfg
        self._consumers: List[TickConsumer] = []
        self._tick_count: int = 0
        self._simulated_time_ms: float = 0.0

    def register_consumer(self, consumer: TickConsumer) -> None:
        """Add a consumer to the schedule.

        Order matters: consumers are ticked in registration order.
        If consumer A's tick depends on consumer B having already
        run this tick (e.g., A reads diagnostic state B just
        updated), B must be registered first.

        Args:
            consumer: any object implementing the TickConsumer
                protocol.
        """
        self._consumers.append(consumer)

    def step(self) -> None:
        """Advance every consumer by exactly one tick (dt_ms).

        This is the core advancement primitive. The runtime calls it
        from run_for_ticks, run_for_ms, and run_until.
        """
        if not self.cfg.enable_event_loop:
            return
        for consumer in self._consumers:
            consumer.tick(self.cfg.dt_ms)
        self._tick_count += 1
        self._simulated_time_ms += self.cfg.dt_ms

    def run_for_ticks(self, n_ticks: int) -> int:
        """Advance exactly n_ticks ticks.

        Args:
            n_ticks: number of ticks to run.

        Returns:
            number of ticks actually run (equal to n_ticks unless
                max_ticks_per_run was hit first).
        """
        if not self.cfg.enable_event_loop:
            return 0
        n_to_run = min(n_ticks, self.cfg.max_ticks_per_run)
        for _ in range(n_to_run):
            self.step()
        return n_to_run

    def run_for_ms(self, duration_ms: float) -> int:
        """Advance enough ticks to cover the given simulated time.

        Args:
            duration_ms: simulated biological time to advance.

        Returns:
            number of ticks run.
        """
        n_ticks = int(round(duration_ms / self.cfg.dt_ms))
        return self.run_for_ticks(n_ticks)

    def run_until(
        self,
        stop_condition: Callable[[], bool],
        max_ticks: Optional[int] = None,
    ) -> int:
        """Advance ticks until stop_condition returns True or the
        max-ticks bound is hit.

        Args:
            stop_condition: callable returning True when the runtime
                should stop. Evaluated after each step.
            max_ticks: optional bound for this call. Defaults to
                cfg.max_ticks_per_run.

        Returns:
            number of ticks run.
        """
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
        """Reset every consumer and the global tick clock."""
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
        """Return the runtime's full diagnostic state.

        Includes the global tick count and simulated time, plus a
        per-consumer diagnostic dict. Used by chat.py to surface the
        runtime's state in the instructor-facing display.
        """
        return {
            "tick_count": self._tick_count,
            "simulated_time_ms": self._simulated_time_ms,
            "n_consumers": len(self._consumers),
            "consumers": [
                c.get_diagnostic_state() for c in self._consumers
            ],
        }
