"""
test_event_loop.py
Tests for the tick-based runtime scheduler and the four TickConsumer
adapters that wrap rate-different modules.

Tests verify:
    1. The TickConsumer protocol is uniform: every consumer accepts
       tick(dt_ms), reset(), and get_diagnostic_state().
    2. The DecaySweepConsumer fires decay_unconfirmed at the configured
       interval and not before.
    3. The LemmaDecayConsumer applies gamma_lemma decay only when
       production is idle.
    4. The SpellOutConsumer drains its pending queue at one item per
       segment_interval_ms.
    5. The ArcuateTransportConsumer advances on every dt_arcuate_ms.
    6. The EventLoop advances every consumer in registration order.
    7. run_for_ms converts time to ticks correctly.
    8. run_until honors the stop condition and the max-ticks bound.
    9. reset() returns every consumer and the global clock to zero.
"""

from __future__ import annotations

from typing import List

import pytest
import torch

from runtime.event_loop_t import (
    ArcuateTransportConfig,
    ArcuateTransportConsumer,
    DecaySweepConfig,
    DecaySweepConsumer,
    EventLoop,
    EventLoopConfig,
    LemmaDecayConfig,
    LemmaDecayConsumer,
    SpellOutConfig,
    SpellOutConsumer,
)


# =========================================================================
# Helpers
# =========================================================================

class _RecordingConsumer:
    """Simple consumer that records every tick for sequencing tests."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.ticks: List[float] = []
        self.resets: int = 0

    def tick(self, dt_ms: float) -> None:
        self.ticks.append(dt_ms)

    def reset(self) -> None:
        self.resets += 1
        self.ticks = []

    def get_diagnostic_state(self) -> dict:
        return {"label": self.label, "n_ticks": len(self.ticks)}


class _FakeAcquisitionModule:
    """Minimal stand-in for the lemma_acquisition module's
    decay_unconfirmed interface. The test verifies the consumer
    invokes it on the correct interval; we do not need the full
    module for that.
    """

    def __init__(self) -> None:
        self.n_decay_calls: int = 0

    def decay_unconfirmed(self) -> None:
        self.n_decay_calls += 1


class _FakeMidMTG:
    """Minimal stand-in for mid-MTG with the cfg.gamma_lemma and
    a_lemma buffer that LemmaDecayConsumer reads.
    """

    def __init__(self, gamma_lemma: float = 0.95, n_lemmas: int = 32) -> None:
        class _Cfg:
            pass
        self.cfg = _Cfg()
        self.cfg.gamma_lemma = gamma_lemma
        self.a_lemma = torch.ones(1, n_lemmas)


class _FakeWernicke:
    """Minimal stand-in for Wernicke's spell_out interface."""

    def __init__(self) -> None:
        self.calls: List = []

    def spell_out_for_slot(self, slot_index: int) -> str:
        self.calls.append(("plain", slot_index))
        return f"WORD_{slot_index}"

    def spell_out_with_polar_question(self, slot_index: int) -> str:
        self.calls.append(("polar", slot_index))
        return f"WORD_{slot_index}?"


class _FakeArcuate:
    """Minimal stand-in for the arcuate. The transport consumer does
    not call into the arcuate directly except to advance its clock,
    so the fake just exists to satisfy the consumer's reference.
    """

    pass


# =========================================================================
# DecaySweepConsumer
# =========================================================================

class TestDecaySweepConsumer:

    def test_decay_fires_at_interval(self):
        acq = _FakeAcquisitionModule()
        consumer = DecaySweepConsumer(
            DecaySweepConfig(sweep_interval_ms=100.0), acq,
        )
        # 19 ticks at 5 ms each = 95 ms < 100 ms threshold.
        for _ in range(19):
            consumer.tick(5.0)
        assert acq.n_decay_calls == 0

        # 20th tick brings us to 100 ms exactly: should fire.
        consumer.tick(5.0)
        assert acq.n_decay_calls == 1

    def test_decay_fires_repeatedly(self):
        acq = _FakeAcquisitionModule()
        consumer = DecaySweepConsumer(
            DecaySweepConfig(sweep_interval_ms=50.0), acq,
        )
        # 1 second total, sweep every 50 ms = 20 sweeps.
        for _ in range(200):
            consumer.tick(5.0)
        assert acq.n_decay_calls == 20

    def test_disabled_does_not_fire(self):
        acq = _FakeAcquisitionModule()
        consumer = DecaySweepConsumer(
            DecaySweepConfig(enable_decay_sweep=False), acq,
        )
        for _ in range(1000):
            consumer.tick(5.0)
        assert acq.n_decay_calls == 0

    def test_reset_zeros_counters(self):
        acq = _FakeAcquisitionModule()
        consumer = DecaySweepConsumer(DecaySweepConfig(), acq)
        for _ in range(50):
            consumer.tick(5.0)
        consumer.reset()
        diag = consumer.get_diagnostic_state()
        assert diag["time_since_last_sweep_ms"] == 0.0
        assert diag["n_sweeps_run"] == 0


# =========================================================================
# LemmaDecayConsumer
# =========================================================================

class TestLemmaDecayConsumer:

    def test_decay_applies_when_idle(self):
        mid = _FakeMidMTG(gamma_lemma=0.5)  # Visible decay.
        consumer = LemmaDecayConsumer(
            LemmaDecayConfig(decay_interval_ms=50.0),
            mid,
            idle_flag_getter=lambda: True,
        )
        # 10 ticks at 5 ms = 50 ms = one decay step.
        for _ in range(10):
            consumer.tick(5.0)
        # a_lemma should have been multiplied by 0.5 once.
        assert mid.a_lemma[0, 0].item() == pytest.approx(0.5, abs=1e-6)

    def test_decay_skipped_when_production_active(self):
        mid = _FakeMidMTG(gamma_lemma=0.5)
        consumer = LemmaDecayConsumer(
            LemmaDecayConfig(decay_interval_ms=50.0),
            mid,
            idle_flag_getter=lambda: False,  # Production active.
        )
        for _ in range(50):  # Plenty of time for several decays.
            consumer.tick(5.0)
        # Decay should not have been applied.
        assert mid.a_lemma[0, 0].item() == pytest.approx(1.0, abs=1e-6)

    def test_disabled_does_not_decay(self):
        mid = _FakeMidMTG(gamma_lemma=0.5)
        consumer = LemmaDecayConsumer(
            LemmaDecayConfig(enable_lemma_decay=False),
            mid,
            idle_flag_getter=lambda: True,
        )
        for _ in range(100):
            consumer.tick(5.0)
        assert mid.a_lemma[0, 0].item() == pytest.approx(1.0, abs=1e-6)


# =========================================================================
# SpellOutConsumer
# =========================================================================

class TestSpellOutConsumer:

    def test_emits_at_segment_interval(self):
        wernicke = _FakeWernicke()
        consumer = SpellOutConsumer(
            SpellOutConfig(segment_interval_ms=25.0),
            wernicke,
        )
        outputs: List[str] = []
        consumer.enqueue_spell_out(
            slot_index=9, polar_question=False,
            callback=outputs.append,
        )

        # 4 ticks at 5 ms = 20 ms < 25 ms threshold.
        for _ in range(4):
            consumer.tick(5.0)
        assert outputs == []

        # 5th tick brings us to 25 ms: should emit.
        consumer.tick(5.0)
        assert outputs == ["WORD_9"]

    def test_polar_question_uses_polar_spell_out(self):
        wernicke = _FakeWernicke()
        consumer = SpellOutConsumer(SpellOutConfig(), wernicke)
        outputs: List[str] = []
        consumer.enqueue_spell_out(
            slot_index=17, polar_question=True,
            callback=outputs.append,
        )
        for _ in range(5):
            consumer.tick(5.0)
        assert outputs == ["WORD_17?"]
        # The fake records which spell-out method was called.
        assert wernicke.calls == [("polar", 17)]

    def test_drains_queue_in_order(self):
        wernicke = _FakeWernicke()
        consumer = SpellOutConsumer(
            SpellOutConfig(segment_interval_ms=25.0),
            wernicke,
        )
        outputs: List[str] = []
        for slot in (9, 10, 11):
            consumer.enqueue_spell_out(
                slot_index=slot, polar_question=False,
                callback=outputs.append,
            )

        # Run for 75 ms = 15 ticks. Should emit all three.
        for _ in range(15):
            consumer.tick(5.0)
        assert outputs == ["WORD_9", "WORD_10", "WORD_11"]


# =========================================================================
# ArcuateTransportConsumer
# =========================================================================

class TestArcuateTransportConsumer:

    def test_advances_on_arcuate_dt(self):
        arcuate = _FakeArcuate()
        consumer = ArcuateTransportConsumer(
            ArcuateTransportConfig(dt_arcuate_ms=5.0),
            arcuate,
        )
        # Each 5 ms tick should advance once.
        for _ in range(10):
            consumer.tick(5.0)
        diag = consumer.get_diagnostic_state()
        assert diag["n_steps"] == 10

    def test_disabled_does_not_advance(self):
        arcuate = _FakeArcuate()
        consumer = ArcuateTransportConsumer(
            ArcuateTransportConfig(enable_arcuate_transport=False),
            arcuate,
        )
        for _ in range(100):
            consumer.tick(5.0)
        diag = consumer.get_diagnostic_state()
        assert diag["n_steps"] == 0


# =========================================================================
# EventLoop
# =========================================================================

class TestEventLoopBasics:

    def test_step_advances_all_consumers(self):
        loop = EventLoop(EventLoopConfig(dt_ms=5.0))
        c1 = _RecordingConsumer("c1")
        c2 = _RecordingConsumer("c2")
        loop.register_consumer(c1)
        loop.register_consumer(c2)
        loop.step()
        assert c1.ticks == [5.0]
        assert c2.ticks == [5.0]

    def test_consumers_ticked_in_registration_order(self):
        """Order matters for consumers that depend on each other.
        Verify that registration order is preserved per tick.
        """
        sequence: List[str] = []

        class _OrderedConsumer:
            def __init__(self, label: str) -> None:
                self.label = label

            def tick(self, dt_ms: float) -> None:
                sequence.append(self.label)

            def reset(self) -> None:
                pass

            def get_diagnostic_state(self) -> dict:
                return {"label": self.label}

        loop = EventLoop(EventLoopConfig())
        loop.register_consumer(_OrderedConsumer("first"))
        loop.register_consumer(_OrderedConsumer("second"))
        loop.register_consumer(_OrderedConsumer("third"))
        loop.step()
        assert sequence == ["first", "second", "third"]

    def test_run_for_ticks_runs_exact_count(self):
        loop = EventLoop(EventLoopConfig(dt_ms=5.0))
        c = _RecordingConsumer("c")
        loop.register_consumer(c)
        loop.run_for_ticks(7)
        assert len(c.ticks) == 7
        assert loop.tick_count == 7
        assert loop.simulated_time_ms == pytest.approx(35.0)

    def test_run_for_ms_converts_correctly(self):
        loop = EventLoop(EventLoopConfig(dt_ms=5.0))
        c = _RecordingConsumer("c")
        loop.register_consumer(c)
        loop.run_for_ms(100.0)
        # 100 / 5 = 20 ticks.
        assert len(c.ticks) == 20

    def test_run_until_stops_on_condition(self):
        loop = EventLoop(EventLoopConfig(dt_ms=5.0))
        c = _RecordingConsumer("c")
        loop.register_consumer(c)
        # Stop after the 5th tick.
        loop.run_until(lambda: loop.tick_count >= 5)
        assert loop.tick_count == 5

    def test_run_until_honors_max_ticks_bound(self):
        """If the stop condition never fires, the bound limits the run."""
        loop = EventLoop(EventLoopConfig(dt_ms=5.0, max_ticks_per_run=42))
        c = _RecordingConsumer("c")
        loop.register_consumer(c)
        n = loop.run_until(lambda: False)
        assert n == 42

    def test_reset_zeros_clock_and_consumers(self):
        loop = EventLoop(EventLoopConfig(dt_ms=5.0))
        c = _RecordingConsumer("c")
        loop.register_consumer(c)
        loop.run_for_ticks(10)
        loop.reset()
        assert loop.tick_count == 0
        assert loop.simulated_time_ms == 0.0
        assert c.resets == 1

    def test_disabled_loop_does_not_step(self):
        loop = EventLoop(EventLoopConfig(enable_event_loop=False))
        c = _RecordingConsumer("c")
        loop.register_consumer(c)
        loop.step()
        loop.run_for_ticks(50)
        assert len(c.ticks) == 0


# =========================================================================
# Integration: multiple consumers running together
# =========================================================================

class TestEventLoopIntegration:
    """Verify that multiple consumers with different rates compose
    correctly under the same scheduler.
    """

    def test_decay_sweep_plus_spell_out_compose(self):
        """Run a decay-sweep consumer (100 ms interval) and a spell-out
        consumer (25 ms interval) together for 200 ms. Decay should
        fire 2 times; spell-out should drain a queue of pending
        emissions.
        """
        acq = _FakeAcquisitionModule()
        wernicke = _FakeWernicke()
        loop = EventLoop(EventLoopConfig(dt_ms=5.0))

        decay_consumer = DecaySweepConsumer(
            DecaySweepConfig(sweep_interval_ms=100.0), acq,
        )
        spell_consumer = SpellOutConsumer(
            SpellOutConfig(segment_interval_ms=25.0), wernicke,
        )
        loop.register_consumer(decay_consumer)
        loop.register_consumer(spell_consumer)

        outputs: List[str] = []
        for slot in (9, 10, 11, 12):
            spell_consumer.enqueue_spell_out(
                slot_index=slot, polar_question=False,
                callback=outputs.append,
            )

        loop.run_for_ms(200.0)

        # Decay sweep at 100 ms intervals over 200 ms = 2 sweeps.
        assert acq.n_decay_calls == 2
        # Spell-out at 25 ms intervals over 200 ms can drain up to 8;
        # we queued 4 so all should emit.
        assert outputs == ["WORD_9", "WORD_10", "WORD_11", "WORD_12"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
