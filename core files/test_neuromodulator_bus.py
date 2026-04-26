"""
test_neuromodulator_bus.py
Tests for the NeuromodulatorBus adapter.

These tests verify:
    1. Neutral defaults match the cognitive-loop convention.
    2. Standalone mode (no upstream broadcast) holds and returns scalars
       correctly through get/set/reset.
    3. Upstream-delegation mode reads through to the upstream broadcast
       when one is provided, with local-registry fallback for keys the
       upstream does not have.
    4. Local writes do not perturb the upstream broadcast.
    5. Serialization round-trip preserves all local scalars.
    6. Ablation flag forces every get() call to return the neutral default.
"""

from __future__ import annotations

import pytest
import torch

from coordination.neuromodulator_bus_t import (
    NeuromodulatorBus,
    NeuromodulatorBusConfig,
    NEUTRAL_DEFAULTS,
)


# =========================================================================
# Helpers
# =========================================================================

class FakeUpstreamBroadcast:
    """Minimal fake of the cognitive-loop NeuromodulatorBroadcast interface.

    Implements only the parts the bus reads through: a get() method that
    raises KeyError on unknown keys. This is the duck-typing contract the
    bus relies on, so a fake that obeys it is a sufficient stand-in for
    integration tests without pulling in the upstream module.
    """

    def __init__(self, scalars: dict) -> None:
        self._scalars = {
            key: torch.tensor(value) for key, value in scalars.items()
        }

    def get(self, key: str) -> torch.Tensor:
        if key not in self._scalars:
            raise KeyError(key)
        return self._scalars[key]


# =========================================================================
# Standalone mode
# =========================================================================

class TestStandaloneMode:
    """Behavior of the bus when no upstream broadcast is connected."""

    def test_neutral_defaults_match_cognitive_loop_convention(self):
        """The neutral defaults must match the values the cognitive-loop
        README documents for ablation: DA=0, NE=1, ACh_inc=ACh_dec=0.5,
        5HT=0.
        """
        assert NEUTRAL_DEFAULTS["DA"] == 0.0
        assert NEUTRAL_DEFAULTS["NE"] == 1.0
        assert NEUTRAL_DEFAULTS["ACh_inc"] == 0.5
        assert NEUTRAL_DEFAULTS["ACh_dec"] == 0.5
        assert NEUTRAL_DEFAULTS["5HT"] == 0.0

    def test_get_returns_neutral_at_construction(self):
        """A fresh bus must return the neutral default for every key."""
        bus = NeuromodulatorBus(NeuromodulatorBusConfig())
        for key, expected in NEUTRAL_DEFAULTS.items():
            assert bus.get(key).item() == expected

    def test_set_then_get_round_trips(self):
        """Writing a value then reading it back must return the same value."""
        bus = NeuromodulatorBus(NeuromodulatorBusConfig())
        bus.set("DA", torch.tensor(0.7))
        assert bus.get("DA").item() == pytest.approx(0.7)

    def test_set_accepts_plain_float(self):
        """The set() method must coerce a plain float to a tensor."""
        bus = NeuromodulatorBus(NeuromodulatorBusConfig())
        bus.set("NE", 2.5)
        result = bus.get("NE")
        assert isinstance(result, torch.Tensor)
        assert result.item() == pytest.approx(2.5)

    def test_reset_restores_neutral_defaults(self):
        """After writes and a reset, get() must return the neutral defaults."""
        bus = NeuromodulatorBus(NeuromodulatorBusConfig())
        bus.set("DA", 0.9)
        bus.set("NE", 3.0)
        bus.reset()
        assert bus.get("DA").item() == NEUTRAL_DEFAULTS["DA"]
        assert bus.get("NE").item() == NEUTRAL_DEFAULTS["NE"]

    def test_unknown_key_in_get_raises(self):
        """Reading an unknown modulator key must raise KeyError."""
        bus = NeuromodulatorBus(NeuromodulatorBusConfig())
        with pytest.raises(KeyError, match="Unknown modulator"):
            bus.get("nonexistent_modulator")

    def test_unknown_key_in_set_raises(self):
        """Writing an unknown modulator key must raise KeyError."""
        bus = NeuromodulatorBus(NeuromodulatorBusConfig())
        with pytest.raises(KeyError, match="Unknown modulator"):
            bus.set("nonexistent_modulator", 1.0)


# =========================================================================
# Upstream delegation
# =========================================================================

class TestUpstreamDelegation:
    """Behavior of the bus when an upstream broadcast is connected."""

    def test_get_delegates_to_upstream_when_key_present(self):
        """When the upstream has the key, get() must return the upstream
        value rather than the local default.
        """
        upstream = FakeUpstreamBroadcast({"DA": 0.42, "NE": 1.7})
        bus = NeuromodulatorBus(NeuromodulatorBusConfig(), upstream)
        assert bus.get("DA").item() == pytest.approx(0.42)
        assert bus.get("NE").item() == pytest.approx(1.7)

    def test_get_falls_back_to_local_when_upstream_missing_key(self):
        """When the upstream does not have the key, get() must fall back
        to the local registry (which starts at neutral defaults).
        """
        upstream = FakeUpstreamBroadcast({"DA": 0.42})
        bus = NeuromodulatorBus(NeuromodulatorBusConfig(), upstream)
        # DA comes from upstream.
        assert bus.get("DA").item() == pytest.approx(0.42)
        # NE is missing from upstream, so falls back to local default.
        assert bus.get("NE").item() == NEUTRAL_DEFAULTS["NE"]

    def test_set_does_not_perturb_upstream(self):
        """A local set() must not modify the upstream broadcast. Writes
        are scoped to the local registry so v2 modulation experiments
        do not contaminate the rest of the Genesis stack.
        """
        upstream = FakeUpstreamBroadcast({"DA": 0.42})
        bus = NeuromodulatorBus(NeuromodulatorBusConfig(), upstream)
        # Write locally.
        bus.set("DA", 0.99)
        # The upstream's value is unchanged.
        assert upstream.get("DA").item() == pytest.approx(0.42)
        # The bus still reports the upstream value because get() prefers
        # the upstream when the upstream has the key.
        assert bus.get("DA").item() == pytest.approx(0.42)


# =========================================================================
# Serialization
# =========================================================================

class TestSerialization:
    """Round-trip serialization of the local scalar registry."""

    def test_serialize_restore_preserves_local_scalars(self):
        """Save and restore must produce a bus that returns the same
        scalars as the original on every key.
        """
        bus = NeuromodulatorBus(NeuromodulatorBusConfig())
        bus.set("DA", 0.3)
        bus.set("NE", 1.4)
        bus.set("ACh_inc", 0.8)
        bus.set("ACh_dec", 0.2)
        bus.set("5HT", -0.1)

        state = bus.serialize()

        restored = NeuromodulatorBus(NeuromodulatorBusConfig())
        restored.restore(state)

        for key in NEUTRAL_DEFAULTS:
            assert restored.get(key).item() == pytest.approx(bus.get(key).item())

    def test_restore_with_missing_keys_uses_defaults(self):
        """A partial state must restore the keys it has and fall back to
        neutral defaults for the keys it does not.
        """
        bus = NeuromodulatorBus(NeuromodulatorBusConfig())
        bus.restore({"DA": 0.7})  # Only DA provided.
        assert bus.get("DA").item() == pytest.approx(0.7)
        # Other keys should be at neutral defaults.
        assert bus.get("NE").item() == NEUTRAL_DEFAULTS["NE"]
        assert bus.get("5HT").item() == NEUTRAL_DEFAULTS["5HT"]

    def test_restore_ignores_unknown_keys(self):
        """An unknown key in a saved state must be silently ignored
        rather than raising. Tolerance is appropriate here because the
        bus is HOT-tier state and the cost of a partial mismatch is
        small.
        """
        bus = NeuromodulatorBus(NeuromodulatorBusConfig())
        # Should not raise.
        bus.restore({"DA": 0.5, "totally_made_up_modulator": 99.9})
        assert bus.get("DA").item() == pytest.approx(0.5)


# =========================================================================
# Ablation
# =========================================================================

class TestAblation:
    """When enable_neuromodulator_bus is False, every get() must return
    the neutral default regardless of what was written.
    """

    def test_ablation_forces_neutral_returns(self):
        bus = NeuromodulatorBus(
            NeuromodulatorBusConfig(enable_neuromodulator_bus=False)
        )
        # Even after writing, get() returns neutral.
        bus.set("DA", 0.99)
        assert bus.get("DA").item() == NEUTRAL_DEFAULTS["DA"]

    def test_ablation_overrides_upstream(self):
        """Even with an upstream broadcast holding non-neutral values,
        ablation must force neutral returns. This is the standard
        ablation behavior: turn off the system entirely.
        """
        upstream = FakeUpstreamBroadcast({"DA": 0.42, "NE": 2.5})
        bus = NeuromodulatorBus(
            NeuromodulatorBusConfig(enable_neuromodulator_bus=False),
            upstream,
        )
        assert bus.get("DA").item() == NEUTRAL_DEFAULTS["DA"]
        assert bus.get("NE").item() == NEUTRAL_DEFAULTS["NE"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
