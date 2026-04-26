"""
neuromodulator_bus_t.py
The Neuromodulator Bus: Runtime Broadcast Channel for the Four Modulators

BIOLOGICAL GROUNDING
====================
This file implements the runtime channel through which v2 regions read the
four major neuromodulators of the Genesis stack: dopamine (DA),
norepinephrine (NE), acetylcholine (ACh, decomposed into incremental and
decremental pathways), and serotonin (5HT). It does not model a brain
region. It models the volume-transmission property of neuromodulatory
systems: that diffuse chemical signals broadcast simultaneously to large
populations of cells, where each cell reads the local concentration at
forward time and adjusts its dynamics accordingly.

The four modeled modulators and their primary functions:

    Dopamine (DA): Reward prediction error. Encodes the difference between
    expected and received reward. In the v2 lexical-acquisition pathway, DA
    carries the perception-production agreement signal during Phase 3
    training. Originates biologically from the ventral tegmental area.
    Schultz W, Dayan P, Montague PR (1997). DOI: 10.1126/science.275.5306.1593

    Norepinephrine (NE): Arousal and unexpected uncertainty. Signals when
    something surprising or salient occurs, increasing gain on sensory
    processing. Originates from the locus coeruleus.
    Yu AJ, Dayan P (2005). DOI: 10.1016/j.neuron.2005.04.026

    Acetylcholine (ACh): Expected uncertainty and encoding-versus-retrieval
    gating. The incremental pathway from nucleus basalis to neocortex
    enhances cue salience and encoding mode; the decremental pathway from
    medial septum to hippocampus and cingulate supports latent inhibition
    and retrieval mode. v2 reads both as separate scalars because they have
    different anatomical sources and opposite functional effects.
    Hasselmo ME (2006). DOI: 10.1016/j.conb.2006.09.002
    Avery MC, Krichmar JL (2017). DOI: 10.3389/fncir.2017.00108

    Serotonin (5HT): Behavioral inhibition and tolerance to aversive
    uncertainty. Modulates persistence versus withdrawal in ambiguous
    situations.
    Dayan P, Huys QJM (2009). DOI: 10.1146/annurev.neuro.051508.135507

The Genesis stack already implements the four-modulator broadcast in the
cognitive-loop repository as the NeuromodulatorBroadcast class. That class
is the source of truth: it computes the modulator scalars from biological
sources (VTA for DA, LC for NE, NBM for ACh_inc, raphe for 5HT) and from
task-driven controllers during ablation studies. v2 must not reimplement
this logic, because doing so would create a divergent copy that has to be
kept in sync, which is the failure mode the continuity frame in Appendix C
exists to prevent.

ENGINEERING NOTE
================
This module is an adapter, not a reimplementation. It exposes the
get/set/reset interface that v2 regions need, and when the runtime
constructs a NeuromodulatorBroadcast instance from cognitive-loop, the bus
holds a reference to it and delegates reads to it. When no upstream
broadcast is provided (as in the scaffold's standalone tests), the bus
holds the modulator scalars directly with the same neutral defaults the
cognitive-loop README documents:

    DA = 0.0       (no reward prediction error)
    NE = 1.0       (neutral arousal gain)
    ACh_inc = 0.5  (balanced encoding)
    ACh_dec = 0.5  (balanced retrieval)
    5HT = 0.0      (neutral uncertainty tolerance)

Reference: Pragmi-Cognitive-Loop_v1 README, Engineering Note on Section 10.

The standalone mode is what the scaffold's region tests use. The integrated
mode is what the runtime uses when running the full cold-start dialogue
with the kernel and broadcast wired in. Both modes present the same
interface to v2 regions, so a region implementation does not need to know
which mode the bus is in.

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
from torch import Tensor


# =========================================================================
# Configuration
# =========================================================================

# Neutral default values for each modulator key. These are the values a
# region reads when no biological source has been wired up, and they are
# chosen so that a region using the modulator at its neutral value
# behaves as if the modulator were not present at all.
#
# Reference: Pragmi-Cognitive-Loop_v1 README, ablation-as-neutral pattern.
# NOT a biological quantity. Engineering convention for ablation studies.
NEUTRAL_DEFAULTS: Dict[str, float] = {
    "DA": 0.0,
    "NE": 1.0,
    "ACh_inc": 0.5,
    "ACh_dec": 0.5,
    "5HT": 0.0,
}


@dataclass
class NeuromodulatorBusConfig:
    """Configuration for a NeuromodulatorBus instance.

    The master flag follows the cognitive-loop ablation flag standard.
    Setting enable_neuromodulator_bus to False forces every get() call to
    return the neutral default for that key, which is the standard
    ablation behavior for the whole modulation system.

    NOT a biological quantity. Engineering convention.

    Attributes:
        enable_neuromodulator_bus: master flag for the whole module.
        device: torch device for the modulator scalar tensors. Default CPU.
    """

    enable_neuromodulator_bus: bool = True
    device: str = "cpu"


# =========================================================================
# The NeuromodulatorBus
# =========================================================================

class NeuromodulatorBus:
    """Runtime adapter for the four neuromodulator scalars.

    BIOLOGICAL STRUCTURE: Volume-transmission diffusion of neuromodulators
    from their respective brainstem and forebrain sources to widespread
    cortical and subcortical targets. The bus represents the
    locally-readable concentration that any cell in the target field
    encounters at any given time.

    BIOLOGICAL FUNCTION: Provides v2 regions with a single object to hold
    a reference to when they need to read modulator state at forward
    time. Regions that care about a modulator subscribe to its key at
    init time by storing a reference to the bus, then call bus.get(key)
    inside forward() to read the current value.

    Reference: Avery MC, Krichmar JL (2017). DOI: 10.3389/fncir.2017.00108

    INTERFACE CONTRACT:
        Inputs: get(key) returns the current scalar for that modulator.
        Outputs: set(key, value) writes a new scalar for that modulator.
                 reset() restores all modulators to their neutral defaults.

    UPSTREAM INTEGRATION:
        When constructed with an upstream NeuromodulatorBroadcast (the
        cognitive-loop class), get() delegates to the upstream broadcast's
        own scalar registry. This is how v2 stays consistent with the rest
        of the Genesis stack: there is one source of truth and v2 reads
        it through this adapter.

        When constructed without an upstream broadcast, the bus holds
        scalars directly. This is the standalone mode used by the
        scaffold's region tests and by smoke tests that do not need the
        full broadcast wired up.
    """

    def __init__(
        self,
        cfg: NeuromodulatorBusConfig,
        upstream_broadcast: Optional[Any] = None,
    ) -> None:
        """Initialize the bus, optionally with an upstream broadcast.

        Args:
            cfg: NeuromodulatorBusConfig.
            upstream_broadcast: optional reference to a
                cognitive-loop NeuromodulatorBroadcast instance. When
                provided, get() reads through to the upstream broadcast.
                When None, the bus holds modulator scalars directly.
        """
        self.cfg = cfg
        self.upstream = upstream_broadcast

        # Local scalar registry. Used in standalone mode and as a fallback
        # when the upstream broadcast is missing a key the v2 code reads.
        # Each value is a 0-dimensional tensor so that downstream consumers
        # can perform tensor arithmetic uniformly without dispatching on
        # type.
        device = torch.device(cfg.device)
        self._scalars: Dict[str, Tensor] = {
            key: torch.tensor(value, device=device)
            for key, value in NEUTRAL_DEFAULTS.items()
        }

    def get(self, key: str) -> Tensor:
        """Read the current scalar for a modulator.

        When the bus is ablated (enable flag False), returns the neutral
        default for the key. Otherwise, when an upstream broadcast is
        connected and has the key, reads through to it. Otherwise reads
        from the local scalar registry.

        Args:
            key: one of "DA", "NE", "ACh_inc", "ACh_dec", "5HT".

        Returns:
            0-dimensional Tensor holding the current scalar value.

        Raises:
            KeyError: if the key is not a recognized modulator.
        """
        if key not in NEUTRAL_DEFAULTS:
            raise KeyError(
                f"Unknown modulator key '{key}'. "
                f"Expected one of {sorted(NEUTRAL_DEFAULTS.keys())}."
            )

        if not self.cfg.enable_neuromodulator_bus:
            return torch.tensor(
                NEUTRAL_DEFAULTS[key],
                device=torch.device(self.cfg.device),
            )

        # Upstream broadcast takes precedence when available. The cognitive-loop
        # NeuromodulatorBroadcast exposes its scalars through a get() method
        # of its own; we duck-type rather than import to keep this adapter
        # decoupled from the upstream class for testing.
        if self.upstream is not None and hasattr(self.upstream, "get"):
            try:
                return self.upstream.get(key)
            except (KeyError, AttributeError):
                # Upstream does not have this key. Fall through to local.
                pass

        return self._scalars[key]

    def set(self, key: str, value: Tensor) -> None:
        """Write a new scalar for a modulator into the local registry.

        Does not write to the upstream broadcast. Writes are scoped to the
        local registry so that v2 modulation experiments do not perturb
        the rest of the Genesis stack.

        Args:
            key: one of the recognized modulator keys.
            value: 0-dimensional Tensor.

        Raises:
            KeyError: if the key is not a recognized modulator.
        """
        if key not in NEUTRAL_DEFAULTS:
            raise KeyError(
                f"Unknown modulator key '{key}'. "
                f"Expected one of {sorted(NEUTRAL_DEFAULTS.keys())}."
            )
        # Coerce to tensor on the configured device. Plain floats and
        # tensors on other devices are both common in caller code; this
        # normalization keeps downstream consumers from having to dispatch.
        device = torch.device(self.cfg.device)
        if not isinstance(value, Tensor):
            value = torch.tensor(value, device=device)
        else:
            value = value.to(device)
        self._scalars[key] = value

    def reset(self) -> None:
        """Reset all local modulator scalars to their neutral defaults.

        Does not reset the upstream broadcast. The upstream broadcast has
        its own reset method and is reset by its own runtime when the
        runtime decides to do so.
        """
        device = torch.device(self.cfg.device)
        for key, value in NEUTRAL_DEFAULTS.items():
            self._scalars[key] = torch.tensor(value, device=device)

    def serialize(self) -> Dict[str, float]:
        """Serialize the local scalar registry for the .soul checkpoint.

        Returns the local scalars as plain floats. The upstream broadcast
        serializes itself separately on the cognitive-loop side and is not
        the v2 substrate's responsibility to capture.

        Returns:
            dict mapping modulator key to scalar value.
        """
        return {key: value.item() for key, value in self._scalars.items()}

    def restore(self, state: Dict[str, float]) -> None:
        """Restore the local scalar registry from a .soul checkpoint.

        Tolerates missing keys (silently uses the neutral default) and
        unknown keys (silently ignores them) so that the bus does not
        break when a checkpoint format evolves. This tolerance is
        appropriate here because modulator scalars are HOT-tier state:
        they reset to reasonable values quickly during normal operation.

        Args:
            state: dict from a previous serialize() call.
        """
        device = torch.device(self.cfg.device)
        for key in NEUTRAL_DEFAULTS:
            if key in state:
                self._scalars[key] = torch.tensor(state[key], device=device)
            else:
                self._scalars[key] = torch.tensor(
                    NEUTRAL_DEFAULTS[key], device=device,
                )
