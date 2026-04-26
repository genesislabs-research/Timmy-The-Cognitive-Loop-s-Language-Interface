"""
identity_module_t.py
The Identity Module: Self-Other Distinction Through Pronoun Routing

BIOLOGICAL GROUNDING
====================
This file implements the substrate's self-other distinction, the
architectural mechanism that lets the system understand "your name"
refers to itself and "my name" refers to the speaker. The mechanism is
a pronoun routing layer that binds first-person and second-person
pronouns to two pre-allocated identity lemmas in mid-MTG: self_lemma
for the substrate's own identity, and other_lemma for the conversational
partner's identity.

The neuroscience grounding is the medial prefrontal cortex (mPFC) and
the precuneus, both of which participate in self-referential processing.
The default-mode network (DMN), whose core structure is in place by
approximately six months of age according to the Gilmore-Knickmeyer
review of structural brain development, supports the persistent
self-model. The temporo-parietal junction (TPJ) is involved in
distinguishing self from other in social contexts. A full implementation
of the self-model would simulate these regions explicitly with a
context-sensitive learned routing that incorporates autobiographical
memory and theory of mind. The v2 spec correctly defers that
implementation to a later version and uses a static routing table for
the cold-start dialogue milestone.

The reason the routing is hardcoded in v2 is that the meaning of
pronouns is not learnable from positive examples alone. A child cannot
learn that YOU refers to whoever the speaker is addressing without
already understanding that there is a distinction between speaker and
addressee. The pronoun mapping is part of language's grammatical
scaffold rather than its lexical content. Hardcoding the routing here
is equivalent to the human language acquisition device's pre-installed
grammatical sensitivity, not a claim that the system has learned
something it has not. The phonological codes for the pronouns
themselves are still acquired through Phase 3 lexical acquisition; only
the structural binding from pronoun-meaning to identity-lemma is
pre-wired.

The other side of the identity module is the episode tagging
mechanism. When a perceived input activates the self_lemma above
threshold, episodes stored to CA3 during that input get a flag in
their metadata indicating "this concerns me." This flag is what
enables preferential consolidation of self-relevant memories during
sleep, which is the architectural enactment of the closing-ritual
personal-memory tier from the Genesis Teaching for Timmy specification.

Primary grounding papers:

Northoff G, Bermpohl F (2004). "Cortical midline structures and the self."
Trends in Cognitive Sciences, 8(3), 102-107.
DOI: 10.1016/j.tics.2004.01.004

Saxe R, Kanwisher N (2003). "People thinking about thinking people: the
role of the temporo-parietal junction in 'theory of mind.'" NeuroImage,
19(4), 1835-1842. DOI: 10.1016/S1053-8119(03)00230-1

Gilmore JH, Knickmeyer RC, Gao W (2018). "Imaging structural and
functional brain development in early childhood." Nature Reviews
Neuroscience, 19(3), 123-137. DOI: 10.1038/nrn.2018.1

Cross-linguistic note: the pronoun routing table is language-specific.
A system raised in a language with multiple second-person pronouns
(formal versus informal, plural versus singular, gendered versus
ungendered) needs a routing table that captures those distinctions.
A system raised in a language without obligatory pronouns needs a
different routing pattern. The architecture commits to having an
identity-routing layer; the contents of the layer adapt to the
language. The English routing here is the v2 default, configurable
through the routing_table argument.

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor


# =========================================================================
# Configuration and routing tables
# =========================================================================

# Default English pronoun routing. Maps the perceived pronoun string to
# the identity slot it activates. SELF means activate self_lemma (the
# substrate's own identity); OTHER means activate other_lemma (the
# conversational partner's identity).
#
# The asymmetry in English: second-person pronouns (you, your) refer to
# the listener, which from the substrate's perspective is itself.
# First-person pronouns (I, me, my, mine) refer to the speaker, which
# from the substrate's perspective is the partner. This inversion is
# what makes the routing hardcoded rather than learnable: the substrate
# cannot derive from examples alone that "your" referring to itself
# is correct, because the data the substrate sees uses "your" to
# refer to the substrate without ever explaining why.
#
# Reference: v2 Spec Section 24a.5.
DEFAULT_PRONOUN_ROUTING: Dict[str, str] = {
    "you": "SELF",
    "your": "SELF",
    "yours": "SELF",
    "yourself": "SELF",
    "i": "OTHER",
    "me": "OTHER",
    "my": "OTHER",
    "mine": "OTHER",
    "myself": "OTHER",
}


@dataclass
class IdentityModuleConfig:
    """Configuration for the IdentityModule.

    The master flag follows the cognitive-loop ablation flag standard.
    Setting enable_identity_module to False forces all routing to
    return no activation boost, which is the standard ablation behavior
    and is what a substrate without self-other distinction would
    produce.

    Attributes:
        enable_identity_module: master flag for the whole module.
        enable_self_episode_tagging: enable the "this concerns me" flag
            on CA3 episodes when self_lemma fires above threshold.
            Disabling does not affect routing; it just stops the
            kernel-side tagging.
        pronoun_routing: dict mapping perceived pronoun strings to
            "SELF" or "OTHER". Defaults to English; can be overridden
            for other languages.
        identity_boost_magnitude: how much activation to add to the
            target identity lemma when a routed pronoun is perceived.
            Calibrated so the boost is large enough to drive the
            corresponding lemma above threshold but not so large that
            it overrides the substrate's other lemma activations.
            NOT a biological quantity. Engineering tuning constant.
        self_activation_threshold: minimum self_lemma activation level
            at which an episode gets the "this concerns me" tag. Higher
            values are more conservative (fewer episodes tagged); lower
            values are more inclusive.
            NOT a biological quantity. Engineering tuning constant.
    """

    enable_identity_module: bool = True
    enable_self_episode_tagging: bool = True
    pronoun_routing: Dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_PRONOUN_ROUTING)
    )
    identity_boost_magnitude: float = 5.0
    self_activation_threshold: float = 1.0


# =========================================================================
# The IdentityModule
# =========================================================================

class IdentityModule:
    """The substrate's self-other distinction layer.

    BIOLOGICAL STRUCTURE: A simplified placeholder for the
    self-referential processing network: medial prefrontal cortex,
    precuneus, and temporo-parietal junction. The full DMN-based
    self-model is deferred to a later version; the static routing
    table is the minimum viable identity layer that supports the
    cold-start dialogue.

    BIOLOGICAL FUNCTION: Routes perceived pronouns to the appropriate
    identity lemma in mid-MTG (self_lemma for second-person pronouns
    referring to the substrate, other_lemma for first-person pronouns
    referring to the speaker). Tags episodes with a "this concerns me"
    flag when self_lemma activates above threshold during the episode,
    enabling preferential consolidation of self-relevant memories.

    Reference: v2 Spec Section 24a.5.

    INTERFACE CONTRACT:
        Inputs:
            route_pronoun(token, midmtg_region) - given a perceived
                pronoun token and the mid-MTG region instance, applies
                the appropriate identity-lemma activation boost.
            check_self_active(midmtg_region) - returns True when the
                self_lemma activation in mid-MTG is above the
                self-tagging threshold.
            tag_episode(episode_metadata, midmtg_region) - in-place
                annotates the episode metadata dict with a self-relevance
                flag based on current self_lemma activation.

        State: stateless. The state lives in mid-MTG (the lemma
            activations) and in the kernel (the tagged episodes).
            This module is pure routing logic plus a configuration.
    """

    def __init__(self, cfg: IdentityModuleConfig) -> None:
        """Initialize the identity module with the given routing
        configuration.

        Args:
            cfg: IdentityModuleConfig.
        """
        self.cfg = cfg
        # Normalize pronoun tokens to lowercase for case-insensitive
        # matching. The substrate's perception pipeline does not
        # guarantee case normalization, so we do it here.
        self._routing: Dict[str, str] = {
            k.lower(): v for k, v in cfg.pronoun_routing.items()
        }

    # ---------------------------------------------------------------
    # Pronoun routing
    # ---------------------------------------------------------------

    def is_pronoun(self, token: str) -> bool:
        """Return True if the given token is a routed pronoun.

        Args:
            token: the perceived word as a lowercase string.

        Returns:
            True if the token is in the routing table, False otherwise.
        """
        if not self.cfg.enable_identity_module:
            return False
        return token.lower() in self._routing

    def route_pronoun(
        self,
        token: str,
        midmtg_region: Any,
    ) -> Optional[str]:
        """Apply the identity-lemma activation boost for a perceived
        pronoun.

        When the token is a recognized pronoun, the corresponding
        identity lemma in mid-MTG receives an additive activation
        boost equal to identity_boost_magnitude. The boost is applied
        in place to the persistent lemma activation buffer, so it
        persists across subsequent integration ticks (subject to the
        normal gamma_lemma decay).

        Args:
            token: the perceived word as a string. Case-insensitive.
            midmtg_region: a MidMTG instance. The boost is applied to
                its persistent a_lemma buffer through identity_slot
                lookup.

        Returns:
            "SELF" if the routing activated self_lemma, "OTHER" if it
                activated other_lemma, None if the token was not a
                recognized pronoun or the module is ablated.
        """
        if not self.cfg.enable_identity_module:
            return None

        normalized = token.lower()
        if normalized not in self._routing:
            return None

        target = self._routing[normalized]
        # Map the SELF/OTHER tag to the corresponding identity slot
        # name in mid-MTG. The mid-MTG region owns the slot indices
        # through its identity_slot accessor.
        if target == "SELF":
            slot_name = "self_lemma"
        elif target == "OTHER":
            slot_name = "other_lemma"
        else:
            # Unknown routing target. The configuration is invalid;
            # return None rather than corrupting mid-MTG's activation.
            return None

        slot_idx = midmtg_region.identity_slot(slot_name)
        # Apply the boost in place to the persistent activation buffer.
        # The buffer might be (B, n_lemmas); we add the boost across
        # the batch dimension. With batch size 1 (the typical scaffold
        # case), this just adds to row 0.
        with torch.no_grad():
            midmtg_region.a_lemma[:, slot_idx] += (
                self.cfg.identity_boost_magnitude
            )

        return target

    def route_perceived_phrase(
        self,
        tokens: List[str],
        midmtg_region: Any,
    ) -> List[Tuple[str, str]]:
        """Route every pronoun in a perceived phrase.

        Convenience method for the perception pipeline: takes a
        sequence of tokens and applies the appropriate routing to
        each one that is a pronoun.

        Args:
            tokens: list of perceived word tokens in order.
            midmtg_region: a MidMTG instance.

        Returns:
            list of (token, target) pairs for every pronoun that was
                routed, in order. Useful for diagnostics and for the
                chat interface's instructor-facing display.
        """
        if not self.cfg.enable_identity_module:
            return []
        routed: List[Tuple[str, str]] = []
        for tok in tokens:
            target = self.route_pronoun(tok, midmtg_region)
            if target is not None:
                routed.append((tok, target))
        return routed

    # ---------------------------------------------------------------
    # Self-relevance check and episode tagging
    # ---------------------------------------------------------------

    def check_self_active(self, midmtg_region: Any) -> bool:
        """Return True when the self_lemma activation is above the
        self-tagging threshold.

        Read by the kernel boundary code at episode-storage time to
        decide whether to flag the episode as self-relevant for
        preferential consolidation.

        Args:
            midmtg_region: a MidMTG instance.

        Returns:
            True if self_lemma activation exceeds
                self_activation_threshold, False otherwise.
        """
        if not self.cfg.enable_identity_module:
            return False
        if not self.cfg.enable_self_episode_tagging:
            return False
        slot_idx = midmtg_region.identity_slot("self_lemma")
        # Average across the batch dimension. With batch size 1 this
        # is just the scalar at that slot.
        activation = midmtg_region.a_lemma[:, slot_idx].mean().item()
        return activation >= self.cfg.self_activation_threshold

    def tag_episode(
        self,
        episode_metadata: Dict[str, Any],
        midmtg_region: Any,
    ) -> Dict[str, Any]:
        """Annotate episode metadata with the self-relevance flag.

        Called by the kernel boundary at episode-storage time. Adds a
        "self_relevant" key to the metadata dict, set to True when the
        self_lemma is currently active above threshold and False
        otherwise. The kernel's sleep_consolidation routine reads this
        flag to bias replay-prioritization toward self-relevant
        episodes, which is the architectural enactment of the closing
        ritual's personal-memory tier from the Genesis Teaching for
        Timmy specification.

        Args:
            episode_metadata: dict to annotate. Modified in place and
                also returned for chaining.
            midmtg_region: a MidMTG instance.

        Returns:
            the (modified) episode_metadata dict.
        """
        episode_metadata["self_relevant"] = self.check_self_active(midmtg_region)
        return episode_metadata

    # ---------------------------------------------------------------
    # Production direction: emitting pronouns from identity activation
    # ---------------------------------------------------------------

    def get_production_pronoun(
        self,
        midmtg_region: Any,
        register: str = "informal",
    ) -> Optional[str]:
        """Determine which pronoun to emit in production based on the
        currently active identity lemma.

        Used by the production loop when the substrate is constructing
        an utterance and the next slot calls for a pronoun. Reads the
        relative activation of self_lemma versus other_lemma in
        mid-MTG; whichever is higher determines whether the substrate
        emits a first-person or second-person pronoun.

        The asymmetry mirrors perception: when the substrate produces
        an utterance about itself, it uses the first-person pronouns
        (I/my/me) that the pronoun routing table maps to OTHER under
        perception. This is correct: the substrate is the speaker
        when producing, and "I" when produced by the substrate is the
        equivalent of "you" when perceived from the partner.

        Args:
            midmtg_region: a MidMTG instance.
            register: "informal" or "formal" or other dialect markers
                that the routing table supports. Default "informal";
                the default English routing only has informal forms.

        Returns:
            the pronoun string to emit, or None if neither identity
                lemma is active enough to drive a pronoun choice.
        """
        if not self.cfg.enable_identity_module:
            return None
        self_idx = midmtg_region.identity_slot("self_lemma")
        other_idx = midmtg_region.identity_slot("other_lemma")
        self_activation = midmtg_region.a_lemma[:, self_idx].mean().item()
        other_activation = midmtg_region.a_lemma[:, other_idx].mean().item()

        # Neither identity active enough to drive production.
        if (
            self_activation < self.cfg.self_activation_threshold
            and other_activation < self.cfg.self_activation_threshold
        ):
            return None

        # When the substrate is producing about itself, it uses the
        # first-person pronoun. The perception routing maps
        # "i" -> OTHER (the speaker, who from the listener's
        # perspective is OTHER), but in production the substrate IS
        # the speaker, so the substrate uses "i" to refer to itself
        # when self_lemma is active.
        if self_activation >= other_activation:
            return "i"
        else:
            return "you"

    # ---------------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------------

    def serialize(self) -> Dict[str, Any]:
        """Serialize the identity module configuration for the .soul
        checkpoint.

        The module is stateless (state lives in mid-MTG and the
        kernel). The serialization captures the routing configuration
        so that a substrate restored from a checkpoint with a
        non-default routing table (e.g., for a non-English language
        deployment) preserves that configuration across sessions.

        Returns:
            dict with the routing table and the threshold parameters.
        """
        return {
            "pronoun_routing": dict(self._routing),
            "identity_boost_magnitude": self.cfg.identity_boost_magnitude,
            "self_activation_threshold": self.cfg.self_activation_threshold,
            "enable_identity_module": self.cfg.enable_identity_module,
            "enable_self_episode_tagging": (
                self.cfg.enable_self_episode_tagging
            ),
        }

    def restore(self, state: Dict[str, Any]) -> None:
        """Restore the identity module configuration from a .soul
        checkpoint.

        Tolerates missing keys (uses the existing config value for
        any field not present in the saved state), since the identity
        module's configuration is more stable than its mid-MTG and
        kernel-side state. A configuration mismatch on a routing
        table is not a structural error in the way that a weight
        shape mismatch would be; the module continues to function
        with the saved or default routing.

        Args:
            state: dict from a previous serialize() call.
        """
        if "pronoun_routing" in state:
            self._routing = {
                k.lower(): v for k, v in state["pronoun_routing"].items()
            }
        if "identity_boost_magnitude" in state:
            self.cfg.identity_boost_magnitude = float(
                state["identity_boost_magnitude"]
            )
        if "self_activation_threshold" in state:
            self.cfg.self_activation_threshold = float(
                state["self_activation_threshold"]
            )
        if "enable_identity_module" in state:
            self.cfg.enable_identity_module = bool(
                state["enable_identity_module"]
            )
        if "enable_self_episode_tagging" in state:
            self.cfg.enable_self_episode_tagging = bool(
                state["enable_self_episode_tagging"]
            )
