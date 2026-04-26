"""
test_identity_module.py
Tests for the IdentityModule.

These tests verify:
    1. Pronoun routing maps second-person pronouns to SELF and
       first-person pronouns to OTHER correctly per English convention.
    2. Routing applies the activation boost to the right mid-MTG slot.
    3. Self-active check fires when self_lemma activation crosses the
       threshold and not before.
    4. Episode tagging adds the self_relevant flag based on the
       self-active check.
    5. Production-direction pronoun selection mirrors the perception
       routing (self_lemma active -> "i" emitted).
    6. The full perception-to-production round-trip works for the
       cold-start dialogue's identity-establishment phase.
    7. Configuration serializes round-trip preserving routing table
       and thresholds.
    8. Ablation flag forces no routing and no tagging.
"""

from __future__ import annotations

import pytest
import torch

from coordination.identity_module_t import (
    IdentityModule,
    IdentityModuleConfig,
    DEFAULT_PRONOUN_ROUTING,
)
from regions.mid_mtg_t import MidMTG, MidMTGConfig


# =========================================================================
# Helpers
# =========================================================================

def _fresh_midmtg() -> MidMTG:
    """Construct a small mid-MTG for testing."""
    return MidMTG(MidMTGConfig(n_concepts=64, n_lemmas=128))


# =========================================================================
# Routing table
# =========================================================================

class TestPronounRouting:
    """The routing table maps perceived pronouns to identity lemmas."""

    def test_second_person_routes_to_self(self):
        """Second-person pronouns (you, your, etc.) refer to the
        substrate from the speaker's perspective, so they activate
        self_lemma.
        """
        for pronoun in ("you", "your", "yours", "yourself"):
            assert DEFAULT_PRONOUN_ROUTING[pronoun] == "SELF"

    def test_first_person_routes_to_other(self):
        """First-person pronouns refer to the speaker, who from the
        substrate's perspective is the conversational partner, so they
        activate other_lemma.
        """
        for pronoun in ("i", "me", "my", "mine", "myself"):
            assert DEFAULT_PRONOUN_ROUTING[pronoun] == "OTHER"

    def test_is_pronoun_recognizes_routed_tokens(self):
        ident = IdentityModule(IdentityModuleConfig())
        for pronoun in ("you", "your", "i", "my", "me"):
            assert ident.is_pronoun(pronoun)
        assert not ident.is_pronoun("name")
        assert not ident.is_pronoun("timmy")

    def test_is_pronoun_is_case_insensitive(self):
        ident = IdentityModule(IdentityModuleConfig())
        assert ident.is_pronoun("You")
        assert ident.is_pronoun("YOUR")
        assert ident.is_pronoun("I")


# =========================================================================
# Activation boost application
# =========================================================================

class TestActivationBoost:
    """Routing applies the identity-lemma boost to the right slot."""

    def test_route_you_boosts_self_lemma(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        self_idx = midmtg.identity_slot("self_lemma")
        before = midmtg.a_lemma[0, self_idx].item()

        target = ident.route_pronoun("you", midmtg)
        assert target == "SELF"

        after = midmtg.a_lemma[0, self_idx].item()
        assert after > before, (
            f"self_lemma activation should have increased, "
            f"before={before:.3f} after={after:.3f}."
        )

    def test_route_my_boosts_other_lemma(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        other_idx = midmtg.identity_slot("other_lemma")
        before = midmtg.a_lemma[0, other_idx].item()

        target = ident.route_pronoun("my", midmtg)
        assert target == "OTHER"

        after = midmtg.a_lemma[0, other_idx].item()
        assert after > before

    def test_route_non_pronoun_returns_none_and_does_not_boost(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        self_idx = midmtg.identity_slot("self_lemma")
        other_idx = midmtg.identity_slot("other_lemma")
        before_self = midmtg.a_lemma[0, self_idx].item()
        before_other = midmtg.a_lemma[0, other_idx].item()

        result = ident.route_pronoun("name", midmtg)
        assert result is None
        assert midmtg.a_lemma[0, self_idx].item() == before_self
        assert midmtg.a_lemma[0, other_idx].item() == before_other

    def test_route_perceived_phrase_routes_all_pronouns(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()

        # "your name is timmy" - "your" should route to SELF.
        routed = ident.route_perceived_phrase(
            ["your", "name", "is", "timmy"], midmtg,
        )
        assert routed == [("your", "SELF")]

    def test_route_perceived_phrase_with_multiple_pronouns(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()

        routed = ident.route_perceived_phrase(
            ["my", "name", "is", "amellia", "and", "you", "are", "timmy"],
            midmtg,
        )
        assert routed == [("my", "OTHER"), ("you", "SELF")]


# =========================================================================
# Self-active check and episode tagging
# =========================================================================

class TestSelfActiveCheck:

    def test_check_self_active_returns_false_at_baseline(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        assert not ident.check_self_active(midmtg)

    def test_check_self_active_returns_true_after_routing_you(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        ident.route_pronoun("you", midmtg)
        assert ident.check_self_active(midmtg)

    def test_check_self_active_returns_false_after_routing_my_only(self):
        """Routing 'my' boosts other_lemma, not self_lemma. Self should
        not be active.
        """
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        ident.route_pronoun("my", midmtg)
        assert not ident.check_self_active(midmtg)


class TestEpisodeTagging:

    def test_tag_episode_marks_self_relevant_when_self_active(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        ident.route_pronoun("you", midmtg)

        episode = {"content": "your name is timmy"}
        ident.tag_episode(episode, midmtg)
        assert episode["self_relevant"] is True

    def test_tag_episode_marks_not_self_relevant_at_baseline(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()

        episode = {"content": "the sky is blue"}
        ident.tag_episode(episode, midmtg)
        assert episode["self_relevant"] is False

    def test_tag_episode_returns_metadata_for_chaining(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        episode = {"content": "test"}
        result = ident.tag_episode(episode, midmtg)
        assert result is episode

    def test_tagging_disabled_keeps_flag_false(self):
        cfg = IdentityModuleConfig(enable_self_episode_tagging=False)
        ident = IdentityModule(cfg)
        midmtg = _fresh_midmtg()
        ident.route_pronoun("you", midmtg)  # self_lemma is now active.

        episode = {"content": "your name is timmy"}
        ident.tag_episode(episode, midmtg)
        # Even though self_lemma is active, tagging is disabled.
        assert episode["self_relevant"] is False


# =========================================================================
# Production direction
# =========================================================================

class TestProductionPronounSelection:

    def test_self_active_emits_first_person(self):
        """When self_lemma is active in production, the substrate
        emits a first-person pronoun ('i'), because the substrate is
        the speaker referring to itself.
        """
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        # Boost self_lemma.
        self_idx = midmtg.identity_slot("self_lemma")
        with torch.no_grad():
            midmtg.a_lemma[0, self_idx] = 5.0
        assert ident.get_production_pronoun(midmtg) == "i"

    def test_other_active_emits_second_person(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        other_idx = midmtg.identity_slot("other_lemma")
        with torch.no_grad():
            midmtg.a_lemma[0, other_idx] = 5.0
        assert ident.get_production_pronoun(midmtg) == "you"

    def test_neither_active_returns_none(self):
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()
        assert ident.get_production_pronoun(midmtg) is None


# =========================================================================
# Cold-start dialogue identity establishment
# =========================================================================

class TestColdStartDialogueIdentity:
    """Integration test: the perception-to-production round-trip for
    the cold-start dialogue's identity-establishment phase. This is a
    small slice of the full Section 24a trace, exercising only the
    identity routing.
    """

    def test_your_name_is_timmy_routing(self):
        """The phrase 'your name is timmy' must activate self_lemma so
        the substrate associates the new lemma with itself.
        """
        ident = IdentityModule(IdentityModuleConfig())
        midmtg = _fresh_midmtg()

        ident.route_perceived_phrase(
            ["your", "name", "is", "timmy"], midmtg,
        )

        # self_lemma is now active.
        assert ident.check_self_active(midmtg)

        # Producing about itself, the substrate would emit "i" (or
        # "my" in the actual dialogue, which is part of the same
        # first-person paradigm).
        assert ident.get_production_pronoun(midmtg) == "i"


# =========================================================================
# Serialization
# =========================================================================

class TestSerialization:

    def test_round_trip_preserves_routing(self):
        cfg = IdentityModuleConfig()
        original = IdentityModule(cfg)

        state = original.serialize()

        restored_cfg = IdentityModuleConfig()
        restored = IdentityModule(restored_cfg)
        restored.restore(state)

        # The routing table is preserved.
        for pronoun, target in DEFAULT_PRONOUN_ROUTING.items():
            assert restored._routing[pronoun] == target

    def test_round_trip_preserves_thresholds(self):
        cfg = IdentityModuleConfig(
            identity_boost_magnitude=7.5,
            self_activation_threshold=2.0,
        )
        original = IdentityModule(cfg)

        state = original.serialize()

        restored_cfg = IdentityModuleConfig()
        restored = IdentityModule(restored_cfg)
        restored.restore(state)

        assert restored.cfg.identity_boost_magnitude == pytest.approx(7.5)
        assert restored.cfg.self_activation_threshold == pytest.approx(2.0)

    def test_round_trip_with_custom_routing(self):
        """A non-English routing table must survive serialization
        for cross-language deployments.
        """
        custom_routing = {"tu": "SELF", "yo": "OTHER", "mi": "OTHER"}
        cfg = IdentityModuleConfig(pronoun_routing=custom_routing)
        original = IdentityModule(cfg)

        state = original.serialize()

        restored = IdentityModule(IdentityModuleConfig())
        restored.restore(state)

        assert restored._routing["tu"] == "SELF"
        assert restored._routing["yo"] == "OTHER"
        # Default pronouns were overridden, so they should not be
        # present.
        assert "you" not in restored._routing


# =========================================================================
# Ablation
# =========================================================================

class TestAblation:

    def test_ablation_returns_none_from_route(self):
        cfg = IdentityModuleConfig(enable_identity_module=False)
        ident = IdentityModule(cfg)
        midmtg = _fresh_midmtg()

        result = ident.route_pronoun("you", midmtg)
        assert result is None

    def test_ablation_does_not_modify_activation(self):
        cfg = IdentityModuleConfig(enable_identity_module=False)
        ident = IdentityModule(cfg)
        midmtg = _fresh_midmtg()
        self_idx = midmtg.identity_slot("self_lemma")
        before = midmtg.a_lemma[0, self_idx].item()

        ident.route_pronoun("you", midmtg)

        assert midmtg.a_lemma[0, self_idx].item() == before

    def test_ablation_returns_false_from_check_self_active(self):
        cfg = IdentityModuleConfig(enable_identity_module=False)
        ident = IdentityModule(cfg)
        midmtg = _fresh_midmtg()
        # Manually boost self_lemma above threshold.
        self_idx = midmtg.identity_slot("self_lemma")
        with torch.no_grad():
            midmtg.a_lemma[0, self_idx] = 10.0
        # Module is ablated, so check returns False regardless.
        assert not ident.check_self_active(midmtg)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
