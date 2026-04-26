"""
wernicke_t.py
Wernicke's: The Lexical Phonological Code Store

BIOLOGICAL GROUNDING
====================
This file models left posterior superior temporal gyrus together with
posterior middle temporal gyrus, the cortical substrate for lexical
phonological code retrieval and storage. The Indefrey and Levelt 2004
diagnostic that this is specifically the lexical code substrate is its
activation in picture naming, word generation, word reading, and word
listening, but not in pseudoword reading or pseudoword listening. The
contrast distinguishes a lexical code substrate (active when a word is
involved, regardless of input or output direction) from a sublexical
phonological substrate (which would also activate for pseudowords).

The phonological code is a distributed representation that encodes
morphemic structure and segmental content, distinct from the syntactic
content of the lemma upstream and from the syllabic structure built
downstream by Broca's. Production retrieval window 250 to 400 ms post
picture onset. Production-side code becomes available around 330 ms.
Segmental spell-out proceeds at roughly 25 ms per segment, giving a
100 to 150 ms window for syllabification input assembly downstream.

Wernicke's is bidirectional in the strong sense. The lemma-to-code
projection W_L_to_P runs in both directions through a single tied matrix.
Production reads from lemma to code, perception reads from code to lemma.
This is what makes the picture-word interference paradigm produce
phonological facilitation rather than impairment when a phonologically
related distracter is presented at SOA 0 to 150 ms: the distracter's
perception pre-activates the same code that production is about to
retrieve.

The Indefrey-Levelt diagnostic for Wernicke's-as-lexical-code (rather than
sublexical phonology) implies that lesions to W_L_to_P should produce
fluent semantically empty production with severe comprehension deficit
and neologistic paraphasias, the clinical picture of Wernicke's aphasia.
The tied-weights commitment is what makes the lesion affect both
directions, which is what matches the clinical picture better than
separate-store models.

This file implements the three equations from Section 13 of the Broca's
corpus:

    phi_l*(t) = W_L_to_P * e_l*(t)                                     (13.1)

    s(t) = sum_m spell(phi_m) * indicator[m in morphemes(l*)]          (13.2)

    d/dt a_l_percept(t) = W_P_to_L * phi_input(t)
                          - tau_decay^-1 * a_l_percept(t)              (13.3)

The phonological_confidence signal from Section 24.2.4 of the v2 spec is
implemented as the local stability of the phonological code under small
lemma perturbations.

PLACEHOLDER NOTE
================
Equation 13.2 specifies spell as a sequential decoder over the phonological
code embedding. The scaffold uses a small GRU-cell-based decoder that emits
one segment per call, conditioned on the phonological code and a hidden
state carrying previous-segment context. This is a reasonable engineering
realization of "sequential decoder over the embedding, conditioned on
embedding plus previous output." The decoder is trainable but the scaffold
does not run training; tests plant a known mapping the same way mid-MTG
tests plant a known concept-lemma pairing.

The dimensionality choices are engineering defaults:

    phonological code dimension d_phon = n_lemmas (no compression at this
        boundary; compression would imply lossy phonology which is not
        the architectural intent).

    segment vocabulary n_segments = 64 (covers IPA inventory plus stress,
        boundary, and silence markers with room).

Both are configurable. Replacing them with values from a real phonological
inventory does not change the surrounding code.

Primary grounding papers:

Indefrey P, Levelt WJM (2004). "The spatial and temporal signatures of
word production components." Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001

Hickok G, Poeppel D (2007). "The cortical organization of speech
processing." Nature Reviews Neuroscience, 8(5), 393-402.
DOI: 10.1038/nrn2113

Schwartz MF, Dell GS, Martin N, Gahl S, Sobel P (2006). "A case-series
test of the interactive two-step model of lexical access: Evidence from
picture naming." Journal of Memory and Language, 54(2), 228-264.
DOI: 10.1016/j.jml.2005.10.001

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from substrate.tied_substrate_t import TiedSubstrate, TiedSubstrateConfig


# =========================================================================
# Configuration
# =========================================================================

# Special segment indices in the segment vocabulary. Reserved at the low
# indices of the segment space. The biology has functional analogs for
# each of these (silence, syllable boundary marker, end-of-word marker)
# but they are encoded continuously in the spectrotemporal dynamics
# rather than as discrete tokens. The discrete encoding here is an
# engineering choice that makes the spell-out output legible to the
# downstream syllabification machinery.
# NOT a biological quantity in this exact form. Engineering convention.
SEGMENT_SILENCE: int = 0
SEGMENT_SYLLABLE_BOUNDARY: int = 1
SEGMENT_WORD_END: int = 2
N_RESERVED_SEGMENTS: int = 3


@dataclass
class WernickeConfig:
    """Configuration for the Wernicke's region.

    Master flag and sub-flags follow the cognitive-loop ablation flag
    standard. Each sub-flag corresponds to a cited mechanism.

    Attributes:
        enable_wernicke: master flag.
        enable_persistence: enable a_l_percept decay across timesteps.
        enable_spell_out: enable incremental segment emission. When
            disabled, production direction stops at the phonological
            code without emitting segments.
        n_lemmas: must match the upstream mid-MTG n_lemmas. The
            substrate cannot have a different lemma vocabulary on the
            production and perception sides.
        d_phon: dimensionality of the phonological code embedding.
            Default equal to n_lemmas (no compression at this boundary).
        n_segments: size of the segment vocabulary. Default 64,
            covering IPA inventory plus stress and boundary markers
            with margin.
        d_decoder_hidden: hidden dimensionality of the spell-out GRU
            cell. NOT a biological quantity. Engineering choice;
            64 is enough to carry per-syllable context.
        tau_decay_steps: perceptual activation decay constant in
            timesteps. From Equation 13.3, perceptual activation
            decays exponentially. With dt = 5 ms, tau = 30 steps gives
            a 150 ms decay constant which matches the documented
            comprehension-direction transit window.
        spell_out_max_steps: safety cap on spell-out length. Prevents
            runaway emission if the decoder fails to emit a word-end
            marker. NOT a biological quantity.
        confidence_perturbation_scale: standard deviation of the
            Gaussian perturbation applied to lemma activation when
            measuring phonological_confidence. Small enough not to
            flip lemma identity, large enough to expose unstable
            retrievals.
        confidence_floor: numerical floor on the confidence signal.
    """

    enable_wernicke: bool = True
    enable_persistence: bool = True
    enable_spell_out: bool = True
    n_lemmas: int = 512
    d_phon: int = 512
    n_segments: int = 64
    d_decoder_hidden: int = 64
    tau_decay_steps: int = 30
    spell_out_max_steps: int = 32
    confidence_perturbation_scale: float = 0.05
    confidence_floor: float = 1e-6


# =========================================================================
# Wernicke's Region
# =========================================================================

class Wernicke(nn.Module):
    """The lexical phonological code store.

    BIOLOGICAL STRUCTURE: Left posterior superior temporal gyrus and
    posterior middle temporal gyrus.

    BIOLOGICAL FUNCTION: Lexical phonological code retrieval and storage.
    Activates during picture naming, word generation, word reading, and
    word listening, but not during pseudoword reading or pseudoword
    listening. The diagnostic identifies it specifically as the lexical
    code substrate.

    Reference: Indefrey P, Levelt WJM (2004). DOI: 10.1016/j.cognition.2002.06.001

    ANATOMICAL INTERFACE (production input):
        Sending structure: Lemma stratum in mid-MTG.
        Receiving structure: Lexical phonological code store in Wernicke's
            (this module).
        Connection: Lemma-to-phonological-code projection W_L_to_P.

    ANATOMICAL INTERFACE (production output):
        Sending structure: Lexical phonological code store in Wernicke's
            (this module).
        Receiving structure: Arcuate fasciculus white-matter tract.
        Connection: Output of the segmental spell-out function.

    ANATOMICAL INTERFACE (comprehension input):
        Sending structure: Auditory cortex (or text-input pipeline in
            v2 text mode).
        Receiving structure: Lexical phonological code store in Wernicke's
            (this module).
        Connection: Phonological code recognition input phi_input.

    ANATOMICAL INTERFACE (comprehension output):
        Sending structure: Lexical phonological code store in Wernicke's
            (this module).
        Receiving structure: Lemma stratum in mid-MTG.
        Connection: Phonological-code-to-lemma projection (the same
            W_L_to_P matrix, accessed in the reverse direction).

    STATE:
        Persistent perceptual lemma activation a_l_percept (a vector of
            shape (B, n_lemmas), decaying with tau_decay).
        Spell-out decoder hidden state during incremental emission
            (shape (B, d_decoder_hidden), reset between words).
    """

    def __init__(self, cfg: WernickeConfig) -> None:
        """Initialize Wernicke's with the W_L_to_P tied substrate and the
        spell-out decoder.

        Args:
            cfg: WernickeConfig.
        """
        super().__init__()
        self.cfg = cfg

        # The lemma-to-phonological-code tied substrate. W_L_to_P has
        # shape (d_phon, n_lemmas). forward_a_to_b takes a lemma
        # activation and produces a phonological code; forward_b_to_a
        # takes a perceived phonological code and drives lemma
        # activation backward into mid-MTG.
        # Reference: Equations 13.1 and 13.3 of the Broca's corpus.
        self.w_l_to_p = TiedSubstrate(TiedSubstrateConfig(
            in_dim=cfg.n_lemmas,
            out_dim=cfg.d_phon,
        ))

        # Spell-out decoder. A GRU cell that emits one segment logit
        # vector per step, conditioned on the phonological code (held
        # constant across the spell-out of a single word) and the
        # previous hidden state.
        # Reference: Equation 13.2 of the Broca's corpus, with the
        # placeholder substitution (GRU as the sequential decoder)
        # flagged in this file's header docstring.
        # NOT a biological quantity. The biology realizes sequential
        # phonological emission through coupled oscillator dynamics
        # in pSTG-Broca's interaction; the GRU is a tractable
        # engineering analog.
        self.spell_out_input_to_hidden = nn.Linear(
            cfg.d_phon + cfg.n_segments, cfg.d_decoder_hidden,
        )
        self.spell_out_hidden_to_hidden = nn.Linear(
            cfg.d_decoder_hidden, cfg.d_decoder_hidden,
        )
        self.spell_out_hidden_to_segment = nn.Linear(
            cfg.d_decoder_hidden, cfg.n_segments,
        )

        # Persistent perceptual lemma activation. Equation 13.3.
        # Decays exponentially with tau_decay_steps between calls.
        # Reference: v2 Spec Section 9 (state serialization).
        self.register_buffer(
            "a_l_percept", torch.zeros(1, cfg.n_lemmas),
        )

        # Spell-out decoder hidden state. Reset at the start of each
        # word's spell-out via reset_spell_out_state().
        self.register_buffer(
            "_spell_out_hidden", torch.zeros(1, cfg.d_decoder_hidden),
        )

        # Last emitted segment, used as input for the next decoder step.
        # Initialized to silence so the first segment of each word is
        # emitted with no prior-segment context.
        self.register_buffer(
            "_last_segment", torch.zeros(1, cfg.n_segments),
        )

    # ---------------------------------------------------------------
    # Forward: production direction
    # ---------------------------------------------------------------

    def retrieve_phonological_code(self, l_star: Tensor) -> Tensor:
        """Equation 13.1: lemma to phonological code.

        Takes the selected lemma (one-hot from mid-MTG) and projects it
        through W_L_to_P to produce the phonological code embedding.
        This is the production-direction read of the tied substrate.

        Args:
            l_star: (B, n_lemmas) selected lemma. Typically one-hot but
                the substrate accepts any normalized distribution; a
                multi-peak distribution will produce a blended code.

        Returns:
            (B, d_phon) phonological code embedding.
        """
        if not self.cfg.enable_wernicke:
            return torch.zeros(
                l_star.shape[0], self.cfg.d_phon,
                device=l_star.device, dtype=l_star.dtype,
            )
        return self.w_l_to_p.forward_a_to_b(l_star)

    def reset_spell_out_state(self, batch_size: int) -> None:
        """Reset the spell-out decoder hidden state for a new word.

        Called at the start of each word's spell-out so that the decoder
        does not carry hidden state from a previous word into the
        current one. Words are independent units in the spell-out
        machinery.

        Args:
            batch_size: batch dimension to allocate hidden state for.
        """
        device = self._spell_out_hidden.device
        dtype = self._spell_out_hidden.dtype
        self._spell_out_hidden = torch.zeros(
            batch_size, self.cfg.d_decoder_hidden,
            device=device, dtype=dtype,
        )
        self._last_segment = torch.zeros(
            batch_size, self.cfg.n_segments,
            device=device, dtype=dtype,
        )

    def emit_next_segment(self, phon_code: Tensor) -> Tensor:
        """Equation 13.2: emit one segment given the phonological code.

        Each call advances the spell-out decoder by one timestep,
        conditioned on the phonological code (held constant across the
        spell-out of a single word) and the previous segment. Returns
        the segment logits; the runtime is responsible for argmax or
        sampling depending on the operating mode.

        The spell-out is incremental: segments become available one at
        a time at 25 ms intervals (one per runtime tick when dt = 5 ms
        and the runtime calls this every 5 ticks). This incrementality
        is load-bearing for the self-monitoring window because mid-word
        error detection depends on syllables becoming available before
        the full word has been emitted.

        Args:
            phon_code: (B, d_phon) phonological code from
                retrieve_phonological_code(). Held constant across
                successive emit_next_segment calls within a single
                word's spell-out.

        Returns:
            (B, n_segments) segment logits for this step.
        """
        if not self.cfg.enable_wernicke or not self.cfg.enable_spell_out:
            return torch.zeros(
                phon_code.shape[0], self.cfg.n_segments,
                device=phon_code.device, dtype=phon_code.dtype,
            )

        B = phon_code.shape[0]
        if self._spell_out_hidden.shape[0] != B:
            self.reset_spell_out_state(B)

        # GRU-style decoder step. Concatenate phonological code with
        # previous segment one-hot, project to hidden, apply tanh
        # nonlinearity, project to segment logits.
        decoder_input = torch.cat(
            [phon_code, self._last_segment], dim=1,
        )
        h_input = self.spell_out_input_to_hidden(decoder_input)
        h_recurrent = self.spell_out_hidden_to_hidden(self._spell_out_hidden)
        new_hidden = torch.tanh(h_input + h_recurrent)
        segment_logits = self.spell_out_hidden_to_segment(new_hidden)

        # Update state for the next step.
        self._spell_out_hidden = new_hidden
        # Use the segment one-hot of the argmax as the previous-segment
        # context for the next step. The tests plant deterministic
        # mappings; downstream argmax uses the same reading.
        argmax_idx = segment_logits.argmax(dim=1)
        new_last_segment = torch.zeros_like(self._last_segment)
        new_last_segment.scatter_(1, argmax_idx.unsqueeze(1), 1.0)
        self._last_segment = new_last_segment

        return segment_logits

    def spell_out_word(
        self,
        phon_code: Tensor,
        max_steps: Optional[int] = None,
    ) -> Tensor:
        """Run the spell-out to completion or until max_steps.

        Convenience method for tests and for runtime contexts that do not
        need the per-tick incrementality. Calls emit_next_segment
        repeatedly until a SEGMENT_WORD_END token is emitted or the step
        cap is reached.

        Args:
            phon_code: (B, d_phon) phonological code.
            max_steps: maximum number of segments to emit. Default
                cfg.spell_out_max_steps. Prevents runaway emission.

        Returns:
            (B, T_emitted) segment indices including the terminating
                word-end marker.
        """
        if max_steps is None:
            max_steps = self.cfg.spell_out_max_steps

        B = phon_code.shape[0]
        self.reset_spell_out_state(B)

        emitted_segments = []
        word_ended = torch.zeros(B, dtype=torch.bool, device=phon_code.device)

        for _ in range(max_steps):
            logits = self.emit_next_segment(phon_code)
            argmax_idx = logits.argmax(dim=1)
            emitted_segments.append(argmax_idx)
            word_ended = word_ended | (argmax_idx == SEGMENT_WORD_END)
            if word_ended.all():
                break

        return torch.stack(emitted_segments, dim=1)

    # ---------------------------------------------------------------
    # Forward: comprehension direction
    # ---------------------------------------------------------------

    def perceive_phonological_code(self, phi_input: Tensor) -> Tensor:
        """Equation 13.3: perceive a phonological code.

        Takes an incoming phonological code (from the auditory cortex
        path or from the text-input pipeline in v2 text mode) and
        updates the persistent perceptual lemma activation through the
        tied substrate accessed in the reverse direction.

        The persistent activation decays with tau_decay between calls.
        New input adds to the decayed activation, producing the
        Equation 13.3 dynamics in discrete-time form.

        Args:
            phi_input: (B, d_phon) perceived phonological code.

        Returns:
            (B, n_lemmas) updated perceptual lemma activation.
        """
        if not self.cfg.enable_wernicke:
            return torch.zeros(
                phi_input.shape[0], self.cfg.n_lemmas,
                device=phi_input.device, dtype=phi_input.dtype,
            )

        B = phi_input.shape[0]
        if self.a_l_percept.shape[0] != B:
            self.a_l_percept = torch.zeros(
                B, self.cfg.n_lemmas,
                device=phi_input.device, dtype=phi_input.dtype,
            )

        # Discrete-time form of Equation 13.3. The continuous-time
        # equation is d/dt a = W * phi - tau^-1 * a; the discrete-time
        # update is a_new = a_old * exp(-1/tau) + W * phi, where
        # exp(-1/tau) is the per-step decay factor.
        if self.cfg.enable_persistence:
            decay_factor = torch.exp(
                torch.tensor(-1.0 / self.cfg.tau_decay_steps)
            )
            decayed = self.a_l_percept * decay_factor
        else:
            decayed = torch.zeros_like(self.a_l_percept)

        new_input = self.w_l_to_p.forward_b_to_a(phi_input)
        self.a_l_percept = decayed + new_input

        return self.a_l_percept

    # ---------------------------------------------------------------
    # Confidence signal
    # ---------------------------------------------------------------

    def get_phonological_confidence(self, l_star: Tensor) -> Tensor:
        """Compute the phonological confidence signal from Section 24.2.4.

        The signal corresponds to "I cannot reliably produce the
        phonological form for this lemma" when low. It is measured by
        perturbing the lemma activation slightly and observing the
        variance of the resulting phonological codes. A confident
        retrieval is locally stable (small variance under perturbation).
        An unconfident retrieval flips between codes under small
        perturbations (large variance).

        Reference: v2 Spec Section 24.2.4.

        Args:
            l_star: (B, n_lemmas) selected lemma to measure confidence
                for. Typically the same one-hot indicator that
                retrieve_phonological_code received.

        Returns:
            (B,) tensor of phonological confidence values in [0, 1].
        """
        if not self.cfg.enable_wernicke:
            return torch.zeros(l_star.shape[0])

        # Reference code without perturbation.
        with torch.no_grad():
            phi_ref = self.retrieve_phonological_code(l_star)

            # Multiple perturbed retrievals. Each perturbation is small
            # enough not to flip the lemma identity. The variance of the
            # resulting codes is the instability measure.
            n_perturbations = 8
            perturbed_codes = []
            for _ in range(n_perturbations):
                noise = torch.randn_like(l_star) * (
                    self.cfg.confidence_perturbation_scale
                )
                phi_perturbed = self.retrieve_phonological_code(l_star + noise)
                perturbed_codes.append(phi_perturbed)
            stacked = torch.stack(perturbed_codes, dim=0)
            # (n_perturbations, B, d_phon)

            # Variance per code dimension, summed across the code, normalized
            # against the reference code's L2 norm. This gives the relative
            # instability: how much the code wobbles relative to its own
            # magnitude.
            code_variance = stacked.var(dim=0).sum(dim=1)  # (B,)
            ref_norm_sq = (phi_ref ** 2).sum(dim=1).clamp(
                min=self.cfg.confidence_floor
            )
            relative_instability = code_variance / ref_norm_sq

            # Map relative_instability to confidence in [0, 1] through an
            # exponential decay. A relative instability of zero gives
            # confidence one; large instability drives confidence toward
            # zero. The exponential gives smooth interpolation.
            # NOT a biological quantity. Engineering normalization for the
            # epistemic monitor.
            confidence = torch.exp(-relative_instability)

        return confidence

    # ---------------------------------------------------------------
    # State management
    # ---------------------------------------------------------------

    def reset_state(self) -> None:
        """Reset all persistent state to zero.

        Called between unrelated sessions. Within-session resets between
        utterances should use reset_spell_out_state() for the decoder
        and let the perceptual activation decay naturally.
        """
        self.a_l_percept.zero_()
        self._spell_out_hidden.zero_()
        self._last_segment.zero_()

    def serialize(self) -> dict:
        """Serialize Wernicke's state for the .soul checkpoint.

        COLD-tier: the W_L_to_P weight matrix, the spell-out decoder
            parameters.
        WARM-tier: the persistent perceptual activation, the spell-out
            hidden state, the last emitted segment.

        Returns:
            dict with sub-dicts "cold" and "warm".
        """
        return {
            "cold": {
                "w_l_to_p": self.w_l_to_p.serialize(),
                "decoder_input_to_hidden": (
                    self.spell_out_input_to_hidden.state_dict()
                ),
                "decoder_hidden_to_hidden": (
                    self.spell_out_hidden_to_hidden.state_dict()
                ),
                "decoder_hidden_to_segment": (
                    self.spell_out_hidden_to_segment.state_dict()
                ),
            },
            "warm": {
                "a_l_percept": self.a_l_percept.detach().cpu().clone(),
                "spell_out_hidden": (
                    self._spell_out_hidden.detach().cpu().clone()
                ),
                "last_segment": self._last_segment.detach().cpu().clone(),
            },
        }

    def restore(self, state: dict) -> None:
        """Restore Wernicke's state from a .soul checkpoint.

        Args:
            state: dict from a previous serialize() call.
        """
        cold = state["cold"]
        self.w_l_to_p.restore(cold["w_l_to_p"])
        self.spell_out_input_to_hidden.load_state_dict(
            cold["decoder_input_to_hidden"]
        )
        self.spell_out_hidden_to_hidden.load_state_dict(
            cold["decoder_hidden_to_hidden"]
        )
        self.spell_out_hidden_to_segment.load_state_dict(
            cold["decoder_hidden_to_segment"]
        )

        warm = state["warm"]
        a_l_percept = warm["a_l_percept"].to(self.a_l_percept.device)
        if a_l_percept.shape[1] != self.cfg.n_lemmas:
            raise ValueError(
                f"Wernicke restore: a_l_percept n_lemmas mismatch. "
                f"Saved {a_l_percept.shape[1]}, current {self.cfg.n_lemmas}."
            )
        self.a_l_percept = a_l_percept
        self._spell_out_hidden = warm["spell_out_hidden"].to(
            self._spell_out_hidden.device
        )
        self._last_segment = warm["last_segment"].to(self._last_segment.device)
