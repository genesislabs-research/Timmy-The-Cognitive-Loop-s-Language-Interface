"""
wernicke_t.py
Wernicke's: Lexical Phonological Code Store and Surface Spell-Out

BIOLOGICAL GROUNDING
====================
This file implements the lexical phonological code stratum in the
v2 speech pathway. The biological commitment is that the cortical
locus of stored phonological forms is left posterior superior
temporal gyrus (pSTG) and the immediately adjacent posterior middle
temporal gyrus (pMTG), the region traditionally called Wernicke's
area (Hickok and Poeppel 2007). This region holds the cortical
representations that link lemmas in mid-MTG to the segmental
spell-out machinery in pIFG and the speech-motor circuit.

The substrate has two directions of operation. The production
direction takes a lemma activation from mid-MTG, projects it
through W_L_to_P to obtain a phonological code, and emits the code
into the spell-out machinery. The comprehension direction takes a
perceived phonological code (from upstream auditory processing or,
in the v2 text-only context, from a phonological-code encoding of
the input text), drives lemma activations backward through W_L_to_P
into mid-MTG, and integrates them into the persistent perceptual
lemma buffer.

This module is the refactored consumer of the LexicalSubstrate
parent. It does not construct its own W_L_to_P; it holds a
reference to the parent and reads through parent.tied_w_l_to_p for
both directions. The persistent a_l_percept buffer remains in this
module because perceptual lemma activation is state of the
phonological-code stratum specifically, not part of the lexical
knowledge.

The spell-out machinery is implemented through two paths. The
direct-text path is a slot-index-keyed dictionary that maps
known lemma slots to their canonical surface text. Reserved slots
(uncertainty markers, wh-words) populate the dictionary at
construction; acquired lemmas populate it at allocation time when
the runtime extracts the surface text from the frame recognizer's
wildcard binding. The GRU spell-out path runs on phonological codes
without slot-index context and is the architectural placeholder for
a learned segmental decoder. v2 cold-start uses the direct-text
path; the GRU path is reserved for trained-substrate operation
where novel phonological codes need to be spelled out without prior
slot-text registration.

Primary grounding papers:

Hickok G, Poeppel D (2007). "The cortical organization of speech
processing." Nature Reviews Neuroscience, 8(5), 393-402.
DOI: 10.1038/nrn2113.

Indefrey P, Levelt WJM (2004). "The spatial and temporal signatures
of word production components." Cognition, 92(1-2), 101-144.
DOI: 10.1016/j.cognition.2002.06.001.

Bohland JW, Bullock D, Guenther FH (2010). "Neural representations
and mechanisms for the performance of simple speech sequences."
Journal of Cognitive Neuroscience, 22(7), 1504-1529.
DOI: 10.1162/jocn.2009.21306. (Architecture for the GRU spell-out
path: Lashley's serial-order problem solved through gradient
activation patterns over phoneme inventory.)

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from substrate.lexical_substrate_t import LexicalSubstrate
from substrate.lemma_slots_t import (
    PHONOLOGICAL_TEXT_BY_NAME,
    SLOT_BY_NAME,
)


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class WernickeConfig:
    """Configuration for the Wernicke's lexical store.

    Master flag is first per the Genesis Labs ablation flag standard.
    NOT a biological quantity.

    Attributes:
        enable_wernicke: master flag.
        enable_persistence: when False, a_l_percept is reset between
            forward calls.
        enable_spell_out: when False, spell-out methods return
            empty strings or zero tensors.
        d_decoder_hidden: hidden width of the GRU spell-out decoder.
            NOT a biological quantity, training artifact only.
        n_segments: phoneme/segment inventory size for the GRU
            spell-out decoder. NOT a biological quantity in this
            exact form; corresponds to the cardinality of the
            language's phoneme set in the speech-motor literature.
        spell_out_max_steps: hard cap on GRU decoder iterations to
            prevent runaway emission on degenerate phonological
            codes.
        tau_decay_steps: time constant for the persistent
            a_l_percept buffer's decay between perception calls.
            NOT a biological quantity.
    """

    enable_wernicke: bool = True
    enable_persistence: bool = True
    enable_spell_out: bool = True
    d_decoder_hidden: int = 64
    n_segments: int = 64
    spell_out_max_steps: int = 32
    tau_decay_steps: int = 30


# =========================================================================
# Wernicke
# =========================================================================

class Wernicke(nn.Module):
    """Lexical phonological code store, refactored to consume the
    parent substrate.

    BIOLOGICAL STRUCTURE: Left posterior superior temporal gyrus
    and posterior middle temporal gyrus, the region called
    Wernicke's area in classical neuroanatomy.

    BIOLOGICAL FUNCTION: Stores the lexical phonological codes
    that link lemmas in mid-MTG to the segmental spell-out
    machinery. In production, takes a selected lemma and produces
    a phonological code that drives surface emission. In
    comprehension, takes a perceived phonological code and drives
    lemma activations backward into mid-MTG.

    Reference: Equations 13.1, 13.2, 13.3 of the Broca's corpus.
    Architect's Phase 0 gap document, resolution 3.

    ANATOMICAL INTERFACE (production input):
        Sending structure: Lemma stratum in mid-MTG.
        Receiving structure: Lexical phonological code store
            (this module).
        Connection: Lemma-to-phonological-code projection W_L_to_P
            (held by the LexicalSubstrate parent).

    ANATOMICAL INTERFACE (production output):
        Sending structure: Lexical phonological code store (this
            module).
        Receiving structure: Spell-out / segmental output machinery
            (BA44 sequencing buffer per Bohland et al. 2010).
        Connection: Phonological-code-to-segment readout (this
            module's GRU decoder, or the slot-index-to-text
            shortcut).

    ANATOMICAL INTERFACE (comprehension):
        Sending structure: Upstream auditory processing or text
            input.
        Receiving structure: Lexical phonological code store, then
            backward through W_L_to_P into mid-MTG's lemma stratum.
        Connection: same W_L_to_P matrix accessed in reverse.

    STATE: Persistent a_l_percept buffer for the perceptual lemma
    drive, decays at tau_decay_steps between perception events.
    Slot-index-to-text dictionary for the direct spell-out path.
    GRU decoder parameters for the segmental spell-out path. Both
    serialize through this module's state_dict.
    """

    def __init__(
        self,
        cfg: WernickeConfig,
        substrate: LexicalSubstrate,
    ) -> None:
        """Initialize Wernicke's against a parent substrate.

        Args:
            cfg: WernickeConfig.
            substrate: a LexicalSubstrate instance whose
                tied_w_l_to_p wrapper this module reads through.
        """
        super().__init__()
        self.cfg = cfg
        self.substrate = substrate
        self.n_lemmas = substrate.cfg.n_lemmas
        self.d_phon = substrate.cfg.d_phon

        # Persistent perceptual lemma activation. Holds the
        # accumulated drive from comprehension across the duration
        # of a perceived utterance.
        self.register_buffer(
            "a_l_percept", torch.zeros(1, self.n_lemmas),
        )

        # Slot-text registry for the direct spell-out path. Pre-
        # populated with reserved-slot canonical text from the slot
        # inventory; extended at allocation time by the runtime.
        # Plain dict rather than a buffer because str values do not
        # serialize through state_dict; the runtime is responsible
        # for re-registering acquired-lemma texts at restore time
        # from the .soul checkpoint's side metadata.
        self._slot_text: Dict[int, str] = {}
        for name, slot_index in SLOT_BY_NAME.items():
            text = PHONOLOGICAL_TEXT_BY_NAME.get(name, "")
            if text:
                self._slot_text[slot_index] = text

        # GRU spell-out decoder. Stub-shaped for v2 cold-start; the
        # learned weights will be trained against a phonological-
        # code-to-segment dataset in a subsequent phase.
        self.spell_out_gru = nn.GRUCell(
            input_size=cfg.n_segments,
            hidden_size=cfg.d_decoder_hidden,
        )
        self.spell_out_input_proj = nn.Linear(
            self.d_phon, cfg.d_decoder_hidden,
        )
        self.spell_out_output_proj = nn.Linear(
            cfg.d_decoder_hidden, cfg.n_segments,
        )

        self.register_buffer(
            "_spell_out_hidden",
            torch.zeros(1, cfg.d_decoder_hidden),
        )

    # =====================================================================
    # Production direction: lemma to phonological code
    # =====================================================================

    def retrieve_phonological_code(
        self, lemma_activation: Tensor,
    ) -> Tensor:
        """Project lemma activation through W_L_to_P to a
        phonological code.

        Implements Equation 13.1 of the corpus: the production
        direction reads through tied_w_l_to_p.forward_a_to_b,
        which computes lemma_activation @ W_L_to_P.T and produces
        a (B, d_phon) phonological code.

        Args:
            lemma_activation: (B, n_lemmas) selected lemma. May be
                a one-hot of the chosen slot, or a soft activation
                from mid-MTG's a_lemma.

        Returns:
            (B, d_phon) phonological code.
        """
        if not self.cfg.enable_wernicke:
            return torch.zeros(
                lemma_activation.shape[0], self.d_phon,
                device=lemma_activation.device,
                dtype=lemma_activation.dtype,
            )
        return self.substrate.tied_w_l_to_p.forward_a_to_b(
            lemma_activation,
        )

    # =====================================================================
    # Comprehension direction: phonological code to lemma drive
    # =====================================================================

    def perceive_phonological_code(
        self, phon: Tensor,
    ) -> Tensor:
        """Drive lemma activation from a perceived phonological
        code.

        Implements Equation 13.3 of the corpus: the comprehension
        direction reads through tied_w_l_to_p.forward_b_to_a, which
        computes phon @ W_L_to_P. Updates the persistent
        a_l_percept buffer with decay.

        Args:
            phon: (B, d_phon) perceived phonological code.

        Returns:
            (B, n_lemmas) lemma drive. The runtime feeds this into
                mid-MTG's a_lemma for settling.
        """
        if not self.cfg.enable_wernicke:
            return torch.zeros(
                phon.shape[0], self.n_lemmas,
                device=phon.device, dtype=phon.dtype,
            )

        drive = self.substrate.tied_w_l_to_p.forward_b_to_a(phon)

        if self.cfg.enable_persistence:
            decay = 1.0 - 1.0 / max(self.cfg.tau_decay_steps, 1)
            if self.a_l_percept.shape[0] != phon.shape[0]:
                with torch.no_grad():
                    self.a_l_percept = torch.zeros(
                        phon.shape[0], self.n_lemmas,
                        device=phon.device, dtype=phon.dtype,
                    )
            with torch.no_grad():
                self.a_l_percept.copy_(
                    decay * self.a_l_percept + drive
                )
            return self.a_l_percept.clone()
        return drive

    # =====================================================================
    # Spell-out: surface emission
    # =====================================================================

    def register_slot_text(self, slot_index: int, text: str) -> None:
        """Register surface text for a slot.

        Called by the runtime at allocation time. The frame
        recognizer's wildcard binding tells the runtime what surface
        text the new lemma should emit; the runtime calls this
        method to register that text against the allocated slot.

        Args:
            slot_index: slot to register.
            text: surface text. Empty string clears the registration.
        """
        if text:
            self._slot_text[slot_index] = text
        elif slot_index in self._slot_text:
            del self._slot_text[slot_index]

    def get_slot_text(self, slot_index: int) -> Optional[str]:
        """Return the registered surface text for a slot, or None
        if no text has been registered.

        Args:
            slot_index: slot to look up.

        Returns:
            registered surface text, or None.
        """
        return self._slot_text.get(slot_index)

    def spell_out_for_slot(self, slot_index: int) -> str:
        """Direct-text spell-out path.

        Returns the canonical surface text registered for the slot.
        This is the v2 cold-start spell-out path: substrate emits
        the slot's text directly without going through the GRU
        decoder. The mechanism is appropriate for slots whose text
        is known from acquisition (reserved slots from the slot
        inventory, acquired lemmas from frame-recognizer wildcard
        bindings).

        Args:
            slot_index: slot to spell out.

        Returns:
            surface text, or empty string if not registered.
        """
        if not self.cfg.enable_spell_out:
            return ""
        return self._slot_text.get(slot_index, "")

    def spell_out_with_polar_question(
        self, slot_index: int, polar_question: bool,
    ) -> str:
        """Slot-text spell-out with polar-question form when
        requested.

        Adds a question mark to the surface text when the production
        pathway has flagged polar-question co-activation. This is
        the architectural mechanism that turns "my name is Timmy"
        into "my name is Timmy?" during turn 3 of the cold-start
        dialogue, before confirmation arrives.

        Args:
            slot_index: slot to spell out.
            polar_question: if True, append a question mark.

        Returns:
            surface text, optionally with a trailing question mark.
        """
        text = self.spell_out_for_slot(slot_index)
        if not text:
            return ""
        if polar_question and not text.endswith("?"):
            return text + "?"
        return text

    def spell_out_gru_decoder(
        self, phon: Tensor, n_steps: Optional[int] = None,
    ) -> Tensor:
        """GRU-based segmental spell-out from a phonological code.

        The architectural placeholder for the learned segmental
        decoder. v2 cold-start does not exercise this path; it is
        present so the architectural shape supports trained-
        substrate operation where novel phonological codes need to
        be spelled out without prior slot-text registration.

        Implements Equation 13.2 of the corpus: a recurrent decoder
        that emits segments conditioned on the phonological code.

        Args:
            phon: (B, d_phon) phonological code.
            n_steps: number of decoder steps. Defaults to
                spell_out_max_steps.

        Returns:
            (B, n_steps, n_segments) segment logits per step.
        """
        if not self.cfg.enable_spell_out:
            return torch.zeros(
                phon.shape[0], 0, self.cfg.n_segments,
                device=phon.device, dtype=phon.dtype,
            )

        if n_steps is None:
            n_steps = self.cfg.spell_out_max_steps

        h = self.spell_out_input_proj(phon)
        seg_in = torch.zeros(
            phon.shape[0], self.cfg.n_segments,
            device=phon.device, dtype=phon.dtype,
        )
        outputs = []
        for _ in range(n_steps):
            h = self.spell_out_gru(seg_in, h)
            seg_logits = self.spell_out_output_proj(h)
            outputs.append(seg_logits)
            # Feed teacher-forcing-style argmax back as next input.
            seg_in = torch.zeros_like(seg_logits)
            seg_in.scatter_(
                1, seg_logits.argmax(dim=1, keepdim=True), 1.0,
            )
        return torch.stack(outputs, dim=1)

    # =====================================================================
    # State management
    # =====================================================================

    def reset_state(self) -> None:
        """Clear persistent buffers but preserve registered slot
        texts and learned decoder weights."""
        with torch.no_grad():
            self.a_l_percept.zero_()
            self._spell_out_hidden.zero_()

    def reset_slot_text_registry(self) -> None:
        """Reset the slot-text registry to the reserved-slot
        defaults.

        Used between unrelated dialogue sessions or when the
        runtime needs to discard all acquired-lemma text bindings
        without affecting the substrate's matrices.
        """
        self._slot_text.clear()
        for name, slot_index in SLOT_BY_NAME.items():
            text = PHONOLOGICAL_TEXT_BY_NAME.get(name, "")
            if text:
                self._slot_text[slot_index] = text

    def get_diagnostic_state(self) -> Dict[str, float]:
        """Return a dict of internal norms and counters."""
        with torch.no_grad():
            return {
                "a_l_percept_norm": float(
                    self.a_l_percept.norm().item()
                ),
                "n_registered_slots": len(self._slot_text),
            }
