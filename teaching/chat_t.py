"""
chat_t.py
The Chat Interface: Where a Teacher Teaches and a Substrate Learns

BIOLOGICAL GROUNDING
====================
This file does not model a brain region. It models the conversational
medium through which a teacher and a learner exchange information. In
biology this is the speech channel: the teacher speaks acoustic
waveforms into the air, the learner's auditory cortex transduces them
to phonological codes, the learner produces speech back through the
articulator chain, and the teacher's auditory cortex transduces those
acoustic outputs back to phonological codes. The substrate operates on
the phonological-code stratum at both ends; the chat interface stands
in for the auditory and articulatory transduction that biology
performs and that the v2 text-mode substrate stubs out.

The biological commitment that makes this stub honest is that the
substrate's perception and production are real even though the channel
is text. The instructor types a phrase. The chat interface tokenizes
it, runs it through the frame recognizer (which fires ACh into the
NeuromodulatorBus when a teaching frame matches), runs it through the
identity module (which boosts self_lemma or other_lemma when a pronoun
is present), runs it through Wernicke's perception direction
(which drives lemma activation in mid-MTG), and runs it through the
confirmation detector (which fires dopamine into the bus when a
confirmation pattern matches the substrate's most recent emission).
This is the same sequence the auditory pathway would drive in a
biological system; the stub is just the front-end that maps
characters to phonological codes.

The instructor-facing display surface is what the Genesis Teaching
for Timmy specification calls for: the four-component epistemic
confidence vector, the global maturity scalar, the current ACh
level, the allocation status counts (provisional, confirmed,
unallocated), and the substrate's most recent emission. The
instructor reads these to pace the conversation: when ACh is low
the substrate is in retrieval mode, when ACh is high the substrate
is in encoding mode, when allocation has a pending provisional row
the next thing to say is something that confirms or corrects.
The display is not for the instructor to override the substrate;
it is for the instructor to teach effectively.

The grapheme-to-phoneme stub maps known words through a small lookup
table and unknown words through character-level decomposition. The
known-word lookup matches the test_lexical_substrate convention:
each text string hashes to a deterministic phonological code so the
same word always produces the same code. The character-level fallback
hashes the string itself, which produces stable codes for unknown
words at the cost of slower acquisition (the substrate sees the same
code for the same string but does not benefit from a learned
phonological structure across similar words). Both paths are
sufficient for the cold-start dialogue.

Primary grounding papers:

Hickok G, Poeppel D (2007). "The cortical organization of speech
processing." Nature Reviews Neuroscience, 8(5), 393-402.
DOI: 10.1038/nrn2113. (The dual-stream model under which the chat
interface stands in for the auditory ventral stream and the
articulator stands in for the dorsal-stream output.)

Hasselmo ME (2006). "The role of acetylcholine in learning and memory."
Current Opinion in Neurobiology, 16(6), 710-715.
DOI: 10.1016/j.conb.2006.09.002. (The encoding-versus-retrieval
modulation that the frame recognizer drives through ACh, and that
the chat display surfaces for the instructor.)

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor


# =========================================================================
# Configuration
# =========================================================================

@dataclass
class ChatConfig:
    """Configuration for the chat interface.

    Attributes:
        enable_chat: master flag.
        show_diagnostics: when True, the format_state() method renders
            the instructor-facing diagnostic panel alongside the
            conversation. When False, only the conversation lines are
            shown. Useful for end-user-facing deployments versus
            instructor-mode teaching sessions.
        d_phon: phonological-code dimensionality. Must match the
            substrate's d_phon. Used by the grapheme-to-phoneme stub
            to produce codes of the correct shape.
        n_concepts: concept-space dimensionality. Used to construct
            concept vectors with frame-bias contributions for
            allocation events.
        ach_decay_per_turn: how much ACh fades between turns when no
            teaching frame fires. Models the leak that Hasselmo 2006
            documents at the time-scale of conversational turns.
            Default 0.7.
        ach_resting_level: the level ACh decays toward in the absence
            of frame matches. Default 0.5 (the bus's neutral value).
        emission_history_length: how many recent substrate emissions
            to retain for the confirmation detector's repeat-match
            pattern.
    """

    enable_chat: bool = True
    show_diagnostics: bool = True
    d_phon: int = 64
    n_concepts: int = 1024
    ach_decay_per_turn: float = 0.7
    ach_resting_level: float = 0.5
    emission_history_length: int = 8


# =========================================================================
# Grapheme-to-phoneme stub
# =========================================================================

def _hash_to_tensor(text: str, dim: int) -> Tensor:
    """Map a text string to a deterministic unit-norm tensor.

    Same text always produces the same tensor; different texts
    produce nearly orthogonal tensors in expectation. The hash is
    Python's built-in hash, seeded into a generator that produces a
    Gaussian random vector. The vector is unit-normalized so the
    phonological code has consistent magnitude across words.

    NOT a biological quantity. Engineering convenience for the
    text-mode chat interface; replaced by a real grapheme-to-phoneme
    pipeline when speech I/O comes online.

    Args:
        text: input string.
        dim: target dimensionality.

    Returns:
        (dim,) unit-norm Tensor.
    """
    if not text:
        return torch.zeros(dim)
    seed = abs(hash(text)) % (2 ** 31)
    gen = torch.Generator()
    gen.manual_seed(seed)
    v = torch.randn(dim, generator=gen)
    return v / (v.norm() + 1e-9)


def text_to_phonological_code(text: str, d_phon: int) -> Tensor:
    """Stub grapheme-to-phoneme: map a single word to its phonological
    code.

    Uses the same hash-based encoding for all words, so the substrate
    sees a stable code per word but does not benefit from learned
    phonological structure across similar words. Sufficient for the
    cold-start dialogue and for early teaching sessions; the real
    grapheme-to-phoneme pipeline replaces this when phonological
    structure becomes part of the substrate's input.

    Args:
        text: the word to encode.
        d_phon: phonological-code dimensionality.

    Returns:
        (d_phon,) phonological code tensor.
    """
    return _hash_to_tensor(text.lower(), d_phon)


def text_to_concept(text: str, n_concepts: int) -> Tensor:
    """Stub conceptual encoder: map a word to a concept vector.

    The first 1000 dimensions are derived from the word's hash so
    different words produce different concept vectors. Dimensions
    1000 through 1023 are reserved for frame-bias and uncertainty
    flags that the chat interface fills in based on the recognized
    teaching frame.

    Args:
        text: the word to encode.
        n_concepts: concept-space dimensionality.

    Returns:
        (n_concepts,) concept vector.
    """
    base = _hash_to_tensor(text.lower(), n_concepts)
    # Zero out the reserved tail (frame-bias and uncertainty
    # subspaces). The frame recognizer fills these dimensions through
    # the bias_dim_index returned by recognize_and_get_bias.
    base[1000:] = 0.0
    return base


# =========================================================================
# Turn record
# =========================================================================

@dataclass
class TurnRecord:
    """One full turn of the conversation: instructor input plus any
    substrate emission that followed.

    Attributes:
        instructor_text: the raw text the instructor typed. Empty
            when this is a substrate-initiated turn (e.g., a
            spontaneous question).
        instructor_tokens: the tokenized instructor input.
        substrate_emission: the text the substrate emitted, or None
            if the substrate did not produce a turn (e.g., the
            instructor's input did not require a response).
        recognized_frame: the name of any teaching frame that matched,
            or None.
        confirmation_event: the polarity of any confirmation event
            that fired, or None.
        diagnostics: snapshot of the substrate's diagnostic state
            after the turn completes.
    """

    instructor_text: str = ""
    instructor_tokens: List[str] = field(default_factory=list)
    substrate_emission: Optional[str] = None
    recognized_frame: Optional[str] = None
    confirmation_event: Optional[int] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# =========================================================================
# The Chat interface
# =========================================================================

class Chat:
    """The instructor-facing chat interface.

    BIOLOGICAL FUNCTION: routes typed input through the substrate's
    perception-side machinery (frame recognizer, identity module,
    Wernicke's, confirmation detector), drives one production turn
    when appropriate, and surfaces the substrate's internal state
    for the instructor to read. Holds a small history of recent
    emissions so the confirmation detector's repeat-match pattern
    can fire on yes_repeat events.

    INTERFACE CONTRACT:
        Inputs:
            handle_input(text): one full turn driven by instructor
                input. Routes through perception, optionally drives
                a production turn, returns a TurnRecord.

        Outputs:
            format_state(): one-line summary of the substrate's
                internal state for the instructor display.
            history(): list of TurnRecord, in conversation order.

        State: list of recent emissions for the confirmation
            detector. Conversation history. The substrate state
            lives in the substrate components and does not duplicate
            here.
    """

    def __init__(
        self,
        cfg: ChatConfig,
        lexical_substrate: Any,
        mid_mtg: Any,
        wernicke: Any,
        lemma_acquisition: Any,
        frame_recognizer: Any,
        confirmation_detector: Any,
        identity_module: Optional[Any] = None,
        epistemic_monitor: Optional[Any] = None,
        neuromodulator_bus: Optional[Any] = None,
        spell_out_consumer: Optional[Any] = None,
    ) -> None:
        """Initialize the chat interface against a fully-wired
        substrate.

        Args:
            cfg: ChatConfig.
            lexical_substrate: the LexicalSubstrate parent.
            mid_mtg: the MidMTG region.
            wernicke: the Wernicke region.
            lemma_acquisition: the LemmaAcquisitionModule.
            frame_recognizer: the TeachingFrameRecognizer.
            confirmation_detector: the ConfirmationDetector.
            identity_module: optional IdentityModule for pronoun
                routing. When None, pronouns are not routed (only
                the substrate's natural lemma activation runs).
            epistemic_monitor: optional EpistemicMonitor for the
                confidence-band display. When None, the diagnostic
                panel omits the confidence vector.
            neuromodulator_bus: optional bus to fire ACh and
                dopamine into. When None, the chat interface still
                runs but the modulator events are no-ops.
            spell_out_consumer: optional SpellOutConsumer for
                streaming production output through the runtime.
                When None, spell-out happens synchronously in the
                handle_input call.
        """
        self.cfg = cfg
        self.substrate = lexical_substrate
        self.mid_mtg = mid_mtg
        self.wernicke = wernicke
        self.acquisition = lemma_acquisition
        self.frame_recognizer = frame_recognizer
        self.confirmation_detector = confirmation_detector
        self.identity_module = identity_module
        self.epistemic_monitor = epistemic_monitor
        self.bus = neuromodulator_bus
        self.spell_out_consumer = spell_out_consumer

        self._history: List[TurnRecord] = []
        # Production-idle flag, read by LemmaDecayConsumer to know
        # whether to apply gamma_lemma decay outside the production
        # forward pass.
        self._production_idle: bool = True

    # ---------------------------------------------------------------
    # Production-idle accessor (used by LemmaDecayConsumer)
    # ---------------------------------------------------------------

    def is_production_idle(self) -> bool:
        return self._production_idle

    # ---------------------------------------------------------------
    # Tokenization
    # ---------------------------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, strip punctuation, split on whitespace.

        Matches the convention the frame_recognizer and
        confirmation_detector tokenize their pattern lists with so
        the patterns and the input use the same token form.
        """
        # Map common contractions and apostrophes to a stable form.
        normalized = text.lower()
        # Preserve the apostrophe inside words like "that's" because
        # the confirmation patterns include those literal forms.
        # Strip leading and trailing punctuation only.
        for ch in [".", ",", "!", "?", ":", ";"]:
            normalized = normalized.replace(ch, " ")
        return [t for t in normalized.split() if t]

    # ---------------------------------------------------------------
    # Perception pipeline
    # ---------------------------------------------------------------

    def _decay_ach(self) -> None:
        """Apply per-turn ACh decay so the bus drifts back to resting
        level when no teaching frame fires.

        Models the leak documented in Hasselmo 2006: ACh elevation is
        phasic in response to encoding-relevant events, and absent
        such events the level returns to baseline. The decay runs at
        the start of every turn before the frame recognizer fires
        any new ACh, so a frame match reaches the bus as a phasic
        elevation rather than as a saturated level.
        """
        if self.bus is None:
            return
        try:
            current = self.bus.get("ACh_inc")
            if hasattr(current, "item"):
                current = current.item()
            current = float(current)
        except (KeyError, AttributeError):
            return
        rest = self.cfg.ach_resting_level
        decay = self.cfg.ach_decay_per_turn
        new_level = rest + decay * (current - rest)
        try:
            self.bus.set("ACh_inc", new_level)
        except (KeyError, AttributeError):
            pass

    def _drive_perception(
        self,
        tokens: List[str],
    ) -> Tuple[Optional[str], Optional[Tensor]]:
        """Run frame recognition and return (frame_name, frame_bias).

        The frame_bias is the one-hot concept-space contribution that
        the frame recognizer flagged for this turn. It will be added
        to the concept vector at allocation time so the lemma
        acquisition module can distinguish a naming frame from a
        vocabulary frame.

        Side effect: fires ACh into the bus at the amplitude the
        recognizer reports.
        """
        result = None
        if self.frame_recognizer is None:
            return None, None
        try:
            result = self.frame_recognizer.recognize_and_get_bias(tokens)
        except Exception:
            return None, None
        if result is None or not result.recognized:
            return None, None
        # Fire ACh.
        if self.bus is not None and result.ach_amplitude > 0.0:
            try:
                # Take the max of current ACh and the new amplitude so
                # successive frame matches do not lower a previously
                # higher ACh.
                current = self.bus.get("ACh_inc")
                if hasattr(current, "item"):
                    current = current.item()
                new_level = max(float(current), float(result.ach_amplitude))
                self.bus.set("ACh_inc", new_level)
            except (KeyError, AttributeError):
                pass
        # Build the bias tensor.
        bias = None
        if result.bias_dim_index is not None:
            bias = torch.zeros(self.cfg.n_concepts)
            bias[result.bias_dim_index] = 1.0
        return result.frame_name, bias

    def _route_pronouns(self, tokens: List[str]) -> None:
        """Apply identity-routing to any pronouns in the token stream.

        Boosts self_lemma or other_lemma in mid-MTG's persistent
        activation as a side effect. No return value; the activation
        change is what subsequent perception steps read.
        """
        if self.identity_module is None:
            return
        self.identity_module.route_perceived_phrase(tokens, self.mid_mtg)

    def _perceive_phonological_codes(self, tokens: List[str]) -> None:
        """Drive Wernicke's perception direction with the phonological
        code for each token.

        Each token's hash-based phonological code flows into
        Wernicke's perceive method, which projects backward through
        W_L_to_P (perception direction of the tied substrate) into
        lemma activation in mid-MTG.
        """
        for tok in tokens:
            phon = text_to_phonological_code(tok, self.cfg.d_phon)
            self.wernicke.perceive_phonological_code(phon)

    def _detect_confirmation(
        self,
        tokens: List[str],
    ) -> Optional[int]:
        """Run confirmation detection and fire dopamine if a pattern
        matches.

        The detector reads its own most-recent emission record,
        compares against the input tokens, fires dopamine of the
        appropriate amplitude into the bus when a positive or
        negative confirmation pattern matches, and either confirms
        or decays the relevant provisional lemma row.

        Returns:
            polarity (+1 or -1) of the matched pattern, or None if
                no pattern matched.
        """
        if self.confirmation_detector is None:
            return None
        try:
            result = self.confirmation_detector.detect_confirmation(
                tokens, bus=self.bus, acquisition=self.acquisition,
            )
        except Exception:
            return None
        if result is None or not result.matched:
            return None
        return result.polarity

    # ---------------------------------------------------------------
    # Production pipeline
    # ---------------------------------------------------------------

    def _drive_production(
        self,
        frame_bias: Optional[Tensor],
    ) -> Optional[str]:
        """Run one production turn and return the emitted text.

        Reads the current concept vector from mid-MTG (whatever is
        active in the persistent buffer after perception), applies
        the frame bias if any, calls
        lemma_acquisition.select_lemma_for_production to pick a
        slot and a polar-question flag, and routes through
        Wernicke's spell-out (or the spell_out_consumer's queue if
        one is provided) to render the text.

        Returns the emitted text, or None if production fails (no
        allocated lemma scores above theta_production).
        """
        # Build the current concept vector. The mid-MTG persistent
        # a_lemma is in lemma space; we need to project back to
        # concept space through the comprehension direction to get
        # the substrate's current conceptual state.
        with torch.no_grad():
            concept = self.mid_mtg.forward_comprehension(
                self.mid_mtg.a_lemma
            ).squeeze(0)
        if frame_bias is not None:
            concept = concept + frame_bias

        slot_index, polar_q = self.acquisition.select_lemma_for_production(
            concept,
        )
        if slot_index < 0:
            return None

        self._production_idle = False
        try:
            if polar_q:
                text = self.wernicke.spell_out_with_polar_question(slot_index)
            else:
                text = self.wernicke.spell_out_for_slot(slot_index)
        finally:
            self._production_idle = True

        # Record the emission for the confirmation detector's repeat
        # pattern.
        if self.confirmation_detector is not None:
            try:
                self.confirmation_detector.record_emission(text, slot_index)
            except (AttributeError, TypeError):
                pass

        return text

    # ---------------------------------------------------------------
    # Allocation pipeline
    # ---------------------------------------------------------------

    def _maybe_allocate_novel(
        self,
        tokens: List[str],
        frame_bias: Optional[Tensor],
    ) -> Optional[int]:
        """Check whether the input contains a novel phonological code
        that should trigger an allocation, and if so allocate it.

        For each token, computes its phonological code and asks the
        acquisition module whether the code is novel. If novel and
        the current ACh level is high enough (encoding mode), runs
        allocate_row with the token's concept vector plus the frame
        bias and returns the new slot index.

        Returns the slot index of the most recently allocated row,
        or None if no allocation happened.
        """
        if self.acquisition is None:
            return None
        last_allocated: Optional[int] = None
        for tok in tokens:
            phon = text_to_phonological_code(tok, self.cfg.d_phon)
            # Skip tokens that are already in the reserved slot table
            # (they are pre-confirmed at construction).
            if not self.acquisition.is_novel(phon):
                continue
            # Build the concept vector for this token.
            concept = text_to_concept(tok, self.cfg.n_concepts)
            if frame_bias is not None:
                concept = concept + frame_bias
            # Bind the slot's text in Wernicke's so spell-out
            # can render it later.
            slot = self.acquisition.allocate_row(concept, phon)
            if slot >= 0:
                self.wernicke.register_slot_text(slot, tok)
                last_allocated = slot
        return last_allocated

    # ---------------------------------------------------------------
    # Top-level turn handler
    # ---------------------------------------------------------------

    def handle_input(self, text: str) -> TurnRecord:
        """Route one full instructor turn through the substrate.

        The order of operations matters:

            1. Decay ACh from any previous frame elevation so a fresh
               match reads as phasic.
            2. Tokenize the input.
            3. Run frame recognition (fires ACh, returns frame name
               and bias).
            4. Route pronouns through the identity module (boosts
               self_lemma or other_lemma in mid-MTG).
            5. Run confirmation detection (fires dopamine, transitions
               provisional rows to confirmed).
            6. Drive Wernicke's perception with each token's
               phonological code (drives lemma activation in mid-MTG).
            7. Optionally allocate any novel tokens (provisional rows
               in lemma_acquisition).
            8. Drive one production turn through the lemma_acquisition
               selector and Wernicke's spell-out.
            9. Snapshot diagnostics for the turn record.

        Args:
            text: the instructor's typed input.

        Returns:
            TurnRecord describing the turn.
        """
        if not self.cfg.enable_chat:
            return TurnRecord(instructor_text=text)

        self._decay_ach()
        tokens = self._tokenize(text)

        frame_name, frame_bias = self._drive_perception(tokens)
        self._route_pronouns(tokens)
        confirm_polarity = self._detect_confirmation(tokens)
        self._perceive_phonological_codes(tokens)

        # Allocation runs after perception so that mid-MTG's lemma
        # activation reflects the just-received input. The
        # acquisition module reads from the substrate's current
        # state when it allocates.
        self._maybe_allocate_novel(tokens, frame_bias)

        emission = self._drive_production(frame_bias)

        record = TurnRecord(
            instructor_text=text,
            instructor_tokens=tokens,
            substrate_emission=emission,
            recognized_frame=frame_name,
            confirmation_event=confirm_polarity,
            diagnostics=self._snapshot_diagnostics(),
        )
        self._history.append(record)
        return record

    # ---------------------------------------------------------------
    # Diagnostic snapshot for the instructor display
    # ---------------------------------------------------------------

    def _snapshot_diagnostics(self) -> Dict[str, Any]:
        """Capture the substrate's internal state for the instructor
        panel. Per the Genesis Teaching for Timmy specification, the
        panel surfaces ACh, allocation status counts, and (when
        available) the four-component confidence vector.
        """
        diag: Dict[str, Any] = {}
        if self.bus is not None:
            try:
                ach = self.bus.get("ACh_inc")
                if hasattr(ach, "item"):
                    ach = ach.item()
                diag["ach_inc"] = float(ach)
            except (KeyError, AttributeError):
                pass
            try:
                maturity = self.bus.get("global_maturity")
                if hasattr(maturity, "item"):
                    maturity = maturity.item()
                diag["global_maturity"] = float(maturity)
            except (KeyError, AttributeError):
                pass
        if self.acquisition is not None:
            try:
                diag["allocation"] = self.acquisition.get_diagnostic_state()
            except Exception:
                pass
        if self.epistemic_monitor is not None:
            try:
                # The monitor needs a one-hot lemma indicator. Use
                # the current peak.
                with torch.no_grad():
                    peak_idx = self.mid_mtg.a_lemma.abs().argmax(dim=1)
                lemma_one_hot = torch.zeros_like(self.mid_mtg.a_lemma)
                lemma_one_hot[0, peak_idx.item()] = 1.0
                report = self.epistemic_monitor.compute_confidence(
                    self.mid_mtg, self.wernicke, lemma_one_hot,
                    neuromod_bus=self.bus,
                )
                diag["confidence"] = {
                    "aggregate": report.aggregate,
                    "register": report.register.value,
                    "lemma": report.lemma_confidence,
                    "phonological": report.phonological_confidence,
                }
            except Exception:
                pass
        return diag

    # ---------------------------------------------------------------
    # Display helpers
    # ---------------------------------------------------------------

    def format_state(self) -> str:
        """Render a one-line summary of the substrate's current state.

        Used by the runner script to display the instructor panel
        alongside each turn. When show_diagnostics is False this
        returns an empty string.

        Returns:
            human-readable state line, or empty string if diagnostics
                are disabled.
        """
        if not self.cfg.show_diagnostics:
            return ""
        diag = self._snapshot_diagnostics()
        parts: List[str] = []
        if "ach_inc" in diag:
            parts.append(f"ACh={diag['ach_inc']:.2f}")
        if "global_maturity" in diag:
            parts.append(f"maturity={diag['global_maturity']:.2f}")
        if "confidence" in diag:
            c = diag["confidence"]
            parts.append(f"conf={c['aggregate']:.2f}({c['register']})")
        if "allocation" in diag:
            a = diag["allocation"]
            parts.append(
                f"allocated={a.get('n_provisional', 0)}P+"
                f"{a.get('n_confirmed', 0)}C"
            )
        return " ".join(parts)

    def history(self) -> List[TurnRecord]:
        """Return the conversation history in order."""
        return list(self._history)

    def reset(self) -> None:
        """Reset the chat history. Does not reset the substrate."""
        self._history.clear()
