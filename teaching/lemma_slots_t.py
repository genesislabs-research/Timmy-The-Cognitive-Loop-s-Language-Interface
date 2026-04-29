"""
lemma_slots_t.py
Canonical Reserved-Slot Inventory for the Speech Substrate

BIOLOGICAL GROUNDING
====================
This file defines the inventory of pre-allocated lemma slots in the
v2 speech pathway. The biological commitment is that certain lemma
content is part of the language's grammatical scaffolding rather than
acquirable from positive examples alone, and that certain epistemic
content is part of the substrate's honest reporting register from cold
start. Both kinds of content are realized as lemma slots that exist at
substrate construction with allocation_status = STATUS_CONFIRMED, so
that they fire reliably during the cold-start dialogue and during all
subsequent phases.

The inventory is divided into three groups by function. The identity
group holds the self-and-other distinction that pronoun routing
depends on. The question group holds the wh-words plus the
polar-question marker that the production pathway uses to form
interrogatives. The uncertainty group holds the eight epistemic
register markers that the production fall-through path uses when no
allocated lemma exceeds the production threshold.

The contiguous slot ordering is deliberate. Reserved slots occupy
indices 0 through 16 inclusive; acquired lemmas start at index 17.
Numbered slot constants make the cold-start dialogue's polar-question
co-activation testable without dictionary lookups, and the contiguous
range makes RESERVED_SLOTS = 17 a simple loop boundary in
find_free_slot.

The architect's concept-subspace reservation is orthogonal to this
slot reservation and lives at concept-space dimensions 1000 through
1023. The frame-bias subspace at 1000-1015 and the uncertainty
subspace at 1016-1023 are reserved in the conceptual stratum, not in
the lemma stratum, and are referenced by the frame_recognizer and the
uncertainty_router respectively. The two reservations exist
independently and serve different purposes.

The grammatical-scaffolding rationale for hardcoding identity routing
is that pronouns cannot be learned from positive examples without
already understanding the speaker-addressee distinction; the rationale
for hardcoding wh-words is that they similarly carry abstract
question-type semantics that depend on the language's grammatical
machinery rather than on accumulated experience; the rationale for
hardcoding the uncertainty markers is that an architecture without
honest "I don't know" fall-through cannot be honest at all, and the
production register has to land somewhere when nothing is known.

Primary grounding papers:

Goddard CL, Wierzbicka A, eds. (2002). "Meaning and Universal Grammar:
Theory and Empirical Findings." John Benjamins.
DOI: 10.1075/slcs.60. (NSM semantic primes account for the wh-words
as conceptually irreducible.)

Tomasello M (2003). "Constructing a Language: A Usage-Based Theory of
Language Acquisition." Harvard University Press.
DOI: 10.4159/9780674017641. (Pronoun acquisition requires prior
understanding of speaker-addressee roles.)

Tulving E (1985). "Memory and consciousness." Canadian Psychology,
26(1), 1-12. DOI: 10.1037/h0080017. (Epistemic register distinctions
in metamemory.)

Genesis Labs Research, April 2026.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional


# =========================================================================
# Reserved slot indices
# =========================================================================
#
# The full inventory. Each index is the canonical slot for that lemma in
# both W_C_to_L (row index) and W_L_to_P (column index). The naming uses
# snake_case strings so that runtime code can refer to slots by name
# through SLOT_BY_NAME or by index through the named constants.

# Identity group. Slots 0 through 1.
SLOT_SELF_LEMMA: int = 0
SLOT_OTHER_LEMMA: int = 1

# Wh-word group. Slots 2 through 7.
SLOT_WHAT: int = 2
SLOT_WHO: int = 3
SLOT_WHERE: int = 4
SLOT_WHEN: int = 5
SLOT_WHY: int = 6
SLOT_HOW: int = 7

# Polar-question marker. Slot 8. Distinct from the wh-words because it
# carries no conceptual content of its own; it is a syntactic marker
# that switches the production pathway into rising-intonation question
# form.
SLOT_POLAR_QUESTION: int = 8

# Uncertainty register group. Slots 9 through 16. Each register
# corresponds to a distinct epistemic state that the production
# fall-through can land on, with i_dont_know as the canonical
# zero-knowledge fallback for the cold-start dialogue.
SLOT_I_DONT_KNOW: int = 9
SLOT_IM_NOT_SURE: int = 10
SLOT_MAYBE: int = 11
SLOT_I_THINK: int = 12
SLOT_PROBABLY: int = 13
SLOT_I_DONT_REMEMBER: int = 14
SLOT_NO_IDEA: int = 15
SLOT_CANNOT_SAY: int = 16

# Total number of reserved slots. Acquired lemmas start at this index.
RESERVED_SLOTS: int = 17


# =========================================================================
# Filtered views
# =========================================================================
#
# The three named subgroups, each as a dict mapping name to slot index.
# Modules that need to iterate over a subgroup (e.g., the identity
# module iterating over identity slots, or the uncertainty router
# iterating over uncertainty slots) read these views rather than the
# full table. The views are read-only at the convention level; the
# canonical source of truth is SLOT_BY_NAME below.

IDENTITY_LEMMA_SLOTS: Dict[str, int] = {
    "self_lemma": SLOT_SELF_LEMMA,
    "other_lemma": SLOT_OTHER_LEMMA,
}

QUESTION_LEMMA_SLOTS: Dict[str, int] = {
    "what": SLOT_WHAT,
    "who": SLOT_WHO,
    "where": SLOT_WHERE,
    "when": SLOT_WHEN,
    "why": SLOT_WHY,
    "how": SLOT_HOW,
    "polar_question": SLOT_POLAR_QUESTION,
}

UNCERTAINTY_LEMMA_SLOTS: Dict[str, int] = {
    "i_dont_know": SLOT_I_DONT_KNOW,
    "im_not_sure": SLOT_IM_NOT_SURE,
    "maybe": SLOT_MAYBE,
    "i_think": SLOT_I_THINK,
    "probably": SLOT_PROBABLY,
    "i_dont_remember": SLOT_I_DONT_REMEMBER,
    "no_idea": SLOT_NO_IDEA,
    "cannot_say": SLOT_CANNOT_SAY,
}


# =========================================================================
# Canonical lookup table
# =========================================================================

SLOT_BY_NAME: Dict[str, int] = {
    **IDENTITY_LEMMA_SLOTS,
    **QUESTION_LEMMA_SLOTS,
    **UNCERTAINTY_LEMMA_SLOTS,
}

# Reverse lookup. Built once at import time.
NAME_BY_SLOT: Dict[int, str] = {
    slot: name for name, slot in SLOT_BY_NAME.items()
}

# All reserved slot indices as a frozenset, useful for membership tests
# in find_free_slot and similar code paths.
RESERVED_SLOT_INDICES: FrozenSet[int] = frozenset(SLOT_BY_NAME.values())


# =========================================================================
# Uncertainty register metadata
# =========================================================================
#
# Each uncertainty marker carries a register tag that identifies which
# epistemic state it expresses. The uncertainty_router consults this
# metadata to select the appropriate marker given the substrate's
# current confidence vector. The tags are fixed at construction; they
# are not mutable. The architect's specification of the eight registers
# lives in question 1 of the Phase 3 four-item handoff.

UNCERTAINTY_REGISTER_TAGS: Dict[str, str] = {
    "i_dont_know": "floor",
    "im_not_sure": "low",
    "maybe": "hedged",
    "i_think": "tentative",
    "probably": "leaning",
    "i_dont_remember": "decayed",
    "no_idea": "floor_strong",
    "cannot_say": "refuse",
}


# =========================================================================
# Phonological code text (for the construction-time hash-based stub)
# =========================================================================
#
# Each reserved lemma needs a phonological code at construction. The
# text strings here feed into the existing word_to_phonological_code
# stub used elsewhere in the speech repo. The runtime constructs the
# actual phonological code tensor by hashing the text and unpacking
# into d_phon dimensions, which is the same procedure that
# Phase 3 acquisition uses for novel lemmas. Identity slots have an
# empty string because their phonological code is acquired at runtime
# (TIMMY for self_lemma during the cold-start naming dialogue, and
# whatever name the instructor goes by for other_lemma in subsequent
# sessions).

PHONOLOGICAL_TEXT_BY_NAME: Dict[str, str] = {
    "self_lemma": "",
    "other_lemma": "",
    "what": "what",
    "who": "who",
    "where": "where",
    "when": "when",
    "why": "why",
    "how": "how",
    "polar_question": "",
    "i_dont_know": "i don't know",
    "im_not_sure": "i'm not sure",
    "maybe": "maybe",
    "i_think": "i think",
    "probably": "probably",
    "i_dont_remember": "i don't remember",
    "no_idea": "no idea",
    "cannot_say": "i cannot say",
}


# =========================================================================
# API helpers
# =========================================================================

def slot_for(name: str) -> int:
    """Return the slot index for a reserved lemma name.

    Args:
        name: one of the reserved lemma names (self_lemma,
            other_lemma, what, who, where, when, why, how,
            polar_question, i_dont_know, im_not_sure, maybe,
            i_think, probably, i_dont_remember, no_idea,
            cannot_say).

    Returns:
        the canonical slot index for the named lemma.

    Raises:
        KeyError: if the name is not a reserved lemma.
    """
    if name not in SLOT_BY_NAME:
        raise KeyError(
            f"Unknown reserved lemma name '{name}'. "
            f"Expected one of {sorted(SLOT_BY_NAME.keys())}."
        )
    return SLOT_BY_NAME[name]


def name_for(slot_index: int) -> Optional[str]:
    """Return the reserved lemma name for a slot index, or None if
    the slot is not reserved.

    Args:
        slot_index: lemma slot index in [0, RESERVED_SLOTS) or beyond.

    Returns:
        the lemma name if slot_index is a reserved slot, None
            otherwise.
    """
    return NAME_BY_SLOT.get(slot_index)


def is_reserved(slot_index: int) -> bool:
    """Return True if the given slot index is in the reserved range.

    The reserved range is [0, RESERVED_SLOTS). Slots in this range
    are pre-allocated at construction and cannot be reused for
    runtime acquisition.

    Args:
        slot_index: lemma slot index.

    Returns:
        True if 0 <= slot_index < RESERVED_SLOTS.
    """
    return 0 <= slot_index < RESERVED_SLOTS


def is_uncertainty_marker(slot_index: int) -> bool:
    """Return True if the slot is one of the uncertainty markers.

    Used by the production fall-through to distinguish uncertainty
    emissions (which suppress polar-question co-activation regardless
    of provisional status) from regular lemma emissions.

    Args:
        slot_index: lemma slot index.

    Returns:
        True if slot_index is in UNCERTAINTY_LEMMA_SLOTS.values().
    """
    return slot_index in UNCERTAINTY_LEMMA_SLOTS.values()


def is_identity_marker(slot_index: int) -> bool:
    """Return True if the slot is one of the identity markers.

    Used by the identity module to detect when a perceived pronoun
    has activated an identity slot above threshold.

    Args:
        slot_index: lemma slot index.

    Returns:
        True if slot_index is in IDENTITY_LEMMA_SLOTS.values().
    """
    return slot_index in IDENTITY_LEMMA_SLOTS.values()


def is_question_marker(slot_index: int) -> bool:
    """Return True if the slot is one of the question primes or the
    polar-question marker.

    Args:
        slot_index: lemma slot index.

    Returns:
        True if slot_index is in QUESTION_LEMMA_SLOTS.values().
    """
    return slot_index in QUESTION_LEMMA_SLOTS.values()


# =========================================================================
# Status constants for the lemma acquisition module
# =========================================================================
#
# Re-exported here so that consumers of the slot table can refer to
# allocation status without needing to import lemma_acquisition_t. The
# numerical values are part of the public contract; changing them
# requires updating every consumer.

STATUS_UNALLOCATED: int = 0
STATUS_PROVISIONAL: int = 1
STATUS_CONFIRMED: int = 2
