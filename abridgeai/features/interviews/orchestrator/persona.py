"""Trait-based interviewer persona (value object).

Why this exists
---------------
The interview config stores ``persona`` as one of three string labels
(``strict`` / ``neutral`` / ``supportive``). Those labels drive the spoken
TONE only — never difficulty, scoring, or pass/fail (see
:mod:`...orchestrator.utterance` and the ``adaptive`` convention
"TONE ONLY — never the decision"). This module keeps that contract while
expressing a persona as a small set of INDEPENDENT tone traits instead of an
opaque label, so:

* a teacher can eventually tune warmth / directness without a new enum value
  rippling through 12 lookup sites, and
* the LLM phrasing layer can be handed explicit trait numbers rather than a
  bare word it has to interpret.

Hard boundary
-------------
Every trait here shapes LANGUAGE. None of them may be read by
``decision.py``, ``selection.py``, difficulty targeting, or the rubric. Two
candidates of equal ability MUST receive identical decisions and scores under
different personas — the guardrail tests in
``tests/unit/test_interview_persona_invariants.py`` lock this in.

Backwards compatibility
-----------------------
The three presets reproduce the existing ``strict`` / ``neutral`` /
``supportive`` behaviour exactly. ``PersonaProfile.key`` always resolves to one
of the three legacy :class:`~...orchestrator.utterance.Persona` enum values, so
the deterministic fallback tables (``_ACK`` / ``_TRANSITION``) — which are keyed
by that enum — keep working unchanged. Traits ride ALONGSIDE the label; they do
not replace the label the fallback path depends on.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from enum import Enum

from abridgeai.features.interviews.orchestrator.utterance import Persona, persona_from

# Trait scale bounds. Traits are small integers so they render cleanly into a
# prompt and clamp trivially; the exact ceiling is arbitrary but fixed.
TRAIT_MIN = 0
TRAIT_MAX = 4


class OpeningStyle(str, Enum):  # noqa: UP042 -- match codebase Enum convention
    """How the interviewer opens the session (ceremony phrasing only).

    ``COMFORT`` adds one short put-the-candidate-at-ease line before the first
    question — borrowed from the Sotopia interviewer example. It is a phrasing
    choice on the OPENING turn, not an extra turn and not a timer change.
    """

    BRIEF = "brief"
    STANDARD = "standard"
    COMFORT = "comfort"


def _clamp(value: int) -> int:
    return max(TRAIT_MIN, min(TRAIT_MAX, value))


@dataclass(frozen=True)
class PersonaProfile:
    """A resolved interviewer persona expressed as independent tone traits.

    ``key`` is the legacy label the deterministic fallback tables key on; it is
    always one of ``strict`` / ``neutral`` / ``supportive``. The trait fields
    only influence the LLM phrasing layer.
    """

    key: str
    warmth: int  # affective language density (cold → warm)
    directness: int  # hedged → blunt/imperative
    verbosity: int  # terse → expansive
    formality: int  # informal → formal register
    ack_frequency: int  # how often an acknowledgement is emitted at all
    opening_style: OpeningStyle

    def persona(self) -> Persona:
        """The legacy enum the fallback tables key on."""
        return persona_from(self.key)

    def clamped(self) -> PersonaProfile:
        """Return a copy with every trait clamped into ``[TRAIT_MIN, TRAIT_MAX]``."""
        return replace(
            self,
            warmth=_clamp(self.warmth),
            directness=_clamp(self.directness),
            verbosity=_clamp(self.verbosity),
            formality=_clamp(self.formality),
            ack_frequency=_clamp(self.ack_frequency),
        )

    def as_prompt_traits(self) -> dict[str, object]:
        """Serialise the traits for the utterance LLM prompt.

        Only the tone traits + the preset key. No decision-bearing data ever
        appears here — the phrasing model receives tone guidance, nothing that
        could shift difficulty or scoring.
        """
        return {
            "key": self.key,
            "warmth": self.warmth,
            "directness": self.directness,
            "verbosity": self.verbosity,
            "formality": self.formality,
            "ack_frequency": self.ack_frequency,
            "opening_style": self.opening_style.value,
        }


# ── Presets: reproduce today's three personas exactly ────────────────────────
# The trait numbers are chosen so the LLM guidance matches the tone the existing
# deterministic templates already express (strict = sparse/blunt/formal,
# supportive = warm/gentle/expansive, neutral in the middle).
PRESETS: dict[str, PersonaProfile] = {
    "strict": PersonaProfile(
        key="strict",
        warmth=0,
        directness=4,
        verbosity=1,
        formality=4,
        ack_frequency=1,
        opening_style=OpeningStyle.BRIEF,
    ),
    "neutral": PersonaProfile(
        key="neutral",
        warmth=2,
        directness=2,
        verbosity=2,
        formality=3,
        ack_frequency=2,
        opening_style=OpeningStyle.STANDARD,
    ),
    "supportive": PersonaProfile(
        key="supportive",
        warmth=4,
        directness=1,
        verbosity=3,
        formality=2,
        ack_frequency=3,
        opening_style=OpeningStyle.COMFORT,
    ),
}

_DEFAULT_KEY = "neutral"


def profile_from(value: str | None) -> PersonaProfile:
    """Resolve a persona label to its preset :class:`PersonaProfile`.

    Unknown / None labels fall back to the neutral preset — mirroring
    :func:`...utterance.persona_from`, so the two resolution helpers never
    disagree about what an unrecognised value means.
    """
    key = persona_from(value).value
    return PRESETS.get(key, PRESETS[_DEFAULT_KEY])


# The trait fields that a teacher override may set. ``key`` and the legacy
# label are NOT overridable here — the preset (via ``persona``) always decides
# which deterministic fallback tables key on, so an override can only reshape
# the LLM tone dials, never the scoring-adjacent selector.
_OVERRIDABLE_TRAITS = ("warmth", "directness", "verbosity", "formality", "ack_frequency")


def profile_from_config(
    persona: str | None,
    persona_profile_json: dict | None = None,
) -> PersonaProfile:
    """Resolve the effective persona profile for an interview config (Phase 3).

    Resolution order:

    1. ``persona_profile_json`` present → the ``persona`` preset merged with the
       overrides. Only the known trait keys are read (unknown keys ignored), each
       value must be int-coercible or it is skipped, and every result is clamped
       to ``[TRAIT_MIN, TRAIT_MAX]``. ``opening_style`` may also be overridden
       with a valid :class:`OpeningStyle` value; anything else is ignored.
    2. ``persona`` only → the preset (:func:`profile_from`).
    3. neither → the neutral preset.

    The merge is defensive by construction: a malformed override can never raise
    and never produce an out-of-range trait — worst case it is ignored and the
    preset value stands. Persona is tone-only, so a bad override degrades to
    "slightly different tone", never to a scoring or fairness change.
    """
    base = profile_from(persona)
    if not isinstance(persona_profile_json, dict) or not persona_profile_json:
        return base

    updates: dict[str, object] = {}
    for trait in _OVERRIDABLE_TRAITS:
        if trait not in persona_profile_json:
            continue
        raw = persona_profile_json[trait]
        try:
            updates[trait] = _clamp(int(raw))
        except (TypeError, ValueError):
            continue  # non-numeric override → keep the preset value

    raw_opening = persona_profile_json.get("opening_style")
    if raw_opening is not None:
        # Unknown opening style → keep the preset value. An override that names a
        # style this build does not know must not break persona resolution.
        with contextlib.suppress(ValueError):
            updates["opening_style"] = OpeningStyle(raw_opening)

    if not updates:
        return base
    return replace(base, **updates)


__all__ = [
    "TRAIT_MIN",
    "TRAIT_MAX",
    "OpeningStyle",
    "PersonaProfile",
    "PRESETS",
    "profile_from_config",
    "profile_from",
]
