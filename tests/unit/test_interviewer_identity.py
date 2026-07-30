"""Unit tests for the interviewer identity feature.

Two tests here are load-bearing rather than incidental:

* :func:`test_default_opening_is_unchanged_from_before_the_feature` pins that a
  config which has not opted in renders the pre-feature sentence byte for byte.
  Identity is a presentation change on the most-read turn of the interview, so
  "existing sessions are untouched" has to be verifiable, not asserted.
* :func:`test_every_named_identity_still_discloses_the_ai` pins the ethical
  constraint. Giving the interviewer a human name is only acceptable while the
  candidate is still told they are talking to an AI, and that is exactly the
  property a future wording tweak could silently drop.
"""

from __future__ import annotations

import pytest

from abridgeai.features.interviews.orchestrator.interviewer_identity import (
    PRESETS,
    InterviewerRole,
    as_prompt_identity,
    identity_from,
    identity_from_config,
)
from abridgeai.features.interviews.orchestrator.persona import OpeningStyle
from abridgeai.features.interviews.services.ceremony import opening_text, room_intro_text

# The exact sentences shipped before identity existed. Written out rather than
# derived so a change to the production string is a visible diff here.
_PRE_FEATURE_EN = (
    "Hi, An. It’s nice to meet you. I’m aBridgeAI’s virtual interview "
    "assistant, and I’ll guide your “Data Structures” technical course interview "
    "today. First, can you confirm that I’m speaking with An?"
)
_PRE_FEATURE_VI = (
    "Xin chào An. Rất vui được gặp bạn. Tôi là trợ lý phỏng vấn ảo của "
    "aBridgeAI và sẽ hướng dẫn buổi phỏng vấn kỹ thuật “Data Structures” hôm nay. "
    "Trước tiên, bạn có thể xác nhận tôi đang trao đổi với bạn An không?"
)

_NAMED_ROLES = [r for r in InterviewerRole if r is not InterviewerRole.GENERIC_ASSISTANT]


def _open(**over: object) -> str:
    kwargs: dict[str, object] = {
        "title": "Data Structures",
        "name": "An",
        "persona": "neutral",
        "language": "en",
    }
    kwargs.update(over)
    return opening_text(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("language", "expected"), [("en", _PRE_FEATURE_EN), ("vi", _PRE_FEATURE_VI)]
)
def test_default_opening_is_unchanged_from_before_the_feature(language: str, expected: str) -> None:
    """No identity chosen → the original sentence, byte for byte."""
    assert _open(language=language) == expected


@pytest.mark.parametrize("persona", ["strict", "neutral", "supportive"])
@pytest.mark.parametrize("style", list(OpeningStyle))
def test_opening_style_does_nothing_without_an_identity(persona: str, style: OpeningStyle) -> None:
    """Activating OpeningStyle must not change configs that did not opt in.

    ``strict`` presets to BRIEF and ``supportive`` to COMFORT, so without this
    gate simply shipping the feature would have reworded every existing
    interview's first turn.
    """
    assert _open(persona=persona, opening_style=style) == _PRE_FEATURE_EN


@pytest.mark.parametrize("role", _NAMED_ROLES, ids=lambda r: r.value)
@pytest.mark.parametrize("language", ["en", "vi"])
def test_every_named_identity_still_discloses_the_ai(role: InterviewerRole, language: str) -> None:
    """A human name never displaces the AI disclosure.

    Presenting an AI interviewer as a person would be a consent problem, and the
    research on AI-conducted interviews ties misleading candidates about the
    system's nature to worse perceived fairness and less honest answers.
    """
    identity = identity_from(role.value)
    text = _open(language=language, identity=identity)

    assert identity.name is not None
    assert identity.name in text
    assert identity.title(language) in text
    # "AI interviewer" / "người phỏng vấn AI", plus the platform name.
    assert "AI" in text
    assert "aBridgeAI" in text


@pytest.mark.parametrize("language", ["en", "vi"])
def test_comfort_adds_a_line_and_brief_drops_one(language: str) -> None:
    identity = identity_from(InterviewerRole.BACKEND_TECH_LEAD.value)
    brief = _open(language=language, identity=identity, opening_style=OpeningStyle.BRIEF)
    standard = _open(language=language, identity=identity, opening_style=OpeningStyle.STANDARD)
    comfort = _open(language=language, identity=identity, opening_style=OpeningStyle.COMFORT)

    assert len(brief) < len(standard) < len(comfort)
    # The question being asked is identical in all three; only ceremony differs.
    for text in (brief, standard, comfort):
        assert text.endswith("?")


def test_room_intro_is_absent_by_default_and_present_when_named() -> None:
    """The voice gap: today a voice candidate's first audio is a bank question."""
    assert room_intro_text(identity=identity_from(None), language="en") is None
    named = room_intro_text(
        identity=identity_from(InterviewerRole.STAFF_ENGINEER.value), language="en"
    )
    assert named is not None
    assert "Quân" in named


@pytest.mark.parametrize(
    "blob",
    [None, {}, "not-a-dict", {"interviewer_role": "ceo"}, {"interviewer_role": 42}],
)
def test_resolution_never_raises_and_degrades_to_the_default(blob: object) -> None:
    """A malformed blob must fall back, never blow up a session start."""
    assert identity_from_config(blob).role is InterviewerRole.GENERIC_ASSISTANT  # type: ignore[arg-type]


def test_prompt_identity_carries_no_decision_bearing_keys() -> None:
    """Mirrors the persona invariant: identity shapes language, nothing else.

    The persona test guards ``as_prompt_traits``; identity travels on its own
    payload, so it needs the same guard or the boundary has a hole.
    """
    payload = as_prompt_identity(identity_from("eng_manager"), "en")
    assert set(payload) == {"role", "name", "title", "register"}
    forbidden = ("difficulty", "score", "weight", "question", "outcome", "pass")
    for key in payload:
        assert not any(bad in key.lower() for bad in forbidden)


def test_presets_are_complete_and_carry_no_subject_matter() -> None:
    """Every enum member has a preset, and none smuggles in domain content.

    A preset that mentioned a technology would push the interviewer toward
    talking about it, which the leading-question judge scores as introducing
    content the student had not produced.
    """
    assert set(PRESETS) == {r.value for r in InterviewerRole}
    banned = ("postgres", "kubernetes", "python", "algorithm", "database", "api")
    for preset in PRESETS.values():
        blob = " ".join(
            [preset.title_en, preset.title_vi, preset.register_en, preset.register_vi]
        ).lower()
        assert not any(term in blob for term in banned), preset.role.value
