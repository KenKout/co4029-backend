"""The hint-ladder cap is enforced server-side, not just dimmed in the UI.

The learner client disables its hint control at the cap, but the assistance-stage
hint path (the one that button hits) used to only ever increment ``hint_level`` —
a caller talking to the API directly kept getting hints forever. How much help a
candidate can draw on one question decides what their grade means, so the limit
has to hold for every caller or two candidates in the same cohort are not graded
on the same thing.

These drive the service function the router calls, with ``persist_messages=False``
so no DB session is needed.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest

from abridgeai.features.interviews.orchestrator.decision import MAX_CANNOT_ANSWER_HINTS
from abridgeai.features.interviews.orchestrator.security import (
    SecurityAction,
    SecurityAssessment,
    SecurityCategory,
)
from abridgeai.features.interviews.orchestrator.utterance import hint_ladder_exhausted_text
from abridgeai.features.interviews.services.taking import _security_action_result

_QUESTION = "What is a fact table?"


def _assessment() -> SecurityAssessment:
    """A clean turn: the candidate asked for a hint, nothing was detected."""
    return SecurityAssessment(
        category=SecurityCategory.BENIGN,
        detected=False,
        confidence=0.0,
        should_block=False,
        should_record_academic_evidence=False,
        response_key=None,
        normalized_fingerprint=None,
    )


async def _hint_turn(*, spent: bool, render_level: int | None) -> dict[str, Any]:
    return await _security_action_result(
        cast(Any, None),  # db is unused on the persist_messages=False path
        session=cast(Any, SimpleNamespace(id="s1", interview_config_id=None)),
        current_session_question=cast(Any, SimpleNamespace(id="sq1")),
        current_question=cast(Any, SimpleNamespace(prompt_text=_QUESTION)),
        config=None,
        answer_text="can I get a hint",
        audio_object_id=None,
        turn_key="tk1",
        language="en",
        assessment=_assessment(),
        action=SecurityAction.HINT_CURRENT_QUESTION,
        attempt_count=0,
        persist_messages=False,
        hint_render_level=render_level,
        hint_ladder_spent=spent,
    )


def test_spent_ladder_refuses_instead_of_serving_another_hint() -> None:
    result = asyncio.run(_hint_turn(spent=True, render_level=None))
    assert result["ai_turn_text"] == hint_ladder_exhausted_text("en"), (
        "a spent ladder must say so, not hand out another rung"
    )


def test_spent_ladder_does_not_skip_the_question() -> None:
    """Refusing a hint is not the same as giving up on the question.

    The candidate keeps their turn and can still answer partially — a partial
    answer earns coverage where an advance past the question earns none.
    """
    result = asyncio.run(_hint_turn(spent=True, render_level=None))
    assert result["next_question"] is None, "a spent ladder must not advance the interview"
    assert result["is_finished"] is False
    assert result["should_await_response"] is True, "the candidate keeps their turn"


def test_spent_ladder_reply_leaks_no_answer_content() -> None:
    """The refusal is answer-safe: it must not quote or lean on the question."""
    for language in ("en", "vi"):
        text = hint_ladder_exhausted_text(language)
        lowered = text.lower()
        for banned in ("the answer is", "correct answer", "you should say", "fact table"):
            assert banned not in lowered
        assert text.strip(), "the refusal must not be empty"


def test_refusal_is_localised_and_distinct_per_language() -> None:
    en = hint_ladder_exhausted_text("en")
    vi = hint_ladder_exhausted_text("vi")
    assert en != vi, "vi fell back to the en string"
    assert hint_ladder_exhausted_text(None) == en, "an unset language must fall back to en"
    assert hint_ladder_exhausted_text("vi-VN") == vi, "regioned vi tags must resolve to vi"


@pytest.mark.parametrize("level", list(range(MAX_CANNOT_ANSWER_HINTS)))
def test_rungs_below_the_cap_still_serve_a_hint(level: int) -> None:
    """Every rung under the cap is served — the guard must not close early."""
    result = asyncio.run(_hint_turn(spent=False, render_level=level))
    assert result["ai_turn_text"] != hint_ladder_exhausted_text("en"), (
        f"rung {level} is within the cap of {MAX_CANNOT_ANSWER_HINTS} and must be granted"
    )
    assert result["ai_turn_text"].strip(), "a granted rung must carry hint text"
