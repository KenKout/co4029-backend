"""Unit tests for the follow-up stage's semantic repetition guard + PII rules.

The follow-up stage used to see exactly one question and one answer, so it could
re-ask — in different words — something already covered earlier in the session.
Exact-id de-duplication in ``selection.py`` cannot catch that, because a
generated follow-up has no question id at all. The fix feeds the model the recent
interviewer turns; these tests pin the wiring (window size, ordering, filtering)
and the prompt rules, not the model's judgement.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from abridgeai.features.interviews.ai.stages.followup.logic import (
    RECENT_QUESTION_WINDOW,
    _recent_interviewer_questions,
    maybe_generate_followup,
)

_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "abridgeai/features/interviews/ai/stages/followup/prompts/system.j2"
)
_GENERATION_PROMPT = (
    Path(__file__).resolve().parents[2]
    / "abridgeai/features/interviews/ai/stages/generation/prompts/system.j2"
)


def _savepoint_cm() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _db(*, existing_followup: bool = False, recent: list[str] | None = None) -> AsyncMock:
    """A db whose two SELECTs answer independently.

    ``maybe_generate_followup`` issues the cap check first (count via
    ``scalar_one``) and then the recent-questions query (``scalars().all()``),
    so one shared result object would conflate them.
    """
    db = AsyncMock()
    cap_result = MagicMock()
    cap_result.scalar_one = MagicMock(return_value=1 if existing_followup else 0)
    recent_result = MagicMock()
    recent_result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=list(recent or [])))
    )
    db.execute = AsyncMock(side_effect=[cap_result, recent_result])
    db.begin_nested = MagicMock(side_effect=_savepoint_cm)
    return db


def _gateway(payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        generate_json=AsyncMock(return_value=SimpleNamespace(content_json=payload))
    )


def _sent_user_prompt(gateway: SimpleNamespace) -> dict[str, object]:
    kwargs = gateway.generate_json.await_args.kwargs
    return json.loads(kwargs["user_prompt"])


# ── the recent-questions loader ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recent_questions_returned_chronologically() -> None:
    """The query is newest-first for the LIMIT; the model must read them in order."""
    db = AsyncMock()
    result = MagicMock()
    result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=["newest", "middle", "oldest"]))
    )
    db.execute = AsyncMock(return_value=result)

    got = await _recent_interviewer_questions(db, session_id=uuid.uuid4(), limit=3)

    assert got == ["oldest", "middle", "newest"]


@pytest.mark.asyncio
async def test_recent_questions_drops_blanks_and_strips() -> None:
    db = AsyncMock()
    result = MagicMock()
    result.scalars = MagicMock(
        return_value=MagicMock(all=MagicMock(return_value=["  Q2  ", "   ", None, "Q1"]))
    )
    db.execute = AsyncMock(return_value=result)

    got = await _recent_interviewer_questions(db, session_id=uuid.uuid4(), limit=5)

    assert got == ["Q1", "Q2"]


@pytest.mark.asyncio
async def test_recent_questions_zero_limit_skips_the_query() -> None:
    """A zero window must not cost a round-trip."""
    db = AsyncMock()
    db.execute = AsyncMock()

    got = await _recent_interviewer_questions(db, session_id=uuid.uuid4(), limit=0)

    assert got == []
    db.execute.assert_not_awaited()


# ── wiring into the prompt ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_recent_questions_reach_the_user_prompt() -> None:
    db = _db(recent=["What is an index?", "How does a B-tree help?"])
    gateway = _gateway({"is_sufficient": True, "followup": None, "rationale": "ok"})

    await maybe_generate_followup(
        db,
        session=SimpleNamespace(id=uuid.uuid4()),
        current_question=SimpleNamespace(id=uuid.uuid4(), prompt_text="Explain indexes."),
        student_answer="They make lookups faster.",
        gateway=gateway,
    )

    payload = _sent_user_prompt(gateway)
    assert payload["recent_interviewer_questions"] == [
        "How does a B-tree help?",
        "What is an index?",
    ]


@pytest.mark.asyncio
async def test_empty_history_still_sends_the_field() -> None:
    """First question of a session: the key must exist so the template is stable."""
    db = _db(recent=[])
    gateway = _gateway({"is_sufficient": True, "followup": None, "rationale": "ok"})

    await maybe_generate_followup(
        db,
        session=SimpleNamespace(id=uuid.uuid4()),
        current_question=SimpleNamespace(id=uuid.uuid4(), prompt_text="Q?"),
        student_answer="A.",
        gateway=gateway,
    )

    assert _sent_user_prompt(gateway)["recent_interviewer_questions"] == []


@pytest.mark.asyncio
async def test_cap_rule_still_short_circuits_before_any_llm_call() -> None:
    """Adding a second query must not weaken the existing single-follow-up cap."""
    db = _db(existing_followup=True)
    gateway = _gateway({"is_sufficient": False, "followup": "probe?", "rationale": "x"})

    result = await maybe_generate_followup(
        db,
        session=SimpleNamespace(id=uuid.uuid4()),
        current_question=SimpleNamespace(id=uuid.uuid4(), prompt_text="Q?"),
        student_answer="A.",
        gateway=gateway,
        # The cap became a configurable budget (default 2); the rule this test
        # pins is "an exhausted budget short-circuits before any LLM call", so
        # exercise it with a 1-cap and one existing follow-up.
        max_follow_ups_per_question=1,
    )

    assert result is None
    gateway.generate_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_answer_short_circuits_before_any_query() -> None:
    db = _db()
    gateway = _gateway({"is_sufficient": True, "followup": None, "rationale": "ok"})

    result = await maybe_generate_followup(
        db,
        session=SimpleNamespace(id=uuid.uuid4()),
        current_question=SimpleNamespace(id=uuid.uuid4(), prompt_text="Q?"),
        student_answer="   ",
        gateway=gateway,
    )

    assert result is None
    db.execute.assert_not_awaited()
    gateway.generate_json.assert_not_awaited()


def test_window_is_bounded() -> None:
    """The prompt must not grow with session length."""
    assert 1 <= RECENT_QUESTION_WINDOW <= 10


# ── prompt rules (repetition + PII) ──────────────────────────────────────────


def test_followup_prompt_documents_the_repetition_rule() -> None:
    body = _PROMPT.read_text(encoding="utf-8")
    assert "recent_interviewer_questions" in body
    assert "semantically overlaps" in body


def test_followup_prompt_forbids_pii() -> None:
    body = _PROMPT.read_text(encoding="utf-8").lower()
    assert "personally identifiable information" in body
    # The specific categories matter: a vague "don't ask personal things" is not
    # actionable for a model.
    for category in ("date of birth", "phone number", "passport"):
        assert category in body


def test_generation_prompt_forbids_pii() -> None:
    """Authored questions outlive a session, so the guard belongs here too."""
    body = _GENERATION_PROMPT.read_text(encoding="utf-8").lower()
    assert "personally identifiable information" in body
    assert "never elicit personal data" in body
