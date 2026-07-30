"""Integration tests for the interview follow-up stage (T6.7).

Plan §6414-6453. Asserts:

* Sufficient answers do NOT trigger a follow-up (return None).
* Shallow answers return the LLM-generated probing question text.
* The single-cap rule short-circuits before any LLM call when a
  follow-up already exists for the (session, question) pair.
* The gateway is invoked with ``LLMRole.INTERVIEW_FOLLOWUP`` (small tier
  per ``ai/llm/roles.py``) and ``stage_name='interview_followup'``.
* Prompts live in ``prompts/*.j2`` only (no Python string interpolation).

The tests inject ``AsyncMock`` for ``db`` and ``gateway`` because this stage
is exercised in unit-shaped isolation: the runtime contract is that
``maybe_generate_followup`` consults the DB for the cap check and the
gateway for the sufficiency judgement, both via injectable seams.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from abridgeai.ai.llm import LLMRole
from abridgeai.features.interviews.ai.stages.followup import (
    FOLLOWUP_STAGE_NAME,
    FollowupVerdict,
    maybe_generate_followup,
    parse_followup_response,
)


def _question_stub() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        prompt_text="Explain how event loops schedule I/O callbacks.",
    )


def _session_stub() -> SimpleNamespace:
    return SimpleNamespace(id=uuid.uuid4())


def _db_with_no_existing_followup() -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=None)
    db.execute = AsyncMock(return_value=result)
    db.begin_nested = MagicMock(side_effect=_savepoint_cm)
    return db


def _savepoint_cm() -> MagicMock:
    """A ``begin_nested()`` stand-in yielding an async context manager.

    ``maybe_generate_followup`` wraps the gateway call in
    ``async with db.begin_nested()`` (SAVEPOINT). A plain ``AsyncMock``
    attribute returns a coroutine, which is not an async CM, so the helper
    must hand back an object exposing ``__aenter__``/``__aexit__``.
    """
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=cm)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def _db_with_existing_followup() -> AsyncMock:
    db = AsyncMock()
    result = MagicMock()
    result.scalar_one_or_none = MagicMock(return_value=uuid.uuid4())
    db.execute = AsyncMock(return_value=result)
    return db


def _gateway_returning(payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        generate_json=AsyncMock(return_value=SimpleNamespace(content_json=payload))
    )


@pytest.mark.asyncio
async def test_returns_none_for_sufficient_answer() -> None:
    db = _db_with_no_existing_followup()
    gateway = _gateway_returning(
        {
            "is_sufficient": True,
            "followup": None,
            "rationale": "Answer covers scheduling and callback queue.",
        }
    )

    result = await maybe_generate_followup(
        db,
        session=_session_stub(),
        current_question=_question_stub(),
        student_answer="Node uses libuv to poll fds and drains microtasks first.",
        gateway=gateway,
    )

    assert result is None
    gateway.generate_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_returns_followup_text_for_shallow_answer() -> None:
    db = _db_with_no_existing_followup()
    expected = "Can you walk through what happens when a microtask is queued?"
    gateway = _gateway_returning(
        {
            "is_sufficient": False,
            "followup": expected,
            "rationale": "Answer was a single sentence with no mechanism.",
        }
    )

    result = await maybe_generate_followup(
        db,
        session=_session_stub(),
        current_question=_question_stub(),
        student_answer="It runs callbacks.",
        gateway=gateway,
    )

    assert result == expected


@pytest.mark.asyncio
async def test_returns_none_when_followup_already_exists() -> None:
    db = _db_with_existing_followup()
    gateway = _gateway_returning({"is_sufficient": False, "followup": "should not be asked"})

    result = await maybe_generate_followup(
        db,
        session=_session_stub(),
        current_question=_question_stub(),
        student_answer="It runs callbacks.",
        gateway=gateway,
    )

    assert result is None
    gateway.generate_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_uses_followup_role_binding() -> None:
    db = _db_with_no_existing_followup()
    gateway = _gateway_returning({"is_sufficient": True, "followup": None})

    pipeline_run_id = uuid.uuid4()
    await maybe_generate_followup(
        db,
        session=_session_stub(),
        current_question=_question_stub(),
        student_answer="Some non-trivial answer about event loops.",
        gateway=gateway,
        pipeline_run_id=pipeline_run_id,
    )

    gateway.generate_json.assert_awaited_once()
    kwargs = gateway.generate_json.await_args.kwargs
    assert kwargs["role"] is LLMRole.INTERVIEW_FOLLOWUP
    assert kwargs["stage_name"] == FOLLOWUP_STAGE_NAME == "interview_followup"
    assert kwargs["pipeline_run_id"] == pipeline_run_id


def test_jinja_prompts_in_j2_only() -> None:
    here = Path(__file__).resolve().parents[2]
    stage_dir = here / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "followup"
    prompts_dir = stage_dir / "prompts"

    assert (prompts_dir / "system.j2").is_file()
    # No user.j2: this stage builds its user prompt with json.dumps(...) in
    # logic.py, so only the system prompt is a template. The rule this test
    # actually guards is "prompt PROSE lives in .j2, never in Python".

    for python_path in stage_dir.glob("*.py"):
        body = python_path.read_text(encoding="utf-8")
        assert "You are a strict but fair technical interviewer" not in body, (
            f"{python_path.name} embeds prompt prose; move it to prompts/system.j2"
        )
        assert "{{ student_answer" not in body, (
            f"{python_path.name} embeds Jinja templating; keep it in prompts/*.j2"
        )


def test_parser_handles_malformed_payload() -> None:
    verdict = parse_followup_response(None)
    assert isinstance(verdict, FollowupVerdict)
    assert verdict.is_sufficient is True
    assert verdict.followup is None


def test_parser_promotes_followup_when_text_present_but_flag_missing() -> None:
    verdict = parse_followup_response({"followup": "Why specifically?"})
    assert verdict.is_sufficient is False
    assert verdict.followup == "Why specifically?"


@pytest.mark.asyncio
async def test_empty_answer_short_circuits_without_db_or_llm() -> None:
    db = AsyncMock()
    gateway = SimpleNamespace(generate_json=AsyncMock())

    result = await maybe_generate_followup(
        db,
        session=_session_stub(),
        current_question=_question_stub(),
        student_answer="   ",
        gateway=gateway,
    )

    assert result is None
    db.execute.assert_not_called()
    gateway.generate_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_gateway_failure_is_swallowed_inside_savepoint() -> None:
    """A failing gateway/audit write must not bubble to /respond.

    Regression for the ck_ai_model_calls_parent_ref 500: the audit flush
    raised, the bare ``except`` swallowed it, but the poisoned session then
    500'd the request. The SAVEPOINT now contains the failure so the stage
    returns None and the request transaction stays usable.
    """
    db = _db_with_no_existing_followup()
    gateway = SimpleNamespace(
        generate_json=AsyncMock(side_effect=RuntimeError("audit flush failed"))
    )

    result = await maybe_generate_followup(
        db,
        session=_session_stub(),
        current_question=_question_stub(),
        student_answer="It runs callbacks.",
        gateway=gateway,
    )

    assert result is None
    db.begin_nested.assert_called_once()
    gateway.generate_json.assert_awaited_once()
