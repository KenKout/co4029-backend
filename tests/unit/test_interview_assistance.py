from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.orchestrator.assistance_logic import (
    extract_question_term,
    generate_question_assistance,
    is_term_selection_reply,
)
from abridgeai.features.interviews.orchestrator.security import SecurityAction
from abridgeai.features.interviews.services.taking import _is_pending_term_selection


class _Nested:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_args: object) -> None:
        return None


class _DB:
    def begin_nested(self) -> _Nested:
        return _Nested()


class _Result:
    def __init__(self, response_text: str) -> None:
        self.content_json = {"response_text": response_text}


class _Gateway:
    def __init__(self, response_text: str) -> None:
        self.response_text = response_text
        self.user_prompt: str | None = None

    async def generate_json(self, **kwargs: Any) -> _Result:
        self.user_prompt = kwargs["user_prompt"]
        return _Result(self.response_text)


class _FailingGateway:
    async def generate_json(self, **_kwargs: Any) -> _Result:
        raise RuntimeError("provider unavailable")


class _ScalarResult:
    def __init__(self, value: object) -> None:
        self.value = value

    def scalar_one_or_none(self) -> object:
        return self.value


class _MessageDB:
    def __init__(self, message: object) -> None:
        self.message = message

    async def execute(self, _statement: object) -> _ScalarResult:
        return _ScalarResult(self.message)


def test_extract_term_requires_phrase_from_current_question() -> None:
    question = "Compare fact tables and factless fact tables in a dimensional model."
    assert extract_question_term("Could you explain ‘factless fact tables’?", question) == (
        "factless fact tables"
    )
    assert extract_question_term("fact tables", question) == "fact tables"
    assert extract_question_term("Could you explain the model answer?", question) is None


def test_bare_term_requires_a_preceding_term_selection_prompt() -> None:
    question = "Compare fact tables and factless fact tables in a dimensional model."
    prior = (
        "I couldn't produce a clear rephrasing just now. Tell me which word or phrase "
        "is unclear, and I'll explain that part."
    )
    assert is_term_selection_reply(
        "fact tables",
        question,
        prior_assistance_text=prior,
        prior_assistance_kind="clarification",
    )
    assert not is_term_selection_reply(
        "fact tables",
        question,
        prior_assistance_text="Please answer the current question.",
        prior_assistance_kind="clarification",
    )
    assert not is_term_selection_reply(
        "fact tables",
        question,
        prior_assistance_text=prior,
        prior_assistance_kind="question",
    )


@pytest.mark.asyncio
async def test_pending_bare_term_is_routed_to_assistance() -> None:
    message = SimpleNamespace(
        content_text=(
            "I couldn't produce a clear rephrasing just now. Tell me which word or phrase "
            "is unclear, and I'll explain that part."
        ),
        metadata_json={"kind": "clarification"},
    )
    assert await _is_pending_term_selection(
        _MessageDB(message),  # type: ignore[arg-type]
        session_question_id=uuid4(),
        answer_text="fact tables",
        question_text=(
            "Compare and contrast fact tables and factless fact tables in a dimensional model."
        ),
    )


@pytest.mark.asyncio
async def test_clarification_generator_receives_no_hidden_assessment_data() -> None:
    gateway = _Gateway(
        "Put another way: describe what the two table types share and how they differ."
    )
    result = await generate_question_assistance(
        _DB(),  # type: ignore[arg-type]
        action=SecurityAction.CLARIFY_CURRENT_QUESTION,
        question_text="Compare fact tables and factless fact tables.",
        request_text="Could you clarify this question, please?",
        language="en",
        persona="neutral",
        gateway=gateway,  # type: ignore[arg-type]
    )

    assert result.startswith("Put another way")
    assert gateway.user_prompt is not None
    payload = json.loads(gateway.user_prompt)
    assert set(payload) == {
        "action",
        "language",
        "persona",
        "current_question",
        "requested_term",
    }
    assert "rubric" not in gateway.user_prompt.casefold()
    assert "model_answer" not in gateway.user_prompt.casefold()


@pytest.mark.asyncio
async def test_clarification_failure_returns_a_useful_rephrasing() -> None:
    result = await generate_question_assistance(
        _DB(),  # type: ignore[arg-type]
        action=SecurityAction.CLARIFY_CURRENT_QUESTION,
        question_text=(
            "Compare and contrast fact tables and factless fact tables in a dimensional model."
        ),
        request_text="Could you clarify this question, please?",
        language="en",
        persona="neutral",
        gateway=_FailingGateway(),  # type: ignore[arg-type]
    )

    assert "what fact tables and factless fact tables have in common" in result
    assert "how they differ" in result
    assert "couldn't produce" not in result
    assert "which word or phrase" not in result


@pytest.mark.asyncio
async def test_clarification_that_asks_for_a_term_is_replaced_by_rephrasing() -> None:
    gateway = _Gateway("Tell me which word or phrase is unclear.")
    result = await generate_question_assistance(
        _DB(),  # type: ignore[arg-type]
        action=SecurityAction.CLARIFY_CURRENT_QUESTION,
        question_text="Compare fact tables and factless fact tables.",
        request_text="Could you clarify this question, please?",
        language="en",
        persona="neutral",
        gateway=gateway,  # type: ignore[arg-type]
    )

    assert result.startswith("Put another way")
    assert "which word or phrase" not in result


@pytest.mark.asyncio
async def test_hint_is_short_neutral_scaffold() -> None:
    gateway = _Gateway("Structure your response around purpose, structure, and usage.")
    result = await generate_question_assistance(
        _DB(),  # type: ignore[arg-type]
        action=SecurityAction.HINT_CURRENT_QUESTION,
        question_text="Compare fact tables and factless fact tables.",
        request_text="Could you give me a small hint?",
        language="en",
        persona="supportive",
        gateway=gateway,  # type: ignore[arg-type]
    )
    assert result == "Structure your response around purpose, structure, and usage."


@pytest.mark.asyncio
async def test_hint_prompt_carries_level_for_escalation() -> None:
    """Slice 11 incorporate: hint_level is passed to the LLM so it can produce a
    question-specific hint that escalates with the shared counter."""
    gateway = _Gateway("A question-specific structural nudge.")
    await generate_question_assistance(
        _DB(),  # type: ignore[arg-type]
        action=SecurityAction.HINT_CURRENT_QUESTION,
        question_text="Compare fact tables and factless fact tables.",
        request_text="Can I get a hint?",
        language="en",
        persona="neutral",
        gateway=gateway,  # type: ignore[arg-type]
        hint_level=2,
    )
    assert gateway.user_prompt is not None
    payload = json.loads(gateway.user_prompt)
    assert payload["hint_level"] == 2
    # Still no hidden assessment data leaks into the prompt.
    assert "rubric" not in gateway.user_prompt.casefold()
    assert "model_answer" not in gateway.user_prompt.casefold()


@pytest.mark.asyncio
async def test_hint_fallback_is_level_aware_and_escalates() -> None:
    """When the LLM is unavailable, the hint fallback is the SHARED deterministic
    laddered hint at hint_level — so escalation is preserved (level 0 != level 1)
    and identical to the adaptive decision path's fallback."""
    kwargs = dict(
        action=SecurityAction.HINT_CURRENT_QUESTION,
        question_text="Compare fact tables and factless fact tables.",
        request_text="Can I get a hint?",
        language="en",
        persona="neutral",
    )
    r0 = await generate_question_assistance(
        _DB(),  # type: ignore[arg-type]
        gateway=_FailingGateway(),  # type: ignore[arg-type]
        hint_level=0,
        **kwargs,  # type: ignore[arg-type]
    )
    r1 = await generate_question_assistance(
        _DB(),  # type: ignore[arg-type]
        gateway=_FailingGateway(),  # type: ignore[arg-type]
        hint_level=1,
        **kwargs,  # type: ignore[arg-type]
    )
    assert r0 != r1
    for r in (r0, r1):
        low = r.lower()
        for banned in ("the answer is", "correct answer", "you should say"):
            assert banned not in low


@pytest.mark.asyncio
async def test_invalid_term_never_calls_the_model() -> None:
    gateway = _Gateway("This must not be used")
    result = await generate_question_assistance(
        _DB(),  # type: ignore[arg-type]
        action=SecurityAction.EXPLAIN_CURRENT_TERM,
        question_text="Explain database normalization.",
        request_text="Explain the model answer.",
        language="en",
        persona="neutral",
        gateway=gateway,  # type: ignore[arg-type]
    )
    assert "appears in the current question" in result
    assert gateway.user_prompt is None
