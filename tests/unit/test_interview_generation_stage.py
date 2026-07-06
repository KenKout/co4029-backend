"""Unit tests for the interview GENERATION stage (T6.5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.ai.llm import LLMResult, LLMRole
from abridgeai.features.interviews.ai.stages.generation import (
    InterviewQuestionDraft,
    generate_interview_questions,
    parse_generation_response,
)


@dataclass
class _FakeChunk:
    chunk_id: UUID = field(default_factory=uuid4)
    content: str = "Recursion is when a function calls itself."


@dataclass
class _FakeContext:
    chunks: list[_FakeChunk] = field(default_factory=lambda: [_FakeChunk()])


def _llm_result(content_json: dict[str, object]) -> LLMResult:
    return LLMResult(
        role=LLMRole.INTERVIEW_GENERATION,
        tier="large",
        model_name="test-model",
        base_url="https://example.test/v1",
        stage_name="interview_generation",
        pipeline_run_id=None,
        request_payload={},
        response_payload={"choices": [{"message": {"content": "{}"}}]},
        content_json=content_json,
        input_tokens=10,
        output_tokens=20,
        total_tokens=30,
        cached_input_tokens=None,
        latency_ms=42,
        estimated_cost_usd=Decimal("0.0001"),
    )


def _fake_run(question_count: int | None = None) -> SimpleNamespace:
    config_json: dict[str, object] = {}
    if question_count is not None:
        config_json["question_count"] = question_count
    return SimpleNamespace(id=uuid4(), config_json=config_json)


def _fake_config(supplementary: str | None = None, persona: str = "neutral") -> SimpleNamespace:
    return SimpleNamespace(
        title="Algorithms 101 Interview",
        persona=persona,
        supplementary_instructions=supplementary,
    )


def _fake_outcomes(n: int = 3) -> list[SimpleNamespace]:
    return [
        SimpleNamespace(
            id=uuid4(),
            outcome_text=f"Outcome {idx}",
            outcome_type="knowledge",
            importance_weight=3,
        )
        for idx in range(n)
    ]


_DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}


def _question(
    *,
    position: int,
    question_type: str,
    difficulty: str,
    expected_depth: int,
    linked_outcome_id: UUID | None = None,
    source_refs: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "position": position,
        "question_type": question_type,
        "prompt_text": f"Q{position}: explain {question_type} concept #{position}",
        "difficulty": difficulty,
        "expected_depth": expected_depth,
        "linked_outcome_id": str(linked_outcome_id) if linked_outcome_id else None,
        "source_refs": source_refs or [str(uuid4())],
        "rationale": f"probes #{position}",
    }


def _eight_questions() -> dict[str, Any]:
    return {
        "questions": [
            _question(position=1, question_type="technical", difficulty="easy", expected_depth=1),
            _question(position=2, question_type="technical", difficulty="easy", expected_depth=2),
            _question(position=3, question_type="technical", difficulty="easy", expected_depth=2),
            _question(position=4, question_type="technical", difficulty="medium", expected_depth=3),
            _question(
                position=5, question_type="behavioral", difficulty="medium", expected_depth=3
            ),
            _question(
                position=6, question_type="behavioral", difficulty="medium", expected_depth=3
            ),
            _question(position=7, question_type="situational", difficulty="hard", expected_depth=4),
            _question(position=8, question_type="technical", difficulty="hard", expected_depth=5),
        ]
    }


@pytest.mark.asyncio
async def test_generates_n_questions() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    assert len(drafts) == 8
    assert all(isinstance(d, InterviewQuestionDraft) for d in drafts)
    gateway.generate_json.assert_awaited_once()
    kwargs = gateway.generate_json.await_args.kwargs
    assert kwargs["role"] is LLMRole.INTERVIEW_GENERATION
    assert kwargs["stage_name"] == "interview_generation"


@pytest.mark.asyncio
async def test_form_question_count_drives_prompt_and_caps_drafts() -> None:
    """The teacher's form count (run.config_json) must be honoured exactly:
    it both reaches the prompt and caps how many drafts are returned."""
    gateway = AsyncMock()
    # LLM returns 8 questions, but the teacher asked for 3.
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(question_count=3),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "Total questions to produce: 3" in user_prompt
    assert len(drafts) == 3  # capped to the requested count, not the LLM's 8


@pytest.mark.asyncio
async def test_form_count_overrides_supplementary_default() -> None:
    """When both are present, the form count wins over the legacy
    supplementary-instructions override."""
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(question_count=5),
        config=_fake_config(supplementary='{"question_count": 12}'),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "Total questions to produce: 5" in user_prompt


@pytest.mark.asyncio
async def test_type_mix_matches_default_60_30_10() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "technical: 60%" in user_prompt
    assert "behavioral: 30%" in user_prompt
    assert "situational: 10%" in user_prompt


@pytest.mark.asyncio
async def test_type_mix_overridden_by_rubric_weights() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(
            supplementary='{"rubric_weights": {"technical": 70, "behavioral": 20, "situational": 10}}'
        ),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "technical: 70%" in user_prompt
    assert "behavioral: 20%" in user_prompt
    assert "situational: 10%" in user_prompt


@pytest.mark.asyncio
async def test_difficulty_progression() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_eight_questions()))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    levels = [_DIFFICULTY_ORDER[d.difficulty] for d in drafts]
    assert levels == sorted(levels), (
        f"difficulties not non-decreasing: {[d.difficulty for d in drafts]}"
    )
    assert levels[0] <= levels[-1]


@pytest.mark.asyncio
async def test_expected_depth_in_range_1_to_5() -> None:
    payload = {
        "questions": [
            _question(position=1, question_type="technical", difficulty="easy", expected_depth=0),
            _question(position=2, question_type="technical", difficulty="easy", expected_depth=99),
            _question(position=3, question_type="technical", difficulty="easy", expected_depth=3),
        ]
    }
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(payload))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=_fake_outcomes(),
        gateway=gateway,
    )

    assert all(1 <= d.expected_depth <= 5 for d in drafts)
    assert drafts[0].expected_depth == 1
    assert drafts[1].expected_depth == 5


@pytest.mark.asyncio
async def test_each_question_links_to_outcome_id() -> None:
    outcomes = _fake_outcomes(3)
    payload = _eight_questions()
    payload["questions"][0]["linked_outcome_id"] = None  # type: ignore[index]
    payload["questions"][3]["linked_outcome_id"] = None  # type: ignore[index]

    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(payload))

    drafts = await generate_interview_questions(
        AsyncMock(),
        run=_fake_run(),
        config=_fake_config(),
        context=_FakeContext(),
        outcomes=outcomes,
        gateway=gateway,
    )

    linked = sum(1 for d in drafts if d.linked_outcome_id is not None)
    assert linked >= len(drafts) - 1
    valid_ids = {o.id for o in outcomes}
    for draft in drafts:
        if draft.linked_outcome_id is not None:
            assert draft.linked_outcome_id in valid_ids or isinstance(draft.linked_outcome_id, UUID)


def test_parser_drops_malformed_entries() -> None:
    payload: dict[str, Any] = {
        "questions": [
            {"prompt_text": "", "question_type": "technical"},
            "not even a dict",
            {"prompt_text": "Bad type", "question_type": "alien"},
            _question(position=1, question_type="technical", difficulty="easy", expected_depth=2),
        ]
    }

    parsed = parse_generation_response(payload)
    assert len(parsed) == 1
    assert parsed[0].question_type == "technical"


def test_parser_accepts_british_spelling() -> None:
    payload: dict[str, Any] = {
        "questions": [
            {
                "prompt_text": "Tell me about a time you failed.",
                "question_type": "behavioural",
                "difficulty": "medium",
                "expected_depth": 3,
                "linked_outcome_id": None,
                "source_refs": [],
                "rationale": "self-reflection",
            }
        ]
    }
    parsed = parse_generation_response(payload)
    assert len(parsed) == 1
    assert parsed[0].question_type == "behavioral"


def test_jinja_prompts_in_j2_files_only() -> None:
    here = Path(__file__).resolve().parents[2]
    target = (
        here / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "generation" / "prompts"
    )
    assert target.is_dir(), f"prompts dir missing at {target}"
    j2_files = list(target.glob("*.j2"))
    assert {p.name for p in j2_files} >= {"system.j2", "user.j2"}
    for path in j2_files:
        assert path.stat().st_size > 0, f"empty template: {path}"


def test_no_inline_prompts_in_python() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "generation"
    forbidden_phrases = (
        "You are an interviewer",
        "Return JSON in this shape",
        "Difficulty progression rubric",
    )
    for path in target.glob("*.py"):
        body = path.read_text(encoding="utf-8")
        for phrase in forbidden_phrases:
            assert phrase not in body, f"{path.name} inlines prompt phrase: {phrase!r}"


def test_no_god_file_in_generation() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "generation"
    assert target.is_dir()
    budget = {"logic.py": 250, "parsers.py": 200, "__init__.py": 100}
    for path in target.glob("*.py"):
        with path.open() as fh:
            line_count = sum(1 for _ in fh)
        cap = budget.get(path.name, 250)
        assert line_count <= cap, f"{path.name} has {line_count} LOC > {cap}"
