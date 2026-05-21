"""Unit tests for the quiz GENERATION stage (T5.6).

Covers acceptance items in plan §5658-5697:
* Mocked LLM → parsed :class:`GeneratedQuestion` list.
* Audit threading: stage_name="generation" + role=GENERATION + pipeline_run_id.
* ``previous_questions`` arg threaded into the user prompt (regen path).
* Parser drops malformed-options entries instead of crashing.
* No file in ``stages/generation/`` exceeds 250 LOC.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.ai.llm import LLMResult, LLMRole
from abridgeai.ai.retrieval import ChunkWithDistance
from abridgeai.features.quizzes.ai.stages.generation import (
    GeneratedQuestion,
    generate_questions,
    parse_generation_response,
)


def _chunk(content: str = "Recursion is...") -> ChunkWithDistance:
    return ChunkWithDistance(
        chunk_id=uuid4(),
        material_version_id=uuid4(),
        course_id=uuid4(),
        lesson_id=uuid4(),
        content=content,
        distance=0.1,
    )


def _llm_result(content_json: dict[str, object]) -> LLMResult:
    return LLMResult(
        role=LLMRole.GENERATION,
        tier="large",
        model_name="test-model",
        base_url="https://example.test/v1",
        stage_name="generation",
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


def _good_payload(question: str = "What is recursion?") -> dict[str, object]:
    return {
        "questions": [
            {
                "position": 1,
                "question_type": "mcq",
                "question": question,
                "options": {
                    "A": "A function calling itself",
                    "B": "An array iteration",
                    "C": "A loop",
                    "D": "A pointer",
                },
                "correct_answer": "A",
                "explanation": "Recursion is when a function invokes itself.",
                "bloom_level": "understand",
                "difficulty": "medium",
                "source_refs": [str(uuid4())],
            }
        ]
    }


@pytest.mark.asyncio
async def test_generate_returns_parsed_questions() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_good_payload()))

    questions = await generate_questions(
        title="Algorithms 101",
        config={"difficulty": "medium", "question_types": ["mcq"]},
        chunks=[_chunk()],
        templates=[],
        kg_context=None,
        db=AsyncMock(),
        pipeline_run_id=uuid4(),
        gateway=gateway,
    )

    assert len(questions) == 1
    only = questions[0]
    assert isinstance(only, GeneratedQuestion)
    assert only.prompt_text == "What is recursion?"
    assert only.bloom_level == "understand"
    assert {opt.option_key for opt in only.options} == {"A", "B", "C", "D"}
    assert sum(1 for opt in only.options if opt.is_correct) == 1


@pytest.mark.asyncio
async def test_generation_passes_audit_metadata() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_good_payload()))
    run_id = uuid4()
    parent_id = uuid4()

    await generate_questions(
        title="Q",
        config={},
        chunks=[_chunk()],
        templates=[],
        kg_context=None,
        db=AsyncMock(),
        pipeline_run_id=run_id,
        parent_run_id=parent_id,
        gateway=gateway,
    )

    gateway.generate_json.assert_awaited_once()
    kwargs = gateway.generate_json.await_args.kwargs
    assert kwargs["role"] is LLMRole.GENERATION
    assert kwargs["stage_name"] == "generation"
    assert kwargs["pipeline_run_id"] == run_id
    assert kwargs["parent_run_id"] == parent_id
    assert "Source context:" in kwargs["user_prompt"]


@pytest.mark.asyncio
async def test_avoids_duplicates_via_previous_questions() -> None:
    gateway = AsyncMock()
    gateway.generate_json = AsyncMock(return_value=_llm_result(_good_payload("New question?")))

    previous = [
        "What is the time complexity of binary search?",
        "Explain memoization with a code example.",
    ]
    await generate_questions(
        title="Q",
        config={},
        chunks=[_chunk()],
        templates=[],
        kg_context=None,
        db=AsyncMock(),
        pipeline_run_id=uuid4(),
        previous_questions=previous,
        gateway=gateway,
    )

    user_prompt: str = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "Previously generated questions" in user_prompt
    for stem in previous:
        assert stem in user_prompt


def test_parser_handles_malformed_options() -> None:
    payload: dict[str, object] = {
        "questions": [
            {
                "question": "Bad MCQ — only 3 options",
                "options": {"A": "x", "B": "y", "C": "z"},
                "correct_answer": "A",
                "explanation": "n/a",
                "bloom_level": "understand",
                "difficulty": "medium",
            },
            {
                "question": "Good MCQ",
                "options": {"A": "1", "B": "2", "C": "3", "D": "4"},
                "correct_answer": "B",
                "explanation": "OK",
                "bloom_level": "understand",
                "difficulty": "medium",
            },
            "not even a dict",
            {"question": "", "options": {}},
        ]
    }

    parsed = parse_generation_response(payload)
    assert len(parsed) == 1
    assert parsed[0].prompt_text == "Good MCQ"
    correct = [opt for opt in parsed[0].options if opt.is_correct]
    assert len(correct) == 1
    assert correct[0].option_key == "B"


def test_parser_emits_word_bank_for_fill_blank() -> None:
    """fill_blank questions ship a drag-and-drop word bank as ``options``.

    Each correct answer must be present verbatim and flagged
    ``is_correct=True``; distractors must be flagged ``is_correct=False``;
    keys + positions must satisfy the ``quiz_question_options``
    UNIQUE(question_id, option_key) and UNIQUE(question_id, position)
    constraints (option_key fits VARCHAR(5)).
    """
    payload: dict[str, object] = {
        "questions": [
            {
                "question": "A data warehouse is ___, ___, time-variant, and ___.",
                "question_type": "fill_blank",
                "correct_answer": ["subject-oriented", "integrated", "non-volatile"],
                "options": [
                    "subject-oriented",
                    "integrated",
                    "non-volatile",
                    "transactional",
                    "operational",
                ],
                "explanation": "Inmon's four-property definition.",
                "bloom_level": "remember",
                "difficulty": "medium",
            }
        ]
    }
    parsed = parse_generation_response(payload)
    assert len(parsed) == 1
    q = parsed[0]
    assert q.question_type == "fill_blank"
    assert len(q.options) == 5
    correct_texts = {opt.option_text for opt in q.options if opt.is_correct}
    assert correct_texts == {"subject-oriented", "integrated", "non-volatile"}
    distractor_texts = {opt.option_text for opt in q.options if not opt.is_correct}
    assert distractor_texts == {"transactional", "operational"}
    keys = [opt.option_key for opt in q.options]
    positions = [opt.position for opt in q.options]
    assert keys == ["O01", "O02", "O03", "O04", "O05"]
    assert positions == [1, 2, 3, 4, 5]
    assert all(len(k) <= 5 for k in keys)


def test_parser_rejects_fill_blank_without_distractor() -> None:
    """A bank with only correct answers gives the answer away — reject."""
    payload: dict[str, object] = {
        "questions": [
            {
                "question": "A data warehouse is ___ updatable.",
                "question_type": "fill_blank",
                "correct_answer": ["non"],
                "options": ["non"],
                "explanation": "x",
                "bloom_level": "remember",
                "difficulty": "easy",
            }
        ]
    }
    assert parse_generation_response(payload) == []


def test_parser_rejects_fill_blank_without_options() -> None:
    """Missing word bank means the FE has nothing to render — reject."""
    payload: dict[str, object] = {
        "questions": [
            {
                "question": "A data warehouse is ___ updatable.",
                "question_type": "fill_blank",
                "correct_answer": ["non"],
                "explanation": "x",
                "bloom_level": "remember",
                "difficulty": "easy",
            }
        ]
    }
    assert parse_generation_response(payload) == []


def test_parser_backfills_missing_correct_answer_into_bank() -> None:
    """If the LLM forgets to put a correct answer in ``options``, the
    parser prepends it so the bank still contains the right answer
    (the validator can still reject on quality grounds, but we don't
    drop the question outright).
    """
    payload: dict[str, object] = {
        "questions": [
            {
                "question": "A data warehouse is ___ updatable.",
                "question_type": "fill_blank",
                "correct_answer": ["non"],
                "options": ["always", "sometimes", "never"],
                "explanation": "x",
                "bloom_level": "remember",
                "difficulty": "easy",
            }
        ]
    }
    parsed = parse_generation_response(payload)
    assert len(parsed) == 1
    texts = [opt.option_text for opt in parsed[0].options]
    assert "non" in texts
    correct = [opt for opt in parsed[0].options if opt.is_correct]
    assert len(correct) == 1
    assert correct[0].option_text == "non"


def test_no_god_file_in_generation() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "quizzes" / "ai" / "stages" / "generation"
    assert target.is_dir(), f"generation stage dir not found at {target}"
    budget = {"logic.py": 250, "parsers.py": 250, "__init__.py": 250}
    for path in target.glob("*.py"):
        with path.open() as fh:
            line_count = sum(1 for _ in fh)
        cap = budget.get(path.name, 250)
        assert line_count <= cap, f"{path.name} has {line_count} LOC > {cap}"
