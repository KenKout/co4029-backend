"""Unit tests for the quiz validation stage (T5.7).

Covers acceptance items in plan §5703-5743:
* Verdicts correctly partition into accepted vs rejected with reasons
* Audit row carries ``stage_name="validation"``
* Apply-verdicts preserves question identity through the partition
* Parser is permissive against malformed verdict rows
* No file in ``stages/validation/`` exceeds 250 LOC
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.features.quizzes.ai.stages.validation import (
    Verdict,
    apply_verdicts,
    parse_validation_response,
    validate_questions,
)
from abridgeai.features.quizzes.ai.stages.validation.logic import VALIDATION_STAGE_NAME


def _question(position: int, prompt: str = "Stem?") -> dict[str, object]:
    return {
        "position": position,
        "id": f"qid-{position}",
        "prompt_text": f"{prompt} ({position})",
        "options": [
            {"option_key": "A", "option_text": "alpha", "is_correct": position == 1},
            {"option_key": "B", "option_text": "beta", "is_correct": position == 2},
            {"option_key": "C", "option_text": "gamma", "is_correct": False},
            {"option_key": "D", "option_text": "delta", "is_correct": False},
        ],
        "explanation": f"because-{position}",
        "bloom_level": "understand",
        "difficulty": "medium",
    }


def _chunk_stub(text: str = "x") -> SimpleNamespace:
    return SimpleNamespace(id=uuid4(), content=text, metadata={"section_title": "S"})


@pytest.mark.asyncio
async def test_validation_partitions_verdicts() -> None:
    questions = [_question(i) for i in range(1, 6)]
    chunks = [_chunk_stub("source")]

    fake_payload = {
        "verdicts": [
            {"position": 1, "verdict": "accept", "reason": "OK"},
            {"position": 2, "verdict": "reject", "reason": "SOURCE_LEAK: 'in the slides'"},
            {"position": 3, "verdict": "accept", "reason": "OK"},
            {
                "position": 4,
                "verdict": "reject",
                "reason": "AMBIGUOUS: B and C both work",
                "evidence_excerpt": "two correct options",
            },
            {"position": 5, "verdict": "accept"},
        ]
    }

    fake_result = SimpleNamespace(content_json=fake_payload)
    gateway = SimpleNamespace(generate_json=AsyncMock(return_value=fake_result))
    db = AsyncMock()

    llm_result, verdicts = await validate_questions(
        "Quiz Title",
        chunks,
        questions,
        db,
        pipeline_run_id=uuid4(),
        gateway=gateway,
    )

    assert llm_result is fake_result
    assert [v.verdict for v in verdicts] == ["accept", "reject", "accept", "reject", "accept"]

    accepted, rejected, reasons = apply_verdicts(questions, verdicts)
    assert len(accepted) == 3
    assert len(rejected) == 2
    rejected_positions = [r["position"] for r in rejected]
    assert rejected_positions == [2, 4]
    assert any("SOURCE_LEAK" in reason for reason in reasons)
    assert any("AMBIGUOUS" in reason for reason in reasons)
    assert rejected[1]["evidence_excerpt"] == "two correct options"


@pytest.mark.asyncio
async def test_validation_audit_stage_name() -> None:
    questions = [_question(1)]
    chunks = [_chunk_stub()]
    pipeline_run_id = uuid4()

    fake_result = SimpleNamespace(content_json={"verdicts": [{"position": 1, "verdict": "accept"}]})
    generate_mock = AsyncMock(return_value=fake_result)
    gateway = SimpleNamespace(generate_json=generate_mock)
    db = AsyncMock()

    await validate_questions(
        "Title",
        chunks,
        questions,
        db,
        pipeline_run_id=pipeline_run_id,
        gateway=gateway,
    )

    generate_mock.assert_awaited_once()
    kwargs = generate_mock.await_args.kwargs
    assert kwargs["stage_name"] == "validation"
    assert VALIDATION_STAGE_NAME == "validation"
    assert kwargs["pipeline_run_id"] == pipeline_run_id
    from abridgeai.ai.llm import LLMRole

    assert kwargs["role"] == LLMRole.VALIDATION


def test_apply_verdicts_preserves_question_id() -> None:
    questions = [_question(1, "First"), _question(2, "Second")]
    verdicts = [
        Verdict(position=1, verdict="accept", reasons=[]),
        Verdict(
            position=2,
            verdict="reject",
            reasons=["SHAPE: only 3 options"],
            evidence_excerpt="A, B, C",
        ),
    ]

    accepted, rejected, reasons = apply_verdicts(questions, verdicts)

    assert len(accepted) == 1
    assert accepted[0]["id"] == "qid-1"
    assert accepted[0]["prompt_text"] == "First (1)"
    assert len(rejected) == 1
    assert rejected[0]["question_id"] == "qid-2"
    assert rejected[0]["prompt_text"] == "Second (2)"
    assert rejected[0]["reasons"] == ["SHAPE: only 3 options"]
    assert reasons == ["SHAPE: only 3 options"]


def test_parser_handles_malformed_verdict() -> None:
    payload = {
        "verdicts": [
            {"position": "not-a-number", "verdict": "accept"},
            {"position": 99, "verdict": "accept", "reason": "out of range"},
            {"position": 1, "verdict": "garbage"},
            "this is not a dict",
            {"position": 2, "verdict": "REJECT", "reason": "UNGROUNDED: invented"},
        ]
    }

    verdicts = parse_validation_response(payload, question_count=3)

    assert len(verdicts) == 3
    assert verdicts[0].verdict == "accept"
    assert "did not return" in verdicts[0].reasons[0]
    assert verdicts[1].verdict == "reject"
    assert verdicts[1].reasons == ["UNGROUNDED: invented"]
    assert verdicts[2].verdict == "accept"


def test_parser_returns_empty_for_zero_questions() -> None:
    assert parse_validation_response({"verdicts": []}, question_count=0) == []


def test_parser_handles_none_payload() -> None:
    verdicts = parse_validation_response(None, question_count=2)
    assert len(verdicts) == 2
    assert all(v.verdict == "accept" for v in verdicts)


def test_no_god_file_in_validation_stage() -> None:
    here = Path(__file__).resolve().parents[2]
    target = here / "abridgeai" / "features" / "quizzes" / "ai" / "stages" / "validation"
    assert target.is_dir(), f"validation stage dir not found at {target}"
    for path in target.glob("*.py"):
        line_count = sum(1 for _ in path.open())
        assert line_count <= 250, f"{path.name} has {line_count} LOC > 250"
