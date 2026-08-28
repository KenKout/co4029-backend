"""Unit tests for the interview validation stage (T6.6).

Mirrors the partition + parser coverage style from
``test_quiz_validation_stage.py`` but targets the interview-specific
five-criterion verdict shape:

* GROUNDED — empty / mismatched ``source_refs`` rejects the question.
* DIFFICULTY_COHERENT — abrupt rank drops (hard → easy) reject.
* TYPE_MATCHES_CONFIG — overrepresented buckets fail (9/1/0 vs 60/30/10).
* NOT_LEADING — LLM judgement plumbed through; failure flips accepted.
* LENGTH_REASONABLE — prompt under 20 or over 500 chars rejects.

Plus an architectural test enforcing prompts live in ``.j2`` files
only.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.features.interviews.ai.stages.generation.parsers import (
    InterviewQuestionDraft,
)
from abridgeai.features.interviews.ai.stages.validation import (
    ValidationCriterion,
    Verdict,
    parse_leading_verdicts,
    validate_interview_questions,
)
from abridgeai.features.interviews.ai.stages.validation.logic import (
    VALIDATION_STAGE_NAME,
)


def _draft(
    *,
    question_type: str = "technical",
    difficulty: str = "medium",
    expected_depth: int = 3,
    prompt_text: str = "Walk me through how you would design a caching layer.",
    source_refs: list[UUID] | None = None,
    linked_outcome_id: UUID | None = None,
    variant_group_id: UUID | None = None,
) -> InterviewQuestionDraft:
    return InterviewQuestionDraft(
        question_type=question_type,  # type: ignore[arg-type]
        prompt_text=prompt_text,
        difficulty=difficulty,  # type: ignore[arg-type]
        expected_depth=expected_depth,
        linked_outcome_id=linked_outcome_id,
        variant_group_id=variant_group_id,
        source_refs=list(source_refs) if source_refs is not None else [],
        rationale="",
    )


def _context(chunk_ids: list[UUID]) -> SimpleNamespace:
    return SimpleNamespace(chunks=[SimpleNamespace(id=cid, content="src") for cid in chunk_ids])


def _run(
    *, source_chunk_ids: list[UUID] | None = None, type_weights: dict | None = None
) -> SimpleNamespace:
    config_json: dict = {}
    if source_chunk_ids is not None:
        config_json["retrieval"] = {"source_chunk_ids": [str(cid) for cid in source_chunk_ids]}
    if type_weights is not None:
        config_json["type_weights"] = type_weights
    return SimpleNamespace(id=uuid4(), config_json=config_json)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        title="Designing distributed systems",
        persona="neutral",
    )


def _gateway_returning(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(
        generate_json=AsyncMock(return_value=SimpleNamespace(content_json=payload)),
    )


def _accept_all(question_count: int) -> dict:
    return {
        "verdicts": [
            {"question_index": index, "not_leading": True} for index in range(question_count)
        ]
    }


@pytest.mark.asyncio
async def test_rejects_ungrounded_question() -> None:
    chunk = uuid4()
    drafts = [_draft(source_refs=[])]
    context = _context([chunk])
    run = _run(source_chunk_ids=[chunk])
    gateway = _gateway_returning(_accept_all(1))

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=run,
        config=_config(),
        drafts=drafts,
        context=context,
        gateway=gateway,
    )

    assert len(verdicts) == 1
    assert verdicts[0].accepted is False
    assert ValidationCriterion.GROUNDED in verdicts[0].failed_criteria


@pytest.mark.asyncio
async def test_accepts_well_formed_question() -> None:
    chunk = uuid4()
    drafts = [
        _draft(source_refs=[chunk]),
        _draft(question_type="behavioral", source_refs=[chunk]),
    ]
    context = _context([chunk])
    run = _run(source_chunk_ids=[chunk])
    gateway = _gateway_returning(_accept_all(2))

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=run,
        config=_config(),
        drafts=drafts,
        context=context,
        gateway=gateway,
    )

    assert all(v.accepted for v in verdicts)
    assert all(v.failed_criteria == [] for v in verdicts)
    assert [v.question_index for v in verdicts] == [0, 1]


@pytest.mark.asyncio
async def test_length_check() -> None:
    chunk = uuid4()
    drafts = [
        _draft(prompt_text="too?", source_refs=[chunk]),
        _draft(prompt_text="x" * 1000, source_refs=[chunk]),
        _draft(source_refs=[chunk]),
    ]
    context = _context([chunk])
    run = _run(source_chunk_ids=[chunk])
    gateway = _gateway_returning(_accept_all(3))

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=run,
        config=_config(),
        drafts=drafts,
        context=context,
        gateway=gateway,
    )

    assert verdicts[0].accepted is False
    assert ValidationCriterion.LENGTH_REASONABLE in verdicts[0].failed_criteria
    assert verdicts[1].accepted is False
    assert ValidationCriterion.LENGTH_REASONABLE in verdicts[1].failed_criteria
    assert verdicts[2].accepted is True


@pytest.mark.asyncio
async def test_type_mix_check() -> None:
    chunk = uuid4()
    drafts = [_draft(source_refs=[chunk]) for _ in range(9)]
    drafts.append(_draft(question_type="behavioral", source_refs=[chunk]))
    context = _context([chunk])
    run = _run(source_chunk_ids=[chunk])
    gateway = _gateway_returning(_accept_all(len(drafts)))

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=run,
        config=_config(),
        drafts=drafts,
        context=context,
        gateway=gateway,
    )

    technical_failures = [
        v for v, d in zip(verdicts, drafts, strict=True) if d.question_type == "technical"
    ]
    behavioral_failures = [
        v for v, d in zip(verdicts, drafts, strict=True) if d.question_type == "behavioral"
    ]
    assert all(
        ValidationCriterion.TYPE_MATCHES_CONFIG in v.failed_criteria for v in technical_failures
    )
    assert all(
        ValidationCriterion.TYPE_MATCHES_CONFIG not in v.failed_criteria
        for v in behavioral_failures
    )


@pytest.mark.asyncio
async def test_skip_type_mix_in_variant_mode() -> None:
    chunk = uuid4()
    drafts = [_draft(source_refs=[chunk]) for _ in range(9)]
    drafts.append(_draft(question_type="behavioral", source_refs=[chunk]))
    context = _context([chunk])
    run = _run(source_chunk_ids=[chunk])
    gateway = _gateway_returning(_accept_all(len(drafts)))

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=run,
        config=_config(),
        drafts=drafts,
        context=context,
        gateway=gateway,
        skip_type_mix=True,
    )

    assert all(v.accepted for v in verdicts)
    assert all(ValidationCriterion.TYPE_MATCHES_CONFIG not in v.failed_criteria for v in verdicts)


@pytest.mark.asyncio
async def test_variant_group_allows_partial_distinct_angles() -> None:
    chunk, outcome, group = uuid4(), uuid4(), uuid4()
    drafts = [
        _draft(
            question_type="technical",
            source_refs=[chunk],
            linked_outcome_id=outcome,
            variant_group_id=group,
        ),
        _draft(
            question_type="system_design",
            source_refs=[chunk],
            linked_outcome_id=outcome,
            variant_group_id=group,
        ),
    ]

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=_run(source_chunk_ids=[chunk]),
        config=_config(),
        drafts=drafts,
        context=_context([chunk]),
        gateway=_gateway_returning(_accept_all(2)),
        skip_type_mix=True,
    )

    assert all(v.accepted for v in verdicts)


@pytest.mark.asyncio
async def test_variant_group_rejects_duplicate_angle_or_mismatched_outcome() -> None:
    chunk, group = uuid4(), uuid4()
    drafts = [
        _draft(
            question_type="technical",
            source_refs=[chunk],
            linked_outcome_id=uuid4(),
            variant_group_id=group,
        ),
        _draft(
            question_type="technical",
            source_refs=[chunk],
            linked_outcome_id=uuid4(),
            variant_group_id=group,
        ),
    ]

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=_run(source_chunk_ids=[chunk]),
        config=_config(),
        drafts=drafts,
        context=_context([chunk]),
        gateway=_gateway_returning(_accept_all(2)),
        skip_type_mix=True,
    )

    assert all(
        ValidationCriterion.VARIANT_GROUP_COHERENT in verdict.failed_criteria
        for verdict in verdicts
    )


@pytest.mark.asyncio
async def test_variant_group_rejects_different_outcomes_with_distinct_angles() -> None:
    chunk, group = uuid4(), uuid4()
    drafts = [
        _draft(
            question_type="technical",
            source_refs=[chunk],
            linked_outcome_id=uuid4(),
            variant_group_id=group,
        ),
        _draft(
            question_type="system_design",
            source_refs=[chunk],
            linked_outcome_id=uuid4(),
            variant_group_id=group,
        ),
    ]

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=_run(source_chunk_ids=[chunk]),
        config=_config(),
        drafts=drafts,
        context=_context([chunk]),
        gateway=_gateway_returning(_accept_all(2)),
        skip_type_mix=True,
    )

    assert all(
        verdict.failed_criteria == [ValidationCriterion.VARIANT_GROUP_COHERENT]
        for verdict in verdicts
    )


@pytest.mark.asyncio
async def test_difficulty_progression_rejects_abrupt_drop() -> None:
    chunk = uuid4()
    drafts = [
        _draft(difficulty="hard", source_refs=[chunk]),
        _draft(difficulty="easy", source_refs=[chunk]),
    ]
    context = _context([chunk])
    run = _run(source_chunk_ids=[chunk])
    gateway = _gateway_returning(_accept_all(2))

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=run,
        config=_config(),
        drafts=drafts,
        context=context,
        gateway=gateway,
    )

    assert verdicts[0].accepted is True
    assert ValidationCriterion.DIFFICULTY_COHERENT in verdicts[1].failed_criteria


@pytest.mark.asyncio
async def test_not_leading_check_threads_through_llm() -> None:
    chunk = uuid4()
    drafts = [_draft(source_refs=[chunk])]
    context = _context([chunk])
    run = _run(source_chunk_ids=[chunk])
    payload = {
        "verdicts": [
            {
                "question_index": 0,
                "not_leading": False,
                "rationale": "Stem assumes microservices win.",
            }
        ]
    }
    gateway = _gateway_returning(payload)

    verdicts = await validate_interview_questions(
        AsyncMock(),
        run=run,
        config=_config(),
        drafts=drafts,
        context=context,
        gateway=gateway,
    )

    assert verdicts[0].accepted is False
    assert ValidationCriterion.NOT_LEADING in verdicts[0].failed_criteria


@pytest.mark.asyncio
async def test_audit_stage_name_and_role() -> None:
    chunk = uuid4()
    drafts = [_draft(source_refs=[chunk])]
    context = _context([chunk])
    run = _run(source_chunk_ids=[chunk])
    gateway = _gateway_returning(_accept_all(1))
    db = AsyncMock()

    await validate_interview_questions(
        db,
        run=run,
        config=_config(),
        drafts=drafts,
        context=context,
        gateway=gateway,
    )

    gateway.generate_json.assert_awaited_once()
    kwargs = gateway.generate_json.await_args.kwargs
    assert kwargs["stage_name"] == "interview_validation"
    assert VALIDATION_STAGE_NAME == "interview_validation"
    assert kwargs["pipeline_run_id"] == run.id
    from abridgeai.ai.llm import LLMRole

    assert kwargs["role"] == LLMRole.INTERVIEW_VALIDATION


def test_parse_leading_verdicts_handles_malformed() -> None:
    payload = {
        "verdicts": [
            "not a dict",
            {"question_index": "nope"},
            {"question_index": 5, "not_leading": True},
            {"question_index": 0, "not_leading": False, "rationale": "leading"},
        ]
    }
    parsed = parse_leading_verdicts(payload, question_count=2)
    assert len(parsed) == 2
    assert parsed[0].not_leading is False
    assert parsed[0].rationale == "leading"
    assert parsed[1].not_leading is True
    assert parsed[1].is_default is True


def test_parse_leading_verdicts_zero_questions() -> None:
    assert parse_leading_verdicts({"verdicts": []}, question_count=0) == []


def test_parse_leading_verdicts_none_payload() -> None:
    parsed = parse_leading_verdicts(None, question_count=2)
    assert all(v.not_leading and v.is_default for v in parsed)


def test_verdict_dataclass_shape() -> None:
    verdict = Verdict(
        question_index=0,
        accepted=False,
        failed_criteria=[ValidationCriterion.GROUNDED],
        rationale="ungrounded",
    )
    assert verdict.question_index == 0
    assert verdict.accepted is False
    assert verdict.failed_criteria == [ValidationCriterion.GROUNDED]
    assert ValidationCriterion.GROUNDED.value == "grounded"


def test_jinja_prompts_in_j2_only() -> None:
    here = Path(__file__).resolve().parents[2]
    prompts_dir = (
        here / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "validation" / "prompts"
    )
    assert prompts_dir.is_dir()
    j2_files = sorted(p.name for p in prompts_dir.glob("*.j2"))
    assert j2_files == ["system.j2", "user.j2"]
    other_files = [p for p in prompts_dir.iterdir() if p.is_file() and p.suffix != ".j2"]
    assert other_files == []
