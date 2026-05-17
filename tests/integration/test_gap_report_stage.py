"""Integration tests for the GAP REPORT stage (T6.9).

These tests mock :class:`LLMGateway` and the DB so they don't touch the
network or live Postgres. The stage's contract is: take a completed
session + rubric scores + quiz attempts + a course/module scope, return
a :class:`GapReportDraft` ready for T6.11 services to persist.

Tests cover:

* discrepancy formula (theory − practice)
* strengths picked from criteria scoring ≥ 4.0
* weaknesses picked from criteria scoring < 3.0
* study plan surfaces ≥ 3 distinct resources across all items
* gateway is invoked with ``LLMRole.GAP_REPORT_GENERATION``
* prompts live in ``.j2`` files only (no string literal prompts in code)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from abridgeai.features.interviews.ai.stages.evaluation import (
    CriterionScore,
    ResponseEvaluation,
    RubricScores,
)
from abridgeai.features.interviews.ai.stages.gap_report import (
    GAP_REPORT_STAGE_NAME,
    GapReportDraft,
    StudyPlanItem,
    generate_gap_report,
    parse_gap_report_response,
)


def _make_session() -> SimpleNamespace:
    return SimpleNamespace(id=uuid4(), student_id=uuid4())


def _rubric(
    *,
    aggregated: dict[str, float],
    total_score: float,
    response_evaluations: list[ResponseEvaluation] | None = None,
) -> RubricScores:
    return RubricScores(
        response_evaluations=response_evaluations or [],
        aggregated=aggregated,
        total_score=total_score,
    )


def _quiz_attempt(score_percent: float | None) -> SimpleNamespace:
    return SimpleNamespace(score_percent=score_percent)


def _lesson_row(lesson_id: UUID, title: str = "Lesson", summary: str = "") -> dict[str, object]:
    return {"id": lesson_id, "title": title, "summary": summary}


def _resource_row(
    resource_id: UUID,
    lesson_id: UUID,
    *,
    title: str = "Resource",
    resource_type: str = "pdf",
) -> dict[str, object]:
    return {
        "id": resource_id,
        "lesson_id": lesson_id,
        "title": title,
        "resource_type": resource_type,
    }


def _db_returning_library(
    lessons: list[dict[str, object]],
    resources: list[dict[str, object]],
) -> AsyncMock:
    """Mock db.execute to return lesson rows then resource rows via .mappings().all()."""

    lesson_result = MagicMock()
    lesson_mappings = MagicMock()
    lesson_mappings.all = MagicMock(return_value=lessons)
    lesson_result.mappings = MagicMock(return_value=lesson_mappings)

    resource_result = MagicMock()
    resource_mappings = MagicMock()
    resource_mappings.all = MagicMock(return_value=resources)
    resource_result.mappings = MagicMock(return_value=resource_mappings)

    db = AsyncMock()
    db.execute = AsyncMock(side_effect=[lesson_result, resource_result])
    return db


def _gateway_returning(payload: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        generate_json=AsyncMock(return_value=SimpleNamespace(content_json=payload))
    )


def _llm_payload(
    *,
    strengths: list[str] | None = None,
    weaknesses: list[str] | None = None,
    study_plan: list[dict[str, object]] | None = None,
    student_summary: str = "You are doing well overall.",
    teacher_summary: str = "Student shows mixed performance.",
) -> dict[str, object]:
    return {
        "strengths": strengths or [],
        "weaknesses": weaknesses or [],
        "study_plan": study_plan or [],
        "student_summary": student_summary,
        "teacher_summary": teacher_summary,
    }


@pytest.mark.asyncio
async def test_discrepancy_calculated_correctly() -> None:
    """theory=80, practice=60 → discrepancy=20."""

    session = _make_session()
    quiz_attempts = [_quiz_attempt(80.0), _quiz_attempt(80.0)]
    rubric = _rubric(
        aggregated={"technical_accuracy": 3.0},
        total_score=60.0,
    )
    db = _db_returning_library([], [])
    gateway = _gateway_returning(_llm_payload())

    draft = await generate_gap_report(
        db,
        session=session,
        rubric_scores=rubric,
        quiz_attempts=quiz_attempts,
        course_id=uuid4(),
        gateway=gateway,
    )

    assert draft.theory_score_avg == pytest.approx(80.0)
    assert draft.practice_score == pytest.approx(60.0)
    assert draft.discrepancy_score == pytest.approx(20.0)


@pytest.mark.asyncio
async def test_theory_average_zero_when_no_quiz_attempts() -> None:
    session = _make_session()
    rubric = _rubric(aggregated={"a": 3.0}, total_score=70.0)
    db = _db_returning_library([], [])
    gateway = _gateway_returning(_llm_payload())

    draft = await generate_gap_report(
        db,
        session=session,
        rubric_scores=rubric,
        quiz_attempts=[],
        course_id=uuid4(),
        gateway=gateway,
    )

    assert draft.theory_score_avg == 0.0
    assert draft.discrepancy_score == pytest.approx(-70.0)


@pytest.mark.asyncio
async def test_strengths_picked_from_high_score_criteria() -> None:
    """rubric.aggregated has 1 score ≥ 4.0 → strength list non-empty."""

    session = _make_session()
    rubric = _rubric(
        aggregated={"technical_accuracy": 4.5, "communication": 2.0},
        total_score=65.0,
    )
    db = _db_returning_library([], [])
    gateway = _gateway_returning(_llm_payload())

    draft = await generate_gap_report(
        db,
        session=session,
        rubric_scores=rubric,
        quiz_attempts=[_quiz_attempt(70.0)],
        course_id=uuid4(),
        gateway=gateway,
    )

    assert len(draft.strengths) >= 1
    assert any("technical_accuracy" in bullet for bullet in draft.strengths)


@pytest.mark.asyncio
async def test_weaknesses_picked_from_low_score_criteria() -> None:
    """rubric.aggregated has 1 score < 3.0 → weakness list non-empty."""

    session = _make_session()
    rubric = _rubric(
        aggregated={"technical_accuracy": 4.0, "communication": 1.5},
        total_score=55.0,
    )
    db = _db_returning_library([], [])
    gateway = _gateway_returning(_llm_payload())

    draft = await generate_gap_report(
        db,
        session=session,
        rubric_scores=rubric,
        quiz_attempts=[_quiz_attempt(70.0)],
        course_id=uuid4(),
        gateway=gateway,
    )

    assert len(draft.weaknesses) >= 1
    assert any("communication" in bullet for bullet in draft.weaknesses)


@pytest.mark.asyncio
async def test_study_plan_links_to_at_least_3_resources_total() -> None:
    """Happy path with 3 weaknesses → at least 3 resources across study_plan items."""

    session = _make_session()
    lesson_a, lesson_b, lesson_c = uuid4(), uuid4(), uuid4()
    res_a1, res_a2, res_b1, res_c1 = uuid4(), uuid4(), uuid4(), uuid4()
    lessons = [
        _lesson_row(lesson_a, title="A"),
        _lesson_row(lesson_b, title="B"),
        _lesson_row(lesson_c, title="C"),
    ]
    resources = [
        _resource_row(res_a1, lesson_a),
        _resource_row(res_a2, lesson_a),
        _resource_row(res_b1, lesson_b),
        _resource_row(res_c1, lesson_c),
    ]
    db = _db_returning_library(lessons, resources)

    rubric = _rubric(
        aggregated={
            "technical_accuracy": 1.5,
            "communication": 1.5,
            "problem_solving": 1.5,
        },
        total_score=30.0,
    )

    payload = _llm_payload(
        weaknesses=[
            "technical_accuracy: gaps on definitions",
            "communication: rambling answers",
            "problem_solving: misses trade-offs",
        ],
        study_plan=[
            {
                "topic": "Refresh fundamentals",
                "weakness_summary": "Definitions are shaky.",
                "suggested_lesson_id": str(lesson_a),
                "suggested_resource_ids": [str(res_a1), str(res_a2)],
                "priority": "high",
            },
            {
                "topic": "Structure your answers",
                "weakness_summary": "Rambling answers without conclusion.",
                "suggested_lesson_id": str(lesson_b),
                "suggested_resource_ids": [str(res_b1)],
                "priority": "medium",
            },
            {
                "topic": "Practice trade-off analysis",
                "weakness_summary": "Missing pros/cons.",
                "suggested_lesson_id": str(lesson_c),
                "suggested_resource_ids": [str(res_c1)],
                "priority": "medium",
            },
        ],
    )
    gateway = _gateway_returning(payload)

    draft = await generate_gap_report(
        db,
        session=session,
        rubric_scores=rubric,
        quiz_attempts=[_quiz_attempt(75.0)],
        course_id=uuid4(),
        module_id=uuid4(),
        gateway=gateway,
    )

    assert len(draft.study_plan) == 3
    distinct = {rid for item in draft.study_plan for rid in item.suggested_resource_ids}
    assert len(distinct) >= 3
    for item in draft.study_plan:
        assert len(item.suggested_resource_ids) >= 1


@pytest.mark.asyncio
async def test_study_plan_backfills_resources_when_llm_omits_them() -> None:
    """LLM returns lesson_ids but no resources → fallback to lesson resources + backfill."""

    session = _make_session()
    lesson_a, lesson_b = uuid4(), uuid4()
    res_a1, res_b1, res_b2 = uuid4(), uuid4(), uuid4()
    lessons = [_lesson_row(lesson_a), _lesson_row(lesson_b)]
    resources = [
        _resource_row(res_a1, lesson_a),
        _resource_row(res_b1, lesson_b),
        _resource_row(res_b2, lesson_b),
    ]
    db = _db_returning_library(lessons, resources)

    rubric = _rubric(aggregated={"a": 2.0}, total_score=40.0)
    payload = _llm_payload(
        study_plan=[
            {
                "topic": "Topic A",
                "weakness_summary": "weak A",
                "suggested_lesson_id": str(lesson_a),
                "suggested_resource_ids": [],
                "priority": "high",
            },
            {
                "topic": "Topic B",
                "weakness_summary": "weak B",
                "suggested_lesson_id": str(lesson_b),
                "suggested_resource_ids": [],
                "priority": "medium",
            },
        ],
    )
    gateway = _gateway_returning(payload)

    draft = await generate_gap_report(
        db,
        session=session,
        rubric_scores=rubric,
        quiz_attempts=[_quiz_attempt(80.0)],
        course_id=uuid4(),
        gateway=gateway,
    )

    distinct = {rid for item in draft.study_plan for rid in item.suggested_resource_ids}
    assert len(distinct) >= 3


@pytest.mark.asyncio
async def test_uses_gap_report_role_binding() -> None:
    """LLM call uses LLMRole.GAP_REPORT_GENERATION."""

    session = _make_session()
    rubric = _rubric(aggregated={"a": 3.0}, total_score=60.0)
    db = _db_returning_library([], [])
    pipeline_run_id = uuid4()
    gateway = _gateway_returning(_llm_payload())

    await generate_gap_report(
        db,
        session=session,
        rubric_scores=rubric,
        quiz_attempts=[_quiz_attempt(80.0)],
        course_id=uuid4(),
        pipeline_run_id=pipeline_run_id,
        gateway=gateway,
    )

    kwargs = gateway.generate_json.await_args.kwargs
    from abridgeai.ai.llm import LLMRole

    assert kwargs["role"] == LLMRole.GAP_REPORT_GENERATION
    assert kwargs["stage_name"] == "gap_report"
    assert GAP_REPORT_STAGE_NAME == "gap_report"
    assert kwargs["pipeline_run_id"] == pipeline_run_id
    assert kwargs["parent_run_id"] == pipeline_run_id


def test_jinja_prompts_in_j2_only() -> None:
    here = Path(__file__).resolve().parents[1]
    stage_dir = (
        here.parent / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "gap_report"
    )
    assert (stage_dir / "prompts" / "system.j2").is_file()
    assert (stage_dir / "prompts" / "user.j2").is_file()

    for py_file in stage_dir.glob("*.py"):
        text = py_file.read_text()
        assert 'system_prompt = "' not in text, f"{py_file.name}: inline system prompt"
        assert "system_prompt = '" not in text, f"{py_file.name}: inline system prompt"


def test_parser_drops_invalid_uuids_and_unknown_priority() -> None:
    payload = {
        "strengths": ["a: ok", 42, "  "],
        "weaknesses": ["b: weak"],
        "study_plan": [
            {
                "topic": "Topic A",
                "weakness_summary": "weak",
                "suggested_lesson_id": "not-a-uuid",
                "suggested_resource_ids": ["nope", str(uuid4())],
                "priority": "URGENT",
            },
            {
                "topic": "  ",
                "weakness_summary": "skipped",
            },
        ],
        "student_summary": "Fine.",
        "teacher_summary": "Detail.",
    }
    parsed = parse_gap_report_response(payload)
    assert parsed["strengths"] == ["a: ok"]
    assert parsed["weaknesses"] == ["b: weak"]
    assert len(parsed["study_plan"]) == 1
    item = parsed["study_plan"][0]
    assert item["suggested_lesson_id"] is None
    assert len(item["suggested_resource_ids"]) == 1
    assert item["priority"] == "medium"


def test_parser_returns_empty_for_garbage_payload() -> None:
    parsed = parse_gap_report_response(None)
    assert parsed["strengths"] == []
    assert parsed["study_plan"] == []
    assert parsed["student_summary"] == ""


def test_study_plan_item_dataclass_shape() -> None:
    item = StudyPlanItem(
        topic="t",
        weakness_summary="w",
        suggested_lesson_id=None,
        suggested_resource_ids=[uuid4()],
        priority="high",
    )
    assert item.priority == "high"
    assert len(item.suggested_resource_ids) == 1


def test_gap_report_draft_dataclass_shape() -> None:
    draft = GapReportDraft(
        discrepancy_score=10.0,
        theory_score_avg=70.0,
        practice_score=60.0,
        strengths=["a"],
        weaknesses=["b"],
        study_plan=[],
        student_summary="s",
        teacher_summary="t",
    )
    assert draft.discrepancy_score == 10.0
    assert draft.report_json == {}


@pytest.mark.asyncio
async def test_evidence_excerpts_surfaced_to_llm_user_prompt() -> None:
    """Per-response justifications surface in the user prompt for grounding."""

    session = _make_session()
    qid = uuid4()
    response_evaluations = [
        ResponseEvaluation(
            session_question_id=qid,
            criterion_scores=[
                CriterionScore(
                    criterion="technical_accuracy",
                    score=1.0,
                    justification="Confused recursion with iteration.",
                ),
                CriterionScore(
                    criterion="communication",
                    score=2.0,
                    justification="Unclear pronoun referents.",
                ),
            ],
        ),
    ]
    rubric = _rubric(
        aggregated={"technical_accuracy": 1.0, "communication": 2.0},
        total_score=30.0,
        response_evaluations=response_evaluations,
    )
    db = _db_returning_library([], [])
    gateway = _gateway_returning(_llm_payload())

    await generate_gap_report(
        db,
        session=session,
        rubric_scores=rubric,
        quiz_attempts=[_quiz_attempt(80.0)],
        course_id=uuid4(),
        gateway=gateway,
    )

    user_prompt = gateway.generate_json.await_args.kwargs["user_prompt"]
    assert "Confused recursion" in user_prompt
    assert "Unclear pronoun referents" in user_prompt
