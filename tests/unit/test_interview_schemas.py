from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from abridgeai.features.interviews.schemas import (
    GapReportAuthoringRead,
    GapReportRead,
    InterviewConfigAuthoring,
    InterviewConfigCreate,
    InterviewConfigPublic,
    InterviewConfigUpdate,
    InterviewForAuthoringPublic,
    InterviewForTakingPublic,
    InterviewGenerationRequest,
    InterviewGenerationRunPublic,
    InterviewOutcomeAuthoring,
    InterviewOutcomeCreate,
    InterviewOutcomePublic,
    InterviewQuestionAuthoring,
    InterviewQuestionCreate,
    InterviewQuestionPublic,
    InterviewSessionFinishResponse,
    InterviewSessionPublic,
    InterviewSessionStartRequest,
    InterviewSessionStartResponse,
    InterviewSubmitAnswerRequest,
    InterviewSubmitAnswerResponse,
    StudyPlanItem,
)

_PUBLIC_FORBIDDEN_QUESTION = frozenset(
    {
        "difficulty",
        "review_status",
        "ai_generated",
        "source_refs_json",
        "reviewed_by",
        "reviewed_at",
        "linked_outcome_id",
        "position",
        "interview_config_id",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
    }
)

_PUBLIC_FORBIDDEN_OUTCOME = frozenset(
    {
        "importance_weight",
        "interview_config_id",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
    }
)

_PUBLIC_FORBIDDEN_CONFIG = frozenset(
    {
        "supplementary_instructions",
        "generation_run_id",
        "min_outcomes_to_pass",
        "draft_question_count",
        "total_importance_weight",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
    }
)

_STUDENT_REPORT_FORBIDDEN = frozenset(
    {
        "raw_evaluation_json",
        "teacher_summary",
        "source_quiz_attempt_id",
        "source_interview_session_id",
    }
)


def test_no_hint_leak_in_public_question() -> None:
    leaked = _PUBLIC_FORBIDDEN_QUESTION & set(InterviewQuestionPublic.model_fields.keys())
    assert not leaked, f"InterviewQuestionPublic leaks fields: {leaked}"
    assert "difficulty" not in InterviewQuestionPublic.model_fields
    assert "review_status" not in InterviewQuestionPublic.model_fields
    assert "source_refs_json" not in InterviewQuestionPublic.model_fields
    assert "ai_generated" not in InterviewQuestionPublic.model_fields


def test_no_weight_leak_in_public_outcome() -> None:
    leaked = _PUBLIC_FORBIDDEN_OUTCOME & set(InterviewOutcomePublic.model_fields.keys())
    assert not leaked, f"InterviewOutcomePublic leaks fields: {leaked}"
    assert "importance_weight" not in InterviewOutcomePublic.model_fields


def test_no_internal_leak_in_public_config() -> None:
    leaked = _PUBLIC_FORBIDDEN_CONFIG & set(InterviewConfigPublic.model_fields.keys())
    assert not leaked, f"InterviewConfigPublic leaks fields: {leaked}"


def test_no_internal_leak_in_student_gap_report() -> None:
    leaked = _STUDENT_REPORT_FORBIDDEN & set(GapReportRead.model_fields.keys())
    assert not leaked, f"GapReportRead (student) leaks fields: {leaked}"


def test_authoring_inherits_public() -> None:
    assert issubclass(InterviewQuestionAuthoring, InterviewQuestionPublic)
    assert issubclass(InterviewOutcomeAuthoring, InterviewOutcomePublic)
    assert issubclass(InterviewConfigAuthoring, InterviewConfigPublic)
    assert issubclass(GapReportAuthoringRead, GapReportRead)

    public_q_fields = set(InterviewQuestionPublic.model_fields.keys())
    auth_q_fields = set(InterviewQuestionAuthoring.model_fields.keys())
    assert public_q_fields.issubset(auth_q_fields)

    public_o_fields = set(InterviewOutcomePublic.model_fields.keys())
    auth_o_fields = set(InterviewOutcomeAuthoring.model_fields.keys())
    assert public_o_fields.issubset(auth_o_fields)

    public_c_fields = set(InterviewConfigPublic.model_fields.keys())
    auth_c_fields = set(InterviewConfigAuthoring.model_fields.keys())
    assert public_c_fields.issubset(auth_c_fields)


def test_authoring_question_includes_hidden_fields() -> None:
    assert "difficulty" in InterviewQuestionAuthoring.model_fields
    assert "review_status" in InterviewQuestionAuthoring.model_fields
    assert "ai_generated" in InterviewQuestionAuthoring.model_fields
    assert "source_refs_json" in InterviewQuestionAuthoring.model_fields
    assert "reviewed_by" in InterviewQuestionAuthoring.model_fields
    assert "reviewed_at" in InterviewQuestionAuthoring.model_fields


def test_authoring_outcome_includes_importance_weight() -> None:
    assert "importance_weight" in InterviewOutcomeAuthoring.model_fields


def test_session_start_response_has_first_question_only() -> None:
    fields = InterviewSessionStartResponse.model_fields
    assert "first_question" in fields
    assert "questions" not in fields


def test_submit_answer_response_has_next_question_singular() -> None:
    fields = InterviewSubmitAnswerResponse.model_fields
    assert "next_question" in fields
    assert "questions" not in fields
    assert "is_finished" in fields


def test_for_taking_public_has_first_question_only() -> None:
    fields = InterviewForTakingPublic.model_fields
    assert "first_question" in fields
    assert "questions" not in fields
    assert "outcomes" not in fields


def test_config_public_status_narrows_to_published() -> None:
    course_id = uuid4()
    module_id = uuid4()
    with pytest.raises(ValidationError):
        InterviewConfigPublic(
            id=uuid4(),
            course_id=course_id,
            module_id=module_id,
            title="Draft",
            status="draft",  # type: ignore[arg-type]
        )

    ok = InterviewConfigPublic(
        id=uuid4(),
        course_id=course_id,
        module_id=module_id,
        title="Live",
        status="published",
    )
    assert ok.status == "published"


def test_config_authoring_widens_status() -> None:
    course_id = uuid4()
    module_id = uuid4()
    now = datetime.now(UTC)
    for status in ("draft", "published", "archived"):
        dto = InterviewConfigAuthoring(
            id=uuid4(),
            course_id=course_id,
            module_id=module_id,
            title="x",
            status=status,  # type: ignore[arg-type]
            created_at=now,
            updated_at=now,
        )
        assert dto.status == status


def test_persona_literal_rejects_friendly() -> None:
    course_id = uuid4()
    module_id = uuid4()
    with pytest.raises(ValidationError):
        InterviewConfigPublic(
            id=uuid4(),
            course_id=course_id,
            module_id=module_id,
            title="x",
            status="published",
            persona="friendly",  # type: ignore[arg-type]
        )

    for persona in ("strict", "neutral", "supportive"):
        ok = InterviewConfigPublic(
            id=uuid4(),
            course_id=course_id,
            module_id=module_id,
            title="x",
            status="published",
            persona=persona,  # type: ignore[arg-type]
        )
        assert ok.persona == persona


def test_session_status_literal_5_values() -> None:
    valid = ("in_progress", "completed", "timed_out", "abandoned", "failed")
    now = datetime.now(UTC)
    for status in valid:
        dto = InterviewSessionPublic(
            session_id=uuid4(),
            interview_config_id=uuid4(),
            status=status,  # type: ignore[arg-type]
            input_mode="text",
            attempt_number=1,
            started_at=now,
        )
        assert dto.status == status

    with pytest.raises(ValidationError):
        InterviewSessionPublic(
            session_id=uuid4(),
            interview_config_id=uuid4(),
            status="expired",  # type: ignore[arg-type]
            input_mode="text",
            attempt_number=1,
            started_at=now,
        )


def test_input_mode_literal_three_values() -> None:
    for mode in ("voice", "text", "hybrid"):
        dto = InterviewSessionStartRequest(input_mode=mode)  # type: ignore[arg-type]
        assert dto.input_mode == mode

    with pytest.raises(ValidationError):
        InterviewSessionStartRequest(input_mode="audio")  # type: ignore[arg-type]


def test_question_type_literal_five_values() -> None:
    for qtype in (
        "conceptual",
        "behavioral",
        "technical",
        "situational",
        "system_design",
    ):
        dto = InterviewQuestionPublic(
            id=uuid4(),
            prompt_text="x",
            question_type=qtype,  # type: ignore[arg-type]
        )
        assert dto.question_type == qtype

    with pytest.raises(ValidationError):
        InterviewQuestionPublic(
            id=uuid4(),
            prompt_text="x",
            question_type="multiple_choice",  # type: ignore[arg-type]
        )


def test_create_schemas_basic_validation() -> None:
    config = InterviewConfigCreate(
        title="ML Foundations",
        course_id=uuid4(),
        module_id=uuid4(),
        persona="strict",
        time_limit_minutes=30,
        max_attempts=3,
        cooldown_hours=24,
        supplementary_instructions="Probe deeply on edge cases",
    )
    assert config.title == "ML Foundations"
    assert config.persona == "strict"
    # FR-5.3 retake cooldown knob is accepted at config creation.
    assert config.cooldown_hours == 24

    # ``status`` is NOT a patchable field — transitions go through the
    # dedicated /publish|/archive|/unarchive endpoints (extra="forbid").
    update = InterviewConfigUpdate(title="Renamed", cooldown_hours=12)
    assert update.title == "Renamed"
    assert update.cooldown_hours == 12
    assert update.persona is None

    outcome = InterviewOutcomeCreate(
        position=1,
        outcome_text="Explains gradient descent",
        outcome_type="knowledge",
        importance_weight=4,
    )
    assert outcome.importance_weight == 4

    question = InterviewQuestionCreate(
        prompt_text="Walk me through SGD",
        question_type="conceptual",
        difficulty="mid_level",
        position=1,
    )
    assert question.difficulty == "mid_level"

    gen = InterviewGenerationRequest(
        course_id=uuid4(),
        module_id=uuid4(),
        question_count=8,
    )
    assert gen.question_count == 8

    submit = InterviewSubmitAnswerRequest(
        session_id=uuid4(),
        session_question_id=uuid4(),
        answer_text="Because the loss surface is non-convex",
    )
    assert submit.answer_text is not None
    assert submit.audio_object_id is None


def test_create_schemas_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        InterviewConfigCreate.model_validate(
            {
                "title": "x",
                "course_id": uuid4(),
                "module_id": uuid4(),
                "extra_field": "boom",
            }
        )

    with pytest.raises(ValidationError):
        InterviewGenerationRequest.model_validate(
            {
                "course_id": uuid4(),
                "module_id": uuid4(),
                "unexpected": True,
            }
        )


def test_generation_request_rejects_the_removed_mode_field() -> None:
    """``mode`` was removed (2026-08-30): no stage ever read it.

    ``extra="forbid"`` turns a stale client into a loud 422 rather than a run
    that silently ignores the teacher's three-way choice.
    """
    with pytest.raises(ValidationError):
        InterviewGenerationRequest.model_validate(
            {
                "mode": "coverage",
                "course_id": uuid4(),
                "module_id": uuid4(),
            }
        )


def test_generation_request_question_count_bounds() -> None:
    course_id = uuid4()
    module_id = uuid4()
    ok = InterviewGenerationRequest(
        course_id=course_id,
        module_id=module_id,
        question_count=10,
    )
    assert ok.question_count == 10

    with pytest.raises(ValidationError):
        InterviewGenerationRequest(
                course_id=course_id,
            module_id=module_id,
            question_count=0,
        )

    with pytest.raises(ValidationError):
        InterviewGenerationRequest(
                course_id=course_id,
            module_id=module_id,
            question_count=999,
        )


def test_orm_compat_via_from_attributes() -> None:
    obj = SimpleNamespace(
        id=uuid4(),
        prompt_text="Explain backprop",
        question_type="technical",
    )
    dto = InterviewQuestionPublic.model_validate(obj)
    assert dto.question_type == "technical"
    assert "difficulty" not in dto.model_dump()


def test_session_finish_response_binary_only_defaults() -> None:
    """§4.3: the student finish response is binary pass/fail. ``total_score`` /
    ``rubric_scores`` remain on the schema for API stability but default to
    None / [] — the router never populates them for the learner."""
    finish = InterviewSessionFinishResponse(
        session_id=uuid4(),
        status="completed",
        pass_verdict=True,
    )
    assert finish.pass_verdict is True
    assert finish.total_score is None
    assert finish.rubric_scores == []


def test_for_authoring_compose() -> None:
    course_id = uuid4()
    module_id = uuid4()
    now = datetime.now(UTC)
    config = InterviewConfigAuthoring(
        id=uuid4(),
        course_id=course_id,
        module_id=module_id,
        title="x",
        status="draft",
        created_at=now,
        updated_at=now,
    )
    bundle = InterviewForAuthoringPublic(config=config, outcomes=[], questions=[])
    assert bundle.config.status == "draft"


def test_for_taking_compose() -> None:
    config = InterviewConfigPublic(
        id=uuid4(),
        course_id=uuid4(),
        module_id=uuid4(),
        title="x",
        status="published",
    )
    first_q = InterviewQuestionPublic(
        id=uuid4(),
        prompt_text="x",
        question_type="conceptual",
    )
    take = InterviewForTakingPublic(config=config, first_question=first_q)
    assert take.first_question is not None
    assert take.first_question.question_type == "conceptual"
    assert "outcomes" not in take.model_dump()


def test_gap_report_authoring_inherits_student_view() -> None:
    now = datetime.now(UTC)
    student = GapReportRead(
        id=uuid4(),
        student_id=uuid4(),
        course_id=uuid4(),
        discrepancy_summary="Strong on theory, weak on application",
        study_plan=[StudyPlanItem(topic="Backprop", suggested_resources=["Lesson 4"])],
        generated_at=now,
    )
    dumped = student.model_dump()
    assert "raw_evaluation_json" not in dumped
    # FR-5.7: numeric per-criterion rubric scores must NOT reach the student.
    assert "per_criterion_breakdown" not in dumped

    teacher = GapReportAuthoringRead(
        id=uuid4(),
        student_id=uuid4(),
        course_id=uuid4(),
        discrepancy_summary="x",
        study_plan=[],
        per_criterion_breakdown={"technical_accuracy": 3.2},
        generated_at=now,
        raw_evaluation_json={"q1": {"verdict": "met", "rationale": "..."}},
        teacher_summary="Notable strength on theory.",
        source_interview_session_id=uuid4(),
    )
    assert teacher.teacher_summary is not None
    assert "rationale" in teacher.raw_evaluation_json["q1"]
    # FR-5.7: per-criterion breakdown is teacher-only and preserved here.
    assert teacher.per_criterion_breakdown["technical_accuracy"] == 3.2


def test_generation_run_orm_compat() -> None:
    obj = SimpleNamespace(
        run_id=uuid4(),
        status="running",
        config_json={"question_count": 5},
        started_at=datetime.now(UTC),
        finished_at=None,
        failure_message=None,
    )
    dto = InterviewGenerationRunPublic.model_validate(obj)
    assert dto.status == "running"


def test_submit_answer_voice_mode() -> None:
    submit = InterviewSubmitAnswerRequest(
        session_id=uuid4(),
        session_question_id=uuid4(),
        audio_object_id=uuid4(),
        latency_ms=1234,
    )
    assert submit.audio_object_id is not None
    assert submit.latency_ms == 1234


def test_submit_answer_supports_backward_compatible_turn_actions() -> None:
    legacy = InterviewSubmitAnswerRequest(
        session_id=uuid4(),
        session_question_id=uuid4(),
        answer_text="My answer",
    )
    clarification = InterviewSubmitAnswerRequest(
        session_id=uuid4(),
        session_question_id=uuid4(),
        answer_text="Could you clarify this question, please?",
        turn_action="clarify",
    )

    assert legacy.turn_action is None
    assert clarification.turn_action == "clarify"


def test_submit_answer_extra_forbidden() -> None:
    with pytest.raises(ValidationError):
        InterviewSubmitAnswerRequest.model_validate(
            {
                "session_id": uuid4(),
                "session_question_id": uuid4(),
                "answer_text": "x",
                "stuff": "boom",
            }
        )
