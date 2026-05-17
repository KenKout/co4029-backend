from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from pydantic import ValidationError

from abridgeai.features.quizzes.schemas import (
    QuizAttemptRead,
    QuizAttemptStart,
    QuizAttemptSubmit,
    QuizAttemptSubmitAnswer,
    QuizAuthoring,
    QuizForAuthoringPublic,
    QuizForTakingPublic,
    QuizGenerationRequest,
    QuizGenerationRunRead,
    QuizPublic,
    QuizQuestionAuthoring,
    QuizQuestionOptionAuthoring,
    QuizQuestionOptionPublic,
    QuizQuestionPublic,
)

_PUBLIC_FORBIDDEN_OPTION = frozenset(
    {
        "is_correct",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
    }
)

_PUBLIC_FORBIDDEN_QUESTION = frozenset(
    {
        "review_status",
        "original_generated_payload",
        "source_refs",
        "reviewed_by",
        "reviewed_at",
        "expected_response_time_ms",
        "expected_ef_ceiling",
        "difficulty",
        "bloom_level",
        "explanation",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
    }
)

_PUBLIC_FORBIDDEN_QUIZ = frozenset(
    {
        "internal_notes",
        "draft_count",
        "course_id",
        "module_id",
        "shuffle_questions",
        "shuffle_options",
        "initial_ef",
        "min_ef_for_unlock",
        "coverage_threshold",
        "reminders_enabled",
        "generation_instructions",
        "generation_run_id",
        "created_by",
        "updated_by",
        "deleted_at",
        "deleted_by",
    }
)


def test_public_schemas_importable() -> None:
    assert QuizPublic is not None
    assert QuizQuestionPublic is not None
    assert QuizQuestionOptionPublic is not None
    assert QuizForTakingPublic is not None


def test_public_no_is_correct_in_options() -> None:
    assert "is_correct" not in QuizQuestionOptionPublic.model_fields


def test_public_option_excludes_internal_fields() -> None:
    leaked = _PUBLIC_FORBIDDEN_OPTION & set(QuizQuestionOptionPublic.model_fields.keys())
    assert not leaked, f"QuizQuestionOptionPublic leaks fields: {leaked}"


def test_public_question_excludes_internal_fields() -> None:
    leaked = _PUBLIC_FORBIDDEN_QUESTION & set(QuizQuestionPublic.model_fields.keys())
    assert not leaked, f"QuizQuestionPublic leaks fields: {leaked}"


def test_no_internal_notes_or_draft_count_in_public_quiz() -> None:
    leaked = _PUBLIC_FORBIDDEN_QUIZ & set(QuizPublic.model_fields.keys())
    assert not leaked, f"QuizPublic leaks fields: {leaked}"


def test_authoring_includes_is_correct() -> None:
    assert "is_correct" in QuizQuestionOptionAuthoring.model_fields


def test_authoring_inherits_public() -> None:
    assert issubclass(QuizQuestionAuthoring, QuizQuestionPublic)
    assert issubclass(QuizQuestionOptionAuthoring, QuizQuestionOptionPublic)
    assert issubclass(QuizAuthoring, QuizPublic)


def test_quiz_public_status_literal_narrows_to_published() -> None:
    with pytest.raises(ValidationError):
        QuizPublic(
            id=uuid4(),
            title="Draft Quiz",
            status="draft",  # type: ignore[arg-type]
            passing_score_percent=Decimal("70.00"),
        )

    with pytest.raises(ValidationError):
        QuizPublic(
            id=uuid4(),
            title="Archived Quiz",
            status="archived",  # type: ignore[arg-type]
            passing_score_percent=Decimal("70.00"),
        )

    ok = QuizPublic(
        id=uuid4(),
        title="Live Quiz",
        status="published",
        passing_score_percent=Decimal("70.00"),
    )
    assert ok.status == "published"


def test_quiz_authoring_status_widens_to_full_enum() -> None:
    quiz_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    now = datetime.now(UTC)
    for status in ("draft", "published", "archived"):
        dto = QuizAuthoring(
            id=quiz_id,
            title="x",
            status=status,  # type: ignore[arg-type]
            passing_score_percent=Decimal("70.00"),
            course_id=course_id,
            module_id=module_id,
            created_at=now,
            updated_at=now,
        )
        assert dto.status == status


def test_question_type_literal_narrowing() -> None:
    with pytest.raises(ValidationError):
        QuizQuestionPublic(
            id=uuid4(),
            quiz_id=uuid4(),
            position=1,
            question_type="mcq",  # type: ignore[arg-type]
            prompt_text="bad type",
        )


def test_attempt_submit_answer_optional_fields() -> None:
    qid = uuid4()
    only_options = QuizAttemptSubmitAnswer(
        question_id=qid,
        selected_option_ids=[uuid4()],
    )
    assert only_options.text_answer is None
    assert only_options.hint_used is False

    only_text = QuizAttemptSubmitAnswer(question_id=qid, text_answer="42")
    assert only_text.selected_option_ids is None

    bare = QuizAttemptSubmitAnswer(question_id=qid)
    assert bare.selected_option_ids is None
    assert bare.text_answer is None


def test_attempt_submit_extras_rejected() -> None:
    with pytest.raises(ValidationError):
        QuizAttemptSubmit.model_validate(
            {
                "answers": [],
                "unexpected_field": True,
            }
        )


def test_attempt_start_optional_idempotency_key() -> None:
    quiz_id = uuid4()
    bare = QuizAttemptStart(quiz_id=quiz_id)
    assert bare.idempotency_key is None

    keyed = QuizAttemptStart(quiz_id=quiz_id, idempotency_key=uuid4())
    assert keyed.idempotency_key is not None


def test_orm_compat_via_from_attributes() -> None:
    obj = SimpleNamespace(
        id=uuid4(),
        quiz_id=uuid4(),
        position=1,
        question_type="multiple_choice",
        prompt_text="Pick one",
        hint_text=None,
        options=[
            SimpleNamespace(
                id=uuid4(),
                option_key="A",
                option_text="apple",
                position=1,
            ),
            SimpleNamespace(
                id=uuid4(),
                option_key="B",
                option_text="banana",
                position=2,
            ),
        ],
    )
    dto = QuizQuestionPublic.model_validate(obj)
    assert dto.question_type == "multiple_choice"
    assert len(dto.options) == 2
    assert dto.options[0].option_key == "A"
    assert "is_correct" not in dto.options[0].model_dump()


def test_quiz_for_taking_public_compose() -> None:
    quiz = QuizPublic(
        id=uuid4(),
        title="x",
        status="published",
        passing_score_percent=Decimal("70.00"),
    )
    take = QuizForTakingPublic(quiz=quiz, questions=[])
    assert take.quiz.id == quiz.id
    assert take.questions == []


def test_quiz_for_authoring_public_compose() -> None:
    course_id = uuid4()
    module_id = uuid4()
    now = datetime.now(UTC)
    quiz = QuizAuthoring(
        id=uuid4(),
        title="x",
        status="draft",
        passing_score_percent=Decimal("70.00"),
        course_id=course_id,
        module_id=module_id,
        created_at=now,
        updated_at=now,
    )
    bundle = QuizForAuthoringPublic(quiz=quiz, questions=[])
    assert bundle.quiz.status == "draft"


def test_generation_request_extras_rejected() -> None:
    with pytest.raises(ValidationError):
        QuizGenerationRequest.model_validate(
            {
                "mode": "full",
                "unknown": "boom",
            }
        )


def test_generation_request_target_count_bounds() -> None:
    ok = QuizGenerationRequest(mode="full", target_count=10)
    assert ok.target_count == 10

    with pytest.raises(ValidationError):
        QuizGenerationRequest(mode="full", target_count=0)
    with pytest.raises(ValidationError):
        QuizGenerationRequest(mode="full", target_count=999)


def test_generation_run_orm_compat() -> None:
    obj = SimpleNamespace(
        id=uuid4(),
        quiz_id=uuid4(),
        status="running",
        started_at=datetime.now(UTC),
        completed_at=None,
        error_message=None,
        pipeline_run_id=None,
    )
    dto = QuizGenerationRunRead.model_validate(obj)
    assert dto.status == "running"


def test_attempt_read_score_columns_optional() -> None:
    obj = SimpleNamespace(
        id=uuid4(),
        quiz_id=uuid4(),
        attempt_number=1,
        status="in_progress",
        started_at=datetime.now(UTC),
        submitted_at=None,
        graded_at=None,
        time_taken_seconds=None,
        score_points=None,
        score_percent=None,
        passed=None,
        total_questions=None,
        correct_count=None,
    )
    dto = QuizAttemptRead.model_validate(obj)
    assert dto.score_points is None
    assert dto.passed is None
