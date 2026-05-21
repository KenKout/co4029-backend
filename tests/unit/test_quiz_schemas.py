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
                "title": "Untitled",
                "unknown": "boom",
            }
        )


def test_generation_request_question_count_bounds() -> None:
    ok = QuizGenerationRequest(title="x", question_count=10)
    assert ok.question_count == 10

    with pytest.raises(ValidationError):
        QuizGenerationRequest(title="x", question_count=0)
    with pytest.raises(ValidationError):
        QuizGenerationRequest(title="x", question_count=999)


def test_generation_request_minimal_topic_mode_defaults() -> None:
    """A bare-minimum FR-5 payload (just ``title``) should preserve the
    legacy topic-mode defaults so existing callers don't break."""
    r = QuizGenerationRequest(title="Untitled")
    assert r.generation_mode == "topic"
    assert r.question_count == 3
    assert r.question_types == ["multiple_choice"]
    assert r.difficulty == "mixed"
    assert r.focus_topics == []
    assert r.avoid_topics == []
    assert r.append is False
    assert r.coverage_options is None


def test_generation_request_title_optional_when_quiz_id_set() -> None:
    """When the route already pins the quiz (``POST /quizzes/{id}/generate``)
    the body shouldn't have to carry a redundant title — the route's
    ``{quiz_id}`` becomes the body's ``quiz_id`` defence-in-depth and
    the title is ignored. This unblocks the SPA's regenerate flow which
    has no UX surface for an unused title."""
    quiz_id = uuid4()
    r = QuizGenerationRequest(quiz_id=quiz_id, question_count=5)
    assert r.title is None
    assert r.quiz_id == quiz_id


def test_generation_request_title_required_when_creating_new_quiz() -> None:
    """Conversely, when ``quiz_id is None`` the service has to mint a
    fresh ``Quiz`` row and needs a title, so the schema must reject
    titleless payloads up front rather than letting them fail at the
    ORM layer."""
    with pytest.raises(ValidationError) as exc_info:
        QuizGenerationRequest(question_count=5)
    msg = str(exc_info.value)
    assert "title is required" in msg


def test_generation_request_question_type_rejects_legacy_mcq() -> None:
    """The DB CHECK constraint accepts ``multiple_choice`` only; the
    legacy ``mcq`` alias is removed and must fail at the schema layer."""
    with pytest.raises(ValidationError) as exc_info:
        QuizGenerationRequest(title="x", question_types=["mcq"])  # type: ignore[list-item]
    assert exc_info.value.errors()[0]["type"] == "literal_error"


def test_generation_request_topic_strings_cleaned() -> None:
    """``focus_topics`` / ``avoid_topics`` strip whitespace, drop empties,
    and clamp each entry to 200 chars."""
    long = "x" * 300
    r = QuizGenerationRequest(
        title="x",
        focus_topics=["  vectors  ", "", "   ", "matrices", long],
        avoid_topics=["  systems  "],
    )
    assert r.focus_topics == ["vectors", "matrices", "x" * 200]
    assert r.avoid_topics == ["systems"]


def test_generation_request_topic_list_max_length() -> None:
    """At most 10 entries per topic list (post-cleanup)."""
    with pytest.raises(ValidationError):
        QuizGenerationRequest(
            title="x",
            focus_topics=[f"topic-{i}" for i in range(11)],
        )


def test_generation_request_bloom_distribution_total_capped() -> None:
    """Sum of bloom counts must be ``<= question_count``."""
    ok = QuizGenerationRequest(
        title="x",
        question_count=5,
        bloom_distribution={"remember": 2, "understand": 3},
    )
    assert sum(ok.bloom_distribution.values()) == 5

    with pytest.raises(ValidationError) as exc_info:
        QuizGenerationRequest(
            title="x",
            question_count=3,
            bloom_distribution={"remember": 5},
        )
    assert "exceeds question_count" in exc_info.value.errors()[0]["msg"]


def test_generation_request_bloom_distribution_negative_rejected() -> None:
    with pytest.raises(ValidationError):
        QuizGenerationRequest(
            title="x",
            question_count=5,
            bloom_distribution={"remember": -1},
        )


def test_generation_request_bloom_distribution_unknown_key_rejected() -> None:
    """Unknown bloom keys fail at the Pydantic boundary, not in the AI stage."""
    with pytest.raises(ValidationError):
        QuizGenerationRequest(
            title="x",
            bloom_distribution={"transcend": 1},  # type: ignore[dict-item]
        )


def test_generation_request_coverage_mode_defaults_materialised() -> None:
    """Opt-in coverage mode without explicit options gets the default
    ``CoverageOptions`` block so downstream code can rely on it."""
    r = QuizGenerationRequest(title="x", generation_mode="coverage")
    assert r.coverage_options is not None
    assert r.coverage_options.min_per_section == 1
    assert r.coverage_options.max_per_section == 5


def test_generation_request_topic_mode_keeps_options_none() -> None:
    """Topic mode does NOT materialise coverage_options."""
    r = QuizGenerationRequest(title="x", generation_mode="topic")
    assert r.coverage_options is None


def test_generation_request_full_fr5_roundtrip() -> None:
    """Full FR-5 payload survives ``model_validate -> model_dump`` cleanly."""
    payload = {
        "title": "Final Exam",
        "description": "End of unit",
        "question_count": 10,
        "question_types": ["multiple_choice", "short_answer"],
        "difficulty": "medium",
        "bloom_distribution": {"remember": 2, "understand": 3, "apply": 5},
        "include_prerequisites": True,
        "model_preference": "openai:gpt-4o-mini",
        "generation_mode": "coverage",
        "focus_topics": ["vectors", "matrices"],
        "avoid_topics": ["systems"],
        "extra_instructions": "Avoid trick questions.",
        "append": False,
        "coverage_options": {
            "min_per_section": 1,
            "max_per_section": 3,
            "skip_summaries": True,
            "slides_per_section": 4,
            "parallelism": 8,
        },
    }
    r = QuizGenerationRequest.model_validate(payload)
    dumped = r.model_dump()
    # Re-validate the dump — guarantees the schema is its own fixed point.
    r2 = QuizGenerationRequest.model_validate(dumped)
    assert r2.coverage_options is not None
    assert r2.coverage_options.parallelism == 8
    assert r2.bloom_distribution["apply"] == 5
    assert r2.focus_topics == ["vectors", "matrices"]


def test_coverage_options_min_le_max_validator() -> None:
    from abridgeai.features.quizzes.schemas import CoverageOptions  # noqa: PLC0415

    ok = CoverageOptions(min_per_section=2, max_per_section=5)
    assert ok.min_per_section == 2

    with pytest.raises(ValidationError) as exc_info:
        CoverageOptions(min_per_section=8, max_per_section=2)
    assert "min_per_section cannot exceed max_per_section" in str(exc_info.value)


def test_coverage_options_parallelism_bounds() -> None:
    from abridgeai.features.quizzes.schemas import CoverageOptions  # noqa: PLC0415

    assert CoverageOptions(parallelism=None).parallelism is None
    assert CoverageOptions(parallelism=1).parallelism == 1
    assert CoverageOptions(parallelism=32).parallelism == 32
    with pytest.raises(ValidationError):
        CoverageOptions(parallelism=0)
    with pytest.raises(ValidationError):
        CoverageOptions(parallelism=33)


def test_coverage_options_extras_rejected() -> None:
    from abridgeai.features.quizzes.schemas import CoverageOptions  # noqa: PLC0415

    with pytest.raises(ValidationError):
        CoverageOptions.model_validate(
            {"min_per_section": 1, "max_per_section": 5, "unknown": "x"}
        )


def test_question_regeneration_request_extras_rejected() -> None:
    from abridgeai.features.quizzes.schemas import QuestionRegenerationRequest  # noqa: PLC0415

    ok = QuestionRegenerationRequest(question_id=uuid4())
    assert ok.question_id is not None
    with pytest.raises(ValidationError):
        QuestionRegenerationRequest.model_validate(
            {"question_id": str(uuid4()), "wat": 1}
        )


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
