from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from abridgeai.features.quizzes.routers import authoring as authoring_router
from abridgeai.features.quizzes.routers import authoring_admin, authoring_audit, authoring_extended
from abridgeai.features.quizzes.services import audit as audit_service
from abridgeai.features.quizzes.services import generation
from abridgeai.features.spaced_repetition.services import remediation


class _Rows:
    def __init__(self, rows: list[tuple] | None = None, mappings: list[dict] | None = None):
        self._rows = rows or []
        self._mappings = mappings or []

    def all(self):
        return self._rows

    def first(self):
        return self._mappings[0] if self._mappings else None

    def mappings(self):
        return self


@pytest.mark.asyncio
async def test_authoring_contacts_and_attempt_projection(monkeypatch: pytest.MonkeyPatch) -> None:
    student_id = uuid4()
    db = SimpleNamespace(execute=AsyncMock(return_value=_Rows([(student_id, "Ada", "ada@example.test", "b", "k")])))
    stream = AsyncMock(return_value=("https://cdn/avatar", None))
    monkeypatch.setattr(authoring_router, "create_stream_url", stream)

    contacts = await authoring_router._resolve_student_contacts(db, {student_id}, include_avatar=True)
    assert contacts[student_id] == ("Ada", "ada@example.test", "https://cdn/avatar")
    stream.assert_awaited_once()

    attempt = SimpleNamespace(
        id=uuid4(),
        quiz_id=uuid4(),
        student_id=student_id,
        attempt_number=2,
        status="graded",
        started_at=datetime.now(UTC),
        submitted_at=datetime.now(UTC),
        time_taken_seconds=12,
        score_percent=Decimal("88.50"),
        passed=True,
        integrity_score=4,
        integrity_policy_snapshot={"score_threshold": 3},
        integrity_warning_issued=True,
    )
    projected = authoring_router._attempt_teacher_view(attempt, "Quiz", "Ada", 2)
    assert projected.student_name == "Ada"
    assert projected.integrity_score_threshold == 3
    assert projected.integrity_flagged is True


@pytest.mark.asyncio
async def test_quiz_audit_route_filters_and_resolves_actor(monkeypatch: pytest.MonkeyPatch) -> None:
    quiz_id = uuid4()
    actor_id = uuid4()
    now = datetime.now(UTC)
    event = SimpleNamespace(
        id=uuid4(),
        event_name="answer_graded",
        quiz_id=quiz_id,
        actor_user_id=actor_id,
        actor_name=None,
        actor_email=None,
        subject_attempt_id=None,
        subject_question_id=None,
        subject_user_id=None,
        payload_json={"score": 1},
        occurred_at=now,
    )
    monkeypatch.setattr(
        audit_service,
        "list_events_for_quiz",
        AsyncMock(return_value=[event]),
    )
    monkeypatch.setattr(
        authoring_audit,
        "_resolve_student_contacts",
        AsyncMock(return_value={actor_id: ("Teacher", "teacher@example.test", None)}),
    )

    result = await authoring_audit.list_quiz_audit_events(
        quiz_id,
        None,
        SimpleNamespace(),
        event_name="answer_graded",
        page=0,
        page_size=10,
    )
    assert result.total == 1
    assert result.items[0].actor_name == "Teacher"
    assert result.items[0].actor_email == "teacher@example.test"


@pytest.mark.asyncio
async def test_authoring_admin_projection_and_download() -> None:
    run = SimpleNamespace(
        id=uuid4(),
        quiz_id=uuid4(),
        status="dry_run",
        attempts_affected=2,
        answers_changed=1,
        created_at=datetime.now(UTC),
        committed_at=None,
        items=[],
    )
    projected = authoring_admin._serialize_regrade_run(run)
    assert projected.status == "dry_run"
    assert projected.items == []

    response = authoring_admin._report_download(
        ["Student", "Grade"], [["Ada", Decimal("92.50")]], "csv", filename_stem="quiz-report"
    )
    assert response.media_type == "text/csv"
    assert "quiz-report-" in response.headers["content-disposition"]


def test_authoring_extended_attr_shim_and_deep_report_filters() -> None:
    shim = authoring_extended._AttrShim(
        {"title": "Quiz", "browser_security": "retired", "nested": {"value": 3}}
    )
    assert shim.model_dump() == {"title": "Quiz", "nested": {"value": 3}}
    assert shim.nested.value == 3
    with pytest.raises(AttributeError):
        _ = shim.missing


@pytest.mark.asyncio
async def test_generation_config_helpers_and_coverage_precompute(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = uuid4(), uuid4()
    config = {
        "target_outcome_ids": [str(first), "not-a-uuid", str(second)],
        "source_lesson_ids": [str(first)],
        "question_count": 4,
        "coverage_options": {"section_grouping": "fixed", "slides_per_section": 2},
    }
    assert generation._config_uuid(config, "target_outcome_ids") is None
    assert generation._config_uuid_list(config, "target_outcome_ids") == [first, second]
    merged = generation._inject_outcome_guidance({"extra_instructions": "Be precise"}, ["L.O.1: Basics"])
    assert "Be precise" in merged["extra_instructions"]
    assert "L.O.1: Basics" in merged["extra_instructions"]

    outlines = [SimpleNamespace(section_id="s1")]
    build = AsyncMock(return_value=outlines)
    budget = {"s1": 4}
    def allocate(*args: object, **kwargs: object) -> dict[str, int]:
        del args, kwargs
        return budget
    monkeypatch.setattr(generation, "build_lesson_outline", build)
    monkeypatch.setattr(generation, "allocate_question_budget", allocate)
    result = await generation._precompute_coverage_inputs(SimpleNamespace(), config)
    assert result == (outlines, budget)
    build.assert_awaited_once()


@pytest.mark.asyncio
async def test_generation_entrypoint_marks_success(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = uuid4()
    quiz_id = uuid4()
    run = SimpleNamespace(
        id=run_id,
        requested_by=uuid4(),
        course_id=uuid4(),
        config_json={"quiz_id": str(quiz_id)},
        status="pending",
        started_at=None,
        finished_at=None,
    )
    quiz = SimpleNamespace(id=quiz_id, title="Quiz", status="draft")
    db = SimpleNamespace(
        get=AsyncMock(side_effect=[run, quiz]),
        refresh=AsyncMock(),
        commit=AsyncMock(),
        rollback=AsyncMock(),
    )
    pipeline = AsyncMock()
    notify = AsyncMock()
    monkeypatch.setattr(generation.full_pipeline, "run_full_pipeline", pipeline)
    monkeypatch.setattr(generation, "_resolve_outcome_texts", AsyncMock(return_value=[]))
    monkeypatch.setattr(generation, "notify_quiz_generation_outcome", notify)

    await generation.run_quiz_generation(db, run_id)

    assert run.status == "completed"
    pipeline.assert_awaited_once()
    notify.assert_awaited_once()
    assert db.commit.await_count >= 2


@pytest.mark.asyncio
async def test_remediation_links_and_empty_context(monkeypatch: pytest.MonkeyPatch) -> None:
    lesson_id, material_id = uuid4(), uuid4()
    assert remediation.build_deep_link(
        course_slug="course",
        lesson_id=lesson_id,
        material_id=material_id,
        material_type="video",
        source_location={"timestamp_start_ms": 2500},
    ).endswith("?t=2")
    assert remediation.build_deep_link(
        course_slug="course",
        lesson_id=lesson_id,
        material_id=material_id,
        material_type="pdf",
        source_location={"page_number": 7},
    ).endswith("?p=7")
    assert remediation.build_deep_link(
        course_slug="course",
        lesson_id=lesson_id,
        material_id=material_id,
        material_type="html",
        source_location={"anchor": "#intro"},
    ).endswith("#intro")
    assert remediation._extract_chunk_ids([str(material_id), {"id": "bad"}, {"chunk_id": str(lesson_id)}]) == [
        material_id,
        lesson_id,
    ]

    monkeypatch.setattr(remediation, "_load_question_context", AsyncMock(return_value=None))
    send = AsyncMock()
    monkeypatch.setattr(remediation, "send_notification", send)
    await remediation.dispatch_remediation_for_card_failure(
        SimpleNamespace(), student_id=uuid4(), question_id=uuid4(), quiz_attempt_id=None
    )
    send.assert_not_awaited()


@pytest.mark.asyncio
async def test_authoring_router_validation_and_empty_read_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    course_id, quiz_id = uuid4(), uuid4()
    db = SimpleNamespace(get=AsyncMock(return_value=None), commit=AsyncMock())
    actor = SimpleNamespace(user_id=uuid4())

    with pytest.raises(Exception) as missing:
        await authoring_router.create_quiz_under_course(course_id, {}, actor, db)
    assert getattr(missing.value, "status_code", None) == 400

    with pytest.raises(Exception) as malformed:
        await authoring_router.create_quiz_under_course(course_id, {"module_id": "bad"}, actor, db)
    assert getattr(malformed.value, "status_code", None) == 400

    with pytest.raises(Exception) as absent:
        await authoring_router.get_quiz_authoring(quiz_id, actor, db)
    assert getattr(absent.value, "status_code", None) == 404

    from abridgeai.features.quizzes.queries import analytics as analytics_queries

    monkeypatch.setattr(
        analytics_queries,
        "list_attempts_for_course",
        AsyncMock(return_value=SimpleNamespace(items=[], next_cursor=None)),
    )
    monkeypatch.setattr(authoring_router, "_resolve_student_names", AsyncMock(return_value={}))
    page = await authoring_router.list_course_quiz_attempts(course_id, actor, db)
    assert page.items == []
    assert page.next_cursor is None

    monkeypatch.setattr(
        analytics_queries,
        "course_assessment_facets",
        AsyncMock(
            return_value={
                "quiz_attempt_count": 0,
                "quiz_pass_rate": None,
                "quiz_titles": [],
            }
        ),
    )
    monkeypatch.setattr(analytics_queries, "course_quiz_student_ids", AsyncMock(return_value=set()))
    from abridgeai.features.interviews.api import public as interviews_api

    monkeypatch.setattr(
        interviews_api,
        "course_interview_facets",
        AsyncMock(return_value={"session_count": 0, "student_ids": set(), "titles": []}),
    )
    summary = await authoring_router.course_assessment_summary(course_id, actor, db)
    assert summary.students_assessed == 0
    assert summary.quiz_titles == []


@pytest.mark.asyncio
async def test_authoring_admin_error_and_empty_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    quiz_id, run_id = uuid4(), uuid4()
    actor = SimpleNamespace(user_id=uuid4())
    db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock(), refresh=AsyncMock())

    from abridgeai.features.quizzes.services import regrade as regrade_service

    monkeypatch.setattr(regrade_service, "compute_regrade", AsyncMock(side_effect=RuntimeError("missing")))
    with pytest.raises(RuntimeError, match="missing"):
        await authoring_admin.regrade_dry_run(quiz_id, SimpleNamespace(attempt_ids=[], question_ids=[]), actor, db)

    monkeypatch.setattr(regrade_service, "get_regrade_run", AsyncMock(return_value=None))
    with pytest.raises(Exception) as missing:
        await authoring_admin.get_regrade_run(quiz_id, run_id, actor, db)
    assert getattr(missing.value, "status_code", None) == 404

    from abridgeai.features.quizzes.services import manual_grading

    monkeypatch.setattr(manual_grading, "list_needs_grading", AsyncMock(return_value=[]))
    queue = await authoring_admin.list_needs_grading(quiz_id, actor, db)
    assert queue.total == 0
    assert queue.items == []

    from abridgeai.features.quizzes.queries import overrides as override_queries

    monkeypatch.setattr(override_queries, "list_overrides", AsyncMock(return_value=[]))
    assert await authoring_admin.list_quiz_overrides(quiz_id, actor, db) == []

    from abridgeai.features.quizzes.services import feedback as feedback_service

    monkeypatch.setattr(feedback_service, "list_bands", AsyncMock(return_value=[]))
    assert await authoring_admin.list_feedback_bands(quiz_id, actor, db) == []

    from abridgeai.features.quizzes.services import gradebook

    monkeypatch.setattr(gradebook, "list_quiz_grades", AsyncMock(return_value=[]))
    monkeypatch.setattr(authoring_admin, "_resolve_student_contacts", AsyncMock(return_value={}))
    gradebook_page = await authoring_admin.get_quiz_gradebook(quiz_id, actor, db)
    assert gradebook_page.total == 0
    export = await authoring_admin.export_quiz_gradebook(quiz_id, actor, db, format="csv")
    assert export.media_type == "text/csv"

    with pytest.raises(Exception) as invalid:
        await authoring_admin.import_questions_from_file(
            quiz_id, SimpleNamespace(content="x", format="bad"), actor, db
        )
    assert getattr(invalid.value, "status_code", None) == 400


@pytest.mark.asyncio
async def test_extended_results_and_mutation_error_branches(monkeypatch: pytest.MonkeyPatch) -> None:
    quiz_id = uuid4()
    actor = SimpleNamespace(user_id=uuid4())
    db = SimpleNamespace(get=AsyncMock(return_value=None), commit=AsyncMock())

    with pytest.raises(Exception) as missing:
        await authoring_extended.get_quiz_results(quiz_id, actor, db)
    assert getattr(missing.value, "status_code", None) == 404

    from abridgeai.features.quizzes.queries import analytics as analytics_queries

    monkeypatch.setattr(analytics_queries, "quiz_question_breakdown", AsyncMock(return_value=[]))
    question_page = await authoring_extended.get_quiz_results_questions(
        quiz_id, actor, db, page=0, page_size=10
    )
    assert question_page.total == 0

    from abridgeai.features.quizzes.services import authoring as authoring_service

    monkeypatch.setattr(
        authoring_service,
        "update_quiz",
        AsyncMock(side_effect=authoring_service.NotFoundError("missing")),
    )
    with pytest.raises(Exception) as missing_update:
        await authoring_extended.update_quiz(quiz_id, {}, actor, db)
    assert getattr(missing_update.value, "status_code", None) == 404

    with pytest.raises(Exception) as bad_format:
        await authoring_admin.export_quiz_gradebook(quiz_id, actor, db, format="pdf")
    assert getattr(bad_format.value, "status_code", None) == 400


@pytest.mark.asyncio
async def test_generation_entrypoint_records_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    run_id, quiz_id = uuid4(), uuid4()
    run = SimpleNamespace(
        id=run_id,
        requested_by=uuid4(),
        course_id=uuid4(),
        config_json={"quiz_id": str(quiz_id)},
        status="pending",
        started_at=None,
        finished_at=None,
    )
    quiz = SimpleNamespace(id=quiz_id, title="Quiz", status="draft")
    db = SimpleNamespace(
        get=AsyncMock(side_effect=[run, quiz, run]),
        refresh=AsyncMock(),
        commit=AsyncMock(),
        rollback=AsyncMock(),
    )
    monkeypatch.setattr(generation, "_resolve_outcome_texts", AsyncMock(return_value=[]))
    monkeypatch.setattr(
        generation.full_pipeline,
        "run_full_pipeline",
        AsyncMock(side_effect=RuntimeError("LLM unavailable")),
    )
    monkeypatch.setattr(generation, "notify_quiz_generation_outcome", AsyncMock())

    with pytest.raises(RuntimeError, match="LLM unavailable"):
        await generation.run_quiz_generation(db, run_id)
    assert run.status == "failed"
    assert run.config_json["failure"]["message"] == "LLM unavailable"
    db.rollback.assert_awaited_once()


@pytest.mark.asyncio
async def test_remediation_resolves_deduplicated_resources_and_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    course_id, lesson_id = uuid4(), uuid4()
    chunk_a, chunk_b = uuid4(), uuid4()
    material_a, material_b = uuid4(), uuid4()
    version_a, version_b = uuid4(), uuid4()
    rows = [
        {
            "chunk_id": chunk_a,
            "chunk_index": 2,
            "chunk_metadata": {"page": 4},
            "lesson_id": lesson_id,
            "material_version_id": version_a,
            "material_id": material_a,
            "material_title": "Slides",
            "material_type": "pdf",
            "lesson_title": "Lesson",
            "course_slug": "course",
        },
        {
            "chunk_id": chunk_b,
            "chunk_index": 3,
            "chunk_metadata": {"page": 5},
            "lesson_id": lesson_id,
            "material_version_id": version_a,
            "material_id": material_a,
            "material_title": "Slides",
            "material_type": "pdf",
            "lesson_title": "Lesson",
            "course_slug": "course",
        },
        {
            "chunk_id": uuid4(),
            "chunk_index": 1,
            "chunk_metadata": {},
            "lesson_id": lesson_id,
            "material_version_id": version_b,
            "material_id": material_b,
            "material_title": "Video",
            "material_type": "video",
            "lesson_title": "Lesson",
            "course_slug": "course",
        },
    ]

    class _Result:
        def mappings(self):
            return self

        def all(self):
            return rows

    db = SimpleNamespace(execute=AsyncMock(return_value=_Result()))
    resources = await remediation._resolve_chunks_to_materials(
        db, chunk_ids=[chunk_a, chunk_b], course_id=course_id
    )
    assert [resource.material_id for resource in resources] == [material_a, material_b]
    assert resources[0].deep_link.endswith("?p=4")

    context = {
        "course_slug": "course",
        "course_id": course_id,
        "source_refs": [{"chunk_id": str(chunk_a)}],
    }
    monkeypatch.setattr(remediation, "_load_question_context", AsyncMock(return_value=context))
    monkeypatch.setattr(remediation, "_concepts_for_chunks", AsyncMock(return_value=["graphs"]))
    monkeypatch.setattr(remediation, "organization_id_for_course", AsyncMock(return_value=uuid4()))
    monkeypatch.setattr(
        remediation,
        "retrieve_kg_context_for_anchors",
        AsyncMock(return_value=SimpleNamespace(concepts=[SimpleNamespace(name="graphs")])),
    )
    monkeypatch.setattr(remediation, "_chunks_for_concepts", AsyncMock(return_value=[chunk_b]))
    monkeypatch.setattr(remediation, "_resolve_chunks_to_materials", AsyncMock(return_value=resources[:1]))
    monkeypatch.setattr(
        remediation,
        "send_notification",
        AsyncMock(),
    )
    monkeypatch.setattr(
        "abridgeai.features.identity.api.public.get_user_locale",
        AsyncMock(return_value="en"),
    )
    await remediation.dispatch_remediation_for_card_failure(
        SimpleNamespace(), student_id=uuid4(), question_id=uuid4(), quiz_attempt_id=None
    )
    remediation.send_notification.assert_awaited_once()
