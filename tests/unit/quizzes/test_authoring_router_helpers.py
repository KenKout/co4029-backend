from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.responses import Response, StreamingResponse

from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.features.quizzes.routers import authoring


def test_http_error_helpers_keep_stable_machine_readable_details():
    item_id = uuid4()
    missing = authoring._not_found("quiz", item_id)
    bad = authoring._bad_request("invalid")
    conflict = authoring._conflict("locked")

    assert missing.status_code == 404
    assert missing.detail == {"error": "not_found", "resource": "quiz", "id": str(item_id)}
    assert bad.status_code == 400
    assert bad.detail == {"error": "bad_request", "message": "invalid"}
    assert conflict.status_code == 409
    assert conflict.detail == {"error": "conflict", "message": "locked"}


def test_attr_shim_filters_retired_keys_and_wraps_nested_values():
    shim = authoring._AttrShim(
        {
            "title": "Quiz",
            "browser_security": True,
            "nested": {"value": 3},
            "options": [{"option_text": "A"}],
        }
    )

    assert shim.title == "Quiz"
    assert shim.browser_security is None
    assert shim.nested.value == 3
    assert shim.options[0].option_text == "A"
    assert shim.model_dump(include={"title", "nested"}) == {
        "title": "Quiz",
        "nested": {"value": 3},
    }
    assert shim.model_dump(exclude={"nested"}) == {
        "title": "Quiz",
        "options": [{"option_text": "A"}],
    }


def test_attempt_teacher_view_defaults_and_integrity_snapshot():
    attempt = SimpleNamespace(
        id=uuid4(),
        quiz_id=uuid4(),
        student_id=uuid4(),
        attempt_number=2,
        status="submitted",
        started_at=datetime(2026, 9, 16, tzinfo=UTC),
        submitted_at=None,
        time_taken_seconds=120,
        score_percent=Decimal("75"),
        passed=True,
        integrity_score=None,
        integrity_policy_snapshot={"score_threshold": 6},
        integrity_warning_issued=True,
    )

    result = authoring._attempt_teacher_view(attempt, "Quiz", "Student", 3)

    assert result.quiz_title == "Quiz"
    assert result.student_name == "Student"
    assert result.integrity_flags == 3
    assert result.integrity_score == 0
    assert result.integrity_score_threshold == 6
    assert result.integrity_flagged is True


def test_generation_run_view_projects_failure_and_tolerates_bad_progress():
    quiz_id = uuid4()
    created_at = datetime(2026, 9, 16, tzinfo=UTC)
    run = SimpleNamespace(
        id=uuid4(),
        status="failed",
        started_at=None,
        created_at=created_at,
        finished_at=created_at,
        config_json={"failure": {"message": "retrieval produced zero chunks"}},
        progress_json={"stage_index": "not-an-int"},
    )

    result = authoring._generation_run_view(run, quiz_id)

    assert result.quiz_id == quiz_id
    assert result.started_at == created_at
    assert result.error_message == "retrieval produced zero chunks"
    assert result.progress is None

    run.status = "running"
    run.config_json = None
    run.progress_json = {"current_stage": "generation", "stage_index": 2, "total_stages": 4}
    result = authoring._generation_run_view(run, quiz_id)
    assert result.error_message is None
    assert result.progress is not None
    assert result.progress.current_stage == "generation"


def test_fill_outcome_positions_handles_resolved_missing_and_unlinked_questions():
    linked = SimpleNamespace(learning_outcome_id=uuid4())
    deleted = SimpleNamespace(learning_outcome_id=uuid4())
    unlinked = SimpleNamespace(learning_outcome_id=None)

    authoring._fill_outcome_positions([linked, deleted, unlinked], {linked.learning_outcome_id: (2, "1.2")})

    assert linked.outcome_position == 2
    assert linked.outcome_code == "1.2"
    assert deleted.outcome_position is None
    assert deleted.outcome_code is None
    assert unlinked.outcome_position is None


def test_serialize_regrade_run_projects_each_delta():
    created_at = datetime(2026, 9, 16, tzinfo=UTC)
    item = SimpleNamespace(
        attempt_id=uuid4(),
        question_id=uuid4(),
        old_is_correct=False,
        new_is_correct=True,
        old_points=Decimal("0"),
        new_points=Decimal("1"),
    )
    run = SimpleNamespace(
        id=uuid4(),
        quiz_id=uuid4(),
        status="dry_run",
        attempts_affected=1,
        answers_changed=1,
        created_at=created_at,
        committed_at=None,
        items=[item],
    )

    result = authoring._serialize_regrade_run(run)

    assert result.id == run.id
    assert result.items[0].question_id == item.question_id
    assert result.items[0].new_points == Decimal("1")


def test_report_download_builds_csv_or_xlsx(monkeypatch):
    from abridgeai.features.quizzes.services import reports_export

    stream_csv = Mock(return_value=iter(["a,b\n", "1,2\n"]))
    build_xlsx = Mock(return_value=b"xlsx")
    monkeypatch.setattr(reports_export, "stream_csv", stream_csv)
    monkeypatch.setattr(reports_export, "build_xlsx", build_xlsx)

    csv = authoring._report_download(["a", "b"], [[1, 2]], "csv", filename_stem="report")
    xlsx = authoring._report_download(["a", "b"], [[1, 2]], "xlsx", filename_stem="report")

    assert isinstance(csv, StreamingResponse)
    assert csv.media_type == "text/csv"
    assert csv.headers["content-disposition"].endswith('.csv"')
    assert isinstance(xlsx, Response)
    assert xlsx.body == b"xlsx"
    assert xlsx.headers["content-disposition"].endswith('.xlsx"')
    stream_csv.assert_called_once_with(["a", "b"], [[1, 2]])
    build_xlsx.assert_called_once_with(["a", "b"], [[1, 2]])


@pytest.mark.asyncio
async def test_arq_pool_default_is_none():
    assert await authoring.get_arq_pool() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "status_code"),
    [(NotFoundError("missing"), 404), (ConflictError("locked"), 409), (AppError("bad"), 400)],
)
async def test_update_quiz_maps_service_errors(monkeypatch, failure: Exception, status_code: int):
    monkeypatch.setattr(
        authoring.authoring_service, "update_quiz", AsyncMock(side_effect=failure)
    )
    db = SimpleNamespace(commit=AsyncMock())

    with pytest.raises(HTTPException) as raised:
        await authoring.update_quiz(uuid4(), {"title": "New"}, object(), db)

    assert raised.value.status_code == status_code
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_set_expected_time_commits_and_returns_count(monkeypatch):
    update = AsyncMock(return_value=2)
    monkeypatch.setattr(
        authoring.authoring_service, "bulk_set_expected_response_time", update
    )
    db = SimpleNamespace(commit=AsyncMock())
    question_ids = [uuid4(), uuid4()]
    payload = authoring.BulkSetExpectedTimeRequest(
        items=[
            authoring.BulkSetItem(question_id=question_ids[0], expected_response_time_ms=1000),
            authoring.BulkSetItem(question_id=question_ids[1], expected_response_time_ms=2000),
        ]
    )
    actor = object()
    quiz_id = uuid4()

    result = await authoring.bulk_set_expected_time(quiz_id, payload, actor, db)

    assert result.updated == 2
    update.assert_awaited_once_with(
        db, quiz_id, [(question_ids[0], 1000), (question_ids[1], 2000)], actor
    )
    db.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_bulk_approve_maps_conflict(monkeypatch):
    monkeypatch.setattr(
        authoring.authoring_service,
        "bulk_approve_questions",
        AsyncMock(side_effect=ConflictError("published")),
    )
    payload = authoring.BulkApproveRequest(question_ids=[uuid4()])
    with pytest.raises(HTTPException) as raised:
        await authoring.bulk_approve_questions(uuid4(), payload, object(), object())
    assert raised.value.status_code == 409


@pytest.mark.asyncio
async def test_delete_quiz_commits_and_maps_missing(monkeypatch):
    delete = AsyncMock(return_value=None)
    monkeypatch.setattr(authoring.authoring_service, "delete_quiz", delete)
    db = SimpleNamespace(commit=AsyncMock())
    quiz_id = uuid4()
    actor = object()

    await authoring.delete_quiz(quiz_id, actor, db)
    delete.assert_awaited_once_with(db, quiz_id, actor)
    db.commit.assert_awaited_once_with()

    delete.side_effect = NotFoundError("gone")
    with pytest.raises(HTTPException) as raised:
        await authoring.delete_quiz(quiz_id, actor, db)
    assert raised.value.status_code == 404


@pytest.mark.asyncio
async def test_generation_run_endpoints_scope_results(monkeypatch):
    quiz_id = uuid4()
    run_id = uuid4()
    monkeypatch.setattr(
        authoring.authoring_service, "get_latest_generation_run", AsyncMock(return_value=None)
    )
    assert await authoring.get_latest_quiz_generation_run(quiz_id, object(), object()) is None

    db = SimpleNamespace(get=AsyncMock(return_value=None))
    with pytest.raises(HTTPException) as raised:
        await authoring.get_generation_run(quiz_id, run_id, object(), db)
    assert raised.value.status_code == 404

    db.get.return_value = SimpleNamespace(config_json={"quiz_id": str(uuid4())})
    with pytest.raises(HTTPException) as raised:
        await authoring.get_generation_run(quiz_id, run_id, object(), db)
    assert raised.value.status_code == 404

    scoped = SimpleNamespace(config_json={"quiz_id": str(quiz_id)})
    db.get.return_value = scoped
    projected = object()
    monkeypatch.setattr(authoring, "_generation_run_view", Mock(return_value=projected))
    assert await authoring.get_generation_run(quiz_id, run_id, object(), db) is projected


@pytest.mark.asyncio
async def test_import_file_validates_format_rolls_back_errors_and_commits_success(monkeypatch):
    from abridgeai.features.quizzes.services import quiz_io

    db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock())
    actor = object()
    quiz_id = uuid4()
    with pytest.raises(HTTPException) as raised:
        await authoring.import_questions_from_file(
            quiz_id, authoring._ImportBody(content="x", format="csv"), actor, db
        )
    assert raised.value.status_code == 400

    import_file = AsyncMock(side_effect=ValueError("malformed"))
    monkeypatch.setattr(quiz_io, "import_questions_from_file", import_file)
    with pytest.raises(HTTPException) as raised:
        await authoring.import_questions_from_file(
            quiz_id, authoring._ImportBody(content="x", format="gift"), actor, db
        )
    assert raised.value.status_code == 422
    db.rollback.assert_awaited_once_with()

    import_file.side_effect = AppError("unsupported")
    with pytest.raises(HTTPException) as raised:
        await authoring.import_questions_from_file(
            quiz_id, authoring._ImportBody(content="x", format="gift"), actor, db
        )
    assert raised.value.status_code == 400
    assert db.rollback.await_count == 2

    expected = {"imported": 1, "skipped": 0, "warnings": []}
    import_file.side_effect = None
    import_file.return_value = expected
    assert (
        await authoring.import_questions_from_file(
            quiz_id, authoring._ImportBody(content="x", format="gift"), actor, db
        )
        == expected
    )
    db.commit.assert_awaited_once_with()


@pytest.mark.asyncio
async def test_export_questions_validates_format_and_builds_download(monkeypatch):
    from abridgeai.features.quizzes.services import quiz_io

    quiz_id = uuid4()
    with pytest.raises(HTTPException) as raised:
        await authoring.export_quiz_questions(quiz_id, object(), object(), format="csv")
    assert raised.value.status_code == 400

    export = AsyncMock(side_effect=AppError("bad export"))
    monkeypatch.setattr(quiz_io, "export_quiz_questions", export)
    with pytest.raises(HTTPException) as raised:
        await authoring.export_quiz_questions(quiz_id, object(), object(), format="gift")
    assert raised.value.status_code == 400

    export.side_effect = None
    export.return_value = "::Question:: body"
    response = await authoring.export_quiz_questions(
        quiz_id, object(), object(), format="gift"
    )
    assert response.media_type == "text/plain"
    assert response.body == b"::Question:: body"
    assert response.headers["content-disposition"].endswith('.txt"')
