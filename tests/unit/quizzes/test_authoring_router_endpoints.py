"""What the quiz authoring endpoints do around the service call.

``test_authoring_router_helpers.py`` covers the projections and a first set
of endpoints; this file covers the rest, and it asks two questions of each:
does a failure reach the client as the right status, and does a failed
request leave the transaction uncommitted.

The second is the one worth the file. Every one of these endpoints owns its
commit -- the services flush and the router decides -- so a ``commit`` that
runs despite a refusal persists exactly the state the refusal was meant to
prevent. A quiz created against a module in someone else's course, an
override that tripped a unique constraint, a regrade committed twice: each
is a write the service already made in the session and the router is the
only thing standing between it and the database.

The services are mocked. These tests are about the router's own handling,
and the services have their own suites.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.features.quizzes.routers import authoring
from abridgeai.features.quizzes.services.authoring import QuizPublishValidationError
from abridgeai.features.quizzes.services.publish_gate import QuizApprovalRequiredError

pytestmark = pytest.mark.asyncio


def _db() -> SimpleNamespace:
    return SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock(), refresh=AsyncMock())


def _quiz(course_id: UUID | None = None) -> SimpleNamespace:
    """A quiz row carrying every field ``QuizAuthoring`` projects."""
    now = datetime.now(UTC)
    return SimpleNamespace(
        id=uuid4(),
        course_id=course_id or uuid4(),
        module_id=uuid4(),
        title="Quiz",
        slug="quiz",
        description=None,
        status="draft",
        passing_score_percent=Decimal("70.00"),
        time_limit_seconds=None,
        allow_retakes=True,
        max_attempts=None,
        cooldown_hours=None,
        shuffle_questions=False,
        shuffle_options=False,
        show_hints=True,
        require_camera=False,
        initial_ef=None,
        min_ef_for_unlock=None,
        coverage_threshold=None,
        reminders_enabled=False,
        generation_instructions=None,
        available_from=None,
        available_until=None,
        due_at=None,
        published_at=None,
        created_at=now,
        updated_at=now,
    )


class TestCreatingAQuizUnderACourse:
    """The module arrives in the body while the course comes from the URL.

    Authorisation is checked against the course, so a module from a
    different course would be a write the permission check never saw.
    """

    async def test_a_missing_module_id_is_rejected_before_any_write(self) -> None:
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.create_quiz_under_course(uuid4(), {}, object(), db)
        assert raised.value.status_code == 400
        assert raised.value.detail["message"] == "module_id is required"
        db.commit.assert_not_awaited()

    async def test_a_module_id_that_is_not_a_uuid_is_rejected(self) -> None:
        """The body is a loose dict, so this cannot be left to FastAPI."""
        with pytest.raises(HTTPException) as raised:
            await authoring.create_quiz_under_course(
                uuid4(), {"module_id": "not-a-uuid"}, object(), _db()
            )
        assert raised.value.status_code == 400

    async def test_an_unknown_module_is_reported_as_the_missing_module(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Naming the quiz here would be wrong -- no quiz was ever named."""
        monkeypatch.setattr(
            authoring.authoring_service, "create_quiz", AsyncMock(side_effect=NotFoundError("no"))
        )
        module_id = uuid4()
        with pytest.raises(HTTPException) as raised:
            await authoring.create_quiz_under_course(
                uuid4(), {"module_id": str(module_id)}, object(), _db()
            )
        assert raised.value.status_code == 404
        assert raised.value.detail["resource"] == "module"
        assert raised.value.detail["id"] == str(module_id)

    async def test_a_module_belonging_to_another_course_is_refused_uncommitted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The quiz has already been built in the session by this point.

        The check is after the service call because the module's course is
        only known once the row exists. Committing anyway would attach a
        quiz to a course whose permissions were never checked -- the reason
        the check exists at all.
        """
        monkeypatch.setattr(
            authoring.authoring_service,
            "create_quiz",
            AsyncMock(return_value=_quiz(course_id=uuid4())),
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.create_quiz_under_course(
                uuid4(), {"module_id": str(uuid4())}, object(), db
            )
        assert raised.value.status_code == 400
        assert raised.value.detail["message"] == "module does not belong to course"
        db.commit.assert_not_awaited()

    async def test_a_module_in_the_right_course_commits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        course_id = uuid4()
        monkeypatch.setattr(
            authoring.authoring_service,
            "create_quiz",
            AsyncMock(return_value=_quiz(course_id=course_id)),
        )
        db = _db()
        result = await authoring.create_quiz_under_course(
            course_id, {"module_id": str(uuid4())}, object(), db
        )
        assert result.course_id == course_id
        db.commit.assert_awaited_once_with()


class TestPublishing:
    """Both gates refuse with 422 and name the questions at fault.

    A bare refusal would leave the teacher to find the offending questions
    by hand in a quiz that may hold dozens, so the ids are part of the
    contract, not a convenience.
    """

    async def test_questions_without_an_expected_time_are_listed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        missing = [uuid4(), uuid4()]
        monkeypatch.setattr(
            authoring.authoring_service,
            "publish_quiz",
            AsyncMock(side_effect=QuizPublishValidationError(missing)),
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.publish_quiz(uuid4(), object(), db)

        assert raised.value.status_code == 422
        assert raised.value.detail["error"] == "publish_gate_t_exp_required"
        assert raised.value.detail["missing_t_exp_question_ids"] == [str(q) for q in missing]
        db.commit.assert_not_awaited()

    async def test_a_quiz_with_nothing_approved_is_a_separate_refusal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The two 422s mean different things and the client shows different
        screens: one sends the teacher to set times, the other to review."""
        monkeypatch.setattr(
            authoring.authoring_service,
            "publish_quiz",
            AsyncMock(side_effect=QuizApprovalRequiredError([])),
        )
        with pytest.raises(HTTPException) as raised:
            await authoring.publish_quiz(uuid4(), object(), _db())

        assert raised.value.status_code == 422
        assert raised.value.detail["error"] == "pending_review"
        assert raised.value.detail["pending_question_ids"] == []

    @pytest.mark.parametrize(
        ("failure", "status_code"),
        [(NotFoundError("gone"), 404), (AppError("archived"), 400)],
    )
    async def test_the_remaining_failures_keep_their_ordinary_statuses(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception, status_code: int
    ) -> None:
        monkeypatch.setattr(
            authoring.authoring_service, "publish_quiz", AsyncMock(side_effect=failure)
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.publish_quiz(uuid4(), object(), db)
        assert raised.value.status_code == status_code
        db.commit.assert_not_awaited()

    async def test_a_successful_publish_commits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            authoring.authoring_service, "publish_quiz", AsyncMock(return_value=_quiz())
        )
        db = _db()
        await authoring.publish_quiz(uuid4(), object(), db)
        db.commit.assert_awaited_once_with()


class TestArchiving:
    async def test_a_successful_archive_commits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        archive = AsyncMock(return_value=_quiz())
        monkeypatch.setattr(authoring.authoring_service, "archive_quiz", archive)
        db = _db()
        quiz_id = uuid4()
        actor = object()

        await authoring.archive_quiz(quiz_id, actor, db)

        archive.assert_awaited_once_with(db, quiz_id, actor)
        db.commit.assert_awaited_once_with()

    async def test_a_failed_archive_does_not_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            authoring.authoring_service,
            "archive_quiz",
            AsyncMock(side_effect=NotFoundError("gone")),
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.archive_quiz(uuid4(), object(), db)

        assert raised.value.status_code == 404
        db.commit.assert_not_awaited()


class TestQuestionWrites:
    """A frozen quiz must answer 409, not 400.

    The client retries a 400 after fixing the body and gives up on a 409;
    a published quiz is not something a different body will get past.
    """

    @pytest.mark.parametrize(
        ("failure", "status_code"),
        [
            (NotFoundError("gone"), 404),
            (ConflictError("quiz_published_readonly"), 409),
            (AppError("bad options"), 400),
        ],
    )
    async def test_create_maps_each_failure_without_committing(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception, status_code: int
    ) -> None:
        monkeypatch.setattr(
            authoring.authoring_service, "create_question", AsyncMock(side_effect=failure)
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.create_question(uuid4(), {"prompt_text": "Q"}, object(), db)
        assert raised.value.status_code == status_code
        db.commit.assert_not_awaited()

    @pytest.mark.parametrize(
        ("failure", "status_code"),
        [
            (NotFoundError("gone"), 404),
            (ConflictError("quiz_published_readonly"), 409),
            (AppError("bad options"), 400),
        ],
    )
    async def test_update_maps_each_failure_without_committing(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception, status_code: int
    ) -> None:
        monkeypatch.setattr(
            authoring.authoring_service, "update_question", AsyncMock(side_effect=failure)
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.update_question(uuid4(), uuid4(), {"prompt_text": "Q"}, object(), db)
        assert raised.value.status_code == status_code
        db.commit.assert_not_awaited()

    async def test_a_missing_question_is_named_as_a_question_not_a_quiz(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The client keys its error copy off the resource name."""
        monkeypatch.setattr(
            authoring.authoring_service, "update_question", AsyncMock(side_effect=NotFoundError(""))
        )
        question_id = uuid4()
        with pytest.raises(HTTPException) as raised:
            await authoring.update_question(uuid4(), question_id, {}, object(), _db())
        assert raised.value.detail["resource"] == "quiz_question"
        assert raised.value.detail["id"] == str(question_id)

    async def test_delete_commits_only_when_the_service_agreed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        delete = AsyncMock(return_value=None)
        monkeypatch.setattr(authoring.authoring_service, "delete_question", delete)
        db = _db()
        question_id = uuid4()
        actor = object()

        await authoring.delete_question(uuid4(), question_id, actor, db)
        delete.assert_awaited_once_with(db, question_id, actor)
        db.commit.assert_awaited_once_with()

        delete.side_effect = ConflictError("quiz_published_readonly")
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.delete_question(uuid4(), question_id, actor, db)
        assert raised.value.status_code == 409
        db.commit.assert_not_awaited()

    @pytest.mark.parametrize(
        ("failure", "status_code"), [(NotFoundError("gone"), 404), (AppError("no"), 400)]
    )
    async def test_duplicate_maps_its_failures(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception, status_code: int
    ) -> None:
        monkeypatch.setattr(
            authoring.question_bank_service, "duplicate_question", AsyncMock(side_effect=failure)
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.duplicate_question(uuid4(), uuid4(), object(), db)
        assert raised.value.status_code == status_code
        db.commit.assert_not_awaited()


class TestOverrides:
    """An override is an exception granted to one student.

    Creating one twice for the same target is the common mistake -- two rows
    granting different deadlines to the same person, with nothing saying
    which applies -- so the unique constraint is the real guard and the
    router's job is to turn it into a 409 the teacher can act on.
    """

    async def test_a_duplicate_target_is_rolled_back_and_reported_as_conflict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rollback matters: the failed INSERT has poisoned the
        transaction, and without it the audit write that follows would fail
        too, replacing a clear 409 with a driver error."""
        import abridgeai.features.quizzes.queries.overrides as overrides_queries

        monkeypatch.setattr(
            overrides_queries, "create_override", AsyncMock(side_effect=Exception("unique"))
        )
        db = SimpleNamespace(
            commit=AsyncMock(), rollback=AsyncMock(), refresh=AsyncMock(), flush=AsyncMock()
        )
        body = authoring.QuizOverrideIn(scope="user", user_id=uuid4())

        with pytest.raises(HTTPException) as raised:
            await authoring.create_quiz_override(uuid4(), body, object(), db)

        assert raised.value.status_code == 409
        db.rollback.assert_awaited_once_with()
        db.commit.assert_not_awaited()

    async def test_updating_an_override_that_is_gone_does_not_commit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import abridgeai.features.quizzes.queries.overrides as overrides_queries

        monkeypatch.setattr(overrides_queries, "update_override", AsyncMock(return_value=None))
        db = _db()
        override_id = uuid4()

        with pytest.raises(HTTPException) as raised:
            await authoring.update_quiz_override(
                uuid4(),
                override_id,
                authoring.QuizOverrideIn(scope="user", user_id=uuid4()),
                object(),
                db,
            )
        assert raised.value.status_code == 404
        assert raised.value.detail["resource"] == "override"
        db.commit.assert_not_awaited()

    async def test_deleting_an_override_that_is_gone_is_a_404(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Not a silent success: the teacher is told the row was not there,
        rather than believing they revoked an exception that still stands."""
        import abridgeai.features.quizzes.queries.overrides as overrides_queries

        monkeypatch.setattr(overrides_queries, "delete_override", AsyncMock(return_value=False))
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.delete_quiz_override(uuid4(), uuid4(), object(), db)
        assert raised.value.status_code == 404
        db.commit.assert_not_awaited()

    async def test_a_successful_delete_commits(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import abridgeai.features.quizzes.queries.overrides as overrides_queries
        import abridgeai.features.quizzes.services.audit as quiz_audit

        monkeypatch.setattr(overrides_queries, "delete_override", AsyncMock(return_value=True))
        audit = AsyncMock()
        monkeypatch.setattr(quiz_audit, "record_event", audit)
        db = _db()
        await authoring.delete_quiz_override(uuid4(), uuid4(), object(), db)
        audit.assert_awaited_once()
        db.commit.assert_awaited_once_with()


class TestFeedbackBands:
    """Overlapping bands are a 422, not a 400.

    The bands are edited as a grid in the form, and 422 is what the client
    renders inline against the offending row; a 400 surfaces as a generic
    failure banner with the grid unchanged.
    """

    async def test_overlapping_bands_are_unprocessable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import abridgeai.features.quizzes.services.feedback as feedback_service

        monkeypatch.setattr(
            feedback_service,
            "set_feedback_bands",
            AsyncMock(side_effect=AppError("bands overlap")),
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.set_feedback_bands(
                uuid4(), authoring._FeedbackBandsBody(bands=[]), object(), db
            )
        assert raised.value.status_code == 422
        assert "overlap" in str(raised.value.detail)
        db.commit.assert_not_awaited()

    async def test_an_unknown_quiz_is_a_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import abridgeai.features.quizzes.services.feedback as feedback_service

        monkeypatch.setattr(
            feedback_service, "set_feedback_bands", AsyncMock(side_effect=NotFoundError("gone"))
        )
        with pytest.raises(HTTPException) as raised:
            await authoring.set_feedback_bands(
                uuid4(), authoring._FeedbackBandsBody(bands=[]), object(), _db()
            )
        assert raised.value.status_code == 404

    async def test_replacing_the_bands_commits_and_projects_each_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import abridgeai.features.quizzes.services.feedback as feedback_service

        rows = [
            SimpleNamespace(
                id=uuid4(),
                min_grade=Decimal("80"),
                max_grade=Decimal("100"),
                feedback_text="Well done",
                feedback_format="markdown",
            )
        ]
        monkeypatch.setattr(
            feedback_service, "set_feedback_bands", AsyncMock(return_value=rows)
        )
        db = _db()
        result = await authoring.set_feedback_bands(
            uuid4(), authoring._FeedbackBandsBody(bands=[]), object(), db
        )
        assert [band.feedback_text for band in result] == ["Well done"]
        db.commit.assert_awaited_once_with()


class TestManualGrading:
    """The queue and the mark, which the teacher uses in sequence."""

    async def test_the_queue_reads_each_id_from_the_right_row(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Answer, question and attempt each carry an ``id``.

        The row the teacher clicks is addressed by ``answer_id``; taking the
        wrong one from the triple would send every mark to a different
        answer than the one on screen.
        """
        import abridgeai.features.quizzes.services.manual_grading as manual_grading

        answer = SimpleNamespace(id=uuid4(), answer_text="Because of the light")
        question = SimpleNamespace(
            id=uuid4(), question_type="short_answer", prompt_text="Why?"
        )
        attempt = SimpleNamespace(
            id=uuid4(), student_id=uuid4(), submitted_at=datetime.now(UTC)
        )
        monkeypatch.setattr(
            manual_grading,
            "list_needs_grading",
            AsyncMock(return_value=[(answer, question, attempt)]),
        )

        rows = await authoring.list_needs_grading(uuid4(), object(), _db())

        assert len(rows) == 1
        assert rows[0].answer_id == answer.id
        assert rows[0].question_id == question.id
        assert rows[0].attempt_id == attempt.id
        assert rows[0].student_id == attempt.student_id
        assert rows[0].answer_text == "Because of the light"

    async def test_an_empty_queue_is_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import abridgeai.features.quizzes.services.manual_grading as manual_grading

        monkeypatch.setattr(manual_grading, "list_needs_grading", AsyncMock(return_value=[]))
        assert await authoring.list_needs_grading(uuid4(), object(), _db()) == []

    async def test_the_grader_is_taken_from_the_session_not_the_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A body-supplied grader id would let a mark be attributed to
        someone who never gave it."""
        import abridgeai.features.quizzes.services.manual_grading as manual_grading

        graded = SimpleNamespace(
            id=uuid4(),
            attempt_id=uuid4(),
            question_id=uuid4(),
            manual_score=Decimal("4"),
            manual_feedback="Good",
            is_correct=True,
            points_awarded=Decimal("4"),
            needs_manual_grade=False,
            graded_by=uuid4(),
            graded_at=datetime.now(UTC),
        )
        grade = AsyncMock(return_value=graded)
        monkeypatch.setattr(manual_grading, "grade_answer_manually", grade)
        actor = SimpleNamespace(user_id=uuid4())
        db = _db()
        quiz_id, answer_id = uuid4(), uuid4()

        await authoring.grade_answer_manually(
            quiz_id,
            answer_id,
            authoring.ManualGradeIn(score=Decimal("4"), feedback="Good"),
            actor,
            db,
        )

        assert grade.await_args.kwargs["grader_id"] == actor.user_id
        assert grade.await_args.kwargs["quiz_id"] == quiz_id
        assert grade.await_args.kwargs["answer_id"] == answer_id
        db.commit.assert_awaited_once_with()

    @pytest.mark.parametrize(
        ("failure", "status_code"),
        [(NotFoundError("gone"), 404), (AppError("score above the maximum"), 400)],
    )
    async def test_a_refused_mark_is_not_committed(
        self, monkeypatch: pytest.MonkeyPatch, failure: Exception, status_code: int
    ) -> None:
        import abridgeai.features.quizzes.services.manual_grading as manual_grading

        monkeypatch.setattr(
            manual_grading, "grade_answer_manually", AsyncMock(side_effect=failure)
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.grade_answer_manually(
                uuid4(),
                uuid4(),
                authoring.ManualGradeIn(score=Decimal("4")),
                SimpleNamespace(user_id=uuid4()),
                db,
            )
        assert raised.value.status_code == status_code
        db.commit.assert_not_awaited()


class TestCommittingARegrade:
    """A regrade rewrites marks students have already seen.

    It is reviewed as a dry run first and committed once; a second commit
    would apply the same deltas to the already-corrected scores.
    """

    async def test_recommitting_a_run_is_a_conflict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import abridgeai.features.quizzes.services.regrade as regrade_service

        monkeypatch.setattr(
            regrade_service, "commit_regrade", AsyncMock(side_effect=AppError("already committed"))
        )
        db = _db()
        with pytest.raises(HTTPException) as raised:
            await authoring.commit_regrade_run(uuid4(), uuid4(), object(), db)
        assert raised.value.status_code == 409
        db.commit.assert_not_awaited()

    async def test_an_unknown_run_is_a_404(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import abridgeai.features.quizzes.services.regrade as regrade_service

        monkeypatch.setattr(
            regrade_service, "commit_regrade", AsyncMock(side_effect=NotFoundError("gone"))
        )
        run_id = uuid4()
        with pytest.raises(HTTPException) as raised:
            await authoring.commit_regrade_run(uuid4(), run_id, object(), _db())
        assert raised.value.status_code == 404
        assert raised.value.detail["resource"] == "regrade run"
        assert raised.value.detail["id"] == str(run_id)

    async def test_a_committed_run_is_read_back_and_returned(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The response is re-read after the commit rather than projected
        from the pre-commit object, so it carries the committed timestamp."""
        import abridgeai.features.quizzes.services.regrade as regrade_service

        run = SimpleNamespace(
            id=uuid4(),
            quiz_id=uuid4(),
            status="committed",
            attempts_affected=2,
            answers_changed=3,
            created_at=datetime.now(UTC),
            committed_at=datetime.now(UTC),
            items=[],
        )
        monkeypatch.setattr(regrade_service, "commit_regrade", AsyncMock(return_value=None))
        monkeypatch.setattr(regrade_service, "get_regrade_run", AsyncMock(return_value=run))
        db = _db()

        result = await authoring.commit_regrade_run(run.quiz_id, run.id, object(), db)

        assert result.status == "committed"
        assert result.committed_at is not None
        db.commit.assert_awaited_once_with()
