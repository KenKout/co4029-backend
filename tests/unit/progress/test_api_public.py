from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from abridgeai.features.progress.api.public import (
    AtRiskStudentDTO,
    LessonProgressDTO,
    count_students_needing_attention,
    get_at_risk_students,
    get_course_health_signals,
    get_course_progress_for_user,
    get_lesson_progress,
    list_students_needing_attention,
)


def test_module_exports() -> None:
    from abridgeai.features.progress.api import public

    assert {"get_lesson_progress", "get_at_risk_students"} <= set(public.__all__)


def test_dto_is_frozen() -> None:
    dto = LessonProgressDTO(
        id=uuid4(),
        user_id=uuid4(),
        lesson_id=uuid4(),
        status="not_started",
        completion_percent=Decimal("0"),
        last_activity_at=None,
        total_time_seconds=0,
    )
    with pytest.raises(ValidationError):
        dto.status = "completed"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_get_lesson_progress_returns_none_when_missing(
    test_engine: AsyncEngine,
) -> None:
    async with AsyncSession(test_engine) as session:
        result = await get_lesson_progress(
            session, student_id=uuid4(), lesson_id=uuid4()
        )
    assert result is None


@pytest.mark.asyncio
async def test_get_lesson_progress_roundtrip(test_engine: AsyncEngine) -> None:
    user_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    lesson_id = uuid4()
    org_id = uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.connect() as conn:
        trans = await conn.begin()
        session = AsyncSession(bind=conn, expire_on_commit=False)
        try:
            await session.execute(
                text(
                    "INSERT INTO organizations (id, slug, name, status) "
                    "VALUES (:id, :slug, :name, 'active')"
                ),
                {"id": str(org_id), "slug": f"o-{suffix}", "name": "O"},
            )
            await session.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": str(user_id), "email": f"u-{suffix}@e.com"},
            )
            await session.execute(
                text(
                    "INSERT INTO courses (id, organization_id, owner_user_id, "
                    "slug, title, status) "
                    "VALUES (:id, :org, :owner, :slug, 'C', 'draft')"
                ),
                {
                    "id": str(course_id),
                    "org": str(org_id),
                    "owner": str(user_id),
                    "slug": f"c-{suffix}",
                },
            )
            await session.execute(
                text(
                    "INSERT INTO modules (id, course_id, position, title, status) "
                    "VALUES (:id, :course, 1, 'M', 'draft')"
                ),
                {"id": str(module_id), "course": str(course_id)},
            )
            await session.execute(
                text(
                    "INSERT INTO lessons (id, module_id, slug, title, status) "
                    "VALUES (:id, :module, :slug, 'L', 'draft')"
                ),
                {
                    "id": str(lesson_id),
                    "module": str(module_id),
                    "slug": f"l-{suffix}",
                },
            )
            await session.execute(
                text(
                    "INSERT INTO lesson_progress (id, user_id, lesson_id, "
                    "status, completion_percent, total_time_seconds, "
                    "last_activity_at) "
                    "VALUES (:id, :u, :l, 'in_progress', 42.5, 120, :at)"
                ),
                {
                    "id": str(uuid4()),
                    "u": str(user_id),
                    "l": str(lesson_id),
                    "at": datetime.now(UTC),
                },
            )
            await session.flush()

            result = await get_lesson_progress(
                session, student_id=user_id, lesson_id=lesson_id
            )
        finally:
            await trans.rollback()

    assert result is not None
    assert isinstance(result, LessonProgressDTO)
    assert result.user_id == user_id
    assert result.lesson_id == lesson_id
    assert result.status == "in_progress"
    assert result.completion_percent == Decimal("42.50")
    assert result.total_time_seconds == 120


@pytest.mark.asyncio
async def test_get_at_risk_students_empty_course(test_engine: AsyncEngine) -> None:
    async with AsyncSession(test_engine) as session:
        rows = await get_at_risk_students(session, uuid4())
    assert rows == []
    assert all(isinstance(r, AtRiskStudentDTO) for r in rows)


# ---------------------------------------------------------------------------
# Projections
#
# The four functions below hold no query of their own: each calls a progress
# service and reshapes the result into a DTO. The services are covered by
# ``test_at_risk_monitoring.py`` and the reporting suite, so these tests mock
# them and assert only the reshaping -- which is where a cross-feature caller
# can be handed a number that means something other than what it says.
# ---------------------------------------------------------------------------


def _risk_row(
    *,
    user_id: uuid.UUID,
    course_id: uuid.UUID,
    severity: str = "high",
    signal_count: int = 2,
) -> SimpleNamespace:
    return SimpleNamespace(
        user_id=user_id,
        course_id=course_id,
        completion_percent=Decimal("12.5"),
        last_engagement_at=datetime(2026, 9, 1, tzinfo=UTC),
        days_since_last_engagement=16,
        primary_reason="No activity for 16 days (threshold: 7)",
        signal_count=signal_count,
        severity=severity,
    )


class TestTheAtRiskRoster:
    async def test_the_primary_reason_is_the_first_of_the_scored_reasons(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The service orders reasons by severity, so the first is the one
        worth showing. Picking any other would name a lesser threshold than
        the one that actually put the student on the list.
        """
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.services import monitoring

        student = SimpleNamespace(
            user_id=uuid4(),
            completion_percent=Decimal("30"),
            days_since_last_engagement=9,
            reasons=[
                SimpleNamespace(detail="No activity for 9 days (threshold: 7)"),
                SimpleNamespace(detail="Completion 30% (threshold: 40%)"),
            ],
        )
        monkeypatch.setattr(
            monitoring,
            "get_at_risk_students",
            AsyncMock(return_value=SimpleNamespace(students=[student])),
        )

        rows = await public.get_at_risk_students(object(), uuid4())

        assert len(rows) == 1
        assert rows[0].primary_reason == "No activity for 9 days (threshold: 7)"
        assert rows[0].signal_count == 2, "the count spans every reason, not just the shown one"

    async def test_a_student_with_no_reasons_carries_no_primary_reason(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Indexing an empty reason list would raise rather than project."""
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.services import monitoring

        student = SimpleNamespace(
            user_id=uuid4(),
            completion_percent=Decimal("0"),
            days_since_last_engagement=None,
            reasons=[],
        )
        monkeypatch.setattr(
            monitoring,
            "get_at_risk_students",
            AsyncMock(return_value=SimpleNamespace(students=[student])),
        )

        rows = await public.get_at_risk_students(object(), uuid4())

        assert rows[0].primary_reason is None
        assert rows[0].signal_count == 0
        assert rows[0].days_since_last_engagement is None

    async def test_days_since_engagement_crosses_as_a_float(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The query returns a Decimal; the DTO is typed float.

        Without the cast the DTO would reject the value outright, taking the
        whole roster with it.
        """
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.services import monitoring

        student = SimpleNamespace(
            user_id=uuid4(),
            completion_percent=Decimal("10"),
            days_since_last_engagement=Decimal("12.5"),
            reasons=[SimpleNamespace(detail="inactive")],
        )
        monkeypatch.setattr(
            monitoring,
            "get_at_risk_students",
            AsyncMock(return_value=SimpleNamespace(students=[student])),
        )

        rows = await public.get_at_risk_students(object(), uuid4())
        assert rows[0].days_since_last_engagement == 12.5
        assert isinstance(rows[0].days_since_last_engagement, float)


class TestTheHeadlineCountAndItsRows:
    """The tile and the list answer different questions and may differ.

    The count is DISTINCT students; the list is one row per (student,
    course). A learner behind in three courses is one person to follow up
    with but three things to do, so the API exposes both rather than letting
    a caller derive one from the other.
    """

    async def test_the_count_is_passed_through_untouched(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.services import monitoring

        count = AsyncMock(return_value=7)
        monkeypatch.setattr(monitoring, "count_students_needing_attention", count)
        db, course_ids = object(), [uuid4(), uuid4()]

        assert await public.count_students_needing_attention(db, course_ids) == 7
        count.assert_awaited_once_with(db, course_ids)

    async def test_one_student_in_two_courses_yields_two_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Deduplicating here would hide the second course's follow-up."""
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.services import monitoring

        student, course_a, course_b = uuid4(), uuid4(), uuid4()
        monkeypatch.setattr(
            monitoring,
            "list_students_needing_attention",
            AsyncMock(
                return_value=[
                    _risk_row(user_id=student, course_id=course_a),
                    _risk_row(user_id=student, course_id=course_b, severity="medium"),
                ]
            ),
        )

        rows = await public.list_students_needing_attention(object(), [course_a, course_b])

        assert [row.course_id for row in rows] == [course_a, course_b]
        assert {row.user_id for row in rows} == {student}
        assert [row.severity for row in rows] == ["high", "medium"]

    async def test_the_service_ordering_is_preserved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Worst first: the service sorts, and the projection must not resort."""
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.services import monitoring

        first, second = uuid4(), uuid4()
        course = uuid4()
        monkeypatch.setattr(
            monitoring,
            "list_students_needing_attention",
            AsyncMock(
                return_value=[
                    _risk_row(user_id=first, course_id=course),
                    _risk_row(user_id=second, course_id=course, severity="medium"),
                ]
            ),
        )

        rows = await public.list_students_needing_attention(object(), [course])
        assert [row.user_id for row in rows] == [first, second]

    async def test_no_one_at_risk_is_an_empty_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.services import monitoring

        monkeypatch.setattr(
            monitoring, "list_students_needing_attention", AsyncMock(return_value=[])
        )
        assert await public.list_students_needing_attention(object(), [uuid4()]) == []


class TestCourseHealthSignals:
    """``at_risk_students`` must stay comparable with ``student_count``.

    The table reads "8 of 40". A signal tally rather than a headcount could
    print "52 of 40", which tells a teacher nothing and undermines the rest
    of the row.
    """

    async def test_a_student_with_several_signals_counts_once(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.queries import analytics
        from abridgeai.features.progress.services import monitoring

        course = uuid4()
        student = uuid4()
        monkeypatch.setattr(
            analytics,
            "summarize_progress_by_course",
            AsyncMock(
                return_value={
                    course: SimpleNamespace(student_count=40, avg_completion_percent=61.5)
                }
            ),
        )
        monkeypatch.setattr(
            monitoring,
            "list_students_needing_attention",
            AsyncMock(
                return_value=[
                    _risk_row(user_id=student, course_id=course),
                    _risk_row(user_id=student, course_id=course, severity="medium"),
                ]
            ),
        )

        signals = await public.get_course_health_signals(object(), [course])

        assert signals[course].at_risk_students == 1
        assert signals[course].student_count == 40
        assert signals[course].avg_completion_percent == 61.5

    async def test_each_course_counts_only_its_own_students(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One dashboard request spans a teacher's whole course set, so the
        rows arrive interleaved and have to be bucketed before counting."""
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.queries import analytics
        from abridgeai.features.progress.services import monitoring

        course_a, course_b = uuid4(), uuid4()
        monkeypatch.setattr(
            analytics,
            "summarize_progress_by_course",
            AsyncMock(
                return_value={
                    course_a: SimpleNamespace(student_count=10, avg_completion_percent=50.0),
                    course_b: SimpleNamespace(student_count=20, avg_completion_percent=75.0),
                }
            ),
        )
        monkeypatch.setattr(
            monitoring,
            "list_students_needing_attention",
            AsyncMock(
                return_value=[
                    _risk_row(user_id=uuid4(), course_id=course_a),
                    _risk_row(user_id=uuid4(), course_id=course_b),
                    _risk_row(user_id=uuid4(), course_id=course_b),
                ]
            ),
        )

        signals = await public.get_course_health_signals(object(), [course_a, course_b])

        assert signals[course_a].at_risk_students == 1
        assert signals[course_b].at_risk_students == 2

    async def test_a_course_with_no_summary_is_omitted_entirely(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A course nobody is enrolled in is absent rather than zeroed.

        "Nobody has enrolled yet" and "everyone is at 0%" look identical as a
        zero row, and only one of them is a problem the teacher can act on.
        The caller is left to render the difference.
        """
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.queries import analytics
        from abridgeai.features.progress.services import monitoring

        enrolled, empty = uuid4(), uuid4()
        monkeypatch.setattr(
            analytics,
            "summarize_progress_by_course",
            AsyncMock(
                return_value={
                    enrolled: SimpleNamespace(student_count=5, avg_completion_percent=20.0)
                }
            ),
        )
        monkeypatch.setattr(
            monitoring, "list_students_needing_attention", AsyncMock(return_value=[])
        )

        signals = await public.get_course_health_signals(object(), [enrolled, empty])

        assert set(signals) == {enrolled}
        assert signals[enrolled].at_risk_students == 0


class TestPerUserCourseProgress:
    async def test_the_summary_crosses_the_boundary_as_plain_data(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reporting model is feature-internal, so it is dumped rather
        than returned: handing the model itself across would make a progress
        schema part of another feature's contract.
        """
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.services import reporting

        summary = SimpleNamespace(
            model_dump=lambda: {
                "course_id": "c",
                "completed_lessons": 3,
                "in_progress_lessons": 1,
                "not_started_lessons": 6,
                "completion_percent": 30.0,
                "last_activity_at": None,
            }
        )
        get_summary = AsyncMock(return_value=summary)
        monkeypatch.setattr(reporting, "get_my_course_progress_summary", get_summary)
        db, user_id, course_id = object(), uuid4(), uuid4()

        result = await public.get_course_progress_for_user(
            db, user_id=user_id, course_id=course_id
        )

        assert isinstance(result, dict)
        assert result["completed_lessons"] == 3
        get_summary.assert_awaited_once_with(db, user_id=user_id, course_id=course_id)

    async def test_the_student_is_named_by_argument_not_by_session(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The learner endpoint serves whoever is signed in; this is the same
        projection for any user, which is what makes it usable from the
        manager's user-detail page.
        """
        from abridgeai.features.progress.api import public
        from abridgeai.features.progress.services import reporting

        get_summary = AsyncMock(return_value=SimpleNamespace(model_dump=dict))
        monkeypatch.setattr(reporting, "get_my_course_progress_summary", get_summary)
        someone_else = uuid4()

        await public.get_course_progress_for_user(
            object(), user_id=someone_else, course_id=uuid4()
        )

        assert get_summary.await_args.kwargs["user_id"] == someone_else
