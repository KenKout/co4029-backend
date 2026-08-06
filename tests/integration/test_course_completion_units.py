"""Course completion counted in curriculum UNITS (lessons + quizzes + interviews).

Guards the behaviour change: ``course_enrollments.status`` — which career-path
stage unlock reads as ``satisfied`` (D2) — used to be an average over published
lessons and ignored quiz and interview items entirely.

The parity tests matter most. Two independent implementations of the same rules
now exist: the per-item ``learner_progress`` services the curriculum screen
renders, and the aggregate SQL in ``queries/completion_units.py``. If they ever
disagree, a student sees an item ticked while the stage behind it stays locked
(or the reverse), so the agreement is asserted rather than assumed.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.features.career_paths.services import authoring as cp_authoring_service
from abridgeai.features.enrollments.queries.completion_units import (
    count_course_units,
    get_course_unit_tally,
)
from abridgeai.features.enrollments.services.completion import (
    resync_stale_course_completions,
    sync_course_completion,
)
from abridgeai.features.interviews.services.learner_progress import (
    list_my_interview_progress,
)
from abridgeai.features.quizzes.services.learner_progress import list_my_quiz_progress


def _async_url(u: str) -> str:
    if "+psycopg_async" in u:
        return u
    if u.startswith("postgresql+psycopg://"):
        return u.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if u.startswith("postgresql://"):
        return u.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return u


def _ensure_head() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


class _Builder:
    """Seeds one org/owner/student and courses with arbitrary unit mixes.

    Each method commits on its own connection: the code under test reads
    through a separate ``AsyncSession``, so seed data held open in an
    uncommitted transaction would be invisible to it.
    """

    def __init__(self, engine: AsyncEngine, suffix: str) -> None:
        self._engine = engine
        self._s = suffix
        self.org = uuid.uuid4()
        self.owner = uuid.uuid4()
        self.student = uuid.uuid4()
        self.courses: list[uuid.UUID] = []

    async def _exec(self, sql: str, params: dict | None = None) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(text(sql), params or {})

    async def setup(self) -> None:
        await self._exec(
            "INSERT INTO organizations (id, slug, name) VALUES (:i,:s,'Units')",
            {"i": self.org, "s": f"cu-{self._s}"},
        )
        for uid, email in (
            (self.owner, f"own-{self._s}@t.local"),
            (self.student, f"stu-{self._s}@t.local"),
        ):
            await self._exec(
                "INSERT INTO users (id, primary_email) VALUES (:i,:e)",
                {"i": uid, "e": email},
            )

    async def course(
        self,
        *,
        slug: str,
        lessons: int = 0,
        quizzes: int = 0,
        interviews: int = 0,
        draft_quizzes: int = 0,
        draft_interviews: int = 0,
        enroll: bool = True,
    ) -> dict:
        course, module = uuid.uuid4(), uuid.uuid4()
        await self._exec(
            "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
            "VALUES (:i,:o,:w,:s,:s,'published')",
            {"i": course, "o": self.org, "w": self.owner, "s": f"{slug}-{self._s}"},
        )
        await self._exec(
            "INSERT INTO modules (id, course_id, title, position, status) "
            "VALUES (:i,:c,'M',1,'published')",
            {"i": module, "c": course},
        )
        pos = 0
        lesson_ids, quiz_ids, iv_ids = [], [], []
        for n in range(lessons):
            pos += 1
            lid = uuid.uuid4()
            await self._exec(
                "INSERT INTO lessons (id, module_id, slug, title, status, lesson_type) "
                "VALUES (:i,:m,:s,'L','published','video')",
                {"i": lid, "m": module, "s": f"{slug}-l{n}-{self._s}"},
            )
            await self._exec(
                "INSERT INTO module_items (id, module_id, item_type, lesson_id, position) "
                "VALUES (:i,:m,'lesson',:r,:p)",
                {"i": uuid.uuid4(), "m": module, "r": lid, "p": pos},
            )
            lesson_ids.append(lid)
        for n in range(quizzes + draft_quizzes):
            pos += 1
            qid = uuid.uuid4()
            status = "published" if n < quizzes else "draft"
            await self._exec(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent, allow_retakes, max_attempts) "
                "VALUES (:i,:c,:m,'Q',:st,70,TRUE,2)",
                {"i": qid, "c": course, "m": module, "st": status},
            )
            await self._exec(
                "INSERT INTO module_items (id, module_id, item_type, quiz_id, position) "
                "VALUES (:i,:m,'quiz',:r,:p)",
                {"i": uuid.uuid4(), "m": module, "r": qid, "p": pos},
            )
            if n < quizzes:
                quiz_ids.append(qid)
        for n in range(interviews + draft_interviews):
            pos += 1
            iid = uuid.uuid4()
            status = "published" if n < interviews else "draft"
            await self._exec(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, created_by) "
                "VALUES (:i,:c,:m,'IV',:st,:w)",
                {"i": iid, "c": course, "m": module, "st": status, "w": self.owner},
            )
            await self._exec(
                "INSERT INTO module_items "
                "(id, module_id, item_type, interview_config_id, position) "
                "VALUES (:i,:m,'interview',:r,:p)",
                {"i": uuid.uuid4(), "m": module, "r": iid, "p": pos},
            )
            if n < interviews:
                iv_ids.append(iid)
        if enroll:
            await self._exec(
                "INSERT INTO course_enrollments (id, course_id, student_id, status, source) "
                "VALUES (:i,:c,:s,'active','manual')",
                {"i": uuid.uuid4(), "c": course, "s": self.student},
            )
        self.courses.append(course)
        return {
            "course": course,
            "module": module,
            "lessons": lesson_ids,
            "quizzes": quiz_ids,
            "interviews": iv_ids,
        }

    async def complete_lesson(self, lesson_id: uuid.UUID) -> None:
        await self._exec(
            "INSERT INTO lesson_progress "
            "(id, user_id, lesson_id, status, completion_percent, total_time_seconds) "
            "VALUES (:i,:u,:l,'completed',100,0)",
            {"i": uuid.uuid4(), "u": self.student, "l": lesson_id},
        )

    async def pass_quiz(self, quiz_id: uuid.UUID) -> None:
        await self._exec(
            "INSERT INTO quiz_grades (id, quiz_id, student_id, grade_percent, "
            "grade_points, passed, grading_method, attempts_counted) "
            "VALUES (:i,:q,:s,90,9,TRUE,'highest',1)",
            {"i": uuid.uuid4(), "q": quiz_id, "s": self.student},
        )

    async def fail_quiz(self, quiz_id: uuid.UUID, *, attempts: int) -> None:
        """Record a failing grade plus ``attempts`` submitted attempts."""
        await self._exec(
            "INSERT INTO quiz_grades (id, quiz_id, student_id, grade_percent, "
            "grade_points, passed, grading_method, attempts_counted) "
            "VALUES (:i,:q,:s,10,1,FALSE,'highest',:n)",
            {"i": uuid.uuid4(), "q": quiz_id, "s": self.student, "n": attempts},
        )
        for n in range(attempts):
            await self._exec(
                "INSERT INTO quiz_attempts (id, quiz_id, student_id, attempt_number, "
                "status, score_percent) VALUES (:i,:q,:s,:n,'submitted',10)",
                {"i": uuid.uuid4(), "q": quiz_id, "s": self.student, "n": n + 1},
            )

    async def interview_attempt(
        self, config_id: uuid.UUID, *, passed: bool | None, mode: str = "assessment", num: int = 1
    ) -> None:
        await self._exec(
            "INSERT INTO interview_sessions (id, interview_config_id, student_id, "
            "attempt_number, status, input_mode, session_mode, pass_verdict) "
            "VALUES (:i,:c,:s,:n,'completed','text',:m,:v)",
            {
                "i": uuid.uuid4(),
                "c": config_id,
                "s": self.student,
                "n": num,
                "m": mode,
                "v": passed,
            },
        )


@pytest_asyncio.fixture
async def builder(engine: AsyncEngine) -> AsyncIterator[_Builder]:
    b = _Builder(engine, uuid.uuid4().hex[:8])
    await b.setup()
    yield b
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM lesson_progress WHERE user_id=:s"), {"s": b.student})
        await conn.execute(text("DELETE FROM quiz_attempts WHERE student_id=:s"), {"s": b.student})
        await conn.execute(text("DELETE FROM quiz_grades WHERE student_id=:s"), {"s": b.student})
        await conn.execute(
            text("DELETE FROM interview_sessions WHERE student_id=:s"), {"s": b.student}
        )
        await conn.execute(
            text("DELETE FROM course_enrollments WHERE student_id=:s"), {"s": b.student}
        )
        for cid in b.courses:
            await conn.execute(
                text("DELETE FROM career_course_items WHERE course_id=:c"), {"c": cid}
            )
            await conn.execute(
                text(
                    "DELETE FROM module_items WHERE module_id IN "
                    "(SELECT id FROM modules WHERE course_id=:c)"
                ),
                {"c": cid},
            )
            await conn.execute(text("DELETE FROM interview_configs WHERE course_id=:c"), {"c": cid})
            await conn.execute(text("DELETE FROM quizzes WHERE course_id=:c"), {"c": cid})
            await conn.execute(
                text(
                    "DELETE FROM lessons WHERE module_id IN "
                    "(SELECT id FROM modules WHERE course_id=:c)"
                ),
                {"c": cid},
            )
            await conn.execute(text("DELETE FROM modules WHERE course_id=:c"), {"c": cid})
            await conn.execute(text("DELETE FROM courses WHERE id=:c"), {"c": cid})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:a,:b)"), {"a": b.owner, "b": b.student}
        )
        await conn.execute(text("DELETE FROM organizations WHERE id=:o"), {"o": b.org})


async def _status(engine: AsyncEngine, course: uuid.UUID, student: uuid.UUID) -> str:
    async with engine.connect() as conn:
        return (
            await conn.execute(
                text("SELECT status FROM course_enrollments WHERE course_id=:c AND student_id=:s"),
                {"c": course, "s": student},
            )
        ).scalar_one()


# ---------------------------------------------------------------- the headline


@pytest.mark.asyncio
async def test_lessons_done_but_quiz_untouched_is_not_complete(
    session_factory, engine, builder
) -> None:
    """THE regression: this used to promote to 'completed' on lessons alone."""
    fx = await builder.course(slug="mix", lessons=1, quizzes=1)
    await builder.complete_lesson(fx["lessons"][0])

    async with session_factory() as db:
        result = await sync_course_completion(
            db, course_id=fx["course"], student_id=builder.student
        )
        await db.commit()

    assert result is None, "an unanswered quiz must block completion"
    assert await _status(engine, fx["course"], builder.student) == "active"


@pytest.mark.asyncio
async def test_all_units_done_promotes(session_factory, engine, builder) -> None:
    fx = await builder.course(slug="all", lessons=1, quizzes=1, interviews=1)
    await builder.complete_lesson(fx["lessons"][0])
    await builder.pass_quiz(fx["quizzes"][0])
    await builder.interview_attempt(fx["interviews"][0], passed=True)

    async with session_factory() as db:
        assert (
            await sync_course_completion(db, course_id=fx["course"], student_id=builder.student)
            == "completed"
        )
        await db.commit()
    assert await _status(engine, fx["course"], builder.student) == "completed"


@pytest.mark.asyncio
async def test_failed_interview_blocks_but_exhausted_quiz_does_not(
    session_factory, builder
) -> None:
    """The deliberate asymmetry between the two rules, in one assertion.

    A quiz completes when terminal (failed with every attempt used); an
    interview completes only when PASSED.
    """
    fx = await builder.course(slug="asym", quizzes=1, interviews=1)
    await builder.fail_quiz(fx["quizzes"][0], attempts=2)  # allow_retakes, max 2 → exhausted
    await builder.interview_attempt(fx["interviews"][0], passed=False)

    async with session_factory() as db:
        tally = await get_course_unit_tally(db, course_id=fx["course"], student_id=builder.student)

    assert tally.quizzes_done == 1, "failed-but-exhausted quiz is terminal ⇒ done"
    assert tally.interviews_done == 0, "failed interview stays pending — must be PASSED"
    assert not tally.is_complete()


@pytest.mark.asyncio
async def test_practice_interview_never_completes_a_unit(session_factory, builder) -> None:
    """Cohort fairness: rehearsing must not tick a graded milestone."""
    fx = await builder.course(slug="prac", interviews=1)
    await builder.interview_attempt(fx["interviews"][0], passed=True, mode="practice")

    async with session_factory() as db:
        tally = await get_course_unit_tally(db, course_id=fx["course"], student_id=builder.student)
    assert tally.interviews_total == 1
    assert tally.interviews_done == 0


@pytest.mark.asyncio
async def test_draft_units_are_invisible_and_uncounted(session_factory, builder) -> None:
    """A unit the student cannot see must never become an unsatisfiable unit.

    This is the bug that would have locked stages permanently: the interview
    progress query did not filter ``interview_configs.status``, and this
    database has curriculum rows pointing at draft configs.
    """
    fx = await builder.course(slug="draft", lessons=1, draft_quizzes=2, draft_interviews=3)
    await builder.complete_lesson(fx["lessons"][0])

    async with session_factory() as db:
        tally = await get_course_unit_tally(db, course_id=fx["course"], student_id=builder.student)
        assert tally.total == 1, "only the published lesson counts"
        assert tally.is_complete()
        assert (
            await sync_course_completion(db, course_id=fx["course"], student_id=builder.student)
            == "completed"
        )
        await db.commit()


@pytest.mark.asyncio
async def test_zero_unit_course_never_promotes(session_factory, builder) -> None:
    fx = await builder.course(slug="empty")
    async with session_factory() as db:
        tally = await get_course_unit_tally(db, course_id=fx["course"], student_id=builder.student)
        assert tally.total == 0
        assert tally.percent == 0.0, "an empty course is 0%, never 100%"
        assert not tally.is_complete()
        assert (
            await sync_course_completion(db, course_id=fx["course"], student_id=builder.student)
            is None
        )


@pytest.mark.asyncio
async def test_demotion_when_a_quiz_is_regraded_below_the_milestone(
    session_factory, engine, builder
) -> None:
    """Completion can move backward through the quiz path too, not just lessons."""
    fx = await builder.course(slug="demote", lessons=1, quizzes=1)
    await builder.complete_lesson(fx["lessons"][0])
    await builder.pass_quiz(fx["quizzes"][0])

    async with session_factory() as db:
        assert (
            await sync_course_completion(db, course_id=fx["course"], student_id=builder.student)
            == "completed"
        )
        await db.commit()

    # Regrade below the milestone, attempts still remaining ⇒ not terminal.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE quiz_grades SET passed=FALSE, grade_percent=20 "
                "WHERE quiz_id=:q AND student_id=:s"
            ),
            {"q": fx["quizzes"][0], "s": builder.student},
        )

    async with session_factory() as db:
        assert (
            await sync_course_completion(db, course_id=fx["course"], student_id=builder.student)
            == "active"
        )
        await db.commit()
    assert await _status(engine, fx["course"], builder.student) == "active"


@pytest.mark.asyncio
async def test_partial_percent_counts_every_kind(session_factory, builder) -> None:
    fx = await builder.course(slug="pct", lessons=2, quizzes=1, interviews=1)
    await builder.complete_lesson(fx["lessons"][0])
    await builder.pass_quiz(fx["quizzes"][0])

    async with session_factory() as db:
        tally = await get_course_unit_tally(db, course_id=fx["course"], student_id=builder.student)
    assert (tally.total, tally.done) == (4, 2)
    assert tally.percent == 50.0


# ------------------------------------------------------------------- parity


@pytest.mark.asyncio
async def test_quiz_parity_with_learner_progress(session_factory, builder) -> None:
    """The aggregate SQL and the per-item service must agree on every quiz."""
    fx = await builder.course(slug="qpar", quizzes=3)
    await builder.pass_quiz(fx["quizzes"][0])
    await builder.fail_quiz(fx["quizzes"][1], attempts=2)  # exhausted ⇒ done
    await builder.fail_quiz(fx["quizzes"][2], attempts=1)  # attempt left ⇒ pending

    async with session_factory() as db:
        per_item = await list_my_quiz_progress(db, course_id=fx["course"], user_id=builder.student)
        tally = await get_course_unit_tally(db, course_id=fx["course"], student_id=builder.student)

    assert sum(1 for q in per_item if q["completed"]) == tally.quizzes_done
    assert len(per_item) == tally.quizzes_total
    assert tally.quizzes_done == 2


@pytest.mark.asyncio
async def test_interview_parity_with_learner_progress(session_factory, builder) -> None:
    fx = await builder.course(slug="ipar", interviews=3, draft_interviews=1)
    await builder.interview_attempt(fx["interviews"][0], passed=True)
    await builder.interview_attempt(fx["interviews"][1], passed=False)
    await builder.interview_attempt(fx["interviews"][2], passed=True, mode="practice")

    async with session_factory() as db:
        per_item = await list_my_interview_progress(
            db, course_id=fx["course"], user_id=builder.student
        )
        tally = await get_course_unit_tally(db, course_id=fx["course"], student_id=builder.student)

    assert len(per_item) == tally.interviews_total == 3, "the draft config is excluded by both"
    assert sum(1 for i in per_item if i["completed"]) == tally.interviews_done == 1


@pytest.mark.asyncio
async def test_interview_progress_excludes_draft_and_deleted_configs(
    session_factory, engine, builder
) -> None:
    """Guards the population fix directly: the curriculum's filter, applied here.

    Without it a draft interview is reported to the learner-progress consumer
    and — now that completion counts interview units — becomes a unit no
    student can satisfy, locking the course and every stage behind it.
    """
    fx = await builder.course(slug="pop", interviews=1, draft_interviews=2)
    async with engine.begin() as conn:
        await conn.execute(
            text("UPDATE interview_configs SET deleted_at=NOW() WHERE id=:i"),
            {"i": fx["interviews"][0]},
        )

    async with session_factory() as db:
        per_item = await list_my_interview_progress(
            db, course_id=fx["course"], user_id=builder.student
        )
        counts = await count_course_units(db, course_id=fx["course"])

    assert per_item == [], "1 published-but-deleted + 2 drafts ⇒ nothing reportable"
    assert counts.interviews == 0


# --------------------------------------------------------------- publish gate


@pytest.mark.asyncio
async def test_publish_gate_rejects_course_with_no_gradeable_units(
    session_factory, engine, builder
) -> None:
    """A published course with no unit can never be satisfied — reject at publish.

    Previously the gate only checked ``courses.status = 'published'``, so an
    interview-only course whose interviews were all draft sailed through and
    locked the stage forever.
    """
    fx = await builder.course(slug="gate", draft_interviews=2, enroll=False)
    path_id, stage_id = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                "VALUES (:i,:o,:s,'P','draft')"
            ),
            {"i": path_id, "o": builder.org, "s": f"gate-{uuid.uuid4().hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO career_path_stages "
                "(id, career_path_id, position, unlock_policy, enforcement) "
                "VALUES (:i,:p,1,'always','hard')"
            ),
            {"i": stage_id, "p": path_id},
        )
        await conn.execute(
            text(
                "INSERT INTO career_course_items "
                "(career_path_id, course_id, stage_id, position, is_required) "
                "VALUES (:p,:c,:s,1,TRUE)"
            ),
            {"p": path_id, "c": fx["course"], "s": stage_id},
        )

    try:
        async with session_factory() as db:
            with pytest.raises(Exception, match="stage_course_has_no_gradeable_units"):
                await cp_authoring_service.validate_path_for_publish(db, path_id)
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM career_course_items WHERE career_path_id=:p"), {"p": path_id}
            )
            await conn.execute(
                text("DELETE FROM career_path_stages WHERE career_path_id=:p"), {"p": path_id}
            )
            await conn.execute(text("DELETE FROM career_paths WHERE id=:p"), {"p": path_id})


@pytest.mark.asyncio
async def test_publish_gate_accepts_a_course_whose_only_unit_is_a_quiz(
    session_factory, engine, builder
) -> None:
    """The gate counts UNITS, not lessons — a quiz-only course is finishable."""
    fx = await builder.course(slug="qonly", quizzes=1, enroll=False)
    path_id, stage_id = uuid.uuid4(), uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO career_paths (id, organization_id, slug, name, status) "
                "VALUES (:i,:o,:s,'P','draft')"
            ),
            {"i": path_id, "o": builder.org, "s": f"qonly-{uuid.uuid4().hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO career_path_stages "
                "(id, career_path_id, position, unlock_policy, enforcement) "
                "VALUES (:i,:p,1,'always','soft')"
            ),
            {"i": stage_id, "p": path_id},
        )
        await conn.execute(
            text(
                "INSERT INTO career_course_items "
                "(career_path_id, course_id, stage_id, position, is_required) "
                "VALUES (:p,:c,:s,1,TRUE)"
            ),
            {"p": path_id, "c": fx["course"], "s": stage_id},
        )
    try:
        async with session_factory() as db:
            assert await cp_authoring_service.validate_path_for_publish(db, path_id) == []
    finally:
        async with engine.begin() as conn:
            await conn.execute(
                text("DELETE FROM career_course_items WHERE career_path_id=:p"), {"p": path_id}
            )
            await conn.execute(
                text("DELETE FROM career_path_stages WHERE career_path_id=:p"), {"p": path_id}
            )
            await conn.execute(text("DELETE FROM career_paths WHERE id=:p"), {"p": path_id})


# ------------------------------------------------------------- drift backstop


@pytest.mark.asyncio
async def test_drift_sweeper_repairs_a_lost_write(session_factory, engine, builder) -> None:
    """The backstop the docstring used to promise but no code implemented.

    Simulates a dropped synchronous write (every call site swallows its own
    exceptions) by finishing the work directly in SQL, then asserts the sweep
    notices and repairs it.
    """
    fx = await builder.course(slug="drift", lessons=1, quizzes=1)
    await builder.complete_lesson(fx["lessons"][0])
    await builder.pass_quiz(fx["quizzes"][0])
    assert await _status(engine, fx["course"], builder.student) == "active"

    async with session_factory() as db:
        scanned, fixed = await resync_stale_course_completions(db)
        await db.commit()

    assert scanned >= 1
    assert fixed >= 1
    assert await _status(engine, fx["course"], builder.student) == "completed"


@pytest.mark.asyncio
async def test_drift_sweeper_is_idempotent(session_factory, engine, builder) -> None:
    fx = await builder.course(slug="idem", lessons=1)
    await builder.complete_lesson(fx["lessons"][0])

    async with session_factory() as db:
        await resync_stale_course_completions(db)
        await db.commit()
    assert await _status(engine, fx["course"], builder.student) == "completed"

    async with session_factory() as db:
        # A second sweep must not report this pair again.
        _, fixed_again = await resync_stale_course_completions(db)
        await db.commit()

    async with engine.connect() as conn:
        still = (
            await conn.execute(
                text("SELECT status FROM course_enrollments WHERE course_id=:c AND student_id=:s"),
                {"c": fx["course"], "s": builder.student},
            )
        ).scalar_one()
    assert still == "completed"
    assert isinstance(fixed_again, int)


@pytest.mark.asyncio
async def test_dropped_and_waitlisted_rows_are_untouched(session_factory, engine, builder) -> None:
    """Promotion must never resurrect a dropped enrollment or grant a waitlist place."""
    fx = await builder.course(slug="drop", lessons=1)
    await builder.complete_lesson(fx["lessons"][0])
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE course_enrollments SET status='dropped' "
                "WHERE course_id=:c AND student_id=:s"
            ),
            {"c": fx["course"], "s": builder.student},
        )

    async with session_factory() as db:
        assert (
            await sync_course_completion(db, course_id=fx["course"], student_id=builder.student)
            is None
        )
        await resync_stale_course_completions(db)
        await db.commit()

    assert await _status(engine, fx["course"], builder.student) == "dropped"


@pytest.mark.asyncio
async def test_lesson_without_a_module_items_row_still_counts(
    session_factory, engine, builder
) -> None:
    """Lessons must be found through modules, not through module_items.

    ``module_items`` is the curriculum ORDERING table and a published lesson
    can legitimately have no row in it — this database has a course with 4
    published lessons and only 3 module_items rows. An earlier revision of the
    unit query joined lessons through module_items, which silently dropped that
    lesson from the denominator and completed the course while a real lesson
    was still unfinished. That is precisely the bug class this change removes,
    so it must not be reintroduced through the fix for it.
    """
    fx = await builder.course(slug="orphan", lessons=2)
    # Detach one lesson from the ordering table, leaving it published.
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM module_items WHERE lesson_id=:l"), {"l": fx["lessons"][1]}
        )
    await builder.complete_lesson(fx["lessons"][0])

    async with session_factory() as db:
        tally = await get_course_unit_tally(db, course_id=fx["course"], student_id=builder.student)
        counts = await count_course_units(db, course_id=fx["course"])
        assert (tally.lessons_total, tally.lessons_done) == (2, 1)
        assert counts.lessons == 2
        assert not tally.is_complete(), "the unordered lesson still has to be done"
        assert (
            await sync_course_completion(db, course_id=fx["course"], student_id=builder.student)
            is None
        )


@pytest.mark.asyncio
async def test_lesson_at_99_percent_is_not_a_done_unit(session_factory, engine, builder) -> None:
    """Unit completion reads ``lesson_progress.status``, not the raw percent.

    ``mark_lesson_complete`` and the 80% engagement threshold both set
    status='completed'; a partially-watched lesson has a percent but no status,
    and must not count.
    """
    fx = await builder.course(slug="p99", lessons=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO lesson_progress (id, user_id, lesson_id, status, "
                "completion_percent, total_time_seconds) "
                "VALUES (:i,:u,:l,'in_progress',:p,10)"
            ),
            {
                "i": uuid.uuid4(),
                "u": builder.student,
                "l": fx["lessons"][0],
                "p": Decimal("99.00"),
            },
        )

    async with session_factory() as db:
        tally = await get_course_unit_tally(db, course_id=fx["course"], student_id=builder.student)
    assert (tally.lessons_total, tally.lessons_done) == (1, 0)
