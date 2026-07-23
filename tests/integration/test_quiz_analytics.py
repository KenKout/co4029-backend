"""Integration tests for :func:`quiz_results_summary` (grading-method-aware).

Seeds a single quiz with two students and a mix of completed attempts
(plus one in-progress attempt that must be ignored), then asserts the
per-method headline reduction, the aggregate stats (mean / median /
percentiles / pass-rate / mean-time) and the 11-bucket score histogram.

The function reduces COMPLETED attempts (``status IN ('submitted','graded')``
AND ``score_percent IS NOT NULL``) to one headline row per student
according to ``grading_method`` (highest/average/first/last), then
aggregates over that headline set.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
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
from abridgeai.features.quizzes.queries.analytics import (
    quiz_question_breakdown,
    quiz_results_summary,
)


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    cfg_path = Path(__file__).resolve().parents[2] / "alembic.ini"
    cfg = Config(str(cfg_path))
    cfg.set_main_option(
        "script_location",
        str(Path(__file__).resolve().parents[2] / "migrations"),
    )
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


@pytest_asyncio.fixture
async def scenario(engine: AsyncEngine) -> AsyncIterator[dict[str, uuid.UUID]]:
    """Seed one course/module/quiz with two students and mixed attempts.

    Completed attempts (passing threshold 70):
      student A: #1 score 40 (passed=False), #2 score 100 (passed=True)
      student B: #1 score 60 (passed=False), #2 score 80  (passed=True)
    Plus one in_progress attempt (score NULL) that MUST be ignored.

    A second empty quiz is seeded for the zero-attempts case.
    """
    org = uuid.uuid4()
    teacher = uuid.uuid4()
    student_a = uuid.uuid4()
    student_b = uuid.uuid4()
    course = uuid.uuid4()
    module = uuid.uuid4()
    quiz = uuid.uuid4()
    empty_quiz = uuid.uuid4()

    now = datetime.now(UTC)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Analytics Org', 'active')"
            ),
            {"id": org, "slug": f"an-org-{org.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO users (id, primary_email, status) VALUES "
                "(:t, :te, 'active'), (:a, :ae, 'active'), (:b, :be, 'active')"
            ),
            {
                "t": teacher,
                "a": student_a,
                "b": student_b,
                "te": f"t-{teacher.hex[:8]}@test.local",
                "ae": f"a-{student_a.hex[:8]}@test.local",
                "be": f"b-{student_b.hex[:8]}@test.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :u, :slug, 'Analytics Course', 'published')"
            ),
            {"id": course, "org": org, "u": teacher, "slug": f"an-course-{course.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :c, 'Module', 1, 'published')"
            ),
            {"id": module, "c": course},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) VALUES "
                "(:q, :c, :m, 'Results Quiz', 'published'), "
                "(:eq, :c, :m, 'Empty Quiz', 'published')"
            ),
            {"q": quiz, "eq": empty_quiz, "c": course, "m": module},
        )

        # student A: #1 = 40 (fail), #2 = 100 (pass)
        # student B: #1 = 60 (fail), #2 = 80  (pass)
        attempts = [
            (student_a, 1, "graded", 40, False, 300),
            (student_a, 2, "graded", 100, True, 200),
            (student_b, 1, "graded", 60, False, 400),
            (student_b, 2, "submitted", 80, True, 500),
        ]
        for sid, num, status, score, passed, secs in attempts:
            await conn.execute(
                text(
                    "INSERT INTO quiz_attempts "
                    "(id, quiz_id, student_id, attempt_number, status, "
                    " score_percent, passed, time_taken_seconds, started_at, submitted_at) "
                    "VALUES (:id, :q, :s, :n, :st, :sc, :p, :secs, :start, :submit)"
                ),
                {
                    "id": uuid.uuid4(),
                    "q": quiz,
                    "s": sid,
                    "n": num,
                    "st": status,
                    "sc": score,
                    "p": passed,
                    "secs": secs,
                    "start": now - timedelta(hours=2),
                    "submit": now - timedelta(hours=1),
                },
            )

        # in_progress attempt (score NULL) — MUST be ignored by the summary.
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts "
                "(id, quiz_id, student_id, attempt_number, status, "
                " score_percent, passed, time_taken_seconds, started_at) "
                "VALUES (:id, :q, :s, :n, 'in_progress', NULL, NULL, NULL, :start)"
            ),
            {
                "id": uuid.uuid4(),
                "q": quiz,
                "s": student_a,
                "n": 3,
                "start": now,
            },
        )

    yield {
        "course_id": course,
        "quiz_id": quiz,
        "empty_quiz_id": empty_quiz,
        "student_a": student_a,
        "student_b": student_b,
        "org_id": org,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM quiz_attempts WHERE quiz_id IN (:q, :eq)"),
            {"q": quiz, "eq": empty_quiz},
        )
        await conn.execute(
            text("DELETE FROM quizzes WHERE id IN (:q, :eq)"),
            {"q": quiz, "eq": empty_quiz},
        )
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :a, :b)"),
            {"t": teacher, "a": student_a, "b": student_b},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org})


async def test_highest_headline_and_aggregates(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """grading_method='highest' → headline A=100, B=80."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["quiz_id"], "highest")

    assert result["total_attempts"] == 4  # 4 completed; in_progress excluded
    assert result["unique_students"] == 2
    assert result["mean_score"] == pytest.approx(90.0)
    assert result["median_score"] == pytest.approx(90.0)
    assert result["p25"] == pytest.approx(85.0)
    assert result["p75"] == pytest.approx(95.0)
    assert result["pass_rate"] == pytest.approx(1.0)  # both headline attempts passed
    assert result["mean_time_seconds"] == pytest.approx(350.0)  # A#2=200, B#2=500 → 350

    assert len(result["histogram"]) == 11
    assert sum(b["count"] for b in result["histogram"]) == 2


async def test_average_headline_and_aggregates(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """grading_method='average' → headline A=(40+100)/2=70, B=(60+80)/2=70."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["quiz_id"], "average")

    assert result["total_attempts"] == 4
    assert result["unique_students"] == 2
    assert result["mean_score"] == pytest.approx(70.0)
    assert result["median_score"] == pytest.approx(70.0)
    assert result["pass_rate"] == pytest.approx(1.0)  # bool_or → both students have a pass
    assert len(result["histogram"]) == 11
    assert sum(b["count"] for b in result["histogram"]) == 2


async def test_first_headline(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """grading_method='first' → headline A=40, B=60 (attempt #1 each)."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["quiz_id"], "first")

    assert result["total_attempts"] == 4
    assert result["unique_students"] == 2
    assert result["mean_score"] == pytest.approx(50.0)  # (40+60)/2
    assert result["pass_rate"] == pytest.approx(0.0)  # neither first attempt passed
    assert sum(b["count"] for b in result["histogram"]) == 2


async def test_last_headline(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """grading_method='last' → headline A=100, B=80 (attempt #2 each)."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["quiz_id"], "last")

    assert result["total_attempts"] == 4
    assert result["unique_students"] == 2
    assert result["mean_score"] == pytest.approx(90.0)  # (100+80)/2
    assert result["pass_rate"] == pytest.approx(1.0)
    assert sum(b["count"] for b in result["histogram"]) == 2


async def test_zero_attempts(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    """A quiz with no attempts → zeroed counts, None stats, empty histogram."""
    async with session_factory() as db:
        result = await quiz_results_summary(db, scenario["empty_quiz_id"], "highest")

    assert result["total_attempts"] == 0
    assert result["unique_students"] == 0
    assert result["mean_score"] is None
    assert result["median_score"] is None
    assert result["p25"] is None
    assert result["p75"] is None
    assert result["pass_rate"] is None
    assert result["mean_time_seconds"] is None
    assert len(result["histogram"]) == 11
    assert sum(b["count"] for b in result["histogram"]) == 0


async def test_invalid_grading_method_raises(
    session_factory: async_sessionmaker[AsyncSession],
    scenario: dict[str, uuid.UUID],
) -> None:
    async with session_factory() as db:
        with pytest.raises(ValueError):
            await quiz_results_summary(db, scenario["quiz_id"], "bogus")


# ---------------------------------------------------------------------------
# quiz_question_breakdown (Task 1.3 + 1.4)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def breakdown_scenario(engine: AsyncEngine) -> AsyncIterator[dict[str, uuid.UUID]]:
    """Seed one quiz with 4 questions and 3 students' completed attempts.

    Questions:
      Q1 (pos 1, multiple_choice, B correct) — 2 correct / 1 wrong  -> rate 0.667
      Q2 (pos 2, multiple_choice, B correct) — 0 correct / 3 wrong  -> rate 0.0
      Q3 (pos 3, short_answer)               — 1 correct / 3 answered -> rate 0.333
      Q4 (pos 4, multiple_choice, B correct) — 0 answers (unreached) -> rate None

    Every MCQ has options A/B/C/D. The distractor students pick is C.

    Completed answers (status submitted/graded):
      student A: Q1->B(correct), Q2->C(wrong), Q3->text(correct)
      student B: Q1->B(correct), Q2->C(wrong), Q3->text(wrong)
      student C: Q1->C(wrong),   Q2->C(wrong), Q3->text(wrong)

    Plus one in_progress attempt (student A) answering Q1->B(correct)
    that MUST be excluded — if counted Q1 would be 3/4.
    """
    org = uuid.uuid4()
    teacher = uuid.uuid4()
    student_a = uuid.uuid4()
    student_b = uuid.uuid4()
    student_c = uuid.uuid4()
    course = uuid.uuid4()
    module = uuid.uuid4()
    quiz = uuid.uuid4()

    q1 = uuid.uuid4()
    q2 = uuid.uuid4()
    q3 = uuid.uuid4()
    q4 = uuid.uuid4()

    # option ids keyed by (question, option_key)
    opts: dict[tuple[uuid.UUID, str], uuid.UUID] = {}
    now = datetime.now(UTC)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Breakdown Org', 'active')"
            ),
            {"id": org, "slug": f"bd-org-{org.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO users (id, primary_email, status) VALUES "
                "(:t, :te, 'active'), (:a, :ae, 'active'), "
                "(:b, :be, 'active'), (:c, :ce, 'active')"
            ),
            {
                "t": teacher,
                "a": student_a,
                "b": student_b,
                "c": student_c,
                "te": f"t-{teacher.hex[:8]}@test.local",
                "ae": f"a-{student_a.hex[:8]}@test.local",
                "be": f"b-{student_b.hex[:8]}@test.local",
                "ce": f"c-{student_c.hex[:8]}@test.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :u, :slug, 'Breakdown Course', 'published')"
            ),
            {"id": course, "org": org, "u": teacher, "slug": f"bd-course-{course.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :c, 'Module', 1, 'published')"
            ),
            {"id": module, "c": course},
        )
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status) "
                "VALUES (:q, :c, :m, 'Breakdown Quiz', 'published')"
            ),
            {"q": quiz, "c": course, "m": module},
        )

        questions = [
            (q1, 1, "multiple_choice", "Q1 prompt"),
            (q2, 2, "multiple_choice", "Q2 prompt"),
            (q3, 3, "short_answer", "Q3 prompt"),
            (q4, 4, "multiple_choice", "Q4 prompt"),
        ]
        for qid, position, qtype, prompt in questions:
            await conn.execute(
                text(
                    "INSERT INTO quiz_questions "
                    "(id, quiz_id, position, question_type, prompt_text, review_status) "
                    "VALUES (:id, :qz, :p, :qt, :pt, 'approved')"
                ),
                {"id": qid, "qz": quiz, "p": position, "qt": qtype, "pt": prompt},
            )

        # MCQ options for Q1, Q2, Q4 (A/B/C/D, B correct)
        for qid in (q1, q2, q4):
            for option_key, option_text, is_correct, position in (
                ("A", "opt A", False, 1),
                ("B", "opt B", True, 2),
                ("C", "opt C", False, 3),
                ("D", "opt D", False, 4),
            ):
                oid = uuid.uuid4()
                opts[(qid, option_key)] = oid
                await conn.execute(
                    text(
                        "INSERT INTO quiz_question_options "
                        "(id, question_id, option_key, option_text, is_correct, position) "
                        "VALUES (:id, :q, :k, :t, :c, :p)"
                    ),
                    {
                        "id": oid,
                        "q": qid,
                        "k": option_key,
                        "t": option_text,
                        "c": is_correct,
                        "p": position,
                    },
                )

        # Completed attempts + answers.
        completed = [
            (
                student_a,
                1,
                "graded",
                {
                    q1: ("B", True, None),
                    q2: ("C", False, None),
                    q3: (None, True, "a correct text"),
                },
            ),
            (
                student_b,
                1,
                "submitted",
                {
                    q1: ("B", True, None),
                    q2: ("C", False, None),
                    q3: (None, False, "b wrong text"),
                },
            ),
            (
                student_c,
                1,
                "graded",
                {
                    q1: ("C", False, None),
                    q2: ("C", False, None),
                    q3: (None, False, "c wrong text"),
                },
            ),
        ]
        for sid, num, status, answers in completed:
            attempt_id = uuid.uuid4()
            await conn.execute(
                text(
                    "INSERT INTO quiz_attempts "
                    "(id, quiz_id, student_id, attempt_number, status, started_at, submitted_at) "
                    "VALUES (:id, :q, :s, :n, :st, :start, :submit)"
                ),
                {
                    "id": attempt_id,
                    "q": quiz,
                    "s": sid,
                    "n": num,
                    "st": status,
                    "start": now - timedelta(hours=2),
                    "submit": now - timedelta(hours=1),
                },
            )
            for qid, (opt_key, is_correct, answer_text) in answers.items():
                sel = opts[(qid, opt_key)] if opt_key is not None else None
                await conn.execute(
                    text(
                        "INSERT INTO quiz_attempt_answers "
                        "(id, attempt_id, question_id, selected_option_id, "
                        " answer_text, is_correct, points_awarded) "
                        "VALUES (:id, :a, :q, :sel, :txt, :c, 1)"
                    ),
                    {
                        "id": uuid.uuid4(),
                        "a": attempt_id,
                        "q": qid,
                        "sel": sel,
                        "txt": answer_text,
                        "c": is_correct,
                    },
                )

        # in_progress attempt — answers here MUST be excluded.
        ip_attempt = uuid.uuid4()
        await conn.execute(
            text(
                "INSERT INTO quiz_attempts "
                "(id, quiz_id, student_id, attempt_number, status, started_at) "
                "VALUES (:id, :q, :s, :n, 'in_progress', :start)"
            ),
            {"id": ip_attempt, "q": quiz, "s": student_a, "n": 2, "start": now},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_attempt_answers "
                "(id, attempt_id, question_id, selected_option_id, is_correct, points_awarded) "
                "VALUES (:id, :a, :q, :sel, TRUE, 1)"
            ),
            {"id": uuid.uuid4(), "a": ip_attempt, "q": q1, "sel": opts[(q1, "B")]},
        )

    yield {
        "quiz_id": quiz,
        "q1": q1,
        "q2": q2,
        "q3": q3,
        "q4": q4,
    }

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM quiz_attempt_answers WHERE attempt_id IN ("
                "SELECT id FROM quiz_attempts WHERE quiz_id = :q)"
            ),
            {"q": quiz},
        )
        await conn.execute(text("DELETE FROM quiz_attempts WHERE quiz_id = :q"), {"q": quiz})
        await conn.execute(
            text(
                "DELETE FROM quiz_question_options WHERE question_id IN ("
                "SELECT id FROM quiz_questions WHERE quiz_id = :q)"
            ),
            {"q": quiz},
        )
        await conn.execute(text("DELETE FROM quiz_questions WHERE quiz_id = :q"), {"q": quiz})
        await conn.execute(text("DELETE FROM quizzes WHERE id = :q"), {"q": quiz})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :a, :b, :c)"),
            {"t": teacher, "a": student_a, "b": student_b, "c": student_c},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org})


async def test_breakdown_order_hardest_first(
    session_factory: async_sessionmaker[AsyncSession],
    breakdown_scenario: dict[str, uuid.UUID],
) -> None:
    """Rows ordered by correctness_rate ASC; unanswered (None) sorts last."""
    async with session_factory() as db:
        rows = await quiz_question_breakdown(db, breakdown_scenario["quiz_id"])

    assert [r["question_id"] for r in rows] == [
        breakdown_scenario["q2"],  # 0.0
        breakdown_scenario["q3"],  # 0.333
        breakdown_scenario["q1"],  # 0.667
        breakdown_scenario["q4"],  # None (unanswered) last
    ]


async def test_breakdown_counts_and_rates(
    session_factory: async_sessionmaker[AsyncSession],
    breakdown_scenario: dict[str, uuid.UUID],
) -> None:
    async with session_factory() as db:
        rows = await quiz_question_breakdown(db, breakdown_scenario["quiz_id"])
    by_id = {r["question_id"]: r for r in rows}

    q1 = by_id[breakdown_scenario["q1"]]
    assert q1["prompt"] == "Q1 prompt"
    assert q1["correct_count"] == 2
    assert q1["answered_count"] == 3  # in_progress answer excluded (else 4)
    assert q1["correctness_rate"] == pytest.approx(2 / 3)

    q2 = by_id[breakdown_scenario["q2"]]
    assert q2["correct_count"] == 0
    assert q2["answered_count"] == 3
    assert q2["correctness_rate"] == pytest.approx(0.0)

    q3 = by_id[breakdown_scenario["q3"]]
    assert q3["correct_count"] == 1
    assert q3["answered_count"] == 3
    assert q3["correctness_rate"] == pytest.approx(1 / 3)


async def test_breakdown_option_distribution(
    session_factory: async_sessionmaker[AsyncSession],
    breakdown_scenario: dict[str, uuid.UUID],
) -> None:
    async with session_factory() as db:
        rows = await quiz_question_breakdown(db, breakdown_scenario["quiz_id"])
    by_id = {r["question_id"]: r for r in rows}

    # Q1: A/B/C/D, B correct. 2 chose B (correct), 1 chose C (distractor).
    q1_dist = by_id[breakdown_scenario["q1"]]["option_distribution"]
    assert len(q1_dist) == 4
    q1_by_key = {o["option_key"]: o for o in q1_dist}
    assert q1_by_key["A"]["chosen_count"] == 0
    assert q1_by_key["B"]["chosen_count"] == 2
    assert q1_by_key["B"]["is_correct"] is True
    assert q1_by_key["C"]["chosen_count"] == 1  # tempting distractor
    assert q1_by_key["C"]["is_correct"] is False
    assert q1_by_key["D"]["chosen_count"] == 0
    assert (
        sum(o["chosen_count"] for o in q1_dist) == by_id[breakdown_scenario["q1"]]["answered_count"]
    )

    # Q2: all 3 chose C.
    q2_dist = by_id[breakdown_scenario["q2"]]["option_distribution"]
    q2_by_key = {o["option_key"]: o for o in q2_dist}
    assert q2_by_key["C"]["chosen_count"] == 3
    assert sum(o["chosen_count"] for o in q2_dist) == 3


async def test_breakdown_short_answer_has_empty_distribution(
    session_factory: async_sessionmaker[AsyncSession],
    breakdown_scenario: dict[str, uuid.UUID],
) -> None:
    async with session_factory() as db:
        rows = await quiz_question_breakdown(db, breakdown_scenario["quiz_id"])
    by_id = {r["question_id"]: r for r in rows}
    assert by_id[breakdown_scenario["q3"]]["option_distribution"] == []


async def test_breakdown_unanswered_question_present(
    session_factory: async_sessionmaker[AsyncSession],
    breakdown_scenario: dict[str, uuid.UUID],
) -> None:
    """A question nobody reached still appears with zeroed counts / None rate."""
    async with session_factory() as db:
        rows = await quiz_question_breakdown(db, breakdown_scenario["quiz_id"])
    by_id = {r["question_id"]: r for r in rows}

    q4 = by_id[breakdown_scenario["q4"]]
    assert q4["answered_count"] == 0
    assert q4["correct_count"] == 0
    assert q4["correctness_rate"] is None
    # MCQ options still enumerated, each chosen_count == 0.
    assert len(q4["option_distribution"]) == 4
    assert all(o["chosen_count"] == 0 for o in q4["option_distribution"])
