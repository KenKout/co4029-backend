"""Integration tests for the teacher review-queue drill-down (GAP-1 invariant).

The dashboard badge counts and the ``/teacher/dashboard/review-queue/{kind}``
drill-down rows are computed by TWO different queries (aggregate COUNT vs
grouped COUNT). They must agree: for ``quiz-cards`` / ``interview-questions``
/ ``materials`` the expanded rows' counts must SUM to the badge, and for
``missing-texp`` (a DISTINCT-quiz badge) the row COUNT must equal the badge.
Editing one query without its counterpart breaks the invariant silently —
that is exactly what this file guards.

Also asserts the FIX-SEC-1-style perimeter: rows are scoped to the caller's
authorable courses; a sibling course in the same org must never leak in.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest_asyncio
from alembic import command
from alembic.config import Config
from conftest import SeededUsers
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import abridgeai.ai.models  # noqa: F401  -- register processing_jobs
import abridgeai.features.access_control.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.quizzes.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token
from abridgeai.features.courses.routers import authoring as courses_authoring_router


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
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(courses_authoring_router.router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


async def _seed_session(engine: AsyncEngine, user_id: uuid.UUID) -> uuid.UUID:
    sid = uuid.uuid4()
    expires_at = datetime.now(tz=UTC) + timedelta(hours=1)
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :hash, :exp)"
            ),
            {"id": sid, "uid": user_id, "hash": f"revq-{sid.hex}", "exp": expires_at},
        )
    return sid


@pytest_asyncio.fixture
async def teacher_bearer(engine: AsyncEngine, seeded_users: SeededUsers) -> AsyncIterator[str]:
    sid = await _seed_session(engine, seeded_users.teacher_id)
    yield create_access_token(user_id=seeded_users.teacher_id, session_id=sid)
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM auth_sessions WHERE id = :id"), {"id": sid})


@pytest_asyncio.fixture
async def review_scenario(
    engine: AsyncEngine, seeded_users: SeededUsers
) -> AsyncIterator[dict[str, uuid.UUID]]:
    """Seed review work on the in-scope course and a sibling (out-of-scope).

    Course A (``seeded_users.course_id``) is the teacher's assigned course;
    Course B is a sibling in the same org with NO teacher scope — the
    FIX-SEC-1 perimeter. Expected badge numbers (see the test):

    * quiz-cards: 2 pending questions on quiz A; sibling quiz B carries 2
      pending that must NOT count.
    * interview-questions: 2 pending on config A; sibling has 1.
    * materials: lesson A has one ingested version, no quiz source; sibling
      lesson B has one too (excluded).
    * missing-texp: published quiz A holds 2 approved questions with NULL
      t_exp (counts 1 quiz); a calibrated published quiz and a sibling
      published quiz missing t_exp must not count.
    """
    course_a = seeded_users.course_id
    module_a = uuid.uuid4()
    module_b = uuid.uuid4()
    course_b = uuid.uuid4()
    lesson_a = uuid.uuid4()
    lesson_b = uuid.uuid4()
    storage_obj = uuid.uuid4()
    suffix = course_b.hex[:8]

    quiz_a = uuid.uuid4()
    quiz_b = uuid.uuid4()
    quiz_texp = uuid.uuid4()
    quiz_texp_b = uuid.uuid4()
    quiz_calibrated = uuid.uuid4()
    cfg_a = uuid.uuid4()
    cfg_b = uuid.uuid4()
    material_a = uuid.uuid4()
    version_a = uuid.uuid4()
    job_a = uuid.uuid4()
    material_b = uuid.uuid4()
    version_b = uuid.uuid4()
    job_b = uuid.uuid4()

    async with engine.begin() as conn:
        # --- Course B (sibling, out of scope) + modules -------------------
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'Sibling Course', 'draft')"
            ),
            {
                "id": course_b,
                "org": seeded_users.organization_id,
                "owner": seeded_users.manager_id,
                "slug": f"review-sibling-{suffix}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) VALUES "
                "(:ma, :ca, 'Module A', 1, 'draft'), "
                "(:mb, :cb, 'Module B', 1, 'draft')"
            ),
            {"ma": module_a, "mb": module_b, "ca": course_a, "cb": course_b},
        )
        await conn.execute(
            text(
                "INSERT INTO lessons (id, module_id, slug, title, status) VALUES "
                "(:la, :ma, 'lesson-a', 'Lesson A', 'draft'), "
                "(:lb, :mb, 'lesson-b', 'Lesson B', 'draft')"
            ),
            {"la": lesson_a, "lb": lesson_b, "ma": module_a, "mb": module_b},
        )
        await conn.execute(
            text("INSERT INTO storage_objects (id, bucket, object_key) VALUES (:id, :b, :k)"),
            {"id": storage_obj, "b": "test-bucket", "k": f"review/{suffix}.pdf"},
        )

        # --- Quiz A: 2 pending + 1 approved-with-t_exp ---------------------
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent) VALUES "
                "(:qa, :ca, :ma, 'Quiz A Pending', 'draft', 70.00)"
            ),
            {"qa": quiz_a, "ca": course_a, "ma": module_a},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions (id, quiz_id, position, question_type, "
                "prompt_text, expected_response_time_ms, source_refs, review_status) VALUES "
                "(:q1, :qa, 1, 'multiple_choice', 'Pending Q1', NULL, '[]'::jsonb, 'pending'), "
                "(:q2, :qa, 2, 'multiple_choice', 'Pending Q2', NULL, '[]'::jsonb, 'pending'), "
                "(:q3, :qa, 3, 'multiple_choice', 'Approved Q3', 45000, '[]'::jsonb, 'approved')"
            ),
            {
                "q1": uuid.uuid4(),
                "q2": uuid.uuid4(),
                "q3": uuid.uuid4(),
                "qa": quiz_a,
            },
        )
        # --- Quiz B (sibling): 2 pending — must NOT count ------------------
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent) VALUES "
                "(:qb, :cb, :mb, 'Quiz B Pending', 'draft', 70.00)"
            ),
            {"qb": quiz_b, "cb": course_b, "mb": module_b},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions (id, quiz_id, position, question_type, "
                "prompt_text, expected_response_time_ms, source_refs, review_status) VALUES "
                "(:q1, :qb, 1, 'multiple_choice', 'Sibling Pending Q1', NULL, "
                "'[]'::jsonb, 'pending'), "
                "(:q2, :qb, 2, 'multiple_choice', 'Sibling Pending Q2', NULL, "
                "'[]'::jsonb, 'pending')"
            ),
            {"q1": uuid.uuid4(), "q2": uuid.uuid4(), "qb": quiz_b},
        )

        # --- Interview config A: 2 pending ---------------------------------
        await conn.execute(
            text(
                "INSERT INTO interview_configs (id, course_id, module_id, title, status) "
                "VALUES (:ia, :ca, :ma, 'IC A', 'draft')"
            ),
            {"ia": cfg_a, "ca": course_a, "ma": module_a},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions (id, interview_config_id, position, "
                "question_type, prompt_text, review_status, source_refs_json) VALUES "
                "(:i1, :ia, 1, 'technical', 'Pending I1', 'pending', '[]'::jsonb), "
                "(:i2, :ia, 2, 'behavioral', 'Pending I2', 'pending', '[]'::jsonb)"
            ),
            {"i1": uuid.uuid4(), "i2": uuid.uuid4(), "ia": cfg_a},
        )
        # --- Interview config B (sibling): 1 pending — must NOT count ------
        await conn.execute(
            text(
                "INSERT INTO interview_configs (id, course_id, module_id, title, status) "
                "VALUES (:ib, :cb, :mb, 'IC B', 'draft')"
            ),
            {"ib": cfg_b, "cb": course_b, "mb": module_b},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions (id, interview_config_id, position, "
                "question_type, prompt_text, review_status, source_refs_json) VALUES "
                "(:i1, :ib, 1, 'technical', 'Sibling Pending I1', 'pending', '[]'::jsonb)"
            ),
            {"i1": uuid.uuid4(), "ib": cfg_b},
        )

        # --- Materials: lesson A ingested, no quiz source ------------------
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:m, :l, 'Mat A', 'pdf')"
            ),
            {"m": material_a, "l": lesson_a},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions (id, material_id, "
                "storage_object_id, version_no, processing_status) VALUES "
                "(:v, :m, :so, 1, 'ready')"
            ),
            {"v": version_a, "m": material_a, "so": storage_obj},
        )
        await conn.execute(
            text(
                "INSERT INTO processing_jobs (id, entity_type, entity_id, job_type, "
                "status, progress_percent) VALUES "
                "(:j, 'material_version', :v, 'full_pipeline', 'completed', 100)"
            ),
            {"j": job_a, "v": version_a},
        )
        # --- Materials on sibling lesson B — must NOT count ----------------
        await conn.execute(
            text(
                "INSERT INTO learning_materials (id, lesson_id, title, material_type) "
                "VALUES (:m, :l, 'Mat B', 'pdf')"
            ),
            {"m": material_b, "l": lesson_b},
        )
        await conn.execute(
            text(
                "INSERT INTO learning_material_versions (id, material_id, "
                "storage_object_id, version_no, processing_status) VALUES "
                "(:v, :m, :so, 1, 'ready')"
            ),
            {"v": version_b, "m": material_b, "so": storage_obj},
        )
        await conn.execute(
            text(
                "INSERT INTO processing_jobs (id, entity_type, entity_id, job_type, "
                "status, progress_percent) VALUES "
                "(:j, 'material_version', :v, 'full_pipeline', 'completed', 100)"
            ),
            {"j": job_b, "v": version_b},
        )

        # --- Published quiz missing t_exp (course A): 2 approved, NULL/0 ---
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent) VALUES "
                "(:q, :ca, :ma, 'Quiz Missing Texp', 'published', 70.00)"
            ),
            {"q": quiz_texp, "ca": course_a, "ma": module_a},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions (id, quiz_id, position, question_type, "
                "prompt_text, expected_response_time_ms, source_refs, review_status) VALUES "
                "(:t1, :q, 1, 'multiple_choice', 'Texp NULL', NULL, '[]'::jsonb, 'approved'), "
                "(:t2, :q, 2, 'multiple_choice', 'Texp NULL 2', NULL, '[]'::jsonb, 'approved')"
            ),
            {"t1": uuid.uuid4(), "t2": uuid.uuid4(), "q": quiz_texp},
        )
        # --- Calibrated published quiz (course A) — must NOT count ---------
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent) VALUES "
                "(:q, :ca, :ma, 'Quiz Calibrated', 'published', 70.00)"
            ),
            {"q": quiz_calibrated, "ca": course_a, "ma": module_a},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions (id, quiz_id, position, question_type, "
                "prompt_text, expected_response_time_ms, source_refs, review_status) VALUES "
                "(:c1, :q, 1, 'multiple_choice', 'Calibrated Q', 30000, '[]'::jsonb, 'approved')"
            ),
            {"c1": uuid.uuid4(), "q": quiz_calibrated},
        )
        # --- Published quiz missing t_exp on sibling — must NOT count ------
        await conn.execute(
            text(
                "INSERT INTO quizzes (id, course_id, module_id, title, status, "
                "passing_score_percent) VALUES "
                "(:q, :cb, :mb, 'Sibling Missing Texp', 'published', 70.00)"
            ),
            {"q": quiz_texp_b, "cb": course_b, "mb": module_b},
        )
        await conn.execute(
            text(
                "INSERT INTO quiz_questions (id, quiz_id, position, question_type, "
                "prompt_text, expected_response_time_ms, source_refs, review_status) VALUES "
                "(:t1, :q, 1, 'multiple_choice', 'Sibling Texp NULL', NULL, "
                "'[]'::jsonb, 'approved')"
            ),
            {"t1": uuid.uuid4(), "q": quiz_texp_b},
        )

    yield {
        "course_a": course_a,
        "course_b": course_b,
        "module_a": module_a,
        "lesson_a": lesson_a,
        "quiz_a": quiz_a,
        "cfg_a": cfg_a,
        "quiz_texp": quiz_texp,
    }

    # Teardown: drop the fixture's OWN rows (both courses) so the next test
    # re-seeds cleanly. Course-B children are deleted by course_id; the
    # seeded course-A rows are deleted by the explicit ids this fixture
    # created. Leftover processing_jobs / storage_objects rows are wiped by
    # the session-scoped purge on the next run.
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "DELETE FROM quiz_questions WHERE quiz_id IN "
                "(SELECT id FROM quizzes WHERE course_id = :id)"
            ),
            {"id": course_b},
        )
        await conn.execute(text("DELETE FROM quizzes WHERE course_id = :id"), {"id": course_b})
        await conn.execute(
            text(
                "DELETE FROM interview_questions WHERE interview_config_id IN "
                "(SELECT id FROM interview_configs WHERE course_id = :id)"
            ),
            {"id": course_b},
        )
        await conn.execute(
            text("DELETE FROM interview_configs WHERE course_id = :id"), {"id": course_b}
        )
        await conn.execute(
            text(
                "DELETE FROM learning_material_versions WHERE material_id IN "
                "(SELECT id FROM learning_materials WHERE lesson_id IN "
                " (SELECT id FROM lessons WHERE module_id IN "
                "  (SELECT id FROM modules WHERE course_id = :id)))"
            ),
            {"id": course_b},
        )
        await conn.execute(
            text(
                "DELETE FROM learning_materials WHERE lesson_id IN "
                "(SELECT id FROM lessons WHERE module_id IN "
                " (SELECT id FROM modules WHERE course_id = :id))"
            ),
            {"id": course_b},
        )
        await conn.execute(
            text(
                "DELETE FROM lessons WHERE module_id IN "
                "(SELECT id FROM modules WHERE course_id = :id)"
            ),
            {"id": course_b},
        )
        await conn.execute(text("DELETE FROM modules WHERE course_id = :id"), {"id": course_b})
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course_b})
        # --- Fixture rows on the seeded course A --------------------------
        await conn.execute(
            text("DELETE FROM quiz_questions WHERE quiz_id = ANY(:qids)"),
            {"qids": [quiz_a, quiz_texp, quiz_calibrated]},
        )
        await conn.execute(
            text("DELETE FROM quizzes WHERE id = ANY(:qids)"),
            {"qids": [quiz_a, quiz_texp, quiz_calibrated]},
        )
        await conn.execute(
            text("DELETE FROM interview_questions WHERE interview_config_id = :cid"),
            {"cid": cfg_a},
        )
        await conn.execute(
            text("DELETE FROM interview_configs WHERE id = :cid"), {"cid": cfg_a}
        )
        await conn.execute(
            text("DELETE FROM learning_material_versions WHERE material_id = :mid"),
            {"mid": material_a},
        )
        await conn.execute(
            text("DELETE FROM learning_materials WHERE id = :mid"), {"mid": material_a}
        )
        await conn.execute(text("DELETE FROM lessons WHERE id = :lid"), {"lid": lesson_a})
        await conn.execute(text("DELETE FROM modules WHERE id = :mid"), {"mid": module_a})


async def _stats(client: httpx.AsyncClient, bearer: str) -> dict:
    resp = await client.get(
        "/api/v1/teacher/dashboard/stats", headers={"Authorization": f"Bearer {bearer}"}
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def _drilldown(client: httpx.AsyncClient, bearer: str, kind: str) -> list[dict]:
    resp = await client.get(
        f"/api/v1/teacher/dashboard/review-queue/{kind}",
        headers={"Authorization": f"Bearer {bearer}"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


async def test_review_queue_rows_sum_to_badge(
    client: httpx.AsyncClient, teacher_bearer: str, review_scenario: dict
) -> None:
    """Expanded drill-down rows must SUM to the dashboard badge per kind.

    This is the invariant that breaks when one query is edited and its
    counterpart is not: the badge is a bare COUNT, the drill-down is a
    grouped COUNT — they are separate statements and nothing but this test
    ties them together.
    """
    stats = await _stats(client, teacher_bearer)
    assert stats["quiz_cards_pending_review"] == 2
    assert stats["interview_questions_pending_review"] == 2
    assert stats["materials_ready_for_quiz_gen"] == 1

    for kind, badge_key in (
        ("quiz-cards", "quiz_cards_pending_review"),
        ("interview-questions", "interview_questions_pending_review"),
        ("materials", "materials_ready_for_quiz_gen"),
    ):
        rows = await _drilldown(client, teacher_bearer, kind)
        assert sum(row["count"] for row in rows) == stats[badge_key], (
            f"{kind}: drill-down rows ({rows}) do not sum to badge "
            f"{stats[badge_key]}"
        )
        # Every row belongs to the in-scope course — never the sibling.
        assert all(row["course_id"] == str(review_scenario["course_a"]) for row in rows)
        assert not any(row["course_id"] == str(review_scenario["course_b"]) for row in rows)


async def test_review_queue_row_shapes(
    client: httpx.AsyncClient, teacher_bearer: str, review_scenario: dict
) -> None:
    """Drill-down rows carry the deep-link coordinates the UI renders."""
    quiz_rows = await _drilldown(client, teacher_bearer, "quiz-cards")
    assert len(quiz_rows) == 1
    assert quiz_rows[0]["target_id"] == str(review_scenario["quiz_a"])
    assert quiz_rows[0]["target_title"] == "Quiz A Pending"
    assert quiz_rows[0]["module_title"] == "Module A"
    assert quiz_rows[0]["count"] == 2

    interview_rows = await _drilldown(client, teacher_bearer, "interview-questions")
    assert len(interview_rows) == 1
    assert interview_rows[0]["target_id"] == str(review_scenario["cfg_a"])
    assert interview_rows[0]["count"] == 2

    material_rows = await _drilldown(client, teacher_bearer, "materials")
    assert len(material_rows) == 1
    assert material_rows[0]["target_id"] == str(review_scenario["lesson_a"])
    assert material_rows[0]["count"] == 1


async def test_missing_texp_drilldown_len_equals_badge(
    client: httpx.AsyncClient, teacher_bearer: str, review_scenario: dict
) -> None:
    """missing-texp badge counts DISTINCT quizzes, so len(rows) == badge.

    The per-row count is the number of uncalibrated questions inside that
    quiz (the work the teacher clears), so it is deliberately NOT summed.
    """
    stats = await _stats(client, teacher_bearer)
    assert stats["published_quizzes_missing_texp"] == 1

    rows = await _drilldown(client, teacher_bearer, "missing-texp")
    assert len(rows) == stats["published_quizzes_missing_texp"]
    assert len(rows) == 1
    assert rows[0]["target_id"] == str(review_scenario["quiz_texp"])
    assert rows[0]["target_title"] == "Quiz Missing Texp"
    assert rows[0]["count"] == 2  # both approved questions have NULL t_exp
    assert rows[0]["course_id"] == str(review_scenario["course_a"])


async def test_review_queue_unknown_kind_rejected(
    client: httpx.AsyncClient, teacher_bearer: str,
) -> None:
    # The router's `kind: Literal[...]` rejects unknown kinds at validation,
    # so an unregistered category is 422 before it ever reaches the service
    # (the service's NotFoundError fallback is unreachable over HTTP).
    resp = await client.get(
        "/api/v1/teacher/dashboard/review-queue/not-a-kind",
        headers={"Authorization": f"Bearer {teacher_bearer}"},
    )
    assert resp.status_code == 422
