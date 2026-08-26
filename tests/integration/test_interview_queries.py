"""Integration tests for the interviews queries layer (T6.3).

Plan §6.3. Asserts:

* :mod:`abridgeai.features.interviews.queries.published` excludes
  drafts/archived/soft-deleted, filters questions by review_status,
  and enforces the max-attempts gate.
* :mod:`abridgeai.features.interviews.queries.authoring` returns all
  states and provides position helpers.
* :mod:`abridgeai.features.interviews.queries.sessions` enforces
  one-active-session-per-config and surfaces per-session artefacts
  (messages, evaluations, gap reports).
* The queries directory is single-namespace + sql/ co-located —
  no ``orm/`` / ``raw/`` mechanism split (locked decision).
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
from abridgeai.features.interviews.queries import (
    get_active_session,
    get_gap_report_for_session,
    get_interview_for_authoring,
    get_interview_for_taking,
    get_interview_with_full_config,
    get_outcome_evaluations,
    get_published_interview,
    get_session,
    get_session_attempt_number,
    get_session_with_responses,
    get_user_interview_sessions,
    list_interviews_for_course,
    list_outcomes_for_config,
    list_published_interviews_for_course,
    list_published_interviews_for_module,
    list_published_questions_for_config,
    list_questions_for_config,
    list_session_messages,
    next_outcome_position,
    next_question_position,
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
async def session_factory(
    engine: AsyncEngine,
) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def fixture_data(engine: AsyncEngine) -> AsyncIterator[dict]:
    """Seed an interview tree with mixed statuses, sessions, evaluations."""

    org = uuid.uuid4()
    teacher = uuid.uuid4()
    student = uuid.uuid4()
    fresh_student = uuid.uuid4()
    course = uuid.uuid4()
    module_a = uuid.uuid4()
    module_b = uuid.uuid4()

    cfg_pub = uuid.uuid4()
    cfg_draft = uuid.uuid4()
    cfg_archived = uuid.uuid4()
    cfg_soft_deleted = uuid.uuid4()
    cfg_capped = uuid.uuid4()
    cfg_module_b = uuid.uuid4()

    outcome_pub = uuid.uuid4()
    q_approved = uuid.uuid4()
    q_pending = uuid.uuid4()
    q_rejected = uuid.uuid4()

    session_active = uuid.uuid4()
    session_completed = uuid.uuid4()
    session_other_user = uuid.uuid4()

    session_question = uuid.uuid4()
    msg_user = uuid.uuid4()
    msg_ai = uuid.uuid4()
    eval_row = uuid.uuid4()
    gap_row = uuid.uuid4()

    now = datetime.now(UTC)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO organizations (id, slug, name, status) "
                "VALUES (:id, :slug, 'Interview Org', 'active')"
            ),
            {"id": org, "slug": f"iv-org-{org.hex[:8]}"},
        )
        await conn.execute(
            text(
                "INSERT INTO users (id, primary_email, status) VALUES "
                "(:t, :te, 'active'), (:s, :se, 'active'), (:f, :fe, 'active')"
            ),
            {
                "t": teacher,
                "s": student,
                "f": fresh_student,
                "te": f"t-{teacher.hex[:8]}@test.local",
                "se": f"s-{student.hex[:8]}@test.local",
                "fe": f"f-{fresh_student.hex[:8]}@test.local",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :u, :slug, 'Interview Course', 'published')"
            ),
            {
                "id": course,
                "org": org,
                "u": teacher,
                "slug": f"iv-course-{course.hex[:8]}",
            },
        )
        await conn.execute(
            text(
                "INSERT INTO modules "
                "(id, course_id, title, position, status) VALUES "
                "(:a, :c, 'Module A', 1, 'published'), "
                "(:b, :c, 'Module B', 2, 'published')"
            ),
            {"a": module_a, "b": module_b, "c": course},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_configs (id, course_id, module_id, title, status, max_attempts, slug) VALUES (:p, :c, :m, 'Pub Interview', 'published', NULL, 'slug-' || uuid_generate_v4()::text), (:d, :c, :m, 'Draft Interview', 'draft', NULL, 'slug-' || uuid_generate_v4()::text), (:a, :c, :m, 'Archived Interview', 'archived', NULL, 'slug-' || uuid_generate_v4()::text), (:sd, :c, :m, 'Deleted Interview', 'published', NULL, 'slug-' || uuid_generate_v4()::text), (:cp, :c, :m, 'Capped Interview', 'published', 2, 'slug-' || uuid_generate_v4()::text), (:mb, :c, :mb_id, 'Module B Interview', 'published', NULL, 'slug-' || uuid_generate_v4()::text);"
            ),
            {
                "p": cfg_pub,
                "d": cfg_draft,
                "a": cfg_archived,
                "sd": cfg_soft_deleted,
                "cp": cfg_capped,
                "mb": cfg_module_b,
                "c": course,
                "m": module_a,
                "mb_id": module_b,
            },
        )
        await conn.execute(
            text("UPDATE interview_configs SET deleted_at = :ts WHERE id = :id"),
            {"ts": now - timedelta(days=1), "id": cfg_soft_deleted},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_outcomes ("
                "id, interview_config_id, position, outcome_text, "
                "outcome_type, importance_weight) VALUES "
                "(:o, :c, 1, 'Demonstrate concept X', 'knowledge', 3)"
            ),
            {"o": outcome_pub, "c": cfg_pub},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_questions ("
                "id, interview_config_id, position, question_type, "
                "prompt_text, review_status, source_refs_json) VALUES "
                "(:a, :c, 1, 'technical', 'Approved Q?', 'approved', "
                " '[]'::jsonb), "
                "(:p, :c, 2, 'behavioral', 'Pending Q?', 'pending', "
                " '[]'::jsonb), "
                "(:r, :c, 3, 'situational', 'Rejected Q?', 'rejected', "
                " '[]'::jsonb)"
            ),
            {"a": q_approved, "p": q_pending, "r": q_rejected, "c": cfg_pub},
        )
        await conn.execute(
            text(
                "INSERT INTO module_items ("
                "id, module_id, item_type, interview_config_id, position) "
                "VALUES (:i, :m, 'interview', :c, 1)"
            ),
            {"i": uuid.uuid4(), "m": module_a, "c": cfg_pub},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions ("
                "id, interview_config_id, student_id, attempt_number, "
                "status, input_mode, started_at) VALUES "
                "(:a, :c, :s, 1, 'in_progress', 'text', :ts1), "
                "(:cm, :c, :s, 2, 'completed', 'text', :ts2), "
                "(:o, :c, :f, 1, 'in_progress', 'text', :ts3)"
            ),
            {
                "a": session_active,
                "cm": session_completed,
                "o": session_other_user,
                "c": cfg_pub,
                "s": student,
                "f": fresh_student,
                "ts1": now - timedelta(minutes=5),
                "ts2": now - timedelta(hours=2),
                "ts3": now - timedelta(minutes=10),
            },
        )
        # Cap-test: 2 sessions for student against cfg_capped (== max_attempts)
        for n in range(1, 3):
            await conn.execute(
                text(
                    "INSERT INTO interview_sessions ("
                    "id, interview_config_id, student_id, attempt_number, "
                    "status, input_mode, started_at) VALUES "
                    "(:id, :c, :s, :n, 'completed', 'text', :ts)"
                ),
                {
                    "id": uuid.uuid4(),
                    "c": cfg_capped,
                    "s": student,
                    "n": n,
                    "ts": now - timedelta(days=n + 1),
                },
            )
        await conn.execute(
            text(
                "INSERT INTO interview_session_questions ("
                "id, session_id, interview_question_id, sequence_no, "
                "asked_at) VALUES "
                "(:id, :s, :q, 1, :ts)"
            ),
            {
                "id": session_question,
                "s": session_completed,
                "q": q_approved,
                "ts": now - timedelta(hours=2),
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_session_messages ("
                "id, session_id, session_question_id, role, content_text) "
                "VALUES "
                "(:u, :s, :sq, 'user', 'My answer'), "
                "(:a, :s, :sq, 'ai', 'Follow up')"
            ),
            {
                "u": msg_user,
                "a": msg_ai,
                "s": session_completed,
                "sq": session_question,
            },
        )
        await conn.execute(
            text(
                "INSERT INTO interview_outcome_evaluations ("
                "id, session_id, outcome_id, verdict_met, hidden_reasoning) "
                "VALUES (:id, :s, :o, TRUE, 'private rationale')"
            ),
            {"id": eval_row, "s": session_completed, "o": outcome_pub},
        )
        await conn.execute(
            text(
                "INSERT INTO gap_reports ("
                "id, student_id, course_id, module_id, "
                "source_interview_session_id, report_json) "
                "VALUES (:id, :s, :c, :m, :ss, '{}'::jsonb)"
            ),
            {
                "id": gap_row,
                "s": student,
                "c": course,
                "m": module_a,
                "ss": session_completed,
            },
        )

    data = {
        "org": org,
        "teacher": teacher,
        "student": student,
        "fresh_student": fresh_student,
        "course": course,
        "module_a": module_a,
        "module_b": module_b,
        "cfg_pub": cfg_pub,
        "cfg_draft": cfg_draft,
        "cfg_archived": cfg_archived,
        "cfg_soft_deleted": cfg_soft_deleted,
        "cfg_capped": cfg_capped,
        "cfg_module_b": cfg_module_b,
        "outcome_pub": outcome_pub,
        "q_approved": q_approved,
        "q_pending": q_pending,
        "q_rejected": q_rejected,
        "session_active": session_active,
        "session_completed": session_completed,
        "session_other_user": session_other_user,
        "session_question": session_question,
        "msg_user": msg_user,
        "msg_ai": msg_ai,
        "eval_row": eval_row,
        "gap_row": gap_row,
    }
    yield data

    config_ids = [
        cfg_pub,
        cfg_draft,
        cfg_archived,
        cfg_soft_deleted,
        cfg_capped,
        cfg_module_b,
    ]
    async with engine.begin() as conn:
        await conn.execute(text("DELETE FROM gap_reports WHERE id = :id"), {"id": gap_row})
        await conn.execute(
            text(
                "DELETE FROM interview_session_messages WHERE session_id IN ("
                "SELECT id FROM interview_sessions WHERE interview_config_id = "
                "ANY(:ids))"
            ),
            {"ids": config_ids},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_outcome_evaluations WHERE session_id IN ("
                "SELECT id FROM interview_sessions WHERE interview_config_id = "
                "ANY(:ids))"
            ),
            {"ids": config_ids},
        )
        await conn.execute(
            text(
                "DELETE FROM interview_session_questions WHERE session_id IN ("
                "SELECT id FROM interview_sessions WHERE interview_config_id = "
                "ANY(:ids))"
            ),
            {"ids": config_ids},
        )
        await conn.execute(
            text("DELETE FROM interview_sessions WHERE interview_config_id = ANY(:ids)"),
            {"ids": config_ids},
        )
        await conn.execute(
            text("DELETE FROM interview_questions WHERE interview_config_id = ANY(:ids)"),
            {"ids": config_ids},
        )
        await conn.execute(
            text("DELETE FROM interview_outcomes WHERE interview_config_id = ANY(:ids)"),
            {"ids": config_ids},
        )
        await conn.execute(
            text("DELETE FROM module_items WHERE interview_config_id = ANY(:ids)"),
            {"ids": config_ids},
        )
        await conn.execute(
            text("DELETE FROM interview_configs WHERE id = ANY(:ids)"),
            {"ids": config_ids},
        )
        await conn.execute(
            text("DELETE FROM modules WHERE id = ANY(:ids)"),
            {"ids": [module_a, module_b]},
        )
        await conn.execute(text("DELETE FROM courses WHERE id = :id"), {"id": course})
        await conn.execute(
            text("DELETE FROM users WHERE id = ANY(:ids)"),
            {"ids": [teacher, student, fresh_student]},
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :id"), {"id": org})


async def test_get_published_interview_excludes_draft(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        pub = await get_published_interview(session, fixture_data["cfg_pub"])
        draft = await get_published_interview(session, fixture_data["cfg_draft"])
        archived = await get_published_interview(session, fixture_data["cfg_archived"])
    assert pub is not None
    assert pub.status == "published"
    assert draft is None
    assert archived is None


async def test_get_published_interview_excludes_soft_deleted(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        deleted = await get_published_interview(session, fixture_data["cfg_soft_deleted"])
    assert deleted is None


async def test_list_published_questions_filters_to_approved(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        questions = await list_published_questions_for_config(session, fixture_data["cfg_pub"])
    ids = {q.id for q in questions}
    assert ids == {fixture_data["q_approved"]}


async def test_list_published_interviews_for_module_filters_status(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        rows = await list_published_interviews_for_module(session, fixture_data["module_a"])
    ids = {c.id for c in rows}
    assert fixture_data["cfg_pub"] in ids
    assert fixture_data["cfg_capped"] in ids
    assert fixture_data["cfg_draft"] not in ids
    assert fixture_data["cfg_archived"] not in ids
    assert fixture_data["cfg_soft_deleted"] not in ids
    assert fixture_data["cfg_module_b"] not in ids


async def test_list_published_interviews_for_course(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        rows = await list_published_interviews_for_course(session, fixture_data["course"])
    ids = {c.id for c in rows}
    assert fixture_data["cfg_pub"] in ids
    assert fixture_data["cfg_module_b"] in ids
    assert fixture_data["cfg_capped"] in ids
    assert fixture_data["cfg_draft"] not in ids


async def test_get_interview_for_taking_returns_triple(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        result = await get_interview_for_taking(
            session,
            fixture_data["cfg_pub"],
            fixture_data["fresh_student"],
        )
    assert result is not None
    config, outcomes, questions = result
    assert config.id == fixture_data["cfg_pub"]
    assert {o.id for o in outcomes} == {fixture_data["outcome_pub"]}
    assert {q.id for q in questions} == {fixture_data["q_approved"]}


async def test_get_interview_for_taking_max_attempts(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        capped_for_student = await get_interview_for_taking(
            session,
            fixture_data["cfg_capped"],
            fixture_data["student"],
        )
        capped_for_fresh = await get_interview_for_taking(
            session,
            fixture_data["cfg_capped"],
            fixture_data["fresh_student"],
        )
    assert capped_for_student is None
    assert capped_for_fresh is not None


async def test_authoring_returns_all_statuses(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        draft = await get_interview_for_authoring(session, fixture_data["cfg_draft"])
        archived = await get_interview_for_authoring(session, fixture_data["cfg_archived"])
        course_interviews = await list_interviews_for_course(
            session, fixture_data["course"], include_archived=True
        )
    assert draft is not None
    assert draft.status == "draft"
    assert archived is not None
    assert archived.status == "archived"
    statuses = {c.status for c in course_interviews}
    assert {"draft", "published", "archived"}.issubset(statuses)


async def test_authoring_excludes_archived_by_default(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        without_flag = await list_interviews_for_course(
            session, fixture_data["course"], include_archived=False
        )
        with_flag = await list_interviews_for_course(
            session, fixture_data["course"], include_archived=True
        )
    assert fixture_data["cfg_archived"] not in {c.id for c in without_flag}
    assert fixture_data["cfg_archived"] in {c.id for c in with_flag}


async def test_get_interview_with_full_config(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        result = await get_interview_with_full_config(session, fixture_data["cfg_pub"])
    assert result is not None
    config, outcomes, questions = result
    assert config.id == fixture_data["cfg_pub"]
    assert len(outcomes) == 1
    assert {q.id for q in questions} == {
        fixture_data["q_approved"],
        fixture_data["q_pending"],
        fixture_data["q_rejected"],
    }


async def test_list_questions_for_config_filters_review_status(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        all_qs = await list_questions_for_config(session, fixture_data["cfg_pub"])
        approved = await list_questions_for_config(
            session, fixture_data["cfg_pub"], review_status="approved"
        )
        pending = await list_questions_for_config(
            session, fixture_data["cfg_pub"], review_status="pending"
        )
    assert len(all_qs) == 3
    assert {q.id for q in approved} == {fixture_data["q_approved"]}
    assert {q.id for q in pending} == {fixture_data["q_pending"]}


async def test_position_helpers(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        next_outcome = await next_outcome_position(session, fixture_data["cfg_pub"])
        next_question = await next_question_position(session, fixture_data["cfg_pub"])
        next_outcome_empty = await next_outcome_position(session, fixture_data["cfg_draft"])
    assert next_outcome == 2
    assert next_question == 4
    assert next_outcome_empty == 1


async def test_get_active_session_one_per_config(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        active = await get_active_session(
            session,
            fixture_data["student"],
            fixture_data["cfg_pub"],
        )
        inactive_for_capped = await get_active_session(
            session,
            fixture_data["student"],
            fixture_data["cfg_capped"],
        )
    assert active is not None
    assert active.id == fixture_data["session_active"]
    assert active.status == "in_progress"
    assert inactive_for_capped is None


async def test_active_session_unique_constraint_blocks_duplicate(
    engine: AsyncEngine,
    fixture_data: dict,
) -> None:
    """DB-level UNIQUE on (interview_config_id, student_id, attempt_number)
    enforces no-duplicate-attempt invariant complementing the query."""
    from sqlalchemy.exc import IntegrityError

    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO interview_sessions ("
                    "id, interview_config_id, student_id, attempt_number, "
                    "status, input_mode) VALUES "
                    "(:id, :c, :s, 1, 'in_progress', 'text')"
                ),
                {
                    "id": uuid.uuid4(),
                    "c": fixture_data["cfg_pub"],
                    "s": fixture_data["student"],
                },
            )


async def test_get_session_attempt_number(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        next_for_student = await get_session_attempt_number(
            session,
            fixture_data["student"],
            fixture_data["cfg_pub"],
        )
        next_fresh = await get_session_attempt_number(
            session,
            fixture_data["fresh_student"],
            fixture_data["cfg_capped"],
        )
    assert next_for_student == 3
    assert next_fresh == 1


async def test_get_user_interview_sessions(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        all_sessions = await get_user_interview_sessions(session, fixture_data["student"])
        only_active = await get_user_interview_sessions(
            session, fixture_data["student"], status="in_progress"
        )
    assert len(all_sessions) == 4
    assert {s.id for s in only_active} == {fixture_data["session_active"]}


async def test_get_session_with_responses(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        result = await get_session_with_responses(session, fixture_data["session_completed"])
        missing = await get_session(session, uuid.uuid4())
    assert result is not None
    sess, asked, messages = result
    assert sess.id == fixture_data["session_completed"]
    assert {q.id for q in asked} == {fixture_data["session_question"]}
    assert {m.id for m in messages} == {
        fixture_data["msg_user"],
        fixture_data["msg_ai"],
    }
    assert missing is None


async def test_outcome_evaluations_and_gap_report(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        evals = await get_outcome_evaluations(session, fixture_data["session_completed"])
        gap = await get_gap_report_for_session(session, fixture_data["session_completed"])
        outcomes = await list_outcomes_for_config(session, fixture_data["cfg_pub"])
    assert {e.id for e in evals} == {fixture_data["eval_row"]}
    assert evals[0].verdict_met is True
    assert gap is not None
    assert gap.id == fixture_data["gap_row"]
    assert {o.id for o in outcomes} == {fixture_data["outcome_pub"]}


async def test_list_session_messages_filtered_by_question(
    session_factory: async_sessionmaker[AsyncSession],
    fixture_data: dict,
) -> None:
    async with session_factory() as session:
        all_msgs = await list_session_messages(session, fixture_data["session_completed"])
        per_question = await list_session_messages(
            session,
            fixture_data["session_completed"],
            session_question_id=fixture_data["session_question"],
        )
    assert {m.id for m in all_msgs} == {
        fixture_data["msg_user"],
        fixture_data["msg_ai"],
    }
    assert {m.id for m in per_question} == {
        fixture_data["msg_user"],
        fixture_data["msg_ai"],
    }


def test_no_mechanism_split() -> None:
    queries_dir = (
        Path(__file__).resolve().parents[2] / "abridgeai" / "features" / "interviews" / "queries"
    )
    subdirs = {p.name for p in queries_dir.iterdir() if p.is_dir() and not p.name.startswith("__")}
    assert "orm" not in subdirs
    assert "raw" not in subdirs
    assert subdirs.issubset({"sql"}), (
        f"Locked decision: queries/ must contain only sql/ as subdir, got {subdirs}"
    )
