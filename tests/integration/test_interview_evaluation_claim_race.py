"""The evaluation claim must hold under two genuinely concurrent transactions.

The unit suite pins the ownership RULES in Python. This file pins the SQL, which
is where the guarantee actually lives: the claim is one conditional UPDATE, and
its correctness depends on Postgres re-evaluating the WHERE clause against the
committed row after a competing row lock is released.

The dangerous interleaving is the same shape as the runtime-state race:

1. T1 issues its claim UPDATE and holds the row lock uncommitted.
2. T2's claim runs inside that window. Under READ COMMITTED its snapshot still
   shows the row UNCLAIMED, so a check done in a preceding SELECT would pass.
3. T2's UPDATE blocks on T1's lock, then re-checks after T1 commits — and matches
   zero rows.

With the predicate inside the UPDATE, exactly one job wins. A test that claims
sequentially would pass even with the check in a separate SELECT, so it would
prove nothing about the race we care about.

These tests drive real concurrent transactions and therefore build their own
sessions from the engine rather than sharing the per-test session fixture.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from abridgeai.features.interviews.queries import sessions as sessions_queries
from abridgeai.features.interviews.services.evaluation_claim import (
    lease_expiry_for,
    new_claim_token,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def claim_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    """A minimal FK chain plus one terminal, ungraded, unclaimed session."""
    org_id = uuid.uuid4()
    teacher_id = uuid.uuid4()
    student_id = uuid.uuid4()
    course_id = uuid.uuid4()
    module_id = uuid.uuid4()
    config_id = uuid.uuid4()
    session_id = uuid.uuid4()
    suffix = org_id.hex[:8]

    async with test_engine.begin() as conn:
        await conn.execute(
            text("INSERT INTO organizations (id, slug, name) VALUES (:id, :slug, :name)"),
            {"id": org_id, "slug": f"ec-{suffix}", "name": "Eval Claim Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"ec-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'EC Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"ec-course-{suffix}"},
        )
        await conn.execute(
            text(
                "INSERT INTO modules (id, course_id, title, position, status) "
                "VALUES (:id, :course, 'Module', 1, 'published')"
            ),
            {"id": module_id, "course": course_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_configs "
                "(id, course_id, module_id, title, status, persona, created_by, slug) "
                "VALUES (:id, :course, :module, 'EC Interview', 'published', 'neutral', "
                ":teacher, 'slug-' || uuid_generate_v4()::text)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, assessment_started_at, ended_at, onboarding_stage, "
                "interview_language) "
                "VALUES (:id, :cfg, :student, 1, 'completed', 'text', now(), now(), now(), "
                "'completed', 'en')"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )

    yield {"session_id": session_id, "student_id": student_id, "course_id": course_id}

    async with test_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM gap_reports WHERE source_interview_session_id = :s"),
            {"s": session_id},
        )
        await conn.execute(text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id})
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"), {"t": teacher_id, "s": student_id}
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


async def _claim_columns(test_engine: AsyncEngine, session_id: UUID) -> Any:
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    async with maker() as db:
        return (
            await db.execute(
                text(
                    "SELECT evaluation_claim_token AS token, "
                    "       evaluation_claim_expires_at AS expires_at "
                    "  FROM interview_sessions WHERE id = :s"
                ),
                {"s": session_id},
            )
        ).first()


async def test_only_one_of_two_concurrent_claims_wins(
    claim_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Both jobs start while the verdict is NULL; exactly one may grade."""
    session_id = claim_probe["session_id"]
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    now = datetime.now(UTC)
    first, second = new_claim_token(), new_claim_token()

    async def attempt(token: uuid.UUID) -> bool:
        async with maker() as db:
            return await sessions_queries.claim_session_evaluation(
                db, session_id, token=token, now=now, lease_expires_at=lease_expiry_for(now)
            )

    results = await asyncio.gather(attempt(first), attempt(second))

    assert sum(1 for won in results if won) == 1, (
        f"exactly one claim must win, got {results} — two graders would race to publish"
    )
    row = await _claim_columns(test_engine, session_id)
    winner = first if results[0] else second
    assert row.token == winner
    assert row.expires_at is not None


async def test_the_loser_is_refused_from_inside_the_winners_lock_window(
    claim_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The interleaving a preceding-SELECT guard would let through.

    T2's claim is issued while T1's UPDATE is uncommitted, so T2's snapshot shows
    an unclaimed row. It must still lose once T1 commits.
    """
    session_id = claim_probe["session_id"]
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    now = datetime.now(UTC)
    winner_token, loser_token = new_claim_token(), new_claim_token()

    t1 = maker()
    t2 = maker()
    try:
        # T1 claims but does NOT commit yet — it holds the row lock.
        held = await t1.execute(
            text(
                "UPDATE interview_sessions "
                "   SET evaluation_claim_token = :tok, evaluation_claim_expires_at = :exp "
                " WHERE id = :s AND pass_verdict IS NULL "
                "   AND (evaluation_claim_token IS NULL "
                "        OR evaluation_claim_expires_at IS NULL "
                "        OR evaluation_claim_expires_at <= :now) "
                "RETURNING id"
            ),
            {
                "tok": winner_token,
                "exp": lease_expiry_for(now),
                "s": session_id,
                "now": now,
            },
        )
        assert held.scalar_one_or_none() is not None

        async def competing_claim() -> bool:
            return await sessions_queries.claim_session_evaluation(
                t2,
                session_id,
                token=loser_token,
                now=now,
                lease_expires_at=lease_expiry_for(now),
            )

        task = asyncio.create_task(competing_claim())
        await asyncio.sleep(0.2)  # let T2 reach the lock and block on it
        assert not task.done(), "T2 should be blocked on T1's row lock"

        await t1.commit()
        assert await task is False, (
            "the second claimer saw an unclaimed snapshot and must still be refused"
        )
    finally:
        await t1.close()
        await t2.close()

    row = await _claim_columns(test_engine, session_id)
    assert row.token == winner_token


async def test_a_graded_session_cannot_be_claimed(
    claim_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """``pass_verdict=False`` is a published judgement, not an absence of one."""
    session_id = claim_probe["session_id"]
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    async with test_engine.begin() as conn:
        await conn.execute(
            text("UPDATE interview_sessions SET pass_verdict = FALSE WHERE id = :s"),
            {"s": session_id},
        )

    now = datetime.now(UTC)
    async with maker() as db:
        claimed = await sessions_queries.claim_session_evaluation(
            db, session_id, token=new_claim_token(), now=now, lease_expires_at=lease_expiry_for(now)
        )

    assert claimed is False


async def test_an_expired_lease_is_reclaimed_and_the_dead_owner_loses_ownership(
    claim_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """A worker killed at job_timeout must not hold the session hostage."""
    session_id = claim_probe["session_id"]
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    now = datetime.now(UTC)
    dead = new_claim_token()
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE interview_sessions "
                "   SET evaluation_claim_token = :tok, evaluation_claim_expires_at = :exp "
                " WHERE id = :s"
            ),
            {"tok": dead, "exp": now - timedelta(minutes=1), "s": session_id},
        )

    fresh = new_claim_token()
    async with maker() as db:
        assert (
            await sessions_queries.claim_session_evaluation(
                db, session_id, token=fresh, now=now, lease_expires_at=lease_expiry_for(now)
            )
            is True
        )
        assert (
            await sessions_queries.holds_session_evaluation_claim(
                db, session_id, token=fresh, now=now
            )
            is True
        )
        assert (
            await sessions_queries.holds_session_evaluation_claim(
                db, session_id, token=dead, now=now
            )
            is False
        ), "the dead owner must not be able to publish or stamp a failure"
        await db.rollback()


async def test_releasing_is_scoped_to_our_own_token(
    claim_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """A superseded job must not clear the claim of whoever took over."""
    session_id = claim_probe["session_id"]
    maker = async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)
    now = datetime.now(UTC)
    owner, stale = new_claim_token(), new_claim_token()

    async with maker() as db:
        await sessions_queries.claim_session_evaluation(
            db, session_id, token=owner, now=now, lease_expires_at=lease_expiry_for(now)
        )
        await sessions_queries.release_session_evaluation_claim(db, session_id, token=stale)

    row = await _claim_columns(test_engine, session_id)
    assert row.token == owner, "a stale job released a claim it did not own"

    async with maker() as db:
        await sessions_queries.release_session_evaluation_claim(db, session_id, token=owner)

    row = await _claim_columns(test_engine, session_id)
    assert row.token is None
    assert row.expires_at is None


async def test_a_second_gap_report_for_one_session_is_refused(
    claim_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The partial unique index from 0107 is what stops the read-then-insert race.

    Both jobs saw "no report" and both inserted; readers take the newest row, so
    the stale duplicate stayed invisible while the data was wrong.
    """
    from sqlalchemy.exc import IntegrityError

    session_id = claim_probe["session_id"]
    values = {
        "student": claim_probe["student_id"],
        "course": claim_probe["course_id"],
        "s": session_id,
    }
    insert = text(
        "INSERT INTO gap_reports (id, student_id, course_id, source_interview_session_id, "
        "report_json) VALUES (uuid_generate_v4(), :student, :course, :s, '{}'::jsonb)"
    )
    async with test_engine.begin() as conn:
        await conn.execute(insert, values)

    with pytest.raises(IntegrityError, match="uq_gap_reports_source_interview_session"):
        async with test_engine.begin() as conn:
            await conn.execute(insert, values)


async def test_quiz_sourced_reports_are_unaffected_by_the_unique_index(
    claim_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """The index is partial: many NULL-session (quiz) reports remain legal."""
    values = {
        "student": claim_probe["student_id"],
        "course": claim_probe["course_id"],
    }
    insert = text(
        "INSERT INTO gap_reports (id, student_id, course_id, source_interview_session_id, "
        "report_json) VALUES (uuid_generate_v4(), :student, :course, NULL, '{}'::jsonb) "
        "RETURNING id"
    )
    created: list[uuid.UUID] = []
    async with test_engine.begin() as conn:
        for _ in range(2):
            created.append((await conn.execute(insert, values)).scalar_one())

    assert len(created) == 2

    async with test_engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM gap_reports WHERE id = ANY(:ids)"), {"ids": created}
        )
