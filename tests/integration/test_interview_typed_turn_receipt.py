"""Durable typed-turn receipts against real Postgres.

The receipt is the seam that turns the ACK's meaning from "in RAM" into "your
answer is durable". Everything that matters is SQL: the unique idempotency
index decides the race, and the received→applied transition is a CAS under the
processing token, so only the fold that claimed the receipt can settle it.

Pinned here:

* persisting a receipt commits a user row with the right metadata;
* a second persist of the SAME turn key loads the first row (never a second
  transcript entry) — the migration-0023 unique index doing the work;
* the received→applied CAS succeeds exactly once under the right token and
  refuses a wrong token or an already-applied row.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker

from abridgeai.features.interviews.realtime import native_typed_turn as ntt

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncEngine

pytestmark = pytest.mark.asyncio


@pytest_asyncio.fixture
async def receipt_probe(test_engine: AsyncEngine) -> AsyncIterator[dict[str, Any]]:
    """Minimal FK chain plus one live, onboarding-complete session."""
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
            {"id": org_id, "slug": f"tt-{suffix}", "name": "Typed Turn Org"},
        )
        for uid, label in ((teacher_id, "teacher"), (student_id, "student")):
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"tt-{label}-{suffix}@test.local"},
            )
        await conn.execute(
            text(
                "INSERT INTO courses (id, organization_id, owner_user_id, slug, title, status) "
                "VALUES (:id, :org, :owner, :slug, 'TT Course', 'published')"
            ),
            {"id": course_id, "org": org_id, "owner": teacher_id, "slug": f"tt-course-{suffix}"},
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
                "VALUES (:id, :course, :module, 'TT Interview', 'published', 'neutral', "
                ":teacher, 'slug-' || uuid_generate_v4()::text)"
            ),
            {"id": config_id, "course": course_id, "module": module_id, "teacher": teacher_id},
        )
        await conn.execute(
            text(
                "INSERT INTO interview_sessions "
                "(id, interview_config_id, student_id, attempt_number, status, input_mode, "
                "started_at, assessment_started_at, onboarding_stage, interview_language) "
                "VALUES (:id, :cfg, :student, 1, 'in_progress', 'hybrid', now(), now(), "
                "'completed', 'en')"
            ),
            {"id": session_id, "cfg": config_id, "student": student_id},
        )

    yield {"session_id": session_id}

    async with test_engine.begin() as conn:
        await conn.execute(text("DELETE FROM interview_sessions WHERE id = :s"), {"s": session_id})
        await conn.execute(text("DELETE FROM interview_configs WHERE id = :c"), {"c": config_id})
        await conn.execute(text("DELETE FROM modules WHERE id = :m"), {"m": module_id})
        await conn.execute(text("DELETE FROM courses WHERE id = :c"), {"c": course_id})
        await conn.execute(
            text("DELETE FROM users WHERE id IN (:t, :s)"), {"t": teacher_id, "s": student_id}
        )
        await conn.execute(text("DELETE FROM organizations WHERE id = :o"), {"o": org_id})


def _maker(test_engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(test_engine, expire_on_commit=False, autoflush=False)


async def _count_rows(test_engine: AsyncEngine, session_id: UUID) -> int:
    async with test_engine.begin() as conn:
        return (
            await conn.execute(
                text(
                    "SELECT count(*) FROM interview_session_messages "
                    "WHERE session_id = :s AND role = 'user'"
                ),
                {"s": session_id},
            )
        ).scalar_one()


async def test_persisting_a_receipt_commits_a_received_user_row(
    receipt_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    session_id = receipt_probe["session_id"]
    turn_key = "tk-receipt-0001"

    async with _maker(test_engine)() as db:
        row, created = await ntt.persist_receipt(
            db,
            session_id=session_id,
            session_question_id=None,
            bank_question_id=None,
            text="my answer",
            turn_key=turn_key,
        )
        assert created is True
        assert row.role == "user"
        assert row.content_text == "my answer"
        meta = row.metadata_json
        assert meta["turn_key"] == turn_key
        assert meta["turn_state"] == "received"
        assert meta["source"] == "native_agent"
        assert meta["kind"] == "answer"
        assert meta["processing_token"]

    assert await _count_rows(test_engine, session_id) == 1


async def test_a_second_persist_of_the_same_key_loads_the_first_row(
    receipt_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """THE unique index: two copies of one turn make ONE transcript row."""
    session_id = receipt_probe["session_id"]
    turn_key = "tk-receipt-0002"

    async with _maker(test_engine)() as db:
        first, created_first = await ntt.persist_receipt(
            db,
            session_id=session_id,
            session_question_id=None,
            bank_question_id=None,
            text="my answer",
            turn_key=turn_key,
        )
        assert created_first is True

    async with _maker(test_engine)() as db:
        second, created_second = await ntt.persist_receipt(
            db,
            session_id=session_id,
            session_question_id=None,
            bank_question_id=None,
            text="my answer",
            turn_key=turn_key,
        )
        assert created_second is False
        assert second.id == first.id

    assert await _count_rows(test_engine, session_id) == 1


async def test_the_received_to_applied_cas_is_one_shot(
    receipt_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    session_id = receipt_probe["session_id"]

    async with _maker(test_engine)() as db:
        row, _ = await ntt.persist_receipt(
            db,
            session_id=session_id,
            session_question_id=None,
            bank_question_id=None,
            text="answer",
            turn_key="tk-receipt-0003",
        )
        token = uuid.UUID(row.metadata_json["processing_token"])

        async def _apply(row_: Any) -> bool:
            from sqlalchemy import update  # noqa: PLC0415

            from abridgeai.features.interviews.models import (  # noqa: PLC0415
                InterviewSessionMessage,
            )

            result = await db.execute(
                update(InterviewSessionMessage)
                .where(
                    InterviewSessionMessage.id == row_.id,
                    InterviewSessionMessage.metadata_json["turn_state"].as_string()
                    == "received",
                    InterviewSessionMessage.metadata_json["processing_token"].as_string()
                    == str(token),
                )
                .values(
                    metadata_json={
                        **row_.metadata_json,
                        "turn_state": "applied",
                        "applied_at": datetime.now(UTC).isoformat(),
                    }
                )
            )
            await db.commit()
            return result.rowcount > 0

        assert await _apply(row) is True
        # A second application (a duplicate fold) must not land: the row is
        # already applied, and the CAS refuses.
        assert await _apply(row) is False

        # A WRONG token (a superseded folder) cannot settle anything.
        row2, _ = await ntt.persist_receipt(
            db,
            session_id=session_id,
            session_question_id=None,
            bank_question_id=None,
            text="answer two",
            turn_key="tk-receipt-0004",
        )
        wrong = uuid.uuid4()
        result = await db.execute(
            text(
                "UPDATE interview_session_messages SET metadata_json = "
                "jsonb_set(metadata_json, '{turn_state}', to_jsonb('applied'::text)) "
                "WHERE id = :id AND metadata_json->>'processing_token' = :tok"
            ),
            {"id": row2.id, "tok": str(wrong)},
        )
        assert result.rowcount == 0


async def test_receipt_state_readers(
    receipt_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    session_id = receipt_probe["session_id"]

    async with _maker(test_engine)() as db:
        row, _ = await ntt.persist_receipt(
            db,
            session_id=session_id,
            session_question_id=None,
            bank_question_id=None,
            text="answer",
            turn_key="tk-receipt-0005",
        )
        assert ntt.receipt_state(row) == "received"
        assert ntt.receipt_is_owned_by_current_caller(row, None) is True

        # An expired lease reads as NOT owned: the folder is dead.
        stale = dict(row.metadata_json)
        stale["processing_expires_at"] = (
            datetime.now(UTC) - timedelta(minutes=1)
        ).isoformat()
        stale_row = SimpleNamespace(metadata_json=stale)
        assert ntt.receipt_is_owned_by_current_caller(stale_row, None) is False
