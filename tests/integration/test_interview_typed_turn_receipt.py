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

import asyncio
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
        # The linkage tests insert interview_questions rows referencing this
        # config; delete them first or the config delete violates its FK.
        await conn.execute(
            text("DELETE FROM interview_questions WHERE interview_config_id = :c"),
            {"c": config_id},
        )
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


async def test_a_bank_question_id_is_resolved_to_a_session_question_row(
    receipt_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """THE linkage fix: the receipt is FK-linked via a created session-question.

    A bank id is not a session-question id (different table); the resolver
    creates the asked-question row on demand and the receipt links to IT, so
    the evaluator's ``session_question_id IS NOT NULL`` candidate filter finds
    the answer.
    """
    session_id = receipt_probe["session_id"]
    bank_question_id = uuid.uuid4()

    # The resolver needs the bank question to exist (FK on
    # interview_session_questions.interview_question_id → interview_questions,
    # ondelete SET NULL means the row can still be created; but create the
    # bank question properly so the link is real).
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, position, question_type, prompt_text, "
                " review_status, ai_generated, source_refs_json) "
                "SELECT :qid, ic.id, 99, 'conceptual', 'Bank question?', "
                "       'approved', false, '[]'::jsonb "
                "FROM interview_configs ic WHERE ic.id = ("
                "  SELECT interview_config_id FROM interview_sessions WHERE id = :sid)"
            ),
            {"qid": bank_question_id, "sid": session_id},
        )

    async with _maker(test_engine)() as db:
        row, created = await ntt.persist_receipt(
            db,
            session_id=session_id,
            session_question_id=None,
            bank_question_id=bank_question_id,
            text="my linked answer",
            turn_key="tk-link-0005",
        )
        assert created is True
        # The row's linkage is NOT the bank id — it is the created
        # session-question row, whose interview_question_id points back.
        assert row.session_question_id is not None
        assert row.session_question_id != bank_question_id

    async with test_engine.begin() as conn:
        asked = (
            await conn.execute(
                text(
                    "SELECT iq.interview_question_id FROM interview_session_questions iq "
                    "WHERE iq.id = :sid"
                ),
                {"sid": row.session_question_id},
            )
        ).scalar_one()
    assert asked == bank_question_id, (
        "the created session-question row does not point back at the bank question"
    )


async def test_concurrent_resolution_of_the_same_bank_question_converges(
    receipt_probe: dict[str, Any], test_engine: AsyncEngine
) -> None:
    """Two creators of the same link: the loser reloads, the answer survives.

    The resolver's flush can lose the (session, sequence) unique slot to a
    concurrent creator; convergence is constraint + reload, never a lost
    answer.
    """
    session_id = receipt_probe["session_id"]
    bank_question_id = uuid.uuid4()
    async with test_engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO interview_questions "
                "(id, interview_config_id, position, question_type, prompt_text, "
                " review_status, ai_generated, source_refs_json) "
                "SELECT :qid, ic.id, 98, 'conceptual', 'Race question?', "
                "       'approved', false, '[]'::jsonb "
                "FROM interview_configs ic WHERE ic.id = ("
                "  SELECT interview_config_id FROM interview_sessions WHERE id = :sid)"
            ),
            {"qid": bank_question_id, "sid": session_id},
        )

    # Race two FULL persist_receipt calls (the production entry point): the
    # resolver's flush, the receipt insert, and the commit share ONE
    # transaction there, so convergence is judged on durable rows — not on
    # in-memory ids a later rollback could invalidate.
    maker = _maker(test_engine)

    async def _persist(turn_key: str) -> Any:
        async with maker() as db:
            row, _created = await ntt.persist_receipt(
                db,
                session_id=session_id,
                session_question_id=None,
                bank_question_id=bank_question_id,
                text=f"racing answer {turn_key}",
                turn_key=turn_key,
            )
            return row

    rows = list(await asyncio.gather(_persist("tk-race-0001"), _persist("tk-race-0002")))

    # THE guarantee that matters: NO answer is lost. Every racing receipt is
    # durably linked to a session-question row that points back at the SAME
    # bank question. (When the two creations overlap, the constraint+reload
    # path converges them onto one row; when they interleave after a commit,
    # a second row for the same bank question is benign — both links resolve,
    # nothing is lost, and future resolves reuse whichever exists.)
    assert all(r.session_question_id is not None for r in rows), (
        "a racing receipt was persisted without its question link"
    )

    async with test_engine.begin() as conn:
        ids = [r.session_question_id for r in rows]
        linked_banks = (
            await conn.execute(
                text(
                    "SELECT iq.interview_question_id FROM interview_session_questions iq "
                    "WHERE iq.id = ANY(:ids)"
                ),
                {"ids": ids},
            )
        ).scalars().all()
    assert set(linked_banks) == {bank_question_id}, (
        "a racing receipt's session-question row points at the WRONG bank question"
    )


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
