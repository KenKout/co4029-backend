from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from abridgeai.features.interviews.api.public import (
    SessionSummaryDTO,
    get_session_summary,
)


def test_module_exports() -> None:
    from abridgeai.features.interviews.api import public

    assert {"get_session_summary"} <= set(public.__all__)


def test_dto_is_frozen() -> None:
    dto = SessionSummaryDTO(
        id=uuid4(),
        interview_config_id=uuid4(),
        student_id=uuid4(),
        attempt_number=1,
        status="in_progress",
        input_mode="text",
        started_at=datetime.now(UTC),
        ended_at=None,
        pass_verdict=None,
        outcomes_total=0,
        outcomes_met=0,
    )
    with pytest.raises(ValidationError):
        dto.status = "completed"  # type: ignore[misc]


@pytest.mark.asyncio
async def test_get_session_summary_missing_returns_none(
    test_engine: AsyncEngine,
) -> None:
    async with AsyncSession(test_engine) as session:
        result = await get_session_summary(session, uuid4())
    assert result is None


@pytest.mark.asyncio
async def test_get_session_summary_rolls_up_outcomes(
    test_engine: AsyncEngine,
) -> None:
    org_id = uuid4()
    user_id = uuid4()
    course_id = uuid4()
    module_id = uuid4()
    config_id = uuid4()
    session_id = uuid4()
    outcome_a = uuid4()
    outcome_b = uuid4()
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
                "INSERT INTO interview_configs (id, course_id, module_id, title, status, slug) VALUES (:id, :c, :m, 'IC', 'draft', 'slug-' || uuid_generate_v4()::text);"
            ),
                {
                    "id": str(config_id),
                    "c": str(course_id),
                    "m": str(module_id),
                },
            )
            await session.execute(
                text(
                    "INSERT INTO interview_outcomes (id, interview_config_id, "
                    "position, outcome_text, outcome_type, importance_weight) "
                    "VALUES (:a, :cfg, 1, 'A', 'knowledge', 1), "
                    "(:b, :cfg, 2, 'B', 'knowledge', 1)"
                ),
                {
                    "a": str(outcome_a),
                    "b": str(outcome_b),
                    "cfg": str(config_id),
                },
            )
            await session.execute(
                text(
                    "INSERT INTO interview_sessions (id, interview_config_id, "
                    "student_id, attempt_number, status, input_mode, "
                    "started_at) "
                    "VALUES (:id, :cfg, :s, 1, 'completed', 'text', :at)"
                ),
                {
                    "id": str(session_id),
                    "cfg": str(config_id),
                    "s": str(user_id),
                    "at": datetime.now(UTC),
                },
            )
            await session.execute(
                text(
                    "INSERT INTO interview_outcome_evaluations "
                    "(id, session_id, outcome_id, verdict_met) "
                    "VALUES (:e1, :sess, :oa, TRUE), "
                    "(:e2, :sess, :ob, FALSE)"
                ),
                {
                    "e1": str(uuid4()),
                    "e2": str(uuid4()),
                    "sess": str(session_id),
                    "oa": str(outcome_a),
                    "ob": str(outcome_b),
                },
            )
            await session.flush()

            summary = await get_session_summary(session, session_id)
        finally:
            await trans.rollback()

    assert summary is not None
    assert isinstance(summary, SessionSummaryDTO)
    assert summary.id == session_id
    assert summary.attempt_number == 1
    assert summary.status == "completed"
    assert summary.input_mode == "text"
    assert summary.outcomes_total == 2
    assert summary.outcomes_met == 1
