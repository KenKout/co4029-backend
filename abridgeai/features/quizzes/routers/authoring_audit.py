"""Quiz audit-event authoring route."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.pagination import PageResponse, paginate_sequence
from abridgeai.core.security import CurrentUser
from abridgeai.features.quizzes.routers.authoring import (
    _REQUIRE_QUIZ,
    _resolve_student_contacts,
    router,
)


class _AuditEventRow(BaseModel):
    model_config = {"from_attributes": True}

    id: UUID
    event_name: str
    quiz_id: UUID
    actor_user_id: UUID | None = None
    actor_name: str | None = None
    actor_email: str | None = None
    subject_attempt_id: UUID | None = None
    subject_question_id: UUID | None = None
    subject_user_id: UUID | None = None
    payload_json: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime


@router.get(
    "/quizzes/{quiz_id}/audit-events",
    response_model=PageResponse[_AuditEventRow],
)
async def list_quiz_audit_events(
    quiz_id: UUID,
    current_user: Annotated[CurrentUser, Depends(_REQUIRE_QUIZ)],
    db: Annotated[AsyncSession, Depends(get_db)],
    search: Annotated[str | None, Query(max_length=200)] = None,
    event_name: str | None = None,
    sort: str | None = None,
    sort_dir: Annotated[str, Query(pattern="^(asc|desc)$")] = "desc",
    page: Annotated[int, Query(ge=0)] = 0,
    page_size: Annotated[int, Query(ge=1, le=200)] = 25,
) -> PageResponse[_AuditEventRow]:
    """Most-recent-first append-only audit trail for a quiz."""
    del current_user
    from abridgeai.features.quizzes.services import audit as _audit  # noqa: PLC0415

    events = await _audit.list_events_for_quiz(db, quiz_id, limit=None)
    actor_contacts = await _resolve_student_contacts(
        db, {event.actor_user_id for event in events if event.actor_user_id is not None}
    )
    rows = []
    for event in events:
        row = _AuditEventRow.model_validate(event)
        if event.actor_user_id is not None:
            actor_name, actor_email, _ = actor_contacts.get(event.actor_user_id, (None, None, None))
            row = row.model_copy(update={"actor_name": actor_name, "actor_email": actor_email})
        rows.append(row)
    result = paginate_sequence(
        rows,
        page=page,
        page_size=page_size,
        search=search,
        search_text=lambda row: (
            f"{row.event_name} {row.actor_name or ''} {row.actor_email or ''} {row.payload_json}"
        ),
        predicate=lambda row: event_name is None or row.event_name == event_name,
        sort=sort,
        sort_dir=sort_dir,
        sortable={"event": lambda row: row.event_name, "when": lambda row: row.occurred_at},
    )
    return PageResponse[_AuditEventRow](**result.__dict__)


__all__ = ["_AuditEventRow", "list_quiz_audit_events"]
