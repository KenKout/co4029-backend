"""Quiz override DTOs (Phase 5).

Back the teacher override-management endpoints:

* ``GET    /teacher/quizzes/{quiz_id}/overrides``          -> list[QuizOverrideRead]
* ``POST   /teacher/quizzes/{quiz_id}/overrides``          (QuizOverrideIn) -> QuizOverrideRead
* ``PATCH  /teacher/quizzes/{quiz_id}/overrides/{id}``     (QuizOverrideIn) -> QuizOverrideRead
* ``DELETE /teacher/quizzes/{quiz_id}/overrides/{id}``     -> 204

A user override sets ``user_id`` (group_id null); a group override sets
``group_id`` (user_id null). NULL exception columns mean "no exception — fall
through to the base quiz value" (see the precedence resolver).
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, model_validator


class QuizOverrideIn(BaseModel):
    """Create/update payload for one override row."""

    model_config = ConfigDict(extra="forbid")

    scope: Literal["user", "group"]
    user_id: UUID | None = None
    group_id: UUID | None = None
    available_from: datetime | None = None
    available_until: datetime | None = None
    due_at: datetime | None = None
    time_limit_seconds: int | None = None
    max_attempts: int | None = None
    allow_retakes: bool | None = None
    cooldown_hours: int | None = None

    @model_validator(mode="after")
    def _check_scope_target(self) -> QuizOverrideIn:
        if self.scope == "user" and (self.user_id is None or self.group_id is not None):
            raise ValueError("user-scope override requires user_id and no group_id")
        if self.scope == "group" and (self.group_id is None or self.user_id is not None):
            raise ValueError("group-scope override requires group_id and no user_id")
        return self


class QuizOverrideRead(BaseModel):
    """An override row as returned to the teacher."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    quiz_id: UUID
    scope: str
    user_id: UUID | None = None
    group_id: UUID | None = None
    available_from: datetime | None = None
    available_until: datetime | None = None
    due_at: datetime | None = None
    time_limit_seconds: int | None = None
    max_attempts: int | None = None
    allow_retakes: bool | None = None
    cooldown_hours: int | None = None


__all__ = ["QuizOverrideIn", "QuizOverrideRead"]
