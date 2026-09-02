"""Interview outcome authoring (T6.11).

Outcome CRUD for an interview config, extracted verbatim from
``services.authoring`` (2026-09-01) to keep that module under the
interviews LOC ratchet.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.db.recursive_delete import soft_delete_cascade
from abridgeai.core.exceptions import AppError
from abridgeai.core.security import CurrentUser
from abridgeai.features.interviews.models import InterviewOutcome
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.services import published_freeze
from abridgeai.features.interviews.services._shared import (
    _assert_threshold_within_outcome_count,
    _require_config,
    _require_outcome,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


async def add_outcome(
    db: AsyncSession,
    config_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewOutcome:
    config = await _require_config(db, config_id)
    # The outcomes are the grading criteria; a published interview must not
    # grow a criterion mid-cohort (see published_freeze.py).
    published_freeze.assert_learning_outcomes_editable(config)
    data = payload.model_dump(exclude_unset=True)
    outcome_text = str(data["outcome_text"] or "").strip()
    if not outcome_text:
        raise AppError("Outcome text is required")
    next_position = await authoring_queries.next_outcome_position(db, config_id)
    outcome = InterviewOutcome(
        interview_config_id=config_id,
        position=data.get("position") or next_position,
        outcome_text=outcome_text,
        outcome_type=data["outcome_type"],
        importance_weight=data.get("importance_weight", 1),
        created_by=actor.user_id,
    )
    db.add(outcome)
    await flush_or_conflict(db)
    await db.refresh(outcome)
    return outcome


async def update_outcome(
    db: AsyncSession,
    config_id: UUID,
    outcome_id: UUID,
    payload: Any,  # noqa: ANN401
    actor: CurrentUser,
) -> InterviewOutcome:
    config = await _require_config(db, config_id)
    # Reweighting an outcome changes how every answer scores; frozen on publish.
    published_freeze.assert_learning_outcomes_editable(config)
    outcome = await _require_outcome(db, config_id, outcome_id)
    data = payload.model_dump(exclude_unset=True)
    if "outcome_text" in data:
        outcome_text = str(data["outcome_text"] or "").strip()
        if not outcome_text:
            raise AppError("Outcome text is required")
        data["outcome_text"] = outcome_text
    for key in ("outcome_text", "outcome_type", "importance_weight", "position"):
        if key in data:
            setattr(outcome, key, data[key])
    if data:
        outcome.updated_by = actor.user_id
    await flush_or_conflict(db)
    await db.refresh(outcome)
    return outcome


async def delete_outcome(
    db: AsyncSession,
    config_id: UUID,
    outcome_id: UUID,
    actor: CurrentUser,
) -> None:
    config = await _require_config(db, config_id)
    # Dropping a criterion retroactively rewrites what the interview measured.
    published_freeze.assert_learning_outcomes_editable(config)
    outcome = await _require_outcome(db, config_id, outcome_id)
    outcomes = await authoring_queries.list_outcomes_for_config(db, config_id)
    _assert_threshold_within_outcome_count(
        config.min_outcomes_to_pass,
        max(0, len(outcomes) - 1),
    )
    await soft_delete_cascade(db, outcome, actor_id=actor.user_id)
