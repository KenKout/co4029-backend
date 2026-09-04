"""Persistence and protected-context services for interview security.

No function in this module returns protected content to a router. Protected
phrases live only long enough to evaluate a proposed student-facing output.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from abridgeai.core.config import get_settings
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewOutcome,
    InterviewQuestion,
    InterviewSecurityEvent,
)
from abridgeai.features.interviews.orchestrator.security import (
    OUTPUT_GUARD_VERSION,
    SECURITY_POLICY_VERSION,
    SECURITY_PROMPT_VERSION,
    SECURITY_RULES_VERSION,
    ProtectedContent,
    SecurityAction,
    SecurityAssessment,
    confidence_band,
)
from abridgeai.features.interviews.orchestrator.security_logic import assess_output_leakage
from abridgeai.features.interviews.realtime import observability as obs

if TYPE_CHECKING:
    from collections.abc import Iterable

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(frozen=True)
class GuardedOutput:
    text: str
    output_leakage_blocked: bool
    output_fallback_used: bool
    protected_content_category: str | None


@dataclass(frozen=True)
class SecuritySessionMetrics:
    assessment_count: int = 0
    blocked_attempt_count: int = 0
    repeated_attempt_count: int = 0
    output_leakage_prevented: int = 0
    security_fallback_rate: float = 0.0
    average_classification_latency_ms: float | None = None


async def protected_content_for_config(
    db: AsyncSession,
    *,
    config_id: UUID,
    allowed_question_ids: Iterable[UUID] = (),
) -> list[ProtectedContent]:
    """Load hidden content just in time for deterministic output comparison."""
    allowed = set(allowed_question_ids)
    questions = list(
        (
            await db.execute(
                select(InterviewQuestion).where(
                    InterviewQuestion.interview_config_id == config_id,
                    InterviewQuestion.deleted_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    outcomes = list(
        (
            await db.execute(
                select(InterviewOutcome).where(
                    InterviewOutcome.interview_config_id == config_id,
                    InterviewOutcome.deleted_at.is_(None),
                )
            )
        )
        .scalars()
        .all()
    )
    config = await db.get(InterviewConfig, config_id)

    protected: list[ProtectedContent] = []
    for question in questions:
        if question.id in allowed:
            protected.append(
                ProtectedContent(
                    category="allowed_question_text",
                    text=question.prompt_text,
                    content_id=str(question.id),
                )
            )
        else:
            protected.append(
                ProtectedContent(
                    category="question_text",
                    text=question.prompt_text,
                    content_id=str(question.id),
                )
            )
        if question.model_answer:
            protected.append(
                ProtectedContent(
                    category="model_answer",
                    text=question.model_answer,
                    content_id=str(question.id),
                )
            )
    protected.extend(
        ProtectedContent(
            category="rubric_text",
            text=outcome.outcome_text,
            content_id=str(outcome.id),
        )
        for outcome in outcomes
    )
    protected.extend(
        ProtectedContent(
            category="scoring_weight",
            text=f"{outcome.outcome_text} importance weight {outcome.importance_weight}",
            content_id=str(outcome.id),
        )
        for outcome in outcomes
    )
    if config is not None and config.supplementary_instructions:
        protected.append(
            ProtectedContent(
                category="supplementary_instructions",
                text=config.supplementary_instructions,
                content_id=str(config.id),
            )
        )
        protected.extend(
            _protected_authoring_values(
                config.supplementary_instructions,
                content_id=str(config.id),
            )
        )
    if config is not None and config.min_outcomes_to_pass is not None:
        protected.append(
            ProtectedContent(
                category="passing_threshold",
                text=f"minimum outcomes to pass {config.min_outcomes_to_pass}",
                content_id=str(config.id),
            )
        )
    protected.extend(_internal_prompt_phrases())
    return protected


def _protected_authoring_values(value: str, *, content_id: str) -> list[ProtectedContent]:
    """Extract exact rubric/threshold snippets from trusted authoring JSON."""
    try:
        payload = json.loads(value)
    except (TypeError, ValueError):
        return []
    if not isinstance(payload, dict):
        return []
    protected: list[ProtectedContent] = []
    raw_weights = payload.get("rubric_weights")
    if isinstance(raw_weights, dict):
        for criterion, weight in raw_weights.items():
            protected.append(
                ProtectedContent(
                    category="scoring_weight",
                    text=f"{criterion} rubric weight {weight}",
                    content_id=content_id,
                )
            )
    for key in ("passing_threshold", "min_outcomes_to_pass"):
        if key in payload:
            protected.append(
                ProtectedContent(
                    category="passing_threshold",
                    text=f"{key} {payload[key]}",
                    content_id=content_id,
                )
            )
    return protected


@lru_cache(maxsize=1)
def _internal_prompt_phrases() -> tuple[ProtectedContent, ...]:
    """Load reviewable prompt lines once; never expose them through a router."""
    feature_root = Path(__file__).resolve().parents[1]
    phrases: list[ProtectedContent] = []
    for path in sorted(feature_root.rglob("*system.j2")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            stripped = line.strip()
            if len(stripped) >= 24 and len(stripped.split()) >= 4:
                phrases.append(
                    ProtectedContent(category="internal_prompt", text=stripped)
                )
    return tuple(phrases)


async def record_security_event(
    db: AsyncSession,
    *,
    session_id: UUID,
    turn_key: str,
    event_type: str,
    assessment: SecurityAssessment,
    action: SecurityAction,
    attempt_count: int,
    latency_ms: float | None = None,
    fallback_status: bool = False,
    turn_id: str | None = None,
    protected_content_category: str | None = None,
) -> bool:
    """Persist and emit one redacted event; return False for a duplicate."""
    stable_turn_id = (turn_id or turn_key)[:255]
    existing = await db.scalar(
        select(InterviewSecurityEvent.id).where(
            InterviewSecurityEvent.session_id == session_id,
            InterviewSecurityEvent.turn_id == stable_turn_id,
            InterviewSecurityEvent.event_type == event_type,
        )
    )
    if existing is not None:
        return False

    row = InterviewSecurityEvent(
        session_id=session_id,
        turn_id=stable_turn_id,
        event_type=event_type,
        category=assessment.category.value,
        confidence_band=confidence_band(assessment.confidence),
        action=action.value,
        attempt_count=max(0, attempt_count),
        fallback_status=fallback_status,
        normalized_fingerprint=assessment.normalized_fingerprint,
        protected_content_category=protected_content_category,
        policy_version=SECURITY_POLICY_VERSION,
        rules_version=SECURITY_RULES_VERSION,
        prompt_version=SECURITY_PROMPT_VERSION,
        output_guard_version=OUTPUT_GUARD_VERSION,
        latency_ms=latency_ms,
    )
    try:
        async with db.begin_nested():
            db.add(row)
            await db.flush()
    except IntegrityError:
        return False

    obs.emit(
        event_type,
        session_id=session_id,
        turn_id=stable_turn_id,
        category=assessment.category.value,
        confidence_band=confidence_band(assessment.confidence),
        action=action.value,
        attempt_count=attempt_count,
        fallback_status=fallback_status,
        policy_version=SECURITY_POLICY_VERSION,
        latency_ms=latency_ms,
        normalized_fingerprint=assessment.normalized_fingerprint,
        protected_content_category=protected_content_category,
    )
    return True


async def guard_student_output(
    db: AsyncSession,
    *,
    session_id: UUID,
    config_id: UUID,
    turn_key: str,
    proposed_text: str,
    fallback_text: str,
    allowed_question_ids: Iterable[UUID],
    assessment: SecurityAssessment,
    action: SecurityAction,
    attempt_count: int,
    turn_id: str | None = None,
    force_record_only: bool = False,
) -> GuardedOutput:
    """Apply the output guard immediately before persistence/serialization.

    ``force_record_only`` downgrades even ``enforce`` mode to record-only for
    this single call: the leakage event is still persisted/emitted for audit,
    but ``proposed_text`` is returned unmodified and no fallback is
    substituted. Used by the gap-report boundary, where product decisions
    require the learner-facing AI feedback to always be displayed.
    """
    mode = get_settings().interview_security_guard_mode
    if mode == "off":
        return GuardedOutput(proposed_text, False, False, None)

    protected = await protected_content_for_config(
        db,
        config_id=config_id,
        allowed_question_ids=allowed_question_ids,
    )
    leakage = assess_output_leakage(proposed_text, protected)
    if not leakage.blocked:
        return GuardedOutput(proposed_text, False, False, None)

    enforcing = mode == "enforce" and not force_record_only
    await record_security_event(
        db,
        session_id=session_id,
        turn_key=turn_key,
        event_type=obs.EV_SECURITY_OUTPUT_LEAKAGE_BLOCKED,
        assessment=assessment,
        action=action if enforcing else SecurityAction.ALLOW,
        attempt_count=attempt_count,
        fallback_status=enforcing,
        turn_id=turn_id,
        protected_content_category=leakage.protected_content_category,
    )
    if not enforcing:
        return GuardedOutput(
            proposed_text,
            # force_record_only still reports the detection so callers can
            # distinguish "flagged but displayed" from a clean pass.
            output_leakage_blocked=force_record_only,
            output_fallback_used=False,
            protected_content_category=leakage.protected_content_category,
        )

    fallback_check = assess_output_leakage(fallback_text, protected)
    safe_text = fallback_text
    if fallback_check.blocked:
        safe_text = "I can repeat or clarify the current question."
    return GuardedOutput(
        safe_text,
        output_leakage_blocked=True,
        output_fallback_used=True,
        protected_content_category=leakage.protected_content_category,
    )


async def get_security_session_metrics(
    db: AsyncSession,
    session_id: UUID,
) -> SecuritySessionMetrics:
    assessed = func.count().filter(
        InterviewSecurityEvent.event_type == obs.EV_SECURITY_ASSESSED
    )
    blocked = func.count().filter(InterviewSecurityEvent.event_type == obs.EV_SECURITY_BLOCKED)
    repeated = func.count().filter(
        InterviewSecurityEvent.event_type == obs.EV_SECURITY_REPEATED_ATTEMPT
    )
    prevented = func.count().filter(
        (InterviewSecurityEvent.event_type == obs.EV_SECURITY_OUTPUT_LEAKAGE_BLOCKED)
        & InterviewSecurityEvent.fallback_status.is_(True)
    )
    # ``security_fallback_rate`` is the share of CLASSIFIER assessments that
    # fell back to the deterministic verdict, so both sides of the ratio must
    # come from the same event stream. Counting ``fallback_status`` across ALL
    # event types mixed in ``output_leakage_blocked`` rows (which set the flag
    # when the guard substitutes safe text) and produced rates above 1.0, or a
    # non-zero rate on a session whose classifier never failed once.
    fallbacks = func.count().filter(
        (InterviewSecurityEvent.event_type == obs.EV_SECURITY_ASSESSED)
        & InterviewSecurityEvent.fallback_status.is_(True)
    )
    row = (
        await db.execute(
            select(
                assessed,
                blocked,
                repeated,
                prevented,
                fallbacks,
                func.avg(InterviewSecurityEvent.latency_ms).filter(
                    InterviewSecurityEvent.event_type == obs.EV_SECURITY_ASSESSED
                ),
            ).where(InterviewSecurityEvent.session_id == session_id)
        )
    ).one()
    assessment_count = int(row[0] or 0)
    fallback_count = int(row[4] or 0)
    return SecuritySessionMetrics(
        assessment_count=assessment_count,
        blocked_attempt_count=int(row[1] or 0),
        repeated_attempt_count=int(row[2] or 0),
        output_leakage_prevented=int(row[3] or 0),
        security_fallback_rate=(fallback_count / assessment_count if assessment_count else 0.0),
        average_classification_latency_ms=(float(row[5]) if row[5] is not None else None),
    )


__all__ = [
    "GuardedOutput",
    "SecuritySessionMetrics",
    "get_security_session_metrics",
    "guard_student_output",
    "protected_content_for_config",
    "record_security_event",
]
