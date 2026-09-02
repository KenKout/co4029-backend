"""Interview taking service (T6.11).

Student-side session lifecycle: ``start_session``, ``take_session_step``
(record answer + optional follow-up + advance to next question), and
``submit_session`` (mark completed + enqueue async evaluation).

Security invariants:

* ``start_session`` enforces "1 active session per (config, student)"
  by short-circuiting on ``queries.sessions.get_active_session`` —
  duplicate POSTs return the existing live session.
* ``take_session_step`` and ``submit_session`` raise
  :class:`ForbiddenError` when ``session.student_id != actor.user_id``
  (router maps to HTTP 403). Cross-user session leak = security bug.

Submit dispatches the post-session evaluation via ARQ task
``evaluate_interview_session_task`` — never run synchronously in the
HTTP path (latency + retry budget concerns).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from sqlalchemy.exc import IntegrityError

from abridgeai.core.config import get_settings
from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.core.security import CurrentUser, utcnow
from abridgeai.features.interviews.ai.stages.followup import maybe_generate_followup
from abridgeai.features.interviews.models import (
    InterviewConfig,
    InterviewQuestion,
    InterviewSession,
    InterviewSessionMessage,
    InterviewSessionQuestion,
)
from abridgeai.features.interviews.orchestrator.assistance_logic import (
    generate_question_assistance,
    is_term_selection_reply,
)
from abridgeai.features.interviews.orchestrator.decision import (
    DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
    MAX_CANNOT_ANSWER_HINTS,
)
from abridgeai.features.interviews.orchestrator.interviewer_identity import (
    identity_from_config,
)
from abridgeai.features.interviews.orchestrator.security import (
    SecurityAction,
    SecurityAssessment,
    SecurityCategory,
    SecurityResponsePolicy,
)
from abridgeai.features.interviews.orchestrator.security_logic import (
    assess_by_rules,
    assess_security,
    choose_security_action,
    extract_separable_academic_text,
    is_explicit_end_request,
    requested_current_question_action,
    safe_closing,
    safe_security_response,
    should_flag_session,
)
from abridgeai.features.interviews.orchestrator.selection import CandidateQuestion
from abridgeai.features.interviews.orchestrator.utterance import (
    hint_ladder_exhausted_text,
)
from abridgeai.features.interviews.orchestrator.variant_selection import (
    select_logical_variants,
)
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.queries import sessions as sessions_queries
from abridgeai.features.interviews.realtime import observability as security_obs
from abridgeai.features.interviews.services import security as security_service
from abridgeai.features.interviews.services.ceremony import (
    FinishReason,
    ensure_ceremony_message,
    normalize_language,
    onboarding_ceremony_kind,
)
from abridgeai.features.interviews.services.retake import (
    InterviewCooldownActive,  # noqa: F401  -- re-exported: routers catch taking_service.InterviewCooldownActive
    InterviewMaxAttemptsReached,  # noqa: F401  -- re-exported: routers catch taking_service.InterviewMaxAttemptsReached
    RetakeStatus,
    _enforce_retake_policy,
    compute_retake_status,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData


logger = logging.getLogger(__name__)

_EVALUATE_INTERVIEW_SESSION_TASK = "evaluate_interview_session_task"


async def _require_session(db: AsyncSession, session_id: UUID) -> InterviewSession:
    session = await sessions_queries.get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Interview session {session_id} not found")
    return session


def _assert_owns_session(session: InterviewSession, actor: CurrentUser) -> None:
    if session.student_id != actor.user_id:
        raise ForbiddenError("Session does not belong to the calling user")


async def _first_published_question(
    db: AsyncSession,
    config_id: UUID,
    *,
    session_seed: str,
) -> InterviewQuestion | None:
    questions = await authoring_queries.list_questions_for_config(
        db,
        config_id,
        review_status="approved",
    )
    questions = await _runtime_question_pool(db, config_id, questions, session_seed)
    return questions[0] if questions else None


async def _runtime_question_pool(
    db: AsyncSession,
    config_id: UUID,
    questions: list[InterviewQuestion],
    session_seed: str,
) -> list[InterviewQuestion]:
    config = await db.get(InterviewConfig, config_id)
    if config is None:
        return questions

    candidates = [
        CandidateQuestion(
            question_id=str(question.id),
            linked_outcome_id=(
                str(question.linked_outcome_id) if question.linked_outcome_id is not None else None
            ),
            question_type=question.question_type,
            difficulty=question.difficulty,
            position=question.position,
            variant_group_id=(
                str(question.variant_group_id) if question.variant_group_id is not None else None
            ),
        )
        for question in questions
    ]
    selected = select_logical_variants(
        candidates,
        role=identity_from_config(config.persona_profile_json).role,
        session_seed=session_seed,
    )
    selected_ids = {candidate.question_id for candidate in selected}
    return [question for question in questions if str(question.id) in selected_ids]


async def _next_published_question_after(
    db: AsyncSession,
    config_id: UUID,
    asked_question_ids: set[UUID],
    *,
    session_seed: str,
) -> InterviewQuestion | None:
    questions = await authoring_queries.list_questions_for_config(
        db,
        config_id,
        review_status="approved",
    )
    questions = await _runtime_question_pool(db, config_id, questions, session_seed)
    for question in questions:
        if question.id not in asked_question_ids:
            return question
    return None


async def _next_session_sequence(db: AsyncSession, session_id: UUID) -> int:
    from sqlalchemy import func, select  # noqa: PLC0415

    row = await db.execute(
        select(func.coalesce(func.max(InterviewSessionQuestion.sequence_no), 0)).where(
            InterviewSessionQuestion.session_id == session_id
        )
    )
    return int(row.scalar_one()) + 1


async def start_session(
    db: AsyncSession,
    config_id: UUID,
    payload: Any,  # noqa: ANN401  -- InterviewSessionStartRequest at router edge.
    actor: CurrentUser,
    *,
    language: str = "en",
) -> InterviewSession:
    """Create (or return the existing live) :class:`InterviewSession`.

    Idempotent: a second ``start`` call while a previous session is
    still ``in_progress`` returns the same row instead of allocating a
    new attempt number — enforces plan §6.3 "1 active per config".

    Self-heal: if the existing live session has zero attached questions
    (e.g. it was created before any question was approved), attach the
    first approved question now so the learner doesn't get stuck on an
    empty session forever.

    Input mode: every session is a hybrid native-agent room (mic is an
    in-room toggle, typed turns ride ``lk.chat``). The request body may
    still carry ``input_mode`` for older clients; it is ignored.
    """
    data = payload.model_dump(exclude_unset=True) if payload is not None else {}
    del data  # accepted-and-ignored (see docstring); no field is read

    # Thesis §4.3: the pass/fail verdict is judged per learning outcome. Starting
    # an interview whose config has no outcomes guarantees an automatic fail with
    # nothing to evaluate — block it so the student never enters a no-win session.
    outcomes = await authoring_queries.list_outcomes_for_config(db, config_id)
    if not outcomes:
        raise AppError("interview_no_outcomes: this interview has no learning outcomes configured")

    existing = await sessions_queries.get_active_session(db, actor.user_id, config_id)
    if existing is not None:
        await _ensure_first_question_attached(db, existing.id, config_id)
        # A live session created before the mode unification may be recorded
        # "text"/"voice"; it now runs as the unified hybrid room too, so align
        # the row once (analytics read it, nothing branches on it any more).
        if existing.input_mode != "hybrid":
            existing.input_mode = "hybrid"
            await flush_or_conflict(db)
        await ensure_ceremony_message(
            db,
            session=existing,
            kind=onboarding_ceremony_kind(existing.onboarding_stage),
            language=existing.interview_language,
        )
        return existing

    # FR-5.3 retake policy — gate a *new* attempt on the config's
    # ``max_attempts`` ceiling and ``cooldown_minutes`` window. Only reached
    # when there is no live session (the idempotent short-circuit above
    # returns the in-progress row without consuming a fresh attempt).
    await _enforce_retake_policy(db, config_id=config_id, student_id=actor.user_id)
    attempt_number = await sessions_queries.get_session_attempt_number(db, actor.user_id, config_id)

    session = InterviewSession(
        interview_config_id=config_id,
        student_id=actor.user_id,
        attempt_number=attempt_number,
        status="in_progress",
        input_mode="hybrid",
        onboarding_stage="identity_check",
        interview_language=normalize_language(language),
    )
    db.add(session)
    await flush_or_conflict(db)

    first_question = await _first_published_question(db, config_id, session_seed=str(session.id))
    if first_question is not None:
        db.add(
            InterviewSessionQuestion(
                session_id=session.id,
                interview_question_id=first_question.id,
                sequence_no=1,
            )
        )
        await flush_or_conflict(db)

    await db.refresh(session)
    await ensure_ceremony_message(
        db,
        session=session,
        kind="candidate_confirmation",
        language=session.interview_language,
    )
    return session


async def _ensure_first_question_attached(
    db: AsyncSession, session_id: UUID, config_id: UUID
) -> None:
    """Attach question #1 to ``session_id`` if it has none yet.

    Closes the gap where ``start_session`` short-circuited on a stale
    ``in_progress`` row created before any question reached
    ``review_status='approved'``. Without this, approving questions
    after the fact never reaches the learner's existing session.
    """
    from sqlalchemy import func, select  # noqa: PLC0415

    count = (
        await db.execute(
            select(func.count(InterviewSessionQuestion.id)).where(
                InterviewSessionQuestion.session_id == session_id
            )
        )
    ).scalar_one()
    if int(count) > 0:
        return
    first_question = await _first_published_question(db, config_id, session_seed=str(session_id))
    if first_question is None:
        return
    db.add(
        InterviewSessionQuestion(
            session_id=session_id,
            interview_question_id=first_question.id,
            sequence_no=1,
        )
    )
    await flush_or_conflict(db)


async def take_session_step(  # noqa: C901 - shared legacy/adaptive turn coordinator
    db: AsyncSession,
    session_id: UUID,
    answer_text: str,
    actor: CurrentUser,
    *,
    audio_object_id: UUID | None = None,
    turn_key: str | None = None,
    language: str = "en",
    turn_action: str = "answer",
) -> dict[str, Any]:
    """Record the student's answer + advance the session.

    Returns a superset dict. The legacy keys ``{next_question, is_finished,
    followup_text}`` are ALWAYS present (existing callers unaffected). When the
    adaptive interviewer runs (``adaptive_interviewer_enabled`` AND the session
    is text/hybrid AND the pipeline succeeds), extra structured keys (``action``,
    ``reason_code``, ``ai_turn_text``, ``state_version``, …) are added and the
    canonical mapper guarantees the legacy + structured views never contradict.

    Adaptive is HARD-GATED to text/hybrid using the authoritative backend
    ``session.input_mode`` — voice sessions always run the exact legacy path
    until Phase 18 updates the LiveKit bridge. On ANY adaptive failure the
    savepoint rolls back (no orphan AI turn / partial state) and the legacy
    sequential path runs; the student's answer, recorded once in the outer
    transaction, always survives.
    """
    session = await _require_session(db, session_id)
    _assert_owns_session(session, actor)
    language = getattr(session, "interview_language", language)
    if getattr(session, "onboarding_stage", "completed") != "completed":
        raise AppError("Complete interview onboarding before answering questions")
    if session.status != "in_progress":
        raise AppError(f"Cannot record answer on session with status={session.status}")

    current_session_question = await _current_session_question(db, session_id)
    if current_session_question is None:
        raise AppError("Session has no current question to answer")

    current_question: InterviewQuestion | None = None
    if current_session_question.interview_question_id is not None:
        current_question = await db.get(
            InterviewQuestion, current_session_question.interview_question_id
        )

    settings = get_settings()
    try:
        security_stage = await _run_security_stage(
            db,
            session=session,
            current_session_question=current_session_question,
            current_question=current_question,
            answer_text=answer_text,
            audio_object_id=audio_object_id,
            client_turn_key=turn_key,
            language=language,
            turn_action=turn_action,
        )
    except Exception:  # noqa: BLE001 -- enforcement must fail closed, never bypass
        logger.exception("interview security stage failed (session=%s)", session_id)
        if settings.interview_security_guard_mode == "enforce":
            return _security_emergency_result(language, current_question)
        security_stage = _SecurityStageResult(
            turn_key=turn_key,
            assessment=_benign_assessment(source="stage_fallback", classifier_failed=True),
            action=SecurityAction.ALLOW,
            academic_text=answer_text,
            handled_result=None,
            attempt_count=0,
        )

    if security_stage.handled_result is not None:
        return security_stage.handled_result

    # ── Adaptive gate ────────────────────────────────────────────────────────
    # Resolved via the master switch AND the per-mode flag
    # (``adaptive_enabled_for_mode``). Text/hybrid default ON under the master
    # switch; voice defaults OFF until human sign-off flips
    # ADAPTIVE_INTERVIEWER_VOICE_ENABLED. When adaptive runs, the LiveKit bridge
    # consumes the canonical result (ai_turn_text / should_finish) exactly like
    # the REST path, so all enabled modes share the adaptive brain. Any adaptive
    # failure degrades to the legacy sequential path via the same fallback ladder.
    # Full gate = static mode gate AND the per-student percentage rollout
    # (Phase 11). rollout_percent=100 (default) makes this identical to the old
    # ``adaptive_enabled_for_mode`` check, so existing deployments are unaffected.
    adaptive_on = settings.adaptive_enabled_for_student(
        session.input_mode,
        student_id=str(session.student_id),
        config_id=str(session.interview_config_id),
    )

    # v2 phase-progression sub-feature (Slice 7). Resolved once here and threaded
    # into both the live and shadow adaptive calls. Defaults OFF → v1 behaviour.
    phases_enabled = settings.adaptive_v2_feature_enabled(session.input_mode, "phases")
    # v2 depth-probe sub-feature (Slice 8): dig into strong answers for the ceiling.
    depth_probe_enabled = settings.adaptive_v2_feature_enabled(session.input_mode, "depth_probe")
    # v2 cross-turn memory (Slice 9): feed prior claims into analysis for contradiction.
    cross_turn_enabled = settings.adaptive_v2_feature_enabled(session.input_mode, "cross_turn")
    # v2 affect (Slice 10): read candidate affect to warm utterance tone (phrasing only).
    affect_enabled = settings.adaptive_v2_feature_enabled(session.input_mode, "affect")
    # v2 hint ladder (Slice 11): escalate hints + vary rephrasing on the same question.
    hint_ladder_enabled = settings.adaptive_v2_feature_enabled(session.input_mode, "hint_ladder")
    # v2 per-outcome difficulty (Slice 12): calibrate question difficulty per topic competence.
    per_outcome_difficulty_enabled = settings.adaptive_v2_feature_enabled(
        session.input_mode, "per_outcome_difficulty"
    )
    # v2 rich closing (Slice 13): self-reflection + invite-questions closing sub-steps.
    rich_closing_enabled = settings.adaptive_v2_feature_enabled(session.input_mode, "rich_closing")
    # v2 self-correction (Slice 15): reward a candidate who fixes their own mistake.
    self_correction_enabled = settings.adaptive_v2_feature_enabled(
        session.input_mode, "self_correction"
    )
    # v2 confident-but-wrong challenge (Slice 16): lean in on a confidently wrong answer.
    confident_wrong_challenge_enabled = settings.adaptive_v2_feature_enabled(
        session.input_mode, "confident_wrong_challenge"
    )
    # v2 rambling redirect (Slice 17): steer a long, on-topic, low-substance ramble.
    rambling_redirect_enabled = settings.adaptive_v2_feature_enabled(
        session.input_mode, "rambling_redirect"
    )
    # v2 outcome backtracking (Slice 18): revisit under-covered outcomes before closing.
    backtrack_undercovered_enabled = settings.adaptive_v2_feature_enabled(
        session.input_mode, "backtrack_undercovered"
    )
    # v2 communication polish (Slice 20): time-pressure signaling + recovery framing (tone only).
    comms_polish_enabled = settings.adaptive_v2_feature_enabled(session.input_mode, "comms_polish")
    # v2 frustration de-escalation (Slice 19A): acknowledge + resume on frustration.
    frustration_deescalation_enabled = settings.adaptive_v2_feature_enabled(
        session.input_mode, "frustration_deescalation"
    )
    # v2 mid-interview question deferral (Slice 19B): defer a mid-interview candidate question.
    question_deferral_enabled = settings.adaptive_v2_feature_enabled(
        session.input_mode, "question_deferral"
    )
    # Emergent outcome evidence (SparkMe-inspired): let answer analysis credit an
    # outcome the current question did not target when the candidate
    # demonstrated it in passing, instead of discarding that evidence.
    emergent_evidence_enabled = settings.adaptive_v2_feature_enabled(
        session.input_mode, "emergent_evidence"
    )
    # v2 role-conditioned question filter: a config-scoped interviewer role
    # HARD-filters the candidate pool to its preferred question_type before
    # scoring. Off -> selection is role-blind (byte-for-byte v1).
    role_question_filter_enabled = settings.adaptive_v2_feature_enabled(
        session.input_mode, "role_question_filter"
    )

    if adaptive_on:
        adaptive_result = await _try_adaptive_step(
            db,
            session=session,
            current_session_question=current_session_question,
            current_question=current_question,
            answer_text=answer_text,
            audio_object_id=audio_object_id,
            client_turn_key=security_stage.turn_key,
            language=language,
            security_assessment=security_stage.assessment,
            security_action=security_stage.action,
            security_attempt_count=security_stage.attempt_count,
            phases_enabled=phases_enabled,
            depth_probe_enabled=depth_probe_enabled,
            cross_turn_enabled=cross_turn_enabled,
            affect_enabled=affect_enabled,
            hint_ladder_enabled=hint_ladder_enabled,
            per_outcome_difficulty_enabled=per_outcome_difficulty_enabled,
            rich_closing_enabled=rich_closing_enabled,
            self_correction_enabled=self_correction_enabled,
            confident_wrong_challenge_enabled=confident_wrong_challenge_enabled,
            rambling_redirect_enabled=rambling_redirect_enabled,
            backtrack_undercovered_enabled=backtrack_undercovered_enabled,
            comms_polish_enabled=comms_polish_enabled,
            frustration_deescalation_enabled=frustration_deescalation_enabled,
            question_deferral_enabled=question_deferral_enabled,
            emergent_evidence_enabled=emergent_evidence_enabled,
            role_question_filter_enabled=role_question_filter_enabled,
        )
        if adaptive_result is not None:
            return adaptive_result
        # adaptive declined/failed → answer already recorded; run legacy advance
        # WITHOUT re-inserting the answer (safeguard #2). Emit a live fallback
        # event (Slice 6) so the aggregator can track the adaptive→legacy
        # fallback rate that gates the staged rollout.
        security_obs.emit(
            security_obs.EV_FALLBACK,
            session_id=session.id,
            shadow=False,
            input_mode=session.input_mode,
        )
        return await _legacy_advance(
            db,
            session=session,
            current_session_question=current_session_question,
            current_question=current_question,
            answer_text=answer_text,
            language=language,
            turn_key=security_stage.turn_key,
            security_assessment=security_stage.assessment,
            security_action=security_stage.action,
            security_attempt_count=security_stage.attempt_count,
        )

    # ── Pure legacy path (flag off, voice session, or gated out by rollout) ───
    # Adaptive turns replay from canonical runtime state. Rebuild legacy retries
    # from allowlisted persisted records so REST retries and LiveKit reconnects
    # cannot insert a second answer or security event.
    if security_stage.assessment.source == "replay" and security_stage.turn_key:
        legacy_replay = await _replay_legacy_step(
            db,
            session_id=session.id,
            current_session_question=current_session_question,
            current_question=current_question,
            turn_key=security_stage.turn_key,
        )
        if legacy_replay is not None:
            return legacy_replay

    _add_student_answer(
        db,
        session_id,
        current_session_question.id,
        answer_text,
        audio_object_id,
        turn_key=security_stage.turn_key,
        metadata_extra={"security_assessed": True},
    )
    await flush_or_conflict(db)

    # ── Shadow mode (Phase 10) ────────────────────────────────────────────────
    # When shadowing this mode, COMPUTE the adaptive decision for comparison but
    # NEVER let it drive the student: the legacy result below is what the student
    # gets. The shadow computation runs in a savepoint that ALWAYS rolls back, so
    # it can never persist an AI turn or bump state — identical isolation to the
    # proven adaptive-fallback path. Best-effort: any failure is swallowed.
    if settings.shadow_enabled_for_mode(session.input_mode):
        await _run_shadow_step(
            db,
            session=session,
            current_session_question=current_session_question,
            current_question=current_question,
            answer_text=answer_text,
            language=language,
            turn_key=security_stage.turn_key,
            security_assessment=security_stage.assessment,
            security_action=security_stage.action,
            security_attempt_count=security_stage.attempt_count,
            phases_enabled=phases_enabled,
            depth_probe_enabled=depth_probe_enabled,
            cross_turn_enabled=cross_turn_enabled,
            affect_enabled=affect_enabled,
            hint_ladder_enabled=hint_ladder_enabled,
            per_outcome_difficulty_enabled=per_outcome_difficulty_enabled,
            rich_closing_enabled=rich_closing_enabled,
            self_correction_enabled=self_correction_enabled,
            confident_wrong_challenge_enabled=confident_wrong_challenge_enabled,
            rambling_redirect_enabled=rambling_redirect_enabled,
            backtrack_undercovered_enabled=backtrack_undercovered_enabled,
            comms_polish_enabled=comms_polish_enabled,
            frustration_deescalation_enabled=frustration_deescalation_enabled,
            question_deferral_enabled=question_deferral_enabled,
            emergent_evidence_enabled=emergent_evidence_enabled,
            role_question_filter_enabled=role_question_filter_enabled,
        )

    return await _legacy_advance(
        db,
        session=session,
        current_session_question=current_session_question,
        current_question=current_question,
        answer_text=answer_text,
        language=language,
        turn_key=security_stage.turn_key,
        security_assessment=security_stage.assessment,
        security_action=security_stage.action,
        security_attempt_count=security_stage.attempt_count,
    )


def _add_student_answer(
    db: AsyncSession,
    session_id: UUID,
    session_question_id: UUID,
    answer_text: str,
    audio_object_id: UUID | None,
    *,
    turn_key: str | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> None:
    """Add the student-answer message row (does not flush)."""
    metadata: dict[str, Any] = {"kind": "answer"}
    if turn_key is not None:
        metadata["turn_key"] = turn_key
    if metadata_extra:
        metadata.update(metadata_extra)
    db.add(
        InterviewSessionMessage(
            session_id=session_id,
            session_question_id=session_question_id,
            role="user",
            content_text=answer_text,
            audio_object_id=audio_object_id,
            metadata_json=metadata,
        )
    )


def _derive_turn_key(session_question_id: UUID, state_version: int, answer_text: str) -> str:
    """Stable idempotency key when the client did not supply one (safeguard #1).

    Folds the runtime ``state_version`` in so a *legitimate* second turn (which
    runs at a bumped version) never collides with the first even when the answer
    text repeats (e.g. two \"I don't know\"s), while a genuine retry (same
    version + same answer) resolves to the same key and is deduplicated.
    """
    import hashlib  # noqa: PLC0415

    digest = hashlib.sha256(answer_text.encode("utf-8")).hexdigest()[:16]
    return f"{session_question_id}:{state_version}:{digest}"


@dataclass(frozen=True)
class _SecurityStageResult:
    turn_key: str | None
    assessment: SecurityAssessment
    action: SecurityAction
    academic_text: str
    handled_result: dict[str, Any] | None
    attempt_count: int


def _benign_assessment(
    *, source: str = "rules", classifier_failed: bool = False
) -> SecurityAssessment:
    return SecurityAssessment(
        category=SecurityCategory.BENIGN,
        detected=False,
        confidence=1.0,
        should_block=False,
        should_record_academic_evidence=True,
        response_key=None,
        normalized_fingerprint=None,
        source=source,
        classifier_failed=classifier_failed,
    )


async def _config_max_hints(db: AsyncSession, config_id: UUID) -> int:
    """The teacher-configured hint cap for a session's config.

    Falls back to the shipped constant only when the config value is absent.
    Zero is a deliberate teacher setting that disables hints.
    """
    row = await db.get(InterviewConfig, config_id)
    value = getattr(row, "max_hints_per_question", None)
    return value if value is not None else MAX_CANNOT_ANSWER_HINTS


async def _run_security_stage(  # noqa: C901 -- precedence is kept explicit and auditable
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    answer_text: str,
    audio_object_id: UUID | None,
    client_turn_key: str | None,
    language: str,
    turn_action: str = "answer",
) -> _SecurityStageResult:
    """Shared pre-academic gate for adaptive, legacy, and LiveKit turns."""
    from abridgeai.features.interviews.orchestrator import repository as state_repo

    settings = get_settings()
    # Slice 11 upgrade: unify the hint ladder across BOTH hint paths. When the
    # hint_ladder v2 sub-flag is on for this mode, the assistance-stage hint
    # renders the SAME deterministic laddered hint (keyed on the shared
    # data.hint_level) that the adaptive decision path uses, and advances the
    # shared counter — so escalation is consistent no matter which path fires.
    # Off → the assistance stage keeps its one-hint-per-question reuse (v1).
    hint_ladder_enabled = settings.adaptive_v2_feature_enabled(session.input_mode, "hint_ladder")
    explicit_actions = {
        "repeat": SecurityAction.REPEAT_CURRENT_QUESTION,
        "clarify": SecurityAction.CLARIFY_CURRENT_QUESTION,
        "explain_term": SecurityAction.EXPLAIN_CURRENT_TERM,
        "hint": SecurityAction.HINT_CURRENT_QUESTION,
    }
    structured_action = explicit_actions.get(turn_action)
    if structured_action is not None and assess_by_rules(answer_text).detected:
        # A client-supplied control cannot override a mixed request for hidden
        # answers, rubrics, future questions, or internal instructions.
        structured_action = None
    explicit_end = is_explicit_end_request(answer_text) if turn_action == "answer" else False
    control_action = structured_action or (
        requested_current_question_action(answer_text) if turn_action == "answer" else None
    )
    if (
        control_action is None
        and turn_action == "answer"
        and current_question is not None
        and await _is_pending_term_selection(
            db,
            session_question_id=current_session_question.id,
            answer_text=answer_text,
            question_text=current_question.prompt_text,
        )
    ):
        control_action = SecurityAction.EXPLAIN_CURRENT_TERM
    if (
        settings.interview_security_guard_mode == "off"
        and not explicit_end
        and control_action is None
    ):
        return _SecurityStageResult(
            turn_key=client_turn_key,
            assessment=_benign_assessment(source="off"),
            action=SecurityAction.ALLOW,
            academic_text=answer_text,
            handled_result=None,
            attempt_count=0,
        )

    loaded = await state_repo.load_or_init(db, session.id)
    turn_key = client_turn_key or _derive_turn_key(
        current_session_question.id, loaded.version, answer_text
    )
    data = loaded.data

    # Security replay short-circuit happens before classifier/event/message
    # writes. The safe action can be rebuilt from bounded state.
    if data.last_security_turn_key == turn_key:
        if data.last_security_action == "normal_end":
            replay_assessment = _benign_assessment(source="replay")
            return _SecurityStageResult(
                turn_key=turn_key,
                assessment=replay_assessment,
                action=SecurityAction.ALLOW,
                academic_text="",
                handled_result=_controlled_result_dict(
                    safe_closing(language), language=language, is_finished=True
                ),
                attempt_count=data.security_attempt_count,
            )
        try:
            prior_action = SecurityAction(data.last_security_action or "allow")
        except ValueError:
            prior_action = SecurityAction.ALLOW
        try:
            replay_category = SecurityCategory(data.last_security_category or "benign")
        except ValueError:
            replay_category = SecurityCategory.BENIGN
        replay_assessment = SecurityAssessment(
            category=replay_category,
            detected=replay_category is not SecurityCategory.BENIGN,
            confidence=1.0,
            should_block=prior_action is not SecurityAction.ALLOW,
            should_record_academic_evidence=False,
            response_key=(
                f"security.{replay_category.value}"
                if replay_category is not SecurityCategory.BENIGN
                else None
            ),
            normalized_fingerprint=data.last_security_fingerprint,
            source="replay",
        )
        handled = None
        if prior_action is not SecurityAction.ALLOW:
            config = await db.get(InterviewConfig, session.interview_config_id)
            handled = await _security_action_result(
                db,
                session=session,
                current_session_question=current_session_question,
                current_question=current_question,
                config=config,
                answer_text=answer_text,
                audio_object_id=audio_object_id,
                turn_key=turn_key,
                language=language,
                assessment=replay_assessment,
                action=prior_action,
                attempt_count=data.security_attempt_count,
                persist_messages=False,
            )
        return _SecurityStageResult(
            turn_key=turn_key,
            assessment=replay_assessment,
            action=prior_action,
            academic_text=answer_text,
            handled_result=handled,
            attempt_count=data.security_attempt_count,
        )

    prior_category: SecurityCategory | None = None
    if data.last_security_category:
        try:
            prior_category = SecurityCategory(data.last_security_category)
        except ValueError:
            prior_category = None

    started = time.monotonic()
    if explicit_end or control_action is not None:
        assessment = _benign_assessment(source="control_rules")
        action = control_action or SecurityAction.ALLOW
    else:
        assessment = await assess_security(
            db,
            student_utterance=answer_text,
            last_category=prior_category,
        )
        config = await db.get(InterviewConfig, session.interview_config_id)
        try:
            response_policy = SecurityResponsePolicy(
                getattr(config, "security_response_policy", "warn_and_continue")
            )
        except ValueError:
            response_policy = SecurityResponsePolicy.WARN_AND_CONTINUE
        max_attempts = int(getattr(config, "security_max_consecutive_attempts", 3) or 3)
        action = choose_security_action(
            assessment,
            consecutive_attempts=data.consecutive_security_attempts,
            max_consecutive_attempts=max_attempts,
            response_policy=response_policy,
            guard_mode=settings.interview_security_guard_mode,
        )
        if (
            action is SecurityAction.END_AND_FLAG
            and not settings.interview_security_allow_session_termination
        ):
            action = SecurityAction.WARN_AND_REDIRECT
    latency_ms = round((time.monotonic() - started) * 1000.0, 2)

    config = await db.get(InterviewConfig, session.interview_config_id)
    max_attempts = int(getattr(config, "security_max_consecutive_attempts", 3) or 3)
    data.security_assessment_count += 1
    was_flagged = data.session_security_flagged
    if assessment.detected:
        data.security_attempt_count += 1
        data.consecutive_security_attempts += 1
        if data.consecutive_security_attempts >= 2:
            data.repeated_security_attempt_count += 1
        data.last_security_category = assessment.category.value
        data.last_security_fingerprint = assessment.normalized_fingerprint
        if action in (SecurityAction.WARN_AND_REDIRECT, SecurityAction.END_AND_FLAG):
            data.security_warning_issued = True
        if should_flag_session(
            attempt_count=data.consecutive_security_attempts,
            max_consecutive_attempts=max_attempts,
        ):
            data.session_security_flagged = True
            session.session_security_flagged = True
    else:
        data.consecutive_security_attempts = 0
    if assessment.classifier_failed:
        data.security_fallback_count += 1
    data.last_security_action = "normal_end" if explicit_end else action.value
    data.last_security_turn_key = turn_key

    academic_text = answer_text
    if control_action is not None:
        academic_text = ""
    if assessment.detected and action is not SecurityAction.ALLOW:
        separable = extract_separable_academic_text(answer_text)
        if separable:
            academic_text = separable
            assessment = replace(assessment, should_record_academic_evidence=True)
            await _record_separable_evidence(
                db,
                data=data,
                current_question=current_question,
                academic_text=separable,
                turn_id=str(current_session_question.id),
            )
        else:
            academic_text = ""

    # Slice 11 upgrade: shared hint ladder. When enabled and this turn is a hint
    # request, render at the CURRENT shared level then advance it (before the
    # save below), so a repeated hint on the same question escalates — unified
    # with the adaptive decision path, which reads/resets the same counter.
    #
    # The ladder's depth is ENFORCED here, not just reflected in the UI: the
    # learner client disables its hint control at the cap, but a caller hitting
    # the API directly used to keep getting hints forever (the counter only ever
    # incremented). How much help a candidate can draw on one question decides
    # what their grade means, so a client-side limit is not a limit — two
    # candidates in the same cohort could sit on different ladders.
    #
    # Past the cap the request is refused rather than served, and the counter is
    # left alone so it cannot drift away from the adaptive path's view of it.
    hint_render_level: int | None = None
    hint_ladder_spent = False
    if hint_ladder_enabled and action is SecurityAction.HINT_CURRENT_QUESTION:
        hint_cap = await _config_max_hints(db, session.interview_config_id)
        if data.hint_level >= hint_cap:
            hint_ladder_spent = True
        else:
            hint_render_level = data.hint_level
            data.hint_level += 1

    security_handled = explicit_end or action is not SecurityAction.ALLOW
    await state_repo.save(
        db,
        session.id,
        data,
        expected_version=loaded.version,
        turn_idempotency_key=turn_key if security_handled else None,
    )

    await security_service.record_security_event(
        db,
        session_id=session.id,
        turn_key=turn_key,
        event_type=security_obs.EV_SECURITY_ASSESSED,
        assessment=assessment,
        action=action,
        attempt_count=data.security_attempt_count,
        latency_ms=latency_ms,
        fallback_status=assessment.classifier_failed,
    )
    if assessment.detected and action is not SecurityAction.ALLOW:
        await security_service.record_security_event(
            db,
            session_id=session.id,
            turn_key=turn_key,
            event_type=security_obs.EV_SECURITY_BLOCKED,
            assessment=assessment,
            action=action,
            attempt_count=data.security_attempt_count,
        )
    if assessment.detected and data.consecutive_security_attempts >= 2:
        await security_service.record_security_event(
            db,
            session_id=session.id,
            turn_key=turn_key,
            event_type=security_obs.EV_SECURITY_REPEATED_ATTEMPT,
            assessment=assessment,
            action=action,
            attempt_count=data.security_attempt_count,
        )
    if data.session_security_flagged and not was_flagged:
        await security_service.record_security_event(
            db,
            session_id=session.id,
            turn_key=turn_key,
            event_type=security_obs.EV_SECURITY_SESSION_FLAGGED,
            assessment=assessment,
            action=action,
            attempt_count=data.security_attempt_count,
        )

    handled_result: dict[str, Any] | None = None
    if explicit_end:
        handled_result = await _normal_end_result(
            db,
            session=session,
            current_session_question=current_session_question,
            current_question=current_question,
            answer_text=answer_text,
            audio_object_id=audio_object_id,
            turn_key=turn_key,
            language=language,
            assessment=assessment,
            attempt_count=data.security_attempt_count,
        )
    elif action is not SecurityAction.ALLOW:
        handled_result = await _security_action_result(
            db,
            session=session,
            current_session_question=current_session_question,
            current_question=current_question,
            config=config,
            answer_text=answer_text,
            audio_object_id=audio_object_id,
            turn_key=turn_key,
            language=language,
            assessment=assessment,
            action=action,
            attempt_count=data.security_attempt_count,
            academic_text=academic_text or None,
            hint_render_level=hint_render_level,
            hint_ladder_spent=hint_ladder_spent,
        )

    return _SecurityStageResult(
        turn_key=turn_key,
        assessment=assessment,
        action=action,
        academic_text=academic_text,
        handled_result=handled_result,
        attempt_count=data.security_attempt_count,
    )


async def _record_separable_evidence(
    db: AsyncSession,
    *,
    data: InterviewRuntimeStateData,
    current_question: InterviewQuestion | None,
    academic_text: str,
    turn_id: str,
) -> None:
    if current_question is None or current_question.linked_outcome_id is None:
        return
    from abridgeai.features.interviews.models import InterviewOutcome  # noqa: PLC0415
    from abridgeai.features.interviews.orchestrator.analysis_logic import (  # noqa: PLC0415
        analyze_turn,
    )
    from abridgeai.features.interviews.orchestrator.coverage import (  # noqa: PLC0415
        apply_evidence_to_coverage,
    )
    from abridgeai.features.interviews.orchestrator.state import (  # noqa: PLC0415
        OutcomeCoverageState,
    )

    outcome = await db.get(InterviewOutcome, current_question.linked_outcome_id)
    # ``academic_text`` survived a turn the guard already flagged, so it is
    # known-adversarial: the sharpest case for routing through analyze_turn.
    analysis = await analyze_turn(
        db,
        question_text=current_question.prompt_text,
        student_answer=academic_text,
        turn_id=turn_id,
        outcome_id=str(current_question.linked_outcome_id),
        outcome_text=outcome.outcome_text if outcome is not None else None,
    )
    if analysis.confidence <= 0.0:
        return
    for evidence in analysis.evidence:
        coverage = data.outcome_coverage.get(evidence.outcome_id)
        if coverage is None:
            coverage = OutcomeCoverageState(outcome_id=evidence.outcome_id)
            data.outcome_coverage[evidence.outcome_id] = coverage
        apply_evidence_to_coverage(coverage, evidence)


async def _normal_end_result(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    answer_text: str,
    audio_object_id: UUID | None,
    turn_key: str,
    language: str,
    assessment: SecurityAssessment,
    attempt_count: int,
) -> dict[str, Any]:
    closing = safe_closing(language)
    return await _persist_controlled_result(
        db,
        session=session,
        current_session_question=current_session_question,
        current_question=current_question,
        answer_text=answer_text,
        audio_object_id=audio_object_id,
        turn_key=turn_key,
        language=language,
        assessment=assessment,
        action=SecurityAction.ALLOW,
        attempt_count=attempt_count,
        response_text=closing,
        is_finished=True,
        message_kind="end_request",
    )


async def _security_action_result(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    config: InterviewConfig | None,
    answer_text: str,
    audio_object_id: UUID | None,
    turn_key: str,
    language: str,
    assessment: SecurityAssessment,
    action: SecurityAction,
    attempt_count: int,
    academic_text: str | None = None,
    persist_messages: bool = True,
    hint_render_level: int | None = None,
    hint_ladder_spent: bool = False,
) -> dict[str, Any]:
    assistance_kind = {
        SecurityAction.REPEAT_CURRENT_QUESTION: "repeat",
        SecurityAction.CLARIFY_CURRENT_QUESTION: "clarification",
        SecurityAction.EXPLAIN_CURRENT_TERM: "term",
        SecurityAction.HINT_CURRENT_QUESTION: "hint",
    }.get(action)
    custom = None
    if config is not None:
        custom = (
            config.security_custom_refusal_vi
            if (language or "en").lower().startswith("vi")
            else config.security_custom_refusal_en
        )
    response = safe_security_response(
        action,
        language=language,
        current_question=(current_question.prompt_text if current_question else None),
        custom_refusal=custom,
    )
    if hint_ladder_spent and action is SecurityAction.HINT_CURRENT_QUESTION:
        # Ladder spent on this question: say so instead of serving another rung.
        # Deliberately NOT the persisted-reuse path below — replaying the last
        # hint would read as a fresh grant and hide the limit from the candidate.
        # Answer-safe by construction: it carries no content from the question.
        response = hint_ladder_exhausted_text(language)
    elif (
        hint_render_level is not None
        and action is SecurityAction.HINT_CURRENT_QUESTION
        and current_question is not None
    ):
        # Slice 11 upgrade: generate a question-SPECIFIC hint at the escalation
        # level the caller captured from data.hint_level. The model produces
        # phrasing tied to THIS question at this rung, and the shared
        # deterministic laddered hint (same one the adaptive decision path uses)
        # is the answer-safe fallback baked into generate_question_assistance if
        # the model is unavailable — so escalation holds either way. We do NOT
        # reuse the persisted one-hint here: each rung is a fresh, harder hint.
        response = await generate_question_assistance(
            db,
            action=action,
            question_text=current_question.prompt_text,
            request_text=answer_text,
            language=language,
            persona=getattr(config, "persona", None),
            hint_level=hint_render_level,
        )
    elif current_question is not None and action in {
        SecurityAction.CLARIFY_CURRENT_QUESTION,
        SecurityAction.EXPLAIN_CURRENT_TERM,
        SecurityAction.HINT_CURRENT_QUESTION,
    }:
        persisted = await _existing_assistance_text(
            db,
            session_question_id=current_session_question.id,
            kind=assistance_kind or "clarification",
        )
        # Generate a clarification once and allow only one neutral hint per
        # question. Reusing persisted text also makes repeated controls stable.
        if persisted is not None and action in {
            SecurityAction.CLARIFY_CURRENT_QUESTION,
            SecurityAction.HINT_CURRENT_QUESTION,
        }:
            response = persisted
        else:
            response = await generate_question_assistance(
                db,
                action=action,
                question_text=current_question.prompt_text,
                request_text=answer_text,
                language=language,
                persona=getattr(config, "persona", None),
            )
    if not persist_messages:
        return _controlled_result_dict(
            response,
            language=language,
            is_finished=action is SecurityAction.END_AND_FLAG,
            assistance_kind=assistance_kind,
        )
    return await _persist_controlled_result(
        db,
        session=session,
        current_session_question=current_session_question,
        current_question=current_question,
        answer_text=answer_text,
        audio_object_id=audio_object_id,
        turn_key=turn_key,
        language=language,
        assessment=assessment,
        action=action,
        attempt_count=attempt_count,
        response_text=response,
        is_finished=action is SecurityAction.END_AND_FLAG,
        message_kind=(
            "security"
            if assessment.detected
            else {
                "repeat": "turn_control",
                "clarification": "clarification",
                "term": "term_explanation",
                "hint": "hint",
            }.get(assistance_kind or "", "turn_control")
        ),
        academic_text=academic_text,
        assistance_kind=assistance_kind,
    )


async def _existing_assistance_text(
    db: AsyncSession,
    *,
    session_question_id: UUID,
    kind: str,
) -> str | None:
    """Return the latest persisted assistance of this kind for one question."""
    from sqlalchemy import select  # noqa: PLC0415

    rows = (
        await db.execute(
            select(InterviewSessionMessage)
            .where(
                InterviewSessionMessage.session_question_id == session_question_id,
                InterviewSessionMessage.role == "ai",
            )
            .order_by(InterviewSessionMessage.created_at.desc())
        )
    ).scalars()
    metadata_kind = {
        "clarification": "clarification",
        "term": "term_explanation",
        "hint": "hint",
    }.get(kind, kind)
    for message in rows:
        metadata = message.metadata_json if isinstance(message.metadata_json, dict) else {}
        content = (message.content_text or "").strip()
        if metadata.get("kind") == metadata_kind and content:
            return content
    return None


async def _is_pending_term_selection(
    db: AsyncSession,
    *,
    session_question_id: UUID,
    answer_text: str,
    question_text: str,
) -> bool:
    """Treat a bare term as assistance only after an explicit term prompt."""
    from sqlalchemy import select  # noqa: PLC0415

    message = (
        await db.execute(
            select(InterviewSessionMessage)
            .where(
                InterviewSessionMessage.session_question_id == session_question_id,
                InterviewSessionMessage.role == "ai",
            )
            .order_by(InterviewSessionMessage.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
    if message is None:
        return False
    metadata = message.metadata_json if isinstance(message.metadata_json, dict) else {}
    return is_term_selection_reply(
        answer_text,
        question_text,
        prior_assistance_text=message.content_text or "",
        prior_assistance_kind=metadata.get("kind"),
    )


async def _persist_controlled_result(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    answer_text: str,
    audio_object_id: UUID | None,
    turn_key: str,
    language: str,
    assessment: SecurityAssessment,
    action: SecurityAction,
    attempt_count: int,
    response_text: str,
    is_finished: bool,
    message_kind: str,
    academic_text: str | None = None,
    assistance_kind: str | None = None,
) -> dict[str, Any]:
    allowed_ids = [current_question.id] if current_question is not None else []
    deterministic_fallback = safe_security_response(
        action,
        language=language,
        current_question=(current_question.prompt_text if current_question else None),
    )
    guarded = await security_service.guard_student_output(
        db,
        session_id=session.id,
        config_id=session.interview_config_id,
        turn_key=turn_key,
        proposed_text=response_text,
        fallback_text=deterministic_fallback,
        allowed_question_ids=allowed_ids,
        assessment=assessment,
        action=action,
        attempt_count=attempt_count,
    )
    _add_student_answer(
        db,
        session.id,
        current_session_question.id,
        answer_text,
        audio_object_id,
        turn_key=turn_key,
        metadata_extra={
            "kind": message_kind,
            "security_category": assessment.category.value,
            "academic_evidence_allowed": bool(academic_text),
            **({"safe_academic_text": academic_text} if academic_text else {}),
        },
    )
    db.add(
        InterviewSessionMessage(
            session_id=session.id,
            session_question_id=current_session_question.id,
            role="ai",
            content_text=guarded.text,
            metadata_json={
                "kind": message_kind,
                "security_action": action.value,
                "turn_key": turn_key,
                "output_leakage_blocked": guarded.output_leakage_blocked,
                "output_fallback_used": guarded.output_fallback_used,
                "protected_content_category": guarded.protected_content_category,
            },
        )
    )
    await flush_or_conflict(db)
    return _controlled_result_dict(
        guarded.text,
        language=language,
        is_finished=is_finished,
        assistance_kind=assistance_kind,
    )


def _controlled_result_dict(
    text: str,
    *,
    language: str,
    is_finished: bool,
    assistance_kind: str | None = None,
) -> dict[str, Any]:
    result = {
        "next_question": None,
        "is_finished": is_finished,
        "followup_text": text,
        "ai_turn_text": text,
        "language": language,
        "should_narrate": True,
        "should_await_response": not is_finished,
        "should_finish": is_finished,
    }
    if assistance_kind is not None:
        result["assistance_kind"] = assistance_kind
    return result


def _security_emergency_result(
    language: str, current_question: InterviewQuestion | None
) -> dict[str, Any]:
    text = safe_security_response(
        SecurityAction.REFUSE_AND_REDIRECT,
        language=language,
        current_question=(current_question.prompt_text if current_question else None),
    )
    result = _controlled_result_dict(text, language=language, is_finished=False)
    result["_security_fallback"] = True
    return result


async def _try_adaptive_step(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    answer_text: str,
    audio_object_id: UUID | None,
    client_turn_key: str | None,
    language: str = "en",
    security_assessment: SecurityAssessment | None = None,
    security_action: SecurityAction = SecurityAction.ALLOW,
    security_attempt_count: int = 0,
    phases_enabled: bool = False,
    depth_probe_enabled: bool = False,
    cross_turn_enabled: bool = False,
    affect_enabled: bool = False,
    hint_ladder_enabled: bool = False,
    per_outcome_difficulty_enabled: bool = False,
    rich_closing_enabled: bool = False,
    self_correction_enabled: bool = False,
    confident_wrong_challenge_enabled: bool = False,
    rambling_redirect_enabled: bool = False,
    backtrack_undercovered_enabled: bool = False,
    comms_polish_enabled: bool = False,
    frustration_deescalation_enabled: bool = False,
    question_deferral_enabled: bool = False,
    emergent_evidence_enabled: bool = False,
    role_question_filter_enabled: bool = False,
) -> dict[str, Any] | None:
    """Attempt one adaptive turn. Returns the canonical result, or None to fall back.

    Order (safeguards #1, #2, #3):
      1. idempotency PRE-CHECK (before any answer insert);
      2. insert the student answer ONCE in the outer transaction;
      3. run the adaptive pipeline inside a SAVEPOINT — any failure rolls the
         savepoint back (no orphan AI turn) and returns None so the caller runs
         the legacy advance without re-inserting the answer.
    """
    from abridgeai.features.interviews.orchestrator import repository as state_repo
    from abridgeai.features.interviews.orchestrator.adaptive import run_adaptive_turn

    session_id = session.id

    # Config is required for persona / timing / supplementary instructions.
    config = await db.get(InterviewConfig, session.interview_config_id)
    if config is None:
        return None  # cannot run adaptive without config → legacy

    # 1. Idempotency pre-check (before inserting the answer).
    try:
        loaded = await state_repo.load_or_init(db, session_id)
    except Exception:  # noqa: BLE001 — runtime state unavailable → legacy path
        logger.exception("adaptive: runtime state load failed (session=%s)", session_id)
        return None

    effective_turn_key = client_turn_key or _derive_turn_key(
        current_session_question.id, loaded.version, answer_text
    )

    if state_repo.is_duplicate_turn(loaded, effective_turn_key):
        replay = _replay_from_state(loaded)
        if replay is not None:
            return await _rehydrate_step_result(db, replay)
        # No stored replay → safest is to not double-process; fall back.
        return None

    # 2. Insert the student answer ONCE (outer transaction). A concurrent
    #    duplicate trips the partial-unique index; treat that as a replay.
    try:
        async with db.begin_nested():
            _add_student_answer(
                db,
                session_id,
                current_session_question.id,
                answer_text,
                audio_object_id,
                turn_key=effective_turn_key,
            )
            await db.flush()
    except IntegrityError:
        logger.info("adaptive: duplicate answer insert (session=%s) — replaying", session_id)
        loaded = await state_repo.load_or_init(db, session_id)
        replay = _replay_from_state(loaded)
        return await _rehydrate_step_result(db, replay) if replay is not None else None

    # 3. Adaptive pipeline inside a savepoint.
    try:
        async with db.begin_nested():
            outcome = await run_adaptive_turn(
                db,
                session=session,
                config=config,
                current_session_question=current_session_question,
                current_question=current_question,
                answer_text=answer_text,
                turn_key=effective_turn_key,
                language=language,
                security_assessment=security_assessment,
                security_action=security_action,
                security_attempt_count=security_attempt_count,
                phases_enabled=phases_enabled,
                depth_probe_enabled=depth_probe_enabled,
                cross_turn_enabled=cross_turn_enabled,
                affect_enabled=affect_enabled,
                hint_ladder_enabled=hint_ladder_enabled,
                per_outcome_difficulty_enabled=per_outcome_difficulty_enabled,
                rich_closing_enabled=rich_closing_enabled,
                self_correction_enabled=self_correction_enabled,
                confident_wrong_challenge_enabled=confident_wrong_challenge_enabled,
                rambling_redirect_enabled=rambling_redirect_enabled,
                backtrack_undercovered_enabled=backtrack_undercovered_enabled,
                comms_polish_enabled=comms_polish_enabled,
                frustration_deescalation_enabled=frustration_deescalation_enabled,
                question_deferral_enabled=question_deferral_enabled,
                emergent_evidence_enabled=emergent_evidence_enabled,
                role_question_filter_enabled=role_question_filter_enabled,
            )
        if outcome.result is not None:
            _emit_live_decision(session_id, outcome.result)
            return _project_adaptive_result(outcome.result)
    except Exception:  # noqa: BLE001 — adaptive is best-effort; savepoint rolled back
        logger.exception("adaptive: pipeline failed (session=%s) — legacy fallback", session_id)
    return None


def _emit_live_decision(session_id: Any, canonical: dict[str, Any]) -> None:  # noqa: ANN401 - session_id is UUID|str
    """Emit the LIVE adaptive decision event (Slice 6).

    Mirrors the shadow-path emit but with ``shadow=False`` so the aggregator
    counts it toward the live adaptive-success / fallback rates that gate the
    staged rollout. Transcript-free (same privacy contract). Never raises.
    """
    from abridgeai.features.interviews.realtime import observability as obs  # noqa: PLC0415

    next_q = canonical.get("next_question")
    obs.emit(
        obs.EV_DECISION,
        session_id=session_id,
        shadow=False,
        adaptive=canonical.get("action") is not None,
        action=canonical.get("action"),
        reason_code=canonical.get("reason_code"),
        selected_question_id=canonical.get("current_question_id"),
        selected_question_type=getattr(next_q, "question_type", None),
        selected_question_difficulty=getattr(next_q, "difficulty", None),
        target_outcome_id=canonical.get("target_outcome_id"),
        state_version=canonical.get("state_version"),
        utterance_status=canonical.get("_utterance_status"),
        interaction_state=canonical.get("interaction_state"),
        pending_confirmation=bool(canonical.get("pending_confirmation")),
        finished=bool(canonical.get("should_finish")),
    )


def _project_adaptive_result(canonical: dict[str, Any]) -> dict[str, Any]:
    """Drop internal (underscore-prefixed) keys before returning to the router."""
    return {k: v for k, v in canonical.items() if not k.startswith("_")}


async def _run_shadow_step(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    answer_text: str,
    language: str = "en",
    turn_key: str | None = None,
    security_assessment: SecurityAssessment | None = None,
    security_action: SecurityAction = SecurityAction.ALLOW,
    security_attempt_count: int = 0,
    phases_enabled: bool = False,
    depth_probe_enabled: bool = False,
    cross_turn_enabled: bool = False,
    affect_enabled: bool = False,
    hint_ladder_enabled: bool = False,
    per_outcome_difficulty_enabled: bool = False,
    rich_closing_enabled: bool = False,
    self_correction_enabled: bool = False,
    confident_wrong_challenge_enabled: bool = False,
    rambling_redirect_enabled: bool = False,
    backtrack_undercovered_enabled: bool = False,
    comms_polish_enabled: bool = False,
    frustration_deescalation_enabled: bool = False,
    question_deferral_enabled: bool = False,
    emergent_evidence_enabled: bool = False,
    role_question_filter_enabled: bool = False,
) -> None:
    """Compute the adaptive decision for comparison WITHOUT driving the student.

    Shadow mode (Phase 10): the student has already been served the legacy path;
    here we run the adaptive pipeline purely to log WHAT it *would* have decided,
    so we can compare adaptive vs legacy on real traffic with zero student-facing
    risk. The pipeline runs inside a savepoint that ALWAYS rolls back — it can
    never persist an AI turn, bump state, or otherwise leak into the live
    session (identical isolation to the adaptive-fallback path). The student
    answer is NOT re-inserted here (the caller already recorded it).

    Best-effort and silent: any failure is swallowed so shadow computation can
    never affect the interview the student is actually taking.
    """
    from abridgeai.features.interviews.orchestrator.adaptive import run_adaptive_turn
    from abridgeai.features.interviews.realtime import observability as obs

    session_id = session.id
    config = await db.get(InterviewConfig, session.interview_config_id)
    if config is None:
        return

    shadow_turn_key = f"shadow:{turn_key or current_session_question.id}:{uuid4().hex}"
    try:
        # begin_nested() opens a savepoint; we roll it back unconditionally in
        # the finally block so nothing the pipeline wrote ever commits.
        savepoint = await db.begin_nested()
        try:
            outcome = await run_adaptive_turn(
                db,
                session=session,
                config=config,
                current_session_question=current_session_question,
                current_question=current_question,
                answer_text=answer_text,
                turn_key=shadow_turn_key,
                language=language,
                security_assessment=security_assessment,
                security_action=security_action,
                security_attempt_count=security_attempt_count,
                phases_enabled=phases_enabled,
                depth_probe_enabled=depth_probe_enabled,
                cross_turn_enabled=cross_turn_enabled,
                affect_enabled=affect_enabled,
                hint_ladder_enabled=hint_ladder_enabled,
                per_outcome_difficulty_enabled=per_outcome_difficulty_enabled,
                rich_closing_enabled=rich_closing_enabled,
                self_correction_enabled=self_correction_enabled,
                confident_wrong_challenge_enabled=confident_wrong_challenge_enabled,
                rambling_redirect_enabled=rambling_redirect_enabled,
                backtrack_undercovered_enabled=backtrack_undercovered_enabled,
                comms_polish_enabled=comms_polish_enabled,
                frustration_deescalation_enabled=frustration_deescalation_enabled,
                question_deferral_enabled=question_deferral_enabled,
                emergent_evidence_enabled=emergent_evidence_enabled,
                role_question_filter_enabled=role_question_filter_enabled,
            )
            canonical = outcome.result or {}
            # Emit the shadow decision (no transcript content — same privacy
            # contract as the live decision event), tagged shadow=True so the
            # aggregator can separate it from real decisions.
            next_q = canonical.get("next_question")
            obs.emit(
                obs.EV_DECISION,
                session_id=session_id,
                shadow=True,
                adaptive=canonical.get("action") is not None,
                action=canonical.get("action"),
                reason_code=canonical.get("reason_code"),
                selected_question_id=canonical.get("current_question_id"),
                selected_question_type=getattr(next_q, "question_type", None),
                selected_question_difficulty=getattr(next_q, "difficulty", None),
                target_outcome_id=canonical.get("target_outcome_id"),
                state_version=canonical.get("state_version"),
                utterance_status=canonical.get("_utterance_status"),
                answer_chars=len(answer_text),
                finished=bool(canonical.get("should_finish")),
            )
        finally:
            # ALWAYS roll back — shadow must never persist anything.
            await savepoint.rollback()
    except Exception:  # noqa: BLE001 — shadow is best-effort; never affect the live turn
        logger.exception("shadow: adaptive computation failed (session=%s)", session_id)


def _replay_from_state(loaded: Any) -> dict[str, Any] | None:  # noqa: ANN401
    """Extract a stored canonical replay dict from runtime state, if present."""
    prior = getattr(loaded.data, "last_answer_analysis", None)
    if isinstance(prior, dict):
        replay = prior.get("_canonical_result")
        if isinstance(replay, dict):
            return replay
    return None


async def _rehydrate_step_result(db: AsyncSession, replay: dict[str, Any]) -> dict[str, Any]:
    """Turn a stored replay into a live result dict (re-fetch next_question ORM)."""
    result = dict(replay)
    nq = result.get("next_question")
    if isinstance(nq, str):
        result["next_question"] = await db.get(InterviewQuestion, UUID(nq))
    return result


async def _replay_legacy_step(
    db: AsyncSession,
    *,
    session_id: UUID,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    turn_key: str,
) -> dict[str, Any] | None:
    """Reconstruct a safe legacy response without re-running a turn."""
    messages = await sessions_queries.list_session_messages(db, session_id)
    matching_user = next(
        (
            message
            for message in messages
            if message.role == "user"
            and isinstance(message.metadata_json, dict)
            and message.metadata_json.get("turn_key") == turn_key
        ),
        None,
    )
    if matching_user is None:
        return None

    matching_ai = next(
        (
            message
            for message in messages
            if message.role == "ai"
            and isinstance(message.metadata_json, dict)
            and message.metadata_json.get("turn_key") == turn_key
        ),
        None,
    )
    if matching_ai is not None:
        return {
            "next_question": None,
            "is_finished": False,
            "followup_text": matching_ai.content_text,
            "ai_turn_text": matching_ai.content_text,
        }

    if matching_user.session_question_id != current_session_question.id:
        return {
            "next_question": current_question,
            "is_finished": False,
            "followup_text": None,
        }
    return {
        "next_question": None,
        "is_finished": True,
        "followup_text": None,
    }


async def _legacy_advance(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    answer_text: str,
    language: str = "en",
    turn_key: str | None = None,
    security_assessment: SecurityAssessment | None = None,
    security_action: SecurityAction = SecurityAction.ALLOW,
    security_attempt_count: int = 0,
) -> dict[str, Any]:
    """The exact legacy sequential progression (unchanged behaviour).

    Assumes the student answer has ALREADY been recorded by the caller. Runs the
    single-follow-up stage, then advances to the next approved question or
    signals ``is_finished``.
    """
    assessment = security_assessment or _benign_assessment(source="legacy")
    effective_turn_key = turn_key or f"legacy:{current_session_question.id}:{uuid4().hex}"
    followup_text: str | None = None
    if current_question is not None:
        config = await db.get(InterviewConfig, session.interview_config_id)
        max_follow_ups_per_question = (
            config.max_follow_ups_per_question
            if config is not None and config.max_follow_ups_per_question is not None
            else DEFAULT_MAX_FOLLOWUPS_PER_QUESTION
        )
        followup_text = await maybe_generate_followup(
            db,
            session=session,
            current_question=current_question,
            student_answer=answer_text,
            max_follow_ups_per_question=max_follow_ups_per_question,
        )

    if followup_text:
        fallback_followup = (
            "Bạn có thể làm rõ câu trả lời của mình cho câu hỏi hiện tại không?"
            if (language or "en").lower().startswith("vi")
            else "Could you clarify your answer to the current question?"
        )
        guarded = await security_service.guard_student_output(
            db,
            session_id=session.id,
            config_id=session.interview_config_id,
            turn_key=effective_turn_key,
            proposed_text=followup_text,
            fallback_text=fallback_followup,
            allowed_question_ids=([current_question.id] if current_question else []),
            assessment=assessment,
            action=security_action,
            attempt_count=security_attempt_count,
        )
        followup_text = guarded.text
        db.add(
            InterviewSessionMessage(
                session_id=session.id,
                session_question_id=current_session_question.id,
                role="ai",
                content_text=followup_text,
                metadata_json={
                    "kind": "followup",
                    "turn_key": effective_turn_key,
                    "output_leakage_blocked": guarded.output_leakage_blocked,
                    "output_fallback_used": guarded.output_fallback_used,
                    "protected_content_category": guarded.protected_content_category,
                },
            )
        )
        await flush_or_conflict(db)
        return {
            "next_question": None,
            "is_finished": False,
            "followup_text": followup_text,
        }

    asked_ids = await _asked_question_ids(db, session.id)
    next_question = await _next_published_question_after(
        db,
        session.interview_config_id,
        asked_ids,
        session_seed=str(session.id),
    )
    if next_question is None:
        # No further question: persist the short final-question transition as
        # its OWN AI turn (Natural Interview Transitions spec §ending). The
        # separate goodbye/closing turn is added later by the finish flow, so
        # the ending is two short turns. Idempotent via a stable turn key.
        from abridgeai.features.interviews.orchestrator.utterance import (  # noqa: PLC0415
            persona_from,
            transition_text,
        )

        config = await db.get(InterviewConfig, session.interview_config_id)
        persona = persona_from(config.persona if config is not None else None)
        final_turn_key = f"transition-final:{effective_turn_key}"
        final_text = transition_text(persona, language, final=True)
        existing_messages = await sessions_queries.list_session_messages(db, session.id)
        final_exists = any(
            message.role == "ai"
            and isinstance(message.metadata_json, dict)
            and message.metadata_json.get("turn_key") == final_turn_key
            for message in existing_messages
        )
        if not final_exists:
            db.add(
                InterviewSessionMessage(
                    session_id=session.id,
                    session_question_id=current_session_question.id,
                    role="ai",
                    content_text=final_text,
                    metadata_json={
                        "kind": "transition",
                        "turn_key": final_turn_key,
                        "transition_target": "closing",
                    },
                )
            )
            await flush_or_conflict(db)
        return {
            "next_question": None,
            "is_finished": True,
            "followup_text": final_text,
            "ai_turn_text": final_text,
            "transition_id": final_turn_key,
            "transition_text": final_text,
            "transition_target": "closing",
        }

    allowed_ids = [next_question.id]
    if current_question is not None:
        allowed_ids.append(current_question.id)
    next_guard = await security_service.guard_student_output(
        db,
        session_id=session.id,
        config_id=session.interview_config_id,
        turn_key=effective_turn_key,
        proposed_text=next_question.prompt_text,
        fallback_text=(
            "Tôi có thể nhắc lại hoặc giải thích câu hỏi hiện tại."
            if (language or "en").lower().startswith("vi")
            else "I can repeat or clarify the current question."
        ),
        allowed_question_ids=allowed_ids,
        assessment=assessment,
        action=security_action,
        attempt_count=security_attempt_count,
    )
    if next_guard.output_leakage_blocked:
        db.add(
            InterviewSessionMessage(
                session_id=session.id,
                session_question_id=current_session_question.id,
                role="ai",
                content_text=next_guard.text,
                metadata_json={
                    "kind": "output_guard_fallback",
                    "turn_key": effective_turn_key,
                    "output_leakage_blocked": True,
                    "output_fallback_used": True,
                    "protected_content_category": next_guard.protected_content_category,
                },
            )
        )
        await flush_or_conflict(db)
        return {
            "next_question": None,
            "is_finished": False,
            "followup_text": next_guard.text,
            "ai_turn_text": next_guard.text,
        }

    # Persist a standardized, persona-aware transition as its OWN AI turn before
    # the next question turn (Natural Interview Transitions spec). The question
    # text is NOT included here — it rides in ``next_question`` — so it is never
    # duplicated. A stable turn key keyed to the destination question makes
    # retries idempotent (no duplicate transition on resend).
    from abridgeai.features.interviews.orchestrator.utterance import (  # noqa: PLC0415
        persona_from,
        transition_text,
    )

    config = await db.get(InterviewConfig, session.interview_config_id)
    persona = persona_from(config.persona if config is not None else None)
    transition_turn_key = f"transition:{effective_turn_key}:{next_question.id}"
    transition_text_value = transition_text(persona, language)
    existing_messages = await sessions_queries.list_session_messages(db, session.id)
    transition_exists = any(
        message.role == "ai"
        and isinstance(message.metadata_json, dict)
        and message.metadata_json.get("turn_key") == transition_turn_key
        for message in existing_messages
    )
    if not transition_exists:
        db.add(
            InterviewSessionMessage(
                session_id=session.id,
                session_question_id=current_session_question.id,
                role="ai",
                content_text=transition_text_value,
                metadata_json={
                    "kind": "transition",
                    "turn_key": transition_turn_key,
                    "transition_target": "next_question",
                },
            )
        )
        await flush_or_conflict(db)

    sequence_no = await _next_session_sequence(db, session.id)
    db.add(
        InterviewSessionQuestion(
            session_id=session.id,
            interview_question_id=next_question.id,
            sequence_no=sequence_no,
        )
    )
    await flush_or_conflict(db)
    return {
        "next_question": next_question,
        "is_finished": False,
        "followup_text": transition_text_value,
        "ai_turn_text": transition_text_value,
        "transition_id": transition_turn_key,
        "transition_text": transition_text_value,
        "transition_target": "next_question",
    }


async def submit_session(
    db: AsyncSession,
    session_id: UUID,
    actor: CurrentUser,
    *,
    arq_pool: object | None = None,
    reason: FinishReason = "natural",
    language: str = "en",
) -> InterviewSession:
    """Persist the final ceremony, terminalize the session, and enqueue evaluation.

    Natural and explicit early completion are ``completed`` and graded; unanswered
    questions score zero, because reaching the assessment and skipping questions
    still counts as having been assessed.

    A run that never left onboarding (identity / audio check / readiness) has
    nothing to grade — grading one fabricated verdicts from onboarding chatter and
    burned an attempt. Those terminalize as ``abandoned`` and are never enqueued,
    matching the stale-session sweep. A deadline completion is ``timed_out`` when
    answers exist, ``abandoned`` otherwise. Commits inline so the worker sees
    terminal state before dequeueing.
    """
    session = await _require_session(db, session_id)
    _assert_owns_session(session, actor)
    language = getattr(session, "interview_language", language)
    if session.status != "in_progress":
        return session

    config = await db.get(InterviewConfig, session.interview_config_id)
    ended_at = utcnow()
    if reason == "timed_out":
        if config is None or not config.time_limit_minutes:
            raise AppError("Cannot time out an interview without a time limit")
        started_at = session.assessment_started_at or session.started_at
        if started_at.tzinfo is None:
            started_at = started_at.replace(tzinfo=UTC)
        deadline = started_at + timedelta(minutes=config.time_limit_minutes)
        # Allow a small client/server scheduling tolerance at the boundary.
        if ended_at + timedelta(seconds=2) < deadline:
            raise AppError("Interview time limit has not elapsed")

    from sqlalchemy import func, select  # noqa: PLC0415

    user_message_count = int(
        (
            await db.execute(
                select(func.count(InterviewSessionMessage.id)).where(
                    InterviewSessionMessage.session_id == session.id,
                    InterviewSessionMessage.role == "user",
                    InterviewSessionMessage.session_question_id.is_not(None),
                )
            )
        ).scalar_one()
    )
    await ensure_ceremony_message(
        db,
        session=session,
        kind="closing",
        language=language,
        reason=reason,
    )
    # A run that never reached the assessment has no gradeable transcript, no
    # matter HOW it ended (previously only the timed_out branch checked, so an
    # onboarding quit + submit was marked ``completed`` and graded: 14 production
    # sessions carry verdicts derived from onboarding chatter alone).
    never_reached_assessment = session.assessment_started_at is None
    session.status = (
        "abandoned"
        if never_reached_assessment or (user_message_count == 0 and reason == "timed_out")
        else "timed_out"
        if reason == "timed_out"
        else "completed"
    )
    session.ended_at = ended_at
    await db.commit()
    await db.refresh(session)

    # Reaching the assessment is the line, not answering: a candidate who got to
    # the questions and skipped them HAS been assessed and is still graded
    # (unanswered questions score zero).
    has_content = not never_reached_assessment and (user_message_count > 0 or reason != "timed_out")
    if arq_pool is not None and has_content:
        await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            _EVALUATE_INTERVIEW_SESSION_TASK,
            actor.user_id,
            session.id,
            _job_id=f"interview-evaluation:{session.id}",
        )
    return session


async def session_time_remaining_seconds(db: AsyncSession, session: InterviewSession) -> int | None:
    """Return the authoritative whole-second countdown for a live session."""
    config = await db.get(InterviewConfig, session.interview_config_id)
    if config is None or not config.time_limit_minutes:
        return None
    started_at = session.assessment_started_at
    if started_at is None:
        return int(config.time_limit_minutes * 60)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)
    deadline = started_at + timedelta(minutes=config.time_limit_minutes)
    return max(0, int((deadline - datetime.now(UTC)).total_seconds()))


async def get_session_for_user(
    db: AsyncSession, session_id: UUID, user_id: UUID
) -> InterviewSession | None:
    session = await sessions_queries.get_session(db, session_id)
    if session is None or session.student_id != user_id:
        return None
    return session


async def get_user_sessions(
    db: AsyncSession,
    user_id: UUID,
    *,
    status: str | None = None,
    interview_config_id: UUID | None = None,
) -> list[InterviewSession]:
    return await sessions_queries.get_user_interview_sessions(
        db, user_id, status=status, interview_config_id=interview_config_id
    )


async def _current_session_question(
    db: AsyncSession, session_id: UUID
) -> InterviewSessionQuestion | None:
    from sqlalchemy import select  # noqa: PLC0415

    stmt = (
        select(InterviewSessionQuestion)
        .where(InterviewSessionQuestion.session_id == session_id)
        .order_by(InterviewSessionQuestion.sequence_no.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _asked_question_ids(db: AsyncSession, session_id: UUID) -> set[UUID]:
    from sqlalchemy import select  # noqa: PLC0415

    stmt = select(InterviewSessionQuestion.interview_question_id).where(
        InterviewSessionQuestion.session_id == session_id
    )
    rows = (await db.execute(stmt)).scalars().all()
    return {row for row in rows if row is not None}


__all__ = [
    "RetakeStatus",
    "compute_retake_status",
    "get_session_for_user",
    "get_user_sessions",
    "start_session",
    "session_time_remaining_seconds",
    "submit_session",
    "take_session_step",
]
