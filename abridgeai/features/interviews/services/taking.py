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
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.queries import sessions as sessions_queries

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession


logger = logging.getLogger(__name__)

_EVALUATE_INTERVIEW_SESSION_TASK = "evaluate_interview_session_task"

# Terminal session statuses — an attempt in any of these has been "used up"
# and its ``ended_at`` anchors the retake cooldown (FR-5.3). ``in_progress``
# is excluded: a live session is handled by the idempotent active-session
# short-circuit, not the cooldown gate.
_TERMINAL_SESSION_STATUSES = ("completed", "timed_out", "abandoned", "failed")


class InterviewCooldownActive(AppError):  # noqa: N818  # mirrors quiz CooldownActive
    """FR-5.3 — student tried to start a new attempt inside the config's
    ``cooldown_hours`` window since their last attempt ended.

    Carries ``retry_after`` so the router can surface a ``Retry-After``
    hint (mapped to HTTP 429). Mirrors the quiz-side
    :class:`~abridgeai.features.quizzes.queries.published.CooldownActive`.
    """

    def __init__(self, config_id: UUID, retry_after: datetime) -> None:
        super().__init__("Interview retake cooldown is still active")
        self.config_id = config_id
        self.retry_after = retry_after


class InterviewMaxAttemptsReached(AppError):  # noqa: N818  # mirrors quiz MaxAttemptsReached
    """FR-5.3 — student has used every attempt allowed by ``max_attempts``.

    Mapped to HTTP 409 by the router.
    """

    def __init__(self, config_id: UUID, max_attempts: int) -> None:
        super().__init__("Interview attempt limit reached")
        self.config_id = config_id
        self.max_attempts = max_attempts


async def _require_session(db: AsyncSession, session_id: UUID) -> InterviewSession:
    session = await sessions_queries.get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Interview session {session_id} not found")
    return session


def _assert_owns_session(session: InterviewSession, actor: CurrentUser) -> None:
    if session.student_id != actor.user_id:
        raise ForbiddenError("Session does not belong to the calling user")


async def _first_published_question(db: AsyncSession, config_id: UUID) -> InterviewQuestion | None:
    questions = await authoring_queries.list_questions_for_config(
        db, config_id, review_status="approved"
    )
    return questions[0] if questions else None


async def _next_published_question_after(
    db: AsyncSession,
    config_id: UUID,
    asked_question_ids: set[UUID],
) -> InterviewQuestion | None:
    questions = await authoring_queries.list_questions_for_config(
        db, config_id, review_status="approved"
    )
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


async def _enforce_retake_policy(
    db: AsyncSession,
    *,
    config_id: UUID,
    student_id: UUID,
) -> None:
    """FR-5.3 — block a new attempt on ``max_attempts`` / ``cooldown_hours``.

    Loads the config, then applies two gates against the student's
    *terminal* sessions (live sessions never reach here):

    * **Cooldown** — if ``cooldown_hours`` is set and the most recent
      terminal attempt ended less than that many hours ago, raise
      :class:`InterviewCooldownActive` (router → 429).
    * **Max attempts** — if ``max_attempts`` is set and the count of
      terminal attempts already meets it, raise
      :class:`InterviewMaxAttemptsReached` (router → 409).

    NULL / non-positive knobs disable the corresponding gate. The gates
    are read-time only — no data migration normalises historical rows.
    """
    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415

    config = await db.get(InterviewConfig, config_id)
    if config is None:
        return

    cooldown_hours = config.cooldown_hours
    if cooldown_hours is not None and cooldown_hours > 0:
        last_ended = await sessions_queries.get_last_terminal_ended_at(
            db, student_id, config_id, _TERMINAL_SESSION_STATUSES
        )
        if last_ended is not None:
            if last_ended.tzinfo is None:
                last_ended = last_ended.replace(tzinfo=UTC)
            retry_after = last_ended + timedelta(hours=cooldown_hours)
            if retry_after > datetime.now(UTC):
                raise InterviewCooldownActive(config_id, retry_after)

    max_attempts = config.max_attempts
    if max_attempts is not None and max_attempts > 0:
        used = await sessions_queries.count_terminal_sessions(
            db, student_id, config_id, _TERMINAL_SESSION_STATUSES
        )
        if used >= max_attempts:
            raise InterviewMaxAttemptsReached(config_id, max_attempts)


async def start_session(
    db: AsyncSession,
    config_id: UUID,
    payload: Any,  # noqa: ANN401  -- InterviewSessionStartRequest at router edge.
    actor: CurrentUser,
) -> InterviewSession:
    """Create (or return the existing live) :class:`InterviewSession`.

    Idempotent: a second ``start`` call while a previous session is
    still ``in_progress`` returns the same row instead of allocating a
    new attempt number — enforces plan §6.3 "1 active per config".

    Self-heal: if the existing live session has zero attached questions
    (e.g. it was created before any question was approved), attach the
    first approved question now so the learner doesn't get stuck on an
    empty session forever.

    Input-mode upgrade: if the caller specifies an ``input_mode`` that
    differs from the existing session's recorded mode AND the parent
    config allows it, update the session in-place. Without this, a
    student whose first ``start`` request landed as ``text`` could never
    escalate to ``voice``/``hybrid`` — the realtime-token endpoint then
    rejects them with HTTP 409 "session is not a voice interview".
    """
    # Interviews are always hybrid now (AI speaks + writes; student types or
    # speaks). Ignore any client-supplied input_mode and pin to hybrid so a
    # stale/legacy client can't downgrade the session to text/voice-only.
    requested_input_mode = "hybrid"

    # Thesis §4.3: the pass/fail verdict is judged per learning outcome. Starting
    # an interview whose config has no outcomes guarantees an automatic fail with
    # nothing to evaluate — block it so the student never enters a no-win session.
    outcomes = await authoring_queries.list_outcomes_for_config(db, config_id)
    if not outcomes:
        raise AppError("interview_no_outcomes: this interview has no learning outcomes configured")

    existing = await sessions_queries.get_active_session(db, actor.user_id, config_id)
    if existing is not None:
        await _ensure_first_question_attached(db, existing.id, config_id)
        await _maybe_upgrade_input_mode(db, existing, config_id, requested_input_mode)
        return existing

    # FR-5.3 retake policy — gate a *new* attempt on the config's
    # ``max_attempts`` ceiling and ``cooldown_hours`` window. Only reached
    # when there is no live session (the idempotent short-circuit above
    # returns the in-progress row without consuming a fresh attempt).
    await _enforce_retake_policy(db, config_id=config_id, student_id=actor.user_id)

    attempt_number = await sessions_queries.get_session_attempt_number(db, actor.user_id, config_id)
    session = InterviewSession(
        interview_config_id=config_id,
        student_id=actor.user_id,
        attempt_number=attempt_number,
        status="in_progress",
        input_mode=requested_input_mode,
    )
    db.add(session)
    await flush_or_conflict(db)

    first_question = await _first_published_question(db, config_id)
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
    return session


_CONFIG_MODE_ALLOWS: dict[str, set[str]] = {
    "text": {"text"},
    "voice": {"voice"},
    "hybrid": {"text", "voice", "hybrid"},
}


async def _maybe_upgrade_input_mode(
    db: AsyncSession,
    session: InterviewSession,
    config_id: UUID,
    requested: str,
) -> None:
    """Promote ``session.input_mode`` to ``requested`` when permitted.

    No-op when the modes already match or when the parent config does
    not allow the requested mode. Silent on mismatch by design — the
    realtime-token endpoint is the right place to reject voice on a
    text-only config; here we just sync the session to what the caller
    asked for.
    """
    if session.input_mode == requested:
        return
    from abridgeai.features.interviews.models import InterviewConfig  # noqa: PLC0415

    config = await db.get(InterviewConfig, config_id)
    if config is None:
        return
    allowed = _CONFIG_MODE_ALLOWS.get(config.supported_modes, set())
    if requested not in allowed:
        return
    session.input_mode = requested
    await flush_or_conflict(db)


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
    first_question = await _first_published_question(db, config_id)
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


async def take_session_step(
    db: AsyncSession,
    session_id: UUID,
    answer_text: str,
    actor: CurrentUser,
    *,
    audio_object_id: UUID | None = None,
    turn_key: str | None = None,
    language: str = "en",
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

    # ── Security guard (prompt-injection defense, Phase S1) ───────────────────
    # Runs BEFORE the academic pipeline (and the intent rules). One choke point
    # for text/hybrid/voice + adaptive/legacy; internally no-ops when the guard
    # mode is "off" and only blocks in "enforce" — see security_handler.
    settings = get_settings()
    blocked = await maybe_security_block(
        db,
        session=session,
        current_session_question=current_session_question,
        answer_text=answer_text,
        audio_object_id=audio_object_id,
        language=language,
    )
    if blocked is not None:
        return blocked

    # ── Adaptive is ALWAYS ON for interviews (2026-07 product decision) ───────
    # Interviews are a single hybrid, always-adaptive experience. The master
    # flag ``adaptive_interviewer_enabled`` is now an emergency KILL SWITCH:
    # only an explicit ``=false`` forces the legacy sequential path. On ANY
    # adaptive failure the savepoint rolls back and legacy runs; the student's
    # answer, recorded once, always survives.
    adaptive_on = settings.adaptive_interviewer_enabled is not False

    if adaptive_on:
        adaptive_result = await _try_adaptive_step(
            db,
            session=session,
            current_session_question=current_session_question,
            current_question=current_question,
            answer_text=answer_text,
            audio_object_id=audio_object_id,
            client_turn_key=turn_key,
            language=language,
        )
        if adaptive_result is not None:
            return adaptive_result
        # adaptive declined/failed → answer already recorded; run legacy advance
        # WITHOUT re-inserting the answer (safeguard #2).
        return await _legacy_advance(
            db,
            session=session,
            current_session_question=current_session_question,
            current_question=current_question,
            answer_text=answer_text,
        )

    # ── Pure legacy path (flag off, voice session, or gated out by rollout) ───
    _add_student_answer(db, session_id, current_session_question.id, answer_text, audio_object_id)
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
        )

    return await _legacy_advance(
        db,
        session=session,
        current_session_question=current_session_question,
        current_question=current_question,
        answer_text=answer_text,
    )


def _add_student_answer(
    db: AsyncSession,
    session_id: UUID,
    session_question_id: UUID,
    answer_text: str,
    audio_object_id: UUID | None,
    *,
    turn_key: str | None = None,
) -> None:
    """Add the student-answer message row (does not flush)."""
    metadata: dict[str, Any] = {"kind": "answer"}
    if turn_key is not None:
        metadata["turn_key"] = turn_key
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
            )
        if outcome.result is not None:
            return _project_adaptive_result(outcome.result)
    except Exception:  # noqa: BLE001 — adaptive is best-effort; savepoint rolled back
        logger.exception("adaptive: pipeline failed (session=%s) — legacy fallback", session_id)
    return None


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

    shadow_turn_key = f"shadow:{current_session_question.id}:{uuid4().hex}"
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


async def _legacy_advance(
    db: AsyncSession,
    *,
    session: InterviewSession,
    current_session_question: InterviewSessionQuestion,
    current_question: InterviewQuestion | None,
    answer_text: str,
) -> dict[str, Any]:
    """The exact legacy sequential progression (unchanged behaviour).

    Assumes the student answer has ALREADY been recorded by the caller. Runs the
    single-follow-up stage, then advances to the next approved question or
    signals ``is_finished``.
    """
    followup_text: str | None = None
    if current_question is not None:
        followup_text = await maybe_generate_followup(
            db,
            session=session,
            current_question=current_question,
            student_answer=answer_text,
        )

    if followup_text:
        db.add(
            InterviewSessionMessage(
                session_id=session.id,
                session_question_id=current_session_question.id,
                role="ai",
                content_text=followup_text,
                metadata_json={"kind": "followup"},
            )
        )
        await flush_or_conflict(db)
        return {
            "next_question": None,
            "is_finished": False,
            "followup_text": followup_text,
        }

    asked_ids = await _asked_question_ids(db, session.id)
    next_question = await _next_published_question_after(db, session.interview_config_id, asked_ids)
    if next_question is None:
        return {
            "next_question": None,
            "is_finished": True,
            "followup_text": None,
        }

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
        "followup_text": None,
    }


async def submit_session(
    db: AsyncSession,
    session_id: UUID,
    actor: CurrentUser,
    *,
    arq_pool: object | None = None,
) -> InterviewSession:
    """Mark session ``completed`` + enqueue evaluation.

    Commits inline so the ARQ worker sees the terminal status before
    the evaluation job dequeues. Tolerates ``arq_pool=None`` (test
    mode); production routers inject the real pool via Depends.
    """
    session = await _require_session(db, session_id)
    _assert_owns_session(session, actor)
    if session.status != "in_progress":
        return session

    session.status = "completed"
    session.ended_at = utcnow()
    await db.commit()
    await db.refresh(session)

    if arq_pool is not None:
        await arq_pool.enqueue_job(  # type: ignore[attr-defined]
            _EVALUATE_INTERVIEW_SESSION_TASK, actor.user_id, session.id
        )
    return session


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
) -> list[InterviewSession]:
    return await sessions_queries.get_user_interview_sessions(db, user_id, status=status)


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
    "get_session_for_user",
    "get_user_sessions",
    "start_session",
    "submit_session",
    "take_session_step",
]
