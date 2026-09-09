"""Interview evaluation + gap-report orchestrator (T6.11).

ARQ entrypoint for the ``evaluate_interview_session_task`` job.
Composes:

* T6.8 :func:`evaluate_session` → :class:`RubricScores` (per-criterion
  + aggregated total).
* T6.9 :func:`generate_gap_report` → :class:`GapReportDraft`
  (theory/practice discrepancy + study plan).

Then persists :class:`InterviewOutcomeEvaluation` rows, updates the
session's ``internal_summary_json`` (canonical home for ``total_score``
+ ``rubric_aggregated`` per the baseline schema — there is no separate
``total_score`` column), inserts the :class:`GapReport` row, and
commits.

Failure path: on exception, the transaction is rolled back, the
session row is re-fetched, ``internal_summary_json['evaluation_failure']``
is stamped with the message, the failure is committed, and the
exception is re-raised so ARQ records the job-level failure for retry.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.ai.models import GenerationRun
from abridgeai.core.db.conflict_mapper import register_conflict_mappings
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.observability import get_logger
from abridgeai.core.security import utcnow
from abridgeai.features.interviews.ai.stages.evaluation import evaluate_outcomes, evaluate_session
from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    resolve_rubric_definition,
)
from abridgeai.features.interviews.ai.stages.gap_report import (
    generate_gap_report,
)
from abridgeai.features.interviews.ai.stages.persona_adherence import (
    audit_persona_adherence,
)
from abridgeai.features.interviews.models import (
    InterviewSession,
)
from abridgeai.features.interviews.orchestrator.interviewer_identity import (
    identity_from_config,
)
from abridgeai.features.interviews.orchestrator.persona import profile_from_config
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.queries import sessions as sessions_queries
from abridgeai.features.interviews.services.evaluation_claim import (
    lease_expiry_for,
    new_claim_token,
)
from abridgeai.features.interviews.services.gap_report_writer import persist_gap_report
from abridgeai.features.quizzes.api import public as quizzes_public

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)

# The gap-report writer is read-then-insert on a re-runnable path, and migration
# 0107 made ``source_interview_session_id`` unique where NOT NULL. Register the
# constraint so a losing racer surfaces as a ConflictError instead of a raw
# IntegrityError/HTTP 500. Both spellings are registered because PostgreSQL may
# report either the declarative name or the index name.
register_conflict_mappings(
    {
        "uq_gap_reports_source_interview_session": (
            "interview_gap_report_exists: this interview session already has a gap report"
        ),
        "gap_reports_source_interview_session_id_key": (
            "interview_gap_report_exists: this interview session already has a gap report"
        ),
    }
)


def _ungradeable_reason(session: object) -> str | None:
    """Why this run may not be graded, or None when it may.

    A single refusal: ``evaluate_and_generate_report`` is the ONLY writer of
    ``pass_verdict``, so a missed guard at any enqueue site must not be able to
    reach that write.

    ``"never_started_assessment"`` — still in onboarding (identity check /
    audio check / readiness), so there are no answers to judge: answering is
    gated on ``onboarding_stage == 'completed'`` in ``taking.record_answer``,
    making ``assessment_started_at IS NULL`` equivalent to "no answer can
    exist". Grading one fabricated outcome verdicts and a pass/fail from
    onboarding chatter alone, against a student who never saw a question, and
    consumed one of their attempts.
    """
    if getattr(session, "assessment_started_at", None) is None:
        return "never_started_assessment"
    return None


async def evaluate_and_generate_report(
    db: AsyncSession, session_id: UUID, *, is_final_attempt: bool = False
) -> None:
    """Run evaluation + gap-report stages, persist results, commit.

    Side effects (all in a single transaction):

    1. ``InterviewOutcomeEvaluation`` rows — one per outcome with its OWN
       met/not-met verdict + reasoning + evidence (thesis §4.3), NOT a copied
       session-total.
    2. ``InterviewSession.pass_verdict`` — derived from
       ``met_count >= min_outcomes_to_pass`` (NULL threshold → all outcomes
       must be met). ``internal_summary_json`` also gains the rubric
       ``total_score`` / ``rubric_aggregated`` for teacher diagnostics.
    3. ``GapReport`` row — student / teacher summary + ``report_json``.

    Parameters
    ----------
    is_final_attempt
        True when the caller (ARQ task wrapper) has exhausted
        ``WorkerSettings.max_tries`` on this job. When an exception hits
        on the final attempt, ``InterviewSession.status`` is stamped
        ``'failed'`` in addition to the ``evaluation_failure`` note so the
        student-facing poll (``course-interview.tsx``) can detect the
        terminal failure and stop waiting instead of polling forever
        for a ``pass_verdict`` that will never arrive.

    On exception: rollback, stamp ``internal_summary_json['evaluation_failure']``
    (plus ``status='failed'`` when ``is_final_attempt``), commit, and re-raise.
    """
    session = await sessions_queries.get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Interview session {session_id} not found")

    ungradeable = _ungradeable_reason(session)
    if ungradeable is not None:
        # See _ungradeable_reason: practice rehearsals and runs that never
        # reached the assessment are refused here, not only at the enqueue sites.
        _logger.info(
            "interview.evaluation.skipped",
            extra={"session_id": str(session_id), "reason": ungradeable},
        )
        return None

    # Exclusive claim BEFORE any judging. A published verdict or a live claim
    # held by another job both refuse here, in ONE atomic conditional UPDATE —
    # reading ``pass_verdict`` and deciding in Python would leave the whole
    # grading pass (1-2 min of LLM calls) as a TOCTOU window, which is how two
    # jobs used to run concurrently and race to publish. The recovery sweep makes
    # that reachable by design: it enqueues under a per-attempt job ID (ARQ
    # cannot dedupe it) while the original job may still be running.
    claim_token = new_claim_token()
    claimed_at = utcnow()
    if not await sessions_queries.claim_session_evaluation(
        db,
        session_id,
        token=claim_token,
        now=claimed_at,
        lease_expires_at=lease_expiry_for(claimed_at),
    ):
        _logger.info(
            "interview.evaluation.not_claimed",
            extra={"session_id": str(session_id), "pass_verdict": session.pass_verdict},
        )
        return None

    # The claim was a core UPDATE, so the ORM row we loaded above still shows the
    # pre-claim values (the sessionmaker uses expire_on_commit=False). Sync them
    # by hand: without this, clearing the claim at publish time would be a
    # None -> None no-op, the ORM would emit no UPDATE for those columns, and the
    # row would keep a lease for work that had already finished.
    session.evaluation_claim_token = claim_token
    session.evaluation_claim_expires_at = lease_expiry_for(claimed_at)

    # Shutdown cancellation is NOT an evaluation failure. A worker being
    # stopped (deploy, OOM guard, restart) delivers CancelledError to this
    # task; treating it like an exception used to stamp the session
    # `evaluation_failure` (and `status='failed'` on the final attempt) for
    # work that never failed — the recovery sweep would then see a "failed"
    # row and the student an error for a verdict that was never attempted.
    # Roll back, release the claim (scoped to our token, bounded) and
    # re-raise the cancellation untouched.
    try:
        return await _evaluate_claimed(
            db,
            session,
            session_id=session_id,
            claim_token=claim_token,
            claimed_at=claimed_at,
            is_final_attempt=is_final_attempt,
        )
    except asyncio.CancelledError:
        await db.rollback()
        await _release_claim_on_cancellation(
            db,
            session_id=session_id,
            claim_token=claim_token,
        )
        raise


async def _release_claim_on_cancellation(
    db: AsyncSession, *, session_id: UUID, claim_token: UUID
) -> None:
    """Release our claim in a FRESH session, bounded and shielded.

    The cancellation may have arrived mid-transaction on ``db``; the claim
    release must not inherit that state, and it must not be interrupted
    itself — a shielded, bounded cleanup on a new connection is the last
    chance to make the session recoverable before the process dies.
    """
    from abridgeai.core.db import get_sessionmaker  # noqa: PLC0415

    try:
        sessionmaker = get_sessionmaker()

        async def _release() -> None:
            async with sessionmaker() as fresh:
                await sessions_queries.release_session_evaluation_claim(
                    fresh, session_id, token=claim_token
                )

        await asyncio.wait_for(asyncio.shield(_release()), timeout=10.0)
        _logger.info(
            "interview.evaluation.claim_released_cancelled",
            extra={"session_id": str(session_id)},
        )
    except Exception:  # noqa: BLE001 -- cleanup is best-effort; the lease lapses anyway
        _logger.warning(
            "interview.evaluation.claim_release_on_cancel_failed",
            extra={"session_id": str(session_id)},
        )


async def _evaluate_claimed(
    db: AsyncSession,
    session: InterviewSession,
    *,
    session_id: UUID,
    claim_token: UUID,
    claimed_at: datetime,
    is_final_attempt: bool,
) -> None:
    """The grading pipeline proper. Caller owns claim bookkeeping."""
    # Parent generation_run for this evaluation so it surfaces on the admin
    # processing dashboard and its LLM calls attribute to a pipeline run.
    # Created + committed with status='running' BEFORE the stages so a later
    # failure (and rollback) still leaves a visible 'failed' run row. Declared
    # here so the except handler can stamp it even if creation itself raised.
    eval_run_id: UUID | None = None
    try:
        outcomes = await authoring_queries.list_outcomes_for_config(db, session.interview_config_id)
        all_questions = await authoring_queries.list_questions_for_config(
            db,
            session.interview_config_id,
        )
        candidate_answers = await _list_candidate_answers(db, session_id)
        snapshot = await sessions_queries.get_session_with_responses(db, session_id)
        if snapshot is None:
            raise NotFoundError(f"Interview session {session_id} not found")
        _, session_questions, _ = snapshot
        (
            questions,
            question_prompts,
            expected_question_ids,
            answered_question_ids,
        ) = _build_question_evaluation_context(
            all_questions,
            session_questions,
            candidate_answers,
        )

        config = await authoring_queries.get_interview_for_authoring(
            db, session.interview_config_id
        )
        if config is None:
            raise NotFoundError(f"Interview config {session.interview_config_id} not found")

        # Create the parent generation_run (status='running') and commit it on
        # its own so it is durably visible on the admin processing dashboard
        # even if a stage below fails and rolls back the evaluation work.
        eval_run = await quizzes_public.create_generation_run(
            db,
            kind="interview_evaluation",
            source_scope_kind="module" if config.module_id is not None else "course",
            course_id=config.course_id,
            module_id=config.module_id,
            requested_by=session.student_id,
            config_json={
                "interview_config_id": str(session.interview_config_id),
                "interview_session_id": str(session_id),
            },
        )
        eval_run_id = eval_run.id
        run_row = await db.get(GenerationRun, eval_run_id)
        if run_row is not None:
            run_row.status = "running"
            run_row.started_at = utcnow()
        await db.commit()

        # Thesis §4.3 gate: per-outcome met/not-met verdicts decide pass/fail.
        outcome_verdicts = await evaluate_outcomes(
            db,
            session=session,
            outcomes=outcomes,
            questions=questions,
            answers=candidate_answers,
            question_prompts=question_prompts,
            pipeline_run_id=eval_run_id,
        )
        outcome_verdicts = _fail_unanswered_outcomes(
            outcome_verdicts,
            questions=questions,
            answered_question_ids=answered_question_ids,
        )

        # Rubric stays as a teacher-facing diagnostic feeding the Gap Report;
        # it no longer gates pass/fail (phase-03).
        #
        # Resolve the teacher's scoring rubric from the config's
        # ``supplementary_instructions``. Before this was wired up, no caller
        # passed a rubric at all, so EVERY session was silently graded against
        # the four-criterion equal-weight default and a teacher-authored rubric
        # had no effect. Malformed / prose-only fields still fall back to that
        # default, so grading cannot break on a bad config.
        rubric = resolve_rubric_definition(getattr(config, "supplementary_instructions", None))
        rubric_scores = await evaluate_session(
            db,
            session=session,
            outcomes=outcomes,
            questions=questions,
            answers=candidate_answers,
            rubric=rubric,
            question_prompts=question_prompts,
            expected_question_ids=expected_question_ids,
            pipeline_run_id=eval_run_id,
        )

        min_outcomes_to_pass = getattr(config, "min_outcomes_to_pass", None)
        course_id, module_id = config.course_id, config.module_id

        quiz_attempts = await _load_student_quiz_attempts(
            db, student_id=session.student_id, course_id=course_id, module_id=module_id
        )

        report_draft = await generate_gap_report(
            db,
            session=session,
            rubric_scores=rubric_scores,
            quiz_attempts=quiz_attempts,
            course_id=course_id,
            module_id=module_id,
            pipeline_run_id=eval_run_id,
        )

        # Tone-only diagnostic (never gates pass/fail): did the AI interviewer
        # hold the configured persona? Runs over the stored transcript. It is
        # best-effort — audit_persona_adherence never raises, returning an
        # unavailable() sentinel on no-turns / LLM failure — so a tone audit
        # problem can never block a student's evaluation from completing.
        transcript = await sessions_queries.list_session_messages(db, session_id)
        persona_adherence = await audit_persona_adherence(
            db,
            persona=profile_from_config(
                getattr(config, "persona", None),
                getattr(config, "persona_profile_json", None),
            ),
            messages=transcript,
            # The judge must know WHICH interviewer was declared, or it reads a
            # role's register (concrete vs. trade-off-oriented wording) as tone
            # drift and flags a config that behaved exactly as intended.
            identity=identity_from_config(getattr(config, "persona_profile_json", None)),
            language=getattr(session, "interview_language", None),
            pipeline_run_id=eval_run_id,
        )

        # Judging is done; publishing starts here. Re-check ownership first: if
        # our lease lapsed mid-run another job may legitimately have taken over,
        # and a stale owner must not publish over it. ``FOR UPDATE`` holds the
        # row until the commit below, so the check cannot be overtaken.
        if not await sessions_queries.holds_session_evaluation_claim(
            db, session_id, token=claim_token, now=utcnow()
        ):
            _logger.warning(
                "interview.evaluation.publish_skipped_lease_lost",
                extra={"session_id": str(session_id)},
            )
            await db.rollback()
            return

        await _persist_outcome_evaluations(db, session_id=session_id, verdicts=outcome_verdicts)
        _stamp_session_summary(
            session,
            rubric_scores=rubric_scores,
            verdicts=outcome_verdicts,
            min_outcomes_to_pass=min_outcomes_to_pass,
            question_count=len(expected_question_ids),
            answered_question_count=len(answered_question_ids),
            persona_adherence=persona_adherence,
        )
        await persist_gap_report(
            db,
            session=session,
            course_id=course_id,
            module_id=module_id,
            draft=report_draft,
        )

        # The claim has done its job — the verdict below is what refuses any
        # further grading from now on. Cleared in the SAME transaction as the
        # verdict so the row can never be left holding a lease for work that
        # already finished.
        session.evaluation_claim_token = None
        session.evaluation_claim_expires_at = None

        # Mark the evaluation run completed now that all work has committed.
        run_row = await db.get(GenerationRun, eval_run_id)
        if run_row is not None:
            run_row.status = "completed"
            run_row.finished_at = utcnow()
        await db.commit()
    except Exception as exc:
        await db.rollback()
        await _record_evaluation_failure(
            db,
            session_id=session_id,
            eval_run_id=eval_run_id,
            claim_token=claim_token,
            exc=exc,
            is_final_attempt=is_final_attempt,
        )
        raise

    # An interview is a gradeable course unit, completed by a passing verdict, so
    # this is the moment the unit can change state — fire the D2 course-completion
    # writer so the student's last interview unlocks the next career-path stage now
    # instead of at the nightly drift sweep.
    #
    # AFTER the commit, deliberately. It used to run inside the transaction above,
    # which made a side effect able to destroy the verdict it was reacting to:
    # ``sync_course_completion`` flushes through ``flush_or_conflict``, which
    # ROLLS BACK on IntegrityError before raising. That rollback discarded the
    # uncommitted verdict, gap report and outcome rows; the swallow in
    # ``_sync_course_completion`` then let execution continue to ``db.commit()``,
    # which committed an empty transaction. The job logged success and the
    # student's graded interview silently went back to ungraded — and because the
    # claim was cleared in the same discarded transaction, the row still held a
    # 30-minute lease that blocked the recovery sweep from re-driving it.
    await _sync_course_completion(db, student_id=session.student_id, course_id=course_id)


async def _record_evaluation_failure(
    db: AsyncSession,
    *,
    session_id: UUID,
    eval_run_id: UUID | None,
    claim_token: UUID,
    exc: Exception,
    is_final_attempt: bool,
) -> None:
    """Persist the failure trail for a crashed evaluation, then let it re-raise.

    Runs AFTER the caller's rollback, on its own commits, and never masks the
    original exception.

    Writes the session-level trail ONLY while we still own the claim. A job whose
    lease lapsed has been superseded — stamping ``status='failed'`` from it would
    relabel work another job is doing (or has already published), and since the
    recovery query filters on ``pass_verdict IS NULL`` nothing would repair it.
    The ``generation_run`` row is stamped either way: it records THIS run's
    outcome and is not shared.
    """
    # Stamp the evaluation run as failed so the dashboard shows the terminal
    # state even though the evaluation work itself rolled back.
    if eval_run_id is not None:
        failed_run = await db.get(GenerationRun, eval_run_id)
        if failed_run is not None:
            failed_run.status = "failed"
            failed_run.finished_at = utcnow()
            failed_run.config_json = dict(failed_run.config_json or {}) | {
                "failure": {"message": str(exc)}
            }
            await db.commit()

    if not await sessions_queries.holds_session_evaluation_claim(
        db, session_id, token=claim_token, now=utcnow()
    ):
        # Superseded: our lease lapsed and/or another job owns this session now
        # (possibly having already published a verdict). Writing the failure
        # trail from here would relabel someone else's work.
        _logger.warning(
            "interview.evaluation.failure_discarded_not_owner",
            extra={"session_id": str(session_id), "error": str(exc)},
        )
        await db.rollback()
        return

    fresh = await db.get(InterviewSession, session_id)
    if fresh is None:
        await db.rollback()
        return

    fresh.internal_summary_json = dict(fresh.internal_summary_json or {}) | {
        "evaluation_failure": {
            "message": str(exc),
            "failed_at": utcnow().isoformat(),
            "final_attempt": is_final_attempt,
        }
    }
    # Only stamp the terminal 'failed' status once ARQ has exhausted its retry
    # budget. Marking it failed on attempt 1/3 would tell the student the
    # interview is dead while a retry is still queued — but NOT stamping it on
    # the LAST attempt leaves the session stuck at 'completed' with
    # pass_verdict forever null, so the frontend poll never resolves and the
    # student waits indefinitely.
    if is_final_attempt:
        fresh.status = "failed"
    # Release our claim in the same transaction. The grading pass failed, so the
    # next ARQ retry or recovery sweep must be able to claim immediately instead
    # of waiting out a 30-minute lease held by a job that is already dead.
    fresh.evaluation_claim_token = None
    fresh.evaluation_claim_expires_at = None
    await db.commit()


# Persistence helpers live in evaluation_persistence.py (LOC gate);
# re-exported here so existing import sites (and tests patching these names
# on `evaluation`) keep working.
from abridgeai.features.interviews.services.evaluation_persistence import (  # noqa: E402,F401
    _build_question_evaluation_context,
    _derive_pass_verdict,
    _fail_unanswered_outcomes,
    _gradable_answer_text,
    _list_candidate_answers,
    _load_student_quiz_attempts,
    _persist_outcome_evaluations,
    _QuizAttemptRow,
    _stamp_session_summary,
    _sync_course_completion,
)
