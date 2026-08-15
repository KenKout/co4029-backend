"""Regrade service (Phase 1).

Moodle-style dry-run → commit regrade. When a teacher edits a question's
answer key, historical ``QuizAttemptAnswer`` rows keep their old
``is_correct``/``points_awarded`` (and the attempt headline scores stay frozen)
until a regrade runs. This service:

* :func:`compute_regrade` — a **dry run**: re-grades every in-scope stored answer
  against the CURRENT (live) question definition, diffs against the stored
  values, and persists a :class:`QuizRegradeRun` + one :class:`QuizRegradeItem`
  per changed answer. It does NOT mutate attempts.
* :func:`commit_regrade` — applies a dry run's deltas to the live answer rows,
  recomputes each affected attempt's headline score (reusing the same helper as
  ``submit_attempt`` — DRY), optionally reconciles SM-2, and marks the run
  committed. Idempotent-guarded (a committed run cannot be re-committed).

Layering: this service owns its own DB reads/writes (same precedent as
``services/taking.py``), so routers call here rather than touching queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from uuid import UUID

from sqlalchemy import select

from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import AppError, NotFoundError
from abridgeai.core.observability import get_logger
from abridgeai.core.security import utcnow
from abridgeai.features.quizzes.models import (
    Quiz,
    QuizAttempt,
    QuizAttemptAnswer,
    QuizQuestion,
    QuizQuestionOption,
    QuizRegradeItem,
    QuizRegradeRun,
)
from abridgeai.features.quizzes.services.grader import grade_answer_against_revision
from abridgeai.features.quizzes.services.taking import _recompute_attempt_score

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)


async def _require_quiz(db: AsyncSession, quiz_id: UUID) -> Quiz:
    quiz = (
        await db.execute(select(Quiz).where(Quiz.id == quiz_id))
    ).scalar_one_or_none()
    if quiz is None:
        raise NotFoundError(f"Quiz {quiz_id} not found")
    return quiz


async def _snapshot_live_question(
    db: AsyncSession, question: QuizQuestion
) -> dict[str, Any]:
    """Serialize the CURRENT question definition into the dict shape that
    :func:`grade_answer_against_revision` expects (``question_type`` + either
    ``options`` w/ ``option_key``+``is_correct`` for MCQ/TF, or ``correct_answer``
    for short_answer/fill_blank from ``original_generated_payload``).
    """
    payload: dict[str, Any] = {"question_type": question.question_type}
    if question.question_type in {"multiple_choice", "true_false"}:
        options = (
            (
                await db.execute(
                    select(QuizQuestionOption)
                    .where(QuizQuestionOption.question_id == question.id)
                    .order_by(QuizQuestionOption.position)
                )
            )
            .scalars()
            .all()
        )
        payload["options"] = [
            {"option_key": o.option_key, "is_correct": o.is_correct} for o in options
        ]
    else:
        gen = question.original_generated_payload or {}
        payload["correct_answer"] = gen.get("correct_answer")
    return payload


async def _current_revision_id(db: AsyncSession, question_id: UUID) -> UUID | None:
    from abridgeai.features.quizzes.models import QuizQuestionRevision  # noqa: PLC0415

    return (
        await db.execute(
            select(QuizQuestionRevision.id)
            .where(QuizQuestionRevision.question_id == question_id)
            .order_by(QuizQuestionRevision.revision_no.desc())
            .limit(1)
        )
    ).scalar_one_or_none()


async def compute_regrade(
    db: AsyncSession,
    *,
    quiz_id: UUID,
    attempt_ids: list[UUID] | None,
    question_ids: list[UUID] | None,
    requested_by: UUID | None,
) -> QuizRegradeRun:
    """Dry run: persist a :class:`QuizRegradeRun` with one item per changed answer.

    Does NOT mutate attempts. ``attempt_ids`` / ``question_ids`` empty or None =
    whole quiz (all submitted/graded attempts, all approved questions).
    """
    quiz = await _require_quiz(db, quiz_id)

    # In-scope questions (default: all approved for the quiz).
    q_stmt = select(QuizQuestion).where(
        QuizQuestion.quiz_id == quiz.id,
        QuizQuestion.review_status == "approved",
    )
    if question_ids:
        q_stmt = q_stmt.where(QuizQuestion.id.in_(question_ids))
    questions = list((await db.execute(q_stmt)).scalars().all())
    snapshots: dict[UUID, dict[str, Any]] = {
        q.id: await _snapshot_live_question(db, q) for q in questions
    }
    scoped_qids = set(snapshots.keys())

    # In-scope attempts (default: all submitted/graded for the quiz).
    a_stmt = select(QuizAttempt).where(
        QuizAttempt.quiz_id == quiz.id,
        QuizAttempt.status.in_(["submitted", "graded"]),
    )
    if attempt_ids:
        a_stmt = a_stmt.where(QuizAttempt.id.in_(attempt_ids))
    attempts = list((await db.execute(a_stmt)).scalars().all())
    scoped_attempt_ids = [a.id for a in attempts]

    run = QuizRegradeRun(
        quiz_id=quiz.id,
        status="dry_run",
        scope_json={
            "attempt_ids": [str(x) for x in (attempt_ids or [])],
            "question_ids": [str(x) for x in (question_ids or [])],
        },
        requested_by=requested_by,
    )
    db.add(run)
    await flush_or_conflict(db)
    await db.refresh(run)

    if not scoped_attempt_ids or not scoped_qids:
        run.attempts_affected = 0
        run.answers_changed = 0
        await flush_or_conflict(db)
        await db.refresh(run)
        return run

    answers = list(
        (
            await db.execute(
                select(QuizAttemptAnswer).where(
                    QuizAttemptAnswer.attempt_id.in_(scoped_attempt_ids),
                    QuizAttemptAnswer.question_id.in_(scoped_qids),
                )
            )
        )
        .scalars()
        .all()
    )

    # Batch-resolve selected option keys (avoid N+1).
    option_ids = [a.selected_option_id for a in answers if a.selected_option_id]
    option_key_by_id: dict[UUID, str] = {}
    if option_ids:
        for oid, okey in (
            await db.execute(
                select(QuizQuestionOption.id, QuizQuestionOption.option_key).where(
                    QuizQuestionOption.id.in_(option_ids)
                )
            )
        ).all():
            option_key_by_id[oid] = okey

    changed_attempts: set[UUID] = set()
    items_created = 0
    for ans in answers:
        snapshot = snapshots.get(ans.question_id)
        if snapshot is None:
            continue
        selected_key = (
            option_key_by_id.get(ans.selected_option_id)
            if ans.selected_option_id
            else None
        )
        new = grade_answer_against_revision(
            snapshot,
            selected_option_key=selected_key,
            answer_text=ans.answer_text,
        )
        if (
            new.is_correct != ans.is_correct
            or new.points_awarded != ans.points_awarded
        ):
            db.add(
                QuizRegradeItem(
                    run_id=run.id,
                    attempt_id=ans.attempt_id,
                    answer_id=ans.id,
                    question_id=ans.question_id,
                    old_is_correct=ans.is_correct,
                    new_is_correct=new.is_correct,
                    old_points=ans.points_awarded,
                    new_points=new.points_awarded,
                )
            )
            changed_attempts.add(ans.attempt_id)
            items_created += 1

    run.attempts_affected = len(changed_attempts)
    run.answers_changed = items_created
    await flush_or_conflict(db)
    await db.refresh(run)
    return run


async def get_regrade_run(
    db: AsyncSession, *, quiz_id: UUID, run_id: UUID
) -> QuizRegradeRun | None:
    """Load a run with its items eagerly loaded (async — no lazy load), scoped
    to the quiz."""
    from sqlalchemy.orm import selectinload  # noqa: PLC0415

    run = (
        await db.execute(
            select(QuizRegradeRun)
            .where(QuizRegradeRun.id == run_id, QuizRegradeRun.quiz_id == quiz_id)
            .options(selectinload(QuizRegradeRun.items))
        )
    ).scalar_one_or_none()
    return run


async def commit_regrade(
    db: AsyncSession,
    *,
    quiz_id: UUID,
    run_id: UUID,
    reconcile_sr: bool = False,
) -> QuizRegradeRun:
    """Apply a dry run's deltas, recompute affected attempt scores, mark committed.

    Guards: the run must exist for the quiz and be in ``dry_run`` status (else
    :class:`AppError` → 409 upstream). Transactional via ``flush_or_conflict``.
    """
    quiz = await _require_quiz(db, quiz_id)
    run = await get_regrade_run(db, quiz_id=quiz_id, run_id=run_id)
    if run is None:
        raise NotFoundError(f"Regrade run {run_id} not found")
    if run.status != "dry_run":
        raise AppError(f"Regrade run {run_id} is not in dry_run status")

    items = list(
        (
            await db.execute(
                select(QuizRegradeItem).where(QuizRegradeItem.run_id == run.id)
            )
        )
        .scalars()
        .all()
    )
    if not items:
        run.status = "committed"
        run.committed_at = utcnow()
        await flush_or_conflict(db)
        await db.refresh(run)
        return run

    # Apply per-answer deltas.
    answer_ids = [it.answer_id for it in items]
    answers_by_id = {
        a.id: a
        for a in (
            await db.execute(
                select(QuizAttemptAnswer).where(QuizAttemptAnswer.id.in_(answer_ids))
            )
        )
        .scalars()
        .all()
    }
    affected_attempt_ids: set[UUID] = set()
    for it in items:
        ans = answers_by_id.get(it.answer_id)
        if ans is None:
            continue
        ans.is_correct = it.new_is_correct
        ans.points_awarded = it.new_points
        ans.graded_revision_id = await _current_revision_id(db, it.question_id)
        affected_attempt_ids.add(it.attempt_id)

    await flush_or_conflict(db)

    # Recompute each affected attempt's headline score (shared helper — DRY).
    attempts = list(
        (
            await db.execute(
                select(QuizAttempt).where(QuizAttempt.id.in_(affected_attempt_ids))
            )
        )
        .scalars()
        .all()
    )
    for attempt in attempts:
        score_points, score_percent, _correct, _count = await _recompute_attempt_score(
            db, attempt, quiz
        )
        attempt.score_points = score_points
        attempt.score_percent = score_percent
        attempt.passed = score_percent >= quiz.passing_score_percent

    # Phase 9: refresh the materialised grade-of-record for each affected student
    # (dedup — grade is per student, not per attempt).
    from abridgeai.features.quizzes.services.gradebook import (  # noqa: PLC0415
        recompute_final_grade,
    )

    for student_id in {a.student_id for a in attempts}:
        await recompute_final_grade(db, quiz, student_id)

    if reconcile_sr:
        # SM-2 reconciliation is deferred behind a flag (v1 default off). Re-firing
        # a card review here risks double-counting; we log intent instead so the
        # grade fix is never blocked by SR. A future task wires a real
        # reconcile_card_review() in features/spaced_repetition.
        _logger.warning(
            "sr_regrade_reconcile_requested",
            run_id=str(run.id),
            changed_answers=len(items),
        )

    run.status = "committed"
    run.committed_at = utcnow()
    await flush_or_conflict(db)

    # Phase 13: append a correctness-bearing audit event in the same transaction.
    from abridgeai.features.quizzes.services.audit import record_event  # noqa: PLC0415

    await record_event(
        db,
        event_name="attempt_regraded",
        quiz_id=quiz_id,
        payload={
            "run_id": str(run.id),
            "answers_changed": run.answers_changed,
            "attempts_affected": run.attempts_affected,
        },
    )

    await db.refresh(run)
    return run


__all__ = ["commit_regrade", "compute_regrade", "get_regrade_run"]
