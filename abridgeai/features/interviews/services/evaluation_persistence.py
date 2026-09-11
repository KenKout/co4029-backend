"""Persistence + summary helpers for the interview evaluation pipeline.

Split from :mod:`evaluation` (the LOC gate). These helpers own everything that
TURNS a judged evaluation into durable rows: candidate answers from the
transcript, per-outcome evaluation rows, the pass verdict, the summary stamp
and the course-completion sync. The main pipeline in :mod:`evaluation` calls
into these; nothing else should write the same columns.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.security import utcnow
from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
    OutcomeVerdict,
    OutcomeVerdicts,
    build_outcome_verdicts,
)
from abridgeai.features.interviews.ai.stages.evaluation.rubric import RubricScores
from abridgeai.features.interviews.ai.stages.persona_adherence.parsers import PersonaAdherence
from abridgeai.features.interviews.models import (
    InterviewOutcomeEvaluation,
    InterviewQuestion,
    InterviewSession,
    InterviewSessionMessage,
    InterviewSessionQuestion,
)
from abridgeai.features.interviews.queries import sessions as sessions_queries

_logger = logging.getLogger("abridgeai.features.interviews.services.evaluation_persistence")


async def _list_candidate_answers(
    db: AsyncSession, session_id: UUID
) -> list[InterviewSessionMessage]:
    """The gradeable user turns for one session.

    Single predicate chain: user row, linked to a session question, not
    onboarding, and — for typed-turn receipts — fully APPLIED. A receipt still
    in ``received`` (fold in flight at finish) or ``failed`` (fold raised) is
    NOT evidence and must not enter the rubric/outcome prompts; ordinary
    REST/voice user rows have no receipt marker and keep the old behavior.
    The receipt state check lives in the shared evaluation predicate
    (:func:`abridgeai.features.interviews.ai.stages.evaluation.logic.
    _is_candidate_answer`) so direct/internal callers cannot bypass it.
    """
    from abridgeai.features.interviews.ai.stages.evaluation.logic import (  # noqa: PLC0415
        _is_candidate_answer,
    )

    messages = await sessions_queries.list_session_messages(db, session_id)
    return [
        m
        for m in messages
        if _is_candidate_answer(m)
        and getattr(m, "session_question_id", None) is not None
        and (getattr(m, "metadata_json", None) or {}).get("kind") != "onboarding"
    ]


def _build_question_evaluation_context(
    all_questions: list[InterviewQuestion],
    session_questions: list[InterviewSessionQuestion],
    candidate_answers: list[InterviewSessionMessage],
) -> tuple[list[InterviewQuestion], dict[UUID, str], list[UUID], set[UUID]]:
    """Resolve answer FKs and the complete gradeable question set.

    Message rows reference ``InterviewSessionQuestion.id``, not
    ``InterviewQuestion.id``. The old evaluator indexed prompts by the latter,
    so production judge prompts silently had an empty question. This mapping
    also includes every approved but unasked question in the score denominator
    when a candidate ends the interview early.
    """
    question_by_id = {question.id: question for question in all_questions}
    asked_config_ids = {
        asked.interview_question_id
        for asked in session_questions
        if asked.interview_question_id is not None
    }
    questions = [
        question
        for question in all_questions
        if question.review_status == "approved" or question.id in asked_config_ids
    ]
    gradeable_config_ids = {question.id for question in questions}

    prompts: dict[UUID, str] = {}
    first_session_id_by_question: dict[UUID, UUID] = {}
    config_id_by_session_id: dict[UUID, UUID] = {}
    for asked in session_questions:
        config_question_id = asked.interview_question_id
        if config_question_id is None or config_question_id not in gradeable_config_ids:
            continue
        question = question_by_id.get(config_question_id)
        if question is None:
            continue
        prompts[asked.id] = question.prompt_text
        config_id_by_session_id[asked.id] = config_question_id
        first_session_id_by_question.setdefault(config_question_id, asked.id)

    answered_session_ids = {
        answer.session_question_id
        for answer in candidate_answers
        if answer.session_question_id in config_id_by_session_id and _gradable_answer_text(answer)
    }
    answered_session_id_by_question: dict[UUID, UUID] = {}
    for asked in session_questions:
        if asked.id in answered_session_ids and asked.interview_question_id is not None:
            answered_session_id_by_question.setdefault(asked.interview_question_id, asked.id)

    expected_question_ids = [
        answered_session_id_by_question.get(
            question.id,
            first_session_id_by_question.get(question.id, question.id),
        )
        for question in questions
    ]
    answered_question_ids = {
        config_id_by_session_id[answer.session_question_id]
        for answer in candidate_answers
        if answer.session_question_id in config_id_by_session_id and _gradable_answer_text(answer)
    }
    return questions, prompts, expected_question_ids, answered_question_ids


def _gradable_answer_text(message: InterviewSessionMessage) -> str:
    """Mirror the evaluation-stage evidence filter for answer counting."""
    metadata = message.metadata_json or {}
    if metadata.get("kind") in {
        "security",
        "turn_control",
        "clarification",
        "term_explanation",
        "hint",
        "end_request",
    }:
        safe = metadata.get("safe_academic_text")
        return safe.strip() if isinstance(safe, str) else ""
    return (message.content_text or "").strip()


def _fail_unanswered_outcomes(
    verdicts: OutcomeVerdicts,
    *,
    questions: list[InterviewQuestion],
    answered_question_ids: set[UUID],
) -> OutcomeVerdicts:
    """Make outcomes with no submitted linked answer deterministically fail."""
    question_ids_by_outcome: dict[UUID, set[UUID]] = {}
    for question in questions:
        if question.linked_outcome_id is not None:
            question_ids_by_outcome.setdefault(question.linked_outcome_id, set()).add(question.id)

    adjusted: list[OutcomeVerdict] = []
    for verdict in verdicts.verdicts:
        linked_question_ids = question_ids_by_outcome.get(verdict.outcome_id, set())
        if linked_question_ids and linked_question_ids.isdisjoint(answered_question_ids):
            adjusted.append(
                OutcomeVerdict(
                    outcome_id=verdict.outcome_id,
                    met=False,
                    reasoning="No answer was submitted for questions linked to this outcome.",
                    evidence=None,
                )
            )
        else:
            adjusted.append(verdict)
    return build_outcome_verdicts(adjusted)


async def _load_student_quiz_attempts(
    db: AsyncSession,
    *,
    student_id: UUID,
    course_id: UUID,
    module_id: UUID | None,
) -> list[Any]:
    """Pull quiz_attempt scores via raw SQL.

    Cross-feature read into the ``quiz_attempts`` table — direct ORM
    import would break the ``Features are independent`` import-linter
    contract. The Gap Report stage consumes objects with a
    ``score_percent`` attribute (via :class:`_QuizAttemptLike`
    Protocol); raw rows fit that shape.
    """
    from sqlalchemy import text  # noqa: PLC0415

    sql = (
        "SELECT qa.score_percent FROM quiz_attempts qa "
        "JOIN quizzes q ON q.id = qa.quiz_id "
        "WHERE qa.student_id = :student_id "
        "  AND q.course_id = :course_id "
    )
    params: dict[str, Any] = {"student_id": student_id, "course_id": course_id}
    if module_id is not None:
        sql += "  AND q.module_id = :module_id "
        params["module_id"] = module_id

    rows = (await db.execute(text(sql), params)).mappings().all()
    return [_QuizAttemptRow(score_percent=row["score_percent"]) for row in rows]


class _QuizAttemptRow:
    """Lightweight stand-in for the Gap Report's ``_QuizAttemptLike`` Protocol."""

    __slots__ = ("score_percent",)

    def __init__(self, *, score_percent: Decimal | float | int | None) -> None:
        self.score_percent = score_percent


async def _persist_outcome_evaluations(
    db: AsyncSession,
    *,
    session_id: UUID,
    verdicts: OutcomeVerdicts,
) -> None:
    """Insert one :class:`InterviewOutcomeEvaluation` per outcome verdict.

    Each row carries that outcome's OWN met/not-met verdict, hidden reasoning,
    and evidence excerpt — the genuine per-outcome judgement from the §4.3
    verdict stage (not a copied session-total).

    Written as an idempotent upsert because this is now a RE-RUNNABLE path. The
    table has a unique ``(session_id, outcome_id)``, and an evaluation can fail
    *after* these rows commit (the gap-report stage is downstream). A plain
    INSERT would then make every recovery attempt die on that constraint, so the
    session could never be graded — the failure would be permanent and silent.
    ON CONFLICT DO UPDATE lets the newest verdict win instead.
    """
    if not verdicts.verdicts:
        return
    from sqlalchemy.dialects.postgresql import insert as pg_insert  # noqa: PLC0415

    stmt = pg_insert(InterviewOutcomeEvaluation).values(
        [
            {
                "session_id": session_id,
                "outcome_id": verdict.outcome_id,
                "verdict_met": verdict.met,
                "hidden_reasoning": verdict.reasoning,
                "evidence_excerpt": verdict.evidence,
            }
            for verdict in verdicts.verdicts
        ]
    )
    await db.execute(
        stmt.on_conflict_do_update(
            index_elements=["session_id", "outcome_id"],
            set_={
                "verdict_met": stmt.excluded.verdict_met,
                "hidden_reasoning": stmt.excluded.hidden_reasoning,
                "evidence_excerpt": stmt.excluded.evidence_excerpt,
            },
        )
    )


def _derive_pass_verdict(verdicts: OutcomeVerdicts, min_outcomes_to_pass: int | None) -> bool:
    """Pass when enough outcomes are met (thesis §4.3).

    ``min_outcomes_to_pass`` is the teacher-configured threshold. When it is
    NULL/unset we require EVERY outcome to be met — the documented-safe
    default (a teacher who configured no threshold has not opted into a
    partial pass). A session with no outcomes cannot pass.
    """
    if verdicts.total == 0:
        return False
    threshold = min_outcomes_to_pass if min_outcomes_to_pass is not None else verdicts.total
    return verdicts.met_count >= threshold


def _stamp_session_summary(
    session: InterviewSession,
    *,
    rubric_scores: RubricScores,
    verdicts: OutcomeVerdicts,
    min_outcomes_to_pass: int | None,
    question_count: int,
    answered_question_count: int,
    persona_adherence: PersonaAdherence | None = None,
) -> None:
    summary: dict[str, Any] = dict(session.internal_summary_json or {})
    summary.pop("evaluation_failure", None)
    summary["total_score"] = float(rubric_scores.total_score)
    summary["rubric_aggregated"] = dict(rubric_scores.aggregated)
    summary["outcomes_met"] = verdicts.met_count
    summary["outcomes_total"] = verdicts.total
    summary["min_outcomes_to_pass"] = min_outcomes_to_pass
    summary["questions_total"] = question_count
    summary["questions_answered"] = answered_question_count
    summary["questions_unanswered"] = max(0, question_count - answered_question_count)
    summary["evaluated_at"] = utcnow().isoformat()
    # Teacher-only tone diagnostic. Only stored when the audit produced
    # something usable — an unavailable() sentinel (no interviewer turns / LLM
    # down) is not persisted, so the teacher UI can tell "audited" from "not".
    if persona_adherence is not None and persona_adherence.available:
        summary["persona_adherence"] = persona_adherence.to_json()
    session.internal_summary_json = summary
    session.pass_verdict = _derive_pass_verdict(verdicts, min_outcomes_to_pass)
    # A recovered run must leave the failure state behind. ``status='failed'``
    # says only "the grader never finished" — it is stamped by the ARQ wrapper
    # when the retry budget runs out, so the student-facing poll can stop
    # waiting. Now that the recovery sweep re-drives those rows, keeping the
    # status after a verdict has been written would show an error for an
    # interview that is in fact graded, and would hide the row from every reader
    # that filters on a terminal status. ``completed`` is the honest label: the
    # session ran through to a verdict. ``timed_out`` is deliberately NOT
    # rewritten — that one is a real assessment outcome, not a grader fault.
    # (The stale ``evaluation_failure`` note is already dropped at the top of
    # this function, so only the status needs correcting here.)
    if session.status == "failed":
        session.status = "completed"


async def _sync_course_completion(db: AsyncSession, *, student_id: UUID, course_id: UUID) -> None:
    """Fire the D2 course-completion writer for this interview's course.

    Course completion counts interview units, and an interview unit is done
    only when an attempt has ``pass_verdict = TRUE``. Since
    ``course_enrollments.status`` is what career-path stage unlock reads as
    ``satisfied``, the verdict write above has to be followed by this call or
    passing the last interview leaves the next stage locked until the nightly
    drift sweep.

    Runs in its OWN transaction, after the verdict has been committed, and owns
    the commit. Sharing the verdict's transaction let this side effect destroy the
    verdict: ``sync_course_completion`` flushes via ``flush_or_conflict``, which
    rolls back before raising, so a conflict here discarded the grading work that
    had not been committed yet — and the swallow below hid it.

    Never raises into the caller: an evaluation that produced a real verdict must
    not be reported as failed because a completion side-effect was. The nightly
    sweeper (``enrollments...resync_stale_course_completions``) repairs any miss —
    the same contract ``progress.services.tracking`` uses.
    """
    from abridgeai.features.enrollments.api import public as enrollments_api  # noqa: PLC0415

    try:
        await enrollments_api.sync_course_completion(db, course_id=course_id, student_id=student_id)
        await db.commit()
    except Exception:  # noqa: BLE001 -- side-effect; nightly sweeper repairs drift
        await db.rollback()
        _logger.warning(
            "interview.course_completion_sync_failed",
            exc_info=True,
            extra={"course_id": str(course_id), "student_id": str(student_id)},
        )
