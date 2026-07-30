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

from decimal import Decimal
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.models import GenerationRun
from abridgeai.core.db.conflict_mapper import flush_or_conflict
from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.observability import get_logger
from abridgeai.core.security import utcnow
from abridgeai.features.interviews import practice
from abridgeai.features.interviews.ai.stages.evaluation import evaluate_outcomes, evaluate_session
from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
    OutcomeVerdict,
    OutcomeVerdicts,
    build_outcome_verdicts,
)
from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    RubricScores,
    resolve_rubric_definition,
)
from abridgeai.features.interviews.ai.stages.gap_report import (
    GapReportDraft,
    generate_gap_report,
)
from abridgeai.features.interviews.ai.stages.persona_adherence import (
    audit_persona_adherence,
)
from abridgeai.features.interviews.ai.stages.persona_adherence.parsers import (
    PersonaAdherence,
)
from abridgeai.features.interviews.models import (
    GapReport,
    InterviewOutcomeEvaluation,
    InterviewQuestion,
    InterviewSession,
    InterviewSessionMessage,
    InterviewSessionQuestion,
)
from abridgeai.features.interviews.orchestrator.interviewer_identity import (
    identity_from_config,
)
from abridgeai.features.interviews.orchestrator.persona import profile_from_config
from abridgeai.features.interviews.orchestrator.security import (
    SecurityAction,
    SecurityAssessment,
    SecurityCategory,
)
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.queries import sessions as sessions_queries
from abridgeai.features.interviews.services import security as security_service
from abridgeai.features.quizzes.api import public as quizzes_public

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

_logger = get_logger(__name__)


def _ungradeable_reason(session: object) -> str | None:
    """Why this run may not be graded, or None when it may.

    Two refusals, both defence in depth against the same class of damage:
    ``evaluate_and_generate_report`` is the ONLY writer of ``pass_verdict``, so a
    missed guard at any enqueue site must not be able to reach that write.

    * ``"practice"`` — a rehearsal. A verdict on it would unlock the SR lesson
      and quiz gates that read ``pass_verdict``.
    * ``"never_started_assessment"`` — still in onboarding (identity check /
      audio check / readiness), so there are no answers to judge: answering is
      gated on ``onboarding_stage == 'completed'`` in ``taking.record_answer``,
      making ``assessment_started_at IS NULL`` equivalent to "no answer can
      exist". Grading one fabricated outcome verdicts and a pass/fail from
      onboarding chatter alone, against a student who never saw a question, and
      consumed one of their attempts.
    """
    if practice.is_practice(getattr(session, "session_mode", None)):
        return "practice"
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
        return

    # Parent generation_run for this evaluation so it surfaces on the admin
    # processing dashboard and its LLM calls attribute to a pipeline run.
    # Created + committed with status='running' BEFORE the stages so a later
    # failure (and rollback) still leaves a visible 'failed' run row. Declared
    # here so the except handler can stamp it even if creation itself raised.
    eval_run_id: UUID | None = None

    try:
        outcomes = await authoring_queries.list_outcomes_for_config(db, session.interview_config_id)
        # Partitioned by session mode, and this is the highest-consequence of the
        # partition filters — it is not about leaking questions. This query takes
        # every review state, and the questions it returns become
        # ``expected_question_ids``: a question from the other partition would be
        # counted as unanswered, inflate ``questions_total``, and hard-fail any
        # outcome linked to it via ``_fail_unanswered_outcomes``. Unfiltered, the
        # partition would corrupt real grades rather than merely reveal a bank.
        all_questions = await authoring_queries.list_questions_for_config(
            db,
            session.interview_config_id,
            practice_only=practice.partition_for_mode(session.session_mode),
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
        await _persist_gap_report(
            db,
            session=session,
            course_id=course_id,
            module_id=module_id,
            draft=report_draft,
        )

        # Mark the evaluation run completed now that all work has committed.
        run_row = await db.get(GenerationRun, eval_run_id)
        if run_row is not None:
            run_row.status = "completed"
            run_row.finished_at = utcnow()
            await db.commit()
    except Exception as exc:
        await db.rollback()
        # Stamp the evaluation run as failed so the dashboard shows the
        # terminal state. This rides its own commit after the rollback above
        # and never masks the original exception.
        if eval_run_id is not None:
            failed_run = await db.get(GenerationRun, eval_run_id)
            if failed_run is not None:
                failed_run.status = "failed"
                failed_run.finished_at = utcnow()
                failed_run.config_json = dict(failed_run.config_json or {}) | {
                    "failure": {"message": str(exc)}
                }
                await db.commit()
        fresh = await db.get(InterviewSession, session_id)
        if fresh is not None:
            fresh.internal_summary_json = dict(fresh.internal_summary_json or {}) | {
                "evaluation_failure": {
                    "message": str(exc),
                    "failed_at": utcnow().isoformat(),
                    "final_attempt": is_final_attempt,
                }
            }
            # Only stamp the terminal 'failed' status once ARQ has exhausted
            # its retry budget. Marking it failed on attempt 1/3 would tell
            # the student the interview is dead while a retry is still
            # queued — but NOT stamping it on the LAST attempt leaves the
            # session stuck at 'completed' with pass_verdict forever null,
            # so the frontend poll in course-interview.tsx never resolves
            # and the student waits indefinitely (the bug we're fixing).
            if is_final_attempt:
                fresh.status = "failed"
            await db.commit()
        raise


async def _list_candidate_answers(
    db: AsyncSession, session_id: UUID
) -> list[InterviewSessionMessage]:
    messages = await sessions_queries.list_session_messages(db, session_id)
    return [
        m
        for m in messages
        if getattr(m, "role", None) == "user"
        and getattr(m, "session_question_id", None) is not None
        and (getattr(m, "metadata_json", None) or {}).get("kind") != "onboarding"
    ]


async def generate_practice_feedback(db: AsyncSession, session_id: UUID) -> None:
    """Judge a practice run against the criteria, without grading it.

    A deliberately separate function rather than a branch inside
    :func:`evaluate_and_generate_report`. That function is the only writer of
    ``pass_verdict``, and ``pass_verdict = TRUE`` is what opens the SR-lesson
    and quiz gates — keeping the two apart makes "a rehearsal can never unlock
    anything" a structural property instead of a conditional someone could
    later invert. Each function refuses the other's mode outright.

    What it runs, and what it deliberately does not:

    * ``evaluate_outcomes`` — per-criterion met/not-met. This is the whole
      point: the student saw those criteria before the run, so the feedback
      that closes the loop is which ones they actually demonstrated.
    * NOT ``evaluate_session`` — that produces numeric rubric scores, which
      thesis §4.3 keeps away from students in any mode.
    * NOT ``generate_gap_report`` — it writes a ``gap_reports`` row that the
      results screen reads as a graded outcome, and it needs the learner-facing
      output guard. A rehearsal should not manufacture either.
    * NOT ``_derive_pass_verdict`` / ``_stamp_session_summary`` — no verdict,
      no score, no ``questions_total`` that a teacher view would read as a
      graded attempt.

    Failures are recorded rather than raised. Unlike a real evaluation there is
    no verdict anyone is blocked on, nothing downstream is gated on this, and
    the recovery sweep skips practice — so there is nothing for a retry loop to
    repair. But the failure IS stamped, because the alternative is a client that
    cannot tell "still thinking" from "never coming" and spins forever.
    """
    session = await sessions_queries.get_session(db, session_id)
    if session is None:
        raise NotFoundError(f"Interview session {session_id} not found")
    if not practice.is_practice(session.session_mode):
        # The mirror of the refusal in evaluate_and_generate_report. A graded
        # session reaching here would get outcome rows written without the
        # verdict, score and gap report that make them meaningful.
        return

    try:
        outcomes = await authoring_queries.list_outcomes_for_config(db, session.interview_config_id)
        all_questions = await authoring_queries.list_questions_for_config(
            db,
            session.interview_config_id,
            practice_only=True,
        )
        candidate_answers = await _list_candidate_answers(db, session_id)
        snapshot = await sessions_queries.get_session_with_responses(db, session_id)
        if snapshot is None:
            raise NotFoundError(f"Interview session {session_id} not found")
        _, session_questions, _ = snapshot
        questions, question_prompts, _expected, answered_question_ids = (
            _build_question_evaluation_context(all_questions, session_questions, candidate_answers)
        )

        verdicts = await evaluate_outcomes(
            db,
            session=session,
            outcomes=outcomes,
            questions=questions,
            answers=candidate_answers,
            question_prompts=question_prompts,
            pipeline_run_id=None,
        )
        # Same honesty as the graded path: a criterion whose question was never
        # answered was not demonstrated, whatever the judge inferred elsewhere.
        verdicts = _fail_unanswered_outcomes(
            verdicts,
            questions=questions,
            answered_question_ids=answered_question_ids,
        )

        await _persist_outcome_evaluations(db, session_id=session_id, verdicts=verdicts)
        session.internal_summary_json = dict(session.internal_summary_json or {}) | {
            "practice": {
                "outcomes_met": verdicts.met_count,
                "outcomes_total": verdicts.total,
                "evaluated_at": utcnow().isoformat(),
            }
        }
        await db.commit()
    except Exception as exc:  # noqa: BLE001 -- see the docstring: recorded, not raised
        await db.rollback()
        fresh = await db.get(InterviewSession, session_id)
        if fresh is not None:
            fresh.internal_summary_json = dict(fresh.internal_summary_json or {}) | {
                "practice": {"failed": True, "message": str(exc), "at": utcnow().isoformat()}
            }
            await db.commit()


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
    """
    for verdict in verdicts.verdicts:
        db.add(
            InterviewOutcomeEvaluation(
                session_id=session_id,
                outcome_id=verdict.outcome_id,
                verdict_met=verdict.met,
                hidden_reasoning=verdict.reasoning,
                evidence_excerpt=verdict.evidence,
            )
        )
    await flush_or_conflict(db)


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


async def _persist_gap_report(
    db: AsyncSession,
    *,
    session: InterviewSession,
    course_id: UUID,
    module_id: UUID | None,
    draft: GapReportDraft,
) -> None:
    # Gap-report prose is also learner-facing AI output. Guard the complete
    # learner projection (summary + generated study-plan text) before it can be
    # serialized by REST. Numeric rubric details remain teacher-only.
    from sqlalchemy import select  # noqa: PLC0415

    asked_question_ids = list(
        (
            await db.execute(
                select(InterviewSessionQuestion.interview_question_id).where(
                    InterviewSessionQuestion.session_id == session.id,
                    InterviewSessionQuestion.interview_question_id.is_not(None),
                )
            )
        )
        .scalars()
        .all()
    )
    learner_parts = [draft.student_summary]
    for item in draft.study_plan:
        learner_parts.extend((item.topic, item.weakness_summary))
    assessment = SecurityAssessment(
        category=SecurityCategory.BENIGN,
        detected=False,
        confidence=1.0,
        should_block=False,
        should_record_academic_evidence=False,
        response_key=None,
        normalized_fingerprint=None,
        source="gap_report_boundary",
    )
    guarded = await security_service.guard_student_output(
        db,
        session_id=session.id,
        config_id=session.interview_config_id,
        turn_key=f"gap-report:{session.id}",
        proposed_text="\n".join(part for part in learner_parts if part),
        fallback_text=(
            "Your interview feedback could not be displayed safely. "
            "Please ask your instructor for learning guidance."
        ),
        allowed_question_ids=[qid for qid in asked_question_ids if qid is not None],
        assessment=assessment,
        action=SecurityAction.ALLOW,
        attempt_count=0,
    )
    report_json = dict(draft.report_json)
    student_summary = draft.student_summary
    if guarded.output_fallback_used:
        student_summary = guarded.text
        report_json["study_plan"] = []
        report_json["strengths"] = []
        report_json["weaknesses"] = []
    db.add(
        GapReport(
            student_id=session.student_id,
            course_id=course_id,
            module_id=module_id,
            source_quiz_attempt_id=None,
            source_interview_session_id=session.id,
            student_summary=student_summary,
            teacher_summary=draft.teacher_summary,
            report_json=report_json,
        )
    )
    await flush_or_conflict(db)


__all__ = ["evaluate_and_generate_report"]
