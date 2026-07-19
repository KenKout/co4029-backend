"""Interview EVALUATION stage orchestrator (T6.8).

Runs after ``interview_sessions.status='completed'`` to compute per-criterion
rubric scores. Each candidate response is judged in **one** LLM call against
**all** configured rubric criteria (per-criterion calls would N×M too many
round-trips for a typical 10-question interview). Per-response evaluations
are then folded into a session-level :class:`RubricScores` by
:func:`abridgeai.features.interviews.ai.stages.evaluation.rubric.aggregate_rubric_scores`.

This stage **returns** ``RubricScores``; it does **not** persist. Persistence
to ``interview_outcome_evaluations`` and ``interview_sessions.total_score /
rubric_scores`` is owned by the T6.11 services layer so the stage stays
side-effect-free and reusable from any caller (worker, replay, integration
test).

Audit fields
------------
* ``stage_name="evaluation"`` — required by Reconciliation §B1 so
  ``ai_model_calls`` rows roll up to the evaluation phase of the run.
* ``pipeline_run_id`` — threaded into the gateway so per-call cost rolls
  up to the parent ``ai_pipeline_runs`` row.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
    OutcomeVerdicts,
    build_outcome_verdicts,
)
from abridgeai.features.interviews.ai.stages.evaluation.parsers import (
    parse_evaluation_response,
)
from abridgeai.features.interviews.ai.stages.evaluation.parsers_outcome_verdicts import (
    parse_outcome_verdicts,
)
from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    ResponseEvaluation,
    RubricScores,
    aggregate_rubric_scores,
    resolve_rubric,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.interviews.models import (
        InterviewOutcome,
        InterviewQuestion,
        InterviewSession,
        InterviewSessionMessage,
    )


EVALUATION_STAGE_NAME = "evaluation"


async def evaluate_outcomes(
    db: AsyncSession,
    *,
    session: InterviewSession,
    outcomes: Sequence[InterviewOutcome],
    questions: Sequence[InterviewQuestion],
    answers: Sequence[InterviewSessionMessage],
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> OutcomeVerdicts:
    """Judge the session transcript against EACH outcome (thesis §4.3).

    Produces one binary met/not-met :class:`OutcomeVerdict` per configured
    outcome in a SINGLE LLM call (the judge sees the whole transcript and all
    outcomes at once). This is the authoritative pass signal — the services
    layer compares ``met_count`` against ``InterviewConfig.min_outcomes_to_pass``.

    Unlike :func:`evaluate_session` (rubric, retained as a teacher diagnostic),
    this stage does NOT produce numeric scores and never references a model
    answer. Returns empty verdicts when there are no outcomes or no candidate
    answers. The stage is side-effect-free; the caller persists.
    """
    expected_outcome_ids = [outcome.id for outcome in outcomes]
    if not expected_outcome_ids:
        return build_outcome_verdicts([])

    transcript = _build_transcript(questions, answers)
    if not transcript:
        # No candidate answers → every outcome defaults to not-met via the parser.
        return build_outcome_verdicts(
            parse_outcome_verdicts(None, expected_outcome_ids=expected_outcome_ids)
        )

    outcome_views = [_outcome_for_verdict(outcome) for outcome in outcomes]

    gateway = gateway or LLMGateway()
    system_prompt = render_prompt("prompts/outcome_system.j2")
    user_prompt = json.dumps(
        {"transcript": transcript, "outcomes": outcome_views},
        ensure_ascii=False,
    )

    llm_result = await gateway.generate_json(
        role=LLMRole.INTERVIEW_EVALUATION,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        db=db,
        stage_name=EVALUATION_STAGE_NAME,
        pipeline_run_id=pipeline_run_id,
        parent_run_id=pipeline_run_id,
    )

    payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
    verdicts = parse_outcome_verdicts(payload, expected_outcome_ids=expected_outcome_ids)
    return build_outcome_verdicts(verdicts)


def _build_transcript(
    questions: Sequence[InterviewQuestion],
    answers: Sequence[InterviewSessionMessage],
) -> list[dict[str, str]]:
    """Pair each candidate answer with its question prompt, in answer order.

    Answers are ``role='user'`` rows; each carries ``session_question_id`` ->
    ``InterviewQuestion.id``. Answers with no resolvable question still surface
    (question text empty) so the judge sees the full candidate input.
    """
    prompts_by_question = _question_prompts(questions)
    transcript: list[dict[str, str]] = []
    for answer in answers:
        if not _is_candidate_answer(answer):
            continue
        response_text = _candidate_response_text(answer)
        if not response_text:
            continue
        question_id = getattr(answer, "session_question_id", None)
        question_prompt = (
            prompts_by_question.get(question_id, "") if isinstance(question_id, UUID) else ""
        )
        transcript.append({"question": question_prompt, "answer": response_text})
    return transcript


def _outcome_for_verdict(outcome: InterviewOutcome) -> dict[str, Any]:
    return {
        "outcome_id": str(outcome.id),
        "outcome_text": getattr(outcome, "outcome_text", "") or "",
        "outcome_type": getattr(outcome, "outcome_type", "") or "",
    }



async def evaluate_session(
    db: AsyncSession,
    *,
    session: InterviewSession,
    outcomes: Sequence[InterviewOutcome],
    questions: Sequence[InterviewQuestion],
    answers: Sequence[InterviewSessionMessage],
    config: Mapping[str, Any] | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> RubricScores:
    """Score every candidate answer against the configured rubric.

    Parameters
    ----------
    db
        Async session — passed through to :class:`LLMGateway` so each
        per-response judge call writes one ``ai_model_calls`` row.
    session
        Completed :class:`InterviewSession`. Only ``id`` is read here;
        callers fetch related rows separately to keep the stage
        composable.
    outcomes
        Linked learning outcomes for the interview config. Surfaced in
        the user prompt so the judge has scope context.
    questions
        :class:`InterviewQuestion` rows in display order. Used to
        resolve a question's prompt text for each answer via the answer
        message's ``session_question_id``.
    answers
        Candidate response messages (``role='user'`` rows in
        ``interview_session_messages``). One LLM judge call is fired
        per answer.
    config
        Interview run config (``rubric_weights``, ``rubric_criteria``).
        Falls back to four-criterion equal-weight default — see
        :func:`resolve_rubric`.
    pipeline_run_id
        Parent pipeline-run id; threaded into the gateway for cost
        attribution.
    gateway
        Inject a custom :class:`LLMGateway` (test seam). Defaults to
        ``LLMGateway()``.

    Returns
    -------
    RubricScores
        Per-response evaluations + aggregated per-criterion means and
        the session ``total_score`` (0-100). Caller (T6.11 services)
        persists this; the stage does not write to the DB itself.
    """

    rubric_weights = resolve_rubric(config)
    expected_criteria = tuple(rubric_weights.keys())

    if not answers or not expected_criteria:
        return aggregate_rubric_scores([], rubric_weights)

    prompts_by_question = _question_prompts(questions)
    outcome_views = [_outcome_for_prompt(outcome) for outcome in outcomes]
    gateway = gateway or LLMGateway()
    system_prompt = render_prompt("prompts/system.j2")

    response_evaluations: list[ResponseEvaluation] = []
    for answer in answers:
        if not _is_candidate_answer(answer):
            continue
        question_id = getattr(answer, "session_question_id", None)
        question_prompt = (
            prompts_by_question.get(question_id, "") if isinstance(question_id, UUID) else ""
        )
        response_text = _candidate_response_text(answer)
        if not response_text:
            continue

        user_prompt = json.dumps(
            {
                "question": question_prompt,
                "outcomes": outcome_views,
                "candidate_response": response_text,
                "rubric_criteria": list(expected_criteria),
            },
            ensure_ascii=False,
        )

        llm_result = await gateway.generate_json(
            role=LLMRole.INTERVIEW_EVALUATION,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            db=db,
            stage_name=EVALUATION_STAGE_NAME,
            pipeline_run_id=pipeline_run_id,
            parent_run_id=pipeline_run_id,
        )

        payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else {}
        criterion_scores = parse_evaluation_response(payload, expected_criteria=expected_criteria)
        response_evaluations.append(
            ResponseEvaluation(
                session_question_id=_resolve_session_question_id(answer, session),
                criterion_scores=criterion_scores,
            )
        )

    return aggregate_rubric_scores(response_evaluations, rubric_weights)


def _question_prompts(questions: Sequence[InterviewQuestion]) -> dict[UUID, str]:
    """Index InterviewQuestion rows by id so we can look up prompts by message FK."""

    return {
        question.id: question.prompt_text
        for question in questions
        if getattr(question, "prompt_text", None)
    }


def _outcome_for_prompt(outcome: InterviewOutcome) -> dict[str, Any]:
    return {
        "outcome_text": getattr(outcome, "outcome_text", "") or "",
        "outcome_type": getattr(outcome, "outcome_type", "") or "",
        "importance_weight": getattr(outcome, "importance_weight", 1) or 1,
    }


def _is_candidate_answer(message: InterviewSessionMessage) -> bool:
    """Filter for student utterances; AI / system messages are not scored."""

    return getattr(message, "role", None) == "user"


def _candidate_response_text(message: InterviewSessionMessage) -> str:
    """Return only evidence-eligible text for post-session grading."""
    metadata = getattr(message, "metadata_json", None) or {}
    if isinstance(metadata, dict) and metadata.get("kind") in {
        "security",
        "turn_control",
        "end_request",
    }:
        safe = metadata.get("safe_academic_text")
        return safe.strip() if isinstance(safe, str) else ""
    return (getattr(message, "content_text", None) or "").strip()


def _resolve_session_question_id(
    message: InterviewSessionMessage, session: InterviewSession
) -> UUID:
    candidate = getattr(message, "session_question_id", None)
    if isinstance(candidate, UUID):
        return candidate
    return session.id


__all__ = ["EVALUATION_STAGE_NAME", "evaluate_session"]
