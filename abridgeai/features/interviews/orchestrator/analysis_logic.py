"""Answer-analysis LLM call (Phase 3).

Wires :mod:`orchestrator.analysis` to the LLM gateway. Pure types + parser
live in that module; this one owns the I/O and mirrors the follow-up / intent
conventions:

* SAVEPOINT (``begin_nested``) around the gateway call so a failed
  ``ai_model_calls`` audit flush can't poison the request session.
* Malformed output is retried ONCE (per the brief's Phase 3 requirement),
  then falls back to a safe not-assessable analysis — the student's turn is
  still recorded and the session still advances.
* Never raises: analysis is guidance for the next interviewer action, never a
  hard dependency of recording the answer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING

from abridgeai.ai.llm import LLMGateway, LLMRole
from abridgeai.ai.prompts import render_prompt
from abridgeai.core.config import get_settings
from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    OutcomeEvidence,
    fallback_analysis,
    parse_analysis_response,
)
from abridgeai.features.interviews.orchestrator.extraction import AnswerClaims, ClaimKind

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

ANALYSIS_STAGE_NAME = "interview_analysis"
# Distinct stage name so telemetry can separate legacy analysis from the split
# path's rubric half while both run in shadow.
MATCHING_STAGE_NAME = "interview_matching"


async def analyze_answer(
    db: AsyncSession,
    *,
    question_text: str,
    student_answer: str,
    turn_id: str,
    outcome_id: str | None = None,
    outcome_text: str | None = None,
    expected_evidence: Sequence[str] | None = None,
    common_misconceptions: Sequence[str] | None = None,
    supplementary_instructions: str | None = None,
    prior_claims: Sequence[str] | None = None,
    other_outcomes: Sequence[Mapping[str, str]] | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> AnswerAnalysis:
    """Analyze a student answer into a structured :class:`AnswerAnalysis`.

    Never raises. On malformed model output, retries once; if still unusable
    (or the call fails), returns :func:`fallback_analysis` so the orchestrator
    advances neutrally without inventing a score.

    ``other_outcomes`` enables *emergent* evidence: a list of
    ``{"id": ..., "text": ...}`` for the config's OTHER outcomes, letting the
    analyzer credit an outcome the question did not target when the candidate
    demonstrably covered it in passing. Omit/empty → v1 behaviour (only the
    linked outcome is visible). Attributions are sanitized against this
    whitelist by :func:`_sanitize_evidence`, so a hallucinated id can never
    create a phantom outcome.
    """
    answer = (student_answer or "").strip()
    if not answer:
        return fallback_analysis(turn_id)

    allowed_other = {
        str(o["id"]) for o in (other_outcomes or []) if isinstance(o, Mapping) and o.get("id")
    }
    try:
        system_prompt = render_prompt("prompts/analysis_system.j2")
        user_prompt = json.dumps(
            {
                "current_question": question_text,
                "student_answer": answer,
                "turn_id": turn_id,
                "outcome": {"id": outcome_id or "", "text": outcome_text or ""},
                "expected_evidence": list(expected_evidence or []),
                "common_misconceptions": list(common_misconceptions or []),
                "supplementary_instructions": supplementary_instructions or "",
                "prior_claims": list(prior_claims or []),
                "other_outcomes": [
                    {"id": str(o["id"]), "text": str(o.get("text", ""))}
                    for o in (other_outcomes or [])
                    if isinstance(o, Mapping) and o.get("id")
                ],
            },
            ensure_ascii=False,
        )
        return await _call_and_parse(
            db,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            turn_id=turn_id,
            outcome_id=outcome_id,
            allowed_other=allowed_other,
            pipeline_run_id=pipeline_run_id,
            gateway=gateway,
        )
    except Exception:  # noqa: BLE001 — analysis is best-effort; never bubble to /respond
        logger.exception("interview analysis stage failed; using fallback")

    return fallback_analysis(turn_id)


async def match_claims_to_outcomes(
    db: AsyncSession,
    *,
    claims: AnswerClaims,
    question_text: str,
    turn_id: str,
    outcome_id: str | None = None,
    outcome_text: str | None = None,
    expected_evidence: Sequence[str] | None = None,
    common_misconceptions: Sequence[str] | None = None,
    prior_claims: Sequence[str] | None = None,
    other_outcomes: Sequence[Mapping[str, str]] | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> AnswerAnalysis:
    """Rubric-side half of the split analysis: claims in, evidence out.

    This is the call that holds the protected surface — outcome text, other
    outcomes, expected evidence, misconceptions. It never receives the
    candidate's raw words; only the bounded, rules-screened claims produced by
    :func:`orchestrator.extraction_logic.extract_answer_claims`. That is the
    isolation property the split exists to create.

    Question-level judgements (relevance / completeness / specificity /
    self_corrected) are NOT asked of this stage — only the extractor saw the
    text they describe, so they are copied over deterministically afterwards
    and whatever the model says about them is discarded.

    Never raises; degrades to :func:`fallback_analysis` like its sibling.
    """
    if not claims.claims:
        # No screened claims survived. Returning a neutral analysis is correct:
        # an empty projection is not evidence that the answer was bad, and
        # inventing evidence from nothing is exactly the failure to avoid.
        return _carry_extractor_fields(fallback_analysis(turn_id), claims)

    allowed_other = {
        str(o["id"]) for o in (other_outcomes or []) if isinstance(o, Mapping) and o.get("id")
    }
    try:
        system_prompt = render_prompt("prompts/matching_system.j2")
        user_prompt = json.dumps(
            {
                "current_question": question_text,
                "claims": [{"text": c.text, "kind": c.kind.value} for c in claims.claims],
                "turn_id": turn_id,
                "outcome": {"id": outcome_id or "", "text": outcome_text or ""},
                "expected_evidence": list(expected_evidence or []),
                "common_misconceptions": list(common_misconceptions or []),
                "prior_claims": list(prior_claims or []),
                "other_outcomes": [
                    {"id": str(o["id"]), "text": str(o.get("text", ""))}
                    for o in (other_outcomes or [])
                    if isinstance(o, Mapping) and o.get("id")
                ],
            },
            ensure_ascii=False,
        )
        parsed = await _call_and_parse(
            db,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            turn_id=turn_id,
            outcome_id=outcome_id,
            allowed_other=allowed_other,
            pipeline_run_id=pipeline_run_id,
            gateway=gateway,
            stage_name=MATCHING_STAGE_NAME,
        )
        return _carry_extractor_fields(parsed, claims)
    except Exception:  # noqa: BLE001 — matching is best-effort; never bubble to /respond
        logger.exception("interview matching stage failed; using fallback")

    return _carry_extractor_fields(fallback_analysis(turn_id), claims)


async def analyze_turn(
    db: AsyncSession,
    *,
    question_text: str,
    student_answer: str,
    turn_id: str,
    outcome_id: str | None = None,
    outcome_text: str | None = None,
    expected_evidence: Sequence[str] | None = None,
    common_misconceptions: Sequence[str] | None = None,
    prior_claims: Sequence[str] | None = None,
    other_outcomes: Sequence[Mapping[str, str]] | None = None,
    pipeline_run_id: UUID | None = None,
    gateway: LLMGateway | None = None,
) -> AnswerAnalysis:
    """Analyse one turn under ``interview_analysis_split_mode``.

    The single entry point both runtime call sites use, so the legacy and split
    paths cannot diverge per-caller:

    * ``off`` — one legacy call holding the raw answer and the rubric together.
    * ``shadow`` — run both, log the divergence, **return the legacy result**.
      Measurement must never change what a learner experiences.
    * ``enforce`` — return the split result.

    ``supplementary_instructions`` is deliberately not a parameter: neither path
    may place the author-wide rubric blob in a runtime context, and leaving it
    off the signature makes that unforgettable rather than a per-caller habit.
    """
    mode = get_settings().interview_analysis_split_mode

    async def _legacy() -> AnswerAnalysis:
        return await analyze_answer(
            db,
            question_text=question_text,
            student_answer=student_answer,
            turn_id=turn_id,
            outcome_id=outcome_id,
            outcome_text=outcome_text,
            expected_evidence=expected_evidence,
            common_misconceptions=common_misconceptions,
            supplementary_instructions=None,
            prior_claims=prior_claims,
            other_outcomes=other_outcomes,
            pipeline_run_id=pipeline_run_id,
            gateway=gateway,
        )

    async def _split() -> AnswerAnalysis:
        # extract_answer_claims is the ONLY call that sees student_answer.
        from abridgeai.features.interviews.orchestrator.extraction_logic import (  # noqa: PLC0415
            extract_answer_claims,
        )

        claims = await extract_answer_claims(
            db,
            question_text=question_text,
            student_answer=student_answer,
            pipeline_run_id=pipeline_run_id,
            gateway=gateway,
        )
        return await match_claims_to_outcomes(
            db,
            claims=claims,
            question_text=question_text,
            turn_id=turn_id,
            outcome_id=outcome_id,
            outcome_text=outcome_text,
            expected_evidence=expected_evidence,
            common_misconceptions=common_misconceptions,
            prior_claims=prior_claims,
            other_outcomes=other_outcomes,
            pipeline_run_id=pipeline_run_id,
            gateway=gateway,
        )

    if mode == "enforce":
        return await _split()
    if mode == "shadow":
        legacy = await _legacy()
        try:
            shadow = await _split()
            log_analysis_divergence(legacy=legacy, shadow=shadow, turn_id=turn_id)
        except Exception:  # noqa: BLE001 -- shadow is observation only
            logger.exception("analysis split shadow failed; legacy result stands")
        return legacy
    return await _legacy()


def log_analysis_divergence(
    *, legacy: AnswerAnalysis, shadow: AnswerAnalysis, turn_id: str
) -> None:
    """Record how the split path differed from legacy, for shadow rollout.

    Emits only bounded, non-sensitive fields — enum values, counts, booleans.
    Student text and evidence summaries are never logged: the split exists to
    stop that content spreading, so a measurement channel that re-leaked it
    would defeat the purpose.
    """
    legacy_ids = sorted({(e.outcome_id or "") for e in legacy.evidence})
    shadow_ids = sorted({(e.outcome_id or "") for e in shadow.evidence})
    logger.info(
        "analysis_split_divergence",
        extra={
            "turn_id": turn_id,
            "correctness_legacy": legacy.correctness.value,
            "correctness_shadow": shadow.correctness.value,
            "correctness_agrees": legacy.correctness is shadow.correctness,
            "evidence_count_legacy": len(legacy.evidence),
            "evidence_count_shadow": len(shadow.evidence),
            "outcome_ids_agree": legacy_ids == shadow_ids,
            "probe_agrees": legacy.recommended_probe_type is shadow.recommended_probe_type,
        },
    )


def _carry_extractor_fields(analysis: AnswerAnalysis, claims: AnswerClaims) -> AnswerAnalysis:
    """Overwrite the fields only the extractor could have observed.

    The matcher never saw the raw answer, so any value it produced for these is
    a guess from a lossy projection. ``has_concrete_example`` is derived
    deterministically from claim kinds rather than asked of either model.
    """
    analysis.relevance = claims.relevance
    analysis.completeness = claims.completeness
    analysis.specificity = claims.specificity
    analysis.self_corrected = claims.self_corrected
    analysis.has_concrete_example = any(c.kind is ClaimKind.EXAMPLE for c in claims.claims)
    return analysis


async def _call_and_parse(
    db: AsyncSession,
    *,
    system_prompt: str,
    user_prompt: str,
    turn_id: str,
    outcome_id: str | None,
    allowed_other: set[str],
    pipeline_run_id: UUID | None,
    gateway: LLMGateway | None,
    stage_name: str = ANALYSIS_STAGE_NAME,
) -> AnswerAnalysis:
    """One gateway call + parse + evidence sanitize, retried once.

    Shared by :func:`analyze_answer` and :func:`match_claims_to_outcomes` so the
    retry policy and the id allowlist cannot drift apart between the legacy and
    split paths — which matters while both run side by side in shadow mode.
    """
    gateway = gateway or LLMGateway()
    for attempt in (1, 2):  # one retry on malformed output (Phase 3 spec)
        async with db.begin_nested():
            llm_result = await gateway.generate_json(
                role=LLMRole.INTERVIEW_ANALYSIS,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                db=db,
                stage_name=stage_name,
                pipeline_run_id=pipeline_run_id,
            )
        payload = llm_result.content_json if isinstance(llm_result.content_json, dict) else None
        parsed = parse_analysis_response(payload, default_turn_id=turn_id)
        if parsed is not None:
            _sanitize_evidence(parsed, target_outcome_id=outcome_id, allowed_other=allowed_other)
            return parsed
        logger.warning("%s: malformed payload (attempt %d/2)", stage_name, attempt)
    return fallback_analysis(turn_id)


def _sanitize_evidence(
    analysis: AnswerAnalysis,
    *,
    target_outcome_id: str | None,
    allowed_other: set[str],
) -> None:
    """Drop untrustworthy evidence attributions in place (defence in depth).

    The model is asked to only cite ids it was given, but a wrong id would
    silently create a phantom outcome in the coverage map, so we enforce it here
    rather than trusting the prompt:

    * evidence for the linked outcome → kept, ``secondary=False``;
    * evidence for an id in ``allowed_other`` → kept, ``secondary=True``
      (forced, regardless of what the model claimed);
    * anything else (unknown/invented/empty id) → dropped.

    When ``allowed_other`` is empty (emergent evidence disabled) this reduces to
    v1 behaviour: only target-outcome evidence survives.
    """
    target = str(target_outcome_id) if target_outcome_id else None
    kept: list[OutcomeEvidence] = []
    for ev in analysis.evidence:
        oid = (ev.outcome_id or "").strip()
        if target is not None and oid == target:
            ev.secondary = False
            kept.append(ev)
        elif oid and oid in allowed_other:
            ev.secondary = True
            kept.append(ev)
        elif not oid and target is not None:
            # Model omitted the id on a target-outcome item — repair it rather
            # than discarding real evidence for the question actually asked.
            ev.outcome_id = target
            ev.secondary = False
            kept.append(ev)
    analysis.evidence = kept


__all__ = [
    "ANALYSIS_STAGE_NAME",
    "MATCHING_STAGE_NAME",
    "analyze_answer",
    "match_claims_to_outcomes",
]
