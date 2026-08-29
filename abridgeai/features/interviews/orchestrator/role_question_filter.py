"""Role-conditioned question filter — WHICH question type a role prefers.

This is the ONE place the ``interviewer_role`` is allowed to reach question
*selection*, and it does so as a **pre-filter on the candidate list** at the
call site (``adaptive.py``), never as an input to the scorer. See
:mod:`orchestrator.interviewer_identity` for the hard boundary: identity /
persona shape LANGUAGE ONLY; here we relax that in exactly one, deliberately
narrow way — the owner decided a config-scoped role may choose the question
*type* it asks.

Why this is defensible (and not the fairness bug the persona invariant guards
against): ``interviewer_role`` is **config-scoped** (stored on
``InterviewConfig.persona_profile_json``, one value per assessment). Every
candidate of the same config meets the SAME role, so they face the SAME
type-filtered selection — two candidates of equal ability still score
identically. This is a design choice about *what is assessed* (a tech-lead
interview vs a behavioural screen), not per-candidate interviewer variance.

Guardrails that keep it safe:
- ``preferred_type`` is a pure 1:1 map. ``GENERIC_ASSISTANT`` maps to ``None``
  → ``filter_candidates_by_role`` returns the pool unchanged, so every config
  that never opted into a role is byte-for-byte the prior behaviour.
- The filter NEVER reads ``SelectionContext`` / ``DecisionInputs`` / rubric. It
  only trims the candidate list before ``select_next_question`` runs, so the
  scorer stays role-blind.
- It never leaves an outcome with zero eligible questions: an outcome whose
  bank has no preferred-type question keeps one non-preferred question
  (fallback), so coverage can never be blocked by a partial or legacy bank.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.interviewer_identity import InterviewerRole
from abridgeai.features.interviews.orchestrator.selection import CandidateQuestion

#: 1:1 mapping from interviewer role to the ``question_type`` it prefers.
#: ``None`` (generic assistant) means "no preference" → no filter.
_ROLE_PREFERRED_TYPE: dict[InterviewerRole, str | None] = {
    InterviewerRole.BACKEND_TECH_LEAD: "technical",
    InterviewerRole.STAFF_ENGINEER: "system_design",
    InterviewerRole.ENG_MANAGER: "situational",
    InterviewerRole.HR_SCREENER: "behavioral",
    InterviewerRole.GENERIC_ASSISTANT: None,
}


def preferred_type(role: InterviewerRole) -> str | None:
    """Return the ``question_type`` this role prefers, or ``None`` for no preference.

    ``None`` is the generic assistant (and any future role with no type bias),
    which is what keeps opted-out configs identical to the pre-feature engine.
    """
    return _ROLE_PREFERRED_TYPE.get(role)


def filter_candidates_by_role(
    candidates: list[CandidateQuestion],
    role: InterviewerRole,
    *,
    strict: bool = False,
) -> list[CandidateQuestion]:
    """HARD-filter the candidate pool to the role's preferred question type.

    Returns a new list; the input is never mutated. Behaviour:

    - ``preferred_type(role)`` is ``None`` → return ``candidates`` unchanged.
    - Otherwise keep every preferred-type question. For any outcome that has
      NO preferred-type question, also keep its non-preferred questions as a
      fallback so an outcome is never left un-coverable (covers the "role was
      set after a single-type generation" / partial-bank case; a no-op on a
      well-formed variant bank).
    - If the pool has no preferred-type question at all, degrade to the full
      pool rather than returning an empty selection.
    """
    preferred = preferred_type(role)
    if preferred is None:
        return candidates

    kept = [c for c in candidates if c.question_type == preferred]
    if strict:
        return kept + [c for c in candidates if c.variant_group_id is None]
    if not kept:
        return candidates

    covered_outcomes = {c.linked_outcome_id for c in kept if c.linked_outcome_id is not None}
    fallback = [
        c
        for c in candidates
        if c.variant_group_id is None
        or (
            c.linked_outcome_id is not None
            and c.linked_outcome_id not in covered_outcomes
            and c.question_type != preferred
        )
    ]
    return kept + fallback


__all__ = ["filter_candidates_by_role", "preferred_type"]
