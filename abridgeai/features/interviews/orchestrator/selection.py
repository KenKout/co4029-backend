"""Adaptive question selection (Phase 5).

Turns the teacher-approved question bank from a fixed script into a *pool*. A
deterministic scorer ranks un-asked approved questions by:

* uncovered-outcome priority (target outcomes with the least evidence first),
* outcome importance weight (1..5, from ``InterviewOutcome.importance_weight``),
* teacher-order preference (preserve ``position`` as a tie-breaker / fallback),
* difficulty fit (ramp easy→hard; ease off when time is low or answers weak),
* time fit (avoid long/hard questions when little time remains),
* duplicate / recently-asked penalties.

Determinism is deliberate (brief non-goal: "do not depend entirely on an LLM for
deterministic state transitions"). No LLM is used for selection — only for the
semantic analysis that *feeds* it (Phase 3). The existing strictly-sequential
selector is preserved verbatim as :func:`sequential_pick` and is the guaranteed
fallback whenever adaptive selection can't run or returns nothing.

Pure ranking logic here — the DB fetch of candidate questions lives in the
caller (Slice 4), so this module stays unit-testable with plain objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from abridgeai.features.interviews.orchestrator.coverage import COVERAGE_SUFFICIENT_POINTS


class Difficulty(str, Enum):  # noqa: UP042 -- match codebase convention
    JUNIOR = "junior"
    MID_LEVEL = "mid_level"
    SENIOR = "senior"


_DIFFICULTY_RANK: dict[str, int] = {
    Difficulty.JUNIOR.value: 1,
    Difficulty.MID_LEVEL.value: 2,
    Difficulty.SENIOR.value: 3,
}


@dataclass(frozen=True)
class CandidateQuestion:
    """A poolable question, decoupled from the ORM row.

    ``position`` preserves the teacher's authored order (None sorts last).
    ``importance_weight`` is the linked outcome's weight (1..5; default 1 when
    the question has no linked outcome).
    """

    question_id: str
    linked_outcome_id: str | None
    question_type: str
    difficulty: str | None
    position: int | None
    importance_weight: int = 1


@dataclass(frozen=True)
class SelectionContext:
    """Runtime inputs that shape scoring for the current decision point."""

    asked_question_ids: frozenset[str]
    skipped_question_ids: frozenset[str]
    # outcome_id -> provisional evidence count (0 = uncovered).
    outcome_evidence_counts: dict[str, int]
    # outcome_ids that still need coverage to pass (required & not yet sufficient).
    uncovered_required_outcome_ids: frozenset[str]
    # Rough difficulty the student is handling well so far (1..3); None = unknown.
    student_difficulty_level: int | None = None
    time_fraction_remaining: float | None = None
    # outcome_id most recently targeted — penalise to avoid hammering one outcome.
    last_targeted_outcome_id: str | None = None
    # Per-outcome competence estimates (Slice 12, v2): outcome_id -> 0..1. When a
    # candidate's linked outcome has an entry, difficulty fit targets THAT
    # competence (push harder on strengths, ease on weak areas) instead of the
    # single global student level. Empty/missing → fall back to the global level.
    outcome_competence: dict[str, float] | None = None
    # Outcome backtracking (Slice 18, v2): when True, reward un-asked questions
    # on UNDER-covered outcomes (touched but below the sufficiency threshold) so
    # the selector circles back to rushed partials before spending turns on
    # already-covered ones. Below the uncovered reward, so fresh outcomes still
    # lead (breadth-first preserved). Off → the under-covered band is ignored (v1).
    backtrack_undercovered: bool = False


@dataclass(frozen=True)
class ScoredCandidate:
    candidate: CandidateQuestion
    score: float
    breakdown: dict[str, float]


# ── scoring weights (documented; single source of truth) ─────────────────────
# Each term is additive. Tuned so *required-uncovered outcome coverage* dominates,
# then outcome importance, then difficulty/time fit, with teacher order as a
# small nudge and penalties for duplication / single-outcome fixation.
_W_UNCOVERED_OUTCOME = 100.0  # question targets an uncovered outcome
_W_UNDERCOVERED_OUTCOME = 40.0  # targets a touched-but-insufficient outcome (Slice 18)
_W_REQUIRED_OUTCOME = 60.0  # ...and that outcome is required to pass
_W_IMPORTANCE = 8.0  # × importance_weight (1..5) → up to 40
_W_TEACHER_ORDER = 5.0  # earlier position preferred (small nudge)
_W_DIFFICULTY_FIT = 15.0  # difficulty matches student's demonstrated level
_W_TIME_FIT = 12.0  # easier questions preferred when time is low
_P_ALREADY_COVERED = 25.0  # penalty: outcome already has ample evidence
_P_RECENT_OUTCOME = 20.0  # penalty: same outcome as last targeted


def _competence_to_rank(competence: float) -> int:
    """Map a 0..1 competence estimate to a target difficulty rank (1..3).

    Low competence → aim junior (1); mid → mid-level (2); high → senior (3).
    Thresholds mirror the neutral-0.5 prior: below 0.4 eases, above 0.6 pushes.
    """
    if competence >= 0.6:
        return 3
    if competence <= 0.4:
        return 1
    return 2


def _difficulty_fit_score(
    ctx: SelectionContext, difficulty: str | None, *, outcome_id: str | None = None
) -> float:
    """Reward a difficulty near the student's demonstrated level.

    Slice 12 (v2): when a per-outcome competence estimate exists for this
    candidate's linked outcome, target THAT competence's difficulty (calibrate
    per topic); otherwise fall back to the single global student level. Unknown
    difficulty or no target level → neutral (half credit), so a question is
    never excluded merely for missing metadata.
    """
    rank = _DIFFICULTY_RANK.get(difficulty or "", 0)
    if rank == 0:
        return _W_DIFFICULTY_FIT * 0.5

    target_level: int | None = None
    if outcome_id is not None and ctx.outcome_competence:
        competence = ctx.outcome_competence.get(outcome_id)
        if competence is not None:
            target_level = _competence_to_rank(competence)
    if target_level is None:
        target_level = ctx.student_difficulty_level
    if target_level is None:
        return _W_DIFFICULTY_FIT * 0.5

    distance = abs(rank - target_level)
    # distance 0 → full, 1 → half, 2 → zero.
    return _W_DIFFICULTY_FIT * max(0.0, 1.0 - distance / 2.0)


def _time_fit_score(ctx: SelectionContext, difficulty: str | None) -> float:
    """When time is low, prefer easier questions; otherwise neutral."""
    if ctx.time_fraction_remaining is None or ctx.time_fraction_remaining > 0.2:
        return _W_TIME_FIT * 0.5
    rank = _DIFFICULTY_RANK.get(difficulty or "", 2)
    # low time: junior(1) → full, mid(2) → half, senior(3) → zero.
    return _W_TIME_FIT * max(0.0, 1.0 - (rank - 1) / 2.0)


def score_candidate(candidate: CandidateQuestion, ctx: SelectionContext) -> ScoredCandidate:
    """Score one candidate. Higher is better. Returns a full breakdown for audit."""
    breakdown: dict[str, float] = {}
    oid = candidate.linked_outcome_id

    evidence = ctx.outcome_evidence_counts.get(oid, 0) if oid else 0
    is_uncovered = evidence == 0
    breakdown["uncovered_outcome"] = _W_UNCOVERED_OUTCOME if is_uncovered else 0.0

    # Under-covered band (Slice 18, v2): the outcome was touched but has not yet
    # reached the sufficiency threshold. Rewarded ONLY when backtracking is
    # enabled, so the selector circles back to rushed partials before spending a
    # turn on an already-covered outcome. Below the uncovered reward, so a fresh
    # (0-evidence) outcome still ranks higher (breadth-first preserved).
    is_undercovered = 0 < evidence < COVERAGE_SUFFICIENT_POINTS
    breakdown["undercovered_outcome"] = (
        _W_UNDERCOVERED_OUTCOME if (ctx.backtrack_undercovered and is_undercovered) else 0.0
    )

    is_required_uncovered = oid is not None and oid in ctx.uncovered_required_outcome_ids
    breakdown["required_outcome"] = _W_REQUIRED_OUTCOME if is_required_uncovered else 0.0

    breakdown["importance"] = _W_IMPORTANCE * float(max(1, min(5, candidate.importance_weight)))

    # Teacher order: earlier position scores higher; None (unordered) → 0 nudge.
    if candidate.position is not None:
        breakdown["teacher_order"] = _W_TEACHER_ORDER / float(candidate.position + 1)
    else:
        breakdown["teacher_order"] = 0.0

    breakdown["difficulty_fit"] = _difficulty_fit_score(ctx, candidate.difficulty, outcome_id=oid)
    breakdown["time_fit"] = _time_fit_score(ctx, candidate.difficulty)

    # Penalty: outcome already amply covered (>= 2 pieces of evidence).
    breakdown["already_covered_penalty"] = -_P_ALREADY_COVERED if (oid and evidence >= 2) else 0.0

    # Penalty: same outcome as the immediately-previous target (avoid fixation).
    breakdown["recent_outcome_penalty"] = (
        -_P_RECENT_OUTCOME if (oid is not None and oid == ctx.last_targeted_outcome_id) else 0.0
    )

    score = sum(breakdown.values())
    return ScoredCandidate(candidate=candidate, score=score, breakdown=breakdown)


def select_next_question(
    candidates: list[CandidateQuestion], ctx: SelectionContext
) -> ScoredCandidate | None:
    """Pick the highest-scoring un-asked candidate, or None if the pool is empty.

    Already-asked and skipped questions are filtered out first (never re-ask
    unless the caller explicitly repeats/reframes, which is a *decision* action,
    not a selection). Ties break by teacher ``position`` then ``question_id`` for
    full determinism.
    """
    pool = [
        c
        for c in candidates
        if c.question_id not in ctx.asked_question_ids
        and c.question_id not in ctx.skipped_question_ids
    ]
    if not pool:
        return None

    scored = [score_candidate(c, ctx) for c in pool]
    scored.sort(
        key=lambda s: (
            -s.score,
            s.candidate.position if s.candidate.position is not None else 1_000_000,
            s.candidate.question_id,
        )
    )
    return scored[0]


def sequential_pick(
    candidates: list[CandidateQuestion], asked_question_ids: frozenset[str]
) -> CandidateQuestion | None:
    """The legacy strictly-sequential selector, preserved as the fallback.

    Mirrors ``taking._next_published_question_after``: first candidate (in
    teacher/position order) not yet asked. Guarantees the adaptive path can
    always degrade to the exact prior behaviour.
    """
    ordered = sorted(
        candidates,
        key=lambda c: (c.position if c.position is not None else 1_000_000, c.question_id),
    )
    for c in ordered:
        if c.question_id not in asked_question_ids:
            return c
    return None


__all__ = [
    "CandidateQuestion",
    "Difficulty",
    "ScoredCandidate",
    "SelectionContext",
    "score_candidate",
    "select_next_question",
    "sequential_pick",
]
