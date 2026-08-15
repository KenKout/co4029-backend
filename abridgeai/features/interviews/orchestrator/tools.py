"""Server-authoritative tool surface for the interview agent.

The conversational LLM owns the words and the rhythm. The server owns WHAT may be
asked, WHETHER the interview may end, and WHAT counts as covered. These functions
are that boundary — pure, DB-free and LLM-free, so the guarantees they enforce can
be property-tested with plain objects exactly like ``decision.py`` is.

Design notes that are load-bearing:

* **The checklist is derived, never stored.** A tick is
  ``coverage_points >= COVERAGE_SUFFICIENT_POINTS``, read live from
  ``OutcomeCoverageState``. Storing a parallel boolean would eventually disagree
  with ``coverage.is_provisionally_sufficient`` and there would be no way to tell
  which one was right.

* **Ending is gated on three conditions, not one.** Refusing requires an unticked
  required outcome AND a question left to ask AND time above the closing
  threshold. Dropping any of them makes "refuse forever" reachable: an empty
  question pool with an unticked outcome can never be satisfied.

* **Refusals are bounded.** ``MAX_END_REFUSALS`` is the second anti-deadlock
  layer, independent of the first: after it the server stops arguing and lets the
  session close. (The third layer — a wall-clock stop for a model that never
  tries to end at all — lives in the session runtime, not here.)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from abridgeai.features.interviews.orchestrator.coverage import COVERAGE_SUFFICIENT_POINTS
from abridgeai.features.interviews.orchestrator.decision import MAX_CANNOT_ANSWER_HINTS
from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData

# How many times the server refuses `end_interview` before giving way. Bounded so
# a model that keeps trying cannot trap the candidate in a session that will not
# close; the refusal message names what is missing, so a cooperative model
# converges well before this.
MAX_END_REFUSALS = 2

# Same idea for advancing: the agent must resolve the current question before
# moving to a new one, but refusing forever must be unreachable. Independent bound
# from MAX_END_REFUSALS because the two failure modes are different — a model that
# rushes through questions vs one that refuses to finish.
MAX_ADVANCE_REFUSALS = 2


def _is_ticked(data: InterviewRuntimeStateData, outcome_id: str) -> bool:
    coverage = data.outcome_coverage.get(outcome_id)
    if coverage is None:
        return False
    return coverage.coverage_points >= COVERAGE_SUFFICIENT_POINTS


@dataclass(frozen=True)
class OutcomeProgress:
    outcome_id: str
    title: str | None
    ticked: bool
    points: int


@dataclass(frozen=True)
class ProgressReport:
    """What the model may read about its own progress.

    Exists so the agent can orient itself with a tool call instead of the server
    re-stuffing the whole runtime state into the system prompt every turn.
    """

    outcomes: list[OutcomeProgress] = field(default_factory=list)
    required_unticked: list[str] = field(default_factory=list)
    questions_remaining: int = 0
    hint_level: int = 0
    hints_left_here: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "outcomes": [
                {
                    "outcome_id": o.outcome_id,
                    "title": o.title,
                    "ticked": o.ticked,
                    "points": o.points,
                }
                for o in self.outcomes
            ],
            "required_unticked": list(self.required_unticked),
            "questions_remaining": self.questions_remaining,
            "hint_level": self.hint_level,
            "hints_left_here": self.hints_left_here,
        }


def build_progress_report(
    data: InterviewRuntimeStateData,
    *,
    required_outcome_ids: list[str],
    questions_remaining: int,
    outcome_titles: dict[str, str] | None = None,
) -> ProgressReport:
    titles = outcome_titles or {}
    outcomes = [
        OutcomeProgress(
            outcome_id=oid,
            title=titles.get(oid),
            ticked=_is_ticked(data, oid),
            points=(
                data.outcome_coverage[oid].coverage_points if oid in data.outcome_coverage else 0
            ),
        )
        for oid in required_outcome_ids
    ]
    return ProgressReport(
        outcomes=outcomes,
        required_unticked=[o.outcome_id for o in outcomes if not o.ticked],
        questions_remaining=questions_remaining,
        hint_level=data.hint_level,
        hints_left_here=max(0, MAX_CANNOT_ANSWER_HINTS - data.hint_level),
    )


@dataclass(frozen=True)
class EndInterviewVerdict:
    allowed: bool
    message: str
    refusal_count: int


def resolve_end_interview(
    data: InterviewRuntimeStateData,
    *,
    required_outcome_ids: list[str],
    questions_remaining: int,
    below_closing_threshold: bool,
    outcome_titles: dict[str, str] | None = None,
) -> EndInterviewVerdict:
    """Decide whether the agent may end the interview now.

    Refuses ONLY when continuing could still complete coverage: an unticked
    required outcome, a question left to ask, and time above the closing
    threshold. If any of those is false, refusing could never be satisfied, so the
    interview must be allowed to close.
    """
    unticked = [oid for oid in required_outcome_ids if not _is_ticked(data, oid)]
    could_still_cover = bool(unticked) and questions_remaining > 0 and not below_closing_threshold
    if not could_still_cover:
        return EndInterviewVerdict(True, "", data.end_refusal_count)

    if data.end_refusal_count >= MAX_END_REFUSALS:
        return EndInterviewVerdict(True, "", data.end_refusal_count)

    data.end_refusal_count += 1
    titles = outcome_titles or {}
    named = ", ".join(f"'{titles.get(oid, oid)}'" for oid in unticked)
    return EndInterviewVerdict(
        False,
        (
            f"Cannot end yet: these required outcomes are not covered: {named}. "
            f"{questions_remaining} question(s) remain. "
            "Call next_question to continue the interview."
        ),
        data.end_refusal_count,
    )


def build_turn_reminder(
    data: InterviewRuntimeStateData,
    *,
    current_outcome_id: str | None,
    required_outcome_ids: list[str],
    questions_remaining: int,
    max_follow_ups_per_question: int,
    below_closing_threshold: bool,
    outcome_titles: dict[str, str] | None = None,
    time_remaining_seconds: int | None = None,
    current_question_text: str | None = None,
    server_advanced: bool = False,
    opening: bool = False,
) -> str:
    """A one-paragraph state note appended to the agent's context each turn.

    Why append instead of letting the agent call ``get_progress``: a tool call
    costs an extra LLM round-trip, and the COMMON turn (probe / follow-up) needs
    no tool at all. Pushing the state in keeps the frequent path at one round-trip
    and still leaves the model unable to act on stale information.

    Carries PROGRESS ONLY — coverage counts, budgets, and the permitted next move.
    Never rubric text, expected evidence or answer content: this string enters the
    LLM context verbatim on every single turn, so anything sensitive here would be
    one prompt-injection away from the candidate.
    """
    titles = outcome_titles or {}
    advance = resolve_next_question(
        _peek(data),
        current_outcome_id=current_outcome_id,
        questions_remaining=questions_remaining,
        below_closing_threshold=below_closing_threshold,
        max_follow_ups_per_question=max_follow_ups_per_question,
    )
    unticked = [oid for oid in required_outcome_ids if not _is_ticked(data, oid)]

    parts: list[str] = []
    if current_question_text and opening:
        parts.append(_opening_clause(current_question_text))
    elif current_question_text:
        parts.append(_live_question_clause(current_question_text, server_advanced))
    if current_outcome_id is not None:
        current = data.outcome_coverage.get(current_outcome_id)
        points = current.coverage_points if current is not None else 0
        title = titles.get(current_outcome_id, current_outcome_id)
        covered = points >= COVERAGE_SUFFICIENT_POINTS
        parts.append(
            f"Current question targets '{title}': "
            f"{'covered' if covered else 'NOT yet covered'} "
            f"({points}/{COVERAGE_SUFFICIENT_POINTS} evidence points)."
        )
    hints_left = max(0, MAX_CANNOT_ANSWER_HINTS - data.hint_level)
    parts.append(
        f"Follow-ups used here: {data.current_question_follow_up_count}"
        f"/{max_follow_ups_per_question}. Hints left: {hints_left}."
    )
    if not server_advanced and not opening:
        parts.append(
            "You MAY call next_question when this exchange is finished."
            if advance.allowed
            else (
                "Do NOT call next_question yet — probe this answer further, "
                "or call request_hint if the candidate is stuck."
            )
        )
    # Cap the outcome list: this rides in context every turn, and a long syllabus
    # would crowd out the conversation the note exists to support.
    if unticked:
        shown = ", ".join(f"'{titles.get(oid, oid)}'" for oid in unticked[:3])
        more = f" (+{len(unticked) - 3} more)" if len(unticked) > 3 else ""
        parts.append(f"Still uncovered overall: {shown}{more}.")
    else:
        parts.append("All required outcomes are covered; you may call end_interview.")
    # The clock. Omitted entirely for an untimed session rather than reported as
    # zero, which would push the agent to rush a session that has no limit. When a
    # timed session's clock stops reaching the client its auto-close dies
    # silently, so this is a contract requirement, not a nicety.
    if time_remaining_seconds is not None:
        minutes = max(0, int(time_remaining_seconds) // 60)
        unit = "minute" if minutes == 1 else "minutes"
        parts.append(f"About {minutes} {unit} remain.")
        if below_closing_threshold:
            parts.append("Time is nearly up — wrap up and move to closing now.")
    return " ".join(parts)


def _opening_clause(current_question_text: str) -> str:
    """The join turn: nothing has been asked in the room yet."""
    return (
        "OPEN THE INTERVIEW NOW. Greet the candidate warmly in ONE short sentence, "
        "then put this question to them in your own words, assessing exactly what "
        f'it assesses: "{current_question_text}". Do not read it out verbatim, do '
        "not re-introduce yourself, do not call any tool, and do not add anything "
        "after the question."
    )


def _live_question_clause(current_question_text: str, server_advanced: bool) -> str:
    """Name the question the SERVER believes is live, verbatim, every turn.

    Without it the model announced "Moving on to the next question:" and asked a new
    one in its own words, never calling the tool. Server-side the interview stayed
    on the previous question, so the card, the counter and the grading all tracked a
    question nobody was answering — and the new one was scored against the old one's
    outcome.

    ``server_advanced`` flips the instruction from "you must call the tool to move"
    to "you have already been moved". Telling the model to call ``next_question``
    after the server advanced would consume a SECOND question for one transition.
    """
    if server_advanced:
        return (
            "The previous question is finished and the server has ALREADY moved the "
            f'interview to a NEW question: "{current_question_text}". Ask it now — '
            "acknowledge the candidate's last answer in one short sentence, then put "
            "this question to them, in your own words but assessing exactly what it "
            "assesses. Do NOT call next_question: it is already done, and calling it "
            "would skip a question. Do NOT ask anything else."
        )
    return (
        f'The candidate is answering THIS question: "{current_question_text}". '
        "Do NOT ask any other question in your own words. To move on you MUST "
        "call next_question and ask the question it returns; asking without it "
        "leaves the interview stuck here and the answer unscored."
    )


def _peek(data: InterviewRuntimeStateData) -> InterviewRuntimeStateData:
    """A throwaway copy for read-only gate evaluation.

    ``resolve_next_question`` increments the refusal counter as a side effect —
    that is correct when the agent actually calls the tool, but the reminder only
    ASKS what the answer would be. Passing the live object would burn the refusal
    budget on every turn and unblock advancing after two reminders.
    """
    return InterviewRuntimeStateData.from_dict(data.to_dict())


@dataclass(frozen=True)
class NextQuestionVerdict:
    allowed: bool
    message: str
    refusal_count: int


def resolve_next_question(
    data: InterviewRuntimeStateData,
    *,
    current_outcome_id: str | None,
    questions_remaining: int,
    below_closing_threshold: bool,
    max_follow_ups_per_question: int,
) -> NextQuestionVerdict:
    """Decide whether the agent may move to a NEW question.

    The current question must be resolved first — otherwise a model that finds
    silence uncomfortable walks the candidate through the whole bank without ever
    probing. "Resolved" is any of:

    * its outcome is provisionally ticked (answered well enough), or
    * the hint ladder is spent (the candidate genuinely cannot answer), or
    * the per-question follow-up budget is spent (we have probed enough), or
    * there is no current question yet, or
    * time is past the closing threshold.

    The last four are all escape hatches, and at least one must exist or a
    candidate who cannot answer would be held on question one forever.
    """
    if current_outcome_id is None or questions_remaining <= 0 or below_closing_threshold:
        return NextQuestionVerdict(True, "", data.advance_refusal_count)
    if _is_ticked(data, current_outcome_id):
        return NextQuestionVerdict(True, "", data.advance_refusal_count)
    if data.hint_level >= MAX_CANNOT_ANSWER_HINTS:
        return NextQuestionVerdict(True, "", data.advance_refusal_count)
    if data.current_question_follow_up_count >= max_follow_ups_per_question:
        return NextQuestionVerdict(True, "", data.advance_refusal_count)

    if data.advance_refusal_count >= MAX_ADVANCE_REFUSALS:
        return NextQuestionVerdict(True, "", data.advance_refusal_count)

    data.advance_refusal_count += 1
    return NextQuestionVerdict(
        False,
        (
            "The current question is not resolved yet — its learning outcome is "
            "still uncovered and you have follow-up budget left. Probe the "
            "candidate's answer or call request_hint before moving on."
        ),
        data.advance_refusal_count,
    )


def current_question_resolved(
    data: InterviewRuntimeStateData,
    *,
    current_outcome_id: str | None,
    max_follow_ups_per_question: int,
) -> bool:
    """True when the LIVE question has nothing left to give.

    The substantive half of :func:`resolve_next_question`, with no side effects and
    none of its escape hatches: "no question yet" and "time is nearly up" both let
    that gate stand open, which is right for a model ASKING permission and wrong as
    a trigger for the server advancing on its own.

    Pure, so the caller can evaluate it every turn without burning the refusal
    budget that ``resolve_next_question`` debits.
    """
    if not current_outcome_id:
        return False
    return (
        _is_ticked(data, current_outcome_id)
        or data.hint_level >= MAX_CANNOT_ANSWER_HINTS
        or data.current_question_follow_up_count >= max_follow_ups_per_question
    )


def reset_for_new_question(data: InterviewRuntimeStateData) -> None:
    """Clear the counters whose budget belongs to ONE question.

    Called when the interview advances. ``end_refusal_count`` is deliberately NOT
    cleared: ending is a decision about the SESSION, so resetting its budget per
    question would hand a model that keeps trying to quit a fresh argument on
    every question and the bound would stop bounding anything.

    ``advance_refusal_count`` must be here. Left as a session counter it means the
    gate refuses twice in the whole interview and then stands open, so the first
    question gets probed and every question after it can be rushed — the opposite
    of what the gate is for.
    """
    data.hint_level = 0
    data.reframe_count = 0
    data.current_question_follow_up_count = 0
    data.current_question_hint_refunds = 0
    data.advance_refusal_count = 0


@dataclass(frozen=True)
class HintGrant:
    granted: bool
    level: int
    is_final: bool


def resolve_hint_request(data: InterviewRuntimeStateData) -> HintGrant:
    """Advance the hint ladder for the CURRENT question, server-side.

    Returns the rung the model should scaffold at. The model is given the rung and
    the question, never text to reproduce verbatim: handing back a fixed string is
    what made hints read as boilerplate.
    """
    if data.hint_level >= MAX_CANNOT_ANSWER_HINTS:
        return HintGrant(False, data.hint_level, True)
    level = data.hint_level
    data.hint_level += 1
    return HintGrant(True, level, data.hint_level >= MAX_CANNOT_ANSWER_HINTS)


__all__ = [
    "MAX_ADVANCE_REFUSALS",
    "MAX_END_REFUSALS",
    "EndInterviewVerdict",
    "HintGrant",
    "NextQuestionVerdict",
    "OutcomeProgress",
    "ProgressReport",
    "build_progress_report",
    "build_turn_reminder",
    "current_question_resolved",
    "reset_for_new_question",
    "resolve_end_interview",
    "resolve_next_question",
    "resolve_hint_request",
]
