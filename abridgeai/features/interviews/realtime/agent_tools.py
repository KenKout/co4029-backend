"""LiveKit function tools that expose the interview's server-authoritative state.

This is the boundary the model cannot route around: the agent's words and rhythm
are its own, but ``next_question`` / ``request_hint`` / ``end_interview`` /
``get_progress`` here enforce what may be asked, how many hints exist, and
whether the interview may end — backed by the pure helpers in :mod:`tools` and
the persisted ``InterviewRuntimeStateData``.

Refusals are raised as ``ToolError`` so the model SEES the reason (the SDK hands
tool errors back to the LLM verbatim) — and the reason names the specific unmet
outcomes rather than a generic "keep going", so a cooperative model converges.
The pure logic in :mod:`tools` already bounds repeated refusals, so a stubborn
model cannot deadlock the session.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from livekit.agents import RunContext, ToolError, function_tool

from abridgeai.features.interviews.orchestrator.tools import (
    EndInterviewVerdict,
    NextQuestionVerdict,
    build_progress_report,
    reset_for_new_question,
    resolve_end_interview,
    resolve_hint_request,
    resolve_next_question,
)
from abridgeai.features.interviews.realtime import observability as obs

if TYPE_CHECKING:
    from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData

logger = logging.getLogger(__name__)

# Dataclass carrying the per-session runtime state + the question-selection
# context the tools need. Kept as the generic `InterviewUserdata` the agent
# session is typed on, mirroring the `hotel_receptionist` `Userdata` pattern:
# one mutable object, created per job, shared by every tool call.
_Userdata = object  # placeholder; the real generic is defined in session_runtime


_FINAL_HINT_NOTE = "— this is the FINAL hint; if the candidate still cannot answer, advance."


class InterviewToolsMixin:
    """Mix-in providing the interview tools on the agent.

    Follows the ``hotel_receptionist`` composition pattern: capability modules
    are mixins on the single agent class, so ONE chat context carries the whole
    conversation and every tool shares the same ``userdata``.
    """

    @function_tool
    async def interview_get_progress(self, ctx: RunContext[object]) -> str:
        """Read current interview progress: which outcomes are covered, what is
        still required, and how many questions/hints remain.

        Use this to decide whether to keep probing, request a hint, or advance.
        Returns a short JSON summary; it never contains answer content.
        """
        data = _runtime_state(ctx)
        report = build_progress_report(
            data,
            required_outcome_ids=_required_outcomes(ctx),
            questions_remaining=_questions_remaining(ctx),
            outcome_titles=_outcome_titles(ctx),
        )
        import json

        return json.dumps(report.to_dict(), ensure_ascii=False)

    @function_tool
    async def interview_next_question(self, ctx: RunContext[object]) -> str:
        """Advance to the NEXT question in the interview.

        The server chooses the question (you do not); this tool returns its text.
        Only allowed when the CURRENT question is resolved: its outcome is
        covered, the hint ladder or follow-up budget is spent, or no question is
        active. If refused, the error tells you what is still missing — probe the
        candidate further or call interview_request_hint.
        """
        data = _runtime_state(ctx)
        # The server may have advanced already (`native_advance`): the model calling
        # the tool for a transition that has happened must be handed the SAME
        # question, or one candidate transition spends two of the bank's questions.
        pending = _take_pending_question(ctx)
        if pending is not None:
            return (
                "Already advanced — the interview is on this question, targeting "
                f"outcome '{data.current_outcome_id}':\n{pending}"
            )
        verdict = resolve_next_question(
            data,
            current_outcome_id=data.current_outcome_id,
            questions_remaining=_questions_remaining(ctx),
            below_closing_threshold=_below_closing_threshold(ctx),
            max_follow_ups_per_question=_max_followups(ctx),
        )
        if not verdict.allowed:
            raise _refused(ctx, "interview_next_question", verdict)
        selected = _select_question(ctx)
        if selected is None:
            raise ToolError("No questions remain. Move to the closing exchange.")
        data.current_outcome_id = selected.outcome_id
        reset_for_new_question(data)
        # Read back by the turn grader (`native_runtime.fold_turn`) and the hint
        # ladder below. Left unwritten it pins both to question ONE for the whole
        # session.
        _set_current_question_text(ctx, selected.prompt_text)
        # The plain int on the userdata does not follow the pool on its own —
        # only `fold_turn` and `native_advance` recomputed it. Left stale here,
        # the snapshot published below said `total - remaining` for the PREVIOUS
        # question: the card moved to the new question while the header stayed
        # "Question 1 of 3" until the next graded answer.
        _sync_questions_remaining(ctx)
        # The client learns the question changed ONLY from here. The model's spoken
        # words arrive as transcription, which carries no question identity, so
        # without this the UI stays on the previous question card for the rest of
        # the interview.
        await _publish_state(ctx)
        return (
            f"Ask this question next, targeting outcome '{selected.outcome_id}':\n"
            f"{selected.prompt_text}"
        )

    @function_tool
    async def interview_request_hint(self, ctx: RunContext[object]) -> str:
        """Request the server to advance the hint ladder for the CURRENT question.

        Returns the escalation rung (0 = gentle nudge, 1 = break it into parts,
        2+ = walk through one entry point) plus the question text, so you can
        scaffold THAT question without inventing a generic one. When the ladder is
        spent the tool says so and you must either probe or call
        interview_next_question.
        """
        data = _runtime_state(ctx)
        grant = resolve_hint_request(data)
        question = _current_question_text(ctx)
        if not grant.granted:
            raise ToolError(
                "No hints remain for this question. Probe the candidate's answer "
                "or call interview_next_question."
            )
        # Mirror of the routed path's STUDENT_REQUESTED_HINT exemption
        # (turn_state.py: a hint request "does not consume the academic probe
        # budget"). On the native path `fold_turn` charges one follow-up for
        # EVERY turn that does not advance — including a typed hint request,
        # which arrives as an `answer` turn when the candidate types it instead
        # of using a hint button. Without this refund the follow-up budget (2)
        # exhausts before the hint ladder (3), the question advances with no
        # transition, and the candidate never gets the rungs they asked for.
        #
        # Bounded to ONE per question: the model also calls this tool on its
        # own scaffolding instinct, and an unbounded refund let those calls
        # cancel the budget every turn — the count ping-ponged below its
        # threshold and the question could never auto-advance.
        if (
            data.current_question_follow_up_count > 0
            and data.current_question_hint_refunds < 1
        ):
            data.current_question_follow_up_count -= 1
            data.current_question_hint_refunds += 1
        # The next thing the model says IS the hint: mark it for the transcript
        # recorder (so a reload renders kind="hint", not FOLLOW-UP) and tell the
        # client (so the live utterance gets the HINT badge). A refused hint
        # sets neither — the refusal speech is an ordinary probe.
        ctx.userdata.pending_assistant_kind = "hint"  # type: ignore[attr-defined]
        await ctx.userdata.publish_agent_action("hint")  # type: ignore[attr-defined]
        if question:
            return (
                f"Hint rung {grant.level} for the current question. "
                f"Scaffold this question specifically: {question} "
                f"{_FINAL_HINT_NOTE if grant.is_final else ''}"
            )
        return f"Hint rung {grant.level} — scaffold the current question specifically."

    @function_tool
    async def interview_end_interview(self, ctx: RunContext[object]) -> str:
        """End the interview and submit it for evaluation.

        Refused while required outcomes remain uncovered AND questions remain AND
        time is left — the error names the missing outcomes. When allowed, this
        finalizes the session; say a short closing line to the candidate first.
        """
        data = _runtime_state(ctx)
        verdict = resolve_end_interview(
            data,
            required_outcome_ids=_required_outcomes(ctx),
            questions_remaining=_questions_remaining(ctx),
            below_closing_threshold=_below_closing_threshold(ctx),
            outcome_titles=_outcome_titles(ctx),
        )
        if not verdict.allowed:
            raise _refused(ctx, "interview_end_interview", verdict)
        await _finalize_session(ctx)
        return "Interview finalized and submitted for evaluation. Deliver the closing message now."


# ── wiring helpers: everything below is a small seam between the generic
#    RunContext and the persisted runtime state. The values are resolved from
#    the per-session `userdata`, which the session runtime populates. These
#    read-only getters keep the mixin readable and the mapping in one place.


def _runtime_state(ctx: RunContext[object]) -> InterviewRuntimeStateData:
    return ctx.userdata.state  # type: ignore[attr-defined]


def _required_outcomes(ctx: RunContext[object]) -> list[str]:
    return list(ctx.userdata.required_outcome_ids)  # type: ignore[attr-defined]


def _outcome_titles(ctx: RunContext[object]) -> dict[str, str]:
    return dict(ctx.userdata.outcome_titles)  # type: ignore[attr-defined]


def _questions_remaining(ctx: RunContext[object]) -> int:
    return int(ctx.userdata.questions_remaining)  # type: ignore[attr-defined]


def _below_closing_threshold(ctx: RunContext[object]) -> bool:
    return bool(ctx.userdata.below_closing_threshold)  # type: ignore[attr-defined]


def _max_followups(ctx: RunContext[object]) -> int:
    return int(ctx.userdata.max_follow_ups_per_question)  # type: ignore[attr-defined]


def _current_question_text(ctx: RunContext[object]) -> str | None:
    return ctx.userdata.current_question_text  # type: ignore[attr-defined]


def _set_current_question_text(ctx: RunContext[object], text: str) -> None:
    ctx.userdata.current_question_text = text  # type: ignore[attr-defined]


def _take_pending_question(ctx: RunContext[object]) -> str | None:
    """Consume a server-side advance, returning the question it selected.

    Consuming rather than peeking: the flag exists to make ONE model call after a
    server advance a no-op, not to disable advancing for the rest of the turn.
    """
    userdata = ctx.userdata
    if not getattr(userdata, "pending_new_question", False):
        return None
    userdata.pending_new_question = False  # type: ignore[attr-defined]
    return _current_question_text(ctx)


def _select_question(ctx: RunContext[object]) -> object | None:
    # The selection scorer lives server-side (selection.py). The userdata carries
    # a small callable/selector handle so the tool stays thin and the pure
    # selection logic stays where the property tests can reach it.
    selector = ctx.userdata.select_next  # type: ignore[attr-defined]
    return selector()


def _sync_questions_remaining(ctx: RunContext[object]) -> None:
    selector = ctx.userdata.select_next  # type: ignore[attr-defined]
    ctx.userdata.questions_remaining = selector.remaining()  # type: ignore[attr-defined]


async def _finalize_session(ctx: RunContext[object]) -> None:
    finalizer = ctx.userdata.finalize_session  # type: ignore[attr-defined]
    await finalizer()


async def _publish_state(ctx: RunContext[object]) -> None:
    publish = ctx.userdata.publish_state  # type: ignore[attr-defined]
    await publish()


def _refused(
    ctx: RunContext[object],
    tool: str,
    verdict: EndInterviewVerdict | NextQuestionVerdict,
) -> ToolError:
    """Record a refusal, then hand the model the reason.

    The native path emits no ``ReasonCode`` of its own, so without this a refusal
    is invisible: "the interview stayed on question one" looks identical whether
    the gate refused, the model never asked, or grading was broken. The refusal
    count is the load-bearing field — it is what the anti-deadlock bound counts,
    so a session approaching that bound is visible before it trips.
    """
    obs.emit(
        obs.EV_TOOL_REFUSED,
        session_id=_session_id(ctx),
        tool=tool,
        refusal_count=verdict.refusal_count,
    )
    return ToolError(verdict.message)


def _session_id(ctx: RunContext[object]) -> object:
    return getattr(ctx.userdata, "interview_session_id", "unknown")


__all__ = ["InterviewToolsMixin"]
