"""DB-side setup for the native interview agent — no LiveKit imports.

Builds the one :class:`InterviewUserdata` a job runs on, wires its two injected
callables, and stops there. Split from :mod:`native_runtime` so the whole
question-selection and finalization path can be tested with a database and no
room, and so the LiveKit surface stays in one file.

Everything here is a composition of code that already exists:
``turn_perception.load_candidates`` for the bank, ``selection.select_next_question``
for the pick, ``repository.load_readonly`` for the runtime state, and
``orchestration_bridge.finalize_session`` for the submit. Nothing re-implements a
rule that lives elsewhere — the point of the native agent is a different
conversation loop, not a second set of policies.

Read-only on purpose: setup never writes runtime state. ``load_or_init`` would
create a row this short-lived transaction is not the owner of, and the native
path has no save step yet (see the module TODO seam in the report) — a row
created here and never updated reads as a session that made no progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.db import get_sessionmaker
from abridgeai.features.interviews.orchestrator import repository as state_repo
from abridgeai.features.interviews.orchestrator import turn_perception
from abridgeai.features.interviews.orchestrator.coverage import is_provisionally_sufficient
from abridgeai.features.interviews.orchestrator.decision import (
    DEFAULT_MAX_FOLLOWUPS_PER_QUESTION,
    MAX_CANNOT_ANSWER_HINTS,
    DecisionInputs,
)
from abridgeai.features.interviews.orchestrator.interviewer_identity import identity_from_config
from abridgeai.features.interviews.orchestrator.role_question_filter import (
    filter_candidates_by_role,
)
from abridgeai.features.interviews.orchestrator.selection import (
    SelectionContext,
    select_next_question,
)
from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData
from abridgeai.features.interviews.orchestrator.turn_state import sync_question_history
from abridgeai.features.interviews.realtime import orchestration_bridge as bridge
from abridgeai.features.interviews.realtime.agent_userdata import InterviewUserdata
from abridgeai.features.interviews.services.taking import session_time_remaining_seconds
from abridgeai.features.interviews.workers.analysis import RECONCILE_TURN_ANALYSIS_TASK

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from abridgeai.features.interviews.orchestrator.selection import CandidateQuestion
    from abridgeai.features.interviews.orchestrator.state import InterviewRuntimeStateData
    from abridgeai.features.interviews.orchestrator.sufficiency import SufficiencyVerdict

logger = logging.getLogger(__name__)

# The fraction of remaining time below which the interview must head for the
# closing. Read off ``DecisionInputs`` rather than restated, so the native path
# and the routed path cannot drift apart on the one number that decides whether
# ending is allowed. NOT ``decision._LOW_TIME_FRACTION`` (0.2), which is the
# looser "stop probing" threshold.
_CLOSING_TIME_FRACTION: float = DecisionInputs.closing_time_fraction


class NativeSetupError(RuntimeError):
    """The session or its config is missing, so no interview can be run."""


@dataclass(frozen=True)
class SelectedBankQuestion:
    """One question the server chose, in the shape ``agent_tools`` expects."""

    outcome_id: str
    prompt_text: str


@dataclass(frozen=True)
class BankSelector:
    """The deterministic scorer, bound to one session's LIVE runtime state.

    Callable so it satisfies ``InterviewUserdata.select_next``, which must be
    synchronous — the tool that calls it is already inside a turn and cannot
    afford a database round trip mid-reply. The candidate pool and prompt texts
    are therefore snapshotted at setup; only the state read on each call is live.

    Marking the pick as asked is this class's job, not the tool's: the scorer
    filters on ``asked_question_ids``, so without the append the same question
    scores highest forever and the model is handed question one repeatedly.
    """

    candidates: list[CandidateQuestion]
    prompts: dict[str, str]
    state: InterviewRuntimeStateData
    required_outcome_ids: tuple[str, ...]
    time_fraction_remaining: float | None

    def _context(self) -> SelectionContext:
        points = {oid: cov.coverage_points for oid, cov in self.state.outcome_coverage.items()}
        return SelectionContext(
            asked_question_ids=frozenset(self.state.asked_question_ids),
            skipped_question_ids=frozenset(self.state.skipped_question_ids),
            outcome_evidence_counts=points,
            uncovered_required_outcome_ids=frozenset(
                oid
                for oid in self.required_outcome_ids
                if not is_provisionally_sufficient(points.get(oid, 0))
            ),
            time_fraction_remaining=self.time_fraction_remaining,
            last_targeted_outcome_id=self.state.current_outcome_id,
        )

    def total(self) -> int:
        """The size of this session's question pool.

        The one source for the counter's denominator. `asked + remaining` is NOT
        that: `asked_question_ids` also carries ids merged from the REST transcript
        that may not be in the filtered pool, so it grows while `remaining` does
        not shrink — which is how the header walked from "1 of 3" to "of 4".
        """
        return len(self.candidates)

    def remaining(self) -> int:
        """How many bank questions are still askable right now."""
        ctx = self._context()
        return sum(
            1
            for candidate in self.candidates
            if candidate.question_id not in ctx.asked_question_ids
            and candidate.question_id not in ctx.skipped_question_ids
        )

    def last_picked_id(self) -> UUID | None:
        """The question the interview is currently on, for the deferred analysis.

        Read from runtime state rather than a field of its own so a rejoin — which
        rebuilds this selector but reloads the same state — still knows which
        question a turn belongs to.
        """
        raw = self.state.current_question_id
        if not raw:
            return None
        try:
            return UUID(str(raw))
        except (ValueError, AttributeError, TypeError):
            return None

    def __call__(self) -> SelectedBankQuestion | None:
        scored = select_next_question(self.candidates, self._context())
        if scored is None:
            return None
        question_id = scored.candidate.question_id
        # Guarded: `remaining()` filters on a frozenset of these ids, so a repeated
        # append does not reduce the remaining count but does inflate any length-based
        # count of asked questions — which is how the header drifted from "1 of 3" to
        # "3 of 4" mid-interview. `sync_question_history` merges the REST transcript's
        # ids into this same list, so a collision is expected, not hypothetical.
        if question_id not in self.state.asked_question_ids:
            self.state.asked_question_ids.append(question_id)
        self.state.current_question_id = question_id
        # A question with no linked outcome yields "" rather than None, because
        # the tool assigns this straight onto ``current_outcome_id``. The gates
        # then see an outcome that can never tick, so advancing falls through to
        # the follow-up budget and the bounded refusal counter — correct, if
        # blunter than a linked question.
        return SelectedBankQuestion(
            outcome_id=str(scored.candidate.linked_outcome_id or ""),
            prompt_text=self.prompts.get(question_id, ""),
        )


@dataclass
class NativeSetup:
    """Everything one native job needs, resolved before the room starts.

    ``close_session`` returns the closing text to speak; ``userdata.finalize_session``
    is the same submit with the return dropped, because ``agent_tools`` types it
    as returning None. Both land in ``orchestration_bridge.finalize_session``.
    """

    userdata: InterviewUserdata
    language: str
    input_mode: str
    close_session: Callable[[], Awaitable[str | None]]
    selector: BankSelector
    tts_voice: str | None = None
    interviewer_name: str | None = None
    onboarding_turns: list[tuple[str, str]] = field(default_factory=list)
    time_remaining_seconds: int | None = None
    # Grade one answer and persist the turn's state mutations. Injected rather
    # than imported so `native_runtime` stays free of DB/gateway concerns, and so
    # the ordering guarantee (grade BEFORE the note is built) is testable.
    grade_turn: Callable[..., Awaitable[None]] | None = None
    # Persist runtime state on its own, for the mutations no grade follows: a tool
    # advancing the question or ending the session. Without it those survive only
    # as a side effect of the NEXT graded answer, so a rejoin right after an
    # advance re-asks the question the candidate was already moved past.
    save_state: Callable[[], Awaitable[None]] | None = None


class StateWriter:
    """Persists the ONE runtime-state object the tools and the grader mutate.

    Two bugs it exists to close, both of which made a rejoin start over:

    * The previous save was a no-op whenever the session had no
      ``interview_runtime_state`` row yet — it called ``load_readonly`` and
      returned on ``None``. Nothing was ever written, so coverage, the hint
      ladder and the refusal counters did not survive a rejoin.
    * It re-read the version immediately before saving and passed that as
      ``expected_version``, so the optimistic check could never fail and a
      concurrent write from the out-of-process re-analysis worker was silently
      overwritten.

    Holding the version between saves makes a foreign write detectable. When one
    happens the in-room state still wins — it is what the interviewer is acting
    on right now, and the final grade is re-judged post-session from the
    transcript anyway — but it is logged rather than silent.
    """

    def __init__(self, session_id: UUID, state: InterviewRuntimeStateData) -> None:
        self._session_id = session_id
        self._state = state
        self._version: int | None = None

    def adopt_version(self, version: int) -> None:
        self._version = version

    async def save(self) -> None:
        async with get_sessionmaker()() as db:
            loaded = await state_repo.load_or_init(db, self._session_id)
            expected = self._version if self._version is not None else loaded.version
            if expected != loaded.version:
                logger.warning(
                    "runtime state changed underneath the agent; in-room state wins "
                    "(session=%s, held=%s, found=%s)",
                    self._session_id,
                    expected,
                    loaded.version,
                )
                expected = loaded.version
            try:
                self._version = await state_repo.save(
                    db, self._session_id, self._state, expected_version=expected
                )
            except state_repo.StaleStateError:
                logger.warning(
                    "runtime state save lost a race; will re-sync next turn (session=%s)",
                    self._session_id,
                )
                self._version = None
                return
            await db.commit()


async def _submit_and_discard_closing(session_id: UUID, student_id: UUID, *, language: str) -> None:
    """``finalize_session`` adapted to the tool's ``-> None`` contract.

    The model delivers its own closing in its own words (the instructions tell it
    to), so the canonical ceremony text is written but not spoken here. The hard
    stop is the path that speaks it, because there is no model turn left to.
    """
    await bridge.finalize_session(session_id, student_id, language=language, reason="natural")


def _make_turn_grader(
    session_id: UUID,
    student_id: UUID,
    *,
    question_id_getter: Callable[[], UUID | None],
    writer: StateWriter,
) -> Callable[..., Awaitable[None]]:
    """Bind the fast probe, the coverage fold and the deferred re-analysis together.

    Returns the callable ``NativeSetup.grade_turn`` expects. Each turn opens its
    own session: the agent runs outside any request scope, and holding one open
    across a conversation would pin a connection for the whole interview.

    Persistence goes through the shared :class:`StateWriter` rather than a local
    save, so the grader and the tools cannot disagree about which version they
    hold — see that class for the two bugs a local save reintroduces.
    """

    async def _grade(
        *,
        state: InterviewRuntimeStateData,
        answer_text: str,
        question_text: str,
        turn_id: str,
    ) -> None:
        from abridgeai.core.db import get_sessionmaker  # noqa: PLC0415
        from abridgeai.features.interviews.orchestrator.sufficiency_logic import (  # noqa: PLC0415
            probe_sufficiency,
        )
        from abridgeai.features.interviews.realtime.native_grading import (  # noqa: PLC0415
            grade_native_turn,
        )

        async with get_sessionmaker()() as db:

            async def _probe(
                *,
                question_text: str,
                answer_text: str,
                outcome_id: str | None,
                allowed_other: tuple[str, ...],
            ) -> SufficiencyVerdict:
                del allowed_other  # emergent-evidence ids are not offered here yet
                return await probe_sufficiency(
                    db,
                    question_text=question_text,
                    student_answer=answer_text,
                    outcome_id=outcome_id,
                )

            async def _enqueue(*, turn_id: str, probe_verdict: dict[str, Any]) -> None:
                question_id = question_id_getter()
                if question_id is None:
                    # Nothing to re-analyse against; the probe's fold stands.
                    return
                pool = await bridge._get_arq_pool()  # noqa: SLF001
                if pool is None:
                    return
                await pool.enqueue_job(
                    RECONCILE_TURN_ANALYSIS_TASK,
                    str(student_id),
                    str(session_id),
                    {
                        "question_id": str(question_id),
                        "turn_id": turn_id,
                        "probe_verdict": probe_verdict,
                    },
                )

            await grade_native_turn(
                state=state,
                answer_text=answer_text,
                question_text=question_text,
                turn_id=turn_id,
                probe=_probe,
                enqueue_reconcile=_enqueue,
                save_state=writer.save,
            )

    return _grade


async def load_native_setup(
    session_id: UUID, student_id: UUID, *, language: str = "en"
) -> NativeSetup:
    """Resolve the session's bank, outcomes, budgets and runtime state."""
    from abridgeai.features.interviews.models import (  # noqa: PLC0415
        InterviewConfig,
        InterviewSession,
    )

    async with get_sessionmaker()() as db:
        session = await db.get(InterviewSession, session_id)
        if session is None:
            raise NativeSetupError(f"interview session {session_id} not found")
        config = await db.get(InterviewConfig, session.interview_config_id)
        if config is None:
            raise NativeSetupError(f"interview config for session {session_id} not found")

        loaded = await state_repo.load_readonly(db, session_id)
        state = loaded.data if loaded is not None else InterviewRuntimeStateData()
        loaded_version = loaded.version if loaded is not None else None

        candidates, orm_by_id = await turn_perception.load_candidates(db, config.id)
        if getattr(config, "generation_variant_strategy", None) == "role_only":
            candidates = filter_candidates_by_role(
                candidates,
                identity_from_config(getattr(config, "persona_profile_json", None)).role,
                strict=True,
            )
            orm_by_id = {
                question_id: question
                for question_id, question in orm_by_id.items()
                if any(candidate.question_id == UUID(question_id) for candidate in candidates)
            }
        asked_ids = await turn_perception.persisted_question_ids(db, session_id)
        # Merges the REST transcript into state that the agent path never wrote,
        # so the scorer cannot re-offer a question the candidate already saw.
        sync_question_history(
            state,
            asked_ids,
            current_question=orm_by_id.get(str(asked_ids[-1])) if asked_ids else None,
        )

        outcomes = await turn_perception.list_outcomes(db, config.id)
        time_fraction = turn_perception.time_fraction_remaining(session, config)
        remaining_seconds = await session_time_remaining_seconds(db, session)
        session_language = getattr(session, "interview_language", language) or language
        input_mode = session.input_mode
        identity = identity_from_config(getattr(config, "persona_profile_json", None))
        tts_voice = getattr(config, "tts_voice", None)

    # Every config outcome is required — ``InterviewOutcome`` carries no
    # required flag, and ``adaptive.run_adaptive_turn`` reads them the same way.
    required_outcome_ids = tuple(str(outcome.id) for outcome in outcomes)
    selector = BankSelector(
        candidates=candidates,
        prompts={question_id: question.prompt_text for question_id, question in orm_by_id.items()},
        state=state,
        required_outcome_ids=required_outcome_ids,
        time_fraction_remaining=time_fraction,
    )

    userdata = InterviewUserdata(
        interview_session_id=session_id,
        student_id=student_id,
        language=session_language,
        state=state,
        required_outcome_ids=list(required_outcome_ids),
        outcome_titles={str(outcome.id): outcome.outcome_text for outcome in outcomes},
        questions_remaining=selector.remaining(),
        questions_total=selector.total(),
        # Teacher-configured budgets (Phase 16). The defaults mirror the
        # pre-config constants so an old row that somehow lacks values — or a
        # test constructing a config-less session — keeps the shipped behaviour.
        max_follow_ups_per_question=(
            getattr(config, "max_follow_ups_per_question", None)
            if getattr(config, "max_follow_ups_per_question", None) is not None
            else DEFAULT_MAX_FOLLOWUPS_PER_QUESTION
        ),
        max_hints_per_question=(
            getattr(config, "max_hints_per_question", None)
            if getattr(config, "max_hints_per_question", None) is not None
            else MAX_CANNOT_ANSWER_HINTS
        ),
        below_closing_threshold=(
            time_fraction is not None and time_fraction <= _CLOSING_TIME_FRACTION
        ),
        current_question_text=await bridge.get_current_question_text(
            session_id, language=session_language
        ),
        select_next=selector,
        finalize_session=partial(
            _submit_and_discard_closing, session_id, student_id, language=session_language
        ),
        time_remaining_seconds=remaining_seconds,
    )
    writer = StateWriter(session_id, state)
    if loaded_version is not None:
        writer.adopt_version(loaded_version)
    return NativeSetup(
        userdata=userdata,
        language=session_language,
        input_mode=input_mode,
        close_session=partial(
            bridge.finalize_session,
            session_id,
            student_id,
            language=session_language,
            reason="timed_out",
        ),
        selector=selector,
        tts_voice=tts_voice,
        interviewer_name=identity.name,
        onboarding_turns=await bridge.get_onboarding_turns(session_id),
        time_remaining_seconds=remaining_seconds,
        grade_turn=_make_turn_grader(
            session_id,
            student_id,
            question_id_getter=selector.last_picked_id,
            writer=writer,
        ),
        save_state=writer.save,
    )


__all__ = [
    "BankSelector",
    "NativeSetup",
    "NativeSetupError",
    "SelectedBankQuestion",
    "load_native_setup",
]
