"""Unit tests for the practice/assessment split.

Two properties are worth pinning here, and they fail in opposite directions.

The first is the bank partition. Its obvious purpose is to stop a rehearsal
revealing the graded questions, and that is the direction people remember to
test. The dangerous direction is the other one: the evaluator loads questions
across *every* review state and turns them into ``expected_question_ids``, so an
unfiltered practice question is counted as unanswered and hard-fails the outcome
linked to it. Unfiltered, the partition corrupts real grades rather than merely
leaking a bank — so both directions are asserted, and the evaluator's filter is
asserted separately from the selection paths.

The second is that "ungraded" is a real state rather than an absent one. A
terminal session with a NULL ``pass_verdict`` is what the stalled-evaluation
sweep re-enqueues, so practice is only safe while ``session_mode`` distinguishes
the two.

A rehearsal is still judged — against the criteria the student saw beforehand —
just never graded, and that runs through a separate function on a separate ARQ
task. Two functions each refusing the other's mode is what makes "practice can
never write ``pass_verdict``" structural rather than a conditional someone could
invert; ``pass_verdict = TRUE`` is what opens the SR-lesson and quiz gates. Both
refusals are asserted below, along with the fail-safe direction of the mode
helper: anything we cannot positively identify as practice gets graded.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from abridgeai.features.interviews import practice
from abridgeai.features.interviews.orchestrator import turn_perception
from abridgeai.features.interviews.queries import authoring as authoring_queries
from abridgeai.features.interviews.services import evaluation as evaluation_service
from abridgeai.features.interviews.services import taking as taking_service

# --------------------------------------------------------------------------- #
# The mode helper — the fail-safe direction matters more than the happy path
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value", [None, "", "assessment", "Practice", "graded", "unknown"])
def test_only_the_exact_practice_literal_reads_as_practice(value: str | None) -> None:
    """Anything we cannot positively identify is graded.

    Chosen deliberately over the reverse: grading a run that should have been a
    rehearsal is recoverable and visible, while silently discarding a real
    attempt is neither.
    """
    assert practice.is_practice(value) is False


def test_practice_literal_reads_as_practice() -> None:
    assert practice.is_practice(practice.MODE_PRACTICE) is True
    assert practice.partition_for_mode(practice.MODE_PRACTICE) is True
    assert practice.partition_for_mode(practice.MODE_ASSESSMENT) is False


def test_practice_attempt_number_sits_outside_the_real_attempt_space() -> None:
    """Real attempts are allocated as ``MAX(...) + 1``, so they start at 1.

    If someone renumbers practice into that space, ``uq_interview_sessions_number``
    starts rejecting a student's first real attempt after a rehearsal.
    """
    assert practice.PRACTICE_ATTEMPT_NUMBER < 1


# --------------------------------------------------------------------------- #
# The query actually emits the filter
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("partition", [True, False])
async def test_partition_reaches_the_compiled_sql(partition: bool) -> None:
    """Assert the emitted WHERE clause, not just that a kwarg was accepted.

    A filter that is threaded through every call site but silently dropped when
    building the statement would pass every mock-level test in this file.
    """
    captured: dict[str, object] = {}

    class _Result:
        @staticmethod
        def scalars() -> object:
            return SimpleNamespace(all=list)

    async def _execute(stmt: object) -> object:
        captured["sql"] = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        return _Result()

    db = SimpleNamespace(execute=_execute)
    await authoring_queries.list_questions_for_config(
        db, uuid4(), review_status="approved", practice_only=partition
    )

    sql = str(captured["sql"]).lower()
    assert "practice_only" in sql
    assert ("practice_only = true" if partition else "practice_only = false") in sql


@pytest.mark.asyncio
async def test_no_partition_kwarg_emits_no_partition_filter() -> None:
    """The default has to stay "both", or teacher-facing listings lose half the
    bank the moment this column exists."""
    captured: dict[str, object] = {}

    class _Result:
        @staticmethod
        def scalars() -> object:
            return SimpleNamespace(all=list)

    async def _execute(stmt: object) -> object:
        captured["sql"] = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        return _Result()

    await authoring_queries.list_questions_for_config(SimpleNamespace(execute=_execute), uuid4())

    # The column is always in the SELECT list (the query loads whole rows), so
    # the assertion has to be about the predicate, not the string.
    where = str(captured["sql"]).lower().split(" where ", 1)
    assert len(where) == 1 or "practice_only" not in where[1]


# --------------------------------------------------------------------------- #
# Every path that picks a question for a running session passes the partition
# --------------------------------------------------------------------------- #


def _capture_list_questions() -> tuple[AsyncMock, object]:
    mock = AsyncMock(return_value=[])
    return mock, patch.object(authoring_queries, "list_questions_for_config", mock)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [(practice.MODE_ASSESSMENT, False), (practice.MODE_PRACTICE, True)],
)
async def test_adaptive_candidate_pool_is_partitioned(mode: str, expected: bool) -> None:
    """``SelectionContext`` exposes only asked/skipped exclusions, so there is
    nowhere downstream to drop a partition — it has to happen at load time."""
    mock, patcher = _capture_list_questions()
    with patcher, patch.object(turn_perception, "list_outcomes", AsyncMock(return_value=[])):
        await turn_perception.load_candidates(MagicMock(), uuid4(), session_mode=mode)

    assert mock.await_args.kwargs["practice_only"] is expected
    assert mock.await_args.kwargs["review_status"] == "approved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [(practice.MODE_ASSESSMENT, False), (practice.MODE_PRACTICE, True)],
)
async def test_first_question_is_partitioned(mode: str, expected: bool) -> None:
    mock, patcher = _capture_list_questions()
    with patcher:
        await taking_service._first_published_question(MagicMock(), uuid4(), session_mode=mode)

    assert mock.await_args.kwargs["practice_only"] is expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "expected"),
    [(practice.MODE_ASSESSMENT, False), (practice.MODE_PRACTICE, True)],
)
async def test_legacy_sequential_path_is_partitioned(mode: str, expected: bool) -> None:
    """The legacy path is not a dead branch — every adaptive failure rolls back
    to it, so leaving it unfiltered leaks the bank on the error path only."""
    mock, patcher = _capture_list_questions()
    with patcher:
        await taking_service._next_published_question_after(
            MagicMock(), uuid4(), set(), session_mode=mode
        )

    assert mock.await_args.kwargs["practice_only"] is expected


@pytest.mark.asyncio
async def test_publish_gate_counts_only_gradable_questions() -> None:
    """A config whose approved questions are all practice-only has nothing to
    assess with; counting the whole bank would let it publish anyway."""
    mock, patcher = _capture_list_questions()
    from abridgeai.features.interviews.services import authoring as authoring_service

    with (
        patcher,
        patch.object(
            authoring_service,
            "_require_config",
            AsyncMock(return_value=SimpleNamespace(status="draft")),
        ),
        pytest.raises(Exception, match="interview_no_approved_questions"),
    ):
        await authoring_service.publish_interview_config(MagicMock(), uuid4(), MagicMock())

    assert mock.await_args.kwargs["practice_only"] is False


# --------------------------------------------------------------------------- #
# Choosing the mode: three gates, all of which raise rather than downgrade
# --------------------------------------------------------------------------- #


async def _resolve(*, requested: str | None, config: object, questions: int, used: int) -> str:
    with (
        patch.object(taking_service, "count_practice_questions", AsyncMock(return_value=questions)),
        patch.object(taking_service, "count_practice_runs_used", AsyncMock(return_value=used)),
    ):
        return await taking_service._resolve_session_mode(
            MagicMock(),
            config=config,
            config_id=uuid4(),
            student_id=uuid4(),
            requested={} if requested is None else {"session_mode": requested},
        )


@pytest.mark.asyncio
async def test_omitted_mode_resolves_to_assessment() -> None:
    """An old client that has never heard of this field must keep being graded."""
    mode = await _resolve(
        requested=None, config=SimpleNamespace(practice_mode_enabled=True), questions=5, used=0
    )
    assert mode == practice.MODE_ASSESSMENT


@pytest.mark.asyncio
async def test_practice_is_granted_when_all_three_gates_pass() -> None:
    mode = await _resolve(
        requested="practice",
        config=SimpleNamespace(practice_mode_enabled=True),
        questions=2,
        used=0,
    )
    assert mode == practice.MODE_PRACTICE


@pytest.mark.asyncio
async def test_practice_is_refused_when_the_teacher_has_not_enabled_it() -> None:
    with pytest.raises(taking_service.InterviewPracticeUnavailable) as exc:
        await _resolve(
            requested="practice",
            config=SimpleNamespace(practice_mode_enabled=False),
            questions=5,
            used=0,
        )
    assert exc.value.reason == "not_enabled"


@pytest.mark.asyncio
async def test_practice_is_refused_when_the_partition_is_empty() -> None:
    """Enabled but unusable is its own reason, not a generic failure.

    The fix belongs to the teacher, and the student cannot act on it — telling
    them apart is the difference between a support ticket and a shrug.
    """
    with pytest.raises(taking_service.InterviewPracticeUnavailable) as exc:
        await _resolve(
            requested="practice",
            config=SimpleNamespace(practice_mode_enabled=True),
            questions=0,
            used=0,
        )
    assert exc.value.reason == "no_practice_questions"


@pytest.mark.asyncio
async def test_practice_ceiling_is_separate_from_the_attempt_ceiling() -> None:
    """A used-up rehearsal must not raise the "out of attempts" error.

    They are different exceptions on purpose: no graded attempt was consumed, so
    reporting an attempt ceiling here would be a lie to the student.
    """
    with pytest.raises(taking_service.InterviewPracticeLimitReached):
        await _resolve(
            requested="practice",
            config=SimpleNamespace(practice_mode_enabled=True),
            questions=3,
            used=practice.PRACTICE_MAX_RUNS,
        )


# --------------------------------------------------------------------------- #
# Practice feedback: judged, but never graded
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "task"),
    [
        (practice.MODE_ASSESSMENT, "evaluate_interview_session_task"),
        (practice.MODE_PRACTICE, "generate_practice_feedback_task"),
    ],
)
async def test_finishing_queues_the_task_that_matches_the_mode(mode: str, task: str) -> None:
    """One task writes verdicts, the other cannot. Picking the wrong one for a
    rehearsal would either grade it or leave it with no feedback at all."""
    session = SimpleNamespace(
        id=uuid4(),
        interview_config_id=uuid4(),
        student_id=uuid4(),
        status="in_progress",
        session_mode=mode,
        interview_language="en",
        ended_at=None,
        # This test is about mode routing, so the run must be gradeable at all:
        # a session that never reached the assessment is abandoned and enqueues
        # nothing regardless of mode.
        assessment_started_at=datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
    )
    count_result = MagicMock()
    count_result.scalar_one.return_value = 0
    db = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(time_limit_minutes=30)),
        execute=AsyncMock(return_value=count_result),
        commit=AsyncMock(),
        refresh=AsyncMock(),
    )
    arq = SimpleNamespace(enqueue_job=AsyncMock(return_value=object()))

    with (
        patch.object(taking_service, "_require_session", AsyncMock(return_value=session)),
        patch.object(taking_service, "_assert_owns_session"),
        patch.object(taking_service, "ensure_ceremony_message", AsyncMock()),
    ):
        await taking_service.submit_session(
            db, session.id, SimpleNamespace(user_id=session.student_id), arq_pool=arq
        )

    assert arq.enqueue_job.await_count == 1
    assert arq.enqueue_job.await_args.args[0] == task


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "task"),
    [
        (practice.MODE_ASSESSMENT, "evaluate_interview_session_task"),
        (practice.MODE_PRACTICE, "generate_practice_feedback_task"),
    ],
)
async def test_sweep_queues_the_task_that_matches_the_mode(mode: str, task: str) -> None:
    """The stale-session sweep is a second finish path, and it has to make the
    same choice. A swept rehearsal answered real questions; sending it to the
    grader would write a verdict, sending it nowhere would lose the feedback."""
    from datetime import timedelta

    from abridgeai.core.security import utcnow
    from abridgeai.features.interviews.services import lifecycle as lifecycle_service

    stale = utcnow() - timedelta(hours=2)
    session = SimpleNamespace(
        id=uuid4(),
        student_id=uuid4(),
        started_at=stale,
        assessment_started_at=stale,
        status="in_progress",
        session_mode=mode,
        ended_at=None,
    )
    arq = SimpleNamespace(enqueue_job=AsyncMock(return_value=object()))

    with (
        patch.object(
            lifecycle_service.sessions_queries,
            "list_in_progress_voice_sessions_with_limit",
            AsyncMock(return_value=[(session, 1)]),
        ),
        patch.object(
            lifecycle_service.sessions_queries, "count_user_messages", AsyncMock(return_value=2)
        ),
    ):
        await lifecycle_service.sweep_stale_voice_sessions(SimpleNamespace(commit=AsyncMock()), arq)

    assert arq.enqueue_job.await_count == 1
    assert arq.enqueue_job.await_args.args[0] == task


@pytest.mark.asyncio
async def test_evaluator_refuses_a_run_that_never_reached_the_assessment() -> None:
    """Defence in depth against grading onboarding chatter.

    ``evaluate_and_generate_report`` is the only writer of ``pass_verdict``, so
    the refusal lives here as well as at the enqueue site — the same belt-and-
    braces reasoning as the practice-mode refusal above. A session still in
    identity check / audio check / readiness has no answers, and grading one
    fabricated outcome verdicts and a pass/fail against a student who never saw
    a question (14 such rows in production before this guard).
    """
    session = SimpleNamespace(
        id=uuid4(),
        session_mode=practice.MODE_ASSESSMENT,
        interview_config_id=uuid4(),
        student_id=uuid4(),
        assessment_started_at=None,
    )
    outcomes = AsyncMock(return_value=[])
    db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock(), add=MagicMock())

    with (
        patch.object(
            evaluation_service.sessions_queries, "get_session", AsyncMock(return_value=session)
        ),
        patch.object(evaluation_service.authoring_queries, "list_outcomes_for_config", outcomes),
    ):
        await evaluation_service.evaluate_and_generate_report(db, session.id)

    # Bailed before touching outcomes, and wrote nothing at all.
    outcomes.assert_not_awaited()
    db.commit.assert_not_awaited()
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_practice_feedback_refuses_graded_sessions() -> None:
    """The mirror of the evaluator's refusal.

    Two functions that each reject the other's mode is what makes "a rehearsal
    can never write pass_verdict" structural rather than conditional. A graded
    session reaching here would get outcome rows without the verdict, score and
    gap report that give them meaning.
    """
    session = SimpleNamespace(
        id=uuid4(),
        session_mode=practice.MODE_ASSESSMENT,
        interview_config_id=uuid4(),
        student_id=uuid4(),
    )
    outcomes = AsyncMock(return_value=[])
    db = SimpleNamespace(commit=AsyncMock(), rollback=AsyncMock(), add=MagicMock())

    with (
        patch.object(
            evaluation_service.sessions_queries, "get_session", AsyncMock(return_value=session)
        ),
        patch.object(evaluation_service.authoring_queries, "list_outcomes_for_config", outcomes),
    ):
        await evaluation_service.generate_practice_feedback(db, session.id)

    outcomes.assert_not_awaited()
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_practice_feedback_never_writes_a_verdict() -> None:
    """The property the whole split rests on.

    ``pass_verdict = TRUE`` opens the SR-lesson and quiz gates. This path judges
    a rehearsal against the same criteria and must still leave that column
    untouched — along with the score and question counts a teacher view would
    read as a graded attempt.
    """
    from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
        OutcomeVerdict,
        OutcomeVerdicts,
    )

    outcome_id = uuid4()
    session = SimpleNamespace(
        id=uuid4(),
        session_mode=practice.MODE_PRACTICE,
        interview_config_id=uuid4(),
        student_id=uuid4(),
        pass_verdict=None,
        internal_summary_json={},
    )
    verdicts = OutcomeVerdicts(
        verdicts=[OutcomeVerdict(outcome_id=outcome_id, met=True, reasoning="r")]
    )
    db = SimpleNamespace(
        commit=AsyncMock(), rollback=AsyncMock(), add=MagicMock(), flush=AsyncMock()
    )

    with (
        patch.object(
            evaluation_service.sessions_queries, "get_session", AsyncMock(return_value=session)
        ),
        patch.object(
            evaluation_service.authoring_queries,
            "list_outcomes_for_config",
            AsyncMock(return_value=[SimpleNamespace(id=outcome_id, outcome_text="x")]),
        ),
        patch.object(authoring_queries, "list_questions_for_config", AsyncMock(return_value=[])),
        patch.object(evaluation_service, "_list_candidate_answers", AsyncMock(return_value=[])),
        patch.object(
            evaluation_service.sessions_queries,
            "get_session_with_responses",
            AsyncMock(return_value=(session, [], [])),
        ),
        patch.object(evaluation_service, "evaluate_outcomes", AsyncMock(return_value=verdicts)),
        patch.object(evaluation_service, "flush_or_conflict", AsyncMock()),
    ):
        await evaluation_service.generate_practice_feedback(db, session.id)

    assert session.pass_verdict is None
    summary = session.internal_summary_json
    assert summary["practice"]["outcomes_met"] == 1
    assert "total_score" not in summary
    assert "questions_total" not in summary
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_practice_feedback_draws_only_from_the_practice_partition() -> None:
    """It judges what was actually asked. Loading the graded bank here would
    mark the student down for criteria their rehearsal never covered."""
    session = SimpleNamespace(
        id=uuid4(),
        session_mode=practice.MODE_PRACTICE,
        interview_config_id=uuid4(),
        student_id=uuid4(),
        pass_verdict=None,
        internal_summary_json={},
    )
    mock, patcher = _capture_list_questions()
    db = SimpleNamespace(
        commit=AsyncMock(), rollback=AsyncMock(), add=MagicMock(), flush=AsyncMock()
    )

    with (
        patch.object(
            evaluation_service.sessions_queries, "get_session", AsyncMock(return_value=session)
        ),
        patch.object(
            evaluation_service.authoring_queries,
            "list_outcomes_for_config",
            AsyncMock(return_value=[]),
        ),
        patcher,
        patch.object(evaluation_service, "_list_candidate_answers", AsyncMock(return_value=[])),
        patch.object(
            evaluation_service.sessions_queries,
            "get_session_with_responses",
            AsyncMock(return_value=(session, [], [])),
        ),
        patch.object(
            evaluation_service,
            "evaluate_outcomes",
            AsyncMock(return_value=_empty_verdicts()),
        ),
        patch.object(evaluation_service, "flush_or_conflict", AsyncMock()),
    ):
        await evaluation_service.generate_practice_feedback(db, session.id)

    assert mock.await_args.kwargs["practice_only"] is True


def _empty_verdicts() -> object:
    from abridgeai.features.interviews.ai.stages.evaluation.outcome_verdicts import (
        OutcomeVerdicts,
    )

    return OutcomeVerdicts(verdicts=[])


# --------------------------------------------------------------------------- #
# The evaluator: the partition filter that protects grades, not secrets
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_evaluator_refuses_practice_sessions() -> None:
    """Defence in depth for the single function that writes ``pass_verdict``.

    The enqueue sites already skip practice. This refusal exists because that
    column is what the SR-lesson and quiz unlock gates read, so one missed
    enqueue anywhere must not be able to reach the write.
    """
    session = SimpleNamespace(
        id=uuid4(), session_mode="practice", interview_config_id=uuid4(), student_id=uuid4()
    )
    list_questions = AsyncMock(return_value=[])
    db = SimpleNamespace(commit=AsyncMock(), add=MagicMock(), flush=AsyncMock())

    with (
        patch.object(
            evaluation_service.sessions_queries, "get_session", AsyncMock(return_value=session)
        ),
        patch.object(authoring_queries, "list_questions_for_config", list_questions),
    ):
        result = await evaluation_service.evaluate_and_generate_report(db, session.id)

    assert result is None
    # Returned before touching the transcript, the LLM stages, or the run row.
    list_questions.assert_not_awaited()
    db.add.assert_not_called()


@pytest.mark.asyncio
async def test_graded_evaluation_corpus_excludes_the_practice_partition() -> None:
    """This query takes every review state and feeds ``expected_question_ids``.

    Unfiltered, a practice question is counted as unanswered, inflates
    ``questions_total``, and hard-fails the outcome linked to it. That is a
    corrupted grade, not a leaked bank — which is why it is asserted apart from
    the selection paths above.

    Only the assessment direction is exercised: a practice session is refused
    before it ever reaches this load (see the test above), so there is no
    practice case to assert here.
    """
    session = SimpleNamespace(
        id=uuid4(),
        session_mode=practice.MODE_ASSESSMENT,
        interview_config_id=uuid4(),
        student_id=uuid4(),
        # Must be a run that actually reached the assessment, or the evaluator's
        # never-started guard refuses it before this corpus load happens.
        assessment_started_at=datetime(2026, 7, 24, 10, 0, tzinfo=UTC),
    )
    mock, patcher = _capture_list_questions()
    # ``get`` returns None so the failure handler runs cleanly and re-raises,
    # keeping the assertion about the corpus query rather than the error path.
    db = SimpleNamespace(
        commit=AsyncMock(), add=MagicMock(), flush=AsyncMock(), get=AsyncMock(return_value=None)
    )

    with (
        patch.object(
            evaluation_service.sessions_queries, "get_session", AsyncMock(return_value=session)
        ),
        patch.object(
            evaluation_service.authoring_queries,
            "list_outcomes_for_config",
            AsyncMock(return_value=[]),
        ),
        patcher,
        patch.object(evaluation_service, "_list_candidate_answers", AsyncMock(return_value=[])),
        patch.object(
            evaluation_service.sessions_queries,
            "get_session_with_responses",
            AsyncMock(return_value=None),
        ),
        pytest.raises(Exception),  # noqa: B017, PT011 -- a None snapshot aborts after the load
    ):
        await evaluation_service.evaluate_and_generate_report(db, session.id)

    assert mock.await_args.kwargs["practice_only"] is False
