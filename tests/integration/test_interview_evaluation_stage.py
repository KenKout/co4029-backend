"""Integration tests for the interview EVALUATION stage (T6.8).

These tests mock :class:`LLMGateway` so they don't touch the network or
the DB. The stage's contract is: take a completed session + answers,
return a :class:`RubricScores` shape ready for T6.11 services to
persist. Tests cover:

* per-response judgement (N answers → N ResponseEvaluation entries)
* aggregation lands in [0, 100]
* config-supplied criteria + weights are honoured
* default 4-criterion equal-weight fallback when config is silent
* prompts live in ``.j2`` files only (no string literal prompts in code)
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.features.interviews.ai.stages.evaluation import (
    DEFAULT_CRITERIA,
    EVALUATION_STAGE_NAME,
    CriterionScore,
    ResponseEvaluation,
    aggregate_rubric_scores,
    evaluate_outcomes,
    evaluate_session,
    parse_evaluation_response,
    resolve_rubric,
)


def _make_session(session_id: UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(id=session_id or uuid4())


def _make_question(question_id: UUID, prompt: str = "Explain idempotency.") -> SimpleNamespace:
    return SimpleNamespace(id=question_id, prompt_text=prompt)


def _make_outcome(text: str = "Understands idempotency") -> SimpleNamespace:
    return SimpleNamespace(outcome_text=text, outcome_type="knowledge", importance_weight=3)


def _make_answer(
    session_question_id: UUID,
    text: str,
    *,
    role: str = "user",
) -> SimpleNamespace:
    return SimpleNamespace(
        session_question_id=session_question_id,
        role=role,
        content_text=text,
    )


def _llm_payload(scores: dict[str, int], justification: str = "Reasonable.") -> dict:
    return {
        "criterion_scores": [
            {"criterion": name, "score": value, "justification": justification}
            for name, value in scores.items()
        ]
    }


def _gateway_returning(payloads: list[dict]) -> SimpleNamespace:
    side_effects = [SimpleNamespace(content_json=payload) for payload in payloads]
    generate = AsyncMock(side_effect=side_effects)
    return SimpleNamespace(generate_json=generate)


@pytest.mark.asyncio
async def test_evaluates_each_response() -> None:
    session = _make_session()
    q1, q2 = uuid4(), uuid4()
    questions = [_make_question(q1, "Q1?"), _make_question(q2, "Q2?")]
    outcomes = [_make_outcome()]
    answers = [
        _make_answer(q1, "Idempotency means repeated calls have the same effect."),
        _make_answer(q2, "I would partition by user id to keep retries safe."),
    ]
    payloads = [
        _llm_payload(dict.fromkeys(DEFAULT_CRITERIA, 4)),
        _llm_payload(dict.fromkeys(DEFAULT_CRITERIA, 3)),
    ]
    gateway = _gateway_returning(payloads)
    db = AsyncMock()

    result = await evaluate_session(
        db,
        session=session,
        outcomes=outcomes,
        questions=questions,
        answers=answers,
        config=None,
        pipeline_run_id=uuid4(),
        gateway=gateway,
    )

    assert len(result.response_evaluations) == 2
    assert gateway.generate_json.await_count == 2
    assert {ev.session_question_id for ev in result.response_evaluations} == {q1, q2}


@pytest.mark.asyncio
async def test_unanswered_questions_receive_zero_and_reduce_total_score() -> None:
    session = _make_session()
    answered_id, unanswered_id = uuid4(), uuid4()
    questions = [
        _make_question(answered_id, "Answered?"),
        _make_question(unanswered_id, "Skipped?"),
    ]
    gateway = _gateway_returning([_llm_payload(dict.fromkeys(DEFAULT_CRITERIA, 4))])

    result = await evaluate_session(
        AsyncMock(),
        session=session,
        outcomes=[],
        questions=questions,
        answers=[_make_answer(answered_id, "My submitted answer.")],
        gateway=gateway,
    )

    assert gateway.generate_json.await_count == 1
    by_question = {
        evaluation.session_question_id: evaluation for evaluation in result.response_evaluations
    }
    assert set(by_question) == {answered_id, unanswered_id}
    assert all(score.score == 0 for score in by_question[unanswered_id].criterion_scores)
    assert result.total_score == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_production_session_question_mapping_supplies_prompt_to_judge() -> None:
    session = _make_session()
    configured_question_id = uuid4()
    session_question_id = uuid4()
    gateway = _gateway_returning([_llm_payload(dict.fromkeys(DEFAULT_CRITERIA, 3))])

    await evaluate_session(
        AsyncMock(),
        session=session,
        outcomes=[],
        questions=[_make_question(configured_question_id, "Configured prompt")],
        answers=[_make_answer(session_question_id, "Candidate response")],
        question_prompts={session_question_id: "Configured prompt"},
        expected_question_ids=[session_question_id],
        gateway=gateway,
    )

    payload = json.loads(gateway.generate_json.await_args.kwargs["user_prompt"])
    assert payload["question"] == "Configured prompt"


@pytest.mark.asyncio
async def test_followup_turns_for_one_question_are_graded_once() -> None:
    session = _make_session()
    question_id = uuid4()
    gateway = _gateway_returning([_llm_payload(dict.fromkeys(DEFAULT_CRITERIA, 3))])

    result = await evaluate_session(
        AsyncMock(),
        session=session,
        outcomes=[],
        questions=[_make_question(question_id)],
        answers=[
            _make_answer(question_id, "Initial answer."),
            _make_answer(question_id, "Follow-up detail."),
        ],
        gateway=gateway,
    )

    assert gateway.generate_json.await_count == 1
    assert len(result.response_evaluations) == 1
    payload = json.loads(gateway.generate_json.await_args.kwargs["user_prompt"])
    assert payload["candidate_response"] == "Initial answer.\n\nFollow-up detail."


@pytest.mark.asyncio
async def test_aggregates_to_total_score_in_0_100() -> None:
    session = _make_session()
    q1 = uuid4()
    questions = [_make_question(q1)]
    answers = [_make_answer(q1, "A solid answer.")]
    perfect = _llm_payload(dict.fromkeys(DEFAULT_CRITERIA, 5))
    gateway = _gateway_returning([perfect])
    db = AsyncMock()

    result = await evaluate_session(
        db,
        session=session,
        outcomes=[],
        questions=questions,
        answers=answers,
        config=None,
        pipeline_run_id=uuid4(),
        gateway=gateway,
    )

    assert 0.0 <= result.total_score <= 100.0
    assert result.total_score == pytest.approx(100.0, abs=0.01)


@pytest.mark.asyncio
async def test_uses_config_criteria_when_present() -> None:
    session = _make_session()
    q1 = uuid4()
    questions = [_make_question(q1)]
    answers = [_make_answer(q1, "Reasoned response.")]

    config = {
        "rubric_weights": {
            "depth": 3.0,
            "clarity": 1.0,
        }
    }
    payload = _llm_payload({"depth": 5, "clarity": 1})
    gateway = _gateway_returning([payload])
    db = AsyncMock()

    result = await evaluate_session(
        db,
        session=session,
        outcomes=[],
        questions=questions,
        answers=answers,
        config=config,
        pipeline_run_id=uuid4(),
        gateway=gateway,
    )

    assert set(result.aggregated.keys()) == {"depth", "clarity"}
    expected = (5 * (3.0 / 4.0) + 1 * (1.0 / 4.0)) * (100.0 / 5.0)
    assert result.total_score == pytest.approx(expected, abs=0.05)
    assert result.aggregated["depth"] == pytest.approx(5.0)
    assert result.aggregated["clarity"] == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_falls_back_to_default_4_criterion_equal_weight() -> None:
    weights = resolve_rubric(None)
    assert tuple(weights.keys()) == DEFAULT_CRITERIA
    assert all(w == pytest.approx(0.25) for w in weights.values())

    weights_empty_cfg = resolve_rubric({})
    assert tuple(weights_empty_cfg.keys()) == DEFAULT_CRITERIA

    weights_bad_cfg = resolve_rubric({"rubric_weights": {"x": -1, "y": 0}})
    assert tuple(weights_bad_cfg.keys()) == DEFAULT_CRITERIA


def test_jinja_prompts_in_j2_only() -> None:
    here = Path(__file__).resolve().parents[1]
    stage_dir = (
        here.parent / "abridgeai" / "features" / "interviews" / "ai" / "stages" / "evaluation"
    )
    assert (stage_dir / "prompts" / "system.j2").is_file()
    assert (stage_dir / "prompts" / "user.j2").is_file()

    for py_file in stage_dir.glob("*.py"):
        text = py_file.read_text()
        assert 'system_prompt = "' not in text, f"{py_file.name}: inline system prompt"
        assert "system_prompt = '" not in text, f"{py_file.name}: inline system prompt"


def test_aggregate_rubric_scores_pure_helper() -> None:
    qid = uuid4()
    evaluations = [
        ResponseEvaluation(
            session_question_id=qid,
            criterion_scores=[
                CriterionScore(criterion="a", score=4.0, justification="ok"),
                CriterionScore(criterion="b", score=2.0, justification="ok"),
            ],
        ),
        ResponseEvaluation(
            session_question_id=qid,
            criterion_scores=[
                CriterionScore(criterion="a", score=2.0, justification="ok"),
                CriterionScore(criterion="b", score=4.0, justification="ok"),
            ],
        ),
    ]
    scores = aggregate_rubric_scores(evaluations, {"a": 1.0, "b": 1.0})
    assert scores.aggregated["a"] == pytest.approx(3.0)
    assert scores.aggregated["b"] == pytest.approx(3.0)
    assert scores.total_score == pytest.approx(60.0, abs=0.01)


def test_parser_fills_missing_criteria_with_zero() -> None:
    payload = {
        "criterion_scores": [
            {"criterion": "a", "score": 4, "justification": "good"},
        ]
    }
    parsed = parse_evaluation_response(payload, expected_criteria=("a", "b"))
    assert len(parsed) == 2
    assert parsed[0].criterion == "a"
    assert parsed[0].score == 4.0
    assert parsed[1].criterion == "b"
    assert parsed[1].score == 0.0
    assert "did not return" in parsed[1].justification


def test_parser_clips_out_of_range_scores() -> None:
    payload = {
        "criterion_scores": [
            {"criterion": "a", "score": 99, "justification": "x"},
            {"criterion": "b", "score": -3, "justification": "x"},
        ]
    }
    parsed = parse_evaluation_response(payload, expected_criteria=("a", "b"))
    assert parsed[0].score == 5.0
    assert parsed[1].score == 0.0


@pytest.mark.asyncio
async def test_skips_non_user_messages_and_empty_answers() -> None:
    session = _make_session()
    q1 = uuid4()
    questions = [_make_question(q1)]
    answers = [
        _make_answer(q1, "Spoken AI line.", role="ai"),
        _make_answer(q1, "   "),
        _make_answer(q1, "Real candidate response."),
    ]
    payload = _llm_payload(dict.fromkeys(DEFAULT_CRITERIA, 3))
    gateway = _gateway_returning([payload])
    db = AsyncMock()

    result = await evaluate_session(
        db,
        session=session,
        outcomes=[],
        questions=questions,
        answers=answers,
        config=None,
        pipeline_run_id=uuid4(),
        gateway=gateway,
    )

    assert gateway.generate_json.await_count == 1
    assert len(result.response_evaluations) == 1


@pytest.mark.asyncio
async def test_audit_stage_name_and_role() -> None:
    session = _make_session()
    q1 = uuid4()
    questions = [_make_question(q1)]
    answers = [_make_answer(q1, "Answer.")]
    pipeline_run_id = uuid4()
    payload = _llm_payload(dict.fromkeys(DEFAULT_CRITERIA, 3))
    gateway = _gateway_returning([payload])
    db = AsyncMock()

    await evaluate_session(
        db,
        session=session,
        outcomes=[],
        questions=questions,
        answers=answers,
        config=None,
        pipeline_run_id=pipeline_run_id,
        gateway=gateway,
    )

    kwargs = gateway.generate_json.await_args.kwargs
    assert kwargs["stage_name"] == "evaluation"
    assert EVALUATION_STAGE_NAME == "evaluation"
    assert kwargs["pipeline_run_id"] == pipeline_run_id
    from abridgeai.ai.llm import LLMRole

    assert kwargs["role"] == LLMRole.INTERVIEW_EVALUATION


# ─────────────────── Per-outcome verdict stage (thesis §4.3) ───────────────────


def _make_outcome_with_id(
    outcome_id: UUID, text: str = "Understands idempotency"
) -> SimpleNamespace:
    return SimpleNamespace(
        id=outcome_id, outcome_text=text, outcome_type="knowledge", importance_weight=3
    )


def _verdict_payload(verdicts: list[dict]) -> SimpleNamespace:
    return SimpleNamespace(content_json={"verdicts": verdicts})


def _gateway_one(payload: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(generate_json=AsyncMock(return_value=payload))


@pytest.mark.asyncio
async def test_evaluate_outcomes_one_verdict_per_outcome() -> None:
    session = _make_session()
    q1 = uuid4()
    o1, o2 = uuid4(), uuid4()
    outcomes = [_make_outcome_with_id(o1, "Knows REST"), _make_outcome_with_id(o2, "Knows SQL")]
    questions = [_make_question(q1, "Explain REST.")]
    answers = [_make_answer(q1, "REST is stateless; resources via verbs.")]
    gateway = _gateway_one(
        _verdict_payload(
            [
                {
                    "outcome_id": str(o1),
                    "met": True,
                    "reasoning": "Covered statelessness.",
                    "evidence": "REST is stateless",
                },
                {
                    "outcome_id": str(o2),
                    "met": False,
                    "reasoning": "No SQL shown.",
                    "evidence": None,
                },
            ]
        )
    )
    db = AsyncMock()

    result = await evaluate_outcomes(
        db,
        session=session,
        outcomes=outcomes,
        questions=questions,
        answers=answers,
        pipeline_run_id=uuid4(),
        gateway=gateway,
    )

    assert result.total == 2
    assert result.met_count == 1
    by_id = {v.outcome_id: v for v in result.verdicts}
    assert by_id[o1].met is True
    assert by_id[o2].met is False
    # Single batch call for the whole transcript.
    assert gateway.generate_json.await_count == 1


@pytest.mark.asyncio
async def test_evaluate_outcomes_missing_verdict_defaults_not_met() -> None:
    session = _make_session()
    q1 = uuid4()
    o1, o2 = uuid4(), uuid4()
    outcomes = [_make_outcome_with_id(o1), _make_outcome_with_id(o2)]
    questions = [_make_question(q1)]
    answers = [_make_answer(q1, "A partial answer.")]
    # LLM only returns a verdict for o1.
    gateway = _gateway_one(
        _verdict_payload([{"outcome_id": str(o1), "met": True, "reasoning": "ok"}])
    )
    db = AsyncMock()

    result = await evaluate_outcomes(
        db,
        session=session,
        outcomes=outcomes,
        questions=questions,
        answers=answers,
        pipeline_run_id=None,
        gateway=gateway,
    )

    by_id = {v.outcome_id: v for v in result.verdicts}
    assert by_id[o1].met is True
    assert by_id[o2].met is False  # defaulted — safe direction
    assert result.met_count == 1


@pytest.mark.asyncio
async def test_evaluate_outcomes_no_answers_all_not_met_no_llm() -> None:
    session = _make_session()
    o1 = uuid4()
    outcomes = [_make_outcome_with_id(o1)]
    gateway = SimpleNamespace(generate_json=AsyncMock())
    db = AsyncMock()

    result = await evaluate_outcomes(
        db,
        session=session,
        outcomes=outcomes,
        questions=[],
        answers=[],
        pipeline_run_id=None,
        gateway=gateway,
    )

    assert result.total == 1
    assert result.met_count == 0
    gateway.generate_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_outcomes_empty_outcomes_returns_empty() -> None:
    session = _make_session()
    gateway = SimpleNamespace(generate_json=AsyncMock())
    db = AsyncMock()

    result = await evaluate_outcomes(
        db,
        session=session,
        outcomes=[],
        questions=[_make_question(uuid4())],
        answers=[_make_answer(uuid4(), "answer")],
        pipeline_run_id=None,
        gateway=gateway,
    )

    assert result.total == 0
    gateway.generate_json.assert_not_awaited()


@pytest.mark.asyncio
async def test_evaluate_outcomes_audit_stage_and_role() -> None:
    session = _make_session()
    q1 = uuid4()
    o1 = uuid4()
    pipeline_run_id = uuid4()
    gateway = _gateway_one(
        _verdict_payload([{"outcome_id": str(o1), "met": True, "reasoning": "ok"}])
    )
    db = AsyncMock()

    await evaluate_outcomes(
        db,
        session=session,
        outcomes=[_make_outcome_with_id(o1)],
        questions=[_make_question(q1)],
        answers=[_make_answer(q1, "Answer.")],
        pipeline_run_id=pipeline_run_id,
        gateway=gateway,
    )

    kwargs = gateway.generate_json.await_args.kwargs
    assert kwargs["stage_name"] == "evaluation"
    assert kwargs["pipeline_run_id"] == pipeline_run_id
    from abridgeai.ai.llm import LLMRole

    assert kwargs["role"] == LLMRole.INTERVIEW_EVALUATION
