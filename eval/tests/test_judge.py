"""Tests for the LLM-as-judge module (T8.3).

Monkey-patches ``eval.judges.judge._call_judge_llm`` so no real network
traffic is generated. The patched stub returns canned ``(content, cost)``
tuples mimicking the upstream provider's chat-completion payload.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from eval.judges import judge as judge_module
from eval.judges.judge import (
    JudgeScore,
    PairwiseVerdict,
    judge_pairwise,
    judge_response,
)

_LLMStub = Callable[..., Awaitable[tuple[str, float]]]


def _install_llm_stub(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[tuple[str, float]],
    *,
    capture: list[str] | None = None,
) -> None:
    iterator = iter(responses)

    async def _stub(
        *,
        judge_model: str,  # noqa: ARG001 - signature matches real impl
        prompt: str,
        settings: Any,  # noqa: ARG001
    ) -> tuple[str, float]:
        if capture is not None:
            capture.append(prompt)
        try:
            return next(iterator)
        except StopIteration as exc:
            raise AssertionError(
                "LLM stub exhausted; test wired wrong number of responses"
            ) from exc

    monkeypatch.setattr(judge_module, "_call_judge_llm", _stub)


@pytest.fixture(autouse=True)
def _reset_template_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the Jinja2 environment to reload from disk for each test.

    Otherwise template tests that swap loaders would leak between cases.
    """
    monkeypatch.setattr(judge_module, "_TEMPLATE_ENV", judge_module._build_environment())


async def test_judge_response_parses_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = (
        '{"score": 4, "justification": "well-grounded question", "confidence": 0.85}',
        0.00012,
    )
    _install_llm_stub(monkeypatch, [payload])

    result = await judge_response(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="groundedness",
        criterion_description="Question is grounded in the source",
        candidate_response="What is 2+2?",
        source_context="Arithmetic basics",
    )

    assert isinstance(result, JudgeScore)
    assert result.criterion_id == "groundedness"
    assert result.score == 4.0
    assert result.justification == "well-grounded question"
    assert result.confidence == 0.85
    assert result.judge_model == "gpt-4o-mini"
    assert result.cost_usd == pytest.approx(0.00012)


async def test_judge_response_handles_markdown_fenced_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fenced = '```json\n{"score": 3, "justification": "ok", "confidence": 0.5}\n```'
    _install_llm_stub(monkeypatch, [(fenced, 0.0)])

    result = await judge_response(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="answerability",
        criterion_description="Question is answerable",
        candidate_response="Q",
    )

    assert result.score == 3.0
    assert result.justification == "ok"
    assert result.confidence == 0.5


async def test_judge_response_handles_parse_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_llm_stub(monkeypatch, [("not json at all, just prose", 0.0)])

    result = await judge_response(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="groundedness",
        criterion_description="x",
        candidate_response="Q",
    )

    assert result.score == 1.0
    assert result.confidence == 0.0
    assert result.justification.startswith("JUDGE_PARSE_FAILED:")
    assert "not json at all" in result.justification


async def test_judge_response_caps_justification_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    long_justification = "x" * 5000
    payload = (
        '{"score": 5, "justification": "' + long_justification + '", "confidence": 0.7}',
        0.0,
    )
    _install_llm_stub(monkeypatch, [payload])

    result = await judge_response(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="groundedness",
        criterion_description="x",
        candidate_response="Q",
    )

    assert len(result.justification) == 1000
    assert result.justification == "x" * 1000


async def test_judge_response_clamps_out_of_range_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_llm_stub(
        monkeypatch,
        [('{"score": 10, "justification": "too high", "confidence": 0.9}', 0.0)],
    )

    result = await judge_response(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="groundedness",
        criterion_description="x",
        candidate_response="Q",
    )

    assert result.score == 5.0


async def test_judge_response_uses_correct_template_for_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []
    _install_llm_stub(
        monkeypatch,
        [('{"score": 3, "justification": "ok", "confidence": 0.5}', 0.0)],
        capture=captured,
    )

    await judge_response(
        judge_model="gpt-4o-mini",
        scenario_capability="interview_generation",
        criterion_id="open_endedness",
        criterion_description="Questions invite explanation",
        candidate_response="Why does X happen?",
        source_context="Some passage",
    )

    assert len(captured) == 1
    rendered = captured[0]
    assert "interview question" in rendered.lower()
    assert "Questions invite explanation" in rendered
    assert "Why does X happen?" in rendered


async def test_judge_response_unknown_capability_raises() -> None:
    with pytest.raises(ValueError, match="unknown scenario capability"):
        await judge_response(
            judge_model="gpt-4o-mini",
            scenario_capability="not_a_real_capability",
            criterion_id="x",
            criterion_description="x",
            candidate_response="Q",
        )


async def test_judge_response_handles_llm_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _boom(**_kwargs: Any) -> tuple[str, float]:
        raise RuntimeError("provider down")

    monkeypatch.setattr(judge_module, "_call_judge_llm", _boom)

    result = await judge_response(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="groundedness",
        criterion_description="x",
        candidate_response="Q",
    )

    assert result.score == 1.0
    assert result.confidence == 0.0
    assert result.justification.startswith("JUDGE_PARSE_FAILED:")
    assert "provider down" in result.justification


async def test_judge_pairwise_position_swap_detection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge picks 'a' first call, then picks 'a' again when positions
    swapped — meaning original 'b' beat original 'a' in the second run.
    Disagreement → tie."""
    _install_llm_stub(
        monkeypatch,
        [
            ('{"winner": "a", "justification": "first", "confidence": 0.9}', 0.0001),
            ('{"winner": "a", "justification": "swapped", "confidence": 0.9}', 0.0001),
        ],
    )

    verdict = await judge_pairwise(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="groundedness",
        criterion_description="x",
        candidate_a="alpha",
        candidate_b="bravo",
    )

    assert isinstance(verdict, PairwiseVerdict)
    assert verdict.winner == "tie"
    assert verdict.confidence == 0.0
    assert verdict.cost_usd == pytest.approx(0.0002)


async def test_judge_pairwise_consistent_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First call picks 'b'; swapped call picks 'a' (which is original
    'b'). Both runs agree original 'b' wins."""
    _install_llm_stub(
        monkeypatch,
        [
            ('{"winner": "b", "justification": "b is better", "confidence": 0.8}', 0.0001),
            ('{"winner": "a", "justification": "still b", "confidence": 0.7}', 0.0001),
        ],
    )

    verdict = await judge_pairwise(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="groundedness",
        criterion_description="x",
        candidate_a="alpha",
        candidate_b="bravo",
    )

    assert verdict.winner == "b"
    assert verdict.confidence == pytest.approx(0.75)
    assert "b is better" in verdict.justification
    assert "still b" in verdict.justification


async def test_judge_pairwise_tie_when_either_run_unparseable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_llm_stub(
        monkeypatch,
        [
            ('{"winner": "b", "justification": "ok", "confidence": 0.8}', 0.0001),
            ("garbage non-json", 0.0001),
        ],
    )

    verdict = await judge_pairwise(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="groundedness",
        criterion_description="x",
        candidate_a="alpha",
        candidate_b="bravo",
    )

    assert verdict.winner == "tie"
    assert verdict.confidence == 0.0


async def test_judge_pairwise_invalid_winner_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_llm_stub(
        monkeypatch,
        [
            ('{"winner": "neither", "justification": "?", "confidence": 0.5}', 0.0),
            ('{"winner": "neither", "justification": "?", "confidence": 0.5}', 0.0),
        ],
    )

    verdict = await judge_pairwise(
        judge_model="gpt-4o-mini",
        scenario_capability="quiz_generation",
        criterion_id="groundedness",
        criterion_description="x",
        candidate_a="alpha",
        candidate_b="bravo",
    )

    assert verdict.winner == "tie"
    assert verdict.confidence == 0.0
