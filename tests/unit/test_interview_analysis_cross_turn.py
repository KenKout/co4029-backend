"""Unit tests for cross-turn prior-claims injection into answer analysis (Slice 9).

Stubs the LLM gateway and captures the user prompt to prove the current
outcome's prior claims are passed into the analysis call (so the model can flag
a cross-turn contradiction). When no prior claims are supplied, the payload
carries an empty list — v1 parity.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from abridgeai.features.interviews.orchestrator import analysis_logic


class _CapturingDB:
    """A DB whose begin_nested() is an async-context no-op (no real txn)."""

    def begin_nested(self) -> Any:
        class _Ctx:
            async def __aenter__(self) -> None:
                return None

            async def __aexit__(self, *a: Any) -> bool:
                return False

        return _Ctx()


def _gateway_capturing(calls: list[dict[str, Any]]) -> SimpleNamespace:
    async def _gen(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            content_json={
                "relevance": "relevant",
                "completeness": "complete",
                "correctness": "correct",
                "specificity": "specific",
                "recommended_probe_type": "none",
                "confidence": 0.9,
                "evidence": [],
            }
        )

    return SimpleNamespace(generate_json=AsyncMock(side_effect=_gen))


@pytest.mark.asyncio
async def test_prior_claims_are_injected_into_analysis_prompt() -> None:
    calls: list[dict[str, Any]] = []
    gw = _gateway_capturing(calls)
    await analysis_logic.analyze_answer(
        _CapturingDB(),  # type: ignore[arg-type]
        question_text="Explain fact tables.",
        student_answer="They are wide and store attributes.",
        turn_id="t-2",
        outcome_id="o-1",
        outcome_text="Understands warehouse modeling",
        prior_claims=["A fact table stores measures, not attributes"],
        gateway=gw,
    )
    assert calls, "gateway was not called"
    payload = json.loads(calls[0]["user_prompt"])
    assert payload["prior_claims"] == ["A fact table stores measures, not attributes"]


@pytest.mark.asyncio
async def test_no_prior_claims_sends_empty_list() -> None:
    calls: list[dict[str, Any]] = []
    gw = _gateway_capturing(calls)
    await analysis_logic.analyze_answer(
        _CapturingDB(),  # type: ignore[arg-type]
        question_text="Explain fact tables.",
        student_answer="They store measures.",
        turn_id="t-1",
        outcome_id="o-1",
        gateway=gw,
    )
    payload = json.loads(calls[0]["user_prompt"])
    assert payload["prior_claims"] == []
