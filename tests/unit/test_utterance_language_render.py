"""Regression test: the utterance system prompt must render with ``language``.

A live strict-adaptive voice run surfaced that ``generate_utterance`` rendered
``prompts/utterance_system.j2`` with NO arguments, while that template
references ``{{ language }}``. Under Jinja ``StrictUndefined`` this raised
``UndefinedError`` on EVERY turn, so the LLM phrasing path always fell through
to the deterministic fallback — voice adaptive would have shipped sounding
robotic even though the decision model worked.

This test drives ``generate_utterance`` with a stub gateway that returns a
valid rewrite. If the system prompt fails to render, the code swallows the
exception and returns ``status == "fallback"``; a green LLM path (``"llm"``)
proves the template rendered with ``language`` bound. Covers EN and VI.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from abridgeai.features.interviews.orchestrator import utterance_logic
from abridgeai.features.interviews.orchestrator.decision import (
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.utterance import Persona


class _FakeGateway:
    """Returns a valid rewrite that preserves the question verbatim so
    ``_validated_rewrite`` accepts it and the status is ``"llm"``."""

    def __init__(self, question_or_probe: str) -> None:
        self._qp = question_or_probe
        self.generate_json = AsyncMock(
            return_value=SimpleNamespace(
                content_json={
                    "acknowledgement": "Thanks.",
                    "transition": "",
                    "ai_turn_text": f"Thanks. {question_or_probe}",
                }
            )
        )


@pytest.mark.parametrize("language", ["en", "vi"])
@pytest.mark.asyncio
async def test_utterance_system_prompt_renders_with_language(language: str) -> None:
    decision = InterviewerDecision(
        action=InterviewerActionType.ASK_FOR_EXAMPLE,
        reason_code=ReasonCode.MISSING_EXAMPLE,
    )

    # A DB whose begin_nested() is an async-context no-op (no real transaction).
    class _DB:
        def begin_nested(self) -> Any:
            class _Ctx:
                async def __aenter__(self) -> None:
                    return None

                async def __aexit__(self, *a: Any) -> None:
                    return None

            return _Ctx()

    # Build the fallback first to learn the exact question/probe text, then feed
    # the gateway a rewrite that preserves it.
    fallback = utterance_logic.build_fallback_utterance(
        decision, persona=Persona.NEUTRAL, language=language, question_text="Explain indexes."
    )
    gateway = _FakeGateway(fallback.question_or_probe)

    utterance, status = await utterance_logic.generate_utterance(
        _DB(),  # type: ignore[arg-type]
        decision,
        persona=Persona.NEUTRAL,
        language=language,
        question_text="Explain indexes.",
        use_llm=True,
        gateway=gateway,  # type: ignore[arg-type]
    )

    # The LLM path was reached and accepted → the system prompt rendered without
    # the historical 'language' UndefinedError.
    assert status == "llm", (
        "utterance system prompt failed to render (language likely unbound) — "
        "fell back to deterministic template"
    )
    assert utterance.question_or_probe == fallback.question_or_probe
    gateway.generate_json.assert_awaited_once()
