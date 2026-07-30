"""Golden-file parity for the deterministic persona utterance layer.

Purpose
-------
This is the safety net that makes the Phase 1 ``_ACK`` / ``_TRANSITION`` re-key
(from ``(Persona, style, lang)`` to ``(warmth_band, style, lang)``) provably
behaviour-preserving. It snapshots the EXACT ``ai_turn_text`` that
``build_fallback_utterance`` produces for every reachable
``(persona × action × acknowledgement_style × language)`` combination, plus the
laddered-hint and reframe escalation paths that vary independently of that grid.

If a refactor changes any rendered string, this test fails with the precise key
that drifted — so a re-key that is meant to be a pure internal reorganisation
cannot silently alter what a candidate hears.

Regenerating (ONLY when a wording change is intentional)
--------------------------------------------------------
Run this module as a script:

    python -m tests.unit.test_interview_persona_golden --update

and review the fixture diff in the PR. Never regenerate to "make the test pass"
without reading the diff — that defeats the entire point of the snapshot.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from abridgeai.features.interviews.orchestrator.decision import (
    AcknowledgementStyle,
    InterviewerActionType,
    InterviewerDecision,
    ReasonCode,
)
from abridgeai.features.interviews.orchestrator.utterance import (
    Persona,
    build_fallback_utterance,
    laddered_hint,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_GOLDEN = _FIXTURES / "persona_golden_utterances.json"
_GOLDEN_EXTRA = _FIXTURES / "persona_golden_extra.json"

# A fixed question so the grid isolates persona/action/style/language wording —
# the question text itself is passed through verbatim and is not what we snapshot.
_QUESTION = "Explain idempotency."
_LANGS = ("en", "vi")
_PERSONAS = (Persona.STRICT, Persona.NEUTRAL, Persona.SUPPORTIVE)
_STYLES = (
    AcknowledgementStyle.NONE,
    AcknowledgementStyle.NEUTRAL,
    AcknowledgementStyle.POSITIVE,
    AcknowledgementStyle.CORRECTIVE,
)
# Every action the fallback renderer dispatches on.
_ACTIONS = tuple(InterviewerActionType)


def _decision(action: InterviewerActionType, style: AcknowledgementStyle) -> InterviewerDecision:
    return InterviewerDecision(
        action=action,
        reason_code=ReasonCode.OUTCOME_NOT_COVERED,
        acknowledgement_style=style,
    )


def _build_matrix() -> dict[str, str]:
    """Render ai_turn_text for the full (persona × action × style × lang) grid."""
    out: dict[str, str] = {}
    for persona in _PERSONAS:
        for action in _ACTIONS:
            for style in _STYLES:
                for lang in _LANGS:
                    utt = build_fallback_utterance(
                        _decision(action, style),
                        persona=persona,
                        language=lang,
                        question_text=_QUESTION,
                    )
                    key = f"{persona.value}|{action.value}|{style.value}|{lang}"
                    out[key] = utt.ai_turn_text
    return out


def _build_extra() -> dict[str, str]:
    """Escalation paths that vary independently of the persona×style grid.

    Laddered hints (hint_level) and reframe signposts (reframe_count) escalate
    on their own axis; snapshot them across levels 0-3 (3 clamps to the last
    rung) so the re-key can't perturb the ladder either.
    """
    out: dict[str, str] = {}
    for lang in _LANGS:
        for level in range(4):
            hint = build_fallback_utterance(
                _decision(InterviewerActionType.PROVIDE_NEUTRAL_HINT, AcknowledgementStyle.NONE),
                persona=Persona.NEUTRAL,
                language=lang,
                question_text="",
                hint_level=level,
            )
            out[f"hint|{lang}|{level}"] = hint.ai_turn_text
            # Empty question_text so the probe falls through to the deterministic
            # per-language generic reframe probe — that is the escalation path we
            # are snapshotting, not a passed-through question string.
            reframe = build_fallback_utterance(
                _decision(InterviewerActionType.REFRAME_QUESTION, AcknowledgementStyle.NONE),
                persona=Persona.NEUTRAL,
                language=lang,
                question_text="",
                reframe_count=level,
            )
            out[f"reframe|{lang}|{level}"] = reframe.ai_turn_text
    # Sanity: the public laddered_hint helper agrees with the rendered ladder.
    for lang in _LANGS:
        assert laddered_hint(0, lang) in out[f"hint|{lang}|0"]
    return out


def test_persona_utterance_grid_matches_golden() -> None:
    expected = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    actual = _build_matrix()
    assert actual == expected, (
        "Deterministic persona wording drifted from the golden snapshot. "
        "If intentional, regenerate with `python -m "
        "tests.unit.test_interview_persona_golden --update` and review the diff."
    )


def test_persona_escalation_paths_match_golden() -> None:
    expected = json.loads(_GOLDEN_EXTRA.read_text(encoding="utf-8"))
    actual = _build_extra()
    assert actual == expected


def test_golden_grid_is_complete() -> None:
    """Guard against the grid silently shrinking (e.g. an action dropped)."""
    grid = _build_matrix()
    assert len(grid) == len(_PERSONAS) * len(_ACTIONS) * len(_STYLES) * len(_LANGS)


if __name__ == "__main__":
    import sys

    if "--update" in sys.argv:
        _GOLDEN.write_text(
            json.dumps(_build_matrix(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _GOLDEN_EXTRA.write_text(
            json.dumps(_build_extra(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print("golden fixtures regenerated")
    else:
        print("pass --update to regenerate the fixtures")
