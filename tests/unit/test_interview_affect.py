"""Unit tests for deterministic affect heuristics (Slice 10, v2).

Pure — no DB, no LLM. The heuristic reads lightweight signals from the answer
text + analysis to classify candidate affect (terse / rambling / nervous /
confident / neutral). This is used ONLY to warm the utterance tone; it never
changes control flow, so a wrong read is cosmetic, never harmful.
"""

from __future__ import annotations

from abridgeai.features.interviews.orchestrator.affect import Affect, detect_affect
from abridgeai.features.interviews.orchestrator.analysis import (
    AnswerAnalysis,
    Completeness,
    Correctness,
    Relevance,
    Specificity,
)


def _strong() -> AnswerAnalysis:
    return AnswerAnalysis(
        relevance=Relevance.RELEVANT,
        completeness=Completeness.COMPLETE,
        correctness=Correctness.CORRECT,
        specificity=Specificity.SPECIFIC,
        confidence=0.9,
    )


def test_very_short_answer_is_terse() -> None:
    assert detect_affect(answer_text="Yes.", analysis=None) is Affect.TERSE


def test_long_low_specificity_answer_is_rambling() -> None:
    text = "well " * 120  # very long, no substance
    assert detect_affect(answer_text=text, analysis=None) is Affect.RAMBLING


def test_hedging_phrases_read_as_nervous() -> None:
    got = detect_affect(
        answer_text="Um, I'm not really sure, but maybe it could possibly be a fact table?",
        analysis=None,
    )
    assert got is Affect.NERVOUS


def test_strong_confident_answer_is_confident() -> None:
    text = "A fact table stores quantitative measures at a defined grain, "
    text += "with foreign keys to conformed dimensions."
    assert detect_affect(answer_text=text, analysis=_strong()) is Affect.CONFIDENT


def test_ordinary_answer_is_neutral() -> None:
    got = detect_affect(
        answer_text="A fact table holds measures and links to dimension tables.",
        analysis=None,
    )
    assert got is Affect.NEUTRAL
