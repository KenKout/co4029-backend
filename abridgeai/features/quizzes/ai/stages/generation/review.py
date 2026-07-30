"""Validation-stage (T5.7) reshaping + dedup helpers for generation output.

Split out of ``parsers.py`` (250-LOC god-file budget). This module owns the
transform from a normalised :class:`GeneratedQuestion` into the compact dict the
downstream validation stage consumes, plus the answer-key rendering and the
dedup-key normaliser those rely on.

``parsers.py`` re-exports ``question_for_review`` and ``normalize_question_text``
so existing imports keep working.
"""

from __future__ import annotations

import string
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from abridgeai.features.quizzes.ai.stages.generation.parsers import (
        GeneratedQuestion,
    )


def _render_answer_for_review(qtype: str, data: dict[str, Any]) -> str:
    """Render a Phase 7 answer key as readable text for the validation stage.

    The validator judges groundedness from a compact text view of the question,
    so these answer shapes have to be flattened into prose rather than passed
    as raw JSON:

    * ``numerical`` → ``"3 (tolerance 0)"``
    * ``matching``  → ``"Extract -> Reads data; Load -> Writes data"``
    * ``ordering``  → ``"1. Extract; 2. Transform; 3. Load"``
    """
    if qtype == "numerical":
        answer = data.get("numeric_answer")
        if answer is None:
            return ""
        tolerance = data.get("numeric_tolerance")
        if tolerance is None:
            return str(answer)
        return f"{answer} (tolerance {tolerance})"

    if qtype == "matching":
        pairs = data.get("match_pairs")
        if not isinstance(pairs, list):
            return ""
        return "; ".join(
            f"{pair.get('left')} -> {pair.get('right')}"
            for pair in pairs
            if isinstance(pair, dict)
        )

    if qtype == "ordering":
        items = data.get("ordering_sequence")
        if not isinstance(items, list):
            return ""
        return "; ".join(
            f"{index}. {item}" for index, item in enumerate(items, start=1)
        )

    return ""


def question_for_review(question: GeneratedQuestion | dict[str, Any]) -> dict[str, Any]:
    """Reshape one question into the validation-stage (T5.7) input dict.

    The validator needs to see the question_type so it can apply the
    right shape rules per type. For non-MCQ questions we surface the
    expected answer text/list so the validator can judge groundedness.
    """
    data: dict[str, Any]
    if isinstance(question, dict):
        data = question
    elif hasattr(question, "model_dump"):
        data = question.model_dump()
    else:
        # Fallback for duck-typed candidate objects used in tests — read
        # their attributes directly.
        data = {
            "prompt_text": getattr(question, "prompt_text", None),
            "question_type": getattr(question, "question_type", None),
            "options": getattr(question, "options", None),
            "explanation": getattr(question, "explanation", None),
            "bloom_level": getattr(question, "bloom_level", None),
            "difficulty": getattr(question, "difficulty", None),
            "source_refs": getattr(question, "source_refs", None),
            "original_generated_payload": getattr(
                question, "original_generated_payload", None
            ),
        }
    options = data.get("options") or []
    options_dict: dict[str, str] = {}
    correct: str | None = None
    if isinstance(options, list):
        for opt in options:
            key = opt.get("option_key") if isinstance(opt, dict) else None
            if isinstance(key, str):
                options_dict[key] = opt.get("option_text", "")
                if opt.get("is_correct"):
                    correct = key
    payload = data.get("original_generated_payload") or {}
    qtype = data.get("question_type")
    if qtype in {"short_answer", "fill_blank"}:
        correct_text = payload.get("correct_answer")
    elif qtype == "true_false":
        # Map the canonical option keys ("T"/"F") to the literal strings
        # the validator's system prompt expects ("True"/"False").
        if correct == "T":
            correct_text = "True"
        elif correct == "F":
            correct_text = "False"
        else:
            correct_text = correct
    elif qtype in {"numerical", "matching", "ordering"}:
        # Phase 7: these carry their answer on dedicated fields, not option
        # rows. Render it as readable text so the validator can judge
        # groundedness — without this it would see an EMPTY correct_answer and
        # reject every question of these types as unsupported by the source.
        correct_text = _render_answer_for_review(qtype, data)
    else:
        correct_text = correct
    if qtype == "multiple_choice" and not data.get("single_answer", True):
        # Multi-select: surface every correct letter, not just the last one
        # seen, so the validator judges the whole answer set.
        correct_keys = sorted(
            str(opt.get("option_key"))
            for opt in (options if isinstance(options, list) else [])
            if isinstance(opt, dict) and opt.get("is_correct")
        )
        if correct_keys:
            correct_text = ", ".join(correct_keys)
    return {
        "prompt_text": data.get("prompt_text"),
        "question_type": data.get("question_type"),
        "options": options_dict,
        "correct_answer": correct_text,
        "explanation": data.get("explanation"),
        "bloom_level": data.get("bloom_level"),
        "difficulty": data.get("difficulty"),
    }


def normalize_question_text(text: str) -> str:
    """Lowercase + strip punctuation/whitespace for dedup keys."""
    translator = str.maketrans("", "", string.punctuation)
    return " ".join(text.lower().translate(translator).split())


__all__ = [
    "_render_answer_for_review",
    "normalize_question_text",
    "question_for_review",
]
