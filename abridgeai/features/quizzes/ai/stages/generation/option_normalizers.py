"""Option-list normalisers for the quiz GENERATION stage parser.

Per-type helpers that the main parser delegates to. ``multiple_choice``
keeps the LLM's option dict / list, ``true_false`` synthesizes a
canonical T/F pair from the correct flag, and ``short_answer`` /
``fill_blank`` return ``[]`` (their answers live on
``original_generated_payload`` instead).
"""

from __future__ import annotations

from typing import Any


def normalize_options(
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
    correct: Any,  # noqa: ANN401 -- raw LLM JSON
    question_type: str,
) -> list[dict[str, Any]]:
    if question_type == "multiple_choice":
        return _normalize_mcq_options(options_raw, correct)
    if question_type == "true_false":
        return _normalize_true_false_options(options_raw, correct)
    return []


def coerce_fill_blank_answer(raw: Any) -> list[str]:  # noqa: ANN401 -- raw LLM JSON
    """Return a list of blank strings, accepting list / semicolon /
    comma-separated string."""
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        candidate = raw.strip()
        if not candidate:
            return []
        for sep in (";", ","):
            if sep in candidate:
                return [piece.strip() for piece in candidate.split(sep) if piece.strip()]
        return [candidate]
    return []


def _normalize_mcq_options(
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
    correct: Any,  # noqa: ANN401 -- raw LLM JSON
) -> list[dict[str, Any]]:
    correct_key = correct.strip().upper() if isinstance(correct, str) else None
    if isinstance(options_raw, dict):
        cleaned = {
            str(key).strip().upper(): str(value).strip()
            for key, value in options_raw.items()
            if isinstance(value, str)
        }
        return [
            {
                "option_key": key,
                "option_text": cleaned.get(key, ""),
                "is_correct": key == correct_key,
                "position": pos,
            }
            for pos, key in enumerate(["A", "B", "C", "D"], start=1)
            if key in cleaned
        ]
    if isinstance(options_raw, list):
        out: list[dict[str, Any]] = []
        for pos, item in enumerate(options_raw, start=1):
            if not isinstance(item, dict):
                continue
            key = item.get("option_key") or item.get("key")
            if not isinstance(key, str):
                continue
            out.append(
                {
                    "option_key": key.strip().upper(),
                    "option_text": str(item.get("option_text") or item.get("text") or "").strip(),
                    "is_correct": bool(item.get("is_correct")),
                    "position": int(item.get("position") or pos),
                }
            )
        return out
    return []


def _normalize_true_false_options(
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
    correct: Any,  # noqa: ANN401 -- raw LLM JSON
) -> list[dict[str, Any]]:
    """Return canonical T/F option rows.

    Generators sometimes emit no options (just `correct_answer: "True"`)
    so we always synthesize the pair from the correct flag rather than
    trusting the LLM's option array verbatim.
    """
    correct_label = _coerce_true_false_correct(correct, options_raw)
    return [
        {
            "option_key": "T",
            "option_text": "True",
            "is_correct": correct_label is True,
            "position": 1,
        },
        {
            "option_key": "F",
            "option_text": "False",
            "is_correct": correct_label is False,
            "position": 2,
        },
    ]


def _coerce_true_false_correct(
    correct: Any,  # noqa: ANN401 -- raw LLM JSON
    options_raw: Any,  # noqa: ANN401 -- raw LLM JSON
) -> bool | None:
    """Coerce a true/false answer hint into a boolean."""
    if isinstance(correct, bool):
        return correct
    if isinstance(correct, str):
        token = correct.strip().lower()
        if token in {"true", "t", "1", "yes"}:
            return True
        if token in {"false", "f", "0", "no"}:
            return False
    if isinstance(options_raw, list):
        for item in options_raw:
            if isinstance(item, dict) and item.get("is_correct"):
                key = str(item.get("option_key") or item.get("key") or "").strip().lower()
                if key in {"t", "true"}:
                    return True
                if key in {"f", "false"}:
                    return False
    return None


__all__ = ["coerce_fill_blank_answer", "normalize_options"]
