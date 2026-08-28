"""Parsing helpers for the interview GENERATION stage (T6.5)."""

from __future__ import annotations

from typing import Any


def coerce_logical_question_index(value: Any) -> int | None:  # noqa: ANN401 -- raw LLM JSON
    """Accept non-negative integer group ordinals, never booleans.

    Strings and whole floats are accepted too — some API layers stringify
    the LLM JSON, and a group whose ordinal was serialised as ``"0"`` must
    not silently lose its whole bank.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str)):
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return None
        return coerced if coerced >= 0 else None
    return None
