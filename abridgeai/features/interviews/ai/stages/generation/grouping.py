"""Parsing helpers for the interview GENERATION stage (T6.5)."""

from __future__ import annotations

from typing import Any


def coerce_logical_question_index(value: Any) -> int | None:  # noqa: ANN401 -- raw LLM JSON
    """Accept non-negative integer group ordinals, never booleans/floats."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None
