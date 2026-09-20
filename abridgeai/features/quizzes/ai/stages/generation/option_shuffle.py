"""Randomisation helpers for generated multiple-choice options."""

from __future__ import annotations

import random
from typing import Any


def randomize_mcq_options(
    options: list[dict[str, Any]],
    *,
    rng: random.Random | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Shuffle MCQ choices and return the remapped correct option keys.

    LLMs commonly put the answer first or second. Re-keying after the model
    responds removes that test-taking signal while keeping the answer key and
    persisted option rows consistent. ``rng`` is injectable for tests;
    production uses a system-backed RNG.
    """
    shuffled = [dict(option) for option in options]
    (rng or random.SystemRandom()).shuffle(shuffled)
    keys = ("A", "B", "C", "D")
    normalized: list[dict[str, Any]] = []
    correct_keys: list[str] = []
    for position, (key, option) in enumerate(zip(keys, shuffled, strict=True), start=1):
        row = {
            "option_key": key,
            "option_text": option["option_text"],
            "is_correct": bool(option["is_correct"]),
            "position": position,
        }
        normalized.append(row)
        if row["is_correct"]:
            correct_keys.append(key)
    return normalized, correct_keys


__all__ = ["randomize_mcq_options"]
