"""Deterministic per-attempt question/option shuffle (Phase 6).

The ``quiz.shuffle_questions`` / ``quiz.shuffle_options`` flags were inert before
this phase. Here we realize them into a concrete order that is:

* **deterministic** — seeded by the attempt UUID, so the same attempt always
  produces the same order (reproducible on resume / review), and
* **persisted** — written to ``quiz_attempts.layout`` at start time so the order
  survives a server restart and is authoritative for grading/review.

No DB access here — pure functions, trivially unit-testable.
"""

from __future__ import annotations

import random
import uuid
from typing import Any


def _seed_for(attempt_id: uuid.UUID) -> int:
    return int(attempt_id.hex, 16)


def build_layout(
    attempt_id: uuid.UUID,
    ordered_question_ids: list[uuid.UUID],
    options_by_question: dict[uuid.UUID, list[uuid.UUID]],
    *,
    shuffle_questions: bool,
    shuffle_options: bool,
) -> dict[str, Any]:
    """Deterministically realize question/option order from a per-attempt seed."""
    rng = random.Random(_seed_for(attempt_id))  # noqa: S311 -- deterministic reproducible shuffle, not cryptographic
    q_ids = [str(q) for q in ordered_question_ids]
    if shuffle_questions:
        rng.shuffle(q_ids)
    opt_order: dict[str, list[str]] = {}
    for qid in ordered_question_ids:
        opts = [str(o) for o in options_by_question.get(qid, [])]
        if shuffle_options and opts:
            # Per-question sub-seed so option shuffles are independent of each
            # other and of the question shuffle.
            sub = random.Random(  # noqa: S311 -- deterministic reproducible shuffle, not cryptographic
                _seed_for(attempt_id) ^ int(uuid.UUID(str(qid)).hex, 16)
            )
            sub.shuffle(opts)
        opt_order[str(qid)] = opts
    return {
        "question_order": q_ids,
        "option_order": opt_order,
        "seed": attempt_id.hex,
        "v": 1,
    }


def apply_layout(questions: list, layout: dict | None) -> list:
    """Reorder already-loaded question objects (+ their options) per a stored layout.

    Defensive: questions/options not present in the layout (e.g. added after the
    attempt started) are appended in their original order rather than dropped.
    """
    if not layout:
        return questions
    order = layout.get("question_order", [])
    q_by_id = {str(q.id): q for q in questions}
    ordered = [q_by_id[qid] for qid in order if qid in q_by_id]
    for q in questions:
        if str(q.id) not in order:
            ordered.append(q)
    opt_order = layout.get("option_order", {})
    for q in ordered:
        oorder = opt_order.get(str(q.id))
        if oorder and getattr(q, "options", None):
            o_by_id = {str(o.id): o for o in q.options}
            q.options = [o_by_id[oid] for oid in oorder if oid in o_by_id] + [
                o for o in q.options if str(o.id) not in oorder
            ]
    return ordered


__all__ = ["apply_layout", "build_layout"]
