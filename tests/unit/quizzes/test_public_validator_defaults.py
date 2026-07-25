"""The no-leak validator must not clobber schema defaults with source ``None``.

Bug this guards against:

``QuizQuestionPublic`` derives shuffled matching/ordering projections in a
``model_validator(mode="before")``. For an ORM source it rebuilds a dict of the
declared fields — and it used to copy every value verbatim, including ``None``.

Several columns are backed by a Postgres ``server_default`` (``prompt_format``,
``hint_format``, ``single_answer``). On a row that hasn't been flushed/refreshed
yet those attributes read as ``None`` in Python even though the schema declares
a perfectly good default (``"plain"`` / ``True``). Copying that ``None`` over the
default raised ``ValidationError: Input should be a valid string``.

Fix: skip a source ``None`` when the field is non-required AND its annotation
does not permit ``None``. Genuinely nullable fields (``hint_text: str | None``)
must still accept ``None``.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

from abridgeai.features.quizzes.schemas.public import QuizQuestionPublic

_PAIRS = [{"left": "A", "right": "1"}, {"left": "B", "right": "2"}]


def _row(**over: Any) -> SimpleNamespace:
    base: dict[str, Any] = {
        "id": uuid.uuid4(),
        "quiz_id": uuid.uuid4(),
        "position": 1,
        "question_type": "matching",
        "prompt_text": "Match them",
        "hint_text": None,
        # Unmaterialized server_defaults — the crux of the bug.
        "prompt_format": None,
        "hint_format": None,
        "single_answer": None,
        "options": [],
        "learning_outcome_id": None,
        "outcome_position": None,
        "match_pairs": _PAIRS,
        "ordering_sequence": None,
    }
    base.update(over)
    return SimpleNamespace(**base)


def test_unflushed_server_defaults_do_not_break_validation() -> None:
    """The exact row shape that raised ValidationError."""
    pub = QuizQuestionPublic.model_validate(_row())
    assert pub.prompt_format == "plain"
    assert pub.hint_format == "plain"
    assert pub.single_answer is True


def test_nullable_field_still_accepts_none() -> None:
    """``hint_text`` is genuinely ``str | None`` — None must survive."""
    pub = QuizQuestionPublic.model_validate(_row(hint_text=None))
    assert pub.hint_text is None


def test_explicit_value_is_not_replaced_by_default() -> None:
    """A real value must win over the schema default."""
    pub = QuizQuestionPublic.model_validate(_row(prompt_format="markdown"))
    assert pub.prompt_format == "markdown"


def test_no_leak_invariant_still_holds_on_such_a_row() -> None:
    """Defaults fix must not weaken the security projection."""
    pub = QuizQuestionPublic.model_validate(_row())
    dumped = pub.model_dump()
    assert "match_pairs" not in dumped
    assert "ordering_sequence" not in dumped
    assert pub.match_prompts == ["A", "B"]
    assert sorted(pub.match_choices) == ["1", "2"]


def test_ordering_row_with_unflushed_defaults() -> None:
    seq = ["s1", "s2", "s3", "s4"]
    pub = QuizQuestionPublic.model_validate(
        _row(question_type="ordering", match_pairs=None, ordering_sequence=seq)
    )
    assert pub.prompt_format == "plain"
    assert sorted(pub.ordering_items) == sorted(seq)
    assert pub.ordering_items != seq
    assert "ordering_sequence" not in pub.model_dump()
