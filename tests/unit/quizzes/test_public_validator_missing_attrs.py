"""Regression: the no-leak validator must not fabricate ``None`` for absent attrs.

``QuizQuestion`` has NO ``options`` ORM relationship — callers attach the list
manually (see ``_attach_question_options`` in the authoring router). The Phase 7
no-leak validator rebuilds a dict of declared fields off the ORM row; an earlier
version used ``getattr(row, name, None)`` unconditionally, which injected
``None`` into the required ``options: list[...]`` field and raised
ValidationError for ANY matching/ordering question loaded without options
pre-attached (i.e. the student take path).

The validator must skip attributes the source object does not have, letting the
schema default apply instead.
"""

from __future__ import annotations

import uuid

from abridgeai.features.quizzes.schemas.public import QuizQuestionPublic


class _RowWithoutOptions:
    """ORM-like row that deliberately has NO ``options`` attribute."""

    def __init__(self, **kw: object) -> None:
        for key, value in kw.items():
            setattr(self, key, value)


def _row(**over: object) -> _RowWithoutOptions:
    base: dict[str, object] = {
        "id": uuid.uuid4(),
        "quiz_id": uuid.uuid4(),
        "position": 1,
        "question_type": "matching",
        "prompt_text": "Match them",
        "hint_text": None,
        "prompt_format": "plain",
        "hint_format": "plain",
        "single_answer": True,
        "learning_outcome_id": None,
        "outcome_position": None,
        "match_pairs": [{"left": "A", "right": "1"}, {"left": "B", "right": "2"}],
        "ordering_sequence": None,
    }
    base.update(over)
    return _RowWithoutOptions(**base)


def test_matching_validates_when_options_attr_absent() -> None:
    """The take path loads questions without attaching options — must not raise."""
    pub = QuizQuestionPublic.model_validate(_row())
    # Schema default applies instead of a fabricated None.
    assert pub.options == []
    # Derived no-leak projection still works.
    assert pub.match_prompts == ["A", "B"]
    assert sorted(pub.match_choices) == ["1", "2"]


def test_ordering_validates_when_options_attr_absent() -> None:
    pub = QuizQuestionPublic.model_validate(
        _row(
            question_type="ordering",
            match_pairs=None,
            ordering_sequence=["s1", "s2", "s3", "s4"],
        )
    )
    assert pub.options == []
    assert sorted(pub.ordering_items) == ["s1", "s2", "s3", "s4"]


def test_answer_keys_still_never_serialize() -> None:
    """The security invariant must survive the missing-attr fix."""
    dumped = QuizQuestionPublic.model_validate(_row()).model_dump()
    assert "match_pairs" not in dumped
    assert "ordering_sequence" not in dumped
