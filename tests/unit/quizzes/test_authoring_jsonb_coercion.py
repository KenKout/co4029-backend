"""Regression tests for JSONB coercion on the authoring write path (Phase 7).

Bug this guards against (live 500 on POST /teacher/quizzes/{id}/questions with
``question_type=matching``):

The authoring router accepts loose ``dict`` bodies and projects them through a
private ``_AttrShim``. That shim turns any *list of dicts* into a list of
``_AttrShim`` objects so services can use attribute access (needed for
``options``). But ``match_pairs`` / ``ordering_sequence`` are written straight
into **JSONB columns**, and psycopg cannot ``json.dumps`` a ``_AttrShim`` →
``TypeError: Object of type _AttrShim is not JSON serializable`` → HTTP 500.

``_as_plain_json`` unwraps shims (and real Pydantic models) back to plain
dicts/lists before assignment. These tests pin that behaviour without touching
the DB.
"""

from __future__ import annotations

import json
from typing import Any

from abridgeai.features.quizzes.services.authoring import _as_plain_json


class _FakeShim:
    """Stands in for the router's ``_AttrShim`` (same contract: model_dump)."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = dict(data)

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return dict(self._data)


def test_unwraps_list_of_shims_to_plain_dicts() -> None:
    """The exact shape that caused the live 500."""
    shimmed = [
        _FakeShim({"left": "Term 1", "right": "Match 1"}),
        _FakeShim({"left": "Term 2", "right": "Match 2"}),
    ]
    result = _as_plain_json(shimmed)

    assert result == [
        {"left": "Term 1", "right": "Match 1"},
        {"left": "Term 2", "right": "Match 2"},
    ]
    # The whole point: it must now survive JSON serialization.
    json.dumps(result)


def test_shim_list_is_not_json_serializable_without_coercion() -> None:
    """Proves the guard is load-bearing, not decorative."""
    shimmed = [_FakeShim({"left": "A", "right": "1"})]
    try:
        json.dumps(shimmed)
    except TypeError:
        pass  # expected — this is the bug being prevented
    else:  # pragma: no cover
        raise AssertionError("expected raw shim list to fail json.dumps")


def test_plain_data_passes_through_unchanged() -> None:
    pairs = [{"left": "A", "right": "1"}]
    assert _as_plain_json(pairs) == pairs
    assert _as_plain_json(["one", "two"]) == ["one", "two"]
    assert _as_plain_json("scalar") == "scalar"
    assert _as_plain_json(7) == 7
    assert _as_plain_json(None) is None


def test_unwraps_nested_shims_inside_dicts() -> None:
    nested = {"outer": [_FakeShim({"k": "v"})], "plain": 1}
    result = _as_plain_json(nested)
    assert result == {"outer": [{"k": "v"}], "plain": 1}
    json.dumps(result)


def test_ordering_sequence_of_scalars_is_untouched() -> None:
    seq = ["Step 1", "Step 2", "Step 3"]
    assert _as_plain_json(seq) == seq
    json.dumps(_as_plain_json(seq))
