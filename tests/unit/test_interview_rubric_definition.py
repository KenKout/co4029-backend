"""Unit tests for the teacher-authored scoring rubric (``evaluation_rubric``).

Context / regression being locked in
------------------------------------
``evaluate_session`` accepted a ``config`` mapping from which it resolved rubric
weights, but the real caller (``services.evaluation.evaluate_and_generate_report``)
never passed one. Every session was therefore graded against the four-criterion
equal-weight default and a teacher-authored rubric had no effect whatsoever.

The scoring rubric lives under the ``evaluation_rubric`` key inside
``supplementary_instructions`` — deliberately NOT ``rubric_weights``, which the
GENERATION stage already claims for the question TYPE MIX
(technical/behavioral/situational). Reusing that key would have graded candidates
against criteria literally named "technical"/"behavioral"; the test at the bottom
of this module asserts the two namespaces stay independent.
"""

from __future__ import annotations

import json

import pytest

from abridgeai.features.interviews.ai.stages.evaluation.rubric import (
    DEFAULT_CRITERIA,
    resolve_rubric_definition,
    resolve_supplementary_notes,
)
from abridgeai.features.interviews.ai.stages.generation.resolve import resolve_type_mix


def _supplementary(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ── Full form: name + weight + description ───────────────────────────────────


def test_full_form_reads_weights_and_descriptions() -> None:
    raw = _supplementary(
        {
            "evaluation_rubric": {
                "criteria": [
                    {"name": "depth", "weight": 3, "description": "Cites concrete evidence."},
                    {"name": "clarity", "weight": 1},
                ]
            }
        }
    )

    definition = resolve_rubric_definition(raw)

    assert definition.criteria == ("depth", "clarity")
    assert definition.weights["depth"] == pytest.approx(0.75)
    assert definition.weights["clarity"] == pytest.approx(0.25)
    assert definition.descriptions == {"depth": "Cites concrete evidence."}


def test_weight_omitted_defaults_to_one() -> None:
    raw = _supplementary({"evaluation_rubric": {"criteria": [{"name": "a"}, {"name": "b"}]}})

    definition = resolve_rubric_definition(raw)

    assert definition.weights["a"] == pytest.approx(0.5)
    assert definition.weights["b"] == pytest.approx(0.5)


def test_criterion_alias_key_is_accepted() -> None:
    raw = _supplementary(
        {"evaluation_rubric": {"criteria": [{"criterion": "reasoning", "weight": 2}]}}
    )

    assert resolve_rubric_definition(raw).criteria == ("reasoning",)


# ── Shorthand forms ──────────────────────────────────────────────────────────


def test_weight_mapping_shorthand() -> None:
    raw = _supplementary({"evaluation_rubric": {"depth": 3, "clarity": 1}})

    definition = resolve_rubric_definition(raw)

    assert set(definition.criteria) == {"depth", "clarity"}
    assert definition.weights["depth"] == pytest.approx(0.75)
    assert definition.descriptions == {}


def test_name_list_shorthand_is_equal_weight() -> None:
    raw = _supplementary({"evaluation_rubric": ["depth", "clarity", "rigour"]})

    definition = resolve_rubric_definition(raw)

    assert definition.criteria == ("depth", "clarity", "rigour")
    assert all(w == pytest.approx(1 / 3) for w in definition.weights.values())


# ── Fallback safety: grading must never break on a bad config ────────────────


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "Please focus on database normalisation.",  # prose, not JSON
        "{not valid json",
        _supplementary({}),  # no evaluation_rubric key
        _supplementary({"evaluation_rubric": {}}),
        _supplementary({"evaluation_rubric": []}),
        _supplementary({"evaluation_rubric": {"criteria": []}}),
        _supplementary({"evaluation_rubric": {"a": -1, "b": 0}}),  # non-positive weights
        _supplementary({"evaluation_rubric": {"criteria": [{"name": "   "}]}}),
        _supplementary({"evaluation_rubric": 42}),
    ],
)
def test_falls_back_to_default_rubric(raw: str | None) -> None:
    definition = resolve_rubric_definition(raw)

    assert definition.criteria == DEFAULT_CRITERIA
    assert all(w == pytest.approx(0.25) for w in definition.weights.values())
    assert definition.descriptions == {}


def test_weights_are_normalised_to_sum_one() -> None:
    raw = _supplementary({"evaluation_rubric": {"a": 70, "b": 20, "c": 10}})

    weights = resolve_rubric_definition(raw).weights

    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["a"] == pytest.approx(0.7)


def test_duplicate_names_keep_the_first_entry() -> None:
    raw = _supplementary(
        {
            "evaluation_rubric": {
                "criteria": [
                    {"name": "depth", "weight": 3, "description": "first"},
                    {"name": "depth", "weight": 9, "description": "second"},
                ]
            }
        }
    )

    definition = resolve_rubric_definition(raw)

    assert definition.criteria == ("depth",)
    assert definition.descriptions["depth"] == "first"


def test_criteria_count_is_capped() -> None:
    raw = _supplementary({"evaluation_rubric": [f"criterion_{i}" for i in range(25)]})

    assert len(resolve_rubric_definition(raw).criteria) == 10


def test_long_criterion_name_is_truncated() -> None:
    raw = _supplementary({"evaluation_rubric": ["x" * 200]})

    (name,) = resolve_rubric_definition(raw).criteria
    assert len(name) == 64


# ── Namespace independence from the generation stage ─────────────────────────


def test_scoring_rubric_and_type_mix_do_not_collide() -> None:
    """``rubric_weights`` drives the question type mix; it must NOT be graded on.

    A config carrying BOTH keys must yield the type mix from ``rubric_weights``
    and the scoring criteria from ``evaluation_rubric`` — never the other way
    round, and never one bleeding into the other.
    """
    raw = _supplementary(
        {
            "rubric_weights": {"technical": 70, "behavioral": 20, "situational": 10},
            "evaluation_rubric": {"criteria": [{"name": "depth", "weight": 1}]},
        }
    )

    assert resolve_type_mix(raw) == {"technical": 70, "behavioral": 20, "situational": 10}
    assert resolve_rubric_definition(raw).criteria == ("depth",)


def test_type_mix_only_config_still_grades_on_defaults() -> None:
    """A teacher who only set a type mix must not be graded on type names."""
    raw = _supplementary({"rubric_weights": {"technical": 60, "behavioral": 30, "situational": 10}})

    definition = resolve_rubric_definition(raw)

    assert definition.criteria == DEFAULT_CRITERIA
    assert "technical" not in definition.criteria


# ── Prose-notes extraction for the generation prompt ─────────────────────────


def test_plain_prose_is_returned_unchanged() -> None:
    """The common case: the field is free text, not JSON — return it verbatim."""
    assert (
        resolve_supplementary_notes("Focus on real-world scenarios, avoid rote recall.")
        == "Focus on real-world scenarios, avoid rote recall."
    )


def test_none_and_blank_yield_empty_string() -> None:
    assert resolve_supplementary_notes(None) == ""
    assert resolve_supplementary_notes("   ") == ""


def test_json_config_returns_only_the_notes_key() -> None:
    """When the field holds JSON, only the human prose reaches the prompt."""
    raw = _supplementary(
        {
            "notes": "Prioritise applied questions.",
            "evaluation_rubric": {"criteria": [{"name": "depth", "weight": 2}]},
            "rubric_weights": {"technical": 70, "behavioral": 30, "situational": 0},
            "question_count": 8,
        }
    )

    assert resolve_supplementary_notes(raw) == "Prioritise applied questions."


def test_json_config_without_notes_yields_empty_string() -> None:
    """Structured-only config must NOT leak its JSON blob into the prompt."""
    raw = _supplementary({"evaluation_rubric": {"criteria": [{"name": "depth"}]}})

    assert resolve_supplementary_notes(raw) == ""
