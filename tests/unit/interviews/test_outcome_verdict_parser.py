"""Unit tests for the per-outcome verdict parser (thesis §4.3 gate).

Pure parser tests — no DB, no LLM, no network. Covers the contract that
``parse_outcome_verdicts`` returns exactly one verdict per expected outcome id,
in order, defaulting any outcome the judge omitted (or returned malformed) to
``met=False`` (the safe direction — never inflates a pass).
"""

from __future__ import annotations

from uuid import uuid4

from abridgeai.features.interviews.ai.stages.evaluation.parsers_outcome_verdicts import (
    parse_outcome_verdicts,
)


def test_well_formed_payload_maps_each_outcome() -> None:
    o1, o2 = uuid4(), uuid4()
    payload = {
        "verdicts": [
            {"outcome_id": str(o1), "met": True, "reasoning": "Solid.", "evidence": "did X"},
            {"outcome_id": str(o2), "met": False, "reasoning": "Vague.", "evidence": None},
        ]
    }
    parsed = parse_outcome_verdicts(payload, expected_outcome_ids=[o1, o2])
    assert [v.outcome_id for v in parsed] == [o1, o2]
    assert parsed[0].met is True
    assert parsed[0].evidence == "did X"
    assert parsed[1].met is False


def test_missing_outcome_defaults_to_not_met() -> None:
    o1, o2 = uuid4(), uuid4()
    payload = {"verdicts": [{"outcome_id": str(o1), "met": True, "reasoning": "ok"}]}
    parsed = parse_outcome_verdicts(payload, expected_outcome_ids=[o1, o2])
    assert parsed[1].outcome_id == o2
    assert parsed[1].met is False
    assert "no verdict" in parsed[1].reasoning.lower()


def test_unknown_outcome_id_is_ignored() -> None:
    o1 = uuid4()
    stray = uuid4()
    payload = {
        "verdicts": [
            {"outcome_id": str(o1), "met": True, "reasoning": "ok"},
            {"outcome_id": str(stray), "met": True, "reasoning": "not expected"},
        ]
    }
    parsed = parse_outcome_verdicts(payload, expected_outcome_ids=[o1])
    assert len(parsed) == 1
    assert parsed[0].outcome_id == o1


def test_malformed_rows_dropped() -> None:
    o1 = uuid4()
    payload = {
        "verdicts": [
            "not-a-dict",
            {"outcome_id": "not-a-uuid", "met": True},
            {"outcome_id": str(o1), "met": "garbage"},  # bad bool → dropped
        ]
    }
    parsed = parse_outcome_verdicts(payload, expected_outcome_ids=[o1])
    assert parsed[0].met is False  # defaulted because the row was dropped


def test_string_boolean_spellings_accepted() -> None:
    o1, o2 = uuid4(), uuid4()
    payload = {
        "verdicts": [
            {"outcome_id": str(o1), "met": "met", "reasoning": "ok"},
            {"outcome_id": str(o2), "met": "not met", "reasoning": "ok"},
        ]
    }
    parsed = parse_outcome_verdicts(payload, expected_outcome_ids=[o1, o2])
    assert parsed[0].met is True
    assert parsed[1].met is False


def test_none_payload_all_not_met() -> None:
    o1, o2 = uuid4(), uuid4()
    parsed = parse_outcome_verdicts(None, expected_outcome_ids=[o1, o2])
    assert all(v.met is False for v in parsed)
    assert len(parsed) == 2


def test_empty_expected_returns_empty() -> None:
    assert parse_outcome_verdicts({"verdicts": []}, expected_outcome_ids=[]) == []
