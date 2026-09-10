"""Browser-integrity scoring: pure rules + authoring-schema validation.

Covers the decision contract (2026-09-10):

* weights are three INDEPENDENT 1..5 numbers, defaulting 3/1/2;
* the warning threshold defaults to 3 (range 1..20);
* only tab_switch / focus_lost / fullscreen_exit score — reconnect,
  disconnect and the server-only warning_issued never do;
* a snapshot with a corrupt/missing key degrades to the shipped default
  (scoring can never be silently switched off by bad data);
* the crossing decision is inclusive (>=) and computed from the score AFTER
  the batch, so one tab_switch (weight 3) hits the default threshold 3
  immediately while focus_lost (weight 1) needs three events;
* the custom-refusal write contract carries EN only (VI was removed by
  migration 0112) and the range bounds are enforced on write.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from pydantic import ValidationError

from abridgeai.features.interviews.schemas.authoring import (
    InterviewConfigCreate,
    InterviewConfigUpdate,
)
from abridgeai.features.interviews.schemas.integrity import (
    DEFAULT_WEIGHTS,
    add_weighted_score,
    is_scored_event,
    reaches_threshold,
    warn_once_and_score,
    weight_for,
)


def _create_kwargs(**overrides: Any) -> dict[str, Any]:
    return {
        "title": "Interview",
        "course_id": uuid.uuid4(),
        "module_id": uuid.uuid4(),
        **overrides,
    }


# ── defaults & validation ────────────────────────────────────────────────────


def test_default_weights_and_threshold_match_the_decision() -> None:
    config = InterviewConfigCreate(**_create_kwargs())
    assert config.integrity_weight_tab_switch == 3
    assert config.integrity_weight_focus_lost == 1
    assert config.integrity_weight_fullscreen_exit == 2
    assert config.integrity_score_threshold == 3
    assert DEFAULT_WEIGHTS == {
        "tab_switch": 3,
        "focus_lost": 1,
        "fullscreen_exit": 2,
    }


@pytest.mark.parametrize(
    ("field", "low", "high"),
    [
        ("integrity_weight_tab_switch", 1, 5),
        ("integrity_weight_focus_lost", 1, 5),
        ("integrity_weight_fullscreen_exit", 1, 5),
        ("integrity_score_threshold", 1, 20),
    ],
)
def test_create_rejects_out_of_range_policy(field: str, low: int, high: int) -> None:
    with pytest.raises(ValidationError):
        InterviewConfigCreate(**_create_kwargs(**{field: low - 1}))
    with pytest.raises(ValidationError):
        InterviewConfigCreate(**_create_kwargs(**{field: high + 1}))


def test_update_accepts_partial_policy_patch() -> None:
    patch = InterviewConfigUpdate(**{"integrity_weight_focus_lost": 5})
    assert patch.integrity_weight_focus_lost == 5
    # Everything else stays "unchanged" (None), never a spurious 0.
    assert patch.integrity_weight_tab_switch is None
    assert patch.integrity_score_threshold is None


def test_update_rejects_out_of_range_weight() -> None:
    with pytest.raises(ValidationError) as excinfo:
        InterviewConfigUpdate(**{"integrity_weight_tab_switch": 6})
    assert "integrity_weight_tab_switch" in str(excinfo.value)


def test_custom_refusal_write_contract_is_english_only() -> None:
    config = InterviewConfigCreate(
        **_create_kwargs(security_custom_refusal_en="Focus on the question.")
    )
    assert config.security_custom_refusal_en == "Focus on the question."
    assert "security_custom_refusal_vi" not in InterviewConfigCreate.model_fields
    assert "security_custom_refusal_vi" not in InterviewConfigUpdate.model_fields


# ── scoring rules ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("event_type", ["tab_switch", "focus_lost", "fullscreen_exit"])
def test_scored_events_are_exactly_the_three_browser_signals(event_type: str) -> None:
    assert is_scored_event(event_type)


@pytest.mark.parametrize("event_type", ["warning_issued", "reconnect", "disconnect"])
def test_server_and_connectivity_events_never_score(event_type: str) -> None:
    assert not is_scored_event(event_type)
    assert add_weighted_score(2, event_type, {}) == 2


def test_one_tab_switch_reaches_the_default_threshold() -> None:
    policy = {"tab_switch": 3, "focus_lost": 1, "fullscreen_exit": 2, "score_threshold": 3}
    assert reaches_threshold(0, weight_for("tab_switch", policy), 3)


def test_focus_lost_needs_three_events_at_the_default_weight() -> None:
    policy = {"tab_switch": 3, "focus_lost": 1, "fullscreen_exit": 2, "score_threshold": 3}
    score = 0
    for _ in range(3):
        score = add_weighted_score(score, "focus_lost", policy)
        if score < 3:
            assert not warn_once_and_score(score, 3).reaches_threshold
    assert score == 3
    assert warn_once_and_score(score, 3).reaches_threshold


def test_mixed_accumulation() -> None:
    policy = {"tab_switch": 3, "focus_lost": 1, "fullscreen_exit": 2, "score_threshold": 3}
    score = 0
    score = add_weighted_score(score, "focus_lost", policy)  # 1
    score = add_weighted_score(score, "fullscreen_exit", policy)  # 3
    assert score == 3
    assert warn_once_and_score(score, 3).reaches_threshold


def test_crossing_is_inclusive_not_strictly_greater() -> None:
    assert reaches_threshold(2, 1, 3)
    assert not reaches_threshold(1, 1, 3)


def test_corrupt_snapshot_degrades_to_the_default_weight() -> None:
    # A bad key must never switch scoring OFF (direction-safe fallback).
    assert weight_for("tab_switch", {"tab_switch": "garbage"}) == 3
    assert weight_for("focus_lost", {}) == 1
    assert weight_for("tab_switch", {"tab_switch": None}) == 3
