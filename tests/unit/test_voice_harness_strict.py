"""Unit tests for the voice harness's strict adaptive verification.

These exercise the pure decision helpers ``_strict_adaptive_ok`` /
``_strict_failure_reason`` against synthetic ``ScenarioResult`` objects — NO
live services, NO DB. They lock in the contract that strict mode fails unless
the adaptive brain genuinely ran, using PERSISTED state/decision signals as the
source of truth (never utterance text).

The live end-to-end assertion lives in the opt-in ``voice_live`` wrapper.
"""

from __future__ import annotations

from scripts.voice_harness.run_harness import (
    ScenarioResult,
    _security_failure_reason,
    _security_requirement_ok,
    _strict_adaptive_ok,
    _strict_failure_reason,
)


def _adaptive_result(**overrides: object) -> ScenarioResult:
    """A ScenarioResult that represents a healthy adaptive run by default."""
    base = ScenarioResult(
        ok=True,
        language="en",
        session_id="s",
        runtime_state_count=1,
        adaptive_ran=True,
        message_count=4,
        adaptive_actions=["ask_for_example", "transition_topic"],
        state_version=2,
        decision_count=2,
        fallback_count=0,
    )
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def test_strict_ok_for_genuine_adaptive_run() -> None:
    assert _strict_adaptive_ok(_adaptive_result()) is True


def test_strict_fails_when_no_runtime_state() -> None:
    """Legacy run: no runtime-state row, no adaptive execution."""
    r = _adaptive_result(
        runtime_state_count=0,
        adaptive_ran=False,
        state_version=None,
        adaptive_actions=[],
        decision_count=0,
    )
    assert _strict_adaptive_ok(r) is False
    # Failure message must point at the disabled agent flag.
    reason = _strict_failure_reason(r)
    assert "adaptive feature flag DISABLED" in reason
    assert "ADAPTIVE_INTERVIEWER_VOICE_ENABLED" in reason


def test_strict_fails_when_no_decision_persisted() -> None:
    r = _adaptive_result(adaptive_actions=[], decision_count=0)
    assert _strict_adaptive_ok(r) is False
    assert "no structured adaptive action" in _strict_failure_reason(r)


def test_strict_fails_when_state_version_null() -> None:
    r = _adaptive_result(state_version=None)
    assert _strict_adaptive_ok(r) is False
    assert "version is null" in _strict_failure_reason(r)


def test_strict_fails_when_every_turn_fell_back() -> None:
    """A run where all adaptive turns used the deterministic fallback is not a
    genuine adaptive exercise."""
    r = _adaptive_result(decision_count=2, fallback_count=2)
    assert _strict_adaptive_ok(r) is False
    assert "deterministic fallback" in _strict_failure_reason(r)


def test_strict_ok_with_partial_fallback() -> None:
    """Some fallback is fine as long as not EVERY turn fell back."""
    r = _adaptive_result(decision_count=3, fallback_count=1)
    assert _strict_adaptive_ok(r) is True


def test_security_requirement_is_opt_in() -> None:
    assert _security_requirement_ok(_adaptive_result()) is True


def test_security_requirement_uses_persisted_assessed_and_blocked_events() -> None:
    result = _adaptive_result(
        required_security_blocks=3,
        security_assessment_count=3,
        security_blocked_count=3,
    )
    assert _security_requirement_ok(result) is True


def test_security_requirement_fails_when_voice_turn_was_not_blocked() -> None:
    result = _adaptive_result(
        required_security_blocks=2,
        security_assessment_count=2,
        security_blocked_count=1,
    )
    assert _security_requirement_ok(result) is False
    reason = _security_failure_reason(result)
    assert "required at least 2" in reason
    assert "blocked=1" in reason
