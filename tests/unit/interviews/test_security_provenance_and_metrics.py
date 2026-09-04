"""Regression guards for interview security provenance, metrics, and audit dedupe.

Covers three defects found by auditing the security chain against the live
schema:

1. ``interview_sessions.security_*_version`` only had migration
   ``server_default`` values, so every session claimed the baseline versions
   while ``interview_security_events`` recorded the real ones from the code
   constants. Live dev data had 215/215 sessions on ``rules_version='1.0.0'``
   against events on ``'1.2.0'``.
2. ``security_fallback_rate`` summed ``fallback_status`` over EVERY event type
   but divided by the ``assessed`` count only, so an ``output_leakage_blocked``
   row (which also sets the flag) inflated the rate — observed as ``1.000`` on a
   session whose classifier never failed, and a non-zero numerator on sessions
   with zero assessments.
3. Both the follow-up and next-question output guards ran under the same
   ``turn_key``; since ``record_security_event`` dedupes on
   ``(session_id, turn_id, event_type)`` the second leakage in one turn was
   dropped from the audit log.
"""

from __future__ import annotations

import ast
import inspect

from abridgeai.features.interviews.orchestrator.security import (
    OUTPUT_GUARD_VERSION,
    SECURITY_POLICY_VERSION,
    SECURITY_PROMPT_VERSION,
    SECURITY_RULES_VERSION,
)
from abridgeai.features.interviews.services import security as security_service
from abridgeai.features.interviews.services import taking as taking_service


def _interview_session_construction_kwargs() -> dict[str, str]:
    """Keyword names -> source of the ``InterviewSession(...)`` call in taking.py.

    Parsed via ``ast`` rather than string splitting: the call spans multiple
    lines and contains nested calls, so a naive split on ``)`` truncates it.
    """
    tree = ast.parse(inspect.getsource(taking_service))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "InterviewSession"
        ):
            return {
                keyword.arg: ast.unparse(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
    raise AssertionError("no InterviewSession(...) construction found in taking.py")


def test_start_session_stamps_security_versions_from_code_constants() -> None:
    """The session row must record the versions that actually ran."""
    kwargs = _interview_session_construction_kwargs()

    expected = {
        "security_policy_version": "SECURITY_POLICY_VERSION",
        "security_rules_version": "SECURITY_RULES_VERSION",
        "security_prompt_version": "SECURITY_PROMPT_VERSION",
        "output_guard_version": "OUTPUT_GUARD_VERSION",
    }
    for field, constant in expected.items():
        assert field in kwargs, (
            f"{field} is not stamped at session creation, so the row keeps the "
            "migration server_default and misreports which policy graded the attempt"
        )
        assert kwargs[field] == constant, (
            f"{field} must be stamped from {constant}, not {kwargs[field]!r} — "
            "a hardcoded literal drifts the moment the rules are revised"
        )


def test_security_version_constants_are_the_single_source_of_truth() -> None:
    """Sessions and events must stamp from the same constants."""
    taking_source = inspect.getsource(taking_service)
    events_source = inspect.getsource(security_service.record_security_event)

    for constant in (
        "SECURITY_POLICY_VERSION",
        "SECURITY_RULES_VERSION",
        "SECURITY_PROMPT_VERSION",
        "OUTPUT_GUARD_VERSION",
    ):
        assert constant in taking_source, f"session row does not use {constant}"

    assert "OUTPUT_GUARD_VERSION" in events_source
    # Guard against a literal creeping back in place of the constant.
    assert SECURITY_RULES_VERSION not in ('"1.0.0"', "'1.0.0'")
    assert all(
        isinstance(value, str) and value
        for value in (
            SECURITY_POLICY_VERSION,
            SECURITY_RULES_VERSION,
            SECURITY_PROMPT_VERSION,
            OUTPUT_GUARD_VERSION,
        )
    )


def test_fallback_rate_numerator_is_scoped_to_assessed_events() -> None:
    """Numerator and denominator must come from the same event stream."""
    source = inspect.getsource(security_service.get_security_session_metrics)

    fallback_stmt = source.split("fallbacks = ", 1)[1].split("row = ", 1)[0]
    assert "EV_SECURITY_ASSESSED" in fallback_stmt, (
        "fallback numerator must filter to assessed events; counting every "
        "event type mixes in output_leakage_blocked rows and inflates the rate"
    )
    # The old implementation used an unfiltered sum(case(...)).
    assert "case(" not in fallback_stmt


def _guard_student_output_calls() -> list[dict[str, str]]:
    """Every ``guard_student_output(...)`` call in taking.py, as kwarg maps."""
    tree = ast.parse(inspect.getsource(taking_service))
    calls: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "guard_student_output"
        ):
            calls.append(
                {
                    keyword.arg: ast.unparse(keyword.value)
                    for keyword in node.keywords
                    if keyword.arg is not None
                }
            )
    return calls


def test_output_guard_calls_in_one_turn_use_distinct_turn_ids() -> None:
    """Follow-up and next-question guards must not collide in the audit log."""
    calls = _guard_student_output_calls()
    assert len(calls) >= 3, f"expected at least three output-guard call sites, got {len(calls)}"

    # The two guards that can both fire within a SINGLE answered turn share an
    # ``effective_turn_key``, so each needs its own turn_id namespace.
    shared_turn_key_calls = [
        call for call in calls if call.get("turn_key") == "effective_turn_key"
    ]
    assert len(shared_turn_key_calls) >= 2, (
        "expected the follow-up and next-question guards to share effective_turn_key"
    )

    turn_ids = [call.get("turn_id") for call in shared_turn_key_calls]
    assert all(turn_ids), (
        "guards sharing effective_turn_key must pass an explicit turn_id; otherwise "
        f"record_security_event dedupes the second leakage away (got {turn_ids})"
    )
    assert len(set(turn_ids)) == len(turn_ids), f"turn_id values collide: {turn_ids}"


def test_record_security_event_dedupe_key_still_includes_turn_id() -> None:
    """The turn_id namespacing fix relies on this dedupe key shape."""
    source = inspect.getsource(security_service.record_security_event)
    assert "(turn_id or turn_key)" in source
    assert "InterviewSecurityEvent.turn_id == stable_turn_id" in source
