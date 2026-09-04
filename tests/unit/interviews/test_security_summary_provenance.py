"""The teacher security summary reports WHICH guard judged an attempt.

``interview_sessions`` carries four provenance columns
(``security_policy_version``, ``security_rules_version``,
``security_prompt_version``, ``output_guard_version``). Commit 1637196 made the
session row stamp them from the code constants at creation, but nothing read
them back: a grep of schemas/, routers/ and queries/ returned nothing, so a
flagged attempt — the exact row a teacher inspects during a cohort dispute —
could not answer "flagged under which rules?".

They are surfaced on ``SecuritySessionSummary`` (already the teacher-only,
redacted projection) rather than on the session schemas, so they inherit the
per-config ``security_incident_summary_enabled`` toggle and never leak to a
learner. The values must come off the SESSION ROW, not from today's constants —
an attempt graded months ago was judged by the rules live back then.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest

from abridgeai.features.interviews.routers import authoring_sessions
from abridgeai.features.interviews.schemas.authoring import (
    InterviewSessionSummary,
    SecuritySessionSummary,
)

_PROVENANCE_FIELDS = (
    "policy_version",
    "rules_version",
    "prompt_version",
    "output_guard_version",
)


class _FakeMetrics:
    assessment_count = 4
    blocked_attempt_count = 1
    repeated_attempt_count = 0
    output_leakage_prevented = 0
    security_fallback_rate = 0.25
    average_classification_latency_ms = 12.5


def _session(**overrides: Any) -> SimpleNamespace:
    """A session row as the routers hand it to the summary builder."""
    base = {
        "id": uuid4(),
        "session_security_flagged": True,
        "security_policy_version": "2026-07-19",
        "security_rules_version": "1.2.0",
        "security_prompt_version": "1.1.0",
        "output_guard_version": "1.0.0",
    }
    base.update(overrides)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _stub_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Metrics come from SQL; this test is about provenance plumbing."""

    async def _fake_get_metrics(_db: object, _session_id: object) -> _FakeMetrics:
        return _FakeMetrics()

    from abridgeai.features.interviews.services import security as security_service

    monkeypatch.setattr(security_service, "get_security_session_metrics", _fake_get_metrics)


@pytest.mark.asyncio
async def test_summary_reports_the_versions_stored_on_the_session() -> None:
    summary = await authoring_sessions._security_summary_view(
        None,  # type: ignore[arg-type] -- db unused once metrics are stubbed
        _session(),
        enabled=True,
    )
    assert summary is not None
    assert summary.policy_version == "2026-07-19"
    assert summary.rules_version == "1.2.0"
    assert summary.prompt_version == "1.1.0"
    assert summary.output_guard_version == "1.0.0"
    assert summary.session_flagged is True


@pytest.mark.asyncio
async def test_summary_does_not_substitute_todays_constants() -> None:
    """A historical attempt must report ITS rules, not the current build's."""
    from abridgeai.features.interviews.orchestrator.security import (
        SECURITY_RULES_VERSION,
    )

    legacy = _session(security_rules_version="1.0.0")
    summary = await authoring_sessions._security_summary_view(
        None,  # type: ignore[arg-type]
        legacy,
        enabled=True,
    )
    assert summary is not None
    assert summary.rules_version == "1.0.0"
    if SECURITY_RULES_VERSION != "1.0.0":
        assert summary.rules_version != SECURITY_RULES_VERSION


@pytest.mark.asyncio
async def test_provenance_respects_the_incident_summary_toggle() -> None:
    """Disabling the teacher report must not leak provenance either."""
    summary = await authoring_sessions._security_summary_view(
        None,  # type: ignore[arg-type]
        _session(),
        enabled=False,
    )
    assert summary is None


@pytest.mark.asyncio
async def test_missing_columns_degrade_to_none_instead_of_raising() -> None:
    """Some callers pass a lightweight row shim; absence must not 500."""
    shim = SimpleNamespace(id=uuid4(), session_security_flagged=False)
    summary = await authoring_sessions._security_summary_view(
        None,  # type: ignore[arg-type]
        shim,
        enabled=True,
    )
    assert summary is not None
    for field in _PROVENANCE_FIELDS:
        assert getattr(summary, field) is None


def test_provenance_is_teacher_only_not_on_the_learner_surface() -> None:
    """The fields live on the security summary, never on a learner schema."""
    from abridgeai.features.interviews.schemas.session import InterviewSessionPublic

    learner_fields = set(InterviewSessionPublic.model_fields)
    for field in _PROVENANCE_FIELDS:
        assert field in SecuritySessionSummary.model_fields
        assert field not in learner_fields

    # And the teacher list row reaches them only through security_summary.
    assert "security_summary" in InterviewSessionSummary.model_fields
    for field in _PROVENANCE_FIELDS:
        assert field not in InterviewSessionSummary.model_fields
