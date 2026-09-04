"""``end_and_flag`` is refused on write but still readable on existing rows.

The teacher UI offered a "End and flag after threshold" policy that the platform
never delivered:

* ``services/taking.py`` downgrades ``END_AND_FLAG`` to ``WARN_AND_REDIRECT``
  whenever ``interview_security_allow_session_termination`` is false, and that
  setting defaults to false (``core/config.py``);
* even with the flag on, the action only puts ``is_finished`` in the response
  payload — nothing stamps ``interview_sessions.status`` or ``ended_at``, so a
  learner whose client ignores the signal simply keeps answering.

Two candidates in the same cohort could therefore be graded under different
rules depending on their client. The option is withdrawn at the API boundary
(not merely hidden in the UI), while the read schema keeps the value so configs
authored earlier still serialise.
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

_WRITE_ALLOWED = ("continue_and_log", "warn_and_continue")


def _create_kwargs(**overrides: Any) -> dict[str, Any]:
    """Minimum valid ``InterviewConfigCreate`` body."""
    return {
        "title": "Interview",
        "course_id": uuid.uuid4(),
        "module_id": uuid.uuid4(),
        **overrides,
    }


@pytest.mark.parametrize("policy", _WRITE_ALLOWED)
def test_create_accepts_the_enforceable_policies(policy: str) -> None:
    config = InterviewConfigCreate(**_create_kwargs(security_response_policy=policy))
    assert config.security_response_policy == policy


@pytest.mark.parametrize("policy", _WRITE_ALLOWED)
def test_update_accepts_the_enforceable_policies(policy: str) -> None:
    patch = InterviewConfigUpdate(**{"security_response_policy": policy})
    assert patch.security_response_policy == policy


def test_create_rejects_end_and_flag() -> None:
    with pytest.raises(ValidationError) as excinfo:
        InterviewConfigCreate(**_create_kwargs(security_response_policy="end_and_flag"))
    assert "security_response_policy" in str(excinfo.value)


def test_update_rejects_end_and_flag() -> None:
    with pytest.raises(ValidationError) as excinfo:
        InterviewConfigUpdate(**{"security_response_policy": "end_and_flag"})
    assert "security_response_policy" in str(excinfo.value)


def test_default_policy_is_unchanged() -> None:
    """Withdrawing a value must not move the default for new configs."""
    assert InterviewConfigCreate(**_create_kwargs()).security_response_policy == (
        "warn_and_continue"
    )


def test_read_schema_still_accepts_a_legacy_end_and_flag_row() -> None:
    """A config stored before the withdrawal must keep serialising, not 500.

    The DB CHECK still permits ``end_and_flag`` and no migration rewrites those
    rows, so the READ literal has to stay wider than the WRITE literal.
    """
    from abridgeai.features.interviews.schemas.authoring import (
        SecurityResponsePolicyLiteral,
        SecurityResponsePolicyWriteLiteral,
    )

    read_values = set(SecurityResponsePolicyLiteral.__args__)
    write_values = set(SecurityResponsePolicyWriteLiteral.__args__)

    assert "end_and_flag" in read_values, "read schema must tolerate legacy rows"
    assert "end_and_flag" not in write_values, "write schema must refuse it"
    assert write_values < read_values, "write set must be a strict subset of read"
