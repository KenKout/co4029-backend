"""Integration coverage placeholder for configured-deadline interview sweeps.

The executable unit suite in ``test_interview_lifecycle_sweep.py`` covers the
lifecycle contract without PostgreSQL: only sessions with an explicit
``time_limit_minutes`` can expire. Untimed text, voice, and hybrid sessions
remain resumable regardless of inactivity.

Add database-backed coverage here when a stable Postgres fixture is available:
* an expired timed session becomes ``timed_out`` or ``abandoned``;
* an untimed session remains ``in_progress`` after more than 30 minutes.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.skip(
    reason="Requires a dedicated Postgres lifecycle fixture; unit coverage is executable."
)
