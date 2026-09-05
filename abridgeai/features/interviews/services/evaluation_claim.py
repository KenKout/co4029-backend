"""Exclusive-owner tokens and lease arithmetic for interview evaluation.

Why a DB claim and not a simpler guard
--------------------------------------
Checking ``pass_verdict`` at the start of the job and refusing to overwrite it at
the end is a TOCTOU check: a full grading pass (one to two minutes of LLM calls)
sits between the read and the write, so two jobs that both start while the
verdict is NULL both pass the gate. The recovery sweep makes that reachable by
design — it enqueues under a per-attempt job ID (ARQ cannot dedupe it) while the
original job may still be running, since ``job_timeout`` (20 min) overlaps the
recovery grace (15 min).

The cost of two concurrent graders is not just wasted LLM spend: the judge is an
LLM and is not deterministic, so two passes over one transcript can disagree, and
the published verdict becomes whichever job commits last. A student's pass/fail
decided by scheduling is a fairness defect.

Why a lease and not an advisory lock
------------------------------------
``evaluate_and_generate_report`` commits several times mid-run (the
``generation_run`` row, the outcome upserts, the gap report). A transaction-scoped
advisory lock is released by the first of those commits, so it cannot protect the
span that matters.

Why no renewal machinery
------------------------
The lease is deliberately longer than ``WorkerSettings.job_timeout``. ARQ kills a
task at that timeout, so a job that is still alive ALWAYS holds a valid lease and
never needs to renew; a job that died has its claim reclaimed automatically once
the lease lapses. A test pins the ordering, because a lease shorter than the
timeout would let a healthy-but-slow evaluation have its claim stolen mid-run —
reintroducing exactly the concurrency this module removes.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from uuid import UUID, uuid4

# Must stay ABOVE ``WorkerSettings.job_timeout`` (1200s). The margin covers the
# gap between ARQ's timeout firing and the row being reclaimable, so a fresh
# recovery cannot start while the killed job's transaction is still unwinding.
EVALUATION_LEASE_SECONDS = 1800


def new_claim_token() -> UUID:
    """A fresh owner token, unique per claim attempt.

    Identity is per-CLAIM rather than per-session or per-job: it is what a job
    presents at publish time to prove the lease it acquired is still its own.
    """
    return uuid4()


def lease_expiry_for(claimed_at: datetime) -> datetime:
    """When a claim taken at ``claimed_at`` stops being authoritative."""
    return claimed_at + timedelta(seconds=EVALUATION_LEASE_SECONDS)


__all__ = [
    "EVALUATION_LEASE_SECONDS",
    "lease_expiry_for",
    "new_claim_token",
]
