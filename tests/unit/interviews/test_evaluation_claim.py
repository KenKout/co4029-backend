"""One evaluation owner per session, enforced by a DB claim with a lease.

The publish guard shipped in the previous commit reads ``pass_verdict`` at the
START of the job and refuses to overwrite a verdict at the END. That closes the
two observed symptoms (a stale failure relabelling a graded session, and a
double grade) but it is a TOCTOU check: between the read and the write sits a
full grading pass of one to two minutes. Two jobs that both start while
``pass_verdict`` is still NULL both pass the gate and run concurrently.

What that still costs:

* the published verdict becomes whichever job commits LAST. The judge is an LLM
  and is not deterministic, so two passes over the same transcript can disagree
  on ``met_count`` — one pass, one fail — and the student's result is decided by
  scheduling. That is a cohort-fairness problem, not a performance one.
* two gap reports for one interview (``_persist_gap_report`` is
  read-then-insert, and the table had no unique constraint).
* double LLM spend.

So exclusivity moves into the database: an atomic conditional UPDATE claims the
session for one owner token with a lease expiry. The lease is deliberately
longer than ``WorkerSettings.job_timeout``, so a job that is still alive always
holds a valid lease and a job that died has its claim reclaimed automatically —
no renewal machinery, no advisory lock (evaluation commits several times
mid-run, and every commit would release a transaction-scoped lock).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest

from abridgeai.features.interviews.services.evaluation_claim import (
    EVALUATION_LEASE_SECONDS,
    lease_expiry_for,
    new_claim_token,
)
from abridgeai.workers.arq_app import WorkerSettings

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


def test_the_lease_outlives_the_hard_job_timeout() -> None:
    """A live job must always hold a valid lease.

    ARQ kills a task at ``job_timeout``, so a lease longer than that cannot
    expire under a job that is still running — which is what lets us skip
    renewal entirely. If someone lowers the lease below the timeout, a
    long-but-healthy evaluation would have its claim stolen mid-run and two
    jobs would grade the same session again.
    """
    assert WorkerSettings.job_timeout < EVALUATION_LEASE_SECONDS, (
        "a lease shorter than job_timeout can expire under a running job"
    )


def test_the_lease_expiry_is_derived_from_the_claim_time() -> None:
    assert lease_expiry_for(_NOW) == _NOW + timedelta(seconds=EVALUATION_LEASE_SECONDS)


def test_each_claim_gets_a_distinct_token() -> None:
    """The token is what proves ownership at publish time."""
    assert new_claim_token() != new_claim_token()


class _Row:
    """Stand-in for the claim columns of one ``interview_sessions`` row."""

    def __init__(
        self,
        *,
        pass_verdict: bool | None = None,
        token: object | None = None,
        expires_at: datetime | None = None,
    ) -> None:
        self.pass_verdict = pass_verdict
        self.evaluation_claim_token = token
        self.evaluation_claim_expires_at = expires_at


class _FakeClaimDB:
    """Executes the claim predicates in Python against one in-memory row.

    The real helpers live in ``queries.sessions`` and run as a single
    conditional UPDATE. This mirrors the predicate so the OWNERSHIP RULES can be
    asserted without Postgres; the SQL itself is covered by the integration
    suite.
    """

    def __init__(self, row: _Row) -> None:
        self.row = row
        self.commits = 0

    def try_claim(self, *, token: object, now: datetime) -> bool:
        row = self.row
        if row.pass_verdict is not None:
            return False
        held = row.evaluation_claim_token is not None and (
            row.evaluation_claim_expires_at is not None
            and row.evaluation_claim_expires_at > now
        )
        if held:
            return False
        row.evaluation_claim_token = token
        row.evaluation_claim_expires_at = lease_expiry_for(now)
        self.commits += 1
        return True

    def owns(self, *, token: object, now: datetime) -> bool:
        row = self.row
        return (
            row.evaluation_claim_token == token
            and row.evaluation_claim_expires_at is not None
            and row.evaluation_claim_expires_at > now
        )


def test_the_first_claimer_wins_and_the_second_is_refused() -> None:
    db = _FakeClaimDB(_Row())
    first, second = new_claim_token(), new_claim_token()

    assert db.try_claim(token=first, now=_NOW) is True
    assert db.try_claim(token=second, now=_NOW) is False, (
        "two concurrent jobs would grade the same session and race to publish"
    )
    assert db.owns(token=first, now=_NOW) is True
    assert db.owns(token=second, now=_NOW) is False


def test_an_expired_lease_is_reclaimable() -> None:
    """A worker that died must not hold the session hostage."""
    dead = new_claim_token()
    db = _FakeClaimDB(_Row(token=dead, expires_at=_NOW - timedelta(minutes=1)))
    fresh = new_claim_token()

    assert db.try_claim(token=fresh, now=_NOW) is True
    assert db.owns(token=fresh, now=_NOW) is True
    assert db.owns(token=dead, now=_NOW) is False, (
        "the dead owner must not be able to publish or stamp a failure"
    )


def test_an_already_graded_session_cannot_be_claimed() -> None:
    db = _FakeClaimDB(_Row(pass_verdict=False))

    assert db.try_claim(token=new_claim_token(), now=_NOW) is False, (
        "pass_verdict=False is a published judgement, not an absence of one"
    )


def test_a_stale_owner_does_not_own_after_its_lease_lapses() -> None:
    """The publish check is ownership, not merely 'did I claim once'."""
    token = new_claim_token()
    db = _FakeClaimDB(_Row())
    assert db.try_claim(token=token, now=_NOW) is True

    later = _NOW + timedelta(seconds=EVALUATION_LEASE_SECONDS + 1)
    assert db.owns(token=token, now=later) is False


@pytest.mark.parametrize("verdict", [True, False])
def test_ownership_is_independent_of_the_verdict_value(verdict: bool) -> None:
    token = new_claim_token()
    db = _FakeClaimDB(_Row())
    db.try_claim(token=token, now=_NOW)
    db.row.pass_verdict = verdict

    assert db.owns(token=token, now=_NOW) is True, (
        "the owner must still be able to finish its own transaction"
    )
    assert db.try_claim(token=uuid4(), now=_NOW) is False
