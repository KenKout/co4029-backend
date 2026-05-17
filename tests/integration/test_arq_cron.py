"""ARQ cron-job registration introspection (T4.7).

The cron jobs themselves land with later phases (T4.5 cleanup-orphans,
Phase 7.5 SR scan_due_cards, Phase 7 daily_compliance_summary). T4.7
only guarantees the slot is wired so those tasks can register without
edits to ``arq_app``.
"""

from __future__ import annotations

from arq.cron import CronJob

from abridgeai.workers.arq_app import WorkerSettings


def test_cron_jobs_attribute_exists() -> None:
    assert hasattr(WorkerSettings, "cron_jobs")


def test_cron_jobs_is_list() -> None:
    assert isinstance(WorkerSettings.cron_jobs, list)


def test_cron_entries_are_well_typed() -> None:
    for entry in WorkerSettings.cron_jobs:
        assert isinstance(entry, CronJob)
