"""ARQ worker app — aggregates feature ``JOBS`` lists into a single ``WorkerSettings``.

Each feature module exposes ``JOBS: list[Callable]``. ``arq_app`` imports
those lists explicitly (per plan §5138 — explicit imports keep startup
fast and surface missing features at import time, not at first job run).
"""

from __future__ import annotations

from arq.connections import RedisSettings
from arq.cron import CronJob, cron

from abridgeai.core.config import get_settings
from abridgeai.features.interviews.workers import JOBS as INTERVIEW_JOBS
from abridgeai.features.materials.workers import JOBS as MATERIAL_JOBS
from abridgeai.features.materials.workers.cron import cleanup_orphaned_uploads_task
from abridgeai.features.notifications.workers import JOBS as NOTIFICATION_JOBS
from abridgeai.features.quizzes.workers import JOBS as QUIZ_JOBS
from abridgeai.features.spaced_repetition.workers import JOBS as SR_JOBS
from abridgeai.features.spaced_repetition.workers import scan_due_cards_task


class WorkerSettings:
    """ARQ worker configuration (composed jobs + cron + tunables).

    Tunables locked by plan §5130:
      * ``max_jobs=10``  — concurrency per worker
      * ``job_timeout=600`` — 10 min hard cap per task
      * ``keep_result_seconds=3600`` — Redis result retention
      * ``max_tries=3`` — exponential backoff retry budget

    Cron jobs are feature-owned and registered here as they land:
      * Phase 4.5 — ``cleanup_orphaned_uploads_task`` (daily 03:00)
      * Phase 7.5 — ``scan_due_cards_task`` (every hour)
      * Phase 7   — ``daily_compliance_summary_task`` (daily 09:00)
    """

    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    functions = (
        list(MATERIAL_JOBS)
        + list(QUIZ_JOBS)
        + list(INTERVIEW_JOBS)
        + list(NOTIFICATION_JOBS)
        + list(SR_JOBS)
    )
    max_jobs = 10
    job_timeout = 600
    keep_result_seconds = 3600
    max_tries = 3
    cron_jobs: list[CronJob] = [
        cron(cleanup_orphaned_uploads_task, hour={3}, minute=0),
        cron(scan_due_cards_task, minute=0),
    ]


__all__ = ["WorkerSettings"]
