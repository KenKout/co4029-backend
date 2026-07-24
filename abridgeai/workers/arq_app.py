"""ARQ worker app — aggregates feature ``JOBS`` lists into a single ``WorkerSettings``.

Each feature module exposes ``JOBS: list[Callable]``. ``arq_app`` imports
those lists explicitly (per plan §5138 — explicit imports keep startup
fast and surface missing features at import time, not at first job run).
"""

from __future__ import annotations

from arq.connections import RedisSettings
from arq.cron import CronJob, cron

import abridgeai.ai.models  # noqa: F401
import abridgeai.features.career_paths.models  # noqa: F401
import abridgeai.features.courses.models  # noqa: F401
import abridgeai.features.enrollments.models  # noqa: F401
import abridgeai.features.identity.models  # noqa: F401
import abridgeai.features.interviews.models  # noqa: F401
import abridgeai.features.materials.models  # noqa: F401
import abridgeai.features.notifications.models  # noqa: F401
import abridgeai.features.progress.models  # noqa: F401
import abridgeai.features.quizzes.models  # noqa: F401
import abridgeai.features.spaced_repetition.models  # noqa: F401
from abridgeai.core.config import get_settings
from abridgeai.features.career_paths.workers import snapshot_career_readiness_task
from abridgeai.features.interviews.workers import JOBS as INTERVIEW_JOBS
from abridgeai.features.interviews.workers.lifecycle import sweep_interview_sessions_task
from abridgeai.features.materials.workers import JOBS as MATERIAL_JOBS
from abridgeai.features.materials.workers.cron import cleanup_orphaned_uploads_task
from abridgeai.features.materials.workers.reaper import reconcile_orphaned_ingests_task
from abridgeai.features.notifications.workers import JOBS as NOTIFICATION_JOBS
from abridgeai.features.quizzes.workers import JOBS as QUIZ_JOBS
from abridgeai.features.quizzes.workers.timing import sweep_overdue_attempts_task
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
    job_timeout = 1200  # Nam 23/07/2026: Increase timeout window for long pdf context
    keep_result_seconds = 3600
    max_tries = 3
    cron_jobs: list[CronJob] = [
        cron(cleanup_orphaned_uploads_task, hour={3}, minute=0),
        # Reconcile ingest ProcessingJob rows stuck pending/running with no
        # live ARQ job (dropped across a worker restart / Redis blip). Runs
        # every 3 minutes so a stranded document auto-recovers within one
        # interval instead of spinning forever with no signal.
        cron(reconcile_orphaned_ingests_task, minute=set(range(0, 60, 3))),
        cron(scan_due_cards_task, minute=0),
        # Finalise stale in-progress voice interview sessions every 5 minutes.
        cron(sweep_interview_sessions_task, minute=set(range(0, 60, 5))),
        # FR-6.8 — nightly career readiness snapshots (one row per active
        # career enrollment) feeding manager aggregates + student history.
        cron(snapshot_career_readiness_task, hour={2}, minute=30),
        # Phase 6 — finalize/expire overdue in-progress quiz attempts every minute
        # so a timed quiz closes even if the student never hits submit.
        cron(sweep_overdue_attempts_task, minute=set(range(0, 60))),
    ]


__all__ = ["WorkerSettings"]
