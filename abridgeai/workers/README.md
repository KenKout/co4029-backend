# ARQ Workers — abridgeai

## Convention

Every task signature MUST be:

```python
async def task(ctx: dict, actor_id: UUID, ...args) -> ...
```

- `actor_id` is the FIRST argument after `ctx`. Propagated to audit columns
  (`created_by`, `updated_by`) via `set_worker_actor` (T0.8) which sets the
  `current_actor_var` ContextVar that the SQLAlchemy `before_flush` listener
  reads on every persisted instance.
- Workers pull source files from S3 directly using the **internal endpoint**
  (`aws_endpoint_url`). They MUST NOT proxy bytes through the backend HTTP
  API — the bandwidth-saving architecture (browser → S3 direct via presigned
  URLs; worker → S3 direct via internal endpoint) was designed to keep
  large media off the application server's NIC.
- Bind structured-log context with `bind_request_context(...)` at task start
  and call `clear_request_context()` in a `finally` so neighbouring tasks in
  the worker pool never inherit identifiers.

## WorkerSettings composition

`abridgeai/workers/arq_app.py` aggregates the per-feature `JOBS` lists into
a single `WorkerSettings.functions`:

```python
from abridgeai.features.materials.workers import JOBS as MATERIAL_JOBS
# Phase 5/6/7: from abridgeai.features.{quizzes,interviews,notifications,sr}.workers import JOBS as ...

class WorkerSettings:
    functions = list(MATERIAL_JOBS)  # extend as features land
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = 10
    job_timeout = 600
    keep_result_seconds = 3600
    max_tries = 3
    cron_jobs: list[CronJob] = []
```

Each feature owns a `workers/` package that exports `JOBS: list[Callable]`.
`arq_app` imports those lists explicitly (no auto-discovery scan) so import
errors surface at startup rather than at first job dispatch.

## Run locally

```bash
cd backend-new
uv run arq abridgeai.workers.arq_app.WorkerSettings
```

## Cron jobs

Registered in `WorkerSettings.cron_jobs`. Each cron task is owned by its
feature; `arq_app` re-exposes them so a single worker process drives the
schedule. Pending registrations:

- `cleanup_orphaned_uploads_task` — Phase 4.5, daily 03:00 UTC.
- `scan_due_cards_task` — Phase 7.5 (SR engine), every hour at minute 0.
- `daily_compliance_summary_task` — Phase 7 admin, daily 09:00 UTC.

Add a new cron by importing the task in `arq_app.py`, then appending
`cron(my_task, hour={3}, minute=0)` to `cron_jobs`.

## Enqueueing from API code

API routers obtain an `ArqRedis` pool via FastAPI dependency injection and
enqueue by name:

```python
await arq_pool.enqueue_job(
    "ingest_material_version_task",
    actor_id,
    material_version_id,
    pipeline_run_id,
)
```

The first positional argument after the task name is `actor_id` — never
omit it. The audit listener relies on it.
