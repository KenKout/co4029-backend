"""Admin queries -- raw SQL aggregations for the operate-mode surface (T7.5).

Each submodule corresponds to a router slice:

* :mod:`job_metrics` -- the canonical job failure-rate + queue-state contract
  shared by the dashboard and the processing surface (ADM-004).
* :mod:`stats`      -- overview / active-users / content / health counts.
* :mod:`audit`      -- role-change + data-change + http-audit lookups.
* :mod:`processing` -- ARQ queue depth + processing_jobs triage.
* :mod:`users`      -- user listings + detail + status flips.

SQL lives in ``./sql/<slice>/<query>.sql`` and is loaded at module import via
``importlib.resources`` (T1.4a / T3.4 pattern). Wrapping the raw text in
``sqlalchemy.text(...)`` happens once at module load to avoid per-call parsing.

Org-scoping convention: every query that may be Manager-scoped accepts a
``CAST(:organization_id AS uuid)`` parameter. Pass ``None`` from the service layer
to short-circuit the scope filter (IT Admin global mode).
"""

from abridgeai.features.admin.queries import audit, job_metrics, processing, stats, users

__all__ = ["audit", "job_metrics", "processing", "stats", "users"]
