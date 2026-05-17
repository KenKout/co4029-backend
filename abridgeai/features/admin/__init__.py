"""Admin feature -- system stats, audit dashboards, processing visibility, user management (T7.5).

NO models here -- this feature is purely a read/operate-over-existing-tables surface.

Modules:
* :mod:`queries`  -- raw SQL aggregations across multiple feature tables.
* :mod:`services` -- thin orchestration over the queries plus user-disable side-effects
  (session revocation).
* :mod:`routers`  -- four FastAPI routers (stats / audit / processing / users) all mounted
  under ``/admin`` by the integration layer (T7.6).

Permission perimeter:
* IT Admin (``system.administer``) -- bypasses org scope, sees global aggregates.
* Manager / HOD (``system.stats.read``) -- org-bound; queries filter by the actor's
  ``organization_memberships``.
* Other roles -- 403.

T0.23 (HTTP audit log middleware) is deferred. The ``audit/http`` endpoint is registered
unconditionally and returns 503 at request time when the ``http_audit_log`` table is
absent. This avoids hard-coupling T7.5 to T0.23.
"""
