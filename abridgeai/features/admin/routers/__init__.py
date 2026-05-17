"""Admin feature routers (T7.5).

Four sibling routers all mounted under ``/admin`` by the integration layer
(T7.6):

* :data:`stats_router`      -- ``/admin/stats/*``
* :data:`audit_router`      -- ``/admin/audit/*``
* :data:`processing_router` -- ``/admin/processing/*``
* :data:`users_router`      -- ``/admin/users*``
"""

from __future__ import annotations

from .audit import router as audit_router
from .processing import router as processing_router
from .stats import router as stats_router
from .users import router as users_router

__all__ = ["audit_router", "processing_router", "stats_router", "users_router"]
