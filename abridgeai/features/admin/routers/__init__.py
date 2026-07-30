"""Admin feature routers (T7.5).

Five sibling routers all mounted under ``/admin`` by the integration layer
(T7.6):

* :data:`stats_router`      -- ``/admin/stats/*``
* :data:`audit_router`      -- ``/admin/audit/*``
* :data:`processing_router` -- ``/admin/processing/*``
* :data:`users_router`      -- ``/admin/users*``
* :data:`ai_costs_router`   -- ``/admin/ai/costs/*`` (T0.27)
* :data:`ai_pricing_router` -- ``/admin/ai/pricing/*`` (model cost config CRUD)
* :data:`settings_router`   -- ``/admin/settings/*`` + per-org overrides
"""

from __future__ import annotations

from .ai_costs import router as ai_costs_router
from .ai_model_pricing import router as ai_pricing_router
from .audit import router as audit_router
from .processing import router as processing_router
from .settings import router as settings_router
from .stats import router as stats_router
from .users import router as users_router

__all__ = [
    "ai_costs_router",
    "ai_pricing_router",
    "audit_router",
    "processing_router",
    "settings_router",
    "stats_router",
    "users_router",
]
