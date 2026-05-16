"""Identity feature routers.

Each module exports a single ``APIRouter``; mounting happens in Phase 1
integration (``abridgeai/api.py`` / ``abridgeai/main.py`` will call
``app.include_router(auth_router, prefix="/api/v1")`` once those files
exist). Until then this package only exposes the routers for direct
import by tests.
"""

from __future__ import annotations

from .auth import router as auth_router

__all__ = ["auth_router"]
