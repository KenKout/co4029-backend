"""Policy routers: a public read surface and an admin authoring surface."""

from abridgeai.features.policies.routers.admin import router as admin_router
from abridgeai.features.policies.routers.public import router as public_router

__all__ = ["admin_router", "public_router"]
