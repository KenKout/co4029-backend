"""Admin services -- thin orchestration over queries plus user-disable side-effects (T7.5)."""

from abridgeai.features.admin.services import audit, processing, stats, users

__all__ = ["audit", "processing", "stats", "users"]
