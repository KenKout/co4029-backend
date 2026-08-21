"""Config-resolution helpers for the interview generation pipeline.

Extracted from :mod:`.generation` to keep that orchestrator under its LOC
ratchet. ``resolve_variant_mode`` folds the run's ``variant_strategy`` form
value, the config's interviewer role, and the target question count into a
single plan; ``config_uuid`` is a defensive UUID extractor for
``generation_runs.config_json``.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from abridgeai.features.interviews.ai.stages.generation.resolve import (
    VARIANT_ANGLES,
    resolve_variant_strategy,
)
from abridgeai.features.interviews.models import InterviewConfig


def resolve_variant_mode(
    config: InterviewConfig,
    config_json: dict[str, Any],
    target_count: int,
) -> tuple[str | None, str | None, int]:
    """Resolve variant strategy, role type, and adjusted target count.

    Returns ``(variant_strategy, role_type, target_count)``. Legacy (no
    strategy) returns ``(None, None, target_count)`` unchanged. ``all_angles``
    multiplies the count by the angle count; ``role_only`` fixes every question
    to the config role's preferred type, degrading to legacy when the role has
    no preferred type (generic assistant).
    """
    from abridgeai.features.interviews.orchestrator.interviewer_identity import (  # noqa: PLC0415
        identity_from_config,
    )
    from abridgeai.features.interviews.orchestrator.role_question_filter import (  # noqa: PLC0415
        preferred_type,
    )

    variant_strategy = resolve_variant_strategy(config_json)
    role_type: str | None = None
    if variant_strategy == "all_angles":
        target_count = target_count * len(VARIANT_ANGLES)
    elif variant_strategy == "role_only":
        role_type = preferred_type(identity_from_config(config.persona_profile_json).role)
        if role_type is None:
            variant_strategy = None
    return variant_strategy, role_type, target_count


def config_uuid(config_json: dict[str, Any] | None, key: str) -> UUID | None:
    """Defensive UUID extractor for a ``generation_runs.config_json`` key."""
    if not config_json:
        return None
    raw = config_json.get(key)
    if raw is None:
        return None
    if isinstance(raw, UUID):
        return raw
    try:
        return UUID(str(raw))
    except (TypeError, ValueError):
        return None


__all__ = ["config_uuid", "resolve_variant_mode"]
