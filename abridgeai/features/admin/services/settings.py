"""Admin surface over runtime settings.

Returns each registered setting alongside *where its effective value comes
from*, which is the part that makes the page usable. A form showing only the
current number cannot answer "is this 800 because someone set it, or because
nobody has?" — and that is exactly the question an operator has when ingestion
behaves unexpectedly. Every row carries the org value, the global value, the
environment value and the code default, plus which of them won.

Writes are validated against :mod:`abridgeai.core.settings_registry` before
they reach the table: the column is JSONB and would otherwise accept anything,
with the failure surfacing much later inside an ingest worker.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from abridgeai.core.exceptions import NotFoundError
from abridgeai.core.runtime_settings import invalidate_settings_cache
from abridgeai.core.settings_registry import (
    SETTINGS_REGISTRY,
    SettingSpec,
    SettingValidationError,
    coerce_and_validate,
)
from abridgeai.features.admin.queries import settings as settings_queries

if TYPE_CHECKING:
    from abridgeai.core.db import AsyncSession  # type: ignore[attr-defined]


@dataclass(frozen=True)
class ResolvedSetting:
    key: str
    group: str
    type: str
    label: str
    description: str
    env_var: str | None
    minimum: float | None
    maximum: float | None
    requires_reprocess: bool
    default_value: bool | int | float
    env_value: bool | int | float | None
    global_value: bool | int | float | None
    org_value: bool | int | float | None
    effective_value: bool | int | float
    # "organization" | "global" | "environment" | "default"
    source: str


def _parse_env(spec: SettingSpec) -> bool | int | float | None:
    if spec.env_var is None:
        return None
    raw = os.environ.get(spec.env_var)
    if raw is None or not raw.strip():
        return None
    value = raw.strip()
    try:
        if spec.type == "bool":
            lowered = value.lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
            return None
        return int(value) if spec.type == "int" else float(value)
    except ValueError:
        return None


def _valid_or_none(key: str, raw: Any) -> bool | int | float | None:  # noqa: ANN401
    """Coerce a stored value, discarding one that no longer fits its spec."""
    if raw is None:
        return None
    try:
        return coerce_and_validate(key, raw)
    except SettingValidationError:
        return None


async def list_settings(
    db: AsyncSession, organization_id: UUID | None = None
) -> list[ResolvedSetting]:
    """Every registered setting with its provenance, in registry order."""
    rows = await settings_queries.load_rows(db, organization_id)
    globals_: dict[str, Any] = {}
    org_values: dict[str, Any] = {}
    for row in rows:
        target = globals_ if row["organization_id"] is None else org_values
        target[row["setting_key"]] = row["setting_value_json"]

    out: list[ResolvedSetting] = []
    for key, spec in SETTINGS_REGISTRY.items():
        org_value = _valid_or_none(key, org_values.get(key))
        global_value = _valid_or_none(key, globals_.get(key))
        env_value = _parse_env(spec)

        # Same precedence the resolver applies at ingest time. Kept in step by
        # ordering alone rather than shared code because the two need different
        # shapes — one returns a value, the other has to name the winner.
        if org_value is not None:
            effective, source = org_value, "organization"
        elif global_value is not None:
            effective, source = global_value, "global"
        elif env_value is not None:
            effective, source = env_value, "environment"
        else:
            effective, source = spec.default, "default"

        out.append(
            ResolvedSetting(
                key=key,
                group=spec.group,
                type=spec.type,
                label=spec.label,
                description=spec.description,
                env_var=spec.env_var,
                minimum=spec.minimum,
                maximum=spec.maximum,
                requires_reprocess=spec.requires_reprocess,
                default_value=spec.default,
                env_value=env_value,
                global_value=global_value,
                org_value=org_value,
                effective_value=effective,
                source=source,
            )
        )
    return out


async def set_setting(
    db: AsyncSession,
    *,
    key: str,
    value: Any,  # noqa: ANN401 -- arrives as raw JSON; validated below
    organization_id: UUID | None,
    actor_id: UUID | None,
) -> ResolvedSetting:
    """Write one setting and return its new resolved state.

    Raises :class:`SettingValidationError` for an unknown key or an
    out-of-bounds value. The unknown-key case matters as much as the bounds:
    the registry is a closed allowlist, so a typo cannot quietly create a row
    that nothing ever reads.
    """
    coerced = coerce_and_validate(key, value)
    # Cross-field invariant: max teachers must stay >= min teachers. Validated
    # here so a stale org override can't be written into a self-contradicting
    # state (a min above the max would make every course unpublishable).
    if key == "courses.min_teachers_per_course":
        max_val = await _resolved_int(db, "courses.max_teachers_per_course", organization_id)
        if coerced > max_val:
            raise SettingValidationError(
                "courses.min_teachers_per_course cannot exceed "
                f"courses.max_teachers_per_course ({max_val})"
            )
    elif key == "courses.max_teachers_per_course":
        min_val = await _resolved_int(db, "courses.min_teachers_per_course", organization_id)
        if coerced < min_val:
            raise SettingValidationError(
                "courses.max_teachers_per_course cannot be below "
                f"courses.min_teachers_per_course ({min_val})"
            )
    await settings_queries.upsert(
        db,
        setting_key=key,
        value_json=json.dumps(coerced),
        organization_id=organization_id,
        updated_by=actor_id,
    )
    invalidate_settings_cache()
    return await _one(db, key, organization_id)


async def clear_setting(
    db: AsyncSession,
    *,
    key: str,
    organization_id: UUID | None,
) -> ResolvedSetting:
    """Remove an override so the next level down takes effect.

    Clearing an organization row falls back to the global row; clearing the
    global row falls back to the environment, then the code default. Deleting
    something that was never set is not an error — the end state is what the
    caller asked for either way.
    """
    if key not in SETTINGS_REGISTRY:
        raise SettingValidationError(f"Unknown setting {key!r}")
    await settings_queries.delete(db, setting_key=key, organization_id=organization_id)
    invalidate_settings_cache()
    return await _one(db, key, organization_id)


async def _one(
    db: AsyncSession, key: str, organization_id: UUID | None
) -> ResolvedSetting:
    for row in await list_settings(db, organization_id):
        if row.key == key:
            return row
    raise NotFoundError(f"Setting {key} not found")


async def _resolved_int(
    db: AsyncSession, key: str, organization_id: UUID | None
) -> int:
    """Effective integer value of a registered int setting."""
    row = await _one(db, key, organization_id)
    return int(row.effective_value)


__all__ = ["ResolvedSetting", "clear_setting", "list_settings", "set_setting"]
