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
from abridgeai.features.admin.queries import setting_changes as change_queries
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
    # Cross-field invariant: max teachers must stay >= min teachers, so a stale
    # org override cannot be written into a self-contradicting state (a min
    # above the max makes every course unpublishable). Shared with
    # ``preview_change`` so the dry run enforces the same rules as the write.
    await _assert_cross_field_invariants(db, key, coerced, organization_id)
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


# ---------------------------------------------------------------------------
# Controlled change workflow (PRD ADM-030 -> ADM-034)
# ---------------------------------------------------------------------------
#
# Settings used to save on every keystroke-commit: the field's onBlur wrote
# straight through to the table. That made "I was reading the form" and "I
# changed the deployment" the same gesture, with no record of who, why, or what
# it had been. The flow below separates them:
#
#     preview_change()  -> validate + show the blast radius, writing nothing
#     apply_change()    -> write the value AND the audit row, in one transaction
#     rollback_change() -> re-apply a previous before-value as a new change
#
# The audit row and the setting write are deliberately not separable. A change
# that lands without its audit row is precisely the state this feature exists
# to prevent, so they share the caller's transaction and commit together.

CHANGE_SOURCE_ADMIN_CONSOLE = "admin_console"

#: Reason length bounds. A reason is required (ADM-033) but must not become a
#: place to paste an incident report the audit table then has to carry forever.
REASON_MIN_LENGTH = 3
REASON_MAX_LENGTH = 500


@dataclass(frozen=True)
class ChangeImpact:
    """What applying a pending change would do, computed without writing.

    ``affected_organizations`` is the number of live organizations that would
    actually feel a global change — those with no override of their own, since
    an org row wins over the global row. Reporting the total org count instead
    would overstate the blast radius for a key most tenants have customised.
    """

    key: str
    scope: str
    organization_id: UUID | None
    current_value: bool | int | float | None
    current_source: str
    new_value: bool | int | float | None
    #: True when the write is a no-op — same value, same scope.
    unchanged: bool
    #: Global changes reach every organization that has not overridden the key.
    affected_organizations: int
    total_organizations: int
    #: ADM-034: this setting only takes effect on the NEXT ingest. Already
    #: processed material is not reprocessed and is not retroactively changed.
    requires_reprocess: bool
    label: str
    description: str


def _validate_reason(reason: str) -> str:
    """A reason is mandatory and must say something (ADM-033)."""
    cleaned = (reason or "").strip()
    if len(cleaned) < REASON_MIN_LENGTH:
        raise SettingValidationError(
            f"A reason of at least {REASON_MIN_LENGTH} characters is required "
            "for every configuration change"
        )
    if len(cleaned) > REASON_MAX_LENGTH:
        raise SettingValidationError(
            f"Reason must be at most {REASON_MAX_LENGTH} characters"
        )
    return cleaned


def _scope_of(organization_id: UUID | None) -> str:
    return "organization" if organization_id is not None else "global"


def _stored_value_at_scope(
    row: ResolvedSetting, organization_id: UUID | None
) -> bool | int | float | None:
    """The override stored AT THIS SCOPE, or None when the value is inherited.

    Not the same as ``effective_value``: an org with no override of its own has
    an effective value but nothing stored, and recording the inherited number
    as the "before" would make a later rollback pin a value nobody ever chose.
    """
    return row.org_value if organization_id is not None else row.global_value


async def preview_change(
    db: AsyncSession,
    *,
    key: str,
    value: Any,  # noqa: ANN401 -- raw JSON from the client; validated here
    organization_id: UUID | None,
) -> ChangeImpact:
    """Validate a pending change and describe its effect. Writes nothing.

    Raises :class:`SettingValidationError` exactly as ``apply_change`` would,
    so the preview step is a genuine dry run rather than a second, weaker set
    of rules the real write might still reject.
    """
    coerced = coerce_and_validate(key, value)
    await _assert_cross_field_invariants(db, key, coerced, organization_id)

    current = await _one(db, key, organization_id)
    stored = _stored_value_at_scope(current, organization_id)
    spec = SETTINGS_REGISTRY[key]
    affected, total = await change_queries.affected_org_counts(db, setting_key=key)

    return ChangeImpact(
        key=key,
        scope=_scope_of(organization_id),
        organization_id=organization_id,
        current_value=current.effective_value,
        current_source=current.source,
        new_value=coerced,
        unchanged=stored == coerced,
        # An org-scoped change touches exactly one tenant; only a global one
        # spreads, and only to organizations that have not overridden the key.
        affected_organizations=1 if organization_id is not None else affected,
        total_organizations=total,
        requires_reprocess=spec.requires_reprocess,
        label=spec.label,
        description=spec.description,
    )


async def preview_clear(
    db: AsyncSession, *, key: str, organization_id: UUID | None
) -> ChangeImpact:
    """Describe what removing an override would do."""
    if key not in SETTINGS_REGISTRY:
        raise SettingValidationError(f"Unknown setting {key!r}")
    current = await _one(db, key, organization_id)
    stored = _stored_value_at_scope(current, organization_id)
    spec = SETTINGS_REGISTRY[key]
    affected, total = await change_queries.affected_org_counts(db, setting_key=key)
    return ChangeImpact(
        key=key,
        scope=_scope_of(organization_id),
        organization_id=organization_id,
        current_value=current.effective_value,
        current_source=current.source,
        # None means "inherit from the next level down" — the caller renders
        # that as the inherited value, not as an empty field.
        new_value=None,
        unchanged=stored is None,
        affected_organizations=1 if organization_id is not None else affected,
        total_organizations=total,
        requires_reprocess=spec.requires_reprocess,
        label=spec.label,
        description=spec.description,
    )


async def apply_change(
    db: AsyncSession,
    *,
    key: str,
    value: Any,  # noqa: ANN401 -- raw JSON from the client; validated here
    organization_id: UUID | None,
    actor_id: UUID | None,
    reason: str,
    source: str = CHANGE_SOURCE_ADMIN_CONSOLE,
) -> tuple[ResolvedSetting, dict[str, Any]]:
    """Apply one change and record it. Returns ``(new state, audit row)``.

    The audit row is written in the same transaction as the value. There is no
    ordering in which a change can land without its record.
    """
    cleaned_reason = _validate_reason(reason)
    before = _stored_value_at_scope(await _one(db, key, organization_id), organization_id)

    updated = await set_setting(
        db,
        key=key,
        value=value,
        organization_id=organization_id,
        actor_id=actor_id,
    )
    after = _stored_value_at_scope(updated, organization_id)

    audit = await change_queries.insert(
        db,
        setting_key=key,
        organization_id=organization_id,
        scope=_scope_of(organization_id),
        action="set",
        before_value=None if before is None else json.dumps(before),
        after_value=None if after is None else json.dumps(after),
        reason=cleaned_reason,
        actor_id=actor_id,
        source=source,
    )
    return updated, audit


async def apply_clear(
    db: AsyncSession,
    *,
    key: str,
    organization_id: UUID | None,
    actor_id: UUID | None,
    reason: str,
    source: str = CHANGE_SOURCE_ADMIN_CONSOLE,
) -> tuple[ResolvedSetting, dict[str, Any]]:
    """Remove an override and record it."""
    cleaned_reason = _validate_reason(reason)
    before = _stored_value_at_scope(await _one(db, key, organization_id), organization_id)

    updated = await clear_setting(db, key=key, organization_id=organization_id)

    audit = await change_queries.insert(
        db,
        setting_key=key,
        organization_id=organization_id,
        scope=_scope_of(organization_id),
        action="clear",
        before_value=None if before is None else json.dumps(before),
        after_value=None,
        reason=cleaned_reason,
        actor_id=actor_id,
        source=source,
    )
    return updated, audit


async def rollback_change(
    db: AsyncSession,
    *,
    change_id: UUID,
    actor_id: UUID | None,
    reason: str,
    organization_id: UUID | None = None,
) -> tuple[ResolvedSetting, dict[str, Any]]:
    """Restore the value a previous change replaced.

    Restores ``before_value_json`` — a value if the scope had an override, or
    inheritance if it did not. Both directions matter: rolling back a change
    that *created* an override must remove it again, not pin it to whatever it
    happened to inherit.

    The rollback is appended as its own change pointing at what it undid; the
    original row is never rewritten. ``organization_id``, when given, pins the
    tenant the caller is allowed to act on so an org-scoped admin cannot roll
    back a global change or another tenant's.
    """
    cleaned_reason = _validate_reason(reason)
    original = await change_queries.get_change(db, change_id=change_id)
    if original is None:
        raise NotFoundError(f"Setting change {change_id} not found")

    target_org: UUID | None = original["organization_id"]
    if organization_id is not None and target_org != organization_id:
        # Caller is scoped to one org; anything else is not theirs to undo.
        raise NotFoundError(f"Setting change {change_id} not found")

    key = original["setting_key"]
    if key not in SETTINGS_REGISTRY:
        # The key was removed from the registry after the change was recorded.
        # Re-applying it would write a row nothing reads.
        raise SettingValidationError(
            f"Setting {key!r} is no longer registered and cannot be rolled back"
        )

    before = _stored_value_at_scope(await _one(db, key, target_org), target_org)
    restore_to = original["before_value_json"]

    if restore_to is None:
        updated = await clear_setting(db, key=key, organization_id=target_org)
    else:
        updated = await set_setting(
            db,
            key=key,
            value=restore_to,
            organization_id=target_org,
            actor_id=actor_id,
        )
    after = _stored_value_at_scope(updated, target_org)

    audit = await change_queries.insert(
        db,
        setting_key=key,
        organization_id=target_org,
        scope=_scope_of(target_org),
        action="rollback",
        before_value=None if before is None else json.dumps(before),
        after_value=None if after is None else json.dumps(after),
        reason=cleaned_reason,
        actor_id=actor_id,
        source=CHANGE_SOURCE_ADMIN_CONSOLE,
        reverted_change_id=change_id,
    )
    return updated, audit


async def list_changes(
    db: AsyncSession,
    *,
    key: str | None = None,
    organization_id: UUID | None = None,
    global_only: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Change history, newest first.

    ``global_only`` and ``organization_id`` are separate because a NULL
    organization cannot express the difference between "global rows" and "no
    scope filter at all".
    """
    scope_filter = (
        "global"
        if global_only
        else ("organization" if organization_id is not None else None)
    )
    return await change_queries.list_changes(
        db,
        setting_key=key,
        organization_id=organization_id,
        scope_filter=scope_filter,
        limit=limit,
    )


async def _assert_cross_field_invariants(
    db: AsyncSession,
    key: str,
    coerced: bool | int | float,
    organization_id: UUID | None,
) -> None:
    """Teacher-count bounds must stay consistent with each other.

    Extracted from ``set_setting`` so ``preview_change`` enforces exactly the
    same rules; a preview that passes and an apply that then fails would make
    the preview worthless.
    """
    if key == "courses.min_teachers_per_course":
        max_val = await _resolved_int(
            db, "courses.max_teachers_per_course", organization_id
        )
        if coerced > max_val:
            raise SettingValidationError(
                "courses.min_teachers_per_course cannot exceed "
                f"courses.max_teachers_per_course ({max_val})"
            )
    elif key == "courses.max_teachers_per_course":
        min_val = await _resolved_int(
            db, "courses.min_teachers_per_course", organization_id
        )
        if coerced < min_val:
            raise SettingValidationError(
                "courses.max_teachers_per_course cannot be below "
                f"courses.min_teachers_per_course ({min_val})"
            )


__all__ = [
    "CHANGE_SOURCE_ADMIN_CONSOLE",
    "REASON_MAX_LENGTH",
    "REASON_MIN_LENGTH",
    "ChangeImpact",
    "ResolvedSetting",
    "apply_change",
    "apply_clear",
    "clear_setting",
    "list_changes",
    "list_settings",
    "preview_change",
    "preview_clear",
    "rollback_change",
    "set_setting",
]
