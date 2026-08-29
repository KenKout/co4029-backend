"""The runtime-config change contract: reasons, scope, and what "before" means.

These pin the parts of the controlled-change flow that carry the safety
properties (PRD ADM-030 -> ADM-034) and that a DB-backed test would obscure
rather than clarify:

* a change without a real reason is refused, so the audit trail cannot fill up
  with blank rows;
* "before" is the value stored AT THAT SCOPE, not the effective value — the
  distinction rollback depends on;
* global and organization scope are never inferred from each other.

Endpoint-level behaviour (transactional apply, org isolation on rollback) needs
Postgres and lives in the integration suite.
"""

import pytest

from abridgeai.core.settings_registry import SettingValidationError
from abridgeai.features.admin.services.settings import (
    REASON_MAX_LENGTH,
    REASON_MIN_LENGTH,
    ResolvedSetting,
    _scope_of,
    _stored_value_at_scope,
    _validate_reason,
)
from uuid import uuid4

ORG_ID = uuid4()


def _row(
    *, global_value: float | int | bool | None, org_value: float | int | bool | None
) -> ResolvedSetting:
    """A resolved setting with the two override slots under test."""
    effective = (
        org_value
        if org_value is not None
        else (global_value if global_value is not None else 512)
    )
    return ResolvedSetting(
        key="chunking.max_tokens",
        group="chunking",
        type="int",
        label="Max tokens",
        description="",
        env_var=None,
        minimum=None,
        maximum=None,
        requires_reprocess=True,
        default_value=512,
        env_value=None,
        global_value=global_value,
        org_value=org_value,
        effective_value=effective,
        source="organization" if org_value is not None else "global",
    )


class TestReasonIsMandatory:
    def test_empty_reason_is_refused(self) -> None:
        with pytest.raises(SettingValidationError):
            _validate_reason("")

    def test_whitespace_only_reason_is_refused(self) -> None:
        # Otherwise a required field is satisfied by pressing space.
        with pytest.raises(SettingValidationError):
            _validate_reason("   \n\t ")

    def test_too_short_reason_is_refused(self) -> None:
        with pytest.raises(SettingValidationError):
            _validate_reason("x" * (REASON_MIN_LENGTH - 1))

    def test_reason_is_trimmed(self) -> None:
        assert _validate_reason("  raising chunk size  ") == "raising chunk size"

    def test_over_long_reason_is_refused(self) -> None:
        # The audit table is not the place to paste an incident report.
        with pytest.raises(SettingValidationError):
            _validate_reason("x" * (REASON_MAX_LENGTH + 1))

    def test_reason_at_the_bounds_is_accepted(self) -> None:
        assert _validate_reason("x" * REASON_MIN_LENGTH)
        assert _validate_reason("x" * REASON_MAX_LENGTH)


class TestScope:
    def test_no_organization_is_global(self) -> None:
        assert _scope_of(None) == "global"

    def test_an_organization_is_org_scoped(self) -> None:
        assert _scope_of(ORG_ID) == "organization"


class TestStoredValueSemantics:
    """`before` must be the override at that scope, never the inherited value.

    This is the property rollback rests on. If an org that merely *inherits*
    the global value recorded that number as its "before", rolling back would
    pin the org to a value nobody ever chose for it — and it would then stop
    tracking future global changes, silently.
    """

    def test_inherited_org_value_has_nothing_stored(self) -> None:
        row = _row(global_value=800, org_value=None)
        assert row.effective_value == 800
        assert _stored_value_at_scope(row, ORG_ID) is None

    def test_org_override_is_read_at_org_scope(self) -> None:
        row = _row(global_value=800, org_value=1200)
        assert _stored_value_at_scope(row, ORG_ID) == 1200

    def test_global_scope_reads_the_global_slot_not_the_org_one(self) -> None:
        # An org override must not leak into the global "before".
        row = _row(global_value=800, org_value=1200)
        assert _stored_value_at_scope(row, None) == 800

    def test_unset_global_has_nothing_stored(self) -> None:
        row = _row(global_value=None, org_value=None)
        assert _stored_value_at_scope(row, None) is None

    def test_a_zero_override_is_stored_not_absent(self) -> None:
        # `0` is falsy; treating it as "no override" would make clearing and
        # setting-to-zero indistinguishable in the audit trail.
        row = _row(global_value=0, org_value=None)
        assert _stored_value_at_scope(row, None) == 0
        assert _stored_value_at_scope(row, None) is not None

    def test_a_false_override_is_stored_not_absent(self) -> None:
        row = _row(global_value=False, org_value=None)
        assert _stored_value_at_scope(row, None) is False
        assert _stored_value_at_scope(row, None) is not None
