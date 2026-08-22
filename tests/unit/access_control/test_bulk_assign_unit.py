"""Unit tests for the bulk org-unit membership assign.

Pure service logic over patched queries, so no database is needed.

The tests that matter here are the tenancy ones. This endpoint takes a list
of membership ids from a request body and writes to them, which is exactly
the shape that leaks across tenants if the org check is wrong — and the
codebase already flags membership writes as its most consequential org
check. A cross-tenant id must not merely be filtered out; the whole call
must fail, and nothing may be written.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any

import pytest

from abridgeai.core.exceptions import AppError
from abridgeai.features.access_control.schemas.admin import BulkAssignUnitRequest
from abridgeai.features.access_control.services import organizations as svc

ORG_A = uuid.uuid4()
ORG_B = uuid.uuid4()
UNIT_A = uuid.uuid4()


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Patched query layer + a record of what would have been written."""
    ids = {name: uuid.uuid4() for name in ("m1", "m2", "foreign", "missing")}
    rows = {
        ids["m1"]: SimpleNamespace(id=ids["m1"], organization_id=ORG_A, deleted_at=None),
        ids["m2"]: SimpleNamespace(id=ids["m2"], organization_id=ORG_A, deleted_at=None),
        ids["foreign"]: SimpleNamespace(
            id=ids["foreign"], organization_id=ORG_B, deleted_at=None
        ),
    }
    units = {UNIT_A: SimpleNamespace(id=UNIT_A, organization_id=ORG_A, deleted_at=None)}
    written: dict[str, Any] = {}

    async def _get_org(_db: Any, org_id: uuid.UUID) -> Any:
        return SimpleNamespace(id=org_id)

    async def _get_unit(_db: Any, unit_id: uuid.UUID) -> Any:
        return units.get(unit_id)

    async def _list(_db: Any, wanted: list[uuid.UUID]) -> list[Any]:
        return [rows[i] for i in wanted if i in rows]

    async def _bulk(_db: Any, wanted: list[uuid.UUID], unit_id: uuid.UUID | None) -> int:
        written["ids"] = list(wanted)
        written["unit"] = unit_id
        return len(wanted)

    monkeypatch.setattr(svc, "get_organization", _get_org)
    monkeypatch.setattr(svc.org_queries, "get_unit", _get_unit)
    monkeypatch.setattr(svc.org_queries, "list_memberships_by_ids", _list)
    monkeypatch.setattr(svc.org_queries, "bulk_update_membership_unit", _bulk)
    return {"ids": ids, "units": units, "written": written}


async def test_assigns_and_reports_stale_ids(world: dict[str, Any]) -> None:
    """A selection that went stale mid-flow reports, it does not fail.

    Someone removed between opening the picker and confirming is an ordinary
    race, not an attack — failing the whole cohort over it would be hostile.
    """
    ids = world["ids"]
    result = await svc.assign_memberships_to_unit(
        None,
        ORG_A,
        BulkAssignUnitRequest(
            membership_ids=[ids["m1"], ids["m2"], ids["missing"]], org_unit_id=UNIT_A
        ),
    )
    assert result.assigned == 2
    assert result.skipped == [ids["missing"]]
    assert world["written"]["unit"] == UNIT_A


async def test_foreign_membership_rejects_whole_call_without_writing(
    world: dict[str, Any],
) -> None:
    """The security case: one cross-tenant id poisons the entire request.

    Silently dropping it would be worse than erroring — the caller would
    believe the cohort moved and never look again.
    """
    ids = world["ids"]
    with pytest.raises(AppError, match="different organization"):
        await svc.assign_memberships_to_unit(
            None,
            ORG_A,
            BulkAssignUnitRequest(
                membership_ids=[ids["m1"], ids["foreign"]], org_unit_id=UNIT_A
            ),
        )
    assert world["written"] == {}, "a rejected call must not write anything"


async def test_target_unit_from_another_org_is_rejected(world: dict[str, Any]) -> None:
    """Filing your people under another tenant's department would leak them
    into that tenant's scope filters."""
    world["units"][UNIT_A].organization_id = ORG_B
    with pytest.raises(AppError, match="different organization"):
        await svc.assign_memberships_to_unit(
            None,
            ORG_A,
            BulkAssignUnitRequest(
                membership_ids=[world["ids"]["m1"]], org_unit_id=UNIT_A
            ),
        )
    assert world["written"] == {}


async def test_missing_target_unit_is_rejected(world: dict[str, Any]) -> None:
    with pytest.raises(AppError, match="does not exist or is deleted"):
        await svc.assign_memberships_to_unit(
            None,
            ORG_A,
            BulkAssignUnitRequest(
                membership_ids=[world["ids"]["m1"]], org_unit_id=uuid.uuid4()
            ),
        )
    assert world["written"] == {}


async def test_detach_writes_null_and_skips_unit_validation(
    world: dict[str, Any],
) -> None:
    """``org_unit_id=None`` is the same "no unit" state a single PATCH
    produces, so bulk and single cannot drift."""
    result = await svc.assign_memberships_to_unit(
        None,
        ORG_A,
        BulkAssignUnitRequest(membership_ids=[world["ids"]["m1"]], org_unit_id=None),
    )
    assert result.assigned == 1
    assert world["written"]["unit"] is None


def test_schema_rejects_an_empty_selection() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        BulkAssignUnitRequest(membership_ids=[])


def test_schema_caps_the_id_list() -> None:
    """An unbounded IN list from a request body is a DoS shape."""
    with pytest.raises(ValueError, match="at most 500"):
        BulkAssignUnitRequest(membership_ids=[uuid.uuid4()] * 501)
