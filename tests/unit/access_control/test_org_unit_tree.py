"""Unit tests for the org-unit tree builder.

Pure assembly over rows the query layer returned, so these run without a
database: ``list_units_for_organization`` is patched and the service is
handed plain row objects.

The regression that motivated this file is
:func:`test_does_not_touch_the_children_relationship`. ``OrgUnitNode`` has
a ``children`` field, ``OrgUnit`` has a ``children`` *relationship* of the
same name, and the schema base sets ``from_attributes=True`` — so building
a node with ``model_validate(row)`` silently read the relationship and
tried to lazy-load it on an async session, turning every request into a
``MissingGreenlet`` 500.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from abridgeai.features.access_control.services import organizations as svc

ORG_ID = uuid.uuid4()
_NOW = datetime.now(tz=UTC)


def _row(
    unit_id: uuid.UUID,
    name: str,
    parent_unit_id: uuid.UUID | None = None,
    unit_type: str = "department",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=unit_id,
        organization_id=ORG_ID,
        parent_unit_id=parent_unit_id,
        unit_type=unit_type,
        name=name,
        code=None,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _patch_rows(
    monkeypatch: pytest.MonkeyPatch, rows: list[SimpleNamespace]
) -> None:
    async def _fake(_db: Any, _org_id: uuid.UUID, **_kw: Any) -> list[SimpleNamespace]:
        return rows

    monkeypatch.setattr(svc.org_queries, "list_units_for_organization", _fake)


@pytest.fixture
def faculty_rows() -> dict[str, Any]:
    """Engineering → {Computer Science → AI Program, Electrical}."""
    ids = {key: uuid.uuid4() for key in ("fac", "cse", "ee", "ai")}
    return {
        "ids": ids,
        "rows": [
            _row(ids["fac"], "Engineering", None, "faculty"),
            _row(ids["cse"], "Computer Science", ids["fac"]),
            _row(ids["ee"], "Electrical", ids["fac"]),
            _row(ids["ai"], "AI Program", ids["cse"], "program"),
        ],
    }


async def test_nests_by_parent_unit_id(
    monkeypatch: pytest.MonkeyPatch, faculty_rows: dict[str, Any]
) -> None:
    _patch_rows(monkeypatch, faculty_rows["rows"])
    roots = await svc.list_unit_tree(None, ORG_ID)

    assert [r.name for r in roots] == ["Engineering"]
    faculty = roots[0]
    assert [c.name for c in faculty.children] == ["Computer Science", "Electrical"]
    assert [g.name for g in faculty.children[0].children] == ["AI Program"]


async def test_descendant_count_is_the_whole_subtree(
    monkeypatch: pytest.MonkeyPatch, faculty_rows: dict[str, Any]
) -> None:
    """Not just direct children — the delete cascade takes the whole branch."""
    _patch_rows(monkeypatch, faculty_rows["rows"])
    faculty = (await svc.list_unit_tree(None, ORG_ID))[0]

    assert faculty.descendant_count == 3  # CSE + AI + Electrical
    assert faculty.children[0].descendant_count == 1  # CSE -> AI
    assert faculty.children[1].descendant_count == 0


async def test_siblings_sort_by_name_case_insensitively(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = uuid.uuid4()
    _patch_rows(
        monkeypatch,
        [
            _row(root, "Root", None, "faculty"),
            _row(uuid.uuid4(), "zebra", root),
            _row(uuid.uuid4(), "Alpha", root),
        ],
    )
    faculty = (await svc.list_unit_tree(None, ORG_ID))[0]
    assert [c.name for c in faculty.children] == ["Alpha", "zebra"]


async def test_orphan_is_surfaced_as_a_root_not_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A unit whose parent was deleted must stay visible.

    Dropping it would hide real courses and real people from the manager's
    tree; showing it at top level is visibly odd, which is the point.
    """
    orphan_id = uuid.uuid4()
    _patch_rows(
        monkeypatch,
        [_row(orphan_id, "Detached Dept", uuid.uuid4())],
    )
    roots = await svc.list_unit_tree(None, ORG_ID)
    assert [r.id for r in roots] == [orphan_id]


async def test_empty_organization_yields_no_roots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_rows(monkeypatch, [])
    assert await svc.list_unit_tree(None, ORG_ID) == []


async def test_does_not_touch_the_children_relationship(
    monkeypatch: pytest.MonkeyPatch, faculty_rows: dict[str, Any]
) -> None:
    """Regression: reading ORM ``children`` lazy-loads and 500s.

    The rows here raise if anything reads ``children`` or ``parent``, which
    is exactly what ``OrgUnitNode.model_validate(row)`` used to do via
    ``from_attributes=True``. The builder must compose nodes from scalar
    columns and derive nesting from ``parent_unit_id`` alone.
    """

    class ExplodingRow(SimpleNamespace):
        @property
        def children(self) -> None:
            raise AssertionError(
                "read ORM .children — this lazy-loads on an async session"
            )

        @property
        def parent(self) -> None:
            raise AssertionError("read ORM .parent — this lazy-loads too")

    rows = [ExplodingRow(**vars(r)) for r in faculty_rows["rows"]]
    _patch_rows(monkeypatch, rows)

    roots = await svc.list_unit_tree(None, ORG_ID)
    assert roots[0].descendant_count == 3
