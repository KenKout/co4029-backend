"""Characterization snapshot helpers for behavior preservation.

The workflow:
    1. Pre-migration: call ``capture_snapshot`` once to save the current
       query result to ``tests/fixtures/snapshots/{name}.json``. The
       fixture is then committed to git as the behavior baseline.
    2. Post-migration: ``assert_snapshot_match`` re-runs the (possibly
       refactored) query callable and compares row-by-row against the
       baseline. Any drift dumps a diff to
       ``.sisyphus/evidence/orm-consolidation/snapshot-mismatch-{name}.json``
       and raises ``AssertionError``.

Determinism is the caller's responsibility: the query callable must
return rows in a stable order (e.g. ``ORDER BY id``). The helpers do not
sort -- order-sensitive comparison is intentional so that ordering
regressions are caught.

Serialization rules (locked):
    * SQLAlchemy ``Row`` -> dict via ``row._mapping``
    * SQLAlchemy ORM instances -> dict of mapped column attributes
    * ``UUID`` -> ``str(uuid)``
    * ``datetime`` -> ``isoformat()`` with microseconds dropped (rounded
      to the second). Sub-second precision is unstable across DB
      round-trips and noisy in diffs; we trade it away on purpose.
    * ``Decimal`` -> ``str(decimal)`` (preserves precision; ``float``
      would lose it)
    * ``bytes`` -> base64-encoded ASCII string
    * ``None`` -> ``null``
    * ``set`` / ``frozenset`` -> sorted list (JSON has no set type, and
      iteration order would otherwise be nondeterministic)
"""

from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

from sqlalchemy import Row
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.exc import NoInspectionAvailable
from sqlalchemy.ext.asyncio import AsyncSession

_BACKEND_NEW_ROOT = Path(__file__).resolve().parents[2]
_SNAPSHOTS_DIR = _BACKEND_NEW_ROOT / "tests" / "fixtures" / "snapshots"
_EVIDENCE_DIR = _BACKEND_NEW_ROOT.parent / ".sisyphus" / "evidence" / "orm-consolidation"


QueryCallable = Callable[..., Awaitable[Sequence[Any]]]


class _SnapshotEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, Row):
            return dict(o._mapping)
        if isinstance(o, UUID):
            return str(o)
        if isinstance(o, datetime):
            return o.replace(microsecond=0).isoformat()
        if isinstance(o, Decimal):
            return str(o)
        if isinstance(o, (bytes, bytearray)):
            return base64.b64encode(bytes(o)).decode("ascii")
        if isinstance(o, (set, frozenset)):
            return sorted(list(o), key=lambda x: (type(x).__name__, str(x)))
        try:
            mapper = sa_inspect(o).mapper
        except (NoInspectionAvailable, AttributeError):
            return super().default(o)
        return {col.key: getattr(o, col.key) for col in mapper.columns}


def _serialize_row(row: Any) -> dict[str, Any]:
    if isinstance(row, dict):
        return json.loads(json.dumps(row, cls=_SnapshotEncoder))
    if isinstance(row, Row):
        return json.loads(json.dumps(dict(row._mapping), cls=_SnapshotEncoder))
    try:
        mapper = sa_inspect(row).mapper
    except (NoInspectionAvailable, AttributeError):
        return json.loads(json.dumps({"value": row}, cls=_SnapshotEncoder))
    payload = {col.key: getattr(row, col.key) for col in mapper.columns}
    return json.loads(json.dumps(payload, cls=_SnapshotEncoder))


def _serialize_result(result: Sequence[Any]) -> list[dict[str, Any]]:
    return [_serialize_row(r) for r in result]


def _snapshot_path(name: str) -> Path:
    if not name or "/" in name or "\\" in name:
        raise ValueError(f"invalid snapshot name: {name!r}")
    return _SNAPSHOTS_DIR / f"{name}.json"


def _evidence_path(name: str) -> Path:
    return _EVIDENCE_DIR / f"snapshot-mismatch-{name}.json"


async def capture_snapshot(
    db: AsyncSession,
    query_callable: QueryCallable,
    *,
    name: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Run ``query_callable`` and persist its serialized result.

    Run-once semantics: if the snapshot file already exists, this is a
    no-op. To rebuild a snapshot, delete the file on disk and re-run.
    """
    path = _snapshot_path(name)
    if path.exists():
        return
    rows = await query_callable(db, **(params or {}))
    serialized = _serialize_result(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialized, indent=2, sort_keys=True) + "\n")


def _row_diff(expected: list[dict[str, Any]], actual: list[dict[str, Any]]) -> str:
    n_old, n_new = len(expected), len(actual)
    if n_old != n_new:
        first = min(n_old, n_new)
        return f"rows={n_old}->{n_new}, first mismatch at row[{first}] (length differs)"
    for i, (e_row, a_row) in enumerate(zip(expected, actual, strict=True)):
        if e_row == a_row:
            continue
        keys = sorted(set(e_row) | set(a_row))
        for k in keys:
            ev, av = e_row.get(k, "<MISSING>"), a_row.get(k, "<MISSING>")
            if ev != av:
                return (
                    f"rows={n_old}->{n_new}, "
                    f"first mismatch at row[{i}] field '{k}': "
                    f"expected={ev!r} actual={av!r}"
                )
    return f"rows={n_old}->{n_new}, no field-level diff (deep equality false)"


async def assert_snapshot_match(
    db: AsyncSession,
    query_callable: QueryCallable,
    *,
    name: str,
    params: dict[str, Any] | None = None,
) -> None:
    """Compare a fresh query result against the saved baseline.

    On mismatch, dumps the full expected/actual payloads to
    ``.sisyphus/evidence/orm-consolidation/snapshot-mismatch-{name}.json``
    and raises ``AssertionError`` with a row-by-row summary.
    """
    path = _snapshot_path(name)
    if not path.exists():
        raise AssertionError(
            f"Snapshot '{name}' not found at {path}. "
            f"Run capture_snapshot(..., name='{name}') first."
        )
    expected = json.loads(path.read_text())
    rows = await query_callable(db, **(params or {}))
    actual = _serialize_result(rows)

    if expected == actual:
        return

    diff = _row_diff(expected, actual)
    _EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    evidence = _evidence_path(name)
    evidence.write_text(
        json.dumps(
            {"name": name, "diff": diff, "expected": expected, "actual": actual},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    raise AssertionError(f"Snapshot '{name}' drift: {diff}. Full diff: {evidence}")
