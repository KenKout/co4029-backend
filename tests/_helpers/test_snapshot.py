from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

from tests._helpers import snapshot as snap_mod
from tests._helpers.snapshot import (
    _EVIDENCE_DIR,
    _SNAPSHOTS_DIR,
    assert_snapshot_match,
    capture_snapshot,
)

_SENTINEL_DB: Any = object()


def _row(idx: int, *, drift: bool = False) -> dict[str, Any]:
    return {
        "id": UUID(f"00000000-0000-0000-0000-00000000000{idx}"),
        "title": f"Row {idx}" if not drift else f"DRIFTED {idx}",
        "amount": Decimal("10.50") + Decimal(idx),
        "created_at": datetime(2025, 1, 1, 12, 0, idx, 123456, tzinfo=UTC),
    }


async def _query_three_rows(db: Any) -> Sequence[Any]:
    return [_row(1), _row(2), _row(3)]


async def _query_three_rows_with_drift(db: Any) -> Sequence[Any]:
    return [_row(1), _row(2, drift=True), _row(3)]


@pytest.fixture
def cleanup_test_capture() -> Path:
    path = _SNAPSHOTS_DIR / "test_capture.json"
    if path.exists():
        path.unlink()
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def staged_match_unchanged() -> Path:
    path = _SNAPSHOTS_DIR / "test_match_unchanged.json"
    serialized = snap_mod._serialize_result([_row(1), _row(2), _row(3)])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialized, indent=2, sort_keys=True) + "\n")
    yield path
    if path.exists():
        path.unlink()


@pytest.fixture
def staged_match_drift() -> tuple[Path, Path]:
    snap_path = _SNAPSHOTS_DIR / "test_match_drift.json"
    evidence_path = _EVIDENCE_DIR / "snapshot-mismatch-test_match_drift.json"
    serialized = snap_mod._serialize_result([_row(1), _row(2), _row(3)])
    snap_path.parent.mkdir(parents=True, exist_ok=True)
    snap_path.write_text(json.dumps(serialized, indent=2, sort_keys=True) + "\n")
    if evidence_path.exists():
        evidence_path.unlink()
    yield snap_path, evidence_path
    if snap_path.exists():
        snap_path.unlink()
    if evidence_path.exists():
        evidence_path.unlink()


async def test_capture(cleanup_test_capture: Path) -> None:
    path = cleanup_test_capture
    assert not path.exists()

    await capture_snapshot(
        _SENTINEL_DB,
        _query_three_rows,
        name="test_capture",
        params=None,
    )

    assert path.exists(), f"snapshot file not created at {path}"
    payload = json.loads(path.read_text())
    assert isinstance(payload, list)
    assert len(payload) == 3
    assert payload[0]["title"] == "Row 1"
    assert payload[0]["id"] == "00000000-0000-0000-0000-000000000001"
    assert payload[0]["amount"] == "11.50"
    assert payload[0]["created_at"] == "2025-01-01T12:00:01+00:00"

    await capture_snapshot(
        _SENTINEL_DB,
        _query_three_rows,
        name="test_capture",
    )
    assert json.loads(path.read_text()) == payload


async def test_match_unchanged(staged_match_unchanged: Path) -> None:
    await assert_snapshot_match(
        _SENTINEL_DB,
        _query_three_rows,
        name="test_match_unchanged",
    )


async def test_match_drift(staged_match_drift: tuple[Path, Path]) -> None:
    _, evidence_path = staged_match_drift

    with pytest.raises(AssertionError) as exc:
        await assert_snapshot_match(
            _SENTINEL_DB,
            _query_three_rows_with_drift,
            name="test_match_drift",
        )

    msg = str(exc.value)
    assert "test_match_drift" in msg
    assert "drift" in msg.lower()
    assert "row[1]" in msg
    assert "title" in msg

    assert evidence_path.exists(), f"evidence file not created at {evidence_path}"
    evidence = json.loads(evidence_path.read_text())
    assert evidence["name"] == "test_match_drift"
    assert "expected" in evidence
    assert "actual" in evidence
    assert evidence["expected"][1]["title"] == "Row 2"
    assert evidence["actual"][1]["title"] == "DRIFTED 2"
