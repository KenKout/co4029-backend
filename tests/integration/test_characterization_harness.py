from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from characterization.contract import ParityContract, load_contract, slugify
from characterization.harness import (
    Snapshot,
    capture_snapshot,
    snapshot_path,
)
from characterization.replay import compare, replay


def _mock_factory(handler):
    async def factory(base_url: str) -> httpx.AsyncClient:
        transport = httpx.MockTransport(handler)
        return httpx.AsyncClient(base_url=base_url, transport=transport, timeout=5.0)

    return factory


def _healthz_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"status": "ok"},
        headers={"content-type": "application/json"},
    )


def test_slugify_normalizes_endpoint():
    assert slugify("/healthz") == "healthz"
    assert slugify("/api/v1/users") == "api_v1_users"
    assert slugify("/") == "_root"
    assert slugify("/healthz/") == "healthz"
    assert slugify("/healthz?x=1") == "healthz"


def test_default_contract_loads():
    contract = load_contract("/some/unknown/endpoint")
    assert isinstance(contract, ParityContract)
    assert "$.id" in contract.redact_fields
    assert "$.created_at" in contract.redact_fields
    assert "$.created_by" in contract.allow_extra_fields


def test_healthz_contract_loads():
    contract = load_contract("/healthz")
    assert contract.endpoint == "/healthz"
    assert contract.behavior.status_code == 200
    assert "$.status" in contract.strict_match_fields


@pytest.mark.asyncio
async def test_snapshot_and_replay_healthz_self(tmp_path: Path):
    factory = _mock_factory(_healthz_handler)

    captured = await capture_snapshot(
        "/healthz",
        backend="new",
        client_factory=factory,
        snapshots_dir=tmp_path,
    )
    assert captured.status_code == 200
    assert captured.body_json == {"status": "ok"}

    persisted = snapshot_path("/healthz", backend="new", snapshots_dir=tmp_path)
    assert persisted.is_file()
    on_disk = json.loads(persisted.read_text())
    assert on_disk["body_json"] == {"status": "ok"}

    result = await replay(
        "/healthz",
        backend="new",
        client_factory=factory,
        captured_backend="new",
        snapshots_dir=tmp_path,
    )
    assert result.passed, f"self-consistency replay failed: {result.divergences}"


@pytest.mark.asyncio
async def test_inject_extra_field_passes_with_allow(tmp_path: Path):
    parity_dir = tmp_path / "parity"
    parity_dir.mkdir()
    (parity_dir / "_default.yaml").write_text(
        "redact_fields: []\n"
        "allow_extra_fields:\n"
        '  - "$.debug_token"\n'
        "strict_match_fields: []\n"
        "behavior:\n"
        "  status_code: 200\n"
    )

    captured = Snapshot(
        backend="old",
        endpoint="/widgets",
        method="GET",
        status_code=200,
        headers={"content-type": "application/json"},
        body_json={"name": "widget", "size": 42},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"name": "widget", "size": 42, "debug_token": "abc-123"},
            headers={"content-type": "application/json"},
        )

    factory = _mock_factory(handler)

    result = await replay(
        "/widgets",
        backend="new",
        client_factory=factory,
        captured=captured,
        parity_dir=parity_dir,
        snapshots_dir=tmp_path / "snaps",
    )
    assert result.passed, f"allow_extra_fields did not absorb the extra: {result.divergences}"


@pytest.mark.asyncio
async def test_unallowed_extra_field_fails(tmp_path: Path):
    parity_dir = tmp_path / "parity"
    parity_dir.mkdir()
    (parity_dir / "_default.yaml").write_text(
        "redact_fields: []\n"
        "allow_extra_fields: []\n"
        "strict_match_fields: []\n"
        "behavior:\n"
        "  status_code: 200\n"
    )

    captured = Snapshot(
        backend="old",
        endpoint="/widgets",
        method="GET",
        status_code=200,
        headers={},
        body_json={"name": "widget"},
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"name": "widget", "rogue": "field"})

    result = await replay(
        "/widgets",
        backend="new",
        client_factory=_mock_factory(handler),
        captured=captured,
        parity_dir=parity_dir,
        snapshots_dir=tmp_path / "snaps",
    )
    assert not result.passed
    assert any("body diverges" in d for d in result.divergences)


def test_redaction_normalizes_timestamps():
    contract = ParityContract(
        redact_fields=["$.created_at", "$.id"],
        allow_extra_fields=[],
        strict_match_fields=[],
    )
    captured = Snapshot(
        backend="old",
        endpoint="/x",
        method="GET",
        status_code=200,
        headers={},
        body_json={"id": "uuid-A", "created_at": "2024-01-01T00:00:00Z", "name": "alice"},
    )
    replayed = Snapshot(
        backend="new",
        endpoint="/x",
        method="GET",
        status_code=200,
        headers={},
        body_json={"id": "uuid-B", "created_at": "2025-12-31T23:59:59Z", "name": "alice"},
    )
    result = compare(captured, replayed, contract)
    assert result.passed, result.divergences


def test_strict_match_field_must_be_equal():
    contract = ParityContract(
        redact_fields=[],
        allow_extra_fields=[],
        strict_match_fields=["$.status"],
    )
    captured = Snapshot(
        backend="old",
        endpoint="/healthz",
        method="GET",
        status_code=200,
        headers={},
        body_json={"status": "ok"},
    )
    replayed = Snapshot(
        backend="new",
        endpoint="/healthz",
        method="GET",
        status_code=200,
        headers={},
        body_json={"status": "degraded"},
    )
    result = compare(captured, replayed, contract)
    assert not result.passed
    assert any("strict_match_fields" in d for d in result.divergences)


def test_status_code_mismatch_diverges():
    contract = ParityContract()
    captured = Snapshot(
        backend="old",
        endpoint="/x",
        method="GET",
        status_code=200,
        headers={},
        body_json={"a": 1},
    )
    replayed = Snapshot(
        backend="new",
        endpoint="/x",
        method="GET",
        status_code=500,
        headers={},
        body_json={"a": 1},
    )
    result = compare(captured, replayed, contract)
    assert not result.passed
    assert any("status_code" in d.lower() or "status" in d.lower() for d in result.divergences)


def test_row_count_behavior_compares_list_length():
    contract = ParityContract(
        behavior={"status_code": 200, "row_count": "$.items"},
    )
    captured = Snapshot(
        backend="old",
        endpoint="/list",
        method="GET",
        status_code=200,
        headers={},
        body_json={"items": [{"id": 1}, {"id": 2}, {"id": 3}]},
    )
    replayed_same = Snapshot(
        backend="new",
        endpoint="/list",
        method="GET",
        status_code=200,
        headers={},
        body_json={"items": [{"id": 10}, {"id": 20}, {"id": 30}]},
    )
    contract_with_redact = ParityContract(
        redact_fields=["$.items[*].id"],
        behavior={"status_code": 200, "row_count": "$.items"},
    )
    assert compare(captured, replayed_same, contract_with_redact).passed

    replayed_short = Snapshot(
        backend="new",
        endpoint="/list",
        method="GET",
        status_code=200,
        headers={},
        body_json={"items": [{"id": 1}, {"id": 2}]},
    )
    result = compare(captured, replayed_short, contract_with_redact)
    assert not result.passed
    assert any("row_count" in d for d in result.divergences)
