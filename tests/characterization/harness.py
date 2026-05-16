"""Snapshot capture for characterization testing.

Hits a running backend over HTTP and persists the response to JSON.
Snapshots live under ``tests/characterization/snapshots/`` and are checked into git
so replay can reproduce assertions without the legacy backend running.
"""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx

from .contract import slugify

Backend = Literal["old", "new"]
ClientFactory = Callable[[str], Awaitable[httpx.AsyncClient]]

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"

DEFAULT_OLD_URL = "http://localhost:8000"
DEFAULT_NEW_URL = "http://localhost:8001"


@dataclass
class Snapshot:
    backend: Backend
    endpoint: str
    method: str
    status_code: int
    headers: dict[str, str]
    body_json: Any
    request_body: dict[str, Any] | None = None
    request_headers: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "endpoint": self.endpoint,
            "method": self.method,
            "status_code": self.status_code,
            "headers": self.headers,
            "body_json": self.body_json,
            "request_body": self.request_body,
            "request_headers": self.request_headers,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Snapshot:
        return cls(
            backend=data["backend"],
            endpoint=data["endpoint"],
            method=data.get("method", "GET"),
            status_code=data["status_code"],
            headers=data.get("headers", {}),
            body_json=data.get("body_json"),
            request_body=data.get("request_body"),
            request_headers=data.get("request_headers", {}),
        )


def resolve_base_url(backend: Backend) -> str:
    if backend == "old":
        return os.environ.get("OLD_BACKEND_URL", DEFAULT_OLD_URL)
    return os.environ.get("NEW_BACKEND_URL", DEFAULT_NEW_URL)


def snapshot_path(endpoint: str, *, backend: Backend, snapshots_dir: Path | None = None) -> Path:
    base = snapshots_dir or SNAPSHOT_DIR
    return base / f"{slugify(endpoint)}.{backend}.json"


async def _default_client(base_url: str) -> httpx.AsyncClient:
    return httpx.AsyncClient(base_url=base_url, timeout=10.0)


async def capture_snapshot(
    endpoint: str,
    *,
    backend: Backend,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
    client_factory: ClientFactory | None = None,
    snapshots_dir: Path | None = None,
    persist: bool = True,
) -> Snapshot:
    """Call ``endpoint`` against the chosen backend and save the response.

    ``client_factory`` lets tests inject an ``httpx.AsyncClient`` over an ASGI app
    (via ``ASGITransport``) instead of opening a real socket. Returning a client
    bound to ``base_url`` is the contract — the factory is awaited to support
    async setup such as starting an in-process server.
    """
    factory = client_factory or _default_client
    base_url = resolve_base_url(backend)
    request_headers = dict(headers or {})

    async with await factory(base_url) as client:
        response = await client.request(
            method,
            endpoint,
            headers=request_headers or None,
            json=json_body,
        )

    try:
        body_json: Any = response.json()
    except (ValueError, json.JSONDecodeError):
        body_json = response.text

    snapshot = Snapshot(
        backend=backend,
        endpoint=endpoint,
        method=method.upper(),
        status_code=response.status_code,
        headers={k.lower(): v for k, v in response.headers.items()},
        body_json=body_json,
        request_body=json_body,
        request_headers=request_headers,
    )

    if persist:
        target = snapshot_path(endpoint, backend=backend, snapshots_dir=snapshots_dir)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(snapshot.to_json(), indent=2, sort_keys=True))

    return snapshot


def load_snapshot(
    endpoint: str, *, backend: Backend, snapshots_dir: Path | None = None
) -> Snapshot:
    target = snapshot_path(endpoint, backend=backend, snapshots_dir=snapshots_dir)
    if not target.is_file():
        raise FileNotFoundError(f"No snapshot at {target}; run capture first")
    return Snapshot.from_json(json.loads(target.read_text()))
