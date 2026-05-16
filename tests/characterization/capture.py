from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from .harness import Backend, capture_snapshot


def _parse_headers(raw: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in raw or []:
        if ":" not in item:
            raise SystemExit(f"--header must be 'Name: Value', got {item!r}")
        name, value = item.split(":", 1)
        out[name.strip()] = value.strip()
    return out


def _parse_body(raw: str | None) -> dict[str, Any] | None:
    if raw is None:
        return None
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            return json.load(f)
    return json.loads(raw)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tests.characterization.capture",
        description="Capture an HTTP snapshot from old or new backend.",
    )
    parser.add_argument("--endpoint", required=True, help="Path, e.g. /healthz")
    parser.add_argument("--backend", required=True, choices=("old", "new"))
    parser.add_argument("--method", default="GET")
    parser.add_argument(
        "--header",
        action="append",
        help="HTTP header 'Name: Value' (repeatable)",
    )
    parser.add_argument(
        "--json-body",
        help="JSON request body string, or @path/to/file.json",
    )
    return parser


async def _run(args: argparse.Namespace) -> int:
    backend: Backend = args.backend
    snap = await capture_snapshot(
        args.endpoint,
        backend=backend,
        method=args.method,
        headers=_parse_headers(args.header),
        json_body=_parse_body(args.json_body),
    )
    print(
        json.dumps(
            {
                "endpoint": snap.endpoint,
                "backend": snap.backend,
                "status_code": snap.status_code,
            },
            indent=2,
        )
    )
    return 0 if 200 <= snap.status_code < 500 else 1


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return asyncio.run(_run(args))


if __name__ == "__main__":
    sys.exit(main())
