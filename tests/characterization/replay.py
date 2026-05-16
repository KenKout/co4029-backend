"""Replay: re-call an endpoint and compare against a saved snapshot under a parity contract."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jsonpath_ng.exceptions import JsonPathParserError
from jsonpath_ng.ext import parse as jsonpath_parse

from .contract import ParityContract, load_contract
from .harness import (
    Backend,
    ClientFactory,
    Snapshot,
    capture_snapshot,
    load_snapshot,
)

REDACTED = "[REDACTED]"


@dataclass
class ReplayResult:
    passed: bool
    divergences: list[str] = field(default_factory=list)
    captured: Snapshot | None = None
    replayed: Snapshot | None = None

    def assert_ok(self) -> None:
        if not self.passed:
            raise AssertionError("Parity FAILED:\n  - " + "\n  - ".join(self.divergences))


def _compile(expr: str) -> Any:
    try:
        return jsonpath_parse(expr)
    except (JsonPathParserError, Exception) as exc:  # noqa: BLE001 — surface bad contracts loudly
        raise ValueError(f"Invalid JSONPath {expr!r}: {exc}") from exc


def _apply_redactions(body: Any, paths: Iterable[str]) -> Any:
    """Replace every value matched by ``paths`` with the REDACTED sentinel.

    Operates on a deep copy so the snapshot on disk is never mutated.
    """
    out = deepcopy(body)
    for expr in paths:
        compiled = _compile(expr)
        for match in compiled.find(out):
            full = match.full_path
            full.update(out, REDACTED)
    return out


def _strip_paths(body: Any, paths: Iterable[str]) -> Any:
    """Remove every value matched by ``paths`` (used for allow_extra_fields).

    Missing paths are silently ignored — that is the entire point of the allow-list.
    """
    out = deepcopy(body)
    for expr in paths:
        compiled = _compile(expr)
        matches = list(compiled.find(out))
        for match in matches:
            parent = match.context.value if match.context else None
            field_or_index = str(match.path).strip("[]'\"")
            if isinstance(parent, dict) and field_or_index in parent:
                parent.pop(field_or_index)
            elif isinstance(parent, list):
                try:
                    idx = int(field_or_index)
                    if 0 <= idx < len(parent):
                        parent.pop(idx)
                except ValueError:
                    pass
    return out


def _strict_match(captured: Any, replayed: Any, paths: Iterable[str]) -> list[str]:
    divergences: list[str] = []
    for expr in paths:
        compiled = _compile(expr)
        cap_matches = [m.value for m in compiled.find(captured)]
        rep_matches = [m.value for m in compiled.find(replayed)]
        if cap_matches != rep_matches:
            divergences.append(
                f"strict_match_fields[{expr!r}] differ: captured={cap_matches!r} replayed={rep_matches!r}"
            )
    return divergences


def _is_2xx(status: int) -> bool:
    return 200 <= status < 300


def _check_behavior(
    contract: ParityContract,
    captured: Snapshot,
    replayed: Snapshot,
) -> list[str]:
    divergences: list[str] = []
    expected = contract.behavior.status_code
    if expected is not None:
        if captured.status_code != expected:
            divergences.append(
                f"captured status {captured.status_code} != contract.behavior.status_code {expected}"
            )
        if replayed.status_code != expected:
            divergences.append(
                f"replayed status {replayed.status_code} != contract.behavior.status_code {expected}"
            )
    else:
        if not _is_2xx(captured.status_code):
            divergences.append(f"captured status {captured.status_code} is not 2xx")
        if not _is_2xx(replayed.status_code):
            divergences.append(f"replayed status {replayed.status_code} is not 2xx")
    if captured.status_code != replayed.status_code:
        divergences.append(
            f"status_code mismatch: captured={captured.status_code} replayed={replayed.status_code}"
        )

    row_count_path = contract.behavior.row_count
    if row_count_path:
        compiled = _compile(row_count_path)
        cap_list = next((m.value for m in compiled.find(captured.body_json)), None)
        rep_list = next((m.value for m in compiled.find(replayed.body_json)), None)
        cap_n = len(cap_list) if isinstance(cap_list, list) else None
        rep_n = len(rep_list) if isinstance(rep_list, list) else None
        if cap_n is None or rep_n is None:
            divergences.append(
                f"behavior.row_count path {row_count_path!r} did not resolve to a list "
                f"(captured={cap_n!r}, replayed={rep_n!r})"
            )
        elif cap_n != rep_n:
            divergences.append(
                f"behavior.row_count mismatch at {row_count_path!r}: captured={cap_n} replayed={rep_n}"
            )
    return divergences


def compare(
    captured: Snapshot,
    replayed: Snapshot,
    contract: ParityContract,
) -> ReplayResult:
    """Diff two snapshots under a contract; the heart of the harness."""
    divergences: list[str] = _check_behavior(contract, captured, replayed)
    divergences.extend(
        _strict_match(captured.body_json, replayed.body_json, contract.strict_match_fields)
    )

    cap_redacted = _apply_redactions(captured.body_json, contract.redact_fields)
    rep_redacted = _apply_redactions(replayed.body_json, contract.redact_fields)
    rep_pruned = _strip_paths(rep_redacted, contract.allow_extra_fields)
    cap_pruned = _strip_paths(cap_redacted, contract.allow_extra_fields)

    if cap_pruned != rep_pruned:
        divergences.append(
            "body diverges after redactions/allowances:\n"
            f"  captured={cap_pruned!r}\n  replayed={rep_pruned!r}"
        )

    return ReplayResult(
        passed=not divergences,
        divergences=divergences,
        captured=captured,
        replayed=replayed,
    )


async def replay(
    endpoint: str,
    *,
    backend: Backend,
    headers: dict[str, str] | None = None,
    method: str = "GET",
    json_body: dict[str, Any] | None = None,
    client_factory: ClientFactory | None = None,
    captured_backend: Backend = "old",
    snapshots_dir: Path | None = None,
    parity_dir: Path | None = None,
    captured: Snapshot | None = None,
) -> ReplayResult:
    """Replay an endpoint against ``backend`` and compare with a previously captured snapshot.

    By default the captured snapshot comes from the legacy backend (``captured_backend="old"``)
    so the contract describes "does the new backend behave like the old one?". Tests may pass
    an explicit ``captured`` snapshot to short-circuit disk lookup (e.g. self-consistency).
    """
    contract = load_contract(endpoint, parity_dir=parity_dir)

    if captured is None:
        captured = load_snapshot(endpoint, backend=captured_backend, snapshots_dir=snapshots_dir)

    replayed = await capture_snapshot(
        endpoint,
        backend=backend,
        headers=headers,
        method=method,
        json_body=json_body,
        client_factory=client_factory,
        snapshots_dir=snapshots_dir,
        persist=False,
    )

    return compare(captured, replayed, contract)
