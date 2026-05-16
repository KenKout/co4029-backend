"""Parity contracts: declarative rules for behavioral equality between old and new backends.

A characterization test compares a *captured* response (from one backend) against a *replayed*
response (from another backend) using a contract. The contract specifies which fields can be
ignored (timestamps, IDs), which extra fields are tolerated, and which fields must match exactly.

Contracts are YAML files under ``tests/parity/``:
  - ``<endpoint-slug>.yaml`` — endpoint-specific overrides
  - ``_default.yaml`` — baseline, applied when no specific contract exists

The slug rule is identical to the snapshot slug (see ``slugify``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field

PARITY_DIR = Path(__file__).resolve().parents[1] / "parity"


class BehaviorContract(BaseModel):
    """Behavioral expectations enforced before payload diffing."""

    status_code: int | None = None
    """Expected HTTP status. ``None`` means: any 2xx is acceptable."""

    row_count: str | None = None
    """JSONPath to a list whose length must match between snapshots (list endpoints)."""


class ParityContract(BaseModel):
    """Declarative parity rules for a single endpoint."""

    endpoint: str | None = None
    redact_fields: list[str] = Field(default_factory=list)
    allow_extra_fields: list[str] = Field(default_factory=list)
    strict_match_fields: list[str] = Field(default_factory=list)
    behavior: BehaviorContract = Field(default_factory=BehaviorContract)


def slugify(endpoint: str) -> str:
    """Normalize an endpoint path into a filesystem-safe slug.

    ``/healthz`` -> ``healthz``
    ``/api/v1/users/{id}`` -> ``api_v1_users_{id}``
    Trailing slashes are stripped; query strings are dropped.
    """
    path = endpoint.split("?", 1)[0].rstrip("/")
    if not path:
        return "_root"
    slug = path.replace("/", "_").lstrip("_")
    return slug or "_root"


def load_contract(endpoint: str, parity_dir: Path | None = None) -> ParityContract:
    """Load the contract for an endpoint.

    Resolution order:
      1. ``<parity_dir>/<slug>.yaml`` if present
      2. ``<parity_dir>/_default.yaml`` otherwise

    The default contract MUST exist; missing it is a configuration error.
    """
    base = parity_dir or PARITY_DIR
    specific = base / f"{slugify(endpoint)}.yaml"
    default = base / "_default.yaml"
    target = specific if specific.is_file() else default
    if not target.is_file():
        raise FileNotFoundError(
            f"No parity contract for {endpoint!r}: looked for {specific} and {default}"
        )
    raw = yaml.safe_load(target.read_text()) or {}
    return ParityContract.model_validate(raw)
