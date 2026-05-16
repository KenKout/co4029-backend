"""Pydantic models that validate the permission catalog and role seeds YAML.

Two YAML files are the single source of truth for the access-control system:

* ``catalog.yaml`` declares every permission code the platform understands.
* ``role_seeds.yaml`` declares the five built-in roles and the permission
  codes each role is granted at bootstrap.

Both files are loaded through :mod:`abridgeai.access_control.permissions.loader`.
The schemas in this module enforce structural correctness (versioning, code
formatting, uniqueness) so typos and drift fail loudly instead of silently
granting or denying access.

Cross-validation between the two files (every role-permission code must
resolve to a catalog entry) lives in the loader because it requires both
parsed objects.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# Permission codes use lowercase dotted segments, e.g. ``course.read.draft``.
# Hyphens, slashes, uppercase, and trailing dots are rejected so the codes
# remain greppable and stable across SQL, YAML, and Python literals.
_CODE_PATTERN = re.compile(r"^[a-z_]+(\.[a-z_]+)+$")

# The five roles below match the platform's bootstrap roster. Adding or
# removing a role here is intentional and requires a coordinated migration;
# Literal[...] catches accidental renames in YAML.
RoleCode = Literal["student", "teacher", "hod", "manager", "admin"]


class PermissionDef(BaseModel):
    """A single permission entry in :file:`catalog.yaml`.

    Attributes:
        code: Dotted permission identifier (e.g. ``course.read.draft``).
        description: Human-readable explanation surfaced in admin tooling.
        category: Top-level grouping (``course``, ``user``, ``system`` ...);
            used by admin UIs to render the catalog.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: str
    description: str
    category: str

    @field_validator("code")
    @classmethod
    def _code_must_match_format(cls, value: str) -> str:
        if not _CODE_PATTERN.fullmatch(value):
            msg = (
                f"permission code {value!r} is not in the required format "
                f"'^[a-z_]+(\\.[a-z_]+)+$' (lowercase dotted segments, e.g. "
                f"'course.read.draft')"
            )
            raise ValueError(msg)
        return value

    @field_validator("description")
    @classmethod
    def _description_must_be_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("permission description must be non-empty")
        return value

    @field_validator("category")
    @classmethod
    def _category_must_be_lowercase_word(cls, value: str) -> str:
        if not re.fullmatch(r"[a-z_]+", value):
            raise ValueError(
                f"permission category {value!r} must be lowercase letters/underscores",
            )
        return value


class Catalog(BaseModel):
    """Top-level model for :file:`catalog.yaml`.

    Validators enforce that ``code`` values are unique across the file and
    that the catalog never accidentally ships empty.
    """

    model_config = ConfigDict(extra="forbid")

    version: int
    permissions: list[PermissionDef]

    @field_validator("version")
    @classmethod
    def _version_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("catalog version must be >= 1")
        return value

    @model_validator(mode="after")
    def _codes_must_be_unique_and_nonempty(self) -> Catalog:
        if not self.permissions:
            raise ValueError("catalog.permissions must not be empty")
        seen: set[str] = set()
        duplicates: list[str] = []
        for perm in self.permissions:
            if perm.code in seen:
                duplicates.append(perm.code)
            seen.add(perm.code)
        if duplicates:
            raise ValueError(
                f"duplicate permission codes in catalog: {sorted(set(duplicates))!r}",
            )
        return self

    def codes(self) -> set[str]:
        """Return the catalog's permission codes as a set for membership checks."""
        return {p.code for p in self.permissions}


class RoleDef(BaseModel):
    """A single role entry in :file:`role_seeds.yaml`.

    ``permissions`` is either an explicit list of catalog codes or the
    sentinel ``"ALL"``, which the loader expands into the full catalog
    after the catalog has been parsed. We use a sentinel (not a wildcard
    glob) so YAML stays statically auditable.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    code: RoleCode
    name: str
    description: str
    permissions: list[str] | Literal["ALL"]

    @field_validator("name", "description")
    @classmethod
    def _string_fields_nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("role name/description must be non-empty")
        return value

    @field_validator("permissions")
    @classmethod
    def _permissions_shape(
        cls,
        value: list[str] | Literal["ALL"],
    ) -> list[str] | Literal["ALL"]:
        if value == "ALL":
            return value
        if not isinstance(value, list):  # pragma: no cover - pydantic guards
            raise ValueError("role.permissions must be a list of codes or 'ALL'")
        if not value:
            raise ValueError("role.permissions list must not be empty")
        bad = [p for p in value if not _CODE_PATTERN.fullmatch(p)]
        if bad:
            raise ValueError(
                f"role.permissions contains malformed codes: {sorted(set(bad))!r}",
            )
        # Reject duplicates inside a single role to surface accidental copy/paste.
        if len(set(value)) != len(value):
            dupes = sorted({p for p in value if value.count(p) > 1})
            raise ValueError(f"role.permissions has duplicate codes: {dupes!r}")
        return value


class RoleSeeds(BaseModel):
    """Top-level model for :file:`role_seeds.yaml`."""

    model_config = ConfigDict(extra="forbid")

    version: int
    roles: list[RoleDef]

    @field_validator("version")
    @classmethod
    def _version_must_be_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("role_seeds version must be >= 1")
        return value

    @model_validator(mode="after")
    def _roles_must_match_canonical_set(self) -> RoleSeeds:
        expected: set[str] = {"student", "teacher", "hod", "manager", "admin"}
        codes = [r.code for r in self.roles]
        if len(codes) != len(set(codes)):
            dupes = sorted({c for c in codes if codes.count(c) > 1})
            raise ValueError(f"duplicate role codes in role_seeds: {dupes!r}")
        if set(codes) != expected:
            missing = sorted(expected - set(codes))
            extra = sorted(set(codes) - expected)
            raise ValueError(
                "role_seeds.roles must declare exactly the canonical 5 roles "
                f"{sorted(expected)!r}; missing={missing!r}, extra={extra!r}",
            )
        return self
