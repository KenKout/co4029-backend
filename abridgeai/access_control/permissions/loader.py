"""Load and cross-validate the permission catalog and role seeds.

The two YAML files live next to this module so they ship inside the wheel.
We use :mod:`importlib.resources` to locate them, which works in editable
installs (``uv sync``) and in installed wheels alike.

Cross-validation (every role-permission code must resolve to a catalog
entry) happens here because it needs both parsed objects. The loader is
also where the ``"ALL"`` sentinel for the admin role is expanded into the
full catalog code list.
"""

from __future__ import annotations

from importlib import resources

import yaml

from abridgeai.access_control.permissions.schema import (
    Catalog,
    RoleDef,
    RoleSeeds,
)

_PACKAGE = "abridgeai.access_control.permissions"
_CATALOG_FILENAME = "catalog.yaml"
_ROLE_SEEDS_FILENAME = "role_seeds.yaml"


def _read_yaml(filename: str) -> object:
    text = (resources.files(_PACKAGE) / filename).read_text(encoding="utf-8")
    return yaml.safe_load(text)


def load_catalog() -> Catalog:
    """Parse and validate ``catalog.yaml``.

    Raises:
        pydantic.ValidationError: structural problems (missing fields,
            duplicate codes, malformed code formats, unknown keys).
    """
    raw = _read_yaml(_CATALOG_FILENAME)
    return Catalog.model_validate(raw)


def load_role_seeds(catalog: Catalog | None = None) -> RoleSeeds:
    """Parse, validate, and (when ``catalog`` is supplied) cross-validate
    ``role_seeds.yaml``.

    When ``catalog`` is provided, two extra steps run after schema validation:

    1. Every concrete permission code on every role must exist in the
       catalog. Any unknown codes are reported in a single ``ValueError``
       so multiple typos surface in one run.
    2. Roles whose ``permissions`` field is the sentinel ``"ALL"`` get
       their permissions list expanded to ``catalog.codes()``. We materialise
       the full list so downstream consumers (Alembic data migrations, admin
       UIs, audit reports) never have to know about the sentinel.

    Without ``catalog``, only the schema validators run; this is useful for
    cheap structural checks in environments where the catalog file is not
    available (e.g. unit tests of the YAML shape).
    """
    raw = _read_yaml(_ROLE_SEEDS_FILENAME)
    seeds = RoleSeeds.model_validate(raw)
    if catalog is None:
        return seeds

    known = catalog.codes()
    missing: dict[str, list[str]] = {}
    expanded_roles: list[RoleDef] = []
    for role in seeds.roles:
        if role.permissions == "ALL":
            expanded = sorted(known)
            expanded_roles.append(role.model_copy(update={"permissions": expanded}))
            continue
        unknown = [code for code in role.permissions if code not in known]
        if unknown:
            missing[role.code] = sorted(set(unknown))
        expanded_roles.append(role)

    if missing:
        details = "; ".join(f"{role}={codes!r}" for role, codes in sorted(missing.items()))
        msg = f"role_seeds.yaml references permission codes not present in catalog.yaml: {details}"
        raise ValueError(msg)

    return seeds.model_copy(update={"roles": expanded_roles})


__all__ = ["load_catalog", "load_role_seeds"]
