from __future__ import annotations

import textwrap

import pytest
import yaml
from pydantic import ValidationError

from abridgeai.access_control.permissions import (
    Catalog,
    RoleSeeds,
    load_catalog,
    load_role_seeds,
)


def test_catalog_count_at_least_30() -> None:
    catalog = load_catalog()
    assert len(catalog.permissions) >= 30
    assert len(catalog.codes()) == len(catalog.permissions)


def test_role_perms_resolve_against_catalog() -> None:
    catalog = load_catalog()
    seeds = load_role_seeds(catalog)
    known = catalog.codes()
    assert {r.code for r in seeds.roles} == {
        "student",
        "teacher",
        "hod",
        "manager",
        "admin",
    }
    for role in seeds.roles:
        assert isinstance(role.permissions, list)
        for code in role.permissions:
            assert code in known, f"role {role.code} references unknown code {code}"


def test_admin_has_all_permissions() -> None:
    catalog = load_catalog()
    seeds = load_role_seeds(catalog)
    admin = next(r for r in seeds.roles if r.code == "admin")
    assert isinstance(admin.permissions, list)
    assert set(admin.permissions) == catalog.codes()


def test_load_role_seeds_without_catalog_keeps_all_sentinel() -> None:
    seeds = load_role_seeds()
    admin = next(r for r in seeds.roles if r.code == "admin")
    assert admin.permissions == "ALL"


def test_malformed_catalog_missing_code_rejected() -> None:
    bad = textwrap.dedent(
        """
        version: 1
        permissions:
          - description: missing code field
            category: course
        """,
    )
    with pytest.raises(ValidationError):
        Catalog.model_validate(yaml.safe_load(bad))


def test_malformed_catalog_duplicate_code_rejected() -> None:
    bad = textwrap.dedent(
        """
        version: 1
        permissions:
          - {code: course.read, description: a, category: course}
          - {code: course.read, description: b, category: course}
        """,
    )
    with pytest.raises(ValidationError):
        Catalog.model_validate(yaml.safe_load(bad))


def test_malformed_catalog_invalid_code_format_rejected() -> None:
    bad = textwrap.dedent(
        """
        version: 1
        permissions:
          - {code: Course-Read, description: bad casing/dash, category: course}
        """,
    )
    with pytest.raises(ValidationError):
        Catalog.model_validate(yaml.safe_load(bad))


def test_malformed_catalog_extra_field_rejected() -> None:
    bad = textwrap.dedent(
        """
        version: 1
        permissions:
          - code: course.read
            description: ok
            category: course
            unknown_extra: nope
        """,
    )
    with pytest.raises(ValidationError):
        Catalog.model_validate(yaml.safe_load(bad))


def test_malformed_role_seeds_wrong_role_set_rejected() -> None:
    bad = textwrap.dedent(
        """
        version: 1
        roles:
          - {code: student, name: S, description: d, permissions: [course.read]}
        """,
    )
    with pytest.raises(ValidationError):
        RoleSeeds.model_validate(yaml.safe_load(bad))


def test_role_with_unknown_permission_code_rejected_by_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog = load_catalog()
    bad_seeds = textwrap.dedent(
        """
        version: 1
        roles:
          - code: student
            name: Student
            description: d
            permissions: [course.read, totally.bogus.code]
          - code: teacher
            name: Teacher
            description: d
            permissions: [course.read]
          - code: hod
            name: HOD
            description: d
            permissions: [course.read]
          - code: manager
            name: Manager
            description: d
            permissions: [course.read]
          - code: admin
            name: Admin
            description: d
            permissions: ALL
        """,
    )

    from abridgeai.access_control.permissions import loader as loader_mod

    def fake_read(filename: str) -> object:
        if filename == loader_mod._ROLE_SEEDS_FILENAME:
            return yaml.safe_load(bad_seeds)
        return yaml.safe_load(
            (__import__("importlib").resources.files(loader_mod._PACKAGE) / filename).read_text(
                encoding="utf-8"
            ),
        )

    monkeypatch.setattr(loader_mod, "_read_yaml", fake_read)
    with pytest.raises(ValueError, match="totally.bogus.code"):
        load_role_seeds(catalog)
