from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from abridgeai.access_control.permissions.loader import load_catalog, load_role_seeds
from abridgeai.core.config import get_settings

SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"
SYSTEM_USER_EMAIL = "system@abridgeai.local"
PRIOR_HEAD = "0003_fk_cascade_no_action"
SEED_HEAD = "0004_seed_permission_catalog"

BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _alembic_config() -> Config:
    cfg = Config(str(BACKEND_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_ROOT / "migrations"))
    return cfg


@pytest_asyncio.fixture
async def engine() -> AsyncEngine:
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest.fixture
def alembic_cfg() -> Config:
    return _alembic_config()


@pytest_asyncio.fixture
async def at_seed_head(alembic_cfg: Config) -> None:
    command.upgrade(alembic_cfg, SEED_HEAD)
    yield


async def _count_permissions(engine: AsyncEngine) -> int:
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT count(*) FROM permissions"))
        return int(result.scalar_one())


async def _count_roles(engine: AsyncEngine) -> int:
    async with engine.connect() as conn:
        result = await conn.execute(text("SELECT count(*) FROM roles"))
        return int(result.scalar_one())


async def _count_role_permissions_for(engine: AsyncEngine, role_code: str) -> int:
    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                SELECT count(*) FROM role_permissions rp
                JOIN roles r ON r.id = rp.role_id
                WHERE r.code = :code
                """
            ),
            {"code": role_code},
        )
        return int(result.scalar_one())


async def test_catalog_seeded_after_upgrade(
    at_seed_head: None,
    engine: AsyncEngine,
) -> None:
    catalog = load_catalog()
    expected_perm_count = len(catalog.permissions)

    perm_count = await _count_permissions(engine)
    assert perm_count >= expected_perm_count

    async with engine.connect() as conn:
        role_codes_result = await conn.execute(text("SELECT code FROM roles ORDER BY code"))
        role_codes = [row[0] for row in role_codes_result]
    assert {"admin", "hod", "manager", "student", "teacher"}.issubset(set(role_codes))

    async with engine.connect() as conn:
        sys_result = await conn.execute(
            text("SELECT primary_email, status FROM users WHERE id = CAST(:uid AS uuid)"),
            {"uid": SYSTEM_USER_ID},
        )
        row = sys_result.one()
    assert row[0] == SYSTEM_USER_EMAIL
    assert row[1] == "inactive"


async def test_role_permissions_match_yaml(
    at_seed_head: None,
    engine: AsyncEngine,
) -> None:
    catalog = load_catalog()
    seeds = load_role_seeds(catalog)

    catalog_count = len(catalog.permissions)
    by_code = {role.code: role for role in seeds.roles}

    admin_count = await _count_role_permissions_for(engine, "admin")
    assert admin_count == catalog_count

    student_role = by_code["student"]
    assert isinstance(student_role.permissions, list)
    student_count = await _count_role_permissions_for(engine, "student")
    assert student_count == len(student_role.permissions)
    assert student_count == 4

    async with engine.connect() as conn:
        student_perms_result = await conn.execute(
            text(
                """
                SELECT p.code FROM role_permissions rp
                JOIN roles r ON r.id = rp.role_id
                JOIN permissions p ON p.id = rp.permission_id
                WHERE r.code = 'student'
                ORDER BY p.code
                """
            )
        )
        student_codes = [row[0] for row in student_perms_result]
    assert sorted(student_codes) == sorted(student_role.permissions)


async def test_idempotent_re_run(
    at_seed_head: None,
    alembic_cfg: Config,
    engine: AsyncEngine,
) -> None:
    perm_before = await _count_permissions(engine)
    role_before = await _count_roles(engine)

    command.upgrade(alembic_cfg, SEED_HEAD)

    perm_after = await _count_permissions(engine)
    role_after = await _count_roles(engine)
    assert perm_after == perm_before
    assert role_after == role_before


@pytest.mark.destructive
async def test_round_trip_upgrade_downgrade_upgrade(
    at_seed_head: None,
    alembic_cfg: Config,
    engine: AsyncEngine,
) -> None:
    catalog = load_catalog()
    seeded_codes = sorted(catalog.codes())
    seeded_role_codes = ["admin", "hod", "manager", "student", "teacher"]

    async with engine.connect() as conn:
        result = await conn.execute(
            text(
                """
                DELETE FROM user_role_assignments
                WHERE role_id IN (
                    SELECT id FROM roles WHERE code = ANY(:codes)
                )
                """
            ),
            {"codes": seeded_role_codes},
        )
        await conn.commit()
        del result

    command.downgrade(alembic_cfg, PRIOR_HEAD)

    async with engine.connect() as conn:
        seeded_perm_left = await conn.execute(
            text("SELECT count(*) FROM permissions WHERE code = ANY(:codes)"),
            {"codes": seeded_codes},
        )
        seeded_role_left = await conn.execute(
            text("SELECT count(*) FROM roles WHERE code = ANY(:codes)"),
            {"codes": seeded_role_codes},
        )
        sys_left = await conn.execute(
            text("SELECT count(*) FROM users WHERE id = CAST(:uid AS uuid)"),
            {"uid": SYSTEM_USER_ID},
        )
    assert seeded_perm_left.scalar_one() == 0
    assert seeded_role_left.scalar_one() == 0
    assert sys_left.scalar_one() == 0

    command.upgrade(alembic_cfg, SEED_HEAD)

    async with engine.connect() as conn:
        seeded_perm_back = await conn.execute(
            text("SELECT count(*) FROM permissions WHERE code = ANY(:codes)"),
            {"codes": seeded_codes},
        )
        seeded_role_back = await conn.execute(
            text("SELECT count(*) FROM roles WHERE code = ANY(:codes)"),
            {"codes": seeded_role_codes},
        )
    assert seeded_perm_back.scalar_one() == len(seeded_codes)
    assert seeded_role_back.scalar_one() == len(seeded_role_codes)
