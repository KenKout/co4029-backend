"""Coverage for ``access_control/routers/organizations.py``.

The service layer owns the business rules (tested in
``test_org_admin_service.py``); what the ROUTER owns is who may see what, and
how a rule's refusal is rendered. Both are security-shaped:

* **Listing visibility.** ``_REQUIRE_ORG_MANAGE`` accepts ``org_unit.manage``
  and ``user.bulk_import``, and the flat permission set behind it cannot tell
  a grant made inside one tenant from a global one. The module's own docstring
  records what that cost: "Unrestricted listing therefore leaked the whole
  tenant roster — names, slugs and statuses of every customer — to any
  manager." Only ``system.administer`` keeps the global view. That is the
  single most important assertion in this file.

* **Absent and foreign must be indistinguishable.** Routes addressed by a
  child resource's own id (a domain, a unit, a membership) resolve the owning
  organization first. A resource that does not exist and one belonging to
  another tenant both return 404 with the same body — otherwise the endpoint
  becomes an oracle for which ids exist elsewhere.

* **Error mapping.** ``NotFoundError`` → 404, ``ForbiddenError`` → 403,
  ``AppError`` → 422. A rule that raises correctly but surfaces as a 500 is
  still a broken endpoint.

Isolation: every test builds its own two tenants and tears them down.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from fastapi import FastAPI
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.core.db import get_db
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.access_control.routers.organizations import router as org_router


def _async_url(database_url: str) -> str:
    if "+psycopg_async" in database_url:
        return database_url
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return database_url


def _ensure_head() -> None:
    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    cfg.set_main_option("script_location", str(root / "migrations"))
    command.upgrade(cfg, "head")


@pytest_asyncio.fixture
async def engine() -> AsyncIterator[AsyncEngine]:
    _ensure_head()
    eng = create_async_engine(_async_url(get_settings().database_url), pool_pre_ping=True)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


@pytest_asyncio.fixture
async def client(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[httpx.AsyncClient]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    app = FastAPI()
    app.include_router(org_router, prefix="/api/v1")
    app.dependency_overrides[get_db] = _override_get_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
    app.dependency_overrides.clear()


async def _bearer(engine: AsyncEngine, user_id: uuid.UUID) -> str:
    sid = uuid.uuid4()
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (:id, :uid, :h, NOW() + INTERVAL '1 hour')"
            ),
            {"id": sid, "uid": user_id, "h": hash_secret(generate_token())},
        )
    return create_access_token(user_id=user_id, session_id=sid)


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


class World:
    """Two tenants. ``master`` runs tenant A; ``platform`` is a global admin.

    ``other_master`` runs tenant B and exists so "another tenant's admin"
    is a real caller rather than a hypothetical.
    """

    def __init__(self) -> None:
        self.tag = uuid.uuid4().hex[:10]
        self.org_a = uuid.uuid4()
        self.org_b = uuid.uuid4()
        self.faculty_a = uuid.uuid4()
        self.faculty_b = uuid.uuid4()
        self.master = uuid.uuid4()
        self.other_master = uuid.uuid4()
        self.platform = uuid.uuid4()
        self.teacher = uuid.uuid4()
        self.membership_a = uuid.uuid4()
        self.membership_b = uuid.uuid4()
        self.domain_a = uuid.uuid4()


@pytest_asyncio.fixture
async def world(engine: AsyncEngine) -> AsyncIterator[World]:
    w = World()
    users = {
        "master": w.master,
        "other": w.other_master,
        "platform": w.platform,
        "teacher": w.teacher,
    }
    async with engine.begin() as conn:
        for org, name in ((w.org_a, "Router Tenant A"), (w.org_b, "Router Tenant B")):
            await conn.execute(
                text(
                    "INSERT INTO organizations (id, slug, name, status) "
                    "VALUES (:id, :slug, :name, 'active')"
                ),
                {"id": org, "slug": f"rt-{org.hex[:10]}", "name": name},
            )
        for label, uid in users.items():
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"rt-{w.tag}-{label}@test.local"},
            )
        for fid, org, name in (
            (w.faculty_a, w.org_a, "A Faculty"),
            (w.faculty_b, w.org_b, "B Faculty"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO org_units (id, organization_id, unit_type, name, code) "
                    "VALUES (:id, :org, 'faculty', :name, :code)"
                ),
                {"id": fid, "org": org, "name": name, "code": f"RF{fid.hex[:6]}"},
            )
        for mid, uid, org in (
            (w.membership_a, w.master, w.org_a),
            (w.membership_b, w.other_master, w.org_b),
        ):
            await conn.execute(
                text(
                    "INSERT INTO organization_memberships "
                    "(id, user_id, organization_id, status) "
                    "VALUES (:id, :uid, :org, 'active')"
                ),
                {"id": mid, "uid": uid, "org": org},
            )
        await conn.execute(
            text(
                "INSERT INTO organization_memberships "
                "(id, user_id, organization_id, status) "
                "VALUES (gen_random_uuid(), :uid, :org, 'active')"
            ),
            {"uid": w.teacher, "org": w.org_a},
        )
        await conn.execute(
            text(
                "INSERT INTO organization_domains (id, organization_id, domain, auto_provision) "
                "VALUES (:id, :org, :domain, FALSE)"
            ),
            {"id": w.domain_a, "org": w.org_a, "domain": f"rt-{w.tag}.edu"},
        )
        # Master Deans are org-scoped hod in their own tenant; the platform
        # admin is global. The teacher holds no management permission at all.
        for uid, role, scope, org in (
            (w.master, "hod", "organization", w.org_a),
            (w.other_master, "hod", "organization", w.org_b),
            (w.platform, "admin", "global", None),
            (w.teacher, "teacher", "organization", w.org_a),
        ):
            await conn.execute(
                text(
                    "INSERT INTO user_role_assignments "
                    "(user_id, role_id, scope_kind, organization_id) "
                    "SELECT :uid, r.id, :scope, :org FROM roles r WHERE r.code = :role"
                ),
                {"uid": uid, "scope": scope, "org": org, "role": role},
            )
    yield w
    async with engine.begin() as conn:
        for stmt in (
            "DELETE FROM auth_sessions WHERE user_id = ANY(:users)",
            "DELETE FROM user_faculty_assignments WHERE organization_id = ANY(:orgs)",
            "DELETE FROM user_role_assignments WHERE user_id = ANY(:users)",
            "DELETE FROM organization_memberships WHERE organization_id = ANY(:orgs)",
            "DELETE FROM organization_domains WHERE organization_id = ANY(:orgs)",
            "DELETE FROM org_units WHERE organization_id = ANY(:orgs)",
            "DELETE FROM users WHERE id = ANY(:users)",
            "DELETE FROM organizations WHERE id = ANY(:orgs)",
        ):
            await conn.execute(
                text(stmt), {"orgs": [w.org_a, w.org_b], "users": list(users.values())}
            )


# ---------------------------------------------------------------------------
# Listing visibility — the tenant-roster leak this module documents
# ---------------------------------------------------------------------------


async def test_a_tenant_admin_does_not_see_other_tenants(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    """The regression the module's docstring records.

    ``org_unit.manage`` is granted INSIDE a tenant, but the permission set the
    dependency checks is flat, so the route cannot infer scope from it. If the
    listing is not narrowed explicitly, every manager reads the names, slugs
    and statuses of every customer.
    """
    token = await _bearer(engine, world.master)
    resp = await client.get("/api/v1/admin/organizations", headers=_auth(token))
    assert resp.status_code == 200, resp.text

    ids = {row["id"] for row in resp.json()["items"]}
    assert str(world.org_a) in ids
    assert str(world.org_b) not in ids, "another tenant leaked into the listing"


async def test_a_platform_admin_keeps_the_global_view(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    """``system.administer`` is the one caller that stays unrestricted.

    Asserted alongside the narrowing test: a fix that simply returned nothing
    would pass that one and break the operator console.
    """
    token = await _bearer(engine, world.platform)
    resp = await client.get(
        "/api/v1/admin/organizations", params={"limit": 200}, headers=_auth(token)
    )
    assert resp.status_code == 200, resp.text

    ids = {row["id"] for row in resp.json()["items"]}
    assert {str(world.org_a), str(world.org_b)} <= ids


async def test_search_is_narrowed_the_same_way(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    """Search is a second listing route and needs the same guard.

    A narrowing applied only to the cursor list would leave the page-numbered
    table as an open door — and search is the easier one to probe with.
    """
    token = await _bearer(engine, world.master)
    resp = await client.get(
        "/api/v1/admin/organizations/search",
        params={"search": "Router Tenant"},
        headers=_auth(token),
    )
    assert resp.status_code == 200, resp.text
    ids = {row["id"] for row in resp.json()["items"]}
    assert ids == {str(world.org_a)}


async def test_a_caller_without_management_permission_is_refused(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.teacher)
    resp = await client.get("/api/v1/admin/organizations", headers=_auth(token))
    assert resp.status_code == 403, resp.text


async def test_an_anonymous_caller_is_refused(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/v1/admin/organizations")
    assert resp.status_code in {401, 403}, resp.text


# ---------------------------------------------------------------------------
# Per-resource org access — absent and foreign must look identical
# ---------------------------------------------------------------------------


async def test_reading_another_tenants_organization_is_refused(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    resp = await client.get(f"/api/v1/admin/organizations/{world.org_b}", headers=_auth(token))
    assert resp.status_code in {403, 404}, resp.text


@pytest.mark.parametrize(
    ("template", "foreign_attr"),
    [
        ("/api/v1/admin/org-units/{id}", "faculty_b"),
        ("/api/v1/admin/organization-memberships/{id}", "membership_b"),
    ],
)
async def test_child_routes_hide_foreign_and_absent_alike(
    client: httpx.AsyncClient,
    engine: AsyncEngine,
    world: World,
    template: str,
    foreign_attr: str,
) -> None:
    """The same answer for "never existed" and "belongs to someone else".

    These routes are addressed by the child's own id with no ``org_id`` in the
    path, so the owning organization is resolved first. If the two cases
    diverged — 404 for one, 403 for the other — an admin of tenant A could
    enumerate which ids are real in tenant B by watching the status code.

    Compared to each other rather than to a hardcoded number: what matters is
    that the two are INDISTINGUISHABLE, whichever code is chosen.
    """
    token = await _bearer(engine, world.master)
    absent = await client.get(template.format(id=uuid.uuid4()), headers=_auth(token))
    foreign = await client.get(
        template.format(id=getattr(world, foreign_attr)), headers=_auth(token)
    )
    assert absent.status_code == 404, absent.text
    assert foreign.status_code == absent.status_code, (
        f"foreign={foreign.status_code} absent={absent.status_code}: "
        "a resource in another tenant must look exactly like a missing one"
    )
    assert foreign.json() == absent.json(), "the bodies must not differ either"


# ---------------------------------------------------------------------------
# Organization CRUD + error mapping
# ---------------------------------------------------------------------------


async def test_organization_crud_round_trip(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.platform)
    headers = _auth(token)
    slug = f"rt-new-{world.tag}"

    created = await client.post(
        "/api/v1/admin/organizations",
        json={"slug": slug, "name": "Created Tenant", "status": "active"},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    org_id = created.json()["id"]

    fetched = await client.get(f"/api/v1/admin/organizations/{org_id}", headers=headers)
    assert fetched.status_code == 200, fetched.text
    assert fetched.json()["slug"] == slug

    patched = await client.patch(
        f"/api/v1/admin/organizations/{org_id}",
        json={"name": "Renamed Tenant"},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "Renamed Tenant"

    deleted = await client.delete(f"/api/v1/admin/organizations/{org_id}", headers=headers)
    assert deleted.status_code in {200, 204}, deleted.text

    gone = await client.get(f"/api/v1/admin/organizations/{org_id}", headers=headers)
    assert gone.status_code == 404, gone.text

    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM organizations WHERE id = CAST(:id AS uuid)"), {"id": org_id}
        )


async def test_a_duplicate_slug_is_422_not_500(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    """``AppError`` maps to 422 with the validation envelope.

    A business-rule refusal that escapes as a 500 tells the client nothing and
    pages whoever is on call.
    """
    token = await _bearer(engine, world.platform)
    existing = await client.get(f"/api/v1/admin/organizations/{world.org_a}", headers=_auth(token))
    slug = existing.json()["slug"]

    resp = await client.post(
        "/api/v1/admin/organizations",
        json={"slug": slug, "name": "Clash", "status": "active"},
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text
    assert resp.json()["detail"]["error"] == "validation"


async def test_patching_a_missing_organization_is_404(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.platform)
    resp = await client.patch(
        f"/api/v1/admin/organizations/{uuid.uuid4()}",
        json={"name": "Nope"},
        headers=_auth(token),
    )
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------


async def test_domain_endpoints_round_trip(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    headers = _auth(token)

    listed = await client.get(f"/api/v1/admin/organizations/{world.org_a}/domains", headers=headers)
    assert listed.status_code == 200, listed.text
    assert [d["id"] for d in listed.json()] == [str(world.domain_a)]

    created = await client.post(
        f"/api/v1/admin/organizations/{world.org_a}/domains",
        json={"domain": f"rt-second-{world.tag}.edu", "auto_provision": True},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    domain_id = created.json()["id"]

    patched = await client.patch(
        f"/api/v1/admin/organization-domains/{domain_id}",
        json={"auto_provision": False},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["auto_provision"] is False

    removed = await client.delete(
        f"/api/v1/admin/organization-domains/{domain_id}", headers=headers
    )
    assert removed.status_code in {200, 204}, removed.text


async def test_claiming_a_mapped_domain_is_422(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    resp = await client.post(
        f"/api/v1/admin/organizations/{world.org_a}/domains",
        json={"domain": f"rt-{world.tag}.edu", "auto_provision": False},
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------


async def test_unit_listing_and_tree(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    headers = _auth(token)

    listed = await client.get(f"/api/v1/admin/organizations/{world.org_a}/units", headers=headers)
    assert listed.status_code == 200, listed.text
    assert str(world.faculty_a) in {u["id"] for u in listed.json()}

    tree = await client.get(
        f"/api/v1/admin/organizations/{world.org_a}/units/tree", headers=headers
    )
    assert tree.status_code == 200, tree.text
    assert str(world.faculty_a) in {n["id"] for n in tree.json()}


async def test_creating_a_faculty_requires_a_master_dean(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    """``ForbiddenError`` from the service maps to 403.

    The platform admin holds ``system.administer`` so the dependency lets them
    in, but they are not a Master Dean of this tenant — the service refuses and
    the router must not turn that into a 500.
    """
    token = await _bearer(engine, world.platform)
    resp = await client.post(
        f"/api/v1/admin/organizations/{world.org_a}/units",
        json={"name": "Platform Faculty", "code": f"PF{world.tag[:4]}"},
        headers=_auth(token),
    )
    assert resp.status_code == 403, resp.text


async def test_a_master_dean_can_create_and_rename_a_faculty(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    headers = _auth(token)

    created = await client.post(
        f"/api/v1/admin/organizations/{world.org_a}/units",
        json={"name": "Dean Faculty", "code": f"DF{world.tag[:4]}"},
        headers=headers,
    )
    assert created.status_code == 201, created.text
    unit_id = created.json()["id"]

    patched = await client.patch(
        f"/api/v1/admin/org-units/{unit_id}",
        json={"name": "Dean Faculty Renamed"},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["name"] == "Dean Faculty Renamed"

    removed = await client.delete(f"/api/v1/admin/org-units/{unit_id}", headers=headers)
    assert removed.status_code in {200, 204}, removed.text


# ---------------------------------------------------------------------------
# Bulk assign
# ---------------------------------------------------------------------------


async def test_bulk_assign_reports_assigned_and_skipped(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    stale = str(uuid.uuid4())
    resp = await client.post(
        f"/api/v1/admin/organizations/{world.org_a}/memberships/assign-unit",
        json={
            "membership_ids": [str(world.membership_a), stale],
            "org_unit_id": str(world.faculty_a),
        },
        headers=_auth(token),
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["assigned"] == 1
    assert body["skipped"] == [stale]


async def test_bulk_assign_rejects_a_foreign_membership_with_422(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    resp = await client.post(
        f"/api/v1/admin/organizations/{world.org_a}/memberships/assign-unit",
        json={
            "membership_ids": [str(world.membership_a), str(world.membership_b)],
            "org_unit_id": str(world.faculty_a),
        },
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# Faculty staffing
# ---------------------------------------------------------------------------


async def test_faculty_assignment_endpoints(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    headers = _auth(token)

    added = await client.post(
        f"/api/v1/admin/faculties/{world.faculty_a}/members",
        json={"user_ids": [str(world.teacher)]},
        headers=headers,
    )
    assert added.status_code in {200, 201}, added.text

    listed = await client.get(
        f"/api/v1/admin/organizations/{world.org_a}/faculty-assignments",
        headers=headers,
    )
    assert listed.status_code == 200, listed.text
    assert str(world.teacher) in {row["user_id"] for row in listed.json()}

    removed = await client.delete(
        f"/api/v1/admin/faculties/{world.faculty_a}/members/{world.teacher}",
        headers=headers,
    )
    assert removed.status_code in {200, 204}, removed.text


async def test_adding_yourself_to_a_faculty_is_403(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    resp = await client.post(
        f"/api/v1/admin/faculties/{world.faculty_a}/members",
        json={"user_ids": [str(world.master)]},
        headers=_auth(token),
    )
    assert resp.status_code == 403, resp.text


async def test_staffing_another_tenants_faculty_is_refused(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    resp = await client.post(
        f"/api/v1/admin/faculties/{world.faculty_b}/members",
        json={"user_ids": [str(world.teacher)]},
        headers=_auth(token),
    )
    assert resp.status_code in {403, 404}, resp.text


# ---------------------------------------------------------------------------
# Memberships
# ---------------------------------------------------------------------------


async def test_membership_patch_and_delete(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    token = await _bearer(engine, world.master)
    headers = _auth(token)

    patched = await client.patch(
        f"/api/v1/admin/organization-memberships/{world.membership_a}",
        json={"status": "suspended"},
        headers=headers,
    )
    assert patched.status_code == 200, patched.text
    assert patched.json()["status"] == "suspended"

    removed = await client.delete(
        f"/api/v1/admin/organization-memberships/{world.membership_a}", headers=headers
    )
    assert removed.status_code in {200, 204}, removed.text

    # Leaving is not deleting: the row survives with status 'left'.
    async with engine.begin() as conn:
        row = (
            (
                await conn.execute(
                    text("SELECT status, deleted_at FROM organization_memberships WHERE id = :id"),
                    {"id": world.membership_a},
                )
            )
            .mappings()
            .one()
        )
    assert row["status"] == "left"
    assert row["deleted_at"] is None


async def test_membership_patch_rejects_an_unknown_status(
    client: httpx.AsyncClient, engine: AsyncEngine, world: World
) -> None:
    """FastAPI validates the literal before the service is reached."""
    token = await _bearer(engine, world.master)
    resp = await client.patch(
        f"/api/v1/admin/organization-memberships/{world.membership_a}",
        json={"status": "left"},
        headers=_auth(token),
    )
    assert resp.status_code == 422, resp.text
