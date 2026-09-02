"""Coverage for ``access_control/services/organizations.py`` and the query
module underneath it.

This is tenant administration: organizations, their email domains, their
faculties, who staffs those faculties, and who belongs to the tenant. Almost
every rule in it is a containment rule, and the failure mode when one breaks
is not an error — it is one tenant's data quietly becoming reachable from
another, or a person keeping access they were supposed to lose.

The rules worth stating, because they are what these tests defend:

* **Two kinds of Dean.** A *Master Dean* holds ``hod`` at ORGANIZATION scope
  and governs the whole tenant. A *Faculty Dean* holds ``hod`` at ORG_UNIT
  scope AND an active ``user_faculty_assignments`` row for that faculty —
  both halves, so a stale role row alone grants nothing. A Faculty Dean may
  staff their own faculty but may not appoint or remove another Dean.
* **Bulk assignment is all-or-nothing across tenants but tolerant of
  staleness.** A membership id from another organization rejects the whole
  batch; an id that has simply vanished is reported in ``skipped``.
* **A faculty is archived only when nothing lives in it.** Programs, courses
  and active staff assignments each block it.
* **Leaving is not deleting.** ``delete_membership`` sets ``status='left'``
  with a timestamp, because every org-scope check filters on ``active`` and
  the history is worth keeping.

Isolation: the shared Postgres means every test builds its own organization
and tears the whole graph down; nothing here asserts a global count.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from abridgeai.core.config import get_settings
from abridgeai.core.exceptions import AppError, ForbiddenError, NotFoundError
from abridgeai.features.access_control.queries import organizations as org_queries
from abridgeai.features.access_control.schemas.admin import (
    BulkAssignUnitRequest,
    FacultyMembersAddRequest,
    MembershipPatch,
    OrganizationCreate,
    OrganizationDomainCreate,
    OrganizationDomainPatch,
    OrganizationPatch,
    OrgUnitCreate,
    OrgUnitPatch,
)
from abridgeai.features.access_control.services import organizations as svc


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
async def db(engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session


class Tenant:
    """One organization with two faculties and a cast of staff.

    ``master`` holds org-scoped ``hod``; ``dean`` holds org_unit-scoped ``hod``
    on ``faculty_a`` plus the matching faculty assignment; ``teacher`` and
    ``manager`` are assignable staff; ``student`` is deliberately not.
    """

    def __init__(self) -> None:
        self.tag = uuid.uuid4().hex[:10]
        self.org_id = uuid.uuid4()
        self.other_org_id = uuid.uuid4()
        self.faculty_a = uuid.uuid4()
        self.faculty_b = uuid.uuid4()
        self.foreign_faculty = uuid.uuid4()
        self.master = uuid.uuid4()
        self.dean = uuid.uuid4()
        self.teacher = uuid.uuid4()
        self.manager = uuid.uuid4()
        self.student = uuid.uuid4()
        self.outsider = uuid.uuid4()
        self.membership_ids: dict[uuid.UUID, uuid.UUID] = {}
        self.foreign_membership_id = uuid.uuid4()


@pytest_asyncio.fixture
async def tenant(engine: AsyncEngine) -> AsyncIterator[Tenant]:
    t = Tenant()
    users = {
        "master": t.master,
        "dean": t.dean,
        "teacher": t.teacher,
        "manager": t.manager,
        "student": t.student,
        "outsider": t.outsider,
    }
    async with engine.begin() as conn:
        for org, name in ((t.org_id, "Org Admin Tenant"), (t.other_org_id, "Org Admin Other")):
            await conn.execute(
                text(
                    "INSERT INTO organizations (id, slug, name, status) "
                    "VALUES (:id, :slug, :name, 'active')"
                ),
                {"id": org, "slug": f"oadm-{org.hex[:10]}", "name": name},
            )
        for label, uid in users.items():
            await conn.execute(
                text("INSERT INTO users (id, primary_email) VALUES (:id, :email)"),
                {"id": uid, "email": f"oadm-{t.tag}-{label}@test.local"},
            )
        # 0094_flat_faculties: every LIVE org_unit must be a top-level faculty.
        for fid, org, name in (
            (t.faculty_a, t.org_id, "Faculty A"),
            (t.faculty_b, t.org_id, "Faculty B"),
            (t.foreign_faculty, t.other_org_id, "Foreign Faculty"),
        ):
            await conn.execute(
                text(
                    "INSERT INTO org_units "
                    "(id, organization_id, unit_type, name, code) "
                    "VALUES (:id, :org, 'faculty', :name, :code)"
                ),
                {"id": fid, "org": org, "name": name, "code": f"F-{fid.hex[:6]}"},
            )
        # Memberships: everyone but the outsider belongs to the tenant.
        for label, uid in users.items():
            if label == "outsider":
                continue
            mid = uuid.uuid4()
            t.membership_ids[uid] = mid
            await conn.execute(
                text(
                    "INSERT INTO organization_memberships "
                    "(id, user_id, organization_id, status) "
                    "VALUES (:id, :uid, :org, 'active')"
                ),
                {"id": mid, "uid": uid, "org": t.org_id},
            )
        await conn.execute(
            text(
                "INSERT INTO organization_memberships "
                "(id, user_id, organization_id, status) "
                "VALUES (:id, :uid, :org, 'active')"
            ),
            {"id": t.foreign_membership_id, "uid": t.outsider, "org": t.other_org_id},
        )
        # Roles. Master Dean = org-scoped hod. Faculty Dean = org_unit-scoped
        # hod, which only counts alongside the faculty assignment added below.
        for uid, role, scope, unit in (
            (t.master, "hod", "organization", None),
            (t.dean, "hod", "org_unit", t.faculty_a),
            (t.teacher, "teacher", "organization", None),
            (t.manager, "manager", "organization", None),
            (t.student, "student", "organization", None),
        ):
            await conn.execute(
                text(
                    "INSERT INTO user_role_assignments "
                    "(user_id, role_id, scope_kind, organization_id, org_unit_id) "
                    "SELECT :uid, r.id, :scope, :org, :unit FROM roles r WHERE r.code = :role"
                ),
                {"uid": uid, "scope": scope, "org": t.org_id, "unit": unit, "role": role},
            )
        await conn.execute(
            text(
                "INSERT INTO user_faculty_assignments "
                "(id, user_id, organization_id, faculty_id, status) "
                "VALUES (gen_random_uuid(), :uid, :org, :faculty, 'active')"
            ),
            {"uid": t.dean, "org": t.org_id, "faculty": t.faculty_a},
        )
    yield t
    async with engine.begin() as conn:
        for stmt in (
            "DELETE FROM user_faculty_assignments WHERE organization_id = ANY(:orgs)",
            "DELETE FROM user_role_assignments WHERE user_id = ANY(:users)",
            "DELETE FROM organization_memberships WHERE organization_id = ANY(:orgs)",
            "DELETE FROM organization_domains WHERE organization_id = ANY(:orgs)",
            "DELETE FROM courses WHERE organization_id = ANY(:orgs)",
            "DELETE FROM org_units WHERE organization_id = ANY(:orgs)",
            "DELETE FROM users WHERE id = ANY(:users)",
            "DELETE FROM organizations WHERE id = ANY(:orgs)",
        ):
            await conn.execute(
                text(stmt),
                {"orgs": [t.org_id, t.other_org_id], "users": list(users.values())},
            )


# ---------------------------------------------------------------------------
# Tenancy resolvers — what lets a router scope a request addressed by child id
# ---------------------------------------------------------------------------


async def test_resolvers_find_the_owning_organization(db: AsyncSession, tenant: Tenant) -> None:
    """Domains, units and memberships are addressed by their OWN id.

    The permission dependency has no ``org_id`` in the path to scope against,
    so these resolvers are what let the router call ``require_org_access``
    before touching anything.
    """
    domain = await svc.create_domain(
        db,
        tenant.org_id,
        OrganizationDomainCreate(domain=f"res-{tenant.tag}.edu", auto_provision=False),
    )
    await db.commit()

    assert await svc.organization_id_for_domain(db, domain.id) == tenant.org_id
    assert await svc.organization_id_for_unit(db, tenant.faculty_a) == tenant.org_id
    membership_id = tenant.membership_ids[tenant.teacher]
    assert await svc.organization_id_for_membership(db, membership_id) == tenant.org_id


async def test_resolvers_return_none_for_unknown_ids(db: AsyncSession) -> None:
    """None means "does not exist", which the router renders as the SAME 404 a
    foreign-tenant caller gets — the two must be indistinguishable."""
    missing = uuid.uuid4()
    assert await svc.organization_id_for_domain(db, missing) is None
    assert await svc.organization_id_for_unit(db, missing) is None
    assert await svc.organization_id_for_membership(db, missing) is None


async def test_organization_ids_for_user(db: AsyncSession, tenant: Tenant) -> None:
    ids = await svc.organization_ids_for_user(db, tenant.teacher)
    assert tenant.org_id in ids
    assert tenant.other_org_id not in ids


# ---------------------------------------------------------------------------
# Organizations
# ---------------------------------------------------------------------------


async def test_get_organization_hides_a_soft_deleted_tenant(
    db: AsyncSession, tenant: Tenant
) -> None:
    org = await svc.get_organization(db, tenant.org_id)
    assert org.id == tenant.org_id

    await svc.delete_organization(db, tenant.org_id, actor_id=tenant.master)
    await db.commit()
    with pytest.raises(NotFoundError):
        await svc.get_organization(db, tenant.org_id)


async def test_delete_of_an_unknown_organization_is_not_found(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await svc.delete_organization(db, uuid.uuid4(), actor_id=None)


async def test_slug_uniqueness_is_enforced_on_create_and_patch(
    db: AsyncSession, tenant: Tenant
) -> None:
    """The slug is how a tenant is addressed, so a duplicate is a real clash.

    Both paths matter: creating onto an existing slug, and renaming one
    tenant onto another's.
    """
    taken = (await svc.get_organization(db, tenant.org_id)).slug
    with pytest.raises(AppError, match="already exists"):
        await svc.create_organization(
            db, OrganizationCreate(slug=taken, name="Impostor", status="active")
        )

    with pytest.raises(AppError, match="already exists"):
        await svc.patch_organization(db, tenant.other_org_id, OrganizationPatch(slug=taken))


async def test_patching_an_organization_to_its_own_slug_is_allowed(
    db: AsyncSession, tenant: Tenant
) -> None:
    """Renaming without changing the slug must not collide with itself.

    The clash check has to exempt the row being edited, or every save of an
    unrelated field fails once the slug is set.
    """
    current = await svc.get_organization(db, tenant.org_id)
    updated = await svc.patch_organization(
        db, tenant.org_id, OrganizationPatch(slug=current.slug, name="Renamed")
    )
    await db.commit()
    assert updated.name == "Renamed"
    assert updated.slug == current.slug


async def test_patch_of_an_unknown_organization_is_not_found(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await svc.patch_organization(db, uuid.uuid4(), OrganizationPatch(name="x"))


# ---------------------------------------------------------------------------
# Listing / search / visibility
# ---------------------------------------------------------------------------


async def test_list_pages_by_cursor_without_repeating_a_row(
    db: AsyncSession, tenant: Tenant
) -> None:
    """Ordered by ``(name, id)`` and paged on both.

    Restricted to this test's own two tenants so the assertion does not
    depend on what else is in the shared database.
    """
    visible = [tenant.org_id, tenant.other_org_id]
    first = await svc.list_organizations(db, limit=1, visible_to_ids=visible)
    assert len(first.items) == 1
    assert first.next_cursor is not None

    second = await svc.list_organizations(
        db, limit=1, cursor=first.next_cursor, visible_to_ids=visible
    )
    assert len(second.items) == 1
    assert second.items[0].id != first.items[0].id

    full = await svc.list_organizations(db, limit=50, visible_to_ids=visible)
    assert full.next_cursor is None, "a partial page cannot have more behind it"
    assert {o.id for o in full.items} == set(visible)


async def test_list_rejects_a_malformed_cursor(db: AsyncSession) -> None:
    with pytest.raises(ValueError, match="cursor"):
        await svc.list_organizations(db, cursor="not-a-real-cursor")


async def test_visibility_restricts_what_a_scoped_caller_sees(
    db: AsyncSession, tenant: Tenant
) -> None:
    """``visible_to_ids=None`` is unrestricted and reserved for system admins.

    Every other caller passes the set it belongs to, because the permission
    behind the route is flat and cannot tell an ``org_unit.manage`` granted
    in one tenant from a global one.
    """
    page = await svc.search_organizations(db, visible_to_ids=[tenant.org_id])
    assert [o.id for o in page.items] == [tenant.org_id]
    assert page.total == 1


async def test_search_matches_name_and_slug(db: AsyncSession, tenant: Tenant) -> None:
    page = await svc.search_organizations(
        db, search="Org Admin Tenant", visible_to_ids=[tenant.org_id, tenant.other_org_id]
    )
    assert [o.id for o in page.items] == [tenant.org_id]


async def test_inactivity_filter_narrows_and_never_widens(db: AsyncSession, tenant: Tenant) -> None:
    """``inactive_days`` INTERSECTS with the visibility set.

    The dangerous implementation replaces ``visible_to_ids`` with the
    inactive ids, which would show a scoped caller other tenants purely
    because those tenants are quiet. Asserted by asking for inactive tenants
    while visible to only one: the answer can never contain the other.
    """
    page = await svc.search_organizations(db, inactive_days=1, visible_to_ids=[tenant.org_id])
    assert tenant.other_org_id not in {o.id for o in page.items}


async def test_an_empty_intersection_pages_to_zero_rows(db: AsyncSession, tenant: Tenant) -> None:
    """An empty allowlist must mean "nothing", not "unrestricted".

    Passing an empty list down to the query layer would be read as None —
    the exact bug that turns a filter into a full-table leak.
    """
    page = await svc.search_organizations(
        db,
        inactive_days=100000,  # no tenant has been quiet that long
        visible_to_ids=[tenant.org_id],
    )
    assert page.items == []
    assert page.total == 0
    assert page.total_pages == 0


async def test_count_organizations_is_reachable(db: AsyncSession, tenant: Tenant) -> None:
    del tenant
    assert await svc.count_organizations(db) >= 1


# ---------------------------------------------------------------------------
# Domains
# ---------------------------------------------------------------------------


async def test_domain_lifecycle(db: AsyncSession, tenant: Tenant) -> None:
    created = await svc.create_domain(
        db,
        tenant.org_id,
        OrganizationDomainCreate(domain=f"life-{tenant.tag}.edu", auto_provision=True),
    )
    await db.commit()
    assert created.auto_provision is True
    assert [d.id for d in await svc.list_domains(db, tenant.org_id)] == [created.id]

    patched = await svc.patch_domain(db, created.id, OrganizationDomainPatch(auto_provision=False))
    await db.commit()
    assert patched.auto_provision is False

    await svc.delete_domain(db, created.id, actor_id=tenant.master)
    await db.commit()
    assert await svc.list_domains(db, tenant.org_id) == []


async def test_a_domain_can_only_be_mapped_once_anywhere(db: AsyncSession, tenant: Tenant) -> None:
    """Uniqueness is GLOBAL, not per tenant.

    The domain is what auto-provisioning matches a new sign-in against, so
    two tenants claiming one domain would make "which organization does this
    person belong to?" ambiguous at login.
    """
    name = f"shared-{tenant.tag}.edu"
    await svc.create_domain(
        db, tenant.org_id, OrganizationDomainCreate(domain=name, auto_provision=False)
    )
    await db.commit()

    with pytest.raises(AppError, match="already mapped"):
        await svc.create_domain(
            db,
            tenant.other_org_id,
            OrganizationDomainCreate(domain=name, auto_provision=False),
        )


async def test_renaming_a_domain_onto_another_is_rejected(db: AsyncSession, tenant: Tenant) -> None:
    first = await svc.create_domain(
        db,
        tenant.org_id,
        OrganizationDomainCreate(domain=f"one-{tenant.tag}.edu", auto_provision=False),
    )
    await svc.create_domain(
        db,
        tenant.org_id,
        OrganizationDomainCreate(domain=f"two-{tenant.tag}.edu", auto_provision=False),
    )
    await db.commit()

    with pytest.raises(AppError, match="already mapped"):
        await svc.patch_domain(
            db, first.id, OrganizationDomainPatch(domain=f"two-{tenant.tag}.edu")
        )


async def test_domain_operations_on_unknown_ids_are_not_found(
    db: AsyncSession, tenant: Tenant
) -> None:
    del tenant
    missing = uuid.uuid4()
    with pytest.raises(NotFoundError):
        await svc.patch_domain(db, missing, OrganizationDomainPatch(auto_provision=True))
    with pytest.raises(NotFoundError):
        await svc.delete_domain(db, missing, actor_id=None)


async def test_domain_create_requires_a_live_organization(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await svc.create_domain(
            db, uuid.uuid4(), OrganizationDomainCreate(domain="x.edu", auto_provision=False)
        )


# ---------------------------------------------------------------------------
# Faculties (org units)
# ---------------------------------------------------------------------------


async def test_only_a_master_dean_may_create_a_faculty(db: AsyncSession, tenant: Tenant) -> None:
    """Org-scoped ``hod`` governs the tenant; a Faculty Dean does not.

    A Faculty Dean creating faculties would let them expand their own remit,
    which is the whole point of separating the two levels.
    """
    with pytest.raises(ForbiddenError, match="Master Dean"):
        await svc.create_unit(
            db,
            tenant.org_id,
            OrgUnitCreate(name="Sneaky Faculty", code="SNK"),
            actor_id=tenant.dean,
        )

    created = await svc.create_unit(
        db,
        tenant.org_id,
        OrgUnitCreate(name="Legit Faculty", code=f"LGT{tenant.tag[:4]}"),
        actor_id=tenant.master,
    )
    await db.commit()
    assert created.unit_type == "faculty"
    assert created.parent_unit_id is None, "0094 keeps every live unit top-level"


async def test_only_a_master_dean_may_edit_or_archive_a_faculty(
    db: AsyncSession, tenant: Tenant
) -> None:
    with pytest.raises(ForbiddenError, match="Master Dean"):
        await svc.patch_unit(
            db, tenant.faculty_a, OrgUnitPatch(name="Renamed"), actor_id=tenant.dean
        )
    with pytest.raises(ForbiddenError, match="Master Dean"):
        await svc.delete_unit(db, tenant.faculty_a, actor_id=tenant.dean)


async def test_a_master_dean_can_rename_a_faculty(db: AsyncSession, tenant: Tenant) -> None:
    updated = await svc.patch_unit(
        db, tenant.faculty_b, OrgUnitPatch(name="Faculty B Renamed"), actor_id=tenant.master
    )
    await db.commit()
    assert updated.name == "Faculty B Renamed"


async def test_a_faculty_with_live_dependencies_cannot_be_archived(
    db: AsyncSession, engine: AsyncEngine, tenant: Tenant
) -> None:
    """Courses, programs and active staff each hold a faculty open.

    Archiving underneath them would leave rows pointing at a dead faculty,
    and every scope filter built on ``faculty_id`` would start returning
    nothing for them.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO courses "
                "(id, organization_id, faculty_id, owner_user_id, slug, title) "
                "VALUES (gen_random_uuid(), :org, :faculty, :owner, :slug, 'Held open')"
            ),
            {
                "org": tenant.org_id,
                "faculty": tenant.faculty_b,
                "owner": tenant.master,
                "slug": f"hold-{tenant.tag}",
            },
        )

    with pytest.raises(AppError, match="faculty_has_dependencies"):
        await svc.delete_unit(db, tenant.faculty_b, actor_id=tenant.master)


async def test_an_active_staff_assignment_also_holds_a_faculty_open(
    db: AsyncSession, tenant: Tenant
) -> None:
    """faculty_a has the Dean assigned to it by the fixture."""
    with pytest.raises(AppError, match="faculty_has_dependencies"):
        await svc.delete_unit(db, tenant.faculty_a, actor_id=tenant.master)


async def test_an_empty_faculty_archives_cleanly(db: AsyncSession, tenant: Tenant) -> None:
    await svc.delete_unit(db, tenant.faculty_b, actor_id=tenant.master)
    await db.commit()
    with pytest.raises(NotFoundError):
        await svc.get_unit(db, tenant.faculty_b)


async def test_unit_lookups_of_unknown_ids_are_not_found(db: AsyncSession) -> None:
    with pytest.raises(NotFoundError):
        await svc.get_unit(db, uuid.uuid4())


async def test_list_units_returns_the_tenants_faculties(db: AsyncSession, tenant: Tenant) -> None:
    ids = {u.id for u in await svc.list_units(db, tenant.org_id)}
    assert {tenant.faculty_a, tenant.faculty_b} <= ids
    assert tenant.foreign_faculty not in ids


async def test_unit_tree_is_returned_for_the_tenant(db: AsyncSession, tenant: Tenant) -> None:
    nodes = await svc.list_unit_tree(db, tenant.org_id)
    ids = {n.id for n in nodes}
    assert {tenant.faculty_a, tenant.faculty_b} <= ids
    # 0094 makes every live unit a root, so the tree is one level deep.
    assert all(not n.children for n in nodes)


# ---------------------------------------------------------------------------
# Bulk membership assignment
# ---------------------------------------------------------------------------


async def test_bulk_assign_moves_the_named_memberships(db: AsyncSession, tenant: Tenant) -> None:
    ids = [tenant.membership_ids[tenant.teacher], tenant.membership_ids[tenant.manager]]
    result = await svc.assign_memberships_to_unit(
        db,
        tenant.org_id,
        BulkAssignUnitRequest(membership_ids=ids, org_unit_id=tenant.faculty_a),
    )
    await db.commit()
    assert result.assigned == 2
    assert result.skipped == []

    row = await org_queries.get_membership(db, ids[0])
    assert row is not None
    assert row.org_unit_id == tenant.faculty_a


async def test_bulk_assign_rejects_the_whole_batch_on_a_foreign_id(
    db: AsyncSession, tenant: Tenant
) -> None:
    """One membership from another tenant fails everything.

    Ids arrive in a request body. Skipping the foreign one quietly would file
    that person under this tenant's faculty — or, read the other way, tell
    the caller a cohort moved when it partly did not. A partial write nobody
    is told about is worse than an error.
    """
    mine = tenant.membership_ids[tenant.teacher]
    with pytest.raises(AppError, match="different organization"):
        await svc.assign_memberships_to_unit(
            db,
            tenant.org_id,
            BulkAssignUnitRequest(
                membership_ids=[mine, tenant.foreign_membership_id],
                org_unit_id=tenant.faculty_a,
            ),
        )
    await db.rollback()

    row = await org_queries.get_membership(db, mine)
    assert row is not None
    assert row.org_unit_id is None, "nothing may be written when the batch is rejected"


async def test_bulk_assign_tolerates_a_stale_selection(db: AsyncSession, tenant: Tenant) -> None:
    """A membership removed mid-flow is reported, not fatal.

    That is a stale UI selection, not an attack — failing the batch because
    one person left while the manager was choosing would be hostile.
    """
    gone = uuid.uuid4()
    result = await svc.assign_memberships_to_unit(
        db,
        tenant.org_id,
        BulkAssignUnitRequest(
            membership_ids=[tenant.membership_ids[tenant.teacher], gone],
            org_unit_id=tenant.faculty_a,
        ),
    )
    await db.commit()
    assert result.assigned == 1
    assert result.skipped == [gone]


async def test_bulk_assign_rejects_a_unit_from_another_tenant(
    db: AsyncSession, tenant: Tenant
) -> None:
    with pytest.raises(AppError, match="different organization"):
        await svc.assign_memberships_to_unit(
            db,
            tenant.org_id,
            BulkAssignUnitRequest(
                membership_ids=[tenant.membership_ids[tenant.teacher]],
                org_unit_id=tenant.foreign_faculty,
            ),
        )


async def test_bulk_assign_rejects_a_missing_unit(db: AsyncSession, tenant: Tenant) -> None:
    with pytest.raises(AppError, match="does not exist"):
        await svc.assign_memberships_to_unit(
            db,
            tenant.org_id,
            BulkAssignUnitRequest(
                membership_ids=[tenant.membership_ids[tenant.teacher]],
                org_unit_id=uuid.uuid4(),
            ),
        )


async def test_bulk_assign_can_clear_the_unit(db: AsyncSession, tenant: Tenant) -> None:
    """``org_unit_id=None`` unfiles people, and must skip the unit checks."""
    ids = [tenant.membership_ids[tenant.teacher]]
    await svc.assign_memberships_to_unit(
        db, tenant.org_id, BulkAssignUnitRequest(membership_ids=ids, org_unit_id=tenant.faculty_a)
    )
    await db.commit()

    result = await svc.assign_memberships_to_unit(
        db, tenant.org_id, BulkAssignUnitRequest(membership_ids=ids, org_unit_id=None)
    )
    await db.commit()
    assert result.assigned == 1
    row = await org_queries.get_membership(db, ids[0])
    assert row is not None
    assert row.org_unit_id is None


# ---------------------------------------------------------------------------
# Faculty staffing
# ---------------------------------------------------------------------------


async def test_a_faculty_dean_may_staff_their_own_faculty(db: AsyncSession, tenant: Tenant) -> None:
    rows = await svc.add_faculty_members(
        db,
        tenant.faculty_a,
        FacultyMembersAddRequest(user_ids=[tenant.teacher]),
        actor_id=tenant.dean,
    )
    await db.commit()
    assert [r.user_id for r in rows] == [tenant.teacher]


async def test_a_faculty_dean_may_not_staff_a_faculty_they_do_not_hold(
    db: AsyncSession, tenant: Tenant
) -> None:
    """The role row alone is not authority — the faculty assignment is.

    The Dean holds org_unit ``hod`` on faculty_a only, so faculty_b must be
    closed to them even though the role code is identical.
    """
    with pytest.raises(ForbiddenError, match="Faculty Dean"):
        await svc.add_faculty_members(
            db,
            tenant.faculty_b,
            FacultyMembersAddRequest(user_ids=[tenant.teacher]),
            actor_id=tenant.dean,
        )


async def test_nobody_may_add_themselves(db: AsyncSession, tenant: Tenant) -> None:
    with pytest.raises(ForbiddenError, match="yourself"):
        await svc.add_faculty_members(
            db,
            tenant.faculty_a,
            FacultyMembersAddRequest(user_ids=[tenant.master]),
            actor_id=tenant.master,
        )


async def test_only_active_members_of_the_tenant_are_assignable(
    db: AsyncSession, tenant: Tenant
) -> None:
    """Someone outside the organization cannot be staffed into its faculty."""
    with pytest.raises(AppError, match="active members"):
        await svc.add_faculty_members(
            db,
            tenant.faculty_a,
            FacultyMembersAddRequest(user_ids=[tenant.outsider]),
            actor_id=tenant.master,
        )


async def test_students_are_not_faculty_staff(db: AsyncSession, tenant: Tenant) -> None:
    """Eligibility is by role: hod / manager / teacher only.

    A student assigned to a faculty would pick up staff-side scope filters
    built on ``user_faculty_assignments``.
    """
    with pytest.raises(AppError, match="not eligible faculty staff"):
        await svc.add_faculty_members(
            db,
            tenant.faculty_a,
            FacultyMembersAddRequest(user_ids=[tenant.student]),
            actor_id=tenant.master,
        )


async def test_a_faculty_dean_cannot_appoint_another_dean(
    db: AsyncSession, engine: AsyncEngine, tenant: Tenant
) -> None:
    """Only the Master Dean appoints Deans.

    Otherwise a Faculty Dean can grow the set of people holding their own
    level of authority, which the two-tier split exists to prevent.
    """
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "INSERT INTO user_role_assignments "
                "(user_id, role_id, scope_kind, organization_id) "
                "SELECT :uid, r.id, 'organization', :org FROM roles r WHERE r.code = 'hod'"
            ),
            {"uid": tenant.teacher, "org": tenant.org_id},
        )

    with pytest.raises(ForbiddenError, match="another Faculty Dean"):
        await svc.add_faculty_members(
            db,
            tenant.faculty_a,
            FacultyMembersAddRequest(user_ids=[tenant.teacher]),
            actor_id=tenant.dean,
        )


async def test_adding_an_existing_member_is_idempotent(db: AsyncSession, tenant: Tenant) -> None:
    """Re-adding returns the existing assignment rather than a second row.

    A duplicate would violate the partial unique index on live assignments;
    returning the existing one makes the call safe to retry.
    """
    first = await svc.add_faculty_members(
        db,
        tenant.faculty_a,
        FacultyMembersAddRequest(user_ids=[tenant.teacher]),
        actor_id=tenant.master,
    )
    await db.commit()
    again = await svc.add_faculty_members(
        db,
        tenant.faculty_a,
        FacultyMembersAddRequest(user_ids=[tenant.teacher]),
        actor_id=tenant.master,
    )
    await db.commit()
    assert [r.id for r in again] == [r.id for r in first]


async def test_add_requires_a_live_top_level_faculty(db: AsyncSession, tenant: Tenant) -> None:
    del tenant
    with pytest.raises(NotFoundError):
        await svc.add_faculty_members(
            db,
            uuid.uuid4(),
            FacultyMembersAddRequest(user_ids=[uuid.uuid4()]),
            actor_id=uuid.uuid4(),
        )


async def test_removal_mirrors_the_appointment_rules(db: AsyncSession, tenant: Tenant) -> None:
    await svc.add_faculty_members(
        db,
        tenant.faculty_a,
        FacultyMembersAddRequest(user_ids=[tenant.teacher]),
        actor_id=tenant.master,
    )
    await db.commit()

    with pytest.raises(ForbiddenError, match="yourself"):
        await svc.remove_faculty_member(db, tenant.faculty_a, tenant.dean, actor_id=tenant.dean)

    await svc.remove_faculty_member(db, tenant.faculty_a, tenant.teacher, actor_id=tenant.master)
    await db.commit()
    assert (
        await org_queries.get_active_faculty_assignment(
            db, user_id=tenant.teacher, faculty_id=tenant.faculty_a
        )
        is None
    )


async def test_removing_someone_who_is_not_assigned_is_not_found(
    db: AsyncSession, tenant: Tenant
) -> None:
    with pytest.raises(NotFoundError):
        await svc.remove_faculty_member(
            db, tenant.faculty_a, tenant.manager, actor_id=tenant.master
        )


async def test_an_unauthorized_actor_cannot_remove_staff(db: AsyncSession, tenant: Tenant) -> None:
    with pytest.raises(ForbiddenError, match="Faculty Dean"):
        await svc.remove_faculty_member(db, tenant.faculty_a, tenant.dean, actor_id=tenant.teacher)


# ---------------------------------------------------------------------------
# Faculty assignment visibility
# ---------------------------------------------------------------------------


async def test_a_master_dean_sees_every_faculty(db: AsyncSession, tenant: Tenant) -> None:
    await svc.add_faculty_members(
        db,
        tenant.faculty_b,
        FacultyMembersAddRequest(user_ids=[tenant.teacher]),
        actor_id=tenant.master,
    )
    await db.commit()

    rows = await svc.list_faculty_assignments(db, tenant.org_id, actor_id=tenant.master)
    faculties = {r.faculty_id for r in rows}
    assert {tenant.faculty_a, tenant.faculty_b} <= faculties


async def test_a_faculty_dean_sees_only_their_own_faculties(
    db: AsyncSession, tenant: Tenant
) -> None:
    """The list is narrowed to the actor's faculties, not the whole tenant."""
    await svc.add_faculty_members(
        db,
        tenant.faculty_b,
        FacultyMembersAddRequest(user_ids=[tenant.teacher]),
        actor_id=tenant.master,
    )
    await db.commit()

    rows = await svc.list_faculty_assignments(db, tenant.org_id, actor_id=tenant.dean)
    assert {r.faculty_id for r in rows} == {tenant.faculty_a}


async def test_a_faculty_dean_asking_for_another_faculty_is_forbidden(
    db: AsyncSession, tenant: Tenant
) -> None:
    with pytest.raises(ForbiddenError, match="assigned faculties"):
        await svc.list_faculty_assignments(
            db, tenant.org_id, faculty_id=tenant.faculty_b, actor_id=tenant.dean
        )


async def test_a_system_admin_bypasses_the_faculty_narrowing(
    db: AsyncSession, tenant: Tenant
) -> None:
    """``allow_system_admin`` is the platform operator's escape hatch.

    They hold no role inside the tenant at all, so without it a global admin
    would see an empty roster.
    """
    rows = await svc.list_faculty_assignments(
        db, tenant.org_id, actor_id=tenant.outsider, allow_system_admin=True
    )
    assert {r.faculty_id for r in rows} == {tenant.faculty_a}


async def test_listing_reports_each_members_role_codes(db: AsyncSession, tenant: Tenant) -> None:
    """The roster shows what someone IS, not just that they are attached."""
    rows = await svc.list_faculty_assignments(
        db, tenant.org_id, faculty_id=tenant.faculty_a, actor_id=tenant.master
    )
    dean_row = next(r for r in rows if r.user_id == tenant.dean)
    assert "hod" in dean_row.role_codes


async def test_listing_rejects_a_faculty_from_another_tenant(
    db: AsyncSession, tenant: Tenant
) -> None:
    """Reported as not-found: whether another tenant has that faculty is not
    this caller's to learn."""
    with pytest.raises(NotFoundError, match="Faculty not found"):
        await svc.list_faculty_assignments(
            db, tenant.org_id, faculty_id=tenant.foreign_faculty, actor_id=tenant.master
        )


# ---------------------------------------------------------------------------
# Memberships
# ---------------------------------------------------------------------------


async def test_patch_membership_validates_the_target_unit(db: AsyncSession, tenant: Tenant) -> None:
    membership_id = tenant.membership_ids[tenant.teacher]
    with pytest.raises(AppError, match="does not exist"):
        await svc.patch_membership(db, membership_id, MembershipPatch(org_unit_id=uuid.uuid4()))
    with pytest.raises(AppError, match="different organization"):
        await svc.patch_membership(
            db, membership_id, MembershipPatch(org_unit_id=tenant.foreign_faculty)
        )


async def test_patch_membership_can_change_status(db: AsyncSession, tenant: Tenant) -> None:
    membership_id = tenant.membership_ids[tenant.teacher]
    updated = await svc.patch_membership(db, membership_id, MembershipPatch(status="suspended"))
    await db.commit()
    assert updated.status == "suspended"


def test_patch_cannot_express_left_so_only_delete_can_set_it() -> None:
    """``patch_membership`` has a ``status == 'left'`` branch that stamps
    ``left_at`` — and no request can reach it.

    ``organization_memberships.status`` allows ``left`` at the database level
    and ``delete_membership`` writes it directly, but ``MembershipPatch.status``
    is ``Literal['active', 'inactive', 'suspended']``, so a PATCH carrying
    ``left`` is rejected by validation before the service sees it.

    Pinned rather than silently worked around: either the literal should gain
    ``left`` (making the branch live) or the branch should go. Whichever is
    chosen, this test fails and says so.
    """
    with pytest.raises(ValidationError):
        MembershipPatch(status="left")


async def test_delete_membership_marks_left_rather_than_deleting(
    db: AsyncSession, tenant: Tenant
) -> None:
    """The row survives with its history; access stops because every
    org-scope check filters on ``active``."""
    membership_id = tenant.membership_ids[tenant.manager]
    await svc.delete_membership(db, membership_id, actor_id=tenant.master)
    await db.commit()

    row = await org_queries.get_membership(db, membership_id)
    assert row is not None, "the membership must not be deleted"
    assert row.status == "left"
    assert row.left_at is not None
    assert row.deleted_at is None


async def test_membership_operations_on_unknown_ids_are_not_found(
    db: AsyncSession,
) -> None:
    missing = uuid.uuid4()
    with pytest.raises(NotFoundError):
        await svc.patch_membership(db, missing, MembershipPatch(status="left"))
    with pytest.raises(NotFoundError):
        await svc.delete_membership(db, missing, actor_id=None)


# ---------------------------------------------------------------------------
# Query-layer details the service depends on
# ---------------------------------------------------------------------------


async def test_role_scope_check_requires_both_halves_for_a_faculty_dean(
    db: AsyncSession, engine: AsyncEngine, tenant: Tenant
) -> None:
    """org_unit ``hod`` counts only while the faculty assignment is live.

    Deactivating the assignment must revoke the Dean's authority immediately,
    without touching the role row — that is how a Dean is stood down.
    """
    assert await org_queries.user_has_role_scope(
        db,
        user_id=tenant.dean,
        role_code="hod",
        organization_id=tenant.org_id,
        faculty_id=tenant.faculty_a,
    )

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE user_faculty_assignments SET status = 'inactive' "
                "WHERE user_id = :uid AND faculty_id = :faculty"
            ),
            {"uid": tenant.dean, "faculty": tenant.faculty_a},
        )

    assert not await org_queries.user_has_role_scope(
        db,
        user_id=tenant.dean,
        role_code="hod",
        organization_id=tenant.org_id,
        faculty_id=tenant.faculty_a,
    )


async def test_an_expired_role_assignment_does_not_grant_scope(
    db: AsyncSession, engine: AsyncEngine, tenant: Tenant
) -> None:
    async with engine.begin() as conn:
        await conn.execute(
            text(
                "UPDATE user_role_assignments SET active_until = :past "
                "WHERE user_id = :uid AND scope_kind = 'organization'"
            ),
            {"uid": tenant.master, "past": datetime.now(tz=UTC) - timedelta(days=1)},
        )
    assert not await org_queries.user_has_role_scope(
        db, user_id=tenant.master, role_code="hod", organization_id=tenant.org_id
    )


async def test_active_org_member_user_ids_filters_to_the_tenant(
    db: AsyncSession, tenant: Tenant
) -> None:
    members = await org_queries.active_org_member_user_ids(
        db, tenant.org_id, [tenant.teacher, tenant.outsider]
    )
    assert tenant.teacher in members
    assert tenant.outsider not in members


async def test_bulk_update_membership_unit_short_circuits_on_empty(
    db: AsyncSession,
) -> None:
    """An empty id list must not become ``IN ()`` — that is a SQL error."""
    assert await org_queries.bulk_update_membership_unit(db, [], None) == 0


async def test_list_memberships_by_ids_short_circuits_on_empty(
    db: AsyncSession,
) -> None:
    assert await org_queries.list_memberships_by_ids(db, []) == []
