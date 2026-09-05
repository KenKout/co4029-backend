"""Coverage for the policies feature: models, service state machine, both routers.

Policies are the platform's binding text. What makes them worth testing is not
the CRUD — it is the two properties that make a version number mean anything:

1. **Published text is immutable.** An admin fixing a clause must open a new
   draft; the version readers already agreed to is never rewritten under them.
2. **Exactly one version is current per (policy, language).** Publishing v3
   retires v2 in the same transaction, or ``published_version`` starts picking
   a winner by ordering and two documents both claim to be the terms.

The third property is about reach rather than correctness: the reader endpoints
are UNAUTHENTICATED on purpose, because the terms must be readable before an
account exists. A regression that puts them behind auth breaks the feature
without breaking a single response body, so it is asserted directly.

Audience is asserted as what it is — a relevance filter over public documents,
not access control. ``GET /policies/{slug}`` opens regardless of audience; only
the index narrows. That distinction is easy to "fix" into a security control it
was never meant to be, so both halves are pinned.

Isolation: the suite shares one Postgres, so every test owns policies under a
``test-`` slug prefix and the autouse fixture removes them (and their versions
and audience rows) on the way in and out.
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
from conftest import SeededUsers
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
from abridgeai.core.exceptions import AppError, ConflictError, NotFoundError
from abridgeai.core.security import create_access_token, generate_token, hash_secret
from abridgeai.features.policies import queries as policy_queries
from abridgeai.features.policies import services as policy_service
from abridgeai.features.policies.routers.admin import router as admin_router
from abridgeai.features.policies.routers.public import router as public_router
from abridgeai.features.policies.schemas import (
    PolicyAudienceUpdate,
    PolicyCreate,
    PolicyVersionCreate,
    PolicyVersionPatch,
)

SLUG_PREFIX = "test-policy-"


def _slug(name: str) -> str:
    return f"{SLUG_PREFIX}{name}-{uuid.uuid4().hex[:8]}"


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
async def db(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture
async def app(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[FastAPI]:
    async def _override_get_db() -> AsyncIterator[AsyncSession]:
        async with session_factory() as session:
            yield session

    fastapi_app = FastAPI()
    fastapi_app.include_router(public_router, prefix="/api/v1")
    fastapi_app.include_router(admin_router, prefix="/api/v1")
    fastapi_app.dependency_overrides[get_db] = _override_get_db
    yield fastapi_app
    fastapi_app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac


_ISSUED_SESSIONS: list[uuid.UUID] = []


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
    _ISSUED_SESSIONS.append(sid)
    return create_access_token(user_id=user_id, session_id=sid)


@pytest_asyncio.fixture(autouse=True)
async def _purge_issued_sessions(engine: AsyncEngine) -> AsyncIterator[None]:
    yield
    if not _ISSUED_SESSIONS:
        return
    async with engine.begin() as conn:
        await conn.execute(
            text("DELETE FROM auth_sessions WHERE id = ANY(CAST(:ids AS uuid[]))"),
            {"ids": [str(s) for s in _ISSUED_SESSIONS]},
        )
    _ISSUED_SESSIONS.clear()


@pytest_asyncio.fixture(autouse=True)
async def _clean_policies(engine: AsyncEngine) -> AsyncIterator[None]:
    """Own every ``test-policy-*`` row and its children, before and after.

    Cleaning on the way IN as well as out matters: a leftover row from a run
    that failed before teardown would otherwise make the index assertions here
    a test of that leftover instead.
    """

    async def _wipe() -> None:
        # The policy FKs are ON DELETE NO ACTION by design — policies are
        # audited, soft-deleted records and a hard delete must not silently
        # take their version history with it. So children go first, in order.
        async with engine.begin() as conn:
            for table in ("policy_audience_roles", "policy_versions"):
                await conn.execute(
                    text(
                        f"DELETE FROM {table} WHERE policy_id IN "  # noqa: S608
                        "(SELECT id FROM policies WHERE slug LIKE :p)"
                    ),
                    {"p": f"{SLUG_PREFIX}%"},
                )
            await conn.execute(
                text("DELETE FROM policies WHERE slug LIKE :p"),
                {"p": f"{SLUG_PREFIX}%"},
            )

    await _wipe()
    yield
    await _wipe()


def _auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def _published_policy(
    db: AsyncSession,
    *,
    slug: str,
    title: str = "Test Policy",
    body: str = "## Section\n\nText.",
    category: str = "legal",
    audience: list[str] | None = None,
    actor_id: uuid.UUID | None = None,
) -> uuid.UUID:
    """Create a policy and publish v1. Returns the policy id."""
    detail = await policy_service.create_policy(
        db,
        PolicyCreate(slug=slug, category=category, title=title),  # type: ignore[arg-type]
        actor_id=actor_id,
    )
    draft = detail.versions[0]
    await policy_service.update_draft(
        db, draft.id, PolicyVersionPatch(body=body), actor_id=actor_id
    )
    await policy_service.publish_version(db, draft.id, actor_id=actor_id)
    if audience:
        await policy_service.set_audience(
            db, detail.id, PolicyAudienceUpdate(role_codes=audience), actor_id=actor_id
        )
    await db.commit()
    return detail.id


# ---------------------------------------------------------------------------
# Service: the draft/publish state machine
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_opens_an_empty_v1_draft(db: AsyncSession) -> None:
    """A policy with no version would be unreachable and unopenable."""
    detail = await policy_service.create_policy(
        db, PolicyCreate(slug=_slug("create"), category="legal", title="Terms"), actor_id=None
    )
    await db.commit()

    assert len(detail.versions) == 1
    v = detail.versions[0]
    assert v.version_no == 1
    assert v.status == "draft"
    assert v.title == "Terms"


@pytest.mark.asyncio
async def test_duplicate_slug_is_rejected(db: AsyncSession) -> None:
    slug = _slug("dupe")
    await policy_service.create_policy(
        db, PolicyCreate(slug=slug, category="legal", title="First"), actor_id=None
    )
    await db.commit()

    with pytest.raises(ConflictError):
        await policy_service.create_policy(
            db, PolicyCreate(slug=slug, category="legal", title="Second"), actor_id=None
        )


@pytest.mark.asyncio
async def test_published_version_cannot_be_edited(db: AsyncSession) -> None:
    """The core immutability rule: readers agreed to this exact text."""
    slug = _slug("immutable")
    policy_id = await _published_policy(db, slug=slug)
    published = await policy_queries.published_version(db, policy_id, language="en")
    assert published is not None

    with pytest.raises(AppError):
        await policy_service.update_draft(
            db, published.id, PolicyVersionPatch(body="rewritten"), actor_id=None
        )


@pytest.mark.asyncio
async def test_new_draft_is_seeded_from_the_published_text(db: AsyncSession) -> None:
    """Copy-on-write: fixing one clause should not mean retyping the document."""
    slug = _slug("seeded")
    policy_id = await _published_policy(db, slug=slug, body="## Original\n\nBody.")

    draft = await policy_service.open_new_draft(db, policy_id, PolicyVersionCreate(), actor_id=None)
    await db.commit()

    assert draft.version_no == 2
    assert draft.status == "draft"
    full = await policy_service.read_version(db, policy_id, draft.id)
    assert "## Original" in full.body


@pytest.mark.asyncio
async def test_only_one_draft_per_language(db: AsyncSession) -> None:
    """Two open drafts make "the draft" ambiguous for the editor AND publish."""
    slug = _slug("one-draft")
    policy_id = await _published_policy(db, slug=slug)
    await policy_service.open_new_draft(db, policy_id, PolicyVersionCreate(), actor_id=None)
    await db.commit()

    with pytest.raises(ConflictError):
        await policy_service.open_new_draft(db, policy_id, PolicyVersionCreate(), actor_id=None)


@pytest.mark.asyncio
async def test_publishing_retires_the_previous_version(db: AsyncSession) -> None:
    """Exactly one current version, or two documents both claim to be the terms."""
    slug = _slug("supersede")
    policy_id = await _published_policy(db, slug=slug, body="## V1\n\nOld.")

    draft = await policy_service.open_new_draft(db, policy_id, PolicyVersionCreate(), actor_id=None)
    await policy_service.update_draft(
        db, draft.id, PolicyVersionPatch(body="## V2\n\nNew."), actor_id=None
    )
    await policy_service.publish_version(db, draft.id, actor_id=None)
    await db.commit()

    versions = await policy_queries.list_versions(db, policy_id)
    published = [v for v in versions if v.status == "published"]
    archived = [v for v in versions if v.status == "archived"]
    assert len(published) == 1
    assert published[0].version_no == 2
    assert [v.version_no for v in archived] == [1]

    doc = await policy_service.read_document(db, slug)
    assert "## V2" in doc.body


@pytest.mark.asyncio
async def test_empty_body_cannot_be_published(db: AsyncSession) -> None:
    """create_policy opens v1 empty, so this is the very first reachable state."""
    detail = await policy_service.create_policy(
        db, PolicyCreate(slug=_slug("empty"), category="legal", title="Blank"), actor_id=None
    )
    await db.commit()

    with pytest.raises(AppError):
        await policy_service.publish_version(db, detail.versions[0].id, actor_id=None)


@pytest.mark.asyncio
async def test_publication_is_attributed_to_the_publisher(
    db: AsyncSession, seeded_users: SeededUsers
) -> None:
    """``published_by`` is stamped by the service, never accepted from a client."""
    slug = _slug("attributed")
    await _published_policy(db, slug=slug, actor_id=seeded_users.admin_id)

    doc = await policy_service.read_document(db, slug)
    assert doc.published_by_name is not None
    assert doc.version_no == 1


@pytest.mark.asyncio
async def test_unpublished_policy_is_not_readable(db: AsyncSession) -> None:
    """A draft is not the terms; reading one would show text nobody released."""
    slug = _slug("draft-only")
    detail = await policy_service.create_policy(
        db, PolicyCreate(slug=slug, category="legal", title="Draft"), actor_id=None
    )
    await policy_service.update_draft(
        db, detail.versions[0].id, PolicyVersionPatch(body="Unreleased."), actor_id=None
    )
    await db.commit()

    with pytest.raises(NotFoundError):
        await policy_service.read_document(db, slug)


@pytest.mark.asyncio
async def test_body_is_sanitized_on_save(db: AsyncSession) -> None:
    """Bodies are admin-authored markdown, but admin-authored is not trusted."""
    slug = _slug("sanitize")
    detail = await policy_service.create_policy(
        db, PolicyCreate(slug=slug, category="legal", title="XSS"), actor_id=None
    )
    await policy_service.update_draft(
        db,
        detail.versions[0].id,
        PolicyVersionPatch(body="Safe.<script>alert(1)</script>"),
        actor_id=None,
    )
    await db.commit()

    stored = await policy_service.read_version(db, detail.id, detail.versions[0].id)
    assert "<script>" not in stored.body


# ---------------------------------------------------------------------------
# Audience: a relevance filter, not access control
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_index_hides_policies_the_reader_is_not_party_to(db: AsyncSession) -> None:
    public_slug = _slug("public")
    scoped_slug = _slug("scoped")
    await _published_policy(db, slug=public_slug)
    await _published_policy(db, slug=scoped_slug, audience=["teacher"])

    anonymous = {p.slug for p in await policy_service.list_documents(db)}
    assert public_slug in anonymous
    assert scoped_slug not in anonymous

    teacher = {p.slug for p in await policy_service.list_documents(db, role_codes=["teacher"])}
    assert scoped_slug in teacher
    # Public documents stay in the list; the audience widens, never replaces.
    assert public_slug in teacher

    student = {p.slug for p in await policy_service.list_documents(db, role_codes=["student"])}
    assert scoped_slug not in student


@pytest.mark.asyncio
async def test_student_audience_means_students_only(db: AsyncSession) -> None:
    """The audience is LITERAL: naming ``student`` excludes every other role.

    There is no universal role any more. A teacher or an admin reading their
    index is not shown a policy governed by students specifically, and an
    anonymous visitor sees only the public set.
    """
    student_slug = _slug("students-only")
    await _published_policy(db, slug=student_slug, audience=["student"])

    students = {p.slug for p in await policy_service.list_documents(db, role_codes=["student"])}
    assert student_slug in students

    for roles in (None, ["teacher"], ["hod"], ["admin"], ["teacher", "hod"]):
        reader = {p.slug for p in await policy_service.list_documents(db, role_codes=roles)}
        assert student_slug not in reader, f"leaked to {roles}"


@pytest.mark.asyncio
async def test_multi_role_reader_sees_each_role_s_audience(db: AsyncSession) -> None:
    """A reader holding several roles unions those audiences."""
    for_slug = _slug("for-students")
    teach_slug = _slug("for-teachers")
    await _published_policy(db, slug=for_slug, audience=["student"])
    await _published_policy(db, slug=teach_slug, audience=["teacher"])

    both = {p.slug for p in await policy_service.list_documents(db, role_codes=["student", "teacher"])}
    assert for_slug in both
    assert teach_slug in both


@pytest.mark.asyncio
async def test_empty_audience_is_the_only_everyone(db: AsyncSession) -> None:
    """Everyone = no audience rows. Any named role narrows to that role."""
    public_slug = _slug("truly-public")
    await _published_policy(db, slug=public_slug)

    for roles in (None, ["student"], ["teacher"], ["student", "teacher"]):
        reader = {p.slug for p in await policy_service.list_documents(db, role_codes=roles)}
        assert public_slug in reader, f"public policy hidden from {roles}"


@pytest.mark.asyncio
async def test_a_scoped_policy_still_opens_by_its_own_url(db: AsyncSession) -> None:
    """The audience scopes the INDEX. A shared or bookmarked link must open.

    Hiding a public document from the person it governs was never a security
    boundary, and turning this into one would break emailed policy links.
    """
    slug = _slug("direct")
    await _published_policy(db, slug=slug, audience=["manager"])

    doc = await policy_service.read_document(db, slug)
    assert doc.slug == slug


@pytest.mark.asyncio
async def test_unknown_role_code_filters_rather_than_errors(db: AsyncSession) -> None:
    """Reader-supplied codes are untrusted input to a courtesy filter."""
    slug = _slug("unknown-role")
    await _published_policy(db, slug=slug)

    slugs = {p.slug for p in await policy_service.list_documents(db, role_codes=["not-a-role"])}
    assert slug in slugs


@pytest.mark.asyncio
async def test_setting_an_unknown_audience_role_is_rejected(db: AsyncSession) -> None:
    """Admin input is different: a typo here would silently hide a policy."""
    slug = _slug("bad-audience")
    policy_id = await _published_policy(db, slug=slug)

    with pytest.raises(AppError):
        await policy_service.set_audience(
            db, policy_id, PolicyAudienceUpdate(role_codes=["nope"]), actor_id=None
        )


@pytest.mark.asyncio
async def test_empty_audience_makes_a_policy_public_again(db: AsyncSession) -> None:
    slug = _slug("re-public")
    policy_id = await _published_policy(db, slug=slug, audience=["teacher"])

    await policy_service.set_audience(
        db, policy_id, PolicyAudienceUpdate(role_codes=[]), actor_id=None
    )
    await db.commit()

    slugs = {p.slug for p in await policy_service.list_documents(db)}
    assert slug in slugs


# ---------------------------------------------------------------------------
# Reader router: unauthenticated on purpose
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reader_endpoints_need_no_session(
    client: httpx.AsyncClient, db: AsyncSession
) -> None:
    """The terms must be readable BEFORE an account exists.

    A regression nesting these behind auth breaks the feature without changing
    a single response body, so it is asserted at the HTTP edge with no header.
    """
    slug = _slug("anon")
    await _published_policy(db, slug=slug, title="Anon Readable")

    index = await client.get("/api/v1/policies")
    assert index.status_code == 200
    assert slug in {p["slug"] for p in index.json()}

    doc = await client.get(f"/api/v1/policies/{slug}")
    assert doc.status_code == 200
    assert doc.json()["title"] == "Anon Readable"


@pytest.mark.asyncio
async def test_reader_gets_404_for_an_unknown_slug(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/v1/policies/no-such-policy-anywhere")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_reader_index_accepts_repeated_role_params(
    client: httpx.AsyncClient, db: AsyncSession
) -> None:
    """The frontend sends one ``role`` param per role, not a joined string."""
    scoped = _slug("multi-role")
    await _published_policy(db, slug=scoped, audience=["hod"])

    resp = await client.get("/api/v1/policies", params=[("role", "student"), ("role", "hod")])
    assert resp.status_code == 200
    assert scoped in {p["slug"] for p in resp.json()}


# ---------------------------------------------------------------------------
# Admin router: gated on system.administer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_admin_endpoints_reject_a_non_admin(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    token = await _bearer(engine, seeded_users.teacher_id)
    resp = await client.get("/api/v1/admin/policies", headers=_auth(token))
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_endpoints_reject_anonymous(client: httpx.AsyncClient) -> None:
    resp = await client.get("/api/v1/admin/policies")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_admin_can_author_and_publish_over_http(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers
) -> None:
    """The whole authoring loop the editor drives, through the real endpoints."""
    token = await _bearer(engine, seeded_users.admin_id)
    headers = _auth(token)
    slug = _slug("http-flow")

    created = await client.post(
        "/api/v1/admin/policies",
        headers=headers,
        json={"slug": slug, "category": "academic", "title": "Interview Policy"},
    )
    assert created.status_code == 201
    policy = created.json()
    version_id = policy["versions"][0]["id"]

    patched = await client.patch(
        f"/api/v1/admin/policies/{policy['id']}/versions/{version_id}",
        headers=headers,
        json={"body": "## Scope\n\nApplies to interviews.", "changelog": "First release."},
    )
    assert patched.status_code == 200

    # The editor loads the body from this endpoint, not from the detail payload.
    fetched = await client.get(
        f"/api/v1/admin/policies/{policy['id']}/versions/{version_id}", headers=headers
    )
    assert fetched.status_code == 200
    assert "## Scope" in fetched.json()["body"]

    published = await client.post(
        f"/api/v1/admin/policies/{policy['id']}/versions/{version_id}/publish",
        headers=headers,
        json={},
    )
    assert published.status_code == 200
    assert published.json()["status"] == "published"

    # Visible to an anonymous reader the moment it is published.
    public = await client.get(f"/api/v1/policies/{slug}")
    assert public.status_code == 200
    assert public.json()["version_no"] == 1


@pytest.mark.asyncio
async def test_admin_audience_update_returns_role_names(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, db: AsyncSession
) -> None:
    """Names come from the roles catalogue, so a renamed role renames here too."""
    token = await _bearer(engine, seeded_users.admin_id)
    slug = _slug("audience-http")
    policy_id = await _published_policy(db, slug=slug)

    resp = await client.put(
        f"/api/v1/admin/policies/{policy_id}/audience",
        headers=_auth(token),
        json={"role_codes": ["student", "teacher"]},
    )
    assert resp.status_code == 200
    audience = resp.json()["audience"]
    assert {r["code"] for r in audience} == {"student", "teacher"}
    assert all(r["name"] for r in audience)


@pytest.mark.asyncio
async def test_admin_version_fetch_checks_the_parent_policy(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, db: AsyncSession
) -> None:
    """A mismatched id pair is a caller bug, not a document to serve."""
    token = await _bearer(engine, seeded_users.admin_id)
    a_id = await _published_policy(db, slug=_slug("parent-a"))
    b_id = await _published_policy(db, slug=_slug("parent-b"))
    b_version = await policy_queries.published_version(db, b_id, language="en")
    assert b_version is not None

    resp = await client.get(
        f"/api/v1/admin/policies/{a_id}/versions/{b_version.id}", headers=_auth(token)
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_admin_conflict_surfaces_as_409(
    client: httpx.AsyncClient, engine: AsyncEngine, seeded_users: SeededUsers, db: AsyncSession
) -> None:
    """A rule that raises correctly but surfaces as a 500 is still broken."""
    token = await _bearer(engine, seeded_users.admin_id)
    policy_id = await _published_policy(db, slug=_slug("conflict"))
    headers = _auth(token)

    first = await client.post(
        f"/api/v1/admin/policies/{policy_id}/versions", headers=headers, json={}
    )
    assert first.status_code == 201

    second = await client.post(
        f"/api/v1/admin/policies/{policy_id}/versions", headers=headers, json={}
    )
    assert second.status_code == 409
