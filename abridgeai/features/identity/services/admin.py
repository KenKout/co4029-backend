"""Admin-side identity reads (T1.9 ``GET /users/{id}`` + ``GET /users``).

Two helpers feed the admin lookup router:

* :func:`get_user_with_profile` -- single-user lookup by id.
* :func:`list_users` -- cursor-paginated list (base64 of last user id).

Routers must not import ``queries.*`` directly (import-linter contract #2),
so this service translates the raw row reads into ``UserRead`` instances.
The cursor format is a base64-encoded UUID of the last item on the previous
page; the implementation is intentionally minimal (sort by ``users.id``).
Future work may add timestamp-based opaque cursors per Reconciliation §A10.
"""

from __future__ import annotations

import base64
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.core.pagination import Page
from abridgeai.features.access_control.api import public as access_control_api
from abridgeai.features.identity.models import User, UserProfile
from abridgeai.features.identity.queries import users as user_queries
from abridgeai.features.identity.schemas import (
    AssignedCourseRead,
    CareerPathProgressRead,
    CourseProgressRead,
    ProgramProgressRead,
    UserCreate,
    UserListPage,
    UserOverviewRead,
    UserRead,
)
from abridgeai.features.identity.services.profile import serialize_user
from abridgeai.infrastructure.s3 import create_stream_url

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from abridgeai.features.access_control.api._dto import OrgDTO
    from abridgeai.features.identity.models import StorageObject


async def _mint_avatar_url(
    storage_map: dict[UUID, StorageObject],
    profile: UserProfile | None,
) -> str | None:
    """Presign a profile's avatar object from the page's batch-loaded map.

    The sign call is local-only (no network/DB round-trip), so minting one
    URL per row does not reintroduce an N+1; a storage blip drops that one
    avatar (FE falls back to initials) rather than failing the page.
    """
    if profile is None or profile.avatar_object_id is None:
        return None
    storage = storage_map.get(profile.avatar_object_id)
    if storage is None:
        return None
    try:
        url, _ = await create_stream_url(storage)
    except Exception:  # noqa: BLE001 — never let a storage blip break the list
        return None
    return url


async def _serialize_search_row(
    user: User,
    profile: UserProfile | None,
    storage_map: dict[UUID, StorageObject],
    role_map: dict[UUID, list[str]],
    org: OrgDTO | None,
) -> UserRead:
    """Serialize one search row with minted avatar URL + role/org aggregates."""
    avatar_url = await _mint_avatar_url(storage_map, profile)
    row = serialize_user(user, profile)
    if avatar_url is not None and row.profile is not None:
        row.profile.avatar_url = avatar_url
    return row.model_copy(
        update={
            "roles": role_map.get(user.id, []),
            "organization_id": org.id if org else None,
            "organization_name": org.name if org else None,
        }
    )


_DEFAULT_LIMIT = 50
_MAX_LIMIT = 100


def _encode_cursor(user_id: UUID) -> str:
    return base64.urlsafe_b64encode(str(user_id).encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> UUID:
    padding = "=" * (-len(cursor) % 4)
    raw = base64.urlsafe_b64decode((cursor + padding).encode()).decode()
    return UUID(raw)


async def get_user_with_profile(db: AsyncSession, user_id: UUID) -> UserRead | None:
    """Return ``UserRead`` for ``user_id`` (with profile if present), or ``None``."""
    user = await user_queries.get_user(db, user_id)
    if user is None:
        return None
    profile = await user_queries.get_profile(db, user_id)
    return serialize_user(user, profile)


async def list_users(
    db: AsyncSession,
    *,
    cursor: str | None = None,
    limit: int = _DEFAULT_LIMIT,
) -> UserListPage:
    """Cursor-paginated user list ordered by ``users.id``.

    ``cursor`` is the base64-encoded UUID of the last item on the previous
    page. ``limit`` is clamped to ``_MAX_LIMIT``. ``next_cursor`` is set when
    the page was full (i.e. there may be more rows); ``None`` otherwise.
    """
    capped = min(max(limit, 1), _MAX_LIMIT)
    after: UUID | None = None
    if cursor:
        try:
            after = _decode_cursor(cursor)
        except (ValueError, UnicodeDecodeError) as exc:
            raise ValueError("Invalid cursor") from exc

    rows = await user_queries.list_users(db, after=after, limit=capped)
    items: list[UserRead] = []
    profiles = {p.user_id: p for p in await user_queries.list_profiles(db, [u.id for u in rows])}
    for user in rows:
        items.append(serialize_user(user, profiles.get(user.id)))

    next_cursor = _encode_cursor(rows[-1].id) if len(rows) == capped else None
    return UserListPage(items=items, next_cursor=next_cursor)


async def search_users(
    db: AsyncSession,
    *,
    status: str | None = None,
    search: str | None = None,
    role: str | None = None,
    organization: UUID | None = None,
    org_unit: UUID | None = None,
    sort: str | None = None,
    sort_dir: str = "asc",
    page: int = 0,
    page_size: int = 25,
) -> Page[UserRead]:
    """Offset page of users (server-side search + whitelisted sort) as
    ``UserRead``. Delegates the SQLAlchemy statement to the query layer, then
    batch-loads profiles, role codes, and primary orgs for the page.

    ``role`` / ``organization`` filter to users holding that role code (any
    scope) / belonging to that org. Both id sets are resolved via
    ``access_control.api.public`` (feature independence) and intersected into
    a single allowlist passed to the query; an empty intersection
    short-circuits to an empty page.
    """
    restrict_sets: list[set[UUID]] = []
    if role:
        restrict_sets.append(set(await access_control_api.list_user_ids_with_role(db, role)))
    if organization:
        restrict_sets.append(set(await access_control_api.list_user_ids_in_org(db, organization)))
    if org_unit:
        # The parameter name remains backward compatible; it now narrows to
        # active staff affiliations in one top-level Faculty. Students are
        # intentionally excluded because they have no Faculty affiliation.
        restrict_sets.append(set(await access_control_api.list_user_ids_in_org_unit(db, org_unit)))

    restrict_ids: list[UUID] | None = None
    if restrict_sets:
        intersection = set.intersection(*restrict_sets)
        if not intersection:
            return Page(items=[], total=0, page=page, page_size=page_size, total_pages=0)
        restrict_ids = list(intersection)

    result = await user_queries.search_users(
        db,
        status=status,
        search=search,
        restrict_ids=restrict_ids,
        sort=sort,
        sort_dir=sort_dir,
        page=page,
        page_size=page_size,
    )
    user_ids = [u.id for u in result.items]
    profiles = {p.user_id: p for p in await user_queries.list_profiles(db, user_ids)}
    # Batch-load avatar storage objects for the page, then presign each (the
    # sign call is local-only, so minting stays O(1) network for the page).
    avatar_ids = [p.avatar_object_id for p in profiles.values() if p.avatar_object_id is not None]
    storage_map = await user_queries.list_storage_objects(db, avatar_ids)
    role_map = await access_control_api.get_role_codes_for_users(db, user_ids)
    org_map = await access_control_api.get_primary_orgs_for_users(db, user_ids)
    items = []
    for u in result.items:
        org = org_map.get(u.id)
        items.append(await _serialize_search_row(u, profiles.get(u.id), storage_map, role_map, org))
    return Page(
        items=items,
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        total_pages=result.total_pages,
    )


async def get_user_overview(db: AsyncSession, *, user_id: UUID) -> UserOverviewRead:
    """Assemble the manager/HOD user-detail payload for ``user_id``.

    Starts from the user's identity (same ``UserRead`` the list rows use),
    then adds role-dependent sections:

    * ``student`` — enrolled courses with per-course progress, career-path
      enrolments with progress, and the latest activity timestamp.
    * ``teacher`` — courses the user is actively assigned to teach.
    * manager / HOD / admin — identity only (no learning surface).

    All cross-feature reads go through the sibling ``api.public`` surfaces
    (enrollments, progress, career_paths, courses); this service only
    assembles. The caller (router) is responsible for the org-scope guard.
    """
    user = await user_queries.get_user(db, user_id)
    if user is None:
        raise NotFoundError(f"user {user_id} not found")
    profile = await user_queries.get_profile(db, user_id)
    role_map = await access_control_api.get_role_codes_for_users(db, [user_id])
    org_map = await access_control_api.get_primary_orgs_for_users(db, [user_id])
    org = org_map.get(user_id)
    user_read = await _serialize_search_row(user, profile, {}, role_map, org)

    role_codes = role_map.get(user_id, [])
    overview = UserOverviewRead(user=user_read)

    if "student" in role_codes:
        await _attach_student_sections(db, overview, user_id)
    if "teacher" in role_codes:
        from abridgeai.features.courses.api import public as courses_api

        assigned = await courses_api.list_courses_for_teacher(db, user_id)
        overview.assigned_courses = [
            AssignedCourseRead(course_id=c.id, title=c.title, slug=c.slug, status=c.status)
            for c in assigned
        ]
    return overview


async def _attach_student_sections(
    db: AsyncSession, overview: UserOverviewRead, student_id: UUID
) -> None:
    """Fill ``courses`` / ``career_paths`` / ``last_active_at`` for a student."""
    from abridgeai.features.career_paths.api import public as career_paths_api
    from abridgeai.features.courses.api import public as courses_api
    from abridgeai.features.enrollments.api import public as enrollments_api
    from abridgeai.features.learning_programs.api import (  # noqa: PLC0415
        public as learning_programs_api,
    )
    from abridgeai.features.progress.api import public as progress_api

    active_at: datetime | None = overview.user.last_login_at

    enrollments = await enrollments_api.list_user_course_enrollments(db, user_id=student_id)
    courses: list[CourseProgressRead] = []
    for enrollment in enrollments:
        # Dropped enrolments are KEPT. They were skipped, which meant a
        # student who left a course simply vanished from this page and a
        # manager had no way to see it ever happened. `enrollment_status`
        # already distinguishes them and the SPA renders a muted "dropped"
        # badge, so showing the row costs nothing and restores the history.
        course = await courses_api.get_course_by_id(db, enrollment.course_id)
        if course is None:
            continue
        progress = await progress_api.get_course_progress_for_user(
            db, user_id=student_id, course_id=enrollment.course_id
        )
        completion = float(progress.get("completion_percent") or 0)
        last = progress.get("last_activity_at")
        if last is not None:
            last_dt = last if isinstance(last, datetime) else datetime.fromisoformat(str(last))
            active_at = max(active_at, last_dt) if active_at else last_dt
        courses.append(
            CourseProgressRead(
                course_id=enrollment.course_id,
                title=course.title,
                slug=course.slug,
                status=course.status,
                enrollment_status=enrollment.status,
                enrolled_at=enrollment.enrolled_at,
                completion_percent=completion,
                completed_lessons=int(progress.get("completed_lessons") or 0),
                total_lessons=int(progress.get("total_lessons") or 0),
            )
        )
    overview.courses = courses

    path_rows = await career_paths_api.list_user_career_enrollments(db, student_id=student_id)
    paths: list[CareerPathProgressRead] = []
    for row in path_rows:
        career_path_id = UUID(str(row["career_path_id"]))
        course_rows = await career_paths_api.get_path_course_progress_for_user(
            db, career_path_id=career_path_id, student_id=student_id
        )
        completed = sum(1 for r in course_rows if r.get("satisfied"))
        total = len(course_rows)
        percent = (
            round(
                sum(float(r.get("completion_percent") or 0) for r in course_rows) / total,
                2,
            )
            if total
            else 0.0
        )
        paths.append(
            CareerPathProgressRead(
                career_path_id=career_path_id,
                name=str(row["name"]),
                slug=str(row["slug"]),
                status=str(row["status"]),
                started_at=row["started_at"],  # type: ignore[arg-type]
                completed_at=row.get("completed_at"),  # type: ignore[arg-type]
                completed_courses=completed,
                course_count=total,
                completion_percent=percent,
            )
        )
    overview.career_paths = paths

    # Learning programs sit beside career paths: a program pins a specific
    # path VERSION, so its progress is measured against what the student was
    # actually enrolled onto rather than the path's current head.
    program_rows = await learning_programs_api.list_student_program_enrollments(
        db, student_id=student_id
    )
    overview.programs = [ProgramProgressRead.model_validate(row) for row in program_rows]
    overview.last_active_at = active_at


async def create_user_account(
    db: AsyncSession,
    *,
    payload: UserCreate,
    actor_id: UUID,
) -> UserRead:
    """Admin invite: create a user with profile + org membership + role.

    The account is created ``active`` so the invited email can sign in via
    Google OAuth right away (the pre-registration gate accepts existing
    ``users`` rows), and it is attached to ``payload.organization_id`` with
    an org-scoped ``payload.role_code`` assignment via
    :func:`abridgeai.features.access_control.api.public.grant_org_role_access`
    — the same least-privilege default path auto-provisioning uses, but with
    the admin-chosen role recorded in ``granted_by``.

    Raises :class:`ConflictError` when the email already exists.
    """
    existing = await user_queries.get_user_by_email(db, payload.primary_email)
    if existing is not None:
        raise ConflictError(f"user with email '{payload.primary_email}' already exists")

    user = User(primary_email=payload.primary_email, status="active")
    db.add(user)
    await db.flush()

    profile = UserProfile(
        user_id=user.id,
        given_name=payload.given_name,
        family_name=payload.family_name,
        display_name=payload.display_name or payload.primary_email.split("@")[0],
    )
    db.add(profile)

    await access_control_api.grant_org_role_access(
        db,
        user_id=user.id,
        organization_id=payload.organization_id,
        role_code=payload.role_code,
        granted_by=actor_id,
    )

    return serialize_user(user, profile)


__all__ = [
    "create_user_account",
    "get_user_overview",
    "get_user_with_profile",
    "list_users",
    "search_users",
]
