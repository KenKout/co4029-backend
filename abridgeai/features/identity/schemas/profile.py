from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class _ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserProfileRead(_ORMModel):
    user_id: UUID
    given_name: str | None = None
    family_name: str | None = None
    display_name: str
    avatar_object_id: UUID | None = None
    # Presigned GET URL for the avatar image, minted server-side on each read
    # (short TTL). ``None`` when no avatar is set. Not persisted — a projection.
    avatar_url: str | None = None
    bio: str | None = None
    locale: str | None = None


class UserRead(_ORMModel):
    id: UUID
    primary_email: str
    status: str
    last_login_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    profile: UserProfileRead | None = None
    # Distinct active role codes across all scopes (admin user-list "Role"
    # column). Empty when the user holds no active role assignment. Populated
    # by the admin list/search services; other UserRead producers leave it [].
    roles: list[str] = Field(default_factory=list)
    # Primary organization (most recent active membership) for the admin
    # user-list "Organization" column. None when the user belongs to no org
    # (e.g. platform admins). Populated by the search service only.
    organization_id: UUID | None = None
    organization_name: str | None = None


class UserProfileUpdate(BaseModel):
    given_name: str | None = Field(default=None, max_length=100)
    family_name: str | None = Field(default=None, max_length=100)
    display_name: str | None = Field(default=None, max_length=200)
    bio: str | None = None
    # Preferred UI + notification language. Constrained to the frontend
    # SUPPORTED_LOCALES; the DB CHECK constraint (migration 0024) is the
    # backstop. ``None`` on PATCH means "leave unchanged" (exclude_unset).
    locale: Literal["en", "vi"] | None = None


class UserCreate(BaseModel):
    """Admin invite payload — create a user and attach them to an org.

    The account is created ``active`` with an org-scoped role assignment
    (``role_code``, default ``student``) plus an active membership, so the
    invited email can sign in via Google OAuth immediately (the
    pre-registration gate accepts existing ``users`` rows) and is already
    scoped to its organization.

    ``organization_id`` is optional at the schema level because a non-admin
    inviter (manager) is never allowed to pick: the router forces the
    caller's own primary organization server-side. Platform admins MUST
    still provide it.
    """

    primary_email: str = Field(min_length=3, max_length=320)
    given_name: str | None = Field(default=None, max_length=100)
    family_name: str | None = Field(default=None, max_length=100)
    display_name: str | None = Field(default=None, max_length=200)
    organization_id: UUID | None = None
    role_code: str = Field(default="student", min_length=1, max_length=50)
    student_code: str | None = Field(default=None, max_length=50)
    employee_code: str | None = Field(default=None, max_length=50)


class UserProfileLinkIn(BaseModel):
    link_type: str = Field(max_length=30)
    url: str
    label: str | None = Field(default=None, max_length=100)


class UserProfileLinkRead(_ORMModel):
    id: UUID
    user_id: UUID
    link_type: str
    url: str
    label: str | None = None
    created_at: datetime
    updated_at: datetime


class AuthSessionRead(_ORMModel):
    id: UUID
    expires_at: datetime
    revoked_at: datetime | None = None
    mfa_verified_at: datetime | None = None
    ip_address: str | None = None
    user_agent: str | None = None
    created_at: datetime
    updated_at: datetime


class UserPermissionsRead(BaseModel):
    """Effective permission codes the current user holds (T1.9 ``/users/me/permissions``)."""

    permissions: list[str]


class UserListPage(BaseModel):
    """Cursor-paginated list of users (T1.9 ``GET /users``)."""

    items: list[UserRead]
    next_cursor: str | None = None


class CourseProgressRead(BaseModel):
    """Per-course progress for the manager/HOD user-detail page."""

    course_id: UUID
    title: str
    slug: str
    status: str
    enrollment_status: str
    enrolled_at: datetime
    completion_percent: float
    completed_lessons: int
    total_lessons: int


class CareerPathProgressRead(BaseModel):
    """Career-path enrolment + progress for the manager/HOD user-detail page."""

    career_path_id: UUID
    name: str
    slug: str
    status: str
    started_at: datetime
    completed_at: datetime | None = None
    completed_courses: int
    course_count: int
    completion_percent: float


class ProgramPathAttemptRead(BaseModel):
    """One path a student took inside a learning program.

    A switch is recorded as a NEW attempt rather than by mutating the old
    one, so the list is the student's path history: what they picked, what
    they abandoned, and when.
    """

    career_path_id: UUID
    career_path_name: str | None = None
    status: str
    selected_at: datetime
    ended_at: datetime | None = None


class ProgramProgressRead(BaseModel):
    """Learning-program enrolment + progress, sibling of the career-path row.

    ``completion_percent`` is measured against the path version PINNED to
    this enrolment, not the path's current head — which is the whole point
    of program versioning, and why this cannot be derived from the
    career-path section beside it.
    """

    enrollment_id: UUID
    learning_program_id: UUID
    program_name: str
    program_version_no: int
    status: str
    enrolled_at: datetime
    completed_at: datetime | None = None
    withdrawn_at: datetime | None = None
    completed_courses: int = 0
    course_count: int = 0
    completion_percent: float = 0
    max_path_switches: int = 0
    approved_switch_count: int = 0
    attempts: list[ProgramPathAttemptRead] = Field(default_factory=list)


class AssignedCourseRead(BaseModel):
    """A course assigned to a teacher (manager/HOD user-detail page)."""

    course_id: UUID
    title: str
    slug: str
    status: str


class UserOverviewRead(BaseModel):
    """Org-scoped user detail for managers / HODs (``GET /users/{id}/overview``).

    The caller must hold ``user.read`` and the target must belong to the
    caller's organization (cross-org lookups 404). What is populated depends
    on the target's role:

    * ``student`` — ``courses`` (enrolments + per-course progress),
      ``career_paths`` (enrolments + progress) and ``last_active_at``.
    * ``teacher`` — ``assigned_courses``.
    * manager / HOD / admin — basic identity only (``user``).
    """

    user: UserRead
    courses: list[CourseProgressRead] = Field(default_factory=list)
    career_paths: list[CareerPathProgressRead] = Field(default_factory=list)
    programs: list[ProgramProgressRead] = Field(default_factory=list)
    assigned_courses: list[AssignedCourseRead] = Field(default_factory=list)
    last_active_at: datetime | None = None
