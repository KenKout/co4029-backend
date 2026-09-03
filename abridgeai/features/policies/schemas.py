"""Request/response models for the policies feature."""

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

PolicyCategory = Literal["legal", "academic"]
PolicyVersionStatus = Literal["draft", "published", "archived"]


class PolicyAudienceRoleRead(BaseModel):
    """A role the policy names as a party.

    Carries the role's ``code`` AND its display ``name``, both read from the
    roles catalogue — the point of the join table is that the label shown to
    an admin is the product's own role name, never a second copy of it.
    """

    role_id: UUID
    code: str
    name: str


class PolicyVersionSummary(BaseModel):
    """A version without its body — for history lists and pickers."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    version_no: int
    language: str
    status: PolicyVersionStatus
    title: str
    changelog: str | None = None
    published_at: datetime | None = None
    published_by: UUID | None = None
    updated_at: datetime


class PolicyVersionRead(PolicyVersionSummary):
    """A version with its body."""

    body: str
    format: str


class PolicyRead(BaseModel):
    """A policy's identity plus its audience."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    slug: str
    category: PolicyCategory
    audience: list[PolicyAudienceRoleRead] = Field(default_factory=list)


class PolicyDetail(PolicyRead):
    """Identity, audience, and every version — the admin detail payload."""

    versions: list[PolicyVersionSummary] = Field(default_factory=list)


class PolicyDocument(BaseModel):
    """What a READER gets: one policy resolved to its current published text.

    Deliberately flat and free of ids — this is the public payload that
    replaces the hardcoded ``PolicyDocument`` constant the front end used to
    import, so it carries exactly the fields a rendered page needs plus the
    provenance the entity now makes real (version, publisher, date).
    """

    slug: str
    category: PolicyCategory
    title: str
    body: str
    format: str
    language: str
    version_no: int
    published_at: datetime
    #: Display name of the publisher, resolved at read time. ``None`` when the
    #: publishing account has since been deleted — the FK is ``SET NULL``, and
    #: the document stays valid regardless of who is still employed.
    published_by_name: str | None = None
    changelog: str | None = None


class PolicySummary(BaseModel):
    """An index entry — the list a reader browses."""

    slug: str
    category: PolicyCategory
    title: str
    language: str
    version_no: int
    published_at: datetime


class PolicyCreate(BaseModel):
    slug: str = Field(min_length=1, max_length=100, pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    category: PolicyCategory
    title: str = Field(min_length=1, max_length=255)
    language: str = Field(default="en", min_length=2, max_length=10)


class PolicyVersionCreate(BaseModel):
    """Open a new draft. Body defaults to a copy of the latest version."""

    language: str = Field(default="en", min_length=2, max_length=10)
    title: str | None = Field(default=None, min_length=1, max_length=255)
    body: str | None = None
    changelog: str | None = Field(default=None, max_length=2000)


class PolicyVersionPatch(BaseModel):
    """Edit a DRAFT. Every field optional; only what is sent is written."""

    title: str | None = Field(default=None, min_length=1, max_length=255)
    body: str | None = None
    changelog: str | None = Field(default=None, max_length=2000)


class PolicyAudienceUpdate(BaseModel):
    """Replace the audience set.

    An empty list is meaningful and is NOT the same as "unchanged": it makes
    the policy public. That is why this is a PUT of the whole set rather than
    add/remove endpoints — the empty case has to be expressible.
    """

    role_codes: list[str] = Field(default_factory=list)


__all__ = [
    "PolicyAudienceRoleRead",
    "PolicyAudienceUpdate",
    "PolicyCategory",
    "PolicyCreate",
    "PolicyDetail",
    "PolicyDocument",
    "PolicyRead",
    "PolicySummary",
    "PolicyVersionCreate",
    "PolicyVersionPatch",
    "PolicyVersionRead",
    "PolicyVersionStatus",
    "PolicyVersionSummary",
]
