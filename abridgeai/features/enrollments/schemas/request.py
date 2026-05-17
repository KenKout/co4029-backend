from __future__ import annotations

import re
from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _validate_email(value: str) -> str:
    if not _EMAIL_RE.match(value):
        raise ValueError(f"invalid email: {value!r}")
    return value


class BulkEnrollRequest(BaseModel):
    user_ids: list[UUID] = Field(default_factory=list)
    emails: list[str] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @field_validator("emails")
    @classmethod
    def _validate_emails(cls, value: list[str]) -> list[str]:
        return [_validate_email(item) for item in value]

    @model_validator(mode="after")
    def _at_least_one(self) -> BulkEnrollRequest:
        if not self.user_ids and not self.emails:
            raise ValueError("BulkEnrollRequest requires at least one user_id or email")
        return self


class BulkEnrollFailure(BaseModel):
    identifier: str
    reason: str


class BulkEnrollResult(BaseModel):
    enrolled: list[UUID]
    failures: list[BulkEnrollFailure]


class CSVImportRow(BaseModel):
    email: str
    given_name: str | None = None
    family_name: str | None = None
    display_name: str | None = None

    model_config = ConfigDict(extra="forbid")

    @field_validator("email")
    @classmethod
    def _email_format(cls, value: str) -> str:
        return _validate_email(value)


class CSVImportFailure(BaseModel):
    row_number: int
    identifier: str | None = None
    reason: str


class CSVImportResult(BaseModel):
    enrolled: list[UUID]
    created_users: list[UUID]
    failures: list[CSVImportFailure]


class InvitationCodeCreate(BaseModel):
    code: str = Field(min_length=4, max_length=64)
    max_uses: int | None = Field(default=None, gt=0)
    expires_at: datetime | None = None

    model_config = ConfigDict(extra="forbid")


class InvitationCodePatch(BaseModel):
    expires_at: datetime | None = None
    max_uses: int | None = Field(default=None, gt=0)
    is_active: bool | None = None

    model_config = ConfigDict(extra="forbid")


__all__ = [
    "BulkEnrollFailure",
    "BulkEnrollRequest",
    "BulkEnrollResult",
    "CSVImportFailure",
    "CSVImportResult",
    "CSVImportRow",
    "InvitationCodeCreate",
    "InvitationCodePatch",
]
