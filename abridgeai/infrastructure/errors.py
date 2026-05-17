"""Typed S3 / object-storage errors for the service layer.

Service code catches these instead of raw ``botocore.exceptions.ClientError``
so the boundary stays HTTP-agnostic and the API layer can map them to
status codes (404, 403, 502, 503).
"""

from __future__ import annotations

from abridgeai.core.exceptions import AppError


class S3Error(AppError):
    """Base class for object-storage failures."""


class S3NotFoundError(S3Error):
    """The requested object does not exist."""


class S3PermissionError(S3Error):
    """S3 rejected the request as forbidden (bad creds / bucket policy)."""


class S3UploadError(S3Error):
    """Generic upload / multipart failure (network, 5xx, parse error)."""


class S3NotConfiguredError(S3Error):
    """Settings.aws_access_key_id / secret / bucket are missing.

    Raised eagerly at the call site instead of silently substituting a
    placeholder URL — production must have real S3, tests must use ``moto``.
    """


__all__ = [
    "S3Error",
    "S3NotConfiguredError",
    "S3NotFoundError",
    "S3PermissionError",
    "S3UploadError",
]
