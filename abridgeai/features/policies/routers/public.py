"""Public policy reads.

Deliberately unauthenticated, and that is load-bearing: someone who cannot
sign in is exactly who needs the terms, and a user must be able to read them
BEFORE creating an account. The front end pins the same guarantee in
``help-policy.test.tsx``.

Audience scoping applies to the INDEX, not to a document reached by its own
slug. These links are shared, emailed and bookmarked; a policy that 404s for
the person it governs would be a worse failure than showing them a document
they are not a named party to — and the text is public either way.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from abridgeai.core.db import get_db
from abridgeai.core.exceptions import NotFoundError
from abridgeai.features.policies import services as policy_service
from abridgeai.features.policies.schemas import PolicyDocument, PolicySummary

router = APIRouter(prefix="/policies", tags=["policies"])


@router.get("", response_model=list[PolicySummary])
async def list_policies_endpoint(
    db: Annotated[AsyncSession, Depends(get_db)],
    role: Annotated[
        list[str] | None,
        Query(description="Reader's role codes; omit for the public set."),
    ] = None,
    language: str = policy_service.DEFAULT_LANGUAGE,
) -> list[PolicySummary]:
    """Policies this reader is a party to, plus every public one.

    ``role`` is supplied by the client from the signed-in user's own roles.
    That is safe precisely BECAUSE the documents are public: the parameter
    widens a courtesy filter, it does not unlock anything. Nothing here is
    gated on it, so a forged value reveals no more than reading the slugs.
    """
    return await policy_service.list_documents(db, role_codes=role, language=language)


@router.get("/{slug}", response_model=PolicyDocument)
async def get_policy_endpoint(
    slug: str,
    db: Annotated[AsyncSession, Depends(get_db)],
    language: str = policy_service.DEFAULT_LANGUAGE,
) -> PolicyDocument:
    try:
        return await policy_service.read_document(db, slug, language=language)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found", "message": str(exc)},
        ) from exc


__all__ = ["router"]
