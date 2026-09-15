"""LiveKit webhook receiver — raw-body, signature-verified Egress callbacks.

Deliberately OUTSIDE the ``/api/v1`` authenticated surface: LiveKit delivers
these server-to-server with an HMAC-signed JWT in the ``Authorization``
header, not a user token. Verification therefore REPLACES user auth:

* missing / malformed signature header → 401, event never decoded;
* invalid signature or stale timestamp (beyond the skew leeway) → 401;
* verified → the event is handed to ``services.recording`` which persists
  idempotently. Duplicate and out-of-order deliveries are no-ops.

The endpoint ACKs (200) even for events about rooms/egresses it does not
know — a foreign room sharing this LiveKit project must not make LiveKit
retry the delivery forever. Only signature failures are rejected.

Route shape: registered directly on the app root (see ``api/__init__``) as
``POST /internal/livekit/webhook``. No feature-scoped prefix keeps the
provider contract URL stable across feature refactors.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated

from fastapi import APIRouter, Header, HTTPException, Request, Response, status

from abridgeai.core.config import get_settings
from abridgeai.features.interviews.services import recording as recording_service

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

router = APIRouter(tags=["internal-livekit"])

logger = logging.getLogger(__name__)


@router.post("/internal/livekit/webhook")
async def livekit_webhook(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> Response:
    """Receive one LiveKit webhook delivery (Egress lifecycle events)."""
    settings = get_settings()
    if not settings.livekit_webhook_enabled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "not_found"},
        )

    body = (await request.body()).decode("utf-8", errors="replace")
    try:
        event = recording_service.verify_webhook_signature(body, authorization)
    except ValueError as exc:
        logger.warning(
            "interview.recording.webhook_rejected",
            extra={"reason": str(exc)},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"error": "invalid_signature"},
        ) from exc

    # Its own short-lived session: the callback must be persisted even though
    # no request-scoped ``get_db`` dependency exists on an unauthenticated
    # route, and one bad event must not poison a shared session.
    sessionmaker = _get_sessionmaker()
    try:
        async with sessionmaker() as db:
            verb = await recording_service.handle_egress_webhook(db, event)
            await db.commit()
    except Exception:  # noqa: BLE001 -- log and ACK: retries cannot fix a bug
        logger.exception("interview.recording.webhook_processing_failed")
        return Response(status_code=status.HTTP_200_OK)
    logger.info(
        "interview.recording.webhook_accepted",
        extra={"event": verb},
    )
    return Response(status_code=status.HTTP_200_OK)


def _get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    from abridgeai.core.db import get_sessionmaker  # noqa: PLC0415

    return get_sessionmaker()


__all__ = ["livekit_webhook", "router"]
