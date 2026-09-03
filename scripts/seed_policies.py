"""Seed the policy catalogue from ``scripts/data/policy_seed.json``.

The five policy documents used to live as hardcoded constants in the
frontend (``lib/help-content.ts``). Now that policies are a real entity with
versions, publishers and audiences, that text has to exist as rows before the
reader pages can be pointed at the API — this script performs that import.

Idempotent by slug: a policy that already exists is left completely alone,
including its versions. Re-running after an admin has edited a document must
not quietly revert their work, so this script only ever creates.

Usage::

    cd src/backend
    uv run python scripts/seed_policies.py
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from abridgeai.features.policies import queries as policy_queries
from abridgeai.features.policies import services as policy_services
from abridgeai.features.policies.schemas import (
    PolicyAudienceUpdate,
    PolicyCreate,
    PolicyVersionPatch,
)

SEED_FILE = pathlib.Path(__file__).parent / "data" / "policy_seed.json"


def _async_db_url(url: str) -> str:
    """Ensure the URL uses the psycopg async driver."""
    if "+psycopg_async" in url:
        return url
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return url


async def _publisher_id(db: AsyncSession) -> UUID | None:
    """A platform admin to credit as publisher, or ``None``.

    Attribution matters on a policy — "published by" is shown to readers — so
    prefer a real admin account. ``None`` is tolerated rather than fatal: a
    fresh database may have no admin yet, and an unattributed policy is far
    better than no policy.
    """
    from abridgeai.features.access_control.models import Role, UserRoleAssignment

    stmt = (
        select(UserRoleAssignment.user_id)
        .join(Role, Role.id == UserRoleAssignment.role_id)
        .where(
            Role.code == "admin",
            Role.deleted_at.is_(None),
            UserRoleAssignment.deleted_at.is_(None),
        )
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def _seed_one(db: AsyncSession, spec: dict, actor_id: UUID | None) -> str:
    slug = spec["slug"]
    if await policy_queries.get_policy_by_slug(db, slug) is not None:
        return f"skip   {slug} (already present)"

    detail = await policy_services.create_policy(
        db,
        PolicyCreate(
            slug=slug,
            category=spec["category"],
            title=spec["title"],
            language=spec["language"],
        ),
        actor_id=actor_id,
    )

    # create_policy opens v1 as an empty draft; fill it, then release it.
    draft = detail.versions[0]
    await policy_services.update_draft(
        db,
        draft.id,
        PolicyVersionPatch(body=spec["body"], changelog=spec["changelog"]),
        actor_id=actor_id,
    )
    await policy_services.publish_version(db, draft.id, actor_id=actor_id)

    if spec["audience"]:
        await policy_services.set_audience(
            db,
            detail.id,
            PolicyAudienceUpdate(role_codes=spec["audience"]),
            actor_id=actor_id,
        )

    audience = ", ".join(spec["audience"]) or "public"
    return f"create {slug} v1 published  [{audience}]"


async def main() -> None:
    from abridgeai.core.config import get_settings

    specs = json.loads(SEED_FILE.read_text(encoding="utf-8"))

    engine = create_async_engine(_async_db_url(get_settings().database_url), pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)

    async with session_factory() as db:
        actor_id = await _publisher_id(db)
        if actor_id is None:
            print("warning: no admin account found; policies will be unattributed", file=sys.stderr)
        for spec in specs:
            print(await _seed_one(db, spec, actor_id))
        await db.commit()

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
