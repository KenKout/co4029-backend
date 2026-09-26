"""Load-test user minter — creates N synthetic students and prints JWTs.

Adapted from scripts/seed_voice_interview_demo.py (same upsert patterns).
For each user i in 0..N-1 (stable uuid5 ids → idempotent re-runs):

  * users + user_profiles row (status=active, email k6-loadtest-{i}@…)
  * organization_memberships into the course's org (same unit as the
    course owner's membership, so tenant gates pass)
  * org-scoped 'student' role assignment
  * course_enrollments (active) for --course-id
  * auth_sessions row (stable session uuid5) → 30-day JWT via the app's
    own create_access_token (runs with the deployment's JWT secret)

Output: JSON array [{user_id, session_id, access_token}] on STDOUT
(progress logs go to STDERR) — redirect into perf/k6/.state/users.json.

Usage (on the deployment server, from the backend checkout):

  cd /root/co4029/backend && sudo .venv/bin/python /tmp/mint_loadtest_users.py \
      --count 20 --course-id <uuid> [--domain abridgeai.test]

Cleanup (optional, after the evaluation):
  DELETE FROM users WHERE primary_email LIKE 'k6-loadtest-%';  -- cascades
  are NOT configured — see --print-cleanup for the explicit statement list.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from datetime import timedelta
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

NS = NAMESPACE_URL
EMAIL_DOMAIN_DEFAULT = "abridgeai.local"


def uid(i: int) -> str:
    return str(uuid5(NS, f"k6-loadtest-user-{i}"))


def sid(i: int) -> str:
    return str(uuid5(NS, f"k6-loadtest-session-{i}"))


def _async_db_url(url: str) -> str:
    if "+psycopg_async" in url:
        return url
    if url.startswith("postgresql+psycopg://"):
        return url.replace("postgresql+psycopg://", "postgresql+psycopg_async://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    return url


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


async def seed(session, *, count: int, course_id: str, domain: str) -> list[dict]:
    # Resolve the course's org (memberships are the tenant boundary; the
    # schema enforces one live org per user, which is all we need here).
    course = (
        await session.execute(
            text("SELECT organization_id, owner_user_id FROM courses WHERE id = CAST(:c AS uuid)"),
            {"c": course_id},
        )
    ).first()
    if course is None:
        raise SystemExit(f"course {course_id} not found")
    org_id = str(course.organization_id)
    log(f"org={org_id}")

    student_role = (
        await session.execute(text("SELECT id FROM roles WHERE code = 'student' LIMIT 1"))
    ).first()
    if student_role is None:
        raise SystemExit("student role missing — seed the catalog first")

    out = []
    for i in range(count):
        user_id, sess_id, email = uid(i), sid(i), f"k6-loadtest-{i:03d}@{domain}"

        await session.execute(
            text(
                "INSERT INTO users (id, primary_email, status) "
                "VALUES (CAST(:id AS uuid), :email, 'active') "
                "ON CONFLICT (id) DO UPDATE SET primary_email = EXCLUDED.primary_email, status = 'active'"
            ),
            {"id": user_id, "email": email},
        )
        await session.execute(
            text(
                "INSERT INTO user_profiles (user_id, given_name, family_name, display_name) "
                "VALUES (CAST(:uid AS uuid), 'K6', :dn, :dn) "
                "ON CONFLICT (user_id) DO UPDATE SET display_name = EXCLUDED.display_name"
            ),
            {"uid": user_id, "dn": f"k6 loadtest {i:03d}"},
        )
        await session.execute(
            text(
                "INSERT INTO organization_memberships (id, user_id, organization_id, status) "
                "SELECT gen_random_uuid(), CAST(:uid AS uuid), CAST(:oid AS uuid), 'active' "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM organization_memberships WHERE user_id = CAST(:uid AS uuid) "
                "  AND deleted_at IS NULL AND status <> 'left')"
            ),
            {"uid": user_id, "oid": org_id},
        )
        await session.execute(
            text(
                "INSERT INTO user_role_assignments (user_id, role_id, scope_kind, organization_id) "
                "SELECT CAST(:uid AS uuid), :rid, 'organization', CAST(:oid AS uuid) "
                "WHERE NOT EXISTS ("
                "  SELECT 1 FROM user_role_assignments a WHERE a.user_id = CAST(:uid AS uuid) "
                "  AND a.role_id = :rid AND a.scope_kind = 'organization' "
                "  AND a.organization_id = CAST(:oid AS uuid) AND a.deleted_at IS NULL)"
            ),
            {"uid": user_id, "rid": str(student_role.id), "oid": org_id},
        )
        await session.execute(
            text(
                "INSERT INTO course_enrollments (id, course_id, student_id, status, source) "
                "VALUES (gen_random_uuid(), CAST(:cid AS uuid), CAST(:uid AS uuid), 'active', 'manual') "
                "ON CONFLICT (course_id, student_id) DO UPDATE SET status = 'active'"
            ),
            {"cid": course_id, "uid": user_id},
        )
        token_hash = hashlib.sha256(sess_id.encode()).hexdigest()
        await session.execute(
            text(
                "INSERT INTO auth_sessions (id, user_id, refresh_token_hash, expires_at) "
                "VALUES (CAST(:sid AS uuid), CAST(:uid AS uuid), :hash, NOW() + INTERVAL '30 days') "
                "ON CONFLICT (id) DO UPDATE SET expires_at = NOW() + INTERVAL '30 days', revoked_at = NULL"
            ),
            {"sid": sess_id, "uid": user_id, "hash": token_hash},
        )
        out.append({"user_id": user_id, "session_id": sess_id, "email": email})
        log(f"seeded {email}")

    await session.commit()
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--course-id", required=True)
    ap.add_argument("--domain", default=EMAIL_DOMAIN_DEFAULT)
    args = ap.parse_args()

    from abridgeai.core.config import get_settings
    from abridgeai.core.security import create_access_token

    settings = get_settings()
    engine = create_async_engine(_async_db_url(settings.database_url), pool_pre_ping=True)
    Session = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with Session() as session:
        users = await seed(
            session, count=args.count, course_id=args.course_id, domain=args.domain
        )
    await engine.dispose()

    for u in users:
        u["access_token"] = create_access_token(
            user_id=UUID(u["user_id"]),
            session_id=UUID(u["session_id"]),
            expires_delta=timedelta(days=30),
        )
    json.dump(users, sys.stdout, indent=1)
    print(f"minted {len(users)} users", file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
