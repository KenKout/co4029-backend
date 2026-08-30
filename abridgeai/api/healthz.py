"""Health check endpoints (T0.26).

Three endpoints for three consumers, each with a deliberate scope:

* ``GET /healthz`` -- lightweight liveness probe. Returns 200 always with
  no dependency checks. Target latency < 50ms. Used by load balancers and
  the audit-log middleware exemption list (see
  ``core/observability/audit_log.py``).
* ``GET /healthz/deep`` -- composite dependency-status report. Admin-only
  (requires ``system.administer``). Reports per-check status + latency for
  postgres / redis / neo4j / garage_s3 / llm_provider plus the deployed
  version + git sha.
* ``GET /readyz`` -- Kubernetes readiness probe. Returns 200 only if
  postgres + redis are reachable AND the alembic schema is at head.
  Returns 503 otherwise. No auth (K8s probe path).

Status semantics for ``/healthz/deep``:

* ``ok``        -- all critical checks (postgres, redis) ok.
* ``degraded``  -- a non-critical check fails (neo4j, garage_s3, llm)
                   but postgres + redis are still ok.
* ``unhealthy`` -- postgres OR redis is down.

Per-check timeout is 3s; composite timeout is 10s. Each check runs
concurrently via ``asyncio.gather`` with ``asyncio.wait_for`` bounding
the slow path.

Implementation notes:

* The five ``_check_*`` helpers are top-level so tests can monkeypatch
  any of them to drive the composite/readyz code paths without touching
  real infrastructure.
* ``_get_version`` and ``_get_git_sha`` are ``@cache``-decorated so the
  subprocess + import cost is paid once per process, not per request.
* The Neo4j / Garage checks fall through to "disabled" when
  the corresponding settings are absent, never "unhealthy" -- a
  feature being intentionally off must not flip the overall status. The
  LLM check defaults to "skipped" and only pings the provider when
  "LLM_HEALTH_PING" opts in (see ``_check_llm_provider``).
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from collections.abc import Awaitable
from functools import cache
from pathlib import Path
from typing import Annotated, Literal

import httpx
from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from abridgeai.core.cache.client import get_cache
from abridgeai.core.config import get_settings
from abridgeai.core.db import get_sessionmaker
from abridgeai.core.security import CurrentUser
from abridgeai.features.access_control.policies import require_permission

logger = logging.getLogger(__name__)

router = APIRouter(tags=["health"])

PER_CHECK_TIMEOUT_S = 3.0
COMPOSITE_TIMEOUT_S = 10.0
_FALLBACK_VERSION = "0.1.0"


CheckStatusLiteral = Literal["ok", "unhealthy", "disabled", "skipped"]
DeepStatusLiteral = Literal["ok", "degraded", "unhealthy"]


class CheckStatus(BaseModel):
    """Per-dependency probe result."""

    status: CheckStatusLiteral
    latency_ms: float | None = None


class DeepHealthOut(BaseModel):
    """Composite payload returned by ``GET /healthz/deep``."""

    status: DeepStatusLiteral
    checks: dict[str, CheckStatus]
    version: str
    git_sha: str | None


@cache
def _get_version() -> str:
    """Resolve the running version, preferring ``abridgeai.__version__``.

    Falls back to a hard-coded sentinel so missing metadata never breaks
    the health endpoint.
    """
    import abridgeai

    pkg_version = getattr(abridgeai, "__version__", None)
    if pkg_version is None:
        return _FALLBACK_VERSION
    rendered = str(pkg_version).strip()
    return rendered or _FALLBACK_VERSION


@cache
def _get_git_sha() -> str | None:
    """Return the short git SHA of the deployed checkout, or ``None``.

    Resolution order:
      1. ``GIT_SHA`` env var (typically set by CI / image build).
      2. ``git rev-parse --short HEAD`` against the backend root.
      3. ``None`` if neither is available (e.g. running from a tarball).
    """
    env_sha = os.environ.get("GIT_SHA")
    if env_sha:
        return env_sha[:12]
    backend_root = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.check_output(  # noqa: S603 - hard-coded argv, no shell
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 - command lookup is fine
            cwd=str(backend_root),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    sha = out.strip()
    return sha or None


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 2)


async def _check_postgres() -> CheckStatus:
    """Probe Postgres with a single ``SELECT 1`` round-trip."""
    start = time.perf_counter()
    try:
        sm = get_sessionmaker()
        async with sm() as session:
            await asyncio.wait_for(
                session.execute(text("SELECT 1")),
                timeout=PER_CHECK_TIMEOUT_S,
            )
    except Exception as exc:  # noqa: BLE001 - probe must classify any failure as unhealthy
        logger.warning("healthz.postgres_unhealthy", extra={"err": repr(exc)})
        return CheckStatus(status="unhealthy", latency_ms=_ms_since(start))
    return CheckStatus(status="ok", latency_ms=_ms_since(start))


async def _check_redis() -> CheckStatus:
    """Probe Redis with a single ``PING``."""
    start = time.perf_counter()
    try:
        client = get_cache()
        await asyncio.wait_for(client.ping(), timeout=PER_CHECK_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        logger.warning("healthz.redis_unhealthy", extra={"err": repr(exc)})
        return CheckStatus(status="unhealthy", latency_ms=_ms_since(start))
    return CheckStatus(status="ok", latency_ms=_ms_since(start))


async def _check_neo4j() -> CheckStatus:
    """Probe Neo4j via ``verify_connectivity`` -- ``disabled`` if not configured."""
    settings = get_settings()
    if not settings.knowledge_graph_enabled or not settings.neo4j_uri:
        return CheckStatus(status="disabled", latency_ms=None)
    start = time.perf_counter()
    try:
        from abridgeai.infrastructure.neo4j import get_neo4j_driver

        driver = get_neo4j_driver()
        await asyncio.wait_for(driver.verify_connectivity(), timeout=PER_CHECK_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001
        logger.warning("healthz.neo4j_unhealthy", extra={"err": repr(exc)})
        return CheckStatus(status="unhealthy", latency_ms=_ms_since(start))
    return CheckStatus(status="ok", latency_ms=_ms_since(start))


async def _check_garage_s3() -> CheckStatus:
    """Probe Garage / S3 via ``HeadBucket`` -- ``disabled`` if creds missing."""
    settings = get_settings()
    if not (settings.aws_access_key_id and settings.aws_secret_access_key):
        return CheckStatus(status="disabled", latency_ms=None)
    start = time.perf_counter()
    try:
        import aioboto3
        from botocore.config import Config

        endpoint = settings.aws_endpoint_url or None
        config = Config(
            signature_version="s3v4",
            s3={"addressing_style": "path" if endpoint else "virtual"},
            region_name=settings.aws_region,
        )
        access_key = settings.aws_access_key_id.get_secret_value()
        secret_key = settings.aws_secret_access_key.get_secret_value()
        async with aioboto3.Session().client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=settings.aws_region,
            config=config,
        ) as client:
            await asyncio.wait_for(
                client.head_bucket(Bucket=settings.s3_bucket_name),
                timeout=PER_CHECK_TIMEOUT_S,
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("healthz.s3_unhealthy", extra={"err": repr(exc)})
        return CheckStatus(status="unhealthy", latency_ms=_ms_since(start))
    return CheckStatus(status="ok", latency_ms=_ms_since(start))


async def _check_llm_provider() -> CheckStatus:
    """LLM probe: skipped unless ``LLM_HEALTH_PING`` opts in.

    A live ping costs the provider quota on every health view, so the
    default is a deliberate no-op that reports ``skipped`` — dashboards
    render it as "Not checked". With ``LLM_HEALTH_PING=1`` the probe
    issues a cheap ``GET {base}/models`` (a capabilities list, never a
    generation) against the same credential the app uses, bounded by the
    same per-check timeout as every other probe. Missing API key keeps
    the check ``skipped`` with a warning instead of failing.
    """
    settings = get_settings()
    if not os.environ.get("LLM_HEALTH_PING") or not settings.llm_api_key:
        return CheckStatus(status="skipped", latency_ms=None)
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=PER_CHECK_TIMEOUT_S) as client:
            resp = await client.get(
                f"{settings.llm_base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {settings.llm_api_key}"},
            )
    except httpx.HTTPError as exc:
        logger.warning("healthz.llm_ping_failed: %s", exc)
        return CheckStatus(status="unhealthy", latency_ms=None)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)
    if resp.status_code >= 400:
        logger.warning("healthz.llm_ping_http_%s", resp.status_code)
        return CheckStatus(status="unhealthy", latency_ms=latency_ms)
    return CheckStatus(status="ok", latency_ms=latency_ms)


def _check_alembic_at_head_sync() -> bool:
    """Return ``True`` iff the live DB revision matches the alembic head.

    Runs blocking SQLAlchemy + alembic APIs; callers must wrap with
    ``asyncio.to_thread`` to keep the event loop unblocked.
    """
    try:
        from alembic.config import Config
        from alembic.runtime.migration import MigrationContext
        from alembic.script import ScriptDirectory
        from sqlalchemy import create_engine

        backend_root = Path(__file__).resolve().parents[2]
        cfg = Config(str(backend_root / "alembic.ini"))
        cfg.set_main_option("script_location", str(backend_root / "migrations"))
        script = ScriptDirectory.from_config(cfg)
        head = script.get_current_head()

        settings = get_settings()
        sync_url = settings.database_url.replace("+psycopg_async", "+psycopg")
        engine = create_engine(sync_url)
        try:
            with engine.connect() as connection:
                ctx = MigrationContext.configure(connection)
                current = ctx.get_current_revision()
        finally:
            engine.dispose()
    except Exception as exc:  # noqa: BLE001
        logger.warning("healthz.alembic_check_failed", extra={"err": repr(exc)})
        return False
    return bool(current == head)


async def _check_alembic_at_head() -> bool:
    return await asyncio.to_thread(_check_alembic_at_head_sync)


@router.get("/healthz", status_code=status.HTTP_200_OK)
async def healthz_simple() -> dict[str, str]:
    """Lightweight liveness probe. Always 200, no dependency checks."""
    return {"status": "ok"}


@router.get("/healthz/deep", response_model=DeepHealthOut)
async def healthz_deep(
    _admin: Annotated[CurrentUser, Depends(require_permission("system.administer"))],
) -> DeepHealthOut:
    """Composite dependency-status report. Admin-only.

    Each check runs concurrently with a 3s individual ceiling; the whole
    response is bounded at 10s. A check that exceeds either ceiling is
    reported ``unhealthy`` so a single slow dependency cannot starve the
    probe.
    """

    async def _bounded(name: str, factory: Awaitable[CheckStatus]) -> tuple[str, CheckStatus]:
        try:
            res = await asyncio.wait_for(factory, timeout=PER_CHECK_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "healthz.check_timeout_or_error", extra={"check": name, "err": repr(exc)}
            )
            return name, CheckStatus(status="unhealthy", latency_ms=None)
        return name, res

    coros = [
        _bounded("postgres", _check_postgres()),
        _bounded("redis", _check_redis()),
        _bounded("neo4j", _check_neo4j()),
        _bounded("garage_s3", _check_garage_s3()),
        _bounded("llm_provider", _check_llm_provider()),
    ]
    try:
        results: list[tuple[str, CheckStatus]] = await asyncio.wait_for(
            asyncio.gather(*coros, return_exceptions=False),
            timeout=COMPOSITE_TIMEOUT_S,
        )
    except TimeoutError:
        logger.warning("healthz.deep_composite_timeout")
        results = [
            (k, CheckStatus(status="unhealthy", latency_ms=None))
            for k in ("postgres", "redis", "neo4j", "garage_s3", "llm_provider")
        ]

    checks = {name: status_ for name, status_ in results}
    composite = _classify_composite(checks)
    return DeepHealthOut(
        status=composite,
        checks=checks,
        version=_get_version(),
        git_sha=_get_git_sha(),
    )


def _classify_composite(checks: dict[str, CheckStatus]) -> DeepStatusLiteral:
    """Reduce per-check statuses to ``ok | degraded | unhealthy``."""
    pg = checks.get("postgres", CheckStatus(status="unhealthy")).status
    rd = checks.get("redis", CheckStatus(status="unhealthy")).status
    if pg == "unhealthy" or rd == "unhealthy":
        return "unhealthy"
    if any(c.status == "unhealthy" for c in checks.values()):
        return "degraded"
    return "ok"


@router.get("/readyz")
async def readyz(response: Response) -> dict[str, object]:
    """Kubernetes readiness probe.

    Returns 200 iff postgres + redis are reachable AND the alembic schema
    is at head; 503 otherwise. No auth required so kubelet can hit it.
    """
    pg = await _check_postgres()
    rd = await _check_redis()
    at_head = await _check_alembic_at_head()
    if pg.status != "ok" or rd.status != "ok" or not at_head:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "postgres": pg.status,
        "redis": rd.status,
        "alembic_at_head": at_head,
    }


__all__ = [
    "CheckStatus",
    "DeepHealthOut",
    "router",
]
