"""Integration tests for the T0.28 CORS middleware wiring."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from abridgeai.core.config import get_settings


@pytest.fixture
def cors_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:5173,https://app.abridgeai.com")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 64)
    get_settings.cache_clear()
    from abridgeai.api import create_app

    app = create_app()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        get_settings.cache_clear()


@pytest.fixture
def no_cors_client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("ALLOWED_ORIGINS", "")
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("JWT_SECRET_KEY", "x" * 64)
    get_settings.cache_clear()
    from abridgeai.api import create_app

    app = create_app()
    try:
        with TestClient(app) as c:
            yield c
    finally:
        get_settings.cache_clear()


def test_cors_preflight_allowed_origin(cors_client: TestClient) -> None:
    r = cors_client.options(
        "/healthz",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("Access-Control-Allow-Origin") == "http://localhost:5173"


def test_cors_preflight_disallowed_origin(cors_client: TestClient) -> None:
    r = cors_client.options(
        "/healthz",
        headers={
            "Origin": "http://evil.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )
    assert "Access-Control-Allow-Origin" not in r.headers


def test_cors_credentials_allowed(cors_client: TestClient) -> None:
    r = cors_client.options(
        "/healthz",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )
    assert r.headers.get("Access-Control-Allow-Credentials") == "true"


def test_cors_no_wildcard_with_credentials(cors_client: TestClient) -> None:
    r = cors_client.options(
        "/healthz",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        },
    )
    assert r.headers.get("Access-Control-Allow-Origin") != "*"


def test_cors_methods_allowed(cors_client: TestClient) -> None:
    r = cors_client.options(
        "/healthz",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization,Content-Type",
        },
    )
    assert r.status_code == 200
    allow_methods = r.headers.get("Access-Control-Allow-Methods", "")
    assert "POST" in allow_methods


def test_cors_no_origins_configured_fallback(no_cors_client: TestClient) -> None:
    r = no_cors_client.get("/healthz", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 200
    assert "Access-Control-Allow-Origin" not in r.headers
