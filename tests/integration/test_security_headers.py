"""Integration tests for the T0.28 security-headers middleware."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from abridgeai.core.config import get_settings
from abridgeai.core.security.headers import SecurityHeadersMiddleware


def _build_app(environment: str = "production") -> FastAPI:
    app = FastAPI()
    app.add_middleware(SecurityHeadersMiddleware, environment=environment)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/login")
    def login() -> dict[str, str]:
        return {"token": "dummy"}

    @app.get("/quiz")
    def quiz() -> dict[str, str]:
        return {"id": "q1"}

    return app


@pytest.fixture
def prod_client() -> Iterator[TestClient]:
    with TestClient(_build_app(environment="production")) as c:
        yield c


@pytest.fixture
def dev_client() -> Iterator[TestClient]:
    with TestClient(_build_app(environment="dev")) as c:
        yield c


def test_strict_transport_security_present(prod_client: TestClient) -> None:
    r = prod_client.get("/healthz")
    assert r.status_code == 200
    assert r.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"


def test_x_content_type_options_present(prod_client: TestClient) -> None:
    r = prod_client.get("/healthz")
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_x_frame_options_deny(prod_client: TestClient) -> None:
    r = prod_client.get("/healthz")
    assert r.headers["X-Frame-Options"] == "DENY"


def test_referrer_policy_present(prod_client: TestClient) -> None:
    r = prod_client.get("/healthz")
    assert r.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"


def test_permissions_policy_present(prod_client: TestClient) -> None:
    r = prod_client.get("/healthz")
    assert "geolocation=()" in r.headers["Permissions-Policy"]
    assert "microphone=()" in r.headers["Permissions-Policy"]
    assert "camera=()" in r.headers["Permissions-Policy"]


def test_csp_present(prod_client: TestClient) -> None:
    r = prod_client.get("/healthz")
    assert "default-src 'self'" in r.headers["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]


def test_csp_no_unsafe_inline_in_production(prod_client: TestClient) -> None:
    r = prod_client.get("/healthz")
    csp = r.headers["Content-Security-Policy"]
    script_src = next(part for part in csp.split(";") if part.strip().startswith("script-src"))
    assert "'unsafe-inline'" not in script_src
    assert "'unsafe-eval'" not in script_src


def test_csp_relaxed_in_dev(dev_client: TestClient) -> None:
    r = dev_client.get("/healthz")
    csp = r.headers["Content-Security-Policy"]
    script_src = next(part for part in csp.split(";") if part.strip().startswith("script-src"))
    assert "'unsafe-inline'" in script_src
    assert "https://cdn.jsdelivr.net" in script_src


def test_security_headers_on_all_endpoints(prod_client: TestClient) -> None:
    for method, path in [
        ("GET", "/healthz"),
        ("POST", "/login"),
        ("GET", "/quiz"),
    ]:
        r = prod_client.request(method, path)
        assert r.headers["Strict-Transport-Security"] == "max-age=31536000; includeSubDomains"
        assert r.headers["X-Content-Type-Options"] == "nosniff"
        assert r.headers["X-Frame-Options"] == "DENY"
        assert r.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
        assert "Permissions-Policy" in r.headers
        assert "Content-Security-Policy" in r.headers


def test_real_app_emits_security_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("ALLOWED_ORIGINS", "")
    get_settings.cache_clear()
    try:
        from abridgeai.api import create_app

        app = create_app()
        with TestClient(app) as client:
            r = client.get("/healthz")
            assert r.status_code == 200
            assert r.headers["X-Content-Type-Options"] == "nosniff"
            assert r.headers["X-Frame-Options"] == "DENY"
            csp = r.headers["Content-Security-Policy"]
            script_src = next(p for p in csp.split(";") if p.strip().startswith("script-src"))
            assert "'unsafe-inline'" not in script_src
    finally:
        get_settings.cache_clear()
