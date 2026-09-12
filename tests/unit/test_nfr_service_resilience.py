"""Automated NFR checks for graceful optional-service degradation."""

from abridgeai.api.healthz import CheckStatus, _classify_composite


def test_ai_dependency_failure_degrades_without_downing_core_services() -> None:
    checks = {
        "postgres": CheckStatus(status="ok", latency_ms=4.0),
        "redis": CheckStatus(status="ok", latency_ms=2.0),
        "neo4j": CheckStatus(status="disabled", latency_ms=None),
        "garage_s3": CheckStatus(status="ok", latency_ms=8.0),
        "llm": CheckStatus(status="unhealthy", latency_ms=None),
    }

    assert _classify_composite(checks) == "degraded"


def test_disabled_optional_services_keep_core_platform_healthy() -> None:
    checks = {
        "postgres": CheckStatus(status="ok", latency_ms=4.0),
        "redis": CheckStatus(status="ok", latency_ms=2.0),
        "neo4j": CheckStatus(status="disabled", latency_ms=None),
        "garage_s3": CheckStatus(status="disabled", latency_ms=None),
        "llm": CheckStatus(status="skipped", latency_ms=None),
    }

    assert _classify_composite(checks) == "ok"
