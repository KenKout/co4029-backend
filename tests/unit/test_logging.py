from __future__ import annotations

import json
import logging

import pytest
import structlog

from abridgeai.core.config import get_settings
from abridgeai.core.observability import (
    bind_request_context,
    clear_request_context,
    configure_structlog,
    get_logger,
)
from abridgeai.core.observability.logging import _reset_for_tests


@pytest.fixture(autouse=True)
def _isolated_logging():
    """Wipe configure-once flag and contextvars between tests."""
    _reset_for_tests()
    get_settings.cache_clear()
    yield
    _reset_for_tests()
    get_settings.cache_clear()


def _set_format(monkeypatch: pytest.MonkeyPatch, fmt: str, level: str = "INFO") -> None:
    monkeypatch.setenv("LOG_FORMAT", fmt)
    monkeypatch.setenv("LOG_LEVEL", level)
    get_settings.cache_clear()


def test_json_renderer_emits_parseable_line(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_format(monkeypatch, "json")
    log = get_logger("abridgeai.test")

    log.info("user_signed_in", user_id="u-123", status="ok")

    captured = capsys.readouterr().out.strip().splitlines()
    assert captured, "expected at least one log line on stdout"
    payload = json.loads(captured[-1])
    assert payload["event"] == "user_signed_in"
    assert payload["user_id"] == "u-123"
    assert payload["status"] == "ok"
    assert payload["level"] == "info"
    assert "timestamp" in payload


def test_console_renderer_is_human_readable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_format(monkeypatch, "console")
    log = get_logger("abridgeai.test")

    log.info("dev_event", request_id="abc")

    out = capsys.readouterr().out
    assert "dev_event" in out
    assert "request_id" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out.strip().splitlines()[-1])


def test_contextvars_merge_into_event(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_format(monkeypatch, "json")
    log = get_logger("abridgeai.test")

    bind_request_context(request_id="req-42", user_id="u-7")
    try:
        log.info("during_request")
    finally:
        clear_request_context()
    log.info("after_clear")

    lines = capsys.readouterr().out.strip().splitlines()
    during = json.loads(lines[-2])
    after = json.loads(lines[-1])
    assert during["request_id"] == "req-42"
    assert during["user_id"] == "u-7"
    assert "request_id" not in after
    assert "user_id" not in after


def test_configure_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_format(monkeypatch, "json")
    configure_structlog()
    handler_count_after_first = len(logging.getLogger().handlers)

    for _ in range(5):
        configure_structlog()

    assert len(logging.getLogger().handlers) == handler_count_after_first


def test_stdlib_bridge_routes_through_structlog(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _set_format(monkeypatch, "json")
    configure_structlog()

    logging.getLogger("abridgeai.cache.test").warning("legacy_event %s", "payload")

    out = capsys.readouterr().out.strip().splitlines()
    assert out, "stdlib log should reach the structlog handler"
    payload = json.loads(out[-1])
    assert "legacy_event" in payload["event"]
    assert payload["level"] == "warning"


def test_noisy_libraries_are_silenced(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_format(monkeypatch, "json")
    configure_structlog()

    for noisy in ("httpx", "httpcore", "asyncio", "urllib3", "boto3"):
        assert logging.getLogger(noisy).level >= logging.WARNING


def test_get_logger_returns_bound_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_format(monkeypatch, "json")
    log = get_logger("abridgeai.smoke").bind()
    assert isinstance(log, structlog.stdlib.BoundLogger)


def test_relayed_structlog_record_still_formats(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A record formatted in one process and re-dispatched in another must print.

    livekit-agents relays its job subprocess' logs by formatting the record,
    replacing ``msg`` with the resulting string and pickling it
    (``ipc/log_queue.py``), then re-dispatching in the parent. Both of structlog's
    ``_logger`` / ``_name`` markers survive that trip, so without the repair
    filter ``ProcessorFormatter`` calls ``.copy()`` on a ``str`` and every agent
    log line becomes an ``AttributeError`` traceback.
    """
    _set_format(monkeypatch, "json")
    log = get_logger("abridgeai.test")

    grabbed: list[logging.LogRecord] = []

    class _Grab(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            grabbed.append(record)

    logging.getLogger("abridgeai.test").addHandler(_Grab())
    log.info("voice.agent_dispatch", native=True)
    assert grabbed, "expected the emitted record to be observable"

    handler = logging.getLogger().handlers[0]
    relayed = logging.makeLogRecord(grabbed[0].__dict__)
    relayed.msg = handler.format(grabbed[0])
    relayed.args = None

    assert hasattr(relayed, "_logger"), "structlog's marker must survive the relay"

    errors: list[logging.LogRecord] = []
    monkeypatch.setattr(handler, "handleError", errors.append)
    logging.getLogger(relayed.name).callHandlers(relayed)

    assert not errors, "the relayed record failed to format"
    assert capsys.readouterr().out.strip(), "the relayed record produced no output"
