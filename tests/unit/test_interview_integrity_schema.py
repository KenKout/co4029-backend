"""Unit tests for interview integrity-event schemas (Phase 4, no DB).

Tests validation of integrity event batches. Ensures event_type, severity,
batch size constraints are enforced correctly. No database required.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from abridgeai.features.interviews.schemas.integrity import (
    IntegrityEventBatchRequest,
    IntegrityEventItem,
    MAX_EVENTS_PER_BATCH,
)


class TestIntegrityEventItem:
    """Single integrity event validation."""

    def test_valid_event_minimum(self):
        """Minimal event with only event_type is valid."""
        event = IntegrityEventItem(event_type="focus_lost")
        assert event.event_type == "focus_lost"
        assert event.severity == "info"  # default
        assert event.metadata == {}

    def test_valid_event_with_severity(self):
        """Event with explicit severity is valid."""
        event = IntegrityEventItem(
            event_type="tab_switch",
            severity="warning",
        )
        assert event.event_type == "tab_switch"
        assert event.severity == "warning"

    def test_valid_event_with_metadata(self):
        """Event with metadata dict is valid."""
        event = IntegrityEventItem(
            event_type="reconnect",
            severity="info",
            metadata={"latency_ms": 150, "attempt": 2},
        )
        assert event.metadata == {"latency_ms": 150, "attempt": 2}

    def test_valid_all_event_types(self):
        """All allowed event types are accepted."""
        event_types = [
            "focus_lost",
            "tab_switch",
            "fullscreen_exit",
            "warning_issued",
            "reconnect",
            "disconnect",
        ]
        for et in event_types:
            event = IntegrityEventItem(event_type=et)
            assert event.event_type == et

    def test_valid_all_severity_levels(self):
        """All allowed severity levels are accepted."""
        severities = ["info", "warning", "critical"]
        for sev in severities:
            event = IntegrityEventItem(event_type="focus_lost", severity=sev)
            assert event.severity == sev

    def test_invalid_event_type(self):
        """Invalid event_type raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IntegrityEventItem(event_type="invalid_type")
        assert "event_type" in str(exc_info.value).lower()

    def test_invalid_severity(self):
        """Invalid severity raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IntegrityEventItem(event_type="focus_lost", severity="urgent")
        assert "severity" in str(exc_info.value).lower()

    def test_default_severity_is_info(self):
        """Severity defaults to 'info' if not provided."""
        event = IntegrityEventItem(event_type="focus_lost")
        assert event.severity == "info"

    def test_metadata_accepts_various_scalar_types(self):
        """Metadata can contain strings, ints, floats, bools."""
        metadata = {
            "name": "test",
            "count": 42,
            "ratio": 0.95,
            "flag": True,
        }
        event = IntegrityEventItem(
            event_type="focus_lost",
            metadata=metadata,
        )
        assert event.metadata == metadata


class TestIntegrityEventBatchRequest:
    """Batch request validation."""

    def test_valid_single_event(self):
        """Batch with one event is valid."""
        batch = IntegrityEventBatchRequest(
            events=[IntegrityEventItem(event_type="focus_lost")]
        )
        assert len(batch.events) == 1

    def test_valid_multiple_events(self):
        """Batch with multiple events is valid."""
        events = [
            IntegrityEventItem(event_type="focus_lost"),
            IntegrityEventItem(event_type="tab_switch", severity="warning"),
            IntegrityEventItem(event_type="reconnect"),
        ]
        batch = IntegrityEventBatchRequest(events=events)
        assert len(batch.events) == 3

    def test_valid_max_events(self):
        """Batch with exactly MAX_EVENTS_PER_BATCH events is valid."""
        events = [
            IntegrityEventItem(event_type="focus_lost")
            for _ in range(MAX_EVENTS_PER_BATCH)
        ]
        batch = IntegrityEventBatchRequest(events=events)
        assert len(batch.events) == MAX_EVENTS_PER_BATCH

    def test_empty_batch_rejected(self):
        """Batch with zero events raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IntegrityEventBatchRequest(events=[])
        assert "at least 1 item" in str(exc_info.value).lower()

    def test_batch_exceeds_max_events(self):
        """Batch with >MAX_EVENTS_PER_BATCH events raises ValidationError."""
        events = [
            IntegrityEventItem(event_type="focus_lost")
            for _ in range(MAX_EVENTS_PER_BATCH + 1)
        ]
        with pytest.raises(ValidationError) as exc_info:
            IntegrityEventBatchRequest(events=events)
        assert "at most" in str(exc_info.value).lower() or "max" in str(exc_info.value).lower()

    def test_invalid_event_in_batch(self):
        """Batch containing invalid event raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            IntegrityEventBatchRequest(
                events=[
                    IntegrityEventItem(event_type="focus_lost"),
                    IntegrityEventItem(event_type="invalid_type"),  # invalid
                ]
            )
        assert "event_type" in str(exc_info.value).lower()

    def test_batch_dict_construction(self):
        """Batch can be constructed from dict with event dicts."""
        batch = IntegrityEventBatchRequest(
            events=[
                {"event_type": "focus_lost"},
                {"event_type": "tab_switch", "severity": "warning"},
                {"event_type": "reconnect", "metadata": {"code": 200}},
            ]
        )
        assert len(batch.events) == 3
        assert batch.events[0].severity == "info"
        assert batch.events[1].severity == "warning"
        assert batch.events[2].metadata == {"code": 200}

    def test_batch_respects_max_cap(self):
        """MAX_EVENTS_PER_BATCH is enforced at 50."""
        # This is a validation of the constant itself
        assert MAX_EVENTS_PER_BATCH == 50

        # Batch with 51 events should fail
        events = [
            IntegrityEventItem(event_type="focus_lost")
            for _ in range(51)
        ]
        with pytest.raises(ValidationError):
            IntegrityEventBatchRequest(events=events)
