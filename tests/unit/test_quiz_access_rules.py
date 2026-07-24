"""Unit tests for Phase 12 quiz access-rule helpers."""

from __future__ import annotations

import pytest

from abridgeai.features.quizzes.queries.access_rules import (
    ip_in_allowlist,
    parse_subnet_allowlist,
    password_matches,
)


def test_parse_subnet_allowlist_accepts_cidr_and_bare_ip():
    nets = parse_subnet_allowlist("10.0.0.0/8, 192.168.1.5 , 2001:db8::/32")
    assert len(nets) == 3


def test_parse_subnet_allowlist_empty_is_empty_list():
    assert parse_subnet_allowlist(None) == []
    assert parse_subnet_allowlist("   ") == []


def test_parse_subnet_allowlist_rejects_malformed():
    with pytest.raises(ValueError):
        parse_subnet_allowlist("10.0.0.0/8, not-an-ip")
    with pytest.raises(ValueError):
        parse_subnet_allowlist("999.1.1.1")


def test_ip_in_allowlist_matches_cidr():
    nets = parse_subnet_allowlist("10.0.0.0/8, 192.168.1.5")
    assert ip_in_allowlist("10.4.5.6", nets) is True
    assert ip_in_allowlist("192.168.1.5", nets) is True
    assert ip_in_allowlist("172.16.0.1", nets) is False


def test_ip_in_allowlist_bad_client_ip_is_false():
    nets = parse_subnet_allowlist("10.0.0.0/8")
    assert ip_in_allowlist(None, nets) is False
    assert ip_in_allowlist("garbage", nets) is False


def test_password_matches_constant_time_correct():
    assert password_matches("hunter2", "hunter2") is True
    assert password_matches("hunter2", "wrong") is False
    assert password_matches("hunter2", None) is False
