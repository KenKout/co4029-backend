"""Pure helpers for quiz access-rule enforcement (Phase 12).

Security notes:
- Passwords are compared in constant time (secrets.compare_digest).
- Subnet strings are parsed with stdlib ipaddress and MUST reject malformed
  input at teacher-save time so a typo cannot silently lock out or admit
  everyone.
- Never log the stored/submitted password or the client IP alongside the
  password.
"""

from __future__ import annotations

import ipaddress
import secrets

IpNetwork = ipaddress.IPv4Network | ipaddress.IPv6Network


def parse_subnet_allowlist(raw: str | None) -> list[IpNetwork]:
    """Parse a comma-separated CIDR/IP allowlist into networks.

    Raises ``ValueError`` if any non-empty entry is malformed.
    """
    if not raw or not raw.strip():
        return []
    nets: list[IpNetwork] = []
    for part in raw.split(","):
        entry = part.strip()
        if not entry:
            continue
        # strict=False lets a bare host or a network with host bits set parse.
        nets.append(ipaddress.ip_network(entry, strict=False))
    return nets


def ip_in_allowlist(client_ip: str | None, allowlist: list[IpNetwork]) -> bool:
    """True if ``client_ip`` falls inside any allowlisted network.

    A missing/unparseable client IP is treated as NOT allowed (fail closed).
    """
    if not client_ip:
        return False
    try:
        addr = ipaddress.ip_address(client_ip.strip())
    except ValueError:
        return False
    return any(addr in net for net in allowlist)


def password_matches(stored: str | None, submitted: str | None) -> bool:
    """Constant-time compare of the quiz password against a submission.

    ``stored`` empty/None means no password is configured — callers gate on
    that separately, so this returns True (nothing to match).
    """
    if not stored:
        return True
    if submitted is None:
        return False
    return secrets.compare_digest(stored, submitted)


__all__ = ["ip_in_allowlist", "parse_subnet_allowlist", "password_matches"]
