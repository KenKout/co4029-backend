"""Masking for personal data in aggregate views (PRD ADM-024).

Admin list views exist to spot patterns — which account, which source, how
often — and a pattern is visible from a partial identifier. Full email
addresses and IPs are only needed once an operator has a specific target and a
reason, which is the detail view. Masking in the list therefore costs nothing
operationally and removes a standing bulk-export of personal data from every
screen an admin leaves open.

Masking happens SERVER-SIDE, in the projection, not in the UI. A client-side
mask is decoration: the full value is still in the response body, still in the
browser's memory, still in any HAR file or proxy log, and still one devtools
tab away. If the operator is not entitled to the value, it must not be sent.

The masks keep enough to correlate rows without identifying a person outright:

* ``na***@domain.com`` — the domain survives, so "everyone from one tenant" is
  still visible, and the first two characters let an operator match against a
  full address they already have.
* ``14.224.xxx.xxx`` — the /16 survives, which is what makes "these forty
  failures share a network" readable. The host portion is what identifies a
  subscriber.
"""

from __future__ import annotations

import ipaddress

#: Leading characters of the local part kept in a masked email.
_EMAIL_PREFIX_CHARS = 2


def mask_email(email: str | None) -> str | None:
    """``nam.nguyen@example.com`` -> ``na***@example.com``.

    ``None`` passes through: an absent address is not a redacted one, and the
    caller needs to keep telling those apart.
    """
    if not email:
        return email
    local, separator, domain = email.partition("@")
    if not separator:
        # Not an address shape. Mask the whole thing rather than guessing —
        # whatever it is, it was stored in a field that holds personal data.
        return "***"
    # A one- or two-character local part would otherwise pass through intact.
    prefix = local[:_EMAIL_PREFIX_CHARS] if len(local) > _EMAIL_PREFIX_CHARS else local[:1]
    return f"{prefix}***@{domain}"


def mask_ip(address: str | None) -> str | None:
    """``14.224.10.7`` -> ``14.224.xxx.xxx``; IPv6 keeps its first two groups.

    An unparseable value is fully masked rather than echoed: the field holds
    network identifiers, and something that failed to parse is not thereby
    safe to publish.
    """
    if not address:
        return address
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return "***"
    if parsed.version == 4:
        octets = address.split(".")
        return f"{octets[0]}.{octets[1]}.xxx.xxx"
    groups = parsed.exploded.split(":")
    return ":".join([*groups[:2], *(["xxxx"] * (len(groups) - 2))])


__all__ = ["mask_email", "mask_ip"]
