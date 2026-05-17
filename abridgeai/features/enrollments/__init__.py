"""Enrollments feature package (T7.1).

Manager-assigned only — there is intentionally no student-facing
self-enroll route and no invitation-code redemption endpoint per the
locked plan decision. Invitation codes are Manager tracking artefacts
that Managers hand out via email; the redeem path was rejected from
the thesis self-enroll flow.
"""

from __future__ import annotations
