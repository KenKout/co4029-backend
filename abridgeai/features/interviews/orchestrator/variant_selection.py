"""Pick one stable angle from each complete logical-question group."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Sequence
from typing import Protocol

from abridgeai.features.interviews.orchestrator.interviewer_identity import (
    InterviewerRole,
)
from abridgeai.features.interviews.orchestrator.role_question_filter import preferred_type


class VariantCandidate(Protocol):
    question_id: str
    question_type: str
    variant_group_id: str | None


_POLICY_VERSION = "logical-angle-v1"


def select_logical_variants(
    candidates: Sequence[VariantCandidate],
    *,
    role: InterviewerRole,
    session_seed: str,
) -> list[VariantCandidate]:
    """Collapse every complete four-angle group to one stable member.

    A named role receives its preferred angle. The generic role hashes the
    immutable session id with each group id, so its one-of-four choice is random
    across sessions but stable across retries, reconnects, and runtime paths.
    Ungrouped or malformed groups retain legacy behavior.
    """
    groups: dict[str, list[VariantCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.variant_group_id is not None:
            groups[candidate.variant_group_id].append(candidate)

    chosen_ids: set[str] = set()
    complete_ids: set[str] = set()
    preferred = preferred_type(role)
    for group_id, members in groups.items():
        if len(members) != 4 or len({member.question_type for member in members}) != 4:
            continue
        complete_ids.add(group_id)
        if preferred is not None:
            chosen = next((member for member in members if member.question_type == preferred), None)
        else:
            ordered = sorted(members, key=lambda member: member.question_id)
            digest = hashlib.sha256(
                f"{_POLICY_VERSION}:{session_seed}:{group_id}".encode()
            ).digest()
            chosen = ordered[int.from_bytes(digest[:8], "big") % len(ordered)]
        if chosen is not None:
            chosen_ids.add(chosen.question_id)

    return [
        candidate
        for candidate in candidates
        if candidate.variant_group_id not in complete_ids or candidate.question_id in chosen_ids
    ]


__all__ = ["select_logical_variants"]
