"""Public re-exports for the spaced-repetition schema module (T7.5.12).

Pydantic DTOs for the dashboard surface composed in T7.5.12. Internal
dataclasses (``StudentLessonSummary``, ``ClassKRDistribution``,
``DifficultCard``, ``AtRiskStudent``) live in
:mod:`abridgeai.features.spaced_repetition.queries`; this module wraps
them in HTTP-shaped Pydantic models so the routers stay typed without
leaking dataclass internals across the API boundary.
"""

from __future__ import annotations

from abridgeai.features.spaced_repetition.schemas.dashboards import (
    AtRiskStudentRead,
    CardsDueItem,
    CardsDuePage,
    ClassKRDistributionRead,
    DifficultCardRead,
    LessonOverviewItem,
    LessonStatus,
    StudentLessonSummaryRead,
    StudentSrDetailLessonRead,
    StudentSrDetailRead,
    StudentSrDetailReviewRead,
)

__all__ = [
    "AtRiskStudentRead",
    "CardsDueItem",
    "CardsDuePage",
    "ClassKRDistributionRead",
    "DifficultCardRead",
    "LessonOverviewItem",
    "LessonStatus",
    "StudentLessonSummaryRead",
    "StudentSrDetailLessonRead",
    "StudentSrDetailRead",
    "StudentSrDetailReviewRead",
]
