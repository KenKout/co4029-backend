"""Career paths feature package (T7.3).

Career paths group multiple courses into ordered learning tracks.
Manager-assigned only -- there is intentionally no student-facing
self-enroll route, mirroring the T7.1 enrollments invariant.

A path can be ``published`` while individual constituent courses are
still ``draft``; the published-course filter on the learner tree query
hides drafts from learners while authoring queries surface the full
set for Manager / Teacher review.
"""

from __future__ import annotations

from abridgeai.features.career_paths import models

__all__ = ["models"]
