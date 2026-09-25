"""Which version of a material a learner reads.

The obvious answer — ``learning_materials.current_version_id`` — is wrong for
the minutes a re-upload spends in the pipeline, and wrong forever if that
pipeline fails.

``complete_upload`` promotes the new version the moment the bytes land: it
flips ``is_current``, points ``current_version_id`` at the new row, sets
``processing_status='pending'`` and only then enqueues the ingest. Every
learner read pairs the current version with ``processing_status='ready'``, so
between those two moments no row satisfies both and the material simply
vanishes for students. Nothing in the pipeline restores the old version if the
ingest fails, so a timed-out re-upload takes a working material dark
indefinitely, and only a teacher noticing and calling ``rollback_to_version``
brings it back.

The rule here is *prefer the current version, fall back to the newest ready
one*:

    ORDER BY is_current DESC, version_no DESC   (over ready versions only)

Ordering on ``is_current`` first is what keeps a deliberate rollback working.
``rollback_to_version`` refuses any target that is not ``ready``, so after a
rollback to v2 with v3 still ready, v2 is current AND ready, wins the sort, and
the newer v3 is correctly ignored. The fallback only ever engages when the
current version cannot be served — which is exactly the case it exists for.

A material with no ready version at all (a first upload still processing) stays
invisible, as before: there is nothing to fall back to.

Teacher-facing reads deliberately do NOT use this. A teacher re-uploading wants
to see what they just uploaded, mid-pipeline and all; only the learner surface
needs the last good version.

``chunks._STREAM_TARGET_SQL`` writes the same rule out by hand, because it is a
raw ``text()`` query (it reaches ``storage_objects``, which belongs to another
feature) and interpolating SQL from a constant trips the injection lint. The
two must agree; ``test_materials_queries`` covers both.
"""

from __future__ import annotations

import uuid

from sqlalchemy import ScalarSelect, select
from sqlalchemy.orm import InstrumentedAttribute, aliased

from abridgeai.features.materials.models import LearningMaterial, LearningMaterialVersion


def learner_version_id(
    material_id: InstrumentedAttribute[uuid.UUID],
) -> ScalarSelect[uuid.UUID]:
    """Correlated scalar subquery: the version id a learner should be served.

    ``material_id`` is the outer query's material column, normally
    ``LearningMaterial.id``. Returns ``NULL`` when the material has no ready
    version, which turns the caller's join into no rows — the same 404 the
    learner got before.

    ``deleted_at`` is spelled out rather than left to the global soft-delete
    listener: that listener rewrites ORM entity loads, and this is a column
    subquery on an alias. Being explicit costs one line and does not depend on
    how the criteria propagates.
    """
    version = aliased(LearningMaterialVersion, name="learner_version")
    return (
        select(version.id)
        .where(
            version.material_id == material_id,
            version.processing_status == "ready",
            version.deleted_at.is_(None),
        )
        .order_by(version.is_current.desc(), version.version_no.desc())
        .limit(1)
        .correlate(LearningMaterial)
        .scalar_subquery()
    )


__all__ = ["learner_version_id"]
