from __future__ import annotations

from sqlalchemy import CheckConstraint, UniqueConstraint

from abridgeai.ai.models import ProcessingJob
from abridgeai.features.materials.models import (
    ChunkingEnrichmentCache,
    DocumentChunk,
    LearningMaterial,
    LearningMaterialVersion,
)

_AUDIT_COLUMNS = {"created_by", "updated_by", "deleted_at", "deleted_by"}
_TIMESTAMP_COLUMNS = {"created_at", "updated_at"}


def test_models_importable() -> None:
    assert LearningMaterial.__tablename__ == "learning_materials"
    assert LearningMaterialVersion.__tablename__ == "learning_material_versions"
    assert DocumentChunk.__tablename__ == "document_chunks"
    assert ProcessingJob.__tablename__ == "processing_jobs"
    assert ChunkingEnrichmentCache.__tablename__ == "chunking_enrichment_cache"


def test_audit_columns_present_on_soft_delete_models() -> None:
    for model in (LearningMaterial, LearningMaterialVersion):
        cols = set(model.__table__.columns.keys())
        for required in _AUDIT_COLUMNS | _TIMESTAMP_COLUMNS | {"id"}:
            assert required in cols, f"{model.__tablename__} missing {required}"


def test_material_type_check_constraint() -> None:
    checks = [c for c in LearningMaterial.__table__.constraints if isinstance(c, CheckConstraint)]
    assert any(c.name == "learning_materials_material_type_check" for c in checks), [
        c.name for c in checks
    ]
    src = next(c.sqltext.text for c in checks if c.name == "learning_materials_material_type_check")
    for value in ("video", "pdf", "code", "audio", "image", "docx", "pptx", "xlsx", "text"):
        assert f"'{value}'" in src


def test_processing_status_check_constraint_has_enriching() -> None:
    checks = [
        c for c in LearningMaterialVersion.__table__.constraints if isinstance(c, CheckConstraint)
    ]
    target = next(
        c for c in checks if c.name == "learning_material_versions_processing_status_check"
    )
    src = target.sqltext.text
    for value in (
        "pending",
        "extracting",
        "chunking",
        "enriching",
        "embedding",
        "building_kg",
        "ready",
        "failed",
        "cancelled",
    ):
        assert f"'{value}'" in src, f"{value!r} missing from {src!r}"


def test_document_chunk_denormalized_fks() -> None:
    cols = DocumentChunk.__table__.columns
    for required in ("course_id", "module_id", "lesson_id", "material_version_id"):
        assert required in cols, required
        assert not cols[required].nullable, f"{required} must be NOT NULL"


def test_document_chunk_no_softdelete_no_updated_at() -> None:
    cols = set(DocumentChunk.__table__.columns.keys())
    assert "deleted_at" not in cols
    assert "deleted_by" not in cols
    assert "updated_at" not in cols
    assert "created_at" in cols


def test_storage_object_fk_on_version() -> None:
    fk_targets = {
        next(iter(c.foreign_keys)).target_fullname
        for c in LearningMaterialVersion.__table__.columns
        if c.foreign_keys
    }
    assert "storage_objects.id" in fk_targets


def test_current_version_dual_source_invariant() -> None:
    assert "current_version_id" in LearningMaterial.__table__.columns
    assert "is_current" in LearningMaterialVersion.__table__.columns


def test_chunking_enrichment_cache_unique_constraint() -> None:
    uniques = [
        c for c in ChunkingEnrichmentCache.__table__.constraints if isinstance(c, UniqueConstraint)
    ]
    target = next(c for c in uniques if c.name == "uq_chunking_enrichment_cache_hash_prompt")
    cols = {col.name for col in target.columns}
    assert cols == {"content_hash", "prompt_version"}


def test_processing_job_status_check_constraint() -> None:
    checks = [c for c in ProcessingJob.__table__.constraints if isinstance(c, CheckConstraint)]
    target = next(c for c in checks if c.name == "processing_jobs_status_check")
    src = target.sqltext.text
    for value in ("pending", "running", "completed", "failed", "cancelled"):
        assert f"'{value}'" in src


def test_chunk_metadata_renamed_to_metadata_json_attribute() -> None:
    assert hasattr(DocumentChunk, "metadata_json")
    assert DocumentChunk.__table__.columns["metadata"].name == "metadata"
