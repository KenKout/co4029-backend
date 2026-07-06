"""Unit tests for material version history + rollback (FR-3.4, phase-04).

Service-level with mocked queries/helpers, plus router wiring call-site
coverage. Rollback is a pointer swap: chunks are version-scoped, so a
previously-ready version needs no reprocess.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.features.materials.services.authoring import _versions as versions_service

_MATERIAL = uuid.uuid4()
_VERSION = uuid.uuid4()
_ACTOR = SimpleNamespace(user_id=uuid.uuid4())
_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)

_SVC = "abridgeai.features.materials.services.authoring._versions"


def _version_row(
    *,
    version_no: int = 1,
    is_current: bool = False,
    processing_status: str = "ready",
    material_id: uuid.UUID = _MATERIAL,
    version_id: uuid.UUID = _VERSION,
) -> SimpleNamespace:
    return SimpleNamespace(
        id=version_id,
        material_id=material_id,
        version_no=version_no,
        storage_object_id=uuid.uuid4(),
        is_current=is_current,
        processing_status=processing_status,
        processing_error=None,
        extracted_metadata={},
        uploaded_by=None,
        uploaded_at=_NOW,
        processed_at=_NOW,
        created_by=None,
        updated_by=None,
        created_at=_NOW,
        updated_at=_NOW,
        deleted_at=None,
        deleted_by=None,
    )


def _material_row() -> SimpleNamespace:
    return SimpleNamespace(id=_MATERIAL, current_version_id=uuid.uuid4(), updated_by=None)


class TestListMaterialVersions:
    async def test_lists_newest_first_projection(self) -> None:
        rows = [_version_row(version_no=2, version_id=uuid.uuid4()), _version_row(version_no=1)]
        with patch(
            f"{_SVC}.get_material_with_versions",
            new=AsyncMock(return_value=(_material_row(), rows)),
        ):
            result = await versions_service.list_material_versions(AsyncMock(), _MATERIAL)
        assert [v.version_no for v in result] == [2, 1]
        assert result[1].processing_status == "ready"

    async def test_missing_material_raises_not_found(self) -> None:
        with (
            patch(f"{_SVC}.get_material_with_versions", new=AsyncMock(return_value=None)),
            pytest.raises(NotFoundError),
        ):
            await versions_service.list_material_versions(AsyncMock(), _MATERIAL)


class TestRollbackMaterialVersion:
    def _patches(self, material: SimpleNamespace, version: SimpleNamespace) -> list:
        return [
            patch(f"{_SVC}.require_material", new=AsyncMock(return_value=material)),
            patch(f"{_SVC}.require_version", new=AsyncMock(return_value=version)),
            patch(f"{_SVC}.reset_other_versions_current", new=AsyncMock()),
            patch(f"{_SVC}.flush_or_conflict", new=AsyncMock()),
        ]

    async def test_happy_path_swaps_pointer(self) -> None:
        material = _material_row()
        version = _version_row()
        patches = self._patches(material, version)
        with patches[0], patches[1], patches[2] as reset, patches[3] as flush:
            result = await versions_service.rollback_material_version(
                AsyncMock(), _MATERIAL, _VERSION, _ACTOR
            )
        assert version.is_current is True
        assert material.current_version_id == _VERSION
        assert material.updated_by == _ACTOR.user_id
        reset.assert_awaited_once()
        flush.assert_awaited_once()
        assert result.id == _VERSION

    async def test_already_current_conflicts(self) -> None:
        patches = self._patches(_material_row(), _version_row(is_current=True))
        with patches[0], patches[1], pytest.raises(ConflictError, match="already"):
            await versions_service.rollback_material_version(
                AsyncMock(), _MATERIAL, _VERSION, _ACTOR
            )

    async def test_not_ready_conflicts(self) -> None:
        patches = self._patches(_material_row(), _version_row(processing_status="failed"))
        with patches[0], patches[1], pytest.raises(ConflictError, match="not ready"):
            await versions_service.rollback_material_version(
                AsyncMock(), _MATERIAL, _VERSION, _ACTOR
            )

    async def test_foreign_version_not_found(self) -> None:
        patches = self._patches(_material_row(), _version_row(material_id=uuid.uuid4()))
        with patches[0], patches[1], pytest.raises(NotFoundError):
            await versions_service.rollback_material_version(
                AsyncMock(), _MATERIAL, _VERSION, _ACTOR
            )


class TestRouterWiring:
    async def test_list_route_calls_service(self) -> None:
        from abridgeai.features.materials.routers import authoring as router_module

        svc = AsyncMock(return_value=[])
        with patch.object(router_module.authoring_service, "list_material_versions", svc):
            await router_module.list_material_versions(_MATERIAL, _ACTOR, AsyncMock())
        svc.assert_awaited_once()

    async def test_rollback_route_maps_conflict_to_409(self) -> None:
        from fastapi import HTTPException

        from abridgeai.features.materials.routers import authoring as router_module

        svc = AsyncMock(side_effect=ConflictError("not ready"))
        with (
            patch.object(router_module.authoring_service, "rollback_material_version", svc),
            pytest.raises(HTTPException) as exc_info,
        ):
            await router_module.rollback_material_version(_MATERIAL, _VERSION, _ACTOR, AsyncMock())
        assert exc_info.value.status_code == 409
        assert exc_info.value.detail["error"] == "rollback_rejected"
