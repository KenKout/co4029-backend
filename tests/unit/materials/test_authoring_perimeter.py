"""The material-scoped authorization perimeter (FIX-SEC-1).

Every material and version endpoint in the authoring router is guarded by
one of two dependency factories rather than by a check written into each
handler. That is the whole design: permissions in this system are granted
per *course*, but these routes are addressed by material or version id, so
something has to walk the chain -- version to material, material to lesson
to module to course -- and enforce the course's permissions at the end of
it. A handler that skipped the walk would be reachable by anyone holding a
valid token, which is the legacy bug the factories were introduced to close.

The existing integration suite proves the perimeter exists (a student is
refused, no endpoint uses a bare ``get_current_user``). What it does not
cover is how the walk behaves once it gets going, and that is where the
interesting cases are: who is let through without a permission check at
all, which of several permission codes suffices, and what happens when a
request pairs a version id with a material id from somewhere else.

The database and the permission engine are mocked. These tests are about
the decisions the factories make from what those two report back.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from abridgeai.features.materials.routers import authoring
from abridgeai.features.materials.routers.authoring import (
    _enforce_course_permission,
    _not_found,
    _permission_denied,
    _resolve_material_to_course,
    require_material_authoring_access,
    require_version_authoring_access,
)


def _user(user_id: UUID | None = None) -> SimpleNamespace:
    return SimpleNamespace(user_id=user_id or uuid4())


def _request(**path_params: Any) -> SimpleNamespace:
    """Only ``path_params`` is read off the request."""
    return SimpleNamespace(path_params=path_params)


@pytest.fixture
def chain(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """A material that resolves to a course owned by ``owner``."""
    course_id, owner_id, material_id = uuid4(), uuid4(), uuid4()
    monkeypatch.setattr(
        authoring.materials_api,
        "get_material_with_lesson_context",
        AsyncMock(return_value=SimpleNamespace(course_id=course_id)),
    )
    monkeypatch.setattr(
        authoring.courses_api,
        "get_course_by_id",
        AsyncMock(return_value=SimpleNamespace(owner_user_id=owner_id)),
    )
    return {"course_id": course_id, "owner_id": owner_id, "material_id": material_id}


def _allow(monkeypatch: pytest.MonkeyPatch, *granted: str) -> AsyncMock:
    """Stand in for the permission engine, granting only ``granted`` codes."""

    async def _can_manage(
        _db: object, _user_id: UUID, _course_id: UUID, *, manage_perm: str
    ) -> bool:
        return manage_perm in granted

    spy = AsyncMock(side_effect=_can_manage)
    monkeypatch.setattr(authoring, "can_manage_course", spy)
    return spy


class TestWalkingFromAMaterialToItsCourse:
    async def test_a_material_that_does_not_resolve_is_absent(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            authoring.materials_api,
            "get_material_with_lesson_context",
            AsyncMock(return_value=None),
        )
        assert await _resolve_material_to_course(object(), uuid4()) is None

    async def test_a_material_whose_course_is_gone_does_not_resolve(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Failing open here would leave a material with no course to check
        its permissions against, and the walk has nothing else to enforce."""
        monkeypatch.setattr(
            authoring.materials_api,
            "get_material_with_lesson_context",
            AsyncMock(return_value=SimpleNamespace(course_id=uuid4())),
        )
        monkeypatch.setattr(
            authoring.courses_api, "get_course_by_id", AsyncMock(return_value=None)
        )
        assert await _resolve_material_to_course(object(), uuid4()) is None

    async def test_a_resolved_material_yields_its_course_and_owner(
        self, chain: dict[str, Any]
    ) -> None:
        resolved = await _resolve_material_to_course(object(), chain["material_id"])
        assert resolved == (chain["course_id"], chain["owner_id"])


class TestWhoIsLetThrough:
    async def test_the_course_owner_is_never_asked_for_a_permission(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ownership short-circuits ahead of the engine.

        A teacher cannot be locked out of their own course by a role
        assignment going missing, and the check costs nothing.
        """
        spy = _allow(monkeypatch)
        owner = _user()
        course_id = uuid4()

        result = await _enforce_course_permission(
            object(), owner, course_id, owner.user_id, ("course.update",)
        )

        assert result is owner
        spy.assert_not_awaited()

    async def test_a_non_owner_needs_the_permission(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _allow(monkeypatch, "course.update")
        actor = _user()

        result = await _enforce_course_permission(
            object(), actor, uuid4(), uuid4(), ("course.update",)
        )
        assert result is actor

    async def test_any_one_of_several_codes_suffices(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The codes are alternatives, not a set to hold all of.

        Requiring all of them would lock out every role that legitimately
        holds one -- a reviewer with ``course.publish`` but no
        ``course.update``, say.
        """
        _allow(monkeypatch, "course.publish")
        actor = _user()

        result = await _enforce_course_permission(
            object(), actor, uuid4(), uuid4(), ("course.update", "course.publish")
        )
        assert result is actor

    async def test_holding_none_of_them_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _allow(monkeypatch, "course.read")
        course_id = uuid4()

        with pytest.raises(HTTPException) as raised:
            await _enforce_course_permission(
                object(), _user(), course_id, uuid4(), ("course.update",)
            )

        assert raised.value.status_code == 403
        assert raised.value.detail["required"] == ["course.update"]
        assert raised.value.detail["course_id"] == str(course_id)

    async def test_every_code_is_tried_before_refusing(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stopping at the first miss would make the order of the codes
        decide the outcome."""
        spy = _allow(monkeypatch, "course.publish")

        await _enforce_course_permission(
            object(), _user(), uuid4(), uuid4(), ("course.update", "course.publish")
        )

        tried = [call.kwargs["manage_perm"] for call in spy.await_args_list]
        assert tried == ["course.update", "course.publish"]


class TestTheMaterialScopedDependency:
    async def test_an_unknown_material_is_a_404_before_any_permission_work(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            authoring.materials_api,
            "get_material_with_lesson_context",
            AsyncMock(return_value=None),
        )
        spy = _allow(monkeypatch, "course.update")
        material_id = uuid4()

        with pytest.raises(HTTPException) as raised:
            await require_material_authoring_access()(material_id, _user(), object())

        assert raised.value.status_code == 404
        assert raised.value.detail["resource"] == "material"
        assert raised.value.detail["id"] == str(material_id)
        spy.assert_not_awaited()

    async def test_the_default_requirement_is_course_update(
        self, monkeypatch: pytest.MonkeyPatch, chain: dict[str, Any]
    ) -> None:
        """Called with no codes at every call site, so the default is the
        real requirement for most of the material surface."""
        spy = _allow(monkeypatch, "course.update")

        await require_material_authoring_access()(chain["material_id"], _user(), object())

        assert spy.await_args.kwargs["manage_perm"] == "course.update"

    async def test_an_explicit_requirement_replaces_the_default(
        self, monkeypatch: pytest.MonkeyPatch, chain: dict[str, Any]
    ) -> None:
        spy = _allow(monkeypatch, "course.delete")

        await require_material_authoring_access("course.delete")(
            chain["material_id"], _user(), object()
        )

        assert [c.kwargs["manage_perm"] for c in spy.await_args_list] == ["course.delete"]

    async def test_the_owner_passes_the_dependency_untouched(
        self, monkeypatch: pytest.MonkeyPatch, chain: dict[str, Any]
    ) -> None:
        _allow(monkeypatch)
        owner = _user(chain["owner_id"])

        assert (
            await require_material_authoring_access()(chain["material_id"], owner, object())
            is owner
        )

    async def test_a_stranger_is_refused_with_the_courses_id(
        self, monkeypatch: pytest.MonkeyPatch, chain: dict[str, Any]
    ) -> None:
        _allow(monkeypatch)

        with pytest.raises(HTTPException) as raised:
            await require_material_authoring_access()(chain["material_id"], _user(), object())

        assert raised.value.status_code == 403
        assert raised.value.detail["course_id"] == str(chain["course_id"])


class TestTheVersionScopedDependency:
    """A version id is one hop further from the course than a material id.

    It also appears in paths that carry a material id beside it, which is
    the case the guard below exists for.
    """

    @pytest.fixture
    def version_chain(self, chain: dict[str, Any]) -> dict[str, Any]:
        """A version whose parent material is the one from ``chain``."""
        version_id = uuid4()

        class _Result:
            def scalar_one_or_none(self) -> UUID:
                return chain["material_id"]

        db = SimpleNamespace(execute=AsyncMock(return_value=_Result()))
        return {**chain, "version_id": version_id, "db": db}

    async def test_an_unknown_version_is_a_404(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _Empty:
            def scalar_one_or_none(self) -> None:
                return None

        db = SimpleNamespace(execute=AsyncMock(return_value=_Empty()))
        spy = _allow(monkeypatch, "course.update")
        version_id = uuid4()

        with pytest.raises(HTTPException) as raised:
            await require_version_authoring_access()(
                _request(), version_id, _user(), db
            )

        assert raised.value.status_code == 404
        assert raised.value.detail["resource"] == "material_version"
        assert raised.value.detail["id"] == str(version_id)
        spy.assert_not_awaited()

    async def test_a_version_of_a_material_the_caller_owns_is_allowed(
        self, monkeypatch: pytest.MonkeyPatch, version_chain: dict[str, Any]
    ) -> None:
        _allow(monkeypatch)
        owner = _user(version_chain["owner_id"])

        result = await require_version_authoring_access()(
            _request(), version_chain["version_id"], owner, version_chain["db"]
        )
        assert result is owner

    async def test_a_matching_material_id_in_the_path_is_accepted(
        self, monkeypatch: pytest.MonkeyPatch, version_chain: dict[str, Any]
    ) -> None:
        _allow(monkeypatch, "course.update")
        actor = _user()

        result = await require_version_authoring_access()(
            _request(material_id=str(version_chain["material_id"])),
            version_chain["version_id"],
            actor,
            version_chain["db"],
        )
        assert result is actor

    async def test_a_smuggled_material_id_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, version_chain: dict[str, Any]
    ) -> None:
        """The guard this dependency exists for.

        The route reads ``/materials/{material_id}/versions/{version_id}``,
        and permissions are enforced against the *version's* real parent.
        Without this check a caller could pair a version they may not touch
        with a material they may -- or, worse, act on a version while the
        handler below reads the material id from the path and operates on
        something else entirely.
        """
        _allow(monkeypatch, "course.update")

        with pytest.raises(HTTPException) as raised:
            await require_version_authoring_access()(
                _request(material_id=str(uuid4())),
                version_chain["version_id"],
                _user(),
                version_chain["db"],
            )

        assert raised.value.status_code == 404, (
            "404 rather than 403: a mismatch must not confirm that the "
            "version exists to someone guessing ids"
        )
        assert raised.value.detail["resource"] == "material_version"

    async def test_a_path_without_a_material_id_is_fine(
        self, monkeypatch: pytest.MonkeyPatch, version_chain: dict[str, Any]
    ) -> None:
        """Some version routes are addressed by version alone; the guard
        applies only when there is something to compare."""
        _allow(monkeypatch, "course.update")
        actor = _user()

        result = await require_version_authoring_access()(
            _request(), version_chain["version_id"], actor, version_chain["db"]
        )
        assert result is actor

    async def test_the_comparison_survives_string_and_uuid_forms(
        self, monkeypatch: pytest.MonkeyPatch, version_chain: dict[str, Any]
    ) -> None:
        """Path params arrive as strings, the resolved id as a ``UUID``.

        Comparing them without normalising would never match, and every
        version route carrying a material id would 404 for everyone.
        """
        _allow(monkeypatch, "course.update")

        result = await require_version_authoring_access()(
            _request(material_id=version_chain["material_id"]),
            version_chain["version_id"],
            _user(),
            version_chain["db"],
        )
        assert result is not None


class TestTheRefusalShapes:
    """Both are read by the SPA, which branches on them."""

    def test_not_found_names_the_resource_and_id(self) -> None:
        resource_id = uuid4()
        exc = _not_found("material_version", resource_id)
        assert exc.status_code == 404
        assert exc.detail == {
            "error": "not_found",
            "resource": "material_version",
            "id": str(resource_id),
        }

    def test_permission_denied_says_what_was_required(self) -> None:
        """The client renders "you need X on this course", so the codes and
        the course have to travel with the refusal."""
        course_id = uuid4()
        exc = _permission_denied(codes=("course.update", "course.publish"), course_id=course_id)
        assert exc.status_code == 403
        assert exc.detail == {
            "error": "permission_denied",
            "required": ["course.update", "course.publish"],
            "scope": "course",
            "course_id": str(course_id),
        }
