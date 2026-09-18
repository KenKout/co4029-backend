"""Which career paths a program offers, and which one is the default.

A program version carries an ordered list of career paths and exactly one
default. Editing that list in a draft is where two rules live that are easy
to get backwards, and both of them are about *not* silently changing what
enrolled students are walking.

**An existing path keeps the version it was pinned to.** When a draft is
re-saved, a path already on the program is re-attached with its stored
``career_path_version_id``, not with whatever the path's current head
happens to be. Only newly added paths resolve to the published head. Losing
that would re-pin every enrolled student to a newer version of their path
the next time an author touched an unrelated field.

**Removing the default is refused, not absorbed.** A program with no
default has nothing to enrol a student onto, so the author is made to
nominate a replacement rather than have one picked for them.

``_resolve_draft_default`` is the trickiest piece and is pure: it has to
tell "the client explicitly sent null" apart from "the client omitted the
field", which are different requests that a plain ``None`` cannot express.

Queries are mocked; the default resolver needs nothing at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from abridgeai.core.exceptions import ConflictError, NotFoundError
from abridgeai.features.learning_programs import services


def _payload(**fields: Any) -> SimpleNamespace:
    """A ``ProgramUpdate`` stand-in carrying ``model_fields_set``."""
    return SimpleNamespace(model_fields_set=set(fields), **fields)


def _path_row(path_id: UUID, *, default: bool = False, version_id: UUID | None = None) -> dict:
    return {
        "career_path_id": path_id,
        "career_path_version_id": version_id or uuid4(),
        "is_default": default,
    }


class TestResolvingTheDefaultOnADraftSave:
    """The field is absent, null, or set -- three requests, three answers."""

    def test_an_omitted_field_keeps_the_existing_default(self) -> None:
        """An author editing the title must not lose the default they chose."""
        kept = uuid4()
        result = services._resolve_draft_default(
            payload=_payload(),
            path_ids=[kept, uuid4()],
            existing_paths=[_path_row(kept, default=True)],
        )
        assert result == kept

    def test_an_explicit_null_is_refused_while_a_default_exists(self) -> None:
        """Clearing the default would leave the program unable to enrol
        anyone, so the author is told to nominate a replacement instead of
        having the request quietly absorbed.
        """
        existing = uuid4()
        with pytest.raises(ConflictError, match="default_path_must_be_replaced_before_removal"):
            services._resolve_draft_default(
                payload=_payload(default_career_path_id=None),
                path_ids=[existing],
                existing_paths=[_path_row(existing, default=True)],
            )

    def test_an_explicit_null_is_fine_when_there_was_no_default(self) -> None:
        """A draft that never had one is not being asked to give one up."""
        result = services._resolve_draft_default(
            payload=_payload(default_career_path_id=None),
            path_ids=[uuid4()],
            existing_paths=[],
        )
        assert result is None

    def test_a_nominated_default_must_be_one_of_the_paths(self) -> None:
        """Otherwise the program's default points at a path it does not
        offer, and enrolment has nowhere to send the student."""
        with pytest.raises(ConflictError, match="default_path_must_belong_to_program_version"):
            services._resolve_draft_default(
                payload=_payload(default_career_path_id=uuid4()),
                path_ids=[uuid4(), uuid4()],
                existing_paths=[],
            )

    def test_a_nominated_default_that_is_offered_is_accepted(self) -> None:
        chosen = uuid4()
        result = services._resolve_draft_default(
            payload=_payload(default_career_path_id=chosen),
            path_ids=[uuid4(), chosen],
            existing_paths=[],
        )
        assert result == chosen

    def test_dropping_the_default_path_from_the_list_is_refused(self) -> None:
        """The subtle one: the author did not touch the default field, they
        removed the path that *was* the default.

        Accepting it would leave a default pointing outside the list, so the
        same refusal applies as if they had cleared it directly.
        """
        removed = uuid4()
        with pytest.raises(ConflictError, match="default_path_must_be_replaced_before_removal"):
            services._resolve_draft_default(
                payload=_payload(),
                path_ids=[uuid4()],
                existing_paths=[_path_row(removed, default=True)],
            )

    def test_replacing_the_default_in_the_same_save_is_allowed(self) -> None:
        """Nominating the new one and dropping the old one is one edit, and
        refusing it would make the default impossible to change."""
        old, new = uuid4(), uuid4()
        result = services._resolve_draft_default(
            payload=_payload(default_career_path_id=new),
            path_ids=[new],
            existing_paths=[_path_row(old, default=True)],
        )
        assert result == new

    def test_a_draft_with_no_paths_and_no_default_resolves_to_nothing(self) -> None:
        assert (
            services._resolve_draft_default(
                payload=_payload(), path_ids=[], existing_paths=[]
            )
            is None
        )


class TestReplacingTheDraftsPathList:
    @pytest.fixture
    def program(self) -> SimpleNamespace:
        return SimpleNamespace(id=uuid4(), organization_id=uuid4())

    @pytest.fixture
    def version(self) -> SimpleNamespace:
        return SimpleNamespace(id=uuid4())

    @pytest.fixture
    def added(self, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
        rows: list[Any] = []
        monkeypatch.setattr(
            services,
            "LearningProgramVersionPath",
            lambda **kwargs: SimpleNamespace(**kwargs),
        )
        return rows

    def _db(self, added: list[Any]) -> SimpleNamespace:
        return SimpleNamespace(add=added.append)

    def _resolve(
        self, monkeypatch: pytest.MonkeyPatch, rows: list[dict]
    ) -> AsyncMock:
        stub = AsyncMock(return_value=rows)
        monkeypatch.setattr(services.queries, "resolve_published_path_versions", stub)
        return stub

    async def test_a_repeated_path_is_refused(
        self, program: SimpleNamespace, version: SimpleNamespace, added: list[Any]
    ) -> None:
        """Position is meaningful, so the same path twice has no coherent
        ordering -- and the unique index would reject it anyway."""
        path_id = uuid4()
        with pytest.raises(ConflictError, match="career_path_ids_must_be_unique"):
            await services._replace_draft_paths(
                self._db(added), program, version, [path_id, path_id]
            )

    async def test_a_single_path_program_defaults_to_its_only_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        program: SimpleNamespace,
        version: SimpleNamespace,
        added: list[Any],
    ) -> None:
        """Asking an author to nominate the default when there is exactly
        one candidate is a question with one answer.
        """
        only = uuid4()
        self._resolve(
            monkeypatch,
            [{"career_path_id": only, "version_id": uuid4(), "career_path_status": "published"}],
        )

        await services._replace_draft_paths(self._db(added), program, version, [only])

        assert [row.is_default for row in added] == [True]

    async def test_a_default_outside_the_list_is_refused(
        self, program: SimpleNamespace, version: SimpleNamespace, added: list[Any]
    ) -> None:
        with pytest.raises(ConflictError, match="default_path_must_belong_to_program_version"):
            await services._replace_draft_paths(
                self._db(added),
                program,
                version,
                [uuid4(), uuid4()],
                default_career_path_id=uuid4(),
            )

    async def test_an_unpublished_path_cannot_be_attached(
        self,
        monkeypatch: pytest.MonkeyPatch,
        program: SimpleNamespace,
        version: SimpleNamespace,
        added: list[Any],
    ) -> None:
        """The resolver only returns paths that are published and live, so a
        short result means one of the requested paths is neither.

        Attaching a draft path would let a program publish a route students
        cannot actually walk.
        """
        self._resolve(monkeypatch, [])

        with pytest.raises(ConflictError, match="all_paths_must_be_published_and_not_archived"):
            await services._replace_draft_paths(
                self._db(added), program, version, [uuid4(), uuid4()]
            )

    async def test_an_existing_path_keeps_the_version_it_was_pinned_to(
        self,
        monkeypatch: pytest.MonkeyPatch,
        program: SimpleNamespace,
        version: SimpleNamespace,
        added: list[Any],
    ) -> None:
        """The rule that protects students already walking the path.

        Re-saving a draft must not re-pin an existing path to the career
        path's current head: students are enrolled against the pinned
        version, and moving it would change what they have to finish
        because an author renamed the program.
        """
        pinned_version = uuid4()
        existing_id = uuid4()
        resolve = self._resolve(monkeypatch, [])

        await services._replace_draft_paths(
            self._db(added),
            program,
            version,
            [existing_id],
            existing_paths=[_path_row(existing_id, version_id=pinned_version)],
        )

        assert [row.career_path_version_id for row in added] == [pinned_version]
        assert resolve.await_args.kwargs["career_path_ids"] == [], (
            "an already-attached path is not re-resolved at all"
        )

    async def test_a_newly_added_path_takes_the_published_head(
        self,
        monkeypatch: pytest.MonkeyPatch,
        program: SimpleNamespace,
        version: SimpleNamespace,
        added: list[Any],
    ) -> None:
        """Nobody is enrolled onto it yet, so the newest published version
        is the right thing to pin."""
        new_id, head = uuid4(), uuid4()
        self._resolve(
            monkeypatch,
            [{"career_path_id": new_id, "version_id": head, "career_path_status": "published"}],
        )

        await services._replace_draft_paths(self._db(added), program, version, [new_id])

        assert added[0].career_path_version_id == head

    async def test_an_archived_path_is_refused_by_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        program: SimpleNamespace,
        version: SimpleNamespace,
        added: list[Any],
    ) -> None:
        """A distinct message from the general "must be published" one,
        because archiving is a deliberate act the author can reverse."""
        path_id = uuid4()
        self._resolve(
            monkeypatch,
            [{"career_path_id": path_id, "version_id": uuid4(), "career_path_status": "archived"}],
        )

        with pytest.raises(ConflictError, match="archived_path_cannot_be_added"):
            await services._replace_draft_paths(self._db(added), program, version, [path_id])

    async def test_positions_follow_the_order_the_author_gave(
        self,
        monkeypatch: pytest.MonkeyPatch,
        program: SimpleNamespace,
        version: SimpleNamespace,
        added: list[Any],
    ) -> None:
        """The list is what a student is shown when choosing, so the order
        is the author's editorial decision rather than an implementation
        detail.
        """
        first, second, third = uuid4(), uuid4(), uuid4()
        self._resolve(
            monkeypatch,
            [
                {"career_path_id": pid, "version_id": uuid4(), "career_path_status": "published"}
                for pid in (first, second, third)
            ],
        )

        await services._replace_draft_paths(
            self._db(added), program, version, [first, second, third],
            default_career_path_id=second,
        )

        assert [(row.career_path_id, row.position) for row in added] == [
            (first, 1),
            (second, 2),
            (third, 3),
        ]
        assert [row.is_default for row in added] == [False, True, False]

    async def test_existing_and_new_paths_mix_in_one_save(
        self,
        monkeypatch: pytest.MonkeyPatch,
        program: SimpleNamespace,
        version: SimpleNamespace,
        added: list[Any],
    ) -> None:
        """The ordinary edit: keep what is there and add one more."""
        kept, kept_version, fresh, fresh_head = uuid4(), uuid4(), uuid4(), uuid4()
        resolve = self._resolve(
            monkeypatch,
            [
                {
                    "career_path_id": fresh,
                    "version_id": fresh_head,
                    "career_path_status": "published",
                }
            ],
        )

        await services._replace_draft_paths(
            self._db(added),
            program,
            version,
            [kept, fresh],
            default_career_path_id=kept,
            existing_paths=[_path_row(kept, version_id=kept_version)],
        )

        assert resolve.await_args.kwargs["career_path_ids"] == [fresh]
        assert [row.career_path_version_id for row in added] == [kept_version, fresh_head]


class TestPublishingAProgram:
    @pytest.fixture
    def world(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        program = SimpleNamespace(
            id=uuid4(), organization_id=uuid4(), status="draft", updated_by=None
        )
        version = SimpleNamespace(
            id=uuid4(),
            status="draft",
            max_career_paths_per_enrollment=1,
            published_at=None,
            updated_by=None,
        )
        monkeypatch.setattr(services, "_require_operator", AsyncMock())
        monkeypatch.setattr(services, "flush_or_conflict", AsyncMock())
        monkeypatch.setattr(services, "_program_out", AsyncMock(return_value="published"))
        monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=program))
        monkeypatch.setattr(
            services.queries, "get_current_version", AsyncMock(return_value=version)
        )
        monkeypatch.setattr(
            services.queries,
            "list_version_paths",
            AsyncMock(return_value=[{"is_default": True}]),
        )
        monkeypatch.setattr(
            services.queries, "list_unpublishable_version_path_ids", AsyncMock(return_value=[])
        )
        return {
            "db": SimpleNamespace(),
            "program": program,
            "version": version,
            "actor": SimpleNamespace(user_id=uuid4()),
        }

    async def _publish(self, world: dict[str, Any]):
        return await services.publish_program(
            world["db"], program_id=world["program"].id, actor=world["actor"]
        )

    async def test_an_unknown_program_is_not_found(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        monkeypatch.setattr(services.queries, "get_program", AsyncMock(return_value=None))
        with pytest.raises(NotFoundError, match="learning_program_not_found"):
            await self._publish(world)

    async def test_an_archived_program_cannot_be_published(
        self, world: dict[str, Any]
    ) -> None:
        """Archive is the end of a program's life, not a pause."""
        world["program"].status = "archived"
        with pytest.raises(ConflictError, match="archived_program_cannot_be_published"):
            await self._publish(world)

    async def test_publishing_needs_a_draft_to_publish(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Re-publishing an already-published version would move its
        ``published_at`` and tell nobody anything new."""
        world["version"].status = "published"
        with pytest.raises(ConflictError, match="program_has_no_draft_version"):
            await self._publish(world)

    async def test_a_program_with_no_paths_cannot_be_published(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """There would be nothing to enrol a student onto."""
        monkeypatch.setattr(
            services.queries, "list_version_paths", AsyncMock(return_value=[])
        )
        with pytest.raises(ConflictError, match="program_requires_at_least_one_path"):
            await self._publish(world)

    async def test_the_student_path_limit_cannot_exceed_what_is_offered(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """A limit of three on a two-path program is not a limit, it is a
        setting that can never bind -- and the refusal carries both numbers
        so the author knows which one to change.
        """
        world["version"].max_career_paths_per_enrollment = 3
        monkeypatch.setattr(
            services.queries,
            "list_version_paths",
            AsyncMock(return_value=[{"is_default": True}, {"is_default": False}]),
        )

        with pytest.raises(services.ProgramConflictError) as raised:
            await self._publish(world)

        assert raised.value.code == "career_path_limit_exceeds_program_paths", (
            "the machine code stays stable for the client to branch on"
        )
        assert raised.value.fields == {"requested": 3, "path_count": 2}
        assert "cannot exceed" in raised.value.message, (
            "the sentence is what a manager is shown, rather than the code"
        )

    @pytest.mark.parametrize(
        "paths",
        [
            [{"is_default": False}, {"is_default": False}],
            [{"is_default": True}, {"is_default": True}],
        ],
        ids=["none", "two"],
    )
    async def test_exactly_one_default_is_required(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any], paths: list[dict]
    ) -> None:
        """Zero leaves enrolment with no destination; two leaves it with no
        rule for choosing between them."""
        world["version"].max_career_paths_per_enrollment = 1
        monkeypatch.setattr(
            services.queries, "list_version_paths", AsyncMock(return_value=paths)
        )

        with pytest.raises(ConflictError, match="program_requires_exactly_one_default_path"):
            await self._publish(world)

    async def test_a_path_that_became_unavailable_blocks_publishing(
        self, monkeypatch: pytest.MonkeyPatch, world: dict[str, Any]
    ) -> None:
        """Paths are validated again at publish because a draft can sit for
        weeks, and a path that was published when it was attached may have
        been archived since.
        """
        monkeypatch.setattr(
            services.queries,
            "list_unpublishable_version_path_ids",
            AsyncMock(return_value=[uuid4()]),
        )

        with pytest.raises(ConflictError, match="program_contains_unavailable_paths"):
            await self._publish(world)

    async def test_a_clean_program_publishes_both_rows(
        self, world: dict[str, Any]
    ) -> None:
        """The version and the program flip together: a published version
        under a draft program would be invisible, and the reverse would be
        a program advertising nothing.
        """
        result = await self._publish(world)

        assert result == "published"
        assert world["version"].status == "published"
        assert world["version"].published_at is not None
        assert world["program"].status == "published"
        assert world["program"].updated_by == world["actor"].user_id
