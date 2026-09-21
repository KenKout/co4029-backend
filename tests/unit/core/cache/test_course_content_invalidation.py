"""When a write drops the shared published-course-tree key, and when it must not.

Two properties matter here, and they pull in opposite directions.

The first is reach. The cached tree is assembled from courses, modules,
module items, lessons, quizzes, interview configs, the teaching team and the
career-path placements; a write to any of them can change what a learner sees,
so each has to resolve back to a course id. Two of those tables — ``lessons``
and ``module_items`` — only know their module, so that hop is a real query.

The second is timing. The key is shared by everyone on the course and costs
several batch queries to rebuild, so a delete at flush time is not enough on
its own: between that delete and the commit, a concurrent reader can refill
the key from the pre-write snapshot, and that stale tree would then sit there,
for every learner, until the TTL ran out. The delete is therefore repeated
after the commit, where nothing can put the old snapshot back.

No Redis: the delete is a recorded call.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest
from sqlalchemy import ForeignKey, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# Side-effect import: registers the cache listeners on the global Session.
import abridgeai.core.db  # noqa: F401
from abridgeai.core.cache import COURSE_CONTENT_PUBLISHED
from abridgeai.core.cache import invalidator as invalidator_module
from abridgeai.core.cache.course_content import collect_course_content_keys


class _Base(DeclarativeBase):
    pass


class FakeCourse(_Base):
    __tablename__ = "courses"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    title: Mapped[str] = mapped_column(String(64), default="t")


class FakeModule(_Base):
    __tablename__ = "modules"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    course_id: Mapped[str] = mapped_column(ForeignKey("courses.id"))


class FakeLesson(_Base):
    __tablename__ = "lessons"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    module_id: Mapped[str] = mapped_column(ForeignKey("modules.id"))
    title: Mapped[str] = mapped_column(String(64), default="t")


COURSE_ID = str(uuid4())
MODULE_ID = str(uuid4())


def _key(course_id: str = COURSE_ID) -> str:
    return COURSE_CONTENT_PUBLISHED.format(course_id=course_id)


_NAMESPACE = COURSE_CONTENT_PUBLISHED.pattern.split("{", 1)[0]


@pytest.fixture
def deletions(monkeypatch: pytest.MonkeyPatch) -> list[set[str]]:
    """Capture the course-content keys the invalidator would send to Redis.

    Other namespaces ride the same scheduler (a lesson write also drops that
    learner's unlock keys); they are filtered out so these tests describe one
    cache.
    """
    recorded: list[set[str]] = []

    def _record(keys: set[str], user_ids: set[str], globs: set[str]) -> None:
        mine = {k for k in keys if k.startswith(_NAMESPACE)}
        if mine:
            recorded.append(mine)

    monkeypatch.setattr(invalidator_module, "_schedule_invalidation", _record)
    return recorded


@pytest.fixture
def session(deletions: list[set[str]]) -> Any:
    engine = create_engine("sqlite://")
    _Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add(FakeCourse(id=COURSE_ID))
        db.add(FakeModule(id=MODULE_ID, course_id=COURSE_ID))
        db.commit()
        deletions.clear()  # the fixture's own seed writes are not under test
        yield db
    _Base.metadata.drop_all(engine)
    engine.dispose()


class TestResolvingAWriteToItsCourse:
    def test_a_row_that_names_its_course_resolves_directly(self, session: Any) -> None:
        keys = collect_course_content_keys(
            session, [FakeModule(id=str(uuid4()), course_id=COURSE_ID)]
        )

        assert keys == {_key()}

    def test_the_course_row_itself_resolves_through_its_primary_key(self, session: Any) -> None:
        assert collect_course_content_keys(session, [FakeCourse(id=COURSE_ID)]) == {_key()}

    def test_a_lesson_resolves_through_its_module(self, session: Any) -> None:
        # The lesson knows only its module; the course id is one hop away.
        keys = collect_course_content_keys(
            session, [FakeLesson(id=str(uuid4()), module_id=MODULE_ID)]
        )

        assert keys == {_key()}

    def test_a_table_the_tree_does_not_read_is_ignored(self, session: Any) -> None:
        class Unrelated:
            __tablename__ = "notification_deliveries"
            course_id = COURSE_ID  # same column name, different aggregate

        assert collect_course_content_keys(session, [Unrelated()]) == set()

    def test_one_flush_touching_two_courses_drops_both(self, session: Any) -> None:
        other = str(uuid4())
        keys = collect_course_content_keys(
            session,
            [
                FakeModule(id=str(uuid4()), course_id=COURSE_ID),
                FakeModule(id=str(uuid4()), course_id=other),
            ],
        )

        assert keys == {_key(), _key(other)}


class TestWhenTheDeletionHappens:
    def test_the_flush_deletes_immediately(self, session: Any, deletions: list[set[str]]) -> None:
        # Covers the long transaction, and the session that flushes and is
        # then thrown away without ever committing.
        session.add(FakeLesson(id=str(uuid4()), module_id=MODULE_ID))
        session.flush()

        assert deletions == [{_key()}]

    def test_the_commit_deletes_again(self, session: Any, deletions: list[set[str]]) -> None:
        # The repeat is the one that guarantees correctness: a reader that
        # refilled the key from the pre-commit snapshot is undone here.
        session.add(FakeLesson(id=str(uuid4()), module_id=MODULE_ID))
        session.commit()

        assert deletions == [{_key()}, {_key()}]

    def test_a_rollback_sends_no_second_delete(
        self, session: Any, deletions: list[set[str]]
    ) -> None:
        session.add(FakeLesson(id=str(uuid4()), module_id=MODULE_ID))
        session.flush()
        deletions.clear()
        session.rollback()

        assert deletions == []

    def test_a_later_commit_does_not_replay_an_earlier_write(
        self, session: Any, deletions: list[set[str]]
    ) -> None:
        session.add(FakeLesson(id=str(uuid4()), module_id=MODULE_ID))
        session.commit()
        deletions.clear()
        session.commit()

        assert deletions == []
