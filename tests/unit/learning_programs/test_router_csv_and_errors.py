"""The two things the learning-program router does that aren't delegation.

Most endpoints here are three lines: call the service, commit, map the
error. The parts worth testing are the two helpers those three lines sit
around.

**The error map** is where a refusal becomes something a person reads.
Most of the service raises ``ConflictError("some_machine_code")`` and the
client branches on the code, which is fine until a code starts carrying
identifiers -- a manager hitting the concurrency cap was once shown
``concurrent_program_limit_reached:71acb8a2-…:1`` verbatim in a toast.
``ProgramConflictError`` exists to fix that, and this mapper is the half
that has to notice the difference.

**The CSV reader** takes a file a human exported from Excel, and every
quirk it handles is one that produced an unhelpful failure: a byte-order
mark that makes the first column header unmatchable, a base64 envelope for
files whose encoding would not survive JSON, and rows with more values than
headers.

All of this is pure -- no database, no session.
"""

from __future__ import annotations

import csv
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from abridgeai.core.exceptions import ConflictError, ForbiddenError, NotFoundError
from abridgeai.features.learning_programs import routers
from abridgeai.features.learning_programs.services import ProgramConflictError

_BOM = "\ufeff"
EMAIL_FILE = "email\na@test.local\n"
TWO_ROW_FILE = "email\na@test.local\nb@test.local\n"


class TestTurningARefusalIntoAResponse:
    def test_a_missing_thing_is_a_404(self) -> None:
        exc = routers._http_error(NotFoundError("learning_program_not_found"))
        assert exc.status_code == 404
        assert exc.detail == {"error": "learning_program_not_found"}

    def test_a_refusal_to_act_is_a_403(self) -> None:
        """Distinct from 404 on purpose here: the caller is a dean who *can*
        see the request, they just may not decide this one."""
        exc = routers._http_error(ForbiddenError("self_approval_is_not_allowed"))
        assert exc.status_code == 403
        assert exc.detail == {"error": "self_approval_is_not_allowed"}

    def test_a_plain_conflict_carries_only_its_code(self) -> None:
        """The client branches on the code; there is no sentence to show."""
        exc = routers._http_error(ConflictError("program_has_no_draft_version"))
        assert exc.status_code == 409
        assert exc.detail == {"error": "program_has_no_draft_version"}

    def test_a_structured_conflict_carries_the_sentence_and_its_fields(self) -> None:
        """The case the richer error class was added for.

        Without this branch the manager is shown the machine code -- which
        is what happened, and why ``message`` exists. The code still travels
        beside it so the client can keep branching on it.
        """
        exc = routers._http_error(
            ProgramConflictError(
                "career_path_limit_exceeds_program_paths",
                "The student path limit cannot exceed the number of Career Paths.",
                requested=3,
                path_count=2,
            )
        )

        assert exc.status_code == 409
        assert exc.detail["error"] == "career_path_limit_exceeds_program_paths"
        assert exc.detail["message"].startswith("The student path limit")
        assert exc.detail["requested"] == 3
        assert exc.detail["path_count"] == 2

    def test_the_structured_branch_is_checked_before_the_plain_one(self) -> None:
        """``ProgramConflictError`` subclasses ``ConflictError``, so the
        order of these two checks is the whole behaviour: reversed, the
        richer error would be flattened to its code and the message lost.
        """
        rich = ProgramConflictError("code", "A readable sentence.", limit=1)
        assert isinstance(rich, ConflictError)
        assert "message" in routers._http_error(rich).detail

    def test_an_unclassified_error_still_becomes_a_conflict(self) -> None:
        """A 409 with the text is a worse answer than a mapped one and a
        better answer than an unhandled 500."""
        exc = routers._http_error(ValueError("something unexpected"))
        assert exc.status_code == 409
        assert exc.detail == {"error": "something unexpected"}


class TestGettingTheFileOutOfTheRequest:
    def test_plain_text_is_taken_as_it_comes(self) -> None:
        payload = SimpleNamespace(csv_text="email\na@test.local\n", csv_base64=None)
        assert routers._decode_csv_text(payload) == "email\na@test.local\n"

    def test_a_base64_envelope_is_decoded(self) -> None:
        """The alternative exists so a file whose encoding would not survive
        a JSON string can be shipped byte-exact."""
        import base64

        raw = "email\nb@test.local\n"
        payload = SimpleNamespace(
            csv_text=None, csv_base64=base64.b64encode(raw.encode()).decode()
        )
        assert routers._decode_csv_text(payload) == raw

    def test_a_byte_order_mark_is_stripped_on_decode(self) -> None:
        """Excel writes a BOM, and a BOM on the header line makes the first
        column name unmatchable -- so every row fails validation on a file
        that looks perfectly correct when opened.
        """
        import base64

        # ``utf-8-sig`` on encode is what Excel does: it writes the mark
        # itself, so the source text must not already carry one or the file
        # ends up with two and only the outer one is stripped.
        raw = "email\nc@test.local\n".encode("utf-8-sig")
        assert raw.startswith(b"\xef\xbb\xbf"), "the fixture really is BOM-prefixed"
        payload = SimpleNamespace(csv_text=None, csv_base64=base64.b64encode(raw).decode())

        decoded = routers._decode_csv_text(payload)

        assert not decoded.startswith(_BOM)
        assert decoded.startswith("email")

    def test_text_wins_when_both_are_sent(self) -> None:
        payload = SimpleNamespace(csv_text="email\nfrom-text\n", csv_base64="aWdub3JlZA==")
        assert routers._decode_csv_text(payload) == "email\nfrom-text\n"

    def test_a_request_with_neither_is_a_400(self) -> None:
        """A body that is well-formed JSON but carries no file. The refusal
        names both accepted fields rather than saying "invalid"."""
        with pytest.raises(HTTPException) as raised:
            routers._decode_csv_text(SimpleNamespace(csv_text=None, csv_base64=None))

        assert raised.value.status_code == 400
        assert raised.value.detail["error"] == "invalid_csv"
        assert "csv_text or csv_base64" in raised.value.detail["message"]

    def test_an_empty_string_is_a_file_not_an_absence(self) -> None:
        """``""`` is a file with nothing in it, which imports zero rows --
        a different outcome from sending no file at all, and the check is
        ``is not None`` precisely so it can tell them apart.
        """
        assert routers._decode_csv_text(SimpleNamespace(csv_text="", csv_base64=None)) == ""


class TestReadingTheRows:
    def test_a_simple_file_becomes_one_dict_per_row(self) -> None:
        rows = routers._parse_csv_text(
            "email,given_name\na@test.local,Van A\nb@test.local,Thi B\n"
        )
        assert rows == [
            {"email": "a@test.local", "given_name": "Van A"},
            {"email": "b@test.local", "given_name": "Thi B"},
        ]

    def test_a_leading_byte_order_mark_is_stripped_again(self) -> None:
        """Stripped in both places because the text path never went through
        the utf-8-sig decode: a BOM pasted into ``csv_text`` would otherwise
        survive and rename the first column to ``\\ufeffemail``.
        """
        rows = routers._parse_csv_text(_BOM + "email\na@test.local\n")
        assert rows == [{"email": "a@test.local"}]

    def test_values_are_trimmed(self) -> None:
        """Spreadsheet exports carry padding, and an email with a trailing
        space is a different string from the same email without one."""
        rows = routers._parse_csv_text("email,given_name\n  a@test.local , Van A \n")
        assert rows == [{"email": "a@test.local", "given_name": "Van A"}]

    def test_an_empty_cell_becomes_an_empty_string(self) -> None:
        """The CSV reader yields ``None`` for a short row's missing tail;
        the row schema expects strings or absence, not nulls."""
        rows = routers._parse_csv_text("email,given_name\na@test.local\n")
        assert rows[0]["given_name"] == ""

    def test_extra_values_beyond_the_headers_are_dropped(self) -> None:
        """A ragged row collects its surplus under a ``None`` key.

        Left in, that key reaches a row schema that forbids extra fields,
        and the line fails for a reason the manager cannot see in their
        spreadsheet -- there is no column to point at.
        """
        rows = routers._parse_csv_text("email\na@test.local,surplus,more\n")

        assert None not in rows[0]
        assert rows[0] == {"email": "a@test.local"}

    def test_a_header_only_file_yields_no_rows(self) -> None:
        assert routers._parse_csv_text("email,given_name\n") == []

    def test_an_empty_file_yields_no_rows(self) -> None:
        assert routers._parse_csv_text("") == []

    def test_quoted_values_containing_commas_survive(self) -> None:
        """A display name with a comma in it is ordinary, and splitting on
        commas by hand would silently shift every later column."""
        rows = routers._parse_csv_text('email,display_name\na@test.local,"Nguyen, Van A"\n')
        assert rows[0]["display_name"] == "Nguyen, Van A"

    def test_windows_line_endings_are_handled(self) -> None:
        """The file comes off a Windows machine more often than not."""
        rows = routers._parse_csv_text("email\r\na@test.local\r\nb@test.local\r\n")
        assert [r["email"] for r in rows] == ["a@test.local", "b@test.local"]

    def test_unknown_columns_are_preserved_for_the_row_schema_to_reject(self) -> None:
        """The reader does not vet headers -- the row schema does, and it
        forbids extras so a misspelled column fails loudly rather than being
        silently ignored on every line.
        """
        rows = routers._parse_csv_text("email,nickname\na@test.local,Bo\n")
        assert rows[0] == {"email": "a@test.local", "nickname": "Bo"}


class TestTheImportEndpoint:
    async def test_an_unreadable_file_is_a_400_before_the_service_runs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A malformed file is the caller's problem, not a conflict, and it
        must not be mapped through the service's 409 vocabulary."""

        def _explode(_text: str) -> list[dict[str, str]]:
            raise csv.Error("field larger than field limit")

        monkeypatch.setattr(routers, "_parse_csv_text", _explode)
        service = AsyncMock()
        monkeypatch.setattr(routers.services, "import_students_from_csv", service)
        db = SimpleNamespace(commit=AsyncMock())

        with pytest.raises(HTTPException) as raised:
            await routers.import_students_csv(
                uuid4(),
                SimpleNamespace(csv_text="anything", csv_base64=None),
                object(),
                db,
            )

        assert raised.value.status_code == 400
        assert raised.value.detail["error"] == "invalid_csv"
        service.assert_not_awaited()
        db.commit.assert_not_awaited()

    async def test_a_successful_import_commits(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            routers.services, "import_students_from_csv", AsyncMock(return_value="done")
        )
        db = SimpleNamespace(commit=AsyncMock())

        result = await routers.import_students_csv(
            uuid4(),
            SimpleNamespace(csv_text=EMAIL_FILE, csv_base64=None),
            object(),
            db,
        )

        assert result == "done"
        db.commit.assert_awaited_once_with()

    async def test_a_refused_import_is_mapped_and_not_committed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The gate refusals -- unpublished program, no version, not an
        operator -- still apply to a file, and they abort it as a whole
        rather than appearing once per row.
        """
        monkeypatch.setattr(
            routers.services,
            "import_students_from_csv",
            AsyncMock(
                side_effect=ConflictError("only_published_programs_accept_enrollments")
            ),
        )
        db = SimpleNamespace(commit=AsyncMock())

        with pytest.raises(HTTPException) as raised:
            await routers.import_students_csv(
                uuid4(),
                SimpleNamespace(csv_text=EMAIL_FILE, csv_base64=None),
                object(),
                db,
            )

        assert raised.value.status_code == 409
        assert raised.value.detail["error"] == "only_published_programs_accept_enrollments"
        db.commit.assert_not_awaited()

    async def test_the_service_only_ever_sees_parsed_rows(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The router owns the file format entirely.

        Pushing the text down would make the service care about BOMs and
        base64 envelopes, which is why the two halves are split here.
        """
        service = AsyncMock(return_value="done")
        monkeypatch.setattr(routers.services, "import_students_from_csv", service)

        await routers.import_students_csv(
            uuid4(),
            SimpleNamespace(csv_text=TWO_ROW_FILE, csv_base64=None),
            object(),
            SimpleNamespace(commit=AsyncMock()),
        )

        assert service.await_args.kwargs["rows"] == [
            {"email": "a@test.local"},
            {"email": "b@test.local"},
        ]


def test_the_import_endpoint_answers_200_not_201() -> None:
    """A run can legitimately create nothing: re-uploading last week's file
    reports everyone under ``already_enrolled`` and writes no new rows, so
    201 would be a lie about half the time.
    """
    route = next(
        r
        for r in routers.management_router.routes
        if getattr(r, "name", None) == "import_students_csv"
    )
    assert route.status_code in (None, 200)


def test_the_roster_payload_refuses_unknown_fields() -> None:
    """``extra="forbid"`` turns a typo in the SPA's request body into a 422
    rather than a silently ignored field."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        routers.ProgramCsvImportPayload(csv_text=EMAIL_FILE, rows=[{"email": "x"}])
