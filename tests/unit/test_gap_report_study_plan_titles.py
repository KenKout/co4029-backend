"""Study-plan resource-title projection (gap report).

The gap report persists only resource UUIDs in ``report_json``. The projection
layer resolves those UUIDs to human titles at read time (repairing existing
reports without a migration). These cover the two pure helpers that shape the
study plan; the async DB resolver (`_resolve_resource_titles`) is exercised by
the router integration tests.
"""

from abridgeai.features.interviews.routers.learner import (
    _apply_resource_titles,
    _study_plan_from_report,
)


def test_study_plan_from_report_extracts_entries_with_uuid_resources() -> None:
    report_json = {
        "study_plan": [
            {
                "topic": "Data Warehouse Foundations",
                "suggested_lesson_id": "46741c12-ff9d-48ac-9804-24753f6386eb",
                "suggested_resource_ids": ["e601a288-8c83-4ffe-91fb-0519d3265986"],
            },
            {
                "topic": "Professional Interviewing",
                "suggested_lesson_id": None,
                "suggested_resource_ids": ["e601a288-8c83-4ffe-91fb-0519d3265986"],
            },
        ]
    }

    plan = _study_plan_from_report(report_json)

    assert [item["topic"] for item in plan] == [
        "Data Warehouse Foundations",
        "Professional Interviewing",
    ]
    assert plan[0]["lesson_id"] == "46741c12-ff9d-48ac-9804-24753f6386eb"
    # Resources start as raw UUID strings before resolution.
    assert plan[0]["suggested_resources"] == ["e601a288-8c83-4ffe-91fb-0519d3265986"]


def test_study_plan_from_report_tolerates_missing_or_malformed_payload() -> None:
    assert _study_plan_from_report(None) == []
    assert _study_plan_from_report({}) == []
    assert _study_plan_from_report({"study_plan": "not-a-list"}) == []
    # Non-dict entries are skipped rather than crashing.
    assert _study_plan_from_report({"study_plan": ["oops", 42]}) == []


def test_apply_resource_titles_replaces_uuids_with_titles() -> None:
    plan = [
        {
            "topic": "Data Warehouse Foundations",
            "lesson_id": None,
            "suggested_resources": ["e601a288-8c83-4ffe-91fb-0519d3265986"],
        }
    ]
    titles = {"e601a288-8c83-4ffe-91fb-0519d3265986": "Chapter 1 - Overview.pdf"}

    _apply_resource_titles(plan, titles)

    assert plan[0]["suggested_resources"] == ["Chapter 1 - Overview.pdf"]


def test_apply_resource_titles_drops_unresolvable_ids() -> None:
    # An id with no title (deleted resource / stale report) is dropped, never
    # shown as a raw UUID. The item can legitimately end up with no resources.
    plan = [
        {
            "topic": "Professional Interviewing",
            "lesson_id": None,
            "suggested_resources": [
                "e601a288-8c83-4ffe-91fb-0519d3265986",
                "deadbeef-0000-0000-0000-000000000000",
            ],
        }
    ]
    titles = {"e601a288-8c83-4ffe-91fb-0519d3265986": "Chapter 1 - Overview.pdf"}

    _apply_resource_titles(plan, titles)

    assert plan[0]["suggested_resources"] == ["Chapter 1 - Overview.pdf"]
