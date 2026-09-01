"""End-to-end exercise of the path-change review flow against the live dev DB.

Drives the real service layer (not HTTP) for: request -> mark in progress ->
reject with reason, and request -> approve, asserting the DB state and the
notifications produced at each step. Cleans up everything it creates.

Run: uv run --no-sync python scripts/verify_path_change_review.py
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from abridgeai.core.config import get_settings
from abridgeai.core.security import CurrentUser
from abridgeai.features.learning_programs import services


@dataclass
class Ctx:
    org_id: uuid.UUID
    faculty_id: uuid.UUID
    dean_id: uuid.UUID
    student_id: uuid.UUID
    program_id: uuid.UUID
    version_id: uuid.UUID
    enrollment_id: uuid.UUID
    path_a: uuid.UUID
    path_b: uuid.UUID


async def _pick_existing(db) -> Ctx:
    """Reuse live rows: an org, a faculty, two published paths with versions."""
    org_id = (await db.execute(text("SELECT id FROM organizations LIMIT 1"))).scalar_one()
    faculty_id = (
        await db.execute(
            text("SELECT id FROM org_units WHERE organization_id = :o LIMIT 1"), {"o": org_id}
        )
    ).scalar_one()
    paths = (
        await db.execute(
            text("""
            SELECT cp.id AS path_id, cpv.id AS version_id
            FROM career_paths cp
            JOIN career_path_versions cpv ON cpv.career_path_id = cp.id
            WHERE cp.organization_id = :o AND cp.status = 'published'
              AND cpv.status = 'published' AND cp.deleted_at IS NULL
            ORDER BY cp.created_at LIMIT 2
            """),
            {"o": org_id},
        )
    ).all()
    if len(paths) < 2:
        raise SystemExit("need two published career paths in the dev DB")

    dean_id = uuid.uuid4()
    student_id = uuid.uuid4()
    for uid, email in ((dean_id, "verify-dean"), (student_id, "verify-student")):
        # Membership-first identity: `users` carries no organization_id; org
        # affiliation lives in organization_memberships.
        await db.execute(
            text("""
            INSERT INTO users (id, primary_email, status, created_at, updated_at)
            VALUES (:id, :email, 'active', NOW(), NOW())
            """),
            {"id": uid, "email": f"{email}-{uid.hex[:8]}@verify.local"},
        )
        await db.execute(
            text("""
            INSERT INTO organization_memberships (id, user_id, organization_id, org_unit_id,
                                                  status, joined_at, created_at, updated_at)
            VALUES (gen_random_uuid(), :u, :o, :f, 'active', NOW(), NOW(), NOW())
            """),
            {"u": uid, "o": org_id, "f": faculty_id},
        )

    # Faculty-dean scope: org_unit-scoped 'hod' role assignment PLUS an active
    # faculty assignment — actor_has_program_role requires both.
    hod_role_id = (
        await db.execute(text("SELECT id FROM roles WHERE code = 'hod' LIMIT 1"))
    ).scalar_one()
    await db.execute(
        text("""
        INSERT INTO user_role_assignments (id, user_id, role_id, scope_kind, organization_id,
                                           org_unit_id, active_from, created_at, updated_at)
        VALUES (gen_random_uuid(), :u, :r, 'org_unit', :o, :f, NOW(), NOW(), NOW())
        """),
        {"u": dean_id, "r": hod_role_id, "o": org_id, "f": faculty_id},
    )
    await db.execute(
        text("""
        INSERT INTO user_faculty_assignments (id, user_id, organization_id, faculty_id, status,
                                              active_from, created_at, updated_at)
        VALUES (gen_random_uuid(), :u, :o, :f, 'active', NOW(), NOW(), NOW())
        """),
        {"u": dean_id, "o": org_id, "f": faculty_id},
    )

    program_id = uuid.uuid4()
    version_id = uuid.uuid4()
    slug = f"verify-{program_id.hex[:8]}"
    await db.execute(
        text("""
        INSERT INTO learning_programs (id, organization_id, faculty_id, owner_faculty_dean_id,
                                       slug, name, status, created_at, updated_at)
        VALUES (:id, :o, :f, :d, :slug, 'Verify Program', 'published', NOW(), NOW())
        """),
        {"id": program_id, "o": org_id, "f": faculty_id, "d": dean_id, "slug": slug},
    )
    await db.execute(
        text("""
        INSERT INTO learning_program_versions (id, learning_program_id, version_no, status,
                                               max_path_switches, created_at, updated_at)
        VALUES (:id, :p, 1, 'published', 3, NOW(), NOW())
        """),
        {"id": version_id, "p": program_id},
    )
    for pos, row in enumerate(paths, start=1):
        # Composite PK (program_version_id, career_path_id) — no id, no updated_at.
        await db.execute(
            text("""
            INSERT INTO learning_program_version_paths
                (program_version_id, career_path_id, career_path_version_id, position, created_at)
            VALUES (:v, :cp, :cpv, :pos, NOW())
            """),
            {"v": version_id, "cp": row.path_id, "cpv": row.version_id, "pos": pos},
        )

    enrollment_id = uuid.uuid4()
    await db.execute(
        text("""
        INSERT INTO program_enrollments (id, learning_program_id, program_version_id, student_id,
                                          status, enrolled_at, created_at, updated_at)
        VALUES (:id, :p, :v, :s, 'active', NOW(), NOW(), NOW())
        """),
        {"id": enrollment_id, "p": program_id, "v": version_id, "s": student_id},
    )
    await db.execute(
        text("""
        INSERT INTO program_path_attempts (id, program_enrollment_id, career_path_id,
                                           career_path_version_id, status, selected_at,
                                           created_at, updated_at)
        VALUES (gen_random_uuid(), :e, :cp, :cpv, 'active', NOW(), NOW(), NOW())
        """),
        {"e": enrollment_id, "cp": paths[0].path_id, "cpv": paths[0].version_id},
    )
    await db.commit()
    return Ctx(
        org_id=org_id,
        faculty_id=faculty_id,
        dean_id=dean_id,
        student_id=student_id,
        program_id=program_id,
        version_id=version_id,
        enrollment_id=enrollment_id,
        path_a=paths[0].path_id,
        path_b=paths[1].path_id,
    )


async def _notifications(db, student_id: uuid.UUID) -> list[tuple[str, str, str]]:
    rows = (
        await db.execute(
            text("""
            SELECT category, title, body FROM notifications
            WHERE user_id = :u ORDER BY created_at
            """),
            {"u": student_id},
        )
    ).all()
    return [(r.category, r.title, r.body) for r in rows]


async def _cleanup(db, ctx: Ctx) -> None:
    for stmt, params in [
        ("DELETE FROM notifications WHERE user_id IN (:s, :d)", {"s": ctx.student_id, "d": ctx.dean_id}),
        ("DELETE FROM path_change_requests WHERE program_enrollment_id = :e", {"e": ctx.enrollment_id}),
        ("DELETE FROM course_enrollment_entitlements WHERE source_id IN (SELECT id FROM program_path_attempts WHERE program_enrollment_id = :e)", {"e": ctx.enrollment_id}),
        ("DELETE FROM program_path_attempts WHERE program_enrollment_id = :e", {"e": ctx.enrollment_id}),
        ("DELETE FROM program_enrollments WHERE id = :e", {"e": ctx.enrollment_id}),
        ("DELETE FROM learning_program_version_paths WHERE program_version_id = :v", {"v": ctx.version_id}),
        ("DELETE FROM learning_program_versions WHERE id = :v", {"v": ctx.version_id}),
        ("DELETE FROM learning_programs WHERE id = :p", {"p": ctx.program_id}),
        ("DELETE FROM user_faculty_assignments WHERE user_id = :d", {"d": ctx.dean_id}),
        ("DELETE FROM user_role_assignments WHERE user_id = :d", {"d": ctx.dean_id}),
        ("DELETE FROM student_career_enrollments WHERE student_id = :s", {"s": ctx.student_id}),
        ("DELETE FROM course_enrollments WHERE student_id = :s", {"s": ctx.student_id}),
        ("DELETE FROM organization_memberships WHERE user_id IN (:s, :d)", {"s": ctx.student_id, "d": ctx.dean_id}),
        ("DELETE FROM users WHERE id IN (:s, :d)", {"s": ctx.student_id, "d": ctx.dean_id}),
    ]:
        await db.execute(text(stmt), params)
    await db.commit()


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(str(settings.database_url), echo=False)
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"{'PASS' if ok else 'FAIL'}  {label}{(' — ' + detail) if detail else ''}")
        if not ok:
            failures.append(label)

    async with factory() as db:
        ctx = await _pick_existing(db)

    try:
        dean = CurrentUser(ctx.dean_id, uuid.uuid4())

        # ── 1. student files a request ────────────────────────────────────
        async with factory() as db:
            req = await services.request_path_change(
                db,
                enrollment_id=ctx.enrollment_id,
                target_path_id=ctx.path_b,
                reason="I want to move into data engineering.",
                student_id=ctx.student_id,
            )
            await db.commit()
        check("request created as pending", req.status == "pending", req.status)

        # ── 2. dean marks it in progress ──────────────────────────────────
        async with factory() as db:
            ack = await services.mark_change_request_in_progress(
                db, request_id=req.id, actor=dean
            )
            await db.commit()
        check("status -> in_progress", ack.status == "in_progress", ack.status)
        check("in_progress_at stamped", ack.in_progress_at is not None)
        check("in_progress_by = dean", ack.in_progress_by == ctx.dean_id)

        async with factory() as db:
            notes = await _notifications(db, ctx.student_id)
        check("in_progress notification sent", len(notes) == 1, f"{len(notes)} rows")
        if notes:
            cat, title, body = notes[0]
            check("notification category", cat == "path_change_review", cat)
            check(
                "body says nothing changed yet",
                "Nothing has changed yet" in body,
                body[:90],
            )

        # ── 3. acknowledging twice is idempotent, not a 409 ───────────────
        async with factory() as db:
            again = await services.mark_change_request_in_progress(
                db, request_id=req.id, actor=dean
            )
            await db.commit()
        check("re-acknowledge is a no-op", again.status == "in_progress")
        async with factory() as db:
            notes = await _notifications(db, ctx.student_id)
        check("no duplicate notification", len(notes) == 1, f"{len(notes)} rows")

        # ── 4. a second open request is refused while one is in progress ──
        async with factory() as db:
            try:
                await services.request_path_change(
                    db,
                    enrollment_id=ctx.enrollment_id,
                    target_path_id=ctx.path_b,
                    reason="second attempt",
                    student_id=ctx.student_id,
                )
                await db.commit()
                check("second request blocked while in_progress", False, "no error raised")
            except Exception as exc:  # noqa: BLE001
                await db.rollback()
                check(
                    "second request blocked while in_progress",
                    "pending_path_change" in str(exc),
                    str(exc),
                )

        # ── 5. reject without a reason code is refused ────────────────────
        async with factory() as db:
            try:
                await services.decide_change_request(
                    db, request_id=req.id, approve=False, decision_reason=None, actor=dean
                )
                await db.commit()
                check("reject requires a reason code", False, "no error raised")
            except Exception as exc:  # noqa: BLE001
                await db.rollback()
                check(
                    "reject requires a reason code",
                    "rejection_reason_code_is_required" in str(exc),
                    str(exc),
                )

        # ── 6. 'other' without detail is refused ─────────────────────────
        async with factory() as db:
            try:
                await services.decide_change_request(
                    db,
                    request_id=req.id,
                    approve=False,
                    decision_reason="   ",
                    decision_reason_code="other",
                    actor=dean,
                )
                await db.commit()
                check("'other' requires written detail", False, "no error raised")
            except Exception as exc:  # noqa: BLE001
                await db.rollback()
                check(
                    "'other' requires written detail",
                    "code_is_other" in str(exc),
                    str(exc),
                )

        # ── 7. reject with a real reason ─────────────────────────────────
        async with factory() as db:
            rejected = await services.decide_change_request(
                db,
                request_id=req.id,
                approve=False,
                decision_reason="Finish stage 2 first.",
                decision_reason_code="progress_loss_too_high",
                actor=dean,
            )
            await db.commit()
        check("status -> rejected", rejected.status == "rejected", rejected.status)
        check(
            "reason code persisted",
            rejected.decision_reason_code == "progress_loss_too_high",
            str(rejected.decision_reason_code),
        )
        check(
            "in_progress_by survived the decision",
            rejected.in_progress_by == ctx.dean_id,
        )

        async with factory() as db:
            notes = await _notifications(db, ctx.student_id)
        check("rejection notification sent", len(notes) == 2, f"{len(notes)} rows")
        if len(notes) >= 2:
            _, title, body = notes[1]
            check("rejection title", "rejected" in title.lower(), title)
            check(
                "rejection body carries the canned reason",
                "progress you would lose" in body,
                body[:120],
            )
            check("rejection body carries the note", "Finish stage 2 first." in body, body[:160])
            check(
                "rejection body says no switch consumed",
                "does not use up" in body,
                body[-90:],
            )

        # ── 8. rejection costs no budget, and history is exposed ─────────
        async with factory() as db:
            enrollments = await services.list_my_enrollments(db, ctx.student_id)
        me = next(e for e in enrollments if e.id == ctx.enrollment_id)
        check("approved_switch_count still 0", me.approved_switch_count == 0, str(me.approved_switch_count))
        check("no open request after rejection", me.pending_change_request is None)
        check("history has 1 entry", len(me.change_request_history) == 1, str(len(me.change_request_history)))
        if me.change_request_history:
            h = me.change_request_history[0]
            check("history entry is the rejection", h.status == "rejected", h.status)
            check(
                "history carries the reason code",
                h.decision_reason_code == "progress_loss_too_high",
                str(h.decision_reason_code),
            )

        # ── 9. a fresh request can now be filed and approved ────────────
        async with factory() as db:
            req2 = await services.request_path_change(
                db,
                enrollment_id=ctx.enrollment_id,
                target_path_id=ctx.path_b,
                reason="Re-filing after finishing stage 2.",
                student_id=ctx.student_id,
            )
            await db.commit()
        check("new request allowed after rejection", req2.status == "pending", req2.status)

        # Approve straight from pending — acknowledging is optional, not a gate.
        async with factory() as db:
            approved = await services.decide_change_request(
                db, request_id=req2.id, approve=True, decision_reason=None, actor=dean
            )
            await db.commit()
        check("approve works without prior acknowledgement", approved.status == "approved", approved.status)
        check("new attempt created", approved.new_attempt_id is not None)

        async with factory() as db:
            notes = await _notifications(db, ctx.student_id)
        check("approval notification sent", len(notes) == 3, f"{len(notes)} rows")
        if len(notes) >= 3:
            _, title, body = notes[2]
            check("approval title", "approved" in title.lower(), title)

        async with factory() as db:
            enrollments = await services.list_my_enrollments(db, ctx.student_id)
        me = next(e for e in enrollments if e.id == ctx.enrollment_id)
        check("approved_switch_count now 1", me.approved_switch_count == 1, str(me.approved_switch_count))
        check("history has 2 entries", len(me.change_request_history) == 2, str(len(me.change_request_history)))
        active = [a for a in me.attempts if a.status == "active"]
        check(
            "active attempt moved to target path",
            len(active) == 1 and active[0].career_path_id == ctx.path_b,
        )
    finally:
        async with factory() as db:
            await _cleanup(db, ctx)
        await engine.dispose()

    print()
    if failures:
        print(f"{len(failures)} FAILURE(S): " + "; ".join(failures))
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    asyncio.run(main())
