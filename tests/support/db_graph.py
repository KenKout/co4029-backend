"""FK-graph-driven hard delete for test cleanup.

Shared by ``tests/conftest.py`` (seeded-fixture purge) and per-file cleanup
fixtures. Test suites used to hand-roll DELETE chains per file; every new FK
into a shared parent (courses, quizzes, users) broke a different file's
teardown with a ForeignKeyViolation at setup of unrelated tests. This walks
the LIVE FK graph instead, so a new dependent table is handled the day its
migration lands.

Works with any object exposing ``execute`` (AsyncSession or AsyncConnection).
Test-database only, by construction: the Settings model swaps
``database_url`` to ``test_database_url`` whenever pytest is imported.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import text

from abridgeai.core.audit.maintenance import audit_maintenance


async def _load_fk_graph(session: Any) -> dict[str, list[tuple[str, str]]]:
    """``parent_table -> [(child_table, child_fk_column), ...]`` for public FKs.

    Only FKs that target the parent's ``id`` column are included — that is
    every relationship in this schema, and it lets the deleter chase children
    by primary key alone.
    """
    rows = (
        await session.execute(
            text(
                """
                SELECT ccu.table_name AS parent,
                       tc.table_name  AS child,
                       kcu.column_name AS col
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                  ON kcu.constraint_name = tc.constraint_name
                JOIN information_schema.constraint_column_usage ccu
                  ON ccu.constraint_name = tc.constraint_name
                WHERE tc.constraint_type = 'FOREIGN KEY'
                  AND tc.table_schema = 'public'
                  AND ccu.column_name = 'id'
                """
            )
        )
    ).all()
    graph: dict[str, list[tuple[str, str]]] = {}
    for row in rows:
        graph.setdefault(row.parent, []).append((row.child, row.col))
    return graph


async def _tables_with_id_pk(session: Any) -> set[str]:
    rows = (
        await session.execute(
            text(
                "SELECT table_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND column_name = 'id'"
            )
        )
    ).scalars()
    return set(rows)


async def _append_only_tables(session: Any) -> set[str]:
    """Tables guarded by the ``audit_log_immutable`` trigger.

    The trigger refuses UPDATE unconditionally and DELETE unless
    ``app.audit_maintenance = 'on'``. Both matter here, because an audit row
    written by one test blocks the NEXT test's purge:

    * ``system_setting_changes.actor_id`` and ``http_audit_log.user_id`` are
      nullable FKs to ``users.id``, so the closure walk drags these tables in
      and the explicit DELETE trips the maintenance guard.
    * Both FKs are ``ON DELETE SET NULL``, so even skipping them is not enough —
      deleting the parent ``users`` row makes Postgres issue the UPDATE itself,
      which the trigger rejects outright.

    So the purge both skips them (below) and runs inside the maintenance scope
    (see ``hard_delete_graph``), which is exactly the scope's purpose: retention
    cleanup. Discovered from the live catalog for the same reason the FK graph
    is — a new append-only table is handled the day its migration lands.
    """
    rows = (
        await session.execute(
            text(
                """
                SELECT DISTINCT tg.tgrelid::regclass::text AS table_name
                FROM pg_trigger tg
                JOIN pg_proc p ON p.oid = tg.tgfoid
                WHERE NOT tg.tgisinternal
                  AND p.proname = 'audit_log_immutable'
                """
            )
        )
    ).scalars()
    return set(rows)


async def _nullable_columns(session: Any) -> set[tuple[str, str]]:
    rows = (
        await session.execute(
            text(
                "SELECT table_name, column_name FROM information_schema.columns "
                "WHERE table_schema = 'public' AND is_nullable = 'YES'"
            )
        )
    ).all()
    return {(row.table_name, row.column_name) for row in rows}


async def hard_delete_graph(  # noqa: C901 -- three-phase FK-cycle walk reads better in one place
    session: Any,
    root_table: str,
    root_ids: list[str],
) -> None:
    """Hard-delete rows and everything that transitively references them.

    Three phases, because the schema has FK CYCLES — e.g.
    ``learning_materials.current_version_id -> learning_material_versions`` and
    ``learning_material_versions.material_id -> learning_materials`` reference
    each other, so no pure delete order exists:

    1. COLLECT the closure of doomed row ids per table by walking the live FK
       graph from the roots (a visited set makes self-FK chains terminate).
    2. SEVER every NULLABLE FK among doomed rows with UPDATE ... SET col=NULL.
       Any FK cycle must contain at least one nullable edge (rows could never
       have been inserted otherwise), so this leaves the NOT-NULL subgraph
       acyclic. The rows are about to die, so nulling is harmless.
    3. DELETE child-before-parent (Kahn's ordering over the remaining NOT-NULL
       edges), flushing link tables (no ``id`` column) ahead of each parent.

    Test-database only, by construction: the Settings model swaps
    ``database_url`` to ``test_database_url`` whenever pytest is imported.
    """
    graph = await _load_fk_graph(session)
    has_id_pk = await _tables_with_id_pk(session)
    nullable = await _nullable_columns(session)
    append_only = await _append_only_tables(session)
    if append_only:
        # Enter the retention-maintenance scope for this transaction (the same
        # opt-in production retention jobs use). Deleting a seeded user drags an
        # ``ON DELETE SET NULL`` into the append-only audit tables, and their
        # trigger rejects that UPDATE outright — so the purge cannot complete
        # outside this scope once any test has written an audit row. SET LOCAL
        # means the permission dies with the transaction.
        await audit_maintenance(session)

    # --- Phase 1: collect the doomed closure -----------------------------
    closure: dict[str, set[str]] = {root_table: set(root_ids)}
    frontier: list[tuple[str, list[str]]] = [(root_table, root_ids)]
    while frontier:
        table, ids = frontier.pop()
        for child, col in graph.get(table, ()):
            if child not in has_id_pk:
                continue  # link table: no PK, nothing can reference it
            if child in append_only:
                # Immutable audit trail: cannot be UPDATEd or DELETEd here. Its
                # FK into the doomed parent is ON DELETE SET NULL, so Postgres
                # severs the link itself when the parent goes.
                continue
            child_ids = {
                str(v)
                for v in (
                    await session.execute(
                        text(
                            f"SELECT id FROM {child} "  # noqa: S608 -- identifiers come from information_schema, not user input
                            f"WHERE {col} = ANY(CAST(:ids AS uuid[]))"
                        ),
                        {"ids": ids},
                    )
                ).scalars()
            }
            fresh = child_ids - closure.get(child, set())
            if fresh:
                closure.setdefault(child, set()).update(fresh)
                frontier.append((child, list(fresh)))

    # --- Phase 2: sever nullable FK edges INSIDE cycles only -------------
    # Nulling every nullable FK would trip CHECK constraints (module_items has
    # an XOR check over its three target columns), so first find the cyclic
    # core: run Kahn's elimination over ALL in-closure edges; whatever cannot
    # be ordered is a cycle, and only nullable edges within it are severed.
    # Any FK cycle must contain a nullable edge — the rows could never have
    # been inserted otherwise.
    all_blocked: dict[str, set[str]] = {t: set() for t in closure}
    for parent, edges in graph.items():
        if parent not in closure:
            continue
        for child, _col in edges:
            if child in closure and child != parent:
                all_blocked[parent].add(child)
    acyclic = set(closure)
    progressed = True
    while progressed:
        progressed = False
        for table in list(acyclic):
            if not (all_blocked[table] & acyclic - {table}):
                acyclic.discard(table)
                progressed = True
    cyclic_core = acyclic  # tables Kahn could not eliminate

    severed: set[tuple[str, str, str]] = set()
    for parent, edges in graph.items():
        if parent not in cyclic_core:
            continue
        for child, col in edges:
            if child in cyclic_core and (child, col) in nullable:
                severed.add((parent, child, col))
                await session.execute(
                    text(
                        f"UPDATE {child} SET {col} = NULL "  # noqa: S608
                        f"WHERE id = ANY(CAST(:ids AS uuid[]))"
                    ),
                    {"ids": list(closure[child])},
                )

    # --- Phase 3: delete child-before-parent over surviving edges --------
    blocked: dict[str, set[str]] = {t: set() for t in closure}
    for parent, edges in graph.items():
        if parent not in closure:
            continue
        for child, col in edges:
            if child in closure and child != parent and (parent, child, col) not in severed:
                blocked[parent].add(child)

    # --- Phase 2b: delete append-only rows pointing at doomed parents -----
    # These tables are skipped by the closure walk (their trigger forbids the
    # UPDATE that phase 2 would use to sever), but their ``ON DELETE SET NULL``
    # FKs would still make Postgres attempt that same forbidden UPDATE when the
    # parent row goes. Deleting the referencing rows first — legal inside the
    # maintenance scope set above — removes the edge entirely.
    for parent, edges in graph.items():
        if parent not in closure:
            continue
        for child, col in edges:
            if child not in append_only:
                continue
            await session.execute(
                text(
                    f"DELETE FROM {child} "  # noqa: S608 -- identifiers come from information_schema
                    f"WHERE {col} = ANY(CAST(:ids AS uuid[]))"
                ),
                {"ids": list(closure[parent])},
            )

    remaining = set(closure)
    while remaining:
        ready = [t for t in remaining if not (blocked[t] & remaining)]
        if not ready:  # pragma: no cover - would mean a NOT NULL FK cycle
            ready = list(remaining)
        for table in ready:
            # Link tables referencing this one go first (they carry no PK and
            # are not part of the closure).
            for child, col in graph.get(table, ()):
                if child in append_only:
                    continue  # immutable audit trail — see _append_only_tables
                if child not in has_id_pk:
                    await session.execute(
                        text(
                            f"DELETE FROM {child} "  # noqa: S608
                            f"WHERE {col} = ANY(CAST(:ids AS uuid[]))"
                        ),
                        {"ids": list(closure[table])},
                    )
            await session.execute(
                text(f"DELETE FROM {table} WHERE id = ANY(CAST(:ids AS uuid[]))"),  # noqa: S608
                {"ids": list(closure[table])},
            )
            remaining.discard(table)


__all__ = ["hard_delete_graph"]
