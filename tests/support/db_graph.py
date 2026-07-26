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

    # --- Phase 1: collect the doomed closure -----------------------------
    closure: dict[str, set[str]] = {root_table: set(root_ids)}
    frontier: list[tuple[str, list[str]]] = [(root_table, root_ids)]
    while frontier:
        table, ids = frontier.pop()
        for child, col in graph.get(table, ()):
            if child not in has_id_pk:
                continue  # link table: no PK, nothing can reference it
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

    remaining = set(closure)
    while remaining:
        ready = [t for t in remaining if not (blocked[t] & remaining)]
        if not ready:  # pragma: no cover - would mean a NOT NULL FK cycle
            ready = list(remaining)
        for table in ready:
            # Link tables referencing this one go first (they carry no PK and
            # are not part of the closure).
            for child, col in graph.get(table, ()):
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
