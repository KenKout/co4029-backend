# Characterization Snapshot Helpers

Tiny in-tree snapshot harness for proving query-level behavior preservation
during the ORM consolidation migration. No external deps.

## Workflow

```
[pre-migration]   capture_snapshot(...)  --> tests/fixtures/snapshots/{name}.json   (committed to git)
[migrate code]    refactor the query
[post-migration]  assert_snapshot_match(...)  --> compares; raises on drift
```

## Usage

```python
from tests._helpers.snapshot import capture_snapshot, assert_snapshot_match

async def get_lesson_to_course(db, *, lesson_id):
    result = await db.execute(
        text("SELECT id, title FROM courses WHERE lesson_id = :lid ORDER BY id"),
        {"lid": lesson_id},
    )
    return result.all()

# step 1 -- baseline (run once, then check the JSON into git)
await capture_snapshot(
    db,
    get_lesson_to_course,
    name="courses__lesson_to_course",
    params={"lesson_id": LESSON_UUID},
)

# step 2 -- after the migration, the post-migration test re-runs:
await assert_snapshot_match(
    db,
    get_lesson_to_course,
    name="courses__lesson_to_course",
    params={"lesson_id": LESSON_UUID},
)
```

## Caveats

- **Order-sensitive.** Add a deterministic `ORDER BY` to the query.
  The helper does not sort.
- **Timestamps round to the second.** Microseconds are dropped to keep
  diffs stable across DB round-trips. If a test depends on sub-second
  precision, this helper is the wrong tool.
- **`UUID` becomes `str`.** Comparison is string-based.
- **`Decimal` becomes `str`.** Preserves precision; do not switch to
  `float`.
- **`bytes` becomes base64 ASCII.**
- **`set` / `frozenset` are sorted.**
- **Run-once capture.** `capture_snapshot` skips if the file already
  exists. Delete the JSON manually to re-baseline.
- **Mismatch evidence** lands at
  `.sisyphus/evidence/orm-consolidation/snapshot-mismatch-{name}.json`
  with the full `expected` / `actual` payloads.
- **Time freezing is the caller's job.** This helper does not freeze
  time. If the query reads `NOW()`, freeze in the test.
