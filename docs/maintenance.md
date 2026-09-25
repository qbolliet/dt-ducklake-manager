# Storage lifecycle and maintenance

DuckLake's Parquet files are immutable. Every `UPDATE`/`DELETE`/schema change
produces new files rather than rewriting existing ones, and old files stay
referenced by prior snapshots (time travel). Left alone, small files and
delete tombstones accumulate and degrade read performance. This page explains
the physical model, the operations available, when to run each one, and how
to read what an operation actually did.

## Copy-on-write and delete files

*(measured on DuckDB 1.5.2 / `ducklake` extension, see the specification's
annex for the exact commands.)*

An `UPDATE` of 5 000 rows on a 20 000-row file produces: a new 5 000-row data
file (carrying `_ducklake_internal_row_id`), a `…-delete.parquet` tombstone
file naming 5 000 positions in the *old* file, and the old file itself left
intact — still referenced by the snapshot that preceded the update. A raw
listing of the data directory (`read_parquet('**/*.parquet')`) at that point
shows 30 000 rows across three files from different snapshots: **this is an
artifact of reading the directory directly**, not a data problem — the
DuckLake-aware logical view (`SELECT * FROM catalog.schema.table`) is correct
at every instant.

A `DELETE` that removes *every* row of a file behaves differently: no delete
file is created at all, the file is simply dropped from the current snapshot
(`delete_file_count = 0` after `DELETE FROM t` with no `WHERE` clause).

## Data inlining

Data inlining is active by default: a small `INSERT` produces **no** Parquet
file — the rows live in the catalog itself
(`{inlined_insert=[1]}` in the resulting snapshot). Controlled by the `ATTACH`
option `DATA_INLINING_ROW_LIMIT` (`0` disables it entirely — useful for tests
that inspect files directly) or set later with
`set_option('data_inlining_row_limit', n)`. Inlined rows are written out to
Parquet by
[`DuckLakeMaintenance.flush_inlined_data`][dt_ducklake_manager.maintenance.procedures.DuckLakeProcedures.flush_inlined_data],
which reports `(schema, table, rows_flushed)` per table.

Suggested policy: `0` for the initial build (so every row lands in Parquet
immediately); a few thousand rows during incremental updates; flush as part of
planned maintenance before any process reads the data files directly.

## Operations: effect, when, risk

| Operation | Effect | When | Risk |
|---|---|---|---|
| [`rewrite_data_files(delete_threshold)`][dt_ducklake_manager.maintenance.procedures.DuckLakeProcedures.rewrite_data_files] | Rewrites files whose deleted-row share exceeds the threshold; **without an explicit threshold it is a true no-op**, even at 25% deletions | After every update/delete (`delete_threshold` 0.1–0.3) | None — old files stay readable via time travel |
| [`merge_files(max_file_size)`][dt_ducklake_manager.maintenance.procedures.DuckLakeProcedures.merge_files] | Merges adjacent files into files of at most `max_file_size` bytes (the catalog's `target_file_size` by default). `min_file_size` is a **lower** bound: files *smaller* than it are left out of the merge, so it is not set by default | After every write (through `compact`); after `recluster` | None |
| [`flush_inlined_data`][dt_ducklake_manager.maintenance.procedures.DuckLakeProcedures.flush_inlined_data] | Writes inlined catalog rows out to Parquet | Planned maintenance; before reading files directly | None |
| [`recluster(order_by)`][dt_ducklake_manager.maintenance.policy.DuckLakeMaintenance.recluster] | Rewrites the whole table in `cluster_by` order | When file-range overlap degrades pruning (see [Recluster and the overlap indicator](#recluster-and-the-overlap-indicator)), typically after N updates | Full rewrite; storage doubles until cleanup |
| [`set_partitioned_by` / `repartition`][dt_ducklake_manager.maintenance.procedures.DuckLakeProcedures.repartition] | Changes the partition keys (future writes only, or with a rewrite) | Filter strategy change on a very low cardinality column (see [Partitioning](#partitioning)) | `repartition`: full rewrite, same as `recluster` |
| [`expire_snapshots(older_than_days)`][dt_ducklake_manager.maintenance.procedures.DuckLakeProcedures.expire_snapshots] | Makes snapshots older than the cutoff unreachable | **Planned maintenance only**, with an explicit retention | **Destroys time travel** beyond the retention |
| [`cleanup_files`][dt_ducklake_manager.maintenance.procedures.DuckLakeProcedures.cleanup_files] | Deletes files no live snapshot references | After `expire_snapshots` — the only step that actually frees disk space | Irreversible |
| [`delete_orphaned_files`][dt_ducklake_manager.maintenance.procedures.DuckLakeProcedures.delete_orphaned_files] | Deletes files under `data_path` unknown to the catalog | After an incident (interrupted transaction, manual file copy); always `dry_run` first | Irreversible |

## The cycle: rewrite → merge → expire → cleanup

*Rewrite* (rewrite/merge/flush) happens after every write and is always safe:
old files stay reachable through time travel. *Expire* and *cleanup* only
happen in planned maintenance, because they are what actually destroy
recoverability and free space.

The write operations (`DatabaseUpdater.update_database`,
`DatabaseUpdater.add_columns`, `DatabaseDeleter.delete_rows`) call only the
safe half of this cycle automatically, right after their own commit, through
[`DuckLakeMaintenance.compact`][dt_ducklake_manager.maintenance.procedures.DuckLakeProcedures.compact]
(`merge_files` up to the catalog's `target_file_size`, then
`rewrite_data_files`; `compact_after_update=True` by default). On a
connection with no DuckLake catalog attached (in-memory tests), the compaction
is skipped. They never call `expire_snapshots`, `cleanup_files` or
`delete_orphaned_files` — those are only ever triggered by a deliberate,
policy-driven call to `maintain`.

## Recluster and the overlap indicator

DuckLake prunes files using per-file min/max statistics on `cluster_by`
columns (`ducklake_file_column_stats`, surfaced as `Total Files Read` in
`EXPLAIN ANALYZE`), plus Parquet row-group statistics inside each file. Both
only help when the data is physically grouped by those columns — which
successive updates degrade: each batch is sorted *within itself* at write
time, but not merged into the table's global order, so file ranges
progressively overlap.

[`storage_report()`][dt_ducklake_manager.maintenance.policy.DuckLakeMaintenance.storage_report]
measures `overlap_ratio` — the share of active, non-empty files whose
`[min, max]` range on the first `cluster_by` column overlaps another file's
range (strict inequalities: two files that merely share a boundary value do
not count as overlapping; identical ranges do). This is the indicator that
decides when [`recluster`][dt_ducklake_manager.maintenance.policy.DuckLakeMaintenance.recluster]
is worth its cost: a full-table rewrite (`CREATE TEMP TABLE … AS SELECT *`,
`DELETE FROM table`, `INSERT … ORDER BY cluster_by`, single transaction,
`threads = 1` for the insert so files come out disjoint and monotone rather
than interleaved) that temporarily doubles storage until the previous files
are released by `expire_snapshots` + `cleanup_files`.

**Recluster only matters beyond a few million rows.** File-level
pruning only starts once the table spans several files, i.e. beyond roughly
5.5 million rows. Even with the target forced down to 2 MB (25 files after 10
updates), dashboard queries stayed between 13 and 36 ms whether they read 1 file
or 25. Below a few million rows, leave `recluster=False` and let
`storage_report().overlap_ratio` grow: it is harmless.

## Partitioning

Partitioning complements the physical sort on `cluster_by`; it is not a
substitute for it. DuckLake writes one directory per partition value, so it only
pays off on a column of **very low cardinality** that nearly every query filters
on (e.g. a model version or a year), and it multiplies small files on anything
else. The build accepts `partition_by`; on an existing table:

```python
maintenance.set_partitioned_by("fact_table", partition_by=["model_version"])
maintenance.reset_partitioned_by("fact_table")              # future writes only
maintenance.repartition("fact_table", partition_by=["year(date)"])  # + rewrite
```

`set_partitioned_by` and `reset_partitioned_by` only affect the files written
afterwards; `repartition` resets, applies the new keys and, unless
`run_maintenance=False`, merges and rewrites the existing files so that they
adopt the new layout — a full rewrite, with the same storage cost as
`recluster`. Partition expressions are passed as is (`'country'`,
`'year(ts)'`, `'month(ts)'`, `'bucket(8, user_id)'`).

## `MaintenancePolicy` and `maintain`

[`MaintenancePolicy`][dt_ducklake_manager.maintenance.policy.MaintenancePolicy]
groups every threshold that decides which maintenance step is worth running.
Every destructive or costly behaviour is opt-in — by default nothing expires,
nothing is deleted, and the table is never reclustered:

| Field | Default | Meaning |
|---|---|---|
| `delete_threshold` | `0.1` | Passed to `rewrite_data_files` |
| `min_file_size_bytes` | `100_000_000` | Below this a data file counts as "small"; only decides whether the merge step runs (the merge itself goes up to the catalog's `target_file_size`) |
| `max_small_files` | `10` | `merge_files` runs when more small files than this exist |
| `flush_inlined` | `True` | Flush inlined rows when some exist |
| `max_overlap_ratio` | `0.5` | `recluster` runs when `overlap_ratio` exceeds this (and `recluster=True`) |
| `recluster` | `False` | Opt-in: allow the (full-rewrite) recluster step at all |
| `retention_days` | `None` | Snapshot retention for expire + cleanup; `None` never expires anything |
| `delete_orphaned` | `False` | Opt-in: run `delete_orphaned_files` |
| `dry_run` | `False` | Log what would run without changing anything |

[`DuckLakeMaintenance.maintain(policy)`][dt_ducklake_manager.maintenance.policy.DuckLakeMaintenance.maintain]
reads a fresh `storage_report()`, then considers each step in order — flush →
rewrite → merge → recluster → expire → cleanup → delete_orphaned — running it
only when its own indicator justifies it under the policy:

| Step | Runs when |
|---|---|
| flush | `flush_inlined` and unflushed inlined rows exist |
| rewrite | at least one delete file exists (DuckLake applies `delete_threshold` per file) |
| merge | `small_file_count > max_small_files` |
| recluster | `recluster=True` and `overlap_ratio > max_overlap_ratio` |
| expire, then cleanup | `retention_days is not None` |
| delete_orphaned | `delete_orphaned=True` |

A skipped step is logged explicitly with its reason (e.g. `"recluster
skipped — overlap 0.12 <= max_overlap_ratio 0.5"`) and flagged
`<step>_skipped` in the resulting `OperationReport.maintenance`. Under
`dry_run=True`, the modifying steps (flush, rewrite, merge, recluster) are
only logged as "would run"; expire/cleanup/delete_orphaned are called with
their own `dry_run=True`, which only lists candidates. Every step is
non-fatal: a failure is logged and added to `report.warnings`, and the
remaining steps still run.

There is no `full_maintenance` shortcut — planned maintenance is always an
explicit policy:

```python
maintenance.maintain(MaintenancePolicy(retention_days=30))
```

## Scenarios

### After a large update

The write itself already ran `compact()` (merge + rewrite) right after its
commit. Nothing more is required immediately; `storage_report()` tells you
whether it is worth going further:

```python
report = maintenance.storage_report()
print(report.summary())
```

### After a massive deletion

Same as above — `delete_rows(..., compact_after_update=True)` already merged
and rewrote. If the deletion also emptied every value of some columns,
`delete_rows` drops them automatically (`perform_cleanup`, default follows
`auto_cleanup`); check `report.columns_dropped`.

### After N updates (file-range overlap)

```python
report = maintenance.storage_report()
if report.overlap_ratio is not None and report.overlap_ratio > 0.5:
    maintenance.maintain(MaintenancePolicy(recluster=True))
```

### Freeing disk space (retention)

Space is only actually freed by `cleanup_files`, which only makes sense after
`expire_snapshots`. Both are planned-maintenance-only:

```python
maintenance.maintain(MaintenancePolicy(retention_days=30))
```

### After an interrupted transaction

Review candidates before deleting anything:

```python
candidates = maintenance.delete_orphaned_files(dry_run=True)
# inspect candidates, then:
maintenance.delete_orphaned_files(dry_run=False)
```

### Finding the run behind a given state

```python
recovery = DatabaseRecoveryManager(conn)
print(recovery.list_ducklake_snapshots())  # snapshot_id, author (run_id), commit_message, commit_extra_info, ...
```

See [Traceability of runs](schema.md#traceability-of-runs) for how `run_id` and
`commit_message` land on a snapshot in the first place.

### Time travel (reading or restoring a past state)

```python
# Reading a past state on the connection already open
at = DuckLakeConnector(catalog_path, data_path, snapshot_version=17).at_clause()
conn.execute(f"SELECT * FROM fact_table {at}")

# Restoring the result set to that state (a new, authored snapshot)
DatabaseRecoveryManager(conn).restore_snapshot(17, run_id="rollback-run-42")
```

With the file-based DuckDB backend, a snapshot connection
(`DuckLakeConnector(..., snapshot_version=17).connect()`) cannot coexist with an
already-open connection to the same catalog (the catalog file is locked at the
process level): on an open connection, use the `AT (VERSION => n)` clause built
by `DuckLakeConnector.at_clause`. `restore_snapshot` copies the rows of each
table as they were at the snapshot into temporary tables, then empties and
refills the live tables in one transaction; a table whose columns changed since
the snapshot is refused, and is read with the `AT` clause instead.

### Explicit broadcast of a partial-key column

`add_columns` never spreads a value carried by a partial key (e.g.
`(region, product)`) over a fact table keyed by a fuller key (e.g. `(date,
region, product)`) — see [Result-set granularity](architecture.md#1-result-set-granularity)
for why. To broadcast deliberately:

```python
keys = updater.get_key_combinations(["region", "product"])
df_partial = keys.to_polars().join(df_score, on=["region", "product"])
updater.add_columns(df_partial)
```

## Reading an `OperationReport`

Every write (`build_schema`, `update_database`, `add_columns`, `delete_rows`,
`delete_columns`) and every maintenance call (`recluster`, `maintain`) returns
or exposes an [`OperationReport`][dt_ducklake_manager.reporting.OperationReport],
built from DuckLake's own introspection functions (`ducklake_table_info`,
`ducklake_snapshots`, `ducklake_table_changes`), never estimated in Python.
Zeros are explicit rather than omitted in the underlying counters; the
one-line `summary()` only shows non-trivial deltas:

```python
report = updater.last_report
print(report.summary())
# update main [run-42]: +1240 rows, ~380 updated, +2 columns (score, rank),
# 3 -> 2 files (48.2 -> 31.7 MB), snapshot 17 -> 18, 4.1s
```

Key fields: `rows_inserted` / `rows_updated` / `rows_deleted` (from
`ducklake_table_changes`), `columns_added` / `columns_dropped`,
`snapshot_before` / `snapshot_after`, `files_before` / `files_after`,
`bytes_before` / `bytes_after`, `maintenance` (counters from the post-write
compaction or from `maintain`'s steps), and `warnings` (non-fatal issues
collected during the operation). `to_dict()` gives a JSON-serializable form
for logging pipelines.

## Reading a `StorageReport`

[`StorageReport`][dt_ducklake_manager.maintenance.policy.StorageReport],
returned by `storage_report()`, is the decision input behind `maintain`:
`file_count`, `total_bytes`, `delete_file_count`, `delete_ratio`,
`small_file_count`, `inlined_rows` / `has_inlined_data`, `snapshot_count` /
`oldest_snapshot_age_days`, and `overlap_ratio` / `cluster_column`. Every
field defaults to its "nothing to do" value when it cannot be measured (no
real DuckLake catalog attached, unknown table) rather than raising.

```python
report = maintenance.storage_report()
print(report.summary())
# storage main.fact_table: 3 files (2.4 MB), 0 delete files (0.0%),
# 3 small files, 0 inlined rows, 12 snapshots (oldest 4.2 days), overlap 0.67 on date
```
