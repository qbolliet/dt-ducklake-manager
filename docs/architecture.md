# Architecture & design trade-offs

This page documents the structural decisions behind the storage model. It is intended as
an architectural documentation of the current implementation.

The design space is organized around **two independent axes**, often conflated:

- **Logical axis — result-set granularity**: what determines whether new data
  becomes new columns, new rows, or a new schema altogether?
- **Physical axis — catalog layout**: how many DuckLake catalogs and schemas
  hold the result sets?

These axes are independent: a project can have one catalog per result set, or
several result sets consolidated as schemas of a single catalog, regardless of
how each individual result set's granularity was decided. Each axis is treated
as its own section below.

For context, the storage model (see the [package description](index.md)) is
made of exactly three tables per result set (see [Schema and data
model](schema.md) for the full description):

- a `fact_table` holding the observations, categorical columns carrying their
  original labels directly ;
- a `metadata` table describing each column (label, type, categorical status,
  primary key, hierarchy, UI fields) ;
- a `dataset_metadata` table describing the result set itself.

---

## 1. Result-set granularity

A *result set* is defined by its **key** — the combination of columns that
identifies one observation (e.g. `(date, region, product)`). The question this
section answers is: given a new DataFrame, when does it become new rows of an
existing `fact_table`, new columns of it, or an altogether different schema?

**A DataFrame sharing the fact table's exact key is the same result set.**
New observations for existing (or new) key combinations are an
`update_database` (upsert); new value columns keyed by the same primary keys
are an `add_columns` (outer merge on the keys — see [Schema and data
model](schema.md#design-choices-not-retained) for why a value carried by a
*partial* key is never silently broadcast over a fuller one).

**A DataFrame carrying a different key is a different result set**, and goes
into its **own schema** — never into extra columns of an unrelated fact table.
Predictions keyed by `(date, region, product)` and Shapley values keyed by
`(date, region, product, feature)` are two schemas, `predictions` and
`shapley`, even though they come from the same model run and share several
column *names*. Forcing them into one fact table would mean padding every
prediction row with `NULL` Shapley columns (or the reverse), and would make
the primary key of the combined table ambiguous.

**Cross-result-set analysis joins on labels, not on ids.** A
query correlating `predictions` and `shapley` joins directly on their shared
label columns (`region`, `product`, …), which is exactly what a categorical
column already stores. This is simpler than the dimension-table alternative
that was considered and dropped (see [Schema and data
model](schema.md#design-choices-not-retained)): there is no id↔label
resolution step, and no risk of two databases assigning different ids to the
same label, because no id is ever assigned.

### Multi-catalog reads

A single connection can `ATTACH` several DuckLake catalogs and join across
them freely for reads (measured). Writes are more constrained: **a single
DuckDB transaction can only write to one attached database** (measured:
`a single transaction can only write to a single attached database`).
Consequently:

- cross-catalog access is for **reads only**;
- a secondary catalog attached purely for reading should use
  `DuckLakeConnector(..., read_only=True)` and, when the primary catalog's
  schema must stay active, `attach(activate_schema=False)` so attaching the
  second catalog does not steal the connection's current schema;
- no operation in this package ever writes across two catalogs in the same
  transaction, and application code combining several catalogs should keep the
  same discipline.

---

## 2. Schema organization within a catalog

A DuckLake catalog can hold several schemas. Both layouts are supported (see
[`DuckLakeConnector`](api/connection/DuckLakeConnector.md)): every schema-aware
class accepts a `schema` argument (default `main`) and qualifies its table
references accordingly, so several result sets can live as separate schemas of one
catalog. The question is whether to keep one schema per catalog or to consolidate
several result sets as separate schemas within one catalog.

This axis is **orthogonal to [section 1](#1-result-set-granularity)**: whether
result sets live in separate catalogs or as separate schemas of one catalog,
each keeps its own `fact_table` / `metadata` / `dataset_metadata` triplet —
there is no dimension table in either layout to share or duplicate.

### Option A — One schema per catalog (`main`)

**Status: ✅ Default layout.**

Each catalog contains exactly one schema, holding one fact table and its
companion tables.

**Advantages**

- **Snapshot isolation**: DuckLake snapshots are per catalog, so each result set
  has an independent time-travel history — updating one result set does not
  advance the snapshot version seen by readers of another.
- **Concurrency isolation**: with the file-based DuckDB backend, the catalog file
  is locked at the process level; separate catalogs let result sets be written
  in parallel.
- **Simple routing**: a connection targets a single schema (`USE db.main`), with
  no schema-selection logic.

**Disadvantages (complexity)**

- **Catalog proliferation**: each result set is a separate catalog to attach and
  administer.
- **Cross-result-set joins** require attaching several catalogs within the same
  connection.

### Option B — Multiple schemas within a single catalog

**Status: ✅ Supported (opt-in via the `schema` argument).**

Result sets are stored as separate schemas in one catalog (e.g.
`predictions.fact_table`, `shapley.fact_table`), sharing one catalog backend. A
single shared connection drives all schemas: each builder/manager qualifies its
tables by its `schema`, and `DuckLakeConnector` creates the schema on first use.

**Advantages**

- **Single catalog** to attach and administer; cross-schema joins are
  first-class on a single connection, and — since fact tables carry labels
  directly — need no id-resolution step.
- **Coherent transactions** across schemas.
- Pairs naturally with a server-based catalog backend (PostgreSQL), which
  supports concurrent readers and writers.

**Disadvantages (complexity)**

- **Snapshot granularity is the whole catalog**, not the schema: time-travel to a
  specific result set becomes ambiguous, since updating one schema advances the
  catalog version for all of them.
- **With the file-based backend, a single catalog file serializes writes** across
  all schemas (process-level lock).
- **Requires schema routing** in the connection and management layer.

### Summary

| Option | Key advantage | Main complexity cost | Status |
| --- | --- | --- | --- |
| A — One schema per catalog | Per-result-set snapshots and locking; simple routing | Catalog proliferation; multi-catalog attach for cross joins | ✅ default |
| B — Multiple schemas per catalog | Single catalog; first-class cross-schema joins; fits a server backend | Catalog-level snapshots; serialized writes on the file backend; schema routing | ✅ supported |

---

## 3. Transactions and recovery

**Every public write operation is one DuckDB transaction.**
`update_database`, `add_columns`, `delete_rows` and `delete_columns` each open a
single `BEGIN` / `COMMIT` block (`BaseSchemaManager._transaction`) and `ROLLBACK`
on any exception, so a failure mid-operation leaves the schema exactly as it was
— fact table, `metadata` rows and added columns alike, since DuckDB rolls DDL
back along with DML. There is no application-level transaction machinery: no
registered operations, no savepoints, no compensating `rollback_func`.

`use_transaction=False` keeps the same ordered steps but runs them in autocommit
mode: faster, and a failure then leaves whatever was already written in place.

**Post-write maintenance runs after the commit, never inside it.**
`merge_adjacent_files` and `rewrite_data_files` are called once the transaction
has closed: they are optimizations, and a compaction failure must never undo a
successful write.

**Time travel is the recovery mechanism.** No backup file is written, because
DuckLake already persists the full snapshot history. To recover from a bad write
that was itself committed:

```python
recovery = DatabaseRecoveryManager(conn)
print(recovery.list_ducklake_snapshots())          # pick a snapshot_id
old = DuckLakeConnector(catalog, data, snapshot_version=17).connect()
```

then read the tables from `old` and reinsert them into the current catalog.
`RecoveryStrategy.USE_SNAPSHOT_HISTORY` returns that procedure step by step. The
other strategies (`REPAIR_SCHEMA`, `CLEAN_ORPHANED_DATA`, `VALIDATE_AND_FIX`)
repair structural inconsistencies in place and never touch the history.
Snapshot expiry (`expire_snapshots`) is planned maintenance with an explicit
retention — never a side effect of a write, because it is what destroys the
ability to recover.

---

## Retained decisions

On **result-set granularity** (section 1), a result set's boundary is its key:
a DataFrame sharing the fact table's key is the same result set (new rows via
`update_database`, new columns via `add_columns`); a DataFrame carrying a
different key is a different result set and goes into its own schema, never
into extra columns of an unrelated fact table.
Multiple catalogs may be attached and read together, but a write is never
allowed to span two of them.

On **schema organization** (section 2), both layouts are now available. `main` per
catalog remains the **default**, preserving per-result-set snapshots and
file-backend locking. Consolidating several result sets as separate schemas of a
single catalog (Option B) is opt-in through the `schema` argument and pairs
naturally with the PostgreSQL backend, which supports concurrent readers and
writers; the trade-off is that DuckLake snapshots become catalog-wide rather than
per result set.
