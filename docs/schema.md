# Schema and data model

Each result set produced by a model lives in one DuckLake **schema**, made of
exactly three tables. A catalog can hold several such schemas — see
[Architecture & trade-offs](architecture.md) for how they relate to each other
and to the catalog. This page describes what is inside a single schema and how
its pieces are meant to be read.

![Scheme for table storage](assets/schema_bdd.png)

## The three tables

### `fact_table`

The observations, one column per variable, at their natural SQL width (see
[Types](#types) below). There is **no dimension table and no synthetic code**
anywhere in the schema: a categorical column stores its label directly
(`"Île-de-France"`, not an integer id). Parquet dictionary-encoding already
absorbs the storage cost of the repeated labels, so nothing is gained by
indirecting through an id.

The table is:

- deduplicated on the primary keys at build time;
- written **sorted** by `dataset_metadata.cluster_by` (see [cluster_by](#cluster_by-and-physical-sort-order));
- optionally Hive-partitioned on very-low-cardinality columns that are
  systematically filtered on (year, country) — never to the point of
  producing files under ~100 MB.

### `metadata` — one row per `fact_table` column

| Column | Type | Nullable | Role |
|---|---|---|---|
| `name` | VARCHAR | no | Technical column name |
| `label` | VARCHAR | no | Display label (defaults to `name`) |
| `sql_type` | VARCHAR | no | DuckDB SQL type (`BIGINT`, `DOUBLE`, `VARCHAR`, …) — drives DDL and the front end's type → chart mapping |
| `is_primary_key` | BOOLEAN | no | Part of the logical key (deduplication, upsert) |
| `is_categorical` | BOOLEAN | no | **Pure UI metadata**: the column is filtered through a select menu and can be a `groupBy`. Never drives a storage decision |
| `parent_name` | VARCHAR | yes | Parent column in a column hierarchy (see below) |
| `unit` | VARCHAR | yes | Axis / tooltip suffix (`"€"`, `"%"`, `"MW"`) |
| `display_format` | VARCHAR | yes | d3-format string (`",.2f"`, `".0%"`) |
| `family` | VARCHAR | yes | Thematic grouping of variables in menus |
| `description` | VARCHAR | yes | Contextual help text |
| `default_aggregation` | VARCHAR | yes | One of `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `MEDIAN`, `MODE`, validated on write |

`metadata` is **the contract between the database and the interface**:
everything the UI needs to drive itself (label, type, categorical status,
hierarchy, unit, format, family, default aggregation) lives here, never
deduced from the data at query time.

There is no `python_type` column — it would be redundant with `sql_type`. Type
conflicts arising from a later batch (e.g. an update carrying `Int32` where
`BIGINT` is stored) are resolved on the SQL type ranking `BOOLEAN < TINYINT <
SMALLINT < INTEGER < BIGINT < FLOAT < DOUBLE < VARCHAR`, keeping the greater
width: a stored `BIGINT` is never narrowed by a batch of `Int32`
(`resolve_sql_type_conflict`).

### `dataset_metadata` — one row per schema

| Column | Type | Role |
|---|---|---|
| `label` | VARCHAR | Title of the result set |
| `description` | VARCHAR | Subtitle / description |
| `source` | VARCHAR | Provenance (model, pipeline) |
| `updated_at` | TIMESTAMP | Last successful write (build, update, column add/drop) |
| `schema_version` | INTEGER | Currently always `1`, reserved for a future migration |
| `cluster_by` | VARCHAR | JSON list of the physical sort columns (see below) |

`updated_at` and `schema_version` are always populated; the remaining fields
come from optional builder arguments (`dataset_label`, `dataset_description`,
`dataset_source`).

## Categorical status

`is_categorical` picks the filter widget: a select menu for a categorical
column, a search field otherwise. The interface reads the modalities of a
categorical column with a plain `SELECT DISTINCT` on the fact table (a cheap
query on a dictionary-encoded column), with search and a limit. `label =
value` always: there is no separate label to resolve.

The flag is **inferred exactly once**, when the column is created:

- at `build_schema` time, from the number of non-null distinct values of a
  textual column against `categorical_threshold` — overridable per column with
  `categorical_overrides`;
- for a column added later, through `allow_new_columns=True` on
  `update_database` or through `add_columns`, using the updater's own
  `categorical_threshold`.

It is **never recomputed** by a later write: an update or a deletion that
changes the number of modalities does not touch it. This is deliberate — the
boolean only chooses a UI widget, and a status that flips with every batch
would make the interface unstable for no benefit. To correct it explicitly
(for instance to switch a column's filter from a search box to a select menu),
use:

```python
manager.update_column_metadata("region", is_categorical=True)
```

A column that belongs to a hierarchy is always categorical:
`is_categorical=False` is refused for it, and declaring a hierarchy link
force-sets `is_categorical=True` on both ends, with a warning, if either was
not already categorical.

The UI fields (`label`, `parent_name`, `unit`, `display_format`, `family`,
`description`, `default_aggregation`) and `is_categorical` belong to the
metadata producer: a data update never overwrites them. They are corrected on
an existing column with `update_column_metadata(column, **fields)`.

## Column hierarchies and the `NULL` convention

A hierarchy (a group-options menu, a selection tree) is **a chain of `fact_table`
columns**, declared through `metadata.parent_name`: the parent of `commune` is
`departement`, the parent of `departement` is `region`. There is no depth
limit. The hierarchy of *values* is already in the fact table: `SELECT
DISTINCT region, departement, commune` gives the full tree, no auxiliary table
required.

Declare a hierarchy at build time:

```python
builder = DuckLakeTablesBuilder(
    df, primary_keys=["date", "commune"],
    hierarchies={"commune": "departement", "departement": "region"},
    connection=connection,
)
```

or per column through `column_metadata={"commune": {"parent_name":
"departement"}}` — both sources must agree where they overlap. On an existing
schema, add or correct a link with `update_column_metadata("commune",
parent_name="departement")`.

Invariants validated at write time: the parent column exists; the graph of
`parent_name` pointers is a forest (no cycle, at most one parent per column,
checked by `validate_hierarchy_forest`); every column of a hierarchy is
categorical.

**Convention for irregular trees** (leaves at different depths): the missing
levels are `NULL`. The tree builder stops at the first `NULL` level, and the
API ignores `NULL` in the `DISTINCT`. The value of the level above is never
repeated to fill a hole — that would fabricate a node that does not exist in
the data.

`get_column_hierarchies(conn, schema)` is the reference implementation for API
layers that need the reconstructed root-to-leaf chains (one list per leaf
column) rather than the raw per-column `parent_name` pointers.

## Types

DataFrame (narwhals) → SQL, widths preserved:

| DataFrame (narwhals) | SQL |
|---|---|
| `Boolean` | `BOOLEAN` |
| `Int8` / `Int16` / `Int32` / `Int64` | `TINYINT` / `SMALLINT` / `INTEGER` / `BIGINT` |
| `UInt8` / `UInt16` / `UInt32` / `UInt64` | `UTINYINT` / `USMALLINT` / `UINTEGER` / `UBIGINT` |
| `Float32` / `Float64` | `FLOAT` / `DOUBLE` |
| `String` / `Categorical` / `Enum` | `VARCHAR` |
| `Date` / `Datetime` | `DATE` / `TIMESTAMP` |

The package never silently narrows a width: it is up to the producer to supply
a `Float32` (or a narrower integer) when 32 bits are judged sufficient for
values destined for charts.

## `cluster_by` and physical sort order

DuckLake has no index: pruning relies entirely on per-file column statistics
(`min`/`max`, exposed by `ducklake_file_column_stats`) and Parquet row-group
statistics, both of which only help when the data is physically grouped.
`cluster_by` (`dataset_metadata.cluster_by`, a JSON list) names the columns
the fact table is sorted by at write time — default: the primary keys, in
their declared order (most selective first). Build and update batches are
written `ORDER BY cluster_by`.

Successive updates degrade this ordering: each batch is sorted *within
itself* but not merged into the table's global order, so file ranges
progressively overlap. `DuckLakeMaintenance.recluster()` restores it by
rewriting the whole table in sort order — see
[Storage lifecycle and maintenance](maintenance.md) for when and how to run
it, and for the `overlap_ratio` indicator that tells you it's needed.

## Traceability of runs

Every write operation (`build_schema`, `update_database`, `add_columns`,
`delete_rows`, `delete_columns`, `DuckLakeMaintenance.maintain`) accepts
`run_id: str | None` and `commit_message: str | None`, recorded on the
resulting DuckLake snapshot via `ducklake_set_commit_message(catalog, author
:= run_id, message, extra_info := <JSON>)`, **inside** the write's
transaction. `extra_info` carries at least `operation` and `schema`, plus any
free-form fields passed through `commit_info` (e.g. `{"model_version":
"1.3"}`).

No technical column is ever added to the fact table to answer "which run
produced this state?" — the question is answered by reading the snapshots:

```python
conn.execute("SELECT snapshot_id, author, commit_message, commit_extra_info FROM ducklake_snapshots('db')").fetchall()
# or, through the package:
DatabaseRecoveryManager(conn).list_ducklake_snapshots()
```

On a connection with no real DuckLake catalog attached (in-memory test
connections), `run_id`/`commit_message` are silently skipped with a DEBUG log
line — the operation still runs and still returns a full
[`OperationReport`](api/reporting/OperationReport.md).
