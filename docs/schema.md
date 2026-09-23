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
| `label_for` | VARCHAR | yes | Set on a **label column**: name of the code column whose value labels it carries (see [Codes and value labels](#codes-and-value-labels)) |
| `unit` | VARCHAR | yes | Axis / tooltip suffix (`"€"`, `"%"`, `"MW"`) |
| `display_format` | VARCHAR | yes | d3-format string (`",.2f"`, `".0%"`) |
| `family` | VARCHAR | yes | Thematic grouping of variables in menus |
| `description` | VARCHAR | yes | Contextual help text |
| `default_aggregation` | VARCHAR | yes | One of `SUM`, `AVG`, `MIN`, `MAX`, `COUNT`, `MEDIAN`, `MODE`, validated on write |

`metadata` is **the contract between the database and the interface**:
everything the UI needs to drive itself (label, type, categorical status,
hierarchy, unit, format, family, default aggregation) lives here, never
deduced from the data at query time.

`label` and `label_for` do not overlap: `label` is the display label **of the
column itself** (a header, "NC8 code"); `label_for` names the column whose
**per-value** labels this column carries (code `"01012100"` → label "Chevaux
reproducteurs de race pure"), linking a code column to its label column — see
[Codes and value labels](#codes-and-value-labels).

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
value`, **except** for a code column that has a label column of its own (see
[Codes and value labels](#codes-and-value-labels)): there, `value` is the
code and `label` is its label.

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

Three invariants are validated at write time:

1. the parent column exists;
2. the graph of `parent_name` pointers is a forest (no cycle, at most one
   parent per column, checked by `validate_hierarchy_forest`);
3. every column of a hierarchy is categorical.

**Convention for irregular trees** (leaves at different depths): the missing
levels are `NULL`. The tree builder stops at the first `NULL` level, and the
API ignores `NULL` in the `DISTINCT`. The value of the level above is never
repeated to fill a hole — that would fabricate a node that does not exist in
the data.

`get_column_hierarchies(conn, schema)` is the reference implementation for API
layers that need the reconstructed root-to-leaf chains (one list per leaf
column) rather than the raw per-column `parent_name` pointers.

## Codes and value labels

Some columns carry a **business code** that must be restituted verbatim (a
tariff-nomenclature code, an INSEE code, an ISO country code) and that should
also display a human-readable **label**. The code and the label are **two
`fact_table` columns**; the link is declared in `metadata.label_for`, carried
by the label column, pointing at the code column. Example: an `nc6` → `nc8`
customs-nomenclature hierarchy, with `nc8` labelled in French and English:

| `name` | `parent_name` | `label_for` |
|---|---|---|
| `nc6` | `NULL` | `NULL` |
| `nc8` | `nc6` | `NULL` |
| `nc8_libelle_fr` | `NULL` | `nc8` |
| `nc8_libelle_en` | `NULL` | `nc8` |

The pointer sits on the label column, not the code column, for two reasons:
it is the same shape as `parent_name` (the child points at its target — same
validation, same cascade on deletion), and one code can have **several**
label columns (languages, short/long form) without any schema change. Each
level of a code hierarchy carries its own labels; the hierarchy itself
relates the codes.

Declare a code/label pair at build time:

```python
builder = DuckLakeTablesBuilder(
    df, primary_keys=["date", "nc8"],
    hierarchies={"nc8": "nc6"},
    value_labels={"nc8_libelle_fr": "nc8", "nc8_libelle_en": "nc8"},
    connection=connection,
)
builder.build_schema(
    column_metadata={"nc8_libelle_fr": {"label": "Libellé NC8 (FR)"}},
)
```

or per column through `column_metadata={"nc8_libelle_fr": {"label_for":
"nc8"}}` — both sources must agree where they overlap, exactly like
`hierarchies`. On an existing schema, add or correct a link with
`update_column_metadata("nc8_libelle_fr", label_for="nc8")`.

### Invariants and when they are checked

Four invariants are validated on write, raising `ValueError` (`ROLLBACK` on
failure):

1. the target code column exists in the fact table and differs from the
   label column;
2. no chaining: the target is not itself a label column;
3. the label column is `VARCHAR`, is not a primary key, and belongs to no
   hierarchy (neither `parent_name` set, nor parent of another column);
4. **functional dependency code → label**: for every non-`NULL` value of the
   code, the label column carries a **single** value across the whole table,
   `NULL` included (a code with some labelled and some unlabelled rows is
   invalid); a row whose code is `NULL` must have a `NULL` label.

Invariants 1–3 are structural (`validate_value_labels`) and are checked
whenever a `label_for` link is declared or changed: at `build_schema`, and on
`update_column_metadata(column, label_for=...)` against an existing schema.
Invariant 4 (`check_value_label_dependency`) is a data check and runs: at
build time, on the deduplicated DataFrame; on every `update_database`, after
the upsert, **inside the transaction**, restricted to the codes present in
the batch; on every `add_columns` call that writes a code or a label column;
when a link is declared on an existing base
(`update_column_metadata(col, label_for=...)`, checked against the whole
table); and after `update_value_labels`. The error message lists at most ten
faulty codes with their competing labels.

The two queries behind invariant 4:

```sql
SELECT code FROM fact_table WHERE code IS NOT NULL GROUP BY code
HAVING COUNT(DISTINCT libelle) > 1 OR (COUNT(libelle) > 0 AND COUNT(libelle) < COUNT(*));

SELECT COUNT(*) FROM fact_table WHERE code IS NULL AND libelle IS NOT NULL;
```

There is no constraint on `is_categorical`: a code with 10,000 modalities can
stay non-categorical (a search field, whose autocomplete searches both the
code and the label). The label column's own categorical status is moot for
the interface, which hides label columns from variable lists.

### Changing a label

A nomenclature revision that renames a code violates the dependency if it
arrives through an upsert: the rows already written under the old label keep
it, so some rows of that code would carry the old label and the new batch's
rows the new one — exactly what invariant 4 refuses. `update_database`
raises and points to
`DatabaseUpdater.update_value_labels(label_column, labels)`, where `labels`
is a `(code, label)` DataFrame. The operation is a single `UPDATE … FROM`
over every row of the codes given (a copy-on-write rewrite of those rows,
followed by compaction), inside one transaction, reported as an
`OperationReport` (`operation = 'update_value_labels'`). The displayed label
is therefore always the **current** one; earlier labels stay readable through
DuckLake time travel. A code that changes *meaning* from one vintage to the
next is not a label change: the producer folds the vintage into the code, or
keeps an ordinary column with no `label_for`.

### Reading

`SELECT DISTINCT code, label` gives the correspondence table; an aggregate
grouped by code recovers the label with `ANY_VALUE(label)` (valid precisely
because of the functional dependency); filters and joins between result sets
are made on the code, not the label — see [Architecture &
trade-offs](architecture.md#1-result-set-granularity).
[`get_value_label_columns(conn, schema)`](api/utils/get_value_label_columns.md)
(code → sorted list of label columns) is the reference implementation for
API layers, symmetric to `get_column_hierarchies`.

### Storage

Cost is low: the label is dictionary-encoded and perfectly correlated with
the code.

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

## Design choices not retained

**Value hierarchy in a single column** (adjacency list + materialized path,
`dim_<col>(value, label, parent_value, path, depth)`). This would only beat
flattening into columns if (a) the depth is irregular and unknown to the
producer, or (b) the taxonomy evolves independently of the facts and is
maintained as an external reference. In a model-results pipeline the
producer controls the DataFrame and knows its levels: case (a) is handled by
the `NULL` convention above, and case (b) does not arise. The cost — a second
table to keep consistent with the fact table (orphaned values, atomic
replacement, cycle validation) — is exactly the complexity that dropping
`dim_*` tables removes. If a project ever needs it, the extension is local: a
table declared explicitly at build time (`value_hierarchies={col: df}`),
replaced wholesale on every update, with no impact on the fact table or on
`metadata`.

**Label table `dim_<col>(value, label)` for a business code ≠ its label, or
a reference catalog shared across projects.** Preferable only if the
reference data is maintained independently of the models **and** its labels
must change without touching the facts. The cost is what dropping `dim_*`
tables removes: a join on every query (and, for a separate catalog,
multi-catalog reads, see [Architecture & trade-offs](architecture.md#multi-catalog-reads)),
versioning by vintage (a tariff nomenclature changes every year), orphaned
codes, consistency between two tables. A shared reference table, if one
exists, is used **upstream**: the producer joins it onto its DataFrame before
writing.

**Naming convention (`<col>_label`) with no metadata.** An implicit contract
the interface would have to guess; `label_for` makes it explicit for the
price of one nullable column.

**Pointer on the code column (`label_column`).** Would limit each code to a
single label.

**Broadcasting a column over a partial-key subset.** A DataFrame carrying
`(region, product, score)` against a fact table keyed `(date, region,
product)` is **a different result set** (a different key): it belongs in its
own schema, which the API joins or displays separately. Duplicating `score`
onto every `date` denormalizes the value (rows inserted later don't inherit
it, so it drifts from its source), hides a full rewrite of the fact table
behind what looks like a column add, and makes a cartesian product implicit.
A user who genuinely wants the broadcast does it explicitly in their own
DataFrame — joining it onto the existing key combinations, obtained through
`updater.get_key_combinations(columns)` — then calls `add_columns`. See
[`add_columns`](api/operations/DatabaseUpdater.md) for that recipe.
