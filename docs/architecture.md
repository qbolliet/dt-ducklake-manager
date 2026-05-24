# Architecture & design trade-offs

This page documents the structural decisions behind the storage model, the
alternatives that were considered, and the trade-offs of each. It is intended as
an architectural documentation of the current implementation and a potential reference for future evolutions.

The design space is organized around **two independent axes**, often conflated:

- **Logical axis — dimension sharing**: are the dimension tables *conformed*
  (a single shared source of truth) or *redundant* (recreated per database)?
- **Physical axis — catalog layout**: how many DuckLake catalogs and schemas
  hold the result sets?

These axes are independent: one can have several fact tables with redundant
dimensions, or a single multi-schema catalog whose dimensions are still
redundant. Each axis is treated as its own section below, and each candidate is
a subsection.

> **Comparison criterion.** The options are compared on their **intrinsic
> complexity** and operational properties (encoding coherence, snapshot
> granularity, locking).

For context, the storage model (see the [package description](index.md)) is made
of three layers per result set:

- a `fact_table` holding the observations ;
- `dim_<col>` tables mapping each modality of a low-cardinality categorical
  variable to an integer `id`, where ids are assigned by order of appearance ;
- a `metadata` table describing each column (label, type, categorical status,
  primary key).

---

## 1. Fact table granularity and dimension sharing

Each result set produced by a model — for instance predictions on one side and
Shapley values on the other — may share categorical variables (`country`,
`city`, …) while differing in granularity and in the value columns it carries.
The question is whether each result set should be a self-contained database with
its own dimension tables, or whether several fact tables should coexist and
share their dimension tables.

### Option A — One database per result set (redundant dimensions)

**Status: ✅ Retained in the current implementation.**

Each result set is a self-contained DuckLake catalog with its own `fact_table`,
`metadata`, and `dim_*` tables. A categorical variable common to several result
sets is stored as an independent dimension table in each database.

**Advantages**

- **Full isolation**: building, updating, or deleting one result set cannot
  affect another.
- **Local categorical decisions**: when a column's modality count crosses the
  threshold, the conversion between categorical and non-categorical status is
  self-contained within a single fact table. The divergence problem (a column
  whose cardinality grows in one result set but not another) cannot arise,
  because no structure is shared.
- **Independent lifecycle and snapshots**: DuckLake time-travel is scoped to a
  catalog, so each result set keeps its own snapshot history — well suited to
  auditing a specific model run.
- **Independent concurrency**: separate catalogs can be written in parallel,
  which matters for the file-based DuckDB backend that locks the catalog at the
  process level.
- **Simple model**: a single fact table per catalog, and `metadata` keyed by
  column name alone.

**Disadvantages (complexity)**

- **Dimension redundancy**: a shared categorical variable is stored once per
  database. The storage cost is negligible, since dimension tables are bounded
  by the categorical threshold.
- **Dimension-encoding drift across databases**: integer ids are assigned
  independently in each database (by order of appearance of the modalities), so
  the same label can map to different ids in two databases. Cross-database joins
  must therefore resolve labels through each database's dimension table rather
  than join on the raw integer id. Joining directly on ids is only correct when
  identical encoding is guaranteed.
- **No single source of truth for labels**: a label correction has to be applied
  in every database that carries the variable.

### Option B — Multiple fact tables sharing conformed dimensions

A single database holds several fact tables (one per result set) at different
granularities, all referencing a shared set of dimension tables (*conformed
dimensions* in the Kimball sense).

**Advantages**

- **Coherent encoding by construction**: a shared dimension has a single
  id↔label mapping, so cross-fact joins on dimension ids are correct without
  resolving labels.
- **Single source of truth for labels** across all result sets.
- Marginal storage gain (dimensions are small by construction).

**Disadvantages (complexity)**

- **Metadata must be keyed by `(fact table, column)`** instead of by column
  alone: the same logical column can be categorical in one fact table and not in
  another, depending on its cardinality in each.
- **The categorical status of a shared variable can diverge** between fact tables
  (e.g. 30 modalities in predictions, 60 in Shapley). Two reconciliation
  policies exist, each adding complexity:
    - *Independent encoding per fact table*: the column may be id-encoded in one
      fact and stored raw in another. The dimension is then shared only as a
      label space, not as an encoding, and the join-on-id benefit is lost for
      that column.
    - *Strict conformed dimension*: the categorical status is decided on the
      **union** of modalities across all fact tables; a column may then exceed
      the threshold on the union while remaining small in each individual table.
- **Shared dimensions require reference counting**: a dimension may only be
  dropped (on conversion to non-categorical) once no fact table still references
  it, and orphan-entry cleanup must be computed over the union of ids referenced
  by all fact tables rather than a single one.
- **Cross-table coupling of conversions**: a categorical ↔ non-categorical
  conversion triggered by one fact table must not rewrite or drop structures
  relied upon by another.

### Summary

| Option | Key advantage | Main complexity cost | Retained |
| --- | --- | --- | --- |
| A — One database per result set | Full isolation; local categorical decisions; per-run snapshots | Redundant dimensions; id-drift across databases on cross-database joins | ✅ |
| B — Shared conformed dimensions | Coherent encoding; single source of truth for labels | `(table, column)` metadata; divergent categorical status; reference counting of shared dimensions | |

---

## 2. Schema organization within a catalog

A DuckLake catalog can hold several schemas. Today each catalog uses a single
schema, `main` (see [`DuckLakeConnector`](api/connection/DuckLakeConnector.md)).
The question is whether several result sets should be consolidated as separate
schemas within one catalog.

This axis is **orthogonal to dimension sharing**: separate schemas do not share
dimension tables (each schema carries its own `dim_*`), so multiple schemas
address catalog proliferation, not dimension redundancy.

### Option A — One schema per catalog (`main`)

**Status: ✅ Retained in the current implementation.**

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

Result sets are stored as separate schemas in one catalog (e.g.
`predictions.fact_table`, `shapley.fact_table`), sharing one catalog backend.

**Advantages**

- **Single catalog** to attach and administer; cross-schema joins are
  first-class on a single connection.
- **Coherent transactions** across schemas.
- Pairs naturally with a server-based catalog backend (PostgreSQL), which
  supports concurrent readers and writers.

**Disadvantages (complexity)**

- **Snapshot granularity is the whole catalog**, not the schema: time-travel to a
  specific result set becomes ambiguous, since updating one schema advances the
  catalog version for all of them.
- **With the file-based backend, a single catalog file serializes writes** across
  all schemas (process-level lock).
- **Does not reduce dimension redundancy**: dimension tables are not shared
  across schemas, so the id-drift consideration of section 1 still applies.
- **Requires schema routing** in the connection and management layer.

### Summary

| Option | Key advantage | Main complexity cost | Retained |
| --- | --- | --- | --- |
| A — One schema per catalog | Per-result-set snapshots and locking; simple routing | Catalog proliferation; multi-catalog attach for cross joins | ✅ |
| B — Multiple schemas per catalog | Single catalog; first-class cross-schema joins; fits a server backend | Catalog-level snapshots; serialized writes on the file backend; schema routing | |

---

## Retained decisions

Both axes currently favour **isolation over consolidation**: one self-contained
catalog with a single `main` schema per result set (Option A in both sections).
This keeps each result set independent in its lifecycle, snapshots, concurrency,
and categorical-status decisions, at the cost of redundant dimension tables and
of resolving labels (rather than raw ids) when joining across databases.
