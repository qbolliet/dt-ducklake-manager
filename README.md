# dt-ducklake-manager

Utilities to build and manage a DuckLake database from a tabular dataset, designed for dashboard and ML prediction pipelines.

> Full documentation is available [here](https://qbolliet.github.io/dt-ducklake-manager/)

## Objectives

This package provides a complete lifecycle for a DuckLake database:

- **Build** a structured schema from any tabular dataset
- **Update** the database with new or modified observations (upsert)
- **Delete** rows based on filter conditions
- **Audit & validate** database integrity at configurable levels
- **Maintain** physical storage (file compaction, snapshot expiry)

The schema is built around a **fact table**, a **metadata table**, and **dimension tables** for low-cardinality categorical columns.

Input dataframes are handled via [narwhals](https://narwhals-dev.github.io/narwhals/), making the package compatible with pandas, polars, and any other narwhals-supported backend.

## Installation

```bash
git clone https://github.com/qbolliet/dt-ducklake-manager.git
uv sync
```

### Documentation

```bash
uv sync --group docs
mkdocs serve --port 5000
```

## Usage

```python
import pandas as pd
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from dt_ducklake_manager.operations import DatabaseUpdater, DatabaseDeleter
from dt_ducklake_manager.maintenance import DatabaseAuditor, DuckLakeMaintenance, ValidationLevel

# 1. Build the schema from an initial dataset
df = pd.DataFrame({
    "id": [1, 2, 3],
    "city": ["Paris", "Berlin", "Madrid"],
    "score": [0.9, 0.7, 0.5],
})
connector = DuckLakeConnector(catalog_path="outputs/database.db")
builder = DuckLakeTablesBuilder(connector=connector, df=df, categorical_threshold=200)
builder.build_schema()

# 2. Update the database with new observations (upsert)
df_new = pd.DataFrame({"id": [2, 4], "city": ["Lyon", "Rome"], "score": [0.8, 0.6]})
updater = DatabaseUpdater(connector=connector)
updater.update_database(df=df_new)

# 3. Delete rows matching a condition
deleter = DatabaseDeleter(connector=connector)
deleter.delete_rows(conditions=[("score", "<", 0.6)])

# 4. Audit database integrity
auditor = DatabaseAuditor(connector=connector)
report = auditor.validate_database(level=ValidationLevel.STANDARD)
print(report)

# 5. Run full maintenance (compaction, snapshot expiry)
maintenance = DuckLakeMaintenance(connector=connector)
maintenance.full_maintenance()
```

More detailed examples and parametrization walkthroughs are available in the `notebooks/` folder.

## Catalog backend (concurrent read/write)

DuckLake keeps its **catalog metadata** in a backing database while the **row data**
stays in Parquet files under `data_path`. Two backends are supported through
`DuckLakeConnector`:

- **DuckDB file** (default) — a local `.ducklake` file. Simple, zero-setup, but the
  file is locked at the process level: it **cannot be read by one program while
  another writes it**. Best for local development and one-shot builds.
- **PostgreSQL** — a multi-client server (MVCC). A read-only consumer (e.g. a
  GraphQL API) can query the catalog **while** an update job writes to it, without
  lock conflicts. Best for production deployments with parallel reads and updates.

Pick the backend at the call site; there is no config file to change:

```python
from dt_ducklake_manager.connection import DuckLakeConnector

if PRODUCTION:
    # Update job (read-write) — credentials passed once to build a DuckDB secret
    conn = DuckLakeConnector.from_postgres(
        data_path="s3://my-bucket/data/",
        dbname="ducklake", host="db.internal", user="app", password="***",
    ).connect()
    # The GraphQL API repository connects read-only to the SAME catalog + data_path:
    #   DuckLakeConnector.from_postgres(
    #       data_path="s3://my-bucket/data/", dbname="ducklake",
    #       host="db.internal", user="api", password="***", read_only=True,
    #   ).connect()
else:
    # Local development — single-process DuckDB file catalog (unchanged behaviour)
    conn = DuckLakeConnector("outputs/catalog.ducklake", "outputs/data/").connect()
```

The returned connection is used exactly the same way for both backends (it is passed
to `DuckLakeTablesBuilder`, `DatabaseUpdater`, etc.).

Notes:

- **No extra Python dependency**: PostgreSQL connectivity is provided by DuckDB's
  `postgres` extension, installed at runtime like the `ducklake` extension itself
  (no `psycopg`). PostgreSQL **≥ 12** is required server-side.
- **Credentials** are supplied through a DuckDB secret rather than embedded in the
  connection string. With inline credentials, `from_postgres` creates a session
  (non-persistent) secret; omitted fields fall back to the libpq environment
  variables (`PGHOST`, `PGUSER`, `PGPASSWORD`, ...). In production, create the secret
  out of band and reference it by name with `meta_secret="..."` so no credential
  ever flows through Python.

## Organization

- `docs/` — package documentation and database schema description
- `logs/` — logging files produced by the builders
- `notebooks/` — illustrative notebooks covering various use cases
- `outputs/` — program outputs

## License

MIT
