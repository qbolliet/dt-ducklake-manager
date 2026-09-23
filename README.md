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

The schema is built around exactly three tables per result set, with **no
dimension table anywhere**: a **fact table** holding the observations, a **metadata table** describing every column — one row per column, the
contract between the database and the interface (label, SQL type,
primary-key and categorical flags, column hierarchy, the code → label link
via `label_for`, UI fields), and a
**dataset metadata table** describing the result set itself (title,
description, source, last update, schema version, physical sort key). See
the [schema](https://qbolliet.github.io/dt-ducklake-manager/schema/) and
[maintenance](https://qbolliet.github.io/dt-ducklake-manager/maintenance/)
pages of the documentation site for the full description.

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
from dt_ducklake_manager.maintenance import (
    DatabaseAuditor,
    DuckLakeMaintenance,
    MaintenancePolicy,
    ValidationLevel,
)

# 0. Open a connection attached to the DuckLake catalog
connection = DuckLakeConnector(
    catalog_path="outputs/catalog.ducklake",
    data_path="outputs/data/",
).connect()

# 1. Build the schema from an initial dataset, with a column hierarchy
# (commune -> region) and UI metadata
df = pd.DataFrame({
    "date": ["2026-01-01", "2026-01-01", "2026-01-02"],
    "region": ["Île-de-France", "Bretagne", "Île-de-France"],
    "commune": ["Paris", "Rennes", "Boulogne"],
    "score": [0.9, 0.7, 0.5],
})
builder = DuckLakeTablesBuilder(
    df,
    categorical_threshold=200,
    primary_keys=["date", "region", "commune"],
    hierarchies={"commune": "region"},
    connection=connection,
    dataset_label="City scores",
)
builder.build_schema(
    column_metadata={"score": {"unit": "%", "default_aggregation": "AVG"}},
    cluster_by=["date", "region"],
    run_id="build-2026-01-01",
)

# 2. Update the database with new observations (upsert), adding a new
# column on the fly
df_new = pd.DataFrame({
    "date": ["2026-01-02"], "region": ["Bretagne"], "commune": ["Rennes"],
    "score": [0.8], "rank": [1],
})
updater = DatabaseUpdater(connection=connection)
updater.update_database(
    update_df=df_new,
    allow_new_columns=True,
    column_metadata={"rank": {"label": "Rank", "default_aggregation": "MIN"}},
    run_id="update-2026-01-02",
)

# 3. Delete rows matching a condition
deleter = DatabaseDeleter(connection=connection)
deleter.delete_rows(filters=[("score", "<", 0.6)])

# 4. Audit database integrity
auditor = DatabaseAuditor(connection=connection)
report = auditor.validate_database(ValidationLevel.STANDARD)
print(report.recommendations)

# 5. Run maintenance driven by measured storage indicators (never
# destructive unless explicitly opted into)
maintenance = DuckLakeMaintenance(connection)
maintenance.maintain(MaintenancePolicy(retention_days=30))
```

A code column can also carry its own label column(s) (e.g. an NC8 tariff code
labelled in French and English) via the builder's `value_labels` argument —
see [Codes and value labels](https://qbolliet.github.io/dt-ducklake-manager/schema/#codes-and-value-labels).

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
- `logs/` — logging files produced by the builders and managers, written
  under the current working directory (`Path.cwd() / "logs"`) by default —
  never inside the installed package itself
- `notebooks/` — illustrative notebooks covering various use cases
- `outputs/` — program outputs

## License

MIT
