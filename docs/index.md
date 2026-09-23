# dt-ducklake-manager

Utilities to build and manage a DuckLake database from a tabular dataset, designed for dashboard and ML prediction pipelines.

## Objectives

This package provides a complete lifecycle for a DuckLake database:

- **Build** a structured schema from any tabular dataset
- **Update** the database with new or modified observations (upsert)
- **Delete** rows based on filter conditions
- **Audit & validate** database integrity at configurable levels
- **Maintain** physical storage (file compaction, snapshot expiry)

The schema is structured around exactly three tables per result set, with **no
dimension table anywhere**:

- the `fact_table` holds the observations; categorical columns store their **original labels** directly (Parquet dictionary-encoding absorbs the storage cost), so there is no synthetic code ;
- the `metadata` table describes each column of the fact table — one row per column — and is the contract between the database and the interface (label, SQL type, primary-key and categorical flags, column hierarchy via `parent_name`, the code → label link via `label_for`, and the UI fields `unit`, `display_format`, `family`, `description`, `default_aggregation`) ;
- the `dataset_metadata` table describes the result set itself (title, description, source, last update, schema version, and `cluster_by`, the physical sort key).

See [Schema and data model](schema.md) for the full description, and [Storage
lifecycle and maintenance](maintenance.md) for how the physical storage is
kept efficient over time.

![Scheme for table storage](assets/schema_bdd.png)

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

A code column can also carry its own label column(s) (e.g. a nomenclature code
labelled in French and English) via the builder's `value_labels` argument —
see [Codes and value labels](schema.md#codes-and-value-labels).

More detailed examples and parametrization walkthroughs are available in the `notebooks/` folder.

## License

MIT
