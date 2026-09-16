# dt-ducklake-manager

Utilities to build and manage a DuckLake database from a tabular dataset, designed for dashboard and ML prediction pipelines.

## Objectives

This package provides a complete lifecycle for a DuckLake database:

- **Build** a structured schema from any tabular dataset
- **Update** the database with new or modified observations (upsert)
- **Delete** rows based on filter conditions
- **Audit & validate** database integrity at configurable levels
- **Maintain** physical storage (file compaction, snapshot expiry)

The schema is structured around exactly three tables per result set:

- the `fact_table` holds the observations; categorical columns store their **original labels**, so there are no dimension tables and no synthetic codes ;
- the `metadata` table describes each column of the fact table (label, SQL type, categorical status, primary key) ;
- the `dataset_metadata` table describes the result set itself (title, description, source, last update, schema version).

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

# 1. Build the schema from an initial dataset
df = pd.DataFrame({
    "id": [1, 2, 3],
    "city": ["Paris", "Berlin", "Madrid"],
    "score": [0.9, 0.7, 0.5],
})
builder = DuckLakeTablesBuilder(
    df,
    categorical_threshold=200,
    primary_keys=["id"],
    connection=connection,
    dataset_label="City scores",
)
builder.build_schema()

# 2. Update the database with new observations (upsert)
df_new = pd.DataFrame({"id": [2, 4], "city": ["Lyon", "Rome"], "score": [0.8, 0.6]})
updater = DatabaseUpdater(connection=connection)
updater.update_database(update_df=df_new)

# 3. Delete rows matching a condition
deleter = DatabaseDeleter(connection=connection)
deleter.delete_rows(filters=[("score", "<", 0.6)])

# 4. Audit database integrity
auditor = DatabaseAuditor(connection=connection)
report = auditor.validate_database(ValidationLevel.STANDARD)
print(report.recommendations)

# 5. Run full maintenance (compaction, snapshot expiry)
maintenance = DuckLakeMaintenance(connection)
maintenance.maintain(MaintenancePolicy(retention_days=30))
```

More detailed examples and parametrization walkthroughs are available in the `notebooks/` folder.

## License

MIT
