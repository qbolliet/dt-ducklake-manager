# dt-ducklake-manager

Utilities to build and manage a DuckLake database from a tabular dataset, designed for dashboard and ML prediction pipelines.

## Objectives

This package provides a complete lifecycle for a DuckLake database:

- **Build** a structured schema from any tabular dataset
- **Update** the database with new or modified observations (upsert)
- **Delete** rows based on filter conditions
- **Audit & validate** database integrity at configurable levels
- **Maintain** physical storage (file compaction, snapshot expiry)

The schema is structured around three layers:

- the `metadata` table references general information (label, type, categorical status, etc.) about each column ;
- the `dimension` tables associate each modality of a low-cardinality categorical variable to an `id` used in the `fact` table ;
- the `fact` table reflects the information in the original dataset.

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

## License

MIT
