# Database schema builder for dashboard template

This directory contains utilities to build a specific generic database scheme from a table.
> Full documentation is available [here](https://qbolliet.github.io/dt-ducklake-manager/)

## Objectives

This package is built to create :
* a generic schema to store a a table as a database ;
* export this schema a relational DuckDB database.

## Organization

The repository is organized as follow :

* the `docs` folder contains all the package documentation and descriptions of the database scheme
* the `logs` folder contains the logging files associated with the builders
* the `notebooks` folder contains illustrative notebooks
* the `outputs` folder contains program outputs.
* the `parameters` folder contains default labels for naming columns.
* the `scripts` folder contains a script for building a database from the table and with the paramters listed in the `config/database.yaml` file.

## Installation

### Package and dependencies

```bash
git clone https://github.com/qbolliet/dt-ducklake-manager.git
poetry install
```

The package is then usable as any other python package.

### Parametrisation

File in the `config/database.yaml` file :
```yaml
INPUT_DATA : '../data/df_origin.csv'
OUTPUT_DATA : '../outputs/database.db'
THRESHOLD : 200
``` 

### Documentation

To visualize the documentation :
```
poetry install --with docs
```

```
mkdocs build --port 5000
```

## Usage

Here's an example of how to use the functions in the package:

```python
import pandas as pd
from dt_ducklake_manager.builders.schema import SchemaBuilder
from dt_ducklake_manager.builders.tables import DuckdbTablesBuilder
from dt_ducklake_manager.loaders.local.loader import Loader

# Load a sample DataFrame
loader = Loader()
sample_data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie'],
    'Age': [25, 30, 35],
    'City': ['Paris', 'Berlin', 'Madrid']
})

# Initialize the SchemaBuilder
schema_builder = SchemaBuilder(df=sample_data, categorical_threshold=3)

# Build the schema
metadata, dimension_tables, fact_table = schema_builder.build()

# Initialize the DuckDB tables builder
duckdb_builder = DuckdbTablesBuilder(df=sample_data)

# Create the schema in DuckDB
duckdb_builder.build_schema()

# Display the schema in DuckDB
duckdb_builder.display_schema()
``` 

## License

The package is licensed under the MIT License.