# Database schema builder for dashboard template

This package contains utilities to build a specific generic database scheme from a table.

## Objectives

This package is built to create :

* a generic schema to store a table as a database :

    - the `metadata` table references general information (label, type etc ...) about each dataframe column ;
    -  the `dimension` tables associate each modality of a categorical variable to an `id` used in the `fact` table ;
    - the `fact` table reflects the information in the original table ;

* export this schema a relational DuckDB database.

![Scheme for table storage](assets/schema_bdd.png)

## Installation

### Package and dependencies

```bash
git clone https://github.com/qbolliet/dt-ducklake-manager.git
poetry install
```

The package is then usable as any other python package.

### Parametrisation

File in the `config.yaml` file :
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