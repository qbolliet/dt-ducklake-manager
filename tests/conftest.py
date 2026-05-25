# Importation des modules
# Modules de base
import logging
import warnings
from datetime import datetime

import duckdb
import polars as pl

# Module de tests
import pytest

# Modules du package
from dt_ducklake_manager.schema import DuckLakeTablesBuilder


# Initialisation d'un jeu de données d'exemple
@pytest.fixture
def sample_df() -> pl.DataFrame:
    """Create a sample DataFrame for testing."""
    return pl.DataFrame(
        {
            "id": list(range(1, 6)),
            "category": ["A", "B", "A", "C", "B"],
            "value": [
                0.1,
                0.2,
                0.3,
                0.4,
                0.5,
            ],  # Valeurs fixes pour la reproductibilité
            "date": pl.date_range(
                datetime(2024, 1, 1), datetime(2024, 1, 5), "1d", eager=True
            ),
            "status": ["active", "inactive", "active", "active", "inactive"],
            "high_cardinality": [f"val_{i}" for i in range(100, 105)],
        }
    )


# Initialisation du dictionnaire de labels pour les colonnes
@pytest.fixture
def column_labels() -> dict[str, str]:
    """Fixture for DataFrame column labels."""
    return {
        "id": "Identifier",
        "category": "Category Name",
        "value": "Numeric Value",
        "date": "Date Field",
        "status": "Status Field",
    }


# Fonction d'initalisation de logging
@pytest.fixture(autouse=True)
def setup_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Set logging for tests."""
    caplog.set_level(logging.INFO)


# Fixture fournissant une connexion in-memory avec DEUX schémas dans un même catalogue
@pytest.fixture
def multi_schema_connection() -> duckdb.DuckDBPyConnection:
    """Provide an in-memory connection holding two schemas in one catalog.

    Builds a ``predictions`` schema and a ``shapley`` schema, each with its own
    ``fact_table``, ``metadata`` and ``dim_*`` tables, on a single shared connection.
    Mirrors the multi-schema layout (one catalog, several schemas) without needing a
    real DuckLake catalog file.

    Returns:
        duckdb.DuckDBPyConnection: connection with both schemas fully built.
    """
    conn = duckdb.connect(":memory:")

    # Schéma "predictions" : 3 lignes, catégorielle 'category'
    predictions_df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["A", "B", "A"],
            "value": [0.1, 0.2, 0.3],
        }
    )
    # Schéma "shapley" : granularité et colonnes propres, modalités différentes
    shapley_df = pl.DataFrame(
        {
            "id": [1, 2],
            "category": ["A", "C"],
            "shap_value": [1.5, 2.5],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            predictions_df,
            categorical_threshold=4,
            primary_keys=["id"],
            connection=conn,
            schema="predictions",
        ).build_schema()
        DuckLakeTablesBuilder(
            shapley_df,
            categorical_threshold=4,
            primary_keys=["id"],
            connection=conn,
            schema="shapley",
        ).build_schema()

    return conn


# Fixture pour DataFrame avec doublons
@pytest.fixture
def sample_df_with_duplicates() -> pl.DataFrame:
    """Create a sample DataFrame with duplicates for testing.

    Rows 0&3 and rows 1&4 are fully identical (including 'value'), so they are
    detected as duplicates regardless of the column subset used for deduplication.
    """
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 1, 2],  # Doublons sur id 1 et 2
            "category": ["A", "B", "A", "A", "B"],
            "status": ["active", "inactive", "active", "active", "inactive"],
            "value": [
                10.0,
                20.0,
                30.0,
                10.0,
                20.0,
            ],  # Même valeur pour les lignes dupliquées
        }
    )


# Fixture fournissant une connexion DuckLake (in-memory) avec un schéma déjà construit
@pytest.fixture
def built_ducklake_schema(sample_df: pl.DataFrame) -> duckdb.DuckDBPyConnection:
    """Provide an in-memory DuckDB connection with a fully built schema.

    The schema is built from sample_df with categorical_threshold=4 and
    primary_keys=['id'], ready to be used with operation managers.
    Uses an in-memory connection (no DuckLake catalog file) for test isolation.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
        )
    builder.build_schema()
    return builder.conn
