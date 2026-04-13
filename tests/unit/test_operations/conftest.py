# Importation des modules
# Modules de base
from datetime import datetime

import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.operations import DatabaseDeleter, DatabaseUpdater

# ---------------------------------------------------------------------------
# Fixtures communes aux tests d'opérations
# ---------------------------------------------------------------------------


# Initialisation d'un DatabaseUpdater pour les tests
@pytest.fixture
def updater(built_ducklake_schema):
    """Create a DatabaseUpdater instance for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.

    Returns:
        DatabaseUpdater: initialized with the test connection and categorical_threshold=4.
    """
    return DatabaseUpdater(
        connection=built_ducklake_schema,
        categorical_threshold=4,
        enable_validation=True,
    )


# Initialisation d'un DatabaseDeleter pour les tests
@pytest.fixture
def deleter(built_ducklake_schema):
    """Create a DatabaseDeleter instance for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.

    Returns:
        DatabaseDeleter: initialized with the test connection, categorical_threshold=4
        and auto_cleanup enabled.
    """
    return DatabaseDeleter(
        connection=built_ducklake_schema,
        categorical_threshold=4,
        enable_validation=True,
        auto_cleanup=True,
    )


# Initialisation d'un DataFrame de mise à jour avec de nouvelles lignes (ids inédits)
@pytest.fixture
def update_df():
    """Create a DataFrame with new rows for update testing.

    Returns:
        pl.DataFrame: a small DataFrame with two new rows to insert (ids 10 and 11).
    """
    return pl.DataFrame(
        {
            "id": [10, 11],
            "category": ["A", "C"],
            "value": [1.1, 2.2],
            "date": pl.date_range(
                datetime(2024, 2, 1), datetime(2024, 2, 2), "1d", eager=True
            ),
            "status": ["active", "inactive"],
            "high_cardinality": ["val_200", "val_201"],
        }
    )
