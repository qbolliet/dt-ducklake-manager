# Importation des modules
# Modules de base
import warnings
import polars as pl
from datetime import datetime
# Module de tests
import pytest
# Modules du package à tester
from dashboard_template_database.operations import DatabaseUpdater


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------

# Initialisation d'un DatabaseUpdater pour les tests
@pytest.fixture
def updater(built_ducklake_schema):
    """Create a DatabaseUpdater instance for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.

    Returns:
        DatabaseUpdater: initialized with the test connection.
    """
    return DatabaseUpdater(
        connection=built_ducklake_schema,
        categorical_threshold=4,
        enable_validation=True,
    )


# Initialisation d'un DataFrame de mise à jour avec de nouvelles lignes
@pytest.fixture
def update_df(sample_df):
    """Create a DataFrame with new rows for update testing.

    Args:
        sample_df: Sample polars DataFrame.

    Returns:
        pl.DataFrame: a small DataFrame with rows to insert.
    """
    return pl.DataFrame({
        'id': [10, 11],
        'category': ['A', 'C'],
        'value': [1.1, 2.2],
        'date': pl.date_range(datetime(2024, 2, 1), datetime(2024, 2, 2), '1d', eager=True),
        'status': ['active', 'inactive'],
        'high_cardinality': ['val_200', 'val_201'],
    })


# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------

# Test de l'initialisation correcte de DatabaseUpdater
def test_updater_initialization(built_ducklake_schema):
    """Test that DatabaseUpdater initializes without errors.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, categorical_threshold=4)
    assert updater is not None
    assert updater.categorical_threshold == 4
    assert updater.batch_size > 0


# Test de l'initialisation avec enable_validation=False
def test_updater_initialization_without_validation(built_ducklake_schema):
    """Test that DatabaseUpdater can be initialized with validation disabled.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, enable_validation=False)
    assert updater.auditor is None


# ---------------------------------------------------------------------------
# Tests de validate_operation()
# ---------------------------------------------------------------------------

# Test que validate_operation retourne True pour une insertion valide
def test_validate_operation_insert_returns_bool(updater, update_df):
    """Test that validate_operation returns a boolean for an insert operation.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with new rows.
    """
    result = updater.validate_operation('insert', df=update_df)
    assert isinstance(result, bool)


# Test que validate_operation retourne True quand la validation est désactivée
def test_validate_operation_disabled_returns_true(built_ducklake_schema, update_df):
    """Test that validate_operation always returns True when validation is disabled.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
        update_df: DataFrame with new rows.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, enable_validation=False)
    result = updater.validate_operation('insert', df=update_df)
    assert result is True


# ---------------------------------------------------------------------------
# Tests de update_database()
# ---------------------------------------------------------------------------

# Test d'insertion de nouvelles lignes sans transaction
def test_update_database_insert_new_rows(updater, built_ducklake_schema, update_df):
    """Test that update_database inserts new rows into the fact table.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
        update_df: DataFrame with new rows to insert.
    """
    # Comptage initial
    initial_count = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]

    # Insertion des nouvelles lignes sans transaction
    # Remarque : keep='first' est requis car narwhals ne supporte pas keep=False (valeur par défaut)
    result = updater.update_database(
        update_df=update_df,
        keep='first',
        use_transaction=False,
    )

    # Vérification que l'opération s'est bien déroulée
    assert isinstance(result, bool)
    assert result is True

    # Vérification que le nombre de lignes a augmenté
    final_count = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert final_count > initial_count


# Test d'insertion avec déduplication sur les doublons du DataFrame d'entrée
def test_update_database_with_dedup_on_update(updater, built_ducklake_schema):
    """Test that update_database removes duplicates from the update DataFrame when requested.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # DataFrame avec deux lignes IDENTIQUES sur toutes les colonnes (id=20, même date)
    # La déduplication utilise toutes les colonnes (primary_keys non transmises au niveau du
    # DataFrame d'entrée) : les deux lignes doivent donc être parfaitement identiques.
    df_with_dup = pl.DataFrame({
        'id': [20, 20, 21],
        'category': ['A', 'A', 'B'],
        'value': [5.0, 5.0, 6.0],
        'date': [datetime(2024, 3, 1), datetime(2024, 3, 1), datetime(2024, 3, 2)],
        'status': ['active', 'active', 'inactive'],
        'high_cardinality': ['val_300', 'val_300', 'val_301'],
    })

    result = updater.update_database(
        update_df=df_with_dup,
        check_duplicates_update=True,
        check_duplicates_db=False,
        keep='first',
        use_transaction=False,
    )

    assert result is True

    # Vérification que la déduplication a fonctionné : id=20 ne doit apparaître qu'une seule fois
    count_20 = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 20"
    ).fetchone()[0]
    assert count_20 == 1

    # Vérification que id=21 a bien été inséré
    count_21 = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 21"
    ).fetchone()[0]
    assert count_21 == 1
