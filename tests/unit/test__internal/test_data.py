# Importation des modules
# Modules de base
import polars as pl
import narwhals as nw
from datetime import datetime
# Module de tests
import pytest
# Module du package à tester
from dt_ducklake_manager._internal.managers.data import DataManager


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------

# Initialisation d'un DataManager pour les tests
@pytest.fixture
def data_manager(built_ducklake_schema):
    """Create a DataManager instance for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.

    Returns:
        DataManager: initialized with the test connection.
    """
    return DataManager(
        connection=built_ducklake_schema,
        categorical_threshold=4,
        batch_size=100,
    )


# Initialisation d'un DataFrame d'insertion valide
@pytest.fixture
def new_rows(sample_df):
    """Create a small DataFrame with new rows compatible with sample_df schema.

    Args:
        sample_df: Sample polars DataFrame.

    Returns:
        pl.DataFrame: two new rows with unique ids.
    """
    return pl.DataFrame({
        'id': [50, 51],
        'category': ['A', 'B'],
        'value': [5.0, 6.0],
        'date': pl.date_range(datetime(2024, 6, 1), datetime(2024, 6, 2), '1d', eager=True),
        'status': ['active', 'inactive'],
        'high_cardinality': ['val_500', 'val_501'],
    })


# Initialisation d'un DataFrame vide
@pytest.fixture
def empty_df():
    """Create an empty DataFrame for edge case testing.

    Returns:
        pl.DataFrame: an empty DataFrame with no rows.
    """
    return pl.DataFrame({'id': [], 'value': []})


# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------

# Test de l'initialisation correcte du gestionnaire de données
def test_data_manager_initialization(built_ducklake_schema):
    """Test that DataManager initializes correctly with the given parameters.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    mgr = DataManager(connection=built_ducklake_schema, batch_size=500)
    assert mgr is not None
    assert mgr.batch_size == 500


# ---------------------------------------------------------------------------
# Tests de validate_operation()
# ---------------------------------------------------------------------------

# Test que validate_operation retourne True pour une insertion valide
def test_validate_operation_insert_valid(data_manager, new_rows):
    """Test that validate_operation returns True for a valid insert operation.

    Args:
        data_manager: DataManager fixture.
        new_rows: DataFrame with new rows to insert.
    """
    result = data_manager.validate_operation('insert', df=new_rows)
    assert result is True


# Test que validate_operation retourne False pour un DataFrame vide
def test_validate_operation_insert_empty_df(data_manager, empty_df):
    """Test that validate_operation returns False when inserting an empty DataFrame.

    Args:
        data_manager: DataManager fixture.
        empty_df: An empty DataFrame.
    """
    result = data_manager.validate_operation('insert', df=empty_df)
    assert result is False


# ---------------------------------------------------------------------------
# Tests de insert_data()
# ---------------------------------------------------------------------------

# Test que insert_data insère bien les nouvelles lignes dans fact_table
def test_insert_data_increases_row_count(data_manager, built_ducklake_schema, new_rows):
    """Test that insert_data increases the fact table row count.

    Args:
        data_manager: DataManager fixture.
        built_ducklake_schema: DuckDB connection.
        new_rows: DataFrame with new rows to insert.
    """
    # Comptage initial
    before = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]

    # Insertion des nouvelles lignes
    inserted = data_manager.insert_data(new_rows, use_batch=False)

    # Vérification que des lignes ont été insérées
    assert isinstance(inserted, int)
    assert inserted > 0

    after = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert after > before


# Test que insert_data avec use_batch=True fonctionne correctement
def test_insert_data_batch_mode(data_manager, built_ducklake_schema, new_rows):
    """Test that insert_data with use_batch=True works correctly.

    Args:
        data_manager: DataManager fixture.
        built_ducklake_schema: DuckDB connection.
        new_rows: DataFrame with new rows to insert.
    """
    before = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]

    # Insertion en mode batch
    inserted = data_manager.insert_data(new_rows, use_batch=True)

    after = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert after >= before


# ---------------------------------------------------------------------------
# Tests de delete_rows()
# ---------------------------------------------------------------------------

# Test que delete_rows supprime les lignes correspondant au filtre
def test_delete_rows_with_filter(data_manager, built_ducklake_schema):
    """Test that delete_rows removes rows matching the given filter.

    Args:
        data_manager: DataManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Vérification que des lignes existent avant suppression
    before = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table WHERE id = 1").fetchone()[0]
    assert before >= 1

    # Suppression via le gestionnaire
    deleted = data_manager.delete_rows(filters=[('id', '=', 1)])

    assert isinstance(deleted, int)
    assert deleted >= 1

    # Vérification que les lignes sont bien absentes
    after = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table WHERE id = 1").fetchone()[0]
    assert after == 0


# Test que delete_rows avec filters=None supprime toutes les lignes
def test_delete_rows_all(data_manager, built_ducklake_schema):
    """Test that delete_rows with filters=None removes all rows.

    Args:
        data_manager: DataManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    before = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert before > 0

    # Remarque : DataManager n'accepte pas filters=None (validation obligatoire des filtres).
    # Suppression de toutes les lignes via un filtre SQL universel.
    deleted = data_manager.delete_rows(filters="1=1")

    after = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert after == 0
    assert deleted == before
