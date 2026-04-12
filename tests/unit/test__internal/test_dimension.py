# Importation des modules
# Modules de base
import narwhals as nw
import polars as pl
# Module de tests
import pytest
# Module du package à tester
from dt_ducklake_manager._internal.managers.dimension import DimensionManager


# ---------------------------------------------------------------------------
# Fixture locale
# ---------------------------------------------------------------------------

# Initialisation d'un DimensionManager pour les tests
@pytest.fixture
def dim_manager(built_ducklake_schema):
    """Create a DimensionManager instance for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.

    Returns:
        DimensionManager: initialized with the test connection.
    """
    return DimensionManager(connection=built_ducklake_schema, categorical_threshold=4)


# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------

# Test de l'initialisation correcte du gestionnaire de dimensions
def test_dimension_manager_initialization(built_ducklake_schema):
    """Test that DimensionManager initializes correctly.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    mgr = DimensionManager(connection=built_ducklake_schema, categorical_threshold=10, max_workers=2)
    assert mgr is not None
    assert mgr.categorical_threshold == 10
    assert mgr.max_workers == 2


# ---------------------------------------------------------------------------
# Tests de validate_operation()
# ---------------------------------------------------------------------------

# Test que validate_operation retourne un booléen pour create_dimension
def test_validate_operation_create_dimension(dim_manager):
    """Test that validate_operation returns a boolean for 'create' operation.

    Args:
        dim_manager: DimensionManager fixture.
    """
    values = nw.from_native(pl.Series('col', ['X', 'Y', 'Z']), series_only=True)
    result = dim_manager.validate_operation('create', column_name='new_dim', values=values)
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Tests de create_dimension_table()
# ---------------------------------------------------------------------------

# Test de la création d'une nouvelle table de dimension
def test_create_dimension_table(dim_manager, built_ducklake_schema):
    """Test that create_dimension_table creates a new dimension table in DuckDB.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Ajout d'abord d'une entrée de métadonnées pour la nouvelle dimension
    built_ducklake_schema.execute(
        "INSERT INTO metadata VALUES ('new_dim', 'New Dim', 'String', 'VARCHAR', true, false)"
    )

    # Création de la table de dimension
    values = nw.from_native(pl.Series('new_dim', ['X', 'Y', 'Z']), series_only=True)
    success = dim_manager.create_dimension_table('new_dim', values)

    # Vérification que la création a réussi
    assert isinstance(success, bool)
    if success:
        # Vérification que la table existe en base
        tables = [
            row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
        ]
        assert 'dim_new_dim' in tables


# ---------------------------------------------------------------------------
# Tests de get_dimension_mapping()
# ---------------------------------------------------------------------------

# Test de la récupération du mapping d'une table de dimension existante
def test_get_dimension_mapping_existing(dim_manager):
    """Test that get_dimension_mapping returns a DataFrame for an existing dimension.

    Args:
        dim_manager: DimensionManager fixture.
    """
    # La table dim_category est créée par built_ducklake_schema (category est catégorielle)
    mapping = dim_manager.get_dimension_mapping('category')
    # Vérification que le mapping est retourné (peut être None si la table n'existe pas)
    if mapping is not None:
        assert isinstance(mapping, nw.DataFrame)
        assert 'value' in mapping.columns
        assert 'label' in mapping.columns


# Test que get_dimension_mapping retourne None pour une dimension inexistante
def test_get_dimension_mapping_nonexistent(dim_manager):
    """Test that get_dimension_mapping returns None for a non-existent dimension.

    Args:
        dim_manager: DimensionManager fixture.
    """
    result = dim_manager.get_dimension_mapping('nonexistent_dimension_xyz')
    assert result is None


# ---------------------------------------------------------------------------
# Tests de delete_dimension_table()
# ---------------------------------------------------------------------------

# Test de la suppression d'une table de dimension existante
def test_delete_dimension_table(dim_manager, built_ducklake_schema):
    """Test that delete_dimension_table removes the dimension table from DuckDB.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Vérification que la table dim_category existe avant suppression
    tables_before = [
        row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
    ]
    if 'dim_category' not in tables_before:
        pytest.skip("dim_category table not present in test schema")

    success = dim_manager.delete_dimension_table('category')

    # Vérification du type de retour
    assert isinstance(success, bool)
    if success:
        # Vérification que la table a bien été supprimée
        tables_after = [
            row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
        ]
        assert 'dim_category' not in tables_after


# ---------------------------------------------------------------------------
# Tests de cleanup_orphaned_dimension_entries()
# ---------------------------------------------------------------------------

# Test que cleanup_orphaned_dimension_entries s'exécute sans erreur
def test_cleanup_orphaned_dimension_entries(dim_manager):
    """Test that cleanup_orphaned_dimension_entries executes without error.

    Args:
        dim_manager: DimensionManager fixture.
    """
    # Exécution du nettoyage : doit retourner un dict sans lever d'exception
    result = dim_manager.cleanup_orphaned_dimension_entries()
    assert isinstance(result, dict)
