# Importation des modules
# Modules de base
import narwhals as nw
import polars as pl
# Module de tests
import pytest
# Utilisation de DimensionManager (sous-classe concrète) pour instancier BaseSchemaManager
from dashboard_template_database._internal.managers.dimension import DimensionManager


# ---------------------------------------------------------------------------
# Fixture locale
# ---------------------------------------------------------------------------

# Initialisation d'une instance concrète de BaseSchemaManager pour les tests
@pytest.fixture
def manager(built_ducklake_schema):
    """Create a DimensionManager (concrete subclass) to test BaseSchemaManager methods.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.

    Returns:
        DimensionManager: initialized with the test connection.
    """
    return DimensionManager(connection=built_ducklake_schema, categorical_threshold=4)


# ===========================================================================
# Tests de _load_current_metadata()
# ===========================================================================

# Test que _load_current_metadata retourne un DataFrame narwhals avec les colonnes attendues
def test_load_current_metadata_returns_dataframe(manager):
    """Test that _load_current_metadata returns a narwhals DataFrame with expected columns.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    metadata = manager._load_current_metadata()
    # Vérification du type de retour
    assert isinstance(metadata, nw.DataFrame)
    # Vérification de la présence des colonnes de métadonnées
    assert 'name' in metadata.columns
    assert 'is_categorical' in metadata.columns
    assert 'is_primary_key' in metadata.columns


# Test que _load_current_metadata retourne un DataFrame non vide pour un schéma construit
def test_load_current_metadata_non_empty(manager):
    """Test that _load_current_metadata returns a non-empty DataFrame for a built schema.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    metadata = manager._load_current_metadata()
    # Le schéma built_ducklake_schema a été construit avec sample_df qui a 6 colonnes
    assert len(metadata) > 0


# ===========================================================================
# Tests de _table_exists()
# ===========================================================================

# Test que _table_exists retourne True pour une table existante
def test_table_exists_fact_table(manager):
    """Test that _table_exists returns True for an existing table.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    assert manager._table_exists('fact_table') is True


# Test que _table_exists retourne True pour la table metadata
def test_table_exists_metadata(manager):
    """Test that _table_exists returns True for the metadata table.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    assert manager._table_exists('metadata') is True


# Test que _table_exists retourne False pour une table inexistante
def test_table_exists_nonexistent(manager):
    """Test that _table_exists returns False for a non-existent table.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    assert manager._table_exists('nonexistent_table_xyz') is False


# ===========================================================================
# Tests de _get_primary_key_columns()
# ===========================================================================

# Test que _get_primary_key_columns retourne la liste des clés primaires
def test_get_primary_key_columns(manager):
    """Test that _get_primary_key_columns returns the list of primary key columns.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    pks = manager._get_primary_key_columns()
    # Vérification du type de retour
    assert isinstance(pks, list)
    # Le schéma est construit avec primary_keys=['id'] dans built_ducklake_schema
    assert 'id' in pks


# ===========================================================================
# Tests de _column_exists()
# ===========================================================================

# Test que _column_exists retourne True pour une colonne existante dans fact_table
def test_column_exists_true(manager):
    """Test that _column_exists returns True for an existing column.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    assert manager._column_exists('id', 'fact_table') is True


# Test que _column_exists retourne False pour une colonne inexistante
def test_column_exists_false(manager):
    """Test that _column_exists returns False for a non-existent column.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    assert manager._column_exists('nonexistent_col', 'fact_table') is False


# ===========================================================================
# Tests de _is_primary_key_column()
# ===========================================================================

# Test que _is_primary_key_column retourne True pour une colonne clé primaire
def test_is_primary_key_column_true(manager):
    """Test that _is_primary_key_column returns True for a primary key column.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    assert manager._is_primary_key_column('id') is True


# Test que _is_primary_key_column retourne False pour une colonne non-clé primaire
def test_is_primary_key_column_false(manager):
    """Test that _is_primary_key_column returns False for a non-primary-key column.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    assert manager._is_primary_key_column('value') is False


# ===========================================================================
# Tests de _invalidate_metadata_cache()
# ===========================================================================

# Test que _invalidate_metadata_cache vide le cache
def test_invalidate_metadata_cache(manager):
    """Test that _invalidate_metadata_cache sets the cache to None.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    # Chargement du cache
    _ = manager._load_current_metadata()
    assert manager._metadata_cache is not None

    # Invalidation du cache
    manager._invalidate_metadata_cache()
    assert manager._metadata_cache is None


# ===========================================================================
# Tests de _check_categorical_threshold()
# ===========================================================================

# Test que _check_categorical_threshold retourne True quand le nombre de modalités est inférieur au seuil
def test_check_categorical_threshold_below(manager):
    """Test that _check_categorical_threshold returns True when unique values <= threshold.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    # Série avec 3 modalités, seuil = 4
    series = nw.from_native(pl.Series('col', ['A', 'B', 'C', 'A']), series_only=True)
    result = manager._check_categorical_threshold(series, threshold=4)
    assert result is True


# Test que _check_categorical_threshold retourne False quand le nombre de modalités est supérieur au seuil
def test_check_categorical_threshold_above(manager):
    """Test that _check_categorical_threshold returns False when unique values > threshold.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    # Série avec 5 modalités, seuil = 4
    series = nw.from_native(pl.Series('col', ['A', 'B', 'C', 'D', 'E']), series_only=True)
    result = manager._check_categorical_threshold(series, threshold=4)
    assert result is False


# Test que _check_categorical_threshold retourne False pour une série entièrement nulle
def test_check_categorical_threshold_all_null(manager):
    """Test that _check_categorical_threshold returns False for an all-null series.

    Args:
        manager: DimensionManager fixture with a built schema.
    """
    series = nw.from_native(pl.Series('col', [None, None, None], dtype=pl.Utf8), series_only=True)
    result = manager._check_categorical_threshold(series, threshold=4)
    assert result is False
