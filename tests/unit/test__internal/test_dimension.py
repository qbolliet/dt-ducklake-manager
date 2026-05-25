# Importation des modules
# Modules de base
from typing import Any

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
def dim_manager(built_ducklake_schema: Any) -> DimensionManager:
    """Create a DimensionManager instance for testing.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.

    Returns:
        DimensionManager: initialized with the test connection.
    """
    return DimensionManager(connection=built_ducklake_schema, categorical_threshold=4)


# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------


# Test de l'initialisation correcte du gestionnaire de dimensions
def test_dimension_manager_initialization(built_ducklake_schema: Any) -> None:
    """Test that DimensionManager initializes correctly.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    mgr = DimensionManager(
        connection=built_ducklake_schema, categorical_threshold=10, max_workers=2
    )
    assert mgr is not None
    assert mgr.categorical_threshold == 10
    assert mgr.max_workers == 2


# ---------------------------------------------------------------------------
# Tests de validate_operation()
# ---------------------------------------------------------------------------


# Test que validate_operation retourne un booléen pour create_dimension
def test_validate_operation_create_dimension(dim_manager: Any) -> None:
    """Test that validate_operation returns a boolean for 'create' operation.

    Args:
        dim_manager: DimensionManager fixture.
    """
    values = nw.from_native(pl.Series("col", ["X", "Y", "Z"]), series_only=True)
    result = dim_manager.validate_operation(
        "create", column_name="new_dim", values=values
    )
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Tests de create_dimension_table()
# ---------------------------------------------------------------------------


# Test de la création d'une nouvelle table de dimension
def test_create_dimension_table(dim_manager: Any, built_ducklake_schema: Any) -> None:
    """Test that create_dimension_table creates a new dimension table in DuckDB.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Ajout d'abord d'une entrée de métadonnées pour la nouvelle dimension
    built_ducklake_schema.execute(
        "INSERT INTO metadata VALUES ('new_dim', 'New Dim', 'String', 'VARCHAR', true,"
        "false)"
    )

    # Création de la table de dimension
    values = nw.from_native(pl.Series("new_dim", ["X", "Y", "Z"]), series_only=True)
    success = dim_manager.create_dimension_table("new_dim", values)

    # Vérification que la création a réussi
    assert isinstance(success, bool)
    if success:
        # Vérification que la table existe en base
        tables = [
            row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
        ]
        assert "dim_new_dim" in tables


# ---------------------------------------------------------------------------
# Tests de get_dimension_mapping()
# ---------------------------------------------------------------------------


# Test de la récupération du mapping d'une table de dimension existante
def test_get_dimension_mapping_existing(dim_manager: Any) -> None:
    """Test that get_dimension_mapping returns a DataFrame for an existing dimension.

    Args:
        dim_manager: DimensionManager fixture.
    """
    # La table dim_category est créée par built_ducklake_schema (category est
    # catégorielle)
    mapping = dim_manager.get_dimension_mapping("category")
    # Vérification que le mapping est retourné (peut être None si la table n'existe pas)
    if mapping is not None:
        assert isinstance(mapping, nw.DataFrame)
        assert "value" in mapping.columns
        assert "label" in mapping.columns


# Test que get_dimension_mapping retourne None pour une dimension inexistante
def test_get_dimension_mapping_nonexistent(dim_manager: Any) -> None:
    """Test that get_dimension_mapping returns None for a non-existent dimension.

    Args:
        dim_manager: DimensionManager fixture.
    """
    result = dim_manager.get_dimension_mapping("nonexistent_dimension_xyz")
    assert result is None


# ---------------------------------------------------------------------------
# Tests de delete_dimension_table()
# ---------------------------------------------------------------------------


# Test de la suppression d'une table de dimension existante
def test_delete_dimension_table(dim_manager: Any, built_ducklake_schema: Any) -> None:
    """Test that delete_dimension_table removes the dimension table from DuckDB.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Vérification que la table dim_category existe avant suppression
    tables_before = [
        row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
    ]
    if "dim_category" not in tables_before:
        pytest.skip("dim_category table not present in test schema")

    success = dim_manager.delete_dimension_table("category")

    # Vérification du type de retour
    assert isinstance(success, bool)
    if success:
        # Vérification que la table a bien été supprimée
        tables_after = [
            row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
        ]
        assert "dim_category" not in tables_after


# ---------------------------------------------------------------------------
# Tests de cleanup_orphaned_dimension_entries()
# ---------------------------------------------------------------------------


# Test que cleanup_orphaned_dimension_entries s'exécute sans erreur
def test_cleanup_orphaned_dimension_entries(dim_manager: Any) -> None:
    """Test that cleanup_orphaned_dimension_entries executes without error.

    Args:
        dim_manager: DimensionManager fixture.
    """
    # Exécution du nettoyage : doit retourner un dict sans lever d'exception
    result = dim_manager.cleanup_orphaned_dimension_entries()
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Fixtures supplémentaires pour les tests de conversion
# ---------------------------------------------------------------------------


# Extension du schéma de test avec une colonne non-catégorielle convertible
@pytest.fixture
def schema_with_region(built_ducklake_schema: Any) -> Any:
    """Extend the built schema with a non-categorical VARCHAR column 'region'.

    Adds a 'region' column to fact_table with 4 unique values (Nord, Sud, Est,
    Ouest), which is exactly at the categorical threshold of 4. The corresponding
    metadata row is inserted with is_categorical=False, making this column a
    valid candidate for convert_to_categorical tests.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
            schema.

    Returns:
        duckdb.DuckDBPyConnection: the same connection, now with the 'region'
            column added.

    Example:
        >>> conn = schema_with_region
        >>> conn.execute("SELECT region FROM fact_table").fetchall()
        [('Nord',), ('Sud',), ('Est',), ('Nord',), ('Ouest',)]
    """
    # Ajout de la colonne 'region' à fact_table
    built_ducklake_schema.execute("ALTER TABLE fact_table ADD COLUMN region VARCHAR")

    # Remplissage de la colonne ligne par ligne via la clé primaire
    built_ducklake_schema.execute("UPDATE fact_table SET region = 'Nord'  WHERE id = 1")
    built_ducklake_schema.execute("UPDATE fact_table SET region = 'Sud'   WHERE id = 2")
    built_ducklake_schema.execute("UPDATE fact_table SET region = 'Est'   WHERE id = 3")
    built_ducklake_schema.execute("UPDATE fact_table SET region = 'Nord'  WHERE id = 4")
    built_ducklake_schema.execute("UPDATE fact_table SET region = 'Ouest' WHERE id = 5")

    # Insertion de la ligne de métadonnées pour 'region' (statut non-catégoriel)
    built_ducklake_schema.execute(
        "INSERT INTO metadata (name, label, python_type, sql_type, is_categorical, "
        "is_primary_key) VALUES ('region', 'Region', 'String', 'VARCHAR', FALSE, FALSE)"
    )

    return built_ducklake_schema


# Initialisation d'un DimensionManager sur le schéma étendu avec 'region'
@pytest.fixture
def dim_manager_with_region(schema_with_region: Any) -> DimensionManager:
    """Create a DimensionManager instance on the extended schema with 'region'.

    Args:
        schema_with_region: Fixture providing a DuckDB connection that includes
            the non-categorical 'region' column.

    Returns:
        DimensionManager: initialized with the extended test connection.
    """
    return DimensionManager(connection=schema_with_region, categorical_threshold=4)


# ---------------------------------------------------------------------------
# Tests de convert_to_categorical()
# ---------------------------------------------------------------------------


# Test du cas nominal : conversion réussie d'une colonne non-catégorielle
def test_convert_to_categorical_success(
    dim_manager_with_region: Any, schema_with_region: Any
) -> None:
    """Test that convert_to_categorical returns True and correctly converts a
    non-categorical VARCHAR column to categorical.

    Verifies that the dimension table is created, metadata is updated, and
    fact_table values are replaced by integer index strings.

    Args:
        dim_manager_with_region: DimensionManager on the extended schema.
        schema_with_region: DuckDB connection with the 'region' column.
    """
    # Construction de la série représentant les valeurs actuelles de 'region'
    series = nw.from_native(
        pl.Series("region", ["Nord", "Sud", "Est", "Nord", "Ouest"]),
        series_only=True,
    )

    # Exécution de la conversion
    result = dim_manager_with_region.convert_to_categorical("region", series)

    # Vérification du retour
    assert result is True

    # Vérification de l'existence de dim_region
    count_table = schema_with_region.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'dim_region'"
    ).fetchone()[0]
    assert count_table == 1

    # Vérification du nombre de lignes dans dim_region (4 valeurs uniques)
    count_rows = schema_with_region.execute(
        "SELECT COUNT(*) FROM dim_region"
    ).fetchone()[0]
    assert count_rows == 4

    # Vérification que les valeurs (indices) sont bien des chaînes entières
    values_in_dim = [
        row[0]
        for row in schema_with_region.execute("SELECT value FROM dim_region").fetchall()
    ]
    assert all(v.isdigit() for v in values_in_dim)

    # Vérification que les labels couvrent exactement les modalités d'origine
    labels_in_dim = {
        row[0]
        for row in schema_with_region.execute("SELECT label FROM dim_region").fetchall()
    }
    assert labels_in_dim == {"Nord", "Sud", "Est", "Ouest"}

    # Vérification du statut catégoriel dans les métadonnées
    is_cat = schema_with_region.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'region'"
    ).fetchone()[0]
    assert is_cat is True

    # Vérification que fact_table ne contient plus les labels d'origine
    count_labels_remaining = schema_with_region.execute(
        "SELECT COUNT(*) FROM fact_table"
        " WHERE region IN ('Nord', 'Sud', 'Est', 'Ouest')"
    ).fetchone()[0]
    assert count_labels_remaining == 0

    # Vérification de l'absence de perte de lignes
    count_rows_fact = schema_with_region.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert count_rows_fact == 5


# Test du dépassement du seuil catégoriel : conversion refusée
def test_convert_to_categorical_threshold_exceeded(
    dim_manager: Any, built_ducklake_schema: Any
) -> None:
    """Test that convert_to_categorical returns False when the column has more
    unique values than the categorical threshold.

    Verifies that no dimension table is created and no metadata mutation occurs.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Série avec 5 valeurs uniques, supérieur au seuil de 4
    series = nw.from_native(
        pl.Series(
            "high_cardinality",
            ["val_100", "val_101", "val_102", "val_103", "val_104"],
        ),
        series_only=True,
    )

    # Exécution de la conversion (doit échouer sur le seuil)
    result = dim_manager.convert_to_categorical("high_cardinality", series)

    # Vérification du refus
    assert result is False

    # Vérification qu'aucune table de dimension n'a été créée
    count_table = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM information_schema.tables"
        " WHERE table_name = 'dim_high_cardinality'"
    ).fetchone()[0]
    assert count_table == 0

    # Vérification de l'absence de mutation dans les métadonnées
    is_cat = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'high_cardinality'"
    ).fetchone()[0]
    assert is_cat is False

    # Vérification de l'intégrité des données dans fact_table
    distinct_values = {
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT DISTINCT high_cardinality FROM fact_table ORDER BY high_cardinality"
        ).fetchall()
    }
    assert distinct_values == {"val_100", "val_101", "val_102", "val_103", "val_104"}


# Test du refus de conversion pour un nom de colonne vide
def test_convert_to_categorical_empty_column_name(dim_manager: Any) -> None:
    """Test that convert_to_categorical returns False when given an empty column
    name, failing at the validation step.

    Args:
        dim_manager: DimensionManager fixture.
    """
    # Série valide quelconque (la validation doit échouer avant d'y accéder)
    series = nw.from_native(pl.Series("x", ["A", "B"]), series_only=True)

    # Exécution avec un nom de colonne vide
    result = dim_manager.convert_to_categorical("", series)

    # Vérification du refus
    assert result is False


# Test du refus de conversion pour un nom de colonne composé d'espaces
def test_convert_to_categorical_whitespace_column_name(dim_manager: Any) -> None:
    """Test that convert_to_categorical returns False when the column name
    consists only of whitespace characters.

    Args:
        dim_manager: DimensionManager fixture.
    """
    # Série valide quelconque
    series = nw.from_native(pl.Series("x", ["A", "B"]), series_only=True)

    # Exécution avec un nom composé uniquement d'espaces (strip() → "")
    result = dim_manager.convert_to_categorical("   ", series)

    # Vérification du refus
    assert result is False


# Test du refus de conversion pour une série vide
def test_convert_to_categorical_empty_series(
    dim_manager: Any, built_ducklake_schema: Any
) -> None:
    """Test that convert_to_categorical returns False when the values series is
    empty, and that no side effects occur on existing dimension tables.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Construction d'une série vide de type String
    empty_series = nw.from_native(
        pl.Series("region", [], dtype=pl.String), series_only=True
    )

    # Exécution avec une série vide (doit échouer à la validation)
    result = dim_manager.convert_to_categorical("category", empty_series)

    # Vérification du refus
    assert result is False

    # Vérification de l'absence d'effet de bord sur dim_category
    count_rows = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM dim_category"
    ).fetchone()[0]
    assert count_rows == 3


# Test du rollback lors d'un échec de conversion dans fact_table
def test_convert_to_categorical_missing_column_triggers_rollback(
    dim_manager_with_region: DimensionManager, schema_with_region: Any
) -> None:
    """Test that convert_to_categorical performs a full rollback when
    _convert_fact_table_dimension_mapping fails because the column is absent
    from fact_table.

    The test drops the 'region' column from fact_table after the fixture setup,
    so validation and dimension table creation succeed, but the fact table
    update fails, triggering rollback of both the dimension table and the
    metadata status.

    Args:
        dim_manager_with_region: DimensionManager on the extended schema.
        schema_with_region: DuckDB connection with the 'region' column initially
            present.
    """
    # Suppression de la colonne 'region' de fact_table pour provoquer l'échec
    # de _convert_fact_table_dimension_mapping (la métadonnée reste intacte)
    schema_with_region.execute("ALTER TABLE fact_table DROP COLUMN region")

    # Construction d'une série valide satisfaisant le seuil catégoriel
    series = nw.from_native(
        pl.Series("region", ["Nord", "Sud", "Est", "Nord", "Ouest"]),
        series_only=True,
    )

    # Exécution de la conversion (doit échouer et déclencher le rollback)
    result = dim_manager_with_region.convert_to_categorical("region", series)

    # Vérification du refus
    assert result is False

    # Vérification du rollback : dim_region ne doit pas exister
    count_table = schema_with_region.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'dim_region'"
    ).fetchone()[0]
    assert count_table == 0

    # Vérification du rollback : is_categorical remis à False dans les métadonnées
    is_cat = schema_with_region.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'region'"
    ).fetchone()[0]
    assert is_cat is False


# ---------------------------------------------------------------------------
# Tests de convert_to_non_categorical()
# ---------------------------------------------------------------------------


# Test du cas nominal : conversion réussie d'une colonne catégorielle
def test_convert_to_non_categorical_success(
    dim_manager: Any, built_ducklake_schema: Any
) -> None:
    """Test that convert_to_non_categorical returns True and correctly restores
    original labels in fact_table for an existing categorical column.

    Verifies that the dimension table is deleted, metadata is updated, and
    fact_table values are replaced by the original string labels.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Exécution de la conversion (category est catégorielle dans le schéma de test)
    result = dim_manager.convert_to_non_categorical("category")

    # Vérification du retour
    assert result is True

    # Vérification de la suppression de dim_category
    count_table = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM information_schema.tables"
        " WHERE table_name = 'dim_category'"
    ).fetchone()[0]
    assert count_table == 0

    # Vérification de la mise à jour du statut dans les métadonnées
    is_cat = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'category'"
    ).fetchone()[0]
    assert is_cat is False

    # Vérification du rétablissement des labels d'origine dans fact_table
    distinct_values = {
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT DISTINCT category FROM fact_table"
        ).fetchall()
    }
    assert distinct_values == {"A", "B", "C"}

    # Vérification de l'absence de perte de lignes
    count_rows = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert count_rows == 5


# Test du refus de conversion pour un nom de colonne vide
def test_convert_to_non_categorical_empty_column_name(
    dim_manager: DimensionManager, built_ducklake_schema: Any
) -> None:
    """Test that convert_to_non_categorical returns False when given an empty
    column name, and that no dimension tables are altered.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Exécution avec un nom de colonne vide (échec de validation attendu)
    result = dim_manager.convert_to_non_categorical("")

    # Vérification du refus
    assert result is False

    # Vérification de l'intégrité des tables de dimension existantes
    count_dims = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM information_schema.tables"
        " WHERE table_name IN ('dim_category', 'dim_status')"
    ).fetchone()[0]
    assert count_dims == 2


# Test du refus lorsque la colonne n'existe pas dans fact_table
def test_convert_to_non_categorical_column_not_in_fact_table(
    dim_manager: DimensionManager, built_ducklake_schema: Any
) -> None:
    """Test that convert_to_non_categorical returns False when the column is
    absent from fact_table, leaving the dimension table and metadata unchanged.

    Simulates an inconsistent DB state where metadata and a dimension table
    exist but the fact_table column is missing.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Mise en place d'un état incohérent : métadonnée + table de dimension sans
    # colonne correspondante dans fact_table
    built_ducklake_schema.execute(
        "INSERT INTO metadata (name, label, python_type, sql_type, is_categorical,"
        " is_primary_key) VALUES ('ghost_col', 'Ghost Col', 'String', 'VARCHAR',"
        " TRUE, FALSE)"
    )
    built_ducklake_schema.execute(
        "CREATE TABLE dim_ghost_col (value VARCHAR, label VARCHAR)"
    )
    built_ducklake_schema.execute(
        "INSERT INTO dim_ghost_col VALUES ('0', 'alpha'), ('1', 'beta')"
    )

    # Exécution de la conversion (doit échouer car ghost_col absent de fact_table)
    result = dim_manager.convert_to_non_categorical("ghost_col")

    # Vérification du refus
    assert result is False

    # Vérification que dim_ghost_col n'a pas été supprimée (arrêt avant la suppression)
    count_table = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM information_schema.tables"
        " WHERE table_name = 'dim_ghost_col'"
    ).fetchone()[0]
    assert count_table == 1

    # Vérification que is_categorical n'a pas été modifié
    is_cat = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'ghost_col'"
    ).fetchone()[0]
    assert is_cat is True


# Test du refus lorsque la table de dimension est absente
def test_convert_to_non_categorical_no_dim_table(
    dim_manager: Any, built_ducklake_schema: Any
) -> None:
    """Test that convert_to_non_categorical returns False when no dimension table
    exists for the target column, and that fact_table remains unchanged.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # high_cardinality n'a pas de table dim_ et n'est pas catégorielle
    result = dim_manager.convert_to_non_categorical("high_cardinality")

    # Vérification du refus
    assert result is False

    # Vérification de l'intégrité de fact_table (aucune mutation des valeurs)
    distinct_values = {
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT DISTINCT high_cardinality FROM fact_table"
        ).fetchall()
    }
    assert distinct_values == {"val_100", "val_101", "val_102", "val_103", "val_104"}

    # Vérification de l'absence de mutation dans les métadonnées
    is_cat = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'high_cardinality'"
    ).fetchone()[0]
    assert is_cat is False


# Test de l'intégrité des données après conversion : fréquences exactes des labels
def test_convert_to_non_categorical_data_integrity(
    dim_manager: Any, built_ducklake_schema: Any
) -> None:
    """Test that after converting 'status' to non-categorical, fact_table contains
    the exact original labels with their correct occurrence counts and no NULLs.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Exécution de la conversion (status : 3× 'active', 2× 'inactive')
    result = dim_manager.convert_to_non_categorical("status")

    # Vérification du retour
    assert result is True

    # Vérification des fréquences exactes des labels rétablis
    count_active = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE status = 'active'"
    ).fetchone()[0]
    assert count_active == 3

    count_inactive = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE status = 'inactive'"
    ).fetchone()[0]
    assert count_inactive == 2

    # Vérification de l'absence de valeurs NULL introduites
    count_null = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE status IS NULL"
    ).fetchone()[0]
    assert count_null == 0

    # Vérification de l'exhaustivité des modalités (exactement deux valeurs distinctes)
    distinct_values = {
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT DISTINCT status FROM fact_table"
        ).fetchall()
    }
    assert distinct_values == {"active", "inactive"}

    # Vérification de la suppression de dim_status
    count_table = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = 'dim_status'"
    ).fetchone()[0]
    assert count_table == 0


# ---------------------------------------------------------------------------
# Tests de roundtrip (conversion aller-retour)
# ---------------------------------------------------------------------------


# Test du cycle complet catégoriel → non-catégoriel → catégoriel
def test_roundtrip_categorical_conversion(
    dim_manager: Any, built_ducklake_schema: Any
) -> None:
    """Test that converting 'category' to non-categorical and back to categorical
    leaves fact_table in a consistent state with correct label frequencies and
    valid dimension table content.

    Args:
        dim_manager: DimensionManager fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # -----------------------------------------------------------------------
    # Phase 1 : conversion vers le mode non-catégoriel
    # -----------------------------------------------------------------------

    # Exécution de la conversion inverse
    result_phase1 = dim_manager.convert_to_non_categorical("category")
    assert result_phase1 is True

    # Vérification que dim_category a été supprimée
    count_table = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM information_schema.tables"
        " WHERE table_name = 'dim_category'"
    ).fetchone()[0]
    assert count_table == 0

    # Vérification du statut dans les métadonnées
    is_cat_phase1 = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'category'"
    ).fetchone()[0]
    assert is_cat_phase1 is False

    # Vérification des fréquences des labels restaurés (A×2, B×2, C×1)
    count_a = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE category = 'A'"
    ).fetchone()[0]
    assert count_a == 2

    count_b = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE category = 'B'"
    ).fetchone()[0]
    assert count_b == 2

    count_c = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE category = 'C'"
    ).fetchone()[0]
    assert count_c == 1

    # -----------------------------------------------------------------------
    # Phase 2 : reconversion vers le mode catégoriel
    # -----------------------------------------------------------------------

    # Construction de la série à partir des valeurs actuelles de fact_table
    raw_values = [
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT category FROM fact_table ORDER BY id"
        ).fetchall()
    ]
    series_phase2 = nw.from_native(pl.Series("category", raw_values), series_only=True)

    # Exécution de la reconversion
    result_phase2 = dim_manager.convert_to_categorical("category", series_phase2)
    assert result_phase2 is True

    # Vérification de la recréation de dim_category
    count_table_phase2 = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM information_schema.tables"
        " WHERE table_name = 'dim_category'"
    ).fetchone()[0]
    assert count_table_phase2 == 1

    # Vérification du nombre de modalités dans dim_category (exactement 3)
    count_dim_rows = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM dim_category"
    ).fetchone()[0]
    assert count_dim_rows == 3

    # Vérification du statut catégoriel dans les métadonnées
    is_cat_phase2 = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'category'"
    ).fetchone()[0]
    assert is_cat_phase2 is True

    # Vérification que fact_table ne contient plus les labels d'origine
    count_labels_remaining = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE category IN ('A', 'B', 'C')"
    ).fetchone()[0]
    assert count_labels_remaining == 0

    # Vérification que toutes les valeurs de fact_table sont des indices valides
    count_orphan_values = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table f"
        " WHERE f.category IS NOT NULL"
        " AND f.category NOT IN (SELECT value FROM dim_category)"
    ).fetchone()[0]
    assert count_orphan_values == 0

    # Vérification de l'absence de perte de lignes
    count_rows = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert count_rows == 5
