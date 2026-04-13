# Importation des modules
# Module de tests
# Modules du package à tester
from dt_ducklake_manager.operations import DatabaseDeleter

# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------


# Test de l'initialisation correcte de DatabaseDeleter
def test_deleter_initialization(built_ducklake_schema):
    """Test that DatabaseDeleter initializes without errors.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    deleter = DatabaseDeleter(connection=built_ducklake_schema, categorical_threshold=4)
    assert deleter is not None
    assert deleter.categorical_threshold == 4


# Test de l'initialisation avec enable_validation=False
def test_deleter_initialization_without_validation(built_ducklake_schema):
    """Test that DatabaseDeleter can be initialized with validation disabled.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    deleter = DatabaseDeleter(connection=built_ducklake_schema, enable_validation=False)
    # Vérification que l'auditeur n'est pas initialisé
    assert deleter.enable_validation is False


# ---------------------------------------------------------------------------
# Tests de validate_operation()
# ---------------------------------------------------------------------------


# Test que validate_operation retourne un booléen pour une suppression valide
def test_validate_operation_delete_returns_bool(deleter):
    """Test that validate_operation returns a boolean for a delete operation.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    result = deleter.validate_operation("delete", filters=[("id", "=", 1)])
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Tests de delete_rows()
# ---------------------------------------------------------------------------


# Test de la suppression de lignes avec un filtre simple
def test_delete_rows_with_filter(deleter, built_ducklake_schema):
    """Test that delete_rows removes rows matching the given filter.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Vérification que la ligne id=1 existe avant suppression
    before = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 1"
    ).fetchone()[0]
    assert before >= 1

    # Suppression de la ligne avec id=1
    deleted_count = deleter.delete_rows(
        filters=[("id", "=", 1)],
        use_transaction=False,
    )

    # Vérification que le nombre de lignes supprimées est cohérent
    assert isinstance(deleted_count, int)
    assert deleted_count >= 1

    # Vérification que la ligne est bien absente
    after = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 1"
    ).fetchone()[0]
    assert after == 0


# Test de la suppression de lignes avec un filtre OR (liste de listes de tuples)
def test_delete_rows_with_or_filter(deleter, built_ducklake_schema):
    """Test that delete_rows handles OR filters (list of lists of tuples).

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Suppression des lignes avec id=2 OU id=3
    before = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id IN (2, 3)"
    ).fetchone()[0]
    assert before >= 1

    deleted_count = deleter.delete_rows(
        filters=[[("id", "=", 2)], [("id", "=", 3)]],
        use_transaction=False,
    )
    assert deleted_count >= 1

    # Vérification que les lignes sont bien absentes
    after = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id IN (2, 3)"
    ).fetchone()[0]
    assert after == 0


# Test de la suppression de toutes les lignes avec filters=None
def test_delete_rows_all(deleter, built_ducklake_schema):
    """Test that delete_rows with filters=None removes all rows.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    before = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert before > 0

    # Remarque : DatabaseDeleter n'accepte pas filters=None (validation obligatoire des filtres).
    # Suppression de toutes les lignes via un filtre SQL universel.
    deleted_count = deleter.delete_rows(filters="1=1", use_transaction=False)
    assert deleted_count == before

    after = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[
        0
    ]
    assert after == 0


# ---------------------------------------------------------------------------
# Tests de delete_columns()
# ---------------------------------------------------------------------------


# Test de la suppression d'une colonne non-clé primaire
def test_delete_columns_single_column(deleter, built_ducklake_schema):
    """Test that delete_columns removes a non-primary-key column from the fact table.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Vérification que la colonne 'value' existe avant suppression
    columns_before = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "value" in columns_before

    result = deleter.delete_columns(["value"], use_transaction=False)

    # Vérification que le résultat est un dictionnaire de statuts
    assert isinstance(result, dict)
    assert "value" in result
    # Vérification que la colonne est bien supprimée
    columns_after = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "value" not in columns_after


# ---------------------------------------------------------------------------
# Tests de changement de statut catégoriel lors d'une suppression
# ---------------------------------------------------------------------------


# Test de conversion non-catégorielle → catégorielle après suppression de lignes
def test_delete_rows_non_categorical_becomes_categorical(
    deleter, built_ducklake_schema
):
    """Test that a non-categorical column becomes categorical when its unique value count
    drops to or below the threshold after rows are deleted.

    The sample schema is built with categorical_threshold=4. The column 'high_cardinality'
    initially holds 5 unique values (val_100..val_104) and is NOT categorical. The row
    with id=5 carries the only occurrence of 'val_104'. Deleting that row leaves exactly
    4 distinct values (val_100..val_103) which equals the threshold, triggering conversion
    to categorical and creation of dim_high_cardinality.

    Args:
        deleter: DatabaseDeleter fixture with auto_cleanup=True.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    # Vérification initiale : high_cardinality n'est pas catégorielle (5 valeurs > seuil=4)
    is_cat_before = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'high_cardinality'"
    ).fetchone()[0]
    assert is_cat_before is False

    # Vérification initiale : la table de dimension dim_high_cardinality n'existe pas encore
    tables_before = [
        row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
    ]
    assert "dim_high_cardinality" not in tables_before

    # Suppression de la ligne id=5 (seul porteur de 'val_104') :
    # après suppression, high_cardinality n'aura plus que 4 valeurs uniques (val_100..val_103)
    # ce qui est ≤ seuil=4 → conversion en variable catégorielle déclenchée par le nettoyage
    # automatique (_detect_new_categorical_after_deletion via _cleanup_orphaned_data_comprehensive).
    deleted_count = deleter.delete_rows(
        filters=[("id", "=", 5)],
        use_transaction=False,
    )
    assert deleted_count == 1

    # Vérification : high_cardinality est désormais catégorielle dans les métadonnées
    is_cat_after = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'high_cardinality'"
    ).fetchone()[0]
    assert is_cat_after is True

    # Vérification : la table de dimension dim_high_cardinality a bien été créée
    tables_after = [
        row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
    ]
    assert "dim_high_cardinality" in tables_after
