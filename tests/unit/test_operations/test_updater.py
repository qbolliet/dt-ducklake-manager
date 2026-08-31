# Importation des modules
# Modules de base
from datetime import datetime
from typing import Any

import polars as pl

# Module de tests
# Modules du package à tester
from dt_ducklake_manager.operations import DatabaseUpdater

# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------


# Test de l'initialisation correcte de DatabaseUpdater
def test_updater_initialization(built_ducklake_schema: Any) -> None:
    """Test that DatabaseUpdater initializes without errors.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, categorical_threshold=4)
    assert updater is not None
    assert updater.categorical_threshold == 4
    assert updater.batch_size > 0


# Test de l'initialisation avec enable_validation=False
def test_updater_initialization_without_validation(built_ducklake_schema: Any) -> None:
    """Test that DatabaseUpdater can be initialized with validation disabled.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, enable_validation=False)
    assert updater.auditor is None


# Test que catalog_alias est propagé à tous les sous-gestionnaires
def test_updater_propagates_catalog_alias(built_ducklake_schema: Any) -> None:
    """Test that ``catalog_alias`` reaches every specialized sub-manager.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    updater = DatabaseUpdater(
        connection=built_ducklake_schema,
        catalog_alias="my_lake",
        schema="predictions",
    )
    assert updater.catalog_alias == "my_lake"
    assert updater.dimension_mgr.catalog_alias == "my_lake"
    assert updater.data_mgr.catalog_alias == "my_lake"
    assert updater.transaction_mgr.catalog_alias == "my_lake"
    assert updater.auditor is not None
    assert updater.auditor.catalog_alias == "my_lake"


# Test que catalog_alias vaut 'db' par défaut
def test_updater_default_catalog_alias(built_ducklake_schema: Any) -> None:
    """Test that ``catalog_alias`` defaults to 'db'.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema)
    assert updater.catalog_alias == "db"
    assert updater.transaction_mgr.catalog_alias == "db"


# ---------------------------------------------------------------------------
# Tests de validate_operation()
# ---------------------------------------------------------------------------


# Test que validate_operation retourne True pour une insertion valide
def test_validate_operation_insert_returns_bool(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that validate_operation returns a boolean for an insert operation.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with new rows.
    """
    result = updater.validate_operation("insert", df=update_df)
    assert isinstance(result, bool)


# Test que validate_operation retourne True quand la validation est désactivée
def test_validate_operation_disabled_returns_true(
    built_ducklake_schema: Any, update_df: pl.DataFrame
) -> None:
    """Test that validate_operation always returns True when validation is disabled.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
        update_df: DataFrame with new rows.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, enable_validation=False)
    result = updater.validate_operation("insert", df=update_df)
    assert result is True


# ---------------------------------------------------------------------------
# Tests de update_database()
# ---------------------------------------------------------------------------


# Test d'insertion de nouvelles lignes sans transaction
def test_update_database_insert_new_rows(
    updater: DatabaseUpdater, built_ducklake_schema: Any, update_df: pl.DataFrame
) -> None:
    """Test that update_database inserts new rows into the fact table.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
        update_df: DataFrame with new rows to insert.
    """
    # Comptage initial
    initial_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]

    # Insertion des nouvelles lignes sans transaction
    # Remarque : keep='first' est requis car narwhals ne supporte pas keep=False (valeur
    # par défaut)
    result = updater.update_database(
        update_df=update_df,
        keep="first",
        use_transaction=False,
    )

    # Vérification que l'opération s'est bien déroulée
    assert isinstance(result, bool)
    assert result is True

    # Vérification que le nombre de lignes a augmenté
    final_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert final_count > initial_count


# Test d'insertion avec déduplication sur les doublons du DataFrame d'entrée
def test_update_database_with_dedup_on_update(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that update_database removes duplicates from the update
    DataFrame when requested.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # DataFrame avec deux lignes IDENTIQUES sur toutes les colonnes (id=20, même date)
    # La déduplication utilise toutes les colonnes (primary_keys non transmises au
    # niveau du
    # DataFrame d'entrée) : les deux lignes doivent donc être parfaitement identiques.
    df_with_dup = pl.DataFrame(
        {
            "id": [20, 20, 21],
            "category": ["A", "A", "B"],
            "value": [5.0, 5.0, 6.0],
            "date": [datetime(2024, 3, 1), datetime(2024, 3, 1), datetime(2024, 3, 2)],
            "status": ["active", "active", "inactive"],
            "high_cardinality": ["val_300", "val_300", "val_301"],
        }
    )

    result = updater.update_database(
        update_df=df_with_dup,
        check_duplicates_update=True,
        check_duplicates_db=False,
        keep="first",
        use_transaction=False,
    )

    assert result is True

    # Vérification que la déduplication a fonctionné : id=20 ne doit apparaître qu'une
    # seule fois
    count_20 = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 20"
    ).fetchone()[0]
    assert count_20 == 1

    # Vérification que id=21 a bien été inséré
    count_21 = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 21"
    ).fetchone()[0]
    assert count_21 == 1


# ---------------------------------------------------------------------------
# Tests de changement de statut catégoriel lors d'une mise à jour
# ---------------------------------------------------------------------------


# Test de conversion non-catégorielle → catégorielle après remplacement de lignes
def test_update_database_non_categorical_becomes_categorical(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that a non-categorical column becomes categorical when its
    unique value count drops to or below the threshold after an update.

    The sample schema is built with categorical_threshold=4.
    The column 'high_cardinality'
    initially has 5 unique values (val_100..val_104) and is therefore NOT categorical.
    After replacing all existing rows (id=1..5) with values from a set of only 3 unique
    labels, the column should be converted to categorical and its dimension table
    created.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    # Vérification initiale :
    # high_cardinality n'est pas catégorielle (5 valeurs > seuil=4)
    is_cat_before = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'high_cardinality'"
    ).fetchone()[0]
    assert is_cat_before is False

    # Vérification initiale :
    # la table de dimension dim_high_cardinality n'existe pas encore
    tables_before = [
        row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
    ]
    assert "dim_high_cardinality" not in tables_before

    # Remplacement de toutes les lignes existantes (id=1..5) via upsert :
    # les 5 nouvelles valeurs de high_cardinality
    # n'appartiennent qu'à 3 modalités distinctes :
    # (grp_A, grp_B, grp_C), ce qui est ≤ seuil=4
    # → conversion en variable catégorielle attendue.
    replacing_df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category": ["A", "B", "A", "C", "B"],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
            "date": pl.date_range(
                datetime(2024, 1, 1), datetime(2024, 1, 5), "1d", eager=True
            ),
            "status": ["active", "inactive", "active", "active", "inactive"],
            "high_cardinality": ["grp_A", "grp_B", "grp_C", "grp_A", "grp_B"],
        }
    )

    result = updater.update_database(
        update_df=replacing_df,
        keep="first",
        use_transaction=False,
    )

    assert result is True

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


# Test de conversion catégorielle → non-catégorielle après ajout de nouvelles modalités
def test_update_database_categorical_becomes_non_categorical(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that a categorical column loses its categorical status when the number of
    distinct values exceeds the threshold after inserting new rows.

    The column 'category' starts with 3 unique values (A, B, C) and is categorical
    (threshold=4). After inserting rows that introduce 2 additional values (D, E),
    the dimension table holds 5 entries which exceeds the threshold, triggering
    conversion to non-categorical and deletion of dim_category.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    # Vérification initiale : category est catégorielle et dim_category existe
    is_cat_before = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'category'"
    ).fetchone()[0]
    assert is_cat_before is True

    tables_before = [
        row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
    ]
    assert "dim_category" in tables_before

    # Insertion de nouvelles lignes portant 2 modalités inédites pour category (D et E):
    # après insertion, la table de dimension contiendra
    # [A, B, C, D, E] = 5 entrées > seuil=4
    # → conversion vers non-catégorielle et suppression de dim_category attendues.
    expansion_df = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "category": ["D", "E", "D"],
            "value": [1.0, 2.0, 3.0],
            "date": [datetime(2024, 3, 1), datetime(2024, 3, 2), datetime(2024, 3, 3)],
            "status": ["active", "inactive", "active"],
            "high_cardinality": ["val_200", "val_201", "val_202"],
        }
    )

    result = updater.update_database(
        update_df=expansion_df,
        keep="first",
        use_transaction=False,
    )

    assert result is True

    # Vérification : category n'est plus catégorielle dans les métadonnées
    is_cat_after = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'category'"
    ).fetchone()[0]
    assert is_cat_after is False

    # Vérification : la table de dimension dim_category a bien été supprimée
    tables_after = [
        row[0] for row in built_ducklake_schema.execute("SHOW TABLES").fetchall()
    ]
    assert "dim_category" not in tables_after


# ---------------------------------------------------------------------------
# Tests des valeurs manquantes dans les colonnes catégorielles lors d'un update
# ---------------------------------------------------------------------------


# Test que les NULL insérés via update_database restent NULL dans fact_table
# et n'introduisent ni ligne label=None ni placeholder "-1" dans dim_*
def test_update_with_null_categorical_preserves_null_in_fact_table(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that NULL values in categorical columns stay NULL after update_database.

    Inserts new rows with NULL in the 'category' column and a brand-new modality.
    Verifies that:
      - dim_category contains the new modality but NO 'None' / 'NaN' / '-1' label
      - fact_table has NULL (not '-1') for the rows whose category was NULL
    """
    # DataFrame d'insertion : un NULL en colonne catégorielle + une modalité connue
    df_with_nulls = pl.DataFrame(
        {
            "id": [30, 31, 32],
            "category": ["A", None, "B"],
            "value": [1.0, 2.0, 3.0],
            "date": [datetime(2024, 4, 1), datetime(2024, 4, 2), datetime(2024, 4, 3)],
            "status": ["active", "inactive", "active"],
            "high_cardinality": ["val_400", "val_401", "val_402"],
        }
    )

    result = updater.update_database(
        update_df=df_with_nulls,
        keep="first",
        use_transaction=False,
    )
    assert result is True

    # dim_category ne contient ni None, ni "nan", ni "-1" comme label
    dim_labels = [
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT label FROM dim_category"
        ).fetchall()
    ]
    assert None not in dim_labels
    assert "nan" not in dim_labels
    assert "-1" not in dim_labels

    # dim_category ne contient pas non plus "-1" en value (placeholder bannis)
    dim_values = [
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT value FROM dim_category"
        ).fetchall()
    ]
    assert "-1" not in dim_values

    # La ligne id=31 (category était NULL) doit toujours être NULL dans fact_table
    category_for_31 = built_ducklake_schema.execute(
        "SELECT category FROM fact_table WHERE id = 31"
    ).fetchone()[0]
    assert category_for_31 is None

    # Comptage global : il doit exister au moins une ligne avec category IS NULL
    null_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE category IS NULL"
    ).fetchone()[0]
    assert null_count >= 1


# Test qu'un update entièrement NULL sur une colonne catégorielle n'ajoute aucune
# ligne à dim_*
def test_update_only_null_categorical_does_not_pollute_dimension(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that an update with only NULL categorical values adds no dim_* entry.

    Verifies that dim_category is unchanged after inserting rows whose categorical
    column is entirely NULL.
    """
    # Snapshot de dim_category avant l'update
    labels_before = sorted(
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT label FROM dim_category"
        ).fetchall()
    )

    df_all_null_cat = pl.DataFrame(
        {
            "id": [40, 41],
            "category": [None, None],
            "value": [9.0, 9.5],
            "date": [datetime(2024, 5, 1), datetime(2024, 5, 2)],
            "status": ["active", "inactive"],
            "high_cardinality": ["val_500", "val_501"],
        }
    )

    result = updater.update_database(
        update_df=df_all_null_cat,
        keep="first",
        use_transaction=False,
    )
    assert result is True

    # dim_category doit être strictement inchangée
    labels_after = sorted(
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT label FROM dim_category"
        ).fetchall()
    )
    assert labels_before == labels_after

    # Et les rangées insérées doivent être NULL dans fact_table
    null_for_new_ids = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id IN (40, 41) AND category IS NULL"
    ).fetchone()[0]
    assert null_for_new_ids == 2
