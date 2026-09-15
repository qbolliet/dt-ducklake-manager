# Importation des modules
# Modules de base
import os
import warnings
from datetime import datetime
from typing import Any

import duckdb
import narwhals as nw
import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.operations import DatabaseUpdater
from dt_ducklake_manager.schema import DuckLakeTablesBuilder


def _ducklake_available() -> bool:
    """Vérifie si l'extension DuckLake est disponible dans l'environnement de test."""
    try:
        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        conn.close()
        return True
    except Exception:
        return False


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
    assert updater.data_mgr.catalog_alias == "my_lake"
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
    assert updater.data_mgr.catalog_alias == "db"


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
    labels, only the metadata flag should flip — the fact table is never rewritten.

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

    # Vérification : les libellés d'origine sont toujours stockés tels quels
    stored_labels = {
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT DISTINCT high_cardinality FROM fact_table"
        ).fetchall()
    }
    assert stored_labels == {"grp_A", "grp_B", "grp_C"}


# Test de conversion catégorielle → non-catégorielle après ajout de nouvelles modalités
def test_update_database_categorical_becomes_non_categorical(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that a categorical column loses its categorical status when the number of
    distinct values exceeds the threshold after inserting new rows.

    The column 'category' starts with 3 unique values (A, B, C) and is categorical
    (threshold=4). After inserting rows that introduce 2 additional values (D, E),
    the fact table holds 5 modalities which exceeds the threshold, flipping the
    metadata flag to non-categorical.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    # Vérification initiale : category est catégorielle
    is_cat_before = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'category'"
    ).fetchone()[0]
    assert is_cat_before is True

    # Insertion de nouvelles lignes portant 2 modalités inédites pour category (D et E):
    # après insertion, la fact_table comptera [A, B, C, D, E] = 5 modalités > seuil=4
    # → bascule du seul booléen is_categorical attendue.
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

    # Vérification : les libellés d'origine sont conservés, D et E compris
    stored_labels = {
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT DISTINCT category FROM fact_table"
        ).fetchall()
    }
    assert stored_labels == {"A", "B", "C", "D", "E"}


# ---------------------------------------------------------------------------
# Tests des valeurs manquantes dans les colonnes catégorielles lors d'un update
# ---------------------------------------------------------------------------


# Test que les NULL insérés via update_database restent NULL dans fact_table
def test_update_with_null_categorical_preserves_null_in_fact_table(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that NULL values in categorical columns stay NULL after update_database.

    Inserts new rows with a NULL in the 'category' column alongside regular labels,
    and verifies that the NULL is stored as NULL — never as a placeholder — while
    the other rows keep their original labels.
    """
    # DataFrame d'insertion : un NULL en colonne catégorielle + des modalités connues
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

    # La ligne id=31 (category était NULL) doit toujours être NULL dans fact_table
    category_for_31 = built_ducklake_schema.execute(
        "SELECT category FROM fact_table WHERE id = 31"
    ).fetchone()[0]
    assert category_for_31 is None

    # Les lignes non nulles conservent leur libellé d'origine
    category_for_30 = built_ducklake_schema.execute(
        "SELECT category FROM fact_table WHERE id = 30"
    ).fetchone()[0]
    assert category_for_30 == "A"

    # Aucun placeholder ne doit apparaître parmi les modalités stockées
    stored_labels = {
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT DISTINCT category FROM fact_table WHERE category IS NOT NULL"
        ).fetchall()
    }
    assert "-1" not in stored_labels
    assert "nan" not in stored_labels


# Test qu'une colonne au statut forcé n'est jamais rebasculée par un update
def test_update_does_not_reflip_forced_categorical_column(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that a forced categorical column keeps its status across an update.

    'high_cardinality' is forced to categorical even though it exceeds the
    threshold; an update must leave that decision untouched.
    """
    # Forçage du statut catégoriel, au-delà du seuil
    built_ducklake_schema.execute(
        "UPDATE metadata SET is_categorical = TRUE, is_categorical_forced = TRUE"
        " WHERE name = 'high_cardinality'"
    )
    updater._invalidate_metadata_cache()

    expansion_df = pl.DataFrame(
        {
            "id": [50, 51],
            "category": ["A", "B"],
            "value": [1.0, 2.0],
            "date": [datetime(2024, 6, 1), datetime(2024, 6, 2)],
            "status": ["active", "inactive"],
            "high_cardinality": ["val_600", "val_601"],
        }
    )

    assert (
        updater.update_database(
            update_df=expansion_df, keep="first", use_transaction=False
        )
        is True
    )

    # Le statut forcé est conservé malgré le dépassement du seuil
    is_cat_after = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'high_cardinality'"
    ).fetchone()[0]
    assert is_cat_after is True


# Test que dataset_metadata.updated_at avance après un update réussi
def test_update_stamps_dataset_metadata(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that a successful update refreshes dataset_metadata.updated_at."""
    # Horodatage volontairement ancien pour rendre l'avancée observable
    built_ducklake_schema.execute(
        "UPDATE dataset_metadata SET updated_at = TIMESTAMP '2000-01-01 00:00:00'"
    )
    before = built_ducklake_schema.execute(
        "SELECT updated_at FROM dataset_metadata"
    ).fetchone()[0]

    new_rows = pl.DataFrame(
        {
            "id": [60],
            "category": ["A"],
            "value": [1.0],
            "date": [datetime(2024, 7, 1)],
            "status": ["active"],
            "high_cardinality": ["val_700"],
        }
    )

    assert (
        updater.update_database(update_df=new_rows, keep="first", use_transaction=False)
        is True
    )

    after = built_ducklake_schema.execute(
        "SELECT updated_at FROM dataset_metadata"
    ).fetchone()[0]
    assert after > before


# ---------------------------------------------------------------------------
# Tests de préservation des champs d'UI de la table metadata
# ---------------------------------------------------------------------------


# Test que update_database n'écrase jamais les champs d'UI renseignés par le producteur
def test_update_database_preserves_ui_metadata(
    updater: DatabaseUpdater, built_ducklake_schema: Any, update_df: pl.DataFrame
) -> None:
    """Test that a data update leaves the producer-owned UI metadata untouched.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
        update_df: DataFrame with new rows to insert.
    """
    # Renseignement des champs d'UI avant la mise à jour
    updater.update_column_metadata(
        "value",
        unit="€",
        display_format=",.2f",
        family="kpi",
        description="the value",
        default_aggregation="sum",
    )

    # Mise à jour de données (insertion de nouvelles lignes)
    assert (
        updater.update_database(
            update_df=update_df, keep="first", use_transaction=False
        )
        is True
    )

    # Les champs d'UI sont inchangés après l'update
    row = built_ducklake_schema.execute(
        "SELECT unit, display_format, family, description, default_aggregation"
        " FROM metadata WHERE name = 'value'"
    ).fetchone()
    assert row == ("€", ",.2f", "kpi", "the value", "SUM")


# ---------------------------------------------------------------------------
# Tests de update_database() et des colonnes inconnues (§4.2, allow_new_columns)
# ---------------------------------------------------------------------------


# Test qu'une colonne inconnue sans allow_new_columns lève une ValueError
def test_update_database_unknown_column_without_allow_raises(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that an unknown DataFrame column is refused by default.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
    """
    df = pl.DataFrame({"id": [10], "category": ["A"], "new_col": ["x"]})

    with pytest.raises(ValueError, match="new_col"):
        updater.update_database(update_df=df, keep="first", use_transaction=False)

    # La colonne n'a pas été ajoutée
    columns = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "new_col" not in columns


# Test qu'avec allow_new_columns=True la colonne est ajoutée avec ses métadonnées
def test_update_database_allow_new_columns_adds_column_and_metadata(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that allow_new_columns=True adds the column and a metadata row.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
    """
    df = pl.DataFrame({"id": [10], "category": ["A"], "score": [1.5]})

    result = updater.update_database(
        update_df=df,
        keep="first",
        use_transaction=False,
        allow_new_columns=True,
        column_metadata={"score": {"unit": "%", "default_aggregation": "avg"}},
    )
    assert result is True

    # La colonne existe désormais dans la fact table, avec la bonne valeur
    row = built_ducklake_schema.execute(
        "SELECT score FROM fact_table WHERE id = 10"
    ).fetchone()
    assert row == (1.5,)

    # La ligne metadata a été créée, is_primary_key=FALSE, champs d'UI appliqués
    meta = built_ducklake_schema.execute(
        "SELECT sql_type, is_primary_key, unit, default_aggregation"
        " FROM metadata WHERE name = 'score'"
    ).fetchone()
    assert meta == ("DOUBLE", False, "%", "AVG")


# Test que column_metadata mal formé lève une ValueError avant tout ajout
def test_update_database_allow_new_columns_invalid_metadata_raises(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that an unknown column_metadata key raises ValueError.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
    """
    df = pl.DataFrame({"id": [10], "category": ["A"], "score": [1.5]})

    with pytest.raises(ValueError, match="Unknown column_metadata key"):
        updater.update_database(
            update_df=df,
            keep="first",
            use_transaction=False,
            allow_new_columns=True,
            column_metadata={"score": {"not_a_field": "x"}},
        )


# ---------------------------------------------------------------------------
# Test de bout en bout de la compaction DuckLake après update (§5.4-5.5)
# ---------------------------------------------------------------------------


# Test que update_database réussit avec compaction réelle sur un catalogue sur disque
@pytest.mark.skipif(
    not _ducklake_available(),
    reason="Extension ducklake non disponible dans cet environnement",
)
def test_update_database_compacts_on_real_ducklake_catalog(tmp_path: Any) -> None:
    """Test that update_database succeeds end-to-end against a real DuckLake catalog.

    The in-memory ``built_ducklake_schema`` fixture used elsewhere in this file
    can't exercise ``_run_ducklake_compaction`` for real: DuckLake table functions
    need an actually attached catalog. This test attaches a real one and checks
    that ``update_database`` (with ``compact_after_update=True``, the default)
    still returns True and the new rows land — i.e. the ``DuckLakeMaintenance``
    wiring in ``_run_ducklake_compaction`` doesn't break the write path.

    Args:
        tmp_path: pytest temporary directory.
    """
    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(catalog, data_dir, data_inlining_row_limit=0).connect()

    df = pl.DataFrame(
        {
            "id": list(range(1, 6)),
            "category": ["A", "B", "A", "C", "B"],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=["id"], connection=conn
        ).build_schema()

    updater = DatabaseUpdater(connection=conn, categorical_threshold=4)
    update_df = pl.DataFrame(
        {"id": [10, 11], "category": ["A", "C"], "value": [1.1, 2.2]}
    )

    assert (
        updater.update_database(
            update_df=update_df, keep="first", use_transaction=False
        )
        is True
    )

    row_count = conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert row_count == 7
    conn.close()


# ---------------------------------------------------------------------------
# Tests de add_columns() (§4.3)
# ---------------------------------------------------------------------------


# Test qu'add_columns refuse une base sans clé primaire
def test_add_columns_no_primary_key_raises() -> None:
    """Test that add_columns refuses a fact_table without a primary key."""
    conn = duckdb.connect(":memory:")
    df = pl.DataFrame({"id": [1, 2], "value": [0.1, 0.2]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=[], connection=conn
        ).build_schema()

    updater = DatabaseUpdater(connection=conn, categorical_threshold=4)
    with pytest.raises(ValueError, match="No primary key"):
        updater.add_columns(pl.DataFrame({"id": [1], "score": [1.0]}))


# Test qu'add_columns refuse un DataFrame sans clé primaire
def test_add_columns_missing_primary_key_raises(updater: DatabaseUpdater) -> None:
    """Test that add_columns refuses a df missing the primary key column.

    Args:
        updater: DatabaseUpdater fixture (primary key: 'id').
    """
    df = pl.DataFrame({"score": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing primary key"):
        updater.add_columns(df)


# Test qu'add_columns refuse un DataFrame non-unique sur les clés primaires
def test_add_columns_duplicate_keys_raises(updater: DatabaseUpdater) -> None:
    """Test that add_columns refuses a df with duplicate primary key values.

    Args:
        updater: DatabaseUpdater fixture.
    """
    df = pl.DataFrame({"id": [1, 1], "score": [1.0, 2.0]})
    with pytest.raises(ValueError, match="unique on primary key"):
        updater.add_columns(df)


# Test qu'add_columns refuse un DataFrame ne portant que des clés primaires
def test_add_columns_no_value_column_raises(updater: DatabaseUpdater) -> None:
    """Test that add_columns refuses a df carrying only primary key columns.

    Args:
        updater: DatabaseUpdater fixture.
    """
    df = pl.DataFrame({"id": [1, 2]})
    with pytest.raises(ValueError, match="no value column"):
        updater.add_columns(df)


# Test qu'add_columns ajoute une nouvelle colonne avec ses valeurs et ses métadonnées
def test_add_columns_adds_new_column(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that add_columns adds a new column, sets values, and creates metadata.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5).
    """
    df = pl.DataFrame({"id": [1, 2, 3], "score": [10.0, 20.0, 30.0]})

    result = updater.add_columns(df, column_metadata={"score": {"unit": "pts"}})
    assert result is True

    rows = built_ducklake_schema.execute(
        "SELECT id, score FROM fact_table ORDER BY id"
    ).fetchall()
    assert rows == [(1, 10.0), (2, 20.0), (3, 30.0), (4, None), (5, None)]

    meta = built_ducklake_schema.execute(
        "SELECT sql_type, is_primary_key, unit FROM metadata WHERE name = 'score'"
    ).fetchone()
    assert meta == ("DOUBLE", False, "pts")


# Test qu'add_columns refuse une colonne déjà existante sans overwrite
def test_add_columns_existing_column_without_overwrite_raises(
    updater: DatabaseUpdater,
) -> None:
    """Test that add_columns refuses an already-existing column by default.

    Args:
        updater: DatabaseUpdater fixture ('value' already exists).
    """
    df = pl.DataFrame({"id": [1], "value": [99.0]})
    with pytest.raises(ValueError, match="already exist"):
        updater.add_columns(df)

    # La valeur n'a pas été modifiée
    row = updater.conn.execute("SELECT value FROM fact_table WHERE id = 1").fetchone()
    assert row[0] != 99.0


# Test qu'add_columns avec overwrite=True met à jour les valeurs existantes
def test_add_columns_existing_column_with_overwrite_updates_values(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that overwrite=True lets add_columns replace existing values.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
    """
    df = pl.DataFrame({"id": [1], "value": [99.0]})
    result = updater.add_columns(df, overwrite=True)
    assert result is True

    row = built_ducklake_schema.execute(
        "SELECT value FROM fact_table WHERE id = 1"
    ).fetchone()
    assert row[0] == 99.0


# Test que les combinaisons de df sans correspondance en base ne sont pas insérées
def test_add_columns_unmatched_combination_not_inserted(
    updater: DatabaseUpdater, built_ducklake_schema: Any, caplog: Any
) -> None:
    """Test that a df key combination absent from fact_table is skipped, not inserted.

    Args:
        updater: DatabaseUpdater fixture (ids 1..5).
        built_ducklake_schema: DuckDB connection.
        caplog: pytest fixture capturing log records.
    """
    initial_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]

    df = pl.DataFrame({"id": [1, 999], "score": [10.0, 20.0]})
    result = updater.add_columns(df)
    assert result is True

    final_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert final_count == initial_count

    assert any("no match in fact_table" in record.message for record in caplog.records)


# Test que les lignes de la base sans correspondance dans df restent NULL
def test_add_columns_rows_without_match_left_null(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that fact_table rows absent from df keep NULL in the new column.

    Args:
        updater: DatabaseUpdater fixture (ids 1..5).
        built_ducklake_schema: DuckDB connection.
    """
    df = pl.DataFrame({"id": [1], "score": [10.0]})
    updater.add_columns(df)

    null_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE score IS NULL"
    ).fetchone()[0]
    assert null_count == 4


# Test qu'un échec en cours d'opération ne laisse subsister ni colonne ni métadonnée
def test_add_columns_failure_mid_operation_leaves_no_trace(
    updater: DatabaseUpdater, built_ducklake_schema: Any, monkeypatch: Any
) -> None:
    """Test that a failure during add_columns rolls back the column and metadata.

    ``DuckDBPyConnection.execute`` is a read-only C-extension attribute and can't
    be monkeypatched directly, so the failure is forced on
    ``_touch_dataset_metadata`` instead: it runs last, inside the same
    transaction, right after the ``ALTER TABLE``, the metadata row insert and the
    ``UPDATE ... FROM`` have all succeeded. If the single transaction is real, none
    of them survive the rollback.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection.
        monkeypatch: pytest fixture for patching.
    """

    def failing_touch() -> None:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(updater, "_touch_dataset_metadata", failing_touch)

    df = pl.DataFrame({"id": [1], "score": [10.0]})
    with pytest.raises(RuntimeError, match="simulated failure"):
        updater.add_columns(df)

    columns_after = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "score" not in columns_after
    meta_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM metadata WHERE name = 'score'"
    ).fetchone()[0]
    assert meta_count == 0


# ---------------------------------------------------------------------------
# Tests de get_key_combinations() (§4.3, recette de diffusion explicite)
# ---------------------------------------------------------------------------


# Test que get_key_combinations retourne les clés primaires par défaut
def test_get_key_combinations_default_primary_keys(
    updater: DatabaseUpdater,
) -> None:
    """Test that get_key_combinations defaults to all primary key columns.

    Args:
        updater: DatabaseUpdater fixture (ids 1..5, primary key 'id').
    """
    combos = updater.get_key_combinations()
    assert combos.columns == ["id"]
    assert sorted(combos["id"].to_list()) == [1, 2, 3, 4, 5]


# Test que get_key_combinations projette les colonnes explicitement demandées
def test_get_key_combinations_explicit_columns(updater: DatabaseUpdater) -> None:
    """Test that get_key_combinations projects the requested columns.

    Args:
        updater: DatabaseUpdater fixture.
    """
    combos = updater.get_key_combinations(["category"])
    assert combos.columns == ["category"]
    assert set(combos["category"].to_list()) == {"A", "B", "C"}


# Test que get_key_combinations refuse une colonne absente de la fact table
def test_get_key_combinations_unknown_column_raises(
    updater: DatabaseUpdater,
) -> None:
    """Test that get_key_combinations raises ValueError for an unknown column.

    Args:
        updater: DatabaseUpdater fixture.
    """
    with pytest.raises(ValueError, match="Unknown column"):
        updater.get_key_combinations(["not_a_column"])


# Test de la recette de diffusion explicite (§4.3 : jointure puis add_columns)
def test_get_key_combinations_explicit_broadcast_recipe(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test the explicit broadcast recipe: join partial-key values onto full keys.

    A value carried by a partial key (here 'category') is not spread onto the
    full key ('id') automatically; the user must join it explicitly before
    calling ``add_columns``.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5, categories A/B/A/C/B).
    """
    keys = updater.get_key_combinations(["id", "category"])
    df_partial = pl.DataFrame({"category": ["A", "B", "C"], "bonus": [1, 2, 3]})

    # Jointure explicite puis retrait de la colonne de clé partielle : seule 'id'
    # (clé primaire de la fact table) doit rester à côté de la valeur diffusée.
    broadcast_df = (
        nw.to_native(keys).join(df_partial, on="category", how="inner").drop("category")
    )

    result = updater.add_columns(nw.from_native(broadcast_df, eager_only=True))
    assert result is True

    rows = dict(
        built_ducklake_schema.execute(
            "SELECT id, bonus FROM fact_table ORDER BY id"
        ).fetchall()
    )
    # category : id1=A, id2=B, id3=A, id4=C, id5=B
    assert rows == {1: 1, 2: 2, 3: 1, 4: 3, 5: 2}


# ---------------------------------------------------------------------------
# Test de bout en bout d'add_columns sur un catalogue DuckLake réel (§4.3, §5.1)
# ---------------------------------------------------------------------------


# Test qu'add_columns réussit avec compaction réelle sur un catalogue sur disque
@pytest.mark.skipif(
    not _ducklake_available(),
    reason="Extension ducklake non disponible dans cet environnement",
)
def test_add_columns_on_real_ducklake_catalog(tmp_path: Any) -> None:
    """Test that add_columns succeeds end-to-end against a real DuckLake catalog.

    Args:
        tmp_path: pytest temporary directory.
    """
    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(catalog, data_dir, data_inlining_row_limit=0).connect()

    df = pl.DataFrame(
        {
            "id": list(range(1, 6)),
            "category": ["A", "B", "A", "C", "B"],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=["id"], connection=conn
        ).build_schema()

    updater = DatabaseUpdater(connection=conn, categorical_threshold=4)
    score_df = pl.DataFrame({"id": [1, 2, 3], "score": [10.0, 20.0, 30.0]})

    assert updater.add_columns(score_df) is True

    row_count = conn.execute(
        "SELECT COUNT(*) FROM fact_table WHERE score IS NOT NULL"
    ).fetchone()[0]
    assert row_count == 3
    conn.close()


# ---------------------------------------------------------------------------
# Tests des chemins transactionnels (§7) : un BEGIN/COMMIT par opération
# ---------------------------------------------------------------------------


# Fonction auxiliaire de capture de l'état complet de la base
def _snapshot_state(conn: Any) -> dict[str, Any]:
    """Capture the full state of the schema, for before/after comparison.

    Args:
        conn: DuckDB connection holding a built schema.

    Returns:
        dict: row count, sorted fact table content, metadata content and the
        ``dataset_metadata`` timestamp.
    """
    return {
        "count": conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0],
        "facts": conn.execute("SELECT * FROM fact_table ORDER BY id").pl().to_dicts(),
        "metadata": conn.execute("SELECT * FROM metadata ORDER BY name")
        .pl()
        .to_dicts(),
        "updated_at": conn.execute(
            "SELECT updated_at FROM dataset_metadata"
        ).fetchone(),
    }


# Test qu'un échec de la mise à jour de la table des faits annule tout l'update
def test_update_rolls_back_on_fact_table_failure(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that a failed fact table step leaves the database untouched.

    The fact table update is the 4th of six steps: the metadata update that
    precedes it must be rolled back too.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    before = _snapshot_state(updater.conn)

    # Échec simulé de l'étape de mise à jour de la table des faits
    updater._update_fact_table_direct = lambda df: False  # type: ignore[method-assign]

    assert updater.update_database(update_df, keep="first") is False

    # La base est strictement identique à son état initial
    assert _snapshot_state(updater.conn) == before


# Test qu'une exception en milieu d'update annule également tout l'update
def test_update_rolls_back_on_exception(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that an exception raised mid-update restores the initial state.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    before = _snapshot_state(updater.conn)

    # Exception simulée au sein de l'étape de mise à jour de la table des faits
    def _boom(df: Any) -> bool:
        raise RuntimeError("disque plein")

    updater._update_fact_table_direct = _boom  # type: ignore[method-assign]

    assert updater.update_database(update_df, keep="first") is False
    assert _snapshot_state(updater.conn) == before


# Test qu'un échec de la dernière étape annule aussi l'upsert des faits
def test_update_rolls_back_on_last_step_failure(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that a failure on the last step also rolls the fact upsert back.

    ``_update_categorical_flags`` runs after the rows have been written: its
    failure must undo them.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    before = _snapshot_state(updater.conn)

    updater._update_categorical_flags = lambda: False  # type: ignore[method-assign]

    assert updater.update_database(update_df, keep="first") is False
    assert _snapshot_state(updater.conn) == before


# Test que des problèmes critiques post-update déclenchent l'annulation
def test_update_rolls_back_on_critical_validation_issues(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that critical post-update validation issues roll the update back.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    before = _snapshot_state(updater.conn)

    # Rapport de validation simulant deux problèmes critiques
    class _CriticalReport:
        def get_critical_issues_count(self) -> int:
            return 2

        def get_issues_by_severity(self, severity: Any) -> list[Any]:
            return []

    assert updater.auditor is not None
    updater.auditor.validate_database = (  # type: ignore[method-assign]
        lambda level=None: _CriticalReport()
    )

    assert updater.update_database(update_df, keep="first") is False
    assert _snapshot_state(updater.conn) == before


# Test que sans transaction, l'état partiel d'un update échoué subsiste
def test_update_without_transaction_keeps_partial_state(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that use_transaction=False leaves partial state behind on failure.

    Documents the real difference between the two modes: in autocommit, what was
    written before the failing step survives.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    # Doublons présents en base : leur suppression précède l'étape des faits
    updater.conn.execute(
        "INSERT INTO fact_table (id, category, value, date, status,"
        " high_cardinality) SELECT id, category, value, date, status,"
        " high_cardinality FROM fact_table WHERE id = 1"
    )
    count_with_duplicate = updater.conn.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]

    updater._update_fact_table_direct = lambda d: False  # type: ignore[method-assign]

    assert (
        updater.update_database(update_df, keep="first", use_transaction=False) is False
    )

    # La déduplication, elle, a bien persisté : l'état est partiel
    count_after = updater.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count_after < count_with_duplicate


# Test que les deux modes produisent le même état final en cas de succès
def test_update_transaction_and_direct_agree_on_success(
    built_ducklake_schema: Any, sample_df: pl.DataFrame, update_df: pl.DataFrame
) -> None:
    """Test that a successful update yields the same state in both modes.

    Args:
        built_ducklake_schema: Fixture providing a built schema.
        sample_df: DataFrame the schema was built from, reused to rebuild it.
        update_df: DataFrame with two new rows.
    """
    # Mise à jour transactionnelle
    updater = DatabaseUpdater(connection=built_ducklake_schema, categorical_threshold=4)
    assert updater.update_database(update_df, keep="first") is True
    transactional = _snapshot_state(updater.conn)

    # Même mise à jour, sans transaction, sur une base reconstruite à l'identique
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df, categorical_threshold=4, primary_keys=["id"]
        )
    builder.build_schema()
    direct_updater = DatabaseUpdater(connection=builder.conn, categorical_threshold=4)
    assert (
        direct_updater.update_database(update_df, keep="first", use_transaction=False)
        is True
    )
    direct = _snapshot_state(direct_updater.conn)

    # Les contenus coïncident (l'horodatage, lui, diffère par construction)
    assert transactional["count"] == direct["count"]
    assert transactional["facts"] == direct["facts"]
    assert transactional["metadata"] == direct["metadata"]


# Test qu'un échec d'add_columns ne laisse ni la colonne ni ses métadonnées
def test_add_columns_rolls_back_column_and_metadata(updater: DatabaseUpdater) -> None:
    """Test that a failed add_columns leaves neither column nor metadata row.

    Args:
        updater: DatabaseUpdater fixture.
    """
    before = _snapshot_state(updater.conn)

    # Échec simulé du rafraîchissement du statut catégoriel, après l'UPDATE
    def _boom() -> list[str]:
        raise RuntimeError("échec après écriture")

    updater._refresh_categorical_flags = _boom  # type: ignore[method-assign]

    score_df = pl.DataFrame({"id": [1, 2, 3], "score": [10.0, 20.0, 30.0]})
    with pytest.raises(RuntimeError, match="échec après écriture"):
        updater.add_columns(score_df)

    # Ni la colonne, ni sa ligne de métadonnées, ni les valeurs ne subsistent
    assert "score" not in updater._get_fact_table_columns()
    assert _snapshot_state(updater.conn) == before
