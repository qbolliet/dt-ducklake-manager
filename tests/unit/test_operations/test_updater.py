# Importation des modules
# Modules de base
import logging
import os
import warnings
from datetime import datetime
from typing import Any

import duckdb
import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.maintenance import IssueSeverity, ValidationLevel
from dt_ducklake_manager.operations import DatabaseUpdater
from dt_ducklake_manager.reporting import OperationReport
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from tests.utils.ducklake import requires_ducklake

# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------


# Test de l'initialisation correcte de DatabaseUpdater
def test_updater_initialization(built_ducklake_schema: Any) -> None:
    """Test that DatabaseUpdater initializes with a BASIC post-write audit.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, categorical_threshold=4)
    assert updater.categorical_threshold == 4
    assert updater.audit_level == ValidationLevel.BASIC
    assert updater.auditor is not None


# Test de la désactivation de l'audit post-écriture
def test_updater_initialization_without_audit(built_ducklake_schema: Any) -> None:
    """Test that the post-write audit can be disabled with audit_level=None.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, audit_level=None)
    assert updater.audit_level is None


# Test que batch_size est accepté mais signalé comme obsolète
def test_updater_batch_size_is_deprecated(built_ducklake_schema: Any) -> None:
    """Test that passing batch_size emits a DeprecationWarning.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    with pytest.warns(DeprecationWarning, match="batch_size"):
        DatabaseUpdater(connection=built_ducklake_schema, batch_size=10)


# Test que l'updater n'expose plus de gestionnaire de données public
def test_updater_has_no_public_data_manager(built_ducklake_schema: Any) -> None:
    """Test that no public sub-manager can add columns behind allow_new_columns.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema)
    assert not hasattr(updater, "data_mgr")


# Test que catalog_alias est propagé à l'auditeur et à la maintenance
def test_updater_propagates_catalog_alias(built_ducklake_schema: Any) -> None:
    """Test that ``catalog_alias`` reaches the auditor and the maintenance helper.

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
    assert updater.auditor is not None
    assert updater.auditor.catalog_alias == "my_lake"
    assert updater.maintenance.catalog_alias == "my_lake"


# Test que catalog_alias vaut 'db' par défaut
def test_updater_default_catalog_alias(built_ducklake_schema: Any) -> None:
    """Test that ``catalog_alias`` defaults to 'db'.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema)
    assert updater.catalog_alias == "db"


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


# Test que l'ajout de modalités ne recalcule pas le statut catégoriel
def test_update_database_keeps_categorical_status(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that an update never re-evaluates is_categorical.

    The column 'category' starts with 3 unique values (A, B, C) and is categorical
    (threshold=4). Inserting rows with 2 additional values (D, E) brings it to 5
    modalities, but the status is inferred once, at creation, and stays True.

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
    # la fact_table comptera 5 modalités > seuil=4, sans effet sur le statut.
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

    # Vérification : category reste catégorielle dans les métadonnées
    is_cat_after = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'category'"
    ).fetchone()[0]
    assert is_cat_after is True
    assert updater.last_report is not None
    assert not any(
        c.startswith("is_categorical") for c in updater.last_report.metadata_changes
    )

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
@requires_ducklake
def test_update_database_compacts_on_real_ducklake_catalog(tmp_path: Any) -> None:
    """Test that update_database succeeds end-to-end against a real DuckLake catalog.

    The in-memory ``built_ducklake_schema`` fixture used elsewhere in this file
    can't exercise ``DuckLakeMaintenance.compact`` for real: DuckLake table functions
    need an actually attached catalog. This test attaches a real one and checks
    that ``update_database`` (with ``compact_after_update=True``, the default)
    still returns True and the new rows land — i.e. the ``DuckLakeMaintenance``
    wiring in ``DuckLakeMaintenance.compact`` doesn't break the write path.

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
    assert isinstance(result, OperationReport)
    assert result.columns_added == ["score"]

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
    assert isinstance(result, OperationReport)
    assert result.rows_updated == 1

    row = built_ducklake_schema.execute(
        "SELECT value FROM fact_table WHERE id = 1"
    ).fetchone()
    assert row[0] == 99.0


# Test que les combinaisons de df absentes de la base sont insérées
def test_add_columns_inserts_new_key_combinations(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that a df key combination absent from fact_table is inserted.

    The inserted row carries the primary key and the new column; every former
    value column is NULL. The matched row gets its value.

    Args:
        updater: DatabaseUpdater fixture (ids 1..5).
        built_ducklake_schema: DuckDB connection.
    """
    df = pl.DataFrame({"id": [1, 999], "score": [10.0, 20.0]})
    report = updater.add_columns(df)
    assert isinstance(report, OperationReport)
    assert report.rows_updated == 1
    assert report.rows_inserted == 1

    assert (
        built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        == 6
    )
    new_row = built_ducklake_schema.execute(
        "SELECT score, category, value, date, status, high_cardinality"
        " FROM fact_table WHERE id = 999"
    ).fetchone()
    assert new_row == (20.0, None, None, None, None, None)
    matched = built_ducklake_schema.execute(
        "SELECT score, category FROM fact_table WHERE id = 1"
    ).fetchone()
    assert matched == (10.0, "A")


# Test d'overwrite combiné à de nouvelles clés
def test_add_columns_overwrite_with_new_keys(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test overwrite=True on an existing column with both known and new keys.

    Known keys absent from df keep their previous value; new keys are inserted.

    Args:
        updater: DatabaseUpdater fixture (ids 1..5).
        built_ducklake_schema: DuckDB connection.
    """
    df = pl.DataFrame({"id": [2, 1000], "value": [9.9, 7.7]})
    report = updater.add_columns(df, overwrite=True)
    assert report.rows_updated == 1
    assert report.rows_inserted == 1

    rows = dict(
        built_ducklake_schema.execute(
            "SELECT id, value FROM fact_table WHERE id IN (1, 2, 1000)"
        ).fetchall()
    )
    assert rows == {1: 0.1, 2: 9.9, 1000: 7.7}


# Test qu'une clé primaire nulle dans df est refusée
def test_add_columns_null_primary_key_raises(updater: DatabaseUpdater) -> None:
    """Test that add_columns refuses a df with a null primary key.

    Args:
        updater: DatabaseUpdater fixture.
    """
    df = pl.DataFrame({"id": [1, None], "score": [10.0, 20.0]})
    with pytest.raises(ValueError, match="null value"):
        updater.add_columns(df)


# Test qu'un échec annule aussi les lignes insérées
def test_add_columns_failure_rolls_back_inserted_rows(
    updater: DatabaseUpdater, built_ducklake_schema: Any, monkeypatch: Any
) -> None:
    """Test that a failure after the insert leaves no inserted row behind.

    Args:
        updater: DatabaseUpdater fixture (ids 1..5).
        built_ducklake_schema: DuckDB connection.
        monkeypatch: pytest fixture for patching.
    """

    def failing_touch() -> None:
        raise RuntimeError("simulated failure")

    monkeypatch.setattr(updater, "_touch_dataset_metadata", failing_touch)

    df = pl.DataFrame({"id": [1, 999], "score": [10.0, 20.0]})
    with pytest.raises(RuntimeError, match="simulated failure"):
        updater.add_columns(df)

    ids = {
        row[0]
        for row in built_ducklake_schema.execute("SELECT id FROM fact_table").fetchall()
    }
    assert ids == {1, 2, 3, 4, 5}
    assert "score" not in updater._get_fact_table_columns()


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
    # Les combinaisons sont renvoyées en narwhals (backend pyarrow) : conversion
    # vers le backend de l'appelant avant la jointure
    broadcast_df = (
        keys.to_polars().join(df_partial, on="category", how="inner").drop("category")
    )

    result = updater.add_columns(broadcast_df)
    assert isinstance(result, OperationReport)

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
@requires_ducklake
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

    result = updater.add_columns(score_df)
    assert isinstance(result, OperationReport)
    assert result.rows_updated == 3

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

    The type widening of the metadata precedes the upsert: it must be rolled back
    too.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    before = _snapshot_state(updater.conn)

    # Échec simulé de l'étape d'écriture de la table des faits
    def _fail(*args: object, **kwargs: object) -> None:
        raise duckdb.IOException("simulated I/O error")

    setattr(updater, "_upsert_fact_table", _fail)

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
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disque plein")

    setattr(updater, "_upsert_fact_table", _boom)

    assert updater.update_database(update_df, keep="first") is False
    assert _snapshot_state(updater.conn) == before


# Test qu'un échec de la dernière étape annule aussi l'upsert des faits
def test_update_rolls_back_on_last_step_failure(
    updater: DatabaseUpdater, update_df: pl.DataFrame
) -> None:
    """Test that a failure on the last step also rolls the fact upsert back.

    The post-write audit runs after the rows have been written: its failure must
    undo them.

    Args:
        updater: DatabaseUpdater fixture.
        update_df: DataFrame with two new rows.
    """
    before = _snapshot_state(updater.conn)

    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("échec de la validation")

    assert updater.auditor is not None
    setattr(updater.auditor, "validate_database", _boom)

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

    # Rapport de validation simulant un problème critique
    class _Issue:
        description = "fact_table is missing"

    class _CriticalReport:
        def get_issues_by_severity(self, severity: Any) -> list[Any]:
            return [_Issue()] if severity == IssueSeverity.CRITICAL else []

    assert updater.auditor is not None
    setattr(updater.auditor, "validate_database", lambda level=None: _CriticalReport())

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

    # Nouvelle colonne ajoutée avant l'étape d'écriture des faits, qui échoue
    def _fail(*args: object, **kwargs: object) -> None:
        raise duckdb.IOException("simulated I/O error")

    setattr(updater, "_upsert_fact_table", _fail)
    with_score = update_df.with_columns(pl.lit(1.0).alias("score"))

    assert (
        updater.update_database(
            with_score, allow_new_columns=True, use_transaction=False
        )
        is False
    )

    # La colonne ajoutée, elle, a bien persisté : l'état est partiel
    assert "score" in updater._get_fact_table_columns()


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

    # Échec simulé de l'horodatage de dataset_metadata, après l'UPDATE
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("échec après écriture")

    updater._touch_dataset_metadata = _boom  # type: ignore[method-assign]

    score_df = pl.DataFrame({"id": [1, 2, 3], "score": [10.0, 20.0, 30.0]})
    with pytest.raises(RuntimeError, match="échec après écriture"):
        updater.add_columns(score_df)

    # Ni la colonne, ni sa ligne de métadonnées, ni les valeurs ne subsistent
    assert "score" not in updater._get_fact_table_columns()
    assert _snapshot_state(updater.conn) == before


# ===========================================================================
# Tests des colonnes de libellés (§2.6) : update_database, update_value_labels,
# add_columns
# ===========================================================================

# Jeu de données : deux lignes de code '01' (label 'Chevaux'), une de code '02'
# (label 'Bovins') — la dépendance nc8 -> nc8_libelle est respectée à la construction.


# Fixture d'une connexion DuckLake avec une paire code/libellé déjà déclarée
@pytest.fixture
def value_label_conn() -> Any:
    """Provide a built schema with a declared code/label pair (nc8 -> nc8_libelle)."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "nc8": ["01", "01", "02"],
            "nc8_libelle": ["Chevaux", "Chevaux", "Bovins"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            df,
            categorical_threshold=10,
            primary_keys=["id"],
            value_labels={"nc8_libelle": "nc8"},
        )
    builder.build_schema()
    return builder.conn


# Fixture d'un DatabaseUpdater construit sur ce schéma
@pytest.fixture
def value_label_updater(value_label_conn: Any) -> DatabaseUpdater:
    """Create a DatabaseUpdater over the code/label schema fixture."""
    return DatabaseUpdater(connection=value_label_conn, categorical_threshold=10)


# Test qu'update_database refuse un nouveau libellé partiel pour un code existant
def test_update_database_refuses_partial_new_label(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that relabeling only some rows of an existing code is refused.

    Only id=1 (code '01') gets a new label; id=2 (also code '01') keeps the old
    one: the batch is rejected and rolled back, pointing to update_value_labels.

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    before = _snapshot_state(value_label_updater.conn)

    df = pl.DataFrame({"id": [1], "nc8_libelle": ["NewLabel"]})
    with pytest.raises(ValueError, match="update_value_labels"):
        value_label_updater.update_database(df)

    after = _snapshot_state(value_label_updater.conn)
    assert after["facts"] == before["facts"]


# Test qu'update_database refuse un lot sans libellé insérant un code déjà libellé
def test_update_database_refuses_new_row_without_label_for_existing_code(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that inserting a code without its label is refused when the code
    already carries a label elsewhere.

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    before = _snapshot_state(value_label_updater.conn)

    df = pl.DataFrame({"id": [4], "nc8": ["01"], "value": [4.0]})
    with pytest.raises(ValueError):
        value_label_updater.update_database(df, allow_new_columns=False)

    after = _snapshot_state(value_label_updater.conn)
    assert after["facts"] == before["facts"]


# Test qu'update_database refuse un changement de code non clé primaire sans son
# libellé
def test_update_database_refuses_code_change_without_label(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that changing a non-primary-key code without its label is refused.

    Moving id=1 into code '02' keeps its stale label 'Chevaux', conflicting with
    the label already carried by '02' ('Bovins'). The error is an input error:
    it is raised (after rollback), not turned into a False return value.

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    before = _snapshot_state(value_label_updater.conn)

    df = pl.DataFrame({"id": [1], "nc8": ["02"]})
    with pytest.raises(ValueError):
        value_label_updater.update_database(df)

    after = _snapshot_state(value_label_updater.conn)
    assert after["facts"] == before["facts"]


# Test qu'update_database accepte une mise à jour respectant la dépendance
def test_update_database_accepts_consistent_relabel(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that relabeling every row of a code together is accepted.

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    df = pl.DataFrame({"id": [1, 2], "nc8_libelle": ["NewChevaux", "NewChevaux"]})
    assert value_label_updater.update_database(df) is True

    rows = value_label_updater.conn.execute(
        "SELECT nc8_libelle FROM fact_table WHERE nc8 = '01' ORDER BY id"
    ).fetchall()
    assert rows == [("NewChevaux",), ("NewChevaux",)]


# Test qu'update_value_labels réécrit les libellés d'un code
def test_update_value_labels_rewrites_rows(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that update_value_labels rewrites every row of the given code(s).

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    new_labels = pl.DataFrame({"nc8": ["01"], "nc8_libelle": ["Chevaux reproducteurs"]})
    report = value_label_updater.update_value_labels("nc8_libelle", new_labels)

    assert isinstance(report, OperationReport)
    assert report.operation == "update_value_labels"
    assert report.rows_updated == 2

    rows = value_label_updater.conn.execute(
        "SELECT nc8_libelle FROM fact_table WHERE nc8 = '01' ORDER BY id"
    ).fetchall()
    assert rows == [("Chevaux reproducteurs",), ("Chevaux reproducteurs",)]
    # Le code '02', non concerné, garde son libellé d'origine
    other = value_label_updater.conn.execute(
        "SELECT nc8_libelle FROM fact_table WHERE nc8 = '02'"
    ).fetchone()
    assert other[0] == "Bovins"


# Test qu'update_value_labels signale, sans les insérer, les codes absents
def test_update_value_labels_warns_on_absent_codes(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that a code absent from the fact table is warned about, never inserted.

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    new_labels = pl.DataFrame(
        {"nc8": ["01", "99"], "nc8_libelle": ["Chevaux", "Inconnu"]}
    )
    report = value_label_updater.update_value_labels("nc8_libelle", new_labels)

    assert any("99" in w for w in report.warnings)
    count = value_label_updater.conn.execute(
        "SELECT COUNT(*) FROM fact_table WHERE nc8 = '99'"
    ).fetchone()[0]
    assert count == 0


# Test qu'update_value_labels refuse un label_column sans label_for déclaré
def test_update_value_labels_requires_label_for(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that update_value_labels refuses a column with no declared label_for.

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    labels = pl.DataFrame({"nc8": ["01"], "value": [99.0]})
    with pytest.raises(ValueError, match="label_for"):
        value_label_updater.update_value_labels("value", labels)


# Test qu'update_value_labels refuse un DataFrame de libellés non unique sur le code
def test_update_value_labels_requires_unique_codes(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that update_value_labels refuses labels not unique on the code column.

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    labels = pl.DataFrame({"nc8": ["01", "01"], "nc8_libelle": ["A", "B"]})
    with pytest.raises(ValueError, match="unique"):
        value_label_updater.update_value_labels("nc8_libelle", labels)


# Test qu'add_columns peut déclarer une nouvelle colonne de libellés
def test_add_columns_declares_new_label_column(
    updater: DatabaseUpdater, built_ducklake_schema: Any
) -> None:
    """Test that add_columns can declare a new column as a label column.

    Every row of ``sample_df`` is covered so the functional dependency
    category -> category_libelle holds across the whole table.

    Args:
        updater: DatabaseUpdater fixture.
        built_ducklake_schema: DuckDB connection (ids 1..5, category A/B/A/C/B).
    """
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category_libelle": ["Cat A", "Cat B", "Cat A", "Cat C", "Cat B"],
        }
    )
    report = updater.add_columns(
        df, column_metadata={"category_libelle": {"label_for": "category"}}
    )
    assert isinstance(report, OperationReport)

    meta = built_ducklake_schema.execute(
        "SELECT label_for FROM metadata WHERE name = 'category_libelle'"
    ).fetchone()
    assert meta[0] == "category"


# Test qu'un code absent est signalé même quand la colonne de code contient NULL
def test_update_value_labels_warns_on_absent_code_with_null_codes() -> None:
    """Test that an absent code is reported although the fact table holds a NULL.

    ``x NOT IN (subquery containing NULL)`` is never true: the detection must not
    depend on the absence of null codes in the fact table.
    """
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "nc8": ["01", None, "02"],
            "nc8_libelle": ["Chevaux", None, "Bovins"],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            df,
            categorical_threshold=10,
            primary_keys=["id"],
            value_labels={"nc8_libelle": "nc8"},
        )
    builder.build_schema()
    updater = DatabaseUpdater(connection=builder.conn, categorical_threshold=10)

    labels = pl.DataFrame({"nc8": ["ZZZ"], "nc8_libelle": ["Inconnu"]})
    report = updater.update_value_labels("nc8_libelle", labels)

    assert any("ZZZ" in w and "1 code(s)" in w for w in report.warnings)
    assert report.rows_updated == 0


# Test qu'un libellé inchangé ne réécrit aucune ligne
def test_update_value_labels_same_label_rewrites_nothing(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that relabeling a code with its current label rewrites no row.

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    labels = pl.DataFrame({"nc8": ["01"], "nc8_libelle": ["Chevaux"]})
    report = value_label_updater.update_value_labels("nc8_libelle", labels)
    assert report.rows_updated == 0
    assert report.warnings == []


# Test qu'un code nul dans le lot de libellés est refusé
def test_update_value_labels_null_code_raises(
    value_label_updater: DatabaseUpdater,
) -> None:
    """Test that a null code in labels raises ValueError before any write.

    Args:
        value_label_updater: DatabaseUpdater over the code/label schema fixture.
    """
    labels = pl.DataFrame({"nc8": [None], "nc8_libelle": ["Rien"]})
    with pytest.raises(ValueError, match="null"):
        value_label_updater.update_value_labels("nc8_libelle", labels)


# Test qu'add_columns ne compte comme mises à jour que les lignes modifiées
def test_add_columns_overwrite_counts_only_changed_rows(
    updater: DatabaseUpdater,
) -> None:
    """Test that overwriting a column with its current values rewrites no row.

    Args:
        updater: DatabaseUpdater fixture.
    """
    values = pl.DataFrame({"id": [1, 2, 3], "value": [0.1, 0.2, 99.0]})
    report = updater.add_columns(values, overwrite=True)
    assert report.rows_updated == 1
    assert report.rows_inserted == 0


# Test qu'add_columns journalise les lignes restées sans valeur
def test_add_columns_logs_rows_left_null(updater: DatabaseUpdater, caplog: Any) -> None:
    """Test that the rows of the base absent from df are counted in the log.

    Args:
        updater: DatabaseUpdater fixture.
        caplog: pytest log capture.
    """
    with caplog.at_level(logging.DEBUG, logger="base_schema_manager"):
        updater.add_columns(pl.DataFrame({"id": [1, 2], "score": [1.0, 2.0]}))
    assert any(
        "3 fact_table row(s) left NULL" in r.getMessage() for r in caplog.records
    )
