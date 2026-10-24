# Importation des modules
# Modules de base
import logging
import os
import warnings
from collections.abc import Generator
from pathlib import Path

import duckdb
import polars as pl

# Module de tests
import pytest

# Modules à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.maintenance import (
    DatabaseAuditor,
    DuckLakeMaintenance,
    ValidationLevel,
)
from dt_ducklake_manager.operations import DatabaseDeleter, DatabaseUpdater
from dt_ducklake_manager.reporting import _table_info
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from tests.utils.ducklake import requires_ducklake

# Marqueur appliqué à l'ensemble du module : un vrai catalogue DuckLake 1.1 est
# indispensable aux schémas imbriqués.
pytestmark = requires_ducklake

# Schémas imbriqués partageant le même dernier niveau, et schéma plat homonyme
SALES = "sales.europe"
HR = "hr.europe"


# ===========================================================================
# Fixtures
# ===========================================================================


# Construction d'un jeu de résultats dans un schéma donné
def _build(conn: duckdb.DuckDBPyConnection, schema: str, df: pl.DataFrame) -> None:
    """Build a result set into ``schema`` of the ``db`` catalog.

    Args:
        conn: Connection with the ``db`` catalog attached.
        schema: Target schema path.
        df: Source DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df,
            categorical_threshold=4,
            primary_keys=["id"],
            connection=conn,
            schema=schema,
        ).build_schema()


# Catalogue portant deux schémas imbriqués de même feuille
@pytest.fixture
def nested_lake(
    sample_df: pl.DataFrame, tmp_path: Path
) -> Generator[duckdb.DuckDBPyConnection]:
    """Provide a DuckLake catalog holding 'sales.europe' and 'hr.europe'.

    Both nested schemas share their leaf name ``europe``: any lookup by leaf name
    would mix them. ``hr.europe`` holds the first two rows of ``sample_df`` only.

    Args:
        sample_df: Sample polars DataFrame (5 rows).
        tmp_path: pytest temporary directory.

    Yields:
        duckdb.DuckDBPyConnection: Connection attached to the catalog.
    """
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(
        str(tmp_path / "test.ducklake"), data_dir, data_inlining_row_limit=0
    ).connect()
    _build(conn, SALES, sample_df)
    _build(conn, HR, sample_df.head(2))
    yield conn
    conn.close()


# Comptage des lignes d'une table d'un schéma imbriqué
def _count(conn: duckdb.DuckDBPyConnection, schema: str) -> int:
    """Count the rows of ``db.<schema>.fact_table``.

    Args:
        conn: Connection with the ``db`` catalog attached.
        schema: Schema path.

    Returns:
        int: Row count.
    """
    row = conn.execute(f"SELECT COUNT(*) FROM db.{schema}.fact_table").fetchone()
    assert row is not None
    return int(row[0])


# ===========================================================================
# Construction
# ===========================================================================


# Test que la construction écrit dans le schéma imbriqué, pas dans un schéma plat
def test_build_writes_into_nested_schema(
    nested_lake: duckdb.DuckDBPyConnection,
) -> None:
    """Test that the three tables land in the nested schemas, with no flat twin.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
    """
    assert _count(nested_lake, SALES) == 5
    assert _count(nested_lake, HR) == 2
    for table in ("metadata", "dataset_metadata"):
        row = nested_lake.execute(f"SELECT COUNT(*) FROM db.{SALES}.{table}").fetchone()
        assert row is not None and row[0] > 0
    # Aucun schéma plat dont le nom contient un point
    names = {
        row[0]
        for row in nested_lake.execute(
            "SELECT schema_name FROM duckdb_schemas() WHERE database_name = 'db'"
        ).fetchall()
    }
    assert not any("." in name for name in names)


# Test qu'un schéma imbriqué sans catalogue attaché est rejeté
def test_build_nested_schema_without_catalog_raises(sample_df: pl.DataFrame) -> None:
    """Test that an in-memory build cannot target a nested schema.

    Without an attached catalog, the first level would be read as a catalog name.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df, categorical_threshold=4, primary_keys=["id"], schema=SALES
        )
    with pytest.raises(ValueError, match="requires an attached catalog"):
        builder.build_schema()


# ===========================================================================
# Isolation des opérations entre schémas de même feuille
# ===========================================================================


# Test qu'une mise à jour d'un schéma imbriqué n'affecte pas son homonyme
def test_update_isolated_between_same_leaf_schemas(
    nested_lake: duckdb.DuckDBPyConnection,
) -> None:
    """Test that updating 'sales.europe' leaves 'hr.europe' untouched.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
    """
    updater = DatabaseUpdater(
        connection=nested_lake, categorical_threshold=4, audit_level=None, schema=SALES
    )
    new_rows = pl.DataFrame({"id": [6], "category": ["B"], "value": [0.9]})
    assert updater.update_database(new_rows, use_transaction=False) is True
    assert _count(nested_lake, SALES) == 6
    assert _count(nested_lake, HR) == 2


# Test qu'une nouvelle colonne n'est ajoutée qu'au schéma imbriqué ciblé
def test_add_column_isolated_between_same_leaf_schemas(
    nested_lake: duckdb.DuckDBPyConnection,
) -> None:
    """Test that a new column reaches 'hr.europe' only, metadata included.

    Exercises the metadata-table existence check (``_table_exists``), which must
    resolve the schema by its full path.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
    """
    updater = DatabaseUpdater(
        connection=nested_lake, categorical_threshold=4, audit_level=None, schema=HR
    )
    new_rows = pl.DataFrame(
        {"id": [1, 2], "category": ["A", "B"], "value": [0.1, 0.2], "extra": [1, 2]}
    )
    assert (
        updater.update_database(new_rows, use_transaction=False, allow_new_columns=True)
        is True
    )
    hr_meta = nested_lake.execute(
        f"SELECT name FROM db.{HR}.metadata WHERE name = 'extra'"
    ).fetchall()
    sales_meta = nested_lake.execute(
        f"SELECT name FROM db.{SALES}.metadata WHERE name = 'extra'"
    ).fetchall()
    assert hr_meta == [("extra",)]
    assert sales_meta == []


# Test qu'une suppression d'un schéma imbriqué n'affecte pas son homonyme
def test_delete_isolated_between_same_leaf_schemas(
    nested_lake: duckdb.DuckDBPyConnection,
) -> None:
    """Test that deleting from 'hr.europe' leaves 'sales.europe' untouched.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
    """
    deleter = DatabaseDeleter(
        connection=nested_lake, audit_level=None, auto_cleanup=False, schema=HR
    )
    deleted = deleter.delete_rows(filters=[("id", "=", 1)], use_transaction=False)
    assert deleted.rows_deleted == 1
    assert _count(nested_lake, HR) == 1
    assert _count(nested_lake, SALES) == 5


# Test que l'audit d'un schéma imbriqué reconnaît ses tables
@pytest.mark.parametrize("schema", [SALES, HR])
def test_audit_nested_schema(
    nested_lake: duckdb.DuckDBPyConnection, schema: str
) -> None:
    """Test that the auditor finds the three tables of the nested schema.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
        schema: Audited schema path.
    """
    report = DatabaseAuditor(connection=nested_lake, schema=schema).validate_database(
        ValidationLevel.BASIC
    )
    assert report.get_critical_issues_count() == 0
    assert {"fact_table", "metadata", "dataset_metadata"} <= report.tables_validated


# Test qu'un schéma plat portant le nom de la feuille n'est pas confondu
def test_audit_flat_leaf_name_sees_no_nested_table(
    nested_lake: duckdb.DuckDBPyConnection,
) -> None:
    """Test that a top-level 'europe' schema does not see the nested tables.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
    """
    nested_lake.execute("CREATE SCHEMA db.europe")
    auditor = DatabaseAuditor(connection=nested_lake, schema="europe")
    assert auditor._table_exists("fact_table") is False
    assert DatabaseAuditor(connection=nested_lake, schema=SALES)._table_exists(
        "fact_table"
    )


# ===========================================================================
# Maintenance et rapports sur schémas imbriqués
# ===========================================================================


# Test que les identifiants DuckLake des tables homonymes sont distincts
def test_table_ids_distinct_between_same_leaf_schemas(
    nested_lake: duckdb.DuckDBPyConnection,
) -> None:
    """Test that each nested schema resolves to its own DuckLake table_id.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
    """
    maintenance = DuckLakeMaintenance(nested_lake)
    sales_id = maintenance._table_id("fact_table", SALES)
    hr_id = maintenance._table_id("fact_table", HR)
    assert sales_id is not None and hr_id is not None
    assert sales_id != hr_id
    assert maintenance._table_id("fact_table", "europe") is None


# Test de la clé de tri lue sur un schéma imbriqué
@pytest.mark.parametrize("schema", [SALES, HR])
def test_sorted_by_nested_schema(
    nested_lake: duckdb.DuckDBPyConnection, schema: str
) -> None:
    """Test that the SORTED BY key set at build time is read back per schema.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
        schema: Schema path.
    """
    maintenance = DuckLakeMaintenance(nested_lake, schema=schema)
    assert maintenance.sorted_by() == ["id"]


# Test du rapport de stockage d'un schéma imbriqué
def test_storage_report_nested_schema(nested_lake: duckdb.DuckDBPyConnection) -> None:
    """Test that the storage report measures the targeted nested table.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
    """
    report = DuckLakeMaintenance(nested_lake).storage_report(schema=SALES)
    assert report.schema == SALES
    assert report.file_count >= 1
    assert report.total_bytes > 0


# Test des statistiques de fichiers par schéma imbriqué
def test_table_info_distinct_between_same_leaf_schemas(
    nested_lake: duckdb.DuckDBPyConnection,
) -> None:
    """Test that _table_info reads the files of the targeted nested table.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
    """
    logger = logging.getLogger("test_nested_schema")
    sales = _table_info(nested_lake, "db", SALES, "fact_table", logger)
    hr = _table_info(nested_lake, "db", HR, "fact_table", logger)
    assert sales is not None and hr is not None
    assert sales[1] > hr[1]
    assert _table_info(nested_lake, "db", "europe", "fact_table", logger) is None


# Test des procédures DuckLake ciblant un schéma imbriqué
def test_maintain_nested_schema(nested_lake: duckdb.DuckDBPyConnection) -> None:
    """Test that a full maintenance pass runs on a nested schema without error.

    Args:
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
    """
    maintenance = DuckLakeMaintenance(nested_lake, schema=SALES)
    maintenance.flush_inlined_data()
    maintenance.merge_files()
    maintenance.rewrite_data_files()
    maintenance.maintain()
    assert _count(nested_lake, SALES) == 5
    assert _count(nested_lake, HR) == 2


# Test de l'affichage d'un schéma imbriqué limité à ses propres tables
def test_display_schema_nested_lists_own_tables(
    sample_df: pl.DataFrame,
    nested_lake: duckdb.DuckDBPyConnection,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that display_schema lists the three tables of the nested schema only.

    Args:
        sample_df: Sample polars DataFrame.
        nested_lake: Catalog with 'sales.europe' and 'hr.europe'.
        caplog: pytest logging capture fixture.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df, categorical_threshold=4, connection=nested_lake, schema=HR
        )
    with caplog.at_level(logging.INFO):
        builder.display_schema()
    # Une seule occurrence par table : les tables homonymes de 'sales.europe'
    # ne sont pas listées
    assert caplog.text.count("fact_table Structure:") == 1
    assert caplog.text.count("dataset_metadata Structure:") == 1
