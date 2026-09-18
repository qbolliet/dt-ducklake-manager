# Importation des modules
# Modules de base
import os
import warnings
from pathlib import Path

# DuckDB
import duckdb
import polars as pl

# Module de tests
import pytest

# Modules à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.operations import DatabaseDeleter, DatabaseUpdater
from dt_ducklake_manager.schema import DuckLakeTablesBuilder

# ===========================================================================
# Isolation des opérations entre catalogues DuckLake attachés à une connexion
# ===========================================================================


# Fixture fournissant une connexion avec DEUX catalogues DuckLake attachés,
# chacun portant un schéma 'main' complet (fact_table, metadata, dataset_metadata).
@pytest.fixture
def two_catalogs_connection(tmp_path: Path) -> duckdb.DuckDBPyConnection:
    """Provide a connection with two attached DuckLake catalogs, each fully built.

    ``lake_a`` and ``lake_b`` each hold a ``main`` schema built from the same
    3-row DataFrame. ``lake_b`` is the current catalog of the connection (last
    ``USE``), so any manager targeting ``lake_a`` must still write only there.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        duckdb.DuckDBPyConnection: connection with both catalogs attached.
    """
    cat_a = str(tmp_path / "lake_a.ducklake")
    cat_b = str(tmp_path / "lake_b.ducklake")
    data_a = str(tmp_path / "data_a")
    data_b = str(tmp_path / "data_b")
    os.makedirs(data_a)
    os.makedirs(data_b)

    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL ducklake; LOAD ducklake;")

    # Attachement des deux catalogues sur la même connexion
    DuckLakeConnector(cat_a, data_a, catalog_alias="lake_a").attach(conn)
    DuckLakeConnector(cat_b, data_b, catalog_alias="lake_b").attach(
        conn, activate_schema=False
    )

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["A", "B", "A"],
            "value": [0.1, 0.2, 0.3],
        }
    )

    # Construction du schéma dans chaque catalogue
    for alias in ("lake_a", "lake_b"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            DuckLakeTablesBuilder(
                df,
                categorical_threshold=4,
                primary_keys=["id"],
                connection=conn,
                schema="main",
                catalog_alias=alias,
            ).build_schema()

    # Le dernier USE porte sur lake_b : le catalogue courant est le "mauvais"
    conn.execute("USE lake_b")
    return conn


# Fonction auxiliaire de comptage des lignes d'une fact_table d'un catalogue donné
def _fact_count(conn: duckdb.DuckDBPyConnection, catalog: str) -> int:
    """Return the fact_table row count for a given catalog.

    Args:
        conn: Connection with both catalogs attached.
        catalog: Catalog alias to query.

    Returns:
        int: number of rows in ``<catalog>.main.fact_table``.
    """
    query = f'SELECT COUNT(*) FROM "{catalog}"."main"."fact_table"'
    row = conn.execute(query).fetchone()
    assert row is not None
    return int(row[0])


# Test qu'une mise à jour ciblant lake_a n'écrit jamais dans lake_b
def test_updater_writes_only_to_its_catalog(
    two_catalogs_connection: duckdb.DuckDBPyConnection,
) -> None:
    """Test that a manager configured on lake_a never writes into lake_b.

    The connection's current catalog is ``lake_b`` (last ``USE``); without
    catalog qualification the insert would land there.

    Args:
        two_catalogs_connection: Connection with lake_a and lake_b attached.
    """
    conn = two_catalogs_connection
    a_before = _fact_count(conn, "lake_a")
    b_before = _fact_count(conn, "lake_b")

    updater = DatabaseUpdater(
        connection=conn,
        categorical_threshold=4,
        enable_validation=False,
        schema="main",
        catalog_alias="lake_a",
    )
    new_rows = pl.DataFrame({"id": [4, 5], "category": ["B", "A"], "value": [0.4, 0.5]})
    assert updater.update_database(new_rows, use_transaction=False) is True

    # lake_a a bien grossi, lake_b est strictement inchangé
    assert _fact_count(conn, "lake_a") == a_before + 2
    assert _fact_count(conn, "lake_b") == b_before


# Test qu'une suppression ciblant lake_a n'affecte jamais lake_b
def test_deleter_touches_only_its_catalog(
    two_catalogs_connection: duckdb.DuckDBPyConnection,
) -> None:
    """Test that a deleter configured on lake_a never deletes from lake_b.

    Args:
        two_catalogs_connection: Connection with lake_a and lake_b attached.
    """
    conn = two_catalogs_connection
    a_before = _fact_count(conn, "lake_a")
    b_before = _fact_count(conn, "lake_b")

    deleter = DatabaseDeleter(
        connection=conn,
        enable_validation=False,
        auto_cleanup=False,
        schema="main",
        catalog_alias="lake_a",
    )
    deleted = deleter.delete_rows(filters=[("id", "=", 1)], use_transaction=False)
    assert deleted.rows_deleted >= 1

    assert _fact_count(conn, "lake_a") == a_before - 1
    assert _fact_count(conn, "lake_b") == b_before
