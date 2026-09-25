# Importation des modules
# Modules de base
import os
import warnings
from collections.abc import Generator
from typing import Any

import duckdb
import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.maintenance import DatabaseRecoveryManager
from dt_ducklake_manager.operations import DatabaseDeleter, DatabaseUpdater
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from tests.utils.ducklake import requires_ducklake

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Catalogue DuckLake réel construit, puis modifié par un update et un delete
@pytest.fixture
def lake(tmp_path: Any) -> Generator[tuple[duckdb.DuckDBPyConnection, int]]:
    """Provide a real catalog whose fact table changed after its build.

    The schema is built (ids 1 to 5), then one update inserts id=10 and changes
    id=1, then one deletion removes id=2.

    Args:
        tmp_path: pytest temporary directory.

    Yields:
        tuple: the connection and the snapshot id right after the build.
    """
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(
        str(tmp_path / "recovery.ducklake"), data_dir, data_inlining_row_limit=0
    ).connect()
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category": ["A", "B", "A", "C", "B"],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=["id"], connection=conn
        ).build_schema()
    build_snapshot = conn.execute(
        "SELECT max(snapshot_id) FROM ducklake_snapshots('db')"
    ).fetchone()[0]

    DatabaseUpdater(conn, categorical_threshold=4).update_database(
        pl.DataFrame({"id": [1, 10], "category": ["A", "C"], "value": [9.0, 1.0]}),
        run_id="bad-run",
        compact_after_update=False,
    )
    DatabaseDeleter(conn).delete_rows([("id", "=", 2)], compact_after_update=False)

    yield conn, build_snapshot
    conn.close()


# Lecture triée de la table des faits
def _facts(conn: duckdb.DuckDBPyConnection) -> list[tuple[Any, ...]]:
    """Read the fact table sorted by id.

    Args:
        conn: Connection attached to the catalog.

    Returns:
        list[tuple]: The rows of the fact table.
    """
    return conn.execute("SELECT * FROM fact_table ORDER BY id").fetchall()


# ---------------------------------------------------------------------------
# Tests de list_ducklake_snapshots()
# ---------------------------------------------------------------------------


# Test que l'historique est renvoyé trié, avec l'auteur des écritures
@requires_ducklake
def test_list_ducklake_snapshots_returns_history(lake: Any) -> None:
    """Test that the history is sorted descending and exposes the run_id.

    Args:
        lake: Real catalog changed after its build.
    """
    conn, _ = lake
    snapshots = DatabaseRecoveryManager(conn).list_ducklake_snapshots()

    assert snapshots is not None
    ids = snapshots["snapshot_id"].to_list()
    assert ids == sorted(ids, reverse=True)
    assert "bad-run" in snapshots["author"].to_list()


# Test que l'historique est indisponible sans catalogue DuckLake
def test_list_ducklake_snapshots_without_catalog() -> None:
    """Test that a plain in-memory connection yields None."""
    assert DatabaseRecoveryManager().list_ducklake_snapshots() is None


# ---------------------------------------------------------------------------
# Tests de restore_snapshot()
# ---------------------------------------------------------------------------


# Test que la restauration ramène les lignes de la table des faits
@requires_ducklake
def test_restore_snapshot_restores_rows(lake: Any) -> None:
    """Test that restoring the build snapshot undoes the update and the deletion.

    Args:
        lake: Real catalog changed after its build.
    """
    conn, build_snapshot = lake
    expected = conn.execute(
        f"SELECT * FROM fact_table AT (VERSION => {build_snapshot}) ORDER BY id"
    ).fetchall()
    assert _facts(conn) != expected

    report = DatabaseRecoveryManager(conn).restore_snapshot(
        build_snapshot, run_id="rollback"
    )

    assert _facts(conn) == expected
    assert report.operation == "restore_snapshot"
    assert (report.rows_before, report.rows_after) == (5, 5)
    assert report.snapshot_after is not None
    assert report.snapshot_before is not None
    assert report.snapshot_after > report.snapshot_before


# Test que la restauration est un nouveau snapshot, lui-même annulable
@requires_ducklake
def test_restore_snapshot_keeps_history(lake: Any) -> None:
    """Test that the restoration is a new, authored snapshot and can be undone.

    Args:
        lake: Real catalog changed after its build.
    """
    conn, build_snapshot = lake
    recovery = DatabaseRecoveryManager(conn)
    before_restore = conn.execute(
        "SELECT max(snapshot_id) FROM ducklake_snapshots('db')"
    ).fetchone()[0]
    rows_before_restore = _facts(conn)

    recovery.restore_snapshot(build_snapshot, run_id="rollback")
    author = conn.execute(
        "SELECT author FROM ducklake_snapshots('db') ORDER BY snapshot_id DESC LIMIT 1"
    ).fetchone()[0]
    assert author == "rollback"

    # Annulation de la restauration par une seconde restauration
    recovery.restore_snapshot(before_restore)
    assert _facts(conn) == rows_before_restore


# Test que la restauration peut se limiter à la table des faits
@requires_ducklake
def test_restore_snapshot_single_table(lake: Any) -> None:
    """Test that only the requested table is restored.

    Args:
        lake: Real catalog changed after its build.
    """
    conn, build_snapshot = lake
    stamp = conn.execute("SELECT updated_at FROM dataset_metadata").fetchone()

    DatabaseRecoveryManager(conn).restore_snapshot(
        build_snapshot, tables=["fact_table"]
    )

    assert conn.execute("SELECT updated_at FROM dataset_metadata").fetchone() == stamp


# Test qu'un snapshot inconnu est refusé sans rien modifier
@requires_ducklake
def test_restore_snapshot_unknown_snapshot_raises(lake: Any) -> None:
    """Test that an unknown snapshot raises ValueError before any write.

    Args:
        lake: Real catalog changed after its build.
    """
    conn, _ = lake
    before = _facts(conn)
    with pytest.raises(ValueError, match="does not exist"):
        DatabaseRecoveryManager(conn).restore_snapshot(999_999)
    assert _facts(conn) == before


# Test qu'une table dont les colonnes ont changé est refusée
@requires_ducklake
def test_restore_snapshot_changed_columns_raises(lake: Any) -> None:
    """Test that a table whose columns changed since the snapshot is refused.

    Args:
        lake: Real catalog changed after its build.
    """
    conn, build_snapshot = lake
    DatabaseUpdater(conn, categorical_threshold=4).add_columns(
        pl.DataFrame({"id": [1], "score": [1.0]}), compact_after_update=False
    )
    before = _facts(conn)

    with pytest.raises(ValueError, match="columns of 'fact_table' changed"):
        DatabaseRecoveryManager(conn).restore_snapshot(build_snapshot)
    assert _facts(conn) == before


# Test qu'une restauration en échec est annulée et ne laisse pas de copie
@requires_ducklake
def test_restore_snapshot_failure_rolls_back(
    lake: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a failure while refilling the tables leaves them untouched.

    Args:
        lake: Real catalog changed after its build.
        monkeypatch: pytest fixture used to inject the failure.
    """
    conn, build_snapshot = lake
    before = _facts(conn)
    recovery = DatabaseRecoveryManager(conn)

    # Échec de l'annotation du commit, après le vidage et le remplissage
    def _boom(*args: Any, **kwargs: Any) -> None:
        raise duckdb.IOException("simulated I/O error")

    monkeypatch.setattr(
        "dt_ducklake_manager.maintenance.recovery._set_commit_message", _boom
    )

    with pytest.raises(duckdb.IOException):
        recovery.restore_snapshot(build_snapshot)
    assert _facts(conn) == before
    temp_tables = conn.execute(
        "SELECT table_name FROM duckdb_tables() WHERE temporary"
    ).fetchall()
    assert temp_tables == []


# Test des arguments invalides
def test_restore_snapshot_invalid_arguments() -> None:
    """Test that an empty table list or a missing catalog raise ValueError."""
    recovery = DatabaseRecoveryManager()
    with pytest.raises(ValueError, match="at least one table"):
        recovery.restore_snapshot(1, tables=[])
    with pytest.raises(ValueError, match="time travel is unavailable"):
        recovery.restore_snapshot(1)
