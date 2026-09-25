# Importation des modules
# Modules de base
import os
import warnings
from datetime import datetime
from typing import Any

# DuckDB
import polars as pl

# Module de tests
import pytest

# Modules à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.maintenance import DatabaseRecoveryManager
from dt_ducklake_manager.operations import DatabaseDeleter, DatabaseUpdater
from dt_ducklake_manager.reporting import OperationReport
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from tests.utils.ducklake import requires_ducklake

# ---------------------------------------------------------------------------
# Tests de OperationReport.summary() / to_dict() (aucune base requise)
# ---------------------------------------------------------------------------


# Test que summary() produit une ligne stable et lisible (§6 de la spec)
def test_summary_stable_and_readable() -> None:
    """Test that summary() renders a stable, human-readable one-line string."""
    report = OperationReport(
        operation="update",
        schema="main",
        run_id="run-42",
        started_at=datetime(2026, 1, 1),
        duration_seconds=4.1,
        rows_inserted=1240,
        rows_updated=380,
        columns_added=["score", "rank"],
        files_before=3,
        files_after=2,
        bytes_before=48_200_000,
        bytes_after=31_700_000,
        snapshot_before=17,
        snapshot_after=18,
    )
    assert report.summary() == (
        "update main [run-42]: +1240 rows, ~380 updated, +2 columns (score, rank), "
        "3 -> 2 files (48.2 -> 31.7 MB), snapshot 17 -> 18, 4.1s"
    )


# Test que les parties triviales (delta nul) sont omises de la ligne de synthèse
def test_summary_omits_trivial_parts() -> None:
    """Test that zero-valued counters are omitted from summary(), not zeroed out."""
    report = OperationReport(
        operation="delete_columns",
        schema="main",
        run_id=None,
        started_at=datetime(2026, 1, 1),
        duration_seconds=0.5,
        columns_dropped=["old_col"],
    )
    assert report.summary() == "delete_columns main: -1 columns (old_col), 0.5s"


# Test que to_dict() sérialise tous les champs, started_at en ISO-8601
def test_to_dict_round_trip() -> None:
    """Test that to_dict() returns every field, with started_at as an ISO string."""
    report = OperationReport(
        operation="update",
        schema="main",
        run_id=None,
        started_at=datetime(2026, 1, 1, 12, 30),
        duration_seconds=1.0,
        warnings=["something"],
    )
    data = report.to_dict()
    assert data["operation"] == "update"
    assert data["started_at"] == "2026-01-01T12:30:00"
    assert data["warnings"] == ["something"]
    assert data["maintenance"] == {}


# ---------------------------------------------------------------------------
# Tests sur connexion in-memory (sans catalogue DuckLake réel)
# ---------------------------------------------------------------------------


# Test qu'un run_id sur une connexion sans DuckLake réel n'échoue pas
def test_run_id_ignored_without_real_catalog(built_ducklake_schema: Any) -> None:
    """Test that run_id/commit_message are silently skipped without a real catalog.

    The DEBUG skip line itself is not asserted here: ``_init_logger`` pins every
    named logger back to INFO on each manager construction (by design, so
    repeated instantiation never leaves a logger at a stale level), which makes
    DEBUG output unobservable through ``caplog`` without reaching into that
    internal. What matters and is directly observable is that no exception is
    raised and the DuckLake-only fields stay empty.

    Args:
        built_ducklake_schema: In-memory DuckDB connection (no DuckLake catalog).
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, categorical_threshold=4)
    update_df = pl.DataFrame({"id": [100], "category": ["A"], "value": [1.0]})

    success = updater.update_database(update_df, run_id="run-x", commit_message="test")

    assert success is True
    assert updater.last_report is not None
    assert updater.last_report.snapshot_after is None


# Test qu'un échec produit un rapport partiel exposé via last_report
def test_failure_produces_partial_report(built_ducklake_schema: Any) -> None:
    """Test that a mid-operation failure attaches a partial report to last_report.

    Args:
        built_ducklake_schema: In-memory DuckDB connection with a built schema.
    """
    updater = DatabaseUpdater(connection=built_ducklake_schema, categorical_threshold=4)

    def _boom(*args: Any, **kwargs: Any) -> bool:
        raise RuntimeError("panne simulée")

    updater._upsert_fact_table = _boom  # type: ignore[assignment]

    update_df = pl.DataFrame({"id": [200], "category": ["A"], "value": [1.0]})
    success = updater.update_database(update_df)

    assert success is False
    assert updater.last_report is not None
    assert updater.last_report.operation == "update"
    assert updater.last_report.warnings
    assert "panne simulée" in updater.last_report.warnings[0]
    # État post-échec jamais renseigné (rollback avant toute mesure d'après-état)
    assert updater.last_report.snapshot_after is None


# ---------------------------------------------------------------------------
# Tests de bout en bout sur un catalogue DuckLake réel (§6, annexe A)
# ---------------------------------------------------------------------------


# Fixture d'un catalogue DuckLake réel sur disque, avec un schéma déjà construit
@pytest.fixture
def real_catalog_conn(tmp_path: Any) -> Any:
    """Provide a real on-disk DuckLake catalog with a built schema (ids 1..3).

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        duckdb.DuckDBPyConnection: connection to the built catalog.
    """
    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(catalog, data_dir, data_inlining_row_limit=0).connect()

    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "category": ["A", "B", "A"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=["id"], connection=conn
        ).build_schema()
    return conn


# Test que les comptages de lignes d'un update sont exacts (table_changes)
@requires_ducklake
def test_update_row_counts_from_table_changes(real_catalog_conn: Any) -> None:
    """Test that rows_inserted/rows_updated are exact, measured via table_changes.

    Args:
        real_catalog_conn: Fixture providing a real, on-disk DuckLake catalog.
    """
    updater = DatabaseUpdater(connection=real_catalog_conn, categorical_threshold=4)

    # 1 ligne nouvelle (id=4) + 1 ligne existante mise à jour (id=1)
    update_df = pl.DataFrame(
        {"id": [1, 4], "category": ["A", "B"], "value": [99.0, 4.0]}
    )
    success = updater.update_database(
        update_df, run_id="run-42", commit_message="test update"
    )

    assert success is True
    report = updater.last_report
    assert report is not None
    assert report.rows_inserted == 1
    assert report.rows_updated == 1
    assert report.rows_deleted == 0
    assert report.snapshot_before is not None
    assert report.snapshot_after is not None
    assert report.snapshot_after > report.snapshot_before


# Test que add_columns mesure mises à jour et insertions sur un catalogue réel
@requires_ducklake
def test_add_columns_row_counts_from_table_changes(real_catalog_conn: Any) -> None:
    """Test that add_columns' outer merge reports exact updated/inserted counts.

    Args:
        real_catalog_conn: Fixture providing a real, on-disk DuckLake catalog.
    """
    updater = DatabaseUpdater(connection=real_catalog_conn, categorical_threshold=4)

    # 2 clés existantes (id=1, 2) + 1 clé nouvelle (id=10)
    df = pl.DataFrame({"id": [1, 2, 10], "score": [0.5, 0.6, 0.7]})
    report = updater.add_columns(df, compact_after_update=False)

    assert report.rows_updated == 2
    assert report.rows_inserted == 1
    rows = real_catalog_conn.execute(
        "SELECT id, category, value, score FROM db.main.fact_table ORDER BY id"
    ).fetchall()
    assert rows == [
        (1, "A", 1.0, 0.5),
        (2, "B", 2.0, 0.6),
        (3, "A", 3.0, None),
        (10, None, None, 0.7),
    ]


# Test qu'une compaction sans effet journalise des compteurs explicitement à zéro
@requires_ducklake
def test_maintenance_no_effect_explicit_zero(real_catalog_conn: Any) -> None:
    """Test that a no-op rewrite reports explicit zeros, not silent absence.

    The zero is asserted on the counters of the report, not on the wording of the
    log line.

    Args:
        real_catalog_conn: Fixture providing a real, on-disk DuckLake catalog.
    """
    updater = DatabaseUpdater(connection=real_catalog_conn, categorical_threshold=4)
    # Une seule ligne nouvelle : pas de suppression, rewrite_data_files ne peut
    # rien avoir à réécrire (seuil de suppression jamais atteint).
    update_df = pl.DataFrame({"id": [10], "category": ["A"], "value": [10.0]})

    success = updater.update_database(update_df)

    assert success is True
    report = updater.last_report
    assert report is not None
    assert report.maintenance["rewrite_files_processed"] == 0
    assert report.maintenance["rewrite_files_created"] == 0


# Test que run_id/commit_message se retrouvent dans ducklake_snapshots
@requires_ducklake
def test_run_id_visible_in_snapshots(real_catalog_conn: Any) -> None:
    """Test that run_id/commit_message land on the resulting DuckLake snapshot.

    Args:
        real_catalog_conn: Fixture providing a real, on-disk DuckLake catalog.
    """
    updater = DatabaseUpdater(connection=real_catalog_conn, categorical_threshold=4)
    update_df = pl.DataFrame({"id": [20], "category": ["A"], "value": [20.0]})

    success = updater.update_database(
        update_df,
        run_id="run-42",
        commit_message="test commit",
        commit_info={"model_version": "1.3"},
    )
    assert success is True

    recovery = DatabaseRecoveryManager(connection=real_catalog_conn)
    snapshots = recovery.list_ducklake_snapshots()
    assert snapshots is not None

    # Le snapshot le plus récent peut être celui d'une compaction post-commit
    # (sans message) : on retrouve celui du run par son author, pas par position.
    matches = snapshots.filter(snapshots["author"] == "run-42")
    assert len(matches) == 1
    tagged_snapshot = matches.rows(named=True)[0]
    assert tagged_snapshot["commit_message"] == "test commit"
    assert "model_version" in tagged_snapshot["commit_extra_info"]


# Test que delete_columns (métadonnées seules) laisse rows_inserted/deleted à zéro
@requires_ducklake
def test_delete_columns_report_has_no_row_changes(real_catalog_conn: Any) -> None:
    """Test that a metadata-only delete_columns reports zero row changes.

    Args:
        real_catalog_conn: Fixture providing a real, on-disk DuckLake catalog.
    """
    deleter = DatabaseDeleter(connection=real_catalog_conn)

    report = deleter.delete_columns(["value"], run_id="run-7")

    assert report.columns_dropped == ["value"]
    assert report.rows_inserted == 0
    assert report.rows_updated == 0
    assert report.rows_deleted == 0
    assert report.files_before == report.files_after
