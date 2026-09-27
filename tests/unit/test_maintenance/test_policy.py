# Importation des modules
# Modules de base
import logging
import os
import warnings
from collections.abc import Callable, Generator
from datetime import datetime
from typing import Any

# DuckDB
import duckdb

# Polars
import polars as pl

# Module de tests
import pytest

# Modules à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.maintenance import (
    DuckLakeMaintenance,
    MaintenancePolicy,
    StorageReport,
)
from dt_ducklake_manager.reporting import OperationReport
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from tests.utils.ducklake import requires_ducklake

# ---------------------------------------------------------------------------
# Fonctions auxiliaires
# ---------------------------------------------------------------------------

# Nombre de lignes par lot : avec des row groups de 122 880 lignes, quatre lots
# réordonnés produisent trois fichiers (mesuré, annexe A)
_BATCH_ROWS = 75_000


# Marqueurs appliqués à l'ensemble du module : catalogue DuckLake réel, et tables
# de plusieurs centaines de milliers de lignes reconstruites pour chaque test
pytestmark = [requires_ducklake, pytest.mark.slow]


# Lecture directe des plages [min, max] de la colonne k des fichiers actifs
def _k_ranges(conn: duckdb.DuckDBPyConnection) -> list[tuple[int, int, int]]:
    """Read ``(record_count, min, max)`` of column ``k`` for active non-empty files.

    Queries ``ducklake_file_column_stats`` directly (annexe A), independently of
    the implementation under test.

    Args:
        conn: Connection with the ``db`` catalog attached.

    Returns:
        list[tuple[int, int, int]]: One row per file, ordered by min value.
    """
    rows = conn.execute(
        """
        SELECT f.record_count, CAST(s.min_value AS BIGINT), CAST(s.max_value AS BIGINT)
        FROM __ducklake_metadata_db.ducklake_data_file f
        JOIN __ducklake_metadata_db.ducklake_file_column_stats s
            ON s.data_file_id = f.data_file_id
        JOIN __ducklake_metadata_db.ducklake_column c
            ON c.column_id = s.column_id AND c.table_id = f.table_id
        JOIN __ducklake_metadata_db.ducklake_table t ON t.table_id = f.table_id
        WHERE t.table_name = 'fact_table' AND c.column_name = 'k'
          AND f.end_snapshot IS NULL AND t.end_snapshot IS NULL
          AND c.end_snapshot IS NULL AND f.record_count > 0
        ORDER BY 2, 3
        """
    ).fetchall()
    return [(int(r[0]), int(r[1]), int(r[2])) for r in rows]


# Nombre de threads courant
def _threads(conn: Any) -> int:
    """Return the current ``threads`` setting.

    Args:
        conn: DuckDB connection.

    Returns:
        int: Current thread count.
    """
    return int(conn.execute("SELECT current_setting('threads')").fetchone()[0])


# Connexion factice qui échoue sur l'INSERT
class _FailingInsertConnection:
    """Connection proxy raising on ``INSERT INTO`` statements, delegating the rest."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

    def execute(self, query: str, *args: Any, **kwargs: Any) -> Any:
        # Simulation d'un échec au moment de la réinsertion triée
        if query.lstrip().upper().startswith("INSERT INTO"):
            raise RuntimeError("simulated insert failure")
        return self._conn.execute(query, *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


# Catalogue avec un schéma construit (cluster_by = k) puis des inserts non triés
@pytest.fixture
def clustered_conn(tmp_path: Any) -> Generator[duckdb.DuckDBPyConnection]:
    """Build a schema clustered on ``k`` then degrade it with unsorted inserts.

    Every batch spans the whole ``k`` range (0-999), so each file's range overlaps
    all the others.

    Args:
        tmp_path: pytest temporary directory.

    Yields:
        duckdb.DuckDBPyConnection: Connection to the catalog (alias ``db``).
    """
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(
        str(tmp_path / "test.ducklake"),
        data_dir,
        data_inlining_row_limit=0,
        ducklake_options={"target_file_size": "200KB"},
    ).connect()
    # Construction initiale (écrite triée sur k)
    df = pl.DataFrame(
        {
            "id": list(range(_BATCH_ROWS)),
            "k": [(i * 7919) % 1000 for i in range(_BATCH_ROWS)],
            "v": [float(i) for i in range(_BATCH_ROWS)],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=["id"], connection=conn
        ).build_schema(cluster_by=["k"])
    # Inserts non triés couvrant toute la plage de k
    for batch in range(1, 4):
        offset = batch * _BATCH_ROWS
        conn.execute(
            f"INSERT INTO db.main.fact_table SELECT range + {offset},"
            f" (range * 7919 + {batch}) % 1000, range::DOUBLE FROM range({_BATCH_ROWS})"
        )
    yield conn
    conn.close()


# Catalogue avec une table simple (sans dataset_metadata)
@pytest.fixture
def plain_conn(tmp_path: Any) -> Generator[duckdb.DuckDBPyConnection]:
    """Create a catalog with a plain ``fact_table`` and no ``dataset_metadata``.

    Args:
        tmp_path: pytest temporary directory.

    Yields:
        duckdb.DuckDBPyConnection: Connection to the catalog (alias ``db``).
    """
    # Chemins distincts de clustered_conn : les deux fixtures peuvent coexister
    data_dir = str(tmp_path / "plain_data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(
        str(tmp_path / "plain.ducklake"), data_dir, data_inlining_row_limit=0
    ).connect()
    conn.execute("CREATE TABLE db.main.fact_table (id INTEGER, value DOUBLE)")
    conn.execute(
        "INSERT INTO db.main.fact_table SELECT range, range::DOUBLE FROM range(20000)"
    )
    yield conn
    conn.close()


# Espions remplaçant les étapes de maintenance
def _install_spies(maint: DuckLakeMaintenance, calls: list[str]) -> None:
    """Replace every maintenance step by a recorder returning a neutral result.

    Args:
        maint: Instance to patch.
        calls: List receiving the called step names, in order.
    """

    def spy(name: str, result: Callable[[], Any]) -> Callable[..., Any]:
        def _recorder(*args: Any, **kwargs: Any) -> Any:
            calls.append(name)
            return result()

        return _recorder

    neutral_report = OperationReport(
        operation="recluster",
        schema="main",
        run_id=None,
        started_at=datetime.now(),
        duration_seconds=0.0,
    )
    maint.flush_inlined_data = spy("flush_inlined_data", list)  # type: ignore[method-assign]
    maint.rewrite_data_files = spy(  # type: ignore[method-assign]
        "rewrite_data_files", lambda: ("main", "fact_table", 0, 0)
    )
    maint.merge_files = spy(  # type: ignore[method-assign]
        "merge_files", lambda: ("main", "fact_table", 0, 0)
    )
    maint.recluster = spy("recluster", lambda: neutral_report)  # type: ignore[method-assign]
    maint.expire_snapshots = spy("expire_snapshots", list)  # type: ignore[method-assign]
    maint.cleanup_files = spy("cleanup_files", list)  # type: ignore[method-assign]
    maint.delete_orphaned_files = spy("delete_orphaned_files", list)  # type: ignore[method-assign]


# Rapport de stockage justifiant toutes les étapes
def _dirty_storage(**overrides: Any) -> StorageReport:
    """Build a StorageReport whose indicators justify every maintenance step.

    Args:
        **overrides: Fields to override.

    Returns:
        StorageReport: The fabricated report.
    """
    fields: dict[str, Any] = {
        "schema": "main",
        "table": "fact_table",
        "file_count": 20,
        "delete_file_count": 3,
        "small_file_count": 20,
        "inlined_rows": 10,
        "has_inlined_data": True,
        "overlap_ratio": 0.9,
        "cluster_column": "id",
    }
    fields.update(overrides)
    return StorageReport(**fields)


# ---------------------------------------------------------------------------
# Tests de storage_report
# ---------------------------------------------------------------------------


# Test que des inserts non triés produisent un recouvrement élevé
def test_storage_report_overlap_high_after_unsorted_inserts(
    clustered_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that unsorted inserts spanning the key range yield a high overlap.

    Args:
        clustered_conn: Degraded clustered catalog.
    """
    report = DuckLakeMaintenance(clustered_conn).storage_report()
    assert report.cluster_column == "k"
    assert report.file_count >= 4
    assert report.overlap_ratio is not None and report.overlap_ratio > 0.5
    assert report.delete_file_count == 0
    assert report.inlined_rows == 0 and report.has_inlined_data is False


# Test des indicateurs sans cluster_by et après un UPDATE partiel
def test_storage_report_without_cluster_by_and_deletes(
    plain_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test overlap is None without cluster_by and deletes are measured.

    Args:
        plain_conn: Plain catalog without dataset_metadata.
    """
    plain_conn.execute("UPDATE db.main.fact_table SET value = -1 WHERE id < 5000")
    maint = DuckLakeMaintenance(plain_conn)
    report = maint.storage_report()
    assert report.overlap_ratio is None
    assert report.cluster_column is None
    assert report.delete_file_count >= 1
    assert 0.0 < report.delete_ratio
    assert report.snapshot_count > 0
    assert report.oldest_snapshot_age_days is not None
    assert report.oldest_snapshot_age_days >= 0.0
    # Seuil de petit fichier : aucun sous 1 octet, tous sous 1 To
    assert maint.storage_report(min_file_size_bytes=1).small_file_count == 0
    huge = maint.storage_report(min_file_size_bytes=10**12)
    assert huge.small_file_count == huge.file_count


# Test du recouvrement nul avec un seul fichier
def test_storage_report_single_file_overlap_zero(
    plain_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that a single data file has an overlap ratio of 0.

    Args:
        plain_conn: Plain catalog, one data file.
    """
    plain_conn.execute("CREATE TABLE db.main.dataset_metadata (cluster_by VARCHAR)")
    plain_conn.execute("""INSERT INTO db.main.dataset_metadata VALUES ('["id"]')""")
    report = DuckLakeMaintenance(plain_conn).storage_report()
    assert report.file_count == 1
    assert report.overlap_ratio == 0.0


# Test que des fichiers aux plages identiques se recouvrent
def test_storage_report_identical_ranges_overlap(
    plain_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that two files with the same range count as overlapping.

    Args:
        plain_conn: Plain catalog.
    """
    plain_conn.execute("CREATE TABLE db.main.dataset_metadata (cluster_by VARCHAR)")
    plain_conn.execute("""INSERT INTO db.main.dataset_metadata VALUES ('["id"]')""")
    plain_conn.execute(
        "INSERT INTO db.main.fact_table SELECT range, 0.0 FROM range(20000)"
    )
    assert DuckLakeMaintenance(plain_conn).storage_report().overlap_ratio == 1.0


# Test de la mesure des lignes inlinées
def test_storage_report_inlined_rows(tmp_path: Any) -> None:
    """Test that inlined rows are counted, then cleared by a flush.

    Args:
        tmp_path: pytest temporary directory.
    """
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(str(tmp_path / "test.ducklake"), data_dir).connect()
    conn.execute("CREATE TABLE db.main.fact_table (id INTEGER)")
    conn.execute("INSERT INTO db.main.fact_table VALUES (1), (2), (3)")
    maint = DuckLakeMaintenance(conn)
    report = maint.storage_report()
    assert report.inlined_rows == 3
    assert report.has_inlined_data is True
    maint.flush_inlined_data("fact_table")
    report = maint.storage_report()
    assert report.inlined_rows == 0
    assert report.has_inlined_data is False
    conn.close()


# Test d'un rapport sur une table inexistante
def test_storage_report_unknown_table(plain_conn: duckdb.DuckDBPyConnection) -> None:
    """Test that an unknown table yields a default, empty report.

    Args:
        plain_conn: Plain catalog.
    """
    report = DuckLakeMaintenance(plain_conn).storage_report("missing")
    assert report.file_count == 0
    assert report.small_file_count == 0
    assert report.overlap_ratio is None


# Test de la ligne de synthèse
def test_storage_report_summary() -> None:
    """Test summary() with and without optional indicators."""
    empty = StorageReport(schema="main", table="t")
    assert "overlap n/a" in empty.summary()
    full = StorageReport(
        schema="main",
        table="t",
        file_count=3,
        total_bytes=2_400_000,
        inlined_rows=None,
        has_inlined_data=True,
        oldest_snapshot_age_days=4.25,
        overlap_ratio=0.5,
        cluster_column="k",
    )
    line = full.summary()
    assert "3 files (2.4 MB)" in line
    assert "inlined data" in line
    assert "oldest 4.2 days" in line or "oldest 4.3 days" in line
    assert "overlap 0.50 on k" in line


# ---------------------------------------------------------------------------
# Tests de recluster
# ---------------------------------------------------------------------------


# Test que recluster produit des fichiers disjoints et monotones
def test_recluster_files_disjoint_and_overlap_zero(
    clustered_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that reclustering yields disjoint, monotone files and zero overlap.

    Args:
        clustered_conn: Degraded clustered catalog.
    """
    maint = DuckLakeMaintenance(clustered_conn)
    report = maint.recluster()

    ranges = _k_ranges(clustered_conn)
    assert len(ranges) >= 2
    # Chaque fichier commence au plus bas là où le précédent se termine
    for (_, _, previous_max), (_, next_min, _) in zip(ranges, ranges[1:], strict=False):
        assert previous_max <= next_min
    assert report.operation == "recluster"
    assert report.maintenance["overlap_ratio_before"] > 0.5
    assert report.maintenance["overlap_ratio_after"] == 0.0
    assert maint.storage_report().overlap_ratio == 0.0
    assert report.snapshot_after is not None and report.snapshot_before is not None
    assert report.snapshot_after > report.snapshot_before
    assert report.files_before > 0 and report.files_after > 0
    assert "merge_files_processed" in report.maintenance


# Test que recluster conserve le nombre de lignes, le contenu et l'identité
def test_recluster_preserves_content_and_identity(
    clustered_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that rows, content and table id are unchanged by a recluster.

    Args:
        clustered_conn: Degraded clustered catalog.
    """
    table_id_query = (
        "SELECT table_id FROM __ducklake_metadata_db.ducklake_table"
        " WHERE table_name = 'fact_table' AND end_snapshot IS NULL"
    )
    table_id_before = clustered_conn.execute(table_id_query).fetchone()
    clustered_conn.execute(
        "CREATE TEMP TABLE before_copy AS SELECT * FROM db.main.fact_table"
    )

    report = DuckLakeMaintenance(clustered_conn).recluster()

    assert report.rows_before == report.rows_after == 4 * _BATCH_ROWS
    assert not report.warnings
    for query in (
        "SELECT count(*) FROM (SELECT * FROM db.main.fact_table"
        " EXCEPT ALL SELECT * FROM temp.main.before_copy)",
        "SELECT count(*) FROM (SELECT * FROM temp.main.before_copy"
        " EXCEPT ALL SELECT * FROM db.main.fact_table)",
    ):
        assert clustered_conn.execute(query).fetchone() == (0,)
    assert clustered_conn.execute(table_id_query).fetchone() == table_id_before
    # Un DELETE intégral ne laisse aucun fichier de suppression
    assert DuckLakeMaintenance(clustered_conn).storage_report().delete_file_count == 0


# Test que les fichiers de suppression orphelins ne comptent plus après recluster
def test_recluster_after_update_leaves_no_live_delete_file(
    clustered_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that delete files of removed data files are not counted afterwards.

    Measured: after a full DELETE, ``ducklake_table_info`` still counts the delete
    files of the removed data files, which ``rewrite_data_files`` cannot absorb —
    counting them would make ``maintain`` rerun the rewrite step forever.

    Args:
        clustered_conn: Degraded clustered catalog.
    """
    clustered_conn.execute("UPDATE db.main.fact_table SET v = -1 WHERE id < 1000")
    maint = DuckLakeMaintenance(clustered_conn)
    assert maint.storage_report().delete_file_count >= 1

    maint.recluster()

    report = maint.storage_report()
    assert report.delete_file_count == 0
    assert report.delete_bytes == 0
    assert report.delete_ratio == 0.0
    skipped = maint.maintain(MaintenancePolicy(max_small_files=100))
    assert skipped.maintenance["rewrite_data_files_skipped"] == 1


# Test d'un order_by explicite
def test_recluster_explicit_order_by(
    clustered_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that an explicit order_by is used (overlap measured on its first column).

    Args:
        clustered_conn: Degraded clustered catalog.
    """
    report = DuckLakeMaintenance(clustered_conn).recluster(order_by=["id"])
    assert report.maintenance["overlap_ratio_after"] == 0.0


# Test des erreurs de validation de recluster
def test_recluster_validation_errors(
    clustered_conn: duckdb.DuckDBPyConnection,
    plain_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test ValueError on unknown column, missing sort key and unknown table.

    Args:
        clustered_conn: Degraded clustered catalog.
        plain_conn: Plain catalog without cluster_by.
    """
    with pytest.raises(ValueError, match="do not exist"):
        DuckLakeMaintenance(clustered_conn).recluster(order_by=["unknown"])
    with pytest.raises(ValueError, match="No sort key"):
        DuckLakeMaintenance(plain_conn).recluster()
    with pytest.raises(ValueError, match="No sort key"):
        DuckLakeMaintenance(plain_conn).recluster(order_by=[])
    with pytest.raises(ValueError, match="does not exist"):
        DuckLakeMaintenance(plain_conn).recluster("missing", order_by=["id"])


# Test que threads est restauré après un succès
def test_recluster_restores_threads_on_success(
    clustered_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that the previous threads value is restored after a recluster.

    Args:
        clustered_conn: Degraded clustered catalog.
    """
    clustered_conn.execute("SET threads = 3")
    DuckLakeMaintenance(clustered_conn).recluster()
    assert _threads(clustered_conn) == 3


# Test que threads est restauré et la table intacte après une exception
def test_recluster_restores_threads_and_rolls_back_on_failure(
    plain_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test threads restoration and rollback when the sorted INSERT fails.

    Args:
        plain_conn: Plain catalog.
    """
    plain_conn.execute("SET threads = 3")
    failing = DuckLakeMaintenance(_FailingInsertConnection(plain_conn))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="simulated insert failure"):
        failing.recluster(order_by=["id"])

    assert _threads(plain_conn) == 3
    count = plain_conn.execute("SELECT count(*) FROM db.main.fact_table").fetchone()
    assert count == (20000,)
    # La table temporaire n'a pas survécu et un nouvel essai fonctionne
    report = DuckLakeMaintenance(plain_conn).recluster(order_by=["id"])
    assert report.rows_after == 20000


# ---------------------------------------------------------------------------
# Tests de MaintenancePolicy
# ---------------------------------------------------------------------------


# Test des valeurs par défaut
def test_policy_defaults_are_safe() -> None:
    """Test that the default policy is non-destructive."""
    policy = MaintenancePolicy()
    assert policy.recluster is False
    assert policy.retention_days is None
    assert policy.delete_orphaned is False
    assert policy.dry_run is False
    assert policy.delete_threshold == 0.1
    assert policy.max_overlap_ratio == 0.5


# Test de la validation des bornes
@pytest.mark.parametrize(
    "kwargs",
    [
        {"delete_threshold": -0.1},
        {"delete_threshold": 1.5},
        {"max_overlap_ratio": 2.0},
        {"min_file_size_bytes": -1},
        {"max_small_files": -1},
        {"retention_days": -1},
    ],
)
def test_policy_invalid_values_raise(kwargs: dict[str, Any]) -> None:
    """Test that out-of-range thresholds raise ValueError.

    Args:
        kwargs: Invalid policy arguments.
    """
    with pytest.raises(ValueError):
        MaintenancePolicy(**kwargs)


# Test des bornes incluses
def test_policy_boundary_values_accepted() -> None:
    """Test that boundary values 0 and 1 are accepted."""
    MaintenancePolicy(
        delete_threshold=0.0,
        max_overlap_ratio=1.0,
        min_file_size_bytes=0,
        max_small_files=0,
        retention_days=0,
    )


# ---------------------------------------------------------------------------
# Tests de maintain
# ---------------------------------------------------------------------------


# Test qu'une politique ne justifiant rien n'exécute rien et le journalise
def test_maintain_nothing_justified_runs_nothing(
    plain_conn: duckdb.DuckDBPyConnection, caplog: Any
) -> None:
    """Test that no step runs when no indicator justifies it, each skip logged.

    Args:
        plain_conn: Plain catalog (one file, no delete, nothing inlined).
        caplog: pytest log capture.
    """
    maint = DuckLakeMaintenance(plain_conn)
    calls: list[str] = []
    _install_spies(maint, calls)
    snapshot_before = plain_conn.execute(
        "SELECT max(snapshot_id) FROM ducklake_snapshots('db')"
    ).fetchone()

    with caplog.at_level(logging.INFO):
        report = maint.maintain(MaintenancePolicy(max_small_files=100))

    assert calls == []
    steps = [
        "flush_inlined_data",
        "rewrite_data_files",
        "merge_files",
        "recluster",
        "expire_snapshots",
        "cleanup_files",
        "delete_orphaned_files",
    ]
    for step in steps:
        assert report.maintenance[f"{step}_skipped"] == 1
        assert any(f"{step} skipped" in r.getMessage() for r in caplog.records)
    assert report.operation == "maintenance"
    assert snapshot_before is not None
    assert report.snapshot_after == snapshot_before[0]


# Test que retention_days=None n'appelle jamais expire ni cleanup
def test_maintain_without_retention_never_expires(
    plain_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that expire/cleanup are never called when retention_days is None.

    Args:
        plain_conn: Plain catalog.
    """
    maint = DuckLakeMaintenance(plain_conn)
    calls: list[str] = []
    _install_spies(maint, calls)
    maint.storage_report = lambda *a, **k: _dirty_storage()  # type: ignore[method-assign]

    maint.maintain(MaintenancePolicy(recluster=True, delete_orphaned=True))

    assert "expire_snapshots" not in calls
    assert "cleanup_files" not in calls
    assert "delete_orphaned_files" in calls


# Test de l'ordre des étapes lorsqu'elles sont toutes justifiées
def test_maintain_runs_steps_in_order(plain_conn: duckdb.DuckDBPyConnection) -> None:
    """Test the flush → rewrite → merge → recluster → expire → cleanup → orphaned
    order.

    Args:
        plain_conn: Plain catalog.
    """
    maint = DuckLakeMaintenance(plain_conn)
    calls: list[str] = []
    _install_spies(maint, calls)
    maint.storage_report = lambda *a, **k: _dirty_storage()  # type: ignore[method-assign]

    report = maint.maintain(
        MaintenancePolicy(recluster=True, retention_days=7, delete_orphaned=True)
    )

    assert calls == [
        "flush_inlined_data",
        "rewrite_data_files",
        "merge_files",
        "recluster",
        "expire_snapshots",
        "cleanup_files",
        "delete_orphaned_files",
    ]
    assert not any(key.endswith("_skipped") for key in report.maintenance)


# Test que recluster reste opt-in malgré un recouvrement élevé
def test_maintain_recluster_is_opt_in(
    clustered_conn: duckdb.DuckDBPyConnection, caplog: Any
) -> None:
    """Test recluster is skipped unless enabled, then runs when overlap is high.

    Args:
        clustered_conn: Degraded clustered catalog.
        caplog: pytest log capture.
    """
    maint = DuckLakeMaintenance(clustered_conn)
    with caplog.at_level(logging.INFO):
        report = maint.maintain(MaintenancePolicy(max_small_files=100))
    assert report.maintenance["recluster_skipped"] == 1
    assert any("recluster=False" in r.getMessage() for r in caplog.records)

    report = maint.maintain(MaintenancePolicy(recluster=True, max_small_files=100))
    assert "recluster_skipped" not in report.maintenance
    assert report.maintenance["recluster_overlap_ratio_after"] == 0.0

    # Recouvrement désormais nul : l'étape est sautée avec la raison chiffrée
    with caplog.at_level(logging.INFO):
        report = maint.maintain(MaintenancePolicy(recluster=True, max_small_files=100))
    assert report.maintenance["recluster_skipped"] == 1
    assert any(
        "overlap 0.00 <= max_overlap_ratio 0.5" in r.getMessage()
        for r in caplog.records
    )


# Test que dry_run ne modifie rien
def test_maintain_dry_run_changes_nothing(
    clustered_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that dry_run plans modifying steps without touching the catalog.

    Args:
        clustered_conn: Degraded clustered catalog.
    """
    clustered_conn.execute("UPDATE db.main.fact_table SET v = -1 WHERE id < 1000")
    maint = DuckLakeMaintenance(clustered_conn)
    before = maint.storage_report()
    snapshot_query = "SELECT max(snapshot_id) FROM ducklake_snapshots('db')"
    snapshot_before = clustered_conn.execute(snapshot_query).fetchone()

    report = maint.maintain(
        MaintenancePolicy(
            dry_run=True, recluster=True, max_small_files=0, retention_days=0
        )
    )

    after = maint.storage_report()
    assert clustered_conn.execute(snapshot_query).fetchone() == snapshot_before
    assert after.file_count == before.file_count
    assert after.overlap_ratio == before.overlap_ratio
    assert report.maintenance["rewrite_data_files_planned"] == 1
    assert report.maintenance["merge_files_planned"] == 1
    assert report.maintenance["recluster_planned"] == 1
    assert "expired_snapshots" in report.maintenance
    assert "cleaned_files" in report.maintenance


# Test qu'une étape en échec n'interrompt pas les suivantes
def test_maintain_step_failure_is_non_fatal(
    plain_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that a failing step is reported as a warning and later steps still run.

    Args:
        plain_conn: Plain catalog.
    """
    maint = DuckLakeMaintenance(plain_conn)
    calls: list[str] = []
    _install_spies(maint, calls)
    maint.storage_report = lambda *a, **k: _dirty_storage()  # type: ignore[method-assign]

    def failing_rewrite(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("boom")

    maint.rewrite_data_files = failing_rewrite  # type: ignore[method-assign]

    report = maint.maintain(MaintenancePolicy(retention_days=1))

    assert any("rewrite_data_files failed" in w for w in report.warnings)
    assert "merge_files" in calls
    assert "expire_snapshots" in calls
