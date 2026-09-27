# Importation des modules
# Modules de base
import os
from collections.abc import Generator
from datetime import datetime
from typing import Any

# DuckDB
import duckdb

# Module de tests
import pytest

# Modules à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.maintenance import DuckLakeMaintenance, MaintenancePolicy
from dt_ducklake_manager.reporting import OperationReport
from tests.utils.ducklake import requires_ducklake

# Marqueur appliqué à l'ensemble du module : tous les tests sont ignorés si
# l'extension ducklake n'est pas disponible dans l'environnement.
pytestmark = requires_ducklake


# ---------------------------------------------------------------------------
# Fixtures communes
# ---------------------------------------------------------------------------


# Initialisation d'un catalogue DuckLake temporaire avec une table de test
@pytest.fixture
def ducklake_conn(tmp_path: Any) -> Generator[tuple[Any, str]]:
    """Create a temporary DuckLake catalog with a test table.

    Args:
        tmp_path: pytest temporary directory.

    Yields:
        tuple: (connection, table_name).
    """
    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(catalog, data_dir).connect()
    # Création d'une table de test avec quelques lignes pour que la maintenance
    # ait des données à traiter
    conn.execute("CREATE TABLE fact_table (id INTEGER, value DOUBLE)")
    conn.execute("INSERT INTO fact_table VALUES (1, 1.0), (2, 2.0), (3, 3.0)")
    yield conn, "fact_table"
    conn.close()


# Initialisation d'une instance DuckLakeMaintenance prête à l'emploi
@pytest.fixture
def maint(ducklake_conn: tuple[Any, str]) -> DuckLakeMaintenance:
    """Return a DuckLakeMaintenance instance ready for use.

    Args:
        ducklake_conn: Fixture providing (connection, table_name).

    Returns:
        DuckLakeMaintenance: initialized with the test connection.
    """
    conn, _ = ducklake_conn
    return DuckLakeMaintenance(conn)


# ---------------------------------------------------------------------------
# Tests du constructeur
# ---------------------------------------------------------------------------


# Test de l'initialisation avec l'alias par défaut
def test_init_default_alias(ducklake_conn: Any) -> None:
    """Test that the default catalog_alias is 'db'.

    Args:
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, _ = ducklake_conn
    maint = DuckLakeMaintenance(conn)
    assert maint.catalog_alias == "db"


# Test de l'initialisation avec un alias personnalisé
def test_init_custom_catalog_alias(ducklake_conn: Any) -> None:
    """Test that DuckLakeMaintenance stores a custom catalog alias.

    Args:
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, _ = ducklake_conn
    maint = DuckLakeMaintenance(conn, catalog_alias="my_lake")
    assert maint.catalog_alias == "my_lake"


# Test que le schéma est conservé au même titre que l'alias du catalogue
def test_init_schema_default_and_custom(ducklake_conn: Any) -> None:
    """Test that ``schema`` defaults to 'main' and is stored when provided.

    Args:
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, _ = ducklake_conn
    assert DuckLakeMaintenance(conn).schema == "main"
    assert DuckLakeMaintenance(conn, schema="predictions").schema == "predictions"


# ---------------------------------------------------------------------------
# Tests des méthodes individuelles
# ---------------------------------------------------------------------------


# Test que merge_files s'exécute sans erreur et retourne le 4-uplet attendu
def test_merge_files_executes_without_error(maint: Any, ducklake_conn: Any) -> None:
    """Test that merge_files completes without raising an exception.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    # Aucune exception ne doit être levée
    result = maint.merge_files(table, schema="main")
    schema_name, table_name, files_processed, files_created = result
    assert schema_name == "main"
    assert table_name == table
    assert isinstance(files_processed, int)
    assert isinstance(files_created, int)


# Test que merge_files accepte des valeurs personnalisées des paramètres réels
def test_merge_files_accepts_custom_parameters(maint: Any, ducklake_conn: Any) -> None:
    """Test that merge_files accepts min_file_size/max_file_size/max_compacted_files.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    # Aucune exception ne doit être levée avec des paramètres personnalisés (entiers
    # en octets — DuckLake rejette une valeur avec unité, cf. spécification §5.5)
    maint.merge_files(
        table,
        schema="main",
        min_file_size=1_000,
        max_file_size=10_000_000,
        max_compacted_files=10,
    )


# Test que rewrite_data_files s'exécute sans erreur et retourne le 4-uplet attendu
def test_rewrite_data_files_executes_without_error(
    maint: Any, ducklake_conn: Any
) -> None:
    """Test that rewrite_data_files completes without raising an exception.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    result = maint.rewrite_data_files(table, schema="main")
    schema_name, table_name, files_processed, files_created = result
    assert schema_name == "main"
    assert table_name == table


# Test que rewrite_data_files ne réécrit rien sous le seuil de suppression
def test_rewrite_data_files_zero_when_no_deletions(
    maint: Any, ducklake_conn: Any
) -> None:
    """Test that rewrite_data_files returns explicit zeros when nothing crosses the
    threshold, and that compact reports them in the operation report.

    The zero is asserted on the returned counters, not on the wording of the log.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    # Aucune suppression n'a eu lieu sur cette table : le seuil par défaut (0.1)
    # ne peut pas être atteint
    assert maint.rewrite_data_files(table, schema="main") == ("main", table, 0, 0)

    report = OperationReport(
        operation="update",
        schema="main",
        run_id=None,
        started_at=datetime.now(),
        duration_seconds=0.0,
    )
    maint.compact(table, report=report)
    assert report.maintenance["rewrite_files_processed"] == 0
    assert report.maintenance["rewrite_files_created"] == 0


# Test que rewrite_data_files réécrit effectivement après un UPDATE partiel
def test_rewrite_data_files_low_threshold_rewrites_after_update(
    tmp_path: Any,
) -> None:
    """Test that a low delete_threshold actually rewrites after a partial UPDATE.

    Mirrors the measured scenario from annexe A of the specification: a large
    enough batch INSERT (inlining disabled) followed by a partial UPDATE produces
    a real delete-tombstone file that ``rewrite_data_files`` can then absorb — a
    handful of rows on a freshly-flushed inlined table does not reliably do so.

    Args:
        tmp_path: pytest temporary directory.
    """
    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(catalog, data_dir, data_inlining_row_limit=0).connect()
    conn.execute("CREATE TABLE fact_table (id INTEGER, value DOUBLE)")
    conn.execute("INSERT INTO fact_table SELECT range, range::DOUBLE FROM range(20000)")
    # UPDATE partiel : produit un fichier de suppression (delete file)
    conn.execute("UPDATE fact_table SET value = value + 1 WHERE id < 5000")

    maint = DuckLakeMaintenance(conn)
    _, _, files_processed, files_created = maint.rewrite_data_files(
        "fact_table", delete_threshold=0.01
    )
    assert files_processed > 0
    assert files_created > 0
    conn.close()


# Test que expire_snapshots s'exécute sans erreur avec la valeur par défaut
def test_expire_snapshots_default_days(maint: Any) -> None:
    """Test that expire_snapshots with default older_than_days=30 does not raise.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    result = maint.expire_snapshots()
    assert isinstance(result, list)


# Test que expire_snapshots accepte une valeur personnalisée de older_than_days
def test_expire_snapshots_custom_days(maint: Any) -> None:
    """Test that expire_snapshots accepts a custom older_than_days value.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    maint.expire_snapshots(older_than_days=7)


# Test que expire_snapshots accepte dry_run
def test_expire_snapshots_dry_run(maint: Any) -> None:
    """Test that expire_snapshots accepts and honors dry_run=True.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    result = maint.expire_snapshots(older_than_days=0, dry_run=True)
    assert isinstance(result, list)


# Test que cleanup_files s'exécute sans erreur
def test_cleanup_files_executes_without_error(maint: Any) -> None:
    """Test that cleanup_files completes without raising an exception.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    result = maint.cleanup_files()
    assert isinstance(result, list)


# Test que cleanup_files accepte dry_run
def test_cleanup_files_dry_run(maint: Any) -> None:
    """Test that cleanup_files accepts and honors dry_run=True.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    result = maint.cleanup_files(dry_run=True)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests de flush_inlined_data (§5.2)
# ---------------------------------------------------------------------------


# Test que flush_inlined_data écrit les lignes inlinées et retourne le nombre de
# lignes vidangées
def test_flush_inlined_data_writes_inlined_rows(maint: Any, ducklake_conn: Any) -> None:
    """Test that flush_inlined_data flushes rows inlined by default and returns
    the count.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name) with default
            inlining (the fixture's 3-row INSERT is small enough to be inlined).
    """
    conn, table = ducklake_conn

    # Avec l'inlining par défaut (non désactivé dans cette fixture), l'insertion de
    # 3 lignes de la fixture ne produit encore aucun fichier
    file_count_before = conn.execute(
        f"SELECT file_count FROM ducklake_table_info('db') WHERE table_name = '{table}'"
    ).fetchone()[0]
    assert file_count_before == 0

    rows = maint.flush_inlined_data(table)
    total_flushed = sum(r[2] for r in rows)
    assert total_flushed == 3

    file_count_after = conn.execute(
        f"SELECT file_count FROM ducklake_table_info('db') WHERE table_name = '{table}'"
    ).fetchone()[0]
    assert file_count_after == 1


# Test que flush_inlined_data sans table vidange l'ensemble du catalogue
def test_flush_inlined_data_whole_catalog(maint: Any, ducklake_conn: Any) -> None:
    """Test that flush_inlined_data(table=None) flushes every table.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    rows = maint.flush_inlined_data()
    assert any(r[2] == 3 for r in rows)


# Test que flush_inlined_data retourne une liste vide sans lignes inlinées
def test_flush_inlined_data_nothing_to_flush(maint: Any, ducklake_conn: Any) -> None:
    """Test that flush_inlined_data returns an empty list once already flushed.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    maint.flush_inlined_data(table)
    # Un second appel ne trouve plus rien à vidanger
    assert maint.flush_inlined_data(table) == []


# ---------------------------------------------------------------------------
# Tests de delete_orphaned_files
# ---------------------------------------------------------------------------


# Test que delete_orphaned_files s'exécute sans erreur en dry_run (défaut)
def test_delete_orphaned_files_dry_run_default(maint: Any, ducklake_conn: Any) -> None:
    """Test that delete_orphaned_files defaults to dry_run=True and returns a list.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    result = maint.delete_orphaned_files()
    assert isinstance(result, list)


# Test que delete_orphaned_files accepte older_than
def test_delete_orphaned_files_with_older_than(maint: Any, ducklake_conn: Any) -> None:
    """Test that delete_orphaned_files accepts an older_than cutoff.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    result = maint.delete_orphaned_files(older_than=datetime.now(), dry_run=True)
    assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests de compact et de maintain avec rétention explicite
# ---------------------------------------------------------------------------


# Test que compact renvoie et reporte les quatre compteurs, zéros compris
def test_compact_returns_and_reports_counters(maint: Any, ducklake_conn: Any) -> None:
    """Test that compact returns the merge/rewrite counters and fills the report.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    report = OperationReport(
        operation="update_database",
        schema="main",
        run_id=None,
        started_at=datetime.now(),
        duration_seconds=0.0,
    )
    counters = maint.compact(table, report=report)
    assert set(counters) == {
        "merge_files_processed",
        "merge_files_created",
        "rewrite_files_processed",
        "rewrite_files_created",
    }
    assert all(isinstance(v, int) for v in counters.values())
    assert report.maintenance == counters


# Test que compact n'échoue pas sur une table inexistante
def test_compact_unknown_table_does_not_raise(maint: Any) -> None:
    """Test that compact swallows failures and reports zero counters.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    counters = maint.compact("table_inexistante_xyz")
    assert all(v == 0 for v in counters.values())


# Test que maintain avec rétention vidange les lignes inlinées puis expire et nettoie
def test_maintain_with_retention_flushes_then_expires_and_cleans(
    maint: Any, ducklake_conn: Any
) -> None:
    """Test that maintain with a retention flushes inlined rows, then always runs
    expire_snapshots and cleanup_files, skipping the no-op steps.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name) with 3 inlined rows.
    """
    _, table = ducklake_conn
    call_log: list[str] = []

    original_flush = maint.flush_inlined_data
    original_expire = maint.expire_snapshots

    def tracking_flush(*args: Any, **kwargs: Any) -> Any:
        call_log.append("flush_inlined_data")
        return original_flush(*args, **kwargs)

    def tracking_expire(*args: Any, **kwargs: Any) -> Any:
        call_log.append("expire_snapshots")
        return original_expire(*args, **kwargs)

    maint.flush_inlined_data = tracking_flush
    maint.expire_snapshots = tracking_expire

    report = maint.maintain(MaintenancePolicy(retention_days=30), table, schema="main")

    assert call_log == ["flush_inlined_data", "expire_snapshots"]
    assert report.maintenance["flush_inlined_rows"] == 3
    # Aucun fichier de suppression, un seul petit fichier : étapes sans effet sautées
    assert report.maintenance["rewrite_data_files_skipped"] == 1
    assert report.maintenance["merge_files_skipped"] == 1
    assert "expired_snapshots" in report.maintenance
    assert "cleaned_files" in report.maintenance
    # Jamais de recluster ni de suppression d'orphelins
    assert report.maintenance["recluster_skipped"] == 1
    assert report.maintenance["delete_orphaned_files_skipped"] == 1


# Test que maintain continue après un échec partiel
def test_maintain_continues_on_step_failure(maint: Any, ducklake_conn: Any) -> None:
    """Test that maintain records a warning and continues if one step fails.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    call_log: list[str] = []

    original_expire = maint.expire_snapshots
    original_cleanup = maint.cleanup_files

    def failing_flush(*args: Any, **kwargs: Any) -> None:
        # Simulation d'un échec sur la première étape
        call_log.append("flush_inlined_data")
        raise RuntimeError("Échec simulé de flush_inlined_data")

    def tracking_expire(*args: Any, **kwargs: Any) -> Any:
        call_log.append("expire_snapshots")
        return original_expire(*args, **kwargs)

    def tracking_cleanup(*args: Any, **kwargs: Any) -> Any:
        call_log.append("cleanup_files")
        return original_cleanup(*args, **kwargs)

    maint.flush_inlined_data = failing_flush
    maint.expire_snapshots = tracking_expire
    maint.cleanup_files = tracking_cleanup

    # maintain ne doit pas lever d'exception malgré l'échec du flush
    report = maint.maintain(MaintenancePolicy(retention_days=30), table, schema="main")

    assert call_log == ["flush_inlined_data", "expire_snapshots", "cleanup_files"]
    assert any("flush_inlined_data failed" in w for w in report.warnings)


# ---------------------------------------------------------------------------
# Tests de set_partitioned_by
# ---------------------------------------------------------------------------


# Test que set_partitioned_by applique le DDL sans erreur
def test_set_partitioned_by_simple(maint: Any, ducklake_conn: Any) -> None:
    """Test that set_partitioned_by executes ALTER TABLE without raising.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, table = ducklake_conn
    # Aucune exception ne doit être levée
    maint.set_partitioned_by(table, partition_by=["id"])


# Test que set_partitioned_by lève ValueError sur liste vide
def test_set_partitioned_by_empty_raises(maint: Any, ducklake_conn: Any) -> None:
    """Test that set_partitioned_by raises ValueError when partition_by is empty.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    with pytest.raises(ValueError, match="partition_by must not be empty"):
        maint.set_partitioned_by(table, partition_by=[])


# Test que set_partitioned_by accepte plusieurs clés
def test_set_partitioned_by_multiple_keys(maint: Any, ducklake_conn: Any) -> None:
    """Test that set_partitioned_by accepts a list with multiple partition keys.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, table = ducklake_conn
    # id et value sont toutes les deux des colonnes valides de fact_table
    maint.set_partitioned_by(table, partition_by=["id", "value"])


# ---------------------------------------------------------------------------
# Tests de reset_partitioned_by
# ---------------------------------------------------------------------------


# Test que reset_partitioned_by supprime le partitionnement sans erreur
def test_reset_partitioned_by(maint: Any, ducklake_conn: Any) -> None:
    """Test that reset_partitioned_by executes RESET PARTITIONED BY without raising.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, table = ducklake_conn
    # On définit d'abord un partitionnement pour avoir quelque chose à supprimer
    maint.set_partitioned_by(table, partition_by=["id"])
    # La suppression ne doit pas lever d'exception
    maint.reset_partitioned_by(table)


# Test que reset_partitioned_by fonctionne même sans partitionnement préalable
def test_reset_partitioned_by_without_prior_partitioning(
    maint: Any, ducklake_conn: Any
) -> None:
    """Test that reset_partitioned_by succeeds even if no partitioning was defined.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    # La table de test n'a pas de partitionnement défini.
    # Le reset doit quand même passer
    maint.reset_partitioned_by(table)


# ---------------------------------------------------------------------------
# Tests de repartition
# ---------------------------------------------------------------------------


# Test de repartition avec changement de clé
def test_repartition_change_key(maint: Any, ducklake_conn: Any) -> None:
    """Test that repartition resets and applies new partition keys.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, table = ducklake_conn
    # Partitionnement initial
    maint.set_partitioned_by(table, partition_by=["id"])
    # Repartitionnement sur une autre colonne, sans réécriture pour accélérer le test
    maint.repartition(table, partition_by=["value"], run_maintenance=False)


# Test de repartition avec suppression du partitionnement (partition_by=None)
def test_repartition_remove_partitioning(maint: Any, ducklake_conn: Any) -> None:
    """Test that repartition with partition_by=None removes partitioning entirely.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, table = ducklake_conn
    maint.set_partitioned_by(table, partition_by=["id"])
    # Suppression sans nouveau partitionnement
    maint.repartition(table, partition_by=None, run_maintenance=False)


# Test que repartition déclenche merge_files
# et rewrite_data_files quand run_maintenance=True
def test_repartition_with_maintenance(maint: Any, ducklake_conn: Any) -> None:
    """Test that repartition triggers merge_files
    and rewrite_data_files when run_maintenance=True.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    call_log: list[str] = []

    original_merge = maint.merge_files
    original_rewrite = maint.rewrite_data_files

    def tracking_merge(tbl: Any, **kwargs: Any) -> None:
        call_log.append("merge_files")
        original_merge(tbl, **kwargs)

    def tracking_rewrite(tbl: Any, **kwargs: Any) -> None:
        call_log.append("rewrite_data_files")
        original_rewrite(tbl, **kwargs)

    maint.merge_files = tracking_merge
    maint.rewrite_data_files = tracking_rewrite

    maint.repartition(table, partition_by=["id"], run_maintenance=True)

    # Vérification que les deux méthodes de maintenance ont bien été appelées
    assert "merge_files" in call_log
    assert "rewrite_data_files" in call_log


# Test que repartition ne déclenche pas la maintenance quand run_maintenance=False
def test_repartition_without_maintenance(maint: Any, ducklake_conn: Any) -> None:
    """Test that repartition skips maintenance calls when run_maintenance=False.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    call_log: list[str] = []

    def tracking_merge(tbl: Any, **kwargs: Any) -> None:
        call_log.append("merge_files")

    def tracking_rewrite(tbl: Any, **kwargs: Any) -> None:
        call_log.append("rewrite_data_files")

    maint.merge_files = tracking_merge
    maint.rewrite_data_files = tracking_rewrite

    maint.repartition(table, partition_by=["id"], run_maintenance=False)

    # Aucune méthode de maintenance ne doit avoir été appelée
    assert "merge_files" not in call_log
    assert "rewrite_data_files" not in call_log


# ---------------------------------------------------------------------------
# Fusion effective des petits fichiers
# ---------------------------------------------------------------------------


# Catalogue dont la table est répartie en plusieurs petits fichiers
@pytest.fixture
def fragmented_conn(tmp_path: Any) -> Generator[duckdb.DuckDBPyConnection]:
    """Provide a real catalog whose table is split into six small files.

    Args:
        tmp_path: pytest temporary directory.

    Yields:
        duckdb.DuckDBPyConnection: connection holding ``fact_table`` (6 files).
    """
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(
        str(tmp_path / "frag.ducklake"), data_dir, data_inlining_row_limit=0
    ).connect()
    conn.execute("CREATE TABLE fact_table (id INTEGER, value DOUBLE)")
    for i in range(6):
        conn.execute(
            f"INSERT INTO fact_table SELECT range, range::DOUBLE"
            f" FROM range({i * 1000}, {(i + 1) * 1000})"
        )
    yield conn
    conn.close()


# Nombre de fichiers de données actifs d'une table
def _active_files(conn: duckdb.DuckDBPyConnection) -> int:
    """Count the active data files of ``fact_table``.

    Args:
        conn: Connection attached to the ``db`` catalog.

    Returns:
        int: Number of data files of the current snapshot.
    """
    row = conn.execute(
        "SELECT file_count FROM ducklake_table_info('db')"
        " WHERE table_name = 'fact_table'"
    ).fetchone()
    return int(row[0])


# Test que la fusion par défaut réduit réellement le nombre de fichiers
def test_merge_files_default_reduces_file_count(
    fragmented_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that merging with the default bounds combines the small files.

    The small files are the very ones to merge: no lower bound may exclude them,
    and the procedure result must be fully read for DuckLake to apply the merge.

    Args:
        fragmented_conn: Catalog whose table is split into six small files.
    """
    assert _active_files(fragmented_conn) == 6
    maint = DuckLakeMaintenance(fragmented_conn)

    _, _, processed, created = maint.merge_files()

    assert (processed, created) == (6, 1)
    assert _active_files(fragmented_conn) == 1
    assert fragmented_conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0] == (
        6000
    )


# Test qu'un seuil bas exclut les fichiers plus petits que lui
def test_merge_files_min_file_size_excludes_smaller_files(
    fragmented_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that min_file_size is a lower bound: smaller files are left alone.

    Args:
        fragmented_conn: Catalog whose table is split into six small files.
    """
    maint = DuckLakeMaintenance(fragmented_conn)

    _, _, processed, _ = maint.merge_files(min_file_size=10**12)

    assert processed == 0
    assert _active_files(fragmented_conn) == 6


# Test que la borne haute par défaut est la taille cible du catalogue
def test_merge_files_default_max_is_target_file_size(
    fragmented_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that the default upper bound follows the catalog's target_file_size.

    Args:
        fragmented_conn: Catalog whose table is split into six small files.
    """
    maint = DuckLakeMaintenance(fragmented_conn)
    fragmented_conn.execute("CALL db.set_option('target_file_size', '2MB')")
    assert maint._target_file_size() == 2_000_000


# Test que compact fusionne les petits fichiers après une écriture
def test_compact_merges_small_files(
    fragmented_conn: duckdb.DuckDBPyConnection,
) -> None:
    """Test that the post-write compaction actually merges the small files.

    Args:
        fragmented_conn: Catalog whose table is split into six small files.
    """
    counters = DuckLakeMaintenance(fragmented_conn).compact()
    assert counters["merge_files_processed"] == 6
    assert _active_files(fragmented_conn) == 1


# Test que les procédures à l'échelle du catalogue refusent un schéma positionnel
def test_catalog_wide_procedures_take_no_schema(maint: Any) -> None:
    """Test that expire_snapshots/cleanup_files reject a positional schema.

    They act on the whole catalog: a schema argument would be misleading.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    with pytest.raises(TypeError):
        maint.expire_snapshots("main")
    with pytest.raises(TypeError):
        maint.cleanup_files("main")
