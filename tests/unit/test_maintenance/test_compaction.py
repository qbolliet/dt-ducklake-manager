# Importation des modules
# Modules de base
import os

# DuckDB
import duckdb

# Module de tests
import pytest

# Modules à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.maintenance import DuckLakeMaintenance

# ---------------------------------------------------------------------------
# Fonctions auxiliaires
# ---------------------------------------------------------------------------


def _ducklake_available() -> bool:
    """Vérifie si l'extension DuckLake est disponible dans l'environnement de test."""
    try:
        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        conn.close()
        return True
    except Exception:
        return False


# Marqueur appliqué à l'ensemble du module : tous les tests sont ignorés si
# l'extension ducklake n'est pas disponible dans l'environnement.
pytestmark = pytest.mark.skipif(
    not _ducklake_available(),
    reason="Extension ducklake non disponible dans cet environnement",
)


# ---------------------------------------------------------------------------
# Fixtures communes
# ---------------------------------------------------------------------------


# Initialisation d'un catalogue DuckLake temporaire avec une table de test
@pytest.fixture
def ducklake_conn(tmp_path):
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
def maint(ducklake_conn):
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
def test_init_default_alias(ducklake_conn):
    """Test that the default catalog_alias is 'db'.

    Args:
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, _ = ducklake_conn
    maint = DuckLakeMaintenance(conn)
    assert maint.catalog_alias == "db"


# Test de l'initialisation avec un alias personnalisé
def test_init_custom_catalog_alias(ducklake_conn):
    """Test that DuckLakeMaintenance stores a custom catalog alias.

    Args:
        ducklake_conn: Fixture providing (connection, table_name).
    """
    conn, _ = ducklake_conn
    maint = DuckLakeMaintenance(conn, catalog_alias="my_lake")
    assert maint.catalog_alias == "my_lake"


# ---------------------------------------------------------------------------
# Tests des méthodes individuelles
# ---------------------------------------------------------------------------


# Test que merge_files s'exécute sans erreur
def test_merge_files_executes_without_error(maint, ducklake_conn):
    """Test that merge_files completes without raising an exception.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    # Aucune exception ne doit être levée
    maint.merge_files("main", table)


# Test que rewrite_data_files s'exécute sans erreur
def test_rewrite_data_files_executes_without_error(maint, ducklake_conn):
    """Test that rewrite_data_files completes without raising an exception.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    maint.rewrite_data_files("main", table)


# Test que expire_snapshots s'exécute sans erreur avec la valeur par défaut
def test_expire_snapshots_default_days(maint):
    """Test that expire_snapshots with default older_than_days=30 does not raise.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    maint.expire_snapshots("main")


# Test que expire_snapshots accepte une valeur personnalisée de older_than_days
def test_expire_snapshots_custom_days(maint):
    """Test that expire_snapshots accepts a custom older_than_days value.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    maint.expire_snapshots("main", older_than_days=7)


# Test que cleanup_files s'exécute sans erreur
def test_cleanup_files_executes_without_error(maint):
    """Test that cleanup_files completes without raising an exception.

    Args:
        maint: DuckLakeMaintenance fixture.
    """
    maint.cleanup_files("main")


# ---------------------------------------------------------------------------
# Tests de full_maintenance
# ---------------------------------------------------------------------------


# Test que full_maintenance exécute toutes les étapes sans erreur
def test_full_maintenance_runs_all_steps(maint, ducklake_conn):
    """Test that full_maintenance calls all four maintenance procedures.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn
    # Aucune exception ne doit être levée et toutes les étapes doivent s'exécuter
    maint.full_maintenance("main", table)


# Test que full_maintenance continue après un échec partiel
def test_full_maintenance_continues_on_step_failure(maint, ducklake_conn):
    """Test that full_maintenance logs a warning and continues if one step fails.

    Args:
        maint: DuckLakeMaintenance fixture.
        ducklake_conn: Fixture providing (connection, table_name).
    """
    _, table = ducklake_conn

    # Compteur d'appels pour vérifier que les étapes suivantes ont bien été exécutées
    call_log: list = []

    original_merge = maint.merge_files
    original_rewrite = maint.rewrite_data_files
    original_expire = maint.expire_snapshots
    original_cleanup = maint.cleanup_files

    def failing_merge(schema, tbl):
        # Simulation d'un échec sur la première étape
        call_log.append("merge_files")
        raise RuntimeError("Échec simulé de merge_files")

    def tracking_rewrite(schema, tbl):
        call_log.append("rewrite_data_files")
        original_rewrite(schema, tbl)

    def tracking_expire(schema, older_than_days=30):
        call_log.append("expire_snapshots")
        original_expire(schema, older_than_days=older_than_days)

    def tracking_cleanup(schema):
        call_log.append("cleanup_files")
        original_cleanup(schema)

    maint.merge_files = failing_merge
    maint.rewrite_data_files = tracking_rewrite
    maint.expire_snapshots = tracking_expire
    maint.cleanup_files = tracking_cleanup

    # full_maintenance ne doit pas lever d'exception malgré l'échec de merge_files
    maint.full_maintenance("main", table)

    # Vérification que toutes les étapes ont bien été appelées malgré l'échec
    assert "merge_files" in call_log
    assert "rewrite_data_files" in call_log
    assert "expire_snapshots" in call_log
    assert "cleanup_files" in call_log
