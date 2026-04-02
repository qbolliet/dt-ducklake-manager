# Importation des modules
# Modules de base
import os
import tempfile
# Module de tests
import pytest
# DuckDB
import duckdb
# Module à tester
from dashboard_template_database.builders import DuckLakeConnector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ducklake_available() -> bool:
    """Vérifie si l'extension DuckLake est disponible dans l'environnement de test."""
    try:
        conn = duckdb.connect(':memory:')
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        conn.close()
        return True
    except Exception:
        return False


# Marqueur appliqué à l'ensemble du module : tous les tests sont ignorés si
# l'extension ducklake n'est pas disponible dans l'environnement.
pytestmark = pytest.mark.skipif(
    not _ducklake_available(),
    reason="Extension ducklake non disponible dans cet environnement"
)


# ---------------------------------------------------------------------------
# Fixture commune
# ---------------------------------------------------------------------------

@pytest.fixture
def ducklake_paths(tmp_path):
    """Crée un répertoire temporaire et retourne (catalog_path, data_path)."""
    catalog = str(tmp_path / 'test.ducklake')
    data_dir = str(tmp_path / 'data')
    os.makedirs(data_dir)
    return catalog, data_dir


# ---------------------------------------------------------------------------
# Tests de connect()
# ---------------------------------------------------------------------------

# Test que connect() retourne bien une connexion DuckDB
def test_connect_returns_duckdb_connection(ducklake_paths):
    """Test that connect() returns a valid DuckDBPyConnection."""
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir)
    conn = connector.connect()
    assert isinstance(conn, duckdb.DuckDBPyConnection)
    conn.close()


# Test que la connexion en lecture seule bloque les écritures
def test_connect_read_only_blocks_write(ducklake_paths):
    """Test that a read-only connection raises an error on write operations."""
    catalog, data_dir = ducklake_paths

    # Création initiale du catalogue avec une connexion lecture-écriture
    rw_conn = DuckLakeConnector(catalog, data_dir).connect()
    rw_conn.execute("CREATE TABLE test_ro (id INTEGER)")
    rw_conn.close()

    # Réouverture en lecture seule : toute écriture doit lever une erreur
    ro_conn = DuckLakeConnector(catalog, data_dir, read_only=True).connect()
    with pytest.raises(duckdb.Error):
        ro_conn.execute("INSERT INTO test_ro VALUES (1)")
    ro_conn.close()


# Test que connect() active bien le bon schéma via USE
def test_connect_activates_correct_schema(ducklake_paths):
    """Test that the USE statement activates the configured schema."""
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir, schema='main')
    conn = connector.connect()
    # La création d'une table sans préfixe doit réussir (schéma activé)
    conn.execute("CREATE TABLE schema_check (id INTEGER)")
    result = conn.execute("SELECT COUNT(*) FROM schema_check").fetchone()[0]
    assert result == 0
    conn.close()


# Test que snapshot_version implique READ_ONLY
def test_connect_snapshot_version_is_read_only(ducklake_paths):
    """Test that SNAPSHOT_VERSION implies READ_ONLY and blocks writes."""
    catalog, data_dir = ducklake_paths

    # Création d'un snapshot initial
    rw_conn = DuckLakeConnector(catalog, data_dir).connect()
    rw_conn.execute("CREATE TABLE snap_test (id INTEGER)")
    rw_conn.close()

    # Réouverture sur le snapshot 1 : toute écriture doit être refusée
    snap_conn = DuckLakeConnector(catalog, data_dir, snapshot_version=1).connect()
    with pytest.raises(duckdb.Error):
        snap_conn.execute("INSERT INTO snap_test VALUES (1)")
    snap_conn.close()


# Test que catalog_alias personnalisé est bien utilisé
def test_connect_custom_catalog_alias(ducklake_paths):
    """Test that a custom catalog_alias is used in the ATTACH statement."""
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir, catalog_alias='my_lake')
    conn = connector.connect()
    # Vérification que l'alias est accessible
    result = conn.execute(
        "SELECT database_name FROM information_schema.schemata WHERE database_name = 'my_lake'"
    ).fetchall()
    assert len(result) > 0
    conn.close()


# ---------------------------------------------------------------------------
# Tests de attach()
# ---------------------------------------------------------------------------

# Test que attach() fonctionne sur une connexion existante
def test_attach_on_existing_connection(ducklake_paths):
    """Test that attach() works on an already-open DuckDB connection."""
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir)

    # Création d'une connexion existante avec l'extension déjà chargée
    existing_conn = duckdb.connect(':memory:')
    existing_conn.execute("INSTALL ducklake; LOAD ducklake;")

    # attach() ne doit pas lever d'exception et doit activer le schéma
    returned_conn = connector.attach(existing_conn)
    assert returned_conn is existing_conn

    # Vérification que le schéma est actif (création de table sans préfixe)
    returned_conn.execute("CREATE TABLE attach_check (val VARCHAR)")
    count = returned_conn.execute("SELECT COUNT(*) FROM attach_check").fetchone()[0]
    assert count == 0
    existing_conn.close()


# ---------------------------------------------------------------------------
# Tests de _build_attach_sql()
# ---------------------------------------------------------------------------

# Test de la construction de la clause ATTACH sans options spéciales
def test_build_attach_sql_default(ducklake_paths):
    """Test that default ATTACH SQL contains DATA_PATH but no READ_ONLY."""
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir)
    sql = connector._build_attach_sql()
    assert "DATA_PATH" in sql
    assert "READ_ONLY" not in sql
    assert "SNAPSHOT_VERSION" not in sql


# Test de la construction de la clause ATTACH avec READ_ONLY
def test_build_attach_sql_read_only(ducklake_paths):
    """Test that read_only=True adds READ_ONLY to the ATTACH SQL."""
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir, read_only=True)
    sql = connector._build_attach_sql()
    assert "READ_ONLY" in sql


# Test de la construction de la clause ATTACH avec SNAPSHOT_VERSION
def test_build_attach_sql_snapshot_version(ducklake_paths):
    """Test that snapshot_version adds SNAPSHOT_VERSION to the ATTACH SQL."""
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir, snapshot_version=3)
    sql = connector._build_attach_sql()
    assert "SNAPSHOT_VERSION 3" in sql
    # READ_ONLY ne doit pas être ajouté en doublon (implicite via SNAPSHOT_VERSION)
    assert sql.count("READ_ONLY") == 0


# Test de la construction de la clause ATTACH avec SNAPSHOT_TIME
def test_build_attach_sql_snapshot_time(ducklake_paths):
    """Test that snapshot_time adds SNAPSHOT_TIME to the ATTACH SQL."""
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir, snapshot_time='2025-01-01 00:00:00')
    sql = connector._build_attach_sql()
    assert "SNAPSHOT_TIME '2025-01-01 00:00:00'" in sql
