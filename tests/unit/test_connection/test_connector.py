# Importation des modules
# Modules de base
import logging
import os
from pathlib import Path
from typing import Any

# DuckDB
import duckdb

# Module de tests
import pytest

# Module à tester
from dt_ducklake_manager.connection import CatalogType, DuckLakeConnector

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
# Fixture commune
# ---------------------------------------------------------------------------


# Initialisation des chemins de catalogue et de données temporaires
@pytest.fixture
def ducklake_paths(tmp_path: Path) -> tuple[str, str]:
    """Create a temporary DuckLake catalog and data directory.

    Args:
        tmp_path: pytest temporary directory.

    Returns:
        tuple: (catalog_path, data_path) as strings.
    """
    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    return catalog, data_dir


# ---------------------------------------------------------------------------
# Tests de connect()
# ---------------------------------------------------------------------------


# Test que connect() retourne une connexion DuckDB valide
def test_connect_returns_duckdb_connection(ducklake_paths: tuple[str, str]) -> None:
    """Test that connect() returns a valid DuckDBPyConnection.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir)
    conn = connector.connect()
    assert isinstance(conn, duckdb.DuckDBPyConnection)
    conn.close()


# Test que la connexion en lecture seule bloque les écritures
def test_connect_read_only_blocks_write(ducklake_paths: tuple[str, str]) -> None:
    """Test that a read-only connection raises an error on write operations.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
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


# Test que connect() active bien le schéma configuré via USE
def test_connect_activates_correct_schema(ducklake_paths: tuple[str, str]) -> None:
    """Test that the USE statement activates the configured schema.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir, schema="main")
    conn = connector.connect()
    # La création d'une table sans préfixe doit réussir (schéma activé)
    conn.execute("CREATE TABLE schema_check (id INTEGER)")
    row = conn.execute("SELECT COUNT(*) FROM schema_check").fetchone()
    assert row is not None
    assert row[0] == 0
    conn.close()


# Test que snapshot_version implique une connexion en lecture seule
def test_connect_snapshot_version_is_read_only(ducklake_paths: tuple[str, str]) -> None:
    """Test that SNAPSHOT_VERSION implies READ_ONLY and blocks writes.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
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


# Test que l'alias de catalogue personnalisé est bien utilisé
def test_connect_custom_catalog_alias(ducklake_paths: tuple[str, str]) -> None:
    """Test that a custom catalog_alias is used in the ATTACH statement.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir, catalog_alias="my_lake")
    conn = connector.connect()
    # Vérification que l'alias est accessible (DuckDB expose catalog_name dans
    # information_schema)
    result = conn.execute(
        "SELECT catalog_name FROM information_schema.schemata WHERE catalog_name ="
        "'my_lake'"
    ).fetchall()
    assert len(result) > 0
    conn.close()


# ---------------------------------------------------------------------------
# Tests de attach()
# ---------------------------------------------------------------------------


# Test que attach() fonctionne sur une connexion DuckDB existante
def test_attach_on_existing_connection(ducklake_paths: tuple[str, str]) -> None:
    """Test that attach() works on an already-open DuckDB connection.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir)

    # Création d'une connexion existante avec l'extension déjà chargée
    existing_conn = duckdb.connect(":memory:")
    existing_conn.execute("INSTALL ducklake; LOAD ducklake;")

    # attach() ne doit pas lever d'exception et doit retourner la même connexion
    returned_conn = connector.attach(existing_conn)
    assert returned_conn is existing_conn

    # Vérification que le schéma est actif (création de table sans préfixe)
    returned_conn.execute("CREATE TABLE attach_check (val VARCHAR)")
    row = returned_conn.execute("SELECT COUNT(*) FROM attach_check").fetchone()
    assert row is not None
    assert row[0] == 0
    existing_conn.close()


# ---------------------------------------------------------------------------
# Tests de _build_attach_sql()
# ---------------------------------------------------------------------------


# Test de la construction de la clause ATTACH sans options spéciales
def test_build_attach_sql_default(ducklake_paths: tuple[str, str]) -> None:
    """Test that default ATTACH SQL contains DATA_PATH but no READ_ONLY.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir)
    sql = connector._build_attach_sql()
    assert "DATA_PATH" in sql
    assert "READ_ONLY" not in sql
    assert "SNAPSHOT_VERSION" not in sql


# Test de la construction de la clause ATTACH avec READ_ONLY
def test_build_attach_sql_read_only(ducklake_paths: tuple[str, str]) -> None:
    """Test that read_only=True adds READ_ONLY to the ATTACH SQL.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir, read_only=True)
    sql = connector._build_attach_sql()
    assert "READ_ONLY" in sql


# Test de la construction de la clause ATTACH avec SNAPSHOT_VERSION
def test_build_attach_sql_snapshot_version(ducklake_paths: tuple[str, str]) -> None:
    """Test that snapshot_version adds SNAPSHOT_VERSION to the ATTACH SQL.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir, snapshot_version=3)
    sql = connector._build_attach_sql()
    assert "SNAPSHOT_VERSION 3" in sql
    # READ_ONLY ne doit pas être ajouté en doublon (implicite via SNAPSHOT_VERSION)
    assert sql.count("READ_ONLY") == 0


# Test de la construction de la clause ATTACH avec SNAPSHOT_TIME
def test_build_attach_sql_snapshot_time(ducklake_paths: tuple[str, str]) -> None:
    """Test that snapshot_time adds SNAPSHOT_TIME to the ATTACH SQL.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(
        catalog, data_dir, snapshot_time="2025-01-01 00:00:00"
    )
    sql = connector._build_attach_sql()
    assert "SNAPSHOT_TIME '2025-01-01 00:00:00'" in sql


# ---------------------------------------------------------------------------
# Tests du backend PostgreSQL (construction, sans serveur Postgres requis)
# ---------------------------------------------------------------------------


# Connexion factice enregistrant le SQL exécuté (test de non-fuite du mot de passe)
class _RecordingConn:
    """Minimal stand-in for a DuckDB connection capturing executed statements."""

    def __init__(self) -> None:
        self.executed: list[str] = []

    def execute(self, sql: str) -> "_RecordingConn":
        self.executed.append(sql)
        return self


# Test que le backend par défaut est bien DuckDB (rétrocompatibilité)
def test_default_catalog_type_is_duckdb(ducklake_paths: tuple[str, str]) -> None:
    """Test that the default constructor uses the DuckDB catalog backend.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths
    connector = DuckLakeConnector(catalog, data_dir)
    assert connector.catalog_type == CatalogType.DUCKDB
    assert connector.meta_secret is None
    # Aucun secret n'est préparé pour le backend fichier
    assert connector._secret_sql is None


# Test que from_postgres configure correctement les attributs du connecteur
def test_from_postgres_sets_attributes() -> None:
    """Test that from_postgres configures the connector for the Postgres backend."""
    connector = DuckLakeConnector.from_postgres(
        "data/", dbname="ducklake", host="localhost", user="app", password="secret"
    )
    assert connector.catalog_type == CatalogType.POSTGRES
    # Le nom de la base figure dans la cible ATTACH (non sensible)
    assert connector.catalog_path == "postgres:dbname=ducklake"
    assert connector.meta_secret == "ducklake_pg_secret"


# Test que from_postgres avec identifiants prépare un secret de session
def test_from_postgres_with_credentials_builds_secret_sql() -> None:
    """Test that inline credentials produce a CREATE SECRET statement."""
    connector = DuckLakeConnector.from_postgres(
        "data/", dbname="ducklake", host="localhost", user="app", password="secret"
    )
    assert connector._secret_sql is not None
    assert "CREATE OR REPLACE SECRET ducklake_pg_secret" in connector._secret_sql
    assert "TYPE postgres" in connector._secret_sql
    assert "DATABASE 'ducklake'" in connector._secret_sql
    assert "HOST 'localhost'" in connector._secret_sql
    assert "USER 'app'" in connector._secret_sql
    assert "PASSWORD 'secret'" in connector._secret_sql


# Test que from_postgres avec meta_secret ne crée pas de secret
def test_from_postgres_with_meta_secret_skips_secret_creation() -> None:
    """Test that referencing an external secret avoids any CREATE SECRET."""
    connector = DuckLakeConnector.from_postgres(
        "data/", dbname="ducklake", meta_secret="external_secret"
    )
    assert connector.meta_secret == "external_secret"
    # Aucun identifiant ne transite par Python : pas de SQL de création de secret
    assert connector._secret_sql is None


# Test que les champs d'identifiants non fournis sont omis du secret
def test_from_postgres_omits_missing_credential_fields() -> None:
    """Test that None credential fields are omitted (libpq env fallback)."""
    connector = DuckLakeConnector.from_postgres("data/", dbname="ducklake")
    assert connector._secret_sql is not None
    # Seuls TYPE et DATABASE sont présents ; le reste retombe sur l'environnement
    assert "DATABASE 'ducklake'" in connector._secret_sql
    assert "HOST" not in connector._secret_sql
    assert "USER" not in connector._secret_sql
    assert "PASSWORD" not in connector._secret_sql


# Test de l'échappement des apostrophes dans les valeurs du secret
def test_build_postgres_secret_sql_escapes_quotes() -> None:
    """Test that single quotes in credentials are escaped in the secret SQL."""
    connector = DuckLakeConnector.from_postgres(
        "data/", dbname="ducklake", user="o'brien", password="pa'ss"
    )
    assert connector._secret_sql is not None
    assert "USER 'o''brien'" in connector._secret_sql
    assert "PASSWORD 'pa''ss'" in connector._secret_sql


# Test de la construction de la clause ATTACH pour le backend Postgres
def test_build_attach_sql_postgres() -> None:
    """Test that the Postgres ATTACH SQL targets postgres and references the secret."""
    connector = DuckLakeConnector.from_postgres(
        "data/files/", dbname="ducklake", host="localhost"
    )
    sql = connector._build_attach_sql()
    assert "ducklake:postgres:dbname=ducklake" in sql
    assert "META_SECRET 'ducklake_pg_secret'" in sql
    assert "DATA_PATH 'data/files/'" in sql
    assert "READ_ONLY" not in sql


# Test de la clause ATTACH Postgres en lecture seule
def test_build_attach_sql_postgres_read_only() -> None:
    """Test that read_only=True adds READ_ONLY to the Postgres ATTACH SQL."""
    connector = DuckLakeConnector.from_postgres(
        "data/", dbname="ducklake", meta_secret="external_secret", read_only=True
    )
    sql = connector._build_attach_sql()
    assert "ducklake:postgres:dbname=ducklake" in sql
    assert "META_SECRET 'external_secret'" in sql
    assert "READ_ONLY" in sql


# Test que la création du secret ne divulgue jamais le mot de passe dans les logs
def test_apply_secret_does_not_log_password(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that _apply_secret executes the secret SQL but never logs the password.

    Args:
        caplog: pytest fixture capturing log records.
    """
    connector = DuckLakeConnector.from_postgres(
        "data/",
        dbname="ducklake",
        host="localhost",
        user="app",
        password="sup3rs3cret",
    )
    recording_conn = _RecordingConn()
    with caplog.at_level(logging.INFO):
        connector._apply_secret(recording_conn)  # type: ignore[arg-type]

    # Le mot de passe figure bien dans le SQL exécuté (création du secret)...
    assert any("sup3rs3cret" in stmt for stmt in recording_conn.executed)
    # ... mais ne doit jamais apparaître dans les logs ; seul le nom est journalisé.
    assert "sup3rs3cret" not in caplog.text
    assert "ducklake_pg_secret" in caplog.text


# ---------------------------------------------------------------------------
# Tests du support multi-schémas
# ---------------------------------------------------------------------------


# Test que connect() crée puis active un schéma nommé inexistant
def test_connect_creates_named_schema(ducklake_paths: tuple[str, str]) -> None:
    """Test that connect() creates a non-existent named schema and activates it.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths

    # Connexion ciblant un schéma 'predictions' qui n'existe pas encore
    conn = DuckLakeConnector(catalog, data_dir, schema="predictions").connect()

    # Le schéma a bien été créé dans le catalogue
    schemas = [
        row[0]
        for row in conn.execute(
            "SELECT schema_name FROM information_schema.schemata"
        ).fetchall()
    ]
    assert "predictions" in schemas

    # Le schéma actif permet d'écrire des tables non qualifiées dans 'predictions'
    conn.execute("CREATE TABLE t AS SELECT 1 AS x")
    located = conn.execute(
        "SELECT table_schema FROM information_schema.tables WHERE table_name = 't'"
    ).fetchone()
    assert located is not None and located[0] == "predictions"
    conn.close()


# Test que deux schémas peuvent coexister dans un même catalogue via deux connexions
def test_two_named_schemas_in_one_catalog(ducklake_paths: tuple[str, str]) -> None:
    """Test that two named schemas coexist in a single catalog.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths

    # Création de deux schémas dans le même catalogue
    conn_pred = DuckLakeConnector(catalog, data_dir, schema="predictions").connect()
    conn_pred.execute("CREATE TABLE fact_table AS SELECT 1 AS id")
    conn_pred.close()

    conn_shap = DuckLakeConnector(catalog, data_dir, schema="shapley").connect()
    conn_shap.execute("CREATE TABLE fact_table AS SELECT 2 AS id")

    # Les deux schémas existent et portent chacun leur fact_table
    schemas = [
        row[0]
        for row in conn_shap.execute(
            "SELECT table_schema FROM information_schema.tables "
            "WHERE table_name = 'fact_table' ORDER BY table_schema"
        ).fetchall()
    ]
    assert "predictions" in schemas
    assert "shapley" in schemas
    conn_shap.close()


# Test qu'une connexion en lecture seule ne crée pas de schéma
def test_read_only_does_not_create_schema(ducklake_paths: tuple[str, str]) -> None:
    """Test that a read-only connection does not attempt to create the schema.

    Args:
        ducklake_paths: Fixture providing (catalog_path, data_path).
    """
    catalog, data_dir = ducklake_paths

    # Création préalable du catalogue avec le schéma 'main' (lecture-écriture)
    rw_conn = DuckLakeConnector(catalog, data_dir).connect()
    rw_conn.execute("CREATE TABLE fact_table AS SELECT 1 AS id")
    rw_conn.close()

    # Connexion lecture seule sur 'main' : aucune création de schéma, USE seul
    ro_conn = DuckLakeConnector(catalog, data_dir, read_only=True).connect()
    row = ro_conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()
    assert row is not None
    assert row[0] == 1
    ro_conn.close()


# ---------------------------------------------------------------------------
# Test d'intégration de concurrence PostgreSQL (serveur réel requis)
# ---------------------------------------------------------------------------


# Lecture des paramètres de connexion Postgres depuis l'environnement de test
def _postgres_test_params() -> dict[str, Any] | None:
    """Read PostgreSQL test connection parameters from environment variables.

    Returns:
        Optional[dict]: Connection parameters, or None when host/dbname are unset.
    """
    host = os.environ.get("DUCKLAKE_TEST_PG_HOST")
    dbname = os.environ.get("DUCKLAKE_TEST_PG_DB")
    if not host or not dbname:
        return None
    port = os.environ.get("DUCKLAKE_TEST_PG_PORT")
    return {
        "host": host,
        "dbname": dbname,
        "port": int(port) if port else None,
        "user": os.environ.get("DUCKLAKE_TEST_PG_USER"),
        "password": os.environ.get("DUCKLAKE_TEST_PG_PASSWORD"),
    }


# Test que lecture et écriture concurrentes coexistent sur un catalogue Postgres
@pytest.mark.integration
@pytest.mark.skipif(
    _postgres_test_params() is None,
    reason="Paramètres de connexion PostgreSQL absents (DUCKLAKE_TEST_PG_*)",
)
def test_postgres_concurrent_read_write(tmp_path: Path) -> None:
    """Test that a read-only connection coexists with a writer on a Postgres catalog.

    Requires a running PostgreSQL server with an empty catalog database, configured
    via the ``DUCKLAKE_TEST_PG_*`` environment variables.

    Args:
        tmp_path: pytest temporary directory used as DATA_PATH.
    """
    import uuid

    params = _postgres_test_params()
    assert params is not None
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    # Nom de table unique pour éviter les collisions entre exécutions
    table = f"concurrency_check_{uuid.uuid4().hex[:8]}"

    # Connexion lecture-écriture : création et insertion
    rw_conn = DuckLakeConnector.from_postgres(data_dir, **params).connect()
    try:
        rw_conn.execute(f"CREATE TABLE {table} (id INTEGER)")
        rw_conn.execute(f"INSERT INTO {table} VALUES (1)")

        # Connexion lecture seule ouverte pendant que l'écrivain est actif :
        # impossible avec un catalogue fichier (verrou processus), permis ici.
        ro_conn = DuckLakeConnector.from_postgres(
            data_dir, read_only=True, **params
        ).connect()
        try:
            row = ro_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
            assert row is not None
            count = row[0]
            assert count == 1
            # Toute écriture côté lecture seule doit être refusée
            with pytest.raises(duckdb.Error):
                ro_conn.execute(f"INSERT INTO {table} VALUES (2)")
        finally:
            ro_conn.close()
    finally:
        # Nettoyage du catalogue partagé
        rw_conn.execute(f"DROP TABLE IF EXISTS {table}")
        rw_conn.close()
