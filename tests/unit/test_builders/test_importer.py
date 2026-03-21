# Importation des modules
# Modules de base
import os
import warnings
# DuckDB
import duckdb
# Module de tests
import pytest
# Modules à tester
from dashboard_template_database.builders import ParquetExporter, ParquetImporter
from dashboard_template_database.builders import DuckdbTablesBuilder


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------

# Fixture fournissant un répertoire d'export complet (fact + metadata + dims)
@pytest.fixture
def exported_schema_dir(built_duckdb_schema, tmp_path):
    """Export a full schema to tmp_path and return the export directory path."""
    output_dir = str(tmp_path / "export")
    os.makedirs(output_dir, exist_ok=True)
    exporter = ParquetExporter(connection=built_duckdb_schema)
    exporter.export_schema(
        fact_table='fact_table',
        output_dir=output_dir,
        include_metadata=True,
        include_dimensions=True,
    )
    return output_dir


# Fixture fournissant un répertoire d'export avec partitionnement sur 'id'
@pytest.fixture
def exported_schema_dir_partitioned(built_duckdb_schema, tmp_path):
    """Export a schema with Hive partitioning on 'id' and return the directory."""
    output_dir = str(tmp_path / "export_partitioned")
    os.makedirs(output_dir, exist_ok=True)
    exporter = ParquetExporter(connection=built_duckdb_schema)
    exporter.export_schema(
        fact_table='fact_table',
        output_dir=output_dir,
        partition_keys=['id'],
        include_metadata=True,
        include_dimensions=True,
    )
    return output_dir


# ---------------------------------------------------------------------------
# Tests d'initialisation
# ---------------------------------------------------------------------------

# Test de l'initialisation en mémoire (sans connexion ni chemin)
def test_init_in_memory(exported_schema_dir):
    """Test that ParquetImporter creates an in-memory connection when no args are given."""
    importer = ParquetImporter(exported_schema_dir)
    assert isinstance(importer.conn, duckdb.DuckDBPyConnection)


# Test de l'initialisation avec une connexion existante
def test_init_with_existing_connection(exported_schema_dir):
    """Test that ParquetImporter reuses a provided connection."""
    conn = duckdb.connect(':memory:')
    importer = ParquetImporter(exported_schema_dir, connection=conn)
    assert importer.conn is conn


# Test de l'initialisation avec un chemin de fichier DuckDB
def test_init_with_path(exported_schema_dir, tmp_path):
    """Test that ParquetImporter opens a file-based DuckDB connection from a path."""
    db_path = str(tmp_path / "restored.duckdb")
    importer = ParquetImporter(exported_schema_dir, path=db_path)
    assert isinstance(importer.conn, duckdb.DuckDBPyConnection)


# Test que le répertoire d'entrée est normalisé (suppression du slash final)
def test_init_strips_trailing_slash(exported_schema_dir):
    """Test that trailing slashes are stripped from input_dir."""
    importer = ParquetImporter(exported_schema_dir + "/")
    assert not importer.input_dir.endswith('/')
    assert not importer.input_dir.endswith('\\')


# ---------------------------------------------------------------------------
# Tests de load_table — fichier plat
# ---------------------------------------------------------------------------

# Test du chargement d'un fichier Parquet plat (table de métadonnées)
def test_load_table_flat_metadata(exported_schema_dir, tmp_path):
    """Test that load_table correctly loads a flat Parquet file into DuckDB."""
    importer = ParquetImporter(exported_schema_dir)
    metadata_path = os.path.join(exported_schema_dir, 'metadata.parquet')

    importer.load_table('metadata', metadata_path)

    tables = [row[0] for row in importer.conn.execute("SHOW TABLES").fetchall()]
    assert 'metadata' in tables


# Test que le nombre de lignes est préservé après chargement
def test_load_table_flat_row_count(exported_schema_dir):
    """Test that the loaded table has the same row count as the original."""
    importer = ParquetImporter(exported_schema_dir)
    metadata_path = os.path.join(exported_schema_dir, 'metadata.parquet')
    importer.load_table('metadata', metadata_path)

    count = importer.conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    # Le schéma sample_df a 6 colonnes → 6 lignes dans metadata
    assert count == 6


# Test que ValueError est levé si la table existe sans overwrite=True
def test_load_table_already_exists_raises(exported_schema_dir):
    """Test that ValueError is raised when loading into an existing table without overwrite."""
    importer = ParquetImporter(exported_schema_dir)
    metadata_path = os.path.join(exported_schema_dir, 'metadata.parquet')
    importer.load_table('metadata', metadata_path)

    with pytest.raises(ValueError, match="existe déjà"):
        importer.load_table('metadata', metadata_path)


# Test que overwrite=True permet de remplacer une table existante
def test_load_table_overwrite(exported_schema_dir):
    """Test that overwrite=True replaces an existing table without error."""
    importer = ParquetImporter(exported_schema_dir)
    metadata_path = os.path.join(exported_schema_dir, 'metadata.parquet')
    importer.load_table('metadata', metadata_path)

    # Deuxième chargement avec overwrite=True ne doit pas lever d'exception
    importer.load_table('metadata', metadata_path, overwrite=True)
    count = importer.conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    assert count == 6


# ---------------------------------------------------------------------------
# Tests de load_table — PRIMARY KEY
# ---------------------------------------------------------------------------

# Test que la contrainte PRIMARY KEY est reconstituée correctement
def test_load_table_with_primary_key(exported_schema_dir):
    """Test that load_table enforces a PRIMARY KEY when primary_key_columns is provided."""
    importer = ParquetImporter(exported_schema_dir)
    metadata_path = os.path.join(exported_schema_dir, 'metadata.parquet')

    # Chargement avec PRIMARY KEY sur 'name'
    importer.load_table('metadata', metadata_path, primary_key_columns=['name'])

    # Tentative d'insertion d'un doublon : doit échouer (violation de la contrainte PRIMARY KEY)
    with pytest.raises(Exception):
        importer.conn.execute(
            "INSERT INTO metadata VALUES ('id', 'label', 'int', 'INTEGER', false, true)"
        )


# ---------------------------------------------------------------------------
# Tests de load_table — partitionnement Hive
# ---------------------------------------------------------------------------

# Test du chargement d'une fact table partitionnée
def test_load_table_partitioned(exported_schema_dir_partitioned):
    """Test that load_table reads a Hive-partitioned directory correctly."""
    importer = ParquetImporter(exported_schema_dir_partitioned)
    fact_path = os.path.join(exported_schema_dir_partitioned, 'fact_table')

    importer.load_table('fact_table', fact_path, hive_partitioning=True)

    tables = [row[0] for row in importer.conn.execute("SHOW TABLES").fetchall()]
    assert 'fact_table' in tables


# Test que le nombre de lignes de la fact table partitionnée est correct
def test_load_table_partitioned_row_count(exported_schema_dir_partitioned):
    """Test that the row count is preserved after loading a partitioned fact table."""
    importer = ParquetImporter(exported_schema_dir_partitioned)
    fact_path = os.path.join(exported_schema_dir_partitioned, 'fact_table')
    importer.load_table('fact_table', fact_path, hive_partitioning=True)

    count = importer.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    # sample_df contient 5 lignes
    assert count == 5


# ---------------------------------------------------------------------------
# Tests de load_schema — round-trip complet
# ---------------------------------------------------------------------------

# Test du round-trip complet : construction → export → import
def test_load_schema_full_round_trip(exported_schema_dir):
    """Test the full round-trip: build → export → import, checking table names."""
    importer = ParquetImporter(exported_schema_dir)
    importer.load_schema()

    loaded_tables = sorted(
        row[0] for row in importer.conn.execute("SHOW TABLES").fetchall()
    )
    # Les tables attendues : fact_table, metadata, dim_category, dim_status
    assert 'fact_table' in loaded_tables
    assert 'metadata' in loaded_tables
    # Au moins une table de dimension doit être présente
    dim_tables = [t for t in loaded_tables if t.startswith('dim_')]
    assert len(dim_tables) > 0


# Test que les données de la fact table sont intactes après round-trip
def test_load_schema_fact_table_row_count(exported_schema_dir):
    """Test that the fact table row count is preserved after load_schema."""
    importer = ParquetImporter(exported_schema_dir)
    importer.load_schema()

    count = importer.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count == 5


# Test que les données de la metadata table sont intactes après round-trip
def test_load_schema_metadata_row_count(exported_schema_dir):
    """Test that the metadata table row count is preserved after load_schema."""
    importer = ParquetImporter(exported_schema_dir)
    importer.load_schema()

    count = importer.conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    assert count == 6


# ---------------------------------------------------------------------------
# Tests de load_schema — partitionnement Hive
# ---------------------------------------------------------------------------

# Test du round-trip avec fact table partitionnée
def test_load_schema_partitioned(exported_schema_dir_partitioned):
    """Test that load_schema correctly handles a Hive-partitioned fact table."""
    importer = ParquetImporter(exported_schema_dir_partitioned)
    importer.load_schema()

    count = importer.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count == 5


# ---------------------------------------------------------------------------
# Tests de load_schema — cas limites
# ---------------------------------------------------------------------------

# Test que l'absence de metadata génère un avertissement et ne bloque pas le chargement
def test_load_schema_missing_metadata_warns(tmp_path):
    """Test that a missing metadata file triggers a warning but does not raise."""
    # Création d'un répertoire d'export minimal (fact table uniquement)
    output_dir = str(tmp_path / "no_metadata")
    os.makedirs(output_dir, exist_ok=True)

    # Export d'une simple table sans metadata
    conn = duckdb.connect(':memory:')
    conn.execute("CREATE TABLE fact_table AS SELECT 1 AS id, 'FR' AS country")
    conn.execute(f"COPY fact_table TO '{output_dir}/fact_table' (FORMAT PARQUET)")

    importer = ParquetImporter(output_dir)
    # Ne doit pas lever d'exception
    importer.load_schema()

    tables = [row[0] for row in importer.conn.execute("SHOW TABLES").fetchall()]
    assert 'fact_table' in tables


# Test que l'absence de tables de dimension génère un avertissement (non bloquant)
def test_load_schema_missing_dim_tables_warns(tmp_path, caplog):
    """Test that missing dimension tables trigger a warning."""
    output_dir = str(tmp_path / "no_dims")
    os.makedirs(output_dir, exist_ok=True)

    # Export minimal : metadata + fact table, sans dimensions
    conn = duckdb.connect(':memory:')
    conn.execute("CREATE TABLE fact_table AS SELECT 1 AS id, 42.0 AS value")
    conn.execute(
        "CREATE TABLE metadata (name VARCHAR PRIMARY KEY, label VARCHAR, "
        "python_type VARCHAR, sql_type VARCHAR, is_categorical BOOLEAN, is_primary_key BOOLEAN)"
    )
    conn.execute("COPY fact_table TO '{}' (FORMAT PARQUET)".format(
        os.path.join(output_dir, 'fact_table')
    ))
    conn.execute("COPY metadata TO '{}' (FORMAT PARQUET)".format(
        os.path.join(output_dir, 'metadata.parquet')
    ))

    importer = ParquetImporter(output_dir)
    importer.load_schema()

    tables = [row[0] for row in importer.conn.execute("SHOW TABLES").fetchall()]
    assert 'fact_table' in tables


# Test que load_schema lève ValueError si une table existe sans overwrite=True
def test_load_schema_overwrite_false_raises(exported_schema_dir):
    """Test that a second load_schema() without overwrite=True raises ValueError."""
    importer = ParquetImporter(exported_schema_dir)
    importer.load_schema()

    with pytest.raises(ValueError, match="existe déjà"):
        importer.load_schema(overwrite=False)


# Test que load_schema avec overwrite=True réussit au second appel
def test_load_schema_overwrite_true(exported_schema_dir):
    """Test that a second load_schema() with overwrite=True succeeds."""
    importer = ParquetImporter(exported_schema_dir)
    importer.load_schema()
    importer.load_schema(overwrite=True)

    count = importer.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count == 5


# Test que FileNotFoundError est levé si le répertoire d'entrée n'existe pas
def test_load_schema_invalid_input_dir(tmp_path):
    """Test that FileNotFoundError is raised when input_dir does not exist."""
    importer = ParquetImporter(str(tmp_path / "nonexistent_dir"))
    with pytest.raises(FileNotFoundError):
        importer.load_schema()


# ---------------------------------------------------------------------------
# Tests de reconstruction de la contrainte PRIMARY KEY
# ---------------------------------------------------------------------------

# Test que la PRIMARY KEY est reconstituée depuis la metadata et appliquée sur la fact table
def test_load_schema_primary_key_reconstructed(exported_schema_dir):
    """Test that the PRIMARY KEY constraint is reconstructed from metadata on the fact table."""
    importer = ParquetImporter(exported_schema_dir)
    importer.load_schema()

    # Tentative d'insertion d'une ligne avec un 'id' déjà présent (1 à 5 dans sample_df)
    # La contrainte PRIMARY KEY doit bloquer l'insertion
    with pytest.raises(Exception):
        importer.conn.execute(
            "INSERT INTO fact_table SELECT * FROM fact_table WHERE id = 1"
        )
