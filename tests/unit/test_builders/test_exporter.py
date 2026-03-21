# Importation des modules
# Modules de base
import os
import warnings
# DuckDB
import duckdb
# Module de tests
import pytest
# Module à tester
from dashboard_template_database.builders import ParquetExporter
from dashboard_template_database.builders import DuckdbTablesBuilder


# ---------------------------------------------------------------------------
# Fixtures locales
# ---------------------------------------------------------------------------

# Fixture fournissant un exporteur connecté à une base en mémoire vide
@pytest.fixture
def exporter_empty():
    """ParquetExporter connected to an empty in-memory DuckDB database."""
    return ParquetExporter()


# Fixture fournissant un exporteur connecté à une base avec un schéma complet
@pytest.fixture
def exporter_with_schema(built_duckdb_schema):
    """ParquetExporter reusing the connection from the built_duckdb_schema fixture."""
    return ParquetExporter(connection=built_duckdb_schema)


# ---------------------------------------------------------------------------
# Tests d'initialisation
# ---------------------------------------------------------------------------

# Test de l'initialisation en mémoire (sans argument)
def test_init_in_memory():
    """Test that ParquetExporter creates an in-memory connection when no args are given."""
    exporter = ParquetExporter()
    assert isinstance(exporter.conn, duckdb.DuckDBPyConnection)


# Test de l'initialisation avec une connexion existante
def test_init_with_existing_connection(built_duckdb_schema):
    """Test that ParquetExporter reuses a provided connection."""
    exporter = ParquetExporter(connection=built_duckdb_schema)
    assert exporter.conn is built_duckdb_schema


# Test de l'initialisation avec un chemin de fichier
def test_init_with_path(tmp_path):
    """Test that ParquetExporter opens a file-based DuckDB connection from a path."""
    db_path = str(tmp_path / "test.duckdb")
    exporter = ParquetExporter(path=db_path)
    assert isinstance(exporter.conn, duckdb.DuckDBPyConnection)


# ---------------------------------------------------------------------------
# Tests de export_fact_table — export sans partitionnement
# ---------------------------------------------------------------------------

# Test de l'export à plat d'une table vers un fichier Parquet unique
def test_export_fact_table_flat(exporter_with_schema, tmp_path):
    """Test that export_fact_table creates a single Parquet file without partitioning."""
    output_path = str(tmp_path / "fact.parquet")
    exporter_with_schema.export_fact_table('fact_table', output_path)

    # Le fichier doit exister
    assert os.path.exists(output_path)


# Test que le fichier Parquet exporté est lisible et contient des données
def test_export_fact_table_flat_readable(exporter_with_schema, tmp_path):
    """Test that the exported flat Parquet file is readable and contains data."""
    import pandas as pd

    output_path = str(tmp_path / "fact.parquet")
    exporter_with_schema.export_fact_table('fact_table', output_path)

    df = pd.read_parquet(output_path)
    assert len(df) > 0


# ---------------------------------------------------------------------------
# Tests de export_fact_table — export avec partitionnement
# ---------------------------------------------------------------------------

# Test de l'export avec partitionnement sur une colonne catégorielle
def test_export_fact_table_partitioned(exporter_with_schema, tmp_path):
    """Test that export_fact_table creates partitioned subdirectories."""
    output_dir = str(tmp_path / "fact_partitioned")
    # La colonne 'category' est présente dans la fact_table (après remplacement par des IDs)
    # On partitionne sur la colonne 'id' qui est numérique et petite
    exporter_with_schema.export_fact_table('fact_table', output_dir, partition_keys=['id'])

    # Le répertoire de sortie doit exister
    assert os.path.isdir(output_dir)
    # DuckDB crée des sous-répertoires col=val/
    subdirs = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]
    assert len(subdirs) > 0


# Test que les sous-répertoires Hive ont le bon format (col=val)
def test_export_fact_table_partitioned_hive_format(exporter_with_schema, tmp_path):
    """Test that partitioned export follows the Hive partitioning naming convention."""
    output_dir = str(tmp_path / "fact_hive")
    exporter_with_schema.export_fact_table('fact_table', output_dir, partition_keys=['id'])

    subdirs = [
        d for d in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, d))
    ]
    # Chaque sous-répertoire doit avoir le format 'id=<valeur>'
    assert all(d.startswith('id=') for d in subdirs)


# ---------------------------------------------------------------------------
# Tests des cas d'erreur de export_fact_table
# ---------------------------------------------------------------------------

# Test que ValueError est levé pour une table inexistante
def test_export_fact_table_nonexistent_table(exporter_with_schema, tmp_path):
    """Test that ValueError is raised when the table does not exist."""
    output_path = str(tmp_path / "output.parquet")
    with pytest.raises(ValueError, match="n'existe pas"):
        exporter_with_schema.export_fact_table('nonexistent_table', output_path)


# Test que ValueError est levé pour une clé de partition invalide
def test_export_fact_table_invalid_partition_key(exporter_with_schema, tmp_path):
    """Test that ValueError is raised when a partition key is not a column."""
    output_dir = str(tmp_path / "bad_partition")
    with pytest.raises(ValueError, match="clés de partitionnement"):
        exporter_with_schema.export_fact_table(
            'fact_table', output_dir, partition_keys=['nonexistent_column']
        )


# ---------------------------------------------------------------------------
# Tests de export_schema
# ---------------------------------------------------------------------------

# Test de l'export complet du schéma (fact table + metadata + dimensions)
def test_export_schema_complete(exporter_with_schema, tmp_path):
    """Test that export_schema exports fact table, metadata, and dimension tables."""
    output_dir = str(tmp_path / "schema_export")
    os.makedirs(output_dir, exist_ok=True)

    exporter_with_schema.export_schema(
        fact_table='fact_table',
        output_dir=output_dir,
        include_metadata=True,
        include_dimensions=True,
    )

    # La fact table doit exister comme répertoire (export à plat sans partition)
    assert os.path.exists(os.path.join(output_dir, 'fact_table'))
    # La table de métadonnées doit exister
    assert os.path.exists(os.path.join(output_dir, 'metadata.parquet'))


# Test de l'export sans la table de métadonnées
def test_export_schema_without_metadata(exporter_with_schema, tmp_path):
    """Test that metadata table is skipped when include_metadata=False."""
    output_dir = str(tmp_path / "no_meta")
    os.makedirs(output_dir, exist_ok=True)

    exporter_with_schema.export_schema(
        fact_table='fact_table',
        output_dir=output_dir,
        include_metadata=False,
        include_dimensions=False,
    )

    # La fact table doit exister
    assert os.path.exists(os.path.join(output_dir, 'fact_table'))
    # La table de métadonnées ne doit pas exister
    assert not os.path.exists(os.path.join(output_dir, 'metadata.parquet'))


# Test de l'export avec partitionnement de la fact table
def test_export_schema_with_partition(exporter_with_schema, tmp_path):
    """Test that the fact table is partitioned while auxiliary tables are exported flat."""
    output_dir = str(tmp_path / "partitioned_schema")
    os.makedirs(output_dir, exist_ok=True)

    exporter_with_schema.export_schema(
        fact_table='fact_table',
        output_dir=output_dir,
        partition_keys=['id'],
        include_metadata=True,
        include_dimensions=False,
    )

    # La fact table doit avoir des sous-répertoires de partition
    fact_dir = os.path.join(output_dir, 'fact_table')
    assert os.path.isdir(fact_dir)
    subdirs = [
        d for d in os.listdir(fact_dir)
        if os.path.isdir(os.path.join(fact_dir, d))
    ]
    assert len(subdirs) > 0

    # La metadata doit être à plat
    assert os.path.exists(os.path.join(output_dir, 'metadata.parquet'))


# Test que export_schema lève ValueError pour une fact table inexistante
def test_export_schema_nonexistent_fact_table(exporter_empty, tmp_path):
    """Test that export_schema raises ValueError when fact_table does not exist."""
    output_dir = str(tmp_path / "bad_schema")
    os.makedirs(output_dir, exist_ok=True)

    with pytest.raises(ValueError, match="n'existe pas"):
        exporter_empty.export_schema(
            fact_table='nonexistent_table',
            output_dir=output_dir,
        )
