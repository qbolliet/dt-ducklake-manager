# Importation des modules
# Modules de base
import warnings
import pandas as pd
# DuckDB
import duckdb
# Module de tests
import pytest
# Module à tester
from dashboard_template_database.builders import DuckdbTablesBuilder


# Initialisation d'une instance de la classe utilisée dans l'ensemble des tests
@pytest.fixture
def duckdb_builder(sample_df):
    """Initialization of the DuckdbTablesBuilder class."""
    # Suppression du UserWarning lié à l'absence de clés primaires : ce comportement
    # est testé séparément dans test_warning_propagated_from_duckdb_builder.
    # categorical_threshold=4 garantit la création de tables de dimension pour category
    # et status (qui ont respectivement 3 et 2 modalités dans sample_df).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return DuckdbTablesBuilder(sample_df, categorical_threshold=4)

# Test de l'initialisation du constructeur
def test_duckdb_builder_initialization(sample_df):
    """Test the initialization of the DuckdbTablesBuilder class."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Vérification de la connexion en mémoire
        builder = DuckdbTablesBuilder(sample_df)
        assert isinstance(builder.conn, duckdb.DuckDBPyConnection)

        # Vérification de la connexion à un fichier
        builder = DuckdbTablesBuilder(sample_df, path=':memory:')
        assert isinstance(builder.conn, duckdb.DuckDBPyConnection)

# Test de la création de la table des méta-données
def test_create_duckdb_metadata_table(duckdb_builder):
    """Test the build of the metadata table."""
    # Création de la table des méta-données
    duckdb_builder.create_duckdb_metadata_table(table_name='test_metadata')
    
    # Vérification que la table existe et a la strcuture attendue
    result = duckdb_builder.conn.execute("SELECT * FROM test_metadata").fetchdf()
    assert 'name' in result.columns
    assert 'label' in result.columns
    assert 'python_type' in result.columns
    assert 'sql_type' in result.columns
    assert 'is_categorical' in result.columns

# Test de la création des tables de dimensions
def test_create_duckdb_dimension_tables(duckdb_builder):
    """Test the build of the dimension tables."""
    # Création des tables de dimensions
    duckdb_builder.create_duckdb_dimension_tables(table_prefix='test_dim_')
    
    # Vérification que les tables existent
    tables = duckdb_builder.conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    
    assert 'test_dim_category' in table_names
    assert 'test_dim_status' in table_names
    
    # Vérification de la structure des tables de dimension
    for table in ['category', 'status']:
        result = duckdb_builder.conn.execute(f"SELECT * FROM test_dim_{table}").fetchdf()
        assert 'value' in result.columns
        assert 'label' in result.columns

# Test de la création de la table des faits
def test_create_duckdb_fact_table(duckdb_builder):
    """Test the build of the fact table."""
    # Création des tables nécessaires
    # Création de la table des méta-données
    duckdb_builder.create_duckdb_metadata_table()
    # Création des tables de dimension
    duckdb_builder.create_duckdb_dimension_tables()
    # Création de la table des faits
    duckdb_builder.create_duckdb_fact_table(table_name='test_fact')
    
    # Vérification que la table des faits existe, a la structure et les dimensions attendues
    result = duckdb_builder.conn.execute("SELECT * FROM test_fact").fetchdf()
    assert result.shape[0] == duckdb_builder.df.shape[0]
    assert 'category' in result.columns
    assert 'status' in result.columns
    
    # Vérification des clés étrangères
    # Les valeurs de la fact table sont des entiers (index de la dim table) ; celles
    # de la dim table sont des VARCHAR en DuckDB → comparaison après cast en entier.
    dim_category = duckdb_builder.conn.execute("SELECT * FROM dim_category").fetchdf()
    dim_status = duckdb_builder.conn.execute("SELECT * FROM dim_status").fetchdf()

    assert set(result['category'].unique()) <= set(dim_category['value'].astype(int))
    assert set(result['status'].unique()) <= set(dim_status['value'].astype(int))

# Test de la création de l'ensemble du schéma
def test_build_duckdb_schema(duckdb_builder):
    """Test the build of the complete scheme."""
    # Création du schéma
    duckdb_builder.build_duckdb_schema(
        metadata_table='test_metadata',
        fact_table='test_fact',
        dim_table_prefix='test_dim_'
    )
    
    # Vérification de l'existence de l'ensemble des tables attendues
    tables = duckdb_builder.conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    
    assert 'test_metadata' in table_names
    assert 'test_fact' in table_names
    assert 'test_dim_category' in table_names
    assert 'test_dim_status' in table_names

# Test de l'affichage du schéma
def test_display_schema(duckdb_builder, caplog):
    """Test the display of the built scheme."""
    # Création du schéma
    duckdb_builder.build_duckdb_schema()
    # Affichage du schéma
    duckdb_builder.display_schema()
    
    # Vérification que l'information est bien rendue
    assert "Created Tables:" in caplog.text
    assert "Structure:" in caplog.text

# Test de la suppression des doublons avec check_duplicates=True et keep=False
def test_build_schema_remove_all_duplicates(sample_df_with_duplicates):
    """Test duplicate removal with check_duplicates=True and keep=False."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckdbTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.df)

    # Construction du schéma avec suppression de tous les doublons
    builder.build_duckdb_schema(check_duplicates=True, keep=False)

    final_count = len(builder.df)
    assert final_count < initial_count

# Test de la suppression des doublons avec check_duplicates=True et keep='first'
def test_build_schema_keep_first_duplicate(sample_df_with_duplicates):
    """Test duplicate removal with check_duplicates=True and keep='first'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckdbTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.df)

    # Construction du schéma en gardant le premier doublon
    builder.build_duckdb_schema(check_duplicates=True, keep='first')

    final_count = len(builder.df)
    assert final_count <= initial_count

# Test de la suppression des doublons avec check_duplicates=True et keep='last'
def test_build_schema_keep_last_duplicate(sample_df_with_duplicates):
    """Test duplicate removal with check_duplicates=True and keep='last'."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckdbTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.df)

    # Construction du schéma en gardant le dernier doublon
    builder.build_duckdb_schema(check_duplicates=True, keep='last')

    final_count = len(builder.df)
    assert final_count <= initial_count

# Test de la conservation des données avec check_duplicates=False
def test_build_schema_no_duplicate_check(sample_df_with_duplicates):
    """Test no duplicate removal with check_duplicates=False."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckdbTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.df)

    # Construction du schéma sans vérification des doublons
    builder.build_duckdb_schema(check_duplicates=False)

    final_count = len(builder.df)
    assert final_count == initial_count

# Test du logging des doublons supprimés
def test_duplicate_removal_logging(sample_df_with_duplicates, caplog):
    """Test logging of removed duplicates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckdbTablesBuilder(sample_df_with_duplicates)

    # Construction du schéma avec suppression des doublons
    builder.build_duckdb_schema(check_duplicates=True, keep=False)

    # Vérification que le logging a eu lieu
    assert "Suppression des doublons" in caplog.text
    assert "observations supprimées" in caplog.text

# Test que la déduplication se base sur les clés primaires quand elles sont fournies
def test_duplicate_check_uses_primary_keys(sample_df):
    """Test that duplicate detection uses primary_keys when provided.

    A duplicate row with the same 'id' but a different 'value' is injected after
    the builder is initialized (init validates PK uniqueness on the original df).
    With primary_keys=['id'], build_duckdb_schema must identify the rows as duplicates
    and remove the extra one, even though 'value' differs between them.
    """
    # Initialisation avec sample_df dont les ids sont uniques → validation OK
    builder = DuckdbTablesBuilder(sample_df, categorical_threshold=4, primary_keys=['id'])

    # Injection d'un doublon après l'initialisation : même id=1, valeur différente
    duplicate_row = builder.df.iloc[[0]].copy()
    duplicate_row['value'] = 99.99
    builder.df = pd.concat([builder.df, duplicate_row], ignore_index=True)

    initial_count = len(builder.df)
    assert initial_count == len(sample_df) + 1

    # Construction du schéma avec déduplication basée sur les clés primaires
    builder.build_duckdb_schema(check_duplicates=True, keep='first')

    # Le doublon sur id=1 (value=99.99) doit avoir été supprimé
    final_count = len(builder.df)
    assert final_count < initial_count
    # Chaque valeur d'identifiant ne doit apparaître qu'une seule fois
    assert builder.df['id'].nunique() == len(builder.df)

# Vérification de la génération d'erreur pour des tables absentes
@pytest.mark.parametrize("table_name", ['invalid_table', 'nonexistent'])
def test_query_nonexistent_table(duckdb_builder, table_name):
    """Test error raised for non-existent tables."""
    with pytest.raises(duckdb.Error):
        duckdb_builder.conn.execute(f"SELECT * FROM {table_name}")


# ---------------------------------------------------------------------------
# Nouveaux tests liés aux changements de comportement par défaut
# ---------------------------------------------------------------------------

# Test que categorical_threshold=None ne crée aucune table de dimension
def test_categorical_threshold_none_no_dim_tables(sample_df):
    """Test that no dimension tables are created in DuckDB when categorical_threshold=None."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckdbTablesBuilder(sample_df, categorical_threshold=None)

    builder.build_duckdb_schema()

    # Récupération de l'ensemble des tables créées
    tables = [row[0] for row in builder.conn.execute("SHOW TABLES").fetchall()]

    # Aucune table de dimension (préfixe 'dim_') ne doit exister
    dim_tables = [t for t in tables if t.startswith('dim_')]
    assert len(dim_tables) == 0


# Test que DuckdbTablesBuilder propage bien le UserWarning de SchemaBuilder
def test_warning_propagated_from_duckdb_builder(sample_df):
    """Test that DuckdbTablesBuilder raises UserWarning when primary_keys is absent."""
    with pytest.warns(UserWarning, match="Aucune clé primaire"):
        DuckdbTablesBuilder(sample_df, categorical_threshold=4)


# ---------------------------------------------------------------------------
# Tests liés au paramètre partition_by (nécessite l'extension DuckLake)
# ---------------------------------------------------------------------------

def _ducklake_available() -> bool:
    """Vérifie si l'extension DuckLake est disponible dans l'environnement de test."""
    try:
        import duckdb as _ddb
        conn = _ddb.connect(':memory:')
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        conn.close()
        return True
    except Exception:
        return False


# Test de la création de la fact table avec partitionnement
@pytest.mark.skipif(not _ducklake_available(), reason="Extension ducklake non disponible")
def test_create_duckdb_fact_table_with_partition_by(sample_df):
    """Test that create_duckdb_fact_table accepts partition_by without error."""
    import duckdb as _ddb
    # Création d'une connexion DuckLake temporaire
    conn = _ddb.connect(':memory:')
    conn.execute("INSTALL ducklake; LOAD ducklake;")

    import tempfile, os
    with tempfile.TemporaryDirectory() as tmpdir:
        catalog = os.path.join(tmpdir, 'test.ducklake')
        data_dir = os.path.join(tmpdir, 'data')
        os.makedirs(data_dir)
        conn.execute(f"ATTACH 'ducklake:{catalog}' AS db (DATA_PATH '{data_dir}')")
        conn.execute("USE db.main")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            builder = DuckdbTablesBuilder(sample_df, categorical_threshold=4, connection=conn)

        builder.create_duckdb_metadata_table()
        builder.create_duckdb_dimension_tables()
        # Vérification que partition_by est accepté sans erreur et que la table est créée
        builder.create_duckdb_fact_table(partition_by=['category'])

        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        assert 'fact_table' in tables


# Test de build_duckdb_schema avec partition_by
@pytest.mark.skipif(not _ducklake_available(), reason="Extension ducklake non disponible")
def test_build_duckdb_schema_with_partition_by(sample_df):
    """Test that build_duckdb_schema propagates partition_by to create_duckdb_fact_table."""
    import duckdb as _ddb
    import tempfile, os

    conn = _ddb.connect(':memory:')
    conn.execute("INSTALL ducklake; LOAD ducklake;")

    with tempfile.TemporaryDirectory() as tmpdir:
        catalog = os.path.join(tmpdir, 'test.ducklake')
        data_dir = os.path.join(tmpdir, 'data')
        os.makedirs(data_dir)
        conn.execute(f"ATTACH 'ducklake:{catalog}' AS db (DATA_PATH '{data_dir}')")
        conn.execute("USE db.main")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            builder = DuckdbTablesBuilder(sample_df, categorical_threshold=4, connection=conn)

        # Vérification que le schéma complet se construit sans erreur avec partition_by
        builder.build_duckdb_schema(partition_by=['category'])

        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        assert 'metadata' in tables
        assert 'fact_table' in tables
        assert 'dim_category' in tables