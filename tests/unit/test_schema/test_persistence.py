# Importation des modules
# Modules de base
import warnings

# DuckDB
import duckdb
import narwhals as nw
import polars as pl

# Module de tests
import pytest

# Module à tester
from dt_ducklake_manager.schema import DuckLakeTablesBuilder

# ---------------------------------------------------------------------------
# Fonctions auxiliaires
# ---------------------------------------------------------------------------


def _ducklake_available() -> bool:
    """Vérifie si l'extension DuckLake est disponible dans l'environnement de test."""
    try:
        import duckdb as _ddb

        conn = _ddb.connect(":memory:")
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        conn.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixture locale
# ---------------------------------------------------------------------------


# Initialisation d'une instance de DuckLakeTablesBuilder utilisée dans l'ensemble des
# tests
@pytest.fixture
def ducklake_builder(sample_df):
    """Initialize a DuckLakeTablesBuilder instance with a sample DataFrame.

    Args:
        sample_df: polars DataFrame fixture from conftest.

    Returns:
        DuckLakeTablesBuilder: initialized with categorical_threshold=4.
    """
    # Suppression du UserWarning lié à l'absence de clés primaires : ce comportement
    # est testé séparément dans test_warning_propagated_from_ducklake_builder.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return DuckLakeTablesBuilder(sample_df, categorical_threshold=4)


# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------


# Test de l'initialisation du constructeur avec et sans connexion explicite
def test_ducklake_builder_initialization(sample_df):
    """Test the initialization of the DuckLakeTablesBuilder class.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # Vérification de la connexion en mémoire (fallback pour les tests unitaires)
        builder = DuckLakeTablesBuilder(sample_df)
        assert isinstance(builder.conn, duckdb.DuckDBPyConnection)

        # Vérification que la connexion fournie est bien utilisée
        conn = duckdb.connect(":memory:")
        builder2 = DuckLakeTablesBuilder(sample_df, connection=conn)
        assert builder2.conn is conn


# ---------------------------------------------------------------------------
# Tests de create_duckdb_metadata_table()
# ---------------------------------------------------------------------------


# Test de la création de la table des méta-données dans DuckDB
def test_create_duckdb_metadata_table(ducklake_builder):
    """Test the build of the metadata table in DuckDB.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
    """
    # Création de la table des méta-données
    ducklake_builder.create_duckdb_metadata_table(table_name="test_metadata")

    # Vérification que la table existe et a la structure attendue
    result = ducklake_builder.conn.execute("SELECT * FROM test_metadata").pl()
    assert "name" in result.columns
    assert "label" in result.columns
    assert "python_type" in result.columns
    assert "sql_type" in result.columns
    assert "is_categorical" in result.columns


# ---------------------------------------------------------------------------
# Tests de create_duckdb_dimension_tables()
# ---------------------------------------------------------------------------


# Test de la création des tables de dimension dans DuckDB
def test_create_duckdb_dimension_tables(ducklake_builder):
    """Test the build of the dimension tables in DuckDB.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
    """
    # Création des tables de dimensions
    ducklake_builder.create_duckdb_dimension_tables(table_prefix="test_dim_")

    # Vérification que les tables attendues existent
    tables = ducklake_builder.conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    assert "test_dim_category" in table_names
    assert "test_dim_status" in table_names

    # Vérification de la structure des tables de dimension
    for table in ["category", "status"]:
        result = ducklake_builder.conn.execute(f"SELECT * FROM test_dim_{table}").pl()
        assert "value" in result.columns
        assert "label" in result.columns


# ---------------------------------------------------------------------------
# Tests de create_duckdb_fact_table()
# ---------------------------------------------------------------------------


# Test de la création de la table des faits dans DuckDB
def test_create_duckdb_fact_table(ducklake_builder, sample_df):
    """Test the build of the fact table in DuckDB.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
        sample_df: Sample polars DataFrame.
    """
    # Création des tables nécessaires en amont
    ducklake_builder.create_duckdb_metadata_table()
    ducklake_builder.create_duckdb_dimension_tables()
    # Création de la table des faits
    ducklake_builder.create_duckdb_fact_table(table_name="test_fact")

    # Vérification que la table des faits existe et a les bonnes dimensions
    result = ducklake_builder.conn.execute("SELECT * FROM test_fact").pl()
    assert result.shape[0] == len(sample_df)
    assert "category" in result.columns
    assert "status" in result.columns

    # Vérification que les valeurs de la fact table
    # sont des indices entiers de la dim table
    dim_category = ducklake_builder.conn.execute("SELECT * FROM dim_category").pl()
    dim_status = ducklake_builder.conn.execute("SELECT * FROM dim_status").pl()
    assert set(result["category"].to_list()) <= set(
        dim_category["value"].cast(pl.Int64).to_list()
    )
    assert set(result["status"].to_list()) <= set(
        dim_status["value"].cast(pl.Int64).to_list()
    )


# ---------------------------------------------------------------------------
# Tests de build_schema()
# ---------------------------------------------------------------------------


# Test de la construction de l'ensemble du schéma
def test_build_schema(ducklake_builder):
    """Test the build of the complete schema (metadata, dimensions, fact table).

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
    """
    # Construction du schéma complet
    ducklake_builder.build_schema(
        metadata_table="test_metadata",
        fact_table="test_fact",
        dim_table_prefix="test_dim_",
    )

    # Vérification que toutes les tables attendues ont été créées
    tables = ducklake_builder.conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    assert "test_metadata" in table_names
    assert "test_fact" in table_names
    assert "test_dim_category" in table_names
    assert "test_dim_status" in table_names


# Test de l'affichage du schéma construit
def test_display_schema(ducklake_builder, caplog):
    """Test the display of the built schema.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
        caplog: pytest logging capture fixture.
    """
    ducklake_builder.build_schema()
    ducklake_builder.display_schema()

    # Vérification que les informations de schéma sont bien loguées
    assert "Created Tables:" in caplog.text
    assert "Structure:" in caplog.text


# ---------------------------------------------------------------------------
# Tests de la gestion des doublons
# ---------------------------------------------------------------------------


# Test de la suppression de tous les doublons avec check_duplicates=True et keep=False
def test_build_schema_remove_all_duplicates(sample_df_with_duplicates):
    """Test duplicate removal with check_duplicates=True and keep=False.

    Args:
        sample_df_with_duplicates: DataFrame with duplicate rows.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.schema_builder.df)

    # Construction du schéma avec suppression de tous les doublons
    builder.build_schema(check_duplicates=True, keep="none")

    # Vérification que les doublons ont bien été supprimés
    final_count = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert final_count < initial_count


# Test de la conservation du premier doublon avec check_duplicates=True et keep='first'
def test_build_schema_keep_first_duplicate(sample_df_with_duplicates):
    """Test duplicate removal with check_duplicates=True and keep='first'.

    Args:
        sample_df_with_duplicates: DataFrame with duplicate rows.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.schema_builder.df)

    builder.build_schema(check_duplicates=True, keep="first")

    final_count = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert final_count <= initial_count


# Test de la conservation du dernier doublon avec check_duplicates=True et keep='last'
def test_build_schema_keep_last_duplicate(sample_df_with_duplicates):
    """Test duplicate removal with check_duplicates=True and keep='last'.

    Args:
        sample_df_with_duplicates: DataFrame with duplicate rows.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.schema_builder.df)

    builder.build_schema(check_duplicates=True, keep="last")

    final_count = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert final_count <= initial_count


# Test de la conservation de toutes les lignes avec check_duplicates=False
def test_build_schema_no_duplicate_check(sample_df_with_duplicates):
    """Test no duplicate removal with check_duplicates=False.

    Args:
        sample_df_with_duplicates: DataFrame with duplicate rows.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.schema_builder.df)

    builder.build_schema(check_duplicates=False)

    final_count = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert final_count == initial_count


# Test du logging lors de la suppression des doublons
def test_duplicate_removal_logging(sample_df_with_duplicates, caplog):
    """Test logging of removed duplicates.

    Args:
        sample_df_with_duplicates: DataFrame with duplicate rows.
        caplog: pytest logging capture fixture.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df_with_duplicates)

    # Suppression des duplicats
    builder.build_schema(check_duplicates=True, keep="none")

    # Vérification que le message de log correspond au message réel de la fonction
    assert "Removing duplicates from" in caplog.text


# Test que la déduplication se base sur les clés primaires quand elles sont fournies
def test_duplicate_check_uses_primary_keys(sample_df):
    """Test that duplicate detection uses primary_keys when provided.

    A duplicate row with the same 'id' but a different 'value' is injected after
    the builder is initialized. With primary_keys=['id'], build_schema must identify
    the rows as duplicates even though 'value' differs.

    Args:
        sample_df: Sample polars DataFrame (no duplicates on 'id').
    """
    # Initialisation avec sample_df dont les ids sont uniques → validation OK
    builder = DuckLakeTablesBuilder(
        sample_df, categorical_threshold=4, primary_keys=["id"]
    )

    # Injection d'un doublon après l'initialisation : même id=1, valeur différente
    original_df = builder.schema_builder.df
    duplicate_row = original_df.filter(nw.col("id") == 1).with_columns(
        nw.lit(99.99).alias("value")
    )
    builder.schema_builder.df = nw.concat([original_df, duplicate_row])

    initial_count = len(builder.schema_builder.df)
    assert initial_count == len(sample_df) + 1

    # Construction du schéma avec déduplication basée sur les clés primaires
    builder.build_schema(check_duplicates=True, keep="first")

    # Le doublon sur id=1 doit avoir été supprimé
    final_count = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert final_count < initial_count
    # Chaque valeur d'identifiant ne doit apparaître qu'une seule fois
    id_unique = builder.conn.execute(
        "SELECT COUNT(DISTINCT id) FROM fact_table"
    ).fetchone()[0]
    id_total = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert id_unique == id_total


# ---------------------------------------------------------------------------
# Tests liés à categorical_threshold=None
# ---------------------------------------------------------------------------


# Test que categorical_threshold=None ne crée aucune table de dimension
def test_categorical_threshold_none_no_dim_tables(sample_df):
    """Test that no dimension tables are created when categorical_threshold=None.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df, categorical_threshold=None)

    builder.build_schema()

    # Récupération des tables créées
    tables = [row[0] for row in builder.conn.execute("SHOW TABLES").fetchall()]

    # Aucune table de dimension (préfixe 'dim_') ne doit exister
    dim_tables = [t for t in tables if t.startswith("dim_")]
    assert len(dim_tables) == 0


# ---------------------------------------------------------------------------
# Tests de propagation du UserWarning
# ---------------------------------------------------------------------------


# Test que DuckLakeTablesBuilder propage le UserWarning de SchemaBuilder
def test_warning_propagated_from_ducklake_builder(sample_df):
    """Test that DuckLakeTablesBuilder raises UserWarning when primary_keys is absent.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with pytest.warns(UserWarning, match="No primary key"):
        DuckLakeTablesBuilder(sample_df, categorical_threshold=4)


# ---------------------------------------------------------------------------
# Tests de gestion des tables inexistantes
# ---------------------------------------------------------------------------


# Test de la génération d'erreur pour des tables absentes
@pytest.mark.parametrize("table_name", ["invalid_table", "nonexistent"])
def test_query_nonexistent_table(ducklake_builder, table_name):
    """Test error raised for non-existent tables.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
        table_name: Name of a table that does not exist.
    """
    with pytest.raises(duckdb.Error):
        ducklake_builder.conn.execute(f"SELECT * FROM {table_name}")


# ---------------------------------------------------------------------------
# Tests liés au paramètre partition_by (nécessite l'extension DuckLake)
# ---------------------------------------------------------------------------


# Test de la création de la fact table avec partitionnement
@pytest.mark.skipif(
    not _ducklake_available(), reason="Extension ducklake non disponible"
)
def test_create_duckdb_fact_table_with_partition_by(sample_df):
    """Test that create_duckdb_fact_table accepts partition_by without error.

    Args:
        sample_df: Sample polars DataFrame.
    """
    import os as _os
    import tempfile

    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL ducklake; LOAD ducklake;")

    with tempfile.TemporaryDirectory() as tmpdir:
        catalog = _os.path.join(tmpdir, "test.ducklake")
        data_dir = _os.path.join(tmpdir, "data")
        _os.makedirs(data_dir)
        conn.execute(f"ATTACH 'ducklake:{catalog}' AS db (DATA_PATH '{data_dir}')")
        conn.execute("USE db.main")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            builder = DuckLakeTablesBuilder(
                sample_df, categorical_threshold=4, connection=conn
            )

        builder.create_duckdb_metadata_table()
        builder.create_duckdb_dimension_tables()
        # Vérification que partition_by est accepté sans erreur
        builder.create_duckdb_fact_table(partition_by=["category"])

        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        assert "fact_table" in tables


# Test de build_schema avec partition_by
@pytest.mark.skipif(
    not _ducklake_available(), reason="Extension ducklake non disponible"
)
def test_build_schema_with_partition_by(sample_df):
    """Test that build_schema propagates partition_by to create_duckdb_fact_table.

    Args:
        sample_df: Sample polars DataFrame.
    """
    import os as _os
    import tempfile

    conn = duckdb.connect(":memory:")
    conn.execute("INSTALL ducklake; LOAD ducklake;")

    with tempfile.TemporaryDirectory() as tmpdir:
        catalog = _os.path.join(tmpdir, "test.ducklake")
        data_dir = _os.path.join(tmpdir, "data")
        _os.makedirs(data_dir)
        conn.execute(f"ATTACH 'ducklake:{catalog}' AS db (DATA_PATH '{data_dir}')")
        conn.execute("USE db.main")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            builder = DuckLakeTablesBuilder(
                sample_df, categorical_threshold=4, connection=conn
            )

        # Vérification que le schéma complet se construit sans erreur avec partition_by
        builder.build_schema(partition_by=["category"])

        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        assert "metadata" in tables
        assert "fact_table" in tables
        assert "dim_category" in tables
