# Importation des modules
# Modules de base
import json
import os
import warnings
from pathlib import Path

# DuckDB
import duckdb
import narwhals as nw
import polars as pl

# Module de tests
import pytest

# Module à tester
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from tests.utils.ducklake import requires_ducklake

# ---------------------------------------------------------------------------
# Fixture locale
# ---------------------------------------------------------------------------


# Initialisation d'une instance de DuckLakeTablesBuilder utilisée dans l'ensemble des
# tests
@pytest.fixture
def ducklake_builder(sample_df: pl.DataFrame) -> DuckLakeTablesBuilder:
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
def test_ducklake_builder_initialization(sample_df: pl.DataFrame) -> None:
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

        # Alias du catalogue : défaut 'db' et valeur explicite conservée au même
        # titre que le schéma
        assert DuckLakeTablesBuilder(sample_df).catalog_alias == "db"
        builder3 = DuckLakeTablesBuilder(
            sample_df, schema="predictions", catalog_alias="my_lake"
        )
        assert builder3.catalog_alias == "my_lake"
        assert builder3.schema == "predictions"


# ---------------------------------------------------------------------------
# Tests de create_duckdb_metadata_table()
# ---------------------------------------------------------------------------


# Test de la création de la table des méta-données dans DuckDB
def test_create_duckdb_metadata_table(ducklake_builder: DuckLakeTablesBuilder) -> None:
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
    assert "is_categorical_forced" not in result.columns
    assert "sql_type" in result.columns
    assert "is_categorical" in result.columns


# ---------------------------------------------------------------------------
# Tests de create_duckdb_dataset_metadata_table()
# ---------------------------------------------------------------------------


# Test de la création de la table des méta-données du jeu de résultats
def test_create_duckdb_dataset_metadata_table(
    ducklake_builder: DuckLakeTablesBuilder,
) -> None:
    """Test the build of the single-row dataset_metadata table.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
    """
    # Création de la table descriptive du jeu de résultats
    ducklake_builder.create_duckdb_dataset_metadata_table()

    result = ducklake_builder.conn.execute("SELECT * FROM dataset_metadata").pl()

    # Une seule ligne par schéma
    assert result.shape[0] == 1
    # Champs systématiquement renseignés
    assert result["schema_version"][0] == 1
    assert result["updated_at"][0] is not None
    # cluster_by reste NULL à ce stade
    assert result["cluster_by"][0] is None
    # Champs descriptifs non fournis par le builder de test
    assert result["label"][0] is None


# Test de la propagation des champs descriptifs du jeu de résultats
def test_dataset_metadata_carries_builder_arguments(
    sample_df: pl.DataFrame,
) -> None:
    """Test that the optional dataset arguments reach dataset_metadata.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
            dataset_label="Prédictions",
            dataset_description="Sorties du modèle",
            dataset_source="pipeline-ml",
        )
    builder.create_duckdb_dataset_metadata_table()

    result = builder.conn.execute("SELECT * FROM dataset_metadata").pl()
    assert result["label"][0] == "Prédictions"
    assert result["description"][0] == "Sorties du modèle"
    assert result["source"][0] == "pipeline-ml"


# ---------------------------------------------------------------------------
# Tests de create_duckdb_fact_table()
# ---------------------------------------------------------------------------


# Test de la création de la table des faits dans DuckDB
def test_create_duckdb_fact_table(
    ducklake_builder: DuckLakeTablesBuilder,
    sample_df: pl.DataFrame,
) -> None:
    """Test the build of the fact table in DuckDB.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
        sample_df: Sample polars DataFrame.
    """
    # Création des tables nécessaires en amont
    ducklake_builder.create_duckdb_metadata_table()
    # Création de la table des faits
    ducklake_builder.create_duckdb_fact_table(table_name="test_fact")

    # Vérification que la table des faits existe et a la bonne forme
    result = ducklake_builder.conn.execute("SELECT * FROM test_fact").pl()
    assert result.shape[0] == len(sample_df)
    assert "category" in result.columns
    assert "status" in result.columns

    # Vérification que les colonnes catégorielles portent les libellés d'origine
    assert result["category"].to_list() == sample_df["category"].to_list()
    assert result["status"].to_list() == sample_df["status"].to_list()


# ---------------------------------------------------------------------------
# Tests de build_schema()
# ---------------------------------------------------------------------------


# Test de la construction de l'ensemble du schéma
def test_build_schema(ducklake_builder: DuckLakeTablesBuilder) -> None:
    """Test the build of the complete schema (metadata, fact, dataset_metadata).

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
    """
    # Construction du schéma complet
    ducklake_builder.build_schema(
        metadata_table="test_metadata",
        fact_table="test_fact",
        dataset_metadata_table="test_dataset_metadata",
    )

    # Vérification que les trois tables attendues ont été créées, et elles seules
    tables = ducklake_builder.conn.execute("SHOW TABLES").fetchall()
    table_names = [t[0] for t in tables]
    assert "test_metadata" in table_names
    assert "test_fact" in table_names
    assert "test_dataset_metadata" in table_names
    assert not [name for name in table_names if name.startswith("dim_")]


# Test de l'affichage du schéma construit
def test_display_schema(
    ducklake_builder: DuckLakeTablesBuilder,
    caplog: pytest.LogCaptureFixture,
) -> None:
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
def test_build_schema_remove_all_duplicates(
    sample_df_with_duplicates: pl.DataFrame,
) -> None:
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
    row = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()
    assert row is not None
    final_count = row[0]
    assert final_count < initial_count


# Test de la conservation du premier doublon avec check_duplicates=True et keep='first'
def test_build_schema_keep_first_duplicate(
    sample_df_with_duplicates: pl.DataFrame,
) -> None:
    """Test duplicate removal with check_duplicates=True and keep='first'.

    Args:
        sample_df_with_duplicates: DataFrame with duplicate rows.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.schema_builder.df)

    builder.build_schema(check_duplicates=True, keep="first")

    row = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()
    assert row is not None
    final_count = row[0]
    assert final_count <= initial_count


# Test de la conservation du dernier doublon avec check_duplicates=True et keep='last'
def test_build_schema_keep_last_duplicate(
    sample_df_with_duplicates: pl.DataFrame,
) -> None:
    """Test duplicate removal with check_duplicates=True and keep='last'.

    Args:
        sample_df_with_duplicates: DataFrame with duplicate rows.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.schema_builder.df)

    builder.build_schema(check_duplicates=True, keep="last")

    row = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()
    assert row is not None
    final_count = row[0]
    assert final_count <= initial_count


# Test de la conservation de toutes les lignes avec check_duplicates=False
def test_build_schema_no_duplicate_check(
    sample_df_with_duplicates: pl.DataFrame,
) -> None:
    """Test no duplicate removal with check_duplicates=False.

    Args:
        sample_df_with_duplicates: DataFrame with duplicate rows.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df_with_duplicates)
    initial_count = len(builder.schema_builder.df)

    builder.build_schema(check_duplicates=False)

    row = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()
    assert row is not None
    final_count = row[0]
    assert final_count == initial_count


# Test du logging lors de la suppression des doublons
def test_duplicate_removal_logging(
    sample_df_with_duplicates: pl.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
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
def test_duplicate_check_uses_primary_keys(sample_df: pl.DataFrame) -> None:
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
    row_final = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()
    assert row_final is not None
    final_count = row_final[0]
    assert final_count < initial_count
    # Chaque valeur d'identifiant ne doit apparaître qu'une seule fois
    row_unique = builder.conn.execute(
        "SELECT COUNT(DISTINCT id) FROM fact_table"
    ).fetchone()
    assert row_unique is not None
    id_unique = row_unique[0]
    row_total = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()
    assert row_total is not None
    id_total = row_total[0]
    assert id_unique == id_total


# ---------------------------------------------------------------------------
# Tests liés à categorical_threshold=None
# ---------------------------------------------------------------------------


# Test que categorical_threshold=None n'altère pas les tables construites
def test_categorical_threshold_none_builds_three_tables(
    sample_df: pl.DataFrame,
) -> None:
    """Test that the schema still holds exactly three tables without a threshold.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df, categorical_threshold=None)

    builder.build_schema()

    # Récupération des tables créées
    tables = {row[0] for row in builder.conn.execute("SHOW TABLES").fetchall()}

    assert tables == {"fact_table", "metadata", "dataset_metadata"}


# ---------------------------------------------------------------------------
# Tests de propagation du UserWarning
# ---------------------------------------------------------------------------


# Test que DuckLakeTablesBuilder propage le UserWarning de SchemaBuilder
def test_warning_propagated_from_ducklake_builder(sample_df: pl.DataFrame) -> None:
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
def test_query_nonexistent_table(
    ducklake_builder: DuckLakeTablesBuilder,
    table_name: str,
) -> None:
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
@requires_ducklake
def test_create_duckdb_fact_table_with_partition_by(sample_df: pl.DataFrame) -> None:
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
        # Vérification que partition_by est accepté sans erreur
        builder.create_duckdb_fact_table(partition_by=["category"])

        tables = [row[0] for row in conn.execute("SHOW TABLES").fetchall()]
        assert "fact_table" in tables


# Test de build_schema avec partition_by
@requires_ducklake
def test_build_schema_with_partition_by(sample_df: pl.DataFrame) -> None:
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
        assert "dataset_metadata" in tables


# ---------------------------------------------------------------------------
# Tests du support multi-schémas (plusieurs schémas dans un même catalogue)
# ---------------------------------------------------------------------------


# Test que build_schema construit les tables dans le schéma nommé demandé
def test_build_schema_into_named_schema(sample_df: pl.DataFrame) -> None:
    """Test that build_schema creates tables in the requested named schema.

    Args:
        sample_df: Sample polars DataFrame.
    """
    conn = duckdb.connect(":memory:")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
            connection=conn,
            schema="predictions",
        )
    builder.build_schema()

    # Les tables existent bien dans le schéma 'predictions'
    schemas = [
        row[0]
        for row in conn.execute(
            "SELECT table_schema FROM information_schema.tables "
            "WHERE table_name = 'fact_table'"
        ).fetchall()
    ]
    assert "predictions" in schemas

    # Les requêtes qualifiées par le schéma fonctionnent
    row = conn.execute("SELECT COUNT(*) FROM predictions.fact_table").fetchone()
    assert row is not None
    count = row[0]
    assert count == len(sample_df)


# Test que deux schémas coexistent dans un même catalogue sans interférence
def test_two_schemas_coexist_in_one_catalog(
    multi_schema_connection: duckdb.DuckDBPyConnection,
) -> None:
    """Test that two schemas coexist with their own tables in one catalog.

    Args:
        multi_schema_connection: Connection with 'predictions' and 'shapley' schemas.
    """
    # Inventaire des tables par schéma
    rows = multi_schema_connection.execute(
        "SELECT table_schema, table_name FROM information_schema.tables "
        "WHERE table_schema IN ('predictions', 'shapley') ORDER BY 1, 2"
    ).fetchall()
    tables_by_schema: dict[str, set[str]] = {}
    for schema_name, table_name in rows:
        tables_by_schema.setdefault(schema_name, set()).add(table_name)

    # Chaque schéma possède ses trois couches
    for schema_name in ("predictions", "shapley"):
        assert "fact_table" in tables_by_schema[schema_name]
        assert "metadata" in tables_by_schema[schema_name]
        assert "dataset_metadata" in tables_by_schema[schema_name]

    # Les comptages sont propres à chaque schéma (3 vs 2 lignes)
    pred_row = multi_schema_connection.execute(
        "SELECT COUNT(*) FROM predictions.fact_table"
    ).fetchone()
    assert pred_row is not None
    pred_rows = pred_row[0]
    shap_row = multi_schema_connection.execute(
        "SELECT COUNT(*) FROM shapley.fact_table"
    ).fetchone()
    assert shap_row is not None
    shap_rows = shap_row[0]
    assert pred_rows == 3
    assert shap_rows == 2


# ---------------------------------------------------------------------------
# Tests des champs d'UI de la table metadata (column_metadata)
# ---------------------------------------------------------------------------


# Test que le DDL de metadata porte les colonnes d'UI, toutes VARCHAR nullable
def test_metadata_ddl_carries_ui_columns(
    ducklake_builder: DuckLakeTablesBuilder,
) -> None:
    """Test that the metadata table DDL includes the nullable UI columns.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
    """
    ducklake_builder.create_duckdb_metadata_table(table_name="test_metadata")

    described = ducklake_builder.conn.execute("DESCRIBE test_metadata").fetchall()
    columns = {row[0]: (row[1], row[2]) for row in described}

    for field in (
        "unit",
        "display_format",
        "family",
        "description",
        "default_aggregation",
    ):
        assert field in columns
        # Type VARCHAR et colonne nullable
        assert columns[field][0] == "VARCHAR"
        assert columns[field][1] == "YES"


# Test que build_schema écrit les valeurs de column_metadata dans la table
def test_build_schema_writes_column_metadata(sample_df: pl.DataFrame) -> None:
    """Test that build_schema persists the column_metadata values into metadata.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df, categorical_threshold=4)

    builder.build_schema(
        column_metadata={
            "value": {
                "unit": "%",
                "display_format": ".0%",
                "family": "scores",
                "description": "prediction score",
                "default_aggregation": "median",
            }
        }
    )

    row = builder.conn.execute(
        "SELECT unit, display_format, family, description, default_aggregation"
        " FROM metadata WHERE name = 'value'"
    ).fetchone()
    assert row == ("%", ".0%", "scores", "prediction score", "MEDIAN")

    # Une colonne non renseignée conserve des champs d'UI nuls
    other = builder.conn.execute(
        "SELECT unit, default_aggregation FROM metadata WHERE name = 'category'"
    ).fetchone()
    assert other == (None, None)


# Test que build_schema propage l'erreur de validation de default_aggregation
def test_build_schema_invalid_default_aggregation_raises(
    sample_df: pl.DataFrame,
) -> None:
    """Test that an invalid default_aggregation aborts build_schema with ValueError.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df, categorical_threshold=4)

    with pytest.raises(ValueError, match="Invalid default_aggregation"):
        builder.build_schema(
            column_metadata={"value": {"default_aggregation": "SOMME"}}
        )


# ---------------------------------------------------------------------------
# Tests de la hiérarchie de colonnes (parent_name, §2.5)
# ---------------------------------------------------------------------------


# Test que le DDL de metadata porte la colonne parent_name, VARCHAR nullable
def test_metadata_ddl_carries_parent_name(
    ducklake_builder: DuckLakeTablesBuilder,
) -> None:
    """Test that the metadata table DDL includes the nullable parent_name column.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
    """
    ducklake_builder.create_duckdb_metadata_table(table_name="test_metadata")

    described = ducklake_builder.conn.execute("DESCRIBE test_metadata").fetchall()
    columns = {row[0]: (row[1], row[2]) for row in described}

    assert "parent_name" in columns
    assert columns["parent_name"][0] == "VARCHAR"
    assert columns["parent_name"][1] == "YES"


# Test que build_schema écrit la hiérarchie déclarée via le paramètre hierarchies
def test_build_schema_writes_hierarchies(sample_df: pl.DataFrame) -> None:
    """Test that hierarchies passed to the constructor reach the metadata table.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
            hierarchies={"category": "status"},
        )

    builder.build_schema()

    row = builder.conn.execute(
        "SELECT parent_name FROM metadata WHERE name = 'category'"
    ).fetchone()
    assert row[0] == "status"


# Test que build_schema propage une erreur de cycle dans la hiérarchie
def test_build_schema_hierarchy_cycle_raises(sample_df: pl.DataFrame) -> None:
    """Test that a cyclic hierarchy aborts build_schema with ValueError.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
            hierarchies={"category": "status", "status": "category"},
        )

    with pytest.raises(ValueError, match="Cycle detected"):
        builder.build_schema()


# ---------------------------------------------------------------------------
# Tests des colonnes de libellés (value_labels / label_for, §2.6)
# ---------------------------------------------------------------------------


# Test que build_schema écrit la paire code/libellé déclarée via value_labels
def test_build_schema_writes_value_labels(sample_df: pl.DataFrame) -> None:
    """Test that value_labels passed to the constructor reach the metadata table.

    ``status`` (label) -> ``category`` (code) respects the functional dependency
    on ``sample_df`` (each category maps to a single status).

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
            value_labels={"status": "category"},
        )

    builder.build_schema()

    row = builder.conn.execute(
        "SELECT label_for FROM metadata WHERE name = 'status'"
    ).fetchone()
    assert row[0] == "category"


# Test que build_schema propage une violation de la dépendance fonctionnelle
def test_build_schema_value_label_dependency_violation_raises(
    sample_df: pl.DataFrame,
) -> None:
    """Test that a functional dependency violation aborts build_schema, writing
    nothing.

    ``category`` (label) -> ``status`` (code) violates the dependency on
    ``sample_df``: ``status='active'`` maps to both ``category='A'`` and
    ``category='C'``.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
            value_labels={"category": "status"},
        )

    with pytest.raises(ValueError, match="Functional dependency"):
        builder.build_schema()

    # Rien n'a été écrit, y compris la table metadata (déjà créée avant le contrôle)
    tables = builder.conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()
    assert tables == []


# ---------------------------------------------------------------------------
# Tests de cluster_by (§5.3)
# ---------------------------------------------------------------------------


# Test que cluster_by par défaut reprend les clés primaires dans leur ordre
def test_build_schema_cluster_by_defaults_to_primary_keys(
    sample_df: pl.DataFrame,
) -> None:
    """Test that cluster_by defaults to the primary keys in declared order.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df, categorical_threshold=4, primary_keys=["id"]
        )
    builder.build_schema()

    row = builder.conn.execute("SELECT cluster_by FROM dataset_metadata").fetchone()
    assert json.loads(row[0]) == ["id"]


# Test qu'aucune clé primaire ne produit un cluster_by NULL par défaut
def test_build_schema_cluster_by_none_without_primary_keys(
    sample_df: pl.DataFrame,
) -> None:
    """Test that cluster_by stays NULL by default when there is no primary key.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(sample_df, categorical_threshold=4)
    builder.build_schema()

    row = builder.conn.execute("SELECT cluster_by FROM dataset_metadata").fetchone()
    assert row[0] is None


# Test qu'une valeur explicite de cluster_by est persistée telle quelle
def test_build_schema_cluster_by_explicit_value(sample_df: pl.DataFrame) -> None:
    """Test that an explicit cluster_by is persisted as the given JSON list.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df, categorical_threshold=4, primary_keys=["id"]
        )
    builder.build_schema(cluster_by=["category", "id"])

    row = builder.conn.execute("SELECT cluster_by FROM dataset_metadata").fetchone()
    assert json.loads(row[0]) == ["category", "id"]


# Test qu'une colonne de cluster_by inconnue lève une ValueError
def test_build_schema_cluster_by_unknown_column_raises(
    sample_df: pl.DataFrame,
) -> None:
    """Test that an unknown cluster_by column aborts build_schema with ValueError.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df, categorical_threshold=4, primary_keys=["id"]
        )

    with pytest.raises(ValueError, match="cluster_by columns"):
        builder.build_schema(cluster_by=["not_a_column"])


# Test que la table des faits est effectivement triée selon cluster_by
def test_build_schema_fact_table_sorted_by_cluster_by(
    sample_df: pl.DataFrame,
) -> None:
    """Test that the fact table rows are physically written in cluster_by order.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df, categorical_threshold=4, primary_keys=["id"]
        )
    # Tri décroissant improbable par défaut (id croissant) : category d'abord
    builder.build_schema(cluster_by=["category"])

    categories = [
        row[0]
        for row in builder.conn.execute("SELECT category FROM fact_table").fetchall()
    ]
    assert categories == sorted(categories)


# Test que create_duckdb_dataset_metadata_table appelée directement avec cluster_by
# persiste la liste JSON fournie
def test_create_duckdb_dataset_metadata_table_with_cluster_by(
    ducklake_builder: DuckLakeTablesBuilder,
) -> None:
    """Test that an explicit cluster_by reaches dataset_metadata as a JSON list.

    Args:
        ducklake_builder: DuckLakeTablesBuilder fixture.
    """
    ducklake_builder.create_duckdb_dataset_metadata_table(cluster_by=["id", "date"])

    result = ducklake_builder.conn.execute(
        "SELECT cluster_by FROM dataset_metadata"
    ).fetchone()
    assert json.loads(result[0]) == ["id", "date"]


# Test que le tri physique produit des fichiers Parquet dont les plages ne se
# recouvrent pas (élagage par fichier, §5.3)
@requires_ducklake
def test_build_schema_cluster_by_produces_non_overlapping_files(
    tmp_path: Path,
) -> None:
    """Test that cluster_by-sorted data yields files with non-overlapping ranges.

    Attaches a real on-disk DuckLake catalog with inlining disabled and a very
    small target file size, so a moderately sized DataFrame lands in several
    Parquet files. Reads them back via ``read_parquet`` and checks that the
    per-file min/max ranges of the cluster column do not overlap — the physical
    condition for DuckLake's file-pruning to work (annexe A of the specification).

    With the engine's default parallelism, DuckDB spreads a sorted INSERT across
    files non-monotonically (measured, annexe A — the same effect documented for
    ``recluster``); ``SET threads = 1`` around the write is the same technique the
    specification itself uses to observe the physical effect deterministically. It
    is applied only in this test, not in production code (only
    ``DuckLakeMaintenance.recluster`` writes single-threaded).

    Args:
        tmp_path: pytest temporary directory.
    """
    from dt_ducklake_manager.connection import DuckLakeConnector

    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)

    conn = DuckLakeConnector(
        catalog,
        data_dir,
        data_inlining_row_limit=0,
        ducklake_options={"target_file_size": "1MB"},
    ).connect()

    # Grand nombre de lignes pour dépasser largement la petite taille de fichier
    # cible en un seul INSERT
    n = 500_000
    df = pl.DataFrame(
        {
            "id": list(range(n)),
            "value": [float(i) for i in range(n)],
        }
    )

    conn.execute("SET threads = 1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            df,
            categorical_threshold=4,
            primary_keys=["id"],
            connection=conn,
            catalog_alias="db",
        )
    builder.build_schema(cluster_by=["id"])

    # Fichiers réellement associés à fact_table (et non à metadata/dataset_metadata,
    # qui partagent le même data_path) : ducklake_list_files est la source fiable,
    # un glob sur data_path mélangerait les schémas des différentes tables.
    files = [
        row[0]
        for row in conn.execute(
            "SELECT data_file FROM ducklake_list_files('db', 'fact_table',"
            " schema := 'main')"
        ).fetchall()
    ]
    assert len(files) > 1, "expected build_schema to produce several Parquet files"

    # Un fichier résiduel vide est possible (mesuré, annexe A) : exclu du contrôle
    # de recouvrement, qui ne porte que sur des plages réelles.
    ranges = sorted(
        row
        for row in (
            conn.execute(
                f"SELECT min(id) AS lo, max(id) AS hi FROM read_parquet('{f}')"
            ).fetchone()
            for f in files
        )
        if row is not None and row[0] is not None
    )

    # Les plages [lo, hi] ne doivent pas se chevaucher une fois triées par lo
    for (lo, hi), (next_lo, _next_hi) in zip(ranges, ranges[1:]):
        assert hi <= next_lo, (
            f"overlapping file ranges: ({lo}, {hi}) vs starting at {next_lo}"
        )

    conn.close()
