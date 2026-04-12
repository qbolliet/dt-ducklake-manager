# Importation des modules
# Modules de base
import warnings
import narwhals as nw
# Module de tests
import pytest
# Modules du package à tester
from dt_ducklake_manager.schema import SchemaBuilder


# ---------------------------------------------------------------------------
# Fixture locale
# ---------------------------------------------------------------------------

# Initialisation d'une instance de SchemaBuilder utilisée dans l'ensemble des tests
@pytest.fixture
def schema_builder(sample_df):
    """Initialize a SchemaBuilder instance with a sample DataFrame.

    Args:
        sample_df: polars DataFrame fixture from conftest.

    Returns:
        SchemaBuilder: initialized with categorical_threshold=4.
    """
    # Suppression du UserWarning lié à l'absence de clés primaires : ce comportement
    # est testé séparément dans test_warning_when_no_primary_keys.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return SchemaBuilder(sample_df, categorical_threshold=4)


# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------

# Test de l'initialisation correcte des attributs de la classe
def test_schema_builder_initialization(schema_builder, sample_df):
    """Test the initialization of the SchemaBuilder class.

    Args:
        schema_builder: SchemaBuilder fixture.
        sample_df: Sample polars DataFrame.
    """
    # Vérification du type et de la bonne initialisation des attributs
    assert isinstance(schema_builder, SchemaBuilder)
    assert schema_builder.categorical_threshold == 4
    # Vérification que df est bien encapsulé dans narwhals
    assert isinstance(schema_builder.df, nw.DataFrame)
    # Vérification de l'équivalence de contenu (comparaison via les backends natifs)
    expected = nw.from_native(sample_df, eager_only=True)
    assert schema_builder.df.to_native().equals(expected.to_native())


# ---------------------------------------------------------------------------
# Tests de create_metadata_table()
# ---------------------------------------------------------------------------

# Test de la création de la table des méta-données
def test_create_metadata_table(schema_builder):
    """Test the build of the metadata table.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    # Création de la table des méta-données
    metadata = schema_builder.create_metadata_table()

    # Vérification du type renvoyé (narwhals DataFrame)
    assert isinstance(metadata, nw.DataFrame)
    # Vérification de l'existence de chacune des colonnes attendues
    assert 'name' in metadata.columns
    assert 'label' in metadata.columns
    assert 'python_type' in metadata.columns
    assert 'sql_type' in metadata.columns
    assert 'is_categorical' in metadata.columns
    assert 'is_primary_key' in metadata.columns

    # Vérification de la bonne détection des variables catégorielles
    cat_filter = metadata.filter(nw.col('name') == 'category')['is_categorical'][0]
    status_filter = metadata.filter(nw.col('name') == 'status')['is_categorical'][0]
    high_card_filter = metadata.filter(nw.col('name') == 'high_cardinality')['is_categorical'][0]
    assert cat_filter is True
    assert status_filter is True
    assert high_card_filter is False


# Test de l'ajout de labels personnalisés lors de la création de la table des méta-données
def test_create_metadata_table_with_labels(schema_builder, column_labels):
    """Test the build of the metadata table with custom column labels.

    Args:
        schema_builder: SchemaBuilder fixture.
        column_labels: Dict mapping column names to custom labels.
    """
    # Création de la table des méta-données avec les labels
    metadata = schema_builder.create_metadata_table(column_labels)

    # Vérification de la bonne association des labels fournis
    for col, label in column_labels.items():
        row_label = metadata.filter(nw.col('name') == col)['label'][0]
        assert row_label == label


# ---------------------------------------------------------------------------
# Tests de create_dimension_tables()
# ---------------------------------------------------------------------------

# Test de la création des tables de dimension
def test_create_dimension_tables(schema_builder):
    """Test the build of the dimension tables.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    # Création préalable de la table des méta-données (prérequis)
    schema_builder.create_metadata_table()
    # Création des tables de dimension
    dim_tables = schema_builder.create_dimension_tables()

    # Vérification du type renvoyé et des colonnes catégorielles présentes
    assert isinstance(dim_tables, dict)
    assert 'category' in dim_tables
    assert 'status' in dim_tables

    # Vérification de la structure de chaque table de dimension
    for table in dim_tables.values():
        assert isinstance(table, nw.DataFrame)
        assert 'value' in table.columns
        assert 'label' in table.columns


# ---------------------------------------------------------------------------
# Tests de create_fact_table()
# ---------------------------------------------------------------------------

# Test de la création de la table des faits
def test_create_fact_table(schema_builder, sample_df):
    """Test the build of the fact table.

    Args:
        schema_builder: SchemaBuilder fixture.
        sample_df: Sample polars DataFrame.
    """
    # Création préalable des prérequis
    schema_builder.create_metadata_table()
    schema_builder.create_dimension_tables()
    # Création de la table des faits
    fact_table = schema_builder.create_fact_table()

    # Vérification du type et de la longueur
    assert isinstance(fact_table, nw.DataFrame)
    assert len(fact_table) == len(sample_df)

    # Vérification que les colonnes catégorielles ont été remplacées par des entiers
    category_dtype = fact_table.schema['category']
    status_dtype = fact_table.schema['status']
    assert isinstance(category_dtype, (nw.Int8, nw.Int16, nw.Int32, nw.Int64,
                                       nw.UInt8, nw.UInt16, nw.UInt32, nw.UInt64))
    assert isinstance(status_dtype, (nw.Int8, nw.Int16, nw.Int32, nw.Int64,
                                     nw.UInt8, nw.UInt16, nw.UInt32, nw.UInt64))


# ---------------------------------------------------------------------------
# Tests de build()
# ---------------------------------------------------------------------------

# Test de la construction complète du schéma
def test_build_complete_schema(schema_builder, column_labels):
    """Test the build of the complete schema (metadata, dimensions, fact table).

    Args:
        schema_builder: SchemaBuilder fixture.
        column_labels: Dict mapping column names to custom labels.
    """
    # Construction du schéma complet
    metadata, dim_tables, fact_table = schema_builder.build(column_labels)

    # Vérification des types de chaque composant
    assert isinstance(metadata, nw.DataFrame)
    assert isinstance(dim_tables, dict)
    assert isinstance(fact_table, nw.DataFrame)


# ---------------------------------------------------------------------------
# Tests liés à categorical_threshold=None
# ---------------------------------------------------------------------------

# Test que categorical_threshold=None ne produit aucune colonne catégorielle
def test_categorical_threshold_none_no_categorical_columns(sample_df):
    """Test that no column is marked categorical when categorical_threshold=None.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = SchemaBuilder(sample_df, categorical_threshold=None)

    metadata = builder.create_metadata_table()

    # Aucune colonne ne doit être marquée comme catégorielle
    assert not metadata['is_categorical'].to_list().__contains__(True)


# Test que categorical_threshold=None produit un dictionnaire de dimensions vide
def test_categorical_threshold_none_empty_dimension_tables(sample_df):
    """Test that no dimension tables are created when categorical_threshold=None.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = SchemaBuilder(sample_df, categorical_threshold=None)

    builder.create_metadata_table()
    dim_tables = builder.create_dimension_tables()

    # Le dictionnaire des tables de dimension doit être vide
    assert isinstance(dim_tables, dict)
    assert len(dim_tables) == 0


# ---------------------------------------------------------------------------
# Tests liés aux clés primaires
# ---------------------------------------------------------------------------

# Test que UserWarning est levé quand aucune clé primaire n'est fournie
def test_warning_when_no_primary_keys(sample_df):
    """Test that a UserWarning is raised when no primary keys are specified.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with pytest.warns(UserWarning, match="No primary key"):
        SchemaBuilder(sample_df, categorical_threshold=4)


# Test qu'aucun avertissement n'est levé quand des clés primaires sont fournies
def test_no_warning_when_primary_keys_provided(sample_df):
    """Test that no UserWarning is raised when primary_keys is provided.

    Args:
        sample_df: Sample polars DataFrame.
    """
    # Traitement des warnings comme des erreurs pour détecter toute émission non attendue
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Ne doit pas lever d'exception
        SchemaBuilder(sample_df, categorical_threshold=4, primary_keys=['id'])


# Test que les clés primaires sont marquées dans les méta-données
def test_primary_keys_marked_in_metadata(sample_df):
    """Test that primary key columns are marked as is_primary_key=True in metadata.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(sample_df, categorical_threshold=4, primary_keys=['id'])
    metadata = builder.create_metadata_table()

    # Vérification que la colonne 'id' est marquée comme clé primaire
    id_pk = metadata.filter(nw.col('name') == 'id')['is_primary_key'][0]
    assert id_pk is True

    # Vérification que les autres colonnes ne sont pas marquées comme clés primaires
    value_pk = metadata.filter(nw.col('name') == 'value')['is_primary_key'][0]
    assert value_pk is False


# Test qu'une ValueError est levée pour une clé primaire inexistante
def test_invalid_primary_key_raises_value_error(sample_df):
    """Test that a ValueError is raised when a primary key column does not exist.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with pytest.raises(ValueError, match="nonexistent_col"):
        SchemaBuilder(sample_df, primary_keys=['nonexistent_col'])
