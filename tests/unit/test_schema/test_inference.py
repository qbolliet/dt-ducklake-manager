# Importation des modules
# Modules de base
import warnings
from typing import Any

import narwhals as nw
import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.schema import SchemaBuilder

# ---------------------------------------------------------------------------
# Fixture locale
# ---------------------------------------------------------------------------


# Initialisation d'une instance de SchemaBuilder utilisée dans l'ensemble des tests
@pytest.fixture
def schema_builder(sample_df: Any) -> SchemaBuilder:
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
def test_schema_builder_initialization(schema_builder: Any, sample_df: Any) -> None:
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
    assert schema_builder.df.to_native().equals(expected.to_native())  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Tests de create_metadata_table()
# ---------------------------------------------------------------------------


# Test de la création de la table des méta-données
def test_create_metadata_table(schema_builder: Any) -> None:
    """Test the build of the metadata table.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    # Création de la table des méta-données
    metadata = schema_builder.create_metadata_table()

    # Vérification du type renvoyé (narwhals DataFrame)
    assert isinstance(metadata, nw.DataFrame)
    # Vérification de l'existence de chacune des colonnes attendues
    assert "name" in metadata.columns
    assert "label" in metadata.columns
    assert "is_categorical_forced" not in metadata.columns
    assert "sql_type" in metadata.columns
    assert "is_categorical" in metadata.columns
    assert "is_primary_key" in metadata.columns

    # Vérification de la bonne détection des variables catégorielles
    cat_filter = metadata.filter(nw.col("name") == "category")["is_categorical"][0]
    status_filter = metadata.filter(nw.col("name") == "status")["is_categorical"][0]
    high_card_filter = metadata.filter(nw.col("name") == "high_cardinality")[
        "is_categorical"
    ][0]
    assert cat_filter is True
    assert status_filter is True
    assert high_card_filter is False


# Test de l'ajout de labels personnalisés
# lors de la création de la table des méta-données
def test_create_metadata_table_with_labels(
    schema_builder: Any, column_labels: Any
) -> None:
    """Test the build of the metadata table with custom column labels.

    Args:
        schema_builder: SchemaBuilder fixture.
        column_labels: Dict mapping column names to custom labels.
    """
    # Création de la table des méta-données avec les labels
    metadata = schema_builder.create_metadata_table(column_labels)

    # Vérification de la bonne association des labels fournis
    for col, label in column_labels.items():
        row_label = metadata.filter(nw.col("name") == col)["label"][0]
        assert row_label == label


# ---------------------------------------------------------------------------
# Tests de categorical_overrides
# ---------------------------------------------------------------------------


# Test qu'une colonne peut être forcée catégorielle au-delà du seuil
def test_categorical_override_forces_true(sample_df: Any) -> None:
    """Test that a column above the threshold can be forced as categorical.

    'high_cardinality' has 5 modalities for a threshold of 4, so it would be
    inferred as non-categorical.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(
        sample_df,
        categorical_threshold=4,
        primary_keys=["id"],
        categorical_overrides={"high_cardinality": True},
    )
    metadata = builder.create_metadata_table()

    row = metadata.filter(nw.col("name") == "high_cardinality")
    assert row["is_categorical"][0] is True


# Test qu'une colonne sous le seuil peut être forcée non catégorielle
def test_categorical_override_forces_false(sample_df: Any) -> None:
    """Test that a column below the threshold can be forced as non-categorical.

    'category' has 3 modalities for a threshold of 4, so it would be inferred as
    categorical.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(
        sample_df,
        categorical_threshold=4,
        primary_keys=["id"],
        categorical_overrides={"category": False},
    )
    metadata = builder.create_metadata_table()

    row = metadata.filter(nw.col("name") == "category")
    assert row["is_categorical"][0] is False


# Test qu'une colonne absente des forçages reste pilotée par le seuil
def test_categorical_override_leaves_other_columns_unforced(sample_df: Any) -> None:
    """Test that columns absent from categorical_overrides stay threshold-driven.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(
        sample_df,
        categorical_threshold=4,
        primary_keys=["id"],
        categorical_overrides={"high_cardinality": True},
    )
    metadata = builder.create_metadata_table()

    row = metadata.filter(nw.col("name") == "category")
    assert row["is_categorical"][0] is True


# Test qu'une colonne inconnue dans categorical_overrides lève une ValueError
def test_categorical_override_unknown_column_raises(sample_df: Any) -> None:
    """Test that an unknown column in categorical_overrides raises a ValueError.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with pytest.raises(ValueError, match="categorical_overrides"):
        SchemaBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
            categorical_overrides={"ghost_column": True},
        )


# ---------------------------------------------------------------------------
# Tests de create_fact_table()
# ---------------------------------------------------------------------------


# Test de la création de la table des faits
def test_create_fact_table(schema_builder: Any, sample_df: Any) -> None:
    """Test that the fact table holds the input values verbatim.

    Categorical columns keep their original labels: no synthetic code is ever
    substituted.

    Args:
        schema_builder: SchemaBuilder fixture.
        sample_df: Sample polars DataFrame.
    """
    # Création préalable des méta-données (prérequis)
    schema_builder.create_metadata_table()
    # Création de la table des faits
    fact_table = schema_builder.create_fact_table()

    # Vérification du type et de la longueur
    assert isinstance(fact_table, nw.DataFrame)
    assert len(fact_table) == len(sample_df)

    # Vérification que les colonnes catégorielles restent textuelles
    assert isinstance(fact_table.schema["category"], nw.String)
    assert isinstance(fact_table.schema["status"], nw.String)

    # Vérification que les libellés d'origine sont conservés tels quels
    assert fact_table["category"].to_list() == sample_df["category"].to_list()
    assert fact_table["status"].to_list() == sample_df["status"].to_list()


# ---------------------------------------------------------------------------
# Tests de build()
# ---------------------------------------------------------------------------


# Test de la construction complète du schéma
def test_build_complete_schema(schema_builder: Any, column_labels: Any) -> None:
    """Test the build of the complete schema (metadata and fact table).

    Args:
        schema_builder: SchemaBuilder fixture.
        column_labels: Dict mapping column names to custom labels.
    """
    # Construction du schéma complet
    metadata, fact_table = schema_builder.build(column_labels)

    # Vérification des types de chaque composant
    assert isinstance(metadata, nw.DataFrame)
    assert isinstance(fact_table, nw.DataFrame)
    # Une ligne de méta-données par colonne de la table des faits
    assert len(metadata) == len(fact_table.columns)


# ---------------------------------------------------------------------------
# Tests liés à categorical_threshold=None
# ---------------------------------------------------------------------------


# Test que categorical_threshold=None ne produit aucune colonne catégorielle
def test_categorical_threshold_none_no_categorical_columns(sample_df: Any) -> None:
    """Test that no column is marked categorical when categorical_threshold=None.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = SchemaBuilder(sample_df, categorical_threshold=None)

    metadata = builder.create_metadata_table()

    # Aucune colonne ne doit être marquée comme catégorielle
    assert not metadata["is_categorical"].to_list().__contains__(True)


# Test que categorical_threshold=None laisse la table des faits intacte
def test_categorical_threshold_none_keeps_labels(sample_df: Any) -> None:
    """Test that the fact table keeps its labels when categorical_threshold=None.

    The threshold only drives the UI flag; it never changes what is stored.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = SchemaBuilder(sample_df, categorical_threshold=None)

    _, fact_table = builder.build()

    # Les libellés d'origine sont conservés quel que soit le seuil
    assert fact_table["category"].to_list() == sample_df["category"].to_list()


# ---------------------------------------------------------------------------
# Tests liés aux clés primaires
# ---------------------------------------------------------------------------


# Test que UserWarning est levé quand aucune clé primaire n'est fournie
def test_warning_when_no_primary_keys(sample_df: Any) -> None:
    """Test that a UserWarning is raised when no primary keys are specified.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with pytest.warns(UserWarning, match="No primary key"):
        SchemaBuilder(sample_df, categorical_threshold=4)


# Test qu'aucun avertissement n'est levé quand des clés primaires sont fournies
def test_no_warning_when_primary_keys_provided(sample_df: Any) -> None:
    """Test that no UserWarning is raised when primary_keys is provided.

    Args:
        sample_df: Sample polars DataFrame.
    """
    # Traitement des warnings comme des erreurs
    # pour détecter toute émission non attendue
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        # Ne doit pas lever d'exception
        SchemaBuilder(sample_df, categorical_threshold=4, primary_keys=["id"])


# Test que les clés primaires sont marquées dans les méta-données
def test_primary_keys_marked_in_metadata(sample_df: Any) -> None:
    """Test that primary key columns are marked as is_primary_key=True in metadata.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(sample_df, categorical_threshold=4, primary_keys=["id"])
    metadata = builder.create_metadata_table()

    # Vérification que la colonne 'id' est marquée comme clé primaire
    id_pk = metadata.filter(nw.col("name") == "id")["is_primary_key"][0]
    assert id_pk is True

    # Vérification que les autres colonnes ne sont pas marquées comme clés primaires
    value_pk = metadata.filter(nw.col("name") == "value")["is_primary_key"][0]
    assert value_pk is False


# Test qu'une ValueError est levée pour une clé primaire inexistante
def test_invalid_primary_key_raises_value_error(sample_df: Any) -> None:
    """Test that a ValueError is raised when a primary key column does not exist.

    Args:
        sample_df: Sample polars DataFrame.
    """
    with pytest.raises(ValueError, match="nonexistent_col"):
        SchemaBuilder(sample_df, primary_keys=["nonexistent_col"])


# ---------------------------------------------------------------------------
# Tests des valeurs manquantes dans les colonnes catégorielles
# ---------------------------------------------------------------------------


# Invariant : un NULL/None dans une colonne catégorielle doit rester NULL dans la
# fact_table, et les autres positions conserver leur libellé d'origine.
def test_null_in_categorical_column_preserved_in_fact_table() -> None:
    """Test that NULL values in a categorical column stay NULL in the fact table.

    The non-null positions must keep their original labels verbatim.
    """
    import polars as pl

    df_with_nulls = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "category": ["A", None, "B", "A", None],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = SchemaBuilder(
            df_with_nulls, categorical_threshold=4, primary_keys=["id"]
        )

    metadata, fact_table = builder.build()

    # La colonne 'category' doit être signalée comme catégorielle
    is_cat = metadata.filter(nw.col("name") == "category")["is_categorical"][0]
    assert is_cat is True

    # La fact_table conserve les NULL aux positions originales (lignes id=2 et id=5)
    category_col = fact_table["category"].to_list()
    # Les indices 1 et 4 (id=2 et id=5) doivent rester NULL/None
    assert category_col[1] is None
    assert category_col[4] is None
    # Les autres positions conservent leur libellé d'origine
    assert category_col[0] == "A"
    assert category_col[2] == "B"
    assert category_col[3] == "A"


# Test que tous les NULL d'une colonne catégorielle restent NULL dans la fact_table
def test_all_null_rows_preserved_as_null_in_fact_table() -> None:
    """Test that every NULL row in a categorical column stays NULL in fact_table.

    The number of NULL values in the categorical column of the fact_table must
    equal the number of NULL values in the source DataFrame.
    """
    import polars as pl

    df_with_nulls = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "status": ["active", None, "inactive", None, "active", None],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = SchemaBuilder(
            df_with_nulls, categorical_threshold=4, primary_keys=["id"]
        )

    _, fact_table = builder.build()

    # Comptage des NULL dans la fact_table : doit correspondre au DataFrame source
    status_values = fact_table["status"].to_list()
    status_nulls = sum(1 for v in status_values if v is None)
    assert status_nulls == 3

    # Les positions non nulles conservent leur libellé d'origine
    assert status_values[0] == "active"
    assert status_values[2] == "inactive"


# ---------------------------------------------------------------------------
# Tests de column_metadata (champs d'UI de la table metadata)
# ---------------------------------------------------------------------------


# Test de la propagation des champs d'UI vers la table de métadonnées
def test_column_metadata_populates_ui_fields(schema_builder: Any) -> None:
    """Test that column_metadata fills the UI fields and upper-cases the aggregation.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    metadata = schema_builder.create_metadata_table(
        column_metadata={
            "value": {
                "unit": "€",
                "display_format": ",.2f",
                "family": "kpi",
                "description": "the observed value",
                "default_aggregation": "sum",
            }
        }
    )

    # Toutes les colonnes d'UI sont présentes dans la table
    for field in (
        "unit",
        "display_format",
        "family",
        "description",
        "default_aggregation",
    ):
        assert field in metadata.columns

    row = metadata.filter(nw.col("name") == "value")
    assert row["unit"][0] == "€"
    assert row["display_format"][0] == ",.2f"
    assert row["family"][0] == "kpi"
    assert row["description"][0] == "the observed value"
    # Normalisation en majuscules à l'écriture
    assert row["default_aggregation"][0] == "SUM"


# Test que les colonnes non renseignées portent NULL sur les champs d'UI
def test_column_metadata_absent_columns_are_null(schema_builder: Any) -> None:
    """Test that columns not referenced in column_metadata keep NULL UI fields.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    metadata = schema_builder.create_metadata_table(
        column_metadata={"value": {"unit": "€"}}
    )

    # La colonne 'id' n'est pas référencée : tous ses champs d'UI restent nuls
    id_row = metadata.filter(nw.col("name") == "id")
    for field in (
        "unit",
        "display_format",
        "family",
        "description",
        "default_aggregation",
    ):
        assert id_row[field][0] is None


# Test du renseignement partiel des champs d'UI
def test_column_metadata_partial_fields(schema_builder: Any) -> None:
    """Test that only the supplied UI fields are set, the others staying NULL.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    metadata = schema_builder.create_metadata_table(
        column_metadata={"value": {"unit": "MW", "default_aggregation": "avg"}}
    )

    row = metadata.filter(nw.col("name") == "value")
    assert row["unit"][0] == "MW"
    assert row["default_aggregation"][0] == "AVG"
    # Champs non fournis : NULL
    assert row["display_format"][0] is None
    assert row["family"][0] is None
    assert row["description"][0] is None


# Test que column_metadata prime sur column_labels pour le libellé
def test_column_metadata_label_wins_over_column_labels(schema_builder: Any) -> None:
    """Test that a label given in column_metadata overrides column_labels.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    metadata = schema_builder.create_metadata_table(
        column_labels={"value": "From column_labels"},
        column_metadata={"value": {"label": "From column_metadata"}},
    )

    label = metadata.filter(nw.col("name") == "value")["label"][0]
    assert label == "From column_metadata"


# Test qu'une colonne inconnue dans column_metadata lève une ValueError
def test_column_metadata_unknown_column_raises(schema_builder: Any) -> None:
    """Test that referencing a missing column in column_metadata raises ValueError.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    with pytest.raises(ValueError, match="do not exist in the DataFrame"):
        schema_builder.create_metadata_table(
            column_metadata={"not_a_column": {"unit": "€"}}
        )


# Test qu'une clé inconnue dans un sous-dictionnaire lève une ValueError listant la clé
def test_column_metadata_unknown_key_raises(schema_builder: Any) -> None:
    """Test that an unknown sub-dictionary key raises a ValueError listing it.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    with pytest.raises(ValueError, match="Unknown column_metadata key.*'color'"):
        schema_builder.create_metadata_table(
            column_metadata={"value": {"unit": "€", "color": "red"}}
        )


# Test qu'une agrégation par défaut invalide lève une ValueError
def test_column_metadata_invalid_default_aggregation_raises(
    schema_builder: Any,
) -> None:
    """Test that an invalid default_aggregation raises a ValueError.

    Args:
        schema_builder: SchemaBuilder fixture.
    """
    with pytest.raises(ValueError, match="Invalid default_aggregation"):
        schema_builder.create_metadata_table(
            column_metadata={"value": {"default_aggregation": "TOTAL"}}
        )


# ---------------------------------------------------------------------------
# Tests des hiérarchies de colonnes (parent_name, §2.5)
# ---------------------------------------------------------------------------


# Test que le paramètre hierarchies renseigne parent_name
def test_hierarchies_param_sets_parent_name(sample_df: Any) -> None:
    """Test that the ``hierarchies`` constructor parameter writes ``parent_name``.

    ``category`` and ``status`` are both already categorical under
    categorical_threshold=4, so no forcing warning is expected here.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(
        sample_df,
        categorical_threshold=4,
        primary_keys=["id"],
        hierarchies={"category": "status"},
    )
    metadata = builder.create_metadata_table()

    row = metadata.filter(nw.col("name") == "category")
    assert row["parent_name"][0] == "status"


# Test que parent_name peut être renseigné via column_metadata uniquement
def test_parent_name_via_column_metadata_only(schema_builder: Any) -> None:
    """Test that ``column_metadata``'s ``parent_name`` key alone sets the hierarchy.

    Args:
        schema_builder: SchemaBuilder fixture (categorical_threshold=4).
    """
    metadata = schema_builder.create_metadata_table(
        column_metadata={"category": {"parent_name": "status"}}
    )
    row = metadata.filter(nw.col("name") == "category")
    assert row["parent_name"][0] == "status"


# Test qu'une colonne non catégorielle appartenant à une hiérarchie est forcée
def test_hierarchies_forces_non_categorical_column(sample_df: Any) -> None:
    """Test that a non-categorical hierarchy column is forced categorical with a
    warning.

    'high_cardinality' has 5 distinct values, above categorical_threshold=4, so it
    is NOT categorical by default. Declaring it as a child of 'status' in a
    hierarchy must force is_categorical=True, with a UserWarning, exactly as
    categorical_overrides would.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(
        sample_df,
        categorical_threshold=4,
        primary_keys=["id"],
        hierarchies={"high_cardinality": "status"},
    )
    with pytest.warns(UserWarning, match="high_cardinality.*hierarchy"):
        metadata = builder.create_metadata_table()

    row = metadata.filter(nw.col("name") == "high_cardinality")
    assert row["is_categorical"][0] is True
    assert row["parent_name"][0] == "status"


# Test qu'une colonne enfant inconnue dans hierarchies lève une ValueError à
# l'initialisation
def test_hierarchies_unknown_child_raises(sample_df: Any) -> None:
    """Test that an unknown child column in ``hierarchies`` raises ValueError.

    Args:
        sample_df: polars DataFrame fixture from conftest.
    """
    with pytest.raises(ValueError, match="do not exist in the DataFrame"):
        SchemaBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
            hierarchies={"not_a_column": "status"},
        )


# Test qu'une colonne parente inconnue dans hierarchies lève une ValueError à
# l'initialisation
def test_hierarchies_unknown_parent_raises(sample_df: Any) -> None:
    """Test that an unknown parent column in ``hierarchies`` raises ValueError.

    Args:
        sample_df: polars DataFrame fixture from conftest.
    """
    with pytest.raises(ValueError, match="do not exist in the DataFrame"):
        SchemaBuilder(
            sample_df,
            categorical_threshold=4,
            primary_keys=["id"],
            hierarchies={"category": "not_a_column"},
        )


# Test qu'une colonne parente inconnue fournie via column_metadata lève une ValueError
def test_parent_name_unknown_parent_via_column_metadata_raises(
    schema_builder: Any,
) -> None:
    """Test that an unknown ``parent_name`` in ``column_metadata`` raises ValueError.

    Args:
        schema_builder: SchemaBuilder fixture (categorical_threshold=4).
    """
    with pytest.raises(ValueError, match="do not exist in the DataFrame"):
        schema_builder.create_metadata_table(
            column_metadata={"category": {"parent_name": "not_a_column"}}
        )


# Test qu'une auto-référence (A -> A) est détectée comme un cycle
def test_hierarchies_self_reference_raises(sample_df: Any) -> None:
    """Test that a column declared as its own parent raises a cycle ValueError.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(
        sample_df,
        categorical_threshold=4,
        primary_keys=["id"],
        hierarchies={"category": "category"},
    )
    with pytest.raises(ValueError, match="Cycle detected"):
        builder.create_metadata_table()


# Test qu'un cycle à deux colonnes (A -> B -> A) est détecté
def test_hierarchies_two_node_cycle_raises(sample_df: Any) -> None:
    """Test that a two-column cycle (A -> B -> A) raises a cycle ValueError.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(
        sample_df,
        categorical_threshold=4,
        primary_keys=["id"],
        hierarchies={"category": "status", "status": "category"},
    )
    with pytest.raises(ValueError, match="Cycle detected"):
        builder.create_metadata_table()


# Test que hierarchies et column_metadata contradictoires lèvent une ValueError
def test_hierarchies_and_column_metadata_conflict_raises(sample_df: Any) -> None:
    """Test that conflicting ``hierarchies`` and ``column_metadata`` raise ValueError.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(
        sample_df,
        categorical_threshold=4,
        primary_keys=["id"],
        hierarchies={"category": "status"},
    )
    with pytest.raises(ValueError, match="Conflicting parent_name"):
        builder.create_metadata_table(
            column_metadata={"category": {"parent_name": "high_cardinality"}}
        )


# Test que hierarchies et column_metadata cohérents ne lèvent aucune erreur
def test_hierarchies_and_column_metadata_agree_ok(sample_df: Any) -> None:
    """Test that ``hierarchies`` and ``column_metadata`` agreeing on the same parent
    does not raise.

    Args:
        sample_df: Sample polars DataFrame.
    """
    builder = SchemaBuilder(
        sample_df,
        categorical_threshold=4,
        primary_keys=["id"],
        hierarchies={"category": "status"},
    )
    metadata = builder.create_metadata_table(
        column_metadata={"category": {"parent_name": "status"}}
    )
    row = metadata.filter(nw.col("name") == "category")
    assert row["parent_name"][0] == "status"


# Test d'une hiérarchie profonde (5 niveaux)
def test_hierarchies_deep_chain() -> None:
    """Test that a 5-level column hierarchy is fully declared without error."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "l0": ["a", "b"],
            "l1": ["a1", "b1"],
            "l2": ["a2", "b2"],
            "l3": ["a3", "b3"],
            "l4": ["a4", "b4"],
        }
    )
    builder = SchemaBuilder(
        df,
        categorical_threshold=10,
        primary_keys=["id"],
        hierarchies={"l4": "l3", "l3": "l2", "l2": "l1", "l1": "l0"},
    )
    metadata = builder.create_metadata_table()

    expected_parent = {"l4": "l3", "l3": "l2", "l2": "l1", "l1": "l0", "l0": None}
    for col, parent in expected_parent.items():
        row = metadata.filter(nw.col("name") == col)
        assert row["parent_name"][0] == parent
        # Toutes les colonnes de la chaîne doivent être catégorielles
        assert row["is_categorical"][0] is True


# Test de deux hiérarchies indépendantes déclarées simultanément
def test_hierarchies_two_independent_trees() -> None:
    """Test that two unrelated column hierarchies can be declared together."""
    df = pl.DataFrame(
        {
            "id": [1, 2],
            "region": ["r1", "r2"],
            "departement": ["d1", "d2"],
            "category": ["c1", "c2"],
            "subcategory": ["s1", "s2"],
        }
    )
    builder = SchemaBuilder(
        df,
        categorical_threshold=10,
        primary_keys=["id"],
        hierarchies={"departement": "region", "subcategory": "category"},
    )
    metadata = builder.create_metadata_table()

    dep_row = metadata.filter(nw.col("name") == "departement")
    assert dep_row["parent_name"][0] == "region"
    subcat_row = metadata.filter(nw.col("name") == "subcategory")
    assert subcat_row["parent_name"][0] == "category"
