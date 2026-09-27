# Importation des modules
# Modules de base
import duckdb
import narwhals as nw
import polars as pl

# Module de tests
import pytest

# Gestionnaire de base, instanciable directement
from dt_ducklake_manager.operations._base import BaseSchemaManager
from dt_ducklake_manager.schema import DuckLakeTablesBuilder

# ---------------------------------------------------------------------------
# Fixture locale
# ---------------------------------------------------------------------------


# Initialisation d'une instance de BaseSchemaManager pour les tests
@pytest.fixture
def manager(built_ducklake_schema: duckdb.DuckDBPyConnection) -> BaseSchemaManager:
    """Create a BaseSchemaManager to test its methods.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.

    Returns:
        BaseSchemaManager: initialized with the test connection.
    """
    return BaseSchemaManager(connection=built_ducklake_schema, categorical_threshold=4)


# ===========================================================================
# Tests de _load_current_metadata()
# ===========================================================================


# Test que _load_current_metadata retourne un DataFrame narwhals avec les colonnes
# attendues
def test_load_current_metadata_returns_dataframe(manager: BaseSchemaManager) -> None:
    """Test that _load_current_metadata returns a narwhals DataFrame
    with expected columns.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    metadata = manager._load_current_metadata()
    # Vérification du type de retour
    assert isinstance(metadata, nw.DataFrame)
    # Vérification de la présence des colonnes de métadonnées
    assert "name" in metadata.columns
    assert "is_categorical" in metadata.columns
    assert "is_primary_key" in metadata.columns


# Test que _load_current_metadata retourne un DataFrame non vide pour un schéma
# construit
def test_load_current_metadata_non_empty(manager: BaseSchemaManager) -> None:
    """Test that _load_current_metadata returns a non-empty DataFrame
    for a built schema.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    metadata = manager._load_current_metadata()
    # Le schéma built_ducklake_schema a été construit avec sample_df qui a 6 colonnes
    assert len(metadata) > 0


# ===========================================================================
# Tests de _table_exists()
# ===========================================================================


# Test que _table_exists retourne True pour une table existante
def test_table_exists_fact_table(manager: BaseSchemaManager) -> None:
    """Test that _table_exists returns True for an existing table.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    assert manager._table_exists("fact_table") is True


# Test que _table_exists retourne True pour la table metadata
def test_table_exists_metadata(manager: BaseSchemaManager) -> None:
    """Test that _table_exists returns True for the metadata table.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    assert manager._table_exists("metadata") is True


# Test que _table_exists retourne False pour une table inexistante
def test_table_exists_nonexistent(manager: BaseSchemaManager) -> None:
    """Test that _table_exists returns False for a non-existent table.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    assert manager._table_exists("nonexistent_table_xyz") is False


# ===========================================================================
# Tests de _get_primary_key_columns()
# ===========================================================================


# Test que _get_primary_key_columns retourne la liste des clés primaires
def test_get_primary_key_columns(manager: BaseSchemaManager) -> None:
    """Test that _get_primary_key_columns returns the list of primary key columns.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    pks = manager._get_primary_key_columns()
    # Vérification du type de retour
    assert isinstance(pks, list)
    # Le schéma est construit avec primary_keys=['id'] dans built_ducklake_schema
    assert "id" in pks


# ===========================================================================
# Tests de _column_exists()
# ===========================================================================


# Test que _column_exists retourne True pour une colonne existante dans fact_table
def test_column_exists_true(manager: BaseSchemaManager) -> None:
    """Test that _column_exists returns True for an existing column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    assert manager._column_exists("id", "fact_table") is True


# Test que _column_exists retourne False pour une colonne inexistante
def test_column_exists_false(manager: BaseSchemaManager) -> None:
    """Test that _column_exists returns False for a non-existent column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    assert manager._column_exists("nonexistent_col", "fact_table") is False


# ===========================================================================
# Tests de _is_primary_key_column()
# ===========================================================================


# Test que _is_primary_key_column retourne True pour une colonne clé primaire
def test_is_primary_key_column_true(manager: BaseSchemaManager) -> None:
    """Test that _is_primary_key_column returns True for a primary key column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    assert manager._is_primary_key_column("id") is True


# Test que _is_primary_key_column retourne False pour une colonne non-clé primaire
def test_is_primary_key_column_false(manager: BaseSchemaManager) -> None:
    """Test that _is_primary_key_column returns False for a non-primary-key column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    assert manager._is_primary_key_column("value") is False


# ===========================================================================
# Tests de _invalidate_metadata_cache()
# ===========================================================================


# Test que _invalidate_metadata_cache vide le cache
def test_invalidate_metadata_cache(manager: BaseSchemaManager) -> None:
    """Test that _invalidate_metadata_cache sets the cache to None.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    # Chargement du cache
    _ = manager._load_current_metadata()
    assert manager._metadata_cache is not None

    # Invalidation du cache
    manager._invalidate_metadata_cache()
    assert manager._metadata_cache is None


# ===========================================================================
# Tests du statut catégoriel : inférence unique et correction manuelle
# ===========================================================================


# Test que le statut catégoriel n'est jamais recalculé après une écriture
def test_categorical_status_not_recomputed_on_existing_column(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that refreshing an existing metadata row never touches is_categorical.

    'category' is categorical at build time (3 values <= threshold 4). Re-recording
    it from a batch carrying 6 distinct values only refreshes its SQL type.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    batch = pl.DataFrame({"category": ["A", "B", "C", "D", "E", "F"]})
    manager._add_column_to_metadata("category", batch)
    assert manager._is_categorical_column("category") is True


# Test que l'inférence s'applique à la création d'une colonne, valeurs nulles exclues
def test_categorical_status_inferred_on_new_column(manager: BaseSchemaManager) -> None:
    """Test that a new textual column is inferred once, nulls excluded.

    Args:
        manager: BaseSchemaManager fixture with a built schema (threshold 4).
    """
    low = pl.DataFrame({"low": ["a", "b", None, "a", "c", "d", None]})
    high = pl.DataFrame({"high": ["a", "b", "c", "d", "e"]})
    manager._add_column_to_metadata("low", low)
    manager._add_column_to_metadata("high", high)
    assert manager._is_categorical_column("low") is True
    assert manager._is_categorical_column("high") is False


# Test de la correction manuelle du statut catégoriel
def test_update_column_metadata_sets_is_categorical(manager: BaseSchemaManager) -> None:
    """Test that update_column_metadata corrects is_categorical both ways.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("high_cardinality", is_categorical=True)
    assert manager._is_categorical_column("high_cardinality") is True
    manager.update_column_metadata("high_cardinality", is_categorical=False)
    assert manager._is_categorical_column("high_cardinality") is False


# Test qu'un statut catégoriel non booléen est refusé
@pytest.mark.parametrize("value", ["true", 1, None])
def test_update_column_metadata_is_categorical_must_be_bool(
    manager: BaseSchemaManager, value: object
) -> None:
    """Test that a non-bool is_categorical raises ValueError.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        value: Invalid value.
    """
    with pytest.raises(ValueError, match="is_categorical must be a bool"):
        manager.update_column_metadata("category", is_categorical=value)  # type: ignore[arg-type]


# Test qu'une colonne enfant d'une hiérarchie ne peut pas cesser d'être catégorielle
def test_update_column_metadata_refuses_non_categorical_hierarchy_child(
    manager: BaseSchemaManager,
) -> None:
    """Test that is_categorical=False is refused on a hierarchy child column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("category", parent_name="status")
    with pytest.raises(ValueError, match="hierarchy"):
        manager.update_column_metadata("category", is_categorical=False)


# Test qu'une colonne parente d'une hiérarchie ne peut pas cesser d'être catégorielle
def test_update_column_metadata_refuses_non_categorical_hierarchy_parent(
    manager: BaseSchemaManager,
) -> None:
    """Test that is_categorical=False is refused on a hierarchy parent column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("category", parent_name="status")
    with pytest.raises(ValueError, match="hierarchy"):
        manager.update_column_metadata("status", is_categorical=False)


# Test qu'un détachement de hiérarchie dans le même appel autorise la bascule
def test_update_column_metadata_detach_and_uncategorize_same_call(
    manager: BaseSchemaManager,
) -> None:
    """Test that clearing parent_name in the same call allows is_categorical=False.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("category", parent_name="status")
    manager.update_column_metadata("category", parent_name=None, is_categorical=False)
    assert manager._is_categorical_column("category") is False


# ===========================================================================
# Tests de _resolve_type_conflicts()
# ===========================================================================


# Test qu'un BIGINT enregistré n'est pas rétrogradé par un lot d'Int32
def test_resolve_type_conflicts_keeps_widest_integer(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that a stored BIGINT is not downgraded by a batch of Int32.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    # 'id' est enregistrée en BIGINT (polars infère Int64)
    metadata = manager._load_current_metadata()
    df = nw.from_native(
        pl.DataFrame({"id": pl.Series("id", [1, 2, 3], dtype=pl.Int32)}),
        eager_only=True,
    )

    manager._resolve_type_conflicts("id", df, metadata)

    # Le type le plus large est conservé
    stored = built_ducklake_schema.execute(
        "SELECT sql_type FROM metadata WHERE name = 'id'"
    ).fetchone()
    assert stored is not None
    assert stored[0] == "BIGINT"


# Test qu'un lot plus large élargit le type enregistré
def test_resolve_type_conflicts_widens_to_varchar(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that a textual batch widens a numeric column to VARCHAR.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    metadata = manager._load_current_metadata()
    df = nw.from_native(
        pl.DataFrame({"value": ["a", "b", "c"]}),
        eager_only=True,
    )

    manager._resolve_type_conflicts("value", df, metadata)

    stored = built_ducklake_schema.execute(
        "SELECT sql_type FROM metadata WHERE name = 'value'"
    ).fetchone()
    assert stored is not None
    assert stored[0] == "VARCHAR"


# Test qu'un lot entièrement nul ne modifie jamais le type enregistré
def test_resolve_type_conflicts_ignores_all_null_batch(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that an all-null batch never overwrites a known numeric type.

    Without the guard, narwhals would infer 'Null', which maps to VARCHAR and would
    wrongly promote the column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    metadata = manager._load_current_metadata()
    df = nw.from_native(
        pl.DataFrame({"value": pl.Series("value", [None, None], dtype=pl.Null)}),
        eager_only=True,
    )

    manager._resolve_type_conflicts("value", df, metadata)

    stored = built_ducklake_schema.execute(
        "SELECT sql_type FROM metadata WHERE name = 'value'"
    ).fetchone()
    assert stored is not None
    assert stored[0] == "DOUBLE"


# ===========================================================================
# Tests du support multi-schémas (qualification et isolation)
# ===========================================================================


# Test que _qualified préfixe le nom de table par le schéma du gestionnaire
def test_qualified_prefixes_schema(
    built_ducklake_schema: duckdb.DuckDBPyConnection,
) -> None:
    """Test that _qualified prefixes the table name with the manager's schema.

    The in-memory test connection has no attached catalog, so ``_qualified``
    quotes and schema-qualifies the name without a catalog prefix.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    mgr = BaseSchemaManager(connection=built_ducklake_schema, schema="predictions")
    assert mgr._qualified("fact_table") == '"predictions"."fact_table"'
    assert mgr._qualified("dataset_metadata") == '"predictions"."dataset_metadata"'


# Test que le schéma par défaut est 'main'
def test_default_schema_is_main(
    built_ducklake_schema: duckdb.DuckDBPyConnection,
) -> None:
    """Test that the default schema is 'main'.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    mgr = BaseSchemaManager(connection=built_ducklake_schema)
    assert mgr.schema == "main"
    assert mgr._qualified("metadata") == '"main"."metadata"'


# Test que l'alias du catalogue est conservé au même titre que le schéma
def test_catalog_alias_default_and_custom(
    built_ducklake_schema: duckdb.DuckDBPyConnection,
) -> None:
    """Test that ``catalog_alias`` defaults to 'db' and is stored when provided.

    The base manager carries the catalog alias alongside the schema so that later
    passes can qualify table references by the catalog.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    # Valeur par défaut
    default_mgr = BaseSchemaManager(connection=built_ducklake_schema)
    assert default_mgr.catalog_alias == "db"

    # Valeur explicite propagée jusqu'à la classe de base
    custom_mgr = BaseSchemaManager(
        connection=built_ducklake_schema, catalog_alias="my_lake"
    )
    assert custom_mgr.catalog_alias == "my_lake"


# Test que _table_exists est isolé par schéma : une table d'un schéma n'est pas vue
# depuis un autre schéma du même catalogue
def test_table_exists_isolated_by_schema(
    multi_schema_connection: duckdb.DuckDBPyConnection,
) -> None:
    """Test that _table_exists distinguishes tables across schemas.

    A ``fact_table`` exists in both schemas; a manager bound to one schema must not
    see tables of a schema where they were not built (here a third, empty schema).

    Args:
        multi_schema_connection: Connection with 'predictions' and 'shapley' schemas.
    """
    # Gestionnaires liés à chacun des deux schémas construits
    pred_mgr = BaseSchemaManager(
        connection=multi_schema_connection, schema="predictions"
    )
    shap_mgr = BaseSchemaManager(connection=multi_schema_connection, schema="shapley")

    # Chaque schéma voit bien ses propres tables
    assert pred_mgr._table_exists("fact_table") is True
    assert shap_mgr._table_exists("fact_table") is True

    # Un schéma vide ne voit aucune table fact_table
    empty_mgr = BaseSchemaManager(connection=multi_schema_connection, schema="main")
    assert empty_mgr._table_exists("fact_table") is False


# Test que les métadonnées chargées sont propres au schéma ciblé
def test_metadata_isolated_by_schema(
    multi_schema_connection: duckdb.DuckDBPyConnection,
) -> None:
    """Test that _load_current_metadata reads the targeted schema's metadata only.

    The 'shapley' schema carries a 'shap_value' column absent from 'predictions'.

    Args:
        multi_schema_connection: Connection with 'predictions' and 'shapley' schemas.
    """
    pred_mgr = BaseSchemaManager(
        connection=multi_schema_connection, schema="predictions"
    )
    shap_mgr = BaseSchemaManager(connection=multi_schema_connection, schema="shapley")

    pred_columns = set(pred_mgr._load_current_metadata()["name"].to_list())
    shap_columns = set(shap_mgr._load_current_metadata()["name"].to_list())

    # La colonne 'value' n'existe que dans predictions, 'shap_value' que dans shapley
    assert "value" in pred_columns
    assert "value" not in shap_columns
    assert "shap_value" in shap_columns
    assert "shap_value" not in pred_columns


# ===========================================================================
# Tests des champs d'UI de la table metadata
# ===========================================================================


# Test que _add_column_to_metadata insère NULL sur les champs d'UI
def test_add_column_to_metadata_inserts_null_ui_fields(
    manager: BaseSchemaManager,
) -> None:
    """Test that a newly discovered column gets NULL UI metadata fields.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    df = pl.DataFrame({"id": [1, 2, 3], "score": [0.1, 0.2, 0.3]})
    manager._add_column_to_metadata("score", df)

    row = manager.conn.execute(
        "SELECT unit, display_format, family, description, default_aggregation"
        " FROM metadata WHERE name = 'score'"
    ).fetchone()
    assert row == (None, None, None, None, None)


# Test que update_column_metadata renseigne les champs et invalide le cache
def test_update_column_metadata_sets_fields(manager: BaseSchemaManager) -> None:
    """Test that update_column_metadata writes the fields and refreshes the cache.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    # Amorçage du cache
    _ = manager._load_current_metadata()

    manager.update_column_metadata(
        "value",
        unit="€",
        display_format=",.2f",
        family="kpi",
        description="the value",
        default_aggregation="sum",
    )

    row = manager.conn.execute(
        "SELECT unit, display_format, family, description, default_aggregation"
        " FROM metadata WHERE name = 'value'"
    ).fetchone()
    assert row == ("€", ",.2f", "kpi", "the value", "SUM")

    # Le cache a été invalidé : la relecture reflète la mise à jour
    reloaded = manager._load_current_metadata()
    agg = reloaded.filter(nw.col("name") == "value")["default_aggregation"][0]
    assert agg == "SUM"


# Test que update_column_metadata peut aussi corriger le libellé
def test_update_column_metadata_updates_label(manager: BaseSchemaManager) -> None:
    """Test that update_column_metadata can also fix the display label.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("value", label="Corrected Label")
    row = manager.conn.execute(
        "SELECT label FROM metadata WHERE name = 'value'"
    ).fetchone()
    assert row[0] == "Corrected Label"


# Test que update_column_metadata sur une colonne absente lève une ValueError
def test_update_column_metadata_missing_column_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that update_column_metadata raises on a column absent from metadata.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="no row in the metadata table"):
        manager.update_column_metadata("not_a_column", unit="€")


# Test que update_column_metadata rejette un champ non autorisé
def test_update_column_metadata_unknown_field_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that update_column_metadata rejects a field outside the allowed set.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="Unknown metadata field.*'sql_type'"):
        manager.update_column_metadata("value", sql_type="DOUBLE")


# Test que update_column_metadata valide default_aggregation
def test_update_column_metadata_invalid_aggregation_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that update_column_metadata validates default_aggregation.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="Invalid default_aggregation"):
        manager.update_column_metadata("value", default_aggregation="TOTAL")


# Test que _add_column_to_metadata préserve les champs d'UI d'une colonne existante
def test_add_column_to_metadata_preserves_ui_fields_on_update(
    manager: BaseSchemaManager,
) -> None:
    """Test that refreshing an existing metadata row keeps its UI fields intact.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    # Renseignement initial des champs d'UI par le producteur
    manager.update_column_metadata(
        "value", unit="€", display_format=",.2f", default_aggregation="sum"
    )

    # Nouvel appel simulant un update de données (type inchangé)
    df = pl.DataFrame({"id": [1], "value": [9.9]})
    manager._add_column_to_metadata("value", df)

    row = manager.conn.execute(
        "SELECT unit, display_format, default_aggregation"
        " FROM metadata WHERE name = 'value'"
    ).fetchone()
    # Les champs d'UI ne sont pas remis à NULL
    assert row == ("€", ",.2f", "SUM")


# ===========================================================================
# Tests de update_column_metadata(parent_name=...) et de la hiérarchie (§2.5)
# ===========================================================================


# Test que update_column_metadata renseigne parent_name et force le statut catégoriel
def test_update_column_metadata_parent_name_forces_categorical(
    manager: BaseSchemaManager,
) -> None:
    """Test that setting parent_name forces both ends of the link categorical.

    'value' and 'date' are not categorical in the built schema fixture. Declaring
    'value' as a child of 'date' must force both to categorical, with a warning.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.warns(UserWarning, match="hierarchy"):
        manager.update_column_metadata("value", parent_name="date")

    row = manager.conn.execute(
        "SELECT parent_name, is_categorical FROM metadata WHERE name = 'value'"
    ).fetchone()
    assert row == ("date", True)

    parent_row = manager.conn.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'date'"
    ).fetchone()
    assert parent_row == (True,)


# Test que update_column_metadata refuse une colonne parente inexistante
def test_update_column_metadata_parent_name_missing_parent_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that a parent_name referencing an unknown column raises ValueError.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="no row in the metadata table"):
        manager.update_column_metadata("value", parent_name="not_a_column")


# Test que update_column_metadata refuse une auto-référence (cycle)
def test_update_column_metadata_parent_name_self_reference_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that setting a column as its own parent raises a cycle ValueError.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="Cycle detected"):
        manager.update_column_metadata("value", parent_name="value")


# Test que update_column_metadata refuse une modification créant un cycle
def test_update_column_metadata_parent_name_cycle_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that a change creating a cycle against the current metadata state raises.

    'category' is first declared as the parent of 'status'; then pointing
    'category' back at 'status' would close a two-column cycle.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("status", parent_name="category")
    with pytest.raises(ValueError, match="Cycle detected"):
        manager.update_column_metadata("category", parent_name="status")


# Test que parent_name=None est accepté et efface la valeur sans validation
def test_update_column_metadata_parent_name_clear(manager: BaseSchemaManager) -> None:
    """Test that clearing parent_name (None) works without hierarchy validation.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.warns(UserWarning, match="hierarchy"):
        manager.update_column_metadata("value", parent_name="date")

    # Effacement : aucune validation de forêt ni forçage catégoriel à ce stade
    manager.update_column_metadata("value", parent_name=None)

    row = manager.conn.execute(
        "SELECT parent_name FROM metadata WHERE name = 'value'"
    ).fetchone()
    assert row[0] is None


# Test que _clear_child_parent_references détache les colonnes enfants
def test_clear_child_parent_references_detaches_children(
    manager: BaseSchemaManager,
) -> None:
    """Test that _clear_child_parent_references NULLs out children's parent_name.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.warns(UserWarning, match="hierarchy"):
        manager.update_column_metadata("value", parent_name="date")
    detached = manager._clear_child_parent_references("date")
    assert detached == ["value"]

    row = manager.conn.execute(
        "SELECT parent_name FROM metadata WHERE name = 'value'"
    ).fetchone()
    assert row[0] is None


# Test que _clear_child_parent_references ne fait rien pour une colonne sans enfant
def test_clear_child_parent_references_no_children(manager: BaseSchemaManager) -> None:
    """Test that _clear_child_parent_references is a no-op absent any children.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    detached = manager._clear_child_parent_references("value")
    assert detached == []


# ===========================================================================
# Tests de update_column_metadata(label_for=...) et des colonnes de libellés (§2.6)
# ===========================================================================

# NOTE : sur le schéma construit (sample_df), 'status' -> 'category' respecte la
# dépendance fonctionnelle (chaque catégorie a un statut unique), alors que
# 'category' -> 'status' la viole ('active' correspond aux catégories 'A' et 'C').


# Test qu'une cible inexistante lève une ValueError
def test_update_column_metadata_label_for_missing_target_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that a label_for referencing an unknown column raises ValueError.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="does not exist"):
        manager.update_column_metadata("category", label_for="not_a_column")


# Test qu'une colonne pointant vers elle-même lève une ValueError
def test_update_column_metadata_label_for_self_reference_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that setting a column as its own label_for target raises ValueError.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="own label_for target"):
        manager.update_column_metadata("category", label_for="category")


# Test qu'une chaîne de libellés (cible elle-même colonne de libellés) lève une erreur
def test_update_column_metadata_label_for_chain_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that targeting an existing label column raises a chaining ValueError.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("status", label_for="category")
    with pytest.raises(ValueError, match="chaining"):
        manager.update_column_metadata("high_cardinality", label_for="status")


# Test qu'une colonne de libellés clé primaire lève une ValueError
def test_update_column_metadata_label_for_primary_key_raises() -> None:
    """Test that a VARCHAR label column that is a primary key raises ValueError."""
    df = pl.DataFrame({"code": ["01", "02"], "value_col": [1, 2]})
    conn = duckdb.connect(":memory:")
    DuckLakeTablesBuilder(
        df, categorical_threshold=10, primary_keys=["code"], connection=conn
    ).build_schema()
    manager = BaseSchemaManager(connection=conn, categorical_threshold=10)

    with pytest.raises(ValueError, match="primary key"):
        manager.update_column_metadata("code", label_for="value_col")


# Test qu'une colonne de libellés non VARCHAR lève une ValueError
def test_update_column_metadata_label_for_non_varchar_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that a non-VARCHAR label column raises ValueError.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="VARCHAR"):
        manager.update_column_metadata("value", label_for="category")


# Test que label_for est refusé sur une colonne ayant déjà une parente
def test_update_column_metadata_label_for_refuses_hierarchy_child(
    manager: BaseSchemaManager,
) -> None:
    """Test that label_for is refused on a column that already has a parent_name.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.warns(UserWarning, match="hierarchy"):
        manager.update_column_metadata("high_cardinality", parent_name="date")
    with pytest.raises(ValueError, match="hierarchy"):
        manager.update_column_metadata("high_cardinality", label_for="category")


# Test que label_for est refusé sur une colonne parente d'une hiérarchie
def test_update_column_metadata_label_for_refuses_hierarchy_parent(
    manager: BaseSchemaManager,
) -> None:
    """Test that label_for is refused on a column that is a hierarchy parent.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.warns(UserWarning, match="hierarchy"):
        manager.update_column_metadata("high_cardinality", parent_name="date")
    with pytest.raises(ValueError, match="hierarchy"):
        manager.update_column_metadata("date", label_for="value")


# Test que parent_name est refusé sur une colonne de libellés
def test_update_column_metadata_refuses_parent_name_on_label_column(
    manager: BaseSchemaManager,
) -> None:
    """Test that parent_name is refused on a column that already has label_for.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("status", label_for="category")
    with pytest.raises(ValueError, match="label column"):
        manager.update_column_metadata("status", parent_name="date")


# Test qu'une violation de la dépendance fonctionnelle sur toute la table lève une
# ValueError
def test_update_column_metadata_label_for_functional_violation_raises(
    manager: BaseSchemaManager,
) -> None:
    """Test that declaring a label_for violating the functional dependency raises.

    'category' -> 'status' violates the dependency: 'status'='active' maps to both
    'category'='A' and 'category'='C'.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="Functional dependency"):
        manager.update_column_metadata("category", label_for="status")


# Test qu'une déclaration valide de label_for est acceptée
def test_update_column_metadata_label_for_accepted(manager: BaseSchemaManager) -> None:
    """Test that a label_for respecting the functional dependency is accepted.

    label_for has no effect on is_categorical, unlike parent_name.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    was_categorical = manager.conn.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'status'"
    ).fetchone()[0]

    manager.update_column_metadata("status", label_for="category")

    row = manager.conn.execute(
        "SELECT label_for, is_categorical FROM metadata WHERE name = 'status'"
    ).fetchone()
    assert row == ("category", was_categorical)


# Test que label_for=None efface le lien sans validation
def test_update_column_metadata_label_for_clear(manager: BaseSchemaManager) -> None:
    """Test that clearing label_for (None) works without any validation.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("status", label_for="category")
    manager.update_column_metadata("status", label_for=None)

    row = manager.conn.execute(
        "SELECT label_for FROM metadata WHERE name = 'status'"
    ).fetchone()
    assert row[0] is None


# Test que _clear_label_for_references détache les colonnes de libellés
def test_clear_label_for_references_detaches_label_columns(
    manager: BaseSchemaManager,
) -> None:
    """Test that _clear_label_for_references NULLs out label columns' label_for.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("status", label_for="category")
    detached = manager._clear_label_for_references("category")
    assert detached == ["status"]

    row = manager.conn.execute(
        "SELECT label_for FROM metadata WHERE name = 'status'"
    ).fetchone()
    assert row[0] is None


# Test que _clear_label_for_references ne fait rien pour un code sans libellé
def test_clear_label_for_references_no_label_columns(
    manager: BaseSchemaManager,
) -> None:
    """Test that _clear_label_for_references is a no-op absent any label column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    detached = manager._clear_label_for_references("value")
    assert detached == []


# Test que _get_label_columns_for_code lit les colonnes de libellés d'un code
def test_get_label_columns_for_code(manager: BaseSchemaManager) -> None:
    """Test that _get_label_columns_for_code returns the declared label columns.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_column_metadata("status", label_for="category")
    assert manager._get_label_columns_for_code("category") == ["status"]
    assert manager._get_label_columns_for_code("value") == []


# ===========================================================================
# Tests de _get_cluster_by_columns() et update_cluster_by() (§5.3)
# ===========================================================================


# Test que _get_cluster_by_columns lit la valeur par défaut (clé primaire)
def test_get_cluster_by_columns_reads_default(manager: BaseSchemaManager) -> None:
    """Test that _get_cluster_by_columns reads the primary-key default.

    Args:
        manager: BaseSchemaManager fixture with a built schema (primary_keys=['id']).
    """
    # built_ducklake_schema est construit sans cluster_by explicite : défaut = ['id']
    assert manager._get_cluster_by_columns() == ["id"]


# Test que _get_cluster_by_columns retourne None si dataset_metadata est absente
def test_get_cluster_by_columns_missing_table_returns_none() -> None:
    """Test that _get_cluster_by_columns returns None without dataset_metadata."""
    manager = BaseSchemaManager(connection=duckdb.connect(":memory:"))
    assert manager._get_cluster_by_columns() is None


# Test que update_cluster_by persiste la nouvelle valeur
def test_update_cluster_by_persists_value(manager: BaseSchemaManager) -> None:
    """Test that update_cluster_by writes the new column list as JSON.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_cluster_by(["category", "id"])
    assert manager._get_cluster_by_columns() == ["category", "id"]


# Test que update_cluster_by ne modifie pas les données de fact_table
def test_update_cluster_by_does_not_rewrite_data(manager: BaseSchemaManager) -> None:
    """Test that update_cluster_by is a metadata-only change.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    before = manager.conn.execute("SELECT * FROM fact_table ORDER BY id").fetchall()
    manager.update_cluster_by(["category"])
    after = manager.conn.execute("SELECT * FROM fact_table ORDER BY id").fetchall()
    assert before == after


# Test que update_cluster_by rejette une liste vide
def test_update_cluster_by_empty_raises(manager: BaseSchemaManager) -> None:
    """Test that update_cluster_by raises ValueError on an empty column list.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="columns must not be empty"):
        manager.update_cluster_by([])


# Test que update_cluster_by rejette une colonne absente de fact_table
def test_update_cluster_by_unknown_column_raises(manager: BaseSchemaManager) -> None:
    """Test that update_cluster_by raises ValueError on an unknown column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(ValueError, match="cluster_by columns"):
        manager.update_cluster_by(["not_a_column"])


# Tests de _remove_from_cluster_by() (§4.3, retrait au drop d'une colonne)
# ===========================================================================


# Test que _remove_from_cluster_by retire une colonne parmi plusieurs
def test_remove_from_cluster_by_drops_one_of_several(
    manager: BaseSchemaManager,
) -> None:
    """Test that _remove_from_cluster_by removes only the targeted column.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_cluster_by(["id", "category"])
    manager._remove_from_cluster_by("category")
    assert manager._get_cluster_by_columns() == ["id"]


# Test que _remove_from_cluster_by remet cluster_by à NULL une fois vidée
def test_remove_from_cluster_by_falls_back_to_null_when_emptied(
    manager: BaseSchemaManager,
) -> None:
    """Test that emptying cluster_by resets it to NULL, not an empty list.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    manager.update_cluster_by(["category"])
    manager._remove_from_cluster_by("category")
    assert manager._get_cluster_by_columns() is None


# Test que _remove_from_cluster_by ne fait rien si la colonne n'y est pas
def test_remove_from_cluster_by_noop_when_absent(manager: BaseSchemaManager) -> None:
    """Test that _remove_from_cluster_by is a no-op for an unrelated column.

    Args:
        manager: BaseSchemaManager fixture with a built schema (cluster_by defaults to
            ['id']).
    """
    manager._remove_from_cluster_by("category")
    assert manager._get_cluster_by_columns() == ["id"]


# ===========================================================================
# Tests de _transaction() : point d'accroche transactionnel unique
# ===========================================================================


# Test qu'une transaction validée rend les écritures durables
def test_transaction_commits_on_success(manager: BaseSchemaManager) -> None:
    """Test that a block leaving _transaction normally is committed.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with manager._transaction("test_commit"):
        manager.conn.execute("DELETE FROM fact_table WHERE id = 1")

    # L'écriture survit à la sortie du bloc
    assert manager.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0] == 4
    # La transaction est refermée
    assert manager._in_transaction is False


# Test qu'une exception annule l'ensemble des écritures du bloc
def test_transaction_rolls_back_on_exception(manager: BaseSchemaManager) -> None:
    """Test that an exception inside _transaction rolls every write back.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    initial_count = manager.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[
        0
    ]

    with pytest.raises(RuntimeError, match="boom"):
        with manager._transaction("test_rollback"):
            manager.conn.execute("DELETE FROM fact_table WHERE id = 1")
            manager.conn.execute("DELETE FROM fact_table WHERE id = 2")
            raise RuntimeError("boom")

    # La base est revenue à son état initial
    count = manager.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count == initial_count
    assert manager._in_transaction is False


# Test qu'un ALTER TABLE est lui aussi annulé (DDL transactionnel dans DuckDB)
def test_transaction_rolls_back_ddl(manager: BaseSchemaManager) -> None:
    """Test that a column added inside a rolled-back transaction does not survive.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(RuntimeError):
        with manager._transaction("test_rollback_ddl"):
            manager.conn.execute("ALTER TABLE fact_table ADD COLUMN score DOUBLE")
            raise RuntimeError("boom")

    assert "score" not in manager._get_fact_table_columns()


# Test que use_transaction=False laisse les écritures partielles en place
def test_transaction_disabled_keeps_partial_state(manager: BaseSchemaManager) -> None:
    """Test that use_transaction=False runs in autocommit mode.

    Without a transaction, a failure mid-block leaves whatever was already
    written behind — the documented difference between the two modes.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    initial_count = manager.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[
        0
    ]

    with pytest.raises(RuntimeError):
        with manager._transaction("test_autocommit", use_transaction=False):
            manager.conn.execute("DELETE FROM fact_table WHERE id = 1")
            raise RuntimeError("boom")

    # La suppression a bien persisté
    count = manager.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count == initial_count - 1


# Test qu'un appel imbriqué ne rouvre pas de transaction
def test_transaction_nested_call_reuses_outer(manager: BaseSchemaManager) -> None:
    """Test that a nested _transaction reuses the transaction already open.

    DuckDB rejects a BEGIN inside a BEGIN; the nested block must neither fail nor
    commit early, so an exception raised after it still rolls everything back.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    initial_count = manager.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[
        0
    ]

    with pytest.raises(RuntimeError):
        with manager._transaction("outer"):
            manager.conn.execute("DELETE FROM fact_table WHERE id = 1")
            # Bloc imbriqué : aucun BEGIN supplémentaire, aucun commit prématuré
            with manager._transaction("inner"):
                manager.conn.execute("DELETE FROM fact_table WHERE id = 2")
            assert manager._in_transaction is True
            raise RuntimeError("boom")

    # Les deux suppressions sont annulées ensemble
    count = manager.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count == initial_count


# ===========================================================================
# Tests de _cluster_by_order_clause() (tri des lots d'écriture)
# ===========================================================================


# Test que la clause reprend cluster_by dans son ordre déclaré
def test_cluster_by_order_clause_keeps_declared_order() -> None:
    """Test that the clause follows the cluster_by order, not the batch order."""
    clause = BaseSchemaManager._cluster_by_order_clause(
        ["value", "id", "category"], ["category", "id"]
    )
    assert clause == 'ORDER BY "category", "id"'


# Test que la clause ne conserve que les colonnes présentes dans le lot
def test_cluster_by_order_clause_filters_to_batch_columns() -> None:
    """Test that sort columns absent from the batch are dropped from the clause."""
    assert (
        BaseSchemaManager._cluster_by_order_clause(["id", "value"], ["category", "id"])
        == 'ORDER BY "id"'
    )
    # Lot sans aucune colonne de cluster_by : pas de tri
    assert BaseSchemaManager._cluster_by_order_clause(["value"], ["id"]) == ""


# Test que la clause est vide sans cluster_by
@pytest.mark.parametrize("cluster_by", [None, []])
def test_cluster_by_order_clause_empty_without_cluster_by(
    cluster_by: list[str] | None,
) -> None:
    """Test that the clause is empty when no sort key is declared.

    Args:
        cluster_by: Missing or empty sort key.
    """
    assert BaseSchemaManager._cluster_by_order_clause(["id"], cluster_by) == ""


# Test que la clause quote les identifiants
def test_cluster_by_order_clause_quotes_identifiers() -> None:
    """Test that a sort column with a double quote is safely quoted."""
    clause = BaseSchemaManager._cluster_by_order_clause(['we"ird'], ['we"ird'])
    assert clause == 'ORDER BY "we""ird"'


# ===========================================================================
# Tests de _get_null_only_columns() et de _touch_dataset_metadata()
# ===========================================================================


# Test que la détection des colonnes nulles repère une colonne vidée
def test_get_null_only_columns_detects_emptied_column(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that a column set to NULL everywhere is reported, in column order.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection.
    """
    built_ducklake_schema.execute(
        "UPDATE fact_table SET status = NULL, high_cardinality = NULL"
    )
    assert manager._get_null_only_columns() == ["status", "high_cardinality"]


# Test que la détection renvoie None sur une table vide
def test_get_null_only_columns_empty_table_returns_none(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that an empty fact table yields None (nothing can be decided).

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection.
    """
    built_ducklake_schema.execute("DELETE FROM fact_table")
    assert manager._get_null_only_columns() is None


# Test que l'horodatage met à jour updated_at
def test_touch_dataset_metadata_stamps_updated_at(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that updated_at moves forward when the dataset is touched.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection.
    """
    built_ducklake_schema.execute(
        "UPDATE dataset_metadata SET updated_at = TIMESTAMP '2000-01-01'"
    )
    manager._touch_dataset_metadata()
    row = built_ducklake_schema.execute(
        "SELECT updated_at FROM dataset_metadata"
    ).fetchone()
    assert row is not None and row[0].year > 2000


# Test que l'horodatage ignore un schéma sans dataset_metadata
def test_touch_dataset_metadata_without_table_is_noop(
    built_ducklake_schema: duckdb.DuckDBPyConnection,
) -> None:
    """Test that a schema without dataset_metadata is left untouched, silently.

    Args:
        built_ducklake_schema: DuckDB connection.
    """
    built_ducklake_schema.execute("DROP TABLE dataset_metadata")
    BaseSchemaManager(connection=built_ducklake_schema)._touch_dataset_metadata()


# ===========================================================================
# Tests de _early_report() et du cache des métadonnées dans _transaction()
# ===========================================================================


# Test que le rapport précoce est exposé comme dernier rapport
def test_early_report_sets_last_report(manager: BaseSchemaManager) -> None:
    """Test that an early report carries its warning and becomes last_report.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    report = manager._early_report("update", "run-1", "nothing to write")
    assert report.warnings == ["nothing to write"]
    assert report.run_id == "run-1"
    assert manager.last_report is report


# Test que l'ouverture d'une transaction relit les métadonnées
def test_transaction_invalidates_metadata_cache(
    manager: BaseSchemaManager, built_ducklake_schema: duckdb.DuckDBPyConnection
) -> None:
    """Test that a change made by another manager is seen by the next operation.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
        built_ducklake_schema: DuckDB connection.
    """
    # Cache chargé, puis metadata modifiée par un autre gestionnaire
    manager._load_current_metadata()
    other = BaseSchemaManager(connection=built_ducklake_schema)
    other.update_column_metadata("value", label="Changed elsewhere")

    with manager._transaction("probe"):
        labels = dict(
            zip(
                manager._load_current_metadata()["name"].to_list(),
                manager._load_current_metadata()["label"].to_list(),
                strict=True,
            )
        )
    assert labels["value"] == "Changed elsewhere"


# Test que l'annulation d'une transaction vide le cache des métadonnées
def test_transaction_rollback_invalidates_metadata_cache(
    manager: BaseSchemaManager,
) -> None:
    """Test that metadata cached inside a rolled-back transaction is discarded.

    Args:
        manager: BaseSchemaManager fixture with a built schema.
    """
    with pytest.raises(RuntimeError):
        with manager._transaction("probe"):
            manager.update_column_metadata("value", label="Rolled back")
            manager._load_current_metadata()
            raise RuntimeError("boom")
    labels = dict(
        zip(
            manager._load_current_metadata()["name"].to_list(),
            manager._load_current_metadata()["label"].to_list(),
            strict=True,
        )
    )
    assert labels["value"] != "Rolled back"
