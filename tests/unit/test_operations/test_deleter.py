# Importation des modules
# Modules de base
import os
import warnings

# Module de tests
from typing import Any

import duckdb
import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.connection import DuckLakeConnector
from dt_ducklake_manager.operations import DatabaseDeleter
from dt_ducklake_manager.reporting import OperationReport
from dt_ducklake_manager.schema import DuckLakeTablesBuilder


def _ducklake_available() -> bool:
    """Vérifie si l'extension DuckLake est disponible dans l'environnement de test."""
    try:
        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL ducklake; LOAD ducklake;")
        conn.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Tests de l'initialisation
# ---------------------------------------------------------------------------


# Test de l'initialisation correcte de DatabaseDeleter
def test_deleter_initialization(built_ducklake_schema: Any) -> None:
    """Test that DatabaseDeleter initializes without errors.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    deleter = DatabaseDeleter(connection=built_ducklake_schema)
    assert deleter is not None


# Test de l'initialisation avec enable_validation=False
def test_deleter_initialization_without_validation(built_ducklake_schema: Any) -> None:
    """Test that DatabaseDeleter can be initialized with validation disabled.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    deleter = DatabaseDeleter(connection=built_ducklake_schema, enable_validation=False)
    # Vérification que l'auditeur n'est pas initialisé
    assert deleter.enable_validation is False


# Test que catalog_alias est propagé aux sous-gestionnaires
def test_deleter_propagates_catalog_alias(built_ducklake_schema: Any) -> None:
    """Test that ``catalog_alias`` reaches every specialized sub-manager.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    deleter = DatabaseDeleter(connection=built_ducklake_schema, catalog_alias="my_lake")
    assert deleter.catalog_alias == "my_lake"
    assert deleter.data_mgr.catalog_alias == "my_lake"
    assert deleter.auditor is not None
    assert deleter.auditor.catalog_alias == "my_lake"


# Test que catalog_alias vaut 'db' par défaut
def test_deleter_default_catalog_alias(built_ducklake_schema: Any) -> None:
    """Test that ``catalog_alias`` defaults to 'db'.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    deleter = DatabaseDeleter(connection=built_ducklake_schema)
    assert deleter.catalog_alias == "db"


# ---------------------------------------------------------------------------
# Tests de validate_operation()
# ---------------------------------------------------------------------------


# Test que validate_operation retourne un booléen pour une suppression valide
def test_validate_operation_delete_returns_bool(deleter: DatabaseDeleter) -> None:
    """Test that validate_operation returns a boolean for a delete operation.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    result = deleter.validate_operation("delete", filters=[("id", "=", 1)])
    assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Tests de delete_rows()
# ---------------------------------------------------------------------------


# Test de la suppression de lignes avec un filtre simple
def test_delete_rows_with_filter(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that delete_rows removes rows matching the given filter.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Vérification que la ligne id=1 existe avant suppression
    before = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 1"
    ).fetchone()[0]
    assert before >= 1

    # Suppression de la ligne avec id=1
    report = deleter.delete_rows(
        filters=[("id", "=", 1)],
        use_transaction=False,
    )

    # Vérification que le nombre de lignes supprimées est cohérent
    assert isinstance(report.rows_deleted, int)
    assert report.rows_deleted >= 1

    # Vérification que la ligne est bien absente
    after = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 1"
    ).fetchone()[0]
    assert after == 0


# Test de la suppression de lignes avec un filtre OR (liste de listes de tuples)
def test_delete_rows_with_or_filter(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that delete_rows handles OR filters (list of lists of tuples).

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Suppression des lignes avec id=2 OU id=3
    before = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id IN (2, 3)"
    ).fetchone()[0]
    assert before >= 1

    report = deleter.delete_rows(
        filters=[[("id", "=", 2)], [("id", "=", 3)]],
        use_transaction=False,
    )
    assert report.rows_deleted >= 1

    # Vérification que les lignes sont bien absentes
    after = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id IN (2, 3)"
    ).fetchone()[0]
    assert after == 0


# Test de la suppression de toutes les lignes avec filters=None
def test_delete_rows_all(deleter: DatabaseDeleter, built_ducklake_schema: Any) -> None:
    """Test that delete_rows with filters=None removes all rows.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    before = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert before > 0

    # Remarque : DatabaseDeleter n'accepte pas filters=None (validation obligatoire des
    # filtres).
    # Suppression de toutes les lignes via un filtre SQL universel.
    report = deleter.delete_rows(filters="1=1", use_transaction=False)
    assert report.rows_deleted == before

    after = built_ducklake_schema.execute("SELECT COUNT(*) FROM fact_table").fetchone()[
        0
    ]
    assert after == 0


# ---------------------------------------------------------------------------
# Tests de delete_columns()
# ---------------------------------------------------------------------------


# Test de la suppression d'une colonne non-clé primaire
def test_delete_columns_single_column(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that delete_columns removes a non-primary-key column from the fact table.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # Vérification que la colonne 'value' existe avant suppression
    columns_before = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "value" in columns_before

    result = deleter.delete_columns(["value"], use_transaction=False)

    # Vérification que le résultat est un OperationReport listant la colonne
    assert isinstance(result, OperationReport)
    assert result.columns_dropped == ["value"]
    # Vérification que la colonne est bien supprimée
    columns_after = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "value" not in columns_after


# ---------------------------------------------------------------------------
# Tests de changement de statut catégoriel lors d'une suppression
# ---------------------------------------------------------------------------


# Test que la suppression de lignes ne recalcule pas le statut catégoriel
def test_delete_rows_keeps_categorical_status(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that deleting rows never re-evaluates is_categorical.

    'high_cardinality' holds 5 distinct values (threshold 4) and is not
    categorical. Deleting the only row carrying 'val_104' leaves 4 values, but the
    status is inferred once, at creation, and stays False.

    Args:
        deleter: DatabaseDeleter fixture with auto_cleanup=True.
        built_ducklake_schema: DuckDB connection with the built schema.
    """
    report = deleter.delete_rows(
        filters=[("id", "=", 5)],
        use_transaction=False,
    )
    assert report.rows_deleted == 1
    assert report.metadata_changes == []

    is_cat_after = built_ducklake_schema.execute(
        "SELECT is_categorical FROM metadata WHERE name = 'high_cardinality'"
    ).fetchone()[0]
    assert is_cat_after is False

    # Vérification : les libellés d'origine sont toujours stockés tels quels
    stored_labels = {
        row[0]
        for row in built_ducklake_schema.execute(
            "SELECT DISTINCT high_cardinality FROM fact_table"
        ).fetchall()
    }
    assert stored_labels == {"val_100", "val_101", "val_102", "val_103"}


# ---------------------------------------------------------------------------
# Tests de delete_columns() sur une colonne parente d'une hiérarchie (§2.5)
# ---------------------------------------------------------------------------


# Test que la suppression d'une colonne parente est refusée sans cascade
def test_delete_columns_parent_refused_without_cascade(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that deleting a hierarchy parent column is refused by default.

    'status' is declared as the parent of 'category'; deleting 'status' without
    cascade=True must be refused for the whole batch and leave both columns intact.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # 'category' et 'status' sont déjà catégorielles (seuil=4) : aucun forçage
    deleter.update_column_metadata("category", parent_name="status")

    result = deleter.delete_columns(["status"], use_transaction=False)

    assert result.columns_dropped == []
    assert any("status" in w for w in result.warnings)
    columns_after = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "status" in columns_after


# Test que cascade=True autorise la suppression et détache les enfants
def test_delete_columns_parent_with_cascade_detaches_children(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that cascade=True allows deleting a hierarchy parent and clears
    the children's parent_name.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    # 'category' et 'status' sont déjà catégorielles (seuil=4) : aucun forçage
    deleter.update_column_metadata("category", parent_name="status")

    result = deleter.delete_columns(["status"], use_transaction=False, cascade=True)

    assert result.columns_dropped == ["status"]
    columns_after = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "status" not in columns_after

    # La colonne enfant est toujours là, mais détachée de la hiérarchie
    parent_of_category = built_ducklake_schema.execute(
        "SELECT parent_name FROM metadata WHERE name = 'category'"
    ).fetchone()[0]
    assert parent_of_category is None


# ---------------------------------------------------------------------------
# Tests de delete_columns() et cluster_by (§4.3, §5.3)
# ---------------------------------------------------------------------------


# Test que la suppression d'une colonne de cluster_by la retire de la liste
def test_delete_columns_removes_column_from_cluster_by(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that dropping a cluster_by column removes it from cluster_by.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection with the built schema
            (cluster_by defaults to ['id']).
    """
    deleter.update_cluster_by(["id", "category"])

    result = deleter.delete_columns(["category"], use_transaction=False)

    assert result.columns_dropped == ["category"]
    assert deleter._get_cluster_by_columns() == ["id"]


# Test que vider entièrement cluster_by le remet à NULL (pas une liste vide)
def test_delete_columns_cluster_by_falls_back_to_null_when_emptied(
    deleter: DatabaseDeleter,
) -> None:
    """Test that dropping the only cluster_by column resets it to NULL.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    deleter.update_cluster_by(["category"])

    result = deleter.delete_columns(["category"], use_transaction=False)

    assert result.columns_dropped == ["category"]
    assert deleter._get_cluster_by_columns() is None


# Test que la suppression d'une colonne hors cluster_by ne modifie pas cluster_by
def test_delete_columns_leaves_cluster_by_untouched_when_unrelated(
    deleter: DatabaseDeleter,
) -> None:
    """Test that deleting a column not in cluster_by leaves it unchanged.

    Args:
        deleter: DatabaseDeleter fixture (cluster_by defaults to ['id']).
    """
    result = deleter.delete_columns(["category"], use_transaction=False)

    assert result.columns_dropped == ["category"]
    assert deleter._get_cluster_by_columns() == ["id"]


# ---------------------------------------------------------------------------
# Test de bout en bout de la compaction DuckLake après delete (§5.4-5.5)
# ---------------------------------------------------------------------------


# Test que delete_rows réussit avec compaction réelle sur un catalogue sur disque
@pytest.mark.skipif(
    not _ducklake_available(),
    reason="Extension ducklake non disponible dans cet environnement",
)
def test_delete_rows_compacts_on_real_ducklake_catalog(tmp_path: Any) -> None:
    """Test that delete_rows succeeds end-to-end against a real DuckLake catalog.

    Mirrors ``test_update_database_compacts_on_real_ducklake_catalog``: the
    in-memory fixture used elsewhere in this file can't exercise
    ``DuckLakeMaintenance.compact`` for real.

    Args:
        tmp_path: pytest temporary directory.
    """
    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(catalog, data_dir, data_inlining_row_limit=0).connect()

    df = pl.DataFrame(
        {
            "id": list(range(1, 6)),
            "category": ["A", "B", "A", "C", "B"],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=["id"], connection=conn
        ).build_schema()

    deleter = DatabaseDeleter(connection=conn)
    report = deleter.delete_rows(filters=[("id", "=", 1)], use_transaction=False)

    assert report.rows_deleted == 1
    row_count = conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert row_count == 4
    conn.close()


# Test que delete_columns ne change pas file_count (§4.3 : opération de métadonnées)
@pytest.mark.skipif(
    not _ducklake_available(),
    reason="Extension ducklake non disponible dans cet environnement",
)
def test_delete_columns_does_not_change_file_count(tmp_path: Any) -> None:
    """Test that ALTER TABLE ... DROP COLUMN rewrites no data file (measured, §4.3).

    Against an in-memory connection there is no attached DuckLake catalog for
    ``ducklake_table_info`` to query, so this needs a real one on disk (mirrors
    ``test_delete_rows_compacts_on_real_ducklake_catalog``).

    Args:
        tmp_path: pytest temporary directory.
    """
    catalog = str(tmp_path / "test.ducklake")
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir)
    conn = DuckLakeConnector(catalog, data_dir, data_inlining_row_limit=0).connect()

    df = pl.DataFrame(
        {
            "id": list(range(1, 6)),
            "category": ["A", "B", "A", "C", "B"],
            "value": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        DuckLakeTablesBuilder(
            df, categorical_threshold=4, primary_keys=["id"], connection=conn
        ).build_schema()

    file_count_before = conn.execute(
        "SELECT file_count FROM ducklake_table_info('db') WHERE table_name ="
        " 'fact_table'"
    ).fetchone()[0]
    assert file_count_before > 0

    deleter = DatabaseDeleter(connection=conn)
    result = deleter.delete_columns(["value"], use_transaction=False)
    assert result.columns_dropped == ["value"]

    file_count_after = conn.execute(
        "SELECT file_count FROM ducklake_table_info('db') WHERE table_name ="
        " 'fact_table'"
    ).fetchone()[0]
    assert file_count_after == file_count_before
    conn.close()


# ---------------------------------------------------------------------------
# Tests des chemins transactionnels (§7) : un BEGIN/COMMIT par opération
# ---------------------------------------------------------------------------


# Fonction auxiliaire de capture de l'état complet de la base
def _snapshot_state(conn: Any) -> dict[str, Any]:
    """Capture the full state of the schema, for before/after comparison.

    Args:
        conn: DuckDB connection holding a built schema.

    Returns:
        dict: row count, sorted fact table content and metadata content.
    """
    return {
        "count": conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0],
        "facts": conn.execute("SELECT * FROM fact_table ORDER BY id").pl().to_dicts(),
        "metadata": conn.execute("SELECT * FROM metadata ORDER BY name")
        .pl()
        .to_dicts(),
    }


# Test qu'un échec après suppression restaure les lignes supprimées
def test_delete_rows_rolls_back_on_exception(deleter: DatabaseDeleter) -> None:
    """Test that a failure after the DELETE restores every deleted row.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    before = _snapshot_state(deleter.conn)

    # Exception simulée dans le nettoyage, après la suppression des lignes
    def _boom(report: Any = None) -> dict[str, Any]:
        raise RuntimeError("échec après suppression")

    deleter._cleanup_orphaned_data_comprehensive = _boom  # type: ignore[method-assign]

    report = deleter.delete_rows(filters=[("id", "=", 1)])
    assert report.warnings

    # Les lignes supprimées sont revenues
    assert _snapshot_state(deleter.conn) == before


# Test que des problèmes critiques post-suppression annulent la suppression
def test_delete_rows_rolls_back_on_critical_validation_issues(
    deleter: DatabaseDeleter,
) -> None:
    """Test that critical post-deletion issues roll the deletion back.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    before = _snapshot_state(deleter.conn)

    # Rapport de validation simulant un problème critique
    class _CriticalReport:
        def get_critical_issues_count(self) -> int:
            return 1

        def get_issues_by_severity(self, severity: Any) -> list[Any]:
            return []

    assert deleter.auditor is not None
    deleter.auditor.validate_database = (  # type: ignore[method-assign]
        lambda level=None: _CriticalReport()
    )

    report = deleter.delete_rows(filters=[("id", "=", 1)])
    assert report.warnings
    assert _snapshot_state(deleter.conn) == before


# Test que sans transaction, la suppression subsiste malgré l'échec du nettoyage
def test_delete_rows_without_transaction_keeps_partial_state(
    deleter: DatabaseDeleter,
) -> None:
    """Test that use_transaction=False leaves the deletion in place on failure.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    initial_count = deleter.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[
        0
    ]

    def _boom(report: Any = None) -> dict[str, Any]:
        raise RuntimeError("échec après suppression")

    deleter._cleanup_orphaned_data_comprehensive = _boom  # type: ignore[method-assign]

    report = deleter.delete_rows(filters=[("id", "=", 1)], use_transaction=False)
    assert report.warnings

    # La ligne supprimée ne revient pas : l'état est partiel
    count_after = deleter.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count_after == initial_count - 1


# Test qu'une suppression réussie survit au commit
def test_delete_rows_commits_on_success(deleter: DatabaseDeleter) -> None:
    """Test that a successful transactional deletion is committed.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    initial_count = deleter.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[
        0
    ]

    assert deleter.delete_rows(filters=[("id", "=", 1)]).rows_deleted == 1

    count_after = deleter.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count_after == initial_count - 1


# Test qu'une colonne en échec n'empêche pas la suppression des autres
def test_delete_columns_isolates_failing_column(deleter: DatabaseDeleter) -> None:
    """Test that a column failing to drop does not abort the other deletions.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    # Échec simulé pour la seule colonne 'status'
    original_drop = deleter._drop_fact_table_column

    def _selective_drop(column: str) -> bool:
        if column == "status":
            return False
        return original_drop(column)

    deleter._drop_fact_table_column = _selective_drop  # type: ignore[method-assign]

    result = deleter.delete_columns(["status", "high_cardinality"])

    # La colonne en échec est signalée, l'autre est bien supprimée
    assert result.columns_dropped == ["high_cardinality"]
    assert any("status" in w for w in result.warnings)
    remaining = deleter._get_fact_table_columns()
    assert "status" in remaining
    assert "high_cardinality" not in remaining


# Test que des problèmes critiques annulent toutes les suppressions de colonnes
def test_delete_columns_rolls_back_on_critical_validation_issues(
    deleter: DatabaseDeleter,
) -> None:
    """Test that critical post-deletion issues restore every dropped column.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    before = _snapshot_state(deleter.conn)

    # Rapport de validation simulant un problème critique
    class _CriticalReport:
        def get_critical_issues_count(self) -> int:
            return 3

        def get_issues_by_severity(self, severity: Any) -> list[Any]:
            return []

    assert deleter.auditor is not None
    deleter.auditor.validate_database = (  # type: ignore[method-assign]
        lambda level=None: _CriticalReport()
    )

    result = deleter.delete_columns(["status", "high_cardinality"])

    # Aucune suppression n'est retenue, colonnes et métadonnées sont intactes
    assert result.columns_dropped == []
    assert result.warnings
    assert _snapshot_state(deleter.conn) == before


# Test qu'une exception lors de la suppression de colonnes restaure tout
def test_delete_columns_rolls_back_on_exception(deleter: DatabaseDeleter) -> None:
    """Test that an exception mid-loop restores columns and metadata rows.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    before = _snapshot_state(deleter.conn)

    # Exception non rattrapée par la boucle : levée par la validation finale
    assert deleter.auditor is not None

    def _boom(level: Any = None) -> Any:
        raise RuntimeError("auditeur indisponible")

    deleter.auditor.validate_database = _boom  # type: ignore[method-assign]

    result = deleter.delete_columns(["status"])

    assert result.columns_dropped == []
    assert result.warnings
    assert _snapshot_state(deleter.conn) == before
