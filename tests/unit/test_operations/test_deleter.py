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
from dt_ducklake_manager.maintenance import IssueSeverity, ValidationLevel
from dt_ducklake_manager.operations import DatabaseDeleter
from dt_ducklake_manager.reporting import OperationReport
from dt_ducklake_manager.schema import DuckLakeTablesBuilder
from tests.utils.ducklake import requires_ducklake

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


# Test de la désactivation de l'audit post-écriture
def test_deleter_initialization_without_audit(built_ducklake_schema: Any) -> None:
    """Test that the post-write audit defaults to BASIC and can be disabled.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    assert DatabaseDeleter(built_ducklake_schema).audit_level == ValidationLevel.BASIC
    deleter = DatabaseDeleter(connection=built_ducklake_schema, audit_level=None)
    assert deleter.audit_level is None


# Test que catalog_alias est propagé à l'auditeur et à la maintenance
def test_deleter_propagates_catalog_alias(built_ducklake_schema: Any) -> None:
    """Test that ``catalog_alias`` reaches the auditor and the maintenance helper.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    deleter = DatabaseDeleter(connection=built_ducklake_schema, catalog_alias="my_lake")
    assert deleter.catalog_alias == "my_lake"
    assert deleter.auditor is not None
    assert deleter.auditor.catalog_alias == "my_lake"
    assert deleter.maintenance.catalog_alias == "my_lake"


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

    with pytest.raises(ValueError, match="hierarchy parent"):
        deleter.delete_columns(["status"], use_transaction=False)

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


# Test qu'une colonne de code visée par des colonnes de libellés est refusée sans
# cascade
def test_delete_columns_code_refused_without_cascade(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that deleting a code column targeted by a label column is refused.

    'status' is declared as the label column of 'category' (§2.6); deleting
    'category' without cascade=True must be refused for the whole batch and leave
    both columns intact.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    deleter.update_column_metadata("status", label_for="category")

    with pytest.raises(ValueError, match="label_for target"):
        deleter.delete_columns(["category"], use_transaction=False)

    columns_after = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "category" in columns_after


# Test que cascade=True autorise la suppression et détache les colonnes de libellés
def test_delete_columns_code_with_cascade_detaches_label_columns(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that cascade=True allows deleting a code column and clears the label
    columns' label_for.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    deleter.update_column_metadata("status", label_for="category")

    result = deleter.delete_columns(["category"], use_transaction=False, cascade=True)

    assert result.columns_dropped == ["category"]
    columns_after = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "category" not in columns_after

    # La colonne de libellés est toujours là, mais détachée du code supprimé
    label_for_status = built_ducklake_schema.execute(
        "SELECT label_for FROM metadata WHERE name = 'status'"
    ).fetchone()[0]
    assert label_for_status is None


# Test que supprimer une colonne de libellés elle-même ne demande rien de particulier
def test_delete_columns_label_column_itself(
    deleter: DatabaseDeleter, built_ducklake_schema: Any
) -> None:
    """Test that deleting a label column itself needs no special handling.

    Args:
        deleter: DatabaseDeleter fixture.
        built_ducklake_schema: DuckDB connection.
    """
    deleter.update_column_metadata("status", label_for="category")

    result = deleter.delete_columns(["status"], use_transaction=False)

    assert result.columns_dropped == ["status"]
    columns_after = [
        row[0]
        for row in built_ducklake_schema.execute("DESCRIBE fact_table").fetchall()
    ]
    assert "status" not in columns_after
    # 'category', la colonne de code, est intacte
    assert "category" in columns_after


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
@requires_ducklake
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
@requires_ducklake
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


# Rapport d'audit simulant un problème critique
class _CriticalReport:
    """Audit report double holding one critical issue."""

    class _Issue:
        description = "metadata table is missing"

    def get_issues_by_severity(self, severity: Any) -> list[Any]:
        """Return the critical issue for CRITICAL, nothing otherwise."""
        return [self._Issue()] if severity == IssueSeverity.CRITICAL else []


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


# Test qu'un échec avant le commit restaure les lignes supprimées
def test_delete_rows_rolls_back_on_exception(deleter: DatabaseDeleter) -> None:
    """Test that a failure after the DELETE, before COMMIT, restores every row.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    before = _snapshot_state(deleter.conn)

    # Exception simulée dans l'audit post-suppression, après le DELETE
    def _boom(level: Any = None) -> Any:
        raise RuntimeError("échec après suppression")

    assert deleter.auditor is not None
    setattr(deleter.auditor, "validate_database", _boom)

    with pytest.raises(RuntimeError, match="échec après suppression"):
        deleter.delete_rows(filters=[("id", "=", 1)])
    assert deleter.last_report is not None
    assert deleter.last_report.warnings

    # Les lignes supprimées sont revenues
    assert _snapshot_state(deleter.conn) == before


# Test qu'un échec du nettoyage post-commit conserve la suppression des lignes
def test_delete_rows_keeps_deletion_when_cleanup_fails(
    deleter: DatabaseDeleter,
) -> None:
    """Test that a failing null-only cleanup does not restore the deleted rows.

    The cleanup runs after COMMIT (DuckDB cannot commit a DELETE and a DROP
    COLUMN on the same table together): its failure is only reported.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    initial_count = deleter.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[
        0
    ]

    def _boom(use_transaction: bool = True) -> list[str]:
        raise RuntimeError("échec du nettoyage")

    deleter._cleanup_null_only_columns = _boom  # type: ignore[method-assign]

    report = deleter.delete_rows(filters=[("id", "=", 1)], compact_after_update=False)

    assert report.rows_deleted == 1
    assert any("échec du nettoyage" in w for w in report.warnings)
    assert deleter.last_report is report
    count_after = deleter.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count_after == initial_count - 1


# Test que des problèmes critiques post-suppression annulent la suppression
def test_delete_rows_rolls_back_on_critical_validation_issues(
    deleter: DatabaseDeleter,
) -> None:
    """Test that critical post-deletion issues roll the deletion back.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    before = _snapshot_state(deleter.conn)

    assert deleter.auditor is not None
    setattr(deleter.auditor, "validate_database", lambda level=None: _CriticalReport())

    with pytest.raises(RuntimeError, match="critical"):
        deleter.delete_rows(filters=[("id", "=", 1)])
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

    def _boom(use_transaction: bool = True) -> list[str]:
        raise RuntimeError("échec après suppression")

    deleter._cleanup_null_only_columns = _boom  # type: ignore[method-assign]

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


# Test qu'une colonne en échec annule la suppression de toutes les colonnes
def test_delete_columns_failing_column_rolls_back_all(deleter: DatabaseDeleter) -> None:
    """Test that a column failing to drop restores the columns already dropped.

    Either every requested column is dropped or none is: a failure on the second
    column restores the first one, its metadata row included.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    before = _snapshot_state(deleter.conn)
    # Échec simulé pour la seule colonne 'status', traitée en second
    original_drop = deleter._drop_fact_table_column

    def _selective_drop(column: str) -> None:
        if column == "status":
            raise duckdb.IOException("simulated I/O error")
        original_drop(column)

    deleter._drop_fact_table_column = _selective_drop  # type: ignore[method-assign]

    with pytest.raises(duckdb.IOException):
        deleter.delete_columns(["high_cardinality", "status"])

    assert _snapshot_state(deleter.conn) == before


# Test que des problèmes critiques annulent toutes les suppressions de colonnes
def test_delete_columns_rolls_back_on_critical_validation_issues(
    deleter: DatabaseDeleter,
) -> None:
    """Test that critical post-deletion issues restore every dropped column.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    before = _snapshot_state(deleter.conn)

    assert deleter.auditor is not None
    setattr(deleter.auditor, "validate_database", lambda level=None: _CriticalReport())

    with pytest.raises(RuntimeError, match="critical"):
        deleter.delete_columns(["status", "high_cardinality"])

    # Aucune suppression n'est retenue, colonnes et métadonnées sont intactes
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

    setattr(deleter.auditor, "validate_database", _boom)

    with pytest.raises(RuntimeError, match="auditeur indisponible"):
        deleter.delete_columns(["status"])

    assert _snapshot_state(deleter.conn) == before


# ---------------------------------------------------------------------------
# Validation des demandes de suppression : ValueError, rien d'écrit
# ---------------------------------------------------------------------------


# Test que des filtres invalides sont refusés avant toute écriture
@pytest.mark.parametrize(
    "filters",
    [None, "", "   ", [], {"id": 1}, 42, [("id", "=", 1), [("id", "=", 2)]]],
)
def test_delete_rows_invalid_filters_raise(
    deleter: DatabaseDeleter, filters: Any
) -> None:
    """Test that missing, empty or malformed filters raise ValueError.

    A dict used to be accepted by the signature and silently delete 0 rows.

    Args:
        deleter: DatabaseDeleter fixture.
        filters: Invalid filters.
    """
    before = _snapshot_state(deleter.conn)
    with pytest.raises(ValueError):
        deleter.delete_rows(filters)
    assert _snapshot_state(deleter.conn) == before


# Test qu'un filtre sans correspondance ne supprime rien et n'horodate pas
def test_delete_rows_without_match_is_noop(deleter: DatabaseDeleter) -> None:
    """Test that a filter matching no row deletes nothing and keeps updated_at.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    stamp = deleter.conn.execute("SELECT updated_at FROM dataset_metadata").fetchone()
    report = deleter.delete_rows([("id", "=", 999)])
    assert report.rows_deleted == 0
    assert report.columns_dropped == []
    after = deleter.conn.execute("SELECT updated_at FROM dataset_metadata").fetchone()
    assert after == stamp


# Test qu'une colonne inconnue du filtre remonte l'erreur DuckDB après annulation
def test_delete_rows_unknown_filter_column_raises(deleter: DatabaseDeleter) -> None:
    """Test that a SQL error in the filter propagates after rollback.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    before = _snapshot_state(deleter.conn)
    with pytest.raises(duckdb.Error):
        deleter.delete_rows([("unknown_column", "=", 1)])
    assert _snapshot_state(deleter.conn) == before


# Test que la suppression d'une clé primaire est refusée
def test_delete_columns_primary_key_raises(deleter: DatabaseDeleter) -> None:
    """Test that deleting a primary key column raises ValueError.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    with pytest.raises(ValueError, match="primary key"):
        deleter.delete_columns(["id", "status"])
    assert "status" in deleter._get_fact_table_columns()


# Test que la suppression d'une colonne inconnue ou d'une liste vide est refusée
@pytest.mark.parametrize("columns", [[], ["does_not_exist"], ["status", "nope"]])
def test_delete_columns_invalid_request_raises(
    deleter: DatabaseDeleter, columns: list[str]
) -> None:
    """Test that an empty list or an unknown column raises ValueError.

    Args:
        deleter: DatabaseDeleter fixture.
        columns: Invalid column list.
    """
    with pytest.raises(ValueError):
        deleter.delete_columns(columns)
    assert "status" in deleter._get_fact_table_columns()


# Test que la suppression d'une colonne catégorielle est signalée
def test_delete_columns_categorical_warning(deleter: DatabaseDeleter) -> None:
    """Test that dropping a categorical column warns that it may back a menu.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    report = deleter.delete_columns(["status"])
    assert report.columns_dropped == ["status"]
    assert any("categorical" in w for w in report.warnings)


# Test que les méthodes publiques sans usage ont été retirées
@pytest.mark.parametrize(
    "name",
    ["get_deletion_impact", "get_deletion_status", "cleanup_database", "data_mgr"],
)
def test_deleter_has_no_removed_api(deleter: DatabaseDeleter, name: str) -> None:
    """Test that the unused public methods and attributes are gone.

    Args:
        deleter: DatabaseDeleter fixture.
        name: Removed attribute name.
    """
    assert not hasattr(deleter, name)
