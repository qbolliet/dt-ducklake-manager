# Importation des modules
# Modules de base
from typing import Any

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.operations import DatabaseDeleter, DatabaseUpdater
from dt_ducklake_manager.operations._base import BaseSchemaManager

# ---------------------------------------------------------------------------
# Fonctions auxiliaires
# ---------------------------------------------------------------------------


# Fonction auxiliaire de lecture des colonnes de la table des faits
def _fact_columns(conn: Any) -> list[str]:
    """Return the fact table column names.

    Args:
        conn: DuckDB connection holding a built schema.

    Returns:
        list[str]: Column names, in table order.
    """
    return [row[0] for row in conn.execute("DESCRIBE fact_table").fetchall()]


# Fonction auxiliaire de lecture des noms présents dans metadata
def _metadata_names(conn: Any) -> set[str]:
    """Return the column names recorded in the metadata table.

    Args:
        conn: DuckDB connection holding a built schema.

    Returns:
        set[str]: Names of the metadata rows.
    """
    return {row[0] for row in conn.execute("SELECT name FROM metadata").fetchall()}


# Fonction auxiliaire de mise à NULL d'une colonne
def _nullify(conn: Any, *columns: str) -> None:
    """Set every value of the given fact table columns to NULL.

    Args:
        conn: DuckDB connection holding a built schema.
        *columns: Names of the columns to empty.
    """
    for column in columns:
        conn.execute(f'UPDATE fact_table SET "{column}" = NULL')


# Gestionnaire paramétré : la méthode commune doit se comporter à l'identique
@pytest.fixture(params=["deleter", "updater"])
def manager(request: Any, deleter: DatabaseDeleter, updater: DatabaseUpdater) -> Any:
    """Provide each manager sharing ``_cleanup_null_only_columns``.

    Args:
        request: pytest request carrying the parameter.
        deleter: DatabaseDeleter fixture.
        updater: DatabaseUpdater fixture.

    Returns:
        BaseSchemaManager: the deleter or the updater, both on the same schema.
    """
    return deleter if request.param == "deleter" else updater


# ---------------------------------------------------------------------------
# Tests de _cleanup_null_only_columns()
# ---------------------------------------------------------------------------


# Test du cas nominal : colonne nulle supprimée avec sa ligne de méta-données
def test_cleanup_drops_null_only_column_and_metadata(
    manager: BaseSchemaManager,
) -> None:
    """Test that a null-only column is dropped along with its metadata row.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    _nullify(manager.conn, "value")

    dropped = manager._cleanup_null_only_columns()

    assert dropped == ["value"]
    assert "value" not in _fact_columns(manager.conn)
    assert "value" not in _metadata_names(manager.conn)
    assert manager.last_report is not None
    assert manager.last_report.columns_dropped == ["value"]


# Test qu'aucune colonne nulle ne produit une liste vide
def test_cleanup_without_null_only_column_returns_empty(
    manager: BaseSchemaManager,
) -> None:
    """Test that nothing is dropped when no column is null-only.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    before = _fact_columns(manager.conn)

    assert manager._cleanup_null_only_columns() == []
    assert _fact_columns(manager.conn) == before


# Test qu'une table des faits vide n'est jamais nettoyée
def test_cleanup_skips_empty_fact_table(manager: BaseSchemaManager) -> None:
    """Test that every column of an empty fact table is kept.

    All columns of an empty table are trivially null-only; dropping them would
    wipe the schema out before the next load.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    manager.conn.execute("DELETE FROM fact_table")
    before = _fact_columns(manager.conn)

    assert manager._cleanup_null_only_columns() == []
    assert _fact_columns(manager.conn) == before
    assert _metadata_names(manager.conn) >= set(before)


# Test qu'une clé primaire entièrement nulle n'est jamais supprimée
def test_cleanup_never_drops_primary_key(manager: BaseSchemaManager) -> None:
    """Test that a null-only primary key column is kept.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    _nullify(manager.conn, "id", "value")

    dropped = manager._cleanup_null_only_columns()

    assert dropped == ["value"]
    assert "id" in _fact_columns(manager.conn)


# Test du retrait de la colonne supprimée de cluster_by
def test_cleanup_removes_column_from_cluster_by(manager: BaseSchemaManager) -> None:
    """Test that a dropped column no longer appears in ``cluster_by``.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    manager.update_cluster_by(["value", "id"])
    _nullify(manager.conn, "value")

    manager._cleanup_null_only_columns()

    assert manager._get_cluster_by_columns() == ["id"]


# Test que cluster_by repasse à NULL si la seule colonne de tri est supprimée
def test_cleanup_resets_cluster_by_when_emptied(manager: BaseSchemaManager) -> None:
    """Test that ``cluster_by`` becomes NULL when its only column is dropped.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    manager.update_cluster_by(["value"])
    _nullify(manager.conn, "value")

    manager._cleanup_null_only_columns()

    assert manager._get_cluster_by_columns() is None


# Test qu'une colonne parente dont les enfants ont des valeurs est conservée
def test_cleanup_keeps_parent_with_non_null_children(
    manager: BaseSchemaManager,
) -> None:
    """Test that a null-only hierarchy parent is kept while its children hold data.

    The other null-only columns are still dropped, and the kept parent is reported
    as a warning instead of aborting the whole cleanup.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    manager.update_column_metadata("category", parent_name="status")
    _nullify(manager.conn, "status", "value")

    dropped = manager._cleanup_null_only_columns()

    assert dropped == ["value"]
    assert "status" in _fact_columns(manager.conn)
    assert manager._get_hierarchy_children("status") == ["category"]
    assert manager.last_report is not None
    assert any("status" in w for w in manager.last_report.warnings)


# Test qu'un parent et son enfant tous deux nuls sont supprimés, enfant d'abord
def test_cleanup_drops_null_parent_after_null_children(
    manager: BaseSchemaManager,
) -> None:
    """Test that a parent is dropped once its null-only children are dropped.

    'category' (parent) precedes 'status' (child) in the table: the parent is
    skipped on the first pass and dropped on the second, whatever the column order.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    manager.update_column_metadata("status", parent_name="category")
    _nullify(manager.conn, "category", "status")

    dropped = manager._cleanup_null_only_columns()

    assert dropped == ["status", "category"]
    assert not {"category", "status"} & set(_fact_columns(manager.conn))
    assert manager.last_report is not None
    assert manager.last_report.warnings == []


# Test qu'un échec en cours de nettoyage annule toutes les suppressions
def test_cleanup_rolls_back_on_failure(
    manager: BaseSchemaManager, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that a failure mid-cleanup restores the columns and metadata rows.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
        monkeypatch: pytest fixture used to inject the failure.
    """
    _nullify(manager.conn, "value", "high_cardinality")
    columns_before = _fact_columns(manager.conn)
    metadata_before = _metadata_names(manager.conn)

    # Échec injecté après la suppression physique de la première colonne
    def _boom(column: str) -> None:
        raise RuntimeError("échec cluster_by")

    monkeypatch.setattr(manager, "_remove_from_cluster_by", _boom)

    with pytest.raises(RuntimeError, match="échec cluster_by"):
        manager._cleanup_null_only_columns()

    assert _fact_columns(manager.conn) == columns_before
    assert _metadata_names(manager.conn) == metadata_before
    assert manager._in_transaction is False


# ---------------------------------------------------------------------------
# Tests des briques _drop_fact_table_column / _drop_column_with_references
# ---------------------------------------------------------------------------


# Test qu'une colonne inexistante renvoie False sans lever
def test_drop_fact_table_column_unknown_returns_false(
    manager: BaseSchemaManager,
) -> None:
    """Test that dropping an unknown column returns False.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    assert manager._drop_fact_table_column("does_not_exist") is False


# Test qu'un DROP en échec ne touche ni metadata ni cluster_by
def test_drop_column_with_references_failure_leaves_references(
    manager: BaseSchemaManager,
) -> None:
    """Test that a failed ``DROP COLUMN`` leaves metadata and cluster_by untouched.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    metadata_before = _metadata_names(manager.conn)
    cluster_before = manager._get_cluster_by_columns()

    assert manager._drop_column_with_references("does_not_exist") is False
    assert _metadata_names(manager.conn) == metadata_before
    assert manager._get_cluster_by_columns() == cluster_before


# Test du détachement des enfants avec cascade
def test_drop_column_with_references_cascade_detaches_children(
    manager: BaseSchemaManager,
) -> None:
    """Test that ``cascade=True`` clears the children's ``parent_name``.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    manager.update_column_metadata("category", parent_name="status")

    assert manager._drop_column_with_references("status", cascade=True) is True
    assert manager._get_hierarchy_children("status") == []
    assert "status" not in _metadata_names(manager.conn)


# Test que _get_hierarchy_children renvoie une liste vide hors hiérarchie
def test_get_hierarchy_children_without_children(manager: BaseSchemaManager) -> None:
    """Test that a column with no child returns an empty list.

    Args:
        manager: DatabaseDeleter or DatabaseUpdater fixture.
    """
    assert manager._get_hierarchy_children("value") == []


# ---------------------------------------------------------------------------
# Tests des points d'entrée publics
# ---------------------------------------------------------------------------


# Test que le nettoyage imbriqué de delete_rows contribue au rapport de delete_rows
@pytest.mark.parametrize("use_transaction", [True, False])
def test_delete_rows_cleanup_contributes_to_outer_report(
    deleter: DatabaseDeleter, use_transaction: bool
) -> None:
    """Test that the nested cleanup reports into the ``delete_rows`` report.

    In autocommit mode too, the nested call must not replace ``last_report`` with
    its own report.

    Args:
        deleter: DatabaseDeleter fixture with auto_cleanup=True.
        use_transaction: Whether ``delete_rows`` runs in a transaction.
    """
    # Seules les lignes id=1 et id=2 portent une valeur
    deleter.conn.execute("UPDATE fact_table SET value = NULL WHERE id NOT IN (1, 2)")

    report = deleter.delete_rows(
        filters=[("id", "<=", 2)],
        use_transaction=use_transaction,
        compact_after_update=False,
    )

    assert report.operation == "delete_rows"
    assert report.columns_dropped == ["value"]
    assert deleter.last_report is report
    assert "value" not in _fact_columns(deleter.conn)


# Test que la suppression de toutes les lignes ne supprime aucune colonne
def test_delete_all_rows_keeps_columns(deleter: DatabaseDeleter) -> None:
    """Test that emptying the fact table with cleanup keeps every column.

    Args:
        deleter: DatabaseDeleter fixture with auto_cleanup=True.
    """
    before = _fact_columns(deleter.conn)

    report = deleter.delete_rows(filters="1=1", compact_after_update=False)

    assert report.columns_dropped == []
    assert _fact_columns(deleter.conn) == before


# Test du format de retour de cleanup_database
def test_cleanup_database_returns_dropped_columns(deleter: DatabaseDeleter) -> None:
    """Test that ``cleanup_database`` wraps the dropped columns in a dict.

    Args:
        deleter: DatabaseDeleter fixture.
    """
    _nullify(deleter.conn, "value")

    assert deleter.cleanup_database() == {"null_columns": ["value"]}


# Test que cleanup_database convertit une exception en dictionnaire d'erreur
def test_cleanup_database_returns_error_on_failure(
    deleter: DatabaseDeleter, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test that ``cleanup_database`` returns ``{"error": ...}`` instead of raising.

    Args:
        deleter: DatabaseDeleter fixture.
        monkeypatch: pytest fixture used to inject the failure.
    """

    def _boom(use_transaction: bool = True) -> list[str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(deleter, "_cleanup_null_only_columns", _boom)

    assert deleter.cleanup_database() == {"error": "boom"}


# Test que optimize_database nettoie aussi cluster_by et parent_name côté updater
def test_optimize_database_cleans_references(updater: DatabaseUpdater) -> None:
    """Test that the updater's cleanup now removes the column from ``cluster_by``.

    Args:
        updater: DatabaseUpdater fixture.
    """
    updater.update_cluster_by(["value", "id"])
    _nullify(updater.conn, "value")

    updater.optimize_database()

    assert "value" not in _fact_columns(updater.conn)
    assert "value" not in _metadata_names(updater.conn)
    assert updater._get_cluster_by_columns() == ["id"]
