# Importation des modules
# Modules de base
import duckdb
import polars as pl

# Module de tests
import pytest

# Modules à tester
from dt_ducklake_manager.maintenance import DatabaseAuditor, ValidationLevel
from dt_ducklake_manager.operations import DatabaseDeleter, DatabaseUpdater

# ===========================================================================
# Isolation des opérations entre schémas d'un même catalogue
# ===========================================================================


# Test qu'une mise à jour sur un schéma n'affecte pas l'autre schéma du catalogue
def test_update_isolated_between_schemas(
    multi_schema_connection: duckdb.DuckDBPyConnection,
) -> None:
    """Test that updating one schema leaves the other schema untouched.

    Args:
        multi_schema_connection: Connection with 'predictions' and 'shapley' schemas.
    """
    conn = multi_schema_connection

    # Comptages initiaux
    pred_before_row = conn.execute(
        "SELECT COUNT(*) FROM predictions.fact_table"
    ).fetchone()
    assert pred_before_row is not None
    pred_before = pred_before_row[0]
    shap_before_row = conn.execute("SELECT COUNT(*) FROM shapley.fact_table").fetchone()
    assert shap_before_row is not None
    shap_before = shap_before_row[0]

    # Insertion d'une nouvelle ligne dans 'predictions' uniquement
    new_pred = pl.DataFrame({"id": [4], "category": ["B"], "value": [0.9]})
    updater = DatabaseUpdater(
        connection=conn,
        categorical_threshold=4,
        enable_validation=False,
        schema="predictions",
    )
    assert updater.update_database(new_pred, use_transaction=False) is True

    # 'predictions' a bien augmenté, 'shapley' est inchangé
    pred_after_row = conn.execute(
        "SELECT COUNT(*) FROM predictions.fact_table"
    ).fetchone()
    assert pred_after_row is not None
    pred_after = pred_after_row[0]
    shap_after_row = conn.execute("SELECT COUNT(*) FROM shapley.fact_table").fetchone()
    assert shap_after_row is not None
    shap_after = shap_after_row[0]
    assert pred_after == pred_before + 1
    assert shap_after == shap_before


# Test qu'une suppression sur un schéma n'affecte pas l'autre schéma du catalogue
def test_delete_isolated_between_schemas(
    multi_schema_connection: duckdb.DuckDBPyConnection,
) -> None:
    """Test that deleting from one schema leaves the other schema untouched.

    Args:
        multi_schema_connection: Connection with 'predictions' and 'shapley' schemas.
    """
    conn = multi_schema_connection

    # Comptages initiaux
    pred_before_row = conn.execute(
        "SELECT COUNT(*) FROM predictions.fact_table"
    ).fetchone()
    assert pred_before_row is not None
    pred_before = pred_before_row[0]
    shap_before_row = conn.execute("SELECT COUNT(*) FROM shapley.fact_table").fetchone()
    assert shap_before_row is not None
    shap_before = shap_before_row[0]

    # Suppression d'une ligne dans 'shapley' uniquement
    deleter = DatabaseDeleter(
        connection=conn,
        categorical_threshold=4,
        enable_validation=False,
        auto_cleanup=False,
        schema="shapley",
    )
    deleted = deleter.delete_rows(filters=[("id", "=", 1)], use_transaction=False)
    assert deleted >= 1

    # 'shapley' a diminué, 'predictions' est inchangé
    pred_after_row = conn.execute(
        "SELECT COUNT(*) FROM predictions.fact_table"
    ).fetchone()
    assert pred_after_row is not None
    pred_after = pred_after_row[0]
    shap_after_row = conn.execute("SELECT COUNT(*) FROM shapley.fact_table").fetchone()
    assert shap_after_row is not None
    shap_after = shap_after_row[0]
    assert shap_after == shap_before - 1
    assert pred_after == pred_before


# Test que l'audit cible le bon schéma et ne signale pas de problème critique
@pytest.mark.parametrize("schema_name", ["predictions", "shapley"])
def test_audit_targets_correct_schema(
    multi_schema_connection: duckdb.DuckDBPyConnection,
    schema_name: str,
) -> None:
    """Test that an auditor validates the targeted schema without critical issues.

    Args:
        multi_schema_connection: Connection with 'predictions' and 'shapley' schemas.
        schema_name: Name of the schema to audit.
    """
    auditor = DatabaseAuditor(
        connection=multi_schema_connection,
        categorical_threshold=4,
        schema=schema_name,
    )
    report = auditor.validate_database(ValidationLevel.STANDARD)
    # Un schéma fraîchement construit ne doit pas comporter de problème critique
    assert report.get_critical_issues_count() == 0
    # La fact_table du schéma ciblé est bien reconnue
    assert "fact_table" in report.tables_validated
