# Importation des modules
# Modules de base
import warnings
from datetime import datetime

import polars as pl

# Module de tests
import pytest

from dt_ducklake_manager.maintenance import DatabaseAuditor, ValidationLevel
from dt_ducklake_manager.operations import DatabaseDeleter, DatabaseUpdater

# Modules du package à tester
from dt_ducklake_manager.schema import DuckLakeTablesBuilder

# ===========================================================================
# Scénario 1 : construction complète du schéma
# ===========================================================================


# Test de la construction complète du schéma à partir de données locales
@pytest.mark.integration
def test_full_schema_build_from_local_data(sample_df):
    """Test the full schema build pipeline from a local DataFrame.

    Verifies that DuckLakeTablesBuilder creates all three layers of the schema
    (metadata, dimension tables, fact table) from a sample polars DataFrame.

    Args:
        sample_df: Sample polars DataFrame with categorical and numeric columns.
    """
    # Construction du schéma complet (connexion en mémoire pour l'isolation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df, categorical_threshold=4, primary_keys=["id"]
        )
    builder.build_schema()

    # Vérification de l'existence des tables attendues
    tables = [row[0] for row in builder.conn.execute("SHOW TABLES").fetchall()]
    assert "metadata" in tables
    assert "fact_table" in tables
    # Vérification de la présence des tables de dimension pour les colonnes
    # catégorielles
    assert "dim_category" in tables
    assert "dim_status" in tables

    # Vérification que la fact table contient le bon nombre de lignes
    row_count = builder.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert row_count == len(sample_df)

    # Vérification que la table des méta-données contient des entrées
    meta_count = builder.conn.execute("SELECT COUNT(*) FROM metadata").fetchone()[0]
    assert meta_count == len(sample_df.columns)


# ===========================================================================
# Scénario 2 : construction puis mise à jour
# ===========================================================================


# Test du pipeline : construction du schéma puis mise à jour avec de nouvelles données
@pytest.mark.integration
def test_build_then_update(built_ducklake_schema, sample_df):
    """Test the build-then-update pipeline.

    Verifies that after building a schema, DatabaseUpdater can insert new rows
    and that the fact table correctly reflects the additional data.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
        sample_df: Sample polars DataFrame.
    """
    # Comptage initial des lignes
    initial_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert initial_count == len(sample_df)

    # Définition des nouvelles lignes à insérer
    new_data = pl.DataFrame(
        {
            "id": [100, 101],
            "category": ["A", "B"],
            "value": [9.9, 8.8],
            "date": pl.date_range(
                datetime(2024, 7, 1), datetime(2024, 7, 2), "1d", eager=True
            ),
            "status": ["active", "inactive"],
            "high_cardinality": ["val_900", "val_901"],
        }
    )

    # Mise à jour de la base avec les nouvelles lignes
    updater = DatabaseUpdater(
        connection=built_ducklake_schema,
        categorical_threshold=4,
        enable_validation=True,
    )
    result = updater.update_database(new_data, use_transaction=False)
    assert result is True

    # Vérification que les nouvelles lignes ont bien été insérées
    final_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert final_count > initial_count

    # Vérification que les lignes avec id=100 et id=101 sont présentes
    count_new = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id IN (100, 101)"
    ).fetchone()[0]
    assert count_new >= 2


# ===========================================================================
# Scénario 3 : construction puis suppression de lignes
# ===========================================================================


# Test du pipeline : construction du schéma puis suppression de lignes filtrées
@pytest.mark.integration
def test_build_then_delete_rows(built_ducklake_schema, sample_df):
    """Test the build-then-delete pipeline.

    Verifies that after building a schema, DatabaseDeleter can remove rows
    matching a filter and that the fact table correctly reflects the deletion.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
        sample_df: Sample polars DataFrame.
    """
    # Comptage initial
    initial_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert initial_count == len(sample_df)

    # Vérification de la présence de la ligne id=1 avant suppression
    before_delete = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 1"
    ).fetchone()[0]
    assert before_delete >= 1

    # Suppression de la ligne avec id=1
    deleter = DatabaseDeleter(
        connection=built_ducklake_schema,
        categorical_threshold=4,
        enable_validation=False,  # Désactivé pour simplifier le test d'intégration
        auto_cleanup=False,
    )
    deleted = deleter.delete_rows(filters=[("id", "=", 1)], use_transaction=False)
    assert deleted >= 1

    # Vérification que la ligne est bien absente
    after_delete = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 1"
    ).fetchone()[0]
    assert after_delete == 0

    # Vérification que le nombre total de lignes a diminué
    final_count = built_ducklake_schema.execute(
        "SELECT COUNT(*) FROM fact_table"
    ).fetchone()[0]
    assert final_count < initial_count


# ===========================================================================
# Scénario 4 : construction puis audit de la base
# ===========================================================================


# Test du pipeline : construction du schéma puis audit d'intégrité
@pytest.mark.integration
def test_build_then_audit(built_ducklake_schema):
    """Test the build-then-audit pipeline.

    Verifies that after building a schema, DatabaseAuditor can validate the
    database and returns no critical issues for a healthy database.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    # Audit de la base fraîchement construite
    auditor = DatabaseAuditor(connection=built_ducklake_schema, categorical_threshold=4)
    report = auditor.validate_database(ValidationLevel.STANDARD)

    # Vérification que le rapport est bien retourné
    from dt_ducklake_manager.maintenance import ValidationReport

    assert isinstance(report, ValidationReport)

    # Une base fraîche et correctement construite ne doit pas avoir de problèmes
    # critiques
    assert report.get_critical_issues_count() == 0

    # Vérification que la table des faits est reconnue dans les tables validées
    assert len(report.tables_validated) >= 1


# ===========================================================================
# Scénario 5 : pipeline complet build → update → delete → audit
# ===========================================================================


# Test du pipeline complet : construction → mise à jour → suppression → audit
@pytest.mark.integration
def test_full_pipeline_build_update_delete_audit(sample_df):
    """Test the full pipeline: build schema, update, delete rows, then audit.

    This end-to-end test verifies that the three main operations (build, update,
    delete) work together correctly, and that the final state passes an integrity
    audit without critical issues.

    Args:
        sample_df: Sample polars DataFrame.
    """
    # Étape 1 : Construction du schéma
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            sample_df, categorical_threshold=4, primary_keys=["id"]
        )
    builder.build_schema()
    conn = builder.conn

    # Vérification de la construction
    row_count = conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert row_count == len(sample_df)

    # Étape 2 : Mise à jour avec de nouvelles lignes
    new_data = pl.DataFrame(
        {
            "id": [200],
            "category": ["A"],
            "value": [7.7],
            "date": pl.date_range(
                datetime(2024, 8, 1), datetime(2024, 8, 1), "1d", eager=True
            ),
            "status": ["active"],
            "high_cardinality": ["val_800"],
        }
    )
    updater = DatabaseUpdater(
        connection=conn, categorical_threshold=4, enable_validation=False
    )
    update_result = updater.update_database(new_data, use_transaction=False)
    assert update_result is True

    # Vérification de l'insertion
    count_after_update = conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
    assert count_after_update > row_count

    # Étape 3 : Suppression d'une ligne
    deleter = DatabaseDeleter(
        connection=conn,
        categorical_threshold=4,
        enable_validation=False,
        auto_cleanup=False,
    )
    deleted = deleter.delete_rows(filters=[("id", "=", 200)], use_transaction=False)
    assert deleted >= 1

    # Vérification que id=200 est bien supprimé
    count_200 = conn.execute(
        "SELECT COUNT(*) FROM fact_table WHERE id = 200"
    ).fetchone()[0]
    assert count_200 == 0

    # Étape 4 : Audit de la base après les opérations
    auditor = DatabaseAuditor(connection=conn, categorical_threshold=4)
    report = auditor.validate_database(ValidationLevel.BASIC)
    assert report.get_critical_issues_count() == 0
