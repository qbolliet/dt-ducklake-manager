# Importation des modules
# Modules de base
import warnings
from typing import Any

import polars as pl

# Module de tests
import pytest

# Modules du package à tester
from dt_ducklake_manager.maintenance import (
    DatabaseAuditor,
    IssueSeverity,
    IssueType,
    ValidationIssue,
    ValidationLevel,
    ValidationReport,
)
from dt_ducklake_manager.schema import DuckLakeTablesBuilder

# ===========================================================================
# Tests de ValidationIssue
# ===========================================================================


# Test de l'initialisation d'un ValidationIssue
def test_validation_issue_initialization() -> None:
    """Test that ValidationIssue is correctly initialized with all fields.

    Examples:
        >>> issue = ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.HIGH,
        'fact_table')
        >>> issue.severity
        <IssueSeverity.HIGH: 'high'>
    """
    issue = ValidationIssue(
        issue_type=IssueType.DATA_INTEGRITY,
        severity=IssueSeverity.HIGH,
        table_name="fact_table",
        column_name="id",
        description="Duplicate primary keys",
        suggested_fix="Remove duplicates",
    )
    assert issue.issue_type == IssueType.DATA_INTEGRITY
    assert issue.severity == IssueSeverity.HIGH
    assert issue.table_name == "fact_table"
    assert issue.column_name == "id"
    # Vérification que detected_at est initialisé automatiquement
    assert issue.detected_at > 0


# ===========================================================================
# Tests de ValidationReport
# ===========================================================================


# Initialisation d'un rapport d'audit utilisé dans les tests de ValidationReport
@pytest.fixture
def empty_report() -> ValidationReport:
    """Create an empty ValidationReport for testing.

    Returns:
        ValidationReport: a new empty report at BASIC level.
    """
    return ValidationReport(validation_level=ValidationLevel.BASIC)


# Test de l'ajout d'un problème au rapport
def test_validation_report_add_issue(empty_report: Any) -> None:
    """Test that add_issue appends an issue to the report's issue list.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    # Vérification que le rapport est initialement vide
    assert len(empty_report.issues) == 0

    issue = ValidationIssue(
        issue_type=IssueType.SCHEMA_INCONSISTENCY,
        severity=IssueSeverity.MEDIUM,
        table_name="metadata",
    )
    empty_report.add_issue(issue)

    # Vérification que le problème a bien été ajouté
    assert len(empty_report.issues) == 1


# Test du filtrage des problèmes par sévérité
def test_validation_report_get_issues_by_severity(empty_report: Any) -> None:
    """Test that get_issues_by_severity returns only matching issues.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    # Ajout de problèmes de sévérités différentes
    empty_report.add_issue(
        ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.CRITICAL, "fact_table")
    )
    empty_report.add_issue(
        ValidationIssue(IssueType.SCHEMA_INCONSISTENCY, IssueSeverity.HIGH, "metadata")
    )
    empty_report.add_issue(
        ValidationIssue(
            IssueType.MISSING_METADATA, IssueSeverity.CRITICAL, "dataset_metadata"
        )
    )

    # Vérification du filtrage par sévérité CRITICAL
    critical = empty_report.get_issues_by_severity(IssueSeverity.CRITICAL)
    assert len(critical) == 2
    # Vérification du filtrage par sévérité HIGH
    high = empty_report.get_issues_by_severity(IssueSeverity.HIGH)
    assert len(high) == 1


# Test du filtrage des problèmes par type
def test_validation_report_get_issues_by_type(empty_report: Any) -> None:
    """Test that get_issues_by_type returns only matching issues.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    empty_report.add_issue(
        ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.HIGH, "fact_table")
    )
    empty_report.add_issue(
        ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.LOW, "metadata")
    )
    empty_report.add_issue(
        ValidationIssue(
            IssueType.MISSING_METADATA, IssueSeverity.MEDIUM, "dataset_metadata"
        )
    )

    # Vérification du filtrage par type DATA_INTEGRITY
    data_issues = empty_report.get_issues_by_type(IssueType.DATA_INTEGRITY)
    assert len(data_issues) == 2
    metadata_issues = empty_report.get_issues_by_type(IssueType.MISSING_METADATA)
    assert len(metadata_issues) == 1


# Test du comptage des problèmes critiques
def test_validation_report_get_critical_issues_count(empty_report: Any) -> None:
    """Test that get_critical_issues_count returns the exact count of CRITICAL issues.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    # Ajout de problèmes de sévérités variées
    empty_report.add_issue(
        ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.CRITICAL, "fact_table")
    )
    empty_report.add_issue(
        ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.HIGH, "fact_table")
    )

    # Vérification que seul le CRITICAL est compté
    assert empty_report.get_critical_issues_count() == 1


# Test de la finalisation du rapport
def test_validation_report_finalize(empty_report: Any) -> None:
    """Test that finalize() sets end_time and populates validation_summary.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    # Ajout d'un problème avant la finalisation
    empty_report.add_issue(
        ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.CRITICAL, "fact_table")
    )

    empty_report.finalize()

    # Vérification que end_time est renseigné
    assert empty_report.end_time is not None
    assert empty_report.end_time >= empty_report.start_time
    # Vérification de la présence des statistiques clés
    assert "total_issues" in empty_report.validation_summary
    assert empty_report.validation_summary["total_issues"] == 1
    assert empty_report.validation_summary["critical_issues"] == 1
    # Vérification que des recommandations ont été générées pour les issues critiques
    assert len(empty_report.recommendations) > 0


# Test que finalize génère des recommandations pour les issues d'intégrité de schéma
def test_validation_report_finalize_generates_schema_recommendation(
    empty_report: Any,
) -> None:
    """Test that finalize generates a recommendation when schema issues are present.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    empty_report.add_issue(
        ValidationIssue(IssueType.SCHEMA_INCONSISTENCY, IssueSeverity.HIGH, "metadata")
    )
    empty_report.finalize()

    # Vérification qu'une recommandation de schéma a été générée
    assert any(
        "schema" in r.lower() or "Schema" in r for r in empty_report.recommendations
    )


# ===========================================================================
# Tests de DatabaseAuditor
# ===========================================================================


# Test de l'initialisation de DatabaseAuditor sans connexion
def test_database_auditor_initialization_without_connection() -> None:
    """Test that DatabaseAuditor can be initialized without a connection.

    Examples:
        >>> auditor = DatabaseAuditor()
        >>> auditor is not None
        True
    """
    # L'initialisation sans connexion ne doit pas lever d'exception
    auditor = DatabaseAuditor()
    assert auditor is not None


# Test de l'initialisation de DatabaseAuditor avec une connexion fournie
def test_database_auditor_initialization_with_connection(
    built_ducklake_schema: Any,
) -> None:
    """Test that DatabaseAuditor stores the provided connection.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    auditor = DatabaseAuditor(connection=built_ducklake_schema)
    assert auditor is not None


# Test que l'alias du catalogue est conservé au même titre que le schéma
def test_database_auditor_catalog_alias(built_ducklake_schema: Any) -> None:
    """Test that ``catalog_alias`` defaults to 'db' and is stored when provided.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
    """
    assert DatabaseAuditor(connection=built_ducklake_schema).catalog_alias == "db"
    custom = DatabaseAuditor(
        connection=built_ducklake_schema, schema="predictions", catalog_alias="my_lake"
    )
    assert custom.catalog_alias == "my_lake"
    assert custom.schema == "predictions"


# Fonction auxiliaire d'extraction des descriptions de problèmes
def _descriptions(report: ValidationReport) -> list[str]:
    """Return the descriptions of every issue of a report.

    Args:
        report: Finalized validation report.

    Returns:
        list[str]: Issue descriptions, in detection order.
    """
    return [issue.description for issue in report.issues]


# Test qu'un schéma sain ne présente aucun problème, aux deux niveaux
@pytest.mark.parametrize(
    "level", [ValidationLevel.BASIC, ValidationLevel.COMPREHENSIVE]
)
def test_validate_database_clean_schema_has_no_issue(
    built_ducklake_schema: Any, level: ValidationLevel
) -> None:
    """Test that a freshly built schema passes both audit levels.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built
        schema.
        level: Audit level.
    """
    report = DatabaseAuditor(connection=built_ducklake_schema).validate_database(level)
    assert report.validation_level == level
    assert report.issues == []
    assert report.tables_validated == {"fact_table", "metadata", "dataset_metadata"}
    assert report.end_time is not None


# Test que le niveau BASIC ne balaie jamais la table des faits
def test_validate_database_basic_does_not_scan_fact_table(
    built_ducklake_schema: Any,
) -> None:
    """Test that BASIC ignores data problems only a scan could detect.

    A duplicated primary key is invisible at the BASIC level (no fact table scan)
    and reported at the COMPREHENSIVE level.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    built_ducklake_schema.execute(
        "INSERT INTO fact_table SELECT * FROM fact_table WHERE id = 1"
    )
    auditor = DatabaseAuditor(connection=built_ducklake_schema)

    assert auditor.validate_database(ValidationLevel.BASIC).issues == []
    comprehensive = auditor.validate_database(ValidationLevel.COMPREHENSIVE)
    violations = comprehensive.get_issues_by_type(IssueType.CONSTRAINT_VIOLATION)
    assert len(violations) == 1
    assert violations[0].affected_rows == 1


# Test qu'une table manquante est signalée avec la sévérité attendue
@pytest.mark.parametrize(
    ("table", "severity"),
    [
        ("metadata", IssueSeverity.CRITICAL),
        ("fact_table", IssueSeverity.HIGH),
        ("dataset_metadata", IssueSeverity.HIGH),
    ],
)
def test_validate_database_reports_missing_table(
    built_ducklake_schema: Any, table: str, severity: IssueSeverity
) -> None:
    """Test that each missing table of the result set is reported.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
        table: Table dropped before the audit.
        severity: Expected severity of the issue.
    """
    built_ducklake_schema.execute(f"DROP TABLE {table}")
    report = DatabaseAuditor(connection=built_ducklake_schema).validate_database()
    missing = [
        i for i in report.issues if i.description == f"Table '{table}' is missing"
    ]
    assert len(missing) == 1
    assert missing[0].severity == severity


# Test qu'une connexion illisible produit un problème critique au lieu d'une exception
def test_validate_database_unreadable_state_is_critical(
    built_ducklake_schema: Any,
) -> None:
    """Test that a failure to read the schema state becomes a critical issue.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    auditor = DatabaseAuditor(connection=built_ducklake_schema)

    def _boom() -> Any:
        raise RuntimeError("catalog unreachable")

    setattr(auditor, "_read_state", _boom)
    report = auditor.validate_database()
    assert report.get_critical_issues_count() == 1
    assert "catalog unreachable" in report.issues[0].description


# Test qu'un contrôle en échec est signalé sans interrompre l'audit
def test_validate_database_failing_check_is_reported(
    built_ducklake_schema: Any,
) -> None:
    """Test that a check raising an exception is reported and the others still run.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    auditor = DatabaseAuditor(connection=built_ducklake_schema)

    def _boom(report: Any, state: Any) -> None:
        raise RuntimeError("check crashed")

    setattr(auditor, "_validate_dataset_metadata", _boom)
    built_ducklake_schema.execute("DELETE FROM metadata WHERE name = 'status'")

    report = auditor.validate_database()
    descriptions = _descriptions(report)
    assert any("check crashed" in d for d in descriptions)
    assert any("'status'" in d for d in descriptions)


# Test que la cardinalité de dataset_metadata est contrôlée
def test_validate_dataset_metadata_extra_row(built_ducklake_schema: Any) -> None:
    """Test that a second dataset_metadata row is reported.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    built_ducklake_schema.execute(
        "INSERT INTO dataset_metadata SELECT * FROM dataset_metadata"
    )
    report = DatabaseAuditor(connection=built_ducklake_schema).validate_database()
    issues = report.get_issues_by_type(IssueType.MISSING_METADATA)
    assert len(issues) == 1
    assert issues[0].affected_rows == 2


# Test que les écarts entre metadata et la table des faits sont signalés
def test_validate_metadata_fact_consistency_both_directions(
    built_ducklake_schema: Any,
) -> None:
    """Test that an undescribed column and an orphan metadata row are both reported.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    built_ducklake_schema.execute("ALTER TABLE fact_table ADD COLUMN extra DOUBLE")
    built_ducklake_schema.execute("ALTER TABLE fact_table DROP COLUMN status")

    report = DatabaseAuditor(connection=built_ducklake_schema).validate_database()
    descriptions = _descriptions(report)
    assert any(
        "'extra' exists in fact_table but has no metadata" in d for d in descriptions
    )
    assert any(
        "'status' is described in metadata but missing" in d for d in descriptions
    )


# Test qu'un écart de type entre metadata et la table des faits est signalé
def test_validate_data_types_consistency_detects_mismatch(
    built_ducklake_schema: Any,
) -> None:
    """Test that a declared type differing from the physical one is reported.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    built_ducklake_schema.execute(
        "UPDATE metadata SET sql_type = 'INTEGER' WHERE name = 'value'"
    )
    report = DatabaseAuditor(connection=built_ducklake_schema).validate_database()
    mismatches = report.get_issues_by_type(IssueType.TYPE_MISMATCH)
    assert [issue.column_name for issue in mismatches] == ["value"]


# Test des synonymes de types acceptés
@pytest.mark.parametrize(
    ("expected", "actual", "compatible"),
    [
        ("VARCHAR", "varchar", True),
        ("TEXT", "VARCHAR", True),
        ("INT", "INTEGER", True),
        ("BOOL", "BOOLEAN", True),
        ("INTEGER", "BIGINT", False),
        ("DOUBLE", "FLOAT", False),
    ],
)
def test_types_are_compatible(expected: str, actual: str, compatible: bool) -> None:
    """Test the declared/physical type comparison.

    Args:
        expected: Type declared in metadata.
        actual: Physical type.
        compatible: Expected outcome.
    """
    assert DatabaseAuditor._types_are_compatible(expected, actual) is compatible


# Test que la forêt des parent_name est contrôlée (cycle et parent inconnu)
def test_validate_column_links_detects_cycle_and_unknown_parent(
    built_ducklake_schema: Any,
) -> None:
    """Test that a parent_name cycle and a dangling parent are both reported.

    The links are written directly in metadata, as a base modified outside of the
    package would be; ``update_column_metadata`` refuses them.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    built_ducklake_schema.execute(
        "UPDATE metadata SET parent_name = 'status' WHERE name = 'category'"
    )
    built_ducklake_schema.execute(
        "UPDATE metadata SET parent_name = 'category' WHERE name = 'status'"
    )
    built_ducklake_schema.execute(
        "UPDATE metadata SET parent_name = 'ghost' WHERE name = 'value'"
    )
    report = DatabaseAuditor(connection=built_ducklake_schema).validate_database()
    descriptions = _descriptions(report)
    assert any("Cycle detected" in d for d in descriptions)
    assert any("references 'ghost'" in d for d in descriptions)


# Test que les déclarations label_for invalides sont signalées au niveau BASIC
def test_validate_column_links_detects_invalid_label_for(
    built_ducklake_schema: Any,
) -> None:
    """Test that a label_for on a non-VARCHAR column is reported without a scan.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    built_ducklake_schema.execute(
        "UPDATE metadata SET label_for = 'category' WHERE name = 'value'"
    )
    report = DatabaseAuditor(connection=built_ducklake_schema).validate_database(
        ValidationLevel.BASIC
    )
    assert any("Invalid label_for declaration" in d for d in _descriptions(report))


# Test que la dépendance code -> libellé est contrôlée au niveau COMPREHENSIVE
def test_validate_value_label_dependencies_detects_violation() -> None:
    """Test that a code carrying two labels is a critical COMPREHENSIVE issue."""
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "nc8": ["01", "01", "02"],
            "nc8_libelle": ["Chevaux", "Chevaux", "Bovins"],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        builder = DuckLakeTablesBuilder(
            df,
            categorical_threshold=10,
            primary_keys=["id"],
            value_labels={"nc8_libelle": "nc8"},
        )
    builder.build_schema()
    # Corruption hors du package : un second libellé pour le code '01'
    builder.conn.execute("UPDATE fact_table SET nc8_libelle = 'Autre' WHERE id = 2")

    auditor = DatabaseAuditor(connection=builder.conn)
    assert auditor.validate_database(ValidationLevel.BASIC).issues == []
    report = auditor.validate_database(ValidationLevel.COMPREHENSIVE)
    assert report.get_critical_issues_count() == 1
    assert report.issues[0].column_name == "nc8_libelle"


# Test du contrôle de qualité : colonne entièrement nulle et colonne majoritairement
# nulle
def test_validate_data_quality_distinguishes_null_shares(
    built_ducklake_schema: Any,
) -> None:
    """Test that a fully null column is HIGH and a mostly null column is MEDIUM.

    The fully null case is tested before the 50% threshold, which it also exceeds.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema (5 rows).
    """
    built_ducklake_schema.execute("UPDATE fact_table SET status = NULL")
    built_ducklake_schema.execute("UPDATE fact_table SET value = NULL WHERE id <= 3")
    report = DatabaseAuditor(connection=built_ducklake_schema).validate_database(
        ValidationLevel.COMPREHENSIVE
    )
    by_column = {
        issue.column_name: issue
        for issue in report.get_issues_by_type(IssueType.DATA_INTEGRITY)
    }
    assert by_column["status"].severity == IssueSeverity.HIGH
    assert "only null values" in by_column["status"].description
    assert by_column["value"].severity == IssueSeverity.MEDIUM
    assert by_column["value"].affected_rows == 3


# Test du contrôle de qualité sur une table vide
def test_validate_data_quality_empty_fact_table(built_ducklake_schema: Any) -> None:
    """Test that an empty fact table is reported once, not column by column.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    built_ducklake_schema.execute("DELETE FROM fact_table")
    report = DatabaseAuditor(connection=built_ducklake_schema).validate_database(
        ValidationLevel.COMPREHENSIVE
    )
    assert _descriptions(report) == ["Fact table is empty"]


# Test que l'auditeur audite le schéma demandé
def test_validate_database_targets_its_schema(
    multi_schema_connection: Any,
) -> None:
    """Test that each schema of a shared catalog is audited independently.

    Args:
        multi_schema_connection: Connection with 'predictions' and 'shapley'
            schemas.
    """
    multi_schema_connection.execute("DROP TABLE shapley.dataset_metadata")
    predictions = DatabaseAuditor(multi_schema_connection, schema="predictions")
    shapley = DatabaseAuditor(multi_schema_connection, schema="shapley")
    assert predictions.validate_database().issues == []
    assert any(
        "dataset_metadata" in d for d in _descriptions(shapley.validate_database())
    )


# Test du contrôle de santé rapide
def test_get_quick_health_check(built_ducklake_schema: Any) -> None:
    """Test that the quick health check reports a healthy built schema.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    health = DatabaseAuditor(connection=built_ducklake_schema).get_quick_health_check()
    assert health["status"] == "healthy"
    assert health["fact_table_rows"] == 5
    assert health["metadata_entries"] == 6
    assert health["has_dataset_metadata"] is True
    assert health["critical_issues"] == 0


# Test du contrôle de santé rapide sur un schéma sans métadonnées
def test_get_quick_health_check_critical_without_metadata(
    built_ducklake_schema: Any,
) -> None:
    """Test that a missing metadata table makes the health check critical.

    Args:
        built_ducklake_schema: DuckDB connection with a built schema.
    """
    built_ducklake_schema.execute("DROP TABLE metadata")
    health = DatabaseAuditor(connection=built_ducklake_schema).get_quick_health_check()
    assert health["status"] == "critical"
