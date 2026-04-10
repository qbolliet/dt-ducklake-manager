# Importation des modules
# Modules de base
import time
# Module de tests
import pytest
# Modules du package à tester
from dashboard_template_database.maintenance import (
    DatabaseAuditor,
    ValidationLevel,
    ValidationReport,
    ValidationIssue,
)
from dashboard_template_database.maintenance.auditor import IssueType, IssueSeverity


# ===========================================================================
# Tests de ValidationIssue
# ===========================================================================

# Test de l'initialisation d'un ValidationIssue
def test_validation_issue_initialization():
    """Test that ValidationIssue is correctly initialized with all fields.

    Examples:
        >>> issue = ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.HIGH, 'fact_table')
        >>> issue.severity
        <IssueSeverity.HIGH: 'high'>
    """
    issue = ValidationIssue(
        issue_type=IssueType.DATA_INTEGRITY,
        severity=IssueSeverity.HIGH,
        table_name='fact_table',
        column_name='id',
        description='Duplicate primary keys',
        suggested_fix='Remove duplicates',
    )
    assert issue.issue_type == IssueType.DATA_INTEGRITY
    assert issue.severity == IssueSeverity.HIGH
    assert issue.table_name == 'fact_table'
    assert issue.column_name == 'id'
    # Vérification que detected_at est initialisé automatiquement
    assert issue.detected_at > 0


# ===========================================================================
# Tests de ValidationReport
# ===========================================================================

# Initialisation d'un rapport d'audit utilisé dans les tests de ValidationReport
@pytest.fixture
def empty_report():
    """Create an empty ValidationReport for testing.

    Returns:
        ValidationReport: a new empty report at STANDARD level.
    """
    return ValidationReport(validation_level=ValidationLevel.STANDARD)


# Test de l'ajout d'un problème au rapport
def test_validation_report_add_issue(empty_report):
    """Test that add_issue appends an issue to the report's issue list.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    # Vérification que le rapport est initialement vide
    assert len(empty_report.issues) == 0

    issue = ValidationIssue(
        issue_type=IssueType.SCHEMA_INCONSISTENCY,
        severity=IssueSeverity.MEDIUM,
        table_name='metadata',
    )
    empty_report.add_issue(issue)

    # Vérification que le problème a bien été ajouté
    assert len(empty_report.issues) == 1


# Test du filtrage des problèmes par sévérité
def test_validation_report_get_issues_by_severity(empty_report):
    """Test that get_issues_by_severity returns only matching issues.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    # Ajout de problèmes de sévérités différentes
    empty_report.add_issue(ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.CRITICAL, 'fact_table'))
    empty_report.add_issue(ValidationIssue(IssueType.SCHEMA_INCONSISTENCY, IssueSeverity.HIGH, 'metadata'))
    empty_report.add_issue(ValidationIssue(IssueType.ORPHANED_REFERENCE, IssueSeverity.CRITICAL, 'dim_category'))

    # Vérification du filtrage par sévérité CRITICAL
    critical = empty_report.get_issues_by_severity(IssueSeverity.CRITICAL)
    assert len(critical) == 2
    # Vérification du filtrage par sévérité HIGH
    high = empty_report.get_issues_by_severity(IssueSeverity.HIGH)
    assert len(high) == 1


# Test du filtrage des problèmes par type
def test_validation_report_get_issues_by_type(empty_report):
    """Test that get_issues_by_type returns only matching issues.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    empty_report.add_issue(ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.HIGH, 'fact_table'))
    empty_report.add_issue(ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.LOW, 'metadata'))
    empty_report.add_issue(ValidationIssue(IssueType.ORPHANED_REFERENCE, IssueSeverity.MEDIUM, 'dim_status'))

    # Vérification du filtrage par type DATA_INTEGRITY
    data_issues = empty_report.get_issues_by_type(IssueType.DATA_INTEGRITY)
    assert len(data_issues) == 2
    orphan_issues = empty_report.get_issues_by_type(IssueType.ORPHANED_REFERENCE)
    assert len(orphan_issues) == 1


# Test du comptage des problèmes critiques
def test_validation_report_get_critical_issues_count(empty_report):
    """Test that get_critical_issues_count returns the exact count of CRITICAL issues.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    # Ajout de problèmes de sévérités variées
    empty_report.add_issue(ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.CRITICAL, 'fact_table'))
    empty_report.add_issue(ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.HIGH, 'fact_table'))

    # Vérification que seul le CRITICAL est compté
    assert empty_report.get_critical_issues_count() == 1


# Test de la finalisation du rapport
def test_validation_report_finalize(empty_report):
    """Test that finalize() sets end_time and populates validation_summary.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    # Ajout d'un problème avant la finalisation
    empty_report.add_issue(ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.CRITICAL, 'fact_table'))

    empty_report.finalize()

    # Vérification que end_time est renseigné
    assert empty_report.end_time is not None
    assert empty_report.end_time >= empty_report.start_time
    # Vérification de la présence des statistiques clés
    assert 'total_issues' in empty_report.validation_summary
    assert empty_report.validation_summary['total_issues'] == 1
    assert empty_report.validation_summary['critical_issues'] == 1
    # Vérification que des recommandations ont été générées pour les issues critiques
    assert len(empty_report.recommendations) > 0


# Test que finalize génère des recommandations pour les issues d'intégrité de schéma
def test_validation_report_finalize_generates_schema_recommendation(empty_report):
    """Test that finalize generates a recommendation when schema issues are present.

    Args:
        empty_report: Empty ValidationReport fixture.
    """
    empty_report.add_issue(ValidationIssue(IssueType.SCHEMA_INCONSISTENCY, IssueSeverity.HIGH, 'metadata'))
    empty_report.finalize()

    # Vérification qu'une recommandation de schéma a été générée
    assert any('schema' in r.lower() or 'Schema' in r for r in empty_report.recommendations)


# ===========================================================================
# Tests de DatabaseAuditor
# ===========================================================================

# Test de l'initialisation de DatabaseAuditor sans connexion
def test_database_auditor_initialization_without_connection():
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
def test_database_auditor_initialization_with_connection(built_ducklake_schema):
    """Test that DatabaseAuditor stores the provided connection.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    auditor = DatabaseAuditor(connection=built_ducklake_schema, categorical_threshold=10)
    assert auditor is not None


# Test de validate_database au niveau BASIC
def test_validate_database_basic_level(built_ducklake_schema):
    """Test that validate_database with BASIC level returns a ValidationReport.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    auditor = DatabaseAuditor(connection=built_ducklake_schema)
    report = auditor.validate_database(ValidationLevel.BASIC)

    # Vérification du type de retour et du niveau de validation
    assert isinstance(report, ValidationReport)
    assert report.validation_level == ValidationLevel.BASIC
    # Vérification que le rapport est finalisé (end_time renseigné)
    assert report.end_time is not None


# Test de validate_database au niveau STANDARD
def test_validate_database_standard_level(built_ducklake_schema):
    """Test that validate_database with STANDARD level returns a ValidationReport.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    auditor = DatabaseAuditor(connection=built_ducklake_schema)
    report = auditor.validate_database(ValidationLevel.STANDARD)

    assert isinstance(report, ValidationReport)
    assert report.validation_level == ValidationLevel.STANDARD
    # Une base saine ne doit pas avoir de problèmes critiques
    assert report.get_critical_issues_count() == 0


# Test de validate_database au niveau COMPREHENSIVE
def test_validate_database_comprehensive_level(built_ducklake_schema):
    """Test that validate_database with COMPREHENSIVE level returns a ValidationReport.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    auditor = DatabaseAuditor(connection=built_ducklake_schema)
    report = auditor.validate_database(ValidationLevel.COMPREHENSIVE)

    assert isinstance(report, ValidationReport)
    assert report.validation_level == ValidationLevel.COMPREHENSIVE


# Test de get_quick_health_check
def test_get_quick_health_check(built_ducklake_schema):
    """Test that get_quick_health_check returns a dict with expected keys.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    auditor = DatabaseAuditor(connection=built_ducklake_schema)
    health = auditor.get_quick_health_check()

    # Vérification que le résultat est un dictionnaire
    assert isinstance(health, dict)
    # Vérification de la présence des clés retournées par get_quick_health_check
    assert 'fact_table_rows' in health
    assert 'metadata_entries' in health
    assert 'status' in health


# Test de validate_operation_preconditions pour une opération d'insertion
def test_validate_operation_preconditions_insert(built_ducklake_schema, sample_df):
    """Test that validate_operation_preconditions for 'insert' returns a ValidationReport.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
        sample_df: Sample polars DataFrame.
    """
    import polars as pl
    auditor = DatabaseAuditor(connection=built_ducklake_schema)
    new_row = pl.DataFrame({'id': [99], 'category': ['A'], 'value': [9.9],
                             'date': sample_df['date'][:1],
                             'status': ['active'], 'high_cardinality': ['val_999']})
    report = auditor.validate_operation_preconditions('insert', df=new_row)

    assert isinstance(report, ValidationReport)


# Test de validate_operation_preconditions pour une opération de suppression
def test_validate_operation_preconditions_delete(built_ducklake_schema):
    """Test that validate_operation_preconditions for 'delete' returns a ValidationReport.

    Args:
        built_ducklake_schema: Fixture providing a DuckDB connection with a built schema.
    """
    auditor = DatabaseAuditor(connection=built_ducklake_schema)
    report = auditor.validate_operation_preconditions('delete', filters=[('id', '=', 1)])

    assert isinstance(report, ValidationReport)
