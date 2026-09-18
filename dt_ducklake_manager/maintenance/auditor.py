# Importation des modules
# Modules de base
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

# DuckDB
import duckdb
import narwhals as nw
from narwhals.typing import IntoDataFrame

# Import des utilitaires
from ..utils.logger import _init_logger
from ..utils.sql import SchemaScoped, quote_ident, resolve_catalog
from ..utils.types import METADATA_COLUMNS


# Classe des niveaux de validation sur la base de données
class ValidationLevel(Enum):
    """Validation levels for database auditing."""

    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


# Classe des types de problèmes détectés
class IssueType(Enum):
    """Types of issues detected during audit."""

    SCHEMA_INCONSISTENCY = "schema_inconsistency"
    DATA_INTEGRITY = "data_integrity"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_METADATA = "missing_metadata"
    PERFORMANCE_ISSUE = "performance_issue"
    CONSTRAINT_VIOLATION = "constraint_violation"


# Classe des niveaux de sévérité des problèmes détectés
class IssueSeverity(Enum):
    """Issue severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Classe de problème détecté durant l'audit de la base de données
@dataclass
class ValidationIssue:
    """
    Represents a validation issue found during database audit.

    Attributes:
        issue_type (IssueType): Type of the issue
        severity (IssueSeverity): Severity level of the issue
        table_name (str): Name of the affected table
        column_name (Optional[str]): Name of the affected column (if applicable)
        description (str): Detailed description of the issue
        suggested_fix (str): Suggested fix for the issue
        affected_rows (Optional[int]): Number of affected rows (if applicable)
        detected_at (float): Timestamp when issue was detected
        additional_info (dict): Additional information about the issue
    """

    issue_type: IssueType
    severity: IssueSeverity
    table_name: str
    column_name: str | None = None
    description: str = ""
    suggested_fix: str = ""
    affected_rows: int | None = None
    detected_at: float = field(default_factory=time.time)
    additional_info: dict[str, Any] = field(default_factory=dict)


# Classe de rapport d'audit de la base de données
@dataclass
class ValidationReport:
    """
    Contains the results of a database validation audit.

    Attributes:
        validation_level (ValidationLevel): Level of validation performed
        start_time (float): Timestamp when validation started
        end_time (Optional[float]): Timestamp when validation ended
        issues (List[ValidationIssue]): List of issues found
        tables_validated (Set[str]): Set of tables that were validated
        validation_summary (dict): Summary statistics of the validation
        recommendations (List[str]): General recommendations for database health
    """

    validation_level: ValidationLevel
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    issues: list[ValidationIssue] = field(default_factory=list)
    tables_validated: set[str] = field(default_factory=set)
    validation_summary: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)

    # Méthode d'ajout d'un problème
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)

    # Méthode d'extraction des problèmes par sévérité
    def get_issues_by_severity(self, severity: IssueSeverity) -> list[ValidationIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]

    # Méthode d'extraction des problèmes par type
    def get_issues_by_type(self, issue_type: IssueType) -> list[ValidationIssue]:
        """Get issues by type."""
        return [issue for issue in self.issues if issue.issue_type == issue_type]

    # Méthode de comptage des problèmes critiques
    def get_critical_issues_count(self) -> int:
        """Get the count of critical issues."""
        return len(self.get_issues_by_severity(IssueSeverity.CRITICAL))

    # Méthode de finalisation du rapport d'audit avec des statistiques
    def finalize(self) -> None:
        """Finalize the report with statistics."""
        self.end_time = time.time()

        # Calcul des statistiques
        self.validation_summary = {
            "total_issues": len(self.issues),
            "critical_issues": len(self.get_issues_by_severity(IssueSeverity.CRITICAL)),
            "high_issues": len(self.get_issues_by_severity(IssueSeverity.HIGH)),
            "medium_issues": len(self.get_issues_by_severity(IssueSeverity.MEDIUM)),
            "low_issues": len(self.get_issues_by_severity(IssueSeverity.LOW)),
            "tables_validated": len(self.tables_validated),
            "validation_duration": self.end_time - self.start_time
            if self.end_time
            else 0,
        }

        # Génération des recommandations
        self._generate_recommendations()

    # Méthode auxiliaire de génération de recommandations
    def _generate_recommendations(self) -> None:
        """Generate recommendations based on issues found."""
        # Identification des erreurs critiques
        critical_count = self.get_critical_issues_count()
        if critical_count > 0:
            self.recommendations.append(
                f"Adress immediately the {critical_count} critical issues detected"
            )

        # Identification des erreurs de schéma
        schema_issues = self.get_issues_by_type(IssueType.SCHEMA_INCONSISTENCY)
        if schema_issues:
            self.recommendations.append("Review the consistency of the database schema")

        # Identification des méta-données manquantes
        metadata_issues = self.get_issues_by_type(IssueType.MISSING_METADATA)
        if metadata_issues:
            self.recommendations.append(
                "Complete the metadata and dataset_metadata tables"
            )

        # Identification des problèmes de performance
        performance_issues = self.get_issues_by_type(IssueType.PERFORMANCE_ISSUE)
        if performance_issues:
            self.recommendations.append(
                "Run Ducklake maintenance (compaction, snapshot expiration) and review"
                " partition configuration to improve performance"
            )


# Classe d'audit de la base de données
class DatabaseAuditor(SchemaScoped):
    """
    Provides comprehensive database validation and state checking capabilities.

    Validates schema consistency, data integrity, agreement between the metadata
    table and the fact table, and identifies potential performance issues.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection
        schema (str): DuckLake schema audited by this instance
        catalog_alias (str): Alias of the attached DuckLake catalog, carried
            alongside ``schema``.
        logger: Logger instance for audit tracking
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        log_filename: str | os.PathLike[str] | None = None,
        schema: str = "main",
        catalog_alias: str = "db",
    ):
        """
        Initialize the database auditor.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            log_filename: Path to log file.
            schema: DuckLake schema to audit. A catalog can host several schemas;
                each is audited independently. Defaults to ``'main'``.
            catalog_alias: Alias of the attached DuckLake catalog, matching the one
                passed to ``DuckLakeConnector``. Carried alongside ``schema``.
                Defaults to ``'db'``.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> auditor = DatabaseAuditor(conn)
            >>> report = auditor.validate_database(ValidationLevel.COMPREHENSIVE)
            >>> # Auditer un schéma dédié dans le même catalogue
            >>> auditor = DatabaseAuditor(conn, schema='predictions')
        """
        # Initialisation de la connexion DuckLake.
        self.conn = connection if connection is not None else duckdb.connect(":memory:")

        # Schéma DuckLake audité : toutes les requêtes qualifient les tables par ce
        # schéma pour cibler le bon jeu de résultats dans le catalogue.
        self.schema = schema

        # Alias du catalogue DuckLake attaché : conservé au même titre que le schéma.
        self.catalog_alias = catalog_alias

        # Alias de catalogue effectif : utilisé pour la qualification uniquement s'il
        # correspond à une base réellement attachée (None pour les connexions
        # in-memory des tests).
        self._catalog = resolve_catalog(self.conn, self.catalog_alias)

        # Initialisation du logger nommé pour traçabilité des audits.
        # Chemin par défaut centralisé dans utils.logger : <cwd>/logs/<name>.log.
        self.logger = _init_logger(filename=log_filename, name="database_auditor")

    # Méthodes principales de validation
    # Méthode de validation de la base de données
    def validate_database(
        self, validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationReport:
        """
        Perform comprehensive database validation.

        Args:
            validation_level: Level of validation to perform

        Returns:
            ValidationReport containing all detected issues

        Example:
            >>> report = auditor.validate_database(ValidationLevel.COMPREHENSIVE)
            >>> if report.get_critical_issues_count() > 0:
            ...     print("Critical issues found!")
            >>> for issue in report.issues:
            ...     print(f"{issue.severity.value}: {issue.description}")
        """
        # Création du rapport
        report = ValidationReport(validation_level=validation_level)
        # Logging
        self.logger.info(
            f"Starting database validation at level: {validation_level.value}"
        )

        try:
            # Validation de base (toujours effectuée)
            # Validation du schéma
            self._validate_schema_existence(report)
            # Validation de la consistence des méta-données
            self._validate_metadata_consistency(report)
            # Validation des méta-données du jeu de résultats
            self._validate_dataset_metadata(report)

            # Validation standard
            if validation_level in [
                ValidationLevel.STANDARD,
                ValidationLevel.COMPREHENSIVE,
            ]:
                # Validation de l'accord entre metadata et fact_table
                self._validate_metadata_fact_consistency(report)
                # Validation de la consistance des types des données
                self._validate_data_types_consistency(report)

            # Validation complète
            if validation_level == ValidationLevel.COMPREHENSIVE:
                # Vérification de la configuration de partitionnement Ducklake
                self._validate_partition_configuration(report)
                # Validation de la qualité des données
                self._validate_data_quality(report)
                # Validation de la violation des contraintes
                self._validate_constraint_violations(report)
                # Vérification de l'état de maintenance Ducklake (snapshots, fichiers)
                self._validate_ducklake_maintenance(report)

            # Finalisation du rapport
            report.finalize()

            # Logging
            self.logger.info(
                f"Database validation completed. Found {len(report.issues)} issues."
            )
            return report

        except Exception as e:
            # Logging
            self.logger.error(f"Error during database validation: {e}")

            # Ajout d'un problème critique pour l'erreur de validation
            error_issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.CRITICAL,
                table_name="VALIDATION_SYSTEM",
                description=f"Validation process failed: {str(e)}",
                suggested_fix="Check database connection and schema structure",
            )
            report.add_issue(error_issue)
            # Finalisation du rapport
            report.finalize()

            return report

    # Validation des prérequis d'une opération
    def validate_operation_preconditions(
        self, operation_type: str, **kwargs: Any
    ) -> ValidationReport:
        """
        Validate preconditions before executing specific operations.

        Args:
            operation_type: Type of operation ('insert', 'update', 'delete',
                'schema_change')
            **kwargs: Operation-specific parameters

        Returns:
            ValidationReport with precondition validation results

        Example:
            >>> # Before inserting data
            >>> report = auditor.validate_operation_preconditions('insert', df=new_data)
            >>> if report.get_critical_issues_count() == 0:
            ...     # Safe to proceed with insertion
            ...     pass
        """
        # Initialisation du rapport
        report = ValidationReport(validation_level=ValidationLevel.BASIC)
        # Logging
        self.logger.info(f"Validating preconditions for operation: {operation_type}")

        try:
            # Validation distincte suivant le type d'opération
            if operation_type == "insert":
                self._validate_insert_preconditions(report, **kwargs)
            elif operation_type == "update":
                self._validate_update_preconditions(report, **kwargs)
            elif operation_type == "delete":
                self._validate_delete_preconditions(report, **kwargs)
            elif operation_type == "schema_change":
                self._validate_schema_change_preconditions(report, **kwargs)
            else:
                issue = ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.MEDIUM,
                    table_name="OPERATION_VALIDATION",
                    description=f"Unknown operation type: {operation_type}",
                    suggested_fix="Use a supported operation type",
                )
                report.add_issue(issue)

            # Finalisation du rapport
            report.finalize()
            return report

        except Exception as e:
            # Logging
            self.logger.error(f"Error validating operation preconditions: {e}")

            # Création d'un erreur associée à l'opération inconnue
            error_issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.HIGH,
                table_name="OPERATION_VALIDATION",
                description=f"Precondition validation failed: {str(e)}",
                suggested_fix="Check operation parameters and database state",
            )
            report.add_issue(error_issue)
            # Finalisation du rapport
            report.finalize()

            return report

    # Méthodes de validation spécifiques
    # Méthode de validation de l'existence du schéma
    def _validate_schema_existence(self, report: ValidationReport) -> None:
        """Validate existence of essential schema tables."""
        try:
            # Vérification des tables essentielles
            essential_tables = ["metadata"]
            # Extraction des tables de la base de données
            existing_tables = self._get_existing_tables()

            # Parcours des tables essentielles
            for table in essential_tables:
                # Vérification de l'existence de la table
                if table not in existing_tables:
                    issue = ValidationIssue(
                        issue_type=IssueType.SCHEMA_INCONSISTENCY,
                        severity=IssueSeverity.CRITICAL,
                        table_name=table,
                        description=f"Essential table '{table}' is missing",
                        suggested_fix=f"Create the '{table}' table with proper"
                        f" structure",
                    )
                    report.add_issue(issue)
                else:
                    report.tables_validated.add(table)

            # Vérification de l'existence de la table des faits
            if "fact_table" not in existing_tables:
                issue = ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    table_name="fact_table",
                    description="Fact table is missing",
                    suggested_fix="Create the fact table or check if database is"
                    " properly initialized",
                )
                report.add_issue(issue)
            else:
                report.tables_validated.add("fact_table")

        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.CRITICAL,
                table_name="SCHEMA_VALIDATION",
                description=f"Error validating schema existence: {str(e)}",
                suggested_fix="Check database connection and permissions",
            )
            report.add_issue(issue)

    # Méthode de validation de la cohérence des méta-données
    def _validate_metadata_consistency(self, report: ValidationReport) -> None:
        """Validate metadata consistency."""
        try:
            # Chargement des métadonnées
            metadata_df = self._get_metadata()

            # Vérification que la table n'est pas vide
            if len(metadata_df) == 0:
                issue = ValidationIssue(
                    issue_type=IssueType.MISSING_METADATA,
                    severity=IssueSeverity.HIGH,
                    table_name="metadata",
                    description="Metadata table is empty",
                    suggested_fix="Populate metadata table with column information",
                )
                report.add_issue(issue)
                return

            # Vérification des colonnes requises dans metadata
            required_columns = list(METADATA_COLUMNS)
            missing_columns = [
                col for col in required_columns if col not in metadata_df.columns
            ]

            if missing_columns:
                issue = ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    table_name="metadata",
                    description=f"Missing required columns in metadata:"
                    f"{missing_columns}",
                    suggested_fix="Add missing columns to metadata table",
                )
                report.add_issue(issue)

            # Vérification des valeurs nulles dans les colonnes critiques
            for col in ["name", "label", "sql_type"]:
                if col in metadata_df.columns:
                    null_count = metadata_df[col].is_null().sum()
                    if null_count > 0:
                        issue = ValidationIssue(
                            issue_type=IssueType.DATA_INTEGRITY,
                            severity=IssueSeverity.MEDIUM,
                            table_name="metadata",
                            column_name=col,
                            description=f"Found {null_count} null values in critical"
                            f" metadata column '{col}'",
                            suggested_fix=f"Update null values in metadata.{col}",
                            affected_rows=int(null_count),
                        )
                        report.add_issue(issue)

            # Vérification des doublons dans les noms de colonnes (il s'agit d ela clé
            # primaire de la base de données)
            if "name" in metadata_df.columns:
                duplicate_names = metadata_df.filter(nw.col("name").is_duplicated())[
                    "name"
                ].to_list()
                if duplicate_names:
                    issue = ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.HIGH,
                        table_name="metadata",
                        column_name="name",
                        description=f"Duplicate column names in metadata:"
                        f"{duplicate_names}",
                        suggested_fix="Remove or rename duplicate entries in metadata",
                        affected_rows=len(duplicate_names),
                    )
                    report.add_issue(issue)

        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.HIGH,
                table_name="metadata",
                description=f"Error validating metadata consistency: {str(e)}",
                suggested_fix="Check metadata table structure and content",
            )
            report.add_issue(issue)

    # Méthode de validation de la présence des méta-données du jeu de résultats
    def _validate_dataset_metadata(self, report: ValidationReport) -> None:
        """Validate the presence and cardinality of the ``dataset_metadata`` table.

        The table describes the result set itself and must hold exactly one row per
        schema.

        Args:
            report: Validation report collecting the issues found.

        Examples:
            >>> auditor._validate_dataset_metadata(report)
        """
        try:
            # Vérification de l'existence de la table
            if not self._table_exists("dataset_metadata"):
                issue = ValidationIssue(
                    issue_type=IssueType.MISSING_METADATA,
                    severity=IssueSeverity.HIGH,
                    table_name="dataset_metadata",
                    description="Dataset metadata table is missing",
                    suggested_fix="Rebuild the schema so that dataset_metadata is"
                    " created, or create it manually",
                )
                report.add_issue(issue)
                return

            report.tables_validated.add("dataset_metadata")

            # Vérification de la cardinalité : une seule ligne par schéma
            row = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._qualified('dataset_metadata')}"
            ).fetchone()
            row_count = row[0] if row is not None else 0

            if row_count != 1:
                issue = ValidationIssue(
                    issue_type=IssueType.MISSING_METADATA,
                    severity=IssueSeverity.MEDIUM,
                    table_name="dataset_metadata",
                    description=f"Dataset metadata table holds {row_count} rows,"
                    f" exactly one is expected",
                    suggested_fix="Keep a single descriptive row per schema",
                    affected_rows=row_count,
                )
                report.add_issue(issue)

        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.MISSING_METADATA,
                severity=IssueSeverity.MEDIUM,
                table_name="dataset_metadata",
                description=f"Error validating dataset metadata: {str(e)}",
                suggested_fix="Check the dataset_metadata table structure",
            )
            report.add_issue(issue)

    # Méthode de validation de l'accord entre la table des méta-données et celle
    # des faits
    def _validate_metadata_fact_consistency(self, report: ValidationReport) -> None:
        """Validate that ``metadata`` and ``fact_table`` describe the same columns.

        The metadata table is the contract between the database and the interface:
        it must hold exactly one row per fact table column, no more and no less.

        Args:
            report: Validation report collecting the issues found.

        Examples:
            >>> auditor._validate_metadata_fact_consistency(report)
        """
        try:
            # Validation impossible sans les deux tables
            if not self._table_exists("fact_table") or not self._table_exists(
                "metadata"
            ):
                return

            # Ensembles de colonnes des deux côtés
            metadata_columns = set(self._get_metadata()["name"].to_list())
            fact_columns = set(self._get_fact_table_columns())

            # Colonnes décrites dans metadata mais absentes de la table des faits
            for col_name in sorted(metadata_columns - fact_columns):
                issue = ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    table_name="metadata",
                    column_name=col_name,
                    description=f"Column '{col_name}' is described in metadata but"
                    f" missing from fact_table",
                    suggested_fix=f"Drop the metadata row for '{col_name}' or add the"
                    f" column to fact_table",
                )
                report.add_issue(issue)

            # Colonnes présentes dans la table des faits mais non décrites
            for col_name in sorted(fact_columns - metadata_columns):
                issue = ValidationIssue(
                    issue_type=IssueType.MISSING_METADATA,
                    severity=IssueSeverity.HIGH,
                    table_name="fact_table",
                    column_name=col_name,
                    description=f"Column '{col_name}' exists in fact_table but has no"
                    f" metadata row",
                    suggested_fix=f"Add a metadata row describing '{col_name}'",
                )
                report.add_issue(issue)

        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.HIGH,
                table_name="fact_table",
                description=f"Error validating metadata/fact_table consistency:"
                f" {str(e)}",
                suggested_fix="Check metadata and fact_table structure",
            )
            report.add_issue(issue)

    # Méthode de validation de la cohérence des types de données entre la table des
    # méta-données et la table des faits
    def _validate_data_types_consistency(self, report: ValidationReport) -> None:
        """Validate data type consistency."""
        try:
            if not self._table_exists("fact_table"):
                return

            # Récupération des métadonnées et de la structure de fact_table
            metadata_df = self._get_metadata()
            fact_structure = self._get_table_structure("fact_table")
            # Parcours des colonnes de méta-données
            for metadata_row in metadata_df.iter_rows(named=True):
                # Extraction du nom de la colonne
                col_name = metadata_row["name"]
                # Extraction du type SQL attendu
                expected_sql_type = metadata_row["sql_type"]

                # Recherche du type actuel dans fact_table
                actual_sql_type = None
                for col_info in fact_structure:
                    if col_info[0] == col_name:
                        actual_sql_type = col_info[1]
                        break

                if actual_sql_type is None:
                    # Colonne manquante dans fact_table (déjà signalée dans
                    # _validate_metadata_fact_consistency)
                    continue

                # Comparaison des types (normalisation pour éviter les faux positifs)
                if not self._types_are_compatible(expected_sql_type, actual_sql_type):
                    issue = ValidationIssue(
                        issue_type=IssueType.TYPE_MISMATCH,
                        severity=IssueSeverity.MEDIUM,
                        table_name="fact_table",
                        column_name=col_name,
                        description=f"Type mismatch for column '{col_name}': expected"
                        f" {expected_sql_type}, got {actual_sql_type}",
                        suggested_fix="Update metadata or alter column type in"
                        " fact_table",
                        additional_info={
                            "expected_type": expected_sql_type,
                            "actual_type": actual_sql_type,
                        },
                    )
                    report.add_issue(issue)

        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=IssueSeverity.MEDIUM,
                table_name="fact_table",
                description=f"Error validating data types consistency: {str(e)}",
                suggested_fix="Check metadata and fact_table structure",
            )
            report.add_issue(issue)

    # Méthode de validation de la configuration de partitionnement de la table des faits
    def _validate_partition_configuration(self, report: ValidationReport) -> None:
        """Validate that fact_table has a Ducklake partition configuration.

        In Ducklake, Hive-style partitioning is the primary query optimisation
        mechanism (replaces DuckDB ART indexes). The absence of a partition key
        on fact_table is reported as a LOW-severity PERFORMANCE_ISSUE so that
        the operator knows to add ``PARTITION BY`` when recreating the table.

        Args:
            report: The ValidationReport to which issues are appended.

        Example:
            >>> auditor._validate_partition_configuration(report)
            >>> perf_issues = report.get_issues_by_type(IssueType.PERFORMANCE_ISSUE)
        """
        try:
            # Vérification de l'existence de la table des faits avant tout contrôle
            if not self._table_exists("fact_table"):
                return

            # Tentative de récupération de la clé de partition via duckdb_tables().
            # Filtrage par schéma : plusieurs schémas peuvent avoir une 'fact_table'.
            partition_key: str | None = None
            try:
                result = self.conn.execute(
                    "SELECT partition_key FROM duckdb_tables() "
                    "WHERE table_name = 'fact_table' AND schema_name = ?",
                    [self.schema],
                ).fetchone()
                if result is not None:
                    partition_key = result[0]
            except Exception:
                # Indisponibilité de duckdb_tables() sur cette connexion — utilisation
                # du fallback
                partition_key = None

            # Tentative de détection via SHOW CREATE TABLE en cas d'échec de la méthode
            # principale
            if not partition_key:
                try:
                    ddl_result = self.conn.execute(
                        f"SHOW CREATE TABLE {self._qualified('fact_table')}"
                    ).fetchone()
                    ddl_text: str = ddl_result[0] if ddl_result else ""
                    if "PARTITION BY" in ddl_text.upper():
                        # Partitionnement détecté dans le DDL — aucun problème
                        return
                except Exception:
                    pass

            # Signalement de l'absence de partitionnement comme problème de performance
            # mineur
            if not partition_key:
                issue = ValidationIssue(
                    issue_type=IssueType.PERFORMANCE_ISSUE,
                    severity=IssueSeverity.LOW,
                    table_name="fact_table",
                    description=(
                        "No Ducklake partition key configured on fact_table. "
                        "Hive-style partitioning is the primary optimisation mechanism"
                        " in Ducklake."
                    ),
                    suggested_fix=(
                        "Recreate fact_table with PARTITION BY on the most frequently"
                        " filtered column "
                        "(e.g. a date or category column) to improve query performance."
                    ),
                )
                report.add_issue(issue)

        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur de vérification
            issue = ValidationIssue(
                issue_type=IssueType.PERFORMANCE_ISSUE,
                severity=IssueSeverity.LOW,
                table_name="fact_table",
                description=f"Error validating partition configuration: {str(e)}",
                suggested_fix="Check fact_table definition and DuckDB version"
                " compatibility",
            )
            report.add_issue(issue)

    # Méthode de vérification de l'état de maintenance du catalogue Ducklake
    def _validate_ducklake_maintenance(self, report: ValidationReport) -> None:
        """Validate that Ducklake maintenance tasks are not overdue.

        Checks snapshot accumulation and data file fragmentation via Ducklake
        catalog functions. These queries are silently skipped on in-memory or
        non-Ducklake connections where catalog functions are unavailable.

        Args:
            report: The ValidationReport to which issues are appended.

        Example:
            >>> auditor._validate_ducklake_maintenance(report)
            >>> perf_issues = report.get_issues_by_type(IssueType.PERFORMANCE_ISSUE)
        """
        # Seuils déclenchant une recommandation de maintenance
        snapshot_threshold: int = 100
        data_files_threshold: int = 500

        # Vérification du nombre de snapshots accumulés dans le catalogue Ducklake
        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM ducklake_snapshots()"
            ).fetchone()
            snapshot_count: int = result[0] if result else 0
            if snapshot_count > snapshot_threshold:
                issue = ValidationIssue(
                    issue_type=IssueType.PERFORMANCE_ISSUE,
                    severity=IssueSeverity.LOW,
                    table_name="fact_table",
                    description=(
                        f"Ducklake snapshot history is large ({snapshot_count}"
                        f" snapshots). "
                        f"Threshold: {snapshot_threshold}."
                    ),
                    suggested_fix=(
                        "Run ducklake_expire_snapshots() to purge old snapshots "
                        "and then ducklake_cleanup_old_files() to reclaim storage."
                    ),
                    additional_info={
                        "snapshot_count": snapshot_count,
                        "threshold": snapshot_threshold,
                    },
                )
                report.add_issue(issue)
        except Exception:
            # Indisponibilité de ducklake_snapshots() sur connexion in-memory ou
            # non-Ducklake — ignoré silencieusement
            pass

        # Vérification de la fragmentation des fichiers de données Ducklake
        try:
            result = self.conn.execute(
                "SELECT COUNT(*) FROM ducklake_data_files()"
            ).fetchone()
            data_file_count: int = result[0] if result else 0
            if data_file_count > data_files_threshold:
                issue = ValidationIssue(
                    issue_type=IssueType.PERFORMANCE_ISSUE,
                    severity=IssueSeverity.LOW,
                    table_name="fact_table",
                    description=(
                        f"Ducklake data file count is high ({data_file_count} files). "
                        f"Excessive small files degrade scan performance. Threshold:"
                        f"{data_files_threshold}."
                    ),
                    suggested_fix=(
                        "Run ducklake_merge_adjacent_files() or"
                        " ducklake_rewrite_data_files() "
                        "to compact fragmented Parquet files."
                    ),
                    additional_info={
                        "data_file_count": data_file_count,
                        "threshold": data_files_threshold,
                    },
                )
                report.add_issue(issue)
        except Exception:
            # Indisponibilité de ducklake_data_files() sur connexion in-memory ou
            # non-Ducklake — ignoré silencieusement
            pass

    # Méthode de validation de la qualité des données
    def _validate_data_quality(self, report: ValidationReport) -> None:
        """Validate data quality."""
        try:
            # Vérification de l'existence de la table des faits
            if not self._table_exists("fact_table"):
                return

            # Comptage total des lignes
            _r = self.conn.execute(
                f"SELECT COUNT(*) FROM {self._qualified('fact_table')}"
            ).fetchone()
            total_rows = _r[0] if _r is not None else 0

            # Vérification que la table des faits n'est pas vide
            if total_rows == 0:
                issue = ValidationIssue(
                    issue_type=IssueType.DATA_INTEGRITY,
                    severity=IssueSeverity.MEDIUM,
                    table_name="fact_table",
                    description="Fact table is empty",
                    suggested_fix="Check if data has been loaded properly",
                )
                report.add_issue(issue)
                return

            # Vérification des colonnes avec beaucoup de valeurs nulles
            fact_columns = self._get_fact_table_columns()
            # Parcours des colonnes
            for col_name in fact_columns:
                # Création de la requête de comptage du nombre de valeurs nulles dans la
                # colonne
                null_count_query = (
                    f"SELECT COUNT(*) FROM {self._qualified('fact_table')} "
                    f"WHERE {quote_ident(col_name)} IS NULL"
                )
                # Exécution de la requête
                _rn = self.conn.execute(null_count_query).fetchone()
                null_count = _rn[0] if _rn is not None else 0
                # Calcul du pourcentage de valeurs nulles
                null_percentage = (null_count / total_rows) * 100

                # Ajout des messages au rapport
                if null_percentage > 50:  # Plus de 50% de valeurs nulles
                    issue = ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.MEDIUM,
                        table_name="fact_table",
                        column_name=col_name,
                        description=f"Column '{col_name}' has {null_percentage:.1f}%"
                        f"null values",
                        suggested_fix=f"Review data quality for column '{col_name}' or"
                        f" consider dropping it",
                        affected_rows=null_count,
                        additional_info={"null_percentage": null_percentage},
                    )
                    report.add_issue(issue)
                elif null_percentage == 100:  # Colonne entièrement nulle
                    issue = ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.HIGH,
                        table_name="fact_table",
                        column_name=col_name,
                        description=f"Column '{col_name}' contains only null values",
                        suggested_fix=f"Consider dropping column '{col_name}' or"
                        f" investigate data loading",
                        affected_rows=null_count,
                    )
                    report.add_issue(issue)

        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.DATA_INTEGRITY,
                severity=IssueSeverity.MEDIUM,
                table_name="fact_table",
                description=f"Error validating data quality: {str(e)}",
                suggested_fix="Check fact table structure and data",
            )
            report.add_issue(issue)

    # Méthode de vérification de l'unicité de la clé primaire de la table des faits
    def _validate_constraint_violations(self, report: ValidationReport) -> None:
        """Validate the applicative uniqueness of the fact table primary key.

        DuckLake supports no DDL constraint, so primary-key uniqueness is enforced
        applicatively at build and upsert time. This check verifies it actually
        holds on the stored data.

        Args:
            report: Validation report collecting the issues found.

        Examples:
            >>> auditor._validate_constraint_violations(report)
        """
        try:
            # Validation impossible sans table des faits ni clé primaire déclarée
            if not self._table_exists("fact_table"):
                return
            primary_keys = self._get_primary_key_columns()
            if not primary_keys:
                return

            # Comptage des combinaisons de clés apparaissant plus d'une fois
            key_columns = ", ".join(quote_ident(col) for col in primary_keys)
            duplicates_query = f"""
                SELECT COUNT(*) FROM (
                    SELECT {key_columns}
                    FROM {self._qualified("fact_table")}
                    GROUP BY {key_columns}
                    HAVING COUNT(*) > 1
                )
            """
            row = self.conn.execute(duplicates_query).fetchone()
            duplicate_count = row[0] if row is not None else 0

            # Ajout d'un message au rapport si des duplicats sont présents
            if duplicate_count > 0:
                issue = ValidationIssue(
                    issue_type=IssueType.CONSTRAINT_VIOLATION,
                    severity=IssueSeverity.HIGH,
                    table_name="fact_table",
                    column_name=", ".join(primary_keys),
                    description=(
                        f"Applicative uniqueness violation: {duplicate_count}"
                        f" duplicated primary key combinations found in fact_table"
                        f" (no DDL constraint enforced in Ducklake)"
                    ),
                    suggested_fix="Deduplicate the fact table on its primary keys",
                    affected_rows=duplicate_count,
                    additional_info={"primary_keys": primary_keys},
                )
                report.add_issue(issue)

        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.CONSTRAINT_VIOLATION,
                severity=IssueSeverity.MEDIUM,
                table_name="SYSTEM",
                description=f"Error validating constraint violations: {str(e)}",
                suggested_fix="Check table constraints and data integrity",
            )
            report.add_issue(issue)

    # Méthodes de validation des préconditions d'opération
    # Méthode de validation des conditions préalables à une opération d'insertion
    def _validate_insert_preconditions(
        self, report: ValidationReport, df: IntoDataFrame | None = None, **kwargs: Any
    ) -> None:
        """Validate insert preconditions."""
        # Vérifiation que le jeu de données est spécifié
        if df is None:
            issue = ValidationIssue(
                issue_type=IssueType.DATA_INTEGRITY,
                severity=IssueSeverity.CRITICAL,
                table_name="fact_table",
                description="DataFrame is None for insert operation",
                suggested_fix="Provide a valid DataFrame for insertion",
            )
            report.add_issue(issue)
            return

        # Conversion vers narwhals pour accéder à len() et .columns de façon sûre
        df_nw = nw.from_native(df, eager_only=True)
        # Vérification que le jeu de données est non vide
        if len(df_nw) == 0:
            issue = ValidationIssue(
                issue_type=IssueType.DATA_INTEGRITY,
                severity=IssueSeverity.MEDIUM,
                table_name="fact_table",
                description="DataFrame is empty for insert operation",
                suggested_fix="Provide DataFrame with data for insertion",
            )
            report.add_issue(issue)

        # Vérification des noms de colonnes valides
        invalid_columns = [
            col
            for col in df_nw.columns
            if not col.replace("_", "").replace(" ", "").isalnum()
        ]
        if invalid_columns:
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.HIGH,
                table_name="fact_table",
                description=f"Invalid column names detected: {invalid_columns}",
                suggested_fix="Use valid column names (alphanumeric and underscores"
                " only)",
            )
            report.add_issue(issue)

    # Méthode de validation des conditions préalables à une mise à jour de la base de
    # données
    def _validate_update_preconditions(
        self, report: ValidationReport, df: IntoDataFrame | None = None, **kwargs: Any
    ) -> None:
        """Validate update preconditions.

        Validates the preconditions for an update operation by using
        the primary keys defined in the metadata instead of the merge_keys
        passed as parameters.
        """
        # Vérification des conditions d'insertions
        self._validate_insert_preconditions(report, df, **kwargs)

        # Récupération des clés primaires depuis les métadonnées
        primary_keys = self._get_primary_key_columns()

        # Si pas de clés primaires définies, pas de validation supplémentaire nécessaire
        # (la mise à jour se fera par INSERT simple)
        if not primary_keys:
            return

        # Vérification que les clés primaires sont présentes dans le DataFrame
        if df is not None:
            df_nw = nw.from_native(df, eager_only=True)
            missing_keys = [key for key in primary_keys if key not in df_nw.columns]
            if missing_keys:
                issue = ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.CRITICAL,
                    table_name="fact_table",
                    description=f"Primary keys not found in DataFrame: {missing_keys}",
                    suggested_fix="Ensure all primary keys are present in the"
                    " DataFrame",
                )
                report.add_issue(issue)
                return

            # Vérification des doublons parmi les valeurs de clés primaires dans le
            # DataFrame.
            duplicate_count = (
                nw.from_native(df, eager_only=True)
                .select(primary_keys)
                .is_duplicated()
                .sum()
            )

            if duplicate_count > 0:
                issue = ValidationIssue(
                    issue_type=IssueType.DATA_INTEGRITY,
                    severity=IssueSeverity.HIGH,
                    table_name="fact_table",
                    description=f"Found {duplicate_count} duplicate primary key values"
                    f" in DataFrame",
                    suggested_fix="Remove duplicate entries based on primary key"
                    " columns",
                    affected_rows=int(duplicate_count),
                )
                report.add_issue(issue)

    # Méthode de validation des conditions préalable à la suppression de données
    def _validate_delete_preconditions(
        self, report: ValidationReport, filters: Any = None, **kwargs: Any
    ) -> None:
        """Validate delete preconditions."""
        # Vérification de l'existence du filtre de suppression des données
        if filters is None:
            issue = ValidationIssue(
                issue_type=IssueType.DATA_INTEGRITY,
                severity=IssueSeverity.CRITICAL,
                table_name="fact_table",
                description="No filters provided for delete operation",
                suggested_fix="Provide valid filters for delete operation to avoid"
                " deleting all data",
            )
            report.add_issue(issue)

    # méthode de validation des conditions préalable à un changement de schéma dans la
    # base de données
    def _validate_schema_change_preconditions(
        self, report: ValidationReport, **kwargs: Any
    ) -> None:
        """Validate schema change preconditions."""
        # Vérification de l'existence des tables avant modification
        if not self._table_exists("fact_table"):
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.HIGH,
                table_name="fact_table",
                description="Fact table does not exist for schema change operation",
                suggested_fix="Create fact table before attempting schema changes",
            )
            report.add_issue(issue)

    # Méthodes utilitaires privées
    # Méthode auxiliaire d'obtention des tables de la base de données
    def _get_existing_tables(self) -> list[str]:
        """Get the list of existing tables in the audited schema."""
        try:
            # Filtrage par schéma : SHOW TABLES ne renvoie que le schéma actif de la
            # connexion, qui n'est pas nécessairement le schéma audité.
            result = self.conn.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = ?",
                [self.schema],
            ).fetchall()
            return [row[0] for row in result]
        except Exception:
            return []

    # Méthode auxiliaire d'extraction des colonnes de la table des faits
    def _get_fact_table_columns(self) -> list[str]:
        """Get fact table columns."""
        try:
            result = self.conn.execute(
                f"DESCRIBE {self._qualified('fact_table')}"
            ).fetchall()
            return [row[0] for row in result]
        except Exception:
            return []

    # Méthode auxiliaire d'extraction des coonnes d'une table spécifique
    def _get_table_columns(self, table_name: str) -> list[str]:
        """Get columns from a specific (bare-named) table in the audited schema."""
        try:
            result = self.conn.execute(
                f"DESCRIBE {self._qualified(table_name)}"
            ).fetchall()
            return [row[0] for row in result]
        except Exception:
            return []

    # Méthode auxiliaire de description d'une table
    def _get_table_structure(self, table_name: str) -> list[tuple[Any, ...]]:
        """Get the complete structure of a (bare-named) table in the audited schema."""
        try:
            return self.conn.execute(
                f"DESCRIBE {self._qualified(table_name)}"
            ).fetchall()
        except Exception:
            return []

    # Méthode auxiliaire d'extraction des métadonnées
    def _get_metadata(self) -> nw.DataFrame[Any]:
        """Get metadata table content.

        Returns:
            nw.DataFrame: The metadata rows (pyarrow backend), or an empty frame
            when the table cannot be read.
        """
        try:
            metadata: nw.DataFrame[Any] = nw.from_native(
                self.conn.execute(
                    f"SELECT * FROM {self._qualified('metadata')}"
                ).to_arrow_table(),
                eager_only=True,
            )
            return metadata
        except Exception:
            return nw.from_dict({}, backend="pyarrow")

    # Méthode auxiliaire de vérification de l'existence d'une colonne dans la table des
    # faits
    def _column_exists_in_fact_table(self, column_name: str) -> bool:
        """Check if a column exists in fact_table."""
        fact_columns = self._get_fact_table_columns()
        return column_name in fact_columns

    # Méthode auxiliaire d'extraction des colonnes marquées comme clés primaires
    def _get_primary_key_columns(self) -> list[str]:
        """Get all column names that are marked as primary keys in metadata.

        Récupère la liste des colonnes définies comme clés primaires dans la table
        des métadonnées. Cette méthode est nécessaire car DatabaseAuditor n'hérite
        pas de BaseSchemaManager.

        Returns:
            List of primary key column names. Empty list if no primary keys defined.
        """
        try:
            result = self.conn.execute(
                f"SELECT name FROM {self._qualified('metadata')} "
                "WHERE is_primary_key = true"
            ).fetchall()
            return [row[0] for row in result]
        except Exception:
            return []

    # Méthode auxiliaire de vérification de la compatibilité des types entre eux
    def _types_are_compatible(self, expected_type: str, actual_type: str) -> bool:
        """Check if two SQL types are compatible."""
        # Normalisation des types pour comparaison
        expected_normalized = expected_type.upper().strip()
        actual_normalized = actual_type.upper().strip()

        # Mapping des types équivalents
        type_equivalents = {
            "VARCHAR": ["TEXT", "STRING", "CHAR"],
            "INTEGER": ["INT", "INT64", "BIGINT"],
            "DOUBLE": ["FLOAT", "FLOAT64", "REAL"],
            "BOOLEAN": ["BOOL"],
        }

        if expected_normalized == actual_normalized:
            return True

        # Vérification des équivalences
        for base_type, equivalents in type_equivalents.items():
            if expected_normalized == base_type and actual_normalized in equivalents:
                return True
            if actual_normalized == base_type and expected_normalized in equivalents:
                return True
            if expected_normalized in equivalents and actual_normalized in equivalents:
                return True

        return False

    # Méthodes publiques pour obtenir des rapports spécialisés
    # Méthode de vérification rapide de la base de données
    def get_quick_health_check(self) -> dict[str, Any]:
        """
        Perform a quick health check of the database.

        Returns:
            Dictionary with basic health metrics

        Example:
            >>> health = auditor.get_quick_health_check()
            >>> if health['status'] == 'healthy':
            ...     print("Database is healthy")
        """
        try:
            health_info: dict[str, Any] = {
                "status": "unknown",
                "timestamp": time.time(),
                "tables_count": 0,
                "fact_table_rows": 0,
                "metadata_entries": 0,
                "has_dataset_metadata": False,
                "critical_issues": 0,
            }

            # Comptage des tables
            tables = self._get_existing_tables()
            health_info["tables_count"] = len(tables)

            # Comptage des lignes de fact_table
            if "fact_table" in tables:
                result = self.conn.execute(
                    f"SELECT COUNT(*) FROM {self._qualified('fact_table')}"
                ).fetchone()
                health_info["fact_table_rows"] = result[0] if result else 0

            # Présence des méta-données du jeu de résultats
            health_info["has_dataset_metadata"] = "dataset_metadata" in tables

            # Comptage des entrées de métadonnées
            if "metadata" in tables:
                result = self.conn.execute(
                    f"SELECT COUNT(*) FROM {self._qualified('metadata')}"
                ).fetchone()
                health_info["metadata_entries"] = result[0] if result else 0

            # Validation rapide pour les issues critiques
            quick_report = self.validate_database(ValidationLevel.BASIC)
            health_info["critical_issues"] = quick_report.get_critical_issues_count()

            # Détermination du statut global
            if health_info["critical_issues"] > 0:
                health_info["status"] = "critical"
            elif len(quick_report.issues) > 0:
                health_info["status"] = "warning"
            elif (
                health_info["fact_table_rows"] > 0
                and health_info["metadata_entries"] > 0
            ):
                health_info["status"] = "healthy"
            else:
                health_info["status"] = "empty"

            return health_info

        except Exception as e:
            # Logging
            self.logger.error(f"Error during quick health check: {e}")
            return {"status": "error", "timestamp": time.time(), "error": str(e)}
