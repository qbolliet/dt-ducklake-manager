# Importation des modules
# Modules de base
import os
import pandas as pd
import narwhals as nw
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
# DuckDB
import duckdb

# Import des utilitaires
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


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
    ORPHANED_REFERENCE = "orphaned_reference"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_METADATA = "missing_metadata"
    INVALID_DIMENSION = "invalid_dimension"
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
    column_name: Optional[str] = None
    description: str = ""
    suggested_fix: str = ""
    affected_rows: Optional[int] = None
    detected_at: float = field(default_factory=time.time)
    additional_info: dict = field(default_factory=dict)


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
    end_time: Optional[float] = None
    issues: List[ValidationIssue] = field(default_factory=list)
    tables_validated: Set[str] = field(default_factory=set)
    validation_summary: dict = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    
    # Méthode d'ajout d'un problème
    def add_issue(self, issue: ValidationIssue) -> None:
        """Add an issue to the report."""
        self.issues.append(issue)
    
    # Méthode d'extraction des problèmes par sévérité
    def get_issues_by_severity(self, severity: IssueSeverity) -> List[ValidationIssue]:
        """Get issues by severity level."""
        return [issue for issue in self.issues if issue.severity == severity]
    
    # Méthode d'extraction des problèmes par type
    def get_issues_by_type(self, issue_type: IssueType) -> List[ValidationIssue]:
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
            'total_issues': len(self.issues),
            'critical_issues': len(self.get_issues_by_severity(IssueSeverity.CRITICAL)),
            'high_issues': len(self.get_issues_by_severity(IssueSeverity.HIGH)),
            'medium_issues': len(self.get_issues_by_severity(IssueSeverity.MEDIUM)),
            'low_issues': len(self.get_issues_by_severity(IssueSeverity.LOW)),
            'tables_validated': len(self.tables_validated),
            'validation_duration': self.end_time - self.start_time if self.end_time else 0
        }
        
        # Génération des recommandations
        self._generate_recommendations()
    
    # Méthode auxiliaire de génération de recommandations
    def _generate_recommendations(self) -> None:
        """Generate recommendations based on issues found."""
        # Identification des erreurs critiques
        critical_count = self.get_critical_issues_count()
        if critical_count > 0:
            self.recommendations.append(f"Adress immediately the {critical_count} critical issues detected")
        
        # Identification des erreurs de schéma
        schema_issues = self.get_issues_by_type(IssueType.SCHEMA_INCONSISTENCY)
        if schema_issues:
            self.recommendations.append("Review the consistency of the database schema")
        
        # Identification des références orphelines
        orphaned_issues = self.get_issues_by_type(IssueType.ORPHANED_REFERENCE)
        if orphaned_issues:
            self.recommendations.append("Clean up orphan references in dimension tables")
        
        # Identification des problèmes de performance
        performance_issues = self.get_issues_by_type(IssueType.PERFORMANCE_ISSUE)
        if performance_issues:
            self.recommendations.append("Optimize indexes and table structure to improve performance")


# Classe d'audit de la base de données
class DatabaseAuditor:
    """
    Provides comprehensive database validation and state checking capabilities.
    
    Validates schema consistency, data integrity, referential integrity between
    fact and dimension tables, and identifies potential performance issues.
    
    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection
        categorical_threshold (int): Threshold for categorical determination
        logger: Logger instance for audit tracking
    """
    # Initialisation
    def __init__(self,
                 connection: Optional[duckdb.DuckDBPyConnection] = None,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None):
        """
        Initialize the database auditor.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            categorical_threshold: Threshold for determining categorical variables.
            log_filename: Path to log file.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> auditor = DatabaseAuditor(conn)
            >>> report = auditor.validate_database(ValidationLevel.COMPREHENSIVE)
        """
        # Initialisation de la connexion DuckLake.
        self.conn = connection if connection is not None else duckdb.connect(':memory:')
        
        # Seuil pour déterminer si une variable est catégorielle
        self.categorical_threshold = categorical_threshold
        
        # Initialisation du logger pour traçabilité des audits
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/database_auditor.log")
        self.logger = _init_logger(filename=log_filename)
    
    # Méthodes principales de validation
    # Méthode de validation de la base de données
    def validate_database(self, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> ValidationReport:
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
        self.logger.info(f"Starting database validation at level: {validation_level.value}")
        
        try:
            # Validation de base (toujours effectuée)
            # Validation du schéma
            self._validate_schema_existence(report)
            # Validation de la consistence des méta-données
            self._validate_metadata_consistency(report)
            
            # Validation standard
            if validation_level in [ValidationLevel.STANDARD, ValidationLevel.COMPREHENSIVE]:
                # Validation des tables de dimension
                self._validate_fact_dimension_consistency(report)
                # Validation de la consistance des types des données
                self._validate_data_types_consistency(report)
                # Validation des références orphelines
                self._validate_orphaned_references(report)
            
            # Validation complète
            if validation_level == ValidationLevel.COMPREHENSIVE:
                # Validation des seuils de variables catégorielles
                self._validate_categorical_thresholds(report)
                # Validation de l'efficacité de l'index
                self._validate_index_efficiency(report)
                # Validation de la qualité des données
                self._validate_data_quality(report)
                # Validation de la violation des contraintes
                self._validate_constraint_violations(report)
            
            # Finalisation du rapport
            report.finalize()
            
            # Logging
            self.logger.info(f"Database validation completed. Found {len(report.issues)} issues.")
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
                suggested_fix="Check database connection and schema structure"
            )
            report.add_issue(error_issue)
            # Finalisation du rapport
            report.finalize()
            
            return report
    
    # Validation des prérequis d'une opération
    def validate_operation_preconditions(self, operation_type: str, **kwargs) -> ValidationReport:
        """
        Validate preconditions before executing specific operations.
        
        Args:
            operation_type: Type of operation ('insert', 'update', 'delete', 'schema_change')
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
            if operation_type == 'insert':
                self._validate_insert_preconditions(report, **kwargs)
            elif operation_type == 'update':
                self._validate_update_preconditions(report, **kwargs)
            elif operation_type == 'delete':
                self._validate_delete_preconditions(report, **kwargs)
            elif operation_type == 'schema_change':
                self._validate_schema_change_preconditions(report, **kwargs)
            else:
                issue = ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.MEDIUM,
                    table_name="OPERATION_VALIDATION",
                    description=f"Unknown operation type: {operation_type}",
                    suggested_fix="Use a supported operation type"
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
                suggested_fix="Check operation parameters and database state"
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
            essential_tables = ['metadata']
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
                        suggested_fix=f"Create the '{table}' table with proper structure"
                    )
                    report.add_issue(issue)
                else:
                    report.tables_validated.add(table)
            
            # Vérification de l'existence de la table des faits
            if 'fact_table' not in existing_tables:
                issue = ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    table_name='fact_table',
                    description="Fact table is missing",
                    suggested_fix="Create the fact table or check if database is properly initialized"
                )
                report.add_issue(issue)
            else:
                report.tables_validated.add('fact_table')
            
        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.CRITICAL,
                table_name="SCHEMA_VALIDATION",
                description=f"Error validating schema existence: {str(e)}",
                suggested_fix="Check database connection and permissions"
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
                    table_name='metadata',
                    description="Metadata table is empty",
                    suggested_fix="Populate metadata table with column information"
                )
                report.add_issue(issue)
                return
            
            # Vérification des colonnes requises dans metadata
            required_columns = ['name', 'label', 'python_type', 'sql_type', 'is_categorical']
            missing_columns = [col for col in required_columns if col not in metadata_df.columns]
            
            if missing_columns:
                issue = ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    table_name='metadata',
                    description=f"Missing required columns in metadata: {missing_columns}",
                    suggested_fix="Add missing columns to metadata table"
                )
                report.add_issue(issue)
            
            # Vérification des valeurs nulles dans les colonnes critiques
            for col in ['name', 'python_type', 'sql_type']:
                if col in metadata_df.columns:
                    null_count = metadata_df[col].isna().sum()
                    if null_count > 0:
                        issue = ValidationIssue(
                            issue_type=IssueType.DATA_INTEGRITY,
                            severity=IssueSeverity.MEDIUM,
                            table_name='metadata',
                            column_name=col,
                            description=f"Found {null_count} null values in critical metadata column '{col}'",
                            suggested_fix=f"Update null values in metadata.{col}",
                            affected_rows=null_count
                        )
                        report.add_issue(issue)
            
            # Vérification des doublons dans les noms de colonnes (il s'agit d ela clé primaire de la base de données)
            if 'name' in metadata_df.columns:
                duplicate_names = metadata_df[metadata_df['name'].duplicated()]['name'].tolist()
                if duplicate_names:
                    issue = ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.HIGH,
                        table_name='metadata',
                        column_name='name',
                        description=f"Duplicate column names in metadata: {duplicate_names}",
                        suggested_fix="Remove or rename duplicate entries in metadata",
                        affected_rows=len(duplicate_names)
                    )
                    report.add_issue(issue)
            
        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.HIGH,
                table_name='metadata',
                description=f"Error validating metadata consistency: {str(e)}",
                suggested_fix="Check metadata table structure and content"
            )
            report.add_issue(issue)
    
    # Méthode de validation de la cohérence entre la table de faits et les tables de dimension
    def _validate_fact_dimension_consistency(self, report: ValidationReport) -> None:
        """Validate consistency between fact table and dimension tables."""
        try:
            if not self._table_exists('fact_table'):
                return  # Déjà signalé dans _validate_schema_existence
            
            # Récupération des métadonnées
            metadata_df = self._get_metadata()
            
            if len(metadata_df) == 0:
                return  # Déjà signalé dans _validate_metadata_consistency
            
            # Récupération des colonnes de la fact table
            fact_columns = set(self._get_fact_table_columns())
            
            # Vérification des colonnes catégorielles
            categorical_columns = metadata_df[metadata_df['is_categorical'] == True]['name'].tolist()
            
            # Parcours des colonnes catégorielles
            for col_name in categorical_columns:
                # Nom de la table de dimension
                dim_table_name = f"dim_{col_name}"
                
                # Vérification de l'existence de la table de dimension
                if not self._table_exists(dim_table_name):
                    issue = ValidationIssue(
                        issue_type=IssueType.INVALID_DIMENSION,
                        severity=IssueSeverity.HIGH,
                        table_name=dim_table_name,
                        column_name=col_name,
                        description=f"Dimension table '{dim_table_name}' missing for categorical column '{col_name}'",
                        suggested_fix=f"Create dimension table for '{col_name}' or update metadata"
                    )
                    report.add_issue(issue)
                    continue
                
                report.tables_validated.add(dim_table_name)
                
                # Vérification de la structure de la table de dimension
                dim_columns = set(self._get_table_columns(dim_table_name))
                required_dim_columns = {'value', 'label'}
                
                missing_dim_columns = required_dim_columns - dim_columns
                if missing_dim_columns:
                    issue = ValidationIssue(
                        issue_type=IssueType.SCHEMA_INCONSISTENCY,
                        severity=IssueSeverity.HIGH,
                        table_name=dim_table_name,
                        description=f"Missing required columns in dimension table: {missing_dim_columns}",
                        suggested_fix=f"Add missing columns to {dim_table_name}"
                    )
                    report.add_issue(issue)
                
                # Vérification de la cohérence référentielle si la colonne existe dans fact_table
                if col_name in fact_columns:
                    self._validate_referential_integrity(report, col_name, dim_table_name)
            
            # Vérification des colonnes dans fact_table qui ne sont pas dans metadata
            metadata_columns = set(metadata_df['name'].tolist())
            missing_metadata = fact_columns - metadata_columns
            
            if missing_metadata:
                issue = ValidationIssue(
                    issue_type=IssueType.MISSING_METADATA,
                    severity=IssueSeverity.MEDIUM,
                    table_name='fact_table',
                    description=f"Columns in fact_table missing from metadata: {list(missing_metadata)}",
                    suggested_fix="Add missing columns to metadata table"
                )
                report.add_issue(issue)

            # Vérification des colonnes dans metadata qui ne sont pas dans fact_table (métadonnées orphelines)
            orphaned_metadata = metadata_columns - fact_columns

            if orphaned_metadata:
                issue = ValidationIssue(
                    issue_type=IssueType.MISSING_METADATA,
                    severity=IssueSeverity.MEDIUM,
                    table_name='metadata',
                    description=f"Columns in metadata not found in fact_table: {list(orphaned_metadata)}",
                    suggested_fix="Remove orphaned metadata entries or add missing columns to fact_table"
                )
                report.add_issue(issue)

        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.HIGH,
                table_name='fact_table',
                description=f"Error validating fact-dimension consistency: {str(e)}",
                suggested_fix="Check fact table and dimension tables structure"
            )
            report.add_issue(issue)
    
    # Méthode de validation de l'intégrité référentielle entre la table des faits et la table de dimension
    def _validate_referential_integrity(self, report: ValidationReport, col_name: str, dim_table_name: str) -> None:
        """Validate referential integrity between fact table and dimension."""
        try:
            # Vérification des valeurs orphelines dans fact_table (ie qui ne sont pas référencées dans la table de dimension)
            orphaned_query = f"""
                SELECT COUNT(DISTINCT f.{col_name}) as orphaned_count
                FROM fact_table f
                LEFT JOIN {dim_table_name} d ON f.{col_name} = d.value
                WHERE f.{col_name} IS NOT NULL AND d.value IS NULL
            """
            
            result = self.conn.execute(orphaned_query).fetchone()
            orphaned_count = result[0] if result else 0
            
            if orphaned_count > 0:
                issue = ValidationIssue(
                    issue_type=IssueType.ORPHANED_REFERENCE,
                    severity=IssueSeverity.MEDIUM,
                    table_name='fact_table',
                    column_name=col_name,
                    description=f"Found {orphaned_count} orphaned references in fact_table.{col_name}",
                    suggested_fix=f"Update dimension table {dim_table_name} or clean orphaned references",
                    affected_rows=orphaned_count
                )
                report.add_issue(issue)
            
            # Vérification des valeurs inutilisées dans dimension
            unused_query = f"""
                SELECT COUNT(*) as unused_count
                FROM {dim_table_name} d
                LEFT JOIN fact_table f ON d.value = f.{col_name}
                WHERE f.{col_name} IS NULL
            """
            
            result = self.conn.execute(unused_query).fetchone()
            unused_count = result[0] if result else 0
            
            if unused_count > 0:
                issue = ValidationIssue(
                    issue_type=IssueType.ORPHANED_REFERENCE,
                    severity=IssueSeverity.LOW,
                    table_name=dim_table_name,
                    description=f"Found {unused_count} unused dimension values in {dim_table_name}",
                    suggested_fix=f"Clean unused values from {dim_table_name}",
                    affected_rows=unused_count
                )
                report.add_issue(issue)
            
        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.DATA_INTEGRITY,
                severity=IssueSeverity.MEDIUM,
                table_name=dim_table_name,
                column_name=col_name,
                description=f"Error validating referential integrity for {col_name}: {str(e)}",
                suggested_fix="Check column and table structure"
            )
            report.add_issue(issue)
    
    # Méthode de validation de la cohérence des types de données entre la table des méta-données et la table des faits
    def _validate_data_types_consistency(self, report: ValidationReport) -> None:
        """Validate data type consistency."""
        try:
            if not self._table_exists('fact_table'):
                return
            
            # Récupération des métadonnées et de la structure de fact_table
            metadata_df = self._get_metadata()
            fact_structure = self._get_table_structure('fact_table')
            # Parcours des colonnes de méta-données
            for _, metadata_row in metadata_df.iterrows():
                # Extraction du nom de la colonne
                col_name = metadata_row['name']
                # Extraction du type SQL attendu
                expected_sql_type = metadata_row['sql_type']
                
                # Recherche du type actuel dans fact_table
                actual_sql_type = None
                for col_info in fact_structure:
                    if col_info[0] == col_name:
                        actual_sql_type = col_info[1]
                        break
                
                if actual_sql_type is None:
                    # Colonne manquante dans fact_table (déjà signalé dans _validate_fact_dimension_consistency)
                    continue
                
                # Comparaison des types (normalisation pour éviter les faux positifs)
                if not self._types_are_compatible(expected_sql_type, actual_sql_type):
                    issue = ValidationIssue(
                        issue_type=IssueType.TYPE_MISMATCH,
                        severity=IssueSeverity.MEDIUM,
                        table_name='fact_table',
                        column_name=col_name,
                        description=f"Type mismatch for column '{col_name}': expected {expected_sql_type}, got {actual_sql_type}",
                        suggested_fix="Update metadata or alter column type in fact_table",
                        additional_info={
                            'expected_type': expected_sql_type,
                            'actual_type': actual_sql_type
                        }
                    )
                    report.add_issue(issue)
            
        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.TYPE_MISMATCH,
                severity=IssueSeverity.MEDIUM,
                table_name='fact_table',
                description=f"Error validating data types consistency: {str(e)}",
                suggested_fix="Check metadata and fact_table structure"
            )
            report.add_issue(issue)
    
    # Méthode de validation des variables catégorielles par rapport au seuil et à leur nombre de modalités
    def _validate_categorical_thresholds(self, report: ValidationReport) -> None:
        """Validate categorical thresholds."""
        try:
            # Extraction de méta-données
            metadata_df = self._get_metadata()
            
            # Vérification des colonnes marquées comme catégorielles
            categorical_columns = metadata_df[metadata_df['is_categorical'] == True]['name'].tolist()
            
            # Parcours des colonnes
            for col_name in categorical_columns:
                if not self._column_exists_in_fact_table(col_name):
                    continue # Colonne manquante dans fact_table (déjà signalé dans A AJOUTER)
                
                # Comptage des valeurs uniques
                unique_count_query = f"SELECT COUNT(DISTINCT {col_name}) FROM fact_table WHERE {col_name} IS NOT NULL"
                result = self.conn.execute(unique_count_query).fetchone()
                unique_count = result[0] if result else 0
                
                # Si le nombre de modalités est supérieur au seuil, renvoie un problème
                if unique_count > self.categorical_threshold:
                    issue = ValidationIssue(
                        issue_type=IssueType.INVALID_DIMENSION,
                        severity=IssueSeverity.MEDIUM,
                        table_name='fact_table',
                        column_name=col_name,
                        description=f"Column '{col_name}' marked as categorical but has {unique_count} unique values (threshold: {self.categorical_threshold})",
                        suggested_fix=f"Consider converting '{col_name}' to non-categorical or increase threshold",
                        additional_info={'unique_count': unique_count, 'threshold': self.categorical_threshold}
                    )
                    report.add_issue(issue)
            
            # Vérification des colonnes non-catégorielles qui pourraient l'être
            non_categorical_columns = metadata_df[
                (metadata_df['is_categorical'] == False) & 
                (metadata_df['python_type'] == 'object')
            ]['name'].tolist()
            
            # Parcours des colonnes
            for col_name in non_categorical_columns:
                if not self._column_exists_in_fact_table(col_name):
                    continue
                
                # Comptage des valeurs uniques
                unique_count_query = f"SELECT COUNT(DISTINCT {col_name}) FROM fact_table WHERE {col_name} IS NOT NULL"
                result = self.conn.execute(unique_count_query).fetchone()
                unique_count = result[0] if result else 0
                
                if unique_count <= self.categorical_threshold:
                    issue = ValidationIssue(
                        issue_type=IssueType.PERFORMANCE_ISSUE,
                        severity=IssueSeverity.LOW,
                        table_name='fact_table',
                        column_name=col_name,
                        description=f"Column '{col_name}' could be categorical (only {unique_count} unique values)",
                        suggested_fix=f"Consider converting '{col_name}' to categorical for better performance",
                        additional_info={'unique_count': unique_count, 'threshold': self.categorical_threshold}
                    )
                    report.add_issue(issue)
            
        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.PERFORMANCE_ISSUE,
                severity=IssueSeverity.LOW,
                table_name='metadata',
                description=f"Error validating categorical thresholds: {str(e)}",
                suggested_fix="Check categorical column configuration"
            )
            report.add_issue(issue)
    
    # Méthode de détection des références orphelines
    def _validate_orphaned_references(self, report: ValidationReport) -> None:
        """Validate and detect orphaned references."""
        try:
            # Recherche des tables de dimension orphelines
            # Extraction des tables de dimension existantes
            existing_tables = self._get_existing_tables()
            dimension_tables = [table for table in existing_tables if table.startswith('dim_')]
            # Extraction des métadonnées
            metadata_df = self._get_metadata()
            expected_dimensions = set(f"dim_{row['name']}" for _, row in metadata_df.iterrows() if row['is_categorical'])
            
            # Tables de dimension sans métadonnées correspondantes
            orphaned_dimension_tables = set(dimension_tables) - expected_dimensions
            
            for table_name in orphaned_dimension_tables:
                issue = ValidationIssue(
                    issue_type=IssueType.ORPHANED_REFERENCE,
                    severity=IssueSeverity.MEDIUM,
                    table_name=table_name,
                    description=f"Orphaned dimension table '{table_name}' found",
                    suggested_fix=f"Remove '{table_name}' or add corresponding metadata entry"
                )
                report.add_issue(issue)
            
            # Vérification des entrées orphelines dans les dimensions existantes
            for _, metadata_row in metadata_df.iterrows():
                if metadata_row['is_categorical']:
                    # Nom de la table de dimension
                    col_name = metadata_row['name']
                    dim_table_name = f"dim_{col_name}"
                    # On vérifie les entrées si la table de dimension existe et que la colonne existe dans la table des faits
                    # (On vérifie plus haut que toutes les variables catégroeilles dans les métadonnées ont une table de dimension et que chaque colonne des métadonnées existe dans la table des faits)
                    if self._table_exists(dim_table_name) and self._column_exists_in_fact_table(col_name):
                        self._validate_referential_integrity(report, col_name, dim_table_name)
            
        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.ORPHANED_REFERENCE,
                severity=IssueSeverity.MEDIUM,
                table_name='SYSTEM',
                description=f"Error validating orphaned references: {str(e)}",
                suggested_fix="Check dimension tables and metadata consistency"
            )
            report.add_issue(issue)
    
    # Méthode de validation de l'efficacité des index
    def _validate_index_efficiency(self, report: ValidationReport) -> None:
        """Validate index effectiveness."""
        try:
            # Récupération des index existants
            indexes = self._get_existing_indexes()
            
            # Ajout d'un message au rapport si aucun index n'est attaché à la base de données
            if len(indexes) == 0:
                issue = ValidationIssue(
                    issue_type=IssueType.PERFORMANCE_ISSUE,
                    severity=IssueSeverity.LOW,
                    table_name='fact_table',
                    description="No indexes found on fact_table",
                    suggested_fix="Consider adding indexes on frequently queried columns"
                )
                report.add_issue(issue)
                return
            
            # Vérification des index sur des colonnes inexistantes
            fact_columns = set(self._get_fact_table_columns())
            
            # Parcours des index
            for index_info in indexes:
                # Nom de l'index
                index_name = index_info.get('index_name', '')
                # Expressions
                expressions = index_info.get('expressions', '')
                
                # Analyse simplifiée pour détecter les colonnes référencées
                referenced_columns = []
                for col in fact_columns:
                    if col in expressions or f"fact_table.{col}" in expressions:
                        referenced_columns.append(col)
                
                # Si aucune colonne valide n'est trouvée mais que l'index référence fact_table
                if not referenced_columns and "fact_table" in expressions:
                    issue = ValidationIssue(
                        issue_type=IssueType.ORPHANED_REFERENCE,
                        severity=IssueSeverity.LOW,
                        table_name='fact_table',
                        description=f"Potentially orphaned index '{index_name}'",
                        suggested_fix=f"Review and possibly drop index '{index_name}'",
                        additional_info={'index_expression': expressions}
                    )
                    report.add_issue(issue)
            
        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.PERFORMANCE_ISSUE,
                severity=IssueSeverity.LOW,
                table_name='fact_table',
                description=f"Error validating index efficiency: {str(e)}",
                suggested_fix="Check index configuration and structure"
            )
            report.add_issue(issue)
    
    # Méthode de validation de la qualité des données
    def _validate_data_quality(self, report: ValidationReport) -> None:
        """Validate data quality."""
        try:
            # Vérification de l'existence de la table des faits
            if not self._table_exists('fact_table'):
                return
            
            # Comptage total des lignes
            total_rows = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            # Vérification que la table des faits n'est pas vide
            if total_rows == 0:
                issue = ValidationIssue(
                    issue_type=IssueType.DATA_INTEGRITY,
                    severity=IssueSeverity.MEDIUM,
                    table_name='fact_table',
                    description="Fact table is empty",
                    suggested_fix="Check if data has been loaded properly"
                )
                report.add_issue(issue)
                return
            
            # Vérification des colonnes avec beaucoup de valeurs nulles
            fact_columns = self._get_fact_table_columns()
            # Parcours des colonnes
            for col_name in fact_columns:
                # Création de la requête de comptage du nombre de valeurs nulles dans la colonne
                null_count_query = f"SELECT COUNT(*) FROM fact_table WHERE {col_name} IS NULL"
                # Exécution de la requête
                null_count = self.conn.execute(null_count_query).fetchone()[0]
                # Calcul du pourcentage de valeurs nulles
                null_percentage = (null_count / total_rows) * 100
                
                # Ajout des messages au rapport
                if null_percentage > 50:  # Plus de 50% de valeurs nulles
                    issue = ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.MEDIUM,
                        table_name='fact_table',
                        column_name=col_name,
                        description=f"Column '{col_name}' has {null_percentage:.1f}% null values",
                        suggested_fix=f"Review data quality for column '{col_name}' or consider dropping it",
                        affected_rows=null_count,
                        additional_info={'null_percentage': null_percentage}
                    )
                    report.add_issue(issue)
                elif null_percentage == 100:  # Colonne entièrement nulle
                    issue = ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.HIGH,
                        table_name='fact_table',
                        column_name=col_name,
                        description=f"Column '{col_name}' contains only null values",
                        suggested_fix=f"Consider dropping column '{col_name}' or investigate data loading",
                        affected_rows=null_count
                    )
                    report.add_issue(issue)
            
        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.DATA_INTEGRITY,
                severity=IssueSeverity.MEDIUM,
                table_name='fact_table',
                description=f"Error validating data quality: {str(e)}",
                suggested_fix="Check fact table structure and data"
            )
            report.add_issue(issue)
    
    # Méthode de vérification de l'unicité des clés primaires dans les tables de dimension
    def _validate_constraint_violations(self, report: ValidationReport) -> None:
        """Validate constraint violations."""
        try:
            # Vérification des contraintes de clé primaire dans les tables de dimension
            # Identification des tables de dimension
            existing_tables = self._get_existing_tables()
            dimension_tables = [table for table in existing_tables if table.startswith('dim_')]

            # Parcours des tables de dimension
            for dim_table in dimension_tables:
                # Vérification des doublons dans la colonne 'value' (clé primaire)
                duplicates_query = f"""
                    SELECT value, COUNT(*) as count 
                    FROM {dim_table} 
                    GROUP BY value 
                    HAVING COUNT(*) > 1
                """
                
                duplicates_result = self.conn.execute(duplicates_query).fetchall()
                
                # Ajout d'un message au rapport si des duplicats sont présents
                if duplicates_result:
                    duplicate_count = len(duplicates_result)
                    issue = ValidationIssue(
                        issue_type=IssueType.CONSTRAINT_VIOLATION,
                        severity=IssueSeverity.HIGH,
                        table_name=dim_table,
                        column_name='value',
                        description=f"Primary key constraint violation: {duplicate_count} duplicate values in {dim_table}.value",
                        suggested_fix=f"Remove duplicate values from {dim_table}",
                        affected_rows=duplicate_count,
                        additional_info={'duplicate_values': [row[0] for row in duplicates_result[:5]]}  # Première 5 valeurs
                    )
                    report.add_issue(issue)
            
        except Exception as e:
            # Création d'un problème dans le rapport associé à l'erreur
            issue = ValidationIssue(
                issue_type=IssueType.CONSTRAINT_VIOLATION,
                severity=IssueSeverity.MEDIUM,
                table_name='SYSTEM',
                description=f"Error validating constraint violations: {str(e)}",
                suggested_fix="Check table constraints and data integrity"
            )
            report.add_issue(issue)
    
    # Méthodes de validation des préconditions d'opération
    # Méthode de validation des conditions préalables à une opération d'insertion
    def _validate_insert_preconditions(self, report: ValidationReport, df: pd.DataFrame = None, **kwargs) -> None:
        """Validate insert preconditions."""
        # Vérifiation que le jeu de données est spécifié
        if df is None:
            issue = ValidationIssue(
                issue_type=IssueType.DATA_INTEGRITY,
                severity=IssueSeverity.CRITICAL,
                table_name='fact_table',
                description="DataFrame is None for insert operation",
                suggested_fix="Provide a valid DataFrame for insertion"
            )
            report.add_issue(issue)
            return

        # Vérification que le jeu de données est non vide
        if len(df) == 0:
            issue = ValidationIssue(
                issue_type=IssueType.DATA_INTEGRITY,
                severity=IssueSeverity.MEDIUM,
                table_name='fact_table',
                description="DataFrame is empty for insert operation",
                suggested_fix="Provide DataFrame with data for insertion"
            )
            report.add_issue(issue)
        
        # Vérification des noms de colonnes valides
        invalid_columns = [col for col in df.columns if not col.replace('_', '').replace(' ', '').isalnum()]
        if invalid_columns:
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.HIGH,
                table_name='fact_table',
                description=f"Invalid column names detected: {invalid_columns}",
                suggested_fix="Use valid column names (alphanumeric and underscores only)"
            )
            report.add_issue(issue)
    
    # Méthode de validation des conditions préalables à une mise à jour de la base de données
    def _validate_update_preconditions(self, report: ValidationReport, df = None, **kwargs) -> None:
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
            missing_keys = [key for key in primary_keys if key not in df.columns]
            if missing_keys:
                issue = ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.CRITICAL,
                    table_name='fact_table',
                    description=f"Primary keys not found in DataFrame: {missing_keys}",
                    suggested_fix="Ensure all primary keys are present in the DataFrame"
                )
                report.add_issue(issue)
                return

            # Vérification des doublons parmi les valeurs de clés primaires dans le DataFrame.
            duplicate_count = nw.from_native(df, eager_only=True).select(primary_keys).is_duplicated().sum()

            if duplicate_count > 0:
                issue = ValidationIssue(
                    issue_type=IssueType.DATA_INTEGRITY,
                    severity=IssueSeverity.HIGH,
                    table_name='fact_table',
                    description=f"Found {duplicate_count} duplicate primary key values in DataFrame",
                    suggested_fix="Remove duplicate entries based on primary key columns",
                    affected_rows=duplicate_count
                )
                report.add_issue(issue)
    
    # Méthode de validation des conditions préalable à la suppression de données
    def _validate_delete_preconditions(self, report: ValidationReport, filters: Any = None, **kwargs) -> None:
        """Validate delete preconditions."""
        # Vérification de l'existence du filtre de suppression des données
        if filters is None:
            issue = ValidationIssue(
                issue_type=IssueType.DATA_INTEGRITY,
                severity=IssueSeverity.CRITICAL,
                table_name='fact_table',
                description="No filters provided for delete operation",
                suggested_fix="Provide valid filters for delete operation to avoid deleting all data"
            )
            report.add_issue(issue)
    
    # méthode de validation des conditions préalable à un changement de schéma dans la base de données
    def _validate_schema_change_preconditions(self, report: ValidationReport, **kwargs) -> None:
        """Validate schema change preconditions."""
        # Vérification de l'existence des tables avant modification
        if not self._table_exists('fact_table'):
            issue = ValidationIssue(
                issue_type=IssueType.SCHEMA_INCONSISTENCY,
                severity=IssueSeverity.HIGH,
                table_name='fact_table',
                description="Fact table does not exist for schema change operation",
                suggested_fix="Create fact table before attempting schema changes"
            )
            report.add_issue(issue)
    
    # Méthodes utilitaires privées
    # Méthode auxiliaire d'obtention des tables de la base de données
    def _get_existing_tables(self) -> List[str]:
        """Get the list of existing tables."""
        try:
            result = self.conn.execute("SHOW TABLES").fetchall()
            return [row[0] for row in result]
        except:
            return []
    
    # Méthode auxiliaire d'extraction des colonnes de la table des faits
    def _get_fact_table_columns(self) -> List[str]:
        """Get fact table columns."""
        try:
            result = self.conn.execute("DESCRIBE fact_table").fetchall()
            return [row[0] for row in result]
        except:
            return []
    
    # Méthode auxiliaire d'extraction des coonnes d'une table spécifique
    def _get_table_columns(self, table_name: str) -> List[str]:
        """Get columns from a specific table."""
        try:
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            return [row[0] for row in result]
        except:
            return []
    
    # Méthode auxiliaire de description d'une table
    def _get_table_structure(self, table_name: str) -> List[Tuple]:
        """Get the complete structure of a table."""
        try:
            return self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        except:
            return []
    
    # Méthode auxiliaire d'extraction des métadonnées
    def _get_metadata(self) -> pd.DataFrame:
        """Get metadata table content."""
        try:
            return self.conn.execute("SELECT * FROM metadata").fetchdf()
        except:
            return pd.DataFrame()
    
    # Méthode auxiliaire d'obtention de la liste des indices disponibles dans la base de données
    def _get_existing_indexes(self) -> List[Dict[str, str]]:
        """Get the list of existing indexes."""
        try:
            result = self.conn.execute("SELECT index_name, expressions FROM duckdb_indexes()").fetchall()
            return [{'index_name': row[0], 'expressions': row[1]} for row in result]
        except:
            return []
    
    # méthode auxiliaire de vérification de l'existence d'une table
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        try:
            self.conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            return True
        except:
            return False
    
    # Méthode auxiliaire de vérification de l'existence d'une colonne dans la table des faits
    def _column_exists_in_fact_table(self, column_name: str) -> bool:
        """Check if a column exists in fact_table."""
        fact_columns = self._get_fact_table_columns()
        return column_name in fact_columns

    # Méthode auxiliaire d'extraction des colonnes marquées comme clés primaires
    def _get_primary_key_columns(self) -> List[str]:
        """Get all column names that are marked as primary keys in metadata.

        Récupère la liste des colonnes définies comme clés primaires dans la table
        des métadonnées. Cette méthode est nécessaire car DatabaseAuditor n'hérite
        pas de BaseSchemaManager.

        Returns:
            List of primary key column names. Empty list if no primary keys defined.
        """
        try:
            result = self.conn.execute(
                "SELECT name FROM metadata WHERE is_primary_key = true"
            ).fetchall()
            return [row[0] for row in result]
        except:
            return []
    
    # Méthode auxiliaire de vérification de la compatibilité des types entre eux
    def _types_are_compatible(self, expected_type: str, actual_type: str) -> bool:
        """Check if two SQL types are compatible."""
        # Normalisation des types pour comparaison
        expected_normalized = expected_type.upper().strip()
        actual_normalized = actual_type.upper().strip()
        
        # Mapping des types équivalents
        type_equivalents = {
            'VARCHAR': ['TEXT', 'STRING', 'CHAR'],
            'INTEGER': ['INT', 'INT64', 'BIGINT'],
            'DOUBLE': ['FLOAT', 'FLOAT64', 'REAL'],
            'BOOLEAN': ['BOOL']
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
    def get_quick_health_check(self) -> Dict[str, Any]:
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
            health_info = {
                'status': 'unknown',
                'timestamp': time.time(),
                'tables_count': 0,
                'fact_table_rows': 0,
                'dimension_tables_count': 0,
                'metadata_entries': 0,
                'critical_issues': 0
            }
            
            # Comptage des tables
            tables = self._get_existing_tables()
            health_info['tables_count'] = len(tables)
            
            # Comptage des lignes de fact_table
            if 'fact_table' in tables:
                result = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()
                health_info['fact_table_rows'] = result[0] if result else 0
            
            # Comptage des tables de dimension
            health_info['dimension_tables_count'] = len([t for t in tables if t.startswith('dim_')])
            
            # Comptage des entrées de métadonnées
            if 'metadata' in tables:
                result = self.conn.execute("SELECT COUNT(*) FROM metadata").fetchone()
                health_info['metadata_entries'] = result[0] if result else 0
            
            # Validation rapide pour les issues critiques
            quick_report = self.validate_database(ValidationLevel.BASIC)
            health_info['critical_issues'] = quick_report.get_critical_issues_count()
            
            # Détermination du statut global
            if health_info['critical_issues'] > 0:
                health_info['status'] = 'critical'
            elif len(quick_report.issues) > 0:
                health_info['status'] = 'warning'
            elif health_info['fact_table_rows'] > 0 and health_info['metadata_entries'] > 0:
                health_info['status'] = 'healthy'
            else:
                health_info['status'] = 'empty'
            
            return health_info
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error during quick health check: {e}")
            return {
                'status': 'error',
                'timestamp': time.time(),
                'error': str(e)
            }