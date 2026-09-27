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

# Import des utilitaires
from ..utils.hierarchy import validate_hierarchy_forest
from ..utils.logger import _init_logger
from ..utils.sql import SchemaScoped, quote_ident, resolve_catalog
from ..utils.types import METADATA_COLUMNS
from ..utils.value_labels import check_value_label_dependency, validate_value_labels

# Tables composant un jeu de résultats
RESULT_SET_TABLES: tuple[str, ...] = ("fact_table", "metadata", "dataset_metadata")


# Classe des niveaux de validation sur la base de données
class ValidationLevel(Enum):
    """Validation levels for database auditing.

    Attributes:
        BASIC: Structural checks reading only the catalog and the small
            ``metadata``/``dataset_metadata`` tables, never scanning the fact
            table: cheap enough to run after every write.
        COMPREHENSIVE: ``BASIC`` plus the data checks that scan the fact table
            (primary key uniqueness, code -> label functional dependency, null
            shares), run on demand.
    """

    BASIC = "basic"
    COMPREHENSIVE = "comprehensive"


# Classe des types de problèmes détectés
class IssueType(Enum):
    """Types of issues detected during an audit."""

    SCHEMA_INCONSISTENCY = "schema_inconsistency"
    DATA_INTEGRITY = "data_integrity"
    TYPE_MISMATCH = "type_mismatch"
    MISSING_METADATA = "missing_metadata"
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
    Represents a validation issue found during a database audit.

    Attributes:
        issue_type (IssueType): Type of the issue.
        severity (IssueSeverity): Severity level of the issue.
        table_name (str): Name of the affected table.
        column_name (str | None): Name of the affected column, if applicable.
        description (str): Detailed description of the issue.
        suggested_fix (str): Suggested fix for the issue.
        affected_rows (int | None): Number of affected rows, if applicable.
        detected_at (float): Timestamp when the issue was detected.
        additional_info (dict): Additional information about the issue.

    Examples:
        >>> issue = ValidationIssue(IssueType.DATA_INTEGRITY, IssueSeverity.HIGH,
        ...     "fact_table")
        >>> issue.severity
        <IssueSeverity.HIGH: 'high'>
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
        validation_level (ValidationLevel): Level of validation performed.
        start_time (float): Timestamp when the validation started.
        end_time (float | None): Timestamp when the validation ended.
        issues (list[ValidationIssue]): Issues found.
        tables_validated (set[str]): Tables that were validated.
        validation_summary (dict): Summary statistics of the validation.
        recommendations (list[str]): General recommendations derived from the
            issues.

    Examples:
        >>> report = ValidationReport(validation_level=ValidationLevel.BASIC)
        >>> report.get_critical_issues_count()
        0
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
        """Add an issue to the report.

        Args:
            issue: The issue to append.
        """
        self.issues.append(issue)

    # Méthode d'extraction des problèmes par sévérité
    def get_issues_by_severity(self, severity: IssueSeverity) -> list[ValidationIssue]:
        """Get the issues of a given severity.

        Args:
            severity: Severity to filter on.

        Returns:
            list[ValidationIssue]: The matching issues, in detection order.
        """
        return [issue for issue in self.issues if issue.severity == severity]

    # Méthode d'extraction des problèmes par type
    def get_issues_by_type(self, issue_type: IssueType) -> list[ValidationIssue]:
        """Get the issues of a given type.

        Args:
            issue_type: Type to filter on.

        Returns:
            list[ValidationIssue]: The matching issues, in detection order.
        """
        return [issue for issue in self.issues if issue.issue_type == issue_type]

    # Méthode de comptage des problèmes critiques
    def get_critical_issues_count(self) -> int:
        """Get the count of critical issues.

        Returns:
            int: Number of issues of severity ``CRITICAL``.
        """
        return len(self.get_issues_by_severity(IssueSeverity.CRITICAL))

    # Méthode de finalisation du rapport d'audit avec des statistiques
    def finalize(self) -> None:
        """Stamp the end time, then compute the summary and recommendations."""
        self.end_time = time.time()

        # Calcul des statistiques
        self.validation_summary = {
            "total_issues": len(self.issues),
            "critical_issues": len(self.get_issues_by_severity(IssueSeverity.CRITICAL)),
            "high_issues": len(self.get_issues_by_severity(IssueSeverity.HIGH)),
            "medium_issues": len(self.get_issues_by_severity(IssueSeverity.MEDIUM)),
            "low_issues": len(self.get_issues_by_severity(IssueSeverity.LOW)),
            "tables_validated": len(self.tables_validated),
            "validation_duration": self.end_time - self.start_time,
        }

        # Génération des recommandations
        self._generate_recommendations()

    # Méthode auxiliaire de génération de recommandations
    def _generate_recommendations(self) -> None:
        """Derive general recommendations from the issues found."""
        # Problèmes critiques
        critical_count = self.get_critical_issues_count()
        if critical_count > 0:
            self.recommendations.append(
                f"Address immediately the {critical_count} critical issue(s) detected"
            )

        # Incohérences de schéma
        if self.get_issues_by_type(IssueType.SCHEMA_INCONSISTENCY):
            self.recommendations.append("Review the consistency of the database schema")

        # Méta-données manquantes
        if self.get_issues_by_type(IssueType.MISSING_METADATA):
            self.recommendations.append(
                "Complete the metadata and dataset_metadata tables"
            )


# Photographie de l'état du schéma audité, lue une seule fois par audit
@dataclass
class _AuditState:
    """State of the audited schema, read once and shared by every check.

    Attributes:
        tables (set[str]): Tables present in the audited schema.
        metadata (nw.DataFrame | None): Rows of the ``metadata`` table, or None
            when the table is absent.
        fact_columns (dict[str, str]): Fact table column -> SQL type, in column
            order; empty when the fact table is absent.
    """

    tables: set[str]
    metadata: nw.DataFrame[Any] | None
    fact_columns: dict[str, str]


# Classe d'audit de la base de données
class DatabaseAuditor(SchemaScoped):
    """
    Structural audit of a result set (``fact_table``, ``metadata``,
    ``dataset_metadata``).

    The auditor checks what a write cannot guarantee by construction on a base
    that may have been modified outside of the package: the three tables are
    present, ``metadata`` describes exactly the fact table's columns with
    consistent types, the ``parent_name`` links form a forest, the ``label_for``
    declarations are valid and, at the ``COMPREHENSIVE`` level, the primary key is
    unique, every code carries a single label and no column is mostly null. The
    write operations run the ``BASIC`` level (no fact table scan) inside their
    transaction; the ``COMPREHENSIVE`` level is meant to be run on demand.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection.
        schema (str): DuckLake schema audited by this instance.
        catalog_alias (str): Alias of the attached DuckLake catalog, carried
            alongside ``schema``.
        logger: Logger instance for audit tracking.

    Examples:
        >>> auditor = DatabaseAuditor(conn, schema='predictions')
        >>> report = auditor.validate_database(ValidationLevel.COMPREHENSIVE)
        >>> report.get_critical_issues_count()
        0
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
                passed to ``DuckLakeConnector``. Defaults to ``'db'``.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> auditor = DatabaseAuditor(conn)
            >>> report = auditor.validate_database(ValidationLevel.COMPREHENSIVE)
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

    # Méthode principale de validation
    def validate_database(
        self, validation_level: ValidationLevel = ValidationLevel.BASIC
    ) -> ValidationReport:
        """
        Audit the result set at the requested level.

        ``BASIC`` (catalog and small tables only): the three tables are present;
        ``metadata`` has its required columns, no null ``name``/``label``/
        ``sql_type`` and no duplicated name; ``dataset_metadata`` holds exactly one
        row; ``metadata`` and the fact table describe the same columns with
        compatible types; the ``parent_name`` links reference existing columns and
        form a forest; the ``label_for`` declarations are structurally valid.

        ``COMPREHENSIVE`` adds the checks scanning the fact table: primary key
        uniqueness, code -> label functional dependency of every declared pair,
        and the share of nulls per column (one single scan for every column).

        A check that fails to run is itself reported as an issue rather than
        interrupting the audit.

        Args:
            validation_level: Level of validation to perform. Defaults to
                ``ValidationLevel.BASIC``.

        Returns:
            ValidationReport: The finalized report of every issue detected.

        Examples:
            >>> report = auditor.validate_database(ValidationLevel.COMPREHENSIVE)
            >>> for issue in report.issues:
            ...     print(f"{issue.severity.value}: {issue.description}")
        """
        # Création du rapport
        report = ValidationReport(validation_level=validation_level)

        # Lecture unique de l'état du schéma, partagée par tous les contrôles
        try:
            state = self._read_state()
        except Exception as e:
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.CRITICAL,
                    table_name="VALIDATION_SYSTEM",
                    description=f"Could not read the schema state: {e}",
                    suggested_fix="Check the database connection and the schema",
                )
            )
            report.finalize()
            self.logger.error(f"Database validation could not start: {e}")
            return report

        # Contrôles structurels, toujours exécutés
        checks = [
            self._validate_schema_existence,
            self._validate_metadata_consistency,
            self._validate_dataset_metadata,
            self._validate_metadata_fact_consistency,
            self._validate_data_types_consistency,
            self._validate_column_links,
        ]
        # Contrôles des données, qui balaient la table des faits
        if validation_level == ValidationLevel.COMPREHENSIVE:
            checks += [
                self._validate_primary_key_uniqueness,
                self._validate_value_label_dependencies,
                self._validate_data_quality,
            ]

        # Exécution de chaque contrôle : un échec devient un problème du rapport
        for check in checks:
            try:
                check(report, state)
            except Exception as e:
                report.add_issue(
                    ValidationIssue(
                        issue_type=IssueType.SCHEMA_INCONSISTENCY,
                        severity=IssueSeverity.HIGH,
                        table_name="VALIDATION_SYSTEM",
                        description=f"Check {check.__name__} failed: {e}",
                        suggested_fix="Check the structure of the audited tables",
                    )
                )

        # Finalisation du rapport
        report.finalize()

        # Logging
        self.logger.debug(
            f"Validation ({validation_level.value}) of schema {self.schema}:"
            f" {len(report.issues)} issue(s)"
        )
        return report

    # Méthode de lecture de l'état du schéma audité
    def _read_state(self) -> _AuditState:
        """Read the tables, the metadata rows and the fact table columns once.

        Returns:
            _AuditState: The state shared by every check of one audit.

        Raises:
            duckdb.Error: If the catalog cannot be queried.
        """
        # Tables du schéma audité (filtrées sur le catalogue lorsqu'il est attaché)
        catalog_filter = " AND table_catalog = ?" if self._catalog is not None else ""
        params = [self.schema] + ([self._catalog] if self._catalog is not None else [])
        tables = {
            row[0]
            for row in self.conn.execute(
                "SELECT table_name FROM information_schema.tables"
                f" WHERE table_schema = ?{catalog_filter}",
                params,
            ).fetchall()
        }

        # Lignes de metadata
        metadata: nw.DataFrame[Any] | None = None
        if "metadata" in tables:
            metadata = nw.from_native(
                self.conn.execute(
                    f"SELECT * FROM {self._qualified('metadata')}"
                ).to_arrow_table(),
                eager_only=True,
            )

        # Colonnes et types de la table des faits
        fact_columns: dict[str, str] = {}
        if "fact_table" in tables:
            fact_columns = {
                row[0]: row[1]
                for row in self.conn.execute(
                    f"DESCRIBE {self._qualified('fact_table')}"
                ).fetchall()
            }

        return _AuditState(tables=tables, metadata=metadata, fact_columns=fact_columns)

    # ---------------------------------------------------------------------------
    # Contrôles structurels (niveau BASIC)
    # ---------------------------------------------------------------------------

    # Méthode de validation de l'existence des trois tables
    def _validate_schema_existence(
        self, report: ValidationReport, state: _AuditState
    ) -> None:
        """Check that the three tables of a result set exist.

        A missing ``metadata`` table is critical (the interface cannot describe
        any column); a missing fact table or ``dataset_metadata`` is reported as
        high.

        Args:
            report: Validation report collecting the issues found.
            state: State of the audited schema.
        """
        # Parcours des tables attendues
        for table in RESULT_SET_TABLES:
            # Validation de la table du schéma
            if table in state.tables:
                report.tables_validated.add(table)
                continue
            report.add_issue(
                ValidationIssue(
                    issue_type=(
                        IssueType.MISSING_METADATA
                        if table == "dataset_metadata"
                        else IssueType.SCHEMA_INCONSISTENCY
                    ),
                    severity=(
                        IssueSeverity.CRITICAL
                        if table == "metadata"
                        else IssueSeverity.HIGH
                    ),
                    table_name=table,
                    description=f"Table '{table}' is missing",
                    suggested_fix="Rebuild the schema with"
                    " DuckLakeTablesBuilder.build_schema",
                )
            )

    # Méthode de validation de la cohérence interne de metadata
    def _validate_metadata_consistency(
        self, report: ValidationReport, state: _AuditState
    ) -> None:
        """Check the required columns, null fields and duplicated names of metadata.

        Args:
            report: Validation report collecting the issues found.
            state: State of the audited schema.
        """
        # Extraction des métadonnées
        metadata = state.metadata
        if metadata is None:
            return

        # Table vide : aucune colonne décrite
        if len(metadata) == 0:
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.MISSING_METADATA,
                    severity=IssueSeverity.HIGH,
                    table_name="metadata",
                    description="Metadata table is empty",
                    suggested_fix="Populate metadata with one row per fact_table"
                    " column",
                )
            )
            return

        # Colonnes requises
        missing_columns = [c for c in METADATA_COLUMNS if c not in metadata.columns]
        if missing_columns:
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    table_name="metadata",
                    description=f"Missing required columns in metadata:"
                    f" {missing_columns}",
                    suggested_fix="Add the missing columns to the metadata table",
                )
            )

        # Valeurs nulles dans les champs indispensables à l'interface
        for column in ("name", "label", "sql_type"):
            if column not in metadata.columns:
                continue
            null_count = int(metadata[column].is_null().sum())
            if null_count > 0:
                report.add_issue(
                    ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.MEDIUM,
                        table_name="metadata",
                        column_name=column,
                        description=f"Found {null_count} null value(s) in the"
                        f" critical metadata column '{column}'",
                        suggested_fix=f"Fill metadata.{column}",
                        affected_rows=null_count,
                    )
                )

        # Doublons de nom : une colonne ne peut être décrite qu'une fois
        if "name" in metadata.columns:
            duplicate_names = sorted(
                set(metadata.filter(nw.col("name").is_duplicated())["name"].to_list())
            )
            if duplicate_names:
                report.add_issue(
                    ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.HIGH,
                        table_name="metadata",
                        column_name="name",
                        description=f"Duplicate column names in metadata:"
                        f" {duplicate_names}",
                        suggested_fix="Keep a single metadata row per column",
                        affected_rows=len(duplicate_names),
                    )
                )

    # Méthode de validation de la cardinalité de dataset_metadata
    def _validate_dataset_metadata(
        self, report: ValidationReport, state: _AuditState
    ) -> None:
        """Check that ``dataset_metadata`` holds exactly one row.

        Args:
            report: Validation report collecting the issues found.
            state: State of the audited schema.
        """
        if "dataset_metadata" not in state.tables:
            return
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM {self._qualified('dataset_metadata')}"
        ).fetchone()
        row_count = int(row[0]) if row is not None else 0
        if row_count != 1:
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.MISSING_METADATA,
                    severity=IssueSeverity.MEDIUM,
                    table_name="dataset_metadata",
                    description=f"Dataset metadata table holds {row_count} rows,"
                    " exactly one is expected",
                    suggested_fix="Keep a single descriptive row per schema",
                    affected_rows=row_count,
                )
            )

    # Méthode de validation de l'accord entre metadata et la table des faits
    def _validate_metadata_fact_consistency(
        self, report: ValidationReport, state: _AuditState
    ) -> None:
        """Check that ``metadata`` and the fact table describe the same columns.

        The metadata table is the contract between the database and the interface:
        it must hold exactly one row per fact table column, no more and no less.

        Args:
            report: Validation report collecting the issues found.
            state: State of the audited schema.
        """
        # Cas où les métadonnées ou la table des faits ne sont pas renseignés
        if state.metadata is None or "fact_table" not in state.tables:
            return

        # Extraction des colonnes de métadonnées
        metadata_columns = set(state.metadata["name"].to_list())
        # Extraction des colonnes de la table des faits
        fact_columns = set(state.fact_columns)

        # Colonnes décrites dans metadata mais absentes de la table des faits
        for column in sorted(metadata_columns - fact_columns):
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    table_name="metadata",
                    column_name=column,
                    description=f"Column '{column}' is described in metadata but"
                    " missing from fact_table",
                    suggested_fix=f"Drop the metadata row of '{column}' or add the"
                    " column to fact_table",
                )
            )

        # Colonnes de la table des faits sans ligne de metadata
        for column in sorted(fact_columns - metadata_columns):
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.MISSING_METADATA,
                    severity=IssueSeverity.HIGH,
                    table_name="fact_table",
                    column_name=column,
                    description=f"Column '{column}' exists in fact_table but has no"
                    " metadata row",
                    suggested_fix=f"Add a metadata row describing '{column}'",
                )
            )

    # Méthode de validation de la cohérence des types déclarés et physiques
    def _validate_data_types_consistency(
        self, report: ValidationReport, state: _AuditState
    ) -> None:
        """Check that ``metadata.sql_type`` matches the physical column types.

        Columns missing on one side are skipped: they are already reported by
        :meth:`_validate_metadata_fact_consistency`.

        Args:
            report: Validation report collecting the issues found.
            state: State of the audited schema.
        """
        # Cas les métadonnées ou la table des faits ne sont pas renseignés
        if state.metadata is None or not state.fact_columns:
            return

        # Parcours des lignes de métadonnées
        for row in state.metadata.iter_rows(named=True):
            # Extraction du nom de la colonne et du type SQL
            # attendus de la table des métadonnées
            column = row["name"]
            expected_type = row.get("sql_type")
            # Extraction du type de la colonne dans la table des faits
            actual_type = state.fact_columns.get(column)
            # Vérification que les deux types sont spécifiés
            if actual_type is None or expected_type is None:
                continue
            # Validation de la cohérence des types
            if not self._types_are_compatible(expected_type, actual_type):
                report.add_issue(
                    ValidationIssue(
                        issue_type=IssueType.TYPE_MISMATCH,
                        severity=IssueSeverity.MEDIUM,
                        table_name="fact_table",
                        column_name=column,
                        description=f"Type mismatch for column '{column}': metadata"
                        f" declares {expected_type}, fact_table holds {actual_type}",
                        suggested_fix="Align metadata.sql_type on the fact_table"
                        " column type",
                        additional_info={
                            "expected_type": expected_type,
                            "actual_type": actual_type,
                        },
                    )
                )

    # Méthode de validation des liens parent_name et label_for
    def _validate_column_links(
        self, report: ValidationReport, state: _AuditState
    ) -> None:
        """Check the ``parent_name`` forest and the ``label_for`` declarations.

        ``parent_name`` must reference a column described in ``metadata`` and the
        links must form a forest (no cycle); every ``label_for`` declaration must
        pass the structural checks of :func:`validate_value_labels` (target
        exists, no chaining, label column is a ``VARCHAR`` outside any hierarchy and
        not a primary key). Only ``metadata`` is read.

        Args:
            report: Validation report collecting the issues found.
            state: State of the audited schema.
        """
        # Extraction des données
        metadata = state.metadata
        if metadata is None or len(metadata) == 0:
            return
        # Vérification que les colonnes attendues sont dans la table des métadonnées
        required = {"name", "sql_type", "is_primary_key", "parent_name", "label_for"}
        if not required.issubset(metadata.columns):
            return

        # Extraction des lignes de la table des métadonnées
        rows = list(metadata.iter_rows(named=True))
        # Extraction des noms de colonne et de parents
        names = {row["name"] for row in rows}
        parent_of = {row["name"]: row["parent_name"] for row in rows}

        # Parents déclarés inexistants
        for column, parent in sorted(parent_of.items()):
            if parent is not None and parent not in names:
                report.add_issue(
                    ValidationIssue(
                        issue_type=IssueType.SCHEMA_INCONSISTENCY,
                        severity=IssueSeverity.HIGH,
                        table_name="metadata",
                        column_name=column,
                        description=f"parent_name of '{column}' references"
                        f" '{parent}', which has no metadata row",
                        suggested_fix="Clear or correct the parent_name via"
                        " update_column_metadata",
                    )
                )

        # Forêt : aucun cycle dans les liens de hiérarchie
        try:
            validate_hierarchy_forest(parent_of)
        except ValueError as e:
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    table_name="metadata",
                    description=str(e),
                    suggested_fix="Break the cycle by clearing one parent_name via"
                    " update_column_metadata",
                )
            )

        # Déclarations label_for : contrôles structurels sur l'état complet
        label_for_map = {
            row["name"]: row["label_for"]
            for row in rows
            if row["label_for"] is not None
        }
        if not label_for_map:
            return
        try:
            validate_value_labels(
                label_for_map,
                {row["name"]: row["sql_type"] for row in rows},
                [row["name"] for row in rows if row["is_primary_key"]],
                parent_of,
            )
        except ValueError as e:
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.SCHEMA_INCONSISTENCY,
                    severity=IssueSeverity.HIGH,
                    table_name="metadata",
                    description=f"Invalid label_for declaration: {e}",
                    suggested_fix="Correct or clear the offending label_for via"
                    " update_column_metadata",
                )
            )

    # ---------------------------------------------------------------------------
    # Contrôles des données (niveau COMPREHENSIVE)
    # ---------------------------------------------------------------------------

    # Méthode de vérification de l'unicité de la clé primaire
    def _validate_primary_key_uniqueness(
        self, report: ValidationReport, state: _AuditState
    ) -> None:
        """Check that the fact table is unique on its primary key columns.

        DuckLake supports no DDL constraint, so primary key uniqueness is enforced
        applicatively at build and upsert time; this check verifies it actually
        holds on the stored data.

        Args:
            report: Validation report collecting the issues found.
            state: State of the audited schema.
        """
        # Vérification que les métadonnées et la table des faits sont non-vides
        if state.metadata is None or "fact_table" not in state.tables:
            return
        # Liste des clés primaires
        primary_keys = [
            row["name"]
            for row in state.metadata.iter_rows(named=True)
            if row.get("is_primary_key")
        ]
        if not primary_keys:
            return

        # Combinaisons de clés présentes plus d'une fois
        key_columns = ", ".join(quote_ident(c) for c in primary_keys)
        row = self.conn.execute(f"""
            SELECT COUNT(*) FROM (
                SELECT 1 FROM {self._qualified("fact_table")}
                GROUP BY {key_columns}
                HAVING COUNT(*) > 1
            )
        """).fetchone()

        # Comptage des duplicats de clés primaires
        duplicate_count = int(row[0]) if row is not None else 0
        if duplicate_count > 0:
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.CONSTRAINT_VIOLATION,
                    severity=IssueSeverity.HIGH,
                    table_name="fact_table",
                    column_name=", ".join(primary_keys),
                    description=f"{duplicate_count} primary key combination(s)"
                    " appear more than once in fact_table",
                    suggested_fix="Delete the duplicated rows with"
                    " DatabaseDeleter.delete_rows, or restore a snapshot",
                    affected_rows=duplicate_count,
                    additional_info={"primary_keys": primary_keys},
                )
            )

    # Méthode de validation de la dépendance fonctionnelle code -> libellé
    def _validate_value_label_dependencies(
        self, report: ValidationReport, state: _AuditState
    ) -> None:
        """Check the code -> label functional dependency of every declared pair.

        Args:
            report: Validation report collecting the issues found.
            state: State of the audited schema.
        """
        # Vérification que les métadonnées et la table des faits sont spécifiés
        if state.metadata is None or "fact_table" not in state.tables:
            return
        # Vérification que les métadonnées contiennent une colonne de labels
        if "label_for" not in state.metadata.columns:
            return
        # Extraction de la table des faits
        fact_table = self._qualified("fact_table")
        # Parcours des colonnes dans la table des métadonnées
        for row in state.metadata.iter_rows(named=True):
            # Extraction de la colonne des labels et de la colonne des codes
            label_column, code_column = row["name"], row["label_for"]
            if code_column is None:
                continue
            # Paire incomplète : déjà signalée par les contrôles structurels
            if (
                label_column not in state.fact_columns
                or code_column not in state.fact_columns
            ):
                continue
            try:
                # Vérification des associations code/label
                check_value_label_dependency(
                    self.conn, fact_table, code_column, label_column
                )
            except ValueError as e:
                report.add_issue(
                    ValidationIssue(
                        issue_type=IssueType.CONSTRAINT_VIOLATION,
                        severity=IssueSeverity.CRITICAL,
                        table_name="fact_table",
                        column_name=label_column,
                        description=str(e),
                        suggested_fix="Correct the labels via"
                        " DatabaseUpdater.update_value_labels",
                        additional_info={"code_column": code_column},
                    )
                )

    # Méthode de validation de la part de valeurs nulles par colonne
    def _validate_data_quality(
        self, report: ValidationReport, state: _AuditState
    ) -> None:
        """Report the empty fact table and the columns that are mostly null.

        The non-null count of every column is read in a single scan
        (``SELECT COUNT(*), COUNT(c1), COUNT(c2), …``). A column holding only nulls
        is reported as high; a column more than half null as medium.

        Args:
            report: Validation report collecting the issues found.
            state: State of the audited schema.
        """
        # Vérification que la table des faits est renseignée
        if not state.fact_columns:
            return
        # Extraction des colonnes
        columns = list(state.fact_columns)
        # Comptage des modalités non nulles par colonne
        counts = ", ".join(f"COUNT({quote_ident(c)})" for c in columns)
        row = self.conn.execute(
            f"SELECT COUNT(*), {counts} FROM {self._qualified('fact_table')}"
        ).fetchone()
        if row is None:
            return
        # Extraction du total
        total_rows = int(row[0])

        # Table vide
        if total_rows == 0:
            report.add_issue(
                ValidationIssue(
                    issue_type=IssueType.DATA_INTEGRITY,
                    severity=IssueSeverity.MEDIUM,
                    table_name="fact_table",
                    description="Fact table is empty",
                    suggested_fix="Check that the data has been loaded",
                )
            )
            return

        # Part des valeurs nulles, colonne par colonne
        for column, non_null_count in zip(columns, row[1:], strict=True):
            null_count = total_rows - int(non_null_count)
            null_percentage = 100 * null_count / total_rows
            # Colonne entièrement nulle : testée avant le seuil de 50 %, qu'elle
            # dépasse aussi
            if null_count == total_rows:
                report.add_issue(
                    ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.HIGH,
                        table_name="fact_table",
                        column_name=column,
                        description=f"Column '{column}' contains only null values",
                        suggested_fix=f"Drop the column '{column}' or check the data"
                        " loading",
                        affected_rows=null_count,
                    )
                )
            elif null_percentage > 50:
                report.add_issue(
                    ValidationIssue(
                        issue_type=IssueType.DATA_INTEGRITY,
                        severity=IssueSeverity.MEDIUM,
                        table_name="fact_table",
                        column_name=column,
                        description=f"Column '{column}' has {null_percentage:.1f}%"
                        " null values",
                        suggested_fix=f"Review the data quality of column '{column}'",
                        affected_rows=null_count,
                        additional_info={"null_percentage": null_percentage},
                    )
                )

    # ---------------------------------------------------------------------------
    # Méthodes utilitaires
    # ---------------------------------------------------------------------------

    # Méthode auxiliaire de vérification de la compatibilité des types entre eux
    @staticmethod
    def _types_are_compatible(expected_type: str, actual_type: str) -> bool:
        """Check whether a declared SQL type matches a physical one.

        Types are compared case-insensitively, with a few synonyms accepted
        (``TEXT``/``STRING`` for ``VARCHAR``, ``INT`` for ``INTEGER``, ``FLOAT8``
        for ``DOUBLE``, ``BOOL`` for ``BOOLEAN``).

        Args:
            expected_type: Type declared in ``metadata.sql_type``.
            actual_type: Type reported by ``DESCRIBE fact_table``.

        Returns:
            bool: True if both designate the same type.

        Examples:
            >>> DatabaseAuditor._types_are_compatible("VARCHAR", "text")
            True
            >>> DatabaseAuditor._types_are_compatible("INTEGER", "BIGINT")
            False
        """
        # Synonymes d'un même type physique
        synonyms = {
            "TEXT": "VARCHAR",
            "STRING": "VARCHAR",
            "CHAR": "VARCHAR",
            "INT": "INTEGER",
            "INT4": "INTEGER",
            "INT8": "BIGINT",
            "FLOAT8": "DOUBLE",
            "FLOAT4": "FLOAT",
            "REAL": "FLOAT",
            "BOOL": "BOOLEAN",
        }
        expected = expected_type.upper().strip()
        actual = actual_type.upper().strip()
        return synonyms.get(expected, expected) == synonyms.get(actual, actual)

    # Méthode publique de vérification rapide de la base de données
    def get_quick_health_check(self) -> dict[str, Any]:
        """
        Perform a quick health check of the result set.

        Runs a ``BASIC`` validation and counts the fact table and metadata rows.

        Returns:
            dict[str, Any]: ``status`` (``'healthy'``, ``'warning'``,
            ``'critical'``, ``'empty'`` or ``'error'``), ``timestamp``,
            ``tables_count``, ``fact_table_rows``, ``metadata_entries``,
            ``has_dataset_metadata`` and ``critical_issues``; ``error`` is added
            when the check itself fails.

        Example:
            >>> health = auditor.get_quick_health_check()
            >>> health['status']
            'healthy'
        """
        try:
            # Initialisation des informations de la base de données
            health_info: dict[str, Any] = {
                "status": "unknown",
                "timestamp": time.time(),
                "tables_count": 0,
                "fact_table_rows": 0,
                "metadata_entries": 0,
                "has_dataset_metadata": False,
                "critical_issues": 0,
            }

            # Tables présentes et comptages
            state = self._read_state()
            health_info["tables_count"] = len(state.tables)
            health_info["has_dataset_metadata"] = "dataset_metadata" in state.tables
            health_info["fact_table_rows"] = (
                self._count_rows("fact_table") if "fact_table" in state.tables else 0
            )
            health_info["metadata_entries"] = (
                len(state.metadata) if state.metadata is not None else 0
            )

            # Validation structurelle
            quick_report = self.validate_database(ValidationLevel.BASIC)
            health_info["critical_issues"] = quick_report.get_critical_issues_count()

            # Statut global
            if health_info["critical_issues"] > 0:
                health_info["status"] = "critical"
            elif quick_report.issues:
                health_info["status"] = "warning"
            elif health_info["fact_table_rows"] > 0 and health_info["metadata_entries"]:
                health_info["status"] = "healthy"
            else:
                health_info["status"] = "empty"

            return health_info

        except Exception as e:
            # Logging
            self.logger.error(f"Error during quick health check: {e}")
            return {"status": "error", "timestamp": time.time(), "error": str(e)}
