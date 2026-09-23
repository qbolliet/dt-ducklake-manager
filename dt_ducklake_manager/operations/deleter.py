# Importation des modules
# Modules de base
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# DuckDB
import duckdb

from ..maintenance.auditor import DatabaseAuditor, ValidationLevel
from ..reporting import OperationReport

# Import des utilitaires
from ..utils.sql import _build_where_clause

# Import des gestionnaires
from ._base import BaseSchemaManager
from ._data import DataManager

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe de suppression de données dans la base
class DatabaseDeleter(BaseSchemaManager):
    """
    Database deleter using modular architecture.

    Every public deletion runs as a single DuckDB transaction (``BEGIN`` /
    ``COMMIT``, ``ROLLBACK`` on exception) opened by
    :meth:`BaseSchemaManager._transaction`; post-write compaction runs after the
    commit. Recovery beyond a failed operation relies on DuckLake time travel
    (``DatabaseRecoveryManager.list_ducklake_snapshots`` and
    ``DuckLakeConnector(..., snapshot_version=N)``), not on application backups.

    Attributes:
        data_mgr (DataManager): Manages fact table operations
        auditor (DatabaseAuditor): Validates database state and operations
        enable_validation (bool): Whether to enable validation
        auto_cleanup (bool): Whether to automatically clean up orphaned data
        catalog_alias (str): Alias of the attached DuckLake catalog
        schema (str): DuckLake schema name
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        log_filename: str | os.PathLike[str] | None = None,
        enable_validation: bool = True,
        auto_cleanup: bool = True,
        catalog_alias: str = "db",
        schema: str = "main",
    ):
        """
        Initialize the refactored database deleter.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            log_filename: Path to log file.
            enable_validation: Whether to enable pre/post operation validation.
            auto_cleanup: Whether to automatically clean up orphaned data.
            catalog_alias: Alias used in the DuckLake ATTACH statement.
                Defaults to ``'db'``.
            schema: DuckLake schema to delete from. A single catalog can host several
                schemas; all tables are qualified by this one, and compaction calls
                target it. Defaults to ``'main'``.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> deleter = DatabaseDeleter(conn, enable_validation=True,
            auto_cleanup=True)
            >>> deleter = DatabaseDeleter(conn, schema='predictions')
        """
        # Initialisation du parent
        super().__init__(
            connection=connection,
            log_filename=log_filename,
            schema=schema,
            catalog_alias=catalog_alias,
        )

        # Initialisation des gestionnaires spécialisés
        self.data_mgr = DataManager(
            connection=connection,
            log_filename=log_filename,
            schema=schema,
            catalog_alias=catalog_alias,
        )

        self.auditor = (
            DatabaseAuditor(
                connection=connection,
                log_filename=log_filename,
                schema=schema,
                catalog_alias=catalog_alias,
            )
            if enable_validation
            else None
        )

        # Configuration
        self.enable_validation = enable_validation
        self.auto_cleanup = auto_cleanup

        # Configuration DuckLake pour les appels de compaction :
        # l'alias du catalogue (self.catalog_alias) et le schéma (self.schema) sont
        # tous deux portés par la classe de base BaseSchemaManager.

    # Méthode de validation de l'opération de suppression avant son exécution
    def validate_operation(self, operation_type: str, **kwargs: Any) -> bool:
        """
        Validate delete operations before execution.

        Args:
            operation_type: Type of operation to validate
            **kwargs: Operation-specific parameters

        Returns:
            True if operation is valid
        """
        # Vérifiation qu'un auditeur est bien fourni et que la validation est demandée
        if not self.enable_validation or not self.auditor:
            return True

        # Validation par l'auditeur
        validation_report = self.auditor.validate_operation_preconditions(
            operation_type, **kwargs
        )

        # Vérification des issues critiques
        critical_issues = validation_report.get_critical_issues_count()
        if critical_issues > 0:
            # Logging
            self.logger.error(
                f"Critical validation issues found for {operation_type} operation:"
            )
            for issue in validation_report.issues:
                if issue.severity.value == "critical":
                    self.logger.error(f"  - {issue.description}")
            return False

        # Avertissements pour les issues de haute priorité
        high_issues = [
            issue
            for issue in validation_report.issues
            if issue.severity.value == "high"
        ]
        if high_issues:
            # Logging
            self.logger.warning(
                f"High priority validation issues found for {operation_type} operation:"
            )
            for issue in high_issues:
                self.logger.warning(f"  - {issue.description}")

        return True

    # Méthode principale de suppression de lignes
    def delete_rows(
        self,
        filters: str | list[Any] | dict[Any, Any] | None = None,
        use_transaction: bool = True,
        perform_cleanup: bool | None = None,
        compact_after_update: bool = True,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> OperationReport:
        """
        Delete rows from fact table based on filters with atomic operations.

        Args:
            filters: Filter conditions (string SQL condition or structured filters)
            use_transaction: Whether to run every step inside a single DuckDB
                transaction, so that a failure mid-deletion leaves the fact table
                exactly as it was. Defaults to True. When False the steps run in
                autocommit mode and a failure leaves partial state behind.
            perform_cleanup: Whether to drop the columns left null-only by the
                deletion (None = use auto_cleanup setting). Runs after the deletion
                is committed, in its own transaction: a cleanup failure is reported
                in ``report.warnings`` without restoring the deleted rows.
            compact_after_update: Whether to run DuckLake compaction (merge small delta
                files and rewrite delete files) immediately after a successful deletion.
                Adds write latency but keeps read performance optimal. Defaults to True.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``). Ignored (skipped with a DEBUG
                log) on a connection with no real DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``.

        Returns:
            OperationReport: report describing what was actually deleted. A
            pre-transaction validation failure returns a report with a warning and
            ``rows_deleted == 0`` rather than raising.

        Example:
            >>> # Using string filter
            >>> report = deleter.delete_rows("status = 'inactive'")
            >>>
            >>> # Using structured filter
            >>> filters = [('status', '=', 'inactive'), ('date', '<', '2023-01-01')]
            >>> report = deleter.delete_rows(filters, use_transaction=True,
            compact_after_update=False)
        """
        # Validation préalable
        if not self.validate_operation("delete", filters=filters):
            # Logging
            self.logger.error("Pre-delete validation failed")
            # Rapport
            return self._early_failure_report(
                "delete_rows", run_id, "Pre-delete validation failed"
            )

        # Configuration du nettoyage
        if perform_cleanup is None:
            perform_cleanup = self.auto_cleanup

        # Logging
        self.logger.info(
            f"Starting row deletion (transaction: {use_transaction}, cleanup:"
            f"{perform_cleanup})"
        )

        # Bloc transactionnel unique : la suppression et sa validation forment un
        # tout, annulé en bloc sur exception.
        try:
            with self._transaction(
                "delete_rows",
                use_transaction=use_transaction,
                run_id=run_id,
                commit_message=commit_message,
                commit_info=commit_info,
            ) as report:
                rows_deleted = self._run_delete_rows(filters, report)
        except Exception as e:
            # Logging
            self.logger.error(f"Error during row deletion: {e}")
            final_report = self.last_report
            assert final_report is not None  # posé par _transaction sur tout échec
            return final_report

        # Logging
        self.logger.info(
            f"Row deletion completed successfully: {rows_deleted} rows deleted"
        )

        # Horodatage de la dernière écriture réussie
        self._touch_dataset_metadata()

        final_report = self.last_report
        assert final_report is not None  # posé par _transaction sur tout succès

        # Suppression des colonnes devenues entièrement nulles, après le commit :
        # DuckDB refuse de valider une transaction mêlant DELETE et DROP COLUMN sur
        # la même table. Transaction distincte, non critique : un échec est reporté
        # sans annuler la suppression des lignes.
        if perform_cleanup and rows_deleted > 0:
            self._run_post_delete_cleanup(final_report, use_transaction)

        # Compaction DuckLake optionnelle après le commit (réécriture des delete
        # files) : la maintenance ne fait jamais partie de la transaction.
        if rows_deleted > 0 and compact_after_update:
            self.maintenance.compact(schema=self.schema, report=final_report)

        self._finalize_report_after_write(final_report)
        self.logger.info(final_report.summary())
        return final_report

    # Méthode de nettoyage des colonnes nulles après une suppression de lignes
    def _run_post_delete_cleanup(
        self, report: OperationReport, use_transaction: bool
    ) -> None:
        """Drop the columns left null-only by a committed row deletion.

        Runs ``_cleanup_null_only_columns`` in its own transaction and merges its
        outcome into the ``delete_rows`` report: dropped columns go to
        ``report.columns_dropped``, warnings to ``report.warnings``. A failure is
        logged and reported as a warning; the row deletion stays committed.
        ``self.last_report`` is set back to ``report`` in every case.

        Args:
            report: Report of the committed ``delete_rows`` operation, updated in
                place.
            use_transaction: Whether the cleanup runs inside a DuckDB transaction.
        """
        try:
            # Ajout au rapport et suppression des colonnes de null
            report.columns_dropped.extend(
                self._cleanup_null_only_columns(use_transaction=use_transaction)
            )
            cleanup_report = self.last_report
            if cleanup_report is not None and cleanup_report is not report:
                report.warnings.extend(cleanup_report.warnings)
        except Exception as e:
            # Logging
            self.logger.warning(f"Cleanup failed, but row deletion completed: {e}")
            # Ajout au rapport
            report.warnings.append(f"Null-only column cleanup failed: {e}")
        finally:
            # Mise à jour du dernier rapport
            self.last_report = report

    # Méthode d'exécution de la suppression des lignes
    def _run_delete_rows(
        self,
        filters: str | list[Any] | dict[Any, Any] | None,
        report: OperationReport,
    ) -> int:
        """Run the ordered steps of a row deletion.

        Steps, in order: deletion of the matching rows, post-deletion validation.
        Called from inside the transaction opened by ``delete_rows``: a failure
        raises, so the deletion is rolled back and the rows return. The null-only
        column cleanup runs afterwards, once committed (``_run_post_delete_cleanup``).

        Args:
            filters: Filter conditions (SQL string or structured filters).
            report: In-progress report of the enclosing transaction.

        Returns:
            Number of rows deleted.

        Raises:
            RuntimeError: If post-deletion validation finds critical issues, naming
                the step reached.
        """
        # Comptage initial : le nombre de lignes supprimées est mesuré sur la table
        # elle-même, la valeur retournée par data_mgr ne couvrant pas les suppressions
        # en cascade éventuelles.
        _row = self.conn.execute(
            f"SELECT COUNT(*) FROM {self._qualified('fact_table')}"
        ).fetchone()
        initial_count = _row[0] if _row is not None else 0

        # Étape 1 : suppression des lignes
        self.data_mgr.delete_rows(filters)

        # Comptage du nombre de lignes effectivement supprimées
        _row2 = self.conn.execute(
            f"SELECT COUNT(*) FROM {self._qualified('fact_table')}"
        ).fetchone()
        current_count = _row2[0] if _row2 is not None else 0
        rows_deleted = initial_count - current_count
        # Valeur exacte, calculée en Python : sert de repli tant que
        # _transaction n'a pas pu obtenir la mesure DuckLake réelle (table_changes),
        # qui la remplacera si elle est disponible.
        report.rows_deleted = rows_deleted

        # Aucune suppression : rien à valider
        if rows_deleted == 0:
            return 0

        # Étape 2 : validation post-suppression
        if self.enable_validation and self.auditor:
            validation_report = self.auditor.validate_database(ValidationLevel.BASIC)
            critical_issues = validation_report.get_critical_issues_count()
            # Problèmes critiques : annulation de la suppression
            if critical_issues > 0:
                raise RuntimeError(
                    f"post-deletion validation found {critical_issues} critical"
                    " issue(s)"
                )

        return rows_deleted

    # Méthode principale de suppression de colonnes
    def delete_columns(
        self,
        columns: list[str],
        use_transaction: bool = True,
        validate_dependencies: bool = True,
        cascade: bool = False,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> OperationReport:
        """
        Delete columns from fact table and related structures with dependency analysis.

        A column that is the parent of another column in a hierarchy, or the
        ``label_for`` target of one or more label columns, cannot be deleted
        by default: it would silently orphan its children's ``parent_name`` or its
        label columns' ``label_for``. Pass ``cascade=True`` to allow it anyway; every
        child's ``parent_name`` and every label column's ``label_for`` is then reset
        to ``NULL``, with a warning. A dropped column that is part of
        ``dataset_metadata.cluster_by`` is also removed from it (reset to ``NULL`` if
        it was the only sort column), with a warning. Dropping a label column itself
        needs nothing special. ``ALTER TABLE ... DROP COLUMN`` is a DuckLake
        metadata-only operation: no data file is rewritten.

        Args:
            columns: List of column names to delete
            use_transaction: Whether to run every deletion inside a single DuckDB
                transaction, so that a failure leaves the fact table and its
                ``metadata`` rows exactly as they were. Defaults to True.
            validate_dependencies: Whether to validate column dependencies
            cascade: Whether to allow deleting a column that is the parent of
                another column or the ``label_for`` target of a label column,
                detaching its children/label columns (``parent_name``/``label_for``
                set to ``NULL``) instead of refusing the deletion. Defaults to False.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``). Ignored (skipped with a DEBUG
                log) on a connection with no real DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``.

        Returns:
            OperationReport: successfully dropped columns are in
            ``report.columns_dropped``; a column that failed (not found, refused
            by dependency analysis, or an error mid-deletion) is instead named in
            ``report.warnings``, never silently omitted.

        Example:
            >>> report = deleter.delete_columns(['old_col1', 'old_col2'])
            >>> report.columns_dropped
            ['old_col1', 'old_col2']
            >>> # Deleting a hierarchy parent, detaching its children
            >>> report = deleter.delete_columns(['region'], cascade=True)
        """
        # Validation préalable
        if not self.validate_operation("drop_column", columns=columns):
            # Logging
            self.logger.error("Pre-column-deletion validation failed")
            # Rapport
            return self._early_failure_report(
                "delete_columns", run_id, "Pre-column-deletion validation failed"
            )

        # Analyse des dépendances si activée
        if validate_dependencies:
            dependency_report = self._analyze_column_dependencies(
                columns, cascade=cascade
            )
            if dependency_report["has_critical_dependencies"]:
                # Logging
                self.logger.error(
                    "Critical dependencies found, aborting columns deletion"
                )
                # Rapport
                return self._early_failure_report(
                    "delete_columns",
                    run_id,
                    f"Critical dependencies found for column(s) {columns};"
                    " aborting columns deletion",
                )

        # Logging
        self.logger.info(f"Starting columns deletion (transaction: {use_transaction})")

        # Bloc transactionnel unique : sur exception, ni les colonnes ni leurs
        # lignes metadata ne sont perdues.
        try:
            with self._transaction(
                "delete_columns",
                use_transaction=use_transaction,
                run_id=run_id,
                commit_message=commit_message,
                commit_info=commit_info,
            ) as report:
                results = self._run_delete_columns(columns, cascade=cascade)
                report.columns_dropped = [
                    col for col, success in results.items() if success
                ]
                for col, success in results.items():
                    if not success:
                        report.warnings.append(f"Column '{col}' could not be deleted")
        except Exception as e:
            # Logging
            self.logger.error(f"Error during column deletion: {e}")
            final_report = self.last_report
            assert final_report is not None  # posé par _transaction sur tout échec
            return final_report

        # Horodatage dès qu'au moins une colonne a effectivement été supprimée
        final_report = self.last_report
        assert final_report is not None  # posé par _transaction sur tout succès
        if final_report.columns_dropped:
            self._touch_dataset_metadata()

        self._finalize_report_after_write(final_report)
        self.logger.info(final_report.summary())
        return final_report

    # Méthode d'exécution de la suppression des colonnes
    def _run_delete_columns(self, columns: list[str], cascade: bool) -> dict[str, bool]:
        """Run the ordered steps of a column deletion.

        For every column that exists in the fact table: children detachment (when
        ``cascade``), ``ALTER TABLE ... DROP COLUMN``, metadata row removal and
        ``cluster_by`` update. A column that fails is marked False and the loop
        goes on; only critical post-deletion validation issues abort the whole
        operation. Called from inside the transaction opened by ``delete_columns``.

        Args:
            columns: Column names requested for deletion.
            cascade: Whether to detach the children of a hierarchy parent instead
                of refusing its deletion.

        Returns:
            Dictionary mapping every requested column name to its success status.

        Raises:
            RuntimeError: If post-deletion validation finds critical issues, naming
                the step reached.
        """
        # Initialisation du dictionnaire résultat
        results: dict[str, bool] = {}

        # Filtrage des colonnes existantes
        existing_columns = self._get_fact_table_columns()
        valid_columns = [col for col in columns if col in existing_columns]

        # Vérification que les colonnes sont valides
        if not valid_columns:
            # Logging
            self.logger.warning("No valid columns found for deletion")
            return {col: False for col in columns}

        # Traitement de chaque colonne
        for column in valid_columns:
            try:
                # Suppression de la colonne et de ses références (parent_name si
                # cascade, metadata, cluster_by)
                results[column] = self._drop_column_with_references(
                    column, cascade=cascade
                )

            except Exception as e:
                # Logging
                self.logger.error(f"Error processing column {column}: {e}")
                results[column] = False
                # Continue avec les autres colonnes

        # Validation post-suppression
        if self.enable_validation and self.auditor:
            validation_report = self.auditor.validate_database(ValidationLevel.BASIC)
            critical_issues = validation_report.get_critical_issues_count()
            # Problèmes critiques : annulation de l'ensemble des suppressions
            if critical_issues > 0:
                raise RuntimeError(
                    f"post-deletion validation found {critical_issues} critical"
                    " issue(s)"
                )

        # Calcul du nombre de suppressions
        successful_deletions = sum(results.values())
        # Logging
        self.logger.info(
            f"Column deletion completed: {successful_deletions}"
            f"/{len(valid_columns)} columns deleted"
        )

        # Ajout des colonnes non trouvées au résultat
        for col in columns:
            if col not in results:
                results[col] = False

        return results

    # Méthodes d'analyse des dépendances
    # Méthode auxiliaire d'analyse des dépendances associées à une colonne
    def _analyze_column_dependencies(
        self, columns: list[str], cascade: bool = False
    ) -> dict[str, Any]:
        """Analyze column dependencies for deletion impact assessment.

        Examines each column for its categorical status, primary key status,
        hierarchy parenthood, and critical references.

        Args:
            columns: List of column names to analyze.
            cascade: Whether the caller allows detaching hierarchy children
                (``parent_name`` set to ``NULL``) instead of treating parenthood as
                a critical dependency. Defaults to False.

        Returns:
            Dependency report containing:
            - columns_analyzed: List of analyzed columns
            - has_critical_dependencies: Whether any critical deps exist
            - dependencies: Per-column dependency details
            - warnings: List of warning messages
        """
        try:
            # Initialisation du rapport
            dependency_report: dict[str, Any] = {
                "columns_analyzed": columns,
                "has_critical_dependencies": False,
                "dependencies": {},
                "warnings": [],
            }

            # Parcours des colonnes
            for column in columns:
                # Initialisation des dépendances de la colonne
                column_deps: dict[str, Any] = {
                    "is_categorical": self._is_categorical_column(column),
                    "referenced_in_queries": False,
                }

                # Avertissements
                # Signalement des colonnes exposées comme filtre dans l'interface
                if column_deps["is_categorical"]:
                    dependency_report["warnings"].append(
                        f"Column {column} is flagged as categorical and may back a"
                        f" menu in the interface"
                    )

                # Vérification si la colonne est une clé primaire (dépendance critique)
                if self._is_primary_key_column(column):
                    dependency_report["has_critical_dependencies"] = True
                    dependency_report["warnings"].append(
                        f"CRITICAL: Column {column} is a primary key - deletion will"
                        f" break data integrity"
                    )

                # Vérification si la colonne est parente d'une autre colonne dans une
                # hiérarchie : dépendance critique sauf cascade=True.
                hierarchy_children = self._get_hierarchy_children(column)
                column_deps["hierarchy_children"] = hierarchy_children
                if hierarchy_children:
                    if cascade:
                        # Warning uniquement
                        dependency_report["warnings"].append(
                            f"Column {column} is the parent of {hierarchy_children} in"
                            f" a hierarchy; cascade=True will clear their parent_name"
                        )
                    else:
                        # Dépendantce critique
                        dependency_report["has_critical_dependencies"] = True
                        # Warning
                        dependency_report["warnings"].append(
                            f"CRITICAL: Column {column} is the parent of"
                            f" {hierarchy_children} in a hierarchy - deletion would"
                            f" orphan them (use cascade=True to detach)"
                        )

                # Vérification si la colonne est visée par des colonnes de libellés :
                # dépendance critique sauf cascade=True.
                label_columns = self._get_label_columns_for_code(column)
                column_deps["label_columns"] = label_columns
                if label_columns:
                    if cascade:
                        # Warning uniquement
                        dependency_report["warnings"].append(
                            f"Column {column} is the label_for target of"
                            f" {label_columns}; cascade=True will clear their"
                            f" label_for"
                        )
                    else:
                        # Dépendance critique
                        dependency_report["has_critical_dependencies"] = True
                        # Warning
                        dependency_report["warnings"].append(
                            f"CRITICAL: Column {column} is the label_for target of"
                            f" {label_columns} - deletion would orphan them (use"
                            f" cascade=True to detach)"
                        )

                dependency_report["dependencies"][column] = column_deps

            return dependency_report

        except Exception as e:
            # Logging
            self.logger.error(f"Error analyzing column dependencies: {e}")
            return {
                "columns_analyzed": columns,
                "has_critical_dependencies": False,
                "dependencies": {},
                "warnings": [f"Dependency analysis failed: {str(e)}"],
            }

    # Méthodes publiques additionnelles
    # Méthode d'évaluation de l'impact sur la base de données d'une opération de
    # suppression
    def get_deletion_impact(
        self,
        columns: list[str] | None = None,
        filters: str | list[Any] | dict[Any, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Analyze the impact of a potential deletion operation.

        Args:
            columns: Columns to analyze for deletion impact (None for no column
                analysis)
            filters: Row filters to analyze for deletion impact (None for no row
                analysis)

        Returns:
            Dictionary containing impact analysis

        Example:
            >>> impact = deleter.get_deletion_impact(columns=['old_col'],
            filters="status = 'inactive'")
            >>> print(f"Rows affected: {impact['rows_affected']}")
            >>> print(f"Dependencies: {impact['column_dependencies']}")
        """
        try:
            # Initialisation du rapport
            impact_report: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "rows_affected": 0,
                "columns_affected": [],
                "column_dependencies": {},
                "warnings": [],
                "recommendations": [],
            }

            # Analyse de l'impact sur les lignes
            if filters is not None:
                try:
                    # Construction de la condition associée aux filtres
                    where_clause = _build_where_clause(filters)  # type: ignore[arg-type]
                    # Identification de slignes affectées
                    if where_clause:
                        count_query = (
                            f"SELECT COUNT(*) FROM {self._qualified('fact_table')}"
                            f" {where_clause}"
                        )
                        _impact_row = self.conn.execute(count_query).fetchone()
                        impact_report["rows_affected"] = (
                            _impact_row[0] if _impact_row is not None else 0
                        )
                    # Ajout d'un message
                    if impact_report["rows_affected"] > 0:
                        impact_report["warnings"].append(
                            f"{impact_report['rows_affected']} rows will be deleted"
                        )
                except Exception as e:
                    # Message de défaut
                    impact_report["warnings"].append(
                        f"Could not analyze row impact: {e}"
                    )

            # Analyse de l'impact sur les colonnes
            if columns:
                # Ajout des colonnes affectées
                impact_report["columns_affected"] = columns
                # Analyse des dépendances avec des colonnes affectées
                dependency_analysis = self._analyze_column_dependencies(columns)
                impact_report["column_dependencies"] = dependency_analysis[
                    "dependencies"
                ]
                # Ajout des avertissements
                impact_report["warnings"].extend(dependency_analysis["warnings"])

            # Génération des recommandations
            if impact_report["rows_affected"] > 1000:
                impact_report["recommendations"].append(
                    "Consider using batch processing for large row deletions"
                )

            return impact_report

        except Exception as e:
            # Logging
            self.logger.error(f"Error analyzing deletion impact: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    # Méthode d'identification du statut de la suppression de la base de données
    def get_deletion_status(self) -> dict[str, Any]:
        """
        Get the status of the database deletion system.

        Returns:
            Dictionary containing system status information

        Example:
            >>> status = deleter.get_deletion_status()
            >>> print(f"System health: {status['health_status']}")
        """
        try:
            # Initialisation du statut
            status = {
                "timestamp": datetime.now().isoformat(),
                "health_status": "unknown",
                # Toujours 0 : les transactions sont portées par DuckDB
                # (BEGIN/COMMIT par opération) et ne font plus l'objet d'un suivi
                # applicatif. Clé conservée pour la stabilité du dictionnaire.
                "active_transactions": 0,
                "validation_enabled": self.enable_validation,
                "auto_cleanup": self.auto_cleanup,
            }

            # Vérification de la santé de la base de données
            if self.auditor:
                health_check = self.auditor.get_quick_health_check()
                status["health_status"] = health_check.get("status", "unknown")
                status["database_info"] = health_check

            # Statistiques de la fact table
            if self._table_exists("fact_table"):
                table_stats = self.data_mgr.get_table_stats()
                status["fact_table_stats"] = table_stats

            return status

        except Exception as e:
            # Logging
            self.logger.error(f"Error getting deletion status: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    # Méthode de nettoyage de la base de données
    def cleanup_database(self) -> dict[str, Any]:
        """
        Drop the fact table columns that only hold null values.

        Delegates to ``_cleanup_null_only_columns`` inside a single transaction:
        metadata rows and ``cluster_by`` are updated alongside; primary keys,
        hierarchy parents that still have children and every column of an empty
        fact table are kept.

        Returns:
            Dictionary with cleanup results (``null_columns``: dropped columns), or
            ``{"error": ...}`` on failure (nothing is then dropped).

        Example:
            >>> results = deleter.cleanup_database()
            >>> print(f"Cleaned: {results}")
        """
        try:
            return {"null_columns": self._cleanup_null_only_columns()}

        except Exception as e:
            # Logging
            self.logger.error(f"Error during database cleanup: {e}")
            return {"error": str(e)}
