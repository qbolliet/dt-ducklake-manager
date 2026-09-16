# Importation des modules
# Modules de base
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

# DuckDB
import duckdb
import narwhals as nw

# Import des utilitaires
from ..utils.logger import _init_logger
from ..utils.sql import SchemaScoped, quote_ident, resolve_catalog
from ..utils.types import METADATA_COLUMNS, metadata_table_ddl

# Import des gestionnaires
from .auditor import DatabaseAuditor, ValidationIssue, ValidationLevel

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe des types de stratégies de récupération possibles
class RecoveryStrategy(Enum):
    """Available recovery strategies for database restoration.

    DuckLake retains a complete snapshot history. The recommended strategy for
    handling data corruption is ``USE_SNAPSHOT_HISTORY``: it lists all available
    snapshots with their timestamps and returns step-by-step instructions to
    reopen a time-travel connection via
    ``DuckLakeConnector(..., snapshot_version=N)`` and reload the data.

    For structural or consistency issues that do not require rolling back data,
    the in-place repair strategies (``REPAIR_SCHEMA``,
    ``CLEAN_ORPHANED_DATA``, ``VALIDATE_AND_FIX``) operate directly on the live
    catalog without touching the snapshot history.
    """

    USE_SNAPSHOT_HISTORY = "use_snapshot_history"
    REPAIR_SCHEMA = "repair_schema"
    CLEAN_ORPHANED_DATA = "clean_orphaned_data"
    VALIDATE_AND_FIX = "validate_and_fix"


# Classe d'opération de récupération
@dataclass
class RecoveryOperation:
    """
    Recovery operation to execute.

    Attributes:
        strategy (RecoveryStrategy): Recovery strategy to use
        target_recovery_point (Optional[str]): Target DuckLake ``snapshot_id``
            (as a string) for ``USE_SNAPSHOT_HISTORY``
        parameters (dict): Strategy-specific parameters
        auto_validate (bool): Whether to automatically validate after recovery
        description (str): Description of the operation
    """

    strategy: RecoveryStrategy
    target_recovery_point: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    auto_validate: bool = True
    description: str = ""


# Classe de résultat de récupération
@dataclass
class RecoveryResult:
    """
    Result of a recovery operation.

    Attributes:
        success (bool): Whether the operation succeeded
        strategy_used (RecoveryStrategy): Strategy that was used
        recovery_time (float): Recovery duration in seconds
        operations_performed (List[str]): List of operations performed
        validation_report (Optional[ValidationReport]): Post-recovery validation report
        error_message (Optional[str]): Error message if applicable
        recommendations (List[str]): Recommendations to avoid future issues
    """

    success: bool
    strategy_used: RecoveryStrategy
    recovery_time: float
    operations_performed: list[str] = field(default_factory=list)
    validation_report: Any | None = None
    error_message: str | None = None
    recommendations: list[str] = field(default_factory=list)


# Classe de récupération de la base de données
class DatabaseRecoveryManager(SchemaScoped):
    """
    Manages database recovery, built on DuckLake time travel.

    Recovery from a bad write is **not** an application-level restore: DuckLake
    persists the full snapshot history, so the mechanism is to list the snapshots
    (:meth:`list_ducklake_snapshots`), pick one, and reopen the catalog on it via
    ``DuckLakeConnector(..., snapshot_version=N)`` — the procedure spelled out by
    :attr:`RecoveryStrategy.USE_SNAPSHOT_HISTORY`. No backup file is written, and
    none is needed: a failed operation is already rolled back by its own DuckDB
    transaction.

    The remaining strategies (``REPAIR_SCHEMA``, ``CLEAN_ORPHANED_DATA``,
    ``VALIDATE_AND_FIX``) repair structural or consistency issues in place, on the
    live catalog, without touching the snapshot history.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection
        catalog_alias (str): Alias of the attached DuckLake catalog
        schema (str): DuckLake schema to recover
        auditor (DatabaseAuditor): Database auditor for validation
        logger: Logger instance for recovery tracking
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        log_filename: str | os.PathLike[str] | None = None,
        catalog_alias: str = "db",
        schema: str = "main",
    ):
        """
        Initialize the database recovery manager.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            log_filename: Path to log file.
            catalog_alias: Alias of the attached DuckLake catalog, used to query
                available snapshots via ``ducklake_snapshots()``. Defaults to ``'db'``.
            schema: DuckLake schema to recover. Used for snapshot queries and to
                qualify the result set's tables. A catalog can host several schemas.
                Defaults to ``'main'``.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> recovery_mgr = DatabaseRecoveryManager(conn, schema='predictions')
        """
        # Initialisation de la connexion DuckLake.
        self.conn = connection if connection is not None else duckdb.connect(":memory:")
        self.catalog_alias = catalog_alias
        self.schema = schema

        # Alias de catalogue effectif : qualification par le catalogue uniquement
        # s'il est réellement attaché (None pour les connexions in-memory des tests).
        self._catalog = resolve_catalog(self.conn, self.catalog_alias)

        # Initialisation des composants
        self.auditor = DatabaseAuditor(
            connection,
            log_filename,
            schema=schema,
            catalog_alias=catalog_alias,
        )

        # Initialisation du logger nommé.
        # Chemin par défaut centralisé dans utils.logger : <cwd>/logs/<name>.log.
        self.logger = _init_logger(filename=log_filename, name="database_recovery")

    # Méthodes de récupération
    # Méthode de récupération de la base de données
    def recover_database(
        self, operation: RecoveryOperation, confirm_destructive: bool = False
    ) -> RecoveryResult:
        """
        Perform database recovery using the specified operation.

        Args:
            operation: Recovery operation to perform
            confirm_destructive: Confirm destructive operations (required for some
                strategies)

        Returns:
            RecoveryResult with operation details

        Example:
            >>> op = RecoveryOperation(
            ...     strategy=RecoveryStrategy.RESTORE_BACKUP,
            ...     target_recovery_point="17",
            ...     description="Restore from snapshot 17 after corruption"
            ... )
            >>> result = recovery_mgr.recover_database(op, confirm_destructive=True)
        """
        # Timestamp du début de l'opération
        start_time = time.time()

        try:
            # Logging
            self.logger.info(f"Starting recovery operation: {operation.strategy.value}")

            # Validation de l'opération
            if not self._validate_recovery_operation(operation, confirm_destructive):
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    error_message="Recovery operation validation failed",
                )

            # Aucune sauvegarde applicative avant une réparation in-place :
            # l'état antérieur reste accessible par time travel (snapshot courant
            # relevé avant l'opération via list_ducklake_snapshots).

            # Exécution de la stratégie de récupération
            if operation.strategy == RecoveryStrategy.USE_SNAPSHOT_HISTORY:
                result = self._recover_use_snapshot_history(operation)
            elif operation.strategy == RecoveryStrategy.REPAIR_SCHEMA:
                result = self._recover_repair_schema(operation)
            elif operation.strategy == RecoveryStrategy.CLEAN_ORPHANED_DATA:
                result = self._recover_clean_orphaned_data(operation)
            elif operation.strategy == RecoveryStrategy.VALIDATE_AND_FIX:
                result = self._recover_validate_and_fix(operation)
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    error_message=f"Unknown recovery strategy: {operation.strategy}",
                )

            # Calcul du temps de récupération
            recovery_time = time.time() - start_time
            result.recovery_time = recovery_time

            # Validation post-récupération si demandée
            if result.success and operation.auto_validate:
                validation_report = self.auditor.validate_database(
                    ValidationLevel.STANDARD
                )
                result.validation_report = validation_report

                # Vérification des issues critiques persistantes
                if validation_report.get_critical_issues_count() > 0:
                    result.recommendations.append(
                        f"Recovery completed but"
                        f" {validation_report.get_critical_issues_count()} critical"
                        f" issues remain"
                    )

            # Logging du résultat
            if result.success:
                self.logger.info(
                    f"Recovery operation completed successfully in {recovery_time:.2f}s"
                )
            else:
                self.logger.error(f"Recovery operation failed: {result.error_message}")

            return result

        except Exception as e:
            # Calcul du temps de récupération
            recovery_time = time.time() - start_time
            # Logging
            self.logger.error(f"Error during recovery operation: {e}")

            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=recovery_time,
                error_message=str(e),
            )

    # Méthode de récupération automatique de la base de données sur la base d'un rapport
    # de validation
    def auto_recover_from_validation(
        self,
        validation_report: Any,
        max_attempts: int = 3,
        allow_destructive: bool = False,
    ) -> RecoveryResult:
        """
        Automatically recover from issues found in validation report.

        Args:
            validation_report: ValidationReport with issues to fix
            max_attempts: Maximum number of recovery attempts
            allow_destructive: Whether to allow destructive recovery operations

        Returns:
            RecoveryResult with recovery details

        Example:
            >>> report = auditor.validate_database()
            >>> if report.get_critical_issues_count() > 0:
            ...     result = recovery_mgr.auto_recover_from_validation(report)
        """
        try:
            # Logging
            self.logger.info("Starting auto-recovery from validation issues")

            # Analyse des problèmes pour déterminer la stratégie de récupération
            strategy = self._determine_recovery_strategy(
                validation_report, allow_destructive
            )

            # Renvoi un message si aucune stratégie n'a été trouvée
            if strategy is None:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.VALIDATE_AND_FIX,
                    recovery_time=0,
                    error_message="No suitable recovery strategy found",
                )

            # Tentatives de récupération
            for attempt in range(max_attempts):
                # Logging
                self.logger.info(f"Auto-recovery attempt {attempt + 1}/{max_attempts}")

                # Création de l'opération de récupération
                operation = RecoveryOperation(
                    strategy=strategy,
                    parameters={"validation_report": validation_report},
                    auto_validate=True,
                    description=f"Auto-recovery attempt {attempt + 1}",
                )

                # Exécution de la récupération
                result = self.recover_database(
                    operation, confirm_destructive=allow_destructive
                )

                if result.success:
                    # Vérification que les issues ont été résolues
                    if (
                        result.validation_report
                        and result.validation_report.get_critical_issues_count() == 0
                    ):
                        # Logging
                        self.logger.info("Auto-recovery completed successfully")
                        return result
                    else:
                        # Logging
                        self.logger.warning(
                            "Recovery completed but issues persist, trying again"
                        )
                else:
                    # Logging
                    self.logger.warning(
                        f"Recovery attempt {attempt + 1} failed: {result.error_message}"
                    )

            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                recovery_time=0,
                error_message=f"Auto-recovery failed after {max_attempts} attempts",
            )

        except Exception as e:
            # Logging
            self.logger.error(f"Error during auto-recovery: {e}")

            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.VALIDATE_AND_FIX,
                recovery_time=0,
                error_message=str(e),
            )

    # Méthodes de récupération spécialisées
    # Méthode de récupération par historique des snapshots DuckLake
    def _recover_use_snapshot_history(
        self, operation: RecoveryOperation
    ) -> RecoveryResult:
        """Guide time-travel restoration using DuckLake's native snapshot history.

        DuckLake does not support multi-statement SQL ROLLBACK. Recovery relies on
        native time travel: open a read-only connection pinned to a previous snapshot
        via ``DuckLakeConnector(..., snapshot_version=N)``, read the tables from that
        connection, and reinsert the data into the current catalog.

        This method:
        1. Lists all available snapshots with their IDs and timestamps.
        2. Validates the requested target snapshot when ``target_recovery_point``
           is provided (expected to be a snapshot_id as a string).
        3. Suggests the second-to-last snapshot as a safe restore point when no
           target is specified.
        4. Returns step-by-step restoration instructions in ``recommendations``.

        Args:
            operation: Recovery operation. ``target_recovery_point`` may contain a
                snapshot_id (as a string) to pinpoint the desired restore point.

        Returns:
            RecoveryResult with the snapshot inventory in ``operations_performed``
            and a step-by-step restoration guide in ``recommendations``.
        """
        try:
            # Initialisation de la liste des opérations
            operations_performed: list[str] = []

            # Interrogation de l'historique des snapshots DuckLake.
            try:
                snapshots_df = nw.from_native(
                    self.conn.execute(
                        f"SELECT * FROM ducklake_snapshots('{self.catalog_alias}')"
                        " ORDER BY snapshot_id DESC"
                    ).to_arrow_table(),
                    eager_only=True,
                )
            except Exception as e:
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    error_message=f"Impossible d'interroger les snapshots DuckLake :"
                    f"{e}",
                )

            # Comptage des snapshots
            snapshot_count = len(snapshots_df)
            # Ajout à la liste des opérations
            operations_performed.append(
                f"{snapshot_count} snapshot(s) disponible(s) dans le catalogue "
                f"'{self.catalog_alias}.{self.schema}'"
            )

            # Présentation structurée de chaque snapshot avec horodatage lisible
            for row in snapshots_df.iter_rows(named=True):
                # Extraction de l'identifiant du snapshot
                snap_id = row.get("snapshot_id", "N/A")
                # Récupération de l'horodatage selon le nom de colonne retourné par
                # DuckLake
                snap_time = row.get("snapshot_time", row.get("timestamp", None))
                if snap_time is not None and hasattr(snap_time, "strftime"):
                    snap_time_str = snap_time.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    snap_time_str = str(snap_time) if snap_time is not None else "N/A"
                # Ajout à la liste des opérations
                operations_performed.append(
                    f"  snapshot_id={snap_id}  |  {snap_time_str}"
                )

            # Détermination du snapshot cible : validé si fourni, suggéré sinon
            target_snapshot = operation.target_recovery_point
            suggested_id: str = "<snapshot_id>"

            if target_snapshot:
                # Vérification que le snapshot demandé figure bien dans l'historique
                snapshot_ids = (
                    [str(v) for v in snapshots_df["snapshot_id"].to_list()]
                    if snapshot_count > 0
                    else []
                )
                if target_snapshot in snapshot_ids:
                    operations_performed.append(
                        f"Snapshot cible validé : {target_snapshot}"
                    )
                    suggested_id = target_snapshot
                else:
                    operations_performed.append(
                        f"ATTENTION : snapshot {target_snapshot} introuvable dans"
                        f" l'historique — "
                        f"vérifier la valeur de snapshot_id"
                    )
            elif snapshot_count > 1:
                # Avant-dernier snapshot : état stable avant la dernière écriture
                suggested_id = str(snapshots_df["snapshot_id"][1])
                operations_performed.append(
                    f"Snapshot suggéré (avant-dernier, état stable) : {suggested_id}"
                )
            elif snapshot_count == 1:
                suggested_id = str(snapshots_df["snapshot_id"][0])
                operations_performed.append(
                    "Un seul snapshot disponible — restauration vers l'état initial"
                    " uniquement"
                )

            # Instructions de restauration étape par étape (format prêt à copier-coller)
            recommendations = [
                "─── Procédure de restauration par time-travel DuckLake ───",
                "",
                "Étape 1 — Ouvrir une connexion en lecture seule sur le snapshot cible"
                ":",
                f"  conn_old = DuckLakeConnector(catalog_path, data_path,"
                f" schema='{self.schema}',"
                f" snapshot_version={suggested_id}).connect()",
                "",
                "Étape 2 — Lire les tables depuis cette connexion :",
                "  fact_df = conn_old.execute('SELECT * FROM fact_table')"
                ".to_arrow_table()",
                "  meta_df = conn_old.execute('SELECT * FROM metadata')"
                ".to_arrow_table()",
                "  ds_df   = conn_old.execute('SELECT * FROM dataset_metadata')"
                ".to_arrow_table()",
                "",
                "Étape 3 — Vider les tables du catalogue courant et réinsérer les"
                " données :",
                "  conn.execute('DELETE FROM fact_table')",
                "  conn.register('_restore_fact', fact_df)",
                "  conn.execute('INSERT INTO fact_table SELECT * FROM _restore_fact')",
                "  conn.execute('DROP VIEW _restore_fact')",
                "  # Répéter pour metadata et dataset_metadata",
                "",
                "Étape 4 — Valider l'intégrité après restauration :",
                f"  auditor = DatabaseAuditor(conn, schema='{self.schema}')",
                "  report  = auditor.validate_database(ValidationLevel.COMPREHENSIVE)",
                "  print(report)",
            ]

            return RecoveryResult(
                success=True,
                strategy_used=operation.strategy,
                recovery_time=0,
                operations_performed=operations_performed,
                recommendations=recommendations,
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e),
            )

    # Méthode publique de consultation de l'historique des snapshots DuckLake
    def list_ducklake_snapshots(self) -> nw.DataFrame[Any] | None:
        """Return the full DuckLake snapshot history as a narwhals DataFrame.

        Convenience wrapper around the ``ducklake_snapshots(catalog)`` table
        function, and the entry point of the recovery procedure: pick a
        ``snapshot_id`` here, then reopen the catalog on it with
        ``DuckLakeConnector(..., snapshot_version=N)`` and copy the data back.
        The history is catalog-wide, covering every schema it holds.

        Returns:
            narwhals DataFrame (pyarrow backend, ``.to_native()`` gives the
            ``pyarrow.Table``) with one row per snapshot (columns depend on the
            DuckLake version), sorted by ``snapshot_id`` descending.
            Returns None if the catalog cannot be queried (e.g. a plain in-memory
            connection with no DuckLake catalog attached).

        Example:
            >>> snapshots = recovery_mgr.list_ducklake_snapshots()
            >>> if snapshots is not None:
            ...     print(snapshots)
        """
        try:
            # Inventaire des snapshots
            snapshots = self.conn.execute(
                f"SELECT * FROM ducklake_snapshots('{self.catalog_alias}')"
                " ORDER BY snapshot_id DESC"
            ).to_arrow_table()
            history: nw.DataFrame[Any] = nw.from_native(snapshots, eager_only=True)
            return history
        except Exception as e:
            # Logging d'erreur
            self.logger.error(f"Impossible de lister les snapshots DuckLake : {e}")
            return None

    # Méthode de récupération par réparation du schéma
    def _recover_repair_schema(self, operation: RecoveryOperation) -> RecoveryResult:
        """Recover by repairing database schema issues.

        Args:
            operation: Recovery operation parameters.

        Returns:
            RecoveryResult with details of schema repairs attempted.
        """
        try:
            # Initialisation de la liste des opération appliquées
            operations_performed = []

            # Validation du schéma actuel
            validation_report = self.auditor.validate_database(
                ValidationLevel.COMPREHENSIVE
            )

            # Réparation des problèmes du schéma
            schema_issues = [
                issue
                for issue in validation_report.issues
                if "schema" in issue.issue_type.value.lower()
            ]

            # Parcours des problèmes
            for issue in schema_issues:
                try:
                    # Application du fix automatique si possible
                    fixed = self._apply_schema_fix(issue)
                    if fixed:
                        operations_performed.append(f"Fixed: {issue.description}")
                    else:
                        operations_performed.append(
                            f"Could not auto-fix: {issue.description}"
                        )
                except Exception as e:
                    operations_performed.append(
                        f"Failed to fix: {issue.description} - {e}"
                    )

            return RecoveryResult(
                success=True,
                strategy_used=operation.strategy,
                recovery_time=0,
                operations_performed=operations_performed,
                recommendations=[
                    "Manual schema review may be required for complex issues"
                ],
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e),
            )

    # Méthode d'application des corrections de schéma automatiques
    def _apply_schema_fix(self, issue: ValidationIssue) -> bool:
        """
        Apply automatic fix for schema issues where possible.

        Args:
            issue: The validation issue to fix

        Returns:
            True if fix was applied successfully, False otherwise
        """
        try:
            # Cas 1: Table metadata manquante → création
            if (
                "missing" in issue.description.lower()
                and issue.table_name == "metadata"
            ):
                self.conn.execute(
                    metadata_table_ddl(self._qualified("metadata"), if_not_exists=True)
                )
                self.logger.info("Created missing metadata table")
                return True

            # Cas 2: Colonnes manquantes dans metadata → ajout avec valeurs par défaut
            if "Missing required columns in metadata" in issue.description:
                return self._add_missing_metadata_columns(issue)

            # Autres cas: pas de fix automatique possible
            return False

        except Exception as e:
            self.logger.error(f"Error applying schema fix: {e}")
            return False

    # Méthode auxiliaire pour ajouter les colonnes manquantes à metadata
    def _add_missing_metadata_columns(self, issue: ValidationIssue) -> bool:
        """
        Add missing columns to metadata table.

        Args:
            issue: The validation issue containing missing column info

        Returns:
            True if columns were added successfully
        """
        try:
            # Colonnes requises pour la table metadata
            required_columns = METADATA_COLUMNS

            # Nom qualifié de la table de métadonnées
            metadata_table = self._qualified("metadata")

            # Récupération des colonnes existantes
            existing_columns = set()
            try:
                result = self.conn.execute(f"DESCRIBE {metadata_table}").fetchall()
                existing_columns = {row[0] for row in result}
            except Exception:
                pass

            # Ajout des colonnes manquantes
            for col_name, col_type in required_columns.items():
                if col_name not in existing_columns:
                    self.conn.execute(
                        f"ALTER TABLE {metadata_table} ADD COLUMN"
                        f" {quote_ident(col_name)} {col_type}"
                    )
                    self.logger.info(
                        f"Added missing column {col_name} to metadata table"
                    )

            return True

        except Exception as e:
            self.logger.error(f"Error adding missing metadata columns: {e}")
            return False

    # Méthode de nettoyage des données orphelines
    def _recover_clean_orphaned_data(
        self, operation: RecoveryOperation
    ) -> RecoveryResult:
        """Recover by cleaning orphaned data from the database.

        Args:
            operation: Recovery operation parameters.

        Returns:
            RecoveryResult with details of cleanup operations performed.
        """
        try:
            # Initialisation de la liste des opérations appliquées au jeu de données
            operations_performed = []

            # Import local pour éviter l'import circulaire :
            # recovery → operations/__init__ → atomic → recovery
            from ..operations.deleter import DatabaseDeleter

            # Utilisation du deleter pour nettoyer
            deleter = DatabaseDeleter(
                self.conn,
                enable_validation=False,
                auto_cleanup=True,
                schema=self.schema,
                catalog_alias=self.catalog_alias,
            )
            # Nettoyage de la base de données
            cleanup_results = deleter.cleanup_database()
            # Parcours des résultats
            for category, result in cleanup_results.items():
                if result:
                    operations_performed.append(f"Cleaned {category}: {result}")

            return RecoveryResult(
                success=True,
                strategy_used=operation.strategy,
                recovery_time=0,
                operations_performed=operations_performed,
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e),
            )

    # Méthode de récupération en validant la base de données et appliquant des
    # corrections automatiques
    def _recover_validate_and_fix(self, operation: RecoveryOperation) -> RecoveryResult:
        """Recover by validating database and applying automatic fixes.

        Args:
            operation: Recovery operation parameters.

        Returns:
            RecoveryResult with validation findings and fixes applied.
        """
        try:
            # Initialisation de la liste des opérations appliquées
            operations_performed = []

            # Validation complète
            validation_report = self.auditor.validate_database(
                ValidationLevel.COMPREHENSIVE
            )
            operations_performed.append(
                f"Validated database: found {len(validation_report.issues)} issues"
            )

            # Tentatives de correction automatique des issues
            fixed_count = 0

            # Parcours des problèmes
            for issue in validation_report.issues:
                try:
                    # Correction basée sur le type d'issue
                    if issue.issue_type.value == "data_integrity":
                        # Correction des problèmes d'intégrité des données
                        if "null values" in issue.description.lower():
                            fixed = self._fix_null_value_issue(issue)
                            if fixed:
                                operations_performed.append(
                                    f"Fixed null value issue: {issue.description}"
                                )
                                fixed_count += 1

                    elif issue.issue_type.value == "missing_metadata":
                        # Correction des problèmes de métadonnées manquantes
                        fixed = self._fix_missing_metadata_issue(issue)
                        if fixed:
                            operations_performed.append(
                                f"Fixed missing metadata: {issue.description}"
                            )
                            fixed_count += 1

                    elif issue.issue_type.value == "type_mismatch":
                        # Correction des conflits de types
                        fixed = self._fix_type_mismatch_issue(issue)
                        if fixed:
                            operations_performed.append(
                                f"Fixed type mismatch: {issue.description}"
                            )
                            fixed_count += 1

                    elif issue.issue_type.value == "constraint_violation":
                        # Correction des violations de contraintes
                        fixed = self._fix_constraint_violation_issue(issue)
                        if fixed:
                            operations_performed.append(
                                f"Fixed constraint violation: {issue.description}"
                            )
                            fixed_count += 1

                    elif issue.issue_type.value == "performance_issue":
                        # Pas de fix automatique pour les problèmes de performance,
                        # juste une recommandation
                        operations_performed.append(
                            f"Performance recommendation: {issue.suggested_fix}"
                        )

                except Exception as e:
                    operations_performed.append(
                        f"Failed to fix issue: {issue.description} - {e}"
                    )

            operations_performed.append(
                f"Successfully fixed {fixed_count}/{len(validation_report.issues)}"
                f" issues"
            )

            return RecoveryResult(
                success=True,
                strategy_used=operation.strategy,
                recovery_time=0,
                operations_performed=operations_performed,
                recommendations=[
                    "Re-run validation to verify all fixes were applied correctly"
                ],
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e),
            )

    # Méthodes utilitaires privées
    # Méthode de validation d'une opération de
    def _validate_recovery_operation(
        self, operation: RecoveryOperation, confirm_destructive: bool
    ) -> bool:
        """Validate a recovery operation before execution.

        Args:
            operation: Recovery operation to validate.
            confirm_destructive: Whether destructive operations are confirmed.

        Returns:
            True if operation is valid and can proceed, False otherwise.
        """
        try:
            # Vérification des opérations destructrices (modifications in-place du
            # catalogue)
            destructive_strategies = [RecoveryStrategy.REPAIR_SCHEMA]
            # Vérification de la confirmation d'une opération destructive de données
            if operation.strategy in destructive_strategies and not confirm_destructive:
                # Logging
                self.logger.error(
                    f"Destructive operation {operation.strategy.value} requires"
                    f" confirmation"
                )
                return False

            # L'existence du snapshot cible est vérifiée par la stratégie
            # elle-même (_recover_use_snapshot_history), qui dispose de
            # l'historique complet.

            return True

        except Exception as e:
            # Logging
            self.logger.error(f"Error validating recovery operation: {e}")
            return False

    # Méthode de détermination de la stratégie de récupération
    def _determine_recovery_strategy(
        self, validation_report: Any, allow_destructive: bool
    ) -> RecoveryStrategy | None:
        """
        Determine best recovery strategy based on issue analysis.

        Args:
            validation_report: The validation report with issues
            allow_destructive: Whether destructive operations are allowed

        Returns:
            The recommended recovery strategy or None if error
        """
        try:
            # Scores de sévérité pour le calcul de priorité
            severity_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}

            # Comptage par type et sévérité
            issue_counts: dict[str, dict[str, int]] = {}

            for issue in validation_report.issues:
                issue_type = issue.issue_type.value
                severity = issue.severity.value

                if issue_type not in issue_counts:
                    issue_counts[issue_type] = {"count": 0, "score": 0}

                issue_counts[issue_type]["count"] += 1
                issue_counts[issue_type]["score"] += severity_scores.get(severity, 1)

            # Tri par score décroissant pour identifier le problème dominant
            sorted_issues = sorted(
                issue_counts.items(), key=lambda x: x[1]["score"], reverse=True
            )

            if not sorted_issues:
                return RecoveryStrategy.VALIDATE_AND_FIX

            dominant_issue = sorted_issues[0][0]
            critical_count = validation_report.get_critical_issues_count()

            # Stratégie basée sur le problème dominant avec seuils de gravité
            # Si trop de problèmes critiques → time-travel via l'historique DuckLake
            # (préférable à une restauration CSV qui ne couvre pas les données de faits)
            if critical_count >= 5:
                return RecoveryStrategy.USE_SNAPSHOT_HISTORY

            # Mapping type de problème → stratégie avec prise en compte du score
            strategy_mapping = {
                "schema_inconsistency": RecoveryStrategy.REPAIR_SCHEMA
                if allow_destructive
                else RecoveryStrategy.VALIDATE_AND_FIX,
                "missing_metadata": RecoveryStrategy.VALIDATE_AND_FIX,
                "data_integrity": RecoveryStrategy.VALIDATE_AND_FIX,
                "type_mismatch": RecoveryStrategy.VALIDATE_AND_FIX,
                "constraint_violation": RecoveryStrategy.VALIDATE_AND_FIX,
            }

            return strategy_mapping.get(
                dominant_issue, RecoveryStrategy.VALIDATE_AND_FIX
            )

        except Exception as e:
            # Logging
            self.logger.error(f"Error determining recovery strategy: {e}")
            return None

    # Méthode auxiliaire de correction d'un problème de valeur nulle
    def _fix_null_value_issue(self, issue: ValidationIssue) -> bool:
        """
        Fix null value issues by deleting affected rows.

        Args:
            issue: The validation issue to fix

        Returns:
            True if fix was applied successfully
        """
        try:
            col_name = issue.column_name
            table_name = issue.table_name

            if not col_name or table_name != "fact_table":
                return False

            # Nom qualifié de la table des faits
            fact_table = self._qualified("fact_table")

            # Comptage des lignes avant suppression
            _r1 = self.conn.execute(f"SELECT COUNT(*) FROM {fact_table}").fetchone()
            initial_count = _r1[0] if _r1 is not None else 0

            # Suppression des lignes contenant des valeurs nulles
            self.conn.execute(
                f"DELETE FROM {fact_table} WHERE {quote_ident(col_name)} IS NULL"
            )

            # Comptage des lignes après suppression
            _r2 = self.conn.execute(f"SELECT COUNT(*) FROM {fact_table}").fetchone()
            final_count = _r2[0] if _r2 is not None else 0
            deleted_count = initial_count - final_count

            self.logger.info(
                f"Deleted {deleted_count} rows with NULL values in {col_name}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to fix null values: {e}")
            return False

    # Méthodes de correction des problèmes de métadonnées
    def _fix_missing_metadata_issue(self, issue: ValidationIssue) -> bool:
        """
        Fix missing metadata issues.

        Args:
            issue: The validation issue to fix

        Returns:
            True if fix was applied successfully
        """
        try:
            # Cas 1: Colonne dans fact_table mais pas dans metadata → ajout aux
            # métadonnées
            if issue.table_name == "fact_table" and issue.column_name:
                col_name = issue.column_name
                # Inférence du type depuis la colonne
                result = self.conn.execute(
                    f"DESCRIBE {self._qualified('fact_table')}"
                ).fetchall()
                col_info = next((row for row in result if row[0] == col_name), None)
                if col_info:
                    sql_type = col_info[1]
                    self.conn.execute(
                        f"""
                        INSERT INTO {self._qualified("metadata")} (name, label,
                        sql_type, is_categorical, is_primary_key)
                        VALUES (?, ?, ?, FALSE, FALSE)
                    """,
                        [
                            col_name,
                            col_name.replace("_", " ").title(),
                            sql_type,
                        ],
                    )
                    self.logger.info(f"Added missing metadata for column {col_name}")
                    return True

            # Cas 2: Colonne dans metadata mais pas dans fact_table → suppression des
            # métadonnées
            if issue.table_name == "metadata" and issue.column_name:
                self.conn.execute(
                    f"DELETE FROM {self._qualified('metadata')} WHERE name = ?",
                    [issue.column_name],
                )
                self.logger.info(
                    f"Removed orphaned metadata for column {issue.column_name}"
                )
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to fix missing metadata: {e}")
            return False

    # Méthode de correction d'une incohérence de typage
    def _fix_type_mismatch_issue(self, issue: ValidationIssue) -> bool:
        """
        Fix type mismatch issues by casting or widening column type.

        Args:
            issue: The validation issue to fix

        Returns:
            True if fix was applied successfully
        """
        try:
            col_name = issue.column_name
            if not col_name:
                return False

            # Alignement de la méta-donnée sur le type physique réel de la colonne.
            # La table des faits fait foi : écrire un type arbitraire dans metadata
            # fabriquerait précisément l'incohérence que l'auditeur signale.
            structure = self.conn.execute(
                f"DESCRIBE {self._qualified('fact_table')}"
            ).fetchall()
            col_info = next((row for row in structure if row[0] == col_name), None)
            if col_info is None:
                return False
            actual_sql_type = col_info[1]

            self.conn.execute(
                f"""
                UPDATE {self._qualified("metadata")}
                SET sql_type = ?
                WHERE name = ?
            """,
                [actual_sql_type, col_name],
            )

            self.logger.info(
                f"Aligned metadata type for column {col_name} on"
                f" fact_table ({actual_sql_type})"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to fix type mismatch: {e}")
            return False

    # Méthode de correction d'une violation de contrainte sur les données
    def _fix_constraint_violation_issue(self, issue: ValidationIssue) -> bool:
        """
        Fix constraint violation issues (e.g., remove duplicates).

        Args:
            issue: The validation issue to fix

        Returns:
            True if fix was applied successfully
        """
        try:
            # Traitement des violations de contraintes (doublons, valeurs invalides,
            # etc.)
            if "duplicate" in issue.description.lower():
                # Nom qualifié de la table des faits
                fact_table = self._qualified("fact_table")
                # Suppression des doublons en gardant la première occurrence
                self.conn.execute(f"""
                    CREATE OR REPLACE TABLE {fact_table} AS
                    SELECT * FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY * ORDER BY 1) as rn
                        FROM {fact_table}
                    ) WHERE rn = 1
                """)
                # Suppression de la colonne temporaire
                self.conn.execute(f"ALTER TABLE {fact_table} DROP COLUMN rn")
                self.logger.info("Removed duplicate rows from fact_table")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to fix constraint violation: {e}")
            return False
