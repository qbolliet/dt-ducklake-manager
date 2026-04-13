# Importation des modules
# Modules de base
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# DuckDB
import duckdb
import narwhals as nw
from narwhals.typing import IntoDataFrame

# Import des gestionnaires
from .._internal.managers.transaction import (
    TransactionManager,
    TransactionOperation,
)
from ..maintenance.auditor import DatabaseAuditor, ValidationLevel
from ..maintenance.recovery import (
    DatabaseRecoveryManager,
    RecoveryOperation,
    RecoveryStrategy,
)

# Import des utilitaires
from ..utils.logger import _init_logger
from .deleter import DatabaseDeleter
from .updater import DatabaseUpdater

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Type d'opération
class AtomicOperationType(Enum):
    """Available types of atomic database operations."""

    BATCH_UPDATE = "batch_update"
    SCHEMA_MIGRATION = "schema_migration"
    DATA_MIGRATION = "data_migration"
    CLEANUP_OPERATION = "cleanup_operation"
    VALIDATION_AND_REPAIR = "validation_and_repair"
    BACKUP_AND_UPDATE = "backup_and_update"


# Configuration de l'opération
@dataclass
class AtomicOperationConfig:
    """
    Configuration for atomic database operations.

    Attributes:
        operation_type (AtomicOperationType): Type of atomic operation
        enable_backup (bool): Whether to create backup before operation
        enable_validation (bool): Whether to validate before and after
        rollback_on_validation_fail (bool): Whether to rollback if post-validation fails
        max_retry_attempts (int): Maximum number of retry attempts
        batch_size (Optional[int]): Batch size for data processing
        parallel_workers (Optional[int]): Number of parallel workers
        timeout_seconds (Optional[int]): Operation timeout in seconds
    """

    operation_type: AtomicOperationType
    enable_backup: bool = True
    enable_validation: bool = True
    rollback_on_validation_fail: bool = True
    max_retry_attempts: int = 1
    batch_size: int | None = None
    parallel_workers: int | None = None
    timeout_seconds: int | None = None


# Résultat de l'opération
@dataclass
class AtomicOperationResult:
    """
    Result of an atomic database operation.

    Attributes:
        success (bool): Whether the operation succeeded
        operation_type (AtomicOperationType): Type of operation performed
        execution_time (float): Total execution time in seconds
        backup_created (Optional[str]): ID of backup created (if any)
        validation_passed (bool): Whether validation passed
        operations_performed (List[str]): List of individual operations performed
        error_message (Optional[str]): Error message if operation failed
        rollback_performed (bool): Whether rollback was performed
        recommendations (List[str]): Recommendations for future operations
        metrics (Dict[str, Any]): Additional metrics and statistics
    """

    success: bool
    operation_type: AtomicOperationType
    execution_time: float
    backup_created: str | None = None
    validation_passed: bool = True
    operations_performed: list[str] | None = None
    error_message: str | None = None
    rollback_performed: bool = False
    recommendations: list[str] | None = None
    metrics: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.operations_performed is None:
            self.operations_performed = []
        if self.recommendations is None:
            self.recommendations = []
        if self.metrics is None:
            self.metrics = {}


# Classe d'exécution d'opération atomiques sur la base de données
class AtomicDatabaseOperations:
    """
    Provides high-level atomic operations for database management.

    Combines transaction management, validation, backup/recovery, and specialized
    managers to provide guaranteed atomic operations with proper error handling
    and state consistency.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection
        transaction_mgr (TransactionManager): Transaction manager
        updater (DatabaseUpdater): Database updater
        deleter (DatabaseDeleter): Database deleter
        recovery_mgr (DatabaseRecoveryManager): Recovery manager
        auditor (DatabaseAuditor): Database auditor
        logger: Logger instance
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        categorical_threshold: int | None = 50,
        log_filename: str | os.PathLike[str] | None = None,
        backup_dir: str | os.PathLike[str] | None = None,
        default_batch_size: int = 10000,
        default_max_workers: int = 4,
    ):
        """
        Initialize the atomic operations manager.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            categorical_threshold: Threshold for determining categorical variables.
            log_filename: Path to log file.
            backup_dir: Directory for CSV-based backups.
            default_batch_size: Default batch size for operations.
            default_max_workers: Default number of parallel workers.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> atomic_ops = AtomicDatabaseOperations(conn)
        """
        # Initialisation de la connexion DuckLake.
        self.conn = connection if connection is not None else duckdb.connect(":memory:")

        # Initialisation du seuil pour les variables catégorielles
        self.categorical_threshold = categorical_threshold

        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(
                FILE_PATH.parents[2], "logs/atomic_operations.log"
            )
        self.logger = _init_logger(filename=log_filename)

        # Initialisation des gestionnaires
        self.transaction_mgr = TransactionManager(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
        )

        self.updater = DatabaseUpdater(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
            max_workers=default_max_workers,
            batch_size=default_batch_size,
            enable_validation=True,
        )

        self.deleter = DatabaseDeleter(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
            enable_validation=True,
            auto_cleanup=True,
        )

        self.recovery_mgr = DatabaseRecoveryManager(
            connection, backup_dir, categorical_threshold, log_filename
        )

        self.auditor = DatabaseAuditor(connection, categorical_threshold, log_filename)

        # Configuration par défaut
        self.default_batch_size = default_batch_size
        self.default_max_workers = default_max_workers

    # Opérations atomiques principales
    def execute_batch_update(
        self, update_data: IntoDataFrame, config: AtomicOperationConfig | None = None
    ) -> AtomicOperationResult:
        """
        Execute a batch update operation atomically.

        Args:
            update_data: DataFrame containing data to update
            config: Configuration for the atomic operation

        Returns:
            AtomicOperationResult with operation details

        Example:
            >>> config = AtomicOperationConfig(
            ...     operation_type=AtomicOperationType.BATCH_UPDATE,
            ...     enable_backup=True,
            ...     batch_size=5000
            ... )
            >>> result = atomic_ops.execute_batch_update(new_data_df, config)
            >>> if result.success:
            ...     print(f"Updated {len(new_data_df)} rows successfully")
        """
        if config is None:
            config = AtomicOperationConfig(AtomicOperationType.BATCH_UPDATE)

        start_time = time.time()
        # Conversion vers narwhals pour les opérations nécessitant len()
        _update_nw = nw.from_native(update_data, eager_only=True)

        try:
            self.logger.info(
                f"Starting atomic batch update with {len(_update_nw)} rows"
            )

            # Pré-validation si activée
            if config.enable_validation:
                validation_report = self.auditor.validate_operation_preconditions(
                    "update", df=update_data
                )
                if validation_report.get_critical_issues_count() > 0:
                    return self._create_failed_result(
                        config.operation_type,
                        time.time() - start_time,
                        "Pre-validation failed with critical issues",
                    )

            # Création de sauvegarde si activée
            backup_id = None
            if config.enable_backup:
                backup_id = self.recovery_mgr.create_recovery_point(
                    description="Auto-backup before batch update"
                )
                if not backup_id:
                    self.logger.warning(
                        "Failed to create backup, continuing without backup"
                    )

            # Configuration de l'updater
            batch_size = config.batch_size or self.default_batch_size
            parallel_workers = config.parallel_workers or self.default_max_workers

            # Exécution de la mise à jour
            success = self.updater.update_database(
                update_data,
                check_duplicates_db=True,
                check_duplicates_update=True,
                use_batch_processing=(len(_update_nw) > batch_size),
                use_transaction=True,
            )

            if not success:
                return self._create_failed_result(
                    config.operation_type,
                    time.time() - start_time,
                    "Database update operation failed",
                    backup_created=backup_id,
                )

            # Post-validation si activée
            validation_passed = True
            if config.enable_validation:
                validation_report = self.auditor.validate_database(
                    ValidationLevel.STANDARD
                )
                validation_passed = validation_report.get_critical_issues_count() == 0

                if not validation_passed and config.rollback_on_validation_fail:
                    # Rollback via recovery
                    if backup_id:
                        self._rollback_to_backup(backup_id)

                    return self._create_failed_result(
                        config.operation_type,
                        time.time() - start_time,
                        "Post-validation failed, operation rolled back",
                        backup_created=backup_id,
                        rollback_performed=True,
                    )

            # Création du résultat de succès
            execution_time = time.time() - start_time
            result = AtomicOperationResult(
                success=True,
                operation_type=config.operation_type,
                execution_time=execution_time,
                backup_created=backup_id,
                validation_passed=validation_passed,
                operations_performed=[
                    f"Updated {len(_update_nw)} rows",
                    "Metadata updated",
                    "Dimensions updated",
                    "Fact table updated",
                ],
                metrics={
                    "rows_processed": len(_update_nw),
                    "batch_size_used": batch_size,
                    "parallel_workers_used": parallel_workers,
                },
            )

            if not validation_passed:
                result.recommendations.append(
                    "Review validation issues that were not critical"
                )

            self.logger.info(
                f"Batch update completed successfully in {execution_time:.2f}s"
            )
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error during batch update: {e}")

            return AtomicOperationResult(
                success=False,
                operation_type=config.operation_type,
                execution_time=execution_time,
                backup_created=backup_id,
                error_message=str(e),
            )

    def execute_schema_migration(
        self,
        migration_operations: list[Callable[..., Any]],
        config: AtomicOperationConfig | None = None,
    ) -> AtomicOperationResult:
        """
        Execute schema migration operations atomically.

        Args:
            migration_operations: List of migration functions to execute
            config: Configuration for the atomic operation

        Returns:
            AtomicOperationResult with operation details

        Example:
            >>> def add_column_migration():
            ...     # Add new column logic
            ...     pass
            >>>
            >>> def update_metadata_migration():
            ...     # Update metadata logic
            ...     pass
            >>>
            >>> migrations = [add_column_migration, update_metadata_migration]
            >>> result = atomic_ops.execute_schema_migration(migrations)
        """
        if config is None:
            config = AtomicOperationConfig(AtomicOperationType.SCHEMA_MIGRATION)

        start_time = time.time()

        try:
            self.logger.info(
                f"Starting atomic schema migration with {len(migration_operations)}"
                f" operations"
            )

            # Pré-validation
            if config.enable_validation:
                validation_report = self.auditor.validate_database(
                    ValidationLevel.COMPREHENSIVE
                )
                if validation_report.get_critical_issues_count() > 0:
                    return self._create_failed_result(
                        config.operation_type,
                        time.time() - start_time,
                        "Pre-migration validation failed",
                    )

            # Création de sauvegarde
            backup_id = None
            if config.enable_backup:
                backup_id = self.recovery_mgr.create_recovery_point(
                    description="Auto-backup before schema migration"
                )

            # Début de transaction
            tx_id = self.transaction_mgr.begin_transaction("Schema migration")
            operations_performed = []

            try:
                # Exécution de chaque opération de migration
                for i, migration_op in enumerate(migration_operations):
                    operation = TransactionOperation(
                        operation_type="schema_migration",
                        operation_func=migration_op,
                        description=f"Migration operation {i + 1}",
                    )

                    if not self.transaction_mgr.add_operation(
                        tx_id, **operation.__dict__
                    ):
                        raise Exception(f"Failed to add migration operation {i + 1}")

                    if not self.transaction_mgr.execute_operation(tx_id):
                        raise Exception(
                            f"Failed to execute migration operation {i + 1}"
                        )

                    operations_performed.append(
                        f"Migration operation {i + 1} completed"
                    )

                # Commit des opérations
                if not self.transaction_mgr.commit_transaction(tx_id):
                    raise Exception("Failed to commit migration transaction")

                # Post-validation
                validation_passed = True
                if config.enable_validation:
                    validation_report = self.auditor.validate_database(
                        ValidationLevel.COMPREHENSIVE
                    )
                    validation_passed = (
                        validation_report.get_critical_issues_count() == 0
                    )

                    if not validation_passed and config.rollback_on_validation_fail:
                        if backup_id:
                            self._rollback_to_backup(backup_id)

                        return self._create_failed_result(
                            config.operation_type,
                            time.time() - start_time,
                            "Post-migration validation failed, rolled back",
                            backup_created=backup_id,
                            rollback_performed=True,
                        )

                execution_time = time.time() - start_time
                result = AtomicOperationResult(
                    success=True,
                    operation_type=config.operation_type,
                    execution_time=execution_time,
                    backup_created=backup_id,
                    validation_passed=validation_passed,
                    operations_performed=operations_performed,
                    metrics={"migration_operations_count": len(migration_operations)},
                )

                self.logger.info(
                    f"Schema migration completed successfully in {execution_time:.2f}s"
                )
                return result

            except Exception as e:
                # Rollback de la transaction
                self.transaction_mgr.rollback_transaction(tx_id)
                raise e

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error during schema migration: {e}")

            return AtomicOperationResult(
                success=False,
                operation_type=config.operation_type,
                execution_time=execution_time,
                backup_created=backup_id,
                error_message=str(e),
                rollback_performed=True,
            )

    def execute_cleanup_operation(
        self,
        cleanup_config: dict[str, Any],
        config: AtomicOperationConfig | None = None,
    ) -> AtomicOperationResult:
        """
        Execute comprehensive database cleanup atomically.

        Args:
            cleanup_config: Configuration for cleanup operations
            config: Configuration for the atomic operation

        Returns:
            AtomicOperationResult with operation details

        Example:
            >>> cleanup_config = {
            ...     'remove_orphaned_dimensions': True,
            ...     'remove_null_columns': True,
            ...     'remove_unused_indexes': True,
            ...     'filters': "status = 'deleted'"  # Remove specific rows
            ... }
            >>> result = atomic_ops.execute_cleanup_operation(cleanup_config)
        """
        if config is None:
            config = AtomicOperationConfig(AtomicOperationType.CLEANUP_OPERATION)

        start_time = time.time()

        try:
            self.logger.info("Starting atomic cleanup operation")

            # Pré-validation
            if config.enable_validation:
                validation_report = self.auditor.validate_database(
                    ValidationLevel.BASIC
                )
                if validation_report.get_critical_issues_count() > 0:
                    return self._create_failed_result(
                        config.operation_type,
                        time.time() - start_time,
                        "Pre-cleanup validation failed",
                    )

            # Création de sauvegarde
            backup_id = None
            if config.enable_backup:
                backup_id = self.recovery_mgr.create_recovery_point(
                    description="Auto-backup before cleanup operation"
                )

            operations_performed = []

            # Suppression de lignes si spécifié
            if "filters" in cleanup_config:
                rows_deleted = self.deleter.delete_rows(
                    cleanup_config["filters"],
                    use_transaction=True,
                    perform_cleanup=False,  # On fera le cleanup après
                )
                if rows_deleted >= 0:
                    operations_performed.append(
                        f"Deleted {rows_deleted} rows based on filters"
                    )

            # Suppression de colonnes si spécifié
            if "columns_to_remove" in cleanup_config:
                columns = cleanup_config["columns_to_remove"]
                results = self.deleter.delete_columns(columns, use_transaction=True)
                successful_deletions = sum(results.values())
                operations_performed.append(
                    f"Deleted {successful_deletions}/{len(columns)} columns"
                )

            # Nettoyage complet
            cleanup_results = self.deleter.cleanup_database(comprehensive=True)

            for category, result in cleanup_results.items():
                if result:
                    if isinstance(result, dict):
                        total = sum(result.values())
                        operations_performed.append(
                            f"Cleaned {category}: {total} items"
                        )
                    elif isinstance(result, list):
                        operations_performed.append(
                            f"Cleaned {category}: {len(result)} items"
                        )
                    else:
                        operations_performed.append(f"Cleaned {category}")

            # Post-validation
            validation_passed = True
            if config.enable_validation:
                validation_report = self.auditor.validate_database(
                    ValidationLevel.STANDARD
                )
                validation_passed = validation_report.get_critical_issues_count() == 0

            execution_time = time.time() - start_time
            result = AtomicOperationResult(
                success=True,
                operation_type=config.operation_type,
                execution_time=execution_time,
                backup_created=backup_id,
                validation_passed=validation_passed,
                operations_performed=operations_performed,
                metrics={"cleanup_categories": len(cleanup_results)},
            )

            self.logger.info(
                f"Cleanup operation completed successfully in {execution_time:.2f}s"
            )
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error during cleanup operation: {e}")

            return AtomicOperationResult(
                success=False,
                operation_type=config.operation_type,
                execution_time=execution_time,
                backup_created=backup_id,
                error_message=str(e),
            )

    def execute_validation_and_repair(
        self,
        validation_level: ValidationLevel = ValidationLevel.COMPREHENSIVE,
        config: AtomicOperationConfig | None = None,
    ) -> AtomicOperationResult:
        """
        Execute database validation and automatic repair atomically.

        Args:
            validation_level: Level of validation to perform
            config: Configuration for the atomic operation

        Returns:
            AtomicOperationResult with operation details

        Example:
            >>> result = atomic_ops.execute_validation_and_repair(
            ...     ValidationLevel.COMPREHENSIVE
            ... )
            >>> print(f"Repaired {len(result.operations_performed)} issues")
        """
        if config is None:
            config = AtomicOperationConfig(AtomicOperationType.VALIDATION_AND_REPAIR)

        start_time = time.time()

        try:
            self.logger.info(
                f"Starting atomic validation and repair at level:"
                f"{validation_level.value}"
            )

            # Validation initiale
            validation_report = self.auditor.validate_database(validation_level)
            initial_issues = len(validation_report.issues)
            critical_issues = validation_report.get_critical_issues_count()

            if initial_issues == 0:
                execution_time = time.time() - start_time
                return AtomicOperationResult(
                    success=True,
                    operation_type=config.operation_type,
                    execution_time=execution_time,
                    validation_passed=True,
                    operations_performed=["Validation completed - no issues found"],
                    metrics={"initial_issues": 0, "critical_issues": 0},
                )

            # Création de sauvegarde
            backup_id = None
            if config.enable_backup and critical_issues > 0:
                backup_id = self.recovery_mgr.create_recovery_point(
                    description="Auto-backup before validation and repair"
                )

            # Tentative de réparation automatique
            recovery_result = self.recovery_mgr.auto_recover_from_validation(
                validation_report,
                max_attempts=config.max_retry_attempts,
                allow_destructive=False,  # Sécurité par défaut
            )

            operations_performed = []
            operations_performed.append(
                f"Initial validation found {initial_issues} issues ({critical_issues}"
                f" critical)"
            )

            if recovery_result.success:
                operations_performed.extend(recovery_result.operations_performed)

                # Validation finale
                final_validation = self.auditor.validate_database(validation_level)
                final_issues = len(final_validation.issues)
                final_critical = final_validation.get_critical_issues_count()

                operations_performed.append(
                    f"Final validation: {final_issues} issues ({final_critical}"
                    f" critical)"
                )

                execution_time = time.time() - start_time
                result = AtomicOperationResult(
                    success=True,
                    operation_type=config.operation_type,
                    execution_time=execution_time,
                    backup_created=backup_id,
                    validation_passed=(final_critical == 0),
                    operations_performed=operations_performed,
                    metrics={
                        "initial_issues": initial_issues,
                        "final_issues": final_issues,
                        "initial_critical": critical_issues,
                        "final_critical": final_critical,
                        "issues_resolved": initial_issues - final_issues,
                    },
                )

                if final_critical > 0:
                    result.recommendations.append(
                        "Manual intervention required for remaining critical issues"
                    )

                self.logger.info(
                    f"Validation and repair completed in {execution_time:.2f}s"
                )
                return result
            else:
                return self._create_failed_result(
                    config.operation_type,
                    time.time() - start_time,
                    f"Auto-repair failed: {recovery_result.error_message}",
                    backup_created=backup_id,
                )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error during validation and repair: {e}")

            return AtomicOperationResult(
                success=False,
                operation_type=config.operation_type,
                execution_time=execution_time,
                error_message=str(e),
            )

    def execute_backup_and_update(
        self,
        update_data: IntoDataFrame,
        backup_description: str = "",
        config: AtomicOperationConfig | None = None,
    ) -> AtomicOperationResult:
        """
        Execute backup creation and data update atomically.

        Args:
            update_data: DataFrame containing data to update
            backup_description: Description for the backup
            config: Configuration for the atomic operation

        Returns:
            AtomicOperationResult with operation details

        Example:
            >>> result = atomic_ops.execute_backup_and_update(
            ...     update_df,
            ...     "Weekly data update"
            ... )
        """
        if config is None:
            config = AtomicOperationConfig(
                AtomicOperationType.BACKUP_AND_UPDATE,
                enable_backup=True,  # Force backup for this operation
                enable_validation=True,
            )

        config.enable_backup = True  # Ensure backup is always enabled

        start_time = time.time()
        # Conversion vers narwhals pour les opérations nécessitant len()
        _update_nw2 = nw.from_native(update_data, eager_only=True)

        try:
            self.logger.info(
                f"Starting atomic backup and update with {len(_update_nw2)} rows"
            )

            # Étape 1: Création de la sauvegarde
            backup_id = self.recovery_mgr.create_recovery_point(
                description=backup_description or "Automatic backup before update"
            )

            if not backup_id:
                return self._create_failed_result(
                    config.operation_type,
                    time.time() - start_time,
                    "Failed to create backup - aborting update",
                )

            # Étape 2: Exécution de la mise à jour avec la configuration
            update_config = AtomicOperationConfig(
                AtomicOperationType.BATCH_UPDATE,
                enable_backup=False,  # Backup déjà créé
                enable_validation=config.enable_validation,
                rollback_on_validation_fail=config.rollback_on_validation_fail,
                batch_size=config.batch_size,
                parallel_workers=config.parallel_workers,
            )

            update_result = self.execute_batch_update(update_data, update_config)

            if not update_result.success:
                # Tentative de rollback si l'update échoue
                self.logger.warning("Update failed, attempting rollback to backup")

                recovery_op = RecoveryOperation(
                    strategy=RecoveryStrategy.RESTORE_BACKUP,
                    target_recovery_point=backup_id,
                    description="Rollback after failed update",
                )

                recovery_result = self.recovery_mgr.recover_database(
                    recovery_op, confirm_destructive=True
                )
                rollback_performed = recovery_result.success

                return AtomicOperationResult(
                    success=False,
                    operation_type=config.operation_type,
                    execution_time=time.time() - start_time,
                    backup_created=backup_id,
                    error_message=f"Update failed: {update_result.error_message}",
                    rollback_performed=rollback_performed,
                    operations_performed=[
                        "Backup created successfully",
                        f"Update failed: {update_result.error_message}",
                        f"Rollback {'successful' if rollback_performed else 'failed'}",
                    ],
                )

            # Succès - combinaison des résultats
            execution_time = time.time() - start_time
            operations_performed = ["Backup created successfully"]
            operations_performed.extend(update_result.operations_performed)

            result = AtomicOperationResult(
                success=True,
                operation_type=config.operation_type,
                execution_time=execution_time,
                backup_created=backup_id,
                validation_passed=update_result.validation_passed,
                operations_performed=operations_performed,
                metrics=update_result.metrics,
            )

            result.recommendations.append(
                "Backup created and can be used for future recovery"
            )

            self.logger.info(
                f"Backup and update completed successfully in {execution_time:.2f}s"
            )
            return result

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error during backup and update: {e}")

            return AtomicOperationResult(
                success=False,
                operation_type=config.operation_type,
                execution_time=execution_time,
                error_message=str(e),
            )

    # Méthodes utilitaires
    def get_operation_status(self) -> dict[str, Any]:
        """
        Get the status of all atomic operation components.

        Returns:
            Dictionary containing status information

        Example:
            >>> status = atomic_ops.get_operation_status()
            >>> print(f"Active transactions: {status['active_transactions']}")
        """
        try:
            status: dict[str, Any] = {
                "timestamp": datetime.now(),
                "active_transactions": len(
                    self.transaction_mgr.list_active_transactions()
                ),
                "recovery_points_available": len(
                    self.recovery_mgr.list_recovery_points()
                ),
                "database_health": self.auditor.get_quick_health_check(),
                "updater_status": self.updater.get_update_status(),
                "deleter_status": self.deleter.get_deletion_status(),
                "components_ready": True,
            }

            # Vérification de la santé des composants
            if status["database_health"].get("status") == "error":
                status["components_ready"] = False
                status["warnings"] = ["Database health check failed"]

            return status

        except Exception as e:
            self.logger.error(f"Error getting operation status: {e}")
            return {
                "timestamp": datetime.now(),
                "error": str(e),
                "components_ready": False,
            }

    def validate_system_integrity(self) -> dict[str, Any]:
        """
        Perform comprehensive system integrity validation.

        Returns:
            Dictionary with validation results

        Example:
            >>> integrity = atomic_ops.validate_system_integrity()
            >>> if integrity['overall_status'] == 'healthy':
            ...     print("System integrity verified")
        """
        try:
            # Validation complète de la base de données
            validation_report = self.auditor.validate_database(
                ValidationLevel.COMPREHENSIVE
            )

            # Collecte des métriques de santé
            health_check = self.auditor.get_quick_health_check()

            # Vérification des composants
            components_status = {
                "transaction_manager": len(
                    self.transaction_mgr.list_active_transactions()
                )
                >= 0,
                "updater": True,  # Assume healthy if instantiated
                "deleter": True,  # Assume healthy if instantiated
                "recovery_manager": len(self.recovery_mgr.list_recovery_points()) >= 0,
                "auditor": health_check.get("status") != "error",
            }

            # Détermination du statut global
            critical_issues = validation_report.get_critical_issues_count()
            all_components_ready = all(components_status.values())

            if critical_issues == 0 and all_components_ready:
                overall_status = "healthy"
            elif critical_issues > 0:
                overall_status = "critical_issues"
            else:
                overall_status = "component_issues"

            return {
                "timestamp": datetime.now(),
                "overall_status": overall_status,
                "validation_report": {
                    "total_issues": len(validation_report.issues),
                    "critical_issues": critical_issues,
                    "tables_validated": len(validation_report.tables_validated),
                },
                "components_status": components_status,
                "health_metrics": health_check,
                "recommendations": validation_report.recommendations
                if hasattr(validation_report, "recommendations")
                else [],
            }

        except Exception as e:
            self.logger.error(f"Error validating system integrity: {e}")
            return {
                "timestamp": datetime.now(),
                "overall_status": "error",
                "error": str(e),
            }

    def emergency_recovery(
        self,
        recovery_strategy: RecoveryStrategy | None = None,
        allow_destructive: bool = False,
    ) -> AtomicOperationResult:
        """
        Perform emergency recovery operations.

        Args:
            recovery_strategy: Specific recovery strategy (None for auto-detection)
            allow_destructive: Whether to allow destructive recovery operations

        Returns:
            AtomicOperationResult with recovery details

        Example:
            >>> result = atomic_ops.emergency_recovery(allow_destructive=True)
            >>> if result.success:
            ...     print("Emergency recovery completed")
        """
        start_time = time.time()

        try:
            self.logger.warning("Initiating emergency recovery procedure")

            # Évaluation de la situation
            validation_report = self.auditor.validate_database(
                ValidationLevel.COMPREHENSIVE
            )
            critical_issues = validation_report.get_critical_issues_count()

            # Détermination de la stratégie de récupération
            if recovery_strategy is None:
                if critical_issues > 5:  # Beaucoup d'issues critiques
                    recent_backups = [
                        p
                        for p in self.recovery_mgr.list_recovery_points()
                        if time.time() - p.timestamp < 86400
                    ]  # Moins de 24h

                    if recent_backups and allow_destructive:
                        recovery_strategy = RecoveryStrategy.RESTORE_BACKUP
                    else:
                        recovery_strategy = RecoveryStrategy.CLEAN_ORPHANED_DATA
                else:
                    recovery_strategy = RecoveryStrategy.VALIDATE_AND_FIX

            # Création de l'opération de récupération
            recovery_op = RecoveryOperation(
                strategy=recovery_strategy,
                auto_validate=True,
                description="Emergency recovery operation",
            )

            # Si on restaure une sauvegarde, utiliser la plus récente
            if recovery_strategy == RecoveryStrategy.RESTORE_BACKUP:
                recovery_points = self.recovery_mgr.list_recovery_points()
                if recovery_points:
                    recovery_op.target_recovery_point = recovery_points[0].recovery_id
                else:
                    return self._create_failed_result(
                        AtomicOperationType.VALIDATION_AND_REPAIR,
                        time.time() - start_time,
                        "No backup available for emergency recovery",
                    )

            # Exécution de la récupération
            recovery_result = self.recovery_mgr.recover_database(
                recovery_op, confirm_destructive=allow_destructive
            )

            execution_time = time.time() - start_time

            if recovery_result.success:
                result = AtomicOperationResult(
                    success=True,
                    operation_type=AtomicOperationType.VALIDATION_AND_REPAIR,
                    execution_time=execution_time,
                    operations_performed=[
                        f"Emergency recovery using strategy: {recovery_strategy.value}",
                        f"Initial critical issues: {critical_issues}",
                    ]
                    + recovery_result.operations_performed,
                    validation_passed=recovery_result.validation_report.get_critical_issues_count()
                    == 0
                    if recovery_result.validation_report
                    else True,
                    recommendations=recovery_result.recommendations,
                )

                self.logger.info(
                    f"Emergency recovery completed successfully in {execution_time:.2f}"
                    f" s"
                )
                return result
            else:
                return AtomicOperationResult(
                    success=False,
                    operation_type=AtomicOperationType.VALIDATION_AND_REPAIR,
                    execution_time=execution_time,
                    error_message=f"Emergency recovery failed:"
                                  f"{recovery_result.error_message}",
                )

        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Error during emergency recovery: {e}")

            return AtomicOperationResult(
                success=False,
                operation_type=AtomicOperationType.VALIDATION_AND_REPAIR,
                execution_time=execution_time,
                error_message=str(e),
            )

    # Méthodes privées utilitaires
    def _create_failed_result(
        self,
        operation_type: AtomicOperationType,
        execution_time: float,
        error_message: str,
        backup_created: str | None = None,
        rollback_performed: bool = False,
    ) -> AtomicOperationResult:
        """Créer un résultat d'échec standardisé."""
        return AtomicOperationResult(
            success=False,
            operation_type=operation_type,
            execution_time=execution_time,
            backup_created=backup_created,
            validation_passed=False,
            error_message=error_message,
            rollback_performed=rollback_performed,
        )

    def _rollback_to_backup(self, backup_id: str) -> bool:
        """Effectuer un rollback vers une sauvegarde."""
        try:
            recovery_op = RecoveryOperation(
                strategy=RecoveryStrategy.RESTORE_BACKUP,
                target_recovery_point=backup_id,
                description="Automatic rollback to backup",
            )

            result = self.recovery_mgr.recover_database(
                recovery_op, confirm_destructive=True
            )
            return result.success

        except Exception as e:
            self.logger.error(f"Error during rollback to backup {backup_id}: {e}")
            return False
