# Importation des modules
# Modules de base
import os
import json
import polars as pl
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import shutil
# DuckDB
import duckdb

# Import des gestionnaires
from .auditor import DatabaseAuditor, ValidationLevel, ValidationReport, ValidationIssue
from .._internal.managers.dimension import DimensionManager
from ..operations.deleter import DatabaseDeleter
# Import des utilitaires
from ..utils.logger import _init_logger

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
    the in-place repair strategies (``REPAIR_SCHEMA``, ``REBUILD_DIMENSIONS``,
    ``CLEAN_ORPHANED_DATA``, ``VALIDATE_AND_FIX``) operate directly on the live
    catalog without touching the snapshot history.
    """
    USE_SNAPSHOT_HISTORY = "use_snapshot_history"
    REPAIR_SCHEMA = "repair_schema"
    REBUILD_DIMENSIONS = "rebuild_dimensions"
    CLEAN_ORPHANED_DATA = "clean_orphaned_data"
    VALIDATE_AND_FIX = "validate_and_fix"


# Classe des types de sauvegarde possibles
# Seule la sauvegarde des métadonnées est conservée : DuckLake gère nativement
# l'historique complet des données via ses snapshots, rendant les sauvegardes CSV
# redondantes pour les tables de faits et de dimensions.
class BackupType(Enum):
    """Available backup types for recovery points.

    Only ``METADATA_BACKUP`` is retained. Full-data backups are superseded by
    DuckLake's native snapshot history, which provides immutable time-travel
    access to every previous state of the catalog.
    """
    METADATA_BACKUP = "metadata_backup"


# Classe de point de récupération de la base de données
@dataclass
class RecoveryPoint:
    """
    Recovery point containing the state of the database.

    Attributes:
        recovery_id (str): Unique identifier of the recovery point
        timestamp (float): Creation timestamp
        backup_type (BackupType): Type of backup
        backup_path (str): Path to the backup files
        metadata (dict): Metadata about the database state
        validation_report (Optional[ValidationReport]): Validation report at the time of backup
        description (str): Description of the recovery point
    """
    recovery_id: str
    timestamp: float
    backup_type: BackupType
    backup_path: str
    metadata: dict = field(default_factory=dict)
    validation_report: Optional[Any] = None
    description: str = ""


# Classe d'opération de récupération
@dataclass
class RecoveryOperation:
    """
    Recovery operation to execute.

    Attributes:
        strategy (RecoveryStrategy): Recovery strategy to use
        target_recovery_point (Optional[str]): ID of the target recovery point
        parameters (dict): Strategy-specific parameters
        auto_validate (bool): Whether to automatically validate after recovery
        description (str): Description of the operation
    """
    strategy: RecoveryStrategy
    target_recovery_point: Optional[str] = None
    parameters: dict = field(default_factory=dict)
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
    operations_performed: List[str] = field(default_factory=list)
    validation_report: Optional[Any] = None
    error_message: Optional[str] = None
    recommendations: List[str] = field(default_factory=list)


# Classe de récupération de la base de données
class DatabaseRecoveryManager:
    """
    Manages database recovery operations and backup/restore functionality.
    
    Provides comprehensive error recovery mechanisms including automatic backup creation,
    validation-driven repair, schema consistency restoration, and transaction rollback.
    
    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection
        backup_dir (Path): Directory for storing backups
        auditor (DatabaseAuditor): Database auditor for validation
        logger: Logger instance for recovery tracking
        max_backup_age_days (int): Maximum age for keeping backups
        auto_backup_on_changes (bool): Whether to create automatic backups
    """
    # Initialisation
    def __init__(self,
                 connection: Optional[duckdb.DuckDBPyConnection] = None,
                 backup_dir: Optional[os.PathLike] = None,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None,
                 max_backup_age_days: int = 30,
                 auto_backup_on_changes: bool = True,
                 catalog_alias: str = 'db',
                 ducklake_schema: str = 'main'):
        """
        Initialize the database recovery manager.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            backup_dir: Directory for storing metadata backups (JSON + CSV export of
                the metadata table). Full-data backups are not needed here because
                DuckLake natively persists the complete snapshot history.
            categorical_threshold: Threshold for determining categorical variables.
            log_filename: Path to log file.
            max_backup_age_days: Maximum age for keeping metadata backup files on disk.
            auto_backup_on_changes: Whether to create an automatic metadata backup
                before destructive in-place operations (e.g. REPAIR_SCHEMA).
            catalog_alias: Alias of the attached DuckLake catalog, used to query
                available snapshots via ``ducklake_snapshots()``. Defaults to ``'db'``.
            ducklake_schema: DuckLake schema name used for snapshot queries.
                Defaults to ``'main'``.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> recovery_mgr = DatabaseRecoveryManager(conn, backup_dir='/path/to/backups')
        """
        # Initialisation de la connexion DuckLake.
        self.conn = connection if connection is not None else duckdb.connect(':memory:')
        self.catalog_alias = catalog_alias
        self.ducklake_schema = ducklake_schema
        
        # Configuration des répertoires
        if backup_dir is None:
            backup_dir = os.path.join(FILE_PATH.parents[2], "data/backups")
        
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialisation des composants
        self.auditor = DatabaseAuditor(connection, categorical_threshold, log_filename)
        
        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/database_recovery.log")
        self.logger = _init_logger(filename=log_filename)
        
        # Configuration
        self.categorical_threshold = categorical_threshold
        self.max_backup_age_days = max_backup_age_days
        self.auto_backup_on_changes = auto_backup_on_changes
        
        # État interne
        self._recovery_points: Dict[str, RecoveryPoint] = {}
        self._load_existing_recovery_points()
    
    # Méthodes de gestion des points de récupération
    # Méthode de création d'un point de récupération
    def create_recovery_point(self,
                              backup_type: BackupType = BackupType.METADATA_BACKUP,
                              description: str = "",
                              force_validation: bool = True) -> Optional[str]:
        """
        Create a metadata recovery point with optional pre-validation.

        Only ``BackupType.METADATA_BACKUP`` is supported : it exports the
        ``metadata`` table to CSV and writes a system health snapshot to JSON.
        For full data recovery, rely on DuckLake's native snapshot history via
        ``USE_SNAPSHOT_HISTORY``.

        Args:
            backup_type: Must be ``BackupType.METADATA_BACKUP`` (only supported type).
            description: Human-readable description of the recovery point.
            force_validation: Whether to run a standard validation before backup.

        Returns:
            Recovery point ID if successful, None otherwise.

        Example:
            >>> recovery_id = recovery_mgr.create_recovery_point(
            ...     description="Before schema repair"
            ... )
        """
        try:
            # Génération d'un ID unique
            recovery_id = f"recovery_{int(time.time())}_{len(self._recovery_points)}"
            # Génération d'un timestamp
            timestamp = time.time()

            # Validation préalable si demandée
            validation_report = None
            if force_validation:
                validation_report = self.auditor.validate_database(ValidationLevel.STANDARD)

                # Vérification des issues critiques
                if validation_report.get_critical_issues_count() > 0:
                    self.logger.warning(
                        f"Creating recovery point with "
                        f"{validation_report.get_critical_issues_count()} critical issues"
                    )

            # Création du répertoire de sauvegarde
            backup_path = self.backup_dir / recovery_id
            backup_path.mkdir(exist_ok=True)

            # Seule la sauvegarde des métadonnées est supportée
            success = self._create_metadata_backup(backup_path)
            
            if not success:
                # Nettoyage en cas d'échec
                shutil.rmtree(backup_path, ignore_errors=True)
                return None
            
            # Création de l'objet RecoveryPoint
            recovery_point = RecoveryPoint(
                recovery_id=recovery_id,
                timestamp=timestamp,
                backup_type=backup_type,
                backup_path=str(backup_path),
                metadata=self._collect_database_metadata(),
                validation_report=validation_report,
                description=description
            )
            
            # Sauvegarde des informations du point de récupération
            self._save_recovery_point_info(recovery_point)
            
            # Ajout au cache
            self._recovery_points[recovery_id] = recovery_point
            
            # Nettoyage des anciens points de récupération
            self._cleanup_old_recovery_points()
            
            # Logging
            self.logger.info(f"Created recovery point {recovery_id} ({backup_type.value})")
            return recovery_id
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error creating recovery point: {e}")
            return None
    
    # Méthode d'énumération des points de récupération
    def list_recovery_points(self, backup_type: Optional[BackupType] = None) -> List[RecoveryPoint]:
        """
        List available recovery points.
        
        Args:
            backup_type: Filter by backup type (None for all)
            
        Returns:
            List of recovery points sorted by timestamp (newest first)
            
        Example:
            >>> points = recovery_mgr.list_recovery_points(BackupType.FULL_BACKUP)
            >>> for point in points:
            ...     print(f"{point.recovery_id}: {point.description}")
        """
        try:
            # Extraction des points de récupération
            points = list(self._recovery_points.values())
            
            # Filtrage par type si spécifié
            if backup_type:
                points = [p for p in points if p.backup_type == backup_type]
            
            # Tri par timestamp décroissant
            points.sort(key=lambda p: p.timestamp, reverse=True)
            
            return points
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error listing recovery points: {e}")
            return []
    
    # Méthode de suppression d'un point de récupération
    def delete_recovery_point(self, recovery_id: str) -> bool:
        """
        Delete a specific recovery point.
        
        Args:
            recovery_id: ID of the recovery point to delete
            
        Returns:
            True if deletion was successful
            
        Example:
            >>> success = recovery_mgr.delete_recovery_point("recovery_1234567890_0")
        """
        try:
            # Impossible de supprimer un point de récupération qui n'existe pas
            if recovery_id not in self._recovery_points:
                # Logging
                self.logger.error(f"Recovery point {recovery_id} not found")
                return False
            
            # Extraction du point de récupération à supprimer
            recovery_point = self._recovery_points[recovery_id]
            
            # Suppression des fichiers de sauvegarde
            backup_path = Path(recovery_point.backup_path)
            if backup_path.exists():
                shutil.rmtree(backup_path, ignore_errors=True)
            
            # Suppression du cache
            del self._recovery_points[recovery_id]
            
            # Logging
            self.logger.info(f"Deleted recovery point {recovery_id}")
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error deleting recovery point {recovery_id}: {e}")
            return False
    
    # Méthodes de récupération
    # Méthode de récupération de la base de données
    def recover_database(self, 
                        operation: RecoveryOperation,
                        confirm_destructive: bool = False) -> RecoveryResult:
        """
        Perform database recovery using the specified operation.
        
        Args:
            operation: Recovery operation to perform
            confirm_destructive: Confirm destructive operations (required for some strategies)
            
        Returns:
            RecoveryResult with operation details
            
        Example:
            >>> op = RecoveryOperation(
            ...     strategy=RecoveryStrategy.RESTORE_BACKUP,
            ...     target_recovery_point="recovery_1234567890_0",
            ...     description="Restore from backup after corruption"
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
                    error_message="Recovery operation validation failed"
                )
            
            # Sauvegarde automatique des métadonnées avant les opérations in-place destructrices.
            # Les opérations sur les données sont couvertes par l'historique DuckLake.
            pre_recovery_point = None
            if operation.strategy == RecoveryStrategy.REPAIR_SCHEMA:
                pre_recovery_point = self.create_recovery_point(
                    BackupType.METADATA_BACKUP,
                    f"Auto-backup before {operation.strategy.value}"
                )

            # Exécution de la stratégie de récupération
            if operation.strategy == RecoveryStrategy.USE_SNAPSHOT_HISTORY:
                result = self._recover_use_snapshot_history(operation)
            elif operation.strategy == RecoveryStrategy.REPAIR_SCHEMA:
                result = self._recover_repair_schema(operation)
            elif operation.strategy == RecoveryStrategy.REBUILD_DIMENSIONS:
                result = self._recover_rebuild_dimensions(operation)
            elif operation.strategy == RecoveryStrategy.CLEAN_ORPHANED_DATA:
                result = self._recover_clean_orphaned_data(operation)
            elif operation.strategy == RecoveryStrategy.VALIDATE_AND_FIX:
                result = self._recover_validate_and_fix(operation)
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    error_message=f"Unknown recovery strategy: {operation.strategy}"
                )
            
            # Calcul du temps de récupération
            recovery_time = time.time() - start_time
            result.recovery_time = recovery_time
            
            # Validation post-récupération si demandée
            if result.success and operation.auto_validate:
                validation_report = self.auditor.validate_database(ValidationLevel.STANDARD)
                result.validation_report = validation_report
                
                # Vérification des issues critiques persistantes
                if validation_report.get_critical_issues_count() > 0:
                    result.recommendations.append(
                        f"Recovery completed but {validation_report.get_critical_issues_count()} critical issues remain"
                    )
            
            # Logging du résultat
            if result.success:
                self.logger.info(f"Recovery operation completed successfully in {recovery_time:.2f}s")
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
                error_message=str(e)
            )
    
    # Méthode de récupération automatique de la base de données sur la base d'un rapport de validation
    def auto_recover_from_validation(self, 
                                   validation_report: Any,
                                   max_attempts: int = 3,
                                   allow_destructive: bool = False) -> RecoveryResult:
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
            self.logger.info(f"Starting auto-recovery from validation issues")
            
            # Analyse des problèmes pour déterminer la stratégie de récupération
            strategy = self._determine_recovery_strategy(validation_report, allow_destructive)
            
            # Renvoi un message si aucune stratégie n'a été trouvée
            if strategy is None:
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.VALIDATE_AND_FIX,
                    recovery_time=0,
                    error_message="No suitable recovery strategy found"
                )
            
            # Tentatives de récupération
            for attempt in range(max_attempts):
                # Logging
                self.logger.info(f"Auto-recovery attempt {attempt + 1}/{max_attempts}")
                
                # Création de l'opération de récupération
                operation = RecoveryOperation(
                    strategy=strategy,
                    parameters={'validation_report': validation_report},
                    auto_validate=True,
                    description=f"Auto-recovery attempt {attempt + 1}"
                )
                
                # Exécution de la récupération
                result = self.recover_database(operation, confirm_destructive=allow_destructive)
                
                if result.success:
                    # Vérification que les issues ont été résolues
                    if result.validation_report and result.validation_report.get_critical_issues_count() == 0:
                        # Logging
                        self.logger.info("Auto-recovery completed successfully")
                        return result
                    else:
                        # Logging
                        self.logger.warning("Recovery completed but issues persist, trying again")
                else:
                    # Logging
                    self.logger.warning(f"Recovery attempt {attempt + 1} failed: {result.error_message}")
            
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                recovery_time=0,
                error_message=f"Auto-recovery failed after {max_attempts} attempts"
            )
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error during auto-recovery: {e}")

            return RecoveryResult(
                success=False,
                strategy_used=RecoveryStrategy.VALIDATE_AND_FIX,
                recovery_time=0,
                error_message=str(e)
            )
    
    # Méthodes de sauvegarde privées
    # Méthode de sauvegarde des méta-données
    def _create_metadata_backup(self, backup_path: Path) -> bool:
        """Create a metadata-only backup (metadata table and system info).

        Args:
            backup_path: Directory path where metadata backup will be stored.

        Returns:
            True if backup completed successfully, False on error.
        """
        try:
            # Sauvegarde de la table metadata si elle existe
            if self._table_exists('metadata'):
                metadata_file = backup_path / "metadata.csv"
                export_query = f"COPY metadata TO '{metadata_file}' (FORMAT CSV, HEADER)"
                self.conn.execute(export_query)
            
            # Sauvegarde des informations système
            system_info = {
                'database_info': self.auditor.get_quick_health_check(),
                'timestamp': time.time()
            }
            
            # Exportation dans un fichier json
            system_file = backup_path / "system_info.json"
            with open(system_file, 'w') as f:
                json.dump(system_info, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error creating metadata backup: {e}")
            return False
    
    # Méthodes de récupération spécialisées
    # Méthode de récupération par historique des snapshots DuckLake
    def _recover_use_snapshot_history(self, operation: RecoveryOperation) -> RecoveryResult:
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
            operations_performed: List[str] = []

            # Interrogation de l'historique des snapshots DuckLake
            try:
                snapshots_df = self.conn.execute(
                    f"SELECT * FROM {self.catalog_alias}.ducklake_snapshots"
                    f"('{self.ducklake_schema}') ORDER BY snapshot_id DESC"
                ).pl()
            except Exception as e:
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    error_message=f"Impossible d'interroger les snapshots DuckLake : {e}",
                )

            snapshot_count = len(snapshots_df)
            operations_performed.append(
                f"{snapshot_count} snapshot(s) disponible(s) dans le catalogue "
                f"'{self.catalog_alias}.{self.ducklake_schema}'"
            )

            # Présentation structurée de chaque snapshot avec horodatage lisible
            for row in snapshots_df.iter_rows(named=True):
                snap_id = row.get('snapshot_id', 'N/A')
                # Récupération de l'horodatage selon le nom de colonne retourné par DuckLake
                snap_time = row.get('snapshot_time', row.get('timestamp', None))
                if snap_time is not None and hasattr(snap_time, 'strftime'):
                    snap_time_str = snap_time.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    snap_time_str = str(snap_time) if snap_time is not None else 'N/A'
                operations_performed.append(f"  snapshot_id={snap_id}  |  {snap_time_str}")

            # Détermination du snapshot cible : validé si fourni, suggéré sinon
            target_snapshot = operation.target_recovery_point
            suggested_id: str = '<snapshot_id>'

            if target_snapshot:
                # Vérification que le snapshot demandé figure bien dans l'historique
                snapshot_ids = (
                    [str(v) for v in snapshots_df['snapshot_id'].to_list()]
                    if snapshot_count > 0 else []
                )
                if target_snapshot in snapshot_ids:
                    operations_performed.append(
                        f"Snapshot cible validé : {target_snapshot}"
                    )
                    suggested_id = target_snapshot
                else:
                    operations_performed.append(
                        f"ATTENTION : snapshot {target_snapshot} introuvable dans l'historique — "
                        f"vérifier la valeur de snapshot_id"
                    )
            elif snapshot_count > 1:
                # Avant-dernier snapshot : état stable avant la dernière écriture
                suggested_id = str(snapshots_df['snapshot_id'][1])
                operations_performed.append(
                    f"Snapshot suggéré (avant-dernier, état stable) : {suggested_id}"
                )
            elif snapshot_count == 1:
                suggested_id = str(snapshots_df['snapshot_id'][0])
                operations_performed.append(
                    "Un seul snapshot disponible — restauration vers l'état initial uniquement"
                )

            # Instructions de restauration étape par étape (format prêt à copier-coller)
            recommendations = [
                "─── Procédure de restauration par time-travel DuckLake ───",
                "",
                "Étape 1 — Ouvrir une connexion en lecture seule sur le snapshot cible :",
                f"  conn_old = DuckLakeConnector(catalog_path, data_path, snapshot_version={suggested_id}).connect()",
                "",
                "Étape 2 — Lire les tables depuis cette connexion :",
                "  fact_df = conn_old.execute('SELECT * FROM fact_table').pl()",
                "  meta_df = conn_old.execute('SELECT * FROM metadata').pl()",
                "  # Répéter pour chaque table de dimension :",
                "  #   dim_df = conn_old.execute('SELECT * FROM dim_<col>').pl()",
                "",
                "Étape 3 — Vider les tables du catalogue courant et réinsérer les données :",
                "  conn.execute('DELETE FROM fact_table')",
                "  conn.register('_restore_fact', fact_df)",
                "  conn.execute('INSERT INTO fact_table SELECT * FROM _restore_fact')",
                "  conn.execute('DROP VIEW _restore_fact')",
                "  # Répéter pour metadata et chaque table de dimension",
                "",
                "Étape 4 — Valider l'intégrité après restauration :",
                "  auditor = DatabaseAuditor(conn)",
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
    def list_ducklake_snapshots(self) -> Optional[pl.DataFrame]:
        """Return the full DuckLake snapshot history as a Polars DataFrame.

        Convenience wrapper around the ``ducklake_snapshots()`` table function.
        Useful for inspecting the snapshot inventory directly before choosing a
        ``snapshot_version`` for time-travel restoration.

        Returns:
            Polars DataFrame with one row per snapshot (columns depend on the
            DuckLake version), sorted by ``snapshot_id`` descending.
            Returns None if the catalog cannot be queried.

        Example:
            >>> snapshots = recovery_mgr.list_ducklake_snapshots()
            >>> if snapshots is not None:
            ...     print(snapshots)
        """
        try:
            return self.conn.execute(
                f"SELECT * FROM {self.catalog_alias}.ducklake_snapshots"
                f"('{self.ducklake_schema}') ORDER BY snapshot_id DESC"
            ).pl()
        except Exception as e:
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
            validation_report = self.auditor.validate_database(ValidationLevel.COMPREHENSIVE)

            # Réparation des problèmes du schéma
            schema_issues = [issue for issue in validation_report.issues
                           if 'schema' in issue.issue_type.value.lower()]

            # Parcours des problèmes
            for issue in schema_issues:
                try:
                    # Application du fix automatique si possible
                    fixed = self._apply_schema_fix(issue)
                    if fixed:
                        operations_performed.append(f"Fixed: {issue.description}")
                    else:
                        operations_performed.append(f"Could not auto-fix: {issue.description}")
                except Exception as e:
                    operations_performed.append(f"Failed to fix: {issue.description} - {e}")

            return RecoveryResult(
                success=True,
                strategy_used=operation.strategy,
                recovery_time=0,
                operations_performed=operations_performed,
                recommendations=["Manual schema review may be required for complex issues"]
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e)
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
            if "missing" in issue.description.lower() and issue.table_name == 'metadata':
                self.conn.execute("""
                    CREATE TABLE IF NOT EXISTS metadata (
                        name VARCHAR,
                        label VARCHAR,
                        python_type VARCHAR,
                        sql_type VARCHAR,
                        is_categorical BOOLEAN,
                        is_primary_key BOOLEAN DEFAULT FALSE
                    )
                """)
                self.logger.info("Created missing metadata table")
                return True

            # Cas 2: Colonnes manquantes dans metadata → ajout avec valeurs par défaut
            if "Missing required columns in metadata" in issue.description:
                return self._add_missing_metadata_columns(issue)

            # Cas 3: Table de dimension manquante pour colonne catégorielle
            if "Dimension table" in issue.description and "missing" in issue.description.lower():
                col_name = issue.column_name
                if col_name:
                    # Création de la table de dimension à partir des valeurs existantes
                    dim_mgr = DimensionManager(self.conn, self.categorical_threshold)
                    dim_mgr.create_dimension_table(col_name)
                    # Récupération des valeurs distinctes de la fact_table
                    values = self.conn.execute(
                        f"SELECT DISTINCT {col_name} FROM fact_table WHERE {col_name} IS NOT NULL"
                    ).pl()[col_name]
                    dim_mgr.update_dimension_values(col_name, values)
                    self.logger.info(f"Created missing dimension table for {col_name}")
                    return True

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
            required_columns = {
                'name': 'VARCHAR',
                'label': 'VARCHAR',
                'python_type': 'VARCHAR',
                'sql_type': 'VARCHAR',
                'is_categorical': 'BOOLEAN',
                'is_primary_key': 'BOOLEAN DEFAULT FALSE'
            }

            # Récupération des colonnes existantes
            existing_columns = set()
            try:
                result = self.conn.execute("DESCRIBE metadata").fetchall()
                existing_columns = {row[0] for row in result}
            except:
                pass

            # Ajout des colonnes manquantes
            for col_name, col_type in required_columns.items():
                if col_name not in existing_columns:
                    default_value = "FALSE" if "BOOLEAN" in col_type else "NULL"
                    self.conn.execute(f"ALTER TABLE metadata ADD COLUMN {col_name} {col_type}")
                    self.logger.info(f"Added missing column {col_name} to metadata table")

            return True

        except Exception as e:
            self.logger.error(f"Error adding missing metadata columns: {e}")
            return False
    
    # Méthode de reconstruction des tables de dimension corrompues
    def _recover_rebuild_dimensions(self, operation: RecoveryOperation) -> RecoveryResult:
        """Recover by rebuilding corrupted dimension tables.

        Cleans orphaned entries, recreates missing tables, and synchronizes
        dimension values with fact table data.

        Args:
            operation: Recovery operation parameters.

        Returns:
            RecoveryResult with details of dimension table repairs.
        """
        try:
            # Initialisation de la liste des opérations réalisées
            operations_performed = []

            # Reconstruction des tables de dimension corrompues
            # Initialisation du gestionnaire des dimensions
            dim_mgr = DimensionManager(self.conn, self.categorical_threshold)

            # Étape 1: Nettoyage des entrées orphelines
            cleaned = dim_mgr.cleanup_orphaned_dimension_entries()
            # Parcours des résultats du nettoyage
            for dim_name, count in cleaned.items():
                if count > 0:
                    operations_performed.append(f"Cleaned {count} orphaned entries from {dim_name}")

            # Étape 2: Reconstruction des tables de dimension manquantes
            try:
                metadata = self.conn.execute("SELECT name, is_categorical FROM metadata").pl()
                categorical_cols = metadata.filter(pl.col('is_categorical') == True)['name'].to_list()

                for col_name in categorical_cols:
                    dim_table = f"dim_{col_name}"
                    if not self._table_exists(dim_table):
                        # Recréation de la table de dimension à partir de la fact_table
                        dim_mgr.create_dimension_table(col_name)
                        values = self.conn.execute(
                            f"SELECT DISTINCT {col_name} FROM fact_table WHERE {col_name} IS NOT NULL"
                        ).pl()[col_name]
                        added = dim_mgr.update_dimension_values(col_name, values)
                        operations_performed.append(f"Recreated {dim_table} with {added} values")
            except Exception as e:
                operations_performed.append(f"Error rebuilding missing tables: {e}")

            # Étape 3: Synchronisation des entrées manquantes dans les tables existantes
            try:
                for col_name in categorical_cols:
                    dim_table = f"dim_{col_name}"
                    if self._table_exists(dim_table):
                        # Ajout des valeurs de fact_table manquantes dans dimension
                        fact_values = self.conn.execute(
                            f"SELECT DISTINCT {col_name} FROM fact_table WHERE {col_name} IS NOT NULL"
                        ).pl()[col_name]
                        added = dim_mgr.update_dimension_values(col_name, fact_values)
                        if added > 0:
                            operations_performed.append(f"Added {added} missing entries to {dim_table}")
            except Exception as e:
                operations_performed.append(f"Error syncing dimension entries: {e}")

            operations_performed.append("Dimension table reconstruction completed")

            return RecoveryResult(
                success=True,
                strategy_used=operation.strategy,
                recovery_time=0,
                operations_performed=operations_performed
            )

        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e)
            )
    
    # Méthode de nettoyage des données orphelines
    def _recover_clean_orphaned_data(self, operation: RecoveryOperation) -> RecoveryResult:
        """Recover by cleaning orphaned data from the database.

        Args:
            operation: Recovery operation parameters.

        Returns:
            RecoveryResult with details of cleanup operations performed.
        """
        try:
            # Initialisation de la liste des opérations appliquées au jeu de données
            operations_performed = []
            
            # Utilisation du deleter pour nettoyer            
            deleter = DatabaseDeleter(self.conn, self.categorical_threshold, 
                                      enable_validation=False, auto_cleanup=True)
            # Nettoyage de la base de données
            cleanup_results = deleter.cleanup_database(comprehensive=True)
            # Parcours des résultats
            for category, result in cleanup_results.items():
                if result:
                    operations_performed.append(f"Cleaned {category}: {result}")
            
            return RecoveryResult(
                success=True,
                strategy_used=operation.strategy,
                recovery_time=0,
                operations_performed=operations_performed
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e)
            )
    # Méthode de récupération en validant la base de données et appliquant des corrections automatiques
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
            validation_report = self.auditor.validate_database(ValidationLevel.COMPREHENSIVE)
            operations_performed.append(f"Validated database: found {len(validation_report.issues)} issues")
            
            # Tentatives de correction automatique des issues
            fixed_count = 0
            
            # Parcours des problèmes
            for issue in validation_report.issues:
                try:
                    # Correction basée sur le type d'issue
                    if issue.issue_type.value == 'orphaned_reference':
                        # Nettoyage des références orphelines
                        cleaned = self._cleanup_orphaned_references_for_issue(issue)
                        if cleaned:
                            operations_performed.append(f"Fixed orphaned references: {issue.description}")
                            fixed_count += 1

                    elif issue.issue_type.value == 'data_integrity':
                        # Correction des problèmes d'intégrité des données
                        if 'null values' in issue.description.lower():
                            fixed = self._fix_null_value_issue(issue)
                            if fixed:
                                operations_performed.append(f"Fixed null value issue: {issue.description}")
                                fixed_count += 1

                    elif issue.issue_type.value == 'missing_metadata':
                        # Correction des problèmes de métadonnées manquantes
                        fixed = self._fix_missing_metadata_issue(issue)
                        if fixed:
                            operations_performed.append(f"Fixed missing metadata: {issue.description}")
                            fixed_count += 1

                    elif issue.issue_type.value == 'invalid_dimension':
                        # Correction des problèmes de dimension invalide
                        fixed = self._fix_invalid_dimension_issue(issue)
                        if fixed:
                            operations_performed.append(f"Fixed invalid dimension: {issue.description}")
                            fixed_count += 1

                    elif issue.issue_type.value == 'type_mismatch':
                        # Correction des conflits de types
                        fixed = self._fix_type_mismatch_issue(issue)
                        if fixed:
                            operations_performed.append(f"Fixed type mismatch: {issue.description}")
                            fixed_count += 1

                    elif issue.issue_type.value == 'constraint_violation':
                        # Correction des violations de contraintes
                        fixed = self._fix_constraint_violation_issue(issue)
                        if fixed:
                            operations_performed.append(f"Fixed constraint violation: {issue.description}")
                            fixed_count += 1

                    elif issue.issue_type.value == 'performance_issue':
                        # Pas de fix automatique pour les problèmes de performance, juste une recommandation
                        operations_performed.append(f"Performance recommendation: {issue.suggested_fix}")

                except Exception as e:
                    operations_performed.append(f"Failed to fix issue: {issue.description} - {e}")
            
            operations_performed.append(f"Successfully fixed {fixed_count}/{len(validation_report.issues)} issues")
            
            return RecoveryResult(
                success=True,
                strategy_used=operation.strategy,
                recovery_time=0,
                operations_performed=operations_performed,
                recommendations=["Re-run validation to verify all fixes were applied correctly"]
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e)
            )
    
    # Méthodes utilitaires privées
    # Méthode de validation d'une opération de 
    def _validate_recovery_operation(self, operation: RecoveryOperation, confirm_destructive: bool) -> bool:
        """Validate a recovery operation before execution.

        Args:
            operation: Recovery operation to validate.
            confirm_destructive: Whether destructive operations are confirmed.

        Returns:
            True if operation is valid and can proceed, False otherwise.
        """
        try:
            # Vérification des opérations destructrices (modifications in-place du catalogue)
            destructive_strategies = [RecoveryStrategy.REPAIR_SCHEMA]
            # Vérification de la confirmation d'une opération destructive de données
            if operation.strategy in destructive_strategies and not confirm_destructive:
                # Logging
                self.logger.error(f"Destructive operation {operation.strategy.value} requires confirmation")
                return False
            
            # Vérification du point de récupération si nécessaire
            if operation.target_recovery_point:
                if operation.target_recovery_point not in self._recovery_points:
                    # Logging
                    self.logger.error(f"Target recovery point {operation.target_recovery_point} not found")
                    return False
            
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error validating recovery operation: {e}")
            return False
    
    # Méthode de détermination de la stratégie de récupération
    def _determine_recovery_strategy(self, validation_report: Any, allow_destructive: bool) -> Optional[RecoveryStrategy]:
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
            severity_scores = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}

            # Comptage par type et sévérité
            issue_counts: Dict[str, Dict[str, int]] = {}

            for issue in validation_report.issues:
                issue_type = issue.issue_type.value
                severity = issue.severity.value

                if issue_type not in issue_counts:
                    issue_counts[issue_type] = {'count': 0, 'score': 0}

                issue_counts[issue_type]['count'] += 1
                issue_counts[issue_type]['score'] += severity_scores.get(severity, 1)

            # Tri par score décroissant pour identifier le problème dominant
            sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1]['score'], reverse=True)

            if not sorted_issues:
                return RecoveryStrategy.VALIDATE_AND_FIX

            dominant_issue = sorted_issues[0][0]
            dominant_score = sorted_issues[0][1]['score']
            critical_count = validation_report.get_critical_issues_count()

            # Stratégie basée sur le problème dominant avec seuils de gravité
            # Si trop de problèmes critiques → time-travel via l'historique DuckLake
            # (préférable à une restauration CSV qui ne couvre pas les données de faits)
            if critical_count >= 5:
                return RecoveryStrategy.USE_SNAPSHOT_HISTORY

            # Mapping type de problème → stratégie avec prise en compte du score
            strategy_mapping = {
                'orphaned_reference': RecoveryStrategy.CLEAN_ORPHANED_DATA,
                'invalid_dimension': RecoveryStrategy.REBUILD_DIMENSIONS,
                'schema_inconsistency': RecoveryStrategy.REPAIR_SCHEMA if allow_destructive else RecoveryStrategy.VALIDATE_AND_FIX,
                'missing_metadata': RecoveryStrategy.VALIDATE_AND_FIX,
                'data_integrity': RecoveryStrategy.VALIDATE_AND_FIX,
                'type_mismatch': RecoveryStrategy.VALIDATE_AND_FIX,
                'constraint_violation': RecoveryStrategy.VALIDATE_AND_FIX,
            }

            # Si le score dominant est élevé (>= 8), considérer une stratégie plus agressive
            if dominant_score >= 8 and allow_destructive:
                if dominant_issue == 'schema_inconsistency':
                    return RecoveryStrategy.REPAIR_SCHEMA
                elif dominant_issue in ['invalid_dimension', 'orphaned_reference']:
                    return RecoveryStrategy.REBUILD_DIMENSIONS

            return strategy_mapping.get(dominant_issue, RecoveryStrategy.VALIDATE_AND_FIX)

        except Exception as e:
            # Logging
            self.logger.error(f"Error determining recovery strategy: {e}")
            return None
    
    # Méthode auxiliaire de collecte des méta données d ela base
    def _collect_database_metadata(self) -> Dict[str, Any]:
        """Collect database metadata for recovery point creation.

        Returns:
            Dictionary containing timestamp, tables list, and health check info.
        """
        try:
            metadata = {
                'timestamp': time.time(),
                'tables': self._get_all_tables(),
                'health_check': {}
            }
            
            # Ajout des informations de santé
            try:
                metadata['health_check'] = self.auditor.get_quick_health_check()
            except:
                metadata['health_check'] = {'status': 'unknown'}
            
            return metadata
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error collecting database metadata: {e}")
            return {'error': str(e)}
    
    # Méthode auxiliaire d'extraction de l'ensemble des tables de la base de données
    def _get_all_tables(self) -> List[str]:
        """Get list of all tables in the database.

        Returns:
            List of table names, empty list on error.
        """
        try:
            # Exécution de la requête
            result = self.conn.execute("SHOW TABLES").fetchall()
            return [row[0] for row in result]
        except:
            return []
    
    # Méthode auxiliaire de vérification de l'existence d'une table dans la base de données
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database.

        Args:
            table_name: Name of the table to check.

        Returns:
            True if table exists, False otherwise.
        """
        try:
            # Exécution de la requête
            self.conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            return True
        except:
            return False
    
    # Méthode de sauvegarde des informations d'un point de récupération
    def _save_recovery_point_info(self, recovery_point: RecoveryPoint) -> None:
        """Save recovery point information to disk.

        Args:
            recovery_point: RecoveryPoint object to serialize and save.
        """
        try:
            # Identification du chemin
            info_file = Path(recovery_point.backup_path) / "recovery_point.json"
            
            # Sérialisation des données
            data = {
                'recovery_id': recovery_point.recovery_id,
                'timestamp': recovery_point.timestamp,
                'backup_type': recovery_point.backup_type.value,
                'backup_path': recovery_point.backup_path,
                'metadata': recovery_point.metadata,
                'description': recovery_point.description,
                'validation_report_summary': {
                    'total_issues': len(recovery_point.validation_report.issues) if recovery_point.validation_report else 0,
                    'critical_issues': recovery_point.validation_report.get_critical_issues_count() if recovery_point.validation_report else 0
                } if recovery_point.validation_report else None
            }
            # Exportation en json
            with open(info_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                
        except Exception as e:
            # Logging
            self.logger.error(f"Error saving recovery point info: {e}")
    
    # Méthode de chargement des points de récupération existants
    def _load_existing_recovery_points(self) -> None:
        """Load existing recovery points from backup directory."""
        try:
            # Vérification que le répertoire existe
            if not self.backup_dir.exists():
                return
            
            # Parcours des répertoires
            for backup_dir in self.backup_dir.iterdir():
                if backup_dir.is_dir():
                    # Idnetification du chemin
                    info_file = backup_dir / "recovery_point.json"
                    # Extraction des informations
                    if info_file.exists():
                        try:
                            # Chargement du json
                            with open(info_file, 'r') as f:
                                data = json.load(f)
                            
                            # Reconstruction de l'objet RecoveryPoint
                            recovery_point = RecoveryPoint(
                                recovery_id=data['recovery_id'],
                                timestamp=data['timestamp'],
                                backup_type=BackupType(data['backup_type']),
                                backup_path=data['backup_path'],
                                metadata=data.get('metadata', {}),
                                description=data.get('description', '')
                            )
                            # Ajout aux points de sauvegarde
                            self._recovery_points[recovery_point.recovery_id] = recovery_point
                            
                        except Exception as e:
                            # Logging
                            self.logger.warning(f"Error loading recovery point from {backup_dir}: {e}")
            # Logging
            self.logger.info(f"Loaded {len(self._recovery_points)} existing recovery points")
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error loading existing recovery points: {e}")
    
    # Méthode auxiliaire de nettoyage des anciens points de récupération
    def _cleanup_old_recovery_points(self) -> None:
        """Clean up recovery points older than max_backup_age_days."""
        try:
            # Détermination du moment à partir duquel supprimer des points de sauvegarde
            cutoff_time = time.time() - (self.max_backup_age_days * 24 * 60 * 60)
            # Détermination des points de sauvegarde à supprimer
            old_points = [
                rp for rp in self._recovery_points.values() 
                if rp.timestamp < cutoff_time
            ]
            
            # Suppression des points
            for recovery_point in old_points:
                self.delete_recovery_point(recovery_point.recovery_id)
            
            if old_points:
                # Logging
                self.logger.info(f"Cleaned up {len(old_points)} old recovery points")
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error cleaning up old recovery points: {e}")
    
    def _cleanup_orphaned_references_for_issue(self, issue: ValidationIssue) -> bool:
        """
        Clean orphaned references for a specific issue.

        Args:
            issue: The validation issue to fix

        Returns:
            True if cleanup was successful
        """
        try:
            col_name = issue.column_name
            if not col_name:
                return False

            dim_table = f"dim_{col_name}"

            if issue.table_name == 'fact_table':
                # Références orphelines dans fact_table → mettre à NULL
                self.conn.execute(f"""
                    UPDATE fact_table SET {col_name} = NULL
                    WHERE {col_name} NOT IN (SELECT value FROM {dim_table})
                """)
            else:
                # Entrées orphelines dans dimension → supprimer
                self.conn.execute(f"""
                    DELETE FROM {dim_table}
                    WHERE value NOT IN (SELECT DISTINCT {col_name} FROM fact_table WHERE {col_name} IS NOT NULL)
                """)

            return True

        except Exception as e:
            self.logger.error(f"Failed to cleanup orphaned references: {e}")
            return False

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

            if not col_name or table_name != 'fact_table':
                return False

            # Comptage des lignes avant suppression
            initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]

            # Suppression des lignes contenant des valeurs nulles
            self.conn.execute(f"DELETE FROM fact_table WHERE {col_name} IS NULL")

            # Comptage des lignes après suppression
            final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            deleted_count = initial_count - final_count

            self.logger.info(f"Deleted {deleted_count} rows with NULL values in {col_name}")
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
            # Cas 1: Colonne dans fact_table mais pas dans metadata → ajout aux métadonnées
            if issue.table_name == 'fact_table' and issue.column_name:
                col_name = issue.column_name
                # Inférence du type depuis la colonne
                result = self.conn.execute(f"DESCRIBE fact_table").fetchall()
                col_info = next((row for row in result if row[0] == col_name), None)
                if col_info:
                    sql_type = col_info[1]
                    self.conn.execute("""
                        INSERT INTO metadata (name, label, python_type, sql_type, is_categorical, is_primary_key)
                        VALUES (?, ?, ?, ?, FALSE, FALSE)
                    """, [col_name, col_name.replace('_', ' ').title(), 'object', sql_type])
                    self.logger.info(f"Added missing metadata for column {col_name}")
                    return True

            # Cas 2: Colonne dans metadata mais pas dans fact_table → suppression des métadonnées
            if issue.table_name == 'metadata' and issue.column_name:
                self.conn.execute("DELETE FROM metadata WHERE name = ?", [issue.column_name])
                self.logger.info(f"Removed orphaned metadata for column {issue.column_name}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to fix missing metadata: {e}")
            return False

    def _fix_invalid_dimension_issue(self, issue: ValidationIssue) -> bool:
        """
        Fix invalid dimension table issues.

        Args:
            issue: The validation issue to fix

        Returns:
            True if fix was applied successfully
        """
        try:
            col_name = issue.column_name
            if not col_name:
                return False

            dim_table = f"dim_{col_name}"
            dim_mgr = DimensionManager(self.conn, self.categorical_threshold)

            # Cas 1: Table de dimension manquante → création
            if "missing" in issue.description.lower():
                dim_mgr.create_dimension_table(col_name)
                values = self.conn.execute(
                    f"SELECT DISTINCT {col_name} FROM fact_table WHERE {col_name} IS NOT NULL"
                ).pl()[col_name]
                dim_mgr.update_dimension_values(col_name, values)
                self.logger.info(f"Created missing dimension table {dim_table}")
                return True

            # Cas 2: Table de dimension corrompue → reconstruction
            if self._table_exists(dim_table):
                self.conn.execute(f"DROP TABLE {dim_table}")
                dim_mgr.create_dimension_table(col_name)
                values = self.conn.execute(
                    f"SELECT DISTINCT {col_name} FROM fact_table WHERE {col_name} IS NOT NULL"
                ).pl()[col_name]
                dim_mgr.update_dimension_values(col_name, values)
                self.logger.info(f"Rebuilt corrupted dimension table {dim_table}")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to fix invalid dimension: {e}")
            return False

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

            # Élargissement du type vers le type le moins restrictif (généralement VARCHAR)
            # Note: DuckDB ne supporte pas ALTER COLUMN TYPE directement, donc recréation nécessaire
            self.conn.execute(f"""
                UPDATE metadata SET sql_type = 'VARCHAR', python_type = 'object'
                WHERE name = ?
            """, [col_name])

            self.logger.info(f"Updated type for column {col_name} to VARCHAR")
            return True

        except Exception as e:
            self.logger.error(f"Failed to fix type mismatch: {e}")
            return False

    def _fix_constraint_violation_issue(self, issue: ValidationIssue) -> bool:
        """
        Fix constraint violation issues (e.g., remove duplicates).

        Args:
            issue: The validation issue to fix

        Returns:
            True if fix was applied successfully
        """
        try:
            # Traitement des violations de contraintes (doublons, valeurs invalides, etc.)
            if 'duplicate' in issue.description.lower():
                # Suppression des doublons en gardant la première occurrence
                self.conn.execute("""
                    CREATE OR REPLACE TABLE fact_table AS
                    SELECT * FROM (
                        SELECT *, ROW_NUMBER() OVER (PARTITION BY * ORDER BY 1) as rn
                        FROM fact_table
                    ) WHERE rn = 1
                """)
                # Suppression de la colonne temporaire
                self.conn.execute("ALTER TABLE fact_table DROP COLUMN rn")
                self.logger.info("Removed duplicate rows from fact_table")
                return True

            return False

        except Exception as e:
            self.logger.error(f"Failed to fix constraint violation: {e}")
            return False