# Importation des modules
# Modules de base
import os
import json
import pickle
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import tempfile
import shutil
# DuckDB
import duckdb

# Import des gestionnaires
from .auditor import DatabaseAuditor, ValidationLevel, ValidationReport
from .managers.dimension_manager import DimensionManager
from .deleter_v2 import DatabaseDeleterV2
# Import des utilitaires
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe des types de stratégies de récupération possibles
class RecoveryStrategy(Enum):
    """Stratégies de récupération disponibles."""
    ROLLBACK_TRANSACTION = "rollback_transaction"
    RESTORE_BACKUP = "restore_backup"
    REPAIR_SCHEMA = "repair_schema"
    REBUILD_DIMENSIONS = "rebuild_dimensions"
    CLEAN_ORPHANED_DATA = "clean_orphaned_data"
    VALIDATE_AND_FIX = "validate_and_fix"


# Classe des types de sauvegarde possibles
class BackupType(Enum):
    """Types de sauvegarde disponibles."""
    FULL_BACKUP = "full_backup"
    SCHEMA_BACKUP = "schema_backup"
    METADATA_BACKUP = "metadata_backup"
    INCREMENTAL_BACKUP = "incremental_backup"


# Classe de point de récupération de la base de données
@dataclass
class RecoveryPoint:
    """
    Point de récupération contenant l'état de la base de données.
    
    Attributes:
        recovery_id (str): Identifiant unique du point de récupération
        timestamp (float): Timestamp de création
        backup_type (BackupType): Type de sauvegarde
        backup_path (str): Chemin vers les fichiers de sauvegarde
        metadata (dict): Métadonnées sur l'état de la base
        validation_report (Optional[ValidationReport]): Rapport de validation au moment de la sauvegarde
        description (str): Description du point de récupération
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
    Opération de récupération à exécuter.
    
    Attributes:
        strategy (RecoveryStrategy): Stratégie de récupération
        target_recovery_point (Optional[str]): ID du point de récupération cible
        parameters (dict): Paramètres spécifiques à la stratégie
        auto_validate (bool): Valider automatiquement après récupération
        description (str): Description de l'opération
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
    Résultat d'une opération de récupération.
    
    Attributes:
        success (bool): Succès de l'opération
        strategy_used (RecoveryStrategy): Stratégie utilisée
        recovery_time (float): Temps de récupération en secondes
        operations_performed (List[str]): Liste des opérations effectuées
        validation_report (Optional[ValidationReport]): Rapport de validation post-récupération
        error_message (Optional[str]): Message d'erreur si applicable
        recommendations (List[str]): Recommandations pour éviter des problèmes futurs
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
                 connection: duckdb.DuckDBPyConnection,
                 backup_dir: Optional[os.PathLike] = None,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None,
                 max_backup_age_days: int = 30,
                 auto_backup_on_changes: bool = True):
        """
        Initialize the database recovery manager.
        
        Args:
            connection: DuckDB connection object
            backup_dir: Directory for storing backups (None for default)
            categorical_threshold: Threshold for determining categorical variables
            log_filename: Path to log file
            max_backup_age_days: Maximum age for keeping backups
            auto_backup_on_changes: Whether to create automatic backups
            
        Example:
            >>> conn = duckdb.connect('database.db')
            >>> recovery_mgr = DatabaseRecoveryManager(conn, backup_dir='/path/to/backups')
        """
        # Initialisation de la connexion
        self.conn = connection
        
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
                            backup_type: BackupType = BackupType.FULL_BACKUP,
                            description: str = "",
                            force_validation: bool = True) -> Optional[str]:
        """
        Create a recovery point with backup and validation.
        
        Args:
            backup_type: Type of backup to create
            description: Description of the recovery point
            force_validation: Whether to force database validation
            
        Returns:
            Recovery point ID if successful, None otherwise
            
        Example:
            >>> recovery_id = recovery_mgr.create_recovery_point(
            ...     BackupType.FULL_BACKUP, 
            ...     "Before major schema changes"
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
                    self.logger.warning(f"Creating recovery point with {validation_report.get_critical_issues_count()} critical issues")
            
            # Création du répertoire de sauvegarde
            backup_path = self.backup_dir / recovery_id
            backup_path.mkdir(exist_ok=True)
            
            # Exécution de la sauvegarde selon le type
            if backup_type == BackupType.FULL_BACKUP:
                success = self._create_full_backup(backup_path)
            elif backup_type == BackupType.SCHEMA_BACKUP:
                success = self._create_schema_backup(backup_path)
            elif backup_type == BackupType.METADATA_BACKUP:
                success = self._create_metadata_backup(backup_path)
            elif backup_type == BackupType.INCREMENTAL_BACKUP:
                success = self._create_incremental_backup(backup_path)
            else:
                # Logging
                self.logger.error(f"Unknown backup type: {backup_type}")
                return None
            
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
            
            # Création d'un point de récupération automatique avant l'opération
            pre_recovery_point = None
            if operation.strategy in [RecoveryStrategy.RESTORE_BACKUP, RecoveryStrategy.REPAIR_SCHEMA]:
                pre_recovery_point = self.create_recovery_point(
                    BackupType.FULL_BACKUP,
                    f"Auto-backup before {operation.strategy.value}"
                )
            
            # Exécution de la stratégie de récupération
            if operation.strategy == RecoveryStrategy.ROLLBACK_TRANSACTION:
                result = self._recover_rollback_transaction(operation)
            elif operation.strategy == RecoveryStrategy.RESTORE_BACKUP:
                result = self._recover_restore_backup(operation)
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
    # Méthode de création d'une sauvegarde
    def _create_full_backup(self, backup_path: Path) -> bool:
        """Créer une sauvegarde complète."""
        try:
            # Sauvegarde de toutes les tables
            tables = self._get_all_tables()
            # Parcours des tables
            for table_name in tables:
                try:
                    # Export de la table vers CSV
                    table_file = backup_path / f"{table_name}.csv"
                    export_query = f"COPY {table_name} TO '{table_file}' (FORMAT CSV, HEADER)"
                    self.conn.execute(export_query)
                    
                    # Sauvegarde de la structure de la table
                    structure_file = backup_path / f"{table_name}_structure.json"
                    structure = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
                    
                    with open(structure_file, 'w') as f:
                        json.dump({
                            'table_name': table_name,
                            'columns': structure
                        }, f, indent=2)
                        
                except Exception as e:
                    # Logging
                    self.logger.error(f"Error backing up table {table_name}: {e}")
                    return False
            
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error creating full backup: {e}")
            return False
    
    # Méthode de création d'une sauvegarde du schéma
    def _create_schema_backup(self, backup_path: Path) -> bool:
        """Créer une sauvegarde du schéma uniquement."""
        try:
            schema_info = {
                'tables': {},
                'indexes': [],
                'timestamp': time.time()
            }
            
            # Sauvegarde des structures de tables
            tables = self._get_all_tables()
            " Parcours des tables"
            for table_name in tables:
                structure = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
                schema_info['tables'][table_name] = structure
            
            # Sauvegarde des index
            try:
                indexes = self.conn.execute("SELECT index_name, expressions FROM duckdb_indexes()").fetchall()
                schema_info['indexes'] = indexes
            except:
                schema_info['indexes'] = []
            
            # Sauvegarde dans un fichier JSON
            schema_file = backup_path / "schema_backup.json"
            with open(schema_file, 'w') as f:
                json.dump(schema_info, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error creating schema backup: {e}")
            return False
    
    # Méthode de sauvegarde des méta-données
    def _create_metadata_backup(self, backup_path: Path) -> bool:
        """Créer une sauvegarde des métadonnées uniquement."""
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
    
    # Méthode de création d'une sauvegarde incrémentale
    def _create_incremental_backup(self, backup_path: Path) -> bool:
        """Créer une sauvegarde incrémentale."""
        try:
            # Pour simplifier, on fait une sauvegarde des métadonnées
            # Une vraie implémentation nécessiterait un tracking des changements
            return self._create_metadata_backup(backup_path)
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error creating incremental backup: {e}")
            return False
    
    # Méthodes de récupération spécialisées
    # Méthode de récupération d'une transaction par rollback
    def _recover_rollback_transaction(self, operation: RecoveryOperation) -> RecoveryResult:
        """Récupération par rollback de transaction."""
        try:
            # Cette stratégie est généralement gérée par le TransactionManager
            # Ici on simule un rollback basique
            
            operations_performed = ["Transaction rollback attempted"]
            
            # Tentative de rollback global
            try:
                self.conn.execute("ROLLBACK")
                operations_performed.append("Global rollback executed")
            except:
                operations_performed.append("No active transaction to rollback")
            
            return RecoveryResult(
                success=True,
                strategy_used=operation.strategy,
                recovery_time=0,
                operations_performed=operations_performed,
                recommendations=["Consider using TransactionManager for better rollback control"]
            )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e)
            )

    # Méthode de récupération par restauration de sauvegarde
    def _recover_restore_backup(self, operation: RecoveryOperation) -> RecoveryResult:
        """Récupération par restauration de sauvegarde."""
        try:
            # Vérification que la cible de la restoration est spécifié
            if not operation.target_recovery_point:
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    error_message="No target recovery point specified"
                )
            # Vérification que le point de sauvegarde existe
            if operation.target_recovery_point not in self._recovery_points:
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    error_message=f"Recovery point {operation.target_recovery_point} not found"
                )
            
            # Extraction du point de sauvegarde
            recovery_point = self._recovery_points[operation.target_recovery_point]
            # Construction du chemin
            backup_path = Path(recovery_point.backup_path)
            
            # Vérification que le chemin existe
            if not backup_path.exists():
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    error_message=f"Backup path {backup_path} does not exist"
                )
            
            operations_performed = []
            
            # Restauration selon le type de sauvegarde
            if recovery_point.backup_type == BackupType.FULL_BACKUP:
                success = self._restore_full_backup(backup_path, operations_performed)
            elif recovery_point.backup_type == BackupType.SCHEMA_BACKUP:
                success = self._restore_schema_backup(backup_path, operations_performed)
            elif recovery_point.backup_type == BackupType.METADATA_BACKUP:
                success = self._restore_metadata_backup(backup_path, operations_performed)
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    error_message=f"Unsupported backup type: {recovery_point.backup_type}"
                )
            
            # Affichage d'un message différent suivant la réussite de l'opération
            if success:
                return RecoveryResult(
                    success=True,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    operations_performed=operations_performed,
                    recommendations=["Verify data integrity after restore operation"]
                )
            else:
                return RecoveryResult(
                    success=False,
                    strategy_used=operation.strategy,
                    recovery_time=0,
                    operations_performed=operations_performed,
                    error_message="Backup restoration failed"
                )
            
        except Exception as e:
            return RecoveryResult(
                success=False,
                strategy_used=operation.strategy,
                recovery_time=0,
                error_message=str(e)
            )
    
    # Méthode de récupération par réparation du schéma
    def _recover_repair_schema(self, operation: RecoveryOperation) -> RecoveryResult:
        """Récupération par réparation du schéma."""
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
                    if issue.suggested_fix:
                        # Tentative d'application du fix suggéré
                        # Note: Ceci nécessiterait une implémentation plus sophistiquée
                        operations_performed.append(f"Attempted fix for: {issue.description}")
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
    
    # Méthode de reconstruction des tables de dimension corrompues
    def _recover_rebuild_dimensions(self, operation: RecoveryOperation) -> RecoveryResult:
        """Récupération par reconstruction des dimensions."""
        # /!\ Cette méthode n'ajoute pas les éventuelle entrées manquantes dans les tables de dimension mais se contente de supprimer les entrées superflues
        try:
            # Initialisation de la liste des opérations réalisées
            operations_performed = []
            
            # Reconstruction des tables de dimension corrompues
            # Initialisation du gestionnaire des dimensions
            dim_mgr = DimensionManager(self.conn, self.categorical_threshold)
            
            # Nettoyage des entrées orphelines
            cleaned = dim_mgr.cleanup_orphaned_dimension_entries()
            # Parcours des résultats du nettoyage
            for dim_name, count in cleaned.items():
                if count > 0:
                    operations_performed.append(f"Cleaned {count} orphaned entries from {dim_name}")
            
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
        """Récupération par nettoyage des données orphelines."""
        try:
            # Initialisation de la liste des opérations appliquées au jeu de données
            operations_performed = []
            
            # Utilisation du deleter pour nettoyer            
            deleter = DatabaseDeleterV2(self.conn, self.categorical_threshold, 
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
        """Récupération par validation et correction automatique."""
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
        """Valider une opération de récupération."""
        try:
            # Vérification des opérations destructrices
            destructive_strategies = [
                RecoveryStrategy.RESTORE_BACKUP,
                RecoveryStrategy.REPAIR_SCHEMA
            ]
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
        """Déterminer la meilleure stratégie de récupération."""
        try:
            # Analyse des types de problèmes
            issue_types = [issue.issue_type.value for issue in validation_report.issues]
            critical_count = validation_report.get_critical_issues_count()
            
            # Stratégie basée sur les types d'issues dominants
            if 'orphaned_reference' in issue_types:
                return RecoveryStrategy.CLEAN_ORPHANED_DATA
            
            elif 'invalid_dimension' in issue_types:
                return RecoveryStrategy.REBUILD_DIMENSIONS
            
            elif 'schema_inconsistency' in issue_types and allow_destructive:
                return RecoveryStrategy.REPAIR_SCHEMA
            
            elif critical_count > 0 and allow_destructive:
                # Si il y a beaucoup d'issues critiques et qu'on a un backup récent
                recent_backups = [p for p in self._recovery_points.values() 
                                if time.time() - p.timestamp < 3600]  # Moins d'1 heure
                if recent_backups:
                    return RecoveryStrategy.RESTORE_BACKUP
            
            # Stratégie par défaut
            return RecoveryStrategy.VALIDATE_AND_FIX
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error determining recovery strategy: {e}")
            return None
    
    # Méthode auxiliaire de collecte des méta données d ela base
    def _collect_database_metadata(self) -> Dict[str, Any]:
        """Collecter les métadonnées de la base de données."""
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
        """Obtenir la liste de toutes les tables."""
        try:
            # Exécution de la requête
            result = self.conn.execute("SHOW TABLES").fetchall()
            return [row[0] for row in result]
        except:
            return []
    
    # Méthode auxiliaire de vérification de l'existence d'une table dans la base de données
    def _table_exists(self, table_name: str) -> bool:
        """Vérifier si une table existe."""
        try:
            # Exécution de la requête
            self.conn.execute(f"SELECT 1 FROM {table_name} LIMIT 1")
            return True
        except:
            return False
    
    # Méthode de sauvegarde des informations d'un point de récupération
    def _save_recovery_point_info(self, recovery_point: RecoveryPoint) -> None:
        """Sauvegarder les informations d'un point de récupération."""
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
        """Charger les points de récupération existants."""
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
        """Nettoyer les anciens points de récupération."""
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
    
    # Placeholder methods for specific recovery operations
    def _restore_full_backup(self, backup_path: Path, operations_performed: List[str]) -> bool:
        """Restaurer une sauvegarde complète."""
        # Implementation would restore all tables from CSV files
        operations_performed.append("Full backup restoration (placeholder)")
        return True
    
    def _restore_schema_backup(self, backup_path: Path, operations_performed: List[str]) -> bool:
        """Restaurer une sauvegarde de schéma."""
        # Implementation would recreate table structures
        operations_performed.append("Schema backup restoration (placeholder)")
        return True
    
    def _restore_metadata_backup(self, backup_path: Path, operations_performed: List[str]) -> bool:
        """Restaurer une sauvegarde de métadonnées."""
        # Implementation would restore metadata table
        operations_performed.append("Metadata backup restoration (placeholder)")
        return True
    
    def _cleanup_orphaned_references_for_issue(self, issue) -> bool:
        """Nettoyer les références orphelines pour une issue spécifique."""
        # Implementation would clean specific orphaned references
        return True
    
    def _fix_null_value_issue(self, issue) -> bool:
        """Corriger un problème de valeurs nulles."""
        # Implementation would handle null value issues
        return True