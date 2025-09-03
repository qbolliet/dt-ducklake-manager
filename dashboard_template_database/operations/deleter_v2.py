# Importation des modules
# Modules de base
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
# DuckDB
import duckdb

# Import des gestionnaires
from .managers.base_manager import BaseSchemaManager
from .managers.dimension_manager import DimensionManager
from .managers.data_manager import DataManager
from .managers.transaction_manager import (
    TransactionManager, TransactionOperation, TransactionState
)
from .auditor import DatabaseAuditor, ValidationLevel, ValidationIssue

# Import des utilitaires
from ..utils.data_processing import _build_where_clause

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Classe de suppression de données dans la base
class DatabaseDeleterV2(BaseSchemaManager):
    """
    Database deleter using modular architecture.
    
    Provides atomic, transactional deletion operations on DuckDB databases with proper
    validation, error recovery, and state consistency. Uses specialized managers
    for different aspects of database operations.
    
    Attributes:
        dimension_mgr (DimensionManager): Manages dimension table operations
        data_mgr (DataManager): Manages fact table operations
        transaction_mgr (TransactionManager): Manages transactions and rollback
        auditor (DatabaseAuditor): Validates database state and operations
        enable_validation (bool): Whether to enable validation
        auto_cleanup (bool): Whether to automatically clean up orphaned data
    """
    # Initialisation
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None,
                 enable_validation: bool = True,
                 auto_cleanup: bool = True):
        """
        Initialize the refactored database deleter.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Threshold for determining categorical variables
            log_filename: Path to log file
            enable_validation: Whether to enable pre/post operation validation
            auto_cleanup: Whether to automatically clean up orphaned data
            
        Example:
            >>> conn = duckdb.connect('database.db')
            >>> deleter = DatabaseDeleterV2(conn, enable_validation=True, auto_cleanup=True)
        """
        # Initialisation du parent
        super().__init__(connection, categorical_threshold, log_filename)
        
        # Initialisation des gestionnaires spécialisés
        self.dimension_mgr = DimensionManager(
            connection, categorical_threshold, log_filename
        )
        
        self.data_mgr = DataManager(
            connection, categorical_threshold, log_filename
        )
        
        self.transaction_mgr = TransactionManager(
            connection, categorical_threshold, log_filename
        )
        
        self.auditor = DatabaseAuditor(
            connection, categorical_threshold, log_filename
        ) if enable_validation else None
        
        # Configuration
        self.enable_validation = enable_validation
        self.auto_cleanup = auto_cleanup
    
    # Méthode de validation de l'opération de suppression avant son exécution
    def validate_operation(self, operation_type: str, **kwargs) -> bool:
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
        validation_report = self.auditor.validate_operation_preconditions(operation_type, **kwargs)
        
        # Vérification des issues critiques
        critical_issues = validation_report.get_critical_issues_count()
        if critical_issues > 0:
            # Logging
            self.logger.error(f"Critical validation issues found for {operation_type} operation:")
            for issue in validation_report.issues:
                if issue.severity.value == 'critical':
                    self.logger.error(f"  - {issue.description}")
            return False
        
        # Avertissements pour les issues de haute priorité
        high_issues = [issue for issue in validation_report.issues if issue.severity.value == 'high']
        if high_issues:
            # Logging
            self.logger.warning(f"High priority validation issues found for {operation_type} operation:")
            for issue in high_issues:
                self.logger.warning(f"  - {issue.description}")
        
        return True
    
    # Méthode principale de suppression de lignes
    def delete_rows(self, 
                   filters: Optional[Union[str, List, Dict]] = None,
                   use_transaction: bool = True,
                   perform_cleanup: Optional[bool] = None) -> int:
        """
        Delete rows from fact table based on filters with atomic operations.
        
        Args:
            filters: Filter conditions (string SQL condition or structured filters)
            use_transaction: Whether to use database transactions
            perform_cleanup: Whether to cleanup orphaned data (None = use auto_cleanup setting)
            
        Returns:
            Number of rows deleted, -1 if operation failed
            
        Example:
            >>> # Using string filter
            >>> deleted = deleter.delete_rows("status = 'inactive'")
            >>> 
            >>> # Using structured filter
            >>> filters = [('status', '=', 'inactive'), ('date', '<', '2023-01-01')]
            >>> deleted = deleter.delete_rows(filters, use_transaction=True)
        """
        # Validation préalable
        if not self.validate_operation('delete', filters=filters):
            self.logger.error("Pre-delete validation failed")
            return -1
        
        # Configuration du nettoyage
        if perform_cleanup is None:
            perform_cleanup = self.auto_cleanup
        
        # Logging
        self.logger.info(f"Starting row deletion (transaction: {use_transaction}, cleanup: {perform_cleanup})")
        
        # Suppression des lignes
        if use_transaction:
            # De manière transactionnelle
            return self._delete_rows_transactional(filters, perform_cleanup)
        else:
            # Directement
            return self._delete_rows_direct(filters, perform_cleanup)
    
    # Méthode de suppression des lignes de manière transactionnelle
    def _delete_rows_transactional(self, 
                                 filters: Optional[Union[str, List, Dict]], 
                                 perform_cleanup: bool) -> int:
        """Suppression transactionnelle de lignes."""
        
        # Début de la transaction
        tx_id = self.transaction_mgr.begin_transaction("Row deletion with cleanup and validation")
        
        try:
            # Comptage initial pour le rollback
            initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            # Étape 1: Suppression des lignes
            operation = TransactionOperation(
                operation_type='delete_rows',
                operation_func=self._execute_row_deletion,
                operation_args=(filters,),
                rollback_func=self._restore_deleted_rows,
                rollback_args=(filters, initial_count),
                description="Delete rows from fact table"
            )
            
            # Annulation si l'opération ne peut pas être ajoutée ou exécutée
            if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                self.transaction_mgr.rollback_transaction(tx_id)
                return -1
            
            if not self.transaction_mgr.execute_operation(tx_id):
                self.transaction_mgr.rollback_transaction(tx_id)
                return -1
            
            # Calcul du nombre de lignes supprimées
            current_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            rows_deleted = initial_count - current_count
            
            if rows_deleted == 0:
                # Pas de suppression, commit simple
                self.transaction_mgr.commit_transaction(tx_id)
                return 0
            
            # Création d'un savepoint avant le nettoyage
            self.transaction_mgr.create_savepoint(tx_id, "before_cleanup")
            
            # Étape 2: Nettoyage des données orphelines si activé
            if perform_cleanup:
                operation = TransactionOperation(
                    operation_type='cleanup_orphaned',
                    operation_func=self._cleanup_orphaned_data_comprehensive,
                    operation_args=(),
                    rollback_func=self._restore_orphaned_data,
                    description="Clean up orphaned dimension entries and null columns"
                )
                
                if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                    # Annulation du nettoyage
                    self.transaction_mgr.rollback_to_savepoint(tx_id, "before_cleanup")
                    # Logging
                    self.logger.warning("Cleanup failed, but row deletion completed")
                else:
                    self.transaction_mgr.execute_operation(tx_id)  # Non-critique si échoue
            
            # Validation post-suppression
            if self.enable_validation and self.auditor:
                # Audit d ela base de données
                validation_report = self.auditor.validate_database(ValidationLevel.BASIC)
                # Identification des problèmes critiques
                critical_issues = validation_report.get_critical_issues_count()
                if critical_issues > 0:
                    # Logging
                    self.logger.error("Critical issues found after deletion, rolling back")
                    # Annulation de la transaction
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return -1
            
            # Commit de la transaction
            if self.transaction_mgr.commit_transaction(tx_id):
                # Logging
                self.logger.info(f"Row deletion completed successfully: {rows_deleted} rows deleted")
                return rows_deleted
            else:
                # Logging
                self.logger.error("Failed to commit deletion transaction")
                return -1
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error during transactional deletion: {e}")
            # Annulation de la transaction
            self.transaction_mgr.rollback_transaction(tx_id)
            return -1
    
    # Méthode de suppression directe des lignes
    def _delete_rows_direct(self, 
                          filters: Optional[Union[str, List, Dict]], 
                          perform_cleanup: bool) -> int:
        """Suppression directe sans transaction."""
        try:
            # Exécution de la suppression
            rows_deleted = self._execute_row_deletion(filters)
            
            if rows_deleted > 0 and perform_cleanup:
                # Nettoyage des données orphelines
                self._cleanup_orphaned_data_comprehensive()
            
            # Logging
            self.logger.info(f"Row deletion completed (direct mode): {rows_deleted} rows deleted")
            return rows_deleted
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error during direct deletion: {e}")
            return -1
    
    # Méthode principale de suppression de colonnes
    def delete_columns(self, 
                      columns: List[str], 
                      use_transaction: bool = True,
                      validate_dependencies: bool = True) -> Dict[str, bool]:
        """
        Delete columns from fact table and related structures with dependency analysis.
        
        Args:
            columns: List of column names to delete
            use_transaction: Whether to use database transactions
            validate_dependencies: Whether to validate column dependencies
            
        Returns:
            Dictionary mapping column names to success status
            
        Example:
            >>> results = deleter.delete_columns(['old_col1', 'old_col2'])
            >>> for col, success in results.items():
            ...     print(f"Column {col}: {'deleted' if success else 'failed'}")
        """
        # Validation préalable
        if not self.validate_operation('drop_column', columns=columns):
            # Logging
            self.logger.error("Pre-column-deletion validation failed")
            return {col: False for col in columns}
        
        # Analyse des dépendances si activée
        if validate_dependencies:
            dependency_report = self._analyze_column_dependencies(columns)
            if dependency_report['has_critical_dependencies']:
                # Logging
                self.logger.error("Critical dependencies found, aborting columns deletion")
                return {col: False for col in columns}
        
        # Logging
        self.logger.info(f"Starting columns deletion (transaction: {use_transaction})")
        
        # Suppression des colonnes
        if use_transaction:
            # Avec transaction
            return self._delete_columns_transactional(columns)
        else:
            # Directement
            return self._delete_columns_direct(columns)
    
    # Méthode de suppression de colonnes de manière transactionnelle
    def _delete_columns_transactional(self, columns: List[str]) -> Dict[str, bool]:
        """Suppression transactionnelle de colonnes."""
        
        # Début de la transaction
        tx_id = self.transaction_mgr.begin_transaction("Column deletion with dependency management")
        
        # Initialisation du dictionnaire résultat
        results = {}
        
        try:
            # Filtrage des colonnes existantes
            existing_columns = self._get_fact_table_columns()
            valid_columns = [col for col in columns if col in existing_columns]
            
            # Vérification que les colonnes sont valides
            if not valid_columns:
                # Logging
                self.logger.warning("No valid columns found for deletion")
                # Commit de la transaction
                self.transaction_mgr.commit_transaction(tx_id)
                return {col: False for col in columns}
            
            # Traitement de chaque colonne
            for column in valid_columns:
                try:
                    # Étape 1: Suppression des index liés à la colonne
                    operation = TransactionOperation(
                        operation_type='drop_indexes',
                        operation_func=self._drop_column_indexes_safe,
                        operation_args=(column,),
                        rollback_func=self._restore_column_indexes,
                        rollback_args=(column,),
                        description=f"Drop indexes for column {column}"
                    )
                    
                    self.transaction_mgr.add_operation(tx_id, **operation.__dict__)
                    self.transaction_mgr.execute_operation(tx_id)
                    
                    # Étape 2: Suppression de la table de dimension si applicable
                    if self._is_dimension_column(column):
                        operation = TransactionOperation(
                            operation_type='drop_dimension',
                            operation_func=self.dimension_mgr.delete_dimension_table,
                            operation_args=(column,),
                            rollback_func=self._restore_dimension_table,
                            rollback_args=(column,),
                            description=f"Drop dimension table for {column}"
                        )
                        
                        self.transaction_mgr.add_operation(tx_id, **operation.__dict__)
                        self.transaction_mgr.execute_operation(tx_id)
                    
                    # Étape 3: Suppression de la colonne de la fact table
                    operation = TransactionOperation(
                        operation_type='drop_column',
                        operation_func=self._drop_fact_table_column,
                        operation_args=(column,),
                        rollback_func=self._restore_fact_table_column,
                        rollback_args=(column,),
                        description=f"Drop column {column} from fact table"
                    )
                    
                    self.transaction_mgr.add_operation(tx_id, **operation.__dict__)
                    self.transaction_mgr.execute_operation(tx_id)
                    
                    # Étape 4: Suppression des métadonnées
                    operation = TransactionOperation(
                        operation_type='drop_metadata',
                        operation_func=self.delete_column_metadata,
                        operation_args=(column,),
                        rollback_func=self._restore_column_metadata,
                        rollback_args=(column,),
                        description=f"Drop metadata for column {column}"
                    )
                    
                    self.transaction_mgr.add_operation(tx_id, **operation.__dict__)
                    self.transaction_mgr.execute_operation(tx_id)
                    
                    results[column] = True
                    
                except Exception as e:
                    # Logging
                    self.logger.error(f"Error processing column {column}: {e}")
                    results[column] = False
                    # Continue avec les autres colonnes
            
            # Nettoyage final des index orphelins
            operation = TransactionOperation(
                operation_type='cleanup_indexes',
                operation_func=self._cleanup_orphaned_indexes,
                operation_args=(),
                description="Clean up orphaned indexes"
            )
            
            self.transaction_mgr.add_operation(tx_id, **operation.__dict__)
            self.transaction_mgr.execute_operation(tx_id)
            
            # Validation post-suppression
            if self.enable_validation and self.auditor:
                # Audit de la base de données
                validation_report = self.auditor.validate_database(ValidationLevel.BASIC)
                
                # Identification des problèmes critiques
                critical_issues = validation_report.get_critical_issues_count()
                if critical_issues > 0:
                    # Logging
                    self.logger.error("Critical issues found after column deletion, rolling back")
                    # Annulation de la transaction
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return {col: False for col in columns}
            
            # Commit de la transaction
            if self.transaction_mgr.commit_transaction(tx_id):
                # Calcul du nombre de suppressions
                successful_deletions = sum(results.values())
                # Logging
                self.logger.info(f"Column deletion completed: {successful_deletions}/{len(valid_columns)} columns deleted")
                
                # Ajout des colonnes non trouvées au résultat
                for col in columns:
                    if col not in results:
                        results[col] = False
                
                return results
            else:
                # Logging
                self.logger.error("Failed to commit column deletion transaction")
                return {col: False for col in columns}
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error during transactional column deletion: {e}")
            # Annulation de la transaction
            self.transaction_mgr.rollback_transaction(tx_id)
            return {col: False for col in columns}
    
    # Suppression des colonnes sans transaction
    def _delete_columns_direct(self, columns: List[str]) -> Dict[str, bool]:
        """Suppression directe de colonnes sans transaction."""
        results = {}
        
        try:
            # Filtrage des colonnes existantes
            existing_columns = self._get_fact_table_columns()
            valid_columns = [col for col in columns if col in existing_columns]
            
            # Parcours des colonnes
            for column in valid_columns:
                try:
                    # Suppression des index
                    self._drop_column_indexes_safe(column)
                    
                    # Suppression de la dimension si applicable
                    if self._is_dimension_column(column):
                        self.dimension_mgr.delete_dimension_table(column)
                    
                    # Suppression de la colonne
                    dropped_columns = self.data_mgr.drop_columns([column])
                    
                    # Suppression des métadonnées
                    self.delete_column_metadata(column)
                    
                    results[column] = len(dropped_columns) > 0
                    
                except Exception as e:
                    # Logging
                    self.logger.error(f"Error deleting column {column}: {e}")
                    results[column] = False
            
            # Nettoyage des index orphelins
            try:
                self._cleanup_orphaned_indexes()
            except Exception as e:
                # Logging
                self.logger.warning(f"Error cleaning orphaned indexes: {e}")
            
            # Ajout des colonnes non trouvées au résultat
            for col in columns:
                if col not in results:
                    results[col] = False
            
            # Calcul du nombre de colonnes supprimées
            successful_deletions = sum(results.values())
            # Logging
            self.logger.info(f"Column deletion completed (direct mode): {successful_deletions}/{len(columns)} columns deleted")
            
            return results
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error during direct column deletion: {e}")
            return {col: False for col in columns}
    
    # Méthodes de suppression sécurisées
    # Méthode de suppression des lignes
    def _execute_row_deletion(self, filters: Optional[Union[str, List, Dict]]) -> int:
        """Exécuter la suppression de lignes."""
        try:
            # Comptage initial
            initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            if initial_count == 0:
                return 0
            
            # Utilisation du data manager pour la suppression
            rows_deleted = self.data_mgr.delete_rows(filters)
            
            return rows_deleted
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error executing row deletion: {e}")
            raise
    
    # Méthode auxiliaire de suppression des indexes liés à une colonne
    def _drop_column_indexes_safe(self, column: str) -> List[str]:
        """Supprimer les index liés à une colonne de manière sécurisée."""
        try:
            # Initialisation de la liste des indexes supprimés
            dropped_indexes = []
            
            # Recherche des index utilisant cette colonne
            index_query = """
                SELECT index_name, expressions 
                FROM duckdb_indexes() 
                WHERE expressions LIKE ?
            """
            indexes = self.conn.execute(index_query, [f'%{column}%']).fetchall()
            
            # Parcours des indexes utilisant la colonne et qu'il faut supprimer
            for index_name, expressions in indexes:
                try:
                    # Requête de suppression
                    drop_query = f"DROP INDEX IF EXISTS {index_name}"
                    self.conn.execute(drop_query)
                    # Ajout à la liste des indexes supprimés
                    dropped_indexes.append(index_name)
                    # Logging
                    self.logger.info(f"Dropped index {index_name} for column {column}")
                except Exception as e:
                    # Logging
                    self.logger.warning(f"Failed to drop index {index_name}: {e}")
            
            return dropped_indexes
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error dropping indexes for column {column}: {e}")
            return []
    
    # Méthode auxiliaire de suppression d'une colonne de la table des faits
    def _drop_fact_table_column(self, column: str) -> bool:
        """Supprimer une colonne de la fact table."""
        try:
            # Suppression des colonnes
            dropped_columns = self.data_mgr.drop_columns([column])
            return len(dropped_columns) > 0
        except Exception as e:
            # Logging
            self.logger.error(f"Error dropping fact table column {column}: {e}")
            return False
    
    
    def _cleanup_orphaned_data_comprehensive(self) -> Dict[str, Any]:
        """Nettoyage complet des données orphelines."""
        try:
            results = {
                'orphaned_dimensions': {},
                'null_columns': [],
                'orphaned_indexes': []
            }
            
            # Nettoyage des dimensions orphelines
            results['orphaned_dimensions'] = self.dimension_mgr.cleanup_orphaned_dimension_entries()
            
            # Suppression des colonnes ne contenant que des nulles
            null_only_columns = self._get_null_only_columns()
            if null_only_columns:
                dropped_columns = self.data_mgr.drop_columns(null_only_columns)
                results['null_columns'] = dropped_columns
                
                # Suppression des métadonnées pour ces colonnes
                for col in dropped_columns:
                    try:
                        self.delete_column_metadata(col)
                    except Exception as e:
                        self.logger.warning(f"Failed to delete metadata for {col}: {e}")
            
            # Nettoyage des index orphelins
            results['orphaned_indexes'] = self._cleanup_orphaned_indexes()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during comprehensive cleanup: {e}")
            return {}
    
    def _cleanup_orphaned_indexes(self) -> List[str]:
        """Nettoyer les index orphelins."""
        try:
            cleaned_indexes = []
            existing_columns = set(self._get_fact_table_columns())
            
            # Récupération de tous les index
            all_indexes = self.conn.execute("""
                SELECT index_name, expressions 
                FROM duckdb_indexes()
            """).fetchall()
            
            for index_name, expressions in all_indexes:
                # Analyse des colonnes référencées
                referenced_columns = []
                for col in existing_columns:
                    if col in expressions or f"fact_table.{col}" in expressions:
                        referenced_columns.append(col)
                
                # Si l'index ne référence aucune colonne existante mais référence fact_table
                if not referenced_columns and "fact_table" in expressions:
                    try:
                        self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                        cleaned_indexes.append(index_name)
                        self.logger.info(f"Cleaned orphaned index: {index_name}")
                    except Exception as e:
                        self.logger.error(f"Failed to drop orphaned index {index_name}: {e}")
            
            return cleaned_indexes
            
        except Exception as e:
            self.logger.error(f"Error cleaning orphaned indexes: {e}")
            return []
    
    # Méthodes de rollback
    def _restore_deleted_rows(self, filters: Optional[Union[str, List, Dict]], initial_count: int) -> bool:
        """Restaurer les lignes supprimées (placeholder - géré par transaction DuckDB)."""
        self.logger.info("Deleted rows restoration handled by database transaction")
        return True
    
    def _restore_column_indexes(self, column: str) -> bool:
        """Restaurer les index de colonne (placeholder - géré par transaction DuckDB)."""
        self.logger.info(f"Column indexes restoration for {column} handled by database transaction")
        return True
    
    def _restore_dimension_table(self, column: str) -> bool:
        """Restaurer une table de dimension (placeholder - géré par transaction DuckDB)."""
        self.logger.info(f"Dimension table restoration for {column} handled by database transaction")
        return True
    
    def _restore_fact_table_column(self, column: str) -> bool:
        """Restaurer une colonne de fact table (placeholder - géré par transaction DuckDB)."""
        self.logger.info(f"Fact table column restoration for {column} handled by database transaction")
        return True
    
    def _restore_column_metadata(self, column: str) -> bool:
        """Restaurer les métadonnées de colonne (placeholder - géré par transaction DuckDB)."""
        self.logger.info(f"Column metadata restoration for {column} handled by database transaction")
        return True
    
    def _restore_orphaned_data(self) -> bool:
        """Restaurer les données orphelines (placeholder - géré par transaction DuckDB)."""
        self.logger.info("Orphaned data restoration handled by database transaction")
        return True
    
    # Méthodes d'analyse des dépendances
    def _analyze_column_dependencies(self, columns: List[str]) -> Dict[str, Any]:
        """Analyser les dépendances des colonnes."""
        try:
            dependency_report = {
                'columns_analyzed': columns,
                'has_critical_dependencies': False,
                'dependencies': {},
                'warnings': []
            }
            
            for column in columns:
                column_deps = {
                    'is_categorical': self._is_dimension_column(column),
                    'has_dimension_table': False,
                    'has_indexes': False,
                    'referenced_in_queries': False  # Placeholder - nécessiterait analyse des logs
                }
                
                # Vérification de l'existence de table de dimension
                if column_deps['is_categorical']:
                    dim_table_name = f"dim_{column}"
                    column_deps['has_dimension_table'] = self._table_exists(dim_table_name)
                
                # Vérification des index
                try:
                    index_query = """
                        SELECT COUNT(*) FROM duckdb_indexes() 
                        WHERE expressions LIKE ?
                    """
                    index_count = self.conn.execute(index_query, [f'%{column}%']).fetchone()[0]
                    column_deps['has_indexes'] = index_count > 0
                except:
                    column_deps['has_indexes'] = False
                
                # Avertissements
                if column_deps['is_categorical'] and column_deps['has_dimension_table']:
                    dependency_report['warnings'].append(
                        f"Column {column} has associated dimension table that will be deleted"
                    )
                
                if column_deps['has_indexes']:
                    dependency_report['warnings'].append(
                        f"Column {column} has associated indexes that will be dropped"
                    )
                
                dependency_report['dependencies'][column] = column_deps
            
            return dependency_report
            
        except Exception as e:
            self.logger.error(f"Error analyzing column dependencies: {e}")
            return {
                'columns_analyzed': columns,
                'has_critical_dependencies': False,
                'dependencies': {},
                'warnings': [f"Dependency analysis failed: {str(e)}"]
            }
    
    # Méthodes publiques additionnelles
    def get_deletion_impact(self, 
                          columns: Optional[List[str]] = None,
                          filters: Optional[Union[str, List, Dict]] = None) -> Dict[str, Any]:
        """
        Analyze the impact of a potential deletion operation.
        
        Args:
            columns: Columns to analyze for deletion impact (None for no column analysis)
            filters: Row filters to analyze for deletion impact (None for no row analysis)
            
        Returns:
            Dictionary containing impact analysis
            
        Example:
            >>> impact = deleter.get_deletion_impact(columns=['old_col'], filters="status = 'inactive'")
            >>> print(f"Rows affected: {impact['rows_affected']}")
            >>> print(f"Dependencies: {impact['column_dependencies']}")
        """
        try:
            impact_report = {
                'timestamp': pd.Timestamp.now(),
                'rows_affected': 0,
                'columns_affected': [],
                'column_dependencies': {},
                'dimension_tables_affected': [],
                'indexes_affected': [],
                'warnings': [],
                'recommendations': []
            }
            
            # Analyse de l'impact sur les lignes
            if filters is not None:
                try:
                    where_clause = _build_where_clause(filters)
                    if where_clause:
                        count_query = f"SELECT COUNT(*) FROM fact_table {where_clause}"
                        impact_report['rows_affected'] = self.conn.execute(count_query).fetchone()[0]
                    
                    if impact_report['rows_affected'] > 0:
                        impact_report['warnings'].append(
                            f"{impact_report['rows_affected']} rows will be deleted"
                        )
                except Exception as e:
                    impact_report['warnings'].append(f"Could not analyze row impact: {e}")
            
            # Analyse de l'impact sur les colonnes
            if columns:
                impact_report['columns_affected'] = columns
                dependency_analysis = self._analyze_column_dependencies(columns)
                impact_report['column_dependencies'] = dependency_analysis['dependencies']
                impact_report['warnings'].extend(dependency_analysis['warnings'])
                
                # Collecte des tables de dimension affectées
                for col, deps in dependency_analysis['dependencies'].items():
                    if deps['has_dimension_table']:
                        impact_report['dimension_tables_affected'].append(f"dim_{col}")
                    
                    if deps['has_indexes']:
                        impact_report['indexes_affected'].append(col)
            
            # Génération des recommandations
            if impact_report['rows_affected'] > 1000:
                impact_report['recommendations'].append(
                    "Consider using batch processing for large row deletions"
                )
            
            if len(impact_report['dimension_tables_affected']) > 0:
                impact_report['recommendations'].append(
                    "Review dimension table dependencies before deletion"
                )
            
            if len(impact_report['indexes_affected']) > 0:
                impact_report['recommendations'].append(
                    "Consider recreating important indexes after column deletion"
                )
            
            return impact_report
            
        except Exception as e:
            self.logger.error(f"Error analyzing deletion impact: {e}")
            return {
                'error': str(e),
                'timestamp': pd.Timestamp.now()
            }
    
    def get_deletion_status(self) -> Dict[str, Any]:
        """
        Get the status of the database deletion system.
        
        Returns:
            Dictionary containing system status information
            
        Example:
            >>> status = deleter.get_deletion_status()
            >>> print(f"System health: {status['health_status']}")
        """
        try:
            status = {
                'timestamp': pd.Timestamp.now(),
                'health_status': 'unknown',
                'active_transactions': 0,
                'validation_enabled': self.enable_validation,
                'auto_cleanup': self.auto_cleanup
            }
            
            # Vérification des transactions actives
            active_txs = self.transaction_mgr.list_active_transactions()
            status['active_transactions'] = len(active_txs)
            
            # Vérification de la santé de la base de données
            if self.auditor:
                health_check = self.auditor.get_quick_health_check()
                status['health_status'] = health_check.get('status', 'unknown')
                status['database_info'] = health_check
            
            # Statistiques de la fact table
            if self._table_exists('fact_table'):
                table_stats = self.data_mgr.get_table_stats()
                status['fact_table_stats'] = table_stats
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting deletion status: {e}")
            return {'error': str(e), 'timestamp': pd.Timestamp.now()}
    
    def validate_database_state(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        """
        Validate the current state of the database.
        
        Args:
            validation_level: Level of validation to perform
            
        Returns:
            ValidationReport from the auditor
            
        Example:
            >>> report = deleter.validate_database_state(ValidationLevel.COMPREHENSIVE)
            >>> if report.get_critical_issues_count() > 0:
            ...     print("Critical issues detected!")
        """
        if not self.auditor:
            self.logger.warning("Validation disabled - no auditor available")
            return None
        
        return self.auditor.validate_database(validation_level)
    
    def cleanup_database(self, comprehensive: bool = True) -> Dict[str, Any]:
        """
        Perform comprehensive database cleanup operations.
        
        Args:
            comprehensive: Whether to perform comprehensive cleanup
            
        Returns:
            Dictionary with cleanup results
            
        Example:
            >>> results = deleter.cleanup_database(comprehensive=True)
            >>> print(f"Cleaned: {results}")
        """
        try:
            if comprehensive:
                return self._cleanup_orphaned_data_comprehensive()
            else:
                # Nettoyage basique
                results = {
                    'orphaned_dimensions': self.dimension_mgr.cleanup_orphaned_dimension_entries(),
                    'orphaned_indexes': self._cleanup_orphaned_indexes()
                }
                
                self.logger.info("Basic database cleanup completed")
                return results
            
        except Exception as e:
            self.logger.error(f"Error during database cleanup: {e}")
            return {'error': str(e)}