# Importation des modules
# Modules de base
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any, Tuple
# DuckDB
import duckdb

# Import des gestionnaires
from .managers.base_manager import BaseSchemaManager
from .managers.dimension_manager import DimensionManager
from .managers.data_manager import DataManager
from .managers.transaction_manager import (
    TransactionManager, TransactionOperation, TransactionState
)
from .auditor import DatabaseAuditor, ValidationLevel, IssueSeverity

# Import des utilitaires
from ..utils.data_processing import remove_dataframe_duplicates
from ..builders.indexer import IndexManager

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe de mise à jour d'une base de données
class DatabaseUpdaterV2(BaseSchemaManager):
    """
    Refactored database updater using the new modular architecture.
    
    Provides atomic, transactional updates to DuckDB databases with proper
    validation, error recovery, and state consistency. Uses specialized managers
    for different aspects of database operations.
    
    Attributes:
        dimension_mgr (DimensionManager): Manages dimension table operations
        data_mgr (DataManager): Manages fact table operations
        transaction_mgr (TransactionManager): Manages transactions and rollback
        auditor (DatabaseAuditor): Validates database state and operations
        max_workers (int): Maximum number of parallel workers
        batch_size (int): Size of batches for processing large datasets
    """
    # Initialisation
    def __init__(self, 
                 connection: Optional[duckdb.DuckDBPyConnection] = None, 
                 path: Optional[os.PathLike]=None,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None,
                 max_workers: int = 4,
                 batch_size: int = 10000,
                 enable_validation: bool = True):
        """
        Initialize the refactored database updater.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Threshold for determining categorical variables
            log_filename: Path to log file
            max_workers: Maximum number of parallel workers
            batch_size: Size of batches for processing
            enable_validation: Whether to enable pre/post operation validation
            
        Example:
            >>> conn = duckdb.connect('database.db')
            >>> updater = DatabaseUpdaterV2(conn, max_workers=8, enable_validation=True)
        """
        # Initialisation du parent
        super().__init__(connection=connection, path=path, categorical_threshold=categorical_threshold, log_filename=log_filename)
        
        # Initialisation des gestionnaires spécialisés
        self.dimension_mgr = DimensionManager(
            connection=connection, path=path, categorical_threshold=categorical_threshold, log_filename=log_filename, max_workers=max_workers
        )
        
        self.data_mgr = DataManager(
            connection=connection, path=path, categorical_threshold=categorical_threshold, log_filename=log_filename, batch_size=batch_size
        )
        
        self.transaction_mgr = TransactionManager(
            connection=connection, path=path, categorical_threshold=categorical_threshold, log_filename=log_filename
        )
        
        self.auditor = DatabaseAuditor(
            connection=connection, path=path, categorical_threshold=categorical_threshold, log_filename=log_filename
        ) if enable_validation else None
        
        # Configuration
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.enable_validation = enable_validation
    
    # Méthode de validation d'une opération
    def validate_operation(self, operation_type: str, **kwargs) -> bool:
        """
        Validate update operations before execution.
        
        Args:
            operation_type: Type of operation to validate
            **kwargs: Operation-specific parameters
            
        Returns:
            True if operation is valid
        """
        # Absence de validation si un auditeur n'est pas spécifié ou si elle n'est pas permise
        if not self.enable_validation or not self.auditor:
            return True
        
        # Validation par l'auditeur
        validation_report = self.auditor.validate_operation_preconditions(operation_type, **kwargs)
        
        # Vérification des problèmes critiques
        if validation_report.get_critical_issues_count() > 0:
            # Logging
            self.logger.error(f"Critical validation issues found for {operation_type} operation:")
            # Logging des erreurs critiques
            for issue in validation_report.get_issues_by_severity(IssueSeverity.CRITICAL):
                self.logger.error(f"  - {issue.description}")
            return False
        
        # Avertissements pour les problèmes de priorité haute
        high_issues = validation_report.get_issues_by_severity(IssueSeverity.HIGH)
        if high_issues:
            # Logging
            self.logger.warning(f"High priority validation issues found for {operation_type} operation:")
            # Logging des problèmes de priorité haute
            for issue in high_issues:
                self.logger.warning(f"  - {issue.description}")
        
        return True
    
    # Méthode principale de mise à jour
    def update_database(self,
                       update_df: pd.DataFrame,
                       check_duplicates_db: bool = True,
                       check_duplicates_update: bool = True,
                       keep: Literal[False, 'first', 'last'] = False,
                       index_config: Optional[Dict] = None,
                       use_batch_processing: bool = True,
                       use_transaction: bool = True) -> bool:
        """
        Update the entire database with new data using atomic operations.
        
        Args:
            update_df: DataFrame containing update data
            check_duplicates_db: Whether to check duplicates in existing database
            check_duplicates_update: Whether to check duplicates in update DataFrame
            keep: Which duplicates to keep when removing duplicates
            index_config: Dictionary containing index configuration
            use_batch_processing: Whether to use batch processing for large datasets
            use_transaction: Whether to use database transactions
            
        Returns:
            True if update was successful, False otherwise
            
        Example:
            >>> success = updater.update_database(
            ...     new_data_df,
            ...     check_duplicates_db=True,
            ...     use_transaction=True
            ... )
            >>> if success:
            ...     print("Database updated successfully")
        """
        # Validation préalable
        if not self.validate_operation('update', df=update_df):
            # Logging
            self.logger.error("Pre-update validation failed")
            return False
        
        # Logging
        self.logger.info(f"Starting database update with {len(update_df)} rows (transaction: {use_transaction})")
        
        # Utilisation de la transaction pour la mise à jour de la base de données
        if use_transaction:
            return self._update_database_transactional(
                update_df, check_duplicates_db, check_duplicates_update, 
                keep, index_config, use_batch_processing
            )
        # Sinon mise à jour directe
        else:
            return self._update_database_direct(
                update_df, check_duplicates_db, check_duplicates_update,
                keep, index_config, use_batch_processing
            )
    
    # Méthode de mise à jour de la base de données de manière transactionnelle
    def _update_database_transactional(self,
                                     update_df: pd.DataFrame,
                                     check_duplicates_db: bool,
                                     check_duplicates_update: bool,
                                     keep: Literal[False, 'first', 'last'],
                                     index_config: Optional[Dict],
                                     use_batch_processing: bool) -> bool:
        """Perform transactional database update with validation and rollback.

        Executes the update within a transaction, allowing rollback on failure.
        Steps: preprocess → duplicate removal → metadata → fact table → dimensions → indexes.

        Args:
            update_df: DataFrame containing the update data.
            check_duplicates_db: Whether to check and remove duplicates in database.
            check_duplicates_update: Whether to check and remove duplicates in update data.
            keep: Duplicate handling strategy ('first', 'last', or False).
            index_config: Optional index configuration dictionary.
            use_batch_processing: Whether to use batch processing for large datasets.

        Returns:
            True if update succeeded and committed, False otherwise.
        """
        
        # Début de la transaction
        tx_id = self.transaction_mgr.begin_transaction("Database update with validation and rollback")
        
        try:
            # Étape 1: Suppression des doublons dans les données de mise à jour
            if check_duplicates_update:
                # Initialisation de l'opération de transaction
                operation = TransactionOperation(
                    operation_type='preprocess',
                    operation_func=self._remove_update_duplicates,
                    operation_args=(update_df, keep),
                    description="Remove duplicates from update data"
                )
                
                # Annulation de la transaction si l'opération ne peut être ajoutée
                if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False
                
                # Annulation de la transaction si l'opération ne peut être exécutée
                if not self.transaction_mgr.execute_operation(tx_id):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False
                
                # Récupération des données nettoyées
                update_df = self._get_cleaned_update_data(update_df, keep)
            
            # Étape 2: Suppression des doublons dans la base de données
            if check_duplicates_db:
                # Initialisation de l'opération de transacti
                operation = TransactionOperation(
                    operation_type='cleanup',
                    operation_func=self._remove_database_duplicates,
                    operation_args=(keep,),
                    rollback_func=self._restore_database_state,
                    description="Remove duplicates from existing database"
                )
                
                # Annulation de la transaction si l'opération ne peut être ajoutée
                if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False
                
                # Annulation de la transaction si l'opération ne peut être exécutée
                if not self.transaction_mgr.execute_operation(tx_id):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False
            
            # Création d'un savepoint avant les modifications majeures
            self.transaction_mgr.create_savepoint(tx_id, "before_major_updates")
            
            # Étape 3: Mise à jour des métadonnées
            # Initialisation de l'opération de transaction
            operation = TransactionOperation(
                operation_type='metadata_update',
                operation_func=self._update_metadata_safe,
                operation_args=(update_df,),
                rollback_func=self._rollback_metadata_changes,
                description="Update metadata table"
            )
            
            # Annulation de la transaction si l'opération ne peut être ajoutée
            if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False
            
            # Annulation de la transaction si l'opération ne peut être exécutée
            if not self.transaction_mgr.execute_operation(tx_id):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False
            
            # Étape 4: Mise à jour de la table de faits (avant les dimensions pour refléter l'état actuel)
            # Mise à jour en batch si spécifié
            if use_batch_processing and len(update_df) > self.batch_size:
                # Initialisation de l'opération de transaction
                operation = TransactionOperation(
                    operation_type='fact_update_batch',
                    operation_func=self._update_fact_table_batch,
                    operation_args=(update_df,),
                    rollback_func=self._rollback_fact_changes,
                    description="Update fact table (batch processing)"
                )
            else:
                # Initialisation de l'opération de transaction
                operation = TransactionOperation(
                    operation_type='fact_update_direct',
                    operation_func=self._update_fact_table_direct,
                    operation_args=(update_df,),
                    rollback_func=self._rollback_fact_changes,
                    description="Update fact table (direct)"
                )

            # Annulation de la transaction si l'opération ne peut être ajoutée
            if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Annulation de la transaction si l'opération ne peut être exécutée
            if not self.transaction_mgr.execute_operation(tx_id):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Étape 5: Mise à jour des tables de dimensions (après fact table pour refléter les données actuelles)
            # Initialisation de l'opération de transaction
            operation = TransactionOperation(
                operation_type='dimension_update',
                operation_func=self._update_dimensions_safe,
                operation_args=(update_df,),
                rollback_func=self._rollback_dimension_changes,
                description="Update dimension tables"
            )

            # Annulation de la transaction si l'opération ne peut être ajoutée
            if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Annulation de la transaction si l'opération ne peut être exécutée
            if not self.transaction_mgr.execute_operation(tx_id):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Étape 5b: Nettoyage des entrées orphelines dans les tables de dimension
            cleanup_operation = TransactionOperation(
                operation_type='dimension_cleanup',
                operation_func=self.dimension_mgr.cleanup_orphaned_dimension_entries,
                operation_args=(),
                description="Clean orphaned dimension entries"
            )

            if not self.transaction_mgr.add_operation(tx_id, **cleanup_operation.__dict__):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            if not self.transaction_mgr.execute_operation(tx_id):
                self.transaction_mgr.rollback_transaction(tx_id)
                return False

            # Étape 6: Mise à jour des index
            # Si un index est spécifié
            if index_config:
                # Initialisation de l'opération de transaction
                operation = TransactionOperation(
                    operation_type='index_update',
                    operation_func=self._update_indexes_safe,
                    operation_args=(index_config,),
                    rollback_func=self._rollback_index_changes,
                    description="Update database indexes"
                )
                
                # Annulation de la transaction si l'opération ne peut être ajoutée
                if not self.transaction_mgr.add_operation(tx_id, **operation.__dict__):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False
                
                # Annulation de la transaction si l'opération ne peut être exécutée
                if not self.transaction_mgr.execute_operation(tx_id):
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False
            
            # Validation post-update
            if self.enable_validation and self.auditor:
                # Validation de la base de données
                validation_report = self.auditor.validate_database(ValidationLevel.STANDARD)
                
                # Vérification des problèmes critiques
                if validation_report.get_critical_issues_count() > 0:
                    # Logging
                    self.logger.error("Critical issues found after update, rolling back")
                    # Annulation de la transaction
                    self.transaction_mgr.rollback_transaction(tx_id)
                    return False
            
            # Commit de la transaction
            if self.transaction_mgr.commit_transaction(tx_id):
                # Logging
                self.logger.info("Database update completed successfully")
                # Invaliation du cache
                self._invalidate_metadata_cache()
                return True
            else:
                # Logging
                self.logger.error("Failed to commit transaction")
                return False
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error during transactional update: {e}")
            # Annulation de la transaction
            self.transaction_mgr.rollback_transaction(tx_id)
            return False
    
    # Méthode auxiliaire de mise à jour directe de la base de données
    def _update_database_direct(self,
                              update_df: pd.DataFrame,
                              check_duplicates_db: bool,
                              check_duplicates_update: bool,
                              keep: Literal[False, 'first', 'last'],
                              index_config: Optional[Dict],
                              use_batch_processing: bool) -> bool:
        """Perform direct database update without transaction wrapping.

        Suitable for simple updates where rollback capability is not needed.
        Faster but no automatic rollback on partial failure.

        Args:
            update_df: DataFrame containing the update data.
            check_duplicates_db: Whether to check and remove duplicates in database.
            check_duplicates_update: Whether to check and remove duplicates in update data.
            keep: Duplicate handling strategy ('first', 'last', or False).
            index_config: Optional index configuration dictionary.
            use_batch_processing: Whether to use batch processing for large datasets.

        Returns:
            True if update completed successfully, False otherwise.
        """
        try:
            # Suppression des doublons dans les données de mise à jour
            if check_duplicates_update:
                update_df = remove_dataframe_duplicates(update_df, keep, self.logger, 'update')
            
            # Suppression des doublons dans la base de données
            if check_duplicates_db:
                self._remove_database_duplicates(keep)
            
            # Mise à jour des métadonnées
            if not self._update_metadata_safe(update_df):
                return False

            # Mise à jour de la table de faits (avant les dimensions pour refléter l'état actuel)
            # Mise à jour par batch si spécifié
            if use_batch_processing and len(update_df) > self.batch_size:
                if not self._update_fact_table_batch(update_df):
                    return False
            else:
                if not self._update_fact_table_direct(update_df):
                    return False

            # Mise à jour des tables de dimensions (après fact table pour refléter les données actuelles)
            if not self._update_dimensions_safe(update_df):
                return False

            # Nettoyage des entrées orphelines dans les tables de dimension
            self.dimension_mgr.cleanup_orphaned_dimension_entries()

            # Mise à jour des index
            if index_config:
                if not self._update_indexes_safe(index_config):
                    return False

            # Nettoyage final
            self._cleanup_orphaned_data()
            
            # Logging
            self.logger.info("Database update completed successfully (direct mode)")
            # Invalidation du cache des méta-données
            self._invalidate_metadata_cache()
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error during direct update: {e}")
            return False
    
    # Méthodes de mise à jour sécurisées
    # Méthode auxiliaire de mise à jour des méta-données
    def _update_metadata_safe(self, update_df: pd.DataFrame) -> bool:
        """Safely update metadata table with type conflict resolution.

        Args:
            update_df: DataFrame whose columns may require metadata updates.

        Returns:
            True if metadata updated successfully, False on error.
        """
        try:
            # Chargement des métadonnées actuelles
            current_metadata = self._load_current_metadata()
            current_columns = set(current_metadata['name'].values) if len(current_metadata) > 0 else set()
            new_columns = set(update_df.columns)
            
            # Vérification des conflits de types pour les colonnes existantes
            for col in current_columns.intersection(new_columns):
                self._resolve_type_conflicts(col, update_df, current_metadata)
            
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error updating metadata: {e}")
            return False
    
    # Méthode auxiliaire de mise à jour des tables de dimension
    def _update_dimensions_safe(self, update_df: pd.DataFrame) -> bool:
        """Safely update dimension tables with categorical threshold checks.

        Handles conversion between categorical and non-categorical status
        based on unique value counts relative to the threshold.

        Args:
            update_df: DataFrame containing potential dimension updates.

        Returns:
            True if dimensions updated successfully, False on error.
        """
        try:
            # Chargement des métadonnées
            current_metadata = self._load_current_metadata()

            # Classification des colonnes par statut catégoriel
            categorical_columns = {}
            non_categorical_columns = {}
            
            # Parcours des colonnes
            for _, row in current_metadata.iterrows():
                " Extraction du nom de la colonne"
                col_name = row['name']
                if col_name in update_df.columns:
                    # Vérification du statut catégoriel
                    if row['is_categorical']:
                        categorical_columns[col_name] = update_df[col_name]
                    elif row['python_type'] == 'object':
                        non_categorical_columns[col_name] = update_df[col_name]
            
            # Mise à jour des dimensions existantes
            if categorical_columns:
                results = self.dimension_mgr.batch_update_dimensions(
                    categorical_columns, 
                    use_parallel=(len(categorical_columns) > 1 and self.max_workers > 1)
                )
                
                # Vérification des résultats
                for col_name, added_count in results.items():
                    if added_count < 0:  # Erreur
                        # Logging
                        self.logger.error(f"Failed to update dimension for {col_name}")
                        return False
            
            # Vérification des conversions vers catégoriel
            for col_name, values in non_categorical_columns.items():
                # Vérification du seuil
                if self._check_categorical_threshold(values):
                    # Conversion en catégoriel
                    if not self.dimension_mgr.convert_to_categorical(col_name, values):
                        # Logging
                        self.logger.warning(f"Failed to convert {col_name} to categorical")
            
            # Vérification des conversions vers non-catégoriel
            for col_name, values in categorical_columns.items():
                # Vérification du seuil
                if not self._check_categorical_threshold(values):
                    # Conversion en non-catégroeil
                    if not self.dimension_mgr.convert_to_non_categorical(col_name):
                        # Logging
                        self.logger.warning(f"Failed to convert {col_name} to non-categorical")
            
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error updating dimensions: {e}")
            return False
    
    # Méthode auxiliaire de mise à jour directe de la table des faits
    def _update_fact_table_direct(self, update_df: pd.DataFrame) -> bool:
        """Update fact table directly without batch processing.

        Args:
            update_df: DataFrame containing the data to upsert.

        Returns:
            True if fact table updated successfully, False on error.
        """
        try:
            # Préparation des données pour la fact table
            prepared_df = self._prepare_dataframe_for_fact_table(update_df)
            
            # Détermination des clés de fusion
            existing_columns = self.data_mgr._get_fact_table_columns()
            merge_keys = list(set(prepared_df.columns).intersection(set(existing_columns)))
            
            # Upsert des données
            if merge_keys:
                inserted, updated = self.data_mgr.upsert_data(prepared_df, merge_keys, use_batch=False)
                self.logger.info(f"Fact table update: {inserted} inserted, {updated} updated")
            else:
                inserted = self.data_mgr.insert_data(prepared_df, use_batch=False)
                self.logger.info(f"Fact table insert: {inserted} rows")
            
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error updating fact table: {e}")
            return False
    
    # Méthode auxiliaire de la mise à jour par batch de la table des faits
    def _update_fact_table_batch(self, update_df: pd.DataFrame) -> bool:
        """Update fact table using batch processing for large datasets.

        Args:
            update_df: DataFrame containing the data to upsert in batches.

        Returns:
            True if fact table updated successfully, False on error.
        """
        try:
            # Préparation des données pour la fact table
            prepared_df = self._prepare_dataframe_for_fact_table(update_df)
            
            # Détermination des clés de fusion
            existing_columns = self.data_mgr._get_fact_table_columns()
            merge_keys = list(set(prepared_df.columns).intersection(set(existing_columns)))
            
            # Upsert des données par batch
            if merge_keys:
                inserted, updated = self.data_mgr.upsert_data(prepared_df, merge_keys, use_batch=True)
                # Logging
                self.logger.info(f"Fact table batch update: {inserted} inserted, {updated} updated")
            else:
                inserted = self.data_mgr.insert_data(prepared_df, use_batch=True)
                # Logging
                self.logger.info(f"Fact table batch insert: {inserted} rows")
            
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error updating fact table in batch: {e}")
            return False
    
    # Méthode auxiliaire de mise à jour des index
    def _update_indexes_safe(self, index_config: Dict) -> bool:
        """Safely update database indexes based on configuration.

        Args:
            index_config: Dictionary containing index configuration.
                - drop_existing: Whether to drop existing indexes first.

        Returns:
            True if indexes updated successfully, False on error.
        """
        try:
            # Création d'un gestionnaire d'index
            index_manager = IndexManager(connection=self.conn)
            
            # Suppression des index existants (optionnel selon configuration)
            if index_config.get('drop_existing', True):
                try:
                    # Recherche des index existants
                    existing_indexes = self.conn.execute("""
                        SELECT name FROM sqlite_master 
                        WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
                    """).fetchall()
                    # Parcours des index
                    for idx in existing_indexes:
                        try:
                            # Suppression des index
                            index_manager.drop_index(idx[0])
                        except Exception as e:
                            # Logging
                            self.logger.warning(f"Cannot drop index {idx[0]}: {e}")
                except Exception as e:
                    # Logging
                    self.logger.warning(f"Error retrieving existing indexes: {e}")
            
            # Création des nouveaux index
            index_manager.create_fact_table_indexes("fact_table", index_config)
            # Logging
            self.logger.info("Successfully updated indexes")
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error updating indexes: {e}")
            return False
    
    # Méthodes de rollback
    # Méthode auxiliaire de rollback des changements de métadonnées
    def _rollback_metadata_changes(self) -> bool:
        """Rollback metadata changes by invalidating cache.

        Returns:
            True if rollback succeeded, False on error.
        """
        try:
            # Invalidation du cache pour forcer le rechargement
            self._invalidate_metadata_cache()
            # Logging
            self.logger.info("Metadata changes rolled back")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error rolling back metadata changes: {e}")
            return False
    
    # Méthode auxiliaire de rollback des changements de dimensions
    def _rollback_dimension_changes(self) -> bool:
        """Rollback dimension table changes (handled by DuckDB transaction).

        Returns:
            True if rollback succeeded, False on error.
        """
        try:
            # Les changements de dimension sont gérés par la transaction DuckDB
            self.logger.info("Dimension changes rolled back")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error rolling back dimension changes: {e}")
            return False
    
    # Méthode auxiliaire de rollback des changements de la fact table.
    def _rollback_fact_changes(self) -> bool:
        """Rollback fact table changes (handled by DuckDB transaction).

        Returns:
            True if rollback succeeded, False on error.
        """
        try:
            # Les changements de fact table sont gérés par la transaction DuckDB
            # Logging
            self.logger.info("Fact table changes rolled back")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error rolling back fact table changes: {e}")
            return False
    
    # Méthode auxiliaire de rollback des changements d'index.
    def _rollback_index_changes(self) -> bool:
        """Rollback index changes (handled by DuckDB transaction).

        Returns:
            True if rollback succeeded, False on error.
        """
        try:
            # Les changements d'index sont gérés par la transaction DuckDB
            # Logging
            self.logger.info("Index changes rolled back")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error rolling back index changes: {e}")
            return False
    
    # Méthodes utilitaires
    # Méthode auxiliaire de préparation du jeu de données pour la table des faits
    def _prepare_dataframe_for_fact_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for fact table insertion.

        Converts categorical columns to their dimension table values
        and handles unmapped values.

        Args:
            df: Original DataFrame to prepare.

        Returns:
            Prepared DataFrame with categorical columns mapped to dimension values.
        """
        try:
            # Copie du DataFrame
            prepared_df = df.copy()
            
            # Chargement des métadonnées
            current_metadata = self._load_current_metadata()
            
            # Conversion des colonnes catégorielles
            for _, row in current_metadata.iterrows():
                col_name = row['name']
                is_categorical = row['is_categorical']
                
                if is_categorical and col_name in prepared_df.columns:
                    # Mise à jour préalable de la dimension
                    self.dimension_mgr.update_dimension_values(col_name, prepared_df[col_name])
                    
                    # Récupération du mapping
                    mapping_df = self.dimension_mgr.get_dimension_mapping(col_name)
                    
                    if mapping_df is not None and len(mapping_df) > 0:
                        label_to_value = dict(zip(mapping_df['label'], mapping_df['value']))
                        prepared_df[col_name] = prepared_df[col_name].map(label_to_value)
                        
                        # Gestion des valeurs non mappées
                        unmapped_mask = prepared_df[col_name].isna()
                        if unmapped_mask.any():
                            self.logger.warning(f"Found {unmapped_mask.sum()} unmapped values in {col_name}")
                            prepared_df[col_name] = prepared_df[col_name].fillna(-1)
            
            return prepared_df
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error preparing DataFrame for fact table: {e}")
            return df
    
    # Méthode auxiliaire de suppression des doublons des données de mise à jour
    def _remove_update_duplicates(self, update_df: pd.DataFrame, keep: Literal[False, 'first', 'last']) -> bool:
        """Remove duplicates from update DataFrame.

        Args:
            update_df: DataFrame to deduplicate.
            keep: Strategy for keeping duplicates ('first', 'last', or False).

        Returns:
            True if deduplication succeeded, False on error.
        """
        try:
            # Cette méthode modifie le DataFrame en place via la référence
            cleaned_df = remove_dataframe_duplicates(update_df, keep, self.logger, 'update')
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error removing update duplicates: {e}")
            return False
    
    # Méthode auxiliaire de nettoyage des données mises à jour
    def _get_cleaned_update_data(self, update_df: pd.DataFrame, keep: Literal[False, 'first', 'last']) -> pd.DataFrame:
        """Get deduplicated update data.

        Args:
            update_df: Original DataFrame with potential duplicates.
            keep: Strategy for keeping duplicates ('first', 'last', or False).

        Returns:
            DataFrame with duplicates removed according to keep strategy.
        """
        return remove_dataframe_duplicates(update_df, keep, self.logger, 'update')
    
    # Méthode auxiliaire de suppression des doublons dans la base de données
    def _remove_database_duplicates(self, keep: Literal[False, 'first', 'last']) -> bool:
        """Remove duplicate rows from the database fact table.

        Args:
            keep: Strategy for keeping duplicates ('first', 'last', or False).

        Returns:
            True if deduplication succeeded, False on error.
        """
        try:
            from ..utils.data_processing import build_database_duplicate_removal_query
            
            # Récupération du nombre initial de lignes
            initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            # Récupération des colonnes
            all_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
            columns_to_check = [col for col in all_columns if col != 'value']
            
            if not columns_to_check:
                return True
            
            # Construction et exécution de la requête de suppression des doublons
            delete_query = build_database_duplicate_removal_query(columns_to_check, keep, 'fact_table')
            self.conn.execute(delete_query)
            
            # Calcul des lignes supprimées
            final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            removed_count = initial_count - final_count
            
            if removed_count > 0:
                # Logging
                self.logger.info(f"Database duplicate removal: {removed_count} rows removed")
            
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error removing database duplicates: {e}")
            return False
    
    # Méthode auxuliaire de restoration de la base de données
    def _restore_database_state(self) -> bool:
        """Restore database state (placeholder - handled by DuckDB transaction).

        Returns:
            True (actual restoration handled by database transaction rollback).
        """
        # Cette méthode est un placeholder pour la restauration d'état
        # En pratique, cela serait géré par la transaction DuckDB
        self.logger.info("Database state restoration handled by transaction")
        return True
    
    # Méthode auxiliaire de nettoyage des données orphelines
    def _cleanup_orphaned_data(self) -> None:
        """Clean up orphaned data after update operations.

        Removes orphaned dimension entries and drops null-only columns.
        """
        try:
            # Nettoyage des entrées orphelines dans les dimensions
            removed_counts = self.dimension_mgr.cleanup_orphaned_dimension_entries()
            
            if removed_counts:
                self.logger.info(f"Cleaned orphaned dimension entries: {removed_counts}")
            
            # Suppression des colonnes ne contenant que des nulles
            null_only_columns = self._get_null_only_columns()
            if null_only_columns:
                dropped_columns = self.data_mgr.drop_columns(null_only_columns)
                if dropped_columns:
                    self.logger.info(f"Dropped null-only columns: {dropped_columns}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning orphaned data: {e}")
    
    # Méthodes publiques additionnelles
    # Méthode d'extraction du statut de la base de données
    def get_update_status(self) -> Dict[str, Any]:
        """
        Get the status of the database update system.
        
        Returns:
            Dictionary containing system status information
            
        Example:
            >>> status = updater.get_update_status()
            >>> print(f"System health: {status['health_status']}")
        """
        try:
            status = {
                'timestamp': pd.Timestamp.now(),
                'health_status': 'unknown',
                'active_transactions': 0,
                'validation_enabled': self.enable_validation,
                'batch_size': self.batch_size,
                'max_workers': self.max_workers
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
            # Logging
            self.logger.error(f"Error getting update status: {e}")
            return {'error': str(e), 'timestamp': pd.Timestamp.now()}
    
    # Méthode de validation de l'état de la base de données
    def validate_database_state(self, validation_level: ValidationLevel = ValidationLevel.STANDARD) -> Any:
        """
        Validate the current state of the database.
        
        Args:
            validation_level: Level of validation to perform
            
        Returns:
            ValidationReport from the auditor
            
        Example:
            >>> report = updater.validate_database_state(ValidationLevel.COMPREHENSIVE)
            >>> if report.get_critical_issues_count() > 0:
            ...     print("Critical issues detected!")
        """
        # Vérification qu'un auditeur est renseigné
        if not self.auditor:
            # Logging
            self.logger.warning("Validation disabled - no auditor available")
            return None
        
        return self.auditor.validate_database(validation_level)
    
    # Méthode d'optimisation de la base de données
    def optimize_database(self) -> bool:
        """
        Perform database optimization operations.
        
        Returns:
            True if optimization was successful
            
        Example:
            >>> success = updater.optimize_database()
            >>> if success:
            ...     print("Database optimized successfully")
        """
        try:
            # Optimisation de la fact table
            if not self.data_mgr.optimize_table():
                self.logger.warning("Failed to optimize fact table")
            
            # Nettoyage des données orphelines
            self._cleanup_orphaned_data()
            
            # Nettoyage des anciennes transactions
            cleaned_txs = self.transaction_mgr.cleanup_old_transactions()
            if cleaned_txs > 0:
                self.logger.info(f"Cleaned up {cleaned_txs} old transactions")
            
            # Logging
            self.logger.info("Database optimization completed")
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Error during database optimization: {e}")
            return False