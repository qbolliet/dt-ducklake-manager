# Importation des modules
# Modules de base
import os
import pandas as pd
import numpy as np
from pathlib import Path
# Types
from typing import Dict, List, Optional, Literal
# DuckDB
import duckdb
# Traitement en parallèle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import des éléments du module
# Import du module de construction du schéma
from ..builders.schema import SchemaBuilder
# Import du module d'indexation
from ..builders.indexer import IndexManager
# Import de l'utilitaire de logging
from ..utils.logger import _init_logger
# Utilitaires de traitement des données
from ..utils.data_processing import (map_python_to_sql_type, remove_dataframe_duplicates, 
                                      build_database_duplicate_removal_query, check_categorical_threshold)

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe de mise à jour de la base de données
class DatabaseUpdater:
    """
    A class to handle incremental updates to DuckDB databases.
    
    Supports updating metadata, fact tables, and dimension tables while maintaining
    consistency and handling type conflicts. Includes duplicate removal and index management.
    """
    # Initialisation
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None,
                 max_workers: int = 4,
                 batch_size: int = 10000):
        """
        Initialize the DatabaseUpdater.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Maximum unique values for categorical columns
            log_filename: Path to log file
            max_workers: Maximum number of parallel workers
            batch_size: Size of batches for processing
        """
        # Initialisation des arguments
        self.conn = connection
        self.categorical_threshold = categorical_threshold
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/database_updater.log")
        self.logger = _init_logger(filename=log_filename)
        
        # Cache pour les métadonnées
        self._metadata_cache = None
        
        # Verrou pour les opérations thread-safe
        self._lock = threading.Lock()
    
    # Méthode de mise à jour de la base de données
    def update_database(self,
                       update_df: pd.DataFrame,
                       check_duplicates_db: bool = True,
                       check_duplicates_update: bool = True,
                       keep: Literal[False, 'first', 'last'] = False,
                       index_config: Optional[Dict] = None,
                       use_batch_processing: bool = True) -> None:
        """
        Update the entire database with new data with support for batch and parallel processing.
        
        Args:
            update_df: DataFrame containing update data
            check_duplicates_db: Whether to check duplicates in existing database
            check_duplicates_update: Whether to check duplicates in update DataFrame
            keep: Which duplicates to keep when removing duplicates
            index_config: Dictionary containing index configuration
            use_batch_processing: Whether to use batch processing for large datasets
        """
        # Logging
        self.logger.info(f"Starting database update with {len(update_df)} rows")
        
        # Étape 1: Suppression des doublons dans les données de mise à jour si demandée
        if check_duplicates_update:
            update_df = remove_dataframe_duplicates(update_df, keep, self.logger, 'update')
        
        # Étape 2: Suppression des doublons dans la base de données existante si demandée
        if check_duplicates_db:
            self._remove_duplicates_from_database(keep)
        
        # Détermination si on utilise le traitement par batch
        should_use_batches = use_batch_processing and len(update_df) > self.batch_size
        
        if should_use_batches:
            self.logger.info(f"Using batch processing with batch size {self.batch_size}")
            # Traitement par lots
            self._process_database_update_in_batches(update_df, index_config)
        else:
            # Traitement direct pour les petites données
            self._process_database_update_direct(update_df, index_config)
        
        # Invalidation du cache des métadonnées
        self._metadata_cache = None
        
        # Logging
        self.logger.info("Database update completed")
    
    # Méthode de processing de la mise à jour en batch
    def _process_database_update_in_batches(self, update_df: pd.DataFrame, index_config: Optional[Dict]) -> None:
        """
        Process database update in batches for large datasets.
        
        Args:
            update_df: DataFrame containing update data
            index_config: Dictionary containing index configuration
        """
        # Étape 3: Mise à jour des métadonnées (une seule fois pour toutes les données)
        self.update_metadata(update_df)
        
        # Étape 4: Mise à jour des tables de dimensions (une seule fois avec toutes les valeurs uniques)
        self.update_dimension_tables(update_df)
        
        # Étape 5: Mise à jour de la table de faits par lots
        def process_fact_table_batch(batch_df):
            self.update_fact_table(batch_df)
        
        self._process_large_dataframe_in_batches(update_df, process_fact_table_batch)
        
        # Étape 6: Mise à jour des index si spécifiés
        if index_config is not None:
            self.update_indexes(index_config)
    
    # Méthode de processing direct de la mise à jour
    def _process_database_update_direct(self, update_df: pd.DataFrame, index_config: Optional[Dict]) -> None:
        """
        Process database update directly without batching.
        
        Args:
            update_df: DataFrame containing update data
            index_config: Dictionary containing index configuration
        """
        # Étape 3: Mise à jour des métadonnées
        self.update_metadata(update_df)
        
        # Étape 4: Mise à jour des tables de dimensions (avec parallélisation interne si applicable)
        self.update_dimension_tables(update_df)
        
        # Étape 5: Mise à jour de la table de faits
        self.update_fact_table(update_df)
        
        # Étape 6: Mise à jour des index si spécifiés
        if index_config is not None:
            self.update_indexes(index_config)
    
    
    # Méthode de mise à jour des métadonnées
    def update_metadata(self, update_df: pd.DataFrame, column_labels: Optional[Dict[str, str]] = None) -> None:
        """
        Update metadata table with new columns and resolve type conflicts.
        
        Args:
            update_df: DataFrame containing new data
            column_labels: Dictionary mapping column names to custom labels. If None, uses default labels.
        """
        # Logging
        self.logger.info("Updating metadata table")
        
        # Chargement des métadonnées actuelles
        current_metadata = self._load_current_metadata()
        current_columns = set(current_metadata['name'].values) if len(current_metadata) > 0 else set()
        new_columns = set(update_df.columns)
        
        # Identification des nouvelles colonnes
        columns_to_add = new_columns - current_columns
        
        # Ajout des nouvelles colonnes aux métadonnées
        for col in columns_to_add:
            label = column_labels.get(col) if column_labels else None
            self._add_column_to_metadata(col, update_df, label)
        
        # Vérification des conflits de types pour les colonnes existantes
        for col in current_columns.intersection(new_columns):
            self._resolve_type_conflicts(col, update_df, current_metadata)
    
    # Méthode de mise à jour de la table des faits
    def update_fact_table(self, update_df: pd.DataFrame) -> None:
        """
        Update fact table with new columns, rows, and type changes.
        
        Args:
            update_df: DataFrame containing update data
        """
        # Logging
        self.logger.info("Updating fact table")
        
        # Récupération de la structure actuelle de la fact table
        try:
            current_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
        except:
            # Table n'existe pas encore
            current_columns = []
        
        # Mise à jour préalable des tables de dimension avec les nouvelles valeurs catégorielles
        self._ensure_dimension_values_exist(update_df)
        
        # Conversion des colonnes catégorielles en utilisant les valeurs des tables de dimension
        prepared_df = self._prepare_dataframe_for_fact_table(update_df)
        
        # Ajout des colonnes manquantes
        new_columns = set(prepared_df.columns) - set(current_columns)
        for col in new_columns:
            self._add_column_to_fact_table(col, prepared_df)
        
        # Mise à jour des types de colonnes si nécessaire
        for col in prepared_df.columns:
            self._update_column_type_in_fact_table(col, prepared_df)
        
        # Ajout/mise à jour des lignes
        if len(current_columns) > 0:
            # Mise à jour avec upsert basé sur toutes les colonnes communes
            merge_keys = list(set(prepared_df.columns).intersection(set(current_columns)))
            self._upsert_to_fact_table(prepared_df, merge_keys)
        else:
            # Création de la table avec les données
            self._create_fact_table_with_data(prepared_df)
        
        # Détection post-upsert des variables devenues catégorielles
        self._detect_new_categorical_variables_after_upsert()
    
    # Méthode de mise à jour des tables de dimension
    def update_dimension_tables(self, update_df: pd.DataFrame) -> None:
        """
        Update dimension tables using unified logic with internal parallelization.
        Leverages _update_dimension_values for robust table management.
        
        Args:
            update_df: DataFrame containing update data
        """
        # Logging
        self.logger.info("Updating dimension tables")
        
        # Chargement des métadonnées actuelles
        current_metadata = self._load_current_metadata()
        
        # Classification des colonnes par statut catégoriel
        categorical_columns = []
        non_categorical_columns = []
        
        for _, row in current_metadata.iterrows():
            col_name = row['name']
            if col_name in update_df.columns:
                if row['is_categorical']:
                    categorical_columns.append(col_name)
                elif row['python_type'] == 'object':
                    non_categorical_columns.append(col_name)
        
        # Traitement des colonnes catégorielles existantes avec parallélisation si possible
        if len(categorical_columns) > 1 and self.max_workers > 1:
            # Traitement parallèle
            columns_and_values = [(col, update_df[col]) for col in categorical_columns]
            self._parallel_dimension_update(columns_and_values)
        else:
            # Traitement séquentiel
            for col_name in categorical_columns:
                self._update_dimension_values(col_name, update_df[col_name])
        
        # Traitement des colonnes non-catégorielles (vérification si elles deviennent catégorielles)
        for col_name in non_categorical_columns:
            self._check_and_convert_to_categorical(col_name, update_df[col_name])
        
        # Vérification des colonnes catégorielles qui pourraient dépasser le seuil
        for col_name in categorical_columns:
            self._check_categorical_threshold(col_name, update_df[col_name])
    
    # Méthodes auxilaire convertissant une variable en catégorielle si elle en satisfait les conditions
    def _check_and_convert_to_categorical(self, col_name: str, values: pd.Series) -> None:
        """
        Check if a non-categorical column should become categorical and convert if needed.
        
        Args:
            col_name: Column name
            values: Series with column values
        """
        if check_categorical_threshold(values, self.categorical_threshold):
            
            # Création de la table de dimension via _update_dimension_values
            self._update_dimension_values(col_name, values)
            
            # Mise à jour du statut catégoriel
            self._update_categorical_status(col_name, True)
            
            # Logging
            self.logger.info(f"Converted {col_name} to categorical with dim_{col_name}")
    
    # Méthode auxiliaire convertissant une variable catégorielle en variable non catégorielle si elle excède un certain seuil
    def _check_categorical_threshold(self, col_name: str, values: pd.Series) -> None:
        """
        Check if a categorical column exceeds threshold and convert to non-categorical if needed.
        
        Args:
            col_name: Column name
            values: Series with column values
        """
        # Nom de la table de dimension
        table_name = f"dim_{col_name}"
        
        # Récupération des valeurs existantes
        try:
            existing_result = self.conn.execute(f"SELECT label FROM {table_name}").fetchdf()
            existing_values = set(existing_result['label']) if len(existing_result) > 0 else set()
        except:
            existing_values = set()
        
        # Calcul du nombre total de valeurs uniques
        new_values = set(values.dropna().unique())
        total_count = len(existing_values.union(new_values))
        
        # Vérification du seuil
        if total_count > self.categorical_threshold:
            # Conversion vers non-catégoriel
            self._convert_to_non_categorical(col_name)
    
    # Méthode de conversion vers le statut non-catégoriel
    def _convert_to_non_categorical(self, col_name: str) -> None:
        """
        Convert a column from categorical to non-categorical status.
        
        Args:
            col_name: Column name
        """
        # Nom de la table de dimension
        table_name = f"dim_{col_name}"
        
        # Remplacement des valeurs par les labels dans la fact table
        self._convert_fact_table_dimension_mapping(col_name, values_to_labels=True)
        
        # Suppression de la table de dimension
        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        
        # Mise à jour du statut dans les métadonnées
        self._update_categorical_status(col_name, False)
        
        self.logger.info(f"Converted {col_name} to non-categorical - dropped {table_name}")  
    
    
    # Méthode auxiliaire de mise à jour du statut catégoriel dans les méta-données
    def _update_categorical_status(self, col_name: str, is_categorical: bool) -> None:
        """
        Update categorical status in metadata.
        
        Args:
            col_name: Column name
            is_categorical: New categorical status
        """
        # Exécution de la requête de mise à jour
        self.conn.execute(
            "UPDATE metadata SET is_categorical = ? WHERE name = ?",
            [is_categorical, col_name]
        )
    
    # Méthode générique de conversion entre valeurs et labels dans la table des faits
    def _convert_fact_table_dimension_mapping(self, col_name: str, values_to_labels: bool = True) -> None:
        """
        Convert between dimension values and labels in the fact table.
        
        Args:
            col_name: Column name
            values_to_labels: If True, converts values→labels; if False, converts labels→values
        """
        # Nom de la table de dimension
        table_name = f"dim_{col_name}"
        
        try:
            # Récupération du mapping de la table de dimension
            dim_result = self.conn.execute(f"SELECT value, label FROM {table_name}").fetchdf()
            
            if len(dim_result) > 0:
                # Création d'une vue temporaire pour le mapping
                self.conn.register('temp_dim_mapping', dim_result)
                
                if values_to_labels:
                    # Conversion values → labels (pour revenir aux données originales)
                    update_query = f"""
                        UPDATE fact_table 
                        SET {col_name} = (
                            SELECT label FROM temp_dim_mapping 
                            WHERE temp_dim_mapping.value = fact_table.{col_name}
                        )
                        WHERE {col_name} IS NOT NULL
                    """
                    operation = "values to labels"
                else:
                    # Conversion labels → values (pour utiliser les index de dimension)
                    update_query = f"""
                        UPDATE fact_table 
                        SET {col_name} = (
                            SELECT value FROM temp_dim_mapping 
                            WHERE temp_dim_mapping.label = fact_table.{col_name}
                        )
                        WHERE {col_name} IS NOT NULL
                    """
                    operation = "labels to values"
                
                # Exécution de la mise à jour
                self.conn.execute(update_query)
                
                # Suppression de la vue temporaire
                self.conn.execute('DROP VIEW temp_dim_mapping')
                
                # Logging
                self.logger.info(f"Converted {operation} for {col_name} in fact table")
                
        except Exception as e:
            # Logging
            self.logger.error(f"Error converting dimension mapping for {col_name}: {e}")
    
    # Méthode de détection des variables devenues catégorielles après l'upsert
    def _detect_new_categorical_variables_after_upsert(self) -> None:
        """
        Detect variables that became categorical after the upsert and create dimension tables.
        """
        # Chargement des méta-données
        current_metadata = self._load_current_metadata()
        
        # Parcours des colonnes non-catégorielles dans les métadonnées
        non_categorical_cols = current_metadata[
            (current_metadata['is_categorical'] == False) &
            (current_metadata['python_type'] == 'object')
        ]
        # Parcours des colonnes
        for _, row in non_categorical_cols.iterrows():
            # Nom de la colonne
            col_name = row['name']
            
            try:
                # Vérification que la colonne existe dans la fact table
                fact_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
                if col_name not in fact_columns:
                    continue
                
                # Comptage des modalités uniques dans la fact table
                unique_count_query = f"SELECT COUNT(DISTINCT {col_name}) as count FROM fact_table WHERE {col_name} IS NOT NULL"
                unique_count = self.conn.execute(unique_count_query).fetchone()[0]
                
                # Si le nombre de modalités est inférieur au seuil, créer une table de dimension
                if unique_count <= self.categorical_threshold:
                    # Récupération des valeurs uniques des labels
                    unique_labels_query = f"SELECT DISTINCT {col_name} as label FROM fact_table WHERE {col_name} IS NOT NULL ORDER BY {col_name}"
                    unique_labels_result = self.conn.execute(unique_labels_query).fetchdf()
                    
                    if len(unique_labels_result) > 0:
                        # Création de la table de dimension
                        self._update_dimension_values(col_name, unique_labels_result['label'])
                        
                        # Mise à jour du statut catégoriel dans les métadonnées
                        self.conn.execute(
                            "UPDATE metadata SET is_categorical = ? WHERE name = ?",
                            [True, col_name]
                        )
                        
                        # Remplacement des labels par les valeurs de la table de dimension dans la fact table
                        self._convert_fact_table_dimension_mapping(col_name, values_to_labels=False)
                        # Logging
                        self.logger.info(f"Converted {col_name} to categorical after upsert ({unique_count} unique values)")
                        
            except Exception as e:
                # Logging
                self.logger.error(f"Error detecting categorical status for {col_name} after upsert: {e}")
    
    # Méthode de mise à jour des index
    def update_indexes(self, index_config: Dict) -> None:
        """
        Update database indexes by removing old ones and creating new ones.
        
        Args:
            index_config: Dictionary containing index configuration
        """
        # Logging
        self.logger.info("Updating indexes")
        
        # Création d'un gestionnaire d'index
        index_manager = IndexManager(connection=self.conn)
        
        # Suppression de tous les index existants (sauf les index système)
        try:
            # Récupération des index existants
            existing_indexes = self.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            """).fetchall()
            # Parcours des index
            for idx in existing_indexes:
                try:
                    index_manager.drop_index(idx[0])
                    self.logger.info(f"Dropped index {idx[0]}")
                except Exception as e:
                    self.logger.warning(f"Cannot drop index {idx[0]}: {e}")
        except Exception as e:
            self.logger.warning(f"Error retrieving existing indexes: {e}")
        
        # Création des nouveaux index selon la configuration
        try:
            index_manager.create_fact_table_indexes("fact_table", index_config)
            self.logger.info("Successfully created indexes based on configuration")
        except Exception as e:
            self.logger.error(f"Error creating indexes: {e}")
    
    # Méthodes pour le traitement par lots et la parallélisation
    # Méthode de processing en batchs
    def _process_large_dataframe_in_batches(self, df: pd.DataFrame, 
                                          process_func, batch_size: Optional[int] = None) -> None:
        """
        Process large pandas DataFrames in batches.
        
        Args:
            df: DataFrame to process
            process_func: Processing function to apply to each batch
            batch_size: Batch size (uses self.batch_size by default)
        """
        # Initialisation de la taille des batchs
        batch_size = batch_size or self.batch_size
        
        # Logging
        self.logger.info(f"Processing {len(df)} rows in batches of {batch_size}")
        # Parcours des observations du jeu de données en batch
        for i in range(0, len(df), batch_size):
            # Circonscription du batch
            batch_start = i
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end].copy()
            
            try:
                # Processing du batch
                process_func(batch_df)
            except Exception as e:
                # Logging
                self.logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
    
    # Méthode de mise à jour en parallèle des tables de dimension
    def _parallel_dimension_update(self, columns_and_values: List[tuple]) -> None:
        """
        Parallel update of dimension tables.
        
        Args:
            columns_and_values: List of tuples (column_name, unique_values)
        """
        # Logging
        self.logger.info(f"Parallel update of {len(columns_and_values)} dimensions")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Soumission des tâches
            future_to_dimension = {
                executor.submit(self._update_dimension_values, col_name, values): col_name
                for col_name, values in columns_and_values
            }
            
            # Récupération des résultats
            for future in as_completed(future_to_dimension):
                dimension = future_to_dimension[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Error during parallel update of {dimension}: {e}")
    
    # Méthode de chargement des méta-données
    def _load_current_metadata(self) -> pd.DataFrame:
        """
        Load current metadata from the database.
        
        Returns:
            DataFrame containing current metadata
        """
        # Chargement de la table si elle n'est pas en cache
        if self._metadata_cache is None:
            try:
                # Chargement
                self._metadata_cache = self.conn.execute("SELECT * FROM metadata").fetchdf()
            except:
                # Si la table n'existe pas encore, on l'initialise vide
                self._metadata_cache = pd.DataFrame(columns=['name', 'label', 'python_type', 'sql_type', 'is_categorical'])
        
        return self._metadata_cache
    
    
    # Méthode de préparation du DataFrame pour la fact table
    def _prepare_dataframe_for_fact_table(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for fact table by converting categorical labels to dimension table indices.
        
        Args:
            df: DataFrame to prepare
            
        Returns:
            Prepared DataFrame with categorical values as indices
        """
        # Copie indépendante du jeu de données
        prepared_df = df.copy()
        # Chargement des méta-données
        current_metadata = self._load_current_metadata()
        
        # Parcours des colonnes catégorielles
        for _, row in current_metadata.iterrows():
            # Nom de la colonne
            col_name = row['name']
            # Statut catégoriel
            is_categorical = row['is_categorical']
            
            if is_categorical and col_name in prepared_df.columns:
                try:
                    # Récupération des valeurs et labels de la table de dimension
                    dim_table = f"dim_{col_name}"
                    dim_result = self.conn.execute(f"SELECT value, label FROM {dim_table}").fetchdf()
                    
                    if len(dim_result) > 0:
                        # Création du mapping label -> value pour convertir les labels en values
                        label_to_value = dict(zip(dim_result['label'], dim_result['value']))
                        
                        # Remplacement des labels par les values de la dimension
                        # Toutes les valeurs devraient maintenant exister grâce à car la mise à jour des tables de dimension est réalisé avant et grace à _ensure_dimension_values_exist
                        prepared_df[col_name] = prepared_df[col_name].map(label_to_value)
                        
                        # Vérification des valeurs non mappées (ne devrait pas arriver normalement)
                        unmapped_mask = prepared_df[col_name].isna()
                        if unmapped_mask.any():
                            unmapped_count = unmapped_mask.sum()
                            # Logging
                            self.logger.error(f"Found {unmapped_count} unmapped labels in {col_name} after dimension update. "
                                            f"This indicates an error in the dimension update process.")
                            # Remplace par -1 comme fallback
                            prepared_df[col_name] = prepared_df[col_name].fillna(-1)
                        
                        # Conversion vers le type approprié selon le type des values dans la dimension
                        if len(dim_result) > 0 and dim_result['value'].dtype in ['int64', 'int32']:
                            prepared_df[col_name] = prepared_df[col_name].astype('int64')
                        else:
                            # Garde le type original si les values ne sont pas des entiers
                            pass
                    else:
                        # Table de dimension vide - garder les valeurs originales temporairement
                        self.logger.warning(f"Dimension table {dim_table} is empty, keeping original values")
                    
                except Exception as e:
                    # Logging
                    self.logger.warning(f"Could not convert categorical column {col_name}: {e}")
        
        return prepared_df
    
    # Méthode de mise à jour préalable des tables de dimension avec les nouvelles valeurs
    def _ensure_dimension_values_exist(self, df: pd.DataFrame) -> None:
        """
        Ensure all categorical values from the DataFrame exist in their respective dimension tables.
        This method should be called before preparing the DataFrame for the fact table.
        
        Args:
            df: DataFrame containing the data to be inserted
        """
        # Extraction des métadonnées
        current_metadata = self._load_current_metadata()
        
        # Parcours des colonnes catégorielles
        for _, row in current_metadata.iterrows():
            # Extraction du nom de la colonne
            col_name = row['name']
            # Extraction du statut catégoriel
            is_categorical = row['is_categorical']
            
            if is_categorical and col_name in df.columns:
                # Mise à jour des valeurs de dimension avec les nouvelles valeurs
                self._update_dimension_values(col_name, df[col_name])
    
    
    # Méthode de déduplication de la base de données
    def _remove_duplicates_from_database(self, keep: Literal[False, 'first', 'last']) -> int:
        """
        Remove duplicates from the existing fact table.
        
        Args:
            keep: Strategy for keeping duplicates
            
        Returns:
            Number of rows removed
        """
        try:
            # Récupération du nombre initial de lignes
            initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            # Récupération des colonnes (exclure 'value' si elle existe)
            all_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
            columns_to_check = [col for col in all_columns if col != 'value']
            
            if not columns_to_check:
                return 0
            
            # Construction de la requête de suppression des doublons
            delete_query = build_database_duplicate_removal_query(columns_to_check, keep, 'fact_table')
            
            self.conn.execute(delete_query)
            
            # Calcul des lignes supprimées
            final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            removed_count = initial_count - final_count
            
            if removed_count > 0:
                self.logger.warning(f"Duplicate removal (database): {removed_count} rows removed")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error removing duplicates from database: {e}")
            return 0
    
    # Méthode d'ajout des informations d'une colonne à la table des métadonnées
    def _add_column_to_metadata(self, column: str, df: pd.DataFrame, label: Optional[str] = None) -> None:
        """
        Add a new column to metadata.
        
        Args:
            column: Column name
            df: DataFrame containing the column
            label: Custom label for the column. If None, defaults to formatted column name.
        """
        # Identification du type de la colonne
        dtype = str(df[column].dtype)
        # Conversion du type en SQL
        sql_type = map_python_to_sql_type(dtype)
        # Définition si la variable est catégorielle
        is_categorical = (
            dtype == 'object' and 
            df[column].nunique() <= self.categorical_threshold
        )
        
        # Création de la table metadata si elle n'existe pas
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                name VARCHAR,
                label VARCHAR,
                python_type VARCHAR,
                sql_type VARCHAR,
                is_categorical BOOLEAN
            )
        """)
        
        # Insertion de la nouvelle colonne
        if label is None:
            label = column.replace('_', ' ').title()
        # Exécution de la requête
        self.conn.execute("""
            INSERT INTO metadata (name, label, python_type, sql_type, is_categorical)
            VALUES (?, ?, ?, ?, ?)
        """, [column, label, dtype, sql_type, is_categorical])
        # Logging
        self.logger.info(f"Added column {column} to metadata")
    
    # Méthode de résolutions des conflits de types
    def _resolve_type_conflicts(self, column: str, df: pd.DataFrame, 
                               current_metadata: pd.DataFrame) -> None:
        """
        Resolve type conflicts using the least restrictive strategy.
        
        Args:
            column: Column name
            df: DataFrame with new data
            current_metadata: Current metadata
        """
        # Identification du type actuel de la colonne dans la base de données
        current_type = current_metadata[current_metadata['name'] == column]['python_type'].values[0]
        # Identification du nouveau type
        new_type = str(df[column].dtype)
        # Ne fait rien si inchangé
        if current_type == new_type:
            return None
        
        # Hiérarchie des types (du plus contraignant au moins contraignant)
        type_hierarchy = {
            'bool': 1,
            'int64': 2,
            'float64': 3,
            'object': 4
        }
        
        current_level = type_hierarchy.get(current_type, 0)
        new_level = type_hierarchy.get(new_type, 0)
        
        # Sélection du type le moins contraignant
        if new_level > current_level:
            resolved_type = new_type
        else:
            resolved_type = current_type
        
        # Conversion en type SQL
        sql_type = map_python_to_sql_type(resolved_type)
        # Mise à jour des métadonnées
        self.conn.execute("""
            UPDATE metadata 
            SET python_type = ?, sql_type = ?
            WHERE name = ?
        """, [resolved_type, sql_type, column])
        # Logging
        self.logger.info(f"Type conflict resolution for {column}: {current_type} -> {resolved_type}")
    
    
    # Méthode d'ajout d'une nouvelle colonne à la table des faits
    def _add_column_to_fact_table(self, column: str, df: pd.DataFrame) -> None:
        """
        Add a new column to the fact table.
        
        Args:
            column: Column name
            df: DataFrame containing the column
        """
        # Identification du type de la colonne
        dtype = str(df[column].dtype)
        # Conversion du type en SQL
        sql_type = map_python_to_sql_type(dtype)
        
        # Ajout de la colonne avec des valeurs NULL (seront remplacées dans l'upsert)
        alter_query = f"ALTER TABLE fact_table ADD COLUMN {column} {sql_type} DEFAULT NULL"
        # Exécution de la requête
        self.conn.execute(alter_query)
        # Logging
        self.logger.info(f"Added column {column} to fact table")
    
    # Mise à jour du type de la colonne dans la table des faist
    def _update_column_type_in_fact_table(self, column: str, df: pd.DataFrame) -> None:
        """
        Update column type in fact table using DuckDB ALTER TABLE syntax.
        
        Args:
            column: Column name
            df: DataFrame with the new data type
        """
        try:
            # Récupération du type actuel de la colonne
            current_columns = self.conn.execute("DESCRIBE fact_table").fetchall()
            current_type = None
            for col_info in current_columns:
                if col_info[0] == column:
                    current_type = col_info[1]
                    break
            
            # Détermination du nouveau type SQL
            new_dtype = str(df[column].dtype)
            new_sql_type = map_python_to_sql_type(new_dtype)
            
            # Mise à jour du type si nécessaire
            if current_type and current_type.upper() != new_sql_type.upper():
                # Création de la requête
                alter_query = f"ALTER TABLE fact_table ALTER {column} SET DATA TYPE {new_sql_type}"
                # Exécution de la requête
                self.conn.execute(alter_query)
                # Logging
                self.logger.info(f"Updated column {column} type from {current_type} to {new_sql_type}")
        except Exception as e:
            # Logging
            self.logger.warning(f"Could not update column type for {column}: {e}")
    
    # Méthode de création de la table des faits avec les données
    def _create_fact_table_with_data(self, df: pd.DataFrame) -> None:
        """
        Create the fact table with data.
        
        Args:
            df: DataFrame with data
        """
        # Enregistrement d'une table temporaires qui recopie le jeu de données
        self.conn.register('temp_fact_creation', df)
        # Recopie dans la table des faits
        self.conn.execute("""
            CREATE TABLE fact_table AS 
            SELECT * FROM temp_fact_creation
        """)
        # Suppression de la table temporaire
        self.conn.execute('DROP VIEW temp_fact_creation')
        
        # Logging du nombre de lignes ajoutées
        rows_added = len(df)
        self.logger.info(f"Created fact table with {rows_added} rows")
    
    # Méthode d'insertion et de mise à jour de données dans la table des faits
    def _upsert_to_fact_table(self, df: pd.DataFrame, merge_keys: List[str]) -> None:
        """
        Upsert data into the fact table.
        
        Args:
            df: DataFrame to insert/update
            merge_keys: Columns used to identify duplicates
        """
        # Enregistrement d'une vue temporaire du jeu de données
        self.conn.register('temp_upsert', df)
        
        # Construction des conditions de jointure
        merge_condition = " AND ".join([f"f.{key} = t.{key}" for key in merge_keys])
        
        # Comptage des mises à jour
        update_count_query = f"""
            SELECT COUNT(*) FROM fact_table f
            WHERE EXISTS (
                SELECT 1 FROM temp_upsert t
                WHERE {merge_condition}
            )
        """
        rows_updated = self.conn.execute(update_count_query).fetchone()[0]
        
        # Mise à jour des lignes existantes
        if rows_updated > 0:
            existing_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
            update_columns = [col for col in df.columns if col in existing_columns and col not in merge_keys]
            
            if update_columns:
                set_clause = ", ".join([f"f.{col} = t.{col}" for col in update_columns])
                # Construction de la requête
                update_query = f"""
                    UPDATE fact_table f
                    SET {set_clause}
                    FROM temp_upsert t
                    WHERE {merge_condition}
                """
                # Exécution de la requête
                self.conn.execute(update_query)
        
        # Insertion des nouvelles lignes
        initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        # Construction de la requête
        insert_query = f"""
            INSERT INTO fact_table
            SELECT t.* FROM temp_upsert t
            WHERE NOT EXISTS (
                SELECT 1 FROM fact_table f
                WHERE {merge_condition}
            )
        """
        # Exécution de la requête
        self.conn.execute(insert_query)
        
        # Comptage du nombre de lignes ajoutées
        final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        rows_added = final_count - initial_count
        
        # Suppression de la vue temporaire
        self.conn.execute("DROP VIEW temp_upsert")
        
        # Logging
        self.logger.info(f"Upsert completed: {rows_added} rows added, {rows_updated} rows updated")
    
    # Méthode de mise à jour d'une table de dimension
    def _update_dimension_values(self, dimension_name: str, labels: pd.Series) -> None:
        """
        Update values of a dimension table.
        
        Args:
            dimension_name: Dimension name
            labels: Series with new labels
        """
        with self._lock:
            # Nom de la table de dimension
            table_name = f"dim_{dimension_name}"
            
            # Création de la table si elle n'existe pas
            self.conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    value VARCHAR PRIMARY KEY,
                    label VARCHAR
                )
            """)
            
            # Récupération des valeurs existantes
            try:
                # Exécution de la requête
                dim_result = self.conn.execute(f"SELECT value, label FROM {table_name}").fetchdf()
                existing_labels = set(dim_result['label'])
                existing_values = set(dim_result['value'])
            except:
                existing_labels = set()
                existing_values = set()
            
            # Identification des nouvelles valeurs
            unique_labels = set(labels.dropna().unique()) if isinstance(labels, pd.Series) else set(labels)
            new_labels = unique_labels - existing_labels
            # Construction des nouvelles valeurs 
            if len(existing_values) > 0:
                max_value = max(existing_values)
                new_values = list(range(max_value + 1, max_value + 1 + len(new_labels)))
            else:
                new_values = list(range(len(new_labels)))
            # Insertion des nouvelles valeurs et labels
            values_added = 0
            for value, label in zip(new_values, new_labels):
                try:
                    self.conn.execute(f"""
                        INSERT INTO {table_name} (value, label) 
                        VALUES (?, ?)
                        ON CONFLICT (value) DO NOTHING
                    """, [value, str(label)])
                    values_added += 1
                except Exception as e:
                    # Logging
                    self.logger.warning(f"Cannot insert {value} into {table_name}: {e}")
            
            if values_added > 0:
                # Logging
                self.logger.info(f"Added {values_added} new values to {table_name}")
        