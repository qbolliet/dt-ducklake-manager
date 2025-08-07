# Module pour la mise à jour incrémentale des tables
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import duckdb
import hashlib
import json
from datetime import datetime

from ..builders.schema import SchemaBuilder
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

class DatabaseUpdater:
    """
    A class to handle incremental updates to DuckDB databases.
    
    Supports adding, updating, and merging data while maintaining
    consistency between fact, dimension, and metadata tables.
    """
    
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None):
        """
        Initialize the DatabaseUpdater.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Maximum unique values for categorical columns
            log_filename: Path to log file
        """
        self.conn = connection
        self.categorical_threshold = categorical_threshold
        
        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/database_updater.log")
        self.logger = _init_logger(filename=log_filename)
        
        # Cache pour les métadonnées
        self._metadata_cache = None
        self._dimension_cache = {}
    
    def load_current_metadata(self) -> pd.DataFrame:
        """
        Load current metadata from the database.
        
        Returns:
            DataFrame containing current metadata
        """
        if self._metadata_cache is None:
            try:
                self._metadata_cache = self.conn.execute("SELECT * FROM metadata").fetchdf()
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")
                raise
        
        return self._metadata_cache
    
    def detect_schema_changes(self, new_df: pd.DataFrame) -> Dict:
        """
        Detect schema changes between new data and existing database.
        
        Args:
            new_df: New DataFrame to compare
            
        Returns:
            Dictionary describing detected changes
        """
        changes = {
            'new_columns': [],
            'removed_columns': [],
            'type_changes': [],
            'categorical_changes': []
        }
        
        # Chargement des métadonnées actuelles
        current_metadata = self.load_current_metadata()
        current_columns = set(current_metadata['name'].values)
        new_columns = set(new_df.columns)
        
        # Nouvelles colonnes
        changes['new_columns'] = list(new_columns - current_columns)
        
        # Colonnes supprimées (non présentes dans les nouvelles données)
        changes['removed_columns'] = list(current_columns - new_columns)
        
        # Vérification des changements de type
        for col in current_columns.intersection(new_columns):
            current_type = current_metadata[current_metadata['name'] == col]['python_type'].values[0]
            new_type = str(new_df[col].dtype)
            
            if current_type != new_type:
                changes['type_changes'].append({
                    'column': col,
                    'old_type': current_type,
                    'new_type': new_type
                })
            
            # Vérification si une colonne devient catégorielle ou non
            current_is_cat = current_metadata[current_metadata['name'] == col]['is_categorical'].values[0]
            new_unique_count = new_df[col].nunique()
            new_is_cat = (str(new_df[col].dtype) == 'object' and 
                         new_unique_count <= self.categorical_threshold)
            
            if current_is_cat != new_is_cat:
                changes['categorical_changes'].append({
                    'column': col,
                    'was_categorical': current_is_cat,
                    'is_categorical': new_is_cat
                })
        
        return changes
    
    def add_new_columns(self, new_columns: List[str], new_df: pd.DataFrame) -> None:
        """
        Add new columns to the fact table and update metadata.
        
        Args:
            new_columns: List of new column names
            new_df: DataFrame containing the new columns
        """
        for col in new_columns:
            dtype = str(new_df[col].dtype)
            sql_type = SchemaBuilder._map_python_to_sql_type(dtype)
            
            # Ajout de la colonne à la fact table
            try:
                # Valeur par défaut selon le type
                default_value = "NULL"
                if 'int' in dtype:
                    default_value = "0"
                elif 'float' in dtype:
                    default_value = "0.0"
                elif dtype == 'bool':
                    default_value = "FALSE"
                elif dtype == 'object':
                    default_value = "''"
                
                alter_query = f"ALTER TABLE fact_table ADD COLUMN {col} {sql_type} DEFAULT {default_value}"
                self.conn.execute(alter_query)
                
                self.logger.info(f"Added column {col} to fact_table")
                
            except Exception as e:
                self.logger.error(f"Failed to add column {col}: {e}")
                raise
            
            # Mise à jour des métadonnées
            self.update_metadata_for_column(col, new_df)
            
            # Si la colonne est catégorielle, créer la table de dimension
            if dtype == 'object' and new_df[col].nunique() <= self.categorical_threshold:
                self.create_dimension_table(col, new_df[col].unique())
    
    def update_metadata_for_column(self, column: str, df: pd.DataFrame) -> None:
        """
        Update or insert metadata for a specific column.
        
        Args:
            column: Column name
            df: DataFrame containing the column
        """
        dtype = str(df[column].dtype)
        sql_type = SchemaBuilder._map_python_to_sql_type(dtype)
        is_categorical = (dtype == 'object' and 
                         df[column].nunique() <= self.categorical_threshold)
        
        # Vérification si la colonne existe déjà dans les métadonnées
        existing = self.conn.execute(
            "SELECT COUNT(*) FROM metadata WHERE name = ?", 
            [column]
        ).fetchone()[0]
        
        if existing > 0:
            # Mise à jour
            update_query = """
                UPDATE metadata 
                SET python_type = ?, sql_type = ?, is_categorical = ?
                WHERE name = ?
            """
            self.conn.execute(update_query, [dtype, sql_type, is_categorical, column])
            self.logger.info(f"Updated metadata for column {column}")
        else:
            # Insertion
            insert_query = """
                INSERT INTO metadata (name, label, python_type, sql_type, is_categorical)
                VALUES (?, ?, ?, ?, ?)
            """
            label = column.replace('_', ' ').title()
            self.conn.execute(insert_query, [column, label, dtype, sql_type, is_categorical])
            self.logger.info(f"Inserted metadata for column {column}")
    
    def create_dimension_table(self, dimension_name: str, values: np.ndarray) -> None:
        """
        Create or update a dimension table.
        
        Args:
            dimension_name: Name of the dimension
            values: Unique values for the dimension
        """
        table_name = f"dim_{dimension_name}"
        
        # Création de la table si elle n'existe pas
        create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                value VARCHAR PRIMARY KEY,
                label VARCHAR
            )
        """
        self.conn.execute(create_query)
        
        # Préparation des données
        dim_df = pd.DataFrame({
            'value': values,
            'label': values
        })
        dim_df = dim_df.dropna().drop_duplicates()
        
        # Insertion des nouvelles valeurs uniquement
        for _, row in dim_df.iterrows():
            try:
                insert_query = f"""
                    INSERT INTO {table_name} (value, label) 
                    VALUES (?, ?)
                    ON CONFLICT (value) DO NOTHING
                """
                self.conn.execute(insert_query, [str(row['value']), str(row['label'])])
            except Exception as e:
                self.logger.warning(f"Could not insert value {row['value']} into {table_name}: {e}")
        
        self.logger.info(f"Updated dimension table {table_name}")
    
    def merge_data(self, 
                   new_df: pd.DataFrame, 
                   merge_keys: Optional[List[str]] = None,
                   update_mode: str = 'upsert') -> Dict:
        """
        Merge new data into existing fact table.
        
        Args:
            new_df: New data to merge
            merge_keys: Columns to use as merge keys (for identifying duplicates)
            update_mode: 'append', 'upsert', or 'replace'
            
        Returns:
            Dictionary with merge statistics
        """
        stats = {
            'rows_added': 0,
            'rows_updated': 0,
            'rows_unchanged': 0,
            'new_columns': [],
            'new_dimension_values': {}
        }
        
        # Détection des changements de schéma
        schema_changes = self.detect_schema_changes(new_df)
        
        # Ajout des nouvelles colonnes si nécessaire
        if schema_changes['new_columns']:
            self.add_new_columns(schema_changes['new_columns'], new_df)
            stats['new_columns'] = schema_changes['new_columns']
        
        # Préparation des données pour l'insertion
        prepared_df = self.prepare_dataframe_for_insert(new_df)
        
        if update_mode == 'append':
            # Simple ajout des données
            stats['rows_added'] = self.append_to_fact_table(prepared_df)
            
        elif update_mode == 'upsert':
            # Mise à jour ou insertion basée sur les clés
            if not merge_keys:
                self.logger.warning("No merge keys specified for upsert, falling back to append")
                stats['rows_added'] = self.append_to_fact_table(prepared_df)
            else:
                stats.update(self.upsert_to_fact_table(prepared_df, merge_keys))
                
        elif update_mode == 'replace':
            # Remplacement complet des données
            self.conn.execute("DELETE FROM fact_table")
            stats['rows_added'] = self.append_to_fact_table(prepared_df)
            
        # Mise à jour des tables de dimension
        stats['new_dimension_values'] = self.update_dimension_tables(new_df)
        
        # Invalidation du cache
        self._metadata_cache = None
        self._dimension_cache = {}
        
        return stats
    
    def prepare_dataframe_for_insert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for insertion by converting categorical values to IDs.
        
        Args:
            df: DataFrame to prepare
            
        Returns:
            Prepared DataFrame
        """
        prepared_df = df.copy()
        metadata = self.load_current_metadata()
        
        for _, row in metadata.iterrows():
            if row['is_categorical'] and row['name'] in prepared_df.columns:
                # Remplacement des valeurs par les IDs (dans ce cas, on garde les valeurs)
                # car DuckDB gère bien les jointures sur les strings
                pass
        
        return prepared_df
    
    def append_to_fact_table(self, df: pd.DataFrame) -> int:
        """
        Append data to fact table.
        
        Args:
            df: DataFrame to append
            
        Returns:
            Number of rows added
        """
        initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        
        # Utilisation de la méthode register pour insérer efficacement
        self.conn.register('temp_insert', df)
        
        # Récupération des colonnes existantes dans fact_table
        existing_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
        
        # Sélection uniquement des colonnes existantes
        columns_to_insert = [col for col in df.columns if col in existing_columns]
        columns_str = ", ".join(columns_to_insert)
        
        insert_query = f"""
            INSERT INTO fact_table ({columns_str})
            SELECT {columns_str} FROM temp_insert
        """
        
        self.conn.execute(insert_query)
        self.conn.execute("DROP VIEW temp_insert")
        
        final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        rows_added = final_count - initial_count
        
        self.logger.info(f"Added {rows_added} rows to fact_table")
        return rows_added
    
    def upsert_to_fact_table(self, df: pd.DataFrame, merge_keys: List[str]) -> Dict:
        """
        Perform upsert operation on fact table.
        
        Args:
            df: DataFrame to upsert
            merge_keys: Columns to use as merge keys
            
        Returns:
            Dictionary with upsert statistics
        """
        stats = {'rows_added': 0, 'rows_updated': 0}
        
        # Création d'une table temporaire
        self.conn.register('temp_upsert', df)
        
        # Identification des lignes existantes
        merge_condition = " AND ".join([f"f.{key} = t.{key}" for key in merge_keys])
        
        # Comptage des lignes qui seront mises à jour
        update_count_query = f"""
            SELECT COUNT(*) FROM fact_table f
            WHERE EXISTS (
                SELECT 1 FROM temp_upsert t
                WHERE {merge_condition}
            )
        """
        stats['rows_updated'] = self.conn.execute(update_count_query).fetchone()[0]
        
        # Mise à jour des lignes existantes
        if stats['rows_updated'] > 0:
            # Construction de la clause SET
            existing_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
            update_columns = [col for col in df.columns if col in existing_columns and col not in merge_keys]
            set_clause = ", ".join([f"f.{col} = t.{col}" for col in update_columns])
            
            update_query = f"""
                UPDATE fact_table f
                SET {set_clause}
                FROM temp_upsert t
                WHERE {merge_condition}
            """
            self.conn.execute(update_query)
        
        # Insertion des nouvelles lignes
        insert_query = f"""
            INSERT INTO fact_table
            SELECT t.* FROM temp_upsert t
            WHERE NOT EXISTS (
                SELECT 1 FROM fact_table f
                WHERE {merge_condition}
            )
        """
        
        initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        self.conn.execute(insert_query)
        final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        
        stats['rows_added'] = final_count - initial_count - stats['rows_updated']
        
        self.conn.execute("DROP VIEW temp_upsert")
        
        self.logger.info(f"Upsert complete: {stats['rows_added']} added, {stats['rows_updated']} updated")
        return stats
    
    def update_dimension_tables(self, df: pd.DataFrame) -> Dict:
        """
        Update dimension tables with new values from DataFrame.
        
        Args:
            df: DataFrame containing potential new dimension values
            
        Returns:
            Dictionary mapping dimension names to number of new values added
        """
        new_values = {}
        metadata = self.load_current_metadata()
        
        for _, row in metadata.iterrows():
            if row['is_categorical'] and row['name'] in df.columns:
                dimension_name = row['name']
                table_name = f"dim_{dimension_name}"
                
                # Récupération des valeurs actuelles
                try:
                    current_values = set(
                        self.conn.execute(f"SELECT value FROM {table_name}").fetchdf()['value']
                    )
                except:
                    current_values = set()
                
                # Nouvelles valeurs uniques
                new_unique = set(df[dimension_name].dropna().unique())
                values_to_add = new_unique - current_values
                
                if values_to_add:
                    # Création ou mise à jour de la table de dimension
                    self.create_dimension_table(dimension_name, list(values_to_add))
                    new_values[dimension_name] = len(values_to_add)
        
        return new_values
    
    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """
        Compute hash of DataFrame for change detection.
        
        Args:
            df: DataFrame to hash
            
        Returns:
            SHA256 hash of the data
        """
        # Conversion du DataFrame en bytes pour le hashing
        df_bytes = pd.util.hash_pandas_object(df, index=False).values.tobytes()
        return hashlib.sha256(df_bytes).hexdigest()
    
    def track_update_history(self, operation: str, stats: Dict) -> None:
        """
        Track update history in a dedicated table.
        
        Args:
            operation: Type of operation performed
            stats: Statistics from the operation
        """
        # Création de la table d'historique si elle n'existe pas
        create_history_query = """
            CREATE TABLE IF NOT EXISTS update_history (
                update_id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                operation VARCHAR,
                stats JSON,
                user VARCHAR,
                description VARCHAR
            )
        """
        self.conn.execute(create_history_query)
        
        # Insertion de l'historique
        insert_query = """
            INSERT INTO update_history (timestamp, operation, stats)
            VALUES (?, ?, ?)
        """
        self.conn.execute(insert_query, [
            datetime.now(),
            operation,
            json.dumps(stats)
        ])
        
        self.logger.info(f"Tracked update history for operation: {operation}")