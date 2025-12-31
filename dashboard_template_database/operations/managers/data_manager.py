# Importation des modules
# Modules de base
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
# DuckDB
import duckdb

# Import du gestionnaire de base
from .base_manager import BaseSchemaManager
# Import des utilitaires
from ...utils.data_processing import map_python_to_sql_type, _build_where_clause

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Classe de gestion des opérations sur la table des faits en accord ave la table des méta-données
class DataManager(BaseSchemaManager):
    """
    Manages fact table operations including inserts, updates, upserts, and deletes.
    
    Handles data type management, column operations, and maintains consistency
    with dimension tables. Provides batch processing capabilities for large datasets.
    
    Attributes:
        batch_size (int): Size of batches for processing large datasets
    """
    # Initialisation
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None,
                 batch_size: int = 10000):
        """
        Initialize the data manager.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Threshold for determining categorical variables
            log_filename: Path to log file
            batch_size: Size of batches for processing large datasets
            
        Example:
            >>> conn = duckdb.connect('database.db')
            >>> data_mgr = DataManager(conn, batch_size=5000)
        """
        # Initialisation du parent
        super().__init__(connection, categorical_threshold, log_filename)
        
        # Configuration pour le traitement par lots
        self.batch_size = batch_size
    
    # Méthodes de validation des opérations avant leur exécution
    # Méthode de validation de l'ensemble des opérations
    def validate_operation(self, operation_type: str, **kwargs) -> bool:
        """
        Validate data operations before execution.
        
        Args:
            operation_type: Type of operation ('insert', 'update', 'upsert', 'delete', 'add_column', 'drop_column')
            **kwargs: Operation-specific parameters
            
        Returns:
            True if operation is valid
        """
        # Choix de la méthode appropriée suivant le type d'opération
        if operation_type == 'insert':
            return self._validate_insert(**kwargs)
        elif operation_type == 'update':
            return self._validate_update(**kwargs)
        elif operation_type == 'upsert':
            return self._validate_upsert(**kwargs)
        elif operation_type == 'delete':
            return self._validate_delete(**kwargs)
        elif operation_type == 'add_column':
            return self._validate_add_column(**kwargs)
        elif operation_type == 'drop_column':
            return self._validate_drop_column(**kwargs)
        else:
            # Logging
            self.logger.warning(f"Unknown operation type: {operation_type}")
            return False
    
    # Méthode de validation des insertions de données
    def _validate_insert(self, df: pd.DataFrame, **kwargs) -> bool:
        """Validate data insertion parameters."""
        # Validation du jeu de données à insérer
        if df is None or len(df) == 0:
            # Logging
            self.logger.error("DataFrame cannot be empty for insertion")
            return False
        
        # Vérification des noms de colonnes valides
        invalid_columns = [col for col in df.columns if not col.replace('_', '').isalnum()]
        if invalid_columns:
            self.logger.error(f"Invalid column names: {invalid_columns}")
            return False
        
        return True
    
    # Méthode de validation de la mise à jour des données
    def _validate_update(self, df: pd.DataFrame, merge_keys: List[str], **kwargs) -> bool:
        """Validate data update parameters."""
        # Validation de l'insertion

        if not self._validate_insert(df, **kwargs):
            return False
        # Vérification de l'existence des clés d'appariement
        if not merge_keys:
            # Logging
            self.logger.error("Merge keys cannot be empty for update operation")
            return False
        
        # Vérification que les clés d'appariement existent dans le DataFrame
        missing_keys = [key for key in merge_keys if key not in df.columns]
        if missing_keys:
            # Logging
            self.logger.error(f"Merge keys not found in DataFrame: {missing_keys}")
            return False
        
        return True
    
    # Méthode de validation de la mise à jour et l'insertion de données
    def _validate_upsert(self, df: pd.DataFrame, merge_keys: List[str], **kwargs) -> bool:
        """Validate data upsert parameters."""
        return self._validate_update(df, merge_keys, **kwargs)
    
    # Méthode de validation de la suppression de données
    def _validate_delete(self, filters: Optional[Union[str, List]], **kwargs) -> bool:
        """Validate data deletion parameters."""

        # Vérification de la spécification des filtres
        if filters is None:
            # Logging
            self.logger.error("Filters cannot be None for delete operation")
            return False
        
        return True
    
    # Méthode de validation de l'ajout d'une colonne
    def _validate_add_column(self, column_name: str, df: pd.DataFrame, **kwargs) -> bool:
        """Validate column addition parameters."""
        # Vérification que le nom est valide
        if not column_name or column_name.strip() == "":
            # logging
            self.logger.error("Column name cannot be empty")
            return False
        # Vérification que le nom de colonne n'est pas présent dans le jeu de données
        if column_name not in df.columns:
            self.logger.error(f"Column {column_name} not found in DataFrame")
            return False
        
        return True
    
    # Méthode de validation de la suppression de colonnes de la table des faits
    def _validate_drop_column(self, columns: List[str], **kwargs) -> bool:
        """Validate column drop parameters."""
        # Vérification de la spécification des colonnes
        if not columns:
            # Logging
            self.logger.error("Columns list cannot be empty")
            return False
        
        # Vérification que les colonnes existent
        existing_columns = self._get_fact_table_columns()
        # Identification des colonnes manquantes
        missing_columns = [col for col in columns if col not in existing_columns]
        if missing_columns:
            # Logging
            self.logger.warning(f"Columns not found in fact table: {missing_columns}")
        
        return True
    
    # Méthodes principales de gestion des données
    # Méthode de création de la table des faits à partir d'un jeu de données
    def create_fact_table(self, df: pd.DataFrame) -> bool:
        """
        Create the fact table with initial data.
        
        Args:
            df: DataFrame containing initial data
            
        Returns:
            True if creation was successful
            
        Example:
            >>> success = data_mgr.create_fact_table(initial_df)
        """
        if not self.validate_operation('insert', df=df):
            return False
        
        try:
            # Enregistrement d'une table temporaire
            self.conn.register('temp_fact_creation', df)
            
            # Création de la table des faits
            self.conn.execute("""
                CREATE TABLE fact_table AS 
                SELECT * FROM temp_fact_creation
            """)
            
            # Suppression de la table temporaire
            self.conn.execute('DROP VIEW temp_fact_creation')
            
            # Logging
            self.logger.info(f"Created fact table with {len(df)} rows and {len(df.columns)} columns")
            return True
            
        except Exception as e:
            # Nettoyage en cas d'erreur
            try:
                self.conn.execute('DROP VIEW IF EXISTS temp_fact_creation')
            except:
                pass
            
            # Logging
            self.logger.error(f"Failed to create fact table: {e}")
            return False
    
    # Méthode d'insertion de données dans la table des faits
    def insert_data(self, df: pd.DataFrame, use_batch: bool = True) -> int:
        """
        Insert new data into the fact table.
        
        Args:
            df: DataFrame containing data to insert
            use_batch: Whether to use batch processing for large datasets
            
        Returns:
            Number of rows inserted
            
        Example:
            >>> rows_inserted = data_mgr.insert_data(new_data_df)
            >>> print(f"Inserted {rows_inserted} rows")
        """
        if not self.validate_operation('insert', df=df):
            return 0
        
        try:
            # Vérification de l'existence de la fact table
            if not self._table_exists('fact_table'):
                if self.create_fact_table(df):
                    return len(df)
                else:
                    return 0
            
            # Processing par lot si nécessaire
            if use_batch and len(df) > self.batch_size:
                return self._batch_insert_data(df)
            else:
                return self._direct_insert_data(df)
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to insert data: {e}")
            return 0
    
    # Méthode de mise à jour et d'insertion de données
    def upsert_data(self, df: pd.DataFrame, merge_keys: Optional[List[str]] = None, use_batch: bool = True) -> Tuple[int, int]:
        """
        Upsert data into the fact table (insert new, update existing).
        
        Args:
            df: DataFrame containing data to upsert
            merge_keys: Columns used to identify existing records (if None, uses all common columns)
            use_batch: Whether to use batch processing for large datasets
            
        Returns:
            Tuple of (rows_inserted, rows_updated)
            
        Example:
            >>> inserted, updated = data_mgr.upsert_data(df, merge_keys=['id', 'date'])
            >>> print(f"Inserted: {inserted}, Updated: {updated}")
        """
        # Détermination des clés d'appariement
        if merge_keys is None:
            existing_columns = self._get_fact_table_columns()
            merge_keys = list(set(df.columns).intersection(set(existing_columns)))
        
        # Validation de l'opération
        if not self.validate_operation('upsert', df=df, merge_keys=merge_keys):
            return 0, 0
        
        try:
            # Vérification de l'existence de la fact table
            if not self._table_exists('fact_table'):
                if self.create_fact_table(df):
                    return len(df), 0
                else:
                    return 0, 0
            
            # Processing par lot si nécessaire
            if use_batch and len(df) > self.batch_size:
                return self._batch_upsert_data(df, merge_keys)
            else:
                return self._direct_upsert_data(df, merge_keys)
            
        except Exception as e:
            self.logger.error(f"Failed to upsert data: {e}")
            return 0, 0
    
    # Méthode de suppression de lignes de la table de faits
    def delete_rows(self, filters: Optional[Union[str, List]]) -> int:
        """
        Delete rows from fact table based on filters.
        
        Args:
            filters: Filter conditions (string SQL condition or structured filters)
            
        Returns:
            Number of rows deleted
            
        Example:
            >>> # Using string filter
            >>> deleted = data_mgr.delete_rows("status = 'inactive'")
            >>> 
            >>> # Using structured filter
            >>> filters = [('status', '=', 'inactive'), ('date', '<', '2023-01-01')]
            >>> deleted = data_mgr.delete_rows(filters)
        """
        # Validation de l'opération
        if not self.validate_operation('delete', filters=filters):
            return 0
        
        try:
            # Comptage initial
            initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            # Construction de la clause WHERE
            where_clause = _build_where_clause(filters)
            
            if not where_clause:
                self.logger.warning("No valid filters provided for deletion")
                return 0
            
            # Construction et exécution de la requête de suppression
            delete_query = f"DELETE FROM fact_table {where_clause}"
            self.conn.execute(delete_query)
            
            # Comptage final
            final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            rows_deleted = initial_count - final_count
            
            # Logging
            self.logger.info(f"Successfully deleted {rows_deleted} rows from fact_table")
            return rows_deleted
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to delete rows: {e}")
            return 0
    
    # Méthodes de gestion des colonnes
    # Méthode d'ajout d'une colonne à la table des faits et à la table des méta-données
    def add_column(self, column_name: str, df: pd.DataFrame, default_value: Any = None) -> bool:
        """
        Add a new column to the fact table.
        
        Args:
            column_name: Name of the column to add
            df: DataFrame containing the new column data
            default_value: Default value for existing rows
            
        Returns:
            True if column was added successfully
            
        Example:
            >>> success = data_mgr.add_column('new_field', df_with_new_field)
        """
        # Validation de l'opération
        if not self.validate_operation('add_column', column_name=column_name, df=df):
            return False
        
        try:
            # Vérification que la colonne n'existe pas déjà
            if self._column_exists(column_name):
                # Logging
                self.logger.warning(f"Column {column_name} already exists")
                return False
            
            # Détermination du type SQL
            dtype = str(df[column_name].dtype)
            sql_type = map_python_to_sql_type(dtype)
            
            # Ajout de la colonne avec valeur par défaut
            if default_value is not None:
                alter_query = f"ALTER TABLE fact_table ADD COLUMN {column_name} {sql_type} DEFAULT ?"
                self.conn.execute(alter_query, [default_value])
            else:
                alter_query = f"ALTER TABLE fact_table ADD COLUMN {column_name} {sql_type} DEFAULT NULL"
                self.conn.execute(alter_query)
            
            # Ajout aux métadonnées
            self._add_column_to_metadata(column_name, df)
            
            # Logging
            self.logger.info(f"Added column {column_name} to fact table")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add column {column_name}: {e}")
            return False
    
    # Méthode de suppression de colonnes de la table des faits et de la table des méta-données
    def drop_columns(self, columns: List[str]) -> List[str]:
        """
        Drop columns from the fact table.
        
        Args:
            columns: List of column names to drop
            
        Returns:
            List of successfully dropped columns
            
        Example:
            >>> dropped = data_mgr.drop_columns(['old_col1', 'old_col2'])
            >>> print(f"Dropped columns: {dropped}")
        """
        # Validation de l'opération
        if not self.validate_operation('drop_column', columns=columns):
            return []
        
        # Identification des colonnes à supprimer
        successfully_dropped = []
        existing_columns = self._get_fact_table_columns()
        valid_columns = [col for col in columns if col in existing_columns]
        
        # Parcours des colonnes
        for column in valid_columns:
            try:
                # Suppression de la colonne
                alter_query = f"ALTER TABLE fact_table DROP COLUMN {column}"
                self.conn.execute(alter_query)
                
                # Suppression des métadonnées
                self.delete_column_metadata(column)
                
                successfully_dropped.append(column)
                # Logging
                self.logger.info(f"Dropped column {column} from fact table")
                
            except Exception as e:
                self.logger.error(f"Failed to drop column {column}: {e}")
        
        return successfully_dropped
    
    # Méthode de mise à jour du type d'une colonne
    def update_column_type(self, column_name: str, new_type: str) -> bool:
        """
        Update the data type of an existing column.
        
        Args:
            column_name: Name of the column
            new_type: New SQL data type
            
        Returns:
            True if type was updated successfully
            
        Example:
            >>> success = data_mgr.update_column_type('amount', 'DECIMAL(10,2)')
        """
        try:
            # Validation de l'opération
            if not self._column_exists(column_name):
                # Logging
                self.logger.error(f"Column {column_name} does not exist")
                return False
            
            # Mise à jour du type
            alter_query = f"ALTER TABLE fact_table ALTER {column_name} SET DATA TYPE {new_type}"
            self.conn.execute(alter_query)
            
            # Mise à jour des métadonnées si disponibles
            try:
                self.conn.execute(
                    "UPDATE metadata SET sql_type = ? WHERE name = ?",
                    [new_type, column_name]
                )
                self._invalidate_metadata_cache()
            except:
                pass  # Métadonnées non disponibles
            
            # Logging
            self.logger.info(f"Updated column {column_name} type to {new_type}")
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to update column type for {column_name}: {e}")
            return False
    
    # Méthodes privées pour le traitement par lots
    # Méthode auxiliaire d'insertion par batch
    def _batch_insert_data(self, df: pd.DataFrame) -> int:
        """Insert data in batches for large datasets."""
        # Initialisation du compteur d'insertions
        total_inserted = 0
        # Logging
        self.logger.info(f"Processing {len(df)} rows in batches of {self.batch_size}")
        # Parcours des batch
        for i in range(0, len(df), self.batch_size):
            # Détermination du début et de la fin du batch
            batch_start = i
            batch_end = min(i + self.batch_size, len(df))
            # Construction du jeu de données correspondant
            batch_df = df.iloc[batch_start:batch_end].copy()
            
            try:
                # Insertion des données dans la base
                inserted = self._direct_insert_data(batch_df)
                # Ajout au total
                total_inserted += inserted
                # Vérificaion que le nombre d'observations insérées correspond bien à la longueur du batch
                if inserted != len(batch_df):
                    # Logging
                    self.logger.warning(f"Batch {batch_start}-{batch_end}: Expected {len(batch_df)}, inserted {inserted}")
                    
            except Exception as e:
                # Logging
                self.logger.error(f"Error processing batch {batch_start}-{batch_end}: {e}")
        
        return total_inserted
    
    # Méthode d'insertion directe des données
    def _direct_insert_data(self, df: pd.DataFrame) -> int:
        """Insert data directly without batch processing."""
        try:
            # Préparation des colonnes manquantes
            self._ensure_columns_exist(df)
            
            # Enregistrement d'une vue temporaire
            self.conn.register('temp_insert', df)
            
            # Insertion des données
            column_list = ', '.join(df.columns)
            insert_query = f"""
                INSERT INTO fact_table ({column_list})
                SELECT {column_list} FROM temp_insert
            """
            self.conn.execute(insert_query)
            
            # Suppression de la vue temporaire
            self.conn.execute('DROP VIEW temp_insert')
            
            return len(df)
            
        except Exception as e:
            # Nettoyage en cas d'erreur
            try:
                self.conn.execute('DROP VIEW IF EXISTS temp_insert')
            except:
                pass
            raise e
    
    # Méthode de mise à jour et d'insertion des données en batch
    def _batch_upsert_data(self, df: pd.DataFrame, merge_keys: List[str]) -> Tuple[int, int]:
        """Upsert data in batches for large datasets."""
        # Initialisation des totaux
        total_inserted = 0
        total_updated = 0
        # Logging
        self.logger.info(f"Processing {len(df)} rows in batches of {self.batch_size}")
        # Parcours des batchs
        for i in range(0, len(df), self.batch_size):
            # Détermination du début et de la fin du batch
            batch_start = i
            batch_end = min(i + self.batch_size, len(df))
            # Circonscription du batch
            batch_df = df.iloc[batch_start:batch_end].copy()
            
            try:
                # Mise à jour du jeu de données
                inserted, updated = self._direct_upsert_data(batch_df, merge_keys)
                # Ajout aux totaux
                total_inserted += inserted
                total_updated += updated
                
            except Exception as e:
                # Logging
                self.logger.error(f"Error processing upsert batch {batch_start}-{batch_end}: {e}")
        
        return total_inserted, total_updated
    
    # Méthode de mise à jour et d'insertion des données directement
    def _direct_upsert_data(self, df: pd.DataFrame, merge_keys: List[str]) -> Tuple[int, int]:
        """Upsert data directly without batch processing."""
        try:
            # Préparation des colonnes manquantes
            self._ensure_columns_exist(df)
            
            # Enregistrement d'une vue temporaire
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
                existing_columns = self._get_fact_table_columns()
                update_columns = [col for col in df.columns if col in existing_columns and col not in merge_keys]
                
                if update_columns:
                    set_clause = ", ".join([f"f.{col} = t.{col}" for col in update_columns])
                    update_query = f"""
                        UPDATE fact_table f
                        SET {set_clause}
                        FROM temp_upsert t
                        WHERE {merge_condition}
                    """
                    self.conn.execute(update_query)
            
            # Insertion des nouvelles lignes
            initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            column_list = ', '.join(df.columns)
            insert_query = f"""
                INSERT INTO fact_table ({column_list})
                SELECT {column_list} FROM temp_upsert t
                WHERE NOT EXISTS (
                    SELECT 1 FROM fact_table f
                    WHERE {merge_condition}
                )
            """
            self.conn.execute(insert_query)
            
            # Comptage final
            final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            rows_inserted = final_count - initial_count
            
            # Suppression de la vue temporaire
            self.conn.execute('DROP VIEW temp_upsert')
            
            # Logging
            self.logger.info(f"Upsert completed: {rows_inserted} rows inserted, {rows_updated} rows updated")
            return rows_inserted, rows_updated
            
        except Exception as e:
            # Nettoyage en cas d'erreur
            try:
                self.conn.execute('DROP VIEW IF EXISTS temp_upsert')
            except:
                pass
            raise e
    
    # Méthode auxiliaire de vérification que toutes les colonnes d'un jeu de données existente dans la table des faits
    def _ensure_columns_exist(self, df: pd.DataFrame) -> None:
        """Ensure all DataFrame columns exist in the fact table."""
        # Identification des colonnes manquantes
        existing_columns = set(self._get_fact_table_columns())
        new_columns = set(df.columns) - existing_columns
        
        # Ajout des colonnes manquantes
        for column in new_columns:
            if not self.add_column(column, df):
                raise Exception(f"Failed to add required column: {column}")
    
    # Méthodes utilitaires
    # Méthode produisant des statistiques sur la table des faits
    def get_table_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the fact table.
        
        Returns:
            Dictionary containing table statistics
            
        Example:
            >>> stats = data_mgr.get_table_stats()
            >>> print(f"Rows: {stats['row_count']}, Columns: {stats['column_count']}")
        """
        try:
            stats = {}
            
            # Comptage des lignes
            stats['row_count'] = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            # Informations sur les colonnes
            columns_info = self.conn.execute("DESCRIBE fact_table").fetchall()
            stats['column_count'] = len(columns_info)
            stats['columns'] = [{'name': col[0], 'type': col[1]} for col in columns_info]
            
            # Taille approximative (en MB)
            try:
                size_query = "SELECT pg_size_pretty(pg_total_relation_size('fact_table'))"
                size_result = self.conn.execute(size_query).fetchone()
                stats['table_size'] = size_result[0] if size_result else 'Unknown'
            except:
                stats['table_size'] = 'Unknown'
            
            return stats
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to get table stats: {e}")
            return {'error': str(e)}
    
    # Méthode d'analyse de la table des faits pour optimiser les requêtes
    def optimize_table(self) -> bool:
        """
        Optimize the fact table by analyzing and potentially reorganizing data.
        
        Returns:
            True if optimization was successful
            
        Example:
            >>> success = data_mgr.optimize_table()
        """
        try:
            # Analyse de la table pour optimiser les requêtes
            self.conn.execute("ANALYZE fact_table")
            
            # Logging
            self.logger.info("Table optimization completed")
            return True
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to optimize table: {e}")
            return False