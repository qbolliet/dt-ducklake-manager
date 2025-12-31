# Importation des modules
# Modules de base
import os
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import threading
# DuckDB
import duckdb

# Import des utilitaires
from ...utils.logger import _init_logger
from ...utils.data_processing import map_python_to_sql_type

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe contennat des opérations utilitaires de base sur la base de données au schéma métadonnées - table des faits - tables de dimensions
class BaseSchemaManager(ABC):
    """
    Base class for database schema management operations.
    
    Provides common functionality for metadata management, column operations,
    and database introspection. All concrete managers should inherit from this class.
    
    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection
        categorical_threshold (int): Threshold for categorical determination
        logger: Logger instance for operation tracking
    """
    
    # Initialisation
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None):
        """
        Initialize the base schema manager.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Threshold for determining categorical variables
            log_filename: Path to log file
            
        Example:
            >>> conn = duckdb.connect('database.db')
            >>> manager = ConcreteManager(conn, categorical_threshold=30)
        """
        # Initialisation de la connexion à la base de données
        self.conn = connection
        
        # Seuil pour déterminer si une variable est catégorielle  
        self.categorical_threshold = categorical_threshold
        
        # Initialisation du logger pour traçabilité des opérations
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[3], "logs/base_schema_manager.log")
        self.logger = _init_logger(filename=log_filename)
        
        # Cache thread-safe pour optimiser les accès aux métadonnées
        self._metadata_cache = None
        self._cache_lock = threading.RLock()
    
    # Méthodes de gestion du cache des métadonnées
    # Méthode de chargement des méta-données
    def _load_current_metadata(self) -> pd.DataFrame:
        """
        Load current metadata from the database with thread-safe caching.
        
        Returns:
            DataFrame containing current metadata
        """
        with self._cache_lock:
            # Chargement de la table si elle n'est pas en cache
            if self._metadata_cache is None:
                try:
                    # Chargement
                    self._metadata_cache = self.conn.execute("SELECT * FROM metadata").fetchdf()
                except:
                    # Si la table n'existe pas encore, on l'initialise vide
                    self._metadata_cache = pd.DataFrame(columns=[
                        'name', 'label', 'python_type', 'sql_type', 'is_categorical', 'is_primary_key'
                    ])
            
            return self._metadata_cache.copy()
    
    # Méthode d'invalidation des méta-données mises en cache
    def _invalidate_metadata_cache(self) -> None:
        """
        Invalidate the metadata cache to force reload on next access.
        """
        with self._cache_lock:
            self._metadata_cache = None
    
    # Méthodes d'introspection de la base de données
    # Méthode d'extraction des colonnes de la table des faits
    # /!\ Doit être cohérent avec les colonnes dans meta-données[name]
    def _get_fact_table_columns(self) -> List[str]:
        """
        Get list of columns in fact table.
        
        Returns:
            List of column names
        """
        try:
            # Exécution de la requête
            result = self.conn.execute("DESCRIBE fact_table").fetchall()
            return [row[0] for row in result]
        except:
            return []
    
    # Méthode d'extraction des colonnes catégorielles
    def _get_categorical_columns(self) -> List[str]:
        """
        Get list of categorical columns from metadata.
        
        Returns:
            List of categorical column names
        """
        try:
            # Exécution de la requête
            result = self.conn.execute(
                "SELECT name FROM metadata WHERE is_categorical = true"
            ).fetchall()
            return [row[0] for row in result]
        except:
            return []
    
    # Méthode de vérification de l'existance d'une table dans la base de données
    def _table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.
        
        Args:
            table_name: Name of the table to check
            
        Returns:
            True if table exists
        """
        try:
            # Exécution de la requête
            result = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
                [table_name]
            ).fetchone()
            return result[0] > 0
        except:
            return False
    
    # Méthode de vérification de l'existence d'une colonne dans une table
    def _column_exists(self, column: str, table: str = "fact_table") -> bool:
        """
        Check if a column exists in specified table.
        
        Args:
            column: Column name
            table: Table name (defaults to fact_table)
            
        Returns:
            True if column exists
        """
        try:
            # Extraction des colonnes de la table
            columns = [row[0] for row in self.conn.execute(f"DESCRIBE {table}").fetchall()]
            return column in columns
        except:
            return False
    
    # Méthode de vérification si une colonne a une table de dimension (équivalent à vérifier si elle est catégorielle)
    def _is_dimension_column(self, column: str) -> bool:
        """
        Check if a column is a dimension (categorical).

        Args:
            column: Column name

        Returns:
            True if column is categorical
        """
        try:
            result = self.conn.execute(
                "SELECT is_categorical FROM metadata WHERE name = ?",
                [column]
            ).fetchone()
            return result[0] if result else False
        except:
            return False

    # Méthode de vérification si une colonne est marquée comme clé primaire
    def _is_primary_key_column(self, column: str) -> bool:
        """
        Check if a column is marked as primary key in metadata.

        Args:
            column: Column name to check

        Returns:
            True if column is a primary key

        Example:
            >>> manager._is_primary_key_column('user_id')
            True
        """
        try:
            result = self.conn.execute(
                "SELECT is_primary_key FROM metadata WHERE name = ?",
                [column]
            ).fetchone()
            return result[0] if result else False
        except:
            return False

    # Méthodes de gestion des métadonnées
    # Méthode d'ajout d'une colonne aux méta-données
    def _add_column_to_metadata(self, column: str, df: pd.DataFrame, label: Optional[str] = None) -> None:
        """
        Add a new column to metadata table.
        
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
                name VARCHAR PRIMARY KEY,
                label VARCHAR,
                python_type VARCHAR,
                sql_type VARCHAR,
                is_categorical BOOLEAN,
                is_primary_key BOOLEAN DEFAULT FALSE
            )
        """)
        
        # Insertion de la nouvelle colonne
        if label is None:
            label = column.replace('_', ' ').title()
        
        self.conn.execute("""
            INSERT INTO metadata (name, label, python_type, sql_type, is_categorical, is_primary_key)
            VALUES (?, ?, ?, ?, ?, FALSE)
            ON CONFLICT (name) DO UPDATE SET
                label = excluded.label,
                python_type = excluded.python_type,
                sql_type = excluded.sql_type,
                is_categorical = excluded.is_categorical
        """, [column, label, dtype, sql_type, is_categorical])
        
        # Invalidation du cache
        self._invalidate_metadata_cache()
        
        # Logging
        self.logger.info(f"Added/updated column {column} in metadata")
    
    # Méthode de mise à jour du statut catégoriel d'une donnée
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
        
        # Invalidation du cache
        self._invalidate_metadata_cache()
        
        # Logging
        self.logger.info(f"Updated categorical status for {col_name}: {is_categorical}")
    
    # Méthode de suppression des méta-données pour une colonne
    def delete_column_metadata(self, column_name: str) -> None:
        """
        Delete metadata for a specific column.
        
        Args:
            column_name: Name of the column
        """
        try:
            # Requête de suppression des méta-données
            delete_query = "DELETE FROM metadata WHERE name = ?"
            # Exécution de la requête
            self.conn.execute(delete_query, [column_name])
            
            # Invalidation du cache
            self._invalidate_metadata_cache()
            
            # Logging
            self.logger.info(f"Deleted metadata for column {column_name}")
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to delete metadata for column {column_name}: {e}")
            raise
    
    # Méthodes de résolution des conflits de types
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
        
        # Invalidation du cache
        self._invalidate_metadata_cache()
        
        # Logging
        self.logger.info(f"Type conflict resolution for {column}: {current_type} -> {resolved_type}")
    
    # Méthode utilitaire pour les colonnes contenant uniquement des valeurs nulles
    def _get_null_only_columns(self) -> List[str]:
        """
        Get list of columns that contain only null values in the fact table.
        
        Returns:
            List of column names that contain only null values
        """
        # Initialisation de la liste des colonnes vides
        null_only_columns = []
        
        try:
            # Récupération des colonnes de la fact table
            columns = self._get_fact_table_columns()
            # Parcours des données
            for column in columns:
                # Vérification si la colonne ne contient que des valeurs nulles
                query = f"SELECT COUNT(*) FROM fact_table WHERE {column} IS NOT NULL"
                non_null_count = self.conn.execute(query).fetchone()[0]
                # Ajout à la liste si ne contient que des colonnes nulles
                if non_null_count == 0:
                    null_only_columns.append(column)
            # Logging
            if null_only_columns:
                self.logger.info(f"Columns containing only null values detected: {null_only_columns}")
                
        except Exception as e:
            self.logger.error(f"An error occurred while detecting null values: {e}")
        
        return null_only_columns
    
    # Méthode de vérification du seuil catégoriel
    def _check_categorical_threshold(self, values: pd.Series, threshold: Optional[int] = None) -> bool:
        """
        Check if values meet categorical threshold criteria.
        
        Args:
            values: Series with column values
            threshold: Threshold to use (defaults to instance threshold)
            
        Returns:
            True if values should be categorical
        """
        threshold = threshold or self.categorical_threshold
        unique_count = len(set(values.dropna().unique()))
        return unique_count <= threshold
    
    @abstractmethod
    def validate_operation(self, operation_type: str, **kwargs) -> bool:
        """
        Abstract method to validate operations before execution.
        
        Args:
            operation_type: Type of operation to validate
            **kwargs: Operation-specific parameters
            
        Returns:
            True if operation is valid
        """
        pass