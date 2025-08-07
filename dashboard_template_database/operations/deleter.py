# Module pour la suppression de données et colonnes
import os
from pathlib import Path
from typing import List, Optional, Dict, Union
import duckdb
import pandas as pd

from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

class DatabaseDeleter:
    """
    A class to handle deletion operations on DuckDB databases.
    
    Supports deleting rows based on filters and removing columns
    while maintaining consistency across fact, dimension, and metadata tables.
    """
    
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 log_filename: Optional[os.PathLike] = None):
        """
        Initialize the DatabaseDeleter.
        
        Args:
            connection: DuckDB connection object
            log_filename: Path to log file
        """
        self.conn = connection
        
        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/database_deleter.log")
        self.logger = _init_logger(filename=log_filename)
    
    def delete_rows(self, 
                   filters: Optional[str] = None,
                   structured_filters: Optional[List[Dict]] = None) -> int:
        """
        Delete rows from fact table based on filters.
        
        Args:
            filters: SQL WHERE clause as string (e.g., "column1 > 10 AND column2 = 'value'")
            structured_filters: List of filter dictionaries with keys: column, operator, value
            
        Returns:
            Number of rows deleted
        """
        # Comptage initial
        initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        
        # Construction de la clause WHERE
        where_clause = self._build_where_clause(filters, structured_filters)
        
        if not where_clause:
            self.logger.warning("No filters provided for deletion - operation cancelled")
            return 0
        
        # Suppression avec vérification
        delete_query = f"DELETE FROM fact_table {where_clause}"
        
        # Log de la requête pour audit
        self.logger.info(f"Executing deletion query: {delete_query[:200]}...")
        
        try:
            self.conn.execute(delete_query)
            
            # Comptage final
            final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            rows_deleted = initial_count - final_count
            
            self.logger.info(f"Successfully deleted {rows_deleted} rows from fact_table")
            
            # Nettoyage des dimensions orphelines
            self.clean_orphaned_dimensions()
            
            return rows_deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete rows: {e}")
            raise
    
    def delete_columns(self, columns: List[str]) -> Dict:
        """
        Delete columns from fact table and update related tables.
        
        Args:
            columns: List of column names to delete
            
        Returns:
            Dictionary with deletion statistics
        """
        stats = {
            'columns_deleted': [],
            'dimensions_removed': [],
            'metadata_updated': 0,
            'errors': []
        }
        
        # Vérification des colonnes existantes
        existing_columns = self._get_fact_table_columns()
        
        for column in columns:
            if column not in existing_columns:
                self.logger.warning(f"Column {column} does not exist in fact_table")
                stats['errors'].append(f"Column {column} not found")
                continue
            
            try:
                # Vérification si la colonne est une dimension
                is_dimension = self._is_dimension_column(column)
                
                # Suppression de la colonne de la fact table
                alter_query = f"ALTER TABLE fact_table DROP COLUMN {column}"
                self.conn.execute(alter_query)
                stats['columns_deleted'].append(column)
                
                # Si c'est une dimension, supprimer la table de dimension
                if is_dimension:
                    self.delete_dimension_table(column)
                    stats['dimensions_removed'].append(column)
                
                # Suppression des métadonnées
                self.delete_column_metadata(column)
                stats['metadata_updated'] += 1
                
                self.logger.info(f"Successfully deleted column {column}")
                
            except Exception as e:
                self.logger.error(f"Failed to delete column {column}: {e}")
                stats['errors'].append(f"Failed to delete {column}: {str(e)}")
        
        return stats
    
    def delete_dimension_table(self, dimension_name: str) -> None:
        """
        Delete a dimension table.
        
        Args:
            dimension_name: Name of the dimension
        """
        table_name = f"dim_{dimension_name}"
        
        try:
            # Vérification de l'existence de la table
            table_exists = self.conn.execute(
                f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()[0] > 0
            
            if table_exists:
                drop_query = f"DROP TABLE IF EXISTS {table_name}"
                self.conn.execute(drop_query)
                self.logger.info(f"Deleted dimension table {table_name}")
            else:
                self.logger.warning(f"Dimension table {table_name} does not exist")
                
        except Exception as e:
            self.logger.error(f"Failed to delete dimension table {table_name}: {e}")
            raise
    
    def delete_column_metadata(self, column_name: str) -> None:
        """
        Delete metadata for a specific column.
        
        Args:
            column_name: Name of the column
        """
        try:
            delete_query = "DELETE FROM metadata WHERE name = ?"
            self.conn.execute(delete_query, [column_name])
            self.logger.info(f"Deleted metadata for column {column_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete metadata for column {column_name}: {e}")
            raise
    
    def clean_orphaned_dimensions(self) -> Dict:
        """
        Remove dimension values that are no longer referenced in fact table.
        
        Returns:
            Dictionary mapping dimension names to number of values removed
        """
        cleaned = {}
        
        # Récupération des colonnes catégorielles
        categorical_columns = self._get_categorical_columns()
        
        for column in categorical_columns:
            table_name = f"dim_{column}"
            
            try:
                # Vérification de l'existence de la table de dimension
                if not self._table_exists(table_name):
                    continue
                
                # Comptage initial
                initial_count = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]
                
                # Suppression des valeurs non référencées
                delete_query = f"""
                    DELETE FROM {table_name}
                    WHERE value NOT IN (
                        SELECT DISTINCT {column}
                        FROM fact_table
                        WHERE {column} IS NOT NULL
                    )
                """
                self.conn.execute(delete_query)
                
                # Comptage final
                final_count = self.conn.execute(
                    f"SELECT COUNT(*) FROM {table_name}"
                ).fetchone()[0]
                
                values_removed = initial_count - final_count
                if values_removed > 0:
                    cleaned[column] = values_removed
                    self.logger.info(f"Removed {values_removed} orphaned values from {table_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to clean dimension {table_name}: {e}")
        
        return cleaned
    
    def truncate_fact_table(self, cascade: bool = False) -> None:
        """
        Truncate (empty) the fact table.
        
        Args:
            cascade: If True, also truncate dimension tables
        """
        try:
            # Truncate fact table
            self.conn.execute("TRUNCATE TABLE fact_table")
            self.logger.info("Truncated fact_table")
            
            if cascade:
                # Truncate toutes les tables de dimension
                categorical_columns = self._get_categorical_columns()
                for column in categorical_columns:
                    table_name = f"dim_{column}"
                    if self._table_exists(table_name):
                        self.conn.execute(f"TRUNCATE TABLE {table_name}")
                        self.logger.info(f"Truncated {table_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to truncate tables: {e}")
            raise
    
    def _build_where_clause(self, 
                           filters: Optional[str] = None,
                           structured_filters: Optional[List[Dict]] = None) -> str:
        """
        Build WHERE clause from filters.
        
        Args:
            filters: Raw SQL filter string
            structured_filters: List of filter dictionaries
            
        Returns:
            WHERE clause string
        """
        clauses = []
        
        if filters:
            clauses.append(f"({filters})")
        
        if structured_filters:
            for f in structured_filters:
                column = f.get('column')
                operator = f.get('operator', '=')
                value = f.get('value')
                
                if column and value is not None:
                    # Gestion des différents types de valeurs
                    if isinstance(value, str):
                        value_str = f"'{value}'"
                    elif isinstance(value, (list, tuple)):
                        value_str = f"({','.join([f"'{v}'" if isinstance(v, str) else str(v) for v in value])})"
                        operator = 'IN' if operator == '=' else operator
                    else:
                        value_str = str(value)
                    
                    clauses.append(f"{column} {operator} {value_str}")
        
        if clauses:
            return "WHERE " + " AND ".join(clauses)
        return ""
    
    def _get_fact_table_columns(self) -> List[str]:
        """Get list of columns in fact table."""
        result = self.conn.execute("DESCRIBE fact_table").fetchall()
        return [row[0] for row in result]
    
    def _get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns from metadata."""
        result = self.conn.execute(
            "SELECT name FROM metadata WHERE is_categorical = true"
        ).fetchall()
        return [row[0] for row in result]
    
    def _is_dimension_column(self, column: str) -> bool:
        """Check if a column is a dimension (categorical)."""
        result = self.conn.execute(
            "SELECT is_categorical FROM metadata WHERE name = ?",
            [column]
        ).fetchone()
        return result[0] if result else False
    
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name]
        ).fetchone()
        return result[0] > 0
    
    def validate_deletion(self, 
                         filters: Optional[str] = None,
                         structured_filters: Optional[List[Dict]] = None) -> Dict:
        """
        Preview what would be deleted without actually deleting.
        
        Args:
            filters: SQL WHERE clause
            structured_filters: List of filter dictionaries
            
        Returns:
            Dictionary with deletion preview information
        """
        where_clause = self._build_where_clause(filters, structured_filters)
        
        if not where_clause:
            return {'error': 'No filters provided', 'would_delete': 0}
        
        # Comptage des lignes qui seraient supprimées
        count_query = f"SELECT COUNT(*) FROM fact_table {where_clause}"
        
        try:
            would_delete = self.conn.execute(count_query).fetchone()[0]
            total_rows = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            # Échantillon des lignes qui seraient supprimées
            sample_query = f"SELECT * FROM fact_table {where_clause} LIMIT 5"
            sample = self.conn.execute(sample_query).fetchdf()
            
            return {
                'would_delete': would_delete,
                'total_rows': total_rows,
                'percentage': (would_delete / total_rows * 100) if total_rows > 0 else 0,
                'sample': sample.to_dict('records') if not sample.empty else []
            }
            
        except Exception as e:
            return {'error': str(e), 'would_delete': 0}