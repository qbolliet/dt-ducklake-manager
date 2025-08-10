# Module pour la suppression de données et colonnes dans les bases de données DuckDB
# Gestion des suppressions de lignes, colonnes, dimensions et index avec intégrité référentielle

# Imports des modules système
import os
from pathlib import Path
from typing import List, Optional, Dict, Union, Set, Tuple

# Imports des modules de base de données et data science
import duckdb
import pandas as pd
from enum import Enum
from dataclasses import dataclass

# Import du module de logging personnalisé
from ..utils.logger import _init_logger

# Emplacement du fichier pour la gestion des logs
FILE_PATH = Path(os.path.abspath(__file__))

# Classe de données pour représenter les dépendances entre colonnes
@dataclass
class ColumnDependency:
    """
    Represents a column dependency in the database schema.
    
    Attributes:
        column_name (str): Name of the dependent column
        depends_on (List[str]): List of entities this column depends on
        dependency_type (str): Type of dependency ('foreign_key', 'computed', 'index')
        cascade_action (str): Action to take when dependency is removed
    """
    column_name: str
    depends_on: List[str]
    dependency_type: str  # 'foreign_key', 'computed', 'index'
    cascade_action: str

# Classe principale pour la gestion des suppressions dans la base de données
class DatabaseDeleter:
    """
    A class to handle deletion operations on DuckDB databases.
    
    Supports deleting rows based on filters and removing columns
    while maintaining consistency across fact, dimension, and metadata tables.
    Handles categorical threshold management and index maintenance.
    
    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection
        categorical_threshold (int): Threshold for categorical determination
        logger: Logger instance for operation tracking
    """
    
    # Méthode d'initialisation de la classe DatabaseDeleter
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None):
        """
        Initialize the DatabaseDeleter.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Threshold for determining categorical variables
            log_filename: Path to log file
        
        Example:
            >>> conn = duckdb.connect('database.db')
            >>> deleter = DatabaseDeleter(conn, categorical_threshold=30)
        """
        # Initialisation de la connexion à la base de données
        self.conn = connection
        
        # Seuil pour déterminer si une variable est catégorielle
        self.categorical_threshold = categorical_threshold
        
        # Initialisation du logger pour traçabilité des opérations
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/database_deleter.log")
        self.logger = _init_logger(filename=log_filename)
        
        # Cache pour optimiser les analyses de dépendances
        self._dependency_cache = None
        self._foreign_key_cache = None
        self._indexes_cache = None
    
    # Méthode d'analyse des dépendances entre colonnes
    def analyze_column_dependencies(self, columns: List[str]) -> Dict[str, List[ColumnDependency]]:
        """
        Analyze dependencies for specified columns.
        
        Args:
            columns: List of column names to analyze
            
        Returns:
            Dictionary mapping column names to their dependencies
            
        Example:
            >>> deleter = DatabaseDeleter(connection)
            >>> deps = deleter.analyze_column_dependencies(['customer_id', 'product_id'])
            >>> print(deps['customer_id'])
        """
        self.logger.info(f"Analyse des dépendances pour les colonnes: {columns}")
        
        dependencies = {}
        
        for column in columns:
            column_deps = []
            
            # Analyse des clés étrangères
            foreign_key_deps = self._analyze_foreign_key_dependencies(column)
            column_deps.extend(foreign_key_deps)
            
            # Analyse des index dépendants
            index_deps = self._analyze_index_dependencies(column)
            column_deps.extend(index_deps)
            
            dependencies[column] = column_deps
            
        return dependencies
    
    # Méthode de vérification si une colonne doit rester catégorielle selon le seuil
    def _should_remain_categorical(self, column: str) -> bool:
        """
        Check if a column should remain categorical based on threshold.
        
        Args:
            column: Column name to check
            
        Returns:
            bool: True if column should remain categorical
            
        Example:
            >>> deleter = DatabaseDeleter(conn, categorical_threshold=50)
            >>> deleter._should_remain_categorical('status')
            True
        """
        try:
            # Comptage des modalités distinctes dans la table de faits
            unique_count = self.conn.execute(f"""
                SELECT COUNT(DISTINCT {column}) 
                FROM fact_table 
                WHERE {column} IS NOT NULL
            """).fetchone()[0]
            
            # Vérification du seuil catégoriel
            should_remain = unique_count <= self.categorical_threshold
            
            self.logger.info(f"Colonne {column}: {unique_count} modalités, seuil: {self.categorical_threshold}, reste catégorielle: {should_remain}")
            
            return should_remain
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification du seuil catégoriel pour {column}: {e}")
            return False
    
    # Méthode de conversion inverse des index vers les labels
    def _reverse_index_to_label_transformation(self, column: str) -> None:
        """
        Reverse the index-to-label transformation for a column.
        
        Args:
            column: Column name to reverse transformation for
            
        Example:
            >>> deleter._reverse_index_to_label_transformation('category')
        """
        try:
            dim_table = f"dim_{column}"
            
            # Vérification de l'existence de la table de dimension
            if not self._table_exists(dim_table):
                self.logger.warning(f"Table de dimension {dim_table} n'existe pas pour la transformation inverse")
                return
            
            # Construction du dictionnaire de passage inverse (value -> label)
            # Vérification de la structure de la table de dimension
            dim_columns = self.conn.execute(f"""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = '{dim_table}'
            """).fetchall()
            
            has_value_col = any('value' in col[0].lower() for col in dim_columns)
            has_label_col = any('label' in col[0].lower() for col in dim_columns)
            
            if has_value_col and has_label_col:
                mapping_result = self.conn.execute(f"""
                    SELECT value, label FROM {dim_table}
                """).fetchall()
            else:
                # Structure alternative possible
                self.logger.warning(f"Structure de dimension non standard pour {dim_table}")
                return
            
            if not mapping_result:
                self.logger.warning(f"Aucun mapping trouvé dans {dim_table}")
                return
            
            # Application de la transformation inverse
            for value, label in mapping_result:
                self.conn.execute(f"""
                    UPDATE fact_table 
                    SET {column} = ? 
                    WHERE {column} = ?
                """, [label, value])
            
            self.logger.info(f"Transformation inverse appliquée pour la colonne {column}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la transformation inverse pour {column}: {e}")
            raise
    
    # Méthode de mise à jour de la colonne is_categorical dans la table metadata
    def _update_metadata_categorical_status(self, column: str, is_categorical: bool) -> None:
        """
        Update the is_categorical status in metadata table.
        
        Args:
            column: Column name
            is_categorical: New categorical status
            
        Example:
            >>> deleter._update_metadata_categorical_status('status', False)
        """
        try:
            self.conn.execute("""
                UPDATE metadata 
                SET is_categorical = ? 
                WHERE name = ?
            """, [is_categorical, column])
            
            self.logger.info(f"Statut catégoriel mis à jour pour {column}: {is_categorical}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la mise à jour du statut catégoriel pour {column}: {e}")
            raise
    
    # Méthode de suppression rigoureuse d'une table de dimension
    def _rigorous_dimension_table_removal(self, column: str) -> None:
        """
        Rigorously remove a dimension table and handle all implications.
        
        Args:
            column: Column name whose dimension table to remove
            
        Example:
            >>> deleter._rigorous_dimension_table_removal('category')
        """
        try:
            # Vérification si la colonne doit rester catégorielle
            if self._should_remain_categorical(column):
                self.logger.info(f"La colonne {column} doit rester catégorielle selon le seuil")
                return
            
            # Transformation inverse index -> label dans la table de faits
            self._reverse_index_to_label_transformation(column)
            
            # Suppression de la table de dimension
            self.delete_dimension_table(column)
            
            # Mise à jour du statut catégoriel dans les métadonnées
            self._update_metadata_categorical_status(column, False)
            
            # Suppression des index liés à cette dimension
            self._drop_column_indexes(column)
            
            self.logger.info(f"Suppression rigoureuse terminée pour la dimension {column}")
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression rigoureuse de la dimension {column}: {e}")
            raise
    
    # Méthode d'analyse des dépendances de clés étrangères
    def _analyze_foreign_key_dependencies(self, column: str) -> List[ColumnDependency]:
        """
        Analyse les dépendances de clés étrangères pour une colonne.
        
        Args:
            column: Nom de la colonne
            
        Returns:
            Liste des dépendances de clé étrangère
        """
        dependencies = []
        
        try:
            # Vérification si la colonne est une clé étrangère vers une dimension
            if self._is_dimension_column(column):
                dim_table = f"dim_{column}"
                
                # Vérification de l'existence de la table de dimension
                if self._table_exists(dim_table):
                    dependency = ColumnDependency(
                        column_name=column,
                        depends_on=[dim_table],
                        dependency_type='foreign_key',
                        cascade_action='restrict'
                    )
                    dependencies.append(dependency)
            
            # Recherche d'autres références dans les métadonnées
            # (à étendre selon la structure spécifique)
            
        except Exception as e:
            self.logger.warning(f"Erreur lors de l'analyse des clés étrangères pour {column}: {e}")
        
        return dependencies
    
    # Méthode de suppression des index liés à une colonne
    def _drop_column_indexes(self, column: str) -> List[str]:
        """
        Drop all indexes that involve the specified column.
        
        Args:
            column: Column name
            
        Returns:
            List of dropped index names
            
        Example:
            >>> dropped = deleter._drop_column_indexes('category')
            >>> print(f"Dropped indexes: {dropped}")
        """
        dropped_indexes = []
        
        try:
            # Recherche des index utilisant cette colonne
            index_query = """
                SELECT index_name, expressions 
                FROM duckdb_indexes() 
                WHERE expressions LIKE ?
            """
            
            indexes = self.conn.execute(index_query, [f'%{column}%']).fetchall()
            
            for index_name, expressions in indexes:
                try:
                    # Suppression de l'index
                    drop_query = f"DROP INDEX IF EXISTS {index_name}"
                    self.conn.execute(drop_query)
                    dropped_indexes.append(index_name)
                    
                    self.logger.info(f"Index supprimé: {index_name} (expressions: {expressions})")
                    
                except Exception as e:
                    self.logger.error(f"Erreur lors de la suppression de l'index {index_name}: {e}")
            
            return dropped_indexes
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la recherche des index pour {column}: {e}")
            return dropped_indexes
    
    # Méthode de nettoyage des index orphelins
    def _cleanup_orphaned_indexes(self) -> List[str]:
        """
        Clean up indexes that reference non-existent columns.
        
        Returns:
            List of cleaned up index names
            
        Example:
            >>> cleaned = deleter._cleanup_orphaned_indexes()
        """
        cleaned_indexes = []
        
        try:
            # Récupération des colonnes existantes dans fact_table
            existing_columns = set(self._get_fact_table_columns())
            
            # Récupération de tous les index
            all_indexes = self.conn.execute("""
                SELECT index_name, expressions 
                FROM duckdb_indexes()
            """).fetchall()
            
            for index_name, expressions in all_indexes:
                # Vérification si l'index référence des colonnes qui n'existent plus
                should_drop = False
                
                # Vérification si l'index référence des colonnes supprimées
                # Analyse des colonnes potentiellement référencées dans l'expression
                referenced_columns = []
                for col in existing_columns:
                    if col in expressions or f"fact_table.{col}" in expressions:
                        referenced_columns.append(col)
                
                # Si l'index ne référence aucune colonne existante, il est orphelin
                if not referenced_columns and "fact_table" in expressions:
                    should_drop = True
                
                if should_drop:
                    try:
                        self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                        cleaned_indexes.append(index_name)
                        self.logger.info(f"Index orphelin supprimé: {index_name}")
                    except Exception as e:
                        self.logger.error(f"Erreur lors de la suppression de l'index orphelin {index_name}: {e}")
            
            return cleaned_indexes
            
        except Exception as e:
            self.logger.error(f"Erreur lors du nettoyage des index orphelins: {e}")
            return cleaned_indexes
    
    # Méthode d'analyse des dépendances d'index
    def _analyze_index_dependencies(self, column: str) -> List[ColumnDependency]:
        """
        Analyse les dépendances d'index pour une colonne.
        
        Args:
            column: Nom de la colonne
            
        Returns:
            Liste des dépendances d'index
        """
        dependencies = []
        
        try:
            # Recherche des index utilisant cette colonne
            index_query = """
                SELECT index_name, expressions FROM duckdb_indexes()
                WHERE expressions LIKE ?
            """
            
            indexes = self.conn.execute(index_query, [f'%{column}%']).fetchall()
            
            for index_name, expressions in indexes:
                dependency = ColumnDependency(
                    column_name=column,
                    depends_on=[index_name],
                    dependency_type='index',
                    cascade_action='drop'
                )
                dependencies.append(dependency)
                
        except Exception as e:
            self.logger.warning(f"Erreur lors de l'analyse des index pour {column}: {e}")
        
        return dependencies
    
    # Méthode de vérification de l'existence d'une colonne
    def _column_exists(self, column: str) -> bool:
        """
        Vérifie si une colonne existe dans fact_table.
        
        Args:
            column: Nom de la colonne
            
        Returns:
            True si la colonne existe
        """
        existing_columns = self._get_fact_table_columns()
        return column in existing_columns
    
    # Méthode de vérification de l'intégrité référentielle
    def _verify_referential_integrity(self) -> bool:
        """
        Vérifie l'intégrité référentielle après modifications.
        
        Returns:
            True si l'intégrité est respectée
        """
        try:
            # Vérification des références orphelines
            categorical_columns = self._get_categorical_columns()
            
            for column in categorical_columns:
                dim_table = f"dim_{column}"
                
                if self._table_exists(dim_table) and self._column_exists(column):
                    orphaned = self.conn.execute(f"""
                        SELECT COUNT(*) FROM fact_table
                        WHERE {column} IS NOT NULL
                        AND {column} NOT IN (SELECT value FROM {dim_table})
                    """).fetchone()[0]
                    
                    if orphaned > 0:
                        self.logger.warning(f"Références orphelines détectées pour {column}: {orphaned}")
                        return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification de l'intégrité: {e}")
            return False

    # Méthode de suppression de lignes dans la table de faits
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
            
            # Nettoyage des index orphelins
            self._cleanup_orphaned_indexes()
            
            return rows_deleted
            
        except Exception as e:
            self.logger.error(f"Failed to delete rows: {e}")
            raise
    
    # Méthode de suppression de colonnes avec gestion rigoureuse des dépendances
    def delete_columns(self, columns: List[str], use_transaction: bool = True) -> Dict:
        """
        Delete columns from fact table and update related tables.
        
        Args:
            columns: List of column names to delete
            use_transaction: Whether to use database transaction for atomicity
            
        Returns:
            Dictionary with deletion statistics
            
        Example:
            >>> stats = deleter.delete_columns(['old_column1', 'old_column2'])
            >>> print(f"Deleted columns: {stats['columns_deleted']}")
        """
        stats = {
            'columns_deleted': [],
            'dimensions_removed': [],
            'metadata_updated': 0,
            'errors': []
        }
        
        # Vérification des colonnes existantes
        existing_columns = self._get_fact_table_columns()
        
        # Utilisation de transaction pour l'atomicité si demandée
        if use_transaction:
            try:
                self.conn.execute("BEGIN TRANSACTION")
                self.logger.info("Transaction commencée pour la suppression des colonnes")
            except Exception as e:
                self.logger.warning(f"Impossible de commencer une transaction: {e}")
                use_transaction = False
        
        try:
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
                
                # Si c'est une dimension, suppression rigoureuse
                if is_dimension:
                    self._rigorous_dimension_table_removal(column)
                    stats['dimensions_removed'].append(column)
                else:
                    # Suppression des index pour les colonnes non-catégorielles
                    dropped_indexes = self._drop_column_indexes(column)
                    if dropped_indexes:
                        self.logger.info(f"Index supprimés pour {column}: {dropped_indexes}")
                
                # Suppression des métadonnées
                self.delete_column_metadata(column)
                stats['metadata_updated'] += 1
                
                self.logger.info(f"Successfully deleted column {column}")
                
            except Exception as e:
                self.logger.error(f"Failed to delete column {column}: {e}")
                stats['errors'].append(f"Failed to delete {column}: {str(e)}")
                
                # En cas d'erreur, continuer avec les autres colonnes
                # Le rollback sera géré à la fin si nécessaire
        
            # Nettoyage final des index orphelins
            cleaned_indexes = self._cleanup_orphaned_indexes()
            if cleaned_indexes:
                stats['indexes_cleaned'] = cleaned_indexes
            
            # Commit de la transaction si tout s'est bien passé
            if use_transaction:
                if not stats['errors']:
                    self.conn.execute("COMMIT")
                    self.logger.info("Transaction commitée avec succès")
                else:
                    self.conn.execute("ROLLBACK")
                    self.logger.warning("Transaction annulée à cause d'erreurs")
                    
        except Exception as e:
            # Gestion des erreurs globales
            self.logger.error(f"Erreur globale lors de la suppression des colonnes: {e}")
            stats['errors'].append(f"Global error: {str(e)}")
            
            if use_transaction:
                try:
                    self.conn.execute("ROLLBACK")
                    self.logger.info("Transaction annulée à cause d'une erreur globale")
                except Exception:
                    pass
            
        return stats
    
    # Méthode de suppression d'une table de dimension
    def delete_dimension_table(self, dimension_name: str) -> None:
        """
        Delete a dimension table.
        
        Args:
            dimension_name: Name of the dimension
        """
        table_name = f"dim_{dimension_name}"
        
        try:
            # Vérification de l'existence de la table
            # Vérification sécurisée de l'existence de la table
            table_exists = self.conn.execute("""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = ?
            """, [table_name]).fetchone()[0] > 0
            
            if table_exists:
                drop_query = f"DROP TABLE IF EXISTS {table_name}"
                self.conn.execute(drop_query)
                self.logger.info(f"Deleted dimension table {table_name}")
            else:
                self.logger.warning(f"Dimension table {table_name} does not exist")
                
        except Exception as e:
            self.logger.error(f"Failed to delete dimension table {table_name}: {e}")
            raise
    
    # Méthode de suppression des métadonnées d'une colonne
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
    
    # Méthode de nettoyage des dimensions orphelines
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
    
    # Méthode de troncature de la table de faits
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
    
    # Méthode de construction des clauses WHERE
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
    
    # Méthode d'obtention des colonnes de la table de faits
    def _get_fact_table_columns(self) -> List[str]:
        """Get list of columns in fact table."""
        result = self.conn.execute("DESCRIBE fact_table").fetchall()
        return [row[0] for row in result]
    
    # Méthode d'obtention des colonnes catégorielles
    def _get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns from metadata."""
        result = self.conn.execute(
            "SELECT name FROM metadata WHERE is_categorical = true"
        ).fetchall()
        return [row[0] for row in result]
    
    # Méthode de vérification si une colonne est une dimension
    def _is_dimension_column(self, column: str) -> bool:
        """Check if a column is a dimension (categorical)."""
        result = self.conn.execute(
            "SELECT is_categorical FROM metadata WHERE name = ?",
            [column]
        ).fetchone()
        return result[0] if result else False
    
    # Méthode de vérification de l'existence d'une table
    def _table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database."""
        result = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_name = ?",
            [table_name]
        ).fetchone()
        return result[0] > 0
    
    # Méthode de validation et prévisualisation des suppressions
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
    
    # Méthode de vérification et mise à jour de la cohérence des métadonnées
    def _ensure_metadata_consistency(self) -> Dict[str, int]:
        """
        Ensure metadata table is consistent with actual database state.
        
        Returns:
            Dictionary with consistency check results
            
        Example:
            >>> results = deleter._ensure_metadata_consistency()
            >>> print(f"Corrections applied: {results}")
        """
        results = {
            'metadata_corrections': 0,
            'orphaned_metadata_removed': 0,
            'categorical_status_updated': 0
        }
        
        try:
            # Récupération des colonnes existantes
            existing_columns = set(self._get_fact_table_columns())
            
            # Récupération des métadonnées existantes
            metadata_result = self.conn.execute("""
                SELECT name, is_categorical FROM metadata
            """).fetchall()
            
            metadata_columns = set()
            
            for name, is_categorical in metadata_result:
                metadata_columns.add(name)
                
                # Vérification si la colonne existe toujours
                if name not in existing_columns:
                    # Suppression des métadonnées orphelines
                    self.delete_column_metadata(name)
                    results['orphaned_metadata_removed'] += 1
                    self.logger.info(f"Métadonnées orphelines supprimées pour {name}")
                    continue
                
                # Vérification de la cohérence du statut catégoriel
                should_be_categorical = self._should_remain_categorical(name)
                if bool(is_categorical) != should_be_categorical:
                    self._update_metadata_categorical_status(name, should_be_categorical)
                    results['categorical_status_updated'] += 1
                    self.logger.info(f"Statut catégoriel corrigé pour {name}: {should_be_categorical}")
            
            # Ajout des métadonnées manquantes pour les nouvelles colonnes
            missing_columns = existing_columns - metadata_columns
            for column in missing_columns:
                # Inférence du type SQL basique
                column_type = self.conn.execute(f"""
                    SELECT data_type FROM information_schema.columns 
                    WHERE table_name = 'fact_table' AND column_name = ?
                """, [column]).fetchone()
                
                if column_type:
                    sql_type = column_type[0]
                    is_categorical = sql_type == 'VARCHAR' and self._should_remain_categorical(column)
                    
                    # Insertion des métadonnées manquantes
                    self.conn.execute("""
                        INSERT INTO metadata (name, label, python_type, sql_type, is_categorical)
                        VALUES (?, ?, ?, ?, ?)
                    """, [
                        column,
                        column.replace('_', ' ').title(),
                        'object' if sql_type == 'VARCHAR' else 'numeric',
                        sql_type,
                        is_categorical
                    ])
                    
                    results['metadata_corrections'] += 1
                    self.logger.info(f"Métadonnées ajoutées pour la nouvelle colonne {column}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification de cohérence des métadonnées: {e}")
            return results
    
    # Méthode de maintenance complète de la base de données
    def perform_database_maintenance(self) -> Dict[str, Union[int, List[str]]]:
        """
        Perform comprehensive database maintenance operations.
        
        Returns:
            Dictionary with maintenance operation results
            
        Example:
            >>> maintenance_results = deleter.perform_database_maintenance()
            >>> print(f"Maintenance completed: {maintenance_results}")
        """
        maintenance_results = {}
        
        try:
            self.logger.info("Début de la maintenance complète de la base de données")
            
            # Vérification de l'intégrité référentielle
            integrity_ok = self._verify_referential_integrity()
            maintenance_results['referential_integrity_ok'] = integrity_ok
            
            # Nettoyage des dimensions orphelines
            cleaned_dimensions = self.clean_orphaned_dimensions()
            maintenance_results['cleaned_dimensions'] = cleaned_dimensions
            
            # Nettoyage des index orphelins
            cleaned_indexes = self._cleanup_orphaned_indexes()
            maintenance_results['cleaned_indexes'] = cleaned_indexes
            
            # Vérification et correction des métadonnées
            metadata_corrections = self._ensure_metadata_consistency()
            maintenance_results.update(metadata_corrections)
            
            self.logger.info(f"Maintenance terminée avec succès: {maintenance_results}")
            
            return maintenance_results
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la maintenance: {e}")
            maintenance_results['error'] = str(e)
            return maintenance_results