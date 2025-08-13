# Importation des modules
# Modules de base
import os
from pathlib import Path
import numpy as np
from typing import List, Optional, Dict, Union, Set, Tuple
# Imports des modules de base de données et data science
import duckdb
from dataclasses import dataclass

# Import du module de logging personnalisé
from ..utils.logger import _init_logger
# Importation des utilitaires
from ..utils.data_processing import _build_where_clause

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
    
    # Initialisation
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
        # Logging
        self.logger.info(f"Analyse des dépendances pour les colonnes: {columns}")
        # Initialisation du dictionnaire des dépendances entre colonnes
        dependencies = {}
        # Parcours des colonnes
        for column in columns:
            # Initialisation de la liste des dépendances de la colonne
            column_deps = []
            
            # Analyse des clés étrangères
            foreign_key_deps = self._analyze_foreign_key_dependencies(column)
            column_deps.extend(foreign_key_deps)
            
            # Analyse des index dépendants
            index_deps = self._analyze_index_dependencies(column)
            column_deps.extend(index_deps)
            # Ajout au dictionnaire résultat
            dependencies[column] = column_deps
            
        return dependencies
    
    # Méthode d'analyse des dépendances de clés étrangères
    def _analyze_foreign_key_dependencies(self, column: str) -> List[ColumnDependency]:
        """
        Analyse les dépendances de clés étrangères pour une colonne.
        
        Args:
            column: Nom de la colonne
            
        Returns:
            Liste des dépendances de clé étrangère
        """
        # Initialisation de la liste des dépendances associées à la colonne
        dependencies = []
        
        try:
            # Vérification si la colonne est une clé étrangère vers une dimension
            if self._is_dimension_column(column):
                # Nom de la table
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
            
        except Exception as e:
            # Logging
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
        # Initialisation de la liste des indices supprimés
        dropped_indexes = []
        
        try:
            # Recherche des index utilisant cette colonne
            index_query = """
                SELECT index_name, expressions 
                FROM duckdb_indexes() 
                WHERE expressions LIKE ?
            """
            # Exécution de la requête
            indexes = self.conn.execute(index_query, [f'%{column}%']).fetchall()
            # Parcours des index à supprimer
            for index_name, expressions in indexes:
                try:
                    # Suppression de l'index
                    drop_query = f"DROP INDEX IF EXISTS {index_name}"
                    self.conn.execute(drop_query)
                    dropped_indexes.append(index_name)
                    # Logging
                    self.logger.info(f"Index supprimé: {index_name} (expressions: {expressions})")
                    
                except Exception as e:
                    # Logging
                    self.logger.error(f"Erreur lors de la suppression de l'index {index_name}: {e}")
            if dropped_indexes :
                # Logging
                self.logger.info(f"Index supprimés pour {column}: {dropped_indexes}")
            
        except Exception as e:
            # Logging
            self.logger.error(f"Erreur lors de la recherche des index pour {column}: {e}")

            if dropped_indexes :
                # Logging
                self.logger.info(f"Index supprimés pour {column}: {dropped_indexes}")
    
    # Méthode de nettoyage des index orphelins
    def _cleanup_orphaned_indexes(self) -> List[str]:
        """
        Clean up indexes that reference non-existent columns.
        
        Returns:
            List of cleaned up index names
            
        Example:
            >>> cleaned = deleter._cleanup_orphaned_indexes()
        """
        
        try:
            # Récupération des colonnes existantes dans fact_table
            existing_columns = set(self._get_fact_table_columns())
            
            # Récupération de tous les index
            all_indexes = self.conn.execute("""
                SELECT index_name, expressions 
                FROM duckdb_indexes()
            """).fetchall()
            
            # Parcours des indices
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
                        # Suppression de l'indice
                        self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")
                        # Logging
                        self.logger.info(f"Index orphelin supprimé: {index_name}")
                    except Exception as e:
                        # Logging
                        self.logger.error(f"Erreur lors de la suppression de l'index orphelin {index_name}: {e}")
            
        except Exception as e:
            # Logging
            self.logger.error(f"Erreur lors du nettoyage des index orphelins: {e}")
    
    # Méthode d'analyse des dépendances d'index
    def _analyze_index_dependencies(self, column: str) -> List[ColumnDependency]:
        """
        Analyse les dépendances d'index pour une colonne.
        
        Args:
            column: Nom de la colonne
            
        Returns:
            Liste des dépendances d'index
        """
        # Initialisation de la liste des dépendances indicaire
        dependencies = []
        
        try:
            # Recherche des index utilisant cette colonne
            index_query = """
                SELECT index_name, expressions FROM duckdb_indexes()
                WHERE expressions LIKE ?
            """
            # Exécution de la requête
            indexes = self.conn.execute(index_query, [f'%{column}%']).fetchall()
            
            for index_name, _ in indexes:
                # Ajout de la dépendance
                dependency = ColumnDependency(
                    column_name=column,
                    depends_on=[index_name],
                    dependency_type='index',
                    cascade_action='drop'
                )
                dependencies.append(dependency)
                
        except Exception as e:
            # Logging
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

    # Méthode de suppression de lignes dans la table de faits
    def delete_rows(self, 
                   filters: Optional[str] = None) -> None:
        """
        Delete rows from fact table based on filters.
        
        Args:
            filters (Optional[ Union[List[Tuple[str, str, Any]], List[List[Tuple[str, str, Any]]], str, None] ], optional) : A filter condition for the rows. Filter syntax: [[(column, op, val), …],…] where op is [=, >, >=, <, <=, !=, in, not in].
            The innermost tuples are transposed into a set of filters applied through an AND operation.
            The outer list combines these sets of filters through an OR operation.
            A single list of tuples can also be used, meaning that no OR operation between set of filters is to be conducted. Defaults to None.
            A string can be used and is interpreted as a written sql condition
        """
        # Comptage initial
        initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        
        # Construction de la clause WHERE
        where_clause = _build_where_clause(filters)
        
        if not where_clause:
            self.logger.warning("No filters provided for deletion - operation cancelled")
            return 0
        
        # Construction de la requête de suppression
        delete_query = f"DELETE FROM fact_table {where_clause}"
        
        try:
            # Exécution de la requête
            self.conn.execute(delete_query)
            
            # Comptage final
            final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            rows_deleted = initial_count - final_count
            
            # Logging
            self.logger.info(f"Successfully deleted {rows_deleted} rows from fact_table")
            
            # Nettoyage des dimensions orphelines
            self.clean_orphaned_dimensions()
            
            # Nettoyage des index orphelins
            self._cleanup_orphaned_indexes()

            # Suppression des colonnes ne contenant que des nulles, des index et des tables de dimension associées
            null_only_columns = self._get_null_only_columns()
            if null_only_columns:
                self.delete_columns(null_only_columns)
            
            # Nettoyage des entrées orphelines dans les tables de dimension
            self._cleanup_dimension_orphaned_entries()
            
            return rows_deleted
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to delete rows: {e}")
            raise
    
    # Méthode de suppression de colonnes avec gestion rigoureuse des dépendances
    def delete_columns(self, columns: List[str], use_transaction: bool = True) -> None:
        """
        Delete columns from fact table and update related tables.
        
        Args:
            columns: List of column names to delete
            use_transaction: Whether to use database transaction for atomicity
            
        Example:
            >>> deleter.delete_columns(['old_column1', 'old_column2'])
        """       
        # Vérification des colonnes existantes
        existing_columns = self._get_fact_table_columns()
        valid_columns = np.intersect1d(existing_columns, columns).tolist()

        # Initialisation du booléen d'erreur
        has_errors=False
        
        # Utilisation de transaction pour l'atomicité si demandée
        if use_transaction:
            try:
                # Commencement de la transaction
                self.conn.execute("BEGIN TRANSACTION")
                # Logging
                self.logger.info("Transaction commencée pour la suppression des colonnes")
            except Exception as e:
                # Logging
                self.logger.warning(f"Impossible de commencer une transaction: {e}")
                # Fallback
                use_transaction = False
        
        try:
            # Parcours des colonnes existantes à supprimer
            for column in valid_columns:
                try:                   
                    # Suppression de la colonne de la fact table
                    alter_query = f"ALTER TABLE fact_table DROP COLUMN {column}"
                    self.conn.execute(alter_query)
                    
                    # Si c'est une dimension, suppression de la table associée
                    # Nom de la table de dimension
                    table_name = f"dim_{column}"
                    # Suppression de la table de dimension
                    self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    
                    # Suppression des index pour les colonnes non-catégorielles
                    self._drop_column_indexes(column)
                    
                    # Suppression des métadonnées
                    self.delete_column_metadata(column)
                    
                    # Logging
                    self.logger.info(f"Successfully deleted column {column}")
                    
                except Exception as e:
                    has_errors=True
                    # Logging
                    self.logger.error(f"Failed to delete column {column}: {e}")
                    # En cas d'erreur, continuer avec les autres colonnes
                    # Le rollback sera géré à la fin si nécessaire
        
            # Nettoyage final des index orphelins
            self._cleanup_orphaned_indexes()
            
            # Commit de la transaction si tout s'est bien passé
            if use_transaction:
                # S'il n'y a pas d'erreur, exécution de la suppression
                if not has_errors:
                    # Exécution de la transaction
                    self.conn.execute("COMMIT")
                    # Logging
                    self.logger.info("Transaction commitée avec succès")
                else:
                    # Annulation de la transaction
                    self.conn.execute("ROLLBACK")
                    # Logging
                    self.logger.warning("Transaction annulée à cause d'erreurs")
                    
        except Exception as e:
            # Gestion des erreurs globales
            self.logger.error(f"Erreur globale lors de la suppression des colonnes: {e}")
            # Si on utilise une transaction, annulation à cuase de l'erreir
            if use_transaction:
                try:
                    # Annulation de la transaction
                    self.conn.execute("ROLLBACK")
                    # Logging
                    self.logger.info("Transaction annulée à cause d'une erreur globale")
                except Exception:
                    pass

    # Méthode de suppression des métadonnées d'une colonne
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
            # Logging
            self.logger.info(f"Deleted metadata for column {column_name}")
            
        except Exception as e:
            # Logging
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
                
                # Récupération des valeurs uniques de la colonne
                unique_labels_query = f"SELECT DISTINCT {col_name} as label FROM fact_table WHERE {col_name} IS NOT NULL ORDER BY {col_name}"
                unique_labels_result = self.conn.execute(unique_labels_query).fetchdf()
                
                if len(unique_labels_result) > 0:
                    # Conversion en catégorielle si nécessaire
                    self._check_and_convert_to_categorical(col_name=col_name, values=unique_labels_result['label'])

            except Exception as e:
                # Logging
                self.logger.error(f"Error detecting categorical status for {col_name} after upsert: {e}")
    

    # Méthodes auxilaire convertissant une variable en catégorielle si elle en satisfait les conditions
    def _check_and_convert_to_categorical(self, col_name: str, values: pd.Series) -> None:
        """
        Check if a non-categorical column should become categorical and convert if needed.
        
        Args:
            col_name: Column name
            values: Series with column values
        """
        if self._check_categorical_threshold(values, self.categorical_threshold):
            
            # Création de la table de dimension via _update_dimension_values
            self._update_dimension_values(col_name, values)
            
            # Mise à jour du statut catégoriel
            self._update_categorical_status(col_name, True)

            # Remplacement des labels par les valeurs de la table de dimension dans la fact table
            self._convert_fact_table_dimension_mapping(col_name, values_to_labels=False)  
            
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
            return False
        else :
            return True 
    
    
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
                    """, [str(value), str(label)])
                    values_added += 1
                except Exception as e:
                    # Logging
                    self.logger.warning(f"Cannot insert {value} into {table_name}: {e}")
            
            if values_added > 0:
                # Logging
                self.logger.info(f"Added {values_added} new values to {table_name}")
    
    # Méthode d'identification des colonnes ne contenant que des valeurs nulles
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
            self.logger.error(f"An error occured while detecting null values: {e}")
        
        return null_only_columns
    
    # Méthode de nettoyage des entrées orphelines dans les tables de dimension
    def _cleanup_dimension_orphaned_entries(self) -> None:
        """
        Clean up dimension table entries that are no longer referenced in the fact table.
        """
        try:
            # Chargement des métadonnées pour identifier les colonnes catégorielles
            current_metadata = self._load_current_metadata()
            
            # Parcours des colonnes catégorielles
            for _, row in current_metadata.iterrows():
                col_name = row['name']
                is_categorical = row['is_categorical']
                # Nettotage des variables catégorielles
                if is_categorical:
                    self._cleanup_single_dimension_orphaned_entries(col_name)
                    
        except Exception as e:
            # Logging
            self.logger.error(f"An error occured while cleaning orphaned dimension entries: {e}")
    
    # Méthode de nettoyage des entrées orphelines pour une dimension spécifique
    def _cleanup_single_dimension_orphaned_entries(self, col_name: str) -> None:
        """
        Clean up orphaned entries for a specific dimension table.
        
        Args:
            col_name: Column name associated with the dimension table
        """
        # Nom de la table de dimension
        table_name = f"dim_{col_name}"
        
        try:
            # Vérification que la table de dimension existe
            dimension_exists = self.conn.execute(f"""
                SELECT COUNT(*) FROM information_schema.tables 
                WHERE table_name = '{table_name}'
            """).fetchone()[0] > 0
            
            if dimension_exists:
                # Vérification que la colonne existe dans la fact table
                fact_columns = self._get_fact_table_columns()
                if col_name not in fact_columns:
                    # Si la colonne n'existe plus dans la fact table, supprimer toute la dimension
                    self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                    # Logging
                    self.logger.info(f"Table de dimension {table_name} supprimée car la colonne {col_name} n'existe plus")
                
                # Suppression des entrées orphelines (valeurs qui ne sont plus référencées dans la fact table)
                cleanup_query = f"""
                    DELETE FROM {table_name}
                    WHERE value NOT IN (
                        SELECT DISTINCT {col_name} 
                        FROM fact_table 
                        WHERE {col_name} IS NOT NULL
                    )
                """
                
                # Comptage des entrées avant suppression
                count_before = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                
                # Exécution de la suppression
                self.conn.execute(cleanup_query)
                
                # Comptage des entrées après suppression
                count_after = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                
                removed_count = count_before - count_after
                # Logging
                if removed_count > 0:
                    self.logger.info(f"Removed {removed_count} orphaned entries from {table_name}")
                    
        except Exception as e:
            # Logging
            self.logger.error(f"An error occured while cleaning orphaned entries from {table_name}: {e}")