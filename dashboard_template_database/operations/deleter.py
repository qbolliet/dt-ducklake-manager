# Module pour la suppression de données et colonnes
import os
from pathlib import Path
from typing import List, Optional, Dict, Union, Set, Tuple
import duckdb
import pandas as pd
from enum import Enum
from dataclasses import dataclass

from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

class CascadeMode(Enum):
    """
    Modes de suppression en cascade.
    """
    NO_CASCADE = "no_cascade"
    RESTRICT = "restrict"  # Empêche la suppression si des dépendances existent
    CASCADE = "cascade"    # Supprime automatiquement les dépendances
    SET_NULL = "set_null"  # Met les références à NULL

@dataclass
class ColumnDependency:
    """
    Représente une dépendance de colonne.
    """
    column_name: str
    depends_on: List[str]
    dependency_type: str  # 'foreign_key', 'computed', 'index'
    cascade_action: str
    
@dataclass
class DeletionPlan:
    """
    Plan d'exécution pour une suppression.
    """
    target_table: str
    target_columns: List[str]
    affected_tables: List[str]
    cascade_operations: List[Dict[str, Union[str, List[str]]]]
    warnings: List[str]
    estimated_rows_affected: int

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
        
        # Cache pour les dépendances
        self._dependency_cache = None
        self._foreign_key_cache = None
    
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
            
            # Analyse des colonnes calculées
            computed_deps = self._analyze_computed_dependencies(column)
            column_deps.extend(computed_deps)
            
            # Analyse des contraintes CHECK
            check_deps = self._analyze_check_constraints(column)
            column_deps.extend(check_deps)
            
            dependencies[column] = column_deps
            
        return dependencies
    
    def create_deletion_plan(self, 
                           columns_to_delete: List[str], 
                           cascade_mode: CascadeMode = CascadeMode.RESTRICT) -> DeletionPlan:
        """
        Create a comprehensive deletion plan.
        
        Args:
            columns_to_delete: Columns to delete
            cascade_mode: How to handle cascading deletions
            
        Returns:
            Detailed deletion plan
        """
        self.logger.info(f"Création du plan de suppression pour: {columns_to_delete}")
        
        # Analyse des dépendances
        dependencies = self.analyze_column_dependencies(columns_to_delete)
        
        # Construction du plan
        plan = DeletionPlan(
            target_table="fact_table",
            target_columns=columns_to_delete,
            affected_tables=[],
            cascade_operations=[],
            warnings=[],
            estimated_rows_affected=0
        )
        
        # Estimation du nombre de lignes affectées
        plan.estimated_rows_affected = self.conn.execute(
            "SELECT COUNT(*) FROM fact_table"
        ).fetchone()[0]
        
        # Analyse de chaque colonne
        for column in columns_to_delete:
            column_deps = dependencies.get(column, [])
            
            if not column_deps and cascade_mode == CascadeMode.RESTRICT:
                continue
                
            # Traitement selon le mode cascade
            for dep in column_deps:
                if cascade_mode == CascadeMode.RESTRICT:
                    if dep.dependency_type in ['foreign_key', 'computed']:
                        plan.warnings.append(
                            f"Suppression de {column} bloquée par dépendance: {dep.dependency_type}"
                        )
                        
                elif cascade_mode == CascadeMode.CASCADE:
                    cascade_op = {
                        'operation': f'cascade_delete_{dep.dependency_type}',
                        'target': dep.depends_on,
                        'reason': f'Dépendance de {column}'
                    }
                    plan.cascade_operations.append(cascade_op)
                    
                elif cascade_mode == CascadeMode.SET_NULL:
                    if dep.dependency_type == 'foreign_key':
                        cascade_op = {
                            'operation': 'set_null',
                            'target': dep.depends_on,
                            'reason': f'Référence à {column} supprimée'
                        }
                        plan.cascade_operations.append(cascade_op)
        
        # Identification des tables affectées
        affected_tables = set(['fact_table'])
        for op in plan.cascade_operations:
            if 'target' in op:
                if isinstance(op['target'], list):
                    for target in op['target']:
                        if target.startswith('dim_'):
                            affected_tables.add(target)
                elif op['target'].startswith('dim_'):
                    affected_tables.add(op['target'])
        
        plan.affected_tables = list(affected_tables)
        
        return plan
    
    def execute_deletion_plan(self, plan: DeletionPlan, confirm: bool = False) -> Dict[str, Any]:
        """
        Execute a deletion plan.
        
        Args:
            plan: Deletion plan to execute
            confirm: Whether to actually execute (dry run if False)
            
        Returns:
            Execution results
        """
        if not confirm:
            return {
                'dry_run': True,
                'plan_summary': {
                    'target_columns': plan.target_columns,
                    'affected_tables': plan.affected_tables,
                    'cascade_operations': len(plan.cascade_operations),
                    'warnings': len(plan.warnings)
                }
            }
        
        self.logger.info("Exécution du plan de suppression")
        execution_results = {
            'success': False,
            'operations_completed': [],
            'operations_failed': [],
            'rollback_info': None
        }
        
        # Début de transaction pour atomicité
        try:
            self.conn.execute("BEGIN TRANSACTION")
            
            # Exécution des opérations cascade d'abord
            for cascade_op in plan.cascade_operations:
                try:
                    self._execute_cascade_operation(cascade_op)
                    execution_results['operations_completed'].append(cascade_op)
                except Exception as e:
                    execution_results['operations_failed'].append({
                        'operation': cascade_op,
                        'error': str(e)
                    })
                    raise
            
            # Suppression des colonnes principales
            for column in plan.target_columns:
                try:
                    self._delete_column_safe(column)
                    execution_results['operations_completed'].append(f'delete_column_{column}')
                except Exception as e:
                    execution_results['operations_failed'].append({
                        'operation': f'delete_column_{column}',
                        'error': str(e)
                    })
                    raise
            
            # Validation post-suppression
            self._validate_post_deletion(plan)
            
            self.conn.execute("COMMIT")
            execution_results['success'] = True
            
        except Exception as e:
            self.conn.execute("ROLLBACK")
            execution_results['rollback_performed'] = True
            self.logger.error(f"Échec de l'exécution du plan: {e}")
            raise
        
        return execution_results
    
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
    
    def _analyze_computed_dependencies(self, column: str) -> List[ColumnDependency]:
        """
        Analyse les dépendances de colonnes calculées.
        
        Args:
            column: Nom de la colonne
            
        Returns:
            Liste des dépendances calculées
        """
        dependencies = []
        
        # Dans DuckDB, les colonnes calculées peuvent être des vues
        # Ici on peut étendre pour analyser les vues qui dépendent de cette colonne
        
        return dependencies
    
    def _analyze_check_constraints(self, column: str) -> List[ColumnDependency]:
        """
        Analyse les contraintes CHECK impliquant une colonne.
        
        Args:
            column: Nom de la colonne
            
        Returns:
            Liste des dépendances de contraintes
        """
        dependencies = []
        
        # DuckDB a un support limité des contraintes CHECK
        # Cette méthode peut être étendue selon les besoins
        
        return dependencies
    
    def _execute_cascade_operation(self, operation: Dict[str, Any]) -> None:
        """
        Exécute une opération en cascade.
        
        Args:
            operation: Description de l'opération à exécuter
        """
        op_type = operation.get('operation', '')
        
        if op_type.startswith('cascade_delete_'):
            dependency_type = op_type.replace('cascade_delete_', '')
            targets = operation.get('target', [])
            
            if dependency_type == 'foreign_key':
                for target in targets:
                    if target.startswith('dim_'):
                        self.conn.execute(f"DROP TABLE IF EXISTS {target}")
                        self.logger.info(f"Table de dimension supprimée: {target}")
                        
            elif dependency_type == 'index':
                for target in targets:
                    self.conn.execute(f"DROP INDEX IF EXISTS {target}")
                    self.logger.info(f"Index supprimé: {target}")
                    
        elif op_type == 'set_null':
            # Mise à NULL des références
            targets = operation.get('target', [])
            for target in targets:
                if target.startswith('dim_'):
                    # Cette opération dépend de la structure exacte
                    self.logger.info(f"Mise à NULL des références dans {target}")
    
    def _delete_column_safe(self, column: str) -> None:
        """
        Supprime une colonne de manière sécurisée.
        
        Args:
            column: Nom de la colonne à supprimer
        """
        # Vérification préalable
        if not self._column_exists(column):
            self.logger.warning(f"La colonne {column} n'existe pas")
            return
        
        # Suppression des métadonnées
        self.delete_column_metadata(column)
        
        # Suppression de la colonne de la table des faits
        alter_query = f"ALTER TABLE fact_table DROP COLUMN {column}"
        self.conn.execute(alter_query)
        
        self.logger.info(f"Colonne {column} supprimée avec succès")
    
    def _validate_post_deletion(self, plan: DeletionPlan) -> None:
        """
        Valide l'état après suppression.
        
        Args:
            plan: Plan de suppression exécuté
        """
        # Vérification que les colonnes ont bien été supprimées
        existing_columns = self._get_fact_table_columns()
        
        for column in plan.target_columns:
            if column in existing_columns:
                raise ValueError(f"La colonne {column} existe encore après suppression")
        
        # Vérification de l'intégrité référentielle restante
        self._verify_referential_integrity()
        
        self.logger.info("Validation post-suppression réussie")
    
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