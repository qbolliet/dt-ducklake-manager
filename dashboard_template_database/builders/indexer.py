# Module pour la gestion des index sur les tables DuckDB
import logging
from typing import List, Dict, Optional
import duckdb

class IndexManager:
    """
    Manages indexes on DuckDB tables for optimized query performance.
    
    This class provides functionality to create, drop, and manage indexes
    on fact tables and dimension tables based on configuration.
    """
    
    def __init__(self, connection: duckdb.DuckDBPyConnection):
        """
        Initialize the IndexManager.
        
        Args:
            connection: DuckDB connection object
        """
        self.conn = connection
        self.logger = logging.getLogger(__name__)
    
    def create_index(self, table_name: str, index_name: str, columns: List[str], unique: bool = False) -> None:
        """
        Create an index on specified columns.
        
        Args:
            table_name: Name of the table
            index_name: Name of the index
            columns: List of column names to index
            unique: Whether the index should be unique
        """
        try:
            # Construction de la requête d'index
            unique_clause = "UNIQUE" if unique else ""
            columns_str = ", ".join(columns)
            
            query = f"""
                CREATE {unique_clause} INDEX IF NOT EXISTS {index_name}
                ON {table_name} ({columns_str})
            """
            
            self.conn.execute(query)
            self.logger.info(f"Successfully created index {index_name} on {table_name}({columns_str})")
            
        except Exception as e:
            self.logger.error(f"Failed to create index {index_name}: {e}")
            raise
    
    def create_fact_table_indexes(self, fact_table_name: str, index_config: Dict) -> None:
        """
        Create indexes on fact table based on configuration.
        
        Args:
            fact_table_name: Name of the fact table
            index_config: Dictionary containing index configurations
        
        Example config:
        {
            "primary_indexes": ["column1", "column2"],
            "composite_indexes": [
                {"columns": ["col1", "col2"], "name": "idx_composite"}
            ],
            "dimension_keys": ["dim1", "dim2"]
        }
        """
        # Index sur les colonnes primaires individuelles
        if "primary_indexes" in index_config:
            for column in index_config["primary_indexes"]:
                index_name = f"idx_{fact_table_name}_{column}"
                self.create_index(fact_table_name, index_name, [column])
        
        # Index composites pour les requêtes fréquentes
        if "composite_indexes" in index_config:
            for idx_conf in index_config["composite_indexes"]:
                self.create_index(
                    fact_table_name,
                    idx_conf["name"],
                    idx_conf["columns"]
                )
        
        # Index sur les clés étrangères vers les dimensions
        if "dimension_keys" in index_config:
            for dim_column in index_config["dimension_keys"]:
                index_name = f"idx_{fact_table_name}_fk_{dim_column}"
                self.create_index(fact_table_name, index_name, [dim_column])
    
    def analyze_table(self, table_name: str) -> None:
        """
        Analyze table to update statistics for query optimization.
        
        Args:
            table_name: Name of the table to analyze
        """
        try:
            self.conn.execute(f"ANALYZE {table_name}")
            self.logger.info(f"Successfully analyzed table {table_name}")
        except Exception as e:
            self.logger.error(f"Failed to analyze table {table_name}: {e}")
            raise
    
    def drop_index(self, index_name: str) -> None:
        """
        Drop an existing index.
        
        Args:
            index_name: Name of the index to drop
        """
        try:
            self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")
            self.logger.info(f"Successfully dropped index {index_name}")
        except Exception as e:
            self.logger.error(f"Failed to drop index {index_name}: {e}")
            raise
    
    def list_indexes(self, table_name: Optional[str] = None) -> List[Dict]:
        """
        List all indexes or indexes for a specific table.
        
        Args:
            table_name: Optional table name to filter indexes
            
        Returns:
            List of dictionaries containing index information
        """
        try:
            if table_name:
                query = f"""
                    SELECT * FROM duckdb_indexes()
                    WHERE table_name = '{table_name}'
                """
            else:
                query = "SELECT * FROM duckdb_indexes()"
            
            result = self.conn.execute(query).fetchall()
            
            # Conversion en liste de dictionnaires
            indexes = []
            for row in result:
                indexes.append({
                    'database_name': row[0],
                    'database_oid': row[1],
                    'schema_name': row[2],
                    'schema_oid': row[3],
                    'table_name': row[4],
                    'table_oid': row[5],
                    'index_name': row[6],
                    'index_oid': row[7],
                    'is_unique': row[8],
                    'is_primary': row[9],
                    'expressions': row[10],
                    'sql': row[11]
                })
            
            return indexes
            
        except Exception as e:
            self.logger.error(f"Failed to list indexes: {e}")
            raise
    
    def optimize_for_queries(self, fact_table_name: str, sample_queries: List[str]) -> None:
        """
        Analyze sample queries and suggest/create optimal indexes.
        
        Args:
            fact_table_name: Name of the fact table
            sample_queries: List of sample SQL queries to optimize for
        """
        # Cette méthode pourrait analyser les queries et suggérer des index
        # Pour l'instant, implémentation basique
        self.logger.info("Analyzing queries for optimization...")
        
        # Analyse des colonnes utilisées dans WHERE et JOIN
        columns_to_index = set()
        
        for query in sample_queries:
            # Analyse simplifiée - recherche des colonnes après WHERE
            if 'WHERE' in query.upper():
                where_part = query.upper().split('WHERE')[1]
                # Extraction basique des colonnes (à améliorer)
                tokens = where_part.split()
                for token in tokens:
                    if token not in ['AND', 'OR', 'NOT', '=', '>', '<', '>=', '<=', 'LIKE']:
                        # Vérification si c'est une colonne de la table
                        try:
                            self.conn.execute(f"SELECT {token} FROM {fact_table_name} LIMIT 0")
                            columns_to_index.add(token.lower())
                        except:
                            pass
        
        # Création des index suggérés
        for column in columns_to_index:
            index_name = f"idx_auto_{fact_table_name}_{column}"
            try:
                self.create_index(fact_table_name, index_name, [column])
            except:
                pass