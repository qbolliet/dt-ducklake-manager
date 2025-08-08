# Importation des modules
# Modules de base
import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter, defaultdict
# DuckDB
import duckdb
# SQL parsing
import sqlparse

# Module d'initialisation du logger
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Classe de gestion de l'indexation dans les tables DuckDB
class IndexManager:
    """
    Manages indexes on DuckDB tables for optimized query performance.
    
    This class provides functionality to create, drop, and manage indexes
    on fact tables and dimension tables based on configuration.
    """
    # Initialisation
    def __init__(self, connection: Optional[duckdb.DuckDBPyConnection] = None, path : Optional[os.PathLike]=None, log_filename: Optional[os.PathLike] = os.path.join(FILE_PATH.parents[2], "logs/duckdb_indexer.log")):
        """
        Initialize the IndexManager.
        
        Args:
            connection: DuckDB connection object
        """
        # Initialisation de la connection
        if (connection is None) & (path is None) :
            self.conn = duckdb.connect(':memory:')
        elif (connection is None) :
            self.conn = duckdb.connect(path)
        else :
            self.conn = connection
        # Initialisation du logger
        self.logger = _init_logger(filename=log_filename)
    
    # Méthode de création d'un index
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
            # Définition de l'unicité de l'index
            unique_clause = "UNIQUE" if unique else ""
            # Définition des colonnes servant à construire l'indice
            columns_str = ", ".join(columns)
            # Définition de la requête
            query = f"""
                CREATE {unique_clause} INDEX IF NOT EXISTS {index_name}
                ON {table_name} ({columns_str})
            """
            # Exécution de la requête
            self.conn.execute(query)
            # Logging
            self.logger.info(f"Successfully created index {index_name} on {table_name}({columns_str})")
            
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to create index {index_name}: {e}")
            raise

    # Méthode de création d'index sur la table des faits à partir d'un dictionnaire de configuration
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
            # Parcours des index un à un
            for column in index_config["primary_indexes"]:
                # Nom de l'index
                index_name = f"idx_{fact_table_name}_{column}"
                # Création de l'index
                self.create_index(fact_table_name, index_name, [column])
        
        # Index composites pour les requêtes fréquentes
        if "composite_indexes" in index_config:
            # Parcours des index composites sur plusieurs colonnes
            for idx_conf in index_config["composite_indexes"]:
                # Création de l'index
                self.create_index(
                    fact_table_name,
                    idx_conf["name"],
                    idx_conf["columns"]
                )
        
        # Index sur les clés étrangères vers les dimensions
        if "dimension_keys" in index_config:
            # Parcours des colonnes de dimension
            for dim_column in index_config["dimension_keys"]:
                # Nom de l'index
                index_name = f"idx_{fact_table_name}_fk_{dim_column}"
                # Création de l'index
                self.create_index(fact_table_name, index_name, [dim_column])
    
    # Méthode d'analyse de la table
    def analyze_table(self, table_name: str) -> None:
        """
        Analyze table to update statistics for query optimization.
        
        Args:
            table_name: Name of the table to analyze
        """
        try:
            # Exécution de l'analyse
            self.conn.execute(f"ANALYZE {table_name}")
            # Logging
            self.logger.info(f"Successfully analyzed table {table_name}")
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to analyze table {table_name}: {e}")
            raise
    
    # Méthode de suppression d'un index existant
    def drop_index(self, index_name: str) -> None:
        """
        Drop an existing index.
        
        Args:
            index_name: Name of the index to drop
        """
        try:
            # Exécution de la requête de suppression
            self.conn.execute(f"DROP INDEX IF EXISTS {index_name}")
            # Logging
            self.logger.info(f"Successfully dropped index {index_name}")
        except Exception as e:
            # Logging
            self.logger.error(f"Failed to drop index {index_name}: {e}")
            raise
    
    # Méthode d'énumération des indices
    def list_indexes(self, table_name: Optional[str] = None) -> List[Dict]:
        """
        List all indexes or indexes for a specific table.
        
        Args:
            table_name: Optional table name to filter indexes
            
        Returns:
            List of dictionaries containing index information
        """
        try:
            # Si une table est spécifiée, on extrait les index de cette dernière
            if table_name:
                # Construction de la requête
                query = f"""
                    SELECT * FROM duckdb_indexes()
                    WHERE table_name = '{table_name}'
                """
            # Sinon on sélectionne l'ensemble des index
            else:
                query = "SELECT * FROM duckdb_indexes()"
            
            # Exécution de la requête
            result = self.conn.execute(query).fetchall()
            
            # Conversion en liste de dictionnaires
            indexes = []
            # Parcours du résultat
            for row in result:
                # Parsing du résultat
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
            # Logging
            self.logger.error(f"Failed to list indexes: {e}")
            raise
    
    # Méthode d'extraction des colonnes à partir d'une requête SQL
    def _extract_columns_from_query(self, query: str, table_name: str) -> Dict[str, Set[str]]:
        """
        Extract columns used in different parts of a SQL query.
        
        Args:
            query: SQL query string
            table_name: Name of the table to analyze
            
        Returns:
            Dictionary with 'where', 'join', 'order_by' keys containing column sets
        """
        # Initialisation du dictionnaire résultat
        result = {
            'where': set(),
            'join': set(), 
            'order_by': set()
        }
        
        
        # Parsing de la requête sql
        parsed = sqlparse.parse(query)[0]
        # Analyse de la requête parsée
        self._analyze_parsed_query(parsed, result, table_name)
            
        return result
    
    # Méthode d'analyse d'une requête parsée avec sqlparse
    def _analyze_parsed_query(self, parsed_query, result: Dict[str, Set[str]], table_name: str) -> None:
        """
        Analyze parsed SQL query to extract relevant columns.
        
        Args:
            parsed_query: Parsed SQL query object
            result: Dictionary to store extracted columns
            table_name: Name of the table to analyze
        """
        # Récupération des colonnes valides de la table
        valid_columns = self._get_table_columns(table_name)
        
        # Parcours récursif des tokens
        self._traverse_tokens(parsed_query.tokens, result, valid_columns)
    
    # Méthode de parcours récursif des tokens SQL
    def _traverse_tokens(self, tokens, result: Dict[str, Set[str]], valid_columns: Set[str]) -> None:
        """
        Recursively traverse SQL tokens to extract column references.
        
        Args:
            tokens: SQL tokens to traverse
            result: Dictionary to store extracted columns  
            valid_columns: Set of valid column names for the table
        """
        # Initialisation
        in_where = False
        in_join = False
        in_order_by = False
        
        # Parcours des tokens
        for token in tokens:
            if hasattr(token, 'tokens'):
                # Token composite, parcours récursif
                self._traverse_tokens(token.tokens, result, valid_columns)
            else:
                token_upper = str(token).upper().strip()
                
                # Détection des sections
                if token_upper == 'WHERE':
                    in_where = True
                elif token_upper in ['JOIN', 'INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'FULL JOIN']:
                    in_join = True
                elif token_upper == 'ORDER BY':
                    in_order_by = True
                elif token_upper in ['GROUP BY', 'HAVING', 'SELECT', 'FROM']:
                    in_where = in_join = in_order_by = False
                
                # Extraction des colonnes selon le contexte
                if token_upper in valid_columns:
                    if in_where:
                        result['where'].add(token_upper.lower())
                    elif in_join:
                        result['join'].add(token_upper.lower())
                    elif in_order_by:
                        result['order_by'].add(token_upper.lower())
    
    # Méthode de récupération des colonnes d'une table
    def _get_table_columns(self, table_name: str) -> Set[str]:
        """
        Get all column names for a table.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Set of column names in uppercase
        """
        try:
            # Exécution de la requête des colonnes
            result = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
            # Renvoi des colonnes
            return {row[0].upper() for row in result}
        except Exception:
            return set()
    
    # Méthode d'analyse du plan d'exécution d'une requête
    def _analyze_query_plan(self, query: str) -> Dict[str, any]:
        """
        Analyze query execution plan to identify bottlenecks.
        
        Args:
            query: SQL query to analyze
            
        Returns:
            Dictionary with plan analysis results
        """
        try:
            # Exécution d'EXPLAIN pour obtenir le plan d'exécution
            explain_result = self.conn.execute(f"EXPLAIN {query}").fetchall()
            # Mise en forme des résultats
            plan_text = '\n'.join([row[1] for row in explain_result])
            # Construction du dictionnaire d'analyse
            analysis = {
                'has_table_scan': 'TABLE_SCAN' in plan_text.upper(),
                'has_index_scan': 'INDEX_SCAN' in plan_text.upper(),
                'has_sort': 'SORT' in plan_text.upper(),
                'has_hash_join': 'HASH_JOIN' in plan_text.upper(),
                'estimated_cost': self._extract_cost_from_plan(plan_text)
            }
            
            return analysis
            
        except Exception as e:
            # Logging
            self.logger.warning(f"Échec de l'analyse du plan d'exécution: {e}")
            return {}
    
    # Méthode d'extraction du coût estimé depuis le plan d'exécution
    def _extract_cost_from_plan(self, plan_text: str) -> float:
        """
        Extract estimated cost from execution plan.
        
        Args:
            plan_text: Execution plan text
            
        Returns:
            Estimated cost or 0.0 if not found
        """
        # Recherche de patterns de coût dans le plan
        cost_patterns = [
            r'cost=(\d+\.?\d*)',
            r'Cost:\s*(\d+\.?\d*)',
            r'estimated_cost:\s*(\d+\.?\d*)'
        ]
        # Parcours des patterns
        for pattern in cost_patterns:
            matches = re.findall(pattern, plan_text, re.IGNORECASE)
            if matches:
                return float(matches[0])
        
        return 0.0
    
    # Méthode de calcul du score de bénéfice d'un index
    def _calculate_index_benefit_score(self, column: str, usage_stats: Dict[str, int], 
                                     table_stats: Dict[str, any]) -> float:
        """
        Calculate benefit score for creating an index on a column.
        
        Args:
            column: Column name
            usage_stats: Usage frequency statistics
            table_stats: Table statistics (cardinality, etc.)
            
        Returns:
            Benefit score for the index
        """
        # Fréquence d'utilisation (WHERE, JOIN, ORDER BY)
        frequency_score = (
            usage_stats.get(f'where_{column}', 0) * 3 +  # WHERE a plus de poids
            usage_stats.get(f'join_{column}', 0) * 2 +   # JOIN a un poids moyen
            usage_stats.get(f'order_{column}', 0) * 1    # ORDER BY a moins de poids
        )
        
        # Cardinalité de la colonne (plus élevée = meilleur index)
        cardinality_score = table_stats.get(f'cardinality_{column}', 1)
        
        # Score combiné normalisé
        benefit_score = frequency_score * min(cardinality_score / 1000, 10)
        
        return benefit_score
    
    # Méthode d'analyse d'un échantillon de requêtes pour créer des index optimaux
    def optimize_for_queries(self, fact_table_name: str, sample_queries: List[str], 
                           create_indexes: bool = True, min_benefit_score: float = 2.0) -> Dict[str, any]:
        """
        Analyze sample queries and suggest/create optimal indexes with comprehensive analysis.
        
        Args:
            fact_table_name: Name of the fact table
            sample_queries: List of sample SQL queries to optimize for
            create_indexes: Whether to actually create the suggested indexes
            min_benefit_score: Minimum benefit score required to create an index
            
        Returns:
            Dictionary containing analysis results and recommendations
        
        Example:
            >>> manager = IndexManager()
            >>> queries = ["SELECT * FROM facts WHERE date > '2023-01-01'", 
            ...           "SELECT * FROM facts JOIN dim_user ON facts.user_id = dim_user.id"]
            >>> result = manager.optimize_for_queries("facts", queries)
            >>> print(result['suggested_indexes'])
        """
        self.logger.info(f"Analyse avancée des requêtes pour l'optimisation de la table {fact_table_name}")
        
        # Structure pour stocker les statistiques d'usage
        usage_stats = defaultdict(int)
        column_combinations = defaultdict(int)
        query_plans = []
        
        # Analyse de chaque requête
        for i, query in enumerate(sample_queries):
            try:
                # Extraction des colonnes utilisées dans différentes parties
                extracted_cols = self._extract_columns_from_query(query, fact_table_name)
                
                # Comptage des usages par type
                for where_col in extracted_cols['where']:
                    usage_stats[f'where_{where_col}'] += 1
                for join_col in extracted_cols['join']:
                    usage_stats[f'join_{join_col}'] += 1
                for order_col in extracted_cols['order_by']:
                    usage_stats[f'order_{order_col}'] += 1
                
                # Détection des combinaisons de colonnes pour index composites
                where_cols = sorted(extracted_cols['where'])
                if len(where_cols) > 1:
                    combination_key = tuple(where_cols)
                    column_combinations[combination_key] += 1
                
                # Analyse du plan d'exécution
                plan_analysis = self._analyze_query_plan(query)
                query_plans.append({
                    'query_index': i,
                    'analysis': plan_analysis,
                    'columns': extracted_cols
                })
                
            except Exception as e:
                self.logger.warning(f"Échec de l'analyse de la requête {i}: {e}")
                continue
        
        # Récupération des statistiques de la table
        table_stats = self._get_table_statistics(fact_table_name)
        
        # Calcul des scores de bénéfice pour chaque colonne
        column_scores = {}
        all_columns = set()
        
        # Collecte de toutes les colonnes utilisées
        for key in usage_stats.keys():
            if key.startswith(('where_', 'join_', 'order_')):
                column = key.split('_', 1)[1]
                all_columns.add(column)
        
        # Calcul des scores
        for column in all_columns:
            score = self._calculate_index_benefit_score(column, usage_stats, table_stats)
            column_scores[column] = score
        
        # Tri des colonnes par score de bénéfice
        sorted_columns = sorted(column_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Suggestion d'index composites basée sur les combinaisons fréquentes
        composite_suggestions = []
        for combination, frequency in column_combinations.items():
            if frequency >= 2 and len(combination) <= 3:  # Limite à 3 colonnes max
                composite_score = sum(column_scores.get(col, 0) for col in combination)
                composite_suggestions.append({
                    'columns': list(combination),
                    'frequency': frequency,
                    'score': composite_score
                })
        
        # Tri des index composites par score
        composite_suggestions.sort(key=lambda x: x['score'], reverse=True)
        
        # Préparation des résultats
        analysis_results = {
            'table_name': fact_table_name,
            'total_queries_analyzed': len([q for q in sample_queries if q.strip()]),
            'usage_statistics': dict(usage_stats),
            'table_statistics': table_stats,
            'column_benefit_scores': column_scores,
            'suggested_single_indexes': [],
            'suggested_composite_indexes': composite_suggestions[:5],  # Top 5
            'query_plan_analysis': query_plans,
            'indexes_created': []
        }
        
        # Suggestion et création d'index individuels
        for column, score in sorted_columns:
            if score >= min_benefit_score:
                index_suggestion = {
                    'column': column,
                    'score': score,
                    'usage_types': [key.split('_')[0] for key in usage_stats.keys() 
                                  if key.endswith(f'_{column}') and usage_stats[key] > 0]
                }
                analysis_results['suggested_single_indexes'].append(index_suggestion)
                
                # Création effective de l'index si demandé
                if create_indexes:
                    index_name = f"idx_opt_{fact_table_name}_{column}"
                    try:
                        self.create_index(fact_table_name, index_name, [column])
                        analysis_results['indexes_created'].append({
                            'name': index_name,
                            'type': 'single',
                            'columns': [column],
                            'score': score
                        })
                    except Exception as e:
                        self.logger.warning(f"Échec de création de l'index {index_name}: {e}")
        
        # Création d'index composites recommandés
        if create_indexes:
            for composite in composite_suggestions[:3]:  # Top 3 seulement
                if composite['score'] >= min_benefit_score * 1.5:  # Seuil plus élevé pour composites
                    columns = composite['columns']
                    index_name = f"idx_comp_{fact_table_name}_{'_'.join(columns)}"
                    try:
                        self.create_index(fact_table_name, index_name, columns)
                        analysis_results['indexes_created'].append({
                            'name': index_name,
                            'type': 'composite',
                            'columns': columns,
                            'score': composite['score']
                        })
                    except Exception as e:
                        self.logger.warning(f"Échec de création de l'index composite {index_name}: {e}")
        
        # Intégration de analyze_table pour mise à jour des statistiques
        if create_indexes and analysis_results['indexes_created']:
            self.logger.info("Mise à jour des statistiques de la table après création d'index")
            try:
                self.analyze_table(fact_table_name)
                analysis_results['table_analyzed'] = True
            except Exception as e:
                self.logger.warning(f"Échec de l'analyse de la table: {e}")
                analysis_results['table_analyzed'] = False
        
        # Logging des résultats
        self.logger.info(f"Analyse terminée: {len(analysis_results['suggested_single_indexes'])} index individuels suggérés, "
                        f"{len(analysis_results['suggested_composite_indexes'])} index composites suggérés")
        if create_indexes:
            self.logger.info(f"{len(analysis_results['indexes_created'])} index créés")
        
        return analysis_results
    
    # Méthode de récupération des statistiques d'une table
    def _get_table_statistics(self, table_name: str) -> Dict[str, any]:
        """
        Get table statistics including row count and column cardinality.
        
        Args:
            table_name: Name of the table
            
        Returns:
            Dictionary with table statistics
        """
        # Initialisation du dictionnaire résultat
        stats = {}
        
        try:
            # Nombre total de lignes
            row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            stats['row_count'] = row_count
            
            # Cardinalité approximative pour chaque colonne
            columns = self._get_table_columns(table_name)
            # Parcours des colonnes
            for column in columns:
                try:
                    # Approximation de la cardinalité (distinct values)
                    cardinality = self.conn.execute(
                        f"SELECT COUNT(DISTINCT {column}) FROM {table_name}"
                    ).fetchone()[0]
                    stats[f'cardinality_{column.lower()}'] = cardinality
                except Exception:
                    stats[f'cardinality_{column.lower()}'] = 1
                    
        except Exception as e:
            # Logging
            self.logger.warning(f"Échec de récupération des statistiques: {e}")
            stats['row_count'] = 0
            
        return stats