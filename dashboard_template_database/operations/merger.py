# Module pour la fusion de bases de données
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set, Callable, Any
import duckdb
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum

from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

class ConflictResolutionMode(Enum):
    """
    Enumération des modes de résolution de conflits pour la fusion.
    """
    UNION = "union"  # Combiner toutes les lignes (permettre les doublons)
    MERGE = "merge"  # Upsert basé sur les clés de fusion
    PRIORITIZE_SOURCE = "prioritize_source"  # Priorité é la base source
    PRIORITIZE_TARGET = "prioritize_target"  # Priorité é la base cible
    CUSTOM = "custom"  # Fonction personnalisée de résolution

class MergeStrategy(Enum):
    """
    Stratégies de fusion disponibles.
    """
    FAST = "fast"  # Fusion rapide sans validation extensive
    SAFE = "safe"  # Fusion sécurisée avec validations complétes
    OPTIMIZED = "optimized"  # Fusion optimisée pour les gros volumes

class DatabaseMerger:
    """
    A comprehensive class to handle merging of DuckDB databases.
    
    Supports merging fact tables, dimension tables, and metadata
    while maintaining data integrity and providing conflict resolution.
    """
    
    def __init__(self, 
                 source_connection: duckdb.DuckDBPyConnection,
                 target_connection: Optional[duckdb.DuckDBPyConnection] = None,
                 log_filename: Optional[os.PathLike] = None):
        """
        Initialize the DatabaseMerger.
        
        Args:
            source_connection: Source database connection
            target_connection: Target database connection (if None, uses source)
            log_filename: Path to log file
            
        Example:
            >>> source_conn = duckdb.connect('source.db')
            >>> target_conn = duckdb.connect('target.db')
            >>> merger = DatabaseMerger(source_conn, target_conn)
        """
        self.source_conn = source_connection
        self.target_conn = target_connection or source_connection
        
        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/database_merger.log")
        self.logger = _init_logger(filename=log_filename)
        
        # Cache pour les métadonnées et schémas
        self._source_metadata_cache = None
        self._target_metadata_cache = None
        self._compatibility_cache = None
        
        # Configuration par défaut
        self.default_merge_keys = ['id']
        self.batch_size = 10000
        self.max_workers = 4
    
    def analyze_compatibility(self, source_db_path: str, target_db_path: str = None) -> Dict[str, Any]:
        """
        Analyze compatibility between source and target databases.
        
        Args:
            source_db_path: Path to source database
            target_db_path: Path to target database
            
        Returns:
            Detailed compatibility analysis report
        """
        self.logger.info("Analyse de compatibilité des bases de données")
        
        # Chargement des métadonnées des deux bases
        source_metadata = self._load_database_metadata(self.source_conn, "source")
        target_metadata = self._load_database_metadata(self.target_conn, "target")
        
        compatibility_report = {
            'timestamp': datetime.now().isoformat(),
            'source_info': source_metadata['info'],
            'target_info': target_metadata['info'],
            'schema_compatibility': self._analyze_schema_compatibility(
                source_metadata['schema'], target_metadata['schema']
            ),
            'dimension_compatibility': self._analyze_dimension_compatibility(
                source_metadata['dimensions'], target_metadata['dimensions']
            ),
            'data_overlap': self._analyze_data_overlap(),
            'merge_recommendations': self._generate_merge_recommendations()
        }
        
        self.logger.info(f"Analyse de compatibilité terminée: {compatibility_report['schema_compatibility']['compatibility_score']:.2f}/10")
        return compatibility_report
    
    def merge_databases(self, 
                       merge_strategy: MergeStrategy = MergeStrategy.SAFE,
                       conflict_resolution: ConflictResolutionMode = ConflictResolutionMode.MERGE,
                       merge_keys: Optional[List[str]] = None,
                       custom_resolution_func: Optional[Callable] = None,
                       backup: bool = True) -> Dict[str, Any]:
        """
        Perform complete database merge operation.
        
        Args:
            merge_strategy: Strategy for merge execution
            conflict_resolution: Mode for handling conflicts
            merge_keys: Columns to use for identifying duplicates
            custom_resolution_func: Custom function for conflict resolution
            backup: Whether to create backup before merge
            
        Returns:
            Comprehensive merge statistics and results
            
        Example:
            >>> result = merger.merge_databases(
            ...     merge_strategy=MergeStrategy.SAFE,
            ...     conflict_resolution=ConflictResolutionMode.MERGE,
            ...     merge_keys=['customer_id', 'date']
            ... )
        """
        merge_keys = merge_keys or self.default_merge_keys
        
        self.logger.info(f"Début de la fusion avec stratégie: {merge_strategy.value}")
        
        # Structure pour les statistiques de fusion
        merge_stats = {
            'start_time': datetime.now(),
            'strategy': merge_strategy.value,
            'conflict_resolution': conflict_resolution.value,
            'phases_completed': [],
            'errors': [],
            'warnings': [],
            'backup_created': False
        }
        
        try:
            # Phase 1: Analyse pré-fusion
            if merge_strategy in [MergeStrategy.SAFE, MergeStrategy.OPTIMIZED]:
                self.logger.info("Phase 1: Analyse pré-fusion")
                compatibility_report = self.analyze_compatibility("source", "target")
                merge_stats['compatibility_report'] = compatibility_report
                merge_stats['phases_completed'].append('pre_analysis')
            
            # Phase 2: Création de sauvegarde
            if backup:
                self.logger.info("Phase 2: Création de sauvegarde")
                backup_info = self._create_backup()
                merge_stats['backup_info'] = backup_info
                merge_stats['backup_created'] = True
                merge_stats['phases_completed'].append('backup')
            
            # Phase 3: Harmonisation du schéma
            self.logger.info("Phase 3: Harmonisation du schéma")
            schema_merge_stats = self._merge_schemas(merge_strategy)
            merge_stats['schema_merge'] = schema_merge_stats
            merge_stats['phases_completed'].append('schema_harmonization')
            
            # Phase 4: Fusion des tables de dimension
            self.logger.info("Phase 4: Fusion des tables de dimension")
            dimension_merge_stats = self._merge_dimensions(conflict_resolution)
            merge_stats['dimension_merge'] = dimension_merge_stats
            merge_stats['phases_completed'].append('dimension_merge')
            
            # Phase 5: Fusion de la table des faits
            self.logger.info("Phase 5: Fusion de la table des faits")
            fact_merge_stats = self._merge_fact_table(
                conflict_resolution, merge_keys, custom_resolution_func
            )
            merge_stats['fact_merge'] = fact_merge_stats
            merge_stats['phases_completed'].append('fact_merge')
            
            # Phase 6: Validation post-fusion
            if merge_strategy == MergeStrategy.SAFE:
                self.logger.info("Phase 6: Validation post-fusion")
                validation_results = self._validate_merge_integrity()
                merge_stats['validation'] = validation_results
                merge_stats['phases_completed'].append('validation')
            
            # Phase 7: Optimisation finale
            if merge_strategy == MergeStrategy.OPTIMIZED:
                self.logger.info("Phase 7: Optimisation finale")
                optimization_stats = self._optimize_merged_database()
                merge_stats['optimization'] = optimization_stats
                merge_stats['phases_completed'].append('optimization')
            
            merge_stats['end_time'] = datetime.now()
            merge_stats['duration'] = (merge_stats['end_time'] - merge_stats['start_time']).total_seconds()
            merge_stats['success'] = True
            
            self.logger.info(f"Fusion terminée avec succés en {merge_stats['duration']:.2f}s")
            
        except Exception as e:
            merge_stats['success'] = False
            merge_stats['error'] = str(e)
            merge_stats['end_time'] = datetime.now()
            self.logger.error(f"échec de la fusion: {e}")
            
            # Tentative de restauration si sauvegarde disponible
            if backup and merge_stats['backup_created']:
                self.logger.info("Tentative de restauration depuis la sauvegarde")
                try:
                    self._restore_from_backup(merge_stats['backup_info'])
                    merge_stats['restored_from_backup'] = True
                except Exception as restore_error:
                    merge_stats['restore_error'] = str(restore_error)
                    self.logger.error(f"échec de la restauration: {restore_error}")
            
            raise
        
        return merge_stats
    
    def _load_database_metadata(self, connection: duckdb.DuckDBPyConnection, db_name: str) -> Dict[str, Any]:
        """
        Charge les métadonnées complétes d'une base de données.
        
        Args:
            connection: Connexion é la base de données
            db_name: Nom de la base pour identification
            
        Returns:
            Dictionnaire des métadonnées complétes
        """
        metadata = {
            'info': {
                'name': db_name,
                'timestamp': datetime.now().isoformat()
            },
            'schema': {},
            'dimensions': {},
            'statistics': {}
        }
        
        try:
            # Informations sur la table des faits
            fact_table_info = connection.execute("DESCRIBE fact_table").fetchall()
            metadata['schema']['fact_table'] = {
                col[0]: {'type': col[1], 'nullable': col[2] == 'YES'} 
                for col in fact_table_info
            }
            
            # Métadonnées depuis la table metadata
            metadata_df = connection.execute("SELECT * FROM metadata").fetchdf()
            metadata['schema']['metadata'] = metadata_df.to_dict('records')
            
            # Information sur les tables de dimension
            dimension_tables = connection.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'dim_%'"
            ).fetchall()
            
            for (table_name,) in dimension_tables:
                table_info = connection.execute(f"DESCRIBE {table_name}").fetchall()
                row_count = connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                
                metadata['dimensions'][table_name] = {
                    'schema': {col[0]: col[1] for col in table_info},
                    'row_count': row_count
                }
            
            # Statistiques générales
            fact_row_count = connection.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            metadata['statistics'] = {
                'fact_table_rows': fact_row_count,
                'dimension_tables_count': len(dimension_tables),
                'total_columns': len(fact_table_info)
            }
            
        except Exception as e:
            self.logger.warning(f"échec du chargement des métadonnées pour {db_name}: {e}")
            
        return metadata
    
    def _analyze_schema_compatibility(self, source_schema: Dict, target_schema: Dict) -> Dict[str, Any]:
        """
        Analyse la compatibilité des schémas entre source et cible.
        
        Args:
            source_schema: Schéma de la base source
            target_schema: Schéma de la base cible
            
        Returns:
            Rapport de compatibilité des schémas
        """
        compatibility = {
            'compatible_columns': [],
            'incompatible_columns': [],
            'missing_in_target': [],
            'missing_in_source': [],
            'type_conflicts': [],
            'compatibility_score': 0.0
        }
        
        source_fact = source_schema.get('fact_table', {})
        target_fact = target_schema.get('fact_table', {})
        
        source_columns = set(source_fact.keys())
        target_columns = set(target_fact.keys())
        
        # Colonnes manquantes
        compatibility['missing_in_target'] = list(source_columns - target_columns)
        compatibility['missing_in_source'] = list(target_columns - source_columns)
        
        # Analyse des colonnes communes
        common_columns = source_columns.intersection(target_columns)
        
        for column in common_columns:
            source_type = source_fact[column]['type']
            target_type = target_fact[column]['type']
            
            if self._are_types_compatible(source_type, target_type):
                compatibility['compatible_columns'].append(column)
            else:
                compatibility['incompatible_columns'].append(column)
                compatibility['type_conflicts'].append({
                    'column': column,
                    'source_type': source_type,
                    'target_type': target_type
                })
        
        # Calcul du score de compatibilité
        total_columns = len(source_columns.union(target_columns))
        if total_columns > 0:
            compatibility['compatibility_score'] = (
                len(compatibility['compatible_columns']) / total_columns
            ) * 10
        
        return compatibility
    
    def _analyze_dimension_compatibility(self, source_dims: Dict, target_dims: Dict) -> Dict[str, Any]:
        """
        Analyse la compatibilité des tables de dimension.
        
        Args:
            source_dims: Dimensions de la base source
            target_dims: Dimensions de la base cible
            
        Returns:
            Rapport de compatibilité des dimensions
        """
        compatibility = {
            'common_dimensions': [],
            'source_only': [],
            'target_only': [],
            'value_overlaps': {},
            'schema_conflicts': []
        }
        
        source_dim_names = set(source_dims.keys())
        target_dim_names = set(target_dims.keys())
        
        compatibility['common_dimensions'] = list(source_dim_names.intersection(target_dim_names))
        compatibility['source_only'] = list(source_dim_names - target_dim_names)
        compatibility['target_only'] = list(target_dim_names - source_dim_names)
        
        # Analyse des dimensions communes
        for dim_name in compatibility['common_dimensions']:
            source_schema = source_dims[dim_name]['schema']
            target_schema = target_dims[dim_name]['schema']
            
            if source_schema != target_schema:
                compatibility['schema_conflicts'].append({
                    'dimension': dim_name,
                    'source_schema': source_schema,
                    'target_schema': target_schema
                })
        
        return compatibility
    
    def _analyze_data_overlap(self) -> Dict[str, Any]:
        """
        Analyse le chevauchement des données entre les bases.
        
        Returns:
            Rapport de chevauchement des données
        """
        overlap_analysis = {
            'estimated_duplicates': 0,
            'unique_source_rows': 0,
            'unique_target_rows': 0,
            'sampling_performed': False
        }
        
        try:
            # échantillonnage pour estimation des doublons
            sample_size = min(1000, 
                self.source_conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0],
                self.target_conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            )
            
            if sample_size > 0:
                # échantillon de la source
                source_sample = self.source_conn.execute(
                    f"SELECT * FROM fact_table TABLESAMPLE SYSTEM(10) LIMIT {sample_size}"
                ).fetchdf()
                
                # Estimation des doublons basée sur les colonnes clés
                if not source_sample.empty and len(self.default_merge_keys) > 0:
                    # Estimation approximative (é améliorer avec des techniques plus sophistiquées)
                    overlap_analysis['sampling_performed'] = True
                    overlap_analysis['sample_size'] = len(source_sample)
            
        except Exception as e:
            self.logger.warning(f"Impossible d'analyser le chevauchement des données: {e}")
        
        return overlap_analysis
    
    def _generate_merge_recommendations(self) -> List[Dict[str, str]]:
        """
        Génére des recommandations pour la fusion basées sur l'analyse.
        
        Returns:
            Liste de recommandations
        """
        recommendations = [
            {
                'type': 'strategy',
                'message': "Utiliser MergeStrategy.SAFE pour une premiére fusion",
                'priority': 'high'
            },
            {
                'type': 'backup',
                'message': "Toujours créer une sauvegarde avant fusion",
                'priority': 'high'
            },
            {
                'type': 'validation',
                'message': "Effectuer une validation compléte aprés fusion",
                'priority': 'medium'
            }
        ]
        
        return recommendations
    
    def _create_backup(self) -> Dict[str, str]:
        """
        Crée une sauvegarde de la base cible avant fusion.
        
        Returns:
            Informations sur la sauvegarde crée
        """
        backup_info = {
            'timestamp': datetime.now().isoformat(),
            'backup_path': f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        }
        
        try:
            # Création d'une sauvegarde compléte
            backup_conn = duckdb.connect(backup_info['backup_path'])
            
            # Copie de toutes les tables
            tables = self.target_conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            
            for (table_name,) in tables:
                # Export vers la sauvegarde
                df = self.target_conn.execute(f"SELECT * FROM {table_name}").fetchdf()
                backup_conn.register('temp_table', df)
                backup_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_table")
                backup_conn.execute("DROP VIEW temp_table")
            
            backup_conn.close()
            backup_info['success'] = True
            
            self.logger.info(f"Sauvegarde crée: {backup_info['backup_path']}")
            
        except Exception as e:
            backup_info['success'] = False
            backup_info['error'] = str(e)
            self.logger.error(f"échec de la création de sauvegarde: {e}")
            raise
        
        return backup_info
    
    def _merge_schemas(self, strategy: MergeStrategy) -> Dict[str, Any]:
        """
        Fusionne les schémas des bases de données.
        
        Args:
            strategy: Stratégie de fusion
            
        Returns:
            Statistiques de fusion des schémas
        """
        schema_stats = {
            'columns_added': [],
            'columns_modified': [],
            'metadata_merged': False,
            'conflicts_resolved': []
        }
        
        try:
            # Chargement des métadonnées
            source_metadata = self.source_conn.execute("SELECT * FROM metadata").fetchdf()
            
            # Fusion des métadonnées dans la cible
            for _, row in source_metadata.iterrows():
                # Vérification si la colonne existe déjé dans la cible
                existing = self.target_conn.execute(
                    "SELECT COUNT(*) FROM metadata WHERE name = ?", [row['name']]
                ).fetchone()[0]
                
                if existing == 0:
                    # Nouvelle colonne é ajouter
                    insert_query = """
                        INSERT INTO metadata (name, label, python_type, sql_type, is_categorical)
                        VALUES (?, ?, ?, ?, ?)
                    """
                    self.target_conn.execute(insert_query, [
                        row['name'], row['label'], row['python_type'], 
                        row['sql_type'], row['is_categorical']
                    ])
                    schema_stats['columns_added'].append(row['name'])
                    
                    # Ajout de la colonne é fact_table si nécessaire
                    try:
                        alter_query = f"ALTER TABLE fact_table ADD COLUMN {row['name']} {row['sql_type']}"
                        self.target_conn.execute(alter_query)
                    except Exception as e:
                        # La colonne existe peut-étre déjé
                        self.logger.warning(f"Impossible d'ajouter la colonne {row['name']}: {e}")
                
                else:
                    # Colonne existante - vérification des conflits
                    existing_row = self.target_conn.execute(
                        "SELECT * FROM metadata WHERE name = ?", [row['name']]
                    ).fetchone()
                    
                    if existing_row[3] != row['sql_type']:  # Type différent
                        conflict_resolution = {
                            'column': row['name'],
                            'source_type': row['sql_type'],
                            'target_type': existing_row[3],
                            'resolution': 'kept_target'  # Par défaut, garder le type cible
                        }
                        schema_stats['conflicts_resolved'].append(conflict_resolution)
            
            schema_stats['metadata_merged'] = True
            
        except Exception as e:
            self.logger.error(f"échec de la fusion des schémas: {e}")
            raise
        
        return schema_stats
    
    def _merge_dimensions(self, conflict_resolution: ConflictResolutionMode) -> Dict[str, Any]:
        """
        Fusionne les tables de dimension.
        
        Args:
            conflict_resolution: Mode de résolution des conflits
            
        Returns:
            Statistiques de fusion des dimensions
        """
        dimension_stats = {
            'dimensions_processed': [],
            'values_added': {},
            'conflicts_resolved': {},
            'errors': []
        }
        
        try:
            # Récupération des tables de dimension de la source
            source_dimensions = self.source_conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'dim_%'"
            ).fetchall()
            
            for (dim_table,) in source_dimensions:
                try:
                    # Chargement des données de dimension depuis la source
                    source_dim_data = self.source_conn.execute(f"SELECT * FROM {dim_table}").fetchdf()
                    
                    # Vérification si la table existe dans la cible
                    target_table_exists = self.target_conn.execute(
                        f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{dim_table}'"
                    ).fetchone()[0] > 0
                    
                    if not target_table_exists:
                        # Création de la table de dimension dans la cible
                        create_query = f"""
                            CREATE TABLE {dim_table} (
                                value VARCHAR PRIMARY KEY,
                                label VARCHAR
                            )
                        """
                        self.target_conn.execute(create_query)
                    
                    # Fusion des valeurs de dimension
                    values_added = 0
                    for _, row in source_dim_data.iterrows():
                        try:
                            insert_query = f"""
                                INSERT INTO {dim_table} (value, label) 
                                VALUES (?, ?)
                                ON CONFLICT (value) DO NOTHING
                            """
                            result = self.target_conn.execute(insert_query, [row['value'], row['label']])
                            if self.target_conn.execute(f"SELECT changes()").fetchone()[0] > 0:
                                values_added += 1
                        except Exception as e:
                            dimension_stats['errors'].append(f"Erreur lors de l'insertion dans {dim_table}: {e}")
                    
                    dimension_stats['dimensions_processed'].append(dim_table)
                    dimension_stats['values_added'][dim_table] = values_added
                    
                except Exception as e:
                    dimension_stats['errors'].append(f"Erreur lors du traitement de {dim_table}: {e}")
                    
        except Exception as e:
            self.logger.error(f"échec de la fusion des dimensions: {e}")
            raise
        
        return dimension_stats
    
    def _merge_fact_table(self, 
                         conflict_resolution: ConflictResolutionMode,
                         merge_keys: List[str],
                         custom_resolution_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Fusionne les tables des faits.
        
        Args:
            conflict_resolution: Mode de résolution des conflits
            merge_keys: Clés pour identifier les doublons
            custom_resolution_func: Fonction personnalisée de résolution
            
        Returns:
            Statistiques de fusion de la table des faits
        """
        fact_stats = {
            'rows_processed': 0,
            'rows_added': 0,
            'rows_updated': 0,
            'conflicts_detected': 0,
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        try:
            # Chargement des données de la source par batches
            source_row_count = self.source_conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            fact_stats['rows_processed'] = source_row_count
            
            if conflict_resolution == ConflictResolutionMode.UNION:
                # Union simple - ajout direct de toutes les lignes
                source_data = self.source_conn.execute("SELECT * FROM fact_table").fetchdf()
                
                # Insertion des données dans la cible
                self.target_conn.register('temp_source_fact', source_data)
                self.target_conn.execute("INSERT INTO fact_table SELECT * FROM temp_source_fact")
                self.target_conn.execute("DROP VIEW temp_source_fact")
                
                fact_stats['rows_added'] = len(source_data)
                
            elif conflict_resolution == ConflictResolutionMode.MERGE:
                # Fusion avec gestion des doublons
                fact_stats.update(self._perform_upsert_merge(merge_keys))
                
            elif conflict_resolution in [ConflictResolutionMode.PRIORITIZE_SOURCE, ConflictResolutionMode.PRIORITIZE_TARGET]:
                # Fusion avec priorité
                fact_stats.update(self._perform_priority_merge(conflict_resolution, merge_keys))
                
            elif conflict_resolution == ConflictResolutionMode.CUSTOM and custom_resolution_func:
                # Fusion avec fonction personnalisée
                fact_stats.update(self._perform_custom_merge(custom_resolution_func, merge_keys))
            
        except Exception as e:
            self.logger.error(f"échec de la fusion de la table des faits: {e}")
            raise
        
        fact_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        return fact_stats
    
    def _perform_upsert_merge(self, merge_keys: List[str]) -> Dict[str, int]:
        """
        Effectue une fusion de type upsert.
        
        Args:
            merge_keys: Clés de fusion
            
        Returns:
            Statistiques de l'upsert
        """
        stats = {'rows_added': 0, 'rows_updated': 0}
        
        # Traitement par batches pour optimiser la mémoire
        offset = 0
        
        while True:
            # Chargement d'un batch de données source
            batch_data = self.source_conn.execute(
                f"SELECT * FROM fact_table LIMIT {self.batch_size} OFFSET {offset}"
            ).fetchdf()
            
            if batch_data.empty:
                break
                
            # Enregistrement du batch temporaire
            self.target_conn.register('temp_batch', batch_data)
            
            # Construction des conditions de fusion
            merge_condition = " AND ".join([f"f.{key} = t.{key}" for key in merge_keys])
            
            # Mise é jour des lignes existantes
            existing_columns = [col[0] for col in self.target_conn.execute("DESCRIBE fact_table").fetchall()]
            update_columns = [col for col in batch_data.columns if col in existing_columns and col not in merge_keys]
            
            if update_columns:
                set_clause = ", ".join([f"f.{col} = t.{col}" for col in update_columns])
                
                update_query = f"""
                    UPDATE fact_table f
                    SET {set_clause}
                    FROM temp_batch t
                    WHERE {merge_condition}
                """
                self.target_conn.execute(update_query)
                stats['rows_updated'] += self.target_conn.execute("SELECT changes()").fetchone()[0]
            
            # Insertion des nouvelles lignes
            insert_query = f"""
                INSERT INTO fact_table
                SELECT t.* FROM temp_batch t
                WHERE NOT EXISTS (
                    SELECT 1 FROM fact_table f
                    WHERE {merge_condition}
                )
            """
            self.target_conn.execute(insert_query)
            stats['rows_added'] += self.target_conn.execute("SELECT changes()").fetchone()[0]
            
            # Nettoyage
            self.target_conn.execute("DROP VIEW temp_batch")
            offset += self.batch_size
        
        return stats
    
    def _perform_priority_merge(self, 
                               conflict_resolution: ConflictResolutionMode,
                               merge_keys: List[str]) -> Dict[str, int]:
        """
        Effectue une fusion avec priorité source ou cible.
        
        Args:
            conflict_resolution: Mode de résolution (PRIORITIZE_SOURCE/TARGET)
            merge_keys: Clés de fusion
            
        Returns:
            Statistiques de la fusion avec priorité
        """
        stats = {'rows_added': 0, 'rows_updated': 0}
        
        if conflict_resolution == ConflictResolutionMode.PRIORITIZE_SOURCE:
            # Source a la priorité - remplacer les conflits
            stats.update(self._perform_upsert_merge(merge_keys))
        else:
            # Cible a la priorité - ignorer les conflits
            source_data = self.source_conn.execute("SELECT * FROM fact_table").fetchdf()
            self.target_conn.register('temp_source', source_data)
            
            merge_condition = " AND ".join([f"f.{key} = t.{key}" for key in merge_keys])
            
            insert_query = f"""
                INSERT INTO fact_table
                SELECT t.* FROM temp_source t
                WHERE NOT EXISTS (
                    SELECT 1 FROM fact_table f
                    WHERE {merge_condition}
                )
            """
            self.target_conn.execute(insert_query)
            stats['rows_added'] = self.target_conn.execute("SELECT changes()").fetchone()[0]
            
            self.target_conn.execute("DROP VIEW temp_source")
        
        return stats
    
    def _perform_custom_merge(self, 
                             custom_func: Callable,
                             merge_keys: List[str]) -> Dict[str, int]:
        """
        Effectue une fusion avec fonction personnalisée.
        
        Args:
            custom_func: Fonction personnalisée de résolution
            merge_keys: Clés de fusion
            
        Returns:
            Statistiques de la fusion personnalisée
        """
        stats = {'rows_added': 0, 'rows_updated': 0}
        
        # Implémentation basique - é étendre selon les besoins
        # La fonction personnalisée devrait prendre deux DataFrames en entrée
        # et retourner le DataFrame fusionné
        
        source_data = self.source_conn.execute("SELECT * FROM fact_table").fetchdf()
        target_data = self.target_conn.execute("SELECT * FROM fact_table").fetchdf()
        
        merged_data = custom_func(source_data, target_data, merge_keys)
        
        # Remplacement de la table cible
        self.target_conn.execute("DELETE FROM fact_table")
        self.target_conn.register('merged_result', merged_data)
        self.target_conn.execute("INSERT INTO fact_table SELECT * FROM merged_result")
        self.target_conn.execute("DROP VIEW merged_result")
        
        stats['rows_added'] = len(merged_data)
        
        return stats
    
    def _validate_merge_integrity(self) -> Dict[str, Any]:
        """
        Valide l'intégrité aprés fusion.
        
        Returns:
            Résultats de validation
        """
        validation_results = {
            'referential_integrity': True,
            'data_quality_checks': [],
            'dimension_consistency': True,
            'metadata_consistency': True,
            'errors': []
        }
        
        try:
            # Vérification de l'intégrité référentielle
            validation_results['referential_integrity'] = self._check_referential_integrity()
            
            # Vérification de la cohérence des dimensions
            validation_results['dimension_consistency'] = self._check_dimension_consistency()
            
            # Vérification de la cohérence des métadonnées
            validation_results['metadata_consistency'] = self._check_metadata_consistency()
            
        except Exception as e:
            validation_results['errors'].append(str(e))
            self.logger.error(f"Erreur lors de la validation: {e}")
        
        return validation_results
    
    def _optimize_merged_database(self) -> Dict[str, Any]:
        """
        Optimise la base de données fusionnée.
        
        Returns:
            Statistiques d'optimisation
        """
        optimization_stats = {
            'tables_analyzed': [],
            'indexes_created': [],
            'statistics_updated': False
        }
        
        try:
            # Analyse de toutes les tables
            tables = self.target_conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            
            for (table_name,) in tables:
                self.target_conn.execute(f"ANALYZE {table_name}")
                optimization_stats['tables_analyzed'].append(table_name)
            
            optimization_stats['statistics_updated'] = True
            
            # Ici, on pourrait intégrer avec l'IndexManager pour créer des index optimaux
            # self._create_optimal_indexes()
            
        except Exception as e:
            self.logger.error(f"Erreur lors de l'optimisation: {e}")
            
        return optimization_stats
    
    def _restore_from_backup(self, backup_info: Dict[str, str]) -> None:
        """
        Restaure depuis une sauvegarde.
        
        Args:
            backup_info: Informations de la sauvegarde
        """
        if not backup_info.get('success', False):
            raise ValueError("Sauvegarde invalide pour la restauration")
        
        backup_path = backup_info['backup_path']
        if not os.path.exists(backup_path):
            raise FileNotFoundError(f"Fichier de sauvegarde introuvable: {backup_path}")
        
        # Connexion é la sauvegarde
        backup_conn = duckdb.connect(backup_path)
        
        try:
            # Restauration de toutes les tables
            tables = backup_conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            
            for (table_name,) in tables:
                # Suppression et recréation de la table
                self.target_conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                
                backup_data = backup_conn.execute(f"SELECT * FROM {table_name}").fetchdf()
                self.target_conn.register(f'restore_{table_name}', backup_data)
                self.target_conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM restore_{table_name}")
                self.target_conn.execute(f"DROP VIEW restore_{table_name}")
            
            self.logger.info("Restauration depuis sauvegarde réussie")
            
        finally:
            backup_conn.close()
    
    def _are_types_compatible(self, type1: str, type2: str) -> bool:
        """
        Vérifie si deux types de données sont compatibles.
        
        Args:
            type1: Premier type
            type2: Deuxiéme type
            
        Returns:
            True si les types sont compatibles
        """
        # Mapping des types compatibles
        compatible_types = {
            'INTEGER': ['BIGINT', 'SMALLINT', 'TINYINT'],
            'BIGINT': ['INTEGER', 'SMALLINT', 'TINYINT'],
            'DOUBLE': ['FLOAT', 'REAL'],
            'VARCHAR': ['TEXT', 'STRING'],
            'TIMESTAMP': ['DATETIME'],
        }
        
        if type1 == type2:
            return True
        
        return type2 in compatible_types.get(type1, []) or type1 in compatible_types.get(type2, [])
    
    def _check_referential_integrity(self) -> bool:
        """
        Vérifie l'intégrité référentielle aprés fusion.
        
        Returns:
            True si l'intégrité référentielle est respectée
        """
        try:
            # Vérification que toutes les valeurs de dimension existent
            metadata = self.target_conn.execute(
                "SELECT name FROM metadata WHERE is_categorical = true"
            ).fetchall()
            
            for (column_name,) in metadata:
                dim_table = f"dim_{column_name}"
                
                # Vérification des valeurs orphelines
                orphaned = self.target_conn.execute(f"""
                    SELECT COUNT(*) FROM fact_table f
                    WHERE f.{column_name} IS NOT NULL
                    AND f.{column_name} NOT IN (SELECT value FROM {dim_table})
                """).fetchone()[0]
                
                if orphaned > 0:
                    self.logger.warning(f"Valeurs orphelines détectées dans {column_name}: {orphaned}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification de l'intégrité référentielle: {e}")
            return False
    
    def _check_dimension_consistency(self) -> bool:
        """
        Vérifie la cohérence des tables de dimension.
        
        Returns:
            True si les dimensions sont cohérentes
        """
        try:
            # Vérification que toutes les tables de dimension sont cohérentes
            dimensions = self.target_conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_name LIKE 'dim_%'"
            ).fetchall()
            
            for (dim_table,) in dimensions:
                # Vérification des doublons
                duplicates = self.target_conn.execute(f"""
                    SELECT COUNT(*) - COUNT(DISTINCT value) FROM {dim_table}
                """).fetchone()[0]
                
                if duplicates > 0:
                    self.logger.warning(f"Doublons détectés dans {dim_table}: {duplicates}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification de la cohérence des dimensions: {e}")
            return False
    
    def _check_metadata_consistency(self) -> bool:
        """
        Vérifie la cohérence des métadonnées.
        
        Returns:
            True si les métadonnées sont cohérentes
        """
        try:
            # Vérification que tous les colonnes de fact_table ont des métadonnées
            fact_columns = set([
                col[0] for col in self.target_conn.execute("DESCRIBE fact_table").fetchall()
            ])
            
            metadata_columns = set([
                col[0] for col in self.target_conn.execute("SELECT name FROM metadata").fetchall()
            ])
            
            missing_metadata = fact_columns - metadata_columns
            if missing_metadata:
                self.logger.warning(f"Métadonnées manquantes pour les colonnes: {missing_metadata}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la vérification de la cohérence des métadonnées: {e}")
            return False