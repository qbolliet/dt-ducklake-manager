# Gestionnaire principal pour les opérations sur la base de données
import os
import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union
import duckdb
from datetime import datetime

from ..builders.tables import DuckdbTablesBuilder
from ..builders.indexer import IndexManager
from ..operations.updater import DatabaseUpdater
from ..operations.deleter import DatabaseDeleter
from ..storage.loader import Loader
from ..storage.saver import Saver
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

class DatabaseManager:
    """
    Main manager class for DuckDB database operations.
    
    Provides a unified interface for creating, updating, deleting,
    and managing DuckDB databases with the fact/dimension/metadata schema.
    """
    
    def __init__(self, 
                 database_path: Optional[Union[str, Path]] = None,
                 database_name: Optional[str] = None,
                 config: Optional[Dict] = None):
        """
        Initialize the DatabaseManager.
        
        Args:
            database_path: Path to the database directory
            database_name: Name of the database (without extension)
            config: Configuration dictionary
        """
        # Configuration par défaut
        self.config = config or self._load_default_config()
        
        # Définition du chemin de la base de données
        if database_path:
            self.database_path = Path(database_path)
        else:
            self.database_path = Path(self.config.get('DATABASE_PATH', './databases'))
        
        # Création du répertoire si nécessaire
        self.database_path.mkdir(parents=True, exist_ok=True)
        
        # Nom de la base de données
        self.database_name = database_name or self.config.get('DATABASE_NAME', 'default')
        
        # Chemin complet du fichier de base de données
        self.db_file = self.database_path / f"{self.database_name}.duckdb"
        
        # Initialisation du logger
        log_dir = self.database_path / "logs"
        log_dir.mkdir(exist_ok=True)
        self.logger = _init_logger(
            filename=log_dir / f"{self.database_name}_manager.log"
        )
        
        # Connection à la base de données
        self.conn = None
        self._connect()
        
        # Initialisation des composants
        self.index_manager = None
        self.updater = None
        self.deleter = None
        
        # Cache des métadonnées
        self._metadata_cache = {}
    
    def _load_default_config(self) -> Dict:
        """Load default configuration."""
        config_path = FILE_PATH.parents[2] / "config.yaml"
        
        if config_path.exists():
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f)
        
        # Configuration par défaut si pas de fichier
        return {
            'DATABASE_PATH': './databases',
            'DATABASE_NAME': 'default',
            'CATEGORICAL_THRESHOLD': 50,
            'INDEX_CONFIG': {
                'auto_index': True,
                'primary_indexes': [],
                'composite_indexes': []
            }
        }
    
    def _connect(self) -> None:
        """Establish connection to the database."""
        try:
            self.conn = duckdb.connect(str(self.db_file))
            self.logger.info(f"Connected to database: {self.db_file}")
            
            # Initialisation des composants avec la connexion
            self.index_manager = IndexManager(self.conn)
            self.updater = DatabaseUpdater(
                self.conn,
                categorical_threshold=self.config.get('CATEGORICAL_THRESHOLD', 50)
            )
            self.deleter = DatabaseDeleter(self.conn)
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
    
    def create_database(self, 
                       df: pd.DataFrame,
                       column_labels: Optional[Dict[str, str]] = None,
                       index_config: Optional[Dict] = None) -> Dict:
        """
        Create a new database with the fact/dimension/metadata schema.
        
        Args:
            df: Input DataFrame
            column_labels: Optional mapping of column names to labels
            index_config: Optional index configuration
            
        Returns:
            Dictionary with creation statistics
        """
        stats = {
            'database': str(self.db_file),
            'created_at': datetime.now().isoformat(),
            'rows': len(df),
            'columns': len(df.columns),
            'dimensions': 0,
            'indexes': 0
        }
        
        try:
            # Création du schéma
            builder = DuckdbTablesBuilder(
                df=df,
                categorical_threshold=self.config.get('CATEGORICAL_THRESHOLD', 50),
                connection=self.conn
            )
            
            # Construction des tables
            builder.build_duckdb_schema(column_labels=column_labels)
            
            # Récupération des statistiques
            stats['dimensions'] = len(builder.dimension_tables)
            
            # Création des index si configuré
            if index_config or self.config.get('INDEX_CONFIG', {}).get('auto_index'):
                stats['indexes'] = self.create_indexes(
                    index_config or self.config.get('INDEX_CONFIG')
                )
            
            # Sauvegarde des métadonnées de création
            self._save_database_metadata(stats)
            
            self.logger.info(f"Database created successfully: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to create database: {e}")
            raise
    
    def update_database(self,
                       df: pd.DataFrame,
                       update_mode: str = 'upsert',
                       merge_keys: Optional[List[str]] = None) -> Dict:
        """
        Update existing database with new data.
        
        Args:
            df: New data DataFrame
            update_mode: 'append', 'upsert', or 'replace'
            merge_keys: Keys for upsert operation
            
        Returns:
            Dictionary with update statistics
        """
        if not self.database_exists():
            raise ValueError(f"Database {self.db_file} does not exist")
        
        stats = self.updater.merge_data(
            new_df=df,
            merge_keys=merge_keys,
            update_mode=update_mode
        )
        
        # Mise à jour de l'historique
        self.updater.track_update_history(f"update_{update_mode}", stats)
        
        return stats
    
    def delete_data(self,
                   rows_filter: Optional[str] = None,
                   structured_filters: Optional[List[Dict]] = None,
                   columns_to_delete: Optional[List[str]] = None) -> Dict:
        """
        Delete data from the database.
        
        Args:
            rows_filter: SQL WHERE clause for row deletion
            structured_filters: Structured filters for row deletion
            columns_to_delete: List of columns to remove
            
        Returns:
            Dictionary with deletion statistics
        """
        stats = {
            'rows_deleted': 0,
            'columns_deleted': [],
            'timestamp': datetime.now().isoformat()
        }
        
        # Suppression de lignes
        if rows_filter or structured_filters:
            stats['rows_deleted'] = self.deleter.delete_rows(
                filters=rows_filter,
                structured_filters=structured_filters
            )
        
        # Suppression de colonnes
        if columns_to_delete:
            deletion_result = self.deleter.delete_columns(columns_to_delete)
            stats.update(deletion_result)
        
        # Mise à jour de l'historique
        self.updater.track_update_history("delete", stats)
        
        return stats
    
    def create_indexes(self, index_config: Dict) -> int:
        """
        Create indexes based on configuration.
        
        Args:
            index_config: Index configuration dictionary
            
        Returns:
            Number of indexes created
        """
        count = 0
        
        # Index sur les colonnes primaires
        if 'primary_indexes' in index_config:
            for column in index_config['primary_indexes']:
                try:
                    self.index_manager.create_index(
                        'fact_table',
                        f'idx_fact_{column}',
                        [column]
                    )
                    count += 1
                except Exception as e:
                    self.logger.warning(f"Could not create index for {column}: {e}")
        
        # Index composites
        if 'composite_indexes' in index_config:
            for idx in index_config['composite_indexes']:
                try:
                    self.index_manager.create_index(
                        'fact_table',
                        idx['name'],
                        idx['columns']
                    )
                    count += 1
                except Exception as e:
                    self.logger.warning(f"Could not create composite index {idx['name']}: {e}")
        
        # Analyse des tables pour optimisation
        self.index_manager.analyze_table('fact_table')
        
        return count
    
    def optimize_database(self) -> Dict:
        """
        Optimize database performance.
        
        Returns:
            Dictionary with optimization statistics
        """
        stats = {
            'vacuum': False,
            'analyze': False,
            'checkpoint': False,
            'indexes_rebuilt': 0
        }
        
        try:
            # VACUUM pour récupérer l'espace
            self.conn.execute("VACUUM")
            stats['vacuum'] = True
            
            # ANALYZE pour mettre à jour les statistiques
            self.conn.execute("ANALYZE")
            stats['analyze'] = True
            
            # CHECKPOINT pour persister les changements
            self.conn.execute("CHECKPOINT")
            stats['checkpoint'] = True
            
            # Reconstruction des index si nécessaire
            indexes = self.index_manager.list_indexes()
            for index in indexes:
                try:
                    # Drop et recréation de l'index
                    self.index_manager.drop_index(index['index_name'])
                    # Récréation basée sur la définition SQL stockée
                    self.conn.execute(index['sql'])
                    stats['indexes_rebuilt'] += 1
                except:
                    pass
            
            self.logger.info(f"Database optimized: {stats}")
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to optimize database: {e}")
            raise
    
    def get_database_info(self) -> Dict:
        """
        Get comprehensive database information.
        
        Returns:
            Dictionary with database statistics and metadata
        """
        info = {
            'database': str(self.db_file),
            'size_bytes': self.db_file.stat().st_size if self.db_file.exists() else 0,
            'tables': {},
            'indexes': [],
            'metadata': {}
        }
        
        try:
            # Information sur les tables
            tables_query = """
                SELECT table_name, estimated_size 
                FROM duckdb_tables()
            """
            tables = self.conn.execute(tables_query).fetchall()
            
            for table_name, size in tables:
                # Comptage des lignes
                count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                
                # Colonnes
                columns = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
                
                info['tables'][table_name] = {
                    'rows': count,
                    'estimated_size': size,
                    'columns': len(columns)
                }
            
            # Information sur les index
            info['indexes'] = self.index_manager.list_indexes()
            
            # Métadonnées stockées
            if self._metadata_cache:
                info['metadata'] = self._metadata_cache
            
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get database info: {e}")
            return info
    
    def export_database(self, 
                       export_path: Optional[Path] = None,
                       format: str = 'parquet') -> Path:
        """
        Export database tables to files.
        
        Args:
            export_path: Directory for export
            format: Export format ('parquet', 'csv', 'json')
            
        Returns:
            Path to export directory
        """
        if not export_path:
            export_path = self.database_path / "exports" / self.database_name
        
        export_path = Path(export_path)
        export_path.mkdir(parents=True, exist_ok=True)
        
        # Export de chaque table
        tables = self.conn.execute("SHOW TABLES").fetchall()
        
        for (table_name,) in tables:
            output_file = export_path / f"{table_name}.{format}"
            
            if format == 'parquet':
                query = f"COPY {table_name} TO '{output_file}' (FORMAT PARQUET)"
            elif format == 'csv':
                query = f"COPY {table_name} TO '{output_file}' (FORMAT CSV, HEADER)"
            elif format == 'json':
                query = f"COPY {table_name} TO '{output_file}' (FORMAT JSON)"
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.conn.execute(query)
            self.logger.info(f"Exported {table_name} to {output_file}")
        
        # Export des métadonnées
        metadata_file = export_path / "database_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.get_database_info(), f, indent=2, default=str)
        
        return export_path
    
    def database_exists(self) -> bool:
        """Check if database exists and has the expected schema."""
        if not self.db_file.exists():
            return False
        
        try:
            # Vérification de l'existence des tables principales
            tables = self.conn.execute("SHOW TABLES").fetchdf()['name'].tolist()
            required_tables = ['fact_table', 'metadata']
            
            return all(table in tables for table in required_tables)
            
        except:
            return False
    
    def _save_database_metadata(self, metadata: Dict) -> None:
        """Save database metadata to file."""
        metadata_file = self.database_path / f"{self.database_name}_metadata.json"
        
        # Chargement des métadonnées existantes
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {'created_at': metadata.get('created_at'), 'history': []}
        
        # Ajout à l'historique
        existing['history'].append(metadata)
        existing['last_updated'] = datetime.now().isoformat()
        
        # Sauvegarde
        with open(metadata_file, 'w') as f:
            json.dump(existing, f, indent=2, default=str)
        
        self._metadata_cache = existing
    
    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            self.logger.info(f"Closed connection to {self.db_file}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()