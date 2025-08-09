# Module pour la mise à jour incrémentale des tables
import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import duckdb
import hashlib
import json
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from ..builders.schema import SchemaBuilder
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

class ConflictResolutionStrategy(Enum):
    """
    Stratégies de résolution de conflits de types.
    """
    PROMOTE_TYPES = "promote_types"  # Promotion automatique des types (int -> float, etc.)
    STRICT_VALIDATION = "strict_validation"  # Validation stricte, échec si conflit
    COERCE_TO_STRING = "coerce_to_string"  # Conversion forcée vers string
    USER_DEFINED = "user_defined"  # Fonction personnalisée de résolution

class DataValidationLevel(Enum):
    """
    Niveaux de validation des données.
    """
    NONE = "none"
    BASIC = "basic"      # Validation des types et NULL
    STRICT = "strict"    # Validation complète avec contraintes
    CUSTOM = "custom"    # Validation personnalisée

@dataclass
class ValidationRule:
    """
    Règle de validation pour une colonne.
    """
    column_name: str
    rule_type: str  # 'not_null', 'range', 'regex', 'custom'
    parameters: Dict[str, Any]
    error_message: str
    severity: str  # 'error', 'warning'

@dataclass
class ConflictResolution:
    """
    Configuration de résolution de conflit.
    """
    column_name: str
    source_type: str
    target_type: str
    resolution_strategy: str
    custom_function: Optional[Callable] = None

class DataQualityChecker:
    """
    Composant de vérification de qualité des données.
    """
    
    def __init__(self, validation_rules: List[ValidationRule] = None):
        self.validation_rules = validation_rules or []
        self.default_rules = self._create_default_rules()
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          validation_level: DataValidationLevel = DataValidationLevel.BASIC) -> Dict[str, Any]:
        """
        Valide un DataFrame selon les règles définies.
        
        Args:
            df: DataFrame à valider
            validation_level: Niveau de validation
            
        Returns:
            Résultats de validation
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'column_stats': {},
            'quality_score': 0.0
        }
        
        if validation_level == DataValidationLevel.NONE:
            validation_results['quality_score'] = 1.0
            return validation_results
        
        # Validation basique
        if validation_level in [DataValidationLevel.BASIC, DataValidationLevel.STRICT]:
            validation_results.update(self._basic_validation(df))
        
        # Validation stricte
        if validation_level == DataValidationLevel.STRICT:
            validation_results.update(self._strict_validation(df))
        
        # Validation personnalisée
        if validation_level == DataValidationLevel.CUSTOM:
            validation_results.update(self._custom_validation(df))
        
        # Calcul du score de qualité
        validation_results['quality_score'] = self._calculate_quality_score(validation_results)
        
        return validation_results
    
    def _create_default_rules(self) -> List[ValidationRule]:
        """
        Crée des règles de validation par défaut.
        
        Returns:
            Liste des règles par défaut
        """
        return [
            ValidationRule(
                column_name="*",
                rule_type="excessive_nulls",
                parameters={"max_null_ratio": 0.8},
                error_message="Trop de valeurs NULL dans la colonne",
                severity="warning"
            )
        ]
    
    def _basic_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validation basique du DataFrame.
        
        Args:
            df: DataFrame à valider
            
        Returns:
            Résultats de validation basique
        """
        results = {'errors': [], 'warnings': [], 'column_stats': {}}
        
        for column in df.columns:
            null_count = df[column].isnull().sum()
            null_ratio = null_count / len(df)
            
            results['column_stats'][column] = {
                'null_count': null_count,
                'null_ratio': null_ratio,
                'dtype': str(df[column].dtype),
                'unique_count': df[column].nunique()
            }
            
            # Vérification du ratio de NULL
            if null_ratio > 0.8:
                results['warnings'].append(
                    f"Colonne {column}: {null_ratio:.1%} de valeurs NULL"
                )
        
        return results
    
    def _strict_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validation stricte du DataFrame.
        
        Args:
            df: DataFrame à valider
            
        Returns:
            Résultats de validation stricte
        """
        results = {'errors': [], 'warnings': []}
        
        # Application des règles personnalisées
        for rule in self.validation_rules:
            if rule.column_name == "*" or rule.column_name in df.columns:
                columns_to_check = df.columns if rule.column_name == "*" else [rule.column_name]
                
                for column in columns_to_check:
                    if not self._apply_validation_rule(df[column], rule):
                        if rule.severity == "error":
                            results['errors'].append(f"{column}: {rule.error_message}")
                        else:
                            results['warnings'].append(f"{column}: {rule.error_message}")
        
        return results
    
    def _custom_validation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validation personnalisée du DataFrame.
        
        Args:
            df: DataFrame à valider
            
        Returns:
            Résultats de validation personnalisée
        """
        # À implémenter selon les besoins spécifiques
        return {'errors': [], 'warnings': []}
    
    def _apply_validation_rule(self, series: pd.Series, rule: ValidationRule) -> bool:
        """
        Applique une règle de validation à une série.
        
        Args:
            series: Série à valider
            rule: Règle à appliquer
            
        Returns:
            True si la règle est respectée
        """
        if rule.rule_type == "not_null":
            return series.isnull().sum() == 0
        
        elif rule.rule_type == "excessive_nulls":
            max_ratio = rule.parameters.get("max_null_ratio", 0.5)
            null_ratio = series.isnull().sum() / len(series)
            return null_ratio <= max_ratio
        
        elif rule.rule_type == "range":
            min_val = rule.parameters.get("min")
            max_val = rule.parameters.get("max")
            if min_val is not None and max_val is not None:
                return series.between(min_val, max_val, inclusive="both").all()
        
        return True
    
    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> float:
        """
        Calcule un score de qualité basé sur les résultats de validation.
        
        Args:
            validation_results: Résultats de validation
            
        Returns:
            Score entre 0.0 et 1.0
        """
        error_count = len(validation_results['errors'])
        warning_count = len(validation_results['warnings'])
        
        # Score basé sur le nombre d'erreurs et avertissements
        penalty = error_count * 0.2 + warning_count * 0.1
        score = max(0.0, 1.0 - penalty)
        
        return score

class DatabaseUpdater:
    """
    A class to handle incremental updates to DuckDB databases.
    
    Supports adding, updating, and merging data while maintaining
    consistency between fact, dimension, and metadata tables.
    """
    
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None,
                 conflict_resolution_strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.PROMOTE_TYPES,
                 validation_level: DataValidationLevel = DataValidationLevel.BASIC,
                 max_workers: int = 4,
                 batch_size: int = 10000):
        """
        Initialize the DatabaseUpdater.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Maximum unique values for categorical columns
            log_filename: Path to log file
            conflict_resolution_strategy: Strategy for resolving type conflicts
            validation_level: Level of data validation
            max_workers: Maximum number of parallel workers
            batch_size: Size of batches for processing
        """
        self.conn = connection
        self.categorical_threshold = categorical_threshold
        self.conflict_resolution_strategy = conflict_resolution_strategy
        self.validation_level = validation_level
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/database_updater.log")
        self.logger = _init_logger(filename=log_filename)
        
        # Cache pour les métadonnées
        self._metadata_cache = None
        self._dimension_cache = {}
        
        # Composants de validation et résolution de conflits
        self.data_quality_checker = DataQualityChecker()
        self.conflict_resolutions = {}
        
        # Verrou pour les opérations thread-safe
        self._lock = threading.Lock()
    
    def load_current_metadata(self) -> pd.DataFrame:
        """
        Load current metadata from the database.
        
        Returns:
            DataFrame containing current metadata
        """
        if self._metadata_cache is None:
            try:
                self._metadata_cache = self.conn.execute("SELECT * FROM metadata").fetchdf()
            except Exception as e:
                self.logger.error(f"Failed to load metadata: {e}")
                raise
        
        return self._metadata_cache
    
    def validate_and_cleanse_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and cleanse incoming data.
        
        Args:
            df: DataFrame to validate and cleanse
            
        Returns:
            Tuple of (cleansed_df, validation_results)
            
        Example:
            >>> updater = DatabaseUpdater(connection)
            >>> clean_df, results = updater.validate_and_cleanse_data(raw_df)
            >>> if results['quality_score'] < 0.8:
            ...     print("Data quality issues detected")
        """
        self.logger.info("Validation et nettoyage des données")
        
        # Validation initiale
        validation_results = self.data_quality_checker.validate_dataframe(df, self.validation_level)
        
        # Nettoyage des données
        cleansed_df = df.copy()
        
        # Nettoyage basique
        cleansed_df = self._basic_data_cleansing(cleansed_df)
        
        # Nettoyage avancé si validation stricte
        if self.validation_level == DataValidationLevel.STRICT:
            cleansed_df = self._advanced_data_cleansing(cleansed_df, validation_results)
        
        # Re-validation après nettoyage
        final_validation = self.data_quality_checker.validate_dataframe(
            cleansed_df, DataValidationLevel.BASIC
        )
        
        validation_results['post_cleansing'] = final_validation
        validation_results['improvement'] = (
            final_validation['quality_score'] - validation_results['quality_score']
        )
        
        return cleansed_df, validation_results
    
    def resolve_type_conflicts(self, df: pd.DataFrame, 
                              schema_changes: Dict) -> Tuple[pd.DataFrame, Dict[str, ConflictResolution]]:
        """
        Resolve type conflicts between new data and existing schema.
        
        Args:
            df: DataFrame with potential type conflicts
            schema_changes: Detected schema changes
            
        Returns:
            Tuple of (resolved_df, conflict_resolutions)
        """
        self.logger.info("Résolution des conflits de types")
        
        resolved_df = df.copy()
        conflict_resolutions = {}
        
        for type_change in schema_changes.get('type_changes', []):
            column = type_change['column']
            old_type = type_change['old_type']
            new_type = type_change['new_type']
            
            resolution = ConflictResolution(
                column_name=column,
                source_type=new_type,
                target_type=old_type,
                resolution_strategy=self.conflict_resolution_strategy.value
            )
            
            # Application de la stratégie de résolution
            if self.conflict_resolution_strategy == ConflictResolutionStrategy.PROMOTE_TYPES:
                resolved_df[column] = self._promote_types(resolved_df[column], old_type, new_type)
                
            elif self.conflict_resolution_strategy == ConflictResolutionStrategy.COERCE_TO_STRING:
                resolved_df[column] = resolved_df[column].astype(str)
                
            elif self.conflict_resolution_strategy == ConflictResolutionStrategy.STRICT_VALIDATION:
                # Validation stricte - échec si conflit non résolvable
                if not self._can_convert_type(resolved_df[column], old_type):
                    raise ValueError(f"Conflit de type non résolvable pour {column}: {new_type} -> {old_type}")
                    
            elif self.conflict_resolution_strategy == ConflictResolutionStrategy.USER_DEFINED:
                if column in self.conflict_resolutions:
                    custom_func = self.conflict_resolutions[column].custom_function
                    if custom_func:
                        resolved_df[column] = custom_func(resolved_df[column], old_type, new_type)
            
            conflict_resolutions[column] = resolution
        
        return resolved_df, conflict_resolutions
    
    def parallel_dimension_update(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Update dimension tables in parallel for better performance.
        
        Args:
            df: DataFrame containing dimension values
            
        Returns:
            Statistics of dimension updates
        """
        self.logger.info("Mise à jour parallèle des dimensions")
        
        metadata = self.load_current_metadata()
        categorical_columns = metadata[metadata['is_categorical'] == True]['name'].tolist()
        
        # Filtrage des colonnes catégorielles présentes dans le DataFrame
        dimensions_to_update = [col for col in categorical_columns if col in df.columns]
        
        if not dimensions_to_update:
            return {}
        
        update_stats = {}
        
        # Traitement parallèle des dimensions
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Soumission des tâches
            future_to_dimension = {
                executor.submit(self._update_single_dimension, df[dim].unique(), dim): dim 
                for dim in dimensions_to_update
            }
            
            # Récupération des résultats
            for future in as_completed(future_to_dimension):
                dimension = future_to_dimension[future]
                try:
                    values_added = future.result()
                    update_stats[dimension] = values_added
                except Exception as e:
                    self.logger.error(f"Erreur lors de la mise à jour de {dimension}: {e}")
                    update_stats[dimension] = 0
        
        return update_stats
    
    def memory_efficient_merge(self, df: pd.DataFrame, 
                              merge_keys: List[str],
                              chunk_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform memory-efficient merge for large DataFrames.
        
        Args:
            df: Large DataFrame to merge
            merge_keys: Keys for merging
            chunk_size: Size of chunks for processing
            
        Returns:
            Merge statistics
        """
        chunk_size = chunk_size or self.batch_size
        
        self.logger.info(f"Fusion optimisée mémoire par chunks de {chunk_size}")
        
        merge_stats = {
            'total_rows': len(df),
            'chunks_processed': 0,
            'rows_added': 0,
            'rows_updated': 0,
            'memory_usage': []
        }
        
        # Traitement par chunks
        for chunk_start in range(0, len(df), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(df))
            chunk_df = df.iloc[chunk_start:chunk_end].copy()
            
            # Mesure de l'utilisation mémoire
            memory_before = chunk_df.memory_usage(deep=True).sum()
            
            # Traitement du chunk
            chunk_stats = self._process_chunk(chunk_df, merge_keys)
            
            # Mise à jour des statistiques
            merge_stats['chunks_processed'] += 1
            merge_stats['rows_added'] += chunk_stats.get('rows_added', 0)
            merge_stats['rows_updated'] += chunk_stats.get('rows_updated', 0)
            merge_stats['memory_usage'].append(memory_before)
            
            # Nettoyage explicite
            del chunk_df
            
        # Statistiques finales
        merge_stats['avg_memory_per_chunk'] = np.mean(merge_stats['memory_usage'])
        merge_stats['peak_memory'] = max(merge_stats['memory_usage'])
        
        return merge_stats
    
    def _basic_data_cleansing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Nettoyage basique des données.
        
        Args:
            df: DataFrame à nettoyer
            
        Returns:
            DataFrame nettoyé
        """
        cleansed_df = df.copy()
        
        # Suppression des doublons
        initial_count = len(cleansed_df)
        cleansed_df = cleansed_df.drop_duplicates()
        if len(cleansed_df) != initial_count:
            self.logger.info(f"Suppression de {initial_count - len(cleansed_df)} doublons")
        
        # Nettoyage des chaînes de caractères
        for column in cleansed_df.columns:
            if cleansed_df[column].dtype == 'object':
                # Suppression des espaces en début/fin
                cleansed_df[column] = cleansed_df[column].astype(str).str.strip()
                
                # Remplacement des chaînes vides par NaN
                cleansed_df[column] = cleansed_df[column].replace('', np.nan)
                
                # Normalisation de la casse (optionnel)
                # cleansed_df[column] = cleansed_df[column].str.lower()
        
        return cleansed_df
    
    def _advanced_data_cleansing(self, df: pd.DataFrame, 
                               validation_results: Dict[str, Any]) -> pd.DataFrame:
        """
        Nettoyage avancé basé sur les résultats de validation.
        
        Args:
            df: DataFrame à nettoyer
            validation_results: Résultats de validation
            
        Returns:
            DataFrame nettoyé
        """
        cleansed_df = df.copy()
        
        # Traitement des colonnes avec trop de NULL
        for column, stats in validation_results.get('column_stats', {}).items():
            if stats['null_ratio'] > 0.8:
                self.logger.warning(f"Suppression de la colonne {column} (trop de NULL)")
                cleansed_df = cleansed_df.drop(columns=[column])
        
        return cleansed_df
    
    def _promote_types(self, series: pd.Series, old_type: str, new_type: str) -> pd.Series:
        """
        Promeut les types selon une hiérarchie logique.
        
        Args:
            series: Série à convertir
            old_type: Type actuel
            new_type: Nouveau type
            
        Returns:
            Série avec type promu
        """
        # Hiérarchie de promotion: int -> float -> str
        type_hierarchy = {
            'int64': 1,
            'float64': 2,
            'object': 3
        }
        
        old_level = type_hierarchy.get(old_type, 0)
        new_level = type_hierarchy.get(new_type, 0)
        
        if new_level > old_level:
            # Promotion vers un type plus général
            if new_level == 2:  # float
                return pd.to_numeric(series, errors='coerce')
            elif new_level == 3:  # string
                return series.astype(str)
        
        # Si pas de promotion possible, retourner la série originale
        return series
    
    def _can_convert_type(self, series: pd.Series, target_type: str) -> bool:
        """
        Vérifie si une série peut être convertie vers un type cible.
        
        Args:
            series: Série à tester
            target_type: Type cible
            
        Returns:
            True si la conversion est possible
        """
        try:
            if 'int' in target_type:
                pd.to_numeric(series, errors='raise')
            elif 'float' in target_type:
                pd.to_numeric(series, errors='raise')
            elif target_type == 'object':
                series.astype(str)
            return True
        except (ValueError, TypeError):
            return False
    
    def _update_single_dimension(self, unique_values: np.ndarray, dimension_name: str) -> int:
        """
        Met à jour une table de dimension spécifique (thread-safe).
        
        Args:
            unique_values: Valeurs uniques de la dimension
            dimension_name: Nom de la dimension
            
        Returns:
            Nombre de nouvelles valeurs ajoutées
        """
        with self._lock:
            try:
                return self._create_dimension_table_values(dimension_name, unique_values)
            except Exception as e:
                self.logger.error(f"Erreur lors de la mise à jour de {dimension_name}: {e}")
                return 0
    
    def _create_dimension_table_values(self, dimension_name: str, values: np.ndarray) -> int:
        """
        Crée ou met à jour les valeurs d'une table de dimension.
        
        Args:
            dimension_name: Nom de la dimension
            values: Valeurs à ajouter
            
        Returns:
            Nombre de valeurs ajoutées
        """
        table_name = f"dim_{dimension_name}"
        
        # Création de la table si elle n'existe pas
        create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                value VARCHAR PRIMARY KEY,
                label VARCHAR
            )
        """
        self.conn.execute(create_query)
        
        values_added = 0
        
        # Insertion des nouvelles valeurs
        for value in values:
            if pd.notna(value):
                try:
                    insert_query = f"""
                        INSERT INTO {table_name} (value, label) 
                        VALUES (?, ?)
                        ON CONFLICT (value) DO NOTHING
                    """
                    self.conn.execute(insert_query, [str(value), str(value)])
                    if self.conn.execute("SELECT changes()").fetchone()[0] > 0:
                        values_added += 1
                except Exception as e:
                    self.logger.warning(f"Impossible d'insérer la valeur {value} dans {table_name}: {e}")
        
        return values_added
    
    def _process_chunk(self, chunk_df: pd.DataFrame, merge_keys: List[str]) -> Dict[str, int]:
        """
        Traite un chunk de données.
        
        Args:
            chunk_df: Chunk à traiter
            merge_keys: Clés de fusion
            
        Returns:
            Statistiques du traitement du chunk
        """
        # Validation du chunk
        validation_results = self.data_quality_checker.validate_dataframe(
            chunk_df, DataValidationLevel.BASIC
        )
        
        if validation_results['quality_score'] < 0.5:
            self.logger.warning("Qualité de données faible dans le chunk, nettoyage appliqué")
            chunk_df = self._basic_data_cleansing(chunk_df)
        
        # Insertion du chunk
        return self.append_to_fact_table(chunk_df)

    def detect_schema_changes(self, new_df: pd.DataFrame) -> Dict:
        """
        Detect schema changes between new data and existing database.
        
        Args:
            new_df: New DataFrame to compare
            
        Returns:
            Dictionary describing detected changes
        """
        changes = {
            'new_columns': [],
            'removed_columns': [],
            'type_changes': [],
            'categorical_changes': []
        }
        
        # Chargement des métadonnées actuelles
        current_metadata = self.load_current_metadata()
        current_columns = set(current_metadata['name'].values)
        new_columns = set(new_df.columns)
        
        # Nouvelles colonnes
        changes['new_columns'] = list(new_columns - current_columns)
        
        # Colonnes supprimées (non présentes dans les nouvelles données)
        changes['removed_columns'] = list(current_columns - new_columns)
        
        # Vérification des changements de type
        for col in current_columns.intersection(new_columns):
            current_type = current_metadata[current_metadata['name'] == col]['python_type'].values[0]
            new_type = str(new_df[col].dtype)
            
            if current_type != new_type:
                changes['type_changes'].append({
                    'column': col,
                    'old_type': current_type,
                    'new_type': new_type
                })
            
            # Vérification si une colonne devient catégorielle ou non
            current_is_cat = current_metadata[current_metadata['name'] == col]['is_categorical'].values[0]
            new_unique_count = new_df[col].nunique()
            new_is_cat = (str(new_df[col].dtype) == 'object' and 
                         new_unique_count <= self.categorical_threshold)
            
            if current_is_cat != new_is_cat:
                changes['categorical_changes'].append({
                    'column': col,
                    'was_categorical': current_is_cat,
                    'is_categorical': new_is_cat
                })
        
        return changes
    
    def add_new_columns(self, new_columns: List[str], new_df: pd.DataFrame) -> None:
        """
        Add new columns to the fact table and update metadata.
        
        Args:
            new_columns: List of new column names
            new_df: DataFrame containing the new columns
        """
        for col in new_columns:
            dtype = str(new_df[col].dtype)
            sql_type = SchemaBuilder._map_python_to_sql_type(dtype)
            
            # Ajout de la colonne à la fact table
            try:
                # Valeur par défaut selon le type
                default_value = "NULL"
                if 'int' in dtype:
                    default_value = "0"
                elif 'float' in dtype:
                    default_value = "0.0"
                elif dtype == 'bool':
                    default_value = "FALSE"
                elif dtype == 'object':
                    default_value = "''"
                
                alter_query = f"ALTER TABLE fact_table ADD COLUMN {col} {sql_type} DEFAULT {default_value}"
                self.conn.execute(alter_query)
                
                self.logger.info(f"Added column {col} to fact_table")
                
            except Exception as e:
                self.logger.error(f"Failed to add column {col}: {e}")
                raise
            
            # Mise à jour des métadonnées
            self.update_metadata_for_column(col, new_df)
            
            # Si la colonne est catégorielle, créer la table de dimension
            if dtype == 'object' and new_df[col].nunique() <= self.categorical_threshold:
                self.create_dimension_table(col, new_df[col].unique())
    
    def update_metadata_for_column(self, column: str, df: pd.DataFrame) -> None:
        """
        Update or insert metadata for a specific column.
        
        Args:
            column: Column name
            df: DataFrame containing the column
        """
        dtype = str(df[column].dtype)
        sql_type = SchemaBuilder._map_python_to_sql_type(dtype)
        is_categorical = (dtype == 'object' and 
                         df[column].nunique() <= self.categorical_threshold)
        
        # Vérification si la colonne existe déjà dans les métadonnées
        existing = self.conn.execute(
            "SELECT COUNT(*) FROM metadata WHERE name = ?", 
            [column]
        ).fetchone()[0]
        
        if existing > 0:
            # Mise à jour
            update_query = """
                UPDATE metadata 
                SET python_type = ?, sql_type = ?, is_categorical = ?
                WHERE name = ?
            """
            self.conn.execute(update_query, [dtype, sql_type, is_categorical, column])
            self.logger.info(f"Updated metadata for column {column}")
        else:
            # Insertion
            insert_query = """
                INSERT INTO metadata (name, label, python_type, sql_type, is_categorical)
                VALUES (?, ?, ?, ?, ?)
            """
            label = column.replace('_', ' ').title()
            self.conn.execute(insert_query, [column, label, dtype, sql_type, is_categorical])
            self.logger.info(f"Inserted metadata for column {column}")
    
    def create_dimension_table(self, dimension_name: str, values: np.ndarray) -> None:
        """
        Create or update a dimension table.
        
        Args:
            dimension_name: Name of the dimension
            values: Unique values for the dimension
        """
        table_name = f"dim_{dimension_name}"
        
        # Création de la table si elle n'existe pas
        create_query = f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                value VARCHAR PRIMARY KEY,
                label VARCHAR
            )
        """
        self.conn.execute(create_query)
        
        # Préparation des données
        dim_df = pd.DataFrame({
            'value': values,
            'label': values
        })
        dim_df = dim_df.dropna().drop_duplicates()
        
        # Insertion des nouvelles valeurs uniquement
        for _, row in dim_df.iterrows():
            try:
                insert_query = f"""
                    INSERT INTO {table_name} (value, label) 
                    VALUES (?, ?)
                    ON CONFLICT (value) DO NOTHING
                """
                self.conn.execute(insert_query, [str(row['value']), str(row['label'])])
            except Exception as e:
                self.logger.warning(f"Could not insert value {row['value']} into {table_name}: {e}")
        
        self.logger.info(f"Updated dimension table {table_name}")
    
    def merge_data(self, 
                   new_df: pd.DataFrame, 
                   merge_keys: Optional[List[str]] = None,
                   update_mode: str = 'upsert') -> Dict:
        """
        Merge new data into existing fact table.
        
        Args:
            new_df: New data to merge
            merge_keys: Columns to use as merge keys (for identifying duplicates)
            update_mode: 'append', 'upsert', or 'replace'
            
        Returns:
            Dictionary with merge statistics
        """
        stats = {
            'rows_added': 0,
            'rows_updated': 0,
            'rows_unchanged': 0,
            'new_columns': [],
            'new_dimension_values': {}
        }
        
        # Détection des changements de schéma
        schema_changes = self.detect_schema_changes(new_df)
        
        # Ajout des nouvelles colonnes si nécessaire
        if schema_changes['new_columns']:
            self.add_new_columns(schema_changes['new_columns'], new_df)
            stats['new_columns'] = schema_changes['new_columns']
        
        # Préparation des données pour l'insertion
        prepared_df = self.prepare_dataframe_for_insert(new_df)
        
        if update_mode == 'append':
            # Simple ajout des données
            stats['rows_added'] = self.append_to_fact_table(prepared_df)
            
        elif update_mode == 'upsert':
            # Mise à jour ou insertion basée sur les clés
            if not merge_keys:
                self.logger.warning("No merge keys specified for upsert, falling back to append")
                stats['rows_added'] = self.append_to_fact_table(prepared_df)
            else:
                stats.update(self.upsert_to_fact_table(prepared_df, merge_keys))
                
        elif update_mode == 'replace':
            # Remplacement complet des données
            self.conn.execute("DELETE FROM fact_table")
            stats['rows_added'] = self.append_to_fact_table(prepared_df)
            
        # Mise à jour des tables de dimension
        stats['new_dimension_values'] = self.update_dimension_tables(new_df)
        
        # Invalidation du cache
        self._metadata_cache = None
        self._dimension_cache = {}
        
        return stats
    
    def prepare_dataframe_for_insert(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare DataFrame for insertion by converting categorical values to IDs.
        
        Args:
            df: DataFrame to prepare
            
        Returns:
            Prepared DataFrame
        """
        prepared_df = df.copy()
        metadata = self.load_current_metadata()
        
        for _, row in metadata.iterrows():
            if row['is_categorical'] and row['name'] in prepared_df.columns:
                # Remplacement des valeurs par les IDs (dans ce cas, on garde les valeurs)
                # car DuckDB gère bien les jointures sur les strings
                pass
        
        return prepared_df
    
    def append_to_fact_table(self, df: pd.DataFrame) -> int:
        """
        Append data to fact table.
        
        Args:
            df: DataFrame to append
            
        Returns:
            Number of rows added
        """
        initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        
        # Utilisation de la méthode register pour insérer efficacement
        self.conn.register('temp_insert', df)
        
        # Récupération des colonnes existantes dans fact_table
        existing_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
        
        # Sélection uniquement des colonnes existantes
        columns_to_insert = [col for col in df.columns if col in existing_columns]
        columns_str = ", ".join(columns_to_insert)
        
        insert_query = f"""
            INSERT INTO fact_table ({columns_str})
            SELECT {columns_str} FROM temp_insert
        """
        
        self.conn.execute(insert_query)
        self.conn.execute("DROP VIEW temp_insert")
        
        final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        rows_added = final_count - initial_count
        
        self.logger.info(f"Added {rows_added} rows to fact_table")
        return rows_added
    
    def upsert_to_fact_table(self, df: pd.DataFrame, merge_keys: List[str]) -> Dict:
        """
        Perform upsert operation on fact table.
        
        Args:
            df: DataFrame to upsert
            merge_keys: Columns to use as merge keys
            
        Returns:
            Dictionary with upsert statistics
        """
        stats = {'rows_added': 0, 'rows_updated': 0}
        
        # Création d'une table temporaire
        self.conn.register('temp_upsert', df)
        
        # Identification des lignes existantes
        merge_condition = " AND ".join([f"f.{key} = t.{key}" for key in merge_keys])
        
        # Comptage des lignes qui seront mises à jour
        update_count_query = f"""
            SELECT COUNT(*) FROM fact_table f
            WHERE EXISTS (
                SELECT 1 FROM temp_upsert t
                WHERE {merge_condition}
            )
        """
        stats['rows_updated'] = self.conn.execute(update_count_query).fetchone()[0]
        
        # Mise à jour des lignes existantes
        if stats['rows_updated'] > 0:
            # Construction de la clause SET
            existing_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
            update_columns = [col for col in df.columns if col in existing_columns and col not in merge_keys]
            set_clause = ", ".join([f"f.{col} = t.{col}" for col in update_columns])
            
            update_query = f"""
                UPDATE fact_table f
                SET {set_clause}
                FROM temp_upsert t
                WHERE {merge_condition}
            """
            self.conn.execute(update_query)
        
        # Insertion des nouvelles lignes
        insert_query = f"""
            INSERT INTO fact_table
            SELECT t.* FROM temp_upsert t
            WHERE NOT EXISTS (
                SELECT 1 FROM fact_table f
                WHERE {merge_condition}
            )
        """
        
        initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        self.conn.execute(insert_query)
        final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        
        stats['rows_added'] = final_count - initial_count - stats['rows_updated']
        
        self.conn.execute("DROP VIEW temp_upsert")
        
        self.logger.info(f"Upsert complete: {stats['rows_added']} added, {stats['rows_updated']} updated")
        return stats
    
    def update_dimension_tables(self, df: pd.DataFrame) -> Dict:
        """
        Update dimension tables with new values from DataFrame.
        
        Args:
            df: DataFrame containing potential new dimension values
            
        Returns:
            Dictionary mapping dimension names to number of new values added
        """
        new_values = {}
        metadata = self.load_current_metadata()
        
        for _, row in metadata.iterrows():
            if row['is_categorical'] and row['name'] in df.columns:
                dimension_name = row['name']
                table_name = f"dim_{dimension_name}"
                
                # Récupération des valeurs actuelles
                try:
                    current_values = set(
                        self.conn.execute(f"SELECT value FROM {table_name}").fetchdf()['value']
                    )
                except:
                    current_values = set()
                
                # Nouvelles valeurs uniques
                new_unique = set(df[dimension_name].dropna().unique())
                values_to_add = new_unique - current_values
                
                if values_to_add:
                    # Création ou mise à jour de la table de dimension
                    self.create_dimension_table(dimension_name, list(values_to_add))
                    new_values[dimension_name] = len(values_to_add)
        
        return new_values
    
    def compute_data_hash(self, df: pd.DataFrame) -> str:
        """
        Compute hash of DataFrame for change detection.
        
        Args:
            df: DataFrame to hash
            
        Returns:
            SHA256 hash of the data
        """
        # Conversion du DataFrame en bytes pour le hashing
        df_bytes = pd.util.hash_pandas_object(df, index=False).values.tobytes()
        return hashlib.sha256(df_bytes).hexdigest()
    
    def track_update_history(self, operation: str, stats: Dict) -> None:
        """
        Track update history in a dedicated table.
        
        Args:
            operation: Type of operation performed
            stats: Statistics from the operation
        """
        # Création de la table d'historique si elle n'existe pas
        create_history_query = """
            CREATE TABLE IF NOT EXISTS update_history (
                update_id INTEGER PRIMARY KEY,
                timestamp TIMESTAMP,
                operation VARCHAR,
                stats JSON,
                user VARCHAR,
                description VARCHAR
            )
        """
        self.conn.execute(create_history_query)
        
        # Insertion de l'historique
        insert_query = """
            INSERT INTO update_history (timestamp, operation, stats)
            VALUES (?, ?, ?)
        """
        self.conn.execute(insert_query, [
            datetime.now(),
            operation,
            json.dumps(stats)
        ])
        
        self.logger.info(f"Tracked update history for operation: {operation}")