# Importation des modules
# Modules de base
import os
import pandas as pd
import numpy as np
from pathlib import Path
# Types
from typing import Dict, List, Optional, Union, Literal
from enum import Enum
# DuckDB
import duckdb
# Traitement en parallèle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Import des éléments du module
from ..builders.schema import SchemaBuilder
from ..utils.logger import _init_logger

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))

# Type pour la résolution de conflits
class ConflictResolutionStrategy(Enum):
    """
    Stratégies de résolution de conflits de types.
    """
    PROMOTE_TYPES = "promote_types"  # Promotion automatique des types (int -> float, etc.)
    STRICT_VALIDATION = "strict_validation"  # Validation stricte, échec si conflit
    COERCE_TO_STRING = "coerce_to_string"  # Conversion forcée vers string

# Classe de mise à jour de la base de données
class DatabaseUpdater:
    """
    A class to handle incremental updates to DuckDB databases.
    
    Supports updating metadata, fact tables, and dimension tables while maintaining
    consistency and handling type conflicts. Includes duplicate removal and index management.
    """
    # Initialisation
    def __init__(self, 
                 connection: duckdb.DuckDBPyConnection,
                 categorical_threshold: Optional[int] = 50,
                 log_filename: Optional[os.PathLike] = None,
                 max_workers: int = 4,
                 batch_size: int = 10000):
        """
        Initialize the DatabaseUpdater.
        
        Args:
            connection: DuckDB connection object
            categorical_threshold: Maximum unique values for categorical columns
            log_filename: Path to log file
            max_workers: Maximum number of parallel workers
            batch_size: Size of batches for processing
        """
        # Initialisation des arguments
        self.conn = connection
        self.categorical_threshold = categorical_threshold
        self.max_workers = max_workers
        self.batch_size = batch_size
        
        # Initialisation du logger
        if log_filename is None:
            log_filename = os.path.join(FILE_PATH.parents[2], "logs/database_updater.log")
        self.logger = _init_logger(filename=log_filename)
        
        # Cache pour les métadonnées
        self._metadata_cache = None
        
        # Verrou pour les opérations thread-safe
        self._lock = threading.Lock()
    
    # Méthode de mise à jour de la base de données
    def update_database(self,
                       update_df: pd.DataFrame,
                       check_duplicates_db: bool = True,
                       check_duplicates_update: bool = True,
                       keep: Literal[False, 'first', 'last'] = False,
                       indexes: Optional[Union[List[str], List[List[str]]]] = None) -> Dict:
        """
        Update the entire database with new data.
        
        Args:
            update_df: DataFrame containing update data
            check_duplicates_db: Whether to check duplicates in existing database
            check_duplicates_update: Whether to check duplicates in update DataFrame
            keep: Which duplicates to keep when removing duplicates
            indexes: List of index definitions (single columns or composite indexes)
            
        Returns:
            Dictionary with update statistics
        """
        # Logging
        self.logger.info("Updating the database")
        
        # Statistiques de mise à jour
        stats = {
            'metadata_changes': {},
            'fact_table_changes': {},
            'dimension_changes': {},
            'duplicates_removed': {'database': 0, 'update': 0},
            'indexes_updated': []
        }
        
        # Étape 1: Suppression des doublons dans les données de mise à jour si demandée
        if check_duplicates_update:
            update_df = self._remove_duplicates_from_dataframe(update_df, keep, 'update')
            stats['duplicates_removed']['update'] = len(update_df)
        
        # Étape 2: Suppression des doublons dans la base de données existante si demandée
        if check_duplicates_db:
            removed_count = self._remove_duplicates_from_database(keep)
            stats['duplicates_removed']['database'] = removed_count
        
        # Étape 3: Mise à jour des métadonnées
        stats['metadata_changes'] = self.update_metadata(update_df)
        
        # Étape 4: Mise à jour de la table de faits
        stats['fact_table_changes'] = self.update_fact_table(update_df)
        
        # Étape 5: Mise à jour des tables de dimensions
        stats['dimension_changes'] = self.update_dimension_tables(update_df)
        
        # Étape 6: Mise à jour des index si spécifiés
        if indexes is not None:
            stats['indexes_updated'] = self.update_indexes(indexes)
        
        # Invalidation du cache des métadonnées
        self._metadata_cache = None
        
        # Logging
        self.logger.info("Finishing the update of the database")

        return stats
    
    # Méthode de mise à jour des métadonnées
    def update_metadata(self, update_df: pd.DataFrame) -> Dict:
        """
        Update metadata table with new columns and resolve type conflicts.
        
        Args:
            update_df: DataFrame containing new data
            
        Returns:
            Dictionary with metadata update statistics
        """
        # Logging
        self.logger.info("Updating the metadata table")
        
        stats = {
            'columns_added': [],
            'types_changed': [],
            'categorical_changes': [],
            'conflicts_resolved': []
        }
        
        # Chargement des métadonnées actuelles
        current_metadata = self._load_current_metadata()
        current_columns = set(current_metadata['name'].values) if len(current_metadata) > 0 else set()
        new_columns = set(update_df.columns)
        
        # Identification des nouvelles colonnes
        columns_to_add = new_columns - current_columns
        
        # Ajout des nouvelles colonnes aux métadonnées
        for col in columns_to_add:
            self._add_column_to_metadata(col, update_df)
            stats['columns_added'].append(col)
        
        # Vérification des conflits de types pour les colonnes existantes
        for col in current_columns.intersection(new_columns):
            conflict_resolution = self._resolve_type_conflicts(col, update_df, current_metadata)
            if conflict_resolution:
                stats['conflicts_resolved'].append(conflict_resolution)
                if conflict_resolution['type_changed']:
                    stats['types_changed'].append({
                        'column': col,
                        'old_type': conflict_resolution['old_type'],
                        'new_type': conflict_resolution['new_type'],
                        'strategy': conflict_resolution['strategy']
                    })
        
        # Vérification des changements catégoriels
        categorical_changes = self._update_categorical_status(update_df, current_metadata)
        stats['categorical_changes'] = categorical_changes
        
        return stats
    
    # Méthode de mise à jour de la table des faits
    def update_fact_table(self, update_df: pd.DataFrame) -> Dict:
        """
        Update fact table with new columns, rows, and type changes.
        
        Args:
            update_df: DataFrame containing update data
            
        Returns:
            Dictionary with fact table update statistics
        """
        # Logging
        self.logger.info("Updating the fact table")
        
        stats = {
            'columns_added': [],
            'rows_added': 0,
            'rows_updated': 0,
            'types_updated': []
        }
        
        # Récupération de la structure actuelle de la fact table
        try:
            current_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
        except:
            # Table n'existe pas encore
            current_columns = []
        
        # Ajout des colonnes manquantes
        new_columns = set(update_df.columns) - set(current_columns)
        for col in new_columns:
            self._add_column_to_fact_table(col, update_df)
            stats['columns_added'].append(col)
        
        # Mise à jour des types de colonnes si nécessaire
        for col in set(update_df.columns).intersection(set(current_columns)):
            if self._update_column_type_in_fact_table(col, update_df):
                stats['types_updated'].append(col)
        
        # Ajout/mise à jour des lignes
        if len(current_columns) > 0:
            # Mise à jour avec upsert basé sur toutes les colonnes communes
            merge_keys = list(set(update_df.columns).intersection(set(current_columns)))
            upsert_stats = self._upsert_to_fact_table(update_df, merge_keys)
            stats['rows_added'] = upsert_stats['rows_added']
            stats['rows_updated'] = upsert_stats['rows_updated']
        else:
            # Création de la table avec les données
            stats['rows_added'] = self._create_fact_table_with_data(update_df)
        
        return stats
    
    # Méthode de mise à jour des tables de dimension
    def update_dimension_tables(self, update_df: pd.DataFrame) -> Dict:
        """
        Update dimension tables by removing non-categorical tables and updating values.
        
        Args:
            update_df: DataFrame containing update data
            
        Returns:
            Dictionary with dimension table update statistics
        """
        # Logging
        self.logger.info("Updating dimension tables")
        
        stats = {
            'tables_removed': [],
            'values_added': {},
            'metadata_updated': []
        }
        
        # Chargement des métadonnées actuelles
        current_metadata = self._load_current_metadata()
        
        # Vérification de chaque colonne pour les changements catégoriels
        # Parcours des colonnes référencées dans les méta-données
        for _, row in current_metadata.iterrows():
            # Extraction du nom et
            col_name = row['name']
            # Extraction du statut de la colonne
            was_categorical = row['is_categorical']
            
            # Parcours des colonnes
            if col_name in update_df.columns:
                # Vérification si la colonne est toujours catégorielle
                unique_count = update_df[col_name].nunique()
                is_still_categorical = (
                    str(update_df[col_name].dtype) == 'object' and 
                    unique_count <= self.categorical_threshold
                )
                # Distinction des différents cas
                # N'est plus catégorielle
                if was_categorical and not is_still_categorical:
                    # Suppression de la table de dimension
                    table_name = f"dim_{col_name}"
                    try:
                        # Suppression de la table des métadonnées
                        self.conn.execute(f"DROP TABLE IF EXISTS {table_name}")
                        stats['tables_removed'].append(table_name)
                        
                        # Mise à jour du statut catégoriel dans les métadonnées
                        self.conn.execute(
                            "UPDATE metadata SET is_categorical = ? WHERE name = ?",
                            [False, col_name]
                        )
                        stats['metadata_updated'].append(col_name)
                        # Logging
                        self.logger.info(f"Suppression de la table de dimension {table_name} (trop de modalités)")
                    except Exception as e:
                        # Logging
                        self.logger.error(f"Erreur lors de la suppression de {table_name}: {e}")
                # Est toujours catégorielle
                elif is_still_categorical:
                    # Mise à jour des valeurs de la dimension
                    values_added = self._update_dimension_values(col_name, update_df[col_name])
                    if values_added > 0:
                        stats['values_added'][col_name] = values_added
                # Devient catégorielle
        
        return stats
    
    # Méthode de mise à jour des index
    def update_indexes(self, indexes: Union[List[str], List[List[str]]]) -> List[str]:
        """
        Update database indexes by removing old ones and creating new ones.
        
        Args:
            indexes: List of index definitions (column names or lists of column names)
            
        Returns:
            List of created index names
        """
        # Logging
        self.logger.info("Updating indexes")
        
        # Suppression de tous les index existants (sauf les index système)
        try:
            # Récupération des index existants
            existing_indexes = self.conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type = 'index' AND name NOT LIKE 'sqlite_%'
            """).fetchall()
            # Parcours des index
            for idx in existing_indexes:
                try:
                    self.conn.execute(f"DROP INDEX IF EXISTS {idx[0]}")
                    self.logger.info(f"Index {idx[0]} supprimé")
                except Exception as e:
                    self.logger.warning(f"Impossible de supprimer l'index {idx[0]}: {e}")
        except Exception as e:
            self.logger.warning(f"Erreur lors de la récupération des index existants: {e}")
        
        # Création des nouveaux index
        for i, index_def in enumerate(indexes):
            try:
                if isinstance(index_def, str):
                    # Index sur une seule colonne
                    index_name = f"idx_{index_def}"
                    # Création de la requête
                    create_query = f"CREATE INDEX {index_name} ON fact_table ({index_def})"
                elif isinstance(index_def, list):
                    # Index composite
                    columns_str = ', '.join(index_def)
                    index_name = f"idx_{'_'.join(index_def)}"
                    # Création de la requête
                    create_query = f"CREATE INDEX {index_name} ON fact_table ({columns_str})"
                else:
                    # Logging
                    self.logger.warning(f"Définition d'index invalide: {index_def}")
                    continue
                # Exécution de la requête
                self.conn.execute(create_query)
                # Logging
                self.logger.info(f"Index {index_name} créé")
                
            except Exception as e:
                self.logger.error(f"Erreur lors de la création de l'index {index_def}: {e}")
    
    # Méthodes pour le traitement par lots et la parallélisation
    # Méthode de processing en batchs
    def _process_large_dataframe_in_batches(self, df: pd.DataFrame, 
                                          process_func, batch_size: Optional[int] = None) -> List:
        """
        Method processing large pandas DataFrames in batches
        
        Args:
            df: DataFrame à traiter
            process_func: Fonction de traitement à appliquer à chaque lot
            batch_size: Taille des lots (utilise self.batch_size par défaut)
            
        Returns:
            Liste des résultats de traitement
        """
        # Initialisation de la taille des batchs
        batch_size = batch_size or self.batch_size
        # Initialisation des résultats
        results = []
        
        # Logging
        self.logger.info(f"Traitement par lots de {len(df)} lignes avec des lots de {batch_size}")
        # Parcours des observations du jeu de données en batch
        for i in range(0, len(df), batch_size):
            # Circonscription du batch
            batch_start = i
            batch_end = min(i + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end].copy()
            
            try:
                # Processing du batch
                result = process_func(batch_df)
                # Ajout du résultat
                results.append(result)
            except Exception as e:
                # Logging
                self.logger.error(f"Erreur lors du traitement du lot {batch_start}-{batch_end}: {e}")
                # Ajout d'un résultat vide
                results.append(None)
        
        return results
    
    # Méthode de mise à jour en parallèle des tables de dimension
    def _parallel_dimension_update(self, columns_and_values: List[tuple]) -> Dict[str, int]:
        """
        Mise à jour parallèle des tables de dimensions.
        
        Args:
            columns_and_values: Liste de tuples (nom_colonne, valeurs_uniques)
            
        Returns:
            Dictionnaire des statistiques de mise à jour par dimension
        """
        # Logging
        self.logger.info(f"Mise à jour parallèle de {len(columns_and_values)} dimensions")
        
        update_stats = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Soumission des tâches
            future_to_dimension = {
                executor.submit(self._update_dimension_values, col_name, values): col_name
                for col_name, values in columns_and_values
            }
            
            # Récupération des résultats
            for future in as_completed(future_to_dimension):
                dimension = future_to_dimension[future]
                try:
                    values_added = future.result()
                    update_stats[dimension] = values_added
                except Exception as e:
                    self.logger.error(f"Erreur lors de la mise à jour parallèle de {dimension}: {e}")
                    update_stats[dimension] = 0
        
        return update_stats
    
    # Méthodes privées utilitaires
    
    def _load_current_metadata(self) -> pd.DataFrame:
        """
        Chargement des métadonnées actuelles depuis la base de données.
        
        Returns:
            DataFrame contenant les métadonnées actuelles
        """
        # Chargement de la table si elle n'est pas en cache
        if self._metadata_cache is None:
            try:
                # Chargement
                self._metadata_cache = self.conn.execute("SELECT * FROM metadata").fetchdf()
            except:
                # Si la table n'existe pas encore, on l'initialise vide
                self._metadata_cache = pd.DataFrame(columns=['name', 'label', 'python_type', 'sql_type', 'is_categorical'])
        
        return self._metadata_cache
    
    # Méthode de suppression des doublons d'un jeu de données
    def _remove_duplicates_from_dataframe(self, df: pd.DataFrame, 
                                        keep: Literal[False, 'first', 'last'],
                                        source: str) -> pd.DataFrame:
        """
        Suppression des doublons d'un DataFrame en excluant la colonne 'value'.
        
        Args:
            df: DataFrame à traiter
            keep: Stratégie de conservation des doublons
            source: Source des données pour le logging
            
        Returns:
            DataFrame sans doublons
        """
        # Comptage du nombre d'observations
        initial_count = len(df)
        
        # Identification des colonnes à vérifier (toutes sauf 'value' si elle existe)
        columns_to_check = [col for col in df.columns if col != 'value']
        
        # Suppression des duplicats
        if keep == False:
            df_cleaned = df.drop_duplicates(subset=columns_to_check, keep=False)
        else:
            df_cleaned = df.drop_duplicates(subset=columns_to_check, keep=keep)
        
        # COmptage du nombre d'obserabtion supprimées
        removed_count = initial_count - len(df_cleaned)
        if removed_count > 0:
            self.logger.warning(f"Suppression des doublons ({source}): {removed_count} observations supprimées")
        
        return df_cleaned
    
    # Méthode de déduplication de la base de données
    def _remove_duplicates_from_database(self, keep: Literal[False, 'first', 'last']) -> int:
        """
        Suppression des doublons dans la table de faits existante.
        
        Args:
            keep: Stratégie de conservation des doublons
            
        Returns:
            Nombre de lignes supprimées
        """
        try:
            # Récupération du nombre initial de lignes
            initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            
            # Récupération des colonnes (exclure 'value' si elle existe)
            all_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
            columns_to_check = [col for col in all_columns if col != 'value']
            
            if not columns_to_check:
                return 0
            
            # Construction de la requête de suppression des doublons
            columns_str = ', '.join(columns_to_check)
            
            if keep == False:
                # Suppression de tous les doublons
                delete_query = f"""
                DELETE FROM fact_table 
                WHERE rowid NOT IN (
                    SELECT MIN(rowid) 
                    FROM fact_table 
                    GROUP BY {columns_str}
                    HAVING COUNT(*) = 1
                )
                """
            else:
                # Conservation du premier ou dernier
                order_clause = "ASC" if keep == 'first' else "DESC"
                delete_query = f"""
                DELETE FROM fact_table 
                WHERE rowid NOT IN (
                    SELECT rowid FROM (
                        SELECT rowid, ROW_NUMBER() OVER (
                            PARTITION BY {columns_str} 
                            ORDER BY rowid {order_clause}
                        ) as rn
                        FROM fact_table
                    ) WHERE rn = 1
                )
                """
            
            self.conn.execute(delete_query)
            
            # Calcul des lignes supprimées
            final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
            removed_count = initial_count - final_count
            
            if removed_count > 0:
                self.logger.warning(f"Suppression des doublons (base de données): {removed_count} lignes supprimées")
            
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la suppression des doublons dans la base: {e}")
            return 0
    
    # Méthode d'ajout des informations d'une colonne à la table des métadonnées
    def _add_column_to_metadata(self, column: str, df: pd.DataFrame) -> None:
        """
        Ajout d'une nouvelle colonne aux métadonnées.
        
        Args:
            column: Nom de la colonne
            df: DataFrame contenant la colonne
        """
        # Identification du type de la colonne
        dtype = str(df[column].dtype)
        # Conversion du type en SQL
        sql_type = SchemaBuilder._map_python_to_sql_type(dtype)
        # Définition si la variable est catégorielle
        is_categorical = (
            dtype == 'object' and 
            df[column].nunique() <= self.categorical_threshold
        )
        
        # Création de la table metadata si elle n'existe pas
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (
                name VARCHAR,
                label VARCHAR,
                python_type VARCHAR,
                sql_type VARCHAR,
                is_categorical BOOLEAN
            )
        """)
        
        # Insertion de la nouvelle colonne
        label = column.replace('_', ' ').title()
        self.conn.execute("""
            INSERT INTO metadata (name, label, python_type, sql_type, is_categorical)
            VALUES (?, ?, ?, ?, ?)
        """, [column, label, dtype, sql_type, is_categorical])
        # Logging
        self.logger.info(f"Ajout de la colonne {column} aux métadonnées")
    
    # Méthode de résolutions des conflits de types
    def _resolve_type_conflicts(self, column: str, df: pd.DataFrame, 
                               current_metadata: pd.DataFrame) -> Optional[Dict]:
        """
        Résolution des conflits de types selon la stratégie la moins contraignante.
        
        Args:
            column: Nom de la colonne
            df: DataFrame avec les nouvelles données
            current_metadata: Métadonnées actuelles
            
        Returns:
            Dictionnaire avec les informations de résolution du conflit
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
            strategy = ConflictResolutionStrategy.PROMOTE_TYPES.value
        else:
            resolved_type = current_type
            strategy = ConflictResolutionStrategy.PROMOTE_TYPES.value
        
        # Conversion en python
        sql_type = SchemaBuilder._map_python_to_sql_type(resolved_type)
        # Mise à jour des métadonnées
        self.conn.execute("""
            UPDATE metadata 
            SET python_type = ?, sql_type = ?
            WHERE name = ?
        """, [resolved_type, sql_type, column])
        # Logging
        self.logger.info(f"Résolution de conflit de type pour {column}: {current_type} -> {resolved_type}")
        
        return {
            'column': column,
            'old_type': current_type,
            'new_type': resolved_type,
            'strategy': strategy,
            'type_changed': True
        }
    
    # Méthode de mise à jour du statut de variable catégorielle
    def _update_categorical_status(self, df: pd.DataFrame, 
                                 current_metadata: pd.DataFrame) -> List[Dict]:
        """
        Mise à jour du statut catégoriel des colonnes.
        
        Args:
            df: DataFrame avec les nouvelles données
            current_metadata: Métadonnées actuelles
            
        Returns:
            Liste des changements catégoriels
        """
        changes = []
        # Parcours des colonnes
        for col in df.columns:
            if col in current_metadata['name'].values:
                # Identification si la variable dans la base de données est actuellement catégorielle
                current_is_cat = current_metadata[current_metadata['name'] == col]['is_categorical'].values[0]
                # Comptage du nombre de modalités pour savoir si après la mise à jour la variable est catégorielle
                new_unique_count = df[col].nunique()
                new_is_cat = (
                    str(df[col].dtype) == 'object' and 
                    new_unique_count <= self.categorical_threshold
                )
                
                if current_is_cat != new_is_cat:
                    # Mise à jour du statut dans les métadonnées
                    self.conn.execute("""
                        UPDATE metadata 
                        SET is_categorical = ?
                        WHERE name = ?
                    """, [new_is_cat, col])
                    
                    changes.append({
                        'column': col,
                        'was_categorical': current_is_cat,
                        'is_categorical': new_is_cat
                    })
                    # Logging
                    self.logger.info(f"Changement de statut catégoriel pour {col}: {current_is_cat} -> {new_is_cat}")
        
        return changes
    
    # Méthode d'ajout d'une nouvelle colonne à la table des faits
    def _add_column_to_fact_table(self, column: str, df: pd.DataFrame) -> None:
        """
        Ajout d'une nouvelle colonne à la table de faits.
        
        Args:
            column: Nom de la colonne
            df: DataFrame contenant la colonne
        """
        # Identification du type de la colonne
        dtype = str(df[column].dtype)
        # Conversion du type en SQL
        sql_type = SchemaBuilder._map_python_to_sql_type(dtype)
        
        # Valeur par défaut selon le type
        if 'int' in dtype:
            default_value = "0"
        elif 'float' in dtype:
            default_value = "0.0"
        elif dtype == 'bool':
            default_value = "FALSE"
        else:
            default_value = "''"
        # Requête d'ajout de la colonne
        alter_query = f"ALTER TABLE fact_table ADD COLUMN {column} {sql_type} DEFAULT {default_value}"
        # Exécution de la requête
        self.conn.execute(alter_query)
        # Logging
        self.logger.info(f"Ajout de la colonne {column} à la table de faits")
    
    # Mise à jour du type de la colonne dans la table des faist
    def _update_column_type_in_fact_table(self, column: str, df: pd.DataFrame) -> bool:
        """
        Mise à jour du type d'une colonne dans la table de faits.
        
        Args:
            column: Nom de la colonne
            df: DataFrame avec le nouveau type
            
        Returns:
            True si le type a été mis à jour
        """
        # Note: DuckDB ne supporte pas ALTER COLUMN TYPE directement
        # Cette méthode pourrait être implémentée en recréant la table si nécessaire
        # Pour l'instant, on log seulement le changement
        new_type = str(df[column].dtype)
        # Logging
        self.logger.info(f"Type de colonne {column} mis à jour vers {new_type}")
        return True
    
    # Méthode de création de la table des faits avec les données
    def _create_fact_table_with_data(self, df: pd.DataFrame) -> int:
        """
        Création de la table de faits avec les données.
        
        Args:
            df: DataFrame avec les données
            
        Returns:
            Nombre de lignes ajoutées
        """
        # Enregistrement d'une table temporaires qui recopie le jeu de données
        self.conn.register('temp_fact_creation', df)
        # Recopie dans la table des faits
        self.conn.execute("""
            CREATE TABLE fact_table AS 
            SELECT * FROM temp_fact_creation
        """)
        # Suppression de la table temporaire
        self.conn.execute('DROP VIEW temp_fact_creation')
        
        # Logging du nombre de lignes ajoutées
        rows_added = len(df)
        self.logger.info(f"Création de la table de faits avec {rows_added} lignes")
        return rows_added
    
    # Méthode d'insertion et de mise à jour de données dans la table des faits
    def _upsert_to_fact_table(self, df: pd.DataFrame, merge_keys: List[str]) -> Dict:
        """
        Upsert des données dans la table de faits.
        
        Args:
            df: DataFrame à insérer/mettre à jour
            merge_keys: Colonnes utilisées pour identifier les doublons
            
        Returns:
            Dictionnaire avec les statistiques d'upsert
        """
        # Enregistrement d'une vue temporaire du jeu de données
        self.conn.register('temp_upsert', df)
        
        # Construction des conditions de jointure
        merge_condition = " AND ".join([f"f.{key} = t.{key}" for key in merge_keys])
        
        # Comptage des mises à jour
        update_count_query = f"""
            SELECT COUNT(*) FROM fact_table f
            WHERE EXISTS (
                SELECT 1 FROM temp_upsert t
                WHERE {merge_condition}
            )
        """
        rows_updated = self.conn.execute(update_count_query).fetchone()[0]
        
        # Mise à jour des lignes existantes
        if rows_updated > 0:
            existing_columns = [col[0] for col in self.conn.execute("DESCRIBE fact_table").fetchall()]
            update_columns = [col for col in df.columns if col in existing_columns and col not in merge_keys]
            
            if update_columns:
                set_clause = ", ".join([f"f.{col} = t.{col}" for col in update_columns])
                update_query = f"""
                    UPDATE fact_table f
                    SET {set_clause}
                    FROM temp_upsert t
                    WHERE {merge_condition}
                """
                self.conn.execute(update_query)
        
        # Insertion des nouvelles lignes
        initial_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        
        insert_query = f"""
            INSERT INTO fact_table
            SELECT t.* FROM temp_upsert t
            WHERE NOT EXISTS (
                SELECT 1 FROM fact_table f
                WHERE {merge_condition}
            )
        """
        self.conn.execute(insert_query)
        
        final_count = self.conn.execute("SELECT COUNT(*) FROM fact_table").fetchone()[0]
        rows_added = final_count - initial_count
        
        # Suppression de la vue temporaire
        self.conn.execute("DROP VIEW temp_upsert")
        
        # Logging
        self.logger.info(f"Upsert terminé: {rows_added} lignes ajoutées, {rows_updated} lignes mises à jour")
        
        return {
            'rows_added': rows_added,
            'rows_updated': rows_updated
        }
    
    # Méthode de mise à jour d'une table de dimension
    def _update_dimension_values(self, dimension_name: str, values: pd.Series) -> int:
        """
        Mise à jour des valeurs d'une table de dimension.
        
        Args:
            dimension_name: Nom de la dimension
            values: Série avec les nouvelles valeurs
            
        Returns:
            Nombre de nouvelles valeurs ajoutées
        """
        with self._lock:
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
                existing_values = set(
                    self.conn.execute(f"SELECT value FROM {table_name}").fetchdf()['value']
                )
            except:
                existing_values = set()
            
            # Identification des nouvelles valeurs
            unique_values = set(values.dropna().unique())
            new_values = unique_values - existing_values
            
            # Insertion des nouvelles valeurs
            values_added = 0
            for value in new_values:
                try:
                    self.conn.execute(f"""
                        INSERT INTO {table_name} (value, label) 
                        VALUES (?, ?)
                        ON CONFLICT (value) DO NOTHING
                    """, [str(value), str(value)])
                    values_added += 1
                except Exception as e:
                    # Logging
                    self.logger.warning(f"Impossible d'insérer {value} dans {table_name}: {e}")
            
            if values_added > 0:
                # Logging
                self.logger.info(f"Ajout de {values_added} nouvelles valeurs à {table_name}")
            
            return values_added