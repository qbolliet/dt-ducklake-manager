# Importation des modules
# Modules de base
import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

# DuckDB
import duckdb
import narwhals as nw
import polars as pl
from narwhals.typing import IntoDataFrame

# Import des utilitaires
from ...utils.logger import _init_logger
from ...utils.sql import qualify_table
from ...utils.types import map_python_to_sql_type

# Emplacement du fichier
FILE_PATH = Path(os.path.abspath(__file__))


# Classe contenant des opérations utilitaires de base sur la base de données au schéma
# métadonnées - table des faits - tables de dimensions
class BaseSchemaManager(ABC):
    """
    Base class for database schema management operations.

    Provides common functionality for metadata management, column operations,
    and database introspection. All concrete managers should inherit from this class.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection
        categorical_threshold (int): Threshold for categorical determination
        schema (str): DuckLake schema holding this result set's tables
        catalog_alias (str): Alias of the attached DuckLake catalog (``ATTACH ...
            AS <alias>``), carried alongside ``schema`` so table references can be
            fully qualified by the catalog.
        logger: Logger instance for operation tracking
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        categorical_threshold: int | None = 50,
        log_filename: str | os.PathLike[str] | None = None,
        schema: str = "main",
        catalog_alias: str = "db",
    ):
        """
        Initialize the base schema manager.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory DuckDB
                connection is created (useful for unit tests only).
            categorical_threshold: Threshold for determining categorical variables.
            log_filename: Path to log file.
            schema: DuckLake schema holding the ``fact_table``, ``metadata`` and
                ``dim_*`` tables to operate on. A single catalog can host several
                schemas (one per result set). Defaults to ``'main'``.
            catalog_alias: Alias of the attached DuckLake catalog, matching the
                one passed to ``DuckLakeConnector`` (``ATTACH ... AS <alias>``).
                Carried alongside ``schema`` so that table references can be
                qualified by the catalog rather than resolved against the
                connection's current catalog. Defaults to ``'db'``.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> manager = ConcreteManager(conn, categorical_threshold=30)
            >>> # Cibler un schéma dédié dans le même catalogue
            >>> manager = ConcreteManager(conn, schema='predictions')
        """
        # Initialisation de la connexion DuckLake.
        # Le fallback :memory: est réservé aux tests unitaires ; en production la
        # connexion doit toujours être fournie via DuckLakeConnector.connect().
        self.conn = connection if connection is not None else duckdb.connect(":memory:")

        # Seuil pour déterminer si une variable est catégorielle
        self.categorical_threshold = categorical_threshold

        # Schéma DuckLake cible : toutes les requêtes qualifient les tables par ce
        # schéma, permettant à plusieurs jeux de résultats de coexister dans un même
        # catalogue.
        self.schema = schema

        # Alias du catalogue DuckLake attaché : conservé au même titre que le schéma
        # afin de pouvoir qualifier les tables par le catalogue (indispensable dès
        # que plusieurs catalogues sont attachés à la même connexion).
        self.catalog_alias = catalog_alias

        # Initialisation du logger pour traçabilité des opérations
        if log_filename is None:
            log_filename = os.path.join(
                FILE_PATH.parents[3], "logs/base_schema_manager.log"
            )
        self.logger = _init_logger(filename=log_filename)

        # Cache thread-safe pour optimiser les accès aux métadonnées
        self._metadata_cache: nw.DataFrame[Any] | None = None
        self._cache_lock = threading.RLock()

    # Méthode de qualification d'un nom de table par le schéma du gestionnaire
    def _qualified(self, table: str) -> str:
        """
        Return a table name qualified by this manager's schema.

        Args:
            table: Bare table name (e.g. ``'fact_table'``, ``'dim_country'``).

        Returns:
            The ``'<schema>.<table>'`` identifier targeting :attr:`schema`.

        Example:
            >>> manager.schema = 'predictions'
            >>> manager._qualified('fact_table')
            'predictions.fact_table'
        """
        # Délégation à l'utilitaire central de qualification
        return qualify_table(table, self.schema)

    # Méthodes de gestion du cache des métadonnées
    # Méthode de chargement des méta-données
    def _load_current_metadata(self) -> nw.DataFrame[Any]:
        """
        Load current metadata from the database with thread-safe caching.

        Returns:
            DataFrame containing current metadata (narwhals)
        """
        with self._cache_lock:
            # Chargement de la table si elle n'est pas en cache
            if self._metadata_cache is None:
                try:
                    # Chargement via polars (backend interne) puis encapsulation
                    # narwhals
                    self._metadata_cache = nw.from_native(
                        self.conn.execute(
                            f"SELECT * FROM {self._qualified('metadata')}"
                        ).pl(),
                        eager_only=True,
                    )
                except Exception:
                    # Si la table n'existe pas encore, initialisation d'un DataFrame
                    # vide
                    self._metadata_cache = nw.from_dict(
                        {
                            "name": [],
                            "label": [],
                            "python_type": [],
                            "sql_type": [],
                            "is_categorical": [],
                            "is_primary_key": [],
                        },
                        native_namespace=pl,
                    )

            return self._metadata_cache.clone()

    # Méthode d'invalidation des méta-données mises en cache
    def _invalidate_metadata_cache(self) -> None:
        """
        Invalidate the metadata cache to force reload on next access.
        """
        with self._cache_lock:
            self._metadata_cache = None

    # Méthodes d'introspection de la base de données
    # Méthode d'extraction des colonnes de la table des faits
    # /!\ Doit être cohérent avec les colonnes dans meta-données[name]
    def _get_fact_table_columns(self) -> list[str]:
        """
        Get list of columns in fact table.

        Returns:
            List of column names
        """
        # Exécution de la requête
        result = self.conn.execute(
            f"DESCRIBE {self._qualified('fact_table')}"
        ).fetchall()
        return [row[0] for row in result]

    # Méthode d'extraction des colonnes catégorielles
    def _get_categorical_columns(self) -> list[str]:
        """
        Get list of categorical columns from metadata.

        Returns:
            List of categorical column names
        """
        # Exécution de la requête
        result = self.conn.execute(
            f"SELECT name FROM {self._qualified('metadata')} "
            "WHERE is_categorical IS TRUE"
        ).fetchall()
        return [row[0] for row in result]

    # Méthode de vérification de l'existance d'une table dans la base de données
    def _table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table to check

        Returns:
            True if table exists
        """
        # Exécution de la requête.
        # Filtrage par schéma indispensable : la même table (ex. 'fact_table') peut
        # exister dans plusieurs schémas du catalogue ; sans ce filtre, un schéma
        # voisin produirait un faux positif.
        row = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = ? AND table_schema = ?",
            [table_name, self.schema],
        ).fetchone()
        return row[0] > 0 if row is not None else False

    # Méthode de vérification de l'existence d'une colonne dans une table
    def _column_exists(self, column: str, table: str = "fact_table") -> bool:
        """
        Check if a column exists in specified table.

        Args:
            column: Column name
            table: Table name (defaults to fact_table)

        Returns:
            True if column exists
        """
        # Extraction des colonnes de la table (qualifiée par le schéma)
        columns = [
            row[0]
            for row in self.conn.execute(
                f"DESCRIBE {self._qualified(table)}"
            ).fetchall()
        ]
        return column in columns

    # Méthode de vérification si une colonne a une table de dimension (équivalent à
    # vérifier si elle est catégorielle)
    def _is_dimension_column(self, column: str) -> bool:
        """
        Check if a column is a dimension (categorical).

        Args:
            column: Column name

        Returns:
            True if column is categorical
        """
        # Recherche du statut catégoriel
        result = self.conn.execute(
            f"SELECT is_categorical FROM {self._qualified('metadata')} WHERE name = ?",
            [column],
        ).fetchone()
        return result[0] if result else False

    # Méthode de vérification si une colonne est marquée comme clé primaire
    def _is_primary_key_column(self, column: str) -> bool:
        """
        Check if a column is marked as primary key in metadata.

        Args:
            column: Column name to check

        Returns:
            True if column is a primary key

        Example:
            >>> manager._is_primary_key_column('user_id')
            True
        """
        # Recherche du statut de clé primaire dans les méta-données
        result = self.conn.execute(
            f"SELECT is_primary_key FROM {self._qualified('metadata')} WHERE name = ?",
            [column],
        ).fetchone()
        return result[0] if result else False

    # Méthode d'extraction de toutes les colonnes marquées comme clés primaires
    def _get_primary_key_columns(self) -> list[str]:
        """
        Get all column names that are marked as primary keys in metadata.

        Returns:
            List of primary key column names. Empty list if no primary keys defined.

        Example:
            >>> manager._get_primary_key_columns()
            ['user_id', 'timestamp']
        """
        # Requête pour récupérer les noms des colonnes marquées comme clés primaires
        result = self.conn.execute(
            f"SELECT name FROM {self._qualified('metadata')} "
            "WHERE is_primary_key IS TRUE"
        ).fetchall()
        return [row[0] for row in result]

    # Méthodes de gestion des métadonnées
    # Méthode d'ajout d'une colonne aux méta-données
    def _add_column_to_metadata(
        self, column: str, df: IntoDataFrame, label: str | None = None
    ) -> None:
        """
        Add a new column to metadata table.

        Args:
            column: Column name
            df: DataFrame containing the column (any narwhals-compatible backend)
            label: Custom label for the column. If None, defaults to formatted column
            name.
        """
        # Conversion vers narwhals pour un accès uniforme au schéma
        df_nw = nw.from_native(df, eager_only=True)
        # Extraction du type narwhals de la colonne
        dtype_obj = df_nw.schema[column]
        dtype_str = str(dtype_obj)
        # Conversion du type narwhals en SQL
        sql_type = map_python_to_sql_type(dtype_obj)
        # Définition si la variable est catégorielle (String avec cardinalité ≤ seuil)
        is_categorical = isinstance(dtype_obj, nw.String) and df_nw[
            column
        ].n_unique() <= (self.categorical_threshold or 0)

        # Nom qualifié de la table de métadonnées
        metadata_table = self._qualified("metadata")

        # Création de la table metadata si elle n'existe pas.
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {metadata_table} (
                name VARCHAR,
                label VARCHAR,
                python_type VARCHAR,
                sql_type VARCHAR,
                is_categorical BOOLEAN,
                is_primary_key BOOLEAN DEFAULT FALSE
            )
        """)

        # Insertion de la nouvelle colonne
        if label is None:
            label = column.replace("_", " ").title()

        # Upsert manuel
        # Vérification de l'existence de la colonne avant d'insérer ou de mettre à jour.
        _row = self.conn.execute(
            f"SELECT COUNT(*) FROM {metadata_table} WHERE name = ?", [column]
        ).fetchone()
        existing_count = _row[0] if _row is not None else 0

        if existing_count == 0:
            # Colonne absente : insertion
            self.conn.execute(
                f"""
                INSERT INTO {metadata_table} (name, label, python_type, sql_type,
                is_categorical, is_primary_key)
                VALUES (?, ?, ?, ?, ?, FALSE)
                """,
                [column, label, dtype_str, sql_type, is_categorical],
            )
        else:
            # Colonne déjà présente : mise à jour des champs variables
            self.conn.execute(
                f"""
                UPDATE {metadata_table}
                SET label = ?, python_type = ?, sql_type = ?, is_categorical = ?
                WHERE name = ?
                """,
                [label, dtype_str, sql_type, is_categorical, column],
            )

        # Invalidation du cache
        self._invalidate_metadata_cache()

        # Logging
        self.logger.info(f"Added/updated column {column} in metadata")

    # Méthode de mise à jour du statut catégoriel d'une donnée
    def _update_categorical_status(self, col_name: str, is_categorical: bool) -> None:
        """
        Update categorical status in metadata.

        Args:
            col_name: Column name
            is_categorical: New categorical status
        """
        # Exécution de la requête de mise à jour
        self.conn.execute(
            f"UPDATE {self._qualified('metadata')} "
            "SET is_categorical = ? WHERE name = ?",
            [is_categorical, col_name],
        )

        # Invalidation du cache
        self._invalidate_metadata_cache()

        # Logging
        self.logger.info(f"Updated categorical status for {col_name}: {is_categorical}")

    # Méthode de suppression des méta-données pour une colonne
    def delete_column_metadata(self, column_name: str) -> None:
        """
        Delete metadata for a specific column.

        Args:
            column_name: Name of the column
        """
        try:
            # Requête de suppression des méta-données
            delete_query = f"DELETE FROM {self._qualified('metadata')} WHERE name = ?"
            # Exécution de la requête
            self.conn.execute(delete_query, [column_name])

            # Invalidation du cache
            self._invalidate_metadata_cache()

            # Logging
            self.logger.info(f"Deleted metadata for column {column_name}")

        except Exception as e:
            # Logging
            self.logger.error(
                f"Failed to delete metadata for column {column_name}: {e}"
            )
            raise

    # Méthodes de résolution des conflits de types
    def _resolve_type_conflicts(
        self, column: str, df: nw.DataFrame[Any], current_metadata: nw.DataFrame[Any]
    ) -> None:
        """
        Resolve type conflicts using the least restrictive strategy.

        Args:
            column: Column name
            df: DataFrame with new data (narwhals)
            current_metadata: Current metadata (narwhals)
        """
        # Identification du type actuel de la colonne dans les métadonnées
        matching = current_metadata.filter(nw.col("name") == column)["python_type"]
        if len(matching) == 0:
            return None
        current_type = matching[0]
        # Identification du nouveau type (chaîne narwhals, ex. 'String', 'Int64',
        # 'Float64')
        new_type = str(df.schema[column])
        # Ne fait rien si inchangé
        if current_type == new_type:
            return None

        # Conservation du type existant si toutes les nouvelles valeurs sont nulles :
        # narwhals infère alors 'Null' sans information sur le type réel de la colonne,
        # ce qui écraserait à tort un type numérique connu.
        if df[column].is_null().all():
            return None

        # Hiérarchie des types narwhals (du plus contraignant au moins contraignant).
        # Les types entiers de largeur variable sont tous regroupés au niveau 2.
        type_hierarchy = {
            "Boolean": 1,
            "Int8": 2,
            "Int16": 2,
            "Int32": 2,
            "Int64": 2,
            "UInt8": 2,
            "UInt16": 2,
            "UInt32": 2,
            "UInt64": 2,
            "Float32": 3,
            "Float64": 3,
            "String": 4,
        }

        current_level = type_hierarchy.get(current_type, 0)
        new_level = type_hierarchy.get(new_type, 0)

        # Sélection du type le moins contraignant
        if new_level > current_level:
            # Le nouveau type est moins contraignant : mise à jour vers new_type
            resolved_type = new_type
            sql_type = map_python_to_sql_type(df.schema[column])
            # Mise à jour des métadonnées
            self.conn.execute(
                f"""
                UPDATE {self._qualified("metadata")}
                SET python_type = ?, sql_type = ?
                WHERE name = ?
            """,
                [resolved_type, sql_type, column],
            )
            # Invalidation du cache
            self._invalidate_metadata_cache()
            # Logging
            self.logger.info(
                f"Type conflict resolution for {column}: {current_type} ->"
                f" {resolved_type}"
            )
        # Si current_level >= new_level, le type existant est conservé sans modification

    # Méthode utilitaire pour les colonnes contenant uniquement des valeurs nulles
    def _get_null_only_columns(self) -> list[str]:
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
                query = (
                    f"SELECT COUNT(*) FROM {self._qualified('fact_table')} "
                    f"WHERE {column} IS NOT NULL"
                )
                _row = self.conn.execute(query).fetchone()
                non_null_count = _row[0] if _row is not None else 0
                # Ajout à la liste si ne contient que des colonnes nulles
                if non_null_count == 0:
                    null_only_columns.append(column)
            # Logging
            if null_only_columns:
                self.logger.info(
                    f"Columns containing only null values detected: {null_only_columns}"
                )
            return null_only_columns

        except Exception as e:
            self.logger.error(f"An error occurred while detecting null values: {e}")
            raise

    # Méthode de vérification du seuil catégoriel
    def _check_categorical_threshold(
        self, values: nw.Series[Any], threshold: int | None = None
    ) -> bool:
        """
        Check if values meet categorical threshold criteria.

        Args:
            values: Narwhals Series with column values
            threshold: Threshold to use (defaults to instance threshold)

        Returns:
            True if values should be categorical
        """
        effective_threshold = threshold or self.categorical_threshold
        # Suppression des valeurs nulles avant le comptage
        non_null_values = values.drop_nulls()
        # Une série entièrement nulle ne peut pas être considérée catégorielle :
        # l'absence de modalités observées ne constitue pas une information de
        # cardinalité.
        if len(non_null_values) == 0:
            return False
        # Aucun seuil défini : impossible de déterminer le statut catégoriel
        if effective_threshold is None:
            return False
        unique_count = non_null_values.n_unique()
        return unique_count <= effective_threshold

    @abstractmethod
    def validate_operation(self, operation_type: str, **kwargs: Any) -> bool:
        """
        Abstract method to validate operations before execution.

        Args:
            operation_type: Type of operation to validate
            **kwargs: Operation-specific parameters

        Returns:
            True if operation is valid
        """
        pass
