# Importation des modules
# Modules de base
import json
import os
import threading
import time
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

# DuckDB
import duckdb
import narwhals as nw
import polars as pl
from narwhals.typing import IntoDataFrame

# Import des utilitaires
from ...utils.hierarchy import validate_hierarchy_forest
from ...utils.logger import _init_logger
from ...utils.sql import qualify_table, quote_ident, resolve_catalog
from ...utils.types import (
    COLUMN_METADATA_KEYS,
    map_python_to_sql_type,
    normalize_default_aggregation,
    resolve_sql_type_conflict,
)


# Classe contenant des opérations utilitaires de base sur la base de données au schéma
# (table des faits - méta-données - méta-données du jeu de résultats)
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
                ``dataset_metadata`` tables to operate on. A single catalog can host
                several schemas (one per result set). Defaults to ``'main'``.
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

        # Alias de catalogue effectif : l'alias n'est utilisé pour qualifier les
        # tables que s'il correspond à une base réellement attachée. Les connexions
        # in-memory des tests n'attachent aucun catalogue : la qualification retombe
        # alors sur le seul schéma.
        self._catalog = resolve_catalog(self.conn, self.catalog_alias)

        # Initialisation du logger nommé pour traçabilité des opérations.
        # Chemin par défaut centralisé dans utils.logger : <cwd>/logs/<name>.log.
        self.logger = _init_logger(filename=log_filename, name="base_schema_manager")

        # Cache thread-safe pour optimiser les accès aux métadonnées
        self._metadata_cache: nw.DataFrame[Any] | None = None
        self._cache_lock = threading.RLock()

        # Indicateur de transaction DuckDB ouverte par ce gestionnaire.
        # Garde de ré-entrance : DuckDB rejette un BEGIN imbriqué, et une opération
        # publique peut en appeler une autre (ex. delete_columns depuis le nettoyage
        # de delete_rows).
        self._in_transaction = False

    # Point d'accroche transactionnel unique des opérations publiques
    @contextmanager
    def _transaction(
        self, operation: str, use_transaction: bool = True
    ) -> Iterator[None]:
        """Run one public operation inside a single DuckDB transaction.

        Opens a ``BEGIN`` on entry, ``COMMIT`` on normal exit and ``ROLLBACK`` on
        any exception, which is then re-raised to the caller. This is the single
        entry/exit hook of every public write operation (``update_database``,
        ``add_columns``, ``delete_rows``, ``delete_columns``): post-write
        maintenance (``rewrite_data_files``, ``merge_adjacent_files``) must run
        **after** the commit, hence outside of this block.

        A nested call — a public operation invoked from within another one — reuses
        the transaction already open instead of issuing a second ``BEGIN``, which
        DuckDB rejects.

        Args:
            operation: Name of the operation, used in the log lines (e.g.
                ``'update'``, ``'delete_rows'``).
            use_transaction: Whether to actually wrap the block in a transaction.
                When False the block runs in autocommit mode: each statement is
                durable as soon as it executes and a failure leaves partial state.
                Defaults to True.

        Yields:
            None: control returns to the caller's ``with`` block.

        Raises:
            Exception: any exception raised inside the block, re-raised after the
                ``ROLLBACK``.

        Examples:
            >>> with manager._transaction('update'):
            ...     manager._update_metadata_safe(df)
        """
        # Transaction déjà ouverte ou mode autocommit : exécution directe du bloc,
        # la gestion transactionnelle revient à l'appelant (ou à personne).
        if not use_transaction or self._in_transaction:
            yield
            return

        # Ouverture de la transaction
        start_time = time.time()
        self.conn.begin()
        self._in_transaction = True
        # Logging
        self.logger.debug(f"BEGIN {operation} ({self.schema})")

        try:
            yield
        except Exception as e:
            # Annulation : la base revient à son état d'avant le BEGIN
            self.conn.rollback()
            self._in_transaction = False
            # Logging : l'étape atteinte est portée par le message de l'exception
            self.logger.error(
                f"ROLLBACK {operation} ({self.schema}) after"
                f" {time.time() - start_time:.2f}s: {e}"
            )
            raise
        else:
            # Validation de la transaction
            self.conn.commit()
            self._in_transaction = False
            # Logging
            self.logger.debug(
                f"COMMIT {operation} ({self.schema}) in {time.time() - start_time:.2f}s"
            )

    # Méthode de qualification d'un nom de table par le schéma (et le catalogue)
    def _qualified(self, table: str) -> str:
        """
        Return a table name qualified by this manager's schema and catalog.

        Args:
            table: Bare table name (e.g. ``'fact_table'``, ``'metadata'``).

        Returns:
            The quoted, qualified identifier targeting :attr:`schema` (and the
            catalog alias when one is actually attached).

        Example:
            >>> manager.schema = 'predictions'
            >>> manager._qualified('fact_table')
            '"predictions"."fact_table"'
        """
        # Délégation à l'utilitaire central de qualification, en propageant l'alias
        # de catalogue effectif (None pour les connexions in-memory des tests).
        return qualify_table(table, self.schema, self._catalog)

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
                    # Table absente : DataFrame vide typé sur le schéma cible.
                    # Les dtypes explicites sont indispensables pour que les filtres
                    # booléens des appelants restent valides sur un frame vide.
                    self._metadata_cache = nw.from_native(
                        pl.DataFrame(
                            schema={
                                "name": pl.String,
                                "label": pl.String,
                                "sql_type": pl.String,
                                "is_categorical": pl.Boolean,
                                "is_categorical_forced": pl.Boolean,
                                "is_primary_key": pl.Boolean,
                                "parent_name": pl.String,
                                "unit": pl.String,
                                "display_format": pl.String,
                                "family": pl.String,
                                "description": pl.String,
                                "default_aggregation": pl.String,
                            }
                        ),
                        eager_only=True,
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

    # Méthode de vérification du statut catégoriel d'une colonne
    def _is_categorical_column(self, column: str) -> bool:
        """
        Check if a column is flagged as categorical in the metadata table.

        The flag is UI metadata only: it says the column is browsed through a menu,
        never that its values are stored differently.

        Args:
            column: Column name

        Returns:
            True if column is categorical

        Example:
            >>> manager._is_categorical_column('region')
            True
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

    # Méthode de lecture des colonnes de tri physique (cluster_by)
    def _get_cluster_by_columns(self) -> list[str] | None:
        """
        Get the physical sort key of the fact table from ``dataset_metadata``.

        Returns:
            The ``cluster_by`` column list, or None if ``dataset_metadata`` doesn't
            exist yet or its ``cluster_by`` value is NULL.

        Example:
            >>> manager._get_cluster_by_columns()
            ['date', 'region']
        """
        # Absence de table dataset_metadata (ex. schéma pas encore construit) : pas de
        # tri connu
        if not self._table_exists("dataset_metadata"):
            return None
        # Lecture de l'unique ligne de dataset_metadata
        result = self.conn.execute(
            f"SELECT cluster_by FROM {self._qualified('dataset_metadata')}"
        ).fetchone()
        if result is None or result[0] is None:
            return None
        # Décodage de la liste JSON persistée
        decoded: list[str] = json.loads(result[0])
        return decoded

    # Méthode de mise à jour du tri physique (cluster_by) sur une base existante
    def update_cluster_by(self, columns: list[str]) -> None:
        """
        Correct the physical sort key (``cluster_by``) recorded for the fact table.

        Only updates ``dataset_metadata.cluster_by``: existing data files are left
        untouched, so file pruning does not improve until the fact table is
        physically reordered (``DuckLakeMaintenance.recluster``). Future writes
        (inserts, upserts) sort themselves by the new value.

        Args:
            columns: Non-empty list of column names, in the desired sort order. Every
                column must exist in the fact table.

        Raises:
            ValueError: If ``columns`` is empty, or references a column absent from
                the fact table.

        Example:
            >>> manager.update_cluster_by(['date', 'region'])
        """
        # Une liste vide n'a pas de sens : ce n'est pas équivalent à "aucun tri" (qui
        # se représente par NULL, non par une correction explicite)
        if not columns:
            raise ValueError("columns must not be empty")

        # Validation de l'existence de chaque colonne dans la table des faits
        existing_columns = set(self._get_fact_table_columns())
        unknown = [c for c in columns if c not in existing_columns]
        if unknown:
            raise ValueError(
                f"cluster_by columns {unknown} do not exist in the fact table"
            )

        # Écriture de la nouvelle valeur, sans réécriture des données
        self.conn.execute(
            f"UPDATE {self._qualified('dataset_metadata')} SET cluster_by = ?",
            [json.dumps(columns)],
        )

        # Logging
        self.logger.info(f"Updated cluster_by to {columns}")

    # Méthode de retrait d'une colonne supprimée du tri physique (cluster_by)
    def _remove_from_cluster_by(self, column: str) -> None:
        """
        Remove ``column`` from ``dataset_metadata.cluster_by`` if it is part of it.

        Called after a column has been dropped from the fact table
        (``DatabaseDeleter.delete_columns``): a ``cluster_by`` referencing a column
        that no longer exists would break every future sorted write. When the
        removal empties the list, ``cluster_by`` is reset to ``NULL`` (no known
        sort key) rather than an empty JSON array.

        Args:
            column: Name of the column that was just dropped.

        Example:
            >>> manager._remove_from_cluster_by('region')
        """
        # Valeur courante (None si non définie ou table absente)
        cluster_by = self._get_cluster_by_columns()
        if not cluster_by or column not in cluster_by:
            return

        # Retrait de la colonne, écriture directe (la colonne n'existe déjà plus
        # dans la table des faits : update_cluster_by refuserait la validation
        # d'existence)
        remaining = [c for c in cluster_by if c != column]
        new_value = json.dumps(remaining) if remaining else None
        self.conn.execute(
            f"UPDATE {self._qualified('dataset_metadata')} SET cluster_by = ?",
            [new_value],
        )

        # Logging
        self.logger.warning(
            f"Column {column!r} removed from cluster_by (was {cluster_by}); "
            f"physical sort key is now {remaining or None}"
        )

    # Méthodes de gestion des métadonnées
    # Méthode d'ajout d'une colonne aux méta-données
    def _add_column_to_metadata(
        self, column: str, df: IntoDataFrame, label: str | None = None
    ) -> None:
        """
        Add a new column to the metadata table, or refresh an existing row.

        On an existing row the producer-owned fields are preserved: a ``label``
        already recorded is never overwritten by a data update, and a column whose
        categorical status was forced keeps it.

        Args:
            column: Column name
            df: DataFrame containing the column (any narwhals-compatible backend)
            label: Custom label for the column. If None, a label is derived from the
                column name on insertion, and the recorded label is kept on update.

        Example:
            >>> manager._add_column_to_metadata('score', df)
        """
        # Conversion vers narwhals pour un accès uniforme au schéma
        df_nw = nw.from_native(df, eager_only=True)
        # Extraction du type narwhals de la colonne
        dtype_obj = df_nw.schema[column]
        # Conversion du type narwhals en SQL
        sql_type = map_python_to_sql_type(dtype_obj)
        # Statut catégoriel : colonne textuelle dont la cardinalité respecte le seuil.
        # Les valeurs manquantes sont exclues du comptage des modalités.
        is_categorical = (
            isinstance(dtype_obj, (nw.String, nw.Categorical, nw.Enum))
            and self.categorical_threshold is not None
            and df_nw[column].drop_nulls().n_unique() <= self.categorical_threshold
        )

        # Nom qualifié de la table de métadonnées
        metadata_table = self._qualified("metadata")

        # Création de la table metadata si elle n'existe pas.
        self.conn.execute(f"""
            CREATE TABLE IF NOT EXISTS {metadata_table} (
                name VARCHAR,
                label VARCHAR,
                sql_type VARCHAR,
                is_categorical BOOLEAN,
                is_categorical_forced BOOLEAN DEFAULT FALSE,
                is_primary_key BOOLEAN DEFAULT FALSE,
                parent_name VARCHAR,
                unit VARCHAR,
                display_format VARCHAR,
                family VARCHAR,
                description VARCHAR,
                default_aggregation VARCHAR
            )
        """)

        # Upsert manuel
        # Vérification de l'existence de la colonne avant d'insérer ou de mettre à jour.
        _row = self.conn.execute(
            f"SELECT COUNT(*) FROM {metadata_table} WHERE name = ?", [column]
        ).fetchone()
        existing_count = _row[0] if _row is not None else 0

        if existing_count == 0:
            # Colonne absente : insertion.
            # Libellé par défaut dérivé du nom technique, statut jamais forcé (la
            # colonne est découverte, pas déclarée par le producteur). Les champs
            # d'UI (unit, display_format, family, description, default_aggregation)
            # ne sont pas listés : ils prennent donc NULL, seul le producteur de
            # métadonnées pouvant les renseigner via update_column_metadata.
            insert_label = (
                label if label is not None else column.replace("_", " ").title()
            )
            self.conn.execute(
                f"""
                INSERT INTO {metadata_table} (name, label, sql_type,
                is_categorical, is_categorical_forced, is_primary_key)
                VALUES (?, ?, ?, ?, FALSE, FALSE)
                """,
                [column, insert_label, sql_type, is_categorical],
            )
        else:
            # Colonne déjà présente : mise à jour des seuls champs dérivés des données.
            # COALESCE préserve un libellé déjà renseigné par le producteur ; le CASE
            # protège un statut catégoriel explicitement forcé. Les champs d'UI ne
            # sont jamais touchés ici : un update de données ne doit pas les remettre
            # à NULL.
            self.conn.execute(
                f"""
                UPDATE {metadata_table}
                SET label = COALESCE(?, label),
                    sql_type = ?,
                    is_categorical = CASE WHEN is_categorical_forced
                                          THEN is_categorical ELSE ? END
                WHERE name = ?
                """,
                [label, sql_type, is_categorical, column],
            )

        # Invalidation du cache
        self._invalidate_metadata_cache()

        # Logging
        self.logger.info(f"Added/updated column {column} in metadata")

    # Méthode de renseignement ou de correction des champs d'UI d'une colonne
    def update_column_metadata(self, column: str, **fields: str | None) -> None:
        """
        Set or correct the producer-owned UI fields of an existing column.

        Only ``label``, ``parent_name``, ``unit``, ``display_format``, ``family``,
        ``description`` and ``default_aggregation`` may be updated. The update
        touches nothing else, so a later data update never has to rebuild the base
        to fix a wrong unit or format. ``default_aggregation`` is validated (and
        upper-cased) before the write. Setting ``parent_name`` declares (or
        corrects) a column hierarchy link: the parent column must already
        exist in metadata, the resulting graph must stay a forest (no cycle), and
        both ``column`` and its new parent are forced categorical, with a warning,
        if either is not already.

        Args:
            column: Name of the column, which must already have a row in the
                metadata table.
            **fields: Field/value pairs among ``label``, ``parent_name``, ``unit``,
                ``display_format``, ``family``, ``description`` and
                ``default_aggregation``. A value of ``None`` clears the field.

        Raises:
            ValueError: If a field name is not one of the allowed fields, if
                ``default_aggregation`` is invalid, if the column has no row in the
                metadata table, if a non-``None`` ``parent_name`` references a
                column absent from metadata, or if it would create a cycle in the
                ``parent_name`` graph.

        Example:
            >>> manager.update_column_metadata(
            ...     'value', unit='€', display_format=',.2f',
            ...     default_aggregation='sum')
            >>> manager.update_column_metadata('commune', parent_name='departement')
        """
        # Contrôle des champs autorisés
        unknown = set(fields) - COLUMN_METADATA_KEYS
        if unknown:
            raise ValueError(
                f"Unknown metadata field(s) {sorted(unknown)}; allowed fields are "
                f"{sorted(COLUMN_METADATA_KEYS)}"
            )

        # Aucun champ fourni : rien à écrire
        if not fields:
            return

        # Normalisation et validation de l'agrégation par défaut
        if "default_aggregation" in fields:
            fields["default_aggregation"] = normalize_default_aggregation(
                fields["default_aggregation"]
            )

        # Vérification de l'existence d'une ligne pour la colonne visée
        metadata_table = self._qualified("metadata")
        exists = False
        if self._table_exists("metadata"):
            _row = self.conn.execute(
                f"SELECT COUNT(*) FROM {metadata_table} WHERE name = ?", [column]
            ).fetchone()
            exists = _row is not None and _row[0] > 0
        if not exists:
            raise ValueError(
                f"Column {column!r} has no row in the metadata table; "
                "update_column_metadata only corrects existing columns"
            )

        # Validation spécifique à parent_name : existence de la colonne parente dans
        # l'état courant de metadata, puis détection de cycle sur le graphe complet.
        # Extraction du nouveau parent
        new_parent = fields.get("parent_name")
        if "parent_name" in fields and new_parent is not None:
            # Vérification que la colonne parent existe dans la table des
            # métadonnées (et est donc une colonne valide de la table des faits)
            _prow = self.conn.execute(
                f"SELECT COUNT(*) FROM {metadata_table} WHERE name = ?", [new_parent]
            ).fetchone()
            # Cas d'erreur si la colonne n'est pas trouvée
            if _prow is None or _prow[0] == 0:
                raise ValueError(
                    f"Parent column {new_parent!r} has no row in the metadata table"
                )
            # Extraction des paires parent/enfant
            current_rows = self.conn.execute(
                f"SELECT name, parent_name FROM {metadata_table}"
            ).fetchall()
            parent_of = {name: parent for name, parent in current_rows}
            parent_of[column] = new_parent
            # Validation de la hiérarchie
            validate_hierarchy_forest(parent_of)

        # Construction de la clause SET (identifiants entre guillemets, valeurs liées)
        set_clause = ", ".join(f"{quote_ident(name)} = ?" for name in fields)
        params = [*fields.values(), column]
        self.conn.execute(
            f"UPDATE {metadata_table} SET {set_clause} WHERE name = ?", params
        )

        # Forçage catégoriel des deux extrémités d'un lien de hiérarchie nouvellement
        # déclaré.
        if "parent_name" in fields and new_parent is not None:
            for hierarchy_col in (column, new_parent):
                _crow = self.conn.execute(
                    f"SELECT is_categorical FROM {metadata_table} WHERE name = ?",
                    [hierarchy_col],
                ).fetchone()
                if _crow is not None and not _crow[0]:
                    # Forçage du statut catégoriel
                    self.conn.execute(
                        f"UPDATE {metadata_table} SET is_categorical = TRUE, "
                        "is_categorical_forced = TRUE WHERE name = ?",
                        [hierarchy_col],
                    )
                    # Warning
                    warnings.warn(
                        f"Column {hierarchy_col!r} is part of a column hierarchy but"
                        f" is not categorical; forcing is_categorical=True",
                        UserWarning,
                        stacklevel=2,
                    )
                    # Logging
                    self.logger.info(
                        f"Forced is_categorical=True for hierarchy column"
                        f" {hierarchy_col!r}"
                    )

        # Invalidation du cache
        self._invalidate_metadata_cache()

        # Logging
        self.logger.info(
            f"Updated metadata fields {sorted(fields)} for column {column}"
        )

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

    # Méthode de détachement des colonnes enfants d'une colonne parente supprimée
    def _clear_child_parent_references(self, column: str) -> list[str]:
        """
        Clear ``parent_name`` on every column whose hierarchy parent is ``column``.

        Used when a column that is the parent of another column (§2.5) is deleted
        with ``cascade=True``: rather than leaving children pointing at a column
        that no longer exists, their ``parent_name`` is reset to ``NULL`` and a
        warning is logged.

        Args:
            column: Name of the column about to be dropped, used as the parent
                reference to detach.

        Returns:
            list[str]: Names of the child columns that were detached. Empty when
            ``column`` was not a hierarchy parent.

        Example:
            >>> manager._clear_child_parent_references('region')
            ['departement']
        """
        # Table des méta-données
        metadata_table = self._qualified("metadata")
        # Extraction des enfants associés à la colonne
        children = [
            row[0]
            for row in self.conn.execute(
                f"SELECT name FROM {metadata_table} WHERE parent_name = ?", [column]
            ).fetchall()
        ]
        # Retrait du parent
        if children:
            self.conn.execute(
                f"UPDATE {metadata_table} SET parent_name = NULL WHERE parent_name = ?",
                [column],
            )
            # Invalidation du cache
            self._invalidate_metadata_cache()
            # Logging
            self.logger.warning(
                f"Column {column!r} was the parent of {children}; their"
                f" parent_name was cleared to NULL (cascade=True)"
            )
        return children

    # Méthodes de résolution des conflits de types
    def _resolve_type_conflicts(
        self, column: str, df: nw.DataFrame[Any], current_metadata: nw.DataFrame[Any]
    ) -> None:
        """
        Resolve a type conflict on SQL types, widening only.

        The recorded ``metadata.sql_type`` is never narrowed: a stored ``BIGINT``
        survives a batch of ``Int32``. Non-ordered types (temporal, decimal, binary)
        are left untouched and reported. The fact table column is widened alongside
        the metadata so both stay consistent.

        Args:
            column: Column name
            df: DataFrame with new data (narwhals)
            current_metadata: Current metadata (narwhals)

        Example:
            >>> manager._resolve_type_conflicts('amount', df, metadata)
        """
        # Identification du type SQL actuellement enregistré
        matching = current_metadata.filter(nw.col("name") == column)["sql_type"]
        if len(matching) == 0:
            return None
        current_type = str(matching[0])

        # Conservation du type existant si toutes les nouvelles valeurs sont nulles :
        # narwhals infère alors 'Null', que map_python_to_sql_type replie sur VARCHAR,
        # ce qui promouvrait à tort une colonne numérique connue en texte.
        if df[column].is_null().all():
            return None

        # Type SQL du lot entrant
        new_type = map_python_to_sql_type(df.schema[column])
        # Ne fait rien si inchangé
        if current_type == new_type:
            return None

        # Résolution par élargissement uniquement
        resolved_type = resolve_sql_type_conflict(current_type, new_type)
        if resolved_type is None:
            # Type non ordonné ou lot plus étroit : le type enregistré est conservé
            self.logger.info(
                f"Type conflict for {column}: incoming {new_type} does not widen"
                f" stored {current_type}; metadata left unchanged"
            )
            return None

        # Élargissement de la colonne de la table des faits, pour que le type
        # enregistré et le type physique restent cohérents (contrôlé par l'auditeur).
        # Échec non bloquant : la métadonnée reste la référence déclarative.
        try:
            self.conn.execute(
                f"ALTER TABLE {self._qualified('fact_table')}"
                f" ALTER {quote_ident(column)} SET DATA TYPE {resolved_type}"
            )
        except Exception as e:
            self.logger.warning(
                f"Could not widen fact_table.{column} to {resolved_type}: {e}"
            )

        # Mise à jour du type SQL enregistré
        self.conn.execute(
            f"""
            UPDATE {self._qualified("metadata")}
            SET sql_type = ?
            WHERE name = ?
        """,
            [resolved_type, column],
        )
        # Invalidation du cache
        self._invalidate_metadata_cache()
        # Logging
        self.logger.info(
            f"Type conflict resolution for {column}: {current_type} -> {resolved_type}"
        )

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
                    f"WHERE {quote_ident(column)} IS NOT NULL"
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

    # Méthode d'actualisation du statut catégoriel des colonnes textuelles
    def _refresh_categorical_flags(self) -> list[str]:
        """
        Recompute the ``is_categorical`` flag of every eligible VARCHAR column.

        The status is pure UI metadata: it is derived from the current distinct
        count of the fact table compared against ``categorical_threshold``, and only
        a plain ``UPDATE metadata`` is issued — the fact table is never rewritten.
        Columns whose status was forced by the producer
        (``is_categorical_forced``) are skipped, and an ``UPDATE`` is emitted only
        when the boolean actually changes.

        Returns:
            List of column names whose flag was flipped. Empty when nothing changed
            or when no threshold is configured.

        Example:
            >>> manager._refresh_categorical_flags()
            ['high_cardinality']
        """
        # Liste des colonnes dont le statut a effectivement basculé
        changed: list[str] = []

        # Absence de seuil : le statut catégoriel n'est pas ré-évaluable
        if self.categorical_threshold is None:
            return changed

        try:
            # Chargement des méta-données courantes
            current_metadata = self._load_current_metadata()
            if len(current_metadata) == 0:
                return changed

            # Sélection des colonnes textuelles dont le statut n'a pas été forcé
            candidates = current_metadata.filter(
                (nw.col("sql_type") == "VARCHAR") & (~nw.col("is_categorical_forced"))
            )

            # Colonnes réellement présentes dans la table des faits
            fact_columns = set(self._get_fact_table_columns())
            # Nom qualifié de la table des faits
            fact_table = self._qualified("fact_table")

            for col_name, was_categorical in zip(
                candidates["name"].to_list(),
                candidates["is_categorical"].to_list(),
            ):
                # Colonne absente de la table des faits : rien à recalculer
                if col_name not in fact_columns:
                    continue

                # Comptage des modalités sur l'état courant de la table des faits
                quoted_col = quote_ident(col_name)
                row = self.conn.execute(
                    f"SELECT COUNT(DISTINCT {quoted_col}) FROM {fact_table}"
                    f" WHERE {quoted_col} IS NOT NULL"
                ).fetchone()
                n_distinct = int(row[0]) if row is not None else 0

                # Statut attendu : au moins une modalité observée et seuil respecté.
                # Une colonne entièrement nulle n'est pas catégorielle, l'absence de
                # modalités ne constituant pas une information de cardinalité.
                is_categorical = 0 < n_distinct <= self.categorical_threshold

                # Écriture uniquement en cas de bascule effective
                if bool(was_categorical) is is_categorical:
                    continue

                # Mise à jour du booléen et invalidation du cache
                self._update_categorical_status(col_name, is_categorical)
                changed.append(col_name)
                # Logging
                self.logger.info(
                    f"Categorical status of '{col_name}' set to {is_categorical}"
                    f" ({n_distinct} distinct values, threshold"
                    f" {self.categorical_threshold})"
                )

            return changed

        except Exception as e:
            # Logging
            self.logger.error(f"Error refreshing categorical flags: {e}")
            return changed

    # Méthode d'horodatage de la dernière écriture réussie
    def _touch_dataset_metadata(self) -> None:
        """
        Stamp ``dataset_metadata.updated_at`` with the current timestamp.

        The table holds exactly one row per schema, so no ``WHERE`` clause is
        needed. Failure is non-blocking: the timestamp is descriptive metadata and
        must never invalidate an otherwise successful write.

        Example:
            >>> manager._touch_dataset_metadata()
        """
        try:
            # Horodatage de la dernière écriture réussie.
            # Valeur liée en Python plutôt que via now() : la colonne est un
            # TIMESTAMP sans fuseau, là où now() renvoie un TIMESTAMP WITH TIME ZONE.
            self.conn.execute(
                f"UPDATE {self._qualified('dataset_metadata')} SET updated_at = ?",
                [datetime.now()],
            )
        except Exception as e:
            # Erreur non bloquante : l'horodatage ne conditionne pas l'écriture
            self.logger.warning(f"Could not stamp dataset_metadata.updated_at: {e}")

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
