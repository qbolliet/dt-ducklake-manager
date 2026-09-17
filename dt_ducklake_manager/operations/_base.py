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
from narwhals.typing import IntoDataFrame

# Import des gestionnaires de maintenance
from ..maintenance.auditor import DatabaseAuditor, ValidationLevel, ValidationReport
from ..maintenance.compaction import DuckLakeMaintenance

# Import des utilitaires
from ..reporting import (
    OperationReport,
    _current_snapshot_id,
    _set_commit_message,
    _table_changes_counts,
    _table_info,
)
from ..utils.hierarchy import validate_hierarchy_forest
from ..utils.logger import _init_logger
from ..utils.sql import SchemaScoped, quote_ident, resolve_catalog
from ..utils.types import (
    COLUMN_METADATA_KEYS,
    empty_metadata_frame,
    map_python_to_sql_type,
    metadata_table_ddl,
    normalize_default_aggregation,
    resolve_sql_type_conflict,
)


# Classe contenant des opérations utilitaires de base sur la base de données au schéma
# (table des faits - méta-données - méta-données du jeu de résultats)
class BaseSchemaManager(SchemaScoped, ABC):
    """
    Base class for database schema management operations.

    Provides common functionality for metadata management, column operations,
    and database introspection. All concrete managers should inherit from this class.

    Attributes:
        conn (duckdb.DuckDBPyConnection): Database connection
        categorical_threshold (int | None): Threshold used once, when a column is
            created, to infer its ``is_categorical`` flag
        schema (str): DuckLake schema holding this result set's tables
        catalog_alias (str): Alias of the attached DuckLake catalog (``ATTACH ...
            AS <alias>``), carried alongside ``schema`` so table references can be
            fully qualified by the catalog.
        logger: Logger instance for operation tracking
        auditor (DatabaseAuditor | None): Auditor used for validation, set by the
            concrete managers (None when validation is disabled)
    """

    # Auditeur de la base, renseigné par les gestionnaires concrets
    auditor: DatabaseAuditor | None = None

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
            categorical_threshold: Maximum number of distinct non-null values for a
                textual column to be flagged categorical **when it is created**
                (never re-evaluated afterwards). None flags no new column.
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

        # Seuil d'inférence du statut catégoriel, appliqué à la création des colonnes
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

        # Mainteneur DuckLake, utilisé pour la compaction post-écriture. Le schéma est
        # repassé explicitement à chaque appel : self.schema peut changer après coup.
        self.maintenance = DuckLakeMaintenance(
            self.conn, catalog_alias=self.catalog_alias, schema=self.schema
        )

        # Cache thread-safe pour optimiser les accès aux métadonnées
        self._metadata_cache: nw.DataFrame[Any] | None = None
        self._cache_lock = threading.RLock()

        # Indicateur de transaction DuckDB ouverte par ce gestionnaire.
        # Garde de ré-entrance : DuckDB rejette un BEGIN imbriqué, et une opération
        # publique peut en appeler une autre.
        self._in_transaction = False

        # Rapport de la transaction actuellement ouverte (None hors transaction) :
        # permet à un appel imbriqué (use_transaction=False) de contribuer ses
        # propres colonnes/avertissements au rapport de l'opération englobante.
        self._current_report: OperationReport | None = None

        # Dernier rapport produit par ce gestionnaire, succès ou échec (partiel) :
        # exposé publiquement pour les opérations qui conservent leur contrat de
        # retour historique (ex. update_database -> bool).
        self.last_report: OperationReport | None = None

    # Point d'accroche transactionnel unique des opérations publiques
    @contextmanager
    def _transaction(
        self,
        operation: str,
        use_transaction: bool = True,
        table: str = "fact_table",
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> Iterator[OperationReport]:
        """Run one public operation inside a single DuckDB transaction.

        Opens a ``BEGIN`` on entry, ``COMMIT`` on normal exit and ``ROLLBACK`` on
        any exception, which is then re-raised to the caller. This is the single
        entry/exit hook of every public write operation (``update_database``,
        ``add_columns``, ``delete_rows``, ``delete_columns``): post-write
        maintenance (``rewrite_data_files``, ``merge_adjacent_files``) must run
        **after** the commit, hence outside of this block.

        It is also the single point where an :class:`OperationReport` is built and
        traced. Before-state (``rows_before``, ``files_before``/``bytes_before``,
        ``snapshot_before``) is captured on entry; ``run_id``/``commit_message``/
        ``commit_info`` are recorded via ``ducklake_set_commit_message`` right
        before ``COMMIT`` (skipped with a DEBUG log on a connection with no
        DuckLake catalog attached); after-state and the exact row-change counts
        (``ducklake_table_changes``) are captured right after ``COMMIT``. The
        caller may still refine ``files_after``/``bytes_after``/``snapshot_after``
        afterwards (e.g. once post-commit compaction has run) and fill
        ``report.maintenance``/``columns_added``/``columns_dropped``/
        ``metadata_changes`` before logging ``report.summary()`` — this method only
        guarantees the report exists, is attached to ``self.last_report``, and
        carries accurate before/after-commit state.

        A nested call — a public operation invoked from within another one — reuses
        the transaction already open instead of issuing a second ``BEGIN``, which
        DuckDB rejects, and reuses the enclosing operation's report so its own
        contributions (e.g. columns dropped by a nested ``delete_columns`` call)
        show up in the outer report too.

        Args:
            operation: Name of the operation, used in the log lines and as
                ``report.operation`` (e.g. ``'update'``, ``'delete_rows'``).
            use_transaction: Whether to actually wrap the block in a transaction.
                When False the block runs in autocommit mode: each statement is
                durable as soon as it executes and a failure leaves partial state.
                Defaults to True.
            table: Fact table name the report's file/row measurements target.
                Defaults to ``'fact_table'``.
            run_id: Caller-supplied run identifier, recorded as the resulting
                snapshot's ``author`` and as ``report.run_id``.
            commit_message: Caller-supplied commit message, recorded on the
                resulting snapshot.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``, alongside ``operation``/``schema`` and whatever of
                the report is already known before commit.

        Yields:
            OperationReport: the in-progress report, mutable by the caller (e.g.
            appending to ``columns_added``) for the remainder of the ``with`` block.

        Raises:
            Exception: any exception raised inside the block, re-raised after the
                ``ROLLBACK``. A partial report is attached to ``self.last_report``
                and logged at ERROR before re-raising.

        Examples:
            >>> with manager._transaction('update') as report:
            ...     manager._update_metadata_safe(df)
            ...     report.columns_added.append('score')
        """
        # Transaction déjà ouverte ou mode autocommit : exécution directe du bloc,
        # la gestion transactionnelle revient à l'appelant (ou à personne). Le
        # rapport de l'opération englobante est réutilisé s'il existe, pour que les
        # contributions de l'appel imbriqué y apparaissent ; sinon un rapport
        # minimal est construit (pas de mesure avant/après : aucune transaction
        # n'encadre le bloc pour en délimiter les bornes).
        if not use_transaction or self._in_transaction:
            report = self._current_report or OperationReport(
                operation=operation,
                schema=self.schema,
                run_id=run_id,
                started_at=datetime.now(),
                duration_seconds=0.0,
            )
            try:
                yield report
            except Exception as e:
                report.warnings.append(f"{operation} failed: {e}")
                raise
            finally:
                self.last_report = report
            return

        # Ouverture de la transaction
        start_time = time.time()
        started_at = datetime.now()

        # Avant-état : mesuré avant le BEGIN, donc hors de toute transaction.
        rows_before = self._count_rows(table)
        files_before, bytes_before, _, _ = self._table_info(table) or (0, 0, 0, 0)
        snapshot_before = self._current_snapshot()

        report = OperationReport(
            operation=operation,
            schema=self.schema,
            run_id=run_id,
            started_at=started_at,
            duration_seconds=0.0,
            rows_before=rows_before,
            files_before=files_before,
            bytes_before=bytes_before,
            snapshot_before=snapshot_before,
        )
        self._current_report = report

        # Connexion
        self.conn.begin()
        self._in_transaction = True
        # Logging
        self.logger.debug(f"BEGIN {operation} ({self.schema})")

        try:
            yield report
        except Exception as e:
            # Annulation : la base revient à son état d'avant le BEGIN
            self.conn.rollback()
            self._in_transaction = False
            self._current_report = None
            report.duration_seconds = time.time() - start_time
            report.warnings.append(f"{operation} failed: {e}")
            self.last_report = report
            # Logging : l'étape atteinte est portée par le message de l'exception.
            # Rapport partiel journalisé en ERROR (pas summary(), pensée pour un
            # succès).
            self.logger.error(
                f"{operation} FAILED after {report.duration_seconds:.2f}s"
                f" (schema={self.schema}, run_id={run_id}): {e}"
            )
            raise
        else:
            # Message de commit DuckLake (traçabilité du run) : dans la transaction,
            # avant COMMIT. Ignoré avec un DEBUG sur une connexion sans DuckLake réel.
            extra_info: dict[str, Any] = {
                "operation": operation,
                "schema": self.schema,
                **(commit_info or {}),
            }
            if report.columns_added:
                extra_info["columns_added"] = report.columns_added
            if report.columns_dropped:
                extra_info["columns_dropped"] = report.columns_dropped
            _set_commit_message(
                self.conn,
                self._catalog,
                run_id,
                commit_message,
                extra_info,
                self.logger,
            )

            # Validation de la transaction
            self.conn.commit()
            self._in_transaction = False
            self._current_report = None

            # Après-état : mesuré juste après COMMIT, avant toute compaction
            # post-écriture (qui reste hors de cette méthode, cf. docstring).
            report.rows_after = self._count_rows(table)
            files_after, bytes_after, _, _ = self._table_info(table) or (0, 0, 0, 0)
            report.files_after = files_after
            report.bytes_after = bytes_after
            report.snapshot_after = self._current_snapshot()
            changes = _table_changes_counts(
                self.conn,
                self._catalog,
                self.schema,
                table,
                snapshot_before,
                report.snapshot_after,
                self.logger,
            )
            # N'écrase les comptages que si une mesure DuckLake réelle a pu être
            # obtenue : sur une connexion sans catalogue réel (tests in-memory),
            # `changes` est vide et les comptages déjà déposés par l'appelant dans
            # le corps de la transaction (calculs Python exacts, ex. COUNT(*)
            # avant/après ciblés) doivent rester tels quels plutôt que d'être
            # remis à zéro.
            if changes:
                report.rows_inserted = changes.get("insert", 0)
                report.rows_updated = changes.get("update_postimage", 0)
                report.rows_deleted = changes.get("delete", 0)
            report.duration_seconds = time.time() - start_time
            self.last_report = report

            # Logging
            self.logger.debug(
                f"COMMIT {operation} ({self.schema}) in {report.duration_seconds:.2f}s"
            )

    # Méthode auxiliaire de lecture des statistiques de fichiers de la table
    def _table_info(self, table: str) -> tuple[int, int, int, int] | None:
        """Delegate to :func:`reporting._table_info` for this manager's table."""
        return _table_info(self.conn, self._catalog, self.schema, table, self.logger)

    # Méthode auxiliaire de lecture du snapshot_id courant
    def _current_snapshot(self) -> int | None:
        """Delegate to :func:`reporting._current_snapshot_id`."""
        return _current_snapshot_id(self.conn, self._catalog, self.logger)

    # Méthode de finalisation du rapport après la maintenance post-écriture
    def _finalize_report_after_write(
        self, report: OperationReport, table: str = "fact_table"
    ) -> None:
        """Re-capture file/byte/snapshot state after post-commit maintenance.

        ``_transaction`` already fills ``files_after``/``bytes_after``/
        ``snapshot_after`` right after ``COMMIT``, before any post-write
        compaction runs (``merge_files``/``rewrite_data_files``). Since compaction
        itself creates further DuckLake snapshots and rewrites files, the report's
        *final* file/byte/snapshot numbers — the ones a reader of the INFO summary
        line cares about — are re-measured here, once compaction (if any) has
        completed. Never overwrites ``report.maintenance``, populated separately by
        the caller from the compaction calls' own return values.

        Args:
            report: The report to update in place.
            table: Fact table name. Defaults to ``'fact_table'``.
        """
        # Extraction des infos de la table
        info = self._table_info(table)
        # Décomposition des informations de la table et ajout au rapport
        if info is not None:
            report.files_after, report.bytes_after, _, _ = info
        # Snapshot de la table
        snapshot = self._current_snapshot()
        # Ajout au rapport
        if snapshot is not None:
            report.snapshot_after = snapshot

    # Méthode de construction d'un rapport minimal pour un échec précoce
    def _early_failure_report(
        self, operation: str, run_id: str | None, warning: str
    ) -> OperationReport:
        """Build and store a minimal report for a failure before any transaction.

        Used by validation checks that reject an operation before
        ``_transaction`` ever opens (e.g. pre-operation auditor validation),
        so ``self.last_report``/the method's return value never sits at ``None``
        even on the earliest possible failure.

        Args:
            operation: Name of the operation that failed.
            run_id: Run identifier the caller was about to use, if any.
            warning: Human-readable reason, appended to ``report.warnings`` and
                logged at WARNING.

        Returns:
            OperationReport: the minimal report, also stored on ``self.last_report``.
        """
        # Création du rapport
        report = OperationReport(
            operation=operation,
            schema=self.schema,
            run_id=run_id,
            started_at=datetime.now(),
            duration_seconds=0.0,
            warnings=[warning],
        )
        # Logging
        self.logger.warning(warning)
        # Mise à jour du dernier rapport
        self.last_report = report
        return report

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
                    # Chargement Arrow (backend interne) puis encapsulation narwhals
                    self._metadata_cache = nw.from_native(
                        self.conn.execute(
                            f"SELECT * FROM {self._qualified('metadata')}"
                        ).to_arrow_table(),
                        eager_only=True,
                    )
                except Exception:
                    # Table absente : DataFrame vide typé sur le schéma cible
                    self._metadata_cache = empty_metadata_frame()

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

        ``is_categorical`` is inferred once, on insertion (textual column whose
        distinct non-null count is ``<= categorical_threshold``). On an existing row
        only ``sql_type`` is refreshed: the recorded ``label`` and
        ``is_categorical`` are never overwritten by a data update.

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
        # Statut catégoriel, calculé une seule fois à la création de la colonne :
        # colonne textuelle dont la cardinalité (hors valeurs manquantes) respecte le
        # seuil.
        is_categorical = (
            isinstance(dtype_obj, (nw.String, nw.Categorical, nw.Enum))
            and self.categorical_threshold is not None
            and df_nw[column].drop_nulls().n_unique() <= self.categorical_threshold
        )

        # Nom qualifié de la table de métadonnées
        metadata_table = self._qualified("metadata")

        # Création de la table metadata si elle n'existe pas.
        self.conn.execute(metadata_table_ddl(metadata_table, if_not_exists=True))

        # Upsert manuel
        # Vérification de l'existence de la colonne avant d'insérer ou de mettre à jour.
        _row = self.conn.execute(
            f"SELECT COUNT(*) FROM {metadata_table} WHERE name = ?", [column]
        ).fetchone()
        existing_count = _row[0] if _row is not None else 0

        if existing_count == 0:
            # Colonne absente : insertion.
            # Libellé par défaut dérivé du nom technique. Les champs
            # d'UI (unit, display_format, family, description, default_aggregation)
            # ne sont pas listés : ils prennent donc NULL, seul le producteur de
            # métadonnées pouvant les renseigner via update_column_metadata.
            insert_label = (
                label if label is not None else column.replace("_", " ").title()
            )
            self.conn.execute(
                f"""
                INSERT INTO {metadata_table} (name, label, sql_type,
                is_categorical, is_primary_key)
                VALUES (?, ?, ?, ?, FALSE)
                """,
                [column, insert_label, sql_type, is_categorical],
            )
        else:
            # Colonne déjà présente : mise à jour du seul type SQL. COALESCE préserve
            # un libellé déjà renseigné ; le statut catégoriel et les champs d'UI ne
            # sont jamais touchés par un update de données.
            self.conn.execute(
                f"""
                UPDATE {metadata_table}
                SET label = COALESCE(?, label),
                    sql_type = ?
                WHERE name = ?
                """,
                [label, sql_type, column],
            )

        # Invalidation du cache
        self._invalidate_metadata_cache()

        # Logging
        self.logger.info(f"Added/updated column {column} in metadata")

    # Méthode de renseignement ou de correction des champs d'UI d'une colonne
    def update_column_metadata(self, column: str, **fields: str | bool | None) -> None:
        """
        Set or correct the producer-owned UI fields of an existing column.

        Only ``label``, ``parent_name``, ``unit``, ``display_format``, ``family``,
        ``description``, ``default_aggregation`` and ``is_categorical`` may be
        updated. ``is_categorical`` is inferred only once, when the column is
        created: this method is the way to correct it (e.g. to switch the UI filter
        of a column from a search input to a select menu). The update
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
                ``display_format``, ``family``, ``description``,
                ``default_aggregation`` (strings, ``None`` clears the field) and
                ``is_categorical`` (bool).

        Raises:
            ValueError: If a field name is not one of the allowed fields, if
                ``default_aggregation`` is invalid, if the column has no row in the
                metadata table, if a non-``None`` ``parent_name`` references a
                column absent from metadata, if it would create a cycle in the
                ``parent_name`` graph, if ``is_categorical`` is not a bool, or if
                ``is_categorical=False`` targets a column of a hierarchy.

        Example:
            >>> manager.update_column_metadata(
            ...     'value', unit='€', display_format=',.2f',
            ...     default_aggregation='sum')
            >>> manager.update_column_metadata('commune', parent_name='departement')
            >>> manager.update_column_metadata('model', is_categorical=True)
        """
        # Contrôle des champs autorisés
        allowed = COLUMN_METADATA_KEYS | {"is_categorical"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(
                f"Unknown metadata field(s) {sorted(unknown)}; allowed fields are "
                f"{sorted(allowed)}"
            )
        # Contrôle du type du statut catégoriel
        if "is_categorical" in fields and not isinstance(
            fields["is_categorical"], bool
        ):
            raise ValueError(
                f"is_categorical must be a bool, got {fields['is_categorical']!r}"
            )

        # Aucun champ fourni : rien à écrire
        if not fields:
            return

        # Normalisation et validation de l'agrégation par défaut
        if "default_aggregation" in fields:
            aggregation = fields["default_aggregation"]
            if isinstance(aggregation, bool):
                raise ValueError("default_aggregation must be a string or None")
            fields["default_aggregation"] = normalize_default_aggregation(aggregation)

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
            parent_of[column] = str(new_parent)
            # Validation de la hiérarchie
            validate_hierarchy_forest(parent_of)

        # Une colonne de hiérarchie reste catégorielle : refus de is_categorical=False
        if fields.get("is_categorical") is False:
            # Colonne parente d'une autre colonne
            _crow = self.conn.execute(
                f"SELECT COUNT(*) FROM {metadata_table} WHERE parent_name = ?",
                [column],
            ).fetchone()
            is_parent = _crow is not None and _crow[0] > 0
            # Colonne enfant : le parent déclaré dans ce même appel prime sur l'état
            # stocké
            if "parent_name" in fields:
                has_parent = new_parent is not None
            else:
                _prow2 = self.conn.execute(
                    f"SELECT parent_name FROM {metadata_table} WHERE name = ?",
                    [column],
                ).fetchone()
                has_parent = _prow2 is not None and _prow2[0] is not None
            if is_parent or has_parent:
                raise ValueError(
                    f"Column {column!r} is part of a column hierarchy and must stay"
                    " categorical"
                )

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
                        f"UPDATE {metadata_table} SET is_categorical = TRUE"
                        " WHERE name = ?",
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
        # Extraction des enfants associés à la colonne
        children = self._get_hierarchy_children(column)
        # Retrait du parent
        if children:
            self.conn.execute(
                f"UPDATE {self._qualified('metadata')} SET parent_name = NULL"
                " WHERE parent_name = ?",
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

    # Méthode auxiliaire de suppression physique d'une colonne de la table des faits
    def _drop_fact_table_column(self, column: str) -> bool:
        """
        Drop a column from the fact table (``ALTER TABLE ... DROP COLUMN``).

        Only the physical column is dropped: its ``metadata`` row, ``cluster_by``
        and ``parent_name`` references are handled by
        ``_drop_column_with_references``.

        Args:
            column: Name of the column to drop.

        Returns:
            bool: True if the column was dropped, False on failure (logged).

        Example:
            >>> manager._drop_fact_table_column('old_col')
            True
        """
        try:
            # Exécution de la requête de suppression de la colonne sur la table
            self.conn.execute(
                f"ALTER TABLE {self._qualified('fact_table')}"
                f" DROP COLUMN {quote_ident(column)}"
            )
            # Logging
            self.logger.info(f"Dropped column {column} from fact table")
            return True
        except Exception as e:
            # Logging
            self.logger.error(f"Error dropping fact table column {column}: {e}")
            return False

    # Méthode de suppression d'une colonne et de toutes ses références
    def _drop_column_with_references(self, column: str, cascade: bool = False) -> bool:
        """
        Drop a fact table column together with every reference to it.

        Ordered steps: children detachment (``parent_name`` set to ``NULL``, only
        when ``cascade``), ``ALTER TABLE ... DROP COLUMN``, ``metadata`` row removal
        and ``cluster_by`` update. Shared by ``DatabaseDeleter.delete_columns`` and
        ``_cleanup_null_only_columns`` so that a dropped column never leaves a
        dangling reference behind. No transaction is opened here: the caller owns
        it.

        Args:
            column: Name of the column to drop. Must exist in the fact table.
            cascade: Whether to detach the hierarchy children of ``column`` before
                dropping it. Defaults to False.

        Returns:
            bool: True if the column was dropped, False if the ``DROP COLUMN``
            failed (nothing else is then modified, except detached children).

        Raises:
            Exception: Any error raised while removing the metadata row or updating
                ``cluster_by``, re-raised as is.

        Example:
            >>> manager._drop_column_with_references('region', cascade=True)
            True
        """
        # Détachement des colonnes enfants d'une hiérarchie (cascade uniquement)
        if cascade:
            self._clear_child_parent_references(column)

        # Suppression physique de la colonne
        if not self._drop_fact_table_column(column):
            return False

        # Suppression de la ligne de méta-données correspondante
        self.delete_column_metadata(column)

        # Retrait de la colonne de cluster_by si elle en faisait partie
        self._remove_from_cluster_by(column)
        return True

    # Méthode de nettoyage des colonnes ne contenant que des valeurs nulles
    def _cleanup_null_only_columns(self, use_transaction: bool = True) -> list[str]:
        """
        Drop the fact table columns that only hold null values.

        Each column goes through ``_drop_column_with_references`` (``metadata``
        row, ``cluster_by`` included). Some null-only columns are kept on purpose:

        - every column when the fact table is **empty**: all its columns are then
          trivially null-only, and dropping them would wipe the schema out before
          the next load;
        - primary key columns;
        - a hierarchy parent that still has children once the other null-only
          columns are dropped (a warning is logged and added to the report). A
          parent whose children are all null-only too is dropped after them.

        Runs inside ``_transaction``: nested in another operation (e.g. the cleanup
        step of ``delete_rows``), it reuses its transaction and report; otherwise it
        opens its own, so a failure leaves no column half-dropped.

        Args:
            use_transaction: Whether to run inside a DuckDB transaction when none is
                already open. Defaults to True.

        Returns:
            list[str]: Names of the dropped columns, in drop order. Empty when
            nothing was dropped.

        Raises:
            Exception: Any error raised while detecting or dropping the columns;
                the transaction (if any) is rolled back before re-raising.

        Examples:
            >>> manager.conn.execute("UPDATE fact_table SET score = NULL")
            >>> manager._cleanup_null_only_columns()
            ['score']
        """
        with self._transaction(
            "cleanup_null_only_columns", use_transaction=use_transaction
        ) as report:
            # Table vide : toutes les colonnes sont trivialement nulles, rien à
            # décider
            if self._count_rows("fact_table") == 0:
                # Logging
                self.logger.info("Fact table is empty: null-only cleanup skipped")
                return []

            # Colonnes candidates, clés primaires exclues
            primary_keys = set(self._get_primary_key_columns())
            pending = [
                c for c in self._get_null_only_columns() if c not in primary_keys
            ]

            # Suppression itérative : une colonne parente devient supprimable dès que
            # ses enfants (eux-mêmes nuls) ont été supprimés, quel que soit l'ordre
            # des colonnes dans la table.
            dropped: list[str] = []
            progress = True
            while pending and progress:
                progress = False
                for column in list(pending):
                    if self._get_hierarchy_children(column):
                        continue
                    pending.remove(column)
                    if self._drop_column_with_references(column):
                        dropped.append(column)
                        progress = True
                    else:
                        report.warnings.append(
                            f"Null-only column '{column}' could not be dropped"
                        )

            # Colonnes parentes conservées : enfants non nuls
            for column in pending:
                # Message
                warning = (
                    f"Null-only column '{column}' kept: it is the hierarchy parent of"
                    f" {self._get_hierarchy_children(column)}"
                )
                # Logging
                self.logger.warning(warning)
                # Ajout au rapport
                report.warnings.append(warning)

            if dropped:
                # Ajout au rapport
                report.columns_dropped.extend(dropped)
                self._touch_dataset_metadata()
                # Logging
                self.logger.info(f"Dropped null-only columns: {dropped}")

            return dropped

    # Méthode de lecture des colonnes enfants d'une colonne dans une hiérarchie
    def _get_hierarchy_children(self, column: str) -> list[str]:
        """
        Get the columns whose ``parent_name`` is ``column``.

        Args:
            column: Name of the potential hierarchy parent.

        Returns:
            list[str]: Names of the child columns. Empty when ``column`` is not a
            hierarchy parent.

        Example:
            >>> manager._get_hierarchy_children('region')
            ['departement']
        """
        return [
            row[0]
            for row in self.conn.execute(
                f"SELECT name FROM {self._qualified('metadata')} WHERE parent_name = ?",
                [column],
            ).fetchall()
        ]

    # Méthodes de résolution des conflits de types
    def _resolve_type_conflicts(
        self,
        column: str,
        df: nw.DataFrame[Any],
        current_metadata: nw.DataFrame[Any],
        report: OperationReport | None = None,
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
            report: When given and the type is actually widened, appended to
                ``report.metadata_changes``.

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
        # Ajout au rapport
        if report is not None:
            report.metadata_changes.append(
                f"sql_type({column}): {current_type} -> {resolved_type}"
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

    # Méthode de validation de l'état de la base de données
    def validate_database_state(
        self, validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> ValidationReport | None:
        """
        Validate the current state of the database.

        Args:
            validation_level: Level of validation to perform.

        Returns:
            ValidationReport | None: The auditor's report, or None when validation
            is disabled (no auditor).

        Example:
            >>> report = updater.validate_database_state(ValidationLevel.COMPREHENSIVE)
            >>> if report is not None and report.get_critical_issues_count() > 0:
            ...     print("Critical issues detected!")
        """
        # Vérification qu'un auditeur est renseigné
        if self.auditor is None:
            self.logger.warning("Validation disabled - no auditor available")
            return None
        return self.auditor.validate_database(validation_level)

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
