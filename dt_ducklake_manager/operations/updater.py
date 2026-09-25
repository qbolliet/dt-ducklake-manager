# Importation des modules
# Modules de base
import os
import warnings
from typing import Any, Literal

# DuckDB
import duckdb
import narwhals as nw
from narwhals.typing import IntoDataFrame

from ..maintenance.auditor import DatabaseAuditor, ValidationLevel
from ..reporting import OperationReport

# Import des utilitaires
from ..utils.sql import quote_ident, remove_dataframe_duplicates
from ..utils.types import map_python_to_sql_type, validate_column_metadata
from ..utils.value_labels import check_value_label_dependency, get_value_label_columns

# Import du gestionnaire de base
from ._base import BaseSchemaManager

# Nombre maximal de valeurs citées en exemple dans un message d'erreur
_SAMPLE_SIZE = 10


# Classe de mise à jour d'une base de données
class DatabaseUpdater(BaseSchemaManager):
    """
    Writes into an existing result set: upsert of rows, addition of value
    columns and relabeling of codes.

    Every public write operation runs as a single DuckDB transaction (``BEGIN`` /
    ``COMMIT``, ``ROLLBACK`` on exception) opened by
    :meth:`BaseSchemaManager._transaction`, and ends with a structural audit of the
    schema at ``audit_level`` inside that transaction; post-write compaction runs
    after the commit. Recovery beyond a failed operation relies on DuckLake time
    travel (``DatabaseRecoveryManager``), not on application backups.

    Errors follow one rule: an invalid input (missing or null primary key, unknown
    column, duplicated key, violated code/label dependency) raises ``ValueError``
    after rollback; an execution failure (database error) is reported by
    ``update_database``'s boolean return value and by the other methods' exception.

    Attributes:
        auditor (DatabaseAuditor): Structural auditor of the schema.
        audit_level (ValidationLevel | None): Level of the audit run inside the
            transaction of each write (None disables it).

    Examples:
        >>> updater = DatabaseUpdater(conn, schema='predictions')
        >>> updater.update_database(df, run_id='run-42')
        True
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        categorical_threshold: int | None = 50,
        log_filename: str | os.PathLike[str] | None = None,
        *,
        audit_level: ValidationLevel | None = ValidationLevel.BASIC,
        batch_size: int | None = None,
        catalog_alias: str = "db",
        schema: str = "main",
    ):
        """
        Initialize the database updater.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            categorical_threshold: Maximum number of distinct non-null values for a
                textual column to be flagged categorical when it is created (by
                ``allow_new_columns`` or ``add_columns``). Never re-evaluated.
            log_filename: Path to log file.
            audit_level: Level of the structural audit run inside the transaction
                of every write, before its commit: ``ValidationLevel.BASIC``
                (default) only reads the catalog and the small metadata tables;
                ``ValidationLevel.COMPREHENSIVE`` also scans the fact table; None
                disables it. A full audit can be run at any time with
                ``updater.auditor.validate_database(ValidationLevel.COMPREHENSIVE)``.
            batch_size: Deprecated and ignored: every write is a single SQL
                statement per step, which DuckDB streams on its own. Passing it
                emits a ``DeprecationWarning``.
            catalog_alias: Alias used in the DuckLake ATTACH statement.
                Passed to maintenance and snapshot calls. Defaults to ``'db'``.
            schema: DuckLake schema to update. A single catalog can host several
                schemas; all tables are qualified by this one, and maintenance and
                snapshot calls target it. Defaults to ``'main'``.

        Examples:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> updater = DatabaseUpdater(conn)
            >>> updater = DatabaseUpdater(conn, schema='predictions', audit_level=None)
        """
        # Initialisation du parent
        super().__init__(
            connection=connection,
            categorical_threshold=categorical_threshold,
            log_filename=log_filename,
            schema=schema,
            catalog_alias=catalog_alias,
        )

        # Paramètre de découpage en lots conservé pour compatibilité uniquement
        if batch_size is not None:
            warnings.warn(
                "batch_size is deprecated and ignored: every write is a single SQL"
                " statement per step",
                DeprecationWarning,
                stacklevel=2,
            )

        # Auditeur structurel et niveau de l'audit post-écriture
        self.auditor = DatabaseAuditor(
            connection=self.conn,
            log_filename=log_filename,
            schema=schema,
            catalog_alias=catalog_alias,
        )
        self.audit_level = audit_level

    # Méthode principale de mise à jour
    def update_database(
        self,
        update_df: IntoDataFrame,
        check_duplicates_db: bool = True,
        check_duplicates_update: bool = True,
        keep: Literal["any", "none", "first", "last"] = "none",
        use_batch_processing: bool | None = None,
        use_transaction: bool = True,
        compact_after_update: bool = True,
        allow_new_columns: bool = False,
        column_metadata: dict[str, dict[str, str]] | None = None,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> bool:
        """
        Upsert a batch into the fact table on its primary keys.

        Rows whose primary key combination is new are inserted; rows whose
        combination already exists are updated, but only when at least one of
        their values actually changes (an unchanged row is not rewritten). The
        whole batch is written sorted by ``cluster_by``, in two SQL statements
        (``UPDATE … FROM`` then ``INSERT … SELECT … WHERE NOT EXISTS … ORDER BY``),
        inside a single transaction together with the type widening of the
        metadata, the addition of new columns, the code/label dependency check,
        the primary key uniqueness check, the ``updated_at`` stamp and the
        structural audit.

        Args:
            update_df: DataFrame (any narwhals-compatible backend) carrying every
                primary key column, with no null key.
            check_duplicates_db: Whether to check, before commit, that the fact
                table is still unique on its primary keys; a duplicated key makes
                the update fail (rolled back) instead of being silently deleted.
                Defaults to True.
            check_duplicates_update: Whether to deduplicate ``update_df`` on its
                primary keys before writing, following ``keep``. When False, a
                batch with duplicated keys raises ``ValueError``. Defaults to True.
            keep: Which of the rows sharing a primary key combination to keep
                (``'first'``, ``'last'``, ``'any'``, or ``'none'`` to drop them
                all). Defaults to ``'none'``.
            use_batch_processing: Deprecated and ignored: the batch is always
                written in one statement per step. Passing it emits a
                ``DeprecationWarning``.
            use_transaction: Whether to run every step inside a single DuckDB
                transaction, so that a failure mid-update leaves the database
                exactly as it was. Defaults to True. When False the steps run in
                autocommit mode and a failure leaves partial state behind.
            compact_after_update: Whether to run DuckLake compaction (merge small
                files and rewrite delete files) right after a successful update.
                Defaults to True.
            allow_new_columns: Whether a column of ``update_df`` absent from the
                fact table may be added. Defaults to False: an unknown column then
                raises ``ValueError``. When True, each new column is added (SQL type
                via ``map_python_to_sql_type``) with its ``metadata`` row
                (``is_primary_key=False``, ``is_categorical`` inferred) inside the
                update's transaction, then its UI fields are taken from
                ``column_metadata``.
            column_metadata: Per-new-column UI fields (``label``, ``unit``,
                ``display_format``, ``family``, ``description``,
                ``default_aggregation``, ``parent_name``, ``label_for``). Ignored
                when ``update_df`` carries no new column; otherwise every key must
                be a new column. A ``parent_name`` may reference another new column
                of the same batch.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``), and on ``self.last_report``.
                Ignored (skipped with a DEBUG log) on a connection with no real
                DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info`` (e.g. ``{'model_version': '1.3'}``).

        Returns:
            bool: True if the update was committed (an empty batch is a successful
            no-op), False if it failed during execution and was rolled back. The
            :class:`~dt_ducklake_manager.reporting.OperationReport` describing what
            actually happened is exposed via ``self.last_report`` in every case.

        Raises:
            ValueError: If no primary key is defined on the fact table, if
                ``update_df`` misses a primary key column or holds a null key, if
                it carries a column absent from the fact table while
                ``allow_new_columns`` is False, if it is not unique on the primary
                keys while ``check_duplicates_update`` is False, if
                ``column_metadata`` is invalid, or if the update would violate the
                functional dependency of a code/label column pair (see
                :meth:`update_value_labels`). Raised after rollback.

        Examples:
            >>> updater.update_database(new_data_df, run_id='run-42')
            True
            >>> updater.update_database(
            ...     new_data_df, allow_new_columns=True,
            ...     column_metadata={'score': {'unit': '%'}})
            True
        """
        # Conversion vers narwhals dès le point d'entrée public
        update_df = nw.from_native(update_df, eager_only=True)

        # Paramètre de découpage en lots conservé pour compatibilité uniquement
        if use_batch_processing is not None:
            warnings.warn(
                "use_batch_processing is deprecated and ignored: the batch is always"
                " written in one statement per step",
                DeprecationWarning,
                stacklevel=2,
            )

        # Clés primaires : indispensables pour distinguer insertions et mises à jour
        primary_keys = self._get_primary_key_columns()
        self._check_batch_keys(update_df, primary_keys, "update_df")

        # Colonnes du lot absentes de la table des faits : refusées par défaut
        existing_columns = set(self._get_fact_table_columns())
        new_columns = [c for c in update_df.columns if c not in existing_columns]
        normalized_metadata: dict[str, dict[str, str | None]] = {}
        if new_columns:
            if not allow_new_columns:
                raise ValueError(
                    f"Unknown column(s) in update_df: {sorted(new_columns)}. Pass "
                    "allow_new_columns=True to add them explicitly."
                )
            normalized_metadata = validate_column_metadata(column_metadata, new_columns)

        # Déduplication du lot sur les clés primaires, puis contrôle d'unicité
        if check_duplicates_update:
            update_df = remove_dataframe_duplicates(
                update_df, keep, self.logger, "update_df", primary_keys
            )
        if len(update_df) != len(update_df.unique(subset=primary_keys, keep="any")):
            raise ValueError(
                f"update_df is not unique on the primary key(s) {primary_keys}; pass"
                " check_duplicates_update=True to deduplicate it"
            )

        # Lot vide : aucune écriture, aucun snapshot
        if len(update_df) == 0:
            self._early_report("update", run_id, "update_df is empty: nothing written")
            return True

        # Clé de tri physique, lue une fois pour toute l'opération
        cluster_by = self._get_cluster_by_columns()

        # Logging
        self.logger.debug(
            f"Starting database update with {len(update_df)} rows"
            f" (transaction: {use_transaction})"
        )

        # Bloc transactionnel unique : toutes les étapes ou aucune
        try:
            with self._transaction(
                "update",
                use_transaction=use_transaction,
                run_id=run_id,
                commit_message=commit_message,
                commit_info=commit_info,
            ) as report:
                # Ajout des colonnes nouvelles, annulé avec le reste sur échec
                if new_columns:
                    self._add_new_columns_from_update(
                        update_df, new_columns, normalized_metadata
                    )
                    report.columns_added = list(new_columns)
                self._run_update_steps(
                    update_df, primary_keys, cluster_by, check_duplicates_db, report
                )
        except ValueError:
            # Erreur de saisie : propagée après annulation
            raise
        except Exception as e:
            # Erreur d'exécution : annulée et journalisée par _transaction
            self.logger.debug(f"update_database returns False after: {e}")
            return False

        # Compaction DuckLake optionnelle après le commit : la maintenance ne fait
        # jamais partie de la transaction
        final_report = self.last_report
        assert final_report is not None  # posé par _transaction sur tout succès
        if compact_after_update:
            self._compact_after_write(final_report)
        self._finalize_report_after_write(final_report)
        self.logger.info(final_report.summary())
        return True

    # Méthode de contrôle des clés primaires d'un lot
    def _check_batch_keys(
        self, df: nw.DataFrame[Any], primary_keys: list[str], name: str
    ) -> None:
        """Check that a batch carries every primary key column, with no null key.

        Args:
            df: Batch about to be written.
            primary_keys: Primary key columns of the fact table.
            name: Name of the batch argument, used in the error messages.

        Raises:
            ValueError: If no primary key is defined, if ``df`` misses a primary
                key column, or if a primary key column of ``df`` holds a null.
        """
        # Clé primaire obligatoire
        if not primary_keys:
            raise ValueError(
                "No primary key is defined on the fact_table: rows cannot be matched"
            )
        # Colonnes de clé présentes
        missing_keys = [k for k in primary_keys if k not in df.columns]
        if missing_keys:
            raise ValueError(
                f"{name} is missing primary key column(s): {sorted(missing_keys)}"
            )
        # Valeurs de clé non nulles : une clé nulle ne correspond jamais à aucune
        # ligne existante et serait insérée à chaque écriture
        null_keys = [k for k in primary_keys if df[k].null_count() > 0]
        if null_keys:
            raise ValueError(f"{name} has null value(s) in primary key(s) {null_keys}")

    # Méthode auxiliaire d'ajout explicite des colonnes inconnues d'un update_df
    def _add_new_columns_from_update(
        self,
        update_df: nw.DataFrame[Any],
        new_columns: list[str],
        normalized_metadata: dict[str, dict[str, str | None]],
    ) -> None:
        """Add the columns of ``update_df`` absent from the fact table.

        Called inside the transaction of ``update_database`` when
        ``allow_new_columns=True``, so that a failed update leaves no column
        behind. Every column is added first (``ALTER TABLE ... ADD COLUMN``, type
        from ``map_python_to_sql_type``) together with its ``metadata`` row
        (``is_primary_key=False``, ``is_categorical`` inferred from the batch); the
        UI fields are applied afterwards, so that a ``parent_name`` or
        ``label_for`` may reference another new column of the same batch.

        Args:
            update_df: The full update DataFrame (narwhals).
            new_columns: Columns of ``update_df`` absent from the fact table.
            normalized_metadata: Per-column UI fields, already validated against
                ``new_columns``.

        Raises:
            ValueError: If a UI field is rejected by ``update_column_metadata``
                (e.g. a ``parent_name`` creating a cycle).
            duckdb.Error: If a column cannot be added.
        """
        # Ajout de toutes les colonnes et de leurs lignes de méta-données
        for column in new_columns:
            sql_type = map_python_to_sql_type(update_df.schema[column])
            self.conn.execute(
                f"ALTER TABLE {self._qualified('fact_table')} ADD COLUMN"
                f" {quote_ident(column)} {sql_type} DEFAULT NULL"
            )
            self._add_column_to_metadata(column, update_df)

        # Champs d'UI, une fois toutes les colonnes décrites
        for column in new_columns:
            fields = normalized_metadata.get(column)
            if fields:
                self.update_column_metadata(column, **fields)

        # Logging
        self.logger.debug(f"Added new column(s) from update_df: {new_columns}")

    # Méthode d'exécution ordonnée des étapes d'une mise à jour
    def _run_update_steps(
        self,
        update_df: nw.DataFrame[Any],
        primary_keys: list[str],
        cluster_by: list[str] | None,
        check_duplicates_db: bool,
        report: OperationReport,
    ) -> None:
        """Run the ordered steps of a database update.

        Steps, in order: widening of the column types, upsert of the batch,
        code/label functional dependency check restricted to the batch's codes,
        primary key uniqueness check (when ``check_duplicates_db``), ``updated_at``
        stamp, structural audit. Called from inside the transaction opened by
        ``update_database``: every failure raises, so the whole update is rolled
        back.

        Args:
            update_df: Deduplicated batch, unique on the primary keys.
            primary_keys: Primary key columns of the fact table.
            cluster_by: Physical sort key of the fact table, or None.
            check_duplicates_db: Whether to check the primary key uniqueness of the
                fact table after the upsert.
            report: In-progress report of the enclosing transaction.

        Raises:
            ValueError: If the update violates the functional dependency of a
                code/label column pair.
            RuntimeError: If the fact table is not unique on its primary keys, or
                if the structural audit finds a critical issue.
            duckdb.Error: If a statement fails.
        """
        # Étape 1 : élargissement des types des colonnes existantes
        self._update_metadata_types(update_df, report)

        # Étape 2 : écriture du lot
        self._upsert_fact_table(update_df, primary_keys, cluster_by, report)

        # Étape 3 : dépendance fonctionnelle des colonnes de libellés, sur l'état
        # post-écriture, restreinte aux codes du lot
        self._check_value_label_dependencies(
            update_df, primary_keys, set(update_df.columns)
        )

        # Étape 4 : unicité de la clé primaire de la table des faits
        if check_duplicates_db:
            self._check_primary_key_uniqueness(primary_keys)

        # Étape 5 : horodatage, dans le même snapshot que l'écriture
        self._touch_dataset_metadata()

        # Étape 6 : audit structurel avant validation
        self._post_write_audit(report)

    # Méthode auxiliaire d'élargissement des types des colonnes existantes
    def _update_metadata_types(
        self, update_df: nw.DataFrame[Any], report: OperationReport | None = None
    ) -> None:
        """Widen the recorded and physical types of the columns the batch widens.

        Args:
            update_df: DataFrame whose columns may require a wider type.
            report: When given, each widened type is appended to
                ``report.metadata_changes``.

        Raises:
            duckdb.Error: If a fact table column cannot be widened.
        """
        # Méta-données courantes (lues une fois pour toutes les colonnes)
        current_metadata = self._load_current_metadata()
        if len(current_metadata) == 0:
            return
        current_columns = set(current_metadata["name"].to_list())

        # Résolution des conflits de types, colonne par colonne
        for column in update_df.columns:
            if column in current_columns:
                self._resolve_type_conflicts(
                    column, update_df, current_metadata, report
                )

    # Méthode d'écriture d'un lot par mise à jour puis insertion
    def _upsert_fact_table(
        self,
        update_df: nw.DataFrame[Any],
        primary_keys: list[str],
        cluster_by: list[str] | None,
        report: OperationReport,
    ) -> None:
        """Write a batch into the fact table in two statements.

        1. ``UPDATE fact_table f SET … FROM batch t WHERE <keys match> AND (<at
           least one value IS DISTINCT FROM the stored one>)``: only the rows that
           actually change are rewritten (copy-on-write), which saves both the
           write and the resulting delete files.
        2. ``INSERT INTO fact_table SELECT … FROM batch t WHERE NOT EXISTS (<same
           keys in fact_table>) ORDER BY cluster_by``: the new rows, sorted as a
           whole so that the files written keep narrow ranges.

        The counts returned by the two statements are stored on the report; on a
        real DuckLake catalog, ``_transaction`` replaces them with the counts of
        ``ducklake_table_changes`` after commit.

        Args:
            update_df: Batch, unique on the primary keys.
            primary_keys: Primary key columns of the fact table.
            cluster_by: Physical sort key of the fact table, or None.
            report: In-progress report of the enclosing transaction.

        Raises:
            duckdb.Error: If a statement fails.
        """
        # Éléments de requête
        fact_table = self._qualified("fact_table")
        view_name = "_update_database_batch"
        columns = list(update_df.columns)
        value_columns = [c for c in columns if c not in primary_keys]
        join_condition = " AND ".join(
            f"f.{quote_ident(k)} = t.{quote_ident(k)}" for k in primary_keys
        )

        # Enregistrement du lot comme vue temporaire
        self.conn.register(view_name, nw.to_native(update_df))
        try:
            # Mise à jour des seules lignes existantes dont une valeur change.
            # Noms non qualifiés côté gauche du SET : DuckDB rejette les
            # qualificateurs de table dans la clause SET d'un UPDATE ... FROM.
            rows_updated = 0
            if value_columns:
                set_clause = ", ".join(
                    f"{quote_ident(c)} = t.{quote_ident(c)}" for c in value_columns
                )
                changed = " OR ".join(
                    f"f.{quote_ident(c)} IS DISTINCT FROM t.{quote_ident(c)}"
                    for c in value_columns
                )
                row = self.conn.execute(f"""
                    UPDATE {fact_table} f SET {set_clause}
                    FROM {view_name} t
                    WHERE {join_condition} AND ({changed})
                """).fetchone()
                rows_updated = int(row[0]) if row is not None else 0

            # Insertion des combinaisons de clés absentes, triées selon cluster_by
            column_list = ", ".join(quote_ident(c) for c in columns)
            order_clause = self._cluster_by_order_clause(columns, cluster_by)
            row = self.conn.execute(f"""
                INSERT INTO {fact_table} ({column_list})
                SELECT {column_list} FROM {view_name} t
                WHERE NOT EXISTS (
                    SELECT 1 FROM {fact_table} f WHERE {join_condition}
                )
                {order_clause}
            """).fetchone()
            rows_inserted = int(row[0]) if row is not None else 0
        finally:
            # Suppression de la vue, succès comme échec
            self.conn.unregister(view_name)

        # Comptages de repli, remplacés par la mesure DuckLake après commit
        report.rows_inserted = rows_inserted
        report.rows_updated = rows_updated

        # Logging
        self.logger.debug(
            f"Upsert of {len(update_df)} row(s): {rows_inserted} inserted,"
            f" {rows_updated} updated,"
            f" {len(update_df) - rows_inserted - rows_updated} unchanged"
        )

    # Méthode de contrôle de l'unicité de la clé primaire de la table des faits
    def _check_primary_key_uniqueness(self, primary_keys: list[str]) -> None:
        """Check that the fact table is unique on its primary keys.

        The upsert cannot create a duplicated key (the batch is unique and new
        rows are inserted only when their key is absent), so a duplicate found here
        was introduced outside of the package. The check raises rather than
        deleting rows: which of the duplicates is right cannot be decided here.

        Args:
            primary_keys: Primary key columns of the fact table.

        Raises:
            RuntimeError: If at least one key combination appears several times,
                with a sample of the duplicated combinations.
        """
        # Colonnes de céls primaires
        key_columns = ", ".join(quote_ident(k) for k in primary_keys)
        # Duplicats de clés primaires dans la table des faits
        duplicates = self.conn.execute(f"""
            SELECT {key_columns}, COUNT(*) AS occurrences
            FROM {self._qualified("fact_table")}
            GROUP BY {key_columns}
            HAVING COUNT(*) > 1
            LIMIT {_SAMPLE_SIZE}
        """).fetchall()
        if duplicates:
            raise RuntimeError(
                f"fact_table is not unique on the primary key(s) {primary_keys};"
                f" duplicated combinations (sample, with occurrences): {duplicates}."
                " Delete the duplicates or restore a snapshot before updating"
            )

    # Méthode de contrôle de la dépendance fonctionnelle des colonnes de libellés
    # après l'écriture d'un lot (update_database ou add_columns)
    def _check_value_label_dependencies(
        self,
        df: nw.DataFrame[Any],
        primary_keys: list[str],
        touched_columns: set[str],
    ) -> None:
        """
        Check label -> code association for every declared pair touched by a batch.

        Restricted to the codes of the rows the batch touched, not a full-table
        scan: for each declared code column whose code column or at least one of
        its label columns is in ``touched_columns``, builds the *current*
        (post-write) distinct code values of the fact table rows matching ``df``'s
        primary keys, then checks the dependency against that restriction. This one
        restriction view covers every scenario of a refused update: a new label for
        an existing code, a batch without the label column inserting a code already
        labeled elsewhere, and a non-primary-key code changed without its label.

        Args:
            df: The batch just written (``update_df`` or ``add_columns``' ``df``),
                carrying every primary key column.
            primary_keys: Primary key columns of the fact table, used to join ``df``
                back to it.
            touched_columns: Columns of ``df`` that were actually written.

        Raises:
            ValueError: If the functional dependency is violated for any relevant
                pair, naming the faulty codes and pointing to
                ``DatabaseUpdater.update_value_labels`` for a deliberate relabeling.
        """
        # Paires code -> colonnes de libellés concernées par ce lot
        pairs = get_value_label_columns(
            self.conn, schema=self.schema, catalog_alias=self.catalog_alias
        )
        relevant = {
            code_col: label_cols
            for code_col, label_cols in pairs.items()
            if code_col in touched_columns
            or any(label_col in touched_columns for label_col in label_cols)
        }
        if not relevant:
            return

        # Vues de restriction aux lignes et aux codes du lot
        fact_table = self._qualified("fact_table")
        batch_view = "_value_label_check_batch"
        codes_view = "_value_label_check_codes"
        self.conn.register(batch_view, nw.to_native(df.select(primary_keys)))
        try:
            # Condition de jointure sur les clés primaires
            join_condition = " AND ".join(
                f"f.{quote_ident(k)} = t.{quote_ident(k)}" for k in primary_keys
            )
            # Parcours des colonnes de code et de leurs colonnes de libellés
            for code_col, label_cols in relevant.items():
                # Codes courants des lignes touchées par le lot
                quoted_code = quote_ident(code_col)
                self.conn.execute(f"""
                    CREATE OR REPLACE TEMP VIEW {codes_view} AS
                    SELECT DISTINCT f.{quoted_code} AS {quoted_code}
                    FROM {fact_table} f
                    JOIN {batch_view} t ON {join_condition}
                """)
                try:
                    # Vérification de l'association code -> libellé
                    for label_col in label_cols:
                        check_value_label_dependency(
                            self.conn,
                            fact_table,
                            code_col,
                            label_col,
                            restrict_to=codes_view,
                        )
                finally:
                    self.conn.execute(f"DROP VIEW IF EXISTS {codes_view}")
        finally:
            # Suppression de la vue du lot
            self.conn.unregister(batch_view)

    # ---------------------------------------------------------------------------
    # Gestion explicite des colonnes : ajout de colonnes de valeurs
    # ---------------------------------------------------------------------------

    # Méthode d'ajout explicite de colonnes de valeurs à partir d'un DataFrame
    def add_columns(
        self,
        df: IntoDataFrame,
        column_metadata: dict[str, dict[str, str]] | None = None,
        overwrite: bool = False,
        compact_after_update: bool = True,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> OperationReport:
        """
        Add value column(s) to the fact table from a DataFrame keyed by the
        primary keys, merging rows on those keys (outer merge).

        ``df`` must carry every primary key column plus one or more other columns,
        the values to add. Rows are matched on the primary keys:

        - key combination present in both: the row receives ``df``'s values;
        - key combination only in ``df``: a new row is **inserted** with the primary
          keys and ``df``'s columns, every other value column of the fact table
          left NULL;
        - key combination only in the fact table: the row keeps NULL in the new
          column(s) (and its previous values in overwritten ones).

        Implemented as ``ALTER TABLE ... ADD COLUMN`` for every genuinely new
        column, followed by a **single** ``UPDATE fact_table ... FROM <df> WHERE
        <primary keys match>`` covering all of them (restricted to the rows whose
        values actually change), then a single ``INSERT ... SELECT`` of the
        unmatched combinations sorted by ``cluster_by``, inside one DuckDB
        transaction: on any failure, neither the column(s), their ``metadata``
        row(s) nor the inserted rows survive. An ``UPDATE`` that touches every row
        of the fact table is a complete copy-on-write rewrite; ``rewrite_data_files``
        is called afterwards with a low ``delete_threshold`` to clear the resulting
        delete files.

        Args:
            df: DataFrame carrying every primary key column plus the value
                column(s) to add. Must be unique on the primary keys, with no null
                primary key.
            column_metadata: Per-added-column UI fields (``label``, ``unit``,
                ``display_format``, ``family``, ``description``,
                ``default_aggregation``, ``parent_name``, ``label_for``). Applies to
                newly added columns as well as to overwritten existing ones. A new
                column can be declared a label column this way, e.g.
                ``{'nc8_libelle': {'label_for': 'nc8'}}``.
            overwrite: Whether a column of ``df`` that already exists in the fact
                table may have its values replaced. Defaults to False: an existing
                column then raises ``ValueError`` instead.
            compact_after_update: Whether to run DuckLake compaction
                (``rewrite_data_files`` with a low ``delete_threshold``) after a
                successful commit. Defaults to True.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``). Ignored (skipped with a DEBUG
                log) on a connection with no real DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``.

        Returns:
            OperationReport: report describing what was actually added, updated
            (``rows_updated``, the rows whose values changed) and inserted
            (``rows_inserted``).

        Raises:
            ValueError: If no primary key is defined on the fact table, if ``df``
                is missing a primary key column, if a primary key of ``df`` holds a
                null, if ``df`` is not unique on the primary keys, if ``df``
                carries no value column, if a value column already exists and
                ``overwrite`` is False, if ``column_metadata`` is malformed, or if
                writing a code or label column violates the functional dependency
                of a declared code/label pair.
            RuntimeError: If the structural audit finds a critical issue.
            duckdb.Error: If a statement fails; the transaction is rolled back.

        Examples:
            >>> updater.add_columns(df_with_score)
            >>> updater.add_columns(df_with_score, overwrite=True)
            >>> # Diffusion explicite d'une valeur portée par une clé partielle :
            >>> keys = updater.get_key_combinations(['region', 'produit'])
            >>> df_partial = keys.to_polars().join(df_score, on=['region', 'produit'])
            >>> updater.add_columns(df_partial)
        """
        # Conversion vers narwhals dès le point d'entrée public
        df_nw = nw.from_native(df, eager_only=True)

        # Clés primaires présentes et non nulles
        primary_keys = self._get_primary_key_columns()
        self._check_batch_keys(df_nw, primary_keys, "df")

        # df doit être unique sur les clés primaires : sinon la valeur affectée à
        # une même ligne de la fact table serait indéterminée (dernière ligne du
        # lot gagnante, silencieusement)
        if len(df_nw) != len(df_nw.unique(subset=primary_keys, keep="any")):
            raise ValueError(f"df must be unique on primary key(s) {primary_keys}")

        # Colonnes à ajouter : tout ce qui n'est pas une clé primaire
        new_columns = [c for c in df_nw.columns if c not in primary_keys]
        if not new_columns:
            raise ValueError(
                "df carries no value column to add besides the primary keys"
            )

        # Colonnes déjà existantes dans la fact table : erreur sauf overwrite=True
        existing_columns = set(self._get_fact_table_columns())
        already_existing = [c for c in new_columns if c in existing_columns]
        if already_existing and not overwrite:
            raise ValueError(
                f"Column(s) already exist in fact_table: {sorted(already_existing)}."
                " Pass overwrite=True to update their values instead."
            )
        columns_to_add = [c for c in new_columns if c not in already_existing]

        # Validation des métadonnées d'UI, restreintes aux colonnes concernées
        normalized_metadata = validate_column_metadata(column_metadata, new_columns)

        # Clé de tri physique, lue une fois pour toute l'opération
        cluster_by = self._get_cluster_by_columns()

        fact_table = self._qualified("fact_table")
        view_name = "_add_columns_src"

        # Transaction DuckDB unique : sur échec, ni les colonnes ni les lignes
        # metadata ne subsistent (ALTER TABLE est transactionnel dans DuckDB).
        with self._transaction(
            "add_columns",
            run_id=run_id,
            commit_message=commit_message,
            commit_info=commit_info,
        ) as report:
            report.columns_added = list(columns_to_add)
            # ALTER TABLE ... ADD COLUMN pour chaque colonne réellement nouvelle
            for column in columns_to_add:
                sql_type = map_python_to_sql_type(df_nw.schema[column])
                self.conn.execute(
                    f"ALTER TABLE {fact_table} ADD COLUMN {quote_ident(column)}"
                    f" {sql_type} DEFAULT NULL"
                )
                # Ligne de méta-données (is_primary_key=FALSE, is_categorical
                # inféré du DataFrame)
                self._add_column_to_metadata(column, df_nw)

            # Champs d'UI (nouvelles colonnes et colonnes overwrite confondues),
            # une fois toutes les colonnes décrites
            for column in new_columns:
                fields = normalized_metadata.get(column)
                if fields:
                    self.update_column_metadata(column, **fields)

            # Condition de jointure sur les clés primaires (identifiants qualifiés
            # et quotés des deux côtés)
            join_condition = " AND ".join(
                f"f.{quote_ident(k)} = t.{quote_ident(k)}" for k in primary_keys
            )

            self.conn.register(view_name, nw.to_native(df_nw))
            try:
                # Un seul UPDATE ... FROM pour toutes les colonnes concernées,
                # restreint aux lignes dont une valeur change (noms non qualifiés
                # côté gauche du SET : DuckDB rejette les qualificateurs de table
                # dans la clause SET d'un UPDATE ... FROM)
                set_clause = ", ".join(
                    f"{quote_ident(c)} = t.{quote_ident(c)}" for c in new_columns
                )
                changed = " OR ".join(
                    f"f.{quote_ident(c)} IS DISTINCT FROM t.{quote_ident(c)}"
                    for c in new_columns
                )
                row = self.conn.execute(f"""
                    UPDATE {fact_table} f
                    SET {set_clause}
                    FROM {view_name} t
                    WHERE {join_condition} AND ({changed})
                """).fetchone()
                rows_updated = int(row[0]) if row is not None else 0

                # Insertion des combinaisons absentes de la base (fusion externe) :
                # les autres colonnes de valeur prennent NULL. Tri par cluster_by
                # pour préserver l'élagage par fichier.
                insert_columns = ", ".join(
                    quote_ident(c) for c in [*primary_keys, *new_columns]
                )
                order_clause = self._cluster_by_order_clause(
                    list(df_nw.columns), cluster_by
                )
                row = self.conn.execute(f"""
                    INSERT INTO {fact_table} ({insert_columns})
                    SELECT {insert_columns} FROM {view_name} t
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {fact_table} f WHERE {join_condition}
                    )
                    {order_clause}
                """).fetchone()
                rows_inserted = int(row[0]) if row is not None else 0
            finally:
                # Suppression de la vue, succès comme échec
                self.conn.unregister(view_name)

            # Comptages de repli, remplacés par la mesure DuckLake après commit
            report.rows_updated = rows_updated
            report.rows_inserted = rows_inserted

            # Dépendance fonctionnelle des colonnes de libellés, sur l'état
            # post-écriture, restreinte aux codes du lot. Couvre aussi le cas d'une
            # nouvelle colonne de libellés déclarée via column_metadata : sa
            # validation structurelle a déjà eu lieu dans update_column_metadata.
            self._check_value_label_dependencies(df_nw, primary_keys, set(new_columns))

            # Horodatage et audit structurel, dans le même snapshot que l'écriture
            self._touch_dataset_metadata()
            self._post_write_audit(report)

        # Lignes préexistantes de la base restées NULL : celles qu'aucune ligne de
        # df n'est venue apparier (df étant unique sur la clé, les lignes appariées
        # sont celles qui n'ont pas été insérées)
        rows_matched = len(df_nw) - rows_inserted
        rows_left_null = max(report.rows_before - rows_matched, 0)
        self.logger.debug(
            f"add_columns: {rows_matched} row(s) matched ({rows_updated} changed),"
            f" {rows_inserted} inserted, {rows_left_null} fact_table row(s) left"
            " NULL (no match in df)"
        )

        # Compaction DuckLake optionnelle : un UPDATE massif laisse des fichiers de
        # suppression sur les anciennes versions des lignes touchées ;
        # delete_threshold bas car le taux de suppression peut approcher 100%.
        final_report = self.last_report
        assert final_report is not None  # posé par _transaction sur tout succès
        if compact_after_update:
            self._compact_after_write(final_report, delete_threshold=0.05)

        self._finalize_report_after_write(final_report)

        # Logging
        self.logger.info(final_report.summary())
        return final_report

    # ---------------------------------------------------------------------------
    # Gestion explicite des colonnes de libellés
    # ---------------------------------------------------------------------------

    # Méthode de remplacement explicite des libellés de certains codes
    def update_value_labels(
        self,
        label_column: str,
        labels: IntoDataFrame,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
        compact_after_update: bool = True,
    ) -> OperationReport:
        """
        Replace the label of one or more codes of a code/label column pair.

        The one legitimate way to relabel a code: an upsert cannot do it, since the
        rows of that code already written under the old label would violate the
        functional dependency (``update_database``'s value-label check refuses
        exactly this and points here). Runs a single ``UPDATE fact_table SET
        <label_column> = t.<label_column> FROM <labels> t WHERE fact_table.<code> =
        t.<code>``, restricted to the rows whose label actually changes and
        rewriting them (copy-on-write), inside a single transaction: a change of
        *some* labels either all lands or none does. The displayed label is thus
        always the **current** one; earlier labels remain readable through DuckLake
        time travel.

        Args:
            label_column: Name of the label column to update. Must already have a
                ``label_for`` declared (via the build, ``add_columns`` or
                ``update_column_metadata``).
            labels: Narwhals-compatible DataFrame carrying exactly two columns: the
                code (under the code column's own name) and the new label (under
                ``label_column``'s name). Must be unique on the code column, with
                no null code.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``). Ignored (skipped with a DEBUG
                log) on a connection with no real DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``.
            compact_after_update: Whether to run DuckLake compaction
                (``rewrite_data_files``) after a successful commit. Defaults to True.

        Returns:
            OperationReport: report describing the rows rewritten
            (``rows_updated``, only those whose label changed); codes of ``labels``
            absent from the fact table are counted and sampled in
            ``report.warnings``, never inserted.

        Raises:
            ValueError: If ``label_column`` has no ``label_for`` declared, if
                ``labels`` does not carry exactly the code and label columns, if
                ``labels`` holds a null code or is not unique on the code column,
                or if the update would (still) violate the functional dependency
                code -> label.
            RuntimeError: If the structural audit finds a critical issue.
            duckdb.Error: If a statement fails; the transaction is rolled back.

        Examples:
            >>> import polars as pl
            >>> new_labels = pl.DataFrame({
            ...     'nc8': ['01012100'], 'nc8_libelle': ['Chevaux reproducteurs']})
            >>> report = updater.update_value_labels('nc8_libelle', new_labels)
        """
        # Conversion vers narwhals dès le point d'entrée public
        labels_nw = nw.from_native(labels, eager_only=True)

        # Résolution de la colonne de code associée à la colonne de libellés
        metadata_table = self._qualified("metadata")
        _row = self.conn.execute(
            f"SELECT label_for FROM {metadata_table} WHERE name = ?", [label_column]
        ).fetchone()
        code_column = _row[0] if _row is not None else None
        if code_column is None:
            raise ValueError(
                f"Column {label_column!r} has no label_for; it is not a value-label"
                " column"
            )

        # labels doit porter exactement le code et le libellé
        expected_columns = {code_column, label_column}
        if set(labels_nw.columns) != expected_columns:
            raise ValueError(
                f"labels must carry exactly the columns {sorted(expected_columns)},"
                f" got {sorted(labels_nw.columns)}"
            )

        # Code non nul : une ligne de code nul a toujours un libellé nul
        if labels_nw[code_column].null_count() > 0:
            raise ValueError(
                f"labels holds null value(s) in {code_column!r}: a null code always"
                " has a null label"
            )

        # Unicité sur le code : sinon le libellé affecté à un même code serait
        # indéterminé (dernière ligne du lot gagnante, silencieusement)
        if len(labels_nw) != len(labels_nw.unique(subset=[code_column], keep="any")):
            raise ValueError(f"labels must be unique on {code_column!r}")

        fact_table = self._qualified("fact_table")
        view_name = "_update_value_labels_src"
        quoted_code = quote_ident(code_column)
        quoted_label = quote_ident(label_column)

        # Transaction DuckDB unique : sur échec, aucun libellé n'est modifié.
        with self._transaction(
            "update_value_labels",
            run_id=run_id,
            commit_message=commit_message,
            commit_info=commit_info,
        ) as report:
            # Enregistrement d'une vue temporaire pour la jointure
            self.conn.register(view_name, nw.to_native(labels_nw))
            try:
                # Codes absents de la base : jamais insérés, seulement signalés.
                # NOT EXISTS plutôt que NOT IN : un code NULL dans la table des
                # faits rendrait NOT IN toujours faux.
                absent_rows = self.conn.execute(f"""
                    SELECT t.{quoted_code}, COUNT(*) OVER ()
                    FROM {view_name} t
                    WHERE NOT EXISTS (
                        SELECT 1 FROM {fact_table} f
                        WHERE f.{quoted_code} = t.{quoted_code}
                    )
                    ORDER BY t.{quoted_code}
                    LIMIT {_SAMPLE_SIZE}
                """).fetchall()
                if absent_rows:
                    # Logging
                    warning = (
                        f"{absent_rows[0][1]} code(s) of labels absent from the"
                        f" fact table, never inserted (sample: "
                        f"{[r[0] for r in absent_rows]})"
                    )
                    self.logger.warning(warning)
                    report.warnings.append(warning)

                # UPDATE unique portant sur tous les codes du lot, restreint aux
                # lignes dont le libellé change
                row = self.conn.execute(f"""
                    UPDATE {fact_table} f SET {quoted_label} = t.{quoted_label}
                    FROM {view_name} t
                    WHERE f.{quoted_code} = t.{quoted_code}
                      AND f.{quoted_label} IS DISTINCT FROM t.{quoted_label}
                """).fetchone()
                # Comptage de repli, remplacé par la mesure DuckLake après commit
                report.rows_updated = int(row[0]) if row is not None else 0

                # Contrôle de dépendance après l'UPDATE (garde-fou : l'unicité sur
                # le code et la restriction aux codes du lot devraient déjà le
                # garantir)
                check_value_label_dependency(
                    self.conn,
                    fact_table,
                    code_column,
                    label_column,
                    restrict_to=view_name,
                )
            finally:
                # Suppression de la vue, succès comme échec
                self.conn.unregister(view_name)

            # Horodatage et audit structurel, dans le même snapshot que l'écriture
            self._touch_dataset_metadata()
            self._post_write_audit(report)

        # Compaction DuckLake optionnelle : l'UPDATE réécrit (copy-on-write) les
        # lignes des codes dont le libellé change.
        final_report = self.last_report
        assert final_report is not None  # posé par _transaction sur tout succès
        if compact_after_update:
            self._compact_after_write(final_report)

        self._finalize_report_after_write(final_report)
        self.logger.info(final_report.summary())
        return final_report

    # Méthode d'extraction des combinaisons de clés existantes en base
    def get_key_combinations(
        self, columns: list[str] | None = None
    ) -> nw.DataFrame[Any]:
        """
        Get distinct existing combinations of key column(s) from the fact table.

        A value carried by a partial key (e.g. ``(region, produit)``) is not
        automatically spread over a fuller key (e.g. ``(date, region, produit)``)
        by ``add_columns`` — that would be a different result set, denormalized. To
        broadcast deliberately, join the caller's partial DataFrame against the full
        key combinations returned here, then pass the joined DataFrame to
        ``add_columns``.

        Args:
            columns: Columns to project. Defaults to every primary key column.

        Returns:
            nw.DataFrame: Distinct combinations of ``columns`` present in the fact
            table, one row per combination (pyarrow backend: convert it with
            ``to_polars()``/``to_pandas()`` before joining a native DataFrame).

        Raises:
            ValueError: If no primary key is defined on the fact table and
                ``columns`` is None, or if ``columns`` references a column absent
                from the fact table.

        Examples:
            >>> keys = updater.get_key_combinations(['region', 'produit'])
            >>> df_partial = keys.to_polars().join(df_score, on=['region', 'produit'])
            >>> updater.add_columns(df_partial)
        """
        # Colonnes par défaut : toutes les clés primaires
        if columns is None:
            columns = self._get_primary_key_columns()
            if not columns:
                raise ValueError(
                    "No primary key is defined on the fact_table; pass columns"
                    " explicitly."
                )

        # Validation de l'existence des colonnes demandées
        existing_columns = set(self._get_fact_table_columns())
        unknown = [c for c in columns if c not in existing_columns]
        if unknown:
            raise ValueError(f"Unknown column(s) in fact_table: {sorted(unknown)}")

        # Création de la requête
        column_list = ", ".join(quote_ident(c) for c in columns)
        result = self.conn.execute(
            f"SELECT DISTINCT {column_list} FROM {self._qualified('fact_table')}"
        ).to_arrow_table()
        combinations: nw.DataFrame[Any] = nw.from_native(result, eager_only=True)
        return combinations
