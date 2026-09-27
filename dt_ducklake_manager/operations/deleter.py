# Importation des modules
# Modules de base
import os
from typing import Any

# DuckDB
import duckdb

from ..maintenance.auditor import DatabaseAuditor, ValidationLevel
from ..reporting import OperationReport

# Import des utilitaires
from ..utils.sql import _build_where_clause

# Import du gestionnaire de base
from ._base import BaseSchemaManager

# Filtres acceptés par delete_rows : condition SQL, conjonction de tuples
# (colonne, opérateur, valeur), ou disjonction de conjonctions
RowFilters = str | list[tuple[str, str, Any]] | list[list[tuple[str, str, Any]]]


# Classe de suppression de données dans la base
class DatabaseDeleter(BaseSchemaManager):
    """
    Deletes rows or columns of a result set.

    Every public deletion runs as a single DuckDB transaction (``BEGIN`` /
    ``COMMIT``, ``ROLLBACK`` on exception) opened by
    :meth:`BaseSchemaManager._transaction`, and ends with a structural audit of the
    schema at ``audit_level`` inside that transaction; post-write compaction runs
    after the commit. Recovery beyond a failed operation relies on DuckLake time
    travel (``DatabaseRecoveryManager``), not on application backups.

    An invalid request (missing or malformed filters, unknown column, column that
    cannot be deleted) raises ``ValueError`` before anything is written; an
    execution failure is re-raised after rollback.

    Attributes:
        auditor (DatabaseAuditor): Structural auditor of the schema.
        audit_level (ValidationLevel | None): Level of the audit run inside the
            transaction of each deletion (None disables it).
        auto_cleanup (bool): Whether ``delete_rows`` drops, by default, the columns
            its deletion leaves null-only.

    Examples:
        >>> deleter = DatabaseDeleter(conn, schema='predictions')
        >>> deleter.delete_rows([('date', '<', '2023-01-01')]).rows_deleted
        1200
    """

    # Initialisation
    def __init__(
        self,
        connection: duckdb.DuckDBPyConnection | None = None,
        log_filename: str | os.PathLike[str] | None = None,
        *,
        audit_level: ValidationLevel | None = ValidationLevel.BASIC,
        auto_cleanup: bool = True,
        catalog_alias: str = "db",
        schema: str = "main",
    ):
        """
        Initialize the database deleter.

        Args:
            connection: DuckDB connection attached to a DuckLake catalog, obtained
                via ``DuckLakeConnector.connect()``. If None, an in-memory connection
                is created (for unit tests only).
            log_filename: Path to log file.
            audit_level: Level of the structural audit run inside the transaction
                of every deletion, before its commit: ``ValidationLevel.BASIC``
                (default) only reads the catalog and the small metadata tables;
                ``ValidationLevel.COMPREHENSIVE`` also scans the fact table; None
                disables it.
            auto_cleanup: Whether ``delete_rows`` drops, by default, the columns
                its deletion leaves null-only. Defaults to True.
            catalog_alias: Alias used in the DuckLake ATTACH statement.
                Defaults to ``'db'``.
            schema: DuckLake schema to delete from. A single catalog can host several
                schemas; all tables are qualified by this one, and compaction calls
                target it. Defaults to ``'main'``.

        Example:
            >>> conn = DuckLakeConnector('catalog.ducklake', 'data/').connect()
            >>> deleter = DatabaseDeleter(conn, auto_cleanup=False)
            >>> deleter = DatabaseDeleter(conn, schema='predictions')
        """
        # Initialisation du parent
        super().__init__(
            connection=connection,
            log_filename=log_filename,
            schema=schema,
            catalog_alias=catalog_alias,
        )

        # Auditeur structurel et niveau de l'audit post-écriture
        self.auditor = DatabaseAuditor(
            connection=self.conn,
            log_filename=log_filename,
            schema=schema,
            catalog_alias=catalog_alias,
        )
        self.audit_level = audit_level

        # Nettoyage des colonnes devenues entièrement nulles
        self.auto_cleanup = auto_cleanup

    # Méthode principale de suppression de lignes
    def delete_rows(
        self,
        filters: RowFilters,
        use_transaction: bool = True,
        perform_cleanup: bool | None = None,
        compact_after_update: bool = True,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> OperationReport:
        """
        Delete the fact table rows matching ``filters``, in a single ``DELETE``.

        Args:
            filters: Rows to delete, as a SQL condition (``"status = 'inactive'"``),
                a list of ``(column, operator, value)`` tuples combined with
                ``AND``, or a list of such lists combined with ``OR``. Operators are
                ``=``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``in`` and ``not in``.
                Required: deleting every row takes an explicit condition such as
                ``"TRUE"``.
            use_transaction: Whether to run every step inside a single DuckDB
                transaction, so that a failure mid-deletion leaves the fact table
                exactly as it was. Defaults to True. When False the steps run in
                autocommit mode and a failure leaves partial state behind.
            perform_cleanup: Whether to drop the columns left null-only by the
                deletion (None = use the ``auto_cleanup`` setting). Runs after the
                deletion is committed, in its own transaction (DuckDB refuses to
                commit a transaction mixing a ``DELETE`` and a ``DROP COLUMN`` on the
                same table): a cleanup failure is reported in ``report.warnings``
                without restoring the deleted rows.
            compact_after_update: Whether to run DuckLake compaction (merge small
                files and rewrite delete files) right after a successful deletion.
                Defaults to True.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``). Ignored (skipped with a DEBUG
                log) on a connection with no real DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``.

        Returns:
            OperationReport: report describing what was actually deleted
            (``rows_deleted``, ``columns_dropped`` by the cleanup).

        Raises:
            ValueError: If ``filters`` is None, empty, or of an unsupported type
                (e.g. a dict).
            RuntimeError: If the structural audit finds a critical issue.
            duckdb.Error: If the ``DELETE`` fails (e.g. unknown column in the
                condition); the transaction is rolled back.

        Example:
            >>> report = deleter.delete_rows("status = 'inactive'")
            >>> filters = [('status', '=', 'inactive'), ('date', '<', '2023-01-01')]
            >>> report = deleter.delete_rows(filters, compact_after_update=False)
        """
        # Validation et traduction des filtres avant toute écriture
        where_clause = self._build_row_filter(filters)

        # Configuration du nettoyage
        if perform_cleanup is None:
            perform_cleanup = self.auto_cleanup

        # Logging
        self.logger.debug(
            f"Starting row deletion (transaction: {use_transaction}, cleanup:"
            f" {perform_cleanup})"
        )

        # Bloc transactionnel unique : la suppression, son horodatage et l'audit
        # forment un tout, annulé en bloc sur exception
        with self._transaction(
            "delete_rows",
            use_transaction=use_transaction,
            run_id=run_id,
            commit_message=commit_message,
            commit_info=commit_info,
        ) as report:
            # Suppression des lignes ; le nombre renvoyé par le DELETE sert de
            # comptage de repli, remplacé par la mesure DuckLake après commit
            row = self.conn.execute(
                f"DELETE FROM {self._qualified('fact_table')} {where_clause}"
            ).fetchone()
            rows_deleted = int(row[0]) if row is not None else 0
            report.rows_deleted = rows_deleted

            # Horodatage et audit, uniquement si la table a changé
            if rows_deleted > 0:
                self._touch_dataset_metadata()
                self._post_write_audit(report)

        final_report = self.last_report
        assert final_report is not None  # posé par _transaction sur tout succès

        # Suppression des colonnes devenues entièrement nulles, après le commit
        if perform_cleanup and rows_deleted > 0:
            self._run_post_delete_cleanup(final_report, use_transaction)

        # Compaction DuckLake optionnelle après le commit (réécriture des fichiers
        # de suppression) : la maintenance ne fait jamais partie de la transaction
        if rows_deleted > 0 and compact_after_update:
            self._compact_after_write(final_report)

        self._finalize_report_after_write(final_report)

        # Logging
        self.logger.info(final_report.summary())
        return final_report

    # Méthode de validation et de traduction des filtres de lignes
    @staticmethod
    def _build_row_filter(filters: RowFilters) -> str:
        """Validate ``delete_rows`` filters and translate them into a ``WHERE`` clause.

        Args:
            filters: SQL condition, list of ``(column, operator, value)`` tuples, or
                list of such lists.

        Returns:
            str: The ``WHERE …`` clause.

        Raises:
            ValueError: If ``filters`` is None, an empty string or list, or of an
                unsupported type (e.g. a dict), or if a list mixes tuples and
                lists.

        Examples:
            >>> DatabaseDeleter._build_row_filter("id = 1")
            'WHERE id = 1'
            >>> DatabaseDeleter._build_row_filter({"id": 1})
            Traceback (most recent call last):
                ...
            ValueError: filters must be a SQL condition string or a list of (column, operator, value) tuples, got dict
        """  # noqa: E501
        # Filtre absent : une suppression totale doit être demandée explicitement
        if filters is None:
            raise ValueError(
                "filters is required; pass an explicit condition such as 'TRUE' to"
                " delete every row"
            )
        # Type non pris en charge
        if not isinstance(filters, str | list):
            raise ValueError(
                "filters must be a SQL condition string or a list of (column,"
                f" operator, value) tuples, got {type(filters).__name__}"
            )
        # Filtre vide : produirait une clause WHERE invalide
        if (isinstance(filters, str) and not filters.strip()) or (
            isinstance(filters, list) and not filters
        ):
            raise ValueError("filters must not be empty")
        # Traduction ; une liste mal formée lève une TypeError, rendue en ValueError
        try:
            return _build_where_clause(filters)
        except TypeError as e:
            raise ValueError(str(e)) from e

    # Méthode de nettoyage des colonnes nulles après une suppression de lignes
    def _run_post_delete_cleanup(
        self, report: OperationReport, use_transaction: bool
    ) -> None:
        """Drop the columns left null-only by a committed row deletion.

        Runs ``_cleanup_null_only_columns`` in its own transaction and merges its
        outcome into the ``delete_rows`` report: dropped columns go to
        ``report.columns_dropped``, warnings to ``report.warnings``. A failure is
        logged and reported as a warning; the row deletion stays committed.
        ``self.last_report`` is set back to ``report`` in every case.

        Args:
            report: Report of the committed ``delete_rows`` operation, updated in
                place.
            use_transaction: Whether the cleanup runs inside a DuckDB transaction.
        """
        try:
            # Ajout au rapport et suppression des colonnes de null
            report.columns_dropped.extend(
                self._cleanup_null_only_columns(use_transaction=use_transaction)
            )
            cleanup_report = self.last_report
            if cleanup_report is not None and cleanup_report is not report:
                report.warnings.extend(cleanup_report.warnings)
        except Exception as e:
            # Logging
            self.logger.warning(f"Cleanup failed, but row deletion completed: {e}")
            # Ajout au rapport
            report.warnings.append(f"Null-only column cleanup failed: {e}")
        finally:
            # Mise à jour du dernier rapport
            self.last_report = report

    # Méthode principale de suppression de colonnes
    def delete_columns(
        self,
        columns: list[str],
        use_transaction: bool = True,
        validate_dependencies: bool = True,
        cascade: bool = False,
        run_id: str | None = None,
        commit_message: str | None = None,
        commit_info: dict[str, Any] | None = None,
    ) -> OperationReport:
        """
        Delete columns from the fact table together with every reference to them.

        A primary key column cannot be deleted. A column that is the parent of
        another column in a hierarchy, or the ``label_for`` target of one or more
        label columns, cannot be deleted by default either: it would silently
        orphan its children's ``parent_name`` or its label columns' ``label_for``.
        Pass ``cascade=True`` to allow it anyway; every child's ``parent_name`` and
        every label column's ``label_for`` is then reset to ``NULL``, with a
        warning. A dropped column that is part of ``dataset_metadata.cluster_by``
        is also removed from it (reset to ``NULL`` if it was the only sort column),
        with a warning. Dropping a label column itself needs nothing special.
        ``ALTER TABLE ... DROP COLUMN`` is a DuckLake metadata-only operation: no
        data file is rewritten. Either every requested column is dropped, or none
        is.

        Args:
            columns: Non-empty list of the column names to delete.
            use_transaction: Whether to run every deletion inside a single DuckDB
                transaction, so that a failure leaves the fact table and its
                ``metadata`` rows exactly as they were. Defaults to True.
            validate_dependencies: Whether to refuse the deletion of a primary key,
                of a hierarchy parent or of a code column targeted by a label column
                (unless ``cascade``). Defaults to True.
            cascade: Whether to allow deleting a column that is the parent of
                another column or the ``label_for`` target of a label column,
                detaching its children/label columns (``parent_name``/``label_for``
                set to ``NULL``) instead of refusing the deletion. Defaults to False.
            run_id: Run identifier recorded on the resulting DuckLake snapshot
                (``ducklake_set_commit_message``). Ignored (skipped with a DEBUG
                log) on a connection with no real DuckLake catalog attached.
            commit_message: Commit message recorded alongside ``run_id``.
            commit_info: Extra JSON-serializable fields merged into the commit's
                ``extra_info``.

        Returns:
            OperationReport: the dropped columns are in ``report.columns_dropped``;
            the dependency warnings (e.g. a categorical column backing a menu, a
            detached child) are in ``report.warnings``.

        Raises:
            ValueError: If ``columns`` is empty, names a column absent from the
                fact table, or (with ``validate_dependencies``) a primary key, a
                hierarchy parent or a label_for target without ``cascade``.
            RuntimeError: If the structural audit finds a critical issue.
            duckdb.Error: If a column cannot be dropped; the transaction is rolled
                back.

        Example:
            >>> report = deleter.delete_columns(['old_col1', 'old_col2'])
            >>> report.columns_dropped
            ['old_col1', 'old_col2']
            >>> # Deleting a hierarchy parent, detaching its children
            >>> report = deleter.delete_columns(['region'], cascade=True)
        """
        # Liste non vide de colonnes existantes, sans doublon
        if not columns:
            raise ValueError("columns must name at least one column to delete")
        columns = list(dict.fromkeys(columns))
        existing_columns = set(self._get_fact_table_columns())
        unknown = [c for c in columns if c not in existing_columns]
        if unknown:
            raise ValueError(f"Unknown column(s) in fact_table: {unknown}")

        # Analyse des dépendances : refus des suppressions qui briseraient le schéma
        dependency_warnings = (
            self._check_column_dependencies(columns, cascade=cascade)
            if validate_dependencies
            else []
        )

        # Logging
        self.logger.debug(f"Starting columns deletion (transaction: {use_transaction})")

        # Bloc transactionnel unique : sur exception, ni les colonnes ni leurs
        # lignes metadata ne sont perdues
        with self._transaction(
            "delete_columns",
            use_transaction=use_transaction,
            run_id=run_id,
            commit_message=commit_message,
            commit_info=commit_info,
        ) as report:
            report.warnings.extend(dependency_warnings)
            # Suppression de chaque colonne et de ses références
            for column in columns:
                self._drop_column_with_references(column, cascade=cascade)
                report.columns_dropped.append(column)
            # Horodatage et audit, dans le même snapshot que la suppression
            self._touch_dataset_metadata()
            self._post_write_audit(report)

        # Rapport
        final_report = self.last_report
        assert final_report is not None  # posé par _transaction sur tout succès
        self._finalize_report_after_write(final_report)

        # Rapport
        self.logger.info(final_report.summary())
        return final_report

    # Méthode d'analyse des dépendances des colonnes à supprimer
    def _check_column_dependencies(
        self, columns: list[str], cascade: bool = False
    ) -> list[str]:
        """Refuse the deletions that would break the schema, and list the warnings.

        A primary key column is never deletable. A hierarchy parent or a
        ``label_for`` target is deletable only with ``cascade``, which detaches its
        children or label columns. A categorical column is deletable, but backs a
        menu in the interface: a warning says so.

        Args:
            columns: Columns about to be deleted, all present in the fact table.
            cascade: Whether hierarchy children and label columns may be detached.

        Returns:
            list[str]: Non-blocking warnings, one per categorical column and per
            dependency that ``cascade`` will detach.

        Raises:
            ValueError: If a column is a primary key, or a hierarchy parent or
                label_for target while ``cascade`` is False, listing every refused
                column and why.
        """
        # Initialisation des motifs de refus et des avertissements
        refusals: list[str] = []
        dependency_warnings: list[str] = []

        # Parcours des colonnes
        for column in columns:
            # Clé primaire : suppression toujours refusée
            if self._is_primary_key_column(column):
                refusals.append(f"{column!r} is a primary key")
                continue

            # Colonne catégorielle : avertissement pour l'interface
            if self._is_categorical_column(column):
                dependency_warnings.append(
                    f"Column {column!r} is flagged as categorical and may back a"
                    " menu in the interface"
                )

            # Colonne parente d'une hiérarchie, ou cible de colonnes de libellés
            dependents = {
                "the hierarchy parent of": self._get_hierarchy_children(column),
                "the label_for target of": self._get_label_columns_for_code(column),
            }
            for role, dependent_columns in dependents.items():
                if not dependent_columns:
                    continue
                if cascade:
                    dependency_warnings.append(
                        f"Column {column!r} is {role} {dependent_columns}; they are"
                        " detached (cascade=True)"
                    )
                else:
                    refusals.append(
                        f"{column!r} is {role} {dependent_columns} (use cascade=True"
                        " to detach them)"
                    )

        # Refus groupé : aucune colonne n'est supprimée
        if refusals:
            raise ValueError(f"Column(s) cannot be deleted: {'; '.join(refusals)}")

        # Logging des avertissements
        for warning in dependency_warnings:
            self.logger.warning(warning)
        return dependency_warnings
